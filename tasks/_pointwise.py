"""Shared assembly for the four pointwise tasks (relu / sigmoid / silu / swiglu).

The semantic facts live in ``tasks/<op>/opspec.json`` and the launch facts in the registered
Metal LoweringProfile; this module only turns the track's *registered* implementation
parameters into the ``gen_inputs``/``reference``/baseline closures and hands them to
:func:`tasks._assembly.build_task`.  It never writes a ``Contract``, a grid/threads literal or
a tolerance: those come from ``instantiate``/``lower``.
"""

from __future__ import annotations

from typing import Callable, Mapping

import torch

from tcenv.distributions import sample
from tcenv.task import Task

from ._assembly import build_task, derive


def gen_inputs_from_params(
    params: Mapping, names: tuple[str, ...], shapes: Mapping[str, tuple[int, ...]], dtype: str
) -> Callable[[str, int], dict[str, torch.Tensor]]:
    """@brief 按 track 登记的采样参数构造输入生成闭包（逐元素族共用）。

    @param params ``implementations.gen_inputs.params``。
    @param names 契约槽位顺序的输入名。
    @param shapes 输入名 → 形状。
    @param dtype 元素类型。
    @return ``(case, seed) -> {name: tensor}``。
    """
    normal_mag = float(params["normal_mag"])
    default_mag = float(params["default_mag"])
    large = float(params["large"])
    seed_stride = int(params["seed_stride"])

    def gen_inputs(case: str, seed: int) -> dict[str, torch.Tensor]:
        mag = normal_mag if case == "normal" else default_mag
        return {
            name: sample(case, shapes[name], seed + seed_stride * index, dtype, mag=mag, large=large)
            for index, name in enumerate(names)
        }

    return gen_inputs


def build_pointwise(
    task_dir: str,
    dtype: str,
    size: str,
    *,
    references: Mapping[str, Callable[..., torch.Tensor]],
    baseline_builders: Mapping[str, Callable[[Mapping], Callable]],
) -> Task:
    """@brief 装配一道逐元素题：语义/启动来自 OpSpec 与登记 profile，表示来自 track。

    @param task_dir ``tasks/`` 下的目录名。
    @param dtype 元素类型。
    @param size 尺寸档位。
    @param references 参考实现 op 名 → 接收 float64 输入的闭包。
    @param baseline_builders baseline 名 → 由登记参数构造实现闭包。
    @return 装配好的 ``Task``。
    @raises ValueError track 登记的 reference op 或 baseline 没有实现。
    """
    derived = derive(task_dir, dtype, size)
    track, instance, _specialization = derived
    names = tuple(binding.name for binding in instance.tensors if binding.role == "input")
    shapes = {name: instance.tensor(name).shape for name in names}
    ref_params = dict(track.implementations["reference"]["params"])
    if ref_params.get("accumulate") != "float64":
        raise ValueError(f"{task_dir}: reference registration must accumulate in float64, got {ref_params}")
    op = str(ref_params["op"])
    ref_fn = references.get(op)
    if ref_fn is None:
        raise ValueError(f"{task_dir}: no reference implementation registered for op {op!r}")

    def reference(inp: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"y": ref_fn(*[inp[name].to(torch.float64) for name in names])}

    baseline_fns: dict[str, Callable] = {}
    for entry in track.baselines:
        name = str(entry["name"])
        builder = baseline_builders.get(name)
        if builder is None:
            raise ValueError(f"{task_dir}: baseline {name!r} is registered but has no implementation")
        baseline_fns[name] = builder(dict(entry["params"]))

    return build_task(
        task_dir,
        dtype,
        size,
        implementations={
            "gen_inputs": gen_inputs_from_params(dict(track.implementations["gen_inputs"]["params"]), names, shapes, dtype),
            "reference": reference,
            "error_scale": None,
        },
        baseline_fns=baseline_fns,
        derived=derived,
    )


__all__ = ["build_pointwise", "gen_inputs_from_params"]
