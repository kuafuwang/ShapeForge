"""Shared assembly for the tiled matmul tasks (``gemm`` and ``gemm_silu``).

The operator's semantics and its default tile live in ``opspec.json`` plus the registered
Metal GEMM profile; a ``@<BM>x<BN>x<BK>t<threads>`` suffix is a *launch* variant produced by
the registered variant factory, never a second OpSpec.  This module only assembles the
registered representation/oracle closures.
"""

from __future__ import annotations

from typing import Callable, Mapping

import torch

from tcenv.distributions import sample
from tcenv.task import Task

from ._assembly import build_task, derive, derive_variant


def _gen_inputs(params: Mapping, shape: tuple[int, int, int], dtype: str) -> Callable:
    """@brief 按 track 登记参数构造 A/B 输入生成闭包。

    @param params ``implementations.gen_inputs.params``。
    @param shape ``(M, N, K)``。
    @param dtype 元素类型。
    @return ``(case, seed) -> {"A", "B"}``。
    """
    mags = dict(params["mags"])
    default_mag = float(params["default_mag"])
    large = float(params["large"])
    spike = float(params["spike"])
    offsets = dict(params["seed_offsets"])
    m, n, k = shape

    def gen_inputs(case, seed):
        mag = mags.get(case, default_mag)
        return {
            "A": sample(case, (m, k), seed + int(offsets["A"]), dtype, mag=mag, large=large, spike=spike),
            "B": sample(case, (k, n), seed + int(offsets["B"]), dtype, mag=mag, large=large, spike=spike),
        }

    return gen_inputs


def _reference(params: Mapping, epilogues: Mapping[str, Callable]):
    """@brief 构造参考实现：fp32 乘积 → double → 登记的收尾函数。

    @param params ``implementations.reference.params``。
    @param epilogues 收尾名 → 函数。
    @return ``inp -> {"C"}``。
    """
    if params.get("accumulate") != "float64" or params.get("matmul") != "float32_cpu":
        raise ValueError(f"matmul reference registration changed: {params}")
    epilogue = epilogues.get(str(params.get("epilogue")))
    if epilogue is None:
        raise ValueError(f"matmul reference epilogue {params.get('epilogue')!r} has no implementation")

    def reference(inp):
        c = inp["A"].float() @ inp["B"].float()  # fp32 accumulate on CPU (exact products of rounded inputs)
        return {"C": epilogue(c.double())}

    return reference


def build_matmul(
    task_dir: str,
    dtype: str,
    size: str,
    *,
    epilogues: Mapping[str, Callable],
    baseline_builders: Mapping[str, Callable[[Mapping, Mapping], Callable]],
    variant: str = "",
) -> Task:
    """@brief 装配一道 matmul 题（默认启动或登记的 variant）。

    @param task_dir ``tasks/`` 下的目录名。
    @param dtype 元素类型。
    @param size 尺寸档位。
    @param epilogues 参考实现收尾名 → 函数（``identity``/``silu``）。
    @param baseline_builders baseline 名 → ``(params, epilogues) -> fn``。
    @param variant 空串表示默认 profile，否则为登记 factory 的 variant 文本。
    @return 装配好的 ``Task``。
    """
    derived = derive_variant(task_dir, dtype, size, variant) if variant else derive(task_dir, dtype, size)
    track, instance, _specialization = derived
    shape = tuple(int(instance.symbols[name]) for name in ("M", "N", "K"))
    baseline_fns = {}
    for entry in track.baselines:
        name = str(entry["name"])
        builder = baseline_builders.get(name)
        if builder is None:
            raise ValueError(f"{task_dir}: baseline {name!r} has no implementation")
        baseline_fns[name] = builder(dict(entry["params"]), epilogues)
    return build_task(
        task_dir,
        dtype,
        size,
        implementations={
            "gen_inputs": _gen_inputs(dict(track.implementations["gen_inputs"]["params"]), shape, dtype),
            "reference": _reference(dict(track.implementations["reference"]["params"]), epilogues),
            "error_scale": None,
        },
        baseline_fns=baseline_fns,
        derived=derived,
        task_name=f"{instance.op_name}@{variant}" if variant else None,
    )


__all__ = ["build_matmul"]
