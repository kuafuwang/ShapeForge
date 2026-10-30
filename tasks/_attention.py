"""Shared assembly for the non-causal attention tasks.

``flash_attention`` and the misleading-scale semantic recovery task share the same
representation (q/k/v/o ``[B, H, S, D]``, fp32 online softmax, one output rounding) and the
same registered Metal launch profile; they differ only in *semantics*: the semantic variant
forms its qk scale as a **sum** ``SM_SCALE + log2(e)`` instead of the usual product.  That
difference lives in ``tasks/<op>/opspec.json`` (distinct formula → distinct ``opspec_digest``)
and in the track's registered reference/baseline scale mode, never in a second Contract.

This module only turns the track's *registered* implementation parameters into closures and
hands them to :func:`tasks._assembly.build_task`; it never writes a ``Contract``, a
grid/threads literal or a tolerance.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping

import torch
import torch.nn.functional as F

from tcenv.distributions import sample
from tcenv.task import Task

from ._assembly import build_task, derive, derive_variant, require_attr

#: ``log2(e)``; the spec text folds it into the qk scale so ``exp2`` can be used.
LOG2E = 1.4426950408889634
#: The registered reference/baseline scale modes.  ``eff_scale`` is the usual product form
#: (``exp2(qk * SM_SCALE * log2 e) == exp(qk * SM_SCALE)``); ``sum_with_log2e`` is the
#: semantic variant's sum (``exp2(qk * (SM_SCALE + log2 e)) == exp(qk * (SM_SCALE * ln2 + 1))``).
SCALE_MODES = ("eff_scale", "sum_with_log2e")


def _gen_inputs(params: Mapping, shape: tuple[int, int, int, int], dtype: str) -> Callable:
    """@brief 按 track 登记参数构造 q/k/v 生成闭包。

    @param params ``implementations.gen_inputs.params``。
    @param shape ``(B, H, S, D)``。
    @param dtype 元素类型。
    @return ``(case, seed) -> {"q", "k", "v"}``。
    """
    mags = dict(params["mags"])
    default_mag = float(params["default_mag"])
    large = float(params["large"])
    spike = float(params["spike"])
    offsets = dict(params["seed_offsets"])

    def gen_inputs(case: str, seed: int) -> dict[str, torch.Tensor]:
        mag = mags.get(case, default_mag)
        return {
            name: sample(case, shape, seed + int(offsets[name]), dtype, mag=mag, large=large, spike=spike)
            for name in ("q", "k", "v")
        }

    return gen_inputs


def effective_scale(sm_scale: float, mode: str) -> float:
    """@brief 由契约常量 ``SM_SCALE`` 与登记的模式算参考实现用的自然对数标度。

    @param sm_scale 契约常量 ``SM_SCALE``（``1/sqrt(D)``）。
    @param mode ``eff_scale`` 或 ``sum_with_log2e``。
    @return 传给框架注意力的自然对数标度。
    @raises ValueError 未登记的模式。
    """
    if mode == "eff_scale":
        return sm_scale
    if mode == "sum_with_log2e":
        return (sm_scale + LOG2E) * math.log(2.0)
    raise ValueError(f"flash_attention: unregistered scale mode {mode!r}; known: {SCALE_MODES}")


def _require_mode(params: Mapping, where: str) -> str:
    """@brief 校验登记中的 scale 模式字段。

    @param params 登记参数。
    @param where 诊断前缀。
    @return 模式名。
    """
    mode = str(params.get("scale"))
    if mode not in SCALE_MODES:
        raise ValueError(f"{where} scale registration changed: {params}")
    return mode


def _reference(params: Mapping, scale: float) -> Callable:
    """@brief 注意力参考实现（``F.scaled_dot_product_attention``，给定自然对数标度）。

    @param params ``implementations.reference.params``。
    @param scale 本实例的有效标度。
    @return ``inp -> {"o"}``。
    """
    _require_mode(params, "flash_attention reference")
    if params.get("accumulate") != "double_output":
        raise ValueError(f"flash_attention reference registration changed: {params}")
    if params.get("kernel") != "torch.nn.functional.scaled_dot_product_attention":
        raise ValueError(f"flash_attention reference kernel changed: {params}")

    def reference(inp: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        o = F.scaled_dot_product_attention(inp["q"].float(), inp["k"].float(), inp["v"].float(), scale=scale)
        return {"o": o.double()}

    return reference


def _mlx(params: Mapping, scale: float) -> Callable:
    """@brief mlx 侧注意力基线（``mx.fast.scaled_dot_product_attention``）。

    @param params track 登记参数。
    @param scale 本实例的有效标度。
    @return ``(q, k, v) -> o``。
    """
    if params.get("kernel") != "mx.fast.scaled_dot_product_attention":
        raise ValueError(f"flash_attention mlx baseline registration changed: {params}")
    _require_mode(params, "flash_attention mlx")

    def f(q, k, v):
        import mlx.core as mx

        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

    return f


def _torch(params: Mapping, scale: float) -> Callable:
    """@brief torch 侧注意力基线。

    @param params track 登记参数。
    @param scale 本实例的有效标度。
    @return ``(q, k, v) -> o``。
    """
    if params.get("kernel") != "torch.nn.functional.scaled_dot_product_attention":
        raise ValueError(f"flash_attention torch baseline registration changed: {params}")
    _require_mode(params, "flash_attention torch")
    return lambda q, k, v: F.scaled_dot_product_attention(q, k, v, scale=scale)


_BASELINE_BUILDERS = {"mlx": _mlx, "torch_mps": _torch}


def build_attention(task_dir: str, dtype: str, size: str, *, variant: str = "") -> Task:
    """@brief 装配一道非因果注意力题（默认启动或登记的 variant）。

    @param task_dir ``tasks/`` 下的目录名。
    @param dtype 元素类型。
    @param size 尺寸档位。
    @param variant 空串表示默认 profile，否则为登记 factory 的 variant 文本。
    @return 装配好的 ``Task``。
    @raises ValueError track 登记的模式或 baseline 没有实现。
    """
    derived = derive_variant(task_dir, dtype, size, variant) if variant else derive(task_dir, dtype, size)
    track, instance, _specialization = derived
    shape = tuple(int(instance.symbols[name]) for name in ("B", "H", "S", "D"))
    ref_params = dict(track.implementations["reference"]["params"])
    scale = effective_scale(require_attr(instance, "SM_SCALE"), _require_mode(ref_params, f"{task_dir} reference"))
    baseline_fns: dict[str, Callable] = {}
    for entry in track.baselines:
        name = str(entry["name"])
        builder = _BASELINE_BUILDERS.get(name)
        if builder is None:
            raise ValueError(f"{task_dir}: baseline {name!r} is registered but has no implementation")
        params = dict(entry["params"])
        if _require_mode(params, f"{task_dir} baseline {name}") != _require_mode(
            ref_params, f"{task_dir} reference"
        ):
            raise ValueError(f"{task_dir}: baseline {name!r} and the reference disagree about the scale mode")
        baseline_fns[name] = builder(params, scale)
    return build_task(
        task_dir,
        dtype,
        size,
        implementations={
            "gen_inputs": _gen_inputs(dict(track.implementations["gen_inputs"]["params"]), shape, dtype),
            "reference": _reference(ref_params, scale),
            "error_scale": None,
        },
        baseline_fns=baseline_fns,
        derived=derived,
        task_name=f"{instance.op_name}@{variant}" if variant else None,
    )


__all__ = ["LOG2E", "SCALE_MODES", "build_attention", "effective_scale"]
