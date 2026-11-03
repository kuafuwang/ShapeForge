"""Test-only v2 task packages (no production task is migrated by A1).

The helpers write standard ``tasks/<op>/`` directories into ``tmp_path``: a typed OpSpec v2,
a real (CPU) reference, and whichever optional standard files a test needs.  They exist so
A1's contract, reference boundary and track plane can be exercised without touching the
shipped v1 tasks (whose identity is frozen by the G0 corpus).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from .v2_mapping import TASKS

RMSNORM_REFERENCE = '''"""RMSNorm reference: fp64 accumulation, one rounding at the output."""

import torch


def reference(inputs, attrs):
    x = inputs["x"].to(torch.float64)
    w = inputs["w"].to(torch.float64)
    mean = (x * x).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(mean + float(attrs["EPS"])) * w
    return {"y": y.to(torch.float32)}
'''

RMSNORM_SILU_REFERENCE = '''"""Fused RMSNorm then SiLU, fp64 accumulation, one rounding at the output."""

import torch


def reference(inputs, attrs):
    x = inputs["x"].to(torch.float64)
    w = inputs["w"].to(torch.float64)
    mean = (x * x).mean(dim=-1, keepdim=True)
    r = x * torch.rsqrt(mean + float(attrs["EPS"])) * w
    return {"y": (r * torch.sigmoid(r)).to(torch.float32)}
'''

RMSNORM_READ_WRITE_REFERENCE = '''"""RMSNorm with y and acc read_write: both initial and final values are used."""

import torch


def reference(inputs, attrs):
    x = inputs["x"].to(torch.float64)
    w = inputs["w"].to(torch.float64)
    acc = inputs["acc"].to(torch.float64)
    y0 = inputs["y"].to(torch.float64)
    mean = (x * x).mean(dim=-1, keepdim=True)
    y = y0 + x * torch.rsqrt(mean + float(attrs["EPS"])) * w
    return {"y": y.to(torch.float32), "acc": (acc + y0).to(torch.float32)}
'''

SEM_REFERENCE = '''"""A tiny semantic-recovery fixture: y = -x."""

import torch


def reference(inputs, attrs):
    return {"y": (-inputs["x"].to(torch.float64)).to(torch.float32)}
'''


def shipped_v2_document(task: str) -> dict[str, Any]:
    """@brief 读取已迁移题的标准 v2 ``opspec.json``（不经过 v1 loader）。

    The fixtures used to derive a v2 document from the shipped *v1* OpSpec.  A5 migrated those
    tasks, so the shipped file is now the v2 document itself: reading it keeps the fixture equal
    to the frozen contract (cases/sizes/attrs/tolerances) instead of reviving a v1 document that
    no longer exists.

    @param task 已迁移的任务名（标准目录 ``tasks/<op>/``）。
    @return v2 文档副本。
    @raises AssertionError 该目录不是 v2 标准入口。
    """
    document = json.loads((TASKS / task / "opspec.json").read_text(encoding="utf-8"))
    if document.get("spec_version") != 2:
        raise AssertionError(f"{task}: the standard entry is not v2 (spec_version={document.get('spec_version')!r})")
    return copy.deepcopy(document)


def rmsnorm_opspec(op_name: str = "rmsnorm") -> dict[str, Any]:
    """@brief shipped RMSNorm 的 v2 OpSpec 副本（cases/sizes/attrs 与 G0 完全一致）。

    @param op_name 算子名（必须等于目录名）。
    @return v2 文档。
    """
    document = shipped_v2_document("rmsnorm")
    document["op_name"] = op_name
    return document


SIGMOID_REFERENCE = '''"""Sigmoid reference in fp64 (same math as the frozen v1 task)."""

import torch


def reference(inputs, attrs):
    return {"y": torch.sigmoid(inputs["x"].to(torch.float64))}
'''


RMSNORM_G0_REFERENCE = '''"""RMSNorm with the v1 sampler ported into the standard gen_inputs hook.

This fixture is test-only.  It reproduces the frozen v1 ``tasks/rmsnorm`` sampling stream
(mags, large=300, weight offset/scale/seed offset and the six adversarial cases) so A1 can
compare both the generated inputs and the fp64 reference outputs against the G0 corpus.
"""

import math

import torch

_FLOAT_MAX = {"fp16": 65504.0, "bf16": 3.3895313892515355e38}
_TINY_NORMAL = {"fp16": 2.0 ** -14, "bf16": 2.0 ** -126}
_TINY_SUBNORMAL = {"fp16": 2.0 ** -24, "bf16": 2.0 ** -133}


def _generator(seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    return g


def _sample(case, shape, seed, dtype, *, mag=1.0, large=300.0, spike=100.0):
    g = _generator(seed)
    shape = tuple(int(s) for s in shape)
    if case == "normal":
        return torch.randn(shape, generator=g) * mag
    if case == "wide_range":
        e = torch.rand(shape, generator=g) * 5 - 3
        s = torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0)
        return s * torch.pow(10.0, e) * mag
    if case == "large":
        top = large if large is not None else min(_FLOAT_MAX.get(dtype, 3.0e38) * 0.25, 1.0e4)
        return (torch.rand(shape, generator=g) * 2 - 1) * top
    if case == "tiny":
        if dtype == "fp16":
            lo, hi = _TINY_SUBNORMAL["fp16"], _TINY_NORMAL["fp16"] * 4
        else:
            lo = _TINY_NORMAL.get(dtype, 1.2e-38) * 2.0
            hi = lo * 4096.0
        e = torch.rand(shape, generator=g) * (math.log10(hi) - math.log10(lo)) + math.log10(lo)
        s = torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0)
        return s * torch.pow(10.0, e)
    if case == "outlier":
        x = torch.randn(shape, generator=g) * mag
        mask = torch.rand(shape, generator=g) < 1e-3
        return torch.where(mask, x * spike, x)
    raise ValueError("unknown case " + repr(case))


def gen_inputs(case, seed, instance):
    dtype = instance.dtype
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    shape = tuple(instance.tensor("x").shape)
    base = "normal" if case == "ieee" else case
    x = _sample(base, shape, seed, dtype, mag=1.0, large=300.0).to(torch_dtype)
    w = (1.0 + 0.1 * _sample("normal", (shape[1],), seed + 99, dtype)).to(torch_dtype)
    return {"x": x, "w": w}


def reference(inputs, attrs):
    x = inputs["x"].to(torch.float64)
    w = inputs["w"].to(torch.float64)
    mean = (x * x).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(mean + float(attrs["EPS"])) * w
    return {"y": y}
'''


def sigmoid_opspec(op_name: str = "sigmoid") -> dict[str, Any]:
    """@brief shipped sigmoid 的 v2 OpSpec 副本（cases/sizes 与 G0 一致）。

    @param op_name 算子名。
    @return v2 文档。
    """
    document = shipped_v2_document("sigmoid")
    document["op_name"] = op_name
    return document


def rmsnorm_silu_opspec(op_name: str = "rmsnorm_silu") -> dict[str, Any]:
    """@brief 融合 RMSNorm+SiLU 的 v2 夹具；ABI 与 workspace RMSNorm 相同。

    @param op_name 算子名。
    @return v2 文档。
    """
    document = rmsnorm_workspace_opspec(op_name)
    document["formula"] = "r = rmsnorm(x, w); y = r * sigmoid(r)"
    document["discovery"] = {"phase": 2, "default_instance": {"dtype": "fp16", "size": "small"}}
    return document


def rmsnorm_read_write_opspec(op_name: str = "rmsnorm_rw") -> dict[str, Any]:
    """@brief 带 ``read_write`` 输入与输出的 v2 RMSNorm（终值属于差分合同）。

    @param op_name 算子名。
    @return v2 文档。
    """
    document = rmsnorm_opspec(op_name)
    document["formula"] = "y = y + rmsnorm(x, w); acc = acc + y (y/acc read_write)"
    document["tensors"]["acc"] = {
        "role": "input",
        "dims": ["M", "H"],
        "dtype": "T",
        "layout": "row_major",
        "strides": ["H", "1"],
        "space": "global",
        "access": "read_write",
    }
    document["tensors"]["y"]["access"] = "read_write"
    document["work_accounting"]["min_bytes"]["terms"].append({"tensor": "acc"})
    document["discovery"] = {"phase": 2, "default_instance": {"dtype": "fp16", "size": "small"}}
    return document


def rmsnorm_workspace_opspec(op_name: str = "rmsnorm_ws") -> dict[str, Any]:
    """@brief 带 ``space: workspace`` 暂存张量的 v2 RMSNorm（A2 workspace ABI 夹具）。

    The scratch buffer is not a reference input and not a reference result: it is declared in the
    ABI only so a candidate can point a stage at it.  ``y`` stays the single result tensor.

    @param op_name 算子名。
    @return v2 文档。
    """
    document = rmsnorm_opspec(op_name)
    document["formula"] = "scratch = x; y = rmsnorm(scratch, w)"
    document["tensors"]["scratch"] = {
        "role": "output",
        "dims": ["M", "H"],
        "dtype": "T",
        "layout": "row_major",
        "strides": ["H", "1"],
        "space": "workspace",
        "access": "write",
    }
    document["discovery"] = {"phase": 2, "default_instance": {"dtype": "fp16", "size": "small"}}
    return document


def sem_opspec(op_name: str = "sem_gemm_neg") -> dict[str, Any]:
    """@brief 一个 ``sem_*`` 白名单夹具（用于 display.spec_text 的允许路径）。

    @param op_name 算子名。
    @return v2 文档。
    """
    return {
        "spec_version": 2,
        "op_name": op_name,
        "category": "elementwise",
        "formula": "y = -x",
        "type_vars": {"T": ["fp16"]},
        "symbols": {"N": {"min": 1}},
        "tensors": {
            "x": {
                "role": "input",
                "dims": ["N"],
                "dtype": "T",
                "layout": "contiguous",
                "strides": ["1"],
                "space": "global",
                "access": "read",
            },
            "y": {
                "role": "output",
                "dims": ["N"],
                "dtype": "T",
                "layout": "contiguous",
                "strides": ["1"],
                "space": "global",
                "access": "write",
            },
        },
        "numerics": {
            "acc_dtype": "fp32",
            "rounding": "single_round_at_output",
            "tolerances": {"fp16": {"rtol": 0.002, "atol_rel": 0.001}},
            "exact": False,
        },
        "attrs": {},
        "cases": {"names": ["normal"]},
        "sizes": {"small": {"N": 8, "usage": ["golden"]}},
        "spec_params": ["N"],
        "work_accounting": {
            "flops": {"terms": [{"coefficient": 1, "symbols": ["N"]}]},
            "min_bytes": {"terms": [{"tensor": "x"}, {"tensor": "y"}]},
        },
        "discovery": {"phase": 1, "default_instance": {"dtype": "fp16", "size": "small"}},
    }


def write_task(
    root: Path,
    *,
    op_name: str = "rmsnorm",
    opspec: Mapping[str, Any] | None = None,
    reference: str = RMSNORM_REFERENCE,
    extra: Mapping[str, str] | None = None,
) -> Path:
    """@brief 写一个标准 v2 任务目录。

    @param root 父目录（任务目录将命名为 ``op_name``）。
    @param op_name 算子名。
    @param opspec v2 文档；缺省用 RMSNorm。
    @param reference reference.py 源。
    @param extra 附加文件名 → 文本（reference.json/display.json/...）。
    @return 任务目录。
    """
    task_dir = Path(root) / op_name
    task_dir.mkdir(parents=True, exist_ok=True)
    document = copy.deepcopy(dict(opspec)) if opspec is not None else rmsnorm_opspec(op_name)
    (task_dir / "opspec.json").write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if reference is not None:
        (task_dir / "reference.py").write_text(reference, encoding="utf-8")
    for name, text in (extra or {}).items():
        path = task_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return task_dir


__all__ = [
    "RMSNORM_G0_REFERENCE",
    "RMSNORM_SILU_REFERENCE",
    "RMSNORM_READ_WRITE_GEN_REFERENCE",
    "RMSNORM_READ_WRITE_REFERENCE",
    "RMSNORM_REFERENCE",
    "SEM_REFERENCE",
    "SIGMOID_REFERENCE",
    "rmsnorm_opspec",
    "shipped_v2_document",
    "rmsnorm_read_write_opspec",
    "rmsnorm_silu_opspec",
    "sem_opspec",
    "sigmoid_opspec",
    "write_task",
]

RMSNORM_READ_WRITE_GEN_REFERENCE = '''"""RMSNorm read_write fixture with an explicit gen_inputs for all reference inputs."""

import torch


def gen_inputs(case, seed, instance):
    m = int(instance.symbols["M"])
    h = int(instance.symbols["H"])
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[instance.dtype]
    g = torch.Generator().manual_seed(int(seed) + 17)
    return {
        "x": (torch.randn(m, h, generator=g) * 1.0).to(dtype),
        "w": (1.0 + 0.1 * torch.randn(h, generator=g)).to(dtype),
        "y": (torch.randn(m, h, generator=g) * 0.5).to(dtype),
        "acc": (torch.randn(m, h, generator=g) * 0.5).to(dtype),
    }


def reference(inputs, attrs):
    x = inputs["x"].to(torch.float64)
    w = inputs["w"].to(torch.float64)
    acc = inputs["acc"].to(torch.float64)
    y0 = inputs["y"].to(torch.float64)
    mean = (x * x).mean(dim=-1, keepdim=True)
    y = y0 + x * torch.rsqrt(mean + float(attrs["EPS"])) * w
    return {"y": y.to(torch.float32), "acc": (acc + y0).to(torch.float32)}
'''
# add to __all__
