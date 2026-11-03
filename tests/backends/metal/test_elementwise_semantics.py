"""The migrated relu / sigmoid / silu references agree with their stable formulas on specials.

A5 retired the hand-written ``kernel_triton.py`` / ``naive.metal`` sources this file used to
read.  The v2 standard entry's ``reference.py`` is now the authority, so the same special-value
checks run against that reference instead of a deleted v1 template.  No GPU.
"""

from __future__ import annotations

from pathlib import Path

import torch

from tcenv.backends.metal.testing import _floor, scaled_error
from tcenv.taskcontract import load_task_contract

ROOT = Path(__file__).resolve().parents[3]


def _specials() -> torch.Tensor:
    """@brief 在题面声明的 N 上构造 NaN、±Inf、±0、次正规数和正常值。

    The v2 reference validates the declared shape, so the special bit patterns are embedded in
    a full-size fp16 tensor instead of the old 11-element vector.  The remaining positions are a
    deterministic normal pattern; the comparison still covers every element.

    @return fp16 输入张量（长度 = relu/sigmoid/silu small 的 N）。
    """
    # Stored as int32 first: values above 32767 are fp16 bit patterns, not signed int16 numbers.
    bits = [0x7E00, 0x7C00, 0xFC00, 0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x3C00, 0xBC00]
    from tcenv.taskcontract import load_task_contract

    contract = load_task_contract(ROOT / "tasks/relu", dtype="fp16", size="small")
    n = int(contract.instance.symbols["N"])
    base = torch.linspace(-4.0, 4.0, n, dtype=torch.float16)
    specials = torch.tensor(bits, dtype=torch.int32).to(torch.int16).view(torch.float16)
    base[: specials.numel()] = specials
    return base


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """@brief 逐元素比较两个张量，NaN 与 NaN 视为相同。

    @param a 张量。
    @param b 张量。
    @return 所有元素都相同则为 True。
    """
    both_nan = torch.isnan(a) & torch.isnan(b)
    return bool(torch.all(both_nan | (a == b)).item())


def _reference(task: str, x: torch.Tensor) -> torch.Tensor:
    """@brief 用已迁移题的 v2 reference 计算 specials 输出。

    @param task 已迁移任务名。
    @param x fp16 输入。
    @return fp64 参考输出。
    """
    contract = load_task_contract(ROOT / "tasks" / task, dtype="fp16", size="small")
    return contract.reference.execute(contract.instance, {"x": x})["y"]


def test_relu_reference_matches_the_frozen_semantics_on_specials():
    """@brief relu 的 v2 reference 在 specials 上与 ``max(x, 0)`` 完全一致（NaN 与 NaN 视为相同）。"""
    x = _specials()
    ref = _reference("relu", x)
    written = torch.where(x.double() < 0, torch.zeros_like(x.double()), x.double())
    assert _same(ref, written)


def test_sigmoid_and_silu_references_match_the_stable_formula():
    """@brief sigmoid/silu 的 v2 reference 与稳定公式在 specials 上误差在容差内，NaN/Inf 符号一致。"""
    x = _specials()
    width = x.double()
    e = torch.exp(-torch.abs(width))
    sigmoid = torch.where(width >= 0, 1.0 / (1.0 + e), e / (1.0 + e))

    def agrees(task: str, expected: torch.Tensor) -> None:
        contract = load_task_contract(ROOT / "tasks" / task, dtype="fp16", size="small")
        ref = contract.reference.execute(contract.instance, {"x": x})["y"]
        tolerance = contract.instance.tolerances
        # Round the stable formula to the submitted dtype first, exactly like a real candidate.
        got = expected.to(torch.float16)
        err = scaled_error(
            got.double(), ref, torch.zeros((), dtype=torch.float64),
            tolerance.rtol, tolerance.atol_rel, tolerance.exact, _floor("fp16"),
        )
        assert float(torch.nan_to_num(err, nan=99.0, posinf=99.0).max()) <= 1.0
        assert torch.equal(torch.isnan(got), torch.isnan(ref))
        infinite = torch.isinf(ref)
        assert torch.all(~infinite | (torch.isinf(got) & (torch.sign(got) == torch.sign(ref))))

    agrees("sigmoid", sigmoid)
    agrees("silu", width * sigmoid)
