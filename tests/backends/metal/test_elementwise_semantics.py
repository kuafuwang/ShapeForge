"""The written relu / sigmoid / silu formulas agree with their references on specials. No GPU."""

from __future__ import annotations

from pathlib import Path

import torch

from tcenv.backends.metal.testing import _floor, scaled_error
from tasks.relu.task import build as build_relu
from tasks.sigmoid.task import build as build_sigmoid
from tasks.silu.task import build as build_silu

ROOT = Path(__file__).resolve().parents[3]


def _specials() -> torch.Tensor:
    """@brief 由 fp16 位模式构造 NaN、±Inf、±0、次正规数和正常值。

    @return fp16 specials 张量。
    """
    # Stored as int32 first: values above 32767 are fp16 bit patterns, not signed int16 numbers.
    bits = [0x7E00, 0x7C00, 0xFC00, 0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x3C00, 0xBC00]
    return torch.tensor(bits, dtype=torch.int32).to(torch.int16).view(torch.float16)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """@brief 逐元素比较两个张量，NaN 与 NaN 视为相同。

    @param a 张量。
    @param b 张量。
    @return 所有元素都相同则为 True。
    """
    both_nan = torch.isnan(a) & torch.isnan(b)
    return bool(torch.all(both_nan | (a == b)).item())


def test_relu_spec_and_naive_match_the_reference_on_specials():
    """@brief relu 的 spec 和 naive 写法与参考实现在 specials 上完全一致（NaN 与 NaN 视为相同）。
    """
    spec = (ROOT / "tasks/relu/kernel_triton.py").read_text()
    naive = (ROOT / "tasks/relu/naive.metal").read_text()
    assert "tl.where(x < 0, 0.0, x)" in spec
    assert "v < 0.0f ? 0.0f : v" in naive
    x = _specials().double()
    ref = build_relu("fp16", "small").reference({"x": x})["y"]
    written = torch.where(x < 0, torch.zeros_like(x), x)
    assert _same(ref, written)


def test_sigmoid_and_silu_specials_match_the_stable_formula():
    """@brief sigmoid 和 silu 的 spec/naive 与稳定公式在 specials 上误差在容差内，NaN 与 Inf 符号一致。
    """
    sig_spec = (ROOT / "tasks/sigmoid/kernel_triton.py").read_text()
    sig_naive = (ROOT / "tasks/sigmoid/naive.metal").read_text()
    silu_spec = (ROOT / "tasks/silu/kernel_triton.py").read_text()
    silu_naive = (ROOT / "tasks/silu/naive.metal").read_text()
    assert "tl.exp(-x" in sig_spec and "sigmoid_f" in sig_naive and "v >= 0.0f" in sig_naive
    assert "tl.exp(-x)" in silu_spec and "v * sigmoid_f(v)" in silu_naive
    x = _specials()

    def stable(v):
        e = torch.exp(-torch.abs(v))
        return torch.where(v >= 0, 1.0 / (1.0 + e), e / (1.0 + e))

    def agrees(task, naive_fp16: torch.Tensor) -> None:
        ref = task.reference({"x": x.double()})["y"]
        err = scaled_error(naive_fp16.double(), ref, torch.zeros((), dtype=torch.float64), task.rtol, task.atol_rel, False, _floor("fp16"))
        assert float(torch.nan_to_num(err, nan=99.0, posinf=99.0).max()) <= 1.0
        assert torch.equal(torch.isnan(naive_fp16), torch.isnan(ref))
        infinite = torch.isinf(ref)
        assert torch.all(~infinite | (torch.isinf(naive_fp16) & (torch.sign(naive_fp16) == torch.sign(ref))))

    agrees(build_sigmoid("fp16", "small"), stable(x.float()).to(torch.float16))
    agrees(build_silu("fp16", "small"), (x.float() * stable(x.float())).to(torch.float16))
