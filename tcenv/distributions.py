"""Adversarial input distributions for randomized differential testing.

The paper's tester "generates inputs that stress overflow, underflow, cancellation,
and precision loss". Each case below targets one of those failure modes. Every call
is seeded so that a failing case can be reproduced from the report.
"""

from __future__ import annotations

import math

import torch

from .dtypes import FLOAT_MAX, FLOAT_TINY_NORMAL, FLOAT_TINY_SUBNORMAL

GENERIC_CASES = ("normal", "wide_range", "large", "tiny", "outlier", "cancel", "const")


def generator(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    return g


def sample(case: str, shape, seed: int, dtype: str = "fp16", *, mag: float = 1.0, large: float | None = None, spike: float = 100.0) -> torch.Tensor:
    """Draw a float32 tensor following one of the adversarial cases.

    ``mag``   scales the 'normal' distributions (tasks pick it so reference outputs stay representable).
    ``spike`` is the multiplier of the rare (1e-3) outliers; matmul-like tasks use a small one so that
              the reference output stays representable.
    ``large`` is the magnitude used for the 'large' case; defaults to a quarter of the dtype range
              (the largest value for which x*x of a few terms is still representable in fp32 but not in fp16).
    """
    g = generator(seed)
    shape = tuple(int(s) for s in shape)
    dmax = FLOAT_MAX.get(dtype, 3.0e38)
    if case == "normal":
        return torch.randn(shape, generator=g) * mag
    if case == "uniform":
        return (torch.rand(shape, generator=g) * 2 - 1) * mag
    if case == "wide_range":
        # sign * 10^U(-3, 2): six decades of magnitude in one tensor (precision loss, mixed scales)
        e = torch.rand(shape, generator=g) * 5 - 3
        s = torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0)
        return s * torch.pow(10.0, e) * mag
    if case == "large":
        top = large if large is not None else min(dmax * 0.25, 1.0e4)
        return (torch.rand(shape, generator=g) * 2 - 1) * top
    if case == "tiny":
        # fp16 subnormals are handled correctly by the GPU, so they are fair game. fp32 subnormals (the
        # range of tiny bf16 values) are flushed to zero by GPU arithmetic -- for every implementation,
        # including the vendor baselines -- so for bf16/fp32 stay just above the smallest *normal* value.
        if dtype == "fp16":
            lo, hi = FLOAT_TINY_SUBNORMAL["fp16"], FLOAT_TINY_NORMAL["fp16"] * 4
        else:
            lo = FLOAT_TINY_NORMAL.get(dtype, 1.2e-38) * 2.0
            hi = lo * 4096.0
        e = torch.rand(shape, generator=g) * (math.log10(hi) - math.log10(lo)) + math.log10(lo)
        s = torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0)
        return s * torch.pow(10.0, e)
    if case == "outlier":
        x = torch.randn(shape, generator=g) * mag
        mask = torch.rand(shape, generator=g) < 1e-3
        return torch.where(mask, x * spike, x)
    if case == "cancel":
        # big +/- pairs along the last axis plus a small remainder: catastrophic cancellation
        x = torch.randn(shape, generator=g) * mag
        big = (torch.rand(shape, generator=g) * 0.5 + 0.5) * 512.0
        s = torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0)
        v = big * s
        if shape and shape[-1] % 2 == 0:
            vv = v.reshape(*shape[:-1], shape[-1] // 2, 2)
            vv[..., 1] = -vv[..., 0]
            v = vv.reshape(shape)
        return x + v
    if case == "const":
        return torch.full(shape, 1.0 * mag if mag != 1.0 else 0.75)
    if case == "sparse":
        x = torch.randn(shape, generator=g) * mag
        return torch.where(torch.rand(shape, generator=g) < 0.9, torch.zeros_like(x), x)
    raise ValueError(f"unknown case {case!r}")
