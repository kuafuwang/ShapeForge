"""sem_attention_odd_scale: independent fp64 reference and the frozen v1 input stream.

The module is the whole oracle of the task.  It may not import the harness (``tasks``/
``tcenv``) or any candidate, so the v1 sampler and the harness IEEE injection are carried
as the frozen block below (byte-identical to the A5-migrated references).
The scale mode is ``sum_with_log2e``, not the usual product.
"""

import torch
import torch.nn.functional as F


# ---- frozen G0 sampler block (byte-identical in every A5-migrated reference) ----------------
import math

_FLOAT_MAX = {"fp16": 65504.0, "bf16": 3.3895313892515355e38, "fp32": 3.4028234663852886e38}
_TINY_NORMAL = {"fp16": 2.0**-14, "bf16": 2.0**-126, "fp32": 2.0**-126}
_TINY_SUBNORMAL = {"fp16": 2.0**-24, "bf16": 2.0**-133, "fp32": 2.0**-149}
_SPECIALS = (float("nan"), float("inf"), float("-inf"), -0.0)
_IEEE_SEED_STRIDE = 104_729


def _generator(seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    return g


def _sample(case, shape, seed, dtype, *, mag=1.0, large=None, spike=100.0):
    """The frozen v1 adversarial sampler (normal/wide_range/large/tiny/outlier/cancel/const)."""
    g = _generator(seed)
    shape = tuple(int(s) for s in shape)
    dmax = _FLOAT_MAX.get(dtype, 3.0e38)
    if case == "normal":
        return torch.randn(shape, generator=g) * mag
    if case == "wide_range":
        e = torch.rand(shape, generator=g) * 5 - 3
        s = torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0)
        return s * torch.pow(10.0, e) * mag
    if case == "large":
        top = large if large is not None else min(dmax * 0.25, 1.0e4)
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
    if case == "cancel":
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
    raise ValueError("unknown case " + repr(case))


def _inject(t, seed, max_n=None):
    """The frozen v1 IEEE injection (``prepare_case`` applied it after sampling)."""
    flat = t.reshape(-1).clone()
    numel = flat.numel()
    last = t.shape[-1] if t.dim() else 1
    rows = max(1, numel // max(1, last))
    n = int(min(256, max(8, 0.1 * rows, numel / 65536), max(1, numel // 4)))
    if max_n is not None:
        n = max(2, min(n, max_n))
    perm = torch.randperm(numel, generator=_generator(seed))
    perm = perm[(perm != 0) & (perm != numel - 1)]
    pos = torch.cat([torch.tensor([0, numel - 1]), perm[: n - 2]])[:n]
    kinds = torch.randperm(n, generator=_generator(seed + 1)) % len(_SPECIALS)
    flat[pos] = torch.tensor([_SPECIALS[int(k)] for k in kinds], dtype=flat.dtype)
    return flat.reshape(t.shape)
# ---- end frozen block ----------------------------------------------------------------------


_MAGS = {"normal": 1.0, "wide_range": 0.1}
_DEFAULT_MAG = 1.0
_LARGE = 6.0
_SPIKE = 5.0
_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453
_SEED_OFFSET_q = 0
_SEED_OFFSET_k = 31
_SEED_OFFSET_v = 62


def gen_inputs(case, seed, instance):
    """The v1 case/seed stream (q/k/v at seed + 0/31/62; ieee injects q only)."""
    base = "normal" if case == "ieee" else case
    mag = _MAGS.get(base, _DEFAULT_MAG)
    shape = tuple(instance.tensor("q").shape)
    out = {}
    for name in ("q", "k", "v"):
        offset = {'q': _SEED_OFFSET_q, 'k': _SEED_OFFSET_k, 'v': _SEED_OFFSET_v}[name]
        out[name] = _sample(base, shape, seed + offset, instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE)
    if case == "ieee":
        out["q"] = _inject(out["q"], seed + _IEEE_SEED_STRIDE)
    return out


def reference(inputs, attrs):
    """F.scaled_dot_product_attention with the sum_with_log2e scale, fp64 output."""
    sm_scale = float(attrs["SM_SCALE"])
    # The semantic-recovery scale mode is a SUM, not the usual product:
    # exp2(qk * (SM_SCALE + log2 e)) == exp(qk * (SM_SCALE + log2 e) * ln 2), so the framework's
    # natural-log scale is (SM_SCALE + LOG2E) * ln(2) — exactly v1's effective_scale helper.
    scale = (sm_scale + _LOG2E) * _LN2
    o = F.scaled_dot_product_attention(inputs["q"].float(), inputs["k"].float(), inputs["v"].float(), scale=scale)
    return {"o": o.double()}
