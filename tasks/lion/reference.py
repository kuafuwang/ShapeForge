"""lion: independent fp64 reference and the frozen v1 input stream.

The module is the whole oracle of the task.  It may not import the harness (``tasks``/
``tcenv``) or any candidate, so the v1 sampler and the harness IEEE injection are carried
as the frozen block below (byte-identical to the A5-migrated references).
tiny 的非 fp16 缩放与 sign 间断点的模块级 error scale 都按 v1 登记保留。
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


_MAG = 1.0
_LARGE = 1000.0
_SEED_OFFSET = 7
_TINY_NON_FP16_SCALE = 1000.0
_DISCONTINUITY_REL = 1e-05
_ATOL_STEPS = 2.5


def gen_inputs(case, seed, instance):
    """The v1 case/seed stream (p/g/m at seed + 7*index; tiny scaled off fp16)."""
    base = "normal" if case == "ieee" else case
    shape = tuple(instance.tensor("p").shape)
    out = {}
    for index, name in enumerate(("p", "g", "m")):
        out[name] = _sample(base, shape, seed + _SEED_OFFSET * index, instance.dtype, mag=_MAG, large=_LARGE)
    if case == "tiny" and instance.dtype != "fp16":
        # fp32 subnormals are flushed by the GPU: keep (1-beta)*x in the normal range, otherwise
        # sign(beta1*m + (1-beta1)*g) legitimately flips for every implementation.
        out = {key: value * _TINY_NON_FP16_SCALE for key, value in out.items()}
    if case == "ieee":
        for index, name in enumerate(("p", "g", "m")):
            out[name] = _inject(out[name], seed + _IEEE_SEED_STRIDE * (index + 1))
    return out


def reference(inputs, attrs):
    """The independent fp64 Lion step."""
    lr = float(attrs["LR"])
    wd = float(attrs["WD"])
    beta1 = float(attrs["BETA1"])
    beta2 = float(attrs["BETA2"])
    p, g, m = (inputs[k].to(torch.float64) for k in ("p", "g", "m"))
    u = torch.sign(beta1 * m + (1 - beta1) * g)
    return {"p_out": p * (1 - lr * wd) - lr * u, "m_out": beta2 * m + (1 - beta2) * g}


def error_scale(inputs, reference_outputs, instance, tolerances):
    """The v1 sign-discontinuity allowance.

    The v1 evaluator filtered non-finite values on the ``ieee`` case *before* calling this hook;
    the v2 hook receives no case argument, so it filters here instead — a no-op on finite cases.
    """
    attrs = instance.resolved_attrs
    lr = float(attrs["LR"])
    beta1 = float(attrs["BETA1"])
    beta2 = float(attrs["BETA2"])
    atol_rel = float(tolerances.atol_rel)

    def finite(t):
        return torch.where(torch.isfinite(t), t, torch.zeros_like(t)) if t.is_floating_point() else t

    p, g, m = (finite(inputs[k]).to(torch.float64) for k in ("p", "g", "m"))
    ref = {key: finite(value) for key, value in reference_outputs.items()}
    a, b = beta1 * m, (1 - beta1) * g
    amb = (a + b).abs() <= _DISCONTINUITY_REL * (a.abs() + b.abs())
    base = torch.sqrt(torch.mean(ref["p_out"] ** 2)).expand_as(ref["p_out"]).clone()
    base[amb] = _ATOL_STEPS * lr / atol_rel
    return {"p_out": base, "m_out": torch.sqrt(torch.mean(ref["m_out"] ** 2))}
