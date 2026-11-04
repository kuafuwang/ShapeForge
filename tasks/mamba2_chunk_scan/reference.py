"""mamba2_chunk_scan: independent fp64 reference and the frozen v1 input stream.

The module is the whole oracle of the task.  It may not import the harness (``tasks``/
``tcenv``) or any candidate, so the v1 sampler and the harness IEEE injection are carried
as the frozen block below (byte-identical to the A5-migrated references).
整个 SSD 算子内联；参考实现的 f32 升到 fp64，ieee 只注入 states。
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
_LARGE = 4.0
_SPIKE = 5.0
_OFFSETS = {"Bmat": 31, "Cmat": 62, "Dskip": 124, "decay": 186, "dt": 155, "states": 93, "x": 0}
_DT_LO, _DT_HI = 0.001, 0.05
_DECAY_LO, _DECAY_HI = 1.0, 8.0
_TORCH_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}


def _round_to(t, dtype):
    """The v1 round_to: round to the storage dtype, then widen back to float32."""
    return t.to(_TORCH_DTYPE[dtype]).to(torch.float32)




def ssd_chunk_scan(x, dt, cs, bm, cm, d, st, *, ops, batch, seq, heads, headdim, dstate, chunk, half=False):
    """Mamba-2 chunk scan for one B/C group (the paper's listing 1)."""
    nc = seq // chunk
    f = ops.f32
    mm = (lambda a: a) if half else f
    x5 = x.reshape(batch, nc, chunk, heads, headdim)
    dt4 = ops.permute(f(dt).reshape(batch, nc, chunk, heads), (0, 3, 1, 2))
    cs4 = f(cs)
    br = bm.reshape(batch, nc, chunk, dstate)
    cr = cm.reshape(batch, nc, chunk, dstate)
    cb = f(ops.einsum("bcln,bcsn->bcls", mm(cr), mm(br)))
    decay = ops.exp(cs4[..., :, None] - cs4[..., None, :])
    w = ops.where(ops.causal(chunk), cb[:, None] * decay * dt4[..., None, :], 0.0)
    w = ops.like(w, x) if half else w
    y = f(ops.einsum("bhcls,bcshp->bclhp", w, mm(x5)))
    y = y + f(ops.einsum("bcln,bchpn->bclhp", mm(cr), mm(st))) * ops.permute(ops.exp(cs4), (0, 2, 3, 1))[..., None]
    y = y + f(d)[:, None] * f(x5)
    return y.reshape(batch, seq, heads, headdim)


class _MlxOps:
    """mlx operator set (fp32 decay and sums)."""

    @staticmethod
    def f32(a):
        import mlx.core as mx
        return a.astype(mx.float32)

    @staticmethod
    def permute(a, axes):
        return a.transpose(*axes)

    @staticmethod
    def like(a, ref):
        return a.astype(ref.dtype)

    @staticmethod
    def einsum(spec, *a):
        import mlx.core as mx
        return mx.einsum(spec, *a)

    @staticmethod
    def exp(a):
        import mlx.core as mx
        return mx.exp(a)

    @staticmethod
    def where(m, a, b):
        import mlx.core as mx
        return mx.where(m, a, b)

    @staticmethod
    def causal(n):
        import mlx.core as mx
        return mx.tril(mx.ones((n, n))) > 0


class _TorchOps:
    """torch operator set (fp32 decay and sums)."""

    @staticmethod
    def f32(a):
        return a.float()

    @staticmethod
    def permute(a, axes):
        return a.permute(*axes)

    @staticmethod
    def like(a, ref):
        return a.to(ref.dtype)

    einsum = staticmethod(torch.einsum)
    exp = staticmethod(torch.exp)

    @staticmethod
    def where(m, a, b):
        return torch.where(m, a, torch.as_tensor(b, dtype=a.dtype, device=a.device))

    @staticmethod
    def causal(n):
        return torch.tril(torch.ones((n, n), dtype=torch.bool, device="mps"))


class _RefOps(_TorchOps):
    """The oracle's operator set: f32 widens to fp64 and the causal mask is built on CPU."""

    f32 = staticmethod(lambda a: a.double())
    where = staticmethod(lambda m, a, b: torch.where(m, a, torch.zeros((), dtype=a.dtype)))
    causal = staticmethod(lambda n: torch.tril(torch.ones((n, n), dtype=torch.bool)))



def gen_inputs(case, seed, instance):
    """The v1 case/seed stream, including the dt/decay construction."""
    base = "normal" if case == "ieee" else case
    mag = _MAGS.get(base, _DEFAULT_MAG)
    symbols = instance.symbols
    B = int(symbols["BATCH"])
    S = int(symbols["SEQLEN"])
    H = int(symbols["NHEADS"])
    P = int(symbols["HEADDIM"])
    N = int(symbols["DSTATE"])
    chunk = int(symbols["CHUNK"])
    nc = S // chunk
    dt = _round_to(_DT_LO + (_DT_HI - _DT_LO) * torch.sigmoid(_sample("normal", (B, S, H), seed + _OFFSETS["dt"], instance.dtype)), instance.dtype)
    a = -(_DECAY_LO + (_DECAY_HI - _DECAY_LO) * torch.sigmoid(_sample("normal", (H,), seed + _OFFSETS["decay"], instance.dtype)))
    cs = (dt.float().reshape(B, nc, chunk, H) * a).cumsum(dim=2).permute(0, 3, 1, 2).contiguous()
    out = {
        "x": _sample(base, (B, S, H, P), seed + _OFFSETS["x"], instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE),
        "dt": dt,
        "dA_cumsum": cs,
        "Bmat": _sample(base, (B, S, N), seed + _OFFSETS["Bmat"], instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE),
        "Cmat": _sample(base, (B, S, N), seed + _OFFSETS["Cmat"], instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE),
        "Dskip": _sample(base, (H,), seed + _OFFSETS["Dskip"], instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE),
        "states": _sample(base, (B, nc, H, P, N), seed + _OFFSETS["states"], instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE),
    }
    if case == "ieee":
        # v1 declares cases.inputs.ieee._inject = ["states"], at seed + 104729 * 7.
        out["states"] = _inject(out["states"], seed + 7 * _IEEE_SEED_STRIDE)
    return out


def reference(inputs, attrs):
    """The paper listing-1 scan, evaluated in fp64."""
    x = inputs["x"]
    batch, seq, heads, headdim = (int(v) for v in x.shape)
    kw = dict(
        batch=batch, seq=seq, heads=heads, headdim=headdim,
        dstate=int(inputs["states"].shape[-1]), chunk=int(inputs["dA_cumsum"].shape[-1]),
    )
    y = ssd_chunk_scan(
        *(inputs[key] for key in ("x", "dt", "dA_cumsum", "Bmat", "Cmat", "Dskip", "states")),
        ops=_RefOps, **kw,
    )
    return {"y": y}
