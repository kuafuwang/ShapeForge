"""Performance baselines for mamba2_chunk_scan: the SSD formulation on mlx and torch.

A baseline is a timing comparison object, never the correctness oracle.  The operator is
inlined for the same reason as in reference.py: these modules may not import the harness.
"""

from __future__ import annotations

import torch


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




def _kwargs():
    """@brief small/paper 的实例符号在调用处给出；这里只是缓存编译键的辅助。

    @return 空字典（真正的形状在 _run 里计算）。
    """
    return {}


def _shape_kwargs(x, states, chunk):
    """@brief 由输入形状推出 ssd_chunk_scan 的关键字（与 v1 的实例符号一致）。

    @param x 输入张量。
    @param states 每块开始时的 state。
    @param chunk 每块时间步数。
    @return 形状关键字字典。
    """
    batch, seq, heads, headdim = (int(v) for v in x.shape)
    return dict(batch=batch, seq=seq, heads=heads, headdim=headdim, dstate=int(states.shape[-1]), chunk=int(chunk))


def _run(args, ops, half):
    """@brief 按 v1 的调用形式执行一次 scan，并转回输入 dtype。

    @param args 位置参数（x/dt/dA_cumsum/Bmat/Cmat/Dskip/states）。
    @param ops 算子集。
    @param half 矩阵乘操作数是否保持输入 dtype。
    @return 输出张量。
    """
    x, dt, cs, bm, cm, d, st = args
    chunk = int(cs.shape[-1])
    kw = _shape_kwargs(x, st, chunk)
    return ssd_chunk_scan(*args, ops=ops, half=half, **kw).to(x.dtype) if ops is _TorchOps or ops is _RefOps else ssd_chunk_scan(*args, ops=ops, half=half, **kw).astype(x.dtype)


def mlx(*args):
    """v1 baseline mlx: fp32 formulation, eager."""
    return _run(args, _MlxOps, False)


def mlx_compile(*args):
    """v1 baseline mlx_compile: the whole formulation compiled once."""
    import mlx.core as mx
    return _compiled_mlx("mamba2_full", False)(*args)


def mlx_half_compile(*args):
    """v1 baseline mlx_half_compile: matmul operands stay in the input dtype."""
    import mlx.core as mx
    return _compiled_mlx("mamba2_half", True)(*args)


def _compiled_mlx(key, half):
    """@brief 编译并缓存一条 mlx scan（与 v1 的 mlx_compiled 一致）。

    @param key 缓存键。
    @param half 操作数是否保持输入 dtype。
    @return 编译后的闭包。
    """
    import mlx.core as mx
    cache = _compiled_mlx.cache
    if key not in cache:
        cache[key] = mx.compile(lambda *a: _run(a, _MlxOps, half))
    return cache[key]


_compiled_mlx.cache = {}


def torch_mps(*args):
    """v1 baseline torch_mps: fp32 formulation, eager."""
    return _run(args, _TorchOps, False)


def torch_half(*args):
    """v1 baseline torch_half: matmul operands stay in the input dtype."""
    return _run(args, _TorchOps, True)


BASELINES = {
    "mlx": mlx,
    "mlx_compile": mlx_compile,
    "mlx_half_compile": mlx_half_compile,
    "torch_mps": torch_mps,
    "torch_half": torch_half,
}
