"""Performance baselines for bitdelta_gemm: separate or compiled decode + dense matmul.

A baseline is a timing comparison object, never the correctness oracle.  The packed words arrive
as the framework-native integer tensor; the decode follows the v1 registration (bit j of word kb
is row kb*32 + j), and K/N come from the operand shapes, exactly as the v1 closures captured them.
"""

from __future__ import annotations

_SCALE = 0.02  # the frozen OpSpec attr default
_COMPILED: dict = {}


def _compiled(key, builder):
    """@brief 每个 key 只编译一次（与 v1 的编译缓存一致）。"""
    fn = _COMPILED.get(key)
    if fn is None:
        fn = _COMPILED[key] = builder()
    return fn


def _mlx_decode(x, bits):
    import mlx.core as mx
    k, n = x.shape[1], bits.shape[1]
    j = mx.arange(32, dtype=mx.uint32).reshape(1, 32, 1)
    b = (bits.reshape(k // 32, 1, n) >> j) & 1
    w = (b.astype(x.dtype) * 2 - 1).reshape(k, n)
    return (x @ w) * _SCALE


def _torch_decode(x, bits):
    import torch
    k, n = x.shape[1], bits.shape[1]
    j = torch.arange(32, dtype=torch.int32, device=x.device).reshape(1, 32, 1)
    b = (bits.reshape(k // 32, 1, n) >> j) & 1
    w = (b.to(x.dtype) * 2 - 1).reshape(k, n)
    return (x @ w) * _SCALE


def mlx(x, bits):
    """v1 baseline mlx: separate unpack + dense matmul."""
    return _mlx_decode(x, bits)


def mlx_compile(x, bits):
    """v1 baseline mlx_compile: the decode is compiled once per (K, N)."""
    import mlx.core as mx
    return _compiled(f"bitdelta_{x.shape[1]}_{bits.shape[1]}", lambda: mx.compile(_mlx_decode))(x, bits)


def torch_mps(x, bits):
    """v1 baseline torch_mps: separate unpack + dense matmul."""
    return _torch_decode(x, bits)


def torch_compile(x, bits):
    """v1 baseline torch_compile: Inductor fuses the decode (the strong baseline)."""
    import torch
    return _compiled(f"bitdelta_torch_{x.shape[1]}_{bits.shape[1]}", lambda: torch.compile(_torch_decode))(x, bits)


BASELINES = {"mlx": mlx, "mlx_compile": mlx_compile, "torch_mps": torch_mps, "torch_compile": torch_compile}
