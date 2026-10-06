"""Conversions between torch tensors and the raw bytes that live in Metal buffers.

All host-side data is a CPU torch tensor. Floating point values are exchanged as
float32/float64 "logical" tensors; ``to_raw`` rounds them to the storage dtype and
``from_raw`` widens the stored bits back to float32.
"""

from __future__ import annotations

import numpy as np
import torch

from .dtypes import DType, get_dtype


def _dt(dtype: str | DType) -> DType:
    return dtype if isinstance(dtype, DType) else get_dtype(dtype)


def round_to(t: torch.Tensor, dtype: str | DType) -> torch.Tensor:
    """Round ``t`` to ``dtype`` and return it widened to float32 (or int32 for ints)."""
    d = _dt(dtype)
    if d.is_float:
        return t.to(d.torch).to(torch.float32)
    return t.to(d.torch)


def to_raw(t: torch.Tensor, dtype: str | DType) -> bytes:
    """Serialize ``t`` as little-endian bytes in ``dtype`` storage format."""
    d = _dt(dtype)
    t = t.detach().to("cpu").contiguous()
    if d.name == "u32":
        t = t.to(torch.int64).to(torch.int32)
    else:
        t = t.to(d.torch)
    if d.kind == "float" and d.nbytes == 2:
        return t.view(d.raw_torch).numpy().tobytes()
    return t.numpy().tobytes()


def from_raw(raw: bytes | bytearray | memoryview | np.ndarray, dtype: str | DType, shape) -> torch.Tensor:
    """Inverse of ``to_raw``. Floats come back as float32; ints keep their width (u32 -> int64)."""
    d = _dt(dtype)
    buf = bytearray(bytes(raw)) if not isinstance(raw, bytearray) else raw
    n = int(np.prod(shape)) if len(shape) else 1
    if len(buf) != n * d.nbytes:
        raise ValueError(f"raw has {len(buf)} bytes, expected {n * d.nbytes} for shape {tuple(shape)} {d.name}")
    if d.kind == "float" and d.nbytes == 2:
        t = torch.frombuffer(buf, dtype=d.raw_torch).view(d.torch).to(torch.float32)
    elif d.name == "u32":
        t = torch.frombuffer(buf, dtype=torch.int32).to(torch.int64) & 0xFFFFFFFF
    else:
        t = torch.frombuffer(buf, dtype=d.torch).clone()
        if d.is_float:
            t = t.to(torch.float32)
    return t.reshape(tuple(shape))


def nbytes_of(shape, dtype: str | DType) -> int:
    d = _dt(dtype)
    n = 1
    for s in shape:
        n *= int(s)
    return n * d.nbytes
