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
    """@brief 把 dtype 名称或对象统一成 ``DType``。

    @param dtype 类型名或 ``DType`` 实例。
    @return 对应的 ``DType``。
    """
    return dtype if isinstance(dtype, DType) else get_dtype(dtype)


def round_to(t: torch.Tensor, dtype: str | DType) -> torch.Tensor:
    """@brief 把张量按目标类型舍入后再加宽回逻辑类型。

    Round ``t`` to ``dtype`` and return it widened to float32 (or int32 for ints).

    @param t 输入张量。
    @param dtype 目标存储类型名或 ``DType``。
    @return 浮点加宽为 float32，整数保持其 torch 宽度。
    """
    d = _dt(dtype)
    if d.is_float:
        return t.to(d.torch).to(torch.float32)
    return t.to(d.torch)


def to_raw(t: torch.Tensor, dtype: str | DType) -> bytes:
    """@brief 把张量序列化成 ``dtype`` 存储格式的小端字节。

    Serialize ``t`` as little-endian bytes in ``dtype`` storage format.

    @param t 输入张量，会先转成 CPU 连续布局。
    @param dtype 目标存储类型名或 ``DType``。
    @return 原始字节串。
    """
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
    """@brief ``to_raw`` 的逆操作：把原始字节还原成逻辑张量。

    Inverse of ``to_raw``. Floats come back as float32; ints keep their width (u32 -> int64).

    @param raw 原始字节、bytearray、memoryview 或 numpy 数组。
    @param dtype 存储类型名或 ``DType``。
    @param shape 目标张量形状。
    @return 指定形状的张量；浮点数为 float32，u32 为 int64。
    @raises ValueError 字节数与形状和类型不匹配时抛出。
    """
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
    """@brief 计算指定形状与元素类型占用的字节数。

    @param shape 形状序列。
    @param dtype 元素类型名或 ``DType``。
    @return 元素个数乘以单元素字节数。
    """
    d = _dt(dtype)
    n = 1
    for s in shape:
        n *= int(s)
    return n * d.nbytes
