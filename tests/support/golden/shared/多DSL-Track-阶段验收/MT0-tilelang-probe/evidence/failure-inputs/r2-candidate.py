"""Fixed MT0 TileLang source-only probe candidate.

This file is candidate-owned ``embedded_host_code``: importing it executes
Python host code inside the hmz lower-level Seatbelt Profile.  It must never be
imported by the ShapeForge control plane, by ``verify.py`` or by shared tests.

The program is deliberately tiny and fixed: one fp32 vector scale over a static
1024-element shape, one threadgroup grid of 4 x 256 threads, Metal dialect
``T.Kernel`` and ``T.Parallel``.  It exists only to prove that the pinned
frontend can parse, lower and emit Metal Shading Language source.  It is not a
performance candidate and generates no device dispatch by itself.
"""

from __future__ import annotations

import tilelang  # noqa: F401  (frontend package identity is checked by the entry)
import tilelang.metal.language as T
from tilelang.language.tir import prim_func

N = 1024
THREADS = 256


@prim_func
def scale_kernel(
    a: T.Tensor((N,), "float32"),
    b: T.Tensor((N,), "float32"),
):
    with T.Kernel(N // THREADS, threads=THREADS) as bx:
        for i in T.Parallel(THREADS):
            b[bx * THREADS + i] = a[bx * THREADS + i] * 2.0
