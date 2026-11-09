from __future__ import annotations
import tilelang
import tilelang.metal.language as T

N = 1024
THREADS = 256

@T.prim_func
def scale_kernel(
    x: T.Tensor((N,), "float32"),
    acc: T.Tensor((N,), "float32"),
    y: T.Tensor((N,), "float32"),
):
    with T.Kernel(N // THREADS, threads=THREADS) as bx:
        for i in T.Parallel(THREADS):
            index = bx * THREADS + i
            value = x[index] * 2.0 + 1.0
            y[index] = value
            acc[index] = acc[index] + value

KERNEL = scale_kernel
