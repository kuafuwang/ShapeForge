from __future__ import annotations
import tilelang
import tilelang.metal.language as T

N = 1024
THREADS = 128

@T.prim_func
def scale_kernel(x: T.Tensor((N,), "float32"), tmp: T.Tensor((N,), "float32")):
    with T.Kernel(N // THREADS, threads=THREADS) as bx:
        for i in T.Parallel(THREADS):
            index = bx * THREADS + i
            tmp[index] = x[index] * 2.0
            tmp[index] = tmp[index] + 0.0

@T.prim_func
def bias_kernel(
    tmp: T.Tensor((N,), "float32"),
    acc: T.Tensor((N,), "float32"),
    y: T.Tensor((N,), "float32"),
):
    with T.Kernel(N // THREADS, threads=THREADS) as bx:
        for i in T.Parallel(THREADS):
            index = bx * THREADS + i
            value = tmp[index] + 1.0
            tmp[index] = value
            y[index] = value
            acc[index] = acc[index] + value

KERNELS = (scale_kernel, bias_kernel)
