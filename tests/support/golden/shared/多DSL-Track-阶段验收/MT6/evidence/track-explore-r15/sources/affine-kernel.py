from __future__ import annotations
import tilelang
import tilelang.metal.language as T
N=1000
NS=16
SEG=64
BASE=2049
STEP=2
THREADS=256
@T.prim_func
def affine_ramp(x: T.Tensor((N,), "float16"), y: T.Tensor((N,), "float16"), scale_tab: T.Tensor((NS,), "float32")):
    with T.Kernel(4, threads=THREADS) as bx:
        tid=T.get_thread_binding()
        i=bx*THREADS+tid
        if i<N:
            s=i//SEG
            scale_tab[s]=BASE+STEP*s
            y[i]=T.float16(scale_tab[s]*T.float32(x[i]))
KERNEL=affine_ramp
