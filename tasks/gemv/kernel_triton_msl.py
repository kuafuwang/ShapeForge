# Baseline variant for the triton-msl backend (same computation and I/O as kernel_triton.py): one program per row with a 1-D
# accumulator, because the backend's generic lowerer does not handle the 2-D block + tl.sum(axis=1) form.
# Launch: grid = (M,), num_warps = 4, M = {{M}}, K = {{K}}, BLOCK_K = 1024
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(a_ptr, x_ptr, y_ptr, M: tl.constexpr, K: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(axis=0)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + row * K + cols).to(tl.float32)
        v = tl.load(x_ptr + cols).to(tl.float32)
        acc += a * v                                   # fp32 accumulation, reduced once after the loop
    tl.store(y_ptr + row, tl.sum(acc, axis=0).to({{TL}}))
