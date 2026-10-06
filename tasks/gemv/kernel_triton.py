# Triton specification: matrix-vector product y = A @ x, element type {{TL}}
# Launch: grid = (M // BLOCK_M,), num_warps = 4, M = {{M}}, K = {{K}}, BLOCK_M = 4, BLOCK_K = 1024
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(a_ptr, x_ptr, y_ptr, M: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + rows[:, None] * K + cols[None, :]).to(tl.float32)
        v = tl.load(x_ptr + cols).to(tl.float32)
        acc += tl.sum(a * v[None, :], axis=1)          # fp32 accumulation
    tl.store(y_ptr + rows, acc.to({{TL}}))
