# Triton specification: fused GEMM + SiLU epilogue, C = silu(A @ B), element type {{TL}}
# Layouts: A[M, K], B[K, N], C[M, N], all row-major contiguous.
# Launch: grid = (N // BLOCK_N, M // BLOCK_M), num_warps = 4, M = {{M}}, N = {{N}}, K = {{K}}, BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32
import triton
import triton.language as tl


@triton.jit
def gemm_silu_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K // BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    out = acc * (1.0 / (1.0 + tl.exp(-acc)))                      # SiLU applied to the fp32 accumulator
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], out.to({{TL}}))
