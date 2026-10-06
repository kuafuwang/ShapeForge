# Baseline variant for the triton-msl backend. The backend refuses a matmul whose epilogue is fused into the K-loop kernel ("split the
# epilogue into a separate kernel after the matmul"), so this is what a Triton user gets through that compiler: a GEMM that stores the
# fp32 accumulator, then a SiLU kernel that rounds once to the element type (the numerics of the fused operator are preserved; the
# price is an extra fp32 round trip through memory).
# Launch: M = {{M}}, N = {{N}}, K = {{K}}, BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32, num_warps = 4
import torch
import triton
import triton.language as tl


@triton.jit
def gemm_f32_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
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
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


@triton.jit
def silu_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(y_ptr + offs, (x * (1.0 / (1.0 + tl.exp(-x)))).to({{TL}}))


def pipeline(inp, num_warps):
    a, b = inp["A"], inp["B"]
    M, N, K = {{M}}, {{N}}, {{K}}
    tmp = torch.empty((M, N), dtype=torch.float32, device="mps")
    c = torch.empty((M, N), dtype={{TORCH}}, device="mps")
    gemm_f32_kernel[(M // 64, N // 64)](a, b, tmp, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=num_warps)
    silu_kernel[(M * N // 1024,)](tmp, c, BLOCK=1024, num_warps=num_warps)
    return c
