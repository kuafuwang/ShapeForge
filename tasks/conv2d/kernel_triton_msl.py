# Baseline variant for the triton-msl backend. The backend refuses the implicit-GEMM kernel ("K-loop matmul: A operand stride could not be
# inferred": the operand is gathered through img/oh/ow index arithmetic). What a Triton user gets through that compiler is an explicit
# im2col: a gather kernel that materializes the [M_TOTAL, K_TOTAL] patch matrix, a plain GEMM on it, and the (tiny) weight transpose.
# Launch: N = {{N}}, H = {{H}}, W = {{W}}, C = {{C}}, OC = {{OC}}, OH = {{OH}}, OW = {{OW}}, M_TOTAL = {{M_TOTAL}}, K_TOTAL = {{K_TOTAL}}, num_warps = 4
import torch
import triton
import triton.language as tl


@triton.jit
def im2col_kernel(x_ptr, col_ptr, H, W, C, OH, OW, K_TOTAL,
                  KW: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    offs_m = tl.program_id(axis=0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.program_id(axis=1) * BLOCK_K + tl.arange(0, BLOCK_K)
    img = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh = rem // OW
    ow = rem % OW
    kh = offs_k // (KW * C)
    kw = (offs_k // C) % KW
    c = offs_k % C
    x_idx = ((img[:, None] * H + oh[:, None] + kh[None, :]) * W + ow[:, None] + kw[None, :]) * C + c[None, :]
    tl.store(col_ptr + offs_m[:, None] * K_TOTAL + offs_k[None, :], tl.load(x_ptr + x_idx))


@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
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
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to({{TL}}))


def pipeline(inp, num_warps):
    x, w = inp["x"], inp["w"]
    H, W, C, OC, OH, OW, M_TOTAL, K_TOTAL = {{H}}, {{W}}, {{C}}, {{OC}}, {{OH}}, {{OW}}, {{M_TOTAL}}, {{K_TOTAL}}
    col = torch.empty((M_TOTAL, K_TOTAL), dtype={{TORCH}}, device="mps")
    y = torch.empty((M_TOTAL, OC), dtype={{TORCH}}, device="mps")
    wt = w.reshape(OC, K_TOTAL).t().contiguous()
    im2col_kernel[(M_TOTAL // 64, K_TOTAL // 32)](x, col, H, W, C, OH, OW, K_TOTAL, KW=3, BLOCK_M=64, BLOCK_K=32, num_warps=num_warps)
    gemm_kernel[(M_TOTAL // 64, OC // 64)](col, wt, y, M_TOTAL, OC, K_TOTAL, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=num_warps)
    return y.reshape(x.shape[0], OH, OW, OC)
