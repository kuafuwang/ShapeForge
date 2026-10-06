# Triton specification: 2-D convolution (stride 1, no padding, no bias) as an implicit GEMM, element type {{TL}}
# Layouts (all contiguous): x[N, H, W, C] (NHWC), w[OC, KH, KW, C], y[N, OH, OW, OC].
# GEMM view: M_TOTAL = N*OH*OW rows (one per output pixel), OC columns, K_TOTAL = KH*KW*C reduction (order: kh, kw, c).
# Launch: grid = (OC // BLOCK_N, M_TOTAL // BLOCK_M), num_warps = 4
#         N = {{N}}, H = {{H}}, W = {{W}}, C = {{C}}, OC = {{OC}}, KH = 3, KW = 3, OH = {{OH}}, OW = {{OW}}, M_TOTAL = {{M_TOTAL}}, K_TOTAL = {{K_TOTAL}}, BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(x_ptr, w_ptr, y_ptr,
                  N: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr, OC: tl.constexpr,
                  KH: tl.constexpr, KW: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    K_TOTAL: tl.constexpr = KH * KW * C
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)               # output pixel index
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)               # output channel
    img = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh = rem // OW
    ow = rem % OW
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)           # fp32 accumulator
    for k0 in range(0, K_TOTAL, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        kh = offs_k // (KW * C)
        kw = (offs_k // C) % KW
        c = offs_k % C
        x_idx = ((img[:, None] * H + oh[:, None] + kh[None, :]) * W + ow[:, None] + kw[None, :]) * C + c[None, :]
        a = tl.load(x_ptr + x_idx)                                 # [BLOCK_M, BLOCK_K]
        b = tl.load(w_ptr + offs_n[None, :] * K_TOTAL + offs_k[:, None])   # [BLOCK_K, BLOCK_N]
        acc = tl.dot(a, b, acc)
    tl.store(y_ptr + offs_m[:, None] * OC + offs_n[None, :], acc.to({{TL}}))
