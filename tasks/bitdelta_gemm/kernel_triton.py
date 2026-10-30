# Triton specification: GEMM with 1-bit packed weights (BitDelta-style): y = scale * (x @ W), W[k, n] = +1 if bit k of column n is set else -1
# Layouts: x[M, K] row-major, bits[K // 32, N] uint32 (word kb of column n holds the signs of rows kb*32 .. kb*32+31, bit j = row kb*32 + j), y[M, N].
# Launch: grid = (N // BLOCK_N, M // BLOCK_M), num_warps = {{NW}}, M = {{M}}, N = {{N}}, K = {{K}}, BLOCK_M = {{BM}}, BLOCK_N = {{BN}}, BLOCK_K = {{BK}}
import triton
import triton.language as tl


@triton.jit
def bitdelta_gemm_kernel(x_ptr, bits_ptr, scale, y_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)                                   # BLOCK_K == 32: one packed word per column per step
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, K // BLOCK_K):
        a = tl.load(x_ptr + offs_m[:, None] * K + (kb * BLOCK_K + offs_k)[None, :])          # [BLOCK_M, 32]
        words = tl.load(bits_ptr + kb * N + offs_n)                                           # [BLOCK_N] uint32
        bit = (words[None, :] >> offs_k[:, None].to(tl.uint32)) & 1                           # [32, BLOCK_N]
        w = (bit.to(tl.float32) * 2.0 - 1.0).to({{TL}})                                       # +-1 in the element type
        acc = tl.dot(a, w, acc)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], (acc * scale).to({{TL}}))
