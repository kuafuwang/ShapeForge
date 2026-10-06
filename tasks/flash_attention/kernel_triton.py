# Triton specification: FlashAttention forward (non-causal), element type {{TL}}
# Layouts: q, k, v, o are [B, H, S, D] contiguous. One program handles BLOCK_M query rows of one (batch, head).
# Launch: grid = (S // BLOCK_M, B * H), num_warps = {{NW}}, B = {{B}}, H = {{H}}, S = {{S}}, D = {{D}}, BLOCK_M = {{BM}}, BLOCK_N = {{BN}}, SM_SCALE = {{SM_SCALE}}
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, S: tl.constexpr, D: tl.constexpr, SM_SCALE: tl.constexpr,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    bh = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    base = bh * S * D
    q = tl.load(q_ptr + base + offs_m[:, None] * D + offs_d[None, :])               # [BLOCK_M, D]
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    qk_scale = {{QK_SCALE_EXPR}}                                                    # log2(e) is folded in so exp2 can be used
    for start_n in range(0, S, BLOCK_N):
        k = tl.load(k_ptr + base + (start_n + offs_n)[None, :] * D + offs_d[:, None])   # [D, BLOCK_N]
        qk = tl.dot(q, k) * qk_scale                                                # fp32 scores
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(v_ptr + base + (start_n + offs_n)[:, None] * D + offs_d[None, :])   # [BLOCK_N, D]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(o_ptr + base + offs_m[:, None] * D + offs_d[None, :], acc.to({{TL}}))
