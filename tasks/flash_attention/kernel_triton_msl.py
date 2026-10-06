# Baseline variant for the triton-msl backend (same computation and I/O as kernel_triton.py). The backend recognizes FlashAttention only in the
# form of the Triton tutorial: runtime batch/head/row/column strides, runtime Z/H/N_CTX, the softmax scale folded into q before the first dot.
# Launch: grid = (N_CTX // BLOCK_M, Z * H), num_warps = {{NW}}, Z = {{B}}, H = {{H}}, N_CTX = {{S}}, HEAD_DIM = {{D}}, BLOCK_M = {{BM}}, BLOCK_N = {{BN}}, SM_SCALE = {{SM_SCALE}}
#         stride_qz = {{SZ}}, stride_qh = {{SH}}, stride_qm = {{SR}}, stride_qk = 1, stride_kz = {{SZ}}, stride_kh = {{SH}}, stride_kn = {{SR}}, stride_kk = 1
#         stride_vz = {{SZ}}, stride_vh = {{SH}}, stride_vn = {{SR}}, stride_vk = 1, stride_oz = {{SZ}}, stride_oh = {{SH}}, stride_om = {{SR}}, stride_on = 1
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, sm_scale, Z, H, N_CTX,
                           stride_qz, stride_qh, stride_qm, stride_qk,
                           stride_kz, stride_kh, stride_kn, stride_kk,
                           stride_vz, stride_vh, stride_vn, stride_vk,
                           stride_oz, stride_oh, stride_om, stride_on,
                           HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    qk_scale = sm_scale * 1.4426950408889634
    q = (q * qk_scale).to({{TL}})
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(k_ptr + off_z * stride_kz + off_h * stride_kh + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        qk = tl.dot(q, tl.trans(k))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(v_ptr + off_z * stride_vz + off_h * stride_vh + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        acc = tl.dot(p.to({{TL}}), v, acc)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(o_ptr + off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on, acc.to({{TL}}))
