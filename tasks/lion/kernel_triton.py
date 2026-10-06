# Triton specification: Lion optimizer step (Chen et al. 2023), element type {{TL}}
# Launch: grid = (N // BLOCK_SIZE,), num_warps = 4, N = {{N}}, BLOCK_SIZE = 1024, LR = 0.001, WD = 0.1, BETA1 = 0.9, BETA2 = 0.99
import triton
import triton.language as tl


@triton.jit
def lion_kernel(p_ptr, g_ptr, m_ptr, p_out_ptr, m_out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                LR: tl.constexpr, WD: tl.constexpr, BETA1: tl.constexpr, BETA2: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask).to(tl.float32)
    c = BETA1 * m + (1.0 - BETA1) * g
    update = tl.where(c > 0, 1.0, tl.where(c < 0, -1.0, 0.0))          # sign(c)
    p_new = p * (1.0 - LR * WD) - LR * update                           # decoupled weight decay + signed step
    m_new = BETA2 * m + (1.0 - BETA2) * g
    tl.store(p_out_ptr + offs, p_new.to({{TL}}), mask=mask)
    tl.store(m_out_ptr + offs, m_new.to({{TL}}), mask=mask)
