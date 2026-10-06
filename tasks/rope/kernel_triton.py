# Triton specification: rotary position embedding (GPT-NeoX / half-split style), element type {{TL}}
# Layout: x[B, H, S, D] contiguous. Position of an element is its index along S.
# Launch: grid = (S // BLOCK_S, B * H), num_warps = 4, B = {{B}}, H = {{H}}, S = {{S}}, D = {{D}}, BLOCK_S = 16, BASE = 10000.0
import triton
import triton.language as tl


@triton.jit
def rope_kernel(x_ptr, y_ptr, S: tl.constexpr, D: tl.constexpr, BASE: tl.constexpr, BLOCK_S: tl.constexpr):
    HALF: tl.constexpr = D // 2
    pid_s = tl.program_id(axis=0)
    bh = tl.program_id(axis=1)
    pos = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)                     # [BLOCK_S]
    i = tl.arange(0, HALF)                                             # [HALF]
    inv_freq = tl.exp(-(i.to(tl.float32) / HALF) * tl.log(BASE))       # BASE ** (-i / HALF)
    ang = pos.to(tl.float32)[:, None] * inv_freq[None, :]              # [BLOCK_S, HALF]
    c, s = tl.cos(ang), tl.sin(ang)
    row = (bh * S + pos)[:, None] * D                                  # [BLOCK_S, 1]
    x1 = tl.load(x_ptr + row + i[None, :]).to(tl.float32)
    x2 = tl.load(x_ptr + row + HALF + i[None, :]).to(tl.float32)
    tl.store(y_ptr + row + i[None, :], (x1 * c - x2 * s).to({{TL}}))
    tl.store(y_ptr + row + HALF + i[None, :], (x1 * s + x2 * c).to({{TL}}))
