# Triton specification: RMSNorm, element type {{TL}}
# Launch: grid = (M,), num_warps = 8, M = {{M}}, H = {{H}}, BLOCK_H = {{H}}, eps = 1e-6
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, H: tl.constexpr, EPS: tl.constexpr, BLOCK_H: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_H)
    x = tl.load(x_ptr + row * H + cols).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / H            # reduction in fp32
    rstd = 1.0 / tl.sqrt(mean_sq + EPS)
    w = tl.load(w_ptr + cols).to(tl.float32)
    y = x * rstd * w
    tl.store(y_ptr + row * H + cols, y.to({{TL}}))
