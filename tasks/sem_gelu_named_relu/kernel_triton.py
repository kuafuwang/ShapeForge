# Triton specification: ReLU activation, element type {{TL}}
# Launch: grid = (N // BLOCK_SIZE,), num_warps = {{NUM_WARPS}}, N = {{N}}, BLOCK_SIZE = {{BLOCK_SIZE}}
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """ReLU: y = max(x, 0)"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    # rectified linear unit
    t = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    e = tl.exp(-2.0 * tl.abs(t))
    th = tl.where(t >= 0, (1.0 - e) / (1.0 + e), (e - 1.0) / (1.0 + e))
    y = 0.5 * x * (1.0 + th)
    tl.store(y_ptr + offsets, y.to({{TL}}), mask=mask)
