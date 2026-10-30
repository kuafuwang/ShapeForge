# Triton specification: GELU activation (tanh approximation), element type {{TL}}
# Launch: grid = (N // BLOCK_SIZE,), num_warps = {{NUM_WARPS}}, N = {{N}}, BLOCK_SIZE = {{BLOCK_SIZE}}
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """GELU: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    # apply the GELU nonlinearity
    y = tl.where(x > 0, x, 0.0)
    tl.store(y_ptr + offsets, y.to({{TL}}), mask=mask)
