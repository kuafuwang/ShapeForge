# Triton specification: sigmoid, element type {{TL}}
# Launch: grid = (N // BLOCK_SIZE,), num_warps = {{NUM_WARPS}}, N = {{N}}, BLOCK_SIZE = {{BLOCK_SIZE}}
import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = 1.0 / (1.0 + tl.exp(-x.to(tl.float32)))  # computed in fp32
    tl.store(y_ptr + offsets, y.to({{TL}}), mask=mask)
