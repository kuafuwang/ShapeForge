# Triton specification: SwiGLU activation, element type {{TL}}
# Launch: grid = (N // BLOCK_SIZE,), num_warps = {{NUM_WARPS}}, N = {{N}}, BLOCK_SIZE = {{BLOCK_SIZE}}
import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(gate_ptr, up_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    g = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    u = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)
    y = g * (1.0 / (1.0 + tl.exp(-g))) * u
    tl.store(y_ptr + offsets, y.to({{TL}}), mask=mask)
