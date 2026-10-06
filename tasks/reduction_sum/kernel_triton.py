# Triton specification: row-wise sum reduction, element type {{TL}}
# Launch: grid = (M,), num_warps = 4, M = {{M}}, N = {{N}}, BLOCK_N = {{N}}
import triton
import triton.language as tl


@triton.jit
def reduction_sum_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + row * N + cols).to(tl.float32)
    s = tl.sum(x, axis=0)                      # fp32 accumulation
    tl.store(y_ptr + row, s.to({{TL}}))
