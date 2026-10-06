# Triton specification: row-wise softmax, element type {{TL}}
# Launch: grid = (M,), num_warps = 8, M = {{M}}, N = {{N}}, BLOCK_N = {{N}}
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + row * N + cols).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(y_ptr + row * N + cols, y.to({{TL}}))
