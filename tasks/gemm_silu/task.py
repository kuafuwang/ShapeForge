# 矩阵乘后再做 SiLU。description：C = silu(A @ B)，silu(v) = v * sigmoid(v)，作用在 fp32 累加器上，再舍入一次到元素类型。
# 标准答案沿用 gemm 的 reference：A、B 先用 float 做 @，转成 double 后交给本文件的 epilogue_ref，即 c * torch.sigmoid(c)。
# 基线是 baselines：mlx 把 a @ b 送进 mx.compile 的 t * mx.sigmoid(t)；torch_mps 是 F.silu(torch.matmul(a, b))。
# 启动契约在 build_gemm 里。核名 gemm_silu_kernel，缓冲 A/B/C，grid、threads 以及常量 M/N/K/BLOCK_* 都由环境锁定。
import torch
import torch.nn.functional as F

from tcenv.task import BaselineImpl
from tasks._common import mlx_compiled
from tasks.gemm.task import build_gemm

# 这段填进 gemm/naive.metal 的结尾定义，朴素 Metal 用 silu_f 做乘积之后的 SiLU。模型读到的 Triton 题面是 kernel_triton.py。
SIGMOID_DEF = """inline float sigmoid_f(float v) {
    float e = exp(-fabs(v));
    return v >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}
inline float silu_f(float v) { return v * sigmoid_f(v); }
"""


def _mlx(a, b):
    return mlx_compiled("silu", lambda mx: (lambda t: t * mx.sigmoid(t)))(a @ b)


def _torch(a, b):
    return F.silu(torch.matmul(a, b))


def build(dtype: str, size: str):
    return build_gemm(
        "gemm_silu", dtype, size,
        epilogue_ref=lambda c: c * torch.sigmoid(c),
        epilogue_msl="silu_f", epilogue_def=SIGMOID_DEF,
        baselines=(BaselineImpl("mlx", "mlx", _mlx), BaselineImpl("torch_mps", "torch", _torch)),
        description="C = silu(A @ B), silu(v) = v * sigmoid(v) applied to the fp32 accumulator before the single rounding to the element type.",
        extra_flops=5.0,
    )
