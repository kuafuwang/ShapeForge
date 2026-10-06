import torch
import torch.nn.functional as F

from tcenv.task import BaselineImpl
from tasks._common import mlx_compiled
from tasks.gemm.task import build_gemm

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
