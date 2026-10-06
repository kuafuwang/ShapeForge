import torch
import torch.nn.functional as F

from tcenv.task import BaselineImpl
from tasks._common import mlx_compiled
from tasks._elementwise import build_elementwise


def _mlx(x):
    return mlx_compiled("silu", lambda mx: (lambda a: a * mx.sigmoid(a)))(x)


def build(dtype: str, size: str):
    return build_elementwise(
        "silu",
        dtype,
        size,
        input_names=("x",),
        ref_fn=lambda x: x * torch.sigmoid(x),
        baselines=(BaselineImpl("mlx", "mlx", _mlx), BaselineImpl("torch_mps", "torch", F.silu)),
        flops_per_elem=5,
        description="y[i] = x[i] * sigmoid(x[i]), evaluated in fp32 and rounded once.",
    )
