import torch
import torch.nn.functional as F

from tcenv.task import BaselineImpl
from tasks._common import mlx_compiled
from tasks._elementwise import build_elementwise


def _mlx(g, u):
    return mlx_compiled("swiglu", lambda mx: (lambda a, b: (a * mx.sigmoid(a)) * b))(g, u)


def _torch(g, u):
    return F.silu(g) * u


def build(dtype: str, size: str):
    return build_elementwise(
        "swiglu",
        dtype,
        size,
        input_names=("gate", "up"),
        ref_fn=lambda g, u: g * torch.sigmoid(g) * u,
        baselines=(BaselineImpl("mlx", "mlx", _mlx), BaselineImpl("torch_mps", "torch", _torch)),
        flops_per_elem=7,
        large=200.0,
        description="y[i] = silu(gate[i]) * up[i] with silu(g) = g * sigmoid(g); evaluated in fp32, rounded once.",
    )
