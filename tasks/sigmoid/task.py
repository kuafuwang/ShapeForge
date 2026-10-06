import torch

from tcenv.task import BaselineImpl
from tasks._elementwise import build_elementwise


def _mlx(x):
    import mlx.core as mx

    return mx.sigmoid(x)


def build(dtype: str, size: str):
    return build_elementwise(
        "sigmoid",
        dtype,
        size,
        input_names=("x",),
        ref_fn=lambda x: torch.sigmoid(x),
        baselines=(BaselineImpl("mlx", "mlx", _mlx), BaselineImpl("torch_mps", "torch", torch.sigmoid)),
        flops_per_elem=4,
        description="y[i] = 1 / (1 + exp(-x[i])), evaluated in fp32 and rounded to the element type once.",
    )
