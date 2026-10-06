import torch

from tcenv.task import BaselineImpl
from tasks._elementwise import build_elementwise


def _mlx(x):
    import mlx.core as mx

    return mx.maximum(x, 0)


def build(dtype: str, size: str):
    return build_elementwise(
        "relu",
        dtype,
        size,
        input_names=("x",),
        ref_fn=lambda x: torch.relu(x),
        baselines=(BaselineImpl("mlx", "mlx", _mlx), BaselineImpl("torch_mps", "torch", torch.relu)),
        flops_per_elem=1,
        description="y[i] = max(x[i], 0). Values <= 0 map to +0.",
    )
