# 逐元素 SiLU。语义只写在 opspec.json；契约与启动事实来自登记的 Metal LoweringProfile。
import torch
import torch.nn.functional as F

from tasks._common import mlx_compiled
from tasks._pointwise import build_pointwise

TASK_DIR = "silu"


def _mlx(params: dict):
    """@brief mlx 侧 SiLU 基线，整个式子走一次 ``mx.compile``。

    @param params track 登记参数。
    @return ``x -> mx.compile(a * mx.sigmoid(a))(x)``。
    """
    if params.get("kernel") != "mx.compile" or params.get("expression") != "a * mx.sigmoid(a)":
        raise ValueError(f"silu mlx baseline registration changed: {params}")
    return lambda x: mlx_compiled("silu", lambda mx: (lambda a: a * mx.sigmoid(a)))(x)


def _torch(params: dict):
    """@brief torch 侧 SiLU 基线（``F.silu``）。

    @param params track 登记参数。
    @return ``x -> F.silu(x)``。
    """
    if params.get("kernel") != "torch.nn.functional.silu":
        raise ValueError(f"silu torch baseline registration changed: {params}")
    return lambda x: F.silu(x)


def build(dtype: str, size: str):
    """@brief 装配逐元素 SiLU 题。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    return build_pointwise(
        TASK_DIR,
        dtype,
        size,
        references={"silu": lambda x: x * torch.sigmoid(x)},
        baseline_builders={"mlx": _mlx, "torch_mps": _torch},
    )
