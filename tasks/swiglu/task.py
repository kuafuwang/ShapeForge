# 逐元素 SwiGLU。语义只写在 opspec.json；契约与启动事实来自登记的 Metal LoweringProfile。
import torch
import torch.nn.functional as F

from tasks._common import mlx_compiled
from tasks._pointwise import build_pointwise

TASK_DIR = "swiglu"


def _mlx(params: dict):
    """@brief mlx 侧 SwiGLU 基线，整个式子走一次 ``mx.compile``。

    @param params track 登记参数。
    @return ``(g, u) -> mx.compile((a * mx.sigmoid(a)) * b)(g, u)``。
    """
    if params.get("kernel") != "mx.compile" or params.get("expression") != "(a * mx.sigmoid(a)) * b":
        raise ValueError(f"swiglu mlx baseline registration changed: {params}")
    return lambda g, u: mlx_compiled("swiglu", lambda mx: (lambda a, b: (a * mx.sigmoid(a)) * b))(g, u)


def _torch(params: dict):
    """@brief torch 侧 SwiGLU 基线（``F.silu(g) * u``）。

    @param params track 登记参数。
    @return ``(g, u) -> F.silu(g) * u``。
    """
    if params.get("kernel") != "torch.nn.functional.silu" or params.get("combine") != "multiply":
        raise ValueError(f"swiglu torch baseline registration changed: {params}")
    return lambda g, u: F.silu(g) * u


def build(dtype: str, size: str):
    """@brief 装配逐元素 SwiGLU 题。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    return build_pointwise(
        TASK_DIR,
        dtype,
        size,
        references={"swiglu": lambda g, u: g * torch.sigmoid(g) * u},
        baseline_builders={"mlx": _mlx, "torch_mps": _torch},
    )
