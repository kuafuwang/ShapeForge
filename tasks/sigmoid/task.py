# 逐元素 sigmoid。语义只写在 opspec.json；契约与启动事实来自登记的 Metal LoweringProfile。
import torch

from tasks._pointwise import build_pointwise

TASK_DIR = "sigmoid"


def _mlx(params: dict):
    """@brief mlx 侧 sigmoid 基线（``mx.sigmoid``）。

    @param params track 登记参数。
    @return ``x -> mx.sigmoid(x)``。
    """
    if params.get("kernel") != "mx.sigmoid":
        raise ValueError(f"sigmoid mlx baseline registration changed: {params}")

    def _mlx_sigmoid(x):
        import mlx.core as mx

        return mx.sigmoid(x)

    return _mlx_sigmoid


def _torch(params: dict):
    """@brief torch 侧 sigmoid 基线（``torch.sigmoid``）。

    @param params track 登记参数。
    @return ``x -> torch.sigmoid(x)``。
    """
    if params.get("kernel") != "torch.sigmoid":
        raise ValueError(f"sigmoid torch baseline registration changed: {params}")
    return lambda x: torch.sigmoid(x)


def build(dtype: str, size: str):
    """@brief 装配逐元素 sigmoid 题。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    return build_pointwise(
        TASK_DIR,
        dtype,
        size,
        references={"sigmoid": torch.sigmoid},
        baseline_builders={"mlx": _mlx, "torch_mps": _torch},
    )
