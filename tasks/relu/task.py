# 逐元素 ReLU。语义只写在 opspec.json；契约与启动事实来自登记的 Metal LoweringProfile。
# 这里只登记参考实现、mlx/torch 基线与输入生成参数（track.json），不构造 Contract。
import torch

from tasks._pointwise import build_pointwise

TASK_DIR = "relu"


def _mlx(params: dict):
    """@brief mlx 侧 ReLU 基线（``mx.maximum(x, 0)``）。

    @param params track 登记参数。
    @return ``x -> mx.maximum(x, 0)``。
    """
    if params.get("kernel") != "mx.maximum" or float(params.get("other", -1.0)) != 0.0:
        raise ValueError(f"relu mlx baseline registration changed: {params}")

    def _mlx_relu(x):
        import mlx.core as mx

        return mx.maximum(x, 0)

    return _mlx_relu


def _torch(params: dict):
    """@brief torch 侧 ReLU 基线（``torch.relu``）。

    @param params track 登记参数。
    @return ``x -> torch.relu(x)``。
    """
    if params.get("kernel") != "torch.relu":
        raise ValueError(f"relu torch baseline registration changed: {params}")
    return lambda x: torch.relu(x)


def build(dtype: str, size: str):
    """@brief 装配逐元素 ReLU 题。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    return build_pointwise(
        TASK_DIR,
        dtype,
        size,
        references={"relu": torch.relu},
        baseline_builders={"mlx": _mlx, "torch_mps": _torch},
    )
