# 行主序矩阵乘。语义只写在 opspec.json；默认 tile 与启动事实来自登记的 metal-gemm profile；
# 候选的 @<BM>x<BN>x<BK>t<threads> 由登记的 variant factory 解析，只换 tile/threads。
import torch

from tasks._matmul import build_matmul

TASK_DIR = "gemm"


def _mlx(params: dict, _epilogues: dict):
    """@brief mlx 侧矩阵乘基线（``a @ b``）。

    @param params track 登记参数。
    @param _epilogues 未使用（恒等收尾）。
    @return ``(a, b) -> a @ b``。
    """
    if params.get("kernel") != "matmul":
        raise ValueError(f"gemm mlx baseline registration changed: {params}")
    return lambda a, b: a @ b


def _torch(params: dict, _epilogues: dict):
    """@brief torch 侧矩阵乘基线（``torch.matmul``）。

    @param params track 登记参数。
    @param _epilogues 未使用（恒等收尾）。
    @return ``(a, b) -> torch.matmul(a, b)``。
    """
    if params.get("kernel") != "torch.matmul":
        raise ValueError(f"gemm torch baseline registration changed: {params}")
    return lambda a, b: torch.matmul(a, b)


def build(dtype: str, size: str):
    """@brief 装配矩阵乘题（默认 tile）。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    return build_matmul(
        TASK_DIR,
        dtype,
        size,
        epilogues={"identity": lambda c: c},
        baseline_builders={"mlx": _mlx, "torch_mps": _torch},
    )


def build_variant(dtype: str, size: str, variant: str):
    """@brief 同一算子、同一 reference 与 baselines，只换登记 factory 解析出的 tile/threads。

    @param dtype 元素类型。
    @param size 尺寸档位。
    @param variant ``<BM>x<BN>x<BK>t<threads>``。
    @return 装配好的 ``Task``，名字带 ``@<variant>`` 后缀。
    """
    return build_matmul(
        TASK_DIR,
        dtype,
        size,
        epilogues={"identity": lambda c: c},
        baseline_builders={"mlx": _mlx, "torch_mps": _torch},
        variant=variant,
    )
