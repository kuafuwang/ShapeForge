# 矩阵乘后再做 SiLU。语义只写在 opspec.json；默认 tile 与启动事实来自登记的 metal-gemm-silu profile。
import torch
import torch.nn.functional as F

from tasks._common import mlx_compiled
from tasks._matmul import build_matmul

TASK_DIR = "gemm_silu"


def _mlx(params: dict, epilogues: dict):
    """@brief mlx 侧 GEMM+SiLU 基线，乘积送进一次 ``mx.compile`` 的 SiLU。

    @param params track 登记参数。
    @param epilogues 收尾函数表（用到 ``silu``）。
    @return ``(a, b) -> silu(a @ b)``。
    """
    if params.get("kernel") != "mx.compile" or params.get("expression") != "silu(a @ b)":
        raise ValueError(f"gemm_silu mlx baseline registration changed: {params}")
    silu = epilogues["silu"]
    return lambda a, b: mlx_compiled("silu", lambda mx: (lambda t: t * mx.sigmoid(t)))(a @ b)


def _torch(params: dict, _epilogues: dict):
    """@brief torch 侧 GEMM+SiLU 基线。

    @param params track 登记参数。
    @param _epilogues 未使用（``F.silu`` 自带实现）。
    @return ``(a, b) -> F.silu(torch.matmul(a, b))``。
    """
    if params.get("kernel") != "torch.nn.functional.silu" or params.get("combine") != "matmul":
        raise ValueError(f"gemm_silu torch baseline registration changed: {params}")
    return lambda a, b: F.silu(torch.matmul(a, b))


def build(dtype: str, size: str):
    """@brief 装配 GEMM+SiLU 题（默认 tile）。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    return build_matmul(
        TASK_DIR,
        dtype,
        size,
        epilogues={"silu": lambda c: c * torch.sigmoid(c)},
        baseline_builders={"mlx": _mlx, "torch_mps": _torch},
    )

