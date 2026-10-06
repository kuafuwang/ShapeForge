# 逐元素 SiLU。description：y[i] = x[i] * sigmoid(x[i])，在 fp32 里算完再舍入一次。
# 标准答案是 ref_fn：x * torch.sigmoid(x)。共享骨架先把 x 转成 float64。
# 基线是 baselines：mlx 用 mx.compile 包住 a * mx.sigmoid(a)，torch_mps 用 F.silu。
# 契约由 build_elementwise 按名字 "silu" 和 input_names=("x",) 生成。核名 silu_kernel、缓冲 x/y、grid、threads、常量 N 与 BLOCK_SIZE 模型都不能改。
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
