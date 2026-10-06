# 逐元素 ReLU。description：y[i] = max(x[i], 0)，小于等于 0 的值映到 +0。
# 标准答案是 ref_fn 里的 torch.relu。共享骨架先把输入转成 float64。
# 基线是 baselines：mlx 走 mx.maximum(x, 0)，torch_mps 走 torch.relu。
# 契约由 build_elementwise 按名字 "relu" 和 input_names=("x",) 生成。核名 relu_kernel、缓冲 x/y、grid、threads、常量 N 与 BLOCK_SIZE 模型都不能改。
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
