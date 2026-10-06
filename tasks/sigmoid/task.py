# 逐元素 sigmoid。description：y[i] = 1 / (1 + exp(-x[i]))，在 fp32 里算完，再舍入到元素类型一次。
# 标准答案是 ref_fn 里的 torch.sigmoid。共享骨架会先把输入转成 float64，再调用这个函数。
# 基线是 baselines：mlx 走 mx.sigmoid，torch_mps 走 torch.sigmoid。
# 契约由 build_elementwise 按名字 "sigmoid" 和 input_names=("x",) 生成。核名、缓冲 x/y、grid、threads、常量 N 与 BLOCK_SIZE 模型都不能改。
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
