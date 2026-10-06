# 逐元素 SwiGLU。description：y[i] = silu(gate[i]) * up[i]，silu(g) = g * sigmoid(g)，fp32 计算后舍入一次。
# 标准答案是 ref_fn：g * torch.sigmoid(g) * u。共享骨架按 input_names 的顺序把 gate、up 转成 float64。
# 基线是 baselines：mlx 是 mx.compile 的 (a * mx.sigmoid(a)) * b，torch_mps 是 F.silu(g) * u。
# 契约由 build_elementwise 按名字 "swiglu" 和 input_names=("gate", "up") 生成。核名 swiglu_kernel，缓冲 gate、up、y，以及 grid、threads、常量 N 与 BLOCK_SIZE，模型都不能改。
import torch
import torch.nn.functional as F

from tcenv.task import BaselineImpl
from tasks._common import mlx_compiled
from tasks._elementwise import build_elementwise


def _mlx(g, u):
    return mlx_compiled("swiglu", lambda mx: (lambda a, b: (a * mx.sigmoid(a)) * b))(g, u)


def _torch(g, u):
    return F.silu(g) * u


def build(dtype: str, size: str):
    return build_elementwise(
        "swiglu",
        dtype,
        size,
        input_names=("gate", "up"),
        ref_fn=lambda g, u: g * torch.sigmoid(g) * u,
        baselines=(BaselineImpl("mlx", "mlx", _mlx), BaselineImpl("torch_mps", "torch", _torch)),
        flops_per_elem=7,
        large=200.0,
        description="y[i] = silu(gate[i]) * up[i] with silu(g) = g * sigmoid(g); evaluated in fp32, rounded once.",
    )
