# 按行求和。description：y[r] = sum_c x[r, c]，核内用 fp32 累加（输入可以抵消），再舍入一次。SHAPES 有 paper、small、plan 三档。
# 标准答案是 reference：在 float64 上做 sum(-1)。error_scale 用各行的 sum(abs(x)) 当这行的误差尺度。
# 基线是 baselines：mlx 的 mx.sum(axis=-1)，torch_mps 的 torch.sum(dim=-1)。
# 模型不能改 contract：核名 reduction_sum_kernel，grid=(M, 1, 1)，threads=(128, 1, 1)，缓冲 x 和 y，常量 M、N。
import torch

from tcenv.contract import AbiArg, Contract
from tcenv.distributions import sample
from tcenv.task import BaselineImpl, Task, default_tolerances
from tasks._common import check_dtype, load_text

SHAPES = {"paper": (4096, 4096), "small": (64, 1024), "plan": (256, 4096)}  # "plan": 2 MB, a few microseconds: dispatch-bound, see README
MAGS = {"normal": 1.0, "wide_range": 0.1}


def _mlx(x):
    """@brief mlx 侧的按行求和基线。

    @param x 输入，mlx 数组。
    @return ``mx.sum(x, axis=-1)``。
    """
    import mlx.core as mx

    return mx.sum(x, axis=-1)


def _torch(x):
    """@brief torch 侧的按行求和基线。

    @param x 输入。
    @return ``torch.sum(x, dim=-1)``。
    """
    return torch.sum(x, dim=-1)


def build(dtype: str, size: str) -> Task:
    """@brief 装配按行求和题。

    核名 ``reduction_sum_kernel``，``grid=(M, 1, 1)``，``threads=(128, 1, 1)``，缓冲 x 和 y，常量 M、N。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper``、``small`` 或 ``plan``。
    @return 装配好的 ``Task``。
    """
    check_dtype(dtype)
    M, N = SHAPES[size]
    contract = Contract(
        kernel_name="reduction_sum_kernel",
        grid=(M, 1, 1),
        threads=(128, 1, 1),
        args=(AbiArg("x", 0, "input", dtype, (M, N), doc="row-major [M, N]"), AbiArg("y", 1, "output", dtype, (M,), doc="row sums")),
        constants=(("M", "uint", M), ("N", "uint", N)),
    )

    def gen_inputs(case, seed):
        return {"x": sample(case, (M, N), seed, dtype, mag=MAGS.get(case, 1.0), large=300.0)}

    def reference(inp):
        return {"y": inp["x"].double().sum(-1)}

    def error_scale(inp, ref):
        # 容差按各行绝对值和缩放：正确的 fp32 求和，误差跟着 sum(|x|) 走。
        # The error of a correct fp32 summation scales with sum |x|, not with |sum x| (cancellation).
        return {"y": inp["x"].double().abs().sum(-1)}

    rtol, _ = default_tolerances(dtype)
    return Task(
        name="reduction_sum", dtype=dtype, size=size, contract=contract,
        triton_source=load_text("reduction_sum", "kernel_triton.py", dtype, {"M": str(M), "N": str(N)}),
        description="y[r] = sum_c x[r, c], accumulated in fp32 (inputs may cancel catastrophically) and rounded once to the element type.",
        cases=("normal", "wide_range", "large", "tiny", "outlier", "cancel", "ieee"),
        gen_inputs=gen_inputs, reference=reference, error_scale=error_scale,
        baselines=(BaselineImpl("mlx", "mlx", _mlx), BaselineImpl("torch_mps", "torch", _torch)),
        flops=float(M * N), min_bytes=float((M * N + M) * 2),
        rtol=rtol, atol_rel=3.0e-6,
        naive_msl=load_text("reduction_sum", "naive.metal", dtype),
    )
