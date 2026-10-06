# 按行 softmax。description：y = exp(x - max(x)) / sum(exp(x - max(x)))，统计量在 fp32，结果舍入一次。SHAPES 的 paper 是 (8192, 2048)。
# 标准答案是 reference：对转成 float64 的 x 做 torch.softmax(..., dim=-1)。
# 基线是 baselines：mlx 的 mx.softmax(axis=-1, precise=True)，torch_mps 的 torch.softmax(dim=-1)。
# 模型不能改 contract：核名 softmax_kernel，grid=(M, 1, 1)，threads=(256, 1, 1)，缓冲 x/y，常量 M、N。
import torch

from tcenv.contract import AbiArg, Contract
from tcenv.distributions import sample
from tcenv.task import BaselineImpl, Task, default_tolerances
from tasks._common import check_dtype, load_text

SHAPES = {"paper": (8192, 2048), "small": (64, 512)}
MAGS = {"normal": 2.0}


def _mlx(x):
    import mlx.core as mx

    return mx.softmax(x, axis=-1, precise=True)


def _torch(x):
    return torch.softmax(x, dim=-1)


def build(dtype: str, size: str) -> Task:
    check_dtype(dtype)
    M, N = SHAPES[size]
    contract = Contract(
        kernel_name="softmax_kernel",
        grid=(M, 1, 1),
        threads=(256, 1, 1),
        args=(AbiArg("x", 0, "input", dtype, (M, N), doc="logits, row-major [M, N]"), AbiArg("y", 1, "output", dtype, (M, N))),
        constants=(("M", "uint", M), ("N", "uint", N)),
    )

    def gen_inputs(case, seed):
        return {"x": sample(case, (M, N), seed, dtype, mag=MAGS.get(case, 1.0), large=1.0e4)}

    def reference(inp):
        return {"y": torch.softmax(inp["x"].double(), dim=-1)}

    rtol, atol_rel = default_tolerances(dtype)
    return Task(
        name="softmax", dtype=dtype, size=size, contract=contract,
        triton_source=load_text("softmax", "kernel_triton.py", dtype, {"M": str(M), "N": str(N)}),
        description="Numerically stable row-wise softmax: y = exp(x - max(x)) / sum(exp(x - max(x))); statistics in fp32, result rounded once.",
        cases=("normal", "wide_range", "large", "tiny", "outlier", "const", "ieee"),
        gen_inputs=gen_inputs, reference=reference,
        baselines=(BaselineImpl("mlx", "mlx", _mlx), BaselineImpl("torch_mps", "torch", _torch)),
        flops=5.0 * M * N, min_bytes=float(2 * M * N * 2),
        rtol=rtol, atol_rel=atol_rel,
        naive_msl=load_text("softmax", "naive.metal", dtype),
    )
