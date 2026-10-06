# 矩阵乘向量。description：y = A @ x，A 是行主序 [M, K]，核内乘积按 fp32 累加后舍入一次。BLOCK_M = 4。
# 标准答案是 reference：把 A 和 x 转成 double 再做 @。
# 基线是 baselines：mlx 的 a @ x，torch_mps 的 torch.mv。
# 模型不能改 contract：核名 gemv_kernel，grid=(M/BLOCK_M, 1, 1)，threads=(128, 1, 1)，缓冲 A/x/y，常量 M、K、BLOCK_M。
import torch

from tcenv.contract import AbiArg, Contract
from tcenv.distributions import sample
from tcenv.task import BaselineImpl, Task, default_tolerances
from tasks._common import check_dtype, load_text

BLOCK_M = 4
SHAPES = {"paper": (4096, 4096), "small": (256, 512)}
MAGS = {"normal": 1.0, "wide_range": 0.1}


def _mlx(a, x):
    return a @ x


def _torch(a, x):
    return torch.mv(a, x)


def build(dtype: str, size: str) -> Task:
    check_dtype(dtype)
    M, K = SHAPES[size]
    contract = Contract(
        kernel_name="gemv_kernel",
        grid=(M // BLOCK_M, 1, 1),
        threads=(128, 1, 1),
        args=(
            AbiArg("A", 0, "input", dtype, (M, K), doc="row-major [M, K]"),
            AbiArg("x", 1, "input", dtype, (K,)),
            AbiArg("y", 2, "output", dtype, (M,)),
        ),
        constants=(("M", "uint", M), ("K", "uint", K), ("BLOCK_M", "uint", BLOCK_M)),
    )

    def gen_inputs(case, seed):
        mag = MAGS.get(case, 1.0)
        return {
            "A": sample(case, (M, K), seed, dtype, mag=mag, large=20.0, spike=5.0),
            "x": sample(case, (K,), seed + 31, dtype, mag=mag, large=20.0, spike=5.0),
        }

    def reference(inp):
        return {"y": inp["A"].double() @ inp["x"].double()}

    rtol, atol_rel = default_tolerances(dtype)
    return Task(
        name="gemv", dtype=dtype, size=size, contract=contract,
        triton_source=load_text("gemv", "kernel_triton.py", dtype, {"M": str(M), "K": str(K)}),
        description="y = A @ x with A[M, K] row-major; products accumulated in fp32, result rounded once.",
        cases=("normal", "wide_range", "large", "tiny", "outlier", "ieee"),
        gen_inputs=gen_inputs, reference=reference,
        baselines=(BaselineImpl("mlx", "mlx", _mlx), BaselineImpl("torch_mps", "torch", _torch)),
        flops=2.0 * M * K, min_bytes=float((M * K + K + M) * 2),
        rtol=rtol, atol_rel=atol_rel,
        naive_msl=load_text("gemv", "naive.metal", dtype),
        # ieee 特殊值放在输入 A 上。triton_baseline_variant 是 Triton→MSL 编译器用的另一份题面，速度对照用 baselines。
        meta={"ieee_inputs": ("A",), "triton_baseline_variant": {"source": load_text("gemv", "kernel_triton_msl.py", dtype, {"M": str(M), "K": str(K)}), "note": "one program per row, 1-D accumulator"}},
    )
