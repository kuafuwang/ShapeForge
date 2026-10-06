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
        meta={"ieee_inputs": ("A",), "triton_baseline_variant": {"source": load_text("gemv", "kernel_triton_msl.py", dtype, {"M": str(M), "K": str(K)}), "note": "one program per row, 1-D accumulator"}},
    )
