import torch

from tcenv.contract import AbiArg, Contract
from tcenv.distributions import sample
from tcenv.task import BaselineImpl, Task, default_tolerances
from tasks._common import check_dtype, load_text

SHAPES = {"paper": (4096, 4096), "small": (64, 1024), "plan": (256, 4096)}  # "plan": 2 MB, a few microseconds: dispatch-bound, see README
MAGS = {"normal": 1.0, "wide_range": 0.1}


def _mlx(x):
    import mlx.core as mx

    return mx.sum(x, axis=-1)


def _torch(x):
    return torch.sum(x, dim=-1)


def build(dtype: str, size: str) -> Task:
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
