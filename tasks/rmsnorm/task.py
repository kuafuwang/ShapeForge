import torch
import torch.nn.functional as F

from tcenv.contract import AbiArg, Contract
from tcenv.distributions import sample
from tcenv.task import BaselineImpl, Task, default_tolerances
from tasks._common import check_dtype, load_text

EPS = 1e-6
SHAPES = {"paper": (4096, 4096), "small": (64, 512)}
MAGS = {"normal": 1.0, "wide_range": 1.0, "outlier": 1.0}


def _mlx(x, w):
    import mlx.core as mx

    return mx.fast.rms_norm(x, w, EPS)


def _torch(x, w):
    return F.rms_norm(x, (x.shape[-1],), w, EPS)


def build(dtype: str, size: str) -> Task:
    check_dtype(dtype)
    M, H = SHAPES[size]
    contract = Contract(
        kernel_name="rmsnorm_kernel",
        grid=(M, 1, 1),
        threads=(256, 1, 1),
        args=(
            AbiArg("x", 0, "input", dtype, (M, H), doc="activations, row-major [M, H]"),
            AbiArg("w", 1, "input", dtype, (H,), doc="per-channel weight"),
            AbiArg("y", 2, "output", dtype, (M, H)),
        ),
        constants=(("M", "uint", M), ("H", "uint", H), ("EPS", "float", EPS)),
    )

    def gen_inputs(case, seed):
        x = sample(case, (M, H), seed, dtype, mag=MAGS.get(case, 1.0), large=300.0)
        w = 1.0 + 0.1 * sample("normal", (H,), seed + 99, dtype)
        return {"x": x, "w": w}

    def reference(inp):
        x, w = inp["x"].double(), inp["w"].double()
        return {"y": x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * w}

    rtol, atol_rel = default_tolerances(dtype)
    eb = 2 if dtype != "fp32" else 4
    return Task(
        name="rmsnorm", dtype=dtype, size=size, contract=contract,
        triton_source=load_text("rmsnorm", "kernel_triton.py", dtype, {"M": str(M), "H": str(H)}),
        description="Row-wise RMSNorm: y[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * w; the sum of squares is accumulated in fp32 and the result rounded once.",
        cases=("normal", "wide_range", "large", "tiny", "outlier", "ieee"),
        gen_inputs=gen_inputs, reference=reference,
        baselines=(BaselineImpl("mlx", "mlx", _mlx), BaselineImpl("torch_mps", "torch", _torch)),
        flops=4.0 * M * H, min_bytes=float((2 * M * H + H) * eb),
        rtol=rtol, atol_rel=atol_rel,
        naive_msl=load_text("rmsnorm", "naive.metal", dtype),
    )
