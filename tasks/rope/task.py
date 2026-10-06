import torch

from tcenv.contract import AbiArg, Contract
from tcenv.distributions import sample
from tcenv.task import BaselineImpl, Task, default_tolerances
from tasks._common import check_dtype, load_text

BASE = 10000.0
BLOCK_S = 16
SHAPES = {"paper": (2, 32, 1024, 128), "small": (1, 2, 64, 64)}
MAGS = {"normal": 1.0, "wide_range": 1.0}


def _mlx(x):
    import mlx.core as mx

    return mx.fast.rope(x, x.shape[-1], traditional=False, base=BASE, scale=1.0, offset=0)


def build(dtype: str, size: str) -> Task:
    check_dtype(dtype)
    B, H, S, D = SHAPES[size]
    contract = Contract(
        kernel_name="rope_kernel",
        grid=(S // BLOCK_S, B * H, 1),
        threads=(128, 1, 1),
        args=(AbiArg("x", 0, "input", dtype, (B, H, S, D), doc="[B, H, S, D] contiguous"), AbiArg("y", 1, "output", dtype, (B, H, S, D))),
        constants=(("B", "uint", B), ("H", "uint", H), ("S", "uint", S), ("D", "uint", D), ("BASE", "float", BASE), ("BLOCK_S", "uint", BLOCK_S)),
    )

    def gen_inputs(case, seed):
        return {"x": sample(case, (B, H, S, D), seed, dtype, mag=MAGS.get(case, 1.0), large=4.0e4)}

    def reference(inp):
        x = inp["x"].double()
        half = D // 2
        inv_freq = BASE ** (-torch.arange(half, dtype=torch.float64) / half)
        ang = torch.arange(S, dtype=torch.float64)[:, None] * inv_freq[None, :]  # [S, half]
        c, s = torch.cos(ang), torch.sin(ang)
        x1, x2 = x[..., :half], x[..., half:]
        return {"y": torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)}

    rtol, atol_rel = default_tolerances(dtype)
    n = B * H * S * D
    return Task(
        name="rope", dtype=dtype, size=size, contract=contract,
        triton_source=load_text("rope", "kernel_triton.py", dtype, {"B": str(B), "H": str(H), "S": str(S), "D": str(D)}),
        description=(
            "Rotary position embedding on x[B,H,S,D], position = index along S, frequencies BASE^(-i/(D/2)), "
            "pairs (x[..., i], x[..., i + D/2]) are rotated by angle pos*freq_i. Angles/trig are computed on the fly in fp32 (no tables)."
        ),
        cases=("normal", "wide_range", "large", "tiny", "outlier", "ieee"),
        gen_inputs=gen_inputs, reference=reference,
        baselines=(BaselineImpl("mlx", "mlx", _mlx),),
        flops=8.0 * n / 2, min_bytes=float(2 * n * 2),
        rtol=rtol, atol_rel=atol_rel,
        naive_msl=load_text("rope", "naive.metal", dtype),
    )
