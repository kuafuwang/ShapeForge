import torch

from tcenv.contract import AbiArg, Contract
from tcenv.distributions import sample
from tcenv.task import BaselineImpl, Task, default_tolerances
from tasks._common import check_dtype, load_text

import re

BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
THREADS = 128
DEFAULT_VARIANT = f"{BLOCK_M}x{BLOCK_N}x{BLOCK_K}t{THREADS}"


def parse_variant(variant: str, shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """``"<BM>x<BN>x<BK>t<threads>"`` -> (BM, BN, BK, threads); the tile must divide the problem and the threads must be whole SIMD groups."""
    m = re.fullmatch(r"(\d+)x(\d+)x(\d+)t(\d+)", variant)
    if not m:
        raise ValueError(f"bad GEMM contract variant {variant!r}; expected <BM>x<BN>x<BK>t<threads>, e.g. {DEFAULT_VARIANT}")
    bm, bn, bk, th = map(int, m.groups())
    M, N, K = shape
    if M % bm or N % bn or K % bk:
        raise ValueError(f"tile {bm}x{bn}x{bk} does not divide {M}x{N}x{K}")
    if th % 32 or th > 1024 or (bm * bn) % (th * 8) or bn % 8:
        raise ValueError(f"threads {th} must be a multiple of 32, at most 1024, and BM*BN must be a multiple of 8*threads (the naive kernel gives each thread a TMx8 tile)")
    return bm, bn, bk, th
SHAPES = {"paper": (2048, 2048, 2048), "small": (256, 256, 128)}
MAGS = {"normal": 1.0, "wide_range": 0.1}


def _mlx(a, b):
    return a @ b


def _torch(a, b):
    return torch.matmul(a, b)


def build_gemm(name: str, dtype: str, size: str, *, epilogue_ref, epilogue_msl: str, epilogue_def: str, baselines, description: str, extra_flops: float, variant: str = "") -> Task:
    check_dtype(dtype)
    M, N, K = SHAPES[size]
    BLOCK_M, BLOCK_N, BLOCK_K, THREADS = parse_variant(variant or DEFAULT_VARIANT, (M, N, K))
    full_name = f"{name}@{variant}" if variant else name
    contract = Contract(
        kernel_name=f"{name}_kernel",
        grid=(N // BLOCK_N, M // BLOCK_M, 1),
        threads=(THREADS, 1, 1),
        args=(
            AbiArg("A", 0, "input", dtype, (M, K), doc="row-major [M, K]"),
            AbiArg("B", 1, "input", dtype, (K, N), doc="row-major [K, N]"),
            AbiArg("C", 2, "output", dtype, (M, N), doc="row-major [M, N]"),
        ),
        constants=(
            ("M", "uint", M), ("N", "uint", N), ("K", "uint", K),
            ("BLOCK_M", "uint", BLOCK_M), ("BLOCK_N", "uint", BLOCK_N), ("BLOCK_K", "uint", BLOCK_K),
        ),
    )

    def gen_inputs(case, seed):
        mag = MAGS.get(case, 1.0)
        return {
            "A": sample(case, (M, K), seed, dtype, mag=mag, large=20.0, spike=5.0),
            "B": sample(case, (K, N), seed + 31, dtype, mag=mag, large=20.0, spike=5.0),
        }

    def reference(inp):
        c = inp["A"].float() @ inp["B"].float()  # fp32 accumulate on CPU (exact products of rounded inputs)
        return {"C": epilogue_ref(c.double())}

    rtol, atol_rel = default_tolerances(dtype)
    subs = {"M": str(M), "N": str(N), "K": str(K), "BM": str(BLOCK_M), "BN": str(BLOCK_N), "BK": str(BLOCK_K), "NW": str(THREADS // 32)}
    naive = load_text("gemm", "naive.metal", dtype, {"KERNEL": f"{name}_kernel", "EPILOGUE": epilogue_msl, "EPILOGUE_DEF": epilogue_def,
                                                    "THREADS": str(THREADS), "TM": str(BLOCK_M * BLOCK_N // (THREADS * 8)), "TPR": str(BLOCK_N // 8)})
    meta = {}
    if name == "gemm":  # the only variant triton-msl accepts for this operator (see tcenv.triton_runner)
        meta["triton_baseline_variant"] = {"source": load_text("gemm", "kernel_triton_msl.py", dtype, subs), "note": "program-id roles swapped, M/N/K as runtime scalars"}
    elif name == "gemm_silu":  # the backend refuses a fused epilogue; it asks for a separate epilogue kernel
        meta["triton_baseline_variant"] = {"source": load_text("gemm_silu", "kernel_triton_msl.py", dtype, subs), "note": "GEMM storing the fp32 accumulator + a separate SiLU kernel (the backend refuses the fused epilogue)"}
    return Task(
        name=full_name, dtype=dtype, size=size, contract=contract,
        triton_source=load_text(name, "kernel_triton.py", dtype, subs),
        description=description,
        cases=("normal", "wide_range", "large", "tiny", "outlier", "ieee"),
        gen_inputs=gen_inputs, reference=reference, baselines=baselines,
        flops=2.0 * M * N * K + extra_flops * M * N, min_bytes=float((M * K + K * N + M * N) * 2),
        rtol=rtol, atol_rel=atol_rel, naive_msl=naive, meta=meta,
    )


def build(dtype: str, size: str) -> Task:
    return build_gemm(
        "gemm", dtype, size,
        epilogue_ref=lambda c: c,
        epilogue_msl="", epilogue_def="",
        baselines=(BaselineImpl("mlx", "mlx", _mlx), BaselineImpl("torch_mps", "torch", _torch)),
        description="C = A @ B with row-major A[M,K], B[K,N], C[M,N]; fp32 accumulation, result rounded once.",
        extra_flops=0.0,
    )


def build_variant(dtype: str, size: str, variant: str) -> Task:
    """The same operator under another launch contract (tile and threads); used by the contract-sensitivity study."""
    return build_gemm(
        "gemm", dtype, size,
        epilogue_ref=lambda c: c,
        epilogue_msl="", epilogue_def="",
        baselines=(BaselineImpl("mlx", "mlx", _mlx), BaselineImpl("torch_mps", "torch", _torch)),
        description="C = A @ B with row-major A[M,K], B[K,N], C[M,N]; fp32 accumulation, result rounded once.",
        extra_flops=0.0, variant=variant,
    )
