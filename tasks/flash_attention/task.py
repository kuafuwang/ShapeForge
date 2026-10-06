import math

import torch
import torch.nn.functional as F

from tcenv.contract import AbiArg, Contract
from tcenv.distributions import sample
from tcenv.task import BaselineImpl, Task, default_tolerances
from tasks._common import check_dtype, load_text

import re

BLOCK_M = BLOCK_N = 32
THREADS = 128
DEFAULT_VARIANT = f"{BLOCK_M}x{BLOCK_N}t{THREADS}"


def parse_variant(variant: str, seq: int, head_dim: int) -> tuple[int, int, int]:
    """``"<BLOCK_M>x<BLOCK_N>t<threads>"`` -> (BLOCK_M, BLOCK_N, threads). The naive kernel lets ``threads / BLOCK_M`` threads (a power of two, at most one SIMD group)
    share a query row, so the thread count is tied to the query tile."""
    m = re.fullmatch(r"(\d+)x(\d+)t(\d+)", variant)
    if not m:
        raise ValueError(f"bad FlashAttention contract variant {variant!r}; expected <BLOCK_M>x<BLOCK_N>t<threads>, e.g. {DEFAULT_VARIANT}")
    bm, bn, th = map(int, m.groups())
    tpr = th // bm
    if seq % bm or seq % bn or th % bm or th % 32 or th > 1024 or tpr < 1 or tpr > 32 or tpr & (tpr - 1) or head_dim % tpr:
        raise ValueError(f"tile {bm}x{bn} with {th} threads is not a valid FlashAttention contract (threads must be a multiple of 32 and of BLOCK_M, threads/BLOCK_M a power of two <= 32 dividing D={head_dim}, tiles dividing S={seq})")
    return bm, bn, th
LOG2E = 1.4426950408889634
# (B, H, S, D)
SHAPES = {"paper": (2, 8, 1024, 64), "small": (1, 2, 128, 64)}
MAGS = {"normal": 1.0, "wide_range": 0.1}


def _mlx_factory(scale):
    def f(q, k, v):
        import mlx.core as mx

        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

    return f


def _torch_factory(scale):
    def f(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, scale=scale)

    return f


def build_attention(name: str, dtype: str, size: str, *, odd_scale: bool = False, description: str | None = None, variant: str = "") -> Task:
    """``odd_scale``: the paper's misleading-cue variant, ``qk_scale = SM_SCALE + log2(e)`` (a *sum*, not a product)."""
    check_dtype(dtype)
    B, H, S, D = SHAPES[size]
    BLOCK_M, BLOCK_N, THREADS = parse_variant(variant or DEFAULT_VARIANT, S, D)
    full_name = f"{name}@{variant}" if variant else name
    sm_scale = 1.0 / math.sqrt(D)
    # natural-log scale applied to q.k in the reference: exp2(x * qk_scale) == exp(x * qk_scale * ln 2)
    eff_scale = (sm_scale + LOG2E) * math.log(2.0) if odd_scale else sm_scale
    qk_expr = "SM_SCALE + 1.4426950408889634" if odd_scale else "SM_SCALE * 1.4426950408889634"
    qk_msl = "SM_SCALE + 1.4426950408889634f" if odd_scale else "SM_SCALE * 1.4426950408889634f"
    shape = (B, H, S, D)
    contract = Contract(
        kernel_name="flash_attention_kernel",
        grid=(S // BLOCK_M, B * H, 1),
        threads=(THREADS, 1, 1),
        args=(
            AbiArg("q", 0, "input", dtype, shape, doc="[B, H, S, D]"),
            AbiArg("k", 1, "input", dtype, shape),
            AbiArg("v", 2, "input", dtype, shape),
            AbiArg("o", 3, "output", dtype, shape),
        ),
        constants=(("B", "uint", B), ("H", "uint", H), ("S", "uint", S), ("D", "uint", D), ("BLOCK_M", "uint", BLOCK_M), ("BLOCK_N", "uint", BLOCK_N), ("SM_SCALE", "float", sm_scale)),
    )

    def gen_inputs(case, seed):
        mag = MAGS.get(case, 1.0)
        mk = lambda s: sample(case, shape, seed + s, dtype, mag=mag, large=6.0, spike=5.0)
        return {"q": mk(0), "k": mk(31), "v": mk(62)}

    def reference(inp):
        o = F.scaled_dot_product_attention(inp["q"].float(), inp["k"].float(), inp["v"].float(), scale=eff_scale)
        return {"o": o.double()}

    rtol, atol_rel = default_tolerances(dtype)
    tpr = THREADS // BLOCK_M
    shuffles = "".join(f"        s += simd_shuffle_xor(s, {1 << i});\n" for i in range(tpr.bit_length() - 1))
    subs = {"B": str(B), "H": str(H), "S": str(S), "D": str(D), "SM_SCALE": repr(sm_scale), "QK_SCALE_EXPR": qk_expr, "QK_SCALE_MSL": qk_msl,
            "BM": str(BLOCK_M), "BN": str(BLOCK_N), "NW": str(contract.num_warps), "TPR": str(tpr), "SHUFFLES": shuffles, "SZ": str(H * S * D), "SH": str(S * D), "SR": str(D)}
    meta = {"ieee_inputs": ("q",)}  # a special in k or v would make every output row non-finite
    if name == "flash_attention":  # the form the compiler baseline accepts (see tcenv.triton_runner)
        meta["triton_baseline_variant"] = {"source": load_text("flash_attention", "kernel_triton_msl.py", dtype, subs), "note": "tutorial form: runtime strides and Z/H/N_CTX, scale folded into q"}
    return Task(
        name=full_name, dtype=dtype, size=size, contract=contract,
        triton_source=load_text("flash_attention", "kernel_triton.py", dtype, subs),
        description=description or "Non-causal scaled dot-product attention o = softmax(q k^T * scale) v per (batch, head), computed with the online-softmax (FlashAttention) recurrence; scores, statistics and the accumulator in fp32, output rounded once.",
        cases=("normal", "wide_range", "large", "tiny", "outlier", "ieee"),
        gen_inputs=gen_inputs, reference=reference,
        baselines=(BaselineImpl("mlx", "mlx", _mlx_factory(eff_scale)), BaselineImpl("torch_mps", "torch", _torch_factory(eff_scale))),
        flops=4.0 * B * H * S * S * D, min_bytes=float(4 * B * H * S * D * 2),
        rtol=rtol, atol_rel=atol_rel,
        naive_msl=load_text("flash_attention", "naive.metal", dtype, subs),
        meta=meta,
    )


def build(dtype: str, size: str) -> Task:
    return build_attention("flash_attention", dtype, size)


def build_variant(dtype: str, size: str, variant: str) -> Task:
    """The same operator under another launch contract (query tile, key tile, threads); used by the contract-sensitivity study."""
    return build_attention("flash_attention", dtype, size, variant=variant)
