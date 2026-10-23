# 非因果注意力。默认 description：每个 (batch, head) 上 o = softmax(q k^T * scale) v，用在线 softmax 递推；分数、统计和累加器在 fp32，输出舍入一次。
# 标准答案是 reference：在 float 上调用 F.scaled_dot_product_attention，scale 用 eff_scale。odd_scale 时 eff_scale = (sm_scale + LOG2E) * ln(2)，否则 eff_scale = 1/sqrt(D)。
# 基线是 baselines：mx.fast.scaled_dot_product_attention 和 F.scaled_dot_product_attention，传入的 scale 与 eff_scale 相同。
# 模型不能改 contract：核名 flash_attention_kernel，grid=(S/BLOCK_M, B*H, 1)，threads，缓冲 q/k/v/o，常量 B/H/S/D/BLOCK_M/BLOCK_N/SM_SCALE。其中 SM_SCALE 始终是 1/sqrt(D)。
# 题面来自 kernel_triton.py。odd_scale 只替换题面里的 QK_SCALE_EXPR。build_variant 只改 query tile、key tile 和 threads。
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


# 解析 "<BLOCK_M>x<BLOCK_N>t<threads>"。threads/BLOCK_M 必须是不超过 32 的 2 的幂，并且整除 head 维 D。
def parse_variant(variant: str, seq: int, head_dim: int) -> tuple[int, int, int]:
    """@brief 把 ``"<BLOCK_M>x<BLOCK_N>t<threads>"`` 拆成 tile 和线程数。

    ``"<BLOCK_M>x<BLOCK_N>t<threads>"`` -> (BLOCK_M, BLOCK_N, threads). The naive kernel lets ``threads / BLOCK_M`` threads (a power of two, at most one SIMD group)
    share a query row, so the thread count is tied to the query tile.

    @param variant 启动契约变体文本。
    @param seq 序列长度 S。
    @param head_dim head 维 D。
    @return ``(BLOCK_M, BLOCK_N, threads)``。
    @raises ValueError 文本格式不对，或 tile / 线程数与 S、D 不匹配时抛出。
    """
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
    """@brief 造一个固定 scale 的 mlx 注意力基线。

    @param scale 传给 ``mx.fast.scaled_dot_product_attention`` 的标度。
    @return 签名如 ``f(q, k, v)`` 的基线函数。
    """
    def f(q, k, v):
        import mlx.core as mx

        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

    return f


def _torch_factory(scale):
    """@brief 造一个固定 scale 的 torch 注意力基线。

    @param scale 传给 ``F.scaled_dot_product_attention`` 的标度。
    @return 签名如 ``f(q, k, v)`` 的基线函数。
    """
    def f(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, scale=scale)

    return f


# odd_scale 改的是题面里的 QK_SCALE_EXPR 和 reference 用的 eff_scale。契约常量 SM_SCALE 写入的是 1/sqrt(D)。
def build_attention(name: str, dtype: str, size: str, *, odd_scale: bool = False, description: str | None = None, variant: str = "") -> Task:
    """@brief 装配 FlashAttention 题，含误导性标度的变体。

    ``odd_scale``: the paper's misleading-cue variant, ``qk_scale = SM_SCALE + log2(e)`` (a *sum*, not a product).

    @param name 题目名，用来拼核名和题面目录。
    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @param odd_scale 是否把题面里的 qk 标度写成和而不是积。
    @param description 覆盖默认题面描述；为 None 时用默认文字。
    @param variant 启动契约变体，空串表示默认。
    @return 装配好的 ``Task``。
    """
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
    """@brief 装配非因果 FlashAttention 题。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    return build_attention("flash_attention", dtype, size)


# 同一注意力、同一 eff_scale，只换 query tile、key tile 和线程数。
def build_variant(dtype: str, size: str, variant: str) -> Task:
    """@brief 同一注意力、同一 eff_scale，只换 query tile、key tile 和线程数。

    The same operator under another launch contract (query tile, key tile, threads); used by the contract-sensitivity study.

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @param variant 启动契约变体文本，交给 ``parse_variant``。
    @return 装配好的 ``Task``，名字带 ``@<variant>`` 后缀。
    """
    return build_attention("flash_attention", dtype, size, variant=variant)
