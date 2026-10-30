"""Registered Metal lowering: RMSNorm, row softmax/sum and the elementwise family.

The profile is *registered* here — this module is imported on demand by
``tcenv.lowering.ensure_lowerings("metal")`` — and it is the only source of the
contract defaults for a Metal-backed task: kernel name, ``grid``/``threads``, the
constant target types, the start form and ``Contract.gpu``.  It carries no build
probe result, no device fact and no candidate source.

``apple-m4-max`` is the only target this profile covers.  Any other ``target_id`` is
refused by ``tcenv.lowering.require_target_profile`` instead of silently reusing this
profile (plan §4.2, N-9); adding a target is a separate change that registers its own
profile.
"""

from __future__ import annotations

import math
import re

from ... import lowering as _lowering
from ...contract import Contract
from ...lowering import (
    ConstantSpec,
    LoweringError,
    LoweringProfile,
    ScalarSpec,
    build_contract,
    register_lowerer,
    register_profile,
    register_variant_factory,
)
from ...opspec.model import SemanticInstance

BACKEND_ID = "metal"
#: The only target the R1 registry covers.
TARGET_ID = "apple-m4-max"
#: Registered lowerer id/version; both enter ``lowering_profile_digest``.
LOWERER_ID = "metal.rmsnorm"
LOWERER_VERSION = "1"
#: Registered lowerer for the elementwise family (`y[i] = f(x[i])`, direct index mapping).
ELEMENTWISE_LOWERER_ID = "metal.elementwise"
ELEMENTWISE_LOWERER_VERSION = "1"
#: Registered profile name referenced by ``tasks/rmsnorm/track.json``.
RMSNORM_PROFILE_NAME = "metal-rmsnorm"


def _lower_buffers_and_constants(
    instance: SemanticInstance, profile: LoweringProfile
) -> tuple[Contract, dict[str, str]]:
    """@brief RMSNorm 的 Metal lower：通用 buffer/常量派生（无 family 专属规则）。

    @param instance 语义实例。
    @param profile 登记的 profile。
    @return ``(contract, backend_params)``。
    """
    contract, backend_params = build_contract(instance, profile)
    return contract, dict(backend_params)


register_lowerer(LOWERER_ID, LOWERER_VERSION, _lower_buffers_and_constants)
register_lowerer(ELEMENTWISE_LOWERER_ID, ELEMENTWISE_LOWERER_VERSION, _lower_buffers_and_constants)


def _elementwise_profile(name: str, kernel_name: str, num_warps: int, block_size: int) -> LoweringProfile:
    """@brief 登记一份逐元素 profile：``grid=(N//BLOCK_SIZE,1,1)``，常量 N/BLOCK_SIZE。

    ``BLOCK_SIZE``/``NUM_WARPS`` 是启动事实，由 profile 派生进 ``backend_params``；题面模板
    只能从这里取值（F-22），不能自己写字面量。

    @param name 登记名。
    @param kernel_name 契约核名。
    @param num_warps ``num_warps`` 启动参数。
    @param block_size 每个 threadgroup 处理的元素数。
    @return 已登记的 profile。
    """
    return register_profile(
        LoweringProfile(
            schema_version=_lowering.LOWERING_PROFILE_SCHEMA_VERSION,
            name=name,
            backend_id=BACKEND_ID,
            target_id=TARGET_ID,
            display_name="Apple M4 Max (Apple9 GPU family)",
            lowerer_id=ELEMENTWISE_LOWERER_ID,
            kernel_name=kernel_name,
            grid=({"symbol": "N", "factor": block_size}, 1, 1),
            threads=(num_warps * 32, 1, 1),
            constants=(
                ConstantSpec(name="N", msl_type="uint", source="symbol:N"),
                ConstantSpec(name="BLOCK_SIZE", msl_type="uint", source=f"literal:{block_size}"),
            ),
            variant_params={
                "BLOCK_SIZE": f"literal:{block_size}",
                "NUM_WARPS": f"literal:{num_warps}",
            },
        )
    )

#: The R1 RMSNorm profile. ``grid=(M,1,1)``, ``threads=(256,1,1)``, constants M/H (uint)
#: and EPS (float), and ``gpu`` equal to the historical default string, so the derived
#: contract stays byte-identical to the frozen R0 task.
RMSNORM_PROFILE = register_profile(
    LoweringProfile(
        schema_version=_lowering.LOWERING_PROFILE_SCHEMA_VERSION,
        name=RMSNORM_PROFILE_NAME,
        backend_id=BACKEND_ID,
        target_id=TARGET_ID,
        display_name="Apple M4 Max (Apple9 GPU family)",
        lowerer_id=LOWERER_ID,
        kernel_name="rmsnorm_kernel",
        grid=({"symbol": "M"}, 1, 1),
        threads=(256, 1, 1),
        constants=(
            ConstantSpec(name="M", msl_type="uint", source="symbol:M"),
            ConstantSpec(name="H", msl_type="uint", source="symbol:H"),
            ConstantSpec(name="EPS", msl_type="float", source="attr:EPS"),
        ),
    )
)


#: Row-wise softmax: one threadgroup per row, 256 threads, constants M/N (uint).
SOFTMAX_PROFILE = register_profile(
    LoweringProfile(
        schema_version=_lowering.LOWERING_PROFILE_SCHEMA_VERSION,
        name="metal-softmax",
        backend_id=BACKEND_ID,
        target_id=TARGET_ID,
        display_name="Apple M4 Max (Apple9 GPU family)",
        lowerer_id=LOWERER_ID,
        kernel_name="softmax_kernel",
        grid=({"symbol": "M"}, 1, 1),
        threads=(256, 1, 1),
        constants=(
            ConstantSpec(name="M", msl_type="uint", source="symbol:M"),
            ConstantSpec(name="N", msl_type="uint", source="symbol:N"),
        ),
    )
)

#: Row-wise sum: one threadgroup per row, 128 threads, constants M/N (uint).
REDUCTION_SUM_PROFILE = register_profile(
    LoweringProfile(
        schema_version=_lowering.LOWERING_PROFILE_SCHEMA_VERSION,
        name="metal-reduction-sum",
        backend_id=BACKEND_ID,
        target_id=TARGET_ID,
        display_name="Apple M4 Max (Apple9 GPU family)",
        lowerer_id=LOWERER_ID,
        kernel_name="reduction_sum_kernel",
        grid=({"symbol": "M"}, 1, 1),
        threads=(128, 1, 1),
        constants=(
            ConstantSpec(name="M", msl_type="uint", source="symbol:M"),
            ConstantSpec(name="N", msl_type="uint", source="symbol:N"),
        ),
    )
)


#: Rotary position embedding: one threadgroup per (BLOCK_S positions, batch*head), 128 threads.
#: ``BASE`` is a semantic attr (it changes the values), ``BLOCK_S`` only changes the launch.
ROPE_PROFILE = register_profile(
    LoweringProfile(
        schema_version=_lowering.LOWERING_PROFILE_SCHEMA_VERSION,
        name="metal-rope",
        backend_id=BACKEND_ID,
        target_id=TARGET_ID,
        display_name="Apple M4 Max (Apple9 GPU family)",
        lowerer_id=LOWERER_ID,
        kernel_name="rope_kernel",
        grid=({"symbol": "S", "factor": 16}, {"product": ["B", "H"]}, 1),
        threads=(128, 1, 1),
        constants=(
            ConstantSpec(name="B", msl_type="uint", source="symbol:B"),
            ConstantSpec(name="H", msl_type="uint", source="symbol:H"),
            ConstantSpec(name="S", msl_type="uint", source="symbol:S"),
            ConstantSpec(name="D", msl_type="uint", source="symbol:D"),
            ConstantSpec(name="BASE", msl_type="float", source="attr:BASE"),
            ConstantSpec(name="BLOCK_S", msl_type="uint", source="literal:16"),
        ),
    )
)

#: Matrix-vector: four rows per threadgroup (``BLOCK_M=4``), 128 threads, constants M/K/BLOCK_M.
GEMV_PROFILE = register_profile(
    LoweringProfile(
        schema_version=_lowering.LOWERING_PROFILE_SCHEMA_VERSION,
        name="metal-gemv",
        backend_id=BACKEND_ID,
        target_id=TARGET_ID,
        display_name="Apple M4 Max (Apple9 GPU family)",
        lowerer_id=LOWERER_ID,
        kernel_name="gemv_kernel",
        grid=({"symbol": "M", "factor": 4}, 1, 1),
        threads=(128, 1, 1),
        constants=(
            ConstantSpec(name="M", msl_type="uint", source="symbol:M"),
            ConstantSpec(name="K", msl_type="uint", source="symbol:K"),
            ConstantSpec(name="BLOCK_M", msl_type="uint", source="literal:4"),
        ),
    )
)

#: One Lion step: 1024 elements per threadgroup, 128 threads, LR/WD/BETA1/BETA2 are semantic attrs.
LION_PROFILE = register_profile(
    LoweringProfile(
        schema_version=_lowering.LOWERING_PROFILE_SCHEMA_VERSION,
        name="metal-lion",
        backend_id=BACKEND_ID,
        target_id=TARGET_ID,
        display_name="Apple M4 Max (Apple9 GPU family)",
        lowerer_id=LOWERER_ID,
        kernel_name="lion_kernel",
        grid=({"symbol": "N", "factor": 1024}, 1, 1),
        threads=(128, 1, 1),
        constants=(
            ConstantSpec(name="N", msl_type="uint", source="symbol:N"),
            ConstantSpec(name="BLOCK_SIZE", msl_type="uint", source="literal:1024"),
            ConstantSpec(name="LR", msl_type="float", source="attr:LR"),
            ConstantSpec(name="WD", msl_type="float", source="attr:WD"),
            ConstantSpec(name="BETA1", msl_type="float", source="attr:BETA1"),
            ConstantSpec(name="BETA2", msl_type="float", source="attr:BETA2"),
        ),
    )
)

#: The four pointwise tasks share one launch geometry: 1024 elements per threadgroup, 4 warps.
RELU_PROFILE = _elementwise_profile("metal-relu", "relu_kernel", 4, 1024)
SIGMOID_PROFILE = _elementwise_profile("metal-sigmoid", "sigmoid_kernel", 4, 1024)
SILU_PROFILE = _elementwise_profile("metal-silu", "silu_kernel", 4, 1024)
SWIGLU_PROFILE = _elementwise_profile("metal-swiglu", "swiglu_kernel", 4, 1024)

# ------------------------------------------------------------- generated integer programs
#: ``sem_int_<seed>``: one threadgroup handles ``BLOCK`` int32 elements, each thread
#: ``PER_THREAD`` of them; ``K`` is a runtime scalar whose value comes from the OpSpec attr.
SEM_INT_LOWERER_ID = "metal.int-program"
SEM_INT_LOWERER_VERSION = "1"
SEM_INT_BLOCK = 256

register_lowerer(SEM_INT_LOWERER_ID, SEM_INT_LOWERER_VERSION, _lower_buffers_and_constants)

SEM_INT_PROFILE = register_profile(
    LoweringProfile(
        schema_version=_lowering.LOWERING_PROFILE_SCHEMA_VERSION,
        name="metal-sem-int",
        backend_id=BACKEND_ID,
        target_id=TARGET_ID,
        display_name="Apple M4 Max (Apple9 GPU family)",
        lowerer_id=SEM_INT_LOWERER_ID,
        kernel_name="sem_int_kernel",
        grid=({"symbol": "N", "factor": SEM_INT_BLOCK}, 1, 1),
        threads=(64, 1, 1),
        constants=(
            ConstantSpec(name="N", msl_type="uint", source="symbol:N"),
            ConstantSpec(name="BLOCK", msl_type="uint", source=f"literal:{SEM_INT_BLOCK}"),
        ),
        scalars=(
            ScalarSpec(name="K", msl_type="int", source="attr:SCALAR_K", doc="runtime scalar argument", index=2),
        ),
        variant_params={
            "BLOCK": f"literal:{SEM_INT_BLOCK}",
            "NUM_WARPS": "literal:2",
        },
    )
)


# ------------------------------------------------------------------------ BitDelta GEMM
#: BitDelta: a packed-sign GEMM whose ``scale`` is a runtime scalar *between* two buffers
#: (``x``, ``bits``, scale, ``y``).  The scalar's ABI slot is pinned by the registered profile,
#: so the buffer slots stay the declaration order and the contract is byte-identical to the
#: frozen hand-written one.
BITDELTA_LOWERER_ID = "metal.bitdelta-gemm"
BITDELTA_LOWERER_VERSION = "1"
BITDELTA_BLOCK_M = 64
BITDELTA_BLOCK_N = 64
BITDELTA_BLOCK_K = 32

register_lowerer(BITDELTA_LOWERER_ID, BITDELTA_LOWERER_VERSION, _lower_buffers_and_constants)

BITDELTA_PROFILE = register_profile(
    LoweringProfile(
        schema_version=_lowering.LOWERING_PROFILE_SCHEMA_VERSION,
        name="metal-bitdelta-gemm",
        backend_id=BACKEND_ID,
        target_id=TARGET_ID,
        display_name="Apple M4 Max (Apple9 GPU family)",
        lowerer_id=BITDELTA_LOWERER_ID,
        kernel_name="bitdelta_gemm_kernel",
        grid=(
            {"symbol": "N", "factor": BITDELTA_BLOCK_N},
            {"symbol": "M", "factor": BITDELTA_BLOCK_M},
            1,
        ),
        threads=(128, 1, 1),
        constants=(
            ConstantSpec(name="M", msl_type="uint", source="symbol:M"),
            ConstantSpec(name="N", msl_type="uint", source="symbol:N"),
            ConstantSpec(name="K", msl_type="uint", source="symbol:K"),
            ConstantSpec(name="BLOCK_M", msl_type="uint", source=f"literal:{BITDELTA_BLOCK_M}"),
            ConstantSpec(name="BLOCK_N", msl_type="uint", source=f"literal:{BITDELTA_BLOCK_N}"),
            ConstantSpec(name="BLOCK_K", msl_type="uint", source=f"literal:{BITDELTA_BLOCK_K}"),
        ),
        scalars=(
            ScalarSpec(
                name="scale",
                msl_type="float",
                source="attr:SCALE",
                doc="per-matrix scale",
                index=2,
            ),
        ),
        variant_params={
            "BM": f"literal:{BITDELTA_BLOCK_M}",
            "BN": f"literal:{BITDELTA_BLOCK_N}",
            "BK": f"literal:{BITDELTA_BLOCK_K}",
            "NW": "literal:4",
        },
    )
)


# --------------------------------------------------------------------------- Mamba-2 chunk scan
#: Mamba-2 SSD chunk scan: seven launch constants, eight buffers (one of them the 5-D ``states``).
MAMBA2_LOWERER_ID = "metal.mamba2-chunk-scan"
MAMBA2_LOWERER_VERSION = "1"

register_lowerer(MAMBA2_LOWERER_ID, MAMBA2_LOWERER_VERSION, _lower_buffers_and_constants)

MAMBA2_PROFILE = register_profile(
    LoweringProfile(
        schema_version=_lowering.LOWERING_PROFILE_SCHEMA_VERSION,
        name="metal-mamba2-chunk-scan",
        backend_id=BACKEND_ID,
        target_id=TARGET_ID,
        display_name="Apple M4 Max (Apple9 GPU family)",
        lowerer_id=MAMBA2_LOWERER_ID,
        kernel_name="mamba2_chunk_scan_kernel",
        grid=(
            {"symbol": "NCHUNKS"},
            {"product": ["BATCH", "NHEADS"]},
            1,
        ),
        threads=(128, 1, 1),
        constants=(
            ConstantSpec(name="BATCH", msl_type="uint", source="symbol:BATCH"),
            ConstantSpec(name="SEQLEN", msl_type="uint", source="symbol:SEQLEN"),
            ConstantSpec(name="NHEADS", msl_type="uint", source="symbol:NHEADS"),
            ConstantSpec(name="HEADDIM", msl_type="uint", source="symbol:HEADDIM"),
            ConstantSpec(name="DSTATE", msl_type="uint", source="symbol:DSTATE"),
            ConstantSpec(name="CHUNK", msl_type="uint", source="symbol:CHUNK"),
            ConstantSpec(name="NCHUNKS", msl_type="uint", source="symbol:NCHUNKS"),
        ),
        # ``CHUNK`` reaches the prompt through ``spec_params`` (it is a declared symbol), so the
        # profile must not also provide it: an ambiguous placeholder is refused by the track.
        variant_params={
            "NW": "literal:4",
        },
    )
)


# --------------------------------------------------------------------------- GEMM family
#: Tiled matmul: same operator under a registered tile/threads contract.  The variant text
#: is the historical grammar ``"<BM>x<BN>x<BK>t<threads>"``; illegal/unknown text is refused
#: by the factory and never reaches a profile.
GEMM_VARIANT_FACTORY_ID = "metal.gemm.variant"
GEMM_VARIANT_FACTORY_VERSION = "1"
GEMM_LOWERER_ID = "metal.gemm"
GEMM_LOWERER_VERSION = "1"
DEFAULT_GEMM_VARIANT = "64x64x32t128"

register_lowerer(GEMM_LOWERER_ID, GEMM_LOWERER_VERSION, _lower_buffers_and_constants)


def parse_gemm_variant(variant: str, shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """@brief 解析 GEMM 启动契约变体（登记文法的唯一实现）。

    @param variant ``"<BM>x<BN>x<BK>t<threads>"``。
    @param shape 问题的 ``(M, N, K)``。
    @return ``(BM, BN, BK, threads)``。
    @raises LoweringError 文法非法、tile 不整除问题形状或线程数不合规。
    """
    match = re.fullmatch(r"(\d+)x(\d+)x(\d+)t(\d+)", variant)
    if not match:
        raise LoweringError(
            f"bad GEMM contract variant {variant!r}; expected <BM>x<BN>x<BK>t<threads>, e.g. {DEFAULT_GEMM_VARIANT}"
        )
    bm, bn, bk, th = (int(group) for group in match.groups())
    m, n, k = shape
    if m % bm or n % bn or k % bk:
        raise LoweringError(f"tile {bm}x{bn}x{bk} does not divide {m}x{n}x{k}")
    if th % 32 or th > 1024 or (bm * bn) % (th * 8) or bn % 8:
        raise LoweringError(
            f"threads {th} must be a multiple of 32, at most 1024, and BM*BN must be a multiple of "
            f"8*threads (the naive kernel gives each thread a TMx8 tile), BN a multiple of 8"
        )
    return bm, bn, bk, th


def _gemm_tile_profile(
    *,
    bm: int,
    bn: int,
    bk: int,
    th: int,
    name: str,
    kernel_name: str,
    lowerer_id: str,
    factory_id: str | None,
) -> LoweringProfile:
    """@brief 用已解析的 tile/threads 产生一份 GEMM profile（不登记进构建器表）。

    @param bm 输出行 tile。
    @param bn 输出列 tile。
    @param bk 缩减维 tile。
    @param th 每个 threadgroup 的线程数。
    @param name profile 名。
    @param kernel_name 契约核名。
    @param lowerer_id 登记的 family lowerer。
    @param factory_id 产生它的 variant factory（默认 profile 为 ``None``）。
    @return 完全物化的 profile。
    """
    return LoweringProfile(
        schema_version=_lowering.LOWERING_PROFILE_SCHEMA_VERSION,
        name=name,
        backend_id=BACKEND_ID,
        target_id=TARGET_ID,
        display_name="Apple M4 Max (Apple9 GPU family)",
        lowerer_id=lowerer_id,
        kernel_name=kernel_name,
        grid=({"symbol": "N", "factor": bn}, {"symbol": "M", "factor": bm}, 1),
        threads=(th, 1, 1),
        constants=(
            ConstantSpec(name="M", msl_type="uint", source="symbol:M"),
            ConstantSpec(name="N", msl_type="uint", source="symbol:N"),
            ConstantSpec(name="K", msl_type="uint", source="symbol:K"),
            ConstantSpec(name="BLOCK_M", msl_type="uint", source=f"literal:{bm}"),
            ConstantSpec(name="BLOCK_N", msl_type="uint", source=f"literal:{bn}"),
            ConstantSpec(name="BLOCK_K", msl_type="uint", source=f"literal:{bk}"),
        ),
        variant_params={
            "BM": f"literal:{bm}",
            "BN": f"literal:{bn}",
            "BK": f"literal:{bk}",
            "NW": f"literal:{th // 32}",
            "TM": f"literal:{bm * bn // (th * 8)}",
            "TPR": f"literal:{bn // 8}",
            "THREADS": f"literal:{th}",
        },
        variant_factory_id=factory_id,
    )


def _gemm_variant_profile(instance: SemanticInstance, variant: str) -> LoweringProfile:
    """@brief GEMM variant factory：只换 tile/threads，语义实例与 OpSpec 不变。

    @param instance 语义实例。
    @param variant variant 文本。
    @return 具体 profile。
    """
    bm, bn, bk, th = parse_gemm_variant(variant, tuple(instance.symbols[k] for k in ("M", "N", "K")))
    return _gemm_tile_profile(
        bm=bm,
        bn=bn,
        bk=bk,
        th=th,
        name=f"metal-gemm@{variant}",
        kernel_name="gemm_kernel",
        lowerer_id=GEMM_LOWERER_ID,
        factory_id=GEMM_VARIANT_FACTORY_ID,
    )


register_variant_factory(GEMM_VARIANT_FACTORY_ID, GEMM_VARIANT_FACTORY_VERSION, _gemm_variant_profile)

#: Default GEMM profile: the factory's default tile, spelled out and checked by the parser.
GEMM_PROFILE = register_profile(
    _gemm_tile_profile(
        bm=64,
        bn=64,
        bk=32,
        th=128,
        name="metal-gemm",
        kernel_name="gemm_kernel",
        lowerer_id=GEMM_LOWERER_ID,
        factory_id=None,
    )
)
#: Default GEMM+SiLU profile (same tile, SiLU epilogue kernel name).
GEMM_SILU_PROFILE = register_profile(
    _gemm_tile_profile(
        bm=64,
        bn=64,
        bk=32,
        th=128,
        name="metal-gemm-silu",
        kernel_name="gemm_silu_kernel",
        lowerer_id=GEMM_LOWERER_ID,
        factory_id=None,
    )
)


# ------------------------------------------------------------------------ FlashAttention
FA_VARIANT_FACTORY_ID = "metal.flash-attention.variant"
FA_VARIANT_FACTORY_VERSION = "1"
FA_LOWERER_ID = "metal.flash-attention"
FA_LOWERER_VERSION = "1"
DEFAULT_FA_VARIANT = "32x32t128"

register_lowerer(FA_LOWERER_ID, FA_LOWERER_VERSION, _lower_buffers_and_constants)


def parse_attention_variant(variant: str, seq: int, head_dim: int) -> tuple[int, int, int]:
    """@brief 解析 FlashAttention 启动契约变体（登记文法）。

    @param variant ``"<BLOCK_M>x<BLOCK_N>t<threads>"``。
    @param seq 序列长度 ``S``。
    @param head_dim head 维 ``D``。
    @return ``(BLOCK_M, BLOCK_N, threads)``。
    @raises LoweringError 文法非法或 tile/线程数与 S、D 不匹配。
    """
    match = re.fullmatch(r"(\d+)x(\d+)t(\d+)", variant)
    if not match:
        raise LoweringError(
            f"bad FlashAttention contract variant {variant!r}; expected <BLOCK_M>x<BLOCK_N>t<threads>, e.g. {DEFAULT_FA_VARIANT}"
        )
    bm, bn, th = (int(group) for group in match.groups())
    tpr = th // bm
    if (
        seq % bm
        or seq % bn
        or th % bm
        or th % 32
        or th > 1024
        or tpr < 1
        or tpr > 32
        or tpr & (tpr - 1)
        or head_dim % tpr
    ):
        raise LoweringError(
            f"tile {bm}x{bn} with {th} threads is not a valid FlashAttention contract (threads must be a multiple of "
            f"32 and of BLOCK_M, threads/BLOCK_M a power of two <= 32 dividing D={head_dim}, tiles dividing S={seq})"
        )
    return bm, bn, th


def _attention_tile_profile(*, bm: int, bn: int, th: int, name: str, factory_id: str | None) -> LoweringProfile:
    """@brief 用已解析的 tile/threads 产生一份 FlashAttention profile。

    @param bm query tile。
    @param bn key tile。
    @param th 每个 threadgroup 的线程数。
    @param name profile 名。
    @param factory_id 产生它的 factory（默认 profile 为 ``None``）。
    @return 完全物化的 profile。
    """
    tpr = th // bm
    shuffles = "".join(f"        s += simd_shuffle_xor(s, {1 << index});\n" for index in range(tpr.bit_length() - 1))
    return LoweringProfile(
        schema_version=_lowering.LOWERING_PROFILE_SCHEMA_VERSION,
        name=name,
        backend_id=BACKEND_ID,
        target_id=TARGET_ID,
        display_name="Apple M4 Max (Apple9 GPU family)",
        lowerer_id=FA_LOWERER_ID,
        kernel_name="flash_attention_kernel",
        grid=({"symbol": "S", "factor": bm}, {"product": ["B", "H"]}, 1),
        threads=(th, 1, 1),
        constants=(
            ConstantSpec(name="B", msl_type="uint", source="symbol:B"),
            ConstantSpec(name="H", msl_type="uint", source="symbol:H"),
            ConstantSpec(name="S", msl_type="uint", source="symbol:S"),
            ConstantSpec(name="D", msl_type="uint", source="symbol:D"),
            ConstantSpec(name="BLOCK_M", msl_type="uint", source=f"literal:{bm}"),
            ConstantSpec(name="BLOCK_N", msl_type="uint", source=f"literal:{bn}"),
            ConstantSpec(name="SM_SCALE", msl_type="float", source="attr:SM_SCALE"),
        ),
        variant_params={
            "BM": f"literal:{bm}",
            "BN": f"literal:{bn}",
            "NW": f"literal:{th // 32}",
            "TPR": f"literal:{tpr}",
            "SHUFFLES": shuffles,
            "SZ": {"product": ["H", "S", "D"]},
            "SH": {"product": ["S", "D"]},
            "SR": {"symbol": "D"},
        },
        variant_factory_id=factory_id,
    )


def _attention_variant_profile(instance: SemanticInstance, variant: str) -> LoweringProfile:
    """@brief FlashAttention variant factory：只换 query/key tile 与 threads。

    @param instance 语义实例。
    @param variant variant 文本。
    @return 具体 profile。
    """
    bm, bn, th = parse_attention_variant(variant, int(instance.symbols["S"]), int(instance.symbols["D"]))
    return _attention_tile_profile(
        bm=bm, bn=bn, th=th, name=f"metal-flash-attention@{variant}", factory_id=FA_VARIANT_FACTORY_ID
    )


register_variant_factory(FA_VARIANT_FACTORY_ID, FA_VARIANT_FACTORY_VERSION, _attention_variant_profile)

#: Default FlashAttention profile: the factory's default tile, spelled out.
FLASH_ATTENTION_PROFILE = register_profile(
    _attention_tile_profile(bm=32, bn=32, th=128, name="metal-flash-attention", factory_id=None)
)


# ------------------------------------------------------------------------------- conv2d
CONV2D_LOWERER_ID = "metal.conv2d"
CONV2D_LOWERER_VERSION = "1"
register_lowerer(CONV2D_LOWERER_ID, CONV2D_LOWERER_VERSION, _lower_buffers_and_constants)

CONV2D_PROFILE = register_profile(
    LoweringProfile(
        schema_version=_lowering.LOWERING_PROFILE_SCHEMA_VERSION,
        name="metal-conv2d",
        backend_id=BACKEND_ID,
        target_id=TARGET_ID,
        display_name="Apple M4 Max (Apple9 GPU family)",
        lowerer_id=CONV2D_LOWERER_ID,
        kernel_name="conv2d_kernel",
        grid=(
            {"symbol": "OC", "factor": 64},
            {"ceil_div": {"numerator": {"product": ["N", "OH", "OW"]}, "denominator": 64}},
            1,
        ),
        threads=(128, 1, 1),
        constants=(
            ConstantSpec(name="N", msl_type="uint", source="symbol:N"),
            ConstantSpec(name="H", msl_type="uint", source="symbol:H"),
            ConstantSpec(name="W", msl_type="uint", source="symbol:W"),
            ConstantSpec(name="C", msl_type="uint", source="symbol:C"),
            ConstantSpec(name="OC", msl_type="uint", source="symbol:OC"),
            ConstantSpec(name="KH", msl_type="uint", source="symbol:KH"),
            ConstantSpec(name="KW", msl_type="uint", source="symbol:KW"),
            ConstantSpec(name="OH", msl_type="uint", source="symbol:OH"),
            ConstantSpec(name="OW", msl_type="uint", source="symbol:OW"),
            ConstantSpec(name="M_TOTAL", msl_type="uint", source="product:N,OH,OW"),
            ConstantSpec(name="K_TOTAL", msl_type="uint", source="product:KH,KW,C"),
            ConstantSpec(name="BLOCK_M", msl_type="uint", source="literal:64"),
            ConstantSpec(name="BLOCK_N", msl_type="uint", source="literal:64"),
            ConstantSpec(name="BLOCK_K", msl_type="uint", source="literal:32"),
        ),
        variant_params={
            "BM": "literal:64",
            "BN": "literal:64",
            "BK": "literal:32",
            "NW": "literal:4",
            "M_TOTAL": {"product": ["N", "OH", "OW"]},
            "K_TOTAL": {"product": ["KH", "KW", "C"]},
        },
    )
)

__all__ = [
    "BACKEND_ID",
    "ELEMENTWISE_LOWERER_ID",
    "ELEMENTWISE_LOWERER_VERSION",
    "CONV2D_PROFILE",
    "DEFAULT_FA_VARIANT",
    "DEFAULT_GEMM_VARIANT",
    "FLASH_ATTENTION_PROFILE",
    "GEMV_PROFILE",
    "GEMM_PROFILE",
    "GEMM_SILU_PROFILE",
    "LION_PROFILE",
    "RELU_PROFILE",
    "ROPE_PROFILE",
    "SIGMOID_PROFILE",
    "SILU_PROFILE",
    "SWIGLU_PROFILE",
    "LOWERER_ID",
    "LOWERER_VERSION",
    "REDUCTION_SUM_PROFILE",
    "RMSNORM_PROFILE",
    "RMSNORM_PROFILE_NAME",
    "SOFTMAX_PROFILE",
    "TARGET_ID",
]
