"""Registered Metal lowering: the preserved v1 tasks only.

A5 moved 16 tasks to the v2 standard entry; A6 deleted the nine ``LoweringProfile`` entries,
the two lowerers and the GEMM variant factory those tasks used.  What is registered here is the
v1 geometry for the preserved tasks (flash_attention/bitdelta_gemm/
mamba2_chunk_scan/sem_attention_odd_scale) plus the dynamic ``sem_int`` family and the
``@variant`` mechanism; a migrated task reads its contract from ``tasks/<op>/opspec.json`` and
its geometry from the candidate, never from this module.

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


def _lower_buffers_and_constants(
    instance: SemanticInstance, profile: LoweringProfile
) -> tuple[Contract, dict[str, str]]:
    """@brief 通用 buffer/常量派生（无 family 专属规则）。

    @param instance 语义实例。
    @param profile 登记的 profile。
    @return ``(contract, backend_params)``。
    """
    contract, backend_params = build_contract(instance, profile)
    return contract, dict(backend_params)


#: The four pointwise profiles (metal-relu/-sigmoid/-silu/-swiglu) lived here.  A5 moved
#: those tasks to the v2 standard entry and A6 deleted the profiles together with their
#: ``metal.elementwise`` lowerer: the candidate owns the geometry, so no ``N/BLOCK_SIZE``
#: registration is left for a migrated task to read.

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


#: The public interface of the *preserved* v1 lowering.
#:
#: A6 deleted nine migrated ``LoweringProfile`` entries (``metal-rmsnorm``, ``metal-softmax``,
#: ``metal-reduction-sum``, ``metal-relu``, ``metal-sigmoid``, ``metal-silu``, ``metal-swiglu``,
#: ``metal-gemm``, ``metal-gemm-silu``), the ``metal.elementwise``/``metal.gemm`` lowerers and
#: the ``metal.gemm.variant`` factory.  Their names must not be re-exported here: ``from
#: tcenv.backends.metal.lowering import *`` walks ``__all__`` and would raise ``AttributeError``
#: on the first missing name (a public import regression the A6 registry checks missed).
#: ``tests/backends/metal/test_a6_metal_config.py`` and ``A6/verify.py`` assert that every entry
#: exists and that ``import *`` works.
__all__ = [
    "BACKEND_ID",
    "BITDELTA_LOWERER_ID",
    "BITDELTA_LOWERER_VERSION",
    "BITDELTA_PROFILE",
    "DEFAULT_FA_VARIANT",
    "FA_LOWERER_ID",
    "FA_LOWERER_VERSION",
    "FA_VARIANT_FACTORY_ID",
    "FA_VARIANT_FACTORY_VERSION",
    "FLASH_ATTENTION_PROFILE",
    "MAMBA2_LOWERER_ID",
    "MAMBA2_LOWERER_VERSION",
    "MAMBA2_PROFILE",
    "SEM_INT_BLOCK",
    "SEM_INT_LOWERER_ID",
    "SEM_INT_LOWERER_VERSION",
    "SEM_INT_PROFILE",
    "TARGET_ID",
]
