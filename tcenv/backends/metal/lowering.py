"""Registered Metal lowering: the preserved v1 tasks only.

A5 moved 16 tasks to the v2 standard entry; A6 deleted the nine ``LoweringProfile`` entries,
the two lowerers and the GEMM variant factory those tasks used.  What is registered here is the
v1 geometry for the remaining semantic tasks (sem_int dynamic family;
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


# ------------------------------------------------------------------------ FlashAttention
# flash_attention 已迁往 v2；sem_attention_odd_scale 仍在 v1 并共享同一个 profile，
# 所以 profile 与 lowerer 保留（variant factory 随 flash_attention 退役）。
FA_LOWERER_ID = "metal.flash-attention"
FA_LOWERER_VERSION = "1"
DEFAULT_FA_VARIANT = "32x32t128"

register_lowerer(FA_LOWERER_ID, FA_LOWERER_VERSION, _lower_buffers_and_constants)



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
    "FA_LOWERER_ID",
    "FA_LOWERER_VERSION",
    "FLASH_ATTENTION_PROFILE",
    "SEM_INT_BLOCK",
    "SEM_INT_LOWERER_ID",
    "SEM_INT_LOWERER_VERSION",
    "SEM_INT_PROFILE",
    "TARGET_ID",
]
