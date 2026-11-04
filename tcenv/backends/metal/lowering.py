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
__all__ = ["BACKEND_ID", "TARGET_ID"]
