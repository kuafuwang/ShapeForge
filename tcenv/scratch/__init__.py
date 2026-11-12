"""Candidate-owned scratch declarations.  Empty scratch stays out of canonical documents."""

from .spec import (
    DEFAULT_MAX_SCRATCH_BUFFERS,
    DEFAULT_MAX_TOTAL_SCRATCH_BYTES,
    INIT_POLICIES,
    SCRATCH_ACCESS,
    ResolvedScratch,
    ScratchSpec,
    ScratchUse,
    bundle_delivery_allowed,
    check_scratch_lifecycle,
    parse_scratch_list,
    parse_scratch_uses,
    scratch_nbytes,
)

__all__ = [
    "DEFAULT_MAX_SCRATCH_BUFFERS",
    "DEFAULT_MAX_TOTAL_SCRATCH_BYTES",
    "INIT_POLICIES",
    "SCRATCH_ACCESS",
    "ResolvedScratch",
    "ScratchSpec",
    "ScratchUse",
    "bundle_delivery_allowed",
    "check_scratch_lifecycle",
    "parse_scratch_list",
    "parse_scratch_uses",
    "scratch_nbytes",
]
