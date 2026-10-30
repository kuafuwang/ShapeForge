"""OpSpec package: structural schema, semantic validation and ``instantiate``.

See :mod:`tcenv.opspec.model` for the rules and :mod:`tcenv.lowering` for ``lower``.
"""

from __future__ import annotations

from .model import (
    OPSPEC_SCHEMA_PATH,
    OPSPEC_SCHEMA_VERSION,
    PROGRAM_KINDS,
    SEMANTIC_RULES_VERSION,
    WORK_ACCOUNTING_VERSION,
    CaseDomain,
    OpSpec,
    OpSpecError,
    SemanticInstance,
    TensorBinding,
    Tolerances,
    WorkAccounting,
    instantiate,
    load_opspec,
    opspec_from_document,
    validate_opspec,
)
from .validator import SchemaFailure, validate

__all__ = [
    "OPSPEC_SCHEMA_PATH",
    "OPSPEC_SCHEMA_VERSION",
    "PROGRAM_KINDS",
    "SEMANTIC_RULES_VERSION",
    "WORK_ACCOUNTING_VERSION",
    "CaseDomain",
    "OpSpec",
    "OpSpecError",
    "SchemaFailure",
    "SemanticInstance",
    "TensorBinding",
    "Tolerances",
    "WorkAccounting",
    "instantiate",
    "load_opspec",
    "opspec_from_document",
    "validate",
    "validate_opspec",
]
