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
from .op_desc import OP_DESC_TEMPLATE_KEYS, render_op_desc
from .v2 import (
    OPSPEC_V2_SCHEMA_PATH,
    OPSPEC_V2_SCHEMA_VERSION,
    SEMANTIC_RULES_VERSION_V2,
    VARLEN_CASE_DOMAIN,
    SemanticInstanceV2,
    instantiate_v2,
    load_opspec_v2,
    opspec_v2_from_document,
    validate_opspec_v2,
)
from .validator import SchemaFailure, validate

__all__ = [
    "OP_DESC_TEMPLATE_KEYS",
    "OPSPEC_SCHEMA_PATH",
    "OPSPEC_SCHEMA_VERSION",
    "OPSPEC_V2_SCHEMA_PATH",
    "OPSPEC_V2_SCHEMA_VERSION",
    "PROGRAM_KINDS",
    "SEMANTIC_RULES_VERSION",
    "SEMANTIC_RULES_VERSION_V2",
    "WORK_ACCOUNTING_VERSION",
    "CaseDomain",
    "OpSpec",
    "OpSpecError",
    "SchemaFailure",
    "SemanticInstance",
    "SemanticInstanceV2",
    "TensorBinding",
    "Tolerances",
    "VARLEN_CASE_DOMAIN",
    "WorkAccounting",
    "instantiate",
    "instantiate_v2",
    "load_opspec",
    "load_opspec_v2",
    "opspec_from_document",
    "opspec_v2_from_document",
    "render_op_desc",
    "validate",
    "validate_opspec",
    "validate_opspec_v2",
]
