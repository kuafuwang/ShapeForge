"""Compatibility import for the runtime evaluation interfaces.

The canonical definitions live in :mod:`tcenv.evaluation` so that a deployable
``tcenv.remote`` service never has to import the search package.  This module is kept
only for existing local callers and is intentionally a re-export, not a second model.
"""

from tcenv.evaluation import (
    LOCAL_ONLY_FIELDS,
    EvaluationExecutor,
    EvaluationRequest,
    EvaluationResult,
    EvaluatorProvenance,
    LocalEvaluator,
    local_provenance,
    request_digest,
)

__all__ = [
    "LOCAL_ONLY_FIELDS",
    "EvaluationExecutor",
    "EvaluationRequest",
    "EvaluationResult",
    "EvaluatorProvenance",
    "LocalEvaluator",
    "local_provenance",
    "request_digest",
]
