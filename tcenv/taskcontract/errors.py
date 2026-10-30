"""Typed failures of the v2 task-contract plane (re-exported from ``tcenv.contract_errors``).

The two diagnostic names the plan requires are stable machine-readable codes, not prose:

* :class:`SemanticUnresolved` (``semantic_unresolved``) — the OpSpec, the reference oracle
  and the representation disagree.  Nothing auto-selects a side; a human/known-answer
  arbitration is required.
* :class:`UnsupportedContractCapability` (``unsupported_contract_capability``) — the
  document asks for a capability this contract version does not provide (a v2 ``@variant``,
  a workspace tensor before the A2 ABI exists, an unavailable track).  It is never a
  fallback to v1.
"""

from __future__ import annotations

from ..contract_errors import (
    AssemblyError,
    ReferenceBoundaryError,
    SemanticUnresolved,
    TaskContractError,
    UnsupportedContractCapability,
)

__all__ = [
    "AssemblyError",
    "ReferenceBoundaryError",
    "SemanticUnresolved",
    "TaskContractError",
    "UnsupportedContractCapability",
]
