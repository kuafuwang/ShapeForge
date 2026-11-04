"""Typed failures of the v2 task-contract plane (re-exported from ``tcenv.contract_errors``).

The two diagnostic names the plan requires are stable machine-readable codes, not prose:

* :class:`SemanticUnresolved` (``semantic_unresolved``) — the OpSpec, the reference oracle
  and the representation disagree.  Nothing auto-selects a side; a human/known-answer
  arbitration is required.
* :class:`UnsupportedContractCapability` (``unsupported_contract_capability``) — the
  document asks for a capability this contract version does not provide (a v2 ``@variant``,
  an unavailable track, an undeclared target limit).  It is never a fallback to v1.

A2 adds :class:`GeometryError` (``execution_geometry_invalid``) for typed geometry, derived
scalar and per-stage argument-binding failures; it is a ``TaskContractError`` so a caller that
only handles the base class still fails closed.

V1 adds :class:`VarlenContractError` (``varlen_contract_invalid``): a runtime ``cu_seqlens``
tensor that violates the L1 segmentation contract is refused before any host/device work.  Like
``GeometryError`` it subclasses :class:`TaskContractError`, so a caller that only handles the
base class still fails closed.
"""

from __future__ import annotations

from ..contract_errors import (
    AssemblyError,
    GeometryError,
    ReferenceBoundaryError,
    SemanticUnresolved,
    TaskContractError,
    UnsupportedContractCapability,
    VarlenContractError,
)

__all__ = [
    "AssemblyError",
    "GeometryError",
    "ReferenceBoundaryError",
    "SemanticUnresolved",
    "TaskContractError",
    "UnsupportedContractCapability",
    "VarlenContractError",
]
