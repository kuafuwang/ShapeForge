"""Lower-level hmz Profile fencing for untrusted embedded DSL frontends.

A trusted driver running under hmz's own interpreter owns Proxy/Profile;
ShapeForge owns attempts, resource limits, watchdogs and artifact/evidence
contracts. This package does not enable a production DSL Track and does not
claim that outer attack validation has run.
"""

from .hmz_fence import (
    ArtifactContractError,
    AttemptSpec,
    HmzCapabilityUnavailable,
    HmzFenceExecutor,
    HmzInstallation,
    IsolationResult,
    ResourceLimits,
    collect_artifacts,
    inspect_hmz,
)

__all__ = [
    "ArtifactContractError",
    "AttemptSpec",
    "HmzCapabilityUnavailable",
    "HmzFenceExecutor",
    "HmzInstallation",
    "IsolationResult",
    "ResourceLimits",
    "collect_artifacts",
    "inspect_hmz",
]
