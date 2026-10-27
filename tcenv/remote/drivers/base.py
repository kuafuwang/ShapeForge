"""Typed vendor/device lifecycle boundary shared by production and tests.

``VendorDriver`` runs inside the attempt worker and is cooperative: it maps a
validated execution spec to typed device operations.  ``DeviceController`` runs
outside the worker/frontend attempt container and is the only component allowed
to claim device resources were released after a kill, timeout or cancel.

The P0 source-only NPU contract uses one logical device group and DEVICE_EVENT
timing.  Nothing here exposes a shell, arbitrary path, BDF, rank, arbitrary
vendor key-value or vendor-specific RPC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class VendorFacts:
    """Probe result for one trusted logical device group."""

    vendor_id: str
    device_group_id: str
    profile_digest: str
    topology_digest: str
    resource_generation: str
    logical_devices: int = 1
    #: Rank/die dimensions of the trusted profile.  0 means the vendor shape has
    #: no such dimension (a graph/service vendor); an agent can then never
    #: demand rank or die evidence, and evidence that invents one is refused.
    ranks: int = 0
    dies: int = 0
    driver_version: str = ""
    firmware_version: str = ""
    runtime_version: str = ""
    timing_sources: tuple[str, ...] = ("DEVICE_EVENT",)
    state: str = "available"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HealthProof:
    ok: bool
    device_group_id: str
    resource_generation: str
    detail: str = ""
    executed_known_good: bool = False


@dataclass(frozen=True)
class CleanupProof:
    """Whether one attempt/fence's device resources are provably gone.

    ``complete=False`` must become quarantine, never a successful cancel,
    timeout or retry.  The proof is bound to the exact attempt and lease fence so
    an old worker cannot clear a newer attempt.
    """

    attempt_id: str
    lease_fence: int
    device_group_id: str
    resource_generation: str
    complete: bool
    vendor_id: str = ""
    profile_digest: str = ""
    topology_digest: str = ""
    live_resources: tuple[str, ...] = ()
    released_resources: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class PreparedAttempt:
    """Worker-local preparation result; never carries a host path or command."""

    attempt_id: str
    lease_fence: int
    device_group_id: str
    resource_generation: str
    module_format: str = "ELF/SIMT"
    module_sha256: str = ""
    handles: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VendorOutcome:
    """Typed worker outcome plus the device-side timing evidence."""

    report: Mapping[str, Any]
    evidence: Mapping[str, Any]
    module_format: str = "ELF/SIMT"
    module_sha256: str = ""


@runtime_checkable
class VendorDriver(Protocol):
    """Cooperative, worker-local device driver."""

    vendor_id: str

    def prepare(
        self, spec: Any, artifact: Any, attempt_context: Mapping[str, Any],
    ) -> PreparedAttempt: ...
    #: ``artifact`` is the exact source bytes for the inline-source shape, or the
    #: typed ArtifactBundle for the artifact-set shape.  It is never a path.

    def execute(
        self, prepared: PreparedAttempt, *, deadline_monotonic: float,
        progress: Any = None,
    ) -> VendorOutcome: ...

    def close(self, prepared: PreparedAttempt) -> None: ...


@runtime_checkable
class DeviceController(Protocol):
    """Trusted, worker-external device lifecycle helper."""

    vendor_id: str

    def probe(self, profile: Mapping[str, Any]) -> VendorFacts: ...

    def capabilities(self, facts: VendorFacts) -> Mapping[str, Any]: ...

    def cancel(self, attempt_id: str, lease_fence: int) -> None: ...

    def reconcile(self, attempt_id: str, lease_fence: int) -> CleanupProof: ...

    def health_check(self, profile: Mapping[str, Any]) -> HealthProof: ...
