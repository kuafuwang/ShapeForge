"""Deterministic fake vendor A: source -> ELF/SIMT, DEVICE_EVENT timing.

Test-only.  The driver follows the production ``VendorDriver`` shape and the
controller follows ``DeviceController``; neither is imported by a production
composition root.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tcenv.remote.contracts import NPU_TIMING_EVIDENCE_SCHEMA_ID, npu_execution_document
from tcenv.remote.drivers.base import (
    CleanupProof,
    HealthProof,
    PreparedAttempt,
    VendorFacts,
    VendorOutcome,
)
from tcenv.remote.job_types import DeviceGroupBinding, NpuExecutionSpec

from .ledger import LedgerError, VendorLedger

VENDOR_ID = "vendor-a"
DEVICE_GROUP_ID = "vendor-a-group-0"
RESOURCE_GENERATION = "1"
CLOCK_DOMAIN = "vendor-a-device-event-clock-v1"


def _digest(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


_PROFILE = {
    "vendor_id": VENDOR_ID,
    "device_group_id": DEVICE_GROUP_ID,
    "driver_version": "fake-a-driver-1",
    "firmware_version": "fake-a-fw-1",
    "runtime_version": "fake-a-runtime-1",
    "logical_devices": 1,
    "timing_sources": ["DEVICE_EVENT"],
}
_TOPOLOGY = {
    "device_group_id": DEVICE_GROUP_ID,
    "logical_devices": 1,
    "rank": None,
    "die": None,
    "fabric": "none",
}
PROFILE_DIGEST = _digest(_PROFILE)
TOPOLOGY_DIGEST = _digest(_TOPOLOGY)

VENDOR_A_GROUP = DeviceGroupBinding(
    device_group_id=DEVICE_GROUP_ID,
    vendor_id=VENDOR_ID,
    profile_digest=PROFILE_DIGEST,
    topology_digest=TOPOLOGY_DIGEST,
    resource_generation=RESOURCE_GENERATION,
)


def build_execution(generation: str = RESOURCE_GENERATION) -> NpuExecutionSpec:
    """Return the fake vendor A binding; ``generation`` models a device reset."""

    if generation == RESOURCE_GENERATION:
        return NpuExecutionSpec(device_group=VENDOR_A_GROUP)
    return NpuExecutionSpec(device_group=DeviceGroupBinding(
        device_group_id=VENDOR_A_GROUP.device_group_id,
        vendor_id=VENDOR_A_GROUP.vendor_id,
        profile_digest=VENDOR_A_GROUP.profile_digest,
        topology_digest=VENDOR_A_GROUP.topology_digest,
        resource_generation=generation,
    ))


def _ledger_path() -> Path:
    value = os.environ.get("TCENV_VENDOR_LEDGER")
    if not value:
        raise RuntimeError("TCENV_VENDOR_LEDGER is required for the fake vendor")
    return Path(value)


@dataclass(frozen=True)
class _FakeModule:
    digest: str
    size: int


class FakeVendorADriver:
    """Worker-local cooperative driver used by the fake worker process."""

    vendor_id = VENDOR_ID

    def __init__(self, ledger: VendorLedger | None = None, *, generation: str | None = None) -> None:
        self.ledger = ledger or VendorLedger(_ledger_path())
        self.generation = generation or os.environ.get("TCENV_VENDOR_A_GENERATION") or RESOURCE_GENERATION

    def _compile_source_to_elf(self, source: bytes) -> bytes:
        # A deterministic fake ELF/SIMT container; not a real ABI or compiler.
        header = b"\x7fELF\x02\x01\x01\x00SIMT\x00"
        body = hashlib.sha256(b"vendor-a-source\0" + source).digest()
        return header + body + source

    def prepare(
        self, spec: Any, source: bytes, attempt_context: Mapping[str, Any],
    ) -> PreparedAttempt:
        if not isinstance(spec, NpuExecutionSpec):
            raise ValueError("vendor A requires the typed NpuExecutionSpec")
        normalized = npu_execution_document(spec.to_dict())
        expected_group = {**VENDOR_A_GROUP.to_dict(), "resource_generation": self.generation}
        if normalized["device_group"] != expected_group:
            raise ValueError("vendor A received a different device group/profile/topology/generation")
        attempt_id = attempt_context.get("attempt_id")
        lease_fence = attempt_context.get("lease_fence")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("vendor A requires an attempt id")
        if isinstance(lease_fence, bool) or not isinstance(lease_fence, int) or lease_fence < 1:
            raise ValueError("vendor A requires a positive lease fence")
        module = _FakeModule(
            digest=hashlib.sha256(self._compile_source_to_elf(source)).hexdigest(),
            size=len(source) + 18,
        )
        group = VENDOR_A_GROUP
        self.ledger.record_resource(
            attempt_id=attempt_id, lease_fence=lease_fence,
            device_group_id=group.device_group_id,
            resource_generation=group.resource_generation,
            kind="module", detail={"format": "ELF/SIMT", "sha256": module.digest, "size": module.size},
        )
        self.ledger.record_resource(
            attempt_id=attempt_id, lease_fence=lease_fence,
            device_group_id=group.device_group_id,
            resource_generation=group.resource_generation,
            kind="stream", detail={"synchronization": "stream-event"},
        )
        self.ledger.record_resource(
            attempt_id=attempt_id, lease_fence=lease_fence,
            device_group_id=group.device_group_id,
            resource_generation=group.resource_generation,
            kind="event", detail={"timing_source": "DEVICE_EVENT"},
        )
        return PreparedAttempt(
            attempt_id=attempt_id,
            lease_fence=lease_fence,
            device_group_id=group.device_group_id,
            resource_generation=group.resource_generation,
            module_format="ELF/SIMT",
            module_sha256=module.digest,
            handles={
                "module": module.digest,
                "stream": f"{attempt_id}:stream",
                "source_sha256": hashlib.sha256(source).hexdigest(),
            },
        )

    def execute(
        self, prepared: PreparedAttempt, *, deadline_monotonic: float, progress: Any = None,
    ) -> VendorOutcome:
        if not isinstance(prepared, PreparedAttempt):
            raise ValueError("vendor A requires a prepared attempt")
        started = time.monotonic_ns()
        self.ledger.mark_device_running(
            attempt_id=prepared.attempt_id, lease_fence=prepared.lease_fence,
        )
        if progress is not None:
            progress({"stage": "compile", "completed_reps": 0, "total_reps": 1, "detail": "fake ELF/SIMT prepared"})
        if os.environ.get("TCENV_VENDOR_A_HANG") == "1":
            while time.monotonic() < deadline_monotonic:
                time.sleep(0.01)
            raise TimeoutError("fake vendor A device did not complete before the deadline")
        if os.environ.get("TCENV_VENDOR_A_ABORT_AFTER_LAUNCH") == "1" and prepared.lease_fence == 1:
            os._exit(9)
        if os.environ.get("TCENV_VENDOR_A_SLOW") == "1":
            time.sleep(1.5)
        # Simulated device time is deterministic and independent of host wall time.
        device_elapsed_ns = 1_000_000 + int(prepared.module_sha256[:8], 16) % 500_000
        host_api_elapsed_ns = max(1, time.monotonic_ns() - started)
        time.sleep(0.01)
        if os.environ.get("TCENV_VENDOR_A_HOST_SLOW") == "1":
            time.sleep(0.05)
            host_api_elapsed_ns = max(host_api_elapsed_ns, 50_000_000)
        evidence = {
            "schema_id": NPU_TIMING_EVIDENCE_SCHEMA_ID,
            "candidate_digest": {
                "algorithm": "sha256",
                "value": prepared.handles["source_sha256"],
            },
            "device_group": {**VENDOR_A_GROUP.to_dict(), "resource_generation": self.generation},
            "timing": {
                "source": "DEVICE_EVENT",
                "clock_domain": CLOCK_DOMAIN,
                "synchronization_completed": True,
                "device_elapsed_ns": device_elapsed_ns,
                "host_api_elapsed_ns": host_api_elapsed_ns,
                "launches": 1,
            },
            "measurement_kind": "simulated",
            "publication_allowed": False,
            "complete": True,
            "diagnostics": {
                "vendor_id": VENDOR_ID,
                "artifact_format": prepared.module_format,
                "artifact_sha256": prepared.module_sha256,
                "device_event_sequence": 1,
            },
        }
        return VendorOutcome(report={}, evidence=evidence, module_format="ELF/SIMT", module_sha256=prepared.module_sha256)

    def close(self, prepared: PreparedAttempt) -> None:
        if os.environ.get("TCENV_VENDOR_A_SKIP_CLOSE") == "1":
            return
        self.ledger.release_attempt(prepared.attempt_id, prepared.lease_fence)


class FakeVendorAController:
    """Worker-external controller backed by the durable fake ledger."""

    vendor_id = VENDOR_ID

    def __init__(
        self, ledger_path: Path | str, *, release_supported: bool = True,
        healthy: bool = True, generation: str = RESOURCE_GENERATION,
    ) -> None:
        self.ledger = VendorLedger(ledger_path)
        self.release_supported = bool(release_supported)
        self.healthy = bool(healthy)
        self.generation = generation

    def probe(self, profile: Mapping[str, Any]) -> VendorFacts:
        if profile.get("device_group_id") != DEVICE_GROUP_ID:
            raise ValueError("fake vendor A probe received another device group")
        self.ledger.record_probe(
            vendor_version="fake-a-1", resource_generation=self.generation,
        )
        return VendorFacts(
            vendor_id=VENDOR_ID, device_group_id=DEVICE_GROUP_ID,
            profile_digest=PROFILE_DIGEST, topology_digest=TOPOLOGY_DIGEST,
            resource_generation=self.generation, logical_devices=1,
            driver_version="fake-a-driver-1", firmware_version="fake-a-fw-1",
            runtime_version="fake-a-runtime-1",
            timing_sources=("DEVICE_EVENT",),
            state="available" if self.healthy else "quarantined",
            diagnostics={"simulated": True},
        )

    def capabilities(self, facts: VendorFacts) -> Mapping[str, Any]:
        if facts.device_group_id != DEVICE_GROUP_ID:
            raise ValueError("fake vendor A capability group mismatch")
        return {
            "artifact_kinds": ["source"],
            "launch_kinds": ["KERNEL"],
            "launch_subtypes": ["SIMT"],
            "timing_sources": ["DEVICE_EVENT"],
            "device_groups": [{
                **{**VENDOR_A_GROUP.to_dict(), "resource_generation": self.generation},
                "logical_devices": 1,
                "state": "available" if self.healthy else "quarantined",
                "timing_sources": ["DEVICE_EVENT"],
            }],
        }

    def cancel(self, attempt_id: str, lease_fence: int) -> None:
        self.ledger.mark_cancelled(attempt_id=attempt_id, lease_fence=lease_fence)

    def _proof(self, attempt_id: str, lease_fence: int, **fields: Any) -> CleanupProof:
        """Every proof names the exact device binding it was produced for."""

        group = VENDOR_A_GROUP
        return CleanupProof(
            attempt_id=attempt_id, lease_fence=lease_fence,
            device_group_id=group.device_group_id,
            resource_generation=self.generation,
            vendor_id=group.vendor_id,
            profile_digest=group.profile_digest,
            topology_digest=group.topology_digest,
            **fields,
        )

    def reconcile(self, attempt_id: str, lease_fence: int) -> CleanupProof:
        try:
            resources = self.ledger.attempt_resources(attempt_id, lease_fence)
            live = self.ledger.live_resources()
        except LedgerError as exc:
            return self._proof(
                attempt_id, lease_fence,
                complete=False, detail=f"ledger unreadable: {exc}",
            )
        stale = [
            resource for resource in live
            if resource["attempt_id"] == attempt_id and resource["lease_fence"] != lease_fence
        ]
        if stale:
            return self._proof(
                attempt_id, lease_fence,
                complete=False,
                live_resources=tuple(sorted(r["kind"] for r in stale)),
                detail="resources belong to a stale fence of the same attempt",
            )
        if not resources:
            return self._proof(
                attempt_id, lease_fence,
                complete=True, detail="no live resources for this attempt/fence",
            )
        if not self.release_supported:
            return self._proof(
                attempt_id, lease_fence,
                complete=False,
                live_resources=tuple(sorted(r["kind"] for r in resources)),
                detail="fake controller cannot prove external release",
            )
        released = self.ledger.release_attempt(attempt_id, lease_fence)
        remaining = self.ledger.attempt_resources(attempt_id, lease_fence)
        if remaining:
            return self._proof(
                attempt_id, lease_fence,
                complete=False,
                live_resources=tuple(sorted(r["kind"] for r in remaining)),
                released_resources=tuple(released),
                detail="external release left live resources",
            )
        return self._proof(
            attempt_id, lease_fence,
            complete=True, released_resources=tuple(released),
            detail="external release verified against the attempt/fence ledger",
        )

    def health_check(self, profile: Mapping[str, Any]) -> HealthProof:
        if profile.get("device_group_id") != DEVICE_GROUP_ID:
            return HealthProof(False, DEVICE_GROUP_ID, RESOURCE_GENERATION, "wrong device group")
        self.ledger.record_probe(
            vendor_version="fake-a-1", resource_generation=self.generation,
        )
        return HealthProof(
            ok=self.healthy, device_group_id=DEVICE_GROUP_ID,
            resource_generation=self.generation,
            detail="fake known-good probe" if self.healthy else "fake known-good probe failed",
            executed_known_good=True,
        )
