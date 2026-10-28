"""Test-only fake NPU module vendor: prebuilt or evaluator-built ``TXELF`` -> load.

This vendor is the P1c device-side stand-in.  It is deliberately *not* a
compiler: it accepts a versioned artifact set, re-verifies the exact bytes of
every executable module against the manifest, records what it loaded, and only
then observes a simulated device event.  It has no git/LLM credentials, no
source download and no build command; the source never arrives in the prebuilt
case and is never compiled here.

The controller follows the production ``DeviceController`` protocol and reuses
the cross-process fake ledger, so cleanup and health remain worker-external and
provable (or quarantined).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from tcenv.remote.artifact_builder import (
    BuildArtifact,
    BuildEvidence,
    BuildProvenance,
    BuildProfile,
)
from tcenv.remote.artifact_bundle import ArtifactBundle
from tcenv.remote.contracts import (
    BUNDLE_DIGEST_ALGORITHM,
    NPU_BUILD_EVIDENCE_SCHEMA_ID,
    NPU_BUILD_PROVENANCE_SCHEMA_ID,
    NPU_BUILD_TIMING_EVIDENCE_SCHEMA_ID,
    build_provenance_document,
    npu_build_execution_document,
)
from tcenv.remote.drivers.base import (
    CleanupProof,
    HealthProof,
    PreparedAttempt,
    VendorFacts,
    VendorOutcome,
)
from tcenv.remote.job_types import DeviceGroupBinding, NpuExecutionSpec, NpuResources

from .ledger import LedgerError, VendorLedger, resource_matches_binding

VENDOR_ID = "vendor-c"
DEVICE_GROUP_ID = "vendor-c-group-0"
RESOURCE_GENERATION = "1"
CLOCK_DOMAIN = "vendor-c-device-event-clock-v1"
LEDGER_ENV = "TCENV_VENDOR_LEDGER"
GENERATION_ENV = "TCENV_VENDOR_C_GENERATION"
SKIP_CLOSE_ENV = "TCENV_VENDOR_C_SKIP_CLOSE"
#: Test-only fault: the vendor reports a build identity that differs from the
#: accepted request (a forged BuildProvenance), so the signed-evidence check can
#: be proven to refuse it before a result is committed.
FORGE_PROVENANCE_ENV = "TCENV_VENDOR_C_FORGE_BUILD_PROVENANCE"

MODULE_ROLE = "EXECUTABLE_MODULE"


def _digest(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


_PROFILE = {
    "vendor_id": VENDOR_ID,
    "device_group_id": DEVICE_GROUP_ID,
    "driver_version": "fake-c-driver-1",
    "firmware_version": "fake-c-fw-1",
    "runtime_version": "fake-c-runtime-1",
    "logical_devices": 1,
    "ranks": 0,
    "dies": 0,
    "timing_sources": ["DEVICE_EVENT"],
    "execution_model": "prebuilt-module-load",
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

VENDOR_C_GROUP = DeviceGroupBinding(
    device_group_id=DEVICE_GROUP_ID,
    vendor_id=VENDOR_ID,
    profile_digest=PROFILE_DIGEST,
    topology_digest=TOPOLOGY_DIGEST,
    resource_generation=RESOURCE_GENERATION,
)


def _group(generation: str) -> DeviceGroupBinding:
    if generation == RESOURCE_GENERATION:
        return VENDOR_C_GROUP
    return DeviceGroupBinding(
        device_group_id=VENDOR_C_GROUP.device_group_id,
        vendor_id=VENDOR_C_GROUP.vendor_id,
        profile_digest=VENDOR_C_GROUP.profile_digest,
        topology_digest=VENDOR_C_GROUP.topology_digest,
        resource_generation=generation,
    )


def build_module_execution(
    generation: str = RESOURCE_GENERATION, *, resources: NpuResources | None = None,
) -> NpuExecutionSpec:
    """The prebuilt executable-module shape (document version 5)."""

    return NpuExecutionSpec(
        device_group=_group(generation),
        artifact_kind="bundle",
        launch_kind="KERNEL",
        launch_subtype="SIMT",
        source_media_type="text/plain; charset=utf-8",
        synchronization="stream-event",
        spec_version=5,
        task_family="kernel_module",
        resources=resources or NpuResources(),
    )


def build_evaluator_execution(
    generation: str = RESOURCE_GENERATION, *, resources: NpuResources | None = None,
) -> NpuExecutionSpec:
    """The evaluator-build shape: inline source, built in the same attempt."""

    return NpuExecutionSpec(
        device_group=_group(generation),
        artifact_kind="source",
        launch_kind="KERNEL",
        launch_subtype="SIMT",
        source_media_type="text/plain; charset=utf-8",
        synchronization="stream-event",
        spec_version=4,
        task_family="kernel",
        resources=resources or NpuResources(),
    )


def ledger_path() -> Path:
    value = os.environ.get(LEDGER_ENV)
    if not value:
        raise RuntimeError(f"{LEDGER_ENV} is required for the fake module vendor")
    return Path(value)


class FakeModuleVendorDriver:
    """Worker-local cooperative driver: verify + load, never compile."""

    vendor_id = VENDOR_ID

    def __init__(self, ledger: VendorLedger | None = None, *, generation: str | None = None) -> None:
        self.ledger = ledger or VendorLedger(ledger_path())
        self.generation = generation or os.environ.get(GENERATION_ENV) or RESOURCE_GENERATION

    def binding(self) -> dict[str, str]:
        return {**VENDOR_C_GROUP.to_dict(), "resource_generation": self.generation}

    def prepare(
        self, spec: Any, artifact: Any, attempt_context: Mapping[str, Any],
    ) -> PreparedAttempt:
        if not isinstance(spec, NpuExecutionSpec):
            raise ValueError("vendor C requires the typed NpuExecutionSpec")
        normalized = npu_build_execution_document(spec.to_dict())
        if normalized["launch_kind"] != "KERNEL" or normalized["synchronization"] != "stream-event":
            raise ValueError("vendor C serves only KERNEL/SIMT/stream-event")
        if normalized["device_group"] != self.binding():
            raise ValueError("vendor C received a different device group/profile/topology/generation")
        if not isinstance(artifact, ArtifactBundle):
            raise ValueError("vendor C requires a typed artifact bundle")
        attempt_id = attempt_context.get("attempt_id")
        lease_fence = attempt_context.get("lease_fence")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("vendor C requires an attempt id")
        if isinstance(lease_fence, bool) or not isinstance(lease_fence, int) or lease_fence < 1:
            raise ValueError("vendor C requires a positive lease fence")
        candidate = attempt_context.get("candidate_digest")
        if not isinstance(candidate, Mapping):
            raise ValueError("vendor C requires the accepted CandidateDigest")
        if candidate.get("algorithm") == BUNDLE_DIGEST_ALGORITHM:
            if candidate.get("value") != artifact.artifact_set_digest:
                raise ValueError("vendor C candidate digest does not bind the artifact set")
        elif candidate.get("algorithm") == "sha256":
            # An evaluator build binds the *source* CandidateDigest; the built
            # artifact set is described by the build evidence and is never passed
            # off as the candidate identity.
            source = attempt_context.get("source")
            if not isinstance(source, bytes) or hashlib.sha256(source).hexdigest() != candidate.get("value"):
                raise ValueError("vendor C source candidate digest does not bind the built source")
        else:
            raise ValueError("vendor C received an unknown candidate digest algorithm")
        build_request = attempt_context.get("build")
        if not isinstance(build_request, Mapping) or build_request.get("mode") not in (
            "prebuilt", "evaluator",
        ):
            raise ValueError("vendor C requires the accepted build identity")
        provenance = build_provenance_document(build_request["provenance"])
        if provenance["artifact_set_digest"] is not None and (
            provenance["artifact_set_digest"] != artifact.artifact_set_digest
        ):
            raise ValueError("vendor C build provenance does not bind the artifact set")
        source_entries = [entry for entry in artifact.entries if entry.role == "SOURCE"]
        if len(source_entries) != 1:
            raise ValueError("vendor C requires exactly one SOURCE artifact")
        if source_entries[0].sha256 != provenance["source_sha256"]:
            raise ValueError("vendor C source artifact differs from the build provenance")
        modules = [entry for entry in artifact.entries if entry.role == MODULE_ROLE]
        if not modules:
            raise ValueError("vendor C requires an EXECUTABLE_MODULE artifact")
        loaded_bytes = attempt_context.get("loaded_artifacts") or {}
        # Load each module by re-verifying the exact bytes the caller received.
        load_records = []
        for entry in modules:
            data = loaded_bytes.get(entry.path)
            if data is None:
                raise ValueError(f"vendor C received no bytes for {entry.path!r}")
            if len(data) != entry.size_bytes or hashlib.sha256(data).hexdigest() != entry.sha256:
                raise ValueError(f"vendor C module {entry.path!r} does not match its manifest")
            load_records.append({
                "path": entry.path, "sha256": entry.sha256, "size_bytes": entry.size_bytes,
            })
            self.ledger.record_module_load(
                attempt_id=attempt_id, lease_fence=lease_fence, binding=self.binding(),
                path=entry.path, sha256=entry.sha256, size_bytes=entry.size_bytes,
            )
        group = self.binding()
        for kind, detail in (
            ("module", {"format": "TXELF/SIMT", "modules": load_records}),
            ("stream", {"synchronization": "stream-event"}),
            ("event", {"timing_source": "DEVICE_EVENT"}),
        ):
            self.ledger.record_resource(
                attempt_id=attempt_id, lease_fence=lease_fence,
                device_group_id=group["device_group_id"], vendor_id=group["vendor_id"],
                profile_digest=group["profile_digest"], topology_digest=group["topology_digest"],
                resource_generation=group["resource_generation"], kind=kind, detail=detail,
            )
        # The BuildEvidence the worker prepared: prebuilt => reused, evaluator =>
        # the builder's own outcome.  Both are extended with what was loaded.
        evidence = attempt_context.get("build_evidence")
        if not isinstance(evidence, BuildEvidence):
            raise ValueError("vendor C requires the typed build evidence")
        evidence = replace(
            evidence,
            loaded=tuple((record["path"], record["sha256"]) for record in load_records),
        )
        received_provenance = evidence.provenance.to_dict()
        if received_provenance["artifact_set_digest"] != artifact.artifact_set_digest:
            raise ValueError("the received build evidence does not bind the loaded artifact set")
        for field in (
            "builder_id", "builder_version", "build_host", "build_profile",
            "toolchain", "compile_profile", "source_sha256",
        ):
            if received_provenance[field] != provenance[field]:
                raise ValueError(f"the received build evidence changes {field}")
        # The receiver records the *completed* build identity it accepted and
        # loaded, so the ledger and the signed evidence close over the same
        # provenance.
        self.ledger.record_build(
            attempt_id=attempt_id, lease_fence=lease_fence, binding=group,
            mode=build_request["mode"], provenance=received_provenance,
            artifacts=[entry.manifest() for entry in artifact.entries],
            load_records=load_records,
        )
        reported = evidence
        if os.environ.get(FORGE_PROVENANCE_ENV) == "1":
            # The receiver accepted and recorded the genuine identity above, but
            # now reports another build identity to its own worker.  Only the
            # signed-evidence check (which compares against the accepted request)
            # can refuse this, so it must refuse before any result is written.
            reported = replace(
                evidence,
                provenance=BuildProvenance(
                    profile=replace(evidence.provenance.profile, builder_version="9.9.9"),
                    source_sha256=evidence.provenance.source_sha256,
                    artifact_set_digest=evidence.provenance.artifact_set_digest,
                ),
            )
        return PreparedAttempt(
            attempt_id=attempt_id,
            lease_fence=lease_fence,
            device_group_id=group["device_group_id"],
            resource_generation=group["resource_generation"],
            module_format="TXELF/SIMT",
            module_sha256=load_records[0]["sha256"],
            handles={
                "modules": json.dumps(load_records, sort_keys=True),
                "build": reported.to_dict(),
                "build_mode": build_request["mode"],
                "candidate_digest": {
                    "algorithm": str(candidate["algorithm"]), "value": str(candidate["value"]),
                },
                "candidate_algorithm": str(candidate["algorithm"]),
                "artifact_set_digest": artifact.artifact_set_digest,
                "source_sha256": provenance["source_sha256"],
            },
        )

    def execute(
        self, prepared: PreparedAttempt, *, deadline_monotonic: float, progress: Any = None,
    ) -> VendorOutcome:
        if not isinstance(prepared, PreparedAttempt):
            raise ValueError("vendor C requires a prepared attempt")
        self.ledger.mark_device_running(
            attempt_id=prepared.attempt_id, lease_fence=prepared.lease_fence,
            binding=self.binding(),
        )
        if progress is not None:
            progress({
                "stage": "compile", "completed_reps": 0, "total_reps": 1,
                "detail": "prebuilt module verified and loaded",
            })
        device_elapsed_ns = 1_200_000 + int(prepared.module_sha256[:8], 16) % 600_000
        host_api_elapsed_ns = 2_000_000
        evidence = {
            "schema_id": NPU_BUILD_TIMING_EVIDENCE_SCHEMA_ID,
            "candidate_digest": dict(prepared.handles["candidate_digest"]),
            # The top-level artifact-set identity follows the *candidate*: a
            # prebuilt artifact set is the candidate, while an evaluator build
            # binds the source candidate and describes the built set only in the
            # signed build evidence.
            "artifact_set_digest": (
                prepared.handles["artifact_set_digest"]
                if prepared.handles["candidate_algorithm"] == BUNDLE_DIGEST_ALGORITHM else None
            ),
            "device_group": self.binding(),
            "topology": {
                "logical_devices": 1,
                "declared_devices": ["device-0"],
                "required_rank_spans": None,
                "required_die_spans": None,
            },
            "spans": [{
                "device_id": "device-0", "rank": None, "die": None,
                "clock_domain": CLOCK_DOMAIN, "elapsed_ns": device_elapsed_ns,
            }],
            "timing": {
                "source": "DEVICE_EVENT",
                "clock_domain": CLOCK_DOMAIN,
                "synchronization_completed": True,
                "device_elapsed_ns": device_elapsed_ns,
                "host_api_elapsed_ns": host_api_elapsed_ns,
                "launches": 1,
            },
            "buffer": None,
            "build": dict(prepared.handles["build"]),
            "measurement_kind": "simulated",
            "publication_allowed": False,
            "complete": True,
            "diagnostics": {
                "vendor_id": VENDOR_ID,
                "build_mode": prepared.handles["build_mode"],
                "loaded_modules": prepared.handles["modules"],
                "compiled_in_attempt": prepared.handles["build_mode"] == "evaluator",
            },
        }
        return VendorOutcome(
            report={}, evidence=evidence, module_format="TXELF/SIMT",
            module_sha256=prepared.module_sha256,
        )

    def close(self, prepared: PreparedAttempt) -> None:
        if os.environ.get(SKIP_CLOSE_ENV) == "1":
            return
        try:
            self.ledger.release_attempt(
                prepared.attempt_id, prepared.lease_fence, binding=self.binding(),
            )
        except LedgerError:
            # Unreachable ledger is not a proof of cleanup: the worker-external
            # controller decides, exactly like the other fake vendors.
            return


class FakeModuleVendorController:
    """Worker-external controller for the prebuilt/evaluator module vendor."""

    vendor_id = VENDOR_ID

    def __init__(
        self, ledger_path: Path | str, *, release_supported: bool = True,
        healthy: bool = True, generation: str = RESOURCE_GENERATION,
        request_artifact_kind: str = "bundle",
    ) -> None:
        self.ledger = VendorLedger(ledger_path)
        self.release_supported = bool(release_supported)
        self.healthy = bool(healthy)
        self.generation = generation
        #: What a client may *submit*: a prebuilt artifact set, or inline source
        #: this frontend builds.  The device itself always loads a module.
        if request_artifact_kind not in ("bundle", "source"):
            raise ValueError("request_artifact_kind must be bundle or source")
        self.request_artifact_kind = request_artifact_kind

    def binding(self) -> dict[str, str]:
        return {**VENDOR_C_GROUP.to_dict(), "resource_generation": self.generation}

    def _busy(self) -> bool:
        if not self.ledger.exists():
            return False
        try:
            return bool(self.ledger.running_markers())
        except LedgerError:
            return True

    def probe(self, profile: Mapping[str, Any]) -> VendorFacts:
        if profile.get("device_group_id") != DEVICE_GROUP_ID:
            raise ValueError("fake vendor C probe received another device group")
        if self._busy():
            return VendorFacts(
                vendor_id=VENDOR_ID, device_group_id=DEVICE_GROUP_ID,
                profile_digest=PROFILE_DIGEST, topology_digest=TOPOLOGY_DIGEST,
                resource_generation=self.generation, logical_devices=1, ranks=0, dies=0,
                driver_version="fake-c-driver-1", firmware_version="fake-c-fw-1",
                runtime_version="fake-c-runtime-1",
                timing_sources=("DEVICE_EVENT",), state="busy",
                diagnostics={"simulated": True, "busy": True},
            )
        self.ledger.record_probe(vendor_version="fake-c-1", resource_generation=self.generation)
        return VendorFacts(
            vendor_id=VENDOR_ID, device_group_id=DEVICE_GROUP_ID,
            profile_digest=PROFILE_DIGEST, topology_digest=TOPOLOGY_DIGEST,
            resource_generation=self.generation, logical_devices=1, ranks=0, dies=0,
            driver_version="fake-c-driver-1", firmware_version="fake-c-fw-1",
            runtime_version="fake-c-runtime-1",
            timing_sources=("DEVICE_EVENT",),
            state="available" if self.healthy else "quarantined",
            diagnostics={"simulated": True, "execution_model": "prebuilt-module-load"},
        )

    def capabilities(self, facts: VendorFacts) -> Mapping[str, Any]:
        if facts.device_group_id != DEVICE_GROUP_ID:
            raise ValueError("fake vendor C capability group mismatch")
        return {
            "artifact_kinds": [self.request_artifact_kind],
            "launch_kinds": ["KERNEL"],
            "launch_subtypes": ["SIMT"],
            "timing_sources": ["DEVICE_EVENT"],
            "device_groups": [{
                **self.binding(),
                "logical_devices": 1,
                "state": "available" if self.healthy else "quarantined",
                "timing_sources": ["DEVICE_EVENT"],
            }],
        }

    def cancel(self, attempt_id: str, lease_fence: int) -> None:
        self.ledger.mark_cancelled(attempt_id=attempt_id, lease_fence=lease_fence)

    def _proof(self, attempt_id: str, lease_fence: int, **fields: Any) -> CleanupProof:
        group = VENDOR_C_GROUP
        return CleanupProof(
            attempt_id=attempt_id, lease_fence=lease_fence,
            device_group_id=group.device_group_id, resource_generation=self.generation,
            vendor_id=group.vendor_id, profile_digest=group.profile_digest,
            topology_digest=group.topology_digest, **fields,
        )

    def reconcile(self, attempt_id: str, lease_fence: int) -> CleanupProof:
        if not self.ledger.exists():
            return self._proof(
                attempt_id, lease_fence, complete=False,
                detail="the fake vendor ledger is missing; cleanup cannot be proven",
            )
        try:
            resources = self.ledger.attempt_resources(attempt_id, lease_fence)
            live = self.ledger.live_resources()
        except LedgerError as exc:
            return self._proof(
                attempt_id, lease_fence, complete=False,
                detail=f"ledger unreadable: {exc}",
            )
        stale = [
            resource for resource in live
            if resource["attempt_id"] == attempt_id and resource["lease_fence"] != lease_fence
        ]
        if stale:
            return self._proof(
                attempt_id, lease_fence, complete=False,
                live_resources=tuple(sorted(r["kind"] for r in stale)),
                detail="resources belong to a stale fence of the same attempt",
            )
        bound = self.binding()
        try:
            markers = self.ledger.attempt_running(attempt_id, lease_fence)
        except LedgerError as exc:
            return self._proof(
                attempt_id, lease_fence, complete=False,
                detail=f"ledger unreadable: {exc}",
            )
        foreign_markers = [
            marker for marker in markers
            if marker["binding"] is None or not all(
                marker["binding"].get(key) == bound.get(key)
                for key in (
                    "device_group_id", "vendor_id", "profile_digest",
                    "topology_digest", "resource_generation",
                )
            )
        ]
        if foreign_markers:
            return self._proof(
                attempt_id, lease_fence, complete=False, live_resources=("running",),
                detail="the attempt is still running under another device binding",
            )
        if not resources and not markers:
            return self._proof(
                attempt_id, lease_fence, complete=True,
                detail="no live resources or running marker for this attempt/fence",
            )
        foreign = [
            resource for resource in resources
            if not resource_matches_binding(resource, bound)
        ]
        if foreign:
            return self._proof(
                attempt_id, lease_fence, complete=False,
                live_resources=tuple(sorted(r["kind"] for r in foreign)),
                detail="resources belong to another device binding; not released",
            )
        if not self.release_supported:
            return self._proof(
                attempt_id, lease_fence, complete=False,
                live_resources=tuple(sorted(r["kind"] for r in resources)) or ("running",),
                detail="fake controller cannot prove external release",
            )
        released = self.ledger.release_attempt(attempt_id, lease_fence, binding=bound)
        remaining = self.ledger.attempt_resources(attempt_id, lease_fence)
        still_running = self.ledger.attempt_running(attempt_id, lease_fence)
        if remaining or still_running:
            return self._proof(
                attempt_id, lease_fence, complete=False,
                live_resources=(
                    tuple(sorted(r["kind"] for r in remaining)) if remaining else ("running",)
                ),
                released_resources=tuple(released),
                detail="external release left live resources or a running marker",
            )
        return self._proof(
            attempt_id, lease_fence, complete=True, released_resources=tuple(released),
            detail="external release verified against the attempt/fence ledger",
        )

    def health_check(self, profile: Mapping[str, Any]) -> HealthProof:
        if profile.get("device_group_id") != DEVICE_GROUP_ID:
            return HealthProof(False, DEVICE_GROUP_ID, self.generation, "wrong device group")
        if self._busy():
            return HealthProof(
                False, DEVICE_GROUP_ID, self.generation,
                "the device group is executing an attempt; the known-good probe is not run",
                executed_known_good=False,
            )
        self.ledger.record_probe(vendor_version="fake-c-1", resource_generation=self.generation)
        return HealthProof(
            ok=self.healthy, device_group_id=DEVICE_GROUP_ID,
            resource_generation=self.generation,
            detail="fake known-good prebuilt module" if self.healthy else "known-good load failed",
            executed_known_good=True,
        )


def received_build_evidence(
    *, bundle: ArtifactBundle, provenance: Mapping[str, Any],
    producer: Mapping[str, Any], duration_ms: int, log: bytes,
) -> BuildEvidence:
    """The receiver's ``reused`` evidence for a control-host-built artifact set.

    It repeats the producer's build-evidence link unchanged (so the original
    status/duration/log digest close over the signed result) and records this
    attempt's own verify/load duration and log.  It never claims a compile.
    """

    profile_document = build_provenance_document(provenance)
    profile = BuildProfile(
        build_profile=profile_document["build_profile"],
        builder_id=profile_document["builder_id"],
        builder_version=profile_document["builder_version"],
        build_host=profile_document["build_host"],
        toolchain_id=profile_document["toolchain"]["id"],
        toolchain_version=profile_document["toolchain"]["version"],
        sdk_version=profile_document["toolchain"]["sdk_version"],
        compile_profile=profile_document["compile_profile"],
    )
    return BuildEvidence(
        mode="prebuilt",
        status="reused",
        provenance=BuildProvenance(
            profile=profile, source_sha256=profile_document["source_sha256"],
            artifact_set_digest=profile_document["artifact_set_digest"],
        ),
        artifacts=tuple(BuildArtifact.from_entry(entry) for entry in bundle.entries),
        duration_ms=max(0, int(duration_ms)),
        log=bytes(log),
        placement="received",
        producer=producer,
    )


__all__ = [
    "FakeModuleVendorController",
    "FakeModuleVendorDriver",
    "NPU_BUILD_EVIDENCE_SCHEMA_ID",
    "NPU_BUILD_PROVENANCE_SCHEMA_ID",
    "VENDOR_C_GROUP",
    "build_evaluator_execution",
    "build_module_execution",
    "received_build_evidence",
]
