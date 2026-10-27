"""Deterministic fake vendor B: graph bundle -> asynchronous service, GRAPH.

Test-only.  Vendor B's shape differs from vendor A in every dimension that
matters to the abstraction:

* the artifact is a versioned *artifact set* (graph definition + parameter
  table), not one source module;
* the launch kind is ``GRAPH`` with the ``SERVICE`` subtype: there is no ELF,
  stream, rank or die, and synchronization is a completion token rather than a
  stream event;
* submission returns before the device finishes, so the host launch time and the
  device elapsed time are two different observations;
* the device-side resources are service-side jobs/sessions, so cancellation and
  cleanup must go through the worker-external controller.

Neither the driver nor the controller is imported by a production composition
root.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tcenv.remote.artifact_bundle import ArtifactBundle, BundleEntry
from tcenv.remote.contracts import (
    BUNDLE_DIGEST_ALGORITHM,
    NPU_BUNDLE_TIMING_EVIDENCE_SCHEMA_ID,
    npu_bundle_execution_document,
)
from tcenv.remote.drivers.base import (
    CleanupProof,
    HealthProof,
    PreparedAttempt,
    VendorFacts,
    VendorOutcome,
)
from tcenv.remote.job_types import DeviceGroupBinding, NpuExecutionSpec, NpuResources

from .graph_service import GraphService, ServiceError, matches_binding

VENDOR_ID = "vendor-b"
DEVICE_GROUP_ID = "vendor-b-group-0"
RESOURCE_GENERATION = "1"
CLOCK_DOMAIN = "vendor-b-service-clock-v1"
SERVICE_JOB_ENV = "TCENV_VENDOR_B_SERVICE"
SERVICE_FAULT_ENV = "TCENV_VENDOR_B_FAULT"
ABORT_AFTER_SUBMIT_ENV = "TCENV_VENDOR_B_ABORT_AFTER_SUBMIT"
SLOW_HOST_ENV = "TCENV_VENDOR_B_SLOW_HOST"
SKIP_CLOSE_ENV = "TCENV_VENDOR_B_SKIP_CLOSE"
DROP_CLEANUP_ENV = "TCENV_VENDOR_B_UNPROVEN_CLEANUP"

GRAPH_MEDIA_TYPE = "application/json"
PARAM_MEDIA_TYPE = "application/octet-stream"
#: A binary parameter table with NULs: a bundle entry is opaque bytes, so no
#: text encoding or newline normalization may be involved.
PARAMETER_BLOB = bytes(range(256)) * 4
GRAPH_BLOB = b'{"schema":"fake-graph-v1","nodes":[{"op":"matmul"},{"op":"relu"}]}\n'

VENDOR_B_GRAPH_ENTRIES = (
    ("graph/main.json", "GRAPH_DEFINITION", GRAPH_MEDIA_TYPE, GRAPH_BLOB),
    ("graph/params.bin", "PARAMETER_TABLE", PARAM_MEDIA_TYPE, PARAMETER_BLOB),
)


def _digest(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


_PROFILE = {
    "vendor_id": VENDOR_ID,
    "device_group_id": DEVICE_GROUP_ID,
    "service_version": "fake-b-service-1",
    "runtime_version": "fake-b-runtime-1",
    "logical_devices": 1,
    "ranks": 0,
    "dies": 0,
    "timing_sources": ["DEVICE_EVENT"],
    "execution_model": "graph-service",
}
_TOPOLOGY = {
    "device_group_id": DEVICE_GROUP_ID,
    "logical_devices": 1,
    "rank": None,
    "die": None,
    "fabric": "service",
}
PROFILE_DIGEST = _digest(_PROFILE)
TOPOLOGY_DIGEST = _digest(_TOPOLOGY)

VENDOR_B_GROUP = DeviceGroupBinding(
    device_group_id=DEVICE_GROUP_ID,
    vendor_id=VENDOR_ID,
    profile_digest=PROFILE_DIGEST,
    topology_digest=TOPOLOGY_DIGEST,
    resource_generation=RESOURCE_GENERATION,
)


def build_graph_execution(
    generation: str = RESOURCE_GENERATION, *, resources: NpuResources | None = None,
) -> NpuExecutionSpec:
    """The version 4 graph execution binding served by fake vendor B."""

    group = (
        VENDOR_B_GROUP if generation == RESOURCE_GENERATION
        else DeviceGroupBinding(
            device_group_id=VENDOR_B_GROUP.device_group_id,
            vendor_id=VENDOR_B_GROUP.vendor_id,
            profile_digest=VENDOR_B_GROUP.profile_digest,
            topology_digest=VENDOR_B_GROUP.topology_digest,
            resource_generation=generation,
        )
    )
    return NpuExecutionSpec(
        device_group=group,
        artifact_kind="bundle",
        launch_kind="GRAPH",
        launch_subtype="SERVICE",
        source_media_type=PARAM_MEDIA_TYPE,
        synchronization="service-completion-token",
        spec_version=4,
        task_family="graph",
        resources=resources or NpuResources(),
    )


def build_graph_bundle() -> ArtifactBundle:
    """The deterministic small artifact set this fake vendor executes."""

    entries = [
        BundleEntry(
            path=path, role=role, media_type=media_type, size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(), data=data,
        )
        for path, role, media_type, data in VENDOR_B_GRAPH_ENTRIES
    ]
    return ArtifactBundle(entries=tuple(entries))


def graph_sha256(bundle: ArtifactBundle) -> str:
    """The digest the service records for the graph it really executed."""

    parts = bytearray()
    for entry in bundle.entries:
        parts += f"{entry.path}\0{entry.sha256}\0{entry.size_bytes}\n".encode("utf-8")
    return hashlib.sha256(bytes(parts)).hexdigest()


def service_path() -> Path:
    value = os.environ.get(SERVICE_JOB_ENV) or os.environ.get("TCENV_VENDOR_LEDGER")
    if not value:
        raise RuntimeError(f"{SERVICE_JOB_ENV} is required for the fake graph service vendor")
    return Path(value)


@dataclass(frozen=True)
class _Submission:
    service_job_id: str
    completion_token: str


class FakeVendorBDriver:
    """Worker-local cooperative driver for an asynchronous graph service."""

    vendor_id = VENDOR_ID

    def __init__(self, service: GraphService | None = None, *, generation: str | None = None) -> None:
        self.service = service or GraphService(service_path())
        self.generation = generation or os.environ.get("TCENV_VENDOR_B_GENERATION") or RESOURCE_GENERATION
        self.fault = os.environ.get(SERVICE_FAULT_ENV) or ""

    def binding(self) -> dict[str, str]:
        return {**VENDOR_B_GROUP.to_dict(), "resource_generation": self.generation}

    def prepare(
        self, spec: Any, artifact: Any, attempt_context: Mapping[str, Any],
    ) -> PreparedAttempt:
        """Submit the graph job; the device has not finished when this returns."""

        if not isinstance(spec, NpuExecutionSpec) or spec.version != 4:
            raise ValueError("vendor B requires the version 4 execution spec")
        normalized = npu_bundle_execution_document(spec.to_dict())
        if normalized["task_family"] != "graph" or normalized["launch_kind"] != "GRAPH":
            raise ValueError("vendor B serves only the GRAPH task family")
        if normalized["device_group"] != self.binding():
            raise ValueError("vendor B received a different device group/profile/topology/generation")
        if not isinstance(artifact, ArtifactBundle):
            raise ValueError("vendor B requires a typed artifact bundle")
        attempt_id = attempt_context.get("attempt_id")
        lease_fence = attempt_context.get("lease_fence")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("vendor B requires an attempt id")
        if isinstance(lease_fence, bool) or not isinstance(lease_fence, int) or lease_fence < 1:
            raise ValueError("vendor B requires a positive lease fence")
        candidate = attempt_context.get("candidate_digest")
        if not isinstance(candidate, Mapping) or candidate.get("algorithm") != BUNDLE_DIGEST_ALGORITHM:
            raise ValueError("vendor B requires the accepted artifact-set CandidateDigest")
        if candidate.get("value") != artifact.artifact_set_digest:
            raise ValueError("vendor B candidate digest does not match the artifact set")
        roles = artifact.roles()
        if "GRAPH_DEFINITION" not in roles or "PARAMETER_TABLE" not in roles:
            raise ValueError("vendor B requires a graph definition and a parameter table")
        # Materialize and re-verify the exact bytes in this attempt's scope; the
        # instance never sees a caller-supplied path.
        workdir = Path(os.environ.get("TCENV_REMOTE_STATE_DIR", ".")) / "vendor-b" / attempt_id
        written = artifact.materialize(workdir)
        graph_digest = graph_sha256(artifact)
        for path, target in sorted(written.items()):
            if not target.is_file():
                raise ValueError(f"vendor B could not materialize {path}")
        device_elapsed_ns = 900_000 + int(graph_digest[:6], 16) % 400_000
        submission = _Submission(**self._submit(
            attempt_id, lease_fence, graph_digest, device_elapsed_ns,
        ))
        return PreparedAttempt(
            attempt_id=attempt_id,
            lease_fence=lease_fence,
            device_group_id=self.binding()["device_group_id"],
            resource_generation=self.generation,
            module_format="GRAPH/SERVICE",
            module_sha256=graph_digest,
            handles={
                "service_job_id": submission.service_job_id,
                "completion_token": submission.completion_token,
                "graph_sha256": graph_digest,
                "workdir": str(workdir),
                "candidate_digest": {
                    "algorithm": str(candidate["algorithm"]), "value": str(candidate["value"]),
                },
                "device_elapsed_ns": str(device_elapsed_ns),
            },
        )

    def _submit(
        self, attempt_id: str, lease_fence: int, graph_digest: str, device_elapsed_ns: int,
    ) -> dict[str, Any]:
        return self.service.submit(
            attempt_id=attempt_id, lease_fence=lease_fence, binding=self.binding(),
            graph_sha256=graph_digest, device_elapsed_ns=device_elapsed_ns, fault=self.fault,
            resources={"graph": graph_digest, "service_session": f"{attempt_id}:session"},
        )

    def execute(
        self, prepared: PreparedAttempt, *, deadline_monotonic: float, progress: Any = None,
    ) -> VendorOutcome:
        """Poll the service until the completion token is observed.

        The host launch time is the wall-clock cost of this observation loop; the
        device time comes only from the completed submission.
        """

        if not isinstance(prepared, PreparedAttempt):
            raise ValueError("vendor B requires a prepared attempt")
        started = time.monotonic_ns()
        service_job_id = prepared.handles["service_job_id"]
        token = prepared.handles["completion_token"]
        if progress is not None:
            progress({
                "stage": "queued", "completed_reps": 0, "total_reps": 1,
                "detail": "graph job accepted by the service",
            })
        if os.environ.get(ABORT_AFTER_SUBMIT_ENV) == "1" and prepared.lease_fence == 1:
            # A worker killed while the device is running: the service keeps the
            # submission and the live marker, and only the worker-external
            # controller can prove (or fail to prove) cleanup.
            os._exit(9)
        observed: dict[str, Any] = {}
        for _ in range(64):
            observed = self.service.poll(service_job_id, completion_token=token)
            if observed["state"] in {"done", "cancelled"}:
                break
            if time.monotonic() >= deadline_monotonic:
                break
            time.sleep(0.002)
        if observed.get("state") != "done":
            # The device never signsalled completion.  The host does not invent a
            # device result: it keeps waiting past the durable deadline so the
            # attempt is terminated as a timeout with no device observation.
            while time.monotonic() < deadline_monotonic + 0.4:
                time.sleep(0.01)
            raise TimeoutError("fake vendor B graph service did not complete before the deadline")
        tokens = observed.get("tokens") or []
        if len(tokens) != 1:
            raise ValueError(
                f"vendor B requires exactly one completion token, observed {len(tokens)}"
            )
        if tokens[0] != token:
            raise ValueError("vendor B observed a completion token this attempt never received")
        if os.environ.get(SLOW_HOST_ENV) == "1":
            # A slow host is not a slow device: only host_api_elapsed may change.
            time.sleep(0.05)
        host_api_elapsed_ns = max(1, time.monotonic_ns() - started)
        device_elapsed_ns = int(observed["device_elapsed_ns"])
        declared = prepared.handles["device_elapsed_ns"]
        if int(declared) != device_elapsed_ns:
            raise ValueError("vendor B device elapsed time changed after submission")
        evidence = self._evidence(prepared, device_elapsed_ns, host_api_elapsed_ns, token)
        return VendorOutcome(
            report={}, evidence=evidence, module_format="GRAPH/SERVICE",
            module_sha256=prepared.module_sha256,
        )

    def _evidence(
        self, prepared: PreparedAttempt, device_elapsed_ns: int, host_api_elapsed_ns: int,
        token: str,
    ) -> dict[str, Any]:
        degradation = os.environ.get("TCENV_VENDOR_B_DEGRADE") or ""
        spans = [{
            "device_id": "device-0",
            # No rank and no die: this vendor shape has neither, so the fields are
            # absent rather than invented.
            "rank": None,
            "die": None,
            "clock_domain": CLOCK_DOMAIN,
            "elapsed_ns": device_elapsed_ns,
        }]
        buffer = None
        complete = True
        if degradation == "drop_span":
            # Claims completeness while the declared device span is missing: the
            # frontend must refuse to turn this into an ok/publishable result.
            spans = []
        elif degradation == "buffer_drop":
            buffer = {
                "flushed": True, "records_written": 4, "records_delivered": 4,
                "dropped": 1, "overwritten": 0,
            }
        elif degradation == "clock_unknown":
            spans = []
            complete = False
        timing = {
            "source": "DEVICE_EVENT",
            "clock_domain": CLOCK_DOMAIN if complete else "unobserved",
            "synchronization_completed": complete,
            "device_elapsed_ns": device_elapsed_ns if complete else 0,
            "host_api_elapsed_ns": host_api_elapsed_ns,
            "launches": 1 if complete else 0,
        }
        return {
            "schema_id": NPU_BUNDLE_TIMING_EVIDENCE_SCHEMA_ID,
            "candidate_digest": dict(prepared.handles["candidate_digest"]),
            "artifact_set_digest": prepared.handles["candidate_digest"]["value"],
            "device_group": self.binding(),
            "topology": {
                "logical_devices": 1,
                "declared_devices": ["device-0"],
                "required_rank_spans": None,
                "required_die_spans": None,
            },
            "spans": spans,
            "timing": timing,
            "buffer": buffer,
            "measurement_kind": "simulated",
            "publication_allowed": False,
            "complete": complete,
            "diagnostics": {
                "vendor_id": VENDOR_ID,
                "completion_token": token,
                "launch_kind": "GRAPH",
                "graph_sha256": prepared.module_sha256,
                "service_job_id": prepared.handles["service_job_id"],
                "degradation": degradation or None,
            },
        }

    def close(self, prepared: PreparedAttempt) -> None:
        if os.environ.get(SKIP_CLOSE_ENV) == "1":
            return
        try:
            self.service.collect(
                prepared.attempt_id, prepared.lease_fence, binding=self.binding(),
            )
        except ServiceError:
            # A service that cannot be reached is not a proven cleanup; the
            # worker-external controller has to decide.
            return


class FakeVendorBController:
    """Worker-external controller for the asynchronous graph service."""

    vendor_id = VENDOR_ID

    def __init__(
        self, service_path: Path | str, *, release_supported: bool = True,
        healthy: bool = True, generation: str = RESOURCE_GENERATION,
    ) -> None:
        self.service = GraphService(service_path)
        self.release_supported = bool(release_supported)
        self.healthy = bool(healthy)
        self.generation = generation

    def binding(self) -> dict[str, str]:
        return {**VENDOR_B_GROUP.to_dict(), "resource_generation": self.generation}

    def _facts(self, state: str) -> VendorFacts:
        return VendorFacts(
            vendor_id=VENDOR_ID, device_group_id=DEVICE_GROUP_ID,
            profile_digest=PROFILE_DIGEST, topology_digest=TOPOLOGY_DIGEST,
            resource_generation=self.generation, logical_devices=1,
            ranks=0, dies=0,
            runtime_version="fake-b-runtime-1",
            timing_sources=("DEVICE_EVENT",),
            state=state,
            diagnostics={"simulated": True, "execution_model": "graph-service"},
        )

    def _busy(self) -> bool:
        if not self.service.exists():
            return False
        try:
            return bool(self.service.live_markers())
        except ServiceError:
            return True

    def probe(self, profile: Mapping[str, Any]) -> VendorFacts:
        if profile.get("device_group_id") != DEVICE_GROUP_ID:
            raise ValueError("fake vendor B probe received another device group")
        if self._busy():
            return self._facts("busy")
        self.service.record_probe(
            service_version="fake-b-1", resource_generation=self.generation,
        )
        return self._facts("available" if self.healthy else "quarantined")

    def capabilities(self, facts: VendorFacts) -> Mapping[str, Any]:
        if facts.device_group_id != DEVICE_GROUP_ID:
            raise ValueError("fake vendor B capability group mismatch")
        return {
            "artifact_kinds": ["bundle"],
            "launch_kinds": ["GRAPH"],
            "launch_subtypes": ["SERVICE"],
            "timing_sources": ["DEVICE_EVENT"],
            "device_groups": [{
                **self.binding(),
                "logical_devices": 1,
                "state": "available" if self.healthy else "quarantined",
                "timing_sources": ["DEVICE_EVENT"],
            }],
        }

    def cancel(self, attempt_id: str, lease_fence: int) -> None:
        self.service.cancel(attempt_id, lease_fence)

    def _proof(self, attempt_id: str, lease_fence: int, **fields: Any) -> CleanupProof:
        group = VENDOR_B_GROUP
        return CleanupProof(
            attempt_id=attempt_id, lease_fence=lease_fence,
            device_group_id=group.device_group_id,
            resource_generation=self.generation,
            vendor_id=group.vendor_id, profile_digest=group.profile_digest,
            topology_digest=group.topology_digest, **fields,
        )

    def reconcile(self, attempt_id: str, lease_fence: int) -> CleanupProof:
        if not self.service.exists():
            return self._proof(
                attempt_id, lease_fence, complete=False,
                detail="the fake graph service state is missing; cleanup cannot be proven",
            )
        bound = self.binding()
        try:
            live = self.service.attempt_live(attempt_id, lease_fence)
        except ServiceError as exc:
            return self._proof(
                attempt_id, lease_fence, complete=False, detail=f"service unreadable: {exc}",
            )
        foreign = [job for job in live if not matches_binding(job["binding"], bound)]
        if foreign:
            return self._proof(
                attempt_id, lease_fence, complete=False,
                live_resources=tuple(sorted(job["service_job_id"] for job in foreign)),
                detail=(
                    "the submission belongs to another device group/profile/topology/"
                    "resource generation; this controller cannot release it"
                ),
            )
        if not live:
            markers = [
                marker for marker in self.service.live_markers()
                if marker["marker"] == f"{attempt_id}:{lease_fence}"
            ]
            if markers:
                return self._proof(
                    attempt_id, lease_fence, complete=False,
                    live_resources=("running",),
                    detail="the service still holds a live marker for this attempt/fence",
                )
            return self._proof(
                attempt_id, lease_fence, complete=True,
                detail="no service submission or live marker for this attempt/fence",
            )
        if not self.release_supported:
            return self._proof(
                attempt_id, lease_fence, complete=False,
                live_resources=tuple(sorted(job["service_job_id"] for job in live)),
                detail="fake controller cannot prove external release",
            )
        try:
            collected = self.service.collect(attempt_id, lease_fence, binding=bound)
        except ServiceError as exc:
            return self._proof(
                attempt_id, lease_fence, complete=False,
                detail=f"service collection failed: {exc}",
            )
        if collected["refused"]:
            return self._proof(
                attempt_id, lease_fence, complete=False,
                live_resources=tuple(sorted(collected["refused"])),
                detail="the service refused to release the submission",
            )
        remaining = self.service.attempt_live(attempt_id, lease_fence)
        markers = [
            marker for marker in self.service.live_markers()
            if marker["marker"] == f"{attempt_id}:{lease_fence}"
        ]
        if remaining or markers:
            return self._proof(
                attempt_id, lease_fence, complete=False,
                live_resources=tuple(sorted(job["service_job_id"] for job in remaining)) or ("running",),
                released_resources=tuple(collected["collected"]),
                detail="external release left a live submission or marker",
            )
        return self._proof(
            attempt_id, lease_fence, complete=True,
            released_resources=tuple(collected["collected"]),
            detail="external release verified against the service state",
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
        self.service.record_probe(
            service_version="fake-b-1", resource_generation=self.generation,
        )
        return HealthProof(
            ok=self.healthy, device_group_id=DEVICE_GROUP_ID,
            resource_generation=self.generation,
            detail="fake graph-service known-good probe" if self.healthy
            else "fake graph-service known-good probe failed",
            executed_known_good=True,
        )
