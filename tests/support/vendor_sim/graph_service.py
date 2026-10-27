"""Cross-process fake *asynchronous graph service* for the P1 fake vendor B.

Vendor B has no ELF, no stream, no rank and no die.  Its device is reached
through a service: a submission is accepted immediately (``queued``), the
device runs it on its own timeline (``running``), and completion is only
observable by polling with the completion token the service issued.  This is a
different execution shape from fake vendor A's synchronous ELF/SIMT module, and
it is what makes the P1 evidence and cleanup paths non-trivial.

The service is deliberately test-only, file-backed (worker and
worker-external controller are separate processes) and stores no source bytes,
host paths, credentials or arbitrary vendor commands.

Faults are explicit and recorded before they fire, so a test can assert that the
fault really happened instead of trusting a message:

``lost_token``       completion is reported without any token
``duplicate_token``  completion is reported with two different tokens
``wrong_token``      completion is reported with a token that was never issued
``never_complete``   the submission never leaves ``running``
``keep_service_job`` collection refuses to remove the submission
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

BINDING_KEYS = (
    "device_group_id", "vendor_id", "profile_digest", "topology_digest", "resource_generation",
)
SERVICE_FAULTS = frozenset({
    "", "lost_token", "duplicate_token", "wrong_token", "never_complete", "keep_service_job",
})


class ServiceError(RuntimeError):
    """The fake graph service is missing, unreadable or refused a call."""


def completion_token(attempt_id: str, lease_fence: int, graph_sha256: str) -> str:
    """Deterministic token: one submission has exactly one completion identity."""

    return hashlib.sha256(
        f"vc-token\0{attempt_id}\0{lease_fence}\0{graph_sha256}".encode("utf-8")
    ).hexdigest()[:32]


def matches_binding(document: Mapping[str, Any], binding: Mapping[str, Any]) -> bool:
    return all(document.get(key) == binding.get(key) for key in BINDING_KEYS)


class GraphService:
    """File-backed asynchronous submit/poll service with durable state."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.path.is_file()

    @contextmanager
    def _locked(self, *, create: bool = True) -> Iterator[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if not create and not self.path.is_file():
                # Missing evidence is never proof that the service is idle.
                raise ServiceError("the fake graph service state is missing")
            document = self._read_unlocked()
            yield document
            temporary = self.path.with_name(self.path.name + ".tmp")
            temporary.write_text(
                json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8",
            )
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _read_unlocked(self) -> dict[str, Any]:
        empty = {
            "schema_version": 1, "submissions": {}, "live": [], "cancelled": [],
            "probes": [], "device_events": [], "blob_reads": [],
        }
        if not self.path.is_file():
            return empty
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ServiceError(f"the fake graph service state is unreadable: {exc}") from exc
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            raise ServiceError("the fake graph service state schema is unsupported")
        for key in ("submissions", "live", "cancelled", "probes", "device_events", "blob_reads"):
            if not isinstance(document.get(key), (dict, list)):
                raise ServiceError("the fake graph service state is corrupt")
        return document

    def read(self) -> dict[str, Any]:
        with self._locked(create=False) as document:
            return json.loads(json.dumps(document))

    # -- worker-side calls -------------------------------------------------
    def submit(
        self, *, attempt_id: str, lease_fence: int, binding: Mapping[str, Any],
        graph_sha256: str, device_elapsed_ns: int, fault: str = "",
        resources: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Accept a graph job and return immediately, before the device finishes.

        The submission is durable in the ``queued`` state before this call
        returns: the host launch time is therefore *not* the device time, and a
        worker killed right after this call leaves observable device state.
        """

        if fault not in SERVICE_FAULTS:
            raise ServiceError(f"unknown fake graph service fault {fault!r}")
        marker = f"{attempt_id}:{lease_fence}"
        job_id = "graph-job-" + hashlib.sha256(marker.encode("utf-8")).hexdigest()[:16]
        token = completion_token(attempt_id, lease_fence, graph_sha256)
        with self._locked() as document:
            if job_id in document["submissions"]:
                raise ServiceError("the fake graph service already holds this submission")
            document["submissions"][job_id] = {
                "service_job_id": job_id,
                "attempt_id": attempt_id,
                "lease_fence": lease_fence,
                "binding": {key: binding.get(key) for key in BINDING_KEYS},
                "graph_sha256": graph_sha256,
                "completion_token": token,
                "state": "queued",
                "history": ["queued"],
                "polls": 0,
                "device_elapsed_ns": device_elapsed_ns,
                "fault": fault,
                "launch_returned_before_completion": True,
                "collected": False,
                "resources": dict(resources or {}),
            }
            if marker not in document["live"]:
                document["live"].append(marker)
            document["device_events"].append({
                "event": "submitted", "service_job_id": job_id, "attempt_id": attempt_id,
                "lease_fence": lease_fence, "state": "queued",
            })
        return {"service_job_id": job_id, "completion_token": token}

    def poll(self, service_job_id: str, *, completion_token: str) -> dict[str, Any]:
        """Advance the device timeline one step and report what is observable."""

        with self._locked() as document:
            job = document["submissions"].get(service_job_id)
            if job is None:
                raise ServiceError("the fake graph service does not know this submission")
            if job["completion_token"] != completion_token:
                raise ServiceError("poll used a completion token that was not issued for this job")
            job["polls"] += 1
            fault = job.get("fault") or ""
            if job["state"] == "queued":
                job["state"] = "running"
                job["history"].append("running")
                document["device_events"].append({
                    "event": "running", "service_job_id": service_job_id,
                    "attempt_id": job["attempt_id"], "lease_fence": job["lease_fence"],
                    "state": "running",
                })
            elif job["state"] == "running" and fault != "never_complete":
                job["state"] = "done"
                job["history"].append("done")
                document["device_events"].append({
                    "event": "completed", "service_job_id": service_job_id,
                    "attempt_id": job["attempt_id"], "lease_fence": job["lease_fence"],
                    "state": "done",
                })
            observed = {
                "service_job_id": service_job_id,
                "state": job["state"],
                "device_elapsed_ns": job["device_elapsed_ns"],
                "fault": fault,
            }
            if job["state"] != "done":
                observed["tokens"] = []
                return observed
            # Record that the injected fault really fired, so a test can assert
            # the fault and its consequence instead of trusting an error string.
            if fault:
                document["device_events"].append({
                    "event": "fault_fired", "service_job_id": service_job_id,
                    "attempt_id": job["attempt_id"], "lease_fence": job["lease_fence"],
                    "fault": fault,
                })
            if fault == "lost_token":
                observed["tokens"] = []
            elif fault == "duplicate_token":
                observed["tokens"] = [job["completion_token"], "0" * 32]
            elif fault == "wrong_token":
                observed["tokens"] = ["f" * 32]
            else:
                observed["tokens"] = [job["completion_token"]]
            return observed

    # -- controller-side calls ---------------------------------------------
    def cancel(self, attempt_id: str, lease_fence: int) -> None:
        """Ask the service to stop this attempt's submissions."""

        marker = f"{attempt_id}:{lease_fence}"
        with self._locked(create=False) as document:
            if marker not in document["cancelled"]:
                document["cancelled"].append(marker)
            for job in document["submissions"].values():
                if job["attempt_id"] == attempt_id and job["lease_fence"] == lease_fence:
                    if job["state"] in {"queued", "running"}:
                        job["state"] = "cancelled"
                        job["history"].append("cancelled")
                        document["device_events"].append({
                            "event": "cancelled", "service_job_id": job["service_job_id"],
                            "attempt_id": attempt_id, "lease_fence": lease_fence,
                            "state": "cancelled",
                        })

    def attempt_live(self, attempt_id: str, lease_fence: int) -> list[dict[str, Any]]:
        with self._locked(create=False) as document:
            return [
                dict(job) for job in document["submissions"].values()
                if job["attempt_id"] == attempt_id and job["lease_fence"] == lease_fence
            ]

    def live_markers(self) -> list[dict[str, Any]]:
        with self._locked(create=False) as document:
            marks = []
            for marker in document["live"]:
                job = next(
                    (
                        job for job in document["submissions"].values()
                        if f"{job['attempt_id']}:{job['lease_fence']}" == marker
                    ),
                    None,
                )
                marks.append({
                    "marker": marker,
                    "binding": None if job is None else dict(job["binding"]),
                    "state": None if job is None else job["state"],
                })
            return marks

    def collect(
        self, attempt_id: str, lease_fence: int, *, binding: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Remove this attempt/fence's service rows, never another binding's."""

        marker = f"{attempt_id}:{lease_fence}"
        collected: list[str] = []
        refused: list[str] = []
        with self._locked(create=False) as document:
            for job_id, job in list(document["submissions"].items()):
                if job["attempt_id"] != attempt_id or job["lease_fence"] != lease_fence:
                    continue
                if not matches_binding(job["binding"], binding):
                    # Another device group/generation owns this row; this
                    # controller cannot release it and must not claim it did.
                    refused.append(job_id)
                    continue
                if job.get("fault") == "keep_service_job":
                    refused.append(job_id)
                    continue
                collected.append(job_id)
                del document["submissions"][job_id]
            if not refused:
                if marker in document["live"]:
                    document["live"].remove(marker)
        return {"collected": sorted(collected), "refused": sorted(refused)}

    def record_blob_read(
        self, *, attempt_id: str, lease_fence: int, content_id: str, digests: list[str],
        chunks: int, size_bytes: int, max_chunk_bytes: int,
    ) -> None:
        """Record how the worker consumed a blob-backed bundle entry.

        The entry list proves the read was chunked and every entry was verified
        against the manifest, so a whole-file read or a skipped digest would be
        visible in the test instead of trusted.
        """

        with self._locked() as document:
            document["blob_reads"].append({
                "attempt_id": attempt_id, "lease_fence": lease_fence,
                "content_id": content_id, "entry_digests": list(digests),
                "chunks": chunks, "size_bytes": size_bytes,
                "max_chunk_bytes": max_chunk_bytes,
            })

    def blob_reads(self) -> list[dict[str, Any]]:
        with self._locked(create=False) as document:
            return json.loads(json.dumps(document["blob_reads"]))

    def record_device_failure(self, *, attempt_id: str, lease_fence: int, reason: str) -> None:
        """Record a typed device-side failure the driver observed."""

        with self._locked() as document:
            document["device_events"].append({
                "event": "device_failed", "attempt_id": attempt_id,
                "lease_fence": lease_fence, "reason": reason,
            })

    def record_probe(self, *, service_version: str, resource_generation: str) -> None:
        with self._locked() as document:
            document["probes"].append({
                "service_version": service_version,
                "resource_generation": resource_generation,
            })

    def submissions(self) -> dict[str, dict[str, Any]]:
        """Every submission currently held by the service (test observability)."""

        with self._locked(create=False) as document:
            return json.loads(json.dumps(document["submissions"]))

    def probes(self) -> list[dict[str, Any]]:
        with self._locked(create=False) as document:
            return json.loads(json.dumps(document["probes"]))

    def device_events(self) -> list[dict[str, Any]]:
        with self._locked(create=False) as document:
            return json.loads(json.dumps(document["device_events"]))

    def history(self, service_job_id: str) -> list[str]:
        with self._locked(create=False) as document:
            job = document["submissions"].get(service_job_id)
            return [] if job is None else list(job["history"])

    def clear(self, *, operator: bool = False) -> None:
        """Erase the service state.  Test setup only, never a vendor action."""

        if operator is not True:
            raise TypeError("GraphService.clear() is a test-setup helper; pass operator=True")
        with self._locked() as document:
            document["submissions"] = {}
            document["live"] = []
            document["cancelled"] = []
            document["probes"] = []
            document["device_events"] = []
            document["blob_reads"] = []
