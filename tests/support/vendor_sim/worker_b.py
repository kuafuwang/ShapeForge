"""Private worker entry for fake vendor B (P1 test-only).

The worker receives the artifact set in the private frame, rebuilds the typed
bundle, re-verifies every inline entry against its declared size and digest,
hands it to the vendor driver and reports the asynchronous service result.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

from tcenv.remote.artifact_bundle import BundleError
from tcenv.remote import worker_protocol
from tcenv.remote.blob_channel import BlobReader
from tcenv.schema import PROTOCOL_VERSION

from .vendor_b import FakeVendorBDriver, GraphService, service_path

#: Engine-owned blob root (never an agent-supplied path).
BLOB_ROOT_ENV = "TCENV_BLOB_ROOT"


def _read_blob_entries(bundle, root: str) -> tuple[int, int]:
    """Read every blob-backed entry chunk by chunk and verify its exact digest.

    Mirrors the P0b worker read path: one chunk at a time, whole-digest check at
    the end, and the read granularity is recorded so a whole-file read would be
    visible in the service state.
    """

    reader = BlobReader(Path(root))
    chunks = 0
    max_chunk = 0
    for entry in bundle.blob_entries:
        collected = bytearray()
        for chunk in reader.iter_chunks(entry.blob_ref["content_id"]):
            chunks += 1
            max_chunk = max(max_chunk, len(chunk))
            collected += chunk
            if len(chunk) > entry.blob_ref["chunk_size"]:
                raise RuntimeError("worker read more than one declared chunk")
        data = bytes(collected)
        if len(data) != entry.size_bytes or hashlib.sha256(data).hexdigest() != entry.sha256:
            raise RuntimeError(f"blob-backed bundle entry {entry.path!r} does not match its manifest")
    return chunks, max_chunk


def main() -> int:
    attempt_id, request, request_id, fence = worker_protocol.read_request(sys.stdin.buffer)
    if request.npu is None or request.npu.bundle is None:
        return 12
    if request.source:
        return 13
    if request.npu.candidate_digest.value != request.npu.bundle.artifact_set_digest:
        return 16
    try:
        service = GraphService(service_path())
    except Exception:  # noqa: BLE001 - an unusable service is an infrastructure failure
        return 13
    driver = FakeVendorBDriver(service)
    if request.npu.bundle.blob_entries:
        root = os.environ.get(BLOB_ROOT_ENV)
        if not root:
            return 17
        try:
            chunks, max_chunk = _read_blob_entries(request.npu.bundle, root)
        except Exception:  # noqa: BLE001 - a missing/corrupt blob is an infrastructure failure
            return 17
        service.record_blob_read(
            attempt_id=attempt_id, lease_fence=fence,
            content_id="+".join(
                entry.blob_ref["content_id"] for entry in request.npu.bundle.blob_entries
            ),
            digests=[entry.sha256 for entry in request.npu.bundle.blob_entries],
            chunks=chunks, size_bytes=sum(
                entry.size_bytes for entry in request.npu.bundle.blob_entries
            ),
            max_chunk_bytes=max_chunk,
        )
    context = {
        "attempt_id": attempt_id,
        "lease_fence": fence,
        "device_group": request.npu.execution.device_group.to_dict(),
        "candidate_digest": request.npu.candidate_digest.to_dict(),
    }
    try:
        prepared = driver.prepare(request.npu.execution, request.npu.bundle, context)
    except (ValueError, BundleError):
        return 14

    def progress(payload) -> None:
        worker_protocol.write_progress(
            sys.stdout.buffer, request_id=request_id, attempt_id=attempt_id,
            lease_fence=fence, progress=payload,
        )

    try:
        outcome = driver.execute(
            prepared,
            deadline_monotonic=time.monotonic() + request.candidate_timeout_ms / 1000.0,
            progress=progress,
        )
    except Exception as exc:  # noqa: BLE001 - a typed device failure is an infrastructure outcome
        # A lost/duplicated/wrong completion token, or a wait past the deadline,
        # is an infrastructure failure: the session is released and no result is
        # reported, so the frontend can never turn it into an ok candidate.
        service.record_device_failure(
            attempt_id=attempt_id, lease_fence=fence, reason=f"{type(exc).__name__}: {exc}"[:300],
        )
        driver.close(prepared)
        return 21
    driver.close(prepared)
    evidence = dict(outcome.evidence)
    if evidence.get("candidate_digest") != request.npu.candidate_digest.to_dict():
        return 14
    timing = evidence.get("timing") or {}
    if timing.get("source") != "DEVICE_EVENT":
        return 15
    device_elapsed_ns = int(timing["device_elapsed_ns"])
    from tests.support.report_build import report_for_request

    # The bundle's SOURCE entry is the candidate bytes; an inline entry carries them.
    source_entry = next(
        (entry for entry in request.npu.bundle.entries if entry.role == "SOURCE"), None,
    )
    source_bytes = (
        bytes(source_entry.data)
        if source_entry is not None and source_entry.data is not None
        else b"kernel void demo() {}\n"
    )
    report = report_for_request(request, status="ok", source=source_bytes)
    report.update({
        # The candidate time is the *device* completion time reported by the
        # service, never the host poll time.
        "candidate_us": device_elapsed_ns / 1000.0,
        "speedup": 1.0,
        "elapsed_s": 0.0,
    })
    # A test-only fault: the report claims a device time the service never
    # produced, which the frontend must reject instead of publishing.
    if (evidence.get("diagnostics") or {}).get("degradation") == "wrong_report_time":
        report["candidate_us"] = device_elapsed_ns / 1000.0 + 1_000.0
    if (evidence.get("diagnostics") or {}).get("degradation") == "host_time_report":
        report["candidate_us"] = int(timing["host_api_elapsed_ns"]) / 1000.0
    worker_protocol.write_result(
        sys.stdout.buffer, request_id=request_id, attempt_id=attempt_id,
        lease_fence=fence, report=report, evidence=evidence,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
