"""Private worker entry for fake vendor A (P0 test-only)."""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

from tcenv.remote import worker_protocol
from tcenv.remote.blob_channel import BlobReader
from tcenv.remote.contracts import NPU_BUNDLE_TIMING_EVIDENCE_SCHEMA_ID
from tcenv.schema import PROTOCOL_VERSION

from .ledger import VendorLedger
from .vendor_a import FakeVendorADriver

VENDOR_LEDGER_ENV = "TCENV_VENDOR_LEDGER"
#: Engine-owned blob root (never an agent-supplied path).
BLOB_ROOT_ENV = "TCENV_BLOB_ROOT"


def env_degrade() -> str:
    return os.environ.get("TCENV_VENDOR_A_DEGRADE") or ""


def _read_blob(descriptor: dict, root: str) -> tuple[bytes, int, int]:
    """Read the artifact chunk by chunk and verify it against the descriptor.

    The fake driver needs the whole (small, simulated) candidate, but the read
    path is the channel's: it never loads more than one chunk into the read
    buffer, and the caller records the maximum chunk so a whole-file read would
    be visible.  A real large-artifact worker would stream into the vendor
    driver instead; this simulation does not claim that.
    """

    reader = BlobReader(Path(root))
    collected = bytearray()
    total = 0
    chunks = 0
    max_chunk = 0
    for chunk in reader.iter_chunks(descriptor["content_id"]):
        # ``iter_chunks`` yields one chunk at a time and verifies the whole
        # digest before the last yield; the read granularity is recorded so a
        # whole-file read would be visible in the ledger.
        chunks += 1
        total += len(chunk)
        max_chunk = max(max_chunk, len(chunk))
        collected += chunk
        if len(chunk) > descriptor["chunk_size"]:
            raise RuntimeError("worker read more than one declared chunk")
    data = bytes(collected)
    if total != descriptor["size_bytes"] or hashlib.sha256(data).hexdigest() != descriptor["sha256"]:
        raise RuntimeError("worker blob does not match its descriptor")
    return data, chunks, max_chunk


def main() -> int:
    attempt_id, request, request_id, fence = worker_protocol.read_request(sys.stdin.buffer)
    if request.npu is None:
        return 12
    ledger_path = os.environ.get(VENDOR_LEDGER_ENV)
    if not ledger_path:
        return 13
    ledger = VendorLedger(Path(ledger_path))
    driver = FakeVendorADriver(ledger)
    binding = {**driver.binding()}
    source = request.source
    if request.npu.blob is not None:
        root = os.environ.get(BLOB_ROOT_ENV)
        if not root:
            return 15
        try:
            source, chunks, max_chunk = _read_blob(request.npu.blob.to_dict(), root)
        except Exception:  # noqa: BLE001 - a missing/corrupt blob is an infra failure
            return 17
        ledger.record_blob_read(
            attempt_id=attempt_id, lease_fence=fence,
            content_id=request.npu.blob.content_id, chunks=chunks,
            size_bytes=len(source), digest=hashlib.sha256(source).hexdigest(),
            max_chunk_bytes=max_chunk,
            binding=binding,
        )
    context = {
        "attempt_id": attempt_id,
        "lease_fence": fence,
        "device_group": request.npu.execution.device_group.to_dict(),
        "candidate_digest": request.npu.candidate_digest.to_dict(),
    }
    prepared = driver.prepare(request.npu.execution, source, context)

    def progress(payload) -> None:
        worker_protocol.write_progress(
            sys.stdout.buffer, request_id=request_id, attempt_id=attempt_id,
            lease_fence=fence, progress=payload,
        )

    outcome = None
    try:
        outcome = driver.execute(
            prepared,
            deadline_monotonic=time.monotonic() + request.candidate_timeout_ms / 1000.0,
            progress=progress,
        )
    finally:
        driver.close(prepared)
    evidence = dict(outcome.evidence)
    if request.npu.execution.version == 4:
        # The bundle contract signs TimingEvidence version 2: the same host/device
        # split plus the declared topology and the per-device spans.
        timing = dict(evidence["timing"])
        declared = request.npu.execution.declared_resources
        evidence = {
            "schema_id": NPU_BUNDLE_TIMING_EVIDENCE_SCHEMA_ID,
            "candidate_digest": dict(evidence["candidate_digest"]),
            "artifact_set_digest": None,
            "device_group": dict(evidence["device_group"]),
            "topology": {
                "logical_devices": declared.logical_devices,
                "declared_devices": [f"device-{index}" for index in range(declared.logical_devices)],
                "required_rank_spans": declared.required_rank_spans,
                "required_die_spans": declared.required_die_spans,
            },
            "spans": [{
                "device_id": "device-0",
                "rank": None,
                "die": None,
                "clock_domain": timing["clock_domain"],
                "elapsed_ns": int(timing["device_elapsed_ns"]),
            }],
            "timing": timing,
            "buffer": None,
            "measurement_kind": evidence["measurement_kind"],
            "publication_allowed": evidence["publication_allowed"],
            "complete": evidence["complete"],
            "diagnostics": dict(evidence.get("diagnostics") or {}),
        }
        if env_degrade() == "drop_span":
            # Claims completeness while the only declared device span is gone.
            evidence["spans"] = []
        elif env_degrade() == "buffer_drop":
            evidence["buffer"] = {
                "flushed": True, "records_written": 2, "records_delivered": 2,
                "dropped": 1, "overwritten": 0,
            }
    if evidence.get("candidate_digest") != request.npu.candidate_digest.to_dict():
        return 14
    device_elapsed_ns = int(evidence["timing"]["device_elapsed_ns"])
    from tests.support.report_build import report_for_request

    report = report_for_request(
        request, status="ok", source=source,
    )
    report.update({
        "candidate_us": device_elapsed_ns / 1000.0,
        "speedup": 1.0,
        "elapsed_s": 0.0,
    })
    if os.environ.get("TCENV_VENDOR_A_BAD_REPORT") == "1":
        # Test-only fault: a worker result the frontend must reject.  It makes a
        # candidate fail deterministically (not just time out).
        report["task"] = "forged-task"
    worker_protocol.write_result(
        sys.stdout.buffer, request_id=request_id, attempt_id=attempt_id,
        lease_fence=fence, report=report, evidence=evidence,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
