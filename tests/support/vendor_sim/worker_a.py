"""Private worker entry for fake vendor A (P0 test-only)."""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

from tcenv.remote import worker_protocol
from tcenv.remote.blob_channel import BlobReader
from tcenv.schema import PROTOCOL_VERSION

from .ledger import VendorLedger
from .vendor_a import FakeVendorADriver

VENDOR_LEDGER_ENV = "TCENV_VENDOR_LEDGER"
#: Engine-owned blob root (never an agent-supplied path).
BLOB_ROOT_ENV = "TCENV_BLOB_ROOT"


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
    if evidence.get("candidate_digest") != request.npu.candidate_digest.to_dict():
        return 14
    device_elapsed_ns = int(evidence["timing"]["device_elapsed_ns"])
    report = {
        "task": request.task,
        "dtype": request.dtype,
        "size": request.size,
        "mode": request.mode,
        "seed": request.seed_int,
        "status": "ok",
        "stage": "done",
        "candidate_us": device_elapsed_ns / 1000.0,
        "speedup": 1.0,
        "elapsed_s": 0.0,
        "target_id": request.target_id,
        "protocol_version": PROTOCOL_VERSION,
        "simulated": True,
    }
    worker_protocol.write_result(
        sys.stdout.buffer, request_id=request_id, attempt_id=attempt_id,
        lease_fence=fence, report=report, evidence=evidence,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
