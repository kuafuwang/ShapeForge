"""Private worker entry for fake vendor A (P0 test-only)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from tcenv.remote import worker_protocol
from tcenv.schema import PROTOCOL_VERSION

from .ledger import VendorLedger
from .vendor_a import FakeVendorADriver

VENDOR_LEDGER_ENV = "TCENV_VENDOR_LEDGER"


def main() -> int:
    attempt_id, request, request_id, fence = worker_protocol.read_request(sys.stdin.buffer)
    if request.npu is None:
        return 12
    ledger_path = os.environ.get(VENDOR_LEDGER_ENV)
    if not ledger_path:
        return 13
    ledger = VendorLedger(Path(ledger_path))
    driver = FakeVendorADriver(ledger)
    context = {
        "attempt_id": attempt_id,
        "lease_fence": fence,
        "device_group": request.npu.execution.device_group.to_dict(),
    }
    prepared = driver.prepare(request.npu.execution, request.source, context)

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
