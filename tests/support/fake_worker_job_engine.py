"""CPU-only subprocess worker for durable job-engine regression tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tcenv.remote import worker_protocol


def _append(path: str, document: dict) -> None:
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(document, sort_keys=True) + "\n")


def main() -> int:
    started = os.environ.get("FAKE_JOB_STARTED")
    if started:
        Path(started).write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    attempt_id, request, request_id, fence = worker_protocol.read_request(sys.stdin.buffer)
    source = request.source
    _append(os.environ.get("FAKE_JOB_ORDER", ""), {
        "request_id": request_id, "attempt_id": attempt_id, "lease_fence": fence,
        "run_id": request.run_id, "session_id": request.session_id, "kind": request.kind,
    })
    counter = os.environ.get("FAKE_JOB_COUNTER")
    if counter:
        path = Path(counter)
        count = int(path.read_text(encoding="utf-8")) if path.is_file() else 0
        path.write_text(str(count + 1), encoding="utf-8")
    running = os.environ.get("FAKE_JOB_RUNNING")
    if running:
        Path(running).write_text(json.dumps({
            "request_id": request_id, "attempt_id": attempt_id, "pid": os.getpid(),
        }), encoding="utf-8")
    if b"ORPHAN" in source:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        _append(os.environ.get("FAKE_JOB_ORPHAN_PIDS", ""), {"pid": child.pid})
    if b"PROGRESS" in source:
        for index in range(5 if b"MANY" in source else 2):
            worker_protocol.write_progress(
                sys.stdout.buffer, request_id=request_id, attempt_id=attempt_id,
                lease_fence=fence, progress={
                    "stage": "compile" if index == 0 else "correctness",
                    "completed_reps": index,
                    "total_reps": max(request.reps, 5 if b"MANY" in source else 2),
                    "detail": f"fake progress {index}", "observed_at_ms": 1 + index,
                },
            )
        time.sleep(0.1)
    if b"FAIL_FIRST" in source and fence == 1:
        return 7
    fail_dir = os.environ.get("FAKE_JOB_FAIL_DIR")
    if b"FAIL_ONCE" in source and fail_dir:
        # Per-request failure marker: independent of the global lease fence, so
        # a retry probe need not be the very first attempt in the store.
        marker = Path(fail_dir) / request_id
        marker.parent.mkdir(parents=True, exist_ok=True)
        if marker.exists():
            pass
        else:
            marker.write_text("1", encoding="utf-8")
            return 7
    if b"EXIT" in source:
        return 7
    if b"NO_RESULT" in source:
        return 0
    hold_dir = os.environ.get("FAKE_JOB_HOLD_DIR")
    if b"HOLD_ONCE" in source and hold_dir:
        marker = Path(hold_dir) / request_id
        marker.parent.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text("1", encoding="utf-8")
            time.sleep(30.0)
    if b"RECOVER" in source and fence == 1:
        time.sleep(30.0)
    elif b"FAIR_SLEEP" in source:
        time.sleep(0.4)
    elif b"SLEEP" in source or b"BLOCK" in source:
        time.sleep(30.0)
    release = os.environ.get("FAKE_JOB_RELEASE")
    if release:
        while not Path(release).exists():
            time.sleep(0.01)
    status = "ok"
    if b"COMPILE_ERROR" in source:
        status = "compile_error"
    elif b"INCORRECT" in source:
        status = "incorrect"
    elif b"CRASH" in source:
        status = "crash"
    report = {
        "task": "wrong-task" if b"BAD_REPORT" in source else request.task,
        "dtype": request.dtype, "size": request.size, "mode": request.mode,
        "seed": request.seed_int, "status": status, "stage": "done",
        "candidate_us": 10.0, "speedup": 1.0,
        "elapsed_s": 0.5 if request.mode == "timed" else 0.0,
        "target_id": request.target_id, "protocol_version": "3",
        "simulated": True,
    }
    evidence = None if b"NO_EVIDENCE" in source else {
        "candidate_sha256": request.candidate_sha256,
        "measurement_kind": "simulated",
    }
    if evidence is not None and b"LARGE_RESULT" in source:
        evidence["padding"] = "x" * 2_000_000
    worker_protocol.write_result(
        sys.stdout.buffer, request_id=request_id, attempt_id=attempt_id,
        lease_fence=fence, report=report, evidence=evidence,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
