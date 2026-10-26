"""CPU-only fault worker for container execution-plane regression tests.

It speaks the same private protocol as ``tests.support.fake_worker_job_engine``
and then misbehaves in one selected way, so the engine's result decision can be
tested without a real evaluator: a malformed report followed by a hang, two
results, and a result followed by a progress frame that blocks the reader.
"""

from __future__ import annotations

import os
import sys
import time

from tcenv.remote import worker_protocol


def _report(request) -> dict:
    return {
        "task": request.task, "dtype": request.dtype, "size": request.size,
        "mode": request.mode, "seed": request.seed_int, "status": "ok", "stage": "done",
        "candidate_us": 1.0, "speedup": 1.0, "elapsed_s": 0.0,
        "target_id": request.target_id, "protocol_version": "3", "simulated": True,
    }


def main() -> int:
    mode = os.environ.get("FAULT_WORKER_MODE", "wrong_task_hang")
    attempt_id, request, request_id, fence = worker_protocol.read_request(sys.stdin.buffer)
    report = _report(request)
    evidence = {"candidate_sha256": request.candidate_sha256}
    if mode in {"wrong_task_hang", "wrong_task_second_result"}:
        report["task"] = "not-the-requested-task"

    def write_result() -> None:
        worker_protocol.write_result(
            sys.stdout.buffer, request_id=request_id, attempt_id=attempt_id,
            lease_fence=fence, report=report, evidence=evidence,
        )
        sys.stdout.buffer.flush()

    if mode == "wrong_task_hang":
        write_result()
        time.sleep(120)
    elif mode == "wrong_task_second_result":
        write_result()
        write_result()
        time.sleep(120)
    elif mode == "double_result":
        write_result()
        write_result()
        time.sleep(120)
    elif mode == "valid_hang":
        write_result()
        time.sleep(120)
    elif mode == "result_then_blocked_progress":
        write_result()
        for index in range(3):
            worker_protocol.write_progress(
                sys.stdout.buffer, request_id=request_id, attempt_id=attempt_id, lease_fence=fence,
                progress={
                    "stage": "finalizing", "completed_reps": index, "total_reps": 3,
                    "detail": "late progress", "observed_at_ms": 1 + index,
                },
            )
        sys.stdout.buffer.flush()
    elif mode == "hang_without_result":
        time.sleep(120)
    else:
        raise SystemExit(f"unknown FAULT_WORKER_MODE {mode!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
