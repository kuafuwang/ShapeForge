"""CPU-only fake evaluator worker for the real-VM container acceptance.

Test-only.  It plugs into the *existing legacy worker protocol* exactly like
``tcenv.remote.worker`` does (the container runs the production
``tcenv.remote.legacy_worker_bridge`` around this module), so the private
engine protocol, the legacy bridge and the result envelope are all exercised
unchanged.

``SLEEP60`` starts a child process with a deterministic argv token and then
sleeps, so the supervisor's timeout/Kill can be checked against both the parent
and its child.  Every other source produces a normal simulated report/evidence
pair that the durable job engine validates.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tcenv.remote import worker as legacy_worker
from tcenv.remote.protocol import EvaluationRequest


def _child_token(request_id: str) -> str:
    prefix = os.environ.get("TCENV_SIM_CHILD_MARKER", "taic-container-sim-child")
    return f"{prefix}-{request_id}"


def _progress(stage: str, message: str) -> None:
    raw = os.environ.get("TCENV_REMOTE_PROGRESS_PATH")
    if not raw:
        return
    try:
        Path(raw).write_text(
            json.dumps({
                "stage": stage, "message": message, "updated_at": int(time.time()),
            }),
            encoding="utf-8",
        )
    except OSError:
        return


class FakeContainerBackend:
    """Deterministic CPU backend; never imported by a production root."""

    def evaluate(self, request: EvaluationRequest) -> dict:
        print(
            f"fake-container-worker pid={os.getpid()} uid={os.getuid()} "
            f"attempt_request={request.request_id}",
            file=sys.stderr, flush=True,
        )
        _progress("compile", "fake container worker started")
        if "SLEEP60" in request.source:
            token = _child_token(request.request_id)
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", token])
            print(f"fake-container-worker child pid={child.pid} token={token}", file=sys.stderr, flush=True)
            time.sleep(120)
        _progress("finalizing", "fake container worker done")
        report = {
            "task": request.task, "dtype": request.dtype, "size": request.size,
            "mode": request.mode, "seed": request.seed, "status": "ok", "stage": "done",
            "candidate_us": 10.0, "speedup": 1.0, "elapsed_s": 0.0,
            "target_id": request.target_id, "protocol_version": "3", "simulated": True,
        }
        evidence = {
            "candidate_sha256": request.candidate_sha256,
            "measurement_kind": "simulated",
            "execution": "container-sim",
        }
        return {"report": report, "evidence": evidence}


def main() -> int:
    return legacy_worker.run_once(FakeContainerBackend(), sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
