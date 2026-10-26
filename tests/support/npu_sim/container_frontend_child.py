"""Frontend process used by the crash/restart acceptance check (test-only).

It owns one JobEngine with the production container supervisor, submits a
sleeping request, signals when the container is durably attached and running,
and then waits to be SIGKILLed by the acceptance runner.  That is a real
frontend crash: no signal handler, no cleanup, no graceful shutdown.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from tests.support.npu_sim import container_sim_common as common


def _log(message: str) -> None:
    """Keep a readable frontend log next to the acceptance report."""

    print(f"[frontend-child] {message}", file=sys.stderr, flush=True)


def _wait_running(engine, request_obj, deadline: float) -> dict[str, str]:
    while time.monotonic() < deadline:
        snapshot = common.get(engine, request_obj)
        attempt = snapshot.current_attempt
        if attempt is not None and attempt.container_id:
            return {
                "request_id": request_obj.request_id,
                "request_digest": request_obj.request_digest,
                "attempt_id": attempt.attempt_id,
                "lease_fence": str(attempt.lease_fence),
                "container_id": attempt.container_id,
                "container_name": attempt.container_name,
                "state": snapshot.state,
            }
        time.sleep(0.05)
    raise SystemExit("frontend child did not attach a container before the deadline")


def main() -> int:
    parser = argparse.ArgumentParser(prog="container_frontend_child")
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--digest", required=True)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--ready", required=True, type=Path)
    parser.add_argument("--deadline-s", type=float, default=90.0)
    args = parser.parse_args()
    engine = common.engine(args.state_dir, args.digest, docker=args.docker)
    request_obj = common.request(args.request_id, source=b"SLEEP60", candidate_timeout_ms=600_000)
    try:
        _log(f"starting request={args.request_id} state={args.state_dir}")
        common.submit(engine, request_obj)
        info = _wait_running(engine, request_obj, time.monotonic() + args.deadline_s)
        # Give the attach client a moment to reach the running container so the
        # crash really happens mid-execution.
        time.sleep(1.0)
        _log(
            "attached attempt={attempt_id} fence={lease_fence} container={container_id} "
            "state={state}".format(**info)
        )
        args.ready.write_text(json.dumps(info, sort_keys=True), encoding="utf-8")
        _log("ready; waiting to be SIGKILLed")
        signal.pause()
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
