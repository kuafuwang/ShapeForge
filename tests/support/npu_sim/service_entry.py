"""Test-only composition root for the real remote service broker.

Only the worker command and the simulated identity differ from production; the
service core, framing, authentication, idempotency, queue and lease are the same.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tcenv.remote.service import RemoteService, ServiceIdentity, SessionGrant


TASK_FINGERPRINT = "a" * 64
TASK_PACK_DIGEST = "b" * 64
TOKEN = "test-session-token"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--selfcheck-max", type=int, default=20)
    parser.add_argument("--selfcheck-max-gpu-s", type=float, default=120.0)
    parser.add_argument("--allow-kind", action="append", choices=["formal_eval", "selfcheck"], default=None)
    parser.add_argument("--expires-at", type=float, default=None)
    parser.add_argument("--token", default=TOKEN)
    parser.add_argument("--formal-token", default=None)
    parser.add_argument("--selfcheck-token", default=None)
    parser.add_argument("--run-id", default="run-1")
    parser.add_argument("--session-id", default="session-1")
    parser.add_argument("--alt-token", default=None)
    parser.add_argument("--alt-run-id", default="run-2")
    parser.add_argument("--alt-session-id", default="session-1")
    parser.add_argument("--aging-s", type=float, default=60.0)
    parser.add_argument("--signature-key-file", type=Path, default=None)
    args = parser.parse_args(argv)
    identity = ServiceIdentity(
        site_id="sim-site",
        device_id="sim-device",
        backend="sim-npu",
        backend_version="1",
        target_id="sim-npu-test",
        worker_build="sim-build-1",
        task_pack_digest=TASK_PACK_DIGEST,
        calibration_digest="c" * 64,
        task_fingerprints=frozenset({TASK_FINGERPRINT}),
        measurement_kind="simulated",
        publication_allowed=False,
    )
    grant = SessionGrant(
        token=args.token,
        run_id=args.run_id,
        session_id=args.session_id,
        target_id="sim-npu-test",
        allowed_kinds=frozenset(args.allow_kind or ["formal_eval", "selfcheck"]),
        expires_at=args.expires_at,
    )
    grants = [grant]
    if args.formal_token:
        grants.append(SessionGrant(token=args.formal_token, run_id=args.run_id, session_id=args.session_id, target_id="sim-npu-test", allowed_kinds=frozenset({"formal_eval"}), expires_at=args.expires_at))
    if args.selfcheck_token:
        grants.append(SessionGrant(token=args.selfcheck_token, run_id=args.run_id, session_id=args.session_id, target_id="sim-npu-test", allowed_kinds=frozenset({"selfcheck"}), expires_at=args.expires_at))
    if args.alt_token:
        grants.append(SessionGrant(token=args.alt_token, run_id=args.alt_run_id, session_id=args.alt_session_id, target_id="sim-npu-test"))
    service = RemoteService(
        identity=identity,
        grants=grants,
        worker_command=[sys.executable, "-m", "tests.support.npu_sim.worker_entry"],
        state_dir=args.state_dir,
        selfcheck_max_timed=args.selfcheck_max,
        selfcheck_max_gpu_s=args.selfcheck_max_gpu_s,
        selfcheck_age_s=args.aging_s,
        signing_key=args.signature_key_file.read_bytes() if args.signature_key_file else None,
    )
    service.serve(sys.stdin.buffer, sys.stdout.buffer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
