"""Process-level crash injection for shared durable-job regression tests."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from tcenv.remote.job_store import JobStore
from tcenv.remote.job_types import ExpectedEvaluator, ExpectedIdentities

from .job_engine import TOKEN, engine, grant, identity, request, submit


def _mark(path: str, text: str) -> None:
    target = Path(path)
    target.write_text(text, encoding="utf-8")
    fd = os.open(target, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["after-accept", "after-running", "after-result-commit"])
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--marker", required=True)
    args = parser.parse_args()
    req = request("crash-request", source=b"RECOVER" if args.action == "after-running" else b"OK")
    if args.action == "after-accept":
        req = request("crash-request", source=b"OK", kind="selfcheck", mode="timed")
        store = JobStore(args.state_dir)
        store.register_grant(grant())
        expected = ExpectedIdentities(
            ExpectedEvaluator(
                site_id=identity().site_id, device_id=identity().device_id,
                target_id=identity().target_id, worker_build=identity().worker_build,
                task_pack_digest=identity().task_pack_digest,
                evaluator_epoch=identity().external_epoch(execution_protocol_version="3"),
            )
        )
        provenance = identity().external_provenance(execution_protocol_version="3")
        cap = store.get_capabilities(
            grant_token=TOKEN, client_nonce="1234567890abcdef", expected={},
            expected_identities=expected, provenance=provenance,
            api_contract_ids=[expected.api_contract_id],
            evidence_contract_ids=[expected.evidence_contract_id],
            required_features=[], supported_features=frozenset(),
        )
        snapshot, created = store.submit(
            request=req, request_digest=req.request_digest, expected=expected,
            provenance=provenance, capability_id=cap.capability_id,
            required_features=[], grant_token=TOKEN,
        )
        assert created and snapshot.state == "accepted"
        _mark(args.marker, snapshot.state)
        os._exit(0)

    service = engine(args.state_dir)
    if args.action == "after-result-commit":
        original = service.store.complete_attempt

        def crash_after_commit(*call_args, **call_kwargs):
            snapshot = original(*call_args, **call_kwargs)
            _mark(args.marker, snapshot.state)
            os._exit(0)

        service.store.complete_attempt = crash_after_commit
        submit(service, req)
        time.sleep(60)
        return 0

    submit(service, req)
    running = os.environ.get("FAKE_JOB_RUNNING")
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if running and Path(running).is_file():
            _mark(args.marker, Path(running).read_text(encoding="utf-8"))
            os._exit(0)
        time.sleep(0.01)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
