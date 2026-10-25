"""Shared device-free fixtures/helpers for the transport-neutral job engine."""

from __future__ import annotations

import hashlib
import secrets
import sys
import time
from pathlib import Path

from tcenv.remote.identity import ServiceIdentity, SessionGrant
from tcenv.remote.job_engine import JobEngine
from tcenv.remote.job_types import (
    ExpectedEvaluator, ExpectedIdentities, ExternalEvaluationRequest, JobSnapshot,
)

TOKEN = "external-test-grant-token"
OTHER_TOKEN = "external-other-grant-token"
TASK_FINGERPRINT = "a" * 64
TASK_PACK_DIGEST = "b" * 64
TARGET = "sim-external-test"


def identity() -> ServiceIdentity:
    return ServiceIdentity(
        site_id="sim-external-site", device_id="sim-external-device",
        backend="sim-npu", backend_version="1", target_id=TARGET,
        worker_build="sim-external-build", task_pack_digest=TASK_PACK_DIGEST,
        calibration_digest="c" * 64, task_fingerprints=frozenset({TASK_FINGERPRINT}),
        measurement_kind="simulated", publication_allowed=False,
    )


def grant(
    token: str = TOKEN, *, run_id: str = "run-1", session_id: str = "session-1",
    subject: str = "subject-1", kinds=frozenset({"formal_eval", "selfcheck"}),
    modes=frozenset(), expires_at: float | None = None,
    max_timed: int = 20, max_gpu_s: float = 120.0,
) -> SessionGrant:
    return SessionGrant(
        token=token, run_id=run_id, session_id=session_id, target_id=TARGET,
        allowed_kinds=frozenset(kinds), allowed_modes=frozenset(modes),
        expires_at=expires_at, max_timed=max_timed, max_gpu_s=max_gpu_s,
        subject=subject,
    )


def request(
    request_id: str = "request-1", *, source: bytes = b"OK",
    kind: str = "formal_eval", mode: str = "quick", run_id: str = "run-1",
    session_id: str = "session-1", candidate_timeout_ms: int = 3000,
    queue_timeout_ms: int = 3000, reps: int = 2,
) -> ExternalEvaluationRequest:
    return ExternalEvaluationRequest(
        kind=kind, request_id=request_id, run_id=run_id, session_id=session_id,
        task="relu", dtype="fp16", size="small", target_id=TARGET,
        task_fingerprint=TASK_FINGERPRINT,
        candidate_sha256=hashlib.sha256(source).hexdigest(), source=source,
        mode=mode, seed="3", reps=reps, queue_timeout_ms=queue_timeout_ms,
        candidate_timeout_ms=candidate_timeout_ms, cases=None,
    )


def expected_identities(value: ServiceIdentity | None = None) -> ExpectedIdentities:
    value = value or identity()
    return ExpectedIdentities(ExpectedEvaluator(
        site_id=value.site_id, device_id=value.device_id, target_id=value.target_id,
        worker_build=value.worker_build, task_pack_digest=value.task_pack_digest,
        evaluator_epoch=value.external_epoch(execution_protocol_version="3"),
    ))


def store_capabilities(store, *, token: str = TOKEN, nonce: str | None = None):
    expected = expected_identities()
    nonce = secrets.token_hex(16) if nonce is None else nonce.ljust(16, "0")
    return store.get_capabilities(
        grant_token=token, client_nonce=nonce, expected={},
        expected_identities=expected,
        provenance=identity().external_provenance(execution_protocol_version="3"),
        api_contract_ids=[expected.api_contract_id],
        evidence_contract_ids=[expected.evidence_contract_id],
        required_features=[], supported_features=frozenset(),
    )


def submit_to_store(store, req: ExternalEvaluationRequest, *, token: str = TOKEN) -> JobSnapshot:
    cap = store_capabilities(store, token=token)
    snapshot, created = store.submit(
        request=req, request_digest=req.request_digest,
        expected=expected_identities(),
        provenance=identity().external_provenance(execution_protocol_version="3"),
        capability_id=cap.capability_id, required_features=[], grant_token=token,
    )
    assert created
    return snapshot


def engine(
    state_dir: Path, *, grants: list[SessionGrant] | None = None,
    worker_env: dict[str, str] | None = None, **options,
) -> JobEngine:
    return JobEngine(
        identity=identity(), grants=grants or [grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_job_engine"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=state_dir,
        worker_env=worker_env, **options,
    )


def capabilities(engine: JobEngine, *, token: str = TOKEN, nonce: str | None = None, valid_for_ms: int = 60_000):
    # Test labels remain readable while meeting the protocol's nonce bound;
    # omitted nonces are fresh, as a real client handshake would be.
    nonce = (secrets.token_hex(16) if nonce is None else nonce).ljust(16, "0")
    return engine.get_capabilities(
        grant_token=token, client_nonce=nonce, valid_for_ms=valid_for_ms,
    )


def submit(engine: JobEngine, req: ExternalEvaluationRequest, *, token: str = TOKEN, nonce: str | None = None) -> JobSnapshot:
    cap = capabilities(engine, token=token, nonce=nonce)
    return engine.submit(
        request=req, expected=engine.expected_identities,
        capability_id=cap.capability_id, grant_token=token,
    )


def get(engine: JobEngine, req: ExternalEvaluationRequest, *, token: str = TOKEN) -> JobSnapshot:
    return engine.get_job(
        request_id=req.request_id, request_digest=req.request_digest,
        expected=engine.expected_identities, grant_token=token,
    )


def wait_state(
    engine: JobEngine, req: ExternalEvaluationRequest, states: set[str], *,
    token: str = TOKEN, timeout: float = 8.0,
) -> JobSnapshot:
    deadline = time.monotonic() + timeout
    last = get(engine, req, token=token)
    while last.state not in states and time.monotonic() < deadline:
        time.sleep(0.02)
        last = get(engine, req, token=token)
    assert last.state in states, f"timed out waiting for {states}; last={last.state}, failure={last.failure}"
    return last


def retry(engine: JobEngine, req: ExternalEvaluationRequest, attempt_id: str, *, token: str = TOKEN, nonce: str | None = None) -> JobSnapshot:
    cap = capabilities(engine, token=token, nonce=nonce)
    return engine.retry(
        request_id=req.request_id, request_digest=req.request_digest,
        expected=engine.expected_identities, interrupted_attempt_id=attempt_id,
        capability_id=cap.capability_id, grant_token=token,
    )
