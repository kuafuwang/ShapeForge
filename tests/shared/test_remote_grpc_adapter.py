"""End-to-end tests for the external gRPC adapter over the durable job engine."""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import grpc
import pytest

from tcenv.evaluation import EvaluationRequest, verify_result_signature
from tcenv.remote import contracts
from tcenv.remote import grpc_contract as gc
from tcenv.remote.client import RemoteEvaluatorPool
from tcenv.remote.grpc_client import (
    GrpcEvaluatorFactory,
    GrpcExpected,
    SignatureKeyring,
)
from tcenv.remote.grpc_server import (
    GrpcServerConfig,
    build_server,
    status_for_error,
)
from tcenv.remote.job_types import JobError
from tcenv.remote.v1 import evaluator_pb2 as pb
from tcenv.remote.v1 import evaluator_pb2_grpc as pb_grpc
from tests.support.job_engine import (
    TARGET,
    TASK_FINGERPRINT,
    TASK_PACK_DIGEST,
    TOKEN,
    engine as make_engine,
    grant,
    identity,
    request as external_request,
)

SIGNING_KEY = b"k" * 32
SIGNING_KEY_ID = "sim-external-site/key-1"
LEGACY_KEY = b"legacy-outbox-key"


def _legacy_request(
    *, request_id: str = "grpc-request-1", source: str = "OK",
    kind: str = "formal_eval", mode: str = "quick",
    run_id: str = "run-1", session_id: str = "session-1",
    queue_timeout_s: float = 3.0, candidate_timeout_s: float = 3.0,
    progress_path: Path | None = None,
) -> EvaluationRequest:
    return EvaluationRequest(
        kind=kind, request_id=request_id, run_id=run_id, session_id=session_id,
        task="relu", dtype="fp16", size="small", target_id=TARGET,
        task_fingerprint=TASK_FINGERPRINT, source=source, mode=mode, seed=3, reps=2,
        queue_timeout_s=queue_timeout_s, candidate_timeout_s=candidate_timeout_s,
        progress_path=progress_path,
    )


def _expected_pb(engine) -> pb.ExpectedEvaluator:
    expected = engine.expected_identities.evaluator
    return pb.ExpectedEvaluator(
        site_id=expected.site_id, device_id=expected.device_id,
        target_id=expected.target_id, worker_build=expected.worker_build,
        task_pack_digest=expected.task_pack_digest,
        evaluator_epoch=expected.evaluator_epoch,
    )


def _keyring(*, key: bytes = SIGNING_KEY, key_id: str = SIGNING_KEY_ID) -> SignatureKeyring:
    return SignatureKeyring({key_id: key})


class Harness:
    def __init__(
        self, tmp_path: Path, *, signing_key: bytes | None = SIGNING_KEY,
        worker_env: dict[str, str] | None = None,
        keyring: SignatureKeyring | None = None,
        required_features: tuple[str, ...] = ("durable-jobs", "attempt-receipts", "progress"),
        legacy_signature_key: bytes | None = LEGACY_KEY,
        watch_slice_s: float = 1.0,
        job_margin_s: float = 5.0,
        token: str = TOKEN,
        signing_key_id: str = SIGNING_KEY_ID,
        token_provider=None,
        state_dir: Path | None = None,
        signing_keyring=None,
        auth_max_failures: int = 64,
    ) -> None:
        self.engine = make_engine(
            state_dir or (tmp_path / "state"),
            worker_env=worker_env,
            signing_key=signing_key,
            signing_key_id=signing_key_id if signing_key is not None else "",
            signing_keyring=signing_keyring,
        )
        self.server = build_server(
            self.engine,
            config=GrpcServerConfig(
                address="127.0.0.1:0", allow_insecure=True,
                require_client_identity=False, watch_poll_s=0.02,
                auth_max_failures=auth_max_failures,
            ),
        )
        self.factory = GrpcEvaluatorFactory(
            endpoint=self.server.address,
            token_provider=token_provider or (lambda _run, _session: token),
            run_id="run-1",
            target_id=TARGET,
            expected=GrpcExpected(
                site_id=identity().site_id,
                device_id=identity().device_id,
                target_id=TARGET,
                worker_build=identity().worker_build,
                task_pack_digest=TASK_PACK_DIGEST,
            ),
            required_features=required_features,
            keyring=keyring or _keyring(),
            legacy_signature_key=legacy_signature_key,
            allow_insecure=True,
            watch_slice_s=watch_slice_s,
            job_margin_s=job_margin_s,
        )

    def close(self) -> None:
        self.factory.close()
        self.server.stop(grace_s=0.2)
        self.engine.close()

    def __enter__(self) -> "Harness":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@pytest.fixture
def harness(tmp_path):
    value = Harness(tmp_path)
    yield value
    value.close()


def test_grpc_round_trip_binds_attempt_receipt_and_signature(harness):
    request = _legacy_request()
    result = harness.factory.formal(request)
    external_epoch = harness.engine.expected_identities.evaluator.evaluator_epoch
    assert result.report["status"] == "ok"
    assert result.provenance.epoch == external_epoch
    assert result.provenance.site_id == identity().site_id
    assert result.provenance.measurement_kind == "simulated"
    assert result.provenance.request_digest == request.request_digest
    assert result.receipt["request_id"] == request.request_id
    assert result.receipt["request_digest"] == request.request_digest
    assert result.receipt["candidate_sha256"] == request.candidate_sha256
    assert result.receipt["attempt_id"]
    assert result.receipt["lease_fence"] == "1"
    assert result.evidence["candidate_sha256"] == request.candidate_sha256
    # The host keeps signing its legacy envelope for the outbox.
    assert verify_result_signature(result, LEGACY_KEY)
    # The external epoch is the measurement identity the host must allowlist.
    assert result.provenance.protocol_version == "3"


def test_grpc_duplicate_submit_replays_and_runs_once(tmp_path):
    counter = tmp_path / "count.txt"
    with Harness(tmp_path, worker_env={"FAKE_JOB_COUNTER": str(counter)}) as harness:
        request = _legacy_request()
        first = harness.factory.formal(request)
        second = harness.factory.formal(request)
        assert counter.read_text() == "1"
        assert first.receipt["receipt_id"] == second.receipt["receipt_id"]
        assert first.report == second.report


def test_grpc_required_feature_is_checked_before_submit(tmp_path):
    counter = tmp_path / "count.txt"
    with Harness(
        tmp_path,
        worker_env={"FAKE_JOB_COUNTER": str(counter)},
        required_features=("durable-jobs", "watch-v2"),
    ) as harness:
        with pytest.raises(Exception) as excinfo:
            harness.factory.formal(_legacy_request())
        assert getattr(excinfo.value, "code", "") == "required_feature_missing"
        assert not counter.exists()


def test_grpc_unsigned_result_is_rejected(tmp_path):
    with Harness(tmp_path, signing_key=None) as harness:
        with pytest.raises(Exception) as excinfo:
            harness.factory.formal(_legacy_request())
        assert getattr(excinfo.value, "code", "") == "result_invalid"


def test_grpc_wrong_or_revoked_signing_key_is_rejected(tmp_path):
    with Harness(tmp_path, keyring=_keyring(key=b"w" * 32)) as harness:
        with pytest.raises(Exception) as excinfo:
            harness.factory.formal(_legacy_request(request_id="wrong-key"))
        assert getattr(excinfo.value, "code", "") == "signature_mismatch"

    with Harness(tmp_path) as harness:
        harness.factory.keyring.revoke(SIGNING_KEY_ID)
        with pytest.raises(Exception) as excinfo:
            harness.factory.formal(_legacy_request(request_id="revoked-key"))
        assert getattr(excinfo.value, "code", "") == "signature_key_revoked"


def test_grpc_grant_and_run_scope_are_enforced(tmp_path):
    with Harness(tmp_path, token="wrong-token") as harness:
        with pytest.raises(Exception) as excinfo:
            harness.factory.formal(_legacy_request(request_id="bad-grant"))
        assert getattr(excinfo.value, "code", "") == "grant_rejected"

    with Harness(tmp_path) as harness:
        other_run = _legacy_request(request_id="other-run", run_id="run-2")
        with pytest.raises(Exception) as excinfo:
            harness.factory.formal(other_run)
        assert getattr(excinfo.value, "code", "") == "session_mismatch"


def test_grpc_task_fingerprint_is_bound_to_task_dtype_and_size(harness):
    mismatched = _legacy_request(request_id="wrong-task")
    object.__setattr__(mismatched, "task", "gemm")
    with pytest.raises(Exception) as excinfo:
        harness.factory.formal(mismatched)
    assert getattr(excinfo.value, "code", "") == "precondition_failed"
    wrong_dtype = _legacy_request(request_id="wrong-dtype")
    object.__setattr__(wrong_dtype, "dtype", "fp32")
    with pytest.raises(Exception) as excinfo:
        harness.factory.formal(wrong_dtype)
    assert getattr(excinfo.value, "code", "") == "precondition_failed"


def test_grpc_interrupted_job_requires_explicit_resume(tmp_path):
    counter = tmp_path / "count.txt"
    with Harness(tmp_path, worker_env={"FAKE_JOB_COUNTER": str(counter)}) as harness:
        request = _legacy_request(source="FAIL_FIRST", request_id="resume-me")
        with pytest.raises(Exception) as first:
            harness.factory.formal(request)
        assert getattr(first.value, "code", "") == "worker_failed"
        assert getattr(first.value, "retryable", False) is True
        assert counter.read_text() == "1"

        # Submit only replays the interrupted durable outcome; it never creates
        # an attempt on its own.
        with pytest.raises(Exception) as replay:
            harness.factory.formal(request)
        assert getattr(replay.value, "code", "") == "worker_failed"
        assert counter.read_text() == "1"

        resumed = harness.factory.formal_resume(request)
        assert resumed.report["status"] == "ok"
        assert counter.read_text() == "2"
        assert resumed.receipt["lease_fence"] == "2"


def test_grpc_cancel_reaches_a_terminal_state(tmp_path):
    running = tmp_path / "running.json"
    with Harness(tmp_path, job_margin_s=0.5, worker_env={"FAKE_JOB_RUNNING": str(running)}) as harness:
        request = _legacy_request(source="SLEEP", request_id="cancel-me", candidate_timeout_s=20.0)
        errors: list[BaseException] = []

        def run() -> None:
            try:
                harness.factory.formal(request)
            except BaseException as exc:  # noqa: BLE001 - asserted below
                errors.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5.0
        while not running.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert running.exists()
        harness.factory.cancel(request.request_id)
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert errors
        assert getattr(errors[0], "code", "") in {"cancelled", "worker_failed", "interrupted"}
        final = harness.engine.get_job(
            request_id=request.request_id,
            request_digest=contracts_request_digest(request),
            expected=harness.engine.expected_identities,
            grant_token=TOKEN,
        )
        assert final.state == "cancelled"


def test_grpc_watch_reconnects_and_reconciles_through_getjob(tmp_path):
    with Harness(tmp_path, watch_slice_s=0.05, job_margin_s=10.0) as harness:
        request = _legacy_request(source="FAIR_SLEEP", request_id="reconnect-me")
        result = harness.factory.formal(request)
        assert result.report["status"] == "ok"


def test_grpc_watch_after_terminal_and_gap_returns_snapshot(harness):
    request = _legacy_request(request_id="terminal-watch")
    harness.factory.formal(request)
    stub = harness.factory._stub
    expected = harness.engine.expected_identities
    reference = pb.JobRef(
        request_id=request.request_id,
        request_digest=contracts_request_digest(request),
        expected=gc.expected_identities_to_proto(expected),
    )
    metadata = (("authorization", f"Bearer {TOKEN}"),)
    events = list(stub.Watch(pb.WatchRequest(job=reference, after_sequence=0), metadata=metadata, timeout=5))
    assert events
    assert events[-1].HasField("snapshot")
    assert events[-1].snapshot.state == pb.JOB_STATE_COMPLETED
    last = events[-1].sequence
    replayed = list(stub.Watch(pb.WatchRequest(job=reference, after_sequence=last), metadata=metadata, timeout=5))
    assert len(replayed) == 1
    assert replayed[0].snapshot.state == pb.JOB_STATE_COMPLETED
    assert replayed[0].sequence == last


def test_grpc_getjob_is_scoped_to_the_grant(harness):
    request = _legacy_request(request_id="scoped-job")
    harness.factory.formal(request)
    stub = harness.factory._stub
    reference = pb.JobRef(
        request_id=request.request_id,
        request_digest=contracts_request_digest(request),
        expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
    )
    metadata = (("authorization", "Bearer other-token"),)
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.GetJob(reference, metadata=metadata, timeout=5)
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_grpc_invalid_presence_enums_and_bounds_fail_closed(harness):
    stub = harness.factory._stub
    expected = harness.engine.expected_identities
    metadata = (("authorization", f"Bearer {TOKEN}"),)
    body = pb.EvaluationRequest(
        kind=pb.EVALUATION_KIND_FORMAL_EVAL,
        request_id="bad-enum",
        run_id="run-1",
        session_id="session-1",
        task="relu",
        dtype="fp16",
        size="small",
        target_id=TARGET,
        task_fingerprint=TASK_FINGERPRINT,
        candidate_sha256=hashlib.sha256(b"OK").hexdigest(),
        source=b"OK",
        mode=99,
        reps=2,
        queue_timeout_ms=3000,
        candidate_timeout_ms=3000,
        cache_policy=pb.CACHE_POLICY_REUSE,
    )
    submit = pb.SubmitRequest(
        request=body,
        request_digest="0" * 64,
        expected=gc.expected_identities_to_proto(expected),
    )
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.Submit(submit, metadata=metadata, timeout=5)
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    body.mode = pb.EVALUATION_MODE_QUICK
    body.reps = 0
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.Submit(submit, metadata=metadata, timeout=5)
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    body.reps = 2
    body.source = b"x" * (contracts.MAX_SOURCE_BYTES + 1)
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.Submit(submit, metadata=metadata, timeout=5)
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    # A missing Authorization header is rejected before any business logic.
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.GetCapabilities(pb.CapabilitiesRequest(client_nonce="a" * 16), timeout=5)
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_grpc_submit_rejects_a_digest_that_does_not_bind_source_bytes(harness):
    stub = harness.factory._stub
    metadata = (("authorization", f"Bearer {TOKEN}"),)
    external = harness.engine.expected_identities
    body = pb.EvaluationRequest(
        kind=pb.EVALUATION_KIND_FORMAL_EVAL,
        request_id="bad-digest",
        run_id="run-1",
        session_id="session-1",
        task="relu",
        dtype="fp16",
        size="small",
        target_id=TARGET,
        task_fingerprint=TASK_FINGERPRINT,
        candidate_sha256=hashlib.sha256(b"OK").hexdigest(),
        source=b"OK",
        mode=pb.EVALUATION_MODE_QUICK,
        reps=2,
        queue_timeout_ms=3000,
        candidate_timeout_ms=3000,
        cache_policy=pb.CACHE_POLICY_REUSE,
    )
    capabilities = stub.GetCapabilities(
        pb.CapabilitiesRequest(
            api_contract_ids=[contracts.API_CONTRACT_ID],
            evidence_contract_ids=[contracts.EVIDENCE_CONTRACT_ID],
            client_nonce="b" * 16,
            expected=_expected_pb(harness.engine),
        ),
        metadata=metadata, timeout=5,
    )
    submit = pb.SubmitRequest(
        request=body,
        request_digest="f" * 64,
        expected=gc.expected_identities_to_proto(external),
        capability_id=capabilities.capability_id,
    )
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.Submit(submit, metadata=metadata, timeout=5)
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_grpc_stale_capability_from_a_previous_frontend_is_rejected(tmp_path):
    harness = Harness(tmp_path)
    try:
        stub = harness.factory._stub
        metadata = (("authorization", f"Bearer {TOKEN}"),)
        capabilities = stub.GetCapabilities(
            pb.CapabilitiesRequest(
                api_contract_ids=[contracts.API_CONTRACT_ID],
                evidence_contract_ids=[contracts.EVIDENCE_CONTRACT_ID],
                client_nonce="c" * 16,
                expected=_expected_pb(harness.engine),
            ),
            metadata=metadata, timeout=5,
        )
        engine = harness.engine
    finally:
        harness.close()
    # A different frontend instance owns a different capability namespace.
    second = make_engine(
        tmp_path / "state-2", signing_key=SIGNING_KEY, signing_key_id=SIGNING_KEY_ID,
    )
    server = build_server(
        second,
        config=GrpcServerConfig(address="127.0.0.1:0", allow_insecure=True, require_client_identity=False),
    )
    try:
        channel = grpc.insecure_channel(server.address)
        stub = pb_grpc.EvaluatorStub(channel)
        body = pb.EvaluationRequest(
            kind=pb.EVALUATION_KIND_FORMAL_EVAL,
            request_id="stale-capability",
            run_id="run-1",
            session_id="session-1",
            task="relu",
            dtype="fp16",
            size="small",
            target_id=TARGET,
            task_fingerprint=TASK_FINGERPRINT,
            candidate_sha256=hashlib.sha256(b"OK").hexdigest(),
            source=b"OK",
            mode=pb.EVALUATION_MODE_QUICK,
            reps=2,
            queue_timeout_ms=3000,
            candidate_timeout_ms=3000,
            cache_policy=pb.CACHE_POLICY_REUSE,
        )
        submit = pb.SubmitRequest(
            request=body,
            request_digest=gc.request_from_proto(body).request_digest,
            expected=gc.expected_identities_to_proto(second.expected_identities),
            capability_id=capabilities.capability_id,
        )
        with pytest.raises(grpc.RpcError) as excinfo:
            stub.Submit(submit, metadata=(("authorization", f"Bearer {TOKEN}"),), timeout=5)
        assert excinfo.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        channel.close()
    finally:
        server.stop(grace_s=0.2)
        second.close()


def test_grpc_progress_reaches_the_host_progress_file(harness, tmp_path):
    progress = tmp_path / "eval_progress.json"
    request = _legacy_request(source="PROGRESS", request_id="progress-me", progress_path=progress)
    harness.factory.formal(request)
    assert progress.is_file()
    document = progress.read_text(encoding="utf-8")
    assert "stage" in document


def test_grpc_error_status_carries_typed_detail_and_retry_info():
    error = JobError(
        "failed_precondition", "unsupported required features: ['x']",
        retryable=True, missing_features=("x",),
    )
    status = status_for_error(error, request_id="req", request_digest="a" * 64)
    assert status.code == grpc.StatusCode.FAILED_PRECONDITION.value[0]
    kinds = {item.type_url for item in status.details}
    assert any(url.endswith("google.rpc.ErrorInfo") for url in kinds)
    assert any(url.endswith("google.rpc.RetryInfo") for url in kinds)
    assert any(url.endswith("tcenv.remote.v1.EvaluationErrorDetail") for url in kinds)
    detail = pb.EvaluationErrorDetail()
    assert status.details[1].Unpack(detail)
    assert detail.missing_features == ["x"]
    assert detail.retryable is True


def test_grpc_factory_refuses_an_insecure_channel_without_explicit_opt_in():
    with pytest.raises(ValueError, match="TLS"):
        GrpcEvaluatorFactory(
            endpoint="127.0.0.1:1",
            token_provider=lambda _run, _session: TOKEN,
            run_id="run-1",
            target_id=TARGET,
            keyring=_keyring(),
        )


def contracts_request_digest(request: EvaluationRequest) -> str:
    from tcenv.remote.job_types import ExternalEvaluationRequest

    return ExternalEvaluationRequest.from_legacy(request).request_digest


def test_grpc_grant_renewal_reattaches_and_revocation_cancels(tmp_path):
    current = {"token": TOKEN}

    def provider(_run: str, _session: str) -> str:
        return current["token"]

    with Harness(tmp_path, token_provider=provider) as harness:
        request = _legacy_request(request_id="renew-me")
        harness.factory.formal(request)
        digest = contracts_request_digest(request)
        renewed = "renewed-grant-token"
        harness.engine.renew_grant(grant(renewed))
        # The superseded token can no longer read the job.
        with pytest.raises(Exception) as excinfo:
            harness.engine.get_job(
                request_id=request.request_id, request_digest=digest,
                expected=harness.engine.expected_identities, grant_token=TOKEN,
            )
        assert getattr(excinfo.value, "code", "") == "unauthenticated"
        # The renewed token reattaches to the same durable result.
        current["token"] = renewed
        replayed = harness.factory.formal(request)
        assert replayed.receipt["receipt_id"]

        running = tmp_path / "revoke-running.json"
        harness.engine.worker_env["FAKE_JOB_RUNNING"] = str(running)
        sleeper = _legacy_request(source="SLEEP", request_id="revoke-me", candidate_timeout_s=20.0)
        errors: list[BaseException] = []

        def run() -> None:
            try:
                harness.factory.formal(sleeper)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5.0
        while not running.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert running.exists()
        harness.engine.revoke_grant(renewed)
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert errors
        assert getattr(errors[0], "code", "") in {"permission_denied", "cancelled"}
        deadline = time.monotonic() + 5.0
        final = harness.engine.store._job_by_id(sleeper.request_id)
        while final.state not in {"cancelled", "failed"} and time.monotonic() < deadline:
            time.sleep(0.02)
            final = harness.engine.store._job_by_id(sleeper.request_id)
        assert final.state == "cancelled"


def test_grpc_hmac_key_rotation_keeps_history_and_new_results_separate(tmp_path):
    rotated_id = "sim-external-site/key-2"
    rotated_key = b"r" * 32
    keyring = SignatureKeyring({SIGNING_KEY_ID: SIGNING_KEY, rotated_id: rotated_key})
    state = tmp_path / "state"
    first = Harness(tmp_path, keyring=keyring, state_dir=state)
    try:
        old = first.factory.formal(_legacy_request(request_id="history-key"))
    finally:
        first.close()

    deadline = time.monotonic() + 5.0
    while True:
        try:
            second = Harness(
                tmp_path, keyring=keyring, signing_key=rotated_key,
                signing_key_id=rotated_id, state_dir=state,
                signing_keyring={SIGNING_KEY_ID: SIGNING_KEY, rotated_id: rotated_key},
            )
            break
        except Exception:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
    try:
        new = second.factory.formal(_legacy_request(request_id="current-key"))
        assert old.receipt["signature_key_id"] == SIGNING_KEY_ID
        assert new.receipt["signature_key_id"] == rotated_id
        # The historical job still replays through the rotated frontend.
        replayed = second.factory.formal(_legacy_request(request_id="history-key"))
        assert replayed.receipt["receipt_id"] == old.receipt["receipt_id"]
        # Revoking the historical key rejects only results signed with it.
        keyring.revoke(SIGNING_KEY_ID)
        with pytest.raises(Exception) as excinfo:
            second.factory.formal(_legacy_request(request_id="history-key"))
        assert getattr(excinfo.value, "code", "") == "signature_key_revoked"
        assert second.factory.formal(_legacy_request(request_id="current-key")).receipt["attempt_id"]
    finally:
        second.close()


def test_grpc_presence_of_seed_and_cases_changes_the_request_digest(tmp_path):
    with Harness(tmp_path) as harness:
        none_seed = _legacy_request(request_id="seed-none", source="OK")
        object.__setattr__(none_seed, "seed", None)
        zero_seed = _legacy_request(request_id="seed-zero", source="OK")
        object.__setattr__(zero_seed, "seed", 0)
        assert contracts_request_digest(none_seed) != contracts_request_digest(zero_seed)
        empty_cases = _legacy_request(request_id="cases-empty", source="OK")
        object.__setattr__(empty_cases, "cases", ())
        no_cases = _legacy_request(request_id="cases-none", source="OK")
        assert contracts_request_digest(empty_cases) != contracts_request_digest(no_cases)
        assert harness.factory.formal(none_seed).report["status"] == "ok"
        assert harness.factory.formal(zero_seed).report["status"] == "ok"
        assert harness.factory.formal(empty_cases).report["status"] == "ok"
        assert harness.factory.formal(no_cases).report["status"] == "ok"


def test_grpc_reps_and_seed_overflow_are_rejected(harness):
    stub = harness.factory._stub
    metadata = (("authorization", f"Bearer {TOKEN}"),)
    expected = harness.engine.expected_identities
    capabilities = stub.GetCapabilities(
        pb.CapabilitiesRequest(
            api_contract_ids=[contracts.API_CONTRACT_ID],
            evidence_contract_ids=[contracts.EVIDENCE_CONTRACT_ID],
            client_nonce="d" * 16,
            expected=_expected_pb(harness.engine),
        ),
        metadata=metadata, timeout=5,
    )
    body = pb.EvaluationRequest(
        kind=pb.EVALUATION_KIND_FORMAL_EVAL,
        request_id="overflow-reps",
        run_id="run-1",
        session_id="session-1",
        task="relu",
        dtype="fp16",
        size="small",
        target_id=TARGET,
        task_fingerprint=TASK_FINGERPRINT,
        candidate_sha256=hashlib.sha256(b"OK").hexdigest(),
        source=b"OK",
        mode=pb.EVALUATION_MODE_QUICK,
        reps=contracts.MAX_REPS + 1,
        queue_timeout_ms=3000,
        candidate_timeout_ms=3000,
        cache_policy=pb.CACHE_POLICY_REUSE,
    )
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.Submit(
            pb.SubmitRequest(
                request=body, request_digest="0" * 64,
                expected=gc.expected_identities_to_proto(expected),
                capability_id=capabilities.capability_id,
            ),
            metadata=metadata, timeout=5,
        )
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    body.reps = 2
    body.seed = "18446744073709551616"  # 2^64, outside the protocol range
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.Submit(
            pb.SubmitRequest(
                request=body, request_digest="0" * 64,
                expected=gc.expected_identities_to_proto(expected),
                capability_id=capabilities.capability_id,
            ),
            metadata=metadata, timeout=5,
        )
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def _capabilities_request(engine, nonce: str, *, api=None, evidence=None) -> pb.CapabilitiesRequest:
    request = pb.CapabilitiesRequest(
        api_contract_ids=[contracts.API_CONTRACT_ID] if api is None else api,
        evidence_contract_ids=[contracts.EVIDENCE_CONTRACT_ID] if evidence is None else evidence,
        client_nonce=nonce,
    )
    if engine is not None:
        request.expected.CopyFrom(_expected_pb(engine))
    return request


def test_grpc_capabilities_requires_expected_identities_and_contract_offers(harness):
    stub = harness.factory.stub
    metadata = (("authorization", f"Bearer {TOKEN}"),)
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.GetCapabilities(
            _capabilities_request(None, "cap-no-expected-0000"), metadata=metadata, timeout=5,
        )
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    for kwargs in ({"api": []}, {"evidence": []}):
        with pytest.raises(grpc.RpcError) as excinfo:
            stub.GetCapabilities(
                _capabilities_request(harness.engine, "cap-no-offer-000000", **kwargs),
                metadata=metadata, timeout=5,
            )
        assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    partial = pb.CapabilitiesRequest(
        api_contract_ids=[contracts.API_CONTRACT_ID],
        evidence_contract_ids=[contracts.EVIDENCE_CONTRACT_ID],
        client_nonce="cap-partial-expected",
    )
    partial.expected.CopyFrom(pb.ExpectedEvaluator(
        site_id=identity().site_id, target_id=TARGET,
    ))
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.GetCapabilities(partial, metadata=metadata, timeout=5)
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_grpc_unknown_fields_and_non_utf8_source_are_rejected(harness):
    stub = harness.factory.stub
    metadata = (("authorization", f"Bearer {TOKEN}"),)
    expected = gc.expected_identities_to_proto(harness.engine.expected_identities)
    capabilities = stub.GetCapabilities(
        _capabilities_request(harness.engine, "unknown-field-cap-00"), metadata=metadata, timeout=5,
    )
    body = pb.EvaluationRequest(
        kind=pb.EVALUATION_KIND_FORMAL_EVAL,
        request_id="unknown-field-request",
        run_id="run-1",
        session_id="session-1",
        task="relu",
        dtype="fp16",
        size="small",
        target_id=TARGET,
        task_fingerprint=TASK_FINGERPRINT,
        candidate_sha256=hashlib.sha256(b"OK").hexdigest(),
        source=b"OK",
        mode=pb.EVALUATION_MODE_QUICK,
        reps=2,
        queue_timeout_ms=3000,
        candidate_timeout_ms=3000,
        cache_policy=pb.CACHE_POLICY_REUSE,
    )
    digest = gc.request_from_proto(body).request_digest
    unknown_field = b"\xb8\x3e\x01"  # field 999, varint 1
    parsed_body = pb.EvaluationRequest.FromString(body.SerializeToString() + unknown_field)
    submit = pb.SubmitRequest(
        request=parsed_body, request_digest=digest, expected=expected,
        capability_id=capabilities.capability_id,
    )
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.Submit(submit, metadata=metadata, timeout=5)
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    clean_submit = pb.SubmitRequest(
        request=body, request_digest=digest, expected=expected,
        capability_id=capabilities.capability_id,
    )
    parsed_submit = pb.SubmitRequest.FromString(clean_submit.SerializeToString() + unknown_field)
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.Submit(parsed_submit, metadata=metadata, timeout=5)
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    non_utf8 = pb.EvaluationRequest()
    non_utf8.CopyFrom(body)
    non_utf8.source = b"\xff\xfe"
    non_utf8.candidate_sha256 = hashlib.sha256(non_utf8.source).hexdigest()
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.Submit(
            pb.SubmitRequest(request=non_utf8, request_digest="0" * 64, expected=expected, capability_id=capabilities.capability_id),
            metadata=metadata, timeout=5,
        )
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_grpc_unknown_grant_trips_the_auth_failure_limiter(tmp_path):
    with Harness(tmp_path, auth_max_failures=2) as harness:
        stub = harness.factory.stub
        metadata = (("authorization", "Bearer syntactically-valid-but-unknown"),)
        for index in range(2):
            with pytest.raises(grpc.RpcError) as excinfo:
                stub.GetCapabilities(
                    _capabilities_request(harness.engine, f"limiter-nonce-{index:06d}"),
                    metadata=metadata, timeout=5,
                )
            assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED
        with pytest.raises(grpc.RpcError) as excinfo:
            stub.GetCapabilities(
                _capabilities_request(harness.engine, "limiter-nonce-final"),
                metadata=metadata, timeout=5,
            )
        assert excinfo.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED


def test_receipt_rejects_sub_millisecond_issued_at():
    receipt_id = contracts.attempt_receipt_id(
        request_id="receipt-request", attempt_id="attempt-1", request_digest="a" * 64,
    )
    receipt = pb.Receipt(
        receipt_id=receipt_id,
        request_id="receipt-request",
        attempt_id="attempt-1",
        request_digest="a" * 64,
        candidate_sha256="b" * 64,
        lease_fence=1,
        session_sequence=1,
    )
    receipt.issued_at.seconds = 1_800_000_000
    receipt.issued_at.nanos = 1
    with pytest.raises(JobError, match="sub-millisecond"):
        gc.receipt_from_proto(receipt)


def test_grpc_watch_rejects_an_old_attempt_filter(harness):
    request = _legacy_request(source="FAIL_FIRST", request_id="watch-old-attempt")
    with pytest.raises(Exception):
        harness.factory.formal(request)
    interrupted = harness.engine.store._job_by_id(request.request_id)
    old_attempt = interrupted.current_attempt.attempt_id
    harness.factory.formal_resume(request)
    stub = harness.factory.stub
    watch = pb.WatchRequest(
        job=pb.JobRef(
            request_id=request.request_id,
            request_digest=contracts_request_digest(request),
            expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
        ),
        after_sequence=0,
        attempt_id=old_attempt,
    )
    with pytest.raises(grpc.RpcError) as excinfo:
        list(stub.Watch(
            watch, metadata=(("authorization", f"Bearer {TOKEN}"),), timeout=5,
        ))
    assert excinfo.value.code() == grpc.StatusCode.FAILED_PRECONDITION


def test_grpc_does_not_replace_the_legacy_ssh_pool():
    """The legacy pool remains importable for historical resume compatibility."""

    assert RemoteEvaluatorPool is not None
    assert "grpc" not in Path("tcenv/remote/client.py").read_text(encoding="utf-8")
