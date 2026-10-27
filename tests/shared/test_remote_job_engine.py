"""Device-free regression tests for the transport-neutral durable job engine."""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from tcenv.remote.identity import ServiceIdentity, SessionGrant
from tcenv.remote.job_store import JobStore
from tcenv.remote.job_types import JobError, StateCorrupt
from tcenv.remote.protocol import provenance_epoch

from ..support.job_engine import (
    OTHER_TOKEN, TARGET, TOKEN, capabilities, engine, get, grant, identity,
    request, retry, submit, wait_state,
)


@pytest.fixture
def service(tmp_path):
    value = engine(tmp_path / "state")
    yield value
    value.close()


def test_concurrent_identical_submit_enqueues_once(tmp_path):
    counter = tmp_path / "counter"
    service = engine(tmp_path / "state", worker_env={"FAKE_JOB_COUNTER": str(counter)})
    try:
        req = request("concurrent", source=b"FAIR_SLEEP")
        cap = capabilities(service)
        with ThreadPoolExecutor(max_workers=8) as pool:
            snapshots = list(pool.map(
                lambda _index: service.submit(
                    request=req, expected=service.expected_identities,
                    capability_id=cap.capability_id, grant_token=TOKEN,
                ), range(8),
            ))
        final = wait_state(service, req, {"completed"})
        assert counter.read_text(encoding="utf-8") == "1"
        assert {snapshot.request_id for snapshot in snapshots} == {req.request_id}
        assert final.current_attempt.retry_index == 0
        assert service.store.quota_usage(req).used_count == 0
    finally:
        service.close()


def test_same_id_different_digest_or_identity_or_scope_fails_closed(tmp_path):
    other = grant(OTHER_TOKEN, subject="other-subject")
    service = engine(tmp_path / "state", grants=[grant(), other])
    try:
        req = request("idempotency")
        submit(service, req)
        wait_state(service, req, {"completed"})
        changed = request("idempotency", source=b"OK changed")
        with pytest.raises(JobError, match="different digest") as caught:
            submit(service, changed, nonce="another-nonce-1")
        assert caught.value.code == "already_exists"

        cap = capabilities(service, nonce="another-nonce-2")
        expected = replace(
            service.expected_identities,
            evaluator=replace(service.expected_identities.evaluator, evaluator_epoch="0" * 64),
        )
        with pytest.raises(JobError) as caught:
            service.submit(
                request=req, expected=expected, capability_id=cap.capability_id,
                grant_token=TOKEN,
            )
        assert caught.value.code == "failed_precondition"

        other_cap = capabilities(service, token=OTHER_TOKEN, nonce="another-nonce-3")
        with pytest.raises(JobError) as caught:
            service.submit(
                request=req, expected=service.expected_identities,
                capability_id=other_cap.capability_id, grant_token=OTHER_TOKEN,
            )
        assert caught.value.code == "not_found"
    finally:
        service.close()


def test_external_identity_is_separate_from_legacy_epoch_and_digest():
    value = identity()
    legacy_epoch = provenance_epoch(
        site_id=value.site_id, device_id=value.device_id, backend=value.backend,
        backend_version=value.backend_version, target_id=value.target_id,
        worker_build=value.worker_build, task_pack_digest=value.task_pack_digest,
        calibration_digest=value.calibration_digest, protocol_version="1",
    )
    assert value.epoch == legacy_epoch
    external = value.external_provenance(execution_protocol_version="3", request_digest="d" * 64, attempt_id="attempt-1")
    assert external["evaluator_epoch"] == value.external_epoch(execution_protocol_version="3")
    assert external["evaluator_epoch"] != value.epoch
    assert external["api_contract_id"] == "tcenv.remote.v1.evaluator-api.1"
    assert external["evidence_contract_id"] == "tcenv.remote.v1.evidence.1"


def test_service_identity_requires_a_registered_task_fingerprint():
    """@brief 生产身份不能为空 task fingerprint 集合，否则会接受任意任务。"""
    with pytest.raises(ValueError, match="task fingerprint"):
        ServiceIdentity(site_id="s", device_id="d", backend="b", backend_version="1", target_id="t", worker_build="w", task_pack_digest="p", calibration_digest="c", task_fingerprints=frozenset(), measurement_kind="measured")


@pytest.mark.parametrize("source,status", [
    (b"COMPILE_ERROR", "compile_error"),
    (b"INCORRECT", "incorrect"),
    (b"CRASH", "crash"),
])
def test_candidate_failures_are_completed_reports(service, source, status):
    req = request("candidate-" + status, source=source)
    submit(service, req)
    final = wait_state(service, req, {"completed"})
    assert final.failure is None
    assert final.result.report_json()["status"] == status
    with pytest.raises(JobError, match="interrupted"):
        retry(service, req, final.current_attempt.attempt_id, nonce="retry-" + status)


def test_capability_expiry_and_feature_negotiation(service):
    req = request("capability-expiry")
    cap = capabilities(service, valid_for_ms=1)
    time.sleep(0.01)
    with pytest.raises(JobError, match="re-hello") as caught:
        service.submit(
            request=req, expected=service.expected_identities,
            capability_id=cap.capability_id, grant_token=TOKEN,
        )
    assert caught.value.code == "failed_precondition"
    with pytest.raises(JobError, match="unsupported required") as caught:
        service.get_capabilities(
            grant_token=TOKEN, client_nonce="missing-feature-nonce",
            expected=service.expected_identities.evaluator.to_dict(),
            api_contract_ids=[service.expected_identities.api_contract_id],
            evidence_contract_ids=[service.expected_identities.evidence_contract_id],
            required_features=["not-implemented"],
        )
    assert caught.value.code == "failed_precondition"
    with pytest.raises(JobError) as caught:
        get(service, req)
    assert caught.value.code == "not_found"

    nonce = "one-time-client-nonce"
    service.get_capabilities(
        grant_token=TOKEN, client_nonce=nonce,
        expected=service.expected_identities.evaluator.to_dict(),
        api_contract_ids=[service.expected_identities.api_contract_id],
        evidence_contract_ids=[service.expected_identities.evidence_contract_id],
    )
    with pytest.raises(JobError, match="already used"):
        service.get_capabilities(
            grant_token=TOKEN, client_nonce=nonce,
            expected=service.expected_identities.evaluator.to_dict(),
            api_contract_ids=[service.expected_identities.api_contract_id],
            evidence_contract_ids=[service.expected_identities.evidence_contract_id],
        )
    feature_cap = service.get_capabilities(
        grant_token=TOKEN, client_nonce="feature-binding-nonce",
        expected=service.expected_identities.evaluator.to_dict(),
        api_contract_ids=[service.expected_identities.api_contract_id],
        evidence_contract_ids=[service.expected_identities.evidence_contract_id],
        required_features=["progress"],
    )
    with pytest.raises(JobError, match="required features differ"):
        service.submit(
            request=req, expected=service.expected_identities,
            capability_id=feature_cap.capability_id, required_features=[],
            grant_token=TOKEN,
        )


def test_grant_scope_isolation_covers_get_watch_cancel_and_retry(tmp_path):
    other = grant(
        OTHER_TOKEN, run_id="run-2", session_id="session-2", subject="subject-2",
    )
    service = engine(tmp_path / "state", grants=[grant(), other])
    try:
        req = request("private-job", source=b"SLEEP")
        submit(service, req)
        wait_state(service, req, {"running"})
        cap = capabilities(service, token=OTHER_TOKEN, nonce="other-scope-nonce")
        for operation in (
            lambda: get(service, req, token=OTHER_TOKEN),
            lambda: service.watch(
                request_id=req.request_id, request_digest=req.request_digest,
                expected=service.expected_identities, grant_token=OTHER_TOKEN,
            ),
            lambda: service.cancel(
                request_id=req.request_id, request_digest=req.request_digest,
                expected=service.expected_identities, grant_token=OTHER_TOKEN,
            ),
            lambda: service.retry(
                request_id=req.request_id, request_digest=req.request_digest,
                expected=service.expected_identities, interrupted_attempt_id="unknown",
                capability_id=cap.capability_id, grant_token=OTHER_TOKEN,
            ),
        ):
            with pytest.raises(JobError) as caught:
                operation()
            assert caught.value.code == "not_found"
        assert get(service, req).state == "running"
        service.cancel(
            request_id=req.request_id, request_digest=req.request_digest,
            expected=service.expected_identities, grant_token=TOKEN,
        )
        wait_state(service, req, {"cancelled"})
    finally:
        service.close()


def test_grant_renewal_reattaches_without_cancelling_active_attempt(tmp_path):
    old = grant(expires_at=time.time() + 0.15)
    renewed = grant("renewed-token", expires_at=time.time() + 60)
    service = engine(tmp_path / "state", grants=[old])
    try:
        req = request("renewal", source=b"FAIR_SLEEP")
        submit(service, req, nonce="old-grant-nonce")
        wait_state(service, req, {"running"})
        old_cap = capabilities(service, nonce="pre-renewal-cap")
        service.renew_grant(renewed)
        with pytest.raises(JobError) as caught:
            get(service, req)
        assert caught.value.code == "unauthenticated"
        final = wait_state(service, req, {"completed"}, token="renewed-token")
        assert final.state == "completed"
        # A capability from the superseded grant generation cannot authorize Submit/Retry.
        with pytest.raises(JobError) as caught:
            service.submit(
                request=req, expected=service.expected_identities,
                capability_id=old_cap.capability_id, grant_token="renewed-token",
            )
        assert caught.value.code in {"unauthenticated", "failed_precondition"}
        replay = submit(service, req, token="renewed-token", nonce="post-renewal-submit")
        assert replay.result.receipt == final.result.receipt
    finally:
        service.close()


def test_revocation_cancels_only_matching_active_scope(tmp_path):
    selfcheck = grant(kinds=frozenset({"selfcheck"}), modes=frozenset({"quick", "timed"}))
    formal = grant(
        OTHER_TOKEN, subject="formal-subject", kinds=frozenset({"formal_eval"}),
        modes=frozenset({"quick", "full"}),
    )
    service = engine(tmp_path / "state", grants=[selfcheck, formal])
    try:
        first = request("revoke-selfcheck", source=b"SLEEP", kind="selfcheck", mode="quick")
        second = request("formal-survives", source=b"OK")
        submit(service, first)
        wait_state(service, first, {"running"})
        submit(service, second, token=OTHER_TOKEN, nonce="formal-submit-nonce")
        service.revoke_grant(TOKEN)
        assert wait_state(service, second, {"completed"}, token=OTHER_TOKEN).state == "completed"
        assert service.store._job_by_id(first.request_id).state == "cancelled"
        with pytest.raises(JobError) as caught:
            get(service, first)
        assert caught.value.code == "permission_denied"
    finally:
        service.close()


def test_selfcheck_infrastructure_refund_and_candidate_accounting(tmp_path):
    service = engine(tmp_path / "state", selfcheck_max_timed=1)
    try:
        broken = request("infra-refund", source=b"EXIT", kind="selfcheck", mode="timed")
        submit(service, broken)
        interrupted = wait_state(service, broken, {"interrupted"})
        usage = service.store.quota_usage(broken)
        assert usage.used_count == usage.reserved_count == usage.used_gpu_us == 0
        assert not usage.hang
        retry(service, broken, interrupted.current_attempt.attempt_id)
        # The same logical job can reserve again after an infrastructure refund.
        wait_state(service, broken, {"interrupted"})
        assert service.store.quota_usage(broken).used_count == 0

        good = request("accounted", source=b"OK", kind="selfcheck", mode="timed")
        submit(service, good, nonce="good-selfcheck")
        final = wait_state(service, good, {"completed"})
        usage = service.store.quota_usage(good)
        assert usage.used_count == 1 and usage.used_gpu_us == 500_000
        assert final.result.receipt["attempt_id"] == final.current_attempt.attempt_id
        exhausted = request("exhausted", source=b"OK", kind="selfcheck", mode="timed")
        with pytest.raises(JobError, match="count quota") as caught:
            submit(service, exhausted, nonce="exhausted-selfcheck")
        assert caught.value.code == "resource_exhausted"
    finally:
        service.close()


def test_selfcheck_gpu_quota_and_hang_blocking(tmp_path):
    service = engine(tmp_path / "state", selfcheck_max_gpu_s=0.25)
    try:
        first = request("gpu-seconds", source=b"OK", kind="selfcheck", mode="timed")
        submit(service, first)
        wait_state(service, first, {"completed"})
        with pytest.raises(JobError, match="GPU-time quota"):
            submit(service, request("gpu-next", kind="selfcheck", mode="timed"), nonce="gpu-next")
    finally:
        service.close()

    service = engine(tmp_path / "hang-state")
    try:
        hanging = request("candidate-hang", source=b"SLEEP", kind="selfcheck", mode="timed", candidate_timeout_ms=50)
        submit(service, hanging)
        final = wait_state(service, hanging, {"completed"})
        assert final.result.report_json()["status"] == "timeout"
        assert service.store.quota_usage(hanging).hang
        with pytest.raises(JobError, match="blocked after a candidate hang"):
            submit(service, request("after-hang", kind="selfcheck", mode="timed"), nonce="after-hang")
    finally:
        service.close()


def test_formal_priority_allows_aged_cross_run_selfcheck(tmp_path):
    order = tmp_path / "order.jsonl"
    other = grant(OTHER_TOKEN, run_id="run-2", session_id="session-2", subject="subject-2")
    service = engine(
        tmp_path / "state", grants=[grant(), other], selfcheck_age_s=0.1,
        worker_env={"FAKE_JOB_ORDER": str(order)},
    )
    try:
        first = request("formal-first", source=b"FAIR_SLEEP")
        aged = request("selfcheck-aged", kind="selfcheck", mode="quick", run_id="run-2", session_id="session-2")
        later = request("formal-later", source=b"OK")
        submit(service, first)
        wait_state(service, first, {"running"})
        submit(service, aged, token=OTHER_TOKEN, nonce="aged-submit")
        time.sleep(0.16)
        submit(service, later, nonce="later-formal")
        wait_state(service, first, {"completed"})
        wait_state(service, aged, {"completed"}, token=OTHER_TOKEN)
        wait_state(service, later, {"completed"})
        rows = [json.loads(line) for line in order.read_text(encoding="utf-8").splitlines()]
        assert [row["request_id"] for row in rows] == ["formal-first", "selfcheck-aged", "formal-later"]
    finally:
        service.close()


def test_progress_resume_attempt_filter_and_snapshot_fallback(tmp_path):
    service = engine(tmp_path / "state", event_window=3)
    try:
        req = request("progress", source=b"PROGRESS MANY")
        submitted = submit(service, req)
        final = wait_state(service, req, {"completed"})
        events = service.watch(
            request_id=req.request_id, request_digest=req.request_digest,
            expected=service.expected_identities, grant_token=TOKEN,
        )
        assert len(events) <= 3
        assert [event.sequence for event in events] == sorted(event.sequence for event in events)
        fallback = service.watch(
            request_id=req.request_id, request_digest=req.request_digest,
            expected=service.expected_identities, grant_token=TOKEN, after_sequence=1,
        )
        assert len(fallback) == 1 and fallback[0].snapshot.state == "completed"
        assert fallback[0].sequence == final.last_sequence
        resumed = service.watch(
            request_id=req.request_id, request_digest=req.request_digest,
            expected=service.expected_identities, grant_token=TOKEN,
            after_sequence=final.last_sequence - 1,
            attempt_id=submitted.current_attempt.attempt_id,
        )
        assert all(event.attempt_id == submitted.current_attempt.attempt_id for event in resumed)
        assert all("path" not in (event.progress or {}) for event in resumed)
    finally:
        service.close()


def test_watch_attempt_filter_never_coalesces_another_attempt(tmp_path):
    """@brief 事件窗口越过旧 attempt 时，旧 attempt 的 Watch 不能收到新 attempt 的快照。"""
    service = engine(tmp_path / "state", event_window=2)
    try:
        req = request("filter-gap", source=b"FAIL_FIRST")
        submit(service, req)
        interrupted = wait_state(service, req, {"interrupted"})
        old_attempt = interrupted.current_attempt
        retry(service, req, old_attempt.attempt_id, nonce="fresh-attempt-nonce")
        final = wait_state(service, req, {"completed"})
        assert final.current_attempt.attempt_id != old_attempt.attempt_id
        stale = service.watch(
            request_id=req.request_id, request_digest=req.request_digest,
            expected=service.expected_identities, grant_token=TOKEN,
            after_sequence=0, attempt_id=old_attempt.attempt_id,
        )
        assert stale == []
        current = service.watch(
            request_id=req.request_id, request_digest=req.request_digest,
            expected=service.expected_identities, grant_token=TOKEN,
            after_sequence=0, attempt_id=final.current_attempt.attempt_id,
        )
        assert len(current) == 1
        assert current[0].snapshot.state == "completed"
        assert current[0].attempt_id == final.current_attempt.attempt_id
    finally:
        service.close()


def test_cancel_kills_worker_group_and_orphan(tmp_path):
    running_marker = tmp_path / "running.json"
    orphan_log = tmp_path / "orphans.jsonl"
    service = engine(tmp_path / "state", worker_env={
        "FAKE_JOB_RUNNING": str(running_marker), "FAKE_JOB_ORPHAN_PIDS": str(orphan_log),
    })
    try:
        req = request("cancel-group", source=b"ORPHAN SLEEP", candidate_timeout_ms=30_000)
        submit(service, req)
        wait_state(service, req, {"running"})
        deadline = time.monotonic() + 5
        orphan_lines: list[str] = []
        while time.monotonic() < deadline:
            try:
                orphan_lines = orphan_log.read_text(encoding="utf-8").splitlines()
            except FileNotFoundError:
                orphan_lines = []
            if orphan_lines:
                break
            time.sleep(0.02)
        if not orphan_lines:
            pytest.fail("orphan worker did not record its process ID")
        orphan_pid = json.loads(orphan_lines[0])["pid"]
        service.cancel(
            request_id=req.request_id, request_digest=req.request_digest,
            expected=service.expected_identities, grant_token=TOKEN,
        )
        final = wait_state(service, req, {"cancelled"})
        assert final.result is None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(orphan_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"orphan process {orphan_pid} survived process-group cancellation")
    finally:
        service.close()


def test_duplicate_result_bytes_are_deduplicated_but_conflicts_are_data_loss(service):
    req = request("receipt-dedup")
    submit(service, req)
    final = wait_state(service, req, {"completed"})
    attempt = final.current_attempt
    report = final.result.report_json()
    evidence = final.result.evidence_json()
    replay = service.store.complete_attempt(req.request_id, attempt, report=report, evidence=evidence)
    assert replay.result.receipt == final.result.receipt
    with pytest.raises(StateCorrupt, match="different full result envelope"):
        service.store.complete_attempt(
            req.request_id, attempt, report={**report, "candidate_us": 999.0}, evidence=evidence,
        )


def test_task_fingerprint_must_bind_task_dtype_and_size(tmp_path):
    """@brief 已登记 fingerprint 不能被另一个 task/dtype/size 重用。"""
    service = engine(tmp_path / "state")
    try:
        with pytest.raises(JobError, match="task fingerprint"):
            submit(service, request("wrong-task", task="gemm"))
        with pytest.raises(JobError, match="task fingerprint"):
            submit(service, request("wrong-dtype", dtype="fp32"))
        with pytest.raises(JobError, match="task fingerprint"):
            submit(service, request("wrong-size", size="paper"))
        assert submit(service, request("right-task")).state in {"accepted", "queued", "running", "completed"}
    finally:
        service.close()


def test_formal_selfcheck_and_mode_scopes_cannot_be_escalated(tmp_path):
    scoped = grant(
        kinds=frozenset({"selfcheck"}), modes=frozenset({"quick"}),
    )
    service = engine(tmp_path / "state", grants=[scoped])
    try:
        for req in (
            request("formal-escalation", kind="formal_eval", mode="quick"),
            request("timed-escalation", kind="selfcheck", mode="timed"),
        ):
            cap = capabilities(service)
            with pytest.raises(JobError) as caught:
                service.submit(
                    request=req, expected=service.expected_identities,
                    capability_id=cap.capability_id, grant_token=TOKEN,
                )
            assert caught.value.code == "permission_denied"
            with pytest.raises(JobError) as caught:
                get(service, req)
            assert caught.value.code == "not_found"
    finally:
        service.close()


def test_bounded_clock_skew_widens_only_grant_and_capability_expiry(tmp_path):
    """@brief 有限时钟偏移只放宽 grant/capability 临期，不放宽撤销与 generation。"""
    now = 1_800_000_000.0
    expired = grant(expires_at=now - 1.0)
    service = engine(tmp_path / "skewed", grants=[expired], clock=lambda: now, clock_skew_ms=5_000)
    try:
        capabilities(service, nonce="skewed-grant-nonce")
    except JobError as exc:  # pragma: no cover - asserted for a clear failure
        pytest.fail(f"bounded skew should accept a grant expired by one second: {exc}")
    finally:
        service.close()

    strict = engine(tmp_path / "strict", grants=[expired], clock=lambda: now)
    try:
        with pytest.raises(JobError, match="expired"):
            capabilities(strict, nonce="strict-grant-nonce")
    finally:
        strict.close()

    revoked = engine(tmp_path / "revoked", grants=[expired], clock=lambda: now, clock_skew_ms=5_000)
    try:
        revoked.revoke_grant(expired.token)
        with pytest.raises(JobError, match="revoked"):
            capabilities(revoked, nonce="revoked-grant-nonce")
    finally:
        revoked.close()


def test_plain_grant_expiry_does_not_stop_accepted_job_and_renewal_recovers_read(tmp_path):
    expiring = grant(expires_at=time.time() + 0.12)
    service = engine(tmp_path / "state", grants=[expiring])
    try:
        req = request("grant-expiry", source=b"FAIR_SLEEP")
        submit(service, req)
        wait_state(service, req, {"running"})
        time.sleep(0.15)
        with pytest.raises(JobError, match="expired") as caught:
            get(service, req)
        assert caught.value.code == "unauthenticated"
        service.renew_grant(grant("next-after-expiry", expires_at=time.time() + 60))
        assert wait_state(service, req, {"completed"}, token="next-after-expiry").state == "completed"
    finally:
        service.close()


def test_timed_reservation_is_visible_then_cancel_refunds_without_authority_result(tmp_path):
    service = engine(tmp_path / "state")
    try:
        req = request("reserved-then-cancel", source=b"SLEEP", kind="selfcheck", mode="timed")
        submit(service, req)
        wait_state(service, req, {"running"})
        usage = service.store.quota_usage(req)
        assert usage.reserved_count == 1 and usage.reserved_gpu_us > 0
        service.cancel(
            request_id=req.request_id, request_digest=req.request_digest,
            expected=service.expected_identities, grant_token=TOKEN,
        )
        final = wait_state(service, req, {"cancelled"})
        assert final.result is None and final.cancel_requested
        usage = service.store.quota_usage(req)
        assert usage == type(usage)(0, 0, 0, 0, False)
    finally:
        service.close()


def test_failed_is_terminal_and_cannot_be_retried_or_overwritten(tmp_path):
    from tcenv.remote.job_types import JobFailure

    service = engine(tmp_path / "state")
    try:
        req = request("terminal-failed", source=b"FAIR_SLEEP")
        submit(service, req)
        running = wait_state(service, req, {"running"})
        attempt = running.current_attempt
        assert service.store.fail_attempt(
            req.request_id, attempt,
            JobFailure("policy", False, "device_policy", "device was administratively disabled"),
        )
        service._cancel_active(req.request_id)
        final = wait_state(service, req, {"failed"})
        assert final.result is None and final.failure.code == "device_policy"
        with pytest.raises(JobError, match="current interrupted"):
            retry(service, req, attempt.attempt_id)
        replay = submit(service, req, nonce="failed-replay")
        assert replay.state == "failed" and replay.failure == final.failure
    finally:
        service.close()


def test_external_result_signature_uses_v1_attempt_receipt_and_blob_descriptors(tmp_path):
    from tcenv.remote import contracts

    key = b"external-test-signing-key-32-bytes!"
    service = engine(
        tmp_path / "state", signing_key=key, signing_key_id="sim-external-site/key-1",
    )
    try:
        req = request("signed-result")
        submit(service, req)
        final = wait_state(service, req, {"completed"})
        result = final.result
        document = {
            "provenance": result.provenance, "report": result.report_descriptor,
            "evidence": result.evidence_descriptor, "receipt": result.receipt,
        }
        assert result.signature["algorithm"] == "hmac-sha256-v1"
        assert contracts.verify_result_signature(
            document, key_id="sim-external-site/key-1", key=key,
            signature=result.signature["signature"],
        )
        assert result.receipt["receipt_id"] == contracts.attempt_receipt_id(
            request_id=req.request_id, attempt_id=final.current_attempt.attempt_id,
            request_digest=req.request_digest,
        )
        assert capabilities(service).signing_key_id == "sim-external-site/key-1"
    finally:
        service.close()


def test_cancel_between_result_precheck_and_commit_finishes_atomically(tmp_path):
    service = engine(tmp_path / "state")
    try:
        req = request("cancel-commit-race")
        original = service.store.complete_attempt
        injected = False

        def complete_with_cancel(*args, **kwargs):
            nonlocal injected
            if not injected:
                injected = True
                # _run_worker has already made its pre-complete cancellation
                # check.  Commit Cancel immediately before the result transaction.
                service.store.cancel(
                    grant_token=TOKEN, request_id=req.request_id,
                    request_digest=req.request_digest,
                    expected=service.expected_identities,
                )
            return original(*args, **kwargs)

        service.store.complete_attempt = complete_with_cancel
        submit(service, req)
        final = wait_state(service, req, {"cancelled"})
        assert final.result is None and final.current_attempt.state == "cancelled"
    finally:
        service.close()


def test_defensive_post_complete_running_cancel_snapshot_cannot_wedge_queue(tmp_path):
    service = engine(tmp_path / "state")
    try:
        req = request("cancel-defensive-post-complete")
        original = service.store.complete_attempt

        def legacy_running_response(*_args, **_kwargs):
            service.store.cancel(
                grant_token=TOKEN, request_id=req.request_id,
                request_digest=req.request_digest, expected=service.expected_identities,
            )
            return service.store._job_by_id(req.request_id)

        service.store.complete_attempt = legacy_running_response
        submit(service, req)
        assert wait_state(service, req, {"cancelled"}).state == "cancelled"
        service.store.complete_attempt = original
        good = request("queue-progresses-after-defensive-cancel")
        submit(service, good)
        assert wait_state(service, good, {"completed"}).state == "completed"
    finally:
        service.close()


def test_result_db_budget_rejects_large_blob_and_leaves_bounded_terminal_failure(tmp_path):
    service = engine(tmp_path / "state", max_db_bytes=1_000_000)
    try:
        req = request("large-result-budget", source=b"LARGE_RESULT")
        submit(service, req)
        final = wait_state(service, req, {"failed"})
        assert final.failure.code == "resource_exhausted" and not final.failure.retryable
        assert final.result is None
        assert service.store.database_bytes() <= 1_000_000
        assert service.store._conn.execute(
            "SELECT COUNT(*) FROM results WHERE request_id=?", (req.request_id,),
        ).fetchone()[0] == 0
        # A policy terminal failure does not wedge the queue, and a small
        # durable completion still respects the same advertised byte budget.
        good = request("small-result-within-budget")
        submit(service, good)
        assert wait_state(service, good, {"completed"}).state == "completed"
        assert service.store.database_bytes() <= 1_000_000
    finally:
        service.close()


def test_db_budget_covers_capability_and_progress_growth(tmp_path):
    service = engine(tmp_path / "state")
    try:
        req = request("progress-budget", source=b"SLEEP")
        submit(service, req)
        running = wait_state(service, req, {"running"})
        service.store.max_db_bytes = service.store.database_bytes() + 1
        with pytest.raises(JobError) as caught:
            capabilities(service)
        assert caught.value.code == "resource_exhausted"
        with pytest.raises(JobError) as caught:
            service.store.record_progress(
                req.request_id, running.current_attempt,
                {"stage": "compile", "detail": "bounded"},
            )
        assert caught.value.code == "resource_exhausted"
        service.cancel(
            request_id=req.request_id, request_digest=req.request_digest,
            expected=service.expected_identities, grant_token=TOKEN,
        )
        wait_state(service, req, {"cancelled"})
    finally:
        service.close()


@pytest.mark.parametrize("tamper", ["request_json", "source"])
def test_tampered_queued_request_is_quarantined_without_stopping_scheduler(tmp_path, tamper):
    service = engine(tmp_path / "state")
    try:
        blocker = request("digest-blocker", source=b"FAIR_SLEEP")
        bad = request("digest-tampered-" + tamper)
        good = request("digest-scheduler-survives")
        submit(service, blocker)
        wait_state(service, blocker, {"running"})
        submit(service, bad)
        with service.store._write() as conn:
            if tamper == "source":
                conn.execute("UPDATE jobs SET source=? WHERE request_id=?", (b"tampered source", bad.request_id))
            else:
                raw = conn.execute("SELECT request_json FROM jobs WHERE request_id=?", (bad.request_id,)).fetchone()[0]
                document = json.loads(raw)
                document["reps"] += 1
                conn.execute(
                    "UPDATE jobs SET request_json=? WHERE request_id=?",
                    (json.dumps(document, sort_keys=True, separators=(",", ":")), bad.request_id),
                )
        submit(service, good)
        quarantined = wait_state(service, bad, {"interrupted"})
        assert quarantined.quarantined and "digest/source" in quarantined.recovery_error
        assert quarantined.failure.code == "data_loss" and not quarantined.failure.retryable
        assert wait_state(service, good, {"completed"}).state == "completed"
        assert wait_state(service, blocker, {"completed"}).state == "completed"
    finally:
        service.close()


@pytest.mark.parametrize("phase", ["queued", "running"])
def test_cancel_with_tampered_source_finishes_cancelled_not_data_loss(tmp_path, phase):
    service = engine(tmp_path / "state")
    blocker = None
    try:
        if phase == "queued":
            blocker = request("cancel-corrupt-source-blocker", source=b"SLEEP", candidate_timeout_ms=30_000)
            submit(service, blocker)
            wait_state(service, blocker, {"running"})
        req = request(
            "cancel-corrupt-source-" + phase,
            source=b"SLEEP" if phase == "running" else b"OK",
            candidate_timeout_ms=30_000,
        )
        submit(service, req)
        if phase == "running":
            wait_state(service, req, {"running"})
        else:
            assert get(service, req).state == "queued"
        with service.store._write() as conn:
            conn.execute(
                "UPDATE jobs SET source=? WHERE request_id=?",
                (b"tampered after submission", req.request_id),
            )

        cancelled = service.cancel(
            request_id=req.request_id, request_digest=req.request_digest,
            expected=service.expected_identities, grant_token=TOKEN,
        )
        if phase == "running":
            assert cancelled.state == "running" and cancelled.cancel_requested
            cancelled = wait_state(service, req, {"cancelled"})
        assert cancelled.state == "cancelled" and cancelled.cancel_requested
        assert cancelled.current_attempt.state == "cancelled"
        assert cancelled.failure is not None and cancelled.failure.code == "cancelled"
        assert not cancelled.quarantined
    finally:
        try:
            if blocker is not None:
                service.cancel(
                    request_id=blocker.request_id, request_digest=blocker.request_digest,
                    expected=service.expected_identities, grant_token=TOKEN,
                )
        finally:
            service.close()


def test_fatal_scheduler_store_error_makes_submit_and_get_unavailable(tmp_path):
    service = engine(tmp_path / "state")
    try:
        req = request("fatal-store")
        cap = capabilities(service)

        def fatal_claim(**_kwargs):
            raise StateCorrupt("fatal scheduler storage failure")

        service.store.claim_next = fatal_claim
        deadline = time.monotonic() + 3
        while not service.store.unavailable and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.store.unavailable
        with pytest.raises(JobError) as caught:
            get(service, req)
        assert caught.value.code == "unavailable"
        with pytest.raises(JobError) as caught:
            service.submit(
                request=req, expected=service.expected_identities,
                capability_id=cap.capability_id, grant_token=TOKEN,
            )
        assert caught.value.code == "unavailable"
    finally:
        service.close()


def test_duplicate_completion_replays_the_committed_key_envelope(tmp_path):
    """@brief 重复完成只回放已提交签名，不会用轮换后的新 key 重签历史结果。"""
    key1 = b"external-test-signing-key-32-bytes!"
    key2 = b"another-test-signing-key-32-bytes!"
    service = engine(tmp_path / "state", signing_key=key1, signing_key_id="sim-external-site/key-1")
    try:
        req = request("signed-replay-key-mismatch")
        submit(service, req)
        final = wait_state(service, req, {"completed"})
        replay = service.store.complete_attempt(
            req.request_id, final.current_attempt,
            report=final.result.report_json(), evidence=final.result.evidence_json(),
            signing_key=key2, signing_key_id="sim-external-site/key-2",
        )
        assert replay.result.signature == final.result.signature
        assert replay.result.signature["key_id"] == "sim-external-site/key-1"
        with pytest.raises(StateCorrupt, match="full result envelope"):
            service.store.complete_attempt(
                req.request_id, final.current_attempt,
                report={**final.result.report_json(), "candidate_us": 999.0},
                evidence=final.result.evidence_json(),
            )
    finally:
        service.close()


def test_engine_result_envelope_conflict_on_completed_job_fails_closed(tmp_path):
    from tcenv.remote.job_engine import _Running

    key1 = b"external-test-signing-key-32-bytes!"
    key2 = b"another-test-signing-key-32-bytes!"
    service = engine(tmp_path / "state", signing_key=key1, signing_key_id="sim-external-site/key-1")
    try:
        req = request("signed-replay-engine-fail-closed")
        submit(service, req)
        final = wait_state(service, req, {"completed"})
        report = {**final.result.report_json(), "candidate_us": 999.0}
        evidence = final.result.evidence_json()
        with pytest.raises(StateCorrupt, match="full result envelope"):
            service.store.complete_attempt(
                req.request_id, final.current_attempt, report=report, evidence=evidence,
                signing_key=key2, signing_key_id="sim-external-site/key-2",
            )
        assert not service.store.unavailable

        service._signing_key = key2
        service._signing_key_id = "sim-external-site/key-2"
        service._complete_worker_result(
            _Running(req, final.current_attempt), report=report, evidence=evidence,
        )
        assert service.store.unavailable
        assert service._stop.is_set()
        with pytest.raises(JobError) as caught:
            get(service, req)
        assert caught.value.code == "unavailable"
    finally:
        service.close()


def test_unsigned_result_allows_null_evidence_with_a_valid_envelope(service):
    req = request("unsigned-null-evidence", source=b"NO_EVIDENCE")
    submit(service, req)
    final = wait_state(service, req, {"completed"})
    assert final.result.evidence is None and final.result.evidence_descriptor is None
    assert final.result.signature is None
    assert final.result.receipt["candidate_sha256"] == req.candidate_sha256
    assert capabilities(service).signing_key_id == ""
    provenance = dict(final.result.provenance)
    del provenance["attempt_id"]
    with service.store._write() as conn:
        conn.execute(
            "UPDATE results SET provenance_json=? WHERE request_id=?",
            (json.dumps(provenance, sort_keys=True, separators=(",", ":")), req.request_id),
        )
    with pytest.raises(StateCorrupt, match="evidence contract"):
        get(service, req)


def test_watch_reconstructs_event_time_snapshots(tmp_path):
    service = engine(tmp_path / "state", event_window=16)
    try:
        req = request("historical-snapshots", source=b"PROGRESS FAIR_SLEEP")
        submit(service, req)
        final = wait_state(service, req, {"completed"})
        events = service.watch(
            request_id=req.request_id, request_digest=req.request_digest,
            expected=service.expected_identities, grant_token=TOKEN,
        )
        snapshots = [event.snapshot for event in events if event.snapshot is not None]
        assert [snapshot.state for snapshot in snapshots] == ["accepted", "queued", "running", "completed"]
        accepted, queued, running, completed = snapshots
        assert accepted.current_attempt is None
        attempt_id = queued.current_attempt.attempt_id
        fence = queued.current_attempt.lease_fence
        assert running.current_attempt.attempt_id == completed.current_attempt.attempt_id == attempt_id
        assert running.current_attempt.lease_fence == completed.current_attempt.lease_fence == fence
        assert accepted.result is queued.result is running.result is None
        assert completed.result is not None and completed.result.receipt["receipt_id"] == final.result.receipt["receipt_id"]
        assert running.current_attempt.started_at_ms is not None
        assert completed.current_attempt.finished_at_ms is not None
        snapshot_events = [event for event in events if event.snapshot is not None]
        assert any(event.progress is not None for event in events)
        assert len(snapshots) == len(snapshot_events)
        assert all(snapshot.last_sequence == event.sequence for snapshot, event in zip(snapshots, snapshot_events))
    finally:
        service.close()


def test_cross_scope_submit_with_another_digest_does_not_leak_request_id(tmp_path):
    other = grant(OTHER_TOKEN, subject="other-subject")
    service = engine(tmp_path / "state", grants=[grant(), other])
    try:
        req = request("cross-scope-idempotency")
        submit(service, req)
        wait_state(service, req, {"completed"})
        changed = request(req.request_id, source=b"different invisible payload")
        with pytest.raises(JobError) as caught:
            submit(service, changed, token=OTHER_TOKEN)
        assert caught.value.code == "not_found"
    finally:
        service.close()


def test_cancel_after_claim_before_active_attach_prevents_worker_request(tmp_path):
    startup_marker = tmp_path / "worker-started"
    request_marker = tmp_path / "worker-received-request"
    service = engine(tmp_path / "state", worker_env={
        "FAKE_JOB_STARTED": str(startup_marker), "FAKE_JOB_RUNNING": str(request_marker),
    })
    popen_calls = []
    original_popen = service._popen

    def recording_popen():
        popen_calls.append(True)
        return original_popen()

    service._popen = recording_popen
    try:
        req = request("cancel-claim-attach-race", source=b"SLEEP", candidate_timeout_ms=30_000)
        original = service.store.claim_next
        injected = False

        def claim_and_cancel(**kwargs):
            nonlocal injected
            claimed = original(**kwargs)
            if claimed is not None and claimed[0].request_id == req.request_id and not injected:
                injected = True
                service.store.cancel(
                    grant_token=TOKEN, request_id=req.request_id,
                    request_digest=req.request_digest, expected=service.expected_identities,
                )
            return claimed

        service.store.claim_next = claim_and_cancel
        submit(service, req)
        wait_state(service, req, {"cancelled"})
        time.sleep(0.1)
        assert injected and not popen_calls
        assert not startup_marker.exists() and not request_marker.exists()
    finally:
        service.close()


def test_cancel_after_attach_kills_without_starting_request_writer(tmp_path, monkeypatch):
    import tcenv.remote.job_engine as job_engine_module

    startup_marker = tmp_path / "worker-started"
    request_marker = tmp_path / "worker-received-request"
    service = engine(tmp_path / "state", worker_env={
        "FAKE_JOB_STARTED": str(startup_marker), "FAKE_JOB_RUNNING": str(request_marker),
    })
    encoded = []
    original_encode = job_engine_module.encode_request

    def recording_encode(*args, **kwargs):
        encoded.append(True)
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(job_engine_module, "encode_request", recording_encode)
    try:
        req = request("cancel-after-attach", source=b"SLEEP", candidate_timeout_ms=30_000)
        original_attach = service.store.attach_worker

        def attach_and_cancel(*args, **kwargs):
            attached = original_attach(*args, **kwargs)
            deadline = time.monotonic() + 3
            while not startup_marker.is_file() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert startup_marker.is_file(), "worker did not reach its pre-stdin startup marker"
            service.store.cancel(
                grant_token=TOKEN, request_id=req.request_id,
                request_digest=req.request_digest, expected=service.expected_identities,
            )
            return attached

        service.store.attach_worker = attach_and_cancel
        submit(service, req)
        assert wait_state(service, req, {"cancelled"}).state == "cancelled"
        assert encoded == [], "cancellation must return before starting the request writer"
        assert not request_marker.exists()
    finally:
        service.close()


def test_close_during_blocked_popen_defers_store_close_and_never_writes_request(tmp_path, monkeypatch):
    import tcenv.remote.job_engine as job_engine_module

    request_marker = tmp_path / "worker-received-request"
    service = engine(tmp_path / "state", worker_env={"FAKE_JOB_RUNNING": str(request_marker)})
    req = request("close-during-popen", source=b"SLEEP", candidate_timeout_ms=30_000)
    popen_entered = threading.Event()
    release_popen = threading.Event()
    store_closed = threading.Event()
    processes = []
    encoded = []
    close_scheduler_states = []
    final_snapshots = []
    original_popen = service._popen
    original_encode = job_engine_module.encode_request
    original_store_close = service.store.close

    def blocking_popen():
        popen_entered.set()
        if not release_popen.wait(timeout=10):
            raise TimeoutError("test did not release blocked worker startup")
        process = original_popen()
        processes.append(process)
        return process

    def recording_encode(*args, **kwargs):
        encoded.append(True)
        return original_encode(*args, **kwargs)

    def recording_store_close():
        close_scheduler_states.append(service._scheduler.is_alive())
        final_snapshots.append(service.store._job_by_id(req.request_id))
        original_store_close()
        store_closed.set()

    service._popen = blocking_popen
    service.store.close = recording_store_close
    monkeypatch.setattr(job_engine_module, "encode_request", recording_encode)
    try:
        submit(service, req)
        assert popen_entered.wait(timeout=3), "scheduler did not claim the job"
        started = time.monotonic()
        service.close()
        assert time.monotonic() - started < 5, "close waited for the blocked worker startup"
        assert service._scheduler.is_alive()
        assert close_scheduler_states == [], "store closed while the scheduler was still alive"
        service.close()  # A repeated close must not close the store early or start another finalizer.
        assert close_scheduler_states == []
        assert encoded == [] and not request_marker.exists()

        release_popen.set()
        assert store_closed.wait(timeout=8), "deferred store finalizer did not finish"
        service._scheduler.join(timeout=1)
        assert not service._scheduler.is_alive()
        assert close_scheduler_states == [False]
        assert len(processes) == 1 and processes[0].poll() is not None
        assert encoded == [] and not request_marker.exists()
        assert len(final_snapshots) == 1
        final = final_snapshots[0]
        assert final.state == "interrupted" and not final.quarantined
        assert final.failure is not None and final.failure.retryable
        assert final.failure.code == "frontend_stopped"
    finally:
        release_popen.set()
        service.close()
        for process in processes:
            if process.poll() is None:
                job_engine_module.kill_group(process)
            job_engine_module.forget(process)
        service._scheduler.join(timeout=8)
        if not store_closed.is_set():
            original_store_close()


def test_supervisor_error_after_popen_cleans_unattached_worker(tmp_path, monkeypatch):
    import tcenv.remote.job_engine as job_engine_module

    request_marker = tmp_path / "worker-received-request"
    service = engine(tmp_path / "state", worker_env={"FAKE_JOB_RUNNING": str(request_marker)})
    processes = []
    encoded = []
    original_popen = service._popen
    original_encode = job_engine_module.encode_request

    def recording_popen():
        process = original_popen()
        processes.append(process)
        return process

    def fail_process_identity(*_args, **_kwargs):
        raise RuntimeError("injected supervisor failure after popen")

    def recording_encode(*args, **kwargs):
        encoded.append(True)
        return original_encode(*args, **kwargs)

    service._popen = recording_popen
    monkeypatch.setattr(job_engine_module, "process_start_token", fail_process_identity)
    monkeypatch.setattr(job_engine_module, "encode_request", recording_encode)
    try:
        req = request("supervisor-error-after-popen", source=b"SLEEP", candidate_timeout_ms=30_000)
        submit(service, req)
        final = wait_state(service, req, {"interrupted"})
        assert final.failure is not None and final.failure.retryable
        assert final.failure.code == "worker_supervisor_failed"
        assert "injected supervisor failure" in final.failure.message
        assert len(processes) == 1 and processes[0].poll() is not None
        assert all(
            stream is None or stream.closed
            for stream in (processes[0].stdin, processes[0].stdout, processes[0].stderr)
        )
        assert encoded == [] and not request_marker.exists()
    finally:
        service.close()
        for process in processes:
            if process.poll() is None:
                job_engine_module.kill_group(process)
            job_engine_module.forget(process)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()


def test_supervisor_error_with_uncleaned_worker_quarantines_instead_of_interrupting(tmp_path, monkeypatch):
    import tcenv.remote.job_engine as job_engine_module

    request_marker = tmp_path / "worker-received-request"
    service = engine(tmp_path / "state", worker_env={"FAKE_JOB_RUNNING": str(request_marker)})
    processes = []
    encoded = []
    original_popen = service._popen
    original_kill = job_engine_module.kill_group
    original_encode = job_engine_module.encode_request

    def recording_popen():
        process = original_popen()
        processes.append(process)
        return process

    def failing_kill(process):
        raise OSError("injected kill failure")

    def fail_process_identity(*_args, **_kwargs):
        raise RuntimeError("injected supervisor failure after popen")

    def recording_encode(*args, **kwargs):
        encoded.append(True)
        return original_encode(*args, **kwargs)

    service._popen = recording_popen
    monkeypatch.setattr(job_engine_module, "kill_group", failing_kill)
    monkeypatch.setattr(job_engine_module, "process_start_token", fail_process_identity)
    monkeypatch.setattr(job_engine_module, "encode_request", recording_encode)
    try:
        req = request("supervisor-error-uncleaned", source=b"SLEEP", candidate_timeout_ms=30_000)
        submit(service, req)
        deadline = time.monotonic() + 5
        snapshot = service.store._job_by_id(req.request_id)
        while not snapshot.quarantined and time.monotonic() < deadline:
            time.sleep(0.01)
            snapshot = service.store._job_by_id(req.request_id)
        assert snapshot.quarantined is True
        assert snapshot.state == "running"
        assert "injected kill failure" in snapshot.recovery_error
        assert snapshot.result is None
        assert encoded == [] and not request_marker.exists()
        # Retry is refused while the old worker's death is unproven.
        cap = capabilities(service)
        with pytest.raises(JobError, match="quarantined"):
            service.retry(
                request_id=req.request_id, request_digest=req.request_digest,
                expected=service.expected_identities,
                interrupted_attempt_id=snapshot.current_attempt.attempt_id,
                capability_id=cap.capability_id, grant_token=TOKEN,
            )
    finally:
        monkeypatch.setattr(job_engine_module, "kill_group", original_kill)
        for process in processes:
            if process.poll() is None:
                original_kill(process)
            job_engine_module.forget(process)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        service.close()


def test_attach_failure_never_writes_request_and_is_recoverable(tmp_path, monkeypatch):
    import tcenv.remote.job_engine as job_engine_module

    request_marker = tmp_path / "worker-received-request"
    service = engine(tmp_path / "state", worker_env={"FAKE_JOB_RUNNING": str(request_marker)})
    encoded = []
    original_encode = job_engine_module.encode_request

    def recording_encode(*args, **kwargs):
        encoded.append(True)
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(job_engine_module, "encode_request", recording_encode)
    original_attach = service.store.attach_worker
    service.store.attach_worker = lambda *_args, **_kwargs: False
    try:
        req = request("attach-fence-failure")
        submit(service, req)
        final = wait_state(service, req, {"interrupted", "completed"})
        assert final.state == "interrupted"
        assert final.failure.code == "worker_attach_fenced" and final.failure.retryable
        assert not final.quarantined
        assert encoded == [], "an unattached worker must never receive the candidate"
        assert not request_marker.exists()

        service.store.attach_worker = original_attach
        retry(service, req, final.current_attempt.attempt_id)
        assert wait_state(service, req, {"completed"}).state == "completed"
    finally:
        service.store.attach_worker = original_attach
        service.close()


def test_attach_cleanup_failure_fails_closed_without_starting_request_writer(tmp_path, monkeypatch):
    import tcenv.remote.job_engine as job_engine_module

    request_marker = tmp_path / "worker-received-request"
    service = engine(tmp_path / "state", worker_env={"FAKE_JOB_RUNNING": str(request_marker)})
    encoded = []
    processes = []
    original_encode = job_engine_module.encode_request
    original_kill = job_engine_module.kill_group
    original_attach = service.store.attach_worker

    def recording_encode(*args, **kwargs):
        encoded.append(True)
        return original_encode(*args, **kwargs)

    def failing_kill(process):
        if process not in processes:
            processes.append(process)
        raise OSError("injected kill failure")

    monkeypatch.setattr(job_engine_module, "encode_request", recording_encode)
    monkeypatch.setattr(job_engine_module, "kill_group", failing_kill)
    service.store.attach_worker = lambda *_args, **_kwargs: False
    try:
        req = request("attach-cleanup-failure", source=b"SLEEP", candidate_timeout_ms=30_000)
        cap = capabilities(service)
        service.submit(
            request=req, expected=service.expected_identities,
            capability_id=cap.capability_id, grant_token=TOKEN,
        )
        deadline = time.monotonic() + 3
        while not service.store.unavailable and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.store.unavailable
        service._scheduler.join(timeout=3)
        assert service._stop.is_set()
        assert not service._scheduler.is_alive()
        assert processes, "the injected pre-request cleanup failure was not exercised"
        assert encoded == [], "an uncleaned unattached worker must never receive the candidate"
        assert not request_marker.exists()

        with pytest.raises(JobError) as caught:
            get(service, req)
        assert caught.value.code == "unavailable"
        with pytest.raises(JobError) as caught:
            service.submit(
                request=req, expected=service.expected_identities,
                capability_id=cap.capability_id, grant_token=TOKEN,
            )
        assert caught.value.code == "unavailable"
    finally:
        service.store.attach_worker = original_attach
        monkeypatch.setattr(job_engine_module, "kill_group", original_kill)
        for process in processes:
            original_kill(process)
            job_engine_module.forget(process)
        service.close()


def test_cancel_cleanup_failure_quarantines_without_starting_request_writer(tmp_path, monkeypatch):
    import tcenv.remote.job_engine as job_engine_module

    startup_marker = tmp_path / "worker-started"
    request_marker = tmp_path / "worker-received-request"
    service = engine(tmp_path / "state", worker_env={
        "FAKE_JOB_STARTED": str(startup_marker), "FAKE_JOB_RUNNING": str(request_marker),
    })
    encoded = []
    processes = []
    original_encode = job_engine_module.encode_request
    original_kill = job_engine_module.kill_group

    def recording_encode(*args, **kwargs):
        encoded.append(True)
        return original_encode(*args, **kwargs)

    def failing_kill(process):
        if process not in processes:
            processes.append(process)
        raise OSError("injected kill failure")

    monkeypatch.setattr(job_engine_module, "encode_request", recording_encode)
    monkeypatch.setattr(job_engine_module, "kill_group", failing_kill)
    try:
        req = request("cancel-cleanup-failure", source=b"SLEEP", candidate_timeout_ms=30_000)
        original_attach = service.store.attach_worker

        def attach_and_cancel(*args, **kwargs):
            attached = original_attach(*args, **kwargs)
            deadline = time.monotonic() + 3
            while not startup_marker.is_file() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert startup_marker.is_file(), "worker did not reach its pre-stdin startup marker"
            service.store.cancel(
                grant_token=TOKEN, request_id=req.request_id,
                request_digest=req.request_digest, expected=service.expected_identities,
            )
            return attached

        service.store.attach_worker = attach_and_cancel
        submit(service, req)
        snapshot = get(service, req)
        deadline = time.monotonic() + 3
        while not snapshot.quarantined and not encoded and snapshot.state in {"accepted", "queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = get(service, req)
        assert snapshot.state == "running" and snapshot.quarantined
        assert snapshot.cancel_requested and "injected kill failure" in snapshot.recovery_error
        assert snapshot.current_attempt.worker_pid == processes[0].pid
        assert snapshot.current_attempt.worker_start_token
        assert not service.store.unavailable and not service._stop.is_set()
        assert encoded == [], "failed cleanup must not fall through to the request writer"
        assert not request_marker.exists()
    finally:
        monkeypatch.setattr(job_engine_module, "kill_group", original_kill)
        for process in processes:
            original_kill(process)
            job_engine_module.forget(process)
        service.close()


def test_nonretryable_worker_result_error_is_not_mislabeled_retryable(tmp_path):
    service = engine(tmp_path / "state")
    try:
        req = request("nonretryable-invalid-result", source=b"BAD_REPORT")
        submit(service, req)
        final = wait_state(service, req, {"failed"})
        assert final.failure.code == "worker_result_invalid" and not final.failure.retryable
        with pytest.raises(JobError, match="current interrupted"):
            retry(service, req, final.current_attempt.attempt_id)
    finally:
        service.close()


def test_nonretryable_data_loss_is_quarantined_not_offered_as_retry(tmp_path):
    service = engine(tmp_path / "state")
    try:
        req = request("corrupt-result-commit")

        def corrupt_completion(*_args, **_kwargs):
            raise StateCorrupt("injected corrupt result envelope")

        service.store.complete_attempt = corrupt_completion
        submit(service, req)
        final = wait_state(service, req, {"interrupted"})
        assert final.quarantined and "corrupt result envelope" in final.recovery_error
        assert final.failure.code == "data_loss" and not final.failure.retryable
        with pytest.raises(JobError, match="quarantined"):
            retry(service, req, final.current_attempt.attempt_id)
    finally:
        service.close()


def test_worker_frame_request_id_is_bound_to_the_active_request():
    from tcenv.remote.job_types import Attempt
    from tcenv.remote.protocol import Frame
    from tcenv.remote.worker_protocol import WORKER_PROTOCOL_VERSION, decode_worker_frame

    attempt = Attempt("attempt-1", 1, "running", 0, False, 0)
    payload = {
        "worker_protocol": WORKER_PROTOCOL_VERSION, "attempt_id": "attempt-1",
        "lease_fence": "1", "progress": {"stage": "compile"},
    }
    frame = Frame("progress", "other-request", payload, 0, "0" * 64)
    with pytest.raises(JobError, match="request ID"):
        decode_worker_frame(frame, attempt, request_id="request-1")
    assert decode_worker_frame(
        Frame("progress", "request-1", payload, 0, "0" * 64),
        attempt, request_id="request-1",
    )[0] == "progress"


def test_close_contains_an_arbitrary_supervisor_exception_and_closes_the_store(tmp_path):
    """A shutdown must never raise out of close(), and must not leak the store."""

    state = tmp_path / "state"
    service = engine(state)
    req = request("close-contains-kill-failure", source=b"SLEEP", candidate_timeout_ms=60_000)
    submit(service, req)
    wait_state(service, req, {"running"})
    deadline = time.monotonic() + 20.0
    while (
        time.monotonic() < deadline
        and (service._active is None or service._active.handle is None)
    ):
        time.sleep(0.02)
    assert service._active is not None and service._active.handle is not None

    original_kill = service.supervisor.kill

    def exploding_kill(handle):
        raise RuntimeError("supervisor exploded during shutdown")

    service.supervisor.kill = exploding_kill
    try:
        # The teardown runs on the bounded scheduling path; close(wait) waits
        # for that path and still reports the contained supervisor exception.
        assert service.close(wait=20.0) is False, "an unverified shutdown must be reported, not hidden"
        assert isinstance(service.close_error, RuntimeError)
        # A second close reports the same failure instead of pretending success,
        # and the store is never closed underneath a live scheduler thread.
        assert service.close() is False
    finally:
        # Let the live worker exit, as an out-of-band stop would: the deferred
        # finalizer then closes the store instead of leaking it.
        service.supervisor.kill = original_kill
        handle = service._active.handle if service._active is not None else None
        if handle is not None and handle.process is not None:
            try:
                os.killpg(handle.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    deadline = time.monotonic() + 30.0
    while not service.closed and time.monotonic() < deadline:
        time.sleep(0.02)
    assert service.closed is True, "the store was leaked after a failed shutdown"
    reopened = JobStore(state)
    try:
        assert reopened._job_by_id(req.request_id).state in {
            "running", "interrupted", "completed",
        }
    finally:
        reopened.close()


def test_close_never_reports_closed_while_the_finalizer_is_running(tmp_path):
    state = tmp_path / "state"
    service = engine(state)
    entered = threading.Event()
    release = threading.Event()
    original_close = service.store.close

    def slow_store_close():
        entered.set()
        assert release.wait(30), "the test never released the store close"
        return original_close()

    service.store.close = slow_store_close
    results: dict[str, bool] = {}
    first = threading.Thread(target=lambda: results.__setitem__("first", service.close(wait=30.0)))
    first.start()
    assert entered.wait(30), "store.close was never called"
    second = threading.Thread(target=lambda: results.__setitem__("second", service.close(wait=30.0)))
    second.start()
    # A concurrent close must wait for the first cleanup instead of claiming a
    # shutdown that has not happened yet.
    time.sleep(0.5)
    assert "second" not in results
    release.set()
    first.join(timeout=30)
    second.join(timeout=30)
    assert results == {"first": True, "second": True}
    assert service.closed is True
