"""Restart, Retry, retention, backup and process-crash regression tests."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from tcenv.remote.job_engine import JobEngine
from tcenv.remote.job_store import JobStore
from tcenv.remote.job_types import (
    Attempt, ExpectedEvaluator, ExpectedIdentities, JobError, JobFailure, StateCorrupt,
)

from ..support.job_engine import (
    TOKEN, capabilities, engine, expected_identities, get, grant, identity,
    request, retry, store_capabilities, submit, submit_to_store, wait_state,
)

ROOT = Path(__file__).resolve().parents[2]


def _crash_frontend(action: str, state_dir: Path, marker: Path, *, counter: Path, running: Path | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["FAKE_JOB_COUNTER"] = str(counter)
    if running is not None:
        env["FAKE_JOB_RUNNING"] = str(running)
    completed = subprocess.run(
        [sys.executable, "-m", "tests.support.fake_job_frontend", action,
         "--state-dir", str(state_dir), "--marker", str(marker)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=15, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert marker.is_file()


def test_restart_invalidates_capability_and_replays_result_without_reexecution(tmp_path):
    counter = tmp_path / "counter"
    state = tmp_path / "state"
    first = engine(state, worker_env={"FAKE_JOB_COUNTER": str(counter)})
    req = request("restart-replay")
    old_cap = capabilities(first)
    submit(first, req)
    completed = wait_state(first, req, {"completed"})
    first.close()

    second = engine(state, worker_env={"FAKE_JOB_COUNTER": str(counter)})
    try:
        with pytest.raises(JobError, match="old frontend") as caught:
            second.submit(
                request=req, expected=second.expected_identities,
                capability_id=old_cap.capability_id, grant_token=TOKEN,
            )
        assert caught.value.code == "failed_precondition"
        replay = submit(second, req, nonce="fresh-after-restart")
        assert replay.state == "completed"
        assert replay.current_attempt.attempt_id == completed.current_attempt.attempt_id
        assert replay.result.receipt == completed.result.receipt
        assert counter.read_text(encoding="utf-8") == "1"
        # Durable grants contain only a token lookup digest and subject/scope.
        connection = sqlite3.connect(state / "jobs.sqlite3")
        try:
            grant_sql = connection.execute("SELECT sql FROM sqlite_master WHERE name='grants'").fetchone()[0]
            assert "token_sha256" in grant_sql and " token " not in grant_sql
            rows = connection.execute("SELECT * FROM grants").fetchall()
            assert TOKEN not in str(rows)
        finally:
            connection.close()
    finally:
        second.close()


def test_interrupted_retry_new_fence_no_double_quota_and_stale_result_is_diagnostic(tmp_path):
    service = engine(tmp_path / "state")
    try:
        req = request("retry-success", source=b"FAIL_FIRST", kind="selfcheck", mode="timed")
        submit_cap = capabilities(service)
        service.submit(
            request=req, expected=service.expected_identities,
            capability_id=submit_cap.capability_id, grant_token=TOKEN,
        )
        interrupted = wait_state(service, req, {"interrupted"})
        old_attempt = interrupted.current_attempt
        assert service.store.quota_usage(req).used_count == 0
        with pytest.raises(JobError, match="fresh GetCapabilities"):
            service.retry(
                request_id=req.request_id, request_digest=req.request_digest,
                expected=service.expected_identities,
                interrupted_attempt_id=old_attempt.attempt_id,
                capability_id=submit_cap.capability_id, grant_token=TOKEN,
            )
        queued = retry(service, req, old_attempt.attempt_id)
        new_attempt = queued.current_attempt
        assert new_attempt.attempt_id != old_attempt.attempt_id
        assert new_attempt.lease_fence > old_attempt.lease_fence
        assert new_attempt.retry_index == old_attempt.retry_index + 1
        assert new_attempt.queued_at_ms >= old_attempt.queued_at_ms
        final = wait_state(service, req, {"completed"})
        usage = service.store.quota_usage(req)
        assert usage.used_count == 1 and usage.used_gpu_us == 500_000 and usage.reserved_count == 0
        assert final.result.receipt["receipt_id"] != "" if final.result else False
        assert final.result.receipt["attempt_id"] == new_attempt.attempt_id

        # A result from the replaced fence cannot overwrite the new terminal result.
        stale = service.store.complete_attempt(
            req.request_id, old_attempt, report={"status": "crash"}, evidence={},
        )
        assert stale.state == "completed"
        assert stale.result.receipt == final.result.receipt
        assert stale.result.report_json()["status"] == "ok"
        with pytest.raises(JobError, match="current interrupted"):
            retry(service, req, new_attempt.attempt_id, nonce="terminal-retry")
    finally:
        service.close()


def test_attach_worker_returns_exact_fence_status_and_preserves_stale_diagnostic(tmp_path):
    store = JobStore(tmp_path / "state")
    try:
        store.register_grant(grant())
        req = request("attach-worker-bool")
        submit_to_store(store, req)
        store.enqueue_accepted(req.request_id)
        claimed = store.claim_next()
        assert claimed is not None
        _request, attempt = claimed
        stale = replace(attempt, lease_fence=attempt.lease_fence + 1)
        assert store.attach_worker(
            req.request_id, stale, pid=4242, pgid=4242, start_token="stale-token",
        ) is False
        row = store._conn.execute(
            "SELECT worker_pid,diagnostic_json FROM attempts WHERE attempt_id=?",
            (attempt.attempt_id,),
        ).fetchone()
        assert row["worker_pid"] is None
        assert json.loads(row["diagnostic_json"])["kind"] == "stale_worker_attach"

        assert store.attach_worker(
            req.request_id, attempt, pid=4243, pgid=4243, start_token="current-token",
        ) is True
        attached = store.active_attempt(req.request_id)
        assert attached is not None
        assert attached.worker_pid == attached.worker_pgid == 4243
        assert attached.worker_start_token == "current-token"
    finally:
        store.close()


def test_max_attempts_and_queue_timeout_refund(tmp_path):
    service = engine(tmp_path / "state", max_attempts=2)
    try:
        req = request("attempt-limit", source=b"EXIT")
        submit(service, req)
        first = wait_state(service, req, {"interrupted"})
        retry(service, req, first.current_attempt.attempt_id)
        second = wait_state(service, req, {"interrupted"})
        assert second.current_attempt.retry_index == 1
        with pytest.raises(JobError, match="maximum attempt") as caught:
            retry(service, req, second.current_attempt.attempt_id, nonce="third-attempt")
        assert caught.value.code == "resource_exhausted"
    finally:
        service.close()

    service = engine(tmp_path / "queue-state")
    try:
        blocker = request("queue-blocker", source=b"FAIR_SLEEP")
        queued = request("queue-timeout", kind="selfcheck", mode="timed", queue_timeout_ms=50)
        submit(service, blocker)
        wait_state(service, blocker, {"running"})
        submit(service, queued, nonce="queued-selfcheck")
        interrupted = wait_state(service, queued, {"interrupted"})
        assert interrupted.failure.code == "queue_timeout"
        assert service.store.quota_usage(queued).used_count == 0
        wait_state(service, blocker, {"completed"})
        retry(service, queued, interrupted.current_attempt.attempt_id, nonce="retry-after-queue")
        assert wait_state(service, queued, {"completed"}).state == "completed"
        assert service.store.quota_usage(queued).used_count == 1
    finally:
        service.close()


def test_expired_tombstone_never_reopens_request_id(tmp_path):
    service = engine(tmp_path / "state", retention_ms=60_000)
    try:
        req = request("retention")
        submit(service, req)
        completed = wait_state(service, req, {"completed"})
        assert service.store.expire_jobs(now_ms=completed.expires_at_ms + 1) == 1
        expired = get(service, req)
        assert expired.state == "expired" and expired.result is None
        replay = submit(service, req, nonce="expired-replay")
        assert replay.state == "expired"
        with pytest.raises(JobError):
            retry(service, req, completed.current_attempt.attempt_id, nonce="expired-retry")
        changed = request("retention", source=b"different")
        with pytest.raises(JobError, match="different digest"):
            submit(service, changed, nonce="expired-conflict")
    finally:
        service.close()


def test_single_owner_permissions_and_corrupt_state_fail_closed(tmp_path):
    state = tmp_path / "state"
    service = engine(state)
    assert (state.stat().st_mode & 0o777) == 0o700
    with pytest.raises(JobError, match="another frontend"):
        JobStore(state)
    req = request("durable")
    submit(service, req)
    wait_state(service, req, {"completed"})
    service.close()

    (state / "jobs.sqlite3").write_bytes(b"not a sqlite database")
    with pytest.raises(StateCorrupt):
        engine(state)


def test_semantically_corrupt_result_receipt_binding_fails_closed(tmp_path):
    state = tmp_path / "state"
    service = engine(state)
    req = request("semantic-corruption")
    submit(service, req)
    wait_state(service, req, {"completed"})
    service.close()
    connection = sqlite3.connect(state / "jobs.sqlite3")
    try:
        connection.execute("DELETE FROM receipts WHERE request_id=?", (req.request_id,))
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(StateCorrupt, match="result/receipt"):
        engine(state)


def test_wal_backup_restore_quarantines_nonterminal_and_requires_explicit_retry(tmp_path):
    counter = tmp_path / "counter"
    state = tmp_path / "state"
    backup = tmp_path / "backup.sqlite3"
    original = engine(state, worker_env={"FAKE_JOB_COUNTER": str(counter)})
    req = request("restore-running", source=b"RECOVER")
    submit(original, req)
    running = wait_state(original, req, {"running"})
    deadline = time.monotonic() + 5
    while not counter.is_file() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert counter.read_text(encoding="utf-8") == "1"
    original.backup(backup)
    original.close()  # changes only the original DB, not the consistent backup

    restored_dir = tmp_path / "restored"
    JobStore.restore_backup(backup, restored_dir)
    restored = engine(restored_dir, worker_env={"FAKE_JOB_COUNTER": str(counter)})
    try:
        recovered = get(restored, req)
        assert recovered.state == "interrupted" and recovered.quarantined
        assert recovered.current_attempt.attempt_id == running.current_attempt.attempt_id
        time.sleep(0.2)
        assert counter.read_text(encoding="utf-8") == "1"  # no automatic rerun
        with pytest.raises(JobError, match="quarantined"):
            retry(restored, req, recovered.current_attempt.attempt_id, nonce="blocked-restore-retry")
        restored.release_quarantine(req.request_id)
        retry(restored, req, recovered.current_attempt.attempt_id, nonce="released-restore-retry")
        final = wait_state(restored, req, {"completed"})
        assert final.current_attempt.lease_fence > running.current_attempt.lease_fence
        assert counter.read_text(encoding="utf-8") == "2"
    finally:
        restored.close()


def test_durable_queued_job_restarts_without_a_second_reservation(tmp_path):
    state = tmp_path / "state"
    req = request("queued-restart", kind="selfcheck", mode="timed")
    store = JobStore(state)
    store.register_grant(grant())
    service_identity = identity()
    expected = ExpectedIdentities(ExpectedEvaluator(
        site_id=service_identity.site_id, device_id=service_identity.device_id,
        target_id=service_identity.target_id, worker_build=service_identity.worker_build,
        task_pack_digest=service_identity.task_pack_digest,
        evaluator_epoch=service_identity.external_epoch(execution_protocol_version="3"),
    ))
    provenance = service_identity.external_provenance(execution_protocol_version="3")
    cap = store.get_capabilities(
        grant_token=TOKEN, client_nonce="queued-restart-nonce", expected={},
        expected_identities=expected, provenance=provenance,
        api_contract_ids=[expected.api_contract_id],
        evidence_contract_ids=[expected.evidence_contract_id],
        required_features=[], supported_features=frozenset(),
    )
    store.submit(
        request=req, request_digest=req.request_digest, expected=expected,
        provenance=provenance, capability_id=cap.capability_id,
        required_features=[], grant_token=TOKEN,
    )
    assert store.enqueue_accepted(req.request_id).state == "queued"
    store.close()

    service = engine(state)
    try:
        final = wait_state(service, req, {"completed"})
        assert final.current_attempt.retry_index == 0
        assert service.store.quota_usage(req).used_count == 1
    finally:
        service.close()


def test_process_crash_after_accept_recovers_enqueue_and_one_quota_reservation(tmp_path):
    state = tmp_path / "state"
    counter = tmp_path / "counter"
    _crash_frontend("after-accept", state, tmp_path / "accepted.marker", counter=counter)
    req = request("crash-request", source=b"OK", kind="selfcheck", mode="timed")
    service = engine(state, worker_env={"FAKE_JOB_COUNTER": str(counter)})
    try:
        final = wait_state(service, req, {"completed"})
        assert final.current_attempt.retry_index == 0
        assert counter.read_text(encoding="utf-8") == "1"
        assert service.store.quota_usage(req).used_count == 1
    finally:
        service.close()


def test_process_crash_with_live_worker_cleans_old_group_before_interrupted_retry(tmp_path):
    state = tmp_path / "state"
    counter = tmp_path / "counter"
    running_marker = tmp_path / "worker-running.json"
    _crash_frontend(
        "after-running", state, tmp_path / "crashed.marker",
        counter=counter, running=running_marker,
    )
    worker = json.loads(running_marker.read_text(encoding="utf-8"))
    req = request("crash-request", source=b"RECOVER")
    service = engine(state, worker_env={"FAKE_JOB_COUNTER": str(counter)})
    try:
        recovered = get(service, req)
        assert recovered.state == "interrupted"
        assert counter.read_text(encoding="utf-8") == "1"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(worker["pid"], 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail("old worker survived recovery cleanup")
        time.sleep(0.15)
        assert get(service, req).state == "interrupted"  # explicit Retry only
        retry(service, req, recovered.current_attempt.attempt_id)
        final = wait_state(service, req, {"completed"})
        assert final.current_attempt.lease_fence > recovered.current_attempt.lease_fence
        assert counter.read_text(encoding="utf-8") == "2"
    finally:
        service.close()


def test_process_crash_after_result_commit_replays_terminal_result(tmp_path):
    state = tmp_path / "state"
    counter = tmp_path / "counter"
    _crash_frontend("after-result-commit", state, tmp_path / "committed.marker", counter=counter)
    req = request("crash-request", source=b"OK")
    service = engine(state, worker_env={"FAKE_JOB_COUNTER": str(counter)})
    try:
        completed = wait_state(service, req, {"completed"})
        replay = submit(service, req, nonce="post-crash-replay")
        assert replay.state == "completed"
        assert replay.current_attempt.attempt_id == completed.current_attempt.attempt_id
        assert replay.result.receipt == completed.result.receipt
        assert counter.read_text(encoding="utf-8") == "1"
    finally:
        service.close()


@pytest.mark.parametrize("tamper", ["request_json", "source"])
def test_open_revalidates_stored_request_and_quarantines_only_that_job(tmp_path, tamper):
    state = tmp_path / "state"
    req = request("open-digest-tamper-" + tamper)
    store = JobStore(state)
    store.register_grant(grant())
    submit_to_store(store, req)
    store.close()

    connection = sqlite3.connect(state / "jobs.sqlite3")
    try:
        if tamper == "source":
            connection.execute("UPDATE jobs SET source=? WHERE request_id=?", (b"changed after commit", req.request_id))
        else:
            raw = connection.execute("SELECT request_json FROM jobs WHERE request_id=?", (req.request_id,)).fetchone()[0]
            document = json.loads(raw)
            document["reps"] += 1
            connection.execute(
                "UPDATE jobs SET request_json=? WHERE request_id=?",
                (json.dumps(document, sort_keys=True, separators=(",", ":")), req.request_id),
            )
        connection.commit()
    finally:
        connection.close()

    reopened = JobStore(state)
    try:
        snapshot = reopened._job_by_id(req.request_id)
        assert snapshot.state == "accepted" and snapshot.quarantined
        assert "request_digest" in snapshot.recovery_error
        with pytest.raises(StateCorrupt):
            reopened.release_quarantine(req.request_id)
        assert reopened._job_by_id(req.request_id).quarantined
        # The bad job is isolated; an independent durable record remains usable.
        good = request("open-digest-good")
        submit_to_store(reopened, good)
        assert reopened.enqueue_accepted(good.request_id).state == "queued"
    finally:
        reopened.close()


def test_cleanup_error_quarantine_can_be_released_by_rechecking_cleanup(tmp_path, monkeypatch):
    state = tmp_path / "state"
    req = request("cleanup-error-release")
    store = JobStore(state)
    store.register_grant(grant())
    submit_to_store(store, req)
    store.enqueue_accepted(req.request_id)
    claimed = store.claim_next()
    assert claimed is not None
    store.close()

    monkeypatch.setattr(
        JobEngine, "_cleanup_recovered_worker",
        staticmethod(lambda _attempt: "operator cleanup still blocked"),
    )
    service = engine(state)
    try:
        quarantined = get(service, req)
        assert quarantined.state == "running" and quarantined.quarantined
        assert "cleanup still blocked" in quarantined.recovery_error
        monkeypatch.setattr(JobEngine, "_cleanup_recovered_worker", staticmethod(lambda _attempt: ""))
        released = service.release_quarantine(req.request_id)
        assert released.state == "interrupted" and not released.quarantined
        assert released.recovery_error == ""
        retry(service, req, released.current_attempt.attempt_id)
        assert wait_state(service, req, {"completed"}).state == "completed"
    finally:
        service.close()


def test_interruption_with_outstanding_cancel_finishes_cancelled_and_rejects_retry(tmp_path):
    store = JobStore(tmp_path / "state")
    try:
        store.register_grant(grant())
        req = request("cancel-before-interrupt")
        submit_to_store(store, req)
        store.enqueue_accepted(req.request_id)
        _request, attempt = store.claim_next()
        store.cancel(
            grant_token=TOKEN, request_id=req.request_id,
            request_digest=req.request_digest, expected=expected_identities(),
        )
        assert store.interrupt_attempt(
            req.request_id, attempt,
            JobFailure("infrastructure", True, "injected_interrupt", "interrupt raced cancel"),
        )
        final = store._job_by_id(req.request_id)
        assert final.state == "cancelled" and final.current_attempt.state == "cancelled"
        cap = store_capabilities(store)
        with pytest.raises(JobError, match="current interrupted") as caught:
            store.retry(
                grant_token=TOKEN, request_id=req.request_id,
                request_digest=req.request_digest, expected=expected_identities(),
                interrupted_attempt_id=attempt.attempt_id, capability_id=cap.capability_id,
                required_features=[],
            )
        assert caught.value.code == "failed_precondition"
    finally:
        store.close()


def test_retention_preserves_evidence_then_purges_tombstones_and_old_leases(tmp_path):
    now = [time.time()]
    service = engine(tmp_path / "state", retention_ms=1000, clock=lambda: now[0])
    try:
        req = request("bounded-retention")
        submit(service, req)
        completed = wait_state(service, req, {"completed"})
        short_cap = capabilities(service, valid_for_ms=1)
        service.renew_grant(grant("retention-renewed-token"))

        now[0] = (completed.expires_at_ms - 1) / 1000
        assert service.store.expire_jobs(now_ms=completed.expires_at_ms - 1) == 0
        assert get(service, req, token="retention-renewed-token").result is not None
        now[0] = completed.expires_at_ms / 1000
        expired = get(service, req, token="retention-renewed-token")
        assert expired.state == "expired" and expired.result is None
        assert service.store.expire_jobs(now_ms=completed.expires_at_ms + 999) == 0
        assert service.store._job_by_id(req.request_id).state == "expired"

        service.store.expire_jobs(now_ms=completed.expires_at_ms + 1000)
        with pytest.raises(JobError) as caught:
            get(service, req, token="retention-renewed-token")
        assert caught.value.code == "not_found"
        conn = service.store._conn
        for table in ("jobs", "attempts", "events", "results", "receipts"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM grants WHERE status='superseded'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM capabilities WHERE capability_id=?", (short_cap.capability_id,)).fetchone()[0] == 0
    finally:
        service.close()


def test_expired_unused_grant_is_purged_only_after_retention(tmp_path):
    now = [time.time()]
    store = JobStore(tmp_path / "state", retention_ms=1000, clock=lambda: now[0])
    try:
        store.register_grant(grant(expires_at=now[0] + 0.001))
        expires_at = store._conn.execute("SELECT expires_at_ms FROM grants").fetchone()[0]
        store.expire_jobs(now_ms=expires_at + 999)
        assert store._conn.execute("SELECT COUNT(*) FROM grants").fetchone()[0] == 1
        store.expire_jobs(now_ms=expires_at + 1000)
        assert store._conn.execute("SELECT COUNT(*) FROM grants").fetchone()[0] == 0
    finally:
        store.close()


def test_linux_process_token_includes_boot_identity_and_never_kills_on_mismatch(monkeypatch):
    import tcenv.remote.job_engine as job_engine_module

    boot_id = ["boot-a"]

    def fake_is_file(path):
        return str(path) == "/proc/4242/stat"

    def fake_read_text(path, *, encoding=None, errors=None):
        del encoding, errors
        if str(path) == "/proc/sys/kernel/random/boot_id":
            return boot_id[0] + "\n"
        if str(path) == "/proc/4242/stat":
            tail = ["S"] + ["0"] * 18 + ["12345"] + ["0"] * 5
            return "4242 (fake worker) " + " ".join(tail)
        raise OSError(path)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    monkeypatch.setattr(Path, "read_text", fake_read_text)
    first = job_engine_module._process_start_token_once(4242)
    assert first == "linux:boot-a:12345"
    boot_id[0] = "boot-b"
    assert job_engine_module._process_start_token_once(4242) == "linux:boot-b:12345"

    attempt = Attempt(
        "attempt-boot", 1, "running", 0, False, 0,
        worker_pid=4242, worker_pgid=4242, worker_start_token=first,
    )
    monkeypatch.setattr(
        job_engine_module.os, "killpg",
        lambda *_args: pytest.fail("a reboot-mismatched process group must not be signalled"),
    )
    error = job_engine_module.JobEngine._cleanup_recovered_worker(attempt)
    assert "start token differs" in error


def test_macos_process_token_includes_kern_boottime(monkeypatch):
    import tcenv.remote.job_engine as job_engine_module

    monkeypatch.setattr(Path, "is_file", lambda _path: False)
    monkeypatch.setattr(job_engine_module.sys, "platform", "darwin")

    class Completed:
        returncode = 0

        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(argv, **_kwargs):
        if argv[0] == "/usr/sbin/sysctl":
            return Completed("{ sec = 123, usec = 456 }\n")
        return Completed("Mon Oct 26 01:02:03 2026\n")

    monkeypatch.setattr(job_engine_module.subprocess, "run", fake_run)
    assert job_engine_module._process_start_token_once(5150) == (
        "ps:{ sec = 123, usec = 456 }:Mon Oct 26 01:02:03 2026"
    )


def test_schema_version_is_transactional_and_failed_commit_is_rolled_back(tmp_path):
    store = JobStore(tmp_path / "state")
    try:
        assert store._conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert not store._conn.in_transaction

        def deny_commit(action, arg1, _arg2, _database, _source):
            if action == sqlite3.SQLITE_TRANSACTION and arg1 == "COMMIT":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        store._conn.set_authorizer(deny_commit)
        with pytest.raises(StateCorrupt, match="write failed"):
            with store._write():
                pass
        store._conn.set_authorizer(None)
        assert not store._conn.in_transaction
        assert store.unavailable
    finally:
        store.close()


def test_tampered_stored_signature_is_rejected_when_serving_result(tmp_path):
    key = b"external-test-signing-key-32-bytes!"
    service = engine(tmp_path / "state", signing_key=key, signing_key_id="sim-external-site/key-1")
    try:
        req = request("stored-signature-verification")
        submit(service, req)
        final = wait_state(service, req, {"completed"})
        signature = dict(final.result.signature)
        signature["signature"] = "0" * 64
        with service.store._write() as conn:
            conn.execute(
                "UPDATE results SET signature_json=? WHERE request_id=?",
                (json.dumps(signature, sort_keys=True, separators=(",", ":")), req.request_id),
            )
        with pytest.raises(StateCorrupt, match="does not verify|envelope digest"):
            get(service, req)
    finally:
        service.close()
