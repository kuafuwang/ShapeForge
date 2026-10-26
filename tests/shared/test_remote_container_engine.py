"""Job-engine integration tests for the container execution plane (fake Docker)."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from tcenv.remote.container_supervisor import ContainerSupervisorConfig, DockerAttemptSupervisor
from tcenv.remote.job_engine import JobEngine
from tcenv.remote.job_store import JobStore
from tcenv.remote.job_types import Attempt, JobError
from tcenv.remote.supervisor import AttemptSpec, WorkerHandle
from tests.support import job_engine as helpers
from tests.support.fake_docker import install_fake_docker
from tests.support.process_state import process_state
from tests.support.job_engine import (
    TOKEN, capabilities, get, grant, request, submit, wait_state,
)

IMAGE_DIGEST = "d" * 64

# The fault worker lives here (instead of a new `tests/support` module) so this
# round's diff stays inside the TASK's path scope.  It speaks the private
# worker protocol and blocks on the candidate deadline, exactly like a real
# stuck worker.
_FAULT_WORKER_SOURCE = r'''
import os, sys, time
from pathlib import Path
from tcenv.remote import worker_protocol

mode = os.environ.get("FAULT_WORKER_MODE", "")
ready = os.environ.get("FAULT_WORKER_READY", "")
attempt_id, request, request_id, fence = worker_protocol.read_request(sys.stdin.buffer)
report = {
    "task": request.task, "dtype": request.dtype, "size": request.size,
    "mode": request.mode, "seed": request.seed_int, "status": "ok", "stage": "done",
    "candidate_us": 1.0, "speedup": 1.0, "elapsed_s": 0.0,
    "target_id": request.target_id, "protocol_version": "3", "simulated": True,
}
evidence = {"candidate_sha256": request.candidate_sha256}
if mode in {"wrong_task_hang", "wrong_task_second_result"}:
    report["task"] = "not-the-requested-task"

def write_result():
    worker_protocol.write_result(
        sys.stdout.buffer, request_id=request_id, attempt_id=attempt_id,
        lease_fence=fence, report=report, evidence=evidence,
    )
    sys.stdout.buffer.flush()
    if ready:
        Path(ready).write_text("ready", encoding="utf-8")

if mode == "wrong_task_hang":
    write_result()
    time.sleep(120)
elif mode == "double_result":
    write_result()
    write_result()
    time.sleep(120)
elif mode == "valid_hang":
    write_result()
    time.sleep(120)
else:
    raise SystemExit("unknown fault worker mode")
'''


class _FrozenClock:
    """A monotonic clock that does not advance until the test says so.

    A normal-path test must not race the candidate deadline against container
    setup: the budget stays at its production value (3s) and the clock is simply
    held still until the worker result is committed.
    """

    def __init__(self, start: float = 1_000.0) -> None:
        self._lock = threading.Lock()
        self._value = float(start)

    def now(self) -> float:
        with self._lock:
            return self._value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._value += max(0.0, float(seconds))


class _FakeClock:
    """A monotonic clock the test advances to expire the candidate deadline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._offset = 0.0

    def now(self) -> float:
        with self._lock:
            return time.monotonic() + self._offset

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._offset += max(0.0, float(seconds))


def _wait_for_marker(path: Path, *, timeout: float = 20.0) -> None:
    """Deterministic gate: wait until the worker itself proved it reached a step."""

    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists(), f"the worker never reached {path}"



def container_identity():
    return replace(helpers.identity(), worker_build=f"sim-external-build+img:{IMAGE_DIGEST}")


def container_engine(
    state_dir: Path, docker: Path, *, worker_env: dict[str, str] | None = None,
    identity=None, monotonic=None,
) -> JobEngine:
    supervisor = DockerAttemptSupervisor(ContainerSupervisorConfig(
        image="taic-worker-sim:local", image_digest=IMAGE_DIGEST, docker=str(docker),
        env={"HOME": "/tmp", "PYTHONPATH": "/opt/taic"}, kill_timeout_s=5.0,
    ))
    return JobEngine(
        identity=identity or container_identity(), grants=[grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_job_engine"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=state_dir,
        worker_env=worker_env, supervisor=supervisor, monotonic=monotonic,
    )


def test_container_engine_commits_result_only_after_verified_cleanup(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    clock = _FrozenClock()
    service = container_engine(tmp_path / "state", docker, monotonic=clock.now)
    req = request("container-ok")
    original_attach = service.store.attach_container
    observed: dict[str, object] = {}

    def attach_and_record(*args, **kwargs):
        observed["starts_before_attach"] = [
            call for call in fake.calls() if len(call) > 1 and call[1] == "start"
        ]
        observed["create_calls"] = [call for call in fake.calls() if len(call) > 1 and call[1] == "create"]
        attached = original_attach(*args, **kwargs)
        observed["row_id"] = service.store._job_by_id(req.request_id).current_attempt.container_id
        return attached

    service.store.attach_container = attach_and_record
    try:
        submit(service, req)
        final = wait_state(service, req, {"completed"})
        assert final.result is not None
        # The normal path must really be the ok path: a pre-start deadline
        # timeout would still "complete", but with no create/attach and no
        # digest-bearing report.
        assert final.result.report_json()["status"] == "ok"
        assert final.failure is None
        assert observed["create_calls"], "no container was created"
        assert observed["row_id"], "the durable attach never happened"
        assert final.result.receipt["attempt_id"] == final.current_attempt.attempt_id
        assert final.result.provenance["worker_build"].endswith(f"+img:{IMAGE_DIGEST}")
        assert final.current_attempt.execution_kind == "container"
        assert final.current_attempt.container_image_digest == IMAGE_DIGEST
        assert final.current_attempt.container_runtime == "runc"
        # ``create`` and the durable attach both precede ``start``, and the
        # container never receives request bytes before its id is durable.
        assert observed["starts_before_attach"] == []
        assert len(observed["create_calls"]) == 1
        assert observed["row_id"] == final.current_attempt.container_id
        # The commit happens only after Kill/Delete/verify: no container and no
        # label-matched duplicate survives a completed attempt.
        assert fake.records() == []
        assert not (tmp_path / "state" / "container-env" / f"{final.current_attempt.attempt_id}.env").exists()
        assert service.supervisor.kind == "container"
    finally:
        service.close()


def test_container_timeout_kills_parent_and_child_then_reports_timeout(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    started = tmp_path / "worker-started"
    clock = _FakeClock()
    service = container_engine(
        tmp_path / "state", docker, worker_env={"FAKE_JOB_STARTED": str(started)},
        monotonic=clock.now,
    )
    # The budget is generous: the test expires the absolute deadline itself,
    # after the worker proved it is running.  Setup speed cannot decide this.
    req = request("container-timeout", source=b"SLEEP", candidate_timeout_ms=60_000)
    try:
        submit(service, req)
        _wait_for_marker(started)
        clock.advance(61.0)
        final = wait_state(service, req, {"completed"}, timeout=20.0)
        # Deterministic gate: the worker wrote its start marker, so the timeout
        # was decided while the container was running, not by a setup overrun.
        assert started.is_file()
        assert final.result is not None
        assert final.result.report_json()["status"] == "timeout"
        assert fake.records() == []
        calls = [call[1] for call in fake.calls() if len(call) > 1]
        assert "kill" in calls and "rm" in calls
        # The kill happened before the result: every record is already deleted.
        assert all(
            record["State"]["Status"] == "exited"
            for record in fake.records()
        )
    finally:
        service.close()


def test_container_cancel_tears_down_and_old_fence_cannot_commit(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    service = container_engine(tmp_path / "state", docker)
    req = request("container-cancel", source=b"SLEEP", candidate_timeout_ms=60_000)
    try:
        submit(service, req)
        deadline = time.monotonic() + 10
        while not fake.records() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert fake.records(), "container was never created"
        cancelled = service.cancel(
            request_id=req.request_id, request_digest=req.request_digest,
            expected=service.expected_identities, grant_token=TOKEN,
        )
        assert cancelled.state == "cancelled"
        # Cancel is synchronous for containers: kill+delete+verify finished.
        assert fake.records() == []
        attempt = cancelled.current_attempt
        assert attempt is not None and attempt.state == "cancelled"
        # A late old worker (same attempt/fence) cannot write a result.
        forged = {
            "task": req.task, "dtype": req.dtype, "size": req.size, "mode": req.mode,
            "seed": req.seed_int, "status": "ok", "stage": "done",
            "target_id": req.target_id, "protocol_version": "3", "simulated": True,
        }
        late = service.store.complete_attempt(
            req.request_id, attempt, report=forged,
            evidence={"candidate_sha256": req.candidate_sha256},
        )
        assert late.state == "cancelled" and late.result is None
        assert get(service, req).result is None
    finally:
        service.close()


def test_container_result_is_withheld_when_delete_cannot_be_verified(tmp_path):
    docker, fake = install_fake_docker(
        tmp_path / "docker", image_digest=IMAGE_DIGEST, control={"rm_leaves_running": True},
    )
    service = container_engine(tmp_path / "state", docker)
    req = request("container-cleanup-incomplete")
    try:
        submit(service, req)
        deadline = time.monotonic() + 15.0
        final = get(service, req)
        while not (final.quarantined or final.state in {"completed", "failed"}) and time.monotonic() < deadline:
            time.sleep(0.02)
            final = get(service, req)
        assert final.state == "running" and final.quarantined is True
        assert "still present" in final.recovery_error
        assert final.result is None
        cap = capabilities(service)
        with pytest.raises(JobError, match="quarantined") as retry_error:
            service.retry(
                request_id=req.request_id, request_digest=req.request_digest,
                expected=service.expected_identities,
                interrupted_attempt_id=final.current_attempt.attempt_id,
                capability_id=cap.capability_id, grant_token=TOKEN,
            )
        assert retry_error.value.code == "failed_precondition"
        # Operator release rechecks the durable container id after fixing Docker.
        fake.set_control(rm_leaves_running=False)
        released = service.release_quarantine(req.request_id)
        assert released.state == "interrupted" and released.quarantined is False
        assert fake.records() == []
        retried = helpers.retry(service, req, released.current_attempt.attempt_id)
        assert retried.state in {"queued", "running", "completed"}
        assert wait_state(service, req, {"completed"}, timeout=15.0).state == "completed"
    finally:
        service.close()


def test_container_create_failure_never_falls_back_to_a_process_worker(tmp_path):
    docker, fake = install_fake_docker(
        tmp_path / "docker", image_digest=IMAGE_DIGEST, control={"fail_create": True},
    )
    marker = tmp_path / "worker-started"
    service = container_engine(tmp_path / "state", docker, worker_env={"FAKE_JOB_STARTED": str(marker)})
    req = request("container-create-failure")
    try:
        submit(service, req)
        final = wait_state(service, req, {"interrupted"})
        assert final.failure is not None and final.failure.code == "worker_unavailable"
        assert final.failure.retryable is True
        assert final.quarantined is False
        assert marker.exists() is False
        assert fake.records() == []
        assert service.supervisor.kind == "container"
    finally:
        service.close()


def test_ambiguous_create_failure_cleans_the_container_it_never_learned_about(tmp_path):
    """The daemon can create the container while the CLI still fails."""

    docker, fake = install_fake_docker(
        tmp_path / "docker", image_digest=IMAGE_DIGEST,
        control={"fail_create_after_record": True},
    )
    marker = tmp_path / "worker-started"
    service = container_engine(tmp_path / "state", docker, worker_env={"FAKE_JOB_STARTED": str(marker)})
    req = request("container-create-ambiguous")
    try:
        submit(service, req)
        final = wait_state(service, req, {"interrupted"})
        # The engine never saw a container id; the deterministic name/label had
        # to be cleaned and verified before Retry became available.
        assert fake.records() == []
        assert final.quarantined is False
        assert final.failure is not None and final.failure.code == "worker_unavailable"
        assert marker.exists() is False
        assert not any(len(call) > 1 and call[1] == "start" for call in fake.calls())
        assert any(len(call) > 1 and call[1] == "rm" for call in fake.calls())
    finally:
        service.close()


def test_ambiguous_create_failure_quarantines_when_cleanup_is_unverified(tmp_path):
    docker, fake = install_fake_docker(
        tmp_path / "docker", image_digest=IMAGE_DIGEST,
        control={"fail_create_after_record": True, "fail_rm": True},
    )
    service = container_engine(tmp_path / "state", docker)
    req = request("container-create-ambiguous-quarantine")
    try:
        submit(service, req)
        deadline = time.monotonic() + 15.0
        final = get(service, req)
        while not (final.quarantined or final.state in {"completed", "failed", "interrupted"}) and time.monotonic() < deadline:
            time.sleep(0.02)
            final = get(service, req)
        assert final.state == "running" and final.quarantined is True
        assert "injected rm failure" in final.recovery_error
        assert len(fake.records()) == 1
        cap = capabilities(service)
        with pytest.raises(JobError, match="quarantined"):
            service.retry(
                request_id=req.request_id, request_digest=req.request_digest,
                expected=service.expected_identities,
                interrupted_attempt_id=final.current_attempt.attempt_id,
                capability_id=cap.capability_id, grant_token=TOKEN,
            )
        fake.set_control(fail_rm=False)
        released = service.release_quarantine(req.request_id)
        assert released.state == "interrupted" and released.quarantined is False
        assert fake.records() == []
    finally:
        service.close()


def test_cancel_committed_before_start_never_lets_the_worker_run(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    marker = tmp_path / "worker-started"
    service = container_engine(tmp_path / "state", docker, worker_env={"FAKE_JOB_STARTED": str(marker)})
    req = request("container-cancel-before-start", source=b"SLEEP", candidate_timeout_ms=60_000)
    original_attach = service.store.attach_container
    original_job_by_id = service.store._job_by_id
    state = {"attached": False, "reads": 0, "cancelled": False}

    def attach_and_arm(*args, **kwargs):
        attached = original_attach(*args, **kwargs)
        state["attached"] = True
        return attached

    def job_by_id(request_id):
        if state["attached"] and not state["cancelled"]:
            state["reads"] += 1
            if state["reads"] == 2:
                # A Cancel RPC that commits durably after the fence re-read but
                # before the worker starts: the container exists (created) and
                # must never receive the request.
                state["cancelled"] = True
                service.store.cancel(
                    grant_token=TOKEN, request_id=req.request_id,
                    request_digest=req.request_digest, expected=service.expected_identities,
                )
        return original_job_by_id(request_id)

    service.store.attach_container = attach_and_arm
    service.store._job_by_id = job_by_id
    try:
        submit(service, req)
        final = wait_state(service, req, {"cancelled"}, timeout=15.0)
        assert state["cancelled"] is True and state["reads"] >= 2
        assert final.current_attempt is not None and final.current_attempt.state == "cancelled"
        assert marker.exists() is False, "a cancelled attempt started its worker"
        assert not any(len(call) > 1 and call[1] == "start" for call in fake.calls())
        assert fake.records() == []
        assert final.result is None
    finally:
        service.close()


def test_container_engine_refuses_a_job_accepted_by_another_image(tmp_path):
    from tcenv.remote.contracts import API_CONTRACT_ID, EVIDENCE_CONTRACT_ID

    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    state = tmp_path / "state"
    req = request("container-stale-build")
    # Accept and queue the job under a different (pre-upgrade) execution
    # identity, exactly as a previous deployment would have left it.
    stale_identity = replace(container_identity(), worker_build=f"sim-external-build+img:{'e' * 64}")
    expected = helpers.expected_identities(stale_identity)
    provenance = stale_identity.external_provenance(
        execution_protocol_version="3",
        api_contract_id=API_CONTRACT_ID, evidence_contract_id=EVIDENCE_CONTRACT_ID,
    )
    store = JobStore(state)
    store.register_grant(grant())
    try:
        capability = store.get_capabilities(
            grant_token=TOKEN, client_nonce="1" * 16, expected=expected.evaluator.to_dict(),
            expected_identities=expected, provenance=provenance,
            api_contract_ids=[API_CONTRACT_ID], evidence_contract_ids=[EVIDENCE_CONTRACT_ID],
            required_features=[], supported_features=frozenset(),
        )
        store.submit(
            request=req, request_digest=req.request_digest, expected=expected,
            provenance=provenance, capability_id=capability.capability_id,
            required_features=[], grant_token=TOKEN,
        )
        assert store.enqueue_accepted(req.request_id).state == "queued"
    finally:
        store.close()

    service = container_engine(state, docker)
    try:
        deadline = time.monotonic() + 15.0
        final = service.store._job_by_id(req.request_id)
        while final.state not in {"interrupted", "failed", "completed"} and time.monotonic() < deadline:
            time.sleep(0.02)
            final = service.store._job_by_id(req.request_id)
        assert final.state == "interrupted"
        assert final.failure is not None and final.failure.code == "worker_build_mismatch"
        assert final.failure.retryable is False
        assert final.quarantined is True
        assert "accepted by a different evaluator identity" in final.recovery_error
        assert final.result is None
        assert fake.records() == []
        assert not any(len(call) > 1 and call[1] in {"create", "start"} for call in fake.calls())
    finally:
        service.close()


def test_container_posture_mismatch_before_start_is_never_executed(tmp_path):
    docker, fake = install_fake_docker(
        tmp_path / "docker", image_digest=IMAGE_DIGEST,
        control={"binds": ["/var/run/docker.sock:/var/run/docker.sock"]},
    )
    marker = tmp_path / "worker-started"
    service = container_engine(tmp_path / "state", docker, worker_env={"FAKE_JOB_STARTED": str(marker)})
    req = request("container-posture-mismatch")
    try:
        submit(service, req)
        final = wait_state(service, req, {"interrupted"})
        assert final.failure is not None and final.failure.code == "worker_container_identity"
        assert marker.exists() is False
        assert fake.records() == []
        assert not any(len(call) > 1 and call[1] == "start" for call in fake.calls())
    finally:
        service.close()


def process_engine_with_image_identity(state_dir: Path) -> JobEngine:
    """A process-supervisor engine that would claim an image-pinned build."""

    from tcenv.remote.supervisor import ProcessAttemptSupervisor

    def _spawn():
        return None

    supervisor = ProcessAttemptSupervisor(
        spawn=_spawn, kill=lambda process: None, group_alive=lambda pid: False,
        forget=lambda process: None,
    )
    return JobEngine(
        identity=container_identity(), grants=[grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_job_engine"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=state_dir,
        supervisor=supervisor,
    )


def _supervisor_config(docker: Path) -> ContainerSupervisorConfig:
    return ContainerSupervisorConfig(
        image="taic-worker-sim:local", image_digest=IMAGE_DIGEST, docker=str(docker),
        env={"HOME": "/tmp", "PYTHONPATH": "/opt/taic"}, kill_timeout_s=5.0,
    )


def _crashed_running_attempt(state: Path, docker: Path, req, *, start: bool = False):
    """Build the exact durable state a crashed frontend leaves behind."""

    from tcenv.remote.contracts import API_CONTRACT_ID, EVIDENCE_CONTRACT_ID

    ident = container_identity()
    expected = helpers.expected_identities(ident)
    store = JobStore(state)
    store.register_grant(grant())
    supervisor = DockerAttemptSupervisor(_supervisor_config(docker))
    try:
        capability = store.get_capabilities(
            grant_token=TOKEN, client_nonce="0" * 16, expected=expected.evaluator.to_dict(),
            expected_identities=expected,
            provenance=ident.external_provenance(
                execution_protocol_version="3",
                api_contract_id=API_CONTRACT_ID, evidence_contract_id=EVIDENCE_CONTRACT_ID,
            ),
            api_contract_ids=[API_CONTRACT_ID], evidence_contract_ids=[EVIDENCE_CONTRACT_ID],
            required_features=[], supported_features=frozenset(),
        )
        store.submit(
            request=req, request_digest=req.request_digest, expected=expected,
            provenance=ident.external_provenance(
                execution_protocol_version="3",
                api_contract_id=API_CONTRACT_ID, evidence_contract_id=EVIDENCE_CONTRACT_ID,
            ),
            capability_id=capability.capability_id, required_features=[], grant_token=TOKEN,
        )
        store.enqueue_accepted(req.request_id)
        claimed = store.claim_next(execution_kind="container")
        assert claimed is not None
        _request, attempt = claimed
        spec = AttemptSpec(
            request=req, attempt=attempt, state_dir=store.state_dir,
            command=(sys.executable, "-m", "tests.support.fake_worker_job_engine"),
            env={},
        )
        daemon = supervisor.daemon_identity()
        assert store.attach_daemon(req.request_id, attempt, daemon=daemon)
        attempt = replace(attempt, container_daemon=daemon)
        handle = supervisor.create(spec)
        assert store.attach_container(
            req.request_id, attempt, container_id=handle.container_id,
            container_name=handle.container_name, image_ref=handle.image_ref,
            image_digest=handle.image_digest, runtime=handle.runtime, daemon=daemon,
        )
        if start:
            # A real crash can also happen after the container was started (and
            # its attach client spawned) but before any result was committed.
            supervisor.start(handle)
            handle.attach and time.sleep(0.2)
    finally:
        store.close()
    return attempt, handle, supervisor


def test_container_crash_reconciles_old_worker_and_never_reruns(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    state = tmp_path / "state"
    req = request("container-crash", source=b"SLEEP", candidate_timeout_ms=60_000)
    attempt, _handle, _supervisor = _crashed_running_attempt(state, docker, req)
    assert len(fake.records()) == 1

    recovered = container_engine(state, docker)
    try:
        snapshot = get(recovered, req)
        assert snapshot.state == "interrupted"
        assert snapshot.quarantined is False
        assert snapshot.current_attempt.attempt_id == attempt.attempt_id
        assert snapshot.failure is not None and snapshot.failure.code == "frontend_restarted"
        # The old container was killed and deleted; the attempt was not queued
        # again and no second container was ever started.
        assert fake.records() == []
        assert len(_attempt_rows(recovered, req)) == 1
        assert not any(len(call) > 1 and call[1] == "start" for call in fake.calls())
    finally:
        recovered.close()


def test_container_crash_with_unverified_cleanup_quarantines_without_retry(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    state = tmp_path / "state"
    req = request("container-crash-quarantine", source=b"SLEEP", candidate_timeout_ms=60_000)
    attempt, _handle, _supervisor = _crashed_running_attempt(state, docker, req)
    fake.set_control(fail_rm=True)
    recovered = container_engine(state, docker)
    try:
        snapshot = get(recovered, req)
        assert snapshot.state == "running" and snapshot.quarantined is True
        # This attempt's container was created but never started, so the
        # injected ``rm`` failure is reported when stopping it.
        assert "injected rm failure" in snapshot.recovery_error
        assert snapshot.current_attempt.attempt_id == attempt.attempt_id
        cap = capabilities(recovered)
        with pytest.raises(JobError, match="quarantined"):
            recovered.retry(
                request_id=req.request_id, request_digest=req.request_digest,
                expected=recovered.expected_identities,
                interrupted_attempt_id=snapshot.current_attempt.attempt_id,
                capability_id=cap.capability_id, grant_token=TOKEN,
            )
        fake.set_control(fail_rm=False)
        released = recovered.release_quarantine(req.request_id)
        assert released.state == "interrupted" and released.quarantined is False
        assert fake.records() == []
        assert len(_attempt_rows(recovered, req)) == 1
    finally:
        recovered.close()


def test_process_supervisor_refuses_to_clean_a_container_attempt(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    state = tmp_path / "state"
    req = request("container-kind-mismatch", source=b"SLEEP")
    attempt, handle, container_supervisor = _crashed_running_attempt(state, docker, req)
    # The same durable state opened by a process-supervisor frontend must fail
    # closed instead of being cleaned with the wrong mechanism.
    process_engine = JobEngine(
        identity=helpers.identity(), grants=[grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_job_engine"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=state,
    )
    try:
        snapshot = process_engine.store._job_by_id(req.request_id)
        assert snapshot.state == "running" and snapshot.quarantined is True
        assert "another execution supervisor" in snapshot.recovery_error
        assert snapshot.current_attempt.attempt_id == attempt.attempt_id
        assert len(fake.records()) == 1
        assert not any(len(call) > 1 and call[1] == "kill" for call in fake.calls())
    finally:
        process_engine.close()
        container_supervisor.cleanup(handle)


def _attempt_rows(service: JobEngine, req) -> list:
    row = service.store._conn.execute(
        "SELECT * FROM attempts WHERE request_id=?", (req.request_id,),
    ).fetchall()
    return list(row)


def test_process_engine_cannot_claim_an_image_pinned_build(tmp_path):
    """A local process worker must never sign a pinned-image worker_build."""

    with pytest.raises(ValueError, match="image-pinned"):
        process_engine_with_image_identity(tmp_path / "state")


def test_process_engine_refuses_a_job_accepted_by_the_container_plane(tmp_path):
    """The mirror case: a pinned-image job must not run on local processes."""

    from tcenv.remote.contracts import API_CONTRACT_ID, EVIDENCE_CONTRACT_ID

    state = tmp_path / "state"
    req = request("process-stale-container-build")
    accepted_identity = container_identity()
    expected = helpers.expected_identities(accepted_identity)
    provenance = accepted_identity.external_provenance(
        execution_protocol_version="3",
        api_contract_id=API_CONTRACT_ID, evidence_contract_id=EVIDENCE_CONTRACT_ID,
    )
    store = JobStore(state)
    store.register_grant(grant())
    try:
        capability = store.get_capabilities(
            grant_token=TOKEN, client_nonce="2" * 16, expected=expected.evaluator.to_dict(),
            expected_identities=expected, provenance=provenance,
            api_contract_ids=[API_CONTRACT_ID], evidence_contract_ids=[EVIDENCE_CONTRACT_ID],
            required_features=[], supported_features=frozenset(),
        )
        store.submit(
            request=req, request_digest=req.request_digest, expected=expected,
            provenance=provenance, capability_id=capability.capability_id,
            required_features=[], grant_token=TOKEN,
        )
        assert store.enqueue_accepted(req.request_id).state == "queued"
    finally:
        store.close()

    marker = tmp_path / "worker-started"
    process_engine = JobEngine(
        identity=helpers.identity(), grants=[grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_job_engine"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=state,
        worker_env={"FAKE_JOB_STARTED": str(marker)},
    )
    try:
        assert process_engine.supervisor.kind == "process"
        deadline = time.monotonic() + 15.0
        final = process_engine.store._job_by_id(req.request_id)
        while final.state not in {"interrupted", "failed", "completed"} and time.monotonic() < deadline:
            time.sleep(0.02)
            final = process_engine.store._job_by_id(req.request_id)
        assert final.state == "interrupted"
        assert final.failure is not None and final.failure.code == "worker_build_mismatch"
        assert final.quarantined is True
        assert final.result is None
        assert marker.exists() is False
    finally:
        process_engine.close()


def test_malformed_result_is_not_converted_into_a_candidate_timeout(tmp_path):
    docker, _fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(ContainerSupervisorConfig(
        image="taic-worker-sim:local", image_digest=IMAGE_DIGEST, docker=str(docker),
        env={"HOME": "/tmp", "PYTHONPATH": "/opt/taic"}, kill_timeout_s=5.0,
    ))
    clock = _FakeClock()
    service = JobEngine(
        identity=container_identity(), grants=[grant()],
        worker_command=[sys.executable, "-c", _FAULT_WORKER_SOURCE],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=tmp_path / "state",
        worker_env={
            "FAULT_WORKER_MODE": "wrong_task_hang",
            "FAULT_WORKER_READY": str(tmp_path / "worker-ready"),
        },
        supervisor=supervisor, monotonic=clock.now,
    )
    req = request("container-wrong-task-hang", source=b"OK", candidate_timeout_ms=60_000)
    try:
        submit(service, req)
        _wait_for_marker(tmp_path / "worker-ready")
        clock.advance(61.0)
        final = wait_state(service, req, {"failed", "interrupted", "completed"}, timeout=20.0)
        # The malformed report is known before the deadline; a timeout report
        # would silently hide it.
        # Deterministic gate: the malformed frame reached the pipe before the
        # deadline decision, so this is the validation branch and not a plain
        # pre-frame timeout that happens to look similar.
        assert (tmp_path / "worker-ready").is_file()
        assert final.state == "failed"
        assert final.failure is not None and final.failure.code == "worker_result_invalid"
        assert "task" in final.failure.message
        assert final.result is None
        assert final.quarantined is False
    finally:
        service.close()


def test_second_result_blocks_the_commit_of_the_first(tmp_path):
    docker, _fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(ContainerSupervisorConfig(
        image="taic-worker-sim:local", image_digest=IMAGE_DIGEST, docker=str(docker),
        env={"HOME": "/tmp", "PYTHONPATH": "/opt/taic"}, kill_timeout_s=5.0,
    ))
    clock = _FakeClock()
    service = JobEngine(
        identity=container_identity(), grants=[grant()],
        worker_command=[sys.executable, "-c", _FAULT_WORKER_SOURCE],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=tmp_path / "state",
        worker_env={
            "FAULT_WORKER_MODE": "double_result",
            "FAULT_WORKER_READY": str(tmp_path / "worker-ready"),
        },
        supervisor=supervisor, monotonic=clock.now,
    )
    req = request("container-double-result", source=b"OK", candidate_timeout_ms=60_000)
    try:
        submit(service, req)
        _wait_for_marker(tmp_path / "worker-ready")
        clock.advance(61.0)
        final = wait_state(service, req, {"failed", "interrupted", "completed"}, timeout=20.0)
        assert (tmp_path / "worker-ready").is_file()
        assert final.state == "failed"
        assert final.failure is not None and final.failure.code == "worker_result_invalid"
        assert "multiple results" in final.failure.message
        assert final.result is None
    finally:
        service.close()


def test_reader_that_never_finishes_blocks_the_commit(tmp_path):
    docker, _fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(ContainerSupervisorConfig(
        image="taic-worker-sim:local", image_digest=IMAGE_DIGEST, docker=str(docker),
        env={"HOME": "/tmp", "PYTHONPATH": "/opt/taic"}, kill_timeout_s=5.0,
    ))
    service = JobEngine(
        identity=container_identity(), grants=[grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_bad_output"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=tmp_path / "state",
        worker_env={"FAULT_WORKER_MODE": "result_then_blocked_progress"}, supervisor=supervisor,
    )
    original = service.store.record_progress

    def slow_progress(*args, **kwargs):
        time.sleep(3.0)  # the reader is still inside this when the worker exits
        return original(*args, **kwargs)

    service.store.record_progress = slow_progress
    req = request("container-blocked-reader", source=b"OK", candidate_timeout_ms=20_000)
    try:
        submit(service, req)
        final = wait_state(service, req, {"failed", "interrupted", "completed"}, timeout=30.0)
        assert final.state == "interrupted"
        assert final.failure is not None and final.failure.code == "worker_protocol_error"
        assert "not fully read" in final.failure.message
        assert final.result is None
    finally:
        service.close()


class _ExplodingCreateSupervisor(DockerAttemptSupervisor):
    """Create the container, then fail with an untyped supervisor exception."""

    def create(self, spec):
        super().create(spec)
        raise RuntimeError("unexpected create failure after the container existed")


def test_generic_create_exception_is_cleaned_by_durable_identity(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = _ExplodingCreateSupervisor(ContainerSupervisorConfig(
        image="taic-worker-sim:local", image_digest=IMAGE_DIGEST, docker=str(docker),
        env={"HOME": "/tmp", "PYTHONPATH": "/opt/taic"}, kill_timeout_s=5.0,
    ))
    service = JobEngine(
        identity=container_identity(), grants=[grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_job_engine"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=tmp_path / "state",
        supervisor=supervisor,
    )
    req = request("container-generic-create-failure")
    try:
        submit(service, req)
        final = wait_state(service, req, {"interrupted"}, timeout=20.0)
        assert final.quarantined is False
        assert final.failure is not None and final.failure.code == "worker_unavailable"
        # The container really existed (the daemon created it); the engine
        # never received its id, so only the durable name/label can clean it.
        assert fake.records() == []
        assert any(len(call) > 1 and call[1] == "rm" for call in fake.calls())
        assert service._scheduler.is_alive()
    finally:
        service.close()


def test_generic_cleanup_exception_quarantines_and_keeps_the_scheduler(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(ContainerSupervisorConfig(
        image="taic-worker-sim:local", image_digest=IMAGE_DIGEST, docker=str(docker),
        env={"HOME": "/tmp", "PYTHONPATH": "/opt/taic"}, kill_timeout_s=5.0,
    ))
    original_cleanup = supervisor.cleanup

    def exploding_cleanup(handle):
        raise RuntimeError("untyped cleanup failure")

    supervisor.cleanup = exploding_cleanup
    service = JobEngine(
        identity=container_identity(), grants=[grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_job_engine"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=tmp_path / "state",
        supervisor=supervisor,
    )
    req = request("container-generic-cleanup-failure")
    try:
        submit(service, req)
        deadline = time.monotonic() + 20.0
        final = get(service, req)
        while not final.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            final = get(service, req)
        assert final.quarantined is True
        assert final.state == "running"
        assert "untyped cleanup failure" in final.recovery_error
        assert final.result is None
        cap = capabilities(service)
        with pytest.raises(JobError, match="quarantined"):
            service.retry(
                request_id=req.request_id, request_digest=req.request_digest,
                expected=service.expected_identities,
                interrupted_attempt_id=final.current_attempt.attempt_id,
                capability_id=cap.capability_id, grant_token=TOKEN,
            )
        # The scheduler loop survived the untyped supervisor failure.
        assert service._scheduler.is_alive()
        supervisor.cleanup = original_cleanup
        assert service.store._job_by_id(req.request_id).quarantined is True
    finally:
        supervisor.cleanup = original_cleanup
        service.close()


class _WriterGateLock:
    """``_active_lock`` wrapper that parks the request-writer thread at its gate.

    The test uses it to reach the writer deterministically: the writer thread
    announces itself, waits for the test to release the gate, and only then
    acquires the real lock.  A cancellation that commits in that window must
    make the writer decline, and the durable commit is observable before the
    gate opens.
    """

    def __init__(self, inner, *, writer_name: str) -> None:
        self._inner = inner
        self._writer_name = writer_name
        self.writer_waiting = threading.Event()
        self.writer_gate = threading.Event()
        self.writer_entered = threading.Event()
        self.writer_saw_cancel: list[bool] = []

    def _is_writer(self) -> bool:
        return threading.current_thread().name == self._writer_name

    def __enter__(self):
        if self._is_writer():
            self.writer_waiting.set()
            assert self.writer_gate.wait(30), "the test never released the writer gate"
        result = self._inner.__enter__()
        if self._is_writer():
            self.writer_entered.set()
        return result

    def __exit__(self, *exc_info):
        return self._inner.__exit__(*exc_info)

    def acquire(self, *args, **kwargs):
        return self._inner.acquire(*args, **kwargs)

    def release(self):
        return self._inner.release()


def test_cancel_rpc_committed_before_start_never_runs_the_worker(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    marker = tmp_path / "worker-started"
    order = tmp_path / "worker-order.jsonl"
    service = JobEngine(
        identity=container_identity(), grants=[grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_job_engine"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=tmp_path / "state",
        worker_env={"FAKE_JOB_STARTED": str(marker), "FAKE_JOB_ORDER": str(order)},
        supervisor=DockerAttemptSupervisor(ContainerSupervisorConfig(
            image="taic-worker-sim:local", image_digest=IMAGE_DIGEST, docker=str(docker),
            env={"HOME": "/tmp", "PYTHONPATH": "/opt/taic"}, kill_timeout_s=5.0,
        )),
        cancel_teardown_s=0.3,
    )
    req = request("cancel-rpc-before-start", source=b"SLEEP", candidate_timeout_ms=60_000)
    # The engine reaches the writer (the worker really starts), and the writer
    # is parked at the exact authorization the cancellation must win.
    gate = _WriterGateLock(service._active_lock, writer_name="tcenv-worker-stdin")
    service._active_lock = gate
    committed = threading.Event()
    rpc_snapshot: dict = {}
    original_store_cancel = service.store.cancel

    def cancel_and_signal(*args, **kwargs):
        snapshot = original_store_cancel(*args, **kwargs)
        # Durable cancellation is committed *while* the cancel holds the lock.
        committed.set()
        return snapshot

    service.store.cancel = cancel_and_signal

    def cancel_when_the_writer_is_parked():
        assert gate.writer_waiting.wait(30), "the writer gate was never reached"
        rpc_snapshot["snapshot"] = service.cancel(
            request_id=req.request_id, request_digest=req.request_digest,
            expected=service.expected_identities, grant_token=TOKEN,
        )

    canceller = threading.Thread(target=cancel_when_the_writer_is_parked, name="cancel-rpc")
    canceller.start()
    try:
        submit(service, req)
        assert committed.wait(30), "the cancellation never committed"
        # The worker process really started (so the writer is the thing that has
        # to decline), independent of whether it ever received the request.
        deadline = time.monotonic() + 20.0
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), "the worker never started; the writer gate was not reached"
        gate.writer_gate.set()
        canceller.join(timeout=30)
        assert not canceller.is_alive()
        final = wait_state(service, req, {"cancelled"}, timeout=20.0)
        assert gate.writer_entered.is_set(), "the writer never reached its authorization"
        rpc = rpc_snapshot["snapshot"]
        assert rpc.state in {"cancelled", "running"}
        if rpc.state == "running":
            assert rpc.cancel_requested is True
        # The worker was started and then killed, but it never received one
        # candidate byte: the writer authorized after the durable cancellation.
        assert not order.exists(), "a cancelled attempt received the candidate"
        assert any(len(call) > 1 and call[1] == "start" for call in fake.calls())
        assert "kill" in [call[1] for call in fake.calls() if len(call) > 1]
        assert fake.records() == []
        assert final.result is None
        assert final.current_attempt is not None and final.current_attempt.state == "cancelled"
    finally:
        gate.writer_gate.set()
        service.close()


def test_cancel_rpc_does_not_claim_cancelled_while_teardown_is_outstanding(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(ContainerSupervisorConfig(
        image="taic-worker-sim:local", image_digest=IMAGE_DIGEST, docker=str(docker),
        env={"HOME": "/tmp", "PYTHONPATH": "/opt/taic"}, kill_timeout_s=5.0,
    ))
    service = JobEngine(
        identity=container_identity(), grants=[grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_job_engine"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=tmp_path / "state",
        supervisor=supervisor, cancel_teardown_s=0.2,
    )
    original_kill = supervisor.kill

    def slow_kill(handle):
        time.sleep(1.5)
        return original_kill(handle)

    supervisor.kill = slow_kill
    req = request("cancel-teardown-outstanding", source=b"SLEEP", candidate_timeout_ms=60_000)
    try:
        submit(service, req)
        deadline = time.monotonic() + 15.0
        while not fake.records() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert fake.records(), "container was never created"
        cancelled = service.cancel(
            request_id=req.request_id, request_digest=req.request_digest,
            expected=service.expected_identities, grant_token=TOKEN,
        )
        # The kill consumed the whole teardown window, so the RPC must not
        # report "cancelled" while the container may still be alive.
        assert cancelled.state == "running" and cancelled.cancel_requested is True
        final = wait_state(service, req, {"cancelled"}, timeout=20.0)
        assert final.state == "cancelled"
        assert fake.records() == []
    finally:
        supervisor.kill = original_kill
        service.close()


def test_daemon_switch_quarantines_recovery_instead_of_faking_absence(tmp_path):
    root = tmp_path
    docker_a, fake_a = install_fake_docker(root / "docker-a", image_digest=IMAGE_DIGEST)
    docker_b, fake_b = install_fake_docker(root / "docker-b", image_digest=IMAGE_DIGEST)
    state = root / "state"
    req = request("container-daemon-switch", source=b"SLEEP", candidate_timeout_ms=60_000)
    _attempt, handle, supervisor_a = _crashed_running_attempt(state, docker_a, req)
    assert len(fake_a.records()) == 1
    service_b = container_engine(state, docker_b)
    try:
        snapshot = get(service_b, req)
        assert snapshot.state == "running" and snapshot.quarantined is True
        assert "another Docker daemon" in snapshot.recovery_error
        assert snapshot.result is None
        cap = capabilities(service_b)
        with pytest.raises(JobError, match="quarantined"):
            service_b.retry(
                request_id=req.request_id, request_digest=req.request_digest,
                expected=service_b.expected_identities,
                interrupted_attempt_id=snapshot.current_attempt.attempt_id,
                capability_id=cap.capability_id, grant_token=TOKEN,
            )
        # The old daemon's live container was never touched by the new engine.
        assert len(fake_a.records()) == 1
        assert fake_b.records() == []
    finally:
        service_b.close()
        supervisor_a.cleanup(handle)


def _queued_job_accepted_by(state: Path, req, *, accepted_identity, protocol_version="3") -> None:
    from tcenv.remote.contracts import API_CONTRACT_ID, EVIDENCE_CONTRACT_ID

    expected = helpers.expected_identities(accepted_identity)
    provenance = accepted_identity.external_provenance(
        execution_protocol_version=protocol_version,
        api_contract_id=API_CONTRACT_ID, evidence_contract_id=EVIDENCE_CONTRACT_ID,
    )
    store = JobStore(state)
    store.register_grant(grant())
    try:
        capability = store.get_capabilities(
            grant_token=TOKEN, client_nonce="3" * 16, expected=expected.evaluator.to_dict(),
            expected_identities=expected, provenance=provenance,
            api_contract_ids=[API_CONTRACT_ID], evidence_contract_ids=[EVIDENCE_CONTRACT_ID],
            required_features=[], supported_features=frozenset(),
        )
        store.submit(
            request=req, request_digest=req.request_digest, expected=expected,
            provenance=provenance, capability_id=capability.capability_id,
            required_features=[], grant_token=TOKEN,
        )
        assert store.enqueue_accepted(req.request_id).state == "queued"
    finally:
        store.close()


def _wait_for_state(service: JobEngine, req, states: set[str], timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    snapshot = service.store._job_by_id(req.request_id)
    while snapshot.state not in states and time.monotonic() < deadline:
        time.sleep(0.02)
        snapshot = service.store._job_by_id(req.request_id)
    return snapshot


def test_process_engine_refuses_a_job_accepted_by_another_process_build(tmp_path):
    """Build A -> build B on the same plane must not be signed as A."""

    state = tmp_path / "state"
    req = request("process-build-drift")
    accepted = replace(helpers.identity(), worker_build="process-build-A")
    current = replace(helpers.identity(), worker_build="process-build-B")
    _queued_job_accepted_by(state, req, accepted_identity=accepted)
    service = JobEngine(
        identity=current, grants=[grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_job_engine"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=state,
    )
    try:
        final = _wait_for_state(service, req, {"interrupted", "failed", "completed"})
        assert final.state == "interrupted"
        assert final.failure is not None and final.failure.code == "worker_build_mismatch"
        assert final.result is None and final.quarantined is True
        assert "different evaluator identity" in final.recovery_error
    finally:
        service.close()


def test_container_engine_refuses_the_same_image_at_another_epoch(tmp_path):
    """Same digest but a different evaluator epoch is still another evaluator."""

    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    state = tmp_path / "state"
    req = request("container-epoch-drift")
    _queued_job_accepted_by(state, req, accepted_identity=container_identity())
    engine = JobEngine(
        identity=container_identity(), grants=[grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_job_engine"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=state,
        execution_protocol_version="4",
        supervisor=DockerAttemptSupervisor(ContainerSupervisorConfig(
            image="taic-worker-sim:local", image_digest=IMAGE_DIGEST, docker=str(docker),
            env={"HOME": "/tmp", "PYTHONPATH": "/opt/taic"}, kill_timeout_s=5.0,
        )),
    )
    try:
        final = _wait_for_state(engine, req, {"interrupted", "failed", "completed"})
        assert final.state == "interrupted"
        assert final.failure is not None and final.failure.code == "worker_build_mismatch"
        assert final.result is None and final.quarantined is True
        assert not any(len(call) > 1 and call[1] in {"create", "start"} for call in fake.calls())
    finally:
        engine.close()


def test_a_valid_result_followed_by_a_hang_is_a_timeout_not_ok(tmp_path):
    """The candidate deadline outranks a result the worker never finished."""

    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_supervisor_config(docker))
    clock = _FakeClock()
    service = JobEngine(
        identity=container_identity(), grants=[grant()],
        worker_command=[sys.executable, "-c", _FAULT_WORKER_SOURCE],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=tmp_path / "state",
        worker_env={
            "FAULT_WORKER_MODE": "valid_hang",
            "FAULT_WORKER_READY": str(tmp_path / "worker-ready"),
        },
        supervisor=supervisor, monotonic=clock.now,
    )
    req = request("container-valid-hang", source=b"OK", candidate_timeout_ms=60_000)
    try:
        submit(service, req)
        _wait_for_marker(tmp_path / "worker-ready")
        clock.advance(61.0)
        final = wait_state(service, req, {"completed", "failed", "interrupted"}, timeout=30.0)
        # A complete, valid report exists - but the worker never returned inside
        # its deadline, so committing "ok" would turn a hang into a speed-up.
        assert (tmp_path / "worker-ready").is_file()
        assert final.state == "completed"
        assert final.result is not None
        report = final.result.report_json()
        assert report["status"] == "timeout"
        assert report["stage"] == "timeout"
        assert "deadline" in report["message"]
        assert fake.records() == []
    finally:
        service.close()


def test_an_unverified_cleanup_is_also_an_unverified_shutdown(tmp_path):
    """A quarantined cleanup must not let close() claim a verified teardown."""

    docker, fake = install_fake_docker(
        tmp_path / "docker", image_digest=IMAGE_DIGEST, control={"fail_rm": True},
    )
    service = container_engine(tmp_path / "state", docker)
    req = request("container-cleanup-hazard", source=b"OK")
    try:
        submit(service, req)
        wait_state(service, req, {"interrupted", "failed", "completed", "running"}, timeout=20.0)
        deadline = time.monotonic() + 20.0
        snapshot = service.store._job_by_id(req.request_id)
        while not snapshot.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = service.store._job_by_id(req.request_id)
        assert snapshot.quarantined is True
        assert "injected rm failure" in snapshot.recovery_error
        assert snapshot.result is None
        # The store still closes, but shutdown is *not* verified: the worker
        # cleanup failure must be visible through close() and close_error.
        assert service.close(wait=20.0) is False
        assert service.close_error is not None
        assert "cleanup" in str(service.close_error) or "residue" in str(service.close_error)
        # The injected rm failure really left the container behind: the hazard
        # is justified, not a bookkeeping artifact.
        assert len(fake.records()) == 1
    finally:
        service.close()


def test_close_does_not_ignore_a_nonempty_kill_result(tmp_path):
    """``Supervisor.kill`` reports an unverified stop as a string, not a raise."""

    service = helpers.engine(tmp_path / "state")
    req = request("close-kill-unverified", source=b"SLEEP", candidate_timeout_ms=1500)
    handle = None
    original_kill = None
    try:
        submit(service, req)
        wait_state(service, req, {"running"}, timeout=10.0)
        # Gate on the durable worker handle: "running" is claimed before create,
        # and close() only exercises the kill path once a handle exists.
        deadline = time.monotonic() + 10.0
        running = service._active
        while (running is None or running.handle is None) and time.monotonic() < deadline:
            time.sleep(0.01)
            running = service._active
        assert running is not None and running.handle is not None
        handle = running.handle
        original_kill = service.supervisor.kill
        service.supervisor.kill = lambda target: "injected kill failure"
        assert service.close(wait=20.0) is False
        assert service.close_error is not None
        assert "teardown was not verified" in str(service.close_error)
    finally:
        if original_kill is not None:
            service.supervisor.kill = original_kill
        if handle is not None and handle.process is not None:
            try:
                os.killpg(os.getpgid(handle.process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                handle.process.wait(timeout=10)
            except Exception:
                pass
        service.close()


def test_a_blocked_request_writer_is_a_hazard_not_a_scheduler_deadlock(tmp_path):
    """900 KB into a worker that never reads stdin, with a failed kill.

    The writer thread is blocked inside ``stdin.write``; closing that buffered
    writer from the scheduler used to deadlock on its lock, so the cleanup
    hazard was never recorded and the process could not shut down.
    """

    service = helpers.engine(tmp_path / "state")
    # A worker that sleeps without ever reading its stdin: the 900 KB request
    # fills the pipe and blocks the writer.
    service.worker_command = [sys.executable, "-c", "import time; time.sleep(60)"]
    processes = []
    original_popen = service._popen

    def recording_popen():
        process = original_popen()
        processes.append(process)
        return process

    service._popen = recording_popen
    original_kill = service.supervisor.kill
    service.supervisor.kill = lambda handle: "injected kill failure"
    req = request("blocked-writer", source=b"x" * 900_000, candidate_timeout_ms=1200)
    try:
        submit(service, req)
        deadline = time.monotonic() + 15.0
        snapshot = service.store._job_by_id(req.request_id)
        while not snapshot.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = service.store._job_by_id(req.request_id)
        assert snapshot.quarantined is True, snapshot
        assert "writer" in (snapshot.recovery_error or "")
        # Shutdown must also return: neither the scheduler nor close() may block
        # on the live writer.  It is not a verified shutdown, so close is False.
        started = time.monotonic()
        assert service.close(wait=10.0) is False
        assert time.monotonic() - started < 9.0
        assert service.close_error is not None
    finally:
        service.supervisor.kill = original_kill
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    process.wait(timeout=10)
                except Exception:
                    pass
        service.close()


def test_an_exit_observed_after_the_deadline_is_still_a_timeout(tmp_path):
    """A late exit code must not turn a deadline overrun into an earned ok.

    ``DockerAttemptSupervisor.wait`` re-inspects the daemon after its attach wait
    times out; a slow daemon query can therefore return a *real* exit code after
    the candidate deadline.  The engine's decision is the absolute deadline, not
    the shape of ``wait``'s return value.
    """

    service = helpers.engine(tmp_path / "state")
    observed: list[float] = []
    original_wait = service.supervisor.wait

    def wait_past_the_deadline(handle, timeout):
        observed.append(timeout)
        time.sleep(max(0.001, float(timeout)))
        running = service._active
        if running is not None and not service._overdue(running):
            # The worker is still inside its budget: a real wait would keep
            # waiting.  Never hand back its (already available) exit code yet.
            return None
        return original_wait(handle, 30.0)

    service.supervisor.wait = wait_past_the_deadline
    req = request("late-exit-after-deadline", source=b"OK", candidate_timeout_ms=1200)
    try:
        submit(service, req)
        final = wait_state(service, req, {"completed", "failed", "interrupted"}, timeout=20.0)
        assert observed, "the worker never started before the deadline"
        assert final.state == "completed"
        assert final.result is not None
        report = final.result.report_json()
        assert report["status"] == "timeout"
        assert report["stage"] == "timeout"
        assert "deadline" in report["message"]
    finally:
        service.close()


def test_a_daemon_switch_during_start_never_writes_the_request(tmp_path, monkeypatch):
    """The daemon re-check after the spawn aborts before the request writer."""

    import tcenv.remote.container_supervisor as supervisor_module

    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    order = tmp_path / "worker-order.jsonl"
    service = container_engine(
        tmp_path / "state", docker, worker_env={"FAKE_JOB_ORDER": str(order)},
    )
    daemon_a = service.supervisor.daemon_identity()
    original_popen = supervisor_module.popen_group

    def popen_then_switch(*args, **kwargs):
        fake.set_control(daemon_id="fake-daemon-b")
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(supervisor_module, "popen_group", popen_then_switch)
    req = request("container-daemon-switch-during-start", candidate_timeout_ms=60_000)
    try:
        submit(service, req)
        deadline = time.monotonic() + 20.0
        snapshot = service.store._job_by_id(req.request_id)
        while not snapshot.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = service.store._job_by_id(req.request_id)
        # The cleanup was attempted against A while the endpoint answered B, so
        # the residue hazard is kept (quarantine) instead of a fake absence.
        assert snapshot.quarantined is True, snapshot.recovery_error
        assert "another Docker daemon" in (snapshot.recovery_error or "")
        assert snapshot.result is None
        # The worker never received a request: it only appends after read_request.
        assert not order.exists()
        # Operator release after the endpoint is back on A proves the container
        # was still there and is now gone.
        fake.set_control(daemon_id=daemon_a)
        released = service.release_quarantine(req.request_id)
        assert released.state == "interrupted", released.recovery_error
        assert fake.records() == []
        assert not order.exists()
    finally:
        fake.set_control(daemon_id=daemon_a)
        service.close()


def test_daemon_switch_after_create_quarantines_and_keeps_the_old_container(tmp_path):
    """A / B daemon switch after ``create`` must not read as a clean absence."""

    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_supervisor_config(docker))
    marker = tmp_path / "worker-started"
    service = JobEngine(
        identity=container_identity(), grants=[grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_job_engine"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=tmp_path / "state",
        worker_env={"FAKE_JOB_STARTED": str(marker)}, supervisor=supervisor,
    )
    req = request("container-daemon-switch-after-create", source=b"OK", candidate_timeout_ms=60_000)
    original_create = supervisor.create

    def create_then_switch_daemon(spec):
        handle = original_create(spec)
        # The engine's Docker endpoint now answers as another Engine: the
        # container it created is invisible here but still exists over there.
        fake.set_control(daemon_id="fake-daemon-b")
        return handle

    supervisor.create = create_then_switch_daemon
    try:
        submit(service, req)
        snapshot = wait_state(service, req, {"running", "interrupted", "failed", "completed"}, timeout=30.0)
        deadline = time.monotonic() + 20.0
        while not snapshot.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = service.store._job_by_id(req.request_id)
        assert snapshot.quarantined is True
        assert "another Docker daemon" in snapshot.recovery_error
        assert snapshot.result is None
        # verify_identity and every cleanup/absence proof stayed bound to daemon
        # A, so the container is still there and nothing was started or written.
        assert marker.exists() is False
        assert len(fake.records()) == 1
        assert not any(len(call) > 1 and call[1] == "start" for call in fake.calls())
        cap = capabilities(service)
        with pytest.raises(JobError, match="quarantined"):
            service.retry(
                request_id=req.request_id, request_digest=req.request_digest,
                expected=service.expected_identities,
                interrupted_attempt_id=snapshot.current_attempt.attempt_id,
                capability_id=cap.capability_id, grant_token=TOKEN,
            )
    finally:
        service.close()
        fake.set_control(daemon_id=fake.control().get("daemon_id", ""))
        for record in fake.records():
            fake._cmd_rm(["--force", record["Id"]])


def test_container_engine_refuses_a_queued_job_whose_provenance_drifted(tmp_path):
    """A provenance field other than worker_build still names another evaluator."""

    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    state = tmp_path / "state"
    req = request("container-provenance-drift")
    # Same expected identities (so the build/epoch comparison passes), but the
    # job was accepted with a provenance this frontend must never sign.
    accepted = replace(container_identity(), publication_allowed=True)
    _queued_job_accepted_by(state, req, accepted_identity=accepted)
    service = container_engine(state, docker)
    marker = tmp_path / "worker-started"
    service.worker_env["FAKE_JOB_STARTED"] = str(marker)
    try:
        final = _wait_for_state(service, req, {"interrupted", "failed", "completed"})
        assert final.state == "interrupted"
        assert final.failure is not None and final.failure.code == "worker_build_mismatch"
        assert "publication_allowed" in final.failure.message
        assert final.quarantined is True and final.result is None
        assert fake.records() == []
        assert not any(len(call) > 1 and call[1] in {"create", "start"} for call in fake.calls())
        assert marker.exists() is False
    finally:
        service.close()


class _ResidueStubSupervisor:
    """A container supervisor whose cleanup hides a surviving attach client."""

    kind = "container"
    image_digest = IMAGE_DIGEST

    def __init__(self) -> None:
        self.daemon = "stub-daemon-1"
        self.residue = "container attach client survived cleanup"
        self.calls: list[str] = []
        self.released = 0

    def daemon_identity(self) -> str:
        return self.daemon

    def expected_worker_build(self, name: str) -> str:
        return f"{name}+img:{IMAGE_DIGEST}"

    def create(self, spec):
        return WorkerHandle(
            spec=spec, kind=self.kind, container_id="a" * 64,
            container_name=f"tcenv-attempt-{spec.attempt.attempt_id}",
            image_ref="taic-worker-sim:local", image_digest=IMAGE_DIGEST,
            runtime="runc", container_daemon=self.daemon,
        )

    def verify_identity(self, handle) -> str:
        self.calls.append("verify_identity")
        raise RuntimeError("identity probe exploded")

    def start(self, handle) -> None:  # pragma: no cover - must never run
        raise AssertionError("a failed identity check must never start the worker")

    def streams(self, handle):  # pragma: no cover - must never run
        raise AssertionError("a failed identity check must never stream")

    def kill(self, handle) -> str:  # pragma: no cover - cleanup() is used
        self.calls.append("kill")
        return ""

    def delete(self, handle) -> str:  # pragma: no cover - cleanup() is used
        self.calls.append("delete")
        return ""

    def verify_absent(self, handle) -> str:  # pragma: no cover - cleanup() is used
        self.calls.append("verify_absent")
        return ""

    def cleanup(self, handle) -> str:
        self.calls.append("cleanup")
        return ""

    def cleanup_residue(self, handle) -> str:
        self.calls.append("cleanup_residue")
        return self.residue

    def cleanup_recovered(self, attempt, *, state_dir, daemon: str = "") -> str:
        return ""

    def release(self, handle) -> None:
        self.released += 1


def test_generic_supervisor_exception_verifies_residue_and_quarantines(tmp_path):
    """A verified kill/delete is not a verified residue-free worker."""

    supervisor = _ResidueStubSupervisor()
    service = JobEngine(
        identity=container_identity(), grants=[grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_job_engine"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=tmp_path / "state",
        supervisor=supervisor,
    )
    req = request("container-residue-exception", source=b"OK", candidate_timeout_ms=60_000)
    try:
        submit(service, req)
        snapshot = service.store._job_by_id(req.request_id)
        deadline = time.monotonic() + 20.0
        while not snapshot.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = service.store._job_by_id(req.request_id)
        # cleanup() reported success but the residue check found the surviving
        # attach client, so the attempt must stay quarantined (no Retry, no result).
        assert snapshot.quarantined is True
        assert "attach client survived" in snapshot.recovery_error
        assert snapshot.result is None
        assert supervisor.calls[:2] == ["verify_identity", "cleanup"]
        assert "cleanup_residue" in supervisor.calls
        assert supervisor.released == 1
    finally:
        service.close()


def test_recovery_reaps_a_surviving_attach_client(tmp_path):
    """A crashed frontend's leftover attach client is residue, not a clean end."""

    docker, fake = install_fake_docker(
        tmp_path / "docker", image_digest=IMAGE_DIGEST, control={"hang_attach": True},
    )
    state = tmp_path / "state"
    req = request("container-hang-attach-crash", source=b"OK", candidate_timeout_ms=60_000)
    attempt, handle, supervisor = _crashed_running_attempt(state, docker, req, start=True)
    attach = handle.attach
    assert attach is not None and attach.poll() is None
    fake.set_control(hang_attach=False)
    recovered = container_engine(state, docker)
    try:
        snapshot = get(recovered, req)
        assert snapshot.state == "interrupted"
        assert snapshot.quarantined is False, snapshot.recovery_error
        assert snapshot.failure is not None and snapshot.failure.code == "frontend_restarted"
        assert snapshot.result is None
        assert attach.wait(timeout=20) == -signal.SIGKILL
        assert process_state(attach.pid) is None
        assert fake.records() == []
    finally:
        recovered.close()
        supervisor.cleanup(handle)
        supervisor.release(handle)


def test_operator_release_reaps_a_surviving_attach_client(tmp_path):
    """Operator release rechecks the same residue before it clears quarantine."""

    docker, fake = install_fake_docker(
        tmp_path / "docker", image_digest=IMAGE_DIGEST, control={"hang_attach": True},
    )
    state = tmp_path / "state"
    req = request("container-hang-attach-release", source=b"OK", candidate_timeout_ms=60_000)
    attempt, handle, supervisor = _crashed_running_attempt(state, docker, req, start=True)
    attach = handle.attach
    assert attach is not None and attach.poll() is None
    fake.set_control(fail_rm=True)
    quarantined = container_engine(state, docker)
    try:
        snapshot = get(quarantined, req)
        assert snapshot.state == "running" and snapshot.quarantined is True
        # The injected delete failure is what blocked the first recovery.
        assert "injected rm failure" in snapshot.recovery_error
        fake.set_control(fail_rm=False, hang_attach=False)
        released = quarantined.release_quarantine(req.request_id)
        assert released.state == "interrupted", released.recovery_error
        assert released.quarantined is False, released.recovery_error
        assert attach.wait(timeout=20) == -signal.SIGKILL
        assert process_state(attach.pid) is None
        assert fake.records() == []
    finally:
        quarantined.close()
        supervisor.cleanup(handle)
        supervisor.release(handle)


def test_slow_create_never_starts_the_container_after_the_deadline(tmp_path):
    docker, fake = install_fake_docker(
        tmp_path / "docker", image_digest=IMAGE_DIGEST, control={"slow_create_s": 1.5},
    )
    marker = tmp_path / "worker-started"
    order = tmp_path / "worker-order.jsonl"
    service = container_engine(
        tmp_path / "state", docker,
        worker_env={"FAKE_JOB_STARTED": str(marker), "FAKE_JOB_ORDER": str(order)},
    )
    req = request("container-slow-create", source=b"OK", candidate_timeout_ms=500)
    try:
        submit(service, req)
        final = wait_state(service, req, {"completed", "failed", "interrupted"}, timeout=30.0)
        # ``create`` themselves overran the deadline: the container exists but
        # must be deleted without ever being started or written to.
        assert final.state == "completed"
        assert final.result is not None and final.result.report_json()["status"] == "timeout"
        assert final.quarantined is False
        assert marker.exists() is False
        assert not order.exists()
        assert fake.records() == []
        assert not any(len(call) > 1 and call[1] == "start" for call in fake.calls())
    finally:
        service.close()


def test_identity_probe_that_spends_the_deadline_never_starts_the_container(tmp_path):
    """The container was created, but the identity probe after it ran out the clock."""

    docker, fake = install_fake_docker(
        tmp_path / "docker", image_digest=IMAGE_DIGEST, control={"slow_inspect_s": 0.8},
    )
    marker = tmp_path / "worker-started"
    order = tmp_path / "worker-order.jsonl"
    service = container_engine(
        tmp_path / "state", docker,
        worker_env={"FAKE_JOB_STARTED": str(marker), "FAKE_JOB_ORDER": str(order)},
    )
    req = request("container-slow-identity", source=b"OK", candidate_timeout_ms=500)
    try:
        submit(service, req)
        final = wait_state(service, req, {"completed", "failed", "interrupted"}, timeout=30.0)
        # ``create`` was fast enough; only the post-create identity probe overran
        # the deadline, so the pre-start check must delete the container unstarted.
        assert final.state == "completed"
        assert final.result is not None and final.result.report_json()["status"] == "timeout"
        assert final.quarantined is False
        assert marker.exists() is False
        assert not order.exists()
        assert fake.records() == []
        assert not any(len(call) > 1 and call[1] == "start" for call in fake.calls())
    finally:
        service.close()


def test_overdue_daemon_query_never_creates_or_starts_a_container(tmp_path):
    docker, fake = install_fake_docker(
        tmp_path / "docker", image_digest=IMAGE_DIGEST, control={"slow_info_s": 0.8},
    )
    marker = tmp_path / "worker-started"
    order = tmp_path / "worker-order.jsonl"
    service = container_engine(
        tmp_path / "state", docker,
        worker_env={"FAKE_JOB_STARTED": str(marker), "FAKE_JOB_ORDER": str(order)},
    )
    req = request("container-slow-info", source=b"OK", candidate_timeout_ms=300)
    try:
        submit(service, req)
        final = wait_state(service, req, {"completed", "failed", "interrupted"}, timeout=30.0)
        assert final.state == "completed"
        assert final.result is not None and final.result.report_json()["status"] == "timeout"
        assert marker.exists() is False
        assert not order.exists()
        assert fake.records() == []
        assert not any(
            len(call) > 1 and call[1] in {"create", "start"} for call in fake.calls()
        )
    finally:
        service.close()


def test_revocation_commits_and_cancels_under_the_start_lock(tmp_path):
    """A durable revocation must not be overtaken by a start or a request write."""

    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    order = tmp_path / "worker-order.jsonl"
    service = container_engine(
        tmp_path / "state", docker,
        worker_env={"FAKE_JOB_ORDER": str(order)},
    )
    req = request("revoke-paused", source=b"SLEEP", candidate_timeout_ms=60_000)
    attached = threading.Event()
    attach_release = threading.Event()
    committed = threading.Event()
    resume = threading.Event()
    original_attach = service.store.attach_container
    original_revoke = service.store.revoke_grant

    def attach_then_pause(*args, **kwargs):
        result = original_attach(*args, **kwargs)
        attached.set()
        assert attach_release.wait(30), "the test never released the durable attach"
        return result

    def revoke_then_pause(token):
        scope_id = original_revoke(token)  # durable revocation is committed here
        committed.set()
        assert resume.wait(30), "the test never released the revocation"
        return scope_id

    service.store.attach_container = attach_then_pause
    service.store.revoke_grant = revoke_then_pause
    revoker = threading.Thread(target=lambda: service.revoke_grant(TOKEN), name="revoke-grant")
    try:
        submit(service, req)
        assert attached.wait(30), "the engine never reached the durable container attach"
        revoker.start()
        # The revocation is durably committed but its scope cancellation has not
        # run yet: exactly the window the old code left open.
        assert committed.wait(30), "the revocation never committed"
        attach_release.set()
        # Give the scheduler time to reach (and, before the fix, pass) the start
        # authorization it must not be able to take before the cancellation.
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and not order.exists():
            time.sleep(0.02)
        assert not order.exists(), "the worker received the candidate after the revocation"
        assert not any(len(call) > 1 and call[1] == "start" for call in fake.calls())
        resume.set()
        revoker.join(timeout=30)
        assert not revoker.is_alive()
        # The grant is revoked, so the token can no longer read the job: read the
        # durable snapshot through the store instead.
        final = service.store._job_by_id(req.request_id)
        deadline = time.monotonic() + 20.0
        while final.state != "cancelled" and time.monotonic() < deadline:
            time.sleep(0.02)
            final = service.store._job_by_id(req.request_id)
        assert final.state == "cancelled"
        assert final.result is None
        assert final.cancel_requested is True
        assert final.failure is not None
        assert final.failure.code in {"cancelled", "grant_revoked"}
        assert not order.exists()
        assert fake.records() == []
        assert not any(len(call) > 1 and call[1] == "start" for call in fake.calls())
    finally:
        attach_release.set()
        resume.set()
        service.close()
