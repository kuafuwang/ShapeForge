"""Job-engine integration tests for the container execution plane (fake Docker)."""

from __future__ import annotations

import json
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
from tcenv.remote.supervisor import AttemptSpec
from tests.support import job_engine as helpers
from tests.support.fake_docker import install_fake_docker
from tests.support.job_engine import (
    TOKEN, capabilities, get, grant, request, submit, wait_state,
)

IMAGE_DIGEST = "d" * 64


def container_identity():
    return replace(helpers.identity(), worker_build=f"sim-external-build+img:{IMAGE_DIGEST}")


def container_engine(
    state_dir: Path, docker: Path, *, worker_env: dict[str, str] | None = None,
    identity=None,
) -> JobEngine:
    supervisor = DockerAttemptSupervisor(ContainerSupervisorConfig(
        image="taic-worker-sim:local", image_digest=IMAGE_DIGEST, docker=str(docker),
        env={"HOME": "/tmp", "PYTHONPATH": "/opt/taic"}, kill_timeout_s=5.0,
    ))
    return JobEngine(
        identity=identity or container_identity(), grants=[grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_job_engine"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=state_dir,
        worker_env=worker_env, supervisor=supervisor,
    )


def test_container_engine_commits_result_only_after_verified_cleanup(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    service = container_engine(tmp_path / "state", docker)
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
    service = container_engine(tmp_path / "state", docker)
    req = request("container-timeout", source=b"SLEEP", candidate_timeout_ms=1200)
    try:
        submit(service, req)
        final = wait_state(service, req, {"completed"}, timeout=15.0)
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


def _crashed_running_attempt(state: Path, docker: Path, req):
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
    service = JobEngine(
        identity=container_identity(), grants=[grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_bad_output"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=tmp_path / "state",
        worker_env={"FAULT_WORKER_MODE": "wrong_task_hang"}, supervisor=supervisor,
    )
    req = request("container-wrong-task-hang", source=b"OK", candidate_timeout_ms=1500)
    try:
        submit(service, req)
        final = wait_state(service, req, {"failed", "interrupted", "completed"}, timeout=20.0)
        # The malformed report is known before the deadline; a timeout report
        # would silently hide it.
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
    service = JobEngine(
        identity=container_identity(), grants=[grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_bad_output"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=tmp_path / "state",
        worker_env={"FAULT_WORKER_MODE": "double_result"}, supervisor=supervisor,
    )
    req = request("container-double-result", source=b"OK", candidate_timeout_ms=1500)
    try:
        submit(service, req)
        final = wait_state(service, req, {"failed", "interrupted", "completed"}, timeout=20.0)
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


def test_cancel_rpc_committed_before_start_never_runs_the_worker(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    marker = tmp_path / "worker-started"
    service = JobEngine(
        identity=container_identity(), grants=[grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_job_engine"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=tmp_path / "state",
        worker_env={"FAKE_JOB_STARTED": str(marker)},
        supervisor=DockerAttemptSupervisor(ContainerSupervisorConfig(
            image="taic-worker-sim:local", image_digest=IMAGE_DIGEST, docker=str(docker),
            env={"HOME": "/tmp", "PYTHONPATH": "/opt/taic"}, kill_timeout_s=5.0,
        )),
        cancel_teardown_s=0.3,
    )
    req = request("cancel-rpc-before-start", source=b"SLEEP", candidate_timeout_ms=60_000)
    original_attach = service.store.attach_container
    entered = threading.Event()
    proceed = threading.Event()
    rpc_snapshot: dict = {}

    def attach_then_pause(*args, **kwargs):
        attached = original_attach(*args, **kwargs)
        entered.set()
        proceed.wait(20)
        return attached

    def cancel_while_paused():
        entered.wait(20)
        rpc_snapshot["snapshot"] = service.cancel(
            request_id=req.request_id, request_digest=req.request_digest,
            expected=service.expected_identities, grant_token=TOKEN,
        )
        proceed.set()

    service.store.attach_container = attach_then_pause
    canceller = threading.Thread(target=cancel_while_paused, name="cancel-rpc")
    canceller.start()
    try:
        submit(service, req)
        canceller.join(timeout=30)
        assert not canceller.is_alive()
        final = wait_state(service, req, {"cancelled"}, timeout=20.0)
        # The RPC committed the durable cancellation while the engine was still
        # between the durable attach and the start authorization.
        rpc = rpc_snapshot["snapshot"]
        assert rpc.state == "cancelled" or (rpc.state == "running" and rpc.cancel_requested)
        assert marker.exists() is False, "a cancelled attempt started its worker"
        assert not any(len(call) > 1 and call[1] == "start" for call in fake.calls())
        assert fake.records() == []
        assert final.result is None
        assert final.current_attempt is not None and final.current_attempt.state == "cancelled"
    finally:
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
