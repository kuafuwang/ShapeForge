"""Job-engine integration tests for the container execution plane (fake Docker)."""

from __future__ import annotations

import json
import sys
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
        assert "does not bind this frontend's pinned image" in final.recovery_error
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
        handle = supervisor.create(spec)
        assert store.attach_container(
            req.request_id, attempt, container_id=handle.container_id,
            container_name=handle.container_name, image_ref=handle.image_ref,
            image_digest=handle.image_digest, runtime=handle.runtime,
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
        identity=container_identity(), grants=[grant()],
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
