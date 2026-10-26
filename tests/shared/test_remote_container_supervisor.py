"""Unit tests for the disposable-container execution plane (fake Docker CLI)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from tcenv.remote.container_supervisor import (
    ContainerSupervisorConfig,
    DockerAttemptSupervisor,
    build_create_args,
    container_name_for,
    container_worker_build,
    normalize_image_digest,
)
from tcenv.remote.job_types import Attempt
from tcenv.remote.supervisor import AttemptSpec, SupervisorError, WorkerStatus
from tests.support.fake_docker import install_fake_docker
from tests.support.job_engine import request

IMAGE_DIGEST = "d" * 64


def _attempt(attempt_id: str = "attempt-7-0123456789abcdef", fence: int = 7) -> Attempt:
    return Attempt(
        attempt_id=attempt_id, lease_fence=fence, state="running", retry_index=0,
        cancel_requested=False, queued_at_ms=0, started_at_ms=1,
        worker_pid=None, worker_pgid=None, worker_start_token="",
        frontend_instance="frontend-test",
    )


def _config(tmp_path: Path, docker: Path, **overrides) -> ContainerSupervisorConfig:
    values = {
        "image": "taic-worker-sim:local",
        "image_digest": IMAGE_DIGEST,
        "docker": str(docker),
        "env": {"HOME": "/tmp", "PYTHONPATH": "/opt/taic"},
        "cli_timeout_s": 30.0,
        "kill_timeout_s": 10.0,
    }
    values.update(overrides)
    return ContainerSupervisorConfig(**values)


def _spec(tmp_path: Path, command: tuple[str, ...], *, source: bytes = b"OK", env: dict[str, str] | None = None) -> AttemptSpec:
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    return AttemptSpec(
        request=request("request-container-1", source=source),
        attempt=_attempt(), state_dir=state, command=command, env=env or {},
    )


def test_create_uses_hardened_profile_pins_image_and_hides_source(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    marker = b"UNTRUSTED_CANDIDATE_SOURCE_MARKER"
    spec = _spec(tmp_path, (sys.executable, "-c", "import sys; sys.stdin.read()"), source=marker)
    handle = supervisor.create(spec)
    try:
        assert handle.container_id and handle.image_digest == IMAGE_DIGEST
        assert handle.container_name == container_name_for(spec.attempt.attempt_id)
        record = fake.record(handle.container_id)
        assert record is not None
        host = record["HostConfig"]
        assert host["NetworkMode"] == "none"
        assert host["ReadonlyRootfs"] is True
        assert host["CapDrop"] == ["ALL"]
        assert any(value.startswith("no-new-privileges") for value in host["SecurityOpt"])
        assert host["Privileged"] is False and host["Binds"] == []
        assert host["PidsLimit"] == 256 and host["Memory"] > 0 and host["NanoCpus"] > 0
        assert record["Config"]["User"] == "65534:65534"
        assert {mount["Destination"] for mount in record["Mounts"]} == {"/scratch", "/tmp"}
        env_names = {line.split("=", 1)[0] for line in record["Config"]["Env"]}
        assert "HOME" in env_names and "TCENV_REMOTE_STATE_DIR" in env_names
        assert not any(name in env_names for name in ("DOCKER_HOST", "SSH_AUTH_SOCK"))
        assert supervisor.verify_identity(handle) == ""
        # The candidate source is only ever written to the private stdin
        # stream; it must not appear in any Docker CLI argument.
        assert all(marker.decode() not in part for call in fake.calls() for part in call)
        create_call = next(call for call in fake.calls() if len(call) > 1 and call[1] == "create")
        assert "--privileged" not in create_call
        assert "-v" not in create_call and "--volume" not in create_call and "--mount" not in create_call
        # The per-attempt env file is consumed at create time and removed.
        assert list((spec.state_dir / "container-env").iterdir()) == []
    finally:
        supervisor.cleanup(handle)
    assert fake.records() == []


def test_create_rejects_unpinned_image_and_credential_environment(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    fake.set_control(image_id="sha256:" + "e" * 64)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    with pytest.raises(SupervisorError, match="pinned digest"):
        supervisor.create(_spec(tmp_path, ("python3", "-c", "pass")))
    assert fake.records() == []
    fake.set_control(image_id=f"sha256:{IMAGE_DIGEST}")
    with pytest.raises(SupervisorError, match="credential"):
        supervisor.create(_spec(tmp_path, ("python3", "-c", "pass"), env={"OPENAI_API_KEY": "sk-not-a-real-key"}))
    with pytest.raises(SupervisorError, match="must not carry DOCKER_HOST"):
        supervisor.create(_spec(tmp_path, ("python3", "-c", "pass"), env={"DOCKER_HOST": "unix:///var/run/docker.sock"}))
    with pytest.raises(SupervisorError, match="multiple lines"):
        supervisor.create(_spec(tmp_path, ("python3", "-c", "pass"), env={"WORKER": "a\nb"}))
    assert fake.records() == []


def test_create_failure_is_bounded_and_creates_no_container(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST, control={"fail_create": True})
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    with pytest.raises(SupervisorError, match="docker create failed"):
        supervisor.create(_spec(tmp_path, ("python3", "-c", "pass")))
    assert fake.records() == []
    assert list((tmp_path / "state" / "container-env").iterdir()) == []


def test_verify_identity_detects_posture_drift(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    spec = _spec(tmp_path, ("python3", "-c", "pass"))
    handle = supervisor.create(spec)
    try:
        record = fake.record(handle.container_id)
        assert record is not None
        _, path = fake._find(handle.container_id)
        assert path is not None
        record["HostConfig"]["NetworkMode"] = "bridge"
        path.write_text(json.dumps(record), encoding="utf-8")
        assert "network mode" in supervisor.verify_identity(handle)
        record["HostConfig"]["NetworkMode"] = "none"
        record["HostConfig"]["Binds"] = ["/var/run/docker.sock:/var/run/docker.sock"]
        path.write_text(json.dumps(record), encoding="utf-8")
        assert "bind mount" in supervisor.verify_identity(handle)
        record["HostConfig"]["Binds"] = []
        record["Config"]["Env"].append("OPENAI_API_KEY=sk-not-real")
        path.write_text(json.dumps(record), encoding="utf-8")
        assert "environment carries" in supervisor.verify_identity(handle)
    finally:
        supervisor.cleanup(handle)


def test_start_wait_and_cleanup_roundtrip(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    spec = _spec(tmp_path, (sys.executable, "-c", "import sys; data=sys.stdin.buffer.read(); sys.stdout.buffer.write(b'ok:'+data)"))
    handle = supervisor.create(spec)
    supervisor.start(handle)
    stdin, stdout, stderr = supervisor.streams(handle)
    assert stdin is not None and stdout is not None and stderr is not None
    stdin.write(b"request-bytes")
    stdin.flush()
    stdin.close()
    assert supervisor.wait(handle, 10.0) == 0
    assert stdout.read() == b"ok:request-bytes"
    assert supervisor.status(handle).state == "exited"
    assert supervisor.cleanup(handle) == ""
    assert supervisor.verify_absent(handle) == ""
    assert supervisor.cleanup_residue(handle) == ""
    assert fake.records() == []
    assert not (tmp_path / "state" / "container-env" / f"{handle.attempt_id}.env").exists()
    supervisor.release(handle)


def test_timeout_root_cleanup_kills_worker_and_child(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    worker = (
        "import subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        "time.sleep(60)"
    )
    handle = supervisor.create(_spec(tmp_path, (sys.executable, "-c", worker)))
    supervisor.start(handle)
    assert supervisor.wait(handle, 0.4) is None
    assert supervisor.status(handle).state == "running"
    assert supervisor.cleanup(handle) == ""
    assert supervisor.verify_absent(handle) == ""
    assert fake.records() == []


def _wait_running(supervisor: DockerAttemptSupervisor, handle, *, timeout: float = 10.0) -> None:
    """``start`` returns before the daemon flips the container to running."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if supervisor.status(handle).state == "running":
            return
        time.sleep(0.02)
    raise AssertionError("container never reached the running state")


def test_kill_and_delete_are_idempotent_and_verify(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    handle = supervisor.create(_spec(tmp_path, (sys.executable, "-c", "import time; time.sleep(30)")))
    supervisor.start(handle)
    _wait_running(supervisor, handle)
    assert supervisor.kill(handle) == ""
    assert supervisor.kill(handle) == ""
    assert supervisor.status(handle).state == "exited"
    assert supervisor.delete(handle) == ""
    assert supervisor.delete(handle) == ""
    assert supervisor.verify_absent(handle) == ""
    supervisor.release(handle)

    survivor = install_fake_docker(
        tmp_path / "docker-survivor", image_digest=IMAGE_DIGEST, control={"ignore_kill": True},
    )
    survivor_supervisor = DockerAttemptSupervisor(_config(tmp_path, survivor[0], kill_timeout_s=1.5))
    handle = survivor_supervisor.create(_spec(tmp_path, (sys.executable, "-c", "import time; time.sleep(30)")))
    survivor_supervisor.start(handle)
    _wait_running(survivor_supervisor, handle)
    error = survivor_supervisor.kill(handle)
    assert "did not stop" in error
    assert survivor_supervisor.cleanup(handle) != ""
    survivor_supervisor.delete(handle)
    survivor_supervisor.release(handle)


def test_kill_of_a_created_container_removes_it_before_it_can_start(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    handle = supervisor.create(_spec(tmp_path, (sys.executable, "-c", "import time; time.sleep(30)")))
    # Never started: there is no process to signal, so the only way to prove a
    # concurrent ``docker start`` cannot run the cancelled candidate is removal.
    assert supervisor.kill(handle) == ""
    assert fake.records() == []
    assert supervisor.verify_absent(handle) == ""
    supervisor.release(handle)


def test_attach_ended_while_container_running_is_an_error(tmp_path):
    docker, fake = install_fake_docker(
        tmp_path / "docker", image_digest=IMAGE_DIGEST, control={"detach_leaves_running": True},
    )
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    handle = supervisor.create(_spec(tmp_path, (sys.executable, "-c", "pass")))
    supervisor.start(handle)
    with pytest.raises(SupervisorError, match="still running"):
        supervisor.wait(handle, 10.0)
    fake.set_control(detach_leaves_running=False)
    assert supervisor.cleanup(handle) == ""
    assert supervisor.verify_absent(handle) == ""
    supervisor.release(handle)


def test_exited_container_without_an_exit_code_is_never_a_timeout(tmp_path, monkeypatch):
    docker, _fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    handle = supervisor.create(_spec(tmp_path, (sys.executable, "-c", "import time; time.sleep(30)")))
    supervisor.start(handle)
    # A malformed status must be a supervisor failure, not a candidate timeout:
    # returning None here would be read as "the deadline expired".
    monkeypatch.setattr(supervisor, "_status_for", lambda ref: (WorkerStatus("exited", None), ""))
    with pytest.raises(SupervisorError, match="without an exit code"):
        supervisor.wait(handle, 0.2)
    monkeypatch.undo()
    assert supervisor.cleanup(handle) == ""
    supervisor.release(handle)


def test_cleanup_recovered_uses_persisted_identity_and_refuses_mismatch(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    daemon = supervisor.daemon_identity()
    handle = supervisor.create(_spec(tmp_path, (sys.executable, "-c", "pass")))
    attempt = Attempt(**{**handle.spec.attempt.__dict__, "container_id": handle.container_id,
                        "container_name": handle.container_name, "execution_kind": "container",
                        "container_daemon": daemon})
    assert supervisor.cleanup_recovered(attempt, state_dir=tmp_path / "state") == ""
    assert fake.records() == []

    # A crash between create and the durable id attach leaves no container id,
    # but the daemon identity is persisted before create.
    docker2, fake2 = install_fake_docker(tmp_path / "docker2", image_digest=IMAGE_DIGEST)
    supervisor2 = DockerAttemptSupervisor(_config(tmp_path, docker2))
    daemon2 = supervisor2.daemon_identity()
    handle = supervisor2.create(_spec(tmp_path, (sys.executable, "-c", "pass")))
    named = Attempt(**{**handle.spec.attempt.__dict__, "container_id": "",
                       "container_name": handle.container_name, "execution_kind": "container",
                       "container_daemon": daemon2})
    assert supervisor2.cleanup_recovered(named, state_dir=tmp_path / "state") == ""
    assert fake2.records() == []

    # A process-supervisor attempt must never be guessed at by the container path.
    process_attempt = Attempt(**{**handle.spec.attempt.__dict__, "execution_kind": "process"})
    assert "not started by the container supervisor" in supervisor2.cleanup_recovered(
        process_attempt, state_dir=tmp_path / "state",
    )

    # Without a persisted daemon (create-time crash before the daemon attach),
    # recovery refuses to guess: "absent here" proves nothing.
    handle = supervisor2.create(_spec(tmp_path, (sys.executable, "-c", "pass")))
    no_daemon = Attempt(**{**handle.spec.attempt.__dict__, "container_id": "",
                           "container_name": handle.container_name, "execution_kind": "container"})
    assert "no persisted Docker daemon identity" in supervisor2.cleanup_recovered(
        no_daemon, state_dir=tmp_path / "state",
    )
    # ... but the creating engine may name its own daemon explicitly.
    assert supervisor2.cleanup_recovered(
        no_daemon, state_dir=tmp_path / "state", daemon=daemon2,
    ) == ""
    assert fake2.records() == []

    # A deterministic-name collision with a different attempt label is
    # refused, not deleted.
    docker3, fake3 = install_fake_docker(tmp_path / "docker3", image_digest=IMAGE_DIGEST)
    supervisor3 = DockerAttemptSupervisor(_config(tmp_path, docker3))
    handle = supervisor3.create(_spec(tmp_path, (sys.executable, "-c", "pass")))
    record = fake3.record(handle.container_id)
    assert record is not None
    _, path = fake3._find(handle.container_id)
    assert path is not None
    record["Config"]["Labels"]["tcenv.attempt"] = "attempt-other"
    path.write_text(json.dumps(record), encoding="utf-8")
    unnamed = Attempt(**{**handle.spec.attempt.__dict__, "container_id": "",
                         "container_name": handle.container_name, "execution_kind": "container",
                         "container_daemon": supervisor3.daemon_identity()})
    assert "label does not match" in supervisor3.cleanup_recovered(unnamed, state_dir=tmp_path / "state")
    assert len(fake3.records()) == 1
    supervisor3.delete(handle)
    supervisor3.release(handle)


def test_cleanup_recovered_is_fail_closed_when_delete_fails(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    daemon = supervisor.daemon_identity()
    handle = supervisor.create(_spec(tmp_path, (sys.executable, "-c", "pass")))
    attempt = Attempt(**{**handle.spec.attempt.__dict__, "container_id": handle.container_id,
                        "container_name": handle.container_name, "execution_kind": "container",
                        "container_daemon": daemon})
    fake.set_control(fail_rm=True)
    error = supervisor.cleanup_recovered(attempt, state_dir=tmp_path / "state")
    # The container never started, so stopping it is removal; either way the
    # injected ``rm`` failure must surface instead of being reported as clean.
    assert "could not stop" in error or "could not delete" in error
    assert "injected rm failure" in error
    fake.set_control(fail_rm=False)
    # ``rm`` reports success but leaves the container behind: verification,
    # not the exit code, decides.
    fake.set_control(rm_leaves_running=True)
    handle.started = True
    assert "still present" in supervisor.cleanup_recovered(attempt, state_dir=tmp_path / "state")
    fake.set_control(rm_leaves_running=False)
    assert supervisor.cleanup_recovered(attempt, state_dir=tmp_path / "state") == ""


def test_cleanup_recovered_rechecks_the_daemon_during_the_sequence(tmp_path):
    """A switch after the entry check must stop the kill/delete/absence proof."""

    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    daemon_a = supervisor.daemon_identity()
    handle = supervisor.create(_spec(tmp_path, (sys.executable, "-c", "import time; time.sleep(30)")))
    attempt = Attempt(**{**handle.spec.attempt.__dict__, "container_id": handle.container_id,
                        "container_name": handle.container_name, "execution_kind": "container",
                        "container_daemon": daemon_a})
    state = tmp_path / "state"
    try:
        # Switch right after the inspect: the kill must not run on daemon B.
        original_inspect = supervisor._inspect

        def inspect_then_switch(ref):
            result = original_inspect(ref)
            fake.set_control(daemon_id="fake-daemon-b")
            return result

        supervisor._inspect = inspect_then_switch
        error = supervisor.cleanup_recovered(attempt, state_dir=state)
        assert "another Docker daemon" in error
        records = fake.records()
        assert len(records) == 1, records
        assert records[0]["State"]["Status"] in {"created", "running"}

        # Switch inside the delete: even a successful delete cannot be reported
        # as a verified cleanup for the original daemon.
        supervisor._inspect = original_inspect
        fake.set_control(daemon_id=daemon_a)
        original_delete = supervisor._delete_ref

        def delete_then_switch(ref):
            error = original_delete(ref)
            fake.set_control(daemon_id="fake-daemon-b")
            return error

        supervisor._delete_ref = delete_then_switch
        error = supervisor.cleanup_recovered(attempt, state_dir=state)
        assert "another Docker daemon" in error
        assert fake.records() == []
    finally:
        fake.set_control(daemon_id=daemon_a)
        supervisor.cleanup(handle)
        supervisor.release(handle)


def test_runtime_profile_and_build_identity_are_explicit(tmp_path):
    docker, _fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    assert normalize_image_digest("sha256:" + IMAGE_DIGEST) == IMAGE_DIGEST
    build = container_worker_build("evaluator-1", IMAGE_DIGEST)
    assert IMAGE_DIGEST in build
    with pytest.raises(SupervisorError):
        container_worker_build("bad name", IMAGE_DIGEST)
    with pytest.raises(SupervisorError, match="runtime profile"):
        _config(tmp_path, docker, runtime_profile="runc-noop")
    args = build_create_args(
        _config(tmp_path, docker, runtime="runsc", runtime_profile="gvisor"),
        name="tcenv-attempt-attempt-7-0123456789abcdef", attempt_id="attempt-7-0123456789abcdef",
        request_id="request-container-1", env_file=tmp_path / "env", command=("python3", "-m", "worker"),
    )
    assert args[args.index("--runtime") + 1] == "runsc"
    assert args[-4:] == ["taic-worker-sim:local", "python3", "-m", "worker"]


def test_wait_only_reports_a_deadline_for_a_verified_running_worker(tmp_path, monkeypatch):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    handle = supervisor.create(_spec(tmp_path, (sys.executable, "-c", "import time; time.sleep(30)")))
    supervisor.start(handle)
    _wait_running(supervisor, handle)
    for state in ("unknown", "created", "absent"):
        monkeypatch.setattr(supervisor, "_status_for", lambda ref, s=state: (WorkerStatus(s), ""))
        with pytest.raises(SupervisorError, match="not running"):
            supervisor.wait(handle, 0.2)
    monkeypatch.undo()
    assert supervisor.cleanup(handle) == ""
    supervisor.release(handle)


def test_cli_output_flood_is_bounded_and_the_helper_is_stopped(tmp_path):
    docker, fake = install_fake_docker(
        tmp_path / "docker", image_digest=IMAGE_DIGEST, control={"flood_bytes": 5_000_000},
    )
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    with pytest.raises(SupervisorError, match="unbounded output"):
        supervisor._cli(["info", "--format", "flood"])
    # The helper was killed instead of being drained into memory.
    assert not any("flood" in " ".join(call) and call[-1] == "flood" for call in fake.calls()[1:])


def test_contract_valid_request_ids_stay_in_the_labels(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    config = _config(tmp_path, docker)
    for request_id in (":leading", "a:b.c", "x" * 160):
        args = build_create_args(
            config, name="tcenv-attempt-attempt-7-0123456789abcdef",
            attempt_id="attempt-7-0123456789abcdef", request_id=request_id,
            env_file=tmp_path / "env", command=("python3", "-c", "pass"),
        )
        assert f"tcenv.request={request_id}" in args
    # Request ids outside the name subset but inside the wire contract are
    # still valid labels (the container *name* only uses the attempt id).
    oversized = "y" * 161
    with pytest.raises(SupervisorError):
        build_create_args(
            config, name="tcenv-attempt-attempt-7-0123456789abcdef",
            attempt_id="attempt-7-0123456789abcdef", request_id=oversized,
            env_file=tmp_path / "env", command=("python3", "-c", "pass"),
        )


def test_contract_valid_request_id_executes_end_to_end(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    base = _spec(tmp_path, (sys.executable, "-c", "import sys; sys.stdin.buffer.read(); print('ok')"))
    request_obj = request(":colon.id", source=b"OK")
    spec = AttemptSpec(
        request=request_obj, attempt=base.attempt, state_dir=base.state_dir,
        command=base.command, env=base.env,
    )
    handle = supervisor.create(spec)
    try:
        supervisor.start(handle)
        stdin, stdout, _stderr = supervisor.streams(handle)
        stdin.write(b"request")
        stdin.close()
        assert supervisor.wait(handle, 10.0) == 0
        assert stdout.read() == b"ok\n"
        assert supervisor.cleanup(handle) == ""
    finally:
        supervisor.cleanup(handle)
        supervisor.release(handle)


def test_cgroup_env_file_and_scratch_residue_are_cleanup_errors(tmp_path):
    cgroup_root = tmp_path / "cgroup"
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker, cgroup_root=str(cgroup_root)))
    handle = supervisor.create(_spec(tmp_path, (sys.executable, "-c", "pass")))
    assert supervisor.cleanup(handle) == ""
    # A cgroup that outlived the container record is residue, not cleanup.
    scope = cgroup_root / "system.slice" / f"docker-{handle.container_id}.scope"
    scope.mkdir(parents=True)
    assert "cgroup survived" in supervisor.verify_absent(handle)
    assert "cgroup survived" in supervisor.cleanup_residue(handle)
    scope.rmdir()
    # A per-attempt env file left behind is residue too.
    env_path = tmp_path / "state" / "container-env" / f"{handle.attempt_id}.env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("HOME=/tmp\n", encoding="utf-8")
    assert "environment file" in supervisor.cleanup_residue(handle)
    env_path.unlink()
    # A scratch directory that cannot be removed is a delete failure, not a
    # warning (a read-only directory makes the unlink fail).
    scratch = tmp_path / "scratch"
    (scratch).mkdir()
    (scratch / "keep").write_text("x", encoding="utf-8")
    scratch.chmod(0o500)
    handle.scratch_dir = scratch
    try:
        assert "scratch directory" in supervisor.delete(handle)
    finally:
        scratch.chmod(0o700)
    handle.scratch_dir = None
    supervisor.release(handle)


def test_create_fails_closed_when_the_env_file_cannot_be_removed(tmp_path, monkeypatch):
    from tcenv.remote import container_supervisor as module

    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    original_write = module._write_env_file

    def write_then_seal(path, env):
        original_write(path, env)
        path.parent.chmod(0o500)

    monkeypatch.setattr(module, "_write_env_file", write_then_seal)
    try:
        with pytest.raises(SupervisorError, match="env file"):
            supervisor.create(_spec(tmp_path, (sys.executable, "-c", "pass")))
    finally:
        (tmp_path / "state" / "container-env").chmod(0o700)
        monkeypatch.undo()
    # The container the daemon created is cleaned through its durable identity.
    attempt = _attempt()
    from dataclasses import replace as _replace

    assert supervisor.cleanup_recovered(
        _replace(attempt, execution_kind="container",
                 container_daemon=supervisor.daemon_identity()),
        state_dir=tmp_path / "state",
    ) == ""
    assert fake.records() == []


def test_worker_environment_is_isolated_from_the_host(tmp_path, monkeypatch):
    monkeypatch.setenv("HOST_ONLY_MARKER", "must-not-leak")
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    probe = (
        "import json,os,sys;sys.stdin.buffer.read();"
        "print(json.dumps(dict(os.environ)))"
    )
    handle = supervisor.create(_spec(tmp_path, (sys.executable, "-c", probe)))
    try:
        supervisor.start(handle)
        stdin, stdout, _stderr = supervisor.streams(handle)
        stdin.write(b"request")
        stdin.close()
        assert supervisor.wait(handle, 10.0) == 0
        env = json.loads(stdout.read())
        assert "HOST_ONLY_MARKER" not in env
        assert "PWD" not in env and "SHLVL" not in env
        assert env["HOME"] == "/tmp"
        assert env["TCENV_REMOTE_STATE_DIR"] == "/scratch/state"
        assert supervisor.cleanup(handle) == ""
    finally:
        supervisor.cleanup(handle)
        supervisor.release(handle)


def test_cleanup_recovered_removes_a_create_time_env_file(tmp_path):
    """A SIGKILL during create leaves the per-attempt env file behind."""

    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    daemon = supervisor.daemon_identity()
    attempt = Attempt(**{**_attempt().__dict__, "execution_kind": "container",
                        "container_daemon": daemon})
    env_path = tmp_path / "state" / "container-env" / f"{attempt.attempt_id}.env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("HOME=/tmp\n", encoding="utf-8")
    assert supervisor.cleanup_recovered(
        attempt, state_dir=tmp_path / "state",
    ) == ""
    assert not env_path.exists()

    # A file that cannot be removed is a cleanup failure, not a warning.
    env_path.write_text("HOME=/tmp\n", encoding="utf-8")
    env_path.parent.chmod(0o500)
    try:
        error = supervisor.cleanup_recovered(attempt, state_dir=tmp_path / "state")
        assert "env file" in error
    finally:
        env_path.parent.chmod(0o700)
        env_path.unlink()


def test_cleanup_residue_detects_a_surviving_attach_client(tmp_path):
    docker, fake = install_fake_docker(
        tmp_path / "docker", image_digest=IMAGE_DIGEST, control={"hang_attach": True},
    )
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    handle = supervisor.create(_spec(tmp_path, (sys.executable, "-c", "pass")))
    supervisor.start(handle)
    # The fake attach client stays alive after the container is gone.
    supervisor._stop_attach = lambda target: None
    assert supervisor.cleanup(handle) == ""
    assert "attach client survived" in supervisor.cleanup_residue(handle)
    handle.attach and handle.attach.kill()
    if handle.attach is not None:
        handle.attach.wait(timeout=10)
    supervisor.release(handle)


def test_a_process_scan_failure_is_never_an_empty_residue_list(tmp_path, monkeypatch):
    """Unknown is not "no attach client": enumeration failures fail closed."""

    from tcenv.remote import hostprocs

    docker, _fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))

    def exploding_scan():
        raise hostprocs.HostProcsError("could not enumerate host processes: injected")

    monkeypatch.setattr(hostprocs, "iter_processes", exploding_scan)
    with pytest.raises(hostprocs.HostProcsError):
        hostprocs.pids_mentioning("c" * 64, "--attach")
    survivors, error = supervisor._attach_survivors("c" * 64)
    assert survivors == []
    assert "could not inspect host processes" in error
    # The residue/recovery proofs propagate that error instead of treating the
    # scan as clean.
    assert supervisor._reap_attach_survivors("c" * 64) == error

    # The Linux enumeration itself fails closed on an unreadable /proc and on
    # an over-long process table.
    def denied(path):
        raise PermissionError("injected /proc denial")

    monkeypatch.setattr(hostprocs.os, "scandir", denied)
    with pytest.raises(hostprocs.HostProcsError, match="could not enumerate"):
        list(hostprocs._linux_processes())


def test_a_live_darwin_process_with_a_huge_environment_is_still_scanned():
    """A 20 KB environment must not turn a live attach process into "not found"."""

    import os
    import subprocess

    from tcenv.remote import hostprocs

    if sys.platform != "darwin":
        pytest.skip("KERN_PROCARGS2 only exists on Darwin")
    token = "taic-hostprocs-big-env-token"
    env = dict(os.environ, HUGE="x" * 20_000, TOKEN=token)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)", "--attach", token],
        env=env,
    )
    try:
        argv = hostprocs._darwin_argv(process.pid)
        assert argv[-2:] == ("--attach", token)
        assert process.pid in hostprocs.pids_mentioning(token, "--attach")
    finally:
        process.kill()
        process.wait(timeout=10)


def test_the_darwin_sysctl_prototype_uses_size_t(monkeypatch):
    import ctypes

    from tcenv.remote import hostprocs

    if sys.platform != "darwin":
        pytest.skip("KERN_PROCARGS2 only exists on Darwin")
    hostprocs._LIBC = None
    libc, _libproc = hostprocs._darwin_libraries()
    assert libc.sysctl.argtypes[3] == ctypes.POINTER(ctypes.c_size_t)
    assert libc.sysctl.restype is ctypes.c_int


class _FakeSysctl:
    """Minimal KERN_PROCARGS2 replacement for malformed/over-limit records."""

    def __init__(self, payload: bytes, *, full_size: int | None = None) -> None:
        self.payload = payload
        self.full_size = len(payload) if full_size is None else full_size

    def sysctl(self, mib, namelen, oldp, oldlenp, newp, newlen):
        import ctypes

        size = oldlenp._obj
        if oldp is None:
            size.value = self.full_size
            return 0
        limit = min(size.value, len(self.payload))
        ctypes.memmove(oldp, self.payload, limit)
        size.value = limit
        return 0


def _record(payload: bytes) -> bytes:
    return len(payload).to_bytes(4, "little") + payload


@pytest.mark.parametrize(
    ("payload", "full_size", "message"),
    [
        (b"", 4, "malformed argument record"),
        (b"\x00\x00\x00\x00execpath\x00", None, "malformed argument count"),
        (_record(b"/bin/x\x00a\x00b\x00"), None, "exceeds the bounded read limit"),
    ],
)
def test_malformed_darwin_records_fail_closed(monkeypatch, payload, full_size, message):
    from tcenv.remote import hostprocs

    class _LibProc:
        @staticmethod
        def proc_listpids(kind, subtype, buffer, size):
            return 0

    monkeypatch.setattr(
        hostprocs, "_darwin_libraries",
        lambda: (_FakeSysctl(payload, full_size=full_size), _LibProc()),
    )
    with pytest.raises(hostprocs.HostProcsError, match=message):
        hostprocs._darwin_argv(4242)


def test_darwin_process_enumeration_failure_raises_instead_of_returning_empty(monkeypatch):
    from tcenv.remote import hostprocs

    class _LibProc:
        @staticmethod
        def proc_listpids(kind, subtype, buffer, size):
            return 0

    monkeypatch.setattr(hostprocs, "_darwin_libraries", lambda: (None, _LibProc()))
    with pytest.raises(hostprocs.HostProcsError, match="could not enumerate"):
        hostprocs._darwin_pids()


def test_every_lifecycle_step_is_bound_to_the_creating_daemon(tmp_path):
    """A / B switch: the container on A must never be "absent" on B."""

    docker_a, fake_a = install_fake_docker(tmp_path / "docker-a", image_digest=IMAGE_DIGEST)
    supervisor_a = DockerAttemptSupervisor(_config(tmp_path, docker_a))
    handle = supervisor_a.create(_spec(tmp_path, (sys.executable, "-c", "import time; time.sleep(30)")))
    daemon_a = supervisor_a.daemon_identity()
    assert handle.container_daemon == daemon_a
    # The same physical daemon now answers with another identity: "not found"
    # here proves nothing, because the container is still alive over there.
    fake_a.set_control(daemon_id="fake-daemon-b")
    for name, call in (
        ("verify_identity", lambda: supervisor_a.verify_identity(handle)),
        ("kill", lambda: supervisor_a.kill(handle)),
        ("delete", lambda: supervisor_a.delete(handle)),
        ("verify_absent", lambda: supervisor_a.verify_absent(handle)),
        ("cleanup", lambda: supervisor_a.cleanup(handle)),
        ("cleanup_residue", lambda: supervisor_a.cleanup_residue(handle)),
    ):
        result = call()
        assert "another Docker daemon" in result, (name, result)
    with pytest.raises(SupervisorError, match="another Docker daemon"):
        supervisor_a.start(handle)
    with pytest.raises(SupervisorError, match="another Docker daemon"):
        supervisor_a.status(handle)
    # Nothing was started or deleted through the wrong identity.
    assert len(fake_a.records()) == 1
    assert fake_a.record(handle.container_id)["State"]["Status"] == "created"
    fake_a.set_control(daemon_id=daemon_a)
    assert supervisor_a.cleanup(handle) == ""
    supervisor_a.release(handle)
