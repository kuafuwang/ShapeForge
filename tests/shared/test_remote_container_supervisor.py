"""Unit tests for the disposable-container execution plane (fake Docker CLI)."""

from __future__ import annotations

import json
import sys
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
from tcenv.remote.supervisor import AttemptSpec, SupervisorError, WorkerHandle
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


def test_kill_and_delete_are_idempotent_and_verify(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    handle = supervisor.create(_spec(tmp_path, (sys.executable, "-c", "import time; time.sleep(30)")))
    supervisor.start(handle)
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
    error = survivor_supervisor.kill(handle)
    assert "did not stop" in error
    assert survivor_supervisor.cleanup(handle) != ""
    survivor_supervisor.delete(handle)
    survivor_supervisor.release(handle)


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


def test_cleanup_recovered_uses_persisted_identity_and_refuses_mismatch(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    handle = supervisor.create(_spec(tmp_path, (sys.executable, "-c", "pass")))
    attempt = Attempt(**{**handle.spec.attempt.__dict__, "container_id": handle.container_id,
                        "container_name": handle.container_name, "execution_kind": "container"})
    assert supervisor.cleanup_recovered(attempt) == ""
    assert fake.records() == []

    # A crash between create and the durable id attach leaves no container id.
    docker2, fake2 = install_fake_docker(tmp_path / "docker2", image_digest=IMAGE_DIGEST)
    supervisor2 = DockerAttemptSupervisor(_config(tmp_path, docker2))
    handle = supervisor2.create(_spec(tmp_path, (sys.executable, "-c", "pass")))
    named = Attempt(**{**handle.spec.attempt.__dict__, "container_id": "",
                       "container_name": handle.container_name, "execution_kind": "container"})
    assert supervisor2.cleanup_recovered(named) == ""
    assert fake2.records() == []

    # A process-supervisor attempt must never be guessed at by the container path.
    process_attempt = Attempt(**{**handle.spec.attempt.__dict__, "execution_kind": "process"})
    assert "not started by the container supervisor" in supervisor2.cleanup_recovered(process_attempt)

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
                         "container_name": handle.container_name, "execution_kind": "container"})
    assert "label does not match" in supervisor3.cleanup_recovered(unnamed)
    assert len(fake3.records()) == 1
    supervisor3.delete(handle)
    supervisor3.release(handle)


def test_cleanup_recovered_is_fail_closed_when_delete_fails(tmp_path):
    docker, fake = install_fake_docker(tmp_path / "docker", image_digest=IMAGE_DIGEST)
    supervisor = DockerAttemptSupervisor(_config(tmp_path, docker))
    handle = supervisor.create(_spec(tmp_path, (sys.executable, "-c", "pass")))
    attempt = Attempt(**{**handle.spec.attempt.__dict__, "container_id": handle.container_id,
                        "container_name": handle.container_name, "execution_kind": "container"})
    fake.set_control(fail_rm=True)
    assert "could not delete" in supervisor.cleanup_recovered(attempt)
    fake.set_control(fail_rm=False)
    # ``rm`` reports success but leaves the container behind: verification,
    # not the exit code, decides.
    fake.set_control(rm_leaves_running=True)
    handle.started = True
    assert "still present" in supervisor.cleanup_recovered(attempt)
    fake.set_control(rm_leaves_running=False)
    assert supervisor.cleanup_recovered(attempt) == ""


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
