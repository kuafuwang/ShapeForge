"""Real Docker CPU-only supervisor tests (opt-in, no GPU/NPU).

Set ``TAIC_DOCKER_CONTAINER_TESTS=1`` and optionally
``TAIC_CONTAINER_TEST_IMAGE=<image>`` (default ``taic-remote-sim:local``) to run
these against a reachable Docker Engine.  They exercise the production
supervisor end to end on a real container: create/start/wait/kill/delete, the
read-only root, dropped capabilities, no Docker socket, no network and the
per-attempt scratch tmpfs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tcenv.remote.container_supervisor import (
    ContainerSupervisorConfig,
    DockerAttemptSupervisor,
)
from tcenv.remote.supervisor import AttemptSpec
from tests.support.job_engine import request

ENABLED = os.environ.get("TAIC_DOCKER_CONTAINER_TESTS") == "1"
IMAGE = os.environ.get("TAIC_CONTAINER_TEST_IMAGE", "taic-remote-sim:local")

pytestmark = pytest.mark.skipif(
    not ENABLED, reason="set TAIC_DOCKER_CONTAINER_TESTS=1 to run real container tests",
)


def _image_digest(image: str = IMAGE) -> str:
    completed = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"Docker image {image!r} is unavailable: {completed.stderr.strip()}")
    return completed.stdout.strip().removeprefix("sha256:")


def _spec(tmp_path: Path, command: tuple[str, ...], *, env: dict[str, str] | None = None) -> AttemptSpec:
    from tcenv.remote.job_types import Attempt

    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    attempt = Attempt(
        attempt_id="attempt-1-aaaaaaaaaaaaaaaa", lease_fence=1, state="running",
        retry_index=0, cancel_requested=False, queued_at_ms=0, started_at_ms=1,
        execution_kind="container",
    )
    return AttemptSpec(
        request=request("request-real-container", source=b"OK"), attempt=attempt,
        state_dir=state, command=command, env=env or {},
    )


def _supervisor(tmp_path: Path, digest: str) -> DockerAttemptSupervisor:
    return DockerAttemptSupervisor(ContainerSupervisorConfig(
        image=IMAGE, image_digest=digest, env={"HOME": "/tmp"}, kill_timeout_s=10.0,
    ))


def test_real_container_runs_with_hardened_profile_and_is_removed(tmp_path):
    digest = _image_digest()
    supervisor = _supervisor(tmp_path, digest)
    probe = (
        "import json,os,socket,sys;"
        "result={'uid':os.getuid(),'readonly':False,'docker_socket':os.path.exists('/var/run/docker.sock'),"
        "'home':os.environ.get('HOME'),'network':False};"
        "open('/scratch/probe','w').write('x');"
        "result['scratch']=os.path.exists('/scratch/probe');"
        "\ntry:\n open('/readonly-probe','w').write('x')\nexcept OSError:\n result['readonly']=True\n"
        "try:\n socket.create_connection(('1.1.1.1',80),timeout=1).close()\nexcept OSError:\n result['network']=True\n"
        "sys.stdin.buffer.read();"
        "sys.stdout.write(json.dumps(result))"
    )
    handle = supervisor.create(_spec(tmp_path, ("python3", "-c", probe)))
    try:
        assert supervisor.verify_identity(handle) == ""
        supervisor.start(handle)
        stdin, stdout, stderr = supervisor.streams(handle)
        assert stdin is not None and stdout is not None and stderr is not None
        stdin.write(b"request")
        stdin.flush()
        stdin.close()
        assert supervisor.wait(handle, 30.0) == 0
        observed = json.loads(stdout.read())
        assert observed["uid"] == 65534
        assert observed["readonly"] is True
        assert observed["docker_socket"] is False
        assert observed["scratch"] is True
        assert observed["network"] is True
        assert observed["home"] == "/tmp"
        assert supervisor.cleanup(handle) == ""
        assert supervisor.verify_absent(handle) == ""
        assert supervisor.cleanup_residue(handle) == ""
    finally:
        supervisor.cleanup(handle)
        supervisor.release(handle)


def test_real_container_timeout_kills_parent_and_child(tmp_path):
    digest = _image_digest()
    supervisor = _supervisor(tmp_path, digest)
    token = f"taic-container-child-{os.getpid()}-{int(time.time())}"
    worker = (
        "import subprocess,sys,time;"
        f"child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)','{token}']);"
        "time.sleep(120)"
    )
    handle = supervisor.create(_spec(tmp_path, ("python3", "-c", worker)))
    try:
        supervisor.start(handle)
        assert supervisor.wait(handle, 1.0) is None
        assert supervisor.status(handle).state == "running"
        top = subprocess.run(
            ["docker", "top", handle.container_id, "-o", "pid,args"],
            text=True, capture_output=True, check=False,
        )
        assert token in top.stdout, "child process was not visible inside the container"
        assert supervisor.cleanup(handle) == ""
        assert supervisor.verify_absent(handle) == ""
        # The parent and its child shared the container/pause PID namespace, so
        # after the container record is gone no process of that namespace can
        # remain.  A host-wide process scan is part of the VM acceptance, where
        # the test process can read the host process table.
        remaining = subprocess.run(
            ["docker", "top", handle.container_id], text=True, capture_output=True, check=False,
        )
        assert remaining.returncode != 0
        assert "No such" in (remaining.stderr + remaining.stdout)
    finally:
        supervisor.cleanup(handle)
        supervisor.release(handle)


def test_real_container_identity_check_rejects_a_retagged_image(tmp_path):
    _image_digest()
    from tcenv.remote.supervisor import SupervisorError

    wrong = DockerAttemptSupervisor(ContainerSupervisorConfig(
        image=IMAGE, image_digest="f" * 64, env={"HOME": "/tmp"},
    ))
    with pytest.raises(SupervisorError, match="pinned digest"):
        wrong.create(_spec(tmp_path, ("python3", "-c", "pass")))
    assert sys.version_info >= (3, 12)
