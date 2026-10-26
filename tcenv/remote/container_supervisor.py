"""Disposable-container execution plane for one external evaluator attempt.

The production execution plane runs exactly one container per attempt and lets
the Docker Engine (containerd/runc, or a configured alternative OCI runtime)
own the process/cgroup lifecycle.  This module is deliberately narrow: it
creates, starts, streams, waits, kills, deletes and *verifies* that one
container is gone.  It does not queue, lease, retry, sign or publish anything;
that remains the job engine's business.

Guarantees encoded here:

* the image is pinned by its ``sha256`` image ID before any container exists;
* no host bind mount, Docker socket, control-host home/repo or credential env
  is ever passed to the worker;
* the root filesystem is read-only, capabilities are dropped, no-new-privileges
  is set, networking defaults to ``none``, and pids/memory/CPU are bounded;
* ``create`` and the durable ``container_id`` attach happen before ``start``;
* kill/delete are idempotent and every stop path verifies the container and any
  label-matched duplicate are really absent (fail closed otherwise).

Plain Docker is a cgroup/namespace lifecycle boundary, not a complete
untrusted-code security boundary.  ``runtime``/``runtime_profile`` allow a
gVisor (``runsc``) or Kata profile to be selected later; neither is implemented
or validated by this work.
"""

from __future__ import annotations

import json
import os
import re
import select
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from tcenv.children import _GROUP_LEADERS, forget, kill_group, register, scrubbed_env

from . import hostprocs
from .job_types import Attempt
from .supervisor import (
    EXECUTION_CONTAINER,
    AttemptSpec,
    DaemonChanged,
    StartRefused,
    SupervisorError,
    WorkerHandle,
    WorkerStatus,
)
from .worker_protocol import STATE_DIR_ENV

_IMAGE_DIGEST = re.compile(r"(?:sha256:)?([0-9a-f]{64})")
_CONTAINER_ID = re.compile(r"[0-9a-f]{12,64}")
# Container names must be a safe subset; attempt ids are engine-generated.
_ATTEMPT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}")
# Labels carry contract-valid ids unchanged: the request id is client-chosen and
# may contain ':', lead with punctuation, or be up to 160 characters.
_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,160}")
_DAEMON_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SECRET_ENV = re.compile(
    r"(^|_)(API_?KEY|APIKEY|KEYS?|TOKENS?|SECRETS?|PASSWORDS?|PASSWD|CREDENTIALS?)($|_)",
    re.I,
)
# Host-side Docker control variables and the SSH agent socket must never reach
# the container.  They are only used for the frontend's own Docker CLI calls.
_FORBIDDEN_WORKER_ENV = frozenset({
    "DOCKER_HOST", "DOCKER_CONFIG", "DOCKER_CERT_PATH", "DOCKER_TLS_VERIFY",
    "SSH_AUTH_SOCK", "SSH_AGENT_PID",
})
_CLI_ENV_KEEP = ("PATH", "HOME", "SSH_AUTH_SOCK", "DOCKER_HOST", "DOCKER_CONFIG", "LANG", "LC_ALL")
_MAX_CLI_OUTPUT = 1_000_000
_MAX_CLI_CHUNK = 65_536
_MAX_ERROR_CHARS = 500
_MAX_ENV_FILE_BYTES = 64 * 1024
_KILL_POLL_S = 0.05
_SCRATCH_LABEL_PREFIX = "tcenv.attempt"
_OWNER_LABEL = "tcenv.owner"
_RUNTIME_PROFILES = frozenset({"runc", "gvisor", "kata"})


def _sanitize(raw: str) -> str:
    text = " ".join(str(raw).replace("\x00", " ").split())
    return text[:_MAX_ERROR_CHARS]


def _terminate(process: subprocess.Popen) -> None:
    """Stop a helper CLI process whose output could not be bounded."""

    try:
        process.kill()
    except OSError:
        return
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        return


def normalize_image_digest(value: str) -> str:
    """Accept ``sha256:<hex>`` or bare hex and return the bare 64-char digest."""

    if not isinstance(value, str):
        raise SupervisorError("worker image digest must be text")
    match = _IMAGE_DIGEST.fullmatch(value.strip())
    if match is None:
        raise SupervisorError("worker image digest must be a sha256 hex digest")
    return match.group(1)


def container_worker_build(build_name: str, image_digest: str) -> str:
    """The signed ``worker_build`` identity for a container-executed worker.

    The digest is embedded in the existing provenance field so the external
    evidence contract does not need a schema change: the signed result records
    the exact image that ran the attempt.
    """

    digest = normalize_image_digest(image_digest)
    if not isinstance(build_name, str) or not re.fullmatch(r"[A-Za-z0-9._:+-]{1,120}", build_name):
        raise SupervisorError("worker build name is invalid")
    return f"{build_name}+img:{digest}"


def container_name_for(attempt_id: str) -> str:
    if not isinstance(attempt_id, str) or _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise SupervisorError("attempt id cannot be used as a container name")
    return f"tcenv-attempt-{attempt_id}"


def _validate_env(env: Mapping[str, str]) -> dict[str, str]:
    """Validate the explicit worker environment; fail closed on secrets."""

    normalized: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or _ENV_NAME.fullmatch(key) is None:
            raise SupervisorError("worker environment variable name is invalid")
        if not isinstance(value, str):
            raise SupervisorError(f"worker environment variable {key} must be text")
        if "\x00" in value or "\n" in value or "\r" in value:
            raise SupervisorError(f"worker environment variable {key} spans multiple lines")
        if key in _FORBIDDEN_WORKER_ENV:
            raise SupervisorError(f"worker environment must not carry {key}")
        if _SECRET_ENV.search(key):
            raise SupervisorError(f"worker environment must not carry credential {key}")
        normalized[key] = value
    return normalized


@dataclass(frozen=True)
class ContainerSupervisorConfig:
    """Production execution-plane configuration for one evaluator frontend."""

    image: str
    image_digest: str
    docker: str = "docker"
    docker_host: str = ""
    runtime: str = ""
    runtime_profile: str = "runc"
    user: str = "65534:65534"
    workdir: str = "/opt/taic"
    pids_limit: int = 256
    memory: str = "2g"
    cpus: str = "2"
    scratch_path: str = "/scratch"
    scratch_size: str = "512m"
    scratch_exec: bool = True
    tmp_size: str = "64m"
    env: Mapping[str, str] = field(default_factory=lambda: {
        "HOME": "/tmp",
        "PYTHONPATH": "/opt/taic",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    })
    cli_timeout_s: float = 30.0
    kill_timeout_s: float = 15.0
    cgroup_root: str = "/sys/fs/cgroup"

    def __post_init__(self) -> None:
        if not isinstance(self.image, str) or not self.image or "\x00" in self.image or self.image.startswith("-"):
            raise SupervisorError("worker image reference is invalid")
        object.__setattr__(self, "image_digest", normalize_image_digest(self.image_digest))
        if not isinstance(self.docker, str) or not self.docker or "\x00" in self.docker:
            raise SupervisorError("docker binary is invalid")
        if self.docker_host and (not isinstance(self.docker_host, str) or "\x00" in self.docker_host):
            raise SupervisorError("docker host is invalid")
        if self.runtime and not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", self.runtime):
            raise SupervisorError("container runtime name is invalid")
        if self.runtime_profile not in _RUNTIME_PROFILES:
            raise SupervisorError(f"runtime profile must be one of {sorted(_RUNTIME_PROFILES)}")
        if not re.fullmatch(r"\d+:\d+", self.user):
            raise SupervisorError("worker user must be numeric uid:gid")
        for value, name in ((self.workdir, "workdir"), (self.scratch_path, "scratch path")):
            if not isinstance(value, str) or not value.startswith("/") or ".." in value.split("/"):
                raise SupervisorError(f"{name} must be an absolute container path")
        if isinstance(self.pids_limit, bool) or not isinstance(self.pids_limit, int) or not 16 <= self.pids_limit <= 65_536:
            raise SupervisorError("pids limit must be an integer between 16 and 65536")
        for value, name in ((self.memory, "memory"), (self.cpus, "cpus"), (self.scratch_size, "scratch size"), (self.tmp_size, "tmp size")):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?[bkmgBKMG]?", value):
                raise SupervisorError(f"{name} is not a bounded Docker value")
        if isinstance(self.cli_timeout_s, bool) or not 1 <= float(self.cli_timeout_s) <= 300:
            raise SupervisorError("docker CLI timeout is out of range")
        if isinstance(self.kill_timeout_s, bool) or not 1 <= float(self.kill_timeout_s) <= 300:
            raise SupervisorError("container kill timeout is out of range")
        if not isinstance(self.cgroup_root, str) or not self.cgroup_root.startswith("/") or "\x00" in self.cgroup_root:
            raise SupervisorError("cgroup root must be an absolute host path")
        object.__setattr__(self, "env", _validate_env(dict(self.env)))


def build_create_args(
    config: ContainerSupervisorConfig, *,
    name: str,
    attempt_id: str, request_id: str, env_file: Path, command: tuple[str, ...],
) -> list[str]:
    """Build the exact ``docker create`` arguments (no shell, no interpolation)."""

    if not _ATTEMPT_ID.fullmatch(attempt_id):
        raise SupervisorError("attempt id cannot be used in container labels")
    if _REQUEST_ID.fullmatch(request_id) is None:
        raise SupervisorError("request id cannot be used in container labels")
    argv = [
        "create",
        "--name", name,
        "--label", f"{_SCRATCH_LABEL_PREFIX}={attempt_id}",
        "--label", f"{_OWNER_LABEL}=external-evaluator",
        "--label", f"tcenv.request={request_id}",
        "--interactive",
        "--init",
        "--read-only",
        "--network", "none",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", str(config.pids_limit),
        "--memory", config.memory,
        "--cpus", config.cpus,
        "--user", config.user,
        "--workdir", config.workdir,
        "--tmpfs", _tmpfs_spec(config.scratch_path, config.scratch_size, exec_allowed=config.scratch_exec),
        "--tmpfs", _tmpfs_spec("/tmp", config.tmp_size, exec_allowed=False),
        # Docker reads this file locally and sends only the key/values; the
        # container never sees a host path.  Keeping values off argv avoids
        # exposing them to other local processes through ``ps``.
        "--env-file", str(env_file),
    ]
    if config.runtime:
        argv += ["--runtime", config.runtime]
    argv += [config.image, *command]
    return argv


def build_create_argv(
    config: ContainerSupervisorConfig, *,
    name: str,
    attempt_id: str, request_id: str, env_file: Path, command: tuple[str, ...],
) -> list[str]:
    """Complete Docker CLI command line for one attempt (binary + host + args)."""

    argv = [config.docker]
    if config.docker_host:
        argv += ["--host", config.docker_host]
    return argv + build_create_args(
        config, name=name, attempt_id=attempt_id, request_id=request_id,
        env_file=env_file, command=command,
    )


def _tmpfs_spec(path: str, size: str, *, exec_allowed: bool) -> str:
    flags = "rw,nosuid,nodev" if exec_allowed else "rw,noexec,nosuid,nodev"
    return f"{path}:{flags},size={size},mode=1777"


def _write_env_file(path: Path, env: Mapping[str, str]) -> None:
    lines = [f"{key}={value}" for key, value in sorted(env.items())]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    if len(payload) > _MAX_ENV_FILE_BYTES:
        raise SupervisorError("worker environment file is too large")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class DockerAttemptSupervisor:
    """One disposable Docker/containerd container per evaluator attempt."""

    kind = EXECUTION_CONTAINER

    def __init__(self, config: ContainerSupervisorConfig) -> None:
        if not isinstance(config, ContainerSupervisorConfig):
            raise SupervisorError("container supervisor requires a validated configuration")
        self.config = config
        # Test seam: called before the gate's final check, so a cancellation or
        # revocation can be interleaved deterministically.
        self._before_spawn = None

    # -- configuration surface used by composition roots --------------------
    @property
    def image_digest(self) -> str:
        return self.config.image_digest

    @property
    def runtime_profile(self) -> str:
        return self.config.runtime_profile

    def expected_worker_build(self, build_name: str) -> str:
        return container_worker_build(build_name, self.config.image_digest)

    # -- docker CLI plumbing ----------------------------------------------
    def _cli_env(self) -> dict[str, str]:
        base = scrubbed_env()
        return {key: base[key] for key in _CLI_ENV_KEEP if key in base}

    def _cli(self, args: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess:
        """Run one Docker CLI call with a streaming, bounded output capture.

        ``subprocess.run(capture_output=True)`` would buffer whatever the CLI
        prints before any size check; a flood then costs memory.  The pipes are
        drained in fixed-size chunks and the process is killed as soon as the
        limit is exceeded.
        """

        argv = [self.config.docker]
        if self.config.docker_host:
            argv += ["--host", self.config.docker_host]
        argv += args
        deadline = time.monotonic() + (self.config.cli_timeout_s if timeout is None else timeout)
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._cli_env(),
            )
        except OSError as exc:
            raise SupervisorError(f"docker {args[0]} failed: {_sanitize(str(exc))}") from exc
        streams = {
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        open_streams = {name: stream for name, stream in streams.items() if stream is not None}
        try:
            while open_streams:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SupervisorError(f"docker {args[0]} failed: timed out")
                ready = select.select(list(open_streams.values()), [], [], min(remaining, 0.2))[0]
                for stream in ready:
                    name = next(key for key, value in open_streams.items() if value is stream)
                    chunk = os.read(stream.fileno(), _MAX_CLI_CHUNK)
                    if not chunk:
                        open_streams.pop(name)
                        continue
                    buffers[name].extend(chunk)
                    if len(buffers[name]) > _MAX_CLI_OUTPUT:
                        raise SupervisorError(f"docker {args[0]} produced unbounded output")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SupervisorError(f"docker {args[0]} failed: timed out")
            try:
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                raise SupervisorError(f"docker {args[0]} failed: timed out") from None
        except BaseException:
            _terminate(process)
            raise
        finally:
            for stream in streams.values():
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        return subprocess.CompletedProcess(
            argv, returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"]),
        )

    @staticmethod
    def _missing(completed: subprocess.CompletedProcess) -> bool:
        text = (completed.stdout + completed.stderr).decode("utf-8", errors="replace").lower()
        return "no such" in text or "not found" in text

    def _inspect(self, ref: str) -> tuple[dict[str, Any] | None, str]:
        """Return ``(container document, error)``; ``(None, "")`` means absent."""

        completed = self._cli(["inspect", "--type", "container", ref])
        if completed.returncode != 0:
            if self._missing(completed):
                return None, ""
            return None, _sanitize(completed.stderr.decode("utf-8", errors="replace") or "inspect failed")
        try:
            document = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return None, f"docker inspect output is not JSON: {exc}"[:_MAX_ERROR_CHARS]
        if not isinstance(document, list) or len(document) != 1 or not isinstance(document[0], dict):
            return None, "docker inspect returned an unexpected document"
        return document[0], ""

    def _status_for(self, ref: str) -> tuple[WorkerStatus, str]:
        document, error = self._inspect(ref)
        if error:
            return WorkerStatus("unknown"), error
        if document is None:
            return WorkerStatus("absent"), ""
        state = document.get("State")
        if not isinstance(state, dict):
            return WorkerStatus("unknown"), "docker inspect has no container state"
        status = state.get("Status")
        pid = state.get("Pid")
        exit_code = state.get("ExitCode")
        state_name = {
            "created": "created", "running": "running", "paused": "running",
            "restarting": "running", "exited": "exited", "dead": "exited",
            "removing": "unknown", "stopped": "exited",
        }.get(status, "unknown")
        return WorkerStatus(
            state_name,
            exit_code if isinstance(exit_code, int) and not isinstance(exit_code, bool) else None,
            pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
        ), ""

    # -- identity ----------------------------------------------------------
    def daemon_identity(self) -> str:
        """Stable identity of the Docker Engine this supervisor talks to.

        Recovery persists this with the attempt: reconciling an attempt against
        another daemon would "prove" absence while the old container is still
        running somewhere else.
        """

        completed = self._cli(["info", "--format", "{{.ID}}"])
        if completed.returncode != 0:
            raise SupervisorError(
                "docker info failed: "
                + _sanitize(completed.stderr.decode("utf-8", errors="replace") or "unknown error")
            )
        daemon = completed.stdout.decode("utf-8", errors="replace").strip()
        if _DAEMON_ID.fullmatch(daemon) is None:
            raise SupervisorError("docker info returned an invalid daemon identity")
        return daemon

    def verify_image(self) -> str:
        completed = self._cli(["image", "inspect", "--format", "{{.Id}}", self.config.image])
        if completed.returncode != 0:
            return _sanitize(completed.stderr.decode("utf-8", errors="replace") or "image inspect failed")
        image_id = completed.stdout.decode("utf-8", errors="replace").strip()
        if image_id != f"sha256:{self.config.image_digest}":
            return f"worker image {self.config.image!r} is not the pinned digest"
        return ""

    # -- lifecycle ---------------------------------------------------------
    def create(self, spec: AttemptSpec) -> WorkerHandle:
        # The engine persisted the daemon identity before create.  A switch
        # between that durable attach and this call would create the container
        # on another Engine while every later proof is bound to the persisted
        # one; verify it first, before any container exists.
        persisted_daemon = getattr(spec.attempt, "container_daemon", "") or ""
        if persisted_daemon:
            error = self._guard_daemon(
                persisted_daemon, "could not create the worker container",
            )
            if error:
                raise SupervisorError(error)
        digest_error = self.verify_image()
        if digest_error:
            raise SupervisorError(f"worker image identity check failed: {digest_error}")
        name = container_name_for(spec.attempt.attempt_id)
        env = dict(self.config.env)
        env.update(_validate_env(dict(spec.env)))
        env[STATE_DIR_ENV] = f"{self.config.scratch_path}/state"
        env_file = spec.state_dir / "container-env" / f"{spec.attempt.attempt_id}.env"
        create_error: str = ""
        try:
            env_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(env_file.parent, 0o700)
            _write_env_file(env_file, env)
            argv = build_create_args(
                self.config, name=name, attempt_id=spec.attempt.attempt_id,
                request_id=spec.request.request_id, env_file=env_file, command=spec.command,
            )
            completed = self._cli(argv, timeout=max(self.config.cli_timeout_s, 60.0))
        except OSError as exc:
            create_error = f"could not prepare worker container: {_sanitize(str(exc))}"
            completed = None
        finally:
            # The per-attempt env file holds the worker environment; leaving it
            # behind is a cleanup failure, not a warning.
            try:
                env_file.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                unlink_error = _sanitize(str(exc))
                create_error = (create_error + "; " if create_error else "") + (
                    f"could not remove the per-attempt env file: {unlink_error}"
                )
        if create_error:
            raise SupervisorError(create_error)
        assert completed is not None
        if completed.returncode != 0:
            raise SupervisorError(
                "docker create failed: "
                + _sanitize(completed.stderr.decode("utf-8", errors="replace") or "unknown error")
            )
        container_id = completed.stdout.decode("utf-8", errors="replace").strip()
        if _CONTAINER_ID.fullmatch(container_id) is None:
            raise SupervisorError("docker create returned an unexpected container id")
        # Bind the handle to the daemon the engine already persisted.  A daemon
        # switch between the durable attach and ``create`` must fail closed here
        # rather than let a later "absent" on the new daemon clear quarantine
        # while the container still runs on the old one.
        daemon = getattr(spec.attempt, "container_daemon", "") or ""
        if not daemon:
            daemon = self.daemon_identity()
        handle = WorkerHandle(
            spec=spec, kind=self.kind, container_id=container_id, container_name=name,
            image_ref=self.config.image, image_digest=self.config.image_digest,
            runtime=self.config.runtime or self.config.runtime_profile,
            container_daemon=daemon,
        )
        return handle

    def _bind_daemon(self, handle: WorkerHandle) -> str:
        """Return "" only when this supervisor's engine owns the handle.

        Every lifecycle and absence proof is bound to the daemon the container
        was created on.  Reconciling against another engine cannot observe the
        container, so it must never be mistaken for a verified cleanup.
        """

        bound = getattr(handle, "container_daemon", "") or ""
        if not bound:
            return "worker handle carries no Docker daemon identity; refusing to act on it"
        try:
            current = self.daemon_identity()
        except SupervisorError as exc:
            return f"could not verify the worker's Docker daemon: {exc}"
        if current != bound:
            return (
                "worker container belongs to another Docker daemon "
                f"({bound!r} != {current!r}); refusing to reconcile it here"
            )
        return ""

    def _guard_daemon(self, expected: str, prefix: str) -> str:
        """Re-verify the persisted daemon around one cleanup step.

        A single check at the entry of a multi-step sequence is not enough: the
        endpoint can start answering as another Engine between the inspect, the
        kill, the delete and the absence proof.  Any such switch makes every
        later observation meaningless, so it fails closed and names the step.
        """

        try:
            current = self.daemon_identity()
        except SupervisorError as exc:
            return f"{prefix}: could not verify the Docker daemon: {exc}"
        if current != expected:
            return (
                f"{prefix}: container belongs to another Docker daemon "
                f"({expected!r} != {current!r}); refusing to reconcile it here"
            )
        return ""

    def verify_identity(self, handle: WorkerHandle) -> str:
        error = self._bind_daemon(handle)
        if error:
            return error
        document, error = self._inspect(handle.container_id)
        if error:
            return f"could not verify worker container: {error}"
        if document is None:
            return "worker container disappeared before it was started"
        labels = (document.get("Config") or {}).get("Labels") or {}
        if labels.get(_SCRATCH_LABEL_PREFIX) != handle.attempt_id:
            return "worker container attempt label does not match the leased attempt"
        if document.get("Image") != f"sha256:{self.config.image_digest}":
            return "worker container image id differs from the pinned digest"
        host = document.get("HostConfig") or {}
        container_config = document.get("Config") or {}
        checks = (
            (host.get("NetworkMode") == "none", "network mode is not none"),
            (host.get("ReadonlyRootfs") is True, "root filesystem is not read-only"),
            (list(host.get("CapDrop") or []) == ["ALL"], "capabilities were not dropped"),
            (host.get("Privileged") is not True, "worker container is privileged"),
            (not (host.get("Binds") or []), "worker container has a bind mount"),
            (not (host.get("PidMode") or ""), "worker container shares a host pid namespace"),
            (not str(host.get("IpcMode") or "").startswith("host"), "worker container shares host IPC"),
            (not str(host.get("UsernsMode") or "").startswith("host"), "worker container uses host user namespace"),
            (host.get("PidsLimit") == self.config.pids_limit, "pids limit differs from the profile"),
            (container_config.get("User") == self.config.user, "worker user differs from the profile"),
            (isinstance(host.get("Memory"), int) and host.get("Memory") > 0, "memory limit is not set"),
            (isinstance(host.get("NanoCpus"), int) and host.get("NanoCpus") > 0, "CPU limit is not set"),
            (not self.config.runtime or host.get("Runtime") == self.config.runtime, "container runtime differs from the profile"),
        )
        for ok, message in checks:
            if not ok:
                return message
        security = [str(value) for value in (host.get("SecurityOpt") or [])]
        if not any(value.startswith("no-new-privileges") for value in security):
            return "no-new-privileges is not set"
        mounts = [
            mount for mount in (document.get("Mounts") or [])
            if mount.get("Type") not in {"tmpfs"} or mount.get("Destination") not in {"/tmp", self.config.scratch_path}
        ]
        if mounts:
            return "worker container has an unexpected mount"
        config = document.get("Config") or {}
        for key in config.get("Env") or []:
            name = str(key).split("=", 1)[0]
            if name in _FORBIDDEN_WORKER_ENV or _SECRET_ENV.search(name):
                return f"worker container environment carries {name}"
        return ""

    def start(
        self, handle: WorkerHandle, *, authorize: Callable[[], str] | None = None,
        gate: object | None = None,
    ) -> None:
        if not handle.container_id:
            raise SupervisorError("container start requires a created container id")
        error = self._bind_daemon(handle)
        if error:
            raise SupervisorError(error)
        # Everything above (especially ``docker info``) can block and spend the
        # candidate deadline.  The engine's final start authorization therefore
        # runs here, after the last blocking bind and immediately before the
        # spawn, so a deadline or cancellation that arrived during the bind
        # never results in ``docker start``.
        try:
            if gate is not None:
                with gate:  # type: ignore[attr-defined]
                    attach = self._spawn_attach(handle, gate)
            else:
                if authorize is not None:
                    reason = authorize()
                    if reason:
                        raise StartRefused(reason)
                attach = self._spawn_attach(handle, None)
        except StartRefused:
            raise
        except BaseException as exc:
            # ``popen_group`` can raise after the process already exists (e.g.
            # its process-registry bookkeeping fails), so no Popen handle
            # reaches us.  Reap any attach client by its argv before failing.
            leftover = self._reap_attach_survivors(handle.container_id)
            detail = "" if not leftover else f"; {leftover}"
            raise SupervisorError(
                f"could not attach to worker container: {_sanitize(str(exc))}{detail}"
            ) from exc
        handle.attach = attach
        # The daemon can switch between the check above and the spawn (the CLI
        # process is started by us, but the endpoint is external).  Re-bind
        # before any stream is handed to the engine: an attach client talking to
        # another Engine must never be followed by a request write.
        changed = self._bind_daemon(handle)
        if changed:
            self._stop_attach(handle)
            leftover = self._reap_attach_survivors(handle.container_id)
            if leftover:
                changed = f"{changed}; {leftover}"[:1000]
            raise SupervisorError(changed)
        handle.started = True

    def _spawn_attach(self, handle: WorkerHandle, gate: object | None) -> subprocess.Popen:
        """Spawn the attach client with a final cancel/stop check before Popen.

        ``children.popen_group`` is entered before its fork, so a cancellation
        that commits while a caller gates that helper would still be followed by
        an exec.  This spawn keeps the same process-group registration
        (``_GROUP_LEADERS`` + ``register``) but runs the engine gate's last check
        after any test/caller gate and immediately before the real Popen.
        """

        if self._before_spawn is not None:
            self._before_spawn()
        if gate is not None:
            gate.check()  # type: ignore[attr-defined]
        argv = self._attach_argv(handle)
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=scrubbed_env(self._cli_env()), start_new_session=True,
            )
        except OSError as exc:
            raise SupervisorError(f"could not attach to worker container: {_sanitize(str(exc))}") from exc
        # Same bookkeeping as ``children.popen_group``: the group leader is
        # registered so kill_group/forget keep their process-group semantics.
        _GROUP_LEADERS.add(process.pid)
        register(process)
        return process

    def _attach_argv(self, handle: WorkerHandle) -> list[str]:
        argv = [self.config.docker]
        if self.config.docker_host:
            argv += ["--host", self.config.docker_host]
        argv += ["start", "--attach", "--interactive", handle.container_id]
        return argv

    def streams(self, handle: WorkerHandle) -> tuple[BinaryIO | None, BinaryIO, BinaryIO]:
        attach = handle.attach
        if attach is None or attach.stdin is None or attach.stdout is None or attach.stderr is None:
            raise SupervisorError("container attach stream is not available")
        return attach.stdin, attach.stdout, attach.stderr

    def poll(self, handle: WorkerHandle) -> int | None:
        error = self._bind_daemon(handle)
        if error:
            raise SupervisorError(error)
        status, error = self._status_for(handle.container_id)
        if error:
            raise SupervisorError(f"could not poll the worker container: {error}")
        if status.state == "exited":
            return status.exit_code if status.exit_code is not None else -1
        return None

    def _verify_still_bound(self, handle: WorkerHandle, what: str) -> None:
        error = self._bind_daemon(handle)
        if error:
            raise SupervisorError(f"{what}: {error}")

    def wait(self, handle: WorkerHandle, timeout: float) -> int | None:
        attach = handle.attach
        if attach is None:
            raise SupervisorError("container wait requires an attached start")
        self._verify_still_bound(handle, "worker deadline status is not bound")
        try:
            attach.wait(timeout=max(0.001, timeout))
        except subprocess.TimeoutExpired:
            status, error = self._status_for(handle.container_id)
            if error:
                raise SupervisorError(f"could not verify the worker container deadline: {error}")
            if status.state == "exited":
                # The container is done; only a stuck attach client remains.
                if status.exit_code is None:
                    raise SupervisorError("worker container exited without an exit code")
                self._verify_still_bound(handle, "worker exit status is not bound")
                self._stop_attach(handle)
                return status.exit_code
            if status.state != "running":
                # Absent/created/unknown is not a deadline: returning None here
                # would report a candidate timeout for a status we cannot read.
                raise SupervisorError(
                    f"worker container state at the deadline is {status.state}, not running"
                )
            self._verify_still_bound(handle, "worker deadline status is not bound")
            return None
        status, error = self._status_for(handle.container_id)
        if error:
            raise SupervisorError(f"could not verify worker container exit: {error}")
        if status.state in {"running", "created"}:
            raise SupervisorError("container attach ended while the worker container is still running")
        if status.state != "exited":
            raise SupervisorError(f"worker container state after exit is {status.state}")
        if status.exit_code is None:
            raise SupervisorError("worker container exited without an exit code")
        self._verify_still_bound(handle, "worker exit status is not bound")
        return status.exit_code

    def exit_code(self, handle: WorkerHandle) -> int | None:
        error = self._bind_daemon(handle)
        if error:
            raise SupervisorError(error)
        status, error = self._status_for(handle.container_id)
        if error:
            raise SupervisorError(f"could not read the worker exit code: {error}")
        if status.state == "exited":
            self._verify_still_bound(handle, "worker exit code is not bound")
            return status.exit_code
        return None

    def status(self, handle: WorkerHandle) -> WorkerStatus:
        error = self._bind_daemon(handle)
        if error:
            raise SupervisorError(error)
        status, error = self._status_for(handle.container_id)
        if error:
            raise SupervisorError(error)
        return status

    # -- teardown primitives ----------------------------------------------
    def _stop_attach(self, handle: WorkerHandle) -> None:
        attach = handle.attach
        if attach is None:
            return
        try:
            if attach.poll() is None:
                kill_group(attach)
        except OSError:
            pass
        finally:
            forget(attach)

    def _kill_ref(self, ref: str) -> str:
        status, error = self._status_for(ref)
        if error:
            return f"could not verify container before kill: {error}"
        if status.state in {"absent", "exited"}:
            return ""
        if status.state == "created":
            # A container that never ran has no process to signal, but a
            # concurrent ``docker start`` could still start it after this
            # call.  Force removal is the only proof it can never run the
            # request, so a cancellation cannot be overtaken by a slow start.
            return self._delete_ref(ref)
        completed = self._cli(["kill", "--signal", "SIGKILL", ref], timeout=self.config.kill_timeout_s)
        if completed.returncode != 0:
            again, again_error = self._status_for(ref)
            if again_error:
                return f"could not verify container after kill: {again_error}"
            if again.state not in {"absent", "exited"}:
                return _sanitize(
                    completed.stderr.decode("utf-8", errors="replace") or "docker kill failed"
                )
        deadline = time.monotonic() + self.config.kill_timeout_s
        while time.monotonic() < deadline:
            current, current_error = self._status_for(ref)
            if current_error:
                return f"could not verify container kill: {current_error}"
            if current.state in {"absent", "exited"}:
                return ""
            time.sleep(_KILL_POLL_S)
        return "worker container did not stop before the kill deadline"

    def _delete_ref(self, ref: str) -> str:
        completed = self._cli(["rm", "--force", ref], timeout=self.config.kill_timeout_s)
        if completed.returncode != 0 and not self._missing(completed):
            return _sanitize(completed.stderr.decode("utf-8", errors="replace") or "docker rm failed")
        return ""

    def _attach_executable_names(self) -> frozenset[str]:
        """Executables whose processes can be this supervisor's attach clients."""

        names = {os.path.basename(self.config.docker)}
        try:
            with open(self.config.docker, "rb") as handle:
                first = handle.readline(256)
        except OSError:
            return frozenset(name for name in names if name)
        if first.startswith(b"#!"):
            parts = first[2:].strip().split()
            if parts:
                names.add(os.path.basename(parts[-1].decode("utf-8", errors="replace")))
        return frozenset(name for name in names if name)

    def _attach_survivors(self, ref: str) -> tuple[list[int], str]:
        """Host processes still attaching to ``ref`` (a crash leaves them).

        ``docker start --attach`` is a client, not part of the container: a
        frontend crash can leave it running after the container record is gone.
        Recovery cannot verify "nothing survived" without looking for it.  The
        argv is read from the kernel (``/proc`` or ``sysctl``) because a fenced
        host may deny executing ``ps``.
        """

        if not ref or _CONTAINER_ID.fullmatch(ref) is None:
            return [], ""
        try:
            return hostprocs.pids_mentioning(
                ref, "--attach", executable_names=self._attach_executable_names(),
            ), ""
        except OSError as exc:
            return [], f"could not inspect host processes: {_sanitize(str(exc))}"

    def _reap_attach_survivors(self, ref: str) -> str:
        """Kill leftover attach clients for ``ref`` and prove they are gone."""

        survivors, error = self._attach_survivors(ref)
        if error:
            return error
        for pid in survivors:
            try:
                still = hostprocs.pid_mentions(pid, ref, "--attach")
            except OSError as exc:
                return f"could not re-verify a surviving attach client: {_sanitize(str(exc))}"
            if not still:
                # The PID vanished or was reused by an unrelated process since
                # the scan: never signal it as an attach client.
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
            except OSError as exc:
                return f"could not stop a surviving attach client: {_sanitize(str(exc))}"
        deadline = time.monotonic() + self.config.kill_timeout_s
        while True:
            survivors, error = self._attach_survivors(ref)
            if error:
                return error
            if not survivors:
                return ""
            if time.monotonic() >= deadline:
                return "an attach client for the worker container survived cleanup"
            time.sleep(_KILL_POLL_S)

    def _label_duplicates(self, attempt_id: str) -> tuple[list[str], str]:
        completed = self._cli(["ps", "--all", "--quiet", "--filter", f"label={_SCRATCH_LABEL_PREFIX}={attempt_id}"])
        if completed.returncode != 0:
            return [], _sanitize(completed.stderr.decode("utf-8", errors="replace") or "docker ps failed")
        ids = [line.strip() for line in completed.stdout.decode("utf-8", errors="replace").splitlines() if line.strip()]
        if any(_CONTAINER_ID.fullmatch(value) is None for value in ids):
            return [], "docker ps returned an unexpected container id"
        return ids, ""

    def _verify_ref_absent(self, ref: str, attempt_id: str, *, container_id: str = "") -> str:
        status, error = self._status_for(ref)
        if error:
            return f"could not verify container absence: {error}"
        if status.state != "absent":
            return f"worker container is still present ({status.state})"
        duplicates, duplicates_error = self._label_duplicates(attempt_id)
        if duplicates_error:
            return f"could not verify container labels: {duplicates_error}"
        if duplicates:
            return "a label-matched worker container is still present"
        return self._cgroup_residue(container_id)

    def kill(self, handle: WorkerHandle) -> str:
        error = self._bind_daemon(handle)
        if error:
            return error
        error = self._kill_ref(handle.container_id or handle.container_name)
        self._stop_attach(handle)
        # A switch during the kill means the observation (and possibly the
        # signal) belonged to another Engine; never report a verified stop.
        changed = self._bind_daemon(handle)
        if changed:
            return changed if not error else f"{error}; {changed}"
        return error

    def delete(self, handle: WorkerHandle) -> str:
        error = self._bind_daemon(handle)
        if error:
            return error
        error = self._delete_ref(handle.container_id or handle.container_name)
        self._stop_attach(handle)
        changed = self._bind_daemon(handle)
        if changed:
            return changed if not error else f"{error}; {changed}"
        if handle.scratch_dir is not None:
            scratch = Path(handle.scratch_dir)
            try:
                shutil.rmtree(scratch)
            except FileNotFoundError:
                pass
            except OSError as exc:
                error = error or f"could not remove the per-attempt scratch directory: {_sanitize(str(exc))}"
            if scratch.exists():
                error = error or "per-attempt scratch directory survived cleanup"
        changed = self._bind_daemon(handle)
        if changed:
            return changed if not error else f"{error}; {changed}"
        return error

    def _cgroup_paths(self, container_id: str) -> list[Path]:
        if not container_id or _CONTAINER_ID.fullmatch(container_id) is None:
            return []
        root = Path(self.config.cgroup_root)
        return [
            root / "system.slice" / f"docker-{container_id}.scope",
            root / "docker" / container_id,
        ]

    def _cgroup_residue(self, container_id: str) -> str:
        paths = self._cgroup_paths(container_id)
        deadline = time.monotonic() + 1.0
        while True:
            for path in paths:
                try:
                    present = path.exists()
                except OSError as exc:
                    return f"could not verify the container cgroup: {_sanitize(str(exc))}"
                if present:
                    if time.monotonic() < deadline:
                        time.sleep(_KILL_POLL_S)
                        break
                    return f"container cgroup survived cleanup: {path}"
            else:
                return ""

    @staticmethod
    def _env_file_for(attempt_id: str, state_dir: Path) -> Path:
        return Path(state_dir) / "container-env" / f"{attempt_id}.env"

    def _env_file_residue(self, attempt_id: str, state_dir: Path) -> str:
        path = self._env_file_for(attempt_id, state_dir)
        try:
            if path.exists():
                return "per-attempt container environment file was not removed"
        except OSError as exc:
            return f"could not verify the per-attempt container environment file: {_sanitize(str(exc))}"
        return ""

    def _remove_env_file(self, attempt_id: str, state_dir: Path) -> str:
        path = self._env_file_for(attempt_id, state_dir)
        try:
            path.unlink()
        except FileNotFoundError:
            return ""
        except OSError as exc:
            return f"could not remove the recovered per-attempt env file: {_sanitize(str(exc))}"
        return ""

    def verify_absent(self, handle: WorkerHandle) -> str:
        error = self._bind_daemon(handle)
        if error:
            return error
        error = self._verify_ref_absent(
            handle.container_id or handle.container_name, handle.attempt_id,
            container_id=handle.container_id,
        )
        if error:
            return error
        return self._bind_daemon(handle)

    def cleanup(self, handle: WorkerHandle) -> str:
        error = self.kill(handle)
        if error:
            return f"kill: {error}"
        error = self.delete(handle)
        if error:
            return f"delete: {error}"
        error = self.verify_absent(handle)
        if error:
            return error
        # A lost attach client is not observable through the handle or the
        # container record: prove none survives by argv.
        return self._reap_attach_survivors(handle.container_id or handle.container_name)

    def cleanup_residue(self, handle: WorkerHandle) -> str:
        error = self.verify_absent(handle)
        if error:
            return error
        error = self._env_file_residue(handle.attempt_id, handle.spec.state_dir)
        if error:
            return error
        if handle.scratch_dir is not None and Path(handle.scratch_dir).exists():
            return "per-attempt scratch directory survived cleanup"
        attach = handle.attach
        if attach is not None and attach.poll() is None:
            return "container attach client survived cleanup"
        survivors, survivors_error = self._attach_survivors(handle.container_id)
        if survivors_error:
            return survivors_error
        if survivors:
            return "container attach client survived cleanup"
        # The residue proof is only valid for the daemon it was taken on.
        return self._bind_daemon(handle)

    def cleanup_recovered(self, attempt: Attempt, *, state_dir: Path, daemon: str = "") -> str:
        """Clean one recovered attempt, bound to its daemon and its temp state.

        ``daemon`` is the current engine's daemon identity, used only when the
        attempt never reached the durable attach (a create-time crash), so no
        daemon was persisted.  A persisted daemon that differs from this
        engine's is a hard stop: "absent on another daemon" proves nothing.
        """

        if getattr(attempt, "execution_kind", "") != self.kind:
            return (
                "running attempt was not started by the container supervisor; "
                "refusing to guess its cleanup"
            )
        persisted = getattr(attempt, "container_daemon", "") or ""
        expected_daemon = persisted or daemon
        if not expected_daemon:
            return "recovered container attempt has no persisted Docker daemon identity; manual cleanup required"
        def guard(prefix: str) -> str:
            return self._guard_daemon(expected_daemon, prefix)

        error = guard("could not verify the recovered worker's Docker daemon")
        if error:
            return error
        stored_id = getattr(attempt, "container_id", "") or ""
        stored_name = getattr(attempt, "container_name", "") or container_name_for(attempt.attempt_id)
        if stored_id and _CONTAINER_ID.fullmatch(stored_id) is None:
            return "persisted container id is invalid; manual cleanup required"
        ref = stored_id or stored_name
        error = guard("could not inspect the recovered worker container")
        if error:
            return error
        document, error = self._inspect(ref)
        if error:
            return f"could not inspect the recovered worker container: {error}"
        # The inspect only counts for the daemon it ran on; a switch right after
        # the entry check must not let kill/delete run against another Engine.
        error = guard("could not inspect the recovered worker container")
        if error:
            return error
        resolved_id = stored_id
        if document is not None:
            labels = (document.get("Config") or {}).get("Labels") or {}
            if labels.get(_SCRATCH_LABEL_PREFIX) != attempt.attempt_id:
                return "recovered container label does not match the persisted attempt"
            if stored_id and document.get("Id") != stored_id:
                return "recovered container id does not match the persisted id"
            if not resolved_id:
                resolved_id = str(document.get("Id") or "")
                if not resolved_id or _CONTAINER_ID.fullmatch(resolved_id) is None:
                    return "recovered container has no usable full id; manual cleanup required"
            error = guard("could not stop the recovered worker container")
            if error:
                return error
            error = self._kill_ref(ref)
            if error:
                return f"could not stop the recovered worker container: {error}"
            error = guard("could not stop the recovered worker container")
            if error:
                return error
            error = guard("could not delete the recovered worker container")
            if error:
                return error
            error = self._delete_ref(ref)
            if error:
                return f"could not delete the recovered worker container: {error}"
            error = guard("could not delete the recovered worker container")
            if error:
                return error
            # A name-only recovery must still prove the full id's cgroup is gone.
            error = self._cgroup_residue(resolved_id)
            if error:
                return error
        else:
            # A crash between ``create`` and the durable attach leaves no id.
            # The deterministic name is still authoritative; the label check
            # above already refused a name collision.
            error = guard("could not verify recovered container cleanup")
            if error:
                return error
            duplicates, duplicates_error = self._label_duplicates(attempt.attempt_id)
            if duplicates_error:
                return f"could not verify recovered container cleanup: {duplicates_error}"
            error = guard("could not verify recovered container cleanup")
            if error:
                return error
            for duplicate in duplicates:
                error = guard("could not stop a recovered worker container")
                if error:
                    return error
                error = self._kill_ref(duplicate)
                if error:
                    return f"could not stop a recovered worker container: {error}"
                error = guard("could not stop a recovered worker container")
                if error:
                    return error
                error = guard("could not delete a recovered worker container")
                if error:
                    return error
                error = self._delete_ref(duplicate)
                if error:
                    return f"could not delete a recovered worker container: {error}"
                error = guard("could not delete a recovered worker container")
                if error:
                    return error
                # Every duplicate is verified by its full id, not only by the
                # deterministic name.
                error = self._cgroup_residue(duplicate)
                if error:
                    return error
        error = guard("could not prove the recovered container absent")
        if error:
            return error
        error = self._verify_ref_absent(ref, attempt.attempt_id, container_id=resolved_id)
        if error:
            return error
        error = guard("could not prove the recovered container absent")
        if error:
            return error
        # A crashed frontend can leave the ``docker start --attach`` client of
        # this exact container behind.  Recovery owns the container, so it owns
        # the client too: reap it and prove it is gone before reporting a clean
        # recovery (an operator release would otherwise re-check the residue).
        error = guard("could not reap the recovered worker's attach client")
        if error:
            return error
        error = self._reap_attach_survivors(resolved_id or ref)
        if error:
            return error
        error = guard("could not reap the recovered worker's attach client")
        if error:
            return error
        error = guard("could not clear the recovered worker's environment file")
        if error:
            return error
        error = self._remove_env_file(attempt.attempt_id, state_dir)
        if error:
            return error
        error = self._env_file_residue(attempt.attempt_id, state_dir)
        if error:
            return error
        # Never report a clean recovery without a last binding to the daemon the
        # attempt was created on.
        return guard("could not verify the recovered worker's Docker daemon")

    def release(self, handle: WorkerHandle) -> None:
        self._stop_attach(handle)
