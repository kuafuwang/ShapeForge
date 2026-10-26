"""Attempt execution supervisors for the external evaluator.

The job engine owns the durable queue, quota, lease/fence, idempotency and
result contract.  It delegates only the *execution* of one attempt to an
:class:`AttemptSupervisor`: create one worker, stream the private request into
it, wait, kill, delete and verify that nothing is left behind.

Two supervisors exist:

* :class:`ProcessAttemptSupervisor` is the historical local development path
  (one ``popen_group`` worker per attempt, killed by process group).
* ``tcenv.remote.container_supervisor.DockerAttemptSupervisor`` is the
  production execution plane: one disposable container per attempt, supervised
  by the Docker Engine/containerd lifecycle.

Selection is explicit.  There is no fallback between them: a supervisor that
cannot create, start, stop or verify its worker reports the failure and the
attempt is interrupted or quarantined instead of being retried on the other
kind.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Protocol, runtime_checkable

from .job_types import (
    EXECUTION_CONTAINER,
    EXECUTION_KINDS,
    EXECUTION_PROCESS,
    Attempt,
    ExternalEvaluationRequest,
)

__all__ = [
    "EXECUTION_CONTAINER", "EXECUTION_KINDS", "EXECUTION_PROCESS",
    "AttemptSpec", "AttemptSupervisor", "ProcessAttemptSupervisor",
    "SupervisorError", "WorkerHandle", "WorkerStatus",
]

_CLEANUP_POLL_S = 0.02
_CLEANUP_DEADLINE_S = 3.0


class SupervisorError(RuntimeError):
    """A bounded, sanitized failure of the execution supervisor itself."""


@dataclass(frozen=True)
class AttemptSpec:
    """Everything one supervisor needs to create exactly one worker."""

    request: ExternalEvaluationRequest
    attempt: Attempt
    state_dir: Path
    command: tuple[str, ...]
    env: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.command or any(not isinstance(part, str) or not part for part in self.command):
            raise SupervisorError("worker command must be a non-empty argv list")
        if any("\x00" in part for part in self.command):
            raise SupervisorError("worker command contains a NUL byte")


@dataclass
class WorkerHandle:
    """The in-memory handle of one created worker."""

    spec: AttemptSpec
    kind: str
    process: subprocess.Popen | None = None
    container_id: str = ""
    container_name: str = ""
    image_ref: str = ""
    image_digest: str = ""
    runtime: str = ""
    attach: subprocess.Popen | None = None
    started: bool = False
    scratch_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def attempt_id(self) -> str:
        return self.spec.attempt.attempt_id

    @property
    def lease_fence(self) -> int:
        return self.spec.attempt.lease_fence

    def durable_identity(self) -> dict[str, Any]:
        """The fields persisted *before* the worker may receive any request."""

        return {
            "execution_kind": self.kind,
            "container_id": self.container_id,
            "container_name": self.container_name,
            "container_image_ref": self.image_ref,
            "container_image_digest": self.image_digest,
            "container_runtime": self.runtime,
        }


@dataclass(frozen=True)
class WorkerStatus:
    """Normalized supervisor status for one worker handle."""

    state: str  # created | running | exited | absent | unknown
    exit_code: int | None = None
    pid: int | None = None


@runtime_checkable
class AttemptSupervisor(Protocol):
    """Narrow execution boundary used by :class:`~.job_engine.JobEngine`."""

    kind: str

    def create(self, spec: AttemptSpec) -> WorkerHandle:
        """Create exactly one worker without letting it run the request yet."""

    def start(self, handle: WorkerHandle) -> None:
        """Start the created worker and attach the bounded private stream."""

    def streams(self, handle: WorkerHandle) -> tuple[BinaryIO | None, BinaryIO, BinaryIO]:
        """Return ``(stdin, stdout, stderr)`` for the private worker protocol."""

    def poll(self, handle: WorkerHandle) -> int | None: ...

    def wait(self, handle: WorkerHandle, timeout: float) -> int | None:
        """Wait up to ``timeout``; return the exit code or ``None`` on timeout."""

    def exit_code(self, handle: WorkerHandle) -> int | None: ...

    def status(self, handle: WorkerHandle) -> WorkerStatus: ...

    def verify_identity(self, handle: WorkerHandle) -> str:
        """Return "" when the created worker matches its pinned identity."""

    def kill(self, handle: WorkerHandle) -> str:
        """Idempotently stop the worker; return "" only when it is known stopped."""

    def delete(self, handle: WorkerHandle) -> str:
        """Idempotently remove the worker and its scratch state."""

    def verify_absent(self, handle: WorkerHandle) -> str:
        """Return "" only when no worker process/container remains."""

    def cleanup(self, handle: WorkerHandle) -> str:
        """Kill, delete and verify one worker; return "" when fully gone."""

    def cleanup_residue(self, handle: WorkerHandle) -> str:
        """Check for leftover child processes and per-attempt temp files."""

    def cleanup_recovered(self, attempt: Attempt) -> str:
        """Clean a worker recovered from durable state after a frontend restart."""

    def release(self, handle: WorkerHandle) -> None:
        """Drop in-memory registrations after confirmed cleanup."""


class ProcessAttemptSupervisor:
    """One local process-group worker per attempt (local development only).

    This preserves the historical behavior exactly: the worker is spawned in a
    new session before the durable worker identity is attached, stdin carries
    the private request, and the whole process group is SIGKILLed on any stop
    path.  It is not an isolation boundary; the production composition root uses
    the container supervisor instead.
    """

    kind = EXECUTION_PROCESS

    def __init__(
        self,
        *,
        spawn: Callable[[], subprocess.Popen],
        kill: Callable[[subprocess.Popen], object],
        group_alive: Callable[[int], bool],
        forget: Callable[[subprocess.Popen], object],
    ) -> None:
        self._spawn = spawn
        self._kill = kill
        self._group_alive = group_alive
        self._forget = forget

    @staticmethod
    def _process(handle: WorkerHandle) -> subprocess.Popen:
        process = handle.process
        if process is None:
            raise SupervisorError("process worker handle has no process")
        return process

    def create(self, spec: AttemptSpec) -> WorkerHandle:
        # argv/env for this path are owned by the engine's ``_popen`` hook; the
        # spec still carries them so both supervisors record the same request.
        process = self._spawn()
        return WorkerHandle(spec=spec, kind=self.kind, process=process)

    def start(self, handle: WorkerHandle) -> None:
        handle.started = True

    def streams(self, handle: WorkerHandle) -> tuple[BinaryIO | None, BinaryIO, BinaryIO]:
        process = self._process(handle)
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise SupervisorError("process worker was not created with piped stdio")
        return process.stdin, process.stdout, process.stderr

    def poll(self, handle: WorkerHandle) -> int | None:
        return self._process(handle).poll()

    def wait(self, handle: WorkerHandle, timeout: float) -> int | None:
        try:
            return self._process(handle).wait(timeout=max(0.001, timeout))
        except subprocess.TimeoutExpired:
            return None

    def exit_code(self, handle: WorkerHandle) -> int | None:
        return self._process(handle).returncode

    def status(self, handle: WorkerHandle) -> WorkerStatus:
        process = self._process(handle)
        code = process.poll()
        if code is None:
            return WorkerStatus("running", None, process.pid)
        return WorkerStatus("exited", code, process.pid)

    def verify_identity(self, handle: WorkerHandle) -> str:
        return ""

    def kill(self, handle: WorkerHandle) -> str:
        process = self._process(handle)
        try:
            self._kill(process)
        except OSError as exc:
            return f"could not kill worker process group: {exc}"
        return ""

    def delete(self, handle: WorkerHandle) -> str:
        return ""

    def verify_absent(self, handle: WorkerHandle) -> str:
        pid = self._process(handle).pid
        if self._group_alive(pid):
            return "worker process group survived cleanup"
        return ""

    def cleanup(self, handle: WorkerHandle) -> str:
        error = self.kill(handle)
        if error:
            return error
        pid = self._process(handle).pid
        deadline = time.monotonic() + _CLEANUP_DEADLINE_S
        while self._group_alive(pid) and time.monotonic() < deadline:
            time.sleep(_CLEANUP_POLL_S)
        return self.verify_absent(handle)

    def cleanup_residue(self, handle: WorkerHandle) -> str:
        del handle
        return ""

    def cleanup_recovered(self, attempt: Attempt) -> str:
        raise SupervisorError("process recovery is owned by the job engine")

    def release(self, handle: WorkerHandle) -> None:
        process = handle.process
        if process is not None:
            self._forget(process)
