"""Single-active, transport-neutral external evaluator job/attempt engine."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from tcenv.children import forget, kill_group, popen_group, scrubbed_env
from tcenv.schema import PROTOCOL_VERSION as EXECUTION_PROTOCOL_VERSION

from .contracts import API_CONTRACT_ID, EVIDENCE_CONTRACT_ID
from .identity import ServiceIdentity, SessionGrant
from .job_store import JobStore
from .job_types import (
    EXECUTION_CONTAINER,
    EXECUTION_PROCESS,
    Attempt,
    Capabilities,
    ExpectedEvaluator,
    ExpectedIdentities,
    ExternalEvaluationRequest,
    JobError,
    JobEvent,
    JobFailure,
    JobSnapshot,
    StateCorrupt,
)
from .protocol import FrameDecoder
from .supervisor import (
    AttemptSpec,
    AttemptSupervisor,
    ProcessAttemptSupervisor,
    SupervisorError,
    WorkerHandle,
)
from .worker_protocol import (
    MAX_WORKER_FRAME_PAYLOAD,
    STATE_DIR_ENV,
    decode_worker_frame,
    encode_request,
)

MAX_STDERR = 4000
_CLOSE_SCHEDULER_JOIN_S = 1.0
# Bounded synchronous teardown for a container Cancel RPC (kill + delete +
# verify).  The scheduler performs the same teardown asynchronously if this
# window expires; the job is already durably cancelled either way.
CONTAINER_CANCEL_TEARDOWN_S = 20.0
_PROC_PIDTBSDINFO = 3
_MACOS_BSDPROC_INFO = None


def _macos_libproc_start_time(pid: int) -> tuple[int, int] | None:
    """Read (start_sec, start_usec) through libproc without executing ``ps``.

    A hardened or fenced host can deny executing ``ps`` while still allowing the
    same process-start information through ``proc_pidinfo``.  This is the same
    operating-system source ``ps -o lstart=`` reports, only at higher resolution;
    it never invents an identity for a process that cannot be inspected.
    """

    global _MACOS_BSDPROC_INFO
    if pid <= 0:
        return None
    try:
        import ctypes
        import ctypes.util

        if _MACOS_BSDPROC_INFO is None:
            class ProcBsdInfo(ctypes.Structure):
                _fields_ = [
                    ("pbi_flags", ctypes.c_uint32), ("pbi_status", ctypes.c_uint32),
                    ("pbi_xstatus", ctypes.c_uint32), ("pbi_pid", ctypes.c_uint32),
                    ("pbi_ppid", ctypes.c_uint32), ("pbi_uid", ctypes.c_uint32),
                    ("pbi_gid", ctypes.c_uint32), ("pbi_ruid", ctypes.c_uint32),
                    ("pbi_rgid", ctypes.c_uint32), ("pbi_svuid", ctypes.c_uint32),
                    ("pbi_svgid", ctypes.c_uint32), ("rfu_1", ctypes.c_uint32),
                    ("pbi_comm", ctypes.c_char * 16), ("pbi_name", ctypes.c_char * 32),
                    ("pbi_nfiles", ctypes.c_uint32), ("pbi_pgid", ctypes.c_uint32),
                    ("pbi_pjobc", ctypes.c_uint32), ("e_tdev", ctypes.c_uint32),
                    ("e_tpgid", ctypes.c_uint32), ("pbi_nice", ctypes.c_int32),
                    ("pbi_start_tvsec", ctypes.c_uint64),
                    ("pbi_start_tvusec", ctypes.c_uint64),
                ]

            library = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
            library.proc_pidinfo.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                ctypes.c_void_p, ctypes.c_int,
            ]
            library.proc_pidinfo.restype = ctypes.c_int
            _MACOS_BSDPROC_INFO = (library, ProcBsdInfo)
        library, structure = _MACOS_BSDPROC_INFO
        info = structure()
        written = library.proc_pidinfo(
            int(pid), _PROC_PIDTBSDINFO, 0, ctypes.byref(info), ctypes.sizeof(info),
        )
        if written != ctypes.sizeof(info) or info.pbi_start_tvsec <= 0:
            return None
        return int(info.pbi_start_tvsec), int(info.pbi_start_tvusec)
    except (ImportError, OSError, AttributeError, ValueError):
        return None


def _macos_start_token(pid: int) -> str | None:
    value = _macos_libproc_start_time(pid)
    if value is None:
        return None
    return f"libproc:{value[0]}.{value[1]:06d}"


def _process_start_token_once(pid: int) -> str | None:
    if pid <= 0:
        return None
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.is_file():
        try:
            raw = stat_path.read_text(encoding="utf-8")
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
            tail = raw[raw.rfind(")") + 2 :].split()
            if not boot_id:
                return None
            return f"linux:{boot_id}:{tail[19]}"
        except (OSError, IndexError):
            return None
    boot_time = ""
    if sys.platform == "darwin":
        try:
            boot = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "kern.boottime"], text=True,
                capture_output=True, timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return _macos_start_token(pid)
        boot_time = boot.stdout.strip()
        if boot.returncode != 0 or not boot_time:
            return _macos_start_token(pid)
    try:
        completed = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="], text=True,
            capture_output=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _macos_start_token(pid) if sys.platform == "darwin" else None
    value = completed.stdout.strip()
    if completed.returncode == 0 and value:
        return f"ps:{boot_time}:{value}"
    # ``ps`` can also refuse a live process without raising (fenced hosts and
    # hardened sandboxes return a nonzero exit).  Fall back to the same kernel
    # information instead of turning a healthy worker into an infrastructure
    # failure.
    return _macos_start_token(pid) if sys.platform == "darwin" else None


def process_start_token(pid: int, *, attempts: int = 1) -> str | None:
    """Return a PID-reuse guard, with bounded retries for a loaded host.

    Recovery uses one probe so an already-dead worker is not delayed.  A newly
    spawned worker uses three probes because a short ``ps`` timeout must not be
    mistaken for a candidate/infrastructure failure on a busy frontend.  When
    ``ps`` is unavailable on macOS the token comes from ``libproc``; a process
    that cannot be inspected at all still yields no token (fail closed).
    """

    for index in range(max(1, attempts)):
        token = _process_start_token_once(pid)
        if token is not None:
            return token
        if index + 1 < attempts:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return None
            time.sleep(0.05)
    return None


def _group_exists(pgid: int) -> bool:
    if pgid <= 0:
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@dataclass
class _Running:
    request: ExternalEvaluationRequest
    attempt: Attempt
    started_monotonic: float = field(default_factory=time.monotonic)
    handle: WorkerHandle | None = None
    cancel_requested: bool = False
    stopping: bool = False
    teardown_done: threading.Event = field(default_factory=threading.Event)
    # Serializes the request-write authorization with cancellation.  The
    # decision "this worker may receive the request" is made under this lock,
    # so a durable cancel can never be followed by a newly authorized request.
    request_lock: threading.Lock = field(default_factory=threading.Lock)
    request_authorized: bool = False


class JobEngine:
    """One persistent queue and one active attempt for one device frontend."""

    def __init__(
        self,
        *, identity: ServiceIdentity,
        grants: Iterable[SessionGrant],
        worker_command: list[str],
        state_dir: Path,
        execution_protocol_version: str = EXECUTION_PROTOCOL_VERSION,
        supported_features: frozenset[str] = frozenset({"durable-jobs", "attempt-receipts", "progress"}),
        supported_modes: frozenset[str] = frozenset({"quick", "timed", "full"}),
        worker_cwd: Path | None = None,
        worker_env: Mapping[str, str] | None = None,
        signing_key: bytes | None = None,
        signing_key_id: str = "",
        signing_keyring: Mapping[str, bytes] | None = None,
        selfcheck_age_s: float = 60.0,
        require_evidence: bool = False,
        supervisor: AttemptSupervisor | None = None,
        cancel_teardown_s: float = CONTAINER_CANCEL_TEARDOWN_S,
        **store_options: Any,
    ) -> None:
        if not worker_command or not all(isinstance(part, str) and part for part in worker_command):
            raise ValueError("worker_command must be a non-empty argv list")
        if not isinstance(execution_protocol_version, str) or not execution_protocol_version:
            raise ValueError("execution_protocol_version must be non-empty")
        if not isinstance(require_evidence, bool):
            raise ValueError("require_evidence must be boolean")
        if isinstance(selfcheck_age_s, bool) or not isinstance(selfcheck_age_s, (int, float)) or not math.isfinite(selfcheck_age_s) or selfcheck_age_s <= 0:
            raise ValueError("selfcheck_age_s must be finite and positive")
        if signing_key is not None and (not isinstance(signing_key, bytes) or len(signing_key) < 32):
            raise ValueError("signing key must contain at least 32 bytes")
        if signing_key is not None and not signing_key_id:
            raise ValueError("signing_key_id is required with a signing key")
        if supervisor is None:
            # Historical local development path: one process-group worker per
            # attempt.  The engine's ``_popen`` hook stays the single spawn
            # point so existing local behavior (and its tests) are unchanged.
            supervisor = ProcessAttemptSupervisor(
                spawn=lambda: self._popen(),
                kill=lambda process: kill_group(process),
                group_alive=_group_exists,
                forget=forget,
            )
        if getattr(supervisor, "kind", None) not in (EXECUTION_PROCESS, EXECUTION_CONTAINER):
            raise ValueError("supervisor kind is not a supported execution kind")
        if supervisor.kind == EXECUTION_PROCESS and "+img:" in str(identity.worker_build):
            # A process worker cannot have run a pinned image: signing such a
            # worker_build would claim an image that never executed.
            raise ValueError("process execution cannot claim an image-pinned worker_build")
        if supervisor.kind == EXECUTION_CONTAINER:
            # The signed result provenance must bind the exact image that ran
            # the attempt.  ``worker_build`` carries the pinned digest, and the
            # container supervisor re-verifies the image id before create.
            digest = str(getattr(supervisor, "image_digest", "") or "")
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("container supervisor must expose its pinned image digest")
            if not identity.worker_build.endswith(f"+img:{digest}"):
                raise ValueError(
                    "container execution requires worker_build to bind the pinned image digest"
                )
        self.supervisor: AttemptSupervisor = supervisor
        if (
            isinstance(cancel_teardown_s, bool)
            or not isinstance(cancel_teardown_s, (int, float))
            or not 0 < float(cancel_teardown_s) <= 120
        ):
            raise ValueError("cancel_teardown_s must be a positive bounded number of seconds")
        self.cancel_teardown_s = float(cancel_teardown_s)
        self.identity = identity
        self.execution_protocol_version = execution_protocol_version
        self.supported_features = frozenset(supported_features)
        self.supported_modes = frozenset(supported_modes)
        self.worker_command = list(worker_command)
        self.worker_cwd = worker_cwd or Path(__file__).resolve().parents[2]
        self.worker_env = dict(worker_env or {})
        self._signing_key = signing_key
        self._signing_key_id = signing_key_id
        self.selfcheck_age_ms = int(selfcheck_age_s * 1000)
        self.require_evidence = require_evidence
        self._stop = threading.Event()
        self._condition = threading.Condition()
        # Re-entrant: the Cancel RPC holds it while committing the durable
        # cancellation and then calls the internal cancel helper.
        self._active_lock = threading.RLock()
        self._active: _Running | None = None
        self._closed = False
        self._unreported_hazard = ""
        self._expected = ExpectedIdentities(ExpectedEvaluator(
            site_id=identity.site_id, device_id=identity.device_id, target_id=identity.target_id,
            worker_build=identity.worker_build, task_pack_digest=identity.task_pack_digest,
            evaluator_epoch=identity.external_epoch(execution_protocol_version=execution_protocol_version),
        ), API_CONTRACT_ID, EVIDENCE_CONTRACT_ID)
        self._expected.validate()
        if len(identity.calibration_digest) != 64 or any(c not in "0123456789abcdef" for c in identity.calibration_digest):
            raise ValueError("external evaluator calibration_digest must be 64 lowercase hex characters")
        self.store = JobStore(
            state_dir, signing_key=signing_key, signing_key_id=signing_key_id,
            signing_keyring=signing_keyring, **store_options,
        )
        try:
            configured_scopes: set[str] = set()
            for grant in grants:
                try:
                    configured_scopes.add(self.store.register_grant(grant))
                except JobError as exc:
                    if exc.code != "already_exists":
                        raise
                    # A new token for the same subject/run/session/kind/target
                    # scope is a normal renewal: supersede the old generation
                    # without cancelling accepted jobs.
                    configured_scopes.add(self.store.renew_grant(grant))
            for scope_id in self.store.revoke_unlisted_scopes(configured_scopes):
                self.store.cancel_scope(scope_id)
            self._recover()
            for request_id in self.store.accepted_jobs():
                self.store.enqueue_accepted(request_id)
            self.store.finish_restore()
            self._scheduler = threading.Thread(target=self._schedule_loop, name="tcenv-external-jobs", daemon=True)
            self._scheduler.start()
        except Exception:
            self.store.close()
            raise

    @property
    def expected_identities(self) -> ExpectedIdentities:
        return self._expected

    def _required_features(self, values: Any) -> frozenset[str]:
        features = JobStore._features(values)
        missing = features - self.supported_features
        if missing:
            raise JobError(
                "failed_precondition",
                f"unsupported required features: {sorted(missing)}",
                missing_features=tuple(sorted(missing)),
            )
        return features

    @property
    def frontend_instance(self) -> str:
        return self.store.frontend_instance

    def _base_provenance(self) -> dict[str, Any]:
        return self.identity.external_provenance(
            execution_protocol_version=self.execution_protocol_version,
            api_contract_id=API_CONTRACT_ID, evidence_contract_id=EVIDENCE_CONTRACT_ID,
        )

    # -- transport-neutral RPC surface -------------------------------------
    def get_capabilities(
        self, *, grant_token: str, client_nonce: str,
        expected: Mapping[str, Any] | None = None,
        api_contract_ids: list[str] | None = None,
        evidence_contract_ids: list[str] | None = None,
        required_features: list[str] | tuple[str, ...] = (),
        valid_for_ms: int = 60_000,
    ) -> Capabilities:
        self._required_features(required_features)
        if not isinstance(expected, Mapping) or not expected:
            raise JobError("invalid_argument", "capabilities requires explicit expected evaluator identities")
        for name in ("site_id", "device_id", "target_id"):
            if not isinstance(expected.get(name), str) or not expected.get(name):
                raise JobError("invalid_argument", f"capabilities requires expected {name}")
        for name, values in (("api_contract_ids", api_contract_ids), ("evidence_contract_ids", evidence_contract_ids)):
            if not isinstance(values, (list, tuple)) or not values:
                raise JobError("invalid_argument", f"capabilities requires non-empty {name}")
        capabilities = self.store.get_capabilities(
            grant_token=grant_token, client_nonce=client_nonce, expected=expected,
            expected_identities=self._expected, provenance=self._base_provenance(),
            api_contract_ids=list(api_contract_ids),
            evidence_contract_ids=list(evidence_contract_ids),
            required_features=required_features, supported_features=self.supported_features,
            valid_for_ms=valid_for_ms,
            limits={
                "max_source_bytes": 1_000_000, "max_reps": 10_000,
                "max_timeout_ms": 3_600_000, "max_cases": 128,
                "max_queue_depth": self.store.max_queue_depth,
                "max_active_attempts": 1, "max_job_store_bytes": self.store.max_db_bytes,
                "max_blob_bytes": 16_000_000,
            },
        )
        return replace(capabilities, target_capabilities=({
            "target_id": self.identity.target_id,
            "backend": self.identity.backend,
            "modes": sorted(self.supported_modes),
            "features": sorted(self.supported_features),
        },))

    def _validate_submission(self, request: ExternalEvaluationRequest) -> None:
        if request.target_id != self.identity.target_id:
            raise JobError("failed_precondition", "target is not served by this evaluator")
        if not self.identity.accepts_task(request.task, request.dtype, request.size, request.task_fingerprint):
            raise JobError("failed_precondition", "task fingerprint does not match the deployed task pack")
        if request.mode not in self.supported_modes:
            raise JobError("failed_precondition", f"mode {request.mode!r} is not supported by this evaluator")

    def submit(
        self, *, request: ExternalEvaluationRequest, expected: ExpectedIdentities,
        capability_id: str, required_features: list[str] | tuple[str, ...] = (),
        grant_token: str,
    ) -> JobSnapshot:
        self._validate_submission(request)
        self._required_features(required_features)
        snapshot, _created = self.store.submit(
            request=request, request_digest=request.request_digest, expected=expected,
            provenance=self._base_provenance(), capability_id=capability_id,
            required_features=required_features, grant_token=grant_token,
        )
        if snapshot.state == "accepted" and not snapshot.quarantined:
            snapshot = self.store.enqueue_accepted(request.request_id)
            self._condition.acquire()
            self._condition.notify_all()
            self._condition.release()
        return snapshot

    def get_job(self, *, request_id: str, request_digest: str, expected: ExpectedIdentities, grant_token: str) -> JobSnapshot:
        return self.store.get_job(grant_token=grant_token, request_id=request_id, request_digest=request_digest, expected=expected)

    def watch(
        self, *, request_id: str, request_digest: str, expected: ExpectedIdentities,
        grant_token: str, after_sequence: int = 0, attempt_id: str | None = None,
    ) -> list[JobEvent]:
        return self.store.events(
            grant_token=grant_token, request_id=request_id, request_digest=request_digest,
            expected=expected, after_sequence=after_sequence, attempt_id=attempt_id,
        )

    def cancel(self, *, request_id: str, request_digest: str, expected: ExpectedIdentities, grant_token: str, reason: str = "") -> JobSnapshot:
        """Durably cancel, then stop any active worker.

        The durable cancellation and the start authorization share
        ``_active_lock``: a worker that has not been authorized to receive the
        request can never be started after the cancellation commits.  The RPC
        then waits for teardown within one total deadline; if teardown is still
        outstanding, the returned snapshot is the durable
        ``running``/``cancel_requested`` state (never a synchronous-teardown
        claim) and the scheduler finishes the cancellation.
        """

        deadline = time.monotonic() + self.cancel_teardown_s
        with self._active_lock:
            snapshot = self.store.cancel(
                grant_token=grant_token, request_id=request_id, request_digest=request_digest,
                expected=expected, reason=reason,
            )
            running = self._cancel_active_locked(request_id)
        self._wake()
        if (
            running is not None
            and self.supervisor.kind == EXECUTION_CONTAINER
            and running.request.request_id == request_id
        ):
            if not running.teardown_done.is_set():
                running.teardown_done.wait(timeout=max(0.0, deadline - time.monotonic()))
            # Always report the durable state: cancelled only once the worker is
            # gone, otherwise running+cancel_requested with teardown pending.
            snapshot = self.store._job_by_id(request_id)
        return snapshot

    def retry(
        self, *, request_id: str, request_digest: str, expected: ExpectedIdentities,
        interrupted_attempt_id: str, capability_id: str,
        required_features: list[str] | tuple[str, ...] = (), grant_token: str,
    ) -> JobSnapshot:
        self._required_features(required_features)
        snapshot = self.store.retry(
            grant_token=grant_token, request_id=request_id, request_digest=request_digest,
            expected=expected, interrupted_attempt_id=interrupted_attempt_id,
            capability_id=capability_id, required_features=required_features,
        )
        self._wake()
        return snapshot

    def renew_grant(self, grant: SessionGrant) -> str:
        return self.store.renew_grant(grant)

    def revoke_grant(self, token: str) -> None:
        scope_id = self.store.revoke_grant(token)
        with self._active_lock:
            # The durable revocation and every start authorization share this
            # lock, so a worker cannot be started after its grant was revoked.
            running = self.store.cancel_scope(scope_id)
            for request_id in running:
                self._cancel_active_locked(request_id)
        self._wake()

    def release_quarantine(self, request_id: str) -> JobSnapshot:
        snapshot = self.store._job_by_id(request_id)
        if snapshot.state == "running" and snapshot.quarantined:
            attempt = snapshot.current_attempt
            if attempt is None:
                raise StateCorrupt("quarantined running job has no attempt")
            # Operator release is an explicit request to retry cleanup, not
            # permission to forget a possibly live old worker or device lease.
            error = self._cleanup_recovered_worker(attempt)
            snapshot = self.store.recovered_attempt(request_id, attempt, cleanup_error=error)
            if snapshot.state == "running":
                return snapshot
        snapshot = self.store.release_quarantine(request_id)
        if snapshot.state == "accepted":
            snapshot = self.store.enqueue_accepted(request_id)
        self._wake()
        return snapshot

    def backup(self, destination: Path) -> Path:
        return self.store.backup(destination)

    def expire_jobs(self) -> int:
        return self.store.expire_jobs()

    # -- recovery -----------------------------------------------------------
    def _recover(self) -> None:
        for request_id, attempt, _quarantined in self.store.running_attempts():
            error = self._cleanup_recovered_worker(attempt)
            self.store.recovered_attempt(request_id, attempt, cleanup_error=error)

    def _cleanup_recovered_worker(self, attempt: Attempt) -> str:
        """Clean the worker a crashed frontend left behind, or refuse to guess.

        The attempt's durable ``execution_kind`` decides which supervisor may
        touch it.  A running attempt is never re-executed: recovery either
        confirms the old container/process group is gone and marks the attempt
        interrupted (explicit Retry only), or quarantines the job.
        """

        kind = getattr(attempt, "execution_kind", EXECUTION_PROCESS)
        if kind not in (EXECUTION_PROCESS, EXECUTION_CONTAINER):
            return "persisted execution kind is unknown; manual cleanup required"
        if kind != self.supervisor.kind:
            return "running attempt belongs to another execution supervisor; manual cleanup required"
        if kind == EXECUTION_CONTAINER:
            try:
                return self.supervisor.cleanup_recovered(attempt, state_dir=self.store.state_dir)
            except Exception as exc:
                return f"execution cleanup failed: {exc}"[:1000]
        return self._cleanup_recovered_process(attempt)

    @staticmethod
    def _cleanup_recovered_process(attempt: Attempt) -> str:
        pid = attempt.worker_pid
        pgid = attempt.worker_pgid
        if pid is None and pgid is None:
            # The durable running fence precedes spawn, so a crash between the
            # spawn and the PID attach leaves a live worker with no durable
            # identity.  Stdin EOF does not prove the process (or a child that
            # ignored it) is gone, so this must fail closed: operator release
            # has to recheck it manually rather than get a Retry.
            return (
                "running process attempt has no durable PID/start token; "
                "the old worker cannot be proven dead, so manual cleanup is required"
            )
        if pid is None or pgid != pid or pid <= 1 or pgid == os.getpgid(0):
            return "invalid persisted worker process identity; manual cleanup required"
        current = process_start_token(pid)
        if current is not None and current != attempt.worker_start_token:
            return "worker PID was reused or its start token differs; refusing to signal it"
        if current is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                return f"could not kill recovered worker group: {exc}"
        elif _group_exists(pgid):
            return "worker leader is dead but a process-group survivor cannot be verified"
        deadline = time.monotonic() + 3.0
        while _group_exists(pgid) and time.monotonic() < deadline:
            time.sleep(0.02)
        if _group_exists(pgid):
            return "recovered worker process group did not exit"
        return ""

    # -- scheduler and real process supervision ----------------------------
    def _wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _schedule_loop(self) -> None:
        while not self._stop.is_set():
            try:
                claimed = self.store.claim_next(
                    selfcheck_age_ms=self.selfcheck_age_ms,
                    execution_kind=self.supervisor.kind,
                )
            except Exception as exc:
                # Durable corruption is fail-closed: stop the scheduler and
                # make subsequent Submit/GetJob calls unavailable as well.
                self.store.mark_unavailable(exc)
                self._stop.set()
                break
            if claimed is None:
                with self._condition:
                    self._condition.wait(timeout=0.05)
                continue
            request, attempt = claimed
            running = _Running(request, attempt)
            with self._active_lock:
                self._active = running
            try:
                # Cancel may have committed after claim_next but before the
                # in-memory active record existed for the cancel path to find.
                if self.store._job_by_id(request.request_id).cancel_requested:
                    running.cancel_requested = True
                self._run_worker(running)
            except Exception as exc:
                # This boundary must contain every supervisor failure: an
                # uncaught exception here would kill the scheduler thread and
                # leave a running, unquarantined job (Retry offered) behind.
                cleanup_error = self._cleanup_supervisor_handle(running)
                detail = f"{type(exc).__name__}: {exc}"
                if cleanup_error:
                    detail = f"{detail}; {cleanup_error}"
                try:
                    if cleanup_error:
                        # Fail closed: an unexpected supervisor error must not
                        # turn into "interrupted, safe to Retry" while the old
                        # worker may still be alive.
                        self._record_cleanup_hazard(running, detail[:1000])
                    elif isinstance(exc, JobError):
                        self._handle_worker_job_error(running, exc)
                    else:
                        self.store.interrupt_attempt(request.request_id, attempt, JobFailure(
                            "infrastructure", True, "worker_supervisor_failed", detail[:1000],
                        ))
                except Exception:
                    # Even the durable write failed: keep the loop alive, mark
                    # the store unavailable, and remember the hazard in memory.
                    self._unreported_hazard = detail[:1000]
                    try:
                        self.store.mark_unavailable(exc)
                    finally:
                        self._stop.set()
            finally:
                with self._active_lock:
                    if self._active is running:
                        self._active = None
                running.teardown_done.set()
                self._wake()

    def _cleanup_supervisor_handle(self, running: _Running | None) -> str:
        """Execution cleanup on an unexpected supervisor exit; "" only if proven."""

        if running is None or running.handle is None:
            return ""
        handle = running.handle
        try:
            error = self.supervisor.cleanup(handle)
        except Exception as exc:
            error = f"execution cleanup failed: {exc}"[:1000]
        finally:
            try:
                self.supervisor.release(handle)
            except Exception:
                pass
            self._close_handle_streams(handle)
        return error

    @staticmethod
    def _close_handle_streams(handle: WorkerHandle) -> None:
        for stream in (
            getattr(getattr(handle, "process", None), "stdin", None),
            getattr(getattr(handle, "process", None), "stdout", None),
            getattr(getattr(handle, "process", None), "stderr", None),
            getattr(getattr(handle, "attach", None), "stdin", None),
            getattr(getattr(handle, "attach", None), "stdout", None),
            getattr(getattr(handle, "attach", None), "stderr", None),
        ):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    def _cancel_active(self, request_id: str) -> _Running | None:
        """Stop the in-memory worker after a *durable* terminal transition.

        It writes nothing durable (Cancel and revocation go through
        ``_cancel_active_locked`` while holding ``_active_lock``, after their
        own durable commit); it is for callers that already moved the durable
        state and only need the live worker stopped.
        """

        with self._active_lock:
            return self._cancel_active_locked(request_id)

    def _cancel_active_locked(self, request_id: str) -> _Running | None:
        """Mark (and stop) the active worker; caller holds ``_active_lock``."""

        running = self._active
        if running is None or running.request.request_id != request_id:
            return None
        running.cancel_requested = True
        # The writer authorizes under ``request_lock``; take it here so the
        # cancel and the authorization cannot interleave.
        with running.request_lock:
            running.cancel_requested = True
        handle = running.handle
        if handle is not None:
            try:
                self.supervisor.kill(handle)
            except Exception:  # a failed kill is reported by cleanup/quarantine
                pass
        return running

    def _popen(self) -> subprocess.Popen:
        env = scrubbed_env()
        env.update(self.worker_env)
        # Engine-owned, so a caller cannot point the private worker temp root at
        # another attempt's state directory.
        env[STATE_DIR_ENV] = str(self.store.state_dir)
        root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        return popen_group(
            self.worker_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, cwd=str(self.worker_cwd),
        )

    def _handle_worker_job_error(self, running: _Running, exc: JobError) -> None:
        request, attempt = running.request, running.attempt
        domain = exc.failure_domain or ("infrastructure" if exc.code in {"data_loss", "resource_exhausted"} else "protocol")
        code = exc.code if exc.code in {"data_loss", "resource_exhausted"} else "worker_result_invalid"
        failure = JobFailure(domain, exc.retryable, code, str(exc)[:1000])
        if exc.retryable:
            self.store.interrupt_attempt(request.request_id, attempt, failure)
            return
        if exc.code == "data_loss":
            applied = self.store.interrupt_attempt(
                request.request_id, attempt, failure,
                quarantine=True, recovery_error=str(exc),
            )
        else:
            applied = self.store.fail_attempt(request.request_id, attempt, failure)
        if not applied:
            snapshot = self.store._job_by_id(request.request_id)
            if snapshot.state in {"completed", "failed", "cancelled", "expired"}:
                self.store.mark_unavailable(StateCorrupt(
                    f"could not apply non-retryable {exc.code} to terminal job "
                    f"{snapshot.state}: {exc}"
                ))
                self._stop.set()

    def _complete_worker_result(
        self, running: _Running, *, report: dict[str, Any], evidence: dict[str, Any] | None,
    ) -> None:
        request, attempt = running.request, running.attempt
        if self.require_evidence and evidence is None:
            # Legacy evaluator results require candidate-binding evidence.  The
            # policy is enforced before the terminal/quota commit; the external
            # engine default still permits opaque null evidence.
            self._handle_worker_job_error(
                running,
                JobError("invalid_argument", "worker result has no evidence object"),
            )
            return
        try:
            snapshot = self.store.complete_attempt(
                request.request_id, attempt, report=report, evidence=evidence,
                signing_key=self._signing_key, signing_key_id=self._signing_key_id,
            )
        except JobError as exc:
            self._handle_worker_job_error(running, exc)
            return
        # Defensive fence for a cancellation that races the result pre-check.
        # New stores finish this atomically in complete_attempt; this also
        # prevents a still-running response from wedging the single-active queue.
        if snapshot.state in {"queued", "running"} and snapshot.cancel_requested:
            self.store.finish_cancellation(request.request_id, attempt)

    def _supervisor_daemon(self) -> tuple[str, str]:
        """Return ``(error, daemon_identity)`` for the current execution engine."""

        try:
            daemon = self.supervisor.daemon_identity()
        except Exception as exc:
            return f"could not verify the Docker daemon identity: {exc}"[:1000], ""
        if not daemon:
            return "the execution engine reported no daemon identity", ""
        return "", daemon

    def _cleanup_uncreated_worker(self, attempt: Attempt) -> str:
        """Clean a worker whose creation failed before any id was durable."""

        if self.supervisor.kind != EXECUTION_CONTAINER:
            # The process supervisor spawns synchronously: a raised exception
            # did not leave a registered process behind (the generic spawn
            # path registers only after Popen succeeds).
            return ""
        daemon_error, daemon = self._supervisor_daemon()
        try:
            return self.supervisor.cleanup_recovered(
                attempt, state_dir=self.store.state_dir, daemon=daemon,
            ) or daemon_error
        except Exception as exc:
            return f"execution cleanup failed: {exc}"[:1000]

    def _accepted_identity_error(self, snapshot: JobSnapshot) -> str:
        """Refuse a queued job accepted by a different evaluator identity.

        The job's durable ``expected`` identities (site/device/target,
        worker_build, task pack digest, evaluator epoch, API/evidence contract)
        are what the signed result will claim, and ``provenance`` records the
        accepting frontend's full identity.  Executing the job under any other
        build - another process build, another image, or the same image at a
        different epoch - would sign a result that names an evaluator which
        never ran it.  Both planes must match completely.
        """

        accepted = snapshot.expected
        if accepted is not None and accepted.to_dict() == self._expected.to_dict():
            accepted_build = str((snapshot.provenance or {}).get("worker_build") or "")
            if accepted_build == self.identity.worker_build:
                return ""
            return (
                f"queued job provenance records worker_build {accepted_build!r} but this "
                f"frontend is {self.identity.worker_build!r}; refusing to sign a result for "
                "another build"
            )
        durable = accepted.to_dict() if accepted is not None else None
        return (
            "queued job was accepted by a different evaluator identity "
            f"({durable!r} != {self._expected.to_dict()!r}); refusing to execute it here"
        )

    def _cleanup_handle_quietly(self, handle: WorkerHandle, *, close_streams: bool = True) -> str:
        """Kill, delete and verify a worker that must not receive a request."""

        try:
            error = self.supervisor.cleanup(handle)
        except Exception as exc:
            error = f"execution cleanup failed: {exc}"[:1000]
        if not error:
            try:
                error = self.supervisor.cleanup_residue(handle)
            except Exception as exc:
                error = f"execution residue check failed: {exc}"[:1000]
        try:
            self.supervisor.release(handle)
        except Exception:
            pass
        if close_streams:
            self._close_handle_streams(handle)
        return error

    def _record_cleanup_hazard(self, running: _Running, error: str) -> None:
        """Fail closed: no result commit and no Retry while cleanup is unproven."""

        request, attempt = running.request, running.attempt
        snapshot = self.store._job_by_id(request.request_id)
        current = snapshot.current_attempt
        if (
            snapshot.state == "running" and current is not None
            and current.attempt_id == attempt.attempt_id
            and current.lease_fence == attempt.lease_fence
        ):
            self.store.recovered_attempt(
                request.request_id, attempt, cleanup_error=error[:1000],
            )
            return
        self.store.record_cleanup_hazard(request.request_id, attempt, error[:1000])

    def _abort_worker_before_request(
        self, running: _Running, handle: WorkerHandle, failure: JobFailure,
        *, worker_attached: bool,
    ) -> None:
        request, attempt = running.request, running.attempt
        cleanup_error = self._cleanup_handle_quietly(handle)
        snapshot = self.store._job_by_id(request.request_id)
        if cleanup_error:
            recovery_error = f"{failure.message}; {cleanup_error}"[:1000]
            if handle.kind == EXECUTION_CONTAINER:
                # The container has a durable identity (id and/or deterministic
                # name), so operator release can recheck it; quarantine rather
                # than taking the whole frontend down.
                self._record_cleanup_hazard(running, recovery_error)
                return
            if not worker_attached:
                # No durable PID/start token was confirmed, so quarantine
                # release could mistake the live worker for a no-request
                # recovery and retry while the old process is still alive.
                self.store.mark_unavailable(StateCorrupt(
                    f"uncleaned unattached worker has no durable process identity: {recovery_error}"
                ))
                self._stop.set()
                return
            current = snapshot.current_attempt
            if (
                snapshot.state == "running" and current is not None
                and current.attempt_id == attempt.attempt_id
                and current.lease_fence == attempt.lease_fence
            ):
                # Keep the attempt running and quarantined so operator release
                # must recheck the persisted process identity before retry/cancel.
                self.store.recovered_attempt(
                    request.request_id, attempt, cleanup_error=recovery_error,
                )
            else:
                self.store.mark_unavailable(StateCorrupt(
                    "uncleaned worker could not be bound to a running durable attempt"
                ))
                self._stop.set()
            return
        if snapshot.cancel_requested or running.cancel_requested:
            self.store.finish_cancellation(request.request_id, attempt)
            return
        self.store.interrupt_attempt(request.request_id, attempt, failure)

    def _run_worker(self, running: _Running) -> None:
        request, attempt = running.request, running.attempt
        if running.stopping or self._stop.is_set():
            self.store.interrupt_attempt(request.request_id, attempt, JobFailure(
                "infrastructure", True, "frontend_stopped", "frontend stopped before worker startup",
            ))
            return
        if running.cancel_requested:
            self.store.finish_cancellation(request.request_id, attempt)
            return
        identity_error = self._accepted_identity_error(self.store._job_by_id(request.request_id))
        if identity_error:
            # The accepted job carries the submitting evaluator's identity.
            # Running it under any other build would sign a result that names
            # an evaluator/image which never ran it.
            self.store.interrupt_attempt(
                request.request_id, attempt,
                JobFailure("infrastructure", False, "worker_build_mismatch", identity_error[:1000]),
                quarantine=True, recovery_error=identity_error[:1000],
            )
            return
        if self.supervisor.kind == EXECUTION_CONTAINER:
            # Bind the attempt to this Docker daemon before any container can
            # exist, so a crash between create and the id attach is recoverable
            # only against the same engine (and a daemon switch fails closed).
            daemon_error, daemon = self._supervisor_daemon()
            if daemon_error:
                self.store.interrupt_attempt(request.request_id, attempt, JobFailure(
                    "infrastructure", True, "worker_daemon_identity", daemon_error[:1000],
                ))
                return
            if not self.store.attach_daemon(request.request_id, attempt, daemon=daemon):
                # No container exists yet; the durable fence moved on.
                self.store.interrupt_attempt(request.request_id, attempt, JobFailure(
                    "infrastructure", True, "worker_attach_fenced",
                    "the Docker daemon identity was not attached to the exact running attempt/fence",
                ))
                return
            attempt = replace(attempt, container_daemon=daemon)
        spec = AttemptSpec(
            request=request, attempt=attempt, state_dir=self.store.state_dir,
            command=tuple(self.worker_command), env=dict(self.worker_env),
        )
        try:
            handle = self.supervisor.create(spec)
        except Exception as exc:
            # A create failure is ambiguous by construction (the daemon may have
            # created the container while the CLI failed, timed out or raised),
            # so any exception - not only the typed ones - must be followed by a
            # durable-identity cleanup.
            residue_error = self._cleanup_uncreated_worker(attempt)
            if residue_error:
                self._record_cleanup_hazard(
                    running,
                    f"could not start worker: {exc}; {residue_error}"[:1000],
                )
            elif self.store._job_by_id(request.request_id).cancel_requested or running.cancel_requested:
                self.store.finish_cancellation(request.request_id, attempt)
            else:
                self.store.interrupt_attempt(request.request_id, attempt, JobFailure(
                    "infrastructure", True, "worker_unavailable", f"could not start worker: {exc}"[:1000],
                ))
            return
        running.handle = handle
        if self._stop.is_set() or running.stopping:
            self._abort_worker_before_request(running, handle, JobFailure(
                "infrastructure", True, "frontend_stopped", "frontend stopped during worker startup",
            ), worker_attached=False)
            return
        if handle.kind == EXECUTION_PROCESS:
            process = handle.process
            if process is None:
                cleanup_error = self._cleanup_handle_quietly(handle)
                if cleanup_error:
                    self._record_cleanup_hazard(
                        running, f"process supervisor returned no worker process; {cleanup_error}"[:1000],
                    )
                    return
                self.store.interrupt_attempt(request.request_id, attempt, JobFailure(
                    "infrastructure", True, "worker_unavailable",
                    "process supervisor returned no worker process",
                ))
                return
            # A raised identity probe propagates to the scheduler's generic
            # supervisor-failure path, which cleans the unattached worker.
            token = process_start_token(process.pid, attempts=3)
            if token is None:
                # The process exists but has no durable start-token identity:
                # unless the whole group is proven dead, Retry must be refused.
                cleanup_error = self._cleanup_handle_quietly(handle)
                if cleanup_error:
                    self._record_cleanup_hazard(
                        running,
                        f"could not bind worker PID start token; {cleanup_error}"[:1000],
                    )
                elif self.store._job_by_id(request.request_id).cancel_requested or running.cancel_requested:
                    self.store.finish_cancellation(request.request_id, attempt)
                else:
                    self.store.interrupt_attempt(request.request_id, attempt, JobFailure(
                        "infrastructure", True, "worker_identity_unavailable",
                        "could not bind worker PID start token",
                    ))
                return
            attached_ok = self.store.attach_worker(
                request.request_id, attempt,
                pid=process.pid, pgid=process.pid, start_token=token,
            )
        else:
            identity_error = self.supervisor.verify_identity(handle)
            if identity_error:
                cleanup_error = self._cleanup_handle_quietly(handle)
                detail = identity_error if not cleanup_error else f"{identity_error}; {cleanup_error}"
                failure = JobFailure(
                    "infrastructure", True, "worker_container_identity", detail[:1000],
                )
                if cleanup_error:
                    self._record_cleanup_hazard(running, f"{failure.message}; {cleanup_error}"[:1000])
                elif self.store._job_by_id(request.request_id).cancel_requested or running.cancel_requested:
                    self.store.finish_cancellation(request.request_id, attempt)
                else:
                    self.store.interrupt_attempt(request.request_id, attempt, failure)
                return
            daemon_error, daemon = self._supervisor_daemon()
            if daemon_error:
                cleanup_error = self._cleanup_handle_quietly(handle)
                detail = daemon_error if not cleanup_error else f"{daemon_error}; {cleanup_error}"
                if cleanup_error:
                    self._record_cleanup_hazard(running, detail[:1000])
                else:
                    self.store.interrupt_attempt(request.request_id, attempt, JobFailure(
                        "infrastructure", True, "worker_daemon_identity", detail[:1000],
                    ))
                return
            attached_ok = self.store.attach_container(
                request.request_id, attempt,
                container_id=handle.container_id, container_name=handle.container_name,
                image_ref=handle.image_ref, image_digest=handle.image_digest,
                runtime=handle.runtime, daemon=daemon,
            )
        if not attached_ok:
            self._abort_worker_before_request(running, handle, JobFailure(
                "infrastructure", True, "worker_attach_fenced",
                "worker identity was not attached to the exact running attempt/fence",
            ), worker_attached=False)
            return
        # Attach is another durable boundary.  Re-read cancellation before the
        # request writer can let a just-claimed worker run the whole candidate.
        attached = self.store._job_by_id(request.request_id)
        if attached.cancel_requested or running.cancel_requested:
            running.cancel_requested = True
            self._abort_worker_before_request(running, handle, JobFailure(
                "infrastructure", True, "worker_cleanup_failed",
                "worker cleanup after cancellation failed",
            ), worker_attached=True)
            return
        current = attached.current_attempt
        if (
            attached.state != "running" or current is None
            or current.attempt_id != attempt.attempt_id
            or current.lease_fence != attempt.lease_fence
        ):
            self._abort_worker_before_request(running, handle, JobFailure(
                "infrastructure", True, "worker_attach_fenced",
                "durable attempt/fence changed after worker attach",
            ), worker_attached=True)
            return
        if self._stop.is_set() or running.stopping:
            self._abort_worker_before_request(running, handle, JobFailure(
                "infrastructure", True, "frontend_stopped", "frontend stopped before worker request",
            ), worker_attached=True)
            return
        # The request is only written after the worker is durably attached and
        # started, so a crash can never hide an executing container.  The final
        # cancellation re-read happens under the same lock the Cancel RPC takes,
        # so a cancellation that wins the lock can never be followed by starting
        # this worker.
        start_error: BaseException | None = None
        cancelled_before_start = False
        # The Cancel RPC commits the durable cancellation while holding
        # ``_active_lock``; reading the durable state and starting the worker
        # under the same lock makes "cancelled before start" impossible to miss.
        with self._active_lock:
            before_start = self.store._job_by_id(request.request_id)
            cancelled_before_start = bool(before_start.cancel_requested or running.cancel_requested)
            if not cancelled_before_start:
                try:
                    self.supervisor.start(handle)
                    stdin, stdout, stderr = self.supervisor.streams(handle)
                except Exception as exc:
                    start_error = exc
        if cancelled_before_start:
            running.cancel_requested = True
            self._abort_worker_before_request(running, handle, JobFailure(
                "infrastructure", True, "worker_cleanup_failed",
                "worker cleanup after cancellation failed",
            ), worker_attached=True)
            return
        if start_error is not None:
            self._abort_worker_before_request(running, handle, JobFailure(
                "infrastructure", True, "worker_start_failed",
                f"could not start worker: {start_error}"[:1000],
            ), worker_attached=True)
            return
        stdout_error: list[BaseException] = []
        supervisor_error: list[BaseException] = []
        results: list[dict[str, Any]] = []

        def read_stdout() -> None:
            decoder = FrameDecoder(max_payload=MAX_WORKER_FRAME_PAYLOAD)
            try:
                read_chunk = getattr(stdout, "read1", stdout.read)
                while True:
                    chunk = read_chunk(65_536)
                    if not chunk:
                        break
                    for frame in decoder.feed(chunk):
                        kind, payload = decode_worker_frame(frame, attempt, request_id=request.request_id)
                        if kind == "progress":
                            self.store.record_progress(request.request_id, attempt, payload)
                        else:
                            if results:
                                raise JobError("invalid_argument", "worker emitted multiple results")
                            results.append(payload)
                decoder.finish()
            except BaseException as exc:  # propagated in the supervisor thread
                stdout_error.append(exc)

        stderr_tail: list[str] = []

        def read_stderr() -> None:
            tail = bytearray()
            while True:
                chunk = stderr.read(4096)
                if not chunk:
                    break
                tail = (tail + chunk)[-MAX_STDERR:]
            stderr_tail.append(tail.decode("utf-8", errors="replace"))

        out_thread = threading.Thread(target=read_stdout, name="tcenv-worker-stdout", daemon=True)
        err_thread = threading.Thread(target=read_stderr, name="tcenv-worker-stderr", daemon=True)
        out_thread.start()
        err_thread.start()
        timed_out = False
        returncode: int | None = None
        write_error: list[BaseException] = []

        def write_request() -> None:
            try:
                # The authorization and the cancellation are mutually exclusive:
                # whichever takes ``request_lock`` first wins.  A cancel that
                # wins means the worker never receives candidate bytes.
                with running.request_lock:
                    if self._stop.is_set() or running.stopping or running.cancel_requested:
                        return
                    running.request_authorized = True
                assert stdin is not None
                stdin.write(encode_request(request, attempt))
                stdin.flush()
                stdin.close()
            except BaseException as exc:  # broken pipe is handled by the supervisor
                write_error.append(exc)

        writer_thread = threading.Thread(target=write_request, name="tcenv-worker-stdin", daemon=True)
        writer_thread.start()
        try:
            if running.cancel_requested:
                self.supervisor.kill(handle)
            try:
                # The request writer is concurrent: a broken worker that never
                # reads stdin still cannot escape the candidate deadline.
                remaining = request.candidate_timeout_ms / 1000.0 - (time.monotonic() - running.started_monotonic)
                returncode = self.supervisor.wait(handle, max(0.001, remaining))
                timed_out = returncode is None
            except (BrokenPipeError, OSError) as exc:
                stdout_error.append(exc)
            except SupervisorError as exc:
                supervisor_error.append(exc)
        finally:
            writer_thread.join(timeout=2)
            if write_error and not running.cancel_requested and not timed_out and not supervisor_error:
                stdout_error.append(write_error[0])
            # Kill then delete then verify, on every path (timeout, cancel and
            # a normal exit): a worker process/container must be provably gone
            # before any result may be committed.  The private streams are
            # closed only after the readers reached EOF, so a reader is never
            # turned into a spurious protocol error by the cleanup itself.
            cleanup_error = self._cleanup_handle_quietly(handle, close_streams=False)
            out_thread.join(timeout=2)
            err_thread.join(timeout=2)
            reader_alive = out_thread.is_alive()
            self._close_handle_streams(handle)

        if cleanup_error:
            self._record_cleanup_hazard(running, cleanup_error)
            return
        if stdout_error:
            error = stdout_error[0]
            if isinstance(error, JobError):
                self._handle_worker_job_error(running, error)
            else:
                self.store.interrupt_attempt(request.request_id, attempt, JobFailure(
                    "protocol", True, "worker_protocol_error", str(error)[:1000],
                ))
            return
        if reader_alive:
            # The stdout reader never reached EOF/decoder.finish(): frames may
            # still be unvalidated, so no result may be committed.
            self.store.interrupt_attempt(request.request_id, attempt, JobFailure(
                "protocol", True, "worker_protocol_error",
                "worker output was not fully read and validated before the result decision",
            ))
            return
        if supervisor_error:
            error = supervisor_error[0]
            self.store.interrupt_attempt(request.request_id, attempt, JobFailure(
                "infrastructure", True, "worker_supervisor_failed", str(error)[:1000],
            ))
            return
        snapshot = self.store._job_by_id(request.request_id)
        if snapshot.cancel_requested or running.cancel_requested:
            self.store.finish_cancellation(request.request_id, attempt)
            return
        if running.stopping or self._stop.is_set():
            self.store.interrupt_attempt(request.request_id, attempt, JobFailure(
                "infrastructure", True, "frontend_stopped", "frontend stopped while the attempt was active",
            ))
            return
        if not results:
            if timed_out:
                from tcenv.evaluate import EvalReport

                report = EvalReport(
                    task=request.task, dtype=request.dtype, size=request.size, mode=request.mode,
                    status="timeout", stage="timeout", message="candidate exceeded the durable execution deadline",
                    seed=request.seed_int or 0, protocol_version=self.execution_protocol_version,
                    target_id=request.target_id, backend=self.identity.backend,
                    backend_version=self.identity.backend_version,
                    elapsed_s=request.candidate_timeout_ms / 1000.0,
                ).to_json()
                if self.identity.measurement_kind == "simulated":
                    report["simulated"] = True
                self._complete_worker_result(
                    running, report=report,
                    evidence={"candidate_sha256": request.candidate_sha256},
                )
                return
            if returncode != 0:
                detail = stderr_tail[0].replace(str(self.store.state_dir), "<state>") if stderr_tail else ""
                self.store.interrupt_attempt(request.request_id, attempt, JobFailure(
                    "infrastructure", True, "worker_failed", f"worker exited {returncode}: {detail}"[:1000],
                ))
                return
            self.store.interrupt_attempt(request.request_id, attempt, JobFailure(
                "protocol", True, "worker_protocol_error", "worker did not emit exactly one result",
            ))
            return
        # A complete result was decoded and ``decoder.finish()`` validated the
        # stream (no malformed frame, no second result, reader reached EOF).
        # A worker that then also exited non-zero is still a failure: the result
        # must not be trusted over a crashed supervisor exit.
        if len(results) != 1:
            self.store.interrupt_attempt(request.request_id, attempt, JobFailure(
                "protocol", True, "worker_protocol_error", "worker emitted multiple results",
            ))
            return
        if returncode not in (0, None):
            detail = stderr_tail[0].replace(str(self.store.state_dir), "<state>") if stderr_tail else ""
            self.store.interrupt_attempt(request.request_id, attempt, JobFailure(
                "infrastructure", True, "worker_failed", f"worker exited {returncode}: {detail}"[:1000],
            ))
            return
        self._complete_worker_result(
            running, report=results[0]["report"], evidence=results[0]["evidence"],
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        with self._active_lock:
            running = self._active
            if running is not None:
                running.stopping = True
                handle = running.handle
            else:
                handle = None
        if handle is not None:
            try:
                self.supervisor.kill(handle)
            except (OSError, SupervisorError):
                pass
        self._wake()
        scheduler = getattr(self, "_scheduler", None)
        if scheduler is not None:
            scheduler.join(timeout=_CLOSE_SCHEDULER_JOIN_S)
            if scheduler.is_alive():
                threading.Thread(
                    target=self._close_store_after_scheduler,
                    args=(scheduler, self.store),
                    name="tcenv-external-jobs-close", daemon=True,
                ).start()
                return
        self.store.close()

    @staticmethod
    def _close_store_after_scheduler(scheduler: threading.Thread, store: JobStore) -> None:
        scheduler.join()
        store.close()

    def __enter__(self) -> "JobEngine":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
