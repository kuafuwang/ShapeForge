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
from typing import Any, Callable, Iterable, Mapping

from tcenv.children import forget, kill_group, popen_group, scrubbed_env
from tcenv.schema import PROTOCOL_VERSION as EXECUTION_PROTOCOL_VERSION

from . import contracts
from .blob_channel import BlobError, BlobScope, BlobStore
from .contracts import API_CONTRACT_ID, EVIDENCE_CONTRACT_ID
from .drivers.base import CleanupProof, DeviceController, HealthProof, VendorFacts
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
    NpuExecutionSpec,
    StateCorrupt,
)
from .protocol import FrameDecoder
from .supervisor import (
    AttemptSpec,
    DaemonChanged,
    AttemptSupervisor,
    ProcessAttemptSupervisor,
    StartRefused,
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
# Upper bound on one supervisor wait slice: the candidate deadline is re-checked
# at least this often, so a late exit is classified against the real clock.
_DEADLINE_SLICE_S = 1.0
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
    # Set under ``_active_lock`` when the writer is allowed to send the
    # request; the same lock spans the durable cancel commit, so a durable
    # cancellation can never be followed by a newly authorized request.
    request_authorized: bool = False


class _StartGate:
    """Hold ``_active_lock`` across the final authorization and the spawn.

    ``__enter__`` re-checks stop/cancel/deadline while holding the engine lock
    and refuses (raising :class:`StartRefused`) without spawning.  The lock is
    released only after ``popen_group`` returned, so a durable revocation or
    cancel cannot commit in the gap between the check and the start.  The
    blocking daemon bind happens before this gate, so Cancel/close keep their
    bounded returns.
    """

    def __init__(self, engine: "JobEngine", running: "_Running") -> None:
        self._engine = engine
        self._running = running
        self._entered = False

    def __enter__(self) -> "_StartGate":
        self._engine._active_lock.acquire()
        try:
            self.check()
        except BaseException:
            self._engine._active_lock.release()
            raise
        self._entered = True
        return self

    def check(self) -> None:
        """Re-check the state immediately before the real spawn.

        Called by the supervisor after any pre-spawn gate and before the actual
        Popen; the engine lock is still held, but a Cancel timeout branch may
        have set ``cancel_requested`` without it, so this read is the last
        chance to refuse a start that would follow a durable cancellation.
        """

        if self._engine._stop.is_set() or self._running.stopping:
            raise StartRefused("stopped")
        if self._running.cancel_requested:
            raise StartRefused("cancelled")
        # The Cancel timeout branch commits durably before it can set the
        # in-memory flag; re-read the durable state so a commit that landed
        # between those two steps still refuses the start.
        durable = self._engine.store._job_by_id(self._running.request.request_id)
        if durable.cancel_requested:
            raise StartRefused("cancelled")
        if self._engine._overdue(self._running):
            raise StartRefused("deadline")

    def __exit__(self, *_exc: object) -> None:
        if self._entered:
            self._entered = False
            self._engine._active_lock.release()


#: Blob-channel refusals keep their classification when they surface as job errors.
_BLOB_JOB_ERRORS = {
    "not_found": "not_found",
    "permission_denied": "permission_denied",
    "failed_precondition": "failed_precondition",
    "resource_exhausted": "resource_exhausted",
    "out_of_range": "out_of_range",
    "unavailable": "unavailable",
}


#: Private worker variable naming the engine-owned blob store root.
BLOB_ROOT_ENV = "TCENV_BLOB_ROOT"


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
        monotonic: Callable[[], float] | None = None,
        contract: contracts.ContractDialect = contracts.LEGACY_DIALECT,
        npu_execution: NpuExecutionSpec | None = None,
        device_controller: DeviceController | None = None,
        device_call_timeout_s: float = 2.0,
        blob_store: Any | None = None,
        **store_options: Any,
    ) -> None:
        if not worker_command or not all(isinstance(part, str) and part for part in worker_command):
            raise ValueError("worker_command must be a non-empty argv list")
        if not isinstance(execution_protocol_version, str) or not execution_protocol_version:
            raise ValueError("execution_protocol_version must be non-empty")
        if not isinstance(require_evidence, bool):
            raise ValueError("require_evidence must be boolean")
        if not isinstance(contract, contracts.ContractDialect):
            raise ValueError("contract must be a ContractDialect")
        if contract.npu_source:
            if not isinstance(npu_execution, NpuExecutionSpec):
                raise ValueError("the NPU contract requires a typed execution binding")
        elif npu_execution is not None:
            raise ValueError("the legacy contract cannot carry an NPU execution binding")
        if contract.npu_source and device_controller is None:
            # Cancelling and reconciling device resources is worker-external by
            # construction: an NPU frontend without a controller could never
            # prove a cleanup, so it is a composition error, not a degraded mode.
            raise ValueError("the NPU contract requires a worker-external device controller")
        if device_controller is not None and not contract.npu_source:
            raise ValueError("a device controller is only valid with the NPU contract")
        if contract.npu_blob:
            if not isinstance(blob_store, BlobStore):
                # Without the data plane a blob-reference job could never be
                # verified or dispatched: a composition error, not a degraded mode.
                raise ValueError("the NPU blob contract requires a blob channel store")
        elif blob_store is not None:
            raise ValueError("a blob channel store is only valid with the NPU blob contract")
        if (
            isinstance(device_call_timeout_s, bool)
            or not isinstance(device_call_timeout_s, (int, float))
            or not math.isfinite(device_call_timeout_s)
            or not 0.05 <= device_call_timeout_s <= 30.0
        ):
            raise ValueError("device_call_timeout_s must be a finite number between 0.05 and 30 seconds")
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
        # Monotonic clock seam.  Production always uses ``time.monotonic``; the
        # deadline tests advance a controlled clock instead of widening the
        # candidate budget until a real-time race "usually" goes their way.
        self._monotonic: Callable[[], float] = monotonic or time.monotonic
        if (
            isinstance(cancel_teardown_s, bool)
            or not isinstance(cancel_teardown_s, (int, float))
            or not 0 < float(cancel_teardown_s) <= 120
        ):
            raise ValueError("cancel_teardown_s must be a positive bounded number of seconds")
        self.cancel_teardown_s = float(cancel_teardown_s)
        self.identity = identity
        self.contract = contract
        self.npu_execution = npu_execution
        self.device_controller = device_controller
        self.device_call_timeout_s = float(device_call_timeout_s)
        self.blob_store: BlobStore | None = blob_store
        # Device-operation state: an attempt may not start while a controller
        # call is still in flight (including a call the RPC already timed out
        # on), and a probe may not start while an attempt owns the group.
        self._device_condition = threading.Condition()
        self._device_inflight = 0
        self._device_attempt_active = False
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
        # Shutdown state: ``_close_lock`` serializes close() so a second caller
        # cannot be told "closed" while the first is still tearing down.
        self._close_lock = threading.Lock()
        self._close_started = False
        self._close_error: BaseException | None = None
        self._closed_event = threading.Event()
        self._unreported_hazard = ""
        self._expected = ExpectedIdentities(ExpectedEvaluator(
            site_id=identity.site_id, device_id=identity.device_id, target_id=identity.target_id,
            worker_build=identity.worker_build, task_pack_digest=identity.task_pack_digest,
            evaluator_epoch=identity.external_epoch(execution_protocol_version=execution_protocol_version),
        ), contract.api_contract_id, contract.evidence_contract_id)
        self._expected.validate()
        if len(identity.calibration_digest) != 64 or any(c not in "0123456789abcdef" for c in identity.calibration_digest):
            raise ValueError("external evaluator calibration_digest must be 64 lowercase hex characters")
        self.store = JobStore(
            state_dir, signing_key=signing_key, signing_key_id=signing_key_id,
            signing_keyring=signing_keyring, contract=contract, **store_options,
        )
        if (
            self.blob_store is not None
            and self.blob_store.root.resolve() != (self.store.state_dir / "blobs").resolve()
        ):
            # The worker-side blob root is injected by this engine, so the store
            # must live inside the frontend's own state directory.  A caller
            # supplied path would turn the trusted read into an arbitrary file read.
            raise ValueError("the blob channel store must live in <state_dir>/blobs")
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
            api_contract_id=self.contract.api_contract_id,
            evidence_contract_id=self.contract.evidence_contract_id,
            npu_execution=(
                None if self.npu_execution is None else self.npu_execution.to_dict()
            ),
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
        if self.contract.npu_source:
            # Authenticate before any device operation: probe/health are device
            # calls and must not run for a rejected grant.  This pre-check is
            # read-only (no nonce use, no capability row); the durable write
            # still happens in store.get_capabilities below.
            self.store.require_grant(grant_token)
        # Probe the trusted device: a group that cannot be verified must not
        # consume the client nonce or leave a durable capability behind.  The
        # whole sequence holds one device-group slot, and each controller call
        # adds a reference that outlives a timed-out RPC, so a candidate can
        # never start while any part of the probe is still touching the device.
        device_target: dict[str, Any] = {}
        if self.contract.npu_source:
            self._begin_device_operation()
            try:
                device_target = self._device_capability()
            finally:
                self._end_device_operation()
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
        target = {
            "target_id": self.identity.target_id,
            "backend": self.identity.backend,
            "modes": sorted(self.supported_modes),
            "features": sorted(self.supported_features),
        }
        target.update(device_target)
        return replace(capabilities, target_capabilities=(target,))

    # -- trusted device facts ----------------------------------------------
    def _device_profile(self) -> dict[str, str]:
        if self.npu_execution is None:
            raise JobError("failed_precondition", "this evaluator has no NPU device binding")
        return self.npu_execution.device_group.to_dict()

    def _begin_device_operation(self, *, nested: bool = False) -> None:
        """Claim the device group for one controller call, or refuse.

        ``nested`` is used inside an already-claimed probe sequence: it only adds
        a reference (so a timed-out call keeps the group claimed until it really
        returns) without the exclusive refusal.
        """

        with self._device_condition:
            if not nested and (self._device_inflight or self._device_attempt_active):
                raise JobError(
                    "failed_precondition",
                    "the device group is executing an attempt or a device probe is "
                    "still in flight; GetCapabilities will not probe it",
                )
            self._device_inflight += 1

    def _end_device_operation(self) -> None:
        with self._device_condition:
            self._device_inflight = max(0, self._device_inflight - 1)
            self._device_condition.notify_all()

    def _bounded_device_call(
        self, name: str, call: Callable[[], Any], *, owner: bool = False,
        nested: bool = False,
    ) -> tuple[Any, str]:
        """Run one trusted controller call with a bounded wall-clock wait.

        The device stays marked in-flight until the underlying call really
        returns: a timeout only bounds the RPC, it does not let a candidate start
        while the controller is still touching the device.  The worker thread
        therefore owns the release, and the caller never releases early.

        ``owner=True`` marks the call as the active attempt's own cleanup
        (cancel/reconcile), which the attempt already owns; it must not take the
        probe slot and must not be refused by it.
        """

        outcome: dict[str, Any] = {}
        done = threading.Event()

        def run() -> None:
            try:
                outcome["value"] = call()
            except BaseException as exc:  # noqa: BLE001 - a controller failure is infrastructure
                outcome["error"] = f"{name} raised {type(exc).__name__}: {exc}"[:500]
            finally:
                if not owner:
                    self._end_device_operation()
                done.set()

        if not owner:
            self._begin_device_operation(nested=nested)
        try:
            thread = threading.Thread(target=run, name=f"device-{name}", daemon=True)
            thread.start()
        except BaseException:
            if not owner:
                self._end_device_operation()
            raise
        if not done.wait(self.device_call_timeout_s):
            return None, f"{name} did not return within {self.device_call_timeout_s:.3f}s"
        if "error" in outcome:
            return None, str(outcome["error"])
        return outcome.get("value"), ""

    def _check_device_binding(
        self, value: Mapping[str, Any], *, where: str, keys: tuple[str, ...] | None = None,
    ) -> None:
        binding = self._device_profile()
        for key in (keys if keys is not None else tuple(binding)):
            expected = binding[key]
            if value.get(key) != expected:
                raise JobError(
                    "failed_precondition",
                    f"{where} reports {key}={value.get(key)!r}, this evaluator serves {expected!r}",
                )

    def _device_capability(self) -> dict[str, Any]:
        """Build the P0 capability from trusted probed facts, or fail closed.

        A device group that cannot be probed, is not healthy, or a controller
        that names another profile/topology/resource generation must not be
        advertised as available: the client would otherwise submit into a group
        this frontend cannot cancel or reconcile.
        """

        controller = self.device_controller
        if controller is None:
            raise JobError("failed_precondition", "no device controller is configured")
        profile = self._device_profile()
        facts, error = self._bounded_device_call(
            "probe", lambda: controller.probe(profile), nested=True,
        )
        if error:
            raise JobError("failed_precondition", f"device probe failed: {error}")
        if not isinstance(facts, VendorFacts):
            raise JobError("failed_precondition", "device controller returned untyped probe facts")
        if facts.state != "available":
            raise JobError("failed_precondition", f"device group is {facts.state!r}, not available")
        if isinstance(facts.logical_devices, bool) or facts.logical_devices != 1:
            raise JobError(
                "failed_precondition",
                f"P0 serves exactly one logical device; the probe reports {facts.logical_devices!r}",
            )
        if tuple(facts.timing_sources) != tuple(self.npu_execution.required_timing_sources if self.npu_execution else ()):
            raise JobError(
                "failed_precondition",
                "the probe does not report the served DEVICE_EVENT timing source",
            )
        self._check_device_binding(
            {
                "device_group_id": facts.device_group_id,
                "vendor_id": facts.vendor_id,
                "profile_digest": facts.profile_digest,
                "topology_digest": facts.topology_digest,
                "resource_generation": facts.resource_generation,
            },
            where="device probe",
        )
        health, error = self._bounded_device_call(
            "health_check", lambda: controller.health_check(profile), nested=True,
        )
        if error:
            raise JobError("failed_precondition", f"device health check failed: {error}")
        if not isinstance(health, HealthProof):
            raise JobError("failed_precondition", "device controller returned an untyped health proof")
        if not health.ok:
            raise JobError("failed_precondition", f"device known-good probe failed: {health.detail}"[:500])
        if not health.executed_known_good:
            # "ok" without running the known-good probe proves nothing; the
            # group must not be advertised as available.
            raise JobError(
                "failed_precondition",
                "device health check did not execute the known-good probe",
            )
        self._check_device_binding(
            {"device_group_id": health.device_group_id, "resource_generation": health.resource_generation},
            where="device health proof", keys=("device_group_id", "resource_generation"),
        )
        advertised, error = self._bounded_device_call(
            "capabilities", lambda: controller.capabilities(facts), nested=True,
        )
        if error:
            raise JobError("failed_precondition", f"device capabilities failed: {error}")
        if not isinstance(advertised, Mapping):
            raise JobError("failed_precondition", "device controller returned untyped capabilities")
        execution = self.npu_execution
        if execution is None:
            raise JobError("failed_precondition", "this evaluator has no NPU execution binding")
        artifact_kinds = list(advertised.get("artifact_kinds") or ())
        launch_kinds = list(advertised.get("launch_kinds") or ())
        launch_subtypes = list(advertised.get("launch_subtypes") or ())
        timing_sources = list(advertised.get("timing_sources") or ())
        if artifact_kinds != ["source"]:
            raise JobError("failed_precondition", "device does not advertise the source artifact kind")
        if launch_kinds != [execution.launch_kind] or launch_subtypes != [execution.launch_subtype]:
            raise JobError("failed_precondition", "device does not advertise the served launch kind/subtype")
        if timing_sources != list(execution.required_timing_sources):
            raise JobError("failed_precondition", "device does not advertise the served timing source")
        groups = advertised.get("device_groups")
        if not isinstance(groups, (list, tuple)) or len(groups) != 1:
            raise JobError("failed_precondition", "device must advertise exactly one logical device group")
        group = dict(groups[0]) if isinstance(groups[0], Mapping) else None
        if group is None:
            raise JobError("failed_precondition", "device advertised an untyped device group")
        self._check_device_binding(group, where="device capability")
        if group.get("state") != "available":
            raise JobError("failed_precondition", f"device group is {group.get('state')!r}, not available")
        # The group's own timing and cardinality must match the advertised shape;
        # a group that reports HOST_WALL or zero devices contradicts P0.
        group_timing = list(group.get("timing_sources") or ())
        if group_timing != list(execution.required_timing_sources):
            raise JobError(
                "failed_precondition",
                "device group does not advertise the served DEVICE_EVENT timing source",
            )
        logical_devices = group.get("logical_devices")
        if isinstance(logical_devices, bool) or not isinstance(logical_devices, int) or logical_devices != 1:
            raise JobError(
                "failed_precondition",
                "P0 serves exactly one logical device; the device group reports "
                f"{logical_devices!r}",
            )
        return {
            "artifact_kinds": artifact_kinds,
            "launch_kinds": launch_kinds,
            "launch_subtypes": launch_subtypes,
            "timing_sources": timing_sources,
            "device_groups": [{
                **group,
                "logical_devices": 1,
                "timing_sources": list(execution.required_timing_sources),
            }],
        }

    # -- blob data plane (P0b) ---------------------------------------------
    def _blob_scope(self, grant_token: str) -> BlobScope:
        """Bind the blob scope to this frontend's identity plus the live grant."""

        scope = self.store.blob_scope(grant_token)
        return BlobScope(
            site_id=self.identity.site_id, device_id=self.identity.device_id,
            target_id=scope["target_id"], subject=scope["subject"],
            run_id=scope["run_id"], session_id=scope["session_id"],
        )

    def _lease_blob(self, request: ExternalEvaluationRequest, grant_token: str) -> bool:
        """Require a ready, in-scope blob before anything durable is written.

        An unfinished upload, another scope's content id, or a descriptor that
        disagrees with the bytes on disk can never be submitted as a complete
        artifact.  The lease is what later releases (or safely keeps) the bytes.
        Returns whether a lease was taken, so a rejected Submit can drop it.
        """

        if self.blob_store is None:
            return False
        if request.blob is None:
            return False
        scope = self._blob_scope(grant_token)
        try:
            state = self.blob_store.state(scope, request.blob.content_id)
            if not state.ready:
                raise JobError(
                    "failed_precondition",
                    f"blob is {state.state}; only a committed blob can be submitted",
                )
            if state.descriptor != request.blob.to_dict():
                raise JobError(
                    "failed_precondition",
                    "request blob descriptor differs from the committed blob",
                )
            self.blob_store.attach(scope, request.blob.content_id, request.request_id)
        except BlobError as exc:
            raise JobError(_BLOB_JOB_ERRORS.get(exc.code, "failed_precondition"), exc.detail) from exc
        return True

    def _release_blob_lease(self, request_id: str, *, attempt: Attempt | None = None) -> None:
        """Release a terminal job's blob lease and record a failed cleanup.

        A cleanup failure is never swallowed: the blob stays durable in the
        ``quarantine`` state and the job records the hazard, so a later reader
        cannot mistake a half-deleted artifact for a clean one.
        """

        if self.blob_store is None:
            return
        try:
            records = self.blob_store.release(request_id)
        except BlobError as exc:
            self._record_blob_hazard(request_id, attempt, f"blob release failed: {exc.detail}")
            return
        failed = [record for record in records if record.get("cleanup") == "quarantine"]
        if failed:
            reason = str(failed[0].get("error") or "blob cleanup did not complete")
            self._record_blob_hazard(request_id, attempt, f"blob cleanup failed: {reason}")

    def _record_blob_hazard(self, request_id: str, attempt: Attempt | None, detail: str) -> None:
        if self._close_error is None:
            self._close_error = RuntimeError(detail[:1000])
        if attempt is None:
            return
        try:
            self.store.record_cleanup_hazard(request_id, attempt, detail[:1000])
        except Exception as exc:  # noqa: BLE001 - defensive durable boundary
            self._record_close_error(f"could not persist blob cleanup hazard: {exc}")

    def reconcile_blobs(self) -> dict[str, list[dict[str, Any]]]:
        """Release leases of jobs that can never use their blob again.

        Runs on start (after a crash) and on shutdown: a crash between commit and
        cleanup then cannot leak bytes, while a live job keeps its lease.
        """

        if self.blob_store is None:
            return {}
        try:
            return self.blob_store.reconcile(self.store.live_request_ids())
        except BlobError as exc:
            self._record_close_error(f"blob reconciliation failed: {exc.detail}")
            return {}

    def blob_state(self, content_id: str, grant_token: str) -> Any:
        """Return the durable blob state for one grant (testing/ops surface)."""

        if self.blob_store is None:
            raise JobError("failed_precondition", "this evaluator has no blob channel")
        try:
            return self.blob_store.state(self._blob_scope(grant_token), content_id)
        except BlobError as exc:
            raise JobError(_BLOB_JOB_ERRORS.get(exc.code, "failed_precondition"), exc.detail) from exc

    def _validate_submission(self, request: ExternalEvaluationRequest) -> None:
        if request.contract != self.contract:
            raise JobError(
                "failed_precondition",
                "request contract is not served by this evaluator; re-handshake with the exact contract",
            )
        if request.target_id != self.identity.target_id:
            raise JobError("failed_precondition", "target is not served by this evaluator")
        if self.contract.npu_source:
            if request.npu is None or self.npu_execution is None:
                raise JobError("invalid_argument", "NPU request has no execution binding")
            if request.npu.execution.to_dict() != self.npu_execution.to_dict():
                raise JobError(
                    "failed_precondition",
                    "request device group/profile/topology/resource generation differs from the served group",
                )
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
        # Only a *new* accepted job needs a ready artifact.  A durable replay is
        # identified by the existing (request_id, request_digest) row, and a
        # terminal job's bytes have already been cleaned up; requiring the blob
        # here would turn an idempotent Submit into not_found and break the six
        # RPC's replay semantics.  The store still re-validates every identity
        # and answers a replay from the committed row, so a different digest or
        # another grant scope is refused exactly as before.
        leased = False
        if self.blob_store is not None and not self.store.job_exists(request.request_id):
            leased = self._lease_blob(request, grant_token)
        try:
            snapshot, _created = self.store.submit(
                request=request, request_digest=request.request_digest, expected=expected,
                provenance=self._base_provenance(), capability_id=capability_id,
                required_features=required_features, grant_token=grant_token,
            )
        except BaseException:
            if leased and not self.store.job_exists(request.request_id):
                # A rejected Submit must not leave a lease behind; the bytes stay
                # (the caller may fix the request and retry the same upload).
                # An existing accepted job keeps its lease: a failed replay is
                # not a terminal outcome.
                try:
                    self.blob_store.release(request.request_id, cleanup=False)
                except BlobError:
                    pass
            raise
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
        acquired = self._active_lock.acquire(timeout=self.cancel_teardown_s)
        if acquired:
            try:
                snapshot = self.store.cancel(
                    grant_token=grant_token, request_id=request_id, request_digest=request_digest,
                    expected=expected, reason=reason,
                )
                running = self._cancel_active_locked(request_id)
                self._cancel_device_for(snapshot)
            finally:
                self._active_lock.release()
        else:
            # A spawn is holding the lock: never let cancel_teardown_s become
            # unbounded.  Commit the durable cancellation without the lock and
            # set the in-memory flag (a bool, atomic under the GIL) so both the
            # writer's final check and the scheduler's wait loop observe it; the
            # request bytes are therefore never sent after this commit.
            snapshot = self.store.cancel(
                grant_token=grant_token, request_id=request_id, request_digest=request_digest,
                expected=expected, reason=reason,
            )
            running = self._active
            if running is not None and running.request.request_id == request_id:
                running.cancel_requested = True
            self._cancel_device_for(snapshot)
            if snapshot.state in {"completed", "failed", "cancelled"}:
                self._release_blob_lease(request_id, attempt=snapshot.current_attempt)
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
        with self._active_lock:
            # The durable revocation, the scope cancellation and every start /
            # request-write authorization share this one lock.  Committing the
            # revocation outside it left a window where a worker could still be
            # started (and receive the request) after the grant was durably
            # revoked but before its scope was cancelled.
            scope_id = self.store.revoke_grant(token)
            running = self.store.cancel_scope(scope_id)
            for request_id in running:
                active = self._cancel_active_locked(request_id)
                self._cancel_device(active.attempt if active is not None else None)
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
            if error:
                self._record_close_error(f"operator release cleanup was not verified: {error}")
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
        # Blob leases whose request is terminal or gone are released before any
        # job is dispatched: a crash between commit and cleanup cannot strand
        # bytes, and a live job keeps its lease.
        self.reconcile_blobs()
        for request_id, attempt, _quarantined in self.store.running_attempts():
            error = self._cleanup_recovered_worker(attempt)
            if error:
                # A quarantined running attempt whose container may still be
                # alive is an unverified shutdown, not a clean start.
                self._record_close_error(f"recovered worker cleanup was not verified: {error}")
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
                error = self.supervisor.cleanup_recovered(attempt, state_dir=self.store.state_dir)
            except Exception as exc:
                return f"execution cleanup failed: {exc}"[:1000]
            if error:
                return error
            return self._reconcile_device(attempt)
        error = self._cleanup_recovered_process(attempt)
        if error:
            return error
        return self._reconcile_device(attempt)

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
            # The device-group lease is taken before the durable claim and held
            # for the whole attempt, so a capability probe can never run
            # concurrently with a candidate start or execution.
            claimed = None
            with self._device_condition:
                while self._device_inflight and not self._stop.is_set():
                    # A timed-out controller call still owns the device.
                    self._device_condition.wait(timeout=0.05)
                if self._stop.is_set():
                    break
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
                if claimed is not None:
                    # Claiming and taking the device group are one atomic step:
                    # a probe can never start between them.
                    self._device_attempt_active = True
            if claimed is None:
                with self._condition:
                    self._condition.wait(timeout=0.05)
                continue
            try:
                self._run_claimed(claimed)
            finally:
                with self._device_condition:
                    self._device_attempt_active = False
                    self._device_condition.notify_all()

    def _run_claimed(self, claimed: tuple[ExternalEvaluationRequest, Attempt]) -> None:
        request, attempt = claimed
        running = _Running(request, attempt, started_monotonic=self._monotonic())
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
                    self._interrupt_attempt(request.request_id, attempt, JobFailure(
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
        problems: list[str] = []
        try:
            error = self.supervisor.cleanup(handle)
        except Exception as exc:
            error = f"execution cleanup failed: {exc}"[:1000]
        if error:
            problems.append(error)
        # ``release`` is part of cleanup: a failure means the attach client may
        # still exist, so it cannot be swallowed while the caller is told the
        # teardown was verified.
        try:
            self.supervisor.release(handle)
        except Exception as exc:
            problems.append(f"execution release failed: {exc}"[:1000])
        # Closing a buffered read stream while its reader is still blocked in
        # ``read`` deadlocks on the buffer lock.  ``_run_worker`` records whether
        # the readers were started and whether they finished; only close the
        # read sides when that is provably safe.
        reader_state = getattr(handle, "metadata", {}).get("reader_state")
        writer_state = getattr(handle, "metadata", {}).get("writer_state")
        self._close_handle_streams(
            handle,
            include_reads=reader_state in {None, "done"},
            include_writes=writer_state in {None, "done"},
        )
        if not problems:
            # A verified kill/delete is not a verified *residue-free* worker:
            # a surviving attach client (or scratch/env file) must still block
            # the interrupt/result path exactly as it does everywhere else.
            try:
                error = self.supervisor.cleanup_residue(handle)
            except Exception as exc:
                error = f"execution residue check failed: {exc}"[:1000]
            if error:
                problems.append(error)
        return "; ".join(problems)

    @staticmethod
    def _close_handle_streams(
        handle: WorkerHandle, *, include_reads: bool = True, include_writes: bool = True,
    ) -> None:
        """Close the worker's private streams; never deadlock on a live thread.

        ``BufferedReader.close``/``BufferedWriter.close`` take the buffered
        object's lock, so closing stdout/stderr while a reader is blocked in
        ``read`` (or stdin while the request writer is blocked in ``write``)
        deadlocks the scheduler.  Callers pass ``include_reads=False`` /
        ``include_writes=False`` unless the matching thread provably finished.
        """

        streams: list = []
        if include_writes:
            streams += [
                getattr(getattr(handle, "process", None), "stdin", None),
                getattr(getattr(handle, "attach", None), "stdin", None),
            ]
        if include_reads:
            streams += [
                getattr(getattr(handle, "process", None), "stdout", None),
                getattr(getattr(handle, "process", None), "stderr", None),
                getattr(getattr(handle, "attach", None), "stdout", None),
                getattr(getattr(handle, "attach", None), "stderr", None),
            ]
        for stream in streams:
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
        # The writer authorizes under ``_active_lock``; the flag is therefore
        # never set "after" an authorization that already decided to write.
        # The kill itself happens on the bounded scheduling path: doing it here
        # would let a blocking ``supervisor.kill`` outlive the caller's teardown
        # deadline (Cancel RPC, close(wait), ProductionStack.stop).
        running.cancel_requested = True
        return running

    def _popen(self) -> subprocess.Popen:
        env = scrubbed_env()
        env.update(self.worker_env)
        # Engine-owned, so a caller cannot point the private worker temp root at
        # another attempt's state directory.
        env[STATE_DIR_ENV] = str(self.store.state_dir)
        if self.blob_store is not None:
            # Engine-owned: the worker reads the committed artifact by content id,
            # never a path a caller supplied.
            env[BLOB_ROOT_ENV] = str(self.blob_store.root)
        root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        return popen_group(
            self.worker_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, cwd=str(self.worker_cwd),
        )

    def _accepted_provenance(self, running: _Running) -> dict[str, Any]:
        """The accepted base provenance of this job (no attempt/request fields)."""

        snapshot = self.store._job_by_id(running.request.request_id)
        return {
            key: value for key, value in (snapshot.provenance or {}).items()
            if key not in {"attempt_id", "request_digest"}
        }

    def _attempt_deadline(self, running: _Running) -> float:
        """Absolute monotonic deadline for this attempt's setup and execution."""

        return running.started_monotonic + running.request.candidate_timeout_ms / 1000.0

    def _overdue(self, running: _Running) -> bool:
        return self._monotonic() >= self._attempt_deadline(running)

    def _timeout_report(self, running: _Running, message: str) -> dict[str, Any]:
        """The durable timeout report for one attempt that never finished in time."""

        from tcenv.evaluate import EvalReport

        request = running.request
        report = EvalReport(
            task=request.task, dtype=request.dtype, size=request.size, mode=request.mode,
            status="timeout", stage="timeout", message=message,
            seed=request.seed_int or 0, protocol_version=self.execution_protocol_version,
            target_id=request.target_id, backend=self.identity.backend,
            backend_version=self.identity.backend_version,
            elapsed_s=request.candidate_timeout_ms / 1000.0,
        ).to_json()
        if self.identity.measurement_kind == "simulated":
            report["simulated"] = True
        return report

    def _timeout_evidence(self, running: _Running, message: str) -> dict[str, Any]:
        """Diagnostic evidence for a candidate that never produced a device event.

        The durable deadline fired, so no device synchronization was observed.
        The NPU contract must not launder that into a completed device event:
        the row keeps the exact CandidateDigest/device group binding, reports no
        launch and no device time, names the unobserved clock and is never
        publishable.  ``JobStore`` requires the matching ``status=timeout``.
        """

        request = running.request
        if request.npu is None or self.npu_execution is None:
            raise StateCorrupt("the NPU contract has no execution binding for a timeout report")
        return {
            "schema_id": contracts.NPU_TIMING_EVIDENCE_SCHEMA_ID,
            "candidate_digest": request.npu.candidate_digest.to_dict(),
            "device_group": request.npu.execution.device_group.to_dict(),
            "timing": {
                "source": "DEVICE_EVENT",
                "clock_domain": contracts.NPU_UNOBSERVED_CLOCK_DOMAIN,
                "synchronization_completed": False,
                "device_elapsed_ns": 0,
                "host_api_elapsed_ns": request.candidate_timeout_ms * 1_000_000,
                "launches": 0,
            },
            "measurement_kind": self.identity.measurement_kind,
            "publication_allowed": False,
            "complete": False,
            "diagnostics": {"timeout": True, "detail": message[:400]},
        }

    def _complete_timeout_result(self, running: _Running, *, message: str) -> None:
        evidence = (
            {"candidate_sha256": running.request.candidate_sha256}
            if not self.contract.npu_source
            else self._timeout_evidence(running, message)
        )
        self._complete_worker_result(
            running, report=self._timeout_report(running, message), evidence=evidence,
        )

    def _finish_overdue_attempt(
        self, running: _Running, *, attempt: Attempt, handle: WorkerHandle | None = None,
    ) -> None:
        """The candidate deadline expired before the worker could run it.

        Setup (daemon query, image inspect/create, identity check, start) counts
        against the same deadline as execution.  A worker that has not started
        when the deadline passes must never be started or written to; the
        attempt is cleaned up and reported as the timeout it is.
        """

        if handle is not None:
            cleanup_error = self._cleanup_handle_quietly(handle)
        else:
            cleanup_error = self._cleanup_uncreated_worker(attempt)
        if cleanup_error:
            self._record_cleanup_hazard(
                running,
                f"candidate deadline expired during setup; {cleanup_error}"[:1000],
            )
            return
        snapshot = self.store._job_by_id(running.request.request_id)
        if snapshot.cancel_requested or running.cancel_requested:
            self._finish_cancellation(running.request.request_id, attempt)
            return
        if running.stopping or self._stop.is_set():
            self._interrupt_attempt(running.request.request_id, attempt, JobFailure(
                "infrastructure", True, "frontend_stopped", "frontend stopped while the attempt was active",
            ))
            return
        self._complete_timeout_result(
            running, message="candidate deadline expired before the worker ran it",
        )

    def _handle_worker_job_error(self, running: _Running, exc: JobError) -> None:
        request, attempt = running.request, running.attempt
        domain = exc.failure_domain or ("infrastructure" if exc.code in {"data_loss", "resource_exhausted"} else "protocol")
        code = exc.code if exc.code in {"data_loss", "resource_exhausted"} else "worker_result_invalid"
        failure = JobFailure(domain, exc.retryable, code, str(exc)[:1000])
        if exc.retryable:
            self._interrupt_attempt(request.request_id, attempt, failure)
            return
        if exc.code == "data_loss":
            applied = self._interrupt_attempt(
                request.request_id, attempt, failure,
                quarantine=True, recovery_error=str(exc),
            )
        else:
            applied = self._fail_attempt(request.request_id, attempt, failure)
        if not applied:
            snapshot = self.store._job_by_id(request.request_id)
            if snapshot.state in {"completed", "failed", "cancelled", "expired"}:
                self.store.mark_unavailable(StateCorrupt(
                    f"could not apply non-retryable {exc.code} to terminal job "
                    f"{snapshot.state}: {exc}"
                ))
                self._stop.set()

    # -- device controller boundary ----------------------------------------
    @staticmethod
    def _lease_binding(value: Any) -> dict[str, Any] | None:
        """Normalize the device binding a lease was accepted with."""

        if value is None:
            return None
        try:
            return contracts.device_group_document(value)
        except contracts.ContractError:
            return None

    def _lease_device_error(self, attempt: Attempt) -> tuple[dict[str, Any] | None, str]:
        """Return ``(accepted_binding, "")`` or ``(None, error)`` for one lease.

        The lease is bound to the device group/profile/topology/resource
        generation the job was accepted with.  A frontend that restarted into a
        different group must not reconcile (or interrupt) the old attempt
        through the new group's controller.
        """

        if not self.contract.npu_source:
            return None, ""
        if self.npu_execution is None:
            return None, "this evaluator has no NPU device binding"
        accepted = self._lease_binding(getattr(attempt, "device_binding", None))
        if accepted is None:
            return None, "the accepted attempt has no durable device group binding"
        current = self.npu_execution.device_group.to_dict()
        if accepted != current:
            changed = sorted(
                key for key in current if accepted.get(key) != current[key]
            )
            return None, (
                "the accepted attempt belongs to another device group "
                f"({','.join(changed)}): this controller cannot reconcile it"
            )
        return accepted, ""

    def _reconcile_device(self, attempt: Attempt) -> str:
        """Return "" only when the worker-external controller proved cleanup.

        A worker-local ``VendorDriver.close`` is cooperative, never proof, and
        the proof must name the exact device binding the lease was accepted with.
        """

        controller = self.device_controller
        if controller is None:
            if self.contract.npu_source:
                return "no device controller is configured; device cleanup cannot be proven"
            return ""
        accepted, lease_error = self._lease_device_error(attempt)
        if lease_error:
            return lease_error
        # Bounded: a wedged controller must not stall the scheduler, the result
        # commit or the Cancel RPC.  A timeout is an infrastructure failure and
        # therefore leaves the job quarantined instead of claiming cleanup.
        proof, reconcile_error = self._bounded_device_call(
            "reconcile", lambda: controller.reconcile(attempt.attempt_id, attempt.lease_fence),
            owner=True,
        )
        if reconcile_error:
            return f"device reconcile failed: {reconcile_error}"[:1000]
        if not isinstance(proof, CleanupProof):
            return "device controller returned an untyped cleanup proof"
        if proof.attempt_id != attempt.attempt_id or proof.lease_fence != attempt.lease_fence:
            return (
                "device cleanup proof is bound to a different attempt/fence: "
                f"{proof.attempt_id}/{proof.lease_fence}"
            )
        if accepted is not None:
            proof_binding = {
                "device_group_id": proof.device_group_id,
                "vendor_id": proof.vendor_id,
                "profile_digest": proof.profile_digest,
                "topology_digest": proof.topology_digest,
                "resource_generation": proof.resource_generation,
            }
            if proof_binding != accepted:
                changed = sorted(key for key in accepted if proof_binding.get(key) != accepted[key])
                return (
                    "device cleanup proof names a different device binding "
                    f"({','.join(changed) or 'unknown'}) than the accepted lease"
                )
        if not proof.complete:
            live = ",".join(proof.live_resources) or "unknown"
            return f"device cleanup was not proven; live resources: {live}; {proof.detail}"[:1000]
        if proof.live_resources:
            # A "complete" proof that still names live resources contradicts
            # itself; treat it as an unproven cleanup, never as success.
            live = ",".join(sorted(proof.live_resources))
            return (
                f"device cleanup claimed complete but still reports live resources: {live}; "
                f"{proof.detail}"
            )[:1000]
        return ""

    def _record_device_cleanup_failure(self, request_id: str, attempt: Attempt, error: str) -> None:
        if self._close_error is None:
            self._close_error = RuntimeError(f"device cleanup was not verified: {error}"[:1000])
        try:
            snapshot = self.store._job_by_id(request_id)
            current = snapshot.current_attempt
            if (
                snapshot.state == "running" and current is not None
                and current.attempt_id == attempt.attempt_id
                and current.lease_fence == attempt.lease_fence
            ):
                self.store.recovered_attempt(request_id, attempt, cleanup_error=error[:1000])
                return
            self.store.record_cleanup_hazard(request_id, attempt, error[:1000])
        except Exception as exc:  # noqa: BLE001 - defensive durable boundary
            self._record_close_error(f"could not persist device cleanup hazard: {exc}")
            try:
                self.store.mark_unavailable(exc)
            finally:
                self._stop.set()

    def _cancel_device_for(self, snapshot: JobSnapshot) -> None:
        """Cancel a live device lease; a terminal job has nothing to cancel."""

        if snapshot.state != "running":
            return
        self._cancel_device(snapshot.current_attempt)

    def _cancel_device(self, attempt: Attempt | None) -> None:
        """Best-effort typed cancel; cleanup is still proven by reconcile.

        A cancel is only delivered for a lease this frontend's controller owns;
        another group's attempt is left to quarantine plus operator recovery
        instead of being cancelled through the wrong device group.
        """

        if attempt is None or self.device_controller is None:
            return
        _accepted, lease_error = self._lease_device_error(attempt)
        if lease_error:
            self._record_close_error(f"device cancel was not delivered: {lease_error}")
            return
        controller = self.device_controller
        # Bounded and best effort: reconcile still decides whether the device
        # resources are gone, so a hung cancel can never become a "cancelled"
        # claim on its own.
        _value, cancel_error = self._bounded_device_call(
            "cancel", lambda: controller.cancel(attempt.attempt_id, attempt.lease_fence),
            owner=True,
        )
        if cancel_error:
            self._record_close_error(f"device cancel could not be delivered: {cancel_error}")

    def _interrupt_attempt(
        self, request_id: str, attempt: Attempt, failure: JobFailure,
        *, quarantine: bool = False, recovery_error: str = "",
    ) -> bool:
        error = self._reconcile_device(attempt)
        if error:
            self._record_device_cleanup_failure(request_id, attempt, error)
            return False
        return self.store.interrupt_attempt(
            request_id, attempt, failure,
            quarantine=quarantine, recovery_error=recovery_error,
        )

    def _finish_cancellation(self, request_id: str, attempt: Attempt) -> bool:
        error = self._reconcile_device(attempt)
        if error:
            self._record_device_cleanup_failure(request_id, attempt, error)
            return False
        finished = self.store.finish_cancellation(request_id, attempt)
        if finished:
            self._release_blob_lease(request_id, attempt=attempt)
        return finished

    def _fail_attempt(self, request_id: str, attempt: Attempt, failure: JobFailure) -> bool:
        error = self._reconcile_device(attempt)
        if error:
            self._record_device_cleanup_failure(request_id, attempt, error)
            return False
        failed = self.store.fail_attempt(request_id, attempt, failure)
        if failed:
            self._release_blob_lease(request_id, attempt=attempt)
        return failed

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
        device_error = self._reconcile_device(attempt)
        if device_error:
            self._record_device_cleanup_failure(request.request_id, attempt, device_error)
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
            self._finish_cancellation(request.request_id, attempt)
        if snapshot.state in {"completed", "failed", "cancelled"}:
            self._release_blob_lease(request.request_id, attempt=attempt)

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
            # The provenance is signed into the result byte for byte.  Every
            # field (measurement_kind, publication_allowed, backend, epoch,
            # calibration, contracts) must be the one this frontend signs now;
            # comparing only worker_build would let a queued job be executed and
            # published under a provenance it was never accepted with.
            accepted_provenance = {
                key: value for key, value in (snapshot.provenance or {}).items()
                # These two are attempt/request scoped and are added to the
                # accepted base provenance when the snapshot is built.
                if key not in {"attempt_id", "request_digest"}
            }
            expected_provenance = self._base_provenance()
            if accepted_provenance == expected_provenance:
                return ""
            drifting = sorted(
                key for key in set(accepted_provenance) | set(expected_provenance)
                if accepted_provenance.get(key) != expected_provenance.get(key)
            )
            return (
                "queued job provenance differs from this frontend's provenance "
                f"(fields: {drifting}); refusing to execute and sign it here"
            )
        durable = accepted.to_dict() if accepted is not None else None
        return (
            "queued job was accepted by a different evaluator identity "
            f"({durable!r} != {self._expected.to_dict()!r}); refusing to execute it here"
        )

    def _cleanup_handle_quietly(self, handle: WorkerHandle, *, close_streams: bool = True) -> str:
        """Kill, delete and verify a worker that must not receive a request."""

        problems: list[str] = []
        try:
            error = self.supervisor.cleanup(handle)
        except Exception as exc:
            error = f"execution cleanup failed: {exc}"[:1000]
        if error:
            problems.append(error)
        # ``release`` drops the attach client; a failure there is a residue risk
        # and must not be swallowed, so it runs before the residue proof.
        try:
            self.supervisor.release(handle)
        except Exception as exc:
            problems.append(f"execution release failed: {exc}"[:1000])
        if not problems:
            try:
                error = self.supervisor.cleanup_residue(handle)
            except Exception as exc:
                error = f"execution residue check failed: {exc}"[:1000]
            if error:
                problems.append(error)
        if close_streams:
            self._close_handle_streams(handle)
        return "; ".join(problems)

    def _record_close_error(self, message: str) -> None:
        """Remember an unverified teardown for close()/ProductionStack.stop."""

        if self._close_error is None:
            self._close_error = RuntimeError(message[:1000])

    def _record_cleanup_hazard(self, running: _Running, error: str) -> None:
        """Fail closed: no result commit, no Retry and no verified shutdown.

        An asynchronous cleanup failure is not only a job state: the process
        may still own a surviving worker, so ``close()`` must report the
        unverified teardown instead of returning ``True`` with no error.
        """

        if self._close_error is None:
            self._close_error = RuntimeError(f"worker cleanup was not verified: {error}"[:1000])
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
            # The failed cleanup is also an unverified shutdown: record it for
            # close()/ProductionStack.stop, not only on the container branch.
            if self._close_error is None:
                self._close_error = RuntimeError(
                    f"worker cleanup was not verified: {recovery_error}"[:1000]
                )
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
            self._finish_cancellation(request.request_id, attempt)
            return
        self._interrupt_attempt(request.request_id, attempt, failure)

    def _run_worker(self, running: _Running) -> None:
        request, attempt = running.request, running.attempt
        if running.stopping or self._stop.is_set():
            self._interrupt_attempt(request.request_id, attempt, JobFailure(
                "infrastructure", True, "frontend_stopped", "frontend stopped before worker startup",
            ))
            return
        if running.cancel_requested:
            self._finish_cancellation(request.request_id, attempt)
            return
        identity_error = self._accepted_identity_error(self.store._job_by_id(request.request_id))
        if identity_error:
            # The accepted job carries the submitting evaluator's identity.
            # Running it under any other build would sign a result that names
            # an evaluator/image which never ran it.
            self._interrupt_attempt(
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
                self._interrupt_attempt(request.request_id, attempt, JobFailure(
                    "infrastructure", True, "worker_daemon_identity", daemon_error[:1000],
                ))
                return
            if not self.store.attach_daemon(request.request_id, attempt, daemon=daemon):
                # No container exists yet; the durable fence moved on.
                self._interrupt_attempt(request.request_id, attempt, JobFailure(
                    "infrastructure", True, "worker_attach_fenced",
                    "the Docker daemon identity was not attached to the exact running attempt/fence",
                ))
                return
            attempt = replace(attempt, container_daemon=daemon)
        if self._overdue(running):
            # The daemon query (or anything before it) already spent the whole
            # candidate deadline: no container may be created or started now.
            self._finish_overdue_attempt(running, attempt=attempt)
            return
        spec = AttemptSpec(
            request=request, attempt=attempt, state_dir=self.store.state_dir,
            command=tuple(self.worker_command), env=dict(self.worker_env),
        )
        try:
            handle = self.supervisor.create(spec)
        except DaemonChanged as exc:
            # The container may exist on another daemon; an absence on the
            # persisted one proves nothing, so this must not become a retryable
            # interrupted attempt.
            self._record_cleanup_hazard(
                running, f"could not start worker: {exc}"[:1000],
            )
            return
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
                self._finish_cancellation(request.request_id, attempt)
            else:
                self._interrupt_attempt(request.request_id, attempt, JobFailure(
                    "infrastructure", True, "worker_unavailable", f"could not start worker: {exc}"[:1000],
                ))
            return
        running.handle = handle
        if self._stop.is_set() or running.stopping:
            self._abort_worker_before_request(running, handle, JobFailure(
                "infrastructure", True, "frontend_stopped", "frontend stopped during worker startup",
            ), worker_attached=False)
            return
        if self._overdue(running):
            # ``create``/``image inspect`` overran the candidate deadline.  The
            # container exists but must never be started.
            self._finish_overdue_attempt(running, attempt=attempt, handle=handle)
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
                self._interrupt_attempt(request.request_id, attempt, JobFailure(
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
                    self._finish_cancellation(request.request_id, attempt)
                else:
                    self._interrupt_attempt(request.request_id, attempt, JobFailure(
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
                    self._finish_cancellation(request.request_id, attempt)
                else:
                    self._interrupt_attempt(request.request_id, attempt, failure)
                return
            daemon_error, daemon = self._supervisor_daemon()
            if daemon_error:
                cleanup_error = self._cleanup_handle_quietly(handle)
                detail = daemon_error if not cleanup_error else f"{daemon_error}; {cleanup_error}"
                if cleanup_error:
                    self._record_cleanup_hazard(running, detail[:1000])
                else:
                    self._interrupt_attempt(request.request_id, attempt, JobFailure(
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
        overdue_before_start = False
        # The Cancel RPC commits the durable cancellation while holding
        # ``_active_lock``; reading the durable state and starting the worker
        # under the same lock makes "cancelled before start" impossible to miss.
        # The candidate deadline is re-checked under the same lock so an
        # ``inspect``/identity probe that overran it can never be followed by a
        # start.
        gate = _StartGate(self, running)
        # The blocking supervisor preparation (daemon bind) must not hold
        # ``_active_lock``: Cancel/close take it to commit the durable
        # cancellation and to stop the attempt, and a slow bind would make them
        # exceed their own deadlines.  The supervisor then runs the spawn inside
        # ``gate``, which re-checks the state while holding the lock, so a
        # cancellation/revocation can never commit between the final check and
        # the start.  The writer re-authorizes after encoding as well.
        before_start = self.store._job_by_id(request.request_id)
        cancelled_before_start = bool(before_start.cancel_requested or running.cancel_requested)
        if not cancelled_before_start and self._overdue(running):
            overdue_before_start = True
        elif not cancelled_before_start:
            try:
                self.supervisor.start(handle, gate=gate)
                stdin, stdout, stderr = self.supervisor.streams(handle)
            except StartRefused as exc:
                if exc.reason == "deadline":
                    overdue_before_start = True
                elif exc.reason == "cancelled":
                    cancelled_before_start = True
                    running.cancel_requested = True
                else:
                    start_error = exc
            except Exception as exc:
                start_error = exc
        if overdue_before_start:
            self._finish_overdue_attempt(running, attempt=attempt, handle=handle)
            return
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

        stderr_error: list[BaseException] = []

        def read_stderr() -> None:
            tail = bytearray()
            try:
                while True:
                    chunk = stderr.read(4096)
                    if not chunk:
                        break
                    tail = (tail + chunk)[-MAX_STDERR:]
            except BaseException as exc:  # the reader boundary must record, not die
                stderr_error.append(exc)
            stderr_tail.append(tail.decode("utf-8", errors="replace"))

        out_thread = threading.Thread(target=read_stdout, name="tcenv-worker-stdout", daemon=True)
        err_thread = threading.Thread(target=read_stderr, name="tcenv-worker-stderr", daemon=True)
        out_thread.start()
        # Record the first reader before starting the second: if the second
        # start raises, the first reader may still own stdout and must never be
        # closed by the cleanup boundary.
        handle.metadata["reader_state"] = "running"
        err_thread.start()
        timed_out = False
        returncode: int | None = None
        write_error: list[BaseException] = []

        def write_request() -> None:
            try:
                # Encoding is not free (a large candidate is copied and framed):
                # do it *before* the final authorization, so a deadline that
                # passes during encoding can never be followed by a write.
                payload = encode_request(request, attempt)
                # The authorization and the cancellation are mutually exclusive:
                # whoever takes ``_active_lock`` first wins.  A cancel that wins
                # (including its durable commit) means this worker never
                # receives candidate bytes.
                with self._active_lock:
                    if self._stop.is_set() or running.stopping or running.cancel_requested:
                        return
                    # A Cancel may have committed while a spawn held the lock
                    # (its in-memory flag is set unlocked); re-read the durable
                    # state as well before sending any candidate bytes.
                    if self.store._job_by_id(request.request_id).cancel_requested:
                        return
                    if self._overdue(running):
                        # Never send candidate bytes after the deadline: the
                        # worker is killed at the deadline anyway, so writing
                        # would only claim execution that did not happen.
                        return
                    running.request_authorized = True
                assert stdin is not None
                stdin.write(payload)
                stdin.flush()
                stdin.close()
            except BaseException as exc:  # broken pipe is handled by the supervisor
                write_error.append(exc)

        writer_thread = threading.Thread(target=write_request, name="tcenv-worker-stdin", daemon=True)
        writer_thread.start()
        handle.metadata["writer_state"] = "running"
        try:
            try:
                # The request writer is concurrent: a broken worker that never
                # reads stdin still cannot escape the candidate deadline.
                #
                # Wait in bounded slices so the absolute deadline is re-checked
                # against the clock (and a test can expire it deterministically)
                # instead of trusting one long wait.  The deadline remains
                # absolute: ``DockerAttemptSupervisor.wait`` can notice an exit
                # on a slow daemon query and return a *real* exit code after the
                # deadline, which is still a timeout, never an earned "ok".
                while True:
                    if self._overdue(running):
                        timed_out = True
                        break
                    # Cancel/close must be torn down by this bounded scheduling
                    # path, not by the RPC or the close() caller: their waits
                    # then bound their own return instead of a blocking kill.
                    if running.cancel_requested or running.stopping or self._stop.is_set():
                        break
                    remaining = self._attempt_deadline(running) - self._monotonic()
                    returncode = self.supervisor.wait(
                        handle, max(0.001, min(_DEADLINE_SLICE_S, remaining)),
                    )
                    if returncode is not None:
                        timed_out = self._overdue(running)
                        break
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
            # When cleanup already failed the attempt is a hazard regardless of
            # the streams, so a blocked reader/writer only needs a short bounded
            # probe (a verified cleanup keeps the full window for the late EPIPE
            # or EOF that a real exit produces).
            join_window = 0.2 if cleanup_error else 2.0
            out_thread.join(timeout=join_window)
            err_thread.join(timeout=join_window)
            # The request writer must also be checked *after* cleanup: a worker
            # that never read stdin leaves it blocked inside ``write`` with a
            # full pipe, and closing that buffered writer from this thread would
            # wait on the writer's lock forever (the same deadlock class as the
            # readers).  A late EPIPE after the worker really died gets this
            # second bounded window to finish.
            writer_thread.join(timeout=join_window)
            writer_alive = writer_thread.is_alive()
            # A writer that only failed during cleanup (a late EPIPE when the
            # worker exits before reading the whole request) is still a request
            # that was not fully delivered: it must never be reported as ok.
            if (
                write_error
                and not running.cancel_requested and not running.stopping
                and not self._stop.is_set() and not timed_out and not supervisor_error
            ):
                stdout_error.append(write_error[0])
            reader_alive = out_thread.is_alive()
            readers_alive = out_thread.is_alive() or err_thread.is_alive()
            handle.metadata["reader_state"] = "running" if readers_alive else "done"
            handle.metadata["writer_state"] = "running" if writer_alive else "done"
            self._close_handle_streams(
                handle, include_reads=not readers_alive, include_writes=not writer_alive,
            )

        if cleanup_error:
            if writer_alive:
                cleanup_error = (
                    f"{cleanup_error}; the request writer is still blocked on stdin "
                    "after cleanup"
                )
            self._record_cleanup_hazard(running, cleanup_error[:1000])
            return
        if writer_alive:
            # The request may be only partially delivered and its stdin stream
            # is still owned by the writer thread: neither a result commit nor a
            # verified teardown is acceptable, and the stream must not be closed
            # from here.  Quarantine so no Retry starts over a live worker.
            self._record_cleanup_hazard(
                running,
                "worker request writer did not finish after cleanup; "
                "the attempt is not a verified result or teardown",
            )
            return
        if stdout_error:
            error = stdout_error[0]
            if isinstance(error, JobError):
                self._handle_worker_job_error(running, error)
            else:
                self._interrupt_attempt(request.request_id, attempt, JobFailure(
                    "protocol", True, "worker_protocol_error", str(error)[:1000],
                ))
            return
        if readers_alive:
            # Either reader may still hand the worker's output to the decoder: a
            # live stderr reader can hold back/obscure the child's output just as
            # a live stdout reader can.  No result may be committed until both
            # readers reached EOF.
            self._interrupt_attempt(request.request_id, attempt, JobFailure(
                "protocol", True, "worker_protocol_error",
                "worker output was not fully read and validated before the result decision",
            ))
            return
        if supervisor_error:
            error = supervisor_error[0]
            self._interrupt_attempt(request.request_id, attempt, JobFailure(
                "infrastructure", True, "worker_supervisor_failed", str(error)[:1000],
            ))
            return
        snapshot = self.store._job_by_id(request.request_id)
        if snapshot.cancel_requested or running.cancel_requested:
            self._finish_cancellation(request.request_id, attempt)
            return
        if running.stopping or self._stop.is_set():
            self._interrupt_attempt(request.request_id, attempt, JobFailure(
                "infrastructure", True, "frontend_stopped", "frontend stopped while the attempt was active",
            ))
            return
        if stderr_error:
            # The stderr reader failed; the bridge cannot prove it saw the
            # complete child output, so an "ok" result is not acceptable.
            error = stderr_error[0]
            self._interrupt_attempt(request.request_id, attempt, JobFailure(
                "protocol", True, "worker_protocol_error",
                f"worker stderr could not be read: {error}"[:1000],
            ))
            return
        if timed_out:
            # The candidate deadline takes precedence over every *valid* result:
            # a worker that emitted a result and then hung did not finish inside
            # the durable deadline, so committing it would turn a hang into an
            # unearned "ok".  A result that does not even validate is a protocol
            # failure, not a timeout: laundering it into a timeout would hide a
            # broken worker behind a plausible slow-candidate outcome.
            if results:
                try:
                    self.store.validate_worker_result(
                        request, report=results[0].get("report"),
                        evidence=results[0].get("evidence"),
                        provenance=self._accepted_provenance(running),
                    )
                except JobError as exc:
                    self._handle_worker_job_error(running, exc)
                    return
            self._complete_timeout_result(
                running, message="candidate exceeded the durable execution deadline",
            )
            return
        if not results:
            if returncode != 0:
                detail = stderr_tail[0].replace(str(self.store.state_dir), "<state>") if stderr_tail else ""
                self._interrupt_attempt(request.request_id, attempt, JobFailure(
                    "infrastructure", True, "worker_failed", f"worker exited {returncode}: {detail}"[:1000],
                ))
                return
            self._interrupt_attempt(request.request_id, attempt, JobFailure(
                "protocol", True, "worker_protocol_error", "worker did not emit exactly one result",
            ))
            return
        # A complete result was decoded and ``decoder.finish()`` validated the
        # stream (no malformed frame, no second result, reader reached EOF).
        # A worker that then also exited non-zero is still a failure: the result
        # must not be trusted over a crashed supervisor exit.  A timeout already
        # returned above, so ``returncode`` is a real exit code here.
        if len(results) != 1:
            self._interrupt_attempt(request.request_id, attempt, JobFailure(
                "protocol", True, "worker_protocol_error", "worker emitted multiple results",
            ))
            return
        if returncode != 0:
            detail = stderr_tail[0].replace(str(self.store.state_dir), "<state>") if stderr_tail else ""
            self._interrupt_attempt(request.request_id, attempt, JobFailure(
                "infrastructure", True, "worker_failed", f"worker exited {returncode}: {detail}"[:1000],
            ))
            return
        self._complete_worker_result(
            running, report=results[0]["report"], evidence=results[0]["evidence"],
        )

    def close(self, *, wait: float | None = None) -> bool:
        """Stop the frontend; ``True`` only once the durable store is closed.

        Fails soft but visibly: an arbitrary supervisor exception is contained
        (reported through the return value, never raised out of a shutdown), the
        store is closed only after its writer stopped, and a scheduler that
        refuses to stop leaves the last step to the bounded finalizer instead of
        being closed underneath a live writer.
        """

        if wait is None:
            deadline = None
            acquired = self._close_lock.acquire(blocking=False)
        else:
            deadline = time.monotonic() + max(0.0, float(wait))
            acquired = self._close_lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
        if not acquired:
            # Another close is still starting the teardown: never claim it.
            if wait is None:
                return self._closed_event.is_set() and self._close_error is None
            return False
        try:
            if not self._close_started:
                self._close_started = True
                self._close_once(
                    probe=None if wait is not None else 0.2, deadline=deadline,
                )
        finally:
            self._close_lock.release()
        if self._closed_event.is_set():
            return self._close_error is None
        if wait is None:
            return False
        return self._closed_event.wait(
            timeout=max(0.0, deadline - time.monotonic()),
        ) and self._close_error is None

    def _close_once(self, probe: float | None = None, deadline: float | None = None) -> None:
        """One shutdown attempt; every failure is recorded, never raised.

        A supervisor that returns a non-empty ``kill`` result did *not* prove
        the worker stopped, so shutdown is incomplete even though the store can
        still close.  The error is remembered for ``close()``/ProductionStack.
        """

        try:
            self._stop.set()
            # Never wait indefinitely for a spawn that holds the lock: close(wait)
            # must bound its own return.  ``_stop`` is already set, so the
            # scheduler notices it after the spawn and tears down; the finalizer
            # closes the store once the scheduler exits.
            if deadline is None:
                acquired = self._active_lock.acquire(blocking=False)
            else:
                acquired = self._active_lock.acquire(
                    timeout=max(0.0, deadline - time.monotonic()),
                )
            if acquired:
                try:
                    running = self._active
                    if running is not None:
                        running.stopping = True
                finally:
                    self._active_lock.release()
            # Teardown is performed by the bounded scheduling path (its wait
            # loop notices ``stopping`` and cleans up under the supervisor's own
            # kill timeout).  Calling kill() here would make close(wait) and
            # ProductionStack.stop wait for that kill instead of bounding their
            # own return.
            self._wake()
            scheduler = getattr(self, "_scheduler", None)
            if probe is not None and scheduler is not None:
                # close() without a wait keeps the historical idle behavior (the
                # store is closed before returning); a close(wait=...) never
                # joins here, because its return must be bounded by wait.
                scheduler.join(timeout=probe)
                if not scheduler.is_alive():
                    try:
                        self.store.close()
                    except Exception as exc:
                        self._close_error = exc
                    self._closed_event.set()
                    return
            threading.Thread(
                target=self._close_store_after_scheduler,
                args=(scheduler, self.store),
                name="tcenv-external-jobs-close", daemon=True,
            ).start()
        except Exception as exc:  # pragma: no cover - defensive shutdown boundary
            if self._close_error is None:
                self._close_error = exc
            self._closed_event.set()

    @property
    def close_error(self) -> BaseException | None:
        return self._close_error

    @property
    def closed(self) -> bool:
        """True only once the durable store is really closed."""

        return self._closed_event.is_set()

    def _close_store_after_scheduler(self, scheduler: threading.Thread | None, store: JobStore) -> None:
        try:
            if scheduler is not None:
                scheduler.join()
        except Exception as exc:  # pragma: no cover - defensive shutdown boundary
            self._close_error = exc
        # Release terminal job leases before the store closes; a failing cleanup
        # is recorded (and the blob stays quarantined) instead of being hidden.
        self.reconcile_blobs()
        try:
            store.close()
        except Exception as exc:  # pragma: no cover - defensive shutdown boundary
            self._close_error = exc
        finally:
            self._closed_event.set()

    def __enter__(self) -> "JobEngine":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
