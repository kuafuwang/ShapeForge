"""Single-owner SQLite persistence for the external evaluator job engine.

Only :mod:`sqlite3` from the Python standard library is used.  Every mutation is
serialized by the owner lock, an in-process writer lock and ``BEGIN IMMEDIATE``.
The database is not an HA store: one host-owned frontend owns one device queue.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Iterator, Mapping

from . import contracts
from .identity import SessionGrant
from .job_types import (
    ATTEMPT_STATES,
    CANDIDATE_STATUSES,
    JOB_STATES,
    TERMINAL_JOB_STATES,
    Attempt,
    Capabilities,
    ExpectedIdentities,
    ExternalEvaluationRequest,
    JobError,
    JobEvent,
    JobFailure,
    JobResult,
    JobSnapshot,
    QuotaUsage,
    StateCorrupt,
)

SCHEMA_VERSION = 1
DEFAULT_RETENTION_MS = 30 * 24 * 60 * 60 * 1000
DB_WRITE_SLACK_BYTES = 64 * 1024
_FEATURE = re.compile(r"[A-Za-z0-9._:-]{1,80}")
_FORBIDDEN_STATE_ROOTS = (Path("/tmp"), Path("/var/tmp"), Path("/private/tmp"), Path("/dev/shm"))
_NETWORK_FILESYSTEMS = {"nfs", "nfs4", "cifs", "smbfs", "smb3", "sshfs", "fuse.sshfs", "9p", "afs"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(document: Any) -> str:
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _loads(raw: str | bytes) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise StateCorrupt(f"invalid durable JSON: {exc}") from exc


def _now_ms(clock) -> int:
    value = clock()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("clock must return finite seconds")
    return int(Decimal(str(value)) * 1000)


def _expiry_ms(value: float | None) -> int | None:
    if value is None:
        return None
    return int((Decimal(str(value)) * 1000).to_integral_value(rounding=ROUND_CEILING))


def _gpu_us(value: float | int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise JobError("invalid_argument", "GPU seconds must be finite and non-negative")
    return int((Decimal(str(value)) * 1_000_000).to_integral_value(rounding=ROUND_CEILING))


def _rfc3339(ms: int) -> str:
    moment = datetime.fromtimestamp(ms // 1000, tz=timezone.utc) + timedelta(milliseconds=ms % 1000)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _blob_descriptor(data: bytes, *, schema_id: str, allow_empty: bool = False) -> dict[str, Any]:
    if (not data and not allow_empty) or len(data) > contracts.MAX_BLOB_BYTES:
        raise JobError("invalid_argument", "result blob exceeds its bounds")
    return {
        "schema_id": schema_id,
        "media_type": "application/json",
        "size_bytes": len(data),
        "sha256": _sha256(data),
    }


def _result_digest(result: JobResult) -> str:
    digest = hashlib.sha256()
    for value in (result.report, result.evidence):
        if value is None:
            digest.update(b"\xff" * 8)
        else:
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    digest.update(_json({
        "report_descriptor": result.report_descriptor,
        "evidence_descriptor": result.evidence_descriptor,
        "provenance": result.provenance,
        "receipt": result.receipt,
        "signature": result.signature,
    }).encode("utf-8"))
    return digest.hexdigest()


def _assert_local_filesystem(path: Path) -> None:
    """Reject network/automounted state before relying on SQLite and flock."""

    if sys.platform == "darwin":
        try:
            completed = subprocess.run(
                ["/sbin/mount"], text=True, capture_output=True, timeout=3, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise JobError("unavailable", f"cannot inspect state filesystem: {exc}") from exc
        if completed.returncode != 0:
            raise JobError("unavailable", "cannot inspect macOS mount table")
        best: tuple[int, str, str] | None = None
        for line in completed.stdout.splitlines():
            match = re.match(r"^.* on (.*) \(([^,()]+)(?:,([^)]*))?\)$", line)
            if not match:
                continue
            mount = Path(match.group(1))
            if path.is_relative_to(mount) and (best is None or len(str(mount)) > best[0]):
                best = (len(str(mount)), match.group(2), match.group(3) or "")
        if best is None:
            raise JobError("unavailable", "cannot identify the macOS filesystem hosting job state")
        if best[1] in _NETWORK_FILESYSTEMS or best[1] == "autofs" or "local" not in {part.strip() for part in best[2].split(",")}:
            raise JobError("failed_precondition", "job state cannot reside on a network or automounted filesystem")
        return
    mounts = Path("/proc/mounts")
    if not mounts.is_file():
        return
    best: tuple[int, str] | None = None
    try:
        lines = mounts.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise JobError("unavailable", f"cannot inspect state filesystem: {exc}") from exc
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        mount = Path(parts[1].replace("\\040", " "))
        try:
            inside = path.is_relative_to(mount)
        except (OSError, ValueError):
            inside = False
        if inside and (best is None or len(str(mount)) > best[0]):
            best = (len(str(mount)), parts[2])
    if best and best[1] in _NETWORK_FILESYSTEMS:
        raise JobError("failed_precondition", "job state cannot reside on a network filesystem")


def _prepare_state_dir(path: Path) -> Path:
    if not isinstance(path, Path):
        path = Path(path)
    absolute = Path(os.path.abspath(path))
    if absolute == Path(absolute.anchor) or any(absolute.is_relative_to(root) for root in _FORBIDDEN_STATE_ROOTS):
        raise JobError("failed_precondition", "job state requires a dedicated persistent host directory, not tmp")
    if path.exists() and path.is_symlink():
        raise JobError("failed_precondition", "job state directory cannot be a symlink")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise JobError("failed_precondition", "job state path is not a directory")
    if resolved == Path(resolved.anchor) or any(resolved.is_relative_to(root) for root in _FORBIDDEN_STATE_ROOTS):
        raise JobError("failed_precondition", "job state requires a dedicated persistent host directory, not tmp")
    _assert_local_filesystem(resolved)
    info = resolved.stat()
    if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
        raise JobError("failed_precondition", "job state directory is not owned by the frontend account")
    os.chmod(resolved, 0o700)
    if stat.S_IMODE(resolved.stat().st_mode) != 0o700:
        raise JobError("failed_precondition", "job state directory must have mode 0700")
    return resolved


class JobStore:
    """Durable jobs, attempts, grants, capabilities, quota and event window."""

    def __init__(
        self,
        state_dir: Path,
        *,
        frontend_instance: str | None = None,
        event_window: int = 128,
        retention_ms: int = DEFAULT_RETENTION_MS,
        max_queue_depth: int = 1024,
        max_db_bytes: int = 1 << 30,
        max_attempts: int = 3,
        selfcheck_max_timed: int = 20,
        selfcheck_max_gpu_s: float = 120.0,
        selfcheck_gpu_reservation_s: float = 1.0,
        signing_key: bytes | None = None,
        signing_key_id: str = "",
        clock=time.time,
    ) -> None:
        self.state_dir = _prepare_state_dir(state_dir)
        self.frontend_instance = frontend_instance or secrets.token_hex(16)
        if not _FEATURE.fullmatch(self.frontend_instance):
            raise ValueError("frontend_instance must be a bounded opaque identifier")
        for name, value, minimum in (
            ("event_window", event_window, 1), ("retention_ms", retention_ms, 1),
            ("max_queue_depth", max_queue_depth, 1), ("max_db_bytes", max_db_bytes, 1),
            ("max_attempts", max_attempts, 1), ("selfcheck_max_timed", selfcheck_max_timed, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be a positive integer")
        self.event_window = event_window
        self.retention_ms = retention_ms
        self.max_queue_depth = max_queue_depth
        self.max_db_bytes = max_db_bytes
        self.max_attempts = max_attempts
        self.selfcheck_max_timed = selfcheck_max_timed
        self.selfcheck_max_gpu_us = _gpu_us(selfcheck_max_gpu_s)
        self.selfcheck_gpu_reservation_us = max(1, _gpu_us(selfcheck_gpu_reservation_s))
        if signing_key is not None and (not isinstance(signing_key, bytes) or len(signing_key) < 32):
            raise ValueError("signing key must contain at least 32 bytes")
        if signing_key is not None and not signing_key_id:
            raise ValueError("signing_key_id is required with a signing key")
        self._signing_key = signing_key
        self._signing_key_id = signing_key_id
        self._clock = clock
        self._lock = threading.RLock()
        self._closed = False
        self._fatal_error = ""
        self._restored = (self.state_dir / "restored.marker").is_file()
        self._owner_fd = os.open(self.state_dir / "owner.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self._owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self._owner_fd)
            raise JobError("failed_precondition", "another frontend owns this device job store") from exc
        try:
            os.ftruncate(self._owner_fd, 0)
            os.write(self._owner_fd, _json({"pid": os.getpid(), "frontend_instance": self.frontend_instance}).encode("utf-8"))
            os.fsync(self._owner_fd)
            self._conn = sqlite3.connect(
                self.state_dir / "jobs.sqlite3", timeout=5.0, isolation_level=None,
                check_same_thread=False, cached_statements=32,
            )
            self._conn.row_factory = sqlite3.Row
            self._configure()
            self._initialize()
        except Exception:
            self.close()
            raise

    def _configure(self) -> None:
        conn = self._conn
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            if conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() != "wal":
                raise StateCorrupt("SQLite WAL mode could not be enabled")
            conn.execute("PRAGMA synchronous=FULL")
            if conn.execute("PRAGMA synchronous").fetchone()[0] != 2:
                raise StateCorrupt("SQLite synchronous=FULL could not be enabled")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA secure_delete=ON")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA wal_autocheckpoint=256")
            rows = conn.execute("PRAGMA integrity_check").fetchall()
            if len(rows) != 1 or rows[0][0] != "ok":
                raise StateCorrupt("SQLite integrity check failed")
        except sqlite3.DatabaseError as exc:
            raise StateCorrupt(f"cannot open durable job state: {exc}") from exc

    def _ensure_available(self) -> None:
        if self._closed:
            raise JobError("unavailable", "job store is closed")
        if self._fatal_error:
            raise JobError(
                "unavailable", "job store is unavailable after a fatal storage error",
                retryable=True,
            )

    def mark_unavailable(self, error: BaseException) -> None:
        with self._lock:
            self._fatal_error = f"{type(error).__name__}: {error}"[:1000]

    @property
    def unavailable(self) -> bool:
        return bool(self._fatal_error)

    def _rollback_after_error(self) -> None:
        try:
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK")
        except sqlite3.DatabaseError as exc:
            self._fatal_error = f"SQLite rollback failed: {exc}"[:1000]
            # Closing the connection is the only reliable way to guarantee a
            # failed COMMIT/ROLLBACK cannot leave an open transaction behind.
            self._conn.close()

    @contextlib.contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._ensure_available()
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                yield self._conn
                self._conn.execute("COMMIT")
            except sqlite3.DatabaseError as exc:
                self._rollback_after_error()
                self._fatal_error = f"SQLite write failed: {exc}"[:1000]
                raise StateCorrupt(f"durable job-store write failed: {exc}") from exc
            except Exception:
                self._rollback_after_error()
                raise

    def _initialize(self) -> None:
        # Schema and user_version commit atomically; an initialization error
        # must not leave an implicit transaction open on the owner connection.
        with self._lock:
            conn = self._conn
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise StateCorrupt(f"unsupported job-store schema version {version}")
            if version == 0 and conn.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1").fetchone():
                raise StateCorrupt("unversioned job-store tables are not accepted")
            try:
                conn.executescript(f"""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS counters (
                    name TEXT PRIMARY KEY, value INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_counters (
                    run_id TEXT NOT NULL, session_id TEXT NOT NULL, value INTEGER NOT NULL,
                    PRIMARY KEY(run_id, session_id)
                );
                CREATE TABLE IF NOT EXISTS grants (
                    token_sha256 TEXT PRIMARY KEY, scope_id TEXT NOT NULL, subject TEXT NOT NULL,
                    scope_json TEXT NOT NULL, generation INTEGER NOT NULL,
                    expires_at_ms INTEGER, max_timed INTEGER NOT NULL, max_gpu_us INTEGER NOT NULL,
                    status TEXT NOT NULL, created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS grants_scope ON grants(scope_id, status);
                CREATE TABLE IF NOT EXISTS capabilities (
                    capability_id TEXT PRIMARY KEY, client_nonce TEXT NOT NULL, server_nonce TEXT NOT NULL,
                    token_sha256 TEXT NOT NULL, scope_id TEXT NOT NULL, generation INTEGER NOT NULL,
                    frontend_instance TEXT NOT NULL, expected_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL, features_json TEXT NOT NULL,
                    issued_at_ms INTEGER NOT NULL, expires_at_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    request_id TEXT PRIMARY KEY, request_digest TEXT NOT NULL,
                    request_json TEXT NOT NULL, source BLOB,
                    state TEXT NOT NULL, expected_json TEXT NOT NULL, provenance_json TEXT NOT NULL,
                    evaluator_epoch TEXT NOT NULL, api_contract_id TEXT NOT NULL, evidence_contract_id TEXT NOT NULL,
                    grant_subject TEXT NOT NULL, grant_scope_id TEXT NOT NULL, grant_scope_json TEXT NOT NULL,
                    current_attempt_id TEXT, capability_id TEXT NOT NULL DEFAULT '',
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    quarantined INTEGER NOT NULL DEFAULT 0, recovery_error TEXT NOT NULL DEFAULT '',
                    accepted_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL, expires_at_ms INTEGER,
                    last_sequence INTEGER NOT NULL DEFAULT 0,
                    failure_json TEXT
                );
                CREATE INDEX IF NOT EXISTS jobs_schedule ON jobs(state, quarantined, accepted_at_ms);
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY, request_id TEXT NOT NULL REFERENCES jobs(request_id),
                    lease_fence INTEGER NOT NULL, retry_index INTEGER NOT NULL, state TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0, queued_at_ms INTEGER NOT NULL,
                    started_at_ms INTEGER, finished_at_ms INTEGER, deadline_at_ms INTEGER,
                    session_sequence INTEGER NOT NULL, capability_id TEXT NOT NULL DEFAULT '',
                    frontend_instance TEXT NOT NULL DEFAULT '',
                    worker_pid INTEGER, worker_pgid INTEGER, worker_start_token TEXT NOT NULL DEFAULT '',
                    failure_json TEXT, diagnostic_json TEXT,
                    UNIQUE(request_id, lease_fence), UNIQUE(request_id, retry_index)
                );
                CREATE INDEX IF NOT EXISTS attempts_active ON attempts(state, request_id);
                CREATE TABLE IF NOT EXISTS quotas (
                    run_id TEXT NOT NULL, session_id TEXT NOT NULL, kind TEXT NOT NULL, target_id TEXT NOT NULL,
                    reserved_count INTEGER NOT NULL DEFAULT 0, reserved_gpu_us INTEGER NOT NULL DEFAULT 0,
                    used_count INTEGER NOT NULL DEFAULT 0, used_gpu_us INTEGER NOT NULL DEFAULT 0,
                    hang INTEGER NOT NULL DEFAULT 0, updated_at_ms INTEGER NOT NULL,
                    PRIMARY KEY(run_id, session_id, kind, target_id)
                );
                CREATE TABLE IF NOT EXISTS reservations (
                    request_id TEXT PRIMARY KEY REFERENCES jobs(request_id),
                    run_id TEXT NOT NULL, session_id TEXT NOT NULL, kind TEXT NOT NULL, target_id TEXT NOT NULL,
                    count INTEGER NOT NULL, gpu_us INTEGER NOT NULL, state TEXT NOT NULL,
                    actual_count INTEGER NOT NULL DEFAULT 0, actual_gpu_us INTEGER NOT NULL DEFAULT 0,
                    updated_at_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS results (
                    request_id TEXT PRIMARY KEY REFERENCES jobs(request_id), attempt_id TEXT NOT NULL UNIQUE,
                    receipt_id TEXT NOT NULL UNIQUE, result_sha256 TEXT NOT NULL,
                    report BLOB NOT NULL, report_descriptor_json TEXT NOT NULL,
                    evidence BLOB, evidence_descriptor_json TEXT,
                    provenance_json TEXT NOT NULL, receipt_json TEXT NOT NULL, signature_json TEXT,
                    created_at_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS receipts (
                    receipt_id TEXT PRIMARY KEY, request_id TEXT NOT NULL, attempt_id TEXT NOT NULL,
                    request_digest TEXT NOT NULL, result_sha256 TEXT NOT NULL, created_at_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    request_id TEXT NOT NULL REFERENCES jobs(request_id), sequence INTEGER NOT NULL,
                    attempt_id TEXT, lease_fence INTEGER NOT NULL, kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL, created_at_ms INTEGER NOT NULL,
                    PRIMARY KEY(request_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS events_attempt ON events(request_id, attempt_id, sequence);
                INSERT OR IGNORE INTO counters(name, value) VALUES('lease_fence', 0);
                PRAGMA user_version={SCHEMA_VERSION};
                COMMIT;
            """)
            except sqlite3.DatabaseError:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        corrupt_requests = self._validate_durable_state()
        if corrupt_requests:
            now = _now_ms(self._clock)
            with self._write() as conn:
                for request_id, message in corrupt_requests:
                    conn.execute(
                        "UPDATE jobs SET quarantined=1,recovery_error=?,updated_at_ms=? WHERE request_id=?",
                        (message[:1000], now, request_id),
                    )
        if self._restored:
            with self._write() as conn:
                conn.execute("UPDATE jobs SET quarantined=1 WHERE state NOT IN ('completed','failed','cancelled','expired')")

    def _validate_durable_state(self) -> list[tuple[str, str]]:
        corrupt_requests: list[tuple[str, str]] = []
        with self._lock:
            try:
                jobs = self._conn.execute("SELECT * FROM jobs").fetchall()
                for job in jobs:
                    if job["state"] not in JOB_STATES or not re.fullmatch(r"[0-9a-f]{64}", job["request_digest"]):
                        raise StateCorrupt("invalid durable job state")
                    try:
                        expected = ExpectedIdentities.from_dict(_loads(job["expected_json"]))
                    except (JobError, TypeError, ValueError, KeyError) as exc:
                        raise StateCorrupt("invalid durable expected identities") from exc
                    if (
                        expected.evaluator.evaluator_epoch != job["evaluator_epoch"]
                        or expected.api_contract_id != job["api_contract_id"]
                        or expected.evidence_contract_id != job["evidence_contract_id"]
                    ):
                        raise StateCorrupt("job identity columns disagree with the expected document")
                    if job["source"] is not None:
                        try:
                            self._request(self._conn, job)
                        except (JobError, TypeError, ValueError, KeyError) as exc:
                            corrupt_requests.append((
                                job["request_id"],
                                f"stored request bytes do not match request_digest: {exc}",
                            ))
                    attempts = self._conn.execute("SELECT * FROM attempts WHERE request_id=?", (job["request_id"],)).fetchall()
                    if any(row["state"] not in ATTEMPT_STATES for row in attempts):
                        raise StateCorrupt("invalid durable attempt state")
                    by_id = {row["attempt_id"]: row for row in attempts}
                    current = by_id.get(job["current_attempt_id"])
                    if job["state"] == "accepted":
                        if job["current_attempt_id"] is not None:
                            raise StateCorrupt("accepted job unexpectedly has an attempt")
                    elif job["current_attempt_id"] is None or current is None:
                        raise StateCorrupt("job references a missing current attempt")
                    active = [row for row in attempts if row["state"] in {"queued", "running"}]
                    if len(active) > 1:
                        raise StateCorrupt("a job has multiple active attempts")
                    expected_attempt_state = {
                        "queued": "queued", "running": "running", "interrupted": "interrupted",
                        "completed": "completed", "failed": "failed", "cancelled": "cancelled",
                    }.get(job["state"])
                    if expected_attempt_state is not None and (current is None or current["state"] != expected_attempt_state):
                        raise StateCorrupt("job and current attempt states disagree")
                    if job["state"] == "completed":
                        result = self._conn.execute("SELECT * FROM results WHERE request_id=?", (job["request_id"],)).fetchone()
                        receipt = self._conn.execute("SELECT * FROM receipts WHERE request_id=?", (job["request_id"],)).fetchone()
                        if result is None or receipt is None:
                            raise StateCorrupt("completed job has no durable result/receipt")
                        if (
                            result["attempt_id"] != job["current_attempt_id"]
                            or receipt["attempt_id"] != job["current_attempt_id"]
                            or result["receipt_id"] != receipt["receipt_id"]
                            or result["result_sha256"] != receipt["result_sha256"]
                            or receipt["request_digest"] != job["request_digest"]
                        ):
                            raise StateCorrupt("completed result/receipt binding is inconsistent")
                        provenance = _loads(result["provenance_json"])
                        if (
                            provenance.get("evaluator_epoch") != job["evaluator_epoch"]
                            or provenance.get("api_contract_id") != job["api_contract_id"]
                            or provenance.get("evidence_contract_id") != job["evidence_contract_id"]
                            or provenance.get("request_digest") != job["request_digest"]
                        ):
                            raise StateCorrupt("result provenance differs from the accepted job identities")
                        self._result_from_row(result, job=job, attempt=current)
                    if job["state"] in {"accepted", "queued", "running", "interrupted"} and job["source"] is None:
                        raise StateCorrupt("nonterminal job has no exact candidate bytes")
                    if job["state"] in TERMINAL_JOB_STATES and job["source"] is not None:
                        raise StateCorrupt("terminal job retained raw source unexpectedly")
                running = self._conn.execute("SELECT COUNT(*) FROM jobs WHERE state='running'").fetchone()[0]
                if running > 1:
                    raise StateCorrupt("more than one device attempt is running")
                for row in self._conn.execute("SELECT * FROM quotas"):
                    if any(row[key] < 0 for key in ("reserved_count", "reserved_gpu_us", "used_count", "used_gpu_us")):
                        raise StateCorrupt("negative durable quota accounting")
                return corrupt_requests
            except sqlite3.DatabaseError as exc:
                raise StateCorrupt(f"invalid durable job state: {exc}") from exc

    def close(self) -> None:
        with getattr(self, "_lock", threading.RLock()):
            if getattr(self, "_closed", True):
                return
            self._closed = True
            try:
                conn = getattr(self, "_conn", None)
                if conn is not None:
                    conn.close()
            finally:
                fd = getattr(self, "_owner_fd", None)
                if fd is not None:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    except OSError:
                        pass
                    os.close(fd)
                    self._owner_fd = None

    def __enter__(self) -> "JobStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def database_bytes(self) -> int:
        with self._lock:
            total = 0
            for suffix in ("", "-wal", "-shm"):
                try:
                    total += (self.state_dir / ("jobs.sqlite3" + suffix)).stat().st_size
                except FileNotFoundError:
                    pass
            return total

    def _ensure_db_capacity(self, additional_bytes: int) -> None:
        if isinstance(additional_bytes, bool) or not isinstance(additional_bytes, int) or additional_bytes < 0:
            raise ValueError("projected database growth must be a non-negative integer")
        if self.database_bytes() + additional_bytes + DB_WRITE_SLACK_BYTES > self.max_db_bytes:
            raise JobError(
                "resource_exhausted",
                "job-store byte limit would be exceeded by this durable write",
            )

    # -- grants and capabilities -------------------------------------------
    def _grant_row(self, conn: sqlite3.Connection, token: str, now_ms: int) -> sqlite3.Row:
        if not isinstance(token, str) or not token:
            raise JobError("unauthenticated", "session grant is required")
        row = conn.execute("SELECT * FROM grants WHERE token_sha256=?", (_sha256(token.encode("utf-8")),)).fetchone()
        if row is None:
            raise JobError("unauthenticated", "session grant was refused")
        if row["status"] == "revoked":
            raise JobError("permission_denied", "session grant was revoked")
        if row["status"] != "active":
            raise JobError("unauthenticated", "session grant was superseded; use the renewed grant")
        if row["expires_at_ms"] is not None and now_ms >= row["expires_at_ms"]:
            raise JobError("unauthenticated", "session grant expired")
        return row

    def register_grant(self, grant: SessionGrant, *, renew: bool = False) -> str:
        if not isinstance(grant, SessionGrant):
            raise TypeError("grant must be a SessionGrant")
        scope = grant.scope_document()
        scope_id = _sha256(_json(scope).encode("utf-8"))
        token_hash = _sha256(grant.token.encode("utf-8"))
        now = _now_ms(self._clock)
        with self._write() as conn:
            existing_token = conn.execute("SELECT * FROM grants WHERE token_sha256=?", (token_hash,)).fetchone()
            if existing_token is not None:
                if existing_token["scope_id"] != scope_id:
                    raise JobError("permission_denied", "token hash is already bound to another grant scope")
                if existing_token["status"] == "active":
                    return scope_id
                if not renew:
                    raise JobError("permission_denied", "grant token is no longer active")
                raise JobError("permission_denied", "a revoked or superseded token cannot be renewed")
            existing_scope = conn.execute(
                "SELECT * FROM grants WHERE scope_id=? AND status='active'", (scope_id,)
            ).fetchall()
            if existing_scope and not renew:
                raise JobError("already_exists", "an active grant already exists for this scope")
            if renew and not existing_scope:
                raise JobError("failed_precondition", "cannot renew an unknown grant scope")
            generation = 1 + max([row["generation"] for row in existing_scope] or [0])
            if renew:
                conn.execute(
                    "UPDATE grants SET status='superseded',updated_at_ms=? WHERE scope_id=? AND status='active'",
                    (now, scope_id),
                )
            values = (
                token_hash, scope_id, grant.grant_subject, _json(scope), generation,
                _expiry_ms(grant.expires_at), grant.max_timed, _gpu_us(grant.max_gpu_s),
                "active", now, now,
            )
            self._ensure_db_capacity(sum(len(str(value).encode("utf-8")) for value in values) + 2048)
            conn.execute("INSERT INTO grants VALUES(?,?,?,?,?,?,?,?,?,?,?)", values)
            return scope_id

    def renew_grant(self, grant: SessionGrant) -> str:
        return self.register_grant(grant, renew=True)

    def revoke_grant(self, token: str) -> str:
        token_hash = _sha256(token.encode("utf-8")) if isinstance(token, str) else ""
        with self._write() as conn:
            row = conn.execute("SELECT * FROM grants WHERE token_sha256=?", (token_hash,)).fetchone()
            if row is None:
                raise JobError("unauthenticated", "session grant was refused")
            conn.execute("UPDATE grants SET status='revoked',updated_at_ms=? WHERE token_sha256=?", (_now_ms(self._clock), token_hash))
            conn.execute("UPDATE capabilities SET expires_at_ms=0 WHERE token_sha256=?", (token_hash,))
            return row["scope_id"]

    def scope_jobs(self, scope_id: str) -> list[str]:
        with self._lock:
            return [row[0] for row in self._conn.execute(
                "SELECT request_id FROM jobs WHERE grant_scope_id=? AND state IN ('accepted','queued','running','interrupted')",
                (scope_id,),
            )]

    @staticmethod
    def _features(values) -> frozenset[str]:
        if not isinstance(values, (list, tuple, frozenset, set)):
            raise JobError("invalid_argument", "required_features must be a bounded set")
        result = frozenset(values)
        if any(not isinstance(value, str) or not _FEATURE.fullmatch(value) for value in result):
            raise JobError("invalid_argument", "feature token is invalid")
        return result

    def get_capabilities(
        self,
        *,
        grant_token: str,
        client_nonce: str,
        expected: Mapping[str, Any],
        expected_identities: ExpectedIdentities,
        provenance: dict[str, Any],
        api_contract_ids: list[str],
        evidence_contract_ids: list[str],
        required_features: list[str] | tuple[str, ...],
        supported_features: frozenset[str],
        valid_for_ms: int = 60_000,
        limits: dict[str, int] | None = None,
    ) -> Capabilities:
        if not isinstance(client_nonce, str) or not 16 <= len(client_nonce) <= 128:
            raise JobError("invalid_argument", "client_nonce must be a bounded high-entropy nonce")
        if isinstance(valid_for_ms, bool) or not isinstance(valid_for_ms, int) or not 1 <= valid_for_ms <= 300_000:
            raise JobError("invalid_argument", "valid_for_ms is outside the capability bounds")
        expected_identities.validate()
        actual = expected_identities.evaluator.to_dict()
        if not isinstance(expected, Mapping):
            raise JobError("invalid_argument", "capabilities expected evaluator must be an object")
        if set(expected) - set(actual):
            raise JobError("invalid_argument", "unknown capabilities expectation")
        for key, value in expected.items():
            if value and actual[key] != value:
                raise JobError("failed_precondition", f"expected evaluator {key} does not match")
        if expected_identities.api_contract_id not in api_contract_ids or expected_identities.evidence_contract_id not in evidence_contract_ids:
            raise JobError("failed_precondition", "client does not accept the selected API/evidence contract")
        features = self._features(required_features)
        supported = self._features(supported_features)
        missing = features - supported
        if missing:
            raise JobError("failed_precondition", f"unsupported required features: {sorted(missing)}")
        now = _now_ms(self._clock)
        capability_id = secrets.token_hex(16)
        server_nonce = secrets.token_hex(16)
        self.expire_jobs(now_ms=now)
        with self._write() as conn:
            grant = self._grant_row(conn, grant_token, now)
            if conn.execute(
                "SELECT 1 FROM capabilities WHERE token_sha256=? AND client_nonce=?",
                (grant["token_sha256"], client_nonce),
            ).fetchone():
                raise JobError("failed_precondition", "client_nonce was already used for this grant")
            values = (
                capability_id, client_nonce, server_nonce, grant["token_sha256"], grant["scope_id"],
                grant["generation"], self.frontend_instance, _json(expected_identities.to_dict()),
                _json(provenance), _json(sorted(features)), now, now + valid_for_ms,
            )
            self._ensure_db_capacity(sum(len(str(value).encode("utf-8")) for value in values) + 2048)
            conn.execute("INSERT INTO capabilities VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", values)
        return Capabilities(
            client_nonce=client_nonce, server_nonce=server_nonce, capability_id=capability_id,
            expected=expected_identities, provenance=dict(provenance), supported_features=supported,
            required_features=features, issued_at_ms=now, valid_for_ms=valid_for_ms,
            limits=dict(limits or {}), signing_key_id=self._signing_key_id,
        )

    def _capability(
        self, conn: sqlite3.Connection, capability_id: str, grant: sqlite3.Row,
        expected: ExpectedIdentities, features: frozenset[str], now_ms: int,
    ) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM capabilities WHERE capability_id=?", (capability_id,)).fetchone()
        if row is None or now_ms >= row["expires_at_ms"] or row["frontend_instance"] != self.frontend_instance:
            raise JobError("failed_precondition", "capability is expired or belongs to an old frontend; re-hello")
        if row["token_sha256"] != grant["token_sha256"] or row["scope_id"] != grant["scope_id"] or row["generation"] != grant["generation"]:
            raise JobError("failed_precondition", "capability is not bound to the current grant; re-hello")
        if row["expected_json"] != _json(expected.to_dict()) or self._features(_loads(row["features_json"])) != features:
            raise JobError("failed_precondition", "capability identities or required features differ")
        return row

    # -- request and snapshot helpers --------------------------------------
    @staticmethod
    def _request(conn: sqlite3.Connection, job: sqlite3.Row) -> ExternalEvaluationRequest:
        if job["source"] is None:
            raise StateCorrupt("active job source bytes are unavailable")
        try:
            request = ExternalEvaluationRequest.from_stored(_loads(job["request_json"]), bytes(job["source"]))
        except StateCorrupt:
            raise
        except (JobError, TypeError, ValueError, KeyError) as exc:
            raise StateCorrupt(f"stored request is invalid: {exc}") from exc
        if request.request_id != job["request_id"] or request.request_digest != job["request_digest"]:
            raise StateCorrupt("stored request bytes/source do not match request_id/request_digest")
        return request

    @staticmethod
    def _attempt(row: sqlite3.Row | None) -> Attempt | None:
        if row is None:
            return None
        return Attempt(
            attempt_id=row["attempt_id"], lease_fence=row["lease_fence"], state=row["state"],
            retry_index=row["retry_index"], cancel_requested=bool(row["cancel_requested"]),
            queued_at_ms=row["queued_at_ms"], started_at_ms=row["started_at_ms"],
            finished_at_ms=row["finished_at_ms"], failure=JobFailure.from_dict(_loads(row["failure_json"]) if row["failure_json"] else None),
            session_sequence=row["session_sequence"], worker_pid=row["worker_pid"], worker_pgid=row["worker_pgid"],
            worker_start_token=row["worker_start_token"], frontend_instance=row["frontend_instance"],
        )

    @staticmethod
    def _result_document(result: JobResult) -> dict[str, Any]:
        return {
            "provenance": result.provenance,
            "report": result.report_descriptor,
            "evidence": result.evidence_descriptor,
            "receipt": result.receipt,
        }

    def _validate_result_envelope(
        self, result: JobResult, *, job: sqlite3.Row, attempt: sqlite3.Row,
    ) -> None:
        try:
            contracts.validate_blob(result.report_descriptor, result.report)
            if (result.evidence is None) != (result.evidence_descriptor is None):
                raise StateCorrupt("result evidence bytes and descriptor presence differ")
            if result.evidence_descriptor is not None:
                contracts.validate_blob(result.evidence_descriptor, result.evidence or b"", allow_empty=True)
            signature = result.signature
            if signature is not None:
                if not isinstance(signature, Mapping) or set(signature) != {"algorithm", "key_id", "signature"}:
                    raise StateCorrupt("result signature fields are missing or unknown")
                if signature["algorithm"] != contracts.RESULT_SIGNATURE_ALGORITHM:
                    raise StateCorrupt("result signature algorithm is unsupported")
                if not isinstance(signature["signature"], str) or not re.fullmatch(r"[0-9a-f]{64}", signature["signature"]):
                    raise StateCorrupt("result signature must be 32 bytes of lowercase hex")
                contracts.result_signature_document(self._result_document(result), key_id=signature["key_id"])
            else:
                # Unsigned results still require the complete typed envelope.
                # Use the same contract validators without inventing a signing
                # key ID merely to call result_signature_document.
                provenance = contracts._provenance_document(result.provenance)
                receipt = contracts._receipt_document(result.receipt)
                if provenance["attempt_id"] != receipt["attempt_id"] or provenance["request_digest"] != receipt["request_digest"]:
                    raise StateCorrupt("unsigned result provenance/receipt binding is inconsistent")
        except StateCorrupt:
            raise
        except (contracts.ContractError, TypeError, ValueError, KeyError) as exc:
            raise StateCorrupt(f"persisted result violates the external evidence contract: {exc}") from exc

        provenance = result.provenance
        receipt = result.receipt
        request_document = _loads(job["request_json"])
        if not isinstance(request_document, Mapping):
            raise StateCorrupt("stored request document is not an object")
        if (
            provenance.get("evaluator_epoch") != job["evaluator_epoch"]
            or provenance.get("api_contract_id") != job["api_contract_id"]
            or provenance.get("evidence_contract_id") != job["evidence_contract_id"]
            or provenance.get("request_digest") != job["request_digest"]
            or provenance.get("attempt_id") != attempt["attempt_id"]
            or receipt.get("request_id") != job["request_id"]
            or receipt.get("attempt_id") != attempt["attempt_id"]
            or receipt.get("request_digest") != job["request_digest"]
            or receipt.get("candidate_sha256") != request_document.get("candidate_sha256")
            or receipt.get("lease_fence") != str(attempt["lease_fence"])
            or receipt.get("session_sequence") != str(attempt["session_sequence"])
        ):
            raise StateCorrupt("result provenance/receipt differs from the accepted job and attempt")
        if signature is not None and self._signing_key is not None:
            if signature["key_id"] != self._signing_key_id or not contracts.verify_result_signature(
                self._result_document(result), key_id=self._signing_key_id,
                key=self._signing_key, signature=signature["signature"],
            ):
                raise StateCorrupt("stored result signature does not verify with the configured key")

    def _result_from_row(
        self, row: sqlite3.Row, *, job: sqlite3.Row, attempt: sqlite3.Row,
    ) -> JobResult:
        evidence = None if row["evidence"] is None else bytes(row["evidence"])
        result = JobResult(
            report=bytes(row["report"]), report_descriptor=_loads(row["report_descriptor_json"]),
            evidence=evidence,
            evidence_descriptor=_loads(row["evidence_descriptor_json"]) if row["evidence_descriptor_json"] else None,
            provenance=_loads(row["provenance_json"]), receipt=_loads(row["receipt_json"]),
            signature=_loads(row["signature_json"]) if row["signature_json"] else None,
        )
        self._validate_result_envelope(result, job=job, attempt=attempt)
        if row["result_sha256"] != _result_digest(result):
            raise StateCorrupt("stored result envelope digest does not match its complete contents")
        return result

    def _replay_result(
        self, stored: JobResult, report: bytes, evidence: bytes | None,
        *, signing_key: bytes | None, signing_key_id: str,
    ) -> JobResult:
        try:
            contracts.validate_blob(stored.report_descriptor, report)
            if (evidence is None) != (stored.evidence is None):
                raise StateCorrupt("same attempt changed evidence null/presence")
            if stored.evidence_descriptor is not None:
                contracts.validate_blob(stored.evidence_descriptor, evidence or b"", allow_empty=True)
        except contracts.ContractError as exc:
            raise StateCorrupt("same attempt/receipt produced different result bytes") from exc
        signature = self._make_signature(
            self._result_document(stored), signing_key=signing_key, signing_key_id=signing_key_id,
        )
        return JobResult(
            report, dict(stored.report_descriptor), evidence,
            None if stored.evidence_descriptor is None else dict(stored.evidence_descriptor),
            dict(stored.provenance), dict(stored.receipt), signature,
        )

    @staticmethod
    def _make_signature(
        document: Mapping[str, Any], *, signing_key: bytes | None, signing_key_id: str,
    ) -> dict[str, Any] | None:
        if signing_key is None:
            return None
        if not signing_key_id:
            raise ValueError("signing_key_id is required with a signing key")
        signature_hex = contracts.sign_result(document, key_id=signing_key_id, key=signing_key)
        return {
            "algorithm": contracts.RESULT_SIGNATURE_ALGORITHM,
            "key_id": signing_key_id,
            "signature": signature_hex,
        }

    def _snapshot(self, conn: sqlite3.Connection, job: sqlite3.Row) -> JobSnapshot:
        attempt_row = None
        if job["current_attempt_id"]:
            attempt_row = conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (job["current_attempt_id"],)).fetchone()
            if attempt_row is None:
                raise StateCorrupt("job references a missing attempt")
        attempt = self._attempt(attempt_row)
        provenance = dict(_loads(job["provenance_json"]))
        provenance["request_digest"] = job["request_digest"]
        if attempt is not None:
            provenance["attempt_id"] = attempt.attempt_id
        result = None
        if job["state"] == "completed":
            row = conn.execute("SELECT * FROM results WHERE request_id=?", (job["request_id"],)).fetchone()
            if row is None:
                raise StateCorrupt("completed job result is missing")
            if attempt_row is None:
                raise StateCorrupt("completed job has no current attempt row")
            result = self._result_from_row(row, job=job, attempt=attempt_row)
        return JobSnapshot(
            request_id=job["request_id"], request_digest=job["request_digest"], state=job["state"],
            provenance=provenance, current_attempt=attempt, result=result,
            failure=JobFailure.from_dict(_loads(job["failure_json"]) if job["failure_json"] else None),
            cancel_requested=bool(job["cancel_requested"]), last_sequence=job["last_sequence"],
            accepted_at_ms=job["accepted_at_ms"], updated_at_ms=job["updated_at_ms"],
            expires_at_ms=job["expires_at_ms"], quarantined=bool(job["quarantined"]),
            recovery_error=job["recovery_error"],
        )

    def _scoped_job(self, conn: sqlite3.Connection, token: str, request_id: str, now_ms: int) -> tuple[sqlite3.Row, sqlite3.Row]:
        grant = self._grant_row(conn, token, now_ms)
        job = conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
        if job is None or job["grant_scope_id"] != grant["scope_id"] or job["grant_subject"] != grant["subject"]:
            raise JobError("not_found", "job is not visible in this grant scope", request_id=request_id)
        return grant, job

    @staticmethod
    def _check_ref(job: sqlite3.Row, request_digest: str, expected: ExpectedIdentities) -> None:
        if job["request_digest"] != request_digest:
            raise JobError("failed_precondition", "job request digest differs", request_id=job["request_id"])
        if job["expected_json"] != _json(expected.to_dict()):
            raise JobError("failed_precondition", "job evaluator/API/evidence identities differ", request_id=job["request_id"])

    def _job_by_id(self, request_id: str) -> JobSnapshot:
        with self._lock:
            self._ensure_available()
            row = self._conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
            if row is None:
                raise JobError("not_found", "job does not exist", request_id=request_id)
            return self._snapshot(self._conn, row)

    # -- quota --------------------------------------------------------------
    @staticmethod
    def _quota_key(request: ExternalEvaluationRequest) -> tuple[str, str, str, str]:
        return request.run_id, request.session_id, request.kind, request.target_id

    def _reserve_quota(self, conn: sqlite3.Connection, request: ExternalEvaluationRequest, grant: sqlite3.Row, now: int) -> None:
        existing = conn.execute("SELECT * FROM reservations WHERE request_id=?", (request.request_id,)).fetchone()
        if existing is not None and existing["state"] == "reserved":
            return
        if request.kind != "selfcheck" or request.mode != "timed":
            return
        key = self._quota_key(request)
        quota = conn.execute("SELECT * FROM quotas WHERE run_id=? AND session_id=? AND kind=? AND target_id=?", key).fetchone()
        values = {key_: (quota[key_] if quota else 0) for key_ in ("reserved_count", "reserved_gpu_us", "used_count", "used_gpu_us", "hang")}
        if values["hang"]:
            raise JobError("failed_precondition", "self-check timed mode is blocked after a candidate hang")
        count_limit = min(self.selfcheck_max_timed, grant["max_timed"])
        gpu_limit = min(self.selfcheck_max_gpu_us, grant["max_gpu_us"])
        if values["reserved_count"] + values["used_count"] + 1 > count_limit:
            raise JobError("resource_exhausted", "self-check timed count quota is exhausted")
        remaining_gpu = gpu_limit - values["reserved_gpu_us"] - values["used_gpu_us"]
        if remaining_gpu <= 0:
            raise JobError("resource_exhausted", "self-check GPU-time quota is exhausted")
        reserve_gpu = min(self.selfcheck_gpu_reservation_us, remaining_gpu)
        conn.execute("""
            INSERT INTO quotas(run_id,session_id,kind,target_id,reserved_count,reserved_gpu_us,used_count,used_gpu_us,hang,updated_at_ms)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id,session_id,kind,target_id) DO UPDATE SET
              reserved_count=reserved_count+1,reserved_gpu_us=reserved_gpu_us+excluded.reserved_gpu_us,updated_at_ms=excluded.updated_at_ms
        """, (*key, 1, reserve_gpu, 0, 0, 0, now))
        conn.execute("DELETE FROM reservations WHERE request_id=?", (request.request_id,))
        conn.execute("INSERT INTO reservations VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
            request.request_id, *key, 1, reserve_gpu, "reserved", 0, 0, now,
        ))

    def _release_reservation(
        self, conn: sqlite3.Connection, request_id: str, now: int,
        *, actual_count: int = 0, actual_gpu_us: int = 0, hang: bool = False,
    ) -> None:
        reservation = conn.execute("SELECT * FROM reservations WHERE request_id=?", (request_id,)).fetchone()
        if reservation is None or reservation["state"] != "reserved":
            return
        key = (reservation["run_id"], reservation["session_id"], reservation["kind"], reservation["target_id"])
        conn.execute("""
            UPDATE quotas SET reserved_count=reserved_count-?,reserved_gpu_us=reserved_gpu_us-?,
              used_count=used_count+?,used_gpu_us=used_gpu_us+?,hang=MAX(hang,?),updated_at_ms=?
            WHERE run_id=? AND session_id=? AND kind=? AND target_id=?
        """, (reservation["count"], reservation["gpu_us"], actual_count, actual_gpu_us,
              int(hang), now, *key))
        state = "settled" if actual_count else "refunded"
        conn.execute("UPDATE reservations SET state=?,actual_count=?,actual_gpu_us=?,updated_at_ms=? WHERE request_id=?", (
            state, actual_count, actual_gpu_us, now, request_id,
        ))
        if not actual_count:
            conn.execute("""
                DELETE FROM quotas WHERE run_id=? AND session_id=? AND kind=? AND target_id=?
                  AND reserved_count=0 AND reserved_gpu_us=0 AND used_count=0 AND used_gpu_us=0 AND hang=0
            """, key)

    def quota_usage(self, request: ExternalEvaluationRequest) -> QuotaUsage:
        key = self._quota_key(request)
        with self._lock:
            row = self._conn.execute("SELECT * FROM quotas WHERE run_id=? AND session_id=? AND kind=? AND target_id=?", key).fetchone()
            if row is None:
                return QuotaUsage(0, 0, 0, 0, False)
            return QuotaUsage(row["reserved_count"], row["reserved_gpu_us"], row["used_count"], row["used_gpu_us"], bool(row["hang"]))

    # -- submit/get/retry/cancel -------------------------------------------
    def submit(
        self, *, request: ExternalEvaluationRequest, request_digest: str,
        expected: ExpectedIdentities, provenance: dict[str, Any], capability_id: str,
        required_features: list[str] | tuple[str, ...], grant_token: str,
    ) -> tuple[JobSnapshot, bool]:
        if request.request_digest != request_digest:
            raise JobError("invalid_argument", "request digest does not match the normalized request")
        expected.validate()
        features = self._features(required_features)
        now = _now_ms(self._clock)
        self.expire_jobs(now_ms=now)
        with self._write() as conn:
            grant = self._grant_row(conn, grant_token, now)
            self._capability(conn, capability_id, grant, expected, features, now)
            scope = _loads(grant["scope_json"])
            if request.kind not in scope["kinds"] or request.mode not in scope["modes"] or request.run_id != scope["run_id"] or request.session_id != scope["session_id"] or request.target_id != scope["target_id"]:
                raise JobError("permission_denied", "request is outside the grant run/session/kind/target/mode scope")
            if request.target_id != expected.evaluator.target_id:
                raise JobError("failed_precondition", "request target differs from expected evaluator")
            existing = conn.execute("SELECT * FROM jobs WHERE request_id=?", (request.request_id,)).fetchone()
            if existing is not None:
                if existing["grant_scope_id"] != grant["scope_id"] or existing["grant_subject"] != grant["subject"]:
                    raise JobError("not_found", "job is not visible in this grant scope", request_id=request.request_id)
                if existing["request_digest"] != request_digest:
                    raise JobError("already_exists", "request_id was reused with a different digest", request_id=request.request_id)
                if existing["expected_json"] != _json(expected.to_dict()) or existing["grant_scope_id"] != grant["scope_id"]:
                    raise JobError("failed_precondition", "idempotent replay identities or grant scope differ", request_id=request.request_id)
                return self._snapshot(conn, existing), False
            depth = conn.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('accepted','queued','running')").fetchone()[0]
            if depth >= self.max_queue_depth:
                raise JobError("resource_exhausted", "device queue depth limit is exhausted")
            request_json = _json(request.without_source())
            self._ensure_db_capacity(len(request.source) + len(request_json.encode("utf-8")) + 8192)
            conn.execute("""
                INSERT INTO jobs(request_id,request_digest,request_json,source,state,expected_json,provenance_json,
                  evaluator_epoch,api_contract_id,evidence_contract_id,grant_subject,grant_scope_id,grant_scope_json,
                  capability_id,accepted_at_ms,updated_at_ms,last_sequence)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)
            """, (request.request_id, request_digest, request_json, sqlite3.Binary(request.source),
                  "accepted", _json(expected.to_dict()), _json(provenance), expected.evaluator.evaluator_epoch,
                  expected.api_contract_id, expected.evidence_contract_id, grant["subject"], grant["scope_id"],
                  grant["scope_json"], capability_id, now, now))
            # The reservation references the job row; both commit or roll back
            # together, so a rejected quota never leaves an accepted job.
            self._reserve_quota(conn, request, grant, now)
            self._add_event(conn, request.request_id, "snapshot", {}, None, 0, now)
            row = conn.execute("SELECT * FROM jobs WHERE request_id=?", (request.request_id,)).fetchone()
            return self._snapshot(conn, row), True

    def _allocate_attempt(self, conn: sqlite3.Connection, job: sqlite3.Row, now: int, *, retry_index: int, capability_id: str = "") -> str:
        capability_id = capability_id or job["capability_id"]
        fence = conn.execute("SELECT value FROM counters WHERE name='lease_fence'").fetchone()[0] + 1
        if fence >= 2**63:
            raise StateCorrupt("lease fence exhausted SQLite's signed integer range")
        conn.execute("UPDATE counters SET value=? WHERE name='lease_fence'", (fence,))
        request = self._request(conn, job)
        counter = conn.execute("SELECT value FROM session_counters WHERE run_id=? AND session_id=?", (request.run_id, request.session_id)).fetchone()
        sequence = (counter[0] if counter else 0) + 1
        conn.execute("""
            INSERT INTO session_counters(run_id,session_id,value) VALUES(?,?,?)
            ON CONFLICT(run_id,session_id) DO UPDATE SET value=excluded.value
        """, (request.run_id, request.session_id, sequence))
        attempt_id = f"attempt-{fence}-{uuid.uuid4().hex[:16]}"
        conn.execute("""
            INSERT INTO attempts(attempt_id,request_id,lease_fence,retry_index,state,queued_at_ms,deadline_at_ms,session_sequence,capability_id,frontend_instance)
            VALUES(?,?,?,?,?,?,?,?,?,?)
        """, (attempt_id, request.request_id, fence, retry_index, "queued", now,
              now + request.queue_timeout_ms, sequence, capability_id, self.frontend_instance))
        conn.execute("""
            UPDATE jobs SET current_attempt_id=?,capability_id=?,state='queued',cancel_requested=0,
              failure_json=NULL,expires_at_ms=NULL,updated_at_ms=? WHERE request_id=?
        """, (attempt_id, capability_id, now, request.request_id))
        self._add_event(conn, request.request_id, "snapshot", {}, attempt_id, fence, now)
        return attempt_id

    def enqueue_accepted(self, request_id: str) -> JobSnapshot:
        now = _now_ms(self._clock)
        with self._write() as conn:
            job = conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
            if job is None:
                raise JobError("not_found", "accepted job disappeared", request_id=request_id)
            if job["state"] == "accepted" and not job["quarantined"]:
                if job["current_attempt_id"] is not None:
                    raise StateCorrupt("accepted job already has an attempt")
                self._allocate_attempt(conn, job, now, retry_index=0)
            return self._snapshot(conn, conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone())

    def accepted_jobs(self) -> list[str]:
        with self._lock:
            return [row[0] for row in self._conn.execute("SELECT request_id FROM jobs WHERE state='accepted' AND quarantined=0 ORDER BY accepted_at_ms")]

    def get_job(self, *, grant_token: str, request_id: str, request_digest: str, expected: ExpectedIdentities) -> JobSnapshot:
        now = _now_ms(self._clock)
        self.expire_jobs(now_ms=now)
        with self._write() as conn:
            _grant, job = self._scoped_job(conn, grant_token, request_id, now)
            self._check_ref(job, request_digest, expected)
            return self._snapshot(conn, job)

    def retry(
        self, *, grant_token: str, request_id: str, request_digest: str,
        expected: ExpectedIdentities, interrupted_attempt_id: str, capability_id: str,
        required_features: list[str] | tuple[str, ...],
    ) -> JobSnapshot:
        features = self._features(required_features)
        now = _now_ms(self._clock)
        self.expire_jobs(now_ms=now)
        with self._write() as conn:
            grant, job = self._scoped_job(conn, grant_token, request_id, now)
            self._check_ref(job, request_digest, expected)
            capability = self._capability(conn, capability_id, grant, expected, features, now)
            if job["quarantined"]:
                raise JobError("failed_precondition", "restored job is quarantined pending operator inspection", request_id=request_id)
            if job["state"] != "interrupted" or not interrupted_attempt_id or job["current_attempt_id"] != interrupted_attempt_id:
                raise JobError("failed_precondition", "Retry requires the current interrupted attempt", request_id=request_id)
            old = conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (interrupted_attempt_id,)).fetchone()
            if old is None or old["state"] != "interrupted":
                raise JobError("aborted", "interrupted attempt changed; reconcile with GetJob", request_id=request_id)
            if job["cancel_requested"] or old["cancel_requested"]:
                # Also close legacy interrupted rows written before cancellation
                # was made atomic; Retry must never clear a durable cancel bit.
                conn.execute(
                    "UPDATE attempts SET state='cancelled',cancel_requested=1,finished_at_ms=? WHERE attempt_id=?",
                    (now, old["attempt_id"]),
                )
                self._finish_terminal(
                    conn, job, "cancelled", now,
                    JobFailure("infrastructure", False, "cancelled", "cancelled job cannot be retried"),
                )
                return self._snapshot(conn, conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone())
            if old["finished_at_ms"] is None or capability["issued_at_ms"] < old["finished_at_ms"] or capability_id == old["capability_id"]:
                raise JobError("failed_precondition", "Retry requires a fresh GetCapabilities exchange after the interruption", request_id=request_id)
            count = conn.execute("SELECT COUNT(*) FROM attempts WHERE request_id=?", (request_id,)).fetchone()[0]
            if count >= self.max_attempts:
                raise JobError("resource_exhausted", "maximum attempt count is exhausted", request_id=request_id)
            request = self._request(conn, job)
            self._reserve_quota(conn, request, grant, now)
            self._allocate_attempt(conn, job, now, retry_index=old["retry_index"] + 1, capability_id=capability_id)
            return self._snapshot(conn, conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone())

    def cancel(self, *, grant_token: str, request_id: str, request_digest: str, expected: ExpectedIdentities, reason: str = "") -> JobSnapshot:
        del reason  # sanitized diagnostics are attached by the supervisor, never trusted as results
        now = _now_ms(self._clock)
        self.expire_jobs(now_ms=now)
        with self._write() as conn:
            _grant, job = self._scoped_job(conn, grant_token, request_id, now)
            self._check_ref(job, request_digest, expected)
            if job["state"] in {"completed", "failed", "cancelled", "expired"}:
                return self._snapshot(conn, job)
            if job["state"] in {"accepted", "queued", "interrupted"}:
                if job["current_attempt_id"] is None:
                    self._allocate_attempt(conn, job, now, retry_index=0)
                    job = conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
                attempt_id = job["current_attempt_id"]
                conn.execute("UPDATE attempts SET state='cancelled',cancel_requested=1,finished_at_ms=? WHERE attempt_id=?", (now, attempt_id))
                self._finish_terminal(conn, job, "cancelled", now, JobFailure("infrastructure", False, "cancelled", "request cancelled before execution"))
            else:
                conn.execute("UPDATE jobs SET cancel_requested=1,updated_at_ms=? WHERE request_id=?", (now, request_id))
                if job["current_attempt_id"]:
                    conn.execute("UPDATE attempts SET cancel_requested=1 WHERE attempt_id=?", (job["current_attempt_id"],))
            return self._snapshot(conn, conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone())

    def cancel_scope(self, scope_id: str) -> list[str]:
        """Security revocation path; does not require the revoked token."""

        now = _now_ms(self._clock)
        running: list[str] = []
        with self._write() as conn:
            rows = conn.execute("""
                SELECT * FROM jobs WHERE grant_scope_id=?
                  AND state IN ('accepted','queued','running','interrupted')
            """, (scope_id,)).fetchall()
            for job in rows:
                request_id = job["request_id"]
                if job["state"] == "running":
                    conn.execute("UPDATE jobs SET cancel_requested=1,updated_at_ms=? WHERE request_id=?", (now, request_id))
                    conn.execute("UPDATE attempts SET cancel_requested=1 WHERE attempt_id=?", (job["current_attempt_id"],))
                    running.append(request_id)
                    continue
                if job["current_attempt_id"] is None:
                    self._allocate_attempt(conn, job, now, retry_index=0)
                    job = conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
                conn.execute("UPDATE attempts SET state='cancelled',cancel_requested=1,finished_at_ms=? WHERE attempt_id=?", (now, job["current_attempt_id"]))
                self._finish_terminal(conn, job, "cancelled", now, JobFailure("policy", False, "grant_revoked", "grant revocation cancelled the job"))
        return running

    def _finish_terminal(self, conn: sqlite3.Connection, job: sqlite3.Row, state: str, now: int, failure: JobFailure | None) -> None:
        request_id = job["request_id"]
        self._release_reservation(conn, request_id, now)
        expires = now + self.retention_ms
        conn.execute("""
            UPDATE jobs SET state=?,source=NULL,cancel_requested=?,failure_json=?,expires_at_ms=?,updated_at_ms=?
            WHERE request_id=?
        """, (
            state, int(state == "cancelled" or bool(job["cancel_requested"])),
            _json(failure.to_dict()) if failure else None, expires, now, request_id,
        ))
        attempt = self._attempt(conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (job["current_attempt_id"],)).fetchone()) if job["current_attempt_id"] else None
        self._add_event(conn, request_id, "snapshot", {}, attempt.attempt_id if attempt else None, attempt.lease_fence if attempt else 0, now)

    # -- scheduling and worker fencing -------------------------------------
    def claim_next(self, *, selfcheck_age_ms: int = 60_000) -> tuple[ExternalEvaluationRequest, Attempt] | None:
        now = _now_ms(self._clock)
        with self._write() as conn:
            if conn.execute("SELECT 1 FROM jobs WHERE state='running' LIMIT 1").fetchone():
                return None
            rows = conn.execute("""
                SELECT j.*,a.attempt_id AS a_attempt_id,a.queued_at_ms AS a_queued_at_ms,
                  a.deadline_at_ms AS a_deadline_at_ms,a.lease_fence AS a_lease_fence
                FROM jobs j JOIN attempts a ON a.attempt_id=j.current_attempt_id
                WHERE j.state='queued' AND j.quarantined=0 AND a.state='queued'
            """).fetchall()
            candidates = []
            for row in rows:
                if row["cancel_requested"]:
                    conn.execute("UPDATE attempts SET state='cancelled',cancel_requested=1,finished_at_ms=? WHERE attempt_id=?", (now, row["a_attempt_id"]))
                    self._finish_terminal(conn, row, "cancelled", now, JobFailure("infrastructure", False, "cancelled", "request cancelled"))
                    continue
                try:
                    request = self._request(conn, row)
                except (JobError, TypeError, ValueError, KeyError) as exc:
                    message = f"stored request failed digest/source validation: {exc}"
                    self._interrupt_row(
                        conn, row,
                        JobFailure("infrastructure", False, "data_loss", message[:1000]),
                        now, quarantine=True, recovery_error=message,
                    )
                    continue
                if now >= row["a_deadline_at_ms"]:
                    failure = JobFailure("infrastructure", True, "queue_timeout", "request exceeded its durable queue deadline")
                    self._interrupt_row(conn, row, failure, now)
                    continue
                age = max(0, now - row["a_queued_at_ms"])
                priority = 0.0 if request.kind == "formal_eval" else max(0.0, 1.0 - min(age / selfcheck_age_ms, 1.0))
                candidates.append((priority, row["a_queued_at_ms"], row["request_id"], row, request))
            if not candidates:
                return None
            _priority, _queued, _rid, row, request = min(candidates, key=lambda value: value[:3])
            attempt_id = row["a_attempt_id"]
            updated = conn.execute("""
                UPDATE attempts SET state='running',started_at_ms=?,deadline_at_ms=?,frontend_instance=?
                WHERE attempt_id=? AND state='queued'
            """, (now, now + request.candidate_timeout_ms, self.frontend_instance, attempt_id))
            if updated.rowcount != 1:
                raise JobError("aborted", "queue attempt changed while claiming the device")
            conn.execute("UPDATE jobs SET state='running',updated_at_ms=? WHERE request_id=?", (now, request.request_id))
            conn.execute("INSERT OR IGNORE INTO counters(name,value) VALUES('device_lease',0)")
            conn.execute("UPDATE counters SET value=value+1 WHERE name='device_lease'")
            self._add_event(conn, request.request_id, "snapshot", {}, attempt_id, row["a_lease_fence"], now)
            attempt = self._attempt(conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone())
            assert attempt is not None
            return request, attempt

    def attach_worker(self, request_id: str, attempt: Attempt, *, pid: int, pgid: int, start_token: str) -> bool:
        with self._write() as conn:
            row = conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt.attempt_id,)).fetchone()
            job = conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
            if row is None or job is None or job["state"] != "running" or job["current_attempt_id"] != attempt.attempt_id or row["lease_fence"] != attempt.lease_fence or row["state"] != "running":
                self._diagnostic(conn, request_id, attempt.attempt_id, "stale_worker_attach", {"pid": pid})
                return False
            updated = conn.execute("""
                UPDATE attempts SET worker_pid=?,worker_pgid=?,worker_start_token=?
                WHERE attempt_id=? AND lease_fence=? AND state='running'
            """, (pid, pgid, start_token, attempt.attempt_id, attempt.lease_fence))
            if updated.rowcount != 1:
                self._diagnostic(conn, request_id, attempt.attempt_id, "stale_worker_attach", {"pid": pid})
                return False
            return True

    def active_attempt(self, request_id: str) -> Attempt | None:
        with self._lock:
            job = self._conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
            if job is None or not job["current_attempt_id"]:
                return None
            attempt = self._attempt(self._conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (job["current_attempt_id"],)).fetchone())
            return attempt if attempt and attempt.state in {"queued", "running"} else None

    def record_progress(self, request_id: str, attempt: Attempt, progress: Mapping[str, Any]) -> bool:
        allowed = {"stage", "completed_reps", "total_reps", "detail", "observed_at_ms"}
        if not isinstance(progress, Mapping) or set(progress) - allowed:
            raise JobError("invalid_argument", "progress has unknown fields")
        normalized = dict(progress)
        if normalized.get("stage") not in {"queued", "static_check", "compile", "correctness", "timing", "finalizing"}:
            raise JobError("invalid_argument", "progress stage is invalid")
        for key in ("completed_reps", "total_reps"):
            value = normalized.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10_000:
                raise JobError("invalid_argument", "progress repetition count is invalid")
            normalized[key] = value
        if normalized["completed_reps"] > normalized["total_reps"]:
            raise JobError("invalid_argument", "completed progress repetitions exceed the total")
        detail = normalized.get("detail", "")
        if not isinstance(detail, str):
            raise JobError("invalid_argument", "progress detail must be text")
        normalized["detail"] = detail[:300]
        observed = normalized.get("observed_at_ms", _now_ms(self._clock))
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise JobError("invalid_argument", "progress timestamp is invalid")
        normalized["observed_at_ms"] = observed
        now = _now_ms(self._clock)
        self.expire_jobs(now_ms=now)
        with self._write() as conn:
            job = conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
            row = conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt.attempt_id,)).fetchone()
            if job is None or row is None or job["state"] != "running" or job["current_attempt_id"] != attempt.attempt_id or row["lease_fence"] != attempt.lease_fence or row["state"] != "running":
                self._diagnostic(conn, request_id, attempt.attempt_id, "stale_progress", {"lease_fence": attempt.lease_fence})
                return False
            self._ensure_db_capacity(len(_json(normalized).encode("utf-8")) + 4096)
            self._add_event(conn, request_id, "progress", normalized, attempt.attempt_id, attempt.lease_fence, now)
            return True

    def _snapshot_event_payload(self, conn: sqlite3.Connection, request_id: str) -> dict[str, Any]:
        job = conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
        if job is None:
            raise StateCorrupt("cannot snapshot an event for a missing job")
        snapshot = self._snapshot(conn, job)
        return {
            "version": 1,
            "snapshot": {
                "request_id": snapshot.request_id,
                "request_digest": snapshot.request_digest,
                "state": snapshot.state,
                "provenance": snapshot.provenance,
                "attempt": asdict(snapshot.current_attempt) if snapshot.current_attempt is not None else None,
                "failure": asdict(snapshot.failure) if snapshot.failure is not None else None,
                "cancel_requested": snapshot.cancel_requested,
                "last_sequence": snapshot.last_sequence,
                "accepted_at_ms": snapshot.accepted_at_ms,
                "updated_at_ms": snapshot.updated_at_ms,
                "expires_at_ms": snapshot.expires_at_ms,
                "quarantined": snapshot.quarantined,
                "recovery_error": snapshot.recovery_error,
            },
        }

    def _add_event(self, conn: sqlite3.Connection, request_id: str, kind: str, payload: dict[str, Any], attempt_id: str | None, fence: int, now: int) -> int:
        sequence = conn.execute("SELECT last_sequence FROM jobs WHERE request_id=?", (request_id,)).fetchone()[0] + 1
        conn.execute("UPDATE jobs SET last_sequence=? WHERE request_id=?", (sequence, request_id))
        event_payload = self._snapshot_event_payload(conn, request_id) if kind == "snapshot" else payload
        conn.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?)", (
            request_id, sequence, attempt_id, fence, kind, _json(event_payload), now,
        ))
        conn.execute("DELETE FROM events WHERE request_id=? AND sequence<=?", (request_id, max(0, sequence - self.event_window)))
        return sequence

    def _event_snapshot(
        self, conn: sqlite3.Connection, job: sqlite3.Row, row: sqlite3.Row,
        *, current: JobSnapshot,
    ) -> JobSnapshot:
        payload = _loads(row["payload_json"])
        if not payload:
            # Events written by the first 2C2a schema had no historical payload.
            return current
        if set(payload) != {"version", "snapshot"} or payload["version"] != 1:
            raise StateCorrupt("stored snapshot event envelope is invalid")
        document = payload["snapshot"]
        expected_fields = {
            "request_id", "request_digest", "state", "provenance", "attempt", "failure",
            "cancel_requested", "last_sequence", "accepted_at_ms", "updated_at_ms",
            "expires_at_ms", "quarantined", "recovery_error",
        }
        if not isinstance(document, Mapping) or set(document) != expected_fields:
            raise StateCorrupt("stored snapshot event fields are missing or unknown")
        attempt_document = document["attempt"]
        attempt = None
        if attempt_document is not None:
            if not isinstance(attempt_document, Mapping):
                raise StateCorrupt("stored snapshot attempt is invalid")
            attempt_fields = dict(attempt_document)
            attempt_fields["failure"] = JobFailure.from_dict(attempt_fields.get("failure"))
            try:
                attempt = Attempt(**attempt_fields)
            except (TypeError, ValueError) as exc:
                raise StateCorrupt("stored snapshot attempt is invalid") from exc
        if (
            document["request_id"] != row["request_id"]
            or document["last_sequence"] != row["sequence"]
            or (attempt is None) != (row["attempt_id"] is None)
            or (attempt is not None and (attempt.attempt_id != row["attempt_id"] or attempt.lease_fence != row["lease_fence"]))
        ):
            raise StateCorrupt("stored snapshot event binding is inconsistent")
        result = None
        if (
            document["state"] == "completed" and current.state == "completed"
            and attempt is not None and current.current_attempt is not None
            and attempt.attempt_id == current.current_attempt.attempt_id
        ):
            result = current.result
        return JobSnapshot(
            request_id=document["request_id"], request_digest=document["request_digest"],
            state=document["state"], provenance=dict(document["provenance"]),
            current_attempt=attempt, result=result,
            failure=JobFailure.from_dict(document["failure"]),
            cancel_requested=bool(document["cancel_requested"]),
            last_sequence=document["last_sequence"], accepted_at_ms=document["accepted_at_ms"],
            updated_at_ms=document["updated_at_ms"], expires_at_ms=document["expires_at_ms"],
            quarantined=bool(document["quarantined"]), recovery_error=document["recovery_error"],
        )

    def events(self, *, grant_token: str, request_id: str, request_digest: str, expected: ExpectedIdentities, after_sequence: int = 0, attempt_id: str | None = None) -> list[JobEvent]:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int) or after_sequence < 0:
            raise JobError("invalid_argument", "after_sequence must be non-negative")
        now = _now_ms(self._clock)
        self.expire_jobs(now_ms=now)
        with self._write() as conn:
            _grant, job = self._scoped_job(conn, grant_token, request_id, now)
            self._check_ref(job, request_digest, expected)
            snapshot = self._snapshot(conn, job)
            params: list[Any] = [request_id]
            filter_sql = ""
            if attempt_id is not None:
                filter_sql = " AND attempt_id=?"
                params.append(attempt_id)
            earliest = conn.execute("SELECT MIN(sequence) FROM events WHERE request_id=?", (request_id,)).fetchone()[0]
            if earliest is not None and after_sequence < earliest - 1:
                # Coalesce a retention gap into one complete current snapshot,
                # even when the caller also filters progress by attempt.
                current = snapshot.current_attempt
                return [JobEvent(request_id, current.attempt_id if current else None, current.lease_fence if current else 0, snapshot.last_sequence, snapshot=snapshot)]
            rows = conn.execute(f"SELECT * FROM events WHERE request_id=? AND sequence>?{filter_sql} ORDER BY sequence", [request_id, after_sequence, *params[1:]]).fetchall()
            result = []
            for row in rows:
                result.append(JobEvent(
                    request_id, row["attempt_id"], row["lease_fence"], row["sequence"],
                    progress=_loads(row["payload_json"]) if row["kind"] == "progress" else None,
                    snapshot=self._event_snapshot(conn, job, row, current=snapshot) if row["kind"] == "snapshot" else None,
                ))
            return result

    def _diagnostic(self, conn: sqlite3.Connection, request_id: str, attempt_id: str, kind: str, detail: dict[str, Any]) -> None:
        conn.execute("UPDATE attempts SET diagnostic_json=? WHERE attempt_id=?", (
            _json({"kind": kind, "detail": detail, "observed_at_ms": _now_ms(self._clock)}), attempt_id,
        ))

    def _interrupt_row(
        self, conn: sqlite3.Connection, job: sqlite3.Row, failure: JobFailure, now: int,
        *, quarantine: bool = False, recovery_error: str = "",
    ) -> None:
        attempt_id = job["current_attempt_id"]
        if job["cancel_requested"]:
            if attempt_id:
                conn.execute("UPDATE attempts SET state='cancelled',cancel_requested=1,finished_at_ms=? WHERE attempt_id=?", (now, attempt_id))
            self._finish_terminal(
                conn, job, "cancelled", now,
                JobFailure("infrastructure", False, "cancelled", "cancellation completed while interrupting the attempt"),
            )
            return
        if attempt_id:
            conn.execute("UPDATE attempts SET state='interrupted',finished_at_ms=?,failure_json=? WHERE attempt_id=?", (
                now, _json(failure.to_dict()), attempt_id,
            ))
        self._release_reservation(conn, job["request_id"], now)
        durable_error = recovery_error or (job["recovery_error"] if quarantine else "")
        conn.execute("UPDATE jobs SET state='interrupted',failure_json=?,quarantined=?,recovery_error=?,updated_at_ms=? WHERE request_id=?", (
            _json(failure.to_dict()), int(quarantine), durable_error[:1000], now, job["request_id"],
        ))
        attempt = self._attempt(conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()) if attempt_id else None
        self._add_event(conn, job["request_id"], "snapshot", {}, attempt.attempt_id if attempt else None, attempt.lease_fence if attempt else 0, now)

    def interrupt_attempt(
        self, request_id: str, attempt: Attempt, failure: JobFailure,
        *, quarantine: bool = False, recovery_error: str = "",
    ) -> bool:
        now = _now_ms(self._clock)
        with self._write() as conn:
            job = conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
            row = conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt.attempt_id,)).fetchone()
            if job is None or row is None or job["state"] not in {"queued", "running"} or job["current_attempt_id"] != attempt.attempt_id or row["lease_fence"] != attempt.lease_fence:
                if row is not None:
                    self._diagnostic(conn, request_id, attempt.attempt_id, "stale_interruption", {"lease_fence": attempt.lease_fence})
                return False
            self._interrupt_row(
                conn, job, failure, now,
                quarantine=quarantine, recovery_error=recovery_error,
            )
            return True

    def fail_attempt(self, request_id: str, attempt: Attempt, failure: JobFailure) -> bool:
        """Commit a non-retryable infrastructure/policy/protocol terminal failure."""

        if failure.retryable or failure.domain not in {"infrastructure", "policy", "protocol"}:
            raise ValueError("failed jobs require a non-retryable typed failure")
        now = _now_ms(self._clock)
        with self._write() as conn:
            job = conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
            row = conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt.attempt_id,)).fetchone()
            if job is None or row is None or job["state"] not in {"queued", "running"} or job["current_attempt_id"] != attempt.attempt_id or row["lease_fence"] != attempt.lease_fence:
                return False
            if job["cancel_requested"] or row["cancel_requested"]:
                conn.execute("UPDATE attempts SET state='cancelled',cancel_requested=1,finished_at_ms=? WHERE attempt_id=?", (now, attempt.attempt_id))
                self._finish_terminal(
                    conn, job, "cancelled", now,
                    JobFailure("infrastructure", False, "cancelled", "cancellation won the failure-commit race"),
                )
                return True
            conn.execute("UPDATE attempts SET state='failed',finished_at_ms=?,failure_json=? WHERE attempt_id=?", (
                now, _json(failure.to_dict()), attempt.attempt_id,
            ))
            self._finish_terminal(conn, job, "failed", now, failure)
            return True

    def finish_cancellation(self, request_id: str, attempt: Attempt) -> bool:
        now = _now_ms(self._clock)
        with self._write() as conn:
            job = conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
            row = conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt.attempt_id,)).fetchone()
            if job is None or row is None or job["state"] not in {"queued", "running"} or job["current_attempt_id"] != attempt.attempt_id or row["lease_fence"] != attempt.lease_fence:
                return False
            conn.execute("UPDATE attempts SET state='cancelled',cancel_requested=1,finished_at_ms=? WHERE attempt_id=?", (now, attempt.attempt_id))
            self._finish_terminal(conn, job, "cancelled", now, JobFailure("infrastructure", False, "cancelled", "request cancelled"))
            return True

    # -- result commit ------------------------------------------------------
    def complete_attempt(
        self, request_id: str, attempt: Attempt, *, report: dict[str, Any], evidence: dict[str, Any] | None,
        signing_key: bytes | None = None, signing_key_id: str = "",
    ) -> JobSnapshot:
        report_bytes = _json(report).encode("utf-8")
        evidence_bytes = None if evidence is None else _json(evidence).encode("utf-8")
        effective_signing_key = self._signing_key if signing_key is None else signing_key
        effective_signing_key_id = self._signing_key_id if signing_key is None else signing_key_id
        now = _now_ms(self._clock)
        self.expire_jobs(now_ms=now)
        with self._write() as conn:
            job = conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
            row = conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt.attempt_id,)).fetchone()
            if job is None or row is None:
                raise StateCorrupt("worker result references unknown durable state")
            if job["state"] == "completed" and job["current_attempt_id"] == attempt.attempt_id:
                stored_row = conn.execute("SELECT * FROM results WHERE request_id=?", (request_id,)).fetchone()
                if stored_row is None:
                    raise StateCorrupt("completed job result is missing")
                stored = self._result_from_row(stored_row, job=job, attempt=row)
                replay = self._replay_result(
                    stored, report_bytes, evidence_bytes,
                    signing_key=effective_signing_key, signing_key_id=effective_signing_key_id,
                )
                if _result_digest(replay) != stored_row["result_sha256"]:
                    raise StateCorrupt("same attempt/receipt produced a different full result envelope")
                return self._snapshot(conn, job)
            if job["state"] != "running" or job["current_attempt_id"] != attempt.attempt_id or row["lease_fence"] != attempt.lease_fence or row["state"] != "running":
                self._diagnostic(conn, request_id, attempt.attempt_id, "stale_result", {"lease_fence": attempt.lease_fence})
                return self._snapshot(conn, job)
            if job["cancel_requested"] or row["cancel_requested"]:
                self._diagnostic(conn, request_id, attempt.attempt_id, "result_after_cancel", {"lease_fence": attempt.lease_fence})
                conn.execute(
                    "UPDATE attempts SET state='cancelled',cancel_requested=1,finished_at_ms=? WHERE attempt_id=?",
                    (now, attempt.attempt_id),
                )
                self._finish_terminal(
                    conn, job, "cancelled", now,
                    JobFailure("infrastructure", False, "cancelled", "cancellation won the result-commit race"),
                )
                return self._snapshot(conn, conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone())
            request = self._request(conn, job)
            base_provenance = _loads(job["provenance_json"])
            self._validate_report(request, report, evidence)
            if report.get("protocol_version") != base_provenance.get("protocol_version"):
                raise JobError("invalid_argument", "worker report execution protocol differs from the evaluator epoch")
            if (report.get("simulated") is True) != (base_provenance.get("measurement_kind") == "simulated"):
                raise JobError("invalid_argument", "worker report measurement kind differs from evaluator provenance")
            provenance = {**base_provenance, "attempt_id": attempt.attempt_id, "request_digest": job["request_digest"]}
            receipt_id = contracts.attempt_receipt_id(
                request_id=request_id, attempt_id=attempt.attempt_id, request_digest=job["request_digest"],
            )
            receipt = {
                "receipt_id": receipt_id, "request_id": request_id, "attempt_id": attempt.attempt_id,
                "request_digest": job["request_digest"], "candidate_sha256": request.candidate_sha256,
                "lease_fence": str(attempt.lease_fence), "session_sequence": str(row["session_sequence"]),
                "issued_at": _rfc3339(now),
            }
            report_descriptor = _blob_descriptor(report_bytes, schema_id="tcenv.eval-report.v3")
            evidence_descriptor = None if evidence_bytes is None else _blob_descriptor(
                evidence_bytes, schema_id="tcenv.remote.v1.selfcheck-evidence", allow_empty=True,
            )
            signature = self._make_signature({
                "provenance": provenance, "report": report_descriptor,
                "evidence": evidence_descriptor, "receipt": receipt,
            }, signing_key=effective_signing_key, signing_key_id=effective_signing_key_id)
            result = JobResult(report_bytes, report_descriptor, evidence_bytes, evidence_descriptor, provenance, receipt, signature)
            self._validate_result_envelope(result, job=job, attempt=row)
            result_hash = _result_digest(result)
            projected_growth = (
                len(report_bytes) + (len(evidence_bytes) if evidence_bytes is not None else 0)
                + len(_json(result.report_descriptor)) + len(_json(result.evidence_descriptor))
                + len(_json(result.provenance)) + len(_json(result.receipt)) + len(_json(result.signature))
                + 16 * 1024
            )
            self._ensure_db_capacity(projected_growth)
            duplicate = conn.execute("SELECT * FROM receipts WHERE receipt_id=?", (receipt_id,)).fetchone()
            if duplicate is not None and duplicate["result_sha256"] != result_hash:
                raise StateCorrupt("same receipt_id is bound to different result bytes")
            gpu_us = _gpu_us(report.get("elapsed_s", 0.0)) if request.kind == "selfcheck" and request.mode == "timed" else 0
            hang = request.kind == "selfcheck" and request.mode == "timed" and report.get("status") in {"timeout", "crash"}
            conn.execute("UPDATE attempts SET state='completed',finished_at_ms=? WHERE attempt_id=?", (now, attempt.attempt_id))
            conn.execute("""
                INSERT INTO results VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """, (request_id, attempt.attempt_id, receipt_id, result_hash, sqlite3.Binary(report_bytes),
                  _json(report_descriptor), sqlite3.Binary(evidence_bytes) if evidence_bytes is not None else None,
                  _json(evidence_descriptor) if evidence_descriptor else None, _json(provenance), _json(receipt),
                  _json(signature) if signature else None, now))
            conn.execute("INSERT INTO receipts VALUES(?,?,?,?,?,?)", (
                receipt_id, request_id, attempt.attempt_id, job["request_digest"], result_hash, now,
            ))
            self._release_reservation(conn, request_id, now, actual_count=1 if request.kind == "selfcheck" and request.mode == "timed" else 0, actual_gpu_us=gpu_us, hang=hang)
            expires = now + self.retention_ms
            conn.execute("UPDATE jobs SET state='completed',source=NULL,failure_json=NULL,expires_at_ms=?,updated_at_ms=? WHERE request_id=?", (expires, now, request_id))
            self._add_event(conn, request_id, "snapshot", {}, attempt.attempt_id, attempt.lease_fence, now)
            return self._snapshot(conn, conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone())

    @staticmethod
    def _validate_report(request: ExternalEvaluationRequest, report: dict[str, Any], evidence: dict[str, Any] | None) -> None:
        if not isinstance(report, dict) or report.get("status") not in CANDIDATE_STATUSES:
            raise JobError("invalid_argument", "worker returned an invalid candidate report")
        for key, expected in (("task", request.task), ("dtype", request.dtype), ("size", request.size), ("mode", request.mode), ("target_id", request.target_id)):
            if report.get(key) != expected:
                raise JobError("invalid_argument", f"worker report {key} does not bind the request")
        if request.seed is not None and report.get("seed") != request.seed_int:
            raise JobError("invalid_argument", "worker report seed does not bind the request")
        if evidence is not None and (
            not isinstance(evidence, dict) or evidence.get("candidate_sha256") != request.candidate_sha256
        ):
            raise JobError("invalid_argument", "worker evidence does not bind the exact candidate bytes")
        if report.get("simulated") is not True:
            from tcenv.evaluate import EvalReport
            try:
                EvalReport.from_json(report)
            except Exception as exc:
                raise JobError("invalid_argument", f"worker report fails EvalReport validation: {exc}") from exc

    # -- crash recovery, retention and backup -------------------------------
    def running_attempts(self) -> list[tuple[str, Attempt, bool]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM jobs WHERE state='running'").fetchall()
            result = []
            for job in rows:
                attempt = self._attempt(self._conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (job["current_attempt_id"],)).fetchone())
                if attempt is not None:
                    result.append((job["request_id"], attempt, bool(job["quarantined"])))
            return result

    def recovered_attempt(self, request_id: str, attempt: Attempt, *, cleanup_error: str = "") -> JobSnapshot:
        """Finish recovery only after the caller confirms old group/fence cleanup."""

        now = _now_ms(self._clock)
        with self._write() as conn:
            job = conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
            row = conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt.attempt_id,)).fetchone()
            if job is None or row is None or job["state"] != "running" or row["lease_fence"] != attempt.lease_fence:
                raise JobError("aborted", "running recovery state changed")
            if cleanup_error:
                conn.execute("UPDATE jobs SET quarantined=1,recovery_error=?,updated_at_ms=? WHERE request_id=?", (cleanup_error[:1000], now, request_id))
            elif job["cancel_requested"]:
                conn.execute("UPDATE attempts SET state='cancelled',finished_at_ms=? WHERE attempt_id=?", (now, attempt.attempt_id))
                self._finish_terminal(conn, job, "cancelled", now, JobFailure("infrastructure", False, "cancelled", "cancellation completed during recovery"))
            elif job["recovery_error"].startswith("stored request"):
                failure = JobFailure("infrastructure", False, "data_loss", job["recovery_error"])
                self._interrupt_row(
                    conn, job, failure, now, quarantine=True,
                    recovery_error=job["recovery_error"],
                )
            else:
                failure = JobFailure("infrastructure", True, "frontend_restarted", "old worker and lease were cleaned during recovery")
                self._interrupt_row(conn, job, failure, now, quarantine=bool(job["quarantined"]))
            return self._snapshot(conn, conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone())

    def release_quarantine(self, request_id: str) -> JobSnapshot:
        now = _now_ms(self._clock)
        with self._write() as conn:
            job = conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
            if job is None:
                raise JobError("not_found", "job does not exist")
            if job["state"] == "running":
                raise JobError("failed_precondition", "old worker cleanup must be confirmed before release")
            if job["source"] is not None:
                # Releasing request-integrity quarantine never silently clears
                # the evidence and then hands the same corrupt bytes to a worker.
                self._request(conn, job)
            conn.execute("UPDATE jobs SET quarantined=0,recovery_error='',updated_at_ms=? WHERE request_id=?", (now, request_id))
            return self._snapshot(conn, conn.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone())

    def finish_restore(self) -> None:
        marker = self.state_dir / "restored.marker"
        if marker.exists():
            marker.unlink()
            self._restored = False
            fd = os.open(self.state_dir, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def expire_jobs(self, *, now_ms: int | None = None) -> int:
        now = _now_ms(self._clock) if now_ms is None else now_ms
        purged_tombstones = 0
        with self._write() as conn:
            rows = conn.execute("SELECT * FROM jobs WHERE state IN ('completed','failed','cancelled') AND expires_at_ms<=?", (now,)).fetchall()
            for job in rows:
                request_id = job["request_id"]
                conn.execute("DELETE FROM results WHERE request_id=?", (request_id,))
                conn.execute("DELETE FROM receipts WHERE request_id=?", (request_id,))
                conn.execute("UPDATE jobs SET state='expired',source=NULL,updated_at_ms=? WHERE request_id=?", (now, request_id))
                attempt = self._attempt(conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (job["current_attempt_id"],)).fetchone()) if job["current_attempt_id"] else None
                self._add_event(conn, request_id, "snapshot", {}, attempt.attempt_id if attempt else None, attempt.lease_fence if attempt else 0, now)

            # Capabilities are only handshake leases; expired rows have no
            # evidentiary value.  Inactive grant generations and expired job
            # tombstones get a second full retention interval before deletion.
            conn.execute("DELETE FROM capabilities WHERE expires_at_ms<=?", (now,))
            conn.execute(
                "DELETE FROM grants WHERE status!='active' AND updated_at_ms<=?",
                (now - self.retention_ms,),
            )
            tombstones = conn.execute(
                "SELECT request_id FROM jobs WHERE state='expired' AND updated_at_ms<=?",
                (now - self.retention_ms,),
            ).fetchall()
            for row in tombstones:
                request_id = row["request_id"]
                for table in ("results", "receipts", "reservations", "events", "attempts"):
                    conn.execute(f"DELETE FROM {table} WHERE request_id=?", (request_id,))
                conn.execute("DELETE FROM jobs WHERE request_id=?", (request_id,))
                purged_tombstones += 1
            conn.execute("""
                DELETE FROM grants
                WHERE status='active' AND expires_at_ms IS NOT NULL AND expires_at_ms<=?
                  AND NOT EXISTS (
                    SELECT 1 FROM jobs WHERE jobs.grant_scope_id=grants.scope_id
                  )
            """, (now - self.retention_ms,))
        if purged_tombstones:
            with self._lock:
                self._ensure_available()
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    self._conn.execute("VACUUM")
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.DatabaseError as exc:
                    self._fatal_error = f"SQLite retention reclaim failed: {exc}"[:1000]
                    raise StateCorrupt(f"could not reclaim expired job storage: {exc}") from exc
        return len(rows)

    def backup(self, destination: Path) -> Path:
        """Create a consistent SQLite online backup, including committed WAL state."""

        destination = Path(destination)
        if destination.exists() and destination.is_symlink():
            raise JobError("failed_precondition", "backup destination cannot be a symlink")
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(destination.parent, 0o700)
        with self._lock:
            if self._closed:
                raise JobError("unavailable", "job store is closed")
            temp = destination.with_name(destination.name + ".tmp-" + secrets.token_hex(4))
            try:
                target = sqlite3.connect(temp)
                try:
                    self._conn.backup(target)
                    rows = target.execute("PRAGMA integrity_check").fetchall()
                    if len(rows) != 1 or rows[0][0] != "ok":
                        raise StateCorrupt("backup integrity check failed")
                finally:
                    target.close()
                os.replace(temp, destination)
                os.chmod(destination, 0o600)
                fd = os.open(destination, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                directory_fd = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                return destination
            except (sqlite3.DatabaseError, OSError) as exc:
                try:
                    temp.unlink()
                except OSError:
                    pass
                raise JobError("unavailable", f"SQLite backup failed: {exc}") from exc

    @classmethod
    def restore_backup(cls, backup: Path, state_dir: Path) -> Path:
        """Restore into a fresh state dir; opening it quarantines every nonterminal job."""

        backup = Path(backup).resolve(strict=True)
        source = sqlite3.connect(f"file:{backup}?mode=ro", uri=True)
        try:
            rows = source.execute("PRAGMA integrity_check").fetchall()
            if len(rows) != 1 or rows[0][0] != "ok" or source.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                raise StateCorrupt("backup is corrupt or has an unsupported schema")
        except sqlite3.DatabaseError as exc:
            raise StateCorrupt(f"cannot restore backup: {exc}") from exc
        finally:
            source.close()
        target_dir = _prepare_state_dir(state_dir)
        if any(target_dir.iterdir()):
            raise JobError("failed_precondition", "restore requires an empty state directory")
        target = target_dir / "jobs.sqlite3"
        shutil.copyfile(backup, target)
        os.chmod(target, 0o600)
        target_fd = os.open(target, os.O_RDONLY)
        try:
            os.fsync(target_fd)
        finally:
            os.close(target_fd)
        marker = target_dir / "restored.marker"
        marker.write_text("nonterminal jobs require operator inspection\n", encoding="utf-8")
        os.chmod(marker, 0o600)
        marker_fd = os.open(marker, os.O_RDONLY)
        try:
            os.fsync(marker_fd)
        finally:
            os.close(marker_fd)
        fd = os.open(target_dir, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return target
