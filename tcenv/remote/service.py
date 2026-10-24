"""The server half of the remote evaluator protocol.

The service core is deliberately backend-agnostic.  A deployment supplies a
fixed worker command (the production command is ``python -m tcenv.remote.worker``);
tests supply their own composition root under ``tests/support/npu_sim``.  The
worker is always a separate process and its stdout is a frame stream.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from tcenv.children import forget, kill_group, popen_group, scrubbed_env
from tcenv.schema import PROTOCOL_VERSION as EVAL_PROTOCOL_VERSION
from tcenv.schema import write_json_atomic
from tcenv.evaluation import EvaluationResult, sign_result, validate_receipt

from .protocol import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    EvaluationRequest,
    EvaluatorProvenance,
    Frame,
    FrameDecoder,
    HelloRequest,
    ProtocolError,
    encode_frame,
    provenance_epoch,
    sha256_hex,
    write_frame,
)

MAX_WORKER_STDERR = 4_000


@dataclass(frozen=True)
class ServiceIdentity:
    """Stable description of the evaluator site and build."""

    site_id: str
    device_id: str
    backend: str
    backend_version: str
    target_id: str
    worker_build: str
    task_pack_digest: str
    calibration_digest: str
    task_fingerprints: frozenset[str] = frozenset()
    measurement_kind: str = "measured"
    publication_allowed: bool = False

    def __post_init__(self) -> None:
        for name in ("site_id", "device_id", "backend", "backend_version", "target_id", "worker_build", "task_pack_digest", "calibration_digest"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"service identity {name} must be non-empty")
        if not self.task_fingerprints:
            raise ValueError("service identity must register at least one task fingerprint")
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in self.task_fingerprints):
            raise ValueError("service identity task fingerprints must be 64 lowercase hex characters")
        if self.measurement_kind not in {"measured", "simulated"}:
            raise ValueError("measurement_kind must be measured or simulated")

    @property
    def epoch(self) -> str:
        return provenance_epoch(
            site_id=self.site_id,
            device_id=self.device_id,
            backend=self.backend,
            backend_version=self.backend_version,
            target_id=self.target_id,
            worker_build=self.worker_build,
            task_pack_digest=self.task_pack_digest,
            calibration_digest=self.calibration_digest,
            protocol_version=PROTOCOL_VERSION,
        )

    def provenance(self, *, request_digest: str = "") -> EvaluatorProvenance:
        return EvaluatorProvenance(
            schema_version=SCHEMA_VERSION,
            epoch=self.epoch,
            site_id=self.site_id,
            device_id=self.device_id,
            backend=self.backend,
            backend_version=self.backend_version,
            target_id=self.target_id,
            worker_build=self.worker_build,
            task_pack_digest=self.task_pack_digest,
            calibration_digest=self.calibration_digest,
            protocol_version=PROTOCOL_VERSION,
            request_digest=request_digest,
            measurement_kind=self.measurement_kind,
            publication_allowed=self.publication_allowed,
        )


@dataclass(frozen=True)
class SessionGrant:
    """The only authority a session token grants to an evaluator client."""

    token: str
    run_id: str
    session_id: str
    target_id: str
    allowed_kinds: frozenset[str] = frozenset({"formal_eval", "selfcheck"})
    expires_at: float | None = None
    max_timed: int = 20
    max_gpu_s: float = 120.0

    def __post_init__(self) -> None:
        for name in ("token", "run_id", "session_id", "target_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"session grant {name} must be non-empty")
        if not self.allowed_kinds or not self.allowed_kinds <= {"formal_eval", "selfcheck"}:
            raise ValueError("session grant kinds are invalid")
        if self.expires_at is not None and (isinstance(self.expires_at, bool) or not isinstance(self.expires_at, (int, float)) or not math.isfinite(self.expires_at)):
            raise ValueError("session grant expiry must be a finite timestamp or null")
        if isinstance(self.max_timed, bool) or not isinstance(self.max_timed, int) or self.max_timed < 1:
            raise ValueError("session grant timed quota must be a positive integer")
        if isinstance(self.max_gpu_s, bool) or not isinstance(self.max_gpu_s, (int, float)) or not math.isfinite(self.max_gpu_s) or self.max_gpu_s <= 0:
            raise ValueError("session grant GPU quota must be finite and positive")


@dataclass
class _Job:
    request: EvaluationRequest
    submitted: float
    cancel: threading.Event = field(default_factory=threading.Event)
    process: subprocess.Popen | None = None


class _JobFailed(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class RemoteService:
    """Minimal authenticated broker with idempotency, priority and one lease."""

    def __init__(
        self,
        *,
        identity: ServiceIdentity,
        grants: list[SessionGrant] | tuple[SessionGrant, ...],
        worker_command: list[str],
        state_dir: Path,
        selfcheck_max_timed: int = 20,
        selfcheck_max_gpu_s: float = 120.0,
        max_source_bytes: int = 1_000_000,
        signing_key: bytes | None = None,
        selfcheck_age_s: float = 60.0,
    ) -> None:
        if not grants:
            raise ValueError("at least one session grant is required")
        if not worker_command or not all(isinstance(part, str) and part for part in worker_command):
            raise ValueError("worker_command must be a non-empty argv list")
        self.identity = identity
        self.worker_command = list(worker_command)
        self.grants = {grant.token: grant for grant in grants}
        if len(self.grants) != len(grants):
            raise ValueError("session grant tokens must be unique")
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.selfcheck_max_timed = int(selfcheck_max_timed)
        self.selfcheck_max_gpu_s = float(selfcheck_max_gpu_s)
        if isinstance(selfcheck_age_s, bool) or not isinstance(selfcheck_age_s, (int, float)) or not math.isfinite(selfcheck_age_s) or selfcheck_age_s <= 0:
            raise ValueError("selfcheck_age_s must be finite and positive")
        self.selfcheck_age_s = float(selfcheck_age_s)
        self.max_source_bytes = int(max_source_bytes)
        if signing_key is not None and (not isinstance(signing_key, bytes) or not signing_key):
            raise ValueError("signing_key must be non-empty bytes or None")
        self._signing_key = signing_key
        self._reader: BinaryIO | None = None
        self._writer: BinaryIO | None = None
        self._writer_lock = threading.Lock()
        self._queue: list[_Job] = []
        self._sequence = 0
        self._jobs: dict[str, _Job] = {}
        self._active: _Job | None = None
        self._authenticated = False
        self._grant: SessionGrant | None = None
        self._seen_nonces: set[str] = set()
        self._selfcheck_counts: dict[tuple[str, str], int] = {}
        self._selfcheck_gpu_s: dict[tuple[str, str], float] = {}
        self._selfcheck_hangs: set[tuple[str, str]] = set()
        self._selfcheck_usage_path = self.state_dir / "selfcheck_usage.json"
        self._load_selfcheck_usage()
        self._stop = threading.Event()
        self._handler: threading.Thread | None = None
        self._lock = threading.Lock()
        self._usage_lock = threading.Lock()

    def _load_selfcheck_usage(self) -> None:
        if not self._selfcheck_usage_path.is_file():
            return
        try:
            document = json.loads(self._selfcheck_usage_path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("selfcheck usage is not an object")
            for key, row in document.items():
                if not isinstance(row, dict):
                    raise ValueError("selfcheck usage row is not an object")
                run_id, session_id = key.split("/", 1)
                count = row["count"]
                gpu_s = row["gpu_s"]
                hang = row.get("hang", False)
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    raise ValueError("invalid selfcheck count")
                if not isinstance(gpu_s, (int, float)) or isinstance(gpu_s, bool) or not math.isfinite(gpu_s) or gpu_s < 0:
                    raise ValueError("invalid selfcheck GPU time")
                if not isinstance(hang, bool):
                    raise ValueError("invalid selfcheck hang flag")
                self._selfcheck_counts[(run_id, session_id)] = count
                self._selfcheck_gpu_s[(run_id, session_id)] = float(gpu_s)
                if hang:
                    self._selfcheck_hangs.add((run_id, session_id))
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read selfcheck usage: {exc}") from exc

    def _save_selfcheck_usage(self) -> None:
        keys = set(self._selfcheck_counts) | set(self._selfcheck_gpu_s) | self._selfcheck_hangs
        document = {
            f"{run_id}/{session_id}": {"count": self._selfcheck_counts.get((run_id, session_id), 0), "gpu_s": self._selfcheck_gpu_s.get((run_id, session_id), 0.0), "hang": (run_id, session_id) in self._selfcheck_hangs}
            for run_id, session_id in keys
        }
        write_json_atomic(self._selfcheck_usage_path, document)

    def _mark_selfcheck_hang(self, request: EvaluationRequest) -> None:
        if request.kind == "selfcheck" and request.mode == "timed":
            key = (request.run_id, request.session_id)
            with self._usage_lock:
                self._selfcheck_hangs.add(key)
                self._save_selfcheck_usage()

    def _refund_selfcheck(self, request: EvaluationRequest) -> None:
        """Release a reserved timed slot when no candidate verdict was obtained."""
        if request.kind != "selfcheck" or request.mode != "timed":
            return
        key = (request.run_id, request.session_id)
        with self._usage_lock:
            count = self._selfcheck_counts.get(key, 0)
            if count:
                self._selfcheck_counts[key] = count - 1
                self._save_selfcheck_usage()

    # -- public entry point -------------------------------------------------
    def serve(self, reader: BinaryIO, writer: BinaryIO) -> None:
        """Serve frames until EOF, then stop and reap the worker."""
        self._reader, self._writer = reader, writer
        self._handler = threading.Thread(target=self._worker_loop, name="tcenv-remote-jobs", daemon=True)
        self._handler.start()
        decoder = FrameDecoder()
        try:
            while not self._stop.is_set():
                # ``BufferedReader.read(n)`` on a pipe waits for n bytes or EOF.
                # A short hello frame would therefore sit unprocessed while the
                # control connection remains open. ``read1`` returns what is
                # currently available and keeps the framing incremental.
                read_chunk = getattr(reader, "read1", reader.read)
                chunk = read_chunk(65_536)
                if not chunk:
                    break
                for frame in decoder.feed(chunk):
                    try:
                        self._handle_frame(frame)
                    except ProtocolError as exc:
                        # A malformed request is one rejected job, not a reason to tear
                        # down an otherwise authenticated control connection.
                        self._send_error(frame.request_id, "protocol_error", str(exc))
            decoder.finish()
        except ProtocolError as exc:
            self._send_error("", "protocol_error", str(exc))
        except OSError:
            pass
        finally:
            self._stop.set()
            self._cancel(self._active)
            self._handler.join(timeout=5.0)
            self._reader = self._writer = None

    # -- frame handlers -----------------------------------------------------
    def _handle_frame(self, frame: Frame) -> None:
        if frame.frame_type == "hello":
            self._handle_hello(frame)
            return
        if frame.frame_type == "cancel":
            self._handle_cancel(frame)
            return
        if frame.frame_type != "request":
            raise ProtocolError(f"unexpected frame type {frame.frame_type!r}")
        if not self._authenticated:
            raise ProtocolError("hello must be completed before an evaluation request")
        self._handle_request(frame)

    def _handle_hello(self, frame: Frame) -> None:
        # A connection may re-authenticate with another valid grant. This is
        # required for a shared broker to serve multiple runs without letting
        # one run cancel another run's jobs; each request is still checked
        # against the currently active grant.
        hello = HelloRequest.from_dict(frame.payload)
        if hello.protocol_version != PROTOCOL_VERSION:
            raise ProtocolError("unsupported protocol version")
        grant = self.grants.get(hello.session_token)
        if grant is None:
            raise ProtocolError("session token was refused")
        if grant.expires_at is not None and time.time() >= grant.expires_at:
            raise ProtocolError("session token expired")
        if hello.client_nonce in self._seen_nonces:
            raise ProtocolError("client nonce was already used")
        if hello.expected_site_id and hello.expected_site_id != self.identity.site_id:
            raise ProtocolError("expected site does not match this evaluator")
        if hello.expected_worker_build and hello.expected_worker_build != self.identity.worker_build:
            raise ProtocolError("expected worker build does not match this evaluator")
        if hello.expected_task_pack_digest and hello.expected_task_pack_digest != self.identity.task_pack_digest:
            raise ProtocolError("expected task pack does not match this evaluator")
        if hello.target_id and hello.target_id != self.identity.target_id:
            raise ProtocolError("target is not served by this evaluator")
        self._grant = grant
        self._seen_nonces.add(hello.client_nonce)
        self._authenticated = True
        server_nonce = secrets.token_hex(16)
        capabilities = {
            "protocol_version": PROTOCOL_VERSION,
            "site_id": self.identity.site_id,
            "device_id": self.identity.device_id,
            "targets": [self.identity.target_id],
            "modes": ["quick", "timed", "full"],
            "max_source_bytes": self.max_source_bytes,
            "selfcheck_max_timed": self.selfcheck_max_timed,
            "selfcheck_max_gpu_s": self.selfcheck_max_gpu_s,
            "task_pack_digest": self.identity.task_pack_digest,
        }
        self._send(
            "hello",
            hello.client_nonce,
            {
                "protocol_version": PROTOCOL_VERSION,
                "client_nonce": hello.client_nonce,
                "server_nonce": server_nonce,
                "provenance": self.identity.provenance().to_dict(),
                "capabilities": capabilities,
            },
        )

    def _handle_cancel(self, frame: Frame) -> None:
        if not self._authenticated:
            raise ProtocolError("hello must be completed before cancel")
        request_id = frame.payload.get("request_id")
        if not isinstance(request_id, str):
            raise ProtocolError("cancel needs a request_id")
        job = self._jobs.get(request_id)
        if job is None:
            self._send_error(request_id, "unknown_request", "no such in-flight request")
            return
        # Cancellation is also a privileged operation: a token for one run may
        # not cancel another run's job merely by learning its request_id.
        self._grant_for_request(job.request)
        self._cancel(job)

    def _grant_for_request(self, request: EvaluationRequest) -> SessionGrant:
        grant = self._grant
        if grant is None:
            raise ProtocolError("hello must be completed before an evaluation request")
        if grant.expires_at is not None and time.time() >= grant.expires_at:
            raise ProtocolError("session token expired")
        if request.run_id != grant.run_id:
            raise ProtocolError("session grant run_id does not match")
        if request.session_id != grant.session_id:
            raise ProtocolError("session grant session_id does not match")
        if request.target_id != grant.target_id:
            raise ProtocolError("session grant target does not match")
        if request.kind not in grant.allowed_kinds:
            raise ProtocolError(f"session grant does not allow {request.kind}")
        return grant

    def _handle_request(self, frame: Frame) -> None:
        request = EvaluationRequest.from_dict(frame.payload)
        grant = self._grant_for_request(request)
        if request.request_id != frame.request_id:
            raise ProtocolError("frame request_id and request body disagree")
        if request.target_id != self.identity.target_id:
            raise ProtocolError("target is not served by this evaluator")
        if request.mode == "validate":
            raise ProtocolError("validate mode is not supported by this evaluator")
        if len(request.source_bytes) > self.max_source_bytes:
            raise ProtocolError("source exceeds the evaluator limit")
        if self.identity.task_fingerprints and request.task_fingerprint not in self.identity.task_fingerprints:
            raise ProtocolError("task fingerprint does not match the evaluator task pack")
        digest = request.request_digest
        existing = self._load_state(request.request_id)
        if existing is not None:
            if existing.get("request_digest") != digest:
                self._send_error(request.request_id, "idempotency_conflict", "request_id was reused with a different body")
                return
            status = existing.get("status")
            if status == "completed":
                response = existing.get("response")
                if isinstance(response, dict):
                    self._send("result", request.request_id, response)
                    return
                self._send_error(request.request_id, "worker_protocol_error", "completed request has no response")
                return
            if status == "running":
                if request.request_id in self._jobs:
                    self._send_error(request.request_id, "in_progress", "this request_id is already in progress", retryable=True)
                    return
                # A service restart can leave a durable running row without a live
                # worker. Do not strand it forever: make the abandoned attempt
                # retryable and let the same idempotent request be submitted again.
                self._remember_state(request.request_id, {
                    "schema_version": SCHEMA_VERSION,
                    "request_id": request.request_id,
                    "request_digest": digest,
                    "status": "failed",
                    "error": {"code": "evaluation_unavailable", "message": "worker disappeared after restart", "retryable": True},
                    "updated_at": time.time(),
                })
            elif status == "failed":
                error = existing.get("error") or {}
                if not error.get("retryable"):
                    self._send_error(request.request_id, str(error.get("code") or "worker_failed"), str(error.get("message") or "request failed"), retryable=False)
                    return
                # Retryable failures (for example a lost worker) are eligible for
                # one fresh attempt with the same request id and digest.
            else:
                self._send_error(request.request_id, "state_error", f"unknown request state {status!r}")
                return
        if request.kind == "selfcheck" and request.mode == "timed":
            key = (request.run_id, request.session_id)
            with self._usage_lock:
                if key in self._selfcheck_hangs:
                    self._send_error(request.request_id, "selfcheck_hang", "a previous self-check timed out or crashed")
                    return
                count = self._selfcheck_counts.get(key, 0)
                if count >= min(self.selfcheck_max_timed, grant.max_timed):
                    self._send_error(request.request_id, "selfcheck_quota", "self-check count quota is exhausted")
                    return
                if self._selfcheck_gpu_s.get(key, 0.0) >= min(self.selfcheck_max_gpu_s, grant.max_gpu_s):
                    self._send_error(request.request_id, "selfcheck_quota", "self-check GPU-time quota is exhausted")
                    return
                self._selfcheck_counts[key] = count + 1
                self._save_selfcheck_usage()
        self._remember_state(request.request_id, {
            "schema_version": SCHEMA_VERSION,
            "request_id": request.request_id,
            "request_digest": digest,
            "status": "running",
            "updated_at": time.time(),
        })
        job = _Job(request=request, submitted=time.perf_counter())
        with self._jobs_lock():
            self._jobs[request.request_id] = job
            self._sequence += 1
            self._queue.append(job)

    def _next_job(self) -> _Job | None:
        """Take the oldest effective-priority job, allowing selfcheck to age.

        Formal work starts ahead of selfcheck, but a selfcheck job older than a
        minute reaches the same priority. Ties are FIFO, so one run cannot starve
        another run's selfcheck indefinitely.
        """
        with self._jobs_lock():
            if not self._queue:
                return None
            now = time.perf_counter()

            def key(job: _Job) -> tuple[float, float]:
                base = 0.0 if job.request.kind == "formal_eval" else 1.0
                age = max(0.0, now - job.submitted)
                effective = base if base == 0.0 else max(0.0, base - min(age / self.selfcheck_age_s, 1.0))
                return effective, job.submitted

            index = min(range(len(self._queue)), key=lambda i: key(self._queue[i]))
            return self._queue.pop(index)

    # -- worker thread ------------------------------------------------------
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            job = self._next_job()
            if job is None:
                time.sleep(0.1)
                continue
            if job.cancel.is_set():
                self._refund_selfcheck(job.request)
                self._finish_job(job, "cancelled", "request cancelled before it started")
                with self._jobs_lock():
                    self._jobs.pop(job.request.request_id, None)
                continue
            if time.perf_counter() - job.submitted > job.request.queue_timeout_s:
                self._refund_selfcheck(job.request)
                self._finish_job(job, "queue_timeout", "request exceeded the queue timeout", retryable=True)
                with self._jobs_lock():
                    self._jobs.pop(job.request.request_id, None)
                continue
            self._active = job
            try:
                if job.cancel.is_set():
                    raise _JobFailed("cancelled", "request cancelled")
                response = self._run_worker(job)
                # _run_worker linearizes cancellation before returning a valid
                # result. Once it has returned, commit and deliver that result
                # fully instead of recording usage and then reporting cancelled.
                if job.request.kind == "selfcheck" and job.request.mode == "timed":
                    report = response.get("report") or {}
                    elapsed = report.get("elapsed_s", 0.0)
                    if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and elapsed >= 0 and math.isfinite(elapsed):
                        key = (job.request.run_id, job.request.session_id)
                        with self._usage_lock:
                            self._selfcheck_gpu_s[key] = self._selfcheck_gpu_s.get(key, 0.0) + float(elapsed)
                            if report.get("status") in {"timeout", "crash"}:
                                self._selfcheck_hangs.add(key)
                            self._save_selfcheck_usage()
                self._remember_state(job.request.request_id, {
                    "schema_version": SCHEMA_VERSION,
                    "request_id": job.request.request_id,
                    "request_digest": job.request.request_digest,
                    "status": "completed",
                    "response": response,
                    "updated_at": time.time(),
                })
                self._send("result", job.request.request_id, response)
            except _JobFailed as exc:
                if exc.code == "candidate_timeout":
                    self._mark_selfcheck_hang(job.request)
                else:
                    # Transport, cancellation, worker and protocol failures are
                    # infrastructure outcomes, not candidate crash/hang evidence.
                    self._refund_selfcheck(job.request)
                self._finish_job(job, exc.code, str(exc), retryable=exc.retryable)
            except Exception as exc:  # noqa: BLE001 - a broken worker is a protocol error, not a host crash
                self._refund_selfcheck(job.request)
                self._finish_job(job, "worker_failed", f"{type(exc).__name__}: {exc}"[:MAX_WORKER_STDERR])
            finally:
                self._active = None
                with self._jobs_lock():
                    self._jobs.pop(job.request.request_id, None)

    def _validate_worker_result(self, request: EvaluationRequest, result: dict[str, Any]) -> None:
        report = result.get("report")
        if not isinstance(report, dict):
            raise _JobFailed("worker_protocol_error", "worker result has no report object")
        if report.get("simulated") is True and self.identity.measurement_kind != "simulated":
            raise _JobFailed("provenance_mismatch", "simulated report from a measured evaluator")
        for key, expected in (("task", request.task), ("dtype", request.dtype), ("size", request.size), ("mode", request.mode), ("target_id", request.target_id), ("protocol_version", EVAL_PROTOCOL_VERSION)):
            actual = report.get(key)
            if actual != expected:
                raise _JobFailed("provenance_mismatch", f"report {key} does not match request: {actual!r} != {expected!r}")
        allowed_status = {"ok", "rejected", "compile_error", "abi_error", "unsafe", "incorrect", "runtime_error", "timeout", "crash", "unchanged", "duplicate", "external_duplicate", "refused_hang"}
        if report.get("status") not in allowed_status:
            raise _JobFailed("worker_protocol_error", f"unknown report status {report.get('status')!r}")
        evidence = result.get("evidence")
        if not isinstance(evidence, dict):
            raise _JobFailed("provenance_mismatch", "worker result has no evidence object")
        if evidence.get("candidate_sha256") != request.candidate_sha256:
            raise _JobFailed("provenance_mismatch", "evidence candidate digest does not match request")
        if request.seed is not None and report.get("seed") != request.seed:
            raise _JobFailed("provenance_mismatch", "report seed does not match request")
        try:
            validate_receipt(EvaluationResult.from_dict({
                "report": report,
                "provenance": self.identity.provenance(request_digest=request.request_digest).to_dict(),
                "evidence": evidence,
                "receipt": result.get("receipt"),
                "signature": "",
            }), request)
        except Exception as exc:
            raise _JobFailed("provenance_mismatch", f"invalid result receipt: {exc}") from exc
        if report.get("simulated") is not True:
            try:
                from tcenv.evaluate import EvalReport
                EvalReport.from_json(report)
            except Exception as exc:
                raise _JobFailed("worker_protocol_error", f"invalid EvalReport: {exc}") from exc

    def _relay_progress(self, path: Path, request_id: str, stop: threading.Event) -> None:
        """Forward bounded progress snapshots without exposing worker paths."""
        last: dict[str, Any] | None = None
        allowed = ("task", "dtype", "size", "mode", "stage", "status", "message", "finished", "updated_at")

        def relay_one() -> None:
            nonlocal last
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            if not isinstance(document, dict):
                return
            payload = {key: document[key] for key in allowed if key in document}
            if payload == last:
                return
            if isinstance(payload.get("message"), str):
                payload["message"] = payload["message"][:300]
            if "finished" in payload and not isinstance(payload["finished"], bool):
                payload.pop("finished")
            if "updated_at" in payload and not isinstance(payload["updated_at"], (int, float)):
                payload.pop("updated_at")
            last = payload
            self._send("progress", request_id, payload)

        while not stop.is_set():
            relay_one()
            stop.wait(0.05)
        # A worker commonly writes its terminal snapshot immediately before
        # exiting. Drain it before _run_worker unlinks the progress file.
        relay_one()

    def _run_worker(self, job: _Job) -> dict[str, Any]:
        request = job.request
        env = scrubbed_env()
        root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        progress_dir = self.state_dir / "progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        progress_path = progress_dir / (sha256_hex(request.request_id.encode("utf-8")) + ".json")
        # A retry may reuse the same request id after an interrupted service.
        # Never relay the previous attempt's snapshot as this job's progress.
        try:
            progress_path.unlink()
        except OSError:
            pass
        env["TCENV_REMOTE_PROGRESS_PATH"] = str(progress_path)
        progress_stop = threading.Event()
        progress_thread = threading.Thread(
            target=self._relay_progress,
            args=(progress_path, request.request_id, progress_stop),
            name="tcenv-remote-progress",
            daemon=True,
        )
        progress_thread.start()
        try:
            proc = popen_group(
                self.worker_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=root,
            )
        except OSError as exc:
            progress_stop.set()
            progress_thread.join(timeout=1)
            try:
                progress_path.unlink()
            except OSError:
                pass
            raise _JobFailed("worker_unavailable", f"could not start worker: {exc}", retryable=True) from exc
        job.process = proc
        try:
            out, err = proc.communicate(
                encode_frame("request", request.request_id, request.to_dict()),
                timeout=float(request.candidate_timeout_s),
            )
        except subprocess.TimeoutExpired as exc:
            kill_group(proc)
            proc.communicate()
            raise _JobFailed("candidate_timeout", "worker exceeded the candidate timeout") from exc
        finally:
            job.process = None
            progress_stop.set()
            progress_thread.join(timeout=1)
            # Normal worker exit does not prove that its children exited. Match
            # tcenv.evaluate's cleanup and terminate the whole process group.
            kill_group(proc)
            forget(proc)
            try:
                progress_path.unlink()
            except OSError:
                pass
        if job.cancel.is_set():
            raise _JobFailed("cancelled", "request cancelled")
        if proc.returncode != 0:
            tail = (err or b"").decode("utf-8", errors="replace")[-MAX_WORKER_STDERR:]
            raise _JobFailed("worker_failed", f"worker exited {proc.returncode}: {tail}")
        decoder = FrameDecoder()
        try:
            frames = decoder.feed(out or b"")
            decoder.finish()
        except ProtocolError as exc:
            raise _JobFailed("worker_protocol_error", str(exc)) from exc
        if len(frames) != 1 or frames[0].frame_type != "result" or frames[0].request_id != request.request_id:
            raise _JobFailed("worker_protocol_error", "worker did not emit exactly one matching result frame")
        result = frames[0].payload
        self._validate_worker_result(request, result)
        response = {
            "schema_version": SCHEMA_VERSION,
            "request_id": request.request_id,
            "request_digest": request.request_digest,
            "report": result["report"],
            "provenance": self.identity.provenance(request_digest=request.request_digest).to_dict(),
            "evidence": result.get("evidence"),
            "receipt": result.get("receipt"),
            "signature": str(result.get("signature") or ""),
        }
        if self._signing_key is not None:
            response["signature"] = sign_result(EvaluationResult.from_dict(response), self._signing_key)
        return response

    # -- cancellation/state/wire helpers -----------------------------------
    def _cancel(self, job: _Job | None) -> None:
        if job is None:
            return
        job.cancel.set()
        proc = job.process
        if proc is not None and proc.poll() is None:
            try:
                kill_group(proc)
            except OSError:
                pass

    def _finish_job(self, job: _Job, code: str, message: str, *, retryable: bool = False) -> None:
        self._remember_state(job.request.request_id, {
            "schema_version": SCHEMA_VERSION,
            "request_id": job.request.request_id,
            "request_digest": job.request.request_digest,
            "status": "failed",
            "error": {"code": code, "message": message, "retryable": retryable},
            "updated_at": time.time(),
        })
        self._send_error(job.request.request_id, code, message, retryable=retryable)

    def _load_state(self, request_id: str) -> dict[str, Any] | None:
        path = self._state_path(request_id)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"cannot read request state for {request_id}: {exc}") from exc
        if not isinstance(document, dict):
            raise ProtocolError(f"request state for {request_id} is not an object")
        return document

    def _remember_state(self, request_id: str, document: dict[str, Any]) -> None:
        write_json_atomic(self._state_path(request_id), document)

    def _state_path(self, request_id: str) -> Path:
        return self.state_dir / f"{sha256_hex(request_id.encode('utf-8'))}.json"

    def _jobs_lock(self) -> threading.Lock:
        return self._lock

    def _send(self, frame_type: str, request_id: str, payload: dict[str, Any]) -> None:
        writer = self._writer
        if writer is None:
            return
        with self._writer_lock:
            try:
                write_frame(writer, frame_type, request_id, payload)
            except (BrokenPipeError, OSError):
                self._stop.set()

    def _send_error(self, request_id: str, code: str, message: str, *, retryable: bool = False) -> None:
        self._send("error", request_id or "unknown", {
            "schema_version": SCHEMA_VERSION,
            "code": code,
            "message": message[:MAX_WORKER_STDERR],
            "retryable": retryable,
            "request_id": request_id,
        })


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m tcenv.remote.service")
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--backend-version", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--worker-build", required=True)
    parser.add_argument("--task-pack-digest", required=True)
    parser.add_argument("--calibration-digest", required=True)
    parser.add_argument("--task-fingerprint", action="append", default=[])
    parser.add_argument("--token", action="append", default=[])
    parser.add_argument("--formal-token", action="append", default=[])
    parser.add_argument("--selfcheck-token", action="append", default=[])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--allow-kind", action="append", choices=["formal_eval", "selfcheck"], default=None)
    parser.add_argument("--publication-allowed", action="store_true", help="advertise that the host allowlist may publish this site")
    parser.add_argument("--signature-key-file", type=Path, default=None, help="optional file containing an HMAC-SHA256 signing key")
    parser.add_argument("--worker-module", default="tcenv.remote.worker")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Production entry point; no simulation backend is reachable here."""
    args = _parse_args(argv)
    identity = ServiceIdentity(
        site_id=args.site_id,
        device_id=args.device_id,
        backend=args.backend,
        backend_version=args.backend_version,
        target_id=args.target_id,
        worker_build=args.worker_build,
        task_pack_digest=args.task_pack_digest,
        calibration_digest=args.calibration_digest,
        task_fingerprints=frozenset(args.task_fingerprint),
        measurement_kind="measured",
        publication_allowed=args.publication_allowed,
    )
    grants = [
        SessionGrant(
            token=token,
            run_id=args.run_id,
            session_id=args.session_id,
            target_id=args.target_id,
            allowed_kinds=frozenset(args.allow_kind or ["formal_eval", "selfcheck"]),
        )
        for token in args.token
    ]
    grants.extend(
        SessionGrant(token=token, run_id=args.run_id, session_id=args.session_id, target_id=args.target_id, allowed_kinds=frozenset({"formal_eval"}))
        for token in args.formal_token
    )
    grants.extend(
        SessionGrant(token=token, run_id=args.run_id, session_id=args.session_id, target_id=args.target_id, allowed_kinds=frozenset({"selfcheck"}))
        for token in args.selfcheck_token
    )
    if not grants:
        raise ValueError("at least one --token, --formal-token or --selfcheck-token is required")
    signing_key = args.signature_key_file.read_bytes() if args.signature_key_file else None
    if signing_key is not None and not signing_key:
        raise ValueError("signature key file must not be empty")
    service = RemoteService(
        identity=identity,
        grants=grants,
        worker_command=[sys.executable, "-m", args.worker_module],
        state_dir=args.state_dir,
        signing_key=signing_key,
    )
    service.serve(sys.stdin.buffer, sys.stdout.buffer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
