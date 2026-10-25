"""The legacy frame adapter for the durable remote evaluator job engine.

The wire protocol remains the original length-prefixed frame protocol.  Queueing,
idempotency, quota, attempts, leasing and worker supervision belong to
:class:`tcenv.remote.job_engine.JobEngine`; this module only authenticates legacy
connections and translates requests, events and results.  The existing legacy
worker remains behind ``tcenv.remote.legacy_worker_bridge``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import secrets
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from tcenv.evaluation import EvaluationResult, sign_result, validate_receipt

from .identity import ServiceIdentity, SessionGrant
from .job_engine import JobEngine
from .job_types import ExternalEvaluationRequest, JobError, JobSnapshot
from .protocol import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    EvaluationRequest,
    Frame,
    FrameDecoder,
    HelloRequest,
    ProtocolError,
    sha256_hex,
    write_frame,
)

MAX_WORKER_STDERR = 4_000
_DELIVERY_POLL_S = 0.05
_ENGINE_TIMEOUT_MESSAGE = "candidate exceeded the durable execution deadline"


@dataclass(frozen=True)
class _RequestRef:
    """Non-authoritative connection state used to adapt and scope one job."""

    legacy: EvaluationRequest
    external: ExternalEvaluationRequest
    grant: SessionGrant


@dataclass
class _Delivery:
    ref: _RequestRef
    after_sequence: int
    # Serializes "this delivery still owns the request" with the frame write,
    # without holding the coordinator lock across pipe I/O.
    send_lock: threading.Lock = field(default_factory=threading.Lock)
    superseded: bool = False


class RemoteService:
    """One legacy frame connection backed by one durable ``JobEngine``."""

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

        # There is no lossless automatic migration for the old counters: in
        # particular, a reserved slot may or may not have executed.  Never drop
        # that history silently.  An operator must explicitly export/remove it
        # before opening the SQLite-authoritative service.
        self._selfcheck_usage_path = self.state_dir / "selfcheck_usage.json"
        try:
            self._selfcheck_usage_path.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ValueError(f"cannot inspect legacy selfcheck usage at {self._selfcheck_usage_path}: {exc}") from exc
        else:
            raise ValueError(
                f"legacy selfcheck usage at {self._selfcheck_usage_path} cannot be "
                "safely imported automatically; migrate or remove it explicitly"
            )

        self._reader: BinaryIO | None = None
        self._writer: BinaryIO | None = None
        self._writer_lock = threading.Lock()
        self._authenticated = False
        self._grant: SessionGrant | None = None
        self._capability_id = ""
        self._capability_grant_token = ""
        self._seen_nonces: set[str] = set()
        self._stop = threading.Event()
        self._closed = False
        self._delivery_lock = threading.Lock()
        self._deliveries: dict[str, _Delivery] = {}
        self._delivery_thread: threading.Thread | None = None
        self._requests: dict[str, _RequestRef] = {}

        # The v1 engine correctly requires a 32-byte key, whereas legacy HMAC
        # accepted any non-empty key.  Derive a stable private engine key without
        # weakening that v1 rule.  The external signature never crosses the
        # legacy boundary; results are signed again with the original legacy key.
        engine_signing_key = hashlib.sha256(signing_key).digest() if signing_key is not None else None
        self.engine = JobEngine(
            identity=identity,
            grants=grants,
            worker_command=[sys.executable, "-m", "tcenv.remote.legacy_worker_bridge"],
            worker_env={"TCENV_LEGACY_WORKER_COMMAND": json.dumps(self.worker_command)},
            state_dir=self.state_dir,
            signing_key=engine_signing_key,
            signing_key_id="legacy/v1" if engine_signing_key is not None else "",
            selfcheck_age_s=selfcheck_age_s,
            require_evidence=True,
            selfcheck_max_timed=self.selfcheck_max_timed,
            selfcheck_max_gpu_s=self.selfcheck_max_gpu_s,
        )

    # -- public entry point and lifecycle ----------------------------------
    def serve(self, reader: BinaryIO, writer: BinaryIO) -> None:
        """Serve legacy frames until EOF, then stop delivery and the engine."""

        self._reader, self._writer = reader, writer
        decoder = FrameDecoder()
        try:
            while not self._stop.is_set():
                # ``read1`` handles a short frame without waiting for a full
                # 64-KiB read on a pipe which remains open.
                read_chunk = getattr(reader, "read1", reader.read)
                chunk = read_chunk(65_536)
                if not chunk:
                    break
                for frame in decoder.feed(chunk):
                    try:
                        self._handle_frame(frame)
                    except ProtocolError as exc:
                        self._send_error(frame.request_id, "protocol_error", str(exc))
                    except JobError as exc:
                        self._send_job_error(frame.request_id, exc)
            decoder.finish()
        except ProtocolError as exc:
            self._send_error("", "protocol_error", str(exc))
        except OSError:
            pass
        finally:
            self.close()
            self._reader = self._writer = None

    def close(self) -> None:
        """Stop delivery before closing the engine; safe to call more than once."""

        if self._closed:
            return
        self._closed = True
        self._stop.set()
        with self._delivery_lock:
            delivery_thread = self._delivery_thread
        if delivery_thread is not None and delivery_thread is not threading.current_thread():
            delivery_thread.join(timeout=2.0)
        with self._delivery_lock:
            self._deliveries.clear()
            self._delivery_thread = None
        # JobEngine owns scheduler/store ordering and worker process-group
        # cleanup.  Do not close its store directly under a live scheduler.
        self.engine.close()

    def __enter__(self) -> "RemoteService":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

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

    @staticmethod
    def _capability_nonce(frontend: str, grant: SessionGrant, client_nonce: str) -> str:
        # Capabilities and nonce uniqueness are frontend-scoped in SQLite.
        # Including the instance preserves the legacy ability to reconnect with
        # a client-chosen nonce without weakening replay checks within one
        # frontend.  The raw grant never appears in the derived nonce.
        return hashlib.sha256(
            f"legacy:{frontend}:{grant.token}:{client_nonce}".encode("utf-8")
        ).hexdigest()[:32]

    def _mint_capability(self, grant: SessionGrant, *, legacy_nonce: str | None = None):
        client_nonce = (
            self._capability_nonce(self.engine.frontend_instance, grant, legacy_nonce)
            if legacy_nonce is not None
            else hashlib.sha256(
                f"legacy-refresh:{self.engine.frontend_instance}:{grant.token}:{secrets.token_hex(16)}".encode("utf-8")
            ).hexdigest()[:32]
        )
        return self.engine.get_capabilities(grant_token=grant.token, client_nonce=client_nonce)

    def _handle_hello(self, frame: Frame) -> None:
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

        # Do not mutate the active grant/capability until every legacy and
        # engine check has succeeded: a failed re-hello keeps the old identity.
        capability = self._mint_capability(grant, legacy_nonce=hello.client_nonce)
        self._grant = grant
        self._capability_id = capability.capability_id
        self._capability_grant_token = grant.token
        self._seen_nonces.add(hello.client_nonce)
        self._authenticated = True
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
                "server_nonce": capability.server_nonce,
                "provenance": self.identity.provenance().to_dict(),
                "capabilities": capabilities,
            },
        )

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
        if request.mode not in grant.modes_for(request.kind):
            raise ProtocolError(f"session grant does not allow {request.kind} mode {request.mode}")
        return grant

    def _submit(self, ref: _RequestRef) -> JobSnapshot:
        grant = ref.grant
        capability_id = self._capability_id if self._capability_grant_token == grant.token else ""
        if not capability_id:
            capability = self._mint_capability(grant)
            capability_id = capability.capability_id
            self._capability_id = capability_id
            self._capability_grant_token = grant.token
        try:
            return self.engine.submit(
                request=ref.external,
                expected=self.engine.expected_identities,
                capability_id=capability_id,
                grant_token=grant.token,
            )
        except JobError as exc:
            if exc.code != "failed_precondition" or "capability" not in str(exc):
                raise
        # A long-lived legacy connection can outlive the short v1 capability.
        # Refresh once; never retry the submission for other failures.
        capability = self._mint_capability(grant)
        self._capability_id = capability.capability_id
        self._capability_grant_token = grant.token
        return self.engine.submit(
            request=ref.external,
            expected=self.engine.expected_identities,
            capability_id=capability.capability_id,
            grant_token=grant.token,
        )

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
        ref = _RequestRef(request, ExternalEvaluationRequest.from_legacy(request), grant)
        snapshot: JobSnapshot | None
        try:
            snapshot = self.engine.get_job(
                request_id=request.request_id,
                request_digest=ref.external.request_digest,
                expected=self.engine.expected_identities,
                grant_token=grant.token,
            )
        except JobError as exc:
            if exc.code == "not_found":
                snapshot = None
            elif exc.code == "failed_precondition" and "digest differs" in str(exc):
                self._send_error(request.request_id, "idempotency_conflict", "request_id was reused with a different body")
                return
            else:
                raise

        if snapshot is None:
            # SQLite is authoritative.  Legacy JSON is consulted only when no
            # visible engine job exists, and is never rewritten by this service.
            legacy = self._load_state(request.request_id)
            if legacy is not None:
                if legacy.get("request_id") != request.request_id:
                    raise ProtocolError(f"request state for {request.request_id} has the wrong request_id")
                if legacy.get("request_digest") != request.request_digest:
                    self._send_error(request.request_id, "idempotency_conflict", "request_id was reused with a different body")
                    return
                status = legacy.get("status")
                if status == "completed":
                    response = legacy.get("response")
                    if not isinstance(response, dict):
                        self._send_error(request.request_id, "worker_protocol_error", "completed request has no response")
                        return
                    self._remember_ref(ref)
                    self._send("result", request.request_id, response)
                    return
                if status == "failed":
                    error = legacy.get("error")
                    if (
                        not isinstance(error, dict)
                        or not isinstance(error.get("retryable"), bool)
                        or not isinstance(error.get("code"), str)
                        or not isinstance(error.get("message"), str)
                    ):
                        raise ProtocolError(f"request state for {request.request_id} has invalid failure state")
                    if not error["retryable"]:
                        self._remember_ref(ref)
                        self._send_error(
                            request.request_id,
                            error["code"] or "worker_failed",
                            "legacy worker failed" if error["code"] in {"worker_failed", "worker_unavailable"} else (error["message"] or "request failed"),
                            retryable=False,
                        )
                        return
                    # Retryable legacy failure: the old worker is gone, so this
                    # explicit resubmission creates the first durable attempt.
                elif status != "running":
                    self._send_error(request.request_id, "state_error", f"unknown request state {status!r}")
                    return
                # A legacy running row has no live worker after migration.  Only
                # this explicit resubmission creates a new engine job.

            snapshot = self._submit(ref)
            self._remember_ref(ref)
            if snapshot.state in {"accepted", "queued", "running"}:
                self._start_delivery(ref, after_sequence=0)
                return
            self._send_snapshot(snapshot, ref)
            return

        self._remember_ref(ref)
        if snapshot.state in {"accepted", "queued", "running"}:
            # Submit/GetJob replay never creates an attempt or automatically
            # retries active work.  The original delivery remains the only one.
            self._send_error(request.request_id, "in_progress", "this request_id is already in progress", retryable=True)
            return
        if snapshot.state == "interrupted":
            attempt = snapshot.current_attempt
            if (
                snapshot.failure is not None and snapshot.failure.retryable
                and not snapshot.quarantined and attempt is not None
            ):
                # Retry is an explicit legacy resubmission and always uses a
                # fresh post-interruption capability/attempt/fence.
                capability = self._mint_capability(grant)
                self._capability_id = capability.capability_id
                self._capability_grant_token = grant.token
                snapshot = self.engine.retry(
                    request_id=request.request_id,
                    request_digest=ref.external.request_digest,
                    expected=self.engine.expected_identities,
                    interrupted_attempt_id=attempt.attempt_id,
                    capability_id=capability.capability_id,
                    grant_token=grant.token,
                )
                self._start_delivery(ref, after_sequence=snapshot.last_sequence, replace=True)
                return
        # Terminal, expired and non-retryable interrupted snapshots are durable
        # outcomes.  In particular, none is silently submitted as a new job.
        self._send_snapshot(snapshot, ref)

    def _remember_ref(self, ref: _RequestRef) -> None:
        previous = self._requests.get(ref.legacy.request_id)
        if previous is None:
            self._requests[ref.legacy.request_id] = ref

    # -- cancellation -------------------------------------------------------
    def _handle_cancel(self, frame: Frame) -> None:
        if not self._authenticated:
            raise ProtocolError("hello must be completed before cancel")
        request_id = frame.payload.get("request_id")
        if not isinstance(request_id, str):
            raise ProtocolError("cancel needs a request_id")
        if request_id != frame.request_id:
            raise ProtocolError("cancel frame and body request_id disagree")
        ref = self._requests.get(request_id)
        if ref is None:
            self._send_error(request_id, "unknown_request", "no such in-flight request")
            return
        # Check the current grant, not the grant captured at Submit.  This
        # rejects cross-run/session/target cancellation after a re-hello.
        self._grant_for_request(ref.legacy)
        grant = self._grant
        assert grant is not None
        if grant.scope_document() != ref.grant.scope_document():
            raise ProtocolError("session grant scope does not own this request")
        try:
            snapshot = self.engine.get_job(
                request_id=request_id,
                request_digest=ref.external.request_digest,
                expected=self.engine.expected_identities,
                grant_token=grant.token,
            )
        except JobError as exc:
            if exc.code == "not_found":
                self._send_error(request_id, "unknown_request", "no such in-flight request")
                return
            raise
        # Cancel owns this request's next response.  Supersede any polling
        # delivery first so a terminal frame from the old attempt cannot race
        # the cancellation reply.
        self._supersede_delivery(request_id)
        snapshot = self.engine.cancel(
            request_id=request_id,
            request_digest=ref.external.request_digest,
            expected=self.engine.expected_identities,
            grant_token=grant.token,
        )
        if snapshot.state in {"completed", "failed", "expired"}:
            self._send_error(request_id, "unknown_request", "no such in-flight request")
        elif snapshot.state == "cancelled":
            self._send_snapshot(snapshot, ref)
        else:
            # A running attempt only records cancel_requested here; the new
            # delivery emits the terminal frame after the worker group exits.
            self._start_delivery(ref, after_sequence=snapshot.last_sequence)

    # -- engine delivery and legacy result mapping -------------------------
    def _start_delivery(self, ref: _RequestRef, *, after_sequence: int, replace: bool = False) -> None:
        request_id = ref.legacy.request_id
        with self._delivery_lock:
            previous = self._deliveries.get(request_id)
            if previous is not None and not replace:
                return
            if previous is not None:
                # Wait for an in-flight frame of the replaced delivery so no
                # stale terminal frame can follow the freshly installed Retry.
                with previous.send_lock:
                    previous.superseded = True
            self._deliveries[request_id] = _Delivery(ref, after_sequence)
            if self._delivery_thread is None:
                self._delivery_thread = threading.Thread(
                    target=self._deliver_loop,
                    name="tcenv-legacy-delivery",
                    daemon=True,
                )
                self._delivery_thread.start()

    def _supersede_delivery(self, request_id: str) -> None:
        """Stop a polling delivery from writing after this call returns."""

        with self._delivery_lock:
            previous = self._deliveries.pop(request_id, None)
        if previous is not None:
            with previous.send_lock:
                previous.superseded = True

    def _is_current_delivery(self, delivery: _Delivery) -> bool:
        with self._delivery_lock:
            return self._deliveries.get(delivery.ref.legacy.request_id) is delivery

    @staticmethod
    def _is_delivery_end(snapshot: JobSnapshot) -> bool:
        # Interrupted is terminal for one legacy response.  A later explicit
        # resubmission installs a fresh delivery/attempt instead of auto-retry.
        return snapshot.state in {"completed", "failed", "cancelled", "interrupted", "expired"}

    def _deliver_one(self, delivery: _Delivery) -> bool:
        """Poll one job without blocking; return when its delivery is finished."""

        ref = delivery.ref
        request_id = ref.legacy.request_id
        try:
            events = self.engine.watch(
                request_id=request_id,
                request_digest=ref.external.request_digest,
                expected=self.engine.expected_identities,
                grant_token=ref.grant.token,
                after_sequence=delivery.after_sequence,
            )
        except JobError:
            # Watch gaps already coalesce to snapshots.  For any other watch
            # failure, reconcile once through GetJob rather than spinning.
            try:
                snapshot = self.engine.get_job(
                    request_id=request_id,
                    request_digest=ref.external.request_digest,
                    expected=self.engine.expected_identities,
                    grant_token=ref.grant.token,
                )
            except JobError as exc:
                if not self._stop.is_set() and self._is_current_delivery(delivery):
                    self._send_job_error(request_id, exc)
                return True
            delivery.after_sequence = max(delivery.after_sequence, snapshot.last_sequence)
            if self._is_delivery_end(snapshot):
                self._send_snapshot(snapshot, ref, delivery=delivery)
                return True
            return False
        for event in events:
            if self._stop.is_set() or not self._is_current_delivery(delivery):
                return False
            delivery.after_sequence = max(delivery.after_sequence, event.sequence)
            if event.progress is not None:
                if not self._send_if_current(
                    delivery, "progress", request_id, self._legacy_progress(ref.legacy, event.progress),
                ):
                    return False
            if event.snapshot is not None and self._is_delivery_end(event.snapshot):
                self._send_snapshot(event.snapshot, ref, delivery=delivery)
                return True
        return False

    def _deliver_loop(self) -> None:
        # One coordinator preserves the legacy service's deterministic
        # cross-request terminal ordering while every individual poll remains
        # bounded and non-blocking.  Submission order is only a tie-breaker:
        # active jobs are skipped, so priority scheduling is not changed.
        while not self._stop.is_set():
            with self._delivery_lock:
                deliveries = list(self._deliveries.values())
            for delivery in deliveries:
                if self._stop.is_set():
                    return
                if not self._is_current_delivery(delivery):
                    continue
                if self._deliver_one(delivery):
                    with self._delivery_lock:
                        if self._deliveries.get(delivery.ref.legacy.request_id) is delivery:
                            self._deliveries.pop(delivery.ref.legacy.request_id, None)
            self._stop.wait(_DELIVERY_POLL_S)

    @staticmethod
    def _legacy_progress(request: EvaluationRequest, progress: dict[str, Any]) -> dict[str, Any]:
        stages = {
            "queued": "queued",
            "static_check": "static",
            "compile": "compile",
            "correctness": "correctness",
            "timing": "bench",
            "finalizing": "done",
        }
        stage = stages.get(str(progress.get("stage")), "queued")
        observed = progress.get("observed_at_ms")
        updated_at = observed / 1000.0 if isinstance(observed, int) and not isinstance(observed, bool) else time.time()
        return {
            "task": request.task,
            "dtype": request.dtype,
            "size": request.size,
            "mode": request.mode,
            "stage": stage,
            "status": "ok" if stage == "done" else "running",
            "message": str(progress.get("detail") or "")[:300],
            "updated_at": updated_at,
            "finished": stage == "done",
        }

    def _legacy_response(self, snapshot: JobSnapshot, ref: _RequestRef) -> dict[str, Any]:
        if snapshot.result is None:
            raise JobError("data_loss", "completed job has no result")
        if snapshot.result.signature is not None and self._signing_key is None:
            # Do not turn a previously signed durable result into an unsigned
            # legacy result merely because the restarted adapter lost its key.
            raise JobError("provenance_mismatch", "signed engine result requires the legacy signing key")
        request = ref.legacy
        report = snapshot.result.report_json()
        evidence = snapshot.result.evidence_json()
        if not isinstance(evidence, dict):
            raise JobError("provenance_mismatch", "worker result has no evidence object")
        if evidence.get("candidate_sha256") != request.candidate_sha256:
            raise JobError("provenance_mismatch", "evidence candidate digest does not match request")
        # The attempt receipt supplies attempt/fence/candidate binding, but its
        # digest belongs to the v1 canonical request.  Legacy verification must
        # see the original legacy digest; the legacy HMAC below covers this
        # translated envelope, not the external signature.
        receipt = dict(snapshot.result.receipt)
        receipt["request_digest"] = request.request_digest
        receipt["measurement_kind"] = self.identity.measurement_kind
        response = {
            "schema_version": SCHEMA_VERSION,
            "request_id": request.request_id,
            "request_digest": request.request_digest,
            "report": report,
            "provenance": self.identity.provenance(request_digest=request.request_digest).to_dict(),
            "evidence": evidence,
            "receipt": receipt,
            "signature": "",
        }
        try:
            validate_receipt(EvaluationResult.from_dict(response), request)
        except Exception as exc:
            raise JobError("provenance_mismatch", f"invalid legacy result receipt: {exc}") from exc
        if self._signing_key is not None:
            response["signature"] = sign_result(EvaluationResult.from_dict(response), self._signing_key)
        return response

    def _send_snapshot(
        self, snapshot: JobSnapshot, ref: _RequestRef, *, delivery: _Delivery | None = None,
    ) -> None:
        request_id = ref.legacy.request_id

        def emit(frame_type: str, payload: dict[str, Any]) -> None:
            if delivery is None:
                self._send(frame_type, request_id, payload)
            else:
                # The replaced/Retyred delivery must never append a stale
                # terminal frame after the new attempt already owns the request.
                self._send_if_current(delivery, frame_type, request_id, payload)

        if snapshot.state == "completed":
            try:
                response = self._legacy_response(snapshot, ref)
            except JobError as exc:
                emit("error", self._error_payload(request_id, self._job_error_code(exc), self._job_error_message(exc), retryable=exc.retryable))
                return
            report = response["report"]
            status = report.get("status")
            # A durable engine deadline is the old worker-timeout outcome.  For
            # timed selfcheck, timeout/crash also locks future timed attempts in
            # the same atomic engine commit and is surfaced as candidate_timeout.
            engine_deadline = status == "timeout" and report.get("message") == _ENGINE_TIMEOUT_MESSAGE
            timed_hang = ref.legacy.kind == "selfcheck" and ref.legacy.mode == "timed" and status in {"timeout", "crash"}
            if engine_deadline or timed_hang:
                emit("error", self._error_payload(
                    request_id, "candidate_timeout", "worker exceeded the candidate timeout",
                ))
                return
            emit("result", response)
            return
        if snapshot.state == "expired":
            emit("error", self._error_payload(request_id, "state_error", "request result has expired"))
            return
        failure = snapshot.failure
        code = failure.code if failure is not None else snapshot.state
        retryable = failure.retryable if failure is not None else False
        message = failure.message if failure is not None else f"request {snapshot.state}"
        mapped = self._failure_code(code, message)
        # Engine diagnostics can contain bridge stderr.  Keep the useful legacy
        # failure category without forwarding process output or host paths.
        if mapped in {"worker_failed", "worker_unavailable", "worker_supervisor_failed"}:
            message = "legacy worker failed"
        emit("error", self._error_payload(request_id, mapped, message, retryable=retryable))

    @staticmethod
    def _failure_code(code: str, message: str) -> str:
        if code == "frontend_stopped":
            return "evaluation_unavailable"
        if code == "worker_supervisor_failed":
            return "worker_failed"
        if code != "worker_result_invalid":
            return code
        # JobEngine deliberately separates candidate reports from job failures;
        # recover the legacy validation categories at the adapter boundary.
        if "EvalReport validation" in message or "invalid candidate report" in message:
            return "worker_protocol_error"
        return "provenance_mismatch"

    # -- legacy read-only state --------------------------------------------
    def _load_state(self, request_id: str) -> dict[str, Any] | None:
        path = self._state_path(request_id)
        try:
            path.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ProtocolError(f"cannot inspect request state for {request_id}: {exc}") from exc
        try:
            # Once a legacy row exists, even a broken symlink, disappearing
            # file or permission error is a failed read, never a fresh job.
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"cannot read request state for {request_id}: {exc}") from exc
        if not isinstance(document, dict):
            raise ProtocolError(f"request state for {request_id} is not an object")
        return document

    def _state_path(self, request_id: str) -> Path:
        return self.state_dir / f"{sha256_hex(request_id.encode('utf-8'))}.json"

    # -- wire/error helpers -------------------------------------------------
    def _safe_message(self, message: str) -> str:
        value = str(message)
        for secret in self.grants:
            value = value.replace(secret, "<grant>")
        for path in (str(self.state_dir), str(Path(__file__).resolve().parents[2])):
            if path:
                value = value.replace(path, "<state>")
        return value[:MAX_WORKER_STDERR]

    def _job_error_code(self, error: JobError) -> str:
        message = str(error)
        if error.code == "already_exists" or (error.code == "failed_precondition" and "digest differs" in message):
            return "idempotency_conflict"
        if error.code == "resource_exhausted" and "self-check" in message:
            return "selfcheck_quota"
        if error.code == "failed_precondition" and "candidate hang" in message:
            return "selfcheck_hang"
        if error.code == "not_found":
            return "unknown_request"
        if error.code in {"invalid_argument", "unauthenticated", "permission_denied", "failed_precondition"}:
            return "protocol_error"
        if error.code == "data_loss":
            return "state_error"
        if error.code in {"provenance_mismatch", "worker_protocol_error", "worker_failed", "worker_unavailable"}:
            return error.code
        return "evaluation_unavailable"

    def _send_job_error(self, request_id: str, error: JobError) -> None:
        self._send_error(
            request_id, self._job_error_code(error), self._job_error_message(error),
            retryable=error.retryable,
        )

    def _job_error_message(self, error: JobError) -> str:
        code = self._job_error_code(error)
        # Bridge/process stderr is deliberately not part of the legacy wire.
        return "legacy worker failed" if code == "worker_failed" else self._safe_message(str(error))

    def _send_if_current(
        self, delivery: _Delivery, frame_type: str, request_id: str, payload: dict[str, Any],
    ) -> bool:
        """Send one frame only while this delivery still owns the request.

        The ownership check and the write share ``_delivery_lock`` with
        ``_start_delivery(replace=True)``; a Retry therefore either installs the
        replacement before the stale check (the frame is dropped) or after this
        write completes (the frame is the pre-Retry terminal outcome).
        """

        with delivery.send_lock:
            if delivery.superseded or self._stop.is_set():
                return False
            if not self._is_current_delivery(delivery):
                return False
            self._send(frame_type, request_id, payload)
            return True

    def _send(self, frame_type: str, request_id: str, payload: dict[str, Any]) -> None:
        writer = self._writer
        if writer is None:
            return
        with self._writer_lock:
            try:
                write_frame(writer, frame_type, request_id, payload)
            except (BrokenPipeError, OSError):
                self._stop.set()

    def _error_payload(self, request_id: str, code: str, message: str, *, retryable: bool = False) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "code": code,
            "message": self._safe_message(message),
            "retryable": retryable,
            "request_id": request_id,
        }

    def _send_error(self, request_id: str, code: str, message: str, *, retryable: bool = False) -> None:
        self._send("error", request_id or "unknown", self._error_payload(request_id, code, message, retryable=retryable))


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
