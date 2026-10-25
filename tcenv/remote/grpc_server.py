"""gRPC ingress for the external ``tcenv.remote.v1`` evaluator.

The adapter authenticates metadata, maps Protobuf to the transport-neutral
``JobEngine`` and maps ``JobError`` onto standard gRPC status details.  Queueing,
idempotency, attempts, fencing, quota and result signing stay in the engine, so
this module and the legacy frame adapter cannot drift into two implementations.

Importing this module requires the optional gRPC dependency group.  The local
runtime and ``tcenv.remote.contracts`` never import it.
"""

from __future__ import annotations

import logging
import stat
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import grpc
from google.protobuf import any_pb2, duration_pb2
from google.rpc import error_details_pb2, status_pb2
from grpc_status import rpc_status

from . import grpc_contract as gc
from .job_engine import JobEngine
from .job_types import TERMINAL_JOB_STATES, JobError, JobEvent
from .revocation import (
    normalize_fingerprints,
    peer_certificate_present,
    peer_certificate_sha256,
)
from .v1 import evaluator_pb2 as pb
from .v1 import evaluator_pb2_grpc as pb_grpc

GRANT_METADATA_KEY = "authorization"
ERROR_DOMAIN = "tcenv.remote.v1"
DEFAULT_MAX_MESSAGE_BYTES = 48_000_000
DEFAULT_MAX_WATCH_STREAMS = 8
DEFAULT_WATCH_POLL_S = 0.05

_STATUS_CODES = {
    "cancelled": grpc.StatusCode.CANCELLED,
    "invalid_argument": grpc.StatusCode.INVALID_ARGUMENT,
    "deadline_exceeded": grpc.StatusCode.DEADLINE_EXCEEDED,
    "not_found": grpc.StatusCode.NOT_FOUND,
    "already_exists": grpc.StatusCode.ALREADY_EXISTS,
    "permission_denied": grpc.StatusCode.PERMISSION_DENIED,
    "resource_exhausted": grpc.StatusCode.RESOURCE_EXHAUSTED,
    "failed_precondition": grpc.StatusCode.FAILED_PRECONDITION,
    "aborted": grpc.StatusCode.ABORTED,
    "out_of_range": grpc.StatusCode.OUT_OF_RANGE,
    "unimplemented": grpc.StatusCode.UNIMPLEMENTED,
    "internal": grpc.StatusCode.INTERNAL,
    "unavailable": grpc.StatusCode.UNAVAILABLE,
    "unauthenticated": grpc.StatusCode.UNAUTHENTICATED,
    "data_loss": grpc.StatusCode.DATA_LOSS,
}
_RETRYABLE_CODES = frozenset({
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.ABORTED,
})
_MAX_DETAIL_TEXT = 1000
_MAX_GRANT_BYTES = 512


class AuthenticationLimitExceeded(JobError):
    def __init__(self) -> None:
        super().__init__("resource_exhausted", "too many authentication failures; retry later", retryable=True)


def grpc_code_for(error: JobError) -> grpc.StatusCode:
    return _STATUS_CODES.get(error.code, grpc.StatusCode.INTERNAL)


def sanitize_error_text(text: Any, *, state_dir: Path | None = None, limit: int = _MAX_DETAIL_TEXT) -> str:
    value = str(text)
    if state_dir is not None:
        value = value.replace(str(state_dir), "<state>")
    return value[:limit]


def status_for_error(
    error: JobError,
    *,
    request_id: str = "",
    attempt_id: str = "",
    request_digest: str = "",
    lease_fence: int = 0,
    state_dir: Path | None = None,
    engine: JobEngine | None = None,
) -> status_pb2.Status:
    code = grpc_code_for(error)
    message = sanitize_error_text(error, state_dir=state_dir) or code.name
    metadata = {}
    for name, value in (
        ("request_id", request_id or error.request_id),
        ("attempt_id", attempt_id or error.attempt_id),
        ("request_digest", request_digest),
    ):
        if value:
            metadata[name] = str(value)[:160]
    if engine is not None:
        metadata["evaluator_epoch"] = engine.expected_identities.evaluator.evaluator_epoch
        metadata["api_contract_id"] = engine.expected_identities.api_contract_id
        metadata["evidence_contract_id"] = engine.expected_identities.evidence_contract_id
    details = []
    info = any_pb2.Any()
    info.Pack(error_details_pb2.ErrorInfo(
        reason=(error.code or "internal").upper()[:64],
        domain=ERROR_DOMAIN,
        metadata=metadata,
    ))
    details.append(info)
    evaluation = any_pb2.Any()
    evaluation.Pack(gc.error_detail(
        request_id=request_id or error.request_id,
        attempt_id=attempt_id or error.attempt_id,
        request_digest=request_digest,
        reason=error.code,
        missing_features=error.missing_features,
        failure_domain=error.failure_domain,
        retryable=error.retryable,
        evaluator_epoch=engine.expected_identities.evaluator.evaluator_epoch if engine else "",
        api_contract_id=engine.expected_identities.api_contract_id if engine else "",
        evidence_contract_id=engine.expected_identities.evidence_contract_id if engine else "",
        lease_fence=lease_fence,
    ))
    details.append(evaluation)
    if error.retryable or code in _RETRYABLE_CODES:
        retry = any_pb2.Any()
        retry.Pack(error_details_pb2.RetryInfo(retry_delay=duration_pb2.Duration(seconds=1)))
        details.append(retry)
    return status_pb2.Status(code=code.value[0], message=message, details=details)


def abort_with_error(
    context: grpc.ServicerContext,
    error: JobError,
    *,
    request_id: str = "",
    attempt_id: str = "",
    request_digest: str = "",
    lease_fence: int = 0,
    state_dir: Path | None = None,
    engine: JobEngine | None = None,
) -> None:
    status = status_for_error(
        error,
        request_id=request_id,
        attempt_id=attempt_id,
        request_digest=request_digest,
        lease_fence=lease_fence,
        state_dir=state_dir,
        engine=engine,
    )
    context.abort_with_status(rpc_status.to_status(status))


def _bearer_token(values: Iterable[tuple[str, str]]) -> str:
    for key, value in values:
        if key.lower() != GRANT_METADATA_KEY:
            continue
        scheme, _, token = str(value).partition(" ")
        if scheme.lower() != "bearer":
            raise JobError("unauthenticated", "authorization metadata must use the Bearer scheme")
        token = token.strip()
        if not token:
            raise JobError("unauthenticated", "session grant is required")
        if len(token.encode("utf-8")) > _MAX_GRANT_BYTES:
            raise JobError("unauthenticated", "session grant is too large")
        if any(not 0x21 <= ord(character) <= 0x7E for character in token):
            raise JobError("unauthenticated", "session grant contains invalid characters")
        return token
    raise JobError("unauthenticated", "session grant is required")


class AuthFailureLimiter:
    """Bound authentication-failure processing without storing identities."""

    def __init__(self, *, max_failures: int = 64, window_s: float = 60.0) -> None:
        if isinstance(max_failures, bool) or not isinstance(max_failures, int) or max_failures < 1:
            raise ValueError("max_failures must be a positive integer")
        if not 0 < window_s <= 3600:
            raise ValueError("window_s must be positive and bounded")
        self.max_failures = max_failures
        self.window_s = float(window_s)
        self._lock = threading.Lock()
        self._failures: list[float] = []

    def record_failure(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._failures = [value for value in self._failures if now - value < self.window_s]
            if len(self._failures) >= self.max_failures:
                raise AuthenticationLimitExceeded()
            self._failures.append(now)

    def allow(self) -> bool:
        now = time.monotonic()
        with self._lock:
            self._failures = [value for value in self._failures if now - value < self.window_s]
            return len(self._failures) < self.max_failures


def _rejecting_handler(method: str, code: grpc.StatusCode, details: str):
    if method.endswith("/Watch"):
        def stream_handler(_request: Any, context: grpc.ServicerContext):
            context.abort(code, details)
            yield  # pragma: no cover - abort raises

        return grpc.stream_stream_rpc_method_handler(stream_handler)

    def unary_handler(_request: Any, context: grpc.ServicerContext) -> None:
        context.abort(code, details)

    return grpc.unary_unary_rpc_method_handler(unary_handler)


class SessionGrantInterceptor(grpc.ServerInterceptor):
    """Reject calls without a syntactically valid per-RPC bearer grant.

    A present token is still authorized by the engine on every RPC; the
    connection itself never carries authority.
    """

    def __init__(self, limiter: AuthFailureLimiter | None = None) -> None:
        self.limiter = limiter or AuthFailureLimiter()

    def intercept_service(self, continuation, handler_call_details):
        try:
            _bearer_token(handler_call_details.invocation_metadata)
        except JobError as exc:
            code = grpc_code_for(exc)
            details = sanitize_error_text(exc)
            try:
                self.limiter.record_failure()
            except AuthenticationLimitExceeded as limit:
                return _rejecting_handler(
                    handler_call_details.method, grpc_code_for(limit), sanitize_error_text(limit),
                )
            return _rejecting_handler(handler_call_details.method, code, details)
        return continuation(handler_call_details)


def _peer_certificate_present(context: grpc.ServicerContext) -> bool:
    return peer_certificate_present(context)


class EvaluatorService(pb_grpc.EvaluatorServicer):
    """One authenticated gRPC surface over the single durable device engine."""

    def __init__(
        self,
        engine: JobEngine,
        *,
        max_watch_streams: int = DEFAULT_MAX_WATCH_STREAMS,
        watch_poll_s: float = DEFAULT_WATCH_POLL_S,
        require_client_identity: bool = False,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        auth_limiter: AuthFailureLimiter | None = None,
        revoked_client_certificates: Iterable[str] = (),
        logger: logging.Logger | None = None,
    ) -> None:
        if isinstance(max_watch_streams, bool) or not isinstance(max_watch_streams, int) or max_watch_streams < 1:
            raise ValueError("max_watch_streams must be a positive integer")
        if not 0 < watch_poll_s <= 5:
            raise ValueError("watch_poll_s must be positive and bounded")
        self.engine = engine
        self.max_watch_streams = max_watch_streams
        self.watch_poll_s = float(watch_poll_s)
        self.require_client_identity = bool(require_client_identity)
        self.max_message_bytes = int(max_message_bytes)
        self.auth_limiter = auth_limiter
        self.revoked_client_certificates = normalize_fingerprints(frozenset(revoked_client_certificates))
        self.logger = logger or logging.getLogger("tcenv.remote.grpc")
        self._watch_slots = threading.BoundedSemaphore(max_watch_streams)

    # -- helpers ------------------------------------------------------------
    def _grant(self, context: grpc.ServicerContext) -> str:
        if self.require_client_identity and not _peer_certificate_present(context):
            raise JobError("permission_denied", "a client certificate is required")
        if self.revoked_client_certificates:
            fingerprint = peer_certificate_sha256(context)
            if fingerprint is None:
                raise JobError("permission_denied", "a client certificate is required")
            if fingerprint in self.revoked_client_certificates:
                raise JobError("permission_denied", "client certificate was revoked")
        return _bearer_token(context.invocation_metadata())

    def _record_auth_failure(self, error: JobError) -> JobError:
        if self.auth_limiter is None:
            return error
        if error.code == "unauthenticated" or (error.code == "permission_denied" and "revoked" in str(error)):
            try:
                self.auth_limiter.record_failure()
            except AuthenticationLimitExceeded as limit:
                return limit
        return error

    def _abort(self, context: grpc.ServicerContext, error: JobError, **details: Any) -> None:
        error = self._record_auth_failure(error)
        abort_with_error(context, error, state_dir=self.engine.store.state_dir, engine=self.engine, **details)

    def _unexpected(self, context: grpc.ServicerContext, exc: BaseException, **details: Any) -> None:
        self.logger.warning("gRPC evaluator internal error: %s", type(exc).__name__)
        self._abort(context, JobError("internal", f"{type(exc).__name__}: {exc}"), **details)

    # -- RPCs ---------------------------------------------------------------
    def GetCapabilities(self, request: pb.CapabilitiesRequest, context: grpc.ServicerContext) -> pb.Capabilities:
        grant_token = ""
        try:
            grant_token = self._grant(context)
            gc.reject_unknown_fields(request, "CapabilitiesRequest")
            if not request.HasField("expected"):
                raise JobError("invalid_argument", "capabilities requires expected evaluator identities")
            gc.reject_unknown_fields(request.expected, "CapabilitiesRequest.expected")
            expected = gc.expected_evaluator_document(request.expected, strict=False)
            for name in ("site_id", "device_id", "target_id"):
                if not expected.get(name):
                    raise JobError("invalid_argument", f"capabilities requires expected {name}")
            capabilities = self.engine.get_capabilities(
                grant_token=grant_token,
                client_nonce=request.client_nonce,
                expected=expected,
                api_contract_ids=list(request.api_contract_ids),
                evidence_contract_ids=list(request.evidence_contract_ids),
                required_features=list(request.required_features),
            )
            return gc.capabilities_to_proto(capabilities, extra_limits={
                "max_request_bytes": self.max_message_bytes,
                "max_watch_streams": self.max_watch_streams,
            })
        except JobError as exc:
            self._abort(context, exc)
            raise AssertionError("unreachable")  # pragma: no cover
        except Exception as exc:  # noqa: BLE001 - never leak a stack or candidate verdict
            self._unexpected(context, exc)
            raise AssertionError("unreachable")  # pragma: no cover

    def Submit(self, request: pb.SubmitRequest, context: grpc.ServicerContext) -> pb.JobSnapshot:
        request_id = ""
        try:
            grant_token = self._grant(context)
            gc.reject_unknown_fields(request, "SubmitRequest")
            if not request.HasField("request") or not request.HasField("expected"):
                raise JobError("invalid_argument", "Submit requires request and expected identities")
            gc.reject_unknown_fields(request.expected, "SubmitRequest.expected")
            gc.reject_unknown_fields(request.expected.evaluator, "SubmitRequest.expected.evaluator")
            evaluation = gc.request_from_proto(request.request)
            request_id = evaluation.request_id
            if request.request_digest != evaluation.request_digest:
                raise JobError("invalid_argument", "request_digest does not match the normalized request and source bytes")
            expected = gc.expected_identities_from_proto(request.expected)
            snapshot = self.engine.submit(
                request=evaluation,
                expected=expected,
                capability_id=request.capability_id,
                required_features=list(request.required_features),
                grant_token=grant_token,
            )
            return gc.snapshot_to_proto(snapshot)
        except JobError as exc:
            self._abort(context, exc, request_id=request_id)
            raise AssertionError("unreachable")  # pragma: no cover
        except Exception as exc:  # noqa: BLE001
            self._unexpected(context, exc, request_id=request_id)
            raise AssertionError("unreachable")  # pragma: no cover

    def _job_reference(self, reference: pb.JobRef) -> tuple[str, str, Any]:
        gc.reject_unknown_fields(reference, "JobRef")
        if not reference.HasField("expected"):
            raise JobError("invalid_argument", "job reference requires exact expected identities")
        gc.reject_unknown_fields(reference.expected, "JobRef.expected")
        gc.reject_unknown_fields(reference.expected.evaluator, "JobRef.expected.evaluator")
        request_id = reference.request_id
        gc.identifier(request_id, "request_id")
        digest = gc.hex64(reference.request_digest, "request_digest")
        expected = gc.expected_identities_from_proto(reference.expected)
        return request_id, digest, expected

    def GetJob(self, request: pb.JobRef, context: grpc.ServicerContext) -> pb.JobSnapshot:
        request_id = ""
        digest = ""
        try:
            grant_token = self._grant(context)
            request_id, digest, expected = self._job_reference(request)
            snapshot = self.engine.get_job(
                request_id=request_id,
                request_digest=digest,
                expected=expected,
                grant_token=grant_token,
            )
            return gc.snapshot_to_proto(snapshot)
        except JobError as exc:
            self._abort(context, exc, request_id=request_id, request_digest=digest)
            raise AssertionError("unreachable")  # pragma: no cover
        except Exception as exc:  # noqa: BLE001
            self._unexpected(context, exc, request_id=request_id, request_digest=digest)
            raise AssertionError("unreachable")  # pragma: no cover

    def Watch(self, request: pb.WatchRequest, context: grpc.ServicerContext):
        request_id = ""
        digest = ""
        attempt_filter = None
        try:
            grant_token = self._grant(context)
            gc.reject_unknown_fields(request, "WatchRequest")
            if not request.HasField("job"):
                raise JobError("invalid_argument", "Watch requires a job reference")
            request_id, digest, expected = self._job_reference(request.job)
            attempt_filter = request.attempt_id if request.HasField("attempt_id") else None
            if attempt_filter is not None:
                gc.identifier(attempt_filter, "attempt_id")
            if not self._watch_slots.acquire(blocking=False):
                raise JobError("resource_exhausted", "watch stream limit is exhausted", retryable=True)
        except JobError as exc:
            self._abort(context, exc, request_id=request_id, request_digest=digest)
            raise AssertionError("unreachable")  # pragma: no cover
        except Exception as exc:  # noqa: BLE001
            self._unexpected(context, exc, request_id=request_id, request_digest=digest)
            raise AssertionError("unreachable")  # pragma: no cover
        return self._watch_events(
            context,
            grant_token=grant_token,
            request_id=request_id,
            digest=digest,
            expected=expected,
            after_sequence=int(request.after_sequence),
            attempt_id=attempt_filter,
        )

    def _watch_events(
        self,
        context: grpc.ServicerContext,
        *,
        grant_token: str,
        request_id: str,
        digest: str,
        expected: Any,
        after_sequence: int,
        attempt_id: str | None,
    ):
        try:
            if not 0 <= after_sequence <= 0xFFFFFFFFFFFFFFFF:
                raise JobError("invalid_argument", "after_sequence is outside the unsigned 64-bit range")
            current = self.engine.get_job(
                request_id=request_id, request_digest=digest, expected=expected, grant_token=grant_token,
            )
            current_attempt_id = current.current_attempt.attempt_id if current.current_attempt else None
            if attempt_id is not None and current_attempt_id != attempt_id:
                raise JobError("failed_precondition", "attempt filter does not match the current attempt")
            if current.state in TERMINAL_JOB_STATES and after_sequence >= current.last_sequence:
                # The caller already consumed the terminal event, or asked from
                # beyond it.  Return the durable terminal snapshot once.
                yield gc.event_to_proto(JobEvent(
                    request_id=current.request_id,
                    attempt_id=current.current_attempt.attempt_id if current.current_attempt else None,
                    lease_fence=current.current_attempt.lease_fence if current.current_attempt else 0,
                    sequence=current.last_sequence,
                    snapshot=current,
                ))
                return
            event_window = self.engine.store.event_window
            while context.is_active():
                events = self.engine.watch(
                    request_id=request_id,
                    request_digest=digest,
                    expected=expected,
                    grant_token=grant_token,
                    after_sequence=after_sequence,
                    attempt_id=attempt_id,
                )
                if not events:
                    current = self.engine.get_job(
                        request_id=request_id, request_digest=digest, expected=expected, grant_token=grant_token,
                    )
                    current_attempt_id = current.current_attempt.attempt_id if current.current_attempt else None
                    if attempt_id is not None and current_attempt_id != attempt_id:
                        raise JobError("failed_precondition", "attempt filter does not match the current attempt")
                    if current.state in TERMINAL_JOB_STATES:
                        yield gc.event_to_proto(JobEvent(
                            request_id=current.request_id,
                            attempt_id=current_attempt_id,
                            lease_fence=current.current_attempt.lease_fence if current.current_attempt else 0,
                            sequence=current.last_sequence,
                            snapshot=current,
                        ))
                        return
                    time.sleep(self.watch_poll_s)
                    continue
                for event in events:
                    after_sequence = max(after_sequence, event.sequence)
                    yield gc.event_to_proto(event)
                    if event.snapshot is not None and event.snapshot.state in TERMINAL_JOB_STATES:
                        return
                if len(events) >= event_window:
                    # A slow consumer may be far behind; re-read through the
                    # gap-coalescing path instead of buffering more events.
                    time.sleep(self.watch_poll_s)
        except JobError as exc:
            self._abort(context, exc, request_id=request_id, request_digest=digest)
        except Exception as exc:  # noqa: BLE001
            self._unexpected(context, exc, request_id=request_id, request_digest=digest)
        finally:
            self._watch_slots.release()

    def Cancel(self, request: pb.CancelRequest, context: grpc.ServicerContext) -> pb.JobSnapshot:
        request_id = ""
        digest = ""
        try:
            grant_token = self._grant(context)
            gc.reject_unknown_fields(request, "CancelRequest")
            if not request.HasField("job"):
                raise JobError("invalid_argument", "Cancel requires a job reference")
            request_id, digest, expected = self._job_reference(request.job)
            snapshot = self.engine.cancel(
                request_id=request_id,
                request_digest=digest,
                expected=expected,
                grant_token=grant_token,
            )
            return gc.snapshot_to_proto(snapshot)
        except JobError as exc:
            self._abort(context, exc, request_id=request_id, request_digest=digest)
            raise AssertionError("unreachable")  # pragma: no cover
        except Exception as exc:  # noqa: BLE001
            self._unexpected(context, exc, request_id=request_id, request_digest=digest)
            raise AssertionError("unreachable")  # pragma: no cover

    def Retry(self, request: pb.RetryRequest, context: grpc.ServicerContext) -> pb.JobSnapshot:
        request_id = ""
        digest = ""
        try:
            grant_token = self._grant(context)
            gc.reject_unknown_fields(request, "RetryRequest")
            if not request.HasField("job"):
                raise JobError("invalid_argument", "Retry requires a job reference")
            request_id, digest, expected = self._job_reference(request.job)
            snapshot = self.engine.retry(
                request_id=request_id,
                request_digest=digest,
                expected=expected,
                interrupted_attempt_id=request.interrupted_attempt_id,
                capability_id=request.capability_id,
                required_features=list(request.required_features),
                grant_token=grant_token,
            )
            return gc.snapshot_to_proto(snapshot)
        except JobError as exc:
            self._abort(context, exc, request_id=request_id, request_digest=digest)
            raise AssertionError("unreachable")  # pragma: no cover
        except Exception as exc:  # noqa: BLE001
            self._unexpected(context, exc, request_id=request_id, request_digest=digest)
            raise AssertionError("unreachable")  # pragma: no cover


# -- TLS credentials --------------------------------------------------------
def _read_secret_file(path: Path, name: str) -> bytes:
    path = Path(path)
    try:
        info = path.stat()
    except OSError as exc:
        raise ValueError(f"cannot read {name}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{name} must be a regular file")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError(f"{name} must not be readable by group or others (mode 0600)")
    data = path.read_bytes()
    if not data:
        raise ValueError(f"{name} is empty")
    return data


def load_server_credentials(
    *,
    certificate_file: Path,
    private_key_file: Path,
    client_ca_file: Path | None = None,
    require_client_certificate: bool = True,
) -> grpc.ServerCredentials:
    certificate = _read_secret_file(certificate_file, "server certificate")
    private_key = _read_secret_file(private_key_file, "server private key")
    root = None
    if client_ca_file is not None:
        root = _read_secret_file(client_ca_file, "client CA bundle")
    if require_client_certificate and root is None:
        raise ValueError("client certificate authentication requires a client CA bundle")
    return grpc.ssl_server_credentials(
        [(private_key, certificate)],
        root_certificates=root,
        require_client_auth=bool(require_client_certificate and root is not None),
    )


def load_client_credentials(
    *,
    ca_file: Path,
    certificate_file: Path | None = None,
    private_key_file: Path | None = None,
) -> grpc.ChannelCredentials:
    root = _read_secret_file(ca_file, "server CA bundle")
    if (certificate_file is None) != (private_key_file is None):
        raise ValueError("a client certificate and private key must be supplied together")
    if certificate_file is None:
        return grpc.ssl_channel_credentials(root_certificates=root)
    return grpc.ssl_channel_credentials(
        root_certificates=root,
        private_key=_read_secret_file(private_key_file, "client private key"),
        certificate_chain=_read_secret_file(certificate_file, "client certificate"),
    )


@dataclass(frozen=True)
class GrpcServerConfig:
    address: str = "127.0.0.1:0"
    max_workers: int = 8
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES
    max_watch_streams: int = DEFAULT_MAX_WATCH_STREAMS
    watch_poll_s: float = DEFAULT_WATCH_POLL_S
    require_client_identity: bool = True
    allow_insecure: bool = False
    certificate_file: Path | None = None
    private_key_file: Path | None = None
    client_ca_file: Path | None = None
    auth_max_failures: int = 64
    auth_window_s: float = 60.0
    revoked_client_certificates: frozenset[str] = frozenset()

    def credentials(self) -> grpc.ServerCredentials | None:
        if self.allow_insecure:
            return None
        if self.certificate_file is None or self.private_key_file is None:
            raise ValueError("production gRPC requires a certificate and private key")
        return load_server_credentials(
            certificate_file=self.certificate_file,
            private_key_file=self.private_key_file,
            client_ca_file=self.client_ca_file,
            require_client_certificate=self.require_client_identity,
        )


@dataclass
class RunningServer:
    server: grpc.Server
    service: EvaluatorService
    address: str
    port: int = field(default=0)

    def stop(self, *, grace_s: float = 2.0) -> None:
        self.server.stop(grace_s)

    def wait_for_termination(self, timeout: float | None = None) -> bool:
        return self.server.wait_for_termination(timeout)


def build_server(
    engine: JobEngine,
    *,
    config: GrpcServerConfig | None = None,
    credentials: grpc.ServerCredentials | None = None,
    logger: logging.Logger | None = None,
    service_factory: type[EvaluatorService] | None = None,
) -> RunningServer:
    """Build the authenticated ingress over one durable engine.

    ``service_factory`` is an extension point for the test-only composition root
    that injects acceptance faults (for example, withholding the result
    signature) at the RPC boundary.  Production passes ``None``.
    """
    config = config or GrpcServerConfig()
    if credentials is None and not config.allow_insecure:
        credentials = config.credentials()
    limiter = AuthFailureLimiter(max_failures=config.auth_max_failures, window_s=config.auth_window_s)
    factory = service_factory or EvaluatorService
    service = factory(
        engine,
        max_watch_streams=config.max_watch_streams,
        watch_poll_s=config.watch_poll_s,
        require_client_identity=config.require_client_identity and not config.allow_insecure,
        max_message_bytes=config.max_message_bytes,
        auth_limiter=limiter,
        revoked_client_certificates=config.revoked_client_certificates,
        logger=logger,
    )
    options = [
        ("grpc.max_send_message_length", config.max_message_bytes),
        ("grpc.max_receive_message_length", config.max_message_bytes),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_pings_without_data", 0),
    ]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.max_workers, thread_name_prefix="tcenv-grpc"),
        interceptors=(SessionGrantInterceptor(limiter),),
        options=options,
    )
    pb_grpc.add_EvaluatorServicer_to_server(service, server)
    if credentials is None:
        port = server.add_insecure_port(config.address)
    else:
        port = server.add_secure_port(config.address, credentials)
    if port <= 0:
        raise RuntimeError(f"gRPC evaluator could not bind {config.address!r}")
    server.start()
    host, _, requested = config.address.rpartition(":")
    bound = f"{host or '127.0.0.1'}:{port}" if requested else f"127.0.0.1:{port}"
    return RunningServer(server=server, service=service, address=bound, port=port)


def watch_request(
    *,
    request_id: str,
    request_digest: str,
    expected: Any,
    after_sequence: int = 0,
    attempt_id: str | None = None,
) -> pb.WatchRequest:
    message = pb.WatchRequest(
        job=pb.JobRef(
            request_id=request_id,
            request_digest=request_digest,
            expected=gc.expected_identities_to_proto(expected),
        ),
        after_sequence=int(after_sequence),
    )
    if attempt_id is not None:
        message.attempt_id = attempt_id
    return message
