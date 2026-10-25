"""Host-side gRPC evaluator client for the external ``tcenv.remote.v1`` API.

One process keeps one channel per endpoint.  A ``GrpcEvaluatorFactory`` hands out
one session-scoped ``GrpcEvaluator``; every RPC re-reads the short-lived session
grant, so a renewed token can reattach to an accepted job while formal
credentials never enter the agent environment.

The adapter verifies capabilities, the three fixed identities, the attempt
receipt, opaque blob digests and the ``hmac-sha256-v1`` result signature before
it turns a durable result into the legacy ``EvaluationResult`` envelope used by
the search loop.  It never retries automatically: only ``formal_resume`` (driven
by a durable pending outbox row) may call ``Retry`` on an interrupted job.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import grpc
from grpc_status import rpc_status

from tcenv.evaluation import (
    EvaluationRequest,
    EvaluationResult,
    EvaluationUnavailable,
    EvaluatorProvenance,
    sign_result,
)

from . import contracts, grpc_contract as gc
from .grpc_server import DEFAULT_MAX_MESSAGE_BYTES, load_client_credentials
from .job_types import JobError
from .v1 import evaluator_pb2 as pb
from .v1 import evaluator_pb2_grpc as pb_grpc

DEFAULT_REQUIRED_FEATURES = ("durable-jobs", "attempt-receipts", "progress")
_LEGACY_PROGRESS_STAGES = {
    "queued": "queued",
    "static_check": "static",
    "compile": "compile",
    "correctness": "correctness",
    "timing": "bench",
    "finalizing": "done",
}
_TRANSIENT_GRPC_CODES = frozenset({
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.CANCELLED,
    grpc.StatusCode.ABORTED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
})
_MAX_GRACE_SUBMIT_RETRIES = 2


class KeyringError(EvaluationUnavailable):
    """The host verifier keyring cannot authenticate a result."""


@dataclass(frozen=True)
class KeyringEntry:
    key_id: str
    key: bytes


class SignatureKeyring:
    """Host-controlled ``hmac-sha256-v1`` verifier keyring.

    Keys are indexed by the full protocol ``key_id`` (``site/key-version``) and
    normal rotation keeps old keys for historical results.  A revoked key is
    removed from verification immediately, independent of the measurement
    epoch.
    """

    def __init__(
        self,
        entries: Mapping[str, bytes | str] | None = None,
        *,
        revoked: Iterable[str] = (),
        schema_version: int = 1,
    ) -> None:
        if schema_version != 1:
            raise ValueError("unsupported signature keyring schema")
        self.schema_version = schema_version
        self._lock = threading.RLock()
        self._keys: dict[str, bytes] = {}
        self._revoked: set[str] = set()
        for key_id, key in (entries or {}).items():
            self._keys[key_id] = _key_bytes(key, key_id)
        for key_id in revoked:
            self._revoked.add(str(key_id))
        for key_id in self._revoked & set(self._keys):
            del self._keys[key_id]

    @classmethod
    def from_file(cls, path: Path | str) -> "SignatureKeyring":
        from .keyring import load_keyring_file

        parsed = load_keyring_file(path)
        keyring = cls({key_id: key for key_id, key in parsed.trusted.items()}, revoked=parsed.revoked)
        if not keyring.key_ids:
            raise ValueError("signature keyring has no usable keys")
        return keyring

    @property
    def key_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._keys))

    def key_ids_for_site(self, site_id: str) -> tuple[str, ...]:
        prefix = f"{site_id}/"
        return tuple(key for key in self.key_ids if key.startswith(prefix))

    def is_revoked(self, key_id: str) -> bool:
        with self._lock:
            return key_id in self._revoked

    def rotate(self, key_id: str, key: bytes | str) -> None:
        value = _key_bytes(key, key_id)
        with self._lock:
            if key_id in self._revoked:
                raise KeyringError("key_revoked", f"signature key {key_id!r} is revoked")
            self._keys[key_id] = value

    def revoke(self, key_id: str) -> None:
        with self._lock:
            self._revoked.add(str(key_id))
            self._keys.pop(str(key_id), None)

    def verify(
        self, *, site_id: str, key_id: str, document: Mapping[str, Any], signature: str,
    ) -> None:
        if not isinstance(key_id, str) or not key_id.startswith(f"{site_id}/"):
            raise KeyringError("signature_key_id_mismatch", "result key_id is not scoped to the evaluator site")
        with self._lock:
            if key_id in self._revoked:
                raise KeyringError("signature_key_revoked", "result signature key was revoked")
            key = self._keys.get(key_id)
        if key is None:
            raise KeyringError("signature_key_unknown", "result signature key is not trusted by this host")
        try:
            valid = contracts.verify_result_signature(document, key_id=key_id, key=key, signature=signature)
        except contracts.ContractError as exc:
            raise KeyringError("signature_invalid", f"cannot verify the result signature: {exc}") from exc
        if not valid:
            raise KeyringError("signature_mismatch", "result signature does not verify")


def _key_bytes(value: bytes | str, key_id: str) -> bytes:
    if not isinstance(key_id, str):
        raise ValueError("signature key id must be a bounded opaque identifier")
    try:
        contracts.key_id(key_id)
    except contracts.ContractError as exc:
        raise ValueError("signature key id must be a bounded opaque identifier") from exc
    if isinstance(value, str):
        try:
            raw = bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError(f"signature key {key_id!r} must be hex text or bytes") from exc
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
    else:
        raise ValueError("signature key must be bytes or lowercase hex text")
    if len(raw) < 32:
        raise ValueError("signature keys must contain at least 32 bytes")
    return raw


@dataclass(frozen=True)
class GrpcExpected:
    """Host-side exact and wildcard expectations for one endpoint."""

    site_id: str = ""
    device_id: str = ""
    target_id: str = ""
    worker_build: str = ""
    task_pack_digest: str = ""
    evaluator_epoch: str = ""

    def wildcards(self) -> dict[str, str]:
        return {
            "site_id": self.site_id,
            "device_id": self.device_id,
            "target_id": self.target_id,
            "worker_build": self.worker_build,
            "task_pack_digest": self.task_pack_digest,
            "evaluator_epoch": self.evaluator_epoch,
        }


def _rpc_error(exc: grpc.RpcError) -> EvaluationUnavailable:
    code = exc.code()
    status = None
    try:
        status = rpc_status.from_call(exc)
    except (ValueError, TypeError, AttributeError):
        status = None
    detail = None
    if status is not None:
        for item in status.details:
            if item.Is(pb.EvaluationErrorDetail.DESCRIPTOR):
                detail = pb.EvaluationErrorDetail()
                item.Unpack(detail)
                break
    message = str(exc.details() or code.name)[:1000]
    retryable = bool(detail.retryable) if detail is not None else code in _TRANSIENT_GRPC_CODES
    reason = (detail.reason if detail is not None and detail.reason else code.name.lower())
    mapping = {
        grpc.StatusCode.UNAVAILABLE: "transport_unavailable",
        grpc.StatusCode.DEADLINE_EXCEEDED: "transport_timeout",
        grpc.StatusCode.CANCELLED: "transport_cancelled",
        grpc.StatusCode.NOT_FOUND: "job_not_found",
        grpc.StatusCode.ALREADY_EXISTS: "idempotency_conflict",
        grpc.StatusCode.UNAUTHENTICATED: "grant_rejected",
        grpc.StatusCode.PERMISSION_DENIED: "permission_denied",
        grpc.StatusCode.FAILED_PRECONDITION: "precondition_failed",
        grpc.StatusCode.INVALID_ARGUMENT: "invalid_argument",
        grpc.StatusCode.RESOURCE_EXHAUSTED: "resource_exhausted",
        grpc.StatusCode.DATA_LOSS: "data_loss",
        grpc.StatusCode.ABORTED: "aborted",
        grpc.StatusCode.UNIMPLEMENTED: "unimplemented",
        grpc.StatusCode.INTERNAL: "evaluator_internal",
    }
    if detail is not None and detail.missing_features:
        return EvaluationUnavailable(
            "required_feature_missing",
            f"evaluator does not support required features: {list(detail.missing_features)}",
            retryable=False,
        )
    return EvaluationUnavailable(mapping.get(code, "evaluator_error"), f"{reason}: {message}", retryable=retryable)


def _legacy_provenance(
    provenance: Mapping[str, Any], *, request_digest: str,
) -> EvaluatorProvenance:
    return EvaluatorProvenance(
        schema_version=1,
        epoch=provenance["evaluator_epoch"],
        site_id=provenance["site_id"],
        device_id=provenance["device_id"],
        backend=provenance["backend"],
        backend_version=provenance["backend_version"],
        target_id=provenance["target_id"],
        worker_build=provenance["worker_build"],
        task_pack_digest=provenance["task_pack_digest"],
        calibration_digest=provenance["calibration_digest"],
        protocol_version=provenance["protocol_version"],
        request_digest=request_digest,
        measurement_kind=provenance["measurement_kind"],
        publication_allowed=provenance["publication_allowed"],
    )


class GrpcEvaluator:
    """One grant-scoped view over the factory's shared channel."""

    transport_kind = "grpc"
    supports_explicit_retry = True

    def __init__(
        self,
        *,
        stub: pb_grpc.EvaluatorStub,
        token_provider: Callable[[str, str], str],
        run_id: str,
        session_id: str,
        target_id: str,
        expected: GrpcExpected,
        api_contract_ids: tuple[str, ...],
        evidence_contract_ids: tuple[str, ...],
        required_features: tuple[str, ...],
        keyring: SignatureKeyring,
        legacy_signature_key: bytes | None,
        rpc_timeout_s: float,
        watch_slice_s: float,
        job_margin_s: float,
        progress_publisher: Callable[[Path, dict[str, Any]], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.stub = stub
        self.token_provider = token_provider
        self.run_id = run_id
        self.session_id = session_id
        self.target_id = target_id
        self.expected = expected
        self.api_contract_ids = api_contract_ids
        self.evidence_contract_ids = evidence_contract_ids
        self.required_features = required_features
        self.keyring = keyring
        self.legacy_signature_key = legacy_signature_key
        self.signature_key = legacy_signature_key
        self.rpc_timeout_s = float(rpc_timeout_s)
        self.watch_slice_s = float(watch_slice_s)
        self.job_margin_s = float(job_margin_s)
        self.progress_publisher = progress_publisher
        self.clock = clock
        self._lock = threading.RLock()
        self._provenance: dict[str, Any] | None = None
        self._epoch = ""
        self._capability_id = ""
        self._expected_identities: Any = None
        self._active: dict[str, tuple[str, Any]] = {}

    # -- public adapter surface --------------------------------------------
    @property
    def provenance(self) -> EvaluatorProvenance | None:
        with self._lock:
            if self._provenance is None:
                return None
            return _legacy_provenance(self._provenance, request_digest="")

    @property
    def epoch(self) -> str:
        return self._epoch

    @property
    def api_contract_id(self) -> str:
        return contracts.API_CONTRACT_ID

    @property
    def evidence_contract_id(self) -> str:
        return contracts.EVIDENCE_CONTRACT_ID

    def execution_metadata(self) -> dict[str, Any]:
        return {
            "transport": self.transport_kind,
            "api_contract_id": contracts.API_CONTRACT_ID,
            "evidence_contract_id": contracts.EVIDENCE_CONTRACT_ID,
            "request_digest_algorithm": contracts.REQUEST_DIGEST_ALGORITHM,
            "result_signature_algorithm": contracts.RESULT_SIGNATURE_ALGORITHM,
            "result_signature_key_ids": list(self.keyring.key_ids),
        }

    def ensure_epoch(self, request: EvaluationRequest | None = None) -> str:
        del request
        self._handshake()
        return self._epoch

    def cancel(self, request_id: str) -> None:
        with self._lock:
            active = self._active.get(request_id)
        if active is None:
            return
        digest, expected = active
        try:
            self.stub.Cancel(
                pb.CancelRequest(job=pb.JobRef(
                    request_id=request_id,
                    request_digest=digest,
                    expected=gc.expected_identities_to_proto(expected),
                )),
                metadata=self._metadata(), timeout=self.rpc_timeout_s,
            )
        except grpc.RpcError:
            # Cancellation is best effort from the host; the durable job state
            # is reconciled by the next GetJob.
            return

    def formal(self, request: EvaluationRequest) -> EvaluationResult:
        if request.kind != "formal_eval":
            raise ValueError("formal() requires kind='formal_eval'")
        return self._evaluate(request, allow_retry=False)

    def selfcheck(self, request: EvaluationRequest) -> EvaluationResult:
        if request.kind != "selfcheck":
            raise ValueError("selfcheck() requires kind='selfcheck'")
        return self._evaluate(request, allow_retry=False)

    def formal_resume(self, request: EvaluationRequest) -> EvaluationResult:
        """Resume an interrupted formal job through the explicit Retry RPC."""

        if request.kind != "formal_eval":
            raise ValueError("formal_resume() requires kind='formal_eval'")
        return self._evaluate(request, allow_retry=True)

    def selfcheck_resume(self, request: EvaluationRequest) -> EvaluationResult:
        if request.kind != "selfcheck":
            raise ValueError("selfcheck_resume() requires kind='selfcheck'")
        return self._evaluate(request, allow_retry=True)

    # -- transport helpers --------------------------------------------------
    def _metadata(self) -> tuple[tuple[str, str], ...]:
        token = self.token_provider(self.run_id, self.session_id)
        if not isinstance(token, str) or not token:
            raise EvaluationUnavailable("grant_missing", "no session grant is available for this session")
        return (("authorization", f"Bearer {token}"),)

    def _external_request(self, request: EvaluationRequest) -> Any:
        from .job_types import ExternalEvaluationRequest

        try:
            external = ExternalEvaluationRequest.from_legacy(request)
        except (JobError, contracts.ContractError) as exc:
            raise EvaluationUnavailable("evaluation_request_invalid", str(exc)) from exc
        if external.run_id != self.run_id:
            raise EvaluationUnavailable("session_mismatch", "request run_id differs from this evaluator")
        if external.session_id != self.session_id:
            raise EvaluationUnavailable("session_mismatch", "request session_id differs from this evaluator")
        if external.target_id != self.target_id:
            raise EvaluationUnavailable("target_mismatch", "request target differs from this evaluator")
        return external

    def _handshake(self) -> pb.Capabilities:
        with self._lock:
            return self._handshake_locked()

    def _handshake_locked(self) -> pb.Capabilities:
        nonce = secrets.token_hex(16)
        expected = pb.ExpectedEvaluator(
            site_id=self.expected.site_id,
            device_id=self.expected.device_id,
            target_id=self.expected.target_id or self.target_id,
            worker_build=self.expected.worker_build,
            task_pack_digest=self.expected.task_pack_digest,
            evaluator_epoch=self.expected.evaluator_epoch or self._epoch,
        )
        request = pb.CapabilitiesRequest(
            api_contract_ids=list(self.api_contract_ids),
            evidence_contract_ids=list(self.evidence_contract_ids),
            required_features=list(self.required_features),
            client_nonce=nonce,
            expected=expected,
        )
        try:
            capabilities = self.stub.GetCapabilities(
                request, metadata=self._metadata(), timeout=self.rpc_timeout_s,
            )
        except grpc.RpcError as exc:
            raise _rpc_error(exc) from exc
        self._validate_capabilities(capabilities, nonce=nonce)
        with self._lock:
            self._capability_id = capabilities.capability_id
            self._provenance = gc.provenance_from_proto(capabilities.provenance, require_attempt=False)
            self._epoch = capabilities.evaluator_epoch
            self._expected_identities = gc.expected_identities_from_provenance(self._provenance)
        return capabilities

    def _validate_capabilities(self, capabilities: pb.Capabilities, *, nonce: str) -> None:
        if capabilities.client_nonce != nonce:
            raise EvaluationUnavailable("capability_invalid", "GetCapabilities did not echo the client nonce")
        if not capabilities.capability_id or not capabilities.server_nonce:
            raise EvaluationUnavailable("capability_invalid", "GetCapabilities omitted a capability identity")
        if capabilities.api_contract_id not in self.api_contract_ids:
            raise EvaluationUnavailable("api_contract_mismatch", "evaluator selected an unacceptable API contract")
        if capabilities.evidence_contract_id not in self.evidence_contract_ids:
            raise EvaluationUnavailable("evidence_contract_mismatch", "evaluator selected an unacceptable evidence contract")
        provenance = gc.provenance_from_proto(capabilities.provenance, require_attempt=False)
        if capabilities.evaluator_epoch != provenance["evaluator_epoch"]:
            raise EvaluationUnavailable("capability_invalid", "capabilities epoch and provenance disagree")
        if capabilities.api_contract_id != provenance["api_contract_id"]:
            raise EvaluationUnavailable("capability_invalid", "capabilities API contract and provenance disagree")
        if capabilities.evidence_contract_id != provenance["evidence_contract_id"]:
            raise EvaluationUnavailable("capability_invalid", "capabilities evidence contract and provenance disagree")
        for name, expected in self.expected.wildcards().items():
            if name == "evaluator_epoch":
                expected = expected or self._epoch
            if name == "target_id":
                expected = expected or self.target_id
            if expected and provenance[name] != expected:
                raise EvaluationUnavailable(f"{name}_mismatch", f"evaluator {name} differs from the host expectation")
        if self._epoch and provenance["evaluator_epoch"] != self._epoch:
            raise EvaluationUnavailable("epoch_drift", "evaluator epoch changed during the run")
        supported = set(capabilities.supported_features)
        missing = sorted(set(self.required_features) - supported)
        if missing:
            raise EvaluationUnavailable(
                "required_feature_missing", f"evaluator does not support required features: {missing}",
            )
        limits = capabilities.limits
        if limits.max_blob_bytes and limits.max_blob_bytes < 1:
            raise EvaluationUnavailable("capability_invalid", "evaluator advertises an unusable blob limit")

    # -- job flow -----------------------------------------------------------
    def _expected_for_submit(self) -> Any:
        if self._expected_identities is None:
            self._handshake()
        return self._expected_identities

    def _submit(self, external: Any) -> pb.JobSnapshot:
        expected = self._expected_for_submit()
        request = pb.SubmitRequest(
            request=gc.request_to_proto(external),
            request_digest=external.request_digest,
            expected=gc.expected_identities_to_proto(expected),
            capability_id=self._capability_id,
            required_features=list(self.required_features),
        )
        return self.stub.Submit(request, metadata=self._metadata(), timeout=self.rpc_timeout_s)

    def _get_job(self, external: Any) -> pb.JobSnapshot | None:
        expected = self._expected_for_submit()
        try:
            return self.stub.GetJob(
                pb.JobRef(
                    request_id=external.request_id,
                    request_digest=external.request_digest,
                    expected=gc.expected_identities_to_proto(expected),
                ),
                metadata=self._metadata(), timeout=self.rpc_timeout_s,
            )
        except grpc.RpcError as exc:
            error = _rpc_error(exc)
            if error.code == "job_not_found":
                return None
            raise error from exc

    def _evaluate(self, request: EvaluationRequest, *, allow_retry: bool) -> EvaluationResult:
        external = self._external_request(request)
        try:
            self._handshake()
            with self._lock:
                self._active[external.request_id] = (external.request_digest, self._expected_for_submit())
            snapshot = self._submit_with_grace(external)
            state = gc.state_from_proto(snapshot.state)
            if state == "interrupted" and allow_retry:
                snapshot = self._retry(external, snapshot)
                state = gc.state_from_proto(snapshot.state)
            if state == "completed":
                return self._result_from_snapshot(snapshot, external, request)
            if state in {"failed", "cancelled", "expired", "interrupted"}:
                raise self._snapshot_error(snapshot, state)
            return self._watch_to_result(external, request, snapshot)
        finally:
            with self._lock:
                self._active.pop(external.request_id, None)

    def _submit_with_grace(self, external: Any) -> pb.JobSnapshot:
        grace_retries = 0
        while True:
            try:
                return self._submit(external)
            except grpc.RpcError as exc:
                error = _rpc_error(exc)
                if error.code not in {"transport_unavailable", "transport_timeout", "transport_cancelled"}:
                    raise error from exc
                # Submit outcome unknown: reconcile by identity before resubmitting
                # the exact same payload.  An accepted job is never re-submitted.
                try:
                    snapshot = self._get_job(external)
                except EvaluationUnavailable:
                    snapshot = None
                if snapshot is not None:
                    gc.state_from_proto(snapshot.state)
                    return snapshot
                if grace_retries >= _MAX_GRACE_SUBMIT_RETRIES:
                    raise error from exc
                grace_retries += 1
                time.sleep(0.2)

    def _retry(self, external: Any, snapshot: pb.JobSnapshot) -> pb.JobSnapshot:
        if not snapshot.HasField("current_attempt"):
            raise EvaluationUnavailable("retry_refused", "interrupted job has no attempt to retry")
        attempt = snapshot.current_attempt
        self._handshake()
        expected = self._expected_for_submit()
        request = pb.RetryRequest(
            job=pb.JobRef(
                request_id=external.request_id,
                request_digest=external.request_digest,
                expected=gc.expected_identities_to_proto(expected),
            ),
            interrupted_attempt_id=attempt.attempt_id,
            capability_id=self._capability_id,
            required_features=list(self.required_features),
        )
        try:
            return self.stub.Retry(request, metadata=self._metadata(), timeout=self.rpc_timeout_s)
        except grpc.RpcError as exc:
            raise _rpc_error(exc) from exc

    def _snapshot_error(self, snapshot: pb.JobSnapshot, state: str) -> EvaluationUnavailable:
        if snapshot.HasField("failure"):
            failure = snapshot.failure
            code = failure.code or state
            message = failure.message or f"job {state}"
            return EvaluationUnavailable(code, message[:1000], retryable=bool(failure.retryable))
        if state == "interrupted":
            return EvaluationUnavailable(
                "interrupted", "the durable attempt was interrupted", retryable=True,
            )
        return EvaluationUnavailable(state, f"job ended in {state}")

    def _watch_to_result(
        self, external: Any, request: EvaluationRequest, initial: pb.JobSnapshot,
    ) -> EvaluationResult:
        after_sequence = int(initial.last_sequence)
        deadline = self.clock() + (
            (external.queue_timeout_ms + external.candidate_timeout_ms) / 1000.0 + self.job_margin_s
        )
        backoff = 0.1
        expected = self._expected_for_submit()
        while True:
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise EvaluationUnavailable(
                    "transport_timeout", "durable job did not finish within the host deadline", retryable=True,
                )
            watch = pb.WatchRequest(
                job=pb.JobRef(
                    request_id=external.request_id,
                    request_digest=external.request_digest,
                    expected=gc.expected_identities_to_proto(expected),
                ),
                after_sequence=after_sequence,
            )
            try:
                stream = self.stub.Watch(
                    watch, metadata=self._metadata(), timeout=min(self.watch_slice_s, remaining),
                )
                for event in stream:
                    if event.request_id != external.request_id:
                        raise EvaluationUnavailable("event_mismatch", "job event request_id does not match")
                    after_sequence = max(after_sequence, int(event.sequence))
                    if event.HasField("progress"):
                        self._publish_progress(request, event.progress)
                        continue
                    if not event.HasField("snapshot"):
                        raise EvaluationUnavailable("event_mismatch", "job event has no body")
                    snapshot = event.snapshot
                    state = gc.state_from_proto(snapshot.state)
                    if state == "completed":
                        return self._result_from_snapshot(snapshot, external, request)
                    if state in {"failed", "cancelled", "expired", "interrupted"}:
                        raise self._snapshot_error(snapshot, state)
            except grpc.RpcError as exc:
                error = _rpc_error(exc)
                if error.code not in {"transport_unavailable", "transport_timeout", "transport_cancelled", "aborted"}:
                    raise error from exc
            # The stream ended, was interrupted, or hit its slice deadline.
            # Reconcile with GetJob; Watch is only an optimization.
            try:
                current = self._get_job(external)
            except EvaluationUnavailable as exc:
                if exc.code in {"transport_unavailable", "transport_timeout", "transport_cancelled"}:
                    current = None
                else:
                    raise
            if current is not None:
                self._validate_snapshot_identity(current, external)
                state = gc.state_from_proto(current.state)
                after_sequence = max(after_sequence, int(current.last_sequence))
                if state == "completed":
                    return self._result_from_snapshot(current, external, request)
                if state in {"failed", "cancelled", "expired", "interrupted"}:
                    raise self._snapshot_error(current, state)
            time.sleep(backoff)
            backoff = min(backoff * 2, 1.0)

    def _publish_progress(self, request: EvaluationRequest, progress: pb.JobProgress) -> None:
        if request.progress_path is None:
            return
        stage = {
            pb.PROGRESS_STAGE_QUEUED: "queued",
            pb.PROGRESS_STAGE_STATIC_CHECK: "static",
            pb.PROGRESS_STAGE_COMPILE: "compile",
            pb.PROGRESS_STAGE_CORRECTNESS: "correctness",
            pb.PROGRESS_STAGE_TIMING: "bench",
            pb.PROGRESS_STAGE_FINALIZING: "done",
        }.get(progress.stage, "queued")
        observed = progress.observed_at
        updated_at = time.time()
        if observed.seconds or observed.nanos:
            updated_at = gc.timestamp_milliseconds(observed) / 1000.0
        payload = {
            "task": request.task,
            "dtype": request.dtype,
            "size": request.size,
            "mode": request.mode,
            "stage": stage,
            "status": "ok" if stage == "done" else "running",
            "message": progress.detail[:300],
            "updated_at": updated_at,
            "finished": stage == "done",
        }
        publisher = self.progress_publisher
        try:
            if publisher is not None:
                publisher(request.progress_path, payload)
            else:
                from tcenv.events import publish

                publish(request.progress_path, payload)
        except Exception:
            # Progress is advisory; it never fails an evaluation.
            return

    def _validate_snapshot_identity(self, snapshot: pb.JobSnapshot, external: Any) -> None:
        if snapshot.request_id != external.request_id:
            raise EvaluationUnavailable("snapshot_mismatch", "job snapshot request_id does not match")
        if snapshot.request_digest != external.request_digest:
            raise EvaluationUnavailable("snapshot_mismatch", "job snapshot request digest does not match")
        provenance = gc.provenance_from_proto(
            snapshot.provenance, require_attempt=snapshot.HasField("current_attempt"),
        )
        if provenance["request_digest"] != external.request_digest:
            raise EvaluationUnavailable("snapshot_mismatch", "job provenance request digest does not match")
        pinned = gc.expected_identities_from_provenance(provenance)
        if pinned != self._expected_for_submit():
            raise EvaluationUnavailable("snapshot_mismatch", "job snapshot identities differ from the accepted capability")

    def _result_from_snapshot(
        self, snapshot: pb.JobSnapshot, external: Any, request: EvaluationRequest,
    ) -> EvaluationResult:
        if not snapshot.HasField("result"):
            raise EvaluationUnavailable("result_missing", "completed job snapshot has no result envelope")
        try:
            (
                report_bytes, report_descriptor,
                evidence_bytes, evidence_descriptor,
                provenance, receipt, signature,
            ) = gc.result_from_proto(snapshot.result)
        except JobError as exc:
            raise EvaluationUnavailable("result_invalid", str(exc)) from exc
        if report_descriptor["schema_id"] != gc.REPORT_SCHEMA_ID:
            raise EvaluationUnavailable("result_invalid", "report schema is not accepted by this host")
        if evidence_descriptor is None:
            raise EvaluationUnavailable("result_invalid", "external results require candidate-binding evidence")
        if evidence_descriptor["schema_id"] != gc.EVIDENCE_SCHEMA_ID:
            raise EvaluationUnavailable("result_invalid", "evidence schema is not accepted by this host")
        if provenance["request_digest"] != external.request_digest:
            raise EvaluationUnavailable("provenance_mismatch", "result request digest does not match")
        if receipt["request_id"] != external.request_id:
            raise EvaluationUnavailable("receipt_invalid", "receipt request_id does not match")
        if receipt["request_digest"] != external.request_digest:
            raise EvaluationUnavailable("receipt_invalid", "receipt request digest does not match")
        if receipt["candidate_sha256"] != external.candidate_sha256:
            raise EvaluationUnavailable("receipt_invalid", "receipt candidate digest does not match")
        if provenance["attempt_id"] != receipt["attempt_id"]:
            raise EvaluationUnavailable("receipt_invalid", "provenance and receipt attempt ids differ")
        expected_receipt = contracts.attempt_receipt_id(
            request_id=external.request_id,
            attempt_id=receipt["attempt_id"],
            request_digest=external.request_digest,
        )
        if receipt["receipt_id"] != expected_receipt:
            raise EvaluationUnavailable("receipt_invalid", "receipt id does not bind this attempt")
        pinned = gc.expected_identities_from_provenance(provenance)
        if pinned != self._expected_for_submit():
            raise EvaluationUnavailable("provenance_mismatch", "result identities differ from the accepted capability")
        document = {
            "provenance": provenance,
            "report": report_descriptor,
            "evidence": evidence_descriptor,
            "receipt": receipt,
        }
        self.keyring.verify(
            site_id=provenance["site_id"],
            key_id=signature["key_id"],
            document=document,
            signature=signature["signature"],
        )
        try:
            report = gc.report_json(report_bytes)
            evidence = gc.evidence_json(evidence_bytes)
        except JobError as exc:
            raise EvaluationUnavailable("result_invalid", str(exc)) from exc
        if not isinstance(evidence, dict) or evidence.get("candidate_sha256") != external.candidate_sha256:
            raise EvaluationUnavailable("provenance_mismatch", "evidence does not bind the exact candidate bytes")
        legacy_receipt = dict(receipt)
        legacy_receipt["request_digest"] = request.request_digest
        legacy_receipt["measurement_kind"] = provenance["measurement_kind"]
        legacy_receipt["signature_key_id"] = signature["key_id"]
        legacy_receipt["signature_algorithm"] = signature["algorithm"]
        result = EvaluationResult(
            report=report,
            provenance=_legacy_provenance(provenance, request_digest=request.request_digest),
            evidence=evidence,
            receipt=legacy_receipt,
            signature="",
        )
        if self.legacy_signature_key is not None:
            result.signature = sign_result(result, self.legacy_signature_key)
        return result


class GrpcEvaluatorFactory:
    """Shared-channel factory; replaces ``RemoteEvaluatorPool`` for gRPC runs."""

    transport_kind = "grpc"
    supports_explicit_retry = True

    def __init__(
        self,
        *,
        endpoint: str,
        token_provider: Callable[[str, str], str],
        run_id: str,
        target_id: str,
        session_id: str = "",
        expected: GrpcExpected | None = None,
        api_contract_ids: Iterable[str] = (contracts.API_CONTRACT_ID,),
        evidence_contract_ids: Iterable[str] = (contracts.EVIDENCE_CONTRACT_ID,),
        required_features: Iterable[str] = DEFAULT_REQUIRED_FEATURES,
        keyring: SignatureKeyring | None = None,
        legacy_signature_key: bytes | None = None,
        credentials: grpc.ChannelCredentials | None = None,
        allow_insecure: bool = False,
        server_name_override: str = "",
        rpc_timeout_s: float = 30.0,
        watch_slice_s: float = 30.0,
        job_margin_s: float = 30.0,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        channel: grpc.Channel | None = None,
        progress_publisher: Callable[[Path, dict[str, Any]], None] | None = None,
    ) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("gRPC evaluator run_id is required")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("gRPC endpoint is required")
        if not callable(token_provider):
            raise ValueError("gRPC token_provider must be callable")
        if keyring is None:
            raise ValueError("gRPC results require a host signature keyring")
        if legacy_signature_key is not None and (not isinstance(legacy_signature_key, bytes) or not legacy_signature_key):
            raise ValueError("legacy_signature_key must be non-empty bytes or None")
        if not allow_insecure and credentials is None and channel is None:
            raise ValueError("gRPC requires TLS credentials; insecure channels must be requested explicitly")
        self.endpoint = endpoint
        self.token_provider = token_provider
        self.run_id = run_id
        self.session_id = session_id
        self.target_id = target_id
        expected = expected or GrpcExpected(target_id=target_id)
        if expected.target_id and expected.target_id != target_id:
            raise ValueError("expected target differs from the factory target")
        self.expected = expected
        self.api_contract_ids = tuple(api_contract_ids)
        self.evidence_contract_ids = tuple(evidence_contract_ids)
        if contracts.API_CONTRACT_ID not in self.api_contract_ids:
            raise ValueError("api_contract_ids must include the implemented API contract")
        if contracts.EVIDENCE_CONTRACT_ID not in self.evidence_contract_ids:
            raise ValueError("evidence_contract_ids must include the implemented evidence contract")
        self.required_features = tuple(dict.fromkeys(required_features))
        self.keyring = keyring
        self.legacy_signature_key = legacy_signature_key
        self.signature_key = legacy_signature_key
        self.rpc_timeout_s = float(rpc_timeout_s)
        self.watch_slice_s = float(watch_slice_s)
        self.job_margin_s = float(job_margin_s)
        self.max_message_bytes = int(max_message_bytes)
        self.progress_publisher = progress_publisher
        self._lock = threading.RLock()
        self._sessions: dict[tuple[str, str, str], GrpcEvaluator] = {}
        self._closed = False
        self._owns_channel = channel is None
        if channel is None:
            options = [
                ("grpc.max_send_message_length", self.max_message_bytes),
                ("grpc.max_receive_message_length", self.max_message_bytes),
                ("grpc.keepalive_time_ms", 30_000),
                ("grpc.keepalive_timeout_ms", 10_000),
                ("grpc.keepalive_permit_without_calls", 1),
            ]
            if server_name_override:
                options.append(("grpc.ssl_target_name_override", server_name_override))
            if allow_insecure:
                channel = grpc.insecure_channel(endpoint, options=options)
            else:
                channel = grpc.secure_channel(endpoint, credentials, options=options)
        self._channel = channel
        self._stub = pb_grpc.EvaluatorStub(channel)

    # -- properties used by the search loop -------------------------------
    @property
    def provenance(self) -> EvaluatorProvenance | None:
        return self._last().provenance

    @property
    def epoch(self) -> str:
        return self._last().epoch

    @property
    def api_contract_id(self) -> str:
        return contracts.API_CONTRACT_ID

    @property
    def evidence_contract_id(self) -> str:
        return contracts.EVIDENCE_CONTRACT_ID

    def execution_metadata(self) -> dict[str, Any]:
        return {
            "transport": self.transport_kind,
            "api_contract_id": contracts.API_CONTRACT_ID,
            "evidence_contract_id": contracts.EVIDENCE_CONTRACT_ID,
            "request_digest_algorithm": contracts.REQUEST_DIGEST_ALGORITHM,
            "result_signature_algorithm": contracts.RESULT_SIGNATURE_ALGORITHM,
            "result_signature_key_ids": list(self.keyring.key_ids),
        }

    def _last(self) -> GrpcEvaluator:
        with self._lock:
            if self._sessions:
                return next(iter(self._sessions.values()))
            return self._session(self.run_id, self.session_id or "unbound", self.target_id)

    def _session(self, run_id: str, session_id: str, target_id: str) -> GrpcEvaluator:
        if self._closed:
            raise EvaluationUnavailable("transport_closed", "gRPC evaluator factory is closed")
        if run_id != self.run_id:
            raise EvaluationUnavailable("session_mismatch", "request run_id differs from this evaluator")
        if target_id != self.target_id:
            raise EvaluationUnavailable("target_mismatch", "request target differs from this evaluator")
        key = (run_id, session_id, target_id)
        with self._lock:
            client = self._sessions.get(key)
            if client is None:
                client = GrpcEvaluator(
                    stub=self._stub,
                    token_provider=self.token_provider,
                    run_id=run_id,
                    session_id=session_id,
                    target_id=target_id,
                    expected=self.expected,
                    api_contract_ids=self.api_contract_ids,
                    evidence_contract_ids=self.evidence_contract_ids,
                    required_features=self.required_features,
                    keyring=self.keyring,
                    legacy_signature_key=self.legacy_signature_key,
                    rpc_timeout_s=self.rpc_timeout_s,
                    watch_slice_s=self.watch_slice_s,
                    job_margin_s=self.job_margin_s,
                    progress_publisher=self.progress_publisher,
                )
                self._sessions[key] = client
            return client

    def _for_request(self, request: EvaluationRequest) -> GrpcEvaluator:
        run_id = request.run_id or self.run_id
        session_id = request.session_id or self.session_id
        target_id = request.target_id or self.target_id
        if not session_id:
            raise EvaluationUnavailable("session_mismatch", "request has no session id for a gRPC evaluator")
        return self._session(run_id, session_id, target_id)

    # -- EvaluationExecutor ------------------------------------------------
    def ensure_epoch(self, request: EvaluationRequest) -> str:
        return self._for_request(request).ensure_epoch(request)

    def formal(self, request: EvaluationRequest) -> EvaluationResult:
        if request.kind != "formal_eval":
            raise ValueError("formal() requires kind='formal_eval'")
        return self._for_request(request).formal(request)

    def selfcheck(self, request: EvaluationRequest) -> EvaluationResult:
        if request.kind != "selfcheck":
            raise ValueError("selfcheck() requires kind='selfcheck'")
        return self._for_request(request).selfcheck(request)

    def formal_resume(self, request: EvaluationRequest) -> EvaluationResult:
        if request.kind != "formal_eval":
            raise ValueError("formal_resume() requires kind='formal_eval'")
        return self._for_request(request).formal_resume(request)

    def selfcheck_resume(self, request: EvaluationRequest) -> EvaluationResult:
        if request.kind != "selfcheck":
            raise ValueError("selfcheck_resume() requires kind='selfcheck'")
        return self._for_request(request).selfcheck_resume(request)

    def cancel(self, request_id: str) -> None:
        with self._lock:
            clients = list(self._sessions.values())
        for client in clients:
            client.cancel(request_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._sessions.clear()
        if self._owns_channel:
            self._channel.close()


def grpc_evaluator_factory_from_config(
    *,
    endpoint: str,
    token_provider: Callable[[str, str], str],
    run_id: str,
    target_id: str,
    ca_file: Path,
    certificate_file: Path | None = None,
    private_key_file: Path | None = None,
    keyring_file: Path,
    legacy_signature_key: bytes | None = None,
    expected: GrpcExpected | None = None,
    api_contract_ids: Iterable[str] = (contracts.API_CONTRACT_ID,),
    evidence_contract_ids: Iterable[str] = (contracts.EVIDENCE_CONTRACT_ID,),
    required_features: Iterable[str] = DEFAULT_REQUIRED_FEATURES,
    server_name_override: str = "",
    rpc_timeout_s: float = 30.0,
    watch_slice_s: float = 30.0,
    job_margin_s: float = 30.0,
) -> GrpcEvaluatorFactory:
    """Build a TLS factory from host-only files; there is no insecure fallback."""

    credentials = load_client_credentials(
        ca_file=Path(ca_file),
        certificate_file=None if certificate_file is None else Path(certificate_file),
        private_key_file=None if private_key_file is None else Path(private_key_file),
    )
    keyring = SignatureKeyring.from_file(keyring_file)
    return GrpcEvaluatorFactory(
        endpoint=endpoint,
        token_provider=token_provider,
        run_id=run_id,
        target_id=target_id,
        expected=expected,
        api_contract_ids=api_contract_ids,
        evidence_contract_ids=evidence_contract_ids,
        required_features=required_features,
        keyring=keyring,
        legacy_signature_key=legacy_signature_key,
        credentials=credentials,
        server_name_override=server_name_override,
        rpc_timeout_s=rpc_timeout_s,
        watch_slice_s=watch_slice_s,
        job_margin_s=job_margin_s,
    )
