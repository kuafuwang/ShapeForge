"""Transport-neutral external evaluator job, attempt and evidence types."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from . import contracts

JOB_STATES = frozenset({
    "accepted", "queued", "running", "interrupted", "completed",
    "failed", "cancelled", "expired",
})
TERMINAL_JOB_STATES = frozenset({"completed", "failed", "cancelled", "expired"})
ATTEMPT_STATES = frozenset({"queued", "running", "interrupted", "completed", "failed", "cancelled"})
# Which execution supervisor started one attempt.  A frontend never guesses:
# recovery and release only clean the kind that durably created the attempt.
EXECUTION_PROCESS = "process"
EXECUTION_CONTAINER = "container"
EXECUTION_KINDS = frozenset({EXECUTION_PROCESS, EXECUTION_CONTAINER})
CANDIDATE_STATUSES = frozenset({
    "ok", "rejected", "compile_error", "abi_error", "unsafe", "incorrect",
    "runtime_error", "timeout", "crash", "unchanged", "duplicate",
    "external_duplicate", "refused_hang",
})
_ID = re.compile(r"[A-Za-z0-9._:-]{1,160}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class JobError(ValueError):
    """A sanitized transport-neutral RPC or durable-state error."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        failure_domain: str = "",
        request_id: str = "",
        attempt_id: str = "",
        missing_features: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message[:4000])
        self.code = code
        self.retryable = bool(retryable)
        self.failure_domain = failure_domain
        self.request_id = request_id
        self.attempt_id = attempt_id
        self.missing_features = tuple(missing_features)


class StateCorrupt(JobError):
    def __init__(self, message: str) -> None:
        super().__init__("data_loss", message, failure_domain="infrastructure")


class UntrustedSignature(StateCorrupt):
    """A stored result is signed by a key this frontend no longer trusts.

    Security revocation must not brick the whole frontend: the affected job is
    quarantined and its result is withheld, while other jobs keep serving.
    """


@dataclass(frozen=True)
class ExpectedEvaluator:
    site_id: str
    device_id: str
    target_id: str
    worker_build: str
    task_pack_digest: str
    evaluator_epoch: str

    def validate(self) -> None:
        for name in ("site_id", "device_id", "target_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _ID.fullmatch(value):
                raise JobError("invalid_argument", f"expected {name} is invalid")
        for name in ("worker_build",):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 256:
                raise JobError("invalid_argument", f"expected {name} is invalid")
        for name in ("task_pack_digest", "evaluator_epoch"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _HEX64.fullmatch(value):
                raise JobError("invalid_argument", f"expected {name} is invalid")

    def to_dict(self) -> dict[str, str]:
        return {
            "site_id": self.site_id,
            "device_id": self.device_id,
            "target_id": self.target_id,
            "worker_build": self.worker_build,
            "task_pack_digest": self.task_pack_digest,
            "evaluator_epoch": self.evaluator_epoch,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "ExpectedEvaluator":
        if not isinstance(document, Mapping) or set(document) != {
            "site_id", "device_id", "target_id", "worker_build", "task_pack_digest", "evaluator_epoch"
        }:
            raise JobError("invalid_argument", "expected evaluator fields are missing or unknown")
        value = cls(**dict(document))
        value.validate()
        return value


@dataclass(frozen=True)
class ExpectedIdentities:
    evaluator: ExpectedEvaluator
    api_contract_id: str = contracts.API_CONTRACT_ID
    evidence_contract_id: str = contracts.EVIDENCE_CONTRACT_ID

    def validate(self) -> None:
        self.evaluator.validate()
        if self.api_contract_id != contracts.API_CONTRACT_ID:
            raise JobError("failed_precondition", "unsupported API contract")
        if self.evidence_contract_id != contracts.EVIDENCE_CONTRACT_ID:
            raise JobError("failed_precondition", "unsupported evidence contract")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator": self.evaluator.to_dict(),
            "api_contract_id": self.api_contract_id,
            "evidence_contract_id": self.evidence_contract_id,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "ExpectedIdentities":
        if not isinstance(document, Mapping) or set(document) != {
            "evaluator", "api_contract_id", "evidence_contract_id"
        }:
            raise JobError("invalid_argument", "expected identity fields are missing or unknown")
        value = cls(
            evaluator=ExpectedEvaluator.from_dict(document["evaluator"]),
            api_contract_id=document["api_contract_id"],
            evidence_contract_id=document["evidence_contract_id"],
        )
        value.validate()
        return value


@dataclass(frozen=True)
class ExternalEvaluationRequest:
    """The validated v1 execution request with exact candidate bytes."""

    kind: str
    request_id: str
    run_id: str
    session_id: str
    task: str
    dtype: str
    size: str
    target_id: str
    task_fingerprint: str
    candidate_sha256: str
    source: bytes
    mode: str
    seed: str | None
    reps: int
    queue_timeout_ms: int
    candidate_timeout_ms: int
    cases: tuple[str, ...] | None
    fast_bench: bool = False
    require_contract_launch: bool = False
    validate: bool = False
    measure_serial: bool = False
    cache_policy: str = "reuse"

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not _ID.fullmatch(self.request_id):
            raise JobError("invalid_argument", "request_id must be a bounded opaque identifier")
        if not isinstance(self.cases, tuple) and self.cases is not None:
            raise JobError("invalid_argument", "cases must be a tuple or null")
        # This performs all source, enum, integer, UTF-8 and explicit-default checks.
        self.digest_document()

    def digest_fields(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "task": self.task,
            "dtype": self.dtype,
            "size": self.size,
            "target_id": self.target_id,
            "task_fingerprint": self.task_fingerprint,
            "candidate_sha256": self.candidate_sha256,
            "mode": self.mode,
            "seed": self.seed,
            "reps": self.reps,
            "queue_timeout_ms": self.queue_timeout_ms,
            "candidate_timeout_ms": self.candidate_timeout_ms,
            "cases": None if self.cases is None else list(self.cases),
            "fast_bench": self.fast_bench,
            "require_contract_launch": self.require_contract_launch,
            "validate": self.validate,
            "measure_serial": self.measure_serial,
            "cache_policy": self.cache_policy,
        }

    def digest_document(self) -> dict[str, Any]:
        try:
            return contracts.request_digest_document(self.digest_fields(), source_bytes=self.source)
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc

    @property
    def request_digest(self) -> str:
        try:
            return contracts.request_digest(self.digest_fields(), source_bytes=self.source)
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc

    @property
    def seed_int(self) -> int | None:
        return None if self.seed is None else int(self.seed)

    def without_source(self) -> dict[str, Any]:
        return {"request_id": self.request_id, **self.digest_fields()}

    def to_worker_dict(self) -> dict[str, Any]:
        """Private worker representation; base64 preserves arbitrary bytes."""

        return {**self.without_source(), "source_base64": base64.b64encode(self.source).decode("ascii")}

    @classmethod
    def from_worker_dict(cls, document: Mapping[str, Any]) -> "ExternalEvaluationRequest":
        if not isinstance(document, Mapping) or "source_base64" not in document:
            raise JobError("invalid_argument", "worker request has no exact source bytes")
        allowed = set(cls.without_source_fields()) | {"source_base64"}
        if set(document) != allowed:
            raise JobError("invalid_argument", "worker request fields are missing or unknown")
        try:
            source = base64.b64decode(document["source_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise JobError("invalid_argument", "worker source_base64 is invalid") from exc
        fields = {key: value for key, value in document.items() if key != "source_base64"}
        if fields["cases"] is not None:
            fields["cases"] = tuple(fields["cases"])
        return cls(**fields, source=source)

    @staticmethod
    def without_source_fields() -> tuple[str, ...]:
        return (
            "request_id", "kind", "run_id", "session_id", "task", "dtype", "size",
            "target_id", "task_fingerprint", "candidate_sha256", "mode", "seed", "reps",
            "queue_timeout_ms", "candidate_timeout_ms", "cases", "fast_bench",
            "require_contract_launch", "validate", "measure_serial", "cache_policy",
        )

    @classmethod
    def from_stored(cls, document: Mapping[str, Any], source: bytes) -> "ExternalEvaluationRequest":
        fields = dict(document)
        if fields.get("cases") is not None:
            fields["cases"] = tuple(fields["cases"])
        return cls(**fields, source=source)

    @classmethod
    def from_legacy(cls, request: Any) -> "ExternalEvaluationRequest":
        """Adapt ``tcenv.evaluation.EvaluationRequest`` without changing its digest."""

        from tcenv.evaluation import EvaluationRequest

        if not isinstance(request, EvaluationRequest):
            raise TypeError("request must be an EvaluationRequest")
        if not request.task_fingerprint:
            raise JobError("invalid_argument", "external requests require task_fingerprint")
        try:
            queue_ms = contracts.seconds_to_timeout_ms(request.queue_timeout_s)
            candidate_ms = contracts.seconds_to_timeout_ms(request.candidate_timeout_s)
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc
        return cls(
            kind=request.kind,
            request_id=request.request_id,
            run_id=request.run_id,
            session_id=request.session_id,
            task=request.task,
            dtype=request.dtype,
            size=request.size,
            target_id=request.target_id,
            task_fingerprint=request.task_fingerprint,
            candidate_sha256=request.candidate_sha256,
            source=request.source_bytes,
            mode=request.mode,
            seed=None if request.seed is None else str(request.seed),
            reps=request.reps,
            queue_timeout_ms=queue_ms,
            candidate_timeout_ms=candidate_ms,
            cases=request.cases,
            fast_bench=request.fast_bench,
            require_contract_launch=request.require_contract_launch,
            validate=request.validate,
            measure_serial=request.measure_serial,
            cache_policy=request.cache_policy,
        )

    def to_legacy(self) -> Any:
        """Adapt to the existing private worker/evaluator request when needed."""

        from tcenv.evaluation import EvaluationRequest

        try:
            source = self.source.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise JobError("invalid_argument", "candidate source is not supported UTF-8 text") from exc
        return EvaluationRequest(
            kind=self.kind,
            request_id=self.request_id,
            run_id=self.run_id,
            session_id=self.session_id,
            task=self.task,
            dtype=self.dtype,
            size=self.size,
            target_id=self.target_id,
            task_fingerprint=self.task_fingerprint,
            candidate_sha256=self.candidate_sha256,
            source=source,
            mode=self.mode,
            seed=self.seed_int,
            reps=self.reps,
            queue_timeout_s=self.queue_timeout_ms / 1000.0,
            candidate_timeout_s=self.candidate_timeout_ms / 1000.0,
            cases=self.cases,
            fast_bench=self.fast_bench,
            require_contract_launch=self.require_contract_launch,
            validate=self.validate,
            measure_serial=self.measure_serial,
            cache_policy=self.cache_policy,
        )


@dataclass(frozen=True)
class JobFailure:
    domain: str
    retryable: bool
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "retryable": self.retryable,
            "code": self.code,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any] | None) -> "JobFailure | None":
        if document is None:
            return None
        return cls(
            domain=str(document["domain"]),
            retryable=bool(document["retryable"]),
            code=str(document["code"]),
            message=str(document["message"]),
        )


@dataclass(frozen=True)
class Attempt:
    attempt_id: str
    lease_fence: int
    state: str
    retry_index: int
    cancel_requested: bool
    queued_at_ms: int
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    failure: JobFailure | None = None
    session_sequence: int = 0
    worker_pid: int | None = None
    worker_pgid: int | None = None
    worker_start_token: str = ""
    frontend_instance: str = ""
    execution_kind: str = EXECUTION_PROCESS
    container_id: str = ""
    container_name: str = ""
    container_image_ref: str = ""
    container_image_digest: str = ""
    container_runtime: str = ""


@dataclass(frozen=True)
class JobResult:
    report: bytes
    report_descriptor: dict[str, Any]
    evidence: bytes | None
    evidence_descriptor: dict[str, Any] | None
    provenance: dict[str, Any]
    receipt: dict[str, Any]
    signature: dict[str, Any] | None = None

    def report_json(self) -> dict[str, Any]:
        value = json.loads(self.report.decode("utf-8"))
        if not isinstance(value, dict):
            raise StateCorrupt("stored report is not an object")
        return value

    def evidence_json(self) -> dict[str, Any] | None:
        if self.evidence is None:
            return None
        value = json.loads(self.evidence.decode("utf-8"))
        if not isinstance(value, dict):
            raise StateCorrupt("stored evidence is not an object")
        return value


@dataclass(frozen=True)
class JobSnapshot:
    request_id: str
    request_digest: str
    state: str
    provenance: dict[str, Any]
    current_attempt: Attempt | None
    result: JobResult | None
    failure: JobFailure | None
    cancel_requested: bool
    last_sequence: int
    accepted_at_ms: int
    updated_at_ms: int
    expires_at_ms: int | None
    quarantined: bool = False
    recovery_error: str = ""


@dataclass(frozen=True)
class JobEvent:
    request_id: str
    attempt_id: str | None
    lease_fence: int
    sequence: int
    progress: dict[str, Any] | None = None
    snapshot: JobSnapshot | None = None


@dataclass(frozen=True)
class Capabilities:
    client_nonce: str
    server_nonce: str
    capability_id: str
    expected: ExpectedIdentities
    provenance: dict[str, Any]
    supported_features: frozenset[str]
    required_features: frozenset[str]
    issued_at_ms: int
    valid_for_ms: int
    limits: dict[str, int] = field(default_factory=dict)
    target_capabilities: tuple[dict[str, Any], ...] = ()
    signing_key_id: str = ""


@dataclass(frozen=True)
class QuotaUsage:
    reserved_count: int
    reserved_gpu_us: int
    used_count: int
    used_gpu_us: int
    hang: bool
