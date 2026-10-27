"""Transport-neutral external evaluator job, attempt and evidence types."""

from __future__ import annotations

import base64
import hashlib
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
        try:
            contracts.dialect_for(self.api_contract_id, self.evidence_contract_id)
        except contracts.ContractError as exc:
            raise JobError("failed_precondition", "unsupported API/evidence contract pair") from exc

    @property
    def contract(self) -> contracts.ContractDialect:
        try:
            return contracts.dialect_for(self.api_contract_id, self.evidence_contract_id)
        except contracts.ContractError as exc:
            raise JobError("failed_precondition", "unsupported API/evidence contract pair") from exc

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
class CandidateDigest:
    """P0 source candidate identity; algorithm is always sha256."""

    algorithm: str
    value: str

    def __post_init__(self) -> None:
        try:
            document = contracts.candidate_digest_document(
                {"algorithm": self.algorithm, "value": self.value}
            )
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc
        if document["algorithm"] != self.algorithm or document["value"] != self.value:
            raise JobError("invalid_argument", "candidate digest was not normalized")

    def to_dict(self) -> dict[str, str]:
        return {"algorithm": self.algorithm, "value": self.value}


@dataclass(frozen=True)
class DeviceGroupBinding:
    """Durable logical device group identity used by the NPU contract."""

    device_group_id: str
    vendor_id: str
    profile_digest: str
    topology_digest: str
    resource_generation: str

    def __post_init__(self) -> None:
        try:
            contracts.device_group_document(self.to_dict())
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc

    def to_dict(self) -> dict[str, str]:
        return {
            "device_group_id": self.device_group_id,
            "vendor_id": self.vendor_id,
            "profile_digest": self.profile_digest,
            "topology_digest": self.topology_digest,
            "resource_generation": self.resource_generation,
        }


@dataclass(frozen=True)
class NpuExecutionSpec:
    """P0 source-only execution shape."""

    device_group: DeviceGroupBinding
    source_media_type: str = "text/plain; charset=utf-8"
    artifact_kind: str = "source"
    launch_kind: str = "KERNEL"
    launch_subtype: str = "SIMT"
    required_timing_sources: tuple[str, ...] = ("DEVICE_EVENT",)
    synchronization: str = "stream-event"

    def __post_init__(self) -> None:
        if not isinstance(self.device_group, DeviceGroupBinding):
            raise JobError("invalid_argument", "execution requires a typed DeviceGroupBinding")
        try:
            contracts.npu_execution_document(self.to_dict())
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "launch_kind": self.launch_kind,
            "launch_subtype": self.launch_subtype,
            "source_media_type": self.source_media_type,
            "device_group": self.device_group.to_dict(),
            "required_timing_sources": list(self.required_timing_sources),
            "synchronization": self.synchronization,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "NpuExecutionSpec":
        if not isinstance(document, Mapping):
            raise JobError("invalid_argument", "execution spec must be an object")
        try:
            normalized = contracts.npu_execution_document(document)
            group = DeviceGroupBinding(**normalized["device_group"])
            return cls(
                device_group=group,
                source_media_type=normalized["source_media_type"],
                artifact_kind=normalized["artifact_kind"],
                launch_kind=normalized["launch_kind"],
                launch_subtype=normalized["launch_subtype"],
                required_timing_sources=tuple(normalized["required_timing_sources"]),
                synchronization=normalized["synchronization"],
            )
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc


@dataclass(frozen=True)
class NpuSourceRequest:
    """P0 NPU request payload: CandidateDigest plus source-only execution."""

    candidate_digest: CandidateDigest
    execution: NpuExecutionSpec

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_digest, CandidateDigest):
            raise JobError("invalid_argument", "NPU request requires a typed CandidateDigest")
        if not isinstance(self.execution, NpuExecutionSpec):
            raise JobError("invalid_argument", "NPU request requires a typed execution spec")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_digest": self.candidate_digest.to_dict(),
            "execution": self.execution.to_dict(),
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "NpuSourceRequest":
        if not isinstance(document, Mapping) or set(document) != {"candidate_digest", "execution"}:
            raise JobError("invalid_argument", "NPU request fields are missing or unknown")
        try:
            digest = contracts.candidate_digest_document(document["candidate_digest"])
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc
        return cls(
            candidate_digest=CandidateDigest(digest["algorithm"], digest["value"]),
            execution=NpuExecutionSpec.from_dict(document["execution"]),
        )


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
    npu: NpuSourceRequest | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not _ID.fullmatch(self.request_id):
            raise JobError("invalid_argument", "request_id must be a bounded opaque identifier")
        if not isinstance(self.cases, tuple) and self.cases is not None:
            raise JobError("invalid_argument", "cases must be a tuple or null")
        # The two contracts never share a candidate identity: the NPU request
        # carries CandidateDigest and must not dual-write the legacy field.
        if self.npu is not None and self.candidate_sha256:
            raise JobError(
                "invalid_argument",
                "the NPU contract must not also carry the legacy candidate_sha256",
            )
        # This performs all source, enum, integer, UTF-8 and explicit-default checks.
        self.digest_document()

    def digest_fields(self) -> dict[str, Any]:
        common = {
            "kind": self.kind,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "task": self.task,
            "dtype": self.dtype,
            "size": self.size,
            "target_id": self.target_id,
            "task_fingerprint": self.task_fingerprint,
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
        if self.npu is None:
            return {"candidate_sha256": self.candidate_sha256, **common}
        return {"candidate_digest": self.npu.candidate_digest.to_dict(), "execution": self.npu.execution.to_dict(), **common}

    @property
    def contract(self) -> contracts.ContractDialect:
        return contracts.NPU_SOURCE_DIALECT if self.npu is not None else contracts.LEGACY_DIALECT

    def digest_document(self) -> dict[str, Any]:
        try:
            return contracts.request_digest_document(
                self.digest_fields(), source_bytes=self.source, dialect=self.contract,
            )
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc

    @property
    def request_digest(self) -> str:
        try:
            return contracts.request_digest(
                self.digest_fields(), source_bytes=self.source, dialect=self.contract,
            )
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc

    @property
    def candidate_digest(self) -> dict[str, str]:
        if self.npu is not None:
            return self.npu.candidate_digest.to_dict()
        return {"algorithm": "sha256", "value": self.candidate_sha256}

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
        try:
            source = base64.b64decode(document["source_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise JobError("invalid_argument", "worker source_base64 is invalid") from exc
        if "execution" in document or "candidate_digest" in document:
            allowed = (set(cls.without_source_fields()) - {"candidate_sha256"}) | {
                "candidate_digest", "execution", "source_base64",
            }
            if set(document) != allowed:
                raise JobError("invalid_argument", "worker NPU request fields are missing or unknown")
            fields = {key: value for key, value in document.items() if key not in {"source_base64", "candidate_digest", "execution"}}
            if fields["cases"] is not None:
                fields["cases"] = tuple(fields["cases"])
            npu = NpuSourceRequest.from_dict({
                "candidate_digest": document["candidate_digest"],
                "execution": document["execution"],
            })
            request = cls(**fields, source=source, candidate_sha256="", npu=npu)
            if request.npu.candidate_digest.value != hashlib.sha256(source).hexdigest():
                raise JobError("invalid_argument", "worker NPU candidate digest does not match the source")
            return request
        allowed = set(cls.without_source_fields()) | {"source_base64"}
        if set(document) != allowed:
            raise JobError("invalid_argument", "worker request fields are missing or unknown")
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
        if "execution" in fields or "candidate_digest" in fields:
            if "execution" not in fields or "candidate_digest" not in fields:
                raise JobError("invalid_argument", "stored NPU request is incomplete")
            if "candidate_sha256" in fields:
                # The two candidate identities never coexist.  Dropping the
                # legacy field here would silently accept a mixed request, so it
                # is a stored-state corruption instead.
                raise JobError("invalid_argument", "stored NPU request also carries candidate_sha256")
            allowed = (set(cls.without_source_fields()) - {"candidate_sha256"}) | {
                "candidate_digest", "execution",
            }
            if set(fields) != allowed:
                raise JobError("invalid_argument", "stored NPU request fields are missing or unknown")
            npu = NpuSourceRequest.from_dict({
                "candidate_digest": fields.pop("candidate_digest"),
                "execution": fields.pop("execution"),
            })
            request = cls(**fields, source=source, candidate_sha256="", npu=npu)
            if request.npu.candidate_digest.value != hashlib.sha256(source).hexdigest():
                raise JobError("invalid_argument", "stored NPU candidate digest does not match the source")
            return request
        return cls(**fields, source=source)

    @classmethod
    def from_legacy(
        cls, request: Any, *, npu: NpuSourceRequest | None = None,
    ) -> "ExternalEvaluationRequest":
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
            candidate_sha256="" if npu is not None else request.candidate_sha256,
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
            npu=npu,
        )

    def to_legacy(self) -> Any:
        """Adapt to the existing private worker/evaluator request when needed."""

        if self.npu is not None:
            # The NPU contract has no legacy evaluator mapping: silently
            # downgrading it would drop CandidateDigest, the device group and
            # the device-side timing evidence.
            raise JobError("invalid_argument", "the NPU contract has no legacy request mapping")
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
    container_daemon: str = ""
    # The logical device group/profile/topology/resource generation this lease
    # was accepted under.  Empty for the legacy Metal/source contract.
    device_binding: Mapping[str, str] | None = None


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
    # The evaluator identities the job was accepted under; a frontend must not
    # execute/sign it under any other build or epoch.
    expected: ExpectedIdentities | None = None


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
