"""Evaluation request/result types and the trusted local evaluator adapter.

The same request model is used by the local adapter and by the remote wire protocol.
Local-only paths stay on this dataclass for Phase 0b compatibility, but are never
serialized or included in request identity.  The production ``tcenv.remote`` package
imports these types from here; it never needs the search package ``taic``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import uuid
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Protocol

from tcenv.evaluate import EvalReport, evaluate
from tcenv.schema import PROTOCOL_VERSION

LOCAL_ONLY_FIELDS = frozenset({"workdir", "progress_path", "self_check_dir", "timeout", "eval_cache", "eval_cache_refresh"})
MAX_SOURCE_BYTES = 1_000_000
MAX_TIMEOUT_S = 3_600.0
MAX_REPS = 10_000
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_KINDS = frozenset({"formal_eval", "selfcheck"})
_CACHE_POLICIES = frozenset({"reuse", "refresh", "disabled"})


class EvaluationError(ValueError):
    """A request, provenance or result cannot be accepted."""


class CandidateDeliveryError(EvaluationError):
    """The submitted text itself cannot become a candidate.

    An oversized submission, an unencodable (lone-surrogate) source or an envelope that does not
    decode are properties of what the model wrote, not faults of the harness.  They are a
    subclass of :class:`EvaluationError` so existing callers keep their behaviour, and a distinct
    type so the request boundary can turn exactly this into a typed ``rejected`` report instead of
    killing the run.
    """

    code = "candidate_delivery_invalid"

    def __init__(self, message: str, *, stage: str = "delivery") -> None:
        super().__init__(message)
        self.stage = str(stage or "delivery")


class EvaluationUnavailable(EvaluationError):
    """The evaluator could not provide a usable result.

    This is an infrastructure/policy outcome, not a candidate compile, numerical,
    timeout or crash verdict. ``code`` is stable enough for run bookkeeping and
    ``retryable`` tells resume whether the same request may be submitted again.
    """

    def __init__(self, code: str, message: str, *, retryable: bool = False, request_id: str = "") -> None:
        super().__init__(message)
        self.code = str(code or "evaluation_unavailable")
        self.retryable = bool(retryable)
        self.request_id = request_id


def _canonical_json(document: Any) -> bytes:
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _encode_source(source: str) -> bytes:
    """@brief Encode a submission as UTF-8 or fail as a delivery error.

    ``str.encode`` raises a bare ``UnicodeEncodeError`` for a lone surrogate.  The default
    ``candidate_sha256`` is derived before ``_validate`` runs, so the same normalization has to
    happen here as well.

    @param source submission text.
    @return UTF-8 bytes.
    @raises CandidateDeliveryError the text is not encodable.
    """
    try:
        return source.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CandidateDeliveryError(f"the submitted source is not encodable as UTF-8: {exc}") from exc


def _envelope_candidate_identity(source: str) -> str | None:
    """@brief 解码合法候选 envelope 并返回其 manifest 身份；普通源返回 None。

    Decoding here is deliberate: both the default constructor and ``from_dict`` must never fall
    back to the transport hash when the source is a segmented submission.  An envelope-shaped but
    invalid source is refused instead of being treated as a kernel.

    @param source 提交文本。
    @return 候选 manifest digest，或 None。
    @raises EvaluationError envelope 非法。
    """
    from .track.envelope import EnvelopeError, decode_envelope, is_candidate_envelope

    if not is_candidate_envelope(source):
        return None
    try:
        return decode_envelope(source).candidate_sha256
    except (EnvelopeError, UnicodeDecodeError) as exc:
        raise CandidateDeliveryError(f"candidate envelope is invalid: {exc}") from exc


@dataclass(frozen=True)
class EvaluationRequest:
    """One structured formal or self-check request."""

    task: str
    dtype: str
    size: str
    source: str
    mode: str = "full"
    seed: int | None = None
    reps: int = 15
    timeout: float | None = None  # local evaluate() option; never sent on the wire
    cases: tuple[str, ...] | None = None
    workdir: Path | None = None  # local-only
    progress_path: Path | None = None  # local-only
    fast_bench: bool = False
    require_contract_launch: bool = False
    self_check_dir: Path | None = None  # local-only
    validate: bool = False
    measure_serial: bool = False
    eval_cache: bool = False  # local-only; wire uses cache_policy
    eval_cache_refresh: bool = False  # local-only; bypass read, still write
    target_id: str = "apple-m4-max"
    kind: str = "formal_eval"
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    run_id: str = ""
    session_id: str = ""
    task_fingerprint: str = ""
    candidate_sha256: str = ""
    queue_timeout_s: float = 30.0
    candidate_timeout_s: float = 420.0
    cache_policy: str = "reuse"
    # Host-only NPU wire digest.  When set, the outbox/resume identity uses the
    # new contract's canonical request digest; the legacy to_dict/digest is
    # unchanged and this field never leaves the host.
    contract_request_digest: str = ""
    # Host-only wire CandidateDigest for the P0b blob contract.  A blob
    # candidate's bytes are not in ``source``, so its digest cannot be derived
    # here; the composition root binds the digest of the artifact it uploaded.
    # Empty keeps the historical ``sha256(source)`` identity, and this field
    # never enters ``to_dict``/the legacy digest.
    wire_candidate_digest: str = ""
    # Host-only CandidateDigest *algorithm* for the same value.  The bundle
    # contract has two artifact branches with different algorithms, so neither
    # the contract nor the value identifies it; empty keeps the historical
    # ``sha256``/``sha256-blob-v1`` choice from the contract identity.
    wire_candidate_algorithm: str = ""
    # Host-only BuildProvenance document of the P1c build contract.  It is an
    # independent field of the outbox row and the resume identity (the wire
    # request digest binds it), never part of the artifact set digest; empty
    # keeps every earlier contract byte-identical.
    wire_build_provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str):
            raise EvaluationError("source must be a string")
        envelope_identity = _envelope_candidate_identity(self.source)
        if not self.candidate_sha256:
            # A3: the default identity of a segmented submission is the candidate-entry manifest
            # digest, never sha256(envelope text).  A plain single file keeps the v1 byte rule.
            object.__setattr__(
                self,
                "candidate_sha256",
                envelope_identity if envelope_identity is not None else _sha256(_encode_source(self.source)),
            )
        self._validate(envelope_identity=envelope_identity)

    def _validate(self, *, envelope_identity: str | None = None) -> None:
        for name in ("request_id", "task", "dtype", "size", "target_id", "kind"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise EvaluationError(f"{name} must be a non-empty string")
        if not _REQUEST_ID.fullmatch(self.request_id):
            raise EvaluationError("request_id must be a bounded opaque identifier")
        if self.kind not in _KINDS:
            raise EvaluationError("kind must be formal_eval or selfcheck")
        allowed = {"quick", "full", "validate"} if self.kind == "formal_eval" else {"quick", "timed"}
        if self.mode not in allowed:
            raise EvaluationError(f"mode {self.mode!r} is not allowed for {self.kind}")
        if self.task_fingerprint and not _HEX64.fullmatch(self.task_fingerprint):
            raise EvaluationError("task_fingerprint must be 64 lowercase hex characters when supplied")
        if not _HEX64.fullmatch(self.candidate_sha256):
            raise EvaluationError("candidate_sha256 must be 64 lowercase hex characters")
        if not isinstance(self.source, str):
            raise EvaluationError("source must be a string")
        try:
            source_bytes = self.source.encode("utf-8")
        except UnicodeEncodeError as exc:
            # A lone surrogate cannot travel as UTF-8 at all; it is the submission's own defect.
            raise CandidateDeliveryError(f"the submitted source is not encodable as UTF-8: {exc}") from exc
        if len(source_bytes) > MAX_SOURCE_BYTES:
            raise CandidateDeliveryError(
                f"the submitted source is {len(source_bytes)} bytes; the limit is {MAX_SOURCE_BYTES} bytes"
            )
        if envelope_identity is not None:
            # The envelope is decoded before defaulting/validating; its transport hash is not an
            # accepted identity under any spelling.
            if self.candidate_sha256 != envelope_identity:
                raise EvaluationError(
                    "candidate_sha256 does not match the envelope's candidate manifest; the envelope transport "
                    "hash is not a candidate identity"
                )
        elif _sha256(source_bytes) != self.candidate_sha256:
            raise EvaluationError("candidate_sha256 does not match the actual source bytes")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0):
            raise EvaluationError("seed must be a non-negative integer or null")
        if isinstance(self.reps, bool) or not isinstance(self.reps, int) or not 1 <= self.reps <= MAX_REPS:
            raise EvaluationError("reps is outside the limit")
        for name in ("queue_timeout_s", "candidate_timeout_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= MAX_TIMEOUT_S:
                raise EvaluationError(f"{name} must be a finite positive timeout")
        if self.cases is not None:
            if not isinstance(self.cases, tuple) or any(not isinstance(case, str) or not case for case in self.cases):
                raise EvaluationError("cases must be a tuple of non-empty strings")
        for name in ("fast_bench", "require_contract_launch", "validate", "measure_serial", "eval_cache", "eval_cache_refresh"):
            if not isinstance(getattr(self, name), bool):
                raise EvaluationError(f"{name} must be boolean")
        if self.cache_policy not in _CACHE_POLICIES:
            raise EvaluationError("cache_policy must be reuse, refresh or disabled")
        if self.contract_request_digest and not _HEX64.fullmatch(self.contract_request_digest):
            raise EvaluationError("contract_request_digest must be 64 lowercase hex characters")
        if self.wire_candidate_digest and not _HEX64.fullmatch(self.wire_candidate_digest):
            raise EvaluationError("wire_candidate_digest must be 64 lowercase hex characters")
        if self.wire_candidate_digest and not self.contract_request_digest:
            # A blob candidate only makes sense on the contract that binds its
            # descriptor; a bare candidate digest would be an unreviewed identity.
            raise EvaluationError("wire_candidate_digest requires the contract_request_digest identity")
        if self.wire_build_provenance is not None:
            if not isinstance(self.wire_build_provenance, dict):
                raise EvaluationError("wire_build_provenance must be a document or None")
            if not self.wire_candidate_digest or not self.contract_request_digest:
                # A build provenance without the wire identity it belongs to
                # would be an unreviewed field on a legacy row.
                raise EvaluationError("wire_build_provenance requires the contract wire identity")
        if self.wire_candidate_algorithm and not _IDENTIFIER.fullmatch(self.wire_candidate_algorithm):
            # The algorithm may be bound before (or without) a value: the durable
            # outbox is what refuses an artifact candidate with no digest, so this
            # field stays a plain host-side annotation.
            raise EvaluationError("wire_candidate_algorithm must be a bounded identifier")

    @property
    def kernel_source(self) -> str:
        return self.source

    @property
    def source_bytes(self) -> bytes:
        return self.source.encode("utf-8")

    @property
    def request_digest(self) -> str:
        if self.contract_request_digest:
            return self.contract_request_digest
        return request_digest(self)

    def to_dict(self) -> dict[str, Any]:
        """Return the wire body; local-only fields never leave the host."""
        return {
            "kind": self.kind,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "task": self.task,
            "dtype": self.dtype,
            "size": self.size,
            "target_id": self.target_id,
            "task_fingerprint": self.task_fingerprint,
            "candidate_sha256": self.candidate_sha256,
            "source": self.source,
            "mode": self.mode,
            "seed": self.seed,
            "reps": self.reps,
            "queue_timeout_s": self.queue_timeout_s,
            "candidate_timeout_s": self.candidate_timeout_s,
            "cases": list(self.cases) if self.cases is not None else None,
            "fast_bench": self.fast_bench,
            "require_contract_launch": self.require_contract_launch,
            "validate": self.validate,
            "measure_serial": self.measure_serial,
            "cache_policy": self.cache_policy,
        }

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> "EvaluationRequest":
        if not isinstance(document, dict):
            raise EvaluationError("evaluation request must be an object")
        document = dict(document)
        document.pop("schema_version", None)
        allowed = {
            "kind", "request_id", "run_id", "session_id", "task", "dtype", "size", "target_id",
            "task_fingerprint", "candidate_sha256", "source", "mode", "seed", "reps",
            "queue_timeout_s", "candidate_timeout_s", "cases", "fast_bench", "require_contract_launch",
            "validate", "measure_serial", "cache_policy",
        }
        unknown = set(document) - allowed
        if unknown:
            raise EvaluationError(f"unknown evaluation request fields: {sorted(unknown)}")
        missing = allowed - set(document)
        if missing:
            raise EvaluationError(f"missing evaluation request fields: {sorted(missing)}")
        cases = document.get("cases")
        if cases is not None and not isinstance(cases, list):
            raise EvaluationError("cases must be a list or null")
        request = cls(**{**document, "cases": tuple(cases) if cases is not None else None})
        if not _HEX64.fullmatch(request.task_fingerprint):
            raise EvaluationError("task_fingerprint must be 64 lowercase hex characters")
        return request


@dataclass(frozen=True)
class EvaluatorProvenance:
    """Typed identity of the evaluator that produced a result."""

    schema_version: int
    epoch: str
    site_id: str
    device_id: str
    backend: str
    backend_version: str
    target_id: str
    worker_build: str
    task_pack_digest: str
    calibration_digest: str
    protocol_version: str
    request_digest: str = ""
    measurement_kind: str = "measured"
    publication_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "epoch": self.epoch,
            "site_id": self.site_id,
            "device_id": self.device_id,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "target_id": self.target_id,
            "worker_build": self.worker_build,
            "task_pack_digest": self.task_pack_digest,
            "calibration_digest": self.calibration_digest,
            "protocol_version": self.protocol_version,
            "request_digest": self.request_digest,
            "measurement_kind": self.measurement_kind,
            "publication_allowed": self.publication_allowed,
        }

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> "EvaluatorProvenance":
        if not isinstance(document, dict):
            raise EvaluationError("provenance must be an object")
        allowed = {
            "schema_version", "epoch", "site_id", "device_id", "backend", "backend_version", "target_id",
            "worker_build", "task_pack_digest", "calibration_digest", "protocol_version", "request_digest",
            "measurement_kind", "publication_allowed",
        }
        unknown = set(document) - allowed
        if unknown:
            raise EvaluationError(f"unknown provenance fields: {sorted(unknown)}")
        missing = allowed - set(document)
        if missing:
            raise EvaluationError(f"missing provenance fields: {sorted(missing)}")
        return cls(**document)


@dataclass
class EvaluationResult:
    """A report plus the identity evidence needed to compare it safely."""

    report: EvalReport | dict[str, Any]
    provenance: EvaluatorProvenance
    evidence: dict | None = None
    receipt: dict | None = None
    signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        report = self.report.to_json() if isinstance(self.report, EvalReport) else self.report
        return {
            "report": report,
            "provenance": self.provenance.to_dict(),
            "evidence": self.evidence,
            "receipt": self.receipt,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> "EvaluationResult":
        if not isinstance(document, dict):
            raise EvaluationError("evaluation result must be an object")
        report = document.get("report")
        if not isinstance(report, dict):
            raise EvaluationError("evaluation result report must be an object")
        provenance = document.get("provenance")
        if not isinstance(provenance, dict):
            raise EvaluationError("evaluation result provenance must be an object")
        return cls(
            report=report,
            provenance=EvaluatorProvenance.from_dict(provenance),
            evidence=document.get("evidence"),
            receipt=document.get("receipt"),
            signature=str(document.get("signature") or ""),
        )


def receipt_id_for(request: EvaluationRequest) -> str:
    """Deterministic receipt id for one request/digest pair."""
    return _sha256(_canonical_json({"request_id": request.request_id, "request_digest": request.request_digest}))


def validate_receipt(result: EvaluationResult, request: EvaluationRequest) -> str:
    """Validate the receipt/request binding and return its receipt id."""
    receipt = result.receipt
    if not isinstance(receipt, dict):
        raise EvaluationError("remote result has no receipt object")
    receipt_id = receipt.get("receipt_id")
    if not isinstance(receipt_id, str) or not receipt_id or len(receipt_id) > 160:
        raise EvaluationError("receipt_id must be a non-empty bounded string")
    if receipt.get("request_id") != request.request_id:
        raise EvaluationError("receipt request_id does not match the request")
    if receipt.get("request_digest") != request.request_digest:
        raise EvaluationError("receipt request digest does not match the request")
    if "candidate_sha256" in receipt and receipt["candidate_sha256"] != request.candidate_sha256:
        raise EvaluationError("receipt candidate digest does not match the request")
    if "candidate_digest" in receipt:
        candidate = receipt["candidate_digest"]
        # A blob candidate's digest is the uploaded artifact's, bound by the host
        # composition root (``wire_candidate_digest``); the inline candidate keeps
        # the original sha256(source) identity and algorithm.
        expected_algorithm = (
            "sha256-blob-v1" if request.wire_candidate_digest and not request.wire_candidate_algorithm
            else request.wire_candidate_algorithm or "sha256"
        )
        expected_value = request.wire_candidate_digest or request.candidate_sha256
        if (
            not isinstance(candidate, dict)
            or set(candidate) != {"algorithm", "value"}
            or candidate.get("algorithm") != expected_algorithm
            or candidate.get("value") != expected_value
        ):
            raise EvaluationError("receipt CandidateDigest does not match the request")
        if "candidate_sha256" in receipt:
            raise EvaluationError("the NPU receipt must not dual-write candidate_sha256")
    if "measurement_kind" in receipt and receipt["measurement_kind"] != result.provenance.measurement_kind:
        raise EvaluationError("receipt measurement kind does not match provenance")
    return receipt_id


def result_signature_payload(result: EvaluationResult) -> bytes:
    """Canonical bytes covered by an evaluator result signature."""
    report = result.report.to_json() if isinstance(result.report, EvalReport) else result.report
    return _canonical_json({
        "report": report,
        "provenance": result.provenance.to_dict(),
        "evidence": result.evidence,
        "receipt": result.receipt,
    })


def sign_result(result: EvaluationResult, key: bytes) -> str:
    """Return an HMAC-SHA256 signature for a result envelope."""
    if not isinstance(key, bytes) or not key:
        raise ValueError("a non-empty signature key is required")
    return hmac.new(key, result_signature_payload(result), hashlib.sha256).hexdigest()


def verify_result_signature(result: EvaluationResult, key: bytes) -> bool:
    """Verify a result signature without accepting an empty/malformed signature."""
    if not isinstance(key, bytes) or not key or not isinstance(result.signature, str):
        return False
    expected = sign_result(result, key)
    return hmac.compare_digest(result.signature, expected)


class EvaluationExecutor(Protocol):
    def formal(self, request: EvaluationRequest) -> EvaluationResult: ...
    def selfcheck(self, request: EvaluationRequest) -> EvaluationResult: ...


def request_digest(request: EvaluationRequest) -> str:
    """Digest the wire body, excluding the idempotency key and local-only fields."""
    body = request.to_dict()
    body.pop("request_id", None)
    return _sha256(_canonical_json(body))


def _backend_identity(request: EvaluationRequest) -> tuple[str, str]:
    try:
        from tcenv.target import load_target

        profile = load_target(request.target_id)
        return profile.backend, profile.backend_version
    except Exception:
        return "", ""


def _epoch(*, site_id: str, device_id: str, backend: str, backend_version: str,
           target_id: str, worker_build: str, task_pack_digest: str,
           calibration_digest: str, protocol_version: str) -> str:
    return _sha256(_canonical_json({
        "site_id": site_id, "device_id": device_id, "backend": backend,
        "backend_version": backend_version, "target_id": target_id,
        "worker_build": worker_build, "task_pack_digest": task_pack_digest,
        "calibration_digest": calibration_digest, "protocol_version": protocol_version,
    }))


def local_task_pack_digest(request: EvaluationRequest) -> str:
    """@brief 本地测量上下文的 task-pack 摘要（与部署侧同一 manifest 形状）。

    The local process has no deployed pack directory, so it uses the pack file set registered
    in the task's track binding.  A task that cannot be resolved or has no registration is a
    task-assembly/deployment error: leaving the digest empty would keep an old evaluator epoch
    that no longer describes the measurement context.

    @param request 评测请求。
    @return 十六进制 sha256。
    @raises EvaluationError 任务无法解析或没有 pack 登记。
    """
    from tasks import get_task

    try:
        task = get_task(request.task, request.dtype, request.size)
    except Exception as exc:  # noqa: BLE001 - an unknown task is not a measurement context
        raise EvaluationError(
            f"local task pack: cannot resolve {request.task}/{request.dtype}/{request.size}: {exc}"
        ) from exc
    pack = (task.evaluation_implementations or {}).get("task_pack")
    digest = str(pack.get("digest") or "") if isinstance(pack, dict) else ""
    if not digest:
        raise EvaluationError(
            f"local task pack: {request.task}/{request.dtype}/{request.size} has no registered pack digest"
        )
    return digest


def local_provenance(request: EvaluationRequest) -> EvaluatorProvenance:
    backend, backend_version = _backend_identity(request)
    pack_digest = local_task_pack_digest(request)
    epoch = _epoch(
        site_id="local", device_id=request.target_id, backend=backend,
        backend_version=backend_version, target_id=request.target_id,
        worker_build="", task_pack_digest=pack_digest, calibration_digest="",
        protocol_version=PROTOCOL_VERSION,
    )
    return EvaluatorProvenance(
        schema_version=1, epoch=epoch, site_id="local", device_id=request.target_id,
        backend=backend, backend_version=backend_version, target_id=request.target_id,
        worker_build="", task_pack_digest=pack_digest, calibration_digest="",
        protocol_version=PROTOCOL_VERSION, request_digest=request_digest(request),
        measurement_kind="measured", publication_allowed=True,
    )


class LocalEvaluator:
    """Adapter for the existing trusted local evaluator."""

    def __init__(
        self, evaluate_fn: Callable[..., EvalReport] | None = None, *,
        expected_identity: Callable[..., dict] | None = None,
    ) -> None:
        self._evaluate_fn = evaluate_fn or evaluate
        #: The current, recomputable confirmation expected.  The real local path resolves it from
        #: the target/backend registry; a test double may inject its own (the orchestrator must
        #: never take the search report's own values as today's expectation).
        self._expected_identity = expected_identity

    def confirmation_expected(self, *, task: str, dtype: str, size: str, target_id: str, source: str) -> dict:
        """@brief 当前执行器下的冻结 expected（探测失败时返回 identity 不完整的文档）。

        @param task 任务名。
        @param dtype 元素类型短名。
        @param size 问题规模名。
        @param target_id 目标设备 id。
        @param source 候选源码。
        @return expected 文档；不可重算的分量为 ``None``（读方按 unknown 处理）。
        """
        if self._expected_identity is not None:
            return self._expected_identity(task=task, dtype=dtype, size=size, target_id=target_id, source=source)
        from .evaluate import confirmation_expected_document

        # Only the real local executor can state today's toolchain profile.  A custom evaluate
        # function is not the Metal executor; claiming its profile would be the same fabrication
        # as copying the search report, so it probes nothing and the build identity stays unknown.
        return confirmation_expected_document(
            task, dtype, size, target_id, source, probe=self._evaluate_fn is evaluate,
        )

    def _evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        if request.mode == "validate":
            raise ValueError("local adapter does not support mode='validate'; use quick/timed/full or the worker validate path")
        report = self._evaluate_fn(
            request.task, request.dtype, request.size, request.source,
            mode=request.mode, seed=request.seed, reps=request.reps,
            timeout=request.timeout,
            cases=None if request.cases is None else list(request.cases),
            workdir=request.workdir, progress_path=request.progress_path,
            fast_bench=request.fast_bench,
            require_contract_launch=request.require_contract_launch,
            self_check_dir=request.self_check_dir, validate=request.validate,
            measure_serial=request.measure_serial, eval_cache=request.eval_cache,
            eval_cache_refresh=request.eval_cache_refresh,
            target_id=request.target_id,
        )
        return EvaluationResult(report=report, provenance=local_provenance(request))

    def formal(self, request: EvaluationRequest) -> EvaluationResult:
        if request.kind != "formal_eval":
            raise ValueError("formal() requires kind='formal_eval'")
        return self._evaluate(request)

    def selfcheck(self, request: EvaluationRequest) -> EvaluationResult:
        if request.kind != "selfcheck":
            raise ValueError("selfcheck() requires kind='selfcheck'")
        return self._evaluate(request)
