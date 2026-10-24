"""Evaluation request/result types and the trusted local evaluator adapter.

The same request model is used by the local adapter and by the remote wire protocol.
Local-only paths stay on this dataclass for Phase 0b compatibility, but are never
serialized or included in request identity.  The production ``tcenv.remote`` package
imports these types from here; it never needs the search package ``taic``.
"""

from __future__ import annotations

import hashlib
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
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_KINDS = frozenset({"formal_eval", "selfcheck"})
_CACHE_POLICIES = frozenset({"reuse", "refresh", "disabled"})


class EvaluationError(ValueError):
    """A request, provenance or result cannot be accepted."""


def _canonical_json(document: Any) -> bytes:
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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

    def __post_init__(self) -> None:
        if not isinstance(self.source, str):
            raise EvaluationError("source must be a string")
        if not self.candidate_sha256:
            object.__setattr__(self, "candidate_sha256", _sha256(self.source.encode("utf-8")))
        self._validate()

    def _validate(self) -> None:
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
        source_bytes = self.source.encode("utf-8")
        if len(source_bytes) > MAX_SOURCE_BYTES:
            raise EvaluationError("source exceeds the byte limit")
        if _sha256(source_bytes) != self.candidate_sha256:
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

    @property
    def kernel_source(self) -> str:
        return self.source

    @property
    def source_bytes(self) -> bytes:
        return self.source.encode("utf-8")

    @property
    def request_digest(self) -> str:
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


def local_provenance(request: EvaluationRequest) -> EvaluatorProvenance:
    backend, backend_version = _backend_identity(request)
    epoch = _epoch(
        site_id="local", device_id=request.target_id, backend=backend,
        backend_version=backend_version, target_id=request.target_id,
        worker_build="", task_pack_digest="", calibration_digest="",
        protocol_version=PROTOCOL_VERSION,
    )
    return EvaluatorProvenance(
        schema_version=1, epoch=epoch, site_id="local", device_id=request.target_id,
        backend=backend, backend_version=backend_version, target_id=request.target_id,
        worker_build="", task_pack_digest="", calibration_digest="",
        protocol_version=PROTOCOL_VERSION, request_digest=request_digest(request),
        measurement_kind="measured", publication_allowed=True,
    )


class LocalEvaluator:
    """Adapter for the existing trusted local evaluator."""

    def __init__(self, evaluate_fn: Callable[..., EvalReport] | None = None) -> None:
        self._evaluate_fn = evaluate_fn or evaluate

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
