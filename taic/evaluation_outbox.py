"""Run-local durable outbox for formal evaluation requests.

The outbox is a sidecar, not a replacement for EvalReport or the search ledger.
It contains request identity and the result envelope, never provider credentials
or the candidate source. A request is marked pending before it is sent; a
completed result can be replayed after a host restart, while an unavailable
request keeps the same id for the remote service's idempotency record.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tcenv.evaluation import EvaluationError, EvaluationRequest, EvaluationResult, EvaluationUnavailable, validate_receipt, verify_result_signature
from tcenv.remote.contracts import (
    BUNDLE_DIGEST_ALGORITHM,
    LEGACY_DIALECT,
    NPU_API_CONTRACT_IDS,
    NPU_BUILD_TIMING_EVIDENCE_SCHEMA_ID,
    NPU_BUNDLE_TIMING_EVIDENCE_SCHEMA_ID,
    NPU_EVIDENCE_CONTRACT_IDS,
    NPU_TIMING_EVIDENCE_SCHEMA_ID,
    ContractError,
    candidate_digest_document,
    device_group_document,
    dialect_for,
    npu_evidence_document,
    npu_report_timing_check,
    receipt_document,
)
from tcenv.schema import SchemaError, loads_strict

from .control import write_json_atomic
from .redact import redact_text

#: CandidateDigest algorithm of an artifact set (bundle contract).
BUNDLE_ALGORITHM = BUNDLE_DIGEST_ALGORITHM

#: Directory (under the run directory) that holds one JSON row and one result envelope per
#: request id.  The public read-only helpers below derive paths from it, so a reader that has no
#: outbox (``taic.resume._read_candidates``) still agrees with the writer byte for byte.
EVALUATION_OUTBOX_DIR_NAME = "evaluation_outbox"

#: The three row states a read-only probe can report.  ``missing`` and ``pending`` are an
#: ``unknown`` context for a resume; a row nobody can read is not a state at all (see
#: :func:`row_state`, which raises).
ROW_STATES = ("missing", "pending", "completed")


def _request_key(request_id: str) -> str:
    """@brief 返回一个 ``request_id`` 在 outbox 里的文件主名（它自己的 sha256）。

    行的文件名是随机 ``request_id`` 的哈希，行里不存完整请求：靠 digest 既定位不到也重建不出请求。
    写入侧（``EvaluationOutbox``）与只读侧（公开 path helper）共用这一个函数，不允许各自实现。

    @param request_id 请求 id。
    @return 小写十六进制文件主名。
    """
    return hashlib.sha256(request_id.encode("utf-8")).hexdigest()


def _row_path(run_dir: Path | str, request_id: str) -> Path:
    """@brief 返回一个请求的 outbox 行路径（不构造 outbox）。

    @param run_dir 运行目录。
    @param request_id 请求 id。
    @return ``<run_dir>/evaluation_outbox/<sha256(request_id)>.json``。
    """
    return Path(run_dir) / EVALUATION_OUTBOX_DIR_NAME / f"{_request_key(request_id)}.json"


def _result_path(run_dir: Path | str, request_id: str) -> Path:
    """@brief 返回一个请求的 outbox result 路径（不构造 outbox）。

    @param run_dir 运行目录。
    @param request_id 请求 id。
    @return ``<run_dir>/evaluation_outbox/<sha256(request_id)>.result.json``。
    """
    return Path(run_dir) / EVALUATION_OUTBOX_DIR_NAME / f"{_request_key(request_id)}.result.json"


def row_state(run_dir: Path | str, request_id: str) -> str:
    """@brief 只读探测一个 outbox 行的状态，不构造 outbox、不校验 HMAC。

    ``missing`` 与 ``pending`` 是候选的**未知**上下文，由调用方（``taic.resume``）记成 unknown；
    行存在但读不懂（不可读、重复键、非对象、``request_id`` 不符或 status 不在闭集里）是 ````NeedsInspection``：
    被改写的行不能伪装成“没评过”。这里不做签名或 receipt 校验——续跑在 ``_read_candidates``
    阶段还没有 outbox/HMAC key，那一步属于 ``Orchestrator.restore`` 的
    ``load_completed_context``。

    @param run_dir 运行目录。
    @param request_id 请求 id。
    @return ``ROW_STATES`` 之一。
    @raises NeedsInspection 行存在但内容不可信时。
    """
    from .artifacts import NeedsInspection

    path = _row_path(run_dir, request_id)
    if not path.is_file():
        return "missing"
    try:
        document = loads_strict(path.read_text(encoding="utf-8"), what=str(path))
    except (OSError, UnicodeDecodeError) as e:
        raise NeedsInspection(f"{path} is damaged and needs human inspection") from e
    except SchemaError as e:
        raise NeedsInspection(str(e)) from e
    if not isinstance(document, dict):
        raise NeedsInspection(f"{path} is not a JSON object and needs human inspection")
    if document.get("request_id") != request_id:
        raise NeedsInspection(
            f"{path} names request {document.get('request_id')!r}, not {request_id!r}: it needs human inspection"
        )
    status = document.get("status")
    if status == "completed":
        return "completed"
    if status in {"pending", "unavailable"}:
        return "pending"
    raise NeedsInspection(f"{path} has an unknown outbox status {status!r} and needs human inspection")


#: TimingEvidence schema of the P1 bundle contract (version 2).  It is imported
#: from the contract module rather than repeated, so the outbox cannot accept an
#: evidence identity the frontend never signs.
BUNDLE_EVIDENCE_SCHEMA_ID = NPU_BUNDLE_TIMING_EVIDENCE_SCHEMA_ID
#: TimingEvidence schema of the P1c build contract (version 3).
BUILD_EVIDENCE_SCHEMA_ID = NPU_BUILD_TIMING_EVIDENCE_SCHEMA_ID


def _bind_build_provenance(request: EvaluationRequest, evaluator: Any) -> dict[str, Any] | None:
    """Return the build identity document this evaluator will submit, or ``None``.

    It is the request-level build document (``mode`` plus the independent
    BuildProvenance).  The P1c outbox stores it as its own field so a resumed row
    and the signed BuildEvidence describe one build identity.  An evaluator that
    does not expose the hook (every earlier contract) contributes ``None``; the
    build identity is never inferred from the artifact set digest.
    """

    hook = getattr(evaluator, "wire_build_provenance", None)
    if not callable(hook):
        return None
    value = hook(request)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise EvaluationUnavailable(
            "request_digest_missing", "evaluator returned an invalid BuildProvenance document"
        )
    if request.wire_build_provenance is not None and value != request.wire_build_provenance:
        raise EvaluationUnavailable(
            "request_digest_mismatch",
            "the bound BuildProvenance differs from the evaluator's build identity",
        )
    return value


def bind_wire_digest(
    request: EvaluationRequest, evaluator: Any, outbox: "EvaluationOutbox | None" = None,
    *, npu_contract: bool | None = None,
) -> EvaluationRequest:
    """Return the request keyed by the exact digest the evaluator will submit.

    This is the one host-side wiring point for the NPU contract: the durable
    outbox row, the submitted request, the receipt and a later resume all use
    the same identity.  ``npu_contract`` covers callers without a durable row
    (the selfcheck bridge); it defaults to the outbox's contract.  The legacy
    contract keeps its historical digest and this is a no-op.
    """

    if npu_contract is None:
        npu_contract = bool(outbox is not None and outbox.npu_contract)
    if not npu_contract:
        return request
    wire = getattr(evaluator, "wire_request_digest", None)
    if not callable(wire):
        raise EvaluationUnavailable(
            "request_digest_missing",
            "the NPU contract requires an evaluator that exposes wire_request_digest()",
        )
    digest = wire(request)
    if not isinstance(digest, str) or not digest:
        raise EvaluationUnavailable("request_digest_missing", "evaluator returned no wire request digest")
    candidate_wire = getattr(evaluator, "wire_candidate_digest", None)
    candidate = "" if not callable(candidate_wire) else candidate_wire(request)
    if candidate and not isinstance(candidate, str):
        raise EvaluationUnavailable("request_digest_missing", "evaluator returned an invalid wire CandidateDigest")
    algorithm_wire = getattr(evaluator, "wire_candidate_algorithm", None)
    algorithm = "" if not callable(algorithm_wire) else algorithm_wire(request)
    if algorithm and not isinstance(algorithm, str):
        raise EvaluationUnavailable(
            "request_digest_missing", "evaluator returned an invalid wire CandidateDigest algorithm"
        )
    if algorithm and not candidate:
        raise EvaluationUnavailable(
            "request_digest_missing", "a wire CandidateDigest algorithm requires its value"
        )
    if request.wire_candidate_algorithm and algorithm and algorithm != request.wire_candidate_algorithm:
        raise EvaluationUnavailable(
            "request_digest_mismatch",
            "the bound wire CandidateDigest algorithm differs from the evaluator's candidate",
        )
    bound_algorithm = request.wire_candidate_algorithm or algorithm
    dialect = getattr(evaluator, "contract", None)
    blob_contract = bool(getattr(dialect, "npu_blob", False)) or bool(
        outbox is not None and outbox.dialect.npu_blob
    )
    bundle_contract = bool(getattr(dialect, "npu_bundle", False)) or bool(
        outbox is not None and outbox.dialect.npu_bundle
    )
    if blob_contract or bundle_contract:
        # The blob contract can never fall back to `candidate_sha256` (that is the
        # empty host source).  Check an already-bound candidate *before* the first
        # binding, so a changed evaluator candidate is refused instead of silently
        # rewriting the durable identity of a resume.
        if not candidate:
            raise EvaluationUnavailable(
                "request_digest_missing",
                "the blob/bundle contract requires an evaluator CandidateDigest",
            )
        if bundle_contract and not bound_algorithm:
            raise EvaluationUnavailable(
                "request_digest_missing",
                "the bundle contract requires an evaluator CandidateDigest algorithm",
            )
        if request.wire_candidate_digest and candidate != request.wire_candidate_digest:
            raise EvaluationUnavailable(
                "request_digest_mismatch",
                "the bound wire CandidateDigest differs from the evaluator's candidate",
            )
        return replace(
            request, contract_request_digest=digest, wire_candidate_digest=candidate,
            wire_candidate_algorithm=bound_algorithm,
            wire_build_provenance=_bind_build_provenance(request, evaluator),
        )
    if request.wire_candidate_digest:
        if candidate and candidate != request.wire_candidate_digest:
            raise EvaluationUnavailable(
                "request_digest_mismatch",
                "the bound wire CandidateDigest differs from the evaluator's candidate",
            )
        return replace(
            request, contract_request_digest=digest,
            wire_candidate_digest=request.wire_candidate_digest,
            wire_build_provenance=_bind_build_provenance(request, evaluator),
        )
    if candidate and candidate != request.candidate_sha256:
        # The inline contract must bind the same bytes the request carries.
        raise EvaluationUnavailable(
            "request_digest_mismatch",
            "the evaluator's inline candidate differs from the request source",
        )
    return replace(request, contract_request_digest=digest)


#: The wire (signed) NPU receipt fields.  The host envelope adds its own
#: metadata (measurement kind, signature algorithm/key id), which is not part of
#: the wire receipt and must not be validated as if it were.
_NPU_WIRE_RECEIPT_FIELDS = (
    "receipt_id", "request_id", "attempt_id", "request_digest",
    "candidate_digest", "device_group", "lease_fence", "session_sequence", "issued_at",
)


@dataclass(frozen=True)
class OutboxIdentity:
    """@brief 一个 outbox 行持久化下来的请求身份子集。

    D43: 行里**不存完整请求**，所以续跑不能重建 `EvaluationRequest`，也不做 request 级 replay。
    这个视图只携带能从行/投影读回的字段（request id 与摘要、候选摘要、build 身份、可选的源码字节），
    足够让 ``validate_receipt``/``_wire_receipt``/``_check_npu_row``/``_validate_result`` 这些
    既有校验直接工作，而不必依赖 mode/seed/reps/execution spec 等未持久化字段。
    """

    request_id: str
    request_digest: str
    candidate_sha256: str
    wire_candidate_digest: str | None = None
    wire_candidate_algorithm: str = ""
    wire_build_provenance: dict | None = None
    source_bytes: bytes | None = None

    @classmethod
    def from_request(cls, request: EvaluationRequest, *, source_bytes: bytes | None = None) -> "OutboxIdentity":
        """@brief 从完整请求构造视图（``begin()`` 的重放分支用）。"""
        return cls(
            request_id=request.request_id,
            request_digest=request.request_digest,
            candidate_sha256=request.candidate_sha256,
            wire_candidate_digest=request.wire_candidate_digest,
            wire_candidate_algorithm=request.wire_candidate_algorithm or "",
            wire_build_provenance=request.wire_build_provenance,
            source_bytes=request.source_bytes if source_bytes is None else source_bytes,
        )


@dataclass(frozen=True)
class CompletedEvaluationContext:
    """@brief 一个 completed outbox 行加它保留的、已校验过的 result。

    ``source_hash`` 记录源码字节那一项校验的真实结果：``matched``（调用方传了 ``source_bytes`` 并已重算）、
    ``unreviewed``（inline dialect 且未传字节：该项**未**校验，不能当成已通过）、
    ``not_applicable``（blob/bundle 没有 inline 源码可重算）。
    """

    request_id: str
    request_digest: str
    row: dict[str, Any]
    result: EvaluationResult
    source_hash: str


class EvaluationOutbox:
    """Persist one JSON state and one result envelope per request id."""

    def __init__(
        self,
        run_dir: Path | str,
        *,
        signature_key: bytes | None = None,
        transport: str = "",
        api_contract_id: str = "",
        evidence_contract_id: str = "",
        external_trusted_key_ids: frozenset[str] | None = None,
    ) -> None:
        if signature_key is not None and (not isinstance(signature_key, bytes) or not signature_key):
            raise ValueError("signature_key must be non-empty bytes or None")
        for name, value in (
            ("transport", transport), ("api_contract_id", api_contract_id),
            ("evidence_contract_id", evidence_contract_id),
        ):
            if not isinstance(value, str):
                raise ValueError(f"outbox {name} must be text")
        self.signature_key = signature_key
        self.transport = transport
        self.api_contract_id = api_contract_id
        self.evidence_contract_id = evidence_contract_id
        if external_trusted_key_ids is not None and not isinstance(external_trusted_key_ids, frozenset):
            raise ValueError("external_trusted_key_ids must be a frozenset or None")
        self.external_trusted_key_ids = external_trusted_key_ids
        npu_ids = (
            self.api_contract_id in NPU_API_CONTRACT_IDS
            or self.evidence_contract_id in NPU_EVIDENCE_CONTRACT_IDS
        )
        if npu_ids:
            # The dialect (inline source vs blob reference) comes from the exact
            # contract pair, never from the shape of a request or result.
            try:
                self.dialect = dialect_for(self.api_contract_id, self.evidence_contract_id)
            except ContractError as exc:
                raise ValueError("outbox NPU contract requires both NPU API and evidence contract ids") from exc
        else:
            self.dialect = LEGACY_DIALECT
        self.npu_contract = npu_ids
        if self.npu_contract and self.signature_key is None:
            # A complete receipt id is not an authenticator: without the host
            # HMAC a rewritten attempt/fence/report/evidence would still replay.
            raise ValueError("the NPU outbox requires the host result signature key")
        self.document_version = 2 if self.npu_contract else 1
        # The status read by the last begin(); resume uses it to decide whether
        # an interrupted durable job may be resumed through an explicit Retry.
        self.previous_status = ""
        self.root = Path(run_dir) / "evaluation_outbox"
        self.root.mkdir(parents=True, exist_ok=True)
        self._receipt_index_path = self.root / "receipts.json"
        self._receipts: dict[str, str] = {}
        if self._receipt_index_path.is_file():
            try:
                import json
                document = json.loads(self._receipt_index_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"cannot read evaluation receipt index: {exc}") from exc
            if not isinstance(document, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in document.items()):
                raise ValueError("evaluation receipt index is not a string map")
            self._receipts = document

    @property
    def evidence_schema_id(self) -> str:
        """The TimingEvidence schema this contract version signs."""

        if self.dialect.build_contract:
            return BUILD_EVIDENCE_SCHEMA_ID
        return (
            BUNDLE_EVIDENCE_SCHEMA_ID if self.dialect.npu_bundle
            else NPU_TIMING_EVIDENCE_SCHEMA_ID
        )

    def _key(self, request_id: str) -> str:
        return _request_key(request_id)

    # ---- public read-only path helpers (no outbox construction, no HMAC) --------------
    @staticmethod
    def row_path(run_dir: Path | str, request_id: str) -> Path:
        """@brief 一个请求的 outbox 行路径；只读 helper，不构造 outbox。"""
        return _row_path(run_dir, request_id)

    @staticmethod
    def result_path(run_dir: Path | str, request_id: str) -> Path:
        """@brief 一个请求的 outbox result 路径；只读 helper，不构造 outbox。"""
        return _result_path(run_dir, request_id)

    @staticmethod
    def row_state(run_dir: Path | str, request_id: str) -> str:
        """@brief 只读探测行状态（``missing``/``pending``/``completed``），损坏时 ``NeedsInspection``。"""
        return row_state(run_dir, request_id)

    def _state_path(self, request_id: str) -> Path:
        return self.root / f"{self._key(request_id)}.json"

    def _result_path(self, request_id: str) -> Path:
        return self.root / f"{self._key(request_id)}.result.json"

    def _document(self, request: EvaluationRequest, *, epoch: str, status: str, created_at: float) -> dict[str, Any]:
        now = time.time()
        document = {
            "schema_version": self.document_version,
            "run_id": request.run_id,
            "session_id": request.session_id,
            "request_id": request.request_id,
            "request_digest": request.request_digest,
            "evaluator_epoch": epoch,
            "transport": self.transport,
            "api_contract_id": self.api_contract_id,
            "evidence_contract_id": self.evidence_contract_id,
            "attempt_id": "",
            "signature_key_id": "",
            "signature_algorithm": "",
            "status": status,
            "report_path": "",
            "created_at": created_at,
            "updated_at": now,
        }
        if self.npu_contract:
            document["candidate_digest"] = self._expected_candidate(request)
            if self.dialect.build_contract:
                # The build provenance is an independent row field: the wire
                # request digest binds it, and a resume must not silently swap
                # the build identity behind an otherwise equal candidate.
                if request.wire_build_provenance is None:
                    raise EvaluationUnavailable(
                        "request_digest_missing",
                        "the build outbox requires the wire build identity document "
                        "(mode plus BuildProvenance); call the evaluator's "
                        "wire_build_provenance() before begin",
                    )
                document["build_provenance"] = request.wire_build_provenance
        else:
            document["candidate_sha256"] = request.candidate_sha256
        return document

    def _expected_candidate(self, request: EvaluationRequest) -> dict[str, str]:
        """Return the wire CandidateDigest this row/result must carry.

        The blob contract has no inline bytes, so ``candidate_sha256`` (the empty
        host source) is never a fallback: without the bound wire CandidateDigest
        there is no artifact identity to record, and writing one would fabricate
        a blob nobody uploaded.
        """

        if (self.dialect.npu_blob or self.dialect.npu_bundle) and not request.wire_candidate_digest:
            raise EvaluationUnavailable(
                "request_digest_missing",
                "the artifact outbox requires the wire CandidateDigest; bind it from the "
                "uploaded descriptor or the bundle manifest before begin/complete/unavailable",
            )
        if self.dialect.npu_bundle and not request.wire_candidate_algorithm:
            # Two artifact branches share this contract, so the algorithm is the
            # only thing that separates an inline source from an artifact set.
            # Guessing it from the value would let a rewritten candidate pass.
            raise EvaluationUnavailable(
                "request_digest_missing",
                "the bundle outbox requires the wire CandidateDigest algorithm",
            )
        algorithm = request.wire_candidate_algorithm or self.dialect.candidate_algorithms[0]
        value = request.wire_candidate_digest or request.candidate_sha256
        try:
            return candidate_digest_document(
                {"algorithm": algorithm, "value": value},
                allowed_algorithms=self.dialect.candidate_algorithms,
            )
        except ContractError as exc:
            raise EvaluationUnavailable(
                "evidence_mismatch", f"outbox CandidateDigest is invalid: {exc}",
            ) from exc

    def _device_group(self, value: Any, *, where: str) -> dict[str, str]:
        """Validate the NPU device-group binding this outbox row must carry."""

        try:
            return device_group_document(value)
        except ContractError as exc:
            raise EvaluationUnavailable(
                "evidence_mismatch", f"outbox NPU {where} device group is invalid: {exc}",
            ) from exc

    def _check_binding(self, document: dict[str, Any]) -> None:
        if document.get("schema_version") != self.document_version:
            raise EvaluationUnavailable("outbox_conflict", "evaluation outbox row version differs from the contract")
        for key, expected in (
            ("transport", self.transport),
            ("api_contract_id", self.api_contract_id),
            ("evidence_contract_id", self.evidence_contract_id),
        ):
            previous = document.get(key)
            if self.npu_contract:
                # The NPU row must carry every identity field: a missing field
                # is a rewritten row, not an older one.
                if previous != expected:
                    raise EvaluationUnavailable(
                        f"{key}_mismatch",
                        f"evaluation outbox {key} differs from the current evaluator",
                    )
            elif previous and expected and previous != expected:
                raise EvaluationUnavailable(
                    f"{key}_mismatch", f"evaluation outbox {key} differs from the current evaluator"
                )

    def _check_npu_row(
        self, request: EvaluationRequest, document: Any, *, immutable_completed: bool,
    ) -> None:
        """Validate an existing NPU row before any state or result rewrite.

        Every NPU write path (`begin`, `complete`, `unavailable`) runs this on
        the row it found, so a tampered version, contract id, request id,
        request digest or CandidateDigest can never be silently rewritten, and
        a completed row cannot be replaced by a second `complete()`.
        """

        if not self.npu_contract:
            return
        if not isinstance(document, dict):
            raise EvaluationUnavailable("outbox_corrupt", "evaluation outbox row is not an object")
        self._check_binding(document)
        if document.get("request_id") != request.request_id:
            raise EvaluationUnavailable("outbox_conflict", "outbox request id is bound to a different request")
        if document.get("request_digest") != request.request_digest:
            raise EvaluationUnavailable("outbox_conflict", "outbox request id is bound to a different request digest")
        candidate = document.get("candidate_digest")
        try:
            candidate = candidate_digest_document(
                candidate, allowed_algorithms=self.dialect.candidate_algorithms,
            )
        except ContractError as exc:
            raise EvaluationUnavailable(
                "outbox_corrupt", f"outbox CandidateDigest is invalid: {exc}",
            ) from exc
        if candidate != self._expected_candidate(request):
            raise EvaluationUnavailable(
                "outbox_corrupt", "outbox CandidateDigest does not bind the request candidate",
            )
        if self.dialect.build_contract:
            stored_build = document.get("build_provenance")
            if not isinstance(stored_build, dict):
                raise EvaluationUnavailable(
                    "outbox_corrupt",
                    "outbox row has no build identity document for the build contract",
                )
            if stored_build != request.wire_build_provenance:
                raise EvaluationUnavailable(
                    "outbox_conflict",
                    "outbox build identity is bound to a different BuildProvenance",
                )
        status = document.get("status")
        if status not in {"pending", "unavailable", "completed"}:
            raise EvaluationUnavailable("outbox_corrupt", "unknown evaluation outbox status")
        if immutable_completed and status == "completed":
            raise EvaluationUnavailable(
                "outbox_conflict", "a completed evaluation outbox row is immutable",
            )

    def _wire_receipt(self, request: EvaluationRequest, result: EvaluationResult) -> dict[str, Any]:
        """Validate the signed NPU wire receipt and recompute its id.

        ``complete`` and a later replay share this check, so a retained result
        whose receipt was rewritten (including its ``receipt_id``) can never be
        replayed under a different identity.  Host-only envelope metadata is
        excluded, exactly as the adapter added it.
        """

        receipt = result.receipt if isinstance(result.receipt, dict) else {}
        document = {key: receipt.get(key) for key in _NPU_WIRE_RECEIPT_FIELDS}
        try:
            wire = receipt_document(document, dialect=self.dialect)
        except ContractError as exc:
            raise EvaluationUnavailable(
                "receipt_mismatch", f"outbox NPU receipt is invalid: {exc}",
            ) from exc
        if wire["request_id"] != request.request_id or wire["request_digest"] != request.request_digest:
            raise EvaluationUnavailable("receipt_mismatch", "outbox NPU receipt does not bind this request")
        if wire["candidate_digest"] != self._expected_candidate(request):
            raise EvaluationUnavailable("receipt_mismatch", "outbox NPU receipt CandidateDigest does not match")
        return wire

    def _validate_result(self, request: EvaluationRequest, result: EvaluationResult) -> None:
        receipt_id = validate_receipt(result, request)
        if self.npu_contract:
            receipt = result.receipt if isinstance(result.receipt, dict) else {}
            if "candidate_sha256" in receipt:
                raise EvaluationUnavailable("receipt_mismatch", "outbox NPU receipt dual-writes candidate_sha256")
            wire = self._wire_receipt(request, result)
            candidate = wire["candidate_digest"]
            receipt_group = wire["device_group"]
            evidence = result.evidence
            if not isinstance(evidence, dict) or evidence.get("schema_id") != self.evidence_schema_id:
                raise EvaluationUnavailable("evidence_mismatch", "outbox NPU result has no timing evidence")
            if evidence.get("candidate_digest") != candidate:
                raise EvaluationUnavailable("evidence_mismatch", "outbox NPU evidence CandidateDigest differs from receipt")
            if evidence.get("device_group") != receipt_group:
                raise EvaluationUnavailable("evidence_mismatch", "outbox NPU device group binding differs")
            if not isinstance(result.signature, str) or not result.signature:
                raise EvaluationUnavailable(
                    "signature_missing", "outbox NPU result requires the host envelope signature",
                )
            report = result.report if isinstance(result.report, dict) else {}
            try:
                npu_evidence_document(
                    evidence,
                    # A blob or bundle candidate has no inline bytes to rehash
                    # here; its digest is bound by the receipt/row comparison
                    # above.  The host does not have the execution resources, so
                    # the topology-versus-execution check stays in the frontend.
                    source_bytes=(
                        None if (self.dialect.npu_blob or self.dialect.npu_bundle)
                        else request.source_bytes
                    ),
                    device_group=receipt_group,
                    expected_artifact_set_digest=(
                        candidate["value"]
                        if self.dialect.npu_bundle
                        and candidate["algorithm"] == BUNDLE_ALGORITHM else None
                    ),
                    # The host outbox has no artifact manifest, so it validates
                    # the build identity (builder/toolchain/profile/source) and
                    # the evidence's own manifest digest; the frontend also
                    # compares the prebuilt entries byte for byte.
                    expected_build=(
                        request.wire_build_provenance
                        if self.dialect.build_contract else None
                    ),
                    report_status=report.get("status"),
                    dialect=self.dialect,
                )
                npu_report_timing_check(evidence, report)
            except ContractError as exc:
                raise EvaluationUnavailable(
                    "evidence_mismatch", f"outbox NPU timing evidence is invalid: {exc}",
                ) from exc
        if self.signature_key is not None:
            if not verify_result_signature(result, self.signature_key):
                raise EvaluationUnavailable("signature_mismatch", "outbox result signature does not verify")
        elif result.signature:
            raise EvaluationUnavailable("signature_unverified", "outbox result has an unverifiable signature")
        previous = self._receipts.get(receipt_id)
        if previous is not None and previous != request.request_digest:
            raise EvaluationUnavailable("receipt_replay", "receipt id was already used for a different request digest")

    def require_wire_digest(self, request: EvaluationRequest) -> None:
        """Refuse an NPU row that is not keyed by the exact wire identity.

        The NPU request digest binds CandidateDigest and the capability's device
        group, so it cannot be recomputed from the host request.  Without it the
        row, the receipt and the resume identity would silently describe a
        different job, so this fails closed instead.
        """

        if not self.npu_contract:
            return
        if not isinstance(request.contract_request_digest, str) or not request.contract_request_digest:
            raise EvaluationUnavailable(
                "request_digest_missing",
                "the NPU outbox requires the host wire request digest; "
                "call the evaluator's wire_request_digest() before begin/complete",
            )
        if request.request_digest != request.contract_request_digest:
            raise EvaluationUnavailable(
                "request_digest_mismatch", "the host request digest is not the NPU wire digest",
            )
        if self.dialect.npu_blob and not request.wire_candidate_digest:
            raise EvaluationUnavailable(
                "request_digest_missing",
                "the blob contract requires the wire CandidateDigest; "
                "bind it from the uploaded descriptor before begin/complete",
            )

    def _read_completed_result(self, document: dict[str, Any], *, row_path: Path) -> EvaluationResult:
        """@brief 读回一个 completed 行保留的 result 信封。

        The result lives under the outbox root; a relative ``report_path`` is resolved exactly as
        ``complete()`` wrote it, and a path that escapes the root is damage, not a missing result.

        @param document 已确认 completed 的行文档。
        @param row_path 行文件路径（``report_path`` 的相对基准是它的上两级目录）。
        @return ``EvaluationResult``（尚未做身份/签名校验）。
        @raises EvaluationUnavailable ``report_path`` 缺失、逃出 outbox 根或 result 不可读时。
        """
        report_path = document.get("report_path")
        if not isinstance(report_path, str) or not report_path:
            raise EvaluationUnavailable("outbox_corrupt", "completed outbox row has no report path")
        result_path = Path(report_path)
        if not result_path.is_absolute():
            result_path = row_path.parent.parent / report_path
        result_path = result_path.resolve()
        if not result_path.is_relative_to(self.root.resolve()):
            raise EvaluationUnavailable("outbox_corrupt", "completed result path escapes the outbox")
        try:
            # §6.4: every report read goes through the duplicate-key-refusing loader *before* the
            # document is handed to ``EvalReport.from_json``.  A plain ``json.loads`` here silently
            # kept the last ``schema_version``/``build`` and let a doubly-keyed report look intact;
            # a dict no longer remembers that a key was repeated.
            payload = loads_strict(result_path.read_text(encoding="utf-8"), what=str(result_path))
        except (OSError, ValueError, SchemaError) as exc:
            raise EvaluationUnavailable("outbox_corrupt", f"cannot read completed evaluation result: {exc}") from exc
        try:
            result = EvaluationResult.from_dict(payload)
        except (EvaluationError, TypeError, ValueError) as exc:
            raise EvaluationUnavailable("outbox_corrupt", f"cannot read completed evaluation result: {exc}") from exc
        # §7.2/F-36: the retained report must be an intact schema-4 document in **both** dialects.
        # ``EvaluationResult.from_dict`` keeps ``report`` as the parsed dictionary, so nothing here
        # has validated it.  The closure runs on that dictionary and leaves it untouched: the
        # envelope's own bytes are what the host HMAC covers, so this check never re-serialises (or
        # repairs) anything the signature was computed over.
        from tcenv.evaluate import EvalReport

        try:
            EvalReport.from_json(result.report)
        except (SchemaError, TypeError, ValueError) as exc:
            raise EvaluationUnavailable(
                "outbox_corrupt", f"completed result report is not a schema-4 report: {exc}",
            ) from exc
        return result

    def _validate_completed(
        self, document: dict[str, Any], result: EvaluationResult, *, view: OutboxIdentity, epoch: str = "",
    ) -> EvaluationResult:
        """@brief 用行/视图校验一个已经读回的结果；``begin()`` 重放与续跑共用。

        Only persisted fields are checked: the result's request digest and epoch, the evidence
        binding, the receipt, the host HMAC, the device-group/build identity and the receipt index.
        ``epoch`` is the **current** evaluator epoch to compare against; an empty string means
        "do not compare" — a resumed row's historical epoch *is* the context, and comparing it with
        today's epoch is an identity mismatch/unknown (F-41), not outbox damage.

        @param document 已确认 completed 的行文档。
        @param result ``_read_completed_result`` 读回的结果。
        @param view 行/请求的持久化身份视图。
        @param epoch 当前 evaluator epoch；空字符串表示不比较。
        @return 校验过的 ``EvaluationResult``。
        @raises EvaluationUnavailable result、receipt、evidence 或签名不一致时。
        """
        # The row's *historical* epoch is part of the context the resume hands to the gate, so the
        # retained result must carry the same one even when no current epoch is compared (F-41: the
        # current epoch is what a resume must not compare).  A row and a result that name different
        # epochs describe two different evaluator deployments; that is damage, not an unknown.
        row_epoch = document.get("evaluator_epoch")
        result_epoch = result.provenance.epoch
        if (
            not isinstance(row_epoch, str) or not row_epoch
            or not isinstance(result_epoch, str) or not result_epoch
        ):
            # A completed context has no verified historical epoch if either side cannot state one.
            # ``None``/absent/"" are not epochs, and "missing" must never compare equal to another
            # "missing"; ``epoch=""`` (the parameter) only means "do not compare with *today's*
            # epoch" (F-41), it never relaxes this one.
            raise EvaluationUnavailable(
                "outbox_corrupt", "completed row and retained result state no historical evaluator epoch",
            )
        if row_epoch != result_epoch:
            raise EvaluationUnavailable(
                "outbox_corrupt", "completed row and retained result name different evaluator epochs",
            )
        if epoch and result_epoch != epoch:
            raise EvaluationUnavailable("epoch_drift", "completed result evaluator epoch differs from the current evaluator")
        if result.provenance.request_digest != view.request_digest:
            raise EvaluationUnavailable("outbox_corrupt", "completed result request digest does not match")
        evidence = result.evidence
        if self.npu_contract:
            try:
                candidate = self._wire_receipt(view, result)["candidate_digest"]
            except EvaluationUnavailable as exc:
                raise EvaluationUnavailable("outbox_corrupt", str(exc)) from exc
            if (
                not isinstance(evidence, dict)
                or evidence.get("schema_id") != self.evidence_schema_id
                or evidence.get("candidate_digest") != candidate
            ):
                raise EvaluationUnavailable("outbox_corrupt", "completed NPU result evidence does not match")
        elif not isinstance(evidence, dict) or evidence.get("candidate_sha256") != view.candidate_sha256:
            raise EvaluationUnavailable("outbox_corrupt", "completed result evidence does not match")
        self._validate_result(view, result)
        if self.npu_contract:
            stored_group = self._device_group(document.get("device_group"), where="row")
            receipt_group = result.receipt.get("device_group") if isinstance(result.receipt, dict) else None
            if stored_group != self._device_group(receipt_group, where="result"):
                raise EvaluationUnavailable(
                    "outbox_corrupt",
                    "outbox NPU device group does not match the retained result",
                )
        external_key_id = result.receipt.get("signature_key_id") if isinstance(result.receipt, dict) else ""
        if (
            external_key_id and self.external_trusted_key_ids is not None
            and external_key_id not in self.external_trusted_key_ids
        ):
            raise EvaluationUnavailable(
                "signature_key_revoked",
                "outbox result was signed by an external key the host no longer trusts",
            )
        return result

    def load_completed_context(
        self, request_id: str, request_digest: str, *, source_bytes: bytes | None = None,
    ) -> CompletedEvaluationContext | None:
        """@brief 只读加载一个 completed 行的上下文；不重建请求、不重查 wire descriptor。

        ``missing``/``pending``/``unavailable`` 行返回 ``None``：那是**未知**上下文，不是损坏。
        行、result、receipt、evidence、host HMAC 或 receipt index 不一致抛出 ``NeedsInspection``
        （消息带 ``result_invalid``）——被改写的上下文不能伪装成“没评过”，也不重放或补写任何文件。

        这里**不**比较行的历史 epoch 与当前 evaluator epoch（F-41），也不做 request 级 replay：
        行不存 mode/seed/reps/execution spec，重建不出请求。``inline`` dialect 的源码摘要只在调用方
        传入 ``source_bytes`` 时重算，否则 ``source_hash="unreviewed"``，不得当成已通过。

        @param request_id 行文件名对应的请求 id。
        @param request_digest 调用方（投影）记录的请求摘要；与行不符是损坏。
        @param source_bytes 候选源码字节；给出时重算 inline dialect 的源码摘要。
        @return 校验过的上下文；缺行/pending 时为 ``None``。
        @raises NeedsInspection 行或 result 不可信时。
        """
        from .artifacts import NeedsInspection

        def damaged(detail: object) -> NeedsInspection:
            return NeedsInspection(
                f"evaluation outbox context for {request_id!r} is unusable: result_invalid ({detail})"
            )

        if not isinstance(request_id, str) or not request_id:
            raise damaged("request id is not a non-empty string")
        if not isinstance(request_digest, str) or not request_digest:
            raise damaged("request digest is not a non-empty string")
        if source_bytes is not None and not isinstance(source_bytes, (bytes, bytearray, memoryview)):
            raise damaged("source bytes are not bytes")
        path = self._state_path(request_id)
        if not path.is_file():
            return None
        try:
            document = loads_strict(path.read_text(encoding="utf-8"), what=str(path))
            if not isinstance(document, dict):
                raise EvaluationUnavailable("outbox_corrupt", "evaluation outbox row is not an object")
            if document.get("request_id") != request_id:
                raise EvaluationUnavailable("outbox_conflict", "outbox row is bound to a different request id")
            if document.get("request_digest") != request_digest:
                raise EvaluationUnavailable("outbox_conflict", "outbox row is bound to a different request digest")
            self._check_binding(document)
            status = document.get("status")
            if status in {"pending", "unavailable"}:
                return None
            if status != "completed":
                raise EvaluationUnavailable("outbox_corrupt", "unknown evaluation outbox status")
            result = self._read_completed_result(document, row_path=path)
            view = self._row_identity(
                document, result, request_id=request_id, request_digest=request_digest, source_bytes=source_bytes,
            )
            if self.npu_contract:
                self._check_npu_row(view, document, immutable_completed=False)
            self._validate_completed(document, result, view=view)
            return CompletedEvaluationContext(
                request_id=request_id, request_digest=request_digest, row=document, result=result,
                source_hash=self._source_hash_state(view),
            )
        except NeedsInspection:
            raise
        except (EvaluationError, ContractError, SchemaError, OSError, ValueError) as exc:
            raise damaged(f"{type(exc).__name__}: {exc}") from exc

    def projection_document(self, context: CompletedEvaluationContext) -> dict:
        """@brief 把一份已校验的 completed 上下文投影成固定 schema v1 的 ``evaluation_context.json``。

        The projection is a *projection of persisted facts*, not a re-derivation: every identity
        field except the baseline environment digest is read back from the row the outbox wrote,
        and the environment digest is the accepted report's own top-level field.  That is what lets
        a later resume reconcile the file against the row/result byte for byte (F-36).

        Nothing is invented: a row that is not completed, a context belonging to another row, or an
        empty evaluator epoch is refused instead of being written with a placeholder.  ``None``
        fields stay ``None`` (the generic dialect has no device group or request-level build
        document; a report with a null environment digest keeps it null).

        @param context ``load_completed_context`` 或 ``complete`` 后的已校验上下文。
        @return 固定 8 键的投影文档。
        @raises EvaluationUnavailable 行未完成、上下文不属于该行、或缺 epoch 时。
        """
        from .artifacts import evaluation_context_document

        document = context.row
        if not isinstance(document, dict) or document.get("status") != "completed":
            raise EvaluationUnavailable(
                "outbox_corrupt", "only a completed evaluation row has an evaluation context projection",
            )
        if document.get("request_id") != context.request_id or document.get("request_digest") != context.request_digest:
            raise EvaluationUnavailable(
                "outbox_conflict", "the evaluation context does not belong to this completed row",
            )
        epoch = document.get("evaluator_epoch")
        if not isinstance(epoch, str) or not epoch:
            raise EvaluationUnavailable(
                "epoch_drift", "the completed evaluation row records no evaluator epoch",
            )
        candidate = document.get("candidate_digest") if self.npu_contract else document.get("candidate_sha256")
        if candidate is None:
            raise EvaluationUnavailable(
                "outbox_corrupt", "the completed evaluation row records no candidate identity",
            )
        report = context.result.report
        if hasattr(report, "to_json"):
            report = report.to_json()
        if not isinstance(report, dict):
            report = {}
        environment = report.get("baseline_environment_digest")
        return evaluation_context_document(
            request_id=context.request_id,
            request_digest=context.request_digest,
            candidate=candidate,
            evaluator_epoch=epoch,
            device_group=document.get("device_group") if self.npu_contract else None,
            build_provenance=document.get("build_provenance") if self.dialect.build_contract else None,
            baseline_environment_digest=environment if isinstance(environment, str) and environment else None,
        )

    def _source_hash_state(self, view: OutboxIdentity) -> str:
        """@brief 源码字节校验的真实状态；不把未做的检查标成通过。"""
        if self.npu_contract and (self.dialect.npu_blob or self.dialect.npu_bundle):
            return "not_applicable"
        return "matched" if view.source_bytes is not None else "unreviewed"

    def _row_identity(
        self, document: dict[str, Any], result: EvaluationResult, *,
        request_id: str, request_digest: str, source_bytes: bytes | None,
    ) -> OutboxIdentity:
        """@brief 只从持久化行（必要时加调用方源码字节）构造校验视图。"""
        if self.npu_contract:
            candidate = candidate_digest_document(
                document.get("candidate_digest"), allowed_algorithms=self.dialect.candidate_algorithms,
            )
            inline = not (self.dialect.npu_blob or self.dialect.npu_bundle)
            return OutboxIdentity(
                request_id=request_id,
                request_digest=request_digest,
                candidate_sha256=candidate["value"],
                wire_candidate_digest=candidate["value"],
                wire_candidate_algorithm=candidate["algorithm"],
                wire_build_provenance=(
                    document.get("build_provenance") if self.dialect.build_contract else None
                ),
                # A blob/bundle candidate has no inline bytes, so there is nothing to rehash.
                source_bytes=source_bytes if inline else None,
            )
        # The legacy (Metal/source) row persists ``candidate_sha256``; it must agree with the bytes
        # the caller measured (when given) and with the receipt/evidence the result carries.  The
        # row used to be ignored here, so a rewritten row identity could hand the gate a context
        # whose candidate was never the measured one.
        row_candidate = document.get("candidate_sha256")
        if not isinstance(row_candidate, str) or not row_candidate:
            raise EvaluationUnavailable("outbox_corrupt", "completed outbox row records no candidate sha256")
        if source_bytes is not None:
            candidate_sha256 = hashlib.sha256(bytes(source_bytes)).hexdigest()
            if candidate_sha256 != row_candidate:
                raise EvaluationUnavailable(
                    "outbox_corrupt", "the source bytes are not the candidate the completed row records",
                )
        else:
            candidate_sha256 = row_candidate
        return OutboxIdentity(
            request_id=request_id, request_digest=request_digest,
            candidate_sha256=candidate_sha256, source_bytes=source_bytes,
        )

    def begin(self, request: EvaluationRequest, *, epoch: str = "") -> EvaluationResult | None:
        """Return a completed result or persist a pending request before sending."""
        self.require_wire_digest(request)
        self.previous_status = ""
        path = self._state_path(request.request_id)
        if path.is_file():
            try:
                import json

                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise EvaluationUnavailable("outbox_corrupt", f"cannot read evaluation outbox: {exc}") from exc
            if not isinstance(document, dict):
                raise EvaluationUnavailable("outbox_conflict", "outbox request id is bound to a different request")
            if self.npu_contract:
                self._check_npu_row(request, document, immutable_completed=False)
            else:
                if document.get("request_digest") != request.request_digest:
                    raise EvaluationUnavailable("outbox_conflict", "outbox request id is bound to a different request")
                self._check_binding(document)
            self.previous_status = str(document.get("status") or "")
            previous_epoch = document.get("evaluator_epoch")
            if epoch and previous_epoch and previous_epoch != epoch:
                raise EvaluationUnavailable("epoch_drift", "evaluator epoch differs from the outbox request")
            if document.get("status") == "completed":
                result = self._read_completed_result(document, row_path=path)
                return self._validate_completed(
                    document, result, view=OutboxIdentity.from_request(request), epoch=epoch,
                )
            if document.get("status") not in {"pending", "unavailable"}:
                raise EvaluationUnavailable("outbox_corrupt", "unknown evaluation outbox status")
            document["status"] = "pending"
            document["updated_at"] = time.time()
            document.pop("error", None)
            write_json_atomic(path, document)
            return None
        write_json_atomic(path, self._document(request, epoch=epoch, status="pending", created_at=time.time()))
        return None

    def complete(self, request: EvaluationRequest, result: EvaluationResult) -> None:
        """Atomically retain the result envelope and mark the request completed."""
        self.require_wire_digest(request)
        self._validate_result(request, result)
        path = self._state_path(request.request_id)
        result_path = self._result_path(request.request_id)
        # Validate the durable row *before* any result or state byte is written:
        # a tampered or already completed row must not gain a result file.
        try:
            import json

            if path.is_file():
                document = json.loads(path.read_text(encoding="utf-8"))
                self._check_npu_row(request, document, immutable_completed=True)
            else:
                document = self._document(
                    request, epoch=result.provenance.epoch, status="pending",
                    created_at=time.time(),
                )
        except (OSError, ValueError) as exc:
            raise EvaluationUnavailable("outbox_corrupt", f"cannot update evaluation outbox: {exc}") from exc
        write_json_atomic(result_path, result.to_dict())
        receipt = result.receipt if isinstance(result.receipt, dict) else {}
        document.update({
            "evaluator_epoch": result.provenance.epoch,
            "status": "completed",
            "report_path": str(result_path.relative_to(path.parent.parent)),
            "attempt_id": str(receipt.get("attempt_id") or ""),
            "signature_key_id": str(receipt.get("signature_key_id") or ""),
            "signature_algorithm": str(receipt.get("signature_algorithm") or ""),
            "updated_at": time.time(),
        })
        if self.npu_contract:
            document["candidate_digest"] = receipt.get("candidate_digest")
            document["device_group"] = receipt.get("device_group")
        document.pop("error", None)
        write_json_atomic(path, document)
        self._receipts[validate_receipt(result, request)] = request.request_digest
        write_json_atomic(self._receipt_index_path, self._receipts)

    def unavailable(self, request: EvaluationRequest, error: BaseException) -> None:
        """Record an infrastructure/policy failure without inventing a report.

        The same identity gate as ``begin``/``complete`` runs before any row is
        created or rewritten.  A blob request without the wire CandidateDigest has
        no artifact identity, so it may not seed a row with the empty-source
        digest; with no durable row there is nothing to reconcile and nothing is
        written (the caller already reports the original failure).
        """

        if self.npu_contract:
            detail = ""
            if not request.contract_request_digest:
                detail = "the NPU outbox row is not keyed by the wire request digest"
            elif (self.dialect.npu_blob or self.dialect.npu_bundle) and not request.wire_candidate_digest:
                detail = "the blob outbox row is not keyed by the wire CandidateDigest"
            if detail:
                if not self._state_path(request.request_id).is_file():
                    # A missing wire identity means nothing was durably accepted,
                    # so there is no row to reconcile; never overwrite the failure
                    # the caller is already reporting with a second one.
                    return
                raise EvaluationUnavailable("request_digest_missing", detail)
        path = self._state_path(request.request_id)
        try:
            import json

            if path.is_file():
                document = json.loads(path.read_text(encoding="utf-8"))
                self._check_npu_row(request, document, immutable_completed=False)
            else:
                document = self._document(request, epoch="", status="pending", created_at=time.time())
        except (OSError, ValueError) as exc:
            if self.npu_contract:
                # A corrupt NPU row must never be silently re-initialized: its
                # identity is unknowable, so the caller sees the corruption.
                raise EvaluationUnavailable(
                    "outbox_corrupt", f"cannot read evaluation outbox: {exc}",
                ) from exc
            document = self._document(request, epoch="", status="pending", created_at=time.time())
        code = getattr(error, "code", "evaluation_unavailable")
        retryable = bool(getattr(error, "retryable", False))
        error_document = {"code": str(code), "message": redact_text(str(error))[:1000], "retryable": retryable}
        if document.get("status") == "completed":
            # A completed envelope is immutable.  A revoked external key makes
            # the replay fail closed, but the retained result is not replaced
            # by a pending row.
            document["error"] = error_document
            document["updated_at"] = time.time()
            write_json_atomic(path, document)
            return
        document.update({
            "status": "unavailable",
            "report_path": "",
            "updated_at": time.time(),
            "error": error_document,
        })
        write_json_atomic(path, document)

    def pending(self) -> list[dict[str, Any]]:
        """Return pending/unavailable rows for brief and resume diagnostics."""
        import json

        rows = []
        for path in sorted(self.root.glob("*.json")):
            if path.name.endswith(".result.json") or path.name == "receipts.json":
                continue
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(row, dict) and row.get("status") in {"pending", "unavailable"}:
                rows.append(row)
        return rows
