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
from dataclasses import replace
from pathlib import Path
from typing import Any

from tcenv.evaluation import EvaluationRequest, EvaluationResult, EvaluationUnavailable, validate_receipt, verify_result_signature
from tcenv.remote.contracts import (
    NPU_SOURCE_API_CONTRACT_ID,
    NPU_SOURCE_EVIDENCE_CONTRACT_ID,
    NPU_TIMING_EVIDENCE_SCHEMA_ID,
    ContractError,
    device_group_document,
    npu_timing_evidence_document,
)

from .control import write_json_atomic
from .redact import redact_text


def bind_wire_digest(
    request: EvaluationRequest, evaluator: Any, outbox: "EvaluationOutbox | None",
) -> EvaluationRequest:
    """Return the request keyed by the exact digest the evaluator will submit.

    This is the one host-side wiring point for the NPU contract: the durable
    outbox row, the submitted request, the receipt and a later resume all use
    the same identity.  The legacy contract keeps its historical digest and
    this is a no-op.
    """

    if outbox is None or not outbox.npu_contract:
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
    return replace(request, contract_request_digest=digest)


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
        self.npu_contract = (
            self.api_contract_id == NPU_SOURCE_API_CONTRACT_ID
            or self.evidence_contract_id == NPU_SOURCE_EVIDENCE_CONTRACT_ID
        )
        if self.npu_contract and (
            self.api_contract_id != NPU_SOURCE_API_CONTRACT_ID
            or self.evidence_contract_id != NPU_SOURCE_EVIDENCE_CONTRACT_ID
        ):
            raise ValueError("outbox NPU contract requires both NPU API and evidence contract ids")
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

    def _key(self, request_id: str) -> str:
        return hashlib.sha256(request_id.encode("utf-8")).hexdigest()

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
            document["candidate_digest"] = {"algorithm": "sha256", "value": request.candidate_sha256}
        else:
            document["candidate_sha256"] = request.candidate_sha256
        return document

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
            if previous and expected and previous != expected:
                raise EvaluationUnavailable(
                    f"{key}_mismatch", f"evaluation outbox {key} differs from the current evaluator"
                )

    def _validate_result(self, request: EvaluationRequest, result: EvaluationResult) -> None:
        receipt_id = validate_receipt(result, request)
        if self.npu_contract:
            receipt = result.receipt if isinstance(result.receipt, dict) else {}
            candidate = receipt.get("candidate_digest")
            if not isinstance(candidate, dict) or candidate.get("value") != request.candidate_sha256:
                raise EvaluationUnavailable("receipt_mismatch", "outbox NPU receipt CandidateDigest does not match")
            if "candidate_sha256" in receipt:
                raise EvaluationUnavailable("receipt_mismatch", "outbox NPU receipt dual-writes candidate_sha256")
            evidence = result.evidence
            if not isinstance(evidence, dict) or evidence.get("schema_id") != NPU_TIMING_EVIDENCE_SCHEMA_ID:
                raise EvaluationUnavailable("evidence_mismatch", "outbox NPU result has no timing evidence")
            if evidence.get("candidate_digest") != candidate:
                raise EvaluationUnavailable("evidence_mismatch", "outbox NPU evidence CandidateDigest differs from receipt")
            receipt_group = receipt.get("device_group")
            if not isinstance(receipt_group, dict) or evidence.get("device_group") != receipt_group:
                raise EvaluationUnavailable("evidence_mismatch", "outbox NPU device group binding differs")
            self._device_group(receipt_group, where="receipt")
            report = result.report if isinstance(result.report, dict) else {}
            try:
                npu_timing_evidence_document(
                    evidence,
                    source_bytes=request.source_bytes,
                    device_group=receipt_group,
                    report_status=report.get("status"),
                )
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
            if not isinstance(document, dict) or document.get("request_digest") != request.request_digest:
                raise EvaluationUnavailable("outbox_conflict", "outbox request id is bound to a different request")
            self._check_binding(document)
            if self.npu_contract:
                candidate = document.get("candidate_digest")
                if (
                    not isinstance(candidate, dict)
                    or candidate.get("algorithm") != "sha256"
                    or candidate.get("value") != request.candidate_sha256
                ):
                    raise EvaluationUnavailable(
                        "outbox_corrupt", "outbox CandidateDigest does not bind the request source",
                    )
            self.previous_status = str(document.get("status") or "")
            previous_epoch = document.get("evaluator_epoch")
            if epoch and previous_epoch and previous_epoch != epoch:
                raise EvaluationUnavailable("epoch_drift", "evaluator epoch differs from the outbox request")
            if document.get("status") == "completed":
                report_path = document.get("report_path")
                if not isinstance(report_path, str) or not report_path:
                    raise EvaluationUnavailable("outbox_corrupt", "completed outbox row has no report path")
                result_path = Path(report_path)
                if not result_path.is_absolute():
                    result_path = path.parent.parent / result_path
                result_path = result_path.resolve()
                if not result_path.is_relative_to(self.root.resolve()):
                    raise EvaluationUnavailable("outbox_corrupt", "completed result path escapes the outbox")
                try:
                    import json

                    result = EvaluationResult.from_dict(json.loads(result_path.read_text(encoding="utf-8")))
                except (OSError, ValueError) as exc:
                    raise EvaluationUnavailable("outbox_corrupt", f"cannot read completed evaluation result: {exc}") from exc
                if epoch and result.provenance.epoch != epoch:
                    raise EvaluationUnavailable("epoch_drift", "completed result evaluator epoch differs from the current evaluator")
                if result.provenance.request_digest != request.request_digest:
                    raise EvaluationUnavailable("outbox_corrupt", "completed result request digest does not match")
                evidence = result.evidence
                if self.npu_contract:
                    candidate = result.receipt.get("candidate_digest") if isinstance(result.receipt, dict) else None
                    if (
                        not isinstance(evidence, dict)
                        or evidence.get("schema_id") != NPU_TIMING_EVIDENCE_SCHEMA_ID
                        or not isinstance(candidate, dict)
                        or evidence.get("candidate_digest") != candidate
                        or candidate.get("value") != request.candidate_sha256
                    ):
                        raise EvaluationUnavailable("outbox_corrupt", "completed NPU result evidence does not match")
                elif not isinstance(evidence, dict) or evidence.get("candidate_sha256") != request.candidate_sha256:
                    raise EvaluationUnavailable("outbox_corrupt", "completed result evidence does not match")
                self._validate_result(request, result)
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
        write_json_atomic(result_path, result.to_dict())
        try:
            import json

            document = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else self._document(request, epoch=result.provenance.epoch, status="pending", created_at=time.time())
        except (OSError, ValueError) as exc:
            raise EvaluationUnavailable("outbox_corrupt", f"cannot update evaluation outbox: {exc}") from exc
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
        """Record an infrastructure/policy failure without inventing a report."""
        if self.npu_contract and not request.contract_request_digest:
            if not self._state_path(request.request_id).is_file():
                # A missing wire digest means nothing was durably accepted, so
                # there is no row to reconcile; never overwrite the failure the
                # caller is already reporting with a second one.
                return
            raise EvaluationUnavailable(
                "request_digest_missing", "the NPU outbox row is not keyed by the wire request digest",
            )
        path = self._state_path(request.request_id)
        try:
            import json

            document = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else self._document(request, epoch="", status="pending", created_at=time.time())
        except (OSError, ValueError):
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
