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
from pathlib import Path
from typing import Any

from tcenv.evaluation import EvaluationRequest, EvaluationResult, EvaluationUnavailable, validate_receipt, verify_result_signature

from .control import write_json_atomic
from .redact import redact_text


class EvaluationOutbox:
    """Persist one JSON state and one result envelope per request id."""

    def __init__(self, run_dir: Path | str, *, signature_key: bytes | None = None) -> None:
        if signature_key is not None and (not isinstance(signature_key, bytes) or not signature_key):
            raise ValueError("signature_key must be non-empty bytes or None")
        self.signature_key = signature_key
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

    @staticmethod
    def _document(request: EvaluationRequest, *, epoch: str, status: str, created_at: float) -> dict[str, Any]:
        now = time.time()
        return {
            "schema_version": 1,
            "run_id": request.run_id,
            "session_id": request.session_id,
            "request_id": request.request_id,
            "request_digest": request.request_digest,
            "candidate_sha256": request.candidate_sha256,
            "evaluator_epoch": epoch,
            "status": status,
            "report_path": "",
            "created_at": created_at,
            "updated_at": now,
        }

    def _validate_result(self, request: EvaluationRequest, result: EvaluationResult) -> None:
        receipt_id = validate_receipt(result, request)
        if self.signature_key is not None:
            if not verify_result_signature(result, self.signature_key):
                raise EvaluationUnavailable("signature_mismatch", "outbox result signature does not verify")
        elif result.signature:
            raise EvaluationUnavailable("signature_unverified", "outbox result has an unverifiable signature")
        previous = self._receipts.get(receipt_id)
        if previous is not None and previous != request.request_digest:
            raise EvaluationUnavailable("receipt_replay", "receipt id was already used for a different request digest")

    def begin(self, request: EvaluationRequest, *, epoch: str = "") -> EvaluationResult | None:
        """Return a completed result or persist a pending request before sending."""
        path = self._state_path(request.request_id)
        if path.is_file():
            try:
                import json

                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise EvaluationUnavailable("outbox_corrupt", f"cannot read evaluation outbox: {exc}") from exc
            if not isinstance(document, dict) or document.get("request_digest") != request.request_digest:
                raise EvaluationUnavailable("outbox_conflict", "outbox request id is bound to a different request")
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
                if not isinstance(evidence, dict) or evidence.get("candidate_sha256") != request.candidate_sha256:
                    raise EvaluationUnavailable("outbox_corrupt", "completed result evidence does not match")
                self._validate_result(request, result)
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
        self._validate_result(request, result)
        path = self._state_path(request.request_id)
        result_path = self._result_path(request.request_id)
        write_json_atomic(result_path, result.to_dict())
        try:
            import json

            document = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else self._document(request, epoch=result.provenance.epoch, status="pending", created_at=time.time())
        except (OSError, ValueError) as exc:
            raise EvaluationUnavailable("outbox_corrupt", f"cannot update evaluation outbox: {exc}") from exc
        document.update({
            "evaluator_epoch": result.provenance.epoch,
            "status": "completed",
            "report_path": str(result_path.relative_to(path.parent.parent)),
            "updated_at": time.time(),
        })
        document.pop("error", None)
        write_json_atomic(path, document)
        self._receipts[validate_receipt(result, request)] = request.request_digest
        write_json_atomic(self._receipt_index_path, self._receipts)

    def unavailable(self, request: EvaluationRequest, error: BaseException) -> None:
        """Record an infrastructure/policy failure without inventing a report."""
        path = self._state_path(request.request_id)
        try:
            import json

            document = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else self._document(request, epoch="", status="pending", created_at=time.time())
        except (OSError, ValueError):
            document = self._document(request, epoch="", status="pending", created_at=time.time())
        code = getattr(error, "code", "evaluation_unavailable")
        retryable = bool(getattr(error, "retryable", False))
        document.update({
            "status": "unavailable",
            "report_path": "",
            "updated_at": time.time(),
            "error": {"code": str(code), "message": redact_text(str(error))[:1000], "retryable": retryable},
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
