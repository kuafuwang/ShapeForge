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

from .control import write_json_atomic
from .redact import redact_text

#: CandidateDigest algorithm of an artifact set (bundle contract).
BUNDLE_ALGORITHM = BUNDLE_DIGEST_ALGORITHM
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
                    try:
                        candidate = self._wire_receipt(request, result)["candidate_digest"]
                    except EvaluationUnavailable as exc:
                        raise EvaluationUnavailable("outbox_corrupt", str(exc)) from exc
                    if (
                        not isinstance(evidence, dict)
                        or evidence.get("schema_id") != self.evidence_schema_id
                        or evidence.get("candidate_digest") != candidate
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
