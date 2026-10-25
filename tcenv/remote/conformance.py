"""Wire-level conformance flow for the external evaluator API.

The flow runs against any TLS/mTLS frontend reachable with the supplied host
factory: authenticated GetCapabilities, a full Submit/Watch/GetJob result, an
explicit Retry for an interrupted attempt (or the rejection of Retry for a
completed job), Cancel of a second job, and typed negative checks.  It returns a
JSON-serializable report without secrets.
"""

from __future__ import annotations

import secrets
import time
from typing import Any

import grpc
from grpc_status import rpc_status

from . import contracts, grpc_contract as gc
from .grpc_client import GrpcEvaluatorFactory
from .job_types import ExternalEvaluationRequest
from .v1 import evaluator_pb2 as pb

DEFAULT_REQUIRED_FEATURES = ("durable-jobs", "attempt-receipts", "progress")


def error_detail(exc: grpc.RpcError) -> pb.EvaluationErrorDetail | None:
    try:
        status = rpc_status.from_call(exc)
    except (ValueError, TypeError, AttributeError):
        return None
    if status is None:
        return None
    for item in status.details:
        if item.Is(pb.EvaluationErrorDetail.DESCRIPTOR):
            detail = pb.EvaluationErrorDetail()
            item.Unpack(detail)
            return detail
    return None


def _metadata(token: str) -> tuple[tuple[str, str], ...]:
    return (("authorization", f"Bearer {token}"),)


def _capabilities(stub, token: str, *, expected, features=(), timeout: float = 15.0):
    return stub.GetCapabilities(
        pb.CapabilitiesRequest(
            api_contract_ids=[contracts.API_CONTRACT_ID],
            evidence_contract_ids=[contracts.EVIDENCE_CONTRACT_ID],
            required_features=list(features),
            client_nonce=secrets.token_hex(16),
            expected=pb.ExpectedEvaluator(
                site_id=expected.site_id,
                device_id=expected.device_id,
                target_id=expected.target_id,
                worker_build=expected.worker_build,
                task_pack_digest=expected.task_pack_digest,
                evaluator_epoch=expected.evaluator_epoch,
            ),
        ),
        metadata=_metadata(token),
        timeout=timeout,
    )


def _step(report: dict[str, Any], name: str, *, ok: bool, detail: str = "", **extra: Any) -> None:
    entry = {"name": name, "ok": bool(ok)}
    if detail:
        entry["detail"] = detail[:1000]
    entry.update(extra)
    report["steps"].append(entry)
    report["ok"] = report["ok"] and bool(ok)


def run_conformance(
    *,
    factory: GrpcEvaluatorFactory,
    request,
    second_request=None,
    token: str = "",
    required_features: tuple[str, ...] = DEFAULT_REQUIRED_FEATURES,
    timeout_s: float = 30.0,
    poll_s: float = 0.05,
) -> dict[str, Any]:
    """Run the conformance flow and return a report; never raises for a failed step."""

    report: dict[str, Any] = {
        "schema_version": 1,
        "ok": True,
        "evaluator_endpoint": factory.endpoint,
        "steps": [],
    }
    token = token or factory.token_provider(factory.run_id, factory.session_id or "conformance")
    stub = factory.stub
    external = ExternalEvaluationRequest.from_legacy(request)

    # 1. Authentication is required on every RPC, including capabilities.
    try:
        stub.GetCapabilities(pb.CapabilitiesRequest(client_nonce="a" * 16), timeout=timeout_s)
        _step(report, "unauthenticated_capabilities_rejected", ok=False, detail="call succeeded")
    except grpc.RpcError as exc:
        _step(report, "unauthenticated_capabilities_rejected", ok=exc.code() == grpc.StatusCode.UNAUTHENTICATED, detail=exc.details() or "")

    # 2. A required feature the frontend does not support fails closed before Submit.
    unsupported = "conformance-unsupported-feature"
    try:
        _capabilities(stub, token, expected=factory.expected, features=(*required_features, unsupported), timeout=timeout_s)
        _step(report, "unsupported_required_feature_rejected", ok=False, detail="capabilities succeeded")
    except grpc.RpcError as exc:
        detail = error_detail(exc)
        _step(
            report, "unsupported_required_feature_rejected",
            ok=exc.code() == grpc.StatusCode.FAILED_PRECONDITION and detail is not None and unsupported in detail.missing_features,
            detail=exc.details() or "",
            missing_features=list(detail.missing_features) if detail is not None else [],
        )

    # 3. GetCapabilities and the three identities.
    epoch = factory.ensure_epoch(request)
    capabilities = _capabilities(stub, token, expected=factory.expected, features=required_features, timeout=timeout_s)
    provenance = gc.provenance_from_proto(capabilities.provenance, require_attempt=False)
    _step(
        report, "capabilities",
        ok=bool(capabilities.capability_id) and capabilities.evaluator_epoch == epoch
        and capabilities.api_contract_id == contracts.API_CONTRACT_ID
        and capabilities.evidence_contract_id == contracts.EVIDENCE_CONTRACT_ID,
        evaluator_epoch=epoch,
        api_contract_id=capabilities.api_contract_id,
        evidence_contract_id=capabilities.evidence_contract_id,
        site_id=provenance["site_id"],
        target_id=provenance["target_id"],
        measurement_kind=provenance["measurement_kind"],
        publication_allowed=bool(provenance["publication_allowed"]),
        supported_features=sorted(capabilities.supported_features),
    )
    expected = gc.expected_identities_from_provenance(provenance)

    # 4. Submit + Watch/GetJob.  An interrupted attempt is resumed through the
    # explicit Retry RPC, never by an implicit second Submit.
    result = None
    try:
        result = factory.formal(request)
        _step(report, "submit_watch_result", ok=True, attempt_id=result.receipt.get("attempt_id", ""))
    except Exception as exc:  # noqa: BLE001 - a conformance step, not an exception policy
        code = getattr(exc, "code", "")
        retryable = bool(getattr(exc, "retryable", False))
        if retryable and code in {"interrupted", "frontend_stopped", "worker_failed", "worker_protocol_error"}:
            resume = getattr(factory, "formal_resume", None)
            if not callable(resume):
                _step(report, "submit_watch_result", ok=False, detail=f"{code}: no explicit retry adapter")
            else:
                try:
                    result = resume(request)
                    _step(report, "submit_watch_result_after_retry", ok=True, attempt_id=result.receipt.get("attempt_id", ""))
                except Exception as retry_exc:  # noqa: BLE001
                    _step(report, "submit_watch_result_after_retry", ok=False, detail=str(retry_exc))
        else:
            _step(report, "submit_watch_result", ok=False, detail=f"{code}: {exc}")

    # 5. GetJob must agree with the watched terminal snapshot.
    try:
        snapshot = stub.GetJob(
            pb.JobRef(
                request_id=external.request_id,
                request_digest=external.request_digest,
                expected=gc.expected_identities_to_proto(expected),
            ),
            metadata=_metadata(token), timeout=timeout_s,
        )
        state = gc.state_from_proto(snapshot.state)
        agrees = result is not None and state == "completed" and snapshot.HasField("result")
        if agrees:
            receipt = gc.receipt_from_proto(snapshot.result.receipt)
            agrees = receipt["attempt_id"] == result.receipt.get("attempt_id")
        _step(report, "getjob_terminal_snapshot", ok=agrees, state=state, last_sequence=int(snapshot.last_sequence))
    except grpc.RpcError as exc:
        _step(report, "getjob_terminal_snapshot", ok=False, detail=exc.details() or "")

    # 6. Retry is only for interrupted attempts.  A completed job must be refused.
    try:
        fresh = _capabilities(stub, token, expected=factory.expected, features=required_features, timeout=timeout_s)
        fresh_expected = gc.expected_identities_from_provenance(
            gc.provenance_from_proto(fresh.provenance, require_attempt=False)
        )
        completed_attempt = result.receipt.get("attempt_id", "") if result is not None else "attempt-missing"
        stub.Retry(
            pb.RetryRequest(
                job=pb.JobRef(
                    request_id=external.request_id,
                    request_digest=external.request_digest,
                    expected=gc.expected_identities_to_proto(fresh_expected),
                ),
                interrupted_attempt_id=completed_attempt,
                capability_id=fresh.capability_id,
                required_features=list(required_features),
            ),
            metadata=_metadata(token), timeout=timeout_s,
        )
        _step(report, "retry_completed_rejected", ok=False, detail="Retry accepted a completed job")
    except grpc.RpcError as exc:
        detail = error_detail(exc)
        _step(
            report, "retry_completed_rejected",
            ok=exc.code() == grpc.StatusCode.FAILED_PRECONDITION,
            detail=exc.details() or "",
            reason=detail.reason if detail is not None else "",
        )

    # 7. Cancel a second job and confirm the durable terminal state.
    if second_request is not None:
        second_external = ExternalEvaluationRequest.from_legacy(second_request)
        try:
            fresh = _capabilities(stub, token, expected=factory.expected, features=required_features, timeout=timeout_s)
            fresh_expected = gc.expected_identities_from_provenance(
                gc.provenance_from_proto(fresh.provenance, require_attempt=False)
            )
            submitted = stub.Submit(
                pb.SubmitRequest(
                    request=gc.request_to_proto(second_external),
                    request_digest=second_external.request_digest,
                    expected=gc.expected_identities_to_proto(fresh_expected),
                    capability_id=fresh.capability_id,
                    required_features=list(required_features),
                ),
                metadata=_metadata(token), timeout=timeout_s,
            )
            cancelled = stub.Cancel(
                pb.CancelRequest(job=pb.JobRef(
                    request_id=second_external.request_id,
                    request_digest=second_external.request_digest,
                    expected=gc.expected_identities_to_proto(fresh_expected),
                )),
                metadata=_metadata(token), timeout=timeout_s,
            )
            deadline = time.monotonic() + max(5.0, timeout_s)
            final = cancelled
            while time.monotonic() < deadline:
                state = gc.state_from_proto(final.state)
                if state in {"cancelled", "completed", "failed", "expired"}:
                    break
                time.sleep(poll_s)
                final = stub.GetJob(
                    pb.JobRef(
                        request_id=second_external.request_id,
                        request_digest=second_external.request_digest,
                        expected=gc.expected_identities_to_proto(fresh_expected),
                    ),
                    metadata=_metadata(token), timeout=timeout_s,
                )
            state = gc.state_from_proto(final.state)
            _step(report, "cancel_second_job", ok=state in {"cancelled", "completed"}, state=state)
        except grpc.RpcError as exc:
            _step(report, "cancel_second_job", ok=False, detail=exc.details() or "")

    return report


def conformance_request(
    *, request_id: str, run_id: str, session_id: str, task: str, dtype: str, size: str,
    target_id: str, task_fingerprint: str, source: bytes, mode: str = "quick",
    kind: str = "formal_eval", seed: str | None = "3", reps: int = 2,
    queue_timeout_ms: int = 30_000, candidate_timeout_ms: int = 120_000,
):
    """Build the legacy-shaped request used by the external conformance CLI."""

    from tcenv.evaluation import EvaluationRequest

    return EvaluationRequest(
        task=task, dtype=dtype, size=size, source=source.decode("utf-8"),
        kind=kind, mode=mode, seed=None if seed is None else int(seed), reps=reps,
        run_id=run_id, session_id=session_id, request_id=request_id, target_id=target_id,
        task_fingerprint=task_fingerprint,
        queue_timeout_s=queue_timeout_ms / 1000.0,
        candidate_timeout_s=candidate_timeout_ms / 1000.0,
    )


def result_summary(result) -> dict[str, Any]:
    receipt = result.receipt if isinstance(result.receipt, dict) else {}
    return {
        "status": result.report.get("status") if isinstance(result.report, dict) else str(getattr(result.report, "status", "")),
        "attempt_id": receipt.get("attempt_id", ""),
        "receipt_id": receipt.get("receipt_id", ""),
        "lease_fence": receipt.get("lease_fence", ""),
        "signature_key_id": receipt.get("signature_key_id", ""),
        "evaluator_epoch": result.provenance.epoch,
        "site_id": result.provenance.site_id,
        "measurement_kind": result.provenance.measurement_kind,
        "publication_allowed": bool(result.provenance.publication_allowed),
    }
