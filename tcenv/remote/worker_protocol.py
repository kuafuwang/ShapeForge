"""Versioned private protocol between one job engine and one attempt worker.

This is not the external evaluator API.  It carries exact source bytes as
base64 inside the existing private length-prefixed frame codec and binds every
progress/result frame to ``attempt_id + lease_fence``.  Phase 2C2b can adapt the
existing evaluator worker to this boundary without changing transport code.
"""

from __future__ import annotations

from typing import Any, BinaryIO, Mapping

from .job_types import Attempt, ExternalEvaluationRequest, JobError
from .protocol import Frame, encode_frame, read_frame, write_frame

WORKER_PROTOCOL_VERSION = "tcenv.remote.worker.v1"
# Two maximum external blobs plus a bounded envelope.  Legacy frames keep
# their original 2,000,000-byte default.
MAX_WORKER_FRAME_PAYLOAD = 33_000_000


def _envelope(attempt: Attempt) -> dict[str, str]:
    return {
        "worker_protocol": WORKER_PROTOCOL_VERSION,
        "attempt_id": attempt.attempt_id,
        "lease_fence": str(attempt.lease_fence),
    }


def encode_request(request: ExternalEvaluationRequest, attempt: Attempt) -> bytes:
    return encode_frame("request", request.request_id, {
        **_envelope(attempt), "request": request.to_worker_dict(),
    })


def read_request(reader: BinaryIO) -> tuple[str, ExternalEvaluationRequest, str, int]:
    frame = read_frame(reader)
    if frame.frame_type != "request":
        raise JobError("invalid_argument", "worker expected one request frame")
    payload = frame.payload
    if set(payload) != {"worker_protocol", "attempt_id", "lease_fence", "request"}:
        raise JobError("invalid_argument", "worker request envelope fields are missing or unknown")
    if payload["worker_protocol"] != WORKER_PROTOCOL_VERSION:
        raise JobError("invalid_argument", "unsupported worker protocol")
    request = ExternalEvaluationRequest.from_worker_dict(payload["request"])
    if request.request_id != frame.request_id:
        raise JobError("invalid_argument", "worker frame and request IDs differ")
    attempt_id = payload["attempt_id"]
    try:
        fence = int(payload["lease_fence"])
    except (TypeError, ValueError) as exc:
        raise JobError("invalid_argument", "worker lease fence is invalid") from exc
    if not isinstance(attempt_id, str) or not attempt_id or fence < 1 or str(fence) != payload["lease_fence"]:
        raise JobError("invalid_argument", "worker attempt/fence binding is invalid")
    return attempt_id, request, frame.request_id, fence


def write_progress(
    writer: BinaryIO, *, request_id: str, attempt_id: str, lease_fence: int,
    progress: Mapping[str, Any],
) -> None:
    write_frame(writer, "progress", request_id, {
        "worker_protocol": WORKER_PROTOCOL_VERSION,
        "attempt_id": attempt_id,
        "lease_fence": str(lease_fence),
        "progress": dict(progress),
    })


def write_result(
    writer: BinaryIO, *, request_id: str, attempt_id: str, lease_fence: int,
    report: dict[str, Any], evidence: dict[str, Any] | None,
) -> None:
    write_frame(writer, "result", request_id, {
        "worker_protocol": WORKER_PROTOCOL_VERSION,
        "attempt_id": attempt_id,
        "lease_fence": str(lease_fence),
        "report": report,
        "evidence": evidence,
    }, max_payload=MAX_WORKER_FRAME_PAYLOAD)


def decode_worker_frame(frame: Frame, attempt: Attempt, *, request_id: str) -> tuple[str, Any]:
    payload = frame.payload
    if frame.request_id != request_id:
        raise JobError("invalid_argument", "worker frame request ID does not match the active request")
    if payload.get("worker_protocol") != WORKER_PROTOCOL_VERSION or payload.get("attempt_id") != attempt.attempt_id or payload.get("lease_fence") != str(attempt.lease_fence):
        raise JobError("invalid_argument", "worker frame attempt/fence/protocol binding is invalid")
    if frame.frame_type == "progress":
        if set(payload) != {"worker_protocol", "attempt_id", "lease_fence", "progress"} or not isinstance(payload["progress"], dict):
            raise JobError("invalid_argument", "worker progress frame is invalid")
        return "progress", payload["progress"]
    if frame.frame_type == "result":
        if set(payload) != {"worker_protocol", "attempt_id", "lease_fence", "report", "evidence"} or not isinstance(payload["report"], dict):
            raise JobError("invalid_argument", "worker result frame is invalid")
        if payload["evidence"] is not None and not isinstance(payload["evidence"], dict):
            raise JobError("invalid_argument", "worker evidence must be an object or null")
        return "result", {"report": payload["report"], "evidence": payload["evidence"]}
    raise JobError("invalid_argument", "unexpected worker frame type")
