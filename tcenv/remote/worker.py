"""One-job evaluator worker.

The worker is intentionally a separate process from the broker.  Its stdin is a
single request frame and its stdout is a single result frame; stderr is for a
short diagnostic tail only.
"""

from __future__ import annotations

import sys
from typing import BinaryIO, Protocol

from .protocol import EvaluationRequest, ProtocolError, read_frame, write_frame


class RemoteBackend(Protocol):
    """The small backend boundary used by the worker."""

    def evaluate(self, request: EvaluationRequest) -> dict:
        """Return ``{"report": dict, "evidence": ..., "receipt": ...}``."""
        ...


def run_once(backend: RemoteBackend, reader: BinaryIO, writer: BinaryIO) -> int:
    """Read one request, evaluate it, and write one result frame."""
    frame = read_frame(reader)
    if frame.frame_type != "request":
        raise ProtocolError("worker expected a request frame")
    request = EvaluationRequest.from_dict(frame.payload)
    if request.request_id != frame.request_id:
        raise ProtocolError("worker frame and request id disagree")
    result = backend.evaluate(request)
    if not isinstance(result, dict) or not isinstance(result.get("report"), dict):
        raise ProtocolError("backend returned no report object")
    write_frame(
        writer,
        "result",
        request.request_id,
        {
            "report": result["report"],
            "evidence": result.get("evidence"),
            "receipt": result.get("receipt"),
            "signature": str(result.get("signature") or ""),
        },
    )
    return 0


def main() -> int:
    from .production import evaluate_request

    return run_once(evaluate_request, sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
