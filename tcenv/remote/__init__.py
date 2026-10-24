"""Remote evaluator protocol and service primitives.

The package contains the production service core.  Test-only fake backends and
composition roots live under ``tests/support/npu_sim`` and are deliberately not
imported here.
"""

from .protocol import (
    EvaluationRequest,
    EvaluationResult,
    EvaluatorProvenance,
    Frame,
    FrameDecoder,
    HelloRequest,
    ProtocolError,
    encode_frame,
    read_frame,
    write_frame,
)
from .service import RemoteService, ServiceIdentity, SessionGrant

__all__ = [
    "EvaluationRequest",
    "EvaluationResult",
    "EvaluatorProvenance",
    "Frame",
    "FrameDecoder",
    "HelloRequest",
    "ProtocolError",
    "RemoteService",
    "ServiceIdentity",
    "SessionGrant",
    "encode_frame",
    "read_frame",
    "write_frame",
]
