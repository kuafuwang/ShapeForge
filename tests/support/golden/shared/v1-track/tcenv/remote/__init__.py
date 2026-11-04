"""Remote evaluator production primitives.

The external data plane is gRPC (``tcenv.remote.grpc_service`` and
``tcenv.remote.grpc_client``); the legacy SSH frame client and frontend were
removed.  The length-prefixed frame codec remains because the private
service-to-worker boundary still uses it.  Test-only fake backends and
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
from .identity import ServiceIdentity, SessionGrant
from .job_engine import JobEngine
from .job_store import JobStore
from .job_types import (
    Attempt,
    Capabilities,
    ExpectedEvaluator,
    ExpectedIdentities,
    ExternalEvaluationRequest,
    JobError,
    JobEvent,
    JobFailure,
    JobResult,
    JobSnapshot,
)

__all__ = [
    "EvaluationRequest",
    "EvaluationResult",
    "EvaluatorProvenance",
    "Frame",
    "FrameDecoder",
    "HelloRequest",
    "ProtocolError",
    "ServiceIdentity",
    "SessionGrant",
    "JobEngine",
    "JobStore",
    "Attempt",
    "Capabilities",
    "ExpectedEvaluator",
    "ExpectedIdentities",
    "ExternalEvaluationRequest",
    "JobError",
    "JobEvent",
    "JobFailure",
    "JobResult",
    "JobSnapshot",
    "encode_frame",
    "read_frame",
    "write_frame",
]
