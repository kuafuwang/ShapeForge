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
from .client import CommandTransport, RemoteError, RemoteEvaluator, RemoteEvaluatorPool, Transport, remote_evaluator_from_command, remote_evaluator_pool_from_command
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
from .service import RemoteService

__all__ = [
    "EvaluationRequest",
    "EvaluationResult",
    "EvaluatorProvenance",
    "Frame",
    "FrameDecoder",
    "HelloRequest",
    "ProtocolError",
    "CommandTransport",
    "RemoteError",
    "RemoteEvaluator",
    "RemoteEvaluatorPool",
    "RemoteService",
    "Transport",
    "remote_evaluator_from_command",
    "remote_evaluator_pool_from_command",
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
