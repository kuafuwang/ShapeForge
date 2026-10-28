"""Protobuf mappings for the external ``tcenv.remote.v1`` evaluator API.

This module imports only the generated messages.  ``grpc`` stays in the server
and client adapters so the local default runtime needs neither gRPC nor TLS.
Mapping code performs presence, enum, size and integer validation before the
transport-neutral business core sees a request; the core remains the authority
for grants, identity binding, digests and durable state.
"""

from __future__ import annotations

import calendar
import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping

from google.protobuf.timestamp_pb2 import Timestamp

from . import contracts
from .job_types import (
    Attempt,
    BlobDescriptor,
    CandidateDigest,
    Capabilities,
    DeviceGroupBinding,
    ExpectedEvaluator,
    ExpectedIdentities,
    ExternalEvaluationRequest,
    JobError,
    JobEvent,
    JobFailure,
    JobResult,
    JobSnapshot,
    NpuExecutionSpec,
    NpuSourceRequest,
)
from .v1 import evaluator_pb2 as pb

_ID = re.compile(r"[A-Za-z0-9._:-]{1,160}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_UTC = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})(?:\.([0-9]{1,9}))?Z")

KIND_TO_PROTO = {
    "formal_eval": pb.EVALUATION_KIND_FORMAL_EVAL,
    "selfcheck": pb.EVALUATION_KIND_SELFCHECK,
}
KIND_FROM_PROTO = {value: key for key, value in KIND_TO_PROTO.items()}
MODE_TO_PROTO = {
    "quick": pb.EVALUATION_MODE_QUICK,
    "timed": pb.EVALUATION_MODE_TIMED,
    "full": pb.EVALUATION_MODE_FULL,
    "validate": pb.EVALUATION_MODE_VALIDATE,
}
MODE_FROM_PROTO = {value: key for key, value in MODE_TO_PROTO.items()}
CACHE_TO_PROTO = {
    "reuse": pb.CACHE_POLICY_REUSE,
    "refresh": pb.CACHE_POLICY_REFRESH,
    "disabled": pb.CACHE_POLICY_DISABLED,
}
CACHE_FROM_PROTO = {value: key for key, value in CACHE_TO_PROTO.items()}
MEASUREMENT_TO_PROTO = {
    "measured": pb.MEASUREMENT_KIND_MEASURED,
    "simulated": pb.MEASUREMENT_KIND_SIMULATED,
}
MEASUREMENT_FROM_PROTO = {value: key for key, value in MEASUREMENT_TO_PROTO.items()}
STATE_TO_PROTO = {
    "accepted": pb.JOB_STATE_ACCEPTED,
    "queued": pb.JOB_STATE_QUEUED,
    "running": pb.JOB_STATE_RUNNING,
    "interrupted": pb.JOB_STATE_INTERRUPTED,
    "completed": pb.JOB_STATE_COMPLETED,
    "failed": pb.JOB_STATE_FAILED,
    "cancelled": pb.JOB_STATE_CANCELLED,
    "expired": pb.JOB_STATE_EXPIRED,
}
STATE_FROM_PROTO = {value: key for key, value in STATE_TO_PROTO.items()}
ATTEMPT_STATE_TO_PROTO = {
    "queued": pb.ATTEMPT_STATE_QUEUED,
    "running": pb.ATTEMPT_STATE_RUNNING,
    "interrupted": pb.ATTEMPT_STATE_INTERRUPTED,
    "completed": pb.ATTEMPT_STATE_COMPLETED,
    "failed": pb.ATTEMPT_STATE_FAILED,
    "cancelled": pb.ATTEMPT_STATE_CANCELLED,
}
FAILURE_DOMAIN_TO_PROTO = {
    "infrastructure": pb.FAILURE_DOMAIN_INFRASTRUCTURE,
    "policy": pb.FAILURE_DOMAIN_POLICY,
    "protocol": pb.FAILURE_DOMAIN_PROTOCOL,
}
FAILURE_DOMAIN_FROM_PROTO = {value: key for key, value in FAILURE_DOMAIN_TO_PROTO.items()}
STAGE_TO_PROTO = {
    "queued": pb.PROGRESS_STAGE_QUEUED,
    "static_check": pb.PROGRESS_STAGE_STATIC_CHECK,
    "static": pb.PROGRESS_STAGE_STATIC_CHECK,
    "compile": pb.PROGRESS_STAGE_COMPILE,
    "correctness": pb.PROGRESS_STAGE_CORRECTNESS,
    "timing": pb.PROGRESS_STAGE_TIMING,
    "bench": pb.PROGRESS_STAGE_TIMING,
    "finalizing": pb.PROGRESS_STAGE_FINALIZING,
    "done": pb.PROGRESS_STAGE_FINALIZING,
}

REPORT_SCHEMA_ID = contracts.REPORT_SCHEMA_ID
EVIDENCE_SCHEMA_ID = contracts.LEGACY_EVIDENCE_SCHEMA_ID
BLOB_MEDIA_TYPE = "application/json"


def _enum(value: int, mapping: Mapping[int, str], name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value not in mapping:
        raise JobError("invalid_argument", f"{name} is unspecified or unknown to this contract")
    return mapping[value]


def state_from_proto(value: int) -> str:
    """Map a JobState enum, rejecting the unspecified and unknown values."""

    return _enum(value, STATE_FROM_PROTO, "job state")


def identifier(value: Any, name: str) -> str:
    """Public bounded-identifier validator for adapters."""

    return _identifier(value, name)


def hex64(value: Any, name: str) -> str:
    """Public lowercase SHA-256 digest validator for adapters."""

    return _hex64(value, name)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise JobError("invalid_argument", f"{name} must be a bounded opaque identifier")
    return value


def _hex64(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise JobError("invalid_argument", f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _bounded_text(value: Any, name: str, *, max_bytes: int = 256, empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not empty):
        raise JobError("invalid_argument", f"{name} must be a {'possibly empty ' if empty else 'non-empty '}string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise JobError("invalid_argument", f"{name} must contain valid Unicode scalar values") from exc
    if len(encoded) > max_bytes:
        raise JobError("invalid_argument", f"{name} exceeds its UTF-8 byte limit")
    return value


# -- timestamps -------------------------------------------------------------
def timestamp_message(milliseconds: int) -> Timestamp:
    if isinstance(milliseconds, bool) or not isinstance(milliseconds, int) or milliseconds < 0:
        raise JobError("invalid_argument", "timestamp milliseconds must be a non-negative integer")
    message = Timestamp()
    message.FromNanoseconds(milliseconds * 1_000_000)
    return message


def timestamp_milliseconds(message: Timestamp) -> int:
    seconds = message.seconds
    nanos = message.nanos
    if isinstance(seconds, bool) or isinstance(nanos, bool) or not isinstance(seconds, int) or not isinstance(nanos, int):
        raise JobError("invalid_argument", "timestamp components must be integers")
    if seconds < 0 or not 0 <= nanos < 1_000_000_000:
        raise JobError("invalid_argument", "timestamp is outside the RFC 3339 UTC range")
    return seconds * 1000 + nanos // 1_000_000


def timestamp_from_rfc3339(text: str) -> Timestamp:
    if not isinstance(text, str) or len(text) > 40:
        raise JobError("invalid_argument", "issued_at must be an RFC 3339 UTC timestamp")
    match = _UTC.fullmatch(text)
    if match is None:
        raise JobError("invalid_argument", "issued_at must be an RFC 3339 UTC timestamp")
    try:
        seconds = calendar.timegm(datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S").timetuple())
    except ValueError as exc:
        raise JobError("invalid_argument", "issued_at must be a valid RFC 3339 UTC timestamp") from exc
    nanos = int((match.group(2) or "0").ljust(9, "0"))
    message = Timestamp(seconds=seconds, nanos=nanos)
    timestamp_milliseconds(message)
    return message


# -- request ---------------------------------------------------------------
def request_to_proto(
    request: ExternalEvaluationRequest,
    *, dialect: contracts.ContractDialect = contracts.LEGACY_DIALECT,
) -> pb.EvaluationRequest:
    if request.contract != dialect:
        raise JobError("invalid_argument", "request contract differs from the configured evaluator contract")
    message = pb.EvaluationRequest(
        kind=KIND_TO_PROTO[request.kind],
        request_id=request.request_id,
        run_id=request.run_id,
        session_id=request.session_id,
        task=request.task,
        dtype=request.dtype,
        size=request.size,
        target_id=request.target_id,
        task_fingerprint=request.task_fingerprint,
        candidate_sha256="" if dialect.npu_source else request.candidate_sha256,
        source=request.source,
        mode=MODE_TO_PROTO[request.mode],
        reps=request.reps,
        queue_timeout_ms=request.queue_timeout_ms,
        candidate_timeout_ms=request.candidate_timeout_ms,
        fast_bench=request.fast_bench,
        require_contract_launch=request.require_contract_launch,
        validate=request.validate,
        measure_serial=request.measure_serial,
        cache_policy=CACHE_TO_PROTO[request.cache_policy],
    )
    if request.seed is not None:
        message.seed = request.seed
    if request.cases is not None:
        message.cases.names.extend(request.cases)
    if dialect.npu_source:
        if request.npu is None:
            raise JobError("invalid_argument", "NPU request has no CandidateDigest/execution binding")
        message.candidate_digest.CopyFrom(candidate_digest_to_proto(request.npu.candidate_digest))
        message.execution.CopyFrom(execution_to_proto(request.npu.execution))
        if request.npu.blob is not None:
            if not dialect.npu_blob:
                raise JobError("invalid_argument", "the inline contract cannot carry a blob reference")
            message.blob_ref.CopyFrom(blob_ref_to_proto(request.npu.blob))
        elif dialect.npu_blob:
            raise JobError("invalid_argument", "the blob contract requires a blob reference")
        if request.npu.bundle is not None:
            if not dialect.npu_bundle:
                raise JobError("invalid_argument", "only the bundle contract can carry an artifact bundle")
            message.bundle.CopyFrom(bundle_to_proto(request.npu.bundle))
        elif dialect.npu_bundle and not request.source:
            # The bundle contract exposes a source/bundle oneof: exactly one
            # branch must be present, so an empty request is refused.
            raise JobError("invalid_argument", "the bundle contract requires exactly one artifact branch")
        if dialect.npu_bundle and request.npu.bundle is not None and request.source:
            raise JobError("invalid_argument", "a bundle request must not embed inline source bytes")
        if dialect.build_contract:
            if request.build is None:
                raise JobError("invalid_argument", "the build contract requires a BuildProvenance identity")
            message.build.CopyFrom(build_request_to_proto(request.build))
        elif request.build is not None:
            raise JobError("invalid_argument", "only the build contract carries a build identity")
    return message


def build_request_to_proto(value) -> pb.BuildRequest:
    provenance = value.provenance_document()
    message = pb.BuildRequest(
        schema_id=contracts.NPU_BUILD_REQUEST_SCHEMA_ID, mode=value.mode,
    )
    message.provenance.CopyFrom(pb.BuildProvenance(
        schema_id=provenance["schema_id"],
        builder_id=provenance["builder_id"],
        builder_version=provenance["builder_version"],
        build_host=provenance["build_host"],
        build_profile=provenance["build_profile"],
        source_sha256=provenance["source_sha256"],
    ))
    message.provenance.toolchain.CopyFrom(pb.BuildToolchain(
        id=provenance["toolchain"]["id"],
        version=provenance["toolchain"]["version"],
        sdk_version=provenance["toolchain"]["sdk_version"],
    ))
    message.provenance.compile_profile.update(
        {key: str(item) for key, item in provenance["compile_profile"].items()}
    )
    if provenance["artifact_set_digest"] is not None:
        message.provenance.artifact_set_digest = provenance["artifact_set_digest"]
    return message


def build_request_from_proto(message: pb.BuildRequest):
    """Rebuild the typed build request from the wire message (fail closed)."""

    from .job_types import NpuBuildRequest

    if not isinstance(message, pb.BuildRequest):
        raise JobError("invalid_argument", "build is not a BuildRequest message")
    reject_unknown_fields(message, "EvaluationRequest.build")
    reject_unknown_fields(message.provenance, "BuildRequest.provenance")
    if message.HasField("provenance") is False:
        raise JobError("invalid_argument", "build request has no provenance")
    reject_unknown_fields(message.provenance.toolchain, "BuildProvenance.toolchain")
    try:
        return NpuBuildRequest.from_dict({
            "schema_id": message.schema_id,
            "mode": message.mode,
            "provenance": {
                "schema_id": message.provenance.schema_id,
                "builder_id": message.provenance.builder_id,
                "builder_version": message.provenance.builder_version,
                "build_host": message.provenance.build_host,
                "build_profile": message.provenance.build_profile,
                "toolchain": {
                    "id": message.provenance.toolchain.id,
                    "version": message.provenance.toolchain.version,
                    "sdk_version": message.provenance.toolchain.sdk_version,
                },
                "compile_profile": dict(message.provenance.compile_profile),
                "source_sha256": message.provenance.source_sha256,
                "artifact_set_digest": (
                    message.provenance.artifact_set_digest
                    if message.provenance.HasField("artifact_set_digest") else None
                ),
            },
        })
    except JobError:
        raise
    except (TypeError, ValueError, contracts.ContractError) as exc:
        raise JobError("invalid_argument", f"build request is invalid: {exc}") from exc


def bundle_entry_to_proto(value) -> pb.ArtifactBundleEntry:
    message = pb.ArtifactBundleEntry(
        path=value.path, role=value.role, media_type=value.media_type,
        size_bytes=value.size_bytes, sha256=value.sha256,
    )
    if value.blob_ref is not None:
        message.blob_ref.CopyFrom(blob_ref_to_proto(value.blob_ref))
    else:
        message.data = value.data or b""
    return message


def bundle_to_proto(value) -> pb.ArtifactBundle:
    message = pb.ArtifactBundle(schema_id=value.schema_id)
    message.entries.extend(bundle_entry_to_proto(entry) for entry in value.entries)
    return message


def bundle_from_proto(message: pb.ArtifactBundle):
    """Rebuild the typed bundle from the wire message (verifying exact bytes)."""

    from .artifact_bundle import ArtifactBundle, BundleError, BundleEntry

    if not isinstance(message, pb.ArtifactBundle):
        raise JobError("invalid_argument", "bundle is not an ArtifactBundle message")
    reject_unknown_fields(message, "EvaluationRequest.bundle")
    entries = []
    for entry in message.entries:
        reject_unknown_fields(entry, "ArtifactBundleEntry")
        content = entry.WhichOneof("content")
        if content is None:
            raise JobError("invalid_argument", "a bundle entry must be inline or blob-backed")
        if content == "blob_ref":
            reject_unknown_fields(entry.blob_ref, "ArtifactBundleEntry.blob_ref")
            blob_ref = blob_ref_from_proto(entry.blob_ref).to_dict()
            data = None
        else:
            blob_ref = None
            data = bytes(entry.data)
        entries.append(BundleEntry(
            path=entry.path, role=entry.role, media_type=entry.media_type,
            size_bytes=int(entry.size_bytes), sha256=entry.sha256,
            data=data, blob_ref=blob_ref,
        ))
    try:
        return ArtifactBundle(entries=tuple(entries))
    except BundleError as exc:
        raise JobError("invalid_argument", f"artifact bundle is invalid: {exc}") from exc


def candidate_digest_to_proto(value: CandidateDigest) -> pb.CandidateDigest:
    return pb.CandidateDigest(algorithm=value.algorithm, value=value.value)


def blob_ref_to_proto(value: Any) -> pb.BlobRef:
    """Serialize a descriptor; a typed BlobDescriptor or its document is accepted."""

    document = value.to_dict() if hasattr(value, "to_dict") else dict(value)
    return pb.BlobRef(
        content_id=document["content_id"],
        sha256=document["sha256"],
        size_bytes=int(document["size_bytes"]),
        media_type=document["media_type"],
        chunk_size=int(document["chunk_size"]),
    )


def blob_ref_from_proto(message: pb.BlobRef) -> "BlobDescriptor":
    return BlobDescriptor.from_dict({
        "content_id": message.content_id,
        "sha256": message.sha256,
        "size_bytes": int(message.size_bytes),
        "media_type": message.media_type,
        "chunk_size": int(message.chunk_size),
    })


def device_group_to_proto(value: DeviceGroupBinding) -> pb.DeviceGroupBinding:
    return pb.DeviceGroupBinding(
        device_group_id=value.device_group_id,
        vendor_id=value.vendor_id,
        profile_digest=value.profile_digest,
        topology_digest=value.topology_digest,
        resource_generation=value.resource_generation,
    )


def execution_to_proto(
    value: NpuExecutionSpec, *, dialect: contracts.ContractDialect | None = None,
) -> pb.NpuExecutionSpec:
    message = pb.NpuExecutionSpec(
        artifact_kind=value.artifact_kind,
        launch_kind=value.launch_kind,
        launch_subtype=value.launch_subtype,
        source_media_type=value.source_media_type,
        required_timing_sources=list(value.required_timing_sources),
        synchronization=value.synchronization,
    )
    message.device_group.CopyFrom(device_group_to_proto(value.device_group))
    if value.spec_version >= 4:
        resources = value.declared_resources
        message.spec_version = value.spec_version
        message.task_family = value.task_family or ""
        message.execution_schema_id = contracts.NPU_BUNDLE_EXECUTION_SCHEMA_ID
        message.resources.logical_devices = resources.logical_devices
        if resources.required_rank_spans is not None:
            message.resources.required_rank_spans = resources.required_rank_spans
        if resources.required_die_spans is not None:
            message.resources.required_die_spans = resources.required_die_spans
    return message


def execution_from_proto(
    message: pb.NpuExecutionSpec, *, dialect: contracts.ContractDialect | None = None,
) -> NpuExecutionSpec:
    if not isinstance(message, pb.NpuExecutionSpec):
        raise JobError("invalid_argument", "execution is not an NpuExecutionSpec message")
    reject_unknown_fields(message, "NpuExecutionSpec")
    if not message.HasField("device_group"):
        raise JobError("invalid_argument", "NPU execution requires a device group binding")
    dialect = dialect or contracts.NPU_SOURCE_DIALECT
    has_resources = message.HasField("resources")
    if not dialect.npu_bundle:
        if message.spec_version or message.task_family or message.execution_schema_id or has_resources:
            # Silently ignoring a version 4 execution spec on an earlier contract
            # would accept an execution shape the server never validated.
            raise JobError(
                "invalid_argument",
                "this contract version does not accept the version 4 execution spec fields",
            )
    document: dict[str, Any] = {
        "artifact_kind": message.artifact_kind,
        "launch_kind": message.launch_kind,
        "launch_subtype": message.launch_subtype,
        "source_media_type": message.source_media_type,
        "device_group": {
            "device_group_id": message.device_group.device_group_id,
            "vendor_id": message.device_group.vendor_id,
            "profile_digest": message.device_group.profile_digest,
            "topology_digest": message.device_group.topology_digest,
            "resource_generation": message.device_group.resource_generation,
        },
        "required_timing_sources": list(message.required_timing_sources),
        "synchronization": message.synchronization,
    }
    if dialect.npu_bundle:
        allowed_versions = (4, 5) if dialect.build_contract else (4,)
        if message.spec_version not in allowed_versions or message.execution_schema_id != contracts.NPU_BUNDLE_EXECUTION_SCHEMA_ID:
            raise JobError(
                "invalid_argument",
                f"this contract version requires an execution spec in {sorted(allowed_versions)}",
            )
        if not has_resources:
            raise JobError("invalid_argument", "the version 4 execution spec requires declared resources")
        reject_unknown_fields(message.resources, "NpuResourceRequirements")
        document["schema_id"] = message.execution_schema_id
        document["task_family"] = message.task_family
        document["resources"] = {
            "logical_devices": int(message.resources.logical_devices),
            "required_rank_spans": (
                int(message.resources.required_rank_spans)
                if message.resources.HasField("required_rank_spans") else None
            ),
            "required_die_spans": (
                int(message.resources.required_die_spans)
                if message.resources.HasField("required_die_spans") else None
            ),
        }
    try:
        return NpuExecutionSpec.from_dict(document)
    except JobError:
        raise
    except (TypeError, ValueError, contracts.ContractError) as exc:
        raise JobError("invalid_argument", f"NPU execution is invalid: {exc}") from exc


def reject_unknown_fields(message, name: str) -> None:
    """Reject Protobuf fields this contract version does not understand.

    Protobuf preserves unknown fields, so a newer execution option must not be
    silently ignored before durable acceptance.
    """

    if not hasattr(message, "SerializeToString") or not hasattr(message, "DiscardUnknownFields"):
        raise JobError("invalid_argument", f"{name} is not a Protobuf message")
    clone = message.__class__()
    clone.CopyFrom(message)
    clone.DiscardUnknownFields()
    if message.SerializeToString(deterministic=True) != clone.SerializeToString(deterministic=True):
        raise JobError("invalid_argument", f"{name} contains unknown fields")


def request_from_proto(
    message: pb.EvaluationRequest,
    *, dialect: contracts.ContractDialect = contracts.LEGACY_DIALECT,
) -> ExternalEvaluationRequest:
    if not isinstance(message, pb.EvaluationRequest):
        raise JobError("invalid_argument", "request is not an EvaluationRequest message")
    reject_unknown_fields(message, "EvaluationRequest")
    if message.HasField("cases"):
        reject_unknown_fields(message.cases, "EvaluationRequest.cases")
    if message.HasField("candidate_digest") != dialect.npu_source or message.HasField("execution") != dialect.npu_source:
        raise JobError("invalid_argument", "request CandidateDigest/execution presence differs from the contract")
    if message.HasField("blob_ref") != dialect.npu_blob:
        raise JobError("invalid_argument", "request blob reference presence differs from the contract")
    if message.HasField("bundle") and not dialect.npu_bundle:
        raise JobError("invalid_argument", "only the bundle contract accepts an artifact bundle")
    if message.HasField("build") != dialect.build_contract:
        raise JobError(
            "invalid_argument", "request build provenance presence differs from the contract"
        )
    if dialect.npu_bundle and message.HasField("blob_ref"):
        raise JobError(
            "invalid_argument",
            "the bundle contract transports artifact bytes inline or per entry, not as one request blob",
        )
    build = None
    if dialect.npu_source:
        if message.candidate_sha256:
            raise JobError("invalid_argument", "the NPU contract does not accept candidate_sha256")
        if not message.candidate_digest.algorithm or not message.candidate_digest.value:
            raise JobError("invalid_argument", "request CandidateDigest is incomplete")
        allowed = dialect.candidate_algorithms
        digest = contracts.candidate_digest_document(
            {"algorithm": message.candidate_digest.algorithm, "value": message.candidate_digest.value},
            allowed_algorithms=allowed,
        )
        blob = None if not message.HasField("blob_ref") else blob_ref_from_proto(message.blob_ref)
        bundle = None if not message.HasField("bundle") else bundle_from_proto(message.bundle)
        build = None if not message.HasField("build") else build_request_from_proto(message.build)
        execution = execution_from_proto(message.execution, dialect=dialect)
        if bundle is not None and bytes(message.source):
            raise JobError(
                "invalid_argument", "a bundle request must not embed inline source bytes"
            )
        if bundle is None and dialect.npu_bundle and not bytes(message.source):
            raise JobError("invalid_argument", "the bundle contract requires exactly one artifact branch")
        npu = NpuSourceRequest(
            candidate_digest=CandidateDigest(
                digest["algorithm"], digest["value"], allowed_algorithms=allowed,
            ),
            execution=execution,
            blob=blob,
            bundle=bundle,
        )
    else:
        if message.candidate_digest.ByteSize() or message.execution.ByteSize():
            raise JobError("invalid_argument", "the legacy contract does not accept NPU request fields")
        if message.blob_ref.ByteSize() or message.bundle.ByteSize():
            raise JobError(
                "invalid_argument", "the legacy contract does not accept a blob or bundle reference"
            )
        npu = None
        try:
            message.source.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise JobError("invalid_argument", "candidate source is not valid UTF-8 text") from exc
    kind = _enum(message.kind, KIND_FROM_PROTO, "kind")
    mode = _enum(message.mode, MODE_FROM_PROTO, "mode")
    cache_policy = _enum(message.cache_policy, CACHE_FROM_PROTO, "cache_policy")
    source = bytes(message.source)
    if len(source) > contracts.MAX_SOURCE_BYTES:
        raise JobError("invalid_argument", "source exceeds the protocol byte limit")
    if not 0 <= message.reps <= contracts.MAX_REPS:
        raise JobError("invalid_argument", "reps is outside the protocol bounds")
    for name in ("queue_timeout_ms", "candidate_timeout_ms"):
        value = getattr(message, name)
        if not 1 <= value <= contracts.MAX_TIMEOUT_MS:
            raise JobError("invalid_argument", f"{name} is outside the protocol bounds")
    seed = message.seed if message.HasField("seed") else None
    cases = tuple(message.cases.names) if message.HasField("cases") else None
    if cases is not None and len(cases) > contracts.MAX_CASES:
        raise JobError("invalid_argument", "cases exceeds the protocol bounds")
    try:
        return ExternalEvaluationRequest(
            kind=kind,
            request_id=_identifier(message.request_id, "request_id"),
            run_id=_identifier(message.run_id, "run_id"),
            session_id=_identifier(message.session_id, "session_id"),
            task=_bounded_text(message.task, "task"),
            dtype=_bounded_text(message.dtype, "dtype", max_bytes=64),
            size=_bounded_text(message.size, "size", max_bytes=128),
            target_id=_identifier(message.target_id, "target_id"),
            task_fingerprint=_hex64(message.task_fingerprint, "task_fingerprint"),
            candidate_sha256="" if dialect.npu_source else _hex64(message.candidate_sha256, "candidate_sha256"),
            source=source,
            mode=mode,
            seed=seed,
            reps=int(message.reps),
            queue_timeout_ms=int(message.queue_timeout_ms),
            candidate_timeout_ms=int(message.candidate_timeout_ms),
            cases=cases,
            fast_bench=bool(message.fast_bench),
            require_contract_launch=bool(message.require_contract_launch),
            validate=bool(message.validate),
            measure_serial=bool(message.measure_serial),
            cache_policy=cache_policy,
            npu=npu,
            contract_dialect=dialect if npu is not None else None,
            build=build,
        )
    except contracts.ContractError as exc:
        raise JobError("invalid_argument", str(exc)) from exc
    except JobError:
        raise


# -- identities -------------------------------------------------------------
def expected_evaluator_to_proto(expected: ExpectedEvaluator) -> pb.ExpectedEvaluator:
    return pb.ExpectedEvaluator(
        site_id=expected.site_id,
        device_id=expected.device_id,
        target_id=expected.target_id,
        worker_build=expected.worker_build,
        task_pack_digest=expected.task_pack_digest,
        evaluator_epoch=expected.evaluator_epoch,
    )


def expected_identities_to_proto(expected: ExpectedIdentities) -> pb.ExpectedIdentities:
    message = pb.ExpectedIdentities(
        api_contract_id=expected.api_contract_id,
        evidence_contract_id=expected.evidence_contract_id,
    )
    message.evaluator.CopyFrom(expected_evaluator_to_proto(expected.evaluator))
    return message


def expected_evaluator_document(
    message: pb.ExpectedEvaluator | None, *, strict: bool,
) -> dict[str, str]:
    """Return the non-empty expectation fields; ``strict`` pins every identity."""

    if message is None:
        raise JobError("invalid_argument", "expected evaluator identity is required")
    document = {
        "site_id": message.site_id,
        "device_id": message.device_id,
        "target_id": message.target_id,
        "worker_build": message.worker_build,
        "task_pack_digest": message.task_pack_digest,
        "evaluator_epoch": message.evaluator_epoch,
    }
    present = {key: value for key, value in document.items() if value}
    if strict:
        missing = [key for key, value in document.items() if not value]
        if missing:
            raise JobError("invalid_argument", f"expected evaluator fields are missing: {missing}")
    for key, value in present.items():
        if not isinstance(value, str):
            raise JobError("invalid_argument", f"expected evaluator {key} is invalid")
    if "site_id" in present:
        _identifier(present["site_id"], "expected site_id")
    if "device_id" in present:
        _identifier(present["device_id"], "expected device_id")
    if "target_id" in present:
        _identifier(present["target_id"], "expected target_id")
    if "worker_build" in present:
        _bounded_text(present["worker_build"], "expected worker_build")
    for key in ("task_pack_digest", "evaluator_epoch"):
        if key in present:
            _hex64(present[key], f"expected {key}")
    return present


def expected_identities_from_proto(message: pb.ExpectedIdentities) -> ExpectedIdentities:
    if not isinstance(message, pb.ExpectedIdentities):
        raise JobError("invalid_argument", "expected identities are not an ExpectedIdentities message")
    if not message.HasField("evaluator"):
        raise JobError("invalid_argument", "expected evaluator identity is required")
    fields = expected_evaluator_document(message.evaluator, strict=True)
    try:
        expected = ExpectedIdentities(
            ExpectedEvaluator(**fields),
            api_contract_id=message.api_contract_id,
            evidence_contract_id=message.evidence_contract_id,
        )
        # Resolve the exact pair so an unsupported or mixed contract is rejected
        # before the store ever records a capability.
        expected.validate()
        return expected
    except JobError:
        raise
    except (TypeError, ValueError) as exc:
        raise JobError("invalid_argument", f"expected identities are invalid: {exc}") from exc


def _document_dialect(document: Mapping[str, Any]) -> contracts.ContractDialect:
    try:
        return contracts.dialect_for(document["api_contract_id"], document["evidence_contract_id"])
    except (KeyError, TypeError, contracts.ContractError) as exc:
        raise JobError("invalid_argument", "unknown API/evidence contract pair") from exc


def expected_identities_from_provenance(document: Mapping[str, Any]) -> ExpectedIdentities:
    dialect = _document_dialect(document)
    if isinstance(document, Mapping) and ("attempt_id" in document or "request_digest" in document):
        provenance = contracts.provenance_document(document, require_attempt=True, dialect=dialect)
    else:
        provenance = contracts.provenance_document(document, require_attempt=False, dialect=dialect)
    return ExpectedIdentities(
        ExpectedEvaluator(
            site_id=provenance["site_id"],
            device_id=provenance["device_id"],
            target_id=provenance["target_id"],
            worker_build=provenance["worker_build"],
            task_pack_digest=provenance["task_pack_digest"],
            evaluator_epoch=provenance["evaluator_epoch"],
        ),
        api_contract_id=provenance["api_contract_id"],
        evidence_contract_id=provenance["evidence_contract_id"],
    )


# -- provenance, receipt and signature -------------------------------------
def provenance_to_proto(
    document: Mapping[str, Any], *, require_attempt: bool,
    dialect: contracts.ContractDialect | None = None,
) -> pb.EvaluatorProvenance:
    selected = dialect or _document_dialect(document)
    provenance = contracts.provenance_document(
        document, require_attempt=require_attempt, dialect=selected,
    )
    message = pb.EvaluatorProvenance(
        schema_version=provenance["schema_version"],
        evaluator_epoch=provenance["evaluator_epoch"],
        site_id=provenance["site_id"],
        device_id=provenance["device_id"],
        backend=provenance["backend"],
        backend_version=provenance["backend_version"],
        target_id=provenance["target_id"],
        worker_build=provenance["worker_build"],
        task_pack_digest=provenance["task_pack_digest"],
        calibration_digest=provenance["calibration_digest"],
        protocol_version=provenance["protocol_version"],
        api_contract_id=provenance["api_contract_id"],
        evidence_contract_id=provenance["evidence_contract_id"],
        measurement_kind=MEASUREMENT_TO_PROTO[provenance["measurement_kind"]],
        publication_allowed=provenance["publication_allowed"],
    )
    if selected.npu_source:
        message.device_group.CopyFrom(pb.DeviceGroupBinding(**provenance["device_group"]))
        message.artifact_kind = provenance["artifact_kind"]
        message.launch_kind = provenance["launch_kind"]
        message.launch_subtype = provenance["launch_subtype"]
        message.timing_sources.extend(provenance["timing_sources"])
    if require_attempt:
        message.attempt_id = provenance["attempt_id"]
        message.request_digest = provenance["request_digest"]
    return message


def provenance_from_proto(
    message: pb.EvaluatorProvenance, *, require_attempt: bool,
    dialect: contracts.ContractDialect | None = None,
) -> dict[str, Any]:
    if not isinstance(message, pb.EvaluatorProvenance):
        raise JobError("invalid_argument", "provenance is not an EvaluatorProvenance message")
    if message.HasField("attempt_id") != require_attempt or message.HasField("request_digest") != require_attempt:
        raise JobError("invalid_argument", "provenance attempt/digest presence differs from the contract")
    selected = dialect or _document_dialect({
        "api_contract_id": message.api_contract_id,
        "evidence_contract_id": message.evidence_contract_id,
    })
    if message.HasField("device_group") != selected.npu_source:
        raise JobError("invalid_argument", "provenance device group presence differs from the contract")
    if bool(message.artifact_kind) != selected.npu_source or bool(message.launch_kind) != selected.npu_source:
        raise JobError("invalid_argument", "provenance execution shape presence differs from the contract")
    measurement = _enum(message.measurement_kind, MEASUREMENT_FROM_PROTO, "measurement_kind")
    document: dict[str, Any] = {
        "schema_version": int(message.schema_version),
        "evaluator_epoch": message.evaluator_epoch,
        "site_id": message.site_id,
        "device_id": message.device_id,
        "backend": message.backend,
        "backend_version": message.backend_version,
        "target_id": message.target_id,
        "worker_build": message.worker_build,
        "task_pack_digest": message.task_pack_digest,
        "calibration_digest": message.calibration_digest,
        "protocol_version": message.protocol_version,
        "api_contract_id": message.api_contract_id,
        "evidence_contract_id": message.evidence_contract_id,
        "measurement_kind": measurement,
        "publication_allowed": bool(message.publication_allowed),
    }
    if selected.npu_source:
        if not message.HasField("device_group") or message.launch_subtype == "" or not message.timing_sources:
            raise JobError("invalid_argument", "NPU provenance is missing its device/execution binding")
        document["device_group"] = {
            "device_group_id": message.device_group.device_group_id,
            "vendor_id": message.device_group.vendor_id,
            "profile_digest": message.device_group.profile_digest,
            "topology_digest": message.device_group.topology_digest,
            "resource_generation": message.device_group.resource_generation,
        }
        document["artifact_kind"] = message.artifact_kind
        document["launch_kind"] = message.launch_kind
        document["launch_subtype"] = message.launch_subtype
        document["timing_sources"] = list(message.timing_sources)
    if require_attempt:
        document["attempt_id"] = message.attempt_id
        document["request_digest"] = message.request_digest
    try:
        return contracts.provenance_document(
            document, require_attempt=require_attempt, dialect=selected,
        )
    except contracts.ContractError as exc:
        raise JobError("invalid_argument", f"provenance is invalid: {exc}") from exc


def receipt_to_proto(
    receipt: Mapping[str, Any], *, dialect: contracts.ContractDialect = contracts.LEGACY_DIALECT,
) -> pb.Receipt:
    document = contracts.receipt_document(receipt, dialect=dialect)
    message = pb.Receipt(
        receipt_id=document["receipt_id"],
        request_id=document["request_id"],
        attempt_id=document["attempt_id"],
        request_digest=document["request_digest"],
        candidate_sha256="" if dialect.npu_source else document["candidate_sha256"],
        lease_fence=int(document["lease_fence"]),
        session_sequence=int(document["session_sequence"]),
        issued_at=timestamp_from_rfc3339(document["issued_at"]),
    )
    if dialect.npu_source:
        message.candidate_digest.CopyFrom(pb.CandidateDigest(**document["candidate_digest"]))
        message.device_group.CopyFrom(pb.DeviceGroupBinding(**document["device_group"]))
    return message


def receipt_from_proto(
    message: pb.Receipt, *, dialect: contracts.ContractDialect | None = None,
) -> dict[str, Any]:
    if not isinstance(message, pb.Receipt):
        raise JobError("invalid_argument", "receipt is not a Receipt message")
    if not message.HasField("issued_at"):
        raise JobError("invalid_argument", "receipt issued_at is required")
    selected = dialect
    if selected is None:
        selected = contracts.NPU_SOURCE_DIALECT if message.HasField("candidate_digest") else contracts.LEGACY_DIALECT
    if message.HasField("candidate_digest") != selected.npu_source or message.HasField("device_group") != selected.npu_source:
        raise JobError("invalid_argument", "receipt candidate/device binding presence differs from the contract")
    if selected.npu_source and message.candidate_sha256:
        raise JobError("invalid_argument", "the NPU receipt does not dual-write candidate_sha256")
    if not selected.npu_source and (message.candidate_digest.ByteSize() or message.device_group.ByteSize()):
        raise JobError("invalid_argument", "the legacy receipt does not accept NPU binding fields")
    document = {
        "receipt_id": message.receipt_id,
        "request_id": message.request_id,
        "attempt_id": message.attempt_id,
        "request_digest": message.request_digest,
        "lease_fence": str(int(message.lease_fence)),
        "session_sequence": str(int(message.session_sequence)),
        "issued_at": _rfc3339_from_timestamp(message.issued_at),
    }
    if selected.npu_source:
        document["candidate_digest"] = {
            "algorithm": message.candidate_digest.algorithm,
            "value": message.candidate_digest.value,
        }
        document["device_group"] = {
            "device_group_id": message.device_group.device_group_id,
            "vendor_id": message.device_group.vendor_id,
            "profile_digest": message.device_group.profile_digest,
            "topology_digest": message.device_group.topology_digest,
            "resource_generation": message.device_group.resource_generation,
        }
    else:
        document["candidate_sha256"] = message.candidate_sha256
    try:
        return contracts.receipt_document(document, dialect=selected)
    except contracts.ContractError as exc:
        raise JobError("invalid_argument", f"receipt is invalid: {exc}") from exc


def _rfc3339_from_timestamp(message: Timestamp) -> str:
    # The canonical receipt document is signed verbatim.  A Timestamp with
    # sub-millisecond precision cannot be reconstructed from the canonical
    # three-digit form without changing the signed bytes, so fail closed.
    if message.nanos % 1_000_000:
        raise JobError("invalid_argument", "receipt issued_at has sub-millisecond precision not covered by the canonical document")
    milliseconds = timestamp_milliseconds(message)
    seconds, millis = divmod(milliseconds, 1000)
    try:
        base = datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    except (OverflowError, OSError, ValueError) as exc:
        raise JobError("invalid_argument", "receipt issued_at is outside the supported range") from exc
    return f"{base}.{millis:03d}Z"


def signature_to_proto(signature: Mapping[str, Any]) -> pb.ResultSignature:
    algorithm = signature.get("algorithm")
    key_id = signature.get("key_id")
    value = signature.get("signature")
    if algorithm != contracts.RESULT_SIGNATURE_ALGORITHM:
        raise JobError("invalid_argument", "result signature algorithm is not supported")
    _bounded_text(key_id, "signature key_id", max_bytes=160)
    if not isinstance(value, str) or len(value) != 64 or not _HEX64.fullmatch(value):
        raise JobError("invalid_argument", "result signature must be 32 raw bytes")
    return pb.ResultSignature(algorithm=algorithm, key_id=key_id, signature=bytes.fromhex(value))


def signature_from_proto(message: pb.ResultSignature) -> dict[str, str]:
    if not isinstance(message, pb.ResultSignature):
        raise JobError("invalid_argument", "signature is not a ResultSignature message")
    if message.algorithm != contracts.RESULT_SIGNATURE_ALGORITHM:
        raise JobError("invalid_argument", "result signature algorithm is not supported")
    value = bytes(message.signature)
    if len(value) != 32:
        raise JobError("invalid_argument", "result signature must be exactly 32 raw bytes")
    _bounded_text(message.key_id, "signature key_id", max_bytes=160)
    return {
        "algorithm": message.algorithm,
        "key_id": message.key_id,
        "signature": value.hex(),
    }


# -- blobs and results ------------------------------------------------------
def blob_to_proto(data: bytes, descriptor: Mapping[str, Any]) -> pb.OpaqueBlob:
    try:
        normalized = contracts.validate_blob(descriptor, data, allow_empty=True)
    except contracts.ContractError as exc:
        raise JobError("invalid_argument", f"stored result blob is invalid: {exc}") from exc
    return pb.OpaqueBlob(
        schema_id=normalized["schema_id"],
        media_type=normalized["media_type"],
        size_bytes=normalized["size_bytes"],
        sha256=normalized["sha256"],
        data=data,
    )


def blob_from_proto(message: pb.OpaqueBlob, *, allow_empty: bool) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(message, pb.OpaqueBlob):
        raise JobError("invalid_argument", "blob is not an OpaqueBlob message")
    data = bytes(message.data)
    if len(data) > contracts.MAX_BLOB_BYTES:
        raise JobError("invalid_argument", "opaque blob exceeds the protocol byte limit")
    descriptor = {
        "schema_id": message.schema_id,
        "media_type": message.media_type,
        "size_bytes": int(message.size_bytes),
        "sha256": message.sha256,
    }
    try:
        normalized = contracts.validate_blob(descriptor, data, allow_empty=allow_empty)
    except contracts.ContractError as exc:
        raise JobError("invalid_argument", f"opaque blob is invalid: {exc}") from exc
    return data, normalized


def _result_document(
    report: tuple[bytes, dict[str, Any]],
    evidence: tuple[bytes, dict[str, Any]] | None,
    provenance: dict[str, Any],
    receipt: dict[str, Any],
    *,
    dialect: contracts.ContractDialect = contracts.LEGACY_DIALECT,
) -> dict[str, Any]:
    document = {
        "provenance": provenance,
        "report": report[1],
        "evidence": None if evidence is None else evidence[1],
        "receipt": receipt,
    }
    try:
        contracts.result_signature_document(document, key_id="placeholder/1", dialect=dialect)
    except contracts.ContractError as exc:
        raise JobError("invalid_argument", f"result envelope is invalid: {exc}") from exc
    return document


def result_to_proto(
    result: JobResult, *, dialect: contracts.ContractDialect | None = None,
) -> pb.EvaluationResult:
    selected = dialect or _document_dialect(result.provenance)
    message = pb.EvaluationResult(
        report=blob_to_proto(result.report, result.report_descriptor),
        provenance=provenance_to_proto(result.provenance, require_attempt=True, dialect=selected),
        receipt=receipt_to_proto(result.receipt, dialect=selected),
    )
    if result.evidence is not None and result.evidence_descriptor is not None:
        message.evidence.CopyFrom(blob_to_proto(result.evidence, result.evidence_descriptor))
    if result.signature is not None:
        message.signature.CopyFrom(signature_to_proto(result.signature))
    return message


def result_from_proto(
    message: pb.EvaluationResult,
    *, dialect: contracts.ContractDialect | None = None,
) -> tuple[bytes, dict[str, Any], bytes | None, dict[str, Any] | None, dict[str, Any], dict[str, Any], dict[str, str]]:
    """Validate one result message and return its transport-neutral pieces."""

    if not isinstance(message, pb.EvaluationResult):
        raise JobError("invalid_argument", "result is not an EvaluationResult message")
    if not message.HasField("report") or not message.HasField("provenance") or not message.HasField("receipt") or not message.HasField("signature"):
        raise JobError("invalid_argument", "result is missing report, provenance, receipt or signature")
    report = blob_from_proto(message.report, allow_empty=False)
    evidence = blob_from_proto(message.evidence, allow_empty=True) if message.HasField("evidence") else None
    selected = dialect
    if selected is None:
        selected = _document_dialect({
            "api_contract_id": message.provenance.api_contract_id,
            "evidence_contract_id": message.provenance.evidence_contract_id,
        })
    provenance = provenance_from_proto(message.provenance, require_attempt=True, dialect=selected)
    receipt = receipt_from_proto(message.receipt, dialect=selected)
    signature = signature_from_proto(message.signature)
    _result_document(report, evidence, provenance, receipt, dialect=selected)
    return (
        report[0], report[1],
        None if evidence is None else evidence[0],
        None if evidence is None else evidence[1],
        provenance, receipt, signature,
    )


# -- job snapshots and events ----------------------------------------------
def failure_to_proto(failure: JobFailure) -> pb.JobFailure:
    return pb.JobFailure(
        domain=FAILURE_DOMAIN_TO_PROTO.get(failure.domain, pb.FAILURE_DOMAIN_UNSPECIFIED),
        retryable=bool(failure.retryable),
        code=failure.code,
        message=failure.message,
    )


def failure_from_proto(message: pb.JobFailure) -> JobFailure:
    domain = _enum(message.domain, FAILURE_DOMAIN_FROM_PROTO, "failure domain")
    return JobFailure(domain, bool(message.retryable), message.code, message.message)


def attempt_to_proto(attempt: Attempt) -> pb.Attempt:
    message = pb.Attempt(
        attempt_id=attempt.attempt_id,
        lease_fence=attempt.lease_fence,
        state=ATTEMPT_STATE_TO_PROTO[attempt.state],
        retry_index=attempt.retry_index,
        cancel_requested=attempt.cancel_requested,
        queued_at=timestamp_message(attempt.queued_at_ms),
    )
    if attempt.started_at_ms is not None:
        message.started_at.CopyFrom(timestamp_message(attempt.started_at_ms))
    if attempt.finished_at_ms is not None:
        message.finished_at.CopyFrom(timestamp_message(attempt.finished_at_ms))
    if attempt.failure is not None:
        message.failure.CopyFrom(failure_to_proto(attempt.failure))
    return message


def snapshot_to_proto(snapshot: JobSnapshot) -> pb.JobSnapshot:
    base_provenance = {
        key: value for key, value in snapshot.provenance.items()
        if key not in {"attempt_id", "request_digest"}
    }
    message = pb.JobSnapshot(
        request_id=snapshot.request_id,
        request_digest=snapshot.request_digest,
        state=STATE_TO_PROTO[snapshot.state],
        provenance=provenance_to_proto(base_provenance, require_attempt=False),
        cancel_requested=snapshot.cancel_requested,
        last_sequence=snapshot.last_sequence,
        accepted_at=timestamp_message(snapshot.accepted_at_ms),
        updated_at=timestamp_message(snapshot.updated_at_ms),
    )
    # A job snapshot's provenance always binds the request digest; the attempt
    # is present from the first allocated attempt onward, including results.
    message.provenance.request_digest = snapshot.request_digest
    if snapshot.current_attempt is not None:
        message.provenance.attempt_id = snapshot.current_attempt.attempt_id
        message.current_attempt.CopyFrom(attempt_to_proto(snapshot.current_attempt))
    if snapshot.result is not None:
        message.result.CopyFrom(result_to_proto(snapshot.result))
    if snapshot.failure is not None:
        message.failure.CopyFrom(failure_to_proto(snapshot.failure))
    if snapshot.expires_at_ms is not None:
        message.expires_at.CopyFrom(timestamp_message(snapshot.expires_at_ms))
    return message


def progress_to_proto(progress: Mapping[str, Any]) -> pb.JobProgress:
    stage = STAGE_TO_PROTO.get(str(progress.get("stage")), pb.PROGRESS_STAGE_QUEUED)
    detail = progress.get("detail")
    message = pb.JobProgress(
        stage=stage,
        detail=detail if isinstance(detail, str) else "",
    )
    for key in ("completed_reps", "total_reps"):
        value = progress.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 0xFFFFFFFF:
            setattr(message, key, value)
    observed = progress.get("observed_at_ms")
    if isinstance(observed, int) and not isinstance(observed, bool) and observed >= 0:
        message.observed_at.CopyFrom(timestamp_message(observed))
    return message


def event_to_proto(event: JobEvent) -> pb.JobEvent:
    message = pb.JobEvent(
        request_id=event.request_id,
        lease_fence=event.lease_fence,
        sequence=event.sequence,
    )
    if event.attempt_id is not None:
        message.attempt_id = event.attempt_id
    elif event.progress is not None or event.snapshot is not None:
        # Only the pre-attempt accepted snapshot may omit the attempt id.
        if event.snapshot is None or event.snapshot.current_attempt is not None:
            raise JobError("data_loss", "job event is missing its attempt binding")
    if event.progress is not None:
        message.progress.CopyFrom(progress_to_proto(event.progress))
    elif event.snapshot is not None:
        message.snapshot.CopyFrom(snapshot_to_proto(event.snapshot))
    else:
        raise JobError("data_loss", "job event has neither progress nor snapshot")
    return message


# -- capabilities -----------------------------------------------------------
_LIMIT_FIELDS = (
    "max_source_bytes", "max_reps", "max_timeout_ms", "max_cases", "max_request_bytes",
    "max_queue_depth", "max_active_attempts", "max_watch_streams", "max_job_store_bytes",
    "max_blob_bytes",
)


def capabilities_to_proto(capabilities: Capabilities, *, extra_limits: Mapping[str, int] | None = None) -> pb.Capabilities:
    selected = _document_dialect(capabilities.provenance)
    provenance = contracts.provenance_document(
        capabilities.provenance, require_attempt=False, dialect=selected,
    )
    message = pb.Capabilities(
        client_nonce=capabilities.client_nonce,
        server_nonce=capabilities.server_nonce,
        capability_id=capabilities.capability_id,
        evaluator_epoch=provenance["evaluator_epoch"],
        api_contract_id=provenance["api_contract_id"],
        evidence_contract_id=provenance["evidence_contract_id"],
        provenance=provenance_to_proto(provenance, require_attempt=False, dialect=selected),
        supported_features=sorted(capabilities.supported_features),
        issued_at=timestamp_message(capabilities.issued_at_ms),
        valid_for_ms=int(capabilities.valid_for_ms),
    )
    for target in capabilities.target_capabilities:
        modes = [MODE_TO_PROTO[mode] for mode in target.get("modes", ()) if mode in MODE_TO_PROTO]
        target_message = message.targets.add(
            target_id=target.get("target_id", ""),
            backend=target.get("backend", ""),
            modes=modes,
            features=list(target.get("features", ())),
        )
        if selected.npu_source:
            target_message.artifact_kinds.extend(target.get("artifact_kinds", ()))
            target_message.launch_kinds.extend(target.get("launch_kinds", ()))
            target_message.launch_subtypes.extend(target.get("launch_subtypes", ()))
            target_message.timing_sources.extend(target.get("timing_sources", ()))
            for group in target.get("device_groups", ()):
                target_message.device_groups.add(
                    device_group_id=group.get("device_group_id", ""),
                    vendor_id=group.get("vendor_id", ""),
                    profile_digest=group.get("profile_digest", ""),
                    topology_digest=group.get("topology_digest", ""),
                    resource_generation=group.get("resource_generation", ""),
                    logical_devices=int(group.get("logical_devices", 0)),
                    state=group.get("state", ""),
                    timing_sources=list(group.get("timing_sources", ())),
                )
    limits = dict(capabilities.limits)
    limits.update(extra_limits or {})
    for name in _LIMIT_FIELDS:
        value = limits.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 0xFFFFFFFFFFFFFFFF:
            setattr(message.limits, name, value)
    return message


def error_detail(
    *,
    request_id: str = "",
    attempt_id: str = "",
    request_digest: str = "",
    reason: str = "",
    missing_features: tuple[str, ...] | list[str] = (),
    failure_domain: str = "",
    retryable: bool = False,
    evaluator_epoch: str = "",
    api_contract_id: str = "",
    evidence_contract_id: str = "",
    lease_fence: int = 0,
) -> pb.EvaluationErrorDetail:
    return pb.EvaluationErrorDetail(
        request_id=request_id,
        attempt_id=attempt_id,
        request_digest=request_digest,
        reason=str(reason)[:160],
        missing_features=list(missing_features),
        failure_domain=FAILURE_DOMAIN_TO_PROTO.get(failure_domain, pb.FAILURE_DOMAIN_UNSPECIFIED),
        retryable=bool(retryable),
        evaluator_epoch=evaluator_epoch,
        api_contract_id=api_contract_id,
        evidence_contract_id=evidence_contract_id,
        lease_fence=int(lease_fence),
    )


def request_digest_matches(message: pb.SubmitRequest, request: ExternalEvaluationRequest) -> bool:
    """Compare a submitted digest constant-time-ish without leaking bytes."""

    expected = request.request_digest
    supplied = message.request_digest
    return isinstance(supplied, str) and supplied == expected


def report_json(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise JobError("invalid_argument", f"report blob is not UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise JobError("invalid_argument", "report blob is not a JSON object")
    return value


def evidence_json(data: bytes | None) -> dict[str, Any] | None:
    if data is None:
        return None
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise JobError("invalid_argument", f"evidence blob is not UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise JobError("invalid_argument", "evidence blob is not a JSON object")
    return value
