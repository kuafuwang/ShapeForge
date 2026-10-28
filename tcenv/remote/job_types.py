"""Transport-neutral external evaluator job, attempt and evidence types."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from . import contracts

JOB_STATES = frozenset({
    "accepted", "queued", "running", "interrupted", "completed",
    "failed", "cancelled", "expired",
})
TERMINAL_JOB_STATES = frozenset({"completed", "failed", "cancelled", "expired"})
ATTEMPT_STATES = frozenset({"queued", "running", "interrupted", "completed", "failed", "cancelled"})
# Which execution supervisor started one attempt.  A frontend never guesses:
# recovery and release only clean the kind that durably created the attempt.
EXECUTION_PROCESS = "process"
EXECUTION_CONTAINER = "container"
EXECUTION_KINDS = frozenset({EXECUTION_PROCESS, EXECUTION_CONTAINER})
CANDIDATE_STATUSES = frozenset({
    "ok", "rejected", "compile_error", "abi_error", "unsafe", "incorrect",
    "runtime_error", "timeout", "crash", "unchanged", "duplicate",
    "external_duplicate", "refused_hang",
})
_ID = re.compile(r"[A-Za-z0-9._:-]{1,160}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class JobError(ValueError):
    """A sanitized transport-neutral RPC or durable-state error."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        failure_domain: str = "",
        request_id: str = "",
        attempt_id: str = "",
        missing_features: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message[:4000])
        self.code = code
        self.retryable = bool(retryable)
        self.failure_domain = failure_domain
        self.request_id = request_id
        self.attempt_id = attempt_id
        self.missing_features = tuple(missing_features)


class StateCorrupt(JobError):
    def __init__(self, message: str) -> None:
        super().__init__("data_loss", message, failure_domain="infrastructure")


class UntrustedSignature(StateCorrupt):
    """A stored result is signed by a key this frontend no longer trusts.

    Security revocation must not brick the whole frontend: the affected job is
    quarantined and its result is withheld, while other jobs keep serving.
    """


@dataclass(frozen=True)
class ExpectedEvaluator:
    site_id: str
    device_id: str
    target_id: str
    worker_build: str
    task_pack_digest: str
    evaluator_epoch: str

    def validate(self) -> None:
        for name in ("site_id", "device_id", "target_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _ID.fullmatch(value):
                raise JobError("invalid_argument", f"expected {name} is invalid")
        for name in ("worker_build",):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 256:
                raise JobError("invalid_argument", f"expected {name} is invalid")
        for name in ("task_pack_digest", "evaluator_epoch"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _HEX64.fullmatch(value):
                raise JobError("invalid_argument", f"expected {name} is invalid")

    def to_dict(self) -> dict[str, str]:
        return {
            "site_id": self.site_id,
            "device_id": self.device_id,
            "target_id": self.target_id,
            "worker_build": self.worker_build,
            "task_pack_digest": self.task_pack_digest,
            "evaluator_epoch": self.evaluator_epoch,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "ExpectedEvaluator":
        if not isinstance(document, Mapping) or set(document) != {
            "site_id", "device_id", "target_id", "worker_build", "task_pack_digest", "evaluator_epoch"
        }:
            raise JobError("invalid_argument", "expected evaluator fields are missing or unknown")
        value = cls(**dict(document))
        value.validate()
        return value


@dataclass(frozen=True)
class ExpectedIdentities:
    evaluator: ExpectedEvaluator
    api_contract_id: str = contracts.API_CONTRACT_ID
    evidence_contract_id: str = contracts.EVIDENCE_CONTRACT_ID

    def validate(self) -> None:
        self.evaluator.validate()
        try:
            contracts.dialect_for(self.api_contract_id, self.evidence_contract_id)
        except contracts.ContractError as exc:
            raise JobError("failed_precondition", "unsupported API/evidence contract pair") from exc

    @property
    def contract(self) -> contracts.ContractDialect:
        try:
            return contracts.dialect_for(self.api_contract_id, self.evidence_contract_id)
        except contracts.ContractError as exc:
            raise JobError("failed_precondition", "unsupported API/evidence contract pair") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator": self.evaluator.to_dict(),
            "api_contract_id": self.api_contract_id,
            "evidence_contract_id": self.evidence_contract_id,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "ExpectedIdentities":
        if not isinstance(document, Mapping) or set(document) != {
            "evaluator", "api_contract_id", "evidence_contract_id"
        }:
            raise JobError("invalid_argument", "expected identity fields are missing or unknown")
        value = cls(
            evaluator=ExpectedEvaluator.from_dict(document["evaluator"]),
            api_contract_id=document["api_contract_id"],
            evidence_contract_id=document["evidence_contract_id"],
        )
        value.validate()
        return value


@dataclass(frozen=True)
class CandidateDigest:
    """Candidate identity: ``sha256`` inline, ``sha256-blob-v1`` for a blob ref."""

    algorithm: str
    value: str
    #: The algorithms this module can construct.  The contract dialect decides
    #: which one a request may actually carry.
    allowed_algorithms: tuple[str, ...] = ("sha256", contracts.BLOB_DIGEST_ALGORITHM)

    def __post_init__(self) -> None:
        try:
            document = contracts.candidate_digest_document(
                {"algorithm": self.algorithm, "value": self.value},
                allowed_algorithms=self.allowed_algorithms,
            )
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc
        if document["algorithm"] != self.algorithm or document["value"] != self.value:
            raise JobError("invalid_argument", "candidate digest was not normalized")

    def to_dict(self) -> dict[str, str]:
        return {"algorithm": self.algorithm, "value": self.value}


@dataclass(frozen=True)
class BlobDescriptor:
    """Content-addressed reference to bytes stored in the blob channel.

    The descriptor is the only artifact identity that crosses the control plane:
    it never contains a path, a URL, or the bytes themselves.
    """

    content_id: str
    sha256: str
    size_bytes: int
    media_type: str
    chunk_size: int

    def __post_init__(self) -> None:
        try:
            normalized = contracts.blob_descriptor_document(self.to_dict())
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc
        if normalized != self.to_dict():
            raise JobError("invalid_argument", "blob descriptor was not normalized")

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_id": self.content_id,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "chunk_size": self.chunk_size,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "BlobDescriptor":
        if not isinstance(document, Mapping):
            raise JobError("invalid_argument", "blob descriptor must be an object")
        try:
            normalized = contracts.blob_descriptor_document(document)
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc
        return cls(**normalized)

    @classmethod
    def for_bytes(cls, data: bytes, *, media_type: str, chunk_size: int) -> "BlobDescriptor":
        """Build the descriptor a client would declare for exact bytes."""

        digest = hashlib.sha256(data).hexdigest()
        return cls(
            content_id=contracts.blob_content_id(digest), sha256=digest,
            size_bytes=len(data), media_type=media_type, chunk_size=chunk_size,
        )


@dataclass(frozen=True)
class DeviceGroupBinding:
    """Durable logical device group identity used by the NPU contract."""

    device_group_id: str
    vendor_id: str
    profile_digest: str
    topology_digest: str
    resource_generation: str

    def __post_init__(self) -> None:
        try:
            contracts.device_group_document(self.to_dict())
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc

    def to_dict(self) -> dict[str, str]:
        return {
            "device_group_id": self.device_group_id,
            "vendor_id": self.vendor_id,
            "profile_digest": self.profile_digest,
            "topology_digest": self.topology_digest,
            "resource_generation": self.resource_generation,
        }


@dataclass(frozen=True)
class NpuResources:
    """Declared logical resource requirements of one execution.

    Only *requirements* live here: logical device count and whether the task
    needs per-rank/per-die evidence.  Physical BDF, rank coordinates and fabric
    placement are decided by the trusted server profile and are not
    representable, so an agent cannot ask for a specific die or link.
    """

    logical_devices: int = 1
    required_rank_spans: int | None = None
    required_die_spans: int | None = None

    def __post_init__(self) -> None:
        try:
            contracts.device_group_requirements(self.to_dict())
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_devices": self.logical_devices,
            "required_rank_spans": self.required_rank_spans,
            "required_die_spans": self.required_die_spans,
        }


@dataclass(frozen=True)
class NpuExecutionSpec:
    """NPU execution shape.

    ``spec_version=1`` is the P0/P0b source/KERNEL/SIMT shape and serializes to
    exactly the seven original fields.  ``spec_version=4`` is the P1 shape and
    adds the execution schema id, the task family and the declared resources.
    ``spec_version=5`` keeps that document but adds the ``kernel_module`` task
    family (a prebuilt executable module launched as KERNEL/SIMT); the earlier
    contract versions reject that family, so the version is explicit here.
    """

    device_group: DeviceGroupBinding
    source_media_type: str = "text/plain; charset=utf-8"
    artifact_kind: str = "source"
    launch_kind: str = "KERNEL"
    launch_subtype: str = "SIMT"
    required_timing_sources: tuple[str, ...] = ("DEVICE_EVENT",)
    synchronization: str = "stream-event"
    # -- document version 4 ------------------------------------------------
    spec_version: int = 1
    task_family: str | None = None
    resources: NpuResources | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.device_group, DeviceGroupBinding):
            raise JobError("invalid_argument", "execution requires a typed DeviceGroupBinding")
        if self.spec_version not in (1, 4, 5):
            raise JobError("invalid_argument", "execution spec_version must be 1, 4 or 5")
        if self.spec_version == 5:
            if self.task_family != "kernel_module":
                raise JobError(
                    "invalid_argument", "execution spec_version 5 is the kernel_module build shape"
                )
        if self.spec_version >= 4:
            if self.task_family not in ("kernel", "kernel_module", "graph"):
                raise JobError(
                    "invalid_argument", "execution task_family must be kernel, kernel_module or graph"
                )
            if self.resources is None:
                object.__setattr__(self, "resources", NpuResources())
            elif not isinstance(self.resources, NpuResources):
                raise JobError("invalid_argument", "execution requires typed resource requirements")
        elif self.task_family is not None or self.resources is not None:
            raise JobError(
                "invalid_argument", "resource requirements belong to the version 4 execution spec"
            )
        try:
            self._normalized()
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc

    def _normalized(self) -> dict[str, Any]:
        if self.spec_version == 5:
            return contracts.npu_build_execution_document(
                {
                    "schema_id": contracts.NPU_BUNDLE_EXECUTION_SCHEMA_ID,
                    "task_family": self.task_family,
                    "artifact_kind": self.artifact_kind,
                    "launch_kind": self.launch_kind,
                    "launch_subtype": self.launch_subtype,
                    "source_media_type": self.source_media_type,
                    "device_group": self.device_group.to_dict(),
                    "resources": self.resources.to_dict(),
                    "required_timing_sources": list(self.required_timing_sources),
                    "synchronization": self.synchronization,
                }
            )
        if self.spec_version == 4:
            return contracts.npu_bundle_execution_document(
                {
                    "schema_id": contracts.NPU_BUNDLE_EXECUTION_SCHEMA_ID,
                    "task_family": self.task_family,
                    "artifact_kind": self.artifact_kind,
                    "launch_kind": self.launch_kind,
                    "launch_subtype": self.launch_subtype,
                    "source_media_type": self.source_media_type,
                    "device_group": self.device_group.to_dict(),
                    "resources": self.resources.to_dict(),
                    "required_timing_sources": list(self.required_timing_sources),
                    "synchronization": self.synchronization,
                }
            )
        return contracts.npu_execution_document(self._p0_dict())

    def _p0_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "launch_kind": self.launch_kind,
            "launch_subtype": self.launch_subtype,
            "source_media_type": self.source_media_type,
            "device_group": self.device_group.to_dict(),
            "required_timing_sources": list(self.required_timing_sources),
            "synchronization": self.synchronization,
        }

    @property
    def version(self) -> int:
        return self.spec_version

    @property
    def declared_resources(self) -> NpuResources:
        return self.resources or NpuResources()

    def to_dict(self) -> dict[str, Any]:
        if self.spec_version < 4:
            return self._p0_dict()
        return {
            "schema_id": contracts.NPU_BUNDLE_EXECUTION_SCHEMA_ID,
            "task_family": self.task_family,
            "artifact_kind": self.artifact_kind,
            "launch_kind": self.launch_kind,
            "launch_subtype": self.launch_subtype,
            "source_media_type": self.source_media_type,
            "device_group": self.device_group.to_dict(),
            "resources": self.resources.to_dict() if self.resources else None,
            "required_timing_sources": list(self.required_timing_sources),
            "synchronization": self.synchronization,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "NpuExecutionSpec":
        if not isinstance(document, Mapping):
            raise JobError("invalid_argument", "execution spec must be an object")
        version = 4 if "schema_id" in document else 1
        if version == 4 and document.get("task_family") == "kernel_module":
            version = 5
        try:
            if version == 5:
                normalized = contracts.npu_build_execution_document(document)
            elif version == 4:
                normalized = contracts.npu_bundle_execution_document(document)
            else:
                normalized = contracts.npu_execution_document(document)
            group = DeviceGroupBinding(**normalized["device_group"])
            resources = None
            if version >= 4:
                resources = NpuResources(**normalized["resources"])
            return cls(
                device_group=group,
                source_media_type=normalized["source_media_type"],
                artifact_kind=normalized["artifact_kind"],
                launch_kind=normalized["launch_kind"],
                launch_subtype=normalized["launch_subtype"],
                required_timing_sources=tuple(normalized["required_timing_sources"]),
                synchronization=normalized["synchronization"],
                spec_version=version,
                task_family=normalized["task_family"] if version >= 4 else None,
                resources=resources,
            )
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc


@dataclass(frozen=True)
class NpuSourceRequest:
    """NPU request payload: CandidateDigest plus one artifact branch.

    Three artifact forms exist across the NPU contract versions and exactly one
    is set at a time:

    * ``blob`` — a content-addressed reference (``npu-blob``);
    * ``bundle`` — a versioned artifact set (``npu-bundle``, document version 4);
    * neither — the exact inline source bytes (``npu-source``, or the source
      branch of the bundle contract).

    The contract identity decides which forms are admissible; the payload never
    selects a contract by its own shape.
    """

    candidate_digest: CandidateDigest
    execution: NpuExecutionSpec
    blob: BlobDescriptor | None = None
    bundle: "ArtifactBundle | None" = None

    def __post_init__(self) -> None:
        from .artifact_bundle import ArtifactBundle

        if not isinstance(self.candidate_digest, CandidateDigest):
            raise JobError("invalid_argument", "NPU request requires a typed CandidateDigest")
        if not isinstance(self.execution, NpuExecutionSpec):
            raise JobError("invalid_argument", "NPU request requires a typed execution spec")
        if self.blob is not None and self.bundle is not None:
            raise JobError("invalid_argument", "an NPU request carries exactly one artifact branch")
        if self.bundle is not None:
            if not isinstance(self.bundle, ArtifactBundle):
                raise JobError("invalid_argument", "NPU artifact bundle must be typed")
            if self.candidate_digest.algorithm != contracts.BUNDLE_DIGEST_ALGORITHM:
                raise JobError(
                    "invalid_argument", "a bundle candidate requires the sha256-bundle-v1 digest algorithm"
                )
            if self.candidate_digest.value != self.bundle.artifact_set_digest:
                raise JobError(
                    "invalid_argument", "a bundle candidate digest must equal artifact_set_digest"
                )
            if self.execution.artifact_kind != "bundle":
                raise JobError("invalid_argument", "a bundle requires artifact_kind=bundle")
        elif self.blob is not None:
            if not isinstance(self.blob, BlobDescriptor):
                raise JobError("invalid_argument", "NPU blob reference must be typed")
            if self.candidate_digest.algorithm != contracts.BLOB_DIGEST_ALGORITHM:
                raise JobError("invalid_argument", "a blob candidate requires the sha256-blob-v1 digest algorithm")
            if self.candidate_digest.value != self.blob.sha256:
                raise JobError("invalid_argument", "a blob candidate digest must equal the blob descriptor digest")
            if self.execution.source_media_type != self.blob.media_type:
                raise JobError("invalid_argument", "blob media type differs from the execution media type")
        else:
            if self.candidate_digest.algorithm != "sha256":
                raise JobError("invalid_argument", "an inline candidate requires the sha256 digest algorithm")
            if self.execution.artifact_kind != "source":
                raise JobError("invalid_argument", "an inline candidate requires artifact_kind=source")

    @property
    def artifact_form(self) -> str:
        if self.bundle is not None:
            return "bundle"
        return "blob" if self.blob is not None else "source"

    def blob_entries(self) -> tuple[Any, ...]:
        """Blob-backed bundle entries (empty for a single-blob request)."""

        return () if self.bundle is None else self.bundle.blob_entries

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "candidate_digest": self.candidate_digest.to_dict(),
            "execution": self.execution.to_dict(),
        }
        if self.blob is not None:
            document["blob_ref"] = self.blob.to_dict()
        if self.bundle is not None:
            document["artifact"] = {"bundle": self.bundle.to_dict()}
        return document

    @classmethod
    def from_dict(
        cls, document: Mapping[str, Any], *, dialect: "contracts.ContractDialect | None" = None,
    ) -> "NpuSourceRequest":
        from .artifact_bundle import ArtifactBundle, BundleError

        if not isinstance(document, Mapping):
            raise JobError("invalid_argument", "NPU request fields are missing or unknown")
        expected = {"candidate_digest", "execution"}
        if "blob_ref" in document:
            expected |= {"blob_ref"}
        if "artifact" in document:
            expected |= {"artifact"}
        if set(document) != expected:
            raise JobError("invalid_argument", "NPU request fields are missing or unknown")
        if "blob_ref" in document and "artifact" in document:
            raise JobError("invalid_argument", "an NPU request carries exactly one artifact branch")
        blob = None if "blob_ref" not in document else BlobDescriptor.from_dict(document["blob_ref"])
        bundle = None
        artifact = document.get("artifact")
        if artifact is not None:
            if not isinstance(artifact, Mapping) or len(artifact) != 1:
                raise JobError("invalid_argument", "artifact must carry exactly one branch")
            branch = next(iter(artifact))
            if branch == "bundle":
                try:
                    bundle = ArtifactBundle.from_document(artifact[branch])
                except BundleError as exc:
                    raise JobError("invalid_argument", str(exc)) from exc
            elif branch == "source":
                descriptor = artifact[branch]
                if not isinstance(descriptor, Mapping) or set(descriptor) != {"size_bytes", "sha256"}:
                    raise JobError(
                        "invalid_argument", "the source artifact must carry size_bytes and sha256"
                    )
            else:
                raise JobError("invalid_argument", "artifact must carry exactly one branch")
        if bundle is not None:
            allowed = (contracts.BUNDLE_DIGEST_ALGORITHM,)
        elif blob is not None:
            allowed = (contracts.BLOB_DIGEST_ALGORITHM,)
        else:
            allowed = ("sha256",)
        try:
            digest = contracts.candidate_digest_document(
                document["candidate_digest"], allowed_algorithms=allowed,
            )
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc
        if artifact is not None and "source" in artifact:
            descriptor = artifact["source"]
            if (
                descriptor.get("sha256") != digest["value"]
                or not isinstance(descriptor.get("size_bytes"), int)
                or isinstance(descriptor.get("size_bytes"), bool)
                or descriptor["size_bytes"] < 1
            ):
                raise JobError(
                    "invalid_argument", "the source artifact does not describe the CandidateDigest"
                )
        return cls(
            candidate_digest=CandidateDigest(digest["algorithm"], digest["value"], allowed_algorithms=allowed),
            execution=NpuExecutionSpec.from_dict(document["execution"]),
            blob=blob,
            bundle=bundle,
        )


@dataclass(frozen=True)
class NpuBuildRequest:
    """The request-level BuildProvenance identity (P1c / document version 5).

    A ``prebuilt`` row names the artifact set the control host built and the
    exact provenance it produced; an ``evaluator`` row names the registered
    build profile this attempt must use and leaves the artifact digest unset.
    Only a registered profile can run it: the fields here are an expectation to
    verify, never an authorization to run an arbitrary command.
    """

    mode: str
    builder_id: str
    builder_version: str
    build_host: str
    build_profile: str
    toolchain: Mapping[str, str]
    compile_profile: Mapping[str, Any]
    source_sha256: str
    artifact_set_digest: str | None = None
    #: The producer build-evidence link a prebuilt request carries (status,
    #: duration, log identity and the digest of the control host's evidence
    #: document).  ``None`` for an evaluator build, which has no producer.
    producer: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        try:
            contracts.build_request_document(self.to_dict())
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", f"build request is invalid: {exc}") from exc

    def provenance_document(self) -> dict[str, Any]:
        return {
            "schema_id": contracts.NPU_BUILD_PROVENANCE_SCHEMA_ID,
            "builder_id": self.builder_id,
            "builder_version": self.builder_version,
            "build_host": self.build_host,
            "build_profile": self.build_profile,
            "toolchain": dict(self.toolchain),
            "compile_profile": dict(self.compile_profile),
            "source_sha256": self.source_sha256,
            "artifact_set_digest": self.artifact_set_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        document = {
            "schema_id": contracts.NPU_BUILD_REQUEST_SCHEMA_ID,
            "mode": self.mode,
            "provenance": self.provenance_document(),
        }
        if self.producer is not None:
            document["producer"] = dict(self.producer)
        return document

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "NpuBuildRequest":
        try:
            normalized = contracts.build_request_document(document)
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", f"build request is invalid: {exc}") from exc
        provenance = normalized["provenance"]
        return cls(
            mode=normalized["mode"],
            builder_id=provenance["builder_id"],
            builder_version=provenance["builder_version"],
            build_host=provenance["build_host"],
            build_profile=provenance["build_profile"],
            toolchain=provenance["toolchain"],
            compile_profile=provenance["compile_profile"],
            source_sha256=provenance["source_sha256"],
            artifact_set_digest=provenance["artifact_set_digest"],
            producer=normalized.get("producer"),
        )


@dataclass(frozen=True)
class ExternalEvaluationRequest:
    """The validated v1 execution request with exact candidate bytes."""

    kind: str
    request_id: str
    run_id: str
    session_id: str
    task: str
    dtype: str
    size: str
    target_id: str
    task_fingerprint: str
    candidate_sha256: str
    source: bytes
    mode: str
    seed: str | None
    reps: int
    queue_timeout_ms: int
    candidate_timeout_ms: int
    cases: tuple[str, ...] | None
    fast_bench: bool = False
    require_contract_launch: bool = False
    validate: bool = False
    measure_serial: bool = False
    cache_policy: str = "reuse"
    npu: NpuSourceRequest | None = None
    #: The contract version this request is meant for.  The bundle contract has
    #: two artifact branches (inline source or bundle) under one identity, so the
    #: version cannot be inferred from the payload; the server still rejects a
    #: request whose dialect differs from its own configured one.
    contract_dialect: contracts.ContractDialect | None = None
    #: Independent BuildProvenance identity for the P1c build contract.  It is
    #: part of the canonical request digest, the durable row and the resume
    #: identity, but never part of ``artifact_set_digest``.
    build: NpuBuildRequest | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not _ID.fullmatch(self.request_id):
            raise JobError("invalid_argument", "request_id must be a bounded opaque identifier")
        if not isinstance(self.cases, tuple) and self.cases is not None:
            raise JobError("invalid_argument", "cases must be a tuple or null")
        # The two contracts never share a candidate identity: the NPU request
        # carries CandidateDigest and must not dual-write the legacy field.
        if self.npu is not None and self.candidate_sha256:
            raise JobError(
                "invalid_argument",
                "the NPU contract must not also carry the legacy candidate_sha256",
            )
        # A blob candidate is referenced, never embedded: the request payload
        # must stay metadata-sized even though the artifact may be huge.
        if self.npu is not None and self.npu.blob is not None and self.source:
            raise JobError(
                "invalid_argument",
                "a blob candidate must not also carry inline source bytes",
            )
        if self.npu is not None and self.npu.bundle is not None and self.source:
            raise JobError(
                "invalid_argument",
                "a bundle candidate must not also carry inline source bytes",
            )
        if self.npu is None:
            if self.contract_dialect is not None and self.contract_dialect.npu_source:
                raise JobError(
                    "invalid_argument", "a legacy request cannot claim an NPU contract version"
                )
        else:
            dialect = self.contract
            if not dialect.npu_source:
                raise JobError(
                    "invalid_argument", "an NPU request must name an NPU contract version"
                )
            form = self.npu.artifact_form
            if form == "bundle" and not dialect.npu_bundle:
                raise JobError(
                    "invalid_argument", "a bundle request requires the bundle contract version"
                )
            if form == "blob" and not dialect.npu_blob:
                raise JobError(
                    "invalid_argument", "a blob request requires the blob contract version"
                )
            if form == "source" and dialect.npu_blob:
                raise JobError(
                    "invalid_argument", "the blob contract does not accept an inline source candidate"
                )
            if dialect.build_contract:
                if self.build is None:
                    raise JobError(
                        "invalid_argument", "the build contract requires a BuildProvenance identity"
                    )
                if not isinstance(self.build, NpuBuildRequest):
                    raise JobError("invalid_argument", "build provenance must be typed")
            elif self.build is not None:
                raise JobError(
                    "invalid_argument", "build provenance belongs to the build contract version"
                )
        # This performs all source, enum, integer, UTF-8 and explicit-default checks.
        self.digest_document()

    def digest_fields(self) -> dict[str, Any]:
        common = {
            "kind": self.kind,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "task": self.task,
            "dtype": self.dtype,
            "size": self.size,
            "target_id": self.target_id,
            "task_fingerprint": self.task_fingerprint,
            "mode": self.mode,
            "seed": self.seed,
            "reps": self.reps,
            "queue_timeout_ms": self.queue_timeout_ms,
            "candidate_timeout_ms": self.candidate_timeout_ms,
            "cases": None if self.cases is None else list(self.cases),
            "fast_bench": self.fast_bench,
            "require_contract_launch": self.require_contract_launch,
            "validate": self.validate,
            "measure_serial": self.measure_serial,
            "cache_policy": self.cache_policy,
        }
        if self.npu is None:
            return {"candidate_sha256": self.candidate_sha256, **common}
        fields: dict[str, Any] = {
            "candidate_digest": self.npu.candidate_digest.to_dict(),
            "execution": self.npu.execution.to_dict(),
            **common,
        }
        if self.npu.blob is not None:
            fields["blob_ref"] = self.npu.blob.to_dict()
        if self.contract.npu_bundle:
            fields["artifact"] = self.artifact_document()
        if self.contract.build_contract:
            if self.build is None:
                raise JobError("invalid_argument", "the build contract requires a BuildProvenance identity")
            fields["build"] = self.build.to_dict()
        return fields

    def artifact_document(self) -> dict[str, Any]:
        """The bundle contract's source/bundle oneof document.

        The inline branch carries only the exact size and SHA-256; the bundle
        branch carries the manifest digest and the per-entry manifest.  The
        bytes themselves never enter a digest document.
        """

        if self.npu is None or not self.contract.npu_bundle:
            raise JobError("invalid_argument", "only the bundle contract has an artifact oneof")
        if self.npu.bundle is not None:
            return {"bundle": self.npu.bundle.to_dict()}
        digest = hashlib.sha256(self.source).hexdigest()
        return {"source": {"size_bytes": len(self.source), "sha256": digest}}

    @property
    def contract(self) -> contracts.ContractDialect:
        if self.npu is None:
            return contracts.LEGACY_DIALECT
        if self.contract_dialect is not None:
            return self.contract_dialect
        if self.npu.bundle is not None:
            return contracts.NPU_BUNDLE_DIALECT
        return contracts.NPU_BLOB_DIALECT if self.npu.blob is not None else contracts.NPU_SOURCE_DIALECT

    @property
    def blob(self) -> BlobDescriptor | None:
        return None if self.npu is None else self.npu.blob

    @property
    def artifact_bytes(self) -> bytes:
        """Exact bytes this job persists for its candidate.

        The persisted blob is the inline source for a source candidate and the
        canonical entry payload for an artifact set; both are deleted together
        with the job at a terminal state.  A blob candidate embeds nothing.
        """

        if self.npu is not None and self.npu.bundle is not None:
            from .artifact_bundle import payload_bytes

            return payload_bytes(self.npu.bundle)
        return self.source

    def digest_document(self) -> dict[str, Any]:
        try:
            return contracts.request_digest_document(
                self.digest_fields(), source_bytes=self.source, dialect=self.contract,
            )
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc

    @property
    def request_digest(self) -> str:
        try:
            return contracts.request_digest(
                self.digest_fields(), source_bytes=self.source, dialect=self.contract,
            )
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc

    @property
    def candidate_digest(self) -> dict[str, str]:
        if self.npu is not None:
            return self.npu.candidate_digest.to_dict()
        return {"algorithm": "sha256", "value": self.candidate_sha256}

    @property
    def seed_int(self) -> int | None:
        return None if self.seed is None else int(self.seed)

    def without_source(self) -> dict[str, Any]:
        return {"request_id": self.request_id, **self.digest_fields()}

    def to_worker_dict(self) -> dict[str, Any]:
        """Private worker representation; base64 preserves arbitrary bytes.

        A blob candidate carries no bytes at all: the worker gets the trusted
        content id and reads the store itself, so a large artifact never becomes
        one base64 worker frame.  A bundle carries only its inline entries (the
        ones under the inline limit); blob-backed entries stay references.
        """

        document = {**self.without_source(), "source_base64": base64.b64encode(self.source).decode("ascii")}
        # The artifact oneof document is derived, not transported: a bundle goes
        # as typed entries (with their exact inline bytes), an inline source as
        # the exact bytes below.  Carrying both would let two identities drift.
        document.pop("artifact", None)
        if self.npu is not None and self.npu.bundle is not None:
            document["bundle"] = self.npu.bundle.to_worker_dict()
        if self.build is not None:
            document["build"] = self.build.to_dict()
        return document

    @classmethod
    def from_worker_dict(cls, document: Mapping[str, Any]) -> "ExternalEvaluationRequest":
        if not isinstance(document, Mapping) or "source_base64" not in document:
            raise JobError("invalid_argument", "worker request has no exact source bytes")
        try:
            source = base64.b64decode(document["source_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise JobError("invalid_argument", "worker source_base64 is invalid") from exc
        if "execution" in document or "candidate_digest" in document:
            from .artifact_bundle import ArtifactBundle, BundleError

            extra = {key for key in ("blob_ref", "bundle", "build") if key in document}
            allowed = (set(cls.without_source_fields()) - {"candidate_sha256"}) | {
                "candidate_digest", "execution", "source_base64",
            } | extra
            if set(document) != allowed:
                raise JobError("invalid_argument", "worker NPU request fields are missing or unknown")
            fields = {
                key: value for key, value in document.items()
                if key not in {
                    "source_base64", "candidate_digest", "execution", "blob_ref", "bundle", "build",
                }
            }
            if fields["cases"] is not None:
                fields["cases"] = tuple(fields["cases"])
            execution = NpuExecutionSpec.from_dict(document["execution"])
            build = (
                NpuBuildRequest.from_dict(document["build"]) if "build" in document else None
            )
            bundle = None
            if "bundle" in document:
                if execution.version < 4 or execution.artifact_kind != "bundle":
                    raise JobError(
                        "invalid_argument", "an artifact bundle requires the version 4 bundle execution spec"
                    )
                try:
                    bundle = ArtifactBundle.from_worker_dict(document["bundle"])
                except BundleError as exc:
                    raise JobError("invalid_argument", str(exc)) from exc
                if source:
                    raise JobError("invalid_argument", "worker bundle request must not carry source bytes")
            if "blob_ref" in document:
                npu = NpuSourceRequest.from_dict({
                    "candidate_digest": document["candidate_digest"],
                    "execution": document["execution"],
                    "blob_ref": document["blob_ref"],
                })
                if source:
                    raise JobError("invalid_argument", "worker blob request must not carry source bytes")
                return cls(
                    **fields, source=source, candidate_sha256="", npu=npu,
                    contract_dialect=contracts.NPU_BLOB_DIALECT,
                )
            if bundle is not None:
                allowed_digest = (contracts.BUNDLE_DIGEST_ALGORITHM,)
            else:
                allowed_digest = ("sha256",)
            try:
                digest = contracts.candidate_digest_document(
                    document["candidate_digest"], allowed_algorithms=allowed_digest,
                )
            except contracts.ContractError as exc:
                raise JobError("invalid_argument", str(exc)) from exc
            npu = NpuSourceRequest(
                candidate_digest=CandidateDigest(
                    digest["algorithm"], digest["value"], allowed_algorithms=allowed_digest,
                ),
                execution=execution,
                bundle=bundle,
            )
            if bundle is None and npu.candidate_digest.value != hashlib.sha256(source).hexdigest():
                raise JobError("invalid_argument", "worker NPU candidate digest does not match the source")
            if build is not None:
                dialect = contracts.NPU_BUILD_DIALECT
            elif execution.version >= 4:
                dialect = contracts.NPU_BUNDLE_DIALECT
            else:
                dialect = contracts.NPU_SOURCE_DIALECT
            return cls(
                **fields, source=source, candidate_sha256="", npu=npu, contract_dialect=dialect,
                build=build,
            )
        allowed = set(cls.without_source_fields()) | {"source_base64"}
        if set(document) != allowed:
            raise JobError("invalid_argument", "worker request fields are missing or unknown")
        fields = {key: value for key, value in document.items() if key != "source_base64"}
        if fields["cases"] is not None:
            fields["cases"] = tuple(fields["cases"])
        return cls(**fields, source=source)

    @staticmethod
    def without_source_fields() -> tuple[str, ...]:
        return (
            "request_id", "kind", "run_id", "session_id", "task", "dtype", "size",
            "target_id", "task_fingerprint", "candidate_sha256", "mode", "seed", "reps",
            "queue_timeout_ms", "candidate_timeout_ms", "cases", "fast_bench",
            "require_contract_launch", "validate", "measure_serial", "cache_policy",
        )

    @classmethod
    def from_stored(cls, document: Mapping[str, Any], source: bytes) -> "ExternalEvaluationRequest":
        fields = dict(document)
        if fields.get("cases") is not None:
            fields["cases"] = tuple(fields["cases"])
        if "execution" in fields or "candidate_digest" in fields:
            if "execution" not in fields or "candidate_digest" not in fields:
                raise JobError("invalid_argument", "stored NPU request is incomplete")
            if "candidate_sha256" in fields:
                # The two candidate identities never coexist.  Dropping the
                # legacy field here would silently accept a mixed request, so it
                # is a stored-state corruption instead.
                raise JobError("invalid_argument", "stored NPU request also carries candidate_sha256")
            extra = {key for key in ("blob_ref", "artifact", "build") if key in fields}
            allowed = (set(cls.without_source_fields()) - {"candidate_sha256"}) | {
                "candidate_digest", "execution",
            } | extra
            if set(fields) != allowed:
                raise JobError("invalid_argument", "stored NPU request fields are missing or unknown")
            build = (
                NpuBuildRequest.from_dict(fields.pop("build")) if "build" in fields else None
            )
            npu_document = {
                "candidate_digest": fields.pop("candidate_digest"),
                "execution": fields.pop("execution"),
            }
            if "blob_ref" in fields:
                npu_document["blob_ref"] = fields.pop("blob_ref")
            if "artifact" in fields:
                npu_document["artifact"] = fields.pop("artifact")
            npu = NpuSourceRequest.from_dict(npu_document)
            request_source = source
            if npu.blob is not None:
                if source:
                    raise JobError("invalid_argument", "stored blob request carries unexpected source bytes")
            elif npu.bundle is not None:
                from .artifact_bundle import BundleError, payload_bundle

                if not source:
                    raise JobError("invalid_argument", "stored bundle request carries no artifact payload")
                try:
                    stored = payload_bundle(source)
                except BundleError as exc:
                    raise JobError("invalid_argument", f"stored bundle payload is invalid: {exc}") from exc
                if stored.to_dict() != npu.bundle.to_dict():
                    raise JobError(
                        "invalid_argument", "stored bundle payload does not match the accepted artifact set"
                    )
                npu = NpuSourceRequest(
                    candidate_digest=npu.candidate_digest, execution=npu.execution,
                    blob=npu.blob, bundle=stored,
                )
                # The payload was the persisted artifact container, not inline
                # source bytes: the typed bundle now owns them.
                request_source = b""
            elif npu.candidate_digest.value != hashlib.sha256(source).hexdigest():
                raise JobError("invalid_argument", "stored NPU candidate digest does not match the source")
            if npu.blob is not None:
                dialect = contracts.NPU_BLOB_DIALECT
            elif build is not None:
                dialect = contracts.NPU_BUILD_DIALECT
            elif npu.execution.version >= 4:
                dialect = contracts.NPU_BUNDLE_DIALECT
            else:
                dialect = contracts.NPU_SOURCE_DIALECT
            return cls(
                **fields, source=request_source, candidate_sha256="", npu=npu,
                contract_dialect=dialect, build=build,
            )
        return cls(**fields, source=source)

    @classmethod
    def from_legacy(
        cls, request: Any, *, npu: NpuSourceRequest | None = None,
    ) -> "ExternalEvaluationRequest":
        """Adapt ``tcenv.evaluation.EvaluationRequest`` without changing its digest."""

        from tcenv.evaluation import EvaluationRequest

        if not isinstance(request, EvaluationRequest):
            raise TypeError("request must be an EvaluationRequest")
        if not request.task_fingerprint:
            raise JobError("invalid_argument", "external requests require task_fingerprint")
        try:
            queue_ms = contracts.seconds_to_timeout_ms(request.queue_timeout_s)
            candidate_ms = contracts.seconds_to_timeout_ms(request.candidate_timeout_s)
        except contracts.ContractError as exc:
            raise JobError("invalid_argument", str(exc)) from exc
        return cls(
            kind=request.kind,
            request_id=request.request_id,
            run_id=request.run_id,
            session_id=request.session_id,
            task=request.task,
            dtype=request.dtype,
            size=request.size,
            target_id=request.target_id,
            task_fingerprint=request.task_fingerprint,
            candidate_sha256="" if npu is not None else request.candidate_sha256,
            source=request.source_bytes,
            mode=request.mode,
            seed=None if request.seed is None else str(request.seed),
            reps=request.reps,
            queue_timeout_ms=queue_ms,
            candidate_timeout_ms=candidate_ms,
            cases=request.cases,
            fast_bench=request.fast_bench,
            require_contract_launch=request.require_contract_launch,
            validate=request.validate,
            measure_serial=request.measure_serial,
            cache_policy=request.cache_policy,
            npu=npu,
        )

    def to_legacy(self) -> Any:
        """Adapt to the existing private worker/evaluator request when needed."""

        if self.npu is not None:
            # The NPU contract has no legacy evaluator mapping: silently
            # downgrading it would drop CandidateDigest, the device group and
            # the device-side timing evidence.
            raise JobError("invalid_argument", "the NPU contract has no legacy request mapping")
        from tcenv.evaluation import EvaluationRequest

        try:
            source = self.source.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise JobError("invalid_argument", "candidate source is not supported UTF-8 text") from exc
        return EvaluationRequest(
            kind=self.kind,
            request_id=self.request_id,
            run_id=self.run_id,
            session_id=self.session_id,
            task=self.task,
            dtype=self.dtype,
            size=self.size,
            target_id=self.target_id,
            task_fingerprint=self.task_fingerprint,
            candidate_sha256=self.candidate_sha256,
            source=source,
            mode=self.mode,
            seed=self.seed_int,
            reps=self.reps,
            queue_timeout_s=self.queue_timeout_ms / 1000.0,
            candidate_timeout_s=self.candidate_timeout_ms / 1000.0,
            cases=self.cases,
            fast_bench=self.fast_bench,
            require_contract_launch=self.require_contract_launch,
            validate=self.validate,
            measure_serial=self.measure_serial,
            cache_policy=self.cache_policy,
        )


@dataclass(frozen=True)
class JobFailure:
    domain: str
    retryable: bool
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "retryable": self.retryable,
            "code": self.code,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any] | None) -> "JobFailure | None":
        if document is None:
            return None
        return cls(
            domain=str(document["domain"]),
            retryable=bool(document["retryable"]),
            code=str(document["code"]),
            message=str(document["message"]),
        )


@dataclass(frozen=True)
class Attempt:
    attempt_id: str
    lease_fence: int
    state: str
    retry_index: int
    cancel_requested: bool
    queued_at_ms: int
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    failure: JobFailure | None = None
    session_sequence: int = 0
    worker_pid: int | None = None
    worker_pgid: int | None = None
    worker_start_token: str = ""
    frontend_instance: str = ""
    execution_kind: str = EXECUTION_PROCESS
    container_id: str = ""
    container_name: str = ""
    container_image_ref: str = ""
    container_image_digest: str = ""
    container_runtime: str = ""
    container_daemon: str = ""
    # The logical device group/profile/topology/resource generation this lease
    # was accepted under.  Empty for the legacy Metal/source contract.
    device_binding: Mapping[str, str] | None = None


@dataclass(frozen=True)
class JobResult:
    report: bytes
    report_descriptor: dict[str, Any]
    evidence: bytes | None
    evidence_descriptor: dict[str, Any] | None
    provenance: dict[str, Any]
    receipt: dict[str, Any]
    signature: dict[str, Any] | None = None

    def report_json(self) -> dict[str, Any]:
        value = json.loads(self.report.decode("utf-8"))
        if not isinstance(value, dict):
            raise StateCorrupt("stored report is not an object")
        return value

    def evidence_json(self) -> dict[str, Any] | None:
        if self.evidence is None:
            return None
        value = json.loads(self.evidence.decode("utf-8"))
        if not isinstance(value, dict):
            raise StateCorrupt("stored evidence is not an object")
        return value


@dataclass(frozen=True)
class JobSnapshot:
    request_id: str
    request_digest: str
    state: str
    provenance: dict[str, Any]
    current_attempt: Attempt | None
    result: JobResult | None
    failure: JobFailure | None
    cancel_requested: bool
    last_sequence: int
    accepted_at_ms: int
    updated_at_ms: int
    expires_at_ms: int | None
    quarantined: bool = False
    recovery_error: str = ""
    # The evaluator identities the job was accepted under; a frontend must not
    # execute/sign it under any other build or epoch.
    expected: ExpectedIdentities | None = None


@dataclass(frozen=True)
class JobEvent:
    request_id: str
    attempt_id: str | None
    lease_fence: int
    sequence: int
    progress: dict[str, Any] | None = None
    snapshot: JobSnapshot | None = None


@dataclass(frozen=True)
class Capabilities:
    client_nonce: str
    server_nonce: str
    capability_id: str
    expected: ExpectedIdentities
    provenance: dict[str, Any]
    supported_features: frozenset[str]
    required_features: frozenset[str]
    issued_at_ms: int
    valid_for_ms: int
    limits: dict[str, int] = field(default_factory=dict)
    target_capabilities: tuple[dict[str, Any], ...] = ()
    signing_key_id: str = ""


@dataclass(frozen=True)
class QuotaUsage:
    reserved_count: int
    reserved_gpu_us: int
    used_count: int
    used_gpu_us: int
    hang: bool
