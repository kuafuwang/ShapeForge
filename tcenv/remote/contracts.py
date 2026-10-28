"""Transport-neutral Phase 2C1 external evaluator evidence contracts.

This module deliberately uses only the Python standard library.  It does not
import Protobuf, gRPC, or the legacy evaluator, and it never hashes serialized
Protobuf messages.  The legacy algorithms in :mod:`tcenv.evaluation` are
unchanged; these functions implement only the new ``tcenv.remote.v1`` contract.

Canonical documents use the RFC 8785 subset needed by the protocol: ASCII field
names, Unicode string values, null/bool, safe integers, arrays, and objects.
Execution timeouts have already been normalized to integer milliseconds before
a request document is built.  Candidate source bytes are hashed separately and
never copied into the request digest document.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from typing import Any, Mapping

API_CONTRACT_ID = "tcenv.remote.v1.evaluator-api.1"
EVIDENCE_CONTRACT_ID = "tcenv.remote.v1.evidence.1"
# P0 NPU contract: source-only artifact, public KERNEL (SIMT profile subtype),
# one logical device group and DEVICE_EVENT timing.  It shares the six RPCs and
# the durable job/attempt/fence machinery with the legacy Metal/source
# contract, but has its own identity, canonical document types and receipt.
NPU_SOURCE_API_CONTRACT_ID = "tcenv.remote.npu-source.evaluator-api.1"
NPU_SOURCE_EVIDENCE_CONTRACT_ID = "tcenv.remote.npu-source.evidence.1"
NPU_SOURCE_REQUEST_DOCUMENT_TYPE = "tcenv.remote.npu-source.request-digest"
NPU_SOURCE_RECEIPT_DOCUMENT_TYPE = "tcenv.remote.npu-source.receipt-id"
NPU_SOURCE_RESULT_DOCUMENT_TYPE = "tcenv.remote.npu-source.result-signature"
NPU_SOURCE_DOCUMENT_VERSION = 2
# P0b NPU contract: same source/KERNEL/SIMT/DEVICE_EVENT shape as P0, but the
# candidate bytes live in the independent content-addressed blob channel.  It is
# a later version of the NPU contract, not a new field on `npu-source` and not a
# field on the legacy Metal/source contract.
NPU_BLOB_API_CONTRACT_ID = "tcenv.remote.npu-blob.evaluator-api.1"
NPU_BLOB_EVIDENCE_CONTRACT_ID = "tcenv.remote.npu-blob.evidence.1"
NPU_BLOB_REQUEST_DOCUMENT_TYPE = "tcenv.remote.npu-blob.request-digest"
NPU_BLOB_RECEIPT_DOCUMENT_TYPE = "tcenv.remote.npu-blob.receipt-id"
NPU_BLOB_RESULT_DOCUMENT_TYPE = "tcenv.remote.npu-blob.result-signature"
NPU_BLOB_DOCUMENT_VERSION = 3
#: Content id prefix: the digest algorithm is tagged so a content id can never
#: be confused with a bare hex digest from another algorithm.
BLOB_CONTENT_ID_PREFIX = "blob-sha256-v1:"
#: CandidateDigest algorithm used by the blob-reference contract.
BLOB_DIGEST_ALGORITHM = "sha256-blob-v1"
MAX_BLOB_CHUNK_BYTES = 4 * 1024 * 1024
# P1 NPU contract: the same six RPCs and durable machinery, with a versioned
# ArtifactBundle artifact, an explicit source/bundle artifact oneof and a
# complete ExecutionSpec (task family, launch kind, logical device group,
# declared resources, synchronization and timing requirements).  The bundle is
# part of *this* document version; it is not an optional field on
# `npu-source`/`npu-blob` and not a field on the legacy Metal/source contract.
NPU_BUNDLE_API_CONTRACT_ID = "tcenv.remote.npu-bundle.evaluator-api.1"
NPU_BUNDLE_EVIDENCE_CONTRACT_ID = "tcenv.remote.npu-bundle.evidence.1"
NPU_BUNDLE_REQUEST_DOCUMENT_TYPE = "tcenv.remote.npu-bundle.request-digest"
NPU_BUNDLE_RECEIPT_DOCUMENT_TYPE = "tcenv.remote.npu-bundle.receipt-id"
NPU_BUNDLE_RESULT_DOCUMENT_TYPE = "tcenv.remote.npu-bundle.result-signature"
NPU_BUNDLE_DOCUMENT_VERSION = 4
NPU_BUNDLE_EXECUTION_SCHEMA_ID = "tcenv.remote.npu-bundle.execution.1"
#: CandidateDigest algorithm for an artifact set (bundle).
BUNDLE_DIGEST_ALGORITHM = "sha256-bundle-v1"
#: P1 timing evidence schema.  Version 1 (P0/P0b) stays byte-identical; the
#: versioned successor adds the declared topology, per-device spans and buffer
#: completeness that a completeness decision needs.
NPU_BUNDLE_TIMING_EVIDENCE_SCHEMA_ID = "tcenv.remote.npu-bundle.timing-evidence.2"
NPU_BUNDLE_EVIDENCE_MEDIA_TYPE = "application/vnd.tcenv.npu-bundle.timing-evidence.v2+json"
NPU_TIMING_EVIDENCE_SCHEMA_ID = "tcenv.remote.npu-source.timing-evidence.1"
NPU_EVIDENCE_MEDIA_TYPE = "application/vnd.tcenv.npu-source.timing-evidence.v1+json"
# P1c NPU contract: the same artifact-set contract with a *build* identity.  The
# request declares how the executable artifact set was (or will be) produced and
# the result carries a BuildEvidence document in the signed evidence.  It is a
# later document version, not an optional field on `npu-bundle`, and the build
# provenance is an independent document: `artifact_set_digest` keeps describing
# only the artifact manifest/bytes.
NPU_BUILD_API_CONTRACT_ID = "tcenv.remote.npu-build.evaluator-api.1"
NPU_BUILD_EVIDENCE_CONTRACT_ID = "tcenv.remote.npu-build.evidence.1"
NPU_BUILD_REQUEST_DOCUMENT_TYPE = "tcenv.remote.npu-build.request-digest"
NPU_BUILD_RECEIPT_DOCUMENT_TYPE = "tcenv.remote.npu-build.receipt-id"
NPU_BUILD_RESULT_DOCUMENT_TYPE = "tcenv.remote.npu-build.result-signature"
NPU_BUILD_DOCUMENT_VERSION = 5
NPU_BUILD_REQUEST_SCHEMA_ID = "tcenv.remote.npu-build.build-request.1"
NPU_BUILD_PROVENANCE_SCHEMA_ID = "tcenv.remote.npu-build.provenance.1"
NPU_BUILD_EVIDENCE_SCHEMA_ID = "tcenv.remote.npu-build.evidence.1"
NPU_BUILD_TIMING_EVIDENCE_SCHEMA_ID = "tcenv.remote.npu-build.timing-evidence.3"
NPU_BUILD_EVIDENCE_MEDIA_TYPE = "application/vnd.tcenv.npu-build.timing-evidence.v3+json"
#: How the executable artifact set is produced.  ``prebuilt`` means the control
#: host built it before Submit and the evaluator only verifies/loads it;
#: ``evaluator`` means this attempt builds the source it received.
BUILD_MODES = frozenset({"prebuilt", "evaluator"})
#: Where the build ran.  The two values are distinct host identities, never a
#: free-form label.
BUILD_HOSTS = {"prebuilt": "control_host", "evaluator": "evaluator"}
#: Outcome of a build.  ``reused`` is a verified prebuilt artifact set (no
#: compile happened in this attempt); the other values are one attempt's own
#: build status.  ``not_run`` is a diagnostic row (timeout before the build).
BUILD_STATUSES = frozenset({"reused", "ok", "compile_failed", "not_run"})
#: The shared error classification.  A build failure is a *candidate* compile
#: failure, never an infrastructure timeout; a toolchain/profile mismatch is
#: refused before Submit/create/start and never reaches a result at all.
BUILD_ERROR_CLASSES = frozenset({"compile", "toolchain_profile", "deadline"})
#: Where a BuildEvidence row was produced, which decides which combination of
#: mode/status is coherent:
#: ``control_host`` is a fresh producer build on the control host (the local
#: Metal pipeline and the prebuilt TSM module), ``evaluator`` is a fresh build
#: inside an attempt, and ``received`` is an evaluator-side row for a prebuilt
#: artifact set it verified and loaded (it never compiled).
BUILD_PLACEMENTS = frozenset({"control_host", "evaluator", "received"})
#: A prebuilt request must carry the producer's build-evidence link so the
#: receiver's signed evidence closes over the original log digest, duration and
#: status.  It is a bounded record, not the producer's whole document: the
#: digest binds the document the control host kept.
NPU_BUILD_PRODUCER_RECORD_SCHEMA_ID = "tcenv.remote.npu-build.producer-evidence-link.1"
# Clock domain recorded when no device clock was ever observed (killed worker).
NPU_UNOBSERVED_CLOCK_DOMAIN = "unobserved"
#: The candidate report schema every contract version signs.  It is shared by
#: all versions (`tcenv.eval-report.v3`), so a rewritten descriptor can never
#: relabel the report as another schema without breaking the envelope check.
REPORT_SCHEMA_ID = "tcenv.eval-report.v3"
REQUEST_DIGEST_ALGORITHM = "sha256-rfc8785-v1"
RECEIPT_DIGEST_ALGORITHM = "sha256-rfc8785-v1"
RESULT_SIGNATURE_ALGORITHM = "hmac-sha256-v1"

REQUEST_DOCUMENT_TYPE = "tcenv.remote.v1.request-digest"
RECEIPT_DOCUMENT_TYPE = "tcenv.remote.v1.receipt-id"
RESULT_DOCUMENT_TYPE = "tcenv.remote.v1.result-signature"
CONTRACT_DOCUMENT_VERSION = 1


@dataclass(frozen=True)
class ContractDialect:
    """One versioned external API/evidence contract.

    The dialect selects identity strings, canonical document types and the
    receipt shape.  It never selects behaviour from the shape of a request: the
    frontend's configured dialect is authoritative and a request that does not
    match it is rejected before durable acceptance.
    """

    api_contract_id: str
    evidence_contract_id: str
    request_document_type: str
    receipt_document_type: str
    result_document_type: str
    document_version: int
    npu_source: bool = False
    #: ``source`` (inline candidate bytes), ``blob`` (content-addressed
    #: reference) or ``bundle`` (versioned artifact set).  The contract identity
    #: selects the admissible form; a request shape never does.
    artifact_form: str = "source"
    #: The identity this contract version *writes* into the result evidence
    #: descriptor.  Both ends (frontend, client, durable read, outbox) take it
    #: from here, so the schema id of a version-2 evidence document can never be
    #: the version-1 id.
    evidence_schema_id: str = ""
    evidence_media_type: str = ""
    #: Whether a descriptor that names *another* identity is refused at the
    #: signing/durable-read boundary.  The legacy and P0/P0b wire froze
    #: ``application/json`` before the versioned media type existed, and their
    #: signed goldens document a different media type, so rewriting them is out of
    #: scope; ``document_version=4`` is the first version that enforces what it
    #: writes.
    enforce_evidence_identity: bool = False
    #: Whether this contract version binds a BuildProvenance document in the
    #: canonical request and a BuildEvidence document in the signed evidence.
    #: It is a later version of the artifact-set contract; the earlier versions
    #: reject both fields as unknown and their digests never change.
    build_contract: bool = False

    @property
    def identity_pair(self) -> tuple[str, str]:
        return (self.api_contract_id, self.evidence_contract_id)

    @property
    def npu_blob(self) -> bool:
        return self.artifact_form == "blob"

    @property
    def npu_bundle(self) -> bool:
        return self.artifact_form == "bundle"

    @property
    def candidate_algorithms(self) -> tuple[str, ...]:
        """CandidateDigest algorithms this dialect accepts."""

        if self.npu_bundle or self.build_contract:
            # The bundle contract exposes the source/bundle artifact oneof, so
            # both candidate identities are legitimate *under* the same
            # contract: an inline source keeps `sha256`, an artifact set uses
            # `sha256-bundle-v1`.
            return ("sha256", BUNDLE_DIGEST_ALGORITHM)
        return (BLOB_DIGEST_ALGORITHM,) if self.npu_blob else ("sha256",)


LEGACY_EVIDENCE_SCHEMA_ID = "tcenv.remote.v1.selfcheck-evidence"
LEGACY_DIALECT = ContractDialect(
    api_contract_id=API_CONTRACT_ID,
    evidence_contract_id=EVIDENCE_CONTRACT_ID,
    request_document_type=REQUEST_DOCUMENT_TYPE,
    receipt_document_type=RECEIPT_DOCUMENT_TYPE,
    result_document_type=RESULT_DOCUMENT_TYPE,
    document_version=CONTRACT_DOCUMENT_VERSION,
    npu_source=False,
    evidence_schema_id=LEGACY_EVIDENCE_SCHEMA_ID,
    evidence_media_type="application/json",
)
NPU_SOURCE_DIALECT = ContractDialect(
    api_contract_id=NPU_SOURCE_API_CONTRACT_ID,
    evidence_contract_id=NPU_SOURCE_EVIDENCE_CONTRACT_ID,
    request_document_type=NPU_SOURCE_REQUEST_DOCUMENT_TYPE,
    receipt_document_type=NPU_SOURCE_RECEIPT_DOCUMENT_TYPE,
    result_document_type=NPU_SOURCE_RESULT_DOCUMENT_TYPE,
    document_version=NPU_SOURCE_DOCUMENT_VERSION,
    npu_source=True,
    artifact_form="source",
    evidence_schema_id=NPU_TIMING_EVIDENCE_SCHEMA_ID,
    evidence_media_type="application/json",
)
NPU_BLOB_DIALECT = ContractDialect(
    api_contract_id=NPU_BLOB_API_CONTRACT_ID,
    evidence_contract_id=NPU_BLOB_EVIDENCE_CONTRACT_ID,
    request_document_type=NPU_BLOB_REQUEST_DOCUMENT_TYPE,
    receipt_document_type=NPU_BLOB_RECEIPT_DOCUMENT_TYPE,
    result_document_type=NPU_BLOB_RESULT_DOCUMENT_TYPE,
    document_version=NPU_BLOB_DOCUMENT_VERSION,
    npu_source=True,
    artifact_form="blob",
    evidence_schema_id=NPU_TIMING_EVIDENCE_SCHEMA_ID,
    evidence_media_type="application/json",
)
NPU_BUNDLE_DIALECT = ContractDialect(
    api_contract_id=NPU_BUNDLE_API_CONTRACT_ID,
    evidence_contract_id=NPU_BUNDLE_EVIDENCE_CONTRACT_ID,
    request_document_type=NPU_BUNDLE_REQUEST_DOCUMENT_TYPE,
    receipt_document_type=NPU_BUNDLE_RECEIPT_DOCUMENT_TYPE,
    result_document_type=NPU_BUNDLE_RESULT_DOCUMENT_TYPE,
    document_version=NPU_BUNDLE_DOCUMENT_VERSION,
    npu_source=True,
    artifact_form="bundle",
    # The version-2 evidence document and its descriptor are one identity: the
    # schema id, the media type and the signed bytes must not disagree.
    evidence_schema_id=NPU_BUNDLE_TIMING_EVIDENCE_SCHEMA_ID,
    evidence_media_type=NPU_BUNDLE_EVIDENCE_MEDIA_TYPE,
    enforce_evidence_identity=True,
)
NPU_BUILD_DIALECT = ContractDialect(
    api_contract_id=NPU_BUILD_API_CONTRACT_ID,
    evidence_contract_id=NPU_BUILD_EVIDENCE_CONTRACT_ID,
    request_document_type=NPU_BUILD_REQUEST_DOCUMENT_TYPE,
    receipt_document_type=NPU_BUILD_RECEIPT_DOCUMENT_TYPE,
    result_document_type=NPU_BUILD_RESULT_DOCUMENT_TYPE,
    document_version=NPU_BUILD_DOCUMENT_VERSION,
    npu_source=True,
    # A build request still carries exactly one artifact branch: inline source
    # for an evaluator build, the artifact set for a prebuilt one.  The build
    # document says which, and the two must agree.
    artifact_form="bundle",
    evidence_schema_id=NPU_BUILD_TIMING_EVIDENCE_SCHEMA_ID,
    evidence_media_type=NPU_BUILD_EVIDENCE_MEDIA_TYPE,
    enforce_evidence_identity=True,
    build_contract=True,
)
NPU_DIALECTS = (NPU_SOURCE_DIALECT, NPU_BLOB_DIALECT, NPU_BUNDLE_DIALECT, NPU_BUILD_DIALECT)
NPU_API_CONTRACT_IDS = frozenset(dialect.api_contract_id for dialect in NPU_DIALECTS)
NPU_EVIDENCE_CONTRACT_IDS = frozenset(dialect.evidence_contract_id for dialect in NPU_DIALECTS)
CONTRACT_DIALECTS = {
    LEGACY_DIALECT.api_contract_id: LEGACY_DIALECT,
    NPU_SOURCE_DIALECT.api_contract_id: NPU_SOURCE_DIALECT,
    NPU_BLOB_DIALECT.api_contract_id: NPU_BLOB_DIALECT,
    NPU_BUNDLE_DIALECT.api_contract_id: NPU_BUNDLE_DIALECT,
    NPU_BUILD_DIALECT.api_contract_id: NPU_BUILD_DIALECT,
}


def dialect_for(api_contract_id: str, evidence_contract_id: str | None = None) -> ContractDialect:
    """Resolve an exact contract pair; unknown or mixed pairs fail closed."""

    dialect = CONTRACT_DIALECTS.get(api_contract_id)
    if dialect is None:
        raise ContractError("API contract is not supported")
    if evidence_contract_id is not None and evidence_contract_id != dialect.evidence_contract_id:
        raise ContractError("API and evidence contracts are not a supported pair")
    return dialect

MAX_SOURCE_BYTES = 1_000_000
MAX_BLOB_BYTES = 16_000_000
MAX_TIMEOUT_MS = 3_600_000
MAX_REPS = 10_000
MAX_CASES = 128
MAX_CANONICAL_BYTES = 1_000_000
MAX_UINT64 = 18_446_744_073_709_551_615
_MAX_SAFE_INTEGER = 9_007_199_254_740_991

_HEX64 = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[A-Za-z0-9._:-]{1,160}")
_KEY_ID = re.compile(r"[A-Za-z0-9._:/-]{1,160}")
_DECIMAL = re.compile(r"0|[1-9][0-9]*")
_SECONDS_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,9})?Z")
_KINDS = frozenset({"formal_eval", "selfcheck"})
_MODES = {
    "formal_eval": frozenset({"quick", "full", "validate"}),
    "selfcheck": frozenset({"quick", "timed"}),
}
_CACHE_POLICIES = frozenset({"reuse", "refresh", "disabled"})
_MEASUREMENT_KINDS = frozenset({"measured", "simulated"})

_REQUEST_FIELDS = (
    "kind",
    "run_id",
    "session_id",
    "task",
    "dtype",
    "size",
    "target_id",
    "task_fingerprint",
    "candidate_sha256",
    "mode",
    "seed",
    "reps",
    "queue_timeout_ms",
    "candidate_timeout_ms",
    "cases",
    "fast_bench",
    "require_contract_launch",
    "validate",
    "measure_serial",
    "cache_policy",
)
_REQUEST_FIELD_SET = frozenset(_REQUEST_FIELDS)
# These fields belong to identity, transport, idempotency, or local execution
# envelopes.  They are intentionally not execution inputs.  Unknown fields are
# still rejected so a new execution option cannot silently escape the digest.
_EXCLUDED_REQUEST_FIELDS = frozenset({
    "request_id",
    "attempt_id",
    "authorization",
    "credentials",
    "session_grant",
    "tls_identity",
    "rpc_deadline",
    "rpc_deadline_ms",
    "trace_id",
    "retry_count",
    "stream_sequence",
    "workdir",
    "out",
    "heartbeat",
    "progress_path",
    "self_check_dir",
    "expected",
    "evaluator_epoch",
    "api_contract_id",
    "evidence_contract_id",
    "capability_id",
    "required_features",
    "lease_fence",
})

_PROVENANCE_FIELDS = frozenset({
    "schema_version",
    "evaluator_epoch",
    "site_id",
    "device_id",
    "backend",
    "backend_version",
    "target_id",
    "worker_build",
    "task_pack_digest",
    "calibration_digest",
    "protocol_version",
    "api_contract_id",
    "evidence_contract_id",
    "attempt_id",
    "request_digest",
    "measurement_kind",
    "publication_allowed",
})
_BLOB_FIELDS = frozenset({"schema_id", "media_type", "size_bytes", "sha256"})
_RECEIPT_FIELDS = frozenset({
    "receipt_id",
    "request_id",
    "attempt_id",
    "request_digest",
    "candidate_sha256",
    "lease_fence",
    "session_sequence",
    "issued_at",
})
_RESULT_FIELDS = frozenset({"provenance", "report", "evidence", "receipt"})


class ContractError(ValueError):
    """A canonical request, receipt, or signed result document is invalid."""


def _text(value: Any, name: str, *, max_bytes: int = 256, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value):
        raise ContractError(f"{name} must be a {'possibly empty ' if empty else 'non-empty '}string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ContractError(f"{name} must contain valid Unicode scalar values") from exc
    if len(encoded) > max_bytes:
        raise ContractError(f"{name} exceeds its UTF-8 byte limit")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ContractError(f"{name} must be a bounded opaque identifier")
    return value


def _key_id(value: Any) -> str:
    if not isinstance(value, str) or not _KEY_ID.fullmatch(value):
        raise ContractError("key_id must be a bounded opaque identifier")
    return value


def key_id(value: Any) -> str:
    """Validate a bounded ``site/key-version`` signature identifier."""

    return _key_id(value)


def _hex64(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ContractError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ContractError(f"{name} is outside its integer bounds")
    return value


def _decimal_uint(value: Any, name: str, *, minimum: int = 0) -> str:
    if not isinstance(value, str) or len(value) > 20 or not _DECIMAL.fullmatch(value):
        raise ContractError(f"{name} must be a canonical unsigned decimal string")
    number = int(value)
    if not minimum <= number <= MAX_UINT64:
        raise ContractError(f"{name} is outside its unsigned 64-bit bounds")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be boolean")
    return value


def _source_bytes(value: Any) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ContractError("source_bytes must be bytes-like; implicit text encoding is forbidden")
    source = bytes(value)
    if len(source) > MAX_SOURCE_BYTES:
        raise ContractError("source exceeds the byte limit")
    return source


def _validate_json(value: Any, *, depth: int = 0, budget: list[int] | None = None) -> None:
    if depth > 32:
        raise ContractError("canonical document nesting is too deep")
    if budget is None:
        budget = [10_000]
    budget[0] -= 1
    if budget[0] < 0:
        raise ContractError("canonical document has too many values")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        _text(value, "canonical string", max_bytes=MAX_CANONICAL_BYTES, empty=True)
        return
    if isinstance(value, int):
        if not -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER:
            raise ContractError("canonical integers must be IEEE-754 safe; use a decimal string")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json(item, depth=depth + 1, budget=budget)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key or not key.isascii():
                raise ContractError("canonical document field names must be non-empty ASCII")
            _validate_json(item, depth=depth + 1, budget=budget)
        return
    raise ContractError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_digest_document(document: Mapping[str, Any]) -> bytes:
    """Return RFC 8785 bytes for the protocol's validated JSON subset.

    All protocol field names are ASCII, so Python's key order is the RFC 8785
    order.  Floats, non-finite numbers, non-scalar Unicode, and unsafe integers
    are rejected rather than serialized differently by different languages.
    """

    if not isinstance(document, Mapping):
        raise ContractError("canonical document must be an object")
    _validate_json(document)
    try:
        encoded = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ContractError(f"document is not canonical JSON: {exc}") from exc
    if len(encoded) > MAX_CANONICAL_BYTES:
        raise ContractError("canonical document exceeds the byte limit")
    return encoded


def _document_digest(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_digest_document(document)).hexdigest()


def seconds_to_timeout_ms(seconds: int | float | str | Decimal) -> int:
    """Convert a positive decimal seconds value with deterministic ceiling.

    The conversion happens once at the legacy/host boundary.  Canonical request
    documents accept only the resulting integer milliseconds.  ``str(float)``
    is used for existing Python callers; cross-language callers should supply a
    decimal string or an already normalized integer-millisecond value.
    """

    if isinstance(seconds, bool):
        raise ContractError("timeout seconds must be a finite positive decimal")
    text = str(seconds)
    if not _SECONDS_DECIMAL.fullmatch(text):
        raise ContractError("timeout seconds must use canonical unsigned decimal syntax without whitespace")
    try:
        value = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise ContractError(f"timeout seconds must be a finite positive decimal: {exc}") from exc
    if not value.is_finite() or value <= 0:
        raise ContractError("timeout seconds must be a finite positive decimal")
    milliseconds = int((value * 1000).to_integral_value(rounding=ROUND_CEILING))
    if not 1 <= milliseconds <= MAX_TIMEOUT_MS:
        raise ContractError("timeout is outside the integer-millisecond bounds")
    return milliseconds


def _common_request_fields(document: Mapping[str, Any]) -> dict[str, Any]:
    kind = _text(document["kind"], "kind", max_bytes=32)
    if kind not in _KINDS:
        raise ContractError("kind must be formal_eval or selfcheck")
    mode = _text(document["mode"], "mode", max_bytes=32)
    if mode not in _MODES[kind]:
        raise ContractError(f"mode {mode!r} is not allowed for {kind}")
    seed = document["seed"]
    if seed is not None:
        seed = _decimal_uint(seed, "seed")
    cases_value = document["cases"]
    if cases_value is None:
        cases = None
    else:
        if not isinstance(cases_value, (list, tuple)) or len(cases_value) > MAX_CASES:
            raise ContractError("cases must be a bounded array or null")
        cases = [_text(case, "case", max_bytes=256) for case in cases_value]
    cache_policy = _text(document["cache_policy"], "cache_policy", max_bytes=32)
    if cache_policy not in _CACHE_POLICIES:
        raise ContractError("cache_policy must be reuse, refresh or disabled")
    return {
        "kind": kind,
        "run_id": _identifier(document["run_id"], "run_id"),
        "session_id": _identifier(document["session_id"], "session_id"),
        "task": _text(document["task"], "task"),
        "dtype": _text(document["dtype"], "dtype", max_bytes=64),
        "size": _text(document["size"], "size", max_bytes=128),
        "target_id": _identifier(document["target_id"], "target_id"),
        "task_fingerprint": _hex64(document["task_fingerprint"], "task_fingerprint"),
        "mode": mode,
        "seed": seed,
        "reps": _integer(document["reps"], "reps", 1, MAX_REPS),
        "queue_timeout_ms": _integer(document["queue_timeout_ms"], "queue_timeout_ms", 1, MAX_TIMEOUT_MS),
        "candidate_timeout_ms": _integer(
            document["candidate_timeout_ms"], "candidate_timeout_ms", 1, MAX_TIMEOUT_MS
        ),
        "cases": cases,
        "fast_bench": _boolean(document["fast_bench"], "fast_bench"),
        "require_contract_launch": _boolean(
            document["require_contract_launch"], "require_contract_launch"
        ),
        "validate": _boolean(document["validate"], "validate"),
        "measure_serial": _boolean(document["measure_serial"], "measure_serial"),
        "cache_policy": cache_policy,
    }


def candidate_digest_document(
    value: Any, *, source_bytes: bytes | bytearray | memoryview | None = None,
    allowed_algorithms: tuple[str, ...] = ("sha256",), blob: Any = None,
) -> dict[str, str]:
    """Validate a CandidateDigest and optionally bind exact bytes or a blob.

    The inline source contract accepts only ``sha256`` over the request bytes.
    The blob-reference contract accepts only ``sha256-blob-v1`` and requires the
    value to equal the descriptor digest, so a candidate can never be replayed
    under another artifact form.  ``blob`` is the descriptor document, not a
    second place to declare the bytes.
    """

    if not isinstance(value, Mapping) or set(value) != {"algorithm", "value"}:
        raise ContractError("candidate_digest must have exactly algorithm and value")
    algorithm = _text(value["algorithm"], "candidate_digest.algorithm", max_bytes=32)
    if algorithm not in allowed_algorithms:
        raise ContractError(
            "candidate_digest.algorithm must be one of " + ", ".join(sorted(allowed_algorithms))
        )
    digest = _hex64(value["value"], "candidate_digest.value")
    if source_bytes is not None:
        try:
            source = _source_bytes(source_bytes)
        except ContractError as exc:
            raise ContractError(str(exc)) from exc
        if hashlib.sha256(source).hexdigest() != digest:
            raise ContractError("candidate_digest does not match the exact source bytes")
    if blob is not None:
        if blob_descriptor_document(blob)["sha256"] != digest:
            raise ContractError("candidate_digest does not match the blob descriptor digest")
    return {"algorithm": algorithm, "value": digest}


def candidate_digest(source_bytes: bytes | bytearray | memoryview) -> dict[str, str]:
    return {"algorithm": "sha256", "value": hashlib.sha256(_source_bytes(source_bytes)).hexdigest()}


_BLOB_REF_FIELDS = frozenset({"content_id", "sha256", "size_bytes", "media_type", "chunk_size"})


def blob_content_id(digest: str) -> str:
    """Return the stable algorithm-tagged content id for a content digest."""

    return BLOB_CONTENT_ID_PREFIX + _hex64(digest, "blob sha256")


def blob_descriptor_document(value: Any) -> dict[str, Any]:
    """Validate one content-addressed blob reference document.

    The reference is canonical request input: it carries the content identity,
    the exact byte count, the media type and the chunk boundary.  It never
    carries a host path, a URL or vendor parameters.
    """

    if not isinstance(value, Mapping) or set(value) != _BLOB_REF_FIELDS:
        raise ContractError("blob_ref fields are missing or unknown")
    digest = _hex64(value["sha256"], "blob_ref.sha256")
    content_id = _text(value["content_id"], "blob_ref.content_id", max_bytes=96)
    if content_id != blob_content_id(digest):
        raise ContractError("blob_ref.content_id does not bind blob_ref.sha256")
    return {
        "content_id": content_id,
        "sha256": digest,
        "size_bytes": _integer(value["size_bytes"], "blob_ref.size_bytes", 0, MAX_UINT64),
        "media_type": _text(value["media_type"], "blob_ref.media_type", max_bytes=160),
        "chunk_size": _integer(value["chunk_size"], "blob_ref.chunk_size", 1, MAX_BLOB_CHUNK_BYTES),
    }


_DEVICE_GROUP_FIELDS = frozenset({
    "device_group_id", "vendor_id", "profile_digest", "topology_digest", "resource_generation",
})


def device_group_document(value: Any) -> dict[str, str]:
    """Validate the minimal logical DeviceGroup binding used by P0."""

    if not isinstance(value, Mapping) or set(value) != _DEVICE_GROUP_FIELDS:
        raise ContractError("device_group fields are missing or unknown")
    return {
        "device_group_id": _identifier(value["device_group_id"], "device_group.device_group_id"),
        "vendor_id": _identifier(value["vendor_id"], "device_group.vendor_id"),
        "profile_digest": _hex64(value["profile_digest"], "device_group.profile_digest"),
        "topology_digest": _hex64(value["topology_digest"], "device_group.topology_digest"),
        "resource_generation": _decimal_uint(
            value["resource_generation"], "device_group.resource_generation", minimum=1,
        ),
    }


_NPU_EXECUTION_FIELDS = frozenset({
    "artifact_kind", "launch_kind", "launch_subtype", "source_media_type",
    "device_group", "required_timing_sources", "synchronization",
})
_NPU_TIMING_SOURCES = frozenset({"DEVICE_EVENT"})


def npu_execution_document(value: Any) -> dict[str, Any]:
    """Validate the P0 source-only execution spec.

    P0 deliberately has no bundle, graph, model, multi-rank, die or profiler
    branch.  Unsupported values fail closed before durable acceptance.
    """

    if not isinstance(value, Mapping) or set(value) != _NPU_EXECUTION_FIELDS:
        raise ContractError("execution spec fields are missing or unknown")
    artifact_kind = _text(value["artifact_kind"], "execution.artifact_kind", max_bytes=32)
    if artifact_kind != "source":
        raise ContractError("P0 supports only the source artifact kind")
    launch_kind = _text(value["launch_kind"], "execution.launch_kind", max_bytes=32)
    if launch_kind != "KERNEL":
        raise ContractError("P0 supports only the KERNEL launch kind")
    launch_subtype = _text(value["launch_subtype"], "execution.launch_subtype", max_bytes=32)
    if launch_subtype != "SIMT":
        raise ContractError("P0 supports only the SIMT launch subtype")
    timing_sources = value["required_timing_sources"]
    if not isinstance(timing_sources, (list, tuple)) or not timing_sources:
        raise ContractError("execution requires at least one timing source")
    sources = [_text(item, "execution.required_timing_sources", max_bytes=32) for item in timing_sources]
    if any(item not in _NPU_TIMING_SOURCES for item in sources):
        raise ContractError("P0 supports only DEVICE_EVENT timing")
    synchronization = _text(value["synchronization"], "execution.synchronization", max_bytes=32)
    if synchronization != "stream-event":
        raise ContractError("P0 supports only stream-event synchronization")
    return {
        "artifact_kind": artifact_kind,
        "launch_kind": launch_kind,
        "launch_subtype": launch_subtype,
        "source_media_type": _text(value["source_media_type"], "execution.source_media_type", max_bytes=64),
        "device_group": device_group_document(value["device_group"]),
        "required_timing_sources": sorted(set(sources)),
        "synchronization": synchronization,
    }


_NPU_SOURCE_REQUEST_FIELDS = frozenset(
    (set(_REQUEST_FIELD_SET) - {"candidate_sha256"}) | {"candidate_digest", "execution"}
)
_NPU_BLOB_REQUEST_FIELDS = _NPU_SOURCE_REQUEST_FIELDS | {"blob_ref"}
#: Bundles: the artifact payload is one explicit oneof branch, never an
#: optional field, and never both branches at once.
_NPU_BUNDLE_REQUEST_FIELDS = frozenset(
    (set(_REQUEST_FIELD_SET) - {"candidate_sha256"}) | {"candidate_digest", "execution", "artifact"}
)
#: P1c adds exactly one independent document to the artifact-set request: the
#: BuildProvenance identity (mode plus the builder/toolchain/profile it declares).
_NPU_BUILD_REQUEST_FIELDS = _NPU_BUNDLE_REQUEST_FIELDS | {"build"}
ARTIFACT_SOURCE = "source"
ARTIFACT_BUNDLE = "bundle"
_NPU_BUNDLE_EXECUTION_FIELDS = frozenset({
    "schema_id", "task_family", "artifact_kind", "launch_kind", "launch_subtype",
    "source_media_type", "device_group", "resources", "required_timing_sources",
    "synchronization",
})
_NPU_RESOURCE_FIELDS = frozenset({"logical_devices", "required_rank_spans", "required_die_spans"})
#: P1 keeps device-event timing only; profiler dispatch stays blocked until the
#: P3 per-capability gate, so an agent cannot ask for a source this frontend
#: cannot corroborate.
_NPU_BUNDLE_TIMING_SOURCES = frozenset({"DEVICE_EVENT"})
#: Declared task families and the execution shape each one implies.  The shape
#: is derived, never taken from a client-supplied default: a graph request that
#: claims SIMT/stream semantics is a contract error.
_TASK_FAMILIES = {
    "kernel": ("source", "KERNEL", "SIMT", "stream-event"),
    "graph": ("bundle", "GRAPH", "SERVICE", "service-completion-token"),
}
#: The main artifact media type each task family implies.  It is derived rather
#: than negotiated so a client and a frontend can never disagree about the
#: execution document (which is part of the request digest); a bundle's real
#: media types live on its entries.
_TASK_FAMILY_MEDIA_TYPE = {
    "kernel": "text/plain; charset=utf-8",
    "graph": "application/octet-stream",
}
#: One served device group per frontend in P1, matching the single-active lease.
MAX_LOGICAL_DEVICES = 8
MAX_DECLARED_SPANS = 64


#: P1c adds one derived shape: a *prebuilt* executable module is an artifact set
#: launched as KERNEL/SIMT with stream-event synchronization.  It is a new task
#: family precisely so the earlier contract versions keep rejecting it.
_TASK_FAMILIES_BUILD = {
    **_TASK_FAMILIES,
    "kernel_module": ("bundle", "KERNEL", "SIMT", "stream-event"),
}
_TASK_FAMILY_MEDIA_TYPE_BUILD = {
    **_TASK_FAMILY_MEDIA_TYPE,
    "kernel_module": "text/plain; charset=utf-8",
}


def npu_bundle_execution_document(value: Any) -> dict[str, Any]:
    """Validate the version 4 P1 ExecutionSpec document."""

    return _npu_execution_document(
        value, families=_TASK_FAMILIES, media_types=_TASK_FAMILY_MEDIA_TYPE,
        error_families="kernel or graph",
    )


def npu_build_execution_document(value: Any) -> dict[str, Any]:
    """Validate the version 5 ExecutionSpec document (adds ``kernel_module``)."""

    return _npu_execution_document(
        value, families=_TASK_FAMILIES_BUILD, media_types=_TASK_FAMILY_MEDIA_TYPE_BUILD,
        error_families="kernel, kernel_module or graph",
    )


def npu_execution_document_for(
    value: Any, *, dialect: ContractDialect = NPU_SOURCE_DIALECT,
) -> dict[str, Any]:
    """Validate an execution document with the schema its contract version owns."""

    if dialect.build_contract:
        return npu_build_execution_document(value)
    if dialect.npu_bundle:
        return npu_bundle_execution_document(value)
    return npu_execution_document(value)


def _npu_execution_document(
    value: Any, *, families: Mapping[str, tuple[str, str, str, str]],
    media_types: Mapping[str, str], error_families: str,
) -> dict[str, Any]:
    """Validate the versioned artifact-set ExecutionSpec document.

    It states the task family, the launch kind, the logical device group, the
    declared resource/topology requirements, the synchronization strategy and
    the timing sources.  The cache policy stays a request-level field and is
    rejected here (unknown field).  Physical BDF/rank/fabric placement is not
    representable: the trusted server-side profile decides it.
    """

    if not isinstance(value, Mapping) or set(value) != _NPU_BUNDLE_EXECUTION_FIELDS:
        raise ContractError("execution spec fields are missing or unknown")
    if value["schema_id"] != NPU_BUNDLE_EXECUTION_SCHEMA_ID:
        raise ContractError("execution spec schema_id is not supported by this contract version")
    task_family = _text(value["task_family"], "execution.task_family", max_bytes=32)
    shape = families.get(task_family)
    if shape is None:
        raise ContractError(f"execution.task_family must be {error_families}")
    artifact_kind = _text(value["artifact_kind"], "execution.artifact_kind", max_bytes=32)
    launch_kind = _text(value["launch_kind"], "execution.launch_kind", max_bytes=32)
    launch_subtype = _text(value["launch_subtype"], "execution.launch_subtype", max_bytes=32)
    synchronization = _text(value["synchronization"], "execution.synchronization", max_bytes=32)
    if (artifact_kind, launch_kind, launch_subtype, synchronization) != shape:
        raise ContractError(
            f"execution.task_family {task_family!r} requires "
            f"artifact_kind={shape[0]}/{shape[1]}/{shape[2]}/{shape[3]}"
        )
    timing_sources = value["required_timing_sources"]
    if not isinstance(timing_sources, (list, tuple)) or not timing_sources:
        raise ContractError("execution requires at least one timing source")
    sources = [_text(item, "execution.required_timing_sources", max_bytes=32) for item in timing_sources]
    if any(item not in _NPU_BUNDLE_TIMING_SOURCES for item in sources):
        raise ContractError("P1 supports only DEVICE_EVENT timing")
    # ``null`` states that the dimension does not exist for this vendor shape;
    # it is not the same as declaring zero spans.
    declared = device_group_requirements(value["resources"])
    media_type = _text(value["source_media_type"], "execution.source_media_type", max_bytes=64)
    if media_type != media_types[task_family]:
        raise ContractError(
            f"execution.task_family {task_family!r} requires source_media_type "
            f"{media_types[task_family]!r}"
        )
    return {
        "schema_id": NPU_BUNDLE_EXECUTION_SCHEMA_ID,
        "task_family": task_family,
        "artifact_kind": artifact_kind,
        "launch_kind": launch_kind,
        "launch_subtype": launch_subtype,
        "source_media_type": media_type,
        "device_group": device_group_document(value["device_group"]),
        "resources": declared,
        "required_timing_sources": sorted(set(sources)),
        "synchronization": synchronization,
    }


def npu_artifact_document(value: Any, *, execution: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the source/bundle artifact oneof document.

    Exactly one branch is present.  A bundle is versioned and carries its
    manifest digest; an inline source carries only its size and digest, because
    the exact bytes travel in the message envelope and are re-hashed there.
    """

    if not isinstance(value, Mapping) or len(value) != 1:
        raise ContractError("artifact must carry exactly one of source or bundle")
    branch = next(iter(value))
    if branch not in (ARTIFACT_SOURCE, ARTIFACT_BUNDLE):
        raise ContractError("artifact must carry exactly one of source or bundle")
    if branch == ARTIFACT_SOURCE:
        if execution["artifact_kind"] != "source":
            raise ContractError("an inline source artifact requires artifact_kind=source")
        descriptor = value[branch]
        if not isinstance(descriptor, Mapping) or set(descriptor) != {"size_bytes", "sha256"}:
            raise ContractError("the inline source artifact must carry exactly size_bytes and sha256")
        return {
            ARTIFACT_SOURCE: {
                "size_bytes": _integer(descriptor["size_bytes"], "artifact.source.size_bytes", 1, MAX_SOURCE_BYTES),
                "sha256": _hex64(descriptor["sha256"], "artifact.source.sha256"),
            }
        }
    if execution["artifact_kind"] != "bundle":
        raise ContractError("a bundle artifact requires artifact_kind=bundle")
    from .artifact_bundle import BundleError, bundle_document

    try:
        normalized = bundle_document(value[branch])
    except BundleError as exc:
        raise ContractError(str(exc)) from exc
    return {ARTIFACT_BUNDLE: {
        "schema_id": normalized["schema_id"],
        "artifact_set_digest": normalized["artifact_set_digest"],
        "entries": normalized["entries"],
    }}


_BUILD_TOOLCHAIN_FIELDS = frozenset({"id", "version", "sdk_version"})
_BUILD_PROVENANCE_FIELDS = frozenset({
    "schema_id", "builder_id", "builder_version", "build_host", "build_profile",
    "toolchain", "compile_profile", "source_sha256", "artifact_set_digest",
})
_BUILD_REQUEST_FIELDS = frozenset({"schema_id", "mode", "provenance"})
_BUILD_PREBUILT_REQUEST_FIELDS = _BUILD_REQUEST_FIELDS | {"producer"}
_BUILD_PRODUCER_FIELDS = frozenset({
    "schema_id", "status", "mode", "duration_ms", "log", "evidence_sha256",
})
_BUILD_ARTIFACTS_FIELDS = frozenset({"schema_id", "entries"})
_BUILD_LOG_FIELDS = frozenset({"sha256", "size_bytes"})
_BUILD_EVIDENCE_FIELDS = frozenset({
    "schema_id", "mode", "status", "error_class", "provenance", "artifacts",
    "duration_ms", "log", "loaded",
})
_BUILD_LOADED_FIELDS = frozenset({"path", "sha256"})
MAX_BUILD_DURATION_MS = 3_600_000


def _build_toolchain_document(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _BUILD_TOOLCHAIN_FIELDS:
        raise ContractError("build.toolchain fields are missing or unknown")
    return {
        "id": _identifier(value["id"], "build.toolchain.id"),
        "version": _text(value["version"], "build.toolchain.version", max_bytes=64),
        "sdk_version": _text(value["sdk_version"], "build.toolchain.sdk_version", max_bytes=64),
    }


def build_provenance_document(
    value: Any, *, mode: str | None = None,
) -> dict[str, Any]:
    """Validate a BuildProvenance document.

    The provenance names *who built what with which toolchain*: builder identity
    and version, the build host/profile, the toolchain/SDK versions, the exact
    source digest, the compile profile and the resulting ``artifact_set_digest``.
    It is deliberately a separate document from the artifact manifest: the
    artifact digest keeps describing only the manifest/bytes, while a changed
    build environment is visible as a different provenance.  Nothing here can
    name a shell, a host path or a free-form build command.
    """

    if not isinstance(value, Mapping) or set(value) != _BUILD_PROVENANCE_FIELDS:
        raise ContractError("build.provenance fields are missing or unknown")
    if value["schema_id"] != NPU_BUILD_PROVENANCE_SCHEMA_ID:
        raise ContractError("build.provenance schema_id is not supported by this contract version")
    build_host = _text(value["build_host"], "build.provenance.build_host", max_bytes=32)
    if build_host not in set(BUILD_HOSTS.values()):
        raise ContractError("build.provenance.build_host must be control_host or evaluator")
    if mode is not None and build_host != BUILD_HOSTS[mode]:
        raise ContractError(
            f"build mode {mode!r} requires build_host {BUILD_HOSTS[mode]!r}"
        )
    artifact_set_digest = value["artifact_set_digest"]
    if artifact_set_digest is not None:
        _hex64(artifact_set_digest, "build.provenance.artifact_set_digest")
    compile_profile = value["compile_profile"]
    if not isinstance(compile_profile, Mapping) or len(compile_profile) > 16:
        raise ContractError("build.provenance.compile_profile must be a bounded object")
    return {
        "schema_id": NPU_BUILD_PROVENANCE_SCHEMA_ID,
        "builder_id": _identifier(value["builder_id"], "build.provenance.builder_id"),
        "builder_version": _text(value["builder_version"], "build.provenance.builder_version", max_bytes=64),
        "build_host": build_host,
        "build_profile": _identifier(value["build_profile"], "build.provenance.build_profile"),
        "toolchain": _build_toolchain_document(value["toolchain"]),
        "compile_profile": _npu_diagnostics(compile_profile),
        "source_sha256": _hex64(value["source_sha256"], "build.provenance.source_sha256"),
        "artifact_set_digest": artifact_set_digest,
    }


def build_producer_record(value: Any) -> dict[str, Any]:
    """Validate the producer build-evidence link carried by a prebuilt request.

    The control host keeps its complete ``BuildEvidence`` document (artifacts,
    log bytes, duration).  What crosses the wire is this bounded link: the
    producer's status/mode, build duration, log digest/size and the digest of
    that document.  The receiver's signed evidence must repeat the link
    unchanged, so the original build record is authenticated by the request
    digest and closed over by the result -- without shipping the log bytes to
    the NPU.
    """

    if not isinstance(value, Mapping) or set(value) != _BUILD_PRODUCER_FIELDS:
        raise ContractError("build.producer fields are missing or unknown")
    if value["schema_id"] != NPU_BUILD_PRODUCER_RECORD_SCHEMA_ID:
        raise ContractError("build.producer schema_id is not supported by this contract version")
    status = _text(value["status"], "build.producer.status", max_bytes=32)
    if status != "ok":
        raise ContractError("a producer build-evidence link must describe a successful build")
    mode = _text(value["mode"], "build.producer.mode", max_bytes=32)
    if mode != "prebuilt":
        raise ContractError("a producer build-evidence link describes a control-host build")
    duration_ms = _integer(value["duration_ms"], "build.producer.duration_ms", 0, MAX_BUILD_DURATION_MS)
    log = value["log"]
    if log is not None:
        if not isinstance(log, Mapping) or set(log) != _BUILD_LOG_FIELDS:
            raise ContractError("build.producer.log fields are missing or unknown")
        log = {
            "sha256": _hex64(log["sha256"], "build.producer.log.sha256"),
            "size_bytes": _integer(log["size_bytes"], "build.producer.log.size_bytes", 0, MAX_SOURCE_BYTES),
        }
    evidence_sha256 = _hex64(value["evidence_sha256"], "build.producer.evidence_sha256")
    return {
        "schema_id": NPU_BUILD_PRODUCER_RECORD_SCHEMA_ID,
        "status": status,
        "mode": mode,
        "duration_ms": duration_ms,
        "log": log,
        "evidence_sha256": evidence_sha256,
    }


def build_request_document(value: Any) -> dict[str, Any]:
    """Validate the request-level build document (mode plus provenance).

    A ``prebuilt`` request states the complete BuildProvenance the control host
    produced, including the artifact set digest, plus the producer's
    build-evidence link.  An ``evaluator`` request states the build identity the
    evaluator must use and leaves the artifact digest ``null``: the artifact does
    not exist yet.  The profile is an expectation, not an authorization: only a
    registered build_profile/toolchain may run it.
    """

    if not isinstance(value, Mapping):
        raise ContractError("build fields are missing or unknown")
    if value.get("schema_id") != NPU_BUILD_REQUEST_SCHEMA_ID:
        raise ContractError("build schema_id is not supported by this contract version")
    mode = _text(value.get("mode"), "build.mode", max_bytes=32)
    if mode not in BUILD_MODES:
        raise ContractError("build.mode must be prebuilt or evaluator")
    expected_fields = (
        _BUILD_PREBUILT_REQUEST_FIELDS if mode == "prebuilt" else _BUILD_REQUEST_FIELDS
    )
    if set(value) != expected_fields:
        if mode == "prebuilt" and set(value) == _BUILD_REQUEST_FIELDS:
            raise ContractError(
                "a prebuilt build request requires the producer build-evidence link"
            )
        raise ContractError("build fields are missing or unknown")
    provenance = build_provenance_document(value["provenance"], mode=mode)
    if mode == "evaluator" and provenance["artifact_set_digest"] is not None:
        raise ContractError("an evaluator build request must not claim an artifact_set_digest")
    if mode == "prebuilt" and provenance["artifact_set_digest"] is None:
        raise ContractError("a prebuilt build request requires the artifact_set_digest it produced")
    document = {
        "schema_id": NPU_BUILD_REQUEST_SCHEMA_ID,
        "mode": mode,
        "provenance": provenance,
    }
    if mode == "prebuilt":
        document["producer"] = build_producer_record(value["producer"])
    return document


def _build_artifacts_document(value: Any) -> dict[str, Any] | None:
    """Validate a produced artifact set and recompute its manifest digest."""

    if value is None:
        return None
    allowed_fields = _BUILD_ARTIFACTS_FIELDS | {"artifact_set_digest"}
    if not isinstance(value, Mapping) or set(value) - allowed_fields or not _BUILD_ARTIFACTS_FIELDS <= set(value):
        raise ContractError("build.artifacts fields are missing or unknown")
    from .artifact_bundle import BUNDLE_SCHEMA_ID, BundleError, artifact_set_digest as manifest_digest, bundle_document

    if value["schema_id"] != BUNDLE_SCHEMA_ID:
        raise ContractError("build.artifacts schema_id is not an artifact bundle schema")
    entries = value["entries"]
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ContractError("build.artifacts must list the produced entries")
    from .artifact_bundle import (
        MAX_BUNDLE_ENTRIES, MAX_BUNDLE_TOTAL_BYTES, _alias_key, _check_path_prefixes,
        canonical_manifest,
    )

    if len(entries) > MAX_BUNDLE_ENTRIES:
        raise ContractError("build.artifacts exceeds the entry-count limit")
    total = 0
    seen: dict[str, str] = {}
    manifest_entries = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ContractError("build.artifacts entries must be objects")
        try:
            manifest_entry = canonical_manifest(BUNDLE_SCHEMA_ID, (entry,))["entries"][0]
        except (BundleError, TypeError, KeyError) as exc:
            raise ContractError(f"build.artifacts is not a valid artifact manifest: {exc}") from exc
        alias = _alias_key(manifest_entry["path"])
        if alias in seen:
            raise ContractError("build.artifacts lists the same artifact path twice")
        seen[alias] = manifest_entry["path"]
        total += manifest_entry["size_bytes"]
        manifest_entries.append(manifest_entry)
    if total > MAX_BUNDLE_TOTAL_BYTES:
        raise ContractError("build.artifacts exceeds the declared total byte limit")
    _check_path_prefixes(item["path"] for item in manifest_entries)
    manifest_entries.sort(key=lambda item: item["path"])
    computed = manifest_digest(BUNDLE_SCHEMA_ID, manifest_entries)
    declared = value.get("artifact_set_digest")
    if declared is not None:
        _hex64(declared, "build.artifacts.artifact_set_digest")
        if declared != computed:
            raise ContractError("build.artifacts artifact_set_digest does not bind its entries")
    return {
        "schema_id": BUNDLE_SCHEMA_ID,
        "entries": manifest_entries,
        "artifact_set_digest": computed,
    }


def _build_loaded_document(value: Any, artifacts: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """Validate the artifacts the receiver actually loaded.

    A vendor that reports an artifact it never loaded (or a different digest)
    cannot claim a successful run: every loaded digest must be one of the
    produced entries, and a prebuilt receiver must load every executable entry.
    """

    if not isinstance(value, (list, tuple)):
        raise ContractError("build.loaded must be an array")
    known = {} if artifacts is None else {
        entry["path"]: entry["sha256"] for entry in artifacts["entries"]
    }
    loaded: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _BUILD_LOADED_FIELDS:
            raise ContractError("build.loaded entries are missing or unknown")
        from .artifact_bundle import MAX_BUNDLE_PATH_BYTES, validate_relative_path

        path = validate_relative_path(item["path"])
        digest = _hex64(item["sha256"], "build.loaded.sha256")
        if path in seen:
            raise ContractError("build.loaded reports the same artifact twice")
        seen.add(path)
        loaded.append({"path": path, "sha256": digest})
    if known:
        unknown = [item["path"] for item in loaded if item["path"] not in known]
        if unknown:
            raise ContractError("build.loaded names an artifact the build did not produce: " + ",".join(sorted(unknown)))
        wrong = [item["path"] for item in loaded if known[item["path"]] != item["sha256"]]
        if wrong:
            raise ContractError("build.loaded digest differs from the produced artifact: " + ",".join(sorted(wrong)))
    return loaded


def build_evidence_document(
    value: Any, *, placement: str, remote: bool = False,
    expected_build: Mapping[str, Any] | None = None,
    request_artifact: Mapping[str, Any] | None = None,
    report_status: str | None = None, complete: bool | None = None,
) -> dict[str, Any]:
    """Validate one BuildEvidence row under its placement semantics.

    ``placement`` decides which producer/receiver combination is coherent:

    * ``control_host`` — a fresh producer build on the control host (local Metal,
      or the prebuilt TSM module).  It is allowed to report ``ok`` there; the
      remote NPU rule (a received prebuilt artifact is ``reused``) does not apply
      to the machine that actually compiled it.
    * ``evaluator`` — a fresh build inside an attempt.  A *remote* row (the
      signed result) must also show what was loaded; a local producer row is the
      builder's own outcome and need not.
    * ``received`` — an evaluator-side row for a prebuilt artifact set.  It must
      be ``reused`` (never ``ok``) and must repeat the producer's
      build-evidence link unchanged.

    The classification is shared by every placement: a build that failed to
    compile is ``compile_failed``/``compile`` and may only accompany a
    ``compile_error`` report; a timeout that never built is ``not_run``.  A
    ``prebuilt`` row must reuse the artifact set named by the accepted request,
    and its ``loaded`` list must cover every executable entry, so a vendor
    cannot load other bytes and still claim the candidate.
    """

    if not isinstance(value, Mapping):
        raise ContractError("build evidence fields are missing or unknown")
    if not _BUILD_EVIDENCE_FIELDS <= set(value) or set(value) - (_BUILD_EVIDENCE_FIELDS | {"producer"}):
        raise ContractError("build evidence fields are missing or unknown")
    if value["schema_id"] != NPU_BUILD_EVIDENCE_SCHEMA_ID:
        raise ContractError("build evidence schema_id is not supported by this contract version")
    mode = _text(value["mode"], "build.mode", max_bytes=32)
    if mode not in BUILD_MODES:
        raise ContractError("build.mode must be prebuilt or evaluator")
    status = _text(value["status"], "build.status", max_bytes=32)
    if status not in BUILD_STATUSES:
        raise ContractError("build.status is not a supported build outcome")
    if placement not in BUILD_PLACEMENTS:
        raise ContractError("build evidence placement is not supported")
    if placement == "control_host" and mode != "prebuilt":
        raise ContractError("a control-host producer build must be mode prebuilt")
    if placement == "evaluator" and mode != "evaluator":
        raise ContractError("an evaluator build must be mode evaluator")
    if placement == "received" and mode != "prebuilt":
        raise ContractError("a received prebuilt artifact must be mode prebuilt")
    error_class = value["error_class"]
    if error_class is not None:
        error_class = _text(error_class, "build.error_class", max_bytes=32)
        if error_class not in BUILD_ERROR_CLASSES:
            raise ContractError("build.error_class is not a supported build error class")
    provenance = build_provenance_document(value["provenance"], mode=mode)
    producer = value.get("producer")
    producer = None if producer is None else build_producer_record(producer)
    if expected_build is not None:
        expected = build_request_document(expected_build)
        if expected["mode"] != mode:
            raise ContractError("build evidence mode differs from the accepted request")
        for field in (
            "builder_id", "builder_version", "build_host", "build_profile",
            "toolchain", "compile_profile", "source_sha256",
        ):
            if provenance[field] != expected["provenance"][field]:
                raise ContractError(f"build evidence {field} differs from the accepted request")
        if mode == "prebuilt" and provenance["artifact_set_digest"] != expected["provenance"]["artifact_set_digest"]:
            raise ContractError(
                "prebuilt build evidence artifact_set_digest differs from the accepted request"
            )
        if mode == "evaluator" and provenance["artifact_set_digest"] is None and status == "ok":
            raise ContractError("a completed evaluator build requires an artifact_set_digest")
        if mode == "prebuilt" and status == "reused":
            if producer != expected["producer"]:
                raise ContractError(
                    "the received build evidence does not repeat the accepted producer record"
                )
    artifacts = _build_artifacts_document(value["artifacts"])
    loaded = _build_loaded_document(value["loaded"], artifacts)
    duration_ms = _integer(value["duration_ms"], "build.duration_ms", 0, MAX_BUILD_DURATION_MS)
    log = value["log"]
    if log is not None:
        if not isinstance(log, Mapping) or set(log) != _BUILD_LOG_FIELDS:
            raise ContractError("build.log fields are missing or unknown")
        log = {
            "sha256": _hex64(log["sha256"], "build.log.sha256"),
            "size_bytes": _integer(log["size_bytes"], "build.log.size_bytes", 0, MAX_SOURCE_BYTES),
        }
    if artifacts is not None:
        if provenance["artifact_set_digest"] != artifacts["artifact_set_digest"]:
            raise ContractError("build evidence artifacts do not add up to the provenance digest")
    if placement == "control_host":
        if status != "ok":
            raise ContractError("a control-host producer build must report a fresh compile as ok")
        if error_class is not None:
            raise ContractError("a successful build has no build error")
        if artifacts is None:
            raise ContractError("a successful build requires the produced artifact manifest")
        if producer is not None:
            raise ContractError("a producer build has no received producer record")
    elif placement == "evaluator" and not remote:
        if status != "ok":
            raise ContractError("a local evaluator producer build must report a fresh compile as ok")
        if error_class is not None:
            raise ContractError("a successful build has no build error")
        if artifacts is None:
            raise ContractError("a successful build requires the produced artifact manifest")
        if producer is not None:
            raise ContractError("a producer build has no received producer record")
    elif status == "reused":
        if mode != "prebuilt":
            raise ContractError("only a prebuilt artifact set can be reported as reused")
        if duration_ms < 1:
            # ``0`` is the contract's "this never ran" duration (a ``not_run``
            # row).  A receiver that verified and loaded an artifact set spent
            # its own time doing so, and must record it instead of reusing that
            # sentinel -- otherwise an all-zero row is indistinguishable from an
            # attempt that never touched the artifact.
            raise ContractError(
                "a received artifact set that was loaded must record its own duration"
            )
        if log is None:
            # The receiver records what it verified and loaded, exactly as it
            # records its own duration: a reused row with no log cannot show
            # that anything was read back from the channel.
            raise ContractError(
                "a received artifact set that was loaded must record its own log"
            )
        if producer is not None and log == producer["log"]:
            # The producer's log identity is already authenticated by the link
            # inside the request digest.  Repeating it byte for byte would claim
            # the producer's diagnostics as this attempt's own record and make
            # the two rows indistinguishable, so the receiver must log its own
            # verify/load pass instead.
            raise ContractError(
                "the receiver's log is the producer's log identity"
            )
        if error_class is not None:
            raise ContractError("a reused artifact set has no build error")
        if artifacts is None or provenance["artifact_set_digest"] is None:
            raise ContractError("a reused artifact set requires its manifest identity")
        if not loaded:
            raise ContractError("a reused artifact set requires the receiver's load record")
        if producer is None:
            raise ContractError("a reused artifact set requires the producer build-evidence link")
        if request_artifact is not None and ARTIFACT_BUNDLE in request_artifact:
            tried = request_artifact[ARTIFACT_BUNDLE]
            if artifacts["artifact_set_digest"] != tried["artifact_set_digest"]:
                raise ContractError(
                    "build evidence artifact set differs from the accepted prebuilt bundle"
                )
            tried_entries = [
                {key: entry[key] for key in ("path", "role", "media_type", "size_bytes", "sha256")}
                for entry in tried["entries"]
            ]
            if artifacts["entries"] != tried_entries:
                raise ContractError(
                    "build evidence entries differ from the accepted prebuilt bundle"
                )
        executable = {
            entry["path"]: entry["sha256"]
            for entry in artifacts["entries"] if entry["role"] == "EXECUTABLE_MODULE"
        }
        if not executable:
            raise ContractError("a prebuilt artifact set requires an executable module")
        missing = sorted(set(executable) - {item["path"] for item in loaded})
        if missing:
            raise ContractError(
                "the prebuilt artifact set was not fully loaded: " + ",".join(missing)
            )
    elif status == "ok":
        if mode != "evaluator":
            raise ContractError("only an evaluator build can report a fresh compile as ok")
        if error_class is not None:
            raise ContractError("a successful build has no build error")
        if artifacts is None:
            raise ContractError("a successful build requires the produced artifact manifest")
        if not loaded:
            raise ContractError("a successful build requires the receiver's load record")
        if producer is not None:
            raise ContractError("a reused producer row cannot accompany a fresh evaluator build")
    elif status == "compile_failed":
        if mode != "evaluator":
            raise ContractError("only an evaluator build can report a compile failure")
        if error_class != "compile":
            raise ContractError("a compile failure must be classified as compile")
        if artifacts is not None or provenance["artifact_set_digest"] is not None:
            raise ContractError("a failed build must not claim a produced artifact set")
        if loaded:
            raise ContractError("a failed build must not claim a loaded module")
        if producer is not None:
            raise ContractError("a failed build has no producer build-evidence link")
        if report_status is not None and report_status != "compile_error":
            raise ContractError("a compile failure requires a compile_error candidate report")
        if complete:
            raise ContractError("a failed build cannot also claim a completed device event")
    else:  # not_run
        if error_class is not None:
            raise ContractError("a build that never ran has no build error class")
        if artifacts is not None or loaded:
            raise ContractError("a build that never ran must not claim artifacts or loads")
        if producer is not None:
            raise ContractError("a build that never ran has no producer build-evidence link")
        if mode == "evaluator" and provenance["artifact_set_digest"] is not None:
            raise ContractError("an evaluator build that never ran cannot claim an artifact set")
        if complete:
            raise ContractError("a build that never ran cannot claim a completed device event")
    document = {
        "schema_id": NPU_BUILD_EVIDENCE_SCHEMA_ID,
        "mode": mode,
        "status": status,
        "error_class": error_class,
        "provenance": provenance,
        "artifacts": artifacts,
        "duration_ms": duration_ms,
        "log": log,
        "loaded": loaded,
    }
    if producer is not None:
        document["producer"] = producer
    return document


def build_producer_evidence_document(value: Any, *, placement: str) -> dict[str, Any]:
    """Validate a *producer's own* BuildEvidence (control host or evaluator).

    This is the local placement rule: the machine that compiled the artifact is
    allowed to report ``ok``, and it has not received anything from a producer.
    The remote NPU rule is unchanged and lives in
    :func:`npu_build_evidence_document`.
    """

    if placement not in {"control_host", "evaluator"}:
        raise ContractError("producer placement must be control_host or evaluator")
    return build_evidence_document(value, placement=placement, remote=False)


def npu_build_evidence_document(
    value: Any, *, expected_build: Mapping[str, Any] | None = None,
    request_artifact: Mapping[str, Any] | None = None,
    report_status: str | None = None, complete: bool | None = None,
) -> dict[str, Any]:
    """Validate the signed BuildEvidence of the remote NPU result.

    A prebuilt artifact received by the evaluator must be ``reused`` and must
    repeat the producer link named by the accepted request; an evaluator build
    must report its own fresh compile.  The remote rule is deliberately *not*
    relaxed for the control-host producer: that producer never sends this
    document as a received result.
    """

    mode = value.get("mode") if isinstance(value, Mapping) else None
    placement = "evaluator" if mode == "evaluator" else "received"
    return build_evidence_document(
        value, placement=placement, remote=True,
        expected_build=expected_build, request_artifact=request_artifact,
        report_status=report_status, complete=complete,
    )


def _build_request_consistency(
    build: Mapping[str, Any], *, execution: Mapping[str, Any], artifact: Mapping[str, Any],
    candidate: Mapping[str, Any], source: bytes, bytes_available: bool,
) -> None:
    """Cross-check the build document against the artifact branch and execution."""

    mode = build["mode"]
    provenance = build["provenance"]
    if mode == "prebuilt":
        if ARTIFACT_BUNDLE not in artifact:
            raise ContractError("a prebuilt build must submit the artifact set it built")
        if execution["task_family"] != "kernel_module":
            raise ContractError("a prebuilt executable module requires task_family kernel_module")
        bundle = artifact[ARTIFACT_BUNDLE]
        if provenance["artifact_set_digest"] != candidate["value"]:
            raise ContractError("build provenance artifact_set_digest differs from the CandidateDigest")
        if bundle["artifact_set_digest"] != provenance["artifact_set_digest"]:
            raise ContractError("build provenance artifact_set_digest differs from the submitted bundle")
        sources = [entry for entry in bundle["entries"] if entry["role"] == "SOURCE"]
        modules = [entry for entry in bundle["entries"] if entry["role"] == "EXECUTABLE_MODULE"]
        if len(sources) != 1:
            raise ContractError("a prebuilt artifact set requires exactly one SOURCE entry")
        if not modules:
            raise ContractError("a prebuilt artifact set requires an EXECUTABLE_MODULE entry")
        if provenance["source_sha256"] != sources[0]["sha256"]:
            raise ContractError("build provenance source_sha256 is not the bundled SOURCE digest")
        if source:
            raise ContractError("a prebuilt request must not embed inline source bytes")
        return
    if ARTIFACT_SOURCE not in artifact:
        raise ContractError("an evaluator build must submit the inline source it will build")
    if execution["task_family"] != "kernel":
        raise ContractError("an evaluator build requires task_family kernel")
    if provenance["source_sha256"] != candidate["value"]:
        raise ContractError("build provenance source_sha256 differs from the CandidateDigest")
    if bytes_available and not source:
        raise ContractError("an evaluator build requires the exact source bytes")
    if source and hashlib.sha256(source).hexdigest() != provenance["source_sha256"]:
        raise ContractError("build provenance source_sha256 is not the exact source bytes")


def _npu_bundle_request_fields(
    document: Mapping[str, Any], source: bytes, *, bytes_available: bool = True,
    build_contract: bool = False,
) -> dict[str, Any]:
    """Validate one bundle-contract request (source branch or bundle branch).

    ``bytes_available=False`` is the terminal-storage path: the source bytes are
    gone, so only the persisted descriptor is re-checked.  The digest is
    unaffected because the canonical document never contained the bytes.  A
    build-contract request additionally carries its BuildProvenance document,
    which is cross-checked against the artifact branch.
    """

    request = _common_request_fields(document)
    execution = (
        npu_build_execution_document(document["execution"]) if build_contract
        else npu_bundle_execution_document(document["execution"])
    )
    artifact = npu_artifact_document(document["artifact"], execution=execution)
    if ARTIFACT_BUNDLE in artifact:
        if source:
            raise ContractError("a bundle request must not embed inline source bytes")
        digest = candidate_digest_document(
            document["candidate_digest"], allowed_algorithms=(BUNDLE_DIGEST_ALGORITHM,),
        )
        if digest["value"] != artifact[ARTIFACT_BUNDLE]["artifact_set_digest"]:
            raise ContractError("candidate_digest does not match the bundle artifact_set_digest")
    else:
        descriptor = artifact[ARTIFACT_SOURCE]
        if bytes_available:
            if not source:
                raise ContractError("an inline source artifact requires exact source bytes")
            digest = candidate_digest_document(
                document["candidate_digest"], allowed_algorithms=("sha256",), source_bytes=source,
            )
            if descriptor["size_bytes"] != len(source):
                raise ContractError("artifact.source does not describe the exact source bytes")
        else:
            digest = candidate_digest_document(
                document["candidate_digest"], allowed_algorithms=("sha256",),
            )
        if descriptor["sha256"] != digest["value"]:
            raise ContractError("artifact.source does not describe the exact source bytes")
    request["candidate_digest"] = digest
    request["execution"] = execution
    request["artifact"] = artifact
    if build_contract:
        build = build_request_document(document["build"])
        _build_request_consistency(
            build, execution=execution, artifact=artifact, candidate=digest,
            source=source, bytes_available=bytes_available,
        )
        request["build"] = build
    return request



def _npu_blob_request_fields(document: Mapping[str, Any], source: bytes) -> dict[str, Any]:
    """Validate the blob-reference candidate reference (no inline bytes).

    The blob contract deliberately refuses inline source bytes: a candidate is
    either transferred through the content-addressed channel or it is not
    admissible under this contract.  Admission also requires the descriptor to
    agree with the declared media type and with the CandidateDigest.
    """

    if source:
        raise ContractError("the blob contract must not embed candidate source bytes")
    blob = blob_descriptor_document(document["blob_ref"])
    request = _common_request_fields(document)
    request["candidate_digest"] = candidate_digest_document(
        document["candidate_digest"], allowed_algorithms=(BLOB_DIGEST_ALGORITHM,), blob=blob,
    )
    request["execution"] = npu_execution_document(document["execution"])
    if request["execution"]["source_media_type"] != blob["media_type"]:
        raise ContractError("blob_ref.media_type differs from execution.source_media_type")
    request["blob_ref"] = blob
    return request


def request_digest_document(
    document: Mapping[str, Any], *, source_bytes: bytes | bytearray | memoryview,
    dialect: ContractDialect = LEGACY_DIALECT,
) -> dict[str, Any]:
    """Validate and return the complete versioned request digest document.

    The input must explicitly contain every execution field, including fields
    whose values are defaults.  Known transport/idempotency/local-only fields
    are excluded.  If ``source`` is present it must be byte-identical to the
    separately supplied ``source_bytes``; source text is never implicitly
    encoded and source bytes never enter the JSON document.
    """

    if not isinstance(document, Mapping):
        raise ContractError("request must be an object")
    if any(not isinstance(key, str) for key in document):
        raise ContractError("request field names must be strings")
    source = _source_bytes(source_bytes)
    if "source" in document and _source_bytes(document["source"]) != source:
        raise ContractError("request source and source_bytes differ")

    if dialect.npu_bundle:
        build_contract = bool(dialect.build_contract)
        fields = _NPU_BUILD_REQUEST_FIELDS if build_contract else _NPU_BUNDLE_REQUEST_FIELDS
        allowed = fields | _EXCLUDED_REQUEST_FIELDS | {"source"}
        unknown = set(document) - allowed
        if unknown:
            raise ContractError(f"unknown request fields: {sorted(unknown)}")
        missing = fields - set(document)
        if missing:
            raise ContractError(f"missing explicit request fields/defaults: {sorted(missing)}")
        return {
            "document_type": dialect.request_document_type,
            "document_version": dialect.document_version,
            "digest_algorithm": REQUEST_DIGEST_ALGORITHM,
            "request": _npu_bundle_request_fields(
                document, source, build_contract=build_contract,
            ),
        }

    if dialect.npu_blob:
        allowed = _NPU_BLOB_REQUEST_FIELDS | _EXCLUDED_REQUEST_FIELDS | {"source"}
        unknown = set(document) - allowed
        if unknown:
            raise ContractError(f"unknown request fields: {sorted(unknown)}")
        missing = _NPU_BLOB_REQUEST_FIELDS - set(document)
        if missing:
            raise ContractError(f"missing explicit request fields/defaults: {sorted(missing)}")
        return {
            "document_type": dialect.request_document_type,
            "document_version": dialect.document_version,
            "digest_algorithm": REQUEST_DIGEST_ALGORITHM,
            "request": _npu_blob_request_fields(document, source),
        }

    if not dialect.npu_source:
        allowed = _REQUEST_FIELD_SET | _EXCLUDED_REQUEST_FIELDS | {"source"}
        unknown = set(document) - allowed
        if unknown:
            raise ContractError(f"unknown request fields: {sorted(unknown)}")
        missing = _REQUEST_FIELD_SET - set(document)
        if missing:
            raise ContractError(f"missing explicit request fields/defaults: {sorted(missing)}")
        candidate_sha256 = hashlib.sha256(source).hexdigest()
        if document["candidate_sha256"] != candidate_sha256:
            raise ContractError("candidate_sha256 does not match the exact source bytes")
        request = _common_request_fields(document)
        request["candidate_sha256"] = candidate_sha256
        return {
            "document_type": dialect.request_document_type,
            "document_version": dialect.document_version,
            "digest_algorithm": REQUEST_DIGEST_ALGORITHM,
            "request": request,
        }

    allowed = _NPU_SOURCE_REQUEST_FIELDS | _EXCLUDED_REQUEST_FIELDS | {"source"}
    unknown = set(document) - allowed
    if unknown:
        raise ContractError(f"unknown request fields: {sorted(unknown)}")
    missing = _NPU_SOURCE_REQUEST_FIELDS - set(document)
    if missing:
        raise ContractError(f"missing explicit request fields/defaults: {sorted(missing)}")
    request = _common_request_fields(document)
    request["candidate_digest"] = candidate_digest_document(
        document["candidate_digest"], source_bytes=source,
    )
    request["execution"] = npu_execution_document(document["execution"])
    return {
        "document_type": dialect.request_document_type,
        "document_version": dialect.document_version,
        "digest_algorithm": REQUEST_DIGEST_ALGORITHM,
        "request": request,
    }


def npu_stored_request_document(
    document: Mapping[str, Any], *, dialect: ContractDialect = NPU_SOURCE_DIALECT,
) -> dict[str, Any]:
    """Validate a persisted NPU request whose source bytes are gone.

    A terminal job deletes its source, but the canonical request document never
    contained the bytes: it contains the CandidateDigest (and, for the blob
    contract, the content-addressed reference).  Rebuilding it from the stored
    document therefore still yields the exact digest the frontend accepted,
    while every public/execution field is validated again and the legacy
    ``candidate_sha256`` is rejected instead of being silently dropped.
    """

    if not dialect.npu_source:
        raise ContractError("only an NPU contract has a source-less stored request")
    if not isinstance(document, Mapping):
        raise ContractError("stored request must be an object")
    if any(not isinstance(key, str) for key in document):
        raise ContractError("stored request field names must be strings")
    fields = (
        (_NPU_BUILD_REQUEST_FIELDS if dialect.build_contract else _NPU_BUNDLE_REQUEST_FIELDS)
        if dialect.npu_bundle
        else _NPU_BLOB_REQUEST_FIELDS if dialect.npu_blob
        else _NPU_SOURCE_REQUEST_FIELDS
    )
    expected = fields | {"request_id"}
    if set(document) != expected:
        missing = sorted(expected - set(document))
        unknown = sorted(set(document) - expected)
        raise ContractError(f"stored request fields are missing or unknown: missing={missing} unknown={unknown}")
    # ``request_id`` is transport identity, so it is not part of the canonical
    # digest; it still has to be a well-formed identifier and the caller must
    # compare it with the durable job row.
    _identifier(document["request_id"], "request_id")
    if dialect.npu_bundle:
        # A bundle request is reconstructed from its artifact oneof: the
        # document never contained the inline bytes (only their size and
        # SHA-256), so the digest is still the one the frontend accepted.  A
        # bundle entry list rewritten after the fact no longer adds up to
        # ``artifact_set_digest`` and is refused inside the validator.
        if "blob_ref" in document:
            raise ContractError("the bundle contract does not accept a request-level blob reference")
        request = _npu_bundle_request_fields(
            document, b"", bytes_available=False, build_contract=bool(dialect.build_contract),
        )
    elif dialect.npu_blob:
        request = _npu_blob_request_fields(document, b"")
    else:
        request = _common_request_fields(document)
        request["candidate_digest"] = candidate_digest_document(document["candidate_digest"])
        request["execution"] = npu_execution_document(document["execution"])
    return {
        "document_type": dialect.request_document_type,
        "document_version": dialect.document_version,
        "digest_algorithm": REQUEST_DIGEST_ALGORITHM,
        "request": request,
    }


def npu_stored_request_digest(
    document: Mapping[str, Any], *, dialect: ContractDialect = NPU_SOURCE_DIALECT,
) -> str:
    """Return the digest of a source-less NPU stored request."""

    return _document_digest(npu_stored_request_document(document, dialect=dialect))


def request_digest(
    document: Mapping[str, Any], *, source_bytes: bytes | bytearray | memoryview,
    dialect: ContractDialect = LEGACY_DIALECT,
) -> str:
    """Return ``SHA256(RFC8785(versioned_request_document))`` as lowercase hex."""

    return _document_digest(
        request_digest_document(document, source_bytes=source_bytes, dialect=dialect)
    )


def receipt_id_document(
    *, request_id: str, attempt_id: str, request_digest: str,
    candidate_digest: Mapping[str, Any] | None = None,
    device_group: Mapping[str, Any] | None = None,
    dialect: ContractDialect = LEGACY_DIALECT,
) -> dict[str, Any]:
    """Return the versioned document binding a receipt to exactly one attempt."""

    if not dialect.npu_source:
        if candidate_digest is not None or device_group is not None:
            raise ContractError("the legacy receipt cannot carry NPU binding fields")
        return {
            "document_type": dialect.receipt_document_type,
            "document_version": dialect.document_version,
            "digest_algorithm": RECEIPT_DIGEST_ALGORITHM,
            "request_id": _identifier(request_id, "request_id"),
            "attempt_id": _identifier(attempt_id, "attempt_id"),
            "request_digest": _hex64(request_digest, "request_digest"),
        }
    if candidate_digest is None or device_group is None:
        raise ContractError("the NPU receipt requires a CandidateDigest and DeviceGroup binding")
    return {
        "document_type": dialect.receipt_document_type,
        "document_version": dialect.document_version,
        "digest_algorithm": RECEIPT_DIGEST_ALGORITHM,
        "request_id": _identifier(request_id, "request_id"),
        "attempt_id": _identifier(attempt_id, "attempt_id"),
        "request_digest": _hex64(request_digest, "request_digest"),
        "candidate_digest": candidate_digest_document(
            candidate_digest, allowed_algorithms=dialect.candidate_algorithms,
        ),
        "device_group": device_group_document(device_group),
    }


def attempt_receipt_id(
    *, request_id: str, attempt_id: str, request_digest: str,
    candidate_digest: Mapping[str, Any] | None = None,
    device_group: Mapping[str, Any] | None = None,
    dialect: ContractDialect = LEGACY_DIALECT,
) -> str:
    """Return the attempt-scoped SHA-256 receipt identifier."""

    return _document_digest(
        receipt_id_document(
            request_id=request_id, attempt_id=attempt_id, request_digest=request_digest,
            candidate_digest=candidate_digest, device_group=device_group, dialect=dialect,
        )
    )


def _blob_descriptor(value: Any, name: str, *, allow_empty: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _BLOB_FIELDS:
        raise ContractError(f"{name} must have exactly schema/media type/size/digest fields")
    size = _integer(value["size_bytes"], f"{name}.size_bytes", 0 if allow_empty else 1, MAX_BLOB_BYTES)
    return {
        "schema_id": _text(value["schema_id"], f"{name}.schema_id", max_bytes=160),
        "media_type": _text(value["media_type"], f"{name}.media_type", max_bytes=160),
        "size_bytes": size,
        "sha256": _hex64(value["sha256"], f"{name}.sha256"),
    }


def validate_blob(descriptor: Any, data: Any, *, allow_empty: bool = False) -> dict[str, Any]:
    """Validate an opaque blob descriptor against the exact received bytes.

    Descriptors are signed only after this check; callers must not rely on a
    report's self-declared size or digest. Returns the normalized descriptor.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ContractError("opaque blob data must be bytes-like")
    raw = bytes(data)
    normalized = _blob_descriptor(descriptor, "blob", allow_empty=allow_empty)
    if len(raw) != normalized["size_bytes"]:
        raise ContractError("opaque blob size does not match its descriptor")
    if hashlib.sha256(raw).hexdigest() != normalized["sha256"]:
        raise ContractError("opaque blob digest does not match its descriptor")
    return normalized


_NPU_PROVENANCE_FIELDS = _PROVENANCE_FIELDS | {
    "device_group", "artifact_kind", "launch_kind", "launch_subtype", "timing_sources",
}


def _provenance_document(
    value: Any, *, require_attempt: bool = True, dialect: ContractDialect = LEGACY_DIALECT,
) -> dict[str, Any]:
    expected_fields = _PROVENANCE_FIELDS if require_attempt else _PROVENANCE_FIELDS - {"attempt_id", "request_digest"}
    if dialect.npu_source:
        expected_fields = _NPU_PROVENANCE_FIELDS if require_attempt else _NPU_PROVENANCE_FIELDS - {"attempt_id", "request_digest"}
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ContractError("provenance fields are missing or unknown")
    schema_version = _integer(value["schema_version"], "schema_version", 1, 1)
    try:
        selected = dialect_for(value["api_contract_id"], value["evidence_contract_id"])
    except ContractError as exc:
        raise ContractError("provenance API/evidence contract is not supported") from exc
    if selected != dialect:
        raise ContractError("provenance contract differs from the configured dialect")
    measurement_kind = value["measurement_kind"]
    if measurement_kind not in _MEASUREMENT_KINDS:
        raise ContractError("unknown critical measurement_kind")
    document = {
        "schema_version": schema_version,
        "evaluator_epoch": _hex64(value["evaluator_epoch"], "evaluator_epoch"),
        "site_id": _identifier(value["site_id"], "site_id"),
        "device_id": _identifier(value["device_id"], "device_id"),
        "backend": _text(value["backend"], "backend", max_bytes=64),
        "backend_version": _text(value["backend_version"], "backend_version"),
        "target_id": _identifier(value["target_id"], "target_id"),
        "worker_build": _text(value["worker_build"], "worker_build"),
        "task_pack_digest": _hex64(value["task_pack_digest"], "task_pack_digest"),
        "calibration_digest": _hex64(value["calibration_digest"], "calibration_digest"),
        "protocol_version": _text(value["protocol_version"], "protocol_version", max_bytes=64),
        "api_contract_id": dialect.api_contract_id,
        "evidence_contract_id": dialect.evidence_contract_id,
        "measurement_kind": measurement_kind,
        "publication_allowed": _boolean(value["publication_allowed"], "publication_allowed"),
    }
    if dialect.npu_source:
        document["device_group"] = device_group_document(value["device_group"])
        document["artifact_kind"] = _text(value["artifact_kind"], "artifact_kind", max_bytes=32)
        document["launch_kind"] = _text(value["launch_kind"], "launch_kind", max_bytes=32)
        document["launch_subtype"] = _text(value["launch_subtype"], "launch_subtype", max_bytes=32)
        timing_sources = value["timing_sources"]
        if not isinstance(timing_sources, (list, tuple)) or not timing_sources:
            raise ContractError("provenance timing_sources must be a non-empty array")
        sources = [_text(item, "timing_source", max_bytes=32) for item in timing_sources]
        if any(item not in _NPU_TIMING_SOURCES for item in sources):
            raise ContractError("provenance timing source is not supported by this contract")
        document["timing_sources"] = sorted(set(sources))
        shape = (
            document["artifact_kind"], document["launch_kind"], document["launch_subtype"],
        )
        families = _TASK_FAMILIES_BUILD if dialect.build_contract else _TASK_FAMILIES
        allowed_shapes = (
            {shape for shape in families.values()} if dialect.npu_bundle
            else {("source", "KERNEL", "SIMT")}
        )
        if shape not in {tuple(item[:3]) for item in allowed_shapes}:
            raise ContractError("provenance execution shape is not supported by this contract")
    if require_attempt:
        document["attempt_id"] = _identifier(value["attempt_id"], "attempt_id")
        document["request_digest"] = _hex64(value["request_digest"], "request_digest")
    return document


def provenance_document(
    value: Any, *, require_attempt: bool = True, dialect: ContractDialect = LEGACY_DIALECT,
) -> dict[str, Any]:
    """Validate one typed provenance document.

    Capabilities omit ``attempt_id``/``request_digest``; jobs and results must
    include both.  The returned mapping uses the same field order as the
    canonical signature document.
    """

    return _provenance_document(value, require_attempt=require_attempt, dialect=dialect)


def receipt_document(
    value: Any, *, dialect: ContractDialect = LEGACY_DIALECT,
) -> dict[str, Any]:
    """Validate and return one attempt-scoped receipt document."""

    return _receipt_document(value, dialect=dialect)


def _receipt_document(
    value: Any, *, dialect: ContractDialect = LEGACY_DIALECT,
) -> dict[str, Any]:
    expected_fields = _RECEIPT_FIELDS if not dialect.npu_source else (
        (_RECEIPT_FIELDS - {"candidate_sha256"}) | {"candidate_digest", "device_group"}
    )
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ContractError("receipt fields are missing or unknown")
    receipt = {
        "receipt_id": _hex64(value["receipt_id"], "receipt_id"),
        "request_id": _identifier(value["request_id"], "request_id"),
        "attempt_id": _identifier(value["attempt_id"], "attempt_id"),
        "request_digest": _hex64(value["request_digest"], "request_digest"),
        "lease_fence": _decimal_uint(value["lease_fence"], "lease_fence", minimum=1),
        "session_sequence": _decimal_uint(value["session_sequence"], "session_sequence"),
        "issued_at": _text(value["issued_at"], "issued_at", max_bytes=40),
    }
    if dialect.npu_source:
        receipt["candidate_digest"] = candidate_digest_document(
            value["candidate_digest"], allowed_algorithms=dialect.candidate_algorithms,
        )
        receipt["device_group"] = device_group_document(value["device_group"])
    else:
        receipt["candidate_sha256"] = _hex64(value["candidate_sha256"], "candidate_sha256")
    if not _UTC.fullmatch(receipt["issued_at"]):
        raise ContractError("issued_at must be an RFC 3339 UTC timestamp")
    try:
        datetime.strptime(receipt["issued_at"][:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError as exc:
        raise ContractError("issued_at must be a valid RFC 3339 UTC timestamp") from exc
    expected = attempt_receipt_id(
        request_id=receipt["request_id"],
        attempt_id=receipt["attempt_id"],
        request_digest=receipt["request_digest"],
        candidate_digest=receipt.get("candidate_digest"),
        device_group=receipt.get("device_group"),
        dialect=dialect,
    )
    if not hmac.compare_digest(receipt["receipt_id"], expected):
        raise ContractError("receipt_id does not bind the request, attempt, candidate and device group")
    return receipt


def result_signature_document(
    document: Mapping[str, Any], *, key_id: str, dialect: ContractDialect = LEGACY_DIALECT,
) -> dict[str, Any]:
    """Validate a result envelope and return its versioned HMAC input document.

    Report/evidence bytes are represented by their validated descriptors.  The
    receiver must independently check each descriptor against the opaque bytes.
    The complete provenance and receipt, not only their identifiers, are signed.
    """

    if not isinstance(document, Mapping) or set(document) != _RESULT_FIELDS:
        raise ContractError("signed result must contain exactly provenance/report/evidence/receipt")
    provenance = _provenance_document(document["provenance"], dialect=dialect)
    report = _blob_descriptor(document["report"], "report", allow_empty=False)
    evidence_value = document["evidence"]
    evidence = None if evidence_value is None else _blob_descriptor(evidence_value, "evidence", allow_empty=True)
    if report["schema_id"] != REPORT_SCHEMA_ID:
        raise ContractError("result report descriptor is not the candidate report schema")
    if evidence is not None and dialect.enforce_evidence_identity:
        # A contract that fixes the evidence identity refuses a descriptor that
        # names another schema (for example the version-1 id on a version-2
        # document) or another media type, on both ends of the chain.
        if (
            evidence["schema_id"] != dialect.evidence_schema_id
            or evidence["media_type"] != dialect.evidence_media_type
        ):
            raise ContractError(
                "result evidence descriptor is not the TimingEvidence identity "
                "this contract version signs"
            )
    receipt = _receipt_document(document["receipt"], dialect=dialect)
    if provenance["attempt_id"] != receipt["attempt_id"]:
        raise ContractError("provenance and receipt attempt_id differ")
    if provenance["request_digest"] != receipt["request_digest"]:
        raise ContractError("provenance and receipt request_digest differ")
    if dialect.npu_source and provenance["device_group"] != receipt["device_group"]:
        raise ContractError("provenance and receipt device groups differ")
    return {
        "document_type": dialect.result_document_type,
        "document_version": dialect.document_version,
        "signature_algorithm": RESULT_SIGNATURE_ALGORITHM,
        "key_id": _key_id(key_id),
        "provenance": provenance,
        "report": report,
        "evidence": evidence,
        "receipt": receipt,
    }


def _signature_key(key: Any) -> bytes:
    if not isinstance(key, (bytes, bytearray, memoryview)):
        raise ContractError("an HMAC key must be bytes")
    key_bytes = bytes(key)
    if len(key_bytes) < 32:
        raise ContractError("an HMAC-SHA256 key must contain at least 32 bytes")
    return key_bytes


def sign_result(
    document: Mapping[str, Any], *, key_id: str, key: bytes,
    dialect: ContractDialect = LEGACY_DIALECT,
) -> str:
    """Return lowercase-hex HMAC-SHA256 over the canonical result document."""

    payload = canonical_digest_document(
        result_signature_document(document, key_id=key_id, dialect=dialect)
    )
    return hmac.new(_signature_key(key), payload, hashlib.sha256).hexdigest()


def verify_result_signature(
    document: Mapping[str, Any], *, key_id: str, key: bytes, signature: str,
    dialect: ContractDialect = LEGACY_DIALECT,
) -> bool:
    """Verify a signature and its key identifier without accepting malformed hex."""

    if not isinstance(signature, str) or not _HEX64.fullmatch(signature):
        return False
    expected = sign_result(document, key_id=key_id, key=key, dialect=dialect)
    return hmac.compare_digest(signature, expected)


_NPU_TIMING_FIELDS = frozenset({
    "source", "clock_domain", "synchronization_completed",
    "device_elapsed_ns", "host_api_elapsed_ns", "launches",
})
_NPU_EVIDENCE_FIELDS = frozenset({
    "schema_id", "candidate_digest", "device_group", "timing",
    "measurement_kind", "publication_allowed", "complete",
})


#: One microsecond of tolerance when matching a report's `candidate_us` against
#: the device-event elapsed time (a driver may round the reported microseconds).
NPU_CANDIDATE_US_TOLERANCE = 1e-3


def npu_report_timing_check(evidence: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    """Require an ``ok`` NPU report's candidate time to be the device event time.

    Host launch time is recorded separately, so a report that claims ``ok`` must
    carry the device-event elapsed time as its candidate time.  Diagnostic rows
    (timeout, compile error, ...) are not constrained here; their evidence is
    validated by :func:`npu_timing_evidence_document`.
    """

    if not isinstance(report, Mapping) or report.get("status") != "ok":
        return
    timing = evidence.get("timing") if isinstance(evidence, Mapping) else None
    if not isinstance(timing, Mapping):
        raise ContractError("an ok report requires timing evidence")
    device_elapsed = timing.get("device_elapsed_ns")
    if isinstance(device_elapsed, bool) or not isinstance(device_elapsed, int) or device_elapsed <= 0:
        raise ContractError("an ok report requires a positive device-event elapsed time")
    candidate_us = report.get("candidate_us")
    if isinstance(candidate_us, bool) or not isinstance(candidate_us, (int, float)):
        raise ContractError("an ok report requires a numeric candidate_us")
    if not math.isfinite(float(candidate_us)):
        raise ContractError("an ok report requires a finite candidate_us")
    if abs(float(candidate_us) - device_elapsed / 1000.0) > NPU_CANDIDATE_US_TOLERANCE:
        raise ContractError(
            "report candidate_us does not come from the device-event elapsed time "
            f"({candidate_us} us vs {device_elapsed} ns)"
        )


def npu_timing_evidence_document(
    value: Any, *, source_bytes: bytes | bytearray | memoryview | None = None,
    device_group: Mapping[str, Any] | None = None,
    measurement_kind: str | None = None,
    publication_allowed: bool | None = None,
    report_status: str | None = None,
    dialect: ContractDialect = NPU_SOURCE_DIALECT,
) -> dict[str, Any]:
    """Validate the P0 minimal TimingEvidence document.

    Device and host time are deliberately separate fields.  A simulated result
    can never claim publication, and a known clock domain is required.  P0 has
    no rank/die/profiler fields, so their absence is not treated as missing.

    ``complete`` separates a real device observation from honest diagnostic
    evidence: a report that claims ``ok`` needs a completed device event, and
    evidence that never synchronized must not claim one.  A worker killed by the
    durable deadline therefore reports ``status=timeout`` with
    ``complete=false``, zero device time, no launch and no publish permission,
    instead of inventing a completed device event.
    """

    if not isinstance(value, Mapping) or not _NPU_EVIDENCE_FIELDS <= set(value):
        raise ContractError("timing evidence fields are missing")
    if set(value) - (_NPU_EVIDENCE_FIELDS | {"diagnostics"}):
        raise ContractError("timing evidence has unknown fields")
    if value["schema_id"] != NPU_TIMING_EVIDENCE_SCHEMA_ID:
        raise ContractError("timing evidence schema_id is not supported by P0")
    candidate = candidate_digest_document(
        value["candidate_digest"], source_bytes=source_bytes,
        allowed_algorithms=dialect.candidate_algorithms,
    )
    group = device_group_document(value["device_group"])
    if device_group is not None and group != device_group_document(device_group):
        raise ContractError("timing evidence device group differs from the accepted capability")
    complete = _boolean(value["complete"], "complete")
    timing = value["timing"]
    if not isinstance(timing, Mapping) or set(timing) != _NPU_TIMING_FIELDS:
        raise ContractError("timing fields are missing or unknown")
    source = _text(timing["source"], "timing.source", max_bytes=32)
    if source != "DEVICE_EVENT":
        raise ContractError("P0 timing source must be DEVICE_EVENT")
    clock_domain = _text(timing["clock_domain"], "timing.clock_domain", max_bytes=96)
    synchronized = _boolean(timing["synchronization_completed"], "timing.synchronization_completed")
    device_elapsed = _integer(timing["device_elapsed_ns"], "timing.device_elapsed_ns", 0, _MAX_SAFE_INTEGER)
    host_elapsed = _integer(timing["host_api_elapsed_ns"], "timing.host_api_elapsed_ns", 0, _MAX_SAFE_INTEGER)
    launches = _integer(timing["launches"], "timing.launches", 0, 1_000_000)
    kind = value["measurement_kind"]
    if kind not in _MEASUREMENT_KINDS:
        raise ContractError("timing evidence measurement_kind is invalid")
    publish = _boolean(value["publication_allowed"], "publication_allowed")
    if kind == "simulated" and publish:
        raise ContractError("simulated timing evidence cannot be publishable")
    if complete:
        if not synchronized:
            raise ContractError("timing evidence requires completed synchronization")
        if launches < 1:
            raise ContractError("completed timing evidence requires at least one launch")
        if device_elapsed <= 0:
            raise ContractError("completed timing evidence requires a positive device elapsed time")
        if clock_domain == NPU_UNOBSERVED_CLOCK_DOMAIN:
            raise ContractError("completed timing evidence requires an observed clock domain")
    else:
        # No device event was observed: the row may only be diagnostic.
        if synchronized:
            raise ContractError("incomplete timing evidence cannot claim completed synchronization")
        if launches or device_elapsed:
            raise ContractError("incomplete timing evidence cannot report device time or a launch")
        if clock_domain != NPU_UNOBSERVED_CLOCK_DOMAIN:
            raise ContractError("incomplete timing evidence must name the unobserved clock domain")
        if publish:
            raise ContractError("incomplete timing evidence cannot be publishable")
    if measurement_kind is not None and kind != measurement_kind:
        raise ContractError("timing evidence measurement kind differs from provenance")
    if publication_allowed is not None and publish != publication_allowed and complete:
        raise ContractError("timing evidence publication permission differs from provenance")
    if report_status is not None:
        if report_status == "ok" and not complete:
            raise ContractError("an ok report requires completed device timing evidence")
        if report_status == "timeout" and complete:
            raise ContractError("a timeout report cannot claim a completed device event")
    document: dict[str, Any] = {
        "schema_id": NPU_TIMING_EVIDENCE_SCHEMA_ID,
        "candidate_digest": candidate,
        "device_group": group,
        "timing": {
            "source": source,
            "clock_domain": clock_domain,
            "synchronization_completed": synchronized,
            "device_elapsed_ns": device_elapsed,
            "host_api_elapsed_ns": host_elapsed,
            "launches": launches,
        },
        "measurement_kind": kind,
        "publication_allowed": publish,
        "complete": complete,
    }
    if "diagnostics" in value:
        diagnostics = value["diagnostics"]
        if not isinstance(diagnostics, Mapping) or len(diagnostics) > 32:
            raise ContractError("timing diagnostics must be a bounded object")
        for key, item in diagnostics.items():
            if not isinstance(key, str) or not key or not key.isascii() or len(key) > 64:
                raise ContractError("timing diagnostic key is invalid")
            if isinstance(item, str):
                _text(item, "timing diagnostic", max_bytes=512, empty=True)
            elif isinstance(item, bool):
                pass
            elif isinstance(item, int):
                if not -_MAX_SAFE_INTEGER <= item <= _MAX_SAFE_INTEGER:
                    raise ContractError("timing diagnostic integer is unsafe")
            elif item is not None:
                raise ContractError("timing diagnostics must be scalar")
        document["diagnostics"] = dict(diagnostics)
    return document


# -- P1 TimingEvidence version 2 -------------------------------------------
#: Version 2 keeps the version-1 host/device split and adds the declared
#: topology, the per-device spans and the buffer completeness that a
#: completeness decision needs.  It is a *new* schema: version 1 (P0/P0b) is
#: unchanged, so old evidence validates byte-for-byte as before.
_NPU_V2_TOPOLOGY_FIELDS = frozenset({
    "logical_devices", "declared_devices", "required_rank_spans", "required_die_spans",
})
_NPU_V2_SPAN_FIELDS = frozenset({"device_id", "rank", "die", "clock_domain", "elapsed_ns"})
_NPU_V2_BUFFER_FIELDS = frozenset({
    "flushed", "records_written", "records_delivered", "dropped", "overwritten",
})
_NPU_V2_EVIDENCE_FIELDS = frozenset({
    "schema_id", "candidate_digest", "artifact_set_digest", "device_group", "topology",
    "spans", "timing", "buffer", "measurement_kind", "publication_allowed", "complete",
})
MAX_EVIDENCE_SPANS = 64


def _coverable_shape(
    logical_devices: int, rank_spans: int | None, die_spans: int | None,
) -> int:
    """Return the number of ``device x rank x die`` spans a topology needs.

    Completeness is a coordinate question, not a count: one span per
    ``(device_id, rank, die)`` combination the declared topology names.  A shape
    whose product cannot fit in one bounded evidence array is inadmissible, so a
    request can never declare a topology that no vendor could ever satisfy.
    """

    needed = logical_devices * (rank_spans or 1) * (die_spans or 1)
    if needed > MAX_EVIDENCE_SPANS:
        raise ContractError(
            f"the declared topology needs {needed} device x rank x die spans; "
            f"one evidence row carries at most {MAX_EVIDENCE_SPANS}"
        )
    return needed


def _topology_document(value: Any) -> dict[str, Any]:
    """Validate the declared topology an evidence row claims to cover."""

    if not isinstance(value, Mapping) or set(value) != _NPU_V2_TOPOLOGY_FIELDS:
        raise ContractError("timing evidence topology fields are missing or unknown")
    logical_devices = _integer(
        value["logical_devices"], "topology.logical_devices", 1, MAX_LOGICAL_DEVICES,
    )
    declared = value["declared_devices"]
    if not isinstance(declared, (list, tuple)) or not declared or len(declared) > MAX_EVIDENCE_SPANS:
        raise ContractError("topology.declared_devices must be a bounded non-empty array")
    devices = [_identifier(item, "topology.declared_devices") for item in declared]
    if len(set(devices)) != len(devices):
        raise ContractError("topology.declared_devices must be unique")
    if len(devices) != logical_devices:
        raise ContractError("topology.declared_devices disagrees with logical_devices")
    document: dict[str, Any] = {
        "logical_devices": logical_devices,
        "declared_devices": sorted(devices),
    }
    for name in ("required_rank_spans", "required_die_spans"):
        raw = value[name]
        document[name] = (
            None if raw is None
            else _integer(raw, f"topology.{name}", 1, MAX_DECLARED_SPANS)
        )
    _coverable_shape(
        logical_devices, document["required_rank_spans"], document["required_die_spans"],
    )
    return document


def _span_documents(value: Any, topology: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the spans one measurement reports.

    A span is one ``(device_id, rank, die)`` observation, so one device with two
    ranks reports two spans.  The identity is the whole coordinate: a duplicated
    coordinate is refused, an undeclared device is refused, and a vendor shape
    without a rank/die dimension must report ``null`` instead of inventing one.
    """

    if not isinstance(value, (list, tuple)) or len(value) > MAX_EVIDENCE_SPANS:
        raise ContractError("timing evidence spans must be a bounded array")
    declared = set(topology["declared_devices"])
    spans: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _NPU_V2_SPAN_FIELDS:
            raise ContractError("timing evidence span fields are missing or unknown")
        device_id = _identifier(item["device_id"], "span.device_id")
        if device_id not in declared:
            raise ContractError("a span names a device the declared topology does not have")
        rank = item["rank"]
        die = item["die"]
        if rank is not None:
            rank = _integer(rank, "span.rank", 0, MAX_DECLARED_SPANS - 1)
        if die is not None:
            die = _integer(die, "span.die", 0, MAX_DECLARED_SPANS - 1)
        required_ranks = topology["required_rank_spans"]
        required_dies = topology["required_die_spans"]
        if required_ranks is None and rank is not None:
            # A vendor shape without a rank dimension must not invent ranks: the
            # field is not required, and fabricating it would fake topology
            # evidence the trusted profile never declared.
            raise ContractError("a span must not report a rank the topology does not declare")
        if required_dies is None and die is not None:
            raise ContractError("a span must not report a die the topology does not declare")
        if required_ranks is not None and rank is not None and rank >= required_ranks:
            raise ContractError("a span reports a rank the declared topology does not cover")
        if required_dies is not None and die is not None and die >= required_dies:
            raise ContractError("a span reports a die the declared topology does not cover")
        coordinate = (device_id, rank, die)
        if coordinate in seen:
            raise ContractError(
                "timing evidence reports the same device/rank/die coordinate twice"
            )
        seen.add(coordinate)
        spans.append({
            "device_id": device_id,
            "rank": rank,
            "die": die,
            "clock_domain": _text(item["clock_domain"], "span.clock_domain", max_bytes=96),
            "elapsed_ns": _integer(item["elapsed_ns"], "span.elapsed_ns", 0, _MAX_SAFE_INTEGER),
        })
    return spans


def _buffer_document(value: Any) -> dict[str, Any] | None:
    """Validate the optional profiler/event buffer completeness block."""

    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _NPU_V2_BUFFER_FIELDS:
        raise ContractError("timing evidence buffer fields are missing or unknown")
    document = {
        "flushed": _boolean(value["flushed"], "buffer.flushed"),
        "records_written": _integer(value["records_written"], "buffer.records_written", 0, 1_000_000),
        "records_delivered": _integer(
            value["records_delivered"], "buffer.records_delivered", 0, 1_000_000,
        ),
        "dropped": _integer(value["dropped"], "buffer.dropped", 0, 1_000_000),
        "overwritten": _integer(value["overwritten"], "buffer.overwritten", 0, 1_000_000),
    }
    return document


def npu_timing_evidence_document_v2(
    value: Any, *, source_bytes: bytes | bytearray | memoryview | None = None,
    device_group: Mapping[str, Any] | None = None,
    expected_artifact_set_digest: str | None = None,
    expected_resources: Mapping[str, Any] | None = None,
    measurement_kind: str | None = None,
    publication_allowed: bool | None = None,
    report_status: str | None = None,
    dialect: ContractDialect = NPU_BUNDLE_DIALECT,
) -> dict[str, Any]:
    """Validate P1 TimingEvidence (schema version 2)."""

    return _npu_timing_evidence_document(
        value, source_bytes=source_bytes, device_group=device_group,
        expected_artifact_set_digest=expected_artifact_set_digest,
        expected_resources=expected_resources, measurement_kind=measurement_kind,
        publication_allowed=publication_allowed, report_status=report_status,
        dialect=dialect, schema_id=NPU_BUNDLE_TIMING_EVIDENCE_SCHEMA_ID,
    )


def npu_build_timing_evidence_document(
    value: Any, *, source_bytes: bytes | bytearray | memoryview | None = None,
    device_group: Mapping[str, Any] | None = None,
    expected_artifact_set_digest: str | None = None,
    expected_resources: Mapping[str, Any] | None = None,
    expected_build: Mapping[str, Any] | None = None,
    request_artifact: Mapping[str, Any] | None = None,
    measurement_kind: str | None = None,
    publication_allowed: bool | None = None,
    report_status: str | None = None,
    dialect: ContractDialect = NPU_BUILD_DIALECT,
) -> dict[str, Any]:
    """Validate P1c TimingEvidence (schema version 3: version 2 plus BuildEvidence).

    The timing rules are the version-2 rules; the build document is the shared
    BuildEvidence contract, and it must agree with the accepted request's build
    provenance and artifact branch.
    """

    return _npu_timing_evidence_document(
        value, source_bytes=source_bytes, device_group=device_group,
        expected_artifact_set_digest=expected_artifact_set_digest,
        expected_resources=expected_resources, measurement_kind=measurement_kind,
        publication_allowed=publication_allowed, report_status=report_status,
        dialect=dialect, schema_id=NPU_BUILD_TIMING_EVIDENCE_SCHEMA_ID,
        expected_build=expected_build, request_artifact=request_artifact,
    )


def _npu_timing_evidence_document(
    value: Any, *, source_bytes: bytes | bytearray | memoryview | None = None,
    device_group: Mapping[str, Any] | None = None,
    expected_artifact_set_digest: str | None = None,
    expected_resources: Mapping[str, Any] | None = None,
    measurement_kind: str | None = None,
    publication_allowed: bool | None = None,
    report_status: str | None = None,
    dialect: ContractDialect = NPU_BUNDLE_DIALECT,
    schema_id: str = NPU_BUNDLE_TIMING_EVIDENCE_SCHEMA_ID,
    expected_build: Mapping[str, Any] | None = None,
    request_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the artifact-set TimingEvidence document (schema 2 or 3).

    Completeness is decided by the *declared* topology, not by a fixed field
    list: a single-device or service-style vendor with no rank/die dimension is
    complete without rank/die spans, while a topology that declares them must
    report every one.  A missing required span, an unknown clock domain, an
    incomplete buffer or a contradictory aggregate makes the row diagnostic
    only: ``complete=False`` and ``publication_allowed=False``.

    Host launch time stays in ``host_api_elapsed_ns`` and is never used as the
    device elapsed time.  ``device_elapsed_ns`` must equal the aggregate the
    spans define, which is the slowest device span (the completion time of the
    measurement, not an average).
    """

    build_expected = expected_build is not None or schema_id == NPU_BUILD_TIMING_EVIDENCE_SCHEMA_ID
    allowed_extra = {"diagnostics"} | ({"build"} if build_expected else set())
    if not isinstance(value, Mapping) or not _NPU_V2_EVIDENCE_FIELDS <= set(value):
        raise ContractError("timing evidence fields are missing")
    if set(value) - (_NPU_V2_EVIDENCE_FIELDS | allowed_extra):
        raise ContractError("timing evidence has unknown fields")
    if value["schema_id"] != schema_id:
        raise ContractError("timing evidence schema_id is not supported by this contract version")
    if build_expected and "build" not in value:
        raise ContractError("this timing evidence version requires build evidence")
    candidate = candidate_digest_document(
        value["candidate_digest"], source_bytes=source_bytes,
        allowed_algorithms=dialect.candidate_algorithms,
    )
    group = device_group_document(value["device_group"])
    if device_group is not None and group != device_group_document(device_group):
        raise ContractError("timing evidence device group differs from the accepted capability")
    artifact_set_digest = value["artifact_set_digest"]
    if candidate["algorithm"] == BUNDLE_DIGEST_ALGORITHM:
        if not isinstance(artifact_set_digest, str):
            raise ContractError("a bundle candidate requires artifact_set_digest evidence")
        _hex64(artifact_set_digest, "artifact_set_digest")
        if artifact_set_digest != candidate["value"]:
            raise ContractError("artifact_set_digest differs from the candidate digest")
    elif artifact_set_digest is not None:
        raise ContractError("an inline source candidate must not claim an artifact_set_digest")
    if (
        expected_artifact_set_digest is not None
        and artifact_set_digest != expected_artifact_set_digest
    ):
        raise ContractError("timing evidence artifact_set_digest differs from the accepted artifact")
    topology = _topology_document(value["topology"])
    if expected_resources is not None:
        declared = device_group_requirements(expected_resources)
        if topology["logical_devices"] != declared["logical_devices"] or (
            topology["required_rank_spans"] != declared["required_rank_spans"]
            or topology["required_die_spans"] != declared["required_die_spans"]
        ):
            raise ContractError(
                "timing evidence topology differs from the declared execution resources"
            )
    spans = _span_documents(value["spans"], topology)
    complete = _boolean(value["complete"], "complete")
    timing = value["timing"]
    if not isinstance(timing, Mapping) or set(timing) != _NPU_TIMING_FIELDS:
        raise ContractError("timing fields are missing or unknown")
    source = _text(timing["source"], "timing.source", max_bytes=32)
    if source not in _NPU_BUNDLE_TIMING_SOURCES:
        raise ContractError("P1 timing source must be DEVICE_EVENT")
    clock_domain = _text(timing["clock_domain"], "timing.clock_domain", max_bytes=96)
    synchronized = _boolean(timing["synchronization_completed"], "timing.synchronization_completed")
    device_elapsed = _integer(timing["device_elapsed_ns"], "timing.device_elapsed_ns", 0, _MAX_SAFE_INTEGER)
    host_elapsed = _integer(timing["host_api_elapsed_ns"], "timing.host_api_elapsed_ns", 0, _MAX_SAFE_INTEGER)
    launches = _integer(timing["launches"], "timing.launches", 0, 1_000_000)
    kind = value["measurement_kind"]
    if kind not in _MEASUREMENT_KINDS:
        raise ContractError("timing evidence measurement_kind is invalid")
    publish = _boolean(value["publication_allowed"], "publication_allowed")
    buffer_document = _buffer_document(value["buffer"])
    if kind == "simulated" and publish:
        raise ContractError("simulated timing evidence cannot be publishable")
    if complete:
        if not synchronized:
            raise ContractError("timing evidence requires completed synchronization")
        if launches < 1:
            raise ContractError("completed timing evidence requires at least one launch")
        if device_elapsed <= 0:
            raise ContractError("completed timing evidence requires a positive device elapsed time")
        if clock_domain == NPU_UNOBSERVED_CLOCK_DOMAIN:
            raise ContractError("completed timing evidence requires an observed clock domain")
        # Completeness is exact coverage of the declared topology's coordinates:
        # every declared device, times every declared rank, times every declared
        # die.  A shape without a rank/die dimension needs exactly one span per
        # device and must not be penalised for the dimension it does not have.
        required_ranks = topology["required_rank_spans"]
        required_dies = topology["required_die_spans"]
        expected = {
            (device, rank, die)
            for device in topology["declared_devices"]
            for rank in (range(required_ranks) if required_ranks is not None else (None,))
            for die in (range(required_dies) if required_dies is not None else (None,))
        }
        reported = {(span["device_id"], span["rank"], span["die"]) for span in spans}
        missing_devices = sorted(
            set(topology["declared_devices"]) - {span["device_id"] for span in spans}
        )
        if missing_devices:
            raise ContractError(
                "completed timing evidence is missing a declared device span: "
                + ",".join(missing_devices)
            )
        missing_ranks = sorted(coordinate for coordinate in expected - reported if coordinate[1] is not None)
        if missing_ranks:
            raise ContractError(
                "completed timing evidence is missing a declared rank span: "
                + ",".join(f"{device}/rank {rank}" for device, rank, _die in missing_ranks)
            )
        missing_dies = sorted(coordinate for coordinate in expected - reported if coordinate[2] is not None)
        if missing_dies:
            raise ContractError(
                "completed timing evidence is missing a declared die span: "
                + ",".join(
                    f"{device}/die {die}" for device, _rank, die in missing_dies
                )
            )
        if reported - expected:
            raise ContractError(
                "completed timing evidence reports a span the declared topology does not cover"
            )
        for span in spans:
            if span["clock_domain"] == NPU_UNOBSERVED_CLOCK_DOMAIN:
                raise ContractError("a completed span requires an observed clock domain")
            if span["clock_domain"] != clock_domain:
                raise ContractError("all spans of one measurement must share the reported clock domain")
        aggregate = max(span["elapsed_ns"] for span in spans)
        if aggregate != device_elapsed:
            raise ContractError(
                "device_elapsed_ns is not the aggregate of the declared device spans "
                f"({device_elapsed} vs {aggregate})"
            )
        if buffer_document is not None:
            if not buffer_document["flushed"]:
                raise ContractError("completed timing evidence requires a flushed buffer")
            if buffer_document["dropped"] or buffer_document["overwritten"]:
                raise ContractError("completed timing evidence requires an intact buffer")
            if buffer_document["records_delivered"] < buffer_document["records_written"]:
                raise ContractError("completed timing evidence requires every record to be delivered")
    else:
        # No device event was observed, or the observation is incomplete: the
        # row may only be diagnostic.
        if synchronized:
            raise ContractError("incomplete timing evidence cannot claim completed synchronization")
        if launches or device_elapsed:
            raise ContractError("incomplete timing evidence cannot report device time or a launch")
        if clock_domain != NPU_UNOBSERVED_CLOCK_DOMAIN:
            raise ContractError("incomplete timing evidence must name the unobserved clock domain")
        if publish:
            raise ContractError("incomplete timing evidence cannot be publishable")
    if measurement_kind is not None and kind != measurement_kind:
        raise ContractError("timing evidence measurement kind differs from provenance")
    if publication_allowed is not None and publish != publication_allowed and complete:
        raise ContractError("timing evidence publication permission differs from provenance")
    if report_status is not None:
        if report_status == "ok" and not complete:
            raise ContractError("an ok report requires completed device timing evidence")
        if report_status == "timeout" and complete:
            raise ContractError("a timeout report cannot claim a completed device event")
    build_document = None
    if build_expected:
        build_document = npu_build_evidence_document(
            value["build"], expected_build=expected_build,
            request_artifact=request_artifact, report_status=report_status,
            complete=complete,
        )
    document: dict[str, Any] = {
        "schema_id": schema_id,
        "candidate_digest": candidate,
        "artifact_set_digest": artifact_set_digest,
        "device_group": group,
        "topology": topology,
        "spans": spans,
        "timing": {
            "source": source,
            "clock_domain": clock_domain,
            "synchronization_completed": synchronized,
            "device_elapsed_ns": device_elapsed,
            "host_api_elapsed_ns": host_elapsed,
            "launches": launches,
        },
        "buffer": buffer_document,
        "measurement_kind": kind,
        "publication_allowed": publish,
        "complete": complete,
    }
    if build_document is not None:
        document["build"] = build_document
    if "diagnostics" in value:
        document["diagnostics"] = _npu_diagnostics(value["diagnostics"])
    return document


def device_group_requirements(value: Any) -> dict[str, Any]:
    """Normalize the declared resource requirements of an execution spec."""

    if not isinstance(value, Mapping) or set(value) != _NPU_RESOURCE_FIELDS:
        raise ContractError("execution.resources fields are missing or unknown")
    declared: dict[str, Any] = {
        "logical_devices": _integer(
            value["logical_devices"], "execution.resources.logical_devices", 1, MAX_LOGICAL_DEVICES,
        ),
    }
    for name in ("required_rank_spans", "required_die_spans"):
        raw = value[name]
        declared[name] = (
            None if raw is None else _integer(raw, f"execution.resources.{name}", 1, MAX_DECLARED_SPANS)
        )
    _coverable_shape(
        declared["logical_devices"], declared["required_rank_spans"],
        declared["required_die_spans"],
    )
    return declared


def _npu_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or len(value) > 32:
        raise ContractError("timing diagnostics must be a bounded object")
    document: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or not key.isascii() or len(key) > 64:
            raise ContractError("timing diagnostic key is invalid")
        if isinstance(item, str):
            document[key] = _text(item, "timing diagnostic", max_bytes=512, empty=True)
        elif isinstance(item, bool):
            document[key] = item
        elif isinstance(item, int):
            if not -_MAX_SAFE_INTEGER <= item <= _MAX_SAFE_INTEGER:
                raise ContractError("timing diagnostic integer is unsafe")
            document[key] = item
        elif item is None:
            document[key] = None
        else:
            raise ContractError("timing diagnostics must be scalar")
    return document


def npu_evidence_document(
    value: Any, *, dialect: ContractDialect, source_bytes: bytes | bytearray | memoryview | None = None,
    device_group: Mapping[str, Any] | None = None,
    expected_artifact_set_digest: str | None = None,
    expected_resources: Mapping[str, Any] | None = None,
    expected_build: Mapping[str, Any] | None = None,
    request_artifact: Mapping[str, Any] | None = None,
    measurement_kind: str | None = None,
    publication_allowed: bool | None = None,
    report_status: str | None = None,
) -> dict[str, Any]:
    """Validate TimingEvidence with the schema the contract version defines.

    One dispatch point keeps the version-1 (P0/P0b), version-2 (P1) and
    version-3 (P1c) schemas from drifting apart at the four places that validate
    a result: the job store's durable read and worker-result paths, the host
    outbox and the gRPC client.
    """

    if dialect.build_contract:
        if expected_build is None:
            raise ContractError("a build-contract result requires the accepted build provenance")
        return npu_build_timing_evidence_document(
            value, source_bytes=source_bytes, device_group=device_group,
            expected_artifact_set_digest=expected_artifact_set_digest,
            expected_resources=expected_resources, expected_build=expected_build,
            request_artifact=request_artifact,
            measurement_kind=measurement_kind, publication_allowed=publication_allowed,
            report_status=report_status, dialect=dialect,
        )
    if dialect.npu_bundle:
        return npu_timing_evidence_document_v2(
            value, source_bytes=source_bytes, device_group=device_group,
            expected_artifact_set_digest=expected_artifact_set_digest,
            expected_resources=expected_resources,
            measurement_kind=measurement_kind, publication_allowed=publication_allowed,
            report_status=report_status, dialect=dialect,
        )
    if expected_artifact_set_digest is not None or expected_resources is not None:
        raise ContractError("this contract version has no artifact set or resource requirements")
    if expected_build is not None or request_artifact is not None:
        raise ContractError("this contract version has no build evidence")
    return npu_timing_evidence_document(
        value, source_bytes=source_bytes, device_group=device_group,
        measurement_kind=measurement_kind, publication_allowed=publication_allowed,
        report_status=report_status, dialect=dialect,
    )
