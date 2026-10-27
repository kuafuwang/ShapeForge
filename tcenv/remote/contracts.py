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
NPU_TIMING_EVIDENCE_SCHEMA_ID = "tcenv.remote.npu-source.timing-evidence.1"
NPU_EVIDENCE_MEDIA_TYPE = "application/vnd.tcenv.npu-source.timing-evidence.v1+json"
# Clock domain recorded when no device clock was ever observed (killed worker).
NPU_UNOBSERVED_CLOCK_DOMAIN = "unobserved"
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
    #: ``source`` (inline candidate bytes) or ``blob`` (content-addressed
    #: reference).  The contract identity selects the form; a request shape
    #: never does.
    artifact_form: str = "source"

    @property
    def identity_pair(self) -> tuple[str, str]:
        return (self.api_contract_id, self.evidence_contract_id)

    @property
    def npu_blob(self) -> bool:
        return self.artifact_form == "blob"

    @property
    def candidate_algorithms(self) -> tuple[str, ...]:
        """CandidateDigest algorithms this dialect accepts."""

        return (BLOB_DIGEST_ALGORITHM,) if self.npu_blob else ("sha256",)


LEGACY_DIALECT = ContractDialect(
    api_contract_id=API_CONTRACT_ID,
    evidence_contract_id=EVIDENCE_CONTRACT_ID,
    request_document_type=REQUEST_DOCUMENT_TYPE,
    receipt_document_type=RECEIPT_DOCUMENT_TYPE,
    result_document_type=RESULT_DOCUMENT_TYPE,
    document_version=CONTRACT_DOCUMENT_VERSION,
    npu_source=False,
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
)
NPU_DIALECTS = (NPU_SOURCE_DIALECT, NPU_BLOB_DIALECT)
NPU_API_CONTRACT_IDS = frozenset(dialect.api_contract_id for dialect in NPU_DIALECTS)
NPU_EVIDENCE_CONTRACT_IDS = frozenset(dialect.evidence_contract_id for dialect in NPU_DIALECTS)
CONTRACT_DIALECTS = {
    LEGACY_DIALECT.api_contract_id: LEGACY_DIALECT,
    NPU_SOURCE_DIALECT.api_contract_id: NPU_SOURCE_DIALECT,
    NPU_BLOB_DIALECT.api_contract_id: NPU_BLOB_DIALECT,
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
    fields = _NPU_BLOB_REQUEST_FIELDS if dialect.npu_blob else _NPU_SOURCE_REQUEST_FIELDS
    expected = fields | {"request_id"}
    if set(document) != expected:
        missing = sorted(expected - set(document))
        unknown = sorted(set(document) - expected)
        raise ContractError(f"stored request fields are missing or unknown: missing={missing} unknown={unknown}")
    # ``request_id`` is transport identity, so it is not part of the canonical
    # digest; it still has to be a well-formed identifier and the caller must
    # compare it with the durable job row.
    _identifier(document["request_id"], "request_id")
    if dialect.npu_blob:
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
            raise ContractError("provenance timing source is not supported by P0")
        document["timing_sources"] = sorted(set(sources))
        if document["artifact_kind"] != "source" or document["launch_kind"] != "KERNEL" or document["launch_subtype"] != "SIMT":
            raise ContractError("provenance execution shape is not supported by P0")
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
