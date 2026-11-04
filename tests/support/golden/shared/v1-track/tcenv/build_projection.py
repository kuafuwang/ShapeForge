"""Schema-4 report ``build``: the one projection and its closed-set validator.

The report has exactly one build record (§6.2).  It is either ``null`` (no builder ran,
or the build was not this attempt's candidate evidence) or a closed object whose
``shape`` is ``partial`` (a typed compile failure) or ``full`` (a successful build or a
received prebuilt artifact set).

* ``BuildOutcome`` is the neutral source of a successful build.  :func:`project_full`
  projects it; it never invents an artifact set, an evidence document or a native
  executable.
* ``BuildError`` with a typed ``compile`` class is the only source of ``partial``.
  :func:`project_partial` projects its error class, duration, log digest and the
  identity that was already known before the build.

The projection runs only in the process that holds the source objects (local worker,
remote evaluator worker, prebuilt control host).  Readers call :func:`validate_for_status`
and recompute the cross-checks; they never fill in or rewrite ``build``.

``assembled_sha256`` is the digest of the bytes that actually entered the compiler.
A builder with a separate assembly step (Metal prepends the contract header) hashes
the assembled text, never the candidate again.  A direct-source builder
(simulator/source_only) has no separate representation, so the compilation unit *is*
the source bundle; recording its digest is honest there, and the "do not copy the
candidate" rule targets builders that assemble but skip hashing the result.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping

from .schema import SchemaError

BUILD_SHAPES = frozenset({"partial", "full"})
BUILD_KINDS = frozenset({"native", "simulator", "source_only"})
BUILD_PLACEMENTS = frozenset({"control_host", "evaluator", "received"})
NATIVE_ABSENT_PORTABLE = "portable_artifact_set"
NATIVE_ABSENT_NOT_NATIVE = "not_native_build"
NATIVE_ABSENT_REASONS = frozenset({NATIVE_ABSENT_PORTABLE, NATIVE_ABSENT_NOT_NATIVE})

#: ``status`` -> the ``build`` shapes it may carry (``None`` = ``build: null``).
#: The status set is the shared ``CANDIDATE_STATUSES``; ``pending`` is not a persisted
#: status and is refused before this table is consulted.
STATUS_SHAPES: dict[str, frozenset[str | None]] = {
    "ok": frozenset({"full"}),
    "compile_error": frozenset({"partial"}),
    "abi_error": frozenset({"full"}),
    "unsafe": frozenset({"full"}),
    "incorrect": frozenset({"full"}),
    "runtime_error": frozenset({"full"}),
    "rejected": frozenset({None, "full"}),
    "timeout": frozenset({None}),
    "crash": frozenset({None}),
    "refused_hang": frozenset({None}),
    "unchanged": frozenset({None}),
    "duplicate": frozenset({None}),
    "external_duplicate": frozenset({None}),
}

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")

_COMMON_FIELDS = frozenset({
    "shape", "build_kind", "candidate_sha256", "assembled_sha256",
    "opspec_digest", "lowering_profile_digest", "provenance",
})
_PARTIAL_FIELDS = _COMMON_FIELDS | {"error_class", "duration_ms", "log"}
_FULL_FIELDS = _COMMON_FIELDS | {
    "placement", "artifacts", "evidence", "native", "native_digest", "native_absent_reason",
}
_NATIVE_FIELDS = frozenset({"schema_id", "backend_id", "kind", "entry_path", "entry_sha256", "descriptor"})
_LOG_FIELDS = frozenset({"sha256", "size_bytes"})
_ARTIFACTS_FIELDS = frozenset({"schema_id", "entries", "artifact_set_digest"})
_MANIFEST_ENTRY_FIELDS = frozenset({"path", "role", "media_type", "size_bytes", "sha256"})
_EVIDENCE_FIELDS = frozenset({
    "schema_id", "mode", "status", "error_class", "provenance", "artifacts",
    "duration_ms", "log", "loaded",
})
_LOADED_FIELDS = frozenset({"path", "sha256"})


def _fail(message: str) -> SchemaError:
    return SchemaError(f"report build is invalid: {message}")


def _closed(
    value: Any, fields: frozenset[str], what: str, *, optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{what} must be an object")
    keys = set(value)
    unknown = keys - fields
    if unknown:
        raise _fail(f"{what} has fields this version does not know ({', '.join(sorted(unknown))})")
    missing = (fields - optional) - keys
    if missing:
        raise _fail(f"{what} is missing {', '.join(sorted(missing))}")
    return value


def _hex64(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _HEX64.match(value):
        raise _fail(f"{what} must be 64 lowercase hex characters")
    return value


def _hex64_or_null(value: Any, what: str) -> str | None:
    if value is None:
        return None
    return _hex64(value, what)


def _non_negative_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _fail(f"{what} must be a non-negative integer")
    return value


def _log_document(value: Any, what: str) -> dict[str, Any] | None:
    if value is None:
        return None
    document = _closed(value, _LOG_FIELDS, what)
    return {
        "sha256": _hex64(document["sha256"], f"{what}.sha256"),
        "size_bytes": _non_negative_int(document["size_bytes"], f"{what}.size_bytes"),
    }


def _provenance_document(value: Any, *, candidate_sha256: str) -> dict[str, Any]:
    from .remote import contracts

    try:
        document = contracts.build_provenance_document(value)
    except contracts.ContractError as exc:
        raise _fail(f"provenance is not a build provenance: {exc}") from exc
    if document["source_sha256"] != candidate_sha256:
        raise _fail("provenance source_sha256 differs from candidate_sha256")
    return document


def _artifacts_document(value: Any) -> dict[str, Any]:
    from .remote.artifact_bundle import BUNDLE_SCHEMA_ID, artifact_set_digest

    document = _closed(value, _ARTIFACTS_FIELDS, "artifacts")
    if document["schema_id"] != BUNDLE_SCHEMA_ID:
        raise _fail("artifacts schema_id is not the artifact bundle schema")
    entries = document["entries"]
    if not isinstance(entries, list) or not entries:
        raise _fail("a full build must list its artifacts")
    normalized = []
    for entry in entries:
        entry = _closed(entry, _MANIFEST_ENTRY_FIELDS, "artifacts entry")
        normalized.append(dict(entry))
    try:
        digest = artifact_set_digest(BUNDLE_SCHEMA_ID, normalized)
    except Exception as exc:  # BundleError is the expected failure; do not narrow it to one module
        raise _fail(f"artifacts entries are invalid: {exc}") from exc
    declared = _hex64(document["artifact_set_digest"], "artifacts.artifact_set_digest")
    if declared != digest:
        raise _fail("artifacts.artifact_set_digest does not bind the listed entries")
    return {"schema_id": BUNDLE_SCHEMA_ID, "entries": normalized, "artifact_set_digest": digest}


def _native_document(value: Any, *, artifacts: Mapping[str, Any], build_kind: str) -> tuple[dict[str, Any], str]:
    from .remote import contracts
    from .remote.artifact_bundle import (
        NATIVE_EXECUTABLE_KINDS,
        NATIVE_EXECUTABLE_SCHEMA_ID,
        validate_relative_path,
    )

    document = _closed(value, _NATIVE_FIELDS, "native")
    if document["schema_id"] != NATIVE_EXECUTABLE_SCHEMA_ID:
        raise _fail("native schema_id is not the backend-native executable schema")
    if build_kind == "source_only":
        raise _fail("a source_only build never compiles and cannot carry a native executable")
    backend_id = document["backend_id"]
    if not isinstance(backend_id, str) or not backend_id or len(backend_id) > 32:
        raise _fail("native backend_id is invalid")
    if document["kind"] not in NATIVE_EXECUTABLE_KINDS:
        raise _fail("native kind is not supported")
    try:
        entry_path = validate_relative_path(document["entry_path"])
    except Exception as exc:
        raise _fail(f"native entry_path is invalid: {exc}") from exc
    entry_sha256 = _hex64(document["entry_sha256"], "native.entry_sha256")
    descriptor = document["descriptor"]
    if not isinstance(descriptor, Mapping):
        raise _fail("native descriptor must be an object")
    try:
        contracts.canonical_digest_document(dict(descriptor))
    except contracts.ContractError as exc:
        raise _fail(f"native descriptor is not canonical: {exc}") from exc
    if not any(
        entry["path"] == entry_path and entry["sha256"] == entry_sha256
        for entry in artifacts["entries"]
    ):
        raise _fail("the native executable is not built from a listed artifact")
    normalized = {
        "schema_id": NATIVE_EXECUTABLE_SCHEMA_ID,
        "backend_id": backend_id,
        "kind": document["kind"],
        "entry_path": entry_path,
        "entry_sha256": entry_sha256,
        "descriptor": dict(descriptor),
    }
    digest = hashlib.sha256(contracts.canonical_digest_document(normalized)).hexdigest()
    return normalized, digest


def _evidence_document(
    value: Any, *, placement: str, provenance: Mapping[str, Any], artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    from .remote import contracts

    document = _closed(
        value,
        _EVIDENCE_FIELDS | ({"producer"} if placement == "received" else frozenset()),
        "evidence",
    )
    if document["schema_id"] != contracts.NPU_BUILD_EVIDENCE_SCHEMA_ID:
        raise _fail("evidence schema_id is not the build evidence schema")
    if document["status"] not in {"ok", "reused"}:
        raise _fail("a full build's evidence status must be ok or reused")
    if document["error_class"] is not None:
        raise _fail("a full build's evidence must not carry an error class")
    mode = document["mode"]
    expected = {
        "control_host": ("prebuilt", "ok"),
        "evaluator": ("evaluator", "ok"),
        "received": ("prebuilt", "reused"),
    }[placement]
    if (mode, document["status"]) != expected:
        raise _fail(f"evidence mode/status do not describe a {placement} build")
    duration = _non_negative_int(document["duration_ms"], "evidence.duration_ms")
    log = _log_document(document["log"], "evidence.log")
    loaded = document["loaded"]
    if not isinstance(loaded, list):
        raise _fail("evidence.loaded must be a list")
    normalized_loaded = []
    for item in loaded:
        item = _closed(item, _LOADED_FIELDS, "evidence.loaded entry")
        normalized_loaded.append({
            "path": item["path"],
            "sha256": _hex64(item["sha256"], "evidence.loaded.sha256"),
        })
    if _provenance_document(document["provenance"], candidate_sha256=provenance["source_sha256"]) != dict(provenance):
        raise _fail("evidence provenance differs from the top-level build provenance")
    if document["artifacts"] != dict(artifacts):
        raise _fail("evidence artifacts differ from the top-level build artifacts")
    normalized = {
        "schema_id": contracts.NPU_BUILD_EVIDENCE_SCHEMA_ID,
        "mode": mode,
        "status": document["status"],
        "error_class": None,
        "provenance": dict(provenance),
        "artifacts": dict(artifacts),
        "duration_ms": duration,
        "log": log,
        "loaded": normalized_loaded,
    }
    if placement == "received":
        try:
            normalized["producer"] = contracts.build_producer_record(document["producer"])
        except contracts.ContractError as exc:
            raise _fail(f"evidence producer link is invalid: {exc}") from exc
    return normalized


def validate_shape(document: Any) -> dict[str, Any]:
    """@brief 校验 ``build`` 对象的闭集形状与内部一致性。

    Validate one non-null ``build`` object; return it normalized.  The caller is responsible
    for the ``status``/``shape`` correspondence (:func:`validate_for_status`).

    @param document ``build`` 对象。
    @return 规范化后的同一形状（未补写、未改值）。
    @raises SchemaError 字段闭集、类型、摘要或交叉核对失败。
    """
    if not isinstance(document, Mapping):
        raise _fail("must be null or an object")
    shape = document.get("shape")
    if shape not in BUILD_SHAPES:
        raise _fail("shape must be partial or full")
    fields = _PARTIAL_FIELDS if shape == "partial" else _FULL_FIELDS
    optional = frozenset() if shape == "partial" else frozenset({"native", "native_absent_reason"})
    document = _closed(document, fields, "build", optional=optional)
    build_kind = document["build_kind"]
    if build_kind not in BUILD_KINDS:
        raise _fail("build_kind must be native, simulator or source_only")
    candidate = _hex64(document["candidate_sha256"], "candidate_sha256")
    assembled = _hex64_or_null(document["assembled_sha256"], "assembled_sha256")
    identity = {
        "opspec_digest": _hex64(document["opspec_digest"], "opspec_digest"),
        "lowering_profile_digest": _hex64(document["lowering_profile_digest"], "lowering_profile_digest"),
    }
    provenance = _provenance_document(document["provenance"], candidate_sha256=candidate)
    normalized: dict[str, Any] = {
        "shape": shape,
        "build_kind": build_kind,
        "candidate_sha256": candidate,
        "assembled_sha256": assembled,
        **identity,
        "provenance": provenance,
    }
    if shape == "partial":
        if document["error_class"] != "compile":
            raise _fail("a partial build only describes a typed compile failure")
        normalized["error_class"] = "compile"
        normalized["duration_ms"] = _non_negative_int(document["duration_ms"], "duration_ms")
        normalized["log"] = _log_document(document["log"], "log")
        if provenance["artifact_set_digest"] is not None:
            raise _fail("a partial build must not claim an artifact set")
        return normalized

    # -- full -------------------------------------------------------------------
    placement = document["placement"]
    if placement not in BUILD_PLACEMENTS:
        raise _fail("placement must be control_host, evaluator or received")
    if provenance["artifact_set_digest"] is None:
        raise _fail("a full build must bind its artifact set digest")
    artifacts = _artifacts_document(document["artifacts"])
    if provenance["artifact_set_digest"] != artifacts["artifact_set_digest"]:
        raise _fail("provenance artifact_set_digest differs from the artifact manifest")
    source_entry = next(
        (entry for entry in artifacts["entries"] if entry["role"] == "SOURCE"), None,
    )
    if source_entry is not None and source_entry["sha256"] != candidate:
        raise _fail("the SOURCE artifact does not carry the candidate bytes")
    evidence = _evidence_document(
        document["evidence"], placement=placement,
        provenance=provenance, artifacts=artifacts,
    )
    has_native = "native" in document
    has_reason = "native_absent_reason" in document
    if placement == "received" and has_native:
        raise _fail("a received prebuilt artifact set cannot claim a native executable")
    if has_native and has_reason:
        raise _fail("native and native_absent_reason are mutually exclusive")
    if has_native:
        native, recomputed = _native_document(
            document["native"], artifacts=artifacts, build_kind=build_kind,
        )
        declared = _hex64_or_null(document["native_digest"], "native_digest")
        if declared is None or declared != recomputed:
            raise _fail("native_digest does not bind the native executable document")
        normalized["native"] = native
        normalized["native_digest"] = recomputed
    else:
        if _hex64_or_null(document.get("native_digest"), "native_digest") is not None:
            raise _fail("a build without a native executable must write native_digest null")
        normalized["native_digest"] = None
        if not has_reason:
            raise _fail("a full build without a native executable must state native_absent_reason")
        reason = document["native_absent_reason"]
        expected_reason = NATIVE_ABSENT_PORTABLE if build_kind == "native" else NATIVE_ABSENT_NOT_NATIVE
        if reason != expected_reason:
            raise _fail(f"native_absent_reason {reason!r} does not match build_kind {build_kind!r}")
        normalized["native_absent_reason"] = reason
    normalized["placement"] = placement
    normalized["artifacts"] = artifacts
    normalized["evidence"] = evidence
    # Keep the canonical field order stable for on-disk reports.
    return {key: normalized[key] for key in sorted(normalized)}


def validate_for_status(build: Any, *, status: str) -> dict[str, Any] | None:
    """@brief 校验 ``status`` 与 ``build`` 的闭集对应，并校验非空形状。

    Validate the ``status``/``build`` correspondence table and the object itself.

    @param build ``build`` 键的值（``None`` 或对象）。
    @param status 报告状态。
    @return 规范化后的 ``build``（``None`` 或对象）。
    @raises SchemaError 状态未知、组合在表外或形状非法。
    """
    if status not in STATUS_SHAPES:
        raise _fail(f"status {status!r} is not a persisted candidate status")
    if build is None:
        if None not in STATUS_SHAPES[status]:
            raise _fail(f"status {status!r} requires a build record")
        return None
    shape = build.get("shape") if isinstance(build, Mapping) else None
    if shape not in STATUS_SHAPES[status]:
        raise _fail(f"status {status!r} cannot carry a {shape!r} build")
    return validate_shape(build)


def identity_document(identity: Mapping[str, Any] | None) -> dict[str, str]:
    """@brief 取候选 ``Specialization`` 的构建身份。

    @param identity ``{"opspec_digest", "lowering_profile_digest"}`` 或 ``None``。
    @return 两个摘要。
    @raises ValueError 没有 Specialization（诊断/无任务 builder 不得伪造任务报告）。
    """
    if not isinstance(identity, Mapping):
        raise ValueError("a candidate build record requires the task Specialization identity")
    return {
        "opspec_digest": _hex64(identity.get("opspec_digest"), "opspec_digest"),
        "lowering_profile_digest": _hex64(
            identity.get("lowering_profile_digest"), "lowering_profile_digest",
        ),
    }


def project_full(outcome: Any, *, build_kind: str, identity: Mapping[str, Any] | None) -> dict[str, Any]:
    """@brief 把一次成功的 ``BuildOutcome`` 投影成 ``build`` 的 ``full`` 形状。

    Project the typed outcome; nothing is inferred that the outcome does not hold.

    @param outcome ``BuildOutcome``。
    @param build_kind builder 登记的构建类型。
    @param identity 任务的 ``Specialization`` 身份。
    @return ``build`` 对象（已通过 :func:`validate_shape` 自检）。
    """
    if build_kind not in BUILD_KINDS:
        raise ValueError("build_kind must be native, simulator or source_only")
    provenance = outcome.provenance.to_dict()
    evidence = outcome.evidence.to_dict()
    native = outcome.evidence.native_document()
    document: dict[str, Any] = {
        "shape": "full",
        "build_kind": build_kind,
        "candidate_sha256": outcome.provenance.source_sha256,
        "assembled_sha256": getattr(outcome, "assembled_sha256", None),
        **identity_document(identity),
        "provenance": provenance,
        "placement": outcome.evidence.placement,
        "artifacts": outcome.evidence.artifacts_document(),
        "evidence": evidence,
    }
    if native is None:
        document["native_digest"] = None
        document["native_absent_reason"] = (
            NATIVE_ABSENT_PORTABLE if build_kind == "native" else NATIVE_ABSENT_NOT_NATIVE
        )
    else:
        document["native"] = {key: value for key, value in native.items() if key != "digest"}
        document["native_digest"] = native["digest"]
    validate_shape(document)
    return document


def project_received(
    evidence: Any, *, build_kind: str, identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """@brief 把 evaluator 已校验并加载的 prebuilt artifact set 投影成 ``full``。

    A received prebuilt set has no ``BuildOutcome`` in this process, so it never claims a
    native executable; the producer link and the load record are the evidence itself.

    @param evidence ``BuildEvidence``（``placement=received``、``status=reused``）。
    @param build_kind builder 登记的构建类型。
    @param identity 任务的 ``Specialization`` 身份。
    @return ``build`` 对象（已通过 :func:`validate_shape` 自检）。
    """
    if build_kind not in BUILD_KINDS:
        raise ValueError("build_kind must be native, simulator or source_only")
    if evidence.placement != "received" or evidence.status != "reused":
        raise ValueError("only a received prebuilt artifact set can be projected here")
    provenance = evidence.provenance.to_dict()
    document = {
        "shape": "full",
        "build_kind": build_kind,
        "candidate_sha256": evidence.provenance.source_sha256,
        "assembled_sha256": None,
        **identity_document(identity),
        "provenance": provenance,
        "placement": "received",
        "artifacts": evidence.artifacts_document(),
        "evidence": evidence.to_dict(),
        "native_digest": None,
        "native_absent_reason": (
            NATIVE_ABSENT_PORTABLE if build_kind == "native" else NATIVE_ABSENT_NOT_NATIVE
        ),
    }
    validate_shape(document)
    return document


def project_partial(error: Any, *, build_kind: str, identity: Mapping[str, Any] | None) -> dict[str, Any]:
    """@brief 把带附件的 ``BuildError`` 投影成 ``build`` 的 ``partial`` 形状。

    Project a typed compile failure.  The error must already carry the provenance and the
    assembled digest; the caller classifies a failure without them as infrastructure, not
    as a candidate compile error.

    @param error ``BuildError``。
    @param build_kind builder 登记的构建类型。
    @param identity 任务的 ``Specialization`` 身份。
    @return ``build`` 对象（已通过 :func:`validate_shape` 自检）。
    """
    from .remote.artifact_builder import require_build_error_projection

    if build_kind not in BUILD_KINDS:
        raise ValueError("build_kind must be native, simulator or source_only")
    require_build_error_projection(error)
    document = {
        "shape": "partial",
        "build_kind": build_kind,
        "candidate_sha256": error.provenance.source_sha256,
        "assembled_sha256": error.assembled_sha256,
        **identity_document(identity),
        "provenance": error.provenance.to_dict(),
        "error_class": "compile",
        "duration_ms": error.duration_ms,
        "log": _log_of(error),
    }
    validate_shape(document)
    return document


def _log_of(error: Any) -> dict[str, Any] | None:
    log = getattr(error, "log", b"") or b""
    if not log:
        return None
    return {"sha256": hashlib.sha256(log).hexdigest(), "size_bytes": len(log)}


def require_no_build(value: Any, *, status: str) -> None:
    """@brief 断言某状态不得携带构建记录（父进程改判用）。

    @param value ``build`` 键的值。
    @param status 报告状态。
    @raises SchemaError 该状态要求 ``build: null`` 却不是。
    """
    if value is not None and None in STATUS_SHAPES.get(status, frozenset()):
        raise _fail(f"status {status!r} must not carry a build record")


__all__ = [
    "BUILD_KINDS",
    "BUILD_PLACEMENTS",
    "BUILD_SHAPES",
    "NATIVE_ABSENT_NOT_NATIVE",
    "NATIVE_ABSENT_PORTABLE",
    "NATIVE_ABSENT_REASONS",
    "STATUS_SHAPES",
    "identity_document",
    "project_full",
    "project_partial",
    "project_received",
    "require_no_build",
    "validate_for_status",
    "validate_shape",
]
