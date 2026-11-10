"""Freeze the canonical execution geometry into the build artifact set (A3).

A2 produced the canonical bundle (candidate entries + geometry pair + optional lifecycle) and its
``bundle_digest``; A3 makes that geometry a *build product*:

* the candidate manifest is the bundle's ``SOURCE`` artifact (multi-file submission), so the frozen
  schema-4 rule ``SOURCE.sha256 == candidate_sha256`` keeps holding without a second identity;
* the complete canonical geometry (candidate + fallback resolutions, lifecycle) is written as an
  ``AUXILIARY`` entry at ``execution_geometry.json``;
* *after* that entry exists, ``provenance.artifact_set_digest`` is recomputed over the complete
  artifact set, so the confirmation gate (``two_period`` on that digest) binds the geometry;
* the native descriptor stores only the geometry ``schema_id``/``path``/``sha256`` and that record
  is part of ``native_digest``, which is cross-audited against the entry, never used as the
  confirmation comparison.

Nothing here adds an ``EvalReport`` field or changes a wire contract: it only extends the artifact
set and the local native descriptor that already exist.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Mapping, TYPE_CHECKING

from .artifact_builder import BuildArtifact, BuildEvidence, BuildOutcome, BuildProvenance
from .artifact_bundle import ArtifactBundle, BundleEntry, NativeExecutable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..track.bundle import CanonicalBundle

EXECUTION_GEOMETRY_ARTIFACT_PATH = "execution_geometry.json"
EXECUTION_GEOMETRY_ARTIFACT_MEDIA_TYPE = "application/json"
CANDIDATE_MANIFEST_ARTIFACT_PATH = "candidate_manifest.json"
CANDIDATE_MANIFEST_ARTIFACT_MEDIA_TYPE = "application/json"


class GeometryArtifactError(ValueError):
    """@brief 几何不能作为构建产物冻结进 artifact set。"""


class EvaluatorBuildError(GeometryArtifactError):
    """@brief 评测方（外部有界 worker）实际执行的构建与本节点的 canonical 构建对象不一致。"""


@dataclass(frozen=True)
class EvaluatorBuild:
    """@brief 由外部有界 evaluator 真实构建、并经本节点交叉审计的构建记录。

    它不是第二个 artifact builder：本进程没有编译，也不重新组装 BuildOutcome；它只是把
    worker 实际产出的 schema-4 ``full``/``partial`` build 记录与本节点自己持有的 canonical
    bundle（候选字节 + 冻结几何 + 生命周期）逐项对齐。任何一项不同即拒绝，不做覆盖。
    """

    shape: str
    artifact_set_digest: str | None
    native_digest: str | None
    assembled_sha256: str | None
    entries: tuple[Mapping[str, Any], ...]
    document: Mapping[str, Any]


def canonical_geometry_bytes(canonical_bundle: "CanonicalBundle") -> bytes:
    """@brief 冻结几何 entry 的确切字节（RFC 8785 canonical JSON）。

    @param canonical_bundle A2 canonical bundle。
    @return canonical 字节。
    """
    from ..identity import canonical_bytes

    return canonical_bytes(canonical_bundle.document())


def _entry(path: str, role: str, media_type: str, data: bytes) -> BundleEntry:
    """@brief 由确切字节构造一个 bundle entry。

    @param path 相对路径。
    @param role artifact 角色。
    @param media_type 媒体类型。
    @param data 确切字节。
    @return entry。
    """
    return BundleEntry(
        path=path,
        role=role,
        media_type=media_type,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        data=data,
    )


def _geometry_entry_manifest(canonical_bundle: "CanonicalBundle") -> dict[str, Any]:
    """@brief 冻结几何 artifact 的 manifest（不读也不写字节）。

    @param canonical_bundle A2 canonical bundle。
    @return manifest entry。
    """
    geometry = canonical_geometry_bytes(canonical_bundle)
    return {
        "path": EXECUTION_GEOMETRY_ARTIFACT_PATH,
        "role": "AUXILIARY",
        "media_type": EXECUTION_GEOMETRY_ARTIFACT_MEDIA_TYPE,
        "size_bytes": len(geometry),
        "sha256": hashlib.sha256(geometry).hexdigest(),
    }


def _candidate_entry_manifests(canonical_bundle: "CanonicalBundle") -> list[dict[str, Any]]:
    """@brief 候选 entries 在构建产物里的 AUXILIARY manifest。

    @param canonical_bundle A2 canonical bundle。
    @return 按路径排序的 manifest entry 列表。
    """
    return [
        {
            "path": entry.path,
            "role": "AUXILIARY",
            "media_type": entry.media_type,
            "size_bytes": len(entry.data),
            "sha256": entry.sha256,
        }
        for entry in canonical_bundle.entries
    ]


def verify_evaluator_build(
    document: Mapping[str, Any],
    *,
    canonical_bundle: "CanonicalBundle",
    identity: Mapping[str, str],
    artifact_audit: Callable[..., tuple[list[dict[str, Any]], set[str]]] | None = None,
) -> EvaluatorBuild:
    """@brief 审计外部有界 evaluator 实际构建／执行的 build 记录。

    The evaluator (the bounded v2 worker) owns the first real compile; the control host must not
    compile the candidate a second time.  This function is what keeps that split honest: the
    worker returns the schema-4 build record it projected from the *real* build it executed, and
    the control host re-derives the candidate/geometry artifact set, the geometry artifact digest,
    the native descriptor's ``execution_geometry`` audit and the artifact-set/native digests from
    its own canonical bundle.  A fabricated, omitted, extra or drifted entry is refused.

    @param document worker 返回的 ``build`` 记录（``full`` 或 ``partial``）。
    @param canonical_bundle 本节点冻结的候选 + 几何 + 生命周期。
    @param identity 控制机冻结的 ``opspec_digest``/``lowering_profile_digest``。
    @param artifact_audit DSL builder 的可选闭集审计：在通用候选/几何/native 检查之外，核对
        frontend 生成物闭集并返回 ``(expected_entries, native_paths)``。缺省保持 MSL 原行为。
    @return 已审计的构建身份。
    @raises EvaluatorBuildError 形状、候选/几何 artifact、native 交叉审计或身份不一致。
    """
    from ..build_projection import identity_document, validate_shape
    from ..schema import SchemaError
    from ..track.bundle import CanonicalBundle

    if not isinstance(canonical_bundle, CanonicalBundle):
        raise EvaluatorBuildError("canonical_bundle must be a CanonicalBundle")
    try:
        normalized = validate_shape(document)
    except SchemaError as exc:
        raise EvaluatorBuildError(f"the evaluator build record is not a schema-4 build: {exc}") from exc
    candidate = canonical_bundle.candidate
    candidate_sha256 = canonical_bundle.candidate_sha256
    if normalized["candidate_sha256"] != candidate_sha256:
        raise EvaluatorBuildError("the evaluator build candidate_sha256 is not the canonical candidate identity")
    provenance = normalized["provenance"]
    if provenance["source_sha256"] != candidate_sha256:
        raise EvaluatorBuildError(
            "the evaluator build provenance.source_sha256 is not the canonical candidate identity"
        )
    expected_identity = identity_document(identity)
    actual_identity = {
        "opspec_digest": normalized["opspec_digest"],
        "lowering_profile_digest": normalized["lowering_profile_digest"],
    }
    if actual_identity != expected_identity:
        raise EvaluatorBuildError(
            "the evaluator build task identity drifted from the control-host task identity"
        )
    if normalized["build_kind"] != "native":
        raise EvaluatorBuildError("the evaluator build is not a native build")
    if normalized["shape"] == "partial":
        if provenance["artifact_set_digest"] is not None:
            raise EvaluatorBuildError("a partial evaluator build must not claim an artifact set")
        if normalized["assembled_sha256"] is None:
            raise EvaluatorBuildError(
                "a projectable compile failure must carry the assembled digest of the bytes that reached the compiler"
            )
        return EvaluatorBuild(
            shape="partial",
            artifact_set_digest=None,
            native_digest=None,
            assembled_sha256=normalized["assembled_sha256"],
            entries=(),
            document=normalized,
        )
    if normalized["placement"] != "control_host":
        raise EvaluatorBuildError(
            "the evaluator build is not this control host's fresh build; a received set cannot bind the executed geometry"
        )
    if normalized["assembled_sha256"] is None:
        raise EvaluatorBuildError("a successful native build must carry the assembled digest")
    entries = [dict(item) for item in normalized["artifacts"]["entries"]]
    aliases = [unicodedata.normalize("NFC", item["path"]).casefold() for item in entries]
    if len(set(aliases)) != len(aliases):
        raise EvaluatorBuildError(
            "the evaluator artifact set has duplicate or case/normalization-aliased paths; a bundle cannot "
            "materialize one path twice"
        )
    geometry_entry = _geometry_entry_manifest(canonical_bundle)
    native = normalized.get("native")
    if native is None:
        raise EvaluatorBuildError("the evaluator build has no native executable to cross-audit")
    sources = [item for item in entries if item["role"] == "SOURCE"]
    if len(sources) != 1:
        raise EvaluatorBuildError(
            f"the evaluator artifact set must carry exactly one SOURCE artifact, got {len(sources)}"
        )
    source = sources[0]
    if candidate.single_source:
        candidate_entry = candidate.entries[0]
        expected_source = {
            "path": candidate_entry.path,
            "role": "SOURCE",
            "media_type": candidate_entry.media_type,
            "size_bytes": len(candidate_entry.data),
            "sha256": candidate_entry.sha256,
        }
        if source != expected_source:
            raise EvaluatorBuildError(
                "the evaluator SOURCE artifact is not the canonical candidate entry (path/role/media_type/"
                "size/digest); a renamed or retyped SOURCE still identifies the same bytes but not the same artifact"
            )
        rest = sorted(
            (item for item in entries if item["role"] != "SOURCE"), key=lambda item: item["path"]
        )
        if rest != [geometry_entry]:
            raise EvaluatorBuildError(
                "a single-source build may only add the frozen execution geometry to the candidate SOURCE"
            )
        expected_entries = sorted([expected_source, geometry_entry], key=lambda item: item["path"])
        native_paths = {candidate_entry.path}
    else:
        manifest_entry = {
            "path": CANDIDATE_MANIFEST_ARTIFACT_PATH,
            "role": "SOURCE",
            "media_type": CANDIDATE_MANIFEST_ARTIFACT_MEDIA_TYPE,
            "size_bytes": len(candidate.manifest_bytes),
            "sha256": candidate_sha256,
        }
        if source != manifest_entry:
            raise EvaluatorBuildError(
                "a multi-file build must attach the candidate manifest as its exact SOURCE artifact"
            )
        candidate_entries = _candidate_entry_manifests(canonical_bundle)
        expected_entries = sorted(
            [*candidate_entries, manifest_entry, geometry_entry],
            key=lambda item: item["path"],
        )
        native_paths = {
            entry.path for entry in candidate.entries if entry.role == "device_kernel"
        }
        if artifact_audit is not None:
            expected_entries, native_paths = artifact_audit(
                entries=entries,
                candidate_entries=candidate_entries,
                manifest_entry=manifest_entry,
                geometry_entry=geometry_entry,
                native=native,
                canonical_bundle=canonical_bundle,
            )
        if entries != expected_entries:
            raise EvaluatorBuildError(
                "the evaluator artifact set is not exactly the candidate entries, the candidate manifest, "
                "the frozen geometry and the Track-audited generated artifacts"
            )
    if candidate.single_source and entries != expected_entries:  # pragma: no cover - covered by ``rest`` above
        raise EvaluatorBuildError("the evaluator artifact set does not match the candidate and geometry identity")
    if native["entry_path"] not in native_paths:
        raise EvaluatorBuildError(
            "the evaluator native executable was not built from the single SOURCE or a candidate device kernel"
        )
    by_path = {item["path"]: item for item in entries}
    if native["entry_sha256"] != by_path[native["entry_path"]]["sha256"]:
        raise EvaluatorBuildError("the evaluator native entry digest is not the listed artifact digest")
    audit = native["descriptor"].get("execution_geometry")
    expected_audit = {
        "schema_id": canonical_bundle.schema_id,
        "path": EXECUTION_GEOMETRY_ARTIFACT_PATH,
        "sha256": geometry_entry["sha256"],
    }
    if audit != expected_audit:
        raise EvaluatorBuildError(
            f"the evaluator native descriptor geometry audit is {audit!r}, expected {expected_audit!r}"
        )
    if native["descriptor"].get("assembled_sha256") != normalized["assembled_sha256"]:
        raise EvaluatorBuildError(
            "the evaluator assembled digest disagrees with the native descriptor of the same build"
        )
    if provenance["artifact_set_digest"] != normalized["artifacts"]["artifact_set_digest"]:
        raise EvaluatorBuildError(
            "the evaluator provenance artifact_set_digest does not bind its artifact manifest"
        )
    return EvaluatorBuild(
        shape="full",
        artifact_set_digest=normalized["artifacts"]["artifact_set_digest"],
        native_digest=normalized["native_digest"],
        assembled_sha256=normalized["assembled_sha256"],
        entries=tuple(entries),
        document=normalized,
    )


def attach_execution_geometry(outcome: BuildOutcome, *, canonical_bundle: "CanonicalBundle") -> BuildOutcome:
    """@brief 把候选清单与冻结几何加入构建产物并重算 artifact_set_digest。

    Order is fixed: the candidate identity is already fixed before the build
    (``provenance.source_sha256 == candidate_sha256``), then this function appends the geometry
    entry, rebuilds the native descriptor, and only then derives the new provenance/evidence
    documents.  A caller cannot attach after a report has been written, and cannot attach a
    geometry belonging to another candidate.

    @param outcome 已成功的构建结果（候选源 bytes 已编译）。
    @param canonical_bundle A2 canonical bundle（候选 + 冻结几何 + 可选生命周期）。
    @return 含候选清单与几何 entry 的新构建结果。
    @raises GeometryArtifactError 身份不符、已附加或路径冲突。
    """
    from ..track.bundle import CanonicalBundle

    if not isinstance(outcome, BuildOutcome):
        raise GeometryArtifactError("attach_execution_geometry needs a BuildOutcome")
    if not isinstance(canonical_bundle, CanonicalBundle):
        raise GeometryArtifactError("canonical_bundle must be a CanonicalBundle")
    if outcome.provenance.source_sha256 != canonical_bundle.candidate_sha256:
        raise GeometryArtifactError(
            "the build's source_sha256 is not the candidate identity: a geometry cannot be attached to a "
            "different candidate"
        )
    existing = {entry.path for entry in outcome.bundle.entries}
    if EXECUTION_GEOMETRY_ARTIFACT_PATH in existing:
        raise GeometryArtifactError("the artifact set already carries a frozen execution geometry")
    if canonical_bundle.candidate.single_source:
        sources = [entry for entry in outcome.bundle.entries if entry.role == "SOURCE"]
        if len(sources) != 1 or sources[0].sha256 != canonical_bundle.candidate_sha256:
            raise GeometryArtifactError(
                "a single-source build must carry exactly one SOURCE artifact with the candidate identity"
            )
    elif any(entry.role == "SOURCE" for entry in outcome.bundle.entries):
        raise GeometryArtifactError(
            "a multi-file build attaches its candidate manifest as SOURCE; the builder must not also produce a "
            "SOURCE artifact, or the frozen projection would pick the wrong candidate identity"
        )
    geometry_bytes = canonical_geometry_bytes(canonical_bundle)
    geometry_entry = _entry(
        EXECUTION_GEOMETRY_ARTIFACT_PATH,
        "AUXILIARY",
        EXECUTION_GEOMETRY_ARTIFACT_MEDIA_TYPE,
        geometry_bytes,
    )
    entries: list[BundleEntry] = list(outcome.bundle.entries)
    if not canonical_bundle.candidate.single_source:
        if CANDIDATE_MANIFEST_ARTIFACT_PATH in existing:
            raise GeometryArtifactError("the artifact set already carries a candidate manifest")
        entries.append(
            _entry(
                CANDIDATE_MANIFEST_ARTIFACT_PATH,
                "SOURCE",
                CANDIDATE_MANIFEST_ARTIFACT_MEDIA_TYPE,
                canonical_bundle.candidate.manifest_bytes,
            )
        )
        for candidate_entry in canonical_bundle.entries:
            if candidate_entry.path in existing:
                produced = next(entry for entry in outcome.bundle.entries if entry.path == candidate_entry.path)
                if (
                    produced.role != "AUXILIARY"
                    or produced.data != bytes(candidate_entry.data)
                    or produced.media_type != candidate_entry.media_type
                ):
                    raise GeometryArtifactError(
                        f"the builder's artifact at candidate path {candidate_entry.path!r} is not the candidate entry "
                        "bytes (or is not an AUXILIARY artifact)"
                    )
                continue
            entries.append(
                _entry(candidate_entry.path, "AUXILIARY", candidate_entry.media_type, bytes(candidate_entry.data))
            )
    entries.append(geometry_entry)

    native = outcome.bundle.native
    new_native: NativeExecutable | None = None
    if native is not None:
        descriptor = dict(native.descriptor)
        descriptor["execution_geometry"] = {
            "schema_id": canonical_bundle.schema_id,
            "path": EXECUTION_GEOMETRY_ARTIFACT_PATH,
            "sha256": geometry_entry.sha256,
        }
        new_native = NativeExecutable(
            backend_id=native.backend_id,
            kind=native.kind,
            entry_path=native.entry_path,
            entry_sha256=native.entry_sha256,
            descriptor=descriptor,
            handle=native.handle,
        )
    bundle = ArtifactBundle(entries=tuple(entries), native=new_native)
    provenance = BuildProvenance(
        profile=outcome.provenance.profile,
        source_sha256=outcome.provenance.source_sha256,
        artifact_set_digest=bundle.artifact_set_digest,
    )
    evidence = BuildEvidence(
        mode=outcome.evidence.mode,
        status=outcome.evidence.status,
        provenance=provenance,
        artifacts=tuple(BuildArtifact.from_entry(entry) for entry in bundle.entries),
        error_class=outcome.evidence.error_class,
        detail=outcome.evidence.detail,
        duration_ms=outcome.evidence.duration_ms,
        log=outcome.evidence.log,
        loaded=outcome.evidence.loaded,
        native=new_native,
        placement=outcome.evidence.placement,
        producer=outcome.evidence.producer,
    )
    attached = BuildOutcome(
        bundle=bundle,
        provenance=provenance,
        evidence=evidence,
        payloads=outcome.payloads,
        assembled_sha256=outcome.assembled_sha256,
    )
    verify_execution_geometry_artifact(attached, canonical_bundle)
    return attached


def verify_execution_geometry_artifact(outcome: BuildOutcome, canonical_bundle: "CanonicalBundle") -> None:
    """@brief 校验冻结几何 entry、provenance digest 与 native.descriptor 交叉审计。

    @param outcome 已附加几何的构建结果。
    @param canonical_bundle 同一份 canonical bundle。
    @raises GeometryArtifactError 任一身份/字节/描述符不一致。
    """
    from ..track.bundle import CanonicalBundle

    if not isinstance(canonical_bundle, CanonicalBundle):
        raise GeometryArtifactError("canonical_bundle must be a CanonicalBundle")
    if outcome.provenance.source_sha256 != canonical_bundle.candidate_sha256:
        raise GeometryArtifactError("the build source identity is not the canonical candidate identity")
    try:
        entry = outcome.bundle.entry(EXECUTION_GEOMETRY_ARTIFACT_PATH)
    except Exception as exc:  # noqa: BLE001 - BundleError
        raise GeometryArtifactError(f"the artifact set has no frozen execution geometry: {exc}") from exc
    if entry.role != "AUXILIARY":
        raise GeometryArtifactError("the frozen execution geometry must be an AUXILIARY artifact")
    expected = canonical_geometry_bytes(canonical_bundle)
    if entry.data != expected:
        raise GeometryArtifactError("the frozen execution geometry bytes are not the canonical bundle document")
    if not canonical_bundle.candidate.single_source:
        try:
            manifest = outcome.bundle.entry(CANDIDATE_MANIFEST_ARTIFACT_PATH)
        except Exception as exc:  # noqa: BLE001
            raise GeometryArtifactError(f"a multi-file build has no candidate manifest artifact: {exc}") from exc
        if manifest.role != "SOURCE" or manifest.sha256 != canonical_bundle.candidate_sha256:
            raise GeometryArtifactError("the candidate manifest artifact does not carry the candidate identity")
        if manifest.data != canonical_bundle.candidate.manifest_bytes:
            raise GeometryArtifactError("the candidate manifest artifact bytes are not the candidate manifest")
    native = outcome.bundle.native
    if native is not None:
        audited = native.descriptor.get("execution_geometry")
        expected_audit = {
            "schema_id": canonical_bundle.schema_id,
            "path": EXECUTION_GEOMETRY_ARTIFACT_PATH,
            "sha256": entry.sha256,
        }
        if audited != expected_audit:
            raise GeometryArtifactError(
                f"native.descriptor execution_geometry is {audited!r}, expected {expected_audit!r}"
            )
    if outcome.provenance.artifact_set_digest != outcome.bundle.artifact_set_digest:
        raise GeometryArtifactError("provenance artifact_set_digest does not bind the attached artifact set")
    if outcome.evidence.native != outcome.bundle.native:
        raise GeometryArtifactError("build evidence native record differs from the artifact set native record")
    manifest = [artifact.manifest() for artifact in outcome.evidence.artifacts]
    actual = [item.manifest() for item in outcome.bundle.entries]
    if manifest != actual:
        raise GeometryArtifactError("build evidence artifacts are not the attached artifact set")


__all__ = [
    "CANDIDATE_MANIFEST_ARTIFACT_MEDIA_TYPE",
    "CANDIDATE_MANIFEST_ARTIFACT_PATH",
    "EXECUTION_GEOMETRY_ARTIFACT_MEDIA_TYPE",
    "EXECUTION_GEOMETRY_ARTIFACT_PATH",
    "EvaluatorBuild",
    "EvaluatorBuildError",
    "GeometryArtifactError",
    "attach_execution_geometry",
    "canonical_geometry_bytes",
    "verify_evaluator_build",
    "verify_execution_geometry_artifact",
]
