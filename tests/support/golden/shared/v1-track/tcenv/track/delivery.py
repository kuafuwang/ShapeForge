"""Candidate bundle delivery: geometry entry parsing and the single-submission envelope (A3).

Two delivery facts are defined here and nowhere else:

* a candidate bundle owns its policy by shipping a ``schedule`` entry named
  ``host/execution_geometry.json``.  :func:`candidate_geometry` parses that closed document into
  the A2 typed ``ExecutionGeometry``.  The entry is candidate Host Tiling, not a Track tiling
  table: the parser has no op name, shape or target branch, and a missing/multiple geometry entry
  is refused.  The build's frozen canonical geometry is a separate artifact at
  ``execution_geometry.json`` (A3), so candidate source and build product never collide.
* a multi-file candidate travels to the single-text agent/evaluator chain as one segmented
  envelope (:mod:`tcenv.track.envelope`).  The envelope is transport only: after decoding it is
  an ordinary ``CandidateArtifact`` and its ``candidate_sha256`` is the candidate-entry manifest
  digest, never ``sha256(envelope)``.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..contract_errors import GeometryError, TaskContractError
from ..schema import SchemaError, loads_strict
from .plane import (
    EXECUTION_GEOMETRY_SCHEMA_VERSION,
    CandidateArtifact,
    CandidateEntry,
    ExecutionGeometry,
    GeometryStage,
)

#: The candidate bundle's one geometry entry.  It lives under ``host/`` because it is candidate
#: Host Tiling; the *build product* canonical geometry uses the root path below, so the two never
#: collide in the artifact set.  The path is fixed so the builder, pruner, self-check and worker
#: all agree without a second manifest.
CANDIDATE_GEOMETRY_PATH = "host/execution_geometry.json"
CANDIDATE_GEOMETRY_MEDIA_TYPE = "application/json"
#: The build artifact that carries :meth:`CandidateArtifact.manifest_bytes`.
CANDIDATE_MANIFEST_PATH = "candidate_manifest.json"
#: The build artifact that carries the frozen canonical execution geometry (A3/A4).
EXECUTION_GEOMETRY_ARTIFACT_PATH = "execution_geometry.json"
CANDIDATE_MANIFEST_MEDIA_TYPE = "application/json"

_GEOMETRY_FIELDS = {"schema_version", "source", "stages"}
_STAGE_FIELDS = {
    "kernel",
    "grid",
    "threads",
    "threadgroup_bytes",
    "workspace",
    "when",
    "derived_scalars",
}
_STAGE_DEFAULTS: dict[str, Any] = {
    "threadgroup_bytes": 0,
    "workspace": (),
    "when": None,
    "derived_scalars": (),
}
_REQUIRED_STAGE_FIELDS = {"kernel", "grid", "threads"}


def parse_candidate_geometry(document: Any, *, where: str = CANDIDATE_GEOMETRY_PATH) -> ExecutionGeometry:
    """@brief 把候选交付的几何 JSON 解析成 typed ``ExecutionGeometry``。

    The document is the A1 field set and nothing else.  Expressions stay as JSON terms; A2's
    resolver is the only evaluator.  ``source`` must be ``candidate``: a fallback geometry is
    harness-owned and cannot arrive inside the candidate.

    @param document 候选几何文档。
    @param where 诊断位置。
    @return 候选 ``ExecutionGeometry``。
    @raises GeometryError 字段/版本/来源/stage 非法。
    """
    if not isinstance(document, Mapping):
        raise GeometryError(f"{where}: the candidate geometry must be an object")
    keys = set(document)
    if keys != _GEOMETRY_FIELDS:
        raise GeometryError(
            f"{where}: the candidate geometry has fields {sorted(keys)}; expected exactly {sorted(_GEOMETRY_FIELDS)}"
        )
    if document["schema_version"] != EXECUTION_GEOMETRY_SCHEMA_VERSION:
        raise GeometryError(
            f"{where}: unsupported candidate geometry schema {document['schema_version']!r}"
        )
    if document["source"] != "candidate":
        raise GeometryError(
            f"{where}: the candidate geometry must declare source 'candidate', got {document['source']!r}; "
            "a fallback geometry is harness-owned and never shipped by a candidate"
        )
    stages = document["stages"]
    if not isinstance(stages, (list, tuple)) or not stages:
        raise GeometryError(f"{where}.stages: a candidate geometry needs at least one stage")
    parsed = tuple(_parse_stage(stage, f"{where}.stages[{index}]") for index, stage in enumerate(stages))
    return ExecutionGeometry(source="candidate", stages=parsed)


def _parse_stage(stage: Any, where: str) -> GeometryStage:
    """@brief 解析一个 stage。

    @param stage stage 文档。
    @param where 诊断位置。
    @return stage。
    @raises GeometryError 字段缺失、开放或类型非法。
    """
    if not isinstance(stage, Mapping):
        raise GeometryError(f"{where}: a stage must be an object")
    keys = set(stage)
    unknown = sorted(keys - _STAGE_FIELDS)
    if unknown:
        raise GeometryError(f"{where}: a stage has unregistered fields {unknown}; allowed {sorted(_STAGE_FIELDS)}")
    missing = sorted(_REQUIRED_STAGE_FIELDS - keys)
    if missing:
        raise GeometryError(f"{where}: a stage is missing {missing}")
    kernel = stage["kernel"]
    if not isinstance(kernel, str) or not kernel:
        raise GeometryError(f"{where}.kernel: a stage needs a kernel name")
    for name in ("grid", "threads"):
        value = stage[name]
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise GeometryError(f"{where}.{name}: expected three terms")
    workspace = stage.get("workspace", _STAGE_DEFAULTS["workspace"])
    if not isinstance(workspace, (list, tuple)):
        raise GeometryError(f"{where}.workspace: expected a list of workspace declarations")
    derived = stage.get("derived_scalars", _STAGE_DEFAULTS["derived_scalars"])
    if not isinstance(derived, (list, tuple)):
        raise GeometryError(f"{where}.derived_scalars: expected a list of (name, term) pairs")
    pairs: list[tuple[str, Any]] = []
    for index, item in enumerate(derived):
        if not isinstance(item, Mapping):
            raise GeometryError(f"{where}.derived_scalars[{index}]: expected {{name, term}}")
        if set(item) != {"name", "term"}:
            raise GeometryError(
                f"{where}.derived_scalars[{index}]: expected exactly name and term, got {sorted(item)}"
            )
        pairs.append((str(item["name"]), item["term"]))
    return GeometryStage(
        kernel=kernel,
        grid=tuple(stage["grid"]),
        threads=tuple(stage["threads"]),
        threadgroup_bytes=stage.get("threadgroup_bytes", _STAGE_DEFAULTS["threadgroup_bytes"]),
        workspace=tuple(workspace),
        when=stage.get("when", _STAGE_DEFAULTS["when"]),
        derived_scalars=tuple(pairs),
    )


def candidate_geometry(candidate: CandidateArtifact) -> ExecutionGeometry:
    """@brief 从候选 bundle 的 ``schedule`` entry 取几何。

    Exactly one candidate entry with role ``schedule`` is allowed, its path is the fixed
    ``execution_geometry.json`` and its media type is ``application/json``.  A bundle that ships
    no geometry (or two) is a delivery error, not a reason for the harness to guess a tiling.

    @param candidate 候选。
    @return 候选几何。
    @raises GeometryError 缺失/重复/路径/媒体类型/JSON 非法。
    """
    schedules = [entry for entry in candidate.entries if entry.role == "schedule"]
    if not schedules:
        raise GeometryError(
            f"candidate {candidate.track_id} ships no schedule entry; a candidate bundle owns its geometry at "
            f"{CANDIDATE_GEOMETRY_PATH!r}"
        )
    if len(schedules) > 1:
        raise GeometryError(
            f"candidate {candidate.track_id} ships {len(schedules)} schedule entries; there must be exactly one "
            "execution geometry"
        )
    entry = schedules[0]
    if entry.path != CANDIDATE_GEOMETRY_PATH:
        raise GeometryError(
            f"candidate schedule entry is {entry.path!r}; the fixed candidate geometry path is "
            f"{CANDIDATE_GEOMETRY_PATH!r}"
        )
    if entry.media_type != CANDIDATE_GEOMETRY_MEDIA_TYPE:
        raise GeometryError(
            f"candidate geometry media type is {entry.media_type!r}, expected {CANDIDATE_GEOMETRY_MEDIA_TYPE!r}"
        )
    try:
        text = bytes(entry.data).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GeometryError(f"candidate geometry is not UTF-8 JSON: {exc}") from exc
    try:
        document = loads_strict(text, what=CANDIDATE_GEOMETRY_PATH)
    except SchemaError as exc:
        raise GeometryError(f"candidate geometry is not strict JSON: {exc}") from exc
    return parse_candidate_geometry(document)


def candidate_from_single_source(source: str, *, path: str = "kernel.metal") -> CandidateArtifact:
    """@brief 把单文件提交变成 ``msl-single`` 候选（身份 = 文件字节 SHA-256）。

    @param source 候选源码文本。
    @param path 候选 entry 路径。
    @return 候选。
    @raises TaskContractError 路径或字节非法。
    """
    if not isinstance(source, str) or not source.strip():
        raise TaskContractError("a single-source candidate needs non-empty source text")
    entry = CandidateEntry(
        path=path,
        role="device_kernel",
        media_type="text/plain",
        data=source.encode("utf-8"),
    )
    return CandidateArtifact(track_id="msl-single", entries=(entry,))


__all__ = [
    "CANDIDATE_GEOMETRY_MEDIA_TYPE",
    "CANDIDATE_GEOMETRY_PATH",
    "CANDIDATE_MANIFEST_MEDIA_TYPE",
    "CANDIDATE_MANIFEST_PATH",
    "EXECUTION_GEOMETRY_ARTIFACT_PATH",
    "candidate_from_single_source",
    "candidate_geometry",
    "parse_candidate_geometry",
]
