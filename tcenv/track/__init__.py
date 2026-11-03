"""Track binding: what the model reads and what the harness assembles for evaluation.

A track binding (``tasks/<op>/track.json``) is *not* a second semantic contract.  It
owns exactly the representation/oracle layer of plan §4.4: the spec-text template, the
known-good template, ``description``, the dtype spelling table, the registered
implementations (``gen_inputs``/``reference``/``error_scale``/baselines) and the
task-pack file set.  It never repeats a symbol value, a tolerance, a launch geometry or
a formula.

Prompt rendering has exactly three sources, and each placeholder declares one of them:

* ``spec_params``   — produced by ``instantiate`` (``{{M}}``, ``{{H}}``);
* ``dtype_spelling``— the track's versioned dtype table (``{{T}}``, ``{{TL}}``, ...);
* ``backend_params``— produced by ``lower`` from the registered LoweringProfile.

A placeholder that no declared source provides, that a second source also provides, or
that survives rendering is refused (F-22): the prompt is then guaranteed to agree with
the semantic instance and the derived contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..identity import IdentityError, load_strict_json
from ..lowering import Specialization
from ..opspec.model import SemanticInstance
from ..opspec.validator import SchemaFailure, validate

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRACK_SCHEMA_VERSION = 1
TRACK_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "track.v1.schema.json"

PLACEHOLDER_SOURCES = frozenset({"spec_params", "dtype_spelling", "backend_params"})
_PLACEHOLDER = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")


class TrackError(ValueError):
    """@brief Track 绑定非法：缺文件、未知占位符来源、占位符冲突或 pack 文件集非法。"""


def _relative_posix(value: str, where: str) -> str:
    """@brief 校验一个仓库相对 POSIX 路径。

    @param value 路径文本。
    @param where 诊断位置。
    @return 原路径文本。
    @raises TrackError 绝对路径、``..`` 或反斜杠。
    """
    if value.startswith("/") or "\\" in value or value.startswith("./"):
        raise TrackError(f"{where}: {value!r} must be a plain repo-relative POSIX path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise TrackError(f"{where}: {value!r} must not contain empty, '.' or '..' segments")
    return value


@dataclass(frozen=True)
class _Template:
    """@brief 一个模板及其占位符来源声明。"""

    path: str
    placeholders: Mapping[str, str]

    def text(self) -> str:
        """@brief 读取模板正文。

        @return 模板原文。
        @raises TrackError 文件不存在。
        """
        resolved = REPO_ROOT / self.path
        if not resolved.is_file():
            raise TrackError(f"template {self.path!r} does not exist")
        return resolved.read_text(encoding="utf-8")

    def found_placeholders(self) -> frozenset[str]:
        """@brief 模板里实际出现的占位符名。

        @return 占位符名集合。
        """
        return frozenset(_PLACEHOLDER.findall(self.text()))


@dataclass(frozen=True)
class TrackBinding:
    """@brief 一份通过 schema 与语义校验的 track 绑定。"""

    path: Path
    document: Mapping[str, Any]

    @property
    def track(self) -> str:
        """@brief track 名。

        @return ``document["track"]``。
        """
        return str(self.document["track"])

    @property
    def backend_id(self) -> str:
        """@brief 该 track 的目标后端 id。

        @return ``document["backend_id"]``。
        """
        return str(self.document["backend_id"])

    @property
    def lowering_profile(self) -> str:
        """@brief 登记的 LoweringProfile 名。

        @return ``document["lowering_profile"]``。
        """
        return str(self.document["lowering_profile"])

    @property
    def opspec_path(self) -> Path:
        """@brief OpSpec 的绝对路径。

        A generated track has no single OpSpec file (each ``(seed, size)`` materialises its own
        document), so ``opspec`` is ``null`` there and this accessor refuses instead of inventing
        a path.

        @return 解析后的路径。
        @raises TrackError 生成族 track 没有单一 OpSpec 文件。
        """
        if self.document["opspec"] is None:
            raise TrackError(f"{self.path}: a generated track has no single OpSpec file")
        return REPO_ROOT / str(self.document["opspec"])

    @property
    def description(self) -> str:
        """@brief 题面描述（只属于 track，不进 ``opspec_digest``）。

        @return 描述文本。
        """
        return str(self.document["description"])

    @property
    def dtype_spelling(self) -> Mapping[str, Mapping[str, str]]:
        """@brief 版本化的 dtype 拼写表。

        @return dtype → 占位符名 → 文本。
        """
        return self.document["dtype_spelling"]

    @property
    def implementations(self) -> Mapping[str, Any]:
        """@brief 登记的 ``gen_inputs``/``reference``/``error_scale`` 身份。

        @return 角色 → ``{implementation_id, implementation_version, params}`` 或 null。
        """
        return self.document["implementations"]

    @property
    def baselines(self) -> tuple[Mapping[str, Any], ...]:
        """@brief 登记的 baseline 实现身份。

        @return baseline 登记元组。
        """
        return tuple(self.document["baselines"])

    @property
    def pack_files(self) -> tuple[str, ...]:
        """@brief task-pack 文件集（R4 用它计算本地 ``task_pack_digest``）。

        @return 仓库相对 POSIX 路径元组。
        """
        return tuple(self.document["pack_files"])

    def _template_block(self, name: str) -> _Template:
        """@brief 取一个模板块；生成块不是模板。

        @param name ``spec_text`` 或 ``known_good``。
        @return 模板对象。
        @raises TrackError 该块由登记的生成器产生。
        """
        block = self.document[name]
        if "template" not in block:
            generator = block["generator"]
            raise TrackError(
                f"{self.path}: {name} is generated by {generator['implementation_id']} "
                f"v{generator['implementation_version']}, not a template"
            )
        return _Template(str(block["template"]), block["placeholders"])

    def _generator_block(self, name: str) -> Mapping[str, Any] | None:
        """@brief 取一个生成块（模板块返回 ``None``）。

        @param name ``spec_text`` 或 ``known_good``。
        @return 登记的生成器或 ``None``。
        """
        block = self.document[name]
        return block["generator"] if "template" not in block else None

    @property
    def spec_text(self) -> _Template:
        """@brief 题面模板。

        @return 模板对象。
        @raises TrackError 题面由登记的生成器产生。
        """
        return self._template_block("spec_text")

    @property
    def spec_text_generator(self) -> Mapping[str, Any] | None:
        """@brief 题面生成器登记（模板题面时为 ``None``）。

        @return ``{implementation_id, implementation_version, params}`` 或 ``None``。
        """
        return self._generator_block("spec_text")

    @property
    def known_good_generator(self) -> Mapping[str, Any] | None:
        """@brief known-good 生成器登记（模板源时为 ``None``）。

        @return ``{implementation_id, implementation_version, params}`` 或 ``None``。
        """
        return self._generator_block("known_good")

    @property
    def variant_factory_id(self) -> str | None:
        """@brief 该题登记的动态启动族 factory id（没有则 ``None``）。

        @return factory id 或 ``None``。
        """
        value = self.document.get("variant_factory")
        return None if value is None else str(value)

    @property
    def baseline_variant(self) -> _Template | None:
        """@brief 该题给 triton-msl 后端用的 baseline 变体模板（没有时为 ``None``）。

        @return 模板对象或 ``None``。
        """
        block = self.document.get("baseline_variant")
        if not block:
            return None
        return _Template(str(block["template"]), block["placeholders"])

    @property
    def baseline_variant_note(self) -> str:
        """@brief baseline 变体的说明文本。

        @return 说明；没有变体时为空串。
        """
        block = self.document.get("baseline_variant")
        return str(block["note"]) if block else ""

    @property
    def known_good(self) -> _Template:
        """@brief known-good 源码模板。

        @return 模板对象。
        """
        return self._template_block("known_good")

    def render(self, template: _Template, *, instance: SemanticInstance, specialization: Specialization) -> str:
        """@brief 用三类来源渲染一个模板。

        @param template 模板及其占位符声明。
        @param instance 语义实例（``spec_params`` 来源）。
        @param specialization 派生结果（``backend_params`` 来源）。
        @return 渲染后的文本。
        @raises TrackError 占位符未声明来源、两个来源同名、或渲染后仍有未解析占位符。
        """
        spelling = self.dtype_spelling.get(instance.dtype)
        if spelling is None:
            raise TrackError(f"{self.path}: dtype_spelling has no entry for dtype {instance.dtype!r}")
        values: dict[str, str] = {}
        for name, declared in template.placeholders.items():
            if declared not in PLACEHOLDER_SOURCES:
                raise TrackError(f"{template.path}: placeholder {name!r} declares unknown source {declared!r}")
            provided = {
                source: source_value
                for source, source_value in (
                    ("spec_params", instance.spec_params.get(name)),
                    ("backend_params", specialization.backend_params.get(name)),
                    ("dtype_spelling", spelling.get(name)),
                )
                if source_value is not None
            }
            if declared not in provided:
                raise TrackError(
                    f"{template.path}: placeholder {name!r} declares source {declared!r}, which has no value; "
                    f"sources that do have a value: {sorted(provided)}"
                )
            others = sorted(set(provided) - {declared})
            if others:
                raise TrackError(
                    f"{template.path}: placeholder {name!r} is ambiguous; declared {declared!r} but "
                    f"{others} also provide it"
                )
            values[name] = str(provided[declared])
        missing = sorted(template.found_placeholders() - set(values))
        if missing:
            raise TrackError(f"{template.path}: undeclared placeholders {missing}")
        declared_unused = sorted(set(values) - template.found_placeholders())
        if declared_unused:
            raise TrackError(f"{template.path}: declared placeholders {declared_unused} do not appear in the template")
        rendered = template.text()
        for name, value in values.items():
            rendered = rendered.replace("{{" + name + "}}", value)
        leftover = _PLACEHOLDER.findall(rendered)
        if leftover:
            raise TrackError(f"{template.path}: unresolved placeholders {sorted(set(leftover))}")
        return rendered

    def render_spec_text(self, instance: SemanticInstance, specialization: Specialization) -> str:
        """@brief 渲染题面（``triton_source``）。

        @param instance 语义实例。
        @param specialization 派生结果。
        @return 渲染后的题面。
        """
        return self.render(self.spec_text, instance=instance, specialization=specialization)

    def render_known_good(self, instance: SemanticInstance, specialization: Specialization) -> str:
        """@brief 渲染 known-good 源码（``naive_msl``/``reference_sources``）。

        @param instance 语义实例。
        @param specialization 派生结果。
        @return 渲染后的 known-good 源码。
        """
        return self.render(self.known_good, instance=instance, specialization=specialization)


def pack_digest(pack_files) -> str:
    """@brief 计算一组 task-pack 文件的确定性摘要（路径 + 字节，按路径排序）。

    One implementation: the evaluation cache key (R2) and the local ``task_pack_digest``
    (R4) must describe the same file set with the same bytes.

    @param pack_files 仓库相对 POSIX 路径序列。
    @return 十六进制 sha256。
    @raises TrackError 文件不存在或是符号链接。
    """
    # One implementation: the deployed evaluator hashes the same manifest document shape over
    # its pack directory (``tcenv/remote/task_pack.py``), so the local identity and the
    # deployment identity are never two different algorithms.
    from ..remote.task_pack import TaskPackError, local_pack_digest

    values = [_relative_posix(str(raw), "pack_digest") for raw in pack_files]
    try:
        return local_pack_digest(values, root=REPO_ROOT)
    except TaskPackError as exc:
        raise TrackError(f"pack_digest: {exc}") from exc


def load_track(path: str | Path) -> TrackBinding:
    """@brief 读取并校验一份 track 绑定。

    @param path ``track.json`` 路径。
    @return 通过 schema 与语义校验的 :class:`TrackBinding`。
    @raises TrackError 非法。
    """
    resolved = Path(path)
    try:
        document = load_strict_json(resolved)
        schema = load_strict_json(TRACK_SCHEMA_PATH)
    except IdentityError as exc:
        raise TrackError(f"{resolved}: {exc}") from exc
    try:
        validate(document, schema)
    except SchemaFailure as exc:
        raise TrackError(f"{resolved}: {exc}") from exc

    for block in ("spec_text", "known_good", "baseline_variant"):
        if block == "baseline_variant" and not document.get(block):
            continue
        if "generator" in document[block]:
            # A generated block registers the producer identity instead of a template.
            generator = document[block]["generator"]
            if not str(generator["implementation_id"]).strip():
                raise TrackError(f"{resolved}.{block}.generator: implementation_id must be non-empty")
            if int(generator["implementation_version"]) < 1:
                raise TrackError(f"{resolved}.{block}.generator: implementation_version must be >= 1")
            continue
        _relative_posix(str(document[block]["template"]), f"{resolved}.{block}.template")
    generated_blocks = {
        block for block in ("spec_text", "known_good") if "generator" in document[block]
    }
    if document["opspec"] is None:
        if generated_blocks != {"spec_text", "known_good"}:
            raise TrackError(
                f"{resolved}.opspec: null is only valid for a fully generated track "
                "(spec_text and known_good both declare a generator)"
            )
    else:
        # A generated text block is a *rendering* of the OpSpec, not a replacement for it: the
        # ``gen_*`` family has one semantic file per op and renders its prompt/known-good from
        # that document.  There is no second template to keep in sync, so both combinations are
        # coherent: ``opspec=null`` (no single semantic file, e.g. the seed-addressed integer
        # family) and ``opspec=<path>`` with both blocks generated (R2 N-31).
        if generated_blocks and generated_blocks != {"spec_text", "known_good"}:
            raise TrackError(
                f"{resolved}: spec_text and known_good must both be generated or both be "
                f"templates, got generators for {sorted(generated_blocks)}"
            )
        _relative_posix(str(document["opspec"]), f"{resolved}.opspec")
    for index, entry in enumerate(document["template_literals"]["launch"]):
        _relative_posix(str(entry["template"]), f"{resolved}.template_literals.launch[{index}]")
    for index, entry in enumerate(document["template_literals"]["semantic"]):
        _relative_posix(str(entry["template"]), f"{resolved}.template_literals.semantic[{index}]")

    declared = {
        str(block["template"])
        for block in (document["spec_text"], document["known_good"], document.get("baseline_variant") or {})
        if "template" in block
    }
    for index, entry in enumerate(document["template_literals"]["launch"] + document["template_literals"]["semantic"]):
        if str(entry["template"]) not in declared:
            raise TrackError(f"{resolved}: template_literals entry {index} names an unregistered template")

    seen: dict[str, str] = {}
    for raw in document["pack_files"]:
        value = _relative_posix(str(raw), f"{resolved}.pack_files")
        folded = value.casefold()
        if folded in seen:
            raise TrackError(f"{resolved}.pack_files: {value!r} collides with {seen[folded]!r}")
        seen[folded] = value
        absolute = REPO_ROOT / value
        if absolute.is_symlink():
            raise TrackError(f"{resolved}.pack_files: {value!r} must not be a symlink")
        if not absolute.is_file():
            raise TrackError(f"{resolved}.pack_files: {value!r} does not exist")

    binding = TrackBinding(path=resolved, document=document)
    templates = [binding._template_block(name) for name in ("spec_text", "known_good") if "template" in document[name]]
    if binding.baseline_variant is not None:
        templates.append(binding.baseline_variant)
    for template in templates:
        found = template.found_placeholders()
        declared_names = set(template.placeholders)
        if found != declared_names:
            raise TrackError(
                f"{resolved}: {template.path}: declared placeholders {sorted(declared_names)} but the template has "
                f"{sorted(found)}"
            )
    return binding


__all__ = [
    "REPO_ROOT",
    "TRACK_SCHEMA_PATH",
    "TRACK_SCHEMA_VERSION",
    "TrackBinding",
    "TrackError",
    "load_track",
    "pack_digest",
]

# The A1 track plane is re-exported from the same package as the v1 binding.  The two are
# different objects: ``TrackBinding`` is the per-task v1 representation/oracle binding, while
# ``TrackRegistry``/``TrackSpec`` describe task-agnostic *delivery mechanisms*.
from .plane import (  # noqa: E402  (kept after the v1 binding to keep its import order stable)
    CANDIDATE_ARTIFACT_SCHEMA_VERSION,
    CANDIDATE_MANIFEST_SCHEMA_ID,
    CANDIDATE_ROLES,
    EXECUTION_GEOMETRY_SCHEMA_VERSION,
    TENSOR_MEMORY_SCHEMA_VERSION,
    CandidateArtifact,
    CandidateEntry,
    ExecutionGeometry,
    GeometryStage,
    TensorMemoryArg,
    TensorMemoryContract,
    TrackRenderer,
)
from .registry import (  # noqa: E402
    METAL_TRACKS_PATH,
    TRACKS_SCHEMA_PATH,
    TRACKS_SCHEMA_VERSION,
    TrackImplementation,
    TrackRegistry,
    TrackSpec,
    load_track_registry,
    registry_from_document,
)

# A2: typed geometry, the ABI deriver, the canonical bundle and the read_write/workspace/timing
# protocol.  These are additions next to the frozen A1 plane; the A1 field sets are unchanged.
from .abi import (  # noqa: E402
    ABI_DERIVER_VERSION,
    REQUIRED_TARGET_LIMITS,
    TargetAbiDeriver,
    TargetAbiLimits,
    abi_limits_from_target,
    derive_tensor_memory_contract,
    derived_abi_digest,
    tensor_span_bytes,
)
from .bundle import CANONICAL_BUNDLE_SCHEMA_ID, CanonicalBundle  # noqa: E402
from .delivery import (  # noqa: E402
    CANDIDATE_GEOMETRY_MEDIA_TYPE,
    CANDIDATE_GEOMETRY_PATH,
    CANDIDATE_MANIFEST_MEDIA_TYPE,
    CANDIDATE_MANIFEST_PATH,
    candidate_from_single_source,
    candidate_geometry,
    parse_candidate_geometry,
)
from .envelope import (  # noqa: E402
    ENVELOPE_MAGIC,
    ENVELOPE_SCHEMA_ID,
    MAX_ENVELOPE_BYTES,
    MAX_ENVELOPE_ENTRIES,
    EnvelopeError,
    decode_envelope,
    encode_envelope,
    is_candidate_envelope,
)
from .geometry import (  # noqa: E402
    GEOMETRY_RESOLUTION_VERSION,
    GEOMETRY_SOURCES,
    GeometryPair,
    GeometrySelection,
    ResolvedGeometry,
    ResolvedStage,
    WorkspaceUse,
    dispatch_parity,
    execution_identity,
    geometry_pair,
    resolve_geometry,
    select_geometry,
    selection_parity,
)
from .lifecycle import (  # noqa: E402
    ENTRY_BOUND_ROLES,
    HOST_STEP_ROLES,
    LIFECYCLE_SCHEMA_VERSION,
    CandidateLifecycle,
    HostStep,
    cache_key,
    cache_scope,
)
from .protocol import (  # noqa: E402
    CORRECTNESS_TRIALS,
    GUARD_BYTES,
    GUARD_PATTERN,
    POISON,
    POISON_PATTERN,
    RESTORE_INITIAL,
    UNTOUCHED,
    ZERO,
    LaunchFillPlan,
    PrepareDeclaration,
    TensorFill,
    TimingDiscipline,
    benchmark_launch_plan,
    check_prepare_timing,
    correctness_launch_plan,
    unwritten_check_applies,
    workspace_fills,
)

__all__ += [
    "ABI_DERIVER_VERSION",
    "CANDIDATE_GEOMETRY_MEDIA_TYPE",
    "CANDIDATE_GEOMETRY_PATH",
    "CANDIDATE_MANIFEST_MEDIA_TYPE",
    "CANDIDATE_MANIFEST_PATH",
    "CANONICAL_BUNDLE_SCHEMA_ID",
    "CORRECTNESS_TRIALS",
    "ENTRY_BOUND_ROLES",
    "ENVELOPE_MAGIC",
    "ENVELOPE_SCHEMA_ID",
    "GEOMETRY_RESOLUTION_VERSION",
    "GEOMETRY_SOURCES",
    "GUARD_BYTES",
    "GUARD_PATTERN",
    "HOST_STEP_ROLES",
    "LIFECYCLE_SCHEMA_VERSION",
    "MAX_ENVELOPE_BYTES",
    "MAX_ENVELOPE_ENTRIES",
    "POISON",
    "POISON_PATTERN",
    "REQUIRED_TARGET_LIMITS",
    "RESTORE_INITIAL",
    "UNTOUCHED",
    "ZERO",
    "CandidateLifecycle",
    "CanonicalBundle",
    "EnvelopeError",
    "GeometryPair",
    "GeometrySelection",
    "HostStep",
    "LaunchFillPlan",
    "PrepareDeclaration",
    "ResolvedGeometry",
    "ResolvedStage",
    "TargetAbiDeriver",
    "TargetAbiLimits",
    "TensorFill",
    "TimingDiscipline",
    "WorkspaceUse",
    "abi_limits_from_target",
    "benchmark_launch_plan",
    "cache_key",
    "cache_scope",
    "candidate_from_single_source",
    "candidate_geometry",
    "check_prepare_timing",
    "correctness_launch_plan",
    "decode_envelope",
    "derive_tensor_memory_contract",
    "derived_abi_digest",
    "dispatch_parity",
    "encode_envelope",
    "execution_identity",
    "geometry_pair",
    "is_candidate_envelope",
    "parse_candidate_geometry",
    "resolve_geometry",
    "select_geometry",
    "selection_parity",
    "tensor_span_bytes",
    "unwritten_check_applies",
    "workspace_fills",
]

__all__ += [
    "CANDIDATE_ARTIFACT_SCHEMA_VERSION",
    "CANDIDATE_MANIFEST_SCHEMA_ID",
    "CANDIDATE_ROLES",
    "EXECUTION_GEOMETRY_SCHEMA_VERSION",
    "METAL_TRACKS_PATH",
    "TENSOR_MEMORY_SCHEMA_VERSION",
    "TRACKS_SCHEMA_PATH",
    "TRACKS_SCHEMA_VERSION",
    "CandidateArtifact",
    "CandidateEntry",
    "ExecutionGeometry",
    "GeometryStage",
    "TensorMemoryArg",
    "TensorMemoryContract",
    "TrackImplementation",
    "TrackRegistry",
    "TrackRenderer",
    "TrackSpec",
    "load_track_registry",
    "registry_from_document",
]
