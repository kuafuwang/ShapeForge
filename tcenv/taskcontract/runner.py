"""The v2 ``evaluate_task`` composition: locator → track → ABI → geometry → build → report (A3).

This is the integration seam, not a second evaluator.  A3 owns:
discovery/locator identity, formal track selection, the ABI derivation, candidate delivery
(single file or envelope), the independent fallback pairing, the canonical bundle, the frozen
geometry artifact and the report/evidence identity checks.  The *real* compile/correctness/timing
adapters are A4; they implement :class:`CandidateBuilder` and :class:`CandidateEvaluator` and are
injected here, so no vendor-specific or per-op tiling exists in this module.

Refusals are deliberate:

* a v2 task without a formal track, or an unknown/interface-only track, is refused (v1 is the
  only path with ``track_id=None``);
* a multi-file candidate's host work must be registered in a ``CandidateLifecycle``;
* the builder's source identity must equal the candidate manifest digest before geometry is
  attached; a report whose build digest differs from the attached artifact set is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from ..contract_errors import TaskContractError, UnsupportedContractCapability
from ..identity import fingerprint
from ..target import DEFAULT_TARGET_ID, load_target
from ..track import (
    CandidateArtifact,
    CandidateLifecycle,
    CanonicalBundle,
    ExecutionGeometry,
    GeometryPair,
    GeometrySelection,
    ResolvedGeometry,
    TargetAbiLimits,
    TrackRenderer,
    abi_limits_from_target,
    candidate_from_single_source,
    candidate_geometry,
    decode_envelope,
    derive_tensor_memory_contract,
    geometry_pair,
    is_candidate_envelope,
    load_track_registry,
    registry_from_document,
    resolve_geometry,
    select_geometry,
)
from ..track.registry import TrackSpec
from ..remote.artifact_builder import BuildOutcome
from ..remote.execution_geometry import (
    EvaluatorBuildError,
    attach_execution_geometry,
    verify_evaluator_build,
)
from .loader import TaskContract
from .locator import TaskLocator

EVALUATION_PLAN_SCHEMA_ID = "tcenv.task.evaluation.1"
EVALUATION_PLAN_SCHEMA_VERSION = 1
#: The frozen track-selection envelope a control host hands to an out-of-process evaluator.
FROZEN_TRACK_SCHEMA_ID = "tcenv.track.frozen.1"
FROZEN_TRACK_SCHEMA_VERSION = 1
#: Candidate entry roles that mean the candidate owns host-side work.
HOST_WORK_ROLES = frozenset({"host_tiling", "schedule"})
#: Report statuses that legitimately have no candidate build (the parent killed the worker or the
#: worker refused before building); they may not carry a fabricated ``full`` build.
BUILD_FREE_STATUSES = frozenset({"timeout", "crash", "refused_hang", "rejected"})


def _digest(value: Any, what: str) -> str:
    """@brief 断言一个 64 位小写十六进制摘要。

    @param value 候选值。
    @param what 诊断名。
    @return 原值。
    @raises TaskContractError 不是摘要。
    """
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise TaskContractError(f"{what} must be a 64-character lowercase hex digest")
    return value


@dataclass(frozen=True)
class TrackSelection:
    """@brief 一个已登记的正式 track（未知/未验证一律拒绝）。

    ``registry_document`` is the exact registry the control host selected from.  It travels to the
    out-of-process evaluator (``frozen_document``) so the worker selects the *same* TrackSpec from
    the *same* registry identity instead of re-reading a default ``tracks.json`` that may differ.
    """

    track: TrackSpec
    registry_path: str | None
    registry_document: Mapping[str, Any]
    #: True when this selection was reconstructed from a frozen control-host document.
    frozen: bool = False

    @property
    def registry_digest(self) -> str:
        """@brief registry 文档摘要。

        @return 十六进制 sha256。
        """
        return fingerprint(dict(self.registry_document))

    @property
    def track_digest(self) -> str:
        """@brief 该 TrackSpec 的摘要。

        @return 十六进制 sha256。
        """
        return fingerprint(self.track.document())

    def document(self) -> dict[str, Any]:
        """@brief 选择的 JSON 文档。

        @return 字段字典。
        """
        return {
            "track_id": self.track.track_id,
            "track_version": int(self.track.track_version),
            "delivery": self.track.delivery,
            "track_digest": self.track_digest,
            "registry_digest": self.registry_digest,
            "renderer": self.track.renderer.document(),
            "validator": self.track.validator.document(),
            "build_adapter": self.track.build_adapter.document(),
        }

    def frozen_document(self, *, presentation_sha256: str) -> dict[str, Any]:
        """@brief 交给外部 evaluator 的冻结 track/registry/渲染身份。

        The document names no host path: the worker rebuilds the registry from these bytes and
        verifies both digests.  ``presentation_sha256`` is the control host's rendering identity,
        which the worker must reproduce (and publish) before it may compile.

        @param presentation_sha256 控制机的题面渲染摘要。
        @return 冻结选择文档。
        """
        return {
            "schema_id": FROZEN_TRACK_SCHEMA_ID,
            "schema_version": FROZEN_TRACK_SCHEMA_VERSION,
            "track_id": self.track.track_id,
            "track_version": int(self.track.track_version),
            "track": self.track.document(),
            "track_digest": self.track_digest,
            "registry": dict(self.registry_document),
            "registry_digest": self.registry_digest,
            "presentation_sha256": _digest(presentation_sha256, "frozen presentation_sha256"),
        }


def _frozen_track(
    track_id: str | None,
    *,
    version: int | None,
    frozen: Mapping[str, Any],
) -> TrackSelection:
    """@brief 由控制机冻结的 registry/track 文档重建选择（不读任何默认 registry）。

    @param track_id 期望的正式 track id。
    @param version 期望版本；``None`` 取冻结版本。
    @param frozen 冻结选择文档。
    @return 选择。
    @raises TaskContractError 冻结文档、摘要、registry 或 track 不一致。
    @raises UnsupportedContractCapability track 不可用或未知。
    """
    if not isinstance(frozen, Mapping):
        raise TaskContractError("the frozen track document must be a mapping")
    if frozen.get("schema_id") != FROZEN_TRACK_SCHEMA_ID or frozen.get("schema_version") != FROZEN_TRACK_SCHEMA_VERSION:
        raise TaskContractError(
            f"the frozen track document is not {FROZEN_TRACK_SCHEMA_ID}/{FROZEN_TRACK_SCHEMA_VERSION}"
        )
    if track_id is None:
        raise UnsupportedContractCapability("a v2 task has no implicit Track: a frozen selection still needs a track_id")
    frozen_id = str(frozen.get("track_id"))
    if frozen_id != str(track_id):
        raise TaskContractError(
            f"the frozen track selection is for {frozen_id!r}, not the requested track {track_id!r}"
        )
    frozen_version = frozen.get("track_version")
    if not isinstance(frozen_version, int) or isinstance(frozen_version, bool):
        raise TaskContractError("the frozen track_version must be an integer")
    if version is not None and int(version) != frozen_version:
        raise TaskContractError(
            f"the frozen track is version {frozen_version}, not the requested version {version}"
        )
    registry_document = frozen.get("registry")
    if not isinstance(registry_document, Mapping):
        raise TaskContractError("the frozen track document has no registry document")
    registry = registry_from_document(registry_document)
    recomputed_registry = fingerprint(registry.document())
    declared_registry = _digest(frozen.get("registry_digest"), "frozen registry_digest")
    if recomputed_registry != declared_registry:
        raise TaskContractError(
            "the frozen registry document does not hash to the control-host registry_digest; a worker may not "
            "select from a different registry"
        )
    frozen_track = frozen.get("track")
    if not isinstance(frozen_track, Mapping):
        raise TaskContractError("the frozen track document has no track record")
    declared_track = _digest(frozen.get("track_digest"), "frozen track_digest")
    if fingerprint(dict(frozen_track)) != declared_track:
        raise TaskContractError("the frozen track record does not hash to the control-host track_digest")
    track = registry.get(frozen_id, version=frozen_version)
    if track.document() != dict(frozen_track):
        raise TaskContractError(
            "the frozen track record is not the track the frozen registry registers; the worker selection drifted"
        )
    track.require_available()
    return TrackSelection(
        track=track,
        registry_path=None,
        registry_document=registry.document(),
        frozen=True,
    )


def select_track(
    track_id: str | None,
    *,
    version: int | None = None,
    registry_path: str | Path | None = None,
    frozen: Mapping[str, Any] | None = None,
) -> TrackSelection:
    """@brief 按正式 id/version 选择 track。

    @param track_id 正式 track id；v2 不允许 None。
    @param version 要求的版本；未知版本拒绝。
    @param registry_path registry 路径；缺省 Metal ``tracks.json``。
    @param frozen 控制机冻结的 track/registry 文档；提供时拒绝再读任何 registry 路径。
    @return 选择。
    @raises UnsupportedContractCapability v2 无 track 或 track 无真实证据。
    @raises TaskContractError 未知 track/版本或冻结身份漂移。
    """
    if frozen is not None:
        if registry_path is not None:
            raise TaskContractError(
                "a frozen track selection may not also read a registry path; the worker must not select from a "
                "default or host-local registry"
            )
        return _frozen_track(track_id, version=version, frozen=frozen)
    if track_id is None:
        raise UnsupportedContractCapability(
            "a v2 task has no implicit Track: pass a formal track_id. Only the v1 path may use track_id=None."
        )
    registry = load_track_registry(registry_path) if registry_path is not None else load_track_registry()
    track = registry.get(track_id, version=version)
    track.require_available()
    return TrackSelection(
        track=track,
        registry_path=str(registry_path) if registry_path is not None else None,
        registry_document=registry.document(),
    )


@runtime_checkable
class GeometryResolver(Protocol):
    """@brief 候选交付 → typed ``ExecutionGeometry``（msl-single 的 launch 解析在这里注入）。"""

    def __call__(self, candidate: CandidateArtifact) -> ExecutionGeometry:
        """@brief 解析候选几何。

        @param candidate 候选。
        @return 候选几何。
        """
        ...


@runtime_checkable
class CandidateBuilder(Protocol):
    """@brief 真实构建适配器（A4 的 Metal 实现）。"""

    def build(self, request: "CandidateBuildRequest") -> BuildOutcome:
        """@brief 构建候选。

        @param request 构建请求。
        @return 构建结果。
        """
        ...


@runtime_checkable
class CandidateEvaluator(Protocol):
    """@brief 真实正确性/计时适配器（A4 的 Metal 实现）。"""

    def evaluate(self, request: "CandidateEvaluationRequest") -> Mapping[str, Any]:
        """@brief 评测已附加几何的构建结果。

        @param request 评测请求。
        @return schema-4 报告文档（至少含 build provenance/candidate identity）。
        """
        ...


@dataclass(frozen=True)
class CandidateBuildRequest:
    """@brief 一次 v2 构建的完整输入（只有候选和几何拥有算法）。

    ``track_selection`` carries the frozen registry/track identity the control host selected, so
    an out-of-process evaluator can reproduce the same TrackSpec instead of loading its default
    registry.  The TrackSpec itself stays a delivery mechanism, never an algorithm source.
    """

    locator: TaskLocator
    contract: TaskContract
    abi: Any
    limits: TargetAbiLimits
    track_selection: TrackSelection
    candidate: CandidateArtifact
    pair: GeometryPair
    selection: GeometrySelection
    canonical_bundle: CanonicalBundle
    target_id: str
    presentation_sha256: str

    @property
    def track(self) -> TrackSpec:
        """@brief 选定的 TrackSpec。

        @return track。
        """
        return self.track_selection.track

    def document(self) -> dict[str, Any]:
        """@brief 构建请求投影（不含字节）。

        @return 字段字典。
        """
        return {
            "task": self.locator.document(),
            "track": {
                "track_id": self.track.track_id,
                "track_version": self.track.track_version,
                "track_digest": self.track_selection.track_digest,
                "registry_digest": self.track_selection.registry_digest,
            },
            "target_id": self.target_id,
            "abi_id": self.abi.abi_id,
            "candidate_sha256": self.candidate.candidate_sha256,
            "geometry_identity": self.selection.identity,
            "provider": self.selection.provider,
            "bundle_digest": self.canonical_bundle.bundle_digest,
            "presentation_sha256": self.presentation_sha256,
        }


@dataclass(frozen=True)
class CandidateEvaluationRequest:
    """@brief 一次 v2 评测的输入（构建产物 + 选定几何）。

    ``build`` is ``None`` when the evaluator owns the first real build: the control host must not
    compile the candidate just to hand the evaluator an identity it will not execute.
    """

    build: BuildOutcome | None
    build_request: CandidateBuildRequest
    canonical_bundle: CanonicalBundle
    geometry: ResolvedGeometry

    def document(self) -> dict[str, Any]:
        """@brief 评测请求投影。

        @return 字段字典。
        """
        return {
            "candidate_sha256": (
                self.build.provenance.source_sha256
                if self.build is not None
                else self.build_request.candidate.candidate_sha256
            ),
            "artifact_set_digest": None if self.build is None else self.build.artifact_set_digest,
            "native_digest": (
                None
                if self.build is None or self.build.bundle.native is None
                else self.build.bundle.native.digest
            ),
            "geometry_identity": self.canonical_bundle.geometry_identity,
            "used_fallback": self.build_request.selection.used_fallback,
        }


@dataclass(frozen=True)
class TaskEvaluationResult:
    """@brief v2 一次 evaluate_task 的可重放身份与报告。

    ``artifact_set_digest``/``native_digest`` are ``None`` only when no candidate build exists
    (the evaluator's bounded worker was killed or the candidate never reached the compiler); a
    timed-out or crashed run never carries a synthesised ``full`` build.
    """

    locator: Mapping[str, Any]
    track: Mapping[str, Any]
    candidate_sha256: str
    task_pack_digest: str
    effective_correctness_only: bool
    geometry_identity: str
    bundle_digest: str
    presentation_sha256: str
    artifact_set_digest: str | None
    native_digest: str | None
    report: Mapping[str, Any]
    #: A4 execution record (actual runtime dispatch/provider/host tiling).  It is deliberately
    #: not part of the published schema-4 report and not part of ``document()``; it is the
    #: evaluator-side evidence that a real worker ran the selected geometry.
    execution: Mapping[str, Any] | None = None
    #: The build record the evaluator actually produced and executed (``full`` or ``partial``).
    #: It is the same object projected into ``report["build"]``; it is excluded from
    #: ``document()`` because the published report already carries it.
    build: Mapping[str, Any] | None = None
    schema_version: int = EVALUATION_PLAN_SCHEMA_VERSION

    def document(self) -> dict[str, Any]:
        """@brief 结果文档（可供 run_config schema 3 直接取字段）。

        @return 字段字典。
        """
        return {
            "schema_id": EVALUATION_PLAN_SCHEMA_ID,
            "schema_version": self.schema_version,
            "locator": dict(self.locator),
            "track": dict(self.track),
            "candidate_sha256": self.candidate_sha256,
            "task_pack_digest": self.task_pack_digest,
            "effective_correctness_only": bool(self.effective_correctness_only),
            "geometry_identity": self.geometry_identity,
            "bundle_digest": self.bundle_digest,
            "presentation_sha256": self.presentation_sha256,
            "artifact_set_digest": self.artifact_set_digest,
            "native_digest": self.native_digest,
            "report": dict(self.report),
        }

    @property
    def identity(self) -> str:
        """@brief 结果摘要。

        @return 十六进制 sha256。
        """
        return fingerprint(self.document())


def default_geometry_resolver(track: TrackSpec) -> GeometryResolver:
    """@brief 默认几何来源：候选 bundle 的 ``host/execution_geometry.json``。

    A single-file track's launch declaration is a backend adapter (A4); without one, the
    evaluation is refused rather than guessing a launch geometry.

    @param track 目标 track。
    @return resolver。
    @raises UnsupportedContractCapability msl-single 默认没有解析器。
    """
    if track.delivery == "candidate_bundle":
        return candidate_geometry
    raise UnsupportedContractCapability(
        f"track {track.track_id}: a single-file candidate's launch geometry needs the backend adapter; "
        "inject a GeometryResolver (A4) instead of letting the harness invent a tiling"
    )


def _normalize_report(report: Any) -> dict[str, Any]:
    """@brief 用唯一的 schema-4 读取器验证并规范化评测报告。

    A3 must not accept an arbitrary mapping with two matching digests: the report is the published
    evidence object, so it goes through ``EvalReport.from_json`` (the same closed reader used for
    every persisted report).  A missing version, a missing field or a free field is refused.

    @param report evaluator 返回值（``EvalReport`` 或 mapping）。
    @return schema-4 报告字典（已闭集校验）。
    @raises TaskContractError 不是合法 schema-4 报告。
    """
    from ..evaluate import EvalReport
    from ..schema import SchemaError

    if isinstance(report, EvalReport):
        document: Any = report.to_json()
    elif isinstance(report, Mapping):
        document = dict(report)
    else:
        raise TaskContractError(
            "the candidate evaluator must return an EvalReport or a schema-4 report mapping"
        )
    try:
        parsed = EvalReport.from_json(document)
    except SchemaError as exc:
        raise TaskContractError(f"the candidate evaluator report is not a valid schema-4 EvalReport: {exc}") from exc
    return parsed.to_json()


def _report_build_document(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """@brief 取 schema-4 报告里的 build 文档。

    @param report 已规范化的报告。
    @return build 文档。
    @raises TaskContractError 报告没有 schema-4 build 证据。
    """
    document: Any = report
    if not isinstance(document, Mapping):
        raise TaskContractError("the candidate evaluator must return a report mapping with build evidence")
    build = document.get("build")
    if not isinstance(build, Mapping):
        # A parent-owned external deadline/crash has no candidate build to bind.  The
        # TaskEvaluationResult still carries the attached identity; only a normal candidate
        # status is required to have a build record.
        if document.get("status") in BUILD_FREE_STATUSES:
            return {}
        raise TaskContractError(
            "the candidate evaluator report has no build record; v2 evidence needs the schema-4 build identity"
        )
    return build


def _verify_report_identity(
    report: Mapping[str, Any],
    *,
    artifact_set_digest: str | None,
    candidate_sha256: str,
) -> None:
    """@brief 报告必须绑定实际执行的 artifact set / 候选身份（含无构建的 compile 失败）。

    @param report 报告文档。
    @param artifact_set_digest 实际执行的、含冻结几何的 artifact set；无构建时为 None。
    @param candidate_sha256 候选身份。
    @raises TaskContractError 报告身份漂移。
    """
    document = _report_build_document(report)
    if not document:
        return
    provenance = document.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TaskContractError("the report build record has no provenance")
    if provenance.get("source_sha256") != candidate_sha256:
        raise TaskContractError("the report's build source identity is not the candidate identity")
    if document.get("candidate_sha256") != candidate_sha256:
        raise TaskContractError("the report's build candidate_sha256 is not the candidate identity")
    if document.get("shape") == "partial":
        if provenance.get("artifact_set_digest") is not None:
            raise TaskContractError("a partial compile-failure build must not claim an artifact set")
        if artifact_set_digest is not None:
            raise TaskContractError(
                "the report claims a partial build but the evaluated build has an artifact set"
            )
        return
    if provenance.get("artifact_set_digest") != artifact_set_digest:
        raise TaskContractError(
            "the report's artifact_set_digest is not the attached geometry-bearing artifact set; the confirmation "
            "gate would compare a different build subject"
        )


def evaluate_task(
    locator: TaskLocator,
    *,
    track_id: str | None,
    builder: CandidateBuilder,
    evaluator: CandidateEvaluator,
    candidate_source: str | None = None,
    candidate: CandidateArtifact | None = None,
    geometry_resolver: GeometryResolver | None = None,
    fallback_geometry: ExecutionGeometry | None = None,
    lifecycle: CandidateLifecycle | None = None,
    lifecycle_factory: Callable[[ResolvedGeometry], CandidateLifecycle] | None = None,
    track_version: int | None = None,
    track_registry_path: str | Path | None = None,
    frozen_track: Mapping[str, Any] | None = None,
    target_id: str = DEFAULT_TARGET_ID,
    max_buffer_length: int | None = None,
) -> TaskEvaluationResult:
    """@brief 装配、构建、附加冻结几何并评测一个 v2 任务。

    When the evaluator reports ``owns_build`` (the A4 bounded worker owns the first real compile),
    this function only runs the builder's pre-compile checks here.  It never compiles the candidate
    in the control host and never invents the build the worker executed: the worker returns its
    real schema-4 build record and this function cross-audits it against its own canonical bundle
    (candidate bytes, frozen geometry, lifecycle, native descriptor audit and identity).

    @param locator 已发现/重建的任务。
    @param track_id 正式 track id（v2 不允许 None）。
    @param builder 真实构建适配器。
    @param evaluator 真实评测适配器。
    @param candidate_source 提交文本（单文件或分段 envelope）。
    @param candidate 已解析候选（与 source 二选一）。
    @param geometry_resolver 候选几何解析器；bundle 缺省用固定 geometry entry。
    @param fallback_geometry 独立 fallback（可选）。
    @param lifecycle 候选 prepare/run 生命周期（host work 必须登记）。
    @param lifecycle_factory 由选定几何构造生命周期（调用方无法提前知道解析结果时使用）。
    @param track_version 要求版本。
    @param track_registry_path registry 路径。
    @param frozen_track 控制机冻结的 track/registry/渲染身份（外部 evaluator 必须使用同一份）。
    @param target_id 目标。
    @param max_buffer_length 设备缓冲长度。
    @return 评测结果。
    @raises UnsupportedContractCapability/ TaskContractError/ GeometryError 任一不满足。
    """
    if not isinstance(locator, TaskLocator):
        raise TaskContractError("evaluate_task needs a TaskLocator")
    if (candidate_source is None) == (candidate is None):
        raise TaskContractError("pass exactly one of candidate_source or candidate")
    if frozen_track is not None and track_registry_path is not None:
        raise TaskContractError(
            "a frozen track selection may not also name a registry path; the evaluator must not re-read a host "
            "registry that may differ from the control host's"
        )
    selection = select_track(
        track_id, version=track_version, registry_path=track_registry_path, frozen=frozen_track
    )
    contract = locator.load()
    track = selection.track

    if candidate is None:
        assert candidate_source is not None
        if is_candidate_envelope(candidate_source):
            candidate = decode_envelope(candidate_source)
        elif track.delivery == "single_source":
            candidate = candidate_from_single_source(candidate_source)
        else:
            raise TaskContractError(
                f"track {track.track_id} delivers a candidate bundle; submit the segmented envelope, not one file"
            )
    candidate.validate_for(track)
    if candidate.track_id != track.track_id:
        raise TaskContractError("the decoded candidate and the selected track disagree")

    profile = load_target(target_id)
    abi = derive_tensor_memory_contract(
        contract.instance, target=profile, max_buffer_length=max_buffer_length
    )
    limits = abi_limits_from_target(profile, max_buffer_length=max_buffer_length)
    # The track's registered renderer is the one renderer for this task.  It only formats the
    # declared instance/reference/ABI; it cannot contribute a tiling or a per-op template.
    presentation = TrackRenderer.for_track(track).render(
        backend_id=profile.backend,
        instance=contract.instance,
        reference_source=contract.reference.source,
        contract=abi,
        display=contract.display,
        track=track,
    )
    if presentation.statement != contract.statement:
        raise TaskContractError(
            "the track renderer's statement differs from the assembled task statement; the v2 statement has one "
            "authoritative rendering"
        )
    if presentation.abi_document != abi.document():
        raise TaskContractError("the track renderer dropped or changed the derived TensorMemoryContract")
    presentation_sha256 = fingerprint(presentation.as_document() if hasattr(presentation, "as_document") else {
        "track_id": presentation.track_id,
        "track_version": presentation.track_version,
        "statement": presentation.statement,
        "reference_source": presentation.reference_source,
        "abi_document": presentation.abi_document,
    })
    if frozen_track is not None and frozen_track.get("presentation_sha256") != presentation_sha256:
        raise TaskContractError(
            "the rendering produced here is not the control host's frozen presentation identity; the worker may "
            "not evaluate a task statement the control host did not render"
        )
    resolver = geometry_resolver or default_geometry_resolver(track)
    geometry = resolver(candidate)
    if geometry.source != "candidate":
        raise TaskContractError("the candidate geometry resolver returned a non-candidate geometry")
    pair = geometry_pair(geometry, fallback_geometry, instance=contract.instance, contract=abi, limits=limits)
    chosen = select_geometry(pair)

    if (lifecycle is None) == (lifecycle_factory is None):
        if lifecycle is not None and lifecycle_factory is not None:
            raise TaskContractError("pass either lifecycle or lifecycle_factory, not both")
    elif lifecycle_factory is not None:
        lifecycle = lifecycle_factory(chosen.geometry)
    if lifecycle is not None:
        lifecycle.validate_for(candidate)
        if lifecycle.geometry.document() != chosen.geometry.document():
            raise TaskContractError("the lifecycle was registered for a different resolved geometry")
    elif any(entry.role in HOST_WORK_ROLES for entry in candidate.entries):
        raise UnsupportedContractCapability(
            "candidate-owned host work (Host Tiling/schedule) must be registered in a CandidateLifecycle; "
            "a lifecycle cannot be omitted for a bundle that ships host work"
        )

    canonical = CanonicalBundle(candidate=candidate, pair=pair, lifecycle=lifecycle)
    request = CandidateBuildRequest(
        locator=locator,
        contract=contract,
        abi=abi,
        limits=limits,
        track_selection=selection,
        candidate=candidate,
        pair=pair,
        selection=chosen,
        canonical_bundle=canonical,
        target_id=target_id,
        presentation_sha256=presentation_sha256,
    )
    # An evaluator that owns the build (the A4 bounded v2 worker) performs the candidate's first
    # real compile inside its own process.  The control host only runs the pre-compile checks; a
    # stuck compile or launch is then bounded by the evaluator's external supervisor, not by this
    # process, and the candidate is compiled exactly once.
    owns_build = bool(getattr(evaluator, "owns_build", False))
    attached: BuildOutcome | None = None
    if owns_build:
        prepare = getattr(builder, "prepare", None)
        if not callable(prepare):
            raise TaskContractError(
                "this evaluator performs the candidate's first real build in its bounded worker; the injected "
                "builder must implement prepare() for the pre-build contract checks"
            )
        prepare(request)
    else:
        built = builder.build(request)
        if built.provenance.source_sha256 != candidate.candidate_sha256:
            raise TaskContractError(
                "the builder's source identity is not the candidate manifest digest; the candidate identity is fixed "
                "before the build"
            )
        attached = attach_execution_geometry(built, canonical_bundle=canonical)
    evaluation = CandidateEvaluationRequest(
        build=attached, build_request=request, canonical_bundle=canonical, geometry=chosen.geometry
    )
    report = _normalize_report(evaluator.evaluate(evaluation))
    status = str(report.get("status"))
    build_record: Mapping[str, Any] | None
    if owns_build:
        identity = {
            "opspec_digest": contract.spec.digest,
            "lowering_profile_digest": contract.fingerprint(),
        }
        executed = getattr(evaluator, "evaluated_build", None)
        if executed is None:
            if status not in BUILD_FREE_STATUSES:
                raise TaskContractError(
                    f"the bounded evaluator returned a {status!r} report without the build it actually executed"
                )
            artifact_set_digest: str | None = None
            native_digest: str | None = None
            build_record = None
        else:
            try:
                verified = verify_evaluator_build(
                    executed, canonical_bundle=canonical, identity=identity
                )
            except EvaluatorBuildError as exc:
                raise TaskContractError(
                    f"the bounded evaluator's build is not the build the control host froze: {exc}"
                ) from exc
            if dict(report.get("build") or {}) != dict(verified.document):
                raise TaskContractError(
                    "the bounded evaluator's report build is not the build record its worker executed; the report "
                    "identity drifted instead of being published by the process that built it"
                )
            _verify_report_identity(
                report,
                artifact_set_digest=verified.artifact_set_digest,
                candidate_sha256=candidate.candidate_sha256,
            )
            artifact_set_digest = verified.artifact_set_digest
            native_digest = verified.native_digest
            build_record = dict(verified.document)
    else:
        assert attached is not None
        _verify_report_identity(
            report, artifact_set_digest=attached.artifact_set_digest, candidate_sha256=candidate.candidate_sha256
        )
        artifact_set_digest = attached.artifact_set_digest
        native_digest = attached.bundle.native.digest if attached.bundle.native is not None else None
        build_record = report.get("build") if isinstance(report.get("build"), Mapping) else None
    execution = getattr(evaluator, "last_execution", None)
    if owns_build and status not in BUILD_FREE_STATUSES:
        worker = execution.get("worker") if isinstance(execution, Mapping) else None
        if not isinstance(worker, Mapping):
            raise TaskContractError(
                "the bounded evaluator did not publish which TrackSpec and rendering its worker actually used"
            )
        expected = selection.document()
        if (
            worker.get("track_digest") != expected["track_digest"]
            or worker.get("registry_digest") != expected["registry_digest"]
            or worker.get("track_id") != expected["track_id"]
            or int(worker.get("track_version") or 0) != int(expected["track_version"])
            or worker.get("presentation_sha256") != presentation_sha256
        ):
            raise TaskContractError(
                "the bounded worker's TrackSpec/registry/rendering identity drifted from the control host's frozen "
                "selection; the evidence would belong to a different Track"
            )
    return TaskEvaluationResult(
        locator=locator.document(),
        track=selection.document(),
        candidate_sha256=candidate.candidate_sha256,
        task_pack_digest=locator.deployment_pack_digest,
        effective_correctness_only=contract.correctness_only,
        geometry_identity=canonical.geometry_identity,
        bundle_digest=canonical.bundle_digest,
        presentation_sha256=presentation_sha256,
        artifact_set_digest=artifact_set_digest,
        native_digest=native_digest,
        report=dict(report),
        execution=None if execution is None else dict(execution),
        build=None if build_record is None else dict(build_record),
    )


__all__ = [
    "BUILD_FREE_STATUSES",
    "CandidateBuildRequest",
    "CandidateBuilder",
    "CandidateEvaluationRequest",
    "CandidateEvaluator",
    "EVALUATION_PLAN_SCHEMA_ID",
    "EVALUATION_PLAN_SCHEMA_VERSION",
    "FROZEN_TRACK_SCHEMA_ID",
    "FROZEN_TRACK_SCHEMA_VERSION",
    "GeometryResolver",
    "HOST_WORK_ROLES",
    "TaskEvaluationResult",
    "TrackSelection",
    "default_geometry_resolver",
    "evaluate_task",
    "select_track",
]
