"""MT1b: the TileLang Track adapter — Track contract validation and the source-only boundary.

This is the backend side of the TileLang ``kernel_pipeline`` Track.  It is deliberately lazy and
pure: importing it does **not** import ``tilelang``, ``torch``, ``hmz`` or a Metal runtime, and
nothing here compiles or dispatches.  The real first build belongs to the bounded v2 worker
(reusing the existing lower-level hmz Profile in :mod:`tcenv.worker_isolation`); this module owns
the contract-side checks and the typed refusal that keeps an ``interface_only`` Track from
pretending to be a build.

Three services are real, and they are the ones the rest of the harness must call:

* the **validator** — the frozen candidate source contract (path/media type/role/entry set), not
  an MSL parse;
* the **builder** — pre-build contract checks plus the real source-only build
  (:class:`~tcenv.backends.metal.tilelang_build.TileLangSourceOnlyBuild`): the staged frontend is
  hashed on the control host, the codegen runs inside the existing lower-level hmz Profile, and
  :func:`expected_build_identity` rebuilds the expected subject from independent inputs, whose
  cache key binds the whole candidate manifest (so the geometry), the DSL source, the frontend
  version/entry, options and the device-independent environment;
* the **geometry resolver** — the candidate-owned kernel_pipeline geometry document.  It is the
  shared closed ``ExecutionGeometry`` plan, not an MSL Host Tiling plan, so this Track does not
  borrow ``msl-bundle``'s contract.

Everything else (a Metal evaluator, a fresh Metal confirmation, a performance claim) is a typed
``unsupported_contract_capability`` until MT2 evaluates a real slice.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...contract_errors import TaskContractError, UnsupportedContractCapability
from ...track import tilelang as contract
from ...track import tilelang_confirmation as fresh_contract
from ...track.delivery import candidate_geometry
from ...track.registry import TrackSpec, load_track_registry


class TileLangTrackValidator:
    """@brief TileLang 候选交付校验器（不使用 MSL 合同）。"""

    def __init__(self, spec: TrackSpec) -> None:
        """@brief 构造时先做注册⇄合同核对。

        @param spec 已登记的 TileLang TrackSpec。
        @raises TileLangContractError 登记与冻结合同不一致。
        """
        if spec.track_version == fresh_contract.TILELANG_CONFIRMATION_TRACK_VERSION:
            fresh_contract.audit_fresh_registration(spec)
            self.source_spec = load_track_registry().get(
                contract.TILELANG_TRACK_ID, version=contract.TILELANG_METAL_TRACK_VERSION
            )
        else:
            contract.audit_registration(spec)
            self.source_spec = spec
        self.spec = spec

    def __call__(self, candidate: Any) -> str:
        """@brief 校验一个候选。

        @param candidate ``CandidateArtifact``。
        @return ``"accepted"``。
        """
        return contract.validate_candidate(candidate, self.source_spec)

    def validate(self, candidate: Any) -> str:
        """@brief ``__call__`` 的显式别名。

        @param candidate ``CandidateArtifact``。
        @return ``"accepted"``。
        """
        return self(candidate)


def _check_request(request: Any, spec: TrackSpec) -> None:
    """@brief 构建请求必须属于本 Track 的同一冻结选择。

    @param request ``CandidateBuildRequest``（鸭子类型）。
    @param spec 已登记的 TileLang TrackSpec。
    @raises TaskContractError 选择的 Track/候选身份不一致。
    """
    track = getattr(getattr(request, "track_selection", None), "track", None)
    if track is None or track.key != spec.key:
        raise TaskContractError(
            f"the build request selects {getattr(track, 'key', None)!r}, not the frozen TileLang Track {spec.key!r}; "
            "a DSL build never runs under another Track's identity"
        )
    if track.document() != spec.document():
        raise TaskContractError("the build request's TrackSpec is not the frozen TrackSpec the adapter audited")
    candidate = getattr(request, "candidate", None)
    if candidate is None or str(getattr(candidate, "track_id", "")) != spec.track_id:
        raise TaskContractError("the build request's candidate does not belong to the TileLang Track")
    geometry = getattr(getattr(request, "selection", None), "geometry", None)
    if geometry is None or str(getattr(geometry, "source", "")) != "candidate":
        raise TaskContractError(
            "a TileLang candidate owns a real candidate geometry; the build request has no candidate execution geometry"
        )


class TileLangSourceOnlyBuilder:
    """@brief TileLang 构建适配器：source-only 走真实围栏构建，Metal build 仍 typed unavailable。"""

    def __init__(
        self,
        spec: TrackSpec,
        *,
        attempt_base: Any = None,
        staging: Any = None,
        executor: Any = None,
        limits: Any = None,
    ) -> None:
        """@brief 构造（不在控制机上执行前端）。

        @param spec 已登记的 TileLang TrackSpec。
        @param attempt_base 可写尝试根目录的父目录。
        @param staging staged frontend 根目录；None 时按环境变量/MT0 默认解析。
        @param executor 围栏执行器（测试可注入）。
        @param limits 资源上限。
        @raises TileLangContractError 登记与冻结合同不一致。
        """
        contract.audit_registration(spec)
        self.spec = spec
        self._attempt_base = attempt_base
        self._staging = staging
        self._executor = executor
        self._limits = limits

    def prepare(self, request: Any) -> None:
        """@brief 编译前的合同检查（不编译、不导入前端）。

        @param request ``CandidateBuildRequest``。
        @raises TaskContractError 交付/Track 身份非法。
        """
        _check_request(request, self.spec)
        contract.validate_candidate(request.candidate, self.spec)

    def build(self, request: Any) -> Any:
        """@brief 真实 source-only 构建（围栏内 codegen）；Metal build 是 typed 拒绝。

        The returned document is a source-only *build record*: it has no Metal library, no
        pipeline, no dispatch and no schema-4 evaluation report.  MT2 owns the Metal slice.

        @param request ``CandidateBuildRequest``。
        @return ``SourceOnlyBuildOutcome``。
        @raises TileLangFrontendUnavailable 前端/hmz 能力缺失（typed capability gap）。
        @raises TileLangIdentityDrift 前端安装与冻结身份不符。
        @raises TileLangCodegenError codegen 超时/非零。
        """
        from .tilelang_build import TileLangSourceOnlyBuild

        self.prepare(request)
        geometry = request.selection.geometry
        names = tuple(str(stage.kernel) for stage in getattr(geometry, "stages", ()))
        build = TileLangSourceOnlyBuild(
            self.spec,
            attempt_base=self._attempt_base,
            staging=self._staging,
            executor=self._executor,
            limits=self._limits,
        )
        return build.build(request.candidate, expected_kernel_names=names)

    def build_metal(self, request: Any) -> Any:
        """@brief Metal build 的显式 typed 拒绝（MT2 才拥有）。

        @param request ``CandidateBuildRequest``。
        @raises TileLangBuildModeUnavailable 总是。
        """
        self.prepare(request)
        raise contract.TileLangBuildModeUnavailable(
            f"track {self.spec.track_id}/{self.spec.track_version} is interface_only: this epoch implements only "
            f"{list(contract.IMPLEMENTED_BUILD_MODES)} and has no Metal build.  A real device build, dispatch, "
            "correctness and timing belong to MT2; refusing is the honest outcome"
        )


def expected_build_identity(
    candidate: Any,
    *,
    environment: Mapping[str, Any],
    track_version: int = contract.TILELANG_TRACK_VERSION,
) -> contract.SourceOnlyBuildIdentity:
    """@brief 独立重建 source-only 构建 expected 主语（不读任何自报报告）。

    The candidate bytes come from the projected candidate, never from disk, and every entry
    (including the execution geometry) enters the digest, so a tampered report cannot talk the
    expected identity into the build it wants to prove.  ``full_build`` is False: this is the
    subject a build must match, not a build result.

    @param candidate ``CandidateArtifact``。
    @param environment 设备无关环境事实（由 staged 解释器真实观测）。
    @return 期望主语。
    @raises TaskContractError 候选没有唯一的 ``kernel.py`` entry 或环境不合法。
    """
    sources = [entry for entry in candidate.entries if entry.role == contract.SOURCE_ROLE]
    if len(sources) != 1:
        raise contract.TileLangContractError(
            f"the projected candidate carries {len(sources)} dsl_source entries; expected identity needs exactly one"
        )
    return contract.expected_build_subject(
        candidate=candidate, environment=environment, track_version=track_version
    )


#: The codegen gate lives in the hardware-neutral contract module so a shared test can use it
#: without importing a backend; the worker-facing name stays here for discoverability.
require_codegen_success = contract.require_codegen_success


def source_only_replay(
    *,
    result: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
    candidate_source: bytes,
) -> dict[str, Any]:
    """@brief 归档 source-only 产物的 CPU 重放（委托冻结合同，不执行前端）。

    @param result 归档结果文档。
    @param artifacts 归档工件字节。
    @param candidate_source 归档候选字节。
    @return 重建的身份文档。
    """
    return contract.replay_source_only(result=result, artifacts=artifacts, candidate_source=candidate_source)


def _unavailable(service: str, spec: TrackSpec) -> Any:
    """@brief 本 epoch 不提供的服务（typed 拒绝，不返回 MSL 适配器）。

    @param service 服务名。
    @param spec 已登记的 TileLang TrackSpec。
    @return 工厂。
    """

    def factory() -> Any:
        if spec.status == "interface_only":
            raise UnsupportedContractCapability(
                f"track {spec.track_id}/{spec.track_version} has no {service!r} in this epoch: it is interface_only and "
                "neither Metal execution, fresh Metal confirmation nor publication is available"
            )
        raise UnsupportedContractCapability(
            f"track {spec.track_id}/{spec.track_version} has no registered {service!r} implementation; "
            "an unimplemented service never falls back to an MSL adapter or publication path"
        )

    return factory


def adapter_services(spec: TrackSpec) -> dict[str, Any]:
    """@brief 该 Track 的延迟服务表（供 ``track_factory`` 构造 LazyTrackAdapter）。

    @param spec 已登记的 TileLang TrackSpec。
    @return 服务名 → 工厂。
    """
    if spec.track_version == fresh_contract.TILELANG_CONFIRMATION_TRACK_VERSION:
        fresh_contract.audit_fresh_registration(spec)
    else:
        contract.audit_registration(spec)

    def renderer() -> Any:
        from ...track.plane import TrackRenderer

        return TrackRenderer.for_track(spec)

    def builder() -> Any:
        # Production construction: no frontend import or probe happens here; the real staged
        # install is observed and hashed on the first build.  /2 returns the same callable shape
        # as the MSL builder service so v2_worker can construct it with target_id.
        if spec.track_version == contract.TILELANG_METAL_TRACK_VERSION:
            def construct(*, target_id: str = "apple-m4-max") -> Any:
                from .tilelang_metal import TileLangMetalBuilder

                return TileLangMetalBuilder(spec, target_id=target_id)

            return construct
        if spec.track_version == fresh_contract.TILELANG_CONFIRMATION_TRACK_VERSION:
            def construct(*, target_id: str = "apple-m4-max") -> Any:
                from .tilelang_metal import TileLangFreshMetalBuilder

                return TileLangFreshMetalBuilder(spec, target_id=target_id)

            return construct
        return TileLangSourceOnlyBuilder(spec)

    def evaluator() -> Any:
        if spec.track_version not in (
            contract.TILELANG_METAL_TRACK_VERSION,
            fresh_contract.TILELANG_CONFIRMATION_TRACK_VERSION,
        ):
            return _unavailable("evaluator", spec)()
        from .tilelang_runtime import TileLangTrackEvaluator

        return TileLangTrackEvaluator

    def confirmation() -> Any:
        if spec.track_version == fresh_contract.TILELANG_CONFIRMATION_TRACK_VERSION and spec.status == "available":
            from .track_factory import _confirmation_identity

            return _confirmation_identity(spec)
        return _unavailable("confirmation", spec)()

    def guidance() -> Any:
        if spec.track_version == fresh_contract.TILELANG_CONFIRMATION_TRACK_VERSION and spec.status == "available":
            from .search_candidate import guidance_fingerprint

            return guidance_fingerprint
        return _unavailable("guidance", spec)()

    return {
        "validator": lambda: TileLangTrackValidator(spec),
        # The statement renderer is the shared ABI/task rendering; a submission *prompt* style is a
        # separate, absent service so no MSL bundle text can reach an unregistered DSL Track.
        "renderer": renderer,
        "builder": builder,
        # A real kernel_pipeline geometry document is the candidate's own ExecutionGeometry.  This is
        # not msl-bundle's HostTilingPlan and does not add a lifecycle.
        "geometry_resolver": lambda: candidate_geometry,
        "evaluator": evaluator,
        "confirmation": confirmation,
        "guidance": guidance,
    }


__all__ = [
    "TileLangSourceOnlyBuilder",
    "TileLangTrackValidator",
    "adapter_services",
    "expected_build_identity",
    "require_codegen_success",
    "source_only_replay",
]
