"""P1: the shape-portfolio vertical slice — resolve → CPU preflight → project → evaluate.

This module is the *composition* seam of the portfolio plan, and it is deliberately thin.  It
adds no evaluator, no second geometry resolver and no per-op tiling; it sequences the objects P0
froze:

1. **parse the request** into a :class:`~tcenv.portfolio.dispatch.DispatchRequest` bound to the
   authoritative :class:`~tcenv.opspec.v2.SemanticInstanceV2` of its finite domain point;
2. **exactly-one dispatch** through :class:`~tcenv.portfolio.dispatch.ExactDispatchResolver`,
   with a runner-owned preflight hook that calls ``resolve_geometry(variant_geometry,
   instance=..., contract=abi, limits=limits)`` — pure CPU, no build, no device;
3. when the derived reason is ``resource_unavailable`` **and** the domain fallback declared that
   reason, dispatch switches to the projected fallback; otherwise the variant fails *before*
   ``evaluate_task`` could raise a resource error from the default ``select_geometry``;
   (``msl-single``'s launch geometry is resolved with a factory bound to the **requested**
   instance's contract, so a caller can never hand in a resolver built for another size;)
4. **project only the finally selected provider** into an ordinary ``CandidateArtifact`` (the
   parent portfolio never becomes a Track candidate), and re-derive the geometry the evaluator
   will actually use — proving it is the same geometry dispatch preflighted;
5. call the one ``evaluate_task(..., fallback_geometry=None)`` seam with the ordinary candidate,
   so the build/execute path, the schema-4 report, the artifact/geometry identity and the
   lifecycle checks are exactly the existing ones.  The **domain fallback is an ordinary
   candidate artifact**, never a ``GeometryPair`` fallback: ``fallback_geometry`` stays ``None``
   and the runtime reports ``provider=candidate`` while the record says ``provider=fallback`` —
   both facts are kept, and neither is renamed into the other.

What this module does *not* do — and refuses to pretend it did:

* it never creates an active ``DispatchManifest``, an activation pointer or a
  ``portfolio_run.json`` sidecar (that is P2);
* it makes no performance claim: fallback execution is correctness evidence for the request, not
  the generated variant's correctness or speedup;
* it never lets a provider's declared geometry drift from the projected child: the projected
  ``host/execution_geometry.json`` bytes are hashed against the parent declaration, and the
  evaluated ``geometry_identity`` must equal the runner's own CPU computation;
* it never builds a provider that did not win the dispatch, and it never falls back without a
  declared reason.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..contract_errors import GeometryError, TaskContractError
from ..identity import fingerprint, is_hex64
from ..opspec.model import OpSpecError
from ..opspec.v2 import SemanticInstanceV2, instantiate_v2
from ..portfolio.candidate import (
    BUNDLE_TRACK_ID,
    CandidateProjection,
    DomainFallback,
    PortfolioCandidate,
    PortfolioVariant,
    project_provider,
)
from ..portfolio.dispatch import (
    DispatchRequest,
    ExactDispatchResolver,
    GeometryPreflight,
    PortfolioDispatchRecord,
    preflight_geometry,
)
from ..portfolio.domain import DomainAuthority, SemanticDomainPlan
from ..target import DEFAULT_TARGET_ID, TargetProfile, load_target
from ..track import CandidateLifecycle, HostStep, PrepareDeclaration
from ..track.abi import abi_limits_from_target, derive_tensor_memory_contract
from ..track.bundle import CanonicalBundle
from ..track.delivery import candidate_geometry
from ..track.geometry import ResolvedGeometry, geometry_pair, resolve_geometry
from ..track.lifecycle import ENTRY_BOUND_ROLES
from ..track.plane import ExecutionGeometry
from .loader import TaskContract, load_task_contract
from .locator import TaskLocator
from .pack import PackClosureError, task_pack_closure
from .runner import (
    BUILD_FREE_STATUSES,
    CandidateBuilder,
    CandidateEvaluator,
    GeometryResolver,
    TaskEvaluationResult,
    evaluate_task,
)

PORTFOLIO_RUN_SCHEMA_ID = "tcenv.portfolio.run.1"
PORTFOLIO_RUN_VERSION = 1
#: Outcomes that must produce a built/executed provider.  Everything else must produce none.
EVALUATED_OUTCOMES = frozenset({"resolved", "fallback"})
#: The runtime's ``provider`` is the *GeometryPair* source the executor ran.  A projected domain
#: fallback is delivered as an ordinary candidate, so it must be ``candidate``; a ``fallback``
#: value here would mean the harness used the §A2 geometry fallback instead of a domain fallback.
RUNTIME_GEOMETRY_PROVIDER = "candidate"
#: 没有 candidate dispatch 记录的报告状态：worker 自己给出 provider（编译失败/超时/崩溃/拒绝），
#: 这些是候选的正常判决，不是「执行了几何 fallback」。
NO_DISPATCH_STATUSES = frozenset(BUILD_FREE_STATUSES | {"compile_error"})
#: Diagnostic prefix shared by the runner's preflight and any replay of it (it is part of the
#: dispatch record's ``detail``, so an independent replay must use the same spelling).
PREFLIGHT_WHERE = "portfolio dispatch preflight"


def _require_digest(value: Any, *, where: str) -> str:
    """@brief 校验一个 64 位小写十六进制摘要。

    @param value 值。
    @param where 诊断位置。
    @return 原值。
    @raises TaskContractError 非法。
    """
    if not is_hex64(value):
        raise TaskContractError(f"{where} must be a 64-character lowercase hex digest, got {value!r}")
    return str(value)


def _sha256(data: bytes) -> str:
    """@brief 字节 sha256。

    @param data 字节。
    @return 十六进制摘要。
    """
    return hashlib.sha256(bytes(data)).hexdigest()


def instances_from_plan(plan: SemanticDomainPlan, spec: Any) -> tuple[SemanticInstanceV2, ...]:
    """@brief 由有限域计划声明的点派生权威 ``SemanticInstanceV2``（每个 dtype/size 一个）。

    The domain plan names ``(dtype, size, usage)`` points; the authoritative instance is the one
    the OpSpec derives, not a caller-supplied object.  :class:`DomainAuthority` re-normalizes the
    plan from these instances and refuses any mismatch, so a forged plan/instance pair cannot
    reach dispatch through this path.

    @param plan 有限语义域计划。
    @param spec 权威 OpSpec。
    @return 语义实例元组（按 ``(dtype, size)`` 排序）。
    @raises TaskContractError 实例无法由该 OpSpec 派生。
    """
    pairs = sorted({(point.dtype, point.size) for point in plan.instances})
    instances: list[SemanticInstanceV2] = []
    for dtype, size in pairs:
        try:
            instances.append(instantiate_v2(spec, dtype, size))
        except OpSpecError as exc:
            raise TaskContractError(
                f"the domain plan names {dtype}/{size}, which the authoritative OpSpec cannot derive: {exc}"
            ) from exc
    if not instances:
        raise TaskContractError("a portfolio run needs at least one authoritative instance")
    return tuple(instances)


def locator_for_instance(locator: TaskLocator, dtype: str, size: str) -> TaskLocator:
    """@brief 为请求的 ``(dtype, size)`` 派生同一任务目录的 locator。

    ``evaluate_task`` reads the locator's own instance, so a portfolio request for another named
    size needs the same task pack addressed at that instance.  This uses the existing A1 reader
    (``load_task_contract``) and the existing pack-closure identity; it does not create a task,
    copy a pack or invent a fingerprint.  ``TaskLocator.load`` inside ``evaluate_task`` re-derives
    both digests and refuses any drift.

    @param locator 默认实例的 locator（提供任务目录与 code root）。
    @param dtype 请求的 dtype。
    @param size 请求的 size。
    @return 该实例的 locator。
    @raises TaskContractError 同名/漂移/无法装配。
    """
    if not isinstance(locator, TaskLocator):
        raise TaskContractError("locator_for_instance needs a TaskLocator")
    contract = load_task_contract(locator.task_dir, dtype=dtype, size=size)
    if contract.op_name != locator.op_name:
        raise TaskContractError(
            f"{locator.task_id}: the requested instance belongs to op {contract.op_name!r}, not {locator.op_name!r}"
        )
    try:
        closure = task_pack_closure(locator.task_dir, root=locator.code_root)
    except PackClosureError as exc:
        raise TaskContractError(f"{locator.task_id}: the task pack closure failed: {exc}") from exc
    try:
        return TaskLocator(
            task_id=locator.task_id,
            op_name=contract.op_name,
            dtype=contract.dtype,
            size=contract.size,
            pack_digest=contract.pack_digest,
            deployment_pack_digest=closure.digest,
            task_dir=locator.task_dir,
            code_root=locator.code_root,
        )
    except ValueError as exc:  # TaskDiscoveryError derives from ValueError
        raise TaskContractError(f"{locator.task_id}: cannot address the requested instance: {exc}") from exc


def _stage_signature(stage: Any) -> dict[str, Any]:
    """@brief 一个已求值 stage 的 dispatch 签名（进入证据，可逐字段比较）。

    @param stage ``ResolvedStage``。
    @return 字段字典。
    """
    kernel, grid, threads, threadgroup_bytes, workspace = stage.dispatch_signature()
    return {
        "kernel": str(kernel),
        "grid": [int(value) for value in grid],
        "threads": [int(value) for value in threads],
        "threadgroup_bytes": int(threadgroup_bytes),
        "workspace": [str(name) for name in workspace],
    }


def _observed_dispatch(records: Iterable[Mapping[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    """@brief runtime 实际记录的 dispatch（按协议身份去重，保留每个 stage）。

    The runtime already collapses repeated trials by ``(stage, kernel, grid, threads,
    threadgroup_bytes, workspace, provider)``, so the stage index is part of its identity.  This
    function must not add a second, coarser identity: two legitimate active stages may share the
    same five-tuple (for example two passes over the same buffers with the same launch shape), and
    folding them together would reject an executable candidate.  Deduplication therefore keys on
    ``(stage, signature)`` and the stage index travels with the signature.

    @param records ``result.execution["dispatch"]`` 条目。
    @return ``(stage, signature)`` 列表（按记录顺序）。
    """
    seen: set[tuple] = set()
    observed: list[tuple[int, dict[str, Any]]] = []
    for item in records:
        signature = {
            "kernel": str(item["kernel"]),
            "grid": [int(value) for value in item["grid"]],
            "threads": [int(value) for value in item["threads"]],
            "threadgroup_bytes": int(item["threadgroup_bytes"]),
            "workspace": [str(name) for name in item["workspace"]],
        }
        stage = int(item["stage"])
        key = (
            stage,
            signature["kernel"],
            tuple(signature["grid"]),
            tuple(signature["threads"]),
            signature["threadgroup_bytes"],
            tuple(signature["workspace"]),
        )
        if key in seen:
            continue
        seen.add(key)
        observed.append((stage, signature))
    return observed


def _dispatch_mismatches(
    expected: ResolvedGeometry, observed: list[tuple[int, dict[str, Any]]]
) -> list[str]:
    """@brief 比较 runner 的已求值几何与 runtime 实际派遣（按 stage 索引，不折叠同签名阶段）。

    @param expected 期望几何。
    @param observed ``(stage, 签名)`` 列表。
    @return 不匹配说明（空表示一致）。
    """
    wanted = [(int(stage.index), _stage_signature(stage)) for stage in expected.active_stages]
    if [index for index, _ in wanted] != [index for index, _ in observed]:
        return [
            f"the runtime dispatched stages {[index for index, _ in observed]}, the frozen geometry has active "
            f"stages {[index for index, _ in wanted]}"
        ]
    problems: list[str] = []
    for (index, want), (_, have) in zip(wanted, observed):
        if want != have:
            problems.append(f"stage {index}: runtime dispatched {have}, the frozen geometry says {want}")
    return problems


@dataclass(frozen=True)
class PortfolioRunResult:
    """@brief 一次 portfolio 运行的组合证据：dispatch 记录 + 投影 + 评测身份。

    The object is a *decision + execution* record.  It is not an active manifest, and the
    ``performance_claim``/``active_manifest`` fields document that explicitly instead of leaving
    the absence to prose.
    """

    record: PortfolioDispatchRecord
    projection: CandidateProjection | None = None
    result: TaskEvaluationResult | None = None
    evaluation: Mapping[str, Any] | None = None
    schema_id: str = PORTFOLIO_RUN_SCHEMA_ID
    run_version: int = PORTFOLIO_RUN_VERSION

    def __post_init__(self) -> None:
        """@brief 校验记录、投影与评测三者互相一致（身份串用/伪造 provider 在此拒绝）。

        @raises TaskContractError 任一身分不一致。
        """
        if self.schema_id != PORTFOLIO_RUN_SCHEMA_ID:
            raise TaskContractError(f"unsupported portfolio run schema {self.schema_id!r}")
        if self.run_version != PORTFOLIO_RUN_VERSION:
            raise TaskContractError(f"unsupported portfolio run version {self.run_version!r}")
        if not isinstance(self.record, PortfolioDispatchRecord):
            raise TaskContractError("a portfolio run needs a PortfolioDispatchRecord")
        evaluated = self.record.outcome in EVALUATED_OUTCOMES
        if evaluated:
            if self.projection is None or self.result is None or self.evaluation is None:
                raise TaskContractError(
                    f"a {self.record.outcome!r} record must carry the projection and evaluation it produced"
                )
            self._check_identity()
        elif self.projection is not None or self.result is not None or self.evaluation is not None:
            raise TaskContractError(
                f"outcome {self.record.outcome!r} selected no provider; a projection or evaluation would be a run "
                "that dispatch refused"
            )

    def _check_identity(self) -> None:
        """@brief 校验投影/评测与 dispatch 记录指向同一个 provider 和同一份几何。

        @raises TaskContractError 不一致。
        """
        assert self.projection is not None and self.result is not None and self.evaluation is not None
        record, projection, result, evaluation = self.record, self.projection, self.result, self.evaluation
        if projection.provider != record.provider:
            raise TaskContractError(
                f"the record says provider {record.provider!r} but the projection is {projection.provider!r}; a "
                "provider may not be re-labelled after dispatch"
            )
        if record.provider == "candidate":
            if projection.identity != record.variant_id:
                raise TaskContractError("the projected candidate is not the variant the record selected")
            if projection.binding.variant_id != record.variant_id or projection.binding.fallback_id is not None:
                raise TaskContractError("the projection binding and the dispatch record disagree on the variant")
        else:
            if projection.identity != record.fallback_id:
                raise TaskContractError("the projected fallback is not the domain fallback the record selected")
            if projection.binding.fallback_id != record.fallback_id or projection.binding.variant_id is not None:
                raise TaskContractError("the projection binding and the dispatch record disagree on the fallback")
        if projection.binding.portfolio_digest != record.portfolio_digest:
            raise TaskContractError("the projection belongs to a different portfolio than the dispatch record")
        if projection.binding.domain_plan_digest != record.domain_plan_digest:
            raise TaskContractError("the projection belongs to a different semantic domain plan")
        if projection.binding.candidate_sha256 != projection.candidate.candidate_sha256:
            raise TaskContractError("the projection binding does not name the projected candidate identity")
        if projection.binding.parent_portfolio_sha256 == projection.candidate.candidate_sha256:
            raise TaskContractError("the projected candidate reuses the parent portfolio identity")
        if result.candidate_sha256 != projection.candidate.candidate_sha256:
            raise TaskContractError(
                "the evaluation was built for a different candidate than the projected provider (stale artifact)"
            )
        if result.track["track_id"] != projection.track_id:
            raise TaskContractError("the evaluation used a different Track than the projection")
        if evaluation.get("compiled_candidate_sha256") != projection.candidate.candidate_sha256:
            raise TaskContractError("the evaluation record does not name the compiled projected candidate")
        if evaluation.get("provider") != record.provider:
            raise TaskContractError("the evaluation record renames the dispatch provider")
        if evaluation.get("execution_published") is True:
            if evaluation.get("runtime_provider") != RUNTIME_GEOMETRY_PROVIDER:
                raise TaskContractError(
                    "the runtime executed the §A2 geometry fallback; a domain fallback is an ordinary projected "
                    "candidate and must never be delivered as a GeometryPair fallback"
                )
            _require_digest(evaluation.get("artifact_set_digest"), where="evaluation.artifact_set_digest")
            if evaluation.get("artifact_set_digest") != result.artifact_set_digest:
                raise TaskContractError("the evaluation artifact set is not the built artifact set")
        elif evaluation.get("execution_published") is False:
            if result.report.get("status") not in NO_DISPATCH_STATUSES:
                raise TaskContractError(
                    "a normal evaluation must publish what it executed; only a build-free status may omit it"
                )
            if evaluation.get("runtime_provider") is not None:
                raise TaskContractError("a build-free evaluation cannot publish a runtime geometry provider")
        else:
            raise TaskContractError("the evaluation record does not say whether a runtime record was published")
        if evaluation.get("fallback_geometry") is not None:
            raise TaskContractError("a portfolio run passes fallback_geometry=None; the §A2 fallback is not a stand-in")
        if evaluation.get("execution_published") and evaluation.get("dispatch_stages") != evaluation.get(
            "expected_dispatch_stages"
        ):
            raise TaskContractError(
                "the dispatched stage indices are not the frozen geometry's active stages "
                f"({evaluation.get('dispatch_stages')} != {evaluation.get('expected_dispatch_stages')})"
            )
        if evaluation.get("geometry_identity") != result.geometry_identity:
            raise TaskContractError("the evaluation geometry identity drifted from the evaluated geometry")
        if evaluation.get("expected_geometry_identity") != result.geometry_identity:
            raise TaskContractError(
                "the evaluated geometry is not the geometry the runner resolved on the CPU before the build"
            )
        _require_digest(evaluation.get("evaluation_identity"), where="evaluation.evaluation_identity")
        if evaluation.get("evaluation_identity") != result.identity:
            raise TaskContractError("the evaluation identity is not the returned TaskEvaluationResult identity")
        if evaluation.get("bundle_digest") != result.bundle_digest:
            raise TaskContractError("the evaluation bundle digest drifted")
        if evaluation.get("geometry_origin_path") != projection.binding.geometry_origin_path:
            raise TaskContractError("the evaluation record does not carry the parent geometry path")
        if evaluation.get("geometry_origin_sha256") != projection.binding.geometry_origin_sha256:
            raise TaskContractError("the evaluation record does not carry the parent geometry content SHA-256")
        if evaluation.get("report_status") != result.report.get("status"):
            raise TaskContractError("the evaluation report status drifted from the report")

    @property
    def identity(self) -> str:
        """@brief 本次运行的组合身份。

        @return 十六进制 sha256。
        """
        return fingerprint(self.document())

    def document(self) -> dict[str, Any]:
        """@brief 组合证据的 JSON 文档。

        @return 字段字典。
        """
        projection = None if self.projection is None else self.projection.document()
        evaluation = None if self.evaluation is None else dict(self.evaluation)
        identities: dict[str, Any] = {
            "domain_plan_digest": self.record.domain_plan_digest,
            "portfolio_digest": self.record.portfolio_digest,
            "variant_id": self.record.variant_id,
            "fallback_id": self.record.fallback_id,
            "provider": self.record.provider,
            "fallback_reason": self.record.fallback_reason,
            "dispatch_record_digest": self.record.identity,
            "projection_binding_digest": None if self.projection is None else self.projection.binding.identity,
            "candidate_sha256": None if self.projection is None else self.projection.candidate.candidate_sha256,
            "parent_portfolio_sha256": None if self.projection is None else self.projection.binding.parent_portfolio_sha256,
            "geometry_origin_path": None if self.projection is None else self.projection.binding.geometry_origin_path,
            "geometry_origin_sha256": None if self.projection is None else self.projection.binding.geometry_origin_sha256,
            "geometry_identity": None if self.result is None else self.result.geometry_identity,
            "bundle_digest": None if self.result is None else self.result.bundle_digest,
            "artifact_set_digest": None if self.result is None else self.result.artifact_set_digest,
            "native_digest": None if self.result is None else self.result.native_digest,
            "evaluation_identity": None if self.result is None else self.result.identity,
        }
        return {
            "schema_id": self.schema_id,
            "run_version": self.run_version,
            "record": self.record.document(),
            "projection": projection,
            "evaluation": evaluation,
            "identities": identities,
            # Explicit absences, so a reader cannot mistake this object for activation or a claim.
            "active_manifest": None,
            "activation_generation": None,
            "performance_claim": None,
            "fallback_credits_variant": False,
        }


def _instances_of(plan: SemanticDomainPlan, contract: TaskContract) -> tuple[SemanticInstanceV2, ...]:
    """@brief 派生权威实例（域计划的 dtype/size 点 × 该 OpSpec）。

    @param plan 有限语义域计划。
    @param contract 该任务的合同（提供权威 OpSpec）。
    @return 实例元组。
    """
    return instances_from_plan(plan, contract.spec)


def _bound_point(
    plan: SemanticDomainPlan,
    instances: Iterable[SemanticInstanceV2],
    request: DispatchRequest | Mapping[str, Any],
) -> tuple[DispatchRequest | None, SemanticInstanceV2 | None]:
    """@brief 解析请求并找出它的权威实例（不可解析/不在域内时返回 ``None``）。

    The preflight hook is per request, so the runner has to know which instance it belongs to
    before dispatch runs.  A request that cannot be parsed, or that names no point of the finite
    domain, gets no hook: dispatch will refuse it and never reaches a provider preflight.

    @param plan 有限语义域计划。
    @param instances 权威实例。
    @param request 请求或请求文档。
    @return ``(parsed request or None, instance or None)``。
    """
    parsed: DispatchRequest | None
    if isinstance(request, DispatchRequest):
        parsed = request
    else:
        try:
            parsed = DispatchRequest.parse(request)
        except TaskContractError:
            return None, None
    for instance in instances:
        if (instance.dtype, instance.size) == (parsed.dtype, parsed.size):
            return parsed, instance
    return parsed, None


def _preflight_hook(
    instance: SemanticInstanceV2,
    *,
    target: TargetProfile,
    max_buffer_length: int | None,
    portfolio: PortfolioCandidate,
    authority: DomainAuthority,
    resolver: GeometryResolver | None,
) -> Callable[..., GeometryPreflight]:
    """@brief 该请求的纯 CPU 几何预检钩子（``resolve_geometry`` + 资源分类）。

    A provider whose geometry the portfolio freezes is preflighted with that geometry.  A provider
    that owns its launch in the source (``msl-single``) has no frozen geometry, so the hook derives
    **that provider's** projected launch geometry for the requested instance and preflights it
    too — the resource decision therefore happens on the CPU before any build, for both Tracks.

    @param instance 请求的权威实例。
    @param target 目标档案（与 ``evaluate_task`` 同一份）。
    @param max_buffer_length 设备缓冲长度（与 ``evaluate_task`` 同一值）。
    @param portfolio 已解析 portfolio（用于投影被预检的 provider）。
    @param authority 投影权威（域/规范/目标）。
    @param resolver 请求实例绑定的 ``msl-single`` launch 解析器。
    @return 预检函数。
    """
    abi = derive_tensor_memory_contract(instance, target=target, max_buffer_length=max_buffer_length)
    limits = abi_limits_from_target(target, max_buffer_length=max_buffer_length)
    providers: dict[str, PortfolioVariant | DomainFallback] = {
        variant.variant_key: variant for variant in portfolio.variants
    }
    if portfolio.fallback is not None:
        providers[portfolio.fallback.fallback_id] = portfolio.fallback

    def hook(geometry: ExecutionGeometry | None, *, provider: str | None = None) -> GeometryPreflight:
        """@brief 预检一个 provider 的几何（声明几何或源内 launch 几何）。

        @param geometry 声明几何（``None`` 表示该 provider 的几何在源里）。
        @param provider provider 的稳定键。
        @return 预检结果。
        """
        chosen = geometry
        if chosen is None:
            target_provider = providers.get(str(provider))
            if target_provider is None:
                raise TaskContractError(
                    f"the preflight hook was asked about {provider!r}, which is not a provider of this portfolio"
                )
            projection = project_provider(portfolio, target_provider, authority=authority)
            chosen = _projected_geometry(projection, resolver)
        return preflight_geometry(
            chosen, instance=instance, contract=abi, limits=limits, where=PREFLIGHT_WHERE
        )

    return hook


def _selected_provider(
    portfolio: PortfolioCandidate, record: PortfolioDispatchRecord
) -> PortfolioVariant | DomainFallback:
    """@brief 由记录取最终选中的 provider 对象。

    @param portfolio portfolio。
    @param record dispatch 记录。
    @return variant 或 domain fallback。
    @raises TaskContractError 记录指向不存在的 provider。
    """
    if record.outcome == "resolved":
        variant = portfolio.variant_for_provider(record.variant_id or "")
        if variant is None:
            raise TaskContractError(
                "the dispatch record selected a variant this portfolio does not contain; a provider may not be "
                "borrowed from another portfolio"
            )
        return variant
    fallback = portfolio.fallback
    if fallback is None or fallback.fallback_id != record.fallback_id:
        raise TaskContractError("the dispatch record selected a domain fallback this portfolio does not contain")
    if fallback.kind == "refuse":
        raise TaskContractError("a refusing fallback has no artifact to project or execute")
    return fallback


def _projected_geometry(
    projection: CandidateProjection, resolver: GeometryResolver | None
) -> ExecutionGeometry:
    """@brief 重新导出 evaluator 实际使用的几何（bundle 读投影 entry，single 用注入的 launch 解析器）。

    @param projection 已投影 provider。
    @param resolver msl-single 的几何解析器（bundle 不需要）。
    @return 执行几何。
    @raises TaskContractError 无法导出或来源不是候选。
    """
    if projection.track_id == BUNDLE_TRACK_ID:
        geometry = candidate_geometry(projection.candidate)
    else:
        if resolver is None:
            raise TaskContractError(
                f"{projection.label}: an msl-single candidate needs the backend launch resolver before the build; "
                "the harness does not invent a grid/threads"
            )
        geometry = resolver(projection.candidate)
    if not isinstance(geometry, ExecutionGeometry):
        raise TaskContractError(
            f"{projection.label}: the geometry resolver returned {type(geometry).__name__}, not an ExecutionGeometry"
        )
    if geometry.source != RUNTIME_GEOMETRY_PROVIDER:
        raise TaskContractError(
            f"{projection.label}: the execution geometry declares source {geometry.source!r}; a domain fallback is "
            "an ordinary projected candidate and is never a GeometryPair fallback"
        )
    return geometry


def _host_work_roles(candidate: Any) -> tuple[str, ...]:
    """@brief 投影候选里必须由 prepare 步骤执行的 host 角色。

    @param candidate 投影后的 ``CandidateArtifact``。
    @return 角色元组。
    """
    entry_roles = {entry.role for entry in candidate.entries}
    return tuple(role for role in ENTRY_BOUND_ROLES if role in entry_roles)


def _lifecycle_factory(
    *,
    projection: CandidateProjection,
    resolved: ResolvedGeometry,
    contract: TaskContract,
    target_id: str,
) -> Callable[[ResolvedGeometry], CandidateLifecycle] | None:
    """@brief 为投影的 host 工作构造 ``CandidateLifecycle`` 工厂。

    The candidate ships its own Host Tiling; this registers that the declared host work is
    *executed* rather than merely delivered (the existing A2/A4 rule).  Timing attribution is
    derived, not chosen: a Host Tiling step that writes a workspace the candidate's geometry
    declares ``zero`` must run inside the measured window, because the runtime would otherwise
    clear the workspace after the CPU write.  When no workspace is declared zero the step stays
    outside timing, which is the existing input-independent, uncached preparation boundary.

    @param projection 投影 provider。
    @param resolved 已求值几何。
    @param contract 任务合同（``instance_key`` 身份）。
    @param target_id 目标。
    @return 工厂，或 ``None``（没有 host 工作）。
    """
    roles = _host_work_roles(projection.candidate)
    if not roles:
        return None
    if projection.track_id != BUNDLE_TRACK_ID:
        raise TaskContractError(
            f"{projection.label}: a {projection.track_id} projection carries host roles {roles}; host work belongs to "
            "a bundle projection"
        )
    zeroed = {use.name for use in resolved.workspace_uses if use.zero}
    inside_timing = bool(zeroed)
    candidate_sha256 = projection.candidate.candidate_sha256
    track_id = projection.track_id

    def factory(chosen: ResolvedGeometry) -> CandidateLifecycle:
        steps = tuple(
            HostStep(
                PrepareDeclaration(
                    name=role,
                    input_dependent=False,
                    inside_timing=inside_timing if role == "host_tiling" else False,
                ),
                role=role,
            )
            for role in roles
        )
        return CandidateLifecycle(
            track_id=track_id,
            geometry=chosen,
            host_steps=steps,
            candidate_sha256=candidate_sha256,
            target_id=target_id,
            instance_key=contract.fingerprint(),
        )

    return factory


def _verify_projected_geometry_bytes(projection: CandidateProjection) -> str | None:
    """@brief 投影几何 entry 的字节必须等于父包声明的几何内容。

    @param projection 投影 provider。
    @return 投影几何 entry 的 sha256（无声明几何时为 None）。
    @raises TaskContractError 路径缺失或内容漂移。
    """
    binding = projection.binding
    if binding.geometry_path is None:
        return None
    entry = next((item for item in projection.candidate.entries if item.path == binding.geometry_path), None)
    if entry is None:
        raise TaskContractError(
            f"{projection.label}: the projection does not contain the declared geometry at {binding.geometry_path!r}"
        )
    digest = _sha256(entry.data)
    if digest != binding.geometry_origin_sha256:
        raise TaskContractError(
            f"{projection.label}: the projected geometry bytes are not the parent's declared geometry "
            f"({digest} != {binding.geometry_origin_sha256}); a projection may not rewrite the geometry"
        )
    return digest


def _native_digest(report: Mapping[str, Any]) -> str | None:
    """@brief 由 schema-4 报告的 native 块重算原生可执行文件摘要。

    ``NativeExecutable.digest`` 是 ``SHA256(RFC8785(document()))``，而报告里的 native 块正是同一份
    handle-free 文档。重算它，是为了让"报告说它编译了什么"和"评测结果的身份"互相绑定，而不是各说各话。

    @param report schema-4 报告。
    @return 原生摘要，或 ``None``（没有 native 块）。
    @raises TaskContractError schema id 或块结构不对。
    """
    build = report.get("build")
    if not isinstance(build, Mapping):
        return None
    native = build.get("native")
    if native is None:
        return None
    if not isinstance(native, Mapping):
        raise TaskContractError("the report's native block is not an object")
    from ..remote.artifact_bundle import NATIVE_EXECUTABLE_SCHEMA_ID, NativeExecutable

    if native.get("schema_id") != NATIVE_EXECUTABLE_SCHEMA_ID:
        raise TaskContractError(
            f"the report's native schema is {native.get('schema_id')!r}, not {NATIVE_EXECUTABLE_SCHEMA_ID!r}"
        )
    return NativeExecutable(
        backend_id=str(native["backend_id"]),
        kind=str(native["kind"]),
        entry_path=str(native["entry_path"]),
        entry_sha256=str(native["entry_sha256"]),
        descriptor=dict(native["descriptor"]),
    ).digest


def _check_native_digest(result: TaskEvaluationResult, *, where: str) -> None:
    """@brief 报告 native 块、报告的 ``native_digest`` 与结果身份必须是同一份原生可执行文件。

    @param result 评测结果。
    @param where 诊断位置。
    @raises TaskContractError 三者不一致。
    """
    build = result.report.get("build")
    recorded = build.get("native_digest") if isinstance(build, Mapping) else None
    recomputed = _native_digest(result.report)
    if recomputed is None:
        # No published native executable (build-free report, or an evaluator that received a
        # prebuilt artifact set): the report itself must not claim one.
        if recorded is not None:
            raise TaskContractError(
                f"{where}: the report claims native_digest {recorded!r} but publishes no native block"
            )
        return
    if recorded != recomputed:
        raise TaskContractError(
            f"{where}: the report's native_digest is not SHA256(descriptor) of its own native block "
            f"({recorded} != {recomputed})"
        )
    if result.native_digest != recomputed:
        raise TaskContractError(
            f"{where}: the evaluation's native digest is not the native executable the report describes "
            f"({result.native_digest} != {recomputed})"
        )


def run_portfolio_task(
    *,
    locator: TaskLocator,
    domain_plan: SemanticDomainPlan,
    portfolio: PortfolioCandidate,
    request: DispatchRequest | Mapping[str, Any],
    evidence: Mapping[str, str],
    builder: CandidateBuilder,
    evaluator: CandidateEvaluator,
    geometry_resolver_factory: Callable[[TaskContract], GeometryResolver] | None = None,
    target_id: str = DEFAULT_TARGET_ID,
    max_buffer_length: int | None = None,
    track_registry_path: str | Path | None = None,
    track_version: int | None = None,
) -> PortfolioRunResult:
    """@brief 解析 → exactly-one dispatch → CPU 几何预检 → 投影 → 唯一 ``evaluate_task``。

    @param locator 任务 locator（默认实例；运行时会按请求的 ``(dtype, size)`` 重新定位）。
    @param domain_plan 该 portfolio 绑定的有限语义域计划。
    @param portfolio 已解析的外层 portfolio envelope。
    @param request 请求对象或它的闭集文档。
    @param evidence provider 身份 → 证据状态（``validated``/``unknown``）。
    @param builder 真实构建适配器（注入，A4 的 ``MetalTrackBuilder``）。
    @param evaluator 真实评测适配器（注入，A4 的 ``MetalTrackEvaluator``）。
    @param geometry_resolver_factory ``msl-single`` 的 launch 解析器工厂；它由 runner 用**请求实例**的
        合同调用（``msl_single_geometry_resolver(task_contract)``），所以调用方无法交进一个按默认实例
        构造、几何与请求不符的解析器。``msl-bundle`` 不需要它。
    @param target_id 目标档案。
    @param max_buffer_length 设备缓冲长度（预检与 ``evaluate_task`` 必须一致）。
    @param track_registry_path 可选 registry（测试用）。
    @param track_version 要求版本。
    @return 组合证据。
    @raises TaskContractError 请求/域/投影/身份/几何任一不合法。
    """
    if not isinstance(locator, TaskLocator):
        raise TaskContractError("run_portfolio_task needs a TaskLocator")
    if not isinstance(domain_plan, SemanticDomainPlan):
        raise TaskContractError("run_portfolio_task needs the SemanticDomainPlan the portfolio is bound to")
    if not isinstance(portfolio, PortfolioCandidate):
        raise TaskContractError("run_portfolio_task needs a PortfolioCandidate")
    contract = locator.load()
    if portfolio.domain_plan_digest != domain_plan.digest:
        raise TaskContractError(
            "the portfolio and the semantic domain plan do not share a domain; a run may not borrow another domain"
        )
    target = load_target(target_id)
    instances = _instances_of(domain_plan, contract)
    authority = DomainAuthority(
        plan=domain_plan,
        spec=contract.spec,
        target=target,
        instances=instances,
        contract=contract,
        max_buffer_length=max_buffer_length,
    )
    resolver = ExactDispatchResolver(
        domain_plan=domain_plan,
        portfolio=portfolio,
        evidence=evidence,
        spec=contract.spec,
        target=target,
        instances=instances,
        max_buffer_length=max_buffer_length,
    )
    parsed, instance = _bound_point(domain_plan, instances, request)
    # The requested instance's own contract owns the launch geometry and the lifecycle identity, so
    # it is derived *before* dispatch: the preflight may need the source launch geometry, and a
    # geometry must never be resolved (or a lifecycle bound) for another size.
    run_locator: TaskLocator | None = None
    run_contract: TaskContract | None = None
    launch_resolver: GeometryResolver | None = None
    hook = None
    if instance is not None and parsed is not None:
        run_locator = locator_for_instance(locator, parsed.dtype, parsed.size)
        if run_locator.document()["dtype"] != parsed.dtype or run_locator.document()["size"] != parsed.size:
            raise TaskContractError("the instance locator does not address the requested instance")
        run_contract = run_locator.load()
        if run_contract.instance.dtype != parsed.dtype or run_contract.instance.size != parsed.size:
            raise TaskContractError("the instance contract does not describe the requested instance")
        if geometry_resolver_factory is not None:
            launch_resolver = geometry_resolver_factory(run_contract)
        hook = _preflight_hook(
            instance,
            target=target,
            max_buffer_length=max_buffer_length,
            portfolio=portfolio,
            authority=authority,
            resolver=launch_resolver,
        )
    if launch_resolver is not None:
        # Opt in: a provider that owns its launch in the source is preflighted with the geometry the
        # backend derives for *this* instance, so an over-limit source launch diverts (or fails)
        # before a build instead of surfacing from inside ``evaluate_task``.
        resolver = ExactDispatchResolver(
            domain_plan=domain_plan,
            portfolio=portfolio,
            evidence=evidence,
            spec=contract.spec,
            target=target,
            instances=instances,
            max_buffer_length=max_buffer_length,
            source_preflight=True,
        )
    record = resolver.resolve(request if parsed is None else parsed, preflight=hook)
    if record.outcome not in EVALUATED_OUTCOMES:
        # Refused/unmatched/unknown/overlap/unavailable: no final projection is returned, nothing is
        # built and ``evaluate_task`` is never entered.  The preflight hook may have projected a
        # provider on the CPU to derive its geometry — that temporary projection is dropped here.
        return PortfolioRunResult(record=record)
    if instance is None or parsed is None:  # pragma: no cover - dispatch cannot resolve without them
        raise TaskContractError("an evaluated outcome needs a parsed request bound to an authoritative instance")
    if record.request is None or record.request.get("instance_key") != parsed.instance_key:  # pragma: no cover
        raise TaskContractError("the dispatch record does not describe the request that was resolved")
    if run_locator is None or run_contract is None:  # pragma: no cover - dispatch cannot resolve without them
        raise TaskContractError("an evaluated outcome needs the requested instance's contract")

    provider = _selected_provider(portfolio, record)
    if isinstance(provider, DomainFallback) and provider.kind == "refuse":  # pragma: no cover - dispatch refuses
        raise TaskContractError("a refusing fallback cannot be executed")
    projection = project_provider(portfolio, provider, authority=authority)
    if projection.provider != record.provider:  # pragma: no cover - _selected_provider guarantees it
        raise TaskContractError("the projection provider drifted from the dispatch record")
    projected_geometry_sha256 = _verify_projected_geometry_bytes(projection)

    # The evaluator will use exactly this geometry: for a bundle, the projected frozen entry; for a
    # single source, the candidate's own launch block through the injected backend resolver.  Both
    # are resolved here, on the CPU, before any build, so a resource/invalid geometry failure can
    # never surface from inside ``evaluate_task`` after it has started building.
    resolver = launch_resolver
    abi = derive_tensor_memory_contract(instance, target=target, max_buffer_length=max_buffer_length)
    limits = abi_limits_from_target(target, max_buffer_length=max_buffer_length)
    execution_geometry = _projected_geometry(projection, resolver)
    # The named CPU preflight the plan requires: resolve the *selected* geometry against the real
    # instance/ABI/limits before anything is projected further or built.  A failure is classified
    # with the P0 helper (``preflight_geometry``) only to describe it; either way no build happens.
    try:
        resolved = resolve_geometry(
            execution_geometry,
            instance=instance,
            contract=abi,
            limits=limits,
            where=f"portfolio {projection.label}",
        )
    except GeometryError as exc:
        classified = preflight_geometry(
            execution_geometry, instance=instance, contract=abi, limits=limits, where=f"portfolio {projection.label}"
        )
        raise TaskContractError(
            f"{projection.label}: the geometry of the selected {projection.track_id} artifact is not executable "
            f"({classified.status}: {exc}). A provider without a declared geometry has no dispatch-time resource "
            "preflight, so this is a candidate defect and never a fallback switch"
        ) from exc
    if projection.track_id == BUNDLE_TRACK_ID:
        if record.preflight.status != "ok" or record.preflight.resolved is None:
            raise TaskContractError(
                f"{projection.label}: dispatch did not preflight the declared geometry it selected"
            )
        if record.preflight.resolved.document() != resolved.document():
            raise TaskContractError(
                f"{projection.label}: the projected geometry is not the geometry dispatch preflighted; the projection "
                "changed the execution geometry"
            )
    elif resolver is None:
        if record.preflight.status != "not_required":
            raise TaskContractError(
                f"{projection.label}: a {projection.track_id} projection owns its launch in the source, so dispatch "
                f"must record not_required when no launch resolver was injected, not {record.preflight.status!r}"
            )
    else:
        # Source-aware preflight: dispatch must have resolved *this* instance's launch geometry and
        # recorded it, not a placeholder.  It is the same geometry the evaluator is about to build.
        if record.preflight.status != "ok" or record.preflight.resolved is None:
            raise TaskContractError(
                f"{projection.label}: dispatch did not preflight the source launch geometry it selected "
                f"({record.preflight.status})"
            )
        if record.preflight.resolved.document() != resolved.document():
            raise TaskContractError(
                f"{projection.label}: the projected source launch geometry is not the geometry dispatch preflighted"
            )
    pair = geometry_pair(execution_geometry, None, instance=instance, contract=abi, limits=limits)
    expected_identity = pair.identity
    factory = _lifecycle_factory(
        projection=projection, resolved=resolved, contract=run_contract, target_id=target_id
    )
    lifecycle_document = None if factory is None else factory(resolved).document()
    expected_bundle_digest = CanonicalBundle(
        candidate=projection.candidate,
        pair=pair,
        lifecycle=None if factory is None else factory(resolved),
    ).bundle_digest

    result = evaluate_task(
        run_locator,
        track_id=projection.track_id,
        candidate=projection.candidate,
        builder=builder,
        evaluator=evaluator,
        geometry_resolver=None if projection.track_id == BUNDLE_TRACK_ID else resolver,
        fallback_geometry=None,
        lifecycle_factory=factory,
        track_version=track_version,
        track_registry_path=track_registry_path,
        target_id=target_id,
        max_buffer_length=max_buffer_length,
    )
    if result.candidate_sha256 != projection.candidate.candidate_sha256:
        raise TaskContractError(
            "the evaluation returned a different candidate identity than the projected provider (stale artifact)"
        )
    if result.track["track_id"] != projection.track_id:
        raise TaskContractError("the evaluation ran a different Track than the projected provider")
    if result.geometry_identity != expected_identity:
        raise TaskContractError(
            "the evaluated geometry identity is not the geometry the runner resolved before the build; the provider "
            "did not execute the frozen candidate geometry"
        )
    if result.bundle_digest != expected_bundle_digest:
        raise TaskContractError(
            "the evaluated canonical bundle digest is not the bundle the runner derived from the projected candidate, "
            f"the preflighted geometry and the lifecycle ({result.bundle_digest} != {expected_bundle_digest})"
        )
    report = result.report
    _check_native_digest(result, where=f"{projection.label}")
    status = str(report.get("status"))
    execution = result.execution
    runtime_provider: str | None = None
    execution_published = False
    observed: list[tuple[int, dict[str, Any]]] = []
    host_runs: list[dict[str, Any]] = []
    worker_status: str | None = None
    if execution is None:
        if status not in BUILD_FREE_STATUSES:
            raise TaskContractError(
                f"{projection.label}: the evaluator published no execution record for a {status!r} report; a real run "
                "must say what it executed"
            )
    else:
        runtime_provider = str(execution.get("provider"))
        if runtime_provider == RUNTIME_GEOMETRY_PROVIDER:
            execution_published = True
        elif status in NO_DISPATCH_STATUSES:
            # The bounded worker was killed or crashed before dispatching: it publishes a
            # build-free report plus a provider of its own.  That is not a dispatch and never a
            # GeometryPair fallback; the record keeps the worker status instead of inventing one.
            # A candidate compile error is the same shape: a real (failed) build, zero dispatched
            # stages, and the worker's own provider string instead of a candidate dispatch record.
            if execution.get("dispatch"):
                raise TaskContractError(
                    f"{projection.label}: the worker reported {status!r} but recorded dispatches; a build-free "
                    "status cannot carry executed stages"
                )
            worker_status = runtime_provider
            runtime_provider = None
        else:
            raise TaskContractError(
                f"{projection.label}: the runtime executed geometry source {runtime_provider!r}; a domain fallback is "
                "an ordinary candidate artifact, never a GeometryPair fallback"
            )
        if execution_published:
            observed = _observed_dispatch(execution.get("dispatch") or [])
            problems = _dispatch_mismatches(resolved, observed)
            if problems:
                raise TaskContractError(
                    f"{projection.label}: the real dispatch does not match the frozen geometry: {'; '.join(problems)}"
                )
            host_runs = [dict(item) for item in (execution.get("host_runs") or [])]
            if projection.track_id == BUNDLE_TRACK_ID:
                if execution.get("host_tiling") is None:
                    raise TaskContractError(
                        f"{projection.label}: the bundle executed without its Host Tiling plan; candidate host work "
                        "must run, not be declared"
                    )
                if not host_runs:
                    raise TaskContractError(
                        f"{projection.label}: the bundle never executed its Host Tiling plan"
                    )
    evaluation = {
        "provider": record.provider,
        "runtime_provider": runtime_provider,
        "execution_published": execution_published,
        "worker_status": worker_status,
        "fallback_reason": record.fallback_reason,
        "fallback_geometry": None,
        "compiled_candidate_sha256": result.candidate_sha256,
        "expected_geometry_identity": expected_identity,
        "expected_bundle_digest": expected_bundle_digest,
        "geometry_pair": pair.document(),
        "geometry_identity": result.geometry_identity,
        "bundle_digest": result.bundle_digest,
        "native_digest_expected": _native_digest(report),
        "artifact_set_digest": result.artifact_set_digest,
        "native_digest": result.native_digest,
        "evaluation_identity": result.identity,
        "presentation_sha256": result.presentation_sha256,
        "effective_correctness_only": bool(result.effective_correctness_only),
        "task_pack_digest": result.task_pack_digest,
        "locator": dict(result.locator),
        "track": dict(result.track),
        "report_status": status,
        "report_digest": fingerprint(dict(report)),
        "report_schema_version": report.get("schema_version"),
        "report_mode": report.get("mode"),
        "seed": report.get("seed"),
        "target_id": report.get("target_id"),
        "backend": report.get("backend"),
        "backend_version": report.get("backend_version"),
        "device_name": report.get("device_name"),
        "gpu_family": report.get("gpu_family"),
        "evaluation_fingerprint": report.get("evaluation_fingerprint"),
        "geometry_origin_path": projection.binding.geometry_origin_path,
        "geometry_origin_sha256": projection.binding.geometry_origin_sha256,
        "geometry_path": projection.binding.geometry_path,
        "projected_geometry_sha256": projected_geometry_sha256,
        "geometry": resolved.document(),
        "expected_dispatch": [_stage_signature(stage) for stage in resolved.active_stages],
        "dispatch": [signature for _, signature in observed],
        # The stage each dispatched signature belongs to.  Two active stages may share one signature,
        # so the stage index is what tells them apart (it is the runtime's own protocol identity).
        "dispatch_stages": [stage for stage, _ in observed],
        "expected_dispatch_stages": [int(stage.index) for stage in resolved.active_stages],
        "host_tiling": None if execution is None else execution.get("host_tiling"),
        "host_runs": host_runs,
        "workspace_bytes": {} if execution is None else dict(execution.get("workspace_bytes") or {}),
        "workspace_zero": {} if execution is None else dict(execution.get("workspace_zero") or {}),
        "lifecycle": lifecycle_document,
        "instance_key": parsed.instance_key,
        # A fallback's correctness and time are not the generated variant's credit.
        "credits_variant": False if record.provider == "fallback" else True,
    }
    return PortfolioRunResult(
        record=record,
        projection=projection,
        result=result,
        evaluation=evaluation,
    )


__all__ = [
    "EVALUATED_OUTCOMES",
    "PORTFOLIO_RUN_SCHEMA_ID",
    "PORTFOLIO_RUN_VERSION",
    "PortfolioRunResult",
    "RUNTIME_GEOMETRY_PROVIDER",
    "instances_from_plan",
    "locator_for_instance",
    "run_portfolio_task",
]
