"""把「一颗 msl-single 候选」包成 shape-portfolio 的正式证据，供搜索循环使用。

搜索只交付一个 ``.metal`` 文件（候选自带 ``/* taic-launch */``），但评测与确认**不另起一套**：
这里把该文件包装成一个单 variant、恒真 guard 的 portfolio，然后一切交给 P0/P2 已有的驱动——
``run_portfolio_task``（搜索期评测）与 ``confirm_subject``（fresh 确认 + R5 身份比较）。

纪律（与 P2 完全一致）：

- 语义域只含搜索的那一个 ``(dtype, size)`` 实例；``golden``/``benchmark`` 两个用途点都由
  ``normalize_semantic_domain_plan`` 派生，不由本模块发明。
- 搜索期评测的 seed 恒为 ``SEARCH_SEED``（7）。fresh 只发生在确认：确认 seed 由
  ``ValidationPlan`` 派生，``confirm_subject`` 会拒绝任何与搜索种子相同的确认。
- 被搜索的 benchmark 实例是 **tune**  shard（它确实被调参），确认 sidecar 的
  ``coverage_kind`` 因此是 ``tune_confirmation``；同尺寸的 golden 点是 required unseen。
  只有一个用途点的尺寸无法派生 ValidationPlan（tune/unseen 至少两个不同点），那种运行不做确认。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from taic.portfolio_confirmation import SEARCH_SEED
from tcenv.contract_errors import TaskContractError
from tcenv.identity import fingerprint
from tcenv.portfolio.candidate import PortfolioCandidate, parse_portfolio
from tcenv.portfolio.dispatch import DispatchRequest
from tcenv.portfolio.domain import (
    DomainAuthority,
    SemanticDomainPlan,
    guard_digest,
    instance_key,
    normalize_semantic_domain_plan,
)
from tcenv.portfolio_validation.validation import PortfolioFreeze, ValidationPlan
from tcenv.target import TargetProfile
from tcenv.taskcontract.loader import TaskContract
from tcenv.taskcontract.locator import TaskLocator
from tcenv.taskcontract.portfolio_runner import PortfolioRunResult, run_portfolio_task
from tcenv.track.plane import CandidateEntry

from .portfolio_confirmation import confirm_subject
from .track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver

#: 单候选 portfolio 的固定拼写。guard 不引用任何 symbol：任何 OpSpec 上它都恒真，
#: 所以 dispatch 永远选中这唯一的 variant，不存在「搜索候选落在域外」这种伪状态。
TRACK_ID = "msl-single"
CANDIDATE_PATH = "device/candidate.metal"
PORTFOLIO_ENTRY_PATH = "coverage_plan/portfolio.json"
BUCKET_KEY = "all"
VARIANT_KEY = "candidate"
GUARD_EXPR: list[Any] = ["ge", 0, 0]
PLANNER = {"id": "taic.search.single", "version": 1}
#: freeze.protocol_epoch：搜索侧 portfolio 证据的第一代（P2 的 fixture 用 1；这是证据代次，不是报告 schema）。
PROTOCOL_EPOCH = 1


def usage_for(contract: TaskContract) -> str:
    """@brief 这次搜索评测的用途：有 benchmark 就计时，否则只做 golden 正确性。

    @param contract v2 任务合同（已绑定 dtype/size）。
    @return ``"benchmark"`` 或声明的第一个用途。
    """
    declared = tuple(contract.instance.size_usage)
    if not declared:
        raise TaskContractError(f"{contract.op_name}.{contract.size}: the size declares no usage")
    return "benchmark" if "benchmark" in declared else declared[0]


def mode_for(contract: TaskContract) -> str:
    """@brief 与 ``confirm_subject`` 相同的用途→模式映射。

    @param contract v2 任务合同。
    @return ``"full"``（benchmark）或 ``"quick"``（golden）。
    """
    return "full" if usage_for(contract) == "benchmark" else "quick"


def domain_for(contract: TaskContract, target: TargetProfile) -> tuple[SemanticDomainPlan, DomainAuthority]:
    """@brief 由权威 OpSpec/target/实例派生语义域与 authority（搜索只含当前实例）。

    @param contract v2 任务合同。
    @param target 目标档案。
    @return ``(domain_plan, authority)``。
    """
    plan = normalize_semantic_domain_plan(contract.spec, (contract.instance,), target=target)
    authority = DomainAuthority(
        plan=plan, spec=contract.spec, target=target, instances=(contract.instance,), contract=contract,
    )
    return plan, authority


def envelope_entries(source: str, plan: SemanticDomainPlan) -> tuple[CandidateEntry, ...]:
    """@brief 把单文件候选包成父 bundle entries（候选字节 + 固定 manifest）。

    @param source 候选 MSL 正文。
    @param plan 该候选绑定的语义域计划。
    @return 父 entries（manifest 在最后，字节确定性序列化）。
    """
    if not isinstance(source, str) or not source.strip():
        raise TaskContractError("a search candidate needs non-empty source text")
    digest = guard_digest(GUARD_EXPR, where=VARIANT_KEY)
    document = {
        "schema_id": "tcenv.candidate-portfolio.1",
        "domain_plan_digest": plan.digest,
        "planner": dict(PLANNER),
        "coverage_plan": {"buckets": [{"bucket_key": BUCKET_KEY, "guard_digest": digest}]},
        "variants": [
            {
                "variant_key": VARIANT_KEY,
                "bucket_key": BUCKET_KEY,
                "guard": {"expr": GUARD_EXPR, "digest": digest},
                "entries": [CANDIDATE_PATH],
                "geometry_entry": None,
            }
        ],
    }
    manifest = json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    return (
        CandidateEntry(path=CANDIDATE_PATH, role="device_kernel", media_type="text/plain", data=source.encode("utf-8")),
        CandidateEntry(path=PORTFOLIO_ENTRY_PATH, role="launch", media_type="application/json", data=manifest),
    )


def portfolio_for(source: str, plan: SemanticDomainPlan) -> PortfolioCandidate:
    """@brief 解析这份单候选 portfolio（闭集校验，含 guard/摘要一致性）。

    @param source 候选正文。
    @param plan 语义域计划。
    @return 解析后的 portfolio。
    """
    return parse_portfolio(envelope_entries(source, plan), domain_plan=plan)


def evidence_for(portfolio: PortfolioCandidate) -> dict[str, str]:
    """@brief 全部 provider 标 ``validated``（单候选没有第二个 provider）。

    @param portfolio 解析后的 portfolio。
    @return provider identity → 证据状态。
    """
    evidence = {variant_id: "validated" for variant_id in portfolio.variant_ids}
    if portfolio.fallback_id is not None:
        evidence[portfolio.fallback_id] = "validated"
    return evidence


def request_for(contract: TaskContract, plan: SemanticDomainPlan) -> DispatchRequest:
    """@brief 当前实例的 dispatch 请求（摘要由 from_instance 计算）。

    @param contract v2 任务合同。
    @param plan 语义域计划。
    @return 请求。
    """
    usage = usage_for(contract)
    point = plan.find(contract.dtype, contract.size, usage)
    if point is None:
        raise TaskContractError(
            f"{contract.op_name}.{contract.dtype}.{contract.size}/{usage}: not in the derived domain plan"
        )
    return DispatchRequest.from_instance(contract.instance, target_abi_digest=point.target_abi_digest, usage=usage)


def evaluate_candidate(
    locator: TaskLocator,
    source: str,
    *,
    target: TargetProfile,
    mode: str | None = None,
    seed: int = SEARCH_SEED,
    reps: int = 15,
    cases: tuple[str, ...] | None = None,
    track_registry_path=None,
) -> PortfolioRunResult:
    """@brief 搜索期评测一颗候选：唯一入口是 ``run_portfolio_task``。

    @param locator 任务 locator（实例必须与请求一致）。
    @param source 候选正文。
    @param target 目标档案。
    @param mode ``full``/``quick``；缺省按用途映射。
    @param seed 输入种子；搜索期固定 7，fresh 只属于确认。
    @param reps 计时重复次数。
    @param cases 用例子集；None 跑合同声明的全部用例。
    @param track_registry_path 可选 track registry（测试用）。
    @return 组合证据（dispatch 记录 + 投影 + schema-4 报告）。
    """
    contract = locator.load()
    plan, _authority = domain_for(contract, target)
    portfolio = portfolio_for(source, plan)
    return run_portfolio_task(
        locator=locator,
        domain_plan=plan,
        portfolio=portfolio,
        request=request_for(contract, plan),
        evidence=evidence_for(portfolio),
        builder=MetalTrackBuilder(target_id=target.id),
        evaluator=MetalTrackEvaluator(
            target_id=target.id,
            mode=mode or mode_for(contract),
            seed=seed,
            reps=reps,
            cases=cases,
        ),
        geometry_resolver_factory=msl_single_geometry_resolver,
        target_id=target.id,
        track_registry_path=track_registry_path,
    )


@dataclass(frozen=True)
class SearchConfirmationInput:
    """@brief 一次确认所需的冻结输入（可落盘、可在 resume 时逐字节重建）。"""

    freeze: PortfolioFreeze
    validation: ValidationPlan
    request: DispatchRequest
    evidence: Mapping[str, str]


def confirmation_input(
    locator: TaskLocator,
    source: str,
    *,
    target: TargetProfile,
    guidance_digest: str,
    master_seed: int,
) -> SearchConfirmationInput:
    """@brief 为优胜候选派生 freeze + validation plan（不读任何搜索报告）。

    ``task_pack_digest`` 用 locator 的部署包摘要（``confirm_subject`` 会再核一次）；
    ``guidance_digest`` 由调用方从真实题面字节算出，是本运行的外部信任输入。

    @param locator 任务 locator。
    @param source 优胜候选正文。
    @param target 目标档案。
    @param guidance_digest 题面/guidance 的摘要。
    @param master_seed 运行种子（HMAC key，永不写入文档）。
    @return 冻结输入。
    @raises TaskContractError 尺寸只有一个用途点（tune/unseen 无法同时成立）。
    """
    contract = locator.load()
    plan, authority = domain_for(contract, target)
    entries = envelope_entries(source, plan)
    freeze = PortfolioFreeze(
        protocol_epoch=PROTOCOL_EPOCH,
        authority=authority,
        candidate_entries=entries,
        task_pack_digest=locator.deployment_pack_digest,
        guidance_digest=guidance_digest,
    )
    if usage_for(contract) != "benchmark":
        raise TaskContractError(
            f"{contract.op_name}.{contract.dtype}.{contract.size}: only a benchmark instance has a speedup to "
            "confirm; a golden-only run publishes correctness, not a timing conclusion"
        )
    tuned = instance_key(contract.dtype, contract.size, "benchmark")
    golden = instance_key(contract.dtype, contract.size, "golden")
    # tune/unseen 至少要两个不同的点：被调参的 benchmark 实例是 tune，同尺寸 golden 点是 required unseen。
    # 只有一个用途点的尺寸没有这个自由度，调用方必须跳过确认（没有可发布的计时结论）。
    if golden not in set(freeze.instance_keys()):
        raise TaskContractError(
            f"{contract.op_name}.{contract.dtype}.{contract.size}: a single-usage size cannot derive a "
            "validation plan (the tuned instance and a required unseen instance must be two distinct points)"
        )
    validation = ValidationPlan.derive(
        freeze,
        tune_shards=(tuned,),
        unseen_count=1,
        master_seed=master_seed,
        required_instance_keys=(golden,),
    )
    portfolio = freeze.portfolio()
    return SearchConfirmationInput(
        freeze=freeze,
        validation=validation,
        request=request_for(contract, plan),
        evidence=evidence_for(portfolio),
    )


def confirm_candidate(
    locator: TaskLocator,
    source: str,
    *,
    target: TargetProfile,
    guidance_digest: str,
    master_seed: int,
) -> dict[str, Any]:
    """@brief 对优胜候选做 P2 的 fresh 确认，返回 subject sidecar 文档。

    @param locator 任务 locator。
    @param source 优胜候选正文。
    @param target 目标档案。
    @param guidance_digest 题面摘要。
    @param master_seed 运行种子。
    @return ``confirm_subject`` 的 verdict 文档（含 search/confirmation 两份原始报告与 identity）。
    """
    frozen = confirmation_input(
        locator, source, target=target, guidance_digest=guidance_digest, master_seed=master_seed,
    )
    return confirm_subject(
        locator=locator,
        domain_plan=frozen.freeze.domain_plan,
        portfolio=frozen.freeze.portfolio(),
        request=frozen.request,
        evidence=frozen.evidence,
        validation=frozen.validation,
        freeze=frozen.freeze,
    )


def guidance_fingerprint(*parts: str) -> str:
    """@brief 题面/guidance 的权威摘要（freeze 的外部信任输入）。

    @param parts 组成题面的文本段（系统提示、渲染契约等）。
    @return 十六进制 sha256 指纹。
    """
    return fingerprint({"guidance": list(parts)})


__all__ = [
    "CANDIDATE_PATH",
    "GUARD_EXPR",
    "PROTOCOL_EPOCH",
    "SEARCH_SEED",
    "TRACK_ID",
    "SearchConfirmationInput",
    "confirm_candidate",
    "confirmation_input",
    "domain_for",
    "envelope_entries",
    "evaluate_candidate",
    "evidence_for",
    "guidance_fingerprint",
    "mode_for",
    "portfolio_for",
    "request_for",
    "usage_for",
]
