"""Search-time evaluation of one shape-portfolio parent.

This is the Metal side of ``eval-portfolio``.  It runs the existing portfolio
executor once per domain point and, in timed mode, folds the benchmark points
into the existing two performance books.  It does not confirm or publish.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from taic.portfolio_confirmation import SEARCH_SEED
from tcenv.backends.metal.portfolio_performance import shard_evidence_document
from tcenv.backends.metal.search_candidate import evidence_for
from tcenv.bench import timing_is_uncertain
from tcenv.contract_errors import GeometryError, TaskContractError
from tcenv.evaluation import CandidateDeliveryError
from tcenv.portfolio.candidate import parse_portfolio
from tcenv.portfolio.dispatch import DispatchRequest
from tcenv.portfolio_assembly import check_parent, domain_for_sizes
from tcenv.portfolio_assembly.assembly import CheckRow
from tcenv.portfolio_performance.performance import PerformanceReport, ShardMeasurement

EVALUATION_SCHEMA_ID = "tcenv.portfolio-assembly.evaluation.1"
_RUN_ERRORS = (TaskContractError, GeometryError, CandidateDeliveryError)
_PERFORMANCE_FIELDS = (
    "specialized_hit_speedup",
    "effective_end_to_end_speedup",
    "specialized_hit_latency_ratio",
    "effective_end_to_end_latency_ratio",
    "specialized_geomean_latency_ratio",
    "effective_geomean_latency_ratio",
    "fallback_time_fraction",
    "fallback_count_fraction",
    "publishable",
    "failure_reasons",
)


@dataclass(frozen=True)
class PointRow:
    """@brief 一个域点跑完后交给 ``after_point`` 的那一行。"""

    instance_key: str
    size: str
    usage: str
    provider: str | None
    variant_key: str | None
    track_id: str | None
    status: str
    message: str
    phase: str
    candidate_us: float | None = None
    baseline_us: float | None = None
    latency_ratio: float | None = None
    cv: Any = None
    speedup: Any = None
    elapsed_s: float | None = None

    def as_correctness(self) -> dict[str, Any]:
        """@brief 正确性行的文档字段。

        @return 六个约定字段。
        """
        return {
            "instance_key": self.instance_key,
            "provider": self.provider,
            "variant_key": self.variant_key,
            "track_id": self.track_id,
            "status": self.status,
            "message": self.message,
        }


@dataclass(frozen=True)
class PortfolioEvaluation:
    """@brief 一次父包评测的结果。``publication_allowed`` 恒为 false。"""

    task: str
    dtype: str
    target_id: str
    sizes: tuple[str, ...]
    mode: str
    seed: int
    domain_plan_digest: str
    portfolio_digest: str | None
    parent_sha256: str | None
    check: dict[str, Any]
    correctness: tuple[dict[str, Any], ...]
    shards: tuple[dict[str, Any], ...]
    performance: dict[str, Any] | None
    status: str
    problems: tuple[str, ...]

    def document(self) -> dict[str, Any]:
        """@brief 评测文档。

        @return 可序列化字段。``evidence_kind`` 与 ``publication_allowed`` 是固定值。
        """
        return {
            "schema_id": EVALUATION_SCHEMA_ID,
            "task": self.task,
            "dtype": self.dtype,
            "target_id": self.target_id,
            "sizes": list(self.sizes),
            "mode": self.mode,
            "seed": self.seed,
            "domain_plan_digest": self.domain_plan_digest,
            "portfolio_digest": self.portfolio_digest,
            "parent_sha256": self.parent_sha256,
            "check": self.check,
            "correctness": [dict(row) for row in self.correctness],
            "shards": [dict(row) for row in self.shards],
            "performance": None if self.performance is None else dict(self.performance),
            "status": self.status,
            "problems": list(self.problems),
            "evidence_kind": "search_time",
            "publication_allowed": False,
        }


def evaluate_parent(
    locator,
    entries,
    *,
    target,
    sizes,
    mode: str,
    seed: int = SEARCH_SEED,
    reps: int = 15,
    run_point: Callable | None = None,
    before_point: Callable | None = None,
    after_point: Callable | None = None,
) -> PortfolioEvaluation:
    """@brief 检查父包，再按域点跑正确性；``timed`` 时再计时并算两本账。

    ``run_point`` 缺省时走 :func:`_run_point`。``before_point`` 返回非空字符串就停下，
    原因写入 ``problems``，当前点和后面的点都不再跑。``after_point`` 在每个实际跑过的点
    之后调用，参数是 :class:`PointRow`。

    @param locator 任务定位器，``load()`` 给出合同。
    @param entries 父包 entries。
    @param target 目标档案。
    @param sizes 这次评测的 size，必须和打包时同一组。
    @param mode ``quick`` 或 ``timed``。
    @param seed 搜索期种子，默认 ``SEARCH_SEED``。
    @param reps 计时重复次数。
    @param run_point 跑一个点的函数；测试注入。
    @param before_point 每个点开跑前调用。
    @param after_point 每个点跑完调用。
    @return 评测结果。
    @raises PortfolioAssemblyError ``mode`` 不是 quick 或 timed。
    """
    from tcenv.portfolio_assembly import PortfolioAssemblyError

    if mode not in ("quick", "timed"):
        raise PortfolioAssemblyError(f"mode must be quick or timed, got {mode!r}")
    contract = locator.load()
    domain = domain_for_sizes(contract, target, sizes)
    check = check_parent(entries, domain)
    base = _base(contract, target, domain, check, mode, seed)
    if not check.ok:
        return _finish(base, status="rejected", problems=check.problems, correctness=(), shards=(), performance=None)

    portfolio = parse_portfolio(entries, domain_plan=domain.plan)
    evidence = evidence_for(portfolio)
    runner = run_point or _run_point
    by_size = {(item.dtype, item.size): item for item in domain.instances}
    by_key = {row.instance_key: row for row in check.rows}
    correctness_points = [
        point for point in domain.plan.instances if mode == "quick" or point.usage == "golden"
    ]
    timing_points = [point for point in domain.plan.instances if mode == "timed" and point.usage == "benchmark"]

    problems = list(check.problems)
    correctness: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    measurements: list[ShardMeasurement] = []
    saw_hang = False
    stopped_early = False

    def run_one(point, phase: str, run_mode: str) -> bool:
        nonlocal saw_hang, stopped_early
        if before_point is not None:
            reason = before_point()
            if isinstance(reason, str) and reason != "":
                problems.append(reason)
                stopped_early = True
                return False
        found = domain.plan.find(point.dtype, point.size, point.usage)
        if found is None:
            raise TaskContractError(f"{point.instance_key} is not in the domain plan")
        request = DispatchRequest.from_instance(
            by_size[(point.dtype, point.size)],
            target_abi_digest=found.target_abi_digest,
            usage=point.usage,
        )
        started = time.perf_counter()
        try:
            run = runner(locator, domain.plan, portfolio, request, evidence, mode=run_mode, seed=seed, reps=reps)
        except _RUN_ERRORS as exc:
            row = _row(
                point,
                by_key.get(point.instance_key),
                status="rejected",
                message=str(exc),
                phase=phase,
                elapsed_s=time.perf_counter() - started,
            )
            _keep(row, phase, correctness, shards)
            if after_point is not None:
                after_point(row)
            return True
        elapsed = time.perf_counter() - started
        row, measurement, shard = _interpret(
            run, request, point, by_key.get(point.instance_key), phase, elapsed
        )
        if measurement is not None:
            measurements.append(measurement)
        if phase == "correctness":
            correctness.append(row.as_correctness())
        else:
            shards.append(shard)
        if after_point is not None:
            after_point(row)
        if row.status in ("timeout", "crash"):
            saw_hang = True
            return False
        return True

    for point in correctness_points:
        if not run_one(point, "correctness", "quick"):
            break
    if not saw_hang and not stopped_early:
        for point in timing_points:
            if not run_one(point, "timing", "full"):
                break

    performance = None
    if mode == "timed" and not saw_hang and not stopped_early:
        benchmark_keys = [point.instance_key for point in timing_points]
        if measurements and len(measurements) == len(benchmark_keys):
            try:
                performance = _performance_document(
                    PerformanceReport.evaluate(measurements, expected_benchmark_keys=benchmark_keys)
                )
            except TaskContractError as exc:
                problems.append(str(exc))
    rows = list(correctness) + list(shards)
    status = _final_status(mode, saw_hang, stopped_early, rows, performance)
    for row in rows:
        if row.get("status") == "ok":
            continue
        message = str(row.get("message") or row.get("status") or "rejected")
        problems.append(f"{row.get('instance_key')}: {message}")
    if status == "rejected" and performance is not None:
        # timing_uncertain 会原样复测，不是下一场组装会话能够修复的拒绝原因。
        problems.extend(
            str(reason) for reason in performance["failure_reasons"]
            if not str(reason).endswith(":timing_uncertain")
        )
    return _finish(
        base,
        status=status,
        problems=tuple(dict.fromkeys(str(problem) for problem in problems)),
        correctness=tuple(correctness),
        shards=tuple(shards),
        performance=performance,
    )


def _run_point(locator, plan, portfolio, request, evidence, *, mode, seed, reps):
    """@brief 默认的单点执行：正确性 ``quick``，计时 ``full`` 且 ``cases=("normal",)``。

    @param locator 任务定位器。
    @param plan 域计划。
    @param portfolio 解析后的父包。
    @param request 该点的调度请求。
    @param evidence provider 证据状态。
    @param mode ``quick`` 或 ``full``。
    @param seed 搜索期种子。
    @param reps 重复次数。
    @return ``run_portfolio_task`` 的结果。
    """
    from tcenv.backends.metal.track_adapter import (
        MetalTrackBuilder,
        MetalTrackEvaluator,
        msl_single_geometry_resolver,
    )
    from tcenv.taskcontract.portfolio_runner import run_portfolio_task

    cases = ("normal",) if mode == "full" else None
    return run_portfolio_task(
        locator=locator,
        domain_plan=plan,
        portfolio=portfolio,
        request=request,
        evidence=evidence,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode=mode, seed=seed, reps=reps, cases=cases),
        geometry_resolver_factory=msl_single_geometry_resolver,
    )


def _measurement_from_run(run, request) -> ShardMeasurement:
    """@brief 从一次已经完成的运行里取出 ``ShardMeasurement``。

    字段取法照抄 ``measure_shard_with_evidence``
    （``tcenv/backends/metal/portfolio_performance.py``）。那个函数被验收清单钉住，不能改，
    而且它会自己再跑一次 ``run_portfolio_task``，所以这里不能调用它，只重写字段提取。

    @param run 单点运行结果。
    @param request 该点的调度请求。
    @return 分片测量。
    @raises TaskContractError 结果不完整或没有候选中位数。
    """
    if run.result is None:
        raise TaskContractError("a performance measurement needs a complete evaluation")
    report = run.result.report
    benchmark = report.get("benchmark") or {}
    baseline_name = benchmark.get("best_baseline")
    baseline_us = None
    if baseline_name is not None:
        baseline_us = (report.get("baseline_us") or {}).get(baseline_name)
    candidate_us = report.get("candidate_us")
    if candidate_us is None:
        raise TaskContractError("a benchmark measurement has no candidate median")
    correctness = report.get("correctness") or {}
    dispatch_ok = run.evaluation.get("dispatch") == run.evaluation.get("expected_dispatch")
    return ShardMeasurement(
        instance_key=request.instance_key,
        usage=request.usage,
        provider=run.record.provider,
        candidate_us=float(candidate_us),
        baseline_us=None if baseline_us is None else float(baseline_us),
        timing_uncertain=timing_is_uncertain(report),
        correctness_ok=bool(correctness.get("ok") and report.get("status") == "ok"),
        dispatch_ok=bool(dispatch_ok),
        unknown=report.get("evaluation_fingerprint") is None,
    )


def _interpret(run, request, point, check_row: CheckRow | None, phase: str, elapsed: float):
    """@brief 把一次运行收成行、可选的测量和分片文档。

    @param run 单点运行结果。
    @param request 调度请求。
    @param point 域点。
    @param check_row 检查表里的对应行。
    @param phase ``correctness`` 或 ``timing``。
    @param elapsed 该点耗时（秒）。
    @return ``(PointRow, measurement 或 None, 分片文档或 None)``。
    """
    status, message, report = _status_of(run)
    measurement = None
    shard = None
    if phase == "timing" and status == "ok":
        try:
            measurement = _measurement_from_run(run, request)
            shard = shard_evidence_document(run, measurement)
            shard["status"] = "ok"
            shard["message"] = ""
        except TaskContractError as exc:
            status = "rejected"
            message = str(exc)
            measurement = None
            shard = None
    if phase == "timing" and shard is None:
        shard = {"instance_key": point.instance_key, "status": status, "message": message}
    provider = _provider(run, check_row)
    variant_key = None if provider != "candidate" or check_row is None else check_row.variant_key
    track_id = _track_id(run, check_row)
    candidate_us, baseline_us, ratio, cv, speedup = _timing_fields(report, measurement)
    row = PointRow(
        instance_key=point.instance_key,
        size=point.size,
        usage=point.usage,
        provider=provider,
        variant_key=variant_key,
        track_id=track_id,
        status=status,
        message=message,
        phase=phase,
        candidate_us=candidate_us,
        baseline_us=baseline_us,
        latency_ratio=ratio,
        cv=cv,
        speedup=speedup,
        elapsed_s=elapsed,
    )
    return row, measurement, shard


def _status_of(run) -> tuple[str, str, dict]:
    """@brief 从运行结果读状态。``result`` 为空时记为 rejected，消息是 ``outcome``。

    @param run 单点运行结果。
    @return ``(status, message, report)``。
    """
    if run.result is None:
        return "rejected", str(getattr(run.record, "outcome", "")), {}
    report = run.result.report
    if not isinstance(report, dict):
        return "rejected", "portfolio run report must be an object", {}
    return str(report.get("status") or "rejected"), str(report.get("message") or ""), report


def _provider(run, check_row: CheckRow | None) -> str | None:
    """@brief 优先用这次运行写出的 provider，否则用检查行。

    @param run 单点运行结果。
    @param check_row 检查行。
    @return provider 名字。
    """
    provider = getattr(getattr(run, "record", None), "provider", None)
    if provider in ("candidate", "fallback"):
        return provider
    if check_row is not None:
        return check_row.provider
    return None


def _track_id(run, check_row: CheckRow | None) -> str | None:
    """@brief 从评测文档或检查行取 track。

    @param run 单点运行结果。
    @param check_row 检查行。
    @return track id。
    """
    evaluation = getattr(run, "evaluation", None)
    if isinstance(evaluation, dict):
        track = evaluation.get("track") or {}
        if isinstance(track, dict) and track.get("track_id"):
            return track.get("track_id")
    if check_row is not None:
        return check_row.track_id
    return None


def _timing_fields(report: dict, measurement: ShardMeasurement | None):
    """@brief 取出自测行和分片行要用的时间字段。

    @param report 运行报告。
    @param measurement 已构造的测量；没有时为 None。
    @return ``(candidate_us, baseline_us, latency_ratio, cv, speedup)``。
    """
    benchmark = report.get("benchmark") or {}
    candidate = benchmark.get("candidate") if isinstance(benchmark.get("candidate"), dict) else {}
    cv = candidate.get("cv")
    speedup = report.get("speedup")
    if measurement is None:
        return report.get("candidate_us"), None, None, cv, speedup
    return measurement.candidate_us, measurement.baseline_us, measurement.latency_ratio, cv, speedup


def _row(point, check_row: CheckRow | None, *, status: str, message: str, phase: str, elapsed_s: float) -> PointRow:
    """@brief 构造一个没有运行结果的行（异常或未产出报告）。

    @param point 域点。
    @param check_row 检查行。
    @param status 行状态。
    @param message 消息。
    @param phase ``correctness`` 或 ``timing``。
    @param elapsed_s 耗时。
    @return 行。
    """
    provider = check_row.provider if check_row is not None else None
    variant_key = None if provider != "candidate" or check_row is None else check_row.variant_key
    return PointRow(
        instance_key=point.instance_key,
        size=point.size,
        usage=point.usage,
        provider=provider,
        variant_key=variant_key,
        track_id=None if check_row is None else check_row.track_id,
        status=status,
        message=message,
        phase=phase,
        elapsed_s=elapsed_s,
    )


def _keep(row: PointRow, phase: str, correctness: list, shards: list) -> None:
    """@brief 把异常行放进正确性列表或分片列表。

    @param row 行。
    @param phase ``correctness`` 或 ``timing``。
    @param correctness 正确性文档列表。
    @param shards 分片文档列表。
    """
    if phase == "correctness":
        correctness.append(row.as_correctness())
        return
    shards.append({"instance_key": row.instance_key, "status": row.status, "message": row.message})


def _performance_document(report: PerformanceReport) -> dict[str, Any]:
    """@brief 取出 ``PerformanceReport`` 的十个标量字段。

    @param report 两本账。
    @return 字段字典。``failure_reasons`` 是列表。
    """
    document = {}
    for name in _PERFORMANCE_FIELDS:
        value = getattr(report, name)
        document[name] = list(value) if name == "failure_reasons" else value
    return document


def _final_status(mode: str, saw_hang: bool, stopped_early: bool, rows: Sequence[dict], performance) -> str:
    """@brief 按计划的优先级决定最终状态。

    @param mode ``quick`` 或 ``timed``。
    @param saw_hang 是否已有 timeout 或 crash。
    @param stopped_early ``before_point`` 是否叫停。
    @param rows 已经产生的正确性行和分片行。
    @param performance 两本账；没算时为 None。
    @return ``hang``、``rejected``、``timing_uncertain`` 或 ``ok``。
    """
    if saw_hang:
        return "hang"
    if stopped_early or any(row.get("status") != "ok" for row in rows):
        return "rejected"
    if mode == "quick":
        return "ok"
    if performance is None:
        return "rejected"
    reasons = list(performance["failure_reasons"])
    if reasons and all(str(reason).endswith(":timing_uncertain") for reason in reasons):
        return "timing_uncertain"
    if performance["publishable"]:
        return "ok"
    return "rejected"


def _base(contract, target, domain, check, mode: str, seed: int) -> dict[str, Any]:
    """@brief 评测文档里与逐点结果无关的身份字段。

    @param contract 任务合同。
    @param target 目标档案。
    @param domain 组装域。
    @param check 检查报告。
    @param mode ``quick`` 或 ``timed``。
    @param seed 搜索期种子。
    @return 字段字典。
    """
    return {
        "task": contract.op_name,
        "dtype": contract.dtype,
        "target_id": target.id,
        "sizes": tuple(domain.sizes),
        "mode": mode,
        "seed": seed,
        "domain_plan_digest": check.domain_plan_digest,
        "portfolio_digest": check.portfolio_digest,
        "parent_sha256": check.parent_sha256,
        "check": check.document(),
    }


def _finish(base: dict[str, Any], **fields) -> PortfolioEvaluation:
    """@brief 用身份字段和结果字段组成评测对象。

    @param base 身份字段。
    @param fields 结果字段。
    @return 评测对象。
    """
    return PortfolioEvaluation(**base, **fields)


__all__ = [
    "EVALUATION_SCHEMA_ID",
    "PointRow",
    "PortfolioEvaluation",
    "evaluate_parent",
]
