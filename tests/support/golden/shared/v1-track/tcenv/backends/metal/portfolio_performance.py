"""P3-B: Metal measurement to backend-neutral ShardMeasurement (no speedup trust)."""

from __future__ import annotations

from typing import Any, Mapping

from tcenv.bench import timing_is_uncertain
from tcenv.contract_errors import TaskContractError
from tcenv.portfolio_performance.performance import ShardMeasurement
from tcenv.taskcontract.portfolio_runner import run_portfolio_task
from .track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver


def measure_shard(
    *,
    locator,
    domain_plan,
    portfolio,
    request,
    evidence: Mapping[str, str],
    seed: int = 7,
    reps: int = 15,
) -> ShardMeasurement:
    """@brief 用真实交错计时构造输入，candidate/baseline 均取生产 median_us。"""
    run = run_portfolio_task(
        locator=locator,
        domain_plan=domain_plan,
        portfolio=portfolio,
        request=request,
        evidence=evidence,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="full", seed=seed, reps=reps, cases=("normal",)),
        geometry_resolver_factory=msl_single_geometry_resolver,
    )
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


__all__ = ["measure_shard"]
