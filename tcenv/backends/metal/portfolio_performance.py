"""P3-B/C: Metal measurement to backend-neutral ShardMeasurement + audit evidence."""

from __future__ import annotations

from typing import Any, Mapping

from tcenv.bench import timing_is_uncertain
from tcenv.contract_errors import TaskContractError
from tcenv.portfolio_performance.performance import ShardMeasurement
from tcenv.taskcontract.portfolio_runner import run_portfolio_task
from .track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver


def shard_evidence_document(run: Any, measurement: ShardMeasurement) -> dict[str, Any]:
    """@brief 把一次真实 Metal portfolio 分片测量投成可逐字段独立复核的证据文档。

    包含真实设备名、实际 dispatch、provider、variant/fallback 身份、candidate/geometry/artifact/
    evaluation 身份，以及计时与误差标量。对比 archive vs live 时，标量时间可排除，身份与协议字
    段必须完全一致。
    """
    if run.result is None or run.evaluation is None:
        raise TaskContractError("cannot project shard evidence from an incomplete portfolio run")
    report = run.result.report
    benchmark = report.get("benchmark") or {}
    correctness = report.get("correctness") or {}
    cases = list(correctness.get("cases") or [])
    worst_scaled_err = max((float(item.get("scaled_err", 0.0)) for item in cases), default=0.0)
    record = run.record
    projection = run.projection
    evaluation = run.evaluation
    return {
        "instance_key": measurement.instance_key,
        "usage": measurement.usage,
        "provider": measurement.provider,
        "variant_id": record.variant_id,
        "variant_key": None
        if projection is None or measurement.provider != "candidate"
        else projection.label,
        "fallback_id": record.fallback_id,
        "fallback_reason": record.fallback_reason,
        "credits_variant": bool(evaluation.get("credits_variant")),
        "track_id": evaluation.get("track", {}).get("track_id"),
        "target_id": evaluation.get("target_id"),
        "device_name": evaluation.get("device_name"),
        "gpu_family": evaluation.get("gpu_family"),
        "candidate_sha256": evaluation.get("compiled_candidate_sha256"),
        "geometry_identity": evaluation.get("geometry_identity"),
        "bundle_digest": evaluation.get("bundle_digest"),
        "artifact_set_digest": evaluation.get("artifact_set_digest"),
        "native_digest": evaluation.get("native_digest"),
        "evaluation_fingerprint": evaluation.get("evaluation_fingerprint"),
        "task_pack_digest": evaluation.get("task_pack_digest"),
        "dispatch": list(evaluation.get("dispatch") or []),
        "expected_dispatch": list(evaluation.get("expected_dispatch") or []),
        "dispatch_ok": bool(measurement.dispatch_ok),
        "correctness_ok": bool(measurement.correctness_ok),
        "worst_scaled_err": worst_scaled_err,
        "best_baseline": benchmark.get("best_baseline"),
        "baselines_measured": sorted((benchmark.get("baselines") or {}).keys()),
        "rotation_sets": benchmark.get("rotation_sets"),
        "attempts": benchmark.get("attempts"),
        "noise_cv": benchmark.get("noise_cv"),
        "timing_uncertain": bool(measurement.timing_uncertain),
        "unknown": bool(measurement.unknown),
        "candidate_us": measurement.candidate_us,
        "baseline_us": measurement.baseline_us,
        "latency_ratio": measurement.latency_ratio,
    }


def measure_shard_with_evidence(
    *,
    locator,
    domain_plan,
    portfolio,
    request,
    evidence: Mapping[str, str],
    seed: int = 7,
    reps: int = 15,
) -> tuple[ShardMeasurement, dict[str, Any]]:
    """@brief 执行一次分片测量并返回 ``(ShardMeasurement, 完整证据文档)``。"""
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
    measurement = ShardMeasurement(
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
    return measurement, shard_evidence_document(run, measurement)


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
    measurement, _ = measure_shard_with_evidence(
        locator=locator,
        domain_plan=domain_plan,
        portfolio=portfolio,
        request=request,
        evidence=evidence,
        seed=seed,
        reps=reps,
    )
    return measurement


__all__ = ["measure_shard", "measure_shard_with_evidence", "shard_evidence_document"]
