"""P3-A CPU tests: two performance books and the per-shard 1.10 gate."""

from __future__ import annotations

import pytest

from tcenv.contract_errors import TaskContractError
from tcenv.portfolio_performance.performance import (
    MAX_REGRESSION_RATIO,
    PerformanceReport,
    ShardMeasurement,
)


def row(key="fp16/threshold/benchmark", *, provider="candidate", candidate=10.0, baseline=10.0, **kwargs):
    return ShardMeasurement(
        instance_key=key,
        usage=key.rsplit("/", 1)[1],
        provider=provider,
        candidate_us=candidate,
        baseline_us=baseline,
        **kwargs,
    )


def test_p3_a_failed_report_has_no_fallback_or_speedup_aggregates():
    report = PerformanceReport.evaluate([
        row(candidate=5.0, baseline=10.0),
        row("fp16/hole/benchmark", provider="fallback", candidate=50.0, baseline=10.0),
    ])
    assert report.publishable is False
    assert report.fallback_time_fraction is None
    assert report.fallback_count_fraction is None
    assert report.specialized_hit_speedup is None
    assert report.effective_end_to_end_speedup is None
    assert MAX_REGRESSION_RATIO == 1.10
    golden = row("fp16/threshold/golden")
    report = PerformanceReport.evaluate([golden, row()])
    assert report.verdicts[0].status == "ignored_golden"
    assert report.specialized_hit_speedup == 1.0
    assert report.effective_end_to_end_speedup == 1.0
    assert report.publishable is True


def test_p3_a_two_books_include_fallback_only_in_effective_accounting():
    measurements = [
        row(candidate=8.0, baseline=10.0),
        row("fp16/hole/benchmark", provider="fallback", candidate=20.0, baseline=10.0),
    ]
    report = PerformanceReport.evaluate(measurements)
    assert report.specialized_hit_speedup is None
    assert report.effective_end_to_end_speedup is None
    assert report.specialized_geomean_latency_ratio is None
    assert report.effective_geomean_latency_ratio is None
    assert report.publishable is False
    assert any("effective_ratio" in reason for reason in report.failure_reasons)


def test_p3_a_per_shard_specialized_regression_cannot_be_hidden_by_geomean():
    measurements = [row(candidate=8.0, baseline=10.0), row("fp16/tail/benchmark", candidate=20.0, baseline=10.0)]
    report = PerformanceReport.evaluate(measurements)
    assert report.specialized_geomean_latency_ratio is None
    assert report.publishable is False
    assert any("specialized_ratio" in reason for reason in report.failure_reasons)


def test_p3_a_fallback_regression_blocks_effective_gate_not_variant_credit():
    measurements = [row(candidate=5.0, baseline=10.0), row("fp16/hole/benchmark", provider="fallback", candidate=50.0, baseline=10.0)]
    report = PerformanceReport.evaluate(measurements)
    assert report.specialized_hit_speedup is None
    assert report.effective_end_to_end_speedup is None
    assert report.publishable is False
    assert all("specialized_ratio" not in reason for reason in report.failure_reasons)


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"timing_uncertain": True}, "timing_uncertain"),
        ({"unknown": True}, "unknown"),
        ({"correctness_ok": False}, "correctness_failed"),
        ({"dispatch_ok": False}, "dispatch_failed"),
        ({"baseline": None}, "missing_baseline"),
    ],
)
def test_p3_a_uncertain_unknown_missing_or_failed_evidence_blocks(kwargs, reason):
    report = PerformanceReport.evaluate([row(**kwargs)])
    assert report.publishable is False
    assert any(item.endswith(reason) for item in report.failure_reasons)


def test_p3_a_fallback_never_enters_specialized_hit_speedup():
    report = PerformanceReport.evaluate([
        row(candidate=5.0, baseline=10.0),
        row("fp16/hole/benchmark", provider="fallback", candidate=1.0, baseline=10.0),
    ])
    assert report.specialized_hit_speedup == 2.0
    assert report.effective_end_to_end_speedup == pytest.approx(1 / 0.3)
    assert report.fallback_count_fraction == 0.5


@pytest.mark.parametrize("bad", [True, 0, -1, float("nan"), float("inf"), "10"])
def test_p3_a_invalid_measurements_are_rejected(bad):
    with pytest.raises(TaskContractError, match="finite positive"):
        row(candidate=bad)


def test_p3_a_golden_only_and_empty_measurements_are_not_performance_evidence():
    with pytest.raises(TaskContractError, match="golden-only"):
        PerformanceReport.evaluate([row("fp16/threshold/golden")])
    with pytest.raises(TaskContractError, match="at least one measurement"):
        PerformanceReport.evaluate([])


def test_p3_a_effective_book_uses_total_work_not_median_of_ratios():
    report = PerformanceReport.evaluate([
        row(candidate=9.0, baseline=10.0),
        row("fp16/tail/benchmark", candidate=18.0, baseline=20.0),
        row("fp16/hole/benchmark", candidate=1100.0, baseline=1000.0),
    ])
    assert report.specialized_hit_latency_ratio == 0.9
    assert report.effective_end_to_end_latency_ratio == 1127 / 1030
    assert report.specialized_hit_speedup > 1.0
    assert report.effective_end_to_end_speedup < 1.0
    assert report.publishable is True  # 最大 shard 恰好 1.10，未越界；两本账方向不同。


def test_p3_a_unknown_usage_and_duplicate_instances_are_refused():
    with pytest.raises(TaskContractError, match="unknown measurement usage"):
        row("fp16/threshold/perf")
    with pytest.raises(TaskContractError, match="duplicate performance measurement"):
        PerformanceReport.evaluate([row(), row()])


def test_p3_a_missing_expected_shards_and_no_variant_cannot_publish():
    report = PerformanceReport.evaluate(
        [row("fp16/hole/benchmark", provider="fallback", candidate=1.0, baseline=10.0)],
        expected_benchmark_keys={"fp16/threshold/benchmark"},
    )
    assert report.publishable is False
    assert any("missing_shards" in reason for reason in report.failure_reasons)
    assert any("no_specialized_variant" in reason for reason in report.failure_reasons)
    assert report.specialized_hit_speedup is None
    assert report.effective_end_to_end_speedup is None
