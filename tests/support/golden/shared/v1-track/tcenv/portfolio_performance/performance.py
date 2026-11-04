"""P3-A: per-shard latency ratios and the 1.10 regression gate.

``latency_ratio = candidate_us / baseline_us`` (lower is faster); ``speedup = 1 / latency_ratio``
(the repository's usual baseline/candidate convention).  The effective book uses total measured
work (sum/sum).  The specialized book is the median of per-shard ratios.  Geomeans are descriptive
only, never the gate.  These models judge measurements; they do not prove freshness.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..contract_errors import TaskContractError

MAX_REGRESSION_RATIO = 1.10
BENCHMARK_USAGE = "benchmark"
USAGES = frozenset({"benchmark", "golden"})


def _ratio(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskContractError(f"{where} must be a finite positive number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise TaskContractError(f"{where} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise TaskContractError(f"{where} must be a finite positive number")
    return result


def _median(values: Iterable[float], *, where: str) -> float:
    items = list(values)
    if not items:
        raise TaskContractError(f"{where} needs at least one benchmark measurement")
    return float(statistics.median(items))


def _geomean(values: Iterable[float], *, where: str) -> float:
    items = list(values)
    if not items:
        raise TaskContractError(f"{where} needs at least one ratio")
    return math.exp(statistics.fmean(math.log(value) for value in items))


@dataclass(frozen=True)
class ShardMeasurement:
    """@brief 一个 instance/provider 的实测输入（时间必须是已完成的测量，不是 speedup 自述）。"""

    instance_key: str
    usage: str
    provider: str
    candidate_us: float
    baseline_us: float | None
    timing_uncertain: bool = False
    correctness_ok: bool = True
    dispatch_ok: bool = True
    unknown: bool = False

    def __post_init__(self) -> None:
        parts = self.instance_key.split("/") if isinstance(self.instance_key, str) else []
        if len(parts) != 3 or self.usage != parts[2]:
            raise TaskContractError("measurement usage is not the instance_key's usage")
        if self.usage not in USAGES:
            raise TaskContractError(f"unknown measurement usage {self.usage!r}")
        if self.provider not in ("candidate", "fallback"):
            raise TaskContractError("measurement provider must be candidate or fallback")
        _ratio(self.candidate_us, where=f"{self.instance_key}.candidate_us")
        if self.baseline_us is not None:
            _ratio(self.baseline_us, where=f"{self.instance_key}.baseline_us")
        for name in ("timing_uncertain", "correctness_ok", "dispatch_ok", "unknown"):
            if type(getattr(self, name)) is not bool:
                raise TaskContractError(f"{self.instance_key}.{name} must be bool")

    @property
    def benchmark(self) -> bool:
        return self.usage == BENCHMARK_USAGE

    @property
    def baseline_missing(self) -> bool:
        return self.baseline_us is None

    @property
    def latency_ratio(self) -> float | None:
        if self.baseline_missing:
            return None
        return float(self.candidate_us) / float(self.baseline_us)

    @property
    def specialized_latency_ratio(self) -> float | None:
        if self.provider != "candidate" or self.baseline_missing:
            return None
        return self.latency_ratio


@dataclass(frozen=True)
class ShardVerdict:
    measurement: ShardMeasurement
    status: str
    reason: str | None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class PerformanceReport:
    """@brief 逐 shard 判决与两本账（聚合值只作描述，不作唯一门）。"""

    verdicts: tuple[ShardVerdict, ...]
    specialized_hit_speedup: float | None
    effective_end_to_end_speedup: float | None
    specialized_hit_latency_ratio: float | None
    effective_end_to_end_latency_ratio: float | None
    specialized_geomean_latency_ratio: float | None
    effective_geomean_latency_ratio: float | None
    fallback_time_fraction: float | None
    fallback_count_fraction: float | None
    publishable: bool
    failure_reasons: tuple[str, ...]

    @classmethod
    def evaluate(
        cls,
        measurements: Iterable[ShardMeasurement],
        *,
        expected_benchmark_keys: Iterable[str] | None = None,
    ) -> "PerformanceReport":
        rows = tuple(measurements)
        if not rows:
            raise TaskContractError("a performance report needs at least one measurement")
        keys = [row.instance_key for row in rows]
        if len(set(keys)) != len(keys):
            raise TaskContractError("duplicate performance measurement instance_key")
        benchmark = [row for row in rows if row.benchmark]
        if not benchmark:
            raise TaskContractError("golden-only measurements cannot enter performance comparison")
        failures: list[str] = []
        if expected_benchmark_keys is not None:
            expected = set(expected_benchmark_keys)
            actual = {row.instance_key for row in benchmark}
            if expected - actual:
                failures.append(f"missing_shards:{sorted(expected - actual)}")
            if actual - expected:
                failures.append(f"unexpected_shards:{sorted(actual - expected)}")
        verdicts: list[ShardVerdict] = []
        for row in rows:
            if not row.benchmark:
                verdicts.append(ShardVerdict(row, "ignored_golden", None))
                continue
            reason = None
            if row.unknown:
                reason = "unknown"
            elif row.timing_uncertain:
                reason = "timing_uncertain"
            elif not row.correctness_ok:
                reason = "correctness_failed"
            elif not row.dispatch_ok:
                reason = "dispatch_failed"
            elif row.baseline_missing:
                reason = "missing_baseline"
            if reason is not None:
                failures.append(f"{row.instance_key}:{reason}")
            verdicts.append(ShardVerdict(row, "ok" if reason is None else "rejected", reason))
        for row in benchmark:
            if row.latency_ratio is not None and row.latency_ratio > MAX_REGRESSION_RATIO:
                failures.append(f"{row.instance_key}:effective_ratio>{MAX_REGRESSION_RATIO:.2f}")
            if row.specialized_latency_ratio is not None and row.specialized_latency_ratio > MAX_REGRESSION_RATIO:
                failures.append(f"{row.instance_key}:specialized_ratio>{MAX_REGRESSION_RATIO:.2f}")
        specialized = [row for row in benchmark if row.provider == "candidate" and not row.baseline_missing]
        if not specialized:
            failures.append("no_specialized_variant")
        if failures:
            return cls(
                verdicts=tuple(verdicts),
                specialized_hit_speedup=None,
                effective_end_to_end_speedup=None,
                specialized_hit_latency_ratio=None,
                effective_end_to_end_latency_ratio=None,
                specialized_geomean_latency_ratio=None,
                effective_geomean_latency_ratio=None,
                fallback_time_fraction=None,
                fallback_count_fraction=None,
                publishable=False,
                failure_reasons=tuple(failures),
            )
        specialized_hit_ratio = _median(
            (row.specialized_latency_ratio for row in specialized),
            where="specialized per-shard latency ratios",
        )
        effective_ratio = _ratio(
            sum(row.candidate_us for row in benchmark) / sum(row.baseline_us for row in benchmark),
            where="effective_end_to_end_latency_ratio",
        )
        specialized_geomean = _geomean(
            (row.specialized_latency_ratio for row in specialized),
            where="specialized geomean latency ratio",
        )
        effective_geomean = _geomean(
            (row.latency_ratio for row in benchmark),
            where="effective geomean latency ratio",
        )
        fallback_time = sum(row.candidate_us for row in benchmark if row.provider == "fallback")
        total_time = sum(row.candidate_us for row in benchmark)
        return cls(
            verdicts=tuple(verdicts),
            specialized_hit_speedup=1.0 / specialized_hit_ratio,
            effective_end_to_end_speedup=1.0 / effective_ratio,
            specialized_hit_latency_ratio=specialized_hit_ratio,
            effective_end_to_end_latency_ratio=effective_ratio,
            specialized_geomean_latency_ratio=specialized_geomean,
            effective_geomean_latency_ratio=effective_geomean,
            fallback_time_fraction=fallback_time / total_time,
            fallback_count_fraction=sum(row.provider == "fallback" for row in benchmark) / len(benchmark),
            publishable=True,
            failure_reasons=(),
        )


__all__ = [
    "MAX_REGRESSION_RATIO",
    "PerformanceReport",
    "ShardMeasurement",
    "ShardVerdict",
]
