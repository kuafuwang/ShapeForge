"""Whether one timed self-check would replace the seed, by the search's own rule.

The line is for the coding agent. The search measures the file again and does not read this sentence.
"""

from __future__ import annotations

import math

from .bench import speedup_se_of_report


def _speedup(report) -> float | None:
    if report is None:
        return None
    value = report.get("speedup") if isinstance(report, dict) else getattr(report, "speedup", None)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def replacement_line(candidate, seed: dict, *, min_improvement: float, noise_z: float) -> str:
    """One sentence. A missing time says the check cannot judge."""
    cand = _speedup(candidate)
    old = _speedup(seed)
    if cand is None or old is None:
        return "这次没有计时，不能判断够不够换种子。自测不算数。"
    gain = max(float(min_improvement), 1.0 + float(noise_z) * math.hypot(speedup_se_of_report(candidate), speedup_se_of_report(seed)))
    if cand > old * gain:
        return "这一版按外层规则够换种子。自测不算数。"
    return "这一版按外层规则不够换种子。自测不算数。"

def _bench_gpu_us(value) -> float | None:
    bench = value.get("benchmark") if isinstance(value, dict) else getattr(value, "benchmark", None)
    if not isinstance(bench, dict):
        return None
    gpu = bench.get("candidate_gpu_us_median")
    if isinstance(gpu, (int, float)) and not isinstance(gpu, bool) and gpu > 0:
        return float(gpu)
    return None


def gpu_timestamp_line(candidate, seed) -> str:
    """Compare candidate and seed GPU-timestamp medians when both reports have them."""
    cand = _bench_gpu_us(candidate)
    old = _bench_gpu_us(seed)
    if cand is None or old is None:
        return ""
    return f"GPU-timestamp speedup vs seed: {old / cand:.3f}x (seed {old:.2f} us, candidate {cand:.2f} us)"
