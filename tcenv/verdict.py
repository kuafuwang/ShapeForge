"""Whether one timed self-check would replace the seed, by the search's own rule.

The line is for the coding agent. The search measures the file again and does not read this sentence.
"""

from __future__ import annotations

import math

from .bench import speedup_se_of_report


def _speedup(report) -> float | None:
    """@brief 从报告里取出正的加速比数值。

    @param report 评测报告对象或字典，可为 None。
    @return 大于 0 的浮点加速比；缺失或非法时为 None。
    """
    if report is None:
        return None
    value = report.get("speedup") if isinstance(report, dict) else getattr(report, "speedup", None)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def replacement_line(candidate, seed: dict, *, min_improvement: float, noise_z: float) -> str:
    """@brief 用外层判据给出一句话结论：这一版是否够换种子。

    One sentence. A missing time says the check cannot judge.

    @param candidate 候选的评测报告（对象或字典）。
    @param seed 当前种子的评测报告字典。
    @param min_improvement 外层要求的最小相对提升。
    @param noise_z 噪声倍数，与两侧标准误一起抬高门槛。
    @return 一行中文结论；任一侧没有计时则说明无法判断。
    """
    cand = _speedup(candidate)
    old = _speedup(seed)
    if cand is None or old is None:
        return "这次没有计时，不能判断够不够换种子。自测不算数。"
    gain = max(float(min_improvement), 1.0 + float(noise_z) * math.hypot(speedup_se_of_report(candidate), speedup_se_of_report(seed)))
    if cand > old * gain:
        return "这一版按外层规则够换种子。自测不算数。"
    return "这一版按外层规则不够换种子。自测不算数。"

def _bench_gpu_us(value) -> float | None:
    """@brief 取出报告 benchmark 段里的候选 GPU 时间戳中位数。

    @param value 评测报告对象或字典。
    @return 正的微秒数；缺失或非法时为 None。
    """
    bench = value.get("benchmark") if isinstance(value, dict) else getattr(value, "benchmark", None)
    if not isinstance(bench, dict):
        return None
    gpu = bench.get("candidate_gpu_us_median")
    if isinstance(gpu, (int, float)) and not isinstance(gpu, bool) and gpu > 0:
        return float(gpu)
    return None


def gpu_timestamp_line(candidate, seed) -> str:
    """@brief 比较候选与种子的 GPU 时间戳中位数。

    Compare candidate and seed GPU-timestamp medians when both reports have them.

    @param candidate 候选的评测报告（对象或字典）。
    @param seed 种子的评测报告（对象或字典）。
    @return 一行英文加速比说明；任一侧缺少时间戳时返回空串。
    """
    cand = _bench_gpu_us(candidate)
    old = _bench_gpu_us(seed)
    if cand is None or old is None:
        return ""
    return f"GPU-timestamp speedup vs seed: {old / cand:.3f}x (seed {old:.2f} us, candidate {cand:.2f} us)"
