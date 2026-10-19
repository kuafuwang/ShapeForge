"""Benchmark protocol (paper Section 3.2: "fixed benchmarking protocol").

* ``k`` back-to-back launches are timed as one batch and divided by ``k`` (stream-order serial
  dispatch for candidates), so launch/sync latency is amortized.
* ``k`` is calibrated per runner so one batch lasts about ``target_s``.
* Inputs rotate over several buffer sets so that memory-bound kernels do not run out of cache.
* Candidate and baselines are measured *interleaved* (A B A B ...), which cancels thermal and
  background-load drift; the reported statistic is the median over repetitions plus the
  coefficient of variation.
"""

# 计时协议所在。候选与 MLX / PyTorch 放进同一次 interleaved，比的是同一把 perf_counter 墙钟。
# 候选的窗口是 commit 到 waitUntilCompleted 的 wall_s（编码在窗口外）；MLX 的窗口是 mx.eval（构图在窗口外）；PyTorch 的窗口含启动循环和 mps.synchronize。
# 两边都是「这一批的墙钟 ÷ k」，再取重复的中位数。加速比在 evaluate 里用这份 median_s 计算：基线耗时 ÷ 候选耗时，大于 1 表示候选更快。
# roofline 拿走同一份候选 median_s，对照标定过的峰值带宽和算力：最小字节、FLOP 各除以峰值得到两条时间下界，较大者是硬件下界，
# 用来说明时间卡在访存还是计算、离下界有多远。本文件自己不加锁；评测、基线命令和 triton-msl 在独占锁里调用它。

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, asdict
from typing import Callable


# 对外认 median_s。加速比和 roofline 都用它；cv 留给调用方判断这次测量是否安静。
@dataclass
class BenchStats:
    median_s: float
    mean_s: float
    min_s: float
    cv: float  # stdev / mean across repetitions
    reps: int
    iters: int
    samples_s: tuple[float, ...] = ()  # one wall-clock sample per repetition, same order as the interleaved loop

    @property
    def median_us(self) -> float:
        return self.median_s * 1e6

    def to_json(self) -> dict:
        d = asdict(self)
        d["median_us"] = self.median_us
        if not d["samples_s"]:
            d.pop("samples_s")
        return d


# 一批的墙钟大约 target_s，把单次启动和同步的固定开销摊进 k。
# 从 1 次启动探起：evaluate 对单次 launch 有数秒级看门狗，一上来就跑满 kmax 会把病态核放大成挂起。
def choose_iters(run_k: Callable[[int], float], target_s: float = 0.015, kmin: int = 4, kmax: int = 4000) -> int:
    """Pick the batch size ``k`` so that one timed batch lasts about ``target_s``.

    The probe starts with a single launch so that a pathologically slow candidate is never asked to run a long
    batch (every GPU launch is bounded by an external watchdog; see tcenv.evaluate).
    """
    run_k(1)  # warm-up, discarded: pays one-time costs (pipeline creation, JIT compilation of a compiled baseline)
    per = max(run_k(1), 1e-8)
    # 单次已经超过 0.1 秒时一批只启动 1 次，避免 k 把单次 launch 的看门狗打满。
    if per > 0.1:  # slow: one launch per batch
        return 1
    # 再用最多 8 次、大约 0.05 秒的一小批，把「每次多少秒」从单次估计修正成一小批的平均。
    probe = max(1, min(8, int(0.05 / per)))
    per = max(run_k(probe) / probe, 1e-8)
    return int(max(kmin, min(kmax, target_s / per)))


# 每一轮打乱顺序再各测一批。热和后台负载会同时作用在候选和基线上，中位数之比才稳得住。
# 返回的 median_s 同时是加速比的分子分母，和 roofline 的 time_s。kmin 让每一批至少转过全部缓冲区组。
def interleaved(
    runners: dict[str, Callable[[int], float]],
    *,
    reps: int = 15,
    target_s: float = 0.015,
    warmup_reps: int = 2,
    seed: int = 0,
    kmin: int = 4,
) -> dict[str, BenchStats]:
    """``kmin``: smallest batch size. Callers that rotate over R buffer sets pass ``kmin=R`` so that every batch touches all of them."""
    rng = random.Random(seed)
    ks = {n: choose_iters(r, target_s, kmin) for n, r in runners.items()}
    samples: dict[str, list[float]] = {n: [] for n in runners}
    names = list(runners)
    for rep in range(reps + warmup_reps):
        order = names[:]
        rng.shuffle(order)
        for n in order:
            t = runners[n](ks[n]) / ks[n]
            # 开头几轮不进样本。机器刚从闲置进入计时，这几轮放进中位数会把加速比拉偏。
            if rep >= warmup_reps:
                samples[n].append(t)
    out: dict[str, BenchStats] = {}
    for n, xs in samples.items():
        mean = statistics.fmean(xs)
        out[n] = BenchStats(
            median_s=statistics.median(xs),
            mean_s=mean,
            min_s=min(xs),
            cv=(statistics.pstdev(xs) / mean) if mean > 0 else 0.0,
            reps=len(xs),
            iters=ks[n],
            samples_s=tuple(xs),
        )
    return out


# evaluate.NOISY_CV uses this same cutoff when it marks benchmark.noisy. Kept here so a report can be labelled without importing the evaluator.
QUIET_CV = 0.05


def approximate_speedup_se(candidate_cv, candidate_reps, baseline_cv, baseline_reps) -> float:
    """Approximate relative standard error of a ratio of two medians.

    SE of a median is about sqrt(pi/2) * sigma / sqrt(n). The candidate and baseline errors are combined as if independent.
    The inputs are the cvs and repetition counts already on the report. This is not a confidence interval, and it is not a 95% interval.
    """
    try:
        n = max(1, min(int(candidate_reps), int(baseline_reps)))
        return math.sqrt(math.pi / 2) * math.hypot(float(candidate_cv), float(baseline_cv)) / math.sqrt(n)
    except (TypeError, ValueError):
        return 0.0


def _benchmark(report) -> dict:
    if report is None:
        return {}
    if isinstance(report, dict):
        bench = report.get("benchmark") or {}
        return bench if isinstance(bench, dict) else {}
    bench = getattr(report, "benchmark", None) or {}
    return bench if isinstance(bench, dict) else {}


def speedup_se_of_report(report) -> float:
    """The approximate relative standard error stored by a report, or 0 when the report has no timing statistics."""
    bench = _benchmark(report)
    try:
        candidate = bench["candidate"]
        baseline = bench["baselines"][bench["best_baseline"]]
        if not isinstance(candidate, dict) or not isinstance(baseline, dict):
            return 0.0
        return approximate_speedup_se(candidate.get("cv"), candidate.get("reps"), baseline.get("cv"), baseline.get("reps"))
    except (KeyError, TypeError, AttributeError, ValueError):
        return 0.0


def paired_bootstrap_speedup_ci(cand_samples, base_samples, *, n_boot: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    """2.5 and 97.5 percentiles of (baseline sample / candidate sample) over paired draws.

    The two sequences are the interleaved repetitions already stored on a report. This is a resampling interval, not a proof.
    Returns None when either side has fewer than two samples.
    """
    import random

    cand = [float(x) for x in cand_samples]
    base = [float(x) for x in base_samples]
    n = min(len(cand), len(base))
    if n < 2:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        picks = [rng.randrange(n) for _ in range(n)]
        c = statistics.median(cand[i] for i in picks)
        b = statistics.median(base[i] for i in picks)
        if c > 0:
            draws.append(b / c)
    if len(draws) < 2:
        return None
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return lo, hi


def speedup_ci_of_report(report) -> tuple[float, float] | None:
    """Paired bootstrap interval when the report stored per-repetition samples for the candidate and the best baseline."""
    bench = _benchmark(report)
    try:
        candidate = bench["candidate"]
        baseline = bench["baselines"][bench["best_baseline"]]
        return paired_bootstrap_speedup_ci(candidate.get("samples_s") or (), baseline.get("samples_s") or ())
    except (KeyError, TypeError, AttributeError, ValueError):
        return None


def timing_is_uncertain(report) -> bool:
    """True when the kept measurement is marked noisy or a reported cv is above the quiet cutoff."""
    bench = _benchmark(report)
    if not bench:
        return False
    if bench.get("noisy") is True:
        return True
    candidate = bench.get("candidate")
    if not isinstance(candidate, dict) or "cv" not in candidate:
        # 有计时块却没有可用的 cv，不能把它当成安静的测量。
        return True
    try:
        cvs = [float(candidate["cv"])]
        bases = bench.get("baselines") or {}
        name = bench.get("best_baseline")
        base = bases.get(name) if isinstance(bases, dict) else None
        if isinstance(base, dict) and "cv" in base:
            cvs.append(float(base["cv"]))
    except (KeyError, TypeError, ValueError):
        return True
    return any(cv > QUIET_CV for cv in cvs)
