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
# 候选的窗口含编码和 commit 到 waitUntilCompleted 的 wall_s。MLX 的窗口是 mx.eval（构图在窗口外）。PyTorch 的窗口含启动循环和 mps.synchronize。
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
    """@brief 一次批量计时的统计结果：中位数、均值、最小值、变异系数与原始样本。"""

    median_s: float
    mean_s: float
    min_s: float
    cv: float  # stdev / mean across repetitions
    reps: int
    iters: int
    samples_s: tuple[float, ...] = ()  # one wall-clock sample per repetition, same order as the interleaved loop

    @property
    def median_us(self) -> float:
        """@brief 以微秒表示的中位数耗时。

        @return ``median_s`` 乘以 1e6。
        """
        return self.median_s * 1e6

    def to_json(self) -> dict:
        """@brief 转成可写入报告的字典。

        @return 字段字典；没有原始样本时去掉 ``samples_s``，并补上 ``median_us``。
        """
        d = asdict(self)
        d["median_us"] = self.median_us
        if not d["samples_s"]:
            d.pop("samples_s")
        return d


# 一批的墙钟大约 target_s，把单次启动和同步的固定开销摊进 k。
# 从 1 次启动探起：evaluate 对单次 launch 有数秒级看门狗，一上来就跑满 kmax 会把病态核放大成挂起。
# 一批 k 次 launch 共用一个心跳 B；父进程在 tcenv.evaluate.LAUNCH_DEADLINE_S 之后 SIGKILL。
# 慢核的批量因此必须有上限，否则一条合法但慢的候选会被判成挂死（max_hangs=1 还会终止整次搜索）。
# 0.8 × LAUNCH_DEADLINE_S：留 20% 余量给编码、提交和心跳写入的延迟。一致性由 tests/test_bench_limit.py 盯着。
BATCH_WALL_BUDGET_S = 1.6


def choose_iters(run_k: Callable[[int], float], target_s: float = 0.015, kmin: int = 4, kmax: int = 4000) -> int:
    """@brief 选出一个计时批量的大小，使一批约持续 ``target_s``。

    Pick the batch size ``k`` so that one timed batch lasts about ``target_s``.

    The probe starts with a single launch so that a pathologically slow candidate is never asked to run a long
    batch (every GPU launch is bounded by an external watchdog; see tcenv.evaluate).

    @param run_k 执行 ``k`` 次启动并返回批次墙钟秒数的可调用对象。
    @param target_s 期望的单批持续秒数。
    @param kmin 批量下限。
    @param kmax 批量上限。
    @return 选定的批量大小，已被看门狗预算封顶。
    """
    run_k(1)  # warm-up, discarded: pays one-time costs (pipeline creation, JIT compilation of a compiled baseline)
    per = max(run_k(1), 1e-8)
    # 单次已经超过 0.1 秒时一批只启动 1 次，避免 k 把单次 launch 的看门狗打满。
    if per > 0.1:
        # 慢核：往 kmax 长大没有意义，但仍要尽量转过每一组缓冲区。上限由看门狗反推：
        # 一批的墙钟 ≈ k × per，必须留在 BATCH_WALL_BUDGET_S 以内。
        return int(max(1, min(kmin, BATCH_WALL_BUDGET_S / per)))
    # 再用最多 8 次、大约 0.05 秒的一小批，把「每次多少秒」从单次估计修正成一小批的平均。
    probe = max(1, min(8, int(0.05 / per)))
    per = max(run_k(probe) / probe, 1e-8)
    k = int(max(kmin, min(kmax, target_s / per)))
    # 两条路径共用同一个上界：一批的墙钟必须留在看门狗以内。代价是 k 可能小于旋转组数——那时这一批只会
    # 转过前 k 组缓冲（run() 每批都从第 0 组开始），比"把一批拖过 2 秒被判挂死"可接受。
    return int(max(1, min(k, BATCH_WALL_BUDGET_S / per)))


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
    """@brief 交错测量多个 runner 并汇总各自的统计量。

    ``kmin``: smallest batch size. Callers that rotate over R buffer sets pass ``kmin=R`` so that every batch touches all of them.

    @param runners 名称到 ``run_k`` 的映射。
    @param reps 计入统计的重复次数。
    @param target_s 每批期望持续秒数。
    @param warmup_reps 不计入样本的预热轮数。
    @param seed 每轮打乱顺序的随机种子。
    @param kmin 批量下限。
    @return 名称到 ``BenchStats`` 的字典。
    """
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
        # 一次重复没有可测的离散度。叫它 0.0 会让噪声门把这次测量当成无限精确；
        # 1.0 把它标成最吵，于是 --eval-reps 1 不能只凭一次采样换种子。
        cv = (statistics.pstdev(xs) / mean) if (mean > 0 and len(xs) > 1) else 1.0
        out[n] = BenchStats(
            median_s=statistics.median(xs),
            mean_s=mean,
            min_s=min(xs),
            cv=cv,
            reps=len(xs),
            iters=ks[n],
            samples_s=tuple(xs),
        )
    return out


# evaluate.NOISY_CV uses this same cutoff when it marks benchmark.noisy. Kept here so a report can be labelled without importing the evaluator.
QUIET_CV = 0.05


def approximate_speedup_se(candidate_cv, candidate_reps, baseline_cv, baseline_reps) -> float:
    """@brief 估计两个中位数之比的近似相对标准误。

    Approximate relative standard error of a ratio of two medians.

    SE of a median is about sqrt(pi/2) * sigma / sqrt(n). The candidate and baseline errors are combined as if independent.
    The inputs are the cvs and repetition counts already on the report. This is not a confidence interval, and it is not a 95% interval.

    @param candidate_cv 候选的变异系数。
    @param candidate_reps 候选的重复次数。
    @param baseline_cv 基线的变异系数。
    @param baseline_reps 基线的重复次数。
    @return 相对标准误；输入无法转换时为 0.0。
    """
    try:
        n = max(1, min(int(candidate_reps), int(baseline_reps)))
        return math.sqrt(math.pi / 2) * math.hypot(float(candidate_cv), float(baseline_cv)) / math.sqrt(n)
    except (TypeError, ValueError):
        return 0.0


def _benchmark(report) -> dict:
    """@brief 从报告对象或字典中取出 benchmark 段。

    @param report 评测报告；可为 None。
    @return benchmark 字典；缺失或类型不对时为空字典。
    """
    if report is None:
        return {}
    if isinstance(report, dict):
        bench = report.get("benchmark") or {}
        return bench if isinstance(bench, dict) else {}
    bench = getattr(report, "benchmark", None) or {}
    return bench if isinstance(bench, dict) else {}


def speedup_se_of_report(report) -> float:
    """@brief 取报告里保存的加速比近似相对标准误。

    The approximate relative standard error stored by a report, or 0 when the report has no timing statistics.

    @param report 评测报告对象或字典。
    @return 相对标准误；缺少计时统计时为 0.0。
    """
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
    """@brief 用成对重采样估计加速比的 2.5% 与 97.5% 分位数。

    2.5 and 97.5 percentiles of (baseline sample / candidate sample) over paired draws.

    The two sequences are the interleaved repetitions already stored on a report. This is a resampling interval, not a proof.
    Returns None when either side has fewer than two samples.

    @param cand_samples 候选每次重复的耗时样本。
    @param base_samples 基线每次重复的耗时样本。
    @param n_boot 重采样次数。
    @param seed 随机种子。
    @return ``(下界, 上界)``；样本不足时为 None。
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
    """@brief 从报告里的原始样本计算成对 bootstrap 加速比区间。

    Paired bootstrap interval when the report stored per-repetition samples for the candidate and the best baseline.

    @param report 评测报告对象或字典。
    @return ``(下界, 上界)``；缺少样本时为 None。
    """
    bench = _benchmark(report)
    try:
        candidate = bench["candidate"]
        baseline = bench["baselines"][bench["best_baseline"]]
        return paired_bootstrap_speedup_ci(candidate.get("samples_s") or (), baseline.get("samples_s") or ())
    except (KeyError, TypeError, AttributeError, ValueError):
        return None


def timing_is_uncertain(report) -> bool:
    """@brief 判断保留的这次测量是否不够安静、可信。

    True when the kept measurement is marked noisy or a reported cv is above the quiet cutoff.

    @param report 评测报告对象或字典。
    @return 缺少计时块、cv 缺失、重复次数不足或 cv 超过 ``QUIET_CV`` 时为 True。
    """
    bench = _benchmark(report)
    if not bench:
        # 没有计时块就无法说明这次测量安静。确认报告缺了这一块时，不能把加速比当成已经复核过。
        return True
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
        reps = [candidate.get("reps")]
        if isinstance(base, dict):
            reps.append(base.get("reps"))
        for count in reps:
            if isinstance(count, int) and not isinstance(count, bool) and count < 2:
                return True
    except (KeyError, TypeError, ValueError):
        return True
    return any(cv > QUIET_CV for cv in cvs)
