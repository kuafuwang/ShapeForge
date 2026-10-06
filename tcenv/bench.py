"""Benchmark protocol (paper Section 3.2: "fixed benchmarking protocol").

* ``k`` back-to-back launches are timed as one batch and divided by ``k`` (stream-order serial
  dispatch for candidates), so launch/sync latency is amortized.
* ``k`` is calibrated per runner so one batch lasts about ``target_s``.
* Inputs rotate over several buffer sets so that memory-bound kernels do not run out of cache.
* Candidate and baselines are measured *interleaved* (A B A B ...), which cancels thermal and
  background-load drift; the reported statistic is the median over repetitions plus the
  coefficient of variation.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, asdict
from typing import Callable


@dataclass
class BenchStats:
    median_s: float
    mean_s: float
    min_s: float
    cv: float  # stdev / mean across repetitions
    reps: int
    iters: int

    @property
    def median_us(self) -> float:
        return self.median_s * 1e6

    def to_json(self) -> dict:
        d = asdict(self)
        d["median_us"] = self.median_us
        return d


def choose_iters(run_k: Callable[[int], float], target_s: float = 0.015, kmin: int = 4, kmax: int = 4000) -> int:
    """Pick the batch size ``k`` so that one timed batch lasts about ``target_s``.

    The probe starts with a single launch so that a pathologically slow candidate is never asked to run a long
    batch (every GPU launch is bounded by an external watchdog; see tcenv.evaluate).
    """
    run_k(1)  # warm-up, discarded: pays one-time costs (pipeline creation, JIT compilation of a compiled baseline)
    per = max(run_k(1), 1e-8)
    if per > 0.1:  # slow: one launch per batch
        return 1
    probe = max(1, min(8, int(0.05 / per)))
    per = max(run_k(probe) / probe, 1e-8)
    return int(max(kmin, min(kmax, target_s / per)))


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
        )
    return out
