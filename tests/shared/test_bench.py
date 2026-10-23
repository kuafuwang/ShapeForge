"""The batch-size probe and the interleaved timing loop of ``tcenv.bench``. No GPU."""

from __future__ import annotations

import pytest

from tcenv.bench import choose_iters, interleaved


def test_choose_iters_ignores_a_one_time_warmup_cost():
    """A compiled baseline's first call includes JIT compilation; that must not be mistaken for the time per launch."""
    calls = {"n": 0}

    def run_k(k: int) -> float:
        calls["n"] += 1
        return (1.0 if calls["n"] == 1 else 100e-6) * k  # first call: 1 s of compilation, afterwards 100 us per launch

    assert choose_iters(run_k, target_s=0.015, kmin=4) == 150


def test_choose_iters_still_uses_single_launches_for_slow_kernels():
    calls = []

    def run_k(k: int) -> float:
        calls.append(k)
        return 0.3 * k

    assert choose_iters(run_k) == 4 and max(calls) == 1  # kmin buffer sets, never a long probe batch


def test_interleaved_reports_every_runner():
    st = interleaved({"a": lambda k: 1e-3 * k, "b": lambda k: 2e-3 * k}, reps=4, warmup_reps=1)
    assert st["a"].median_s == pytest.approx(1e-3) and st["b"].median_s == pytest.approx(2e-3)
