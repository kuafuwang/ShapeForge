"""Approximate speedup error and roofline wording. No GPU, and no change to the stored timing schema."""

from __future__ import annotations

import math

from tcenv.bench import QUIET_CV, approximate_speedup_se, speedup_se_of_report, timing_is_uncertain
from taic.feedback_extra import at_roofline, target_note
from taic.orchestrator import speedup_se
from tcenv.evaluate import EvalReport


def test_the_standard_error_matches_an_independent_recomputation_and_is_not_a_confidence_interval():
    se = approximate_speedup_se(0.02, 16, 0.03, 16)
    expect = math.sqrt(math.pi / 2) * math.hypot(0.02, 0.03) / math.sqrt(16)
    assert se == expect
    assert "not a confidence interval" in approximate_speedup_se.__doc__.lower()
    assert "this is a 95%" not in approximate_speedup_se.__doc__.lower()
    report = EvalReport(task="relu", dtype="fp16", size="small", mode="full", status="ok")
    report.benchmark = {
        "candidate": {"cv": 0.02, "reps": 16},
        "baselines": {"mlx": {"cv": 0.03, "reps": 16}},
        "best_baseline": "mlx",
        "noisy": False,
    }
    assert speedup_se(report) == expect
    assert speedup_se_of_report(report) == expect
    assert speedup_se(None) == 0.0


def test_a_noisy_or_high_cv_measurement_is_labelled_uncertain():
    quiet = {"benchmark": {"noisy": False, "candidate": {"cv": 0.01, "reps": 15}, "baselines": {"mlx": {"cv": 0.01, "reps": 15}}, "best_baseline": "mlx"}}
    assert timing_is_uncertain(quiet) is False
    assert timing_is_uncertain({"benchmark": {"noisy": True, "candidate": {"cv": 0.01, "reps": 15}, "baselines": {"mlx": {"cv": 0.01, "reps": 15}}, "best_baseline": "mlx"}}) is True
    loud = {"benchmark": {"noisy": False, "candidate": {"cv": QUIET_CV + 0.01, "reps": 15}, "baselines": {"mlx": {"cv": 0.01, "reps": 15}}, "best_baseline": "mlx"}}
    assert timing_is_uncertain(loud) is True


def test_a_baseline_under_the_bound_is_not_called_a_physical_optimum_or_winograd():
    beaten = {
        "roofline": {"lower_bound_us": 931.8, "time_over_lower_bound": 1.07, "peak_gbps_measured": 400.0},
        "baseline_us": {"torch_mps": 777.0},
        "benchmark": {"noisy": False},
    }
    note = target_note(beaten)
    assert note and "777.0 us" in note and "Winograd" not in note
    assert "physical" not in note.lower()
    assert "operation-count estimate" in note and "peak calibration" in note
    assert at_roofline(beaten, 1.02) is False
    quiet = {
        "roofline": {"lower_bound_us": 100, "time_over_lower_bound": 1.01, "peak_gbps_measured": 400.0},
        "baseline_us": {"mlx": 120},
        "benchmark": {"noisy": False},
    }
    assert at_roofline(quiet, 1.02) is True
    assert at_roofline({**quiet, "benchmark": {"noisy": True}}, 1.02) is False
    missing_peak = {"roofline": {"lower_bound_us": 100, "time_over_lower_bound": 1.01}, "baseline_us": {"mlx": 120}, "benchmark": {"noisy": False}}
    assert at_roofline(missing_peak, 1.02) is False


def test_a_corrupt_timing_block_does_not_raise_and_an_incomplete_benchmark_is_uncertain():
    from tcenv.bench import speedup_se_of_report, timing_is_uncertain
    assert speedup_se_of_report({"benchmark": {"candidate": 5, "baselines": {"mlx": {"cv": 0.01, "reps": 3}}, "best_baseline": "mlx"}}) == 0.0
    assert timing_is_uncertain({"benchmark": {"noisy": False}}) is True
    from experiments.common import summarize

    raw = {
        "task": "relu", "dtype": "fp16", "model": "fake", "stop_reason": "max_rounds", "rounds": 1,
        "evaluations": 1, "spent_usd": 0.1, "budget_usd": 1.0, "best_speedup": 1.8, "confirmed_speedup": 1.1,
        "naive_speedup": 2.0,
    }
    row = summarize(raw, None)
    assert row["published_speedup"] is None
    assert row["best_speedup"] == 1.8 and row["naive_speedup"] is None  # naive is a separate argument, not copied from the result
    assert row["publication"] == "旧版未绑定源码"
