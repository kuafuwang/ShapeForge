"""The measured-number header stays in front of a truncated feedback. No GPU and no API key."""

from tcenv.evaluate import EvalReport
from tcenv.report import format_feedback

FEEDBACK_CHARS = 6000


def _report(**kw) -> EvalReport:
    base = dict(task="conv2d", dtype="fp16", size="paper", mode="full", status="ok", stage="done", message="ok")
    base.update(kw)
    return EvalReport(**base)


def _roof(**kw) -> dict:
    roof = {
        "bound": "compute",
        "compute_fraction": 0.4,
        "bandwidth_fraction": 0.2,
        "lower_bound_us": 931.8,
        "time_over_lower_bound": 1.061,
        "flops": 1.0,
    }
    roof.update(kw)
    return roof


def test_a_long_correctness_section_cannot_cut_off_the_measured_header():
    cases = [
        {"ok": False, "case": f"case{i}", "scaled_err": 2.0, "allow": 1.0, "message": "x" * 400}
        for i in range(40)
    ]
    rep = _report(
        correctness={"cases": cases, "tolerance": {"rtol": 1e-3, "atol_rel": 1e-3}},
        roofline=_roof(),
        benchmark={"candidate": {"median_us": 988.3, "cv": 0.012}},
        pipeline={
            "static_threadgroup_bytes": 12288,
            "max_threadgroup_memory_bytes": 32768,
            "max_total_threads_per_threadgroup": 1024,
            "launch_threads_per_threadgroup": 128,
        },
        baseline_us={"torch_mps": 2000.0},
    )
    cut = format_feedback(rep)[:FEEDBACK_CHARS]
    assert cut.startswith("STATUS: ok")
    assert "bound: compute" in cut
    assert "1.06x the lower bound" in cut
    assert "compute utilization: 40%" in cut
    assert "timing cv: 1.2%" in cut
    assert "threadgroup memory: 12288 B / 32768 B limit" in cut
    assert "threadgroup thread cap 1024, this launch 128" in cut
    assert "maxTotalThreadsPerThreadgroup" not in cut.split("Correctness", 1)[0]


def test_a_baseline_faster_than_the_bound_is_not_reported_as_near_the_peak():
    rep = _report(
        roofline=_roof(),
        baseline_us={"torch_mps": 773.5},
        candidate_us=988.3,
    )
    text = format_feedback(rep)
    assert "1.06x the lower bound" not in text
    assert "x above it" not in text
    assert "lower bound 931.8 us is not the target: fastest baseline (torch_mps) is already 773.5 us" in text
    assert "not a reason to switch algorithm" not in text
    assert "including ieee" in text


def test_a_missing_gap_omits_the_times_line_and_a_missing_roofline_omits_the_head():
    no_gap = format_feedback(_report(roofline=_roof(time_over_lower_bound=None), baseline_us={"mlx": 2000.0}))
    assert "bound: compute" in no_gap
    assert "the lower bound" not in no_gap
    bare = format_feedback(_report(roofline={}, pipeline={}, benchmark={}))
    assert "Measured:" not in bare


def test_the_performance_block_prints_a_gpu_timestamp_speedup_against_the_seed():
    rep = _report(
        benchmark={
            "candidate": {"median_us": 360.17, "cv": 0.02},
            "candidate_gpu_us_median": 342.94,
            "seed_gpu_us_median": 300.03,
        },
    )
    text = format_feedback(rep)
    assert "GPU-timestamp speedup vs seed: 0.875x" in text
    assert "seed 300.03 us" in text
