"""The measured-number header stays in front of a truncated feedback. No GPU and no API key."""

from tcenv.evaluate import EvalReport
from tcenv.report import format_feedback

FEEDBACK_CHARS = 6000


def _report(**kw) -> EvalReport:
    """@brief 以一组合法默认字段构造 ``EvalReport``，供各用例按需覆盖。

    @param kw 要覆盖的 EvalReport 字段（如 roofline、benchmark、baseline_us）。
    @return 构造好的评测报告对象。
    """
    base = dict(task="conv2d", dtype="fp16", size="paper", mode="full", status="ok", stage="done", message="ok")
    base.update(kw)
    return EvalReport(**base)


def _roof(**kw) -> dict:
    """@brief 构造一份 roofline 字典（compute bound，含上下界与利用率字段），供各用例按需覆盖。

    @param kw 要覆盖的 roofline 字段（如 ``time_over_lower_bound=None``）。
    @return 构造好的 roofline 字典。
    """
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
    """@brief 验证正确性段很长时，前 6000 字符仍以 STATUS 与实测数字头部开头，不会被裁掉。

    用例构造 40 条带长消息的失败 case，确认截断后的文本包含状态、roofline 倍数与利用率、
    timing cv、threadgroup 内存与线程上限，且正确性段之前不出现私有的 maxTotalThreadsPerThreadgroup 名字。
    """
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
    """@brief 验证已有 baseline 快于理论下界时，反馈不把它当成贴近峰值，而是改以该 baseline 为目标。

    断言文本不再出现下界倍数与“above it”，而是说明下界不是目标（torch_mps 已 773.5 us），
    并保留“包括 ieee”这一提示。
    """
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
    """@brief 验证缺 gap 值时不输出倍数行，缺 roofline 时不输出 ``Measured:`` 头部。

    两半分别覆盖 ``time_over_lower_bound=None`` 与完全空的 roofline/pipeline/benchmark。
    """
    no_gap = format_feedback(_report(roofline=_roof(time_over_lower_bound=None), baseline_us={"mlx": 2000.0}))
    assert "bound: compute" in no_gap
    assert "the lower bound" not in no_gap
    bare = format_feedback(_report(roofline={}, pipeline={}, benchmark={}))
    assert "Measured:" not in bare


def test_the_performance_block_prints_a_gpu_timestamp_speedup_against_the_seed():
    """@brief 验证性能块输出用 GPU timestamp 中位数算出的相对 seed 加速比与 seed 用时。

    构造 candidate 360.17 us、GPU 中位数 342.94 us、seed 300.03 us，预期 0.875x。
    """
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
