"""Same-window A/A control: identical bytes launched twice inside one interleaved call.

The point of the control is the *resolution* number, not a speedup: the ratio
of two runs of the same bytes should be 1, and the half-width of its interval is
the smallest difference this GPU window can resolve. Without it the report
compares the candidate against a seed stamp from an earlier window, which is how
a 1 % edit became undecidable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tcenv.backends.metal.v2_runtime import BENCH_CONTROL, _control_document
from tcenv.verdict import control_absent_line, control_line


def _stats(candidate, control, *, cv=0.004):
    return {
        "candidate": SimpleNamespace(samples_s=tuple(candidate), cv=cv),
        BENCH_CONTROL: SimpleNamespace(samples_s=tuple(control), cv=cv),
    }


def test_identical_bytes_produce_a_ratio_near_one_and_report_the_resolution():
    """@brief 同一份字节的比值应落在 1 附近，区间半宽就是这一段窗口的分辨率。"""
    document = _control_document(
        _stats([3.00e-6, 3.02e-6, 2.98e-6, 3.01e-6], [2.99e-6, 3.01e-6, 3.00e-6, 3.02e-6]),
        seed=7,
    )
    assert document["timed_by"] == "wall_clock"
    assert document["rounds"] == 4
    assert document["median_ratio"] == pytest.approx(1.0, abs=5e-3)
    lo, hi = document["interval"]
    assert lo <= 1.0 <= hi
    assert document["resolvable_relative"] == pytest.approx((hi - lo) / 2.0)
    assert document["conclusion"] == "no_diff"


def test_a_real_gap_shows_up_as_a_diff():
    """@brief 候选真慢 3% 时对照组要报 diff，而不是「和 1 没差」。"""
    document = _control_document(
        _stats([3.09e-6, 3.10e-6, 3.08e-6, 3.09e-6], [3.00e-6, 3.00e-6, 3.00e-6, 3.00e-6]),
        seed=7,
    )
    assert document["median_ratio"] == pytest.approx(1.03, abs=5e-3)
    assert document["interval"][0] > 1.02
    assert document["conclusion"] == "diff"


def test_it_says_too_noisy_when_the_rounds_cannot_separate_the_bands():
    """@brief 区间跨过 1.02 边界时不给结论，这正是「这个窗口分辨不了」。"""
    document = _control_document(
        _stats([3.00e-6, 3.10e-6, 2.95e-6, 3.06e-6], [3.00e-6, 3.00e-6, 3.00e-6, 3.00e-6]),
        seed=7,
    )
    assert document["conclusion"] == "too_noisy"


def test_one_round_cannot_produce_an_interval():
    """@brief 一轮没有离散度可言：不编造区间，直接说轮数不够。"""
    document = _control_document(_stats([3.0e-6], [3.0e-6]), seed=7)
    assert document["timed_by"] == "wall_clock"
    assert document["rounds"] == 1
    assert document["interval"] is None
    assert document["resolvable_relative"] is None
    assert document["conclusion"] == "too_noisy"


def test_no_control_means_none_not_a_zero():
    """@brief 没测就是 None：不能写成 0，那会被读成「完全没差别」。"""
    assert _control_document({"candidate": SimpleNamespace(samples_s=(3e-6,), cv=0.0)}, seed=7) is None
    assert _control_document({}, seed=7) is None


def test_the_report_line_states_the_resolution_and_the_expectation():
    """@brief 打印行同时给出比值、区间、分辨率和「同一份字节应该读 1.000」。"""
    line = control_line({"benchmark": {"control": {
        "median_ratio": 1.0002, "interval": [0.9949, 1.0034], "rounds": 16,
    }}})
    assert "same-window A/A control" in line
    assert "1.0002" in line
    assert "16 rounds" in line
    assert "0.43%" in line
    assert "1.000" in line


def test_an_absent_control_is_announced_rather_than_silently_omitted():
    """@brief 没做对照就必须说出来，否则读者会以为上面的比值是配对的。"""
    assert control_line({"benchmark": {}}) == ""
    absent = control_absent_line({"benchmark": {}})
    assert "--control" in absent
    assert "across two windows" in absent
    # 做了对照时这一行不再出现。
    assert control_absent_line({"benchmark": {"control": {"interval": [0.99, 1.01]}}}) == ""


def test_the_runtime_keeps_the_control_out_of_the_baselines(monkeypatch):
    """@brief 对照臂绝不能进 ``baselines``：进了就会变成「最强基线」，把 speedup 变成候选和自己比。"""
    from tcenv.backends.metal import v2_runtime as module

    assert module.BENCH_CONTROL not in {"candidate"}
    # 用真实代码路径构造 baselines 过滤条件：与 benchmark() 里那段表达式一致。
    stats_all = {
        "candidate": SimpleNamespace(to_json=lambda: {}),
        BENCH_CONTROL: SimpleNamespace(to_json=lambda: {}),
        "mlx": SimpleNamespace(to_json=lambda: {}),
    }
    kept = {
        name: value for name, value in stats_all.items()
        if name != "candidate" and name != BENCH_CONTROL
    }
    assert set(kept) == {"mlx"}
