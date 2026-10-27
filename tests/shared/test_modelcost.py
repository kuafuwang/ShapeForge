"""Per-model totals and efficiency come from billed agent rows only."""

from __future__ import annotations

import math

import pytest

from taic.modelcost import efficiency_fields, from_events


def test_model_totals_ignore_scheduling_and_non_finite_costs():
    """@brief 验证按模型汇总只统计真实的 agent 计费行，跳过重试调度行与非有限花费。

    断言重试调度行不计入会话数，NaN 花费按 0 计但仍计入会话数，
    同一模型的 token 分类被逐项累加，结果按花费降序排列。
    """
    rows = from_events([
        {"event": "provider_retry", "provider": "p", "model": "m", "cost_usd": 99},
        {"event": "chat", "provider": "p", "model": "m", "cost_usd": 99},
        {"event": "agent", "provider": "p", "model": "m", "committee": "first", "cost_usd": 0.2,
         "tokens": {"input": 1, "output": 2, "cache_read": 3, "cache_write": 4}},
        {"event": "agent", "provider": "p", "model": "m", "committee": "first", "cost_usd": math.nan,
         "tokens": {"input": 5, "output": 6}},
        {"event": "agent", "provider": "q", "model": "n", "committee": "second", "cost_usd": 1.0,
         "tokens": {"output": 10}},
    ])
    assert [(row["provider"], row["model"], row["sessions"], row["cost_usd"]) for row in rows] == [
        ("q", "n", 1, 1.0), ("p", "m", 2, 0.2),
    ]
    p = next(row for row in rows if row["model"] == "m")
    assert (p["input_tokens"], p["output_tokens"], p["cache_read_tokens"], p["cache_write_tokens"]) == (6, 8, 3, 4)
    assert p["cache_read_per_output"] == 0.375
    assert p["usd_per_1m_output"] == 25_000
    assert p["cost_per_session_usd"] == 0.1
    assert p["sessions_with_cost"] == 1  # 有限成本只有一场；场均仍按全部两场。


def test_model_totals_use_ratios_of_sums_not_mean_session_ratios():
    """Aggregate token and cost totals first, then derive all three efficiency fields."""
    row = from_events([
        {"event": "agent", "provider": "p", "model": "m", "committee": "first", "cost_usd": 0.2,
         "tokens": {"output": 2, "cache_read": 3}},
        {"event": "agent", "provider": "p", "model": "m", "committee": "first", "cost_usd": 0.3,
         "tokens": {"output": 6, "cache_read": 5}},
    ])[0]

    assert row["cost_usd"] == 0.5
    assert row["cache_read_per_output"] == 1.0  # (3 + 5) / (2 + 6), not mean(1.5, 5/6)
    assert row["usd_per_1m_output"] == 62_500  # 0.5 / 8, not mean(0.2/2, 0.3/6)
    assert row["cost_per_session_usd"] == 0.25
    assert row["sessions_with_cost"] == 2


@pytest.mark.parametrize("cost", [None, math.nan, math.inf])
def test_model_totals_with_no_finite_cost_have_null_cost_efficiency(cost):
    """Missing/non-finite costs keep the legacy zero total but never create fake ratios."""
    row = from_events([{
        "event": "agent", "provider": "p", "model": "m", "cost_usd": cost,
        "tokens": {"output": 2, "cache_read": 4},
    }])[0]

    assert row["cost_usd"] == 0.0
    assert row["cache_read_per_output"] == 2.0
    assert row["usd_per_1m_output"] is None
    assert row["cost_per_session_usd"] is None
    assert row["sessions_with_cost"] == 0


@pytest.mark.parametrize("output", [None, 0, -1, math.nan, math.inf, "2", True])
def test_efficiency_rejects_zero_missing_and_invalid_output(output):
    """A missing/zero/invalid output denominator yields nulls rather than Infinity."""
    fields = efficiency_fields(tokens_cache_read=4, tokens_output=output, cost_usd=1.0, sessions=2)
    assert fields == {
        "cache_read_per_output": None,
        "usd_per_1m_output": None,
        "cost_per_session_usd": 0.5,
    }


@pytest.mark.parametrize("cost", [None, math.nan, math.inf, -1.0, "1", True])
def test_efficiency_rejects_missing_and_invalid_cost(cost):
    """Cost-derived fields are null independently of a valid cache/output ratio."""
    fields = efficiency_fields(tokens_cache_read=4, tokens_output=2, cost_usd=cost, sessions=2)
    assert fields == {
        "cache_read_per_output": 2.0,
        "usd_per_1m_output": None,
        "cost_per_session_usd": None,
    }


def test_efficiency_zero_sessions_and_invalid_cache_read_are_null():
    """Only aggregate rows expose per-session cost; zero sessions cannot divide."""
    zero_sessions = efficiency_fields(tokens_cache_read=4, tokens_output=2, cost_usd=1.0, sessions=0)
    assert zero_sessions["cost_per_session_usd"] is None
    assert zero_sessions["cache_read_per_output"] == 2.0
    assert zero_sessions["usd_per_1m_output"] == 500_000

    invalid_cache = efficiency_fields(tokens_cache_read=math.nan, tokens_output=2, cost_usd=1.0)
    assert invalid_cache == {"cache_read_per_output": None, "usd_per_1m_output": 500_000}
    assert "cost_per_session_usd" not in invalid_cache
