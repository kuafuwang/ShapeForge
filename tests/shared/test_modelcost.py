"""Per-model totals are built from billed agent rows, never from retry scheduling rows."""

from __future__ import annotations

import math

from taic.modelcost import from_events


def test_model_totals_ignore_scheduling_and_non_finite_costs():
    rows = from_events([
        {"event": "provider_retry", "provider": "p", "model": "m", "cost_usd": 99},
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
