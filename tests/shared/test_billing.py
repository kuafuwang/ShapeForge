import json

import pytest

from experiments.reprice_runs import cost_of, recompute_run
from taic.committees import _priced
from taic.llm import CostLedger, LLMConfig


def _cfg(**kw):
    return LLMConfig(base_url="http://x", api_key="k", model="m", **kw)


def test_priced_keeps_the_new_members_cache_write_price():
    active = _cfg(price_cache_write=0.0)
    priced = _priced(
        active, base_url="http://y", api_key="k2", model="kimi-k3",
        price={"input": 3.0, "output": 15.0, "cacheRead": 0.3, "cacheWrite": 3.0},
    )
    assert priced.price_cache_write == 3.0
    assert priced.price_in == 3.0 and priced.price_out == 15.0 and priced.price_cache_read == 0.3


def test_charge_records_write_tokens_and_the_price_basis():
    ledger = CostLedger(_cfg(price_in=3.0, price_out=15.0, price_cache_read=0.3, price_cache_write=3.0), 10.0)
    ledger.charge(4.87, "r1", tokens={"input": 1000, "output": 2000, "cache_read": 3000, "cache_write": 1_000_000})
    assert ledger.in_tokens_write == 1_000_000
    assert ledger.in_tokens_miss == 1000 and ledger.out_tokens == 2000
    row = ledger.requests[-1]
    assert row["write"] == 1_000_000 and row["basis"]["cache_write"] == 3.0
    summary = ledger.summary()
    assert summary["input_tokens_cache_write"] == 1_000_000
    assert summary["cost_basis"]["cache_write"] == 3.0


def test_add_records_cache_write_from_usage_details():
    ledger = CostLedger(_cfg(price_in=3.0, price_out=15.0, price_cache_read=0.3, price_cache_write=3.0), 10.0)
    ledger.add({"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 4}}, "t1")
    assert ledger.requests[-1]["write"] == 4
    assert ledger.summary()["input_tokens_cache_write"] == 4


def test_recompute_reads_agent_rows_and_leaves_spent_usd_alone(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "result.json").write_text(json.dumps({"spent_usd": 1.0}))
    (run / "transcript.jsonl").write_text("\n".join(json.dumps(row) for row in [
        {"event": "agent", "label": "init", "model": "m", "provider": "p",
         "tokens": {"input": 1_000_000, "output": 0, "cache_read": 0, "cache_write": 1_000_000}, "cost_usd": 0.9},
        {"event": "chat", "messages": [{"role": "user", "content": "x"}]},
    ]))
    table = {"m": {"input": 1.0, "output": 2.0, "cache_read": 0.5, "cache_write": 3.0}}
    doc = recompute_run(run, table)
    assert doc["recorded_usd"] == 1.0
    assert doc["recomputed_usd"] == 4.0
    assert doc["delta_usd"] == 3.0
    assert doc["sessions"][0]["tokens"]["cache_write"] == 1_000_000
    assert json.loads((run / "cost_recomputed.json").read_text())["recomputed_usd"] == 4.0
    assert json.loads((run / "result.json").read_text())["spent_usd"] == 1.0


def test_recompute_marks_a_model_missing_from_the_price_table(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "transcript.jsonl").write_text(json.dumps(
        {"event": "agent", "label": "init", "model": "unknown", "tokens": {"input": 10}, "cost_usd": 1.0}
    ) + "\n")
    doc = recompute_run(run, {})
    assert doc["missing_models"] == ["unknown"]
    assert doc["recomputed_usd"] == 0.0


def test_cost_of_uses_each_kind():
    prices = {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.0}
    assert cost_of({"input": 1_000_000, "output": 1_000_000, "cache_read": 1_000_000, "cache_write": 1_000_000}, prices) == 21.3


def test_a_non_finite_or_negative_charge_is_refused():
    """A NaN charge made total_usd NaN, and every budget comparison then returned false."""
    ledger = CostLedger(_cfg(), 10.0)
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError):
            ledger.charge(bad)
    assert ledger.total_usd == 0.0
    ledger.charge(1.0)
    assert ledger.total_usd == 1.0

