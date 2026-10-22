"""The evaluation cache key and the report round-trip. No GPU."""

from __future__ import annotations

import json

from taic.llm import SearchConfig
from tcenv.evaluate import EvalReport, _cache_read, _cache_write, _eval_cache_key


def _key(**over):
    base = dict(
        task="relu", dtype="fp16", size="small", mode="full", seed=1, reps=15, cases=None,
        fast_bench=False, require_contract_launch=False, validate=False, measure_serial=False,
        target_id="apple-m4-max", kernel_source="kernel void k() {}\n",
    )
    base.update(over)
    return _eval_cache_key(**base)


def test_the_key_separates_seed_serial_protocol_and_source():
    base = _key()
    assert base != _key(seed=2)
    assert base != _key(measure_serial=True)
    assert base != _key(kernel_source="kernel void other() {}\n")
    assert base == _key()


def test_a_cached_report_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("TCENV_EVAL_CACHE_DIR", str(tmp_path))
    report = EvalReport(task="relu", dtype="fp16", size="small", mode="full", status="ok", speedup=1.2, candidate_us=1.0)
    report.benchmark = {"candidate": {"cv": 0.01}}
    _cache_write("k", report)
    cached = _cache_read("k")
    assert cached is not None and cached.speedup == 1.2 and cached.status == "ok"
    assert json.loads((tmp_path / "k.json").read_text())["schema_version"] >= 1


def test_the_search_defaults_keep_serial_off_and_the_cache_on():
    config = SearchConfig()
    assert config.measure_serial is False
    assert config.eval_cache is True
