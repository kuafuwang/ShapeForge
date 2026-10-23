"""A confirmed speedup is shown only when it is bound to the kernel that was remeasured. No GPU."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from experiments.common import conclusion_of_run, published_conclusion, summarize
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator, RunResult
from tcenv.evaluate import EvalReport
from tcenv.schema import PROTOCOL_VERSION

from ..support.fakes import FakeClient

QUIET_BENCH = {"noisy": False, "candidate": {"cv": 0.01, "reps": 15}, "baselines": {"mlx": {"cv": 0.01, "reps": 15}}, "best_baseline": "mlx"}

SOURCE = "kernel void relu_kernel() {}\n"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _result(**extra):
    base = {
        "task": "relu", "dtype": "fp16", "model": "fake", "stop_reason": "max_rounds", "rounds": 1,
        "evaluations": 1, "spent_usd": 0.1, "budget_usd": 1.0, "best_speedup": 1.4, "confirmed_speedup": 0.9,
        "confirmed_status": "incorrect", "confirm_winner_label": "r1.p0", "confirmation_source_sha256": _digest(SOURCE),
        "confirmed_values": [], "target_id": "apple-m4-max",
    }
    base.update(extra)
    return base


def test_an_incorrect_confirmation_is_not_published_as_the_search_best():
    got = published_conclusion(_result(), best_metal_sha256=_digest(SOURCE), confirm_status="incorrect", winner_metal_sha256=_digest(SOURCE))
    assert got == {"speedup": None, "publication": "未确认"}
    row = summarize(_result(), {"status": "incorrect"})
    assert row["confirmed_speedup"] == 0.9 and row["best_speedup"] == 1.4
    assert row["published_speedup"] is None and row["publication"] == "未确认"


def test_a_result_without_the_hash_field_is_an_old_file(tmp_path):
    old = {"task": "relu", "dtype": "fp16", "model": "fake", "stop_reason": "max_rounds", "rounds": 1, "evaluations": 1,
           "spent_usd": 0.1, "budget_usd": 1.0, "best_speedup": 1.4, "confirmed_speedup": 1.1, "confirmed_status": "ok"}
    assert published_conclusion(old)["publication"] == "旧版未绑定源码"
    assert published_conclusion(old)["speedup"] is None


def test_a_bound_confirmation_shows_the_remeasured_number(tmp_path):
    digest = _digest(SOURCE)
    run = tmp_path / "run"
    (run / "candidates" / "000_r1.p0").mkdir(parents=True)
    (run / "best.metal").write_text(SOURCE)
    (run / "candidates" / "000_r1.p0" / "kernel.metal").write_text(SOURCE)
    (run / "confirm_report.json").write_text(json.dumps({
        "status": "ok", "speedup": 1.25, "benchmark": QUIET_BENCH,
        "protocol_version": PROTOCOL_VERSION, "target_id": "apple-m4-max",
    }))
    result = _result(confirmed_status="ok", confirmed_speedup=1.25, confirmation_source_sha256=digest, confirmed_values=[1.25])
    got = published_conclusion(result, best_metal_sha256=digest, confirm_status="ok", winner_metal_sha256=digest)
    assert got == {"speedup": 1.25, "publication": "confirmed"}
    row = summarize(result, None, run_dir=run)
    assert row["published_speedup"] == 1.25 and row["publication"] == "confirmed"
    # a different best.metal does not inherit the search number
    (run / "best.metal").write_text(SOURCE + " ")
    row = summarize(result, None, run_dir=run)
    assert row["published_speedup"] is None and row["publication"] == "未确认"


def test_finish_writes_the_winner_hash_and_leaves_it_null_when_confirmation_failed(tmp_path):
    cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
    search = SearchConfig(budget_usd=1.0, branches=1, max_rounds=0, eval_reps=3, stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False)

    def ev(*_a, **_k):
        return EvalReport(task="relu", dtype="fp16", size="small", mode="full", status="ok")

    orch = Orchestrator("relu", "fp16", "small", cfg, search, run_dir=tmp_path / "run", client=FakeClient([]), evaluate_fn=ev, log=lambda _s: None, seed=1)
    orch._confirm_winner = SimpleNamespace(label="r1.p0", source=SOURCE, speedup=1.25)
    orch.best = orch._confirm_winner
    res = RunResult(run_dir=str(orch.run_dir), task="relu", dtype="fp16", size="small", model="fake")
    res.confirmed_status, res.confirmed_speedup = "ok", 1.25
    orch._finish(res)
    doc = json.loads((orch.run_dir / "result.json").read_text())
    assert doc["confirmation_source_sha256"] == _digest(SOURCE)
    assert (orch.run_dir / "best.metal").read_text() == SOURCE

    orch._confirm_winner = None
    orch.best = SimpleNamespace(label="init", source=SOURCE, speedup=1.4)
    res2 = RunResult(run_dir=str(orch.run_dir), task="relu", dtype="fp16", size="small", model="fake")
    res2.confirmed_status = "incorrect"
    orch._finish(res2)
    doc2 = json.loads((orch.run_dir / "result.json").read_text())
    assert "confirmation_source_sha256" in doc2 and doc2["confirmation_source_sha256"] is None
    assert doc2["best_speedup"] == 1.4 and doc2["confirmed_speedup"] is None


def test_a_confirmation_served_from_the_evaluation_cache_is_not_published(tmp_path):
    """cache_hit means the report was replayed, not remeasured; the gate must not call that confirmed."""
    digest = _digest(SOURCE)
    run = tmp_path / "run"
    (run / "candidates" / "000_r1.p0").mkdir(parents=True)
    (run / "best.metal").write_text(SOURCE)
    (run / "candidates" / "000_r1.p0" / "kernel.metal").write_text(SOURCE)
    (run / "confirm_report.json").write_text(json.dumps({
        "status": "ok", "speedup": 1.25, "cache_hit": True, "benchmark": QUIET_BENCH,
        "protocol_version": PROTOCOL_VERSION, "target_id": "apple-m4-max",
    }))
    result = _result(confirmed_status="ok", confirmed_speedup=1.25, confirmation_source_sha256=digest, confirmed_values=[1.25, 1.25, 1.25])
    assert conclusion_of_run(run, result) == {"speedup": None, "publication": "不确定"}


def _bound_run(tmp_path, **report_extra):
    digest = _digest(SOURCE)
    run = tmp_path / "run"
    (run / "candidates" / "000_r1.p0").mkdir(parents=True, exist_ok=True)
    (run / "best.metal").write_text(SOURCE)
    (run / "candidates" / "000_r1.p0" / "kernel.metal").write_text(SOURCE)
    report = {"status": "ok", "speedup": 1.25, "benchmark": QUIET_BENCH, "protocol_version": PROTOCOL_VERSION, "target_id": "apple-m4-max"}
    report.update(report_extra)
    (run / "confirm_report.json").write_text(json.dumps(report))
    return run, digest


def test_the_published_number_must_be_the_median_of_the_recorded_batches(tmp_path):
    """A hand-edited or mis-merged result.json must not publish a number the batches do not support."""
    run, digest = _bound_run(tmp_path)
    result = _result(confirmed_status="ok", confirmed_speedup=1.9, confirmation_source_sha256=digest,
                     confirmed_values=[1.0, 1.0, 1.0])
    assert conclusion_of_run(run, result)["speedup"] is None
    result["confirmed_speedup"] = 1.0
    assert conclusion_of_run(run, result) == {"speedup": 1.0, "publication": "confirmed"}


def test_a_non_finite_or_textual_confirmed_speedup_is_not_published(tmp_path):
    run, digest = _bound_run(tmp_path)
    for bad in (float("inf"), float("nan"), "1.25", True):
        result = _result(confirmed_status="ok", confirmed_speedup=bad, confirmation_source_sha256=digest,
                         confirmed_values=[1.0])
        assert conclusion_of_run(run, result)["speedup"] is None, bad


def test_a_confirm_report_from_another_epoch_or_target_is_not_published(tmp_path):
    for extra in ({"protocol_version": "0"}, {"target_id": "another-mac"}):
        run, digest = _bound_run(tmp_path / str(extra), **extra)
        result = _result(confirmed_status="ok", confirmed_speedup=1.25, confirmation_source_sha256=digest,
                         confirmed_values=[1.25])
        assert conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}, extra

