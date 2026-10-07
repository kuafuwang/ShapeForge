"""The second-seed gate is off unless asked. A failed recheck does not become the next seed and is not a formal counterexample. No GPU."""

from __future__ import annotations

import asyncio
import json

import pytest

from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator, StopRun, check_recheck_record, eval_seed
from tcenv.evaluate import EvalReport

from .fakes import FakeClient

SOURCE = "kernel void relu_kernel() { /* candidate */ }\n"


def _cfg(on: bool, hangs: int = 3) -> tuple[LLMConfig, SearchConfig]:
    cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
    search = SearchConfig(
        budget_usd=5.0, branches=1, max_rounds=1, eval_reps=3, max_hangs=hangs, differential_recheck=on,
        stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False,
    )
    return cfg, search


def _report(task, dtype, size, *, mode, seed, status, speed):
    report = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=0 if seed is None else int(seed), status=status, stage="done", speedup=speed, message=status)
    report.speedups = {"mlx": speed} if speed else {}
    report.candidate_us = 10.0 if speed else None
    report.baseline_us = {"mlx": 12.0} if speed else {}
    report.benchmark = {"candidate": {"median_us": 10.0, "cv": 0.01, "reps": 3}, "baselines": {"mlx": {"median_us": 12.0, "cv": 0.01, "reps": 3}}, "best_baseline": "mlx"}
    if status == "incorrect":
        report.correctness = {"ok": False, "cases": [{"ok": False, "case": "nan"}], "tolerance": {"rtol": 0.0, "atol_rel": 0.0, "exact": True}}
    return report


def _orch(tmp_path, name, ev, *, on: bool, hangs: int = 3):
    cfg, search = _cfg(on, hangs)
    return Orchestrator("relu", "fp16", "small", cfg, search, run_dir=tmp_path / name, client=FakeClient([]), evaluate_fn=ev, log=lambda _s: None, seed=1)


def test_the_gate_off_promotes_a_candidate_without_a_second_evaluation(tmp_path):
    calls = []

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
        calls.append(mode)
        return _report(task, dtype, size, mode=mode, seed=seed, status="ok", speed=1.3)

    orch = _orch(tmp_path, "off", ev, on=False)
    cand = asyncio.run(orch._eval(SOURCE, "init"))
    assert cand.is_best and orch.best is cand
    assert calls == ["full"]
    assert not (orch.run_dir / "rechecks").exists()


def test_a_second_seed_that_is_incorrect_does_not_become_the_seed(tmp_path):
    calls = []

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
        calls.append((mode, seed))
        status = "incorrect" if mode == "quick" else "ok"
        return _report(task, dtype, size, mode=mode, seed=seed, status=status, speed=1.3 if status == "ok" else None)

    orch = _orch(tmp_path, "fail", ev, on=True)
    cand = asyncio.run(orch._eval(SOURCE, "init"))
    assert cand.is_best is False and orch.best is None
    assert calls[0][0] == "full" and calls[1] == ("quick", eval_seed(1, "recheck", "init", SOURCE))
    doc = json.loads((orch.run_dir / "rechecks" / "init.json").read_text())
    assert doc["class"] == "incorrect" and doc["case"] == "nan" and doc["counts_as_kernel_error"] is True
    assert doc["search_report"]["status"] == "ok" and doc["recheck_report"]["status"] == "incorrect"
    assert "counterexample" not in doc
    assert doc["retest_command"].startswith("uv run python -m tcenv.cli eval relu ")
    assert "--quick" in doc["retest_command"] and doc["human_permission_required"] is True
    assert check_recheck_record(doc) == []
    assert orch.trace[0]["differential_recheck"] == "incorrect"
    # the search measurement stays cached; the failed recheck is not stored as that cache entry
    assert next(iter(orch.cache.values())).status == "ok"


def test_a_second_seed_that_passes_still_becomes_the_seed(tmp_path):
    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
        return _report(task, dtype, size, mode=mode, seed=seed, status="ok", speed=1.2)

    orch = _orch(tmp_path, "pass", ev, on=True)
    cand = asyncio.run(orch._eval(SOURCE, "init"))
    assert cand.is_best and not (orch.run_dir / "rechecks").exists()
    assert orch.trace[0]["differential_recheck"] is None


def test_a_recheck_timeout_stops_the_search_and_is_not_stored_as_a_wrong_kernel(tmp_path):
    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
        status = "timeout" if mode == "quick" else "ok"
        return _report(task, dtype, size, mode=mode, seed=seed, status=status, speed=1.2 if status == "ok" else None)

    orch = _orch(tmp_path, "hang", ev, on=True, hangs=1)
    with pytest.raises(StopRun, match="gpu_hang"):
        asyncio.run(orch._eval(SOURCE, "init"))
    assert orch.best is None
    doc = json.loads((orch.run_dir / "candidates" / "000_init" / "differential_recheck.json").read_text())
    assert doc["class"] == "timeout" and doc["counts_as_kernel_error"] is False
    assert check_recheck_record(doc) == []
    assert next(iter(orch.cache.values())).status == "ok"


def test_a_failed_confirmation_keeps_a_retest_record_and_does_not_run_it(tmp_path):
    calls = []

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
        calls.append(reps)
        status = "incorrect" if reps > 3 else "ok"
        return _report(task, dtype, size, mode=mode, seed=seed, status=status, speed=1.2 if status == "ok" else None)

    orch = _orch(tmp_path, "confirm", ev, on=False)

    async def go():
        from taic.orchestrator import RunResult

        await orch._eval(SOURCE, "init")
        res = RunResult(run_dir=str(orch.run_dir), task="relu", dtype="fp16", size="small", model="fake", budget_usd=5.0)
        await orch._confirm(res)
        return res

    res = asyncio.run(go())
    assert res.confirmed_status == "incorrect"
    doc = json.loads((orch.run_dir / "rechecks" / "confirm_init.json").read_text())
    assert doc["class"] == "incorrect" and doc["speed_relation"] == "confirm"
    assert "counterexample" not in json.dumps(doc)
    assert check_recheck_record(doc) == []
    assert calls == [3, 6]


def test_overlapping_branches_keep_the_faster_kernel(tmp_path):
    fast = SOURCE + "// fast\n"
    slow = SOURCE + "// slow\n"

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
        speed = 2.0 if "fast" in source else 1.2
        return _report(task, dtype, size, mode=mode, seed=seed, status="ok", speed=speed)

    orch = _orch(tmp_path, "race", ev, on=True)

    async def go():
        await asyncio.gather(orch._eval(slow, "slow"), orch._eval(fast, "fast"))

    asyncio.run(go())
    assert orch.best is not None and orch.best.source == fast and orch.best.speedup == pytest.approx(2.0)


def test_a_rejected_recheck_is_left_out_of_confirmation(tmp_path):
    good = SOURCE + "// good\n"
    bad = SOURCE + "// bad\n"
    confirmed = []

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
        if reps > 3:
            confirmed.append(source)
        if "bad" in source and (mode == "quick" or reps > 3):
            return _report(task, dtype, size, mode=mode, seed=seed, status="incorrect", speed=None)
        speed = 2.0 if "bad" in source else 1.4
        return _report(task, dtype, size, mode=mode, seed=seed, status="ok", speed=speed)

    orch = _orch(tmp_path, "pool", ev, on=True)

    async def go():
        from taic.orchestrator import RunResult

        await orch._eval(good, "good")
        await orch._eval(bad, "bad")
        res = RunResult(run_dir=str(orch.run_dir), task="relu", dtype="fp16", size="small", model="fake", budget_usd=5.0)
        await orch._confirm(res)
        return res

    res = asyncio.run(go())
    assert orch.best.source == good
    assert res.confirmed_status == "ok"
    assert confirmed == [good]


def test_a_counterexample_nested_inside_the_report_is_still_refused():
    from taic.orchestrator import _RECHECK_KEYS, _strip_counterexample

    nested = {"class": "incorrect", "recheck_report": {"status": "incorrect", "counterexample": [{"case": 1}]}}
    assert "counterexample" not in json.dumps(_strip_counterexample(nested))
    doc = {key: "x" for key in _RECHECK_KEYS}
    doc["class"] = "incorrect"
    doc["counts_as_kernel_error"] = True
    doc["human_permission_required"] = True
    doc["recheck_report"] = {"counterexample": {"bytes": "no"}}
    assert any("counterexample" in problem for problem in check_recheck_record(doc))
