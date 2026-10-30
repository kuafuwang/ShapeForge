"""The second-seed gate is off unless asked. A failed recheck does not become the next seed and is not a formal counterexample. No GPU."""

from __future__ import annotations

import asyncio
import json

import pytest

from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator, StopRun, check_recheck_record, eval_seed
from tcenv.evaluate import EvalReport

from ..support.fakes import FakeClient

SOURCE = "kernel void relu_kernel() { /* candidate */ }\n"


def _cfg(on: bool, hangs: int = 3) -> tuple[LLMConfig, SearchConfig]:
    """@brief 构造本文件测试用的模型配置与搜索配置，可指定差分复核开关与挂起上限。

    @param on 是否开启 differential_recheck。
    @param hangs 允许的 GPU 挂起次数。
    @return (LLMConfig, SearchConfig) 二元组。
    """
    cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
    search = SearchConfig(
        budget_usd=5.0, branches=1, max_rounds=1, eval_reps=3, max_hangs=hangs, differential_recheck=on,
        stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False,
    )
    return cfg, search


def _report(task, dtype, size, *, mode, seed, status, speed, source="kernel void demo() {}\n"):
    """@brief 造一份带速度数据与基准数据的 EvalReport；状态为 incorrect 时补上失败用例。

    @param task 任务名。
    @param dtype 数据类型。
    @param size 规模名。
    @param mode 评测模式（full 或 quick）。
    @param seed 评测种子，None 时记为 0。
    @param status 报告状态。
    @param speed 报告的速度值，假值表示无速度。
    @param source 这次评测的候选源码（build 记录必须绑定它）。
    @return 填好的 EvalReport。
    """
    from tasks import get_task

    from tests.support.report_build import attach_build, attach_identity

    task_obj = get_task(task, dtype, size)
    report = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=0 if seed is None else int(seed), status=status, stage="done", speedup=speed, message=status, target_id="apple-m4-max",
                        backend="metal", backend_version="1", protocol_version="3")
    attach_build(report, task_obj, source=source.encode("utf-8"))
    attach_identity(report, task_obj, backend_id="metal")
    report.speedups = {"mlx": speed} if speed else {}
    report.candidate_us = 10.0 if speed else None
    report.baseline_us = {"mlx": 12.0} if speed else {}
    report.benchmark = {"candidate": {"median_us": 10.0, "cv": 0.01, "reps": 3}, "baselines": {"mlx": {"median_us": 12.0, "cv": 0.01, "reps": 3}}, "best_baseline": "mlx"}
    if status == "incorrect":
        report.correctness = {"ok": False, "cases": [{"ok": False, "case": "nan"}], "tolerance": {"rtol": 0.0, "atol_rel": 0.0, "exact": True}}
    return report


def _orch(tmp_path, name, ev, *, on: bool, hangs: int = 3):
    """@brief 在给定子目录下创建一个无脚本化回复的 Orchestrator，并注入评测函数。

    @param tmp_path pytest 提供的临时目录。
    @param name 运行子目录名。
    @param ev 替换真实评测的函数。
    @param on 是否开启差分复核。
    @param hangs 允许的 GPU 挂起次数。
    @return 构造好的 Orchestrator。
    """
    cfg, search = _cfg(on, hangs)
    from tcenv.evaluation import LocalEvaluator

    from tests.support.report_build import structural_expected

    # The injection stands in for the local Metal executor *and* for its current expected
    # identity (the default executor over a custom function cannot state today's profile).
    evaluator = LocalEvaluator(ev, expected_identity=structural_expected)
    return Orchestrator("relu", "fp16", "small", cfg, search, run_dir=tmp_path / name, client=FakeClient([]), evaluate_fn=ev, evaluator=evaluator, log=lambda _s: None, seed=1)


def test_the_gate_off_promotes_a_candidate_without_a_second_evaluation(tmp_path):
    """@brief 验证差分复核关闭时只需一次评测就升为最优，不会生成 rechecks 目录。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
        calls.append(mode)
        return _report(task, dtype, size, mode=mode, seed=seed, status="ok", speed=1.3, source=source)

    orch = _orch(tmp_path, "off", ev, on=False)
    cand = asyncio.run(orch._eval(SOURCE, "init"))
    assert cand.is_best and orch.best is cand
    assert calls == ["full"]
    assert not (orch.run_dir / "rechecks").exists()


def test_a_second_seed_that_is_incorrect_does_not_become_the_seed(tmp_path):
    """@brief 验证第二次种子复核失败时该候选不作为种子，且写回可人工重测的 recheck 记录。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
        calls.append((mode, seed))
        status = "incorrect" if mode == "quick" else "ok"
        return _report(task, dtype, size, mode=mode, seed=seed, status=status, speed=1.3 if status == "ok" else None, source=source)

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
    """@brief 验证第二次种子复核通过时候选仍然成为最优，且不写 recheck 记录。

    @param tmp_path pytest 提供的临时目录。
    """
    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
        return _report(task, dtype, size, mode=mode, seed=seed, status="ok", speed=1.2, source=source)

    orch = _orch(tmp_path, "pass", ev, on=True)
    cand = asyncio.run(orch._eval(SOURCE, "init"))
    assert cand.is_best and not (orch.run_dir / "rechecks").exists()
    assert orch.trace[0]["differential_recheck"] is None


def test_a_recheck_timeout_stops_the_search_and_is_not_stored_as_a_wrong_kernel(tmp_path):
    """@brief 验证复核超时会停下搜索（gpu_hang），且不把超时当成内核错误写入缓存。

    @param tmp_path pytest 提供的临时目录。
    @raises StopRun 复核超时且已达挂起上限时抛出，原因为 gpu_hang。
    """
    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
        status = "timeout" if mode == "quick" else "ok"
        return _report(task, dtype, size, mode=mode, seed=seed, status=status, speed=1.2 if status == "ok" else None, source=source)

    orch = _orch(tmp_path, "hang", ev, on=True, hangs=1)
    with pytest.raises(StopRun, match="gpu_hang"):
        asyncio.run(orch._eval(SOURCE, "init"))
    assert orch.best is None
    doc = json.loads((orch.run_dir / "candidates" / "000_init" / "differential_recheck.json").read_text())
    assert doc["class"] == "timeout" and doc["counts_as_kernel_error"] is False
    assert check_recheck_record(doc) == []
    assert next(iter(orch.cache.values())).status == "ok"


def test_an_unresolved_recheck_is_pending_on_resume(tmp_path):
    """@brief timeout/crash 不当核错，但也不能在 resume 时冒充复查通过。"""
    from taic import resume as rs

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
        status = "timeout" if mode == "quick" else "ok"
        return _report(task, dtype, size, mode=mode, seed=seed, status=status, speed=1.2 if status == "ok" else None, source=source)

    orch = _orch(tmp_path, "pending", ev, on=True, hangs=1)
    with pytest.raises(StopRun):
        asyncio.run(orch._eval(SOURCE, "init"))
    candidates = rs._read_candidates(orch.run_dir, [])
    assert candidates[0]["recheck_pending"] is True
    assert candidates[0]["recheck_rejected"] is False


def test_a_failed_confirmation_keeps_a_retest_record_and_does_not_run_it(tmp_path):
    """@brief 验证确认失败时保留待重测记录但不自动执行，并记下失败状态与速度关系。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
        calls.append(reps)
        status = "incorrect" if reps > 3 else "ok"
        return _report(task, dtype, size, mode=mode, seed=seed, status=status, speed=1.2 if status == "ok" else None, source=source)

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
    """@brief 验证并发分支交错完成时，最快的内核最终成为最优。

    @param tmp_path pytest 提供的临时目录。
    """
    fast = SOURCE + "// fast\n"
    slow = SOURCE + "// slow\n"

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
        speed = 2.0 if "fast" in source else 1.2
        return _report(task, dtype, size, mode=mode, seed=seed, status="ok", speed=speed, source=source)

    orch = _orch(tmp_path, "race", ev, on=True)

    async def go():
        await asyncio.gather(orch._eval(slow, "slow"), orch._eval(fast, "fast"))

    asyncio.run(go())
    assert orch.best is not None and orch.best.source == fast and orch.best.speedup == pytest.approx(2.0)


def test_a_later_failed_batch_does_not_publish_the_earlier_success(tmp_path):
    """@brief 验证后续批次失败时不会拿前面批次的成功来发布，确认结果保持未确认。

    @param tmp_path pytest 提供的临时目录。
    """
    seen = {"n": 0}

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
        seen["n"] += 1
        if reps > 3 and seen["n"] >= 3:
            return _report(task, dtype, size, mode=mode, seed=seed, status="incorrect", speed=None, source=source)
        return _report(task, dtype, size, mode=mode, seed=seed, status="ok", speed=1.2, source=source)

    orch = _orch(tmp_path, "partial", ev, on=False)

    async def go():
        from taic.orchestrator import RunResult

        await orch._eval(SOURCE, "init")
        res = RunResult(run_dir=str(orch.run_dir), task="relu", dtype="fp16", size="small", model="fake", budget_usd=5.0)
        await orch._confirm(res)
        return res

    res = asyncio.run(go())
    assert res.confirmed_status == "incorrect"
    assert res.confirmed_speedup is None and res.confirmation_source_sha256 is None
    assert not (orch.run_dir / "confirm_report.json").exists()
    assert res.confirmations[0]["failed_batches"] == 1
    assert res.confirmations[0]["status"] == "incorrect"
    assert (orch.run_dir / "rechecks" / "confirm_init.b2.json").is_file()


def test_tied_candidates_share_one_confirm_seed(tmp_path):
    """@brief 验证速度相近的平局候选在确认时共用同一个确认种子，且最优取更快的那个。

    @param tmp_path pytest 提供的临时目录。
    """
    seeds = []

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
        if reps > 3:
            seeds.append((source, seed))
        speed = 1.20 if "slow" in source else 1.21
        return _report(task, dtype, size, mode=mode, seed=seed, status="ok", speed=speed, source=source)

    orch = _orch(tmp_path, "same-seed", ev, on=False)
    slow = SOURCE + "// slow\n"
    fast = SOURCE + "// fast\n"

    async def go():
        from taic.orchestrator import RunResult

        await orch._eval(slow, "slow")
        await orch._eval(fast, "fast")
        res = RunResult(run_dir=str(orch.run_dir), task="relu", dtype="fp16", size="small", model="fake", budget_usd=5.0)
        await orch._confirm(res)
        return res

    res = asyncio.run(go())
    shared = eval_seed(1, "confirm", "b1", "")
    assert [seed for _source, seed in seeds][0] == [seed for _source, seed in seeds][3] == shared
    assert res.confirmed_status == "ok" and orch.best.source == slow


def test_a_rejected_recheck_is_left_out_of_confirmation(tmp_path):
    """@brief 验证复核被拒的候选不进入确认流程，确认只跑通过复核的那个内核。

    @param tmp_path pytest 提供的临时目录。
    """
    good = SOURCE + "// good\n"
    bad = SOURCE + "// bad\n"
    confirmed = []

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
        if reps > 3:
            confirmed.append(source)
        if "bad" in source and (mode == "quick" or reps > 3):
            return _report(task, dtype, size, mode=mode, seed=seed, status="incorrect", speed=None, source=source)
        speed = 2.0 if "bad" in source else 1.4
        return _report(task, dtype, size, mode=mode, seed=seed, status="ok", speed=speed, source=source)

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
    assert confirmed and set(confirmed) == {good}  # 每次确认都只测 good；bad 一次都没进来


def test_a_counterexample_nested_inside_the_report_is_still_refused():
    """@brief 验证嵌套在 recheck_report 里的 counterexample 也会被剔除，并被记录校验查出。"""
    from taic.orchestrator import _RECHECK_KEYS, _strip_counterexample

    nested = {"class": "incorrect", "recheck_report": {"status": "incorrect", "counterexample": [{"case": 1}]}}
    assert "counterexample" not in json.dumps(_strip_counterexample(nested))
    doc = {key: "x" for key in _RECHECK_KEYS}
    doc["class"] = "incorrect"
    doc["counts_as_kernel_error"] = True
    doc["human_permission_required"] = True
    doc["recheck_report"] = {"counterexample": {"bytes": "no"}}
    assert any("counterexample" in problem for problem in check_recheck_record(doc))
