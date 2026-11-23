"""The second-seed gate is off unless asked. A failed recheck does not become the next seed and is not a formal counterexample. No GPU."""

from __future__ import annotations

import asyncio
import json

import pytest

from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator, StopRun, check_recheck_record, eval_seed
from tcenv.evaluate import EvalReport

from ..support.fakes import FakeClient
from tests.support.v2_specimen import V2_TASK, CallbackV2Evaluator, bind_v2

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
    from tests.support.report_build import attach_build

    from tests.support.v2_specimen import v2_structural_evaluation_task

    task_obj = v2_structural_evaluation_task(task, dtype, size)
    report = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=0 if seed is None else int(seed), status=status, stage="done", speedup=speed, message=status, target_id="apple-m4-max",
                        backend="metal", backend_version="1", protocol_version="3")
    attach_build(report, task_obj, source=source.encode("utf-8"))
    report.speedups = {"mlx": speed} if speed else {}
    report.candidate_us = 10.0 if speed else None
    # 基准中位数必须真的导出 speedup：发布谓词从这些数重算，不从自报字段抄。
    report.baseline_us = {"mlx": 10.0 * float(speed)} if speed else {}
    report.benchmark = {
        "candidate": {"median_us": 10.0, "cv": 0.01, "reps": 3},
        "baselines": {"mlx": {"median_us": 10.0 * float(speed) if speed else 12.0, "cv": 0.01, "reps": 3}},
        "best_baseline": "mlx",
    }
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
    binding = bind_v2(V2_TASK, "fp16", "small")
    evaluator = CallbackV2Evaluator(binding, ev)
    return Orchestrator(
        V2_TASK, "fp16", "small", cfg, search, run_dir=tmp_path / name,
        client=FakeClient([]), evaluate_fn=ev, evaluator=evaluator, binding=binding,
        log=lambda _s: None, seed=1,
    )


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
    assert doc["retest_command"].startswith(f"uv run python -m tcenv.cli eval-v2 {V2_TASK} ")
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
