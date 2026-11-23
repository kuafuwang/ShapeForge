"""v2 搜索的真实 Metal 垂直切片：脚本化模型（不花钱）+ 真 evaluate_task + P2 确认门。"""

from __future__ import annotations

import asyncio
import json

import pytest

from experiments.contract_sensitivity import gemm_source, parse
from taic.kernel_agent import agent_for_script
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator, eval_seed
from taic.task_binding import bind_task
from taic.v2_evaluation import V2LocalEvaluator

from ...support.fakes import FakeClient, make_response

pytestmark = pytest.mark.gpu

VARIANT = "128x64x32t256"


def _candidate(contract, variant: str = VARIANT) -> str:
    bm, bn, bk, threads = parse(variant)
    m, n = int(contract.instance.symbols["M"]), int(contract.instance.symbols["N"])
    assert m % bm == 0 and n % bn == 0
    constants = "\n".join((
        f"constant constexpr uint BLOCK_M = {bm};",
        f"constant constexpr uint BLOCK_N = {bn};",
        f"constant constexpr uint BLOCK_K = {bk};",
    )) + "\n"
    launch = (
        "/* taic-launch\nmath fast\n"
        f"launch gemm_kernel grid={n // bn},{m // bm},1 threads={threads},1,1\n*/\n"
    )
    return constants + launch + gemm_source(variant)


def _seed_sensitive_relu(contract) -> str:
    """@brief 找一份 seed 7 全对、派生 recheck seed 必错的真实候选字节。"""
    cases = ("normal", "wide_range", "large", "tiny", "outlier", "ieee")

    def source_for(threshold: float) -> str:
        return (
            "/* taic-launch\nlaunch relu_kernel grid=N/256,1,1 threads=256,1,1\n*/\n"
            "kernel void relu_kernel(device const half* x [[buffer(0)]], "
            "device half* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n"
            f"  if (x[0] > {threshold - 0.125:.3f}h && x[0] < {threshold + 0.125:.3f}h) {{\n"
            "    y[i] = x[i] + 1.0h;\n"
            "  } else {\n"
            "    y[i] = x[i] <= 0.0h ? 0.0h : x[i];\n"
            "  }\n}\n"
        )

    for step in range(-20, 21):
        threshold = step / 4.0
        source = source_for(threshold)
        recheck = eval_seed(7, "recheck", "init", source)
        search_values = [
            float(contract.reference.generate(contract.instance, case, 7 + 7919 * (index + 1))["x"].reshape(-1)[0])
            for index, case in enumerate(cases)
        ]
        recheck_values = [
            float(contract.reference.generate(contract.instance, case, recheck + 7919 * (index + 1))["x"].reshape(-1)[0])
            for index, case in enumerate(cases)
        ]
        hits = lambda value: abs(value - threshold) < 0.125
        if not any(hits(value) for value in search_values) and any(hits(value) for value in recheck_values):
            return source
    raise AssertionError("no deterministic seed-sensitive ReLU threshold found")


def test_differential_recheck_rejects_a_real_seed_sensitive_candidate(tmp_path):
    """@brief 同一候选真实 full=ok、fresh quick=incorrect，不能成为下一颗搜索种子。"""
    binding = bind_task("relu", "fp16", "small")
    source = _seed_sensitive_relu(binding.contract)
    llm = LLMConfig(
        base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake",
        reasoning_effort="off", use_tools=False, truncation_retry=False,
    )
    off = dict(stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False, plan_ledger=False)
    search = SearchConfig(
        budget_usd=1.0, branches=1, max_rounds=0, eval_reps=3, kernel_agent="pi",
        differential_recheck=True, **off,
    )
    orch = Orchestrator(
        "relu", "fp16", "small", llm, search, run_dir=tmp_path / "run",
        evaluator=V2LocalEvaluator(binding), client=FakeClient([]), log=lambda _s: None, seed=7,
    )
    candidate = asyncio.run(orch._eval(source, "init"))
    assert candidate.is_best is False and orch.best is None
    assert orch.trace[-1]["differential_recheck"] == "incorrect"
    document = json.loads((tmp_path / "run" / "rechecks" / "init.json").read_text())
    assert document["class"] == "incorrect"
    assert document["search_report"]["status"] == "ok"
    assert document["recheck_report"]["status"] == "incorrect"
    for report in (document["search_report"], document["recheck_report"]):
        assert report["build"]["provenance"]["source_sha256"] == document["source_sha256"]


def test_v2_search_end_to_end_with_a_scripted_llm_and_real_metal(tmp_path, monkeypatch):
    """@brief 初稿编译失败 → 修复 → 搜索种子 → confirm_subject 确认 → 读盘发布。

    paper 尺寸的 GEMM 有足够大的真实负载，计时门才能安静地给出 confirmed；small 的
    timing_uncertain 是合法 verdict，由 P2 自己的负例覆盖，不在这里当成成功接受。
    """
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(tmp_path / "catalog"))
    binding = bind_task("gemm", "fp16", "paper")
    correct = _candidate(binding.contract)
    broken = correct.replace("thread_elements", "undefined_thread_elements", 1)
    assert broken != correct and "kernel void gemm_kernel" in broken

    def route(messages, **kw):
        last = messages[-1]["content"]
        if "evaluation of your kernel failed" in last:
            return make_response(f"```metal\n{correct}\n```")
        return make_response(f"```metal\n{broken}\n```")

    llm = LLMConfig(
        base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake",
        reasoning_effort="off", use_tools=False, truncation_retry=False,
    )
    off = dict(stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False, plan_ledger=False)
    search = SearchConfig(
        budget_usd=5.0, branches=1, max_rounds=0, eval_reps=15, kernel_agent="pi",
        differential_recheck=True, **off,
    )
    orch = Orchestrator(
        "gemm", "fp16", "paper", llm, search, run_dir=tmp_path / "run",
        evaluator=V2LocalEvaluator(binding), client=FakeClient(route),
        agent_runner=agent_for_script(route, "gemm_kernel"), log=lambda s: None, seed=7,
    )
    res = asyncio.run(orch.run())
    run_dir = tmp_path / "run"
    assert [c["status"] for c in res.candidates] == ["compile_error", "ok"], res.candidates
    # The corrected seed also goes through a real fresh quick recheck. A pass leaves no negative
    # recheck document and must not replace the cached search report with the quick report.
    assert res.trace[-1]["differential_recheck"] is None
    assert not (run_dir / "rechecks").exists()
    assert res.best_speedup is not None and res.best_speedup > 0
    assert res.hangs == 0
    # 确认门必须真的确认：identity match、fresh seed、计时安静，而不是接受一个不确定出口。
    assert res.confirmed_status == "ok", res.confirmations
    assert res.confirmed_speedup is not None and res.confirmed_speedup > 0
    sidecars = list((run_dir / "confirmations").glob("*.json"))
    assert len(sidecars) == 1
    document = json.loads(sidecars[0].read_text())
    assert document["status"] == "confirmed" and document["identity"]["kind"] == "match"
    assert document["search_report"]["seed"] == 7
    assert document["confirmation_seed"] != 7
    from experiments.common import conclusion_of_run

    result_doc = json.loads((run_dir / "result.json").read_text())
    assert result_doc["task_mode"] == "v2"
    verdict = conclusion_of_run(run_dir, result_doc)
    assert verdict["publication"] == "confirmed", verdict
    assert verdict["speedup"] and verdict["speedup"] > 0
    # 归档走 v2 身份（不再有 v1 get_task 的任务指纹）。
    history = tmp_path / "catalog" / "apple-m4-max" / "gemm" / "fp16" / "paper" / "history.jsonl"
    assert history.is_file()
    record = json.loads(history.read_text().splitlines()[-1])
    assert record["task_mode"] == "v2"
    assert record["task_fingerprint"] == binding.contract.fingerprint()
