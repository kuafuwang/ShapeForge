"""v2 搜索的真实 Metal 垂直切片：脚本化模型（不花钱）+ 真 evaluate_task + P2 确认门。"""

from __future__ import annotations

import asyncio
import json

import pytest

from experiments.contract_sensitivity import gemm_source, parse
from taic.kernel_agent import agent_for_script
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator
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
    search = SearchConfig(budget_usd=5.0, branches=1, max_rounds=0, eval_reps=15, kernel_agent="pi", **off)
    orch = Orchestrator(
        "gemm", "fp16", "paper", llm, search, run_dir=tmp_path / "run",
        evaluator=V2LocalEvaluator(binding), client=FakeClient(route),
        agent_runner=agent_for_script(route, "gemm_kernel"), log=lambda s: None,
    )
    res = asyncio.run(orch.run())
    run_dir = tmp_path / "run"
    assert [c["status"] for c in res.candidates] == ["compile_error", "ok"], res.candidates
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
