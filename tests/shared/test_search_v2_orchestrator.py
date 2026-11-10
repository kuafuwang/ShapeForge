"""v2 搜索循环（假评测器，不上 GPU）：seed 纪律、run_config、结果与 resume 装配。"""

from __future__ import annotations

import asyncio
import json
import re

from taic.kernel_agent import agent_for_script
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator
from taic.task_binding import bind_task
from taic.v2_evaluation import V2LocalEvaluator
from tcenv.evaluation import EvaluationResult
from tcenv.evaluate import EvalReport
from tcenv.schema import PROTOCOL_VERSION

from ..support.fakes import FakeClient, make_response

BROKEN = "// status=compile_error\nkernel void gemm_kernel() {}\n"
GOOD = "// speed=2.0\nkernel void gemm_kernel() {}\n"


class FakeV2Evaluator(V2LocalEvaluator):
    """@brief 与 V2LocalEvaluator 同协议，但按源码标记出报告（不编译、不上 GPU）。"""

    def __init__(self, binding, calls):
        super().__init__(binding)
        self.calls = calls

    def _evaluate(self, request):
        self.calls.append(request)
        match = re.search(r"// (\w+)=([\w.]+)", request.source)
        key, value = match.group(1), match.group(2)
        report = EvalReport(
            task=request.task, dtype=request.dtype, size=request.size, mode=request.mode,
            seed=request.seed or 0, stage="done", target_id=request.target_id,
            protocol_version=PROTOCOL_VERSION, backend="metal", backend_version="1",
        )
        if key == "speed":
            speedup = float(value)
            report.status, report.message = "ok", "ok"
            report.speedup, report.speedups = speedup, {"mlx": speedup}
            report.candidate_us, report.baseline_us = 100.0 / speedup, {"mlx": 100.0}
            report.correctness = {"ok": True}
            report.benchmark = {
                "candidate": {"median_us": 100.0 / speedup, "cv": 0.01},
                "baselines": {"mlx": {"median_us": 100.0}}, "best_baseline": "mlx",
            }
        else:
            report.status, report.stage = "compile_error", "compile"
            report.message = "Metal compiler rejected the kernel"
            report.compile = {"ok": False, "errors": "kernel.metal:3:5: error: boom"}
        # schema-4 闭包要求每种 status 都有对应的 build 记录；用 v1 夹具任务投影出合法形状
        # （这只是结构替身，身份不参与本测试的断言）。
        from tasks import get_task
        from tests.support.report_build import native_build, simulated_compile_failure
        from tests.support.v1_tasks import V1_TASK

        task_obj = get_task(V1_TASK, "fp16", "small")
        if report.status == "ok":
            report.build = native_build(task_obj, request.source.encode("utf-8"))
        else:
            report.build = simulated_compile_failure(task_obj, b"// COMPILE_FAIL\n" + request.source.encode("utf-8"))
        return EvaluationResult(report=report, provenance=self._provenance(request))


def _run(tmp_path):
    calls = []

    def route(messages, **kw):
        last = messages[-1]["content"]
        if "evaluation of your kernel failed" in last:
            return make_response(f"```metal\n{GOOD}```")
        return make_response(f"```metal\n{BROKEN}```")

    binding = bind_task("gemm", "fp16", "small")
    llm = LLMConfig(
        base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake",
        reasoning_effort="off", use_tools=False, truncation_retry=False,
    )
    off = dict(stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False, plan_ledger=False)
    search = SearchConfig(
        budget_usd=5.0, branches=1, max_rounds=0, eval_reps=3, kernel_agent="pi",
        correctness_only=True, **off,
    )
    orch = Orchestrator(
        "gemm", "fp16", "small", llm, search, run_dir=tmp_path / "run",
        evaluator=FakeV2Evaluator(binding, calls), client=FakeClient(route),
        agent_runner=agent_for_script(route, "gemm_kernel"), log=lambda s: None,
    )
    result = asyncio.run(orch.run())
    return result, calls, tmp_path / "run"


def test_the_v2_search_uses_the_portfolio_seed_and_writes_v2_identities(tmp_path):
    result, calls, run_dir = _run(tmp_path)
    assert [c["status"] for c in result.candidates] == ["compile_error", "ok"]
    assert result.best_speedup == 2.0
    # portfolio 纪律：搜索期每个候选都是 SEARCH_SEED=7，fresh 只属于确认。
    assert calls and all(request.seed == 7 for request in calls)
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["task_mode"] == "v2"
    assert config["track"]["track_id"] == "msl-single"
    assert len(config["task_pack_digest"]) == 64
    assert len(config["presentation_sha256"]) == 64
    document = json.loads((run_dir / "result.json").read_text())
    assert document["task_mode"] == "v2"
    assert document["track"]["delivery"] == "single_source"
    # correctness_only：没有确认，也没有可发布的加速比。
    assert result.confirmed_status != "ok"


def test_v2_init_samples_keeps_the_fastest_draft_on_the_portfolio_seed(tmp_path):
    """@brief v2 的三份初稿都用搜索种子 7，最快的一份留下，方向能从 plan.txt 读回。"""
    calls = []
    speeds = iter([1.0, 3.0, 2.0])

    def route(messages, **kw):
        speed = next(speeds)
        return make_response(f"```metal\n// speed={speed}\nkernel void gemm_kernel() {{}}\n```")

    binding = bind_task("gemm", "fp16", "small")
    llm = LLMConfig(
        base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake",
        reasoning_effort="off", use_tools=False, truncation_retry=False,
    )
    off = dict(stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False, plan_ledger=False)
    search = SearchConfig(
        budget_usd=5.0, branches=1, max_rounds=0, eval_reps=3, kernel_agent="pi",
        correctness_only=True, init_samples=3, **off,
    )
    orch = Orchestrator(
        "gemm", "fp16", "small", llm, search, run_dir=tmp_path / "run",
        evaluator=FakeV2Evaluator(binding, calls), client=FakeClient(route),
        agent_runner=agent_for_script(route, "gemm_kernel"), log=lambda _line: None,
    )
    inner = orch.agent_runner

    def wrapped(brief, tag):
        result = inner(brief, tag)
        result.notes = f"direction: family-{brief.label}\n"
        return result

    orch.agent_runner = wrapped
    result = asyncio.run(orch.run())
    assert result.best_label == "init2"
    assert result.best_speedup == 3.0
    assert calls and all(request.seed == 7 for request in calls)
    from taic.resume import _read_candidates

    restored = {row["label"]: row["plan"] for row in _read_candidates(tmp_path / "run", [])}
    assert restored["init2"] == "family-init2"


def test_a_finished_v2_run_still_assembles_for_a_resume_dry_run(tmp_path):
    _result, _calls, run_dir = _run(tmp_path)
    from taic.cli import main

    code = main(["resume", str(run_dir), "--dry-run", "--budget", "10", "--lift-count-limits", "--max-rounds", "2"])
    assert code == 0
