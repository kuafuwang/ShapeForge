"""Reusable scripted orchestrator helpers (no test functions, no network, no GPU)."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from taic.kernel_agent import agent_for_script
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator
from tcenv.evaluate import EvalReport
from tcenv.schema import PROTOCOL_VERSION

from .fakes import FakeClient, make_response, make_usage
from .v2_specimen import V2_TASK, CallbackV2Evaluator, bind_v2, v2_pack_registered_task


#: Candidate-owned script name, not a TaskContract field.  Keeping this stable also lets the
#: explicitly legacy suites parse their archived script without sharing any v1 geometry/identity.
KERNEL = "flash_attention_kernel"

def code(marker: str, kernel: str = KERNEL) -> str:
    """@brief 生成一段带标记注释的 metal 代码块，供脚本化评估器解析标记。

    @param marker 写在注释里的标记，例如 speed=1.3。
    @param kernel kernel 函数名。
    @return 带 metal 围栏的源码文本。
    """
    return f"```metal\n// {marker}\nkernel void {kernel}(device const half* x [[buffer(0)]], device half* y [[buffer(1)]]) {{}}\n```"

def stamp_report(report, task: str, dtype: str, size: str, source: str, *, build_kind: str = "native"):
    """Give an ad-hoc test evaluator's report the build record and identity its status needs.

    A hand-built ``EvalReport`` has no build record; schema 4 and the confirmation gate require
    one.  CPU tests stand in for the local Metal build with the native projection fixture.

    @param report 评测报告。
    @param task 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @param source 候选源码。
    @param build_kind ``native``（默认替身）或 ``simulator``。
    @return 同一份报告。
    """
    from tests.support.report_build import attach_identity, native_build, simulated_build

    from tcenv.schema import PROTOCOL_VERSION

    task_obj = v2_pack_registered_task(task, dtype, size)
    report.build = native_build(task_obj, source.encode("utf-8")) if build_kind == "native" else simulated_build(task_obj, source.encode("utf-8"))
    report.target_id = report.target_id or "apple-m4-max"
    report.protocol_version = report.protocol_version or PROTOCOL_VERSION
    report.backend = report.backend or "metal"
    report.backend_version = report.backend_version or "1"
    attach_identity(report, task_obj, backend_id="metal")
    return report


def fake_eval(calls: list, *, v2: bool = True):
    """@brief 用源码里的 "// speed=1.3" 或 "// status=compile_error" 标记伪造一次评估结果。

    每次调用把 (mode, source) 记入 calls，便于断言搜索走了哪条分支；标记值不是 speed 时直接当作状态。
    报告不含任务身份：v2 的 ``CallbackV2Evaluator`` 负责补上当前 build 与 provenance，历史上
    v1 ``LocalEvaluator`` 的归档身份投影随 v1 执行路径一起退役。

    @param calls 收集每次评估 (mode, source) 的列表。
    @param v2 兼容旧调用点的显式拼写；两种取值都返回同一份无身份报告。
    @return 可直接交给 Orchestrator 的 evaluate_fn。
    """
    del v2

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **kw):
        calls.append((mode, source))
        m = re.search(r"// (\w+)=([\w.]+)", source)
        key, val = m.group(1), m.group(2)
        r = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=1, stage="done",
                       target_id="apple-m4-max", protocol_version=PROTOCOL_VERSION,
                       backend="metal", backend_version="1")
        if key == "speed":
            sp = float(val)
            r.status, r.message, r.speedup, r.speedups, r.candidate_us, r.baseline_us = "ok", "ok", sp, {"mlx": sp}, 100.0 / sp, {"mlx": 100.0}
            r.benchmark = {"candidate": {"median_us": 100.0 / sp, "cv": 0.01}, "baselines": {"mlx": {"median_us": 100.0}}, "best_baseline": "mlx"}
        elif val == "compile_error":
            r.status, r.stage, r.message, r.compile = "compile_error", "compile", "Metal compiler rejected the kernel", {"ok": False, "errors": "kernel.metal:3:5: error: boom"}
        elif val == "timeout":
            r.status, r.stage, r.message = "timeout", "timeout", "a single GPU launch ran for more than 3s"
        else:
            r.status, r.message = val, f"{val} happened"
        return r

    return ev

PLAN = '```json\n{"bottleneck":"memory","analysis":"x","plans":[{"title":"A","change":"do A"},{"title":"B","change":"do B"},{"title":"C","change":"do C"}]}\n```'

def llm_script(init: str, patches: dict[str, str] | None = None, fixes: list[str] | None = None, plan: str = PLAN, usage=None):
    """@brief 按最后一条用户消息路由出脚本化回复，模拟翻译、规划、实现与修复四类提示。

    调用记录挂在返回函数的 seen 上，使测试能断言提示顺序；遇到不认识的提示直接失败。

    @param init 首次翻译请求返回的 kernel 文本。
    @param patches 以计划标题为键的实现回复，缺省时回退到 speed=0.5 的 kernel。
    @param fixes 修复轮的回复队列，按顺序消费；用尽后回退到编译失败的 kernel。
    @param plan 规划请求返回的 JSON 文本。
    @param usage 每次回复携带的用量，缺省用 make_usage()。
    @return 接受 messages 的假 LLM 客户端函数。
    @raises AssertionError 收到无法归类的提示时抛出。
    """
    patches = patches or {}
    fixes = list(fixes or [])
    seen = {"calls": []}

    def route(messages, **kw):
        last = messages[-1]["content"]
        seen["calls"].append(last[:60])
        u = usage or make_usage()
        if "Translate this Triton kernel" in last:
            return make_response(init, usage=u)
        if "Diagnose the dominant bottleneck" in last or "Reply with the JSON object only" in last:
            return make_response(plan, usage=u)
        if "Implement exactly this optimization plan" in last:
            title = re.search(r"\*\*(.+?)\*\*", last).group(1)
            return make_response(patches.get(title, code("speed=0.5")), usage=u)
        if "evaluation of your kernel failed" in last or "did not contain" in last:
            return make_response(fixes.pop(0) if fixes else code("status=compile_error"), usage=u)
        raise AssertionError("unexpected prompt: " + last[:100])

    route.seen = seen
    return route

def agent_from_script(script, kernel: str = KERNEL):
    """@brief 把脚本化 LLM 包装成 agent_runner：每次调用一个会话，首次无种子起草，之后依次要求计划 A、B、C。

    @param script 由 llm_script 产出的假客户端。
    @param kernel 期望从回复中抽出的 kernel 名。
    @return 接受 (brief, tag) 的 agent 运行函数，产出 AgentResult；抽不到源码时状态为 failed。
    """
    from taic.kernel_agent import AgentResult
    from taic.parse import extract_metal

    step = {"n": 0}

    def run(brief, tag):
        del tag
        if brief.seedless:
            last = "Translate this Triton kernel"
        else:
            title = "ABC"[step["n"] % 3]
            step["n"] += 1
            last = f"Implement exactly this optimization plan **{title}**"
        resp = script([{"role": "user", "content": last}])
        text = resp.choices[0].message.content or ""
        source = extract_metal(text, kernel)
        return AgentResult(
            source=source, cost_usd=0.01, cost_incomplete=False,
            status="ok" if source else "failed", error="" if source else "no kernel",
        )

    return run

def make_orch(tmp_path: Path, script, calls: list, *, budget=5.0, branches=1, max_rounds=2, max_repair=2, evaluate_fn=None, max_hangs=1, log=None, from_run=None, evaluator=None, expected_identity=None, session_runner=None, workspace_config=None, selfcheck_bridge_factory=None, publishable_site_ids=None, publishable_epochs=None, task_name=V2_TASK, dtype="fp16", size="small", v2_confirmation=None, binding=None, seed=None, **search_kw):
    """@brief 用假客户端与假评估器搭一个 Orchestrator，并默认关掉生产环境开着的附加提示开关。

    测试只在显式传入时打开某个开关，这样提示断言不会被额外段落干扰。

    @param tmp_path pytest 提供的临时目录，run 目录建在其下。
    @param script 由 llm_script 产出的假客户端。
    @param calls 透传给 fake_eval 的评估调用记录。
    @param budget 搜索预算（美元）。
    @param branches 并行分支数。
    @param max_rounds 最大轮数。
    @param max_repair 最大修复次数。
    @param evaluate_fn 自定义评估函数，缺省用 fake_eval(calls)。
    @param max_hangs 允许的挂起次数。
    @param log 日志回调，缺省丢弃。
    @param from_run 续跑来源 run 目录。
    @param evaluator 可选的统一评测执行器；None 时走 evaluate_fn 的本地适配器。
    @param expected_identity 可选的当前身份提供器；真实 evaluate 必须传真实文档，不能用 structural fixture。
    @param task_name/dtype/size 任务定位；需要稳定真实计时的测试可显式选择 paper 负载。
    @param binding 可选的已装配 v2 绑定（测试要用同一份 binding 计算 epoch 身份时传入）。
    @param seed 可选的 run seed；省略时由 Orchestrator 自己抽。
    @param session_runner 可选的单场会话执行器。
    @param workspace_config 可选的显式远端 workspace 配置。
    @param search_kw 覆盖其余 SearchConfig 字段。
    @return 已接好假依赖的 Orchestrator。
    """
    llm_cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False, truncation_retry=False)
    # The production default is on. Scripted tests pin the switches off unless the test passes one, so a prompt assertion stays about the thing it names.
    off = dict(stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False, plan_ledger=False)
    search = SearchConfig(budget_usd=budget, branches=branches, max_rounds=max_rounds, max_repair_tries=max_repair, max_hangs=max_hangs, eval_reps=3, kernel_agent="pi", **{**off, **search_kw})
    fn = evaluate_fn or fake_eval(calls, v2=True)
    binding = binding if binding is not None else bind_v2(task_name, dtype, size)
    if evaluator is None:
        evaluator = CallbackV2Evaluator(binding, fn, expected_identity=expected_identity)
    orch = Orchestrator(
        task_name, dtype, size, llm_cfg, search, run_dir=tmp_path / "run",
        evaluate_fn=fn, client=FakeClient(script), log=log or (lambda s: None), from_run=from_run,
        seed=seed, binding=binding, evaluator=evaluator, session_runner=session_runner,
        workspace_config=workspace_config, selfcheck_bridge_factory=selfcheck_bridge_factory,
        publishable_site_ids=publishable_site_ids, publishable_epochs=publishable_epochs,
        agent_runner=agent_for_script(
            script, KERNEL if task_name == V2_TASK else f"{binding.contract.op_name}_kernel",
        ),
    )
    orch._scripted_v2_confirmation = v2_confirmation
    return orch

def _with_scripted_confirmation(orch, call):
    """Run one test driver without allowing its v2 confirmation to reach a real device."""
    if not orch._v2:
        return call()
    from unittest.mock import patch

    def default_confirmation(_locator, _source, **_kwargs):
        return {
            "instance_key": "scripted-v2",
            "status": "confirmed",
            "reason": None,
            "identity": {"kind": "match"},
            "coverage_kind": "generated",
            "confirmation_seed": 7,
            "confirmation_report": orch.best.report.to_json(),
        }

    handler = getattr(orch, "_scripted_v2_confirmation", None) or default_confirmation
    with patch("tcenv.backends.metal.search_candidate.confirm_candidate", handler):
        return call()


def run(orch):
    """Drive a scripted search, with an explicit CPU-only v2 confirmation double."""
    return _with_scripted_confirmation(orch, lambda: asyncio.run(orch.run()))


def run_prepared(cli, args, orch, snapshot, info, new_budget):
    """Drive the real CLI resume path with the same scoped confirmation double."""
    return _with_scripted_confirmation(
        orch, lambda: cli.run_prepared(args, orch, snapshot, info, new_budget),
    )

def _user_prompts(orch, marker):
    """@brief 收集证据文本中包含标记的会话提示，用于断言某段提示出现与否。

    @param orch 待检查的 Orchestrator。
    @param marker 在证据文本中查找的子串。
    @return 匹配会话的证据文本列表。
    """
    return [brief.evidence for brief in orch._session_briefs if marker in (brief.evidence or "")]
