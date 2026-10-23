"""Reusable scripted orchestrator helpers (no test functions, no network, no GPU)."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from taic.kernel_agent import agent_for_script
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator
from tcenv.evaluate import EvalReport

from .fakes import FakeClient, make_response, make_usage


KERNEL = "relu_kernel"

def code(marker: str, kernel: str = KERNEL) -> str:
    return f"```metal\n// {marker}\nkernel void {kernel}(device const half* x [[buffer(0)]], device half* y [[buffer(1)]]) {{}}\n```"

def fake_eval(calls: list):
    """Parses '// speed=1.3' or '// status=compile_error' markers out of the source."""

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **kw):
        calls.append((mode, source))
        m = re.search(r"// (\w+)=([\w.]+)", source)
        key, val = m.group(1), m.group(2)
        r = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=1, stage="done")
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
    """Route on the last user message."""
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
    """One session per call. The first call is the seedless draft; later calls ask for plan A, then B, then C."""
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

def make_orch(tmp_path: Path, script, calls: list, *, budget=5.0, branches=1, max_rounds=2, max_repair=2, evaluate_fn=None, max_hangs=1, log=None, from_run=None, **search_kw):
    llm_cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False, truncation_retry=False)
    # The production default is on. Scripted tests pin the switches off unless the test passes one, so a prompt assertion stays about the thing it names.
    off = dict(stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False, plan_ledger=False)
    search = SearchConfig(budget_usd=budget, branches=branches, max_rounds=max_rounds, max_repair_tries=max_repair, max_hangs=max_hangs, eval_reps=3, kernel_agent="pi", **{**off, **search_kw})
    return Orchestrator(
        "relu", "fp16", "small", llm_cfg, search, run_dir=tmp_path / "run",
        evaluate_fn=evaluate_fn or fake_eval(calls), client=FakeClient(script), log=log or (lambda s: None), from_run=from_run,
        agent_runner=agent_for_script(script, KERNEL),
    )

def run(orch):
    return asyncio.run(orch.run())

def _user_prompts(orch, marker):
    return [brief.evidence for brief in orch._session_briefs if marker in (brief.evidence or "")]
