"""Hashes of everything the LLM is sent, computed with the *current* code, for the byte-identity regression of ``search.hw_feedback = false``.

``compute()`` calls the prompt and feedback functions with ``hw_feedback`` off (the arguments from before that switch, or ``hw_feedback=False``).
``tests/support/golden/metal/hw_off.json`` is that output. Regenerate it on purpose when a prompt change other than the switch should become the new off baseline.
A request hash covers the body ``chat.completions.create`` is given, not only ``messages``: model, max_tokens, reasoning_effort, extra_body, and tools.
Two extra searches record the call sites the parameter sample does not: a length-capped draft retried once at the truncation effort, and a search whose planner and patch requests use different efforts.
``REQUEST_KEYS`` is the whitelist of keys ``LLM._params`` may produce; the test fails when the two sets diverge.
Feedback entries store the text. System prompts stay hashes. Orchestrator entries are ordered by the request label, so round 1 and round 2 cannot trade places unnoticed.
``_meta`` (when, which commit, why) is written by ``--record`` and is not part of ``compute()``.

    uv run python -m tests.support.hw_golden --record --reason "why the off baseline should change"
    # more than FORCE_AFTER keys also needs --force
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "metal"
GOLDEN = GOLDEN_DIR / "hw_off.json"
FIXTURES = GOLDEN_DIR / "fixtures"
# Keys ``LLM._params`` is allowed to put on the wire. ``messages`` is added by ``chat`` and is hashed with them.
REQUEST_KEYS = frozenset({"model", "max_tokens", "reasoning_effort", "extra_body", "tools"})
_NOT_SENT = frozenset({"stream", "_label"})
FORCE_AFTER = 40
_LABEL = contextvars.ContextVar("hw_golden_label", default="")


def sha(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def dumps(data: dict, meta: dict) -> str:
    """The only spelling ``--record`` may write and the format test may accept."""
    return json.dumps({"_meta": meta, **data}, indent=0, sort_keys=True)


def fixture_digest(name: str) -> str:
    """sha256 of the metal source and the report, so a fixture edit cannot pass as a prompt change."""
    metal = (FIXTURES / f"{name}.metal").read_bytes()
    report = (FIXTURES / f"{name}.report.json").read_bytes()
    return hashlib.sha256(metal + b"\0" + report).hexdigest()


def fixture(name: str) -> tuple[dict, str, dict]:
    """(index entry, kernel source, report json) of a recorded candidate."""
    idx = json.loads((FIXTURES / "index.json").read_text())[name]
    return idx, (FIXTURES / f"{name}.metal").read_text(), json.loads((FIXTURES / f"{name}.report.json").read_text())


def param_keys() -> set[str]:
    """Every key ``LLM._params`` emits across effort, tools, and max_tokens."""
    from taic.llm import CostLedger, LLM, LLMConfig
    from taic.tools import LOOKUP_TOOL, QUICK_CHECK_TOOL

    tools = [LOOKUP_TOOL, QUICK_CHECK_TOOL]
    keys: set[str] = set()
    for effort in (None, "off", "low", "high"):
        for extra in ({}, {"top_p": 1}):
            cfg = LLMConfig(base_url="http://x", api_key="k", model="m", reasoning_effort="high", extra_body=dict(extra))
            llm = LLM(cfg, CostLedger(1.0), client=object())
            for tool_arg in (None, tools):
                for limit in (None, 1024):
                    keys |= set(llm._params(tool_arg, limit, effort))
    return keys


def _request_body(call: dict) -> dict:
    return {key: value for key, value in call.items() if key not in _NOT_SENT}


def task_grid() -> list[tuple[str, str, str]]:
    from tasks import SEMANTIC_NAMES, list_tasks

    names = list(list_tasks(True)) + list(SEMANTIC_NAMES) + ["flash_attention@64x32t256", "gemm@128x64x32t256"]
    return [(n, dt, "paper") for n in names for dt in ("fp16", "bf16")]


def system_prompts(extra: dict | None = None) -> dict[str, str]:
    from tasks import get_task

    from taic import prompts

    out = {}
    for name, dt, size in task_grid():
        try:
            task = get_task(name, dt, size)
        except Exception:  # noqa: BLE001 - a combination that does not exist is simply not recorded
            continue
        for tools in (False, True):
            for pe in (False, True):
                out[f"{name}|{dt}|{size}|tools={int(tools)}|pe={int(pe)}"] = sha(prompts.system_prompt(task, use_tools=tools, profiler_evidence=pe, **(extra or {})))
    return out


def user_messages() -> dict[str, str]:
    from tasks import get_task

    from taic import prompts

    out = {}
    for name, dt, size in [(n, d, s) for n, d, s in task_grid() if d == "fp16"]:
        try:
            task = get_task(name, dt, size)
        except Exception:  # noqa: BLE001
            continue
        out[f"initial|{name}"] = sha(prompts.initial_user(task))
        out[f"repair|{name}"] = sha(prompts.repair_user("FEEDBACK TEXT", 1, 3, source_fence="metal"))
        for mixed in (False, True):
            out[f"plan|{name}|mixed={int(mixed)}"] = sha(prompts.plan_user(task, "KERNEL SOURCE\n", "FEEDBACK TEXT", ["h1", "h2"], 3, mixed=mixed, source_fence="metal"))
        out[f"patch|{name}"] = sha(prompts.patch_user(task, "KERNEL SOURCE\n", "FEEDBACK TEXT", {"title": "T", "change": "C"}, source_fence="metal"))
    return out


def feedback_texts() -> dict[str, str]:
    from tcenv.evaluate import EvalReport
    from tcenv.report import format_feedback

    out = {}
    idx = json.loads((FIXTURES / "index.json").read_text())
    for name in sorted(idx):
        rep = EvalReport.from_json(json.loads((FIXTURES / f"{name}.report.json").read_text()))
        text = format_feedback(rep)
        out[f"format_feedback|{name}"] = text
        out[f"format_feedback_cut|{name}"] = text[:6000]
    return out


def outbound_requests() -> dict[str, str]:
    """Complete request bodies, including the tool schema and the three effort settings, with the messages held fixed."""
    from taic.llm import CostLedger, LLM, LLMConfig
    from taic.tools import LOOKUP_TOOL, QUICK_CHECK_TOOL

    tools = [LOOKUP_TOOL, QUICK_CHECK_TOOL]
    messages = [{"role": "user", "content": "PING"}]
    specs = (
        ("effort=off|tools=0|max=default|model=fake", "fake", "off", None, None),
        ("effort=high|tools=0|max=default|model=fake", "fake", "high", None, None),
        ("effort=low|tools=0|max=default|model=fake", "fake", "low", None, None),
        ("effort=high|tools=1|max=default|model=fake", "fake", "high", tools, None),
        ("effort=off|tools=1|max=1024|model=other", "other", "off", tools, 1024),
    )
    out = {}
    for name, model, effort, tool_arg, limit in specs:
        cfg = LLMConfig(base_url="http://x", api_key="k", model=model, reasoning_effort="high")
        body = {"messages": messages, **LLM(cfg, CostLedger(1.0), client=object())._params(tool_arg, limit, effort)}
        out[name] = sha(json.dumps(body, sort_keys=True))
    return out


def _scripted_eval(report_name: str):
    """An evaluator that returns the recorded report of ``report_name``, with the speedup taken from a ``// speed=`` marker in the source."""
    import re

    from tcenv.evaluate import EvalReport

    base = json.loads((FIXTURES / f"{report_name}.report.json").read_text())

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **kw):
        r = EvalReport.from_json(base)
        m = re.search(r"// speed=([\d.]+)", source)
        if m:
            sp = float(m.group(1))
            r.speedup, r.speedups, r.candidate_us = sp, {k: sp for k in (r.speedups or {"mlx": 1.0})}, 100.0 / sp
        return r

    return ev


def _bind_labels(client):
    """Remember ``chat``'s label on the recorded call without putting it on the wire."""
    from taic import llm as llm_mod

    original_create = client.chat.completions.create
    original_chat = llm_mod.LLM.chat

    async def create(**kw):
        result = await original_create(**kw)
        client.calls[-1]["_label"] = _LABEL.get()
        return result

    async def chat(self, messages, *, tools=None, label="", max_tokens=None, reasoning_effort=None):
        token = _LABEL.set(label)
        try:
            return await original_chat(self, messages, tools=tools, label=label, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
        finally:
            _LABEL.reset(token)

    client.chat.completions.create = create
    llm_mod.LLM.chat = chat
    return llm_mod, original_chat


def _length_then_kernel(script):
    """First draft hits the length cap with no kernel. The retry asks the same question again."""
    from .fakes import make_response

    state = {"truncated": False}

    def route(messages, **kw):
        last = messages[-1]["content"] if messages else ""
        if not state["truncated"] and "Translate this Triton kernel" in last:
            state["truncated"] = True
            return make_response("", finish="length")
        return script(messages, **kw)

    return route


def _orch_key(task: str, plan_mix: bool, knobs: dict) -> str:
    parts = [f"orchestrator|{task}|plan_mix={int(plan_mix)}"]
    if knobs.get("truncation_retry"):
        parts.append("truncation_retry=1")
    if knobs.get("plan_reasoning_effort"):
        parts.append(f"plan_effort={knobs['plan_reasoning_effort']}")
        parts.append(f"patch_effort={knobs['patch_reasoning_effort']}")
    return "|".join(parts)


def _check_knobs(knobs: dict, calls: list[dict]) -> None:
    """The recorded calls are the ones the search actually sent, not a standalone ``_params`` sample."""
    if knobs.get("truncation_retry"):
        inits = [call for call in calls if call.get("_label") == "init"]
        if len(inits) != 2 or "reasoning_effort" in inits[0] or inits[1].get("reasoning_effort") != "low":
            raise RuntimeError("the length-capped draft was not retried once at effort low")
    plan_effort = knobs.get("plan_reasoning_effort")
    patch_effort = knobs.get("patch_reasoning_effort")
    if plan_effort or patch_effort:
        for call in calls:
            label = str(call.get("_label") or "")
            want = plan_effort if label.endswith(".plan") or ".plan." in label else patch_effort
            if call.get("reasoning_effort") != want:
                raise RuntimeError(f"{label} was sent with effort {call.get('reasoning_effort')}, not {want}")


def orchestrator_messages(extra_search: dict | None = None) -> dict[str, object]:
    """The task contracts one scripted search hands to hmz, in order.

    The search does not send chat requests. Each entry hashes the contract text.
    """
    from taic.llm import LLMConfig, SearchConfig
    from taic.orchestrator import Orchestrator

    from .fakes import FakeClient
    from ..test_orchestrator import KERNEL, code, llm_script

    out: dict[str, object] = {}
    import tempfile

    cases = (
        ("relu", KERNEL, "relu_best", False, {}),
        ("relu", KERNEL, "relu_best", True, {}),
        ("flash_attention", "flash_attention_kernel", "fa_flash_best", False, {}),
        ("gemm", "gemm_kernel", "gemm_paper_best", True, {}),
    )
    for task, kernel, rep_name, plan_mix, knobs in cases:
        script = llm_script(code("speed=0.9", kernel), patches={"A": code("speed=1.0", kernel)})
        llm_cfg = LLMConfig(
            base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False,
            truncation_retry=bool(knobs.get("truncation_retry")),
            plan_reasoning_effort=knobs.get("plan_reasoning_effort"),
            patch_reasoning_effort=knobs.get("patch_reasoning_effort"),
        )
        # This hash is the switches-off prompt. The production default is on; the off path is what the golden file records.
        off = dict(stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False, plan_ledger=False)
        search = SearchConfig(budget_usd=5.0, branches=1, max_rounds=1, max_repair_tries=0, eval_reps=3, plan_mix=plan_mix, kernel_agent="pi", **{**off, **(extra_search or {})})
        with tempfile.TemporaryDirectory() as td:
            client = FakeClient(script)
            llm_mod, original_chat = _bind_labels(client)
            previous_catalog = os.environ.get("TAIC_CATALOG_ROOT")
            os.environ["TAIC_CATALOG_ROOT"] = str(Path(td) / "catalog")
            try:
                orch = Orchestrator(task, "fp16", "small", llm_cfg, search, run_dir=Path(td) / "run", evaluate_fn=_scripted_eval(rep_name), client=client, log=lambda s: None)
                asyncio.run(orch.run())
            finally:
                if previous_catalog is None:
                    os.environ.pop("TAIC_CATALOG_ROOT", None)
                else:
                    os.environ["TAIC_CATALOG_ROOT"] = previous_catalog
                llm_mod.LLM.chat = original_chat
        from taic.kernel_agent import render_contract

        each = []
        for index, brief in enumerate(orch._session_briefs):
            # The contract names this session's absolute paths. The golden tracks the wording, not the temp directory.
            text = render_contract(brief)
            if brief.work_dir:
                work = Path(brief.work_dir)
                text = text.replace(str(work), "<work>")
                # The self-check log lives next to agent_work, still inside this run.
                text = text.replace(str(work.parent.parent), "<run>")
            text = text.replace(str(Path(brief.repo).resolve()), "<repo>")
            each.append({"label": f"{index}", "sha": sha(text)})
        joined = "\n".join(f"{row['label']} {row['sha']}" for row in each)
        out[_orch_key(task, plan_mix, knobs)] = {"requests": len(each), "sha": sha(joined), "each": each}
    return out


def compute(extra_system: dict | None = None, extra_search: dict | None = None) -> dict:
    return {
        "system_prompt": system_prompts(extra_system),
        "user_messages": user_messages(),
        "feedback": feedback_texts(),
        "outbound": outbound_requests(),
        "orchestrator": orchestrator_messages(extra_search),
    }


def changed_keys(got: dict, want: dict) -> list[str]:
    """Section:key pairs whose recorded value differs. ``_meta`` is provenance, not a prompt."""
    lines = []
    sections = (set(want) | set(got)) - {"_meta"}
    for sec in sorted(sections):
        left = got.get(sec) or {}
        right = want.get(sec) or {}
        if not isinstance(left, dict) or not isinstance(right, dict):
            if left != right:
                lines.append(sec)
            continue
        for key in sorted(set(left) | set(right)):
            if left.get(key) != right.get(key):
                lines.append(f"{sec}:{key}")
    return lines


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=GOLDEN_DIR.parents[3], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _reason(argv: list[str]) -> str:
    if "--reason" not in argv:
        return ""
    index = argv.index("--reason")
    if index + 1 >= len(argv):
        return ""
    return argv[index + 1]


if __name__ == "__main__":
    if "--record" in sys.argv:
        reason = _reason(sys.argv)
        if not reason.strip():
            sys.exit("refusing to record without --reason \"why this baseline should change\"")
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        data = compute()
        previous = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        diff = changed_keys(data, previous)
        print(f"{len(diff)} keys differ")
        for line in diff:
            print(line)
        if len(diff) > FORCE_AFTER and "--force" not in sys.argv:
            sys.exit(f"refusing to record {len(diff)} changes without --force")
        meta = {"recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"), "git_commit": _git_commit(), "reason": reason}
        GOLDEN.write_text(dumps(data, meta))
        print({k: len(v) for k, v in data.items()}, "->", GOLDEN)
    else:
        print(__doc__)
