"""Hashes of everything the LLM is sent, computed with the *current* code, for the byte-identity regression of ``search.hw_feedback = false``.

``compute()`` only calls the prompt / feedback functions with the arguments that existed before ``hw_feedback`` was added (or with ``hw_feedback=False``
when ``off_kwargs`` says so), so the same function produced ``tests/golden/hw_off.json`` from the code before the feature and checks the code after it.

    uv run python -m tests.hw_golden --record      # rewrite tests/golden/hw_off.json from the current code (only do this on purpose)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
GOLDEN = GOLDEN_DIR / "hw_off.json"
FIXTURES = GOLDEN_DIR / "fixtures"


def sha(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def fixture(name: str) -> tuple[dict, str, dict]:
    """(index entry, kernel source, report json) of a recorded candidate."""
    idx = json.loads((FIXTURES / "index.json").read_text())[name]
    return idx, (FIXTURES / f"{name}.metal").read_text(), json.loads((FIXTURES / f"{name}.report.json").read_text())


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
        out[f"repair|{name}"] = sha(prompts.repair_user("FEEDBACK TEXT", 1, 3))
        for mixed in (False, True):
            out[f"plan|{name}|mixed={int(mixed)}"] = sha(prompts.plan_user(task, "KERNEL SOURCE\n", "FEEDBACK TEXT", ["h1", "h2"], 3, mixed=mixed))
        out[f"patch|{name}"] = sha(prompts.patch_user(task, "KERNEL SOURCE\n", "FEEDBACK TEXT", {"title": "T", "change": "C"}))
    return out


def feedback_texts() -> dict[str, str]:
    from tcenv.evaluate import EvalReport
    from tcenv.report import format_feedback

    out = {}
    idx = json.loads((FIXTURES / "index.json").read_text())
    for name in sorted(idx):
        rep = EvalReport.from_json(json.loads((FIXTURES / f"{name}.report.json").read_text()))
        text = format_feedback(rep)
        out[f"format_feedback|{name}"] = sha(text)
        out[f"format_feedback_cut|{name}"] = sha(text[:6000])
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


def orchestrator_messages(extra_search: dict | None = None) -> dict[str, object]:
    """Every request an orchestrator sends in a scripted two-round search (hash per request, sorted: the three branches run concurrently)."""
    from taic.llm import LLMConfig, SearchConfig
    from taic.orchestrator import Orchestrator

    from .fakes import FakeClient
    from .test_orchestrator import KERNEL, code, llm_script

    out: dict[str, object] = {}
    import tempfile

    for task, kernel, rep_name, plan_mix in (("relu", KERNEL, "relu_best", False), ("relu", KERNEL, "relu_best", True), ("flash_attention", "flash_attention_kernel", "fa_flash_best", False), ("gemm", "gemm_kernel", "gemm_paper_best", True)):
        script = llm_script(code("speed=0.9", kernel), patches={"A": code("speed=1.0", kernel), "B": code("speed=1.3", kernel), "C": code("speed=0.6", kernel)})
        llm_cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
        search = SearchConfig(budget_usd=5.0, branches=3, max_rounds=2, max_repair_tries=2, eval_reps=3, plan_mix=plan_mix, **(extra_search or {}))
        with tempfile.TemporaryDirectory() as td:
            client = FakeClient(script)
            orch = Orchestrator(task, "fp16", "small", llm_cfg, search, run_dir=Path(td) / "run", evaluate_fn=_scripted_eval(rep_name), client=client, log=lambda s: None)
            asyncio.run(orch.run())
            hashes = sorted(sha(json.dumps(c["messages"], sort_keys=True)) for c in client.calls)
        out[f"orchestrator|{task}|plan_mix={int(plan_mix)}"] = {"requests": len(hashes), "sha": sha("\n".join(hashes)), "each": hashes}
    return out


def compute(extra_system: dict | None = None, extra_search: dict | None = None) -> dict:
    return {"system_prompt": system_prompts(extra_system), "user_messages": user_messages(), "feedback": feedback_texts(), "orchestrator": orchestrator_messages(extra_search)}


if __name__ == "__main__":
    if "--record" in sys.argv:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        data = compute()
        GOLDEN.write_text(json.dumps(data, indent=0, sort_keys=True))
        print({k: len(v) for k, v in data.items()}, "->", GOLDEN)
    else:
        print(__doc__)
