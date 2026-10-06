"""Shared helpers of the experiment scripts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

from taic.llm import load_config
from taic.orchestrator import Orchestrator, RunResult

# Per-kernel USD budgets. The paper spent $1.5 on elementwise and $5-12 on GEMM/conv candidates with a frontier model;
# deepseek-flash costs roughly 1/20 of that per token, so the same *shape* of budgets is much smaller in dollars.
BUDGETS = {
    "sigmoid": 0.5, "relu": 0.5, "silu": 0.5, "swiglu": 0.6, "rmsnorm": 0.8, "softmax": 1.2, "reduction_sum": 0.8,
    "rope": 0.8, "gemv": 1.0, "gemm": 2.5, "gemm_silu": 2.5, "conv2d": 3.0, "flash_attention": 3.0, "mamba2_chunk_scan": 3.0,
}


def budget_for(task: str, scale: float = 1.0) -> float:
    return BUDGETS.get(task, 1.0) * scale


def run_one(
    task: str,
    dtype: str,
    size: str,
    run_dir: Path,
    *,
    budget: float | None = None,
    branches: int | None = None,
    max_rounds: int | None = None,
    model: str | None = None,
    config_path: str | None = None,
    client: Any = None,
    evaluate_fn: Callable | None = None,
    log: Callable[[str], None] = print,
    use_tools: bool | None = None,
    **search_overrides,
) -> RunResult:
    cfg, search = load_config(config_path, model=model, budget_usd=budget, branches=branches, max_rounds=max_rounds, **search_overrides)
    if use_tools is not None:
        cfg.use_tools = use_tools
    orch = Orchestrator(task, dtype, size, cfg, search, run_dir=run_dir, client=client, evaluate_fn=evaluate_fn, log=log)
    return asyncio.run(orch.run())


def read_result(run_dir: Path) -> dict | None:
    p = Path(run_dir) / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def read_confirm(run_dir: Path) -> dict | None:
    p = Path(run_dir) / "confirm_report.json"
    return json.loads(p.read_text()) if p.exists() else None


def summarize(res: dict, confirm: dict | None, naive: dict | None = None) -> dict:
    """One row of the suite table."""
    row = {
        "task": res["task"], "dtype": res["dtype"], "model": res["model"], "stop_reason": res["stop_reason"], "rounds": res["rounds"],
        "evaluations": res["evaluations"], "spent_usd": res["spent_usd"], "budget_usd": res["budget_usd"],
        "best_speedup": res.get("best_speedup"), "confirmed_speedup": res.get("confirmed_speedup"),
        "candidate_us": res.get("candidate_us"), "baseline_us": res.get("baseline_us") or {},
        "profiles": len(res.get("profiles") or []), "spilled_bytes_seen": max([p.get("spilled_bytes") or 0 for p in res.get("profiles") or []], default=0),
    }
    bl = row["baseline_us"]
    row["best_baseline"] = min(bl, key=bl.get) if bl else None
    cb = (confirm or {}).get("compiler_baseline") or {}
    row["naive_speedup"] = (naive or {}).get("speedup")
    row["vs_triton_msl"] = cb.get("speedup")
    row["triton_msl_us"] = cb.get("estimated_us")
    return row


def naive_reference(task: str, dtype: str, size: str, cache_dir: Path, *, evaluate_fn: Callable | None = None) -> dict:
    """Evaluate the task's hand-written *naive* kernel under the same protocol. Its speedup over the baselines is the reference
    point that makes weak baselines visible: if a naive kernel already beats MLX/MPS (as for unfused ops such as Lion), part of
    any AI-lowering speedup is baseline weakness, not compiler skill."""
    from tasks import get_task
    from tcenv.evaluate import evaluate

    p = Path(cache_dir) / f"naive_{task}_{dtype}.json"
    if p.exists():
        return json.loads(p.read_text())
    t = get_task(task, dtype, size)
    rep = (evaluate_fn or evaluate)(task, dtype, size, t.naive_msl, mode="full", seed=None, reps=15)
    out = {"status": rep.status, "speedup": rep.speedup, "candidate_us": rep.candidate_us, "baseline_us": rep.baseline_us}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    return out
