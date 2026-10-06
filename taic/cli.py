"""``taic`` command line: run the AI compiler on a task."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def cmd_run(a: argparse.Namespace) -> int:
    from .llm import load_config
    from .orchestrator import Orchestrator

    cfg, search = load_config(a.config, model=a.model, preset=a.preset, budget_usd=a.budget, branches=a.branches, max_rounds=a.max_rounds)
    if a.no_tools:
        cfg.use_tools = False
    if a.profile_best:
        search.profile_best = True
    if a.profile_poor:
        search.profile_poor = True
    if a.plan_mix:
        search.plan_mix = True
    if a.init_samples:
        search.init_samples = a.init_samples
    if a.patience:
        search.patience = a.patience
    if a.effort:
        cfg.reasoning_effort = a.effort
    orch = Orchestrator(a.task, a.dtype, a.size, cfg, search, run_dir=Path(a.run_dir) if a.run_dir else None)
    res = asyncio.run(orch.run())
    print(json.dumps({k: getattr(res, k) for k in ("task", "dtype", "model", "best_speedup", "confirmed_speedup", "spent_usd", "stop_reason", "run_dir")}, indent=2))
    return 0 if res.best_speedup else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="taic", description="Triton AI Compiler for Apple GPUs (Triton -> MSL)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="lower one task with the LLM agent")
    p.add_argument("task")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--size", default="paper", choices=["paper", "small", "plan"])
    p.add_argument("--config", default=None, help="one config file, used as it is (no preset is layered on it)")
    p.add_argument("--preset", default=None, help="settings layered over taic.toml: 'paper' (default, taic.paper.toml: the paper's $15 budget) or 'base' (taic.toml alone); also $TAIC_PRESET")
    p.add_argument("--model", default=None)
    p.add_argument("--budget", type=float, default=None, help="USD budget for this kernel (default: the preset's budget_usd, the paper's $15)")
    p.add_argument("--branches", type=int, default=None)
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--effort", default=None, choices=["off", "low", "high", "max"])
    p.add_argument("--no-tools", action="store_true")
    p.add_argument("--profile-best", action="store_true", help="xctrace profile of the winning kernel after the search")
    p.add_argument("--profile-poor", action="store_true", help="trace a round's seed with Instruments when it is slower than the best baseline and give the findings to the model")
    p.add_argument("--plan-mix", action="store_true", help="the last plan of every round is a local, low-risk change; the others are structural")
    p.add_argument("--init-samples", type=int, default=None, help="sample this many initial kernels in parallel; the best eligible one seeds the search")
    p.add_argument("--patience", type=int, default=None, help="stop after this many rounds without a new best (default 4)")
    p.add_argument("--run-dir", default=None)
    p.set_defaults(fn=cmd_run)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
