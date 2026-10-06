"""Experiment 2 (paper Table 1): run-to-run variation of AI lowering under a fixed search budget.

Several independent runs (fresh model samples, fresh evaluation seeds) per kernel; the reported number of a run is the
*confirmed* speedup of its best kernel (re-measured with a new seed), which is not biased by selecting a maximum of noisy
measurements.

    uv run python -m experiments.repeatability --tasks conv2d,gemm,swiglu --trials 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from observe.publish import job_state

from .common import budget_for, read_result, run_one

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASKS = ("conv2d", "gemm", "swiglu")


def run_repeatability(tasks, dtype, size, out_dir: Path, *, trials=5, budget_scale=1.0, max_rounds=None, branches=None, model=None, resume=True, client_factory=None, evaluate_fn=None, log=print, **overrides) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    table: dict[str, dict] = {}
    tasks = list(tasks)
    for ti, task in enumerate(tasks):
        vals: list[float | None] = []
        costs: list[float] = []
        for t in range(trials):
            rest = [{"task": task, "trial": i + 1} for i in range(t + 1, trials)]
            rest += [{"task": other, "trial": i + 1} for other in tasks[ti + 1 :] for i in range(trials)]
            job_state(out_dir / "state.json", script="repeatability", current={"task": task, "trial": t + 1, "trials": trials}, queue=rest, done=[{"task": k, **v} for k, v in table.items()])
            rd = out_dir / task / f"trial_{t + 1}"
            res = read_result(rd) if resume else None
            if res is None:
                log(f"\n##### {task} trial {t + 1}/{trials}")
                run_one(task, dtype, size, rd, budget=budget_for(task, budget_scale), max_rounds=max_rounds, branches=branches, model=model, client=client_factory() if client_factory else None, evaluate_fn=evaluate_fn, log=log, **overrides)
                res = read_result(rd)
            vals.append(res.get("confirmed_speedup") or res.get("best_speedup"))
            costs.append(res["spent_usd"])
            if res["stop_reason"].startswith("gpu_hang"):
                log("== aborted: GPU hang circuit breaker")
                break
        ok = [v for v in vals if v]
        table[task] = {
            "budget_usd": budget_for(task, budget_scale), "trials": vals, "mean": statistics.fmean(ok) if ok else None,
            "sd": statistics.stdev(ok) if len(ok) > 1 else 0.0, "best": max(ok) if ok else None, "spent_usd": sum(costs),
        }
        (out_dir / "repeatability.json").write_text(json.dumps(table, indent=1))
    job_state(out_dir / "state.json", script="repeatability", current=None, queue=[], done=[{"task": k, **v} for k, v in table.items()], phase="done")
    return table


def markdown(table: dict) -> str:
    n = max((len(v["trials"]) for v in table.values()), default=0)
    head = "| kernel | budget | " + " | ".join(f"trial {i + 1}" for i in range(n)) + " | mean +- SD | best |"
    L = [head, "| --- | ---: | " + " | ".join("---:" for _ in range(n)) + " | ---: | ---: |"]
    for k, v in table.items():
        cells = [f"{x:.3f}x" if x else "-" for x in v["trials"]] + ["-"] * (n - len(v["trials"]))
        ms = f"{v['mean']:.3f} +- {v['sd']:.3f}" if v["mean"] else "-"
        L.append(f"| {k} | ${v['budget_usd']:.2f} | " + " | ".join(cells) + f" | {ms} | {v['best']:.3f}x |" if v["best"] else f"| {k} | ${v['budget_usd']:.2f} | " + " | ".join(cells) + " | - | - |")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--size", default="paper", choices=["paper", "small"])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--budget-scale", type=float, default=1.0)
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out_dir) if a.out_dir else REPO_ROOT / "runs" / f"repeat_{time.strftime('%Y%m%d-%H%M%S')}"
    table = run_repeatability(a.tasks.split(","), a.dtype, a.size, out, trials=a.trials, budget_scale=a.budget_scale, max_rounds=a.max_rounds, model=a.model)
    md = markdown(table)
    (out / "repeatability.md").write_text(md + "\n")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
