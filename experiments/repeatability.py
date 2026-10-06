"""Experiment 2 (paper Table 1): run-to-run variation of AI lowering under a fixed search budget.

Several independent runs (fresh model samples, fresh evaluation seeds) per kernel; the reported number of a run is the
*confirmed* speedup of its best kernel (re-measured with a new seed), which is not biased by selecting a maximum of noisy
measurements.

    uv run python -m experiments.repeatability --tasks conv2d,gemm,swiglu --trials 5

Budgets are the paper's Table 1: $15 per run for conv2d, gemm and flash_attention, $1 for swiglu (``--budget USD`` gives every run the same
budget, ``--budget-scale`` multiplies the table). The default three tasks, five trials each: at most 5 x (15 + 15 + 1) = $155, and with
flash_attention added $230; runs that converge early (``patience``) cost less. ``--max-total-usd`` is an optional cap on the whole
experiment (default off): a trial is skipped, and logged as such, if its budget plus what the experiment has spent so far would exceed it.
"""

# 论文实验 2（Table 1）：同一道题、同一笔预算独立重复多次，报告每次的确认加速比，再算均值和离散。
# 每次都是新的运行目录。编排器只在内存里保存当前最优核，不会把上一次 trial 的 kernel 当成下一次的种子。

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from observe.publish import job_state

from .common import budget_for, fits_cap, read_result, run_one

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASKS = ("conv2d", "gemm", "swiglu")


# 目录是 `任务/trial_N`。接着上次的最优核会把多次试验收成一条更长的搜索，均值和标准差就不再是运行之间的波动。
# --resume 只认这一次 trial 自己的 result.json；上一格的结果不会被拷进来当起点。
def run_repeatability(tasks, dtype, size, out_dir: Path, *, trials=5, budget_scale=1.0, budget=None, max_total_usd=None, max_rounds=None, branches=None, model=None, resume=True, client_factory=None, evaluate_fn=None, log=print, **overrides) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    table: dict[str, dict] = {}
    tasks = list(tasks)
    for ti, task in enumerate(tasks):
        vals: list[float | None] = []
        costs: list[float] = []
        # table1=True：swiglu 是 $1，conv2d / gemm / flash_attention 是 $15，scale 只乘这笔钱。同一道题的每次 trial 共用这笔预算，而不是把预算拆开。
        b = budget if budget is not None else budget_for(task, budget_scale, table1=True)
        for t in range(trials):
            rest = [{"task": task, "trial": i + 1} for i in range(t + 1, trials)]
            rest += [{"task": other, "trial": i + 1} for other in tasks[ti + 1 :] for i in range(trials)]
            job_state(out_dir / "state.json", script="repeatability", current={"task": task, "trial": t + 1, "trials": trials}, queue=rest, done=[{"task": k, **v} for k, v in table.items()])
            rd = out_dir / task / f"trial_{t + 1}"
            res = read_result(rd) if resume else None
            if res is None:
                spent = sum(v["spent_usd"] for v in table.values()) + sum(costs)
                if not fits_cap(spent, b, max_total_usd):
                    # 放不下就停这道题剩下的 trial。下一道题仍会按它自己的预算再问一次上限（swiglu 的 $1 可能还放得下）。
                    log(f"(skip {task} trial {t + 1}/{trials}: its ${b:.2f} budget does not fit under --max-total-usd ${max_total_usd:.2f}; the experiment has spent ${spent:.2f})")
                    break
                log(f"\n##### {task} trial {t + 1}/{trials} (budget ${b:.2f})")
                # 提供了 client_factory 时，每个新 trial 调用一次。换的是客户端，不是上一格的核。
                run_one(task, dtype, size, rd, budget=b, max_rounds=max_rounds, branches=branches, model=model, client=client_factory() if client_factory else None, evaluate_fn=evaluate_fn, log=log, **overrides)
                res = read_result(rd)
            vals.append(res.get("confirmed_speedup") or res.get("best_speedup"))
            costs.append(res["spent_usd"])
            if res["stop_reason"].startswith("gpu_hang"):
                # 只停这道题剩下的 trial。已经记下的样本仍留在表里，下一道题还会继续。
                log("== aborted: GPU hang circuit breaker")
                break
        # 均值和标准差只用记下了加速比的那些 trial。被上限或挂卡截掉、还没写入 vals 的次数不参与。
        ok = [v for v in vals if v]
        table[task] = {
            "budget_usd": b, "trials": vals, "mean": statistics.fmean(ok) if ok else None,
            "sd": statistics.stdev(ok) if len(ok) > 1 else 0.0, "best": max(ok) if ok else None, "spent_usd": sum(costs),
        }
        (out_dir / "repeatability.json").write_text(json.dumps(table, indent=1))
    job_state(out_dir / "state.json", script="repeatability", current=None, queue=[], done=[{"task": k, **v} for k, v in table.items()], phase="done")
    return table


# 不同题实际留下的 trial 数可能不一样。表头按最多的那一行开列，较短的一行用横杠补齐。
def markdown(table: dict) -> str:
    n = max((len(v["trials"]) for v in table.values()), default=0)
    head = "| kernel | budget | " + " | ".join(f"trial {i + 1}" for i in range(n)) + " | mean +- SD | best |"
    L = [head, "| --- | ---: | " + " | ".join("---:" for _ in range(n)) + " | ---: | ---: |"]
    for k, v in table.items():
        cells = [f"{x:.3f}x" if x else "-" for x in v["trials"]] + ["-"] * (n - len(v["trials"]))
        ms = f"{v['mean']:.3f} +- {v['sd']:.3f}" if v["mean"] else "-"
        L.append(f"| {k} | ${v['budget_usd']:.2f} | " + " | ".join(cells) + f" | {ms} | {v['best']:.3f}x |" if v["best"] else f"| {k} | ${v['budget_usd']:.2f} | " + " | ".join(cells) + " | - | - |")
    return "\n".join(L)


# 默认三道题是 Table 1 里预算不同的那一组：conv2d、gemm 为 $15，swiglu 为 $1。flash_attention 要在 --tasks 里点名才跑。
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--size", default="paper", choices=["paper", "small"])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--budget", type=float, default=None, help="USD budget of every run (default: Table 1 of the paper: $15, swiglu $1)")
    ap.add_argument("--budget-scale", type=float, default=1.0, help="multiply the budgets (e.g. 0.2 for a smoke run)")
    ap.add_argument("--max-total-usd", type=float, default=None, help="optional cap on the whole experiment: skip a trial whose budget would not fit under it (default: off)")
    ap.add_argument("--preset", default=None, help="config preset: 'paper' (default, taic.paper.toml) or 'base' (taic.toml alone)")
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out_dir) if a.out_dir else REPO_ROOT / "runs" / f"repeat_{time.strftime('%Y%m%d-%H%M%S')}"
    table = run_repeatability(a.tasks.split(","), a.dtype, a.size, out, trials=a.trials, budget_scale=a.budget_scale, budget=a.budget, max_total_usd=a.max_total_usd, max_rounds=a.max_rounds, model=a.model, **({"preset": a.preset} if a.preset else {}))
    md = markdown(table)
    (out / "repeatability.md").write_text(md + "\n")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
