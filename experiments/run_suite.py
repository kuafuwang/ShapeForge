"""Experiment 1 (paper Figure 4): speedup of AI lowering over the baselines for the 12 common kernels, FP16 and BF16,
together with the speedup-versus-cumulative-API-cost frontier of every run.

    uv run python -m experiments.run_suite --tasks all --dtypes fp16,bf16
    uv run python -m experiments.run_suite --tasks relu,gemm --dtypes fp16 --budget-scale 0.2 --resume   # $3 per run instead of the paper's $15
    uv run python -m experiments.run_suite --tasks all --dtypes fp16,bf16 --max-total-usd 150           # never start a run whose budget would not fit

Every run gets the paper's budget of $15 (``--budget USD`` sets another, ``--budget-scale`` multiplies it); at most 12 x 2 x $15 = $360 for
the whole suite, much less in practice because a run that stops improving (``patience``) ends early. ``--max-total-usd`` is an optional cap
on the whole suite (default off): a run is skipped, and logged as such, if its budget plus what the suite has spent so far would exceed it.
Runs are sequential (one GPU, no interference). A finished (task, dtype) pair is skipped with ``--resume``. The whole suite stops
at the first GPU hang (circuit breaker).
"""

# 论文实验 1（Figure 4）：一次套件把多道算子和多种小数铺开，每格单独搜、单独记账。
# 每格的美元上限来自论文的 $15（PAPER_BUDGET_USD）。--budget-scale 只乘这个数；轮数、耐心、分支仍由配置预设决定，缩放不改协议。

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tasks import list_tasks

from observe.publish import job_state

from .common import budget_for, fits_cap, naive_reference, read_confirm, read_result, run_one, summarize

REPO_ROOT = Path(__file__).resolve().parent.parent


# 网格是「每种小数 × 每道题」各一个目录 `{task}_{dtype}`，换小数不会接着上一格的最优核。
# --resume 为真时，该目录已经有 result.json 就不再启动搜索。文件不存在才整格重跑。
def run_suite(tasks, dtypes, size, suite_dir: Path, *, budget_scale=1.0, budget=None, max_total_usd=None, branches=None, max_rounds=None, model=None, resume=True, client=None, evaluate_fn=None, log=print, with_naive=True, **overrides) -> list[dict]:
    suite_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    items = [(dtype, task) for dtype in dtypes for task in tasks]
    for i, (dtype, task) in enumerate(items):
        rest = [{"task": t, "dtype": d} for d, t in items[i + 1 :]]
        job_state(suite_dir / "state.json", script="run_suite", current={"task": task, "dtype": dtype}, queue=rest, done=rows)
        rd = suite_dir / f"{task}_{dtype}"
        res = read_result(rd) if resume else None
        # 命令行给了 --budget 就整套共用这笔钱，scale 不再乘。否则用论文的 $15 乘 --budget-scale，协议字段不在这里改。
        b = budget if budget is not None else budget_for(task, budget_scale)
        if res is None:
            spent = sum(r["spent_usd"] for r in rows)
            if not fits_cap(spent, b, max_total_usd):
                # 上限比的是「已花费 + 这一格的整份预算」。提前收敛省下的钱不算进去，放不下就留空这一格。
                log(f"(skip {task} {dtype}: its ${b:.2f} budget does not fit under --max-total-usd ${max_total_usd:.2f}; the suite has spent ${spent:.2f})")
                continue
            log(f"\n##### {task} {dtype} (budget ${b:.2f})")
            r = run_one(task, dtype, size, rd, budget=b, branches=branches, max_rounds=max_rounds, model=model, client=client, evaluate_fn=evaluate_fn, log=log, **overrides)
            res = read_result(rd)
            if r.stop_reason.startswith(("gpu_hang", "repeated_worker_crashes")):
                # 一台 GPU 上连环占卡没有意义：这一格记下来，后面的题不再启动。
                rows.append(summarize(res, read_confirm(rd), None))
                (suite_dir / "summary.json").write_text(json.dumps(rows, indent=1))
                job_state(suite_dir / "state.json", script="run_suite", current={"task": task, "dtype": dtype}, queue=rest, done=rows, phase="aborted")
                log("== suite aborted: GPU hang circuit breaker")
                return rows
        else:
            # 跳过的是搜索。这一格仍进汇总，确认加速比从已写好的 result.json 来。
            log(f"(skip {task} {dtype}: already done)")
        naive = naive_reference(task, dtype, size, suite_dir, evaluate_fn=evaluate_fn) if with_naive else None
        rows.append(summarize(res, read_confirm(rd), naive))
        (suite_dir / "summary.json").write_text(json.dumps(rows, indent=1))
    job_state(suite_dir / "state.json", script="run_suite", current=None, queue=[], done=rows, phase="done")
    return rows


# 写进 summary.md 的那张表。缺测印成横杠，读表的人不会把「没跑」看成 0 倍。
def markdown_table(rows: list[dict]) -> str:
    L = ["| task | dtype | confirmed speedup vs best baseline | naive kernel (reference) | vs triton-msl | candidate us | best baseline | baselines (us) | budget USD | spent USD | evals | stop |", "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |"]
    f = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"
    for r in rows:
        b = r.get("baseline_us") or {}
        L.append(
            f"| {r['task']} | {r['dtype']} | {f(r['confirmed_speedup'])}x | {f(r.get('naive_speedup'))}x | {f(r.get('vs_triton_msl'))}{'x' if r.get('vs_triton_msl') else ''} | {f(r['candidate_us'], 1)} | {r.get('best_baseline') or '-'} | {', '.join(f'{n} {v:.0f}' for n, v in sorted(b.items(), key=lambda kv: kv[1]))} | {'-' if r.get('budget_usd') is None else format(r['budget_usd'], '.2f')} | {r['spent_usd']:.3f} | {r['evaluations']} | {r['stop_reason']} |"
        )
    return "\n".join(L)


# 命令行只负责拆题单和小数。--tasks all 用任务表里的名字铺网格，不是 p1 那种具名变体。
# 没写 --preset 时不往下传，搜索协议留在默认的论文预设上；scale 只影响美元上限。
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--dtypes", default="fp16")
    ap.add_argument("--size", default="paper", choices=["paper", "small"])
    ap.add_argument("--suite-dir", default=None)
    ap.add_argument("--budget", type=float, default=None, help="USD budget of every run (default: the paper's $15)")
    ap.add_argument("--budget-scale", type=float, default=1.0, help="multiply the budget (e.g. 0.2 for a $3 smoke run)")
    ap.add_argument("--max-total-usd", type=float, default=None, help="optional cap on the whole suite: skip a run whose budget would not fit under it (default: off)")
    ap.add_argument("--preset", default=None, help="config preset: 'paper' (default, taic.paper.toml) or 'base' (taic.toml alone)")
    ap.add_argument("--branches", type=int, default=None)
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-naive", action="store_true", help="skip the naive-kernel reference measurement")
    a = ap.parse_args(argv)
    tasks = list(list_tasks()) if a.tasks == "all" else a.tasks.split(",")
    suite_dir = Path(a.suite_dir) if a.suite_dir else REPO_ROOT / "runs" / f"suite_{time.strftime('%Y%m%d-%H%M%S')}"
    # 默认就续跑：只有 --no-resume 才把已有 result.json 的格子再搜一遍。
    rows = run_suite(tasks, a.dtypes.split(","), a.size, suite_dir, budget_scale=a.budget_scale, budget=a.budget, max_total_usd=a.max_total_usd, branches=a.branches, max_rounds=a.max_rounds, model=a.model, resume=not a.no_resume, with_naive=not a.no_naive, **({"preset": a.preset} if a.preset else {}))
    table = markdown_table(rows)
    (suite_dir / "summary.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nTotal spent: ${sum(r['spent_usd'] for r in rows):.3f}   ({suite_dir})")
    try:
        from .plots import plot_suite

        plot_suite(suite_dir)
    except Exception as e:  # noqa: BLE001
        # 图是附带的。汇总文件已经写好，画图失败不算这套实验失败。
        print(f"(plots skipped: {type(e).__name__}: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
