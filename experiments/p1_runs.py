"""Single, named search runs for the questions the suite cannot answer:

* ``fa_flash``  FlashAttention with the paper's budget and the default model (the pilot used a 1.5 USD budget and stopped after 3 rounds);
* ``fa_pro``    the same with a stronger model of the same provider (``deepseek-v4-pro``, 4.4x the price per token);
* ``fa_flash_hw``  ``fa_flash`` with ``search.hw_feedback``: does a short static hardware-feature analysis in the feedback (and Skill 2c) help the model? (see taic/hwfeedback.py);
* ``fa_64x32t256`` / ``fa_32x16t128``  the same operator under another launch contract (query tile, key tile, threads): if the hand-picked
  contract limited the search, one of these would beat ``fa_flash``;
* ``fa_flash_prof``  ``fa_flash`` with ``search.profile_poor``: does feeding Instruments findings (register spills, code size, GPU idle) to the model help?
* ``fa_flash_cnt``  ``fa_flash_prof`` recorded with a GPU-counter template (limiters, occupancy, ...) and with the profiler skill in the system prompt: do the counters add anything?
* ``fa_paper`` / ``gemm_paper`` / ``conv2d_paper``  the paper's regime: 15 USD per run, patience 12, up to 40 rounds, with ``profile_poor``;
  the live search is one session a round, so ``init_samples`` stays 1;
* ``mamba2``    the phase-2 Mamba-2 chunk-scan kernel.

    uv run python -m experiments.p1_runs --runs fa_flash,fa_pro --out runs/apple-m4-max/p1
    uv run python -m experiments.p1_runs --report --out runs/apple-m4-max/p1

Runs within one process are sequential and resumable (a finished run is skipped); several processes may share one ``--out`` with disjoint
``--runs`` (``summary.json`` / ``summary.md`` always list every finished run). Every run is bounded by its USD budget; the GPU lock protects the timings.

Every run has the paper's budget of 15 USD (``--budget USD`` sets another one for the runs of this invocation). The runs in ``runs/apple-m4-max/p1`` that were
finished or started before that change had 3 USD (their ``result.json`` says so, and so does the ``budget`` column of the summary), so they are not directly
comparable; a finished run is skipped by name, so to repeat one at the new budget use a fresh ``--out``. Worst case if every run below is started: 12 x 15 = 180 USD;
``--max-total-usd`` is an optional cap (default off) on what one invocation may spend: a run whose budget would not fit under it is skipped and logged.
"""

# 套件回答不了的单点问题：同一算子换模型、换启动契约、换搜索辅助，或补上阶段 2 的 Mamba-2。
# 和 run_suite 不同，这里没有「题 × 小数」的网格，也没有 --budget-scale。RUNS 里每一条自己带任务名，预算都是论文的 $15。

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taic.llm import PAPER_BUDGET_USD
from tcenv.events import install_file_log, job_state

from .common import experiment_commitment, fits_cap, naive_reference, read_confirm, read_result, run_one, shown_speedup, summarize, write_cap
from .manifest import attach, ensure

REPO_ROOT = Path(__file__).resolve().parent.parent

# PAPER_REGIME 只挂在 fa_paper / gemm_paper / conv2d_paper。fa_flash_prof 和 fa_flash_cnt 的 search 字典都只有 profile_poor。
# fa_flash_cnt 多出来的 GPU 计数器不在这张表里，启动时要另设 TCENV_XCTRACE_TEMPLATE。
# 没有 search 字典的条目沿用默认预设。max_rounds=40 会盖过预设里的轮数上限；套件不会逐题传入这组键。
# init_samples 必须是 1：当前搜索一场一轮，_guard_shape 会拒绝别的值。
PAPER_REGIME = dict(profile_poor=True, patience=12, max_rounds=40)

# 键是 --runs 的名字，也是目录名。task 才是交给搜索的算子；@ 后面是另一套启动契约。预算字段全是论文的 $15。
# fa_pro 改的是模型名；mamba2_bf16 把小数定成 bf16，其余条目用调用时的 dtype（默认 fp16）。
RUNS: dict[str, dict] = {
    "fa_flash": dict(task="flash_attention", model=None, budget=PAPER_BUDGET_USD),
    # A/B of search.hw_feedback: IDENTICAL to fa_flash (same task, shape, contract, model, search options; launch it with the same --budget 3.0 --preset base) except that the
    # static hardware-feature block is appended to the seed feedback of the planner and the patch authors and Skill 2c is in the system prompt. n=1 on a stochastic search: an anecdote.
    "fa_flash_hw": dict(task="flash_attention", model=None, budget=PAPER_BUDGET_USD, search=dict(hw_feedback=True)),
    "fa_pro": dict(task="flash_attention", model="deepseek-v4-pro", budget=PAPER_BUDGET_USD),
    "fa_64x32t256": dict(task="flash_attention@64x32t256", model=None, budget=PAPER_BUDGET_USD),
    "fa_32x16t128": dict(task="flash_attention@32x16t128", model=None, budget=PAPER_BUDGET_USD),
    # the same search as fa_flash, plus: a slow seed is traced with Instruments (spills, code size, GPU idle) and the findings go to the planner and the patch authors
    "fa_flash_prof": dict(task="flash_attention", model=None, budget=PAPER_BUDGET_USD, search=dict(profile_poor=True)),
    # fa_flash_prof plus GPU counters (start with TCENV_XCTRACE_TEMPLATE=<template saved from Instruments, see README) and Skill 2b that teaches the model to read them.
    # Budget, model and every other setting equal fa_flash / fa_flash_prof on purpose: the three runs differ only in the evidence the planner sees. Run the three
    # into the same --out so that they share a budget (runs/apple-m4-max/p1 holds fa_flash and fa_flash_prof from before the budget was raised to the paper's, at 3 USD).
    "fa_flash_cnt": dict(task="flash_attention", model=None, budget=PAPER_BUDGET_USD, search=dict(profile_poor=True)),
    # the paper's regime (Section 4.3: $15 per run, run until the budget is gone) with the search aids of Section 3.3: profiler findings, local+structural plans, repeated sampling of the seed
    "fa_paper": dict(task="flash_attention", model=None, budget=PAPER_BUDGET_USD, search=PAPER_REGIME),
    "gemm_paper": dict(task="gemm", model=None, budget=PAPER_BUDGET_USD, search=PAPER_REGIME),
    "conv2d_paper": dict(task="conv2d", model=None, budget=PAPER_BUDGET_USD, search=PAPER_REGIME),
    "mamba2": dict(task="mamba2_chunk_scan", model=None, budget=PAPER_BUDGET_USD),
    "mamba2_bf16": dict(task="mamba2_chunk_scan", model=None, budget=PAPER_BUDGET_USD, dtype="bf16"),
}


# 汇总扫的是 RUNS 里每一个名字，不限于本次命令点名的那些。多个进程分头写同一个 --out 时，别人已经写完的运行也要出现在表上。
def finished_rows(out: Path) -> list[dict]:
    """@brief 汇总 ``out`` 下所有已完成的具名运行，不限于本次命令点名的那些。

    One row per finished run found in ``out``, whichever process ran it (several processes may share one output directory).

    @param out 运行的输出目录。
    @return 每个已完成运行一行：选项与 ``summarize`` 的字段。
    """
    rows = []
    for n, spec in RUNS.items():
        res = read_result(out / n)
        if not res:
            continue
        nf = out / f"naive_{spec['task']}_{spec.get('dtype', 'fp16')}.json"
        rows.append({"run": n, "options": spec.get("search") or {}, **summarize(res, read_confirm(out / n), json.loads(nf.read_text()) if nf.exists() else None, run_dir=out / n)})
    return rows


# 同目录里已完成的运行如果用的是另一笔预算（例如改预算之前的 $3），只提示不能横比加速比，不会把它们重跑。
def budget_notes(out: Path, names: list[str], budget: float | None) -> list[str]:
    """@brief 提示同目录下预算不同的已完成运行，不能直接横比加速比。

    One line per finished run in ``out`` whose recorded budget differs from the budget the requested runs are about to get (results at different budgets are not comparable).

    @param out 运行的输出目录。
    @param names 本次点名的运行名。
    @param budget 本次的预算；为 ``None`` 时按各运行自己登记的预算算。
    @return 每个预算不一致的运行一条提示。
    """
    wanted = {budget if budget is not None else RUNS[n]["budget"] for n in names}
    notes = []
    for n in RUNS:
        res = read_result(out / n)
        if res and res.get("budget_usd") is not None and res["budget_usd"] not in wanted:
            notes.append(f"note: {out / n} was run with a ${res['budget_usd']:.2f} budget, the runs started now get " + " / ".join(f"${w:.2f}" for w in sorted(wanted)) + "; do not compare their speedups directly")
    return notes


def _arm_delta(name: str, dtype: str, budget: float | None) -> dict:
    """@brief 取出一个具名运行相对预设的覆盖项。

    Switches that define this named run. A missing model means the preset's model, not a model of null.

    @param name ``RUNS`` 里的运行名。
    @param dtype 本次调用的数据类型，运行条目没写 dtype 时用它。
    @param budget 本次的预算；为 ``None`` 时用条目自己登记的预算。
    @return 要写进清单的覆盖字典。
    """
    spec = RUNS[name]
    delta = {
        "task": spec["task"],
        "dtype": spec.get("dtype", dtype),
        "budget_usd": budget if budget is not None else spec["budget"],
        **(spec.get("search") or {}),
    }
    if spec.get("model"):
        delta["model"] = spec["model"]
    return delta


# 按名字顺序跑。result.json 已在就跳过，哪怕当时的预算和这次要给的不一样；要按新预算重跑得换一个 --out。
def run_named(names: list[str], out: Path, *, size: str = "paper", dtype: str = "fp16", log=print, budget: float | None = None, max_total_usd: float | None = None, **overrides) -> list[dict]:
    """@brief 按名字逐个跑搜索；已有 result.json 的跳过，并按总预算上限逐个判断。

    @param names ``RUNS`` 里的运行名列表。
    @param out 输出目录。
    @param size 规模档位。
    @param dtype 数据类型。
    @param log 进度输出函数。
    @param budget 本次统一预算；为 ``None`` 时用各条目登记的预算。
    @param max_total_usd 总花费上限；为 ``None`` 时不限。
    @param overrides 其余透传给 ``run_one`` 的覆盖项。
    @return 本次跑过或跳过的运行的结果行。
    """
    install_file_log()
    out.mkdir(parents=True, exist_ok=True)
    write_cap(out, max_total_usd)
    rows = []
    names = list(names)
    ensure(
        out,
        kind="suite",
        driver="p1_runs",
        frozen={
            "size": size,
            "dtype": dtype,
            "preset": overrides.get("preset"),
            "model": None,
        },
        arms=[
            {
                "arm_id": n,
                "planned_n": 1,
                "delta": _arm_delta(n, dtype, budget),
            }
            for n in names
        ],
    )
    for note in budget_notes(out, names, budget):
        log(note)
    for i, n in enumerate(names):
        spec = RUNS[n]
        # --budget 把本次点名的每一条都改成同一个美元数。不写时用该条目自己的 PAPER_BUDGET_USD，这里没有 scale。
        b = budget if budget is not None else spec["budget"]
        job_state(out / "state.json", script="p1_runs", current={"run": n, "task": spec["task"]}, queue=[{"run": x} for x in names[i + 1 :]], done=rows)
        dt = spec.get("dtype", dtype)
        rd = out / n
        attach(out, n, n)
        res = read_result(rd)
        if res is None:
            spent = experiment_commitment(out)
            if not fits_cap(spent, b, max_total_usd):
                log(f"(skip {n}: its ${b:.2f} budget does not fit under --max-total-usd ${max_total_usd:.2f}; this invocation has spent ${spent:.2f})")
                continue
            log(f"\n##### {n}: {spec['task']} {dt} model={spec['model'] or 'default'} budget=${b:.2f}")
            # spec["search"] 里的键盖过预设。没有这份字典时，协议停在预设上，变的只是任务名、模型和这一条的预算。
            r = run_one(spec["task"], dt, size, rd, budget=b, model=spec["model"], log=log, **{**spec.get("search", {}), **overrides})
            res = read_result(rd)
            if r.stop_reason.startswith(("gpu_hang", "repeated_worker_crashes")):
                log("== aborted: GPU hang circuit breaker")
                rows.append({"run": n, **summarize(res, read_confirm(rd), None, run_dir=rd)})
                (out / "summary.json").write_text(json.dumps(finished_rows(out), indent=1))
                job_state(out / "state.json", script="p1_runs", current={"run": n}, queue=[], done=rows, phase="aborted")
                return rows
        else:
            # 预算对不上也只跳过。旧的 result.json 留在原地，日志里写明两笔数字。
            was = res.get("budget_usd")
            log(f"(skip {n}: already done" + (f", with a ${was:.2f} budget, not ${b:.2f}" if was is not None and abs(was - b) > 1e-9 else "") + ")")
        naive = naive_reference(spec["task"], dt, size, out)
        rows.append({"run": n, **summarize(res, read_confirm(rd), naive, run_dir=rd)})
        (out / "summary.json").write_text(json.dumps(finished_rows(out), indent=1))
    job_state(out / "state.json", script="p1_runs", current=None, queue=[], done=rows, phase="done")
    return rows


# 表的第一列是具名跑法。fa_pro 和 fa_flash 的 task 都是 flash_attention，模型和搜索选项才是这两行的差别。
def markdown(rows: list[dict]) -> str:
    """@brief 把具名运行的结果拍成 markdown 表，末行附总花费。

    @param rows ``run_named`` 或 ``finished_rows`` 返回的结果行。
    @return 表格文本。
    """
    f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"
    L = ["| run | task | model | published speedup | best (search) | candidate us | best baseline | naive kernel | vs triton-msl | rounds | evals | budget USD | spent USD | stop |", "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        published = shown_speedup(r)
        published_cell = f"{published:.3f}x" if isinstance(published, (int, float)) and not isinstance(published, bool) else (r.get("publication") or "-")
        L.append(f"| {r['run']} | {r['task']} | {r['model']} | {published_cell} | {f(r.get('best_speedup'))}x | {f(r.get('candidate_us'), 1)} | {r.get('best_baseline') or '-'} | {f(r.get('naive_speedup'))}x | {f(r.get('vs_triton_msl'))} | {r['rounds']} | {r['evaluations']} | {f(r.get('budget_usd'), 2)} | {r['spent_usd']:.3f} | {r['stop_reason']} |")
    L.append(f"\nTotal spent: ${sum(r['spent_usd'] for r in rows):.3f}")
    return "\n".join(L)


# --report 只读已有结果打表，不启动搜索。开跑之后打表用的仍是磁盘上全部已完成名字，不限于本次 --runs。
def main(argv=None) -> int:
    """@brief ``python -m experiments.p1_runs`` 的入口；``--report`` 只读已有结果打表。

    @param argv 命令行参数；为 ``None`` 时取 ``sys.argv``。
    @return 进程退出码，正常结束时为 0。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="fa_flash")
    ap.add_argument("--out", default="runs/apple-m4-max/p1")
    ap.add_argument("--budget", type=float, default=None, help="USD budget of every run of this invocation (default: the paper's $15)")
    ap.add_argument("--max-total-usd", type=float, default=None, help="optional cap on what this invocation may spend: skip a run whose budget would not fit under it (default: off)")
    ap.add_argument("--preset", default=None, help="config preset: 'paper' (default, taic.paper.toml) or 'base' (taic.toml alone)")
    ap.add_argument("--report", action="store_true", help="only print the table of the finished runs")
    a = ap.parse_args(argv)
    out = REPO_ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    if a.report:
        # 只重写 summary.md。没有 result.json 的名字不会出现，因为 finished_rows 读不到它们。
        md = markdown(finished_rows(out))
        (out / "summary.md").write_text(md + "\n")
        print(md)
        return 0
    run_named(a.runs.split(","), out, budget=a.budget, max_total_usd=a.max_total_usd, **({"preset": a.preset} if a.preset else {}))
    md = markdown(finished_rows(out))
    (out / "summary.md").write_text(md + "\n")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
