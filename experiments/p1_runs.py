"""Single, named search runs for the questions the suite cannot answer:

* ``fa_flash``  FlashAttention with the full budget and the default model (the pilot used a 1.5 USD budget and stopped after 3 rounds);
* ``fa_pro``    the same with a stronger model of the same provider (``deepseek-v4-pro``, 4.4x the price per token);
* ``fa_64x32t256`` / ``fa_32x16t128``  the same operator under another launch contract (query tile, key tile, threads): if the hand-picked
  contract limited the search, one of these would beat ``fa_flash``;
* ``fa_flash_prof``  ``fa_flash`` with ``search.profile_poor``: does feeding Instruments findings (register spills, code size, GPU idle) to the model help?
* ``fa_paper`` / ``gemm_paper`` / ``conv2d_paper``  the paper's regime: 15 USD per run, patience 8, up to 40 rounds, with ``profile_poor``, ``plan_mix`` and
  ``init_samples=3`` (Section 3.3); compare with the 3 USD runs of the suite to see what budget plus search aids buy (the two effects are not separated);
* ``mamba2``    the phase-2 Mamba-2 chunk-scan kernel.

    uv run python -m experiments.p1_runs --runs fa_flash,fa_pro --out runs/p1
    uv run python -m experiments.p1_runs --report --out runs/p1

Runs within one process are sequential and resumable (a finished run is skipped); several processes may share one ``--out`` with disjoint
``--runs`` (``summary.json`` / ``summary.md`` always list every finished run). Every run is bounded by its USD budget; the GPU lock protects the timings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from observe.publish import job_state

from .common import naive_reference, read_confirm, read_result, run_one, summarize

REPO_ROOT = Path(__file__).resolve().parent.parent

PAPER_REGIME = dict(profile_poor=True, plan_mix=True, init_samples=3, patience=8, max_rounds=40)

RUNS: dict[str, dict] = {
    "fa_flash": dict(task="flash_attention", model=None, budget=3.0),
    "fa_pro": dict(task="flash_attention", model="deepseek-v4-pro", budget=3.0),
    "fa_64x32t256": dict(task="flash_attention@64x32t256", model=None, budget=3.0),
    "fa_32x16t128": dict(task="flash_attention@32x16t128", model=None, budget=3.0),
    # the same search as fa_flash, plus: a slow seed is traced with Instruments (spills, code size, GPU idle) and the findings go to the planner and the patch authors
    "fa_flash_prof": dict(task="flash_attention", model=None, budget=3.0, search=dict(profile_poor=True)),
    # the paper's regime (Section 4.3: $15 per run, run until the budget is gone) with the search aids of Section 3.3: profiler findings, local+structural plans, repeated sampling of the seed
    "fa_paper": dict(task="flash_attention", model=None, budget=15.0, search=PAPER_REGIME),
    "gemm_paper": dict(task="gemm", model=None, budget=15.0, search=PAPER_REGIME),
    "conv2d_paper": dict(task="conv2d", model=None, budget=15.0, search=PAPER_REGIME),
    "mamba2": dict(task="mamba2_chunk_scan", model=None, budget=3.0),
    "mamba2_bf16": dict(task="mamba2_chunk_scan", model=None, budget=3.0, dtype="bf16"),
}


def finished_rows(out: Path) -> list[dict]:
    """One row per finished run found in ``out``, whichever process ran it (several processes may share one output directory)."""
    rows = []
    for n, spec in RUNS.items():
        res = read_result(out / n)
        if not res:
            continue
        nf = out / f"naive_{spec['task']}_{spec.get('dtype', 'fp16')}.json"
        rows.append({"run": n, "options": spec.get("search") or {}, **summarize(res, read_confirm(out / n), json.loads(nf.read_text()) if nf.exists() else None)})
    return rows


def run_named(names: list[str], out: Path, *, size: str = "paper", dtype: str = "fp16", log=print, **overrides) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    names = list(names)
    for i, n in enumerate(names):
        spec = RUNS[n]
        job_state(out / "state.json", script="p1_runs", current={"run": n, "task": spec["task"]}, queue=[{"run": x} for x in names[i + 1 :]], done=rows)
        dt = spec.get("dtype", dtype)
        rd = out / n
        res = read_result(rd)
        if res is None:
            log(f"\n##### {n}: {spec['task']} {dt} model={spec['model'] or 'default'} budget=${spec['budget']}")
            r = run_one(spec["task"], dt, size, rd, budget=spec["budget"], model=spec["model"], log=log, **{**spec.get("search", {}), **overrides})
            res = read_result(rd)
            if r.stop_reason.startswith(("gpu_hang", "repeated_worker_crashes")):
                log("== aborted: GPU hang circuit breaker")
                rows.append({"run": n, **summarize(res, read_confirm(rd), None)})
                (out / "summary.json").write_text(json.dumps(finished_rows(out), indent=1))
                job_state(out / "state.json", script="p1_runs", current={"run": n}, queue=[], done=rows, phase="aborted")
                return rows
        else:
            log(f"(skip {n}: already done)")
        naive = naive_reference(spec["task"], dt, size, out)
        rows.append({"run": n, **summarize(res, read_confirm(rd), naive)})
        (out / "summary.json").write_text(json.dumps(finished_rows(out), indent=1))
    job_state(out / "state.json", script="p1_runs", current=None, queue=[], done=rows, phase="done")
    return rows


def markdown(rows: list[dict]) -> str:
    f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"
    L = ["| run | task | model | confirmed speedup | best (search) | candidate us | best baseline | naive kernel | vs triton-msl | rounds | evals | spent USD | stop |", "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        L.append(f"| {r['run']} | {r['task']} | {r['model']} | {f(r.get('confirmed_speedup'))}x | {f(r.get('best_speedup'))}x | {f(r.get('candidate_us'), 1)} | {r.get('best_baseline') or '-'} | {f(r.get('naive_speedup'))}x | {f(r.get('vs_triton_msl'))} | {r['rounds']} | {r['evaluations']} | {r['spent_usd']:.3f} | {r['stop_reason']} |")
    L.append(f"\nTotal spent: ${sum(r['spent_usd'] for r in rows):.3f}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="fa_flash")
    ap.add_argument("--out", default="runs/p1")
    ap.add_argument("--report", action="store_true", help="only print the table of the finished runs")
    a = ap.parse_args(argv)
    out = REPO_ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    if a.report:
        md = markdown(finished_rows(out))
        (out / "summary.md").write_text(md + "\n")
        print(md)
        return 0
    run_named(a.runs.split(","), out)
    md = markdown(finished_rows(out))
    (out / "summary.md").write_text(md + "\n")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
