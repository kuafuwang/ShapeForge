"""Does the dispatch mode of the candidate bias the comparison with the vendor baselines?

The benchmark times ``k`` back-to-back launches. A candidate is dispatched *serially* (each launch starts after the previous one
finished), while MLX evaluates a graph of independent calls with a *concurrent* encoder, so the tail of one launch can overlap the
start of the next. This script measures the same kernel under both dispatch modes, interleaved with the baselines, on every task.

    uv run python -m experiments.dispatch_ab --tasks all --dtype fp16 --out runs/dispatch_ab
    uv run python -m experiments.dispatch_ab --kernel runs/<run>/best.metal --tasks gemm      # an LLM kernel instead of the naive one
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def measure(task_name: str, dtype: str, size: str, source: str, reps: int = 15) -> dict:
    from tasks import get_task
    from tcenv.backends.metal.baselines import build_runners, rotation_count
    from tcenv.bench import interleaved
    from tcenv.gpulock import gpu_lock
    from tcenv.backends.metal.runtime import get_context
    from tcenv.backends.metal.sanitize import ArgSet
    from tcenv.backends.metal.testing import check_case, prepare_case

    task = get_task(task_name, dtype, size)
    c = task.contract
    ctx = get_context()
    full, _ = c.assemble(source)
    kernel = ctx.compile(full, c.kernel_name)
    data = prepare_case(task, task.cases[0], 777)
    R = rotation_count(task)
    sets = []
    for _ in range(R):
        s = ArgSet(ctx, c)
        s.load_inputs(data.raw)
        s.poison_outputs()
        sets.append(s)
    args = [s.args() for s in sets]

    def runner(mode: str):
        return lambda k: ctx.run(kernel, args, c.grid, c.threads, iters=k, dispatch=mode).wall_s

    with gpu_lock(f"dispatch A/B of {task_name} {dtype} {size}"):
        runners = {"serial": runner("serial"), "concurrent": runner("concurrent")}
        base = build_runners(task, data.rounded)
        runners.update({n: r.run_k for n, r in base.items()})
        stats = interleaved(runners, reps=reps, seed=3, kmin=R)
    # the outputs left by the timed concurrent runs must still be right
    ok = all(check_case(task, data, s.read_outputs(), None).ok and not s.safety_violations() for s in sets)
    us = {n: s.median_us for n, s in stats.items()}
    best_base = min((n for n in us if n not in ("serial", "concurrent")), key=us.get)
    return {
        "task": task_name, "dtype": dtype, "size": size, "rotation_sets": R, "median_us": us, "cv": {n: s.cv for n, s in stats.items()},
        "best_baseline": best_base,
        "speedup_serial": us[best_base] / us["serial"], "speedup_concurrent": us[best_base] / us["concurrent"],
        "concurrent_over_serial": us["serial"] / us["concurrent"], "outputs_correct_after_timed_runs": ok,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--size", default="paper")
    ap.add_argument("--kernel", default=None, help="MSL source to use instead of the task's naive kernel (single task only)")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--out", default="runs/dispatch_ab")
    a = ap.parse_args(argv)
    from tasks import get_task, list_tasks

    names = list(list_tasks(include_phase2=True)) if a.tasks == "all" else a.tasks.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in names:
        src = Path(a.kernel).read_text() if a.kernel else get_task(n, a.dtype, a.size).naive_msl
        r = measure(n, a.dtype, a.size, src, a.reps)
        rows.append(r)
        print(f"{n:16s} serial {r['median_us']['serial']:9.1f}us  concurrent {r['median_us']['concurrent']:9.1f}us  (x{r['concurrent_over_serial']:.3f})  "
              f"{r['best_baseline']} {r['median_us'][r['best_baseline']]:9.1f}us  speedup serial {r['speedup_serial']:.3f}x concurrent {r['speedup_concurrent']:.3f}x  correct={r['outputs_correct_after_timed_runs']}", flush=True)
        (out / f"results_{a.dtype}.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
