"""``tcenv`` command line: evaluate kernels, time baselines, calibrate, inspect tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--size", default="paper", choices=["paper", "small", "plan"])


def _live_progress():
    from observe.publish import tcenv_live_path

    return tcenv_live_path()


def cmd_eval(a: argparse.Namespace) -> int:
    from .evaluate import evaluate
    from .report import format_feedback

    src = Path(a.kernel).read_text()
    rep = evaluate(a.task, a.dtype, a.size, src, mode="quick" if a.quick else "full", seed=a.seed, reps=a.reps, progress_path=_live_progress())
    if a.json:
        print(json.dumps(rep.to_json(), indent=2))
    else:
        print(format_feedback(rep))
    return 0 if rep.status == "ok" else 1


def cmd_naive(a: argparse.Namespace) -> int:
    from tasks import get_task

    from .evaluate import evaluate
    from .report import format_feedback

    task = get_task(a.task, a.dtype, a.size)
    rep = evaluate(a.task, a.dtype, a.size, task.naive_msl, mode="quick" if a.quick else "full", seed=a.seed, reps=a.reps, progress_path=_live_progress())
    print(format_feedback(rep))
    return 0 if rep.status == "ok" else 1


def cmd_baseline(a: argparse.Namespace) -> int:
    from tasks import get_task

    from .baselines import build_runners
    from .bench import interleaved
    from .gpulock import gpu_lock
    from .profile import get_calibration, roofline
    from .testing import prepare_case

    task = get_task(a.task, a.dtype, a.size)
    data = prepare_case(task, task.cases[0], 12345)
    with gpu_lock(f"tcenv baseline {a.task} {a.dtype} {a.size}", log=print):
        runners = build_runners(task, data.rounded)
        stats = interleaved({n: r.run_k for n, r in runners.items()}, reps=a.reps)
        calib = get_calibration()
    for n, s in stats.items():
        r = roofline(task, s.median_s, calib)
        print(f"{a.task:14s} {a.dtype} {n:10s} {s.median_us:10.2f} us  cv {s.cv:5.1%}  {r['achieved_gbps']:7.1f} GB/s ({r['bandwidth_fraction']:.0%})  {r['achieved_tflops']:6.2f} TFLOP/s ({r['compute_fraction']:.0%})")
    return 0


def cmd_profile(a: argparse.Namespace) -> int:
    """Instruments trace of an *already validated* kernel (the full evaluation must pass first)."""
    from . import xctrace
    from .evaluate import evaluate

    src = Path(a.kernel).read_text()
    rep = evaluate(a.task, a.dtype, a.size, src, mode="quick", progress_path=_live_progress())
    if rep.status != "ok":
        print(f"refusing to profile: the kernel does not pass the quick evaluation ({rep.status}: {rep.message[:200]})")
        return 1
    summ = xctrace.profile_kernel(a.task, a.dtype, a.size, src)
    print(xctrace.format_summary(summ))
    if a.json:
        print(json.dumps(summ.to_json(), indent=2))
    return 0


def cmd_triton_baseline(a: argparse.Namespace) -> int:
    from tasks import list_tasks

    from . import triton_baseline as tb

    if not tb.available():
        print("no triton-msl environment found at .venvs/triton-msl (create it with scripts/install_triton_msl.sh)")
        return 1
    if not tb.metal_toolchain_ok():
        print(tb.METAL_TOOLCHAIN_HINT)
        return 1
    names = list(list_tasks(include_phase2=True)) if a.task == "all" else [a.task]
    bad = 0
    for n in names:
        r = tb.measure(n, a.dtype, a.size, reps=a.reps, wait_quiet_s=a.wait_quiet * 60.0)
        bad += r["status"] != "ok"
    return 1 if bad == len(names) else 0


def cmd_calibrate(a: argparse.Namespace) -> int:
    from .metal_runtime import get_context
    from .profile import get_calibration

    print(json.dumps(get_context().info(), indent=2))
    print(json.dumps(get_calibration(force=True), indent=2))
    return 0


def cmd_tasks(a: argparse.Namespace) -> int:
    from tasks import get_task, list_tasks

    for n in list_tasks(include_phase2=True):
        t = get_task(n, "fp16", a.size)
        c = t.contract
        print(f"{n:14s} grid={c.grid} threads={c.threads} args={[x.name for x in c.args]} min_bytes={t.min_bytes/1e6:.1f}MB flops={t.flops/1e9:.2f}G")
    return 0


def cmd_show(a: argparse.Namespace) -> int:
    from tasks import get_task

    t = get_task(a.task, a.dtype, a.size)
    print(t.description, "\n")
    print(t.contract.describe(), "\n")
    print(t.contract.abi_stub(), "\n")
    print(t.triton_source)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tcenv", description="Triton Compiler Environment for Apple GPUs (MSL)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("eval", help="evaluate a kernel.metal against a task")
    p.add_argument("task")
    p.add_argument("kernel")
    _common(p)
    p.add_argument("--quick", action="store_true", help="compile + ABI + safety + correctness only (no benchmark, no shader validation)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("naive", help="evaluate the task's built-in naive reference kernel (harness self-check)")
    p.add_argument("task")
    _common(p)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--reps", type=int, default=15)
    p.set_defaults(fn=cmd_naive)

    p = sub.add_parser("baseline", help="time the MLX / torch-MPS baselines")
    p.add_argument("task")
    _common(p)
    p.add_argument("--reps", type=int, default=15)
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("profile", help="Instruments (xctrace) Metal System Trace of a kernel that passes the quick evaluation")
    p.add_argument("task")
    p.add_argument("kernel")
    _common(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_profile)

    p = sub.add_parser("triton-baseline", help="compile the Triton spec of a task to Metal with triton-msl (isolated venv) and cache its time relative to MLX")
    p.add_argument("task", help="task name or 'all'")
    _common(p)
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--wait-quiet", type=float, default=0.0, metavar="MIN", help="wait up to MIN minutes for a quiet machine before each measurement (1-minute load average <= 6 on this 16-core machine, no other tcenv/pytest jobs)")
    p.set_defaults(fn=cmd_triton_baseline)

    p = sub.add_parser("calibrate", help="measure peak bandwidth / TFLOPs for the roofline")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("tasks", help="list tasks")
    p.add_argument("--size", default="paper", choices=["paper", "small", "plan"])
    p.set_defaults(fn=cmd_tasks)

    p = sub.add_parser("show", help="print the contract and Triton source of a task")
    p.add_argument("task")
    _common(p)
    p.set_defaults(fn=cmd_show)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
