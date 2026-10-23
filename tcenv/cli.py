"""``tcenv`` command line: evaluate kernels, time baselines, calibrate, inspect tasks."""

# 流水线入口，验证环境从这些子命令进去。
# eval 读用户给出的 kernel 文件，交给 evaluate()。naive 把任务自带的 naive_msl 交给同一个 evaluate()，
# 调用链上没有模型请求，接口费用为零，用来自检评测架。profile 先跑 quick，通过后再做 Instruments 跟踪。
# tasks 只打印契约（网格、参数、字节、FLOP），停在任务表。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# eval、naive、baseline、profile、show 共用 dtype 和 size，比的是同一道题、同一种宽度。
def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--size", default="paper", choices=["paper", "small", "plan"])


# 进行中的阶段写到按 pid 区分的观察文件，长评测在命令行外面也能看到走到哪一步。
def _live_progress():
    from tcenv.events import tcenv_live_path

    return tcenv_live_path()


# 外部 kernel 文件这条路。quick 停在编译、ABI、安全和正确性；默认 full 会再跑着色器校验子进程和计时。
def cmd_eval(a: argparse.Namespace) -> int:
    from .evaluate import evaluate
    from .report import format_feedback

    if a.quick and a.timed:
        print("pass only one of --quick and --timed", file=sys.stderr)
        return 2
    if a.quick and a.validate:
        print("--validate belongs to the timed evaluation; --quick never validates", file=sys.stderr)
        return 2
    src = Path(a.kernel).read_text()
    mode = "timed" if a.timed else "quick" if a.quick else "full"
    rep = evaluate(
        a.task, a.dtype, a.size, src, mode=mode, seed=a.seed, reps=a.reps, progress_path=_live_progress(),
        require_contract_launch=bool(a.require_contract_launch),
        self_check_dir=Path(a.self_check_dir) if a.self_check_dir else (Path(a.kernel).resolve().parent if a.timed else None),
        validate=bool(a.validate),
        measure_serial=bool(getattr(a, "measure_serial", False)),
        eval_cache=bool(getattr(a, "eval_cache", False)),
        target_id=a.target,
    )
    if a.json:
        print(json.dumps(rep.to_json(), indent=2))
    else:
        print(format_feedback(rep))
    if a.timed and a.seed_report:
        from .verdict import gpu_timestamp_line, replacement_line

        try:
            seed_doc = json.loads(Path(a.seed_report).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"seed report was not read: {exc}", file=sys.stderr)
            return 1
        gpu_line = gpu_timestamp_line(rep, seed_doc)
        if gpu_line:
            print(gpu_line)
        print(replacement_line(rep, seed_doc, min_improvement=a.min_improvement, noise_z=a.noise_z))
    return 0 if rep.status == "ok" else 1


# 自检评测架。源码是任务里写好的 naive_msl，本函数只调用 evaluate()，和 eval 同一条编译、检查、计时路径。
# 调用链上没有模型请求，接口费用为零。
def cmd_naive(a: argparse.Namespace) -> int:
    from tasks import get_task

    from .evaluate import evaluate
    from .report import format_feedback

    from .target import require_active

    target = require_active(a.target)
    task = get_task(a.task, a.dtype, a.size)
    rep = evaluate(a.task, a.dtype, a.size, task.reference_for(target.backend), mode="quick" if a.quick else "full", seed=a.seed, reps=a.reps, progress_path=_live_progress(), target_id=a.target)
    print(format_feedback(rep))
    return 0 if rep.status == "ok" else 1


# 只测 MLX / PyTorch 基线。计时段独占 GPU 锁，测完用同一批中位数算 roofline，看基线离带宽和算力下界有多远。
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
        calib = get_calibration(target_id=a.target)
    for n, s in stats.items():
        r = roofline(task, s.median_s, calib)
        print(f"{a.task:14s} {a.dtype} {n:10s} {s.median_us:10.2f} us  cv {s.cv:5.1%}  {r['achieved_gbps']:7.1f} GB/s ({r['bandwidth_fraction']:.0%})  {r['achieved_tflops']:6.2f} TFLOP/s ({r['compute_fraction']:.0%})")
    return 0


# 跟踪建立在 quick 已经通过的前提下，录制本身在 xctrace 里独占 GPU 锁。
def cmd_profile(a: argparse.Namespace) -> int:
    """Instruments trace of an *already validated* kernel (the full evaluation must pass first)."""
    from . import xctrace
    from .evaluate import evaluate

    src = Path(a.kernel).read_text()
    rep = evaluate(a.task, a.dtype, a.size, src, mode="quick", progress_path=_live_progress(), target_id=a.target)
    if rep.status != "ok":
        # quick 覆盖编译、ABI、安全和正确性；任一步失败，Instruments 录制不会开始。
        print(f"refusing to profile: the kernel does not pass the quick evaluation ({rep.status}: {rep.message[:200]})")
        return 1
    summ = xctrace.profile_kernel(a.task, a.dtype, a.size, src, target_id=a.target)
    print(xctrace.format_summary(summ))
    if a.json:
        print(json.dumps(summ.to_json(), indent=2))
    return 0


# 另一条基线：在隔离虚拟环境里用 triton-msl 把任务的 Triton 规格编成 Metal 再计时。缺环境就停。
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
    # 全部任务都失败才返回非零；只要有一个成功就返回 0。
    return 1 if bad == len(names) else 0


# 测这台机器的峰值带宽和算力，给 roofline 当天花板。
def cmd_calibrate(a: argparse.Namespace) -> int:
    from .metal_runtime import get_context
    from .profile import get_calibration
    from .target import TargetError, require_active

    try:
        require_active(a.target)
    except TargetError as exc:
        print(str(exc), file=sys.stderr)
        return 5
    print(json.dumps(get_context().info(), indent=2))
    print(json.dumps(get_calibration(force=True, target_id=a.target), indent=2))
    return 0


# 进验证前先看任务表。dtype 固定 fp16，只为把契约（网格、线程、参数名、最小字节、FLOP）打印出来。
def cmd_tasks(a: argparse.Namespace) -> int:
    from tasks import get_task, list_tasks

    for n in list_tasks(include_phase2=True):
        t = get_task(n, "fp16", a.size)
        c = t.contract
        print(f"{n:14s} grid={c.grid} threads={c.threads} args={[x.name for x in c.args]} min_bytes={t.min_bytes/1e6:.1f}MB flops={t.flops/1e9:.2f}G")
    return 0


# 打印单个任务的说明、契约、ABI 桩和 Triton 源码，供人对照着写核。
def cmd_show(a: argparse.Namespace) -> int:
    from dataclasses import replace

    from tasks import get_task

    from .target import TargetError, require_active

    try:
        target = require_active(a.target)
    except TargetError as exc:
        print(str(exc), file=sys.stderr)
        return 5
    cached = get_task(a.task, a.dtype, a.size)
    contract = replace(cached.contract, gpu=target.display_name)
    print(cached.description, "\n")
    print(contract.describe(), "\n")
    print(contract.abi_stub(), "\n")
    print(cached.triton_source)
    return 0


# 子命令分发。eval 走外部核，naive 走任务自带参考核，profile 先 quick 再跟踪，tasks 只列契约。
def main(argv: list[str] | None = None) -> int:
    from tcenv.children import install_termination_handlers
    from taic.llm import SearchConfig

    install_termination_handlers()
    ap = argparse.ArgumentParser(prog="tcenv", description="Triton Compiler Environment for Apple GPUs (MSL)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("eval", help="evaluate a kernel.metal against a task")
    p.add_argument("task")
    p.add_argument("kernel")
    _common(p)
    p.add_argument("--quick", action="store_true", help="compile + ABI + safety + correctness only (no benchmark, no shader validation)")
    p.add_argument("--timed", action="store_true", help="correctness plus the benchmark, without shader validation")
    p.add_argument("--validate", action="store_true", help="with --timed: run the instrumented shader-validation pass first (it costs a second subprocess)")
    p.add_argument("--eval-cache", dest="eval_cache", action="store_true", help="reuse an identical (kernel, task, seed, mode, protocol) report")
    p.add_argument("--serial", dest="measure_serial", action="store_true", help="also time the extra strictly serial dispatch (candidate_serial); off by default after the 2026-10-22 A/B")
    p.add_argument("--no-serial", dest="measure_serial", action="store_false", help="keep the default: do not time the extra strictly serial dispatch")
    p.add_argument("--seed-report", default=None, help="the seed report.json; --timed then says whether this version would replace it")
    p.add_argument("--min-improvement", type=float, default=SearchConfig.min_improvement)
    p.add_argument("--noise-z", type=float, default=SearchConfig.noise_z)
    p.add_argument("--require-contract-launch", action="store_true")
    p.add_argument("--self-check-dir", default=None, help="directory for self_checks.jsonl; default is the kernel file's directory")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--reps", type=int, default=SearchConfig.eval_reps)
    p.add_argument("--target", default="apple-m4-max")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("naive", help="evaluate the task's built-in naive reference kernel (harness self-check)")
    p.add_argument("task")
    _common(p)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--target", default="apple-m4-max")
    p.set_defaults(fn=cmd_naive)

    p = sub.add_parser("baseline", help="time the MLX / torch-MPS baselines")
    p.add_argument("task")
    _common(p)
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--target", default="apple-m4-max")
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("profile", help="Instruments (xctrace) Metal System Trace of a kernel that passes the quick evaluation")
    p.add_argument("task")
    p.add_argument("kernel")
    _common(p)
    p.add_argument("--json", action="store_true")
    p.add_argument("--target", default="apple-m4-max")
    p.set_defaults(fn=cmd_profile)

    p = sub.add_parser("triton-baseline", help="compile the Triton spec of a task to Metal with triton-msl (isolated venv) and cache its time relative to MLX")
    p.add_argument("task", help="task name or 'all'")
    _common(p)
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--wait-quiet", type=float, default=0.0, metavar="MIN", help="wait up to MIN minutes for a quiet machine before each measurement (1-minute load average <= 6 on this 16-core machine, no other tcenv/pytest jobs)")
    p.add_argument("--target", default="apple-m4-max")
    p.set_defaults(fn=cmd_triton_baseline)

    p = sub.add_parser("calibrate", help="measure peak bandwidth / TFLOPs for the roofline")
    p.add_argument("--target", default="apple-m4-max")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("tasks", help="list tasks")
    p.add_argument("--size", default="paper", choices=["paper", "small", "plan"])
    p.set_defaults(fn=cmd_tasks)

    p = sub.add_parser("show", help="print the contract and Triton source of a task")
    p.add_argument("task")
    _common(p)
    p.add_argument("--target", default="apple-m4-max")
    p.set_defaults(fn=cmd_show)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
