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
    """@brief 为子命令加上共用的 dtype 与 size 选项。

    @param p 目标参数解析器。
    """
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--size", default="paper")


# 进行中的阶段写到按 pid 区分的观察文件，长评测在命令行外面也能看到走到哪一步。
def _live_progress():
    """@brief 返回当前进程的实时进度文件路径。

    @return ``tcenv.events.tcenv_live_path`` 给出的路径。
    """
    from tcenv.events import tcenv_live_path

    return tcenv_live_path()


# 外部 kernel 文件这条路。quick 停在编译、ABI、安全和正确性；默认 full 会再跑着色器校验子进程和计时。
def cmd_eval(a: argparse.Namespace) -> int:
    """@brief 评测一个外部 kernel 文件并打印反馈或 JSON 报告。

    quick 停在编译、ABI、安全与正确性；默认 full 还包含着色器校验子进程与计时；
    timed 只做计时。带 ``--seed-report`` 时额外给出一行换种子判断。

    @param a 已解析的命令行参数。
    @return 成功为 0；参数冲突或种子报告读取失败为 1/2；报告 status 非 ok 为 1。
    """
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


def cmd_eval_v2(a: argparse.Namespace) -> int:
    """@brief 评测一个 v2 标准目录任务的候选（msl-single 或 msl-bundle envelope）。

    与搜索外层走的是**同一个** ``evaluate_task``（经 ``search_candidate.evaluate_candidate`` 的
    portfolio 包装），报告是同一份 schema-4 文档；agent 的自测数字只供它自己判断，不进种子。

    @param a 已解析命令行参数。
    @return 报告 status 为 ok 返回 0，否则 1；参数/交付非法为 2。
    """
    import hashlib
    import time

    from taic.task_binding import TaskBindingError, bind_task
    from tcenv.evaluate import EvalReport
    from tcenv.evaluation import CandidateDeliveryError
    from tcenv.report import format_feedback
    from tcenv.schema import PROTOCOL_VERSION
    from tcenv.target import load_target

    from .selfcheck import append, budget_reason, earlier_hang

    if a.quick and a.timed:
        print("pass only one of --quick and --timed", file=sys.stderr)
        return 2
    if a.require_contract_launch:
        print("--require-contract-launch is a v1 contract check; a v2 candidate always supplies its own taic-launch", file=sys.stderr)
        return 2
    # Read bytes and decode inside the narrow boundary: a non-UTF-8 submission is the candidate's own
    # defect (exit 2), not a traceback.
    try:
        raw = Path(a.kernel).read_bytes()
    except OSError as exc:
        print(f"cannot read {a.kernel}: {exc}", file=sys.stderr)
        return 2
    try:
        src = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        print(f"{a.kernel} is not UTF-8 text: {exc}", file=sys.stderr)
        return 2
    mode = "timed" if a.timed else "quick" if a.quick else "full"
    self_check_dir = Path(a.self_check_dir) if a.self_check_dir else (Path(a.kernel).resolve().parent if a.timed else None)
    if a.timed and self_check_dir is not None:
        blocked = earlier_hang(self_check_dir) or budget_reason(self_check_dir)
        if blocked:
            print(blocked)
            return 1
    try:
        binding = bind_task(a.task, a.dtype, a.size, target_id=a.target)
        if not binding.is_v2:
            print(f"task {a.task!r} is not a v2 standard entry; use `tcenv eval` for v1 tasks", file=sys.stderr)
            return 2
        from dataclasses import replace

        from tcenv.track.registry import load_track_registry
        from tcenv.contract_errors import UnsupportedContractCapability

        track_id = str(getattr(a, "track", None) or "msl-single")
        try:
            track = load_track_registry().require_available(track_id)
        except UnsupportedContractCapability as exc:
            print(f"track {track_id!r} cannot be selected: {exc}", file=sys.stderr)
            return 2
        binding = replace(binding, track_id=track.track_id)
        from tcenv.backends.metal.search_candidate import evaluate_candidate

        target = load_target(a.target)
        t0 = time.perf_counter()
        run = evaluate_candidate(
            binding.locator, src,
            target=target, mode=mode,
            seed=a.seed if a.seed is not None else 7,
            reps=a.reps, cases=None,
            track_id=binding.track_id,
        )
        elapsed = time.perf_counter() - t0
    except TaskBindingError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except CandidateDeliveryError as exc:
        # A submission whose shape does not match the selected delivery mechanism is the model's
        # mistake: report it as a typed rejection, not as a harness failure.
        print(json.dumps({"status": "rejected", "stage": getattr(exc, "stage", "delivery"), "message": str(exc)}, indent=2), file=sys.stderr)
        return 2
    result = run.result
    if result is None:
        print(json.dumps({"status": "rejected", "dispatch": run.record.document()}, indent=2, default=str))
        return 1
    document = dict(result.report)
    rep = EvalReport.from_json(document)
    if a.timed and self_check_dir is not None:
        bench = document.get("benchmark") if isinstance(document.get("benchmark"), dict) else {}
        cand = bench.get("candidate") if isinstance(bench.get("candidate"), dict) else {}
        append(self_check_dir, {
            "status": rep.status,
            "candidate_us": rep.candidate_us,
            "cv": cand.get("cv"),
            "speedup": rep.speedup,
            "elapsed_s": elapsed,
            "task": a.task, "dtype": a.dtype, "size": a.size,
            "target_id": a.target, "protocol_version": PROTOCOL_VERSION,
            "kernel_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
        })
    if a.json:
        print(json.dumps(document, indent=2, default=str))
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
    """@brief 用任务自带的 naive 参考内核自检评测架。

    调用链上没有模型请求，接口费用为零，与 ``eval`` 共用同一条编译、检查、计时路径。

    @param a 已解析的命令行参数。
    @return 成功为 0；缺少后端参考源码为 5；报告 status 非 ok 为 1。
    """
    from tasks import get_task

    from .evaluate import evaluate
    from .report import format_feedback

    from .target import require_active

    target = require_active(a.target)
    task = get_task(a.task, a.dtype, a.size)
    source = task.reference_for(target.backend)
    if not source.strip():
        print(f"task {a.task} has no reference source for backend {target.backend}", file=sys.stderr)
        return 5
    rep = evaluate(a.task, a.dtype, a.size, source, mode="quick" if a.quick else "full", seed=a.seed, reps=a.reps, progress_path=_live_progress(), target_id=a.target)
    print(format_feedback(rep))
    return 0 if rep.status == "ok" else 1


# 只测 MLX / PyTorch 基线。计时段独占 GPU 锁，测完用同一批中位数算 roofline，看基线离带宽和算力下界有多远。
def cmd_baseline(a: argparse.Namespace) -> int:
    """@brief 只测 MLX / PyTorch 基线并打印每个基线的 roofline 位置。

    @param a 已解析的命令行参数。
    @return 恒为 0。
    """
    from tasks import get_task

    from .backends.metal.baselines import build_runners
    from .bench import interleaved
    from .gpulock import gpu_lock
    from .backends.metal.profile import get_calibration, roofline
    from .backends.metal.testing import prepare_case

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
    """@brief 对已通过 quick 评测的内核做 Instruments 跟踪。

    Instruments trace of an *already validated* kernel (the full evaluation must pass first).

    @param a 已解析的命令行参数。
    @return 成功为 0；内核未通过 quick 评测为 1。
    """
    from .backends.metal import xctrace
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
    """@brief 在隔离环境里用 triton-msl 编译并计时任务的 Triton 规格。

    @param a 已解析的命令行参数。
    @return 环境缺失或全部任务失败时为 1，否则为 0。
    """
    from tasks import list_tasks

    from .backends.metal import triton_baseline as tb

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
    """@brief 测量本机峰值带宽与算力，供 roofline 使用。

    @param a 已解析的命令行参数。
    @return 成功为 0；目标不可用时为 5。
    """
    from .backends.metal.runtime import get_context
    from .backends.metal.profile import get_calibration
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
    """@brief 列出任务表及其契约摘要。

    默认列 **v2 标准目录**（``discover_standard_tasks``），因为那是当前的任务入口；
    ``--legacy`` 才列仍由 v1 装配的保留题。新人照着 README 敲的第一条命令不应再落在
    退役路径上。

    @param a 已解析的命令行参数。
    @return 恒为 0。
    """
    if getattr(a, "legacy", False):
        from tasks import get_task, list_tasks

        for n in list_tasks(include_phase2=True):
            t = get_task(n, "fp16", a.size)
            c = t.contract
            print(f"{n:14s} grid={c.grid} threads={c.threads} args={[x.name for x in c.args]} min_bytes={t.min_bytes/1e6:.1f}MB flops={t.flops/1e9:.2f}G")
        return 0

    from tcenv.contract_errors import TaskContractError
    from tcenv.taskcontract import discover_standard_tasks, load_task_contract

    for locator in discover_standard_tasks():
        try:
            contract = load_task_contract(locator.task_dir, size=a.size)
        except TaskContractError:
            contract = load_task_contract(locator.task_dir)  # 该题没有这个档位时退回它的默认实例
        instance = contract.instance
        work = instance.work_accounting
        tensors = ", ".join(instance.reference_inputs)
        print(
            f"{locator.task_id:20s} v2 dtype={contract.dtype:4s} size={contract.size:6s} "
            f"inputs=[{tensors}] min_bytes={work.min_bytes/1e6:.1f}MB flops={work.flops/1e9:.2f}G"
        )
    return 0


# 打印单个任务的说明、契约、ABI 桩和 Triton 源码，供人对照着写核。
def cmd_show(a: argparse.Namespace) -> int:
    """@brief 打印单个任务的说明、契约、ABI 骨架与 Triton 源码。

    @param a 已解析的命令行参数。
    @return 成功为 0；目标不可用时为 5。
    """
    from tasks import get_task

    from .lowering import LoweringError, require_target_profile
    from .target import TargetError, require_active

    try:
        target = require_active(a.target)
    except TargetError as exc:
        print(str(exc), file=sys.stderr)
        return 5
    try:
        require_target_profile(target.backend, target.id)
    except LoweringError as exc:
        print(str(exc), file=sys.stderr)
        return 5
    cached = get_task(a.task, a.dtype, a.size)
    # R2: ``contract.gpu`` comes from the registered LoweringProfile (the target was checked
    # above), so this display-only command no longer rewrites it.
    print(cached.description, "\n")
    print(cached.contract.describe(), "\n")
    print(cached.contract.abi_stub(), "\n")
    print(cached.triton_source)
    return 0


# 子命令分发。eval 走外部核，naive 走任务自带参考核，profile 先 quick 再跟踪，tasks 只列契约。
def main(argv: list[str] | None = None) -> int:
    """@brief ``tcenv`` 命令行入口：解析子命令并分发到对应处理函数。

    @param argv 参数列表；为 None 时使用 ``sys.argv``。
    @return 子命令处理函数的返回码。
    """
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

    p = sub.add_parser("eval-v2", help="evaluate a v2 candidate (msl-single file or msl-bundle envelope) against a standard task")
    p.add_argument("task")
    p.add_argument("kernel")
    _common(p)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--timed", action="store_true")
    p.add_argument("--self-check-dir", default=None)
    p.add_argument("--seed-report", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--min-improvement", type=float, default=SearchConfig.min_improvement)
    p.add_argument("--noise-z", type=float, default=SearchConfig.noise_z)
    p.add_argument("--require-contract-launch", action="store_true", help="v1-only flag; rejected for v2")
    p.add_argument(
        "--track", default=None,
        help="delivery mechanism: msl-single (one file) or msl-bundle (a packaged envelope); default msl-single",
    )
    p.add_argument("--target", default="apple-m4-max")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_eval_v2)

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

    p = sub.add_parser("tasks", help="list tasks (v2 standard entries; --legacy for the preserved v1 tasks)")
    p.add_argument("--size", default="paper")
    p.add_argument("--legacy", action="store_true", help="list the preserved v1 tasks instead of the v2 standard entries")
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
