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
    p.add_argument("--size", default=None, help="instance size tier; default is the task's own discovery.default_instance.size")


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
        print("--quick and --timed are mutually exclusive", file=sys.stderr)
        return 2
    if a.quick and a.validate:
        print("--quick and --validate are mutually exclusive", file=sys.stderr)
        return 2
    report = evaluate(
        a.kernel, task_name=a.task, dtype=a.dtype, size=a.size,
        quick=a.quick, timed=a.timed, validate=a.validate,
        eval_cache=a.eval_cache, measure_serial=a.measure_serial,
        seed_report=a.seed_report, min_improvement=a.min_improvement,
        noise_z=a.noise_z, require_contract_launch=a.require_contract_launch,
        self_check_dir=a.self_check_dir, seed=a.seed, reps=a.reps,
        target_id=a.target,
    )
    if a.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(format_feedback(report))
    return 0 if report.status == "ok" else 1


def cmd_eval_v2(a: argparse.Namespace) -> int:
    """@brief 评测一个 v2 候选（msl-single 文件或 msl-bundle envelope）。

    quick 停在编译、ABI、安全和正确性；timed 只做计时。
    """
    from .evaluate import evaluate_v2
    from .report import format_feedback

    if a.quick and a.timed:
        print("--quick and --timed are mutually exclusive", file=sys.stderr)
        return 2
    report = evaluate_v2(
        a.kernel, task_name=a.task, dtype=a.dtype, size=a.size,
        quick=a.quick, timed=a.timed, self_check_dir=a.self_check_dir,
        seed_report=a.seed_report, seed=a.seed, reps=a.reps,
        min_improvement=a.min_improvement, noise_z=a.noise_z,
        require_contract_launch=a.require_contract_launch,
        track=a.track, target_id=a.target,
        batch_submission=a.batch_submission,
    )
    if a.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(format_feedback(report))
    return 0 if report.status == "ok" else 1


def cmd_sweep_v2(a: argparse.Namespace) -> int:
    """@brief 对目录里最多 16 个完整候选文件计时并排名。"""
    from .sweep import sweep_v2

    if a.quick and a.timed:
        print("--quick and --timed are mutually exclusive", file=sys.stderr)
        return 2
    report = sweep_v2(
        a.directory, task_name=a.task, dtype=a.dtype, size=a.size,
        reps=a.reps, seed=a.seed, self_check_dir=a.self_check_dir,
        min_improvement=a.min_improvement, noise_z=a.noise_z,
        target_id=a.target, track=a.track,
    )
    if a.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(format_feedback(report))
    return 0 if report.status == "ok" else 1


def cmd_host_probe(a: argparse.Namespace) -> int:
    """@brief 在 CPU 上编译并试跑 Host Program（不占 GPU）。"""
    from .host_probe import host_probe

    return host_probe(
        a.task, source=a.source, case=a.case, seed=a.seed, target_id=a.target,
    )


def cmd_naive(a: argparse.Namespace) -> int:
    """@brief 评测任务自带的 naive 参考核（harness 自检）。"""
    from .evaluate import evaluate_naive

    report = evaluate_naive(
        a.task, dtype=a.dtype, size=a.size, quick=a.quick,
        seed=a.seed, reps=a.reps, target_id=a.target,
    )
    if a.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(format_feedback(report))
    return 0 if report.status == "ok" else 1


def cmd_baseline(a: argparse.Namespace) -> int:
    """@brief 对 MLX / torch-MPS 基线计时。"""
    from .evaluate import evaluate_baseline

    report = evaluate_baseline(
        a.task, dtype=a.dtype, size=a.size, reps=a.reps, target_id=a.target,
    )
    if a.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(format_feedback(report))
    return 0 if report.status == "ok" else 1


def cmd_profile(a: argparse.Namespace) -> int:
    """@brief 对通过 quick 评测的 kernel 做 Instruments (xctrace) Metal System Trace。"""
    from .profile import profile

    return profile(a.task, a.kernel, json=a.json, target_id=a.target)


def cmd_triton_baseline(a: argparse.Namespace) -> int:
    """@brief 用 triton-msl 编译 Triton 规格到 Metal 并缓存其与 MLX 的相对时间。"""
    from .triton_baseline import triton_baseline

    return triton_baseline(a.task, reps=a.reps, wait_quiet=a.wait_quiet, target_id=a.target)


def cmd_calibrate(a: argparse.Namespace) -> int:
    """@brief 测量 roofline 的峰值带宽 / TFLOPs。"""
    from .calibrate import calibrate

    return calibrate(a.target)


def cmd_tasks(a: argparse.Namespace) -> int:
    """@brief 列出任务（v2 标准目录；--legacy 列出保留的 v1 任务）。"""
    from .tasks import list_tasks

    return list_tasks(legacy=a.legacy, size=a.size)


def cmd_show(a: argparse.Namespace) -> int:
    """@brief 打印任务的契约和 Triton 源。"""
    from .show import show

    return show(a.task, dtype=a.dtype, size=a.size, target_id=a.target)


def cmd_portfolio_pack(a: argparse.Namespace) -> int:
    """@brief 把变体源目录打包成 shape-portfolio 父包 envelope。"""
    from .portfolio_assembly import pack

    return pack(a.task, src=a.src, out=a.out, dtype=a.dtype, sizes=a.sizes, target_id=a.target)


def cmd_portfolio_check(a: argparse.Namespace) -> int:
    """@brief 检查 shape-portfolio 父包的 guard 覆盖和投影。"""
    from .portfolio_assembly import check

    return check(a.task, a.file, dtype=a.dtype, sizes=a.sizes, target_id=a.target, json=a.json)


def cmd_eval_portfolio(a: argparse.Namespace) -> int:
    """@brief 评测 shape-portfolio 父包（搜索期；不发布）。"""
    from .portfolio_eval import evaluate_parent

    report = evaluate_parent(
        a.task, a.file, dtype=a.dtype, quick=a.quick, timed=a.timed,
        sizes=a.sizes, reps=a.reps, target_id=a.target, json=a.json,
        self_check_dir=a.self_check_dir,
    )
    if a.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(format_feedback(report))
    return 0 if report.status == "ok" else 1
