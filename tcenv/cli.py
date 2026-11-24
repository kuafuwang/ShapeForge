"""``tcenv`` command line: parse the subcommand and dispatch to its handler.

All handlers live in :mod:`tcenv.cli_common` (anything every backend shares) or
:mod:`tcenv.cli_metal` (Metal only). Adding a backend-specific command there
keeps it out of another target's surface entirely.
"""

from __future__ import annotations

import argparse
import sys

from tcenv.cli_common import (
    _common,
    _live_progress,
    _size_for,
    cmd_calibrate,
    cmd_eval_portfolio,
    cmd_eval_v2,
    cmd_host_probe,
    cmd_portfolio_check,
    cmd_portfolio_pack,
    cmd_profile,
    cmd_show,
    cmd_sweep_v2,
    cmd_tasks,
)
from tcenv.cli_metal import cmd_ab, cmd_paired_compare


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
    p.add_argument(
        "--track", default=None,
        help="delivery mechanism: msl-single (one file) or msl-bundle (a packaged envelope); default msl-single",
    )
    p.add_argument("--target", default="apple-m4-max")
    p.add_argument(
        "--batch-submission",
        choices=("single_buffer", "per_launch_set_event"),
        default="single_buffer",
        help="candidate batch submission; per_launch_set_event uses per-set SharedEvent dependencies for eligible single-stage candidates",
    )
    p.add_argument(
        "--control", action="store_true",
        help="timed only: launch the same candidate bytes a second time inside the same interleaved "
             "window, so the report says how large a difference this window can resolve",
    )
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_eval_v2)

    p = sub.add_parser("sweep-v2", help="time up to 16 complete candidate files and rank them by speedup")
    p.add_argument("task")
    p.add_argument("directory")
    _common(p)
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--self-check-dir", default=None, help="quota and hang log; same directory as eval-v2 --timed")
    p.add_argument("--min-improvement", type=float, default=SearchConfig.min_improvement)
    p.add_argument("--noise-z", type=float, default=SearchConfig.noise_z)
    p.add_argument("--target", default="apple-m4-max")
    p.add_argument("--track", default=None, help="delivery mechanism; default msl-single")
    p.set_defaults(fn=cmd_sweep_v2)

    p = sub.add_parser("host-probe", help="compile and run a host/tiling.c Host Program on the CPU (no GPU)")
    p.add_argument("task")
    p.add_argument("--source", required=True, help="path to the candidate host/tiling.c")
    _common(p)
    p.add_argument("--case", default="canonical")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--target", default="apple-m4-max")
    p.set_defaults(fn=cmd_host_probe)

    p = sub.add_parser("ab", help="interleaved timing of 2-16 candidate files in one GPU window (Metal-only, read-only)")
    p.add_argument("files", nargs="+")
    p.add_argument("--task", required=True)
    p.add_argument("--dtype", required=True, choices=["fp16", "bf16"])
    p.add_argument("--size", required=True)
    p.add_argument("--backend", default="metal", help="only 'metal' is supported")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--track", default="msl-single")
    p.add_argument("--reps", type=int, default=16)
    p.add_argument("--target", default="apple-m4-max")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_ab)

    p = sub.add_parser("paired-compare", help="nominate a paired comparison of two candidate files (local only, no task-book text yet)")
    p.add_argument("left")
    p.add_argument("right")
    p.add_argument("--backend", default="metal")
    p.add_argument("--self-check-dir", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--task", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--size", default=None)
    p.add_argument("--track", default="msl-single")
    p.add_argument("--reps", type=int, default=16, help="interleaved rounds; 4 cannot give a 1%% ratio a usable standard error")
    p.set_defaults(fn=cmd_paired_compare)

    p = sub.add_parser("profile", help="Instruments (xctrace) Metal System Trace of a kernel that passes the quick evaluation")
    p.add_argument("task")
    p.add_argument("kernel")
    _common(p)
    p.add_argument("--json", action="store_true")
    p.add_argument("--target", default="apple-m4-max")
    p.set_defaults(fn=cmd_profile)

    p = sub.add_parser("calibrate", help="measure peak bandwidth / TFLOPs for the roofline")
    p.add_argument("--target", default="apple-m4-max")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("portfolio-pack", help="pack a variant source tree into a shape-portfolio parent envelope")
    p.add_argument("task")
    p.add_argument("--src", required=True, help="directory with coverage.json and variants/")
    p.add_argument("--out", required=True, help="envelope path; not written when packing fails")
    p.add_argument("--dtype", required=True, choices=["fp16", "bf16"])
    p.add_argument("--sizes", default=None, help="comma-separated sizes; default is every size with benchmark usage")
    p.add_argument("--target", default="apple-m4-max")
    p.set_defaults(fn=cmd_portfolio_pack)

    p = sub.add_parser("portfolio-check", help="check a shape-portfolio parent for guard coverage and projection")
    p.add_argument("task")
    p.add_argument("file")
    p.add_argument("--dtype", required=True, choices=["fp16", "bf16"])
    p.add_argument("--sizes", default=None, help="comma-separated sizes; must match the sizes used to pack")
    p.add_argument("--target", default="apple-m4-max")
    p.add_argument("--json", action="store_true", help="print only the check document")
    p.set_defaults(fn=cmd_portfolio_check)

    p = sub.add_parser("eval-portfolio", help="evaluate a shape-portfolio parent (search-time; does not publish)")
    p.add_argument("task")
    p.add_argument("file")
    p.add_argument("--dtype", required=True, choices=["fp16", "bf16"])
    p.add_argument("--quick", action="store_true", help="correctness of every domain point")
    p.add_argument("--timed", action="store_true", help="golden correctness plus benchmark shards and both books")
    p.add_argument("--sizes", default=None, help="comma-separated sizes; must match the sizes used to pack")
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--target", default="apple-m4-max")
    p.add_argument("--json", action="store_true", help="print only the evaluation document")
    p.add_argument("--self-check-dir", default=None, help="timed only: hang/budget log, one row per benchmark point")
    p.set_defaults(fn=cmd_eval_portfolio)

    p = sub.add_parser("tasks", help="list current v2 standard tasks")
    p.add_argument("--size", default="paper")
    p.set_defaults(fn=cmd_tasks)

    p = sub.add_parser("show", help="print the current v2 task statement, reference and ABI")
    p.add_argument("task")
    _common(p)
    p.add_argument("--target", default="apple-m4-max")
    p.set_defaults(fn=cmd_show)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
