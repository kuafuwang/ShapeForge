"""A/B of the search switches added for cost and for feedback direction. Preset ``base``, $3 per run, so the numbers can sit next to ``fa_flash``.

    uv run python -m experiments.search_ab --which conv2d          # conv2d, on and off
    uv run python -m experiments.search_ab --which fa              # flash_attention, 3 on and 3 off
    uv run python -m experiments.search_ab --which fa --arm on     # only the treatment arm
    uv run python -m experiments.search_ab --which fa --runs fa_off_1,fa_off_2

An existing non-empty directory is left alone. ``--max-total-usd`` skips a run whose budget would not fit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import experiment_commitment, fits_cap, run_one, write_cap
from .manifest import attach, ensure

SWITCHES = dict(
    stop_at_roofline=True,
    abi_guard=True,
    target_note=True,
    rich_history=True,
    status_lines=True,
    truncation_retry=True,
)
OFF = {k: False for k in SWITCHES}


def jobs_for(which: str, arm: str, n: int) -> list[tuple[str, str, bool]]:
    """@brief 列出要跑的实验组：目录名、任务名与开关是否打开。

    ``(directory name, task, switches on)``.

    @param which ``conv2d`` 或 ``fa``，决定任务与目录命名。
    @param arm ``on`` / ``off`` / ``both``。
    @param n ``fa`` 时每组的重复次数。
    @return 三个元组的列表：目录名、任务名、开关是否打开。
    """
    if which == "conv2d":
        arms = ("on", "off") if arm == "both" else (arm,)
        return [(f"conv2d_{name}", "conv2d", name == "on") for name in arms]
    arms = ("on", "off") if arm == "both" else (arm,)
    return [(f"fa_{name}_{i}", "flash_attention", name == "on") for name in arms for i in range(1, n + 1)]


def _occupied(path: Path) -> bool:
    """@brief 判断目录是否已存在且非空。

    @param path 待检查的目录。
    @return 目录存在且里面有文件时为 ``True``。
    """
    return path.exists() and any(path.iterdir())


def run_selected(out: Path, jobs: list[tuple[str, str, bool]], *, budget: float, max_total_usd: float | None, runner=None) -> list[dict]:
    """@brief 逐个跑选中的实验组，跳过已有文件的目录和放不下预算的运行。

    Skip occupied directories and runs that do not fit the cap. ``runner`` defaults to ``run_one``.

    @param out 实验输出根目录。
    @param jobs ``jobs_for`` 返回的实验组列表。
    @param budget 每次运行的花费上限，单位美元。
    @param max_total_usd 总预算上限；为 ``None`` 时不限。
    @param runner 实际跑一次的可调用对象；为 ``None`` 时用 ``run_one``。
    @return 每个实验组一项的备注列表，``skipped`` 标明是否被跳过及原因。
    """
    out.mkdir(parents=True, exist_ok=True)
    write_cap(out, max_total_usd)
    counts: dict[str, int] = {}
    for _name, _task, on in jobs:
        counts["on" if on else "off"] = counts.get("on" if on else "off", 0) + 1
    if jobs:
        ensure(
            out,
            kind="search_ab",
            driver="search_ab",
            frozen={
                "tasks": sorted({task for _name, task, _on in jobs}),
                "dtype": "fp16",
                "size": "paper",
                "preset": "base",
                "model": None,
                "committees": None,
                "budget_usd": budget,
                "max_total_usd": max_total_usd,
            },
            arms=[
                {"arm_id": arm, "delta": dict(SWITCHES) if arm == "on" else {}, "planned_n": counts[arm]}
                for arm in ("off", "on")
                if arm in counts
            ],
        )
    go = runner or run_one
    notes = []
    for name, task, on in jobs:
        attach(out, "on" if on else "off", name)
        dest = out / name
        if _occupied(dest):
            notes.append({"name": name, "skipped": "occupied"})
            print(f"(skip {name}: directory already holds files)", flush=True)
            continue
        spent = experiment_commitment(out)
        if not fits_cap(spent, budget, max_total_usd):
            notes.append({"name": name, "skipped": "over_cap"})
            print(f"(skip {name}: ${budget:.2f} does not fit under --max-total-usd)", flush=True)
            continue
        print(f"\n##### {name} switches={'on' if on else 'off'} budget=${budget:.2f}", flush=True)
        go(task, "fp16", "paper", dest, budget=budget, preset="base", **(SWITCHES if on else OFF))
        notes.append({"name": name, "skipped": None})
    return notes


def main(argv=None) -> int:
    """@brief ``python -m experiments.search_ab`` 的入口，按开关 A/B 跑搜索。

    @param argv 命令行参数；为 ``None`` 时取 ``sys.argv``。
    @return 进程退出码，正常结束时为 0。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("conv2d", "fa"), required=True)
    ap.add_argument("--arm", choices=("on", "off", "both"), default="both")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--n", type=int, default=3, help="trials per arm for --which fa")
    ap.add_argument("--out", default="runs/search_ab")
    ap.add_argument("--runs", default=None, help="comma-separated directory names to run; others in the matrix are not started")
    ap.add_argument("--max-total-usd", type=float, default=None)
    a = ap.parse_args(argv)
    selected = jobs_for(a.which, a.arm, a.n)
    if a.runs:
        wanted = {part.strip() for part in a.runs.split(",") if part.strip()}
        selected = [job for job in selected if job[0] in wanted]
    run_selected(Path(a.out), selected, budget=a.budget, max_total_usd=a.max_total_usd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
