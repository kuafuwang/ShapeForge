"""Paired yield experiment for the four search-output switches. Dry-run by default.

The arms match ``docs/实验批准单-搜索产出.md`` exactly. ``--dry-run`` and
``--summarize`` do not call a model or a GPU. ``--execute`` spends money and
occupies a GPU only when ``--allow-spend``, ``--allow-gpu``, a real
``--approved-by`` name and a finite ``--delta`` greater than zero are all present.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from experiments.search_yield import summarize

REPO = Path(__file__).resolve().parents[1]
CONTROL_OVERRIDES = {"init_samples": 1, "task_ledger": False, "plateau_after": 0, "sweep_tool": False}
TREATMENT_OVERRIDES = {"init_samples": 3, "task_ledger": True, "plateau_after": 2, "sweep_tool": True}
DEFAULT_TASKS = ("conv2d", "flash_attention")
DEFAULT_DTYPE = "fp16"
DEFAULT_SIZE = "paper"
DEFAULT_PAIRS = 3
DEFAULT_BUDGET_USD = 3.0
DEFAULT_MAX_TOTAL_USD = 36.0
V2_SEARCH_SEED = 7
_FORBIDDEN_APPROVERS = frozenset({"cli", "agent", "auto"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def build_plan(
    *,
    tasks: tuple[str, ...] = DEFAULT_TASKS,
    dtype: str = DEFAULT_DTYPE,
    size: str = DEFAULT_SIZE,
    pairs: int = DEFAULT_PAIRS,
    budget_usd: float = DEFAULT_BUDGET_USD,
    max_total_usd: float = DEFAULT_MAX_TOTAL_USD,
    git_head: str | None = None,
) -> dict[str, Any]:
    """@brief 冻结 2 题 × 3 对的 AB/BA 计划。不启动任何运行。

    全局配对序号为奇数时先对照后处理，偶数时反过来。

    @return plan.json 的文档。
    """
    cells: list[dict[str, Any]] = []
    pair_index = 0
    for task in tasks:
        for rep in range(1, pairs + 1):
            pair_index += 1
            order = "AB" if pair_index % 2 == 1 else "BA"
            arms = ("control", "treatment") if order == "AB" else ("treatment", "control")
            for arm in arms:
                overrides = dict(CONTROL_OVERRIDES if arm == "control" else TREATMENT_OVERRIDES)
                cells.append({
                    "pair_index": pair_index,
                    "task": task,
                    "dtype": dtype,
                    "size": size,
                    "rep": rep,
                    "arm": arm,
                    "order": order,
                    "directory": f"{arm}/{task}_{size}_rep{rep}",
                    "budget_usd": float(budget_usd),
                    "overrides": overrides,
                })
    return {
        "schema_id": "experiments.search_yield_ab.plan.1",
        "git_head": git_head if git_head is not None else _git_head(),
        "tasks": list(tasks),
        "dtype": dtype,
        "size": size,
        "pairs": pairs,
        "expected_pairs": len(tasks) * pairs,
        "budget_usd": float(budget_usd),
        "max_total_usd": float(max_total_usd),
        "nominal_total_usd": float(len(cells) * budget_usd),
        "control": dict(CONTROL_OVERRIDES),
        "treatment": dict(TREATMENT_OVERRIDES),
        "cells": cells,
    }


def write_plan(out: Path, plan: dict[str, Any]) -> Path:
    """@brief 把计划写到 ``out/plan.json``。``out`` 必须位于仓库 ``runs/`` 之下。"""
    destination = _runs_dir(out)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def execute_plan(
    out: Path,
    plan: dict[str, Any],
    *,
    runner: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """@brief 顺序执行尚未占用的 cell，累计花费顶到上限就停。

    @param out 实验根，必须在 ``runs/`` 下。
    @param plan ``build_plan`` 的文档。
    @param runner 替换 ``experiments.common.run_one`` 的可调用对象。
    @return 每个已考虑 cell 的备注。跳过原因是 ``occupied`` 或 ``over_cap``。
    """
    from experiments.common import experiment_commitment, fits_cap, run_one, write_cap

    destination = _runs_dir(out)
    budget = float(plan["budget_usd"])
    cap = float(plan["max_total_usd"])
    write_cap(destination, cap)
    go = runner or run_one
    notes: list[dict[str, Any]] = []
    for cell in plan["cells"]:
        cell_dir = destination / cell["directory"]
        if _occupied(cell_dir):
            notes.append({"directory": cell["directory"], "skipped": "occupied"})
            continue
        spent = experiment_commitment(destination)
        if not fits_cap(spent, budget, cap):
            notes.append({"directory": cell["directory"], "skipped": "over_cap"})
            break
        go(
            cell["task"],
            cell["dtype"],
            cell["size"],
            cell_dir,
            budget=budget,
            **dict(cell["overrides"]),
        )
        notes.append({"directory": cell["directory"], "skipped": None})
        if experiment_commitment(destination) + 1e-9 >= cap:
            break
    return notes


def summarize_experiment(
    out: Path,
    *,
    delta: float | None = None,
    allowed_regression: float = 0.0,
) -> dict[str, Any]:
    """@brief 只读汇总完整配对的确认加速比之差和四项次指标。

    未确认的一边不用 ``best_speedup`` 填补。完整配对少于计划数，或没有给出 δ，
    结论强制为 ``inconclusive``。允许退化默认是 0：任一配对差为负就不判处理臂获胜。

    @param out 含 ``plan.json`` 和 cell 目录的实验根。
    @param delta 最小中位差。None 表示批准单里的 δ 仍是空白。
    @param allowed_regression 单对允许的负差绝对值。
    @return 汇总文档。
    """
    destination = _runs_dir(out)
    plan_path = destination / "plan.json"
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {plan_path}: {exc}") from exc
    if not isinstance(plan, dict) or not isinstance(plan.get("cells"), list):
        raise ValueError(f"{plan_path} is not a search_yield_ab plan")
    grouped: dict[tuple, dict[str, dict]] = {}
    order: list[tuple] = []
    for cell in plan["cells"]:
        key = (cell["task"], cell["size"], int(cell["rep"]))
        if key not in grouped:
            grouped[key] = {}
            order.append(key)
        result_path = destination / cell["directory"] / "result.json"
        document = _load(result_path)
        grouped[key][cell["arm"]] = document or {}
        grouped[key][cell["arm"] + "_present"] = document
    differences: list[float] = []
    excluded: list[dict[str, Any]] = []
    for key in order:
        arms = grouped[key]
        control = arms.get("control_present")
        treatment = arms.get("treatment_present")
        reason = _pair_reason(control, treatment)
        if reason is not None:
            excluded.append({"task": key[0], "size": key[1], "rep": key[2], "reason": reason})
            continue
        differences.append(float(treatment["confirmed_speedup"]) - float(control["confirmed_speedup"]))
    expected = int(plan.get("expected_pairs") or 0)
    median = _median(differences)
    control_stats = summarize(destination / "control")
    treatment_stats = summarize(destination / "treatment")
    verdict = _verdict(differences, expected, delta, median, allowed_regression)
    return {
        "schema_id": "experiments.search_yield_ab.summary.1",
        "expected_pairs": expected,
        "complete_pairs": len(differences),
        "paired_differences": differences,
        "median_difference": median,
        "excluded_pairs": excluded,
        "delta": delta,
        "allowed_regression": allowed_regression,
        "verdict": verdict,
        "control": _yield_view(control_stats),
        "treatment": _yield_view(treatment_stats),
    }


def main(argv: list[str] | None = None, *, runner: Callable[..., Any] | None = None) -> int:
    """@brief 默认 dry-run。四道人工门禁不全时 ``--execute`` 返回 2 且不调用 runner。

    @param argv 命令行。
    @param runner 测试注入的单次运行函数。
    @return 0 或 2。
    """
    parser = argparse.ArgumentParser(description="Paired search-yield experiment. Dry-run unless every spend gate is present.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-spend", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--approved-by", default="")
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--allowed-regression", type=float, default=0.0)
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD)
    parser.add_argument("--max-total-usd", type=float, default=DEFAULT_MAX_TOTAL_USD)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--size", default=DEFAULT_SIZE)
    args = parser.parse_args(argv)
    modes = [name for name, enabled in (("dry-run", args.dry_run), ("summarize", args.summarize), ("execute", args.execute)) if enabled]
    if len(modes) > 1:
        print("search_yield_ab: choose only one of --dry-run, --summarize, --execute", file=sys.stderr)
        return 2
    if args.execute:
        refusal = _execute_refusal(args)
        if refusal:
            print(f"search_yield_ab: {refusal}", file=sys.stderr)
            return 2
    try:
        out = _runs_dir(Path(args.out))
    except ValueError as exc:
        print(f"search_yield_ab: {exc}", file=sys.stderr)
        return 2
    try:
        if args.summarize:
            summary = summarize_experiment(out, delta=args.delta, allowed_regression=args.allowed_regression)
            print(json.dumps(summary, indent=2))
            return 0
        tasks = tuple(part.strip() for part in str(args.tasks).split(",") if part.strip())
        plan = build_plan(
            tasks=tasks,
            dtype=args.dtype,
            size=args.size,
            pairs=args.pairs,
            budget_usd=args.budget_usd,
            max_total_usd=args.max_total_usd,
        )
        if args.execute:
            path = out / "plan.json"
            if path.is_file():
                plan = json.loads(path.read_text(encoding="utf-8"))
            else:
                write_plan(out, plan)
            notes = execute_plan(out, plan, runner=runner)
            print(json.dumps({"executed": notes}, indent=2))
            return 0
        written = write_plan(out, plan)
    except ValueError as exc:
        print(f"search_yield_ab: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"plan": str(written), "cells": len(plan["cells"]), "nominal_total_usd": plan["nominal_total_usd"]}, indent=2))
    return 0


def _execute_refusal(args: argparse.Namespace) -> str | None:
    """@brief 四道门禁缺任何一道就拒绝，且此时还没调用 runner。

    ``--delta`` 必须是有限的正数。NaN 和 inf 的比较结果不是“大于 0”，不能放行。
    """
    if not args.allow_spend or not args.allow_gpu:
        return "refusing to execute without both --allow-spend and --allow-gpu"
    name = str(args.approved_by or "").strip()
    if not name or name.casefold() in _FORBIDDEN_APPROVERS:
        return "refusing to execute without --approved-by set to a real person's name"
    delta = args.delta
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta <= 0:
        return "refusing to execute without a positive finite --delta"
    return None


def _pair_reason(control: dict | None, treatment: dict | None) -> str | None:
    """@brief 一对不能进入中位数的原因。两边都合格时返回 None。"""
    if not control:
        return "control result.json is missing or unreadable"
    if not treatment:
        return "treatment result.json is missing or unreadable"
    for arm, document in (("control", control), ("treatment", treatment)):
        if _search_seed(document) != V2_SEARCH_SEED:
            return f"{arm} did not record v2 search evaluation seed {V2_SEARCH_SEED}"
        if document.get("confirmed_status") != "ok":
            return f"{arm} confirmed_status is {document.get('confirmed_status')!r}, not ok"
        digest = document.get("confirmation_source_sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            return f"{arm} confirmation_source_sha256 is not 64 hex characters"
        speed = document.get("confirmed_speedup")
        if (
            isinstance(speed, bool)
            or not isinstance(speed, (int, float))
            or not math.isfinite(float(speed))
            or float(speed) <= 0
        ):
            return (
                f"{arm} confirmed_speedup is not a finite number greater than zero; "
                "best_speedup is not a substitute"
            )
    return None


def _search_seed(document: dict) -> int | None:
    """@brief 从 trace 的搜索评测行读出唯一的 eval_seed。"""
    seeds: list[int] = []
    for row in document.get("trace") or []:
        if not isinstance(row, dict) or "eval_seed" not in row:
            continue
        origin = row.get("eval_seed_from")
        if origin not in (None, "search"):
            continue
        value = row.get("eval_seed")
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        if origin is None and value != V2_SEARCH_SEED:
            continue
        seeds.append(value)
    if not seeds:
        top = document.get("search_eval_seed")
        if isinstance(top, bool) or not isinstance(top, int):
            return None
        return top
    if len(set(seeds)) != 1:
        return None
    return seeds[0]


def _verdict(
    differences: list[float],
    expected: int,
    delta: float | None,
    median: float | None,
    allowed_regression: float,
) -> str:
    """@brief 样本不足或 δ 空白时只能是 inconclusive。"""
    if (
        delta is None
        or len(differences) < expected
        or median is None
        or not math.isfinite(median)
        or any(not math.isfinite(diff) for diff in differences)
    ):
        return "inconclusive"
    if any(diff < -float(allowed_regression) for diff in differences):
        return "no_gain"
    if median >= float(delta):
        return "treatment_wins"
    return "no_gain"


def _yield_view(stats: dict) -> dict[str, Any]:
    """@brief 只留下批准单点名的四项次指标，算法仍是 search_yield.summarize。"""
    return {
        "n": stats.get("n"),
        "lift_median": stats.get("lift_median"),
        "unchanged_fraction": stats.get("unchanged_fraction"),
        "spend_after_best_median": stats.get("spend_after_best_median"),
        "self_checks_median": stats.get("self_checks_median"),
    }


def _median(values: list[float]) -> float | None:
    """@brief 偶数个样本取中间两数的平均。"""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _occupied(path: Path) -> bool:
    """@brief 目录里已经有文件时不再启动。"""
    return path.exists() and any(path.iterdir())


def _load(path: Path) -> dict | None:
    """@brief 读一份 JSON 对象。损坏或缺失时返回 None。"""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _runs_dir(path: Path) -> Path:
    """@brief 解析实验目录，拒绝跑出仓库 ``runs/``。"""
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (REPO / candidate).resolve()
    runs = (REPO / "runs").resolve()
    try:
        resolved.relative_to(runs)
    except ValueError as exc:
        raise ValueError(f"{path} is not under {runs}") from exc
    if resolved == runs:
        raise ValueError(f"{path} must be a directory under runs/, not runs/ itself")
    return resolved


def _git_head() -> str:
    """@brief 冻结当前 HEAD。失败时拒绝写一份没有提交号的计划。"""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"git rev-parse HEAD failed: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
