"""Paired yield experiment for the four search-output switches. Dry-run by default.

The arms match ``docs/实验批准单-搜索产出.md`` exactly. ``--dry-run``,
``--summarize`` and ``--render-only`` do not call a model or a GPU.
``--render-only`` writes the session files a treatment or control arm would
read; it produces no candidate and it is not a search result. ``--execute``
spends money and occupies a GPU only when ``--allow-spend``, ``--allow-gpu``,
a real ``--approved-by`` name and a finite ``--delta`` greater than zero are
all present. An experience-overlay plan uses those same four gates.
A compact task overlay stays unapproved: those gates do not execute it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tomllib
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
EXPERIENCE_FACTOR = "experience-overlay"
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


def build_experience_plan(
    *,
    overlay_sha256: str,
    overlay_path: str,
    tasks: tuple[str, ...] = ("gemm_silu",),
    dtype: str = DEFAULT_DTYPE,
    size: str = "small",
    pairs: int = 2,
    budget_usd: float = DEFAULT_BUDGET_USD,
    max_total_usd: float = 12.0,
    git_head: str | None = None,
) -> dict[str, Any]:
    """@brief 冻结「只差经验叠加层」的 AB/BA 计划。两臂搜索开关相同，不启动运行。

    默认四开关计划不受影响。处理臂的真实会话由 ``prepare_session`` 注入这一份
    冻结叠加层；对照臂不注入。``injection_ready`` 只表示这条路径存在。
    批准人、δ 和可容忍超支仍空着时，实弹搜索没有被授权。

    @return plan.json 的文档。
    """
    if not _SHA256.fullmatch(overlay_sha256):
        raise ValueError("experience overlay sha256 is not 64 hex characters")
    if pairs < 1:
        raise ValueError("experience plan needs at least one pair")
    cells: list[dict[str, Any]] = []
    pair_index = 0
    for task in tasks:
        for rep in range(1, pairs + 1):
            pair_index += 1
            order = "AB" if pair_index % 2 == 1 else "BA"
            arms = ("control", "treatment") if order == "AB" else ("treatment", "control")
            for arm in arms:
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
                    "overrides": dict(CONTROL_OVERRIDES),
                    "experience_overlay_sha256": overlay_sha256 if arm == "treatment" else None,
                })
    plan = {
        "schema_id": "experiments.search_yield_ab.experience-plan.1",
        "factor": EXPERIENCE_FACTOR,
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
        "treatment": dict(CONTROL_OVERRIDES),
        "unique_difference": "treatment session receives the frozen experience overlay; control does not. init_samples, task_ledger, plateau_after and sweep_tool stay equal.",
        "overlay_path": overlay_path,
        "overlay_sha256": overlay_sha256,
        "search_eval_seed": V2_SEARCH_SEED,
        "committees": _committees(),
        "metrics": {
            "primary": "median of treatment.confirmed_speedup minus control.confirmed_speedup over complete pairs",
            "secondary": ["lift_median", "unchanged_fraction", "spend_after_best_median", "self_checks_median"],
            "not_measured_in_dry_run": ["fraction of evaluated candidates that declare scratch"],
        },
        "stop": {
            "per_cell_soft_budget_usd": float(budget_usd),
            "hard_cap_usd": float(max_total_usd),
            "hard_cap_usd_meaning": (
                "commitment cap for whether a later cell may start. "
                "An in-flight request may exceed the per-cell soft budget, "
                "so this is not a ceiling the bill cannot pass. "
                "Tolerable overrun is not approved."
            ),
            "in_flight_request_may_exceed_soft_budget": True,
            "tolerable_overrun_approved": None,
            "exclude_unconfirmed": True,
            "best_speedup_is_not_a_substitute": True,
            "blank_delta_is_inconclusive": True,
            "complete_pairs_below_expected_is_inconclusive": True,
            "summarizer_default_allowed_regression": 0.0,
            "allowed_regression_approved": None,
            "publication_allowed": False,
        },
        "executed": False,
        "approved_by": None,
        "delta": None,
        "paid_search": "未授权/未执行",
        "publication_allowed": False,
        "injection_ready": True,
        "injection": _injection_document(),
        "cells": cells,
    }
    from experiments.experience_compact import assert_compact_scope_matches_plan

    assert_compact_scope_matches_plan(plan)
    return plan


def _committees() -> list[dict[str, str]]:
    """@brief 读 taic.toml 里的两个委员会。不读密钥，不发起请求。"""
    document = tomllib.loads((REPO / "taic.toml").read_text(encoding="utf-8"))
    found: list[dict[str, str]] = []
    for committee in document["committees"]:
        member = committee["members"][0]
        found.append({"name": committee["name"], "provider": member["provider"], "model": member["model"]})
    if [item["name"] for item in found] != ["first", "second"]:
        raise ValueError("experience plan expected the first and second committees")
    return found


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
    """@brief 顺序执行尚未占用的 cell。承诺上限只决定还要不要启动后续 cell。

    已经发出的请求可以超过单次软预算，结束后的实际账单也可以超过 ``max_total_usd``。
    这不是一张账单绝不会越过的天花板。可容忍超支要另有批准。

    @param out 实验根，必须在 ``runs/`` 下。
    @param plan ``build_plan`` 的文档。
    @param runner 替换 ``experiments.common.run_one`` 的可调用对象。
    @return 每个已考虑 cell 的备注。跳过原因是 ``occupied`` 或 ``over_cap``。
    """
    from experiments.common import experiment_commitment, fits_cap, run_one, write_cap

    destination = _runs_dir(out)
    if plan.get("factor") == EXPERIENCE_FACTOR:
        _require_experience_plan(plan)
        from experiments.experience_compact import assert_compact_execution_allowed

        assert_compact_execution_allowed(plan)
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
        extra = dict(cell["overrides"])
        if plan.get("factor") == EXPERIENCE_FACTOR:
            # Both arms share this run_seed. A v2 search request still uses the
            # orchestrator's literal 7; passing it here does not make
            # eval_seed(run_seed, phase, label, source) return 7.
            extra["seed"] = _experience_run_seed(plan)
            if cell["arm"] == "treatment":
                extra["experience_overlay_path"] = plan["overlay_path"]
                extra["experience_overlay_sha256"] = plan["overlay_sha256"]
        go(
            cell["task"],
            cell["dtype"],
            cell["size"],
            cell_dir,
            budget=budget,
            **extra,
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
    expected_run_seed = _experience_run_seed(plan) if plan.get("factor") == EXPERIENCE_FACTOR else None
    for key in order:
        arms = grouped[key]
        control = arms.get("control_present")
        treatment = arms.get("treatment_present")
        reason = _pair_reason(control, treatment, expected_run_seed=expected_run_seed)
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


def _injection_document() -> dict[str, Any]:
    """@brief 注入路径已经接上。这份声明不是搜索结果，也不授权实弹。"""
    from taic.experience_overlay import injection_document

    return injection_document()


def _experience_run_seed(plan: dict[str, Any]) -> int:
    """@brief 经验计划两臂共用的 ``run_seed``。必须是冻结的 v2 搜索种子 7。

    这个值传给 ``run_one(..., seed=)``，写进 ``run_config.json`` 的 ``seed`` 和
    ``run_seed``。v2 搜索期评测另用编排器里的字面量 7。``eval_seed(run_seed, phase,
    label, source)`` 派生的是另一颗种子，不能代替那个字面量。
    """
    value = plan.get("search_eval_seed")
    if isinstance(value, bool) or not isinstance(value, int) or value != V2_SEARCH_SEED:
        raise ValueError(f"experience plan search_eval_seed must be {V2_SEARCH_SEED}")
    return value


def _expected_experience_cells(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """@brief 按冻结矩阵重放每一格。奇数 pair 是 AB，偶数 pair 是 BA。"""
    tasks = plan["tasks"]
    dtype = plan["dtype"]
    size = plan["size"]
    pairs = plan["pairs"]
    budget = float(plan["budget_usd"])
    switches = dict(plan["control"])
    overlay_sha256 = plan["overlay_sha256"]
    cells: list[dict[str, Any]] = []
    pair_index = 0
    for task in tasks:
        for rep in range(1, pairs + 1):
            pair_index += 1
            order = "AB" if pair_index % 2 == 1 else "BA"
            arms = ("control", "treatment") if order == "AB" else ("treatment", "control")
            for arm in arms:
                cells.append({
                    "pair_index": pair_index,
                    "task": task,
                    "dtype": dtype,
                    "size": size,
                    "rep": rep,
                    "arm": arm,
                    "order": order,
                    "directory": f"{arm}/{task}_{size}_rep{rep}",
                    "budget_usd": budget,
                    "overrides": switches,
                    "experience_overlay_sha256": overlay_sha256 if arm == "treatment" else None,
                })
    return cells


def _require_experience_matrix(plan: dict[str, Any]) -> None:
    """@brief 格子必须等于任务矩阵。同一 pair 的题面相同，每臂恰好一格。"""
    tasks = plan.get("tasks")
    dtype = plan.get("dtype")
    size = plan.get("size")
    pairs = plan.get("pairs")
    cells = plan.get("cells")
    if (
        not isinstance(tasks, list)
        or not tasks
        or any(not isinstance(task, str) or not task for task in tasks)
    ):
        raise ValueError("experience plan task matrix is missing")
    if not isinstance(dtype, str) or not dtype or not isinstance(size, str) or not size:
        raise ValueError("experience plan dtype or size is missing")
    if isinstance(pairs, bool) or not isinstance(pairs, int) or pairs < 1:
        raise ValueError("experience plan pairs is not a positive integer")
    if not isinstance(cells, list):
        raise ValueError("experience plan cells are missing")
    expected_pairs = len(tasks) * pairs
    if plan.get("expected_pairs") != expected_pairs or len(cells) != expected_pairs * 2:
        raise ValueError("experience plan does not have one control and one treatment per pair")
    budget = plan.get("budget_usd")
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(float(budget))
    ):
        raise ValueError("experience plan budget_usd is missing")
    if not isinstance(plan.get("control"), dict):
        raise ValueError("experience plan search switches are missing")
    expected = _expected_experience_cells(plan)
    for index, (found, want) in enumerate(zip(cells, expected, strict=True)):
        if not isinstance(found, dict):
            raise ValueError(f"experience plan cell {index} is not an object")
        if found.get("arm") == "control" and found.get("experience_overlay_sha256") is not None:
            raise ValueError("control cell carries an overlay")
        for key, value in want.items():
            if found.get(key) != value:
                raise ValueError(f"experience plan cell {index} {key} does not match the frozen pair")
    if plan.get("committees") != _committees():
        raise ValueError("experience plan committees do not match taic.toml")
    from experiments.experience_compact import assert_compact_scope_matches_plan

    assert_compact_scope_matches_plan(plan)


def _require_experience_plan(plan: dict[str, Any]) -> None:
    """@brief 执行或渲染前核对：注入就绪、矩阵成对、委员会与 taic.toml 一致。"""
    from taic.experience_overlay import ExperienceOverlayError, verify_overlay_bytes

    if plan.get("factor") != EXPERIENCE_FACTOR:
        raise ValueError("not an experience-overlay plan")
    _experience_run_seed(plan)
    if plan.get("injection_ready") is not True:
        raise ValueError("experience plan is not injection_ready")
    if plan.get("control") != plan.get("treatment"):
        raise ValueError("experience plan search switches differ")
    if plan.get("executed") is not False or plan.get("paid_search") != "未授权/未执行":
        raise ValueError("experience plan claims an execution this entry did not record")
    try:
        verify_overlay_bytes(str(plan.get("overlay_path") or ""), str(plan.get("overlay_sha256") or ""))
    except ExperienceOverlayError as exc:
        raise ValueError(str(exc)) from exc
    _require_experience_matrix(plan)


def render_experience_plan(out: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """@brief 用 ``prepare_session`` 渲染每个 cell。不产生候选，不调用模型，不是搜索结果。

    @param out 实验根，必须在 ``runs/`` 下。
    @param plan ``build_experience_plan`` 的文档。
    @return ``render-only.json`` 的文档。
    """
    from dataclasses import replace

    from taic.experience_overlay import (
        OVERLAY_MARK,
        dumps,
        freeze_agent_guidance,
        install_run_overlay,
        run_identity,
        seedless_kernel_brief,
    )
    from taic.kernel_agent import prepare_session
    from taic.llm import SearchConfig
    from taic.orchestrator import _resolve_run_track
    from taic.prompts import system_prompt_v2
    from taic.task_binding import bind_task

    destination = _runs_dir(out)
    _require_experience_plan(plan)
    search = SearchConfig(**dict(plan["control"]))
    rendered_cells: list[dict[str, Any]] = []
    normalized_tasks: dict[str, str] = {}
    system_sha = ""
    for cell in plan["cells"]:
        run_dir = destination / "render-only" / cell["directory"]
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"render-only refuses occupied {cell['directory']}")
        guidance = run_dir / "agent_guidance"
        freeze_agent_guidance(guidance)
        if OVERLAY_MARK in (guidance / "experience.md").read_text(encoding="utf-8"):
            raise ValueError("the frozen guidance snapshot already contains the overlay")
        work = run_dir / "agent_work" / "init"
        task_binding = bind_task(cell["task"], cell["dtype"], cell["size"])
        if cell["arm"] == "treatment":
            install_run_overlay(run_dir, plan["overlay_path"], plan["overlay_sha256"])
        elif (run_dir / "experience_overlay.json").exists():
            raise ValueError("control run already has an overlay record")
        brief = seedless_kernel_brief(
            task_binding, search, repo=REPO, run_dir=run_dir, work_dir=work,
            guidance_dir=guidance, overlay=None,
        )
        prepared = prepare_session(brief, work)
        experience = (work / "experience.md").read_text(encoding="utf-8")
        copies = experience.count(OVERLAY_MARK)
        sidecar = work / "experience_overlay.json"
        record = None
        if cell["arm"] == "treatment":
            if copies != 1 or not sidecar.is_file():
                raise ValueError("treatment session does not contain exactly one overlay")
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            if record.get("run_id") != run_identity(run_dir):
                raise ValueError("session record run_id does not match this cell")
            if record.get("overlay_copies") != 1 or record.get("authoritative_task_rules") is not False:
                raise ValueError("treatment overlay record is not a single non-authoritative copy")
            if record.get("rendered_experience_sha256") != hashlib.sha256(experience.encode("utf-8")).hexdigest():
                raise ValueError("rendered experience sha does not match the session record")
            if experience.count(OVERLAY_MARK) != 1:
                raise ValueError("treatment experience repeated the overlay")
        elif copies != 0 or sidecar.exists():
            raise ValueError("control session contains the overlay")
        task_text = (work / "TASK.md").read_text(encoding="utf-8")
        normalized = task_text.replace(str(run_dir.resolve()), "{run}")
        previous = normalized_tasks.get(cell["arm"])
        if previous is None:
            normalized_tasks[cell["arm"]] = normalized
        elif previous != normalized:
            raise ValueError(f"{cell['arm']} task text is not stable across cells")
        track, track_id = _resolve_run_track(task_binding, "msl-single", None)
        prompt_binding = replace(task_binding, track_id=track_id, track_version=int(track.track_version))
        prompt = system_prompt_v2(prompt_binding, use_tools=True, target_id="apple-m4-max")
        if OVERLAY_MARK in prompt:
            raise ValueError("the system prompt contains the overlay")
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if system_sha and prompt_sha != system_sha:
            raise ValueError("system prompt differs across cells")
        system_sha = prompt_sha
        rendered_cells.append({
            "arm": cell["arm"],
            "directory": cell["directory"],
            "run_id": run_identity(run_dir),
            "task_sha256": prepared.task_sha256,
            "normalized_task_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            "experience_sha256": hashlib.sha256(experience.encode("utf-8")).hexdigest(),
            "experience_bytes": len(experience.encode("utf-8")),
            "overlay_copies": copies,
            "system_prompt_sha256": system_sha,
            "experience_overlay": record,
            "produces_candidates": False,
            "calls_model": False,
            "is_search_result": False,
        })
    if normalized_tasks.get("control") != normalized_tasks.get("treatment"):
        raise ValueError("control and treatment task text differ by more than the run path")
    manifest = {
        "schema_id": "experiments.search_yield_ab.render-only.1",
        "produces_candidates": False,
        "calls_model": False,
        "is_search_result": False,
        "publication_allowed": False,
        "paid_search": "未授权/未执行",
        "injection_ready": True,
        "search_eval_seed": plan["search_eval_seed"],
        "committees": plan["committees"],
        "system_prompt_sha256": system_sha,
        "cells": rendered_cells,
    }
    (destination / "render-only.json").write_text(dumps(manifest), encoding="utf-8")
    return manifest


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
    parser.add_argument("--render-only", action="store_true", help="write the session files only; produces no candidate, calls no model, and is not a search result")
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
    parser.add_argument("--factor", choices=("switches", EXPERIENCE_FACTOR), default="switches")
    parser.add_argument("--overlay", default="")
    args = parser.parse_args(argv)
    modes = [name for name, enabled in (("dry-run", args.dry_run), ("summarize", args.summarize), ("execute", args.execute), ("render-only", args.render_only)) if enabled]
    if len(modes) > 1:
        print("search_yield_ab: choose only one of --dry-run, --summarize, --execute, --render-only", file=sys.stderr)
        return 2
    if args.render_only and args.factor != EXPERIENCE_FACTOR:
        print("search_yield_ab: --render-only is only valid with --factor experience-overlay", file=sys.stderr)
        return 2
    if args.execute:
        refusal = _execute_refusal(args)
        if refusal:
            print(f"search_yield_ab: {refusal}", file=sys.stderr)
            return 2
    if args.factor == EXPERIENCE_FACTOR and not args.overlay:
        print("search_yield_ab: --factor experience-overlay requires --overlay", file=sys.stderr)
        return 2
    if args.factor != EXPERIENCE_FACTOR and args.overlay:
        print("search_yield_ab: --overlay is only valid with --factor experience-overlay", file=sys.stderr)
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
        if args.factor == EXPERIENCE_FACTOR:
            if tasks == DEFAULT_TASKS and args.size == DEFAULT_SIZE:
                raise ValueError(
                    "experience-overlay does not reuse the four-switch task matrix; pass --tasks and --size"
                )
            overlay = Path(args.overlay)
            if not overlay.is_file():
                raise ValueError(f"overlay file does not exist: {args.overlay}")
            overlay_sha256 = hashlib.sha256(overlay.read_bytes()).hexdigest()
            from taic.experience_overlay import ExperienceOverlayError, verify_overlay_bytes

            try:
                verify_overlay_bytes(args.overlay, overlay_sha256)
            except ExperienceOverlayError as exc:
                raise ValueError(str(exc)) from exc
            plan = build_experience_plan(
                overlay_sha256=overlay_sha256,
                overlay_path=args.overlay,
                tasks=tasks,
                dtype=args.dtype,
                size=args.size,
                pairs=args.pairs,
                budget_usd=args.budget_usd,
                max_total_usd=args.max_total_usd,
            )
            if args.execute or args.render_only:
                _require_experience_plan(plan)
            if args.render_only:
                write_plan(out, plan)
                manifest = render_experience_plan(out, plan)
                print(json.dumps({
                    "render_only": str(out / "render-only.json"),
                    "cells": len(manifest["cells"]),
                    "injection_ready": True,
                    "produces_candidates": False,
                    "calls_model": False,
                    "is_search_result": False,
                    "paid_search": "未授权/未执行",
                }, indent=2))
                return 0
            if args.execute:
                from experiments.experience_compact import assert_compact_execution_allowed

                path = out / "plan.json"
                if path.is_file():
                    existing = json.loads(path.read_text(encoding="utf-8"))
                    if existing.get("overlay_sha256") != plan["overlay_sha256"]:
                        raise ValueError("overlay sha256 does not match the file")
                    plan = existing
                    _require_experience_plan(plan)
                assert_compact_execution_allowed(plan)
                if not path.is_file():
                    write_plan(out, plan)
                notes = execute_plan(out, plan, runner=runner)
                print(json.dumps({"executed": notes}, indent=2))
                return 0
            written = write_plan(out, plan)
            print(json.dumps({
                "plan": str(written),
                "cells": len(plan["cells"]),
                "nominal_total_usd": plan["nominal_total_usd"],
                "paid_search": plan["paid_search"],
                "injection_ready": plan["injection_ready"],
                "produces_candidates": False,
                "calls_model": False,
                "is_search_result": False,
            }, indent=2))
            return 0
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


def _pair_reason(
    control: dict | None,
    treatment: dict | None,
    *,
    expected_run_seed: int | None = None,
) -> str | None:
    """@brief 一对不能进入中位数的原因。两边都合格时返回 None。

    搜索期 ``eval_seed`` 必须是 v2 的字面量 7。经验计划另外要求两边 ``run_seed``
    都等于冻结值。四开关计划不传 ``expected_run_seed``，不读 ``run_seed``。
    """
    if not control:
        return "control result.json is missing or unreadable"
    if not treatment:
        return "treatment result.json is missing or unreadable"
    for arm, document in (("control", control), ("treatment", treatment)):
        if _search_seed(document) != V2_SEARCH_SEED:
            return f"{arm} did not record v2 search evaluation seed {V2_SEARCH_SEED}"
        if expected_run_seed is not None:
            recorded = document.get("run_seed")
            if isinstance(recorded, bool) or not isinstance(recorded, int) or recorded != expected_run_seed:
                return f"{arm} run_seed is not the frozen search seed {expected_run_seed}"
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
