"""Controlled harness experiments: inspectable dry-run first, explicit live execution later."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from taic.harness_experiment import ExperimentError, collect_arm, materialize_arm, read_plan, summarize_result, toml_document
from tcenv.schema import write_json_atomic


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)


def _cell_id(seed: int, arm: str, ordinal: int) -> str:
    return f"seed-{seed}-pair-{ordinal}-{arm}"


def build_cells(plan: dict, *, out_dir: Path, repo: Path) -> list[dict]:
    """Build deterministic AB/BA pairs. This creates directories, never starts a model or GPU."""
    plan = read_plan(plan)
    out_dir = Path(out_dir)
    repo = Path(repo)
    out_resolved = out_dir.resolve()
    repo_resolved = repo.resolve()
    try:
        relative = out_resolved.relative_to(repo_resolved)
    except ValueError as exc:
        raise ExperimentError("experiment worktrees must stay inside the repository") from exc
    if not relative.parts or relative.parts[0] != "runs":
        raise ExperimentError("experiment worktrees must be created under runs/")
    base = plan["base_commit"]
    if not base:
        raise ExperimentError("an executable plan needs an explicit base_commit")
    _git(repo, "cat-file", "-e", f"{base}^{{commit}}")
    patch = out_dir / "treatment.patch"
    if plan["change_sha256"]:
        if not patch.is_file() or hashlib.sha256(patch.read_bytes()).hexdigest() != plan["change_sha256"]:
            raise ExperimentError("the treatment patch is missing or differs from the plan")
    cells = []
    previous = out_dir / "cells.json"
    if previous.is_file():
        old = json.loads(previous.read_text(encoding="utf-8"))
        if old.get("experiment_id") == plan["experiment_id"] and isinstance(old.get("cells"), list):
            return old["cells"]
    for index, seed in enumerate(plan["seeds"]):
        for arm in (("control", "treatment") if index % 2 == 0 else ("treatment", "control")):
            cell = _cell_id(seed, arm, index)
            worktree = out_dir / "worktrees" / cell
            run_dir = out_dir / "runs" / cell
            config = out_dir / "configs" / f"{cell}.toml"
            raw_config = out_dir / "configs" / f"{cell}.json"
            if worktree.exists() and not (worktree / ".git").exists() or run_dir.exists() and any(run_dir.iterdir()):
                raise ExperimentError(f"refusing to overwrite existing experiment cell {cell}")
            worktree.parent.mkdir(parents=True, exist_ok=True)
            config.parent.mkdir(parents=True, exist_ok=True)
            try:
                if not worktree.exists():
                    _git(repo, "worktree", "add", "--detach", str(worktree), base)
                if arm == "treatment" and subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=worktree, capture_output=True, text=True).returncode != 0:
                    applied = subprocess.run(["git", "apply", str(patch)], cwd=worktree, capture_output=True, text=True)
                    if applied.returncode:
                        raise ExperimentError(f"treatment patch does not apply at {base}: {applied.stderr or applied.stdout}")
                verify_cell_import(worktree)
                material = materialize_arm(plan, arm, seed)
                config.write_text(toml_document(material["raw"]), encoding="utf-8")
                write_json_atomic(raw_config, material["raw"])
            except Exception:
                if worktree.exists():
                    subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo, capture_output=True)
                raise
            cells.append({"cell": cell, "seed": seed, "arm": arm, "pair_index": index, "worktree": str(worktree), "run_dir": str(run_dir), "config": str(config), "raw_config": str(raw_config), "base_commit": base, **material})
    write_json_atomic(out_dir / "cells.json", {"schema_version": 1, "experiment_id": plan["experiment_id"], "cells": cells})
    return cells


def dry_run(plan: dict, *, out_dir: Path, repo: Path) -> dict:
    cells = build_cells(plan, out_dir=out_dir, repo=repo)
    return {"schema_version": 1, "experiment_id": plan["experiment_id"], "executed": False, "authorization": plan["authorization"], "cells": cells, "total_budget_cap_usd": plan["max_total_budget_usd"]}


def _live_run_one(task, dtype, size, run_dir, *, config_path, worktree: Path, **kwargs):
    """Use the cell interpreter, so treatment and control import different source trees."""
    from tcenv.children import forget, kill_group, popen_group
    from types import SimpleNamespace

    request = {"task": task, "dtype": dtype, "size": size, "run_dir": str(run_dir), "config": config_path, "target_id": kwargs.get("target_id") or "apple-m4-max", "seed": kwargs["seed"], "overrides": {key: value for key, value in kwargs.items() if key in {"budget", "branches", "max_rounds"}}}
    request_path = Path(run_dir) / "cell_request.json"
    write_json_atomic(request_path, request)
    proc = popen_group([worktree / ".venv" / "bin" / "python" if (worktree / ".venv" / "bin" / "python").is_file() else __import__("sys").executable, "-m", "experiments.harness_cell", str(request_path)], cwd=worktree, env={**dict(__import__("os").environ), "PYTHONPATH": str(worktree)}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = proc.communicate()
    finally:
        kill_group(proc)
        forget(proc)
    result_path = Path(run_dir) / "cell_result.json"
    if proc.returncode or not result_path.is_file():
        raise ExperimentError("experiment cell failed: " + (stderr or stdout)[-2000:])
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return SimpleNamespace(spent_usd=payload["spent_usd"], stop_reason=payload["stop_reason"], run_seed=payload["run_seed"])


def execute_plan(plan: dict, *, out_dir: Path, repo: Path, allow_spend: bool = False, allow_gpu: bool = False, approved_by: str = "", runner=None) -> dict:
    """Run only an explicitly authorized plan. The default dry-run never reaches this function."""
    plan = read_plan(plan)
    if plan["status"] != "authorized" or not allow_spend or not allow_gpu or not approved_by.strip():
        raise ExperimentError("live execution needs an authorized plan, --allow-spend, --allow-gpu and --by")
    cells = build_cells(plan, out_dir=out_dir, repo=repo)
    if runner is None:
        runner = _live_run_one
    from .common import experiment_commitment, fits_cap
    previous_rows = []
    previous = out_dir / "execution.json"
    if previous.is_file():
        old = json.loads(previous.read_text(encoding="utf-8"))
        if old.get("experiment_id") == plan["experiment_id"]:
            previous_rows = [row for row in old.get("cells", []) if isinstance(row, dict)]
    records = [row for row in previous_rows if row.get("status") == "completed"]
    done = {row["cell"] for row in records}
    for cell in cells:
        if cell["cell"] in done:
            continue
        prior = [row for row in previous_rows if row.get("cell") == cell["cell"] and row.get("status") != "completed"]
        attempt = len(prior) + 1
        base = {**cell, "attempt": attempt}
        if prior:
            base["previous_attempts"] = prior
        commitment = experiment_commitment(out_dir)
        if not fits_cap(commitment, plan["budget_per_arm_usd"], plan["max_total_budget_usd"]):
            records.append({**base, "status": "over_cap"})
            continue
        raw_path = Path(cell["config"])
        if not Path(cell["run_dir"]).exists():
            Path(cell["run_dir"]).mkdir(parents=True)
        try:
            args = dict(cell["run_args"])
            task, dtype, size = args.pop("task"), args.pop("dtype"), args.pop("size")
            result = runner(task, dtype, size, Path(cell["run_dir"]), config_path=str(raw_path), log=lambda _message: None, worktree=Path(cell["worktree"]), **args)
            records.append({**base, "status": "completed", "spent_usd": result.spent_usd})
        except Exception as exc:  # noqa: BLE001 - keep the failed cell for inspection and resume
            records.append({**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json_atomic(out_dir / "execution.json", {"schema_version": 1, "experiment_id": plan["experiment_id"], "cells": records})
    grouped = {}
    for cell in records:
        grouped.setdefault(cell["arm"], []).append(cell)
    if any(cell["status"] != "completed" for cells in grouped.values() for cell in cells):
        return {"schema_version": 1, "experiment_id": plan["experiment_id"], "status": "inconclusive", "cells": records}
    result = {"schema_version": 1, "plan": plan}
    for arm, items in grouped.items():
        measurements = [collect_arm(plan, Path(cell["run_dir"]), arm) for cell in items]
        combined = {key: value for key, value in measurements[0].items() if key not in {"seeds", "repetitions"}}
        combined["seeds"] = plan["seeds"]
        combined["repetitions"] = plan["repetitions"]
        combined["spent_usd"] = sum(cell["spent_usd"] for cell in measurements)
        for key, value in measurements[0].items():
            if isinstance(value, dict) and "samples" in value:
                combined[key] = {**value, "samples": [sample for row in measurements for sample in row[key]["samples"]], "run_digests": [digest for row in measurements for digest in row[key]["run_digests"]]}
        result[arm] = combined
    from taic.harness_experiment import METRICS
    for metric in METRICS:
        if any(metric in result[arm] for arm in ("control", "treatment")) and not all(metric in result[arm] for arm in ("control", "treatment")):
            raise ExperimentError(f"paired metric {metric} is missing from one arm")
    summary = summarize_result(result)
    write_json_atomic(out_dir / "result.json", result)
    return {"schema_version": 1, "experiment_id": plan["experiment_id"], "status": summary["effect"]["status"], "cells": records, "result": result, "summary": summary}


def verify_cell_import(worktree: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(["python", "-c", "import pathlib, taic; assert pathlib.Path(taic.__file__).resolve().is_relative_to(pathlib.Path.cwd().resolve())"], cwd=worktree, env=env, capture_output=True, text=True)
    if proc.returncode:
        raise ExperimentError("the cell does not import its own worktree: " + (proc.stderr or proc.stdout)[-1000:])
