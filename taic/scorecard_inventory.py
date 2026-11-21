"""H0 read-only inventory for the future harness scorecard.

Only run roots are classified. Experiment containers, nested cells and recovery
runs remain explicit, and historical custom directories without experiment
metadata stay unknown rather than becoming natural by name.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .scorecard_contract import SCHEMA, SUFFICIENCY_THRESHOLDS, evidence_kind

ROOT_FILES = ("result.json", "run_config.json", "state.json", "transcript.jsonl", "ledger.json")
ROOT_DIRS = ("candidates",)
PREP_FILES = ("experience_overlay.json",)
PREP_DIRS = ("render-check",)
EXCLUDED_PARTS = frozenset({"agent_work", "sessions", "candidates", ".selfcheck", "__pycache__"})
GENERATED_RUN_NAME = re.compile(r"^[0-9]{8}-[0-9]{6}_")


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _excluded(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part in EXCLUDED_PARTS for part in parts)


def discover_run_roots(runs_root: Path) -> list[Path]:
    """@brief 发现 run root；工作目录和 candidate 内部不会向上制造新 run。"""
    root = Path(runs_root)
    found: set[Path] = set()
    for name in (*ROOT_FILES, *PREP_FILES):
        for path in root.rglob(name):
            if path.is_file() and not _excluded(path.parent, root):
                found.add(path.parent)
    for name in (*ROOT_DIRS, *PREP_DIRS):
        for path in root.rglob(name):
            if path.is_dir() and not _excluded(path.parent, root):
                found.add(path.parent)
    return sorted(found)


def _ancestor_documents(run_root: Path, runs_root: Path) -> tuple[tuple[Path, dict] | None, tuple[Path, dict] | None]:
    """@brief 找最近的显式实验 plan/approval；不读取 runs 之外的文件。"""
    plan = None
    approval = None
    current = run_root
    while current != runs_root and runs_root in current.parents:
        if plan is None and (current / "plan.json").is_file():
            plan = (current / "plan.json", _load(current / "plan.json"))
        if approval is None and (current / "approval.json").is_file():
            approval = (current / "approval.json", _load(current / "approval.json"))
        current = current.parent
    return plan, approval


def _experiment_identity(run_root: Path, runs_root: Path, result: dict, config: dict, state: dict) -> dict[str, Any]:
    """@brief 从显式实验记录确定 cohort、ID 和旧 plan 的分组指纹。"""
    plan, approval = _ancestor_documents(run_root, runs_root)
    plan_doc = plan[1] if plan else {}
    approval_doc = approval[1] if approval else {}
    binding = config.get("harness_experiment") if isinstance(config.get("harness_experiment"), dict) else {}
    search = config.get("search") if isinstance(config.get("search"), dict) else {}
    experiment_id = (
        binding.get("experiment_id") or search.get("harness_experiment_id")
        or result.get("harness_experiment_id") or approval_doc.get("experiment_id") or plan_doc.get("experiment_id")
    )
    fingerprint_source = plan[0] if plan else (approval[0] if approval else None)
    fingerprint = hashlib.sha256(fingerprint_source.read_bytes()).hexdigest() if fingerprint_source else None
    controlled = bool(binding or search.get("harness_experiment_id") or result.get("harness_experiment_id") or plan or approval)
    recovery_of = result.get("recovery_of") or config.get("recovery_of") or state.get("recovery_of")
    standard_natural = (
        run_root.parent.parent == runs_root
        and GENERATED_RUN_NAME.match(run_root.name) is not None
        and not controlled
    )
    cohort = "recovery" if recovery_of else ("controlled" if controlled else ("natural" if standard_natural else "unknown"))
    paid = bool(
        approval_doc.get("approved_by") or plan_doc.get("experiment_kind") == "paid"
        or plan_doc.get("kind") == "paid" or plan_doc.get("paid") is True or plan_doc.get("paid_search") is True
    )
    smoke = bool(
        plan_doc.get("experiment_kind") == "smoke" or plan_doc.get("kind") == "smoke"
        or plan_doc.get("smoke") is True or ".smoke" in str(plan_doc.get("schema_id") or "")
    )
    kind = "paid_smoke" if paid and smoke else ("paid" if paid else ("smoke" if smoke else "unknown"))
    container = plan[0].parent if plan else (approval[0].parent if approval else None)
    cell_id = _relative(run_root, container) if container is not None and run_root != container else None
    return {
        "cohort": cohort,
        "experiment_kind": kind if controlled else "unknown",
        "experiment_id": str(experiment_id) if experiment_id else None,
        "experiment_fingerprint": fingerprint,
        "cell_id": cell_id,
        "recovery_of": str(recovery_of) if recovery_of else None,
        "plan_path": _relative(plan[0], runs_root) if plan else None,
        "approval_path": _relative(approval[0], runs_root) if approval else None,
    }


def classify_run(run_root: Path, runs_root: Path) -> dict[str, Any]:
    """@brief 盘点一个 run root 的生命周期、cohort、身份字段和成本覆盖。"""
    run_root = Path(run_root)
    runs_root = Path(runs_root)
    result_path = run_root / "result.json"
    result = _load(result_path)
    config = _load(run_root / "run_config.json")
    state = _load(run_root / "state.json")
    execution = _load(run_root / "execution.json")
    observed = execution.get("observed") if isinstance(execution.get("observed"), dict) else {}
    has_result = result_path.is_file() and bool(result) and ("trace" in result or "task" in result)
    execution_evidence = any((run_root / name).exists() for name in ("result.json", "state.json", "transcript.jsonl", "candidates"))
    preparation = any((run_root / name).exists() for name in (*PREP_FILES, *PREP_DIRS))
    if has_result:
        lifecycle = "finalized"
    elif not result_path.exists() and execution_evidence:
        lifecycle = "incomplete_or_aborted"
    elif not execution_evidence and preparation:
        lifecycle = "dry_run"
    else:
        lifecycle = "unclassified"
    target = result.get("target_id") or state.get("target_id") or (config.get("target") or {}).get("id") or observed.get("target_id")
    backend = result.get("backend") or state.get("backend") or (config.get("target") or {}).get("backend") or observed.get("backend")
    protocol = result.get("protocol_version") or config.get("protocol_version") or observed.get("protocol_version")
    spent = result.get("spent_usd") if result.get("spent_usd") is not None else state.get("spent_usd")
    kind = evidence_kind(result, state, execution, observed)
    identity = _experiment_identity(run_root, runs_root, result, config, state)
    return {
        "run_root": _relative(run_root, runs_root),
        "lifecycle": lifecycle,
        **identity,
        "target_id": target,
        "backend": backend,
        "task": result.get("task") or state.get("task") or config.get("task"),
        "dtype": result.get("dtype") or state.get("dtype") or config.get("dtype"),
        "size": result.get("size") or state.get("size") or config.get("size"),
        "protocol_version": str(protocol) if protocol is not None else None,
        "system_prompt_sha1": config.get("system_prompt_sha1"),
        "evidence_kind": kind,
        "stop_reason": result.get("stop_reason") or state.get("stop_reason") or None,
        "confirmed_status": result.get("confirmed_status"),
        "best_label": result.get("best_label") or state.get("best_label"),
        "spent_usd": spent if isinstance(spent, (int, float)) and not isinstance(spent, bool) else None,
        "cost_may_be_incomplete": config.get("cost_may_be_incomplete"),
        "source_paths": {
            "result": _relative(result_path, runs_root) if result_path.exists() else None,
            "run_config": _relative(run_root / "run_config.json", runs_root) if (run_root / "run_config.json").exists() else None,
            "state": _relative(run_root / "state.json", runs_root) if (run_root / "state.json").exists() else None,
        },
    }


def inventory(runs_root: Path) -> dict[str, Any]:
    """@brief 生成 H0 只读 inventory；不生成 H1 四张计分表。"""
    runs_root = Path(runs_root)
    roots = discover_run_roots(runs_root)
    rows = [classify_run(path, runs_root) for path in roots]
    lifecycle_counts = Counter(row["lifecycle"] for row in rows)
    cohort_counts = Counter(row["cohort"] for row in rows)
    evidence_counts = Counter(str(row["evidence_kind"] or "unavailable") for row in rows)
    experiment_counts = Counter(row["experiment_kind"] for row in rows)
    # 这些 0 是完整 inventory 的实观测零，不是缺失数据补零。
    lifecycle = {name: lifecycle_counts.get(name, 0) for name in ("finalized", "incomplete_or_aborted", "dry_run", "unclassified")}
    cohort = {name: cohort_counts.get(name, 0) for name in ("natural", "controlled", "recovery", "unknown")}
    evidence = {name: evidence_counts.get(name, 0) for name in ("real", "simulated", "unavailable")}
    experiment = {name: experiment_counts.get(name, 0) for name in ("paid", "smoke", "paid_smoke", "unknown")}
    run_root_set = set(roots)
    containers = []
    container_paths = set()
    for row in rows:
        for key in ("plan_path", "approval_path"):
            value = row.get(key)
            if value:
                container_paths.add((runs_root / value).parent)
    for path in sorted(container_paths):
        if path in run_root_set:
            continue
        children = [row["run_root"] for row in rows if (runs_root / row["run_root"]).is_relative_to(path)]
        containers.append({"path": _relative(path, runs_root), "child_runs": len(children), "counted_as_run": False})
    field_coverage = {
        "protocol_version": sum(row["protocol_version"] is not None for row in rows),
        "system_prompt_sha1": sum(row["system_prompt_sha1"] is not None for row in rows),
        "spent_usd": sum(row["spent_usd"] is not None for row in rows),
        "stop_reason": sum(row["stop_reason"] is not None for row in rows),
        "confirmed_status": sum(row["confirmed_status"] is not None for row in rows),
    }
    return {
        "schema": SCHEMA,
        "runs_root": str(runs_root),
        "generated_at_unix": time.time(),
        "read_only": True,
        "model_calls": 0,
        "gpu_calls": 0,
        "h1_implemented": False,
        "run_count": len(rows),
        "counts": {
            "lifecycle": dict(lifecycle),
            "cohort": dict(cohort),
            "evidence_kind": dict(evidence),
            "experiment_kind": dict(experiment),
        },
        "field_coverage": field_coverage,
        "experiment_containers": containers,
        "runs": rows,
        "sufficiency_contract": dict(SUFFICIENCY_THRESHOLDS),
        "notes": [
            "finalized 只表示 result 已落盘，不表示搜索成功；stop_reason 与确认状态另列。",
            "没有 NPU 自然 run 时设备指标是 unavailable，不从模拟 fixture 补行。",
            "旧 plan 缺 experiment_id 时保留 null，experiment_fingerprint 只作分组，不是原实验 ID。",
            "本文件只是 H0 inventory，不是 H1 计分卡，也不是效果证据。",
        ],
        "publication_allowed": False,
    }


def render_summary(document: dict[str, Any]) -> str:
    counts = document["counts"]
    return "\n".join([
        "Harness scorecard H0 inventory (read-only; H1 not implemented)",
        f"  run roots: {document['run_count']}; lifecycle: {counts['lifecycle']}",
        f"  cohort: {counts['cohort']}",
        f"  experiment kind: {counts['experiment_kind']}",
        f"  evidence kind: {counts['evidence_kind']}",
        f"  experiment containers (not runs): {len(document['experiment_containers'])}",
        f"  field coverage: {document['field_coverage']}",
        "  publication_allowed=false; no GPU/model calls",
    ]) + "\n"


__all__ = ["classify_run", "discover_run_roots", "inventory", "render_summary"]
