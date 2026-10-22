"""Experiment identity written before the first child search.

The file is ``experiment.json`` in the experiment root. Frozen conditions and each arm's switch delta are immutable.
A later invocation may append child directories, add an arm that was not declared yet, or append a budget extension.
Changing the frozen conditions, or an arm's delta, is a new experiment and is refused.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from taic.control import write_json_atomic

FILENAME = "experiment.json"
VERSION = 1
KINDS = ("suite", "repeatability", "search_ab", "contract_menu", "committee_ab", "single")


class ManifestError(ValueError):
    """The directory already belongs to a different experiment."""


def canonical(doc: dict) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(frozen: dict) -> str:
    """SHA-256 of the canonical frozen conditions. Arms are not part of the hash."""
    return hashlib.sha256(canonical(frozen).encode("utf-8")).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def experiment_id_for(root: Path) -> str:
    """Path of ``root`` relative to the runs directory, or the directory name when it lives elsewhere."""
    root = Path(root).resolve()
    bases = []
    env = os.environ.get("AI_LOWERLING_RUNS_DIR")
    if env:
        bases.append(Path(env))
    bases.append(Path(__file__).resolve().parent.parent / "runs")
    for base in bases:
        try:
            return root.relative_to(base.resolve()).as_posix() or "."
        except ValueError:
            continue
    return root.name


@contextmanager
def _exclusive(root: Path):
    """One writer for ``experiment.json``. A top-up and an attach must not replace each other's document."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".experiment.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def load(root: Path) -> dict | None:
    path = Path(root) / FILENAME
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema_version") != VERSION:
        raise ManifestError(f"{path} is not an experiment manifest version {VERSION}")
    return doc


def _arm(arm_id: str, delta: dict, planned_n: int) -> dict:
    return {
        "arm_id": str(arm_id),
        "delta": delta if isinstance(delta, dict) else {},
        "planned_n": int(planned_n),
        "runs": [],
    }


def _same_delta(left: dict, right: dict) -> bool:
    return canonical(left or {}) == canonical(right or {})


def ensure(
    root: Path,
    *,
    kind: str,
    driver: str,
    frozen: dict,
    arms: list[dict],
) -> dict:
    """Create the manifest, or accept this call when the frozen conditions and existing deltas match.

    ``arms`` items are ``{"arm_id", "delta", "planned_n"}``. A new arm may be added. An existing arm's
    delta may not change. ``planned_n`` may rise. The file is written before this function returns,
    so a caller can start a child search only after ``ensure`` succeeds.
    """
    if kind not in KINDS:
        raise ManifestError(f"unknown experiment kind {kind!r}")
    root = Path(root)
    with _exclusive(root):
        return _ensure_locked(root, kind=kind, driver=driver, frozen=frozen, arms=arms)


def _ensure_locked(root: Path, *, kind: str, driver: str, frozen: dict, arms: list[dict]) -> dict:
    existing = load(root) if (root / FILENAME).is_file() else None
    if existing is None:
        doc = {
            "schema_version": VERSION,
            "experiment_id": experiment_id_for(root),
            "kind": kind,
            "driver": driver,
            "created_at": time.time(),
            "git_commit": _git_commit(),
            "frozen": frozen,
            "config_hash": config_hash(frozen),
            "arms": [_arm(item["arm_id"], item.get("delta") or {}, item.get("planned_n") or 0) for item in arms],
            "budget_extensions": [],
        }
        write_json_atomic(root / FILENAME, doc)
        return doc
    if existing.get("kind") != kind or existing.get("driver") != driver:
        raise ManifestError(f"{root} is already a {existing.get('driver')} experiment")
    if existing.get("config_hash") != config_hash(frozen) or canonical(existing.get("frozen") or {}) != canonical(frozen):
        raise ManifestError(f"{root} already records different frozen conditions; start a new experiment directory")
    by_id = {arm["arm_id"]: arm for arm in existing.get("arms") or []}
    changed = False
    for item in arms:
        arm_id = str(item["arm_id"])
        delta = item.get("delta") or {}
        planned = int(item.get("planned_n") or 0)
        current = by_id.get(arm_id)
        if current is None:
            current = _arm(arm_id, delta, planned)
            existing.setdefault("arms", []).append(current)
            by_id[arm_id] = current
            changed = True
            continue
        if not _same_delta(current.get("delta") or {}, delta):
            raise ManifestError(f"arm {arm_id} already has a different switch delta; start a new experiment directory")
        if planned > int(current.get("planned_n") or 0):
            current["planned_n"] = planned
            changed = True
    if changed:
        write_json_atomic(root / FILENAME, existing)
    return existing


def attach(root: Path, arm_id: str, relative: str) -> dict:
    """Record one child directory on an existing arm. Does not change frozen conditions or the arm delta."""
    root = Path(root)
    rel = Path(relative).as_posix()
    if rel.startswith("/") or ".." in Path(rel).parts:
        raise ManifestError(f"child path must stay inside the experiment: {relative}")
    with _exclusive(root):
        doc = load(root)
        if doc is None:
            raise ManifestError(f"{root} has no experiment manifest")
        for arm in doc.get("arms") or []:
            if arm.get("arm_id") != arm_id:
                continue
            runs = arm.setdefault("runs", [])
            changed = False
            if rel not in runs:
                runs.append(rel)
                changed = True
            if len(runs) > int(arm.get("planned_n") or 0):
                arm["planned_n"] = len(runs)
                changed = True
            if changed:
                write_json_atomic(root / FILENAME, doc)
            return doc
    raise ManifestError(f"arm {arm_id} is not in {root / FILENAME}")


def extend_budget(root: Path, *, run: str, new_cap: float, at: float | None = None) -> dict:
    """Append a budget extension. The frozen conditions and the arm deltas stay as they were."""
    root = Path(root)
    with _exclusive(root):
        doc = load(root)
        if doc is None:
            raise ManifestError(f"{root} has no experiment manifest")
        doc.setdefault("budget_extensions", []).append({
            "run": Path(run).as_posix(),
            "new_cap": float(new_cap),
            "at": time.time() if at is None else float(at),
        })
        write_json_atomic(root / FILENAME, doc)
        return doc


def backfill(root: Path, *, kind: str, driver: str, frozen: dict | None = None) -> dict:
    """Create a manifest for an experiment directory that predates manifests.

    Every child directory that holds a ``run_config.json`` (or a ``state.json``) becomes an
    arm and is attached under its relative path. Budget is deliberately not part of the arm
    delta: it changed between reruns, and a delta that embeds it makes a later ``ensure``
    refuse the directory. The frozen block is marked ``backfilled`` so a reader knows the
    conditions were reconstructed, not recorded at launch.
    """
    root = Path(root)
    existing = load(root)
    if existing is not None:
        return existing
    seen: set[str] = set()
    arms: list[dict] = []
    for name in ("run_config.json", "state.json"):
        for path in sorted(root.rglob(name)):
            child = path.parent
            if child == root:
                continue
            relative = child.relative_to(root).as_posix()
            if relative in seen:
                continue
            seen.add(relative)
            config = _read_json(child / "run_config.json")
            state = _read_json(child / "state.json")
            task = config.get("task") or state.get("task")
            if not task:
                continue
            delta = {"task": task}
            for key in ("dtype", "size", "model"):
                value = config.get(key) or state.get(key)
                if value:
                    delta[key] = value
            arms.append({"arm_id": relative, "delta": delta, "planned_n": 1})
    if not arms:
        raise ManifestError(f"{root} has no child run directories to backfill")
    doc = ensure(root, kind=kind, driver=driver, frozen=frozen or {"backfilled": True}, arms=arms)
    for arm in arms:
        attach(root, arm["arm_id"], arm["arm_id"])
    return load(root) or doc


def _read_json(path: Path) -> dict:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m experiments.manifest")
    parser.add_argument("--backfill", metavar="ROOT", required=True)
    parser.add_argument("--kind", default="suite", choices=sorted(KINDS))
    parser.add_argument("--driver", required=True)
    args = parser.parse_args(argv)
    doc = backfill(Path(args.backfill), kind=args.kind, driver=args.driver)
    print(json.dumps({"experiment_id": doc.get("experiment_id"), "arms": len(doc.get("arms") or []), "root": args.backfill}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
