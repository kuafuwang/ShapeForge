"""Read-only hints mined from finished candidate pairs. Nothing here calls a model or launches a kernel."""

from __future__ import annotations

import json
from pathlib import Path

from .hwfeatures import analyze_source
from .policy import feature_signature

_FEATURES = ("matrix_unit", "acc_bucket", "tg_staging", "double_buffer", "register_prefetch", "vector_load", "online_softmax")


def _present(source: str, backend_id: str = "metal") -> set[str]:
    sig = feature_signature(source, backend_id=backend_id)
    return {name for name, bit in zip(_FEATURES, sig) if bit and bit != "0"}


def _bound(report: dict) -> str | None:
    roof = report.get("roofline") if isinstance(report, dict) else None
    bound = roof.get("bound") if isinstance(roof, dict) else None
    return bound if bound in ("memory", "compute") else None


def _target_info(run: Path) -> tuple[str, str]:
    """The run's backend and kernel extension, legacy Metal fallback."""
    try:
        doc = json.loads((run / "run_config.json").read_text())
        target = doc.get("target") if isinstance(doc.get("target"), dict) else {}
        target_id = str(target.get("id") or doc.get("target_id") or "")
        from tcenv.backends.registry import load_backend
        from tcenv.target import load_target

        profile = load_target(target_id or None)
        backend = load_backend(profile.backend)
        ext = backend.file_ext or ".metal"
        if not str(ext).startswith("."):
            ext = "." + str(ext)
        return backend.id, ext
    except Exception:
        return "metal", ".metal"


def build_experience_index(roots: list[Path], *, exclude_runs: set[str] | None = None) -> dict:
    """Pairs where a later ok kernel beat the previous best, and which static features it added.

    ``exclude_runs`` drops a run whose directory name is in the set, so a repeatability trial can leave itself out.
    """
    blocked = exclude_runs or set()
    wins: list[dict] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        runs = [root] if (root / "candidates").is_dir() else [p for p in root.rglob("candidates") if p.is_dir()]
        seen_dirs = {p.parent if p.name == "candidates" else p for p in runs}
        for run in sorted(seen_dirs):
            if run.name in blocked or any(part in blocked for part in run.parts):
                continue
            backend_id, ext = _target_info(run)
            best_source = None
            best_speed = None
            folders = sorted((run / "candidates").iterdir()) if (run / "candidates").is_dir() else []
            for folder in folders:
                report_path = folder / "report.json"
                kernel_path = folder / f"kernel{ext}"
                if not kernel_path.is_file():
                    kernel_path = next(iter(sorted(folder.glob("kernel.*"))), folder / "kernel.metal")
                if not report_path.is_file() or not kernel_path.is_file():
                    continue
                try:
                    report = json.loads(report_path.read_text())
                except ValueError:
                    continue
                if not isinstance(report, dict) or report.get("status") != "ok":
                    continue
                speed = report.get("speedup")
                if not isinstance(speed, (int, float)) or isinstance(speed, bool):
                    continue
                source = kernel_path.read_text()
                if best_source is not None and best_speed is not None and speed > best_speed * 1.005:
                    added = sorted(_present(source, backend_id) - _present(best_source, backend_id))
                    wins.append({"task": report.get("task"), "bound": _bound(report), "added": added, "ratio": speed / best_speed})
                if best_speed is None or speed > best_speed:
                    best_source, best_speed = source, float(speed)
    return {"entries": _aggregate(wins), "wins": len(wins), "pairs": wins}


def load_index(path: Path) -> dict:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {"entries": []}
    return doc if isinstance(doc, dict) else {"entries": []}


def _task_base(name: object) -> str:
    if not isinstance(name, str) or not name:
        return ""
    return name.split("@", 1)[0]


def _aggregate(wins: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[float]] = {}
    for row in wins:
        for feature in row.get("added") or []:
            grouped.setdefault((row.get("bound"), feature), []).append(float(row["ratio"]))
    entries = []
    for (bound, feature), ratios in sorted(grouped.items(), key=lambda item: (-len(item[1]), str(item[0]))):
        ordered = sorted(ratios)
        entries.append({"bound": bound, "feature": feature, "n": len(ratios), "median_ratio": ordered[len(ordered) // 2]})
    return entries


def retrieve_experience_hints(index: dict, task_name: str, bound: str | None, *, exclude_task: bool = False, k: int = 3) -> str:
    """Up to ``k`` lines. ``exclude_task`` drops every pair whose task (ignoring ``@variant``) is ``task_name``.

    An index that has no ``pairs`` list cannot prove a hint came from another task, so the call returns no hints
    instead of the mixed ``entries``. Repeatability passes an empty index path and never reaches this function.
    """
    if exclude_task:
        pairs = index.get("pairs")
        if not isinstance(pairs, list):
            return ""
        base = _task_base(task_name)
        kept = [row for row in pairs if isinstance(row, dict) and _task_base(row.get("task")) != base]
        rows = _aggregate(kept)
    else:
        rows = [row for row in (index.get("entries") or []) if isinstance(row, dict) and row.get("n")]
    if bound in ("memory", "compute"):
        scoped = [row for row in rows if row.get("bound") in (bound, None)]
        if scoped:
            rows = scoped
    lines = []
    for row in rows[:k]:
        lines.append(f"- {row.get('feature')} showed up on a faster kernel in {row.get('n')} pair(s), median {float(row.get('median_ratio') or 1):.3f}x (static, not a requirement)")
    if not lines:
        return ""
    return "Experience from other measured pairs (hints, the timing decides):\n" + "\n".join(lines)


def write_index(path: Path, index: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=1))


# analyze_source is imported so a caller can see that mining is source-level. Re-export keeps the dependency obvious.
__all__ = ["analyze_source", "build_experience_index", "load_index", "retrieve_experience_hints", "write_index"]
