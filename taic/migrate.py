"""Move flat run directories into runs/<target_id>/."""

from __future__ import annotations

from pathlib import Path

from tcenv.target import REPO

_MARKERS = ("state.json", "result.json", "run_config.json", "candidates", "transcript.jsonl")


def migrate_runs(root: Path | None = None, target_id: str = "apple-m4-max") -> list[str]:
    """Return the directories that moved. A directory that is already nested, or not a run, stays put."""
    base = Path(root) if root is not None else REPO / "runs"
    dest_root = base / target_id
    dest_root.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    if not base.is_dir():
        return moved
    for child in sorted(base.iterdir()):
        if not child.is_dir() or child.name in {target_id, "_template"} or child.name.startswith("."):
            continue
        if not any((child / name).exists() for name in _MARKERS):
            continue
        dest = dest_root / child.name
        if dest.exists():
            continue
        child.rename(dest)
        _stamp(dest, target_id)
        moved.append(child.name)
    return moved


def _stamp(run_dir: Path, target_id: str) -> None:
    """Write the target onto files that were recorded before profiles existed."""
    import json

    for name in ("run_config.json", "state.json", "result.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict) or doc.get("target_id"):
            continue
        doc["target_id"] = target_id
        doc["backend"] = doc.get("backend") or "metal"
        if name == "run_config.json" and not isinstance(doc.get("target"), dict):
            doc["target"] = {"id": target_id, "backend": "metal"}
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
