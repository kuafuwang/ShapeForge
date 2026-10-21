"""Skill and prompt files for one target. Builds on ``tcenv.target``."""

from __future__ import annotations

from pathlib import Path

from tcenv.target import DEFAULT_TARGET_ID, TargetProfile, load_target

def profile(target_id: str | None = None) -> TargetProfile:
    return load_target(target_id or DEFAULT_TARGET_ID)


def skill_path(name: str, target_id: str | None = None) -> Path:
    """``experience`` and ``device`` live on the target. Other names are backend skills."""
    current = profile(target_id)
    if name in ("experience", "wiki/experience"):
        return current.experience
    if name in ("device", "device.md"):
        return current.device_doc
    return current.skills / f"{name}.md"


def skill_text(name: str, target_id: str | None = None) -> str:
    return skill_path(name, target_id).read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """``python -m taic.target migrate-runs`` moves flat run directories under ``runs/<target>/``."""
    import argparse
    import json
    from pathlib import Path

    from .migrate import migrate_runs

    parser = argparse.ArgumentParser(prog="python -m taic.target")
    sub = parser.add_subparsers(dest="cmd", required=True)
    move = sub.add_parser("migrate-runs")
    move.add_argument("--target", default="apple-m4-max")
    move.add_argument("--root", default=None)
    args = parser.parse_args(argv)
    root = None if args.root is None else Path(args.root)
    moved = migrate_runs(root, args.target)
    print(json.dumps({"target": args.target, "moved": moved}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
