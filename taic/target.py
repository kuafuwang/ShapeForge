"""Skill and prompt files for one target. Builds on ``tcenv.target``."""

from __future__ import annotations

from pathlib import Path

from tcenv.target import DEFAULT_TARGET_ID, TargetProfile, load_target

def profile(target_id: str | None = None) -> TargetProfile:
    """@brief 取一个目标档。

    @param target_id 目标标识；``None`` 时用 ``tcenv.target`` 的默认目标。
    @return 解析后的 ``TargetProfile``。
    """
    return load_target(target_id or DEFAULT_TARGET_ID)


def skill_path(name: str, target_id: str | None = None) -> Path:
    """@brief 定位一个技能/提示文件的路径。

    ``experience`` 与 ``device`` 挂在目标档上；其它名字当作后端技能。

    @param name 技能名，如 ``experience``、``wiki/experience``、``device`` 或后端技能名。
    @param target_id 目标标识；``None`` 时用默认目标。
    @return 该文件的路径（不保证存在）。
    """
    current = profile(target_id)
    if name in ("experience", "wiki/experience"):
        return current.experience
    if name in ("device", "device.md"):
        return current.device_doc
    return current.skills / f"{name}.md"


def skill_text(name: str, target_id: str | None = None) -> str:
    """@brief 读一个技能/提示文件的全文。

    @param name 技能名，语义同 ``skill_path``。
    @param target_id 目标标识；``None`` 时用默认目标。
    @return 文件内容。
    """
    return skill_path(name, target_id).read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """@brief ``python -m taic.target migrate-runs``：把平铺的运行目录搬进 ``runs/<target>/``。

    @param argv 命令行参数；缺省从 ``sys.argv`` 取。
    @return 进程退出码，成功为 0。
    """
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
