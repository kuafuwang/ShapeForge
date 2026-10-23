"""Move flat run directories into runs/<target_id>/."""

from __future__ import annotations

from pathlib import Path

from tcenv.target import REPO

_MARKERS = ("state.json", "result.json", "run_config.json", "candidates", "transcript.jsonl")


def migrate_runs(root: Path | None = None, target_id: str = "apple-m4-max") -> list[str]:
    """@brief 把平铺的运行目录搬进 ``runs/<target_id>/``。

    已经是嵌套结构、看起来不是运行目录、或仍在写入的目录都留在原处。

    @param root 待整理目录的根；缺省为仓库下的 ``runs``。
    @param target_id 目标档名，同时是目标子目录名。
    @return 实际搬动的目录名列表；没有任何目录可搬时为空列表。
    """
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
        if _live(child):
            # 正在跑的 run 用的是绝对路径；一移动就会把同一场搜索拆成两个目录。
            continue
        dest = dest_root / child.name
        if dest.exists():
            continue
        child.rename(dest)
        _stamp(dest, target_id)
        moved.append(child.name)
    return moved


def _live(run_dir: Path) -> bool:
    """@brief 判断一次运行是否还有活进程占着。

    判据是：还没有 ``result.json``，且 ``state.json`` 里的 pid 仍然活着。

    @param run_dir 运行目录。
    @return 仍在跑时为 ``True``；已有结果或 pid 已死时为 ``False``。
    """
    from . import control

    if (run_dir / "result.json").is_file():
        return False
    state = control.read_json(run_dir / "state.json")
    return isinstance(state, dict) and bool(control.pid_alive(state.get("pid")))


def _stamp(run_dir: Path, target_id: str) -> None:
    """@brief 把 ``target_id``/``backend`` 补写进建档于 profile 机制之前的文件。

    已带 ``target_id`` 的文件不动；三个文件各自原子替换。

    @param run_dir 运行目录。
    @param target_id 要写入的目标档名。
    """
    import json

    from . import control

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
        control.write_json_atomic(path, doc)
