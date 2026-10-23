"""Retention for ``data/deleted``. The default is a report; ``--apply`` is the only deleting mode.

The dashboard deletion path moves a run or experiment under ``data/deleted``. Nothing has ever
removed those directories. This module adds a TTL policy without changing that default: a caller
asks for a cutoff, sees what would go, and only passes ``--apply`` deliberately.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path


def _size(path: Path) -> tuple[int, int]:
    """@brief 统计一个目录下的文件数与总字节数。

    单个文件取不到大小（例如已被删除）时按 0 计，不影响其余统计。

    @param path 要统计的目录。
    @return ``(文件数, 总字节数)``。
    """
    files = 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            files += 1
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return files, total


def scan(root: Path, *, older_than_days: float, now: float | None = None) -> list[dict]:
    """@brief 为每个子目录生成一行记录，``eligible`` 表示年龄已达到截止线。

    只读扫描：不删除任何东西。

    @param root ``data/deleted`` 这类待清理目录的根。
    @param older_than_days 保留天数阈值；目录 mtime 的年龄 >= 该值才算可清理。
    @param now 当前时间戳；缺省取 ``time.time()``，测试可注入。
    @return 每行含 ``path``/``age_days``/``files``/``bytes``/``eligible`` 的列表；根目录不存在时为空列表。
    """
    root = Path(root)
    if not root.is_dir():
        return []
    moment = time.time() if now is None else now
    rows = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.is_symlink():
            # A symlink is not archive content; following it would delete the target.
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError:
            continue
        files, total = _size(child)
        age = (moment - mtime) / 86400.0
        rows.append({
            "path": str(child), "age_days": round(age, 3), "files": files, "bytes": total,
            "eligible": age >= older_than_days,
        })
    return rows


def apply(rows: list[dict], *, log_path: Path | None = None, now: float | None = None, root: Path | None = None) -> dict:
    """@brief 删除 ``eligible`` 为真的行，并在每次删除前追加一条审计记录。

    删除失败不抛错（``shutil.rmtree(..., ignore_errors=True)``）；日志中记录的是扫描时的文件数与字节数。

    @param rows ``scan`` 返回的行列表；只处理 ``eligible`` 为真的行。
    @param log_path 审计日志路径；为 ``None`` 时不写日志。
    @param now 当前时间戳，写入审计行；缺省取 ``time.time()``。
    @return 含 ``removed``（已删除路径）与 ``bytes``（这些行合计字节数）的字典。
    """
    moment = time.time() if now is None else now
    removed = []
    removed_bytes = 0
    log = Path(log_path) if log_path else None
    archive = Path(root).resolve() if root is not None else None
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if not row.get("eligible"):
            continue
        path = Path(row["path"])
        if path.is_symlink():
            continue
        if archive is not None:
            try:
                path.resolve().relative_to(archive)
            except ValueError:
                # A row outside the archive root is not ours to delete.
                continue
        if log is not None:
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "action": "gc", "path": str(path), "files": row.get("files"), "bytes": row.get("bytes"),
                    "age_days": row.get("age_days"), "at": moment,
                }, ensure_ascii=False) + "\n")
        shutil.rmtree(path, ignore_errors=True)
        if path.exists():
            continue
        removed.append(str(path))
        removed_bytes += int(row.get("bytes") or 0)
    return {"removed": removed, "bytes": removed_bytes}
