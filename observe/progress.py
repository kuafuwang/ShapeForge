"""Stage file written by an evaluation worker. The parent copies it next to the heartbeat."""

# 评测子进程写下的阶段。看板不导入这个模块，它读的是父进程抄到运行目录旁的 eval_progress.json（scan 里的 eval_progress）。
# 改这里要等下一次评测进程重新导入。已经开着的看板每次请求都重读磁盘，所以新文件内容不用重启看板；看板自己的 Python 或 HTML 改了，仍要重启 python -m observe。

from __future__ import annotations

import time
from pathlib import Path

from observe.publish import publish

_path: Path | None = None
_doc: dict = {}


# 子进程是新的，路径从请求里来。update 不再收路径，所以先 bind。传入 None 表示这次评测不往看板写。
def bind(path: str | Path | None) -> None:
    """Send later :func:`update` calls to this file. ``None`` turns publishing off."""
    global _path, _doc
    _path = Path(path) if path else None
    # 换路径时清掉内存里的上一份，避免把别的核的阶段字段写进新文件。
    _doc = {}


# 阶段名没变时不刷新 entered_at，页面才能看出这一阶段已经走了多久。每个用例都重置的话，计时就失真了。
def update(**fields) -> None:
    """Merge ``fields`` into the stage file. Never raises. A new ``stage`` records ``entered_at``."""
    global _doc
    if _path is None:
        # 没 bind，或显式关掉。评测照常进行，看板只是看不到阶段。
        return
    try:
        now = time.time()
        if "stage" in fields and fields["stage"] != _doc.get("stage"):
            # 同一阶段里更新用例计数时，entered_at 保持进入这一阶段的时刻。
            fields["entered_at"] = now
        _doc.update(fields)
        _doc["updated_at"] = now
        publish(_path, _doc)
    except Exception:
        return
