"""Atomic JSON snapshots and an append-only GPU event log.

Every function swallows errors. A dashboard file must never fail a search or an evaluation.
"""

# 搜索和评测写给看板看的快照。页面不调用这些写函数，scan 读的是这里约定的缓存路径，以及实验脚本写下的 state.json。
# 写失败被吞掉，是为了看板文件的权限或磁盘问题不能把一次搜索判失败。改这里的写格式后，正在跑的搜索进程仍用旧函数，要等那个进程重启才生效；看板要换上新的读法，还得重启 python -m observe。已经落在磁盘上的 json，看板每次请求都会重读。

from __future__ import annotations

import json
import os
import time
from pathlib import Path


# 缓存默认在用户目录，不进仓库的 runs/。拷走实验结果时不会带上这台机器的 GPU 锁。
def cache_root() -> Path:
    env = os.environ.get("AI_LOWERLING_OBSERVE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "ai_lowerling"


# 每个进程一份锁文件。scan 用里面的 pid 判断谁还占着 GPU，进程退出后的文件不再算数。
def gpu_dir() -> Path:
    return cache_root() / "gpu.d"


# 评测心跳和 GPU 事件的缓存，跟某一次 run 目录分开。命令行评测没有 runs/ 子目录时，看板仍能看到它走到哪一步。
def observe_dir() -> Path:
    return cache_root() / "observe"


# 追加日志，给总览一条时间线。最后一行可能写不完，scan 解析失败时丢掉那一行。
def gpu_events_path() -> Path:
    return observe_dir() / "gpu_events.jsonl"


# 文件名带 pid。父进程退出后，scan 仍能认出这份心跳该不该继续显示。
def tcenv_live_path(pid: int | None = None) -> Path:
    return observe_dir() / f"tcenv-{pid or os.getpid()}.json"


# 先写同目录临时文件再替换。读者只会看见替换前或替换后的完整 JSON，scan 不会把这次快照当成截断。
def publish(path: Path, doc: dict) -> None:
    """Write ``doc`` as JSON via a temp file in the same directory, then replace. Never raises."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(doc, ensure_ascii=False, default=str)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(data, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        # 看板写失败对搜索不可见。调用方照常继续评测或规划。
        return


# 事件只能追加，不能整文件替换，所以崩溃时最后一行可能是半行。scan 会跳过解析失败的那一行。
def append_jsonl(path: Path, doc: dict) -> None:
    """Append one JSON line. Never raises."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(doc, ensure_ascii=False, default=str) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        return


# 长实验把「现在在跑谁、后面还排队哪些」写成带 script 的 state.json。scan 靠 script 把它和单次核运行分开。
# 不传 phase 就是 run；scan 只把 done / aborted 当成已经结束，其余要再看心跳和 pid。
def job_state(path: Path, *, script: str, current, queue, done, phase: str = "run") -> None:
    """Live status of a long experiment script."""
    publish(
        path,
        {
            "pid": os.getpid(),
            "updated_at": time.time(),
            "script": script,
            "phase": phase,
            "current": current,
            "queue": queue,
            "done": done,
        },
    )


# 等待和持有是两个文件。还在排队时不能覆盖持有者那一份。
def _gpu_file(pid: int, waiter: bool) -> Path:
    name = f"{pid}.wait.json" if waiter else f"{pid}.json"
    return gpu_dir() / name


# 删锁文件失败也要吞掉。看板上看不见这把锁，不能反过来让评测崩溃。
def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        return


# 写给看板看的锁状态，不是锁本身。give_up 只清等待文件，release 才同时清持有者；两种都会记一条事件。
def note_gpu(kind: str, mode: str, what: str) -> None:
    """Record a GPU-lock wait, acquire, release, or give-up. ``kind`` is one of those four words."""
    try:
        pid = os.getpid()
        now = time.time()
        if kind == "wait":
            publish(_gpu_file(pid, waiter=True), {"pid": pid, "role": "waiter", "mode": mode, "what": what, "since": now})
        elif kind == "acquire":
            _unlink(_gpu_file(pid, waiter=True))
            publish(_gpu_file(pid, waiter=False), {"pid": pid, "role": "holder", "mode": mode, "what": what, "since": now})
        elif kind in ("release", "give_up"):
            _unlink(_gpu_file(pid, waiter=True))
            if kind == "release":
                # give_up 是放弃等待，持有者文件留着。release 才是把锁交还。
                _unlink(_gpu_file(pid, waiter=False))
        else:
            # 不认识的 kind 不写文件，也不记事件。
            return
        append_jsonl(gpu_events_path(), {"t": now, "pid": pid, "kind": kind, "mode": mode, "what": what})
    except Exception:
        return
