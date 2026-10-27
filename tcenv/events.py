"""In-process events for progress and dashboard files. The standard library only.

With no subscriber, :func:`emit` does nothing. :func:`install_file_log` is the subscriber that writes the JSON files the dashboard reads. Search and evaluation call this module; they do not import the web package.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Callable

Listener = Callable[[str, dict], None]
_listeners: list[Listener] = []
_file_log_on = False

_progress_path: Path | None = None
_progress_doc: dict = {}


def subscribe(listener: Listener) -> None:
    """@brief 注册一个事件订阅者。

    Register ``listener``. A second registration of the same function is ignored.

    @param listener 接收 ``(kind, payload)`` 的可调用对象。
    """
    if listener not in _listeners:
        _listeners.append(listener)


def reset() -> None:
    """@brief 清空全部订阅者与内存中的进度文档。

    Drop every subscriber and the in-memory progress document. Tests use this.
    """
    global _file_log_on, _progress_path, _progress_doc
    _listeners.clear()
    _file_log_on = False
    _progress_path = None
    _progress_doc = {}


def emit(kind: str, **payload) -> None:
    """@brief 把事件发给所有订阅者。

    Tell subscribers. No subscriber means this returns without writing.

    @param kind 事件类型名。
    @param payload 事件负载，以关键字参数形式传入。
    """
    for listener in list(_listeners):
        try:
            listener(kind, payload)
        except Exception:
            continue


def install_file_log() -> None:
    """@brief 订阅内置的文件写入器。

    Subscribe the file writer. Calling it again does not add a second writer.
    """
    global _file_log_on
    if _file_log_on:
        return
    subscribe(_on_event)
    _file_log_on = True


def cache_root() -> Path:
    """@brief 观察数据所在的缓存根目录。

    Dashboard cache. ``AI_LOWERLING_OBSERVE_DIR`` overrides it. Not inside ``runs/``.

    @return 缓存目录路径。
    """
    env = os.environ.get("AI_LOWERLING_OBSERVE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "ai_lowerling"


def gpu_dir() -> Path:
    """@brief GPU 锁状态文件所在目录。

    @return 缓存根目录下的 ``gpu.d``。
    """
    return cache_root() / "gpu.d"


def observe_dir() -> Path:
    """@brief 观察快照所在目录。

    @return 缓存根目录下的 ``observe``。
    """
    return cache_root() / "observe"


def gpu_events_path() -> Path:
    """@brief GPU 锁事件日志的路径。

    @return ``observe/gpu_events.jsonl``。
    """
    return observe_dir() / "gpu_events.jsonl"


def tcenv_live_path(pid: int | None = None) -> Path:
    """@brief 某个进程的实时进度文件路径。

    @param pid 进程号；为 None 时使用当前进程。
    @return ``observe/tcenv-<pid>.json``。
    """
    return observe_dir() / f"tcenv-{pid or os.getpid()}.json"


def publish(path: Path, doc: dict) -> None:
    """@brief 请求文件写入器用 ``doc`` 整体替换 ``path``。

    Ask the file log to replace ``path`` with ``doc``. Never raises. A no-op until :func:`install_file_log`.

    @param path 目标文件路径。
    @param doc 要写入的文档。
    """
    try:
        install_file_log()
        emit("snapshot", path=path, doc=doc)
    except Exception:
        return


def append_jsonl(path: Path, doc: dict) -> None:
    """@brief 请求文件写入器追加一行 JSON。

    Ask the file log to append one JSON line. Never raises.

    @param path 目标 JSONL 文件路径。
    @param doc 要追加的文档。
    """
    try:
        install_file_log()
        emit("append", path=path, doc=doc)
    except Exception:
        return


def job_state(path: Path, *, script: str, current, queue, done, phase: str = "run") -> None:
    """@brief 发布一个长实验脚本的实时状态。

    Live status of a long experiment script.

    @param path 状态快照路径。
    @param script 脚本名。
    @param current 当前处理项。
    @param queue 待处理队列。
    @param done 已完成项。
    @param phase 当前阶段名。
    """
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


def note_gpu(kind: str, mode: str, what: str) -> None:
    """@brief 记录一次 GPU 锁的等待、获取、释放或放弃。

    Record a GPU-lock wait, acquire, release, or give-up. Never raises.

    @param kind 事件类型，如 wait / acquire / release / give_up。
    @param mode 锁模式，``"ex"`` 或 ``"sh"``。
    @param what 本次加锁用途的说明。
    """
    try:
        install_file_log()
        emit("gpu", action=kind, mode=mode, what=what)
    except Exception:
        return


def bind_progress(path: str | Path | None) -> None:
    """@brief 把后续进度更新绑定到一个阶段文件。

    Send later :func:`update_progress` calls to this file. ``None`` turns them off.

    @param path 阶段文件路径；None 表示关闭。
    """
    try:
        install_file_log()
        emit("progress-bind", path=None if path is None else str(path))
    except Exception:
        return


def update_progress(**fields) -> None:
    """@brief 把字段合并进已绑定的阶段文件。

    Merge ``fields`` into the stage file. Never raises. A new ``stage`` records ``entered_at``.

    @param fields 要合并的进度字段。
    """
    try:
        install_file_log()
        emit("progress", fields=fields)
    except Exception:
        return


def _note_index(path: Path) -> None:
    """@brief 文件日志真正落盘后，给索引侧写一行旁路日志。

    写在写入器线程里而不是 :func:`publish` 调用处：发布是排队的，调用点记日志会让索引在文件还没落地时
    就去 stat，把旧内容配上新日志。

    @param path 刚写入的文件路径。
    """
    try:
        from .index_journal import note

        note(path)
    except Exception:  # noqa: BLE001 - 旁路日志不能影响状态发布
        return


def _on_event(kind: str, payload: dict) -> None:
    """@brief 内置文件写入器的分发入口。

    @param kind 事件类型名。
    @param payload 事件负载。
    """
    if kind == "snapshot":
        _write_json(payload["path"], payload["doc"])
    elif kind == "append":
        _append_jsonl(payload["path"], payload["doc"])
    elif kind == "gpu":
        _record_gpu(payload["action"], payload["mode"], payload["what"])
    elif kind == "progress-bind":
        _bind_progress(payload.get("path"))
    elif kind == "progress":
        _update_progress(**payload.get("fields") or {})


def _write_json(path: Path, doc: dict) -> None:
    """@brief 原子地写入一个 JSON 文件。

    @param path 目标文件路径。
    @param doc 要写入的文档。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(doc, ensure_ascii=False, default=str)
    # Evaluator progress can be published concurrently by worker threads.
    # A PID-only temporary name lets those writers replace each other's file.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)
    _note_index(path)


def _append_jsonl(path: Path, doc: dict) -> None:
    """@brief 向 JSONL 文件追加一行。

    @param path 目标文件路径。
    @param doc 要追加的文档。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(doc, ensure_ascii=False, default=str) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
    _note_index(path)


def _gpu_file(pid: int, waiter: bool) -> Path:
    """@brief 某个进程的 GPU 锁状态文件路径。

    @param pid 进程号。
    @param waiter True 表示等待者文件，False 表示持有者文件。
    @return ``gpu.d/<pid>[.wait].json``。
    """
    name = f"{pid}.wait.json" if waiter else f"{pid}.json"
    return gpu_dir() / name


def _unlink(path: Path) -> None:
    """@brief 删除文件，忽略不存在与其他删除错误。

    @param path 要删除的路径。
    """
    try:
        path.unlink(missing_ok=True)
    except Exception:
        return


def _env_int(name: str, default: int) -> int:
    """@brief 读一个非负整数环境变量；非法时用默认值。

    @param name 环境变量名。
    @param default 默认值。
    @return 解析后的整数。
    """
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value >= 0 else default


def _rotate_jsonl(path: Path, *, limit_env: str, default_limit: int) -> None:
    """@brief 超过字节上限时把 JSONL 改名保留成一个分片。

    轮转是“改名保留”，**不删除**旧内容：索引在 inode 变化后按分片重放，删库后也必须能从保留的
    分片重建同样的历史。清理旧分片只能由运维显式做，索引会把缺失的分片标记为历史截断。

    @param path 正在追加的 JSONL 文件。
    @param limit_env 上限的字节数环境变量名。
    @param default_limit 默认字节上限。
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        return
    limit = _env_int(limit_env, default_limit)
    if limit <= 0 or size < limit:
        return
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    target = path.with_name(f"{path.name}.{stamp}")
    suffix = 1
    while target.exists():
        target = path.with_name(f"{path.name}.{stamp}.{suffix}")
        suffix += 1
    try:
        path.rename(target)
    except OSError:
        return


def _record_gpu(kind: str, mode: str, what: str) -> None:
    """@brief 更新 GPU 锁状态快照并追加一条事件。

    @param kind 事件类型，如 wait / acquire / release / give_up。
    @param mode 锁模式，``"ex"`` 或 ``"sh"``。
    @param what 本次加锁用途的说明。
    """
    pid = os.getpid()
    now = time.time()
    if kind == "wait":
        _write_json(_gpu_file(pid, waiter=True), {"pid": pid, "role": "waiter", "mode": mode, "what": what, "since": now})
    elif kind == "acquire":
        _unlink(_gpu_file(pid, waiter=True))
        _write_json(_gpu_file(pid, waiter=False), {"pid": pid, "role": "holder", "mode": mode, "what": what, "since": now})
    elif kind in ("release", "give_up"):
        _unlink(_gpu_file(pid, waiter=True))
        if kind == "release":
            _unlink(_gpu_file(pid, waiter=False))
    else:
        return
    events_path = gpu_events_path()
    _rotate_jsonl(events_path, limit_env="TCENV_GPU_EVENTS_MAX_BYTES", default_limit=8 * 1024 * 1024)
    _append_jsonl(events_path, {"t": now, "pid": pid, "kind": kind, "mode": mode, "what": what})


def _bind_progress(path: str | None) -> None:
    """@brief 切换阶段文件并清空内存中的进度文档。

    @param path 新的阶段文件路径；None 表示关闭。
    """
    global _progress_path, _progress_doc
    _progress_path = Path(path) if path else None
    _progress_doc = {}


def _update_progress(**fields) -> None:
    """@brief 合并进度字段并重写阶段文件。

    @param fields 进度字段；新 ``stage`` 会附带 ``entered_at``。
    """
    global _progress_doc
    if _progress_path is None:
        return
    now = time.time()
    if "stage" in fields and fields["stage"] != _progress_doc.get("stage"):
        fields["entered_at"] = now
    _progress_doc.update(fields)
    _progress_doc["updated_at"] = now
    _write_json(_progress_path, _progress_doc)
