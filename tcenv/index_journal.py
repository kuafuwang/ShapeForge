"""索引侧的写者日志：每个写者进程一个分片，追加一行“哪个 run 的哪个文件变了”。

索引器不能用目录 mtime 发现已结束运行的就地改写（``state.json``/``confirm_report.json``/候选报告
都不改运行目录 mtime），而每次全量 stat 一万个 run 又太贵。这里让**源写者**在落盘成功后追加一行：

```json
{"run": "conv2d/20261025-...", "path": "state.json", "at": 1762000000.0}
```

纪律（见 plan §11.1b）：

* 写入永不抛错；失败只留一行日志。搜索不能因为索引的旁路日志失败而停下。
* 不 fsync。崩掉丢的一行由索引器的周期对账兜底。
* 每个写者一个分片 ``<runs>/.index_journal/<pid>.jsonl``；并发写多个进程时不做共享 rename。
* 分片到 4MB 就原子 rename 成 ``.1``；索引器发现 inode 变化或 ``size < offset`` 就从头重读并强制一次全量对账。
* 只记会影响索引值的文件名白名单；``transcript.jsonl``/``llm.json``/``eval_progress.json`` 不记。

模块放在 ``tcenv`` 里：搜索侧（``tcenv.events``）与看板侧（``taic.control``）都要写，不能反向 import。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

LOG = logging.getLogger(__name__)

DIR_NAME = ".index_journal"
SHARD_LIMIT = 4 * 1024 * 1024

# 顶层文件：落在运行目录（或实验根）里，``run`` 记它所在目录。
TOP_FILES = frozenset({
    "state.json",
    "result.json",
    "confirm_report.json",
    "run_config.json",
    "feedback.jsonl",
    "feedback_review.json",
    "budget_events.json",
    "ledger.json",
    "resume.lock",
    "experiment.json",
})
# 候选目录里的两个文件：``candidates/<dir>/{report.json,kernel.metal}``，``run`` 记运行目录。
CANDIDATE_FILES = frozenset({"report.json", "kernel.metal"})
BEST_PREFIX = "best."

_ROTATE_LOCK = threading.Lock()


def runs_root() -> Path:
    """@brief 运行根：优先 ``AI_LOWERLING_RUNS_DIR``，否则仓库下的 ``runs/``。

    @return 运行根目录（不解析、不创建）。
    """
    env = os.environ.get("AI_LOWERLING_RUNS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "runs"


def journal_dir(root: Path | None = None) -> Path:
    """@brief 日志目录。

    @param root 运行根；None 时取 :func:`runs_root`。
    @return ``<runs>/.index_journal``。
    """
    return Path(root or runs_root()) / DIR_NAME


def shards(root: Path | None = None) -> list[Path]:
    """@brief 当前可消费的分片（``<pid>.jsonl``，不含轮转出来的 ``.1``）。

    @param root 运行根；None 时取 :func:`runs_root`。
    @return 按文件名排序的分片列表。
    """
    folder = journal_dir(root)
    try:
        return sorted(
            (entry for entry in folder.iterdir() if entry.is_file() and entry.suffix == ".jsonl"),
            key=lambda item: item.name,
        )
    except OSError:
        return []


def _root_text(root: Path) -> str:
    """@brief 根的规整文本（不解析符号链接）。

    @param root 运行根。
    @return ``normpath`` 后的文本。
    """
    return os.path.normpath(os.fspath(root))


def relative_run(path: Path, root: Path | None = None) -> str | None:
    """@brief 一个被写的文件对应的“日志 run 值”；不在 ``runs/`` 下或不在白名单时返回 None。

    @param path 刚写完的文件路径。
    @param root 运行根；None 时取 :func:`runs_root`。
    @return 相对 ``runs/`` 的目录路径（候选文件是它上两级，``experiment.json`` 是自己所在目录）。
    """
    path = Path(path)
    name = path.name
    if name in CANDIDATE_FILES:
        if path.parent.parent.name != "candidates":
            return None
        directory = path.parent.parent.parent
    elif name in TOP_FILES or name.startswith(BEST_PREFIX):
        directory = path.parent
    else:
        return None
    text = os.path.normpath(os.fspath(directory))
    base = _root_text(Path(root or runs_root()))
    if text == base:
        return "."
    prefix = base + os.sep
    if not text.startswith(prefix):
        return None
    return text[len(prefix):].replace(os.sep, "/")


def note(path: Path, root: Path | None = None) -> None:
    """@brief 记一行“这个 run 的这个文件落盘了”。永不抛错。

    @param path 刚写完的文件路径。
    @param root 运行根；None 时取 :func:`runs_root`。
    """
    try:
        run = relative_run(path, root)
        if run is None:
            return
        entry = {"run": run, "path": Path(path).name, "at": time.time()}
        _append(Path(root or runs_root()), json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - 旁路日志失败不能影响搜索或看板写入
        LOG.warning("index journal write failed for %s", path)


def _append(root: Path, line: str) -> None:
    """@brief 追加一行到本进程分片；到 4MB 先原子轮转。

    @param root 运行根。
    @param line 已序列化的一行（带换行）。
    """
    folder = journal_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    shard = folder / f"{os.getpid()}.jsonl"
    with _ROTATE_LOCK:
        try:
            if shard.stat().st_size >= SHARD_LIMIT:
                os.replace(shard, folder / f"{os.getpid()}.jsonl.1")
        except OSError:
            pass
    with shard.open("a", encoding="utf-8") as handle:
        handle.write(line)


def read_new(shard: Path, offset: int, inode: int) -> tuple[list[dict], int, int, bool]:
    """@brief 从 ``offset`` 读一个分片的新增完整行。

    轮转/截断检测：inode 变了或当前大小小于已消费偏移，就从 0 重读并返回 ``reset=True``，
    调用方据此强制一次全量对账。末尾没有换行的半行不解析、也不前移偏移。

    @param shard 分片路径。
    @param offset 已消费的字节偏移。
    @param inode 上次记录的 inode；0 表示未知。
    @return ``(entries, new_offset, new_inode, reset)``。
    """
    try:
        stat = shard.stat()
    except OSError:
        return [], 0, 0, False
    reset = False
    # inode 未知（0，例如第一次读或旧库没有这列）不算轮转；大小回退或 inode 变了才算。
    if (inode and inode != stat.st_ino) or stat.st_size < offset:
        offset = 0
        reset = True
    try:
        with shard.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
    except OSError:
        return [], offset, stat.st_ino, reset
    cut = data.rfind(b"\n")
    if cut < 0:
        return [], offset, stat.st_ino, reset
    chunk = data[:cut + 1]
    entries = []
    for line in chunk.splitlines():
        try:
            item = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        if isinstance(item, dict) and isinstance(item.get("run"), str):
            entries.append(item)
    return entries, offset + len(chunk), stat.st_ino, reset
