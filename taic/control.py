"""Budget control of one run directory: the live top-up file, the budget history, the one-process-per-run lock.

Standard library only (no ``tcenv``, no ``openai``), so that the dashboard and the command line can use it without the search stack.

Files in a run directory (all JSON, all written atomically):

* ``control.json``        ``{"budget_usd": <new TOTAL budget>, "who": ..., "amount_usd": ..., "seq": n, "requested_at": t}``. A running search
                          reads it before every budget gate (a ``stat`` of one file) and raises its budget to it; it never goes below what is already spent.
* ``budget_events.json``  list of every budget change: ``kind`` is ``requested`` (written by whoever asked: dashboard or cli), ``applied`` (written by the
                          search process when it took the new budget) or ``resume`` (a stopped run was started again with another budget).
* ``resume.lock``         ``{"pid": ..., "started_at": ...}``: one process per run directory may continue it. A lock of a dead pid is stale and is taken over.
* ``run_config.json``     the settings the search was started with (written by the orchestrator; never holds credentials).
"""

# 流水线里的位置：问模型、记美元、按预算或轮数停，在 taic/llm.py 和编排器。本文件站在运行目录一侧，
# 让人在搜索还没结束时抬高总预算，并保证同一个目录不会被两个活进程同时继续。只用标准库。
#
# control.json：write_control 写，read_control 读。budget_usd 是新的总预算，不是这次追加的金额。
#   本文件只落盘，不改账本。活进程在 taic/llm.py 的 CostLedger.refresh_budget：文件变了才读，
#   总预算变成 max(文件里的数, 已花费)。可以抬高后继续，也可以降低，但不会降到已花费以下。
# budget_events.json：record_event 追加，read_events / budget_history 读。write_control 写的 kind 是 requested。
#   applied（搜索采纳新预算时）和 resume（停下的运行换预算再启动）由调用方传入 kind，不在本文件里分支。
# resume.lock：acquire_run_lock 创建，lock_holder 查看；release_run_lock 只在 pid 是自己时删除。
#   活着的别的 pid 占着就拒绝。接管在独占 flock 里完成：先写上自己的 pid 并 fsync，再放开，读到的文件不会是空的。
# run_config.json：这里只有文件名。模块说明由编排器写入启动时的设置，且不含密钥；本文件不读写它。

from __future__ import annotations

import fcntl
import json
import math
import os
import time
from pathlib import Path

CONTROL_FILE = "control.json"
EVENTS_FILE = "budget_events.json"
LOCK_FILE = "resume.lock"
CONFIG_FILE = "run_config.json"

WHO = ("dashboard", "cli", "api")


class ControlError(ValueError):
    """A budget request that cannot be honoured (bad number, unknown run, ...)."""


class RunLockError(RuntimeError):
    """Another live process already holds the run directory."""


# ---------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------


# 读者可能正在打开同一个文件。先写到旁边的临时文件，再用 replace 换成正式名字：要么看到旧的完整 JSON，要么看到新的，不会看到写了一半的。
def write_json_atomic(path: Path, doc) -> None:
    """Write ``doc`` next to ``path`` and ``os.replace`` it into place: a reader sees the old or the new file, never half of one. Raises on failure."""
    write_text_atomic(path, json.dumps(doc, ensure_ascii=False, indent=1, default=str))


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` next to ``path`` and ``os.replace`` it into place.

    The kernel, its report and its recheck record must land as a matched pair: a kill between two
    plain ``write_text`` calls leaves a truncated kernel beside a complete report, and a resume
    would pair them by directory name and cache the truncated source under the full report's verdict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            # replace 成功时临时名字已经没了，删不到是正常的。写失败时这里把半截临时文件清掉。
            pass


# 文件缺失或不是完整 JSON 都当成没有。原子替换就是为了让读者少碰到「写到一半所以被当成没有」。
def read_json(path: Path):
    """The parsed JSON of ``path``, or ``None`` when it is missing or damaged."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def append_jsonl(path: Path, row, *, sort_keys: bool = False) -> None:
    """Append one JSON record and fsync it, starting a new line if the previous writer was killed.

    ``feedback.jsonl`` is read by sequence number and ``(run_id, seq)`` is the primary key of the
    derived index. A torn last line followed by an append fuses two records into one unparsable
    line; ``max_seq`` then skips it, the sequence number is reused, and one session's history is
    overwritten in the index.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, sort_keys=sort_keys) + "\n"
    needs_newline = False
    try:
        if path.is_file() and path.stat().st_size > 0:
            with path.open("rb") as probe:
                probe.seek(-1, os.SEEK_END)
                needs_newline = probe.read(1) != b"\n"
    except OSError:
        needs_newline = False
    with path.open("a", encoding="utf-8") as handle:
        if needs_newline:
            handle.write("\n")
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


# 信号 0 不杀进程，只问这个 pid 还在不在。锁文件会比进程活得长，所以过期看 pid，不看文件时间。
def pid_alive(pid) -> bool:
    try:
        n = int(pid)
    except (TypeError, ValueError):
        return False
    if n <= 0:
        return False
    try:
        os.kill(n, 0)
    except ProcessLookupError:
        # 没有这个进程：锁过期。
        return False
    except PermissionError:
        # 进程还在，只是不能给它发信号。当成活的，避免把别人的锁当成过期抢走。
        return True
    except OSError:
        return False
    return True


# 预算必须是有限正数。bool 是 int 的子类，不先排除的话 True 会变成 1.0。
def _finite_positive(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def iso(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time() if t is None else t))


# ---------------------------------------------------------------------------------------------
# control file (live top-up)
# ---------------------------------------------------------------------------------------------


def control_path(run_dir: Path) -> Path:
    return Path(run_dir) / CONTROL_FILE


def parse_control(doc) -> tuple[float | None, str]:
    """(requested total budget, problem). ``problem`` is empty when the document is usable."""
    if not isinstance(doc, dict):
        return None, "not a JSON object"
    b = _finite_positive(doc.get("budget_usd"))
    if b is None:
        return None, f"budget_usd must be a positive number, got {doc.get('budget_usd')!r}"
    return b, ""


# 缺失、损坏，或 budget_usd 不是正数，都当成没有加钱请求，避免一份坏文件改掉预算。
def read_control(run_dir: Path) -> dict | None:
    """The control document of ``run_dir`` if it exists and is usable (otherwise ``None``)."""
    doc = read_json(control_path(run_dir))
    return doc if parse_control(doc)[0] is not None else None


# 向活搜索提出新的总预算，并记一条 requested。这里只校验「是正数」，不和已花费比较；下界在读文件的账本里。
def write_control(run_dir: Path, budget_usd: float, *, who: str, amount_usd: float | None = None, old_budget_usd: float | None = None, note: str = "", max_rounds: int | None = None, patience: int | None = None) -> dict:
    """Ask the running search of ``run_dir`` to raise its total budget to ``budget_usd``. Atomic; records a ``requested`` event.

    ``max_rounds`` and ``patience``, when given, are raised the same way (a live search only takes a value higher than the one it already has)."""
    b = _finite_positive(budget_usd)
    if b is None:
        raise ControlError(f"the new budget must be a positive number, got {budget_usd!r}")
    prev = read_json(control_path(run_dir))
    # 旧文件缺失或损坏时序号从 1 重新计，不让坏序号把这次请求写崩。
    seq = int(prev.get("seq", 0)) + 1 if isinstance(prev, dict) and isinstance(prev.get("seq"), (int, float)) else 1
    doc = {"budget_usd": b, "who": who, "amount_usd": amount_usd, "seq": seq, "requested_at": time.time()}
    if max_rounds is not None:
        doc["max_rounds"] = int(max_rounds)
    if patience is not None:
        doc["patience"] = int(patience)
    write_json_atomic(control_path(run_dir), doc)
    record_event(run_dir, "requested", old_budget_usd, b, who=who, note=note)
    return doc


# ---------------------------------------------------------------------------------------------
# budget history
# ---------------------------------------------------------------------------------------------


def events_path(run_dir: Path) -> Path:
    return Path(run_dir) / EVENTS_FILE


def read_events(run_dir: Path) -> list[dict]:
    doc = read_json(events_path(run_dir))
    return [e for e in doc if isinstance(e, dict)] if isinstance(doc, list) else []


# 预算变更的审计。丢一行也不能让搜索停，所以任何异常都吞掉。
# 仪表盘和搜索会同时追加。原子替换挡不住「两边都读到旧列表、后写的覆盖先写的」，所以先拿独占锁再读-改-写。
def record_event(run_dir: Path, kind: str, old_budget_usd: float | None, new_budget_usd: float | None, *, who: str = "cli", note: str = "", **extra) -> dict:
    """Append one entry to ``budget_events.json`` (read-modify-write under an exclusive file lock, so the dashboard and the search process can both write).
    Never raises: losing one line must not stop a search. The returned dict has ``persisted`` False when the line did not land;
    that file is then not a complete bill."""
    now = time.time()
    event = {
        "t": now, "time": iso(now), "kind": kind, "who": who,
        "old_budget_usd": old_budget_usd, "new_budget_usd": new_budget_usd,
        "amount_usd": (new_budget_usd - old_budget_usd) if (old_budget_usd is not None and new_budget_usd is not None) else None,
        "pid": os.getpid(), "note": note, **extra,
    }
    try:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / ".budget_events.lock", "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                events = read_events(run_dir)
                events.append(event)
                write_json_atomic(events_path(run_dir), events)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    except Exception:  # noqa: BLE001
        # 落盘失败的那一行不在文件里。返回值带 persisted=False，调用方不能把账本当成完整账单。
        return {**event, "persisted": False}
    return {**event, "persisted": True}


# 仪表盘用。requested 若还没有同金额的 applied 或 resume，算搜索还没采纳；只把最后一条这种挂起显示出来。
def budget_history(run_dir: Path, current_budget: float | None = None) -> dict:
    """What the dashboard shows: the budget the run started with, the budget now and every change that took effect.
    ``requested`` entries that were never applied are listed as ``pending`` (the search has not read them yet)."""
    events = read_events(run_dir)
    effective = [e for e in events if e.get("kind") in ("applied", "resume")]
    initial = None
    for e in effective:
        if e.get("old_budget_usd") is not None:
            initial = e["old_budget_usd"]
            break
    pending = []
    for index, event in enumerate(events):
        if event.get("kind") != "requested" or event.get("new_budget_usd") is None:
            continue
        try:
            amount = round(float(event["new_budget_usd"]), 9)
        except (TypeError, ValueError):
            continue
        later_hit = False
        for item in events[index + 1:]:
            if item.get("kind") not in ("applied", "resume") or item.get("new_budget_usd") is None:
                continue
            try:
                if round(float(item["new_budget_usd"]), 9) == amount:
                    later_hit = True
                    break
            except (TypeError, ValueError):
                continue
        if later_hit:
            continue
        pending.append(event)
    return {
        "initial_budget_usd": initial if initial is not None else current_budget,
        "budget_usd": current_budget,
        "topped_up": bool(effective),
        "events": events,
        "applied": effective,
        "pending": pending[-1:] if pending else [],
    }


# ---------------------------------------------------------------------------------------------
# one process per run directory
# ---------------------------------------------------------------------------------------------


def lock_path(run_dir: Path) -> Path:
    return Path(run_dir) / LOCK_FILE


# 给外面看谁占着这个目录。锁文件还在但 pid 已死，视为没人占。
def lock_holder(run_dir: Path) -> dict | None:
    """The live holder of the run's lock (``None`` when there is no lock or its pid is dead)."""
    doc = read_json(lock_path(run_dir))
    if isinstance(doc, dict) and pid_alive(doc.get("pid")):
        return doc
    return None


def _lock_is_ours(doc) -> bool:
    """The file names this process, or the dashboard's ``uv`` parent marked ``starting``."""
    return isinstance(doc, dict) and (doc.get("pid") == os.getpid() or (doc.get("starting") and doc.get("pid") == os.getppid()))


# 一个运行目录只许一个活进程继续：两个进程一起写候选内核和状态，会把同一次运行拆成两份互相覆盖的账。
# 独占 flock 盖住「读旧内容」和「写上自己的 pid」。写完并 fsync 之后才放开，对方读到的不会是空文件。
# 对方 pid 还活着就拒绝。pid 已死、文件是空的，或就是自己上次留下的，在这把 flock 里改写成自己的 pid。
def acquire_run_lock(run_dir: Path, who: str = "resume") -> Path:
    """Claim ``run_dir`` for this process. Raises ``RunLockError`` if a live process holds it; a lock whose pid is dead is stale and replaced."""
    path = lock_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        doc = read_json(path)
        if isinstance(doc, dict) and pid_alive(doc.get("pid")) and not _lock_is_ours(doc):
            raise RunLockError(f"{run_dir} is already being resumed by pid {doc.get('pid')} (started {iso(doc.get('started_at'))})")
        payload = json.dumps({"pid": os.getpid(), "started_at": time.time(), "who": who})
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, payload.encode())
        os.fsync(fd)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    return path


# 只删除本进程写下的锁。别的 pid 的锁不动，避免把刚接手的进程清掉。
def release_run_lock(run_dir: Path) -> None:
    doc = read_json(lock_path(run_dir))
    if isinstance(doc, dict) and doc.get("pid") == os.getpid():
        try:
            lock_path(run_dir).unlink()
        except OSError:
            pass
