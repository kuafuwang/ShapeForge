"""Archive one finished experiment directory or one search directory, and append a reason. The log is append-only.

The directories move under the observe data directory, next to the index, and out of ``runs/``.
The spend ledger is not reduced. Nothing in this module rewrites or truncates the log.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from observe import detail
from observe.actions import Refused
from observe.index import borrow, db_path, get_index
from observe.ledger import operator_name
from observe.query import refresh

_SKIP = {"candidates", "__pycache__"}
_REASON_LIMIT = 500


def log_path() -> Path:
    """@brief 返回删除日志 ``deletion_log.jsonl`` 的路径。

    @return 索引所在目录下的删除日志路径。
    """
    return db_path().parent / "deletion_log.jsonl"


def archive_root() -> Path:
    """@brief 返回归档目录 ``deleted/`` 的路径。

    @return 索引所在目录下的归档根目录。
    """
    return db_path().parent / "deleted"


def _reason(text: object) -> str:
    """@brief 校验并规范化删除理由文本。

    @param text 请求中提交的理由，需为非空字符串。
    @return 去掉首尾空白的理由。
    @raises Refused 理由不是字符串、为空或超过 500 字符时抛出（HTTP 400）。
    """
    if not isinstance(text, str):
        raise Refused(400, "a deletion needs a reason")
    reason = text.strip()
    if not reason:
        raise Refused(400, "a deletion needs a reason")
    if len(reason) > _REASON_LIMIT:
        raise Refused(400, f"the reason must be at most {_REASON_LIMIT} characters")
    return reason


def _under_runs(relative: str) -> Path:
    """@brief 把相对名称解析成 ``runs/`` 下的目录，并拒绝越界路径。

    @param relative 相对 ``runs/`` 根目录的路径文本。
    @return 解析后的绝对目录路径。
    @raises Refused 名称为空、绝对路径、含 ``..`` 或越出 ``runs/`` 时抛出；目录不存在时抛 404。
    """
    if not isinstance(relative, str) or not relative.strip() or "\x00" in relative:
        raise Refused(400, "the record name is empty")
    rel = Path(relative.strip())
    if rel.is_absolute() or ".." in rel.parts:
        raise Refused(400, "that path is not inside runs/")
    root = detail.runs_root().resolve()
    try:
        path = (root / rel).resolve()
    except OSError as exc:
        raise Refused(400, "that path is not inside runs/") from exc
    if path == root or root not in path.parents:
        raise Refused(400, "that path is not inside runs/")
    if not path.is_dir():
        raise Refused(404, "no such record")
    return path


def _operator(name: object) -> str:
    """@brief 校验并规范化 operator（任务）名。

    @param name 请求中的 operator 名，需为非空字符串。
    @return 规范化后的 operator 名。
    @raises Refused 名称非法、为 ``unknown`` 或带路径分隔符时抛出（HTTP 400）。
    """
    if not isinstance(name, str):
        raise Refused(400, "the operator name is empty")
    cleaned = operator_name(name.strip())
    if not cleaned or cleaned == "unknown" or "/" in cleaned or "\\" in cleaned or ".." in cleaned:
        raise Refused(400, "the operator name is not a task name")
    return cleaned


def _searches(root: Path) -> list[Path]:
    """@brief 在目录树中找出所有单次搜索目录（含 ``state.json`` 或 ``result.json``）。

    @param root 递归搜索的起点目录。
    @return 搜索目录路径列表，起点目录自身若符合条件也会包含在内。
    """
    found: list[Path] = []
    if (root / "state.json").exists() or (root / "result.json").exists():
        found.append(root)
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _SKIP and not name.startswith(".")]
        path = Path(current)
        if path != root and ("state.json" in filenames or "result.json" in filenames):
            found.append(path)
    return found


def _live(directory: Path) -> bool:
    """@brief 判断一个搜索目录是否仍在运行。

    @param directory 搜索目录。
    @return 存活状态为 ``running`` 时返回 True。
    """
    state = detail._load(directory / "state.json")
    result = detail._load(directory / "result.json") if (directory / "result.json").exists() else None
    return detail.liveness(state, result, directory) == "running"


def _relative(path: Path) -> str:
    """@brief 取搜索目录相对 ``runs/`` 根目录的 POSIX 路径。

    @param path 搜索目录。
    @return 相对路径文本，用于写入日志与索引。
    """
    return path.resolve().relative_to(detail.runs_root().resolve()).as_posix()


def _read_log() -> list[dict]:
    """@brief 读取删除日志的全部条目，跳过空行与无法解析的行。

    @return 日志条目字典列表，按文件顺序（旧到新）。
    """
    path = log_path()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def list_deletions() -> dict:
    """@brief 返回删除日志（旧到新）；这是页面读取该日志的唯一途径。

    原始英文说明：The log, oldest first. This is the only way the page reads it.

    @return 含 ``deletions`` 键、值为日志条目列表的字典。
    """
    return {"deletions": _read_log()}


def _append(entry: dict) -> None:
    """@brief 以追加方式将一条记录写入删除日志，并 fsync 落盘。

    @param entry 待写入的日志条目。
    """
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def _archive(paths: list[Path], stamp: str) -> str:
    """@brief 把给定目录搬到 ``deleted/<stamp>/`` 下，保留相对 ``runs/`` 的路径结构。

    @param paths 待搬走的目录列表。
    @param stamp 归档批次名，同时用作目录名。
    @return 归档批次目录名（``stamp``）。
    """
    folder = archive_root() / stamp
    folder.mkdir(parents=True, exist_ok=False)
    root = detail.runs_root().resolve()
    for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
        relative = path.resolve().relative_to(root).as_posix()
        dest = folder / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))
    return folder.name


def _runs_for_operator(name: str) -> list[str]:
    """@brief 在索引中找出属于某 operator 的全部 run id。

    @param name 已规范化的 operator 名。
    @return 匹配的 run id 列表。
    """
    refresh()
    with borrow() as index:
        rows = index.conn.execute("SELECT run_id, task, operator FROM runs").fetchall()
    matched = []
    for row in rows:
        label = row["operator"] or operator_name(row["task"])
        if label == name:
            matched.append(row["run_id"])
    return matched


def _client_ip(value: object) -> str:
    """@brief 规范化客户端地址，非字符串或空值记为 ``unknown``。

    @param value 请求方地址。
    @return 去空白后的地址；无法识别时返回 ``"unknown"``。
    """
    if not isinstance(value, str):
        return "unknown"
    host = value.strip()
    return host or "unknown"


def delete_records(kind: object, name: object, reason: object, client_ip: object = "") -> dict:
    """@brief 把已结束的记录移出 ``runs/`` 并追加一行日志；有搜索在跑时整个请求被拒。

    @param kind 记录类型，取 ``experiment``、``operator`` 或 ``run``。
    @param name 记录名（实验目录名、operator 名或搜索目录名）。
    @param reason 删除理由，必填。
    @param client_ip 请求方地址，仅记录用，缺省为空。
    @return 含 ``ok``、``archive``、``kind``、``name`` 与 ``runs`` 的结果字典。
    @raises Refused kind 非法、名称或理由非法时抛 400；记录不存在时抛 404；仍有搜索在跑时抛 409。
    """
    kept = _reason(reason)
    peer = _client_ip(client_ip)
    if kind == "experiment":
        if not isinstance(name, str):
            raise Refused(400, "the record name is empty")
        root = _under_runs(name)
        targets = [root]
        run_dirs = _searches(root)
        experiment_ids = [name.strip()]
    elif kind == "operator":
        cleaned = _operator(name)
        found = _runs_for_operator(cleaned)
        if not found:
            raise Refused(404, f"no records for operator {cleaned}")
        targets = []
        run_dirs = []
        for run_id in found:
            path = detail.resolve_run(run_id)
            if path is None:
                continue
            targets.append(path)
            run_dirs.append(path)
        if not targets:
            raise Refused(404, f"no records for operator {cleaned}")
        experiment_ids = []
        name = cleaned
    elif kind == "run":
        if not isinstance(name, str):
            raise Refused(400, "the record name is empty")
        root = _under_runs(name)
        if not (root / "state.json").exists() and not (root / "result.json").exists():
            raise Refused(404, "no such record")
        targets = [root]
        run_dirs = [root]
        experiment_ids = []
    else:
        raise Refused(400, "kind must be experiment, operator or run")
    live = [_relative(path) for path in run_dirs if _live(path)]
    if live:
        raise Refused(409, "still running, nothing was deleted: " + ", ".join(live))
    run_ids = [_relative(path) for path in run_dirs]
    stamp = f"{len(_read_log()) + 1:04d}-{time.strftime('%Y%m%dT%H%M%S')}"
    folder = _archive(targets, stamp)
    recorded = name.strip() if isinstance(name, str) else str(name)
    entry = {
        "archive": folder,
        "kind": kind,
        "name": recorded,
        "reason": kept,
        "runs": run_ids,
        "ip": peer,
        "t": time.time(),
        "who": "dashboard",
    }
    try:
        get_index().drop_archived(run_ids, experiment_ids)
    finally:
        _append(entry)
    return {"ok": True, "archive": folder, "kind": kind, "name": recorded, "runs": run_ids}
