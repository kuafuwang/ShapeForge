"""Tool-topic lifecycle for feedback.

A code change records ``possibly_stale``. It does not close the topic, delete
it, or stand in for ``change_ref``. Zero new reports are not a fix unless a
later run actually exercised the same command or code path, and even then the
note is evidence on the hint, not a close. Closing still takes
``close-topic --dry-run`` and then ``--confirm``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "taic.feedback_tool_topics.1"
CHANGE_REF = re.compile(r"^commit:[0-9a-f]{7,64}$")
OPEN_STATUSES = frozenset({"pending", "info"})


class TopicError(ValueError):
    """@brief 关单或复核提示无法写入。文件保持原样。"""


def _load(path: Path) -> dict:
    if not path.is_file():
        return {"schema": SCHEMA, "topics": {}}
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("topics"), dict):
        raise TopicError(f"{path} is not a tool-topic ledger")
    document.setdefault("schema", SCHEMA)
    return document


def _save(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def record_tool_topic(
    path: Path,
    topic: str,
    *,
    head: str,
    dirty: bool,
    modules: Mapping[str, str],
    claims: list[Mapping[str, Any]] | None = None,
) -> dict:
    """@brief 记下一条工具类 topic 当时的 HEAD、脏状态和相关文件摘要。

    @param path 账本路径。
    @param topic topic 名。
    @param head 记录时的 git HEAD。
    @param dirty 工作区当时是否脏。
    @param modules 相关文件到 SHA-256 的映射。
    @param claims 这条 topic 盖住的 claim。状态必须是待审或 info。
    @return 写下的 topic 记录。
    """
    name = str(topic or "").strip()
    if not name:
        raise TopicError("topic is empty")
    document = _load(path)
    if name in document["topics"]:
        raise TopicError(f"topic {name!r} already exists")
    stored_claims = []
    for item in claims or []:
        status = str(item.get("status") or "pending")
        if status not in OPEN_STATUSES:
            raise TopicError(f"a new tool topic cannot start from status {status!r}")
        stored_claims.append(
            {
                "run_id": str(item.get("run_id") or ""),
                "seq": int(item["seq"]),
                "digest": str(item["digest"]),
                "status": status,
            }
        )
    row = {
        "topic": name,
        "status": "open",
        "head": str(head),
        "dirty": bool(dirty),
        "modules": {str(key): str(value) for key, value in sorted(modules.items())},
        "stale_hint": "",
        "before": {},
        "after": {},
        "observed_at": "",
        "coverage": "",
        "zero_reproduction": False,
        "fixed": False,
        "change_ref": "",
        "claims": stored_claims,
    }
    document["topics"][name] = row
    _save(path, document)
    return row


def refresh_stale_hint(
    path: Path,
    topic: str,
    *,
    modules_now: Mapping[str, str],
    coverage: Mapping[str, Any] | None = None,
    observed_at: str,
) -> dict:
    """@brief 相关文件变了就附加 possibly_stale。没有覆盖证据不能写成已修复。

    @param path 账本路径。
    @param topic topic 名。
    @param modules_now 这些文件现在的摘要。
    @param coverage ``exercised`` 表示后来的 run 有没有走到同一路径；
        ``new_reports`` 是这条路径上的新上报数。缺省当作没执行到。
    @param observed_at 观察时间，由调用方传入，便于测试固定。
    @return 更新后的 topic。状态、``change_ref`` 和 ``fixed`` 都不变成关单。
    """
    document = _load(path)
    row = document["topics"].get(str(topic))
    if not isinstance(row, dict):
        raise TopicError(f"no tool topic {topic!r}")
    before = dict(row.get("modules") or {})
    after = {str(key): str(modules_now.get(key) or "") for key in before}
    changed = any(before.get(key) != after.get(key) for key in before)
    exercised = bool((coverage or {}).get("exercised"))
    new_reports = (coverage or {}).get("new_reports")
    row["fixed"] = False
    row["change_ref"] = str(row.get("change_ref") or "")
    row["status"] = "open" if row.get("status") != "closed" else row["status"]
    if row.get("status") == "closed":
        raise TopicError("a closed topic is not refreshed into a fix")
    if changed:
        row["stale_hint"] = "possibly_stale"
        row["before"] = before
        row["after"] = after
        row["observed_at"] = str(observed_at)
    else:
        row["stale_hint"] = str(row.get("stale_hint") or "")
    if not exercised:
        row["coverage"] = "not_exercised"
        row["zero_reproduction"] = False
    elif new_reports == 0:
        row["coverage"] = "exercised"
        row["zero_reproduction"] = True
    else:
        row["coverage"] = "exercised"
        row["zero_reproduction"] = False
    row["status"] = "open"
    row["change_ref"] = ""
    row["fixed"] = False
    document["topics"][str(topic)] = row
    _save(path, document)
    return row


def close_topic(
    path: Path,
    topic: str,
    change_ref: str,
    *,
    dry_run: bool = False,
    confirm: bool = False,
    live_status: Mapping[tuple[str, int, str], str] | None = None,
) -> dict:
    """@brief 整批关闭一个 topic。任一条对不上就整份不写。

    @param path 账本路径。
    @param topic topic 名。
    @param change_ref ``commit:`` 加提交号。``possibly_stale`` 不能代替它。
    @param dry_run 只返回计划。
    @param confirm 校验通过后一次写入。
    @param live_status 可选的当前状态。给出时，账本里的每一条都必须出现在这张表里，
        并且状态一致。缺一条或对不上就整批拒绝。
    @return 计划或已关闭的 topic。
    @raises TopicError 参数、状态或引用对不上时抛出，账本字节不变。
    """
    if dry_run == confirm:
        raise TopicError("pass exactly one of --dry-run or --confirm")
    ref = str(change_ref or "").strip()
    if not CHANGE_REF.fullmatch(ref):
        raise TopicError("change_ref must look like commit:<sha>")
    if not path.is_file():
        raise TopicError("no tool-topic ledger")
    original = path.read_text(encoding="utf-8")
    document = _load(path)
    row = document["topics"].get(str(topic))
    if not isinstance(row, dict):
        raise TopicError(f"no tool topic {topic!r}")
    claims = list(row.get("claims") or [])
    if not claims:
        raise TopicError("the topic has no claims to close")
    planned = []
    for index, claim in enumerate(claims):
        status = str(claim.get("status") or "")
        key = (str(claim.get("run_id") or ""), int(claim.get("seq") or 0), str(claim.get("digest") or ""))
        if live_status is not None and key not in live_status:
            raise TopicError(f"item {index}: claim is missing from the live status map")
        if live_status is not None and live_status[key] != status:
            raise TopicError(f"item {index}: claim status is {live_status[key]}, expected {status}")
        if status not in OPEN_STATUSES:
            raise TopicError(f"item {index}: status {status!r} does not match an open tool topic")
        if claim.get("stale_hint") == "possibly_stale" and not ref:
            raise TopicError("a stale hint cannot close a topic")
        planned.append({**claim, "to_status": "done", "change_ref": ref})
    if str(row.get("stale_hint") or "") == "possibly_stale" and dry_run:
        # The hint is visible in the plan and is not itself the close.
        pass
    plan = {"dry_run": True, "topic": row["topic"], "items": planned, "change_ref": ref, "closed": False}
    if dry_run:
        if path.read_text(encoding="utf-8") != original:
            raise TopicError("dry-run must not write")
        return plan
    for claim in row["claims"]:
        claim["status"] = "done"
        claim["change_ref"] = ref
    row["status"] = "closed"
    row["change_ref"] = ref
    row["fixed"] = False
    row["stale_hint"] = str(row.get("stale_hint") or "")
    document["topics"][str(topic)] = row
    _save(path, document)
    return {"dry_run": False, "topic": row["topic"], "items": planned, "change_ref": ref, "closed": True, "stale_hint": row["stale_hint"]}


def open_tool_topic_lines(path: Path) -> list[str]:
    """@brief 列出尚未关闭的工具类 topic。文件不存在时为空，不报错。

    @param path 账本路径。
    @return 给人看的短行。
    """
    if not path.is_file():
        return []
    try:
        document = _load(path)
    except (OSError, json.JSONDecodeError, TopicError):
        return [f"open tool topics: could not read {path}"]
    lines = []
    for name, row in sorted(document.get("topics", {}).items()):
        if not isinstance(row, dict) or row.get("status") == "closed":
            continue
        hint = row.get("stale_hint") or "current"
        lines.append(f"open tool topic {name} ({hint})")
    return lines


def ledger_path() -> Path:
    """@brief 工具 topic 账本。测试用 ``TAIC_TOOL_TOPICS`` 指到临时文件。"""
    return Path(os.environ.get("TAIC_TOOL_TOPICS", "runs/tool_topics.json"))


def modules_from_text(text: str) -> dict[str, str]:
    """@brief 从评审说明里找出已经存在的文件，并计算 SHA-256。

    @param text 评审 note。
    @return 路径到摘要的映射。目录和读失败的路径不记。
    """
    found: dict[str, str] = {}
    for token in re.findall(r"\S+", text or ""):
        candidate = Path(token.strip("`'\"()[]{},;"))
        if not candidate.is_file():
            continue
        try:
            found[str(candidate)] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        except OSError:
            continue
    return found


def note_reviewed_topic(
    path: Path,
    *,
    topic: str,
    run_id: str,
    seq: int,
    digest: str,
    status: str,
    note: str,
    head: str,
    dirty: bool,
) -> dict | None:
    """@brief 一条工具类 claim 被评审后，建档或并入已有 topic。

    已关闭的 topic 不再打开。同一 claim 不重复追加。后来的评审不覆盖
    最初记下的文件摘要，这样随后的字节变化仍能对上原来的记录。

    @return 写下的 topic。不是待审或 info 时返回 None。
    """
    name = str(topic or "").strip()
    if not name or str(status) not in OPEN_STATUSES:
        return None
    modules = modules_from_text(note)
    claim = {"run_id": str(run_id), "seq": int(seq), "digest": str(digest), "status": str(status)}
    if not path.is_file() or name not in _load(path)["topics"]:
        return record_tool_topic(path, name, head=str(head), dirty=bool(dirty), modules=modules, claims=[claim])
    document = _load(path)
    row = document["topics"][name]
    if row.get("status") == "closed":
        return row
    claims = list(row.get("claims") or [])
    key = (claim["run_id"], claim["seq"], claim["digest"])
    if not any((str(item.get("run_id")), int(item.get("seq") or 0), str(item.get("digest"))) == key for item in claims):
        claims.append(claim)
    stored = dict(row.get("modules") or {})
    for key_path, digest_now in modules.items():
        stored.setdefault(key_path, digest_now)
    row["claims"] = claims
    row["modules"] = {str(key_path): str(value) for key_path, value in sorted(stored.items())}
    document["topics"][name] = row
    _save(path, document)
    return row


def refresh_open_topics(path: Path, *, observed_at: str) -> None:
    """@brief 按账本里记下的文件重新计算摘要。变化只产生 possibly_stale。

    没有后续执行记录，所以覆盖写成 ``not_exercised``。已关闭的 topic 跳过。

    @param path 账本路径。文件不存在时什么都不做。
    @param observed_at 观察时间。
    """
    if not path.is_file():
        return
    names = [
        name
        for name, row in _load(path).get("topics", {}).items()
        if isinstance(row, dict) and row.get("status") != "closed"
    ]
    for name in names:
        current = _load(path)["topics"][name]
        modules_now = {}
        for key in current.get("modules") or {}:
            file = Path(key)
            try:
                modules_now[key] = hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else ""
            except OSError:
                modules_now[key] = ""
        refresh_stale_hint(path, name, modules_now=modules_now, coverage={"exercised": False}, observed_at=observed_at)


def hint_open_tool_topics(log: Any = print) -> None:
    """@brief 真正付费前打印未关闭的工具 topic。读失败只提示，不拦截搜索。"""
    try:
        announce_open_tool_topics(ledger_path(), log)
    except Exception as exc:
        try:
            log(f"open tool topics: {exc}")
        except Exception:
            return


def announce_open_tool_topics(path: Path, log: Any = print, *, observed_at: str | None = None) -> list[str]:
    """@brief 付费实验开始前按当前文件重算未关闭的工具 topic，然后打印。只提示，不拦截，不关单。

    @param path 账本路径。
    @param log 输出函数。
    @param observed_at 观察时间。缺省用当前 UTC 时间。
    @return 打印过的行。读失败也只提示。
    """
    from datetime import datetime, timezone

    try:
        refresh_open_topics(path, observed_at=observed_at or datetime.now(timezone.utc).isoformat())
    except (OSError, json.JSONDecodeError, TopicError):
        pass
    lines = open_tool_topic_lines(path)
    for line in lines:
        log(line)
    return lines
