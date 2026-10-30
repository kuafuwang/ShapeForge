"""Read agent feedback for the dashboard. Parsing lives in ``taic.agent_feedback``."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from taic.agent_feedback import PARSER_VERSION, _is_placeholder, _repair_record, load_records, read_appended_records

_REVIEWABLE = {"system"}
_PREMISE = {"verified", "unverified", "downgraded"}
_STATUSES = {"pending", "adopted", "rejected", "done", "info", "obsolete"}
_VERIFICATIONS = {"unchecked", "confirmed", "refuted"}
_SOURCES = {"report", "calibration", "probe", "code", "other"}


def _review_key(seq: int, digest: str) -> str:
    """@brief 拼出评审文件里 claim 的键。

    @param seq 反馈条目序号。
    @param digest 条目摘要。
    @return ``"<seq>:<digest>"`` 形式的键。
    """
    return f"{seq}:{digest}"


def _load_review(path: Path) -> dict:
    """@brief 读取评审文件；缺失、损坏或结构不对时返回空骨架。

    @param path ``feedback_review.json`` 路径。
    @return 含 ``v`` 与 ``claims`` 的字典。
    """
    if not path.is_file():
        return {"v": 1, "claims": {}}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"v": 1, "claims": {}}
    if not isinstance(doc, dict):
        return {"v": 1, "claims": {}}
    claims = doc.get("claims")
    if not isinstance(claims, dict):
        claims = {}
    return {"v": 1, "claims": claims}


def _has_system(record: dict) -> bool:
    """@brief 判断一条反馈是否带有面向系统的内容（kinds、求助段或系统建议段）。

    @param record 一条反馈记录。
    @return 含系统内容时为 True。
    """
    kinds = [part for part in str(record.get("kinds") or "").split(",") if part and part != "none"]
    sections = record.get("sections") if isinstance(record.get("sections"), dict) else {}
    return bool(kinds) or bool(str(sections.get("需要帮助") or "").strip()) or bool(str(sections.get("对系统的建议") or "").strip())


def _default_status(item: dict) -> str:
    """@brief 给出条目在未评审时的默认状态。

    @param item 反馈条目。
    @return 需要评审的条目为 ``pending``，否则为 ``info``。
    """
    audience = item.get("audience") or ""
    premise = item.get("premise") or ""
    if audience in _REVIEWABLE or premise in _PREMISE:
        return "pending"
    return "info"


def _claim_rows(record: dict, review: dict) -> list[dict]:
    """@brief 把一条记录的条目与已存评审合并成 claim 行。

    @param record 一条反馈记录。
    @param review 评审文件的 ``claims`` 映射。
    @return claim 行字典列表；序号不是整数时为空。
    """
    seq = record.get("seq")
    if not isinstance(seq, int):
        return []
    rows = []
    for item in record.get("items") or []:
        if not isinstance(item, dict):
            continue
        digest = str(item.get("digest") or "")
        if not digest:
            continue
        saved = review.get(_review_key(seq, digest))
        if not isinstance(saved, dict):
            saved = {}
        status = saved.get("status") or _default_status(item)
        if status not in _STATUSES:
            status = _default_status(item)
        rows.append({
            "seq": seq,
            "digest": digest,
            "section": item.get("section") or "",
            "audience": item.get("audience") or "",
            "premise": item.get("premise") or "",
            "text": item.get("text") or "",
            "status": status,
            "verification": saved.get("verification") or "unchecked",
            "topic": saved.get("topic") or "",
            "change_ref": saved.get("change_ref") or "",
            "reviewed_at": saved.get("t") or saved.get("reviewed_at"),
            "evidence_source": saved.get("evidence_source") or "",
            "evidence": saved.get("evidence") or "",
            "note": saved.get("note") or "",
            "reviewer": saved.get("reviewer") or "",
            "reviewer_source": saved.get("reviewer_source") or "",
            "method": saved.get("method") or "",
            "batch_id": saved.get("batch_id") or "",
            "git_commit": saved.get("git_commit") or "",
            "git_dirty": saved.get("git_dirty"),
            "subclaims": list(saved.get("subclaims") or []),
            "subclaims_sha256": _subclaims_sha256(saved.get("subclaims") or []),
            "open_subclaims": [sub for sub in (saved.get("subclaims") or []) if str(sub.get("status") or "") in ("pending", "info")],
            "backend": saved.get("backend") or record.get("backend") or "",
            "target_id": saved.get("target_id") or record.get("target_id") or "",
            "history": list(saved.get("history") or []),
        })
    return rows


def _feedback_revision(conn) -> int:
    """@brief 取当前的反馈 revision。

    @param conn 索引数据库连接。
    @return revision 整数；表里没有时返回 0。
    """
    row = conn.execute("SELECT value FROM meta WHERE key = 'feedback_revision'").fetchone()
    return int(row[0]) if row else 0


def _take_feedback_rev(conn) -> int:
    """@brief 为一批 claim 编辑取一个新 revision；未变化的 claim 保留原有 revision。

    原始英文说明：One step for a batch of claim edits. Unchanged claims keep the revision they already have.

    @param conn 索引数据库连接。
    @return 新的反馈 revision。
    """
    nxt = _feedback_revision(conn) + 1
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('feedback_revision', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(nxt),),
    )
    return nxt


def _reviewed_key(value) -> str | None:
    """@brief 把评审时间归一成用于比对的键。

    @param value 评审时间或任意值。
    @return 三位小数的字符串；None 或空串时返回 None。
    """
    if value is None or value == "":
        return None
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _subclaims_sha256(subclaims) -> str:
    """@brief 计算子 claim 列表的稳定摘要。

    @param subclaims 子 claim 列表，可为 None。
    @return 十六进制 sha256 文本。
    """
    return hashlib.sha256(
        json.dumps(subclaims or [], sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _claim_signature(row) -> tuple:
    """@brief 取 claim 行的比对签名，用于判断内容是否变化。

    @param row 数据库行或同构字典。
    @return 签名元组。
    """
    subclaims_sha256 = row["subclaims_sha256"] if "subclaims_sha256" in row.keys() else ""
    return (
        row["section"] or "",
        row["audience"] or "",
        row["premise"] or "",
        row["status"] or "",
        row["verification"] or "",
        row["topic"] or "",
        row["change_ref"] or "",
        _reviewed_key(row["reviewed_at"]),
        subclaims_sha256 or "",
    )


def _item_signature(row) -> tuple:
    """@brief 取反馈条目行的比对签名，用于判断整条记录是否变化。

    @param row 数据库行或同构字典。
    @return 签名元组。
    """
    return (
        row["seq"],
        _reviewed_key(row["t"]),
        row["task"] or "",
        row["dtype"] or "",
        row["size"] or "",
        row["label"] or "",
        row["committee"] or "",
        row["provider"] or "",
        row["model"] or "",
        row["outcome"] or "",
        row["exit"] or "",
        row["kinds"] or "",
        int(row["has_system"] or 0),
        int(row["pending_claims"] or 0),
        int(row["unverified_premises"] or 0),
        int(row["placeholder_items"] or 0),
    )


def retire_run_claims(conn, run_id: str) -> bool:
    """@brief 删除一个运行的全部 claim，并逐条留墓石，让已打开的评审页能撤掉卡片。

    原始英文说明：Forget one run's claims and leave a tombstone for each, so an open review page can drop those cards.

    @param conn 索引数据库连接。
    @param run_id 运行 id。
    @return 确实删除过 claim 时为 True；本来就没有 claim 时返回 False。
    """
    rows = list(conn.execute(
        "SELECT seq, digest, section, audience, premise, status FROM feedback_claims WHERE run_id = ?",
        (run_id,),
    ))
    if not rows:
        conn.execute("DELETE FROM feedback_items WHERE run_id = ?", (run_id,))
        return False
    rev = _take_feedback_rev(conn)
    for row in rows:
        conn.execute(
            """
            INSERT INTO feedback_claim_removed (run_id, seq, digest, rev, section, audience, premise, status, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, seq, digest) DO UPDATE SET
                rev = excluded.rev, section = excluded.section, audience = excluded.audience,
                premise = excluded.premise, status = excluded.status, reason = excluded.reason
            """,
            (run_id, row["seq"], row["digest"], rev, row["section"], row["audience"], row["premise"], row["status"], "run retired"),
        )
    conn.execute("DELETE FROM feedback_claims WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM feedback_subclaims WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM feedback_items WHERE run_id = ?", (run_id,))
    return True


def _file_fp(path: Path) -> str:
    """@brief 取文件的指纹（mtime 与大小）。

    @param path 文件路径。
    @return 指纹字符串；读不到时为空串。
    """
    try:
        stat = path.stat()
    except OSError:
        return ""
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _feedback_cursor(conn, run_id: str) -> tuple[int, str, str] | None:
    """@brief 取一个运行已读到的反馈字节偏移、评审文件指纹与解析器版本。

    @param conn 索引数据库连接。
    @param run_id 运行 id。
    @return 三元组（偏移, 评审指纹, 解析器版本）；索引里没有该运行时返回 None。
    """
    row = conn.execute(
        "SELECT feedback_bytes, feedback_review_fp, feedback_parser FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    return int(row["feedback_bytes"] or 0), str(row["feedback_review_fp"] or ""), str(row["feedback_parser"] or "")


def _store_feedback_cursor(conn, run_id: str, offset: int, review_fp: str) -> None:
    """@brief 记录已读到的反馈字节偏移、评审文件指纹与当前解析器版本。

    @param conn 索引数据库连接。
    @param run_id 运行 id。
    @param offset 已读字节数。
    @param review_fp 评审文件指纹。
    """
    conn.execute(
        "UPDATE runs SET feedback_bytes = ?, feedback_review_fp = ?, feedback_parser = ? WHERE run_id = ?",
        (offset, review_fp, PARSER_VERSION, run_id),
    )


def _rows_for_record(record: dict, review: dict, line_at: int | None = None) -> tuple[dict, list[dict]] | None:
    """@brief 把一条反馈记录转成条目行与 claim 行。

    @param record 反馈记录。
    @param review 评审文件的 ``claims`` 映射。
    @param line_at 该记录在 ``feedback.jsonl`` 中的字节偏移，可为 None。
    @return 二元组（条目行, claim 行列表）；序号非法或记录无内容时返回 None。
    """
    seq = record.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or not _has_text(record):
        return None
    built = _claim_rows(record, review)
    pending = sum(1 for claim in built if claim["status"] == "pending")
    unverified = sum(1 for claim in built if claim["status"] == "pending" and claim["premise"] in ("unverified", "downgraded"))
    item = {
        "seq": seq,
        "t": record.get("t"),
        "task": record.get("task") or "",
        "dtype": record.get("dtype") or "",
        "size": record.get("size") or "",
        "label": record.get("label") or "",
        "committee": record.get("committee") or "",
        "provider": record.get("provider") or "",
        "model": record.get("model") or "",
        "outcome": record.get("outcome") or "",
        "exit": record.get("exit") or "",
        "kinds": record.get("kinds") or "",
        "line_at": line_at,
        "has_system": 1 if _has_system(record) else 0,
        "pending_claims": pending,
        "unverified_premises": unverified,
        "placeholder_items": int(record.get("placeholder_items") or 0),
    }
    return item, built


def _write_subclaims(conn, run_id: str, claim: dict, rev: int) -> None:
    """@brief 重写一个 claim 的子 claim 行。

    @param conn 索引数据库连接。
    @param run_id 运行 id。
    @param claim claim 字典，含 ``seq``、``digest`` 与 ``subclaims``。
    @param rev 写入的反馈 revision。
    """
    conn.execute(
        "DELETE FROM feedback_subclaims WHERE run_id = ? AND seq = ? AND parent_digest = ?",
        (run_id, claim["seq"], claim["digest"]),
    )
    for index, sub in enumerate(claim.get("subclaims") or []):
        if not isinstance(sub, dict):
            continue
        text = str(sub.get("text") or "")
        conn.execute(
            """
            INSERT INTO feedback_subclaims (
                run_id, seq, parent_digest, sub_index, text, text_sha256, digest,
                status, verification, evidence_source, evidence, note, change_ref,
                backend, target_id, derived_from, rev
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id, claim["seq"], claim["digest"], index, text,
                hashlib.sha256(text.encode("utf-8")).hexdigest(), str(sub.get("digest") or ""),
                str(sub.get("status") or ""), str(sub.get("verification") or ""),
                str(sub.get("evidence_source") or ""), str(sub.get("evidence") or ""),
                str(sub.get("note") or ""), str(sub.get("change_ref") or ""),
                str(sub.get("backend") or ""), str(sub.get("target_id") or ""),
                str(sub.get("derived_from") or ""), rev,
            ),
        )


def _write_claim(conn, run_id: str, claim: dict, rev: int) -> None:
    """@brief 插入或更新一条 claim，并同步其子 claim。

    @param conn 索引数据库连接。
    @param run_id 运行 id。
    @param claim claim 字典。
    @param rev 写入的反馈 revision。
    """
    conn.execute("DELETE FROM feedback_claim_removed WHERE run_id = ? AND seq = ? AND digest = ?", (run_id, claim["seq"], claim["digest"]))
    conn.execute(
        """
        INSERT INTO feedback_claims (
            run_id, seq, digest, section, audience, premise, status, verification, topic, change_ref, reviewed_at, rev, subclaims_sha256
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(run_id, seq, digest) DO UPDATE SET
            section = excluded.section, audience = excluded.audience, premise = excluded.premise,
            status = excluded.status, verification = excluded.verification, topic = excluded.topic,
            change_ref = excluded.change_ref, reviewed_at = excluded.reviewed_at, rev = excluded.rev,
            subclaims_sha256 = excluded.subclaims_sha256
        """,
        (
            run_id, claim["seq"], claim["digest"], claim["section"], claim["audience"], claim["premise"],
            claim["status"], claim["verification"], claim["topic"], claim["change_ref"], claim["reviewed_at"], rev,
            _subclaims_sha256(claim.get("subclaims") or []),
        ),
    )
    _write_subclaims(conn, run_id, claim, rev)


def _write_item(conn, run_id: str, row: dict) -> None:
    """@brief 插入一条反馈条目行。

    @param conn 索引数据库连接。
    @param run_id 运行 id。
    @param row 条目行字典。
    """
    conn.execute(
        """
        INSERT INTO feedback_items (
            run_id, seq, t, task, dtype, size, label, committee, provider, model, outcome, exit, kinds,
            line_at, has_system, pending_claims, unverified_premises, placeholder_items
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run_id, row["seq"], row["t"], row["task"], row["dtype"], row["size"], row["label"], row["committee"],
            row["provider"], row["model"], row["outcome"], row["exit"], row["kinds"], row.get("line_at"),
            row["has_system"], row["pending_claims"], row["unverified_premises"], row["placeholder_items"],
        ),
    )


def _proposition(text: str) -> str:
    """@brief 取方括号里的命题槽，并归一化以便与旧条目匹配。

    原始英文说明：The proposition slot of a bracket, normalized for a source/old-item match.

    @param text 条目文本。
    @return 归一化后的命题；没有命题槽时为空串。
    """
    import re

    match = re.search(r"\[[^\]]*\|\s*([^|\]]+?)\s*\|", str(text or ""))
    return re.sub(r"\s+", " ", match.group(1).strip().lower()) if match else ""


def _reparsed_record(record: dict, directory: Path) -> dict:
    """@brief 在不改变旧 claim 摘要的前提下，升级被恢复的实测前提。

    ``feedback.jsonl`` 只追加且存的是条目；仅提升解析器版本无法知道会话之后探测已恢复，
    而用新拆分替换所有条目会改变摘要并孤立 ``feedback_review.json``。重建时只读该会话的
    ``FEEDBACK.md``，寻找旧“降级/未验证”命题的已验证版本，复制其文本与前提但保留旧摘要；
    日志本身不被重写。
    原始英文说明：Upgrade a restored measured premise without changing any old claim digest.

    @param record 待修复的反馈记录。
    @param directory 运行目录。
    @return 修复后的记录（可能是新对象）。
    """
    from taic.agent_feedback import FILE_CAP, parse_feedback, read_feedback_text

    record = _repair_record(record, directory)
    items = [dict(item) for item in (record.get("items") or []) if isinstance(item, dict)]
    work_rel = str(record.get("work_dir") or "")
    work = (directory / work_rel).resolve()
    try:
        work.relative_to(directory.resolve())
    except ValueError:
        return record
    path = work / "FEEDBACK.md"
    text = str(record.get("raw") or "")
    file_over = False
    if path.is_file():
        try:
            text, file_over = read_feedback_text(path)
        except OSError:
            text = ""
    if not text.strip():
        return record
    source = parse_feedback(text, work=work, repo=Path(__file__).resolve().parents[1], file_over=file_over or len(text) > FILE_CAP)
    if source.get("unrecognized_version"):
        return record
    verified = [
        item for item in source.get("items") or []
        if isinstance(item, dict) and item.get("premise") == "verified"
    ]
    old_premises = [item for item in items if item.get("premise") in ("downgraded", "unverified")]
    positions = {id(item): index for index, item in enumerate(items)}
    changed = False
    for item in old_premises:
        position = positions[id(item)]
        section_position = sum(
            1 for earlier in items[:position]
            if earlier.get("section") == item.get("section")
        )
        if item.get("premise") not in ("downgraded", "unverified"):
            continue
        prop = _proposition(str(item.get("text") or ""))
        same_section = [candidate for candidate in verified if candidate.get("section") == item.get("section")]
        match = None
        if prop:
            match = next((
                candidate for candidate in same_section
                if prop in _proposition(str(candidate.get("text") or ""))
            ), None)
        if match is None and not prop and section_position < len(same_section):
            # Older lines sometimes stored the wiped tag without the proposition, but kept the
            # explanation. The source's verified lines are in the same section order.
            match = same_section[section_position]
        if match is None:
            continue
        item["text"] = str(match.get("text") or item.get("text") or "")
        item["premise"] = "verified"
        changed = True
    if not changed:
        return record
    record = dict(record)
    record["items"] = items
    record["unverified_premises"] = sum(1 for item in items if item.get("premise") in ("unverified", "downgraded"))
    return record


def _dropped_placeholders(directory: Path) -> list[dict]:
    """@brief 找出当前解析器刻意从会话中删除的空占位条目（仍在只追加的原始日志里）。

    重建会在修复之前读原始日志，因此这些被审计的占位符即使从未作为活跃 claim 进入新索引，
    也能得到审计墓石。
    原始英文说明：Empty placeholder items that the current parser intentionally removed from a session.

    @param directory 运行目录。
    @return 每项含 ``seq``、``digest``、``section``、``audience`` 与 ``premise`` 的列表。
    """
    path = directory / "feedback.jsonl"
    if not path.is_file():
        return []
    rows = []
    seen = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("format") != "agent_feedback":
            continue
        seq = record.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool):
            continue
        for item in record.get("items") or []:
            if not isinstance(item, dict) or not _is_placeholder(str(item.get("text") or "")):
                continue
            digest = str(item.get("digest") or "")
            if not digest or (seq, digest) in seen:
                continue
            seen.add((seq, digest))
            rows.append({
                "seq": seq, "digest": digest,
                "section": item.get("section") or "",
                "audience": item.get("audience") or "",
                "premise": item.get("premise") or "",
            })
    return rows


def _rebuild_feedback(conn, run_id: str, directory: Path, *, reason: str = "feedback log changed") -> bool:
    """@brief 日志被替换或解析器变化时重建一个运行的反馈索引，之后交给游标增量。

    原始英文说明：The log was replaced or the parser changed. Read it once, then the cursor takes over.

    @param conn 索引数据库连接。
    @param run_id 运行 id。
    @param directory 运行目录。
    @param reason 写入墓石的变更原因。
    @return 索引发生变化时为 True。
    """
    review = _load_review(directory / "feedback_review.json")["claims"]
    items = []
    claims = []
    located, _end = read_appended_records(directory / "feedback.jsonl", 0)
    for line_at, record in located:
        record = _reparsed_record(record, directory)
        built = _rows_for_record(record, review, line_at)
        if built is None:
            continue
        item, claim_rows = built
        items.append(item)
        claims.extend(claim_rows)
    previous = {
        (int(row["seq"]), str(row["digest"])): dict(row)
        for row in conn.execute("SELECT * FROM feedback_claims WHERE run_id = ?", (run_id,))
    }
    current = {(int(claim["seq"]), str(claim["digest"])): claim for claim in claims}
    removed = [key for key in previous if key not in current]
    changed = [claim for key, claim in current.items() if key not in previous or _claim_signature(previous[key]) != _claim_signature(claim)]
    stored_items = {
        _item_signature(row)
        for row in conn.execute("SELECT * FROM feedback_items WHERE run_id = ?", (run_id,))
    }
    items_differ = stored_items != {_item_signature(row) for row in items}
    dropped = [
        row for row in _dropped_placeholders(directory)
        if (row["seq"], row["digest"]) not in current
    ]
    if not removed and not changed and not items_differ and not dropped:
        return False
    rev = _take_feedback_rev(conn) if (removed or changed or dropped) else None
    if items_differ or removed or changed:
        conn.execute("DELETE FROM feedback_items WHERE run_id = ?", (run_id,))
        for row in items:
            _write_item(conn, run_id, row)
    for seq, digest in removed:
        old = previous.get((seq, digest), {})
        saved = review.get(_review_key(seq, digest)) if isinstance(review.get(_review_key(seq, digest)), dict) else {}
        status = saved.get("status") or old.get("status") or "removed"
        conn.execute("DELETE FROM feedback_claims WHERE run_id = ? AND seq = ? AND digest = ?", (run_id, seq, digest))
        conn.execute(
            """
            INSERT INTO feedback_claim_removed (run_id, seq, digest, rev, section, audience, premise, status, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, seq, digest) DO UPDATE SET
                rev = excluded.rev, section = excluded.section, audience = excluded.audience,
                premise = excluded.premise, status = excluded.status, reason = excluded.reason
            """,
            (run_id, seq, digest, rev, old.get("section"), old.get("audience"), old.get("premise"), status, reason),
        )
    for claim in changed:
        _write_claim(conn, run_id, claim, rev)
    for row in dropped:
        saved = review.get(_review_key(row["seq"], row["digest"]))
        status = saved.get("status") if isinstance(saved, dict) else "obsolete"
        conn.execute(
            """
            INSERT INTO feedback_claim_removed (run_id, seq, digest, rev, section, audience, premise, status, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, seq, digest) DO UPDATE SET
                rev = excluded.rev, section = excluded.section, audience = excluded.audience,
                premise = excluded.premise, status = excluded.status, reason = excluded.reason
            """,
            (run_id, row["seq"], row["digest"], rev, row["section"], row["audience"], row["premise"], status or "obsolete", f"{PARSER_VERSION} empty placeholder"),
        )
    return True


def _append_feedback(conn, run_id: str, records: list[tuple[int, dict]], directory: Path) -> bool:
    """@brief 插入索引里还没有的会话；已有的序号保持不变。

    原始英文说明：Insert sessions that are not in the index yet. Existing sequence numbers stay untouched.

    @param conn 索引数据库连接。
    @param run_id 运行 id。
    @param records 新增的（字节偏移, 记录）列表。
    @param directory 运行目录。
    @return 写入过内容时为 True。
    """
    have = {
        int(row["seq"])
        for row in conn.execute("SELECT seq FROM feedback_items WHERE run_id = ?", (run_id,))
    }
    review = _load_review(directory / "feedback_review.json")["claims"]
    fresh_items = []
    fresh_claims = []
    for line_at, record in records:
        built = _rows_for_record(record, review, line_at)
        if built is None or built[0]["seq"] in have:
            continue
        item, claim_rows = built
        fresh_items.append(item)
        fresh_claims.extend(claim_rows)
        have.add(item["seq"])
    if not fresh_claims and not fresh_items:
        return False
    rev = _take_feedback_rev(conn)
    for row in fresh_items:
        _write_item(conn, run_id, row)
    for claim in fresh_claims:
        _write_claim(conn, run_id, claim, rev)
    return True


def _desired_review(row, saved: dict | None) -> dict:
    """@brief 由索引行与已存评审合成期望的 claim 状态。

    @param row 现有 claim 行。
    @param saved 评审文件中保存的条目，可为 None。
    @return 期望写回的 claim 字典。
    """
    base = {
        "seq": int(row["seq"]),
        "digest": str(row["digest"]),
        "section": row["section"] or "",
        "audience": row["audience"] or "",
        "premise": row["premise"] or "",
    }
    if not saved:
        base["status"] = _default_status(base)
        base["verification"] = "unchecked"
        base["topic"] = ""
        base["change_ref"] = ""
        base["reviewed_at"] = None
        base["subclaims"] = []
        base["subclaims_sha256"] = _subclaims_sha256([])
        return base
    status = saved.get("status") or _default_status(base)
    if status not in _STATUSES:
        status = _default_status(base)
    base["status"] = status
    base["verification"] = saved.get("verification") or "unchecked"
    base["topic"] = saved.get("topic") or ""
    base["change_ref"] = saved.get("change_ref") or ""
    base["reviewed_at"] = saved.get("t") or saved.get("reviewed_at")
    base["subclaims"] = list(saved.get("subclaims") or [])
    base["subclaims_sha256"] = _subclaims_sha256(base["subclaims"])
    return base


def _apply_review_file(conn, run_id: str, directory: Path) -> bool:
    """@brief 只根据 ``feedback_review.json`` 更新已存 claim，不重读日志。

    原始英文说明：Update stored claims from ``feedback_review.json`` without reading the log.

    @param conn 索引数据库连接。
    @param run_id 运行 id。
    @param directory 运行目录。
    @return 有 claim 被更新时为 True。
    """
    saved = _load_review(directory / "feedback_review.json")["claims"]
    rows = list(conn.execute("SELECT * FROM feedback_claims WHERE run_id = ?", (run_id,)))
    changed = []
    for row in rows:
        key = f"{int(row['seq'])}:{row['digest']}"
        review = saved.get(key)
        desired = _desired_review(row, review if isinstance(review, dict) else None)
        if _claim_signature(row) != _claim_signature(desired):
            changed.append(desired)
    if not changed:
        return False
    rev = _take_feedback_rev(conn)
    for claim in changed:
        _write_claim(conn, run_id, claim, rev)
    return True


def sync_run_index(conn, run_id: str, directory: Path) -> bool:
    """@brief 用一个运行的追加日志与评审文件把索引补齐。

    只增长的日志从已存字节偏移继续读；评审文件单独应用，因此保存评审不会重新解析日志。日志变短或解析器
    换了则整份读一次，之后仍回到偏移。未变化的 claim 保留原 revision。
    原始英文说明：Catch up one run from the append-only log and the review file.

    @param conn 索引数据库连接。
    @param run_id 运行 id。
    @param directory 运行目录。
    @return 索引发生变化时为 True；索引里没有该运行时返回 False。
    """
    cursor = _feedback_cursor(conn, run_id)
    if cursor is None:
        return False
    offset, stored_review, stored_parser = cursor
    path = directory / "feedback.jsonl"
    review_path = directory / "feedback_review.json"
    try:
        size = path.stat().st_size if path.is_file() else 0
    except OSError:
        size = 0
    review_fp = _file_fp(review_path)
    if stored_parser != PARSER_VERSION or size < offset:
        reason = "parser changed" if stored_parser != PARSER_VERSION else "feedback log shortened"
        changed = _rebuild_feedback(conn, run_id, directory, reason=reason)
        _store_feedback_cursor(conn, run_id, size, review_fp)
        return changed
    changed = False
    if size > offset:
        records, new_offset = read_appended_records(path, offset)
        if _append_feedback(conn, run_id, records, directory):
            changed = True
        offset = new_offset
    if review_fp != stored_review:
        if _apply_review_file(conn, run_id, directory):
            changed = True
    if changed or offset != cursor[0] or review_fp != stored_review:
        _store_feedback_cursor(conn, run_id, offset, review_fp)
    return changed


def _candidate_report(directory: Path, label: str) -> tuple[dict | None, str]:
    """@brief 找到某个标签对应的候选报告。

    @param directory 运行目录。
    @param label 候选标签。
    @return 二元组（schema-4 报告字典或 None, 报告路径文本）。
    """
    root = directory / "candidates"
    if not label or not root.is_dir():
        return None, ""
    matches = [path for path in root.iterdir() if path.is_dir() and path.name.endswith("_" + label)]
    if not matches:
        return None, ""
    path = sorted(matches)[-1] / "report.json"
    if not path.is_file():
        return None, ""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, ""
    if not isinstance(doc, dict):
        return None, ""
    from taic.artifacts import document_state

    # Type-strict currentness (``document_state``): ``4.0 == 4`` in Python, so a plain ``!=`` would
    # accept a float version.  An older report is not current evidence: no build identity, no
    # comparable statistics.  The review row keeps the path it consulted but quotes no number.
    if document_state(doc) != "current":
        return None, str(path)
    return doc, str(path)


def _report_brief(doc: dict | None, source: str) -> dict | None:
    """@brief 从报告里抽出评审能直接引用的几个数（耗时、离散度与 roofline）。

    @param doc 报告字典，可为 None。
    @param source 报告来源（candidate 或 seed）。
    @return 摘要字典；没有报告时返回 None。
    """
    if not doc:
        return None
    benchmark = doc.get("benchmark") if isinstance(doc.get("benchmark"), dict) else {}
    candidate = benchmark.get("candidate") if isinstance(benchmark.get("candidate"), dict) else {}
    roof = doc.get("roofline") if isinstance(doc.get("roofline"), dict) else {}
    return {
        "source": source,
        "status": doc.get("status"),
        "candidate_us": candidate.get("median_us", doc.get("candidate_us")),
        "cv": candidate.get("cv"),
        "roofline": {key: roof.get(key) for key in ("lower_bound_us", "bound_kind", "time_over_lower_bound") if key in roof},
    }


def _calibration() -> dict | None:
    """@brief 读取本机标定的峰值算力与带宽。

    @return 含各精度峰值与带宽的字典；标定读不到时返回 None。
    """
    from tcenv.target import TargetError, read_calibration

    try:
        doc, _path, _source = read_calibration()
    except (OSError, json.JSONDecodeError, TargetError):
        return None
    if not isinstance(doc, dict):
        return None
    peaks = doc.get("peak_tflops") if isinstance(doc.get("peak_tflops"), dict) else {}
    return {
        "fp16": peaks.get("fp16"),
        "bf16": peaks.get("bf16"),
        "fp32": peaks.get("fp32"),
        "read_gbps": doc.get("read_gbps"),
        "copy_gbps": doc.get("copy_gbps"),
    }


def _has_text(record: dict) -> bool:
    """@brief 判断会话是否真有反馈；没有反馈的会话只占一个序号，是时钟刻度而不是评审项。

    原始英文说明：A session with no feedback still occupies a sequence number. That row is a clock tick, not a review item.

    @param record 反馈记录。
    @return 有原文或有条目时为 True。
    """
    if str(record.get("raw") or "").strip():
        return True
    return bool(record.get("items"))


def entries(directory: Path, *, with_calibration: bool = True) -> dict:
    """@brief 返回一个运行的完整反馈，带评审状态和评审者可粘贴的数字。

    ``with_calibration=False`` 时不读本机标定文件：紧凑列表不展示它，读它只会白白多开一个文件。

    原始英文说明：One run's feedback, with the review state and the numbers a reviewer can paste.

    @param directory 运行目录。
    @param with_calibration 是否附带本机标定。
    @return 含 ``records`` 与 ``calibration`` 的字典。
    """
    review = _load_review(directory / "feedback_review.json")["claims"]
    records = []
    for record in load_records(directory / "feedback.jsonl"):
        if not _has_text(record):
            continue
        report, report_path = _candidate_report(directory, str(record.get("label") or ""))
        source = "candidate"
        if report is None:
            report, report_path = _candidate_report(directory, str(record.get("seed_label") or ""))
            source = "seed" if report is not None else ""
        records.append({
            "seq": record.get("seq"),
            "t": record.get("t"),
            "label": record.get("label"),
            "round": record.get("round"),
            "committee": record.get("committee"),
            "provider": record.get("provider"),
            "model": record.get("model"),
            "effort": record.get("effort"),
            "outcome": record.get("outcome"),
            "exit": record.get("exit"),
            "best_speedup": record.get("best_speedup"),
            "seed_label": record.get("seed_label"),
            "kinds": record.get("kinds"),
            "work_dir": record.get("work_dir"),
            "backend": record.get("backend") or "",
            "target_id": record.get("target_id") or "",
            "truncated": record.get("truncated"),
            "truncated_sections": list(record.get("truncated_sections") or []),
            "raw": record.get("raw") or "",
            "claims": _claim_rows(record, review),
            "report": _report_brief(report, source),
            "report_path": report_path,
        })
    return {"records": records, "calibration": _calibration() if with_calibration else None}


def summary_from_index(conn, *, task: str | None = None, model: str | None = None, kind: str | None = None, status: str | None = None) -> dict:
    """@brief 返回计数与被筛选的 claim 列表；``pending_total`` 是全部待评 claim，不是筛选后的切片。

    原始英文说明：Counts and the filtered claim index.

    @param conn 索引数据库连接。
    @param task 只看该任务，None 表示不限。
    @param model 只看该模型，None 表示不限。
    @param kind 只看该 kinds 标签，None 表示不限。
    @param status 只看该状态，None 表示不限。
    @return 含待评总数、子 claim 数、高频项、复现项与 ``items`` 的字典。
    """
    pending_total = int(conn.execute("SELECT COUNT(*) FROM feedback_claims WHERE status = 'pending'").fetchone()[0])
    where = ["1 = 1"]
    args: list[object] = []
    if task:
        where.append("i.task = ?")
        args.append(task)
    if model:
        where.append("i.model = ?")
        args.append(model)
    if kind:
        where.append("(',' || i.kinds || ',') LIKE ?")
        args.append(f"%,{kind},%")
    if status:
        where.append("c.status = ?")
        args.append(status)
    items = [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT c.run_id, c.seq, c.digest, c.section, c.audience, c.premise, c.status, c.verification, c.topic,
                   c.change_ref, c.reviewed_at, i.task, i.model, i.label, i.kinds, i.t, r.target_id
            FROM feedback_claims c
            JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
            LEFT JOIN runs r ON r.run_id = c.run_id
            WHERE {' AND '.join(where)}
            ORDER BY i.t DESC, c.seq DESC
            """,
            args,
        )
    ]
    frequent_rows = conn.execute(
        """
        SELECT COALESCE(NULLIF(c.topic, ''), c.digest) AS claim_key,
               COUNT(DISTINCT c.run_id) AS runs,
               GROUP_CONCAT(DISTINCT i.model) AS models
        FROM feedback_claims c
        JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
        GROUP BY claim_key
        HAVING runs >= 1
        ORDER BY runs DESC, claim_key
        LIMIT 20
        """
    )
    frequent = [{"key": row["claim_key"], "runs": row["runs"], "models": row["models"] or ""} for row in frequent_rows]
    done = list(conn.execute(
        """
        SELECT c.run_id, c.digest, c.topic, c.reviewed_at, r.started_at
        FROM feedback_claims c
        JOIN runs r ON r.run_id = c.run_id
        WHERE c.status = 'done' AND c.reviewed_at IS NOT NULL
        """
    ))
    later = list(conn.execute(
        """
        SELECT c.run_id, c.digest, c.topic, i.t, r.started_at
        FROM feedback_claims c
        JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
        JOIN runs r ON r.run_id = c.run_id
        """
    ))
    recurrence = []
    for finished in done:
        key = finished["topic"] or finished["digest"]
        if not key:
            continue
        for other in later:
            if other["run_id"] == finished["run_id"]:
                continue
            same = (finished["topic"] and other["topic"] == finished["topic"]) or other["digest"] == finished["digest"]
            if not same:
                continue
            if other["started_at"] is None or finished["reviewed_at"] is None:
                continue
            if float(other["started_at"]) <= float(finished["reviewed_at"]):
                continue
            if other["t"] is None or float(other["t"]) <= float(finished["reviewed_at"]):
                continue
            recurrence.append({"key": key, "run_id": other["run_id"], "after": finished["run_id"]})
    open_subclaims_total = int(conn.execute(
        "SELECT COUNT(*) FROM feedback_subclaims WHERE status IN ('pending', 'info')"
    ).fetchone()[0])
    return {
        "pending_total": pending_total,
        "open_subclaims_total": open_subclaims_total,
        "open_work_total": pending_total + open_subclaims_total,
        "frequent": frequent,
        "recurrence": recurrence,
        "items": items,
    }


def prior_claims(conn, *, digest: str, task: str, target_id: str, text: str = "", before_t: float | None = None, exclude_run: str = "", exclude_seq: int | None = None, limit: int = 5) -> list[dict]:
    """@brief 找出同一任务、同一 target 下同摘要的更早 claim。

    这是精确文本查找，不是语义相似度。范围刻意收窄：不同任务或后端不构成证据，改写过的 claim 会得到新摘要。
    返回的 note/evidence 来自更早的评审文件；状态与 change_ref 仍以索引为准。
    原始英文说明：Earlier claims with the same digest in the same task and target.

    @param conn 索引数据库连接。
    @param digest 目标摘要。
    @param task 任务名。
    @param target_id 目标硬件 id。
    @param text 条目全文，给出时也会匹配摘录相同的子 claim。
    @param before_t 只要更早的记录，可为 None。
    @param exclude_run 排除的运行 id。
    @param exclude_seq 排除的序号。
    @param limit 最多返回条数。
    @return 带文本、note、evidence 等补齐字段的 claim 行列表。
    """
    if not digest or not task or not target_id:
        return []
    query = """
        SELECT c.run_id, c.seq, c.digest, c.status, c.verification, c.topic,
               c.change_ref, c.reviewed_at, i.task, i.model, i.t, r.target_id,
               NULL AS sub_index, c.digest AS parent_digest
        FROM feedback_claims c
        JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
        LEFT JOIN runs r ON r.run_id = c.run_id
        WHERE c.digest = ? AND i.task = ? AND r.target_id = ?
    """
    params: list[object] = [digest, task, target_id]
    if before_t is not None:
        query += " AND i.t < ?"
        params.append(float(before_t))
    query += " ORDER BY COALESCE(c.reviewed_at, i.t) DESC, i.t DESC"
    rows = []
    for row in conn.execute(query, params):
        item = dict(row)
        if item["run_id"] == exclude_run and item["seq"] == exclude_seq:
            continue
        rows.append(item)
        if len(rows) >= max(1, int(limit)):
            break
    if text:
        text_sha = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
        subquery = """
            SELECT s.run_id, s.seq, s.parent_digest AS digest, s.status, s.verification, NULL AS topic,
                   s.change_ref, NULL AS reviewed_at, i.task, i.model, i.t, r.target_id,
                   s.sub_index, s.parent_digest
            FROM feedback_subclaims s
            JOIN feedback_claims c ON c.run_id = s.run_id AND c.seq = s.seq AND c.digest = s.parent_digest
            JOIN feedback_items i ON i.run_id = s.run_id AND i.seq = s.seq
            LEFT JOIN runs r ON r.run_id = s.run_id
            WHERE s.text_sha256 = ? AND i.task = ? AND r.target_id = ?
        """
        subparams: list[object] = [text_sha, task, target_id]
        if before_t is not None:
            subquery += " AND i.t < ?"
            subparams.append(float(before_t))
        subquery += " ORDER BY i.t DESC"
        for row in conn.execute(subquery, subparams):
            item = dict(row)
            if item["run_id"] == exclude_run and item["seq"] == exclude_seq:
                continue
            rows.append(item)
            if len(rows) >= max(1, int(limit)):
                break
    if not rows:
        return []
    from .detail import resolve_run

    cache: dict[str, dict] = {}
    for item in rows:
        run_id = str(item["run_id"])
        if run_id not in cache:
            directory = resolve_run(run_id)
            cache[run_id] = _load_review(directory / "feedback_review.json")["claims"] if directory is not None else {}
        saved = cache[run_id].get(f"{item['seq']}:{item['digest']}") or {}
        sub_index = item.get("sub_index")
        sub = None
        if isinstance(sub_index, int) and 0 <= sub_index < len(saved.get("subclaims") or []):
            sub = (saved.get("subclaims") or [])[sub_index]
        source = sub if isinstance(sub, dict) else saved
        item["text"] = str(source.get("text") or item.get("text") or "")
        item["note"] = str(source.get("note") or "")
        item["evidence"] = str(source.get("evidence") or "")
        item["evidence_source"] = str(source.get("evidence_source") or "")
        item["reviewer"] = str(saved.get("reviewer") or "")
        item["method"] = str(saved.get("method") or "")
        item["subclaims"] = list(saved.get("subclaims") or [])
    return rows


def _prior_summary(rows: list[dict]) -> str:
    """@brief 把更早 claim 的状态汇总成一句中文摘要。

    @param rows 更早 claim 行列表。
    @return 形如“此前 adopted×2、rejected”的文本；没有行时为空串。
    """
    if not rows:
        return ""
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    parts = [f"{name}×{count}" if count > 1 else name for name, count in sorted(counts.items())]
    return "此前 " + "、".join(parts)


def attach_prior_claims(conn, rows: list[dict]) -> None:
    """@brief 给评审行挂上同摘要的更早结论；就地修改这些行。

    原始英文说明：Attach exact-digest earlier verdicts to review rows. Mutates the rows in place.

    @param conn 索引数据库连接。
    @param rows 评审行列表（就地修改）。
    """
    cache: dict[tuple[str, str, str, str, float | None], list[dict]] = {}
    for row in rows:
        digest = str(row.get("digest") or "")
        task = str(row.get("task") or "")
        target_id = str(row.get("target_id") or "")
        if not digest or not task or not target_id:
            row["prior"] = []
            row["prior_summary"] = ""
            continue
        current_t = row.get("t")
        texts = [str(row.get("text") or "")]
        texts.extend(str(sub.get("text") or "") for sub in row.get("subclaims") or [] if isinstance(sub, dict))
        merged: dict[tuple, dict] = {}
        for text in dict.fromkeys(texts):
            key = (digest, task, target_id, text, current_t)
            if key not in cache:
                cache[key] = prior_claims(
                    conn, digest=digest, task=task, target_id=target_id, text=text,
                    before_t=current_t, limit=6,
                )
            for item in cache[key]:
                if item.get("run_id") == row.get("run_id") and item.get("seq") == row.get("seq"):
                    continue
                identity = (item.get("run_id"), item.get("seq"), item.get("digest"), item.get("sub_index"))
                merged.setdefault(identity, item)
        prior = sorted(
            merged.values(),
            key=lambda item: (float(item.get("reviewed_at") or item.get("t") or 0), str(item.get("run_id") or "")),
            reverse=True,
        )[:5]
        row["prior"] = prior
        row["prior_summary"] = _prior_summary(prior)


def _review_file_cache(run_id: str, cache: dict[str, dict]) -> dict:
    """@brief 按运行缓存其评审文件的 ``claims`` 映射。

    @param run_id 运行 id。
    @param cache 跨调用复用的缓存字典。
    @return 该运行的 ``claims`` 映射；运行解析不到时为空字典。
    """
    if run_id not in cache:
        from .detail import resolve_run

        directory = resolve_run(run_id)
        cache[run_id] = _load_review(directory / "feedback_review.json")["claims"] if directory is not None else {}
    return cache[run_id]


def _candidate_texts(conn, *, task: str, target_id: str, before_t: float | None, limit: int = 500) -> list[dict]:
    """@brief 取出可用于语义比较的候选 claim，并从其反馈行补齐文本。

    原始英文说明：Candidate claims for semantic comparison, with text read from their feedback lines.

    @param conn 索引数据库连接。
    @param task 任务名。
    @param target_id 目标硬件 id。
    @param before_t 只要更早的记录，可为 None。
    @param limit 最大条数。
    @return 带 ``text`` 的 claim 行列表。
    """
    query = """
        SELECT c.run_id, c.seq, c.digest, c.status, c.verification, c.change_ref,
               c.reviewed_at, i.task, i.model, i.t, r.target_id, i.line_at
        FROM feedback_claims c
        JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
        LEFT JOIN runs r ON r.run_id = c.run_id
        WHERE i.task = ? AND r.target_id = ? AND c.digest <> ''
    """
    params: list[object] = [task, target_id]
    if before_t is not None:
        query += " AND i.t < ?"
        params.append(float(before_t))
    query += " ORDER BY i.t DESC LIMIT ?"
    params.append(max(1, int(limit)))
    rows = [dict(row) for row in conn.execute(query, params)]
    subquery = """
        SELECT s.run_id, s.seq, s.parent_digest AS digest, s.status, s.verification,
               s.change_ref, NULL AS reviewed_at, i.task, i.model, i.t, r.target_id,
               s.sub_index, s.text
        FROM feedback_subclaims s
        JOIN feedback_claims c ON c.run_id = s.run_id AND c.seq = s.seq AND c.digest = s.parent_digest
        JOIN feedback_items i ON i.run_id = s.run_id AND i.seq = s.seq
        LEFT JOIN runs r ON r.run_id = s.run_id
        WHERE i.task = ? AND r.target_id = ? AND s.text <> ''
    """
    subparams: list[object] = [task, target_id]
    if before_t is not None:
        subquery += " AND i.t < ?"
        subparams.append(float(before_t))
    subquery += " ORDER BY i.t DESC LIMIT ?"
    subparams.append(max(1, int(limit)))
    rows.extend(dict(row) for row in conn.execute(subquery, subparams))
    if not rows:
        return []
    from .detail import resolve_run

    lines: dict[tuple[str, int], dict] = {}
    for row in rows:
        if row.get("text"):
            continue
        run_id = str(row["run_id"])
        seq = int(row["seq"])
        key = (run_id, seq)
        if key not in lines:
            directory = resolve_run(run_id)
            record = None
            if directory is not None and row.get("line_at") is not None:
                record = _record_at(directory / "feedback.jsonl", int(row["line_at"]))
            if record is None and directory is not None:
                for candidate in load_records(directory / "feedback.jsonl"):
                    if int(candidate.get("seq") or -1) == seq:
                        record = candidate
                        break
            lines[key] = record or {}
        record = lines[key]
        text = ""
        for item in record.get("items") or []:
            if str(item.get("digest") or "") == str(row["digest"]):
                text = str(item.get("text") or "")
                break
        if text:
            row["text"] = text
    return [row for row in rows if row.get("text")]


def similar_claims(conn, *, digest: str, task: str, target_id: str, text: str, before_t: float | None = None, exclude_run: str = "", exclude_seq: int | None = None, limit: int = 3) -> list[dict]:
    """@brief 找出同一任务、同一 target 下句向量相近的更早 claim。

    这是检索提示，不是结论。使用 Apple 本地句向量；不可用时返回 ``[]``，不会静默换成词面分数。
    原始英文说明：Earlier same-task/target claims whose local sentence embedding is close.

    @param conn 索引数据库连接。
    @param digest 目标摘要。
    @param task 任务名。
    @param target_id 目标硬件 id。
    @param text 查询文本。
    @param before_t 只要更早的记录，可为 None。
    @param exclude_run 排除的运行 id。
    @param exclude_seq 排除的序号。
    @param limit 最多返回条数。
    @return 带 ``score`` 与补齐文本的 claim 行列表。
    """
    from . import similarity, simcache

    if not text or not similarity.available():
        return []
    # 近邻只从更早的 claim 里选：语料键不变时结果不可变，直接读持久缓存，不为每次请求重算。
    corpus = _corpus_key(conn, task=task, target_id=target_id, before_t=before_t)
    cache_id = (str(exclude_run or ""), int(exclude_seq or 0), str(digest or ""))
    scored = simcache.get_similar(*cache_id, corpus)
    if scored is None:
        candidates = _candidate_texts(conn, task=task, target_id=target_id, before_t=before_t)
        candidates = [
            row for row in candidates
            if not (row.get("run_id") == exclude_run and row.get("seq") == exclude_seq)
        ]
        scored = similarity.find_similar(text, candidates, limit=limit)
        # 空结果也缓存：找不到近邻的 claim 同样不值得每次重嵌一遍候选语料。
        simcache.put_similar(*cache_id, corpus, scored)
    if not scored:
        return []
    cache: dict[str, dict] = {}
    for row in scored:
        saved = _review_file_cache(str(row["run_id"]), cache).get(f"{row['seq']}:{row['digest']}") or {}
        sub_index = row.get("sub_index")
        sub = None
        if isinstance(sub_index, int) and 0 <= sub_index < len(saved.get("subclaims") or []):
            sub = (saved.get("subclaims") or [])[sub_index]
        source = sub if isinstance(sub, dict) else saved
        row["text"] = str(source.get("text") or row.get("text") or "")
        row["note"] = str(source.get("note") or "")
        row["evidence"] = str(source.get("evidence") or "")
        row["evidence_source"] = str(source.get("evidence_source") or "")
        row["subclaims"] = list(saved.get("subclaims") or [])
        # 缓存只存检索结果；评审状态以当前评审文件为准，不能因为缓存而回到旧状态。
        for field in ("status", "verification", "change_ref"):
            if source.get(field) is not None:
                row[field] = source.get(field)
    return scored


def _corpus_key(conn, *, task: str, target_id: str, before_t: float | None) -> tuple[int, int]:
    """@brief 候选语料键：``(候选条数, 候选 rev)``，任一变化都必须让近邻缓存失效。

    @param conn 索引数据库连接。
    @param task 任务名。
    @param target_id 目标硬件 id。
    @param before_t 只要更早的记录，可为 None。
    @return 两个整数组成的键。
    """
    params: list[object] = [task, target_id]
    clause = ""
    if before_t is not None:
        clause = " AND i.t < ?"
        params.append(float(before_t))
    claims = conn.execute(
        "SELECT COUNT(*), COALESCE(MAX(c.rev), 0) FROM feedback_claims c"
        " JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq"
        " LEFT JOIN runs r ON r.run_id = c.run_id"
        " WHERE i.task = ? AND r.target_id = ? AND c.digest <> ''" + clause,
        params,
    ).fetchone()
    subs = conn.execute(
        "SELECT COUNT(*), COALESCE(MAX(s.rowid), 0) FROM feedback_subclaims s"
        " JOIN feedback_claims c ON c.run_id = s.run_id AND c.seq = s.seq AND c.digest = s.parent_digest"
        " JOIN feedback_items i ON i.run_id = s.run_id AND i.seq = s.seq"
        " LEFT JOIN runs r ON r.run_id = s.run_id"
        " WHERE i.task = ? AND r.target_id = ? AND s.text <> ''" + clause,
        params,
    ).fetchone()
    return (int(claims[0]) + int(subs[0]), int(claims[1]) + int(subs[1]))


def _warm_state(conn) -> str:
    """@brief 预热的工作发现指纹：待审集合、语料大小与评审代际的廉价摘要。

    三条不 JOIN 的聚合查询就够了：任何一条变化都可能让某些缓存失效，所以指纹变了就重新发现；
    指纹不变时预热的空闲轮只花这三条查询，不再逐条算语料键。指纹是保守近似——它多报变化
    （多扫一轮），不会漏报导致读路径拿到错数据：读路径自己仍按 ``_corpus_key`` 核新鲜度。

    @param conn 索引数据库连接。
    @return 指纹字符串。
    """
    pending = conn.execute(
        "SELECT COUNT(*), COALESCE(MAX(rev), 0), COALESCE(SUM(LENGTH(digest) + seq), 0), COALESCE(MAX(digest), '')"
        " FROM feedback_claims WHERE status = 'pending' AND digest <> ''",
    ).fetchone()
    subs = conn.execute(
        "SELECT COUNT(*), COALESCE(MAX(rowid), 0), COALESCE(SUM(LENGTH(text)), 0) FROM feedback_subclaims WHERE text <> ''",
    ).fetchone()
    corpus = conn.execute(
        "SELECT COUNT(*), COALESCE(MAX(rev), 0) FROM feedback_claims WHERE digest <> ''",
    ).fetchone()
    return "|".join(str(part) for part in (*pending, *subs, *corpus))


# 没有正文的待审 claim：一轮里跳过就别每轮重读它的文件。集合随指纹变化清空（正文可能后补）。
_TEXTLESS: set[tuple[str, int, str]] = set()
_TEXTLESS_STATE = ""

WARM_BATCH = 500


def warm_similar_once(conn=None) -> bool:
    """@brief 为一条尚未缓存近邻的待审 claim 做预热；做了工作返回 True。

    每次只推进一条，由调用方睡眠让出 GIL；没有可用句向量或没有工作时返回 False。工作发现是
    廉价的：空闲轮只发三条不 JOIN 的聚合查询，再读一次缓存键，不逐条算语料键（那是全量扫描）。

    @param conn 索引数据库连接；None 时用只读连接（不占主索引写锁，后台预热走这条）。
    @return 是否完成了一条 claim 的近邻计算。
    """
    from . import similarity, simcache

    global _TEXTLESS_STATE

    if not similarity.available():
        return False
    if conn is None:
        from observe.index import reader

        conn = reader()
    state = _warm_state(conn)
    if simcache.get_warm_state() == state:
        return False
    if state != _TEXTLESS_STATE:
        _TEXTLESS.clear()
        _TEXTLESS_STATE = state
    rows = [dict(row) for row in conn.execute(
        _CLAIM_LIST + " WHERE c.status = 'pending' AND c.digest <> '' ORDER BY i.t DESC, c.seq DESC LIMIT ?",
        (WARM_BATCH,),
    )]
    if not rows:
        simcache.set_warm_state(state)
        return False
    cached = simcache.cached_keys([str(row["run_id"]) for row in rows])
    # 语料键按（task, target, t）分组算一次：同场 run 的待审 claim 语料完全相同，不必每条都查。
    corpora: dict[tuple[str, str, float | None], tuple[int, int]] = {}

    def corpus_of(row: dict) -> tuple[int, int]:
        key = (str(row.get("task") or ""), str(row.get("target_id") or ""), row.get("t"))
        if key not in corpora:
            corpora[key] = _corpus_key(conn, task=key[0], target_id=key[1], before_t=key[2])
        return corpora[key]

    for row in rows:
        identity = (str(row["run_id"]), int(row["seq"]), str(row.get("digest") or ""))
        if identity in _TEXTLESS:
            continue
        if cached.get(identity) == corpus_of(row):
            continue
        _attach_claim_text([row])
        text = str(row.get("text") or "")
        if not text:
            _TEXTLESS.add(identity)
            continue
        similar_claims(
            conn,
            digest=identity[2],
            task=str(row.get("task") or ""),
            target_id=str(row.get("target_id") or ""),
            text=text,
            before_t=row.get("t"),
            exclude_run=identity[0],
            exclude_seq=identity[1],
        )
        return True
    # 本批里没有可做的工作：记下指纹，之后的空闲轮只花三条聚合查询；顺手把库压回上限。
    simcache.set_warm_state(state)
    simcache.prune()
    return False


def attach_similar_claims(conn, rows: list[dict], *, limit: int = 3) -> None:
    """@brief 给待评行挂上语义近邻；同摘要的精确历史仍是更廉价的一层。

    原始英文说明：Attach semantic neighbours to pending review rows. Exact prior stays the cheap layer.

    @param conn 索引数据库连接。
    @param rows 评审行列表（就地修改）。
    @param limit 每个行最多挂几条近邻。
    """
    for row in rows:
        if str(row.get("status") or "") != "pending" or not str(row.get("text") or ""):
            row["similar"] = []
            continue
        similar = similar_claims(
            conn,
            digest=str(row.get("digest") or ""),
            task=str(row.get("task") or ""),
            target_id=str(row.get("target_id") or ""),
            text=str(row.get("text") or ""),
            before_t=row.get("t"),
            exclude_run=str(row.get("run_id") or ""),
            exclude_seq=row.get("seq"),
            limit=limit,
        )
        prior_keys = {(p.get("run_id"), p.get("seq"), p.get("digest")) for p in row.get("prior", [])}
        row["similar"] = [item for item in similar if (item.get("run_id"), item.get("seq"), item.get("digest")) not in prior_keys]


def review_board(conn, *, task: str | None = None, model: str | None = None, kind: str | None = None, status: str | None = None) -> dict:
    """@brief 返回跨运行的全部评审 claim，并补齐只存在于运行目录里的文本。

    索引行只给出 claim 的名字；句子、保存的证据与候选报告都从该运行目录读取。
    原始英文说明：Every review claim across runs, with the text that lives in the run directory.

    @param conn 索引数据库连接。
    @param task 只看该任务，None 表示不限。
    @param model 只看该模型，None 表示不限。
    @param kind 只看该 kinds 标签，None 表示不限。
    @param status 只看该状态，None 表示不限。
    @return 含待评计数、标定与 ``items`` 的看板字典。
    """
    from observe.detail import resolve_run

    summary = summary_from_index(conn, task=task, model=model, kind=kind, status=status)
    grouped: dict[str, list[dict]] = {}
    for item in summary["items"]:
        grouped.setdefault(str(item["run_id"]), []).append(item)
    calibration = None
    for run_id, rows in grouped.items():
        directory = resolve_run(run_id)
        found: dict[tuple[int, str], tuple[dict, dict]] = {}
        if directory is not None:
            doc = entries(directory)
            calibration = doc.get("calibration") or calibration
            for record in doc["records"]:
                for claim in record["claims"]:
                    seq = claim.get("seq")
                    digest = str(claim.get("digest") or "")
                    if isinstance(seq, int) and digest:
                        found[(seq, digest)] = (claim, record)
        for row in rows:
            try:
                seq = int(row["seq"])
            except (TypeError, ValueError):
                seq = -1
            pair = found.get((seq, str(row.get("digest") or "")))
            claim, record = pair if pair is not None else ({}, {})
            row["text"] = claim.get("text") or ""
            row["verification"] = claim.get("verification") or row.get("verification") or "unchecked"
            row["evidence_source"] = claim.get("evidence_source") or ""
            row["evidence"] = claim.get("evidence") or ""
            row["note"] = claim.get("note") or ""
            row["reviewer"] = claim.get("reviewer") or ""
            row["reviewer_source"] = claim.get("reviewer_source") or ""
            row["method"] = claim.get("method") or ""
            row["batch_id"] = claim.get("batch_id") or ""
            row["git_commit"] = claim.get("git_commit") or ""
            row["git_dirty"] = claim.get("git_dirty")
            row["subclaims"] = list(claim.get("subclaims") or [])
            row["open_subclaims"] = [sub for sub in (claim.get("subclaims") or []) if str(sub.get("status") or "") in ("pending", "info")]
            row["backend"] = claim.get("backend") or record.get("backend") or row.get("backend") or ""
            row["target_id"] = claim.get("target_id") or record.get("target_id") or row.get("target_id") or ""
            row["topic"] = claim.get("topic") or row.get("topic") or ""
            row["change_ref"] = claim.get("change_ref") or row.get("change_ref") or ""
            row["status"] = claim.get("status") or row.get("status") or ""
            row["round"] = record.get("round")
            row["committee"] = record.get("committee") or ""
            row["report"] = record.get("report")
            row["report_path"] = record.get("report_path") or ""
    attach_prior_claims(conn, summary["items"])
    attach_similar_claims(conn, summary["items"])
    return {
        "pending_total": summary["pending_total"],
        "open_subclaims_total": summary["open_subclaims_total"],
        "open_work_total": summary["open_work_total"],
        "calibration": calibration,
        "items": summary["items"],
    }


def _record_at(path: Path, line_at: int) -> dict | None:
    """@brief 按字节偏移读一行日志并修复成记录；偏移是该行追加时记下的。

    原始英文说明：One log line. The offset was stored when that line was appended.

    @param path ``feedback.jsonl`` 路径。
    @param line_at 该行的字节偏移。
    @return 修复后的记录；读不到或格式不对时返回 None。
    """
    try:
        with path.open("rb") as handle:
            handle.seek(line_at)
            raw = handle.readline()
    except OSError:
        return None
    try:
        line = raw.decode("utf-8")
    except UnicodeDecodeError:
        line = raw.decode("utf-8", errors="replace")
    from taic.agent_feedback import FORMAT, _load_line, _repair_record

    row = _load_line(line)
    if row is None or row.get("format") != FORMAT:
        return None
    return _repair_record(row, path.parent)


def _attach_claim_text(rows: list[dict]) -> dict | None:
    """@brief 按已存行偏移补齐 claim 文本；没有偏移的行退回读取该运行的日志。

    原始英文说明：Fill claim text from the stored line offset. A row without one falls back to reading that run's log.

    @param rows claim 行列表（就地修改）。
    @return 本机标定字典，没有时为 None。
    """
    from observe.detail import resolve_run

    grouped: dict[str, list[dict]] = {}
    for item in rows:
        grouped.setdefault(str(item["run_id"]), []).append(item)
    calibration = _calibration()
    for run_id, group in grouped.items():
        directory = resolve_run(run_id)
        found: dict[tuple[int, str], tuple[dict, dict]] = {}
        if directory is not None and any(row.get("line_at") is None for row in group):
            for record in entries(directory)["records"]:
                for claim in record["claims"]:
                    seq = claim.get("seq")
                    digest = str(claim.get("digest") or "")
                    if isinstance(seq, int) and digest:
                        found[(seq, digest)] = (claim, record)
        elif directory is not None:
            path = directory / "feedback.jsonl"
            review = _load_review(directory / "feedback_review.json")["claims"]
            by_seq: dict[int, dict] = {}
            for row in group:
                try:
                    seq = int(row["seq"])
                except (TypeError, ValueError):
                    continue
                line_at = row.get("line_at")
                if not isinstance(line_at, int) or seq in by_seq:
                    continue
                record = _record_at(path, line_at)
                if record is not None:
                    by_seq[seq] = record
            for record in by_seq.values():
                for claim in _claim_rows(record, review):
                    digest = str(claim.get("digest") or "")
                    seq = claim.get("seq")
                    if isinstance(seq, int) and digest:
                        found[(seq, digest)] = (claim, record)
        for row in group:
            try:
                seq = int(row["seq"])
            except (TypeError, ValueError):
                seq = -1
            pair = found.get((seq, str(row.get("digest") or "")))
            claim, record = pair if pair is not None else ({}, {})
            row["text"] = claim.get("text") or ""
            row["verification"] = claim.get("verification") or row.get("verification") or "unchecked"
            row["evidence_source"] = claim.get("evidence_source") or ""
            row["evidence"] = claim.get("evidence") or ""
            row["note"] = claim.get("note") or ""
            row["reviewer"] = claim.get("reviewer") or ""
            row["reviewer_source"] = claim.get("reviewer_source") or ""
            row["method"] = claim.get("method") or ""
            row["batch_id"] = claim.get("batch_id") or ""
            row["git_commit"] = claim.get("git_commit") or ""
            row["git_dirty"] = claim.get("git_dirty")
            row["subclaims"] = list(claim.get("subclaims") or [])
            row["open_subclaims"] = [sub for sub in (claim.get("subclaims") or []) if str(sub.get("status") or "") in ("pending", "info")]
            row["backend"] = claim.get("backend") or record.get("backend") or row.get("backend") or ""
            row["target_id"] = claim.get("target_id") or record.get("target_id") or row.get("target_id") or ""
            row["topic"] = claim.get("topic") or row.get("topic") or ""
            row["change_ref"] = claim.get("change_ref") or row.get("change_ref") or ""
            row["status"] = claim.get("status") or row.get("status") or ""
            row["round"] = record.get("round") if record else row.get("round")
            row["committee"] = (record.get("committee") if record else None) or row.get("committee") or ""
            if record:
                report, report_path = _candidate_report(directory, str(record.get("label") or "")) if directory is not None else (None, "")
                source = "candidate"
                if report is None and directory is not None:
                    report, report_path = _candidate_report(directory, str(record.get("seed_label") or ""))
                    source = "seed" if report is not None else ""
                row["report"] = _report_brief(report, source)
                row["report_path"] = report_path
            else:
                row["report"] = None
                row["report_path"] = ""
    return calibration


_CLAIM_LIST = """
    SELECT c.run_id, c.seq, c.digest, c.section, c.audience, c.premise, c.status, c.verification, c.topic,
           c.change_ref, c.reviewed_at, c.rev, i.task, i.dtype, i.size, i.model, i.label, i.t, i.line_at,
           r.target_id
    FROM feedback_claims c
    JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
    LEFT JOIN runs r ON r.run_id = c.run_id
"""


FEEDBACK_SNAPSHOT_PAGE = 200
FEEDBACK_FULL_BYTES = 512 * 1024
REVIEW_SIMILAR_MAX_CLAIMS = 200


def review_feed(conn, *, since: int | None = None, limit: int | None = None, offset: int = 0, full: bool = False, similar: bool = False) -> dict:
    """@brief 返回一份快照，或只返回 ``since`` 之后变化的 claim。

    ``since`` 等于当前反馈 revision 时不返回任何行；只为出现在本次答案里的运行读取文本。
    首屏快照默认分页（``limit``/``offset``）；全量只在显式 ``full=True`` 时给出，并带字节上限与
    ``truncated`` 标记，避免一次响应把整份主张列表塞给客户端。
    原始英文说明：A snapshot, or only the claims that changed after ``since``.

    @param conn 索引数据库连接。
    @param since 客户端已持有的反馈 revision，None 表示要全量快照。
    @param limit 单页条数；None 时用 :data:`FEEDBACK_SNAPSHOT_PAGE`。
    @param offset 快照页偏移（只在 ``since is None`` 时生效）。
    @param full 为 True 时返回全量（仍有字节上限）。
    @param similar 为 True 才给每行算语义近邻。默认 False：90 条待审 claim 各算一次句向量会让首屏多花数秒，
        相似结论改由 :func:`review_similar` 在展开某一场运行时按需计算。
    @return 含 ``same``、``snapshot``、``rev``、计数、``upserts`` 与 ``removed`` 的字典。
    """
    rev = _feedback_revision(conn)
    pending_total = int(conn.execute("SELECT COUNT(*) FROM feedback_claims WHERE status = 'pending'").fetchone()[0])
    open_subclaims_total = int(conn.execute(
        "SELECT COUNT(*) FROM feedback_subclaims WHERE status IN ('pending', 'info')"
    ).fetchone()[0])
    if since is not None and int(since) == rev:
        return {
            "same": True, "snapshot": False, "rev": rev, "pending_total": pending_total,
            "open_subclaims_total": open_subclaims_total,
            "open_work_total": pending_total + open_subclaims_total, "upserts": [], "removed": [],
        }
    next_offset = None
    total = None
    if since is None:
        snapshot = True
        removed: list[dict] = []
        # 分页对象是 claim（一条 record 可以有多条 claim），所以总数必须与分页单位一致：
        # 拿 feedback_items 计数会让“一条 item 多 claim”的快照提前判定到底，前端静默丢掉后面的 claim。
        total = int(conn.execute(
            "SELECT COUNT(*) FROM feedback_claims c"
            " JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq",
        ).fetchone()[0])
        if full:
            rows = [dict(row) for row in conn.execute(_CLAIM_LIST + " ORDER BY i.t DESC, c.seq DESC")]
        else:
            page = max(1, int(limit or FEEDBACK_SNAPSHOT_PAGE))
            offset = max(0, int(offset or 0))
            rows = [dict(row) for row in conn.execute(
                _CLAIM_LIST + " ORDER BY i.t DESC, c.seq DESC LIMIT ? OFFSET ?", (page, offset),
            )]
            if offset + len(rows) < total:
                next_offset = offset + len(rows)
    else:
        rows = [dict(row) for row in conn.execute(_CLAIM_LIST + " WHERE c.rev > ? ORDER BY i.t DESC, c.seq DESC", (int(since),))]
        removed = [
            {"run_id": row["run_id"], "seq": row["seq"], "digest": row["digest"], "status": row["status"], "reason": row["reason"]}
            for row in conn.execute(
                "SELECT run_id, seq, digest, status, reason FROM feedback_claim_removed WHERE rev > ?",
                (int(since),),
            )
        ]
        snapshot = False
    calibration = _attach_claim_text(rows)
    attach_prior_claims(conn, rows)
    if similar:
        attach_similar_claims(conn, rows)
    result = {
        "similar_attached": similar,
        "same": False,
        "snapshot": snapshot,
        "rev": rev,
        "pending_total": pending_total,
        "open_subclaims_total": open_subclaims_total,
        "open_work_total": pending_total + open_subclaims_total,
        "upserts": rows,
        "removed": removed,
        "calibration": calibration,
        "total": total,
        "next_offset": next_offset,
    }
    if full:
        # 全量导出也要有字节上限：对**最终序列化结果**做预算，envelope、calibration、truncated
        # 都算在内；超了就二分收缩行数并重编码，直到连 envelope 也达标（只累加行字节会漏掉外壳）。
        result = _cap_full_export(result, FEEDBACK_FULL_BYTES)
    return result


def _encoded_size(doc: dict) -> int:
    """@brief 响应的真实字节数（与 HTTP 层同一套 serialization）。

    @param doc 响应字典。
    @return UTF-8 字节数。
    """
    return len(json.dumps(doc, ensure_ascii=False, default=str).encode("utf-8"))


def _cap_full_export(result: dict, limit: int) -> dict:
    """@brief 把全量导出的最终 body 压到 ``limit`` 字节以内，并如实标 ``truncated``。

    行数对序列化字节单调，所以用二分找最大可保留行数；``truncated`` 在收缩时恒为 True，
    二分区间内不参与单调性判断。空 envelope 本身就超限时保留 0 行并照样标 ``truncated``。

    @param result 完整响应（``upserts`` 为全部行）。
    @param limit 字节上限。
    @return 收缩后的响应副本（未超限时原样返回）。
    """
    rows = list(result.get("upserts") or [])
    if _encoded_size(result) <= limit:
        return result

    def body(count: int) -> dict:
        trimmed = dict(result)
        trimmed["upserts"] = rows[:count]
        trimmed["truncated"] = True
        return trimmed

    low, high = 0, len(rows) - 1 if rows else 0
    best = body(0)
    while low <= high:
        middle = (low + high) // 2
        candidate = body(middle)
        if _encoded_size(candidate) <= limit:
            best, low = candidate, middle + 1
        else:
            high = middle - 1
    return best


def review_similar(conn, run_id: str, *, limit: int = 3) -> dict:
    """@brief 只算一场运行里待审 claim 的语义近邻，供展开该 run 时按需调用。

    首屏快照不为全部待审行算近邻（每条都要嵌入候选语料），把成本缩到用户真正展开的那一场。

    @param conn 索引数据库连接。
    @param run_id 运行 id（索引里的相对路径）。
    @param limit 每条 claim 最多挂几条近邻。
    @return ``{run_id, items: [{run_id, seq, digest, similar}]}``；没有待审行时 items 为空。
    """
    rows = [dict(row) for row in conn.execute(
        _CLAIM_LIST + " WHERE c.run_id = ? ORDER BY i.t DESC, c.seq DESC", (run_id,),
    )]
    pending = [row for row in rows if str(row.get("status") or "") == "pending"]
    # 单场也必须有界：成本与响应都随待审条数增长，超出部分如实标记而不是静默丢掉。
    truncated = len(pending) > REVIEW_SIMILAR_MAX_CLAIMS
    pending = pending[:REVIEW_SIMILAR_MAX_CLAIMS]
    _attach_claim_text(pending)
    attach_prior_claims(conn, pending)
    attach_similar_claims(conn, pending, limit=limit)
    return {
        "run_id": run_id,
        "items": [
            {"run_id": run_id, "seq": row["seq"], "digest": row["digest"], "similar": row.get("similar") or []}
            for row in pending
        ],
        "truncated": truncated,
    }


def _subclaims(body: dict) -> list[dict] | None:
    """@brief 解析可选的子 claim 列表；每项是一个独立判定的命题。

    原始英文说明：The optional subclaim list. Each item is one independently judged proposition.

    @param body 评审请求体。
    @return 归一化后的子 claim 列表；请求体未给出时返回 None。
    @raises ValueError 列表过长、项不是对象或文本为空时抛出。
    """
    value = body.get("subclaims")
    if value is None:
        return None
    if not isinstance(value, list) or len(value) > 128:
        raise ValueError("subclaims must be a list of at most 128 objects")
    out = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"subclaim {index} must be an object")
        text = str(item.get("text") or "").strip()
        if not text:
            raise ValueError(f"subclaim {index} has empty text")
        out.append({
            "text": text,
            "digest": str(item.get("digest") or ""),
            "status": item.get("status"),
            "verification": item.get("verification"),
            "evidence_source": str(item.get("evidence_source") or ""),
            "evidence": str(item.get("evidence") or ""),
            "note": str(item.get("note") or ""),
            "change_ref": str(item.get("change_ref") or ""),
            "derived_from": str(item.get("derived_from") or ""),
            "backend": str(item.get("backend") or ""),
            "target_id": str(item.get("target_id") or ""),
        })
    return out


def rejection(body: dict) -> str | None:
    """@brief 校验评审请求体的状态组合是否合法。

    原始英文说明：None when the review body is a legal status combination.

    @param body 评审请求体。
    @return 拒绝原因文本；合法时返回 None。
    """
    status = body.get("status")
    verification = body.get("verification")
    if status not in _STATUSES or verification not in _VERIFICATIONS:
        return "status or verification is not recognized"
    subclaims = body.get("subclaims")
    if subclaims is not None:
        if not isinstance(subclaims, list) or len(subclaims) > 128:
            return "subclaims must be a list of at most 128 objects"
        for index, item in enumerate(subclaims):
            if not isinstance(item, dict):
                return f"subclaim {index} must be an object"
            if not str(item.get("text") or "").strip():
                return f"subclaim {index} has empty text"
            sub_reason = rejection({
                key: item.get(key) for key in
                ("status", "verification", "evidence_source", "evidence", "note", "change_ref")
            })
            if sub_reason:
                return f"subclaim {index}: {sub_reason}"
        if status not in ("info", "pending"):
            return "a split claim must keep status info or pending"
    evidence_source = str(body.get("evidence_source") or "")
    evidence = str(body.get("evidence") or "").strip()
    note = str(body.get("note") or "").strip()
    change_ref = str(body.get("change_ref") or "").strip()
    if evidence_source and evidence_source not in _SOURCES:
        return "evidence_source is not recognized"
    if status == "pending":
        if verification != "unchecked":
            return "pending requires verification unchecked"
        return None
    if status == "info":
        if verification != "unchecked":
            return "info requires verification unchecked"
        return None
    if status == "obsolete":
        if not note:
            return "obsolete requires a note"
        return None
    if status == "adopted":
        if verification != "confirmed" or not evidence_source or not evidence:
            return "adopted requires a confirmed verification and evidence"
        return None
    if status == "rejected":
        if verification not in ("refuted", "confirmed"):
            return "rejected requires verification confirmed or refuted"
        if not note:
            return "rejected requires a note"
        return None
    if verification != "confirmed" or not change_ref:
        return "done requires a confirmed verification and change_ref"
    return None


def preview_reviews(run_dir: Path, rows: list[dict]) -> list[dict]:
    """@brief 不写入地校验一批评审；第一项非法即抛 ``ValueError``。

    原始英文说明：Validate a batch without writing. Raises ``ValueError`` on the first bad item.

    @param run_dir 运行目录。
    @param rows 待评审项列表。
    @return 每项含新旧状态与备注的计划列表。
    @raises ValueError 批次为空、项非法或 ``expected_status`` 与当前状态不符时抛出。
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("a review batch needs at least one item")
    doc = _load_review(run_dir / "feedback_review.json")
    planned: list[dict] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"item {index} must be an object")
        try:
            seq = int(row["seq"])
            digest = str(row["digest"] or "")
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"item {index} needs an integer seq and a digest") from None
        if not digest:
            raise ValueError(f"item {index} has an empty digest")
        body = {key: value for key, value in row.items() if key not in ("seq", "digest")}
        reason = rejection(body)
        if reason:
            raise ValueError(f"item {index}: {reason}")
        key = _review_key(seq, digest)
        previous = doc["claims"].get(key)
        previous_status = str(previous.get("status") or "pending") if isinstance(previous, dict) else "pending"
        expected = str(body.get("expected_status") or "")
        if expected and expected != previous_status:
            raise ValueError(
                f"item {index}: claim status is {previous_status}, expected {expected}; re-read the claim before reviewing it"
            )
        planned.append({
            "seq": seq,
            "digest": digest,
            "from_status": previous_status,
            "to_status": str(body.get("status") or ""),
            "note": str(body.get("note") or ""),
        })
        # Later items in the same batch see the status this item will write.
        doc["claims"][key] = {"status": body.get("status")}
    return planned


def apply_reviews(
    run_dir: Path,
    rows: list[dict],
    *,
    reviewer: str | None = None,
    method: str | None = None,
    batch_id: str | None = None,
) -> list[dict]:
    """@brief 先逐项校验，再一次性写入评审文件。

    单项评审与批量评审都走这条原子路径；任一项的 ``expected_status`` 过期，整批在改文件前中止。
    调用方持有运行锁，读-校验-写的安全由该锁保证。
    原始英文说明：Validate every item, then write the review file once.

    @param run_dir 运行目录。
    @param rows 待写入的评审项列表。
    @param reviewer 评审者名称，None 表示用 Git 身份。
    @param method 评审来源，取 ``human`` / ``agent`` / ``script`` / ``api``。
    @param batch_id 批次 id，None 时按需生成。
    @return 已写入的 claim 行列表。
    @raises ValueError 任一项非法或状态已变时抛出。
    """
    from taic import control
    from taic.git_identity import state as git_state

    planned = preview_reviews(run_dir, rows)
    path = run_dir / "feedback_review.json"
    doc = _load_review(path)
    run_cfg = {}
    try:
        run_cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    target = run_cfg.get("target") if isinstance(run_cfg.get("target"), dict) else {}
    if reviewer:
        chosen_reviewer = str(reviewer)
        reviewer_source = "explicit"
    else:
        from taic.git_identity import author as git_author

        identity = git_author(run_dir)
        chosen_reviewer = identity.display
        reviewer_source = identity.source
    git_commit, git_dirty = git_state(run_dir)
    chosen_batch = str(batch_id or "")
    if len(rows) > 1 and not chosen_batch:
        import uuid

        chosen_batch = f"batch-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    stored_rows: list[dict] = []
    ledger_rows: list[dict] = []
    for index, row in enumerate(rows):
        seq = int(row["seq"])
        digest = str(row["digest"] or "")
        body = {key: value for key, value in row.items() if key not in ("seq", "digest")}
        chosen_method = str(method or body.get("method") or "human")
        if chosen_method not in ("human", "agent", "script", "api"):
            raise ValueError(f"item {index}: method must be human, agent, script or api")
        backend = str(body.get("backend") or target.get("backend") or "")
        target_id = str(body.get("target_id") or target.get("id") or "")
        key = _review_key(seq, digest)
        previous = doc["claims"].get(key)
        previous_status = str(previous.get("status") or "pending") if isinstance(previous, dict) else "pending"
        expected = str(body.get("expected_status") or "")
        if expected and expected != previous_status:
            raise ValueError(
                f"item {index}: claim status is {previous_status}, expected {expected}; re-read the claim before reviewing it"
            )
        history = list(previous.get("history") or []) if isinstance(previous, dict) else []
        event = {
            "t": time.time(),
            "status": body.get("status"),
            "verification": body.get("verification"),
            "evidence_source": str(body.get("evidence_source") or ""),
            "evidence": str(body.get("evidence") or ""),
            "note": str(body.get("note") or ""),
            "topic": str(body.get("topic") or ""),
            "change_ref": str(body.get("change_ref") or ""),
            "backend": backend,
            "target_id": target_id,
            "reviewer": chosen_reviewer,
            "reviewer_source": reviewer_source,
            "method": chosen_method,
            "batch_id": str(body.get("batch_id") or chosen_batch or ""),
            "expected_status": str(body.get("expected_status") or ""),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        }
        subclaims = _subclaims(body)
        if subclaims is not None:
            for sub in subclaims:
                sub["backend"] = sub.get("backend") or backend
                sub["target_id"] = sub.get("target_id") or target_id
            event["subclaims"] = subclaims
        history.append(event)
        stored = dict(event)
        stored["history"] = history
        stored["reviewed_at"] = event["t"]
        doc["claims"][key] = stored
        stored_rows.append(stored)
        ledger_rows.append({**stored, "seq": seq, "digest": digest})
    control.write_json_atomic(path, doc)
    from taic.review_ledger import append_events

    try:
        run_id = str(run_dir.resolve().relative_to((Path(__file__).resolve().parents[1] / "runs").resolve()))
    except ValueError:
        run_id = run_dir.name
    append_events(ledger_rows, run_id=run_id)
    return stored_rows


def apply_review(run_dir: Path, seq: int, digest: str, body: dict, *, reviewer: str | None = None, method: str | None = None) -> dict:
    """@brief 追加一条评审并返回落盘的 claim；调用方持有运行锁。

    评审者身份默认取 Git 提交作者（``git var GIT_AUTHOR_IDENT``），表单不会问名字；
    显式传入的 ``reviewer`` 供回填与测试使用，并记为 ``explicit``。
    原始英文说明：Append one review and return the stored claim. The caller holds the run lock.

    @param run_dir 运行目录。
    @param seq 反馈条目序号。
    @param digest 条目摘要。
    @param body 评审内容。
    @param reviewer 评审者名称，None 表示用 Git 身份。
    @param method 评审来源，None 表示按请求体或 ``human``。
    @return 已写入的 claim 字典。
    """
    return apply_reviews(
        run_dir,
        [{**body, "seq": seq, "digest": digest}],
        reviewer=reviewer,
        method=method,
    )[0]
