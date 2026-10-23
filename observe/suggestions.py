"""Compact, token-conscious retrieval of free-text system suggestions.

The SQLite index names claims but does not store their text. This module first groups the
index rows by digest and reads only one representative feedback line per group, returning a
short preview. The full sentence is fetched only for the selected (run, seq, digest).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from observe import detail as detail_mod
from observe import feedback
from observe.index import borrow, get_index

DEFAULT_PREVIEW = 120
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
MAX_PREVIEW = 500
MAX_GROUPS = 200


def _record(directory: Path, line_at: int | None) -> dict | None:
    """@brief 取包含系统建议的反馈记录：优先按行号直读，否则扫描全部记录。

    @param directory 运行目录。
    @param line_at ``feedback.jsonl`` 中的行号，可为 None。
    @return 反馈记录字典；找不到时返回 None。
    """
    if line_at is not None:
        row = feedback._record_at(directory / "feedback.jsonl", int(line_at))
        if isinstance(row, dict):
            return row
    for record in feedback.entries(directory).get("records", []):
        for item in record.get("claims", []):
            if item.get("section") == "对系统的建议":
                return record
    return None


def _item_text(record: dict, digest: str) -> str:
    """@brief 在记录的项目或 claim 中按 digest 找对应文本。

    @param record 反馈记录字典。
    @param digest 条目摘要。
    @return 对应文本；没有匹配时返回空字符串。
    """
    for item in record.get("items") or []:
        if isinstance(item, dict) and str(item.get("digest") or "") == str(digest):
            return str(item.get("text") or "")
    for item in record.get("claims") or []:
        if isinstance(item, dict) and str(item.get("digest") or "") == str(digest):
            return str(item.get("text") or "")
    return ""


def _preview(text: str, chars: int) -> str:
    """@brief 压缩空白并截断成不超过 ``chars`` 字符的预览，超长时以省略号结尾。

    @param text 原始文本。
    @param chars 预览字符上限。
    @return 单行预览文本。
    """
    text = " ".join(_CONTROL.sub(" ", str(text or "")).split())
    if len(text) <= chars:
        return text
    return text[: max(0, chars - 1)].rstrip() + "…"


def _read_jsonl(path: Path) -> list[dict]:
    """@brief 宽容地读取 JSONL：文件不存在或某行无法解析时跳过而不报错。

    @param path JSONL 文件路径。
    @return 字典行列表。
    """
    rows = []
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    with handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _linked(directory: Path, *, seq: int, digest: str) -> tuple[str, list[str]]:
    """@brief 找出该 claim 对应的确切提案（而非仅文本相同的首个提案）及其评审结论。

    原始英文说明：The exact proposal for this claim, not merely the first proposal with the same text.

    @param directory 运行目录。
    @param seq 反馈条目序号。
    @param digest 条目摘要。
    @return 二元组：有效提案的 ``proposal_key``（没有时为空串），以及该提案的评审列表（useful/not-useful）。
    """
    key = ""
    for row in _read_jsonl(directory / "harness_proposals.jsonl"):
        if (
            str(row.get("claim_digest") or "") == str(digest)
            and int(row.get("seq") or 0) == int(seq)
            and row.get("status") == "valid"
        ):
            key = str(row.get("proposal_key") or "")
            break
    reviews = []
    if key:
        reviews = [
            "useful" if row.get("useful") is True else "not-useful"
            for row in _read_jsonl(directory / "harness_reviews.jsonl")
            if str(row.get("proposal_key") or "") == key
        ]
    return key, reviews


def compact(
    *,
    task: str | None = None,
    target: str | None = None,
    model: str | None = None,
    status: str | None = None,
    limit: int = 20,
    preview: int = DEFAULT_PREVIEW,
    offset: int = 0,
) -> dict:
    """@brief 按 digest 聚合系统建议，每组只返回短预览。

    这是面向 AI 的第一阶段：不返回全文，也不调用模型。总数与分页由 SQL 对整个筛选集计算，
    因此 ``truncated`` 描述的是返回的这一页，而不是隐形的扫描上限。每个返回的 digest 只读一行代表性
    反馈，其余留在磁盘上直到 ``detail()``。

    @param task 只看该任务名，None 表示不限。
    @param target 只看该 target，None 表示不限。
    @param model 只看该模型的建议，None 表示不限。
    @param status 只看该评审状态，None 表示不限。
    @param limit 本页最大组数，会被夹到 1..200。
    @param preview 每组预览字符数，会被夹到 0..500。
    @param offset 本页在组列表中的偏移。
    @return 含总数、本页组、``next_offset`` / ``truncated`` 分页字段与 ``hint`` 的字典。
    """
    try:
        get_index().tick()
    except OSError:
        pass
    limit = max(1, min(int(limit or 20), MAX_GROUPS))
    offset = max(0, int(offset or 0))
    preview = max(0, min(int(preview if preview is not None else DEFAULT_PREVIEW), MAX_PREVIEW))
    clauses = ["c.section = '对系统的建议'", "c.digest <> ''"]
    args: list[object] = []
    if task:
        clauses.append("i.task = ?")
        args.append(task)
    if target:
        clauses.append("r.target_id = ?")
        args.append(target)
    if model:
        clauses.append("i.model = ?")
        args.append(model)
    if status:
        clauses.append("c.status = ?")
        args.append(status)
    base = f"""
        FROM feedback_claims c
        JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
        LEFT JOIN runs r ON r.run_id = c.run_id
        WHERE {' AND '.join(clauses)}
    """
    groups: dict[str, dict] = {}
    with borrow() as index:
        total_claims = int(index.conn.execute(f"SELECT COUNT(*) {base}", args).fetchone()[0])
        total_groups = int(index.conn.execute(f"SELECT COUNT(DISTINCT c.digest) {base}", args).fetchone()[0])
        page = [dict(row) for row in index.conn.execute(
            f"""
            SELECT c.digest AS digest, COUNT(*) AS claims, COUNT(DISTINCT c.run_id) AS runs,
                   COALESCE(MAX(i.t), 0) AS latest_t
            {base}
            GROUP BY c.digest
            ORDER BY runs DESC, latest_t DESC, c.digest
            LIMIT ? OFFSET ?
            """,
            [*args, limit, offset],
        )]
        digests = [str(row["digest"]) for row in page]
        representative: dict[str, dict] = {}
        dims: dict[str, dict[str, set[str]]] = {}
        if digests:
            marks = ",".join("?" for _ in digests)
            rows = index.conn.execute(
                f"""
                WITH filtered AS (
                    SELECT c.run_id, c.seq, c.digest, c.status,
                           i.task, i.dtype, i.size, i.model, i.label, i.t, i.line_at,
                           r.target_id
                    {base} AND c.digest IN ({marks})
                ), ranked AS (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY digest ORDER BY t DESC, seq DESC) AS rn
                    FROM filtered
                )
                SELECT * FROM ranked WHERE rn = 1
                """,
                [*args, *digests],
            )
            representative = {str(row["digest"]): dict(row) for row in rows}
            for row in index.conn.execute(
                f"""
                SELECT DISTINCT c.digest, i.task, i.model, r.target_id, c.status
                {base} AND c.digest IN ({marks})
                """,
                [*args, *digests],
            ):
                bucket = dims.setdefault(str(row["digest"]), {"tasks": set(), "models": set(), "targets": set(), "statuses": set()})
                if row["task"]:
                    bucket["tasks"].add(str(row["task"]))
                if row["model"]:
                    bucket["models"].add(str(row["model"]))
                if row["target_id"]:
                    bucket["targets"].add(str(row["target_id"]))
                if row["status"]:
                    bucket["statuses"].add(str(row["status"]))
    for page_row in page:
        digest = str(page_row["digest"])
        row = representative.get(digest)
        if row is None:
            continue
        directory = detail_mod.resolve_run(str(row.get("run_id") or ""))
        text = ""
        converted_key = ""
        reviews: list[str] = []
        if directory is not None:
            record = _record(directory, row.get("line_at"))
            if record is not None:
                text = _item_text(record, digest)
            converted_key, reviews = _linked(directory, seq=int(row.get("seq") or 0), digest=digest)
        bucket = dims.get(digest) or {}
        groups[digest] = {
            "digest": digest,
            "claims": int(page_row["claims"]),
            "runs": int(page_row["runs"]),
            "tasks": sorted(bucket.get("tasks") or []),
            "models": sorted(bucket.get("models") or []),
            "targets": sorted(bucket.get("targets") or []),
            "statuses": sorted(bucket.get("statuses") or []),
            "first": {
                "run_id": str(row.get("run_id") or ""),
                "seq": row.get("seq"),
                "label": row.get("label") or "",
                "t": row.get("t"),
            },
            "preview": _preview(text, preview),
            "converted_key": converted_key,
            "link_scope": "first (representative claim only)",
            "reviews": reviews,
        }
    ordered = list(groups.values())
    for group in ordered:
        group["reviewed"] = sorted(set(group.pop("reviews", [])))
    returned = len(ordered)
    # Advance by the page the SQL returned, even if a representative row could not be read,
    # so a damaged run cannot make the caller fetch the same page forever.
    next_offset = offset + len(page)
    truncated = next_offset < total_groups
    return {
        "total_claims": total_claims,
        "total_groups": total_groups,
        "returned_groups": returned,
        "offset": offset,
        "next_offset": next_offset if truncated else None,
        "truncated": truncated,
        "limit": limit,
        "preview_chars": preview,
        "groups": ordered,
        "hint": "Links describe the representative claim only; another claim with the same digest may have a different conversion. Fetch selected run/seq/digest details and follow next_offset for more groups.",
    }


def detail(run_id: str, seq: int, digest: str) -> dict:
    """@brief 取出一条确切的系统建议及其转换/评审关联，而不必读取所有运行。

    原始英文说明：Fetch one exact suggestion and its conversion/review links, without reading every run.

    @param run_id 运行 id。
    @param seq 反馈条目序号。
    @param digest 条目摘要。
    @return 含建议文本、状态、提案、评审、同前出现（``prior``）与语义相似（``similar``）的字典。
    @raises FileNotFoundError 运行目录无法解析时抛出。
    @raises KeyError 该 (run_id, seq, digest) 找不到建议时抛出。
    """
    directory = detail_mod.resolve_run(run_id)
    if directory is None:
        raise FileNotFoundError(run_id)
    records = feedback.entries(directory).get("records", [])
    source = None
    source_record = None
    for record in records:
        for item in record.get("claims", []):
            if int(item.get("seq") or 0) == int(seq) and str(item.get("digest") or "") == str(digest):
                source = item
                source_record = record
                break
        if source:
            break
    if source is None:
        # Old records may not have a claim view for a section that is still in feedback.jsonl.
        from taic import harness

        for item in harness.free_text_suggestions(directory):
            if int(item.get("seq") or 0) == int(seq) and str(item.get("digest") or "") == str(digest):
                source = item
                break
    if source is None:
        raise KeyError(f"{run_id}#{seq}#{digest}")
    from taic import harness

    board = harness.board(directory)
    proposal = next((
        row for row in board["proposals"]
        if str(row.get("claim_digest") or "") == str(digest)
        and int(row.get("seq") or 0) == int(seq)
        and row.get("status") == "valid"
    ), None)
    reviews = []
    if proposal:
        reviews = [row for row in board["reviews"] if row.get("proposal_key") == proposal.get("proposal_key")]
    started = next((
        row.get("session_started") for row in board.get("suggestions") or []
        if int(row.get("seq") or 0) == int(seq) and str(row.get("digest") or "") == str(digest)
    ), None)
    if started is None:
        started = source.get("session_started")
    task = str((source_record or {}).get("task") or "")
    target_id = str((source_record or {}).get("target_id") or "")
    item_t = (source_record or {}).get("t")
    prior: list[dict] = []
    if not task or not target_id:
        with borrow() as index:
            indexed = index.conn.execute(
                """
                SELECT i.task, i.t, r.target_id
                FROM feedback_claims c
                JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
                LEFT JOIN runs r ON r.run_id = c.run_id
                WHERE c.run_id = ? AND c.seq = ? AND c.digest = ?
                """,
                (run_id, int(seq), str(digest)),
            ).fetchone()
        if indexed is not None:
            task = task or str(indexed["task"] or "")
            target_id = target_id or str(indexed["target_id"] or "")
            item_t = item_t if item_t is not None else indexed["t"]
    similar: list[dict] = []
    prior: list[dict] = []
    if task and target_id:
        texts = [str(source.get("text") or "")]
        texts.extend(str(sub.get("text") or "") for sub in source.get("subclaims") or [] if isinstance(sub, dict))
        with borrow() as index:
            merged: dict[tuple, dict] = {}
            for text in dict.fromkeys(texts):
                for item in feedback.prior_claims(
                    index.conn, digest=str(digest), task=task, target_id=target_id,
                    text=text, before_t=item_t,
                    exclude_run=run_id, exclude_seq=int(seq),
                ):
                    identity = (item.get("run_id"), item.get("seq"), item.get("digest"), item.get("sub_index"))
                    merged.setdefault(identity, item)
            prior = sorted(
                merged.values(),
                key=lambda item: (float(item.get("reviewed_at") or item.get("t") or 0), str(item.get("run_id") or "")),
                reverse=True,
            )[:5]
            similar = feedback.similar_claims(
                index.conn, digest=str(digest), task=task, target_id=target_id,
                text=str(source.get("text") or ""), before_t=item_t,
                exclude_run=run_id, exclude_seq=int(seq),
            )
    prior_keys = {(p.get("run_id"), p.get("seq"), p.get("digest")) for p in prior}
    similar = [item for item in similar if (item.get("run_id"), item.get("seq"), item.get("digest")) not in prior_keys]
    return {
        "run_id": run_id,
        "seq": int(seq),
        "digest": str(digest),
        "text": str(source.get("text") or ""),
        "section": source.get("section") or "对系统的建议",
        "status": source.get("status") or "",
        "session_started": started,
        "probe_time_verified": proposal.get("probe_time_verified") if proposal else None,
        "converted_key": proposal.get("proposal_key") if proposal else "",
        "proposal": proposal,
        "reviews": reviews,
        "prior": prior,
        "similar": similar,
    }
