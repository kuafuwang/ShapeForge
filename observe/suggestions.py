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


def _link_tables(directory: Path) -> tuple[dict[tuple[str, int], str], dict[str, list[str]]]:
    """@brief 一次读完提案与评审两份 JSONL，做成 dict 索引。

    @param directory 运行目录。
    @return ``((digest, seq) -> proposal_key, proposal_key -> [useful|not-useful])``。
    """
    proposals: dict[tuple[str, int], str] = {}
    for row in _read_jsonl(directory / "harness_proposals.jsonl"):
        if row.get("status") == "valid":
            proposals.setdefault(
                (str(row.get("claim_digest") or ""), int(row.get("seq") or 0)),
                str(row.get("proposal_key") or ""),
            )
    reviews: dict[str, list[str]] = {}
    for row in _read_jsonl(directory / "harness_reviews.jsonl"):
        reviews.setdefault(str(row.get("proposal_key") or ""), []).append(
            "useful" if row.get("useful") is True else "not-useful"
        )
    return proposals, reviews


def _linked(directory: Path, *, seq: int, digest: str, cache: dict) -> tuple[str, list[str]]:
    """@brief 找出该 claim 对应的确切提案及其评审结论。

    请求内缓存按运行目录保存两份 JSONL 的索引：20 组建议可能来自 20 个不同运行，旧实现每组
    各读两遍文件；现在每个目录只读一次。

    原始英文说明：The exact proposal for this claim, not merely the first proposal with the same text.

    @param directory 运行目录。
    @param seq 反馈条目序号。
    @param digest 条目摘要。
    @param cache 本次请求的目录级缓存。
    @return 二元组：有效提案的 ``proposal_key``（没有时为空串），以及该提案的评审列表（useful/not-useful）。
    """
    tables = cache.get(directory)
    if tables is None:
        tables = _link_tables(directory)
        cache[directory] = tables
    proposals, review_index = tables
    key = proposals.get((str(digest), int(seq)), "")
    return key, list(review_index.get(key, [])) if key else []


def _run_simulated(directory: Path | None) -> bool:
    """@brief 测量类型缺省不是模拟。只看这一个 run 自己的记录。"""
    return feedback.measurement_simulated(directory)


def compact(
    *,
    task: str | None = None,
    target: str | None = None,
    model: str | None = None,
    status: str | None = None,
    backend: str | None = None,
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
    clauses = ["c.section IN ('对系统的建议', '需要帮助')", "c.digest <> ''"]
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
    if backend:
        clauses.append("r.backend = ?")
        args.append(backend)
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
    links: dict[Path, tuple[dict, dict]] = {}
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
        member_runs: dict[str, set[str]] = {}
        if digests:
            marks = ",".join("?" for _ in digests)
            rows = index.conn.execute(
                f"""
                WITH filtered AS (
                    SELECT c.run_id, c.seq, c.digest, c.status,
                           i.task, i.dtype, i.size, i.model, i.label, i.t, i.line_at,
                           r.target_id, r.backend
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
                SELECT DISTINCT c.digest, c.run_id, i.task, i.model, r.target_id, c.status
                {base} AND c.digest IN ({marks})
                """,
                [*args, *digests],
            ):
                bucket = dims.setdefault(str(row["digest"]), {"tasks": set(), "models": set(), "targets": set(), "statuses": set()})
                member_runs.setdefault(str(row["digest"]), set()).add(str(row["run_id"] or ""))
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
            converted_key, reviews = _linked(
                directory, seq=int(row.get("seq") or 0), digest=digest, cache=links)
        bucket = dims.get(digest) or {}
        claim_simulated = []
        for run_id in sorted(member_runs.get(digest) or ()):
            if not run_id:
                continue
            claim_simulated.append({
                "run_id": run_id,
                "simulated": _run_simulated(detail_mod.resolve_run(run_id)),
            })
        flags = {item["simulated"] for item in claim_simulated}
        groups[digest] = {
            "digest": digest,
            "claims": int(page_row["claims"]),
            "runs": int(page_row["runs"]),
            "tasks": sorted(bucket.get("tasks") or []),
            "models": sorted(bucket.get("models") or []),
            "targets": sorted(bucket.get("targets") or []),
            "statuses": sorted(bucket.get("statuses") or []),
            "backend": str(row.get("backend") or ""),
            "claim_simulated": claim_simulated,
            "simulated": flags.pop() if len(flags) == 1 else None,
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


_CLUSTER_INPUT_CAP = 2000


def _claim_source(directory: Path, line_at, digest: str, cache: dict) -> tuple[str, dict | None]:
    """@brief 取正文和它所在的反馈记录。模拟态要用这同一条记录，不能只看 run 目录。"""
    if isinstance(line_at, int):
        key = (str(directory), line_at)
        record = cache.get(key)
        if record is None:
            record = _record(directory, line_at) or {}
            cache[key] = record
        text = _item_text(record, digest) if record else ""
        if text:
            return text, record or None
    scanned = cache.get((str(directory), "all"))
    if scanned is None:
        scanned = [row for row in feedback.load_records(directory / "feedback.jsonl") if isinstance(row, dict)]
        cache[(str(directory), "all")] = scanned
    for record in scanned:
        text = _item_text(record, digest)
        if text:
            return text, record
    return "", None


def _text_for_digest(directory: Path, line_at, digest: str, cache: dict) -> str:
    """@brief 按行号取正文；行号对不上时再扫这一场的记录。"""
    return _claim_source(directory, line_at, digest, cache)[0]


def cluster_view(
    *,
    task: str | None = None,
    target: str | None = None,
    model: str | None = None,
    status: str | None = None,
    backend: str | None = None,
    limit: int | None = 50,
    preview: int = DEFAULT_PREVIEW,
    offset: int = 0,
    scorer=None,
    method: str | None = None,
    link_at: float | None = None,
    since: float | None = None,
) -> dict:
    """@brief 只读聚类视图。覆盖「对系统的建议」和「需要帮助」。

    先按后端、是否模拟、路由意图分区，再在区内聚类。不写评审文件。

    @param task 任务过滤。
    @param target 芯片过滤。
    @param model 模型过滤。
    @param status 评审状态过滤。
    @param backend 后端过滤。区内仍按每条主张自己的后端分开。
    @param limit 返回簇数。
    @param preview 预览长度。
    @param offset 簇页偏移。
    @param scorer 测试注入的打分函数。
    @param method 打分方法名。
    @param link_at 覆盖连接阈值。
    @param since 增量更新的时间戳；只返回 ``t > since`` 的反馈。``None`` 表示全量。
    @return ``observe.suggestion_cluster.cluster_rows`` 的结果。
    """
    from observe.suggestion_cluster import cluster_rows

    try:
        get_index().tick()
    except OSError:
        pass
    clauses = ["c.section IN ('对系统的建议', '需要帮助')", "c.digest <> ''"]
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
    if backend:
        clauses.append("r.backend = ?")
        args.append(backend)
    if status:
        clauses.append("c.status = ?")
        args.append(status)
    if since is not None:
        clauses.append("i.t > ?")
        args.append(float(since))
    where = " AND ".join(clauses)
    count_sql = f"""
        SELECT COUNT(*)
        FROM feedback_claims c
        JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
        LEFT JOIN runs r ON r.run_id = c.run_id
        WHERE {where}
    """
    sql = f"""
        SELECT c.digest AS digest, c.run_id AS run_id, c.seq AS seq, c.section AS section,
               c.status AS status, i.task AS task, i.line_at AS line_at, i.t AS t,
               r.backend AS backend
        FROM feedback_claims c
        JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
        LEFT JOIN runs r ON r.run_id = c.run_id
        WHERE {where}
        ORDER BY c.digest, c.run_id, c.seq
    """
    with borrow() as index:
        matched = int(index.conn.execute(count_sql, args).fetchone()[0])
        raw = [dict(row) for row in index.conn.execute(sql, args)] if matched <= _CLUSTER_INPUT_CAP else []
    input_truncated = matched > _CLUSTER_INPUT_CAP

    def _refused(warning: str, *, read: int, source_errors: list[dict]) -> dict:
        return {
            "readonly": True,
            "method": method or "not_clustered",
            "total_clusters": 0,
            "returned_clusters": 0,
            "offset": max(0, int(offset or 0)),
            "next_offset": None,
            "truncated": False,
            "input_rows": 0,
            "input_truncated": input_truncated,
            "matched": matched,
            "read": read,
            "source_errors": source_errors,
            "incomplete": True,
            "warning": warning,
            "clusters": [],
            "hint": warning,
        }

    if input_truncated:
        return _refused(
            f"cluster input has {matched} claims, above the {_CLUSTER_INPUT_CAP} cap; "
            "the view is incomplete and was not clustered",
            read=0,
            source_errors=[],
        )
    cache: dict = {}
    source_errors: list[dict] = []
    broken: set[str] = set()
    grouped: dict[tuple, dict] = {}
    read = 0
    for row in raw:
        run_id = str(row.get("run_id") or "")
        digest = str(row.get("digest") or "")
        directory = detail_mod.resolve_run(run_id)
        record = None
        if directory is None:
            text = ""
            reason = "missing_run"
        else:
            text, record = _claim_source(directory, row.get("line_at"), digest, cache)
            reason = "unreadable"
        if not text:
            source_errors.append({
                "run_id": run_id,
                "seq": row.get("seq"),
                "digest": digest,
                "reason": reason,
            })
            broken.add(digest)
            continue
        read += 1
        simulated = feedback.measurement_simulated(directory, record if isinstance(record, dict) else None)
        backend_id = str(row.get("backend") or "")
        status = str(row.get("status") or "")
        occurrence = {
            "run_id": run_id,
            "seq": row.get("seq"),
            "digest": digest,
            "status": status,
        }
        bucket = grouped.get((digest, backend_id, simulated))
        claim = {
            "digest": digest,
            "text": text,
            "section": str(row.get("section") or ""),
            "backend": backend_id,
            "simulated": simulated,
            "run_id": run_id,
            "seq": row.get("seq"),
            "claims": 1,
            "runs": 1,
            "status": status,
            "t": row.get("t") or 0,
            "occurrences": [occurrence],
        }
        if bucket is None:
            claim["_run_ids"] = {run_id} if run_id else set()
            grouped[(digest, backend_id, simulated)] = claim
            continue
        bucket["claims"] += 1
        bucket.setdefault("occurrences", []).append(occurrence)
        statuses = [part for part in str(bucket.get("status") or "").split(",") if part]
        if status and status not in statuses:
            bucket["status"] = ",".join([*statuses, status])
        seen = bucket.setdefault("_run_ids", {str(bucket.get("run_id") or "")})
        if run_id:
            seen.add(run_id)
        seen.discard("")
        bucket["runs"] = len(seen)
        if (row.get("t") or 0, row.get("seq") or 0) >= (bucket.get("t") or 0, bucket.get("seq") or 0):
            bucket["run_id"] = run_id
            bucket["seq"] = row.get("seq")
            bucket["text"] = text
            bucket["section"] = claim["section"]
    nodes = []
    for (digest, _backend, _simulated), item in list(grouped.items()):
        if digest in broken:
            grouped.pop((digest, _backend, _simulated), None)
            continue
        item.pop("_run_ids", None)
        nodes.append(item)
    view = cluster_rows(
        nodes,
        scorer=scorer,
        method=method,
        link_at=link_at,
        limit=limit,
        offset=offset,
        preview=preview,
        input_truncated=False,
    )
    view["matched"] = matched
    view["read"] = read
    view["source_errors"] = source_errors
    view["incomplete"] = bool(source_errors)
    if source_errors:
        view["warning"] = (
            f"{len(source_errors)} claims could not be read; "
            "their digests were left out and this view is incomplete"
        )
        view["hint"] = view["warning"]
    else:
        view["warning"] = ""
    return view
