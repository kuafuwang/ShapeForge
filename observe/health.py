"""Read-only model health prototype for the observe panel.

Every number is recomputed from the existing run/session/feedback projection and
from the real route-M gate traces that happen to exist. Nothing here writes a
trust grant, a run config, an overlay or a model weight. Missing data stays
``None``/``unavailable``; it is never turned into a zero rate.

The human feedback counts reuse :func:`observe.trust.metrics` so adopted/done
and rejected/refuted keep exactly one definition.
"""

from __future__ import annotations

import json
from pathlib import Path

from observe import detail, trust
from observe.index import borrow

# 页面与 JSON 共用这一句：这是历史 run 的描述统计，不是授权依据。
NOTICE = "历史 run 的描述统计；P1 尚未 active；不参与授权"

# ``taic/session_supervisor.py::classify_result`` 会写出的显式 outcome。
EVALUATED = "evaluated"
UNCHANGED = "unchanged"
NO_SUBMISSION = "agent_no_submission"
OTHER_FAILURE_OUTCOMES = ("agent_failed", "provider_failed", "delivery_rejected", "cost_incomplete")
KNOWN_OUTCOMES = (EVALUATED, UNCHANGED, NO_SUBMISSION) + OTHER_FAILURE_OUTCOMES
PREMISE_KINDS = ("verified", "unverified", "downgraded")

SETTLE_TRACE_FILE = "harness_settle_traces.jsonl"
ROUTE_M_TRACE_FILE = "harness_route_m.jsonl"
NEVER_AUTO = "never_auto"
_TRACE_CACHE: dict[str, tuple[tuple[int, int], list[dict]]] = {}
_TRACE_CACHE_LIMIT = 512


def _read_jsonl(path: Path) -> list[dict]:
    """@brief 容错读取 JSONL；坏行跳过，不因一行坏字节丢掉整份门迹。

    @param path JSONL 文件。
    @return 字典行列表；文件不存在时为空列表。
    """
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _cached_rows(path: Path) -> list[dict]:
    """@brief 按 ``(mtime_ns, size)`` 缓存门迹，避免面板每帧重读未变的文件。

    @param path 门迹文件。
    @return 字典行列表。
    """
    key = str(path)
    try:
        stat = path.stat()
    except OSError:
        _TRACE_CACHE.pop(key, None)
        return []
    signature = (stat.st_mtime_ns, stat.st_size)
    cached = _TRACE_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    rows = _read_jsonl(path)
    _TRACE_CACHE[key] = (signature, rows)
    while len(_TRACE_CACHE) > _TRACE_CACHE_LIMIT:
        _TRACE_CACHE.pop(next(iter(_TRACE_CACHE)))
    return rows


def clear_cache() -> None:
    """@brief 丢掉门迹文件缓存；测试或换 ``runs/`` 根时用。"""
    _TRACE_CACHE.clear()


def _trace_run_ids(run_id: str | None) -> list[str]:
    """@brief 取要扫门迹的 run id，单 run 过滤与全局各一条 SQL。

    @param run_id 只看这一场运行；None 表示全部。
    @return run id 列表。
    """
    with borrow() as index:
        if run_id:
            row = index.conn.execute("SELECT run_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return [str(row[0])] if row else []
        return [str(row[0]) for row in index.conn.execute("SELECT run_id FROM runs ORDER BY run_id")]


def _gate(row: dict) -> dict | None:
    """@brief 从一条门迹里取 ``never_auto`` 那一项。

    @param row 门迹行。
    @return 门字典；没有该项时返回 None。
    """
    gates = row.get("gates")
    if not isinstance(gates, list):
        return None
    for gate in gates:
        if isinstance(gate, dict) and gate.get("name") == NEVER_AUTO:
            return gate
    return None


def _author(row: dict) -> tuple[str, str] | None:
    """@brief 取留痕里绑定的作者；缺一不记，不猜模型。

    @param row 路线 M 留痕行。
    @return ``(provider, model)`` 或 None。
    """
    author = row.get("author")
    if not isinstance(author, dict):
        return None
    provider = str(author.get("provider") or "")
    model = str(author.get("model") or "")
    return (provider, model) if provider and model else None


def never_auto(*, run_id: str | None = None) -> tuple[dict, dict[tuple[str, str], dict]]:
    """@brief 只统计真实存在的 G1-M ``never_auto`` 门迹，不把它补成 0%。

    ``harness_settle_traces.jsonl`` 只取 ``route=M`` 的行；``harness_route_m.jsonl``
    整份都是路线 M。前一种没有作者字段，只进全局计数；逐模型归属只用后一种里
    绑定 ``author`` 的行。没有任何门迹时所有计数为 ``None``、``coverage`` 为
    ``unavailable``，比例本版一律不计算。

    @param run_id 只看这一场运行；None 表示全部 run。
    @return ``(全局门迹摘要, {(provider, model): 逐模型摘要})``。
    """
    root = detail.root_path()[0]
    run_ids = _trace_run_ids(run_id)
    totals = {"traces": 0, "observed": 0, "passed": 0, "failed": 0, "not_run": 0,
              "attributed": 0, "unattributed": 0, "runs_with_traces": 0}
    per_model: dict[tuple[str, str], dict] = {}
    for name in run_ids:
        directory = detail.resolve_run(name, root=root)
        if directory is None:
            continue
        found = 0
        for filename, event, route_only in (
            (SETTLE_TRACE_FILE, "settle_gate_trace", True),
            (ROUTE_M_TRACE_FILE, "route_m", False),
        ):
            for row in _cached_rows(directory / filename):
                if row.get("event") != event:
                    continue
                if route_only and (row.get("route") != "M" or row.get("surface") == "premise"):
                    continue
                totals["traces"] += 1
                found += 1
                gate = _gate(row) or {}
                status = str(gate.get("status") or "not_run")
                if status == "pass":
                    totals["passed"] += 1
                elif status == "fail":
                    totals["failed"] += 1
                else:
                    totals["not_run"] += 1
                if status != "not_run":
                    totals["observed"] += 1
                identity = _author(row) if filename == ROUTE_M_TRACE_FILE else None
                if identity is None:
                    totals["unattributed"] += 1
                    continue
                totals["attributed"] += 1
                bucket = per_model.setdefault(identity, {"traces": 0, "observed": 0, "passed": 0, "failed": 0, "not_run": 0})
                bucket["traces"] += 1
                bucket[{"pass": "passed", "fail": "failed"}.get(status, "not_run")] += 1
                if status != "not_run":
                    bucket["observed"] += 1
        if found:
            totals["runs_with_traces"] += 1
    note = (
        "只统计盘上真实存在的路线 M 留痕（harness_settle_traces.jsonl 的 route=M 行与 "
        "harness_route_m.jsonl）；没有门迹的历史 run 记为 unavailable，不写成 0%。"
        "本版只报门迹计数，不计算比例；逐模型归属只来自绑定作者的路线 M 行。"
    )
    if totals["traces"]:
        document = {
            "coverage": "partial",
            "ratio": None,
            "note": note,
            "runs_scanned": len(run_ids),
            "runs_with_traces": totals["runs_with_traces"],
            "traces": totals["traces"],
            "gate_observed": totals["observed"],
            "gate_passed": totals["passed"],
            "gate_failed": totals["failed"],
            "gate_not_run": totals["not_run"],
            "attributed_traces": totals["attributed"],
            "unattributed_traces": totals["unattributed"],
        }
    else:
        document = {
            "coverage": "unavailable",
            "ratio": None,
            "note": note,
            "runs_scanned": len(run_ids),
            "runs_with_traces": None,
            "traces": None,
            "gate_observed": None,
            "gate_passed": None,
            "gate_failed": None,
            "gate_not_run": None,
            "attributed_traces": None,
            "unattributed_traces": None,
        }
    models = {}
    for identity, bucket in per_model.items():
        models[identity] = {
            "coverage": "partial",
            "traces": bucket["traces"],
            "gate_observed": bucket["observed"],
            "gate_passed": bucket["passed"],
            "gate_failed": bucket["failed"],
            "gate_not_run": bucket["not_run"],
        }
    return document, models


def _session_rows(conn, run_id: str | None) -> dict[tuple[str, str], dict]:
    """@brief 按 provider/model 汇总会话结局；未知 outcome 单列，不并入未交核。

    @param conn 只读索引连接。
    @param run_id 只看这一场运行；None 表示全部。
    @return ``{(provider, model): 结局计数}``。
    """
    where = " WHERE run_id = ?" if run_id else ""
    args = (run_id,) if run_id else ()
    known = ",".join("?" for _ in KNOWN_OUTCOMES)
    rows = conn.execute(
        f"""
        SELECT provider, model, COUNT(*) AS sessions,
               SUM(CASE WHEN outcome = ? THEN 1 ELSE 0 END) AS evaluated,
               SUM(CASE WHEN outcome = ? THEN 1 ELSE 0 END) AS unchanged,
               SUM(CASE WHEN outcome = ? THEN 1 ELSE 0 END) AS no_submission,
               SUM(CASE WHEN outcome = 'agent_failed' THEN 1 ELSE 0 END) AS agent_failed,
               SUM(CASE WHEN outcome = 'provider_failed' THEN 1 ELSE 0 END) AS provider_failed,
               SUM(CASE WHEN outcome = 'delivery_rejected' THEN 1 ELSE 0 END) AS delivery_rejected,
               SUM(CASE WHEN outcome = 'cost_incomplete' THEN 1 ELSE 0 END) AS cost_incomplete,
               SUM(CASE WHEN COALESCE(outcome, '') NOT IN ({known}) THEN 1 ELSE 0 END) AS unknown
        FROM sessions{where}
        GROUP BY provider, model
        """,
        (EVALUATED, UNCHANGED, NO_SUBMISSION, *KNOWN_OUTCOMES, *args),
    ).fetchall()
    out = {}
    for row in rows:
        item = {key: int(row[key] or 0) for key in (
            "sessions", "evaluated", "unchanged", "no_submission", "agent_failed",
            "provider_failed", "delivery_rejected", "cost_incomplete",
        )}
        item["unknown_outcomes"] = int(row["unknown"] or 0)
        known_sessions = item["sessions"] - item["unknown_outcomes"]
        item["known_outcomes"] = known_sessions
        item["no_submission_rate"] = (item["no_submission"] / known_sessions) if known_sessions else None
        item["evaluated_rate"] = (item["evaluated"] / known_sessions) if known_sessions else None
        out[(str(row["provider"] or ""), str(row["model"] or ""))] = item
    return out


def _premise_rows(conn, run_id: str | None) -> dict[tuple[str, str], dict]:
    """@brief 按 provider/model 汇总反馈前提；没有前提记录时返回 None 而不是 0。

    @param conn 只读索引连接。
    @param run_id 只看这一场运行；None 表示全部。
    @return ``{(provider, model): 前提计数与降级比例}``。
    """
    where = " WHERE c.run_id = ?" if run_id else ""
    args = (run_id,) if run_id else ()
    rows = conn.execute(
        f"""
        SELECT s.provider AS provider, s.model AS model,
               SUM(CASE WHEN c.premise = 'verified' THEN 1 ELSE 0 END) AS verified,
               SUM(CASE WHEN c.premise = 'unverified' THEN 1 ELSE 0 END) AS unverified,
               SUM(CASE WHEN c.premise = 'downgraded' THEN 1 ELSE 0 END) AS downgraded
        FROM feedback_claims c
        JOIN sessions s ON s.run_id = c.run_id AND s.seq = c.seq
        {where}
        GROUP BY s.provider, s.model
        """,
        args,
    ).fetchall()
    out = {}
    for row in rows:
        counts = {key: int(row[key] or 0) for key in PREMISE_KINDS}
        total = sum(counts.values())
        out[(str(row["provider"] or ""), str(row["model"] or ""))] = {
            **counts,
            "total": total,
            "downgrade_rate": (counts["downgraded"] / total) if total else None,
        }
    return out


def _review_rows(conn, run_id: str | None) -> dict[tuple[str, str], dict]:
    """@brief 复用 ``trust.metrics`` 的 adopted/done、rejected+refuted、pending 口径。

    ``trust.metrics`` 按 target/task 分组；这里只把计数相加后重算比例，不另写判定。

    @param conn 只读索引连接。
    @param run_id 只看这一场运行；None 表示全部。
    @return ``{(provider, model): 人工反馈计数}``。
    """
    totals: dict[tuple[str, str], dict] = {}
    for row in trust.metrics(conn, run_id=run_id):
        key = (str(row.get("provider") or ""), str(row.get("model") or ""))
        bucket = totals.setdefault(key, {"claims": 0, "reviewed": 0, "adopted_done": 0, "rejected_refuted": 0, "pending": 0})
        for name, source in (
            ("claims", "claims"), ("reviewed", "reviewed"), ("adopted_done", "reasonable"),
            ("rejected_refuted", "refuted"), ("pending", "pending"),
        ):
            bucket[name] += int(row.get(source) or 0)
    for bucket in totals.values():
        reviewed = bucket["reviewed"]
        bucket["adopted_rate"] = (bucket["adopted_done"] / reviewed) if reviewed else None
        bucket["wilson_lower"] = trust.wilson_lower(bucket["adopted_done"], reviewed)
    return totals


def collect(conn, *, run_id: str | None = None) -> dict:
    """@brief 一次收齐会话结局、反馈前提与人工反馈计数。

    @param conn 只读索引连接。
    @param run_id 只看这一场运行；None 表示全部。
    @return 含 ``sessions`` / ``premise`` / ``feedback`` 三张按 provider/model 索引的映射。
    """
    return {
        "sessions": _session_rows(conn, run_id),
        "premise": _premise_rows(conn, run_id),
        "feedback": _review_rows(conn, run_id),
    }


def attach(rows: list[dict], *, run_id: str | None = None) -> dict:
    """@brief 给 ``model_stats`` 的模型行附上只读健康度，并返回全局门迹摘要。

    @param rows ``observe.query.model_stats`` 的模型行（含 provider/model）。
    @param run_id 只看这一场运行；None 表示全部。
    @return 含 ``health_notice`` 与 ``never_auto`` 的字典；没有模型行时仍给出说明。
    """
    with borrow() as index:
        collected = collect(index.conn, run_id=run_id)
    sessions = collected["sessions"]
    premises = collected["premise"]
    reviews = collected["feedback"]
    gate_document, gate_models = never_auto(run_id=run_id)
    for row in rows:
        key = (str(row.get("provider") or ""), str(row.get("model") or ""))
        session = sessions.get(key) or {}
        premise = premises.get(key)
        review = reviews.get(key)
        gates = gate_models.get(key) or {
            "coverage": "unavailable", "traces": None, "gate_observed": None,
            "gate_passed": None, "gate_failed": None, "gate_not_run": None,
        }
        row["health"] = {
            "scope": "provider/model，跨委员会合计",
            "sessions": session.get("sessions"),
            "known_outcomes": session.get("known_outcomes"),
            "evaluated": session.get("evaluated"),
            "unchanged": session.get("unchanged"),
            "evaluated_rate": session.get("evaluated_rate"),
            "no_submission": session.get("no_submission"),
            "no_submission_rate": session.get("no_submission_rate"),
            "other_failures": {
                name: session.get(name)
                for name in ("agent_failed", "provider_failed", "delivery_rejected", "cost_incomplete")
            } if session else None,
            "unknown_outcomes": session.get("unknown_outcomes"),
            "premise": premise if premise else {
                "verified": None, "unverified": None, "downgraded": None, "total": None, "downgrade_rate": None,
            },
            "feedback": review,
            "never_auto": gates,
        }
    return {"health_notice": NOTICE, "never_auto": gate_document}
