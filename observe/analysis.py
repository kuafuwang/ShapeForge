"""§8.4 分析面板的派生读模型：落差、每美元、委员会有效性、噪声。

只读索引窄列（外加已确认运行的 ``confirm_report.json`` 取 cv 与批次数），不解析 ``row_json``。
聚合永远在**完整过滤集**上做，行/桶上限只限制展示；跨配置的聚合用 PR12 的统一可比性分组键
（``compare_key.comparison_group``）。每个面板都按 UTF-8 字节预算收口（≤8KB）。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from observe import detail
from observe.compare_key import comparison_group
from observe.index import borrow
from taic.modelcost import efficiency_fields

SMALL_SAMPLE = 5
USD_EPS = 1e-9
RANK_MIN = 20
PANEL_ROWS = 20
PANEL_COLUMNS = 20
PANEL_BYTES = 8192
TEXT_CHARS = 48
EPOCH_CHARS = 24
PANEL_EPOCHS = 8
CONFIRM_READ_LIMIT = 1_000_000

EPOCH_SQL = "CASE WHEN protocol_version = 'flame_chase_1' THEN '2' ELSE protocol_version END"


class EpochRequired(ValueError):
    """@brief 请求没有指定协议纪元、而匹配到的运行跨了多个纪元时抛出（HTTP 400）。

    原始英文说明：The request would mix protocol epochs; the caller has to name one.
    """

    def __init__(self, epochs: list[str]):
        """@brief 记录可选的协议纪元。

        @param epochs 可选的纪元列表。
        """
        super().__init__("protocol_epoch required")
        self.epochs = epochs


def _num(value) -> float | None:
    """@brief 只接受有限数值（bool 不算）。

    @param value 原始值。
    @return 有限浮点数，或 None。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _epoch(protocol_version: str) -> str:
    """@brief ``flame_chase_1`` 记成 ``2``，与索引的 ``protocol_epoch`` 口径一致。

    @param protocol_version 原始协议版本。
    @return 纪元字符串。
    """
    return "2" if protocol_version == "flame_chase_1" else protocol_version


def _bytes(document: dict) -> int:
    """@brief 与验收一致的序列化字节数（不小报中文）。

    @param document 面板字典。
    @return UTF-8 字节数。
    """
    return len(json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _short(value, limit: int = TEXT_CHARS) -> str:
    """@brief 展示用的短字符串；过长时截断并去掉换行。

    @param value 原始值。
    @param limit 最大字符数。
    @return 短字符串。
    """
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit]


def _stamp(kind: str, parts: list[str]) -> str:
    """@brief 面板戳：同一份源与同一组过滤得到同一个值。

    @param kind 面板名。
    @param parts 源指纹与过滤条件。
    @return 十六进制摘要。
    """
    return hashlib.sha256("\x1f".join(["analysis-v2", kind, *parts]).encode("utf-8")).hexdigest()


def _epoch_expression(prefix: str = "") -> str:
    """@brief 运行行的协议纪元表达式（``flame_chase_1`` 归到 ``2``）。

    @param prefix 列前缀（``"r."`` 用于 join）。
    @return SQL 表达式。
    """
    return (
        f"CASE WHEN {prefix}protocol_version = 'flame_chase_1' THEN '2' "
        f"ELSE {prefix}protocol_version END"
    )


def _epochs_for(clauses: list[str], args: list) -> list[str]:
    """@brief 某个过滤集里出现过的协议纪元（只看 ``runs``）。

    @param clauses WHERE 子句列表。
    @param args 子句参数。
    @return 排序后的纪元列表。
    """
    expression = _epoch_expression()
    with borrow() as index:
        return [
            str(row[0]) for row in index.conn.execute(
                f"SELECT DISTINCT {expression} AS epoch FROM runs WHERE {' AND '.join(clauses)}"
                " AND protocol_version != '' ORDER BY 1",
                args,
            )
        ]


def _choose_epoch(epochs: list[str], requested: str | None) -> str:
    """@brief 选定要分析的纪元：显式优先；只有一个（或空）时直接用；多个则要求人来选。

    @param epochs 可选纪元（已排序）。
    @param requested 调用方指定的纪元。
    @return 选中的纪元；空数据时为空串。
    @raises EpochRequired 匹配集跨多个纪元且没有指定纪元。
    """
    if requested:
        return requested
    if len(epochs) <= 1:
        return epochs[0] if epochs else ""
    raise EpochRequired(sorted(epochs))


def signed_delta_pct(search_best, published) -> float | None:
    """@brief 落差：``(published / search_best - 1) * 100``，两边都要是正数。

    @param search_best 搜索最优加速比。
    @param published 当前发布的加速比（必须来自 ``conclusion_of_run``）。
    @return 百分比（4 位小数），无法计算时为 None。
    """
    best = _num(search_best)
    shown = _num(published)
    if best is None or shown is None or best <= 0:
        return None
    return round((shown / best - 1.0) * 100.0, 4)


def usd_per_100pct_gain(spent_max, published) -> float | None:
    """@brief 替代口径：走到 100% 加速比（published−1）在每个百分点上花的美元。

    只有 ``published <= 1`` 才没有“增益”可除、返回 None；其余按权威公式除以
    ``max(published-1, ε)``（所以略高于 1 的 published 会得到很大的数，而不是被排除）。

    @param spent_max 单调不减的已花费。
    @param published 当前发布的加速比。
    @return 美元/百分之百增益，或 None。
    """
    spent = _num(spent_max)
    shown = _num(published)
    if spent is None or shown is None or shown <= 1.0:
        return None
    return round(spent / max(shown - 1.0, USD_EPS), 6)


def _percentile(values: list[float], fraction: float) -> float | None:
    """@brief 最近秩分位数（nearest-rank）：第 ``ceil(fraction*n)`` 小的值。

    @param values 已排序的数值列表。
    @param fraction 0–1 之间的比例。
    @return 分位数，空列表时为 None。
    """
    if not values:
        return None
    index = max(0, min(len(values) - 1, math.ceil(fraction * len(values)) - 1))
    return values[index]


def _ranks(values: list[float]) -> list[float]:
    """@brief 平均秩（并列取平均），用于 Spearman。

    @param values 数值列表。
    @return 每个值的秩（从 1 开始）。
    """
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def spearman(first: list[float], second: list[float]) -> float | None:
    """@brief 两个等长序列的 Spearman 秩相关；样本少于 ``RANK_MIN`` 或方差为 0 时返回 None。

    @param first 第一列。
    @param second 第二列。
    @return 相关系数，或 None。
    """
    if len(first) != len(second) or len(first) < RANK_MIN:
        return None
    left, right = _ranks(first), _ranks(second)
    n = len(left)
    mean = (n + 1) / 2.0
    cov = sum((a - mean) * (b - mean) for a, b in zip(left, right))
    var_left = sum((a - mean) ** 2 for a in left)
    var_right = sum((b - mean) ** 2 for b in right)
    if var_left <= 0 or var_right <= 0:
        return None
    return round(cov / math.sqrt(var_left * var_right), 4)


def _confirm_stats(directory: Path | None, run_id: str) -> tuple[float | None, int | None]:
    """@brief 只从 schema-4 确认报告取 cv 与批次数；旧格式/缺失报告时批次数不可判（None）。

    A schema-3 confirm report has no per-batch identity and no frozen expected, so it is not current
    evidence: neither its statistics nor the batch count may be recovered from ``result.json`` — a
    number that no current report supports is ``null`` here, not a remembered 2.  ``cv`` may still
    fall back to the winner candidate's own report in the index (that report's own statistic).

    @param directory 运行目录，可为 None。
    @param run_id 运行 id（退回索引时用）。
    @return ``(cv, n_batches)``；旧格式运行的第二项是 ``None``。
    """
    cv = None
    batches = None
    winner = ""
    current = False
    if directory is not None:
        from taic.artifacts import document_state

        path = directory / "confirm_report.json"
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                doc = json.loads(handle.read(CONFIRM_READ_LIMIT))
        except (OSError, ValueError):
            doc = None
        # Only a current (schema-4, type-strict) confirm report is evidence: an older one has no
        # per-batch identity and no frozen expected, so its ``cv``/batch count must not be shown as
        # today's statistics.  ``document_state`` keeps ``4.0``/``true`` from passing as 4.
        if document_state(doc) != "current":
            doc = None
        current = doc is not None
        if isinstance(doc, dict):
            benchmark = doc.get("benchmark") if isinstance(doc.get("benchmark"), dict) else {}
            candidate = benchmark.get("candidate") if isinstance(benchmark.get("candidate"), dict) else {}
            cv = _num(candidate.get("cv"))
            confirmation = benchmark.get("confirmation") if isinstance(benchmark.get("confirmation"), dict) else {}
            values = confirmation.get("confirmed_values")
            if isinstance(values, list):
                batches = len(values)
        try:
            with path.parent.joinpath("result.json").open("r", encoding="utf-8", errors="replace") as handle:
                result = json.loads(handle.read(CONFIRM_READ_LIMIT))
        except (OSError, ValueError):
            result = None
        if isinstance(result, dict):
            # No batch count from ``result.confirmed_values``: with an old-format report that number
            # is the run's own memory of a confirmation, not evidence this reader can check.
            winner = str(result.get("confirm_winner_label") or result.get("winner_label") or "")
    if cv is None and winner:
        # 旧格式确认报告没有 cv 时，退回胜者候选报告里的 benchmark.candidate.cv（索引窄列）。
        with borrow() as index:
            row = index.conn.execute(
                "SELECT cv FROM candidates WHERE run_id = ? AND label = ? AND cv IS NOT NULL LIMIT 1",
                (run_id, winner),
            ).fetchone()
        cv = _num(row[0]) if row is not None else None
    return cv, batches


def _median(values: list[float]) -> float | None:
    """@brief 中位数（偶数个取中间两个的平均）。

    @param values 数值列表。
    @return 中位数，空列表时为 None。
    """
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[middle], 6)
    return round((ordered[middle - 1] + ordered[middle]) / 2.0, 6)


def _fit(document: dict, *, sections: tuple[str, ...], text_paths: dict[str, tuple[str, ...]],
         aux: tuple[str, ...] = (), aux_keep: int = 8) -> dict:
    """@brief 按 UTF-8 字节预算收口：先截展示文本，再压辅助列表，最后逐项丢主表。

    ``aux`` 是“点名用”的辅助列表（如 ``insufficient_buckets`` 里的长 id）。它不能绕过预算：
    主表先保留（读者最需要），辅助列表先压到 ``aux_keep`` 项；仍超就丢主表，主表空了再继续丢
    辅助项。任何丢失都在 ``truncated`` 里按列表名报出来，完整计数留在文档的其他字段里。

    @param document 面板字典（就地修改）。
    @param sections 可从尾部丢弃的列表路径（点号分隔，支持 ``a.b``）。
    @param text_paths 每个列表元素里要截断的文本字段。
    @param aux 需要受同一字节预算约束的“字符串列表”路径。
    @param aux_keep 预算紧张时辅助列表先压到的项数。
    @return 同一个字典。
    """
    truncated = document.setdefault("truncated", {})

    def rows_at(path: str) -> list:
        node = document
        for part in path.split("."):
            node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                return []
        return node if isinstance(node, list) else []

    for path in sections:
        for field in text_paths.get(path, ()):
            for row in rows_at(path):
                if isinstance(row, dict) and isinstance(row.get(field), str):
                    shortened = _short(row[field])
                    if shortened != row[field]:
                        truncated["text"] = True
                    row[field] = shortened
    for name in ("epochs",):
        values = document.get(name)
        if isinstance(values, list):
            capped = [str(value)[:EPOCH_CHARS] for value in values[:PANEL_EPOCHS]]
            if capped != values:
                truncated["epochs"] = True
            document[name] = capped
    while _bytes(document) > PANEL_BYTES:
        oversized = [(name, rows_at(name)) for name in aux if len(rows_at(name)) > aux_keep]
        if oversized:
            name, rows = max(oversized, key=lambda pair: len(pair[1]))
            rows.pop()
            truncated[name.split(".")[-1]] = True
            continue
        for path in sections:
            rows = rows_at(path)
            if rows:
                rows.pop()
                truncated[path.split(".")[-1]] = True
                break
        else:
            pending = next((name for name in aux if rows_at(name)), None)
            if pending is None:
                break
            rows_at(pending).pop()
            truncated[pending.split(".")[-1]] = True
    return document


def gap(*, protocol_epoch: str | None = None, task: str | None = None,
        target: str | None = None, stamp: str = "") -> dict:
    """@brief 落差面板：每场已发布运行的 ``search_best → published``，以及每美元口径。

    ``published`` 取自索引里由 ``conclusion_of_run`` 写下的 ``published_speedup`` 列（不是
    ``result.confirmed_speedup``）；未发布的运行只进计数。统计（总数、excluded、各组中位数、
    秩相关）在**完整过滤集**上算，``rows`` 只限制展示（最多 ``PANEL_ROWS`` 行）。分组用 PR12 的
    统一可比性键 ``comparison_group``（task/dtype/size/target/epoch/arm），不同配置不混。

    @param protocol_epoch 只取该协议纪元；None 时要求匹配集只有一个纪元（空集给空面板）。
    @param task 只取该任务。
    @param target 只取该目标。
    @param stamp 客户端持有的面板戳。
    @return 面板字典；戳匹配时只回 ``same``。
    """
    clauses = ["status != 'missing'"]
    args: list = []
    if task:
        clauses.append("task = ?")
        args.append(task)
    if target:
        clauses.append("target_id = ?")
        args.append(target)
    epochs = _epochs_for(clauses, args)
    chosen = _choose_epoch(epochs, protocol_epoch)
    where = " AND ".join(clauses)
    epoch_filter = f"{EPOCH_SQL} = ?" if chosen else "1 = 1"
    epoch_args: list = [chosen] if chosen else []
    with borrow() as index:
        full = [
            dict(row) for row in index.conn.execute(
                "SELECT run_id, task, dtype, size, target_id, arm_id, published_speedup,"
                f" search_best_speedup, spent_max_usd, {EPOCH_SQL} AS epoch FROM runs"
                f" WHERE {where} AND {epoch_filter}"
                " AND publication = 'confirmed' AND published_speedup IS NOT NULL"
                " ORDER BY published_speedup DESC, run_id",
                (*args, *epoch_args),
            )
        ]
        unpublished = int(index.conn.execute(
            f"SELECT COUNT(*) FROM runs WHERE {where} AND {epoch_filter} AND status != 'missing'"
            " AND (publication != 'confirmed' OR published_speedup IS NULL)",
            (*args, *epoch_args),
        ).fetchone()[0] or 0)
        revision = index.revision()
    source_parts = [str(revision), chosen, task or "", target or ""]
    items = []
    excluded = 0
    for row in full:
        directory = detail.resolve_run(row["run_id"])
        cv, batches = _confirm_stats(directory, row["run_id"])
        if directory is not None:
            try:
                stat = (directory / "confirm_report.json").stat()
                source_parts.append(f"{row['run_id']}:{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                source_parts.append(f"{row['run_id']}:0:0")
        usd = usd_per_100pct_gain(row["spent_max_usd"], row["published_speedup"])
        if usd is None:
            excluded += 1
        items.append({
            "run_id": row["run_id"],
            "task": row["task"] or "",
            "dtype": row["dtype"] or "",
            "size": row["size"] or "",
            "target_id": row["target_id"] or "",
            "protocol_epoch": row["epoch"] or "",
            "arm_id": row["arm_id"] or "",
            "search_best": _num(row["search_best_speedup"]),
            "published": _num(row["published_speedup"]),
            "signed_delta_pct": signed_delta_pct(row["search_best_speedup"], row["published_speedup"]),
            "cv": cv,
            "n_batches": batches,
            "usd_per_100pct_gain": usd,
            "usd_per_confirmed": _num(row["spent_max_usd"]),
        })
    # 统一可比性分组：同一个 (task,dtype,size,target,epoch,arm) 才能放在一起看中位数。
    buckets: dict[tuple, list[dict]] = {}
    for item in items:
        buckets.setdefault(comparison_group(item), []).append(item)
    groups = []
    for key, members in buckets.items():
        deltas = [item["signed_delta_pct"] for item in members if item["signed_delta_pct"] is not None]
        usds = [item["usd_per_100pct_gain"] for item in members if item["usd_per_100pct_gain"] is not None]
        groups.append({
            "task": key[0],
            "dtype": key[1],
            "size": key[2],
            "target_id": key[3],
            "protocol_epoch": key[4],
            "arm_id": key[5],
            "n": len(members),
            "insufficient": len(members) < SMALL_SAMPLE,
            "median_signed_delta_pct": _median(deltas) if len(members) >= SMALL_SAMPLE else None,
            "median_usd_per_100pct_gain": _median(usds) if len(members) >= SMALL_SAMPLE else None,
        })
    groups.sort(key=lambda group: (-group["n"], group["task"], group["dtype"], group["size"], group["arm_id"]))
    pairs = [
        (item["cv"], item["signed_delta_pct"]) for item in items
        if item["cv"] is not None and item["signed_delta_pct"] is not None
    ]
    correlation = {"n": len(pairs), "insufficient": len(pairs) < RANK_MIN, "spearman": None}
    if len(pairs) >= RANK_MIN:
        correlation["spearman"] = spearman([pair[0] for pair in pairs], [pair[1] for pair in pairs])
    document = {
        "schema_version": "1",
        "protocol_epoch": chosen,
        "epochs": list(epochs),
        "published_total": len(items),
        "rows": items[:PANEL_ROWS],
        "groups": groups,
        "unpublished": unpublished,
        "excluded": excluded,
        "rank_correlation": correlation,
        # 每美元是替代口径：明确标出来，不把 gain_per_usd 的口径混进同一列。
        "usd_note": "替代口径：spent_max_usd / max(published-1, ε)，published ≤ 1 不给值并计入 excluded",
        "truncated": {"rows": len(items) > PANEL_ROWS, "epochs": False, "groups": False},
        "stamp": _stamp("gap", source_parts),
    }
    _fit(document, sections=("rows", "groups"), text_paths={
        "rows": ("task", "dtype", "size", "target_id", "arm_id"),
        "groups": ("task", "dtype", "size", "target_id", "arm_id"),
    })
    if stamp and stamp == document["stamp"]:
        return {"same": True, "stamp": document["stamp"]}
    return {"same": False, **document}


def committees(*, task: str | None = None, protocol_epoch: str | None = None, stamp: str = "") -> dict:
    """@brief 委员会有效性：单位是 session，不是“场”；按统一可比性键 + 委员会分桶。

    分桶键是 ``(task, dtype, size, target_id, protocol_epoch, arm_id, committee, provider, model)``：
    不同配置不混。session ⋈ candidate 走 ``(run_id, candidate_label = label)``；``committee=''`` 的
    会话单列成显式桶（不混进分母）。成本口径与 ``model_stats()`` 一致：先 SUM，再除以**全部**
    session（``efficiency_fields``），覆盖不足时标下界。

    @param task 只取该任务。
    @param protocol_epoch 只取该纪元；None 时要求匹配集只有一个纪元（空集给空面板）。
    @param stamp 客户端持有的面板戳。
    @return 面板字典；戳匹配时只回 ``same``。
    """
    run_clauses = ["status != 'missing'"]
    run_args: list = []
    if task:
        run_clauses.append("task = ?")
        run_args.append(task)
    epochs = _epochs_for(run_clauses, run_args)
    chosen = _choose_epoch(epochs, protocol_epoch)
    clauses = ["r.status != 'missing'"]
    args: list = []
    if task:
        clauses.append("r.task = ?")
        args.append(task)
    epoch_filter = f"{_epoch_expression('r.')} = ?" if chosen else "1 = 1"
    epoch_args: list = [chosen] if chosen else []
    where = " AND ".join(clauses)
    with borrow() as index:
        rows = [
            dict(row) for row in index.conn.execute(
                "SELECT r.task AS task, r.dtype AS dtype, r.size AS size, r.target_id AS target_id,"
                ' COALESCE(r.arm_id, \'\') AS arm_id,'
                f" {_epoch_expression('r.')} AS epoch,"
                " s.committee AS committee, s.provider AS provider, s.model AS model,"
                " COUNT(*) AS sessions, COUNT(DISTINCT s.run_id) AS runs,"
                " ROUND(SUM(COALESCE(s.cost_usd, 0)), 6) AS cost_usd,"
                " COUNT(s.cost_usd) AS sessions_with_cost,"
                " SUM(CASE WHEN c.label IS NOT NULL THEN 1 ELSE 0 END) AS joined_sessions,"
                " SUM(CASE WHEN c.status = 'ok' THEN 1 ELSE 0 END) AS ok_sessions,"
                " ROUND(AVG(c.speedup), 4) AS mean_speedup"
                " FROM sessions s JOIN runs r ON r.run_id = s.run_id"
                " LEFT JOIN candidates c ON c.run_id = s.run_id AND c.label = s.candidate_label"
                f" WHERE {where} AND {epoch_filter}"
                " GROUP BY r.task, r.dtype, r.size, r.target_id, arm_id, epoch,"
                " s.committee, s.provider, s.model"
                " ORDER BY r.task, epoch, sessions DESC, s.committee, s.provider, s.model",
                (*args, *epoch_args),
            )
        ]
        revision = index.revision()
        totals = index.conn.execute(
            "SELECT COUNT(*) AS sessions, COUNT(DISTINCT s.run_id) AS runs,"
            " ROUND(SUM(COALESCE(s.cost_usd, 0)), 6) AS cost_usd,"
            " SUM(CASE WHEN s.committee = '' THEN 1 ELSE 0 END) AS empty_sessions"
            f" FROM sessions s JOIN runs r ON r.run_id = s.run_id WHERE {where} AND {epoch_filter}",
            (*args, *epoch_args),
        ).fetchone()
    buckets = []
    insufficient_buckets = []
    for row in rows:
        sessions = int(row["sessions"] or 0)
        cost_values = int(row["sessions_with_cost"] or 0)
        # 与 model_stats 同口径：没有一条会话记成本时不给成本比；否则 SUM / 全部 session。
        efficiency = efficiency_fields(
            tokens_cache_read=None, tokens_output=None,
            cost_usd=float(row["cost_usd"] or 0.0) if cost_values else None, sessions=sessions,
        )
        identifier = "/".join((
            _short(row["task"]), _short(row["epoch"]), row["committee"] or "(未标注)",
            row["provider"] or "", row["model"] or "",
        ))
        bucket = {
            "id": identifier,
            "task": row["task"] or "",
            "dtype": row["dtype"] or "",
            "size": row["size"] or "",
            "target_id": row["target_id"] or "",
            "protocol_epoch": row["epoch"] or "",
            "arm_id": row["arm_id"] or "",
            "committee": row["committee"] or "",
            "provider": row["provider"] or "",
            "model": row["model"] or "",
            "sessions": sessions,
            "runs": int(row["runs"] or 0),
            "cost_usd": float(row["cost_usd"] or 0.0),
            "sessions_with_cost": cost_values,
            "cost_per_session_usd": efficiency["cost_per_session_usd"],
            "cost_is_lower_bound": cost_values < sessions,
            "joined_sessions": int(row["joined_sessions"] or 0),
            "ok_sessions": int(row["ok_sessions"] or 0),
            "ok_share": round(int(row["ok_sessions"] or 0) / int(row["joined_sessions"]), 4)
            if int(row["joined_sessions"] or 0) else None,
            "mean_speedup": _num(row["mean_speedup"]),
            "insufficient": sessions < SMALL_SAMPLE,
        }
        if bucket["insufficient"]:
            insufficient_buckets.append(identifier)
        buckets.append(bucket)
    document = {
        "schema_version": "1",
        "protocol_epoch": chosen,
        "epochs": list(epochs),
        "buckets": buckets,
        "totals": {
            "sessions": int(totals["sessions"] or 0),
            "runs": int(totals["runs"] or 0),
            "cost_usd": float(totals["cost_usd"] or 0.0),
            "empty_committee_sessions": int(totals["empty_sessions"] or 0),
        },
        # 逐桶判定；全局只是“有没有不足桶”，并点名它们（混合桶时不会静默消失）。
        # 数量始终是完整集统计；名单受字节预算约束，丢了哪些由 truncated.insufficient_buckets 报。
        "insufficient": bool(insufficient_buckets),
        "insufficient_count": len(insufficient_buckets),
        "insufficient_buckets": insufficient_buckets,
        "truncated": {"buckets": len(buckets) > PANEL_COLUMNS, "epochs": False},
        "stamp": _stamp("committees", [str(revision), chosen, task or ""]),
    }
    _fit(document, sections=("buckets",), aux=("insufficient_buckets",), text_paths={
        "buckets": ("task", "dtype", "size", "target_id", "arm_id", "committee", "provider", "model", "id"),
    })
    if stamp and stamp == document["stamp"]:
        return {"same": True, "stamp": document["stamp"]}
    return {"same": False, **document}


def noise(*, task: str | None = None, protocol_epoch: str | None = None, stamp: str = "") -> dict:
    """@brief 噪声面板：候选 cv 的 p10/p50/p90、reps 分布与 ``cache_hit`` 覆盖率。

    与落差面板同一个纪元口径（混纪元且没指定时拒绝），不把不同纪元的噪声悄悄混在一起。
    ``cache_hit`` 缺失是缺失，不当成 false：覆盖率单独报。

    @param task 只取该任务。
    @param protocol_epoch 只取该纪元；None 时要求匹配集只有一个纪元（空集给空面板）。
    @param stamp 客户端持有的面板戳。
    @return 面板字典；戳匹配时只回 ``same``。
    """
    run_clauses = ["status != 'missing'"]
    run_args: list = []
    if task:
        run_clauses.append("task = ?")
        run_args.append(task)
    epochs = _epochs_for(run_clauses, run_args)
    chosen = _choose_epoch(epochs, protocol_epoch)
    clauses = ["r.status != 'missing'"]
    args: list = []
    if task:
        clauses.append("r.task = ?")
        args.append(task)
    epoch_filter = f"{_epoch_expression('r.')} = ?" if chosen else "1 = 1"
    epoch_args: list = [chosen] if chosen else []
    where = " AND ".join(clauses)
    with borrow() as index:
        cvs = [
            float(row[0]) for row in index.conn.execute(
                "SELECT c.cv FROM candidates c JOIN runs r ON r.run_id = c.run_id"
                f" WHERE {where} AND {epoch_filter} AND c.cv IS NOT NULL", (*args, *epoch_args),
            )
        ]
        # 多取一项只为判断“还有更多”：固定 LIMIT 20 会把第 21 种 reps 静默吞掉，同时把
        # truncated.reps 永远写成 False。
        rep_rows = index.conn.execute(
            "SELECT c.reps, COUNT(*) FROM candidates c JOIN runs r ON r.run_id = c.run_id"
            f" WHERE {where} AND {epoch_filter} AND c.reps IS NOT NULL"
            " GROUP BY c.reps ORDER BY COUNT(*) DESC, c.reps LIMIT ?",
            (*args, *epoch_args, PANEL_COLUMNS + 1),
        ).fetchall()
        reps_total = int(index.conn.execute(
            "SELECT COUNT(DISTINCT c.reps) FROM candidates c JOIN runs r ON r.run_id = c.run_id"
            f" WHERE {where} AND {epoch_filter} AND c.reps IS NOT NULL",
            (*args, *epoch_args),
        ).fetchone()[0] or 0)
        reps = [
            {"reps": int(row[0]) if row[0] is not None else None, "n": int(row[1] or 0)}
            for row in rep_rows[:PANEL_COLUMNS]
        ]
        cache = index.conn.execute(
            "SELECT COUNT(*) AS total, COUNT(c.cache_hit) AS known,"
            " SUM(CASE WHEN c.cache_hit = 1 THEN 1 ELSE 0 END) AS hits"
            f" FROM candidates c JOIN runs r ON r.run_id = c.run_id WHERE {where} AND {epoch_filter}",
            (*args, *epoch_args),
        ).fetchone()
        revision = index.revision()
    cvs.sort()
    total = int(cache["total"] or 0)
    known = int(cache["known"] or 0)
    hits = int(cache["hits"] or 0)
    document = {
        "schema_version": "1",
        "protocol_epoch": chosen,
        "epochs": list(epochs),
        "cv": {
            "n": len(cvs),
            "p10": _percentile(cvs, 0.10),
            "p50": _percentile(cvs, 0.50),
            "p90": _percentile(cvs, 0.90),
            "min": cvs[0] if cvs else None,
            "max": cvs[-1] if cvs else None,
            "coverage": round(len(cvs) / total, 4) if total else None,
        },
        "reps": reps,
        "reps_total": reps_total,
        "cache_hit": {
            "total": total,
            "known": known,
            "true": hits,
            "false": known - hits,
            "coverage": round(known / total, 4) if total else None,
        },
        "insufficient": len(cvs) < SMALL_SAMPLE,
        "truncated": {"reps": len(rep_rows) > PANEL_COLUMNS, "epochs": False},
        "stamp": _stamp("noise", [str(revision), chosen, task or ""]),
    }
    _fit(document, sections=("reps",), text_paths={"reps": ()})
    if stamp and stamp == document["stamp"]:
        return {"same": True, "stamp": document["stamp"]}
    return {"same": False, **document}
