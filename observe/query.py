"""Read the observe index. Listing, filtering and arm summaries do not open candidate reports."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from pathlib import Path

from experiments.common import suite_metrics
from experiments.contract_menu import judge
from observe import detail
from observe.budget import budget_status, summarize, summarize_columns
from observe.cluster_names import cluster_name
from observe.compare_key import config_key
from observe.failure_sig import SESSION_FAILURE_OUTCOMES
from observe.index import borrow, get_index, split_extra_fingerprint
from taic.artifacts import report_state
from taic.modelcost import efficiency_fields
from taic.plan_ledger import nodes_from_trace
from taic.policy import feature_signature

PAGE_LIMIT = 100
# §10.4 组级上限：候选一页 50 颗，反馈一页 10 条。缺省值与前端常量一致。
CANDIDATES_PAGE = 50
FEEDBACK_PAGE = 10
_SESSIONS_FEED_VERSION = "session-cost-kpi-v1"
_LIVE = {"last": None, "n": 0}
_SENT = 0
_TICK = {"at": 0.0}
_BEST_DISCOVER = {"at": 0.0}
_HISTORY = {"key": None, "rows": []}
_SORTS = {"started": "started_at", "speedup": "published_speedup", "spent": "spent_usd", "updated": "updated_at"}
# §11.5：两个进程内缓存都必须有上限；无上限的 dict 在长期看板进程里就是泄漏。
_CACHE_LIMIT = 256
_FEATURE_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_GRID_CACHE: "OrderedDict[tuple, list[dict]]" = OrderedDict()


def _cache_get(cache, key):
    """@brief 读一个 LRU 缓存并把命中项移到末尾。

    @param cache ``OrderedDict``。
    @param key 缓存键。
    @return 命中值或 None。
    """
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_put(cache, key, value) -> None:
    """@brief 写一个 LRU 缓存，超过上限时丢最旧的一项。

    @param cache ``OrderedDict``。
    @param key 缓存键。
    @param value 待缓存值。
    """
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_LIMIT:
        cache.popitem(last=False)


def gpu_snapshot() -> dict:
    """@brief 取当前 GPU 快照（转发给 ``detail.gpu_snapshot``）。

    @return GPU 快照字典。
    """
    return detail.gpu_snapshot()


pid_alive = detail.pid_alive


def refresh() -> None:
    """@brief 发现新目录并重读索引已知运行的改动；供必须与磁盘一致的读路径使用。

    这是显式的全量对账：即使写者日志健康也逐 run 复核，所以绕过看板直接 ``mv``/``rm`` 运行目录后
    ``refresh()`` 仍会立刻把行标成 missing。平时的增量帧走 ``tick()`` 的日志消费。

    原始英文说明：Pick up new directories and edits to runs the index already knows. Used by reads that must match the disk.
    """
    get_index().sync_known(force=True)


def _tick_interval() -> float:
    """@brief 取空闲扫描的最小间隔，来自 ``OBSERVE_TICK_INTERVAL``。

    @return 间隔秒数，不小于 0；环境变量非法时返回 0.5。
    """
    try:
        return max(0.0, float(os.environ.get("OBSERVE_TICK_INTERVAL", "0.5")))
    except ValueError:
        return 0.5


def _refresh_on_read() -> bool:
    """@brief 报告是否开启了读时刷新（``OBSERVE_REFRESH_ON_READ=1``）。

    @return 开启时为 True。
    """
    return os.environ.get("OBSERVE_REFRESH_ON_READ") == "1"


def tick() -> None:
    """@brief 空闲帧：每个间隔最多扫一次目录，所有请求与 SSE 客户端共享。

    测试设置 ``OBSERVE_TICK_INTERVAL=0`` 与 ``OBSERVE_REFRESH_ON_READ=1``，让测试内写入的运行立即可见；
    生产请求路径不扫描。
    """
    now = time.monotonic()
    if now - _TICK["at"] < _tick_interval():
        return
    _TICK["at"] = now
    get_index().tick()


def history() -> list[dict]:
    """@brief 返回所有运行的 detail 行，按索引 revision 缓存，不逐请求重扫。

    原始英文说明：Every run as the detail row, cached by index revision. Not re-scanned per request.

    @return 按开始时间降序排列的行字典列表（副本）。
    """
    if _refresh_on_read():
        tick()
    rows = []
    with borrow() as index:
        key = (str(index.path), index.revision())
        if key == _HISTORY["key"]:
            return list(_HISTORY["rows"])
        fetched = [item["row_json"] for item in index.conn.execute("SELECT row_json FROM runs WHERE status != 'missing'")]
    for text in fetched:
        if not text:
            continue
        rows.append(json.loads(text))
    rows.sort(key=lambda row: float(row.get("started_at") or 0), reverse=True)
    _HISTORY["key"] = key
    _HISTORY["rows"] = list(rows)
    return rows


def budget_report(rows: list[dict] | None = None) -> dict:
    """@brief 对给定行（默认取窄投影 ``budget_rows()``）求预算汇总。

    不传 ``rows`` 时只 SELECT 预算列，不解码 ``row_json``；报告字段与旧实现逐字段相同。

    @param rows 历史行列表，None 时现取窄投影。
    @return ``summarize`` 生成的预算报告。
    """
    return summarize(rows) if rows is not None else _budget_summary()


BUDGET_TOTALS_SQL = (
    "SELECT COUNT(*) AS n,"
    " COALESCE(SUM(spent_usd), 0) AS spent,"
    " COALESCE(SUM(CASE WHEN granted_budget_usd > 0 THEN granted_budget_usd END), 0) AS budget,"
    " COALESCE(SUM(CASE WHEN granted_budget_usd > 0 AND spent_usd - granted_budget_usd > 1e-9"
    " THEN spent_usd - granted_budget_usd ELSE 0 END), 0) AS over"
    " FROM runs WHERE status != 'missing'"
)
# 候选集必须覆盖每一个可能 flagged 的行：耗尽只看 stop_reason；超预算/接近预算需要告警预算 > 0。
# 判定“耗尽”的权威语义在 ``observe.budget._grade``：它先 ``strip()`` 再比 ``budget_exhausted``，所以
# SQL 不能写成等值比较——``'  budget_exhausted\n'`` 这类带前后空白的停止原因会先被 SQL 滤掉，窄投影
# 就会少报一条“已耗尽”。这里用“包含”做上界（literal ``instr``，不走 LIKE 的 ``_`` 通配）：带空白、
# 甚至带 ``strip()`` 能去掉的非常规空白字符的行都进候选集，最终判不判由 ``_grade`` 决定。
BUDGET_FLAGGED_SQL = (
    "SELECT run_id, task, dtype, status, stop_reason, spent_usd, granted_budget_usd, budget_usd,"
    " control_action, control_reason, topped_up FROM runs WHERE status != 'missing' AND ("
    " instr(stop_reason, 'budget_exhausted') > 0 OR (granted_budget_usd > 0 AND ("
    " status = 'running' OR spent_usd - granted_budget_usd > 1e-9)))"
)


def _budget_summary() -> dict:
    """@brief 直接从窄列汇总预算；不构造每行的历史行，也不解码 ``row_json``。

    合计用一条 SQL 聚合，逐行判定只跑“可能 flagged”的候选集（真实语料里个位数到几十行），
    所以 10k 行也不再是 O(每行一个 Python 对象)。字段与旧 ``summarize(history())`` 相同。

    @return ``observe.budget.summarize_columns`` 的报告。
    """
    if _refresh_on_read():
        tick()
    with borrow() as index:
        totals = index.conn.execute(BUDGET_TOTALS_SQL).fetchone()
        fetched = index.conn.execute(BUDGET_FLAGGED_SQL).fetchall()
    return summarize_columns(
        fetched,
        totals=(int(totals["n"] or 0), float(totals["spent"] or 0.0),
                float(totals["budget"] or 0.0), float(totals["over"] or 0.0)),
    )


BUDGET_ROW_COLUMNS = (
    "run_id, task, dtype, status, phase, stop_reason, started_at, publication, published_speedup,"
    " spent_usd, budget_usd, granted_budget_usd, control_action, control_reason, topped_up,"
    " initial_budget_usd"
)


def _budget_row(item: dict) -> dict:
    """@brief 把 ``runs`` 的一列行拼成预算汇总要的行（不碰 ``row_json``）。

    ``granted_budget_usd`` 存的就是摄入时算好的告警预算（``disk_budget_usd`` 才是盘上预算）。
    告警用的 ``stop_reason`` 也要按同一规则重算：已记录但尚未被运行采纳的加预算会撤销旧的
    ``budget_exhausted`` 标记，否则窄投影会比旧报告多报一条“已耗尽”。

    @param item 含列值的字典（``run_id`` 等）。
    @return 可用于 ``observe.budget.summarize`` 的行。
    """
    spent = item.get("spent_usd")
    disk = item.get("budget_usd")
    alert_budget = item.get("granted_budget_usd")
    stop = item.get("stop_reason") or ""
    shown = _as_float(alert_budget)
    on_disk = _as_float(disk)
    spent_n = _as_float(spent)
    if (
        shown is not None and on_disk is not None and shown > on_disk + 1e-9
        and stop == "budget_exhausted" and spent_n is not None and shown > spent_n
    ):
        stop = ""
    status = item.get("status") or ""
    return {
        "id": item.get("run_id"),
        "task": item.get("task") or "",
        "dtype": item.get("dtype") or "",
        "status": status,
        "phase": item.get("phase") or "",
        "stop_reason": item.get("stop_reason") or "",
        "started_at": item.get("started_at"),
        "publication": item.get("publication"),
        "published_speedup": item.get("published_speedup"),
        "spent_usd": spent,
        "budget_usd": alert_budget,
        "budget": budget_status(spent, alert_budget, stop, status == "running"),
        "control": {"action": item.get("control_action"), "reason": item.get("control_reason") or ""},
        "topped_up": bool(item.get("topped_up")),
        "initial_budget_usd": item.get("initial_budget_usd"),
    }


def _as_float(value) -> float | None:
    """@brief 把值转成有限浮点数；布尔、非数值、NaN/inf 返回 None。

    @param value 待转换的值。
    @return 有限浮点数或 None。
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def budget_rows(*, status: str | None = None) -> list[dict]:
    """@brief 预算汇总的窄投影：一条 SQL 取预算相关列，不解码 ``row_json``。

    ``status`` 给定时只取该状态（``overview`` 用它只拉活运行）。

    @param status 可选的 ``runs.status`` 过滤。
    @return 可用于 ``observe.budget.summarize`` 的行列表。
    """
    if _refresh_on_read():
        tick()
    # 与 ``history()`` 一致：预算报告不计 ``missing`` 行；只有显式 status 过滤时才按它取。
    where = " WHERE status != 'missing'" if status is None else " WHERE status = ?"
    args: tuple = () if status is None else (status,)
    order = " ORDER BY COALESCE(started_at, 0) DESC, rowid"
    with borrow() as index:
        fetched = [dict(row) for row in index.conn.execute(
            f"SELECT {BUDGET_ROW_COLUMNS} FROM runs{where}{order}", args
        )]
    return [_budget_row(item) for item in fetched]


def resolve_run(run_id: str) -> Path | None:
    """@brief 把 run id 解析成运行目录，并确保该运行已进索引。

    写路径（显式 refresh、删除、迁移）用它；请求路径不要用，因为 ``Index.resolve`` 会先
    ``_ingest()``，那会在 stamp 判断之前读候选报告、反馈与转录。

    @param run_id 运行 id。
    @return 运行目录；索引里没有该 run 时返回 None。
    """
    return get_index().resolve(run_id)


def _read_run(run_id: str) -> Path | None:
    """@brief 只解析运行目录，不摄取索引；详情与面板的读路径用它。

    未进索引的运行也能读出目录并渲染负载，只是 ``detail_stamp`` 为空、没有 stamp 短路。

    @param run_id 运行 id。
    @return 运行目录；不存在或越界时返回 None。
    """
    return detail.resolve_run(run_id)


def run_detail(run_id: str, *, stamp: str = "") -> dict | None:
    """@brief 组装一个运行的完整详情负载（含特征网格、计划结果与反馈全文）。

    ``stamp`` 与当前 detail 戳一致时只回 ``{"same": True, ...}``。这里不摄取索引：
    索引由显式 refresh 或 SSE 的 tick 追平，详情只读文件。

    @param run_id 运行 id。
    @param stamp 客户端已持有的 detail 戳，空串表示不复用。
    @return 详情字典；运行不存在时返回 None；戳匹配时返回标记字典。
    """
    directory = _read_run(run_id)
    if directory is None:
        return None
    current = _full_detail_stamp(run_id)
    if stamp and stamp == current:
        return {"same": True, "id": run_id, "detail_stamp": current}
    detail_doc = detail.run_detail(run_id)
    if detail_doc is None:
        return None
    config = detail._load(directory / "run_config.json")
    detail_doc["feature_grid"] = feature_grid(directory, detail_doc.get("result"))
    detail_doc["plan_outcomes"] = plan_outcomes(directory, config, detail_doc.get("state"), detail_doc.get("result"))
    from observe.feedback import entries

    detail_doc["feedback"] = entries(directory)
    stored_writer = _stored_kernel_writer(run_id)
    if stored_writer:
        detail_doc["kernel_writer"] = stored_writer
    detail_doc["detail_stamp"] = current
    return detail_doc


def run_summary(run_id: str, *, stamp: str = "") -> dict | None:
    """@brief 只装配 L1 摘要：不读候选报告、反馈全文、预算事件与原始 state/result。

    重面板（候选、反馈、预算事件、时间线、特征格子、计划结局、原始 state/result）各自有端点，
    摘要只留首屏要看的字段。``stamp`` 与当前 detail 戳一致时只回 ``{"same": True, ...}``。
    这里不摄取索引，因此未进索引的运行也不会被强行读一遍候选报告与转录。

    @param run_id 运行 id。
    @param stamp 客户端已持有的 detail 戳，空串表示不复用。
    @return 摘要字典；运行不存在时返回 None；戳匹配时返回标记字典。
    """
    directory = _read_run(run_id)
    if directory is None:
        return None
    current = _detail_stamp(run_id)
    if stamp and current and stamp == current:
        return {"same": True, "id": run_id, "detail_stamp": current}
    document = detail.run_detail(run_id, include=frozenset())
    if document is None:
        return None
    # §5.7：待评测计数读窄投影；未摄入的 run 没有表行，才退回读 outbox。
    stored = get_index().stored_pending_evaluation(run_id)
    document["pending_evaluation"] = stored if stored is not None else detail.pending_evaluation(directory)
    # kernel_writer 是摄入窄投影：负载与摘要戳必须同源，不能一个读文件、一个读指纹。
    stored_writer = _stored_kernel_writer(run_id)
    if stored_writer:
        document["kernel_writer"] = stored_writer
    # 不截断、不改写任何字段：摘要与旧详情逐字段同值。§5.2 的 8KB 是 (C) 校准目标
    # （plan §0.4：编写时未实测的校准值），不是对任意/恶意输入的数学保证；真实语料
    # （63 场）实测最大约 3.7KB，固定 fixture 约 1.9KB，回归断言在 test_observe_read_paths.py。
    document["detail_stamp"] = current
    return document


def _detail_stamp(run_id: str) -> str:
    """@brief 生成 L1 摘要的身份戳（有界摘要）；追加对话、改写非 winner 候选都不改变它。

    戳由摘要真正依赖的源组成：state/result/config/确认文件，``fp_extra`` 的摘要半段
    （``budget_events.json``/``ledger.json``/``resume.lock``/``eval_progress.json``/``evaluation_outbox``），
    以及摄入窄投影 ``runs.kernel_writer``（旧 run 上它由 transcript/agent_work 现推，是摘要字段）。
    候选与反馈只在各自面板的 ``panel_stamp`` 里，**不进这里**：把每颗候选的报告/内核指纹拼进
    摘要戳会让它随候选数无界增长（250 颗实测 17.7KB），而且改写一颗非 winner 的 ``report.json``
    会串扰首屏摘要的 ``same`` 判定。

    原始英文说明：Identity of the L1 summary payload; transcript appends and non-winner candidate rewrites do not change it.

    @param run_id 运行 id。
    @return 十六进制摘要；索引里没有该运行时为空串（未摄入的运行没有可缓存的戳）。
    """
    row = get_index().conn.execute(
        "SELECT fp_state, fp_result, fp_config, fp_confirm, fp_extra, kernel_writer FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        return ""
    summary_extra, _ingestion_only = split_extra_fingerprint(row["fp_extra"])
    return _panel_stamp("summary", [
        *(str(row[key] or "") for key in ("fp_state", "fp_result", "fp_config", "fp_confirm")),
        summary_extra,
        str(row["kernel_writer"] or ""),
    ])


def _stored_kernel_writer(run_id: str) -> str:
    """@brief 取摄入窄投影里的 ``kernel_writer``；没有索引行时为空串。

    摘要与完整详情都用它覆盖文件推导值，让“负载里的 pi/chat”与“戳”来自同一个来源。

    @param run_id 运行 id。
    @return ``"pi"``/``"chat"`` 或空串。
    """
    row = get_index().conn.execute(
        "SELECT kernel_writer FROM runs WHERE run_id = ?", (run_id,),
    ).fetchone()
    return str(row["kernel_writer"] or "") if row is not None else ""


def _full_detail_stamp(run_id: str) -> str:
    """@brief 完整详情自己的戳：覆盖它真正读的每个源，而不是只覆盖摘要。

    ``run_detail`` 的负载含 candidates/feedback/llm/timeline/feature_grid/plan，因此戳必须包含
    候选逐文件 stat、反馈日志与评审文件、``llm.json``、``transcript.jsonl``（timeline 的问模型泳道）、
    机全局 GPU 事件日志/分片（等锁泳道与 ``truncated_history``），以及 ``feedback.entries()`` 里
    ``calibration`` 实际读取的那份本机标定文件。摘要戳只是它的一部分：两者形状不同，不能用同一个
    值互相短路（旧实现让完整详情在候选就地改写后一直回 ``same``）。

    原始英文说明：The full-detail stamp covers every source the full payload reads, not just the summary.

    @param run_id 运行 id。
    @return 十六进制摘要；索引里没有该运行或目录不存在时为空串。
    """
    directory = _read_run(run_id)
    if directory is None:
        return ""
    summary = _detail_stamp(run_id)
    if not summary:
        # 未进索引的运行没有可缓存的摘要戳，完整详情也不给戳（保持“无索引无缓存”的旧契约）。
        return ""
    return _panel_stamp("detail", [
        summary,
        *_candidate_source_parts(directory),
        *_best_source_parts(directory),
        _file_stamp(directory / "feedback.jsonl"),
        _file_stamp(directory / "feedback_review.json"),
        # feedback.entries() 默认带的是 read_calibration() 的结果，它按默认目标解析路径（不看该 run 的
        # target_id）。这里用同一优先级拼路径并只 stat，保证“实际读了哪份标定”也进完整详情的戳。
        _file_stamp(_calibration_path("")),
        _file_stamp(directory / "llm.json"),
        _file_stamp(directory / "transcript.jsonl"),
        _file_stamp(directory / "ledger.json"),
        _file_stamp(detail.gpu_events_path()),
        _file_stamp(detail.gpu_events_path().with_suffix(".truncated")),
        *detail.gpu_shard_stamps(),
    ])


def feature_grid(directory: Path, result: dict | None) -> list[dict]:
    """@brief 按源码签名给出每个格子；格子保留搜索中见过的最快候选。

    相同源码只签一次。结果缓存到 ``state.json`` 或 ``result.json`` 变化为止。
    原始英文说明：One cell per source signature. The cell keeps the fastest candidate seen during the search.

    @param directory 运行目录。
    @param result 解析后的 ``result.json``，可为 None。
    @return 每项含 ``signature``、``label``、``speedup`` 与 ``published`` 的列表。
    """
    winner = (result or {}).get("confirm_winner_label")
    from experiments.common import conclusion_of_run

    published_ok = conclusion_of_run(directory, result).get("publication") == "confirmed"
    key = (
        str(directory.resolve()),
        _file_signature(directory / "state.json"),
        _file_signature(directory / "result.json"),
        str(winner or ""),
        published_ok,
        _grid_source_signature(directory),
    )
    cached = _cache_get(_GRID_CACHE, key)
    if cached is not None:
        return cached
    _target_id, backend_id, _ext = detail.target_identity(directory)
    root = directory / "candidates"
    if not root.is_dir():
        _cache_put(_GRID_CACHE, key, [])
        return []
    best: dict[tuple, dict] = {}
    signed: dict[str, tuple] = {}
    order: list[tuple] = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        source_path = path / detail.target_file_names(directory)[0]
        try:
            source = source_path.read_text(encoding="utf-8")
        except OSError:
            continue
        # Only a current (schema-4) report may contribute a speedup to a cell: an old-format
        # report's number is its own memory of a measurement this panel cannot check.
        report_path = path / "report.json"
        report = detail._load(report_path)
        speed = None
        if report_state(report_path, report) == "current" and isinstance(report, dict):
            speed = report.get("speedup")
        if isinstance(speed, bool) or not isinstance(speed, (int, float)):
            speed = None
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
        signature = signed.get(digest)
        if signature is None:
            signature = _cache_get(_FEATURE_CACHE, digest)
            if signature is None:
                signature = feature_signature(source, backend_id=backend_id)
                _cache_put(_FEATURE_CACHE, digest, signature)
            signed[digest] = signature
        label = path.name.split("_", 1)[1] if "_" in path.name else path.name
        current = best.get(signature)
        if current is None or (speed is not None and (current["speedup"] is None or speed > current["speedup"])):
            if current is None:
                order.append(signature)
            best[signature] = {
                "signature": list(signature),
                "label": label,
                "speedup": speed,
                "published": published_ok and label == winner,
            }
    rows = [best[signature] for signature in order if signature in best]
    for stale in [item for item in _GRID_CACHE if item[0] == key[0] and item != key]:
        _GRID_CACHE.pop(stale, None)
    _cache_put(_GRID_CACHE, key, rows)
    return rows


def plan_outcomes(directory: Path, config: dict | None, state: dict | None, result: dict | None) -> list[dict]:
    """@brief 列出计划节点的结果；未开启 ``plan_ledger`` 时返回空列表，不含优先级与置信度。

    原始英文说明：Empty unless this run turned ``plan_ledger`` on. No priority and no confidence.

    @param directory 运行目录。
    @param config 解析后的 ``run_config.json``，可为 None。
    @param state 解析后的 ``state.json``，可为 None。
    @param result 解析后的 ``result.json``，可为 None。
    @return 每项含 ``round``、``label``、``title``、``outcome`` 与 ``speedup`` 的列表。
    """
    search = (config or {}).get("search") or {}
    if search.get("plan_ledger") is not True:
        return []
    trace = (state or {}).get("trace") or (result or {}).get("trace") or []
    history_lines = (state or {}).get("history") or (result or {}).get("history") or []
    if not isinstance(history_lines, list):
        history_lines = []
    lines = [line for line in history_lines if isinstance(line, str)]
    nodes = nodes_from_trace(trace if isinstance(trace, list) else [], lines)
    return [
        {
            "round": node.round_id,
            "label": node.label,
            "title": node.title,
            "outcome": node.outcome_kind,
            "speedup": node.speedup,
        }
        for node in nodes
    ]


# 面板戳只由该面板依赖的源指纹组成，而且只做 stat：面板的读取路径不解析配置、不摄取索引。
# 候选面板必须逐颗候选取 (mtime_ns,size)：远程 evaluation_outbox 回补报告是就地写文件，
# candidates/ 目录的 mtime 不变，只看目录会让面板永远显示旧报告。
_PANEL_VERSION = "run-panel-v2"


def _file_stamp(path: Path) -> str:
    """@brief 取单个文件的 ``(mtime_ns,size)`` 指纹，只做一次 stat。

    @param path 文件路径。
    @return 指纹字符串；读不到 stat 时为 ``0:0``。
    """
    try:
        stat = path.stat()
    except OSError:
        return "0:0"
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _panel_stamp(kind: str, parts: list[str]) -> str:
    """@brief 把面板名、版本与源指纹拼成面板戳。

    @param kind 面板名（同一份源指纹在不同面板上得到不同戳）。
    @param parts 该面板依赖的源指纹。
    @return 十六进制摘要。
    """
    return hashlib.sha256("\x1f".join([_PANEL_VERSION, kind, *parts]).encode("utf-8")).hexdigest()


def _candidate_source_parts(directory: Path) -> list[str]:
    """@brief 每颗候选的 ``report.json`` 与内核文件的 ``(mtime_ns,size)``。

    内核指纹覆盖全部 ``detail.candidate_kernel_files``（只 stat，不读 ``run_config.json``）。

    @param directory 运行目录。
    @return 每颗候选一行的指纹文本，按目录名排序。
    """
    root = directory / "candidates"
    if not root.is_dir():
        return []
    parts = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        # 覆盖全部 kernel.*：只 stat 一条“猜出来的权威名”会在混入 kernel.aaa 时监视错文件。
        kernels = ",".join(
            f"{name}:{_file_stamp(path / name)}" for name in detail.candidate_kernel_files(path)
        )
        parts.append(f"{path.name}={_file_stamp(path / 'report.json')}={kernels}")
    return parts


def _run_clock_stamps(directory: Path) -> list[str]:
    """@brief 运行状态类面板共用的三份快照指纹。

    @param directory 运行目录。
    @return ``state/result/eval_progress`` 的指纹列表。
    """
    return [
        _file_stamp(directory / "state.json"),
        _file_stamp(directory / "result.json"),
        _file_stamp(directory / "eval_progress.json"),
    ]


def _candidates_panel_stamp(
    directory: Path, *, limit: int, status: str | None, stage: str | None, cursor: str | None,
    label: str | None = None,
) -> str:
    """@brief 候选面板戳：候选逐颗指纹 + state/result + 本次页面请求形状。

    戳必须绑定响应形状（limit/status/stage/cursor）：同一份源文件在 ``limit=2`` 与 ``limit=1``
    下返回不同的行与游标，过滤与续页也是不同形状。只绑源文件会让一个形状的戳短路掉另一个形状，
    客户端拿不到应有的行。

    @param directory 运行目录。
    @param limit 规范化后的页大小。
    @param status 状态过滤。
    @param stage 阶段过滤。
    @param cursor 续页游标。
    @return 面板戳。
    """
    return _panel_stamp("candidates", [
        _file_stamp(directory / "state.json"),
        _file_stamp(directory / "result.json"),
        *_candidate_source_parts(directory),
        f"limit={limit}",
        f"status={status or ''}",
        f"stage={stage or ''}",
        f"cursor={cursor or ''}",
        f"label={label or ''}",
    ])


def _best_source_parts(directory: Path) -> list[str]:
    """@brief 顶层所有 ``best.*`` 的 ``(名字, mtime_ns, size)``，按名字排序。

    发布门读的是 ``best.<ext>``（当前仓库是 ``best.metal``）；``feature_grid`` 经
    ``conclusion_of_run`` 依赖它。只盯一个名字会在混入别的 ``best.*`` 时漏掉权威文件的改写。

    @param directory 运行目录。
    @return 指纹文本列表。
    """
    try:
        entries = sorted(
            (item for item in directory.iterdir() if item.name.startswith("best.") and item.is_file()),
            key=lambda item: item.name,
        )
    except OSError:
        return []
    return [f"{item.name}:{_file_stamp(item)}" for item in entries]


def _grid_source_signature(directory: Path) -> str:
    """@brief 特征格子缓存的身份：候选报告/内核、best 与 run_config 的源指纹。

    只按 state/result 做 key 会让“候选报告就地改写”继续返回旧的 speedup。

    @param directory 运行目录。
    @return 十六进制摘要。
    """
    parts = [
        _file_stamp(directory / "run_config.json"),
        *_best_source_parts(directory),
        *_candidate_source_parts(directory),
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _grid_panel_stamp(directory: Path) -> str:
    """@brief 特征格子：它读候选报告与内核、state/result、run_config、确认报告与 best 源码。

    @param directory 运行目录。
    @return 面板戳。
    """
    return _panel_stamp("grid", [
        _file_stamp(directory / "state.json"),
        _file_stamp(directory / "result.json"),
        _file_stamp(directory / "run_config.json"),
        _file_stamp(directory / "confirm_report.json"),
        *_best_source_parts(directory),
        *_candidate_source_parts(directory),
    ])


def _feedback_panel_stamp(
    directory: Path, *, limit: int = FEEDBACK_PAGE, cursor: str | None = None, seq: int | None = None
) -> str:
    """@brief 反馈面板：反馈日志、评审文件，以及每条记录引用的候选报告。

    页码形状（``limit``/``cursor``/``seq``）也进戳，否则不同页会共享一个戳、翻页只能拿到 ``same``。

    @param directory 运行目录。
    @param limit 本页记录数。
    @param cursor 本页游标，可为 None。
    @param seq 定位包含该序号的页（深链用）。
    @return 面板戳。
    """
    return _panel_stamp(f"feedback:{limit}:{cursor or ''}:{seq or ''}", [
        _file_stamp(directory / "feedback.jsonl"),
        _file_stamp(directory / "feedback_review.json"),
        *_candidate_source_parts(directory),
    ])


def _timeline_panel_stamp(directory: Path) -> str:
    """@brief 时间线：问模型读 transcript+ledger，等锁读机全局 GPU 事件日志。

    @param directory 运行目录。
    @return 面板戳。
    """
    return _panel_stamp("timeline", [
        *_run_clock_stamps(directory),
        _file_stamp(directory / "transcript.jsonl"),
        _file_stamp(directory / "ledger.json"),
        _file_stamp(detail.gpu_events_path()),
        _file_stamp(detail.gpu_events_path().with_suffix(".truncated")),
        *detail.gpu_shard_stamps(),
    ])


def _calibration_path(target_id: str) -> Path:
    """@brief 按目标 id 拼出标定文件路径；只 stat，不读标定内容或目标档案。

    ``tcenv.target.calibration_file`` 会先读 ``target.toml``；这里按同一优先级（环境变量 →
    用户缓存 → 目标目录旧文件）直接拼路径，保证 stamp 计算不打开任何文件。传空串得到默认目标
    的那份——``feedback.entries()`` 的 ``calibration`` 正是读它（``read_calibration()`` 不带 target）。

    @param target_id 目标 id；空串时用默认目标。
    @return 标定文件路径。
    """
    from tcenv.target import DEFAULT_TARGET_ID, TARGETS

    target_id = target_id or DEFAULT_TARGET_ID
    override = os.environ.get("TCENV_CALIBRATION")
    if override:
        return Path(override)
    cache = Path.home() / ".cache" / "ai_lowerling" / "calibration" / f"{target_id}.json"
    if cache.is_file():
        return cache
    legacy = Path(TARGETS) / target_id / "calibration.json"
    return legacy if legacy.is_file() else cache


def _same_panel(run_id: str, stamp: str, current: str) -> dict | None:
    """@brief 戳匹配时的短回路负载；不匹配时返回 None。

    @param run_id 运行 id。
    @param stamp 客户端持有的面板戳。
    @param current 现算的面板戳。
    @return 标记字典或 None。
    """
    if stamp and stamp == current:
        return {"run_id": run_id, "same": True, "stamp": current}
    return None


def _candidate_dir(directory: Path, label: str) -> tuple[Path, int | None] | None:
    """@brief 按标签找候选目录；标签的解析规则与 ``detail._candidates`` 一致。

    @param directory 运行目录。
    @param label 候选标签（不带序号前缀）。
    @return ``(目录, 序号)``；找不到时返回 None。
    """
    root = directory / "candidates"
    if not root.is_dir() or not label:
        return None
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        prefix, _, name = path.name.partition("_")
        try:
            idx: int | None = int(prefix)
        except ValueError:
            idx, name = None, path.name
        if name == label:
            return path, idx
    return None


def _after_dir_cursor(entries: list[Path], cursor: str | None) -> list[Path]:
    """@brief 取游标之后的候选目录；游标指不到任何目录时从头开始。

    @param entries 已排序的候选目录。
    @param cursor 上一页最后一颗的游标（目录名）。
    @return 剩余目录。
    """
    decoded = _decode_cursor(cursor)
    if decoded is None:
        return entries
    _value, name = decoded
    for index, path in enumerate(entries):
        if path.name == name:
            return entries[index + 1:]
    return entries


def _after_label_cursor(labels: list[str], cargo, cursor: str | None) -> list[str]:
    """@brief 取游标之后的候选标签；游标记的是候选目录名，所以过滤与未过滤的游标可互换。

    @param labels 完整标签序列。
    @param cargo ``label -> 候选目录名`` 的映射函数。
    @param cursor 上一页最后一颗的游标（目录名）。
    @return 剩余标签。
    """
    decoded = _decode_cursor(cursor)
    if decoded is None:
        return labels
    _value, name = decoded
    for index, label in enumerate(labels):
        if cargo(label) == name:
            return labels[index + 1:]
    return labels


def candidates_panel(
    run_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    limit: int = CANDIDATES_PAGE,
    cursor: str | None = None,
    stamp: str = "",
    label: str | None = None,
) -> dict | None:
    """@brief 候选面板：候选行与轨迹标记一起取，只有打开这块时才读 ``report.json``。

    过滤与游标只影响返回的行，不影响面板戳；戳覆盖全部候选，所以任何一颗被改写都会让戳变化。

    @param run_id 运行 id。
    @param status 只看该状态。
    @param stage 只看该阶段。
    @param limit 单页行数，上限 1000。
    @param cursor 上一页最后一行的游标。
    @param stamp 客户端持有的面板戳。
    @param label 定位包含该候选的页（深链用；此时忽略 status/stage 过滤）。
    @return 含 ``candidates``、``candidate_marks``、``total``、``next``、``page_index`` 与 ``stamp`` 的字典。
    """
    directory = _read_run(run_id)
    if directory is None:
        return None
    limit = max(1, min(int(limit or CANDIDATES_PAGE), 1000))
    current = _candidates_panel_stamp(directory, limit=limit, status=status, stage=stage, cursor=cursor, label=label)
    # 戳已经包含页面形状，所以任何形状都能安全短路；不同形状不会共享戳。
    same = _same_panel(run_id, stamp, current)
    if same is not None:
        return same

    root = directory / "candidates"
    entries = []
    if root.is_dir():
        # 首页必须全局 newest-first：目录名为 `NNN_label`，倒序就是序号从大到小。
        entries = sorted((item for item in root.iterdir() if item.is_dir()), reverse=True)
    by_label = {}
    for path in entries:
        prefix, _, name = path.name.partition("_")
        by_label[name if _ else path.name] = path

    page_index = None
    if label:
        # 深链定位页：按目录名在 newest-first 序列里的位置取整页，不读其它页的报告。
        cargo = by_label.get(label)
        if cargo is None:
            page = []
            page_index = None
            total = len(entries)
            next_cursor = None
        else:
            index = next(position for position, path in enumerate(entries) if path == cargo)
            page_index = index // limit
            start = page_index * limit
            page_paths = entries[start:start + limit]
            page = [detail.candidate_row(directory, path) for path in page_paths]
            total = len(entries)
            has_more = start + limit < len(entries)
            next_cursor = _encode_cursor("", page_paths[-1].name) if has_more and page_paths else None
    elif status or stage:
        # 过滤依赖每一颗的当前 status/stage：索引里的行可能比就地改写的报告旧（report 改写
        # 不改 candidates/ 目录 mtime），混用会返回状态已经变了的行、漏掉新候选。这里按当前
        # 报告过滤；无过滤的常见路径仍然只读当前页。
        scanned = [detail.candidate_row(directory, path) for path in entries]
        if status:
            scanned = [row for row in scanned if row.get("status") == status]
        if stage:
            scanned = [row for row in scanned if row.get("stage") == stage]
        labels_all = [str(row["label"]) for row in scanned]
        total = len(labels_all)
        cargo = lambda label: by_label[label].name if label in by_label else ""
        remaining = _after_label_cursor(labels_all, cargo, cursor)
        labels = remaining[:limit]
        page = [detail.candidate_row(directory, by_label[label]) for label in labels if label in by_label]
        next_cursor = _encode_cursor("", cargo(labels[-1])) if len(remaining) > limit and labels else None
    else:
        # 无过滤：目录列表就够定位本页，只读当前页的报告与内核片段（250 颗时 limit=1 只读 1 份）。
        total = len(entries)
        remaining = _after_dir_cursor(entries, cursor)
        page_paths = remaining[:limit]
        page = [detail.candidate_row(directory, path) for path in page_paths]
        next_cursor = _encode_cursor("", page_paths[-1].name) if len(remaining) > limit and page_paths else None

    trace = (detail._load(directory / "state.json") or {}).get("trace")
    if not trace:
        trace = (detail._load(directory / "result.json") or {}).get("trace")
    return {
        "run_id": run_id,
        "same": False,
        "stamp": current,
        "total": total,
        "limit": limit,
        "next": next_cursor,
        "page_index": page_index,
        "candidates": page,
        "candidate_marks": detail._candidate_marks(trace),
    }


def candidate_panel(run_id: str, label: str, *, stamp: str = "") -> dict | None:
    """@brief 单颗候选：当前（schema-4）报告的正文、有界内核片段与相对 ``runs/`` 的报告路径。

    Only a schema-4 report is current candidate evidence (§8).  An older report has no build or
    evaluation identity, so the panel marks it (``report_state``) and returns an empty ``report``
    body: its own ``status``/``speedup`` must not be shown as today's measurement.  The raw bytes
    stay reachable through the explicit ``/raw`` route, which is a debugging surface and says so.

    @param run_id 运行 id。
    @param label 候选标签。
    @param stamp 客户端持有的面板戳。
    @return 含 ``report``/``report_state``/``excerpt``/``report_path`` 的字典；运行或候选不存在时返回 None。
    """
    directory = _read_run(run_id)
    if directory is None:
        return None
    found = _candidate_dir(directory, label)
    if found is None:
        return None
    path, idx = found
    # 戳只用 stat 能拿到的东西；权威内核名要读 run_config，放到 same 短路之后。
    kernels = ",".join(
        f"{name}:{_file_stamp(path / name)}" for name in detail.candidate_kernel_files(path)
    )
    current = _panel_stamp(f"candidate:{label}", [
        f"dir={path.name}",
        _file_stamp(path / "report.json"),
        kernels,
    ])
    same = _same_panel(run_id, stamp, current)
    if same is not None:
        return {**same, "label": label}
    kernel = detail.target_file_names(directory)[0]
    report_path = path / "report.json"
    document = detail._load(report_path)
    state = report_state(report_path, document)
    return {
        "run_id": run_id,
        "label": label,
        "idx": idx,
        "same": False,
        "stamp": current,
        "report_state": state,
        "report_path": detail._rel(report_path),
        "excerpt": detail._excerpt(path / kernel) if (path / kernel).is_file() else "",
        # Current evidence only: an old/damaged report contributes no body here.
        "report": document if state == "current" else {},
    }


# 反馈列表只放索引性字段：全文（claim 正文、证据、历史、原文）走 /feedback/{seq}。
_COMPACT_CLAIM_KEYS = ("digest", "section", "audience", "premise", "status", "verification", "topic")
_COMPACT_RECORD_KEYS = (
    "seq", "t", "label", "round", "committee", "provider", "model", "effort",
    "outcome", "exit", "best_speedup", "seed_label", "kinds", "backend", "target_id",
    "truncated", "truncated_sections", "report_path",
)


def compact_feedback(document: dict) -> dict:
    """@brief 把 ``observe.feedback.entries()`` 的完整记录压成列表形状。

    每条记录只留 ``seq/label/round/kinds`` 等索引字段、候选报告摘要（证据预填要它，但不含正文）
    与每个 claim 的状态；claim 正文、evidence、history、subclaims、raw 都不进列表。

    @param document ``entries()`` 的输出。
    @return 含 ``records`` 的字典。
    """
    records = []
    for record in document.get("records") or []:
        claims = []
        counts = {"total": 0}
        for claim in record.get("claims") or []:
            row = {key: claim.get(key) for key in _COMPACT_CLAIM_KEYS}
            row["subclaims_count"] = len(claim.get("subclaims") or [])
            row["open_subclaims_count"] = len(claim.get("open_subclaims") or [])
            claims.append(row)
            status = str(claim.get("status") or "")
            counts["total"] += 1
            counts[status] = counts.get(status, 0) + 1
        row = {key: record.get(key) for key in _COMPACT_RECORD_KEYS}
        row["claims"] = claims
        row["claim_counts"] = counts
        row["report"] = record.get("report")
        records.append(row)
    return {"records": records}


def _after_seq_cursor(records: list[dict], cursor: str | None) -> list[dict]:
    """@brief 取 ``seq`` 游标之后的记录；游标记的是上一页末条的序号。

    @param records 完整的紧凑记录列表（文件顺序）。
    @param cursor 上一页最后一条的游标，可为 None。
    @return 剩余记录；游标指不到任何记录时从头开始。
    """
    decoded = _decode_cursor(cursor)
    if decoded is None:
        return records
    _value, seq = decoded
    for index, record in enumerate(records):
        if str(record.get("seq")) == str(seq):
            return records[index + 1:]
    return records


def feedback_panel(
    run_id: str, *, stamp: str = "", limit: int = FEEDBACK_PAGE, cursor: str | None = None, seq: int | None = None
) -> dict | None:
    """@brief 反馈面板：条目与评审状态都在，但每条 claim 的正文与原文不在。

    单条全文走 :func:`feedback_record`；记录按文件顺序一页 ``limit`` 条，`next` 是下一页游标。

    @param run_id 运行 id。
    @param stamp 客户端持有的面板戳。
    @param limit 本页最多几条记录（夹到 1..200）。
    @param cursor 上一页末条的游标，可为 None。
    @param seq 定位包含该序号的页（深链用）；找不到时返回空页。
    @return 含 ``feedback``（`records`/`total`/`next`/`page_index`）与 ``stamp`` 的字典；运行不存在时返回 None。
    """
    directory = _read_run(run_id)
    if directory is None:
        return None
    limit = max(1, min(int(limit or FEEDBACK_PAGE), 200))
    current = _feedback_panel_stamp(directory, limit=limit, cursor=cursor, seq=seq)
    same = _same_panel(run_id, stamp, current)
    if same is not None:
        return same
    from observe.feedback import entries

    records = compact_feedback(entries(directory, with_calibration=False))["records"]
    page_index = None
    if seq is not None:
        index = next((position for position, row in enumerate(records) if str(row.get("seq")) == str(seq)), None)
        if index is None:
            page = []
            following = None
        else:
            page_index = index // limit
            start = page_index * limit
            page = records[start:start + limit]
            has_more = start + limit < len(records)
            following = _encode_cursor(None, str(page[-1].get("seq"))) if has_more and page else None
    else:
        remaining = _after_seq_cursor(records, cursor)
        page = remaining[:limit]
        following = _encode_cursor(None, str(page[-1].get("seq"))) if len(remaining) > limit and page else None
    return {
        "run_id": run_id,
        "same": False,
        "stamp": current,
        "feedback": {"records": page, "total": len(records), "next": following, "page_index": page_index},
    }


def feedback_record(run_id: str, seq, *, stamp: str = "") -> dict | None:
    """@brief 一条反馈记录的全文（含 ``raw``、claim 正文与校准）；序号不存在时返回 None。

    @param run_id 运行 id。
    @param seq 反馈记录序号。
    @param stamp 客户端持有的面板戳。
    @return 含 ``record`` 与 ``calibration`` 的字典；运行或序号不存在时返回 None。
    """
    directory = _read_run(run_id)
    if directory is None:
        return None
    current = _panel_stamp(f"feedback:{seq}", [
        _file_stamp(directory / "feedback.jsonl"),
        _file_stamp(directory / "feedback_review.json"),
        # 负载里的 calibration 来自 entries() → read_calibration()（默认目标，不看该 run 的 target_id），
        # 所以戳必须盯同一份文件；盯 run 自己 target 的那份会让默认标定改写后仍回 same。
        _file_stamp(_calibration_path("")),
        *_candidate_source_parts(directory),
    ])
    same = _same_panel(run_id, stamp, current)
    if same is not None:
        return {**same, "seq": seq}
    from observe.feedback import entries

    document = entries(directory)
    for record in document.get("records") or []:
        if str(record.get("seq")) == str(seq):
            return {
                "run_id": run_id,
                "seq": record.get("seq"),
                "same": False,
                "stamp": current,
                "calibration": document.get("calibration"),
                "record": record,
            }
    return None


def timeline_panel(run_id: str, *, stamp: str = "") -> dict | None:
    """@brief 时间线面板：GPU 锁泳道与评测/问模型跨度，随运行与机全局事件变化。

    @param run_id 运行 id。
    @param stamp 客户端持有的面板戳。
    @return 含 ``timeline`` 与 ``stamp`` 的字典；运行不存在时返回 None。
    """
    directory = _read_run(run_id)
    if directory is None:
        return None
    current = _timeline_panel_stamp(directory)
    same = _same_panel(run_id, stamp, current)
    if same is not None:
        return same
    state = detail._load(directory / "state.json")
    result = detail._load(directory / "result.json")
    progress = detail._age(detail._load(directory / "eval_progress.json"))
    status = detail.liveness(state, result, directory)
    return {
        "run_id": run_id,
        "same": False,
        "stamp": current,
        "timeline": detail._timeline(directory, state, result, progress, status),
    }


def budget_events_panel(run_id: str, *, stamp: str = "") -> dict | None:
    """@brief 预算事件面板：可能很长，首屏不内联。

    @param run_id 运行 id。
    @param stamp 客户端持有的面板戳。
    @return 含 ``budget_events`` 与 ``stamp`` 的字典；运行不存在时返回 None。
    """
    directory = _read_run(run_id)
    if directory is None:
        return None
    current = _panel_stamp("budget-events", [_file_stamp(directory / "budget_events.json")])
    same = _same_panel(run_id, stamp, current)
    if same is not None:
        return same
    return {"run_id": run_id, "same": False, "stamp": current, "budget_events": detail._budget_events(directory)}


def llm_panel(run_id: str, *, stamp: str = "") -> dict | None:
    """@brief 在飞模型请求预览（``llm.json``），频率与回合不同，单开一块。

    @param run_id 运行 id。
    @param stamp 客户端持有的面板戳。
    @return 含 ``llm`` 与 ``stamp`` 的字典；运行不存在时返回 None。
    """
    directory = _read_run(run_id)
    if directory is None:
        return None
    current = _panel_stamp("llm", [_file_stamp(directory / "llm.json")])
    same = _same_panel(run_id, stamp, current)
    if same is not None:
        return same
    return {"run_id": run_id, "same": False, "stamp": current, "llm": detail._load(directory / "llm.json")}


def grid_panel(run_id: str, *, stamp: str = "") -> dict | None:
    """@brief 特征格子面板：按源码签名分组，每格保留最快候选。

    @param run_id 运行 id。
    @param stamp 客户端持有的面板戳。
    @return 含 ``feature_grid`` 与 ``stamp`` 的字典；运行不存在时返回 None。
    """
    directory = _read_run(run_id)
    if directory is None:
        return None
    current = _grid_panel_stamp(directory)
    same = _same_panel(run_id, stamp, current)
    if same is not None:
        return same
    return {
        "run_id": run_id,
        "same": False,
        "stamp": current,
        "feature_grid": feature_grid(directory, detail._load(directory / "result.json")),
    }


def plan_panel(run_id: str, *, stamp: str = "") -> dict | None:
    """@brief 计划结局面板：未开启 ``plan_ledger`` 时为空列表。

    @param run_id 运行 id。
    @param stamp 客户端持有的面板戳。
    @return 含 ``plan_outcomes`` 与 ``stamp`` 的字典；运行不存在时返回 None。
    """
    directory = _read_run(run_id)
    if directory is None:
        return None
    current = _panel_stamp("plan", [
        _file_stamp(directory / "state.json"),
        _file_stamp(directory / "result.json"),
        _file_stamp(directory / "run_config.json"),
    ])
    same = _same_panel(run_id, stamp, current)
    if same is not None:
        return same
    return {
        "run_id": run_id,
        "same": False,
        "stamp": current,
        "plan_outcomes": plan_outcomes(
            directory,
            detail._load(directory / "run_config.json"),
            detail._load(directory / "state.json"),
            detail._load(directory / "result.json"),
        ),
    }


RAW_KEYS = {"state": "state.json", "result": "result.json", "confirm": "confirm_report.json"}


def raw_panel(run_id: str, key: str, *, stamp: str = "") -> dict | None:
    """@brief 原始文件（``state.json`` / ``result.json`` / ``confirm_report.json``），显式调试用。

    @param run_id 运行 id。
    @param key 取 ``state`` / ``result`` / ``confirm``。
    @param stamp 客户端持有的面板戳。
    @return 含 ``doc`` 与 ``stamp`` 的字典；运行不存在或 key 不合法时返回 None。
    """
    directory = _read_run(run_id)
    if directory is None or key not in RAW_KEYS:
        return None
    path = directory / RAW_KEYS[key]
    current = _panel_stamp(f"raw:{key}", [_file_stamp(path)])
    same = _same_panel(run_id, stamp, current)
    if same is not None:
        return {**same, "key": key}
    return {
        "run_id": run_id,
        "key": key,
        "same": False,
        "stamp": current,
        "path": detail._rel(path),
        "doc": detail._load(path),
    }


def _encode_cursor(value, run_id: str) -> str:
    """@brief 把（排序键值, run_id）编成 URL 安全的游标字符串。

    @param value 排序列的取值。
    @param run_id 运行 id，用于唯一化。
    @return 去掉 ``=`` 填充的 base64url 文本。
    """
    raw = json.dumps([value, run_id], separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str | None) -> tuple[object, str] | None:
    """@brief 解析 keyset 游标；非法或缺失时返回 None。

    @param cursor 客户端回传的游标，可为 None。
    @return 二元组（排序键值, run_id），无法解析时返回 None。
    """
    if not cursor:
        return None
    try:
        padded = cursor + "=" * ((4 - len(cursor) % 4) % 4)
        value, run_id = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        return value, str(run_id)
    except (ValueError, TypeError, UnicodeError):
        return None


# 页面与旧 CLI 的 22 列（S1b 前后行为不变）。
_RUN_SELECT = (
    "run_id, experiment_id, arm_id, trial, task, dtype, model, target_id, status, stop_reason, "
    "publication, published_speedup, spent_usd, budget_usd, source, binding, kernel_sha256, protocol_version, "
    "CASE WHEN protocol_version = 'flame_chase_1' THEN '2' ELSE protocol_version END AS protocol_epoch, "
    "committees_json, started_at, updated_at"
)

# §6.1 的 L0 窄投影。plan 的七字段（再加 task/spent_usd/updated_at）在真实的 73 字符 run id 下最坏
# 276B，超过 200B/行；plan §0.3 第 6 条明确“字段放不下就砍字段，不改预算”，所以这里只留四个：
# 句柄（run_id）、可发布性（publication 短码，legacy 与 unconfirmed 的 published_speedup 都是 null，
# 靠它区分）、已发布数值、以及摄入时写好的停止码（冒号前第一段、≤24 字符）。task 走 ``--task`` 过滤、
# spent_usd/updated_at/完整 stop_reason 都在 ``--wide`` 里。
# 发布口径的原始值（``conclusion_of_run`` 的返回值）与 L0 短码的映射。过滤与投影共用这一张表：
# 只读时映射、过滤却拿原始列比，会让 ``--publication legacy`` 静默返回 0 行（一次真实回归）。
PUBLICATION_CODES = {
    "旧版未绑定源码": "legacy",
    "未确认": "unconfirmed",
    "不确定": "uncertain",
    "confirmed": "confirmed",
    "in_progress": "in_progress",
}
PUBLICATION_RAW_VALUES: dict[str, tuple[str, ...]] = {}
for _raw, _code in PUBLICATION_CODES.items():
    PUBLICATION_RAW_VALUES[_code] = PUBLICATION_RAW_VALUES.get(_code, ()) + (_raw,)

_L0_PUBLICATION_CASE = (
    "CASE publication "
    + " ".join(f"WHEN '{raw}' THEN '{code}'" for raw, code in PUBLICATION_CODES.items())
    + " ELSE COALESCE(publication, 'in_progress') END"
)

_RUN_L0_SELECT = (
    "run_id, "
    + _L0_PUBLICATION_CASE + " AS publication, "
    "published_speedup, COALESCE(stop_code, '') AS stop_code"
)
# L0 响应里的字段集合。排序键（started_at/spent_usd/updated_at）不在其中：keyset 游标要它们，
# 但不能把宽行的列泄漏进窄响应，所以在 SQL 里另取一个内部别名、生成游标后再删掉。
_RUN_L0_FIELDS = frozenset({"run_id", "publication", "published_speedup", "stop_code"})
_CURSOR_ALIAS = "_cursor_value"

def runs_page(
    *,
    experiment_id: str | None = None,
    task: str | None = None,
    dtype: str | None = None,
    model: str | None = None,
    status: str | None = None,
    target: str | None = None,
    publication: str | None = None,
    min_speedup: float | None = None,
    since: float | None = None,
    sort: str = "started",
    order: str = "desc",
    cursor: str | None = None,
    limit: int = 50,
    offset: int = 0,
    compact: bool = False,
) -> dict:
    """@brief 返回一页运行；游标在 ``(排序列, run_id)`` 上做 keyset，``offset`` 是旧的退化方式。

    ``compact`` 时只 SELECT L0 的窄列（§6.1：``run_id/publication/published_speedup/stop_code``；
    字段集合按 plan §0.3 第 6 条“字段放不下就砍字段”缩到能守住 200B/行），不打 ``row_json``，
    并把长中文发布状态换成英文短码；默认保持 HTTP 页面与旧 CLI 的 22 列不变。

    原始英文说明：One page of runs. A cursor is keyset on ``(sort column, run_id)``; ``offset`` is the old fallback.

    @param experiment_id 只取该实验的运行。
    @param task 只取该任务名的运行。
    @param dtype 只取该 dtype 的运行。
    @param model 只取该模型的运行。
    @param status 只取该状态的运行。
    @param target 只取该 target 的运行。
    @param publication 只取该发布状态的运行。
    @param min_speedup 已发布加速比下限。
    @param since 开始时间下限。
    @param sort 排序列名，取 ``started`` / ``speedup`` / ``spent`` / ``updated``。
    @param order 排序方向，``asc`` 或 ``desc``。
    @param cursor 上一页返回的 keyset 游标。
    @param limit 本页最大行数，上限 100。
    @param offset 未用游标时的偏移。
    @param compact 只返回 L0 窄投影（run_id / 短发布码 / published_speedup / stop_code）。
    @return 含 ``total``、``limit``、``offset``、``next``、``sort``、``order`` 与 ``runs`` 的字典。
    """
    if _refresh_on_read():
        tick()
    limit = max(1, min(int(limit or 50), PAGE_LIMIT))
    offset = max(0, int(offset or 0))
    column = _SORTS.get(str(sort or "started"), "started_at")
    direction = "ASC" if str(order or "desc").lower() == "asc" else "DESC"
    clauses = ["status != 'missing'"]
    args: list = []
    for name, value in (("experiment_id", experiment_id), ("task", task), ("dtype", dtype), ("model", model), ("status", status), ("target_id", target)):
        if value:
            clauses.append(f"{name} = ?")
            args.append(value)
    if publication:
        # 过滤值可能是 L0 短码（legacy/unconfirmed/uncertain/…），也可能有人直接给了原始口径；
        # 短码必须展开成它对应的全部原始值（一个短码可能对应两个原始短语）。
        wanted = PUBLICATION_RAW_VALUES.get(str(publication), (str(publication),))
        marks = ",".join("?" for _ in wanted)
        clauses.append(f"publication IN ({marks})")
        args.extend(wanted)
    if min_speedup is not None:
        try:
            clauses.append("COALESCE(published_speedup, -1) >= ?")
            args.append(float(min_speedup))
        except (TypeError, ValueError):
            pass
    if since is not None:
        try:
            clauses.append("COALESCE(started_at, 0) >= ?")
            args.append(float(since))
        except (TypeError, ValueError):
            pass
    where = " AND ".join(clauses)
    page_args = list(args)
    page_offset = offset
    decoded = _decode_cursor(cursor)
    if decoded is not None:
        value, run_id = decoded
        op = "<" if direction == "DESC" else ">"
        where += f" AND (COALESCE({column}, -1) {op} ? OR (COALESCE({column}, -1) = ? AND run_id {op} ?))"
        page_args.extend([value, value, run_id])
        page_offset = 0
    select = _RUN_SELECT if not compact else _RUN_L0_SELECT
    hidden = compact and column not in _RUN_L0_FIELDS
    if hidden:
        select += f", {column} AS {_CURSOR_ALIAS}"
    with borrow() as index:
        total = index.conn.execute(f"SELECT COUNT(*) FROM runs WHERE {' AND '.join(clauses)}", args).fetchone()[0]
        rows = [
            dict(row)
            for row in index.conn.execute(
                f"SELECT {select} "
                f"FROM runs WHERE {where} ORDER BY COALESCE({column}, -1) {direction}, run_id {direction} LIMIT ? OFFSET ?",
                [*page_args, limit, page_offset],
            )
        ]
    following = None
    if len(rows) == limit:
        last = rows[-1]
        value = last.get(column) if column in last else last.get(_CURSOR_ALIAS)
        following = _encode_cursor(-1 if value is None else value, str(last["run_id"]))
    if hidden:
        # 内部排序键不出现在响应里：§6.1 的 L0 只有四个字段（§0.3 第 6 条：放不下就砍字段）。
        for row in rows:
            row.pop(_CURSOR_ALIAS, None)
    return {"total": total, "limit": limit, "offset": offset, "next": following, "sort": sort, "order": direction.lower(), "runs": rows}


def candidates_page(run_id: str, *, status: str | None = None, limit: int = 200) -> dict:
    """@brief 从索引返回候选级事实，不打开任何报告文件。

    原始英文说明：Candidate-level facts from the index. No report files are opened.

    @param run_id 运行 id。
    @param status 只看该状态的候选，None 表示不限。
    @param limit 最大行数，上限 1000。
    @return 含 ``run_id``、``limit`` 与 ``candidates`` 的字典。
    """
    tick()
    limit = max(1, min(int(limit or 200), 1000))
    clauses = ["run_id = ?"]
    args: list = [run_id]
    if status:
        clauses.append("status = ?")
        args.append(status)
    with borrow() as index:
        rows = [dict(row) for row in index.conn.execute(
            "SELECT run_id, idx, label, status, stage, speedup, within_noise, eligible, kernel_sha256, updated_at, "
            "report_state, build_shape, build_kind, evaluation_fingerprint "
            f"FROM candidates WHERE {' AND '.join(clauses)} ORDER BY idx LIMIT ?",
            [*args, limit],
        )]
    return {"run_id": run_id, "limit": limit, "candidates": rows}


def _session_source_fingerprint(directory: Path) -> str:
    """@brief 只 stat transcript，得到便宜的源指纹；不打开文件。"""
    path = directory / "transcript.jsonl"
    try:
        stat = path.stat() if path.is_file() else None
    except OSError:
        stat = None
    return f"{stat.st_mtime_ns}:{stat.st_size}" if stat is not None else "0:0"


def _sessions_stamp(source: str, total: int, limit: int) -> str:
    """@brief 由源指纹、总会话数和页大小组成便宜戳；超过 limit 后的新增也会改变它。"""
    payload = f"{_SESSIONS_FEED_VERSION}|{source}|{total}|{limit}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sessions_page(run_id: str, *, limit: int = 500, stamp: str = "") -> dict:
    """@brief 每个 hmz 会话一行：委员会、模型、token 与花费；``stamp`` 相同则不回正文。

    原始英文说明：One row per hmz session: committee, model, tokens and cost.

    @param run_id 运行 id。
    @param limit 最大行数，上限 2000。
    @param stamp 客户端持有的会话指纹。
    @return 含 ``run_id``、``limit``、``same``、``stamp`` 与 ``sessions`` 的字典。
    """
    tick()
    limit = max(1, min(int(limit or 500), 2000))
    directory = _read_run(run_id)
    source = _session_source_fingerprint(directory) if directory is not None else "0:0"
    with borrow() as index:
        projected = None
        if directory is not None:
            row = index.conn.execute("SELECT value FROM meta WHERE key = ?", (f"sessions_fp:{run_id}",)).fetchone()
            projected = row[0] if row is not None else None
        total = int(index.conn.execute("SELECT COUNT(*) FROM sessions WHERE run_id = ?", (run_id,)).fetchone()[0])
    current = _sessions_stamp(source, total, limit)
    if directory is not None and projected == source and stamp and stamp == current:
        return {"run_id": run_id, "limit": limit, "same": True, "stamp": current, "sessions": []}
    if directory is not None:
        source = get_index().ensure_sessions(directory)
        with borrow() as index:
            total = int(index.conn.execute("SELECT COUNT(*) FROM sessions WHERE run_id = ?", (run_id,)).fetchone()[0])
        current = _sessions_stamp(source, total, limit)
        if stamp and stamp == current:
            return {"run_id": run_id, "limit": limit, "same": True, "stamp": current, "sessions": []}
    with borrow() as index:
        rows = [dict(row) for row in index.conn.execute(
            "SELECT run_id, seq, label, candidate_label, attempt, committee, provider, model, outcome, cost_usd, "
            "failure_sig, session_id, failed_turns, last_kind, failure_class, error, tool_calls, diagnostics_path, "
            "tokens_input, tokens_output, tokens_cache_read, tokens_cache_write "
            "FROM sessions WHERE run_id = ? ORDER BY seq LIMIT ?",
            (run_id, limit),
        )]
    for row in rows:
        row.update(efficiency_fields(
            tokens_cache_read=row.get("tokens_cache_read"), tokens_output=row.get("tokens_output"),
            cost_usd=row.get("cost_usd"),
        ))
    return {"run_id": run_id, "limit": limit, "same": False, "stamp": current, "sessions": rows}


def model_stats(run_id: str | None = None, *, limit: int = 50) -> dict:
    """@brief 按会话聚合 provider/模型用量，可只看一个运行。

    原始英文说明：Per-session model totals, so the first-committee ``runs.model`` is not the only view.

    @param run_id 可选的运行 id；缺省聚合全部运行。
    @param limit 最多返回多少个模型行，上限 ``PAGE_LIMIT``；按花费与会话数降序取前几个。
    @return 含 ``models`` 列表的字典（按花费与会话数降序）。
    """
    limit = max(1, min(int(limit or 50), PAGE_LIMIT))
    tick()
    where = " WHERE run_id = ?" if run_id else ""
    args = (run_id,) if run_id else ()
    if run_id:
        directory = _read_run(run_id)
        if directory is not None:
            get_index().ensure_sessions(directory)
    else:
        # 全局 models 也要看到活跃 run 的新会话；活跃数很小，历史 run 不在每次请求里全量追平。
        with borrow() as index:
            running = [row[0] for row in index.conn.execute("SELECT run_id FROM runs WHERE status = 'running'")]
        for active in running:
            directory = _read_run(active)
            if directory is not None:
                get_index().ensure_sessions(directory)
    with borrow() as index:
        rows = [dict(row) for row in index.conn.execute(
            "SELECT committee, provider, model, COUNT(*) AS sessions, ROUND(SUM(COALESCE(cost_usd, 0)), 6) AS cost_usd, "
            "COUNT(cost_usd) AS _cost_values, "
            "SUM(tokens_input) AS input_tokens, SUM(tokens_output) AS output_tokens, "
            "SUM(tokens_cache_read) AS cache_read_tokens, SUM(tokens_cache_write) AS cache_write_tokens "
            f"FROM sessions{where} GROUP BY committee, provider, model ORDER BY cost_usd DESC, sessions DESC "
            "LIMIT ?",
            [*args, limit],
        )]
    for row in rows:
        cost_values = row.pop("_cost_values")
        row["sessions_with_cost"] = cost_values
        cost = row["cost_usd"] if cost_values else None
        row.update(efficiency_fields(
            tokens_cache_read=row.get("cache_read_tokens"), tokens_output=row.get("output_tokens"),
            cost_usd=cost, sessions=row.get("sessions"),
        ))
    return {"models": rows}


# §6.4 的巡检读模型：固定键、无嵌套自由文本、字节级截断。先按项截断（簇 5、best 8、alerts 10），
# 仍然超 1,536B 就逐个丢弃最不重要的项（best → 簇 → alerts）并置 truncated。
PATROL_BYTES = 1536
PATROL_CLUSTERS = 5
PATROL_BEST = 8
PATROL_ALERTS = 10
# 协议纪元是最容易漏掉的一栏：它是 runs 的普通文本列，真实语料只有 2/3/legacy，但库里的取值
# 数量与长度都没有上限。固定元数据也必须受 1,536B 约束，否则“逐项丢弃”救不回来。
PATROL_EPOCHS = 8
PATROL_EPOCH_CHARS = 24
# 告警只受一个全局配额约束：先按类别优先级把所有候选排成一条序列（发布 → 预算 → 会话 → 评测），
# 再取前 PATROL_ALERTS 条。分类配额会把“5 条发布只留 2 条、回退挤掉丢失/不一致”变成静默丢弃；
# 真实语料里 5 条发布 + 3 条预算只有 1,097B，远在 1,536B 之内。


def _patrol_epochs(values: list[str]) -> tuple[list[str], bool]:
    """@brief 有界协议纪元：每项至多 ``PATROL_EPOCH_CHARS`` 字符、总共至多 ``PATROL_EPOCHS`` 项。

    截断后去重（不同长版本可能截成同一个前缀）；任何丢掉字符/丢掉项都算截断，如实报告。

    @param values 索引里 distinct 的协议纪元（已排序）。
    @return ``(纪元列表, 是否截断)``。
    """
    kept: list[str] = []
    dropped = False
    for value in values:
        text = str(value or "")
        if not text:
            continue
        if len(text) > PATROL_EPOCH_CHARS:
            dropped = True
            text = text[:PATROL_EPOCH_CHARS]
        if text not in kept:
            kept.append(text)
    if len(kept) > PATROL_EPOCHS:
        dropped = True
        kept = kept[:PATROL_EPOCHS]
    return kept, dropped


def _doc_bytes(document: dict) -> int:
    """@brief 与测试口径一致的 UTF-8 字节数（不小报中文）。

    @param document 可 JSON 序列化的对象。
    @return 序列化后的字节数。
    """
    return len(json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _patrol_clusters(index) -> tuple[int, list[dict], bool]:
    """@brief 从 ``failure_members`` 派生失败总数与最大几簇，并给每簇一条可复现例子。

    簇计数永远 ``GROUP BY sig`` 派生；例子只查一条成员（带 label 与归一化后的 message），
    不解码 ``row_json``、不打开报告。多取一行只为判断“是否还有更多”，不当作真的多显示一条。

    @param index 打开的索引。
    @return ``(总失败数, 簇列表, 是否因上限截断)``。
    """
    total = int(index.conn.execute(
        "SELECT COUNT(*) FROM failure_members WHERE sig IS NOT NULL"
    ).fetchone()[0] or 0)
    rows = index.conn.execute(
        "SELECT sig, COUNT(*) AS n, MIN(observed_at) AS first, MAX(observed_at) AS last"
        " FROM failure_members WHERE sig IS NOT NULL"
        " GROUP BY sig ORDER BY n DESC, sig LIMIT ?",
        (PATROL_CLUSTERS + 1,),
    ).fetchall()
    capped = len(rows) > PATROL_CLUSTERS
    rows = rows[:PATROL_CLUSTERS]
    clusters = []
    for row in rows:
        example = index.conn.execute(
            "SELECT m.run_id, c.label, m.status, m.stage, c.message"
            " FROM failure_members m LEFT JOIN candidates c ON c.run_id = m.run_id AND c.idx = m.idx"
            " WHERE m.sig = ? ORDER BY COALESCE(m.observed_at, 0) DESC, m.run_id, m.idx LIMIT 1",
            (row["sig"],),
        ).fetchone()
        example = dict(example) if example is not None else {}
        clusters.append({
            "sig": row["sig"],
            "name": cluster_name(row["sig"], example.get("status", ""), example.get("stage", ""), example.get("message", "")),
            "n": int(row["n"] or 0),
            "first": row["first"],
            "last": row["last"],
            "example": {"run": example.get("run_id") or "", "label": example.get("label") or ""},
        })
    return total, clusters, capped


def _patrol_alerts(index, now: float, publication: list[dict] | None = None,
                   publication_counts: dict[str, int] | None = None) -> tuple[list[dict], bool]:
    """@brief 需要注意的项：发布、预算、会话失败（带候选 label）、待评测不可用（只有 run）。

    预算判定复用预算横幅的 ``summarize_columns``（同一套阈值与“已加预算撤销耗尽”规则），
    但它读的是窄列；会话失败给 ``run``+``label``，预算与评测类没有候选 label，也不编造一个。

    告警按**一个全局配额**填充：把全部候选按类别优先级排成一条序列，取前 ``PATROL_ALERTS`` 条；
    不再按类别预切（那会让 5 条发布只剩 2 条、回退挤掉丢失/不一致）。``truncated`` 只在完整候选数
    超过全局项数或完整发布计数超过样本时置真。

    @param index 打开的索引。
    @param now 当前时间戳（本函数暂不使用，保留签名供以后按陈旧度排序）。
    @param publication 已经算好的发布类告警（在 ``borrow()`` 之外算，避免写锁重入）。
    @param publication_counts 三类发布告警的完整条数；None 表示按样本长度判。
    @return ``(告警列表, 是否截断)``。
    """
    publication = list(publication or [])
    flagged = summarize_columns(index.conn.execute(BUDGET_FLAGGED_SQL).fetchall()).get("flagged") or []
    budget_rows = []
    for item in flagged[:PATROL_ALERTS + 1]:
        amount = item.get("over_usd") if item.get("level") == "over" else item.get("spent_usd")
        budget_rows.append({
            "kind": f"budget_{item.get('level')}", "run": item.get("id"),
            "amount_usd": round(float(amount or 0.0), 4),
        })
    # 会话失败的存在条件是 outcome，不是 failure_sig：`session_failure_signature` 在
    # agent_failed 但 failure_class/error 都为空时返回 NULL，真实语料里 6 条 agent_failed 全是 NULL，
    # 用签名过滤会整类漏报。签名只用于聚类。JOIN runs 且排除 missing：物理删除只清
    # runs/candidates，sessions 行会留下来，不排除会为已删除的 run 发无法定位的告警。
    failure_outcomes = sorted(SESSION_FAILURE_OUTCOMES)
    placeholders = ", ".join("?" for _ in failure_outcomes)
    session_rows = [
        {"kind": "session_failure", "run": row["run_id"], "label": row["candidate_label"] or "",
         "n": int(row["n"] or 0)}
        for row in index.conn.execute(
            "SELECT s.run_id, s.candidate_label, COUNT(*) AS n FROM sessions s"
            " JOIN runs r ON r.run_id = s.run_id"
            f" WHERE s.outcome IN ({placeholders}) AND r.status != 'missing'"
            " GROUP BY s.run_id, s.candidate_label ORDER BY n DESC, s.run_id LIMIT ?",
            (*failure_outcomes, PATROL_ALERTS + 1),
        )
    ]
    eval_rows = [
        {"kind": "eval_unavailable", "run": row["run_id"], "n": int(row["unavailable"] or 0)}
        for row in index.conn.execute(
            "SELECT p.run_id, p.unavailable FROM run_pending_eval p"
            " JOIN runs r ON r.run_id = p.run_id"
            " WHERE p.unavailable > 0 AND r.status != 'missing'"
            " ORDER BY p.unavailable DESC, p.run_id LIMIT ?",
            (PATROL_ALERTS + 1,),
        )
    ]
    # 优先级：发布类（最可行动）→ 预算 → 会话 → 评测；全局配额对整条序列生效。
    candidates = publication + budget_rows + session_rows + eval_rows
    publication_total = sum(publication_counts.values()) if publication_counts else len(publication)
    dropped = len(candidates) > PATROL_ALERTS or publication_total > len(publication)
    return candidates[:PATROL_ALERTS], dropped


def patrol() -> dict:
    """@brief §6.4 的巡检读模型：一屏判定，字节有界；不摄取索引、不解码 ``row_json``。

    只读窄列与派生表（``runs``/``confirmed_best``/``failure_members``/``sessions``/``run_pending_eval``/
    ``kernel_history``），不调 ``tick()``、不调 ``pending_evaluation()``、不打开任何 run 目录。发布类
    告警同走索引投影（``runs.publication`` / ``publication_reason`` / ``result_confirmed_speedup``），
    统计基于完整目录史而不是面板的 200 行。超过 ``PATROL_BYTES`` 时逐个丢弃最不重要的项，并在
    ``truncated`` 里报出来。

    @return 固定键的巡检字典。
    """
    now = time.time()
    # §11.4 选 (a)：冷建不投影 sessions；巡检读 sessions 之前先把“有 transcript 但从未投影”的
    # run 懒投影一次（每场只做一次，之后 sessions_fp 让它不再读文件）。这不调用 ``_ingest``。
    scan_index = get_index()
    pending = scan_index.conn.execute(
        "SELECT r.run_id FROM runs r "
        "JOIN meta t ON t.key = 'transcript_bytes:' || r.run_id "
        "LEFT JOIN meta s ON s.key = 'sessions_fp:' || r.run_id "
        "WHERE r.status != 'missing' AND CAST(t.value AS INTEGER) > 0 AND s.key IS NULL"
    ).fetchall()
    for row in pending:
        directory = detail.resolve_run(row["run_id"])
        if directory is not None:
            scan_index.ensure_sessions(directory)
    # 目录史是巡检的数据源；这里按目录史的 stat 增量投影（没变化时只做几次 stat），
    # 不读 run 目录、不调用 _ingest。发布告警在 borrow() 之外算（写锁可重入，且不拖长读窗口）。
    scan_index.ensure_kernel_history()
    from observe.publications import publication_alert_report

    publication_report = publication_alert_report()
    with borrow() as index:
        revision = index.revision()
        live_runs = int(index.conn.execute("SELECT COUNT(*) FROM runs WHERE status = 'running'").fetchone()[0] or 0)
        epochs, epochs_dropped = _patrol_epochs([
            str(row[0]) for row in index.conn.execute(
                "SELECT DISTINCT CASE WHEN protocol_version = 'flame_chase_1' THEN '2' ELSE protocol_version END"
                " FROM runs WHERE status != 'missing' AND protocol_version != '' ORDER BY 1"
            )
        ])
        total, clusters, clusters_dropped = _patrol_clusters(index)
        alerts, alerts_dropped = _patrol_alerts(
            index, now, publication_report["alerts"], publication_report["counts"],
        )
        # §6.4 的数据源是 kernel_history（目录史）：每个 (task, target) 一行——当前最佳取该组
        # confirmed_speedup 的最大值，最后确认取该组 updated_at 的最大值（发布时写下的真实确认时刻，
        # 不是 run 的开始时间；发布丢失时 confirmed_best 会删行，目录史不会）。
        best_rows = index.conn.execute(
            "SELECT task, target_id, MAX(confirmed_speedup) AS best, MAX(updated_at) AS last_confirmed"
            " FROM kernel_history WHERE confirmed_speedup IS NOT NULL"
            " GROUP BY task, target_id ORDER BY best DESC, task, target_id LIMIT ?",
            (PATROL_BEST + 1,),
        ).fetchall()
        best_capped = len(best_rows) > PATROL_BEST
        best = [
            {
                "task": row["task"] or "",
                "target": row["target_id"] or "",
                "published_speedup": round(float(row["best"]), 4),
                "confirmed_at": row["last_confirmed"],
                "stale_days": round((now - float(row["last_confirmed"])) / 86400.0, 1)
                if row["last_confirmed"] is not None else None,
            }
            for row in best_rows[:PATROL_BEST]
        ]
    truncated = {"clusters": clusters_dropped, "best": best_capped, "alerts": alerts_dropped,
                 "epochs": epochs_dropped}
    document = {
        "schema_version": "1",
        "at": int(now),
        "index_revision": revision,
        "protocol_epochs": epochs,
        "alerts": alerts,
        "live": {"runs": live_runs, "evals": len(detail.live_evals()), "gpu": detail.gpu_lock_counts()},
        "failing": {"total": total, "clusters": clusters},
        "best": best,
        "truncated": truncated,
    }
    # 逐项丢弃，按“读者最不需要先丢”的顺序：best（已发布数字最耐放）→ 失败簇 → 告警 → 协议纪元
    # （纪要先被 `_patrol_epochs` 截到 ≤8×24 字符，本身已经有界，最后才轮到它）。每个列表保底 1 条，
    # 只有在都只剩 1 条仍超预算时才清零（只可能发生在单个 id 就超预算的极端输入上），并照样在
    # truncated 里报出来。
    sections = (
        ("best", document["best"]),
        ("clusters", document["failing"]["clusters"]),
        ("alerts", document["alerts"]),
        ("epochs", document["protocol_epochs"]),
    )
    while _doc_bytes(document) > PATROL_BYTES:
        target = next(((name, rows) for name, rows in sections if len(rows) > 1), None)
        if target is None:
            target = next(((name, rows) for name, rows in sections if rows), None)
        if target is None:
            break
        name, rows = target
        rows.pop()
        truncated[name] = True
    return document


SEARCH_LIMIT = 20


def _match_query(raw: str) -> str:
    """@brief 把用户输入变成 FTS5 的 MATCH 表达式：每个词加引号，避免语法字符报错。

    空输入返回空串；调用方直接给空结果，不把 ``MATCH ''`` 交给 SQLite。

    @param raw 用户输入。
    @return MATCH 表达式。
    """
    tokens = [token for token in str(raw or "").split() if token]
    return " ".join('"' + token.replace('"', '""') + '"' for token in tokens)


def search(*, q: str, kind: str = "runs", limit: int = SEARCH_LIMIT) -> dict:
    """@brief §10.3 的跨表搜索：runs/candidates 走 FTS5 投影，失败簇走小表直接匹配。

    只读索引，不调 ``tick()``、不解码 ``row_json``、不打开 run 目录。``kind=runs`` 搜运行标识与
    元数据；``kind=candidates`` 搜候选标签与失败 ``message``；``kind=clusters`` 搜簇名/签名。

    @param q 搜索词。
    @param kind ``runs`` / ``candidates`` / ``clusters``。
    @param limit 最多返回多少行（1–50）。
    @return 含 ``kind``/``q``/``rows``/``total`` 的字典。
    """
    kind = kind if kind in ("runs", "candidates", "clusters") else "runs"
    capped = max(1, min(int(limit or SEARCH_LIMIT), 50))
    match = _match_query(q)
    if not match:
        return {"schema_version": "1", "kind": kind, "q": str(q or ""), "rows": [], "total": 0}
    if kind == "clusters":
        document = failure_clusters(limit=200)
        needle = str(q or "").strip().lower()
        rows = [
            cluster for cluster in document.get("clusters", [])
            if needle in str(cluster.get("name") or "").lower()
            or needle in str(cluster.get("sig") or "").lower()
        ]
        return {
            "schema_version": "1", "kind": "clusters", "q": str(q or ""),
            "rows": rows[:capped], "total": len(rows),
        }
    with borrow() as index:
        if kind == "runs":
            total = int(index.conn.execute(
                "SELECT COUNT(*) FROM runs_fts JOIN runs r ON r.rowid = runs_fts.rowid"
                " WHERE runs_fts MATCH ? AND r.status != 'missing'",
                (match,),
            ).fetchone()[0] or 0)
            rows = [
                dict(row) for row in index.conn.execute(
                    "SELECT r.run_id, r.task, r.dtype, r.target_id, r.model, r.stop_reason, r.publication"
                    " FROM runs_fts JOIN runs r ON r.rowid = runs_fts.rowid"
                    " WHERE runs_fts MATCH ? AND r.status != 'missing'"
                    " ORDER BY rank, r.run_id LIMIT ?",
                    (match, capped),
                )
            ]
        else:
            total = int(index.conn.execute(
                "SELECT COUNT(*) FROM cands_fts JOIN candidates c ON c.rowid = cands_fts.rowid"
                " JOIN runs r ON r.run_id = c.run_id"
                " WHERE cands_fts MATCH ? AND r.status != 'missing'",
                (match,),
            ).fetchone()[0] or 0)
            rows = [
                dict(row) for row in index.conn.execute(
                    "SELECT c.run_id, c.idx, c.label, c.status, c.stage, c.message"
                    " FROM cands_fts JOIN candidates c ON c.rowid = cands_fts.rowid"
                    " JOIN runs r ON r.run_id = c.run_id"
                    " WHERE cands_fts MATCH ? AND r.status != 'missing'"
                    " ORDER BY rank, c.run_id, c.idx LIMIT ?",
                    (match, capped),
                )
            ]
    return {"schema_version": "1", "kind": kind, "q": str(q or ""), "rows": rows, "total": total}


def failure_clusters(*, task: str | None = None, stage: str | None = None, limit: int = 20) -> dict:
    """@brief 按签名聚合失败簇（只从 ``failure_members`` 派生，不写墓碑）。

    成员表在 ``_upsert_candidates`` 同一事务里维护，所以重复摄取不会抬高 ``n``。示例坐标来自
    **一条完整的成员行**（同一 run/idx 再 join 到同一颗候选），不会把不同行的 run_id/idx/label
    拼在一起；找不到完整行时不编造示例。``cache_hit`` 可空，所以返回覆盖率而不是比例。

    @param task 只看该任务（按 ``runs.task``）。
    @param stage 只看该阶段（按成员的 ``stage``）。
    @param limit 最大簇数，上限 100。
    @return 含 ``limit``、``total_failures``、``cache_hit_known``/``cache_hit_total`` 与 ``clusters`` 的字典。
    """
    tick()
    limit = max(1, min(int(limit or 20), PAGE_LIMIT))
    clauses = ["r.status != 'missing'"]
    args: list = []
    if task:
        clauses.append("r.task = ?")
        args.append(task)
    if stage:
        clauses.append("m.stage = ?")
        args.append(stage)
    where = " AND ".join(clauses)
    with borrow() as index:
        total = index.conn.execute(
            "SELECT COUNT(*) FROM failure_members m JOIN runs r ON r.run_id = m.run_id "
            f"WHERE {where}",
            args,
        ).fetchone()[0]
        cache = index.conn.execute(
            "SELECT COUNT(c.cache_hit) AS known, COUNT(*) AS total"
            " FROM candidates c JOIN runs r ON r.run_id = c.run_id"
            " WHERE c.failure_sig IS NOT NULL AND r.status != 'missing'"
            + (" AND r.task = ?" if task else ""),
            [task] if task else [],
        ).fetchone()
        counts = [dict(row) for row in index.conn.execute(
            "SELECT m.sig AS sig, COUNT(*) AS n, MIN(m.status) AS status, MIN(m.stage) AS stage"
            " FROM failure_members m JOIN runs r ON r.run_id = m.run_id"
            f" WHERE {where} GROUP BY m.sig ORDER BY n DESC, m.sig LIMIT ?",
            [*args, limit],
        )]
        clusters = []
        for row in counts:
            # 一个完整成员行：同一 (run_id, idx) 的候选，名称回退也用它自己的 message。
            example = index.conn.execute(
                "SELECT m.run_id AS run_id, m.idx AS idx, m.status AS status, m.stage AS stage,"
                " c.label AS label, c.message AS message"
                " FROM failure_members m JOIN runs r ON r.run_id = m.run_id"
                " JOIN candidates c ON c.run_id = m.run_id AND c.idx = m.idx"
                f" WHERE m.sig = ? AND {where}"
                " ORDER BY COALESCE(m.observed_at, 0) DESC, m.run_id, m.idx LIMIT 1",
                [row["sig"], *args],
            ).fetchone()
            if example is not None:
                name = cluster_name(
                    row["sig"], example["status"] or "", example["stage"] or "", example["message"] or ""
                )
                item = {
                    "sig": row["sig"], "n": int(row["n"] or 0), "name": name,
                    "run_id": example["run_id"], "idx": int(example["idx"]), "label": example["label"] or "",
                }
            else:
                item = {
                    "sig": row["sig"], "n": int(row["n"] or 0),
                    "name": cluster_name(row["sig"], row["status"] or "", row["stage"] or "", ""),
                    "run_id": None, "idx": None, "label": "",
                }
            clusters.append(item)
    return {
        "limit": limit,
        "total_failures": int(total or 0),
        "cache_hit_known": int(cache["known"] or 0),
        "cache_hit_total": int(cache["total"] or 0),
        "clusters": clusters,
    }


def _candidate_relative(run_id: str, idx, label: str) -> str:
    """@brief 候选目录的相对路径（``<run>/candidates/<三位序号>_<label>``）。

    @param run_id 运行 id（相对 ``runs/``）。
    @param idx 候选序号，未知时为 None 或负数。
    @param label 候选标签。
    @return 相对目录路径；序号未知时直接拼 label。
    """
    if isinstance(idx, int) and not isinstance(idx, bool) and idx >= 0:
        folder = f"{idx:03d}_{label}" if label else f"{idx:03d}"
    else:
        folder = str(label or "")
    return f"{run_id}/candidates/{folder}"


def failure_cluster(sig: str, *, limit: int = 20) -> dict | None:
    """@brief 一个失败簇的证据行：候选、消息与可复现的 ``report.json`` / ``feedback.txt`` 路径。

    只从索引读；详情按 ``candidates.failure_sig`` 走 ``candidates_sig`` 索引。零成员返回 None，
    HTTP 层据此给 404（未知签名不是一个空簇）。

    @param sig 失败签名。
    @param limit 最大证据行数，上限 100。
    @return 含 ``sig``/``n``/``name``/``examples`` 的字典；没有成员时返回 None。
    """
    tick()
    sig = str(sig or "")
    if not sig:
        return None
    limit = max(1, min(int(limit or 20), PAGE_LIMIT))
    with borrow() as index:
        total = index.conn.execute(
            "SELECT COUNT(*) FROM failure_members WHERE sig = ?", (sig,)
        ).fetchone()[0]
        if not int(total or 0):
            return None
        rows = [dict(row) for row in index.conn.execute(
            "SELECT c.run_id AS run_id, c.idx AS idx, c.label AS label, c.status AS status,"
            " c.stage AS stage, c.message AS message, r.task AS task, r.dtype AS dtype,"
            " r.target_id AS target_id, r.started_at AS started_at"
            " FROM candidates c JOIN runs r ON r.run_id = c.run_id"
            " WHERE c.failure_sig = ? AND r.status != 'missing'"
            " ORDER BY COALESCE(r.started_at, 0) DESC, c.run_id, c.idx LIMIT ?",
            (sig, limit),
        )]
    examples = []
    for row in rows:
        folder = _candidate_relative(row["run_id"], row["idx"], row["label"] or "")
        examples.append({
            "run_id": row["run_id"],
            "idx": row["idx"],
            "label": row["label"] or "",
            "status": row["status"] or "",
            "stage": row["stage"] or "",
            "message": row["message"] or "",
            "task": row["task"] or "",
            "dtype": row["dtype"] or "",
            "target_id": row["target_id"] or "",
            "report_path": f"{folder}/report.json",
            "feedback_path": f"{folder}/feedback.txt",
        })
    first = examples[0] if examples else {}
    return {
        "sig": sig,
        "n": int(total or 0),
        "name": cluster_name(sig, first.get("status", ""), first.get("stage", ""), first.get("message", "")),
        "examples": examples,
    }


def failures(
    *, status: str | None = None, task: str | None = None, sig: str | None = None,
    run: str | None = None, label: str | None = None, limit: int = 20,
) -> dict:
    """@brief 跨运行列出候选失败；``status`` 为精确匹配（``abi_error``、``incorrect``、``timeout`` 等）。

    ``status`` 同时约束两个分支：候选按 ``c.status``，会话按 ``outcome``（会话行的 status 就是它）。
    ``sig`` 按失败簇指纹过滤（候选主键走 ``candidates_sig``，会话主键走 ``sessions.failure_sig``）：
    两种指纹前缀不同（§7.2），所以一条 ``--sig`` 同时覆盖“候选失败”与“整轮没有内核”的会话失败，
    ``brief`` 的 ``last_failure.sig`` 因此能直接对上。候选行带报告里的 ``message`` 与可复现的
    ``report_path``；会话失败行没有候选报告，``report_path`` 为 None，``message`` 是脱敏后的错误原文。
    ``run``/``label`` 把范围收到一场运行的一颗候选，这样“这场为何失败”只需一次调用。
    冷索引上按 ``sig`` 反查会话时一次只懒投影一批（:data:`SESSIONS_SIG_LAZY_RUNS` 个 run）；
    查完还有没投影的 run，响应多带 ``sessions_unprojected``（数量），表示这次可能没查全。
    原始英文说明：Candidate failures across runs. ``status`` is exact (``abi_error``, ``incorrect``, ``timeout``, ...).

    @param status 只看该失败状态（候选行按 status，会话行按 outcome）。
    @param task 只看该任务。
    @param sig 只看该失败簇签名。
    @param run 只看该运行（相对 ``runs/`` 的 id）。
    @param label 只看该候选标签。
    @param limit 最大行数，上限 100。
    @return 含 ``limit`` 与 ``failures`` 的字典。
    """
    tick()
    limit = max(1, min(int(limit or 20), PAGE_LIMIT))
    # An old/damaged report is undecidable, not a failure: it is not a claim this reader can make
    # (its ``status`` column holds the format marker).  Only current (schema-4) rows are listed.
    clauses = ["c.status != ''", "c.status != 'ok'", "c.report_state = 'current'"]
    args: list = []
    if status:
        clauses.append("c.status = ?")
        args.append(status)
    if task:
        clauses.append("r.task = ?")
        args.append(task)
    if sig:
        clauses.append("c.failure_sig = ?")
        args.append(sig)
    if run:
        clauses.append("c.run_id = ?")
        args.append(run)
    if label:
        clauses.append("c.label = ?")
        args.append(label)
    with borrow() as index:
        rows = [dict(row) for row in index.conn.execute(
            "SELECT c.run_id, c.idx, c.label, c.status, c.stage, c.speedup, c.within_noise, c.kernel_sha256, "
            "c.failure_sig, c.message, c.report_state, c.build_shape, c.build_kind, "
            "r.task, r.dtype, r.target_id, r.started_at "
            f"FROM candidates c JOIN runs r ON r.run_id = c.run_id WHERE {' AND '.join(clauses)} "
            "ORDER BY COALESCE(r.started_at, 0) DESC, c.run_id, c.idx LIMIT ?",
            [*args, limit],
        )]
    for row in rows:
        folder = _candidate_relative(row["run_id"], row["idx"], row["label"] or "")
        row["report_path"] = f"{folder}/report.json"
    unprojected = 0
    # 候选与会话的指纹前缀不同（§7.2），所以候选已经命中时不可能再有会话命中：跳过会话分支，
    # 避免普通的 failures --sig 在冷索引上白做一轮 sessions 懒投影（读 transcript）。
    if sig and not rows:
        # 会话指纹来自“整轮没有内核”的 hmz 会话：它没有候选报告，但同样是一条可定位的失败。
        session_clauses = ["s.failure_sig = ?"]
        session_args: list = [str(sig)]
        if status:
            # 会话的 status 就是 outcome；漏掉它会让 status 过滤在这个分支失效，
            # 返回和请求相反状态的行（CLI/API 的精确匹配契约对两个分支都成立）。
            session_clauses.append("s.outcome = ?")
            session_args.append(status)
        if task:
            session_clauses.append("r.task = ?")
            session_args.append(task)
        if run:
            session_clauses.append("s.run_id = ?")
            session_args.append(run)
        if label:
            session_clauses.append("s.label = ?")
            session_args.append(label)
        sessions, todo, overflow = _sessions_for_sig(session_clauses, session_args, limit, task=task, run=run)
        # 冷建不投影 sessions（§11.4 选 (a)）；按 sig 反查之前，把还没投影的 run 懒投影一次，
        # 否则 brief 给出的会话指纹在冷索引上永远查不到。投影在写锁外逐个 run 做，且只读有
        # transcript 的目录；冷建本身仍然 0 字节 transcript。用 ``_read_run`` 而不是
        # ``resolve_run``：后者会先 ``_ingest``（读候选报告与 transcript），是写路径的入口。
        if todo:
            for run_id in todo:
                directory = _transcript_dir(run_id)
                if directory is None:
                    # 没有 transcript 的 run 写 ``0:0`` 游标：不写的话每批都被同一批空目录
                    # 填满，懒投影永远不前进。
                    get_index().mark_sessions_absent(run_id)
                    continue
                get_index().ensure_sessions(directory)
            sessions, again, overflow = _sessions_for_sig(session_clauses, session_args, limit, task=task, run=run)
            if not sessions:
                # 本次只推进一批。还有带 transcript 又没投影的 run 时，跨场 sig 搜索可能漏；
                # 必须让调用方看见“没查完”，否则一个空列表会被读成“没有这个失败”。
                unprojected = overflow + sum(1 for run_id in (again or []) if _transcript_dir(run_id) is not None)
        for row in sessions:
            rows.append({
                "run_id": row["run_id"], "idx": None, "label": row.get("candidate_label") or row.get("label") or "",
                "status": row.get("outcome") or "", "stage": row.get("failure_class") or "",
                "speedup": None, "within_noise": None, "kernel_sha256": None,
                "failure_sig": row.get("failure_sig"), "message": row.get("error") or "",
                "task": row.get("task"), "dtype": row.get("dtype"), "target_id": row.get("target_id"),
                "started_at": row.get("started_at"), "report_path": None,
            })
    result = {"limit": limit, "failures": rows[:limit]}
    if unprojected:
        # 还有没投影的 run：跨场 sig 搜索可能漏，调用方可以带 --run 精确定位。
        result["sessions_unprojected"] = unprojected
    return result


SESSIONS_SIG_LAZY_RUNS = 200


def _transcript_dir(run_id: str) -> Path | None:
    """@brief 有 ``transcript.jsonl`` 的运行目录；没有 transcript 时返回 None。

    懒投影只碰这些目录：空目录既没有会话可投影，也不该为它读 ``run_config.json``。
    走 ``_read_run``（只解析路径、不摄取索引），所以这里不会拿到写锁、也不会读候选报告。

    @param run_id 运行 id。
    @return 运行目录或 None。
    """
    directory = _read_run(run_id)
    if directory is None or not (directory / "transcript.jsonl").is_file():
        return None
    return directory


def _sessions_for_sig(clauses: list[str], args: list, limit: int, *, task: str | None, run: str | None):
    """@brief 按会话指纹查已投影的 sessions；顺带回没有投影过的 run（最多一批）。

    @param clauses 会话查询的 WHERE 子句。
    @param args 子句参数。
    @param limit 最大行数。
    @param task 任务过滤（懒投影只做匹配的 run）。
    @param run 单场过滤。
    @return ``(会话行, 待投影 run_id 列表或 None, 超出本批上限的未投影条数)``。
    """
    with borrow() as index:
        sessions = [dict(row) for row in index.conn.execute(
            "SELECT s.run_id, s.seq, s.label, s.candidate_label, s.outcome, s.failure_class, s.error,"
            " s.failure_sig, r.task, r.dtype, r.target_id, r.started_at "
            "FROM sessions s JOIN runs r ON r.run_id = s.run_id "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY COALESCE(r.started_at, 0) DESC, s.run_id, s.seq LIMIT ?",
            [*args, limit],
        )]
        if sessions:
            return sessions, None, 0
        pending_clauses = ["r.status != 'missing'"]
        pending_args: list = []
        if task:
            pending_clauses.append("r.task = ?")
            pending_args.append(task)
        if run:
            pending_clauses.append("r.run_id = ?")
            pending_args.append(run)
        pending = index.conn.execute(
            "SELECT r.run_id FROM runs r WHERE " + " AND ".join(pending_clauses)
            + " AND NOT EXISTS (SELECT 1 FROM meta m WHERE m.key = 'sessions_fp:' || r.run_id)"
            # 冷建已写下 transcript_bytes；为 0 的 run 没有可投影的 transcript，不能让它
            # 永久占据每批配额（否则懒投影永远不前进）。没有这条 meta 的老索引仍纳入，
            # 由投影循环补写标记。
            + " AND NOT EXISTS (SELECT 1 FROM meta t"
              " WHERE t.key = 'transcript_bytes:' || r.run_id AND t.value = '0')"
            " ORDER BY COALESCE(r.started_at, 0) DESC LIMIT ?",
            [*pending_args, SESSIONS_SIG_LAZY_RUNS + 1],
        ).fetchall()
    overflow = max(0, len(pending) - SESSIONS_SIG_LAZY_RUNS)
    todo = [str(row["run_id"]) for row in pending[:SESSIONS_SIG_LAZY_RUNS]]
    return sessions, todo or None, overflow


def best_confirmed(*, task: str | None = None, target: str | None = None, limit: int = 20) -> dict:
    """@brief 列出已确认的 kernel，最快在前；发布门已排除未绑定的数字。

    原始英文说明：Confirmed kernels, fastest first. The publication gate already excluded unbound numbers.

    @param task 只看该任务。
    @param target 只看该 target。
    @param limit 最大行数，上限 100。
    @return 含 ``limit`` 与 ``best`` 的字典。
    """
    # 完成 run 的 winner/确认文件可能在 root mtime 不变时被改写；发布查询前
    # 先对“确认指纹变了”的已确认行重新摄取。指纹一律只 stat，且 stat 阶段不持写锁：
    # 不变的行一个字节都不读，也不必每请求重摄取全部已确认运行。
    # 新 run 的发现：节流后只 stat 可能包含 run 的父目录（不扫全部 watched）。旧实现每个 /best
    # 都 tick()，这里保留同样的发现语义，只把 O(全部历史) 的周期核对留给其他入口。
    now = time.monotonic()
    if now - _BEST_DISCOVER["at"] >= _tick_interval():
        _BEST_DISCOVER["at"] = now
        get_index().discover_new()
    stale = get_index().confirmation_fingerprints()
    if stale:
        with borrow() as index:
            for run_id in stale:
                path = detail.resolve_run(run_id)
                if path is not None:
                    index._ingest(path, force=True)
                    continue
                # 目录被删掉：统一走 missing 清理（发布字段、候选、confirmed_best）。
                index._mark_missing(run_id)
            # _ingest 不自己提交；不提交会把写锁留到下一个请求。
            index.conn.commit()
    limit = max(1, min(int(limit or 20), PAGE_LIMIT))
    clauses = ["publication = 'confirmed'", "published_speedup IS NOT NULL"]
    args: list = []
    if task:
        clauses.append("task = ?")
        args.append(task)
    if target:
        clauses.append("target_id = ?")
        args.append(target)
    with borrow() as index:
        rows = [dict(row) for row in index.conn.execute(
            "SELECT run_id, task, dtype, size, target_id, model, published_speedup, started_at, kernel_sha256 "
            f"FROM runs WHERE {' AND '.join(clauses)} ORDER BY published_speedup DESC, COALESCE(started_at, 0) DESC LIMIT ?",
            [*args, limit],
        )]
    return {"limit": limit, "best": rows}


EXPERIMENTS_PAGE = 100


def experiments(*, limit: int = EXPERIMENTS_PAGE, cursor: str | None = None) -> dict:
    """@brief 按创建时间列出实验及其 arm，keyset 游标翻页。

    @param limit 单页上限。
    @param cursor 上一页返回的 ``next`` 游标。
    @return ``{"experiments": [...], "total": n, "next": ...}``。
    """
    tick()
    limit = max(1, min(int(limit or EXPERIMENTS_PAGE), EXPERIMENTS_PAGE))
    conditions = ""
    values: list = []
    if cursor:
        try:
            created, experiment_id = cursor.split("\x1f", 1)
            created_value = float(created)
        except (TypeError, ValueError):
            created_value, experiment_id = None, ""
        if created_value is not None:
            conditions = " WHERE (created_at < ? OR (created_at = ? AND experiment_id > ?))"
            values = [created_value, created_value, experiment_id]
    with borrow() as index:
        total = int(index.conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0])
        items = list(index.conn.execute(
            "SELECT * FROM experiments" + conditions + " ORDER BY created_at DESC, experiment_id ASC LIMIT ?",
            (*values, limit + 1),
        ))
        page = items[:limit]
        rows = []
        for item in page:
            arms = [
                dict(arm)
                for arm in index.conn.execute(
                    "SELECT arm_id, planned_n FROM arms WHERE experiment_id = ? ORDER BY arm_id",
                    (item["experiment_id"],),
                )
            ]
            rows.append({**dict(item), "arms": arms})
    next_cursor = None
    if len(items) > limit:
        last = page[-1]
        next_cursor = f"{last['created_at']}\x1f{last['experiment_id']}"
    return {"experiments": rows, "total": total, "next": next_cursor}


def experiment_jobs() -> list[dict]:
    """@brief 列出停止按钮能看到的队列作业；只读已知实验的状态文件，不扫全树。

    原始英文说明：Queues the stop button can see. Reads the state file of each known experiment, not the whole tree.

    @return 作业字典列表，按 ``updated_at`` 降序。
    """
    tick()
    with borrow() as index:
        ids = [item["experiment_id"] for item in index.conn.execute("SELECT experiment_id FROM experiments")]
    jobs = []
    root = detail.runs_root()
    for experiment_id in ids:
        directory = root / experiment_id
        job = detail.experiment_job(directory)
        if job is not None:
            jobs.append(job)
    jobs.sort(key=lambda row: float(row.get("updated_at") or 0), reverse=True)
    return jobs


def compare(experiment_id: str) -> dict:
    """@brief 对比一个实验内各 arm 的指标，并给出可否下判定。

    漂移（``binding=drift``）与推断（``source=inferred``）的运行不参与比较；样本不足时不下结论。

    @param experiment_id 实验 id。
    @return 含 ``found``、``verdict``、``reason``、``winner``、``arms``、``drift`` 与 ``inferred`` 的字典。
    """
    tick()
    with borrow() as index:
        exp = index.conn.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)).fetchone()
        exp = dict(exp) if exp is not None else None
        runs = [
            dict(row)
            for row in index.conn.execute(
                "SELECT * FROM runs WHERE experiment_id = ? AND status != 'missing'",
                (experiment_id,),
            )
        ]
        arms = [dict(arm) for arm in index.conn.execute("SELECT * FROM arms WHERE experiment_id = ? ORDER BY arm_id", (experiment_id,))]
    if exp is None and not runs:
        return {"experiment_id": experiment_id, "found": False, "verdict": "样本不足以判定", "arms": []}
    if exp is None:
        exp = {"kind": runs[0]["kind"], "config_hash": "", "frozen_json": "{}"}
    frozen = json.loads(exp["frozen_json"] or "{}")
    drift = [row["run_id"] for row in runs if row["binding"] == "drift"]
    inferred = [row["run_id"] for row in runs if row["source"] == "inferred"]
    usable = [row for row in runs if row["binding"] != "drift" and row["source"] != "inferred"]
    # §8.0 的 comparability key：按 (arm_id, config_key) 分组。配置不一致的运行不再静默丢弃，
    # 而是各自成组并列进 ``mixed``（点名 run_id），分析面板可以据此显示“这个臂混了两种配置”。
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in usable:
        grouped.setdefault((row["arm_id"] or "", config_key(row, frozen)), []).append(row)
    mixed: list[dict] = []
    arm_order: list[str] = []
    for arm in arms:
        if arm["arm_id"] not in arm_order:
            arm_order.append(arm["arm_id"])
    for arm_id, _key in grouped:
        if arm_id not in arm_order:
            arm_order.append(arm_id)
    arm_rows = []
    judge_rows = []
    judge_slugs: list[str] = []
    mixed_arms: list[str] = []
    for arm_id in arm_order:
        keys = [key for (group_arm, key) in grouped if group_arm == arm_id]
        if len(keys) > 1:
            mixed_arms.append(arm_id)
            for key in keys:
                mixed.append({
                    "arm_id": arm_id,
                    "config_key": key,
                    "runs": sorted(row["run_id"] for row in grouped[(arm_id, key)]),
                })
        if not keys:
            judge_slugs.append(arm_id)
        for key in keys:
            # 判定身份也带 config_key：混合臂的每个配置各自一个桶，绝不跨配置合并成一个“契约”。
            judge_slug = f"{arm_id}#{key}" if len(keys) > 1 else arm_id
            judge_slugs.append(judge_slug)
            members = grouped[(arm_id, key)]
            speeds = [row["published_speedup"] for row in members if isinstance(row["published_speedup"], (int, float))]
            metrics = suite_metrics([
                {"published_speedup": row["published_speedup"], "spent_usd": row["spent_usd"]}
                for row in members
            ])
            spent = metrics["spent_usd"]
            gain = _gain_per_usd(members, spent)
            row_out = {
                "arm_id": arm_id,
                "n": len(members),
                "published_n": len(speeds),
                "median": _median(speeds),
                "min": min(speeds) if speeds else None,
                "max": max(speeds) if speeds else None,
                "confirmed_n": sum(1 for row in members if row["publication"] == "confirmed"),
                "fast": metrics["fast"],
                "spent_usd": spent,
                "gain_per_usd": gain,
            }
            if len(keys) > 1:
                # 只有混合臂才多带这个键；单配置臂的行保持与旧实现逐字段一致。
                row_out["config_key"] = key
            arm_rows.append(row_out)
            for row in members:
                judge_rows.append({"slug": judge_slug, "published_speedup": row["published_speedup"], "skipped": None})
    if inferred and not usable:
        verdict, reason, winner = "不下判定", "这些运行是从旧目录推断的", None
    elif mixed_arms:
        # 混合臂里的运行不可直接对比：不给跨配置的 winner（判定身份虽然已按配置分桶，
        # 但把“某一种配置的胜利”报成整臂胜利仍然会误导）。
        verdict = "样本不足以判定"
        reason = f"臂 {', '.join(sorted(mixed_arms))} 混了多种可比配置（config_key 不同），不能跨配置判胜负"
        winner = None
    elif exp["kind"] == "contract_menu":
        judged = judge(judge_rows, judge_slugs)
        verdict, reason, winner = judged["verdict"], judged["reason"], judged["winner"]
    elif not any(row["published_n"] >= 2 for row in arm_rows):
        verdict, reason, winner = "样本不足以判定", "发布数字少于 2 个", None
    else:
        verdict, reason, winner = "样本不足以判定", "只给出各臂的数字，不从最高的一次定胜负", None
    return {
        "experiment_id": experiment_id,
        "found": True,
        "kind": exp["kind"],
        "config_hash": exp["config_hash"],
        "frozen": frozen,
        "verdict": verdict,
        "reason": reason,
        "winner": winner,
        "arms": arm_rows,
        "mixed": mixed,
        "drift": drift,
        "inferred": inferred,
    }


def _median(values: list[float]) -> float | None:
    """@brief 取列表的中位数（偶数个时取中间偏上的那个）。

    @param values 数值列表。
    @return 中位数；列表为空时返回 None。
    """
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _gain_per_usd(members: list[dict], spent: float) -> float | None:
    """@brief 计算每美元换来的“发布加速比 - 朴素加速比”均值。

    @param members 该 arm 的运行行列表。
    @param spent 该 arm 的总花费（美元）。
    @return 每美元增益；花费不为正或没有可用样本时返回 None。
    """
    if spent <= 0:
        return None
    pieces = []
    for row in members:
        published = row.get("published_speedup")
        naive = row.get("naive_speedup")
        if isinstance(published, (int, float)) and not isinstance(published, bool) and isinstance(naive, (int, float)) and not isinstance(naive, bool) and naive:
            pieces.append(published - naive)
    if not pieces:
        return None
    return sum(pieces) / spent


def harness_board(run_id: str) -> dict | None:
    """@brief 返回一个运行的结构化提案与覆盖决策；文件是唯一事实源。

    原始英文说明：Structured proposals and overlay decisions for one run. The files are the source.

    @param run_id 运行 id。
    @return harness 看板字典；运行不存在时返回 None。
    """
    from observe.detail import resolve_run
    from taic.harness import board

    directory = resolve_run(run_id)
    if directory is None:
        return None
    return board(directory)


def suggestions(**kwargs) -> dict:
    """@brief 给 AI 的紧凑系统建议列表；全文另走详情调用。

    原始英文说明：Compact free-text suggestions for AI; full text is a separate detail call.

    @param kwargs 透传给 ``suggestions.compact`` 的筛选与分页参数。
    @return 建议分组字典。
    """
    from observe.suggestions import compact

    return compact(**kwargs)


def suggestion(run_id: str, seq: int, digest: str) -> dict:
    """@brief 返回一条系统建议的确切详情。

    @param run_id 运行 id。
    @param seq 反馈条目序号。
    @param digest 条目摘要。
    @return 建议详情字典。
    """
    from observe.suggestions import detail

    return detail(run_id, seq, digest)


def feedback_board(*, since: int | None = None, limit: int | None = None, offset: int = 0, full: bool = False, similar: bool = False) -> dict:
    """@brief 返回 ``since`` 之后变更的 claim；省略 ``since`` 则给一页快照，只为答案内的行读文本。

    全量只在 ``full=True`` 时给出（有字节上限与 ``truncated``）；首次快照默认一页并带
    ``next_offset``。
    原始英文说明：Claims changed after ``since``. Omit ``since`` for the whole list. Text is read only for rows in the answer.

    @param since 反馈 revision 起点，None 表示全量。
    @param limit 快照单页条数。
    @param offset 快照页偏移。
    @param full 是否给全量导出。
    @param similar 是否给每行算语义近邻；默认不算，相似结论走 :func:`feedback_similar` 按需取。
    @return 反馈 feed 字典。
    """
    from observe.feedback import review_feed

    tick()
    with borrow() as index:
        return review_feed(index.conn, since=since, limit=limit, offset=offset, full=full, similar=similar)


def feedback_similar(run_id: str) -> dict | None:
    """@brief 返回一场运行里待审 claim 的语义近邻；运行不存在时返回 None。

    与首屏快照分开：只有用户展开这场运行才算这一场的句向量近邻。

    @param run_id 运行 id。
    @return ``{run_id, items}`` 或 None。
    """
    from observe.feedback import review_similar

    tick()
    if resolve_run(run_id) is None:
        return None
    with borrow() as index:
        return review_similar(index.conn, run_id)


ROLLUP_KEY = ("task", "dtype", "size", "target_id", "protocol_epoch")
ROLLUP_MERGE_AFTER_DAYS = 400


def rollup(*, merge_after_days: int = ROLLUP_MERGE_AFTER_DAYS, now: float | None = None) -> dict:
    """@brief 读 ``run_rollup`` 的按天汇总；超过保留期的桶在读时并成月，不改存储。

    §11.2 的分层是“汇总降采样”，不是冷热迁移：原始 ``runs`` 行永不删，``run_rollup`` 按天存，
    读时把 ``merge_after_days``（默认 400）以前的桶按（月, 分组键）相加。分组键固定为
    ``task/dtype/size/target_id/protocol_epoch``——任何一行都不可能跨 dtype 或跨纪元求和。

    @param merge_after_days 超过这么多天的桶合并成月；0 或负数表示不合。
    @param now 当前时间（测试注入）；None 表示 ``time.time()``。
    @return ``{"rows": [...], "merged": 合并掉的旧桶数, "merge_after_days": n, "keys": [...]}``。
    """
    tick()
    cutoff = ""
    if int(merge_after_days) > 0:
        cutoff = time.strftime("%Y-%m-%d", time.localtime((now if now is not None else time.time()) - 86400 * int(merge_after_days)))
    with borrow() as index:
        rows = [dict(row) for row in index.conn.execute(
            "SELECT day, task, dtype, size, target_id, protocol_epoch, runs, confirmed, spent_usd,"
            " published_sum, published_max FROM run_rollup"
        )]
    merged: dict[tuple[str, ...], dict] = {}
    out: list[dict] = []
    folded = 0
    for row in rows:
        key = tuple(str(row[name] or "") for name in ROLLUP_KEY)
        day = str(row["day"] or "")
        if cutoff and day < cutoff:
            folded += 1
            bucket = merged.get((day[:7], *key))
            if bucket is None:
                bucket = {
                    "day": day[:7], "month": True, **dict(zip(ROLLUP_KEY, key)),
                    "runs": 0, "confirmed": 0, "spent_usd": 0.0, "published_sum": 0.0,
                    "published_max": None,
                }
                merged[(day[:7], *key)] = bucket
            bucket["runs"] += int(row["runs"] or 0)
            bucket["confirmed"] += int(row["confirmed"] or 0)
            bucket["spent_usd"] += float(row["spent_usd"] or 0.0)
            bucket["published_sum"] += float(row["published_sum"] or 0.0)
            value = row["published_max"]
            if value is not None:
                bucket["published_max"] = value if bucket["published_max"] is None else max(bucket["published_max"], float(value))
        else:
            out.append({
                "day": day, "month": False, **dict(zip(ROLLUP_KEY, key)),
                "runs": int(row["runs"] or 0), "confirmed": int(row["confirmed"] or 0),
                "spent_usd": round(float(row["spent_usd"] or 0.0), 6),
                "published_sum": round(float(row["published_sum"] or 0.0), 6),
                "published_max": None if row["published_max"] is None else float(row["published_max"]),
            })
    for bucket in merged.values():
        bucket["spent_usd"] = round(bucket["spent_usd"], 6)
        bucket["published_sum"] = round(bucket["published_sum"], 6)
        out.append(bucket)
    out.sort(key=lambda row: (row["day"], *(row[name] for name in ROLLUP_KEY)))
    return {
        "rows": out,
        "merged": folded,
        "merge_after_days": int(merge_after_days),
        "keys": list(ROLLUP_KEY),
    }


def overview() -> dict:
    """@brief 首屏总览：只带页面真正消费的反馈计数。

    §11.5 瘦身：``gpu``/``live_runs``/``live_evals``/``heartbeat``/``tasks``/``experiments`` 都要读
    机全局文件或列任务，首屏只用到 ``feedback_pending``，这些字段移到 :func:`live_overview`
    的按需端点。响应小于 1KB，且首屏不再为主响应额外读一份 GPU 快照。

    @return 含反馈计数与时间戳的总览字典。
    """
    from observe.feedback import summary_from_index

    with borrow() as index:
        feedback = summary_from_index(index.conn)
    return {
        "feedback_pending": feedback["pending_total"],
        "feedback_frequent": feedback["frequent"],
        "feedback_recurrence": feedback["recurrence"],
        "now": time.time(),
    }


def live_overview() -> dict:
    """@brief 按需的活状态总览：GPU、在跑的运行与评估、心跳、任务与实验队列。

    页面顶部的「进行中」区用 ``/api/v2/patrol``；这块只留给需要 GPU 与任务清单的显式请求。

    @return 含 ``gpu``/``live_runs``/``live_evals``/``heartbeat``/``tasks``/``experiments`` 的字典。
    """
    from tasks import list_tasks

    # §5.7：活运行个位数，只查它们；不再解码全部 ``row_json``。
    live = budget_rows(status="running")
    progress = None
    if live:
        directory = resolve_run(live[0]["id"])
        if directory is not None:
            progress = detail._age(detail._load(directory / "eval_progress.json"))
    evals = detail.live_evals()
    beat = detail._heartbeat(progress) or (detail._heartbeat(evals[0]) if evals else None)
    return {
        "gpu": detail.gpu_snapshot(),
        "live_runs": live,
        "live_evals": evals,
        "heartbeat": beat,
        "tasks": list(list_tasks(include_phase2=True)),
        "experiments": experiment_jobs(),
        "now": time.time(),
    }


def frame(run_id: str | None) -> dict:
    """@brief 组装一帧页面数据：总览、历史、预算与选中运行的详情。

    选中的运行不存在时退回最后一个存活运行，再退回最新一行。

    @param run_id 期望选中的运行 id，可为 None 或空串。
    @return 含 ``overview``、``history``、``budget``、``selected`` 与 ``run`` 的字典。
    """
    hist = history()
    chosen = run_id or ""
    if chosen and resolve_run(chosen) is None:
        chosen = ""
    if not chosen:
        live = next((row for row in hist if row.get("status") == "running"), None)
        chosen = live["id"] if live else (hist[0]["id"] if hist else "")
    return {
        "overview": overview(),
        "history": hist,
        "budget": budget_report(hist),
        "selected": chosen,
        "run": run_detail(chosen) if chosen else None,
    }


def ledger_report() -> dict:
    """@brief 按 operator 与月份汇总花费账本，并带上结转总额。

    @return 含 ``runs``、``spent_usd``、``budget_usd``、``by_operator``、``by_month`` 与 ``carryover_usd`` 的字典。
    """
    tick()
    with borrow() as index:
        by_operator = [
            {"operator": row["operator"], "runs": row["runs"], "spent_usd": round(row["spent"], 6), "budget_usd": round(row["budget"], 6)}
            for row in index.conn.execute(
                "SELECT operator, COUNT(*) AS runs, COALESCE(SUM(spent_max_usd), 0) AS spent, COALESCE(SUM(budget_usd), 0) AS budget FROM ledger GROUP BY operator ORDER BY spent DESC, operator"
            )
        ]
        # 一次 GROUP BY month, operator，不再每个月再查一次（旧实现是 1+N 次）。
        monthly: dict[str, dict] = {}
        for row in index.conn.execute(
            "SELECT month, operator, COUNT(*) AS runs, COALESCE(SUM(spent_max_usd), 0) AS spent "
            "FROM ledger GROUP BY month, operator ORDER BY month, spent DESC, operator"
        ):
            bucket = monthly.setdefault(str(row["month"]), {"runs": 0, "spent": 0.0, "operators": []})
            bucket["runs"] += int(row["runs"])
            bucket["spent"] += float(row["spent"])
            bucket["operators"].append({
                "operator": row["operator"], "runs": row["runs"], "spent_usd": round(row["spent"], 6),
            })
        by_month = [
            {"month": month, "runs": bucket["runs"], "spent_usd": round(bucket["spent"], 6), "operators": bucket["operators"]}
            for month, bucket in monthly.items()
        ]
        counted = index.conn.execute("SELECT COUNT(*), COALESCE(SUM(budget_usd), 0) FROM ledger").fetchone()
        spent = index.total_usd()
        carry = _carry_total(index)
    return {
        "runs": counted[0],
        "spent_usd": spent,
        "budget_usd": round(counted[1], 6),
        "by_operator": by_operator,
        "by_month": by_month,
        "carryover_usd": carry,
    }


def _carry_total(index) -> float:
    """@brief 取账本结转表中的总额。

    @param index 已借用的索引对象。
    @return 结转美元数；表为空时返回 0.0。
    """
    row = index.conn.execute("SELECT total_usd FROM ledger_carryover LIMIT 1").fetchone()
    return 0.0 if row is None else float(row["total_usd"] or 0)


def live_files() -> list[Path]:
    """@brief 列出实时状态相关文件：GPU 目录、观察目录与最近几个 ``tcenv-*.json``。

    @return 路径列表，用于计算实时 revision。
    """
    from observe.publish import gpu_dir, observe_dir

    paths = [gpu_dir(), observe_dir()]
    directory = observe_dir()
    if directory.is_dir():
        paths.extend(sorted(directory.glob("tcenv-*.json"))[:5])
    return paths


def live_revision() -> int:
    """@brief 当实时文件签名变化时递增的计数器。

    @return 单调递增的版本号。
    """
    signature = tuple(_file_signature(path) for path in live_files())
    if signature != _LIVE["last"]:
        _LIVE["last"] = signature
        _LIVE["n"] += 1
    return int(_LIVE["n"])


def _file_signature(path: Path) -> tuple:
    """@brief 取文件或目录的轻量签名（路径、mtime、大小或条目数）。

    @param path 文件或目录路径。
    @return 签名元组；路径不存在或读不了时 mtime/大小记为 0。
    """
    try:
        if path.is_dir():
            stat = path.stat()
            return (str(path), stat.st_mtime_ns, len(list(path.iterdir())))
        stat = path.stat()
        return (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (str(path), 0, 0)


def stream_revision() -> int:
    """@brief 返回帧数据里的 ``revision`` 字段。

    @return 当前流版本号。
    """
    return poll_stream()["revision"]


def poll_stream() -> dict:
    """@brief 生成一个空闲帧；详情 revision 与轮次 revision 各自独立变化。

    原始英文说明：One idle frame. The detail revision and the turn revision move independently.

    @return 含 ``index_revision``、``turns_revision``、``feedback_revision``、``live_revision``、``revision`` 与 ``runs`` 的字典。
    ``index_revision`` 只跟索引版本走；``live_revision`` 单独给现场文件，不参与整表刷新。
    """
    global _SENT
    tick()
    index = get_index()
    rev = index.revision()
    changed = list(index.changed)
    # live_revision 只描述 GPU 锁目录等现场文件，不能让它们触发整表重取。
    _SENT = max(_SENT, rev)
    return {
        "index_revision": _SENT,
        "turns_revision": index.turns_revision(),
        "feedback_revision": index.feedback_revision(),
        "live_revision": live_revision(),
        "revision": _SENT,
        "runs": changed,
    }


def changed_runs() -> list[str]:
    """@brief 返回索引自上次 tick 以来标记为有变化的 run id。

    @return run id 列表。
    """
    return list(get_index().changed)
