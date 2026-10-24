"""Read the observe index. Listing, filtering and arm summaries do not open candidate reports."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path

from experiments.common import suite_metrics
from experiments.contract_menu import judge
from observe import detail
from observe.budget import summarize
from observe.index import borrow, get_index
from taic.plan_ledger import nodes_from_trace
from taic.policy import feature_signature

PAGE_LIMIT = 100
_LIVE = {"last": None, "n": 0}
_SENT = 0
_TICK = {"at": 0.0}
_HISTORY = {"key": None, "rows": []}
_FEATURE_CACHE: dict[str, tuple] = {}
_SORTS = {"started": "started_at", "speedup": "published_speedup", "spent": "spent_usd", "updated": "updated_at"}
_GRID_CACHE: dict[tuple, list[dict]] = {}


def gpu_snapshot() -> dict:
    """@brief 取当前 GPU 快照（转发给 ``detail.gpu_snapshot``）。

    @return GPU 快照字典。
    """
    return detail.gpu_snapshot()


pid_alive = detail.pid_alive


def refresh() -> None:
    """@brief 发现新目录并重读索引已知运行的改动；供必须与磁盘一致的读路径使用。

    原始英文说明：Pick up new directories and edits to runs the index already knows. Used by reads that must match the disk.
    """
    get_index().sync_known()


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
    """@brief 对给定行（默认取 ``history()``）求预算汇总。

    @param rows 历史行列表，None 时现取 ``history()``。
    @return ``summarize`` 生成的预算报告。
    """
    return summarize(history() if rows is None else rows)


def resolve_run(run_id: str) -> Path | None:
    """@brief 把 run id 解析成运行目录。

    @param run_id 运行 id。
    @return 运行目录；索引里没有该 run 时返回 None。
    """
    return get_index().resolve(run_id)


def run_detail(run_id: str, *, stamp: str = "") -> dict | None:
    """@brief 组装一个运行的详情负载（含特征网格、计划结果与反馈）。

    ``stamp`` 与当前 detail 戳一致时只回 ``{"same": True, ...}``，不做完整渲染。

    @param run_id 运行 id。
    @param stamp 客户端已持有的 detail 戳，空串表示不复用。
    @return 详情字典；运行不存在时返回 None；戳匹配时返回标记字典。
    """
    directory = resolve_run(run_id)
    if directory is None:
        return None
    current = _detail_stamp(run_id)
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
    detail_doc["detail_stamp"] = _detail_stamp(run_id)
    return detail_doc


def _detail_stamp(run_id: str) -> str:
    """@brief 生成 detail 负载的身份戳；追加对话不会改变它。

    原始英文说明：Identity of the detail payload. A transcript append does not change it.

    @param run_id 运行 id。
    @return 由五个指纹字段拼接的字符串；无该运行时为空串。
    """
    row = get_index().conn.execute(
        "SELECT fp_state, fp_result, fp_config, fp_confirm, fp_extra FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        return ""
    return "|".join(str(row[key] or "") for key in ("fp_state", "fp_result", "fp_config", "fp_confirm", "fp_extra"))


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
    )
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached
    _target_id, backend_id, _ext = detail.target_identity(directory)
    root = directory / "candidates"
    if not root.is_dir():
        _GRID_CACHE[key] = []
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
        report = detail._load(path / "report.json") or {}
        speed = report.get("speedup")
        if isinstance(speed, bool) or not isinstance(speed, (int, float)):
            speed = None
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
        signature = signed.get(digest)
        if signature is None:
            signature = _FEATURE_CACHE.get(digest)
            if signature is None:
                signature = feature_signature(source, backend_id=backend_id)
                _FEATURE_CACHE[digest] = signature
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
    _GRID_CACHE[key] = rows
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
) -> dict:
    """@brief 返回一页运行；游标在 ``(排序列, run_id)`` 上做 keyset，``offset`` 是旧的退化方式。

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
        clauses.append("publication = ?")
        args.append(publication)
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
    with borrow() as index:
        total = index.conn.execute(f"SELECT COUNT(*) FROM runs WHERE {' AND '.join(clauses)}", args).fetchone()[0]
        rows = [
            dict(row)
            for row in index.conn.execute(
                "SELECT run_id, experiment_id, arm_id, trial, task, dtype, model, target_id, status, stop_reason, "
                "publication, published_speedup, spent_usd, budget_usd, source, binding, kernel_sha256, protocol_version, "
                "CASE WHEN protocol_version = 'flame_chase_1' THEN '2' ELSE protocol_version END AS protocol_epoch, "
                "committees_json, started_at, updated_at "
                f"FROM runs WHERE {where} ORDER BY COALESCE({column}, -1) {direction}, run_id {direction} LIMIT ? OFFSET ?",
                [*page_args, limit, page_offset],
            )
        ]
    following = None
    if len(rows) == limit:
        last = rows[-1]
        following = _encode_cursor(last.get(column) if last.get(column) is not None else -1, str(last["run_id"]))
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
            "SELECT run_id, idx, label, status, stage, speedup, within_noise, eligible, kernel_sha256, updated_at "
            f"FROM candidates WHERE {' AND '.join(clauses)} ORDER BY idx LIMIT ?",
            [*args, limit],
        )]
    return {"run_id": run_id, "limit": limit, "candidates": rows}


def sessions_page(run_id: str, *, limit: int = 500) -> dict:
    """@brief 每个 hmz 会话一行：委员会、模型、token 与花费。

    原始英文说明：One row per hmz session: committee, model, tokens and cost.

    @param run_id 运行 id。
    @param limit 最大行数，上限 2000。
    @return 含 ``run_id``、``limit`` 与 ``sessions`` 的字典。
    """
    tick()
    limit = max(1, min(int(limit or 500), 2000))
    with borrow() as index:
        rows = [dict(row) for row in index.conn.execute(
            "SELECT run_id, seq, label, candidate_label, attempt, committee, provider, model, outcome, cost_usd, "
            "session_id, failed_turns, last_kind, failure_class, error, tool_calls, diagnostics_path, "
            "tokens_input, tokens_output, tokens_cache_read, tokens_cache_write "
            "FROM sessions WHERE run_id = ? ORDER BY seq LIMIT ?",
            (run_id, limit),
        )]
    return {"run_id": run_id, "limit": limit, "sessions": rows}


def model_stats(run_id: str | None = None) -> dict:
    """@brief 按会话聚合 provider/模型用量，可只看一个运行。

    原始英文说明：Per-session model totals, so the first-committee ``runs.model`` is not the only view.

    @param run_id 可选的运行 id；缺省聚合全部运行。
    @return 含 ``models`` 列表的字典（按花费与会话数降序）。
    """
    tick()
    where = " WHERE run_id = ?" if run_id else ""
    args = (run_id,) if run_id else ()
    with borrow() as index:
        rows = [dict(row) for row in index.conn.execute(
            "SELECT committee, provider, model, COUNT(*) AS sessions, ROUND(SUM(COALESCE(cost_usd, 0)), 6) AS cost_usd, "
            "SUM(tokens_input) AS input_tokens, SUM(tokens_output) AS output_tokens, "
            "SUM(tokens_cache_read) AS cache_read_tokens, SUM(tokens_cache_write) AS cache_write_tokens "
            f"FROM sessions{where} GROUP BY committee, provider, model ORDER BY cost_usd DESC, sessions DESC",
            args,
        )]
    return {"models": rows}


def failures(*, status: str | None = None, task: str | None = None, limit: int = 50) -> dict:
    """@brief 跨运行列出候选失败；``status`` 为精确匹配（``abi_error``、``incorrect``、``timeout`` 等）。

    原始英文说明：Candidate failures across runs. ``status`` is exact (``abi_error``, ``incorrect``, ``timeout``, ...).

    @param status 只看该失败状态。
    @param task 只看该任务。
    @param limit 最大行数，上限 100。
    @return 含 ``limit`` 与 ``failures`` 的字典。
    """
    tick()
    limit = max(1, min(int(limit or 50), PAGE_LIMIT))
    clauses = ["c.status != ''", "c.status != 'ok'"]
    args: list = []
    if status:
        clauses.append("c.status = ?")
        args.append(status)
    if task:
        clauses.append("r.task = ?")
        args.append(task)
    with borrow() as index:
        rows = [dict(row) for row in index.conn.execute(
            "SELECT c.run_id, c.idx, c.label, c.status, c.stage, c.speedup, c.within_noise, c.kernel_sha256, "
            "r.task, r.dtype, r.target_id, r.started_at "
            f"FROM candidates c JOIN runs r ON r.run_id = c.run_id WHERE {' AND '.join(clauses)} "
            "ORDER BY COALESCE(r.started_at, 0) DESC, c.run_id, c.idx LIMIT ?",
            [*args, limit],
        )]
    return {"limit": limit, "failures": rows}


def best_confirmed(*, task: str | None = None, target: str | None = None, limit: int = 20) -> dict:
    """@brief 列出已确认的 kernel，最快在前；发布门已排除未绑定的数字。

    原始英文说明：Confirmed kernels, fastest first. The publication gate already excluded unbound numbers.

    @param task 只看该任务。
    @param target 只看该 target。
    @param limit 最大行数，上限 100。
    @return 含 ``limit`` 与 ``best`` 的字典。
    """
    tick()
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


def experiments() -> list[dict]:
    """@brief 列出实验及其 arm（按创建时间降序）。

    @return 每项为实验字典，``arms`` 键下含各臂的 ``arm_id`` 与 ``planned_n``。
    """
    tick()
    rows = []
    with borrow() as index:
        items = list(index.conn.execute("SELECT * FROM experiments ORDER BY created_at DESC"))
        for item in items:
            arms = [
                dict(arm)
                for arm in index.conn.execute(
                    "SELECT arm_id, planned_n FROM arms WHERE experiment_id = ? ORDER BY arm_id",
                    (item["experiment_id"],),
                )
            ]
            rows.append({**dict(item), "arms": arms})
    return rows


def experiment_jobs() -> list[dict]:
    """@brief 列出停止按钮能看到的队列作业；只读已知实验的状态文件，不扫全树。

    原始英文说明：Queues the stop button can see. Reads the state file of each known experiment, not the whole tree.

    @return 作业字典列表，按 ``updated_at`` 降序。
    """
    refresh()
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
    refresh()
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
    hashes = {row_hash(row, exp["config_hash"]) for row in usable}
    if len(hashes) > 1:
        usable = [row for row in usable if row_hash(row, exp["config_hash"]) == exp["config_hash"]]
    by_arm: dict[str, list[dict]] = {}
    for row in usable:
        by_arm.setdefault(row["arm_id"] or "", []).append(row)
    arm_rows = []
    judge_rows = []
    for arm in arms:
        members = by_arm.get(arm["arm_id"], [])
        speeds = [row["published_speedup"] for row in members if isinstance(row["published_speedup"], (int, float))]
        metrics = suite_metrics([
            {"published_speedup": row["published_speedup"], "spent_usd": row["spent_usd"]}
            for row in members
        ])
        spent = metrics["spent_usd"]
        gain = _gain_per_usd(members, spent)
        arm_rows.append({
            "arm_id": arm["arm_id"],
            "n": len(members),
            "published_n": len(speeds),
            "median": _median(speeds),
            "min": min(speeds) if speeds else None,
            "max": max(speeds) if speeds else None,
            "confirmed_n": sum(1 for row in members if row["publication"] == "confirmed"),
            "fast": metrics["fast"],
            "spent_usd": spent,
            "gain_per_usd": gain,
        })
        for row in members:
            judge_rows.append({"slug": arm["arm_id"], "published_speedup": row["published_speedup"], "skipped": None})
    if inferred and not usable:
        verdict, reason, winner = "不下判定", "这些运行是从旧目录推断的", None
    elif exp["kind"] == "contract_menu":
        judged = judge(judge_rows, [row["arm_id"] for row in arm_rows])
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
        "drift": drift,
        "inferred": inferred,
    }


def row_hash(row: dict, fallback: str) -> str:
    """@brief 返回用于分组的运行配置哈希，目前直接返回调用方给的 fallback。

    @param row 运行行字典。
    @param fallback 实验配置哈希，作为结果返回。
    @return 配置哈希字符串。
    """
    return fallback


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


def feedback_board(*, since: int | None = None) -> dict:
    """@brief 返回 ``since`` 之后变更的 claim；省略 ``since`` 则给全量，只为答案内的行读文本。

    原始英文说明：Claims changed after ``since``. Omit ``since`` for the whole list. Text is read only for rows in the answer.

    @param since 反馈 revision 起点，None 表示全量。
    @return 反馈 feed 字典。
    """
    from observe.feedback import review_feed

    tick()
    with borrow() as index:
        return review_feed(index.conn, since=since)


def overview() -> dict:
    """@brief 汇总首屏总览：GPU、存活运行与评估、心跳、任务、实验与反馈计数。

    @return 总览字典。
    """
    from tasks import list_tasks

    rows = history()
    live = [row for row in rows if row.get("status") == "running"]
    progress = None
    if live:
        directory = resolve_run(live[0]["id"])
        if directory is not None:
            progress = detail._age(detail._load(directory / "eval_progress.json"))
    evals = detail.live_evals()
    beat = detail._heartbeat(progress) or (detail._heartbeat(evals[0]) if evals else None)
    from observe.feedback import summary_from_index

    with borrow() as index:
        feedback = summary_from_index(index.conn)
    return {
        "gpu": detail.gpu_snapshot(),
        "live_runs": live,
        "live_evals": evals,
        "heartbeat": beat,
        "tasks": list(list_tasks(include_phase2=True)),
        "experiments": experiment_jobs(),
        "feedback_pending": feedback["pending_total"],
        "feedback_frequent": feedback["frequent"],
        "feedback_recurrence": feedback["recurrence"],
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
    refresh()
    with borrow() as index:
        by_operator = [
            {"operator": row["operator"], "runs": row["runs"], "spent_usd": round(row["spent"], 6), "budget_usd": round(row["budget"], 6)}
            for row in index.conn.execute(
                "SELECT operator, COUNT(*) AS runs, COALESCE(SUM(spent_max_usd), 0) AS spent, COALESCE(SUM(budget_usd), 0) AS budget FROM ledger GROUP BY operator ORDER BY spent DESC, operator"
            )
        ]
        by_month = []
        for row in index.conn.execute(
            "SELECT month, COUNT(*) AS runs, COALESCE(SUM(spent_max_usd), 0) AS spent FROM ledger GROUP BY month ORDER BY month"
        ):
            operators = [
                {"operator": item["operator"], "runs": item["runs"], "spent_usd": round(item["spent"], 6)}
                for item in index.conn.execute(
                    "SELECT operator, COUNT(*) AS runs, COALESCE(SUM(spent_max_usd), 0) AS spent FROM ledger WHERE month = ? GROUP BY operator ORDER BY spent DESC, operator",
                    (row["month"],),
                )
            ]
            by_month.append({"month": row["month"], "runs": row["runs"], "spent_usd": round(row["spent"], 6), "operators": operators})
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

    @return 含 ``index_revision``、``turns_revision``、``feedback_revision``、``revision`` 与 ``runs`` 的字典。
    """
    global _SENT
    tick()
    index = get_index()
    rev = index.revision()
    changed = list(index.changed)
    _SENT = max(_SENT, rev, live_revision())
    return {
        "index_revision": _SENT,
        "turns_revision": index.turns_revision(),
        "feedback_revision": index.feedback_revision(),
        "revision": _SENT,
        "runs": changed,
    }


def changed_runs() -> list[str]:
    """@brief 返回索引自上次 tick 以来标记为有变化的 run id。

    @return run id 列表。
    """
    return list(get_index().changed)
