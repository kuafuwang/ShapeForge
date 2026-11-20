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
import os
from pathlib import Path

from observe import detail, trust
from observe.index import borrow

# 页面与 JSON 共用这一句：这是历史 run 的描述统计，不是授权依据。
NOTICE = "历史 run 的描述统计；P1 的激活状态在 overlay 视图里单列；不参与授权"

# ``taic/session_supervisor.py::classify_result`` 会写出的显式 outcome。
EVALUATED = "evaluated"
UNCHANGED = "unchanged"
NO_SUBMISSION = "agent_no_submission"
OTHER_FAILURE_OUTCOMES = ("agent_failed", "provider_failed", "delivery_rejected", "cost_incomplete")
KNOWN_OUTCOMES = (EVALUATED, UNCHANGED, NO_SUBMISSION) + OTHER_FAILURE_OUTCOMES
PREMISE_KINDS = ("verified", "unverified", "downgraded")
# EvalReport 自己列出的明确失败结局。old_format/unreadable/missing 是读取状态，
# unchanged/duplicate/external_duplicate 是未送评测的状态；它们都不是候选失败。
NON_OK_STATUSES = frozenset({
    "rejected", "compile_error", "abi_error", "unsafe", "incorrect",
    "runtime_error", "timeout", "crash",
})

SETTLE_TRACE_FILE = "harness_settle_traces.jsonl"
ROUTE_M_TRACE_FILE = "harness_route_m.jsonl"
NEVER_AUTO = "never_auto"
OVERLAY_FILE = "harness_overlay.jsonl"
PROPOSAL_FILE = "harness_proposals.jsonl"
DECISION_FILE = "harness_decisions.jsonl"
CONTROL_FILE = "harness_control.json"
RENDER_RECEIPT = "render-check/task_delivery.json"
SESSION_RECEIPT_GLOB = "sessions/*/task_delivery.json"
SMOKE_FILE = "paired_smoke_result.json"
ROUTE_M = "M"
ROUTE_PREMISE = "premise"
ROUTE_TIP = "tip"
ROUTES = (ROUTE_PREMISE, ROUTE_M, ROUTE_TIP)
OVERLAY_NOTICE = (
    "只读：按盘上真实存在的 proposal/overlay/decision/control/送达回执统计；"
    "没有这些事件的历史 run 记 unavailable/None，不写成 0。激活状态不是效果，"
    "本视图不合成任何「有用率」。"
)
_TRACE_CACHE: dict[str, tuple[tuple[int, int], object]] = {}
_TRACE_CACHE_LIMIT = 512
# overlay/送达类文件的发现仅限 runs/ 树内，且不进入候选目录或隐藏目录。
_DISCOVERY_MARKERS = (OVERLAY_FILE, PROPOSAL_FILE, DECISION_FILE, CONTROL_FILE, SMOKE_FILE)
_SKIP_DIRS = {"__pycache__", "candidates"}
_DISCOVERY_DEPTH = 6


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


def _cached_file(path: Path, parse):
    """@brief 按 ``(mtime_ns, size)`` 缓存一份解析结果，避免面板每帧重读未变的文件。

    @param path 文件路径。
    @param parse 收到路径、返回解析结果的可调用对象。
    @return 解析结果；文件缺失时返回 ``parse`` 对缺失路径的结果。
    """
    key = str(path)
    try:
        stat = path.stat()
    except OSError:
        _TRACE_CACHE.pop(key, None)
        return parse(path)
    signature = (stat.st_mtime_ns, stat.st_size)
    cached = _TRACE_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    value = parse(path)
    _TRACE_CACHE[key] = (signature, value)
    while len(_TRACE_CACHE) > _TRACE_CACHE_LIMIT:
        _TRACE_CACHE.pop(next(iter(_TRACE_CACHE)))
    return value


def _cached_rows(path: Path) -> list[dict]:
    """@brief 容错读取并按签名缓存一份 JSONL。

    @param path JSONL 文件。
    @return 字典行列表。
    """
    return _cached_file(path, _read_jsonl)


def _read_json_object(path: Path) -> dict | None:
    """@brief 容错读取一份 JSON 对象；缺失、坏字节或非对象都返回 None。

    @param path JSON 文件。
    @return 字典或 None。
    """
    try:
        document = json.loads(path.read_text(errors="replace"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _cached_object(path: Path) -> dict | None:
    """@brief 按签名缓存一份 JSON 对象。

    @param path JSON 文件。
    @return 字典或 None。
    """
    return _cached_file(path, _read_json_object)


def clear_cache() -> None:
    """@brief 丢掉门迹文件缓存；测试或换 ``runs/`` 根时用。"""
    _TRACE_CACHE.clear()


def _run_ids(conn, run_id: str | None) -> list[str]:
    """@brief 取要扫门迹的 run id，单 run 过滤与全局各一条 SQL。

    @param conn 只读索引连接。
    @param run_id 只看这一场运行；None 表示全部。
    @return run id 列表。
    """
    if run_id:
        row = conn.execute("SELECT run_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return [str(row[0])] if row else []
    return [str(row[0]) for row in conn.execute("SELECT run_id FROM runs ORDER BY run_id")]


def _trace_run_ids(run_id: str | None) -> list[str]:
    """@brief 在独立只读连接上取要扫的 run id。

    @param run_id 只看这一场运行；None 表示全部。
    @return run id 列表。
    """
    with borrow() as index:
        return _run_ids(index.conn, run_id)


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


def _run_directories(root: Path) -> list[tuple[str, Path]]:
    """@brief 只读地在 ``runs/`` 下找确实写过 overlay/送达标记的目录。

    索引只覆盖已摄取的 run；真实语料里的 overlay 证据可能还没进索引，所以这里另做
    一次有界目录遍历，只认带标记文件的目录。不读 JSON、不写索引。

    @param root ``runs/`` 根。
    @return ``[(run_id, 目录)]``，按 run_id 排序。
    """
    found: list[tuple[str, Path]] = []
    stack = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        if depth and any((directory / name).exists() for name in _DISCOVERY_MARKERS):
            try:
                found.append((directory.relative_to(root).as_posix(), directory))
            except ValueError:
                continue
        if depth >= _DISCOVERY_DEPTH:
            continue
        try:
            with os.scandir(directory) as entries:
                children = [(entry.name, entry.is_dir()) for entry in entries]
        except OSError:
            continue
        for name, is_dir in children:
            if not is_dir or name.startswith(".") or name in _SKIP_DIRS:
                continue
            stack.append((Path(directory, name), depth + 1))
    found.sort(key=lambda item: item[0])
    return found


def _overlay_run_directories(run_id: str | None) -> list[tuple[str, Path]]:
    """@brief 取 overlay 视图要扫的 run：索引里的 run 与盘上带标记的目录取并集。

    @param run_id 只看这一场运行；None 表示全部。
    @return ``[(run_id, 目录)]``，每项唯一。
    """
    root = detail.root_path()[0]
    with borrow() as index:
        names = _run_ids(index.conn, run_id)
    seen: dict[str, Path] = {}
    for name in names:
        directory = detail.resolve_run(name, root=root)
        if directory is not None:
            seen[name] = directory
    for name, directory in _run_directories(root):
        if run_id and name != run_id:
            continue
        seen.setdefault(name, directory)
    return sorted(seen.items())


def _route(row: dict) -> str:
    """@brief 把一条 overlay/proposal 行分到路线 P（premise）、M 或普通 tip。

    ``harness_proposals.jsonl`` 把 surface 放在内层 ``proposal`` 里，overlay 行放在顶层。

    @param row overlay 或 proposal 行。
    @return ``premise`` / ``M`` / ``tip``。
    """
    if str(row.get("route") or "") == ROUTE_M:
        return ROUTE_M
    surface = str(row.get("surface") or "")
    if not surface and isinstance(row.get("proposal"), dict):
        surface = str(row["proposal"].get("surface") or "")
    return ROUTE_PREMISE if surface == ROUTE_PREMISE else ROUTE_TIP


def _overlay_layer(document: dict) -> dict | None:
    """@brief 从 TASK 送达回执里取 ``overlay`` 那一层。

    @param document ``taic.task_delivery.1`` 回执。
    @return 层字典；没有 situation/layers 时 None。
    """
    situation = document.get("situation")
    if not isinstance(situation, dict):
        return None
    for layer in situation.get("layers") or []:
        if isinstance(layer, dict) and layer.get("name") == "overlay":
            return layer
    return None


def _receipts(directory: Path) -> list[tuple[str, dict]]:
    """@brief 收本 run 的 TASK 送达回执：渲染检查与控制机每场会话各一份。

    @param directory run 目录。
    @return ``[(相对路径, 回执字典)]``，按路径排序。
    """
    paths = [directory / RENDER_RECEIPT, *sorted(directory.glob(SESSION_RECEIPT_GLOB))]
    out = []
    for path in paths:
        document = _cached_object(path)
        if document is not None:
            out.append((str(path.relative_to(directory)), document))
    return out


def _session_index(conn, run_id: str | None) -> dict[str, list[tuple[int, str | None]]]:
    """@brief 一次取出会话 ``(seq, outcome)``，按 run 分组，避免逐 run 查索引。

    @param conn 只读索引连接。
    @param run_id 只看这一场运行；None 表示全部。
    @return ``{run_id: [(seq, outcome)]}``。
    """
    where = " WHERE run_id = ?" if run_id else ""
    args = (run_id,) if run_id else ()
    out: dict[str, list[tuple[int, str | None]]] = {}
    for row in conn.execute(f"SELECT run_id, seq, outcome FROM sessions{where} ORDER BY run_id, seq", args):
        out.setdefault(str(row[0]), []).append((int(row[1]), row[2]))
    return out


def _sessions_after(rows: list[tuple[int, str | None]] | None, active_from: int | None) -> dict | None:
    """@brief 统计激活界限之后的会话结局；索引里没这场 run 就返回 None，不写成 0 场。

    @param rows 该 run 的 ``(seq, outcome)`` 列表。
    @param active_from overlay 行写下的首个生效场次。
    @return 结局计数或 None。
    """
    if not rows or active_from is None:
        return None
    after = [(seq, outcome) for seq, outcome in rows if seq >= active_from]
    outcomes: dict[str, int] = {}
    for _seq, outcome in after:
        name = str(outcome or "unknown")
        outcomes[name] = outcomes.get(name, 0) + 1
    return {
        "sessions": len(after),
        "outcomes": outcomes,
        "active_from": active_from,
        "scope": "该 run 中 seq >= active_from 的会话；描述统计，不是 overlay 效果",
    }


def _entry_unavailable(entry: dict) -> list[str]:
    """@brief 列出一条 overlay 条目里没有盘上证据的环节，供页面显示 unavailable。

    @param entry 单条 proposal 条目。
    @return 缺失环节名列表。
    """
    stages = entry["stages"]
    missing = [name for name in ("shadow", "active", "delivered", "disabled", "sessions_after_active")
               if stages.get(name) is None]
    return missing


def overlay_activity(*, run_id: str | None = None) -> dict:
    """@brief 只读汇总 active overlay：proposal → shadow → active → delivered → disabled。

    只读盘上真实存在的 ``harness_proposals.jsonl``、``harness_overlay.jsonl``、
    ``harness_decisions.jsonl``、``harness_control.json`` 与 TASK 送达回执；
    “激活后的会话结果”只取索引里该 run 的会话结局。没有事件的历史 run 记
    ``coverage=unavailable`` 或逐环节 ``None``，不写成 0。路线 M 与 premise 路线 P
    分开汇总。激活不是效果，这里不计算任何有用率。

    @param run_id 只看这一场运行；None 表示全部。
    @return 含 ``routes`` / ``entries`` / ``coverage`` 的只读文档。``entries`` 只列已经走到
        shadow/active/decision 的 key；只有 proposal 的 key 计入 ``routes.*.proposals_only``，
        不逐条展开（``proposals`` 总数仍保留）。
    """
    root = detail.root_path()[0]
    scanned = _overlay_run_directories(run_id)
    entries: list[dict] = []
    proposal_only: dict[str, int] = {route: 0 for route in ROUTES}
    proposal_keys: dict[str, int] = {route: 0 for route in ROUTES}
    runs_scanned = 0
    runs_with_events = 0
    with borrow() as index:
        sessions = _session_index(index.conn, run_id)
        for name, directory in scanned:
            runs_scanned += 1
            proposals = [row for row in _cached_rows(directory / PROPOSAL_FILE) if row.get("event") == "proposal"]
            overlays = [row for row in _cached_rows(directory / OVERLAY_FILE) if row.get("event") == "overlay"]
            decisions = [row for row in _cached_rows(directory / DECISION_FILE) if row.get("event") == "decision"]
            if not (proposals or overlays or decisions):
                continue
            runs_with_events += 1
            control = _cached_object(directory / CONTROL_FILE)
            disabled = None if control is None else bool(control.get("disabled"))
            receipts = []
            for relative, document in _receipts(directory):
                layer = _overlay_layer(document)
                if layer is None:
                    continue
                receipts.append({
                    "receipt": relative,
                    "kind": "render_check" if relative.startswith("render-check/") else "session",
                    "status": str(layer.get("status") or "unknown"),
                    "delivered_bytes": int(layer.get("delivered_bytes") or 0),
                    "omitted_bytes": int(layer.get("omitted_bytes") or 0),
                    "content_sha256": str(layer.get("content_sha256") or ""),
                })
            keys: list[str] = []
            for row in (*proposals, *overlays, *decisions):
                key = str(row.get("proposal_key") or "")
                if key and key not in keys:
                    keys.append(key)
            for key in keys:
                proposal_rows = [row for row in proposals if str(row.get("proposal_key") or "") == key]
                overlay_rows = [row for row in overlays if str(row.get("proposal_key") or "") == key]
                decision_rows = [row for row in decisions if str(row.get("proposal_key") or "") == key]
                proposal = proposal_rows[-1] if proposal_rows else None
                head = proposal or (overlay_rows[0] if overlay_rows else {})
                route = _route(head)
                proposal_keys[route] += 1
                if not (overlay_rows or decision_rows):
                    # 只有 proposal：不进逐条条目，只计数量。
                    proposal_only[route] += 1
                    continue
                active_rows = [row for row in overlay_rows if not bool(row.get("shadow"))]
                shadow_rows = [row for row in overlay_rows if bool(row.get("shadow"))]
                if active_rows and not sessions.get(name):
                    # 会话投影本来是懒加载的（``model_stats`` 也走同一条路）；这里只为真有 active
                    # 行的 run 追平一次，未投影不等于 0 场，所以仍然先补再判断。
                    index.ensure_sessions(directory)
                    sessions[name] = [
                        (int(row[0]), row[1]) for row in index.conn.execute(
                            "SELECT seq, outcome FROM sessions WHERE run_id = ? ORDER BY seq", (name,))
                    ]
                active_from = min(
                    (int(row["active_from"]) for row in active_rows if isinstance(row.get("active_from"), int)),
                    default=None,
                )
                # 送达回执只在有 active 行时才归到这条 proposal；没 active 的 run 不把空回执当送达。
                deliveries = receipts if active_rows else []
                delivered = [item for item in deliveries if item["status"] == "delivered" and item["delivered_bytes"] > 0]
                entry = {
                    "proposal_key": key,
                    "route": route,
                    "surface": str(head.get("surface") or (head.get("proposal") or {}).get("surface") or ""),
                    "target_id": str(head.get("target_id") or ""),
                    "task": str(head.get("task") or ""),
                    "run_id": name,
                    "source_run": str(head.get("source_run") or name),
                    "source_seq": head.get("source_seq", head.get("seq")),
                    "status": (proposal or {}).get("status"),
                    "evidence_level": str(head.get("evidence_level") or "") or None,
                    "trusted_record": (proposal or {}).get("trusted_record"),
                    "probe_time_verified": (proposal or {}).get("probe_time_verified"),
                    "decisions": sorted({str(row.get("decision") or "") for row in decision_rows}) or None,
                    "active_from": active_from,
                    "stages": {
                        "proposal": 1 if proposal_rows else None,
                        "shadow": len(shadow_rows) or None,
                        "active": len(active_rows) or None,
                        "delivered": len(delivered) or None,
                        # 关闭是 run 级开关（harness_control.json），不是逐 key 记录；只对有 overlay 行的 key 显示。
                        "disabled": disabled if overlay_rows else None,
                        "sessions_after_active": _sessions_after(sessions.get(name), active_from)
                        if active_rows else None,
                    },
                    "delivery": deliveries or None,
                }
                entry["unavailable"] = _entry_unavailable(entry)
                entries.append(entry)
    routes = {}
    for route in ROUTES:
        subset = [entry for entry in entries if entry["route"] == route]
        if not proposal_keys[route] and not subset:
            routes[route] = {
                "coverage": "unavailable", "proposals": None, "proposals_only": None, "entries": None,
                "with_shadow": None, "with_active": None, "with_delivery": None, "disabled": None,
                "sessions_after_active": None,
            }
            continue
        routes[route] = {
            "coverage": "partial",
            "proposals": proposal_keys[route],
            "proposals_only": proposal_only[route],
            "entries": len(subset),
            "with_shadow": sum(1 for entry in subset if entry["stages"]["shadow"]),
            "with_active": sum(1 for entry in subset if entry["stages"]["active"]),
            "with_delivery": sum(1 for entry in subset if entry["stages"]["delivered"]),
            "disabled": sum(1 for entry in subset if entry["stages"]["disabled"]),
            "sessions_after_active": sum(1 for entry in subset if entry["stages"]["sessions_after_active"]),
        }
    coverage = "partial" if (entries or any(proposal_keys.values())) else "unavailable"
    return {
        "notice": OVERLAY_NOTICE,
        "coverage": coverage,
        "note": (
            "coverage=partial 表示只统计盘上确实写过 overlay/proposal/decision 文件的 run；"
            "更早的历史 run 没有这些文件，整份视图不把它们补成 0。逐条条目按 (run, proposal_key) 给出"
            "proposal/shadow/active/delivered/disabled 与激活后的会话结局；缺失环节列在 unavailable，"
            "只有 proposal 的 key 只计 proposals_only 不逐条展开。激活、送达和效果是三个不同的量："
            "单个 active run 可以开始计数，但不足以做自动停权或效果判定。本视图不做加权，也不输出有用率。"
        ),
        "runs_scanned": runs_scanned,
        "runs_with_events": runs_with_events or None,
        "routes": routes,
        "entries": entries,
        "effect_rate": None,
        "effect_note": "active 状态不是效果。没有任何字段把 active/送达与 confirmed 或速度比合成为「有用率」。",
    }


def smoke_summary(*, run_id: str | None = None) -> dict:
    """@brief 只读摘出盘上的 P1 配对 smoke 结果；没有记录时整份 unavailable。

    只读 ``paired_smoke_result.json`` 里已经写下的结论字段，不重算 speedup，
    也不把一对样本改写成效果结论。

    @param run_id 只看这一场运行；None 表示全部。
    @return 含 ``records`` 的只读文档。
    """
    root = detail.root_path()[0]
    records = []
    for name, directory in _overlay_run_directories(run_id):
        document = _cached_object(directory / SMOKE_FILE)
        if document is None:
            continue
        cells = document.get("cells") if isinstance(document.get("cells"), dict) else {}
        records.append({
            "run_id": name,
            "schema": document.get("schema"),
            "proposal_key": document.get("proposal_key"),
            "task": document.get("task"),
            "dtype": document.get("dtype"),
            "size": document.get("size"),
            "pairs": document.get("pairs"),
            "excluded_pairs": document.get("excluded_pairs"),
            "total_cost_usd": document.get("total_cost_usd"),
            "budget_cap_usd": document.get("budget_cap_usd"),
            "verdict": document.get("verdict"),
            "publication_allowed": document.get("publication_allowed"),
            "speedup_claim_allowed": document.get("speedup_claim_allowed"),
            "adopt_overlay_for_effect": document.get("adopt_overlay_for_effect"),
            "followup": document.get("followup"),
            "cells": [
                {
                    "arm": str(arm),
                    "model": cell.get("model"),
                    "run_seed": cell.get("run_seed"),
                    "overlay_delivered": cell.get("overlay_delivered"),
                    "confirmed_speedup": cell.get("confirmed_speedup"),
                    "confirmed_status": cell.get("confirmed_status"),
                    "cost_usd": cell.get("cost_usd"),
                    "algorithm": cell.get("algorithm"),
                }
                for arm, cell in cells.items() if isinstance(cell, dict)
            ] or None,
        })
    return {
        "coverage": "available" if records else "unavailable",
        "note": (
            "只读 paired_smoke_result.json 的已写结论；一对样本的作用是行为观察，不是效果证据。"
            "active 状态、送达与 confirmed 数字不合成任何率。"
        ),
        "runs_scanned": len(_overlay_run_directories(run_id)),
        "records": records,
        "effect_rate": None,
    }


def _candidate_rows(conn, run_id: str | None) -> dict[tuple[str, str], tuple[int, int, int]]:
    """@brief 按 provider/model 取候选关联数、明确 ok 数与明确非 ok 数。

    关联口径与 ``observe.analysis.committees`` 相同：``c.run_id = s.run_id AND
    c.label = s.candidate_label``。``old_format``/``unreadable``/``missing`` 只是
    报告读取状态，``unchanged``/``duplicate`` 则未送评测；它们都保持不可判定，
    不进入 ok 比例的分母，也不能被显示成非 ok。

    @param conn 只读索引连接。
    @param run_id 只看这一场运行；None 表示全部。
    @return ``{(provider, model): (关联会话数, ok 会话数, 明确非 ok 会话数)}``。
    """
    where = " WHERE s.run_id = ?" if run_id else ""
    args = (run_id,) if run_id else ()
    non_ok = ",".join("?" for _ in NON_OK_STATUSES)
    rows = conn.execute(
        f"""
        SELECT s.provider AS provider, s.model AS model,
               COUNT(DISTINCT s.run_id || '/' || s.seq) AS linked,
               COUNT(DISTINCT CASE WHEN c.status = 'ok' THEN s.run_id || '/' || s.seq END) AS ok,
               COUNT(DISTINCT CASE WHEN c.status IN ({non_ok}) THEN s.run_id || '/' || s.seq END) AS non_ok
        FROM sessions s
        JOIN candidates c ON c.run_id = s.run_id AND c.label = s.candidate_label
        {where}
        GROUP BY s.provider, s.model
        """,
        (*sorted(NON_OK_STATUSES), *args),
    ).fetchall()
    return {
        (str(row["provider"] or ""), str(row["model"] or "")): (
            int(row["linked"] or 0), int(row["ok"] or 0), int(row["non_ok"] or 0),
        )
        for row in rows
    }


def _session_rows(conn, run_id: str | None) -> dict[tuple[str, str], dict]:
    """@brief 按 provider/model 汇总会话结局与候选 ok；未知 outcome 单列，不并入未交核。

    ``evaluated`` 来自显式 outcome；``ok_sessions`` 来自同一会话标签关联的候选
    ``status='ok'``。两者不互相推断：未交核只认 ``agent_no_submission``。

    @param conn 只读索引连接。
    @param run_id 只看这一场运行；None 表示全部。
    @return ``{(provider, model): 结局与候选 ok 计数}``。
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
    candidates = _candidate_rows(conn, run_id)
    for key, item in out.items():
        linked, ok, non_ok = candidates.get(key, (0, 0, 0))
        known = ok + non_ok
        item["ok_known_sessions"] = known
        item["ok_sessions"] = ok if known else None
        item["non_ok_sessions"] = non_ok if known else None
        item["ok_unknown_sessions"] = linked - known
        item["ok_unlinked_sessions"] = item["sessions"] - linked
        item["ok_rate"] = (ok / known) if known else None
        item["ok_coverage"] = (known / item["sessions"]) if item["sessions"] else None
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
        # 这一模型可以只有 premise='' 的反馈。那说明没有任何前提声明，
        # 不是“声明过且三类各为 0”；计数和比例都保持 unknown。
        out[(str(row["provider"] or ""), str(row["model"] or ""))] = (
            {
                **counts,
                "total": total,
                "downgrade_rate": (counts["downgraded"] / total) if total else None,
            }
            if total else
            {key: None for key in (*PREMISE_KINDS, "total", "downgrade_rate")}
        )
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
    @return 含 ``health_notice``、``never_auto``、``overlay`` 与 ``smoke`` 的字典；没有模型行时仍给出说明。
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
            "ok_sessions": session.get("ok_sessions"),
            "non_ok_sessions": session.get("non_ok_sessions"),
            "ok_known_sessions": session.get("ok_known_sessions"),
            "ok_unknown_sessions": session.get("ok_unknown_sessions"),
            "ok_unlinked_sessions": session.get("ok_unlinked_sessions"),
            "ok_rate": session.get("ok_rate"),
            "ok_coverage": session.get("ok_coverage"),
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
    return {
        "health_notice": NOTICE,
        "never_auto": gate_document,
        "overlay": overlay_activity(run_id=run_id),
        "smoke": smoke_summary(run_id=run_id),
    }
