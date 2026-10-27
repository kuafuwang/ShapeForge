"""§8.1-8.2 发布史读模型：目录史、当前发布判定、可复现来源与三类告警。

数据源是 ``kernel_history``（由 ``Index._sync_kernels`` 从 ``kernels/*/*/*/*/history.jsonl`` 投影，
``updated_at`` 是确认时刻）。两个数必须分开命名：``catalog_confirmed_speedup`` 是历史事实，
``current_publication`` / ``current_published_speedup`` 是 ``conclusion_of_run`` 的当前判定。

两种读法必须分开：

* ``publications()`` 是**面板**：为了显示当前判定与来源可用性，它读 run 目录里的
  ``result.json`` / ``confirm_report.json`` / 候选内核；戳必须覆盖这些依赖。
* ``publication_alert_report()`` 是**巡检**：只用索引窄投影（``kernel_history`` ⋈ ``runs``），
  不打开任何 run 目录，也不解码 ``row_json``。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from experiments.common import publication_reason
from observe import detail
from observe.compare_key import protocol_epoch
from taic.kernel_catalog import blob_path
from observe.index import (
    _best_fingerprints,
    _candidate_files_fingerprint,
    borrow,
    get_index,
)

CATALOG_EPS = 1e-9
PUBLICATION_ROWS = 200
PUBLICATION_ALERT_ROWS = 50


def _num(value) -> float | None:
    """@brief 只接受有限数值（bool 不算）。

    @param value 原始值。
    @return 有限浮点数，或 None。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _stamp(parts: list[str]) -> str:
    """@brief 发布史面板的戳：目录史代际与每行实际读的源都进。

    @param parts 源指纹与过滤条件。
    @return 十六进制摘要。
    """
    return hashlib.sha256("\x1f".join(["publications-v2", *parts]).encode("utf-8")).hexdigest()


def comparability_key(row: dict, epoch: str) -> tuple[str, str, str, str, str, str, str]:
    """@brief 发布史的环比键：``(task, dtype, size, target_id, task_fingerprint, calibration_sha256, epoch)``。

    目录史没有协议纪元，纪元从解析到的 run 派生（``flame_chase_1 → 2``）；解析不到时是
    ``unknown``，这种行不参与环比（§8.1 第 7 条）。

    @param row kernel_history 行。
    @param epoch 解析到的运行的协议纪元。
    @return 七元组。
    """
    return (
        str(row.get("task") or ""),
        str(row.get("dtype") or ""),
        str(row.get("size") or ""),
        str(row.get("target_id") or ""),
        str(row.get("task_fingerprint") or ""),
        str(row.get("calibration_sha256") or ""),
        epoch or "unknown",
    )


def source_state(directory: Path | None, winner_label: str, source_sha256: str) -> tuple[bool, bool]:
    """@brief 目录史那一行能否定位到唯一一份候选内核（§8.2 的 sha 消歧）。

    返回 ``(source_available, source_ambiguous)``：同 label 的候选只有一份且 sha 匹配 → 可用；
    多于一份时用 sha 再筛，仍不唯一 → ``source_ambiguous=True``（不随便挑一个）。只给面板用；
    巡检不调用它。

    @param directory 解析到的运行目录，可为 None。
    @param winner_label 胜出候选标签。
    @param source_sha256 目录史记下的内核摘要。
    @return ``(可用, 有歧义)``。
    """
    if directory is None:
        return False, False
    folder = directory / "candidates"
    try:
        dirs = sorted(item for item in folder.iterdir() if item.is_dir())
    except OSError:
        return False, False
    matched = [item for item in dirs if not winner_label or item.name.endswith("_" + winner_label)]
    if not matched:
        return False, False
    if source_sha256:
        hashed = []
        for item in matched:
            for name in detail.candidate_kernel_files(item):
                path = item / name
                try:
                    if hashlib.sha256(path.read_bytes()).hexdigest() == source_sha256:
                        hashed.append(item)
                        break
                except OSError:
                    continue
        matched = hashed
    if len(matched) == 1:
        return True, False
    return False, len(matched) > 1


def _rows() -> tuple[list[dict], int, int]:
    """@brief 读 ``kernel_history`` 全部行（按目录与行序）与两个代际。

    @return ``(行列表, revision, catalog_revision)``。
    """
    with borrow() as index:
        rows = [
            dict(row) for row in index.conn.execute(
                "SELECT target_id, task, dtype, size, seq, run_id_raw, run_id, winner_label,"
                " source_sha256, confirmed_speedup, confirmed_values_json, updated_at, source_unknown"
                " FROM kernel_history ORDER BY task, dtype, size, target_id, seq"
            )
        ]
        return rows, index.revision(), index.catalog_revision()


def _run_epoch(directory: Path | None) -> str:
    """@brief 解析到的运行的协议纪元（result → run_config → state 依次找 ``protocol_version``）。

    目录史本身没有纪元；解析不到运行或三份文件都没记时返回空串（调用方归到 ``unknown``）。

    @param directory 运行目录，可为 None。
    @return 纪元字符串。
    """
    if directory is None:
        return ""
    for name in ("result.json", "run_config.json", "state.json"):
        doc = detail._load(directory / name) or {}
        if doc.get("protocol_version"):
            return protocol_epoch(doc)
    return ""


def _current(directory: Path | None) -> tuple[str, float | None, str, float | None]:
    """@brief 当前发布判定与其原因码，以及 ``result.confirmed_speedup``（catalog_mismatch 用）。

    @param directory 运行目录，可为 None。
    @return ``(publication, published_speedup, reason, result_confirmed_speedup)``。
    """
    if directory is None:
        return "", None, "no_result_json", None
    result = detail._load(directory / "result.json")
    verdict = publication_reason(directory, result)
    conclusion = detail.conclusion_of_run(directory, result) if result else {"speedup": None}
    confirmed = _num((result or {}).get("confirmed_speedup"))
    return str(verdict.get("publication") or ""), _num(conclusion.get("speedup")), str(verdict.get("reason") or ""), confirmed


def _identity(row: dict) -> str:
    """@brief 一行目录史里所有直接进输出的列。

    @param row kernel_history 行。
    @return 用 0x1e 连接的文本。
    """
    return "\x1e".join(str(row.get(key)) for key in (
        "target_id", "task", "dtype", "size", "seq", "run_id_raw", "run_id", "winner_label",
        "source_sha256", "confirmed_speedup", "updated_at", "source_unknown",
    ))


def _stat(path: Path) -> str:
    """@brief 一个文件的 ``(mtime_ns,size)``；读不到时给稳定占位。

    @param path 文件路径。
    @return 指纹文本。
    """
    try:
        stat = path.stat()
    except OSError:
        return "0:0"
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _run_dependencies(directory: Path) -> str:
    """@brief 面板一行实际读的每一份 run 文件（stat 级，不读内容）。

    ``publication_reason`` 读 ``result.json`` / ``confirm_report.json`` / ``best.*`` / 候选内核；
    ``_run_epoch`` 还读 ``run_config.json`` 与 ``state.json``；来源可用性看 ``candidates/``。
    把这些都纳入戳，任一被就地改写都会让旧戳失效——只盯 ``result.json`` 会漏掉确认报告与候选。

    @param directory 运行目录。
    @return 指纹文本。
    """
    parts = [f"{name}:{_stat(directory / name)}" for name in
             ("result.json", "confirm_report.json", "run_config.json", "state.json")]
    parts.append("best:" + _best_fingerprints(directory))
    parts.append("cands:" + _candidate_files_fingerprint(directory))
    return "|".join(parts)


def _row_delta(speed: float | None, previous: float | None) -> tuple[float | None, bool]:
    """@brief 相邻目录史值的展示百分比与回退判定。

    回退判定用**未舍入**的值与 §8.1 的公式（``catalog[i] < catalog[i-1] - 1e-9``）；百分比只用于
    展示，所以先换成百分比再四舍五入会漏掉微小下降（1.0 → 0.999999998 显示 ``-0.0``）。

    @param speed 当前目录史值。
    @param previous 同组上一条可解析的目录史值。
    @return ``(展示百分比, 是否回退)``。
    """
    if speed is None or previous is None:
        return None, False
    delta = round((speed / previous - 1.0) * 100.0, 4) if previous > 0 else None
    return delta, speed < previous - CATALOG_EPS


def publications(*, task: str | None = None, target: str | None = None, dtype: str | None = None,
                 size: str | None = None, stamp: str = "") -> dict:
    """@brief 发布史端点：每行目录事实 + 环比 + 当前发布判定 + 来源可用性。

    环比只在同一个 comparability_key 内、且两端都能解析到运行时计算；``run_id="run"`` 这类解析不到
    的行 ``source_unknown=1``，不进环比、不进告警，端点里明写。

    戳在打开任何 run 文件之前算：目录史代际 + 每行直接列 + 每行解析到的 run 的依赖 stat。戳匹配时
    连 ``result.json`` 都不读。

    @param task 只取该任务。
    @param target 只取该目标。
    @param dtype 只取该 dtype。
    @param size 只取该尺寸。
    @param stamp 客户端持有的戳。
    @return 面板字典；戳匹配时只回 ``same``。
    """
    get_index().ensure_kernel_history()
    rows, revision, catalog_revision = _rows()
    filtered = [
        row for row in rows
        if (not task or row["task"] == task) and (not target or row["target_id"] == target)
        and (not dtype or row["dtype"] == dtype) and (not size or row["size"] == size)
    ]
    directories = [
        detail.resolve_run(row["run_id"]) if row["run_id"] else None for row in filtered
    ]
    source_parts = [
        str(revision), str(catalog_revision), task or "", target or "", dtype or "", size or "",
    ]
    for row, directory in zip(filtered, directories):
        dependencies = _run_dependencies(directory) if directory is not None else "-"
        sha = str(row["source_sha256"] or "")
        blob = int(blob_path(sha).is_file()) if sha else 0
        source_parts.append(f"{_identity(row)}>>{dependencies}>>blob:{blob}")
    document_stamp = _stamp(source_parts)
    if stamp and stamp == document_stamp:
        return {"same": True, "stamp": document_stamp}
    prepared = []
    for row, directory in zip(filtered, directories):
        epoch = _run_epoch(directory)
        publication, published, reason, confirmed = _current(directory)
        available, ambiguous = source_state(directory, row["winner_label"] or "", row["source_sha256"] or "")
        sha = str(row["source_sha256"] or "")
        prepared.append({
            "row": row,
            "key": comparability_key(row, epoch),
            "publication": publication,
            "published": published,
            "reason": reason,
            "confirmed_speedup": confirmed,
            "available": available,
            "ambiguous": ambiguous,
            # §9.1：源目录被删/搬走后，blob 仍是唯一副本；两个可用性必须分开报。
            "blob": bool(sha) and blob_path(sha).is_file(),
        })
    # 环比：同 comparability_key、按目录行序相邻；两端都要能解析（source_unknown=0）。
    grouped: dict[tuple, list[dict]] = {}
    for item in prepared:
        grouped.setdefault(item["key"], []).append(item)
    output = []
    for key, items in grouped.items():
        best_so_far = None
        previous = None
        for item in items:
            speed = _num(item["row"]["confirmed_speedup"])
            unknown = bool(item["row"]["source_unknown"])
            delta, regressed = _row_delta(speed, None if unknown else previous)
            if speed is not None:
                if best_so_far is None or speed > best_so_far + CATALOG_EPS:
                    is_best = True
                    best_so_far = speed
                else:
                    is_best = False
            else:
                is_best = False
            output.append({
                "updated_at": _num(item["row"]["updated_at"]),
                "task": key[0],
                "dtype": key[1],
                "size": key[2],
                "target_id": key[3],
                "comparability_key": list(key),
                "catalog_confirmed_speedup": speed,
                "delta_vs_prev": delta,
                "regressed": regressed,
                "is_best_so_far": is_best,
                "current_publication": item["publication"],
                "current_published_speedup": item["published"],
                "publication_reason": item["reason"],
                "run_id": item["row"]["run_id"],
                "run_id_raw": item["row"]["run_id_raw"],
                "winner_label": item["row"]["winner_label"],
                "source_unknown": int(item["row"]["source_unknown"] or 0),
                "source_available": bool(item["available"]),
                "source_ambiguous": bool(item["ambiguous"]),
                "blob_available": bool(item["blob"]),
            })
            if speed is not None and not unknown:
                previous = speed
    document = {
        "schema_version": "1",
        "rows": output[:PUBLICATION_ROWS],
        "total": len(output),
        "truncated": {"rows": len(output) > PUBLICATION_ROWS},
        "stamp": document_stamp,
    }
    return {"same": False, **document}


def _alert_rows() -> list[dict]:
    """@brief 告警用的索引投影：``kernel_history`` ⋈ ``runs``（窄列、不解码 ``row_json``）。

    发布原因与 ``result.confirmed_speedup`` 在摄入时写成列；纪元取 ``runs.protocol_version``
    （``flame_chase_1 → 2``）。不解析 run 目录，所以巡检可以放心调。

    @return 每一行目录史都带 run 的发布判定、原因码与目录纪元。
    """
    with borrow() as index:
        return [
            dict(row) for row in index.conn.execute(
                "SELECT h.target_id, h.task, h.dtype, h.size, h.seq, h.run_id, h.run_id_raw,"
                " h.winner_label, h.confirmed_speedup, h.task_fingerprint, h.calibration_sha256,"
                " h.updated_at, h.source_unknown,"
                " r.protocol_version AS protocol_version, r.status AS run_status,"
                " r.publication AS run_publication, r.publication_reason AS run_reason,"
                " r.result_confirmed_speedup AS result_confirmed_speedup"
                " FROM kernel_history h LEFT JOIN runs r ON r.run_id = h.run_id"
                " ORDER BY h.task, h.dtype, h.size, h.target_id, h.seq"
            )
        ]


def _row_epoch(row: dict) -> str:
    """@brief 告警行的协议纪元（索引列）。

    @param row ``_alert_rows`` 的一行。
    @return 纪元字符串；没有来源时为 ``unknown``。
    """
    version = str(row.get("protocol_version") or "")
    return protocol_epoch({"protocol_version": version}) if version else "unknown"


def _alert_reasons(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """@brief 三类告警的**完整**列表（互不替代），供分页与巡检各自计数。

    @param rows ``_alert_rows`` 的全部行。
    @return ``(回退, 发布丢失, 目录不一致)``。
    """
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(comparability_key(row, _row_epoch(row)), []).append(row)
    regressed: list[dict] = []
    for items in grouped.values():
        previous = None
        for item in items:
            speed = _num(item["confirmed_speedup"])
            if speed is None or item["source_unknown"]:
                continue
            delta, dropped = _row_delta(speed, previous)
            if dropped:
                regressed.append({
                    "kind": "confirmation_regressed", "run": item["run_id"], "task": item["task"],
                    "delta_pct": delta, "reason": "confirmation_regressed",
                })
            previous = speed
    lost: list[dict] = []
    mismatch: list[dict] = []
    for item in rows:
        catalog = _num(item["confirmed_speedup"])
        if item["source_unknown"] or catalog is None or not item["run_id"]:
            continue
        if not item.get("run_status") or item["run_status"] == "missing":
            # 没有 result.json / 运行已消失的行从来没有发布过，不算“丢失”
            # （§8.1 验收的分类不能重叠成 6 条）。
            continue
        reason = str(item.get("run_reason") or "")
        if reason == "no_result_json":
            continue
        if not reason:
            # 索引里的运行行还没被重新摄取（迁移后的旧库）时原因码为空：不知道就不报“丢了”，
            # 不能编一个 ``not_confirmed`` 把三条都当成丢失（§8.1 验收恰好是 2 条）。
            continue
        if str(item.get("run_publication") or "") != "confirmed":
            lost.append({
                "kind": "publication_lost", "run": item["run_id"], "task": item["task"],
                "reason": reason,
            })
        recorded = _num(item.get("result_confirmed_speedup"))
        if recorded is not None and abs(catalog - recorded) > CATALOG_EPS:
            mismatch.append({
                "kind": "catalog_mismatch", "run": item["run_id"], "task": item["task"],
                "reason": "catalog_mismatch",
            })
    return regressed, lost, mismatch


def publication_alert_report(limit: int | None = None) -> dict:
    """@brief 发布告警的完整计数 + 每类有界样本（巡检据此判断截断）。

    统计在**完整过滤集**上做（不是面板的 200 行）：``counts`` 是三类各自的真实条数，
    ``alerts`` 每类最多 ``limit``（默认 ``PUBLICATION_ALERT_ROWS``）条。巡检只需要前几条，
    但必须知道“还有多少没显示”，否则会静默丢告警。

    @param limit 每类最多返回几条；None 表示用 ``PUBLICATION_ALERT_ROWS``。
    @return ``{"alerts", "counts", "total", "truncated"}``。
    """
    capped = PUBLICATION_ALERT_ROWS if limit is None else max(0, int(limit))
    regressed, lost, mismatch = _alert_reasons(_alert_rows())
    groups = (
        ("confirmation_regressed", regressed),
        ("publication_lost", lost),
        ("catalog_mismatch", mismatch),
    )
    alerts: list[dict] = []
    counts: dict[str, int] = {}
    for name, items in groups:
        counts[name] = len(items)
        alerts.extend(items[:capped])
    return {
        "alerts": alerts,
        "counts": counts,
        "total": sum(counts.values()),
        "truncated": any(len(items) > capped for _name, items in groups),
    }


def publication_alerts(limit: int = 2) -> list[dict]:
    """@brief 三类发布告警（每类最多 ``limit`` 条），供 ``patrol.alerts`` 使用。

    ① ``confirmation_regressed``：同一 comparability_key 内相邻两条目录史下降（未舍入值按
    §8.1 的 1e-9 判）；② ``publication_lost``：目录史说确认过，但当前 ``conclusion_of_run``
    不发布；③ ``catalog_mismatch``：目录史与 ``result.confirmed_speedup`` 不一致。
    解析不到来源的行不进告警。判定全部来自索引投影，不打开 run 目录。

    @param limit 每类最多几条。
    @return 告警列表。
    """
    return publication_alert_report(limit=limit)["alerts"]
