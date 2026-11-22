"""Read-only H1/H2 harness scorecard over existing runs and targets.

The scorecard answers a narrow question: what can the files already on disk
prove about search efficiency, model/session efficiency, delivery and feedback,
and experience/governance? It never opens the ``observe`` index, never writes a
derived database, never calls a model and never runs a GPU kernel.

Four tables are kept separate on purpose. ``run_efficiency``,
``model_session_efficiency``, ``delivery_and_feedback`` and
``experience_and_governance`` have different statistical units. In particular
delivery receipts, claim evidence and route-M traces never share a denominator,
and run-level ledger cost is never added to ``model_totals``.

Every group carries three independent dimensions:

* ``evidence_kind``: ``real`` / ``simulated`` / ``unavailable`` (declared only);
* ``metric_status`` per ratio: ``measured`` / ``insufficient`` / ``unavailable``;
* ``coverage`` per ratio: independent units and known units.

Missing values stay ``None`` (JSON ``null``). Speedups, latencies and
cost-per-ok are never averaged across tasks, shapes, dtypes or measurement
boundaries. H0's frozen sufficiency thresholds are reused, not retuned here.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from observe.health import (
    EVALUATED,
    NO_SUBMISSION,
    NON_OK_STATUSES,
    OTHER_FAILURE_OUTCOMES,
    PREMISE_KINDS,
    ROUTE_M_TRACE_FILE,
    UNCHANGED,
)

from . import dynamic_budget_audit
from .evidence_level import summarize_levels
from .experience_scope import audit_experience_file
from .n1_metrics_audit import _selection_documents
from .route_m_recheck_audit import _first_blocker
from .scorecard_contract import (
    IDENTITY_FIELDS,
    SUFFICIENCY_THRESHOLDS,
    canonical_session_label,
    compare_identity,
    cost_metrics,
    metric_status,
    resolve_candidate_status,
)
from .scorecard_inventory import classify_run, discover_run_roots
from .seed_decision_audit import _evaluate as _pair_evaluate

SCHEMA = "shapeforge.harness-scorecard.1"
RETRY_LABEL_NOTE = "`.retryN` 映射回逻辑会话，原始标签保留在 session_label_raw；每个逻辑会话只计一次。"
STATIC_CONTRACT_NOTE = "静态合同省略必须为 0；这里是字节计数与违例数，不是省略率。"
DELIVERY_NOTE = "字节送达不等于模型已理解，也不等于该信息有效。"
SPEEDUP_NOTE = "speedup 只列单 run 值并带不可比标记；不跨 run/task/shape/窗口求平均。"
NPU_NOTE = "runs/ 没有 NPU 自然 run 时设备指标是 unavailable，不从模拟 fixture 补行。"
TRUST_NOTE = (
    "trust 的 prompt_sha1 是提示面 SHA-256；历史 run_config.system_prompt_sha1 是编排器对当次 self.system 的 SHA-1。"
    "两者不互相比较，也不合成 target 级授权布尔值。"
)
CONTROL_NOTE = "旧 report.json 没有 benchmark.control 时覆盖率是 unavailable，不记成 0 次 A/A 对照。"
IDENTITY_NOTE = "只有两侧同一字段都真实存在且冲突才计 mismatch；缺字段的旧记录是 partial/unavailable。"
# Recognised paired conclusions. A missing/unknown conclusion is missing data,
# not a known "undecidable" observation; only these values count as known.
PAIR_CONCLUSIONS = frozenset({"diff", "no_diff", "regressed", "too_noisy"})

_KEY_FIELDS_RUN = (
    "target_id", "backend", "task", "dtype", "size", "committee_signature",
    "cohort", "experiment_kind", "experiment_id", "experiment_fingerprint", "experiment_group",
    "evidence_kind", "lifecycle",
)
_KEY_FIELDS_SESSION = (
    "target_id", "backend", "task", "dtype", "size", "provider", "model", "committee",
    "cohort", "experiment_kind", "experiment_id", "experiment_fingerprint", "experiment_group",
    "evidence_kind", "lifecycle",
)
_KEY_FIELDS_RECORD = (
    "target_id", "backend", "task", "cohort", "experiment_kind", "experiment_id",
    "experiment_fingerprint", "experiment_group", "evidence_kind",
)


def _json(path: Path) -> dict[str, Any]:
    """@brief 容错读取一个 JSON 对象；坏文件按缺失处理。"""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _jsonl(path: Path) -> list[dict[str, Any]] | None:
    """@brief 读取 JSONL；文件不存在返回 None，坏行跳过但不丢整份记录。"""
    path = Path(path)
    if not path.is_file():
        return None
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _finite(value: Any) -> float | None:
    """@brief 仅接受有限数值；bool、字符串和 NaN/Inf 都按缺失处理。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _int(value: Any) -> int | None:
    """@brief 非负整数计数；缺失保持 None，而不是补 0。"""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _bytes(value: Any) -> int:
    """@brief 回执字节计数；缺失按 0 字节累加，调用方另有完整性标记。"""
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _ratio(part: int, whole: int) -> float | None:
    """@brief 比例；分母为 0 时保持 None。"""
    return part / whole if whole else None


def _count_or_none(groups: list[dict[str, Any]], key: str) -> int | None:
    """@brief 汇总一个可缺失的计数；没有分组、任一组缺字段或该字段是 null 时整体为 null。

    文本摘要不能把"没有数据"打印成 0：那会把缺失证据说成实测零。只有每个组都带
    真实整数计数时求和才是完整值，真实的和为 0 才显示 0。
    """
    if not groups:
        return None
    total = 0
    for group in groups:
        counts = group.get("counts") if isinstance(group, dict) else None
        value = counts.get(key) if isinstance(counts, dict) else None
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        total += value
    return total


def _shown(value: Any) -> str:
    """@brief 文本摘要里的缺失显示 null，不打印 Python 的 None 或补一个 0。"""
    return "null" if value is None else str(value)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def _status_family(status: Any) -> str:
    """@brief 候选状态分组；读取状态与未送评测状态不进入 ok 率分母。"""
    value = str(status or "")
    if value == "ok":
        return "ok"
    if value in NON_OK_STATUSES:
        return "non_ok"
    if value in {UNCHANGED, "duplicate", "external_duplicate"}:
        return "not_evaluated"
    if value in {"old_format", "unreadable", "missing"}:
        return "unreadable"
    if not value:
        return "unavailable"
    return "other"


def _committee_signature(config: dict[str, Any]) -> str | None:
    """@brief 从 run_config 的 first/second 委员会生成稳定签名，不归到单个模型。"""
    committees = config.get("committees")
    if not isinstance(committees, list) or not committees:
        return None
    parts = []
    for committee in sorted(
        (item for item in committees if isinstance(item, dict)),
        key=lambda item: str(item.get("name") or ""),
    ):
        name = str(committee.get("name") or "")
        members = sorted(
            f"{member.get('provider')}/{member.get('model')}"
            for member in committee.get("members") or []
            if isinstance(member, dict)
        )
        parts.append(f"{name}:" + ",".join(members))
    return ";".join(parts) if parts else None


def _experiment_group(row: dict[str, Any]) -> str | None:
    """@brief 有 experiment_id 用它；旧 plan 用字节指纹兜底，不伪造原 ID。"""
    if row.get("experiment_id"):
        return str(row["experiment_id"])
    if row.get("experiment_fingerprint"):
        return f"plan-sha256:{row['experiment_fingerprint']}"
    return None


def _key(row: dict[str, Any], fields: tuple[str, ...], **extra: Any) -> tuple:
    values = []
    for name in fields:
        if name in extra:
            values.append(extra[name])
        elif name == "experiment_group":
            values.append(_experiment_group(row))
        else:
            values.append(row.get(name))
    return tuple(values)


def _key_document(fields: tuple[str, ...], key: tuple) -> dict[str, Any]:
    return dict(zip(fields, key))


def _sort_token(value: Any) -> str:
    return "" if value is None else str(value)


def _group_items(groups: dict[tuple, Any], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {"key": _key_document(fields, key), "bucket": bucket}
        for key, bucket in sorted(groups.items(), key=lambda item: tuple(_sort_token(part) for part in item[0]))
    ]


def _ratio_blocks(
    entries: dict[str, tuple[float | None, int, int, str]],
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, Any]]]:
    """@brief 把每项比例连同独立单位与 H0 阈值判定一起输出。"""
    ratios: dict[str, Any] = {}
    statuses: dict[str, str] = {}
    coverage: dict[str, dict[str, Any]] = {}
    for name, (value, units, known, metric) in entries.items():
        ratios[name] = value
        statuses[name] = metric_status(metric, units=units, known_units=known)
        coverage[name] = {"units": units, "known_units": known}
    return ratios, statuses, coverage


def _identity_fields(document: dict[str, Any]) -> dict[str, Any]:
    """@brief 只抽取文件里真实存在的身份字段；缺失字段不填占位值。"""
    if not isinstance(document, dict):
        return {}
    build = document.get("build") if isinstance(document.get("build"), dict) else {}
    provenance = build.get("provenance") if isinstance(build.get("provenance"), dict) else {}
    artifacts = build.get("artifacts") if isinstance(build.get("artifacts"), dict) else {}
    evaluation = document.get("evaluation") if isinstance(document.get("evaluation"), dict) else {}
    values = {
        "candidate": build.get("candidate_sha256") or document.get("candidate_sha256"),
        "geometry": build.get("geometry_digest") or build.get("execution_geometry_digest"),
        "bundle": build.get("bundle_digest"),
        "artifact": artifacts.get("artifact_set_digest") or provenance.get("artifact_set_digest"),
        "native": build.get("native_digest") or build.get("native"),
        "evaluation": document.get("evaluation_fingerprint") or evaluation.get("evaluation_fingerprint"),
    }
    return {name: value for name, value in values.items() if value not in (None, "")}


def _confirmation_row(path: Path) -> dict[str, Any]:
    """@brief 单份确认记录：旧 identity.kind 与真实字段比较分开报告。"""
    document = _json(path)
    expected = document.get("expected") if isinstance(document.get("expected"), dict) else {}
    report = document.get("confirmation_report") if isinstance(document.get("confirmation_report"), dict) else {}
    expected_fields = _identity_fields(expected)
    actual_fields = _identity_fields(report)
    comparison = compare_identity(expected_fields, actual_fields)
    legacy = document.get("identity") if isinstance(document.get("identity"), dict) else {}
    legacy_kind = legacy.get("kind")
    if (
        comparison["status"] == "match"
        and len(expected_fields) == len(IDENTITY_FIELDS)
        and len(actual_fields) == len(IDENTITY_FIELDS)
    ):
        field_status = "full_match"
    elif comparison["status"] == "mismatch":
        field_status = "mismatch"
    elif comparison["status"] == "partial":
        field_status = "partial"
    else:
        field_status = "unavailable"
    provider = str(document.get("provider") or "")
    fallback_id = document.get("fallback_id")
    return {
        "path": path,
        "field_status": field_status,
        "identity_scope": comparison["identity_scope"],
        "conflicts": comparison["conflicts"],
        "legacy_kind": legacy_kind,
        "legacy_match_without_full_scope": legacy_kind == "match" and field_status != "full_match",
        "fields_present": len(expected_fields) + len(actual_fields),
        "provider": provider,
        "fallback": bool(fallback_id) or provider not in ("", "candidate"),
        "status": document.get("status"),
        "coverage_kind": document.get("coverage_kind"),
    }


def _confirmation_rows(run_root: Path) -> list[dict[str, Any]]:
    directory = run_root / "confirmations"
    if not directory.is_dir():
        return []
    return [_confirmation_row(path) for path in sorted(directory.glob("*.json")) if path.is_file()]


def _report_rows(run_root: Path) -> list[dict[str, Any]]:
    """@brief 逐个候选报告读取 A/A 对照字段与测量边界，不把 elapsed_s 当模型墙钟。"""
    directory = run_root / "candidates"
    if not directory.is_dir():
        return []
    rows = []
    for path in sorted(directory.glob("*/report.json")):
        document = _json(path)
        benchmark = document.get("benchmark") if isinstance(document.get("benchmark"), dict) else {}
        rows.append({
            "path": path,
            "label": path.parent.name.split("_", 1)[-1],
            "has_control_field": "control" in benchmark,
            "control": benchmark.get("control"),
            "protocol": benchmark.get("protocol"),
            "dispatch": benchmark.get("dispatch"),
            "status": document.get("status"),
            "elapsed_s": _finite(document.get("elapsed_s")),
        })
    return rows


def _feedback_rows(run_root: Path) -> tuple[list[dict[str, Any]] | None, int]:
    """@brief 读取并去重反馈会话；retry 映射回逻辑标签，最后一行代表该会话。"""
    raw_rows = _jsonl(run_root / "feedback.jsonl")
    if raw_rows is None:
        return None, 0
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for index, row in enumerate(raw_rows):
        label = str(row.get("label") or "")
        canonical = canonical_session_label(label)
        if canonical not in grouped:
            order.append(canonical)
        enriched = dict(row)
        enriched["_canonical_label"] = canonical
        enriched["_raw_label"] = label
        enriched["_order"] = index
        grouped[canonical].append(enriched)
    selected = []
    deduplicated = 0
    for canonical in order:
        rows = grouped[canonical]
        rows.sort(key=lambda item: (_finite(item.get("t")) or -1.0, int(item.get("seq") or 0), item["_order"]))
        selected.append(rows[-1])
        deduplicated += len(rows) - 1
    return selected, deduplicated


def _pair_rows(run_root: Path, trace_by_label: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """@brief 自然配对与精确 SHA 重测分开；too_noisy 只算不可判。"""
    counts: Counter = Counter()
    rows = []
    for path in sorted(run_root.rglob("paired_comparison.json")):
        document = _json(path)
        if document.get("applied") is not True:
            counts["not_applied"] += 1
            continue
        counts["applied"] += 1
        label = str(document.get("session_label") or path.parent.name.split("_", 1)[-1])
        trace = trace_by_label.get(label) or trace_by_label.get(canonical_session_label(label)) or {}
        direction, actual, verdict = _pair_evaluate(document, trace)
        conclusion = str(document.get("conclusion") or "")
        known_conclusion = conclusion in PAIR_CONCLUSIONS
        counts[f"conclusion:{conclusion or 'unavailable'}"] += 1
        counts[f"timed_by:{document.get('timed_by') or 'unavailable'}"] += 1
        if known_conclusion:
            counts["conclusion_known"] += 1
        else:
            counts["conclusion_unknown"] += 1
        if conclusion == "too_noisy":
            counts["too_noisy"] += 1
        # The rate is about measurement decidability, not about whether the pair
        # agrees with the seed decision: diff/no_diff/regressed are known
        # conclusions, too_noisy is known undecidable, missing is unknown.
        decidable = known_conclusion and conclusion != "too_noisy"
        if decidable:
            counts["decidable"] += 1
        elif known_conclusion:
            counts["undecidable"] += 1
        if verdict in {"agree", "disagree"}:
            counts[f"verdict:{verdict}"] += 1
        rows.append({
            "path": _relative(path, run_root),
            "session_label": label,
            "conclusion": conclusion or None,
            "conclusion_known": known_conclusion,
            "decidable": decidable,
            "direction": direction,
            "actual_decision": actual,
            "verdict": verdict,
            "timed_by": document.get("timed_by"),
            "elapsed_s": _finite(document.get("elapsed_s")),
            "gpu_seconds": None,
            "cross_window": False,
        })
    rechecks = []
    for path in sorted(run_root.rglob("paired_comparison_recheck.json")):
        wrapper = _json(path)
        document = wrapper.get("document") if isinstance(wrapper.get("document"), dict) else {}
        counts["recheck_files"] += 1
        conclusion = str(document.get("conclusion") or "unavailable")
        counts[f"recheck_conclusion:{conclusion}"] += 1
        if conclusion == "too_noisy":
            counts["recheck_too_noisy"] += 1
        counts[f"recheck_timed_by:{document.get('timed_by') or 'unavailable'}"] += 1
        label = str(document.get("session_label") or path.parent.name)
        trace = trace_by_label.get(label) or trace_by_label.get(canonical_session_label(label)) or {}
        _direction, _actual, verdict = _pair_evaluate(document, trace)
        counts["recheck_decidable" if verdict in {"agree", "disagree"} else "recheck_undecidable"] += 1
        rechecks.append({
            "path": _relative(path, run_root),
            "session_label": document.get("session_label") or path.parent.name,
            "conclusion": document.get("conclusion"),
            "timed_by": document.get("timed_by"),
            "cross_window": True,
        })
    return {"counts": counts, "pairs": rows, "rechecks": rechecks}


def _selection_rows(run_root: Path, selection: dict[Path, list[dict[str, Any]]]) -> dict[str, Any]:
    """@brief GPU 秒只来自 session 正本选择记录；普通配对不冒充 GPU 秒。"""
    rows = selection.get(Path(run_root).resolve(), [])
    known = 0
    unavailable = 0
    total = 0.0
    for document in rows:
        if document.get("ran_gpu") is not True:
            continue
        value = _finite(document.get("gpu_seconds"))
        if value is None:
            unavailable += 1
        else:
            known += 1
            total += value
    return {
        "sessions": len(rows),
        "gpu_seconds_known_sessions": known,
        "gpu_seconds_unavailable_sessions": unavailable,
        "gpu_seconds": total if known else None,
        "complete": unavailable == 0 and known > 0,
    }


def _delivery_receipts(run_root: Path) -> tuple[list[dict[str, Any]], int]:
    """@brief 只把 ``sessions/*/task_delivery.json`` 当会话回执；render-check 单列。"""
    receipts = []
    directory = run_root / "sessions"
    if directory.is_dir():
        for path in sorted(directory.glob("*/task_delivery.json")):
            document = _json(path)
            if document:
                receipts.append({"path": path, "label": path.parent.name, "document": document, "surface": "session"})
    render = int((run_root / "render-check" / "task_delivery.json").is_file())
    return receipts, render


def _delivery_bucket(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    """@brief 按 dynamic_budget_audit 的层名与字节字段汇总一个分组。"""
    counts: Counter = Counter()
    counts.update({
        "receipts": 0,
        "static_delivered_bytes": 0,
        "static_omitted_bytes": 0,
        "static_contract_violations": 0,
        "dynamic_delivered_bytes": 0,
        "dynamic_omitted_bytes": 0,
        "receipts_with_dynamic_omission": 0,
        "receipts_over_budget": 0,
        "receipts_with_missing_static_bytes": 0,
        "receipts_with_missing_dynamic_bytes": 0,
        "block_accounting_mismatches": 0,
        "unknown_layer_rows": 0,
    })
    layers: dict[str, Counter] = {name: Counter() for name in dynamic_budget_audit.LAYER_NAMES}
    blocks: Counter = Counter()
    blocks_by_name: dict[str, Counter] = {}
    reasons: Counter = Counter()
    for receipt in receipts:
        document = receipt["document"]
        counts["receipts"] += 1
        static = document.get("static") if isinstance(document.get("static"), dict) else {}
        dynamic = document.get("dynamic") if isinstance(document.get("dynamic"), dict) else {}
        situation = document.get("situation") if isinstance(document.get("situation"), dict) else {}
        if any(
            not isinstance(static.get(field), int) or isinstance(static.get(field), bool) or static.get(field) < 0
            for field in ("delivered_bytes", "omitted_bytes")
        ):
            counts["receipts_with_missing_static_bytes"] += 1
        if any(
            not isinstance(dynamic.get(field), int) or isinstance(dynamic.get(field), bool) or dynamic.get(field) < 0
            for field in ("delivered_bytes", "omitted_bytes")
        ):
            counts["receipts_with_missing_dynamic_bytes"] += 1
        counts["static_delivered_bytes"] += _bytes(static.get("delivered_bytes"))
        counts["static_omitted_bytes"] += _bytes(static.get("omitted_bytes"))
        if _bytes(static.get("omitted_bytes")):
            counts["static_contract_violations"] += 1
        counts["dynamic_delivered_bytes"] += _bytes(dynamic.get("delivered_bytes"))
        counts["dynamic_omitted_bytes"] += _bytes(dynamic.get("omitted_bytes"))
        if _bytes(dynamic.get("omitted_bytes")):
            counts["receipts_with_dynamic_omission"] += 1
        if dynamic.get("over_budget") is True:
            counts["receipts_over_budget"] += 1
        counts["situation_status:" + str(situation.get("status") or "missing")] += 1
        for field in ("source_bytes", "sanitized_bytes", "output_bytes", "delivered_bytes", "omitted_bytes"):
            counts[f"situation_{field}"] += _bytes(situation.get(field))
        for row in situation.get("layers") or []:
            name = str(row.get("name") or "")
            if name not in layers:
                counts["unknown_layer_rows"] += 1
                continue
            for field in ("source_bytes", "sanitized_bytes", "rendered_bytes", "output_bytes", "delivered_bytes", "omitted_bytes"):
                layers[name][field] += _bytes(row.get(field))
            if _bytes(row.get("omitted_bytes")):
                layers[name]["receipts_with_omission"] += 1
        for block in document.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            name = str(block.get("name") or "unknown")
            expected = _bytes(block.get("expected_bytes"))
            delivered = _bytes(block.get("delivered_bytes"))
            omitted = _bytes(block.get("omitted_bytes"))
            if name != "本轮态势":
                bucket = blocks_by_name.setdefault(name, Counter())
                bucket["expected_bytes"] += expected
                bucket["delivered_bytes"] += delivered
                bucket["omitted_bytes"] += omitted
                if omitted:
                    bucket["receipts_with_omission"] += 1
                blocks["expected_bytes"] += expected
                blocks["delivered_bytes"] += delivered
                blocks["omitted_bytes"] += omitted
                if omitted:
                    blocks["receipts_with_omission"] += 1
                    reasons[str(block.get("omission_reason") or "unknown")] += omitted
            if expected != delivered + omitted:
                counts["block_accounting_mismatches"] += 1
    return {"counts": counts, "layers": layers, "blocks": blocks, "blocks_by_name": blocks_by_name, "reasons": reasons}


def _proposal_records(run_root: Path, context: dict[str, Any]) -> list[dict[str, Any]]:
    """@brief 结构化提案是 claim 证据的单位；没有 key/digest 的自由文本单列不可判。"""
    path = run_root / "harness_proposals.jsonl"
    rows = _jsonl(path) or []
    records = []
    for index, row in enumerate(rows):
        if row.get("event") not in (None, "proposal"):
            continue
        key = str(row.get("proposal_key") or "")
        digest = str(row.get("claim_digest") or "")
        records.append({
            "kind": "proposal",
            "identity": key or digest or f"{context['run_root']}#{index}",
            "proposal_key": key or None,
            "claim_digest": digest or None,
            "structured_key": bool(key or digest),
            "evidence_level": row.get("evidence_level"),
            "simulated": row.get("simulated") is True,
            "provider": row.get("provider"),
            "model": row.get("model"),
            "committee": row.get("committee"),
            "session_label": row.get("label"),
            "path": _relative(path, context["runs_root"]),
            "run_root": context["run_root"],
        })
    return records


def _feedback_claims(feedback: list[dict[str, Any]] | None, context: dict[str, Any]) -> list[dict[str, Any]]:
    """@brief 反馈条目是 premise 的单位；空白 premise 保持 None。"""
    records = []
    for row in feedback or []:
        for index, item in enumerate(row.get("items") or []):
            if not isinstance(item, dict):
                continue
            premise = str(item.get("premise") or "")
            records.append({
                "kind": "feedback_claim",
                "identity": f"{row.get('_raw_label') or row.get('label') or ''}#{index}",
                "premise": premise or None,
                "has_text": bool(str(item.get("text") or "").strip()),
                "digest": item.get("digest"),
                "section": item.get("section"),
                "session_label": row.get("_canonical_label") or row.get("label"),
                "path": _relative(Path(context["runs_root"]) / context["run_root"] / "feedback.jsonl", context["runs_root"]),
                "run_root": context["run_root"],
            })
    return records


def _route_records(run_root: Path, context: dict[str, Any]) -> list[dict[str, Any]]:
    """@brief 真实 route-M 门迹按 trace 计，不与送达回执或 claim 共享分母。"""
    path = run_root / ROUTE_M_TRACE_FILE
    rows = _jsonl(path) or []
    records = []
    for index, row in enumerate(rows):
        if row.get("event") != "route_m":
            continue
        records.append({
            "identity": str(row.get("evidence_id") or f"{context['run_root']}#{index}"),
            "conclusion": row.get("conclusion"),
            "direction": row.get("direction"),
            "evidence_level": row.get("evidence_level"),
            "simulated": row.get("simulated") is True,
            "shadow_written": row.get("shadow_written") is True,
            "delivered": bool((row.get("delivery") or {}).get("delivered") is True),
            "injected_bytes": _bytes((row.get("delivery") or {}).get("injected_bytes")),
            "gates": [
                {"name": gate.get("name"), "status": gate.get("status"), "reason": gate.get("reason")}
                for gate in row.get("gates") or [] if isinstance(gate, dict)
            ],
            "blocker": _first_blocker(row),
            "path": _relative(path, context["runs_root"]),
            "run_root": context["run_root"],
        })
    return records


def _run_record(run_root: Path, runs_root: Path, selection: dict[Path, list[dict[str, Any]]]) -> dict[str, Any]:
    """@brief 收集一个 run root 的只读事实；不从这里生成任何跨 run 平均。"""
    base = classify_run(run_root, runs_root)
    config = _json(run_root / "run_config.json")
    result = _json(run_root / "result.json")
    state = _json(run_root / "state.json")
    feedback, retry_rows = _feedback_rows(run_root)
    trace = [row for row in (result.get("trace") or []) if isinstance(row, dict)]
    trace_by_label = {str(row.get("label") or ""): row for row in trace}
    confirmations = _confirmation_rows(run_root)
    reports = _report_rows(run_root)
    report_status = {
        row["label"]: row["status"]
        for row in reports
        if row.get("label") and row.get("status") is not None
    }
    pairs = _pair_rows(run_root, trace_by_label)
    context = {
        "run_root": base["run_root"],
        "runs_root": runs_root,
        "target_id": base["target_id"],
        "backend": base["backend"],
        "task": base["task"],
        "dtype": base["dtype"],
        "size": base["size"],
        "cohort": base["cohort"],
        "experiment_kind": base["experiment_kind"],
        "experiment_id": base["experiment_id"],
        "experiment_fingerprint": base["experiment_fingerprint"],
        "evidence_kind": base.get("evidence_kind") or "unavailable",
        "lifecycle": base["lifecycle"],
    }
    ok_rows = [row for row in trace if row.get("status") == "ok"]
    model_ok_rows = [row for row in ok_rows if str(row.get("label") or "") != "prior"]
    first_model_ok = (
        min(
            model_ok_rows,
            key=lambda row: row.get("idx") if isinstance(row.get("idx"), int) and not isinstance(row.get("idx"), bool) else 10**9,
        )
        if model_ok_rows else None
    )
    prior_only = bool(ok_rows) and not model_ok_rows and any(str(row.get("label") or "") == "prior" for row in ok_rows)
    status_family = Counter(_status_family(row.get("status")) for row in trace)
    status_counts = Counter(str(row.get("status") or "unavailable") for row in trace)
    controls = [row for row in reports if row["has_control_field"]]
    control_runs = [row for row in controls if row["control"] is not None]
    control_contains_one = 0
    control_excludes_one = 0
    for row in control_runs:
        control = row["control"] if isinstance(row["control"], dict) else {}
        interval = control.get("interval")
        if isinstance(interval, list) and len(interval) == 2 and all(_finite(value) is not None for value in interval):
            low, high = float(interval[0]), float(interval[1])
            if low <= 1.0 <= high:
                control_contains_one += 1
            else:
                control_excludes_one += 1
    protocols = {
        str(row["protocol"] or row["dispatch"])
        for row in reports if (row["protocol"] or row["dispatch"])
    }
    outcomes = Counter(str(row.get("outcome") or "unknown") for row in feedback or [])
    resolved_sessions = []
    for row in feedback or []:
        label = row.get("_canonical_label") or str(row.get("label") or "")
        raw_label = str(row.get("label") or "")
        trace_row = trace_by_label.get(raw_label) or trace_by_label.get(label) or {}
        report_value = report_status.get(raw_label)
        if report_value is None:
            report_value = report_status.get(label)
        if report_value is not None:
            outer_status = report_value
            outer_source = "candidate_report"
        elif trace_row:
            outer_status = trace_row.get("status")
            outer_source = "trace"
        else:
            outer_status = None
            outer_source = "unavailable"
        resolved = resolve_candidate_status(row.get("candidate_status"), outer_status)
        resolved_sessions.append({
            "label": label,
            "raw_label": row.get("_raw_label") or row.get("label"),
            "provider": row.get("provider"),
            "model": row.get("model"),
            "committee": row.get("committee"),
            "outcome": str(row.get("outcome") or "unknown"),
            "feedback_status": row.get("candidate_status"),
            "outer_status": outer_status,
            "outer_source": outer_source,
            "resolved_status": resolved["status"],
            "status_source": resolved["source"],
            "conflict": resolved["conflict"],
            "t": _finite(row.get("t")),
            "seq": _int(row.get("seq")),
        })
    spent = _finite(base.get("spent_usd"))
    incomplete = base.get("cost_may_be_incomplete")
    confirmation_counts = Counter(row["field_status"] for row in confirmations)
    return {
        "base": base,
        "config": config,
        "result": result,
        "state": state,
        "context": context,
        "key": _key(base, _KEY_FIELDS_RUN, committee_signature=_committee_signature(config), evidence_kind=context["evidence_kind"]),
        "sessions": None if feedback is None else len({row["_canonical_label"] for row in feedback}),
        "retry_rows_deduplicated": retry_rows,
        "outcomes": outcomes,
        "resolved_sessions": resolved_sessions,
        "trace_present": bool(trace),
        "trace_status_counts": status_counts,
        "trace_family": status_family,
        "ok_candidates": len(ok_rows),
        "model_ok_candidates": len(model_ok_rows),
        "first_model_ok": first_model_ok,
        "first_model_ok_round": _int(first_model_ok.get("idx")) if first_model_ok else None,
        "prior_only_ok": prior_only,
        "confirmed_status": base.get("confirmed_status"),
        "search_best": _finite(result.get("best_speedup")),
        "confirmed_speedup": _finite(result.get("confirmed_speedup")),
        "spent_usd": spent,
        "cost_may_be_incomplete": incomplete,
        "confirmations": confirmations,
        "confirmation_counts": confirmation_counts,
        "fallback_confirmation": any(row["fallback"] for row in confirmations),
        "speedup_flags": sorted({
            *(["fallback"] if any(row["fallback"] for row in confirmations) else []),
            *(["mixed_measurement_boundary"] if len(protocols) > 1 else []),
            *(["missing_confirmation"] if _finite(result.get("confirmed_speedup")) is None else []),
            *(["confirmed_not_ok"] if base.get("confirmed_status") not in (None, "ok") else []),
        }),
        "reports": reports,
        "control_field_reports": len(controls),
        "control_runs": len(control_runs),
        "control_contains_one": control_contains_one,
        "control_excludes_one": control_excludes_one,
        "pairs": pairs,
        "selection": _selection_rows(run_root, selection),
        "delivery": _delivery_receipts(run_root),
        "proposals": _proposal_records(run_root, context),
        "claims": _feedback_claims(feedback, context),
        "routes": _route_records(run_root, context),
        "recovery_of": base.get("recovery_of"),
        "cost_attribution_cell": base.get("recovery_of") or base["run_root"],
    }


def _run_group_bucket(records: list[dict[str, Any]]) -> dict[str, Any]:
    """@brief 汇总一组 run：candidate 与 session 分母分开，成本只读 run 账本。"""
    counts: Counter = Counter()
    counts.update({
        "runs": 0, "finalized": 0, "runs_with_trace": 0,
        "sessions": 0, "sessions_known_runs": 0, "sessions_unavailable_runs": 0,
        "ok_candidates": 0, "model_ok_candidates": 0,
        "first_model_ok_runs": 0, "runs_without_model_ok": 0, "prior_only_ok_runs": 0,
        "confirmations": 0, "cost_known_runs": 0, "cost_incomplete_runs": 0, "cost_unknown_runs": 0,
        "recovery_runs": 0, "recovery_target_unresolved": 0, "speedup_withheld_rows": 0,
        "paired_not_applied": 0, "paired_applied": 0, "paired_decidable": 0,
        "paired_undecidable": 0, "paired_too_noisy": 0, "paired_recheck_files": 0,
        "paired_conclusion_known": 0, "paired_conclusion_unknown": 0,
        "recheck_files": 0, "recheck_decidable": 0, "recheck_undecidable": 0, "recheck_too_noisy": 0,
        "control_field_reports": 0, "control_runs": 0,
        "control_contains_one": 0, "control_excludes_one": 0,
        "cost_missing_runs": 0, "cost_attributed_elsewhere_runs": 0,
        "cost_known_model_ok": 0, "attributed_recovery_runs": 0,
    })
    for status in ("ok", *sorted(NON_OK_STATUSES), UNCHANGED, "duplicate", "old_format", "unreadable", "missing"):
        counts[f"candidate_status:{status}"] = 0
    for outcome in (EVALUATED, UNCHANGED, NO_SUBMISSION, *OTHER_FAILURE_OUTCOMES, "unknown"):
        counts[f"session_outcome:{outcome}"] = 0
    stop_reasons: Counter = Counter()
    confirmed_status: Counter = Counter()
    candidate_statuses: Counter = Counter()
    timed_by: Counter = Counter()
    identity: Counter = Counter()
    rows = []
    pair_rows = []
    recheck_rows = []
    spent_known = 0.0
    cost_known_runs = 0
    cost_incomplete = False
    cost_unknown = False
    cost_missing_runs = 0
    cost_attributed_elsewhere_runs = 0
    cost_known_model_ok = 0
    cost_units_model_ok = 0
    model_ok_total = 0
    selection_gpu = 0.0
    selection_gpu_known = 0
    selection_gpu_unavailable = 0
    reports_total = 0
    group_roots = {record["base"]["run_root"] for record in records}

    def _add_cost(spent: float | None, model_ok: int, incomplete: Any) -> None:
        """@brief 只把已归属且金额已知的成本并入该组账本。"""
        nonlocal spent_known, cost_known_runs, cost_incomplete, cost_unknown
        nonlocal cost_missing_runs, cost_known_model_ok
        if spent is None:
            cost_missing_runs += 1
        else:
            spent_known += float(spent)
            cost_known_runs += 1
            cost_known_model_ok += model_ok
        if incomplete is True:
            cost_incomplete = True
        elif incomplete is None:
            cost_unknown = True

    for record in records:
        counts["runs"] += 1
        counts[f"lifecycle:{record['base']['lifecycle']}"] += 1
        counts[f"cohort:{record['base']['cohort']}"] += 1
        if record["base"]["lifecycle"] == "finalized":
            counts["finalized"] += 1
        if record["trace_present"]:
            counts["runs_with_trace"] += 1
        if record["sessions"] is None:
            counts["sessions_unavailable_runs"] += 1
        else:
            counts["sessions_known_runs"] += 1
            counts["sessions"] += record["sessions"]
        for name, value in record["outcomes"].items():
            counts[f"session_outcome:{name}"] += value
        for name, value in record["trace_status_counts"].items():
            counts[f"candidate_status:{name}"] += value
            candidate_statuses[name] += value
        for family, value in record["trace_family"].items():
            counts[f"candidate_family:{family}"] += value
        counts["ok_candidates"] += record["ok_candidates"]
        counts["model_ok_candidates"] += record["model_ok_candidates"]
        model_ok_total += record["model_ok_candidates"]
        counts["first_model_ok_runs"] += record["first_model_ok"] is not None
        counts["runs_without_model_ok"] += record["first_model_ok"] is None and record["trace_present"]
        counts["prior_only_ok_runs"] += record["prior_only_ok"]
        if record["first_model_ok"] is not None:
            label = str(record["first_model_ok"].get("label") or "")
            counts[f"first_ok_label:{label}"] += 1
        if record["base"].get("stop_reason"):
            stop_reasons[str(record["base"]["stop_reason"])] += 1
        if record["base"].get("confirmed_status") is not None:
            confirmed_status[str(record["base"]["confirmed_status"])] += 1
        own_root = record["base"]["run_root"]
        owner_root = record.get("cost_owner_root", own_root)
        if owner_root == own_root:
            # Cost-ratio units follow the owner, including a self-owned run whose
            # amount is missing. Known-cost denominators are added by _add_cost only
            # when the amount is actually known.
            cost_units_model_ok += record["model_ok_candidates"]
            _add_cost(record["spent_usd"], record["model_ok_candidates"], record["cost_may_be_incomplete"])
        elif owner_root is None:
            # Unresolved recovery: the row keeps the amount, but no group ledger
            # or per-ok denominator may claim it.
            cost_unknown = True
        elif owner_root not in group_roots:
            # Resolved recovery whose target cell sits in another group: the
            # target group books it once; this group must not book it again.
            cost_attributed_elsewhere_runs += 1
        for incoming in record.get("incoming_recovery_costs") or []:
            # A resolved recovery contributes its ok units only to the target
            # group, even when its amount is missing. The recovery's own group
            # gets neither the units nor the known-cost denominator.
            cost_units_model_ok += incoming["model_ok_candidates"]
            _add_cost(incoming["spent_usd"], incoming["model_ok_candidates"], incoming["cost_may_be_incomplete"])
            counts["attributed_recovery_runs"] += 1
            if incoming["spent_usd"] is not None:
                counts["attributed_recovery_cost_known_runs"] += 1
                counts["attributed_recovery_cost_usd"] = round(
                    float(counts.get("attributed_recovery_cost_usd") or 0.0) + float(incoming["spent_usd"]), 12
                )
            else:
                counts["attributed_recovery_cost_missing_runs"] += 1
        if record["cost_may_be_incomplete"] is True:
            counts["cost_incomplete_runs"] += 1
        elif record["cost_may_be_incomplete"] is None:
            counts["cost_unknown_runs"] += 1
        for row in record["confirmations"]:
            identity[row["field_status"]] += 1
            if row["legacy_match_without_full_scope"]:
                identity["legacy_kind_match_without_full_scope"] += 1
        counts["confirmations"] += len(record["confirmations"])
        reports_total += len(record["reports"])
        counts["control_field_reports"] += record["control_field_reports"]
        counts["control_runs"] += record["control_runs"]
        counts["control_contains_one"] += record["control_contains_one"]
        counts["control_excludes_one"] += record["control_excludes_one"]
        run_pairs = record["pairs"]
        counts["paired_not_applied"] += run_pairs["counts"].get("not_applied", 0)
        counts["paired_applied"] += run_pairs["counts"].get("applied", 0)
        counts["paired_decidable"] += run_pairs["counts"].get("decidable", 0)
        counts["paired_undecidable"] += run_pairs["counts"].get("undecidable", 0)
        counts["paired_too_noisy"] += run_pairs["counts"].get("too_noisy", 0)
        counts["paired_conclusion_known"] += run_pairs["counts"].get("conclusion_known", 0)
        counts["paired_conclusion_unknown"] += run_pairs["counts"].get("conclusion_unknown", 0)
        counts["paired_recheck_files"] += run_pairs["counts"].get("recheck_files", 0)
        for name, value in run_pairs["counts"].items():
            if name.startswith("timed_by:"):
                timed_by[name.split(":", 1)[1]] += value
            elif name.startswith("recheck_"):
                counts[name] += value
        selection = record["selection"]
        counts["selection_sessions"] += selection["sessions"]
        selection_gpu_known += selection["gpu_seconds_known_sessions"]
        selection_gpu_unavailable += selection["gpu_seconds_unavailable_sessions"]
        if selection["gpu_seconds"] is not None:
            selection_gpu += selection["gpu_seconds"]
        if record["recovery_of"]:
            counts["recovery_runs"] += 1
            if record["spent_usd"] is not None:
                counts["recovery_cost_known_runs"] += 1
                counts["recovery_cost_usd"] = round(
                    float(counts.get("recovery_cost_usd") or 0.0) + record["spent_usd"], 12
                )
            else:
                counts["recovery_cost_missing_runs"] += 1
            if not record.get("recovery_target_found", True):
                counts["recovery_target_unresolved"] += 1
                if record["spent_usd"] is not None:
                    counts["unattributed_recovery_cost_usd"] = round(
                        float(counts.get("unattributed_recovery_cost_usd") or 0.0) + record["spent_usd"], 12
                    )
                else:
                    counts["unattributed_recovery_cost_missing_runs"] += 1
            elif owner_root not in group_roots and record["spent_usd"] is not None:
                counts["recovery_cost_attributed_elsewhere_usd"] = round(
                    float(counts.get("recovery_cost_attributed_elsewhere_usd") or 0.0) + record["spent_usd"], 12
                )
        rows.append({
            "run_root": record["base"]["run_root"],
            "cell_id": record["base"].get("cell_id"),
            "experiment_id": record["base"].get("experiment_id"),
            "experiment_fingerprint": record["base"].get("experiment_fingerprint"),
            "label": (
                record["first_model_ok"].get("label") if record["first_model_ok"] else None
            ),
            "search_best_speedup": record["search_best"],
            "confirmed_speedup": record["confirmed_speedup"],
            "confirmed_status": record["confirmed_status"],
            "flags": record["speedup_flags"],
            "included_in_average": False,
            "recovery_of": record["recovery_of"],
            "cost_attribution_cell": record["cost_attribution_cell"],
        })
        counts["speedup_withheld_rows"] += bool(record["speedup_flags"])
        for pair in run_pairs["pairs"]:
            pair_rows.append({**pair, "run_root": record["base"]["run_root"]})
        for recheck in run_pairs["rechecks"]:
            recheck_rows.append({**recheck, "run_root": record["base"]["run_root"]})
    if not counts["runs_with_trace"]:
        counts["ok_candidates"] = None
        counts["model_ok_candidates"] = None
        counts["first_model_ok_runs"] = None
        counts["runs_without_model_ok"] = None
        counts["prior_only_ok_runs"] = None
    if not counts.get("sessions_known_runs"):
        counts["sessions"] = None
    family_known = (
        candidate_statuses.get("ok", 0)
        + sum(candidate_statuses.get(name, 0) for name in NON_OK_STATUSES)
    )
    session_known = (
        sum(counts.get(f"session_outcome:{name}", 0) for name in (EVALUATED, UNCHANGED, NO_SUBMISSION))
        + sum(counts.get(f"session_outcome:{name}", 0) for name in OTHER_FAILURE_OUTCOMES)
    )
    # The denominator is every logical session the group knows about, including rows whose
    # outcome is unrecognised. Unknown outcomes lower coverage; they never shrink the unit
    # count to only the rows we could recognise.
    session_units = int(counts.get("sessions") or 0)
    confirmed_known = sum(
        confirmed_status.get(name, 0)
        for name in ("ok", "failed", "mismatch", "incorrect", "unavailable")
    )
    cost = cost_metrics(
        spent_known if cost_known_runs else None,
        cost_known_model_ok,
        incomplete=(
            None if cost_unknown else
            (True if (cost_incomplete or cost_missing_runs or cost_attributed_elsewhere_runs) else False)
        ),
    )
    identity_known = sum(identity.get(name, 0) for name in ("full_match", "partial", "mismatch"))
    ratios, statuses, coverage = _ratio_blocks({
        "ok_rate": (
            _ratio(candidate_statuses.get("ok", 0), family_known),
            family_known, family_known, "run_level_min_units",
        ),
        "first_model_ok_rate": (
            _ratio(counts["first_model_ok_runs"] or 0, counts["runs_with_trace"]),
            counts["runs_with_trace"], counts["runs_with_trace"], "run_level_min_units",
        ),
        "confirmed_ok_rate": (
            _ratio(confirmed_status.get("ok", 0), confirmed_known),
            confirmed_known, confirmed_known, "run_level_min_units",
        ),
        "no_submission_rate": (
            _ratio(counts.get(f"session_outcome:{NO_SUBMISSION}", 0), session_known),
            session_units, session_known, "session_level_min_units",
        ),
        "paired_decidable_rate": (
            _ratio(counts["paired_decidable"], counts["paired_applied"])
            if counts["paired_conclusion_known"] else None,
            counts["paired_applied"], counts["paired_conclusion_known"], "paired_conclusion_min_units",
        ),
        "identity_full_match_rate": (
            _ratio(identity.get("full_match", 0), identity_known),
            identity_known, identity_known, "paired_conclusion_min_units",
        ),
        "control_contains_one_rate": (
            _ratio(counts["control_contains_one"], counts["control_runs"]),
            counts["control_runs"], counts["control_runs"], "paired_conclusion_min_units",
        ),
        "control_field_coverage_rate": (
            # 一份报告都没有 benchmark.control 时该字段是缺失，不是实测零次对照：
            # 比例保持 null，status 由 known_units=0 判 unavailable。字段真的存在但
            # 只覆盖一部分报告时，才用 reports_total、H0 最小单位和 0.8 覆盖判 insufficient。
            # 实际 A/A 对照运行次数由 control_contains_one_rate/control_runs 另算。
            _ratio(counts["control_field_reports"], reports_total)
            if counts["control_field_reports"] else None,
            reports_total, counts["control_field_reports"], "paired_conclusion_min_units",
        ),
        "ledger_cost_per_ok": (
            cost["ledger_cost_per_ok"], cost_units_model_ok,
            cost_known_model_ok if cost["ledger_cost_per_ok"] is not None else 0, "run_level_min_units",
        ),
        "exact_cost_per_ok": (
            cost["exact_cost_per_ok"], cost_units_model_ok,
            cost_known_model_ok if cost["exact_cost_per_ok"] is not None else 0, "run_level_min_units",
        ),
    })
    notes = [SPEEDUP_NOTE]
    if cost["exact_cost_per_ok"] is None:
        notes.append(
            "exact_cost_per_ok 为 null：至少一个 run 的 cost_may_be_incomplete=true、成本缺失或归属不明，"
            "不能把已知账本除到全部 ok 上。"
        )
    if cost_missing_runs:
        notes.append("部分 run 缺 spent_usd；这些 run 的 ok 不进已知成本分母，账本只覆盖成本已知的 run。")
    if cost_attributed_elsewhere_runs:
        notes.append("恢复成本已按 recovery_of 归入目标 cell；本组账本不再重复计一次。")
    if counts["sessions_unavailable_runs"]:
        notes.append("部分 run 没有 feedback.jsonl；会话数为 null，不补 0。")
    if session_units > session_known:
        notes.append(
            "部分会话 outcome 未知；no_submission_rate 的单位是该组全部已知逻辑会话，"
            "未知 outcome 只降低覆盖，不从分母里消失。"
        )
    control_coverage = statuses["control_field_coverage_rate"]
    if reports_total and control_coverage != "measured":
        notes.append(
            CONTROL_NOTE + " 本组 control 字段覆盖为 " + control_coverage
            + "，不因任意一份报告有字段就把 A/A 覆盖当成 measured。"
        )
    if identity_known and identity.get("full_match", 0) == 0:
        notes.append(IDENTITY_NOTE)
    return {
        "counts": {**dict(counts), **{
            "reports_total": reports_total,
            "sessions_complete": counts.get("sessions_unavailable_runs", 0) == 0,
            "candidates_complete": counts.get("runs_with_trace", 0) == counts.get("runs", 0),
            "control_field_coverage": control_coverage,
            "stop_reason": dict(stop_reasons),
            "confirmed_status": dict(confirmed_status),
            "candidate_status": dict(candidate_statuses),
            "timed_by": dict(timed_by),
            "identity": dict(identity),
            "selection_gpu_seconds": selection_gpu if selection_gpu_known else None,
            "selection_gpu_seconds_complete": selection_gpu_unavailable == 0 and selection_gpu_known > 0,
            "selection_gpu_seconds_known_sessions": selection_gpu_known,
            "selection_gpu_seconds_unavailable_sessions": selection_gpu_unavailable,
            "cost_known_runs": cost_known_runs,
            "cost_known_model_ok": cost_known_model_ok,
            "cost_units_model_ok": cost_units_model_ok,
            "cost_missing_runs": cost_missing_runs,
            "cost_attributed_elsewhere_runs": cost_attributed_elsewhere_runs,
            "ledger_spent_usd": spent_known if cost_known_runs else None,
            "cost_complete": cost["cost_complete"],
            "ledger_spent_without_ok_usd": cost["ledger_spent_without_ok_usd"],
            "recovery_cost_usd": counts.get("recovery_cost_usd"),
            "unattributed_recovery_cost_usd": counts.get("unattributed_recovery_cost_usd"),
            "attributed_recovery_cost_usd": counts.get("attributed_recovery_cost_usd"),
            "recovery_cost_attributed_elsewhere_usd": counts.get("recovery_cost_attributed_elsewhere_usd"),
        }},
        "ratios": ratios,
        "metric_status": statuses,
        "coverage": coverage,
        "speedups": rows,
        "pairs": pair_rows,
        "rechecks": recheck_rows,
        "notes": notes,
    }


def _model_session_bucket(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """@brief 汇总一个 provider/model/committee 会话组；成本只读 model_totals。"""
    counts: Counter = Counter()
    outcomes: Counter = Counter()
    statuses: Counter = Counter()
    conflicts = 0
    completed_ok = 0
    failed_outcome_with_ok_status = 0
    rows = []
    for record, session in pairs:
        counts["sessions"] += 1
        outcomes[session["outcome"]] += 1
        counts[f"outcome:{session['outcome']}"] += 1
        family = _status_family(session["resolved_status"])
        statuses[family] += 1
        counts[f"candidate:{family}"] += 1
        if session["outcome"] == EVALUATED and family == "ok":
            completed_ok += 1
        elif session["outcome"] in (NO_SUBMISSION, *OTHER_FAILURE_OUTCOMES) and family == "ok":
            failed_outcome_with_ok_status += 1
        if session["conflict"]:
            conflicts += 1
        rows.append({
            "run_root": record["base"]["run_root"],
            "cell_id": record["base"].get("cell_id"),
            "session_label": session["label"],
            "session_label_raw": session["raw_label"],
            "outcome": session["outcome"],
            "feedback_status": session["feedback_status"],
            "outer_status": session["outer_status"],
            "outer_source": session["outer_source"],
            "resolved_status": session["resolved_status"],
            "status_source": session["status_source"],
            "conflict": session["conflict"],
            "provider": session["provider"],
            "model": session["model"],
            "committee": session["committee"],
        })
    unique_records = {}
    for record, _session in pairs:
        unique_records[record["base"]["run_root"]] = record
    cost_total = 0.0
    cost_total_fully_covered = 0.0
    cost_known = 0
    cost_fully_covered = 0
    totals_sessions = 0
    totals_present = False
    runs_with_totals = 0
    runs_with_known_cost = 0
    runs_without_cost = 0
    runs_with_total_without_cost = 0
    runs_with_partial_cost = 0
    cost_incomplete = False
    cost_unknown = False
    run_cost_known: dict[str, bool] = {}
    identity = (rows[0]["provider"], rows[0]["model"], rows[0]["committee"]) if rows else ("", "", "")
    for record in unique_records.values():
        matched = False
        run_cost = 0.0
        has_cost = False
        partial_coverage = False
        for total in record["result"].get("model_totals") or []:
            if not isinstance(total, dict):
                continue
            if (
                str(total.get("provider") or ""),
                str(total.get("model") or ""),
                str(total.get("committee") or ""),
            ) != (str(identity[0] or ""), str(identity[1] or ""), str(identity[2] or "")):
                continue
            totals_present = True
            matched = True
            value = _finite(total.get("cost_usd"))
            entry_sessions = _int(total.get("sessions"))
            # ``modelcost`` writes cost_usd=0.0 when it saw no billed sample at all and
            # marks that with sessions_with_cost=0 / cost_per_session_usd=null. That 0.0
            # is a missing cost, not a free session. An older row without the coverage
            # field cannot prove zero samples, so its finite cost stays known.
            samples = _int(total.get("sessions_with_cost"))
            if value is not None and (samples is None or samples > 0):
                run_cost += value
                has_cost = True
            totals_sessions += entry_sessions or 0
            if record["cost_may_be_incomplete"] is True:
                cost_incomplete = True
            elif record["cost_may_be_incomplete"] is None:
                cost_unknown = True
            # Partial coverage is its own cost gap: not every session in the run has a
            # billed sample. ``sessions_with_cost`` never exceeds ``sessions``.
            if samples is not None and entry_sessions is not None and samples < entry_sessions:
                partial_coverage = True
        if matched:
            runs_with_totals += 1
        # Completeness is per run with sessions, not only per ok session: a matching
        # entry that lacks a finite cost_usd, or covers only some sessions, is a gap.
        if partial_coverage:
            cost_incomplete = True
        if has_cost:
            cost_total += run_cost
            cost_known += 1
            # Only a fully covered run may enter the exact numerator; a partial run's
            # amount stays in the known-cost total as a fact, but it is not paired with
            # a completed-ok denominator that those sessions are not all part of.
            if not partial_coverage:
                cost_total_fully_covered += run_cost
                cost_fully_covered += 1
        if has_cost and not partial_coverage:
            runs_with_known_cost += 1
        else:
            runs_without_cost += 1
            if matched and not has_cost:
                runs_with_total_without_cost += 1
            if partial_coverage:
                runs_with_partial_cost += 1
        run_cost_known[record["base"]["run_root"]] = has_cost and not partial_coverage
    ok = statuses.get("ok", 0)
    non_ok = statuses.get("non_ok", 0)
    known = ok + non_ok
    outcome_units = counts["sessions"]
    outcome_known = sum(outcomes.get(name, 0) for name in (EVALUATED, UNCHANGED, NO_SUBMISSION, *OTHER_FAILURE_OUTCOMES))
    completed_ok_known_cost = sum(
        1 for row in rows
        if row["outcome"] == EVALUATED and _status_family(row["resolved_status"]) == "ok"
        and run_cost_known.get(row["run_root"], False)
    )
    completed_ok_without_cost = completed_ok - completed_ok_known_cost
    if not totals_present or cost_unknown:
        cost_complete = None
    elif cost_incomplete or not cost_known or runs_without_cost:
        cost_complete = False
    else:
        cost_complete = True
    # The exact numerator only counts fully covered runs and the denominator only counts
    # completed-ok sessions of those same runs: both sides come from one cost-known set.
    # A partial entry keeps the exact value null.
    exact_per_ok = (
        cost_total_fully_covered / completed_ok_known_cost
        if (cost_complete is True and completed_ok_known_cost)
        else None
    )
    ratios, metric, coverage = _ratio_blocks({
        "session_ok_rate": (_ratio(completed_ok, known), known, known, "session_level_min_units"),
        # Outcome denominators are every logical session in the group. An unrecognised
        # outcome lowers coverage instead of leaving the rate to be computed only over
        # the sessions we happened to recognise.
        "no_submission_rate": (
            _ratio(outcomes.get(NO_SUBMISSION, 0), outcome_known),
            outcome_units, outcome_known, "session_level_min_units",
        ),
        "evaluated_rate": (
            _ratio(outcomes.get(EVALUATED, 0), outcome_known),
            outcome_units, outcome_known, "session_level_min_units",
        ),
        "model_totals_cost_per_ok": (
            exact_per_ok, completed_ok,
            completed_ok_known_cost if exact_per_ok is not None else 0,
            "session_level_min_units",
        ),
    })
    notes = [RETRY_LABEL_NOTE]
    if conflicts:
        notes.append("feedback.candidate_status 与外层 report 冲突；已以外层报告为准并保留冲突计数。")
    if outcome_units > outcome_known:
        notes.append(
            "部分会话 outcome 未知；evaluated_rate/no_submission_rate 的分母是该组全部逻辑会话，"
            "未知 outcome 只降低覆盖，不从分母里消失，也不当失败。"
        )
    if not totals_present:
        notes.append("没有匹配该 provider/model/committee 的 model_totals；成本为 null，不补 0。")
    elif runs_with_partial_cost:
        notes.append(
            "部分 model_totals 行只有部分会话带账单样本（sessions_with_cost < sessions）；"
            "该 run 不算成本完整，精确每 ok 成本为 null。"
        )
    elif not cost_known or runs_without_cost or completed_ok_without_cost:
        notes.append(
            "部分有会话的 run 没有该模型的 model_totals/cost_usd；精确每 ok 成本为 null，"
            "已知账本估值只用成本已知 run 的 completed-ok 分母。"
        )
    elif cost_incomplete or cost_unknown:
        notes.append("model_totals 成本带 incomplete 标记或完整性未知；不输出精确每 ok 成本。")
    return {
        "counts": {
            **dict(counts),
            "sessions": counts["sessions"],
            "outcomes": dict(outcomes),
            "candidate_families": dict(statuses),
            "conflicts": conflicts,
            "completed_ok_sessions": completed_ok,
            "completed_ok_with_known_cost": completed_ok_known_cost,
            "completed_ok_without_cost": completed_ok_without_cost,
            "failed_outcome_with_ok_status": failed_outcome_with_ok_status,
            "retry_rows_deduplicated": sum(record["retry_rows_deduplicated"] for record in unique_records.values()),
            "model_totals_cost_usd": cost_total if cost_known else None,
            "model_totals_cost_usd_fully_covered": (
                cost_total_fully_covered if cost_fully_covered else None
            ),
            "model_totals_sessions": totals_sessions if totals_present else None,
            "model_totals_runs": runs_with_totals,
            "runs_with_model_total_cost": runs_with_known_cost,
            "runs_without_model_totals": runs_without_cost,
            "runs_with_model_total_without_cost": runs_with_total_without_cost,
            "runs_with_partial_model_cost": runs_with_partial_cost,
            "runs_with_sessions": len(unique_records),
            "cost_complete": cost_complete,
        },
        "ratios": ratios,
        "metric_status": metric,
        "coverage": coverage,
        "sessions": rows,
        "notes": notes,
    }


def _delivery_bucket_document(bucket: dict[str, Any]) -> dict[str, Any]:
    counts = bucket["counts"]
    dynamic_complete = counts["receipts_with_missing_dynamic_bytes"] == 0
    static_complete = counts["receipts_with_missing_static_bytes"] == 0
    expected = counts["dynamic_delivered_bytes"] + counts["dynamic_omitted_bytes"] if dynamic_complete else 0
    other_expected = bucket["blocks"]["expected_bytes"]
    ratios, metric, coverage = _ratio_blocks({
        "dynamic_omission_rate": (
            _ratio(counts["dynamic_omitted_bytes"], expected) if dynamic_complete else None,
            counts["receipts"], counts["receipts"] if dynamic_complete and expected else 0,
            "delivery_receipt_min_units",
        ),
        "other_block_omission_rate": (
            _ratio(bucket["blocks"]["omitted_bytes"], other_expected),
            counts["receipts"], counts["receipts"] if other_expected else 0,
            "delivery_receipt_min_units",
        ),
    })
    notes = [STATIC_CONTRACT_NOTE, DELIVERY_NOTE]
    if counts["static_contract_violations"]:
        notes.append("静态合同被省略：这是合同失败，不是可接受的比例。")
    if counts["block_accounting_mismatches"]:
        notes.append("部分块的 expected != delivered + omitted，字节账不闭合。")
    if not dynamic_complete or not static_complete:
        notes.append("部分回执缺少字节字段；该块保持 null/unavailable，不把缺失字段补成 0。")
    return {
        "counts": {
            **dict(counts),
            "dynamic_expected_bytes": expected if dynamic_complete else None,
            "static_delivered_bytes": counts["static_delivered_bytes"] if static_complete else None,
            "static_omitted_bytes": counts["static_omitted_bytes"] if static_complete else None,
            "dynamic_delivered_bytes": counts["dynamic_delivered_bytes"] if dynamic_complete else None,
            "dynamic_omitted_bytes": counts["dynamic_omitted_bytes"] if dynamic_complete else None,
            "static_contract_ok": None if not static_complete else counts["static_contract_violations"] == 0,
            "static_bytes_complete": static_complete,
            "dynamic_bytes_complete": dynamic_complete,
            "situation_layers": {name: dict(bucket["layers"][name]) for name in dynamic_budget_audit.LAYER_NAMES},
            "other_blocks": dict(bucket["blocks"]),
            "other_blocks_by_name": {name: dict(value) for name, value in sorted(bucket["blocks_by_name"].items())},
            "omission_reasons_bytes": dict(bucket["reasons"]),
        },
        "ratios": ratios,
        "metric_status": metric,
        "coverage": coverage,
        "notes": notes,
    }


def _claim_bucket(records: list[dict[str, Any]]) -> dict[str, Any]:
    """@brief claim 证据：proposal 与 feedback premise 分开计，不共享分母。"""
    counts: Counter = Counter()
    proposal_rows = [record for record in records if record.get("kind") == "proposal"]
    for record in proposal_rows:
        counts["proposal_claims"] += 1
        if record.get("evidence_level"):
            counts["explicit_evidence_levels"] += 1
        counts["simulated_claims"] += int(record.get("simulated") is True)
        if not record.get("structured_key"):
            counts["claims_without_structural_key"] += 1
    summary = summarize_levels([
        {"evidence_level": record.get("evidence_level")} for record in proposal_rows
    ])
    levels = Counter(summary["counts"])
    for record in records:
        if record.get("kind") != "feedback_claim":
            continue
        counts["feedback_claims"] += 1
        premise = record.get("premise")
        if premise in PREMISE_KINDS:
            counts[f"premise:{premise}"] += 1
            if premise == "downgraded" and record.get("has_text"):
                counts["downgraded_with_record"] += 1
        else:
            counts["premise_blank"] += 1
    counts["premise_declared"] = sum(counts.get(f"premise:{name}", 0) for name in PREMISE_KINDS)
    counts["downgraded_with_text"] = counts.get("downgraded_with_record", 0)
    premise_total = counts["premise_declared"]
    explicit_levels = counts.get("explicit_evidence_levels", 0)
    ratios, metric, coverage = _ratio_blocks({
        "premise_verified_rate": (
            _ratio(counts.get("premise:verified", 0), premise_total),
            premise_total, premise_total, "claim_level_min_units",
        ),
        "premise_downgraded_rate": (
            _ratio(counts.get("premise:downgraded", 0), premise_total),
            premise_total, premise_total, "claim_level_min_units",
        ),
        "host_process_share_explicit": (
            _ratio(levels.get("host_process", 0), explicit_levels),
            explicit_levels, explicit_levels, "claim_level_min_units",
        ),
    })
    notes = ["evidence_level 只挂在单条提案/主张上，不挂在整场 run 上。"]
    if counts.get("premise_blank", 0):
        notes.append("空白 premise 保持 null，不写成 0% 已验证。")
    if not premise_total:
        notes.append("没有显式 premise 声明；前提比例为 null 而不是 0。")
    if counts.get("claims_without_structural_key", 0):
        notes.append("存在没有 proposal_key/claim_digest 的自由文本主张；跨 run 重复状态为 unavailable。")
    return {
        "counts": {**dict(counts), "evidence_levels": dict(levels)},
        "ratios": ratios,
        "metric_status": metric,
        "coverage": coverage,
        "notes": notes,
    }


def _route_bucket(records: list[dict[str, Any]]) -> dict[str, Any]:
    """@brief route-M trace 是独立单位；与送达回执、claim 证据不共享分母。"""
    counts: Counter = Counter()
    for record in records:
        counts["traces"] += 1
        counts["shadow_eligible"] += int(record.get("shadow_written") is True)
        counts["delivered"] += int(record.get("delivered") is True)
        counts["injected_bytes"] += int(record.get("injected_bytes") or 0)
        counts[f"conclusion:{record.get('conclusion') or 'unavailable'}"] += 1
        counts[f"evidence_level:{record.get('evidence_level') or 'unavailable'}"] += 1
        counts["simulated_traces"] += int(record.get("simulated") is True)
        counts[f"blocker:{record.get('blocker')}"] += 1
        for gate in record.get("gates") or []:
            counts[f"gate:{gate.get('name')}:{gate.get('status')}"] += 1
    ratios, metric, coverage = _ratio_blocks({
        "shadow_eligible_rate": (
            _ratio(counts.get("shadow_eligible", 0), counts["traces"]),
            counts["traces"], counts["traces"], "claim_level_min_units",
        ),
        "delivered_rate": (
            _ratio(counts.get("delivered", 0), counts["traces"]),
            counts["traces"], counts["traces"], "claim_level_min_units",
        ),
    })
    return {
        "counts": dict(counts),
        "ratios": ratios,
        "metric_status": metric,
        "coverage": coverage,
        "notes": ["路线 M trace 是独立单位；与送达回执、claim 证据不共享分母。"],
    }


def _experience_documents(targets_root: Path) -> dict[str, Any]:
    """@brief 复用 audit-experience 的指针审计，按 target/task 汇总且不晋升。"""
    directory = Path(targets_root)
    files = []
    if directory.is_dir():
        for path in sorted(directory.glob("*/experience.md")):
            if not path.is_file():
                continue
            try:
                files.append(audit_experience_file(path, repo=directory.parent))
            except (OSError, ValueError):
                continue
    groups: dict[tuple[str, str], Counter] = defaultdict(Counter)
    warnings: Counter = Counter()
    for document in files:
        target = str(document.get("target_id") or "unknown")
        for section in document.get("sections") or []:
            task_scope = (section.get("scope") or {}).get("task") or {}
            task = str(task_scope.get("value") or "unknown")
            bucket = groups[(target, task)]
            bucket["sections"] += 1
            bucket[f"confidence:{section.get('confidence')}"] += 1
            bucket["verified" if section.get("confidence") == "verified" else "not_verified"] += 1
            for pointer in section.get("pointers") or []:
                bucket["pointers"] += 1
                bucket["pointer_valid"] += int(bool(pointer.get("valid")))
                bucket["pointer_existing"] += int(bool(pointer.get("valid") and pointer.get("exists")))
                bucket["pointer_missing"] += int(bool(pointer.get("valid") and not pointer.get("exists")))
                bucket["pointer_invalid"] += int(not pointer.get("valid"))
            bucket["legacy_sections"] += int(bool(section.get("legacy")))
            bucket["simulated_sections"] += int(bool((section.get("simulation") or {}).get("declared")))
            bucket["promotion_review_eligible"] += int(bool((section.get("promotion_review") or {}).get("eligible")))
            for warning in section.get("warnings") or []:
                warnings[str(warning.get("code") or "unknown")] += 1
    rows = []
    for (target, task), bucket in sorted(groups.items()):
        sections = bucket["sections"]
        pointer_total = bucket["pointers"]
        evidence = "simulated" if sections and bucket["simulated_sections"] == sections else "unavailable"
        rows.append({
            "key": {"target_id": target, "task": task},
            "evidence_kind": evidence,
            "counts": dict(bucket),
            "ratios": {
                "pointer_valid_rate": _ratio(bucket["pointer_valid"], pointer_total),
                "pointer_existing_rate": _ratio(bucket["pointer_existing"], pointer_total),
                "verified_rate": _ratio(bucket["verified"], sections),
            },
            # The H0 claim-level minimum units and the 0.8 coverage rule apply here too:
            # one or two pointer/section rows are not sufficient evidence for a rate, and
            # zero units stays unavailable. Every pointer is judged valid or invalid, so
            # known_units equals units; the binding constraint is the minimum unit count.
            "metric_status": {
                "pointer_valid_rate": metric_status(
                    "claim_level_min_units", units=pointer_total, known_units=pointer_total,
                ),
                "pointer_existing_rate": metric_status(
                    "claim_level_min_units", units=pointer_total, known_units=pointer_total,
                ),
                "verified_rate": metric_status(
                    "claim_level_min_units", units=sections, known_units=sections,
                ),
            },
            "coverage": {
                "pointer_valid_rate": {"units": pointer_total, "known_units": pointer_total},
                "pointer_existing_rate": {"units": pointer_total, "known_units": pointer_total},
                "verified_rate": {"units": sections, "known_units": sections},
            },
            "notes": [
                "经验作用域审计只证明指针与作用域，不证明经验有效；本表不晋升。",
                "未声明模拟也不等于 real measurement；经验审计不是测量。",
                "经验比例沿用 H0 的 claim 级最小独立单位 5 与 0.8 覆盖规则；少量指针是 insufficient，零单位是 unavailable。",
            ],
        })
    summary = Counter()
    for document in files:
        values = document.get("summary") or {}
        summary["files"] += 1
        summary["sections"] += int(values.get("sections") or 0)
        for key in ("verified", "reported", "other"):
            summary[f"confidence:{key}"] += int((values.get("confidence") or {}).get(key) or 0)
        for key in ("total", "valid", "invalid", "existing", "missing"):
            summary[f"pointer:{key}"] += int((values.get("pointers") or {}).get(key) or 0)
        summary["simulated_sections"] += int(values.get("simulated_sections") or 0)
        summary["legacy_sections"] += int(values.get("legacy_sections") or 0)
    return {
        "files": [str(document.get("path")) for document in files],
        "groups": rows,
        "summary": dict(summary),
        "warnings_by_code": dict(warnings),
        "status": "measured" if files else "unavailable",
    }


def _target_backend(targets_root: Path) -> dict[str, str]:
    """@brief 只读 target.toml 的 backend 字段，供授权按 target/backend 卷。"""
    import tomllib

    out: dict[str, str] = {}
    directory = Path(targets_root)
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*/target.toml")):
        try:
            document = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        target_id = str(document.get("id") or path.parent.name)
        out[target_id] = str(document.get("backend") or "unknown")
    return out


def _trust_documents(
    targets_root: Path,
    *,
    prompt_sha1: str | None,
    protocol_version: str | None,
    now: float,
) -> dict[str, Any]:
    """@brief 逐 grant 复核 prompt/protocol/status/expires_at；不合成 target 授权布尔值。"""
    from . import trust

    directory = Path(targets_root)
    backends = _target_backend(directory)
    current_protocol = str(protocol_version or trust.current_protocol_version())
    grants = []
    states: Counter = Counter()
    rollup: dict[tuple[str, str], Counter] = defaultdict(Counter)
    errors = []
    reserved = []
    if directory.is_dir():
        for path in sorted(directory.glob("*/trust.toml")):
            target_id = path.parent.name
            if target_id.startswith("_"):
                reserved.append(target_id)
                continue
            try:
                rows = trust.load_grants(target_id, root=directory)
            except trust.TrustError as exc:
                errors.append({"target_id": target_id, "error": str(exc)})
                continue
            current_prompt = prompt_sha1
            if current_prompt is None:
                try:
                    current_prompt = trust.current_prompt_sha1(target_id)
                except (OSError, ValueError):
                    current_prompt = None
            for row in rows:
                status = str(row.get("status") or "unknown")
                expiry = _finite(row.get("expires_at"))
                prompt_ok = None if current_prompt is None else str(row.get("prompt_sha1") or "") == current_prompt
                protocol_ok = str(row.get("protocol_version") or "") == current_protocol
                if status == "revoked":
                    state = "revoked"
                elif status == "suspended":
                    state = "suspended"
                elif expiry is not None and expiry > 0 and expiry <= now:
                    state = "expired"
                elif status != "active":
                    state = "unknown"
                elif prompt_ok is False and not protocol_ok:
                    state = "stale_prompt_protocol"
                elif prompt_ok is False:
                    state = "stale_prompt"
                elif not protocol_ok:
                    state = "stale_protocol"
                elif prompt_ok is None:
                    state = "unknown"
                else:
                    state = "active"
                states[state] += 1
                backend = backends.get(target_id, "unknown")
                rollup[(target_id, backend)][state] += 1
                rollup[(target_id, backend)]["grants"] += 1
                grants.append({
                    "target_id": target_id,
                    "backend": backend,
                    "grant_id": row.get("id"),
                    "task": row.get("task"),
                    "provider": row.get("provider"),
                    "model": row.get("model"),
                    "effort": row.get("effort"),
                    "scopes": list(row.get("scopes") or []),
                    "status": status,
                    "state": state,
                    "prompt_sha1": row.get("prompt_sha1"),
                    "protocol_version": row.get("protocol_version"),
                    "prompt_matches_current": prompt_ok,
                    "protocol_matches_current": protocol_ok,
                    "expires_at": expiry,
                    "granted_by": row.get("granted_by"),
                    "granted_at": _finite(row.get("granted_at")),
                })
    rollup_rows = [
        {
            "key": {"target_id": target, "backend": backend},
            "counts": dict(bucket),
            "states": {name: int(bucket.get(name, 0)) for name in ("active", "stale_prompt", "stale_protocol", "stale_prompt_protocol", "expired", "suspended", "revoked", "unknown")},
            "notes": ["逐 grant 状态；不生成 target 级 authorized=true/false。"],
        }
        for (target, backend), bucket in sorted(rollup.items())
    ]
    return {
        "current_protocol_version": current_protocol,
        "grants": grants,
        "states": {name: int(states.get(name, 0)) for name in ("active", "stale_prompt", "stale_protocol", "stale_prompt_protocol", "expired", "suspended", "revoked", "unknown")},
        "rollup": rollup_rows,
        "errors": errors,
        "reserved_targets_skipped": sorted(reserved),
        "notes": [TRUST_NOTE],
        "status": "measured" if grants else "unavailable",
    }


def _cross_audit(runs_root: Path) -> dict[str, Any]:
    """@brief 复用现有只读审计计算重叠摘要；失败只标记该项 unavailable。"""
    out: dict[str, Any] = {}
    try:
        from . import n1_metrics_audit

        document = n1_metrics_audit.audit(runs_root)
        out["n1"] = {
            "status": "measured",
            "natural_paired_files": document["natural_paired_comparisons"]["counts"].get("files", 0),
            "natural_paired_applied": document["natural_paired_comparisons"]["counts"].get("applied", 0),
            "recheck_files": document["rechecks"]["files"],
            "n0_runs": document["n0"]["summary"]["runs"],
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        out["n1"] = {"status": "unavailable", "error": type(exc).__name__}
    try:
        document = dynamic_budget_audit.audit(runs_root)
        out["dynamic_budget"] = {
            "status": "measured",
            "receipts": document["receipts"],
            "runs": document["runs"],
            "static_omitted_bytes": document["counts"].get("static_omitted", 0),
            "dynamic_omitted_bytes": document["counts"].get("dynamic_receipt_omitted", 0),
            "dynamic_delivered_bytes": document["counts"].get("dynamic_receipt_delivered", 0),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        out["dynamic_budget"] = {"status": "unavailable", "error": type(exc).__name__}
    return out


def _backend_boundary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """@brief Metal 专属测量与通用 harness 分栏；没有 NPU 自然 run 就保持 unavailable。"""
    evidence: Counter = Counter()
    real = set()
    simulated = set()
    for record in records:
        backend = record["base"].get("backend") or "unknown"
        kind = record["base"].get("evidence_kind") or "unavailable"
        evidence[f"{backend}:{kind}"] += 1
        if kind == "real" and backend not in ("unknown", "unavailable"):
            real.add(backend)
        elif kind == "simulated" and backend not in ("unknown", "unavailable"):
            simulated.add(backend)
    npu_real = "npu" in real
    npu_simulated = "npu" in simulated
    npu = {
        "device_metrics": "unavailable",
        "natural_runs": sum(1 for record in records if record["base"].get("backend") == "npu" and record["base"].get("evidence_kind") == "real"),
        "simulated_runs": sum(1 for record in records if record["base"].get("backend") == "npu" and record["base"].get("evidence_kind") == "simulated"),
        "publication_allowed": False,
        "reason": NPU_NOTE,
    }
    if npu_simulated and not npu_real:
        npu["reason"] = "只有模拟声明；模拟不是设备正确性或性能证据。"
    return {
        "by_backend_evidence": dict(sorted(evidence.items())),
        "real_backends": sorted(real),
        "simulated_backends": sorted(simulated),
        "real_backend_count": len(real),
        "cross_backend_status": "measured" if len(real) >= SUFFICIENCY_THRESHOLDS["cross_backend_min_real_backends"] else ("insufficient" if real else "unavailable"),
        "npu": npu,
        "feature_scope": {
            "metal_only": ["gpu_timestamp", "aa_control", "fallback_provider", "device_name"],
            "generic": ["delivery_bytes", "trust_grants", "experience_scope", "run_lifecycle"],
        },
        "notes": [
            "Metal 专属功能只在真实 Metal 证据上讨论；通用 harness 送达/授权/经验审计不是设备性能结论。",
            NPU_NOTE,
        ],
    }


def _effect_assessment(run_groups: list[dict[str, Any]], backend: dict[str, Any]) -> dict[str, Any]:
    """@brief 只判断“是否足够回答跨任务/跨 backend 收益”，不产出收益结论。"""
    measured_tasks = {
        str(group["key"].get("task"))
        for group in run_groups
        if group["key"].get("cohort") == "natural"
        and group.get("metric_status", {}).get("ok_rate") == "measured"
        and group["key"].get("task")
    }
    tasks_ok = len(measured_tasks) >= SUFFICIENCY_THRESHOLDS["cross_task_min_tasks"]
    backends_ok = backend["real_backend_count"] >= SUFFICIENCY_THRESHOLDS["cross_backend_min_real_backends"]
    if not measured_tasks and not backend["real_backends"]:
        status = "unavailable"
    elif tasks_ok and backends_ok:
        status = "measured"
    else:
        status = "insufficient"
    return {
        "status": status,
        "effect_claim_allowed": False,
        "scope": "natural_runs_only",
        "measured_tasks": sorted(measured_tasks),
        "measured_task_count": len(measured_tasks),
        "real_backend_count": backend["real_backend_count"],
        "thresholds": {
            "cross_task_min_tasks": SUFFICIENCY_THRESHOLDS["cross_task_min_tasks"],
            "cross_backend_min_real_backends": SUFFICIENCY_THRESHOLDS["cross_backend_min_real_backends"],
        },
        "reason": (
            "跨任务/跨真实 backend 的覆盖不足；自然 run 的前后差异只能写关联，不能写因果。"
            if status != "measured" else
            "覆盖达到冻结阈值，但本计分卡仍不把自然 run 的前后差异写成因果。"
        ),
    }


def scorecard(
    runs_root: Path,
    *,
    targets_root: Path | None = None,
    prompt_sha1: str | None = None,
    protocol_version: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """@brief 生成 H1 四张只读计分表与 H2 边界，不写派生库。

    @param runs_root 包含 run root 的目录。
    @param targets_root 现有 ``targets/`` 目录；空则没有经验与授权数据。
    @param prompt_sha1 测试或复算用的提示指纹覆盖；空则逐 target 取当前值。
    @param protocol_version 测试或复算用的协议版本覆盖；空则取当前值。
    @param now 授权过期判断使用的时间戳；空则取当前时间。
    @return ``shapeforge.harness-scorecard.1`` 文档。
    """
    runs_root = Path(runs_root)
    targets_root = Path(targets_root) if targets_root is not None else None
    generated = float(now if now is not None else time.time())
    roots = discover_run_roots(runs_root)
    selections, duplicate_selections = _selection_documents(runs_root)
    selection_map: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for run, _path, document in selections:
        selection_map[Path(run).resolve()].append(document)
    records = [_run_record(root, runs_root, selection_map) for root in roots]
    run_root_names = sorted({record["base"]["run_root"] for record in records})
    by_root = {record["base"]["run_root"]: record for record in records}
    # Cost attribution is decided once, before grouping: a resolved recovery
    # books its ledger and known-cost ok denominator on the target cell only.
    for record in records:
        own_root = record["base"]["run_root"]
        record["cost_owner_root"] = own_root
        record["recovery_target_root"] = None
        record["recovery_target_found"] = not record["recovery_of"]
        record["incoming_recovery_costs"] = []
    for record in records:
        if not record["recovery_of"]:
            continue
        own_root = record["base"]["run_root"]
        target = str(record["recovery_of"]).rstrip("/")
        matches = [
            name for name in run_root_names
            if name == target or name.endswith("/" + target)
        ]
        if len(matches) != 1:
            # Missing or ambiguous target: the row keeps the amount as
            # unattributed; no group ledger and no per-ok denominator sees it.
            record["cost_owner_root"] = None
            continue
        target_root = matches[0]
        record["recovery_target_root"] = target_root
        record["recovery_target_found"] = True
        if target_root == own_root:
            continue
        record["cost_owner_root"] = target_root
        by_root[target_root]["incoming_recovery_costs"].append({
            "run_root": own_root,
            "cell_id": record["base"].get("cell_id"),
            "spent_usd": record["spent_usd"],
            "model_ok_candidates": record["model_ok_candidates"],
            "cost_may_be_incomplete": record["cost_may_be_incomplete"],
        })

    run_groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    session_groups: dict[tuple, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    delivery_groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    claim_groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    route_groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    render_only = 0
    for record in records:
        context = record["context"]
        run_groups[record["key"]].append(record)
        delivery, render = record["delivery"]
        render_only += render
        if delivery:
            delivery_groups[_key(context, _KEY_FIELDS_RECORD)].extend(delivery)
        if record["proposals"] or record["claims"]:
            claim_groups[_key(context, _KEY_FIELDS_RECORD)].extend(record["proposals"])
            claim_groups[_key(context, _KEY_FIELDS_RECORD)].extend(record["claims"])
        if record["routes"]:
            route_groups[_key(context, _KEY_FIELDS_RECORD)].extend(record["routes"])
        for session in record["resolved_sessions"]:
            key = _key(
                context, _KEY_FIELDS_SESSION,
                provider=session["provider"], model=session["model"], committee=session["committee"],
            )
            session_groups[key].append((record, session))

    run_table = []
    for item in _group_items(run_groups, _KEY_FIELDS_RUN):
        key, group = item["key"], item["bucket"]
        run_table.append({
            "key": key,
            **_run_group_bucket(group),
            "source_run_roots": sorted({record["base"]["run_root"] for record in group}),
        })
    session_table = []
    for item in _group_items(session_groups, _KEY_FIELDS_SESSION):
        key, group = item["key"], item["bucket"]
        session_table.append({
            "key": key,
            **_model_session_bucket(group),
            "source_run_roots": sorted({record["base"]["run_root"] for record, _session in group}),
        })
    delivery_table = []
    records_by_root = {record["base"]["run_root"]: record for record in records}
    for item in _group_items(delivery_groups, _KEY_FIELDS_RECORD):
        key, group = item["key"], item["bucket"]
        retention = {
            "status": "not_applicable" if key.get("cohort") == "natural" else "unavailable",
            "reason": "自然 run 没有预登记的指定保留项；受控记录若未在 runs/ 留保留清单则不复述。",
        }
        if key.get("cohort") != "natural":
            for receipt in group:
                run_root = _relative(receipt["path"].parent.parent.parent, runs_root)
                record = records_by_root.get(run_root)
                plan_path = (record or {}).get("base", {}).get("plan_path")
                if not plan_path:
                    continue
                plan = _json(runs_root / plan_path)
                archived = plan.get("specified_retention") or plan.get("retention")
                if archived is not None:
                    retention = {
                        "status": "archived",
                        "value": archived,
                        "source": plan_path,
                        "reason": "只读复述受控记录里已有的保留清单，不重新判定。",
                    }
                    break
        delivery_table.append({
            "key": key,
            **_delivery_bucket_document(_delivery_bucket(group)),
            "specified_retention": retention,
            "source_paths": sorted({_relative(item["path"], runs_root) for item in group}),
        })
    claim_table = []
    for item in _group_items(claim_groups, _KEY_FIELDS_RECORD):
        key, group = item["key"], item["bucket"]
        claim_table.append({
            "key": key,
            **_claim_bucket(group),
            "source_paths": sorted({item["path"] for item in group if item.get("path")}),
        })
    route_table = []
    for item in _group_items(route_groups, _KEY_FIELDS_RECORD):
        key, group = item["key"], item["bucket"]
        route_table.append({
            "key": key,
            **_route_bucket(group),
            "source_paths": sorted({item["path"] for item in group if item.get("path")}),
        })

    # Cross-run repetition of structured proposals only: exact key or exact digest.
    # Cohort and experiment groups stay apart: the same key never counts across
    # natural/controlled/unknown or across different experiment plans.
    repetition_buckets: dict[tuple[str, str | None], dict[str, Any]] = {}
    claim_without_key = 0
    for record in records:
        cohort = str(record["base"]["cohort"])
        experiment_group = _experiment_group(record["base"])
        bucket = repetition_buckets.setdefault(
            (cohort, experiment_group), {"keys": defaultdict(set), "free": 0}
        )
        for proposal in record["proposals"]:
            if not proposal["structured_key"]:
                bucket["free"] += 1
                claim_without_key += 1
                continue
            identity = str(proposal["proposal_key"] or proposal["claim_digest"])
            bucket["keys"][identity].add(record["base"]["run_root"])
    repetition_rows = []
    for (cohort, experiment_group), bucket in sorted(
        repetition_buckets.items(), key=lambda item: (item[0][0], _sort_token(item[0][1]))
    ):
        keys = bucket["keys"]
        repeated = {key: runs for key, runs in keys.items() if len(runs) > 1}
        free = bucket["free"]
        repetition_rows.append({
            "cohort": cohort,
            "experiment_group": experiment_group,
            "structured_claims": sum(len(runs) for runs in keys.values()),
            "distinct_structured_keys": len(keys),
            "repeated_keys": len(repeated),
            "repeated_occurrences": sum(len(runs) for runs in repeated.values()),
            "claims_without_structural_key": free,
            "status": "unavailable" if free and not keys else ("measured" if keys else "unavailable"),
        })
    repetition = {
        "by_cohort_and_experiment": repetition_rows,
        "claims_without_structural_key": claim_without_key,
        "note": "只按精确 proposal_key/claim_digest 计同一 cohort/实验分组内的跨 run 重复；自由文本不临时聚类，跨 cohort 不互计。",
    }

    experience = _experience_documents(targets_root) if targets_root is not None else {
        "files": [], "groups": [], "summary": {}, "warnings_by_code": {}, "status": "unavailable",
    }
    trust_documents = _trust_documents(
        targets_root, prompt_sha1=prompt_sha1, protocol_version=protocol_version, now=generated,
    ) if targets_root is not None else {
        "current_protocol_version": None, "grants": [], "states": {}, "rollup": [], "errors": [],
        "notes": [TRUST_NOTE], "status": "unavailable",
    }
    backend = _backend_boundary(records)
    effect = _effect_assessment(run_table, backend)
    inventory_counts = Counter(record["base"]["lifecycle"] for record in records)
    cohort_counts = Counter(record["base"]["cohort"] for record in records)
    evidence_counts = Counter(str(record["base"]["evidence_kind"] or "unavailable") for record in records)
    experiment_counts = Counter(record["base"]["experiment_kind"] for record in records)
    containers = []
    container_paths = set()
    root_set = {Path(path).resolve() for path in roots}
    for record in records:
        for name in ("plan_path", "approval_path"):
            value = record["base"].get(name)
            if value:
                container_paths.add((runs_root / value).parent.resolve())
    for path in sorted(container_paths):
        if path in root_set:
            continue
        children = [
            record["base"]["run_root"] for record in records
            if (runs_root / record["base"]["run_root"]).resolve().is_relative_to(path)
        ]
        containers.append({"path": _relative(path, runs_root), "child_runs": len(children), "counted_as_run": False})

    document = {
        "schema": SCHEMA,
        "runs_root": str(runs_root),
        "targets_root": str(targets_root) if targets_root is not None else None,
        "generated_at_unix": generated,
        "read_only": True,
        "model_calls": 0,
        "gpu_calls": 0,
        "publication_allowed": False,
        # ``h3_started`` is kept for readers of the frozen 2026-11-21 snapshot bytes. It
        # describes this read-only audit only: no H3 measurement window or paid effect
        # experiment is launched from here. The separately authorized H3 measurement-window
        # smoke of 2026-11-22 lives in docs/可信harness/h3-measurement-window-*-2026-11-22.json;
        # this audit neither rewrites nor extends it.
        "h3_started": False,
        "h3_started_by_this_audit": False,
        "inventory": {
            "run_count": len(records),
            "counts": {
                "lifecycle": {name: int(inventory_counts.get(name, 0)) for name in ("finalized", "incomplete_or_aborted", "dry_run", "unclassified")},
                "cohort": {name: int(cohort_counts.get(name, 0)) for name in ("natural", "controlled", "recovery", "unknown")},
                "evidence_kind": {name: int(evidence_counts.get(name, 0)) for name in ("real", "simulated", "unavailable")},
                "experiment_kind": {name: int(experiment_counts.get(name, 0)) for name in ("paid", "smoke", "paid_smoke", "unknown")},
            },
            "experiment_containers": containers,
            "selection_duplicate_copies_ignored": duplicate_selections,
        },
        "tables": {
            "run_efficiency": {
                "unit": "run",
                "group_count": len(run_table),
                "groups": run_table,
                "notes": [
                    "键包含 committee_signature，不把整场 run 算到 result.json['model'] 一个模型上。",
                    "experiment_id 为空时用 experiment_fingerprint 兜底分组；原始行保留 cell_id/recovery_of。",
                    "finalized 只表示 result 已落盘，不表示搜索成功；stop_reason 与确认状态独立。",
                    "cost 只取 result.json.spent_usd；与 model_totals 分开。",
                    SPEEDUP_NOTE,
                ],
            },
            "model_session_efficiency": {
                "unit": "session",
                "group_count": len(session_table),
                "groups": session_table,
                "notes": [
                    "会话结局来自 feedback.jsonl；sessions/ 只提供送达回执。",
                    RETRY_LABEL_NOTE,
                    "feedback 与外层 report 冲突时以外层为准并保留冲突计数。",
                    "runs 计数留在 run_efficiency；一场双委员会 run 可在这里有两行模型会话。",
                ],
            },
            "delivery_and_feedback": {
                "shared_denominator": False,
                "record_types": {
                    "delivery_receipt": {
                        "unit": "session",
                        "coverage": "available" if delivery_table else "unavailable",
                        "render_only_receipts": render_only,
                        "group_count": len(delivery_table),
                        "groups": delivery_table,
                        "notes": [
                            DELIVERY_NOTE,
                            STATIC_CONTRACT_NOTE,
                            "自然 run 的指定保留项是 not_applicable；字节送达不代替已理解。",
                        ],
                    },
                    "claim_evidence": {
                        "unit": "proposal_or_feedback_claim",
                        "coverage": "available" if claim_table else "unavailable",
                        "group_count": len(claim_table),
                        "repetition": repetition,
                        "groups": claim_table,
                        "notes": [
                            "evidence_level 只出现在主张表。",
                            "跨 run 重复只按精确 key/digest；自由文本没有该键时为 unavailable。",
                        ],
                    },
                    "route_m_trace": {
                        "unit": "trace",
                        "coverage": "available" if route_table else "unavailable",
                        "group_count": len(route_table),
                        "groups": route_table,
                        "notes": [
                            "真实门迹不存在时是 unavailable，不是成功率 0%。",
                            "离线重放仍由 audit-route-m 负责；本表只读真实 trace。",
                        ],
                    },
                },
            },
            "experience_and_governance": {
                "units": ["experience_section", "trust_grant"],
                "experience": experience,
                "governance": trust_documents,
                "notes": [
                    "经验指针按 target/task 汇总；不晋升、不跨 target 传播。",
                    TRUST_NOTE,
                ],
            },
        },
        "backend_boundary": backend,
        "effect_assessment": effect,
        "cross_audit": _cross_audit(runs_root),
        "sufficiency_contract": dict(SUFFICIENCY_THRESHOLDS),
        "notes": [
            "四张表不压成一张笛卡尔积；delivery_receipt/claim_evidence/route_m_trace 不共享分母。",
            "evidence_kind=real|simulated|unavailable 与 metric_status=measured|insufficient|unavailable 是两个独立维度。",
            "缺失值保持 null；speedup、延迟和每 ok 成本不跨不兼容分组求平均。",
            NPU_NOTE,
            "本文件是只读计分卡与自然 run 基线，不是因果效果、发布加速比或 H3 授权。",
            "h3_started=false 只描述本只读审计不启动 H3；2026-11-22 的 H3 测量窗口 smoke 是另行授权的独立实验。",
        ],
    }
    return document


def _status_brief(group: dict[str, Any]) -> str:
    statuses = [f"{name}:{value}" for name, value in (group.get("metric_status") or {}).items() if value != "measured"]
    return ",".join(statuses) if statuses else "all_measured"


def _status_counts(groups: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter = Counter()
    for group in groups:
        for value in (group.get("metric_status") or {}).values():
            counts[str(value)] += 1
    return dict(counts)


def render_summary(document: dict[str, Any]) -> str:
    """@brief 人类可读摘要：四张表分开列，缺失与不可判显式写出。"""
    inventory = document["inventory"]
    tables = document["tables"]
    backend = document["backend_boundary"]
    effect = document["effect_assessment"]
    lines = [
        "Harness 收益计分卡 H1/H2（只读；无模型/GPU 调用）",
        f"  runs: {inventory['run_count']}；lifecycle: {inventory['counts']['lifecycle']}",
        f"  cohort: {inventory['counts']['cohort']}；experiment kind: {inventory['counts']['experiment_kind']}",
        f"  evidence kind: {inventory['counts']['evidence_kind']}；experiment containers (not runs): {len(inventory['experiment_containers'])}",
        "",
        "1) run_efficiency（单位 run）",
    ]
    run_groups = sorted(
        tables["run_efficiency"]["groups"],
        key=lambda group: (
            group["key"].get("cohort") != "natural",
            -(group["counts"].get("runs") or 0),
            str(group["key"]),
        ),
    )
    for group in run_groups[:12]:
        key = group["key"]
        counts = group["counts"]
        ratios = group["ratios"]
        lines.append(
            f"  {key['task']}/{key['dtype']}/{key['size']} {key['cohort']}/{key['experiment_kind']} "
            f"{key['evidence_kind']}: runs={counts['runs']} ok={_shown(ratios['ok_rate'])} "
            f"first_model_ok={_shown(counts['first_model_ok_runs'])} confirmed_ok={counts['confirmed_status'].get('ok', 0)} "
            f"exact_cost_per_ok={_shown(ratios['exact_cost_per_ok'])} [{_status_brief(group)}]"
        )
    if len(tables["run_efficiency"]["groups"]) > 12:
        lines.append(f"  ... {len(tables['run_efficiency']['groups']) - 12} more groups")
    lines.extend(["", "2) model_session_efficiency（单位 session）"])
    session_groups = sorted(
        tables["model_session_efficiency"]["groups"],
        key=lambda group: (
            group["key"].get("cohort") != "natural",
            -(group["counts"].get("sessions") or 0),
            str(group["key"]),
        ),
    )
    for group in session_groups[:12]:
        key = group["key"]
        counts = group["counts"]
        lines.append(
            f"  {key['provider']}/{key['model']} {key['committee']} {key['cohort']}: "
            f"sessions={counts['sessions']} no_submission={_shown(group['ratios']['no_submission_rate'])} "
            f"conflicts={counts['conflicts']} cost={_shown(counts['model_totals_cost_usd'])} [{_status_brief(group)}]"
        )
    if len(tables["model_session_efficiency"]["groups"]) > 12:
        lines.append(f"  ... {len(tables['model_session_efficiency']['groups']) - 12} more groups")
    lines.extend(["", "3) delivery_and_feedback（三种 record 类型，不共享分母）"])
    records = tables["delivery_and_feedback"]["record_types"]
    delivery = records["delivery_receipt"]
    claims = records["claim_evidence"]
    routes = records["route_m_trace"]
    delivery_counts = {
        name: _count_or_none(delivery["groups"], name)
        for name in ("receipts", "static_omitted_bytes", "dynamic_omitted_bytes")
    }
    lines.append(
        f"  delivery_receipt: groups={delivery['group_count']} receipts={_shown(delivery_counts['receipts'])} "
        f"static_omitted={_shown(delivery_counts['static_omitted_bytes'])} "
        f"dynamic_omitted={_shown(delivery_counts['dynamic_omitted_bytes'])} render_only={delivery['render_only_receipts']} "
        f"metric_status={_status_counts(delivery['groups'])}"
    )
    claim_counts = Counter()
    level_counts = Counter()
    for group in claims["groups"]:
        claim_counts.update({key: value for key, value in group["counts"].items() if isinstance(value, int)})
        level_counts.update(group["counts"].get("evidence_levels") or {})
    repetition_rows = claims["repetition"]["by_cohort_and_experiment"]
    repeated = ";".join(
        f"{row['cohort']}|{row['experiment_group'] or '-'}={row['repeated_keys']}x{row['repeated_occurrences']}"
        for row in repetition_rows if row["repeated_keys"]
    ) or "none"
    lines.append(
        f"  claim_evidence: groups={claims['group_count']} proposals={claim_counts.get('proposal_claims', 0)} "
        f"feedback_claims={claim_counts.get('feedback_claims', 0)} verified_premises={claim_counts.get('premise:verified', 0)} "
        f"host_process={level_counts.get('host_process', 0)} repeated={repeated} "
        f"free_claims={claims['repetition']['claims_without_structural_key']} "
        f"metric_status={_status_counts(claims['groups'])}"
    )
    route_counts = Counter()
    for group in routes["groups"]:
        route_counts.update({key: value for key, value in group["counts"].items() if isinstance(value, int)})
    lines.append(
        f"  route_m_trace: coverage={routes['coverage']} groups={routes['group_count']} "
        f"traces={route_counts.get('traces', 0)} shadow_eligible={route_counts.get('shadow_eligible', 0)} "
        f"metric_status={_status_counts(routes['groups'])}"
    )
    lines.extend(["", "4) experience_and_governance（experience 指针 / trust 逐 grant）"])
    experience = tables["experience_and_governance"]["experience"]
    governance = tables["experience_and_governance"]["governance"]
    lines.append(
        f"  experience: {experience['status']} files={len(experience['files'])} "
        f"sections={_shown(experience['summary'].get('sections'))} "
        f"pointers={_shown(experience['summary'].get('pointer:total'))} "
        f"missing={_shown(experience['summary'].get('pointer:missing'))}"
    )
    lines.append(f"  trust: {governance['status']} grants={len(governance['grants'])} states={governance['states']}")
    lines.extend([
        "",
        f"backend: real={backend['real_backends']} simulated={backend['simulated_backends']} "
        f"cross_backend={backend['cross_backend_status']}",
        f"NPU device metrics: {backend['npu']['device_metrics']}（{backend['npu']['reason']}）",
        f"effect assessment: {effect['status']}；claim_allowed={effect['effect_claim_allowed']}",
        f"publication_allowed={document['publication_allowed']}；"
        f"this audit starts H3={document.get('h3_started_by_this_audit')}",
    ])
    return "\n".join(lines) + "\n"


__all__ = ["SCHEMA", "render_summary", "scorecard"]
