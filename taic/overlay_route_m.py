"""Route M: a shadow P1 overlay backed by a host paired comparison.

The host measurement is the evidence. A fenced proposal and premise prose are
nominations: they can be quoted beside the measurement, and they cannot satisfy
a gate by themselves. The task line is synthesized here. It names the chip and
the task, keeps the measured conclusion apart from the model's words, and
carries a reusable feature difference when there is one. A point ratio stays in
the comparison file.

Passing every gate writes a shadow overlay row. The row is not injected into
the task book. ``harness.settle(..., shadow=False)`` does not turn this route on.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from taic import harness, trust
from taic.paired_compare import NOMINATION_SCHEMA, PairedCompareError, normalized_pair
from taic.proposal_routing import route_proposal

TRACE_FILE = "harness_route_m.jsonl"
ROUTE = "M"
HOST_SCHEMA = "taic.paired_comparison.1"
GATE_ORDER = (
    "route_proposal",
    "never_auto",
    "scope",
    "non_simulation",
    "overlay_tip_trust",
)
_MODEL_CONCLUSIONS = frozenset({"no_diff", "diff", "regressed", "too_noisy"})
_HOST_CONCLUSIONS = frozenset({"no_diff", "diff", "too_noisy"})
_FEATURE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_DECIMAL = re.compile(r"\d+\.\d+")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SESSION_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def read_traces(run_dir: Path) -> list[dict]:
    """@brief 读出本场路线 M 的逐门留痕。

    @param run_dir run 目录。
    @return ``event=route_m`` 的行，按写入顺序。
    """
    return [
        row for row in harness._read_jsonl(Path(run_dir) / TRACE_FILE)
        if row.get("event") == "route_m"
    ]


def settle_route_m(
    run_dir: Path,
    *,
    target_id: str,
    task: str,
    protocol_version: str,
    trust_root: Path | None = None,
    prompt_sha1: str | None = None,
    backend: str | None = None,
) -> list[dict]:
    """@brief 用宿主配对比较写路线 M 的影子行和逐门留痕。

    已有同一 ``evidence_id`` 的留痕不重复写。通过全部门的比较追加一条
    ``shadow=true``、``route=M`` 的 overlay，不写 active 决策。

    @param run_dir run 目录。
    @param target_id 本次 settle 的芯片。
    @param task 本次 settle 的题目。
    @param protocol_version 协议代次，写入 overlay 行。
    @param trust_root 信任表根目录。
    @param prompt_sha1 信任要核对的提示指纹；``None`` 表示当前指纹。
    @param backend 本次 settle 的后端。空则从 ``run_config.json`` 读取。
        本次后端和证据后端都必须有值并且完全一致，空串不能通过作用域门。
    @return 本次新写下的留痕。
    """
    run_dir = Path(run_dir)
    requested_backend = str(backend or "").strip() or _run_backend(run_dir)
    existing = {str(row.get("evidence_id") or "") for row in read_traces(run_dir)}
    written: list[dict] = []
    for case in load_cases(run_dir):
        evidence_id = str(case.get("evidence_id") or "")
        if not evidence_id or evidence_id in existing:
            continue
        trace = judge(
            case,
            target_id=target_id,
            task=task,
            protocol_version=str(protocol_version),
            trust_root=trust_root,
            prompt_sha1=prompt_sha1,
            backend=requested_backend,
            run_name=run_dir.name,
        )
        overlay = trace.pop("overlay", None)
        harness._append_jsonl(run_dir / TRACE_FILE, trace)
        if trace.get("shadow_written") and overlay:
            harness._append_jsonl(run_dir / harness.OVERLAY_FILE, overlay)
        written.append(trace)
        existing.add(evidence_id)
    return written


def load_cases(run_dir: Path) -> list[dict]:
    """@brief 收集宿主比较，并把提名贴到对应的一对 SHA 上。

    候选目录里的 ``paired_comparison.json`` 和宿主选择记录里的测量是证据候选。
    同一份测量先读 ``sessions/<实际会话>/``，再读候选目录里的副本。模型工作目录里的
    ``paired_comparisons.jsonl`` 只是提名。一对 SHA 已经有合格的宿主测量时，
    只有同一场会话的提名能并进那一条，不再单独成为证据。

    @param run_dir run 目录。
    @return 待判定的案例。合格宿主测量在前，未被覆盖的提名在后。
    """
    run_dir = Path(run_dir)
    feedback = _feedback_by_label(run_dir)
    run_backend = _run_backend(run_dir)
    sources = _source_index(run_dir)
    cases: list[dict] = []
    seen: set[str] = set()
    covered: set[tuple[str, str]] = set()

    def add(case: dict) -> None:
        evidence_id = str(case.get("evidence_id") or "")
        if not evidence_id or evidence_id in seen:
            return
        seen.add(evidence_id)
        cases.append(case)
        if case.get("qualified") and case.get("sha_pair"):
            covered.add(tuple(case["sha_pair"]))

    for path in sorted((run_dir / "candidates").glob("*/paired_comparison.json")):
        add(_case_from_file(path, feedback=feedback, sources=sources, run_backend=run_backend, run_dir=run_dir))
    # Session directory name is the attempt tag, including ``.retryN``. The
    # candidate copy is the same measurement under the original candidate label.
    for path in sorted(run_dir.glob("sessions/*/paired_host_selection.json")):
        for case in _cases_from_selection(path, feedback=feedback, sources=sources, run_backend=run_backend, run_dir=run_dir):
            add(case)
    for path in sorted(run_dir.glob("candidates/*/paired_host_selection.json")):
        for case in _cases_from_selection(path, feedback=feedback, sources=sources, run_backend=run_backend, run_dir=run_dir):
            add(case)
    for nomination in _nomination_cases(run_dir, feedback):
        pair = tuple(nomination.get("sha_pair") or ())
        if len(pair) == 2 and pair in covered:
            _attach_nomination(cases, nomination)
            continue
        add(nomination)
    return cases


def judge(
    case: dict,
    *,
    target_id: str,
    task: str,
    protocol_version: str,
    trust_root: Path | None,
    prompt_sha1: str | None,
    backend: str,
    run_name: str,
) -> dict:
    """@brief 按固定门序判定一条路线 M 案例，并在全通过时组装影子 overlay。

    @param case :func:`load_cases` 的一项。
    @param target_id 芯片。
    @param task 题目。
    @param protocol_version 协议代次。
    @param trust_root 信任表根。
    @param prompt_sha1 提示指纹。
    @param backend 本次 settle 的后端。空串不能通过作用域门。
    @param run_name 写入 overlay 的来源 run 名。
    @return 留痕。``overlay`` 只在内部携带，落盘前由调用方拆走。
    """
    conclusion = str(case.get("conclusion") or "")
    attribution = case.get("attribution") if isinstance(case.get("attribution"), dict) else {
        "kind": "unknown", "only_a": [], "only_b": [],
    }
    base = {
        "schema": "taic.overlay_route_m.1",
        "event": "route_m",
        "route": ROUTE,
        "t": time.time(),
        "evidence_id": case.get("evidence_id") or "",
        "evidence_role": case.get("evidence_role") or "",
        "evidence_path": case.get("evidence_path") or "",
        "session_label": case.get("session_label") or "",
        "nomination_session": case.get("nomination_session") or "",
        "author": _author_trace(case),
        "sha_pair": list(case.get("sha_pair") or []),
        "conclusion": conclusion,
        "direction": case.get("direction"),
        "task": case.get("task") or "",
        "chip": case.get("chip") or "",
        "backend": case.get("backend") or "",
        "simulated": bool(case.get("simulated")),
        "structural_attribution": attribution,
        "model_claim_raw": _bound(case.get("model_claim_raw") or "", 500),
        "model_conclusion": case.get("model_conclusion") or "",
        "changes_activation": False,
    }
    if not case.get("qualified"):
        reason = str(case.get("qualify_reason") or "model nomination is not host evidence")
        return _finish(base, gates=_not_run(reason), task_line="", measured_text="", model_in_line="", withheld=False, withheld_reason=reason, overlay=None)
    deliverable, why = _deliverable(conclusion, attribution)
    if not deliverable:
        return _finish(base, gates=_not_run(why), task_line="", measured_text="", model_in_line="", withheld=False, withheld_reason=why, overlay=None)
    measured, model_in_line, withheld, withheld_reason = _compose(case, attribution)
    line = _join_line(measured, model_in_line)
    gates = [
        _route_gate(line),
        _never_gate(line),
        _scope_gate(case, line, target_id=target_id, task=task, backend=backend),
        _simulation_gate(case),
        _trust_gate(case, trust_root=trust_root, prompt_sha1=prompt_sha1, protocol_version=protocol_version),
    ]
    passed = all(gate["passed"] for gate in gates)
    overlay = None
    if passed and line:
        overlay = _overlay_event(
            case, line=line, target_id=target_id, task=task,
            protocol_version=protocol_version, run_name=run_name,
        )
    return _finish(
        base, gates=gates, task_line=line, measured_text=measured, model_in_line=model_in_line,
        withheld=withheld, withheld_reason=withheld_reason, overlay=overlay,
    )


def source_features(source: str) -> set[str] | None:
    """@brief 从 kernel 源码抽出可复用的特征名。分析失败时返回 ``None``。

    只看源码里已经出现的特征，不编译，也不读取比较文件里自报的特征。

    @param source MSL 源码。
    @return 特征名集合；源码不是一颗可分析 kernel 时为 ``None``。
    """
    try:
        from taic.hwfeatures import analyze_source, detect_features

        found = analyze_source(source)
        if found.get("error") or not found.get("kernel"):
            return None
        detected = detect_features(found, None)
    except Exception:
        return None
    names: set[str] = set()
    for key, row in detected.items():
        if isinstance(row, dict) and row.get("present") and _FEATURE_NAME.fullmatch(str(key)):
            names.add(str(key))
    return names


def _finish(
    base: dict, *, gates: list[dict], task_line: str, measured_text: str, model_in_line: str,
    withheld: bool, withheld_reason: str, overlay: dict | None,
) -> dict:
    """@brief 补上送达回执。影子行的注入字节数是 0。"""
    encoded = task_line.encode("utf-8") if task_line else b""
    first_fail = next((gate["name"] for gate in gates if not gate["passed"]), "")
    if overlay is not None:
        delivery_reason = "shadow"
        shadow_written = True
    elif any(gate["status"] == "not_run" for gate in gates):
        delivery_reason = withheld_reason or first_fail or "not_run"
        shadow_written = False
    else:
        delivery_reason = first_fail or "refused"
        shadow_written = False
    delivery = {
        "delivered": False,
        "reason": delivery_reason,
        "text_sha256": hashlib.sha256(encoded).hexdigest() if encoded else "",
        "text_bytes": len(encoded),
        "injected_bytes": 0,
    }
    if overlay is not None:
        overlay = dict(overlay)
        overlay["delivery"] = delivery
    row = dict(base)
    row.update({
        "gates": gates,
        "measured_text": measured_text,
        "model_claim_in_line": model_in_line,
        "model_claim_withheld": bool(withheld),
        "withheld_reason": withheld_reason,
        "task_line": task_line,
        "shadow_written": shadow_written,
        "delivery": delivery,
        "overlay": overlay,
    })
    return row


def _not_run(reason: str) -> list[dict]:
    """@brief 五道门都记为未跑，顺序不变。"""
    return [
        {"name": name, "passed": False, "status": "not_run", "reason": reason}
        for name in GATE_ORDER
    ]


def _deliverable(conclusion: str, attribution: dict) -> tuple[bool, str]:
    """@brief 没差可以送达。有差必须带可复用的特征差。噪声太大不送达。"""
    if conclusion == "no_diff":
        return True, ""
    if conclusion == "diff" and attribution.get("kind") == "differ":
        return True, ""
    if conclusion == "diff":
        return False, "diff has no reusable structural attribution"
    if conclusion == "too_noisy":
        return False, "too_noisy is not a deliverable conclusion"
    return False, "host comparison has no deliverable conclusion"


_CODE_SYNTAX = re.compile(r"[;；\n]|&&|\|\||\$\(|`")


def _compose(case: dict, attribution: dict) -> tuple[str, str, bool, str]:
    """@brief 拼宿主实测句，再决定模型说法能不能跟在后面。"""
    measured = _measured_sentence(case, attribution)
    raw = str(case.get("model_claim_raw") or "")
    model_conclusion = str(case.get("model_conclusion") or "")
    if not raw and not model_conclusion:
        return measured, "", False, ""
    cleaned = _plain_nomination(raw)
    if cleaned is not None and _model_text_allowed(cleaned):
        clause = f"模型说法（提名）：{_bound(cleaned, 120)}"
        if _line_holds(_join_line(measured, clause)):
            return measured, clause, False, ""
        withheld_reason = "model nomination would fail route_proposal or never-auto"
        because_numbers = False
    else:
        withheld_reason = (
            "model nomination is not a plain sentence"
            if cleaned is None and raw
            else "model nomination is quoted only as a conclusion word"
        )
        because_numbers = bool(re.search(r"\d", raw))
    placeholder = _withheld_clause(model_conclusion, because_numbers=because_numbers)
    if placeholder and _line_holds(_join_line(measured, placeholder)):
        return measured, placeholder, True, withheld_reason
    return measured, "", True, withheld_reason


def _plain_nomination(raw: str) -> str | None:
    """@brief 提名原文要先过 P1 的纯文本清洗，才能考虑放进任务书。

    控制字符和 Markdown 记号按 ``_clean_action`` 去掉。反引号、``&&``、
    命令替换和分号在清洗前就算代码或 Shell 语法，不能靠删掉记号混进去。
    清洗后仍不成一句的，返回 ``None``，调用方只留结论词。
    """
    text = str(raw or "")
    if not text.strip() or _CODE_SYNTAX.search(text):
        return None
    try:
        cleaned = harness._clean_action(text)
    except harness.ProposalError:
        return None
    if _CODE_SYNTAX.search(cleaned):
        return None
    return cleaned


def _join_line(measured: str, clause: str) -> str:
    """@brief 把实测句和模型说法连成任务书里的一行。"""
    if measured and clause:
        return f"{measured} {clause}"
    return measured or clause


def _measured_sentence(case: dict, attribution: dict) -> str:
    """@brief 宿主合成的实测句：芯片、题目、结论、结构归因。不含比值。"""
    chip = _field(case.get("chip"))
    task = _field(case.get("task"))
    pair = list(case.get("sha_pair") or ["", ""])
    left = _prefix(case.get("sha_a") or (pair[0] if pair else ""))
    right = _prefix(case.get("sha_b") or (pair[1] if len(pair) > 1 else ""))
    conclusion = "没差" if case.get("conclusion") == "no_diff" else "有差"
    head = f"在芯片 {chip} 的题目 {task} 上对比 kernel {left} 与 kernel {right}，宿主配对结论是{conclusion}。"
    kind = attribution.get("kind")
    only_a = [name for name in attribution.get("only_a") or [] if _FEATURE_NAME.fullmatch(str(name))]
    only_b = [name for name in attribution.get("only_b") or [] if _FEATURE_NAME.fullmatch(str(name))]
    if kind == "same":
        detail = "结构归因：两侧 kernel 特征相同。"
    elif kind == "differ":
        parts = []
        if only_a:
            parts.append(f"{left} 侧多出 {'、'.join(only_a)}")
        if only_b:
            parts.append(f"{right} 侧多出 {'、'.join(only_b)}")
        detail = "结构归因：" + "；".join(parts) + "。"
        direction = case.get("direction")
        if case.get("conclusion") == "diff" and direction == "a_faster":
            detail += f"较快的是 {left} 侧。"
        elif case.get("conclusion") == "diff" and direction == "b_faster":
            detail += f"较快的是 {right} 侧。"
    else:
        detail = "结构归因：未同时取得两份 kernel 源码，不点名特征差。"
    return _field(head + detail)


def _withheld_clause(model_conclusion: str, *, because_numbers: bool) -> str:
    """@brief 模型原文不能进任务书时，只留结论词。"""
    label = model_conclusion if model_conclusion in _MODEL_CONCLUSIONS else "未标明"
    note = "数值留在提名记录里。" if because_numbers else "原句留在提名记录里。"
    return f"模型说法（提名）：{label}。{note}这一行只保留结论词。"


def _model_text_allowed(raw: str) -> bool:
    """@brief 模型原文要能整句放进任务书：没有点比值，也过 never-auto。"""
    text = re.sub(r"\s+", " ", raw).strip()
    if not text or _DECIMAL.search(text) or re.search(r"\d", text):
        return False
    return _never_ok(text)


def _line_holds(line: str) -> bool:
    """@brief 这一行自己过路由、never-auto，并且没有点比值。"""
    if not line or _DECIMAL.search(line):
        return False
    routing = route_proposal({"action": line, "audience": "peer", "needs": "none"})
    return routing["disposition"] == "overlay_candidate" and _never_ok(line)


def _route_gate(line: str) -> dict:
    """@brief 第一道门：宿主合成的一行必须是 overlay 候选。"""
    if not line or _DECIMAL.search(line):
        return _gate("route_proposal", False, "the host line is empty or contains a point ratio")
    routing = route_proposal({"action": line, "audience": "peer", "needs": "none"})
    if routing["disposition"] == "overlay_candidate":
        return _gate("route_proposal", True, routing["reason"])
    return _gate("route_proposal", False, f"{routing['disposition']}: {routing['reason']}")


def _never_gate(line: str) -> dict:
    """@brief 第二道门：沿用 P1 的 never-auto 词表。"""
    if not _never_ok(line):
        try:
            harness._reject_never_auto(line)
        except harness.ProposalError as exc:
            return _gate("never_auto", False, str(exc))
        return _gate("never_auto", False, "the host line contains a point ratio")
    return _gate("never_auto", True, "")


def _scope_gate(case: dict, line: str, *, target_id: str, task: str, backend: str) -> dict:
    """@brief 第三道门：题目、芯片、后端与这一行里写的范围一致。

    本次后端和证据后端都要有，并且逐字相同。缺一边或空串都拒绝。
    """
    reasons = []
    if case.get("scope_conflict"):
        reasons.append(str(case["scope_conflict"]))
    if not case.get("task") or case.get("task") != task:
        reasons.append(f"scope task {case.get('task')!r} does not match {task!r}")
    if not case.get("chip") or case.get("chip") != target_id:
        reasons.append(f"scope chip {case.get('chip')!r} does not match {target_id!r}")
    case_backend = str(case.get("backend") or "").strip()
    requested = str(backend or "").strip()
    if not requested or not case_backend:
        reasons.append("scope backend is missing")
    elif case_backend != requested:
        reasons.append(f"scope backend {case_backend!r} does not match {requested!r}")
    if case.get("task") and f"题目 {case['task']}" not in line:
        reasons.append("task line does not name the task")
    if case.get("chip") and f"芯片 {case['chip']}" not in line:
        reasons.append("task line does not name the chip")
    return _gate("scope", not reasons, "; ".join(reasons))


def _simulation_gate(case: dict) -> dict:
    """@brief 第四道门：模拟测量和没有上设备的比较不能进影子行。"""
    if case.get("simulated"):
        return _gate("non_simulation", False, "simulated measurement")
    if case.get("gpu") is not True:
        return _gate("non_simulation", False, "host comparison did not run on a device")
    return _gate("non_simulation", True, "")


def _trust_gate(case: dict, *, trust_root: Path | None, prompt_sha1: str | None, protocol_version: str) -> dict:
    """@brief 第五道门：提名作者要有 active ``overlay:tip`` 信任。

    引用了提名时，作者是那一场提名会话的宿主记录，不是测量文件所在目录的作者。
    没有提名时，作者是测量所属会话。比较文件里自报的 provider 不算。
    """
    author, why = _bound_author(case)
    if why:
        return _gate("overlay_tip_trust", False, why)
    provider = str(author.get("provider") or "")
    model = str(author.get("model") or "")
    effort = str(author.get("effort") or "")
    chip = str(case.get("chip") or "")
    task = str(case.get("task") or "")
    if not provider or not model or not chip or not task:
        return _gate("overlay_tip_trust", False, "no nominating author on the host session record")
    ok, why = trust.is_trusted(
        chip, task=task, provider=provider, model=model, effort=effort,
        scope="overlay:tip", root=trust_root, prompt_sha1=prompt_sha1,
        protocol_version=protocol_version,
    )
    if ok:
        return _gate("overlay_tip_trust", True, "")
    return _gate("overlay_tip_trust", False, why or "no active overlay:tip grant")


def _gate(name: str, passed: bool, reason: str) -> dict:
    """@brief 一道门的结果。"""
    return {"name": name, "passed": bool(passed), "status": "pass" if passed else "fail", "reason": reason}


def _never_ok(text: str) -> bool:
    """@brief 文本是否通过 never-auto，并且不含点比值。"""
    if _DECIMAL.search(text or ""):
        return False
    try:
        harness._reject_never_auto(text or "")
    except harness.ProposalError:
        return False
    return True


def _overlay_event(case: dict, *, line: str, target_id: str, task: str, protocol_version: str, run_name: str) -> dict:
    """@brief 一条永远带 ``shadow=true`` 的路线 M overlay。"""
    author, _why = _bound_author(case)
    digest = str(case.get("evidence_id") or "")
    if digest.startswith("host:"):
        digest = digest.split(":", 1)[1]
    return {
        "schema_version": harness.SCHEMA_VERSION,
        "event": "overlay",
        "t": time.time(),
        "target_id": target_id,
        "task": task,
        "protocol_version": str(protocol_version),
        "source_run": run_name,
        "source_seq": author.get("seq"),
        "proposal_key": harness.proposal_key(target_id=target_id, task=task, surface="tip", action=line),
        "surface": "tip",
        "text": line,
        "evidence_sha256": digest,
        "shadow": True,
        "route": ROUTE,
        "active_from": author.get("seq"),
        "active_until": None,
        "changes_activation": False,
    }


def _case_from_file(path: Path, *, feedback: dict, sources: dict, run_backend: str, run_dir: Path) -> dict:
    """@brief 把一份宿主写下的比较文件收成案例。

    文件里的 ``session_label`` 是宿主写下的实际会话。没有这个字段的旧文件
    才用候选目录名。字段在但不是合法标签时，不退回目录名。
    """
    raw = _read_bytes(path)
    document = _parse_object(raw)
    evidence_id = "host:" + hashlib.sha256(raw or b"").hexdigest()
    if document is None:
        return _unqualified(evidence_id, path, run_dir, "host comparison file is not a JSON object")
    return _case_from_document(
        document, evidence_id=evidence_id, path=path, label=_comparison_session_label(path, document),
        feedback=feedback, sources=sources, run_backend=run_backend, run_dir=run_dir,
    )


def _cases_from_selection(path: Path, *, feedback: dict, sources: dict, run_backend: str, run_dir: Path) -> list[dict]:
    """@brief 从宿主选择记录的测量列表取证据。选中项本身仍是提名。

    测量旁边的 ``selected`` 路径只是宿主当时读过的位置。这里按测量上的 SHA
    重读并重算字节；对不上、或路径出了本场 run，就不把那段源码当特征。
    """
    document = _parse_object(_read_bytes(path))
    if not isinstance(document, dict):
        return []
    label = _selection_session_label(path, document)
    # The candidate/paired-sources index is not a substitute for this record.
    # A measurement is attributed only from paths named here whose bytes still
    # hash to the measurement SHA. ``sources`` stays unused on purpose.
    del sources
    bound = _sources_bound_by_selection(document, run_dir)
    cases = []
    for item in document.get("measurements") or []:
        if not isinstance(item, dict):
            continue
        encoded = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        cases.append(_case_from_document(
            item, evidence_id="host:" + hashlib.sha256(encoded).hexdigest(), path=path, label=label,
            feedback=feedback, sources=bound, run_backend=run_backend, run_dir=run_dir,
        ))
    return cases


def _sources_bound_by_selection(document: dict, run_dir: Path) -> dict[str, str]:
    """@brief 用选择记录里的路径找回两侧源码，但只接受字节对得上测量 SHA 的文件。"""
    qualified: set[str] = set()
    for item in document.get("measurements") or []:
        if not isinstance(item, dict):
            continue
        ok, _reason = _qualify(item)
        if not ok:
            continue
        for key in ("sha_a", "sha_b"):
            sha = str(item.get(key) or "").strip().lower()
            if _HEX64.fullmatch(sha):
                qualified.add(sha)
    if not qualified:
        return {}
    found: dict[str, str] = {}
    for row in document.get("selected") or []:
        if not isinstance(row, dict):
            continue
        for path_key, sha_key in (("path_a", "sha_a"), ("path_b", "sha_b")):
            sha = str(row.get(sha_key) or "").strip().lower()
            if sha not in qualified:
                continue
            text = _source_at(row.get(path_key), sha, run_dir)
            if text is None:
                continue
            found.setdefault(sha, text)
    return found


def _source_at(path_value: Any, expected_sha: str, run_dir: Path) -> str | None:
    """@brief 读一个落在本场 run 里的文件。SHA 不是该文件的字节就丢掉。"""
    raw_path = str(path_value or "").strip()
    if not raw_path or "\x00" in raw_path or not _HEX64.fullmatch(expected_sha):
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = Path(run_dir) / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(Path(run_dir).resolve())
    except (OSError, ValueError):
        return None
    if not resolved.is_file():
        return None
    raw = _read_bytes(resolved)
    if raw is None or len(raw) > 2_000_000:
        return None
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        return None
    return raw.decode("utf-8", errors="replace")


def _case_from_document(
    document: dict, *, evidence_id: str, path: Path, label: str, feedback: dict,
    sources: dict, run_backend: str, run_dir: Path,
) -> dict:
    """@brief 统一宿主文件和选择记录里的测量。"""
    qualified, reason = _qualify(document)
    try:
        pair = list(normalized_pair(str(document.get("sha_a") or ""), str(document.get("sha_b") or "")))
    except PairedCompareError:
        pair = []
    session = feedback.get(label) or {}
    report = _parse_object(_read_bytes(path.parent / "report.json")) or {}
    task, chip, backend, simulated, conflict = _scope(document, report, session, run_backend)
    sha_a = str(document.get("sha_a") or "").strip().lower()
    sha_b = str(document.get("sha_b") or "").strip().lower()
    author = _session_author(feedback, label)
    claim, model_conclusion = _claims_for_pair(run_dir, label, sha_a, sha_b)
    attribution = _attribution(sha_a, sha_b, sources) if qualified else {"kind": "unknown", "only_a": [], "only_b": []}
    conclusion = str(document.get("conclusion") or "")
    if conclusion not in _HOST_CONCLUSIONS:
        conclusion = ""
    return {
        "evidence_id": evidence_id,
        "evidence_role": "host_paired_comparison" if qualified else "not_host_evidence",
        "evidence_path": _relative(path, run_dir),
        "qualified": qualified,
        "qualify_reason": reason,
        "sha_pair": pair,
        "sha_a": sha_a,
        "sha_b": sha_b,
        "conclusion": conclusion,
        "direction": document.get("direction"),
        "gpu": document.get("gpu") is True,
        "task": task,
        "chip": chip,
        "backend": backend,
        "simulated": simulated,
        "scope_conflict": conflict,
        "session_label": label,
        "author": author,
        "attribution": attribution,
        "model_claim_raw": claim,
        "model_conclusion": model_conclusion,
    }


def _unqualified(evidence_id: str, path: Path, run_dir: Path, reason: str) -> dict:
    """@brief 读不出来的宿主文件。它不是提名，也不是证据。"""
    return {
        "evidence_id": evidence_id,
        "evidence_role": "not_host_evidence",
        "evidence_path": _relative(path, run_dir),
        "qualified": False,
        "qualify_reason": reason,
        "sha_pair": [],
        "conclusion": "",
        "direction": None,
        "gpu": False,
        "task": "",
        "chip": "",
        "backend": "",
        "simulated": False,
        "scope_conflict": "",
        "author": None,
        "attribution": {"kind": "unknown", "only_a": [], "only_b": []},
        "model_claim_raw": "",
        "model_conclusion": "",
    }


def _qualify(document: dict) -> tuple[bool, str]:
    """@brief 模型行、未执行的比较和身份不完整的文件不是宿主证据。"""
    if document.get("schema") == NOMINATION_SCHEMA or document.get("source") == "model" or document.get("host_evidence") is False:
        return False, "model nomination is not host evidence"
    if document.get("schema") != HOST_SCHEMA:
        return False, "not a host paired comparison"
    if document.get("applied") is not True:
        return False, str(document.get("reason") or "host comparison was not applied")
    try:
        normalized_pair(str(document.get("sha_a") or ""), str(document.get("sha_b") or ""))
    except PairedCompareError as exc:
        return False, str(exc)
    return True, ""


def _scope(document: dict, report: dict, session: dict, run_backend: str) -> tuple[str, str, str, bool, str]:
    """@brief 题目、芯片、后端和模拟标记。文件之间互相矛盾就记冲突。"""
    tasks = [str(item).strip() for item in (document.get("task"), report.get("task"), session.get("task")) if str(item or "").strip()]
    chips = [str(item).strip() for item in (document.get("target_id"), session.get("target_id")) if str(item or "").strip()]
    backends = [str(item).strip() for item in (document.get("backend"), session.get("backend"), run_backend) if str(item or "").strip()]
    conflict = []
    if len(set(tasks)) > 1:
        conflict.append("task labels disagree")
    if len(set(chips)) > 1:
        conflict.append("chip labels disagree")
    if len(set(backends)) > 1:
        conflict.append("backend labels disagree")
    simulated = any(
        item.get("simulated") is True or item.get("measurement_kind") == "simulated"
        for item in (document, session)
        if isinstance(item, dict)
    )
    return (
        tasks[0] if tasks else "",
        chips[0] if chips else "",
        backends[0] if backends else "",
        simulated,
        "; ".join(conflict),
    )


def _attribution(sha_a: str, sha_b: str, sources: dict[str, str]) -> dict:
    """@brief 两侧源码的特征差。缺源码时不把缺的一侧当成没有特征。"""
    text_a = sources.get(sha_a)
    text_b = sources.get(sha_b)
    if text_a is None or text_b is None:
        return {"kind": "unknown", "only_a": [], "only_b": []}
    feats_a = source_features(text_a)
    feats_b = source_features(text_b)
    if feats_a is None or feats_b is None:
        return {"kind": "unknown", "only_a": [], "only_b": []}
    only_a = sorted(feats_a - feats_b)[:6]
    only_b = sorted(feats_b - feats_a)[:6]
    if not only_a and not only_b:
        return {"kind": "same", "only_a": [], "only_b": []}
    return {"kind": "differ", "only_a": only_a, "only_b": only_b}


def _nomination_cases(run_dir: Path, feedback: dict) -> list[dict]:
    """@brief 模型工作目录里的配对提名。工作目录里的同名宿主文件不算证据。

    每条提名带着写出它的会话标签，以及该标签在宿主 ``feedback.jsonl`` 里的作者。
    """
    cases = []
    paths = sorted((run_dir / "agent_work").glob("*/*/.selfcheck/paired_comparisons.jsonl"))
    paths += sorted((run_dir / "agent_work").glob("*/.selfcheck/paired_comparisons.jsonl"))
    seen: set[Path] = set()
    for path in paths:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        session_label = _workdir_session_label(path, run_dir)
        author = _session_author(feedback, session_label)
        for index, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if not line.strip():
                continue
            evidence_id = "nomination:" + hashlib.sha256(f"{path}:{index}:{line}".encode("utf-8")).hexdigest()
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = None
            if not isinstance(row, dict):
                cases.append({
                    "evidence_id": evidence_id,
                    "evidence_role": "nomination",
                    "evidence_path": _relative(path, run_dir),
                    "session_label": session_label,
                    "author": author,
                    "qualified": False,
                    "qualify_reason": "model nomination is not host evidence",
                    "sha_pair": [],
                    "model_claim_raw": _bound(line, 500),
                    "model_conclusion": "",
                })
                continue
            try:
                pair = list(normalized_pair(str(row.get("sha_a") or ""), str(row.get("sha_b") or "")))
            except PairedCompareError:
                pair = []
            conclusion = str(row.get("conclusion") or "")
            claim = str(row.get("claim") or row.get("text") or "")
            cases.append({
                "evidence_id": evidence_id,
                "evidence_role": "nomination",
                "evidence_path": _relative(path, run_dir),
                "session_label": session_label,
                "author": author,
                "qualified": False,
                "qualify_reason": "model nomination is not host evidence",
                "sha_pair": pair,
                "sha_a": str(row.get("sha_a") or "").strip().lower(),
                "sha_b": str(row.get("sha_b") or "").strip().lower(),
                "model_claim_raw": claim,
                "model_conclusion": conclusion if conclusion in _MODEL_CONCLUSIONS else "",
                "conclusion": "",
            })
    return cases


def _attach_nomination(cases: list[dict], nomination: dict) -> bool:
    """@brief 把同一场会话的提名原文贴到已经有宿主证据的同一对 SHA 上。

    另一场会话的提名不贴。贴上之后，信任门看的是这场提名的作者。
    """
    pair = tuple(nomination.get("sha_pair") or ())
    source = str(nomination.get("session_label") or "")
    if not source or len(pair) != 2:
        return False
    author = nomination.get("author") if isinstance(nomination.get("author"), dict) else None
    for case in cases:
        if tuple(case.get("sha_pair") or ()) != pair or not case.get("qualified"):
            continue
        if str(case.get("session_label") or "") != source:
            continue
        if author is None:
            return False
        if not case.get("model_claim_raw") and nomination.get("model_claim_raw"):
            case["model_claim_raw"] = nomination.get("model_claim_raw") or ""
        if not case.get("model_conclusion") and nomination.get("model_conclusion"):
            case["model_conclusion"] = nomination.get("model_conclusion") or ""
        case["nomination_session"] = source
        case["nomination_author"] = author
        case["author"] = author
        return True
    return False


def _claims_for_pair(run_dir: Path, label: str, sha_a: str, sha_b: str) -> tuple[str, str]:
    """@brief 这场会话的围栏行和前提正文里同时点到两颗 SHA 的文字，只作提名。

    围栏按提案行的 ``label`` 和 ``status=valid`` 限定。别的会话、无效提案都不引用。
    """
    if not label or not (_HEX64.fullmatch(sha_a) and _HEX64.fullmatch(sha_b)):
        return "", ""
    texts = []
    for folder in (
        run_dir / "agent_work" / label,
        *_candidate_dirs(run_dir, label),
    ):
        feedback = folder / "FEEDBACK.md"
        if feedback.is_file():
            texts.append(_section(feedback.read_text(encoding="utf-8", errors="replace"), "被否决的前提"))
    premise = "\n".join(line for text in texts for line in text.splitlines() if sha_a in line.lower() and sha_b in line.lower())
    fence = ""
    for row in harness._read_jsonl(run_dir / harness.PROPOSAL_FILE):
        if str(row.get("label") or "") != label or row.get("status") != "valid":
            continue
        proposal = row.get("proposal") if isinstance(row.get("proposal"), dict) else {}
        action = str(proposal.get("action") or "")
        lowered = action.lower()
        if sha_a in lowered and sha_b in lowered:
            fence = action
            break
    raw = fence or premise.strip()
    return raw, ""


def _section(text: str, heading: str) -> str:
    """@brief 取出 Markdown 里一个标题下的正文。"""
    current = None
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            current = stripped.lstrip("#").strip()
            continue
        if current == heading:
            kept.append(line)
    return "\n".join(kept)


def _bound_author(case: dict) -> tuple[dict, str]:
    """@brief 信任门要核对的作者。

    绑了提名时只认那场提名的作者，并且这场提名必须就是测量所属的会话。
    没有提名时认测量会话自己的作者。作者记录上的会话标签对不上就拒绝。
    """
    session_label = str(case.get("session_label") or "")
    nomination_session = str(case.get("nomination_session") or "")
    if nomination_session and nomination_session != session_label:
        return {}, "nomination session does not match the host measurement session"
    raw = case.get("nomination_author") if nomination_session else case.get("author")
    author = raw if isinstance(raw, dict) else {}
    bound = str(author.get("session_label") or "")
    expected = nomination_session or session_label
    if bound and expected and bound != expected:
        return {}, "nominating author is not the source session"
    return author, ""


def _author_trace(case: dict) -> dict | None:
    """@brief 留痕里记下信任门实际核对的作者。对不上就不写作者。"""
    author, why = _bound_author(case)
    provider = str(author.get("provider") or "")
    model = str(author.get("model") or "")
    if why or not provider or not model:
        return None
    return {
        "provider": provider,
        "model": model,
        "effort": str(author.get("effort") or ""),
        "session_label": str(author.get("session_label") or ""),
    }


def _comparison_session_label(path: Path, document: dict) -> str:
    """@brief ``paired_comparison.json`` 属于哪一场会话。

    宿主写入的 ``session_label`` 优先，并且必须是单个合法标签。provider retry
    的候选目录仍叫原来的标签，实际会话在这个字段里。旧文件没有该字段时，
    才用候选目录名。字段存在但不合法时返回空串，不猜回目录上的旧会话。
    """
    if "session_label" in document:
        saved = str(document.get("session_label") or "").strip()
        if _SESSION_LABEL.fullmatch(saved):
            return saved
        return ""
    if path.parent.parent.name == "candidates":
        return _candidate_label(path.parent.name)
    return path.parent.name


def _selection_session_label(path: Path, document: dict) -> str:
    """@brief 选择记录属于哪一场会话。

    ``sessions/<标签>/`` 的目录名就是这场会话，文件里另写的标签不能改它。
    候选目录里的副本才读宿主写下的 ``session_label``：provider retry 的
    候选目录仍叫原来的标签，``.retryN`` 写在这份字段里。
    """
    parent = path.parent
    if parent.parent.name == "sessions":
        return parent.name
    saved = str(document.get("session_label") or "").strip()
    if _SESSION_LABEL.fullmatch(saved):
        return saved
    if parent.parent.name == "candidates":
        return _candidate_label(parent.name)
    return parent.name


def _workdir_session_label(path: Path, run_dir: Path) -> str:
    """@brief 提名属于 ``agent_work`` 下的第一层目录，也就是宿主分给这场会话的工作目录。

    会话里面再套一层同名目录，不能把提名算到另一场会话上。文件解析后跑出
    ``agent_work`` 的也不收。
    """
    root = (Path(run_dir) / "agent_work").resolve()
    try:
        path.resolve().relative_to(root)
        lexical = path.relative_to(Path(run_dir) / "agent_work")
    except (OSError, ValueError):
        return ""
    if not lexical.parts:
        return ""
    label = lexical.parts[0]
    if label in {".", ".."} or "/" in label or "\\" in label:
        return ""
    return label


def _session_author(feedback: dict, label: str) -> dict | None:
    """@brief 一场会话在宿主记录里的作者。没有 provider 和 model 就当没有作者。"""
    if not label or label in {".", ".."} or "/" in label or "\\" in label:
        return None
    session = feedback.get(label) if isinstance(feedback.get(label), dict) else {}
    provider = str(session.get("provider") or "")
    model = str(session.get("model") or "")
    if not provider or not model:
        return None
    return {
        "provider": provider,
        "model": model,
        "effort": str(session.get("effort") or ""),
        "seq": session.get("seq"),
        "session_label": label,
    }


def _feedback_by_label(run_dir: Path) -> dict[str, dict]:
    """@brief 宿主 ``feedback.jsonl`` 里每个 label 的最后一条会话记录。"""
    found: dict[str, dict] = {}
    for row in harness._read_jsonl(run_dir / "feedback.jsonl"):
        label = str(row.get("label") or "")
        if label:
            found[label] = row
    return found


def _run_backend(run_dir: Path) -> str:
    """@brief 从 ``run_config.json`` 读后端。没有配置时为空串。"""
    document = _parse_object(_read_bytes(run_dir / "run_config.json")) or {}
    target = document.get("target") if isinstance(document.get("target"), dict) else {}
    return str(target.get("backend") or "")


def _source_index(run_dir: Path) -> dict[str, str]:
    """@brief 宿主保存的 kernel 源码，按 SHA-256 索引。模型工作目录不参与。"""
    found: dict[str, str] = {}
    paths = list((run_dir / "candidates").glob("*/kernel.metal"))
    paths += list((run_dir / "candidates").glob("*/*.metal"))
    paths += list((run_dir / "paired-sources").glob("*.metal"))
    for path in paths:
        raw = _read_bytes(path)
        if raw is None or len(raw) > 2_000_000:
            continue
        digest = hashlib.sha256(raw).hexdigest()
        found.setdefault(digest, raw.decode("utf-8", errors="replace"))
    return found


def _candidate_dirs(run_dir: Path, label: str) -> list[Path]:
    """@brief 标签对应的候选目录。"""
    return [path for path in (run_dir / "candidates").glob(f"*_{label}") if path.is_dir()]


def _candidate_label(directory_name: str) -> str:
    """@brief ``002_r2.p0`` 里的会话标签是 ``r2.p0``。"""
    prefix, sep, label = directory_name.partition("_")
    if sep and len(prefix) == 3 and prefix.isdigit():
        return label
    return directory_name


def _field(value: Any) -> str:
    """@brief 压成一行。带点比值的字段不写进任务书。"""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if _DECIMAL.search(text):
        return ""
    return text


def _prefix(sha: str) -> str:
    """@brief SHA 的前 8 位，用来在一行里点名 kernel，不代替证据。"""
    text = str(sha or "").strip().lower()
    if _HEX64.fullmatch(text):
        return text[:8]
    return "unknown"


def _bound(value: str, limit: int) -> str:
    """@brief 截断审计用的原文。"""
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _relative(path: Path, run_dir: Path) -> str:
    """@brief 留痕里写 run 内的相对路径。"""
    try:
        return path.resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def _read_bytes(path: Path) -> bytes | None:
    """@brief 读一个小文件。读不到时返回 ``None``。"""
    try:
        if not path.is_file() or path.stat().st_size > 8_000_000:
            return None
        return path.read_bytes()
    except OSError:
        return None


def _parse_object(raw: bytes | None) -> dict | None:
    """@brief 把字节解析成 JSON 对象。"""
    if not raw:
        return None
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None
