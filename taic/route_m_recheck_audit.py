"""Read-only route-M review over exact-SHA paired rechecks.

There may be no real ``harness_route_m.jsonl`` yet. In that case coverage is
unavailable, not zero-percent success. This audit replays the existing judge in
memory with the exact recheck sources and the real trust tables. It writes no
trace, overlay, grant, task book, or activation decision.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from . import trust
from .overlay_route_m import _case_from_document, _feedback_by_label, _run_backend, judge

SCHEMA = "shapeforge.route-m-recheck-audit.1"


def _sources(run_dir: Path, wrapper: dict[str, Any], document: dict[str, Any]) -> dict[str, str]:
    """@brief 只接受重测命令里的精确路径，且字节必须等于记录的 SHA。"""
    command = wrapper.get("command") or []
    if len(command) < 6 or command[3] != "paired-compare":
        return {}
    found: dict[str, str] = {}
    for value, key in ((command[4], "sha_a"), (command[5], "sha_b")):
        expected = str(document.get(key) or "")
        path = Path(str(value)).resolve()
        try:
            path.relative_to(run_dir.resolve())
            raw = path.read_bytes()
        except (OSError, ValueError):
            continue
        if hashlib.sha256(raw).hexdigest() == expected:
            found[expected] = raw.decode("utf-8", errors="replace")
    return found


def _first_blocker(trace: dict[str, Any]) -> str:
    gates = trace.get("gates") or []
    failed = next((gate for gate in gates if gate.get("status") == "fail"), None)
    if failed is not None:
        return f"gate:{failed.get('name')}"
    not_run = next((gate for gate in gates if gate.get("status") == "not_run"), None)
    if not_run is not None:
        return f"not_run:{not_run.get('reason')}"
    return "shadow_eligible" if trace.get("shadow_written") else "unclassified"


def audit(
    runs_root: Path,
    *,
    trust_root: Path,
    prompt_sha1: str | None = None,
    protocol_version: str | None = None,
) -> dict[str, Any]:
    """@brief 在内存中重放路线 M，返回真实门迹覆盖与离线门位分布。

    @param runs_root 包含 ``paired_comparison_recheck.json`` 的 run 根目录。
    @param trust_root 现有 targets 根目录；只读，不授权。
    @param prompt_sha1 信任门使用的当前提示指纹，空则取当前值。
    @param protocol_version 信任门使用的协议代次，空则取当前值。
    @return 可 JSON 序列化的只读审计文档。
    """
    runs_root = Path(runs_root)
    trust_root = Path(trust_root)
    prompt = str(prompt_sha1 or trust.current_prompt_sha1())
    protocol = str(protocol_version or trust.current_protocol_version())
    actual_files = sorted(runs_root.rglob("harness_route_m.jsonl"))
    actual_traces = 0
    actual_shadows = 0
    for path in actual_files:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "route_m":
                continue
            actual_traces += 1
            actual_shadows += row.get("shadow_written") is True

    counts = Counter()
    gate_counts = Counter()
    blocker_counts = Counter()
    rows = []
    for path in sorted(runs_root.rglob("paired_comparison_recheck.json")):
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        document = wrapper.get("document") if isinstance(wrapper.get("document"), dict) else {}
        run_dir = path.parent.parent.parent
        label = str(document.get("session_label") or path.parent.name.split("_", 1)[-1])
        feedback = _feedback_by_label(run_dir)
        run_backend = _run_backend(run_dir)
        evidence_id = "recheck:" + hashlib.sha256(path.read_bytes()).hexdigest()
        case = _case_from_document(
            document,
            evidence_id=evidence_id,
            path=path,
            label=label,
            feedback=feedback,
            sources=_sources(run_dir, wrapper, document),
            run_backend=run_backend,
            run_dir=run_dir,
        )
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        target_id = str((config.get("target") or {}).get("id") or case.get("chip") or "")
        task = str(config.get("task") or case.get("task") or "")
        trace = judge(
            case,
            target_id=target_id,
            task=task,
            protocol_version=protocol,
            trust_root=trust_root,
            prompt_sha1=prompt,
            backend=run_backend,
            run_name=run_dir.name,
        )
        counts["cases"] += 1
        counts[f"conclusion:{trace.get('conclusion')}"] += 1
        counts[f"attribution:{(trace.get('structural_attribution') or {}).get('kind')}"] += 1
        counts[f"timed_by:{document.get('timed_by')}"] += 1
        if trace.get("shadow_written"):
            counts["shadow_eligible"] += 1
        blocker = _first_blocker(trace)
        blocker_counts[blocker] += 1
        gates = []
        for gate in trace.get("gates") or []:
            key = f"{gate.get('name')}:{gate.get('status')}"
            gate_counts[key] += 1
            gates.append({"name": gate.get("name"), "status": gate.get("status"), "reason": gate.get("reason")})
        rows.append({
            "run_id": str(run_dir.relative_to(runs_root)),
            "session_label": label,
            "conclusion": trace.get("conclusion"),
            "direction": trace.get("direction"),
            "timed_by": document.get("timed_by"),
            "attribution_kind": (trace.get("structural_attribution") or {}).get("kind"),
            "author": trace.get("author"),
            "shadow_eligible": trace.get("shadow_written") is True,
            "blocker": blocker,
            "gates": gates,
            "delivery": trace.get("delivery"),
        })
        counts["injected_bytes"] += int((trace.get("delivery") or {}).get("injected_bytes") or 0)
    trust_failures = gate_counts.get("overlay_tip_trust:fail", 0)
    notes = [
        "真实 runs 没有路线 M 门迹时，覆盖率是 unavailable，不是 0% 成功或失败。",
        "离线重放使用精确 SHA 源码与现有信任表，不写 trace/overlay，不激活，也不替代未来真实 shadow run。",
    ]
    if trust_failures:
        notes.append("现有 overlay:tip 信任门有拒绝；本审计不重签、不扩大 scope。真实 shadow 生成若需要该权限，必须另行具名授权。")
    return {
        "schema": SCHEMA,
        "runs_root": str(runs_root),
        "trust_root": str(trust_root),
        "prompt_sha1": prompt,
        "protocol_version": protocol,
        "actual_route_m": {
            "trace_files": len(actual_files),
            "traces": actual_traces,
            "shadow_rows": actual_shadows,
            "coverage": "available" if actual_files else "unavailable",
        },
        "offline_replay": {
            "counts": dict(counts),
            "gate_counts": dict(gate_counts),
            "blocker_counts": dict(blocker_counts),
            "cases": rows,
        },
        "activation_authorized": False,
        "changes_activation": False,
        "verdict": "no_real_route_m_traces_offline_replay_only" if not actual_files else "real_traces_and_offline_replay",
        "notes": notes,
    }


def render_summary(document: dict[str, Any]) -> str:
    actual = document["actual_route_m"]
    replay = document["offline_replay"]
    counts = replay["counts"]
    lines = [
        "Route M shadow-only recheck audit (read-only)",
        f"  real traces: {actual['traces']} in {actual['trace_files']} files; coverage={actual['coverage']}",
        f"  offline cases: {counts.get('cases', 0)}; shadow eligible: {counts.get('shadow_eligible', 0)}; injected bytes: {counts.get('injected_bytes', 0)}",
        f"  conclusions: diff={counts.get('conclusion:diff', 0)}, no_diff={counts.get('conclusion:no_diff', 0)}, too_noisy={counts.get('conclusion:too_noisy', 0)}",
        f"  activation authorized: {document['activation_authorized']}; verdict: {document['verdict']}",
        "  blockers:",
    ]
    for name, count in sorted(replay["blocker_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"    {count:3d}  {name}")
    return "\n".join(lines) + "\n"


__all__ = ["SCHEMA", "audit", "render_summary"]
