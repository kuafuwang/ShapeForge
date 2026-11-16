"""Read-only, per-key traces for the existing harness settlement gates.

The settlement code remains the activation authority.  This module observes its
inputs and result, records why each proposal key did or did not reach an overlay,
and never writes an overlay, decision, trust grant, or control file.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

TRACE_FILE = "harness_settle_traces.jsonl"
SCHEMA = "taic.harness_settle_trace.1"
GATE_ORDER = (
    "route_proposal",
    "never_auto",
    "scope",
    "eligibility",
    "overlay_trust",
    "decision",
)


def _gate(name: str, status: str, reason: str = "") -> dict:
    return {
        "name": name,
        "status": status,
        "passed": status == "pass",
        "reason": str(reason or ""),
    }


def _not_run(reason: str) -> dict:
    return {"passed": False, "status": "not_run", "reason": str(reason or "")}


def _fresh_model_rows(rows: list[dict]) -> dict[tuple[str, str], dict]:
    authors: dict[tuple[str, str], dict] = {}
    for row in rows:
        if row.get("origin") == "conversion" or row.get("probe_time_verified") is not True:
            continue
        identity = (str(row.get("provider") or ""), str(row.get("model") or ""))
        if all(identity):
            authors.setdefault(identity, row)
    return authors


def _fingerprint(row: dict) -> str:
    body = {key: value for key, value in row.items() if key not in {"t", "fingerprint"}}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_traces(run_dir: Path) -> list[dict]:
    """Return all generic per-key settlement traces in append order."""
    from .harness import _read_jsonl

    return [
        row for row in _read_jsonl(Path(run_dir) / TRACE_FILE)
        if row.get("event") == "settle_gate_trace"
    ]


def _judge_key(
    run_dir: Path,
    key: str,
    rows: list[dict],
    *,
    target_id: str,
    task: str,
    protocol_version: str,
    enabled: bool,
    shadow: bool,
    trust_root: Path | None,
    prompt_sha1: str | None,
    result: dict,
    decisions: list[dict],
) -> dict:
    # Imports stay local so this leaf observer cannot create an import cycle.
    from . import harness
    from .proposal_routing import route_proposal

    ordered = sorted(rows, key=lambda row: (int(row.get("seq") or 0), str(row.get("run_id") or "")))
    surfaces = {
        str((row.get("proposal") or {}).get("surface") or "")
        for row in ordered if isinstance(row.get("proposal"), dict)
    }
    surface = sorted(surfaces)[0] if len(surfaces) == 1 else ("mixed" if surfaces else "")
    gates = [_gate(name, "not_run", "an earlier gate did not pass") for name in GATE_ORDER]
    emitted_match = False
    historical_decisions = [
        str(row.get("decision") or "") for row in decisions
        if row.get("proposal_key") == key and row.get("decision") in {"shadow", "apply"}
    ]

    def set_gate(index: int, status: str, reason: str = "") -> None:
        gates[index] = _gate(GATE_ORDER[index], status, reason)

    if not enabled:
        set_gate(0, "fail", "harness_auto is off")
    else:
        valid = [row for row in ordered if row.get("status") == "valid"]
        p1 = [
            row for row in valid
            if str((row.get("proposal") or {}).get("surface") or "") in harness.P1_SURFACES
        ]
        routed = [
            row for row in p1
            if route_proposal(row.get("proposal") or {}).get("disposition") == "overlay_candidate"
        ]
        if not valid:
            reasons = sorted({str(row.get("reason") or "invalid proposal") for row in ordered})
            set_gate(0, "fail", "; ".join(reasons)[:500])
        elif not p1:
            set_gate(0, "fail", "surface is not a P1 overlay surface")
        elif not routed:
            dispositions = sorted({
                str(route_proposal(row.get("proposal") or {}).get("disposition") or "unknown")
                for row in p1
            })
            set_gate(0, "fail", "not an overlay_candidate: " + ", ".join(dispositions))
        else:
            set_gate(0, "pass", route_proposal(routed[0].get("proposal") or {}).get("reason") or "")

        if gates[0]["passed"]:
            never_reasons = []
            for row in routed:
                action = str((row.get("proposal") or {}).get("action") or "")
                try:
                    harness._reject_never_auto(action)
                except harness.ProposalError as exc:
                    never_reasons.append(str(exc))
            if len(never_reasons) == len(routed):
                set_gate(1, "fail", "; ".join(sorted(set(never_reasons)))[:500])
            else:
                set_gate(1, "pass", "")
                routed = [row for row in routed if not _never_rejected(harness, row)]

        if gates[1]["passed"]:
            scoped = [
                row for row in routed
                if row.get("target_id") == target_id and row.get("task") == task
            ]
            if not scoped:
                set_gate(2, "fail", "no valid proposal matches the requested target and task")
            else:
                set_gate(2, "pass", "")
                routed = scoped

        eligible: list[dict] = []
        trusted: list[dict] = []
        if gates[2]["passed"]:
            if surface == "tip":
                all_authors = _fresh_model_rows(routed)
                authors = {
                    identity: row for identity, row in all_authors.items()
                    if row.get("simulated") is not True and row.get("measurement_kind") != "simulated"
                }
                if len(authors) < 2:
                    simulated = any(
                        row.get("simulated") is True or row.get("measurement_kind") == "simulated"
                        for row in all_authors.values()
                    )
                    reason = (
                        "simulated proposal rows cannot activate overlay" if simulated
                        else "fewer than two fresh, non-conversion provider/model authors"
                    )
                    set_gate(3, "fail", reason)
                else:
                    set_gate(3, "pass", "")
                    eligible = list(authors.values())
            elif surface == "premise":
                fresh = [
                    row for row in routed
                    if row.get("origin") != "conversion" and row.get("probe_time_verified") is True
                ]
                with_probe = [row for row in fresh if row.get("probe_sha256")]
                eligible = [
                    row for row in with_probe
                    if row.get("trusted_record")
                    and row.get("simulated") is not True and row.get("measurement_kind") != "simulated"
                ]
                if not fresh:
                    set_gate(3, "fail", "no fresh, non-conversion premise proposal")
                elif not with_probe:
                    set_gate(3, "fail", "the fresh premise has no probe SHA")
                elif not eligible:
                    simulated = any(
                        row.get("simulated") is True or row.get("measurement_kind") == "simulated"
                        for row in with_probe
                    )
                    reasons = (
                        ["simulated proposal rows cannot activate overlay"] if simulated
                        else sorted({str(row.get("trusted_reason") or "trusted_record=false") for row in with_probe})
                    )
                    set_gate(3, "fail", "; ".join(reasons)[:500])
                else:
                    set_gate(3, "pass", eligible[0].get("trusted_reason") or "")
            else:
                set_gate(3, "fail", "mixed or unknown P1 surface")

        if gates[3]["passed"]:
            scope_name = "overlay:tip" if surface == "tip" else "overlay:premise"
            why: list[str] = []
            for row in eligible:
                ok, reason = harness._trusted(
                    row, target_id=target_id, task=task, scope=scope_name,
                    root=trust_root, prompt_sha1=prompt_sha1,
                )
                if ok and row.get("probe_sha256"):
                    trusted.append(row)
                else:
                    why.append(str(reason))
            if surface == "tip":
                committees = {str(row.get("committee") or "") for row in trusted}
                probes = {str(row.get("probe_sha256") or "") for row in trusted}
                models = {(str(row.get("provider") or ""), str(row.get("model") or "")) for row in trusted}
                enough = len(models) >= 2 and len(probes) >= 2 and {"first", "second"}.issubset(committees)
            else:
                enough = bool(trusted)
            if enough:
                set_gate(4, "pass", "")
            else:
                set_gate(4, "fail", "; ".join(sorted(set(why)))[:500] or "no qualifying active overlay grant")

        emitted = list(result.get("applied") or []) + list(result.get("shadow") or [])
        emitted_match = any(row.get("proposal_key") == key for row in emitted)
        if emitted_match:
            set_gate(5, "pass", "shadow" if shadow else "apply")
        elif all(gate["passed"] for gate in gates[:5]):
            expected = "shadow" if shadow else "apply"
            if expected in historical_decisions:
                set_gate(5, "pass", "already_decided:" + expected)
            else:
                set_gate(5, "fail", "all observed gates passed but no decision exists for the requested mode")

    first_failure = next((gate["name"] for gate in gates if gate["status"] == "fail"), "")
    emitted_after_failure = bool(emitted_match and first_failure)
    return {
        "schema": SCHEMA,
        "event": "settle_gate_trace",
        "t": time.time(),
        "run_id": Path(run_dir).name,
        "proposal_key": key,
        "surface": surface,
        "route": "P" if surface == "premise" else ("M" if surface == "route_m" else "tip"),
        "target_id": target_id,
        "task": task,
        "protocol_version": str(protocol_version),
        "enabled": bool(enabled),
        "requested_mode": "shadow" if shadow else "apply",
        "source_rows": len(rows),
        "historical_decisions": historical_decisions,
        "gates": gates,
        "outcome": "trace_mismatch" if emitted_after_failure else ("eligible" if all(gate["passed"] for gate in gates) else (first_failure or "not_run")),
        "trace_consistent": not emitted_after_failure,
        "changes_activation": False,
        "observation_only": True,
    }


def _never_rejected(harness: Any, row: dict) -> bool:
    try:
        harness._reject_never_auto(str((row.get("proposal") or {}).get("action") or ""))
    except harness.ProposalError:
        return True
    return False


def record_settle_trace(
    run_dir: Path,
    *,
    target_id: str,
    task: str,
    protocol_version: str,
    enabled: bool,
    shadow: bool,
    trust_root: Path | None,
    prompt_sha1: str | None,
    result: dict,
) -> list[dict]:
    """Append changed per-key observations; an unchanged re-settle is idempotent."""
    from .harness import DECISION_FILE, PROPOSAL_FILE, _append_jsonl, _read_jsonl

    run_dir = Path(run_dir)
    proposals = _read_jsonl(run_dir / PROPOSAL_FILE)
    decisions = _read_jsonl(run_dir / DECISION_FILE)
    groups: dict[str, list[dict]] = {}
    for row in proposals:
        key = str(row.get("proposal_key") or "")
        if key:
            groups.setdefault(key, []).append(row)
    existing = {str(row.get("fingerprint") or "") for row in read_traces(run_dir)}
    written = []
    for key in sorted(groups):
        trace = _judge_key(
            run_dir, key, groups[key], target_id=target_id, task=task,
            protocol_version=protocol_version, enabled=enabled, shadow=shadow,
            trust_root=trust_root, prompt_sha1=prompt_sha1, result=result,
            decisions=decisions,
        )
        trace["fingerprint"] = _fingerprint(trace)
        if trace["fingerprint"] in existing:
            continue
        _append_jsonl(run_dir / TRACE_FILE, trace)
        existing.add(trace["fingerprint"])
        written.append(trace)
    return written
