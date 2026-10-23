"""Structured harness proposals, two-source quorum, the per-run overlay, and P2 patches.

The parser never treats free prose as an executable harness change. A model has to put a
fenced ``harness-proposal`` JSON object in the system-suggestion section. The object names
its target, task, surface, action and evidence. A tip needs two different trusted
provider/model authors from the two committees; a measured premise needs a fresh probe but
not a vote. Only a shadow/active overlay is written during a run. Repository changes are
generated as patches and need a person to apply them.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from . import control, trust

SCHEMA_VERSION = 1
TEMPLATE = """# Structured harness proposal

A tip that should participate in this run's overlay goes in ``FEEDBACK.md`` under
`# 对系统的建议`, inside one fenced block:

````markdown
```harness-proposal
{
  "surface": "tip",
  "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"},
  "action": "one imperative sentence",
  "evidence": {"probe": "probe.py", "measurement": "measured number and unit"},
  "rationale": "why this is worth trying"
}
```
````

Fields:

- ``surface``: ``tip`` (needs two trusted models) or ``premise`` (needs a fresh local probe, active ``overlay:premise`` trust, and a matching harness-written self-check row).
- ``scope.target_id`` and ``scope.task``: must match this run.
- ``action``: one sentence; no commands, code, paths that redirect writes, or contract changes.
- ``evidence.probe``: relative to this session's work directory and created after the session started.
- ``evidence.measurement``: the number and unit, when there is one.

A free-text suggestion without this block is still reviewed by a person, but it never
reaches the automatic overlay.

`# 需要帮助` is not this block. Write it in plain text as soon as a step needs a person:
a missing tool or permission, information only a person has, or a decision only a person
can make, including a license agreement, sudo, or credentials. Say what is missing and
what you will do once you have it. Do not put a command in `action`. If you can continue
another way, still write the line, then continue. The board shows it to a person. It does
not enter the overlay.

For ``premise``, the harness must also have a ``status=ok`` line in
``.selfcheck/self_checks.jsonl`` (or the outer candidate report) with the same target, protocol, and submitted kernel SHA,
and a number in ``evidence.measurement`` must match that row's ``candidate_us`` or
``speedup``. A probe file alone stays in shadow.
"""
SURFACES = ("tip", "premise", "skill", "task_wording")
P1_SURFACES = frozenset({"tip", "premise"})
MAX_ACTION = 500
MAX_OVERLAY = 1800
PROPOSAL_FILE = "harness_proposals.jsonl"
DECISION_FILE = "harness_decisions.jsonl"
OVERLAY_FILE = "harness_overlay.jsonl"
CONTROL_FILE = "harness_control.json"
PROMOTION_DIR = "harness_promotions"
_BLOCK = re.compile(r"```(?:json[ \t]+)?harness-proposal[ \t]*\r?\n(.*?)```", re.IGNORECASE | re.DOTALL)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MARK = re.compile(r"[`*_>#]")


class ProposalError(ValueError):
    """A structured proposal that cannot be used as written."""


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normal_action(text: str) -> str:
    text = _CONTROL.sub(" ", str(text or ""))
    text = _MARK.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _clean_action(text: str) -> str:
    text = _CONTROL.sub(" ", str(text or ""))
    text = _MARK.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ProposalError("action is empty")
    if len(text) > MAX_ACTION:
        text = text[:MAX_ACTION].rstrip()
    return text


_NEVER_AUTO = (
    "tasks/", "task.md", "tcenv/", "control.json", "budget_usd", "patience",
    "max_rounds", "min_improvement", "noise_z", "confirm_batches", "eval_reps",
    "tolerance", "credentials", "api key", "price", ".env",
    "git push", "git commit", "rm -rf", "subprocess", "os.system", "eval(", "exec(",
)


def _reject_never_auto(text: str) -> None:
    lower = text.lower()
    for token in _NEVER_AUTO:
        if token in lower:
            raise ProposalError(f"a P1 overlay may not contain {token!r}")
    if re.search(r"\babi\b", lower):
        raise ProposalError("a P1 overlay may not change the ABI")


def proposal_key(*, target_id: str, task: str, surface: str, action: str, dtype: str = "") -> str:
    payload = json.dumps(
        [str(target_id), str(task), str(surface), _normal_action(action), str(dtype or "")],
        ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _fresh_probe(token: str, *, work: Path, started: float | None) -> tuple[str, str, float, bool]:
    """(absolute path, sha256, mtime, time_verified) for a file under this session's work directory.

    ``started`` is the session start. Older feedback records do not carry it; in that case
    the file is still identified by path and hash, but its mtime is not compared with a
    substitute time and the result is marked unverified.
    """
    if not token or "\x00" in token:
        raise ProposalError("evidence.probe is empty")
    raw = Path(str(token))
    if raw.is_absolute():
        raise ProposalError("evidence.probe must be relative to this session's work directory")
    candidate = (Path(work) / raw).resolve()
    try:
        candidate.relative_to(Path(work).resolve())
    except ValueError as exc:
        raise ProposalError("evidence.probe escapes the session work directory") from exc
    if not candidate.is_file():
        raise ProposalError("evidence.probe is not a readable file")
    mtime = candidate.stat().st_mtime
    verified = True
    if started is not None:
        if mtime < float(started) - 1.0:
            raise ProposalError("evidence.probe predates this session")
    else:
        verified = False
    return str(candidate), _sha256(candidate), mtime, verified


def validate_proposal(
    doc: dict,
    *,
    target_id: str,
    task: str,
    dtype: str = "",
    work: Path,
    started: float | None,
    allow_premise: bool = True,
    allow_p2: bool = False,
) -> dict:
    """Return a normalized proposal or raise ProposalError. This does not call a model or GPU."""
    if not isinstance(doc, dict):
        raise ProposalError("the block is not a JSON object")
    surface = str(doc.get("surface") or "").strip()
    if surface not in SURFACES:
        raise ProposalError(f"surface must be one of {list(SURFACES)}")
    if surface not in P1_SURFACES and not allow_p2:
        raise ProposalError("only tip and premise may take effect in a run; skill and task_wording need P2")
    if surface == "premise" and not allow_premise:
        raise ProposalError("premise overlays are disabled")
    scope = doc.get("scope")
    if not isinstance(scope, dict):
        raise ProposalError("scope must be an object")
    scope_target = str(scope.get("target_id") or "").strip()
    if scope_target != target_id:
        raise ProposalError("scope.target_id does not match this run")
    scope_task = str(scope.get("task") or "").strip()
    if scope_task != task:
        raise ProposalError("scope.task does not match this run")
    scope_dtype = str(scope.get("dtype") or "").strip()
    if scope_dtype and dtype and scope_dtype != dtype:
        raise ProposalError("scope.dtype does not match this run")
    # A model that omits the dtype still names the run's dtype; otherwise the same action
    # from two models would produce two keys and never reach quorum.
    scope_dtype = scope_dtype or str(dtype or "")
    action = _clean_action(doc.get("action") or "")
    if re.search(r"[;；\n]|&&|\|\||\$\(|`", action):
        raise ProposalError("action must be one plain sentence without shell or code syntax")
    if surface in P1_SURFACES:
        _reject_never_auto(action)
    elif re.search(r"[;；]|&&|\|\||\$\(|`", action):
        raise ProposalError("action must be one plain sentence without shell or code syntax")
    evidence = doc.get("evidence") if isinstance(doc.get("evidence"), dict) else {}
    probe = str(evidence.get("probe") or "").strip()
    measurement = str(evidence.get("measurement") or "").strip()
    probe_path = probe_sha = ""
    probe_mtime = 0.0
    probe_time_verified = True
    if probe:
        probe_path, probe_sha, probe_mtime, probe_time_verified = _fresh_probe(probe, work=work, started=started)
    elif surface == "premise":
        raise ProposalError("a premise needs evidence.probe from this session")
    if surface == "tip" and not probe:
        raise ProposalError("a tip needs evidence.probe from this session")
    rationale = _CONTROL.sub(" ", str(doc.get("rationale") or ""))[:1000]
    return {
        "surface": surface,
        "scope": {"target_id": scope_target, "task": scope_task, "dtype": scope_dtype},
        "action": action,
        "evidence": {"probe": probe, "measurement": _CONTROL.sub(" ", measurement)[:500]},
        "rationale": rationale,
        "proposal_key": proposal_key(target_id=target_id, task=task, surface=surface, action=action, dtype=scope_dtype),
        "probe_path": probe_path,
        "probe_sha256": probe_sha,
        "probe_mtime": probe_mtime,
        "probe_time_verified": probe_time_verified,
    }


def extract_blocks(text: str) -> list[dict]:
    """Parse every fenced ``harness-proposal`` block. A malformed JSON block is reported, not raised."""
    out = []
    for raw in _BLOCK.findall(text or ""):
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError as exc:
            out.append({"error": f"invalid JSON: {exc}", "raw": raw})
            continue
        out.append({"doc": doc, "raw": raw})
    return out


def parse_proposals(
    parsed: dict,
    *,
    target_id: str,
    task: str,
    dtype: str = "",
    work: Path,
    started: float | None,
    allow_premise: bool = True,
) -> list[dict]:
    """Normalized proposals and errors from this session's structured blocks."""
    sections = parsed.get("sections") if isinstance(parsed.get("sections"), dict) else {}
    # Both system sections can carry a structured block; diagnostics and peer tips cannot.
    text = "\n".join(str(sections.get(name) or "") for name in ("需要帮助", "对系统的建议"))
    found = []
    for index, block in enumerate(extract_blocks(text), start=1):
        if "error" in block:
            found.append({"ok": False, "reason": block["error"], "index": index})
            continue
        try:
            proposal = validate_proposal(
                block["doc"], target_id=target_id, task=task, dtype=dtype,
                work=work, started=started, allow_premise=allow_premise,
            )
        except ProposalError as exc:
            found.append({"ok": False, "reason": str(exc), "index": index})
            continue
        proposal["ok"] = True
        proposal["index"] = index
        proposal["raw"] = block.get("raw") or ""
        found.append(proposal)
    return found


def _self_check_rows(work: Path) -> list[dict]:
    rows = []
    for path in (Path(work) / ".selfcheck" / "self_checks.jsonl", Path(work) / "self_checks.jsonl"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _measurement_has(value: str, measurement: str) -> bool:
    """Whether one of the numbers in the proposal's measurement matches a harness row."""
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", measurement or "")]
    for number in numbers:
        if number == 0:
            continue
        for candidate in re.findall(r"\d+(?:\.\d+)?", str(value or "")):
            try:
                other = float(candidate)
            except ValueError:
                continue
            if abs(number - other) <= max(0.01, abs(number) * 0.01):
                return True
    return False


def trusted_premise_record(
    work: Path,
    *,
    target_id: str,
    protocol_version: str,
    kernel_sha256: str,
    measurement: str,
    trusted_report: dict | None = None,
) -> tuple[bool, str]:
    """A premise is active only when a harness-written timed row agrees with it.

    The row must be ``status=ok``, from the same target and protocol, and bind the submitted
    kernel's SHA. At least one numeric value in ``measurement`` must match that row's
    candidate microseconds or speedup. A probe that only exists on disk does not qualify.
    """
    if not kernel_sha256:
        return False, "no submitted kernel SHA"
    rows = _self_check_rows(Path(work))
    if isinstance(trusted_report, dict):
        rows.insert(0, trusted_report)
    for row in rows:
        if row.get("status") != "ok":
            continue
        if str(row.get("target_id") or "") != str(target_id):
            continue
        if str(row.get("protocol_version") or "") != str(protocol_version):
            continue
        if str(row.get("kernel_sha256") or "") != str(kernel_sha256):
            continue
        if _measurement_has(str(row.get("candidate_us") or ""), measurement):
            return True, str(row.get("trusted_source") or "self_checks.jsonl candidate_us")
        if _measurement_has(str(row.get("speedup") or ""), measurement):
            return True, str(row.get("trusted_source") or "self_checks.jsonl speedup")
    return False, "no matching harness self-check row"


def record_proposals(
    run_dir: Path,
    *,
    record: dict,
    parsed: dict,
    target_id: str,
    task: str,
    dtype: str,
    work: Path,
    started: float,
    allow_premise: bool = True,
    kernel_sha256: str = "",
    protocol_version: str = "",
    trusted_report: dict | None = None,
) -> list[dict]:
    """Append this session's valid/invalid proposals, idempotent by (seq, key, reason)."""
    run_dir = Path(run_dir)
    seq = record.get("seq")
    if not isinstance(seq, int):
        return []
    existing = {
        (row.get("seq"), row.get("proposal_key"), row.get("status"), row.get("reason"))
        for row in _read_jsonl(run_dir / PROPOSAL_FILE)
    }
    out = []
    items = [item for item in (parsed.get("items") or []) if isinstance(item, dict)]
    system_items = [item for item in items if item.get("section") in ("需要帮助", "对系统的建议") and item.get("digest")]
    for item in parse_proposals(parsed, target_id=target_id, task=task, dtype=dtype, work=work, started=started, allow_premise=allow_premise):
        if item.get("ok"):
            key = item["proposal_key"]
            row = {
                "schema_version": SCHEMA_VERSION,
                "event": "proposal",
                "t": time.time(),
                "run_id": str(record.get("run_id") or run_dir.name),
                "seq": seq,
                "label": record.get("label") or "",
                "committee": record.get("committee") or "",
                "provider": record.get("provider") or "",
                "model": record.get("model") or "",
                "effort": record.get("effort") or "",
                "target_id": target_id,
                "task": task,
                "dtype": dtype,
                "status": "valid",
                "reason": "",
                "proposal_key": key,
                "proposal": {
                    "surface": item["surface"],
                    "scope": item["scope"],
                    "action": item["action"],
                    "evidence": item["evidence"],
                    "rationale": item["rationale"],
                },
                "probe_path": item["probe_path"],
                "probe_sha256": item["probe_sha256"],
                "probe_mtime": item["probe_mtime"],
                "probe_time_verified": item["probe_time_verified"],
                "work_dir": str(work),
                "kernel_sha256": str(kernel_sha256 or ""),
                "protocol_version": str(protocol_version or ""),
                "claim_digest": next((str(si.get("digest")) for si in system_items if "harness-proposal" in str(si.get("text") or "")), str(system_items[0].get("digest")) if system_items else ""),
                "trusted_record": False,
                "trusted_reason": "",
            }
            if item["surface"] == "premise":
                trusted, why = trusted_premise_record(
                    work, target_id=target_id, protocol_version=protocol_version,
                    kernel_sha256=kernel_sha256, measurement=str(item["evidence"].get("measurement") or ""),
                    trusted_report=trusted_report,
                )
                row["trusted_record"] = trusted
                row["trusted_reason"] = why
        else:
            key = hashlib.sha1(f"{seq}\0{item.get('index')}\0{item.get('reason')}".encode()).hexdigest()[:16]
            row = {
                "schema_version": SCHEMA_VERSION,
                "event": "proposal",
                "t": time.time(),
                "run_id": str(record.get("run_id") or run_dir.name),
                "seq": seq,
                "label": record.get("label") or "",
                "committee": record.get("committee") or "",
                "provider": record.get("provider") or "",
                "model": record.get("model") or "",
                "effort": record.get("effort") or "",
                "target_id": target_id,
                "task": task,
                "dtype": dtype,
                "status": "invalid",
                "reason": item.get("reason") or "",
                "proposal_key": key,
                "proposal": {},
            }
        signature = (row["seq"], row["proposal_key"], row["status"], row["reason"])
        if signature in existing:
            continue
        _append_jsonl(run_dir / PROPOSAL_FILE, row)
        existing.add(signature)
        out.append(row)
    return out


def _trusted(row: dict, *, target_id: str, task: str, scope: str, root: Path | None, prompt_sha1: str | None) -> tuple[bool, str]:
    return trust.is_trusted(
        target_id, task=task, provider=str(row.get("provider") or ""), model=str(row.get("model") or ""),
        effort=str(row.get("effort") or ""), scope=scope, root=root, prompt_sha1=prompt_sha1,
    )


def source_author(row: dict) -> dict:
    """The model that wrote the original claim, separate from who structured a conversion."""
    src = row.get("source_author")
    if isinstance(src, dict) and (src.get("provider") or src.get("model") or src.get("committee")):
        return {
            "provider": str(src.get("provider") or ""), "model": str(src.get("model") or ""),
            "committee": str(src.get("committee") or ""), "effort": str(src.get("effort") or ""),
        }
    return {
        "provider": str(row.get("provider") or ""), "model": str(row.get("model") or ""),
        "committee": str(row.get("committee") or ""), "effort": str(row.get("effort") or ""),
    }


def proposal_committee(row: dict) -> str:
    """The committee a reviewer must not belong to. Converted rows have no structured committee."""
    return str(source_author(row).get("committee") or row.get("committee") or "")


def _author_pairs(row: dict) -> set[tuple[str, str]]:
    """Every model identity that authored this proposal, including a converted source claim."""
    pairs: set[tuple[str, str]] = set()
    for candidate in (row, row.get("source_author") if isinstance(row.get("source_author"), dict) else {}):
        pair = (str(candidate.get("provider") or ""), str(candidate.get("model") or ""))
        if all(pair):
            pairs.add(pair)
    return pairs


def reviewer_conflict(row: dict, provider: str, model: str) -> bool:
    """Whether this model authored the proposal or the free text it was converted from."""
    pair = (str(provider or ""), str(model or ""))
    return bool(all(pair) and pair in _author_pairs(row))


def reviewer_conflicts_with_group(rows: list[dict], provider: str, model: str) -> bool:
    """A shared proposal key may have multiple authors; a reviewer must differ from all."""
    return any(reviewer_conflict(row, provider, model) for row in rows)


def pending_tip_reviews(run_dir: Path, proposals: list[dict]) -> list[dict]:
    """Valid tips that have neither a decision nor an independent review yet."""
    reviews = {str(row.get("proposal_key") or "") for row in _read_jsonl(Path(run_dir) / "harness_reviews.jsonl") if row.get("event") == "review"}
    decisions = {str(row.get("proposal_key") or "") for row in _read_jsonl(Path(run_dir) / DECISION_FILE)}
    out = []
    for row in proposals:
        key = str(row.get("proposal_key") or "")
        if not key or key in reviews or key in decisions:
            continue
        proposal = row.get("proposal") if isinstance(row.get("proposal"), dict) else {}
        if (row.get("status") == "valid" and proposal.get("surface") == "tip"
                and row.get("probe_time_verified") is True and row.get("probe_sha256")):
            out.append(row)
    return out


def settle(
    run_dir: Path,
    *,
    target_id: str,
    task: str,
    protocol_version: str,
    enabled: bool,
    shadow: bool = True,
    trust_root: Path | None = None,
    prompt_sha1: str | None = None,
) -> dict:
    """Find two-source agreement and append a shadow/active overlay event once per key.

    ``enabled`` is the run's harness switch. When false nothing is written. ``shadow`` records
    what would be injected but ``active_overlay`` ignores it until the switch is active.

    A measured premise can become active only when the proposal carries a matching
    harness-written self-check row: same target/protocol/kernel SHA and a measurement that
    agrees with the row. A probe that only exists on disk stays in shadow.
    """
    if not enabled:
        return {"applied": [], "shadow": [], "reason": "harness_auto is off"}
    run_dir = Path(run_dir)
    proposals = _read_jsonl(run_dir / PROPOSAL_FILE)
    decisions = _read_jsonl(run_dir / DECISION_FILE)
    reviews = _read_jsonl(run_dir / "harness_reviews.jsonl")
    decided = {(row.get("proposal_key"), row.get("decision")) for row in decisions}
    groups: dict[str, list[dict]] = {}
    for row in proposals:
        if row.get("status") == "valid" and row.get("target_id") == target_id and row.get("task") == task:
            groups.setdefault(str(row.get("proposal_key") or ""), []).append(row)
    applied, shadowed = [], []
    for key, rows in groups.items():
        if not key:
            continue
        surface = str(rows[0].get("proposal", {}).get("surface") or "")
        if surface not in P1_SURFACES:
            continue
        rows = [row for row in rows if (row.get("proposal") or {}).get("surface") == surface]
        if not rows:
            continue
        # One author may appear several times; only distinct provider/model/committee count.
        authors = {}
        for row in rows:
            # A conversion is not an independent model proposal, even when an older row
            # still carries the source model in the top-level provider/model fields.
            if row.get("origin") == "conversion" or row.get("probe_time_verified") is not True:
                continue
            identity = (str(row.get("provider") or ""), str(row.get("model") or ""))
            authors.setdefault(identity, row)
        distinct_models = {identity for identity in authors if all(identity)}
        if surface == "tip":
            if len(distinct_models) < 2:
                continue
            trusted = []
            for identity, row in authors.items():
                ok, _why = _trusted(row, target_id=target_id, task=task, scope="overlay:tip", root=trust_root, prompt_sha1=prompt_sha1)
                if ok and row.get("probe_sha256"):
                    trusted.append((identity, row))
            if len(trusted) < 2:
                continue
            if not {"first", "second"}.issubset({str(row.get("committee") or "") for _, row in trusted}):
                continue
            # A copied probe is not a second source. Only trusted rows count.
            if len({str(row.get("probe_sha256")) for _, row in trusted}) < 2:
                continue
        else:
            # A measured premise does not need a second model, but it still needs a person
            # to delegate the model for this surface. This also keeps an empty trust table
            # from silently letting agent-written probes into an active overlay.
            eligible = []
            for row in rows:
                if row.get("origin") == "conversion" or row.get("probe_time_verified") is not True:
                    continue
                if not row.get("probe_sha256") or not row.get("trusted_record"):
                    continue
                ok, _why = _trusted(
                    row, target_id=target_id, task=task, scope="overlay:premise",
                    root=trust_root, prompt_sha1=prompt_sha1,
                )
                if ok:
                    eligible.append(row)
            if not eligible:
                continue
        is_shadow = bool(shadow)
        decision = "shadow" if is_shadow else "apply"
        if (key, decision) in decided:
            continue
        first = trusted[0][1] if surface == "tip" else eligible[0]
        event = {
            "schema_version": SCHEMA_VERSION,
            "event": "overlay",
            "t": time.time(),
            "target_id": target_id,
            "task": task,
            "protocol_version": str(protocol_version),
            "source_run": str(first.get("run_id") or run_dir.name),
            "source_seq": first.get("seq"),
            "proposal_key": key,
            "surface": surface,
            "text": str(first.get("proposal", {}).get("action") or ""),
            "evidence_sha256": str(first.get("probe_sha256") or ""),
            "shadow": is_shadow,
            "active_from": first.get("seq"),
            "active_until": None,
        }
        _append_jsonl(run_dir / OVERLAY_FILE, event)
        _append_jsonl(run_dir / DECISION_FILE, {
            "schema_version": SCHEMA_VERSION, "event": "decision", "t": event["t"],
            "target_id": target_id, "task": task, "proposal_key": key,
            "decision": decision, "source_seq": first.get("seq"),
            "models": sorted(identity[0] + "/" + identity[1] for identity in distinct_models),
        })
        (shadowed if is_shadow else applied).append(event)
    # A single-source tip may reach *active for this run* through an independent review.
    # This is run-scoped, not a repository promotion; P2 still needs a person.
    for review in reviews:
        if review.get("useful") is not True:
            continue
        key = str(review.get("proposal_key") or "")
        group = [row for row in (groups.get(key) or []) if (row.get("proposal") or {}).get("surface") == "tip"]
        if not group:
            continue
        if (key, "apply") in decided or (key, "shadow") in decided or (key, "reviewed_active") in decided:
            continue
        reviewer = review.get("reviewer") if isinstance(review.get("reviewer"), dict) else {}
        reviewer_pair = (str(reviewer.get("provider") or ""), str(reviewer.get("model") or ""))
        authors = set().union(*(_author_pairs(row) for row in group))
        if reviewer_pair in authors:
            continue
        ok, _why = _trusted(
            {
                "provider": reviewer.get("provider"), "model": reviewer.get("model"),
                "effort": reviewer.get("effort") or "",
            },
            target_id=target_id, task=task, scope="overlay:tip", root=trust_root, prompt_sha1=prompt_sha1,
        )
        if not ok:
            continue
        source_seq = review.get("source_seq")
        if source_seq is None:
            # Historical review rows omit the source sequence. They are ambiguous when
            # several claims share one action key, so do not activate them in that case.
            first = group[0] if len(group) == 1 else None
        else:
            first = next((row for row in group if row.get("seq") == source_seq), None)
        if first is None or first.get("probe_time_verified") is not True or not first.get("probe_sha256"):
            continue
        event = {
            "schema_version": SCHEMA_VERSION, "event": "overlay", "t": time.time(),
            "target_id": target_id, "task": task, "protocol_version": str(protocol_version),
            "source_run": str(first.get("run_id") or run_dir.name), "source_seq": first.get("seq"),
            "proposal_key": key, "surface": "tip", "text": str(first.get("proposal", {}).get("action") or ""),
            "evidence_sha256": str(first.get("probe_sha256") or ""), "shadow": False,
            "origin": first.get("origin") or "model",
            "source_author": first.get("source_author") if isinstance(first.get("source_author"), dict) else {},
            "reviewed": True, "reviewer": reviewer,
            "active_from": first.get("seq"), "active_until": None,
        }
        _append_jsonl(run_dir / OVERLAY_FILE, event)
        _append_jsonl(run_dir / DECISION_FILE, {
            "schema_version": SCHEMA_VERSION, "event": "decision", "t": event["t"],
            "target_id": target_id, "task": task, "proposal_key": key,
            "decision": "reviewed_active", "source_seq": first.get("seq"),
            "origin": first.get("origin") or "model",
            "source_author": first.get("source_author") if isinstance(first.get("source_author"), dict) else {},
            "reviewer": reviewer,
        })
        applied.append(event)
    return {"applied": applied, "shadow": shadowed}


def _overlay_disabled(run_dir: Path) -> bool:
    doc = control.read_json(Path(run_dir) / CONTROL_FILE)
    return isinstance(doc, dict) and bool(doc.get("disabled"))


def set_overlay_control(run_dir: Path, *, disabled: bool, by: str = "cli", reason: str = "") -> dict:
    """Only a person disables the overlay. This file never grants a new scope."""
    if not disabled:
        raise ProposalError("the run control file can only disable an overlay; re-enable by starting a new run")
    doc = {
        "disabled": bool(disabled),
        "by": str(by),
        "reason": str(reason),
        "t": time.time(),
    }
    control.write_json_atomic(Path(run_dir) / CONTROL_FILE, doc)
    return doc


def sync_trust_status(run_dir: Path, *, target_id: str, task: str, root: Path | None = None, prompt_sha1: str | None = None) -> list[dict]:
    """Suspend a model on this target/task after repeated confirmed overturns.

    The observation is intentionally conservative: at least eight reviewed claims, from at
    least two runs, and a refuted rate of at least 25%. A person can always revoke earlier.
    """
    try:
        from observe.index import get_index
    except Exception:  # noqa: BLE001 - no index is not an error for this best-effort check
        return []
    index = get_index()
    suspended = []
    for entry in trust.load_grants(target_id, root):
        if entry.get("task") != task or entry.get("status") != "active":
            continue
        provider = str(entry.get("provider") or "")
        model = str(entry.get("model") or "")
        if not provider or not model:
            continue
        stats = index.conn.execute(
            """
            SELECT COUNT(*) AS n, COUNT(DISTINCT c.run_id) AS runs,
                   SUM(CASE WHEN c.status = 'rejected' AND c.verification = 'refuted' THEN 1 ELSE 0 END) AS bad
            FROM feedback_claims c
            JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
            JOIN runs r ON r.run_id = c.run_id
            WHERE r.target_id = ? AND i.task = ? AND i.model = ? AND c.status IN ('adopted','rejected','done')
            """,
            (target_id, task, model),
        ).fetchone()
        if not stats or int(stats["n"] or 0) < 8 or int(stats["runs"] or 0) < 2:
            continue
        if int(stats["bad"] or 0) / int(stats["n"]) < 0.25:
            continue
        try:
            entry = trust.suspend(
                target_id, task=task, provider=provider, model=model,
                effort=str(entry.get("effort") or ""), by="auto",
                reason="confirmed refuted rate crossed 25%", root=root,
            )
        except trust.TrustError:
            continue
        _append_jsonl(Path(run_dir) / DECISION_FILE, {
            "schema_version": SCHEMA_VERSION, "event": "suspend", "t": time.time(),
            "target_id": target_id, "task": task, "provider": provider, "model": model,
            "grant_id": entry.get("id"), "reason": "confirmed refuted rate crossed 25%",
        })
        suspended.append(entry)
    return suspended


def overlay_rows(run_dir: Path, *, target_id: str, task: str, protocol_version: str, shadow: bool | None = None) -> list[dict]:
    rows = _read_jsonl(Path(run_dir) / OVERLAY_FILE)
    disabled = _overlay_disabled(run_dir)
    active: dict[str, dict] = {}
    for row in rows:
        if row.get("event") == "disable":
            active.clear()
            continue
        if row.get("event") != "overlay":
            continue
        if disabled:
            continue
        if row.get("target_id") != target_id or row.get("task") != task:
            continue
        if str(row.get("protocol_version") or "") != str(protocol_version):
            continue
        if shadow is not None and bool(row.get("shadow")) != bool(shadow):
            continue
        active[str(row.get("proposal_key") or "")] = row
    return list(active.values())


def overlay_text(run_dir: Path, *, target_id: str, task: str, protocol_version: str, limit: int = MAX_OVERLAY) -> str:
    rows = overlay_rows(run_dir, target_id=target_id, task=task, protocol_version=protocol_version, shadow=False)
    if not rows:
        return ""
    lines = ["", "# 未验证的模型建议（不是合同，不是事实）",
             "以下条目由两个模型在本硬件上各自提出并经人授予的信任放行；它们仍是线索，合同和实测数字优先。"]
    used = sum(len(line) + 1 for line in lines)
    for row in rows:
        line = f"- [{row.get('source_run') or '?'} #{row.get('source_seq')}] {row.get('text') or ''}".strip()
        if used + len(line) + 1 > limit:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines) + "\n"


def overlay_control(run_dir: Path) -> dict:
    doc = control.read_json(Path(run_dir) / CONTROL_FILE)
    return doc if isinstance(doc, dict) else {"disabled": False}


def _target_rel(target_id: str) -> str:
    return f"targets/{target_id}/experience.md"


def candidate_patch(
    proposal: dict,
    *,
    target_id: str,
    proposal_key: str = "",
    root: Path | None = None,
) -> tuple[str, str]:
    """(patch, relative target file) that appends one experience bullet.

    Only the target's experience file is a P2 default. Shared prompt/skill files are never
    patched here; a person has to handle that surface separately.
    """
    if str(proposal.get("surface") or "") not in ("tip", "premise"):
        raise ProposalError("only tip and premise have a target experience default")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(target_id or "")):
        raise ProposalError("invalid target id")
    base = Path(root) if root is not None else trust.REPO_ROOT
    rel = _target_rel(target_id)
    path = base / rel
    old = path.read_text(encoding="utf-8") if path.is_file() else ""
    if old and not old.endswith("\n"):
        old += "\n"
    entry = f"- [harness {proposal_key or '?'}] {proposal.get('action') or ''}\n"
    new = old + entry
    patch = "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile="a/" + rel, tofile="b/" + rel, n=3,
    ))
    return patch, rel


def write_patch(
    run_dir: Path,
    *,
    target_id: str,
    proposal_key: str,
    root: Path | None = None,
) -> dict:
    run_dir = Path(run_dir)
    proposals = [row for row in _read_jsonl(run_dir / PROPOSAL_FILE) if row.get("proposal_key") == proposal_key and row.get("status") == "valid"]
    if not proposals:
        raise ProposalError("no valid proposal with that key")
    row = proposals[0]
    patch, rel = candidate_patch(row.get("proposal") or {}, target_id=target_id, proposal_key=proposal_key, root=root)
    folder = run_dir / PROMOTION_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{proposal_key}.patch"
    path.write_text(patch, encoding="utf-8")
    meta = {
        "schema_version": SCHEMA_VERSION, "proposal_key": proposal_key, "target_id": target_id,
        "path": rel, "patch": str(path), "sha256": _sha256(path), "source_seq": row.get("seq"),
        "source_run": row.get("run_id"), "text": (row.get("proposal") or {}).get("action"),
        "claim_digest": row.get("claim_digest") or "",
    }
    control.write_json_atomic(folder / f"{proposal_key}.json", meta)
    return meta


def apply_patch(run_dir: Path, *, target_id: str, proposal_key: str, root: Path | None = None, check_only: bool = False, run_tests: bool = False, applied_by: str = "cli") -> dict:
    """Apply a generated target-experience patch **locally**, after ``git apply --check``.

    This is local activation, not a git promotion: the running checkout changes now, while
    ``change_ref`` stays empty until a person commits the touched file. That keeps the
    traceable git reference separate from the fast local feedback loop.
    """
    run_dir = Path(run_dir)
    meta = write_patch(run_dir, target_id=target_id, proposal_key=proposal_key, root=root)
    repo = Path(root) if root is not None else trust.REPO_ROOT
    patch = Path(meta["patch"])
    check = subprocess.run(["git", "apply", "--check", str(patch)], cwd=repo, capture_output=True, text=True)
    if check.returncode != 0:
        raise ProposalError((check.stderr or check.stdout or "git apply --check failed").strip())
    if check_only:
        return {**meta, "applied": False, "check": True}
    if run_tests:
        import sys

        tests = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/test_agent_feedback.py", "tests/test_kernel_agent.py", "tests/test_hw_feedback.py"],
            cwd=repo, capture_output=True, text=True,
        )
        if tests.returncode != 0:
            raise ProposalError((tests.stdout + tests.stderr)[-2000:].strip() or "offline tests failed")
    apply = subprocess.run(["git", "apply", str(patch)], cwd=repo, capture_output=True, text=True)
    if apply.returncode != 0:
        raise ProposalError((apply.stderr or apply.stdout or "git apply failed").strip())
    _append_jsonl(run_dir / DECISION_FILE, {
        "schema_version": SCHEMA_VERSION, "event": "promotion", "status": "local", "t": time.time(),
        "target_id": target_id, "proposal_key": proposal_key, "change_ref": "",
        "path": meta["path"], "paths": [meta["path"]], "patch": str(patch),
        "patch_sha256": meta["sha256"], "local_patch_sha256": meta["sha256"],
        "claim_digest": meta.get("claim_digest") or "", "source_seq": meta.get("source_seq"),
        "applied_by": str(applied_by or "cli"),
        "tests_run": "feedback/contract/hw_off" if run_tests else "",
        "note": "applied to the local checkout; change_ref is assigned when the touched file is committed",
    })
    return {**meta, "applied": True, "change_ref": "", "local": True, "commit_pending": True}


def commit_local_promotions(run_dir: Path, *, message: str, by: str = "cli", root: Path | None = None) -> dict:
    """Commit locally promoted target/code paths and attach ``commit:<sha>`` to each event.

    The local checkout already contains the change; this is the later traceability step.
    It deliberately uses ``git commit --only -- <paths>`` so unrelated staged changes do
    not get swept into the promotion commit, and it checks that each recorded patch still
    matches the working tree exactly.
    """
    run_dir = Path(run_dir)
    repo = Path(root) if root is not None else trust.REPO_ROOT
    decisions = _read_jsonl(run_dir / DECISION_FILE)
    committed = {
        (str(row.get("proposal_key") or ""), str(row.get("patch_sha256") or row.get("local_patch_sha256") or ""))
        for row in decisions if row.get("event") == "promotion_commit"
    }
    pending: list[tuple[dict, list[str], Path | None]] = []
    for row in decisions:
        if row.get("status") != "local" or str(row.get("change_ref") or "").startswith("commit:"):
            continue
        signature = (str(row.get("proposal_key") or ""), str(row.get("patch_sha256") or row.get("local_patch_sha256") or ""))
        if signature in committed:
            continue
        if row.get("event") == "promotion" and row.get("path"):
            patch = Path(str(row["patch"])) if row.get("patch") else None
            pending.append((row, [str(row["path"])], patch))
        elif row.get("event") == "code_promotion" and row.get("paths"):
            patch = Path(str(row["patch"])) if row.get("patch") else None
            pending.append((row, [str(path) for path in row["paths"]], patch))
    if not pending:
        raise ProposalError("there is no locally applied promotion waiting for a git commit")
    paths = sorted({path for _, item_paths, _ in pending for path in item_paths})
    target_paths = {path for row, item_paths, _ in pending if row.get("event") == "promotion" for path in item_paths}
    missing = [path for path in target_paths if not (repo / path).is_file()]
    if missing:
        raise ProposalError("promotion path is missing: " + ", ".join(missing))
    # A promotion patch can add or delete files. Before committing, require the live
    # checkout to contain exactly HEAD plus every pending patch, in application order, so
    # unrelated edits in the same files are not swept into the promotion commit.
    patches = []
    for row, item_paths, patch in pending:
        if patch is None or not patch.is_file():
            raise ProposalError(f"promotion patch is missing: {patch or row.get('proposal_key')}")
        expected_sha = str(row.get("patch_sha256") or row.get("local_patch_sha256") or "")
        if not expected_sha or _sha256(patch) != expected_sha:
            raise ProposalError(f"promotion patch hash does not match its application record: {patch}")
        if row.get("event") == "code_promotion":
            from . import code_agent

            if code_agent._validate_patch(patch.read_text(encoding="utf-8", errors="replace")) != item_paths:
                raise ProposalError(f"promotion patch paths changed since application: {patch}")
        patches.append(patch)
    # A private index makes the validated snapshot the exact commit input. The caller's
    # index (including unrelated staged paths) remains untouched.
    from . import code_agent

    with tempfile.TemporaryDirectory(prefix="taic-promotion-index-") as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        def git(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True)

        base = git("read-tree", "HEAD")
        if base.returncode != 0:
            raise ProposalError((base.stderr or base.stdout or "cannot create promotion index").strip())
        staged = git("add", "--", *paths)
        if staged.returncode != 0:
            raise ProposalError((staged.stderr or staged.stdout or "git add failed").strip())
        if patches:
            patch_paths = sorted({path for _row, item_paths, patch in pending if patch is not None for path in item_paths})
            exact, where = code_agent.worktree_matches_patches(repo, patches, patch_paths, staged_env=env)
            if not exact:
                raise ProposalError(f"the applied promotion patches no longer match the staged snapshot at {where}")
        tree = git("write-tree")
        if tree.returncode != 0:
            raise ProposalError((tree.stderr or tree.stdout or "cannot write promotion tree").strip())
        parent = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        created = subprocess.run(
            ["git", "commit-tree", tree.stdout.strip(), "-p", parent], input=str(message) + "\n",
            cwd=repo, capture_output=True, text=True,
        )
        if created.returncode != 0:
            raise ProposalError((created.stderr or created.stdout or "git commit-tree failed").strip())
        sha = created.stdout.strip()
        updated = subprocess.run(["git", "update-ref", "HEAD", sha, parent], cwd=repo, capture_output=True, text=True)
        if updated.returncode != 0:
            raise ProposalError((updated.stderr or updated.stdout or "HEAD changed before promotion commit").strip())
        # Reset only committed paths in the caller's index to the new HEAD. A concurrent
        # working-tree edit must remain unstaged and must never alter the committed tree.
        synced = subprocess.run(["git", "reset", "--quiet", "HEAD", "--", *paths], cwd=repo, capture_output=True, text=True)
        sync_warning = (synced.stderr or synced.stdout or "index sync failed").strip() if synced.returncode else ""
    change_ref = f"commit:{sha}"
    committed_rows = []
    for row, item_paths, _patch in pending:
        committed_rows.append(row.get("proposal_key"))
        _append_jsonl(run_dir / DECISION_FILE, {
            "schema_version": SCHEMA_VERSION, "event": "promotion_commit", "t": time.time(),
            "target_id": row.get("target_id") or "", "proposal_key": row.get("proposal_key") or "",
            "change_ref": change_ref, "path": item_paths[0] if len(item_paths) == 1 else "",
            "paths": item_paths,
            "patch_sha256": row.get("patch_sha256") or row.get("local_patch_sha256") or "",
            "local_patch_sha256": row.get("local_patch_sha256") or row.get("patch_sha256") or "",
            "applied_by": str(by or "cli"),
        })
        if row.get("claim_digest"):
            try:
                from observe.feedback import apply_review

                apply_review(run_dir, int(row.get("source_seq") or 0), str(row["claim_digest"]), {
                    "status": "done", "verification": "confirmed", "change_ref": change_ref,
                    "evidence_source": "code", "evidence": f"applied {', '.join(item_paths)}", "note": "P2 promotion commit",
                })
            except (OSError, ValueError, ImportError):
                pass
    result = {"committed": True, "change_ref": change_ref, "commit": sha, "paths": paths, "proposals": committed_rows}
    if sync_warning:
        result["index_sync_warning"] = sync_warning
    return result


def free_text_suggestions(run_dir: Path) -> list[dict]:
    """System suggestions that never became a fenced ``harness-proposal`` block.

    A human may confirm one of these and then convert it manually or with an AI into a
    structured row. The conversion is explicit: this function only reads; it never writes
    a proposal or sends anything to a model.
    """
    run_dir = Path(run_dir)
    proposals = _read_jsonl(run_dir / PROPOSAL_FILE)
    converted: dict[tuple[int, str], str] = {}
    for row in proposals:
        digest = str(row.get("claim_digest") or "")
        seq = row.get("seq")
        if digest and isinstance(seq, int):
            if row.get("status") == "valid":
                converted[(seq, digest)] = str(row.get("proposal_key") or "")
    out, seen = [], set()
    for record in _read_jsonl(run_dir / "feedback.jsonl"):
        seq = record.get("seq")
        if not isinstance(seq, int):
            continue
        for item in record.get("items") or []:
            if not isinstance(item, dict) or item.get("section") != "对系统的建议":
                continue
            text = str(item.get("text") or "").strip()
            digest = str(item.get("digest") or "")
            if not text or not digest or "harness-proposal" in text:
                continue
            key = (seq, digest)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "seq": seq,
                "digest": digest,
                "text": text,
                "label": record.get("label") or "",
                "run_id": record.get("run_id") or run_dir.name,
                "task": record.get("task") or "",
                "dtype": record.get("dtype") or "",
                "provider": record.get("provider") or "",
                "model": record.get("model") or "",
                "committee": record.get("committee") or "",
                "t": record.get("t"),
                "session_started": record.get("session_started"),
                "work_dir": record.get("work_dir") or "",
                "converted_key": converted.get(key, ""),
            })
    return out


def convert_feedback(
    run_dir: Path,
    *,
    seq: int,
    digest: str,
    target_id: str,
    task: str,
    dtype: str = "",
    surface: str,
    action: str,
    probe: str = "",
    measurement: str = "",
    rationale: str = "",
    by: str = "cli",
    confirm: bool = False,
    mode: str = "manual",
    supersede: str = "",
) -> dict:
    """Turn one human-confirmed free-text suggestion into a structured proposal row.

    The source text remains attached as provenance. This writes only the local run's
    ``harness_proposals.jsonl``; it does not run a reviewer. ``confirm`` is the human gate.
    """
    if not confirm:
        raise ProposalError("converting free text requires explicit human confirmation")
    if str(mode or "manual") not in ("manual", "ai"):
        raise ProposalError("mode must be manual or ai")
    run_dir = Path(run_dir)
    source = next((
        row for row in free_text_suggestions(run_dir)
        if int(row.get("seq") or 0) == int(seq) and str(row.get("digest") or "") == str(digest)
    ), None)
    if source is None:
        raise ProposalError("no unconverted free-text suggestion with that seq/digest")
    existing_key = str(source.get("converted_key") or "")
    if existing_key:
        if not supersede:
            raise ProposalError("this suggestion has already been converted; pass supersede with the current proposal key")
        if str(supersede) != existing_key:
            raise ProposalError("supersede does not match the current proposal key")
    elif supersede:
        raise ProposalError("supersede was given but this suggestion has no conversion")
    records = [row for row in _read_jsonl(run_dir / "feedback.jsonl") if row.get("seq") == int(seq)]
    record = records[0] if records else {}
    fallback_work = "agent_work/" + str(source.get("label") or "unknown").replace("/", "_")
    work_rel = str(record.get("work_dir") or source.get("work_dir") or fallback_work)
    work = (run_dir / work_rel).resolve()
    try:
        work.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ProposalError("the feedback record's work_dir escapes the run directory") from exc
    # The feedback row is written when the session ends, so its own timestamp cannot prove
    # when a probe was created. Only the recorded session start can. Old rows lack it.
    raw_started = record.get("session_started")
    try:
        started = float(raw_started) if raw_started is not None else None
    except (TypeError, ValueError):
        started = None
    doc = {
        "surface": str(surface or "").strip(),
        "scope": {"target_id": target_id, "task": task, "dtype": dtype},
        "action": action,
        "evidence": {"probe": probe, "measurement": measurement},
        "rationale": rationale,
    }
    normalized = validate_proposal(
        doc, target_id=target_id, task=task, dtype=dtype, work=work, started=started,
        allow_premise=True, allow_p2=True,
    )
    if str(normalized["proposal_key"]) == existing_key:
        raise ProposalError("the revised action has the same proposal key; change the action")
    row = {
        "schema_version": SCHEMA_VERSION,
        "event": "proposal",
        "t": time.time(),
        "run_id": str(record.get("run_id") or run_dir.name),
        "seq": int(seq),
        "label": record.get("label") or source.get("label") or "",
        "target_id": target_id,
        "task": task,
        "dtype": dtype,
        "status": "valid",
        "reason": "",
        "proposal_key": normalized["proposal_key"],
        "proposal": {
            "surface": normalized["surface"],
            "scope": normalized["scope"],
            "action": normalized["action"],
            "evidence": normalized["evidence"],
            "rationale": normalized["rationale"],
        },
        "probe_path": normalized["probe_path"],
        "probe_sha256": normalized["probe_sha256"],
        "probe_mtime": normalized["probe_mtime"],
        "probe_time_verified": normalized["probe_time_verified"],
        "probe_time_reason": "" if normalized["probe_time_verified"] else "source feedback record has no session_started",
        "work_dir": str(work),
        "claim_digest": str(digest),
        "trusted_record": False,
        "trusted_reason": "converted free text; no harness self-check gate",
        "origin": "conversion",
        # The model that wrote the original claim is not the author of the structured
        # action. Keeping it under source_author prevents a conversion from counting as
        # a model-authored tip, or from reviewing its own source claim.
        "provider": "",
        "model": "",
        "committee": "",
        "effort": "",
        "source_author": {
            "provider": record.get("provider") or source.get("provider") or "",
            "model": record.get("model") or source.get("model") or "",
            "committee": record.get("committee") or source.get("committee") or "",
            "effort": record.get("effort") or "",
        },
        "structured_by": {
            "filled_by": "ai" if str(mode or "manual") == "ai" else "human",
            "confirmed_by": str(by or "cli"),
            "mode": str(mode or "manual"),
        },
        "conversion": {
            "source_seq": int(seq),
            "source_digest": str(digest),
            "source_text": str(source.get("text") or ""),
            "source_section": "对系统的建议",
            "source_provider": record.get("provider") or source.get("provider") or "",
            "source_model": record.get("model") or source.get("model") or "",
            "source_committee": record.get("committee") or source.get("committee") or "",
            "source_session_started": started,
            "confirmed_by": str(by or "cli"),
            "mode": str(mode or "manual"),
            "replaces": existing_key,
            "confirmed_at": time.time(),
        },
    }
    _append_jsonl(run_dir / PROPOSAL_FILE, row)
    return row


def board(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    proposals = _read_jsonl(run_dir / PROPOSAL_FILE)
    decisions = _read_jsonl(run_dir / DECISION_FILE)
    reviews = _read_jsonl(run_dir / "harness_reviews.jsonl")
    overlay = _read_jsonl(run_dir / OVERLAY_FILE)
    reviews = _read_jsonl(run_dir / "harness_reviews.jsonl")
    return {
        "run_dir": str(run_dir),
        "control": overlay_control(run_dir),
        "proposals": proposals,
        "reviews": reviews,
        "decisions": decisions,
        "overlay": overlay,
        "suggestions": free_text_suggestions(run_dir),
    }


def write_template(work: Path) -> Path:
    """Put the structured-proposal schema in the session work directory (not TASK.md)."""
    path = Path(work) / "HARNESS.md"
    path.write_text(TEMPLATE, encoding="utf-8")
    return path
