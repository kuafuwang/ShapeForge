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

- ``surface``: ``tip`` (needs two trusted models) or ``premise`` (needs a fresh local probe).
- ``scope.target_id`` and ``scope.task``: must match this run.
- ``action``: one sentence; no commands, code, paths that redirect writes, or contract changes.
- ``evidence.probe``: relative to this session's work directory and created after the session started.
- ``evidence.measurement``: the number and unit, when there is one.

A free-text suggestion without this block is still reviewed by a person, but it never
reaches the automatic overlay.
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
    "tasks/", "tcenv/", "control.json", "budget_usd", "credentials", ".env",
    "git push", "git commit", "rm -rf", "subprocess", "os.system", "eval(", "exec(",
)


def _reject_never_auto(text: str) -> None:
    lower = text.lower()
    for token in _NEVER_AUTO:
        if token in lower:
            raise ProposalError(f"a P1 overlay may not contain {token!r}")


def proposal_key(*, target_id: str, task: str, surface: str, action: str, dtype: str = "") -> str:
    payload = json.dumps(
        [str(target_id), str(task), str(surface), _normal_action(action), str(dtype or "")],
        ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _fresh_probe(token: str, *, work: Path, started: float) -> tuple[str, str, float]:
    """(absolute path, sha256, mtime) for a file under this session's work directory."""
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
    if mtime < float(started) - 1.0:
        raise ProposalError("evidence.probe predates this session")
    return str(candidate), _sha256(candidate), mtime


def validate_proposal(
    doc: dict,
    *,
    target_id: str,
    task: str,
    dtype: str = "",
    work: Path,
    started: float,
    allow_premise: bool = True,
) -> dict:
    """Return a normalized proposal or raise ProposalError. This does not call a model or GPU."""
    if not isinstance(doc, dict):
        raise ProposalError("the block is not a JSON object")
    surface = str(doc.get("surface") or "").strip()
    if surface not in SURFACES:
        raise ProposalError(f"surface must be one of {list(SURFACES)}")
    if surface not in P1_SURFACES:
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
    _reject_never_auto(action)
    evidence = doc.get("evidence") if isinstance(doc.get("evidence"), dict) else {}
    probe = str(evidence.get("probe") or "").strip()
    measurement = str(evidence.get("measurement") or "").strip()
    probe_path = probe_sha = ""
    probe_mtime = 0.0
    if probe:
        probe_path, probe_sha, probe_mtime = _fresh_probe(probe, work=work, started=started)
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
    started: float,
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
                "claim_digest": next((str(si.get("digest")) for si in system_items if "harness-proposal" in str(si.get("text") or "")), str(system_items[0].get("digest")) if system_items else ""),
            }
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
    """
    if not enabled:
        return {"applied": [], "shadow": [], "reason": "harness_auto is off"}
    run_dir = Path(run_dir)
    proposals = _read_jsonl(run_dir / PROPOSAL_FILE)
    decisions = _read_jsonl(run_dir / DECISION_FILE)
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
        # One author may appear several times; only distinct provider/model/committee count.
        authors = {}
        for row in rows:
            identity = (str(row.get("provider") or ""), str(row.get("model") or ""))
            authors.setdefault(identity, row)
        distinct_models = {identity for identity in authors if all(identity)}
        if surface == "tip":
            if len(distinct_models) < 2:
                continue
            committees = {str(row.get("committee") or "") for row in authors.values()}
            if not {"first", "second"}.issubset(committees):
                continue
            trusted = []
            for identity, row in authors.items():
                ok, _why = _trusted(row, target_id=target_id, task=task, scope="overlay:tip", root=trust_root, prompt_sha1=prompt_sha1)
                if ok:
                    trusted.append(identity)
            if len(trusted) < 2:
                continue
            # A copied probe is not a second source. Two paths with identical bytes are still one measurement.
            if len({str(row.get("probe_sha256") or "") for row in authors.values() if row.get("probe_sha256")}) < 2:
                continue
        else:
            # A measured premise has direct evidence and does not need a second model.
            if not rows[0].get("probe_sha256"):
                continue
        decision = "shadow" if shadow else "apply"
        if (key, decision) in decided:
            continue
        first = rows[0]
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
            "shadow": bool(shadow),
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
        (shadowed if shadow else applied).append(event)
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


def apply_patch(run_dir: Path, *, target_id: str, proposal_key: str, root: Path | None = None, check_only: bool = False, run_tests: bool = False) -> dict:
    """Apply a generated target-experience patch, after ``git apply --check``.

    This function does not commit. A person runs the offline tests and commits separately;
    the returned ``change_ref`` is the patch hash to put on the reviewed claim.
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
    change_ref = f"patch:{meta['sha256']}"
    if meta.get("claim_digest"):
        try:
            from observe.feedback import apply_review

            apply_review(run_dir, int(meta.get("source_seq") or 0), str(meta["claim_digest"]), {
                "status": "done", "verification": "confirmed", "change_ref": change_ref,
                "evidence_source": "code", "evidence": f"applied {meta['path']}", "note": "P2 promotion",
            })
        except (OSError, ValueError, ImportError):
            # The patch is the authoritative change; a missing review row is reported by the caller.
            pass
    _append_jsonl(run_dir / DECISION_FILE, {
        "schema_version": SCHEMA_VERSION, "event": "promotion", "t": time.time(),
        "target_id": target_id, "proposal_key": proposal_key, "change_ref": change_ref,
        "path": meta["path"], "patch_sha256": meta["sha256"],
    })
    return {**meta, "applied": True, "change_ref": change_ref}


def board(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    proposals = _read_jsonl(run_dir / PROPOSAL_FILE)
    decisions = _read_jsonl(run_dir / DECISION_FILE)
    overlay = _read_jsonl(run_dir / OVERLAY_FILE)
    return {
        "run_dir": str(run_dir),
        "control": overlay_control(run_dir),
        "proposals": proposals,
        "decisions": decisions,
        "overlay": overlay,
    }


def write_template(work: Path) -> Path:
    """Put the structured-proposal schema in the session work directory (not TASK.md)."""
    path = Path(work) / "HARNESS.md"
    path.write_text(TEMPLATE, encoding="utf-8")
    return path
