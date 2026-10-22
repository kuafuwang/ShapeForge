"""Parse one session's FEEDBACK.md and render peer tips for the next session of the same run.

The search records what the agent wrote. It does not treat that text as a measurement.
Peer tips are recomputed from this run's ``feedback.jsonl`` and never read another run.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

FORMAT = "agent_feedback"
VERSION = 1
# Bumped when in-memory repair of old feedback.jsonl lines changes what the board shows.
# observe/index.py appends this to the run fingerprint so a finished run is read again.
PARSER_VERSION = "fbv2"
FILE_CAP = 16384
RAW_CAP = 4096
TIP_CHARS = 250
TIPS_CHARS = 2000
VERIFIED_TIPS_CHARS = 4000
TIP_KEEP = 8
UNVERIFIED_SPAN = 2  # sessions after the one that first wrote an unverified premise; the third session no longer sees it
_PLACEHOLDERS = frozenset({"无", "没有", "空", "none", "na"})

SECTIONS = (
    "需要帮助",
    "对系统的建议",
    "卡住的原因",
    "给下一棒的避坑提示",
    "被否决的前提",
)
SYSTEM_SECTIONS = ("需要帮助", "对系统的建议")
DIAGNOSTIC_SECTIONS = ("卡住的原因",)
PEER_SECTIONS = ("给下一棒的避坑提示", "被否决的前提")
SECTION_CAPS = {
    "需要帮助": 1200,
    "对系统的建议": 1200,
    "卡住的原因": 800,
    "给下一棒的避坑提示": 500,
    "被否决的前提": 1000,
}
KINDS = ("prompt", "skill", "tool", "rule", "decision", "none")
_HEADER = re.compile(r"^#\s+(" + "|".join(SECTIONS) + r")\s*$")
_KIND = re.compile(r"^kind:\s*(.*)$", re.IGNORECASE)
_VERSION = re.compile(r"^v:\s*(\S+)\s*$", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")
_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE = re.compile(r"\s+")
_MEASURED = re.compile(r"\[已实测:\s*([^|\]]+?)\s*\|\s*([^|\]]+?)\s*\|\s*([^\]]+?)\s*\]")
_MARK = re.compile(r"\[(已实测|未验证|分析否决|推理否决)([^\]]*)\]")
_FILE_TOKEN = re.compile(r"[A-Za-z0-9_./-]+\.(?:py|metal|md|json|sh|txt)")
_WIPED = re.compile(r"^\[未验证:\s*探针文件不存在\]\s*$")
_ROUND = re.compile(r"^r(\d+)\.p\d+$")


def read_feedback_text(path: Path) -> tuple[str, bool]:
    """The last 16384 characters, and whether the file was longer.

    Sessions append. A cut that kept the start would drop the newest lines.
    """
    if not path.is_file():
        return "", False
    data = path.read_text(encoding="utf-8", errors="replace")
    if len(data) > FILE_CAP:
        return data[-FILE_CAP:], True
    return data, False


def empty_feedback() -> dict:
    return {
        "format": FORMAT,
        "v": VERSION,
        "kinds": "none",
        "truncated": False,
        "truncated_sections": [],
        "placeholder_sections": [],
        "unrecognized_version": False,
        "sections": {},
        "section_names": [],
        "items": [],
        "unverified_premises": 0,
        "raw": "",
        "chars": 0,
    }


def parse_feedback(text: str, *, work: Path | None = None, repo: Path | None = None, file_over: bool = False) -> dict:
    """Split one FEEDBACK.md. Unknown versions keep the raw text and no items.

    Section caps apply after the split, so a long system section cannot erase the peer sections.
    """
    body = text or ""
    over = bool(file_over) or len(body) > FILE_CAP
    if len(body) > FILE_CAP:
        body = body[-FILE_CAP:]
    version = _declared_version(body)
    if version is not None and version != VERSION:
        raw = body[:RAW_CAP]
        return {
            "format": FORMAT,
            "v": version,
            "kinds": "none",
            "truncated": over or len(body) > RAW_CAP,
            "truncated_sections": [],
            "placeholder_sections": [],
            "unrecognized_version": True,
            "sections": {},
            "section_names": [],
            "items": [],
            "unverified_premises": 0,
            "raw": raw,
            "chars": len(body),
        }
    kinds = _kinds(body)
    sections, truncated_sections = _sections(body)
    items = []
    unverified = 0
    placeholder_sections = []
    for name in SECTIONS:
        chunk = sections.get(name, "")
        if not chunk.strip():
            continue
        if _is_placeholder(chunk):
            placeholder_sections.append(name)
            continue
        for piece in _pieces(chunk):
            item = _item(name, piece, work=work, repo=repo)
            if item["premise"] in ("unverified", "downgraded"):
                unverified += 1
            items.append(item)
    raw = body[:RAW_CAP]
    truncated = over or bool(truncated_sections) or len(body) > RAW_CAP
    names = [name for name in SECTIONS if sections.get(name, "").strip()]
    return {
        "format": FORMAT,
        "v": VERSION,
        "kinds": kinds,
        "truncated": truncated,
        "truncated_sections": truncated_sections,
        "placeholder_sections": placeholder_sections,
        "unrecognized_version": False,
        "sections": {name: sections[name] for name in names},
        "section_names": names,
        "items": items,
        "unverified_premises": unverified,
        "raw": raw,
        "chars": len(body),
    }


def max_seq(path: Path) -> int:
    """The largest seq in a feedback log. A broken line, including one left in the middle, is skipped."""
    if not path.is_file():
        return 0
    highest = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        row = _load_line(line)
        if row is None:
            continue
        seq = row.get("seq")
        if isinstance(seq, int) and not isinstance(seq, bool):
            highest = max(highest, seq)
    return highest


def load_records(path: Path) -> list[dict]:
    """Every complete record. Broken lines are skipped.

    The file is not rewritten. Empty digests, premise text that a downgrade wiped, and
    missing ``truncated_sections`` are filled in on the copy this function returns.
    """
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        row = _load_line(line)
        if row is not None and row.get("format") == FORMAT:
            rows.append(_repair_record(row))
    return rows


def read_appended_records(path: Path, offset: int) -> tuple[list[tuple[int, dict]], int]:
    """Records in the complete lines at and after ``offset``, with each line's byte position.

    The log is append-only. The returned offset stops after the last newline, so a trailing
    partial line is read on a later call. ``offset`` past the end means the file was replaced.
    """
    if not path.is_file():
        return [], 0
    size = path.stat().st_size
    if offset < 0 or offset > size:
        offset = 0
    if offset == size:
        return [], size
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    last_nl = data.rfind(b"\n")
    if last_nl < 0:
        return [], offset
    rows = []
    pos = 0
    blob = data[: last_nl + 1]
    while pos < len(blob):
        nl = blob.find(b"\n", pos)
        if nl < 0:
            break
        raw = blob[pos:nl]
        line_at = offset + pos
        pos = nl + 1
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            line = raw.decode("utf-8", errors="replace")
        row = _load_line(line)
        if row is not None and row.get("format") == FORMAT:
            rows.append((line_at, _repair_record(row)))
    return rows, offset + pos


def peer_tips_for(run_dir: Path, *, now: int | None = None) -> str:
    """Tips for the session about to start. ``now`` is that session's number.

    Every finished session has a sequence number, including one that left no feedback text.
    An unverified or downgraded premise is shown for the next two sessions and is gone on the third.
    Restating it does not move the session where it first appeared. A digest that was verified
    once stays verified; a later downgraded copy does not replace that text. Verified lines use
    their own character budget and do not take a slot from the eight ordinary tips.
    """
    records = load_records(run_dir / "feedback.jsonl")
    if not records:
        return ""
    numbered = [int(row["seq"]) for row in records if isinstance(row.get("seq"), int)]
    if now is None:
        now = (max(numbered) + 1) if numbered else 1
    grouped: dict[str, dict] = {}
    for row in records:
        seq = row.get("seq")
        if not isinstance(seq, int):
            continue
        for index, item in enumerate(row.get("items") or []):
            if not isinstance(item, dict) or item.get("section") not in PEER_SECTIONS:
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            digest = str(item.get("digest") or "")
            if not digest:
                digest = f"undigested-{seq}-{index}"
            premise = item.get("premise") or ""
            current = grouped.get(digest)
            if current is None:
                current = {
                    "first_seq": seq,
                    "seq": seq,
                    "section": item.get("section"),
                    "premise": premise,
                    "text": text,
                    "row": row,
                    "verified_seq": None,
                    "verified_text": "",
                    "verified_row": None,
                }
                grouped[digest] = current
            else:
                current["seq"] = seq
                current["text"] = text
                current["row"] = row
                current["section"] = item.get("section")
                current["premise"] = premise
            if premise == "verified" and item.get("section") == "被否决的前提":
                current["verified_seq"] = seq
                current["verified_text"] = text
                current["verified_row"] = row
    verified = []
    active = []
    ordinary = []
    for entry in grouped.values():
        if entry["verified_row"] is not None:
            verified.append({
                "seq": entry["verified_seq"],
                "text": entry["verified_text"],
                "row": entry["verified_row"],
                "section": "被否决的前提",
                "premise": "verified",
                "first_seq": entry["first_seq"],
            })
            continue
        if entry["section"] == "被否决的前提" and entry["premise"] in ("unverified", "downgraded"):
            # Shown on the next two sessions: ages 1 and 2. Age 3 is the third session and is dropped.
            if 0 < now - entry["first_seq"] <= UNVERIFIED_SPAN:
                active.append(entry)
            continue
        ordinary.append(entry)
    verified.sort(key=lambda entry: (entry["seq"], entry["text"]))
    active.sort(key=lambda entry: (entry["seq"], entry["text"]))
    ordinary.sort(key=lambda entry: entry["seq"], reverse=True)
    if len(active) > TIP_KEEP:
        active = active[-TIP_KEEP:]
    rest = active + ordinary[: max(0, TIP_KEEP - len(active))]
    verified_lines = _fit_lines(verified, VERIFIED_TIPS_CHARS, drop_oldest=True)
    rest_lines = _fit_rest(rest, len(active))
    return "\n".join(verified_lines + rest_lines)


def round_of(label: str) -> int | None:
    match = _ROUND.fullmatch(label or "")
    if not match:
        return None
    return int(match.group(1))


def _declared_version(text: str) -> int | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _VERSION.match(stripped)
        if match is None:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _kinds(text: str) -> str:
    found = None
    for line in text.splitlines():
        match = _KIND.match(line.strip())
        if match:
            found = match.group(1)
            break
    if found is None:
        return "none"
    kept = []
    for part in re.split(r"[,，\s]+", found.strip()):
        name = part.strip().lower()
        if name in KINDS and name != "none" and name not in kept:
            kept.append(name)
    return ",".join(kept) if kept else "none"


def _sections(text: str) -> tuple[dict[str, str], list[str]]:
    current = None
    chunks: dict[str, list[str]] = {name: [] for name in SECTIONS}
    for line in text.splitlines():
        match = _HEADER.match(line.strip())
        if match:
            current = match.group(1)
            continue
        if current is not None:
            chunks[current].append(line)
    out = {}
    truncated = []
    for name, lines in chunks.items():
        body = "\n".join(lines).strip()
        cap = SECTION_CAPS[name]
        if len(body) > cap:
            # 小节是往下追加的，超长时留下末尾，最新的一条才进下一场。
            body = body[-cap:]
            truncated.append(name)
        out[name] = body
    return out, truncated


def _pieces(body: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] | None = None
    saw_bullet = False
    for line in body.splitlines():
        if line.startswith("- "):
            saw_bullet = True
            if current is not None:
                blocks.append("\n".join(current).strip())
            current = [line[2:]]
            continue
        if current is not None:
            current.append(line)
    if current is not None:
        blocks.append("\n".join(current).strip())
    if saw_bullet:
        return [block for block in blocks if block]
    stripped = body.strip()
    return [stripped] if stripped else []


def _item(section: str, text: str, *, work: Path | None, repo: Path | None) -> dict:
    audience = "peer" if section in PEER_SECTIONS else "diagnostic" if section in DIAGNOSTIC_SECTIONS else "system"
    shown = text.strip()
    premise = ""
    # Digest the sentence the agent wrote. Downgrade rewrites the label afterwards and must not change the digest.
    digest = _digest(section, shown)
    if section == "被否决的前提":
        shown, premise = _premise(shown, work=work, repo=repo)
    return {"section": section, "audience": audience, "text": shown, "digest": digest, "premise": premise}


def _premise(text: str, *, work: Path | None, repo: Path | None) -> tuple[str, str]:
    measured = _MEASURED.search(text)
    if measured:
        probe = measured.group(1).strip()
        if _probe_exists(probe, work=work, repo=repo):
            return text, "verified"
        return _downgrade_measured(text), "downgraded"
    return text, "unverified"


def _downgrade_measured(text: str) -> str:
    """Replace the measured tag and keep the proposition and the number."""

    def repl(match: re.Match) -> str:
        prop = match.group(2).strip()
        number = match.group(3).strip()
        return f"[未验证: 探针文件不存在 | {prop} | {number}]"

    return _MEASURED.sub(repl, text, count=1)


def _probe_roots(work: Path | None, repo: Path | None) -> list[Path]:
    roots = []
    if work is not None:
        roots.append(work)
        if work.parent.name == "agent_work":
            roots.append(work.parent)
    if repo is not None:
        roots.append(repo)
    return roots


def _probe_exists(token: str, *, work: Path | None, repo: Path | None) -> bool:
    """True when any named probe file is readable under the work dir, the repo, or a sibling session."""
    if not token or "\x00" in token:
        return False
    roots = _probe_roots(work, repo)
    candidates = [token.strip()]
    candidates.extend(_FILE_TOKEN.findall(token))
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        if _one_probe(name, roots):
            return True
    return False


def _one_probe(token: str, roots: list[Path]) -> bool:
    if not token or "\x00" in token:
        return False
    raw = Path(token)
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [root / raw for root in roots]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not _inside(resolved, roots):
            continue
        if resolved.is_file():
            return True
    return False


def _inside(path: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root.resolve())
        except (OSError, ValueError):
            continue
        return True
    return False


def _digest(section: str, text: str) -> str:
    """Identity of one claim. Numbers and the probe filename do not change it.

    Text outside the bracket is the claim when it is non-empty, so a measured line and the
    same sentence tagged unverified stay one claim. When the bracket holds the whole sentence,
    only the proposition slot is hashed. Two probes, or two reasons, for one proposition match.
    """
    normalized = _digest_text(text or "")
    if not normalized:
        return ""
    payload = f"{section}\n{normalized}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _digest_text(text: str) -> str:
    match = _MARK.search(text)
    if match is None:
        return _normalize(text)
    outside = _normalize(_MARK.sub(" ", text))
    if outside:
        return outside
    parts = _mark_parts(match.group(2))
    probe = parts[0] if parts else ""
    prop = parts[1] if len(parts) >= 2 else ""
    if len(parts) >= 2 and _normalize(prop):
        return _normalize(prop)
    if len(parts) >= 2:
        return _normalize(f"{probe} {prop}")
    return _normalize(prop or probe)


def _mark_parts(body: str) -> list[str]:
    body = body.strip()
    if body.startswith(":"):
        body = body[1:].strip()
    return [part.strip() for part in body.split("|")] if body else []


def _is_placeholder(text: str) -> bool:
    """A section that is only a filler word. A longer sentence that starts with 没有 is kept."""
    cleaned = _PUNCT.sub("", text or "")
    cleaned = _SPACE.sub("", cleaned).strip().lower()
    return cleaned in _PLACEHOLDERS


def _fit_lines(entries: list[dict], limit: int, *, drop_oldest: bool) -> list[str]:
    lines = [_tip_line(entry["row"], entry["text"]) for entry in entries]
    while lines and sum(len(line) + 1 for line in lines) > limit:
        lines.pop(0 if drop_oldest else -1)
    return lines


def _fit_rest(entries: list[dict], active_count: int) -> list[str]:
    """Unverified lines stay in front. Extra characters drop ordinary tips first, then the oldest unverified line."""
    lines = [_tip_line(entry["row"], entry["text"]) for entry in entries]
    kept_active = min(active_count, len(lines))
    while lines and sum(len(line) + 1 for line in lines) > TIPS_CHARS:
        if len(lines) > kept_active:
            lines.pop()
        else:
            lines.pop(0)
            kept_active = max(0, kept_active - 1)
    return lines


def _repair_record(row: dict) -> dict:
    """Fill fields that older parsers left empty. Does not write the file."""
    sections = row.get("sections") if isinstance(row.get("sections"), dict) else {}
    if row.get("truncated") and "truncated_sections" not in row:
        row["truncated_sections"] = [
            name for name in SECTIONS
            if isinstance(sections.get(name), str) and len(sections[name]) >= SECTION_CAPS[name]
        ]
    items = [item for item in (row.get("items") or []) if isinstance(item, dict)]
    pieces = _pieces(str(sections.get("被否决的前提") or ""))
    premise_items = [item for item in items if item.get("section") == "被否决的前提"]
    for item, piece in zip(premise_items, pieces):
        text = str(item.get("text") or "")
        if _WIPED.match(text.strip()) and _MEASURED.search(piece):
            item["text"] = _downgrade_measured(piece).strip()
        if not str(item.get("digest") or "").strip():
            item["digest"] = _digest("被否决的前提", piece)
    for item in items:
        if str(item.get("digest") or "").strip():
            continue
        text = str(item.get("text") or "").strip()
        if text:
            item["digest"] = _digest(str(item.get("section") or ""), text)
    row["items"] = [item for item in items if not _is_placeholder(str(item.get("text") or ""))]
    return row


def check_feedback(path: Path, work: Path, repo: Path) -> str:
    """How the parser reads one FEEDBACK.md. Reads the file and writes nothing."""
    text, file_over = read_feedback_text(path)
    parsed = parse_feedback(text, work=work, repo=repo, file_over=file_over)
    lines = []
    cuts = parsed.get("truncated_sections") or []
    skipped = parsed.get("placeholder_sections") or []
    lines.append("truncated_sections: " + (", ".join(cuts) if cuts else "-"))
    lines.append("placeholder_sections: " + (", ".join(skipped) if skipped else "-"))
    for index, item in enumerate(parsed.get("items") or [], start=1):
        reason = ""
        if item.get("premise") == "downgraded":
            reason = " reason=探针文件不存在"
        digest = str(item.get("digest") or "")
        lines.append(
            f"[{index}] section={item.get('section') or '-'} premise={item.get('premise') or '-'}"
            f"{reason} digest={digest or '-'} empty_digest={'yes' if not digest else 'no'}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def main(argv: list[str] | None = None) -> int:
    """``python -m taic.agent_feedback check FEEDBACK.md --work DIR``."""
    import argparse

    parser = argparse.ArgumentParser(prog="taic.agent_feedback")
    sub = parser.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check")
    check.add_argument("path")
    check.add_argument("--work", required=True)
    check.add_argument("--repo", default=None)
    args = parser.parse_args(argv)
    repo = Path(args.repo) if args.repo else Path(__file__).resolve().parents[1]
    sys.stdout.write(check_feedback(Path(args.path), Path(args.work), repo))
    return 0


def _normalize(text: str) -> str:
    lowered = text.lower()
    without_numbers = _NUMBER.sub(" ", lowered)
    without_punct = _PUNCT.sub(" ", without_numbers)
    return _SPACE.sub(" ", without_punct).strip()


def _tip_line(row: dict, text: str) -> str:
    label = row.get("label") or "?"
    committee = row.get("committee") or "?"
    model = row.get("model") or "?"
    outcome = row.get("outcome") or row.get("exit") or "?"
    speed = row.get("best_speedup")
    suffix = ""
    if isinstance(speed, (int, float)) and not isinstance(speed, bool):
        suffix = f" {float(speed):.3f}x"
    prefix = f"- [{label} {committee}/{model}, {outcome}{suffix}] "
    room = TIP_CHARS - len(prefix)
    body = text if room >= len(text) else (text[: max(room, 0)] + ("…" if room > 0 else ""))
    line = prefix + body
    return line[:TIP_CHARS]


def _load_line(line: str) -> dict | None:
    if not line.strip():
        return None
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(row, dict):
        return None
    return row


if __name__ == "__main__":
    raise SystemExit(main())
