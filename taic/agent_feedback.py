"""Parse one session's FEEDBACK.md and render peer tips for the next session of the same run.

The search records what the agent wrote. It does not treat that text as a measurement.
Peer tips are recomputed from this run's ``feedback.jsonl`` and never read another run.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

FORMAT = "agent_feedback"
VERSION = 1
FILE_CAP = 16384
RAW_CAP = 4096
TIP_CHARS = 250
TIPS_CHARS = 2000
TIP_KEEP = 8
UNVERIFIED_SPAN = 2  # sessions after the one that first wrote an unverified premise; the third session no longer sees it

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
_UNVERIFIED = re.compile(r"\[未验证(?::[^\]]*)?\]")
_ROUND = re.compile(r"^r(\d+)\.p\d+$")


def read_feedback_text(path: Path) -> tuple[str, bool]:
    """The first 16384 characters, and whether the file was longer."""
    if not path.is_file():
        return "", False
    with path.open(encoding="utf-8", errors="replace") as handle:
        data = handle.read(FILE_CAP + 1)
    return data[:FILE_CAP], len(data) > FILE_CAP


def empty_feedback() -> dict:
    return {
        "format": FORMAT,
        "v": VERSION,
        "kinds": "none",
        "truncated": False,
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
    body = body[:FILE_CAP]
    version = _declared_version(body)
    if version is not None and version != VERSION:
        raw = body[:RAW_CAP]
        return {
            "format": FORMAT,
            "v": version,
            "kinds": "none",
            "truncated": over or len(body) > RAW_CAP,
            "unrecognized_version": True,
            "sections": {},
            "section_names": [],
            "items": [],
            "unverified_premises": 0,
            "raw": raw,
            "chars": len(body),
        }
    kinds = _kinds(body)
    sections, section_cut = _sections(body)
    items = []
    unverified = 0
    for name in SECTIONS:
        chunk = sections.get(name, "")
        if not chunk.strip():
            continue
        for piece in _pieces(chunk):
            item = _item(name, piece, work=work, repo=repo)
            if item["premise"] in ("unverified", "downgraded"):
                unverified += 1
            items.append(item)
    raw = body[:RAW_CAP]
    truncated = over or section_cut or len(body) > RAW_CAP
    names = [name for name in SECTIONS if sections.get(name, "").strip()]
    return {
        "format": FORMAT,
        "v": VERSION,
        "kinds": kinds,
        "truncated": truncated,
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
    """Every complete record. Broken lines are skipped."""
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        row = _load_line(line)
        if row is not None and row.get("format") == FORMAT:
            rows.append(row)
    return rows


def peer_tips_for(run_dir: Path, *, now: int | None = None) -> str:
    """Tips for the session about to start. ``now`` is that session's number.

    Every finished session has a sequence number, including one that left no feedback text.
    An unverified premise is shown for the next two sessions and is gone on the third.
    Restating it does not move the session where it first appeared. A later measured
    version of the same claim becomes an ordinary tip.
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
            current = grouped.get(digest)
            premise = item.get("premise") or ""
            if current is None:
                grouped[digest] = {
                    "first_seq": seq,
                    "seq": seq,
                    "section": item.get("section"),
                    "premise": premise,
                    "text": text,
                    "row": row,
                }
                continue
            current["seq"] = seq
            current["text"] = text
            current["row"] = row
            current["section"] = item.get("section")
            current["premise"] = premise
    unverified = []
    ordinary = []
    for entry in grouped.values():
        premise = entry["premise"]
        if entry["section"] == "被否决的前提" and premise in ("unverified", "downgraded"):
            # Shown on the next two sessions: ages 1 and 2. Age 3 is the third session and is dropped.
            if 0 < now - entry["first_seq"] <= UNVERIFIED_SPAN:
                unverified.append(entry)
            continue
        ordinary.append(entry)
    ordinary.sort(key=lambda entry: entry["seq"], reverse=True)
    chosen = unverified + ordinary[:TIP_KEEP]
    chosen.sort(key=lambda entry: (entry["seq"], entry["text"]))
    lines = []
    for entry in chosen:
        lines.append(_tip_line(entry["row"], entry["text"]))
    while lines and sum(len(line) + 1 for line in lines) > TIPS_CHARS:
        lines.pop(0)
    return "\n".join(lines)


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


def _sections(text: str) -> tuple[dict[str, str], bool]:
    current = None
    chunks: dict[str, list[str]] = {name: [] for name in SECTIONS}
    for line in text.splitlines():
        match = _HEADER.match(line.strip())
        if match:
            current = match.group(1)
            continue
        if current is not None:
            chunks[current].append(line)
    cut = False
    out = {}
    for name, lines in chunks.items():
        body = "\n".join(lines).strip()
        cap = SECTION_CAPS[name]
        if len(body) > cap:
            body = body[:cap]
            cut = True
        out[name] = body
    return out, cut


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
    if section == "被否决的前提":
        shown, premise = _premise(shown, work=work, repo=repo)
    digest = _digest(section, shown)
    return {"section": section, "audience": audience, "text": shown, "digest": digest, "premise": premise}


def _premise(text: str, *, work: Path | None, repo: Path | None) -> tuple[str, str]:
    measured = _MEASURED.search(text)
    if measured:
        probe = measured.group(1).strip()
        if _probe_exists(probe, work=work, repo=repo):
            return text, "verified"
        rewritten = _MEASURED.sub("[未验证: 探针文件不存在]", text, count=1)
        return rewritten, "downgraded"
    return text, "unverified"


def _probe_exists(token: str, *, work: Path | None, repo: Path | None) -> bool:
    if not token or "\x00" in token:
        return False
    raw = Path(token)
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = []
        if work is not None:
            candidates.append(work / raw)
        if repo is not None:
            candidates.append(repo / raw)
    roots = [path for path in (work, repo) if path is not None]
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
    stripped = _UNVERIFIED.sub(" ", _MEASURED.sub(" ", text))
    normalized = _normalize(stripped)
    if not normalized:
        return ""
    payload = f"{section}\n{normalized}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


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
