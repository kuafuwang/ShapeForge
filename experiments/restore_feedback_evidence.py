"""Restore probe files named by feedback from the scratch archives.

``prune_work`` used to archive every probe and ``v/`` file. A measured premise names that file;
after the move the premise still exists in the index, but the file does not. This script reads
the session's FEEDBACK.md/NOTES.md, finds each named file in ``scratch/<label>*.tar.gz`` and
puts it back under the work directory. New sessions keep the evidence in place.
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path

TOKEN = re.compile(r"[A-Za-z0-9_./-]+\.(?:py|metal|md|json|sh|txt)")
SKIP_TOP = {"tcenv", "taic", "tasks", "observe", "experiments", "docs", "targets", "backends", "runs"}


def _tokens(work: Path) -> list[str]:
    text = ""
    for name in ("FEEDBACK.md", "NOTES.md"):
        try:
            text += (work / name).read_text(encoding="utf-8", errors="replace") + "\n"
        except OSError:
            continue
    out = []
    for token in TOKEN.findall(text):
        clean = token.strip()
        if clean.startswith("PWD/"):
            clean = clean[4:]
        if not clean or clean.startswith("/"):
            continue
        if Path(clean).parts and Path(clean).parts[0] in SKIP_TOP:
            continue
        out.append(clean)
    return out


def _restore(run: Path, work: Path, token: str, apply: bool) -> dict | None:
    # r11.p0/bench2.py belongs to a sibling session.
    parts = Path(token).parts
    target_work = work
    member_name = token
    if len(parts) > 1 and re.fullmatch(r"r\d+\.p\d+", parts[0]):
        target_work = run / "agent_work" / parts[0]
        member_name = "/".join(parts[1:])
    destination = target_work / member_name
    if destination.is_file():
        return None
    candidates = sorted((run / "scratch").glob("*.tar.gz"))
    for archive in candidates:
        try:
            with tarfile.open(archive, "r:gz") as tar:
                member = next((m for m in tar.getmembers() if m.name.lstrip("./") == member_name or m.name.lstrip("./").endswith("/" + member_name)), None)
                if member is None or not member.isfile():
                    continue
                if apply:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    destination.write_bytes(extracted.read())
                return {"run": str(run), "token": token, "archive": str(archive), "dest": str(destination), "applied": apply}
        except (OSError, tarfile.TarError):
            continue
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.restore_feedback_evidence")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.runs)
    rows = []
    for work in sorted(root.rglob("*/agent_work/*")):
        if not work.is_dir():
            continue
        run = work.parent.parent
        for token in _tokens(work):
            found = _restore(run, work, token, args.apply)
            if found:
                rows.append(found)
    print(json.dumps({"mode": "apply" if args.apply else "dry-run", "restored": len(rows), "rows": rows}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
