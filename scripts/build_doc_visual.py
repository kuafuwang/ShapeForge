#!/usr/bin/env python3
"""Build the offline ShapeForge documentation reader.

The Markdown files remain the source of truth. This script only creates the
index and embedded document data used by docs/visual/{index,reader}.html.
No network access, Git commands, browser, or third-party Python package is
required. Run from the repository root:

    uv run python scripts/build_doc_visual.py
    uv run python scripts/build_doc_visual.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "visual" / "data"
SCHEMA = "shapeforge.doc-visual.1"
MERMAID = re.compile(r"^```mermaid\s*$", re.MULTILINE)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def category_for(relative: str) -> str:
    if relative.startswith("docs/教程/"):
        return "tutorials"
    if relative.startswith("docs/架构/"):
        return "architecture"
    if relative.startswith("docs/可信harness/archive/"):
        return "archive"
    if relative.startswith("docs/可信harness/"):
        return "harness"
    if relative.startswith(("docs/paper/", "docs/一个公众号的论文解读/")):
        return "research"
    if relative.startswith(("docs/运维/", "docs/visual/")) or relative == "docs/security.md":
        return "operations"
    return "overview"


def iter_markdown() -> list[Path]:
    """Return documentation inputs without generated pages or prompt fixtures."""
    files: set[Path] = set()
    for name in ("README.md", "README.zh-CN.md", "AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md"):
        if (ROOT / name).is_file():
            files.add(ROOT / name)
    for path in (ROOT / "docs").rglob("*.md"):
        if "visual" not in path.relative_to(ROOT / "docs").parts:
            files.add(path)
    for name in (
        "backends/README.md",
        "targets/README.md",
        "kernels/README.md",
        "proto/README.md",
        "tests/README.md",
        "tests/shared/README.md",
        "tests/backends/README.md",
        "tests/backends/metal/README.md",
        "tests/support/README.md",
        "tests/support/remotegrpc/README.md",
    ):
        if (ROOT / name).is_file():
            files.add(ROOT / name)
    for path in (ROOT / "backends").glob("*/wiki/**/*.md"):
        files.add(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def plain(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[`*_~]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def slugger(headings: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for heading in headings:
        base = re.sub(r"[^\w\u4e00-\u9fff \-]", "", heading.lower()).replace(" ", "-")
        count = seen.get(base, 0)
        seen[base] = count + 1
        result.append(base if count == 0 else f"{base}-{count}")
    return result


def headings(text: str) -> list[dict]:
    titles: list[tuple[int, str]] = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING.match(line)
        if match:
            titles.append((len(match.group(1)), plain(match.group(2))))
    anchors = slugger([title for _depth, title in titles])
    return [
        {"depth": depth, "title": title, "anchor": anchor}
        for (depth, title), anchor in zip(titles, anchors)
    ]


def excerpt(text: str, title: str) -> str:
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line.strip() or line.startswith(("#", "|", ">", "-", "---")):
            continue
        candidate = plain(line)
        if candidate and candidate != title:
            return candidate[:180]
    return ""


def document(path: Path) -> dict:
    relative = path.relative_to(ROOT).as_posix()
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    hs = headings(text)
    title = hs[0]["title"] if hs else path.stem
    return {
        "id": relative,
        "path": relative,
        "title": title,
        "category": category_for(relative),
        "excerpt": excerpt(text, title),
        "headings": hs,
        "mermaid": len(MERMAID.findall(text)),
        "lines": len(text.splitlines()),
        "bytes": len(raw),
        "content": text,
    }


def build_payload() -> dict:
    docs = [document(path) for path in iter_markdown()]
    digest = hashlib.sha256()
    for doc in docs:
        digest.update(doc["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(doc["content"].encode("utf-8"))
        digest.update(b"\0")
    categories: dict[str, int] = {}
    for doc in docs:
        categories[doc["category"]] = categories.get(doc["category"], 0) + 1
    return {
        "manifest": {
            "schema": SCHEMA,
            "document_count": len(docs),
            "mermaid_count": sum(doc["mermaid"] for doc in docs),
            "source_digest": digest.hexdigest(),
            "categories": dict(sorted(categories.items())),
        },
        "docs": docs,
    }


def render(payload: dict) -> tuple[str, str]:
    data = "window.SF_DOC_VISUAL = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n"
    manifest = json.dumps(payload["manifest"], ensure_ascii=False, indent=2) + "\n"
    return data, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated data differs from the checked-in files")
    args = parser.parse_args()
    payload = build_payload()
    data, manifest = render(payload)
    outputs = {OUT / "docs.js": data, OUT / "manifest.json": manifest}
    if args.check:
        stale = [str(path.relative_to(ROOT)) for path, content in outputs.items() if not path.is_file() or path.read_text() != content]
        if stale:
            print("doc visual data is stale: " + ", ".join(stale), file=sys.stderr)
            print("run: uv run python scripts/build_doc_visual.py", file=sys.stderr)
            return 1
        print(f"doc visual data is current: {len(payload['docs'])} documents")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
    print(f"wrote {len(payload['docs'])} documents to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
