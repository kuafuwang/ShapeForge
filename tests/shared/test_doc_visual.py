"""The offline documentation reader stays current and self-contained. No browser or network required."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VISUAL = ROOT / "docs" / "visual"


def payload() -> dict:
    raw = (VISUAL / "data" / "docs.js").read_text(encoding="utf-8")
    assert raw.startswith("window.SF_DOC_VISUAL = ") and raw.endswith(";\n")
    return json.loads(raw[len("window.SF_DOC_VISUAL = ") : -2])


def test_generated_index_is_current():
    """文档变化后必须重建 data/，否则首页会展示旧内容。"""
    result = subprocess.run(
        [sys.executable, "scripts/build_doc_visual.py", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_manifest_covers_the_tutorials_and_mermaid():
    data = payload()
    manifest = data["manifest"]
    paths = {doc["path"] for doc in data["docs"]}
    assert manifest["schema"] == "shapeforge.doc-visual.1"
    assert manifest["document_count"] == len(data["docs"])
    assert len(manifest["source_digest"]) == 64
    assert manifest["mermaid_count"] >= 140
    for relative in (
        "docs/教程/Agent-新手第一课.md",
        "docs/教程/Kernel-生成新手第一课.md",
        "docs/教程/OpSpec-任务作者专业指南.md",
        "docs/架构/搜索.md",
        "docs/架构/评测.md",
        "README.zh-CN.md",
    ):
        assert relative in paths
    tutorial = next(doc for doc in data["docs"] if doc["path"] == "docs/教程/Agent-新手第一课.md")
    assert tutorial["mermaid"] == 1
    assert any(heading["title"] == "最小改动怎么落地" for heading in tutorial["headings"])


def test_offline_assets_and_licenses_exist():
    for relative in (
        "index.html",
        "reader.html",
        "assets/app.css",
        "assets/common.js",
        "assets/index.js",
        "assets/reader.js",
        "vendor/marked.umd.js",
        "vendor/purify.min.js",
        "vendor/mermaid.min.js",
        "vendor/LICENSE-marked.txt",
        "vendor/LICENSE-dompurify.txt",
        "vendor/LICENSE-mermaid.txt",
        "README.md",
    ):
        path = VISUAL / relative
        assert path.is_file() and path.stat().st_size > 0, relative
    for html in (VISUAL / "index.html", VISUAL / "reader.html"):
        text = html.read_text(encoding="utf-8")
        assert 'src="http' not in text
        assert 'href="http' not in text


def test_the_reader_does_not_fetch_document_content():
    for name in ("common.js", "index.js", "reader.js"):
        text = (VISUAL / "assets" / name).read_text(encoding="utf-8")
        assert "fetch(" not in text
        assert "XMLHttpRequest" not in text
    assert "window.SF_DOC_VISUAL" in (VISUAL / "data" / "docs.js").read_text(encoding="utf-8")[:40]


def test_reader_sidebars_have_independent_accessible_toggles():
    html = (VISUAL / "reader.html").read_text(encoding="utf-8")
    css = (VISUAL / "assets" / "app.css").read_text(encoding="utf-8")
    js = (VISUAL / "assets" / "reader.js").read_text(encoding="utf-8")
    assert 'id="toggle-left"' in html and 'aria-controls="toc-panel"' in html
    assert 'id="toggle-right"' in html and 'aria-controls="side-panel"' in html
    assert ".reader-layout.hide-left" in css
    assert ".reader-layout.hide-right" in css
    assert ".markdown-body { grid-column: 2; }" in css
    assert "sf-doc-reader-sidebars" in js
    assert "sidebarState[side] = !sidebarState[side]" in js


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_javascript_parses():
    for name in ("common.js", "index.js", "reader.js"):
        subprocess.run(["node", "--check", str(VISUAL / "assets" / name)], check=True)
