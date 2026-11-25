"""Opt-in real Microsoft Edge E2E for observe.

Set ``TAIC_BROWSER_E2E=1``, install ``playwright-core`` outside the repository and point
``NODE_PATH`` at it.  No Safari setting is changed and no model/GPU call is made.  The test starts
the real ASGI app with every non-GET route (including HEAD) refused, then exercises layout, focus, clicks and
native EventSource reconnect through an actual service restart.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENABLED = os.environ.get("TAIC_BROWSER_E2E") == "1"
EDGE = Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge")

pytestmark = pytest.mark.skipif(not ENABLED, reason="set TAIC_BROWSER_E2E=1 to run the real Edge browser E2E")


def test_real_edge_layout_clicks_focus_and_native_reconnect(tmp_path):
    if not shutil.which("node") or not EDGE.is_file():
        pytest.skip("Microsoft Edge and Node.js are required for the opt-in browser E2E")
    node_path = os.environ.get("NODE_PATH", "/tmp/shapeforge-browser-e2e/node_modules")
    resolved = subprocess.run(
        ["node", "-e", "console.log(require.resolve('playwright-core'))"],
        text=True, capture_output=True, env={**os.environ, "NODE_PATH": node_path}, check=False,
    )
    if resolved.returncode != 0:
        pytest.skip("playwright-core is not installed in the isolated NODE_PATH")
    output = Path(os.environ.get("TAIC_BROWSER_E2E_EVIDENCE_DIR", str(tmp_path / "evidence")))
    completed = subprocess.run(
        ["node", str(ROOT / "tests/support/observe_e2e.js"), "--repo", str(ROOT), "--out", str(output)],
        cwd=ROOT, text=True, capture_output=True, timeout=240,
        env={**os.environ, "NODE_PATH": node_path, "TAIC_BROWSER_E2E": "1"}, check=False,
    )
    evidence_path = output / "evidence.json"
    evidence = json.loads(evidence_path.read_text()) if evidence_path.is_file() else {}
    assert completed.returncode == 0, completed.stdout + completed.stderr + "\n" + json.dumps(evidence, indent=2)
    assert evidence["status"] == "passed"
    assert evidence["safari_used"] is False and evidence["read_only_http_guard"] is True
    assert evidence["layout_widths"] == [1280, 760, 620, 480, 320]
    assert evidence["reconnect"]["unchanged_revision_no_detail_refetch"] is True
    assert evidence["reconnect"]["changed_revision_frames"] == 1
    assert evidence["reconnect"]["changed_revision_detail_fetches"] == 1
    assert evidence["reconnect"]["constructors"] == 1 and evidence["reconnect"]["opens"] >= 2
    assert evidence["console_errors"] == [] and evidence["page_errors"] == []
    assert evidence["mutation_negative_checks"] == ["HEAD reset", "HEAD stop", "POST reset", "POST stop"]
    expected_disconnects = [
        row for row in evidence["expected_network_errors"]
        if row["url"].endswith("/api/v2/stream")
    ]
    assert len(expected_disconnects) == 1
    assert expected_disconnects[0]["text"] == "Failed to load resource: net::ERR_INCOMPLETE_CHUNKED_ENCODING"
    assert evidence["navigation_count_after_reconnect"] == evidence["navigation_count_before_restart"]
    assert evidence["navigations"], "the initial real browser navigation was not recorded"
