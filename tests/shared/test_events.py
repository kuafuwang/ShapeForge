"""Events stay inside the standard library. The dashboard is a subscriber, not a dependency of judging."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import tcenv.events as events


def test_emit_writes_nothing_until_the_file_log_is_subscribed(tmp_path):
    saved = list(events._listeners)
    events.reset()
    try:
        target = tmp_path / "state.json"
        events.emit("snapshot", path=target, doc={"phase": "plan"})
        assert not target.exists()
        events.install_file_log()
        events.emit("snapshot", path=target, doc={"phase": "plan"})
        assert json.loads(target.read_text())["phase"] == "plan"
    finally:
        events.reset()
        for listener in saved:
            events.subscribe(listener)
        events.install_file_log()


def test_a_gpu_lock_note_is_written(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_LOWERLING_OBSERVE_DIR", str(tmp_path))
    saved = list(events._listeners)
    events.reset()
    try:
        events.note_gpu("acquire", "ex", "benchmark")
        holder = tmp_path / "gpu.d" / f"{os.getpid()}.json"
        assert json.loads(holder.read_text())["role"] == "holder"
        lines = (tmp_path / "observe" / "gpu_events.jsonl").read_text().splitlines()
        assert json.loads(lines[-1])["kind"] == "acquire"
    finally:
        events.note_gpu("release", "ex", "benchmark")
        events.reset()
        for listener in saved:
            events.subscribe(listener)
        events.install_file_log()


def test_core_imports_do_not_load_the_dashboard_stack(tmp_path):
    """A fresh interpreter blocks starlette and uvicorn, then imports judging, policy and config."""
    root = Path(__file__).resolve().parents[2]
    code = r"""
import os, sys
os.environ["AI_LOWERLING_OBSERVE_DIR"] = sys.argv[1]

class Block:
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".", 1)[0] in ("starlette", "uvicorn"):
            raise ImportError("import hook blocked " + fullname)
        return None

sys.meta_path.insert(0, Block())
import tcenv.evaluate
import tcenv.verify
import tcenv.schema
import taic.policy
import taic.llm
import tcenv.gpulock
tcenv.evaluate._note_stage(stage="screen", status="ok", task="relu")
tcenv.gpulock._gpu_note("wait", "sh", "decouple")
from tcenv.events import emit
emit("snapshot", path=os.path.join(sys.argv[1], "unused.json"), doc={"n": 1})
bad = [
    name for name in sys.modules
    if name == "observe" or name.startswith("observe.")
    or name == "starlette" or name.startswith("starlette.")
    or name == "uvicorn" or name.startswith("uvicorn.")
]
if bad:
    sys.stderr.write("ui packages in sys.modules: " + ", ".join(sorted(bad)) + "\n")
    raise SystemExit(1)
assert taic.llm.DEFAULT_PRESET
assert callable(taic.policy.relation_to_best)
assert tcenv.verify.UNSUPPORTED == "unsupported"
print("ok")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["AI_LOWERLING_OBSERVE_DIR"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert proc.stdout.strip() == "ok"
