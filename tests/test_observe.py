"""Dashboard snapshots and the read-only event stream. No GPU and no model calls."""

from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

from observe import scan
from observe.publish import publish
from tcenv.evaluate import EvalReport


def test_publish_swallows_write_errors(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("observe.publish.os.replace", boom)
    publish(tmp_path / "state.json", {"phase": "plan"})


def test_publish_writes_json(tmp_path):
    path = tmp_path / "nested" / "state.json"
    publish(path, {"phase": "evaluate", "n": 1})
    assert json.loads(path.read_text())["phase"] == "evaluate"


def test_from_json_refuses_a_field_this_version_does_not_know():
    from tcenv.schema import SchemaError

    with pytest.raises(SchemaError, match="needs human inspection"):
        EvalReport.from_json({"task": "relu", "dtype": "fp16", "size": "small", "mode": "quick", "status": "ok", "future_field": 1})


def _write_run(root, name, *, phase, pid, stop="", with_result=False):
    directory = root / name
    directory.mkdir(parents=True)
    state = {
        "pid": pid, "updated_at": time.time(), "started_at": time.time(), "phase": phase,
        "task": "gemm", "dtype": "fp16", "spent_usd": 0.2, "best_speedup": 0.9, "trace": [],
    }
    (directory / "state.json").write_text(json.dumps(state))
    if with_result:
        (directory / "result.json").write_text(json.dumps({
            "task": "gemm", "dtype": "fp16", "stop_reason": stop or "max_rounds",
            "confirmed_speedup": 1.1, "spent_usd": 0.4, "trace": [],
        }))
    return directory


def test_history_separates_a_live_run_from_a_finished_one(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(tmp_path / "runs"))
    _write_run(tmp_path / "runs", "live", phase="evaluate", pid=os.getpid())
    _write_run(tmp_path / "runs", "old", phase="done", pid=1, stop="budget_exhausted", with_result=True)
    by_id = {row["id"]: row for row in scan.history()}
    assert by_id["live"]["status"] == "running"
    assert by_id["old"]["status"] == "done"
    assert by_id["old"]["publication"] == "旧版未绑定源码"
    assert by_id["old"]["speedup"] is None


def test_a_dead_gpu_pid_is_not_a_holder(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("AI_LOWERLING_OBSERVE_DIR", str(cache))
    proc = subprocess.Popen(["/usr/bin/true"])
    proc.wait()
    holder = cache / "gpu.d" / f"{proc.pid}.json"
    holder.parent.mkdir(parents=True)
    holder.write_text(json.dumps({"pid": proc.pid, "role": "holder", "mode": "ex", "what": "stale"}))
    assert scan.gpu_snapshot()["holders"] == []


def test_run_ids_cannot_climb_out_of_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    assert scan.resolve_run("..") is None
    assert scan.resolve_run("a/../../etc") is None
    assert scan.resolve_run("/etc") is None


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class _Events(threading.Thread):
    def __init__(self, url: str) -> None:
        super().__init__(daemon=True)
        self.url = url
        self.events: queue.Queue = queue.Queue()
        self._resp = None

    def run(self) -> None:
        self._resp = urllib.request.urlopen(self.url, timeout=5)
        try:
            while True:
                line = self._resp.readline()
                if not line:
                    break
                if line.startswith(b"data: "):
                    self.events.put(json.loads(line[6:]))
        except Exception:
            return

    def close(self) -> None:
        if self._resp is not None:
            self._resp.close()


def _wait(events: queue.Queue, predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            item = events.get(timeout=0.2)
        except queue.Empty:
            continue
        if predicate(item):
            return item
    raise AssertionError("event stream did not reach the expected state")


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    import uvicorn

    from observe.app import app

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("AI_LOWERLING_OBSERVE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "books.sqlite"))
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.02)
    else:
        raise RuntimeError("dashboard did not start")
    yield port, runs
    server.should_exit = True
    thread.join(timeout=2)


def test_stream_reports_a_phase_change(dashboard):
    port, runs = dashboard
    _write_run(runs, "gemm_fp16", phase="plan", pid=os.getpid())
    reader = _Events(f"http://127.0.0.1:{port}/api/stream")
    reader.start()
    _wait(reader.events, lambda frame: frame["run"] and frame["run"]["state"]["phase"] == "plan")
    state_path = runs / "gemm_fp16" / "state.json"
    doc = json.loads(state_path.read_text())
    doc["phase"] = "evaluate"
    doc["updated_at"] = time.time()
    publish(state_path, doc)
    frame = _wait(reader.events, lambda item: item["run"] and item["run"]["state"]["phase"] == "evaluate")
    assert frame["run"]["state"]["phase"] == "evaluate"
    reader.close()


def test_stream_rejects_a_path_that_climbs_out(dashboard):
    port, _runs = dashboard
    target = "/api/runs/" + urllib.parse.quote("a/../../etc", safe="")
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(f"http://127.0.0.1:{port}{target}", timeout=2)
    assert caught.value.code == 404
