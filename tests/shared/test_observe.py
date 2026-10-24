"""Dashboard snapshots and the read-only event stream. No GPU and no model calls."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

from observe import query as scan
from ..support.dashboard import _Events, _wait
from observe.publish import publish
from tcenv.evaluate import EvalReport


def test_publish_swallows_write_errors(tmp_path, monkeypatch):
    """@brief 验证发布快照时原子替换写出失败只被吞掉，不影响调用方。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的 monkeypatch，用于把原子替换替换成抛 OSError 的函数。
    """
    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("tcenv.events.os.replace", boom)
    publish(tmp_path / "state.json", {"phase": "plan"})


def test_publish_writes_json(tmp_path):
    """@brief 验证 publish 会按需建父目录并写出可解析的 JSON 快照。

    @param tmp_path pytest 提供的临时目录。
    """
    path = tmp_path / "nested" / "state.json"
    publish(path, {"phase": "evaluate", "n": 1})
    assert json.loads(path.read_text())["phase"] == "evaluate"


def test_from_json_refuses_a_field_this_version_does_not_know():
    """@brief 验证 EvalReport 反序列化遇到本版本不认识的字段时抛 SchemaError 并提示需人工检查。
    """
    from tcenv.schema import SchemaError

    with pytest.raises(SchemaError, match="needs human inspection"):
        EvalReport.from_json({"task": "relu", "dtype": "fp16", "size": "small", "mode": "quick", "status": "ok", "future_field": 1})


def _write_run(root, name, *, phase, pid, stop="", with_result=False):
    """@brief 在 runs 根下造一个 run 目录，写出带 pid 与阶段的状态，并按需写结果文件。

    @param root runs 根目录。
    @param name run 目录名。
    @param phase 当前阶段。
    @param pid 写入 state.json 的进程号，用于判定是否存活。
    @param stop 结果里的停止原因，仅在没有结果文件时被忽略。
    @param with_result 是否同时写出 result.json。
    @return 新建的 run 目录路径。
    """
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
    """@brief 验证历史列表按 pid 存活区分运行中与已结束的 run，且旧版未绑定源码的结果不给出加速比。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的 monkeypatch，用于把 runs 目录指向临时目录。
    """
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(tmp_path / "runs"))
    _write_run(tmp_path / "runs", "live", phase="evaluate", pid=os.getpid())
    _write_run(tmp_path / "runs", "old", phase="done", pid=1, stop="budget_exhausted", with_result=True)
    by_id = {row["id"]: row for row in scan.history()}
    assert by_id["live"]["status"] == "running"
    assert by_id["old"]["status"] == "done"
    assert by_id["old"]["publication"] == "旧版未绑定源码"
    assert by_id["old"]["speedup"] is None


def test_a_dead_gpu_pid_is_not_a_holder(tmp_path, monkeypatch):
    """@brief 验证持有者记录里的 pid 已退出时不会被当成 GPU 占用者。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的 monkeypatch，用于把观察缓存目录指向临时目录。
    """
    cache = tmp_path / "cache"
    monkeypatch.setenv("AI_LOWERLING_OBSERVE_DIR", str(cache))
    proc = subprocess.Popen(["/usr/bin/true"])
    proc.wait()
    holder = cache / "gpu.d" / f"{proc.pid}.json"
    holder.parent.mkdir(parents=True)
    holder.write_text(json.dumps({"pid": proc.pid, "role": "holder", "mode": "ex", "what": "stale"}))
    assert scan.gpu_snapshot()["holders"] == []


def test_run_ids_cannot_climb_out_of_runs(tmp_path, monkeypatch):
    """@brief 验证 run id 解析拒绝 ..、带穿越的相对路径与绝对路径。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的 monkeypatch，用于把 runs 目录指向临时目录。
    """
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    assert scan.resolve_run("..") is None
    assert scan.resolve_run("a/../../etc") is None
    assert scan.resolve_run("/etc") is None


def test_stream_reports_a_phase_change(dashboard):
    """@brief 验证事件流在状态文件被重新发布后推送新 revision，且按 run 查询详情能看到更新后的阶段。

    @param dashboard 指向临时 runs 目录的 dashboard fixture，产出端口与 runs 目录。
    """
    port, runs = dashboard
    _write_run(runs, "gemm_fp16", phase="plan", pid=os.getpid())
    reader = _Events(f"http://127.0.0.1:{port}/api/v2/stream")
    reader.start()
    _wait(reader.events, lambda frame: "revision" in frame)
    state_path = runs / "gemm_fp16" / "state.json"
    doc = json.loads(state_path.read_text())
    doc["phase"] = "evaluate"
    doc["updated_at"] = time.time()
    publish(state_path, doc)
    frame = _wait(reader.events, lambda item: "gemm_fp16" in item.get("runs", []))
    detail = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v2/runs/gemm_fp16", timeout=5).read().decode())
    assert detail["state"]["phase"] == "evaluate"
    assert frame["revision"] >= 1
    reader.close()


def test_stream_rejects_a_path_that_climbs_out(dashboard):
    """@brief 验证按 run id 取详情的接口对带 .. 的穿越路径返回 404。

    @param dashboard 指向临时 runs 目录的 dashboard fixture，产出端口与 runs 目录。
    """
    port, _runs = dashboard
    target = "/api/runs/" + urllib.parse.quote("a/../../etc", safe="")
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(f"http://127.0.0.1:{port}{target}", timeout=2)
    assert caught.value.code == 404
