"""The SSE frame keeps live state out of the index revision, and every change is paired with a bump."""

from __future__ import annotations

import json

from observe.index import Index
from observe import query


def _use_temp_runs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "books.sqlite"))
    monkeypatch.setenv("OBSERVE_TICK_INTERVAL", "0")
    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "3600")
    return runs


def _write_run(root, name, *, task="relu", spent=0.25, budget=2.0):
    directory = root / name
    directory.mkdir(parents=True)
    common = {
        "task": task, "dtype": "fp16", "started_at": 10.0,
        "spent_usd": spent, "budget_usd": budget,
    }
    (directory / "state.json").write_text(json.dumps({**common, "phase": "done", "updated_at": 10.0}), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({**common, "stop_reason": "max_rounds"}), encoding="utf-8")
    return directory


def test_poll_stream_separates_live_revision(tmp_path, monkeypatch):
    """@brief 现场文件（GPU 锁目录等）的版本号不再并入 index_revision。"""
    runs = _use_temp_runs(tmp_path, monkeypatch)
    _write_run(runs, "plain")
    Index(tmp_path / "observe.sqlite").sync_known()

    monkeypatch.setattr(query, "live_revision", lambda: 1)
    first = query.poll_stream()
    monkeypatch.setattr(query, "live_revision", lambda: 2)
    second = query.poll_stream()

    assert second["live_revision"] == 2
    assert second["index_revision"] == first["index_revision"]
    assert second["revision"] == second["index_revision"]
    assert isinstance(second["runs"], list)


def test_changed_run_is_paired_with_revision_bump(tmp_path, monkeypatch):
    """@brief 摄取变化的 run 必须同时记录 changed 并递增 revision（精准刷新的前提）。"""
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _write_run(runs, "plain")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    assert "plain" in index.changed

    before = index.revision()
    document = json.loads((directory / "state.json").read_text())
    document["updated_at"] = 99.0
    (directory / "state.json").write_text(json.dumps(document), encoding="utf-8")
    index._ingest(directory)

    assert index.revision() > before
    assert index.changed[-1] == "plain"


def test_every_changed_append_goes_through_the_paired_helper():
    """@brief 守卫（§5.1 验收③）：``changed`` 的追加只能出现在 ``_mark_changed`` 里，且它同时 bump。

    三处（现在是五处）写点都必须成对；任何新写点绕过 helper 都会让精准刷新静默失效，
    而只测某一个写点的行为测试抓不到。这里直接扫源码。

    @raises AssertionError 出现第二个追加点，或 helper 少了 bump 时抛出。
    """
    import inspect

    import observe.index as index_mod

    source = inspect.getsource(index_mod)
    assert source.count("changed.append(") == 1, "changed 的追加点必须唯一（走 _mark_changed）"
    for pattern in ("self.changed.extend(", "self.changed +=", "self.changed.insert("):
        assert pattern not in source, f"绕开 _mark_changed 的追加：{pattern}"
    body = inspect.getsource(index_mod.Index._mark_changed)
    assert "self.changed.append(run_id)" in body and "self._bump()" in body, body
    # 每个写点都调 helper，不再自己动手。
    for method in ("_refresh_running", "_set_kernel_writer", "_upsert_missing", "_upsert_run", "_mark_missing"):
        assert "_mark_changed(" in inspect.getsource(getattr(index_mod.Index, method)), method


def test_running_run_catch_up_marks_change(tmp_path, monkeypatch):
    """@brief 进行中运行的逐帧追平也会落到 changed 上。"""
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _write_run(runs, "live")
    document = json.loads((directory / "state.json").read_text())
    document["phase"] = "running"
    (directory / "state.json").write_text(json.dumps(document), encoding="utf-8")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()

    index._refresh_running()

    assert "live" in index.changed


def test_turns_projection_bumps_turns_revision(tmp_path, monkeypatch):
    """@brief 回合投影递增 turns_revision，前端据此只重画回合列表。"""
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _write_run(runs, "plain")
    lines = [json.dumps({"event": "agent", "label": "r1.p0", "model": "bench", "cost_usd": 0.01, "t": index}) for index in range(3)]
    (directory / "transcript.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()

    before = index.turns_revision()
    index.ensure_turns(directory)

    assert index.turns_revision() > before
