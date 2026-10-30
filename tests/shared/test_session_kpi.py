"""Session and model read APIs share the model-cost efficiency definitions."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from types import SimpleNamespace

from taic.modelcost import from_events


def test_sessions_and_models_expose_ratio_of_sums_and_null_legacy_kpis(tmp_path, monkeypatch):
    """Raw indexed values stay intact while both APIs derive the same six-decimal KPIs."""
    from observe import query as scan

    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    directory = runs / "hmz"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({"task": "relu", "dtype": "fp16", "phase": "done"}))
    (directory / "result.json").write_text(json.dumps({"task": "relu", "dtype": "fp16"}))
    (directory / "run_config.json").write_text(json.dumps({"loop_protocol": "flame_chase_1"}))
    events = [
        {"event": "agent", "label": "s1", "committee": "first", "provider": "p", "model": "m",
         "cost_usd": 0.2, "runtime": {"tool_calls": 2},
         "tokens": {"input": 1, "output": 2, "cache_read": 3, "cache_write": 4}},
        {"event": "agent", "label": "s2", "committee": "first", "provider": "p", "model": "m",
         "cost_usd": 0.3, "tokens": {"output": 6, "cache_read": 5}},
        {"event": "agent", "label": "legacy", "committee": "first", "provider": "p", "model": "m"},
        {"event": "agent", "label": "bad-cost", "committee": "second", "provider": "q", "model": "n",
         "cost_usd": math.nan, "tokens": {"output": 4, "cache_read": 8}},
        {"event": "chat", "provider": "p", "model": "m", "cost_usd": 99,
         "tokens": {"output": 100, "cache_read": 100}},
        {"event": "provider_retry", "provider": "p", "model": "m", "cost_usd": 99,
         "tokens": {"output": 100, "cache_read": 100}},
    ]
    (directory / "transcript.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n")
    scan.refresh()

    sessions = scan.sessions_page("hmz")["sessions"]
    assert [row["label"] for row in sessions] == ["s1", "s2", "legacy", "bad-cost"]
    s1, s2, legacy, bad_cost = sessions
    assert (s1["cost_usd"], s1["tokens_input"], s1["tokens_output"],
            s1["tokens_cache_read"], s1["tokens_cache_write"], s1["tool_calls"]) == (0.2, 1, 2, 3, 4, 2)
    assert s1["cache_read_per_output"] == 1.5
    assert s1["usd_per_1m_output"] == 100_000
    assert s2["cache_read_per_output"] == 0.833333
    assert s2["usd_per_1m_output"] == 50_000
    assert legacy["cost_usd"] is None
    assert (legacy["tokens_input"], legacy["tokens_output"],
            legacy["tokens_cache_read"], legacy["tokens_cache_write"]) == (0, 0, 0, 0)
    assert legacy["cache_read_per_output"] is None
    assert legacy["usd_per_1m_output"] is None
    assert bad_cost["cache_read_per_output"] == 2.0
    assert bad_cost["usd_per_1m_output"] is None
    assert all("cost_per_session_usd" not in row and "model_turns" not in row for row in sessions)

    models = scan.model_stats()["models"]
    by_model = {row["model"]: row for row in models}
    assert set(by_model) == {"m", "n"}
    m = by_model["m"]
    assert (m["sessions"], m["cost_usd"], m["input_tokens"], m["output_tokens"],
            m["cache_read_tokens"], m["cache_write_tokens"]) == (3, 0.5, 1, 8, 8, 4)
    assert m["cache_read_per_output"] == 1.0
    assert m["usd_per_1m_output"] == 62_500
    assert m["cost_per_session_usd"] == 0.166667  # 分母按 plan 是全部 3 场；成本覆盖单独给 2/3。
    assert m["sessions_with_cost"] == 2
    n = by_model["n"]
    assert n["cost_usd"] == 0.0
    assert n["sessions_with_cost"] == 0
    assert n["cache_read_per_output"] == 2.0
    assert n["usd_per_1m_output"] is None
    assert n["cost_per_session_usd"] is None
    assert all("model_turns" not in row for row in models)

    # result.model_totals and taic brief use from_events, so they must agree with query.models.
    result_totals = {row["model"]: row for row in from_events(events)}
    kpi_names = ("cache_read_per_output", "usd_per_1m_output", "cost_per_session_usd")
    assert {name: m[name] for name in kpi_names} == {name: result_totals["m"][name] for name in kpi_names}
    assert {name: n[name] for name in kpi_names} == {name: result_totals["n"][name] for name in kpi_names}
    assert result_totals["m"]["sessions_with_cost"] == 2
    assert result_totals["n"]["sessions_with_cost"] == 0
    assert scan.model_stats(run_id="hmz")["models"] == models
    assert scan.model_stats(run_id="missing")["models"] == []

    first = scan.sessions_page("hmz")
    assert first["same"] is False and first["stamp"]
    from observe.index import Index
    original_ensure = Index.ensure_sessions

    def must_not_reproject(self, directory):
        raise AssertionError("a matching cheap stamp must not enter ensure_sessions")

    monkeypatch.setattr(Index, "ensure_sessions", must_not_reproject)
    same = scan.sessions_page("hmz", stamp=first["stamp"])
    assert same == {"run_id": "hmz", "limit": 500, "same": True, "stamp": first["stamp"], "sessions": []}
    from observe import app
    request = SimpleNamespace(path_params={"run_id": "hmz"}, query_params={"stamp": first["stamp"], "limit": "500"})
    response = asyncio.run(app.api_v2_sessions(request))
    assert json.loads(response.body)["same"] is True
    monkeypatch.setattr(Index, "ensure_sessions", original_ensure)
    request = SimpleNamespace(query_params={"run": "hmz"})
    response = asyncio.run(app.api_v2_model_stats(request))
    assert json.loads(response.body)["models"] == models

    events.append({"event": "agent", "label": "s3", "committee": "first", "provider": "p", "model": "m",
                   "cost_usd": 0.1, "tokens": {"output": 2, "cache_read": 1}})
    (directory / "transcript.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n")
    scan.refresh()
    changed = scan.sessions_page("hmz", stamp=first["stamp"])
    assert changed["same"] is False and changed["stamp"] != first["stamp"]
    assert [row["label"] for row in changed["sessions"]] == ["s1", "s2", "legacy", "bad-cost", "s3"]
    limited = scan.sessions_page("hmz", limit=1)
    assert len(limited["sessions"]) == 1
    assert limited["stamp"] != first["stamp"]
    changed_limited = scan.sessions_page("hmz", limit=1, stamp=limited["stamp"])
    assert changed_limited["same"] is True  # 内容未再变时，limit=1 也能短路。
    events.append({"event": "agent", "label": "s4", "committee": "first", "provider": "p", "model": "m",
                   "cost_usd": 0.1, "tokens": {"output": 2, "cache_read": 1}})
    (directory / "transcript.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n")
    changed_limited = scan.sessions_page("hmz", limit=1, stamp=limited["stamp"])
    assert changed_limited["same"] is False  # 新增在 limit 之外，COUNT 仍让 stamp 变化。
    assert len(changed_limited["sessions"]) == 1


def test_global_model_stats_catches_up_running_runs(tmp_path, monkeypatch):
    """不带 run 的 models 对少量 running run 也追平 sessions；state 不变只追加 transcript 仍生效。"""
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    directory = runs / "hmz"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({"task": "relu", "phase": "llm", "updated_at": time.time()}))
    (directory / "transcript.jsonl").write_text(json.dumps({"event": "agent", "label": "s1", "model": "m", "tokens": {"output": 1}}) + "\n")
    from observe import query as scan
    scan.refresh()
    with (directory / "transcript.jsonl").open("a") as handle:
        handle.write(json.dumps({"event": "agent", "label": "s2", "model": "m", "tokens": {"output": 2}}) + "\n")
    models = scan.model_stats()["models"]
    assert len(models) == 1 and models[0]["output_tokens"] == 3


def test_appended_sessions_do_not_rebuild_the_whole_projection(tmp_path, monkeypatch):
    """首次全量后禁止 _upsert_sessions；追加仍必须走 sessions_bytes 的增量路径。"""
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    directory = runs / "hmz"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({"task": "relu", "phase": "done"}))
    (directory / "transcript.jsonl").write_text(json.dumps({"event": "agent", "label": "s1", "tokens": {"output": 1}}) + "\n")
    from observe import query as scan
    from observe.index import Index
    scan.refresh()
    assert len(scan.sessions_page("hmz")["sessions"]) == 1

    def forbidden(self, directory, run_id, config):
        raise AssertionError("append must not rebuild all sessions")

    monkeypatch.setattr(Index, "_upsert_sessions", forbidden)
    with (directory / "transcript.jsonl").open("a") as handle:
        handle.write(json.dumps({"event": "agent", "label": "s2", "tokens": {"output": 2}}) + "\n")
    rows = scan.sessions_page("hmz")["sessions"]
    assert [row["label"] for row in rows] == ["s1", "s2"]


def test_equal_length_transcript_rewrite_rebuilds_sessions(tmp_path, monkeypatch):
    """等长改写不是追加；源指纹变化后必须重建，不能永远显示旧 label。"""
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    directory = runs / "hmz"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({"task": "relu", "phase": "done"}))
    (directory / "transcript.jsonl").write_text(json.dumps({"event": "agent", "label": "s1", "tokens": {"output": 1}}) + "\n")
    from observe import query as scan
    scan.refresh()
    assert scan.sessions_page("hmz")["sessions"][0]["label"] == "s1"
    path = directory / "transcript.jsonl"
    path.write_text(path.read_text().replace('"s1"', '"x1"'))
    assert scan.sessions_page("hmz")["sessions"][0]["label"] == "x1"


def test_shorter_transcript_rebuilds_sessions(tmp_path, monkeypatch):
    """offset > size 时必须全量重建，不能把删除的前缀留在索引里。"""
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    directory = runs / "hmz"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({"task": "relu", "phase": "done"}))
    rows = [{"event": "agent", "label": label, "tokens": {"output": 1}} for label in ("s1", "s2")]
    (directory / "transcript.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    from observe import query as scan
    scan.refresh()
    assert len(scan.sessions_page("hmz")["sessions"]) == 2
    (directory / "transcript.jsonl").write_text(json.dumps(rows[0]) + "\n")
    assert [row["label"] for row in scan.sessions_page("hmz")["sessions"]] == ["s1"]


def test_large_legacy_transcript_is_not_projected(tmp_path, monkeypatch):
    """非 flame_chase_1 的 >1MB transcript 清空投影并跳过；重复读取 stamp 稳定。"""
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    directory = runs / "hmz"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({"task": "relu", "phase": "done"}))
    (directory / "run_config.json").write_text(json.dumps({"loop_protocol": "legacy"}))
    (directory / "transcript.jsonl").write_text(json.dumps({"event": "agent", "label": "big", "pad": "x" * 1_100_000}) + "\n")
    from observe import query as scan
    scan.refresh()
    first = scan.sessions_page("hmz")
    assert first["sessions"] == []
    assert scan.sessions_page("hmz", stamp=first["stamp"])["same"] is True


def test_sessions_page_cannot_deadlock_the_write_lock(tmp_path, monkeypatch):
    """ensure_sessions 若在 borrow() 内被误挪，会在同一个 _WRITE 上自死锁；用 join 超时把挂死变成失败。"""
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    directory = runs / "hmz"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({"task": "relu", "phase": "done"}))
    (directory / "transcript.jsonl").write_text(json.dumps({"event": "agent", "label": "s1", "tokens": {"output": 1}}) + "\n")
    from observe import query as scan
    scan.refresh()
    finished = threading.Event()

    def read_sessions():
        scan.sessions_page("hmz")
        finished.set()

    worker = threading.Thread(target=read_sessions, daemon=True)
    worker.start()
    worker.join(timeout=10)
    assert finished.is_set(), "sessions_page deadlocked on the observe write lock"
