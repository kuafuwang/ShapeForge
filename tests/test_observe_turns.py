"""The turn table is a projection of transcript.jsonl. It does not redraw the detail page."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from observe.detail import turns_feed
from observe.index import get_index
from observe.query import feature_grid, run_detail


def _prepare(tmp_path, monkeypatch) -> tuple[Path, object]:
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "missing-carryover.json"))
    return runs, get_index()


def _running(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "phase": "llm", "pid": os.getpid(),
        "started_at": time.time(), "updated_at": time.time(), "round": 1,
    }))


def _line(event: str, label: str, **fields) -> str:
    return json.dumps({"event": event, "label": label, **fields}, ensure_ascii=False) + "\n"


def _rows(index, run_id: str) -> list[dict]:
    out = []
    for item in index.conn.execute("SELECT seq, kind, label, row_json FROM turns WHERE run_id = ? ORDER BY seq", (run_id,)):
        body = json.loads(item["row_json"])
        body["seq"] = item["seq"]
        out.append(body)
    return out


def test_a_running_search_catches_up_without_redrawing_the_detail(tmp_path, monkeypatch):
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = runs / "conv"
    _running(directory)
    transcript = directory / "transcript.jsonl"
    first = _line("chat", "init", model="kimi-k3", seconds=1.5, reply={"content": "hello"})
    transcript.write_bytes(first.encode() + b'{"event":"chat","label":"init2"')
    revision, _changed = index.tick()
    assert index.stats["turn_parses"] == 1
    assert index.conn.execute("SELECT turn_bytes FROM runs WHERE run_id = 'conv'").fetchone()["turn_bytes"] == len(first.encode())
    stamp = run_detail("conv")["detail_stamp"]
    turns = index.turns_revision()
    with transcript.open("ab") as handle:
        handle.write(b',"model":"kimi-k3","reply":{"content":"second"}}\n')
    index.tick()
    assert index.stats["turn_parses"] == 1
    assert index.stats["turn_seek"] == len(first.encode())
    revision_after, changed = index.tick()
    # the line was already consumed by the tick above; this frame finds nothing new
    assert revision_after == revision
    assert changed == []
    assert index.turns_revision() == turns + 1
    assert run_detail("conv")["detail_stamp"] == stamp
    assert run_detail("conv", stamp=stamp) == {"same": True, "id": "conv", "detail_stamp": stamp}
    rows = _rows(index, "conv")
    assert [row["seq"] for row in rows] == [1, 2]
    assert rows[1]["reply"] == "second"

    transcript.write_text(_line("chat", "only", reply={"content": "kept"}))
    index.tick()
    shrunk = _rows(index, "conv")
    assert [row["label"] for row in shrunk] == ["only"]
    assert index.conn.execute("SELECT turn_bytes FROM runs WHERE run_id = 'conv'").fetchone()["turn_bytes"] == transcript.stat().st_size

    before = [(row["seq"], row["label"], row["reply"]) for row in shrunk]
    index.conn.close()
    import observe.index as index_mod

    index_mod._SINGLE = None
    db = Path(os.environ["OBSERVE_DB"])
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True) if suffix else db.unlink(missing_ok=True)
    rebuilt = get_index()
    rebuilt.tick()
    assert [(row["seq"], row["label"], row["reply"]) for row in _rows(rebuilt, "conv")] == before


def test_turns_hide_the_system_prompt_the_kernel_and_secrets(tmp_path, monkeypatch):
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = runs / "conv"
    _running(directory)
    kernel = "K" * 2000
    thinking = "T" * 2000
    (directory / "transcript.jsonl").write_text(_line(
        "chat", "init", model="kimi-k3",
        messages=[
            {"role": "system", "content": "SYSTEM_ONLY_MARKER"},
            {"role": "user", "content": "please sk-SUPERSECRETTOKEN"},
        ],
        reply={
            "content": kernel,
            "reasoning_content": thinking,
            "tool_calls": [{"function": {"name": "quick_check", "arguments": "KERNEL_ARGUMENT_MARKER" + ("Z" * 3000)}}],
        },
        usage={"prompt_tokens": 3, "completion_tokens": 4},
        cost_usd=1.25,
    ))
    index.tick()
    blob = json.dumps(_rows(index, "conv"), ensure_ascii=False)
    assert "SYSTEM_ONLY_MARKER" not in blob
    assert "SUPERSECRETTOKEN" not in blob
    assert "KERNEL_ARGUMENT_MARKER" not in blob
    assert kernel not in blob
    assert thinking not in blob
    row = _rows(index, "conv")[0]
    assert len(row["reply"]) <= 800
    assert len(row["thinking"]) <= 300
    assert row["tools"] == ["quick_check"]
    assert row["sent"].startswith("please sk-")


def test_same_label_merges_until_the_label_comes_back_later(tmp_path, monkeypatch):
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = runs / "conv"
    _running(directory)
    (directory / "transcript.jsonl").write_text("".join([
        _line("chat", "A", reply={"content": "one"}),
        _line("retry", "B", error="timeout talking"),
        _line("chat", "B", reply={"content": "body"}),
        _line("chat", "A", reply={"content": "two"}),
    ]))
    index.tick()
    tail = turns_feed("conv", limit=1)
    assert tail["rows"][0]["label"] == "A" and tail["rows"][0]["min_seq"] == 4 and tail["rows"][0]["reply"] == "two"
    assert tail["cursor"] == 4
    middle = turns_feed("conv", before=tail["cursor"], limit=1)
    assert middle["rows"][0]["label"] == "B" and middle["rows"][0]["min_seq"] == 2
    assert middle["rows"][0]["reply"] == "body"
    assert any("timeout" in item for item in middle["rows"][0]["retries"])
    assert middle["cursor"] == 2
    oldest = turns_feed("conv", before=middle["cursor"], limit=1)
    assert oldest["rows"][0]["label"] == "A" and oldest["rows"][0]["min_seq"] == 1 and oldest["rows"][0]["reply"] == "one"
    assert oldest["cursor"] is None
    assert turns_feed("missing") is None
    assert turns_feed("a/../../etc") is None


def test_an_unchanged_stamp_does_not_read_the_turns_table(tmp_path, monkeypatch):
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = runs / "conv"
    _running(directory)
    (directory / "transcript.jsonl").write_text(_line("chat", "init", reply={"content": "hello"}))
    index.tick()
    feed = turns_feed("conv")
    index.stats["turn_reads"] = 0
    again = turns_feed("conv", stamp=feed["stamp"])
    assert again == {"same": True, "stamp": feed["stamp"]}
    assert index.stats["turn_reads"] == 0


def _finished(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "phase": "done", "started_at": 1, "updated_at": 1,
    }))
    (directory / "result.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.1, "budget_usd": 1,
    }))


def test_a_finished_run_is_projected_when_its_turn_list_opens(tmp_path, monkeypatch):
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = runs / "conv"
    _finished(directory)
    (directory / "transcript.jsonl").write_text(_line("chat", "init", reply={"content": "kept"}))
    index.tick()
    assert index.conn.execute("SELECT COUNT(*) FROM turns WHERE run_id = 'conv'").fetchone()[0] == 0
    assert index.conn.execute("SELECT turn_bytes FROM runs WHERE run_id = 'conv'").fetchone()[0] == 0
    feed = turns_feed("conv")
    assert [row["reply"] for row in feed["rows"]] == ["kept"]
    assert index.conn.execute("SELECT turn_bytes FROM runs WHERE run_id = 'conv'").fetchone()[0] > 0
    index.stats["turn_parses"] = 0
    again = turns_feed("conv", stamp=feed["stamp"])
    assert again["same"] is True
    assert index.stats["turn_parses"] == 0


def test_a_run_that_already_has_turns_still_catches_the_line_written_as_it_finishes(tmp_path, monkeypatch):
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = runs / "conv"
    _running(directory)
    transcript = directory / "transcript.jsonl"
    transcript.write_text(_line("chat", "init", reply={"content": "one"}))
    index.tick()
    (directory / "result.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.1, "budget_usd": 1,
    }))
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(_line("chat", "init2", reply={"content": "two"}))
    index.tick()
    assert [row["reply"] for row in _rows(index, "conv")] == ["one", "two"]


def test_feature_grid_signs_a_source_once_and_then_uses_the_cache(tmp_path, monkeypatch):
    runs, _index = _prepare(tmp_path, monkeypatch)
    directory = runs / "gemm"
    directory.mkdir()
    state = {"task": "gemm", "dtype": "fp16", "phase": "done", "started_at": 1, "updated_at": 1}
    (directory / "state.json").write_text(json.dumps(state))
    (directory / "result.json").write_text(json.dumps({"task": "gemm", "dtype": "fp16", "stop_reason": "max_rounds"}))
    same = "kernel void same() {}\n"
    other = "kernel void other() {}\n"
    for name, source, speed in (("000_slow", same, 0.2), ("001_fast", same, 1.5), ("002_other", other, 0.9)):
        path = directory / "candidates" / name
        path.mkdir(parents=True)
        (path / "kernel.metal").write_text(source)
        (path / "report.json").write_text(json.dumps({"speedup": speed, "status": "ok"}))
    import observe.query as query

    calls = {"n": 0}
    real = query.feature_signature

    def wrapped(source):
        calls["n"] += 1
        return real(source)

    monkeypatch.setattr(query, "feature_signature", wrapped)
    query._GRID_CACHE.clear()
    grid = feature_grid(directory, {"confirm_winner_label": "fast"})
    assert calls["n"] == 2  # three files, two distinct sources
    assert [row["label"] for row in grid] == ["fast"]
    assert grid[0]["speedup"] == 1.5
    feature_grid(directory, {"confirm_winner_label": "fast"})
    assert calls["n"] == 2
    state["updated_at"] = 2
    (directory / "state.json").write_text(json.dumps(state))
    feature_grid(directory, {"confirm_winner_label": "fast"})
    assert calls["n"] == 4


def test_fa_paper_signs_each_distinct_source_once(tmp_path, monkeypatch):
    directory = Path("runs/apple-m4-max/p1/fa_paper")
    candidates = directory / "candidates"
    if not candidates.is_dir():
        pytest.skip("fa_paper is not in this checkout")
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(Path("runs").resolve()))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "missing-carryover.json"))
    distinct = set()
    for path in candidates.iterdir():
        source = path / "kernel.metal"
        if source.is_file():
            distinct.add(hashlib.sha1(source.read_bytes()).hexdigest())
    import observe.query as query

    calls = {"n": 0}
    real = query.feature_signature

    def wrapped(source):
        calls["n"] += 1
        return real(source)

    monkeypatch.setattr(query, "feature_signature", wrapped)
    query._GRID_CACHE.clear()
    detail = run_detail("apple-m4-max/p1/fa_paper")
    assert calls["n"] == len(distinct)
    assert detail["detail_stamp"]
    run_detail("apple-m4-max/p1/fa_paper")
    assert calls["n"] == len(distinct)
    assert run_detail("apple-m4-max/p1/fa_paper", stamp=detail["detail_stamp"])["same"] is True
    assert calls["n"] == len(distinct)
