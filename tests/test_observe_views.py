"""Timeline, candidate marks, confirmed speedup and committee handoff. No GPU and no model calls."""

from __future__ import annotations

import hashlib
import json

import pytest

from observe import query as scan

SOURCE = "kernel void gemm() {}\n"
T0 = 1_700_000_000.0


def _digest() -> str:
    return hashlib.sha256(SOURCE.encode()).hexdigest()


def test_scan_reads_timeline_marks_confirmed_point_and_handoff(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    cache = tmp_path / "cache"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("AI_LOWERLING_OBSERVE_DIR", str(cache))
    directory = runs / "gemm_fp16"
    directory.mkdir(parents=True)
    digest = _digest()
    (directory / "best.metal").write_text(SOURCE)
    winner = directory / "candidates" / "000_r1.p0"
    winner.mkdir(parents=True)
    (winner / "kernel.metal").write_text(SOURCE)
    (directory / "confirm_report.json").write_text(json.dumps({"status": "ok", "speedup": 1.25}))
    trace = [
        {"label": "init", "status": "ok", "speedup": 1.0, "best_speedup": 1.0, "cost_usd": 0.1, "within_noise": True, "differential_recheck": None},
        {"label": "r1.p0", "status": "ok", "speedup": 1.4, "best_speedup": 1.4, "cost_usd": 0.3, "within_noise": False, "differential_recheck": "incorrect"},
    ]
    (directory / "state.json").write_text(json.dumps({
        "pid": 1, "updated_at": T0 + 30, "started_at": T0, "phase": "done",
        "task": "gemm", "dtype": "fp16", "model": "model-b", "committee": "second", "round": 2,
        "spent_usd": 0.4, "best_speedup": 1.4, "trace": trace,
    }))
    (directory / "result.json").write_text(json.dumps({
        "task": "gemm", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.5,
        "best_speedup": 9.9, "confirmed_speedup": 1.25, "confirmed_status": "ok",
        "confirm_winner_label": "r1.p0", "confirmation_source_sha256": digest, "trace": trace,
    }))
    (directory / "run_config.json").write_text(json.dumps({"committees": [
        {"name": "first", "members": [{"provider": "a", "model": "model-a"}]},
        {"name": "second", "members": [{"provider": "b", "model": "model-b"}]},
    ]}))
    secret = "SECRET prompt sk-abcdefghijklmnop"
    (directory / "transcript.jsonl").write_text(
        json.dumps({
            "event": "chat", "label": "r2.plan", "model": "model-b", "seconds": 2.0, "t": T0 + 12,
            "messages": [{"role": "user", "content": secret}], "reply": {"content": "do not show"},
        }) + "\n"
        + json.dumps({"event": "chat", "label": "init#t0", "model": "model-a", "seconds": 4.0, "messages": [{"role": "user", "content": secret}]}) + "\n"
        + json.dumps({"event": "warning", "label": "r2.plan", "message": secret}) + "\n"
    )
    (directory / "ledger.json").write_text(json.dumps({"requests": [{"t": 4.5, "label": "init#t0", "cost_usd": 0.01}]}))
    (directory / "eval_progress.json").write_text(json.dumps({
        "stage": "bench", "entered_at": T0 + 18, "updated_at": T0 + 20, "finished": True, "task": "gemm", "message": secret,
    }))
    gpu = cache / "observe" / "gpu_events.jsonl"
    gpu.parent.mkdir(parents=True)
    gpu.write_text(
        json.dumps({"t": T0 + 5, "pid": 42, "kind": "wait", "mode": "ex", "what": "benchmark of gemm fp16 paper"}) + "\n"
        + json.dumps({"t": T0 + 6, "pid": 42, "kind": "acquire", "mode": "ex", "what": "benchmark of gemm fp16 paper"}) + "\n"
        + json.dumps({"t": T0 + 5.5, "pid": 7, "kind": "wait", "mode": "ex", "what": "benchmark of relu fp16 paper"}) + "\n"
        + json.dumps({"t": T0 + 5.6, "pid": 7, "kind": "acquire", "mode": "ex", "what": "benchmark of relu fp16 paper"}) + "\n"
    )

    detail = scan.run_detail("gemm_fp16")
    lanes = detail["timeline"]["lanes"]
    assert [lane["label"] for lane in lanes] == ["问模型", "等锁", "评测"]
    by_label = {lane["label"]: lane["spans"] for lane in lanes}
    model = {span["label"]: span for span in by_label["问模型"]}
    assert model["r2.plan"]["start"] == pytest.approx(T0 + 10)
    assert model["r2.plan"]["end"] == pytest.approx(T0 + 12)
    assert model["init#t0"]["start"] == pytest.approx(T0 + 0.5)
    assert model["init#t0"]["end"] == pytest.approx(T0 + 4.5)
    assert len(by_label["等锁"]) == 1
    assert by_label["等锁"][0]["start"] == pytest.approx(T0 + 5)
    assert by_label["等锁"][0]["end"] == pytest.approx(T0 + 6)
    assert len(by_label["评测"]) == 1
    assert by_label["评测"][0]["start"] == pytest.approx(T0 + 18)
    assert by_label["评测"][0]["end"] == pytest.approx(T0 + 20)
    shown = json.dumps({"timeline": detail["timeline"], "handoff": detail["handoff"], "marks": detail["candidate_marks"]}, ensure_ascii=False)
    assert secret not in shown and "do not show" not in shown

    marks = {row["label"]: row["marks"] for row in detail["candidate_marks"]}
    assert marks["init"] == ["噪声内没换种子"]
    assert marks["r1.p0"] == ["正确性复查没过"]

    assert detail["confirmed_point"] == {"cost_usd": 0.5, "speedup": 1.25, "label": "确认"}
    assert detail["confirmed_point"]["speedup"] != detail["result"]["best_speedup"]
    assert detail["publication"] == "confirmed"

    rounds = {(row["round"], row["committee"], row["model"], row["label"]) for row in detail["handoff"]["rounds"]}
    assert detail["handoff"]["recorded"] is True
    assert (0, "first", "model-a", "初稿") in rounds
    assert (1, "first", "model-a", "第 1 轮") in rounds
    assert (2, "second", "model-b", "第 2 轮") in rounds
    assert {row["committee"] for row in detail["handoff"]["rounds"]} == {"first", "second"}

    old = runs / "old"
    old.mkdir()
    (old / "state.json").write_text(json.dumps({
        "pid": 1, "updated_at": 20, "started_at": 10, "phase": "done", "task": "rope", "dtype": "fp16",
        "model": "old-model", "round": 4, "trace": [],
    }))
    (old / "run_config.json").write_text(json.dumps({"committees": [
        {"name": "first", "members": [{"provider": "a", "model": "should-not-appear"}]},
        {"name": "second", "members": [{"provider": "b", "model": "should-not-appear"}]},
    ]}))
    (old / "transcript.jsonl").write_text(json.dumps({"event": "chat", "label": "init", "seconds": 1.0, "t": 1_600_000_000.0}) + "\n")
    old_detail = scan.run_detail("old")
    old_lanes = {lane["label"]: lane["spans"] for lane in old_detail["timeline"]["lanes"]}
    assert list(old_lanes) == ["问模型", "等锁", "评测"]
    assert old_lanes["问模型"] and old_lanes["等锁"] == [] and old_lanes["评测"] == []
    assert old_detail["handoff"] == {"recorded": False, "note": "这次搜索没有委员会记录", "rounds": []}
    assert old_detail["confirmed_point"] is None

    unbound = runs / "unbound"
    unbound.mkdir()
    (unbound / "state.json").write_text(json.dumps({
        "pid": 1, "updated_at": T0, "started_at": T0, "phase": "done", "task": "gemm", "dtype": "fp16",
        "committee": "first", "model": "model-a", "round": 0, "trace": [{"label": "init", "best_speedup": 9.9, "cost_usd": 0.2}],
    }))
    (unbound / "result.json").write_text(json.dumps({
        "task": "gemm", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.2,
        "best_speedup": 9.9, "confirmed_speedup": 1.25, "confirmed_status": "ok",
        "confirm_winner_label": "init", "confirmation_source_sha256": digest,
    }))
    assert scan.run_detail("unbound")["confirmed_point"] is None
