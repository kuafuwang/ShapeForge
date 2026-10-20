"""Timeline, candidate marks, confirmed speedup and committee handoff. No GPU and no model calls."""

from __future__ import annotations

import hashlib
import json

import pytest

from observe import query as scan
from observe.detail import _chat_lines, _model_notes, _pi_lines_in, kernel_writer

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
    (directory / "confirm_report.json").write_text(json.dumps({
        "status": "ok", "speedup": 1.25,
        "benchmark": {"noisy": False, "candidate": {"cv": 0.01, "reps": 15}, "baselines": {"mlx": {"cv": 0.01, "reps": 15}}, "best_baseline": "mlx"},
    }))
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


def test_model_notes_show_a_timeout_and_an_empty_tool_reply(tmp_path):
    run = tmp_path / "kimi"
    run.mkdir()
    secret = "reasoning secret that must not reach the page"
    (run / "transcript.jsonl").write_text("\n".join([
        json.dumps({"event": "retry", "label": "r2.p1#t0", "attempt": 1, "error": "APITimeoutError: Request timed out.", "sleep_s": 2.0}),
        json.dumps({"event": "chat", "label": "r2.p1#t0", "model": "kimi-k3", "seconds": 874.8, "finish_reason": "tool_calls",
                    "reply": {"role": "assistant", "content": "", "reasoning_content": secret, "tool_calls": [{"function": {"name": "quick_check", "arguments": "{}"}}]}}),
        json.dumps({"event": "retry", "label": "init#t0", "attempt": 1, "error": "APIConnectionError: Connection error."}),
        json.dumps({"event": "error", "label": "r2.p0#t0", "status": 400, "error": "bad request"}),
        json.dumps({"event": "warning", "label": "r2.p0#t0", "message": "reply truncated by max_tokens"}),
        json.dumps({"event": "chat", "label": "r1.plan", "model": "deepseek-flash", "seconds": 40, "finish_reason": "stop",
                    "reply": {"role": "assistant", "content": "```json\n{}\n```", "reasoning_content": secret}}),
    ]) + "\n")
    (run / "state.json").write_text(json.dumps({"history": ["round 2: plan -> aborted (BudgetExhausted)"]}))
    notes = _model_notes(run)
    assert [row["kind"] for row in notes] == ["timeout", "tool", "connection", "http", "truncated", "reply", "stopped"]
    assert notes[5]["label"] == "r1.plan" and notes[5]["detail"] == "正文 14 字" and notes[5]["seconds"] == 40
    assert notes[0]["seconds"] is None
    assert notes[1]["model"] == "kimi-k3" and notes[1]["detail"] == "quick_check"
    blob = json.dumps(notes)
    assert secret not in blob and "arguments" not in blob


def test_model_notes_show_an_hmz_turn_that_wrote_a_kernel_and_one_that_failed(tmp_path):
    run = tmp_path / "pi"
    run.mkdir()
    (run / "transcript.jsonl").write_text("\n".join([
        json.dumps({"event": "agent", "label": "r1.p0", "harness": "pi", "provider": "deepseek", "model": "deepseek-flash", "cost_usd": 0.2, "had_kernel": True}),
        json.dumps({"event": "agent", "label": "r1.p1", "harness": "pi", "provider": "moonshot", "model": "kimi-k3", "had_kernel": False, "error": "hmz exited 1"}),
        json.dumps({"event": "agent", "label": "r1.p2", "harness": "pi", "provider": "deepseek", "model": "deepseek-flash", "cost_usd": 0.1, "had_kernel": False}),
    ]) + "\n")
    notes = _model_notes(run)
    assert [row["kind"] for row in notes] == ["agent", "agent_error", "agent_empty"]
    assert notes[0]["model"] == "pi/deepseek/deepseek-flash" and "$0.200" in notes[0]["detail"]
    assert notes[1]["detail"] == "hmz exited 1"


def test_pi_lines_keep_the_tool_name_and_drop_a_key(tmp_path):
    session = tmp_path / "session.jsonl"
    session.write_text("\n".join([
        json.dumps({"type": "message", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "plan the kernel", "thinkingSignature": "secret-sig"},
            {"type": "toolCall", "name": "bash", "id": "1", "arguments": {"command": "uv run python -m tcenv.cli eval relu kernel.metal --quick"}},
            {"type": "text", "text": "wrote kernel.metal"},
        ]}}),
        json.dumps({"type": "message", "message": {"role": "toolResult", "content": [
            {"type": "text", "text": "api_key sk-secret-value status ok"},
        ]}}),
        json.dumps({"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "the whole task prompt"}]}}),
    ]) + "\n")
    lines = _pi_lines_in(session)
    blob = json.dumps(lines)
    assert [row["who"] for row in lines] == ["思考", "工具", "pi", "结果"]
    assert lines[1]["text"].startswith("bash ")
    assert "secret-sig" not in blob and "sk-secret-value" not in blob and "the whole task prompt" not in blob
    assert "api_key" in lines[3]["text"]


def test_a_long_pi_session_only_keeps_the_tail(tmp_path):
    from observe.detail import _PI_KEEP, _PI_TAIL_BYTES

    session = tmp_path / "session.jsonl"
    old = json.dumps({"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "OLD-LINE-SHOULD-DROP"}]}})
    filler = ("x" * 200 + "\n") * (_PI_TAIL_BYTES // 200 + 10)
    new = json.dumps({"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "NEW-TAIL"}]}})
    session.write_text(old + "\n" + filler + new + "\n")
    lines = _pi_lines_in(session)
    blob = json.dumps(lines)
    assert "OLD-LINE-SHOULD-DROP" not in blob
    assert lines[-1]["text"] == "NEW-TAIL"
    assert len(lines) <= _PI_KEEP


def test_an_epic_shows_the_start_the_spend_and_how_it_ended(tmp_path):
    from observe.detail import _epic_lines_in

    epic = tmp_path / "epic.jsonl"
    epic.write_text("\n".join([
        json.dumps({"event": "began", "at": "t0", "task": "the whole prompt", "agents": [
            {"agent": "assistant", "backend": "pi", "model": "deepseek/deepseek-flash", "effort": "auto", "provider": "", "name": "Golem"},
        ], "budget": {"cost": 2.0}}),
        json.dumps({"event": "opened", "backend": "pi", "session": "30840db1-662e-496d-88a6-132c06c20ee7", "where": "/secret/home"}),
        json.dumps({"event": "usage", "cost": 0.0, "output_tokens": 100, "seconds": 12.2}),
        json.dumps({"event": "ended", "how": "done"}),
    ]) + "\n")
    lines = _epic_lines_in(epic)
    blob = json.dumps(lines)
    assert [row["who"] for row in lines] == ["开始", "打开", "花费", "结束"]
    assert "deepseek/deepseek-flash" in lines[0]["text"] and "$2.00" in lines[0]["text"]
    assert lines[1]["text"].endswith("30840db1")
    assert "the whole prompt" not in blob and "/secret/home" not in blob
    assert lines[3]["text"] == "完成"


def test_kernel_writer_follows_the_recorded_mode_and_old_runs_without_it(tmp_path):
    chat = tmp_path / "chat"
    chat.mkdir()
    (chat / "run_config.json").write_text(json.dumps({"search": {"kernel_agent": ""}}))
    (chat / "transcript.jsonl").write_text(json.dumps({"event": "chat", "label": "init"}) + "\n")
    assert kernel_writer(chat) == "chat"

    explicit = tmp_path / "explicit"
    explicit.mkdir()
    (explicit / "run_config.json").write_text(json.dumps({"search": {"kernel_agent": "pi"}}))
    assert kernel_writer(explicit) == "pi"

    old = tmp_path / "old"
    old.mkdir()
    assert kernel_writer(old) == "chat"
    (old / "transcript.jsonl").write_text(json.dumps({"event": "agent", "label": "init", "had_kernel": True}) + "\n")
    assert kernel_writer(old) == "pi"


def test_chat_tail_shows_the_latest_turn_and_drops_the_system_prompt(tmp_path):
    path = tmp_path / "transcript.jsonl"
    secret = "sk-secret-value"
    kernel = "kernel void conv2d_kernel() {" + ("x" * 400) + "}"
    first = json.dumps({"event": "chat", "label": "old", "messages": [{"role": "user", "content": "FIRST_MARKER old turn"}], "reply": {"content": "old reply"}})
    filler = json.dumps({"event": "chat", "label": "pad", "messages": [{"role": "user", "content": "pad"}], "reply": {"content": "pad"}})
    last = json.dumps({
        "event": "chat",
        "label": "init#t1",
        "messages": [
            {"role": "system", "content": f"system {secret} and the whole prompt"},
            {"role": "tool", "content": "earlier tool result"},
            {"role": "user", "content": "LAST_MARKER please write the kernel"},
        ],
        "reply": {
            "content": "a short answer",
            "reasoning_content": "thinking about banks",
            "tool_calls": [
                {"function": {"name": "quick_check", "arguments": json.dumps({"kernel": kernel})}},
                {"function": {"name": "lookup_msl", "arguments": json.dumps({"topic": "simdgroup"})}},
            ],
        },
    })
    path.write_text(first + "\n" + (filler + "\n") * 8000 + last + "\n")
    lines = _chat_lines(path)
    blob = json.dumps(lines)
    assert secret not in blob and kernel not in blob and "whole prompt" not in blob and "FIRST_MARKER" not in blob
    assert any(row["who"] == "发出" and "LAST_MARKER please write the kernel" in row["text"] for row in lines)
    assert any(row["who"] == "收到" and "a short answer" in row["text"] for row in lines)
    assert any(row["who"] == "思考" and "banks" in row["text"] for row in lines)
    assert any(row["text"] == "quick_check" for row in lines)
    assert any(row["text"] == "lookup_msl simdgroup" for row in lines)


def test_chat_inflight_names_a_stream_without_a_prompt(tmp_path):
    from observe.detail import _chat_inflight

    path = tmp_path / "llm.json"
    path.write_text(json.dumps({"inflight": {"init#t4": {"status": "streaming"}}}))
    assert _chat_inflight(path) == [{"who": "正在接收", "text": "init#t4"}]
