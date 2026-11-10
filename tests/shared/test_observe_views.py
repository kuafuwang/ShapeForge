"""Timeline, candidate marks, confirmed speedup and committee handoff. No GPU and no model calls."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from tcenv.schema import PROTOCOL_VERSION

from observe import query as scan
from observe import detail
from observe.detail import _chat_lines, _model_notes, _pi_lines_in, kernel_writer

SOURCE = "kernel void gemm() {}\n"
T0 = 1_700_000_000.0


def _digest() -> str:
    """@brief 算出用例源码 SOURCE 的 sha256，用于把确认结果绑定到被重测的内核。

    @return SOURCE 的 sha256 十六进制摘要。
    """
    return hashlib.sha256(SOURCE.encode()).hexdigest()


def test_scan_reads_timeline_marks_confirmed_point_and_handoff(tmp_path, monkeypatch):
    """@brief 验证扫描器从运行目录读出时间线三条泳道、候选标记、确认点、发布状态与委员会交接，并隐去提示词里的密钥；老运行没有委员会记录时给出说明。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来把运行目录与观察缓存指到临时路径。
    """
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
    from experiments.common import synthetic_confirmation_documents

    result, report = synthetic_confirmation_documents(
        task="gemm", dtype="fp16", size="small", target_id="apple-m4-max",
        speedup=1.25, label="r1.p0", kernel_sha256=digest,
    )
    (directory / "confirm_report.json").write_text(json.dumps(report))
    trace = [
        {"label": "init", "status": "ok", "speedup": 1.0, "best_speedup": 1.0, "cost_usd": 0.1, "within_noise": True, "differential_recheck": None},
        {"label": "r1.p0", "status": "ok", "speedup": 1.4, "best_speedup": 1.4, "cost_usd": 0.3, "within_noise": False, "differential_recheck": "incorrect"},
    ]
    (directory / "state.json").write_text(json.dumps({
        "pid": 1, "updated_at": T0 + 30, "started_at": T0, "phase": "done",
        "task": "gemm", "dtype": "fp16", "model": "model-b", "committee": "second", "round": 2,
        "spent_usd": 0.4, "best_speedup": 1.4, "trace": trace,
    }))
    result.update({
        "stop_reason": "max_rounds", "spent_usd": 0.5, "best_speedup": 9.9, "trace": trace,
        "winner_label": "r1.p0", "winner_provider": "b", "winner_model": "model-b", "winner_committee": "second", "winner_origin": "session",
        "model_totals": [{"provider": "b", "model": "model-b", "committee": "second", "sessions": 1, "cost_usd": 0.3,
                          "input_tokens": 5, "output_tokens": 10, "cache_read_tokens": 20, "cache_write_tokens": 0}],
    })
    (directory / "result.json").write_text(json.dumps(result))
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
    assert detail["timing"] == {
        "started_at": T0,
        "ended_at": T0 + 30,
        "elapsed_s": 30,
        "running": False,
        "ended_at_estimated": False,
    }
    assert detail["winner"] == {"label": "r1.p0", "provider": "b", "model": "model-b", "committee": "second", "origin": "session"}
    assert detail["model_totals"][0]["model"] == "model-b" and detail["model_totals"][0]["cost_usd"] == 0.3
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

    point = detail["confirmed_point"]
    assert point["cost_usd"] == 0.5 and point["speedup"] == 1.25 and point["label"] == "确认"
    assert point["search_speedup"] == detail["result"]["best_speedup"]
    assert point["batches"] == [1.25] and point["worst"] == 1.25 and point["best"] == 1.25
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
    """@brief 验证模型备注区分超时、空的工具应答、连接错误、HTTP 错误、截断、正文回复与提前停止，并隐去思考内容。

    @param tmp_path pytest 提供的临时目录。
    """
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
    """@brief 验证模型备注把 hmz 回合分成写出内核、出错与没交内核三类，并标出模型与花费。

    @param tmp_path pytest 提供的临时目录。
    """
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


def test_agent_diagnostics_reach_model_notes_and_turn_rows(tmp_path):
    """@brief 验证 agent 的失败诊断既进模型备注（失败类型、hmz 失败回合数与诊断文件路径），也进回合行。

    @param tmp_path pytest 提供的临时目录。
    """
    from observe.detail import _present_turn

    event = {"kind": "agent", "event": "agent", "label": "r4.p0", "harness": "pi", "provider": "moonshot",
             "model": "kimi-k3.1-pro", "had_kernel": False, "error": "Connection error.", "outcome": "agent_failed",
             "runtime": {"failure_class": "no_submission_after_result", "failed_turns": 1, "last_terminal_kind": "result"},
             "diagnostics_path": "agent_diagnostics/r4.json"}
    (tmp_path / "transcript.jsonl").write_text(json.dumps(event) + "\n")
    notes = _model_notes(tmp_path)
    assert notes[0]["kind"] == "agent_error"
    assert "正常回答但没有交卷" in notes[0]["detail"] and "1 个 hmz 失败回合" in notes[0]["detail"]
    assert notes[0]["diagnostics_path"] == "agent_diagnostics/r4.json"
    row = _present_turn({"label": "r4.p0", "min_seq": 1, "max_seq": 1, "events": [event]})
    assert row["kind"] == "agent_error" and row["error"] == "Connection error."
    assert "诊断 no_submission_after_result" in row["sent"] and "诊断文件 agent_diagnostics/r4.json" in row["sent"]


def test_pi_lines_keep_the_tool_name_and_drop_a_key(tmp_path):
    """@brief 验证从 pi 会话里提取的行保留工具名与结果摘要，但丢掉签名、api key 与整段用户提示词。

    @param tmp_path pytest 提供的临时目录。
    """
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


def test_pi_edit_and_write_results_say_which_change_they_were(tmp_path):
    """@brief 验证 edit/write 的回执后面带着自己的改动摘要，同一族候选的几次编辑不再读成同一行。

    pi 的结果文本是固定模板，只靠文件名区分；统计与改后的那一行让两次编辑一眼能分开。
    失败的调用与配不上调用的旧会话保持原文，不乱补一句像成功的话。

    @param tmp_path pytest 提供的临时目录。
    """
    def patch(threads: str, rows: str) -> str:
        """@brief 拼出 pi 存在 details 里的那份 unified patch。

        @param threads 改后的 ``threads`` 片段。
        @param rows 改后的 ``ROWS`` 值。
        @return 两处改动的 patch 文本。
        """
        return (
            "--- v2.metal\n+++ v2.metal\n@@ -9,2 +9,2 @@\n"
            "-launch softmax_kernel grid=M,1,1 threads=32,1,1\n"
            f"+launch softmax_kernel grid=M,1,1 {threads}\n"
            "@@ -21,2 +21,2 @@\n"
            "-    constexpr uint ROWS = 1;\n"
            f"+    constexpr uint ROWS = {rows};\n"
        )

    def message(role: str, **rest) -> str:
        """@brief 把一条消息包成会话文件里的一行。

        @param role 消息角色。
        @param rest 消息的其余字段。
        @return JSON 文本行。
        """
        return json.dumps({"type": "message", "message": {"role": role, **rest}})

    session = tmp_path / "session.jsonl"
    session.write_text("\n".join([
        message("assistant", content=[
            {"type": "toolCall", "name": "edit", "id": "e1", "arguments": {"path": "v5_r2.metal", "edits": []}},
            {"type": "toolCall", "name": "write", "id": "w1", "arguments": {"path": "v5_r1.metal", "content": "x\ny\n"}},
            {"type": "toolCall", "name": "edit", "id": "e2", "arguments": {"path": "v5_r4.metal", "edits": []}},
            {"type": "toolCall", "name": "edit", "id": "e3", "arguments": {"path": "v5_r2.metal", "edits": []}},
            {"type": "toolCall", "name": "edit", "id": "e4", "arguments": {"path": "v5_r8.metal", "edits": []}},
        ]),
        message("toolResult", toolCallId="e1", isError=False, details={"patch": patch("threads=64,1,1", "2")},
                content=[{"type": "text", "text": "Successfully replaced 2 block(s) in v5_r2.metal."}]),
        message("toolResult", toolCallId="w1", isError=False,
                content=[{"type": "text", "text": "Successfully wrote to v5_r1.metal"}]),
        message("toolResult", toolCallId="e2", isError=False, details={"patch": patch("threads=128,1,1", "4")},
                content=[{"type": "text", "text": "Successfully replaced 2 block(s) in v5_r4.metal."}]),
        message("toolResult", toolCallId="e3", isError=True, details={"patch": patch("threads=256,1,1", "8")},
                content=[{"type": "text", "text": "Could not edit file: v5_r2.metal. ENOENT."}]),
        # 老写法只留下 ``diff``，它多一列行号；行号不该被当成改动内容。
        message("toolResult", toolCallId="e4", isError=False,
                details={"diff": "  7 /* taic-launch\n- 9 launch softmax_kernel grid=M,1,1 threads=32,1,1\n"
                                 "+ 9 launch softmax_kernel grid=M/2,1,1 threads=64,1,1\n"},
                content=[{"type": "text", "text": "Successfully replaced 1 block(s) in v5_r8.metal."}]),
    ]) + "\n")
    results = [row["text"] for row in _pi_lines_in(session) if row["who"] == "结果"]
    assert len(results) == 5
    assert results[0].endswith("(+2 −2: launch softmax_kernel grid=M,1,1 threads=64,1,1 · constexpr uint ROWS = 2;)")
    assert results[2].endswith("(+2 −2: launch softmax_kernel grid=M,1,1 threads=128,1,1 · constexpr uint ROWS = 4;)")
    assert results[0] != results[2]
    assert results[1].endswith("(4 B / 2 行)")
    assert results[3] == "Could not edit file: v5_r2.metal. ENOENT."
    assert results[4].endswith("(+1 −1: launch softmax_kernel grid=M/2,1,1 threads=64,1,1)")


def test_a_long_pi_session_only_keeps_the_tail(tmp_path):
    """@brief 验证过长的 pi 会话只保留尾部若干字节与条数，最旧的行被丢弃。

    @param tmp_path pytest 提供的临时目录。
    """
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
    """@brief 验证 epic 记录能压成开始、打开、花费、结束四行，并隐去整段提示词与会话目录。

    @param tmp_path pytest 提供的临时目录。
    """
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
    """@brief 验证 kernel_writer 按 run_config.json 记录的模式判断谁写内核：显式 pi 就是 pi，老运行没有记录时按 transcript 里的事件推断。

    @param tmp_path pytest 提供的临时目录。
    """
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
    """@brief 验证对话尾部只展示最新一轮（发出、收到、思考、工具）并隐去系统提示词、密钥与内核全文。

    @param tmp_path pytest 提供的临时目录。
    """
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
    """@brief 验证 llm.json 里正在流式接收的请求被列成「正在接收」一行，不需要提示词内容。

    @param tmp_path pytest 提供的临时目录。
    """
    from observe.detail import _chat_inflight

    path = tmp_path / "llm.json"
    path.write_text(json.dumps({"inflight": {"init#t4": {"status": "streaming"}}}))
    assert _chat_inflight(path) == [{"who": "正在接收", "text": "init#t4"}]


def test_the_confirmed_point_shows_the_search_number_apart_and_calls_it_not_published(tmp_path, monkeypatch):
    """@brief 验证确认点与搜索最优点分开展示：确认点给出重测的 1.6896，同时带上未发布的搜索最大 1.7276。

    读者不能把搜索最大值当成确认结果；两个数都出现，其中一个标明未发布。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来把运行目录指到临时路径。
    """
    from observe import detail

    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(tmp_path))
    run = tmp_path / "run"
    run.mkdir()
    (run / "state.json").write_text(json.dumps({"task": "conv2d", "dtype": "fp16", "phase": "done", "pid": 1}))
    (run / "best.metal").write_text("kernel void conv2d_kernel() {}\n")
    (run / "confirm_report.json").write_text(json.dumps({
        "status": "ok", "benchmark": {"candidate": {"median_us": 1.0, "cv": 0.01}},
        "protocol_version": PROTOCOL_VERSION, "target_id": "apple-m4-max",
    }))
    (run / "result.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "size": "paper", "stop_reason": "budget_exhausted",
        "best_speedup": 1.7276, "confirmed_speedup": 1.6896, "confirmed_status": "ok",
        "confirmed_values": [1.6896], "target_id": "apple-m4-max",
        "confirm_winner_label": "r19.p0", "spent_usd": 15.2,
        "confirmation_source_sha256": __import__("hashlib").sha256(b"kernel void conv2d_kernel() {}\n").hexdigest(),
    }))
    got = detail._confirmed_point(
        json.loads((run / "result.json").read_text()), [], {"speedup": 1.6896, "publication": "confirmed"},
    )
    assert got["speedup"] == 1.6896 and got["search_speedup"] == 1.7276


def test_a_running_run_shows_start_and_elapsed_but_no_end(tmp_path, monkeypatch):
    """@brief 验证仍在运行的搜索只给出开始时间与已运行时长，没有结束时间。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来把运行目录指到临时路径。
    """
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    directory = runs / "live"
    directory.mkdir(parents=True)
    started = time.time() - 125
    (directory / "state.json").write_text(json.dumps({
        "pid": os.getpid(), "started_at": started, "updated_at": time.time(),
        "phase": "plan", "task": "gemm", "dtype": "fp16",
    }))
    detail = scan.run_detail("live")
    assert detail["status"] == "running"
    assert detail["timing"]["started_at"] == pytest.approx(started)
    assert detail["timing"]["ended_at"] is None
    assert detail["timing"]["elapsed_s"] >= 125
    assert detail["timing"]["running"] is True


def test_the_panels_are_collapsed_until_opened_and_kept_across_a_refresh_in_node():
    """@brief 用 Node 执行面板纯函数：合上时不读文件、展开后注入 payload、刷新后保留已读面板。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = Path("observe/static/index.html").read_text()

    def extract(name):
        start = page.index(f"function {name}(")
        brace = page.index("{", start)
        depth = 0
        for index in range(brace, len(page)):
            if page[index] == "{":
                depth += 1
            elif page[index] == "}":
                depth -= 1
                if depth == 0:
                    return page[start:index + 1]
        raise AssertionError(f"unclosed function {name}")

    names = ("esc", "panelLabel", "panelKey", "panelNames", "panelLoaded", "mergeCandidatePage", "mergePanel",
             "carryPanels", "panelSection", "planPanelFetches")
    script = "\n".join(extract(name) for name in names) + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
let navigationEpoch = 0;
function panelBody(name) { return "BODY:" + name; }
const names = panelNames();
const expected = ["candidates", "budget-events", "grid", "plan", "lineage", "feedback"];
if (JSON.stringify(names) !== JSON.stringify(expected)) throw new Error("panel list drifted: " + names);
if (panelKey("budget-events") !== "budget_events" || panelKey("lineage") !== "lineage"
    || panelKey("feedback") !== "feedback") throw new Error("panel key mapping wrong");

const closed = panelSection("candidates", {}, false);
if (!closed.includes("data-panel=\"candidates\"")) throw new Error("panel is not addressable");
if (!closed.includes("收起时不读文件")) throw new Error("cold panel has no placeholder");
if (closed.includes(" open")) throw new Error("cold panel rendered open");

const doc = {};
mergePanel("candidates", doc, {candidates: [{label: "a"}], candidate_marks: [{label: "a"}]});
if (!panelLoaded("candidates", doc)) throw new Error("candidates payload did not land");
if (doc.candidates.length !== 1 || doc.candidate_marks.length !== 1) throw new Error("candidates payload incomplete");
const opened = panelSection("candidates", doc, true);
// 展开态不在 markup 里：插入带 open 的 details 会派发 toggle，展开由 restoreDetailOpenState 恢复。
if (opened.includes(" open")) throw new Error("open must not be serialized into the panel markup");
if (!opened.includes("BODY:candidates")) throw new Error("loaded panel did not render its body");

// 已加载但用户合上：内容留在 DOM 里，但刷新后不重取。
const collapsed = panelSection("candidates", doc, false);
if (collapsed.includes(" open")) throw new Error("collapsed loaded panel rendered open");
if (!collapsed.includes("BODY:candidates")) throw new Error("collapsed panel lost its loaded body");
const openSet = new Set(["candidates", "grid"]);
if (JSON.stringify(planPanelFetches(names, openSet)) !== JSON.stringify(["candidates", "grid"])) {
  throw new Error("only open panels may be refetched: " + JSON.stringify(planPanelFetches(names, openSet)));
}
if (planPanelFetches(names, new Set()) .length !== 0) throw new Error("no open panels must mean no requests");

mergePanel("budget-events", doc, {budget_events: [1, 2]});
mergePanel("grid", doc, {feature_grid: [1]});
mergePanel("plan", doc, {plan_outcomes: [1]});
mergePanel("lineage", doc, {lineage: {nodes: [], edges: []}});
mergePanel("feedback", doc, {feedback: {records: []}});
for (const name of names) if (!panelLoaded(name, doc)) throw new Error(name + " did not land");

// 刷新后的新详情只有摘要：旧 payload 必须搬过去，否则已展开的块会退回占位。
const fresh = {id: "run"};
carryPanels(doc, fresh);
for (const name of names) if (!panelLoaded(name, fresh)) throw new Error(name + " was dropped on refresh");
const untouched = {id: "run2", candidates: [{label: "own"}]};
carryPanels(doc, untouched);
if (untouched.candidates.length !== 1 || untouched.candidates[0].label !== "own") throw new Error("carryPanels clobbered the new payload");
console.log("ok");
'''
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "ok"


def test_the_compare_view_names_mixed_configs_in_node():
    """@brief 验收（plan §8.0 第 3 条）：混合臂在对照视图里点名 config_key 与 run id，纯单配置臂不显示。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = Path("observe/static/index.html").read_text()

    def extract(name, *, is_async=False):
        marker = f"async function {name}(" if is_async else f"function {name}("
        start = page.index(marker)
        brace = page.index("{", start)
        depth = 0
        for index in range(brace, len(page)):
            if page[index] == "{":
                depth += 1
            elif page[index] == "}":
                depth -= 1
                if depth == 0:
                    return page[start:index + 1]
        raise AssertionError(f"unclosed function {name}")

    script = "\n".join((extract("esc"), extract("num"), extract("loadCompare", is_async=True))) + r'''
const hosts = {};
globalThis.document = { getElementById: (id) => (hosts[id] = hosts[id] || { innerHTML: "" }) };
const arm = (key) => ({ arm_id: "relu", median: null, min: null, max: null, published_n: 0, n: 1,
  confirmed_n: 0, fast: { "1": { share: 0, n: 0 } }, spent_usd: 0.1, gain_per_usd: null, config_key: key });
const base = { experiment_id: "exp", found: true, verdict: "样本不足以判定", reason: "发布数字少于 2 个",
  winner: null, drift: [], inferred: [], arms: [arm("aaaaaaaaaaaa"), arm("bbbbbbbbbbbb")],
  mixed: [{ arm_id: "relu", config_key: "aaaaaaaaaaaa", runs: ["exp/relu/trial_1"] },
          { arm_id: "relu", config_key: "bbbbbbbbbbbb", runs: ["exp/relu/trial_2"] }] };
const pureArm = { ...base.arms[0] };
delete pureArm.config_key;
const responses = [base, { ...base, arms: [pureArm], mixed: [] }];
globalThis.fetch = async () => ({ json: async () => responses.shift() });
loadCompare("exp").then(() => {
  const html = hosts["view-compare"].innerHTML;
  if (!html.includes("混合配置")) throw new Error("mixed header missing: " + html);
  if (!html.includes("aaaaaaaaaaaa") || !html.includes("bbbbbbbbbbbb")) throw new Error("config keys missing");
  if (!html.includes("exp/relu/trial_1") || !html.includes("exp/relu/trial_2")) throw new Error("run ids missing");
  if (!html.includes("配置 aaaaaaaaaaaa")) throw new Error("arm row not labelled with config key");
  return loadCompare("exp");
}).then(() => {
  const clean = hosts["view-compare"].innerHTML;
  if (clean.includes("混合配置")) throw new Error("pure arm must not show a mixed block");
  if (clean.includes("配置 ")) throw new Error("pure arm must not label a config key");
  console.log("ok");
}).catch((err) => { console.error(err); process.exit(1); });
'''
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_the_detail_page_renders_each_heavy_block_as_a_collapsed_panel():
    """@brief 验证详情页把五个重块都改成 panelSection，不再在 renderDetail 里内联候选/预算/格子/计划表。"""
    page = Path("observe/static/index.html").read_text()
    start = page.index("function detailSections(doc)")
    end = page.index("function feedbackClaimCard(claim)")
    body = page[start:end]
    for name in ("candidates", "budget-events", "grid", "plan", "feedback"):
        assert f'panelSection("{name}", doc, openPanels.has("{name}"))' in body
    assert "newestFirst(doc.candidates)" not in body
    assert "newestFirst(doc.feature_grid)" not in body
    assert "newestFirst(doc.plan_outcomes)" not in body
    assert "newestFirst(doc.budget_events)" not in body
    load = page[page.index("async function loadDetail(id)"):page.index("function ledgerTable(")]
    assert "bindPanels(host)" in load and "refreshOpenPanels(id)" in load
    # §10.4：详情用 keyed patch，不再整块 innerHTML = renderDetail(...)。
    assert "patchDetail(host, doc)" in load and "renderDetail(doc)" not in load
    refresh = page[page.index("function refreshOpenPanels(id)"):page.index("async function loadDetail(id)")]
    assert "planPanelFetches(panelNames(), openPanels, [pendingPanel, pendingPanelOwner])" in refresh and "loadPanel(name, id)" in refresh


def test_a_cold_detail_open_only_requests_the_summary_in_node():
    """@brief Node 执行详情请求计划：冷开只发摘要；回合/pi/会话成本要展开才请求。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = Path("observe/static/index.html").read_text()

    def extract(name):
        start = page.index(f"function {name}(")
        brace = page.index("{", start)
        depth = 0
        for index in range(brace, len(page)):
            if page[index] == "{":
                depth += 1
            elif page[index] == "}":
                depth -= 1
                if depth == 0:
                    return page[start:index + 1]
        raise AssertionError(f"unclosed function {name}")

    script = extract("planDetailRequests") + r'''
const cold = planDetailRequests({hasRun: true});
if (cold.detail !== true) throw new Error("cold open must request the summary");
if (cold.turns || cold.live || cold.costs) throw new Error("cold open must not request turns/live/costs: " + JSON.stringify(cold));
const warm = planDetailRequests({hasRun: true, turnsOpen: true, piOpen: true, costsOpen: true});
if (!warm.turns || !warm.live || !warm.costs) throw new Error("open sections must be requested on refresh");
const empty = planDetailRequests({hasRun: false, turnsOpen: true});
if (empty.detail || empty.turns) throw new Error("no run must mean no requests");
console.log("ok");
'''
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "ok"

    load = page[page.index("async function loadDetail(id)"):page.index("function ledgerTable(")]
    assert "planDetailRequests(" in load
    assert "refreshTurns(true);\n  refreshLive();" not in load
    assert "loadSessionCosts(id, true);" not in load.split("planDetailRequests", 1)[-1].split("}", 1)[0]
    turns = page[page.index("async function refreshTurns(reset)"):page.index("async function loadEarlier()")]
    assert "!turnsOpen" in turns
    live = page[page.index("async function refreshLive()"):page.index("setInterval(")]
    assert "!piOpen" in live
    panel = page[page.index("async function loadPanel(name, id)"):page.index("function bindPanels(host)")]
    assert "bindFeedback(again)" not in panel
    assert page.count("\n  bindFeedback(host);") == 1 and "bindFeedback(again)" not in page


def test_a_summary_same_still_refreshes_open_panels_in_node():
    """@brief 真正执行异步 ``loadDetail()``：摘要回 ``same`` 时，已展开面板仍发请求；合上的不发。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = Path("observe/static/index.html").read_text()

    def extract(name, *, asynchronous=False):
        start = page.index(("async function " if asynchronous else "function ") + name + "(")
        brace = page.index("{", start)
        depth = 0
        for index in range(brace, len(page)):
            if page[index] == "{":
                depth += 1
            elif page[index] == "}":
                depth -= 1
                if depth == 0:
                    return page[start:index + 1]
        raise AssertionError(f"unclosed function {name}")

    names = ("panelNames", "panelKey", "panelLoaded", "planPanelFetches", "panelQuery")
    functions = "\n".join(extract(name) for name in names)
    functions += "\n" + extract("refreshOpenPanels") + "\n" + extract("consumePendingPanel") + "\n" + extract("loadPanel", asynchronous=True)
    functions += "\n" + extract("loadDetail", asynchronous=True)
    script = functions + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
let navigationEpoch = 0;
function panelBody(name) { return "BODY:" + name; }
function bindCandidatePicks() {}
function bindFeedback() {}
function mergePanel() {}
function refreshLive() {}
function esc(value) { return String(value ?? ""); }
function panelLabel(name) { return name; }
function resetPanels() {}
function carryPanels() {}
function renderDetail() { return ""; }
function patchDetail() {}
function restoreDetailOpenState() {}
function detailPatchedRun() { return ""; }
function clearDetailPatch() {}
function overlayClose() {}
function detailScroller() { return null; }
function rememberDetailScroll() {}
function recallDetailScroll() {}
function bindPanels() {}
function paintDetailHead() {}
function openStage() {}
function applyPendingDetail() {}
function planDetailRequests() { return {detail: true, turns: false, live: false, costs: false}; }
function refreshTurns() {}
function loadSessionCosts() {}
function loadFeedbackRecord() {}
function panelUrl(id, name) { return "/api/v2/runs/" + id.split("/").map(encodeURIComponent).join("/") + "/" + name; }

const hosts = {"panel-candidates": {innerHTML: ""}};
const fetches = [];
globalThis.document = {
  getElementById: (id) => hosts[id] || null,
};
globalThis.fetch = async (url) => {
  fetches.push(String(url));
  return {ok: true, json: async () => ({same: true, detail_stamp: "s1", stamp: "p1"})};
};
let detailToken = 0, detailDoc = {}, detailId = "run", detailStamp = "s1";
let selectedRun = "run", visible = "detail", selectedCandidate = "";
let panelStamps = {candidates: "p1"}, panelTokens = {}, openPanels = new Set(["candidates"]);
const panelInflight = new Map();
let panelRequestSeq = 0;
let turnsOpen = false, piOpen = false, costsOpen = false, handoffOpen = false;
let restoredDetailView = null;
let pendingPanel = "";
let pendingPanelOwner = "";
const CANDIDATES_PAGE = 50, FEEDBACK_PAGE = 10;
let candidateStatus = "", candidateStage = "", candidatePages = 1, feedbackPages = 1;

(async () => {
  await loadDetail("run");
  if (fetches.length !== 2) throw new Error("expected summary + open panel requests, got " + JSON.stringify(fetches));
  if (!fetches[0].startsWith("/api/v2/runs/run?stamp=s1")) throw new Error("summary request wrong: " + fetches[0]);
  if (!fetches[1].includes("/candidates?limit=50&stamp=p1")) throw new Error("open panel was not refreshed: " + fetches[1]);

  // 合上的面板不让摘要 same 触发任何请求。
  fetches.length = 0;
  openPanels = new Set();
  await loadDetail("run");
  if (fetches.length !== 1) throw new Error("collapsed panel must not be refetched, got " + JSON.stringify(fetches));
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "ok"


def test_stale_async_responses_are_discarded_in_node():
    """@brief 用 deferred JSON 驱动三个异步路径：切 run 或改戳后，旧响应不得合并/缓存/渲染。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = Path("observe/static/index.html").read_text()

    def extract(name, *, asynchronous=False):
        start = page.index(("async function " if asynchronous else "function ") + name + "(")
        brace = page.index("{", start)
        depth = 0
        for index in range(brace, len(page)):
            if page[index] == "{":
                depth += 1
            elif page[index] == "}":
                depth -= 1
                if depth == 0:
                    return page[start:index + 1]
        raise AssertionError(f"unclosed function {name}")

    functions = "\n".join(extract(name, asynchronous=True) for name in ("loadMoreCandidates", "loadFeedbackRecord", "openFeedbackClaim"))
    script = functions + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
let navigationEpoch = 0;
let merges = [];
let rendered = [];
let reviews = [];
let fetches = 0;
let releases = [];
function panelUrl(id, name) { return "/api/v2/runs/" + id + "/" + name; }
function mergeCandidatePage(doc, payload, append) { merges.push([doc.id, append, payload.candidates.length]); }
function panelBody(name, doc) { return "BODY"; }
function bindCandidatePicks() {}
function feedbackFullMarkup(record) { return "FULL:" + record.seq; }
function pickFeedbackFull(cache, stamp, key) {
  const entry = cache.get(key);
  return entry && entry.stamp === stamp ? entry.record : null;
}
function openReview(id, claim, record) { reviews.push([id, claim.digest]); }
function revealFeedbackRecord() {}
const hosts = {};
globalThis.document = {getElementById: (id) => hosts[id] || null};
globalThis.fetch = async (url) => {
  fetches += 1;
  await new Promise((resolve) => releases.push(resolve));
  if (String(url).includes("/feedback/")) return {ok: true, json: async () => ({record: {seq: 7, claims: [{digest: "a", text: "T"}]}})};
  return {ok: true, json: async () => ({candidates: [{label: "x"}], total: 2, next: null})};
};
let selectedRun = "A", detailId = "A", detailDoc = {id: "A", candidates: [], candidatesNext: "c1"};
let panelStamps = {feedback: "s1"}, panelTokens = {}, openPanels = new Set();
const panelInflight = new Map();
let panelRequestSeq = 0;
let candidatePageToken = 0, feedbackRecordToken = 0, feedbackClaimToken = 0;
let currentSeq = 0;
const CANDIDATES_PAGE = 50, FEEDBACK_PAGE = 10;
let candidateStatus = "", candidateStage = "", candidatePages = 1, feedbackPages = 1;
function panelQuery() { return "limit=" + CANDIDATES_PAGE; }
function repaintCandidatePanel(doc) { panelBody("candidates", doc); }
function commitUrl() {}
function currentViewState() { return {}; }
const fullFeedback = new Map();
let reviewsCalibration = null;

(async () => {
  // 1) 候选续页：请求在途时切到另一场，响应必须丢弃。
  const paging = loadMoreCandidates();
  detailId = "B";
  detailDoc = {id: "B", candidates: [], candidatesNext: null};
  selectedRun = "B";
  releases.shift()();
  await paging;
  if (merges.length !== 0) throw new Error("stale candidate page merged: " + JSON.stringify(merges));
  if (detailDoc.candidates.length !== 0) throw new Error("stale page mutated the new run");

  // 2) 反馈全文：请求在途时同一个 run 的反馈戳变了，旧响应不得盖新戳。
  detailId = "A";
  detailDoc = {id: "A", feedback: {records: [{seq: 7, claims: [{digest: "a"}]}]}};
  selectedRun = "A";
  panelStamps = {feedback: "s1"};
  hosts["feedback-full-7"] = {innerHTML: "before"};
  const loading = loadFeedbackRecord(7, "A");
  panelStamps = {feedback: "s2"};
  releases.shift()();
  await loading;
  if (fullFeedback.size !== 0) throw new Error("stale record cached under a new stamp");
  if (hosts["feedback-full-7"].innerHTML !== "before") throw new Error("stale record rendered");

  // 3) 单条处理：请求在途时切 run，不能弹到新 run 上。
  panelStamps = {feedback: "s3"};
  const claiming = openFeedbackClaim(7, "a");
  detailId = "B";
  releases.shift()();
  await claiming;
  if (reviews.length !== 0) throw new Error("stale claim opened a review: " + JSON.stringify(reviews));
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "ok"


def test_one_feedback_click_makes_exactly_one_fetch_in_node():
    """@brief 真跑 ``openFeedbackClaim`` 的网络路径：一次点击只发一个反馈全文请求（不靠 stub 计数）。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = Path("observe/static/index.html").read_text()

    def extract(name, *, asynchronous=False):
        start = page.index(("async function " if asynchronous else "function ") + name + "(")
        brace = page.index("{", start)
        depth = 0
        for index in range(brace, len(page)):
            if page[index] == "{":
                depth += 1
            elif page[index] == "}":
                depth -= 1
                if depth == 0:
                    return page[start:index + 1]
        raise AssertionError(f"unclosed function {name}")

    script = extract("bindFeedback") + "\n" + extract("openFeedbackClaim", asynchronous=True) + r'''
let navigationEpoch = 0;
const fetches = [];
const reviews = [];
globalThis.fetch = async (url) => {
  fetches.push(String(url));
  return {ok: true, json: async () => ({record: {seq: 7, claims: [{digest: "a", text: "T"}]}})};
};
function openReview(id, claim, record) { reviews.push([id, claim.digest, record.seq]); }
function pickFeedbackFull() { return null; }
let selectedRun = "run", detailId = "run";
let detailDoc = {id: "run", feedback: {records: [{seq: 7, claims: [{digest: "a"}]}]}};
let panelStamps = {feedback: "s1"};
let feedbackClaimToken = 0;
let currentSeq = 0;
function commitUrl() {}
function currentViewState() { return {}; }
const fullFeedback = new Map();
let reviewsCalibration = null;

const listeners = [];
const host = {dataset: {}, addEventListener: (type, handler) => listeners.push(handler)};
bindFeedback(host);
bindFeedback(host);  // guard：重复安装不再叠加监听
if (listeners.length !== 1) throw new Error("bindFeedback installed " + listeners.length + " listeners");
const button = {dataset: {seq: "7", digest: "a"}};
listeners[0]({target: {closest: (selector) => (selector === ".feedback-open" ? button : null)}});

setTimeout(() => {
  if (fetches.length !== 1) throw new Error("one click must trigger exactly one fetch, got " + JSON.stringify(fetches));
  if (reviews.length !== 1 || reviews[0][1] !== "a") throw new Error("review not opened once: " + JSON.stringify(reviews));
  console.log("ok");
}, 20);
'''
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "ok"


def test_one_feedback_click_makes_exactly_one_request_in_node():
    """@brief Node 驱动反馈委托监听：一次点击只触发一次 openFeedbackClaim，重复安装不会叠加。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = Path("observe/static/index.html").read_text()

    def extract(name):
        start = page.index(f"function {name}(")
        brace = page.index("{", start)
        depth = 0
        for index in range(brace, len(page)):
            if page[index] == "{":
                depth += 1
            elif page[index] == "}":
                depth -= 1
                if depth == 0:
                    return page[start:index + 1]
        raise AssertionError(f"unclosed function {name}")

    script = extract("bindFeedback") + r'''
let calls = [];
function openFeedbackClaim(seq, digest) { calls.push([seq, digest]); }
const listeners = [];
const host = {dataset: {}, addEventListener: (type, handler) => listeners.push(handler)};
bindFeedback(host);
bindFeedback(host);  // 重复安装必须被 guard 拦掉
if (listeners.length !== 1) throw new Error("bindFeedback installed " + listeners.length + " listeners");
const button = {dataset: {seq: "7", digest: "abc"}};
listeners[0]({target: {closest: (selector) => (selector === ".feedback-open" ? button : null)}});
if (calls.length !== 1) throw new Error("one click must make exactly one request, got " + calls.length);
if (calls[0][0] !== "7" || calls[0][1] !== "abc") throw new Error("wrong claim forwarded: " + JSON.stringify(calls[0]));
listeners[0]({target: {closest: () => null}});
if (calls.length !== 1) throw new Error("a click outside a claim must not fetch");
console.log("ok");
'''
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "ok"


def test_candidate_paging_and_full_feedback_freshness_in_node():
    """@brief Node 执行候选分页与全文缓存：250 颗分两页可达；全文缓存按反馈戳失效。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = Path("observe/static/index.html").read_text()

    def extract(name):
        start = page.index(f"function {name}(")
        brace = page.index("{", start)
        depth = 0
        for index in range(brace, len(page)):
            if page[index] == "{":
                depth += 1
            elif page[index] == "}":
                depth -= 1
                if depth == 0:
                    return page[start:index + 1]
        raise AssertionError(f"unclosed function {name}")

    functions = "\n".join(
        extract(name)
        for name in ("mergeCandidatePage", "mergePanel", "carryPanels", "panelKey", "panelNames",
                     "panelLoaded", "pickFeedbackFull")
    )
    script = functions + r'''
let navigationEpoch = 0, candidateContentEpoch = 0, feedbackContentEpoch = 0;
function row(index) { return {idx: index, label: "c" + index}; }
let candidatePages = 1, feedbackPages = 1;
const first = Array.from({length: 200}, (_, index) => row(index));
const second = Array.from({length: 50}, (_, index) => row(200 + index));
const doc = {};
let state = mergeCandidatePage(doc, {candidates: first, candidate_marks: [{label: "c0"}], total: 250, next: "cur1"}, false);
if (doc.candidates.length !== 200 || state.next !== "cur1" || state.total !== 250) throw new Error("first page wrong");
state = mergeCandidatePage(doc, {candidates: second, total: 250, next: null}, true);
if (doc.candidates.length !== 250) throw new Error("second page not appended: " + doc.candidates.length);
if (new Set(doc.candidates.map((item) => item.label)).size !== 250) throw new Error("candidate page dedupe failed");
if (state.next !== null) throw new Error("next should clear after the last page");
mergeCandidatePage(doc, {candidates: second, total: 250, next: null}, true);
if (doc.candidates.length !== 250) throw new Error("re-appending a page duplicated rows");
if (!doc.candidate_marks || doc.candidate_marks.length !== 1) throw new Error("marks lost on page merge");

// 详情刷新：carryPanels 把旧 detailDoc 当 payload 传回来，游标与总数必须原样保留。
const carried = {id: "run"};
const pageOne = {candidates: first, candidate_marks: [{label: "c0"}], candidatesNext: "cur1", candidatesTotal: 250};
carryPanels(pageOne, carried);
if (!Array.isArray(carried.candidates) || carried.candidates.length !== 200) throw new Error("carryPanels lost the first page");
if (carried.candidatesNext !== "cur1") throw new Error("carryPanels lost the cursor: " + carried.candidatesNext);
if (carried.candidatesTotal !== 250) throw new Error("carryPanels reset the total: " + carried.candidatesTotal);
mergeCandidatePage(carried, {candidates: second, total: 250, next: null}, true);
if (carried.candidates.length !== 250) throw new Error("second page lost after carryPanels refresh");
if (new Set(carried.candidates.map((item) => item.label)).size !== 250) throw new Error("carryPanels page merge duplicated rows");

const cache = new Map();
cache.set("run:1", {stamp: "s1", record: {seq: 1, status: "pending"}});
if (pickFeedbackFull(cache, "s1", "run:1") === null) throw new Error("fresh full record must be reused");
if (pickFeedbackFull(cache, "s2", "run:1") !== null) throw new Error("stale full record must be ignored after a stamp change");
if (pickFeedbackFull(cache, "", "run:1") !== null) throw new Error("a missing stamp must not reuse the cache");
console.log("ok");
'''
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "ok"

    markup = page[page.index("function candidatesMarkup(doc)"):page.index("async function loadMoreCandidates()")]
    assert "candidates-more" in markup and "candidatesNext" in markup
    more = page[page.index("async function loadMoreCandidates()"):page.index("// 展开（open）与已加载（loaded）")]
    assert "cursor=" in more and 'panelQuery("candidates")' in more
    assert "fullFeedback.set(key, {stamp:" in page


def test_the_feedback_panel_is_a_compact_list_and_the_full_view_carries_the_text_in_node():
    """@brief Node 执行反馈面板纯函数：列表不含正文/原文，全文视图才带 claim 正文与 raw，且都转义。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = Path("observe/static/index.html").read_text()

    def extract(name):
        start = page.index(f"function {name}(")
        brace = page.index("{", start)
        depth = 0
        for index in range(brace, len(page)):
            if page[index] == "{":
                depth += 1
            elif page[index] == "}":
                depth -= 1
                if depth == 0:
                    return page[start:index + 1]
        raise AssertionError(f"unclosed function {name}")

    names = ("esc", "num", "paginateGroup", "tailWindow", "feedbackClaimCard", "feedbackCompactMarkup", "feedbackFullMarkup")
    script = "\n".join(extract(name) for name in names) + r'''
const FEEDBACK_PAGE = 10;
const TEXT_WINDOW_BYTES = 256 * 1024;
let feedbackPages = 1;
let feedbackNotice = "";
let currentSeq = 0;
function subclaimsMarkup() { return ""; }
function priorMarkup() { return ""; }
function similarMarkup() { return ""; }

const compactRecord = {
  seq: 7, label: "r1.p0", round: 1, committee: "first", provider: "deepseek", model: "deepseek-flash",
  outcome: "evaluated", kinds: "tool", truncated_sections: [],
  claim_counts: {total: 2, pending: 1, adopted: 1},
  claims: [
    {digest: "a", section: "需要帮助", premise: "unverified", audience: "system", status: "pending", subclaims_count: 1, open_subclaims_count: 1, text: "LEAKED compact text"},
    {digest: "b", section: "对系统的建议", premise: "verified", audience: "system", status: "adopted", subclaims_count: 0, open_subclaims_count: 0},
  ],
};
const compact = feedbackCompactMarkup({feedback: {records: [compactRecord]}});
if (compact.includes("LEAKED compact text")) throw new Error("compact list leaked claim text");
if (!compact.includes("需要帮助") || !compact.includes("pending")) throw new Error("compact list lost the index fields");
if (!compact.includes("pending 1")) throw new Error("compact list lost the claim counts");
if (!compact.includes("data-feedback-seq=\"7\"")) throw new Error("compact list has no full-view loader");
if (!compact.includes("data-digest=\"a\"")) throw new Error("compact list lost the review entry point");
if (feedbackCompactMarkup({feedback: {records: []}}).includes("fb-session")) throw new Error("empty state wrong");

const fullRecord = Object.assign({}, compactRecord, {
  raw: "SECRET <img onerror=alert(1)>",
  claims: [{digest: "a", section: "需要帮助", premise: "unverified", audience: "system", status: "pending",
            text: "<script>alert(2)</script> 正文", subclaims: [], open_subclaims: []}],
});
const full = feedbackFullMarkup(fullRecord);
if (!full.includes("&lt;script&gt;alert(2)&lt;/script&gt;")) throw new Error("full view did not escape claim text");
if (full.includes("<script>alert(2)</script>")) throw new Error("full view rendered raw HTML");
if (!full.includes("SECRET &lt;img onerror=alert(1)&gt;")) throw new Error("full view lost the raw text");
if (!full.includes("原文")) throw new Error("full view has no raw section");
console.log("ok");
'''
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "ok"


def test_the_detail_page_has_the_run_clock():
    """@brief 验证详情页 HTML 仍带着运行时钟与预算面板需要的元素 id 与中文文案。
    """
    html = Path("observe/static/index.html").read_text()
    for token in ("detail-run-clock", "开始时间", "已运行", "结束时间", "formatDuration",
                  "detail-budget", "总预算", "已花", "剩余", "money2", "remaining_usd",
                  "胜者：", "模型费用"):
        assert token in html


def test_the_detail_page_exposes_session_and_model_cost_efficiency():
    """@brief 验证成本表使用真实 KPI 字段、转义与 stamp 短路；不只做函数名/排版断言。"""
    html = Path("observe/static/index.html").read_text()
    for token in (
        "会话成本", "工具调用", "cache_read", "cache_write", "cache_read/output",
        "USD per 1M output", "场均（全部会话）", "记录成本", "成本覆盖",
        "esc(row.label ||", "esc(row.model ||", "number2(row.cache_read_per_output)",
        "money2(row.usd_per_1m_output)", "money2(row.cost_per_session_usd)",
        "row.sessions_with_cost", "row.tokens_output", "/sessions", "/api/v2/model-stats?run=",
        "sessionCostStamp", "doc.same", "modelCostTable",
    ):
        assert token in html
    assert "model_turns" not in html
    # 后端 null/NaN/Infinity 由 test_session_kpi 做行为验收；这里没有 JS 运行时，不能假称覆盖运行时渲染。
    assert "loadSessionCosts(id, true);" in html  # 完整重绘必须恢复两张表；只有 turns_revision 用 stamp。
    assert "loadSessionCosts(selectedRun);" in html
    assert "catch (error)" in html
    assert "if (doc.same) return;" in html


def test_cost_tables_escape_and_render_nulls_in_node():
    """@brief 用 Node 真正执行两张成本表：文本转义、成本覆盖与 NaN/Infinity 空值行为。"""
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = Path("observe/static/index.html").read_text()

    def extract(name):
        start = page.index(f"function {name}(")
        brace = page.index("{", start)
        depth = 0
        for index in range(brace, len(page)):
            if page[index] == "{":
                depth += 1
            elif page[index] == "}":
                depth -= 1
                if depth == 0:
                    return page[start:index + 1]
        raise AssertionError(f"unclosed function {name}")

    functions = "\n".join(extract(name) for name in ("esc", "num", "money2", "number2", "modelCostTable", "sessionCostTable"))
    script = functions + r'''
const model = modelCostTable([{provider:"p",model:"<img onerror=alert(1)>",committee:"first",sessions:2,sessions_with_cost:null,cost_usd:0.5,cache_read_tokens:8,cache_write_tokens:4,output_tokens:8,cache_read_per_output:NaN,usd_per_1m_output:Infinity,cost_per_session_usd:0.25}]);
if (!model.includes("&lt;img onerror=alert(1)&gt;")) throw new Error("model escaping failed");
if (!model.includes("—")) throw new Error("model null/NaN placeholder missing");
const session = sessionCostTable([{label:"<img onerror=alert(2)>",model:"m",outcome:"ok",tool_calls:2,tokens_output:2,cache_read_per_output:NaN,usd_per_1m_output:Infinity,cost_usd:0.2}]);
if (!session.includes("&lt;img onerror=alert(2)&gt;")) throw new Error("session escaping failed");
if (!session.includes("—")) throw new Error("session null/NaN placeholder missing");
if (session.includes("Infinity") || session.includes("NaN")) throw new Error("non-finite value rendered");
console.log("ok");
'''
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "ok"


def test_model_stats_failure_preserves_session_table_and_stamp_in_node():
    """@brief 用 Node 模拟 sessions 成功、model-stats 失败：会话表不被覆盖，stamp 也不丢。"""
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = Path("observe/static/index.html").read_text()

    def extract(name, *, asynchronous=False):
        start = page.index(("async function " if asynchronous else "function ") + name + "(")
        brace = page.index("{", start)
        depth = 0
        for index in range(brace, len(page)):
            if page[index] == "{":
                depth += 1
            elif page[index] == "}":
                depth -= 1
                if depth == 0:
                    return page[start:index + 1]
        raise AssertionError(f"unclosed function {name}")

    functions = "\n".join(extract(name) for name in ("esc", "num", "money2", "number2", "modelCostTable", "sessionCostTable"))
    functions += "\n" + extract("loadSessionCosts", asynchronous=True)
    script = functions + r'''
const hosts = {"session-costs": {innerHTML: ""}, "model-costs": {innerHTML: "fallback"}};
globalThis.document = {getElementById: (id) => hosts[id] || null};
function syncDetailSectionCache() {}
globalThis.detailId = "run";
let sessionCostToken = 0, sessionCostStamp = "", sessionCostRun = "";
globalThis.fetch = async (url) => {
  if (url.includes("/sessions")) return {ok: true, json: async () => ({same: false, stamp: "s1", sessions: [{label: "ok", model: "m", outcome: "evaluated", tool_calls: 1, tokens_output: 2, cache_read_per_output: 1, usd_per_1m_output: 1, cost_usd: 0.2}]})};
  throw new Error("model-stats down");
};
(async () => {
  await loadSessionCosts("run");
  if (sessionCostStamp !== "s1") throw new Error("stamp lost after model-stats failure");
  if (!hosts["session-costs"].innerHTML.includes("ok")) throw new Error("session table was overwritten");
  if (hosts["session-costs"].innerHTML.includes("读取失败")) throw new Error("session table shows an error after sessions succeeded");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "ok"


def test_the_detail_payload_carries_the_current_budget_spent_and_remaining(tmp_path, monkeypatch):
    """@brief 验证详情负载给出当前预算、已花与剩余金额。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来把运行目录指到临时路径。
    """
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    directory = runs / "live"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "pid": os.getpid(), "started_at": time.time() - 10, "updated_at": time.time(),
        "phase": "plan", "task": "gemm", "dtype": "fp16", "spent_usd": 1.0, "budget_usd": 5.0,
    }))
    detail = scan.run_detail("live")
    assert detail["budget"]["spent"] == pytest.approx(1.0)
    assert detail["budget"]["budget"] == pytest.approx(5.0)
    assert detail["budget"]["remaining_usd"] == pytest.approx(4.0)



def test_pi_feed_finds_the_session_hmz_kept_inside_the_epic(tmp_path, monkeypatch):
    """@brief 验证跑在 hmz 0.2.2 上的运行仍能看到 pi 输出：会话在 epic 目录里，且 epic 名字把运行名里的下划线换成了连字符。

    hmz 0.2.2 起给每个 epic 建 own home，pi 的会话不再落到 ``~/.pi/agent/sessions``；目录名又会被压成只含
    字母数字与连字符。两条都对不上时「pi 输出」这块永远是空的，运行看着像没在动。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 提供的环境替换器。
    """
    run_name = "20261025-070409_conv2d_fp16_deepseek-flash_g5yab2vc"
    run = tmp_path / "runs" / "apple-m4-max" / run_name
    run.mkdir(parents=True)
    (run / "run_config.json").write_text(json.dumps({"search": {"kernel_agent": "pi"}}))
    # resolve_run 只认有搜索快照的目录；内容是什么不看。
    (run / "state.json").write_text(json.dumps({"phase": "llm", "round": 0}))

    slug = "apple-m4-max-20261025-070409-conv2d-fp16-deepseek-flash-g5yab2vc-agent-work-init"
    epic = tmp_path / "epics" / f"-Users-kuafu-runs-{slug}" / "20261024T230409.781Z-8b40a8"
    session = epic / "sessions" / "pi" / "sessions" / ("-" + slug + "--")
    session.mkdir(parents=True)
    (epic / "epic.jsonl").write_text(json.dumps({
        "event": "began",
        "agents": [{"agent": "assistant", "backend": "pi", "model": "deepseek/deepseek-flash"}],
        "budget": {"cost": 15},
    }) + "\n")
    (session / "2026-10-24T23-04-10-012Z_6ba.jsonl").write_text(json.dumps({
        "message": {"role": "assistant", "content": [{"type": "text", "text": "coalesced epilogue stores"}]},
    }) + "\n")

    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(tmp_path / "runs"))
    empty = tmp_path / "no-pi-sessions"
    empty.mkdir()
    assert detail._epic_files(run, tmp_path / "epics") != []

    feed = detail.pi_feed(f"apple-m4-max/{run_name}", sessions_root=empty, epics_root=tmp_path / "epics")
    assert feed is not None and feed["kernel_writer"] == "pi"
    assert [row["who"] for row in feed["epics"]] == ["开始"]
    assert any("coalesced epilogue stores" in row["text"] for row in feed["lines"])
    assert detail.pi_feed(f"apple-m4-max/{run_name}", sessions_root=empty, epics_root=tmp_path / "epics", stamp=feed["stamp"])["same"] is True
