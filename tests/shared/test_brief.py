"""``taic brief`` reads a run directory and prints facts. No GPU and no API key."""

from __future__ import annotations

import hashlib
import json

from taic import cli
from tcenv.schema import PROTOCOL_VERSION
from tests.support.v1_tasks import V1_TASK

SECRET = "sk-PLANTED-brief-secret-0123456789"
MARKER = "MSG-BODY-ONLY-9f3a"

# §6.1 的 L1 默认形状：候选清单、模型费用与完整失败列表要 ``--expand cands,models,failures``。
_KEYS = {
    "schema_version", "protocol_version", "stamp",
    "task", "dtype", "size", "committee", "model", "phase", "stop_reason", "round",
    "spent_usd", "budget_usd", "published", "publication", "search_best",
    "winner_label", "winner_provider", "winner_model", "winner_committee", "winner_origin",
    "candidate_us", "baseline_us", "last_failure", "counts", "paths", "pending_evaluation",
}
_EXPANDED_KEYS = _KEYS | {"model_totals", "cands", "agent_failures"}
_EXPAND = ("--expand", "cands,models,failures")
_CAND = {"label", "status", "speedup", "within_noise", "differential_recheck", "origin", "report_state"}


def _brief(capsys, run_dir=None, *extra):
    """@brief 调用 taic brief CLI 并返回退出码与捕获的输出。

    @param capsys pytest 的输出捕获器。
    @param run_dir 可选的 run 目录，None 表示不带目录参数。
    @param extra 追加给 brief 的额外参数。
    @return (退出码, stdout, stderr)。
    """
    argv = (["brief"] if run_dir is None else ["brief", str(run_dir)]) + list(extra)
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_a_run_dir_brief_keeps_search_best_off_the_published_number(tmp_path, capsys):
    """@brief 只有 state 时 published 为 null、search_best 保留 state 值，候选只输出最近 8 个且不泄露正文。

    @param tmp_path pytest 提供的临时目录。
    @param capsys pytest 的输出捕获器。
    """
    run = tmp_path / "run"
    run.mkdir()
    trace = []
    for i in range(10):
        trace.append({
            "label": f"c{i}", "status": "ok", "speedup": i, "within_noise": i % 2 == 0,
            "differential_recheck": "incorrect" if i == 9 else None,
            "message": f"{MARKER} {SECRET}",
        })
    # 每个 trace 行都要有对应的当前（schema-4）候选报告：trace 自带的 status/speedup 只有和
    # 报告绑定后才算证据（§8）。这里建目录，另外用一条负例测旧格式。
    for i in range(10):
        folder = run / "candidates" / f"{i:03d}_c{i}"
        folder.mkdir(parents=True)
        (folder / "kernel.metal").write_text("kernel void k() {}\n", encoding="utf-8")
        (folder / "report.json").write_text(json.dumps({
            "schema_version": 4, "status": "ok", "speedup": i,
        }), encoding="utf-8")
    (run / "state.json").write_text(json.dumps({
        "schema_version": 1, "task": V1_TASK, "dtype": "fp16", "size": "small", "committee": "second",
        "model": "model-live", "phase": "evaluate", "stop_reason": "", "round": 3,
        "spent_usd": 0.5, "budget_usd": 2, "best_speedup": 1.5, "trace": trace,
    }))
    (run / "result.json").write_text(json.dumps({
        "schema_version": 1, "task": V1_TASK, "dtype": "fp16", "size": "paper", "model": "model-old",
        "stop_reason": "max_rounds", "rounds": 9, "spent_usd": 0.25, "budget_usd": 15,
        "best_speedup": 9.5, "confirmed_speedup": 9.5, "confirmed_status": "incorrect",
        "confirm_winner_label": "c9", "confirmation_source_sha256": "abc",
        "trace": [{"label": "from-result", "status": "ok", "speedup": 9.5, "within_noise": False, "differential_recheck": None}],
    }))
    (run / "transcript.jsonl").write_text(
        "\n".join([
            json.dumps({"event": "chat", "label": "init", "messages": [{"role": "user", "content": f"{MARKER} {SECRET}"}], "reply": {"content": SECRET}}),
            json.dumps({"event": "retry", "label": "init", "attempt": 1, "error": "timeout"}),
            json.dumps({"event": "resume", "n": 1}),
        ]) + "\n"
    )
    (run / "eval_progress.json").write_text(json.dumps({"stage": "bench", "status": "running", "message": f"{MARKER} {SECRET}"}))

    code, out, _err = _brief(capsys, run, *_EXPAND)
    assert code == 0
    assert out.endswith("\n") and out.strip().count("\n") == 0
    assert '"published":null' in out and '"search_best":1.5' in out
    assert SECRET not in out and MARKER not in out
    doc = json.loads(out)
    assert set(doc) == _EXPANDED_KEYS
    assert doc["task"] == V1_TASK and doc["dtype"] == "fp16" and doc["size"] == "small"
    assert doc["committee"] == "second" and doc["model"] == "model-live" and doc["phase"] == "evaluate"
    assert doc["stop_reason"] == "" and doc["round"] == 3
    assert doc["spent_usd"] == 0.5 and doc["budget_usd"] == 2
    assert doc["published"] is None and doc["publication"] == "未确认" and doc["search_best"] == 1.5
    assert [c["label"] for c in doc["cands"]] == [f"c{i}" for i in range(2, 10)]
    assert all(set(c) == _CAND for c in doc["cands"])
    assert doc["cands"][-1]["differential_recheck"] == "incorrect"
    assert doc["cands"][0]["within_noise"] is True and doc["cands"][-1]["within_noise"] is False
    assert "from-result" not in out
    assert doc["counts"] == {"chat": 1, "retry": 1, "truncation_retries": 0, "eval_stage": "bench"}
    assert doc["paths"]["result"].endswith("result.json") and doc["paths"]["transcript"].endswith("transcript.jsonl")
    assert doc["paths"]["confirm"] is None and doc["paths"]["metal"] is None
    assert "gpu" not in doc


def test_candidate_dirs_supply_outcomes_when_the_trace_is_absent(tmp_path, capsys):
    """@brief trace 缺失时从 candidates 目录读结果和差分复检，屏蔽正文，计数类字段全为 null。

    @param tmp_path pytest 提供的临时目录。
    @param capsys pytest 的输出捕获器。
    """
    run = tmp_path / "run"
    (run / "state.json").parent.mkdir()
    (run / "state.json").write_text(json.dumps({
        "task": "gemm", "dtype": "bf16", "size": "paper", "committee": "first", "model": "m",
        "phase": "llm", "round": 1, "spent_usd": 0.5, "budget_usd": 2, "best_speedup": 1.5,
    }))
    for i in range(9):
        folder = run / "candidates" / f"{i:03d}_c{i}"
        folder.mkdir(parents=True)
        (folder / "kernel.metal").write_text(f"kernel void k() {{}}\n// {SECRET} {MARKER}\n")
        (folder / "report.json").write_text(json.dumps({
            "schema_version": 4, "status": "ok", "speedup": i, "message": f"{SECRET} {MARKER}",
        }))
    last = run / "candidates" / "008_c8"
    (last / "differential_recheck.json").write_text(json.dumps({"class": "timeout", "recheck_report": {"message": SECRET}}))

    code, out, _err = _brief(capsys, run, *_EXPAND)
    assert code == 0
    doc = json.loads(out)
    assert SECRET not in out and MARKER not in out
    assert doc["published"] is None and doc["publication"] is None and doc["search_best"] == 1.5
    assert [c["label"] for c in doc["cands"]] == [f"c{i}" for i in range(1, 9)]
    assert doc["cands"][-1]["status"] == "ok" and doc["cands"][-1]["speedup"] == 8
    assert doc["cands"][-1]["report_state"] == "current"
    assert doc["cands"][-1]["within_noise"] is None and doc["cands"][-1]["differential_recheck"] == "timeout"
    assert doc["counts"] == {"chat": None, "retry": None, "truncation_retries": None, "eval_stage": None}
    assert "kernel void" not in out


def test_a_bound_confirmation_publishes_the_remeasure_not_the_search_best(tmp_path, capsys):
    """@brief 确认门绑定源码哈希时发布重新测量的值，而不是 search best。

    @param tmp_path pytest 提供的临时目录。
    @param capsys pytest 的输出捕获器。
    """
    source = f"kernel void k() {{}}\n// {SECRET}\n"
    digest = hashlib.sha256(source.encode()).hexdigest()
    run = tmp_path / "run"
    (run / "candidates" / "000_win").mkdir(parents=True)
    (run / "best.metal").write_text(source)
    (run / "candidates" / "000_win" / "kernel.metal").write_text(source)
    batch = {
        "label": "win", "shape": "full", "build_kind": "native", "measurement_kind": "measured",
        "cache_hit": False, "timing_uncertain": False,
        "identity": {"kind": "match", "differences": [], "shape": "full", "build_kind": "native",
                     "measurement_kind": "measured", "cache_hit": False, "timing_uncertain": False},
    }
    from ..support.report_build import confirmation_report

    (run / "confirm_report.json").write_text(json.dumps(confirmation_report(
        V1_TASK, "fp16", "small", source=source.encode(), label="win", values=[1.25], speedup=1.25,
    )))
    (run / "state.json").write_text(json.dumps({
        "task": V1_TASK, "dtype": "fp16", "size": "small", "committee": "first", "model": "m",
        "phase": "done", "stop_reason": "max_rounds", "round": 2, "spent_usd": 0.5, "budget_usd": 2, "best_speedup": 9.5,
    }))
    (run / "result.json").write_text(json.dumps({
        "schema_version": 1, "task": V1_TASK, "dtype": "fp16", "size": "small", "model": "m",
        "stop_reason": "max_rounds", "rounds": 2, "spent_usd": 0.25, "budget_usd": 2,
        "best_speedup": 8.5, "confirmed_speedup": 1.25, "confirmed_status": "ok",
        "confirm_winner_label": "win", "confirmation_source_sha256": digest,
        "confirmed_values": [1.25], "target_id": "apple-m4-max",
        "confirmations": [dict(batch, status="ok", failed_batches=0)],
    }))

    code, out, _err = _brief(capsys, run)
    assert code == 0
    doc = json.loads(out)
    assert SECRET not in out
    assert doc["published"] == 1.25 and doc["publication"] == "confirmed" and doc["search_best"] == 9.5
    assert doc["paths"]["metal"].endswith("best.metal") and doc["paths"]["confirm"].endswith("confirm_report.json")


def test_a_result_without_the_hash_field_does_not_publish_its_best(tmp_path, capsys):
    """@brief 旧版 result 缺少源码哈希字段时不发布 best，publication 标为旧版未绑定源码。

    @param tmp_path pytest 提供的临时目录。
    @param capsys pytest 的输出捕获器。
    """
    run = tmp_path / "only"
    run.mkdir()
    (run / "result.json").write_text(json.dumps({
        "schema_version": 1, "task": "gemm", "dtype": "bf16", "size": "paper", "model": "m",
        "stop_reason": "patience", "rounds": 4, "spent_usd": 0.5, "budget_usd": 2, "best_speedup": 3.5,
    }))
    code, out, _err = _brief(capsys, run)
    assert code == 0
    doc = json.loads(out)
    assert doc["published"] is None and doc["publication"] == "旧版未绑定源码"
    assert doc["search_best"] == 3.5 and doc["round"] == 4 and doc["committee"] is None and doc["phase"] is None


def test_a_damaged_result_is_unreadable_and_its_bytes_stay_out(tmp_path, capsys):
    """@brief result.json 损坏时标为 unreadable，不输出其字节，仍保留 state 里的 search_best。

    @param tmp_path pytest 提供的临时目录。
    @param capsys pytest 的输出捕获器。
    """
    run = tmp_path / "bad"
    run.mkdir()
    (run / "state.json").write_text(json.dumps({"task": V1_TASK, "best_speedup": 1.5, "round": 1}))
    (run / "result.json").write_text("{" + SECRET)
    code, out, _err = _brief(capsys, run)
    assert code == 0
    doc = json.loads(out)
    assert SECRET not in out
    assert doc["published"] is None and doc["publication"] == "unreadable" and doc["search_best"] == 1.5
    assert doc["paths"]["result"].endswith("result.json")


def test_an_empty_directory_is_nulls(tmp_path, capsys):
    """@brief 空目录输出固定结构且所有事实字段都是 null。

    @param tmp_path pytest 提供的临时目录。
    @param capsys pytest 的输出捕获器。
    """
    run = tmp_path / "empty"
    run.mkdir()
    code, out, _err = _brief(capsys, run)
    assert code == 0
    doc = json.loads(out)
    assert set(doc) == _KEYS
    assert doc["task"] is None and doc["published"] is None and doc["publication"] is None and doc["search_best"] is None
    assert doc["last_failure"] is None and doc["candidate_us"] is None and doc["baseline_us"] is None
    assert "cands" not in doc and "agent_failures" not in doc and "model_totals" not in doc
    assert doc["counts"] == {"chat": None, "retry": None, "truncation_retries": None, "eval_stage": None}
    assert all(v is None for v in doc["paths"].values())


def test_agent_failures_expose_structured_causes_but_not_response_bodies(tmp_path, capsys):
    """@brief agent_failures 暴露结构化失败原因，但脱敏错误正文和 response_preview。

    @param tmp_path pytest 提供的临时目录。
    @param capsys pytest 的输出捕获器。
    """
    run = tmp_path / "run"
    run.mkdir()
    (run / "state.json").write_text(json.dumps({"task": "conv2d", "phase": "done", "stop_reason": "agent_failure"}))
    rows = [
        {"event": "agent", "label": "r3.p0", "provider": "deepseek", "model": "deepseek-flash", "committee": "first",
         "outcome": "provider_failed", "error": "Connection error.", "had_kernel": False, "cost_usd": 0.1,
         "runtime": {"failure_class": "turn_failed", "failed_turns": 4, "last_terminal_kind": "failed"}},
        {"event": "agent", "label": "r4.p0.retry1", "candidate_label": "r4.p0", "attempt": 1,
         "provider": "moonshot", "model": "kimi-k3.1-pro", "committee": "second",
         "outcome": "agent_no_submission", "error": SECRET, "had_kernel": False, "cost_usd": 0.2,
         "runtime": {"failure_class": "no_submission_after_result", "failed_turns": 1,
                     "last_terminal_kind": "result", "response_preview": MARKER},
         "diagnostics_path": "agent_diagnostics/r4.json"},
    ]
    (run / "transcript.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    code, out, _err = _brief(capsys, run, *_EXPAND)
    assert code == 0 and MARKER not in out and SECRET not in out
    failures = json.loads(out)["agent_failures"]
    assert len(failures) == 2
    assert failures[0]["failure_class"] == "turn_failed" and failures[0]["failed_turns"] == 4
    assert failures[0]["error"] == "Connection error." and failures[0]["had_kernel"] is False
    assert failures[1]["failure_class"] == "no_submission_after_result"
    assert failures[1]["last_terminal_kind"] == "result" and failures[1]["error"] == "<redacted>"
    assert failures[1]["diagnostics_path"] == "agent_diagnostics/r4.json"
    assert failures[1]["candidate_label"] == "r4.p0" and failures[1]["attempt"] == 1


def test_gpu_events_are_the_cache_tail_and_the_lane_is_omitted_when_absent(tmp_path, capsys):
    """@brief 没有 GPU 事件时不输出 gpu 字段；--gpu 只取缓存尾部 8 条，并脱敏 what、丢弃 note。

    @param tmp_path pytest 提供的临时目录。
    @param capsys pytest 的输出捕获器。
    """
    run = tmp_path / "run"
    run.mkdir()
    (run / "state.json").write_text(json.dumps({"task": V1_TASK}))
    code, out, _err = _brief(capsys, run)
    assert code == 0 and "gpu" not in json.loads(out)

    from tcenv.events import gpu_events_path

    path = gpu_events_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    code, out, _err = _brief(capsys, run, "--gpu")
    assert json.loads(out)["gpu"] == []

    lines = []
    for i in range(10):
        what = SECRET if i == 9 else ("k" * 200 if i == 8 else "benchmark")
        lines.append(json.dumps({"t": i, "pid": 10 + i, "kind": f"k{i}", "mode": "ex", "what": what, "note": SECRET}))
    path.write_text("\n".join(lines) + "\n")
    code, out, _err = _brief(capsys, run, "--gpu")
    doc = json.loads(out)
    assert SECRET not in out and "note" not in out
    assert [row["kind"] for row in doc["gpu"]] == [f"k{i}" for i in range(2, 10)]
    assert doc["gpu"][-1]["what"] == "<redacted>" and doc["gpu"][-2]["what"] is None
    assert doc["gpu"][-1]["pid"] == 19 and set(doc["gpu"][-1]) == {"t", "pid", "kind", "mode", "what"}


def test_orientation_mentions_committees_and_skips_a_planted_key(monkeypatch, capsys):
    """@brief 无 run 的 orientation 说明委员会规则，不泄露环境里的密钥和地址，且不超过 80 行。

    @param monkeypatch pytest 补丁工具，用于植入伪造的密钥和地址。
    @param capsys pytest 的输出捕获器。
    """
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://secret.example.invalid/v1")
    code, out, _err = _brief(capsys)
    assert code == 0
    assert "committees" in out and "one new hmz session" in out and "agent_failed" in out
    assert SECRET not in out and "secret.example.invalid" not in out and "sk-" not in out
    assert len(out.splitlines()) <= 80


def test_a_missing_directory_is_refused(tmp_path, capsys):
    """@brief 目录不存在时退出码 2，stdout 为空，stderr 说明不是目录。

    @param tmp_path pytest 提供的临时目录。
    @param capsys pytest 的输出捕获器。
    """
    code, out, err = _brief(capsys, tmp_path / "missing")
    assert code == 2 and out == "" and "not a directory" in err


def test_brief_reports_winner_and_model_totals_with_old_run_fallback(tmp_path, capsys):
    """@brief winner 字段与按模型汇总的费用、会话数来自 result 和 transcript，旧 run 也能回退推导。

    @param tmp_path pytest 提供的临时目录。
    @param capsys pytest 的输出捕获器。
    """
    run = tmp_path / "run"
    run.mkdir()
    trace = [
        {"label": "init", "status": "ok", "speedup": 1.0, "within_noise": False,
         "differential_recheck": None, "origin": {"origin": "session", "provider": "deepseek", "model": "deepseek-flash", "committee": "first", "attempt": 0}},
        {"label": "r1.p0", "status": "ok", "speedup": 1.2, "within_noise": False,
         "differential_recheck": None, "origin": {"origin": "session", "provider": "moonshot", "model": "kimi-k3", "committee": "second", "attempt": 0}},
    ]
    (run / "state.json").write_text(json.dumps({"task": V1_TASK, "best_label": "r1.p0", "trace": trace}))
    (run / "result.json").write_text(json.dumps({
        "task": V1_TASK, "best_label": "r1.p0", "winner_label": "r1.p0",
        "winner_provider": "moonshot", "winner_model": "kimi-k3", "winner_committee": "second", "winner_origin": "session",
        "trace": trace,
    }))
    (run / "transcript.jsonl").write_text("\n".join([
        json.dumps({"event": "agent", "label": "init", "candidate_label": "init", "provider": "deepseek", "model": "deepseek-flash", "committee": "first", "cost_usd": 0.2,
                    "tokens": {"input": 10, "output": 20, "cache_read": 30, "cache_write": 0}}),
        json.dumps({"event": "agent", "label": "r1.p0", "candidate_label": "r1.p0", "provider": "moonshot", "model": "kimi-k3", "committee": "second", "cost_usd": 1.5,
                    "tokens": {"input": 5, "output": 40, "cache_read": 50, "cache_write": 60}}),
    ]) + "\n")
    code, out, _err = _brief(capsys, run, *_EXPAND)
    assert code == 0
    doc = json.loads(out)
    assert doc["winner_model"] == "kimi-k3" and doc["winner_provider"] == "moonshot"
    assert doc["winner_origin"] == "session" and doc["winner_label"] == "r1.p0"
    assert doc["cands"][0]["origin"]["model"] == "deepseek-flash"
    assert [row["model"] for row in doc["model_totals"]] == ["kimi-k3", "deepseek-flash"]
    assert doc["model_totals"][0]["cost_usd"] == 1.5 and doc["model_totals"][0]["sessions"] == 1


def test_brief_derives_winner_model_from_candidate_origin_without_new_result_fields(tmp_path, capsys):
    """@brief 老 result 没有 winner_* 字段时，从候选 origin 推导 provider、model、committee 和来源。

    @param tmp_path pytest 提供的临时目录。
    @param capsys pytest 的输出捕获器。
    """
    run = tmp_path / "legacy"
    run.mkdir()
    trace = [{"label": "r1.p0", "status": "ok", "speedup": 1.2, "within_noise": False, "differential_recheck": None,
              "origin": {"origin": "session", "provider": "moonshot", "model": "kimi-k3", "committee": "second", "attempt": 0}}]
    (run / "state.json").write_text(json.dumps({"task": V1_TASK, "best_label": "r1.p0", "trace": trace}))
    (run / "result.json").write_text(json.dumps({"task": V1_TASK, "best_label": "r1.p0", "trace": trace}))
    code, out, _err = _brief(capsys, run)
    assert code == 0
    doc = json.loads(out)
    assert doc["winner_label"] == "r1.p0"
    assert (doc["winner_provider"], doc["winner_model"], doc["winner_committee"], doc["winner_origin"]) == (
        "moonshot", "kimi-k3", "second", "session",
    )


def test_brief_shows_pending_evaluation_without_report_body(tmp_path):
    """@brief brief 只显示 outbox 状态和错误码，不泄露候选源码或报告正文。"""
    run = tmp_path / "run"
    outbox = run / "evaluation_outbox"
    outbox.mkdir(parents=True)
    (run / "state.json").write_text(json.dumps({"task": V1_TASK, "stop_reason": "evaluation_unavailable"}))
    (outbox / "request.json").write_text(json.dumps({
        "request_id": "ev-1", "status": "unavailable", "error": {"code": "transport_closed", "message": "secret body", "retryable": True},
        "candidate_sha256": "a" * 64,
    }))
    from taic.brief import summarize_run
    doc = summarize_run(run)
    assert doc["pending_evaluation"]["count"] == 1
    assert doc["pending_evaluation"]["requests"][0]["request_id"] == "ev-1"
    assert doc["pending_evaluation"]["requests"][0]["error_code"] == "transport_closed"
    assert "secret body" not in json.dumps(doc)


def test_an_old_format_candidate_report_is_marked_undecidable(tmp_path, capsys):
    """schema-3 候选报告不是当前证据：brief 标 ``old_format``，不照常给 ``ok`` 和数值。

    The report still holds its own historical ``status``/``speedup``; those are that run's memory of
    a measurement this reader cannot check (no build identity, no evaluation identity), so the row
    says so instead of presenting them as today's outcome.
    """
    run = tmp_path / "run"
    (run / "state.json").parent.mkdir()
    (run / "state.json").write_text(json.dumps({
        "task": "gemm", "dtype": "bf16", "size": "paper", "phase": "llm", "round": 1,
        "spent_usd": 0.5, "budget_usd": 2, "best_speedup": 1.5,
    }))
    folder = run / "candidates" / "000_old"
    folder.mkdir(parents=True)
    (folder / "kernel.metal").write_text("kernel void k() {}\n")
    (folder / "report.json").write_text(json.dumps({"schema_version": 3, "status": "ok", "speedup": 9.9}))
    broken = run / "candidates" / "001_broken"
    broken.mkdir(parents=True)
    (broken / "kernel.metal").write_text("kernel void k() {}\n")
    (broken / "report.json").write_text("{not json")
    empty = run / "candidates" / "002_empty"
    empty.mkdir(parents=True)
    (empty / "kernel.metal").write_text("kernel void k() {}\n")

    code, out, _err = _brief(capsys, run, *_EXPAND)
    assert code == 0
    rows = {row["label"]: row for row in json.loads(out)["cands"]}
    assert rows["old"]["report_state"] == "old_format"
    assert rows["old"]["status"] == "old_format" and rows["old"]["speedup"] is None
    assert rows["broken"]["report_state"] == "unreadable" and rows["broken"]["speedup"] is None
    assert rows["empty"]["report_state"] == "missing" and rows["empty"]["speedup"] is None


def test_a_trace_row_cannot_carry_an_old_reports_numbers(tmp_path, capsys):
    """trace 主路径也要与候选报告的 current 性绑定：旧报告不能借 trace 输出 ``ok 9.9``。

    The reviewer's repro: a schema-3 candidate plus ``result.trace=[{status:'ok',speedup:9.9}]``
    returned ``status='ok', speedup=9.9`` from ``summarize_run(..., expand={'cands'})``, because the
    trace path never read the candidate report.  Every trace row is now bound to its candidate's
    report state; a missing report is undecidable too, not ``ok``.
    """
    run = tmp_path / "run"
    run.mkdir()
    old = run / "candidates" / "000_r1.p0"
    old.mkdir(parents=True)
    (old / "kernel.metal").write_text("kernel void k() {}\n", encoding="utf-8")
    (old / "report.json").write_text(json.dumps({"schema_version": 3, "status": "ok", "speedup": 9.9}))
    (run / "state.json").write_text(json.dumps({
        "schema_version": 1, "task": V1_TASK, "dtype": "fp16", "size": "small", "phase": "done",
        "spent_usd": 0.5, "budget_usd": 2, "best_speedup": 1.5,
        "trace": [
            {"label": "r1.p0", "status": "ok", "speedup": 9.9, "within_noise": False},
            {"label": "ghost", "status": "ok", "speedup": 8.8, "within_noise": False},
        ],
    }))
    (run / "result.json").write_text(json.dumps({
        "schema_version": 1, "task": V1_TASK, "dtype": "fp16", "size": "small",
        "stop_reason": "max_rounds", "spent_usd": 0.5, "budget_usd": 2, "best_speedup": 1.5,
    }))

    code, out, _err = _brief(capsys, run, *_EXPAND)
    assert code == 0
    rows = {row["label"]: row for row in json.loads(out)["cands"]}
    assert rows["r1.p0"]["report_state"] == "old_format"
    assert rows["r1.p0"]["status"] == "old_format" and rows["r1.p0"]["speedup"] is None
    # No candidate directory at all: undecidable, not "ok".
    assert rows["ghost"]["report_state"] == "missing"
    assert rows["ghost"]["status"] == "missing" and rows["ghost"]["speedup"] is None
    assert "9.9" not in out and "8.8" not in out
