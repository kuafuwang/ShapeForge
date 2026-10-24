"""``taic brief`` reads a run directory and prints facts. No GPU and no API key."""

from __future__ import annotations

import hashlib
import json

from taic import cli
from tcenv.schema import PROTOCOL_VERSION

SECRET = "sk-PLANTED-brief-secret-0123456789"
MARKER = "MSG-BODY-ONLY-9f3a"

_KEYS = {
    "schema_version", "protocol_version", "stamp",
    "task", "dtype", "size", "committee", "model", "phase", "stop_reason", "round",
    "spent_usd", "budget_usd", "published", "publication", "search_best",
    "winner_label", "winner_provider", "winner_model", "winner_committee", "winner_origin",
    "model_totals", "cands", "agent_failures", "counts", "paths",
}
_CAND = {"label", "status", "speedup", "within_noise", "differential_recheck", "origin"}


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
    (run / "state.json").write_text(json.dumps({
        "schema_version": 1, "task": "relu", "dtype": "fp16", "size": "small", "committee": "second",
        "model": "model-live", "phase": "evaluate", "stop_reason": "", "round": 3,
        "spent_usd": 0.5, "budget_usd": 2, "best_speedup": 1.5, "trace": trace,
    }))
    (run / "result.json").write_text(json.dumps({
        "schema_version": 1, "task": "relu", "dtype": "fp16", "size": "paper", "model": "model-old",
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

    code, out, _err = _brief(capsys, run)
    assert code == 0
    assert out.endswith("\n") and out.strip().count("\n") == 0
    assert '"published":null' in out and '"search_best":1.5' in out
    assert SECRET not in out and MARKER not in out
    doc = json.loads(out)
    assert set(doc) == _KEYS
    assert doc["task"] == "relu" and doc["dtype"] == "fp16" and doc["size"] == "small"
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
        (folder / "report.json").write_text(json.dumps({"status": "ok", "speedup": i, "message": f"{SECRET} {MARKER}"}))
    last = run / "candidates" / "008_c8"
    (last / "differential_recheck.json").write_text(json.dumps({"class": "timeout", "recheck_report": {"message": SECRET}}))

    code, out, _err = _brief(capsys, run)
    assert code == 0
    doc = json.loads(out)
    assert SECRET not in out and MARKER not in out
    assert doc["published"] is None and doc["publication"] is None and doc["search_best"] == 1.5
    assert [c["label"] for c in doc["cands"]] == [f"c{i}" for i in range(1, 9)]
    assert doc["cands"][-1]["status"] == "ok" and doc["cands"][-1]["speedup"] == 8
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
    (run / "confirm_report.json").write_text(json.dumps({
        "status": "ok", "speedup": 1.25,
        "benchmark": {"noisy": False, "candidate": {"cv": 0.01, "reps": 15}, "baselines": {"mlx": {"cv": 0.01, "reps": 15}}, "best_baseline": "mlx", "confirmation": {"kernel_sha256": digest, "winner_label": "win", "confirmed_values": [1.25], "speedup": 1.25, "candidate_us": None}},
        "protocol_version": PROTOCOL_VERSION, "target_id": "apple-m4-max",
    }))
    (run / "state.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": "small", "committee": "first", "model": "m",
        "phase": "done", "stop_reason": "max_rounds", "round": 2, "spent_usd": 0.5, "budget_usd": 2, "best_speedup": 9.5,
    }))
    (run / "result.json").write_text(json.dumps({
        "schema_version": 1, "task": "relu", "dtype": "fp16", "size": "small", "model": "m",
        "stop_reason": "max_rounds", "rounds": 2, "spent_usd": 0.25, "budget_usd": 2,
        "best_speedup": 8.5, "confirmed_speedup": 1.25, "confirmed_status": "ok",
        "confirm_winner_label": "win", "confirmation_source_sha256": digest,
        "confirmed_values": [1.25], "target_id": "apple-m4-max",
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
    (run / "state.json").write_text(json.dumps({"task": "relu", "best_speedup": 1.5, "round": 1}))
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
    assert doc["cands"] == [] and doc["agent_failures"] == []
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
    code, out, _err = _brief(capsys, run)
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
    (run / "state.json").write_text(json.dumps({"task": "relu"}))
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
    (run / "state.json").write_text(json.dumps({"task": "relu", "best_label": "r1.p0", "trace": trace}))
    (run / "result.json").write_text(json.dumps({
        "task": "relu", "best_label": "r1.p0", "winner_label": "r1.p0",
        "winner_provider": "moonshot", "winner_model": "kimi-k3", "winner_committee": "second", "winner_origin": "session",
        "trace": trace,
    }))
    (run / "transcript.jsonl").write_text("\n".join([
        json.dumps({"event": "agent", "label": "init", "candidate_label": "init", "provider": "deepseek", "model": "deepseek-flash", "committee": "first", "cost_usd": 0.2,
                    "tokens": {"input": 10, "output": 20, "cache_read": 30, "cache_write": 0}}),
        json.dumps({"event": "agent", "label": "r1.p0", "candidate_label": "r1.p0", "provider": "moonshot", "model": "kimi-k3", "committee": "second", "cost_usd": 1.5,
                    "tokens": {"input": 5, "output": 40, "cache_read": 50, "cache_write": 60}}),
    ]) + "\n")
    code, out, _err = _brief(capsys, run)
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
    (run / "state.json").write_text(json.dumps({"task": "relu", "best_label": "r1.p0", "trace": trace}))
    (run / "result.json").write_text(json.dumps({"task": "relu", "best_label": "r1.p0", "trace": trace}))
    code, out, _err = _brief(capsys, run)
    assert code == 0
    doc = json.loads(out)
    assert doc["winner_label"] == "r1.p0"
    assert (doc["winner_provider"], doc["winner_model"], doc["winner_committee"], doc["winner_origin"]) == (
        "moonshot", "kimi-k3", "second", "session",
    )
