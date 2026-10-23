"""``taic brief`` reads a run directory and prints facts. No GPU and no API key."""

from __future__ import annotations

import hashlib
import json

from taic import cli

SECRET = "sk-PLANTED-brief-secret-0123456789"
MARKER = "MSG-BODY-ONLY-9f3a"

_KEYS = {
    "schema_version", "protocol_version", "stamp",
    "task", "dtype", "size", "committee", "model", "phase", "stop_reason", "round",
    "spent_usd", "budget_usd", "published", "publication", "search_best",
    "cands", "counts", "paths",
}
_CAND = {"label", "status", "speedup", "within_noise", "differential_recheck"}


def _brief(capsys, run_dir=None, *extra):
    argv = (["brief"] if run_dir is None else ["brief", str(run_dir)]) + list(extra)
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_a_run_dir_brief_keeps_search_best_off_the_published_number(tmp_path, capsys):
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
    source = f"kernel void k() {{}}\n// {SECRET}\n"
    digest = hashlib.sha256(source.encode()).hexdigest()
    run = tmp_path / "run"
    (run / "candidates" / "000_win").mkdir(parents=True)
    (run / "best.metal").write_text(source)
    (run / "candidates" / "000_win" / "kernel.metal").write_text(source)
    (run / "confirm_report.json").write_text(json.dumps({
        "status": "ok", "speedup": 7.5,
        "benchmark": {"noisy": False, "candidate": {"cv": 0.01, "reps": 15}, "baselines": {"mlx": {"cv": 0.01, "reps": 15}}, "best_baseline": "mlx"},
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
    }))

    code, out, _err = _brief(capsys, run)
    assert code == 0
    doc = json.loads(out)
    assert SECRET not in out
    assert doc["published"] == 1.25 and doc["publication"] == "confirmed" and doc["search_best"] == 9.5
    assert doc["paths"]["metal"].endswith("best.metal") and doc["paths"]["confirm"].endswith("confirm_report.json")


def test_a_result_without_the_hash_field_does_not_publish_its_best(tmp_path, capsys):
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
    run = tmp_path / "empty"
    run.mkdir()
    code, out, _err = _brief(capsys, run)
    assert code == 0
    doc = json.loads(out)
    assert set(doc) == _KEYS
    assert doc["task"] is None and doc["published"] is None and doc["publication"] is None and doc["search_best"] is None
    assert doc["cands"] == [] and doc["counts"] == {"chat": None, "retry": None, "truncation_retries": None, "eval_stage": None}
    assert all(v is None for v in doc["paths"].values())


def test_gpu_events_are_the_cache_tail_and_the_lane_is_omitted_when_absent(tmp_path, capsys):
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
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://secret.example.invalid/v1")
    code, out, _err = _brief(capsys)
    assert code == 0
    assert "committees" in out and "one new hmz session" in out and "agent_failed" in out
    assert SECRET not in out and "secret.example.invalid" not in out and "sk-" not in out
    assert len(out.splitlines()) <= 80


def test_a_missing_directory_is_refused(tmp_path, capsys):
    code, out, err = _brief(capsys, tmp_path / "missing")
    assert code == 2 and out == "" and "not a directory" in err
