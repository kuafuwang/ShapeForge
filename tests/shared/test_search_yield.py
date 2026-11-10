"""Yield medians from a fixture tree. The real runs/ directory is not read."""

from __future__ import annotations

import json

import pytest

from experiments.search_yield import main, summarize


def _dump(path, doc):
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_four_medians_follow_the_documented_formulas(tmp_path):
    """@brief 两份完整运行加一份没有合格 init 的运行，四个中位数按脚本里的公式取。

    抬升是 0.5 和 0，中位数 0.25。原样交回按场次汇总是 1/4；按每次运行再取中位数才是 1/6。
    最优之后的花费是 0.75 和 0，中位数 0.375。自测行数 2、4、1，中位数 2。
    """
    run_a = tmp_path / "a"
    (run_a / "agent_work" / "s1").mkdir(parents=True)
    # The winning row is not the last row. Final cumulative cost is 4, last improved cost is 1.
    (run_a / "result.json").write_text(json.dumps({
        "best_speedup": 3.0,
        "candidates": [
            {"label": "init", "status": "ok", "speedup": 2.0},
            {"label": "r1.p0", "status": "ok", "speedup": 3.0},
        ],
        "trace": [
            {"label": "init", "status": "ok", "improved": True, "cost_usd": 1.0},
            {"label": "r1.p0", "status": "ok", "improved": True, "cost_usd": 1.0},
            {"label": "r2.p0", "status": "evaluated", "improved": False, "cost_usd": 4.0},
        ],
    }), encoding="utf-8")
    (run_a / "transcript.jsonl").write_text(
        "\n".join([
            json.dumps({"event": "agent", "outcome": "evaluated", "self_checks": 9}),
            json.dumps({"event": "agent", "outcome": "unchanged", "self_checks": 9}),
            json.dumps({"event": "agent", "outcome": "evaluated", "self_checks": 9}),
        ]) + "\n",
        encoding="utf-8",
    )
    (run_a / "agent_work" / "s1" / "self_checks.jsonl").write_text('{"status":"ok"}\n{"status":"ok"}\n', encoding="utf-8")

    run_b = tmp_path / "b"
    (run_b / "agent_work" / "s1").mkdir(parents=True)
    _dump(run_b / "result.json", {
        "best_speedup": 2.0,
        "candidates": [{"label": "init2", "status": "ok", "speedup": 2.0}],
        "trace": [{"label": "init2", "status": "ok", "improved": True, "cost_usd": 2.0}],
    })
    (run_b / "transcript.jsonl").write_text(
        json.dumps({"event": "agent", "outcome": "evaluated", "self_checks": 0}) + "\n",
        encoding="utf-8",
    )
    (run_b / "agent_work" / "s1" / "self_checks.jsonl").write_text("1\n2\n3\n4\n", encoding="utf-8")

    run_c = tmp_path / "c"
    (run_c / "agent_work" / "s1").mkdir(parents=True)
    _dump(run_c / "result.json", {
        "best_speedup": 1.0,
        "candidates": [{"label": "init", "status": "compile_error", "speedup": None}],
        "trace": [],
    })
    (run_c / "agent_work" / "s1" / "self_checks.jsonl").write_text('{"status":"ok"}\n', encoding="utf-8")

    summary = summarize(tmp_path)
    assert summary["n"] == 3
    assert summary["lift_median"] == 0.25 and summary["lift_n"] == 2
    assert summary["unchanged_fraction"] == pytest.approx(1 / 4)
    assert summary["unchanged_count"] == 1 and summary["unchanged_sessions"] == 4
    assert summary["unchanged_fraction_per_run_median"] == pytest.approx((1 / 3 + 0) / 2)
    assert summary["spend_after_best_median"] == 0.375 and summary["spend_n"] == 2
    assert summary["self_checks_median"] == 2 and summary["self_checks_n"] == 3


def test_spend_after_best_counts_paid_sessions_that_wrote_no_trace_row(tmp_path):
    """@brief 最好内核之后的付费失败没有 trace 行。分母用 ``spent_usd``，不再是最后一行 trace。"""
    run = tmp_path / "late"
    run.mkdir()
    spent = 3.736662136
    improved = 3.538604912
    _dump(run / "result.json", {
        "best_speedup": 2.0,
        "spent_usd": spent,
        "candidates": [{"label": "init", "status": "ok", "speedup": 2.0}],
        "trace": [{"label": "r3.p0", "status": "ok", "improved": True, "cost_usd": improved}],
    })
    (run / "transcript.jsonl").write_text(
        "\n".join([
            json.dumps({"event": "agent", "label": "r4.p0", "outcome": "agent_failed", "cost_usd": 0.1}),
            json.dumps({"event": "agent", "label": "r5.p0", "outcome": "agent_failed", "cost_usd": 0.098057224}),
        ]) + "\n",
        encoding="utf-8",
    )
    summary = summarize(tmp_path)
    assert summary["spend_n"] == 1
    assert summary["spend_after_best_median"] == pytest.approx((spent - improved) / spent)
    assert summary["spend_after_best_median"] > 0


def test_an_empty_root_returns_null_medians(tmp_path, capsys):
    """@brief 空目录的中位数是 null，退出码 0。"""
    summary = summarize(tmp_path)
    assert summary["n"] == 0
    assert summary["lift_median"] is None
    assert summary["unchanged_fraction"] is None
    assert summary["unchanged_sessions"] == 0
    assert summary["spend_after_best_median"] is None
    assert summary["self_checks_median"] is None
    assert main(["--root", str(tmp_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["n"] == 0 and printed["lift_median"] is None


def _result(path, **doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"best_speedup": 1.0, "candidates": [], "trace": [], **doc}), encoding="utf-8")


def test_uneven_runs_pool_sessions_instead_of_medians_of_run_totals(tmp_path):
    """@brief 不等长运行：原样交回是 1/12，自测中位数是每场的 2，不是每次运行合计的 51.5。"""
    short = tmp_path / "short"
    (short / "agent_work" / "s1").mkdir(parents=True)
    (short / "agent_work" / "s2").mkdir()
    _result(short / "result.json")
    (short / "transcript.jsonl").write_text(
        "\n".join([
            json.dumps({"event": "agent", "label": "s1", "outcome": "unchanged"}),
            json.dumps({"event": "agent", "label": "s2", "outcome": "evaluated"}),
        ]) + "\n",
        encoding="utf-8",
    )
    (short / "agent_work" / "s1" / "self_checks.jsonl").write_text("1\n", encoding="utf-8")
    (short / "agent_work" / "s2" / "self_checks.jsonl").write_text("\n".join("x" for _ in range(100)) + "\n", encoding="utf-8")

    long = tmp_path / "long"
    (long / "agent_work" / "only").mkdir(parents=True)
    _result(long / "result.json")
    events = [{"event": "agent", "label": f"m{i}", "outcome": "evaluated"} for i in range(10)]
    (long / "transcript.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    (long / "agent_work" / "only" / "self_checks.jsonl").write_text("1\n2\n", encoding="utf-8")

    summary = summarize(tmp_path)
    assert summary["unchanged_fraction"] == pytest.approx(1 / 12)
    assert summary["unchanged_sessions"] == 12
    assert summary["unchanged_fraction_per_run_median"] == pytest.approx((0.5 + 0) / 2)
    assert summary["self_checks_median"] == 2
    assert summary["self_checks_n"] == 3
    assert summary["self_checks_per_run_median"] == pytest.approx((101 + 2) / 2)


def test_nested_runs_are_found_and_self_checks_follow_one_session(tmp_path):
    """@brief 深层 result.json 也计入。同一场新旧文件只算新文件；一场旧文件不影响其他场和仅归档的场。"""
    deep = tmp_path / "target" / "day" / "nested" / "run"
    _result(deep / "result.json")
    root_run = tmp_path / "result.json"
    _result(root_run)
    also = tmp_path / "a" / "b" / "c"
    _result(also / "result.json")

    mixed = tmp_path / "mixed"
    _result(mixed / "result.json")
    old_only = mixed / "agent_work" / "old-only"
    both = mixed / "agent_work" / "both"
    fresh_only = mixed / "agent_work" / "fresh-only"
    old_only.mkdir(parents=True)
    both.mkdir(parents=True)
    fresh_only.mkdir(parents=True)
    (old_only / "self_checks.jsonl").write_text("1\n2\n3\n", encoding="utf-8")
    (both / "self_checks.jsonl").write_text("\n".join("old" for _ in range(9)) + "\n", encoding="utf-8")
    (both / ".selfcheck").mkdir()
    (both / ".selfcheck" / "self_checks.jsonl").write_text("1\n2\n", encoding="utf-8")
    (fresh_only / ".selfcheck").mkdir()
    (fresh_only / ".selfcheck" / "self_checks.jsonl").write_text("1\n", encoding="utf-8")
    archived = mixed / "candidates" / "004_archive-only"
    archived.mkdir(parents=True)
    (archived / "self_checks.jsonl").write_text("1\n2\n3\n4\n", encoding="utf-8")
    mirrored = mixed / "candidates" / "005_both"
    mirrored.mkdir()
    (mirrored / "self_checks.jsonl").write_text("\n".join("mirror" for _ in range(7)) + "\n", encoding="utf-8")
    (mixed / "transcript.jsonl").write_text(
        "\n".join([
            json.dumps({"event": "agent", "label": "both", "outcome": "evaluated", "self_checks": 9}),
            json.dumps({"event": "agent", "label": "event-only", "outcome": "evaluated", "self_checks": 5}),
        ]) + "\n",
        encoding="utf-8",
    )

    archive_run = tmp_path / "archive"
    _result(archive_run / "result.json")
    slot = archive_run / "candidates" / "001_init"
    slot.mkdir(parents=True)
    (slot / "self_checks.jsonl").write_text("1\n2\n", encoding="utf-8")

    summary = summarize(tmp_path)
    assert summary["n"] == 5
    mixed_counts = sorted([
        3,  # old-only
        2,  # both: the new file, not 9 and not 9+2
        1,  # fresh-only
        4,  # archive-only
        5,  # event-only
    ])
    # The other runs contribute the archive-only init session of 2 lines. Pool every session.
    assert summary["self_checks_n"] == 6
    assert summary["self_checks_median"] == pytest.approx(_median_of([2, *mixed_counts]))


def _median_of(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
