"""Offline paired-yield runner: AB/BA plan, human gates, and exact medians."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import uuid
from argparse import Namespace
from pathlib import Path

import pytest

from experiments.search_yield_ab import (
    CONTROL_OVERRIDES,
    TREATMENT_OVERRIDES,
    _execute_refusal,
    build_plan,
    execute_plan,
    main,
    summarize_experiment,
    write_plan,
)

ROOT = Path(__file__).resolve().parents[2]


def _scratch() -> Path:
    path = ROOT / "runs" / f"_plan_c_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _result(confirmed: float, *, best: float, spent: float, status: str = "ok", seed: int = 7) -> dict:
    return {
        "task": "conv2d",
        "dtype": "fp16",
        "size": "paper",
        "confirmed_status": status,
        "confirmed_speedup": confirmed if status == "ok" else None,
        "confirmation_source_sha256": "ab" * 32 if status == "ok" else None,
        "best_speedup": best,
        "spent_usd": spent,
        "candidates": [{"label": "init", "status": "ok", "speedup": 1.0}],
        "trace": [{
            "label": "init",
            "status": "ok",
            "improved": True,
            "cost_usd": 1.0,
            "eval_seed": seed,
            "eval_seed_from": "search",
        }],
    }


def _fill(out: Path, cell: dict, *, confirmed: float, best: float, spent: float, checks: int, unchanged: int, status: str = "ok") -> None:
    folder = out / cell["directory"]
    session = folder / "agent_work" / "s1" / ".selfcheck"
    session.mkdir(parents=True)
    (session / "self_checks.jsonl").write_text(("ok\n" * checks), encoding="utf-8")
    (folder / "result.json").write_text(
        json.dumps(_result(confirmed, best=best, spent=spent, status=status)),
        encoding="utf-8",
    )
    events = []
    if unchanged:
        events.append({"event": "agent", "label": "s1", "outcome": "unchanged", "self_checks": checks})
        events.append({"event": "agent", "label": "s2", "outcome": "evaluated", "self_checks": checks})
    else:
        events.append({"event": "agent", "label": "s1", "outcome": "evaluated", "self_checks": checks})
        events.append({"event": "agent", "label": "s2", "outcome": "evaluated", "self_checks": checks})
    (folder / "transcript.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


def test_dry_run_builds_ab_ba_twelve_cells_without_running(monkeypatch):
    """@brief 12 个 cell 按 AB/BA 交替，两臂开关与批准单一致，且不调用 runner。"""
    out = _scratch()
    try:
        def boom(*args, **kwargs):
            raise AssertionError("runner was called during dry-run")

        monkeypatch.setattr("experiments.common.run_one", boom)
        assert main(["--out", str(out)]) == 0
        plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        assert plan["git_head"] == head
        assert plan["control"] == CONTROL_OVERRIDES
        assert plan["treatment"] == TREATMENT_OVERRIDES
        assert plan["nominal_total_usd"] == 36.0
        assert plan["max_total_usd"] == 36.0
        assert len(plan["cells"]) == 12
        assert [plan["cells"][index]["order"] for index in range(0, 12, 2)] == ["AB", "BA", "AB", "BA", "AB", "BA"]
        for index in range(0, 12, 2):
            pair = plan["cells"][index:index + 2]
            expected = ["control", "treatment"] if pair[0]["order"] == "AB" else ["treatment", "control"]
            assert [cell["arm"] for cell in pair] == expected
            for cell in pair:
                overrides = CONTROL_OVERRIDES if cell["arm"] == "control" else TREATMENT_OVERRIDES
                assert cell["overrides"] == overrides
                assert cell["budget_usd"] == 3.0
        assert list(out.rglob("result.json")) == []
        assert main(["--out", str(ROOT / "runs" / ".." / "not-runs")]) == 2
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_execute_refuses_without_all_four_human_gates():
    """@brief 缺花费、缺 GPU、批准人是空或 agent、或缺 δ 时 runner 一次都不调用。"""
    calls = []

    def runner(*args, **kwargs):
        calls.append(args)

    complete = [
        "--out", "/tmp/search-yield-ab-should-not-start",
        "--execute",
        "--allow-spend",
        "--allow-gpu",
        "--approved-by", "Ada Lovelace",
        "--delta", "0.2",
    ]
    variants = [
        [item for item in complete if item != "--allow-spend"],
        [item for item in complete if item != "--allow-gpu"],
        ["" if item == "Ada Lovelace" else item for item in complete],
        ["agent" if item == "Ada Lovelace" else item for item in complete],
        ["cli" if item == "Ada Lovelace" else item for item in complete],
        ["auto" if item == "Ada Lovelace" else item for item in complete],
        [item for item in complete if item not in ("--delta", "0.2")],
    ]
    for argv in variants:
        assert main(argv, runner=runner) == 2
    assert calls == []
    for value in (float("nan"), float("inf"), float("-inf")):
        refusal = _execute_refusal(Namespace(
            allow_spend=True, allow_gpu=True, approved_by="Ada Lovelace", delta=value,
        ))
        assert refusal
    out = _scratch()
    try:
        for token in ("nan", "inf", "+inf", "NaN", "--delta=-inf"):
            argv = [
                "--out", str(out),
                "--execute",
                "--allow-spend",
                "--allow-gpu",
                "--approved-by", "Ada Lovelace",
            ]
            argv.extend(["--delta", token] if not token.startswith("--") else [token])
            assert main(argv, runner=runner) == 2
        assert calls == []
        assert not (out / "plan.json").exists()
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_summarize_computes_exact_paired_median_and_never_falls_back_to_best_speedup():
    """@brief 六对的中位数和四项次指标按批准单计算；未确认样本不拿 best_speedup 填。"""
    out = _scratch()
    try:
        plan = build_plan()
        write_plan(out, plan)
        for cell in plan["cells"]:
            if cell["arm"] == "control":
                _fill(out, cell, confirmed=1.0, best=1.5, spent=2.0, checks=2, unchanged=1)
            else:
                _fill(out, cell, confirmed=1.4, best=2.0, spent=4.0, checks=4, unchanged=0)
        summary = summarize_experiment(out, delta=0.1)
        assert summary["complete_pairs"] == 6
        assert summary["paired_differences"] == pytest.approx([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
        assert summary["median_difference"] == pytest.approx(0.4)
        assert summary["verdict"] == "treatment_wins"
        assert summary["control"]["lift_median"] == 0.5
        assert summary["control"]["unchanged_fraction"] == pytest.approx(0.5)
        assert summary["control"]["spend_after_best_median"] == pytest.approx(0.5)
        assert summary["control"]["self_checks_median"] == 2
        assert summary["treatment"]["lift_median"] == 1.0
        assert summary["treatment"]["unchanged_fraction"] == 0.0
        assert summary["treatment"]["spend_after_best_median"] == pytest.approx(0.75)
        assert summary["treatment"]["self_checks_median"] == 4
        treatment = next(cell for cell in plan["cells"] if cell["arm"] == "treatment" and cell["rep"] == 1 and cell["task"] == "conv2d")
        path = out / treatment["directory"] / "result.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["confirmed_status"] = "timing_uncertain"
        document["best_speedup"] = 9.9
        path.write_text(json.dumps(document), encoding="utf-8")
        broken = summarize_experiment(out, delta=0.1)
        assert broken["complete_pairs"] == 5
        assert broken["verdict"] == "inconclusive"
        assert len(broken["excluded_pairs"]) == 1
        assert "timing_uncertain" in broken["excluded_pairs"][0]["reason"]
        assert pytest.approx(8.9) not in broken["paired_differences"]
        assert broken["median_difference"] == pytest.approx(0.4)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_non_finite_or_non_positive_confirmed_speedup_is_excluded_and_cannot_win():
    """@brief NaN、inf、负数和 0 进排除名单，不能靠 inf 差判成 treatment_wins。"""
    out = _scratch()
    try:
        plan = build_plan(tasks=("conv2d",), pairs=1)
        assert plan["expected_pairs"] == 1
        write_plan(out, plan)
        control = next(cell for cell in plan["cells"] if cell["arm"] == "control")
        treatment = next(cell for cell in plan["cells"] if cell["arm"] == "treatment")
        control_doc = _result(1.0, best=1.5, spent=2.0)
        _fill(out, control, confirmed=1.0, best=1.5, spent=2.0, checks=1, unchanged=0)
        _fill(out, treatment, confirmed=1.0, best=9.9, spent=2.0, checks=1, unchanged=0)
        for bad in (float("inf"), float("-inf"), float("nan"), 0, 0.0, -0.2, True):
            document = _result(1.0, best=9.9, spent=2.0)
            document["confirmed_speedup"] = bad
            (out / treatment["directory"] / "result.json").write_text(
                json.dumps(document, allow_nan=True),
                encoding="utf-8",
            )
            summary = summarize_experiment(out, delta=0.1)
            assert summary["complete_pairs"] == 0
            assert summary["paired_differences"] == []
            assert summary["verdict"] == "inconclusive"
            assert summary["verdict"] != "treatment_wins"
            assert len(summary["excluded_pairs"]) == 1
            assert summary["excluded_pairs"][0]["task"] == "conv2d"
            assert "finite number greater than zero" in summary["excluded_pairs"][0]["reason"]
            assert "treatment" in summary["excluded_pairs"][0]["reason"]
        control_doc["confirmed_speedup"] = float("inf")
        control_doc["best_speedup"] = 9.9
        (out / control["directory"] / "result.json").write_text(
            json.dumps(control_doc, allow_nan=True),
            encoding="utf-8",
        )
        treatment_doc = _result(2.0, best=9.9, spent=2.0)
        (out / treatment["directory"] / "result.json").write_text(
            json.dumps(treatment_doc),
            encoding="utf-8",
        )
        control_bad = summarize_experiment(out, delta=0.1)
        assert control_bad["verdict"] == "inconclusive"
        assert control_bad["paired_differences"] == []
        assert "control" in control_bad["excluded_pairs"][0]["reason"]
        assert not any(math.isinf(item) for item in control_bad["paired_differences"])
        (out / control["directory"] / "result.json").write_text(
            json.dumps(_result(1.0, best=1.5, spent=2.0)),
            encoding="utf-8",
        )
        (out / treatment["directory"] / "result.json").write_text(
            json.dumps(_result(1.5, best=9.9, spent=2.0)),
            encoding="utf-8",
        )
        restored = summarize_experiment(out, delta=0.1)
        assert restored["complete_pairs"] == 1
        assert restored["median_difference"] == pytest.approx(0.5)
        assert restored["verdict"] == "treatment_wins"
        assert restored["excluded_pairs"] == []
        missing = _result(1.0, best=9.9, spent=2.0)
        del missing["confirmed_speedup"]
        (out / treatment["directory"] / "result.json").write_text(json.dumps(missing), encoding="utf-8")
        absent = summarize_experiment(out, delta=0.1)
        assert absent["complete_pairs"] == 0
        assert absent["verdict"] == "inconclusive"
        assert absent["paired_differences"] == []
        assert "best_speedup is not a substitute" in absent["excluded_pairs"][0]["reason"]
        assert 8.9 not in absent["paired_differences"]
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_execute_stops_at_max_total_usd_and_skips_occupied_cells():
    """@brief 已有文件的 cell 跳过；假花费顶到总额后不再启动后面的 cell。"""
    out = _scratch()
    try:
        plan = build_plan(budget_usd=3.0, max_total_usd=6.0)
        write_plan(out, plan)
        occupied = out / plan["cells"][0]["directory"]
        occupied.mkdir(parents=True)
        (occupied / "keep.txt").write_text("already here", encoding="utf-8")
        calls = []

        def runner(task, dtype, size, run_dir, *, budget=None, **overrides):
            calls.append(Path(run_dir).name)
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            (Path(run_dir) / "result.json").write_text(json.dumps({"spent_usd": budget, "budget_usd": budget}), encoding="utf-8")

        notes = execute_plan(out, plan, runner=runner)
        assert notes[0]["skipped"] == "occupied"
        assert calls == ["conv2d_paper_rep1", "conv2d_paper_rep2"]
        assert "conv2d_paper_rep1" == Path(plan["cells"][1]["directory"]).name
        assert all(note["directory"] != plan["cells"][3]["directory"] or note.get("skipped") for note in notes[1:])
        assert not (out / plan["cells"][3]["directory"]).exists()
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_approval_sheet_keeps_delta_and_signoff_blank():
    """@brief 批准单补了工具入口，δ 和执行批准栏仍是空位。"""
    text = (ROOT / "docs/实验批准单-搜索产出.md").read_text(encoding="utf-8")
    assert "experiments.task_headroom" in text
    assert "taic.v2_hw_inspect" in text
    assert "experiments.search_yield_ab" in text
    assert "最小差异 δ（数值、单位）：____" in text
    assert "允许退化：____" in text
    assert "owner 审阅与日期：____" in text
    assert "另行批准实际付费/GPU 执行的记录：____" in text
    assert "是否执行留空" in text
