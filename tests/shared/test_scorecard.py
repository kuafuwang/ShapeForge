"""Scorecard order, and TASK.md rendering for every discovered v2 task. No GPU and no model."""

from __future__ import annotations

from pathlib import Path

from taic.kernel_agent import SessionBrief, render_contract
from taic.task_binding import bind_task
from tcenv.evaluate import EvalReport
from tcenv.report import format_feedback, format_scorecard
from tcenv.target import load_target
from tcenv.taskcontract.locator import discover_standard_tasks

ROOT = Path(__file__).resolve().parents[2]
STYLES = ("msl-single", "msl-bundle", "tilelang-pipeline")


def _report(**kw) -> EvalReport:
    base = dict(task="conv2d", dtype="fp16", size="paper", mode="full", status="ok", stage="done", message="ok")
    base.update(kw)
    return EvalReport(**base)


def _bench() -> dict:
    return {
        "candidate": {"median_us": 10.0, "cv": 0.01},
        "baselines": {"mlx": {"median_us": 20.0}},
        "best_baseline": "mlx",
    }


def _cases(n: int, *, fail: int | None = None) -> list[dict]:
    rows = []
    for i in range(n):
        ok = i != fail
        row = {"ok": ok, "case": f"case-{i:04d}", "scaled_err": 0.1 if ok else 4.0, "allow": 1.0}
        if not ok:
            row["message"] = "too far"
        rows.append(row)
    return rows


def test_scorecard_puts_speedup_ahead_of_a_long_pass_list():
    """@brief 长的全通过正确性会把 ``format_feedback`` 前 1000 字里的加速比挤掉，成绩单仍保留它。"""
    rep = _report(
        benchmark=_bench(), speedup=2.0, speedups={"mlx": 2.0},
        correctness={"cases": _cases(40), "tolerance": {"rtol": 1e-3, "atol_rel": 1e-3}},
        compile={"ok": False, "errors": "Metal compiler errors should stay out of the scorecard"},
    )
    feedback = format_feedback(rep)
    card = format_scorecard(rep)
    assert "speedup vs best baseline" not in feedback[:1000]
    assert "speedup vs best baseline" in card
    assert card.index("STATUS:") < card.index("Performance") < card.index("Correctness")
    assert "Correctness: 40 cases PASS" in card
    assert "Metal compiler" not in card
    assert "Correctness (randomized differential tests" in feedback
    failed = _report(
        benchmark=_bench(), speedup=2.0, speedups={"mlx": 2.0},
        correctness={"cases": _cases(3, fail=1), "tolerance": {"rtol": 1e-3, "atol_rel": 1e-3}},
    )
    failed_card = format_scorecard(failed)
    assert "FAIL case-0001" in failed_card
    assert "cases PASS" not in failed_card


def test_every_v2_task_renders_three_styles_with_the_scorecard_line():
    """@brief 每道已发现的 v2 题、三种交付，成绩单那一行都在；历史和 NOTES 要么留下，要么在省略名单里。"""
    entries = discover_standard_tasks(ROOT / "tasks", code_root=ROOT)
    assert entries
    target = load_target("apple-m4-max")
    history = "HISTORY-TOKEN-9f3a " + ("round 1: raw history\n" * 80)
    notes = "NOTES-TOKEN-9f3a " + ("上一轮笔记\n" * 80)
    card = "speedup vs best baseline (mlx): 2.000x\n"
    saw_varlen = False
    rendered = 0
    for entry in entries:
        binding = bind_task(entry.op_name, entry.dtype, entry.size)
        if not binding.is_v2:
            continue
        rendered += 1
        presentation = binding.presentation()
        tolerances = binding.tolerances()
        for style in STYLES:
            dsl = style == "tilelang-pipeline"
            brief = SessionBrief(
                repo=str(ROOT), task_name=entry.op_name, dtype=entry.dtype, size=entry.size,
                kernel_name=f"{entry.op_name}_kernel", triton_source="",
                description=presentation.statement, contract_text="{}",
                abi_stub=binding.signature_stub(), rtol=tolerances["rtol"], atol_rel=tolerances["atol_rel"],
                v2=True, reference_source=presentation.reference_source,
                seed_source="kernel void k() {}\n", seedless=False,
                scorecard=card, history=history, notes_summary=notes,
                reps=15, min_improvement=1.005, noise_z=2.0,
                work_dir=str(ROOT / "agent_work" / style),
                target_id="apple-m4-max", track_id=style,
                file_ext=".taic" if dsl else target.file_ext,
                language_label="TileLang" if dsl else "Metal",
                source_fence="python" if dsl else "metal",
                reference_skill="msl_reference",
                source_name_pattern=r"\bkernel\s+void\s+[A-Za-z_][A-Za-z0-9_]*\b",
            )
            text = render_contract(brief)
            assert "speedup vs best baseline" in text
            assert "HISTORY-TOKEN-9f3a" in text or ("# 本轮省略" in text and "最近的历史" in text)
            assert "NOTES-TOKEN-9f3a" in text or ("# 本轮省略" in text and "上一轮 NOTES" in text)
            if entry.op_name == "varlen_kda" and style == "msl-single":
                saw_varlen = True
                assert len(text.encode("utf-8")) > 50000
    assert rendered
    assert saw_varlen
