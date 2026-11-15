"""Recompute the finished P2-U pair from run files. This test starts no search and no GPU benchmark."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "runs/p2u-experience-overlay-paid-20261114"
VERIFY = ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2-U/verify.py"

CELLS = (
    "control/gemm_silu_small_rep1",
    "treatment/gemm_silu_small_rep1",
    "treatment/gemm_silu_small_rep2",
    "control/gemm_silu_small_rep2",
)


def _verify():
    from tests.support import p2u_paid_report
    return p2u_paid_report


def _result(directory: str) -> dict:
    return json.loads((EXPERIMENT / directory / "result.json").read_text(encoding="utf-8"))


def test_paired_confirmed_speedups_are_no_gain():
    speeds = {directory: _result(directory)["confirmed_speedup"] for directory in CELLS}
    first = speeds["treatment/gemm_silu_small_rep1"] - speeds["control/gemm_silu_small_rep1"]
    second = speeds["treatment/gemm_silu_small_rep2"] - speeds["control/gemm_silu_small_rep2"]
    ordered = sorted((first, second))
    median = (ordered[0] + ordered[1]) / 2
    assert speeds["control/gemm_silu_small_rep1"] == 1.8415238580482542
    assert speeds["treatment/gemm_silu_small_rep1"] == 1.7385653548302318
    assert speeds["treatment/gemm_silu_small_rep2"] == 1.564200200680351
    assert speeds["control/gemm_silu_small_rep2"] == 1.7607524937395098
    assert first == -0.10295850321802247
    assert second == -0.19655229305915878
    assert median == -0.14975539813859062
    assert first < 0 and second < 0
    report = _verify().build_report()
    assert report["paired_differences"] == [first, second]
    assert report["median_difference"] == median
    assert report["verdict"] == "no_gain"
    assert report["experiment_complete"] is True
    assert report["complete_pairs"] == 2
    assert report["excluded_pairs"] == []
    assert report["adopt_overlay"] is False
    assert report["publication_allowed"] is False


def test_secondary_lift_does_not_override_the_primary():
    report = _verify().build_report()
    control = report["secondary"]["control"]["lift_median"]
    treatment = report["secondary"]["treatment"]["lift_median"]
    assert treatment > control
    assert report["verdict"] == "no_gain"
    assert report["secondary_does_not_override_primary"] is True
    assert report["negative_result_scope"].startswith("这次 gemm_silu/fp16/small")


def test_identity_run_seed_and_search_seed():
    report = _verify().build_report()
    assert [cell["confirmation_source_sha256"][:8] for cell in report["cells"]] == [
        "e890effe",
        "b9b34b66",
        "d40fbf25",
        "ea555f09",
    ]
    for cell in report["cells"]:
        assert cell["confirmed_status"] == "ok"
        assert cell["identity"] == "match"
        assert cell["run_seed"] == 7
        assert cell["search_eval_seed"] == 7
        assert cell["confirmation_seed"] != 7
        assert cell["best_speedup"] != cell["confirmed_speedup"]
    assert report["seeds"]["confirmation_seeds_are_derived"] is True


def test_overlay_copies_come_from_session_archives():
    report = _verify().build_report()
    for cell in report["cells"]:
        copies = [row["overlay_copies"] for row in cell["session_archives"]]
        if cell["arm"] == "control":
            assert copies == [0, 0, 0, 0, 0]
            assert {row["experience_bytes"] for row in cell["session_archives"]} == {13415}
        else:
            assert copies == [1, 1, 1, 1, 1]
            assert {row["experience_bytes"] for row in cell["session_archives"]} == {25469}
            assert all(row["overlay_sidecar"] is True for row in cell["session_archives"])
    assert report["system_prompt_contains_overlay"] is False


def test_cost_excludes_interrupted_and_failed_recovery():
    report = _verify().build_report()
    cost = report["cost"]
    assert cost["stated_total_usd"] == "14.094260376"
    assert cost["cell_spent_usd_text"] == [
        "2.92720628",
        "3.665203884",
        "4.5453007240000005",
        "2.9565494880000003",
    ]
    assert cost["interrupted_commitment_usd"] == 0.0
    assert cost["recovery_failed_commitment_usd"] == 0.0
    assert cost["recovery_cost_counted_once_inside_treatment_rep1"] is True
    assert cost["recovery_cost_usd_text"] == "0.21717865199999994"
    assert [item["directory"] for item in cost["in_flight_overruns"]] == [
        "treatment/gemm_silu_small_rep1",
        "treatment/gemm_silu_small_rep2",
    ]
    assert report["recovery"]["failed_attempt_spent_usd"] == 0.0
    assert report["recovery"]["selfcheck_is_not_the_paired_metric"] is True


def test_cap_history_and_recovery_chain():
    report = _verify().build_report()
    assert report["cap"]["original_commitment_cap_usd"] == 12.0
    assert report["cap"]["raised_commitment_cap_usd"] == 15.0
    assert report["cap"]["last_cell_blocked_at_12"] is True
    assert report["cap"]["last_cell_fits_15"] is True
    assert report["cap"]["previous_cells_topped_up"] is False
    assert report["cap"]["plan_file_executed"] is False
    assert report["cap"]["execution_record"] == "four confirmed result.json files"
    assert report["cost"]["stated_commitment_before_last_usd"] == "11.137710888"
    recovery = report["recovery"]
    assert recovery["breaks_pairing"] is False
    assert recovery["candidate_sha256"].startswith("1a305622")
    assert recovery["new_model_call_for_init"] is False
    assert recovery["orphan_outer_evaluations"] == 0
    assert recovery["pre_resume_is_not_the_paired_metric"] is True
    assert recovery["pre_resume_confirmed_speedup"] != report["cells"][1]["confirmed_speedup"]
    stored = json.loads((VERIFY.parent / "evidence" / "conclusion.json").read_text(encoding="utf-8"))
    assert stored == _verify().canonical(report)
