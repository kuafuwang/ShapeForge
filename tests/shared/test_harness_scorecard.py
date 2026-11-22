"""H1 scorecard keeps four statistical units separate and missing data missing.

Synthetic runs under ``tmp_path`` cover every acceptance item in the plan:
ok/no-ok, missing submission, prior-only first-ok, unknown/incomplete cost,
too_noisy, fallback, missing receipts, dual committees, interruption,
render-only, simulated NPU, agent_work overlays, experiment cells, recovery,
agent_failure, retry dedupe, feedback/report conflicts and partial identity.

The last test reads the shipped ``runs/`` and ``targets/`` trees. It pins no run
count; it checks read-only behaviour and overlap with the existing audits.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from taic import dynamic_budget_audit, experience_scope, n1_metrics_audit, trust
from taic.harness_scorecard import SCHEMA, render_summary, scorecard
from taic.scorecard_inventory import inventory

REPO = Path(__file__).resolve().parents[2]


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _trace(*rows: dict) -> list[dict]:
    return [dict(row) for row in rows]


def _result(
    *,
    task: str = "conv2d",
    trace: list[dict] | None = None,
    target: str = "apple-m4-max",
    backend: str = "metal",
    spent: float | None = 0.25,
    **extra,
) -> dict:
    document = {
        "task": task,
        "dtype": "fp16",
        "size": "small",
        "target_id": target,
        "backend": backend,
        "trace": _trace(*(trace if trace is not None else [{"idx": 0, "label": "r1.p0", "status": "ok"}])),
        "best_speedup": 1.2,
        "confirmed_speedup": 1.1,
        "confirmed_status": "ok",
        "stop_reason": "max_rounds",
        "model_totals": [
            {"provider": "deepseek", "model": "deepseek-flash", "committee": "first", "sessions": 1, "cost_usd": 0.2},
        ],
    }
    if spent is not None:
        document["spent_usd"] = spent
    document.update(extra)
    return document


def _config(*, incomplete: bool | None = False, committees: list[dict] | None = None, **extra) -> dict:
    document = {
        "task": "conv2d",
        "dtype": "fp16",
        "size": "small",
        "target": {"id": "apple-m4-max", "backend": "metal"},
        "committees": committees if committees is not None else [
            {"name": "first", "members": [{"provider": "deepseek", "model": "deepseek-flash"}]},
            {"name": "second", "members": [{"provider": "moonshot", "model": "kimi-k3"}]},
        ],
        "cost_may_be_incomplete": incomplete,
        "system_prompt_sha1": "run-system-sha1-not-a-trust-prompt-sha256",
    }
    document.update(extra)
    return document


def _feedback(
    label: str,
    *,
    provider: str = "deepseek",
    model: str = "deepseek-flash",
    committee: str = "first",
    outcome: str | None = "evaluated",
    status: str | None = "ok",
    premises: tuple[str | None, ...] = (),
    **extra,
) -> dict:
    document = {
        "label": label,
        "t": 1.0,
        "seq": 1,
        "provider": provider,
        "model": model,
        "committee": committee,
        "outcome": outcome,
        "candidate_status": status,
        "items": [
            {"section": "观察", "text": f"item {index}", "digest": f"d{index}", "premise": premise or ""}
            for index, premise in enumerate(premises)
        ],
    }
    document.update(extra)
    return document


def _delivery(*, dynamic_omitted: int = 0, static_omitted: int = 0, blocks: list[dict] | None = None) -> dict:
    return {
        "static": {"delivered_bytes": 100, "omitted_bytes": static_omitted},
        "dynamic": {"delivered_bytes": 50, "omitted_bytes": dynamic_omitted, "budget_bytes": 200, "over_budget": bool(dynamic_omitted)},
        "situation": {
            "status": "delivered",
            "layers": [
                {"name": "overlay", "source_bytes": 10, "delivered_bytes": 10, "omitted_bytes": 0},
                {"name": "host_ledger", "source_bytes": 10, "delivered_bytes": 10, "omitted_bytes": 0},
                {"name": "peer_unverified", "source_bytes": 10, "delivered_bytes": 10, "omitted_bytes": 0},
            ],
        },
        "blocks": blocks or [],
    }


def _natural(runs: Path, name: str, result: dict, config: dict | None = None) -> Path:
    root = runs / "apple-m4-max" / name
    _write(root / "result.json", result)
    _write(root / "run_config.json", config if config is not None else _config())
    return root


def _targets(root: Path, *, trust_document: str | None = None, experience: str | None = None) -> Path:
    _write(root / "apple-m4-max" / "target.toml", {"schema_version": 1, "id": "apple-m4-max", "backend": "metal"})
    if trust_document is not None:
        (root / "apple-m4-max" / "trust.toml").write_text(trust_document, encoding="utf-8")
    if experience is not None:
        (root / "apple-m4-max" / "experience.md").write_text(experience, encoding="utf-8")
    return root


def _run_group(document: dict, **match) -> dict:
    for group in document["tables"]["run_efficiency"]["groups"]:
        if all(group["key"].get(key) == value for key, value in match.items()):
            return group
    raise AssertionError(f"no run group matching {match}")


def _session_groups(document: dict, **match) -> list[dict]:
    return [
        group for group in document["tables"]["model_session_efficiency"]["groups"]
        if all(group["key"].get(key) == value for key, value in match.items())
    ]


def test_four_tables_keep_units_apart_and_dual_committees_do_not_duplicate_runs(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010101_conv2d_fp16_deepseek-flash_aaaaaa", _result())
    _jsonl(root / "feedback.jsonl", [
        _feedback("r1.p0", premises=("verified",)),
        _feedback("r2.p0", provider="moonshot", model="kimi-k3", committee="second"),
    ])
    _write(root / "sessions" / "r1.p0" / "task_delivery.json", _delivery())
    _write(root / "sessions" / "r2.p0" / "task_delivery.json", _delivery(dynamic_omitted=10))
    _jsonl(root / "harness_proposals.jsonl", [
        {"event": "proposal", "proposal_key": "k1", "claim_digest": "d1", "evidence_level": "host_process", "provider": "deepseek", "model": "deepseek-flash", "committee": "first", "label": "r1.p0"},
    ])
    _jsonl(root / "harness_route_m.jsonl", [
        {"event": "route_m", "evidence_id": "ev1", "conclusion": "diff", "evidence_level": "host_process", "delivery": {"delivered": True, "injected_bytes": 12}, "gates": [{"name": "scope", "status": "pass"}], "shadow_written": True},
    ])
    document = scorecard(runs, targets_root=_targets(tmp_path / "targets"), prompt_sha1="p", protocol_version="3", now=10.0)
    assert document["schema"] == SCHEMA
    assert set(document["tables"]) == {"run_efficiency", "model_session_efficiency", "delivery_and_feedback", "experience_and_governance"}
    assert document["tables"]["delivery_and_feedback"]["shared_denominator"] is False
    assert set(document["tables"]["delivery_and_feedback"]["record_types"]) == {"delivery_receipt", "claim_evidence", "route_m_trace"}
    assert document["inventory"]["run_count"] == 1
    group = _run_group(document, cohort="natural", evidence_kind="unavailable")
    assert group["counts"]["runs"] == 1
    session_rows = _session_groups(document, cohort="natural")
    assert len(session_rows) == 2
    assert {(row["key"]["provider"], row["key"]["committee"]) for row in session_rows} == {("deepseek", "first"), ("moonshot", "second")}
    assert sum(row["counts"]["sessions"] for row in session_rows) == 2
    by_session = {(row["key"]["provider"], row["key"]["committee"]): row for row in session_rows}
    assert by_session[("deepseek", "first")]["counts"]["model_totals_cost_usd"] == 0.2
    assert by_session[("deepseek", "first")]["ratios"]["model_totals_cost_per_ok"] == 0.2
    assert by_session[("moonshot", "second")]["counts"]["model_totals_cost_usd"] is None
    assert by_session[("moonshot", "second")]["ratios"]["model_totals_cost_per_ok"] is None
    delivery = document["tables"]["delivery_and_feedback"]["record_types"]["delivery_receipt"]
    assert sum(group["counts"]["receipts"] for group in delivery["groups"]) == 2
    assert {group["specified_retention"]["status"] for group in delivery["groups"]} == {"not_applicable"}
    claims = document["tables"]["delivery_and_feedback"]["record_types"]["claim_evidence"]
    assert sum(group["counts"]["proposal_claims"] for group in claims["groups"]) == 1
    route = document["tables"]["delivery_and_feedback"]["record_types"]["route_m_trace"]
    assert sum(group["counts"]["traces"] for group in route["groups"]) == 1
    assert route["groups"][0]["counts"]["shadow_eligible"] == 1


def test_ok_rate_excludes_not_evaluated_and_first_ok_ignores_prior(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010102_conv2d_fp16_deepseek-flash_bbbbbb", _result(trace=_trace(
        {"idx": 0, "label": "prior", "status": "ok"},
        {"idx": 1, "label": "r1.p0", "status": "incorrect"},
        {"idx": 2, "label": "r2.p0", "status": "unchanged"},
        {"idx": 3, "label": "r3.p0", "status": "ok"},
    )))
    _jsonl(root / "feedback.jsonl", [
        _feedback("r1.p0", status="incorrect"),
        _feedback("r2.p0", outcome="unchanged", status=None),
        _feedback("r3.p0"),
    ])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _run_group(document, cohort="natural")
    assert group["counts"]["candidate_status"]["unchanged"] == 1
    # ok rows are prior + r3; incorrect is the only non-ok sent to evaluation.
    assert group["ratios"]["ok_rate"] == 2 / 3
    assert group["counts"]["first_ok_label:r3.p0"] == 1
    assert group["counts"]["runs_without_model_ok"] == 0
    assert group["counts"]["prior_only_ok_runs"] == 0


def test_prior_only_ok_is_not_a_model_first_ok(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010103_conv2d_fp16_deepseek-flash_cccccc", _result(trace=_trace(
        {"idx": 0, "label": "prior", "status": "ok"},
        {"idx": 1, "label": "r1.p0", "status": "incorrect"},
    ), spent=1.0))
    _jsonl(root / "feedback.jsonl", [_feedback("r1.p0", status="incorrect")])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _run_group(document, cohort="natural")
    assert group["counts"]["prior_only_ok_runs"] == 1
    assert group["counts"]["runs_without_model_ok"] == 1
    assert group["ratios"]["first_model_ok_rate"] == 0.0
    assert group["ratios"]["ledger_cost_per_ok"] is None
    assert group["counts"]["ledger_spent_without_ok_usd"] == 1.0
    assert group["counts"]["ok_candidates"] == 1
    assert group["counts"]["model_ok_candidates"] == 0


def test_all_incorrect_trace_reports_zero_ok_without_an_average(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(
        runs, "20261121-010104_conv2d_fp16_deepseek-flash_zzzzzz",
        _result(trace=_trace(
            {"idx": 0, "label": "r1.p0", "status": "incorrect"},
            {"idx": 1, "label": "r2.p0", "status": "incorrect"},
        ), spent=0.5),
    )
    _jsonl(root / "feedback.jsonl", [
        _feedback("r1.p0", status="incorrect"),
        _feedback("r2.p0", status="incorrect"),
    ])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _run_group(document, cohort="natural")
    assert group["counts"]["ok_candidates"] == 0
    assert group["counts"]["model_ok_candidates"] == 0
    assert group["counts"]["first_model_ok_runs"] == 0
    assert group["counts"]["runs_without_model_ok"] == 1
    assert group["ratios"]["ok_rate"] == 0.0
    assert group["ratios"]["first_model_ok_rate"] == 0.0
    assert group["ratios"]["ledger_cost_per_ok"] is None
    assert group["ratios"]["exact_cost_per_ok"] is None
    assert group["counts"]["ledger_spent_without_ok_usd"] == 0.5


def test_unknown_cost_is_null_and_incomplete_cost_suppresses_exact_per_ok(tmp_path: Path):
    runs = tmp_path / "runs"
    unknown = _natural(runs, "20261121-010104_conv2d_fp16_deepseek-flash_dddddd", _result(spent=None), _config(incomplete=None))
    _jsonl(unknown / "feedback.jsonl", [_feedback("r1.p0")])
    incomplete = _natural(
        runs, "20261121-010105_conv2d_fp16_deepseek-flash_eeeeee",
        _result(spent=1.0, task="gemm"), _config(incomplete=True),
    )
    _jsonl(incomplete / "feedback.jsonl", [_feedback("r1.p0")])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    unknown_group = _run_group(document, task="conv2d")
    assert unknown_group["counts"]["cost_known_runs"] == 0
    assert unknown_group["ratios"]["ledger_cost_per_ok"] is None
    assert unknown_group["ratios"]["exact_cost_per_ok"] is None
    assert unknown_group["counts"]["ledger_spent_without_ok_usd"] is None
    incomplete_group = _run_group(document, task="gemm")
    assert incomplete_group["ratios"]["ledger_cost_per_ok"] == 1.0
    assert incomplete_group["ratios"]["exact_cost_per_ok"] is None
    assert incomplete_group["counts"]["cost_complete"] is False
    assert incomplete_group["counts"]["cost_incomplete_runs"] == 1


def test_unknown_cost_is_not_overwritten_by_a_later_complete_run(tmp_path: Path):
    runs = tmp_path / "runs"
    unknown = _natural(runs, "20261121-010105_conv2d_fp16_deepseek-flash_uuuuuu", _result(spent=0.5), _config(incomplete=None))
    _jsonl(unknown / "feedback.jsonl", [_feedback("r1.p0")])
    complete = _natural(runs, "20261121-010106_conv2d_fp16_deepseek-flash_vvvvvv", _result(spent=0.5), _config(incomplete=False))
    _jsonl(complete / "feedback.jsonl", [_feedback("r1.p0")])
    incomplete = _natural(runs, "20261121-010107_conv2d_fp16_deepseek-flash_tttttt", _result(spent=0.5), _config(incomplete=True))
    _jsonl(incomplete / "feedback.jsonl", [_feedback("r1.p0")])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _run_group(document, cohort="natural")
    assert group["counts"]["cost_complete"] is None
    assert group["counts"]["cost_incomplete_runs"] == 1
    assert group["counts"]["cost_unknown_runs"] == 1
    assert group["ratios"]["exact_cost_per_ok"] is None
    assert group["ratios"]["ledger_cost_per_ok"] == 0.5


def test_mixed_missing_run_cost_never_uses_unknown_run_ok_denominator(tmp_path: Path):
    runs = tmp_path / "runs"
    known = _natural(
        runs, "20261121-010140_conv2d_fp16_deepseek-flash_mixk01",
        _result(spent=1.0), _config(incomplete=False),
    )
    _jsonl(known / "feedback.jsonl", [_feedback("r1.p0")])
    missing = _natural(
        runs, "20261121-010141_conv2d_fp16_deepseek-flash_mixm01",
        _result(spent=None), _config(incomplete=False),
    )
    _jsonl(missing / "feedback.jsonl", [_feedback("r1.p0")])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _run_group(document, cohort="natural")
    assert group["counts"]["runs"] == 2
    assert group["counts"]["model_ok_candidates"] == 2
    assert group["counts"]["cost_known_runs"] == 1
    assert group["counts"]["cost_known_model_ok"] == 1
    assert group["counts"]["cost_missing_runs"] == 1
    # One known dollar over one cost-known ok, not over the two raw oks.
    assert group["ratios"]["ledger_cost_per_ok"] == 1.0
    assert group["coverage"]["ledger_cost_per_ok"] == {"units": 2, "known_units": 1}
    assert group["ratios"]["exact_cost_per_ok"] is None
    assert group["counts"]["cost_complete"] is False


def test_too_noisy_pair_is_undecidable_and_gpu_seconds_are_separate(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010106_conv2d_fp16_deepseek-flash_ffffff", _result(trace=_trace(
        {"idx": 0, "label": "r1.p0", "status": "ok", "improved": True},
    )))
    _write(root / "candidates" / "001_r1.p0" / "paired_comparison.json", {
        "applied": True, "conclusion": "too_noisy", "timed_by": "gpu_timestamp",
        "median_a_s": 0.1, "median_b_s": 0.101, "interval": [0.99, 1.01],
    })
    _write(root / "sessions" / "r1.p0" / "paired_host_selection.json", {
        "session_label": "r1.p0", "ran_gpu": True, "gpu_seconds": 0.5,
        "nominations": 1, "unique_eligible": 1, "reviewed": 1, "uncovered_count": 0, "measurements": [],
    })
    _write(root / "paired_comparison_recheck.json", {"document": {"conclusion": "diff", "direction": "a_faster", "timed_by": "gpu_timestamp", "session_label": "r1.p0"}})
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _run_group(document, cohort="natural")
    assert group["counts"]["paired_applied"] == 1
    assert group["counts"]["paired_too_noisy"] == 1
    assert group["counts"]["paired_decidable"] == 0
    assert group["counts"]["paired_conclusion_known"] == 1
    assert group["counts"]["paired_conclusion_unknown"] == 0
    assert group["ratios"]["paired_decidable_rate"] == 0.0
    assert group["coverage"]["paired_decidable_rate"] == {"units": 1, "known_units": 1}
    assert group["counts"]["timed_by"] == {"gpu_timestamp": 1}
    assert group["counts"]["paired_recheck_files"] == 1
    assert group["counts"]["recheck_conclusion:diff"] == 1
    assert group["counts"]["recheck_decidable"] == 1
    assert group["counts"]["recheck_timed_by:gpu_timestamp"] == 1
    assert group["counts"]["selection_gpu_seconds"] == 0.5
    assert group["counts"]["selection_gpu_seconds_complete"] is True


def test_missing_pair_conclusion_is_unknown_not_a_known_undecidable(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010146_conv2d_fp16_deepseek-flash_noconc", _result())
    for index in range(5):
        _write(root / "candidates" / f"{index:03d}_r1.p0" / "paired_comparison.json", {
            "applied": True,
            "timed_by": "gpu_timestamp",
            "median_a_s": 0.1,
            "median_b_s": 0.11,
            "direction": "a_faster",
        })
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _run_group(document, cohort="natural")
    assert group["counts"]["paired_applied"] == 5
    assert group["counts"]["paired_decidable"] == 0
    assert group["counts"]["paired_undecidable"] == 0
    assert group["counts"]["paired_conclusion_known"] == 0
    assert group["counts"]["paired_conclusion_unknown"] == 5
    assert group["ratios"]["paired_decidable_rate"] is None
    assert group["coverage"]["paired_decidable_rate"] == {"units": 5, "known_units": 0}
    assert group["metric_status"]["paired_decidable_rate"] == "unavailable"


def test_no_diff_is_a_known_decidable_conclusion(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010149_conv2d_fp16_deepseek-flash_nodiff", _result())
    for index in range(5):
        _write(root / "candidates" / f"{index:03d}_r1.p0" / "paired_comparison.json", {
            "applied": True,
            "conclusion": "no_diff",
            "timed_by": "gpu_timestamp",
            "median_a_s": 0.1,
            "median_b_s": 0.1001,
            "interval": [0.99, 1.01],
        })
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _run_group(document, cohort="natural")
    assert group["counts"]["paired_applied"] == 5
    assert group["counts"]["paired_conclusion_known"] == 5
    assert group["counts"]["paired_conclusion_unknown"] == 0
    assert group["counts"]["paired_decidable"] == 5
    assert group["counts"]["paired_undecidable"] == 0
    assert group["ratios"]["paired_decidable_rate"] == 1.0
    assert group["metric_status"]["paired_decidable_rate"] == "measured"


def test_wall_clock_pair_never_becomes_gpu_seconds(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010106_conv2d_fp16_deepseek-flash_wwwwww", _result(trace=_trace(
        {"idx": 0, "label": "r1.p0", "status": "ok", "improved": True},
    )))
    _write(root / "candidates" / "001_r1.p0" / "paired_comparison.json", {
        "applied": True, "conclusion": "diff", "timed_by": "wall_clock",
        "median_a_s": 0.1, "median_b_s": 0.09, "elapsed_s": 0.2,
    })
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _run_group(document, cohort="natural")
    assert group["counts"]["timed_by"] == {"wall_clock": 1}
    assert group["pairs"][0]["timed_by"] == "wall_clock"
    assert group["pairs"][0]["elapsed_s"] == 0.2
    assert group["pairs"][0]["gpu_seconds"] is None
    assert group["counts"]["selection_gpu_seconds"] is None
    # No paired_host_selection row means zero known GPU seconds: not complete.
    assert group["counts"]["selection_gpu_seconds_known_sessions"] == 0
    assert group["counts"]["selection_gpu_seconds_complete"] is False


def test_control_field_absent_null_and_interval_are_distinct(tmp_path: Path):
    runs = tmp_path / "runs"
    absent = _natural(runs, "20261121-010107_conv2d_fp16_deepseek-flash_cccccc", _result())
    null = _natural(runs, "20261121-010108_conv2d_fp16_deepseek-flash_dddddd", _result())
    interval = _natural(runs, "20261121-010109_conv2d_fp16_deepseek-flash_eeeeee", _result())
    _write(absent / "candidates" / "r1" / "report.json", {"benchmark": {"protocol": "p1", "dispatch": {"grid": [1, 1, 1]}}})
    _write(null / "candidates" / "r1" / "report.json", {"benchmark": {"protocol": "p1", "dispatch": {"grid": [1, 1, 1]}, "control": None}})
    _write(interval / "candidates" / "r1" / "report.json", {
        "benchmark": {"protocol": "p1", "dispatch": {"grid": [1, 1, 1]}, "control": {"interval": [0.99, 1.01]}},
    })
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _run_group(document, cohort="natural")
    assert group["counts"]["control_field_reports"] == 2
    assert group["counts"]["control_runs"] == 1
    assert group["counts"]["control_contains_one"] == 1
    assert group["counts"]["control_excludes_one"] == 0
    assert group["ratios"]["control_contains_one_rate"] == 1.0
    # Field coverage is a report-level ratio, not "any report has the field".
    assert group["ratios"]["control_field_coverage_rate"] == 2 / 3
    assert group["coverage"]["control_field_coverage_rate"] == {"units": 3, "known_units": 2}
    assert group["metric_status"]["control_field_coverage_rate"] == "insufficient"
    assert group["counts"]["control_field_coverage"] == "insufficient"


def test_control_field_coverage_is_null_when_no_report_carries_the_field(tmp_path: Path):
    """A missing ``benchmark.control`` field is missing data, not a measured 0/3."""
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010149_conv2d_fp16_deepseek-flash_ctrl00", _result())
    for index in range(3):
        _write(root / "candidates" / f"{index:03d}_r1.p0" / "report.json", {
            "benchmark": {"protocol": "p1", "dispatch": {"grid": [1, 1, 1]}},
        })
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _run_group(document, cohort="natural")
    assert group["counts"]["reports_total"] == 3
    assert group["counts"]["control_field_reports"] == 0
    assert group["ratios"]["control_field_coverage_rate"] is None
    assert group["coverage"]["control_field_coverage_rate"] == {"units": 3, "known_units": 0}
    assert group["metric_status"]["control_field_coverage_rate"] == "unavailable"
    assert group["counts"]["control_field_coverage"] == "unavailable"
    # The actual control-run ratio is a separate measurement and stays unknown too.
    assert group["counts"]["control_runs"] == 0
    assert group["ratios"]["control_contains_one_rate"] is None


def test_control_field_coverage_uses_reports_and_h0_threshold(tmp_path: Path):
    runs = tmp_path / "runs"
    sparse = _natural(runs, "20261121-010147_conv2d_fp16_deepseek-flash_ctrl01", _result())
    for index in range(5):
        benchmark = {"protocol": "p1", "dispatch": {"grid": [1, 1, 1]}}
        if index == 0:
            benchmark["control"] = {"interval": [0.99, 1.01]}
        _write(sparse / "candidates" / f"{index:03d}_r1.p0" / "report.json", {"benchmark": benchmark})
    full = _natural(
        runs, "20261121-010148_gemm_fp16_deepseek-flash_ctrl02",
        _result(task="gemm"), _config(task="gemm"),
    )
    for index in range(5):
        _write(full / "candidates" / f"{index:03d}_r1.p0" / "report.json", {
            "benchmark": {
                "protocol": "p1",
                "dispatch": {"grid": [1, 1, 1]},
                "control": {"interval": [0.99, 1.01]},
            },
        })
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    sparse_group = _run_group(document, cohort="natural", task="conv2d")
    assert sparse_group["counts"]["reports_total"] == 5
    assert sparse_group["counts"]["control_field_reports"] == 1
    assert sparse_group["counts"]["control_runs"] == 1
    assert sparse_group["ratios"]["control_field_coverage_rate"] == 0.2
    assert sparse_group["metric_status"]["control_field_coverage_rate"] == "insufficient"
    assert sparse_group["counts"]["control_field_coverage"] == "insufficient"
    full_group = _run_group(document, cohort="natural", task="gemm")
    assert full_group["counts"]["reports_total"] == 5
    assert full_group["counts"]["control_field_reports"] == 5
    assert full_group["ratios"]["control_field_coverage_rate"] == 1.0
    assert full_group["metric_status"]["control_field_coverage_rate"] == "measured"
    assert full_group["ratios"]["control_contains_one_rate"] == 1.0
    assert full_group["metric_status"]["control_contains_one_rate"] == "measured"


def test_missing_delivery_receipt_is_unavailable_not_a_zero_group(tmp_path: Path):
    runs = tmp_path / "runs"
    _natural(runs, "20261121-010107_conv2d_fp16_deepseek-flash_gggggg", _result())
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    delivery = document["tables"]["delivery_and_feedback"]["record_types"]["delivery_receipt"]
    assert delivery["coverage"] == "unavailable"
    assert delivery["groups"] == []
    assert delivery["render_only_receipts"] == 0


def test_missing_receipt_bytes_stay_null_not_zero(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010108_conv2d_fp16_deepseek-flash_missing", _result())
    _jsonl(root / "feedback.jsonl", [_feedback("r1.p0")])
    receipt = _delivery()
    receipt["dynamic"].pop("delivered_bytes")
    _write(root / "sessions" / "r1.p0" / "task_delivery.json", receipt)
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    delivery = document["tables"]["delivery_and_feedback"]["record_types"]["delivery_receipt"]
    group = [row for row in delivery["groups"] if row["key"]["cohort"] == "natural"][0]
    assert group["counts"]["dynamic_bytes_complete"] is False
    assert group["counts"]["receipts_with_missing_dynamic_bytes"] == 1
    assert group["counts"]["dynamic_delivered_bytes"] is None
    assert group["counts"]["dynamic_omitted_bytes"] is None
    assert group["counts"]["dynamic_expected_bytes"] is None
    assert group["ratios"]["dynamic_omission_rate"] is None
    assert group["coverage"]["dynamic_omission_rate"]["known_units"] == 0
    assert group["counts"]["static_bytes_complete"] is True
    assert group["counts"]["static_omitted_bytes"] == 0


def test_feedback_conflict_outer_status_wins_and_retry_is_not_double_counted(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010108_conv2d_fp16_deepseek-flash_hhhhhh", _result(trace=_trace(
        {"idx": 0, "label": "r1.p0", "status": "incorrect"},
    )))
    _jsonl(root / "feedback.jsonl", [
        _feedback("r1.p0.retry1", status="ok"),
        _feedback("r1.p0.retry2", status="ok"),
    ])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _run_group(document, cohort="natural")
    assert group["counts"]["sessions"] == 1
    session_group = _session_groups(document, cohort="natural")[0]
    assert session_group["counts"]["sessions"] == 1
    assert session_group["counts"]["conflicts"] == 1
    assert session_group["counts"]["retry_rows_deduplicated"] == 1
    row = session_group["sessions"][0]
    assert row["resolved_status"] == "incorrect" and row["status_source"] == "outer_report"
    assert row["session_label"] == "r1.p0"
    assert row["session_label_raw"] == "r1.p0.retry2"


def test_candidate_report_wins_over_trace_and_feedback(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010130_conv2d_fp16_deepseek-flash_rep001", _result(trace=_trace(
        {"idx": 0, "label": "r1.p0", "status": "ok"},
    )))
    _jsonl(root / "feedback.jsonl", [_feedback("r1.p0", status="ok")])
    _write(root / "candidates" / "001_r1.p0" / "report.json", {"status": "incorrect"})
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    session_group = _session_groups(document, cohort="natural")[0]
    assert session_group["counts"]["conflicts"] == 1
    assert session_group["counts"]["completed_ok_sessions"] == 0
    assert session_group["counts"]["candidate:non_ok"] == 1
    row = session_group["sessions"][0]
    assert row["resolved_status"] == "incorrect"
    assert row["status_source"] == "outer_report"
    assert row["outer_status"] == "incorrect"
    assert row["outer_source"] == "candidate_report"


def test_agent_failure_is_not_completed_ok_and_no_submission_is_explicit(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010109_conv2d_fp16_deepseek-flash_iiiiii", _result(
        trace=[], stop_reason="agent_failure", confirmed_status=None,
    ))
    _jsonl(root / "feedback.jsonl", [
        _feedback("r1.p0", outcome="agent_failed", status=None),
        _feedback("r2.p0", outcome="agent_no_submission", status=None),
    ])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _run_group(document, cohort="natural")
    assert group["counts"]["runs"] == 1
    assert group["counts"]["lifecycle:finalized"] == 1
    assert group["counts"]["stop_reason"] == {"agent_failure": 1}
    assert group["counts"]["session_outcome:agent_failed"] == 1
    assert group["counts"]["session_outcome:agent_no_submission"] == 1
    assert group["ratios"]["no_submission_rate"] == 0.5
    assert group["counts"]["confirmed_status"] == {}
    assert "completed_ok" not in group["counts"]
    assert group["counts"]["ok_candidates"] is None


def test_failed_outcome_with_ok_status_is_not_a_completed_ok(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010110_conv2d_fp16_deepseek-flash_oooooo", _result(
        trace=[], stop_reason="agent_failure", confirmed_status=None,
    ))
    _jsonl(root / "feedback.jsonl", [_feedback("r1.p0", outcome="agent_failed", status="ok")])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    session_group = _session_groups(document, cohort="natural")[0]
    assert session_group["counts"]["sessions"] == 1
    assert session_group["counts"]["candidate:ok"] == 1
    assert session_group["counts"]["completed_ok_sessions"] == 0
    assert session_group["counts"]["failed_outcome_with_ok_status"] == 1
    assert session_group["ratios"]["session_ok_rate"] == 0.0


def test_agent_work_overlay_and_render_only_keep_dry_run_distinct(tmp_path: Path):
    runs = tmp_path / "runs"
    real = _natural(runs, "20261121-010110_conv2d_fp16_deepseek-flash_jjjjjj", _result())
    _write(real / "agent_work" / "r1.p0" / "experience_overlay.json", {"overlay": True})
    dry = runs / "apple-m4-max" / "20261121-010111_conv2d_fp16_deepseek-flash_kkkkkk"
    _write(dry / "run_config.json", _config())
    _write(dry / "experience_overlay.json", {"overlay": True})
    _write(dry / "render-check" / "task_delivery.json", _delivery())
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    assert document["inventory"]["run_count"] == 2
    assert document["inventory"]["counts"]["lifecycle"]["dry_run"] == 1
    dry_group = _run_group(document, cohort="natural", lifecycle="dry_run")
    assert dry_group["counts"]["runs"] == 1
    delivery = document["tables"]["delivery_and_feedback"]["record_types"]["delivery_receipt"]
    assert delivery["render_only_receipts"] == 1
    assert sum(group["counts"]["receipts"] for group in delivery["groups"]) == 0


def test_parent_container_cells_and_recovery_cost_are_not_double_counted(tmp_path: Path):
    runs = tmp_path / "runs"
    experiment = runs / "experiment"
    _write(experiment / "plan.json", {"schema_id": "experiments.plan.1"})
    _write(experiment / "approval.json", {"approved_by": "alice"})
    _write(experiment / "summary.json", {"spent_usd": 99.0})
    control = experiment / "control" / "cell-1"
    _write(control / "result.json", _result(spent=1.0))
    _write(control / "run_config.json", _config())
    treatment = experiment / "treatment" / "cell-1"
    _write(treatment / "result.json", _result(spent=0.25))
    _write(treatment / "run_config.json", _config())
    recovery = experiment / "recovery" / "cell-1"
    _write(recovery / "result.json", _result(spent=0.5, recovery_of="control/cell-1"))
    _write(recovery / "run_config.json", _config())
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    assert document["inventory"]["run_count"] == 3
    assert document["inventory"]["experiment_containers"] == [{"path": "experiment", "child_runs": 3, "counted_as_run": False}]
    groups = document["tables"]["run_efficiency"]["groups"]
    assert sum(group["counts"]["cost_known_runs"] for group in groups) == 3
    attributed = sum(
        (group["ratios"]["ledger_cost_per_ok"] or 0.0) * (group["counts"]["cost_known_model_ok"] or 0)
        for group in groups
    )
    assert round(attributed, 12) == 1.75
    controlled = [group for group in groups if group["key"]["cohort"] == "controlled"]
    assert sum(group["counts"]["runs"] for group in controlled) == 2
    assert controlled[0]["key"]["experiment_id"] is None
    assert str(controlled[0]["key"]["experiment_group"]).startswith("plan-sha256:")
    recovery_group = [group for group in groups if group["key"]["cohort"] == "recovery"][0]
    assert recovery_group["counts"]["recovery_runs"] == 1
    assert recovery_group["counts"].get("recovery_target_unresolved", 0) == 0
    assert recovery_group["speedups"][0]["recovery_of"] == "control/cell-1"
    # A resolved recovery's cost belongs to the target cell: the controlled group
    # receives 1.0 + 0.25 own cost plus 0.5 incoming, while the recovery group
    # keeps only the raw description and no second ledger entry.
    controlled_group = controlled[0]
    assert round(controlled_group["counts"]["ledger_spent_usd"], 12) == 1.75
    assert controlled_group["counts"]["attributed_recovery_cost_usd"] == 0.5
    assert controlled_group["counts"]["attributed_recovery_runs"] == 1
    assert controlled_group["counts"]["cost_known_runs"] == 3
    assert controlled_group["counts"]["cost_known_model_ok"] == 3
    assert controlled_group["ratios"]["ledger_cost_per_ok"] == 1.75 / 3
    assert round(sum(group["counts"]["ledger_spent_usd"] or 0.0 for group in groups), 12) == 1.75
    assert recovery_group["counts"]["ledger_spent_usd"] is None
    assert recovery_group["counts"]["cost_known_runs"] == 0
    assert recovery_group["counts"]["recovery_cost_usd"] == 0.5
    assert recovery_group["counts"]["recovery_cost_attributed_elsewhere_usd"] == 0.5
    assert recovery_group["counts"]["attributed_recovery_cost_usd"] is None
    assert recovery_group["ratios"]["ledger_cost_per_ok"] is None
    assert recovery_group["ratios"]["exact_cost_per_ok"] is None
    assert recovery_group["counts"]["cost_complete"] is False


def test_missing_recovery_target_is_flagged_not_guessed(tmp_path: Path):
    runs = tmp_path / "runs"
    experiment = runs / "experiment"
    _write(experiment / "plan.json", {"schema_id": "experiments.plan.1"})
    _write(experiment / "recovery" / "cell-1" / "result.json", _result(spent=0.5, recovery_of="control/missing"))
    _write(experiment / "recovery" / "cell-1" / "run_config.json", _config())
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = [item for item in document["tables"]["run_efficiency"]["groups"] if item["key"]["cohort"] == "recovery"][0]
    assert group["counts"]["recovery_target_unresolved"] == 1
    assert group["counts"]["unattributed_recovery_cost_usd"] == 0.5
    # The unresolved cost is not attributed: it must not enter the group ledger,
    # the per-ok ratios, or the no-ok ledger. It stays in the raw row and the
    # unattributed field above.
    assert group["counts"]["ledger_spent_usd"] is None
    assert group["counts"]["cost_known_runs"] == 0
    assert group["ratios"]["ledger_cost_per_ok"] is None
    assert group["ratios"]["exact_cost_per_ok"] is None
    assert group["counts"]["ledger_spent_without_ok_usd"] is None
    assert group["counts"]["cost_complete"] is None


def test_resolved_and_unresolved_recovery_costs_are_attributed_once(tmp_path: Path):
    runs = tmp_path / "runs"
    experiment = runs / "experiment"
    _write(experiment / "plan.json", {"schema_id": "experiments.plan.1"})
    _write(experiment / "approval.json", {"approved_by": "alice"})
    control = experiment / "control" / "cell-1"
    _write(control / "result.json", _result(spent=1.0))
    _write(control / "run_config.json", _config())
    resolved = experiment / "recovery" / "rep-1"
    _write(resolved / "result.json", _result(spent=0.5, recovery_of="control/cell-1"))
    _write(resolved / "run_config.json", _config())
    unresolved = experiment / "recovery" / "rep-2"
    _write(unresolved / "result.json", _result(spent=0.3, recovery_of="control/missing"))
    _write(unresolved / "run_config.json", _config())
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    groups = document["tables"]["run_efficiency"]["groups"]
    controlled = [group for group in groups if group["key"]["cohort"] == "controlled"][0]
    recovery = [group for group in groups if group["key"]["cohort"] == "recovery"][0]
    # The target group books the control plus the resolved recovery exactly once.
    assert controlled["counts"]["ledger_spent_usd"] == 1.5
    assert controlled["counts"]["attributed_recovery_cost_usd"] == 0.5
    assert controlled["counts"]["cost_known_runs"] == 2
    assert controlled["counts"]["model_ok_candidates"] == 1
    assert controlled["counts"]["cost_known_model_ok"] == 2
    assert controlled["counts"]["cost_units_model_ok"] == 2
    assert controlled["coverage"]["ledger_cost_per_ok"] == {"units": 2, "known_units": 2}
    assert controlled["coverage"]["exact_cost_per_ok"] == {"units": 2, "known_units": 2}
    assert controlled["ratios"]["ledger_cost_per_ok"] == 0.75
    # The recovery group keeps the raw description but no second ledger entry.
    assert recovery["counts"]["ledger_spent_usd"] is None
    assert recovery["counts"]["cost_known_runs"] == 0
    assert recovery["counts"]["cost_known_model_ok"] == 0
    assert recovery["counts"]["cost_units_model_ok"] == 0
    assert recovery["coverage"]["ledger_cost_per_ok"] == {"units": 0, "known_units": 0}
    assert recovery["coverage"]["exact_cost_per_ok"] == {"units": 0, "known_units": 0}
    assert recovery["counts"]["model_ok_candidates"] == 2
    assert recovery["counts"]["recovery_cost_usd"] == 0.8
    assert recovery["counts"]["recovery_cost_attributed_elsewhere_usd"] == 0.5
    assert recovery["counts"]["unattributed_recovery_cost_usd"] == 0.3
    assert recovery["counts"]["recovery_target_unresolved"] == 1
    assert recovery["ratios"]["ledger_cost_per_ok"] is None
    assert recovery["ratios"]["exact_cost_per_ok"] is None
    # The unresolved recovery's ok never enters a known-cost denominator.
    assert sum(group["counts"]["cost_known_model_ok"] for group in groups) == 2
    assert sum(group["counts"]["model_ok_candidates"] for group in groups) == 3


def test_interrupted_run_is_counted_but_candidates_are_unavailable(tmp_path: Path):
    runs = tmp_path / "runs"
    root = runs / "apple-m4-max" / "20261121-010123_conv2d_fp16_deepseek-flash_wwwwww"
    _write(root / "state.json", {"task": "conv2d", "target_id": "apple-m4-max", "backend": "metal", "spent_usd": 0.4})
    _write(root / "run_config.json", _config(incomplete=True))
    _jsonl(root / "transcript.jsonl", [{"role": "user", "text": "start"}])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    assert document["inventory"]["counts"]["lifecycle"]["incomplete_or_aborted"] == 1
    group = _run_group(document, cohort="natural", lifecycle="incomplete_or_aborted")
    assert group["counts"]["runs"] == 1
    assert group["counts"]["ok_candidates"] is None
    assert group["counts"]["candidates_complete"] is False
    assert group["counts"]["cost_known_runs"] == 1
    assert group["ratios"]["exact_cost_per_ok"] is None


def test_unknown_cohort_is_not_merged_into_natural(tmp_path: Path):
    runs = tmp_path / "runs"
    _natural(runs, "20261121-010124_conv2d_fp16_deepseek-flash_xxxxxx", _result())
    custom = runs / "apple-m4-max" / "custom-layout"
    _write(custom / "result.json", _result(spent=0.4))
    _write(custom / "run_config.json", _config())
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    assert document["inventory"]["counts"]["cohort"]["natural"] == 1
    assert document["inventory"]["counts"]["cohort"]["unknown"] == 1
    assert _run_group(document, cohort="natural")["counts"]["runs"] == 1
    assert _run_group(document, cohort="unknown")["counts"]["runs"] == 1


def test_static_contract_omission_is_a_violation_not_a_ratio(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010125_conv2d_fp16_deepseek-flash_yyyyyy", _result())
    _jsonl(root / "feedback.jsonl", [_feedback("r1.p0")])
    _write(root / "sessions" / "r1.p0" / "task_delivery.json", _delivery(static_omitted=5, blocks=[
        {"name": "上一轮 NOTES", "expected_bytes": 10, "delivered_bytes": 4, "omitted_bytes": 6, "omission_reason": "budget"},
    ]))
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    delivery = document["tables"]["delivery_and_feedback"]["record_types"]["delivery_receipt"]
    group = delivery["groups"][0]
    assert group["counts"]["static_omitted_bytes"] == 5
    assert group["counts"]["static_contract_ok"] is False
    assert "static_omission_rate" not in group["ratios"]
    assert group["counts"]["other_blocks"]["omitted_bytes"] == 6
    assert group["counts"]["other_blocks_by_name"]["上一轮 NOTES"]["omitted_bytes"] == 6
    assert set(group["counts"]["situation_layers"]) == {"overlay", "host_ledger", "peer_unverified"}


def test_controlled_retention_list_is_restated_not_invented(tmp_path: Path):
    runs = tmp_path / "runs"
    experiment = runs / "experiment"
    _write(experiment / "plan.json", {"schema_id": "experiments.plan.1", "retention": ["上一轮 NOTES"]})
    _write(experiment / "approval.json", {"approved_by": "alice"})
    cell = experiment / "control" / "cell-1"
    _write(cell / "result.json", _result())
    _write(cell / "run_config.json", _config())
    _write(cell / "sessions" / "r1.p0" / "task_delivery.json", _delivery())
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    delivery = document["tables"]["delivery_and_feedback"]["record_types"]["delivery_receipt"]
    retention = delivery["groups"][0]["specified_retention"]
    assert retention["status"] == "archived"
    assert retention["value"] == ["上一轮 NOTES"]
    assert retention["source"] == "experiment/plan.json"


def test_simulated_npu_declaration_never_fills_device_metrics(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(
        runs, "20261121-010112_relu_fp16_deepseek-flash_llllll",
        _result(task="relu", target="qingwei-npu", backend="npu", simulated=True, measurement_kind="simulated"),
    )
    _jsonl(root / "feedback.jsonl", [_feedback("r1.p0", simulated=True)])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    assert document["inventory"]["counts"]["evidence_kind"]["simulated"] == 1
    assert document["backend_boundary"]["simulated_backends"] == ["npu"]
    assert document["backend_boundary"]["npu"]["device_metrics"] == "unavailable"
    assert document["backend_boundary"]["npu"]["natural_runs"] == 0
    assert document["backend_boundary"]["npu"]["publication_allowed"] is False
    group = _run_group(document, evidence_kind="simulated")
    assert group["counts"]["runs"] == 1
    # evidence kind and metric status are independent dimensions.
    assert group["key"]["evidence_kind"] == "simulated"
    assert group["metric_status"]["ok_rate"] in {"measured", "insufficient", "unavailable"}


def test_trust_is_per_grant_and_never_compares_system_prompt_sha1(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010113_conv2d_fp16_deepseek-flash_mmmmmm", _result())
    _jsonl(root / "feedback.jsonl", [_feedback("r1.p0")])
    trust_document = """
version = 1

[[grants]]
id = "g-active"
target_id = "apple-m4-max"
task = "conv2d"
provider = "deepseek"
model = "deepseek-flash"
effort = "high"
prompt_sha1 = "current-prompt"
protocol_version = "3"
status = "active"
granted_by = "alice"
expires_at = 0.0
scopes = ["overlay:premise"]

[[grants]]
id = "g-stale"
target_id = "apple-m4-max"
task = "conv2d"
provider = "moonshot"
model = "kimi-k3"
effort = "high"
prompt_sha1 = "run-system-sha1-not-a-trust-prompt-sha256"
protocol_version = "3"
status = "active"
granted_by = "alice"
expires_at = 0.0
scopes = ["overlay:tip"]

[[grants]]
id = "g-revoked"
target_id = "apple-m4-max"
task = "gemm"
provider = "deepseek"
model = "deepseek-flash"
effort = "high"
prompt_sha1 = "current-prompt"
protocol_version = "3"
status = "revoked"
granted_by = "bob"
expires_at = 0.0
scopes = ["overlay:premise"]
"""
    targets = _targets(tmp_path / "targets", trust_document=trust_document)
    document = scorecard(runs, targets_root=targets, prompt_sha1="current-prompt", protocol_version="3", now=10.0)
    governance = document["tables"]["experience_and_governance"]["governance"]
    assert governance["states"]["active"] == 1
    assert governance["states"]["stale_prompt"] == 1
    assert governance["states"]["revoked"] == 1
    assert {(grant["model"], grant["state"]) for grant in governance["grants"]} == {
        ("deepseek-flash", "active"), ("kimi-k3", "stale_prompt"), ("deepseek-flash", "revoked"),
    }
    assert {(grant["task"], tuple(grant["scopes"])) for grant in governance["grants"]} == {
        ("conv2d", ("overlay:premise",)), ("conv2d", ("overlay:tip",)), ("gemm", ("overlay:premise",)),
    }
    assert all(grant["granted_by"] in {"alice", "bob"} for grant in governance["grants"])
    assert all("authorized_target" not in row for row in governance["rollup"])
    # The active grant's prompt matches the override, not run_config.system_prompt_sha1.
    run_group = _run_group(document, cohort="natural")
    assert run_group["counts"]["runs"] == 1
    assert "system_prompt_sha1" not in governance["grants"][0]


def test_experience_pointers_group_by_target_and_task_without_promotion(tmp_path: Path):
    runs = tmp_path / "runs"
    _natural(runs, "20261121-010114_conv2d_fp16_deepseek-flash_nnnnnn", _result())
    experience = """# 经验

## 有题条目

Confidence: verified
题目: conv2d
芯片: apple-m4-max
落点: targets/apple-m4-max/experience.md
观察: 一条本机实测。
来源: `runs/apple-m4-max/20261121-010114_conv2d_fp16_deepseek-flash_nnnnnn/result.json`
本机实测: 1.0 µs
边界: conv2d fp16 small
反例: 换题后不成立。
宿主重跑: runs/apple-m4-max/20261121-010114_conv2d_fp16_deepseek-flash_nnnnnn/result.json

## 无题条目

Confidence: reported
观察: 没有题目。
本机实测: 2.0 µs
边界: 未声明
反例: 未知。
"""
    targets = _targets(tmp_path / "targets", experience=experience)
    document = scorecard(runs, targets_root=targets, prompt_sha1="p", protocol_version="3", now=10.0)
    table = document["tables"]["experience_and_governance"]["experience"]
    assert table["status"] == "measured"
    tasks = {group["key"]["task"] for group in table["groups"]}
    assert "conv2d" in tasks and "unknown" in tasks
    assert table["summary"]["pointer:valid"] == 1
    assert table["summary"]["pointer:existing"] >= 1
    assert table["summary"]["sections"] == 2


def test_claim_repetition_uses_exact_keys_and_delivery_metrics_stay_separate(tmp_path: Path):
    runs = tmp_path / "runs"
    first = _natural(runs, "20261121-010115_conv2d_fp16_deepseek-flash_oooooo", _result())
    second = _natural(runs, "20261121-010116_conv2d_fp16_deepseek-flash_pppppp", _result())
    for root, level in ((first, "model_claim"), (second, "host_process")):
        _jsonl(root / "harness_proposals.jsonl", [
            {"event": "proposal", "proposal_key": "same-key", "claim_digest": "same-digest", "evidence_level": level, "provider": "deepseek", "model": "deepseek-flash", "committee": "first", "label": "r1.p0"},
        ])
        _jsonl(root / "feedback.jsonl", [_feedback("r1.p0", premises=("verified", "unverified", None))])
        _write(root / "sessions" / "r1.p0" / "task_delivery.json", _delivery())
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    claims = document["tables"]["delivery_and_feedback"]["record_types"]["claim_evidence"]
    repetition = claims["repetition"]["by_cohort_and_experiment"]
    assert len(repetition) == 1
    assert repetition[0]["cohort"] == "natural"
    assert repetition[0]["structured_claims"] == 2
    assert repetition[0]["distinct_structured_keys"] == 1
    assert repetition[0]["repeated_keys"] == 1
    assert repetition[0]["repeated_occurrences"] == 2
    assert repetition[0]["status"] == "measured"
    counts = claims["groups"][0]["counts"]
    assert counts["proposal_claims"] == 2
    assert counts["explicit_evidence_levels"] == 2
    assert counts["premise:verified"] == 2
    assert counts["premise:unverified"] == 2
    assert counts["premise_blank"] == 2
    assert claims["groups"][0]["ratios"]["premise_verified_rate"] == 0.5
    delivery = document["tables"]["delivery_and_feedback"]["record_types"]["delivery_receipt"]
    delivery_counts = delivery["groups"][0]["counts"]
    assert "proposal_claims" not in delivery_counts
    assert "premise:verified" not in delivery_counts


def test_claims_without_structural_key_make_repetition_unavailable(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010117_conv2d_fp16_deepseek-flash_qqqqqq", _result())
    _jsonl(root / "harness_proposals.jsonl", [
        {"event": "proposal", "evidence_level": "model_claim", "proposal": {"action": "free text"}},
    ])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    claims = document["tables"]["delivery_and_feedback"]["record_types"]["claim_evidence"]
    repetition = claims["repetition"]["by_cohort_and_experiment"]
    assert repetition[0]["claims_without_structural_key"] == 1
    assert repetition[0]["status"] == "unavailable"
    assert claims["repetition"]["claims_without_structural_key"] == 1


def test_claim_repetition_does_not_cross_cohorts(tmp_path: Path):
    runs = tmp_path / "runs"
    first = _natural(runs, "20261121-010131_conv2d_fp16_deepseek-flash_aaaaa1", _result())
    second = _natural(runs, "20261121-010132_conv2d_fp16_deepseek-flash_aaaaa2", _result())
    experiment = runs / "experiment"
    _write(experiment / "plan.json", {"schema_id": "experiments.plan.1"})
    _write(experiment / "approval.json", {"approved_by": "alice"})
    cell = experiment / "cell-1"
    _write(cell / "result.json", _result())
    _write(cell / "run_config.json", _config())
    for root in (first, second, cell):
        _jsonl(root / "harness_proposals.jsonl", [
            {"event": "proposal", "proposal_key": "shared", "claim_digest": "shared", "evidence_level": "model_claim"},
        ])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    repetition = document["tables"]["delivery_and_feedback"]["record_types"]["claim_evidence"]["repetition"]["by_cohort_and_experiment"]
    by_cohort = {row["cohort"]: row for row in repetition}
    assert by_cohort["natural"]["structured_claims"] == 2
    assert by_cohort["natural"]["distinct_structured_keys"] == 1
    assert by_cohort["natural"]["repeated_keys"] == 1
    assert by_cohort["natural"]["repeated_occurrences"] == 2
    # The same key appears once in the controlled cell; it must not be counted
    # as a repetition against the natural group.
    assert by_cohort["controlled"]["structured_claims"] == 1
    assert by_cohort["controlled"]["repeated_keys"] == 0
    assert by_cohort["controlled"]["repeated_occurrences"] == 0
    assert str(by_cohort["controlled"]["experiment_group"]).startswith("plan-sha256:")


def test_partial_identity_is_not_a_full_match(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010118_conv2d_fp16_deepseek-flash_rrrrrr", _result())
    common = {
        "candidate_sha256": "c",
        "geometry_digest": "g",
        "bundle_digest": "b",
        "artifacts": {"artifact_set_digest": "a"},
        "native_digest": "n",
    }
    _write(root / "confirmations" / "full.json", {
        "provider": "candidate",
        "identity": {"kind": "match"},
        "expected": {"build": common, "evaluation": {"evaluation_fingerprint": "e"}},
        "confirmation_report": {"build": common, "evaluation_fingerprint": "e"},
    })
    _write(root / "confirmations" / "partial.json", {
        "provider": "candidate",
        "identity": {"kind": "match"},
        "expected": {"build": {"candidate_sha256": "c"}},
        "confirmation_report": {"build": {"candidate_sha256": "c"}},
    })
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _run_group(document, cohort="natural")
    assert group["counts"]["identity"]["full_match"] == 1
    assert group["counts"]["identity"]["partial"] == 1
    assert group["counts"]["identity"]["legacy_kind_match_without_full_scope"] == 1
    assert group["ratios"]["identity_full_match_rate"] == 0.5
    assert group["metric_status"]["identity_full_match_rate"] == "insufficient"


def test_fallback_marks_the_speedup_row_and_never_averages(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010119_conv2d_fp16_deepseek-flash_ssssss", _result())
    _write(root / "confirmations" / "bench.json", {
        "label": "r1.p0", "status": "ok", "fallback_id": "general", "provider": "fallback",
    })
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _run_group(document, cohort="natural")
    assert "fallback" in group["speedups"][0]["flags"]
    assert group["speedups"][0]["included_in_average"] is False
    assert "speedup" not in group["ratios"]
    assert all("average" not in name for name in group["ratios"])


def test_unknown_feedback_outcome_is_null_not_zero(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010120_conv2d_fp16_deepseek-flash_tttttt", _result())
    _jsonl(root / "feedback.jsonl", [{"label": "r1.p0", "provider": "deepseek", "model": "deepseek-flash", "committee": "first", "items": []}])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    session_group = _session_groups(document, cohort="natural")[0]
    assert session_group["counts"]["sessions"] == 1
    assert session_group["ratios"]["evaluated_rate"] is None
    assert session_group["ratios"]["no_submission_rate"] is None
    assert session_group["metric_status"]["evaluated_rate"] == "unavailable"


def test_model_totals_cost_is_not_added_to_run_ledger(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010121_conv2d_fp16_deepseek-flash_uuuuuu", _result(spent=1.0, model_totals=[
        {"provider": "deepseek", "model": "deepseek-flash", "committee": "first", "sessions": 1, "cost_usd": 2.0},
    ]))
    _jsonl(root / "feedback.jsonl", [_feedback("r1.p0")])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _run_group(document, cohort="natural")
    assert group["ratios"]["ledger_cost_per_ok"] == 1.0
    session_group = _session_groups(document, cohort="natural")[0]
    assert session_group["counts"]["model_totals_cost_usd"] == 2.0
    assert group["ratios"]["ledger_cost_per_ok"] + session_group["counts"]["model_totals_cost_usd"] == 3.0


def test_model_totals_partial_cost_uses_only_known_run_completed_ok(tmp_path: Path):
    runs = tmp_path / "runs"
    known = _natural(
        runs, "20261121-010142_gemm_fp16_deepseek-flash_totk01",
        _result(task="gemm", spent=1.0), _config(task="gemm", incomplete=False),
    )
    _jsonl(known / "feedback.jsonl", [_feedback("r1.p0")])
    missing = _natural(
        runs, "20261121-010143_gemm_fp16_deepseek-flash_totm01",
        _result(task="gemm", spent=1.0, model_totals=[]), _config(task="gemm", incomplete=False),
    )
    _jsonl(missing / "feedback.jsonl", [_feedback("r1.p0")])
    complete = _natural(
        runs, "20261121-010144_softmax_fp16_deepseek-flash_totc01",
        _result(task="softmax", spent=1.0), _config(task="softmax", incomplete=False),
    )
    _jsonl(complete / "feedback.jsonl", [_feedback("r1.p0")])
    complete2 = _natural(
        runs, "20261121-010145_softmax_fp16_deepseek-flash_totc02",
        _result(task="softmax", spent=1.0), _config(task="softmax", incomplete=False),
    )
    _jsonl(complete2 / "feedback.jsonl", [_feedback("r1.p0")])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    partial = _session_groups(document, task="gemm")[0]
    assert partial["counts"]["sessions"] == 2
    assert partial["counts"]["completed_ok_sessions"] == 2
    assert partial["counts"]["completed_ok_with_known_cost"] == 1
    assert partial["counts"]["completed_ok_without_cost"] == 1
    assert partial["counts"]["model_totals_cost_usd"] == 0.2
    assert partial["counts"]["runs_without_model_totals"] == 1
    assert partial["counts"]["cost_complete"] is False
    # Not 0.2 / 2: the unknown-cost session is outside the exact denominator.
    assert partial["ratios"]["model_totals_cost_per_ok"] is None
    assert partial["coverage"]["model_totals_cost_per_ok"] == {"units": 2, "known_units": 0}
    complete_group = _session_groups(document, task="softmax")[0]
    assert complete_group["counts"]["model_totals_cost_usd"] == 0.4
    assert complete_group["counts"]["completed_ok_with_known_cost"] == 2
    assert complete_group["counts"]["runs_without_model_totals"] == 0
    assert complete_group["counts"]["cost_complete"] is True
    assert complete_group["ratios"]["model_totals_cost_per_ok"] == 0.2


def test_model_total_entry_without_cost_on_a_failed_run_blocks_exact_cost(tmp_path: Path):
    runs = tmp_path / "runs"
    known = _natural(
        runs, "20261121-010150_gemm_fp16_deepseek-flash_costok",
        _result(task="gemm", spent=1.0), _config(task="gemm", incomplete=False),
    )
    _jsonl(known / "feedback.jsonl", [_feedback("r1.p0")])
    no_cost = _natural(
        runs, "20261121-010151_gemm_fp16_deepseek-flash_nocost",
        _result(
            task="gemm", spent=1.0,
            trace=_trace({"idx": 0, "label": "r1.p0", "status": "incorrect"}),
            model_totals=[
                {"provider": "deepseek", "model": "deepseek-flash", "committee": "first", "sessions": 1},
            ],
        ),
        _config(task="gemm", incomplete=False),
    )
    _jsonl(no_cost / "feedback.jsonl", [_feedback("r1.p0", status="incorrect")])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _session_groups(document, task="gemm")[0]
    assert group["counts"]["sessions"] == 2
    assert group["counts"]["completed_ok_sessions"] == 1
    assert group["counts"]["model_totals_runs"] == 2
    assert group["counts"]["runs_with_model_total_cost"] == 1
    assert group["counts"]["runs_without_model_totals"] == 1
    assert group["counts"]["runs_with_model_total_without_cost"] == 1
    assert group["counts"]["completed_ok_with_known_cost"] == 1
    # The failed session has no ok, but its run still has a cost gap.
    assert group["counts"]["completed_ok_without_cost"] == 0
    assert group["counts"]["model_totals_cost_usd"] == 0.2
    assert group["counts"]["cost_complete"] is False
    assert group["ratios"]["model_totals_cost_per_ok"] is None


def test_model_totals_zero_cost_without_a_billed_sample_is_not_a_known_cost(tmp_path: Path):
    """``cost_usd=0.0`` with ``sessions_with_cost=0`` is a missing cost, not a free run."""
    runs = tmp_path / "runs"
    free = _natural(
        runs, "20261121-010152_gemm_fp16_deepseek-flash_zero01",
        _result(task="gemm", spent=1.0, model_totals=[{
            "provider": "deepseek", "model": "deepseek-flash", "committee": "first",
            "sessions": 1, "sessions_with_cost": 0, "cost_usd": 0.0, "cost_per_session_usd": None,
        }]),
        _config(task="gemm", incomplete=False),
    )
    _jsonl(free / "feedback.jsonl", [_feedback("r1.p0")])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _session_groups(document, task="gemm")[0]
    assert group["counts"]["sessions"] == 1
    assert group["counts"]["model_totals_runs"] == 1
    # The aggregate row exists but carries no billed sample.
    assert group["counts"]["model_totals_cost_usd"] is None
    assert group["counts"]["runs_with_model_total_cost"] == 0
    assert group["counts"]["runs_with_model_total_without_cost"] == 1
    assert group["counts"]["completed_ok_with_known_cost"] == 0
    assert group["counts"]["completed_ok_without_cost"] == 1
    assert group["counts"]["cost_complete"] is False
    assert group["ratios"]["model_totals_cost_per_ok"] is None


def test_model_totals_mixed_known_and_zero_cost_keeps_exact_null(tmp_path: Path):
    runs = tmp_path / "runs"
    known = _natural(
        runs, "20261121-010153_gemm_fp16_deepseek-flash_mixk01",
        _result(task="gemm", spent=1.0), _config(task="gemm", incomplete=False),
    )
    _jsonl(known / "feedback.jsonl", [_feedback("r1.p0")])
    zero = _natural(
        runs, "20261121-010154_gemm_fp16_deepseek-flash_mixz01",
        _result(task="gemm", spent=1.0, model_totals=[{
            "provider": "deepseek", "model": "deepseek-flash", "committee": "first",
            "sessions": 1, "sessions_with_cost": 0, "cost_usd": 0.0, "cost_per_session_usd": None,
        }]),
        _config(task="gemm", incomplete=False),
    )
    _jsonl(zero / "feedback.jsonl", [_feedback("r1.p0")])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _session_groups(document, task="gemm")[0]
    assert group["counts"]["sessions"] == 2
    # Only the run that actually has a billed sample enters the known ledger and
    # the completed-ok denominator; both sides stay on the same known-cost set.
    assert group["counts"]["model_totals_cost_usd"] == 0.2
    assert group["counts"]["completed_ok_with_known_cost"] == 1
    assert group["counts"]["completed_ok_without_cost"] == 1
    assert group["counts"]["runs_without_model_totals"] == 1
    assert group["counts"]["cost_complete"] is False
    assert group["ratios"]["model_totals_cost_per_ok"] is None


def test_model_totals_partial_billing_coverage_is_not_complete(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(
        runs, "20261121-010155_gemm_fp16_deepseek-flash_part01",
        _result(task="gemm", spent=1.0, model_totals=[{
            "provider": "deepseek", "model": "deepseek-flash", "committee": "first",
            "sessions": 4, "sessions_with_cost": 3, "cost_usd": 0.6,
            "cost_per_session_usd": 0.2,
        }]),
        _config(task="gemm", incomplete=False),
    )
    _jsonl(root / "feedback.jsonl", [_feedback(f"r{index}.p0") for index in range(1, 5)])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _session_groups(document, task="gemm")[0]
    assert group["counts"]["sessions"] == 4
    assert group["counts"]["completed_ok_sessions"] == 4
    assert group["counts"]["model_totals_cost_usd"] == 0.6
    # The exact numerator may not use this amount: the run's sessions are not all in
    # the cost-known set, so the fully-covered numerator stays null.
    assert group["counts"]["model_totals_cost_usd_fully_covered"] is None
    assert group["counts"]["runs_with_partial_model_cost"] == 1
    # One of four sessions has no billed sample, so the run is not a complete cost.
    assert group["counts"]["runs_with_model_total_cost"] == 0
    assert group["counts"]["completed_ok_with_known_cost"] == 0
    assert group["counts"]["completed_ok_without_cost"] == 4
    assert group["counts"]["cost_complete"] is False
    assert group["ratios"]["model_totals_cost_per_ok"] is None
    assert any("sessions_with_cost" in note for note in group["notes"])


def test_model_totals_exact_numerator_only_uses_fully_covered_runs(tmp_path: Path):
    """The per-ok numerator and denominator come from the same cost-known set."""
    runs = tmp_path / "runs"
    full = _natural(
        runs, "20261121-010161_gemm_fp16_deepseek-flash_full01",
        _result(task="gemm", spent=1.0, model_totals=[{
            "provider": "deepseek", "model": "deepseek-flash", "committee": "first",
            "sessions": 2, "sessions_with_cost": 2, "cost_usd": 0.4,
            "cost_per_session_usd": 0.2,
        }]),
        _config(task="gemm", incomplete=False),
    )
    _jsonl(full / "feedback.jsonl", [_feedback("r1.p0"), _feedback("r2.p0")])
    partial = _natural(
        runs, "20261121-010162_gemm_fp16_deepseek-flash_part02",
        _result(task="gemm", spent=1.0, model_totals=[{
            "provider": "deepseek", "model": "deepseek-flash", "committee": "first",
            "sessions": 4, "sessions_with_cost": 3, "cost_usd": 0.6,
            "cost_per_session_usd": 0.2,
        }]),
        _config(task="gemm", incomplete=False),
    )
    _jsonl(partial / "feedback.jsonl", [_feedback(f"r{index}.p0") for index in range(1, 5)])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = _session_groups(document, task="gemm")[0]
    assert group["counts"]["sessions"] == 6
    # The known amount is a fact and keeps both runs; the exact numerator keeps only
    # the fully covered run, matching its completed-ok denominator (2 sessions).
    assert group["counts"]["model_totals_cost_usd"] == 1.0
    assert group["counts"]["model_totals_cost_usd_fully_covered"] == 0.4
    assert group["counts"]["completed_ok_sessions"] == 6
    assert group["counts"]["completed_ok_with_known_cost"] == 2
    assert group["counts"]["completed_ok_without_cost"] == 4
    assert group["counts"]["runs_with_partial_model_cost"] == 1
    assert group["counts"]["cost_complete"] is False
    assert group["ratios"]["model_totals_cost_per_ok"] is None


def test_unknown_session_outcome_stays_in_the_evaluated_rate_units(tmp_path: Path):
    """Unrecognised outcomes lower coverage; they do not leave the denominator."""
    runs = tmp_path / "runs"
    covered = _natural(runs, "20261121-010156_conv2d_fp16_deepseek-flash_unkn01", _result())
    _jsonl(covered / "feedback.jsonl", [
        *[_feedback(f"r{index}.p0") for index in range(1, 6)],
        _feedback("r6.p0", outcome="unknown", status=None),
    ])
    sparse = _natural(
        runs, "20261121-010157_softmax_fp16_deepseek-flash_unkn02",
        _result(task="softmax"), _config(task="softmax"),
    )
    _jsonl(sparse / "feedback.jsonl", [
        *[_feedback(f"r{index}.p0") for index in range(1, 6)],
        _feedback("r6.p0", outcome="unknown", status=None),
        _feedback("r7.p0", outcome="unknown", status=None),
    ])
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    measured = _session_groups(document, task="conv2d")[0]
    assert measured["counts"]["sessions"] == 6
    assert measured["coverage"]["evaluated_rate"] == {"units": 6, "known_units": 5}
    assert measured["ratios"]["evaluated_rate"] == 1.0
    assert measured["metric_status"]["evaluated_rate"] == "measured"
    assert measured["coverage"]["no_submission_rate"] == {"units": 6, "known_units": 5}
    insufficient = _session_groups(document, task="softmax")[0]
    assert insufficient["counts"]["sessions"] == 7
    assert insufficient["counts"]["outcomes"]["unknown"] == 2
    assert insufficient["coverage"]["evaluated_rate"] == {"units": 7, "known_units": 5}
    # Five recognised outcomes are no longer enough when two sessions are unknown.
    assert insufficient["metric_status"]["evaluated_rate"] == "insufficient"
    assert insufficient["metric_status"]["no_submission_rate"] == "insufficient"
    # Unknown is not a failure and not a completed-ok session.
    assert insufficient["counts"]["completed_ok_sessions"] == 5
    assert insufficient["counts"]["failed_outcome_with_ok_status"] == 0
    assert any("未知" in note for note in insufficient["notes"])
    run_group = _run_group(document, cohort="natural", task="softmax")
    assert run_group["coverage"]["no_submission_rate"] == {"units": 7, "known_units": 5}
    assert run_group["metric_status"]["no_submission_rate"] == "insufficient"


def test_experience_rates_need_h0_minimum_units(tmp_path: Path):
    runs = tmp_path / "runs"
    _natural(runs, "20261121-010158_conv2d_fp16_deepseek-flash_expu01", _result())
    experience = """# 经验

## 单条验证

Confidence: verified
题目: conv2d
芯片: apple-m4-max
落点: targets/apple-m4-max/experience.md
观察: 一条本机实测。
来源: `runs/apple-m4-max/20261121-010158_conv2d_fp16_deepseek-flash_expu01/result.json`
本机实测: 1.0 µs
边界: conv2d fp16 small
反例: 换题后不成立。

## 无指针条目

Confidence: reported
观察: 没有落点。
本机实测: 2.0 µs
边界: 未声明。
反例: 未知。
"""
    targets = _targets(tmp_path / "targets", experience=experience)
    document = scorecard(runs, targets_root=targets, prompt_sha1="p", protocol_version="3", now=10.0)
    table = document["tables"]["experience_and_governance"]["experience"]
    by_task = {group["key"]["task"]: group for group in table["groups"]}
    one_pointer = by_task["conv2d"]
    assert one_pointer["coverage"]["pointer_valid_rate"] == {"units": 1, "known_units": 1}
    # One pointer is not the H0 claim-level minimum of five.
    assert one_pointer["metric_status"]["pointer_valid_rate"] == "insufficient"
    assert one_pointer["metric_status"]["verified_rate"] == "insufficient"
    no_pointer = by_task["unknown"]
    assert no_pointer["coverage"]["pointer_valid_rate"] == {"units": 0, "known_units": 0}
    assert no_pointer["metric_status"]["pointer_valid_rate"] == "unavailable"
    assert no_pointer["metric_status"]["verified_rate"] == "insufficient"


def test_text_summary_prints_null_for_missing_omitted_bytes(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010159_conv2d_fp16_deepseek-flash_null01", _result())
    _jsonl(root / "feedback.jsonl", [_feedback("r1.p0")])
    receipt = _delivery()
    receipt["static"].pop("delivered_bytes")
    receipt["dynamic"].pop("delivered_bytes")
    _write(root / "sessions" / "r1.p0" / "task_delivery.json", receipt)
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    group = [
        row for row in document["tables"]["delivery_and_feedback"]["record_types"]["delivery_receipt"]["groups"]
        if row["key"]["cohort"] == "natural"
    ][0]
    # The underlying table keeps the missing bytes as null.
    assert group["counts"]["static_omitted_bytes"] is None
    assert group["counts"]["dynamic_omitted_bytes"] is None
    text = render_summary(document)
    assert "static_omitted=null" in text
    assert "dynamic_omitted=null" in text
    assert "static_omitted=0" not in text
    assert "dynamic_omitted=0" not in text


def test_text_summary_keeps_a_true_zero_omission_at_zero(tmp_path: Path):
    runs = tmp_path / "runs"
    root = _natural(runs, "20261121-010160_conv2d_fp16_deepseek-flash_zero02", _result())
    _jsonl(root / "feedback.jsonl", [_feedback("r1.p0")])
    _write(root / "sessions" / "r1.p0" / "task_delivery.json", _delivery())
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    text = render_summary(document)
    assert "static_omitted=0" in text
    assert "dynamic_omitted=0" in text
    assert "static_omitted=null" not in text
    assert "dynamic_omitted=null" not in text


def test_text_summary_reports_no_receipts_as_unavailable_not_zero(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    delivery = document["tables"]["delivery_and_feedback"]["record_types"]["delivery_receipt"]
    assert delivery["groups"] == []
    assert delivery["coverage"] == "unavailable"
    text = render_summary(document)
    assert "receipts=null" in text
    assert "static_omitted=null" in text
    assert "dynamic_omitted=null" in text
    assert "receipts=0" not in text


def test_scorecard_renders_all_four_tables_and_explicit_unknowns(tmp_path: Path):
    runs = tmp_path / "runs"
    _natural(runs, "20261121-010122_conv2d_fp16_deepseek-flash_vvvvvv", _result())
    document = scorecard(runs, targets_root=tmp_path / "targets", prompt_sha1="p", protocol_version="3", now=10.0)
    text = render_summary(document)
    assert "run_efficiency" in text
    assert "model_session_efficiency" in text
    assert "delivery_and_feedback" in text
    assert "experience_and_governance" in text
    assert "NPU device metrics: unavailable" in text
    assert document["publication_allowed"] is False
    # The frozen snapshot's h3_started=false is kept, and the live document says
    # explicitly that it only describes this read-only audit.
    assert document["h3_started"] is False
    assert document["h3_started_by_this_audit"] is False
    assert any("本只读审计不启动 H3" in note for note in document["notes"])
    assert "this audit starts H3=False" in text


def test_cli_audit_scorecard_prints_schema_json(tmp_path: Path, capsys):
    from taic.cli import main

    runs = tmp_path / "runs"
    _natural(runs, "20261121-010126_conv2d_fp16_deepseek-flash_zzzzzz", _result())
    targets = _targets(tmp_path / "targets")
    assert main(["harness", "audit-scorecard", "--runs", str(runs), "--targets", str(targets), "--json"]) == 0
    captured = json.loads(capsys.readouterr().out)
    assert captured["schema"] == SCHEMA
    assert captured["read_only"] is True
    assert captured["model_calls"] == 0 and captured["gpu_calls"] == 0


def test_real_scorecard_is_read_only_and_overlaps_existing_audits():
    runs = REPO / "runs"
    targets = REPO / "targets"
    if not runs.is_dir() or not targets.is_dir():
        return
    before = {path: path.stat().st_mtime_ns for path in runs.rglob("*.json")}
    document = scorecard(runs, targets_root=targets)
    after = {path: path.stat().st_mtime_ns for path in runs.rglob("*.json")}
    assert after == before
    reference = inventory(runs)
    assert document["inventory"]["run_count"] == reference["run_count"]
    assert document["inventory"]["counts"]["lifecycle"] == reference["counts"]["lifecycle"]
    assert document["inventory"]["counts"]["cohort"] == reference["counts"]["cohort"]
    # Overlap with audit-n1: natural paired files/applied and recheck files.
    n1 = n1_metrics_audit.audit(runs)
    cross = document["cross_audit"]["n1"]
    assert cross["natural_paired_files"] == n1["natural_paired_comparisons"]["counts"].get("files", 0)
    assert cross["natural_paired_applied"] == n1["natural_paired_comparisons"]["counts"].get("applied", 0)
    assert cross["recheck_files"] == n1["rechecks"]["files"]
    # Overlap with audit-dynamic-budget: session receipts plus the separate render-only receipt
    # and any receipt that lives outside both places. The latter is still part of the audit's
    # total, so the test accounts for it instead of assuming the whole tree is static.
    dynamic = dynamic_budget_audit.audit(runs)
    delivery = document["tables"]["delivery_and_feedback"]["record_types"]["delivery_receipt"]
    run_roots = {
        runs / root
        for group in document["tables"]["run_efficiency"]["groups"]
        for root in group["source_run_roots"]
    }
    all_files = sorted(runs.rglob("task_delivery.json"))
    session_files = [
        path for path in all_files
        if path.parent.parent.name == "sessions" and path.parent.parent.parent in run_roots
    ]
    render_files = [
        path for path in all_files
        if path.parent.name == "render-check" and path.parent.parent in run_roots
    ]
    extra_files = [path for path in all_files if path not in set(session_files) | set(render_files)]

    def _omitted(path: Path, section: str) -> int:
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get(section) or {}
        except (OSError, ValueError):
            return 0
        return int(value.get("omitted_bytes") or 0)

    sessions = sum(group["counts"]["receipts"] for group in delivery["groups"])
    assert sessions == len(session_files)
    assert delivery["render_only_receipts"] == len(render_files)
    assert sessions + delivery["render_only_receipts"] + len(extra_files) == dynamic["receipts"]
    dynamic_bytes = sum(
        group["counts"]["dynamic_omitted_bytes"] for group in delivery["groups"]
    )
    assert dynamic_bytes + sum(_omitted(path, "dynamic") for path in extra_files) == dynamic["counts"].get("dynamic_receipt_omitted", 0)
    static_bytes = sum(
        group["counts"]["static_omitted_bytes"] for group in delivery["groups"]
    )
    assert static_bytes + sum(_omitted(path, "static") for path in extra_files) == dynamic["counts"].get("static_omitted", 0)
    # Overlap with audit-experience: pointers are the same unit, no promotion.
    experience = experience_scope.audit_repo(REPO)
    table_summary = document["tables"]["experience_and_governance"]["experience"]["summary"]
    assert table_summary.get("pointer:total", 0) == experience["summary"]["pointers"]["total"]
    assert table_summary.get("pointer:valid", 0) == experience["summary"]["pointers"]["valid"]
    # Trust is per grant, never a target boolean.
    grant_count = 0
    for path in sorted(targets.glob("*/trust.toml")):
        if path.parent.name.startswith("_"):
            continue
        grant_count += len(trust.load_grants(path.parent.name, root=targets))
    assert len(document["tables"]["experience_and_governance"]["governance"]["grants"]) == grant_count
    assert all("authorized_target" not in row for row in document["tables"]["experience_and_governance"]["governance"]["rollup"])
    # No speedup or latency averaging and no simulated NPU rows.
    allowed_evidence = {"real", "simulated", "unavailable"}
    allowed_status = {"measured", "unavailable", "undecidable", "not_applicable", "insufficient"}
    for table_name in ("run_efficiency", "model_session_efficiency"):
        for group in document["tables"][table_name]["groups"]:
            assert group["key"]["evidence_kind"] in allowed_evidence
            assert all(value in allowed_status for value in group["metric_status"].values())
    for record_type in document["tables"]["delivery_and_feedback"]["record_types"].values():
        for group in record_type["groups"]:
            assert group["key"]["evidence_kind"] in allowed_evidence
            assert all(value in allowed_status for value in group["metric_status"].values())
    for group in document["tables"]["run_efficiency"]["groups"]:
        assert not any("speedup" in name or "latency" in name for name in group["ratios"])
        assert all("average" not in name for name in group["ratios"])
    assert document["backend_boundary"]["npu"]["device_metrics"] == "unavailable"
    assert document["publication_allowed"] is False
    assert document["h3_started"] is False
    source = (REPO / "taic" / "harness_scorecard.py").read_text(encoding="utf-8")
    assert "observe.index" not in source
    assert "health.collect" not in source
    assert "overlay_activity" not in source
    assert "borrow(" not in source
    for group in document["tables"]["run_efficiency"]["groups"]:
        if group["key"]["cohort"] == "natural":
            assert re.match(r"^[0-9]{8}-[0-9]{6}_", group["source_run_roots"][0].rsplit("/", 1)[-1])
