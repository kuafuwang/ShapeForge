"""Real-Metal replay: gemm_silu REG candidates, fresh confirmation and the scratch probe."""

from __future__ import annotations

from pathlib import Path

import pytest

from tcenv.backends.metal.search_candidate import evaluate_candidate
from tcenv.target import load_target
from tests.backends.metal.gemm_silu_scratch_record import (
    GUIDANCE,
    locator,
    master_key,
    run_control,
)
from tests.support import gemm_silu_scratch as scratch

gpu = pytest.mark.gpu


def _evaluate(size: str):
    source = scratch.stored_source(size)
    run = evaluate_candidate(
        locator("gemm_silu", size),
        source,
        target=load_target("apple-m4-max"),
        mode="quick",
        reps=1,
    )
    report = run.result.report
    assert report["status"] == "ok", report.get("message")
    cases = (report["correctness"] or {}).get("cases") or []
    assert [item["case"] for item in cases] == list(scratch.DECLARED_CASES)
    assert all(item["ok"] is True for item in cases), cases
    assert report["safety"]["ok"] is True and report["safety"]["nondeterminism"] == []
    assert (report.get("pipeline") or {}).get("stages") == 1
    assert run.projection.candidate.candidate_sha256 == scratch.candidate_sha256(size)
    assert run.document()["fallback_credits_variant"] is False
    assert run.document()["performance_claim"] is None
    return report


@gpu
def test_small_reg_candidate_passes_all_six_cases():
    _evaluate("small")


@gpu
def test_paper_reg_candidate_passes_all_six_cases():
    _evaluate("paper")


@gpu
@pytest.mark.slow
def test_fresh_confirmation_replays_the_archived_identity_and_gate():
    from tcenv.backends.metal.search_candidate import confirm_candidate, guidance_fingerprint

    for size in scratch.SIZES:
        source = scratch.stored_source(size)
        replay = confirm_candidate(
            locator("gemm_silu", size),
            source,
            target=load_target("apple-m4-max"),
            guidance_digest=guidance_fingerprint(GUIDANCE, source),
            master_seed=master_key(),
        )
        archived = scratch.load_json(
            Path("docs/可信harness/gemm_silu-fresh-20261124-remove-v1.json")
        )["subjects"][size]
        assert scratch.stable_identity(replay) == scratch.stable_identity(archived)
        assert replay["status"] == "confirmed"
        assert (replay["identity"] or {}).get("kind") == "match"
        assert replay["confirmation_seed"] != replay["search_seed"]
        report = replay["confirmation_report"]
        assert report["status"] == "ok"
        best = min(float(value) for value in report["baseline_us"].values())
        ratio = float(report["candidate_us"]) / best
        assert ratio <= 1.10, (size, ratio, report["candidate_us"], report["baseline_us"])


@gpu
@pytest.mark.slow
def test_scratch_probe_still_measures_global_as_slower(tmp_path):
    from experiments.gemm_silu_scratch_probe import run_probe

    document = run_probe(tmp_path / "scratch-probe.json")
    scratch.assert_probe(document)
    decision = scratch.load_json(scratch.DECISION)
    for size in scratch.SIZES:
        one = document["sizes"][size]
        archived = decision["sizes"][size]
        m, n, _k = scratch.SHAPES[size]
        consumer_us = 6 * m * n / (archived["peak_gbps_measured"] * 1e9) * 1e6
        assert consumer_us > max(archived["fusion_cost_us"], 0.0), size
        assert abs(consumer_us - archived["consumer_scratch_roofline_us"]) < 1e-6
        if size == "small":
            assert one["derived"]["two_stage_floor_us"] > archived["reg_candidate_us"]
        else:
            assert one["derived"]["texture_roundtrip_over_single_us"] > archived["fusion_cost_us"]
    assert document["not_a_candidate"] is True
    assert document["publication_allowed"] is False


@gpu
@pytest.mark.slow
def test_texture_roundtrip_covers_every_scratch_element_and_detects_quarter_coverage(tmp_path):
    """@brief 回归：每个 fp32 元素都必须有自己的 R32 texel 写/读；旧的 quarter 版必须被抓到。"""
    from experiments.gemm_silu_scratch_probe import run_probe

    document = run_probe(tmp_path / "coverage-probe.json")
    scratch.assert_probe(document)
    for size in scratch.SIZES:
        one = document["sizes"][size]
        m, n, _k = scratch.SHAPES[size]
        texture = one["arms"]["two_stage_texture_roundtrip"]
        coverage = texture["coverage"]
        assert coverage["elements"] == m * n
        assert coverage["scratch_elements_read_back"] == m * n
        assert coverage["output_elements_read_back"] == m * n
        assert coverage["scratch_elements_match"] is True
        assert coverage["output_elements_match"] is True
        assert coverage["ok"] is True
        negative = one["quarter_coverage_negative"]
        assert negative["detected"] is True
        assert negative["scratch_elements_match"] is False
        assert negative["output_elements_match"] is False
        assert negative["first_scratch_mismatch"]["index"] >= 1
        # The old kernel claimed 10 bytes/element while writing only one of four
        # texels; the full read-back is what makes that claim impossible now.
        assert one["arms"]["two_stage_texture_roundtrip"]["scratch_bytes"] == 4 * m * n


@gpu
@pytest.mark.slow
def test_plain_gemm_control_replays_the_same_algorithm_identity():
    """The control is a tasks/gemm run; only its candidate median feeds the fusion delta."""
    for size in scratch.SIZES:
        document = run_control(size)
        report = document["report"]
        archived = scratch.load_json(scratch.SMALL_CONTROL if size == "small" else scratch.PAPER_CONTROL)
        assert document["publication_allowed"] is False
        assert report["status"] == "ok", report.get("message")
        cases = (report["correctness"] or {}).get("cases") or []
        assert [item["case"] for item in cases] == list(scratch.DECLARED_CASES)
        assert all(item["ok"] is True for item in cases)
        assert report["build"]["candidate_sha256"] == archived["report"]["build"]["candidate_sha256"]
        assert report["candidate_us"] > 0
