"""P2-P real-Metal replay: REG candidates, fresh confirmation and the unpack probe."""

from __future__ import annotations

import pytest

from tcenv.backends.metal.search_candidate import evaluate_candidate, confirm_candidate, guidance_fingerprint
from tcenv.target import load_target
from tests.backends.metal.bitdelta_unpack_record import GUIDANCE, locator, master_key
from tests.support import bitdelta_unpack as unpack

gpu = pytest.mark.gpu


def _evaluate(size: str):
    source = unpack.stored_source(size)
    run = evaluate_candidate(
        locator(size),
        source,
        target=load_target("apple-m4-max"),
        mode="quick",
        reps=1,
    )
    report = run.result.report
    assert report["status"] == "ok", report.get("message")
    cases = (report["correctness"] or {}).get("cases") or []
    assert [item["case"] for item in cases] == list(unpack.DECLARED_CASES)
    assert all(item["ok"] is True for item in cases), cases
    assert all(item["n_bad"] == 0 and item["unwritten"] == 0 for item in cases)
    assert report["safety"]["ok"] is True and report["safety"]["nondeterminism"] == []
    assert (report.get("pipeline") or {}).get("stages") == 1
    assert run.projection.candidate.candidate_sha256 == unpack.candidate_sha256(size)
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
    for size in unpack.SIZES:
        source = unpack.stored_source(size)
        replay = confirm_candidate(
            locator(size),
            source,
            target=load_target("apple-m4-max"),
            guidance_digest=guidance_fingerprint(GUIDANCE, source),
            master_seed=master_key(),
        )
        archived = unpack.load_json(
            unpack.SMALL_CONFIRMATION if size == "small" else unpack.PAPER_CONFIRMATION
        )
        assert unpack.stable_identity(replay) == unpack.stable_identity(archived)
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
def test_probe_reproduces_the_paired_ordering_and_the_decode_coverage(tmp_path):
    """@brief 重跑 probe：每个合法 SHARED 臂仍慢于 REG，覆盖与 staging 审计不变。"""
    from experiments.bitdelta_unpack_probe import run_probe

    document = run_probe(tmp_path / "unpack-probe.json")
    unpack.assert_probe(document)
    decision = unpack.load_json(unpack.DECISION)
    for size in unpack.SIZES:
        one = document["sizes"][size]
        archived = decision["sizes"][size]
        ratios = one["interleaved"]["paired_ratio_over_reg"]
        for name, ratio in ratios.items():
            assert ratio["median"] > 1.0, (size, name, ratio["median"])
            assert ratio["rounds_over_one"] >= 10, (size, name, ratio["rounds_over_one"])
        assert one["no_fill"]["shared_fp16"]["wall_us_median"] < one["arms"]["shared_fp16"]["wall_us_median"]
        # The conclusions must reproduce, not the exact microseconds: the same regime (the cheapest
        # legal form only slightly slower at paper, an order of magnitude slower at small) and,
        # where it is stable, the same ordering. The 1.06 bound is a magnitude check, not the gate.
        # A future machine that flipped a stable comparison would have to fail here and force a new
        # decision instead of silently keeping the old one.
        if size == "paper":
            # At paper the ordering is stable in every recorded run (1.02 < 1.08 < 1.47).
            assert min(ratios, key=lambda name: ratios[name]["median"]) == "shared_packed16"
            assert 1.0 < ratios["shared_packed16"]["median"] < 1.06
            assert ratios["shared_packed16"]["median"] < ratios["shared_fp32"]["median"] < ratios["shared_fp16"]["median"]
        else:
            # At small the two cheapest legal forms sit within a few percent of each other and have
            # swapped places between runs (current 7.9445/8.4109, replay 8.2818/8.8025, and an
            # earlier review run recorded 8.5589/8.5482).
            # Only the order-of-magnitude gap is asserted; a strict ranking here was flaky.
            assert ratios["shared_packed16"]["median"] > 4.0
            assert ratios["shared_fp32"]["median"] > 4.0
            cheapest = min(ratios["shared_packed16"]["median"], ratios["shared_fp32"]["median"])
            dearest = max(ratios["shared_packed16"]["median"], ratios["shared_fp32"]["median"])
            assert dearest / cheapest < 1.25
        assert abs(one["arms"]["packed"]["wall_us_median"] / archived["reg_candidate_us"] - 1.0) < 0.25
        assert one["arms"]["packed"]["wall_us_median"] > archived["roofline_lower_bound_us"]
    assert document["not_a_candidate"] is True
    assert document["publication_allowed"] is False


@gpu
@pytest.mark.slow
def test_legal_shared_producer_carries_every_half_bit_through_the_texture(tmp_path):
    """@brief 回归：packed-fp32 poison producer 必须把每个 half 位型逐位写进 R32 纹理。"""
    from experiments.bitdelta_unpack_probe import run_probe

    document = run_probe(tmp_path / "coverage-probe.json")
    unpack.assert_probe(document)
    for size in unpack.SIZES:
        one = document["sizes"][size]
        m, n, k = unpack.SHAPES[size]
        coverage = one["arms"]["shared_packed16"]["coverage"]
        assert coverage["expected_elements"] == k * n
        assert coverage["read_back_elements"] == k * n
        assert coverage["mismatched_elements"] == 0
        assert coverage["ok"] is True
        assert one["output_agreement"]["packed_vs_shared_packed16_equal"] is True
        # Two decoded halves per 4-byte texel keeps the fp16 footprint, not a quarter of it.
        assert one["arms"]["shared_packed16"]["scratch_bytes"] == 2 * k * n
        assert one["traffic"]["shared_fp16"]["consumer_b_read_bytes"] == (m // 64) * 2 * k * n
