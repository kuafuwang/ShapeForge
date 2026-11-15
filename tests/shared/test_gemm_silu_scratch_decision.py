"""CPU replay: gemm_silu REG sources, the scratch probe and the negative decision."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tests.support import gemm_silu_scratch as scratch

P2 = scratch.ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收"
HISTORY = (("P2-I", 71), ("P2-J", 22), ("P2-K", 13), ("P2-L", 12), ("P2-N", 8))
N3_CHANGES = {
    ("P2-I", "docs/架构/shape-portfolio实施计划.md"): (
        "6d0e43bea94efe9dfe7aa333c5be81dabcb6ed5df0d9662c282ec869013e7438",
        "97d498d8a986ff0afb36ccf1c2b44ffcd9b2627564bb7ce3f403cf432936c630",
    ),
    ("P2-K", "tests/backends/metal/conv2d_portfolio_fresh_record.py"): (
        "a551a288406585d1d878d9f8fa75ac4524df74e169e6c265cbfd6dd48e5c6410",
        "ba55cf8ab493dae0a9c90caed35d93aa408f6614b55f62f9c4932a160923612b",
    ),
    ("P2-L", "experiments/gemm_splitk_probe.py"): (
        "e38e21100cca82313dc23f899c1438f8e0c25defbf09b8d37c332d4dd324aee1",
        "bbde260eb15d07c704bd9b3a69ec8abbc06e1fd15aba768c6a9b7cb464044c99",
    ),
    ("P2-L", "tests/support/gemm_splitk.py"): (
        "5b1a12459a0e3fca5bbb6ec9790c2e62f6aadcce738d9da14eed8607d8c8abd6",
        "7ff5f23f41ccd0297e8652a8b0099f11f3370f8b1d9cc27d0b375b0b3da4e7b9",
    ),
}


def _subjects() -> dict[str, dict]:
    return {
        "small": scratch.load_json(scratch.SMALL_CONFIRMATION),
        "paper": scratch.load_json(scratch.PAPER_CONFIRMATION),
    }


def _controls() -> dict[str, dict]:
    return {
        "small": scratch.load_json(scratch.SMALL_CONTROL),
        "paper": scratch.load_json(scratch.PAPER_CONTROL),
    }


def test_reg_candidates_are_one_stage_and_declare_no_scratch():
    """@brief 两个 REG 候选都是 fixture + 片段上的 SiLU，单 stage、零 scratch。"""
    scratch.assert_source_provenance()
    scratch.assert_no_global_candidate()
    for size in scratch.SIZES:
        source = scratch.stored_source(size)
        launch = source.split("/* taic-launch", 1)[1].split("*/", 1)[0]
        assert launch.count("launch ") == 1
        assert "scratch" not in launch
        assert "math fast" in launch
        assert "atomic" not in source
        assert "threadgroup_barrier" in source  # the GEMM K-loop barrier, not a cross-TG spin
        assert scratch.candidate_sha256(size) == (
            scratch.SMALL_SHA256 if size == "small" else scratch.PAPER_SHA256
        )


def test_scratch_probe_records_the_full_global_byte_flow():
    """@brief probe 不是 candidate，且 fill/写/读/输出字节数与形状代数一致。"""
    document = scratch.load_json(scratch.PROBE)
    scratch.assert_probe(document)
    assert document["not_a_candidate"] is True
    assert document["publication_allowed"] is False
    for size, one in document["sizes"].items():
        m, n, _k = scratch.SHAPES[size]
        assert one["scratch_fp32_bytes"] == 4 * m * n
        assert one["fp32_fill_bytes"] == 4 * m * n
        assert one["fp32_write_bytes"] == 4 * m * n
        assert one["fp32_read_bytes"] == 4 * m * n
        assert one["fp16_output_bytes"] == 2 * m * n
        assert one["scratch_roundtrip_bytes"] == 8 * m * n
        arms = one["arms"]
        assert arms["two_stage_texture_roundtrip"]["fill_bytes_per_iteration"] == 0
        assert arms["two_stage_buffer_roundtrip"]["fill_bytes_per_iteration"] == 4 * m * n
        assert arms["two_stage_floor"]["stages"] == 2
        assert arms["two_stage_texture_roundtrip"]["stages"] == 2
        assert one["derived"]["texture_roundtrip_over_single_us"] > 0
        for name in ("two_stage_texture_roundtrip", "two_stage_buffer_roundtrip"):
            coverage = arms[name]["coverage"]
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


def test_decision_keeps_reg_because_global_is_measured_slower():
    """@brief 决策由证据重算；两档的保守下界和 probe 外推都慢于 REG。"""
    subjects, controls, probe = _subjects(), _controls(), scratch.load_json(scratch.PROBE)
    document = scratch.load_json(scratch.DECISION)
    scratch.assert_decision(document, subjects, controls, probe)
    assert document["adopted_route"] == "REG" and document["decision"] == "reg_only"
    assert document["implementation"]["reg_candidate_added"] is True
    assert document["implementation"]["global_candidate_source_added"] is False
    assert document["implementation"]["same_algorithm_ab_run"] is False
    assert document["publication_allowed"] is False
    for size, one in document["sizes"].items():
        assert one["global_would_win"] is False
        assert one["hard_lower_bound_us"] >= one["reg_candidate_us"]
        assert one["best_probe_global_estimate_us"] >= one["reg_candidate_us"]
        assert one["hard_lower_bound_over_reg"] > 1.0
        assert one["probe_estimates_are_not_candidates"] is True
        assert one["serialized_scratch_roofline_us"] > one["consumer_scratch_roofline_us"]
        assert "global_saves_no_flops" in one["stop_reasons"]
        assert "hard_lower_bound_exceeds_reg" in one["stop_reasons"]
        assert one["fusion_cost_fraction_of_reg"] < 0.01
    small, paper = document["sizes"]["small"], document["sizes"]["paper"]
    assert small["two_stage_floor_us"] > small["reg_candidate_us"]
    assert "two_stage_floor_exceeds_reg" in small["stop_reasons"]
    assert paper["consumer_scratch_roofline_us"] > paper["fusion_cost_us"]
    assert "consumer_scratch_roofline_exceeds_fusion_saving" in paper["stop_reasons"]
    assert paper["hard_lower_bound_us"] == paper["reg_candidate_us"] + paper["net_consumer_roofline_us"]
    assert paper["serialized_global_estimate_us"] > paper["hard_lower_bound_us"]
    assert paper["two_stage_buffer_roundtrip_us"] > paper["two_stage_texture_roundtrip_us"]


def test_fresh_subjects_are_six_case_confirmed_and_not_published():
    """@brief 两档 fresh subject 都是六 case、identity match、非 7 的 fresh seed。"""
    for size, subject in _subjects().items():
        scratch.assert_subject(subject, size)
        assert subject["status"] == "confirmed"
        assert subject["confirmation_seed"] != subject["search_seed"]
        assert subject["confirmation_attempts"] == 1
        assert (subject["identity"] or {}).get("kind") == "match"
        assert "publication_allowed" not in subject  # a subject is not a publication


def test_historical_acceptance_manifests_are_byte_identical():
    """@brief 闭合 manifest 逐行核对；仅 N3 已命名的活跃 record 使用固定 after。"""
    for phase, count in HISTORY:
        path = P2 / phase / "manifest.sha256"
        entries = [line.split("  ", 1) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(entries) == count, phase
        for digest, name in entries:
            actual = hashlib.sha256((scratch.ROOT / name).read_bytes()).hexdigest()
            change = N3_CHANGES.get((phase, name))
            if change is None:
                assert actual == digest, name
            else:
                before, after = change
                assert digest == before, name
                assert actual == after, name


def test_prompt_pack_and_trust_boundaries_are_unchanged():
    """@brief 提示 SHA、79 文件闭包和 trust.toml 都停在 P2-M 的状态。"""
    from taic.trust import current_prompt_sha1
    from tcenv.taskcontract.pack import code_import_closure

    from tests.support.live_anchors import load_live_anchors
    anchors = load_live_anchors(scratch.ROOT)
    assert current_prompt_sha1() == anchors["prompt_sha1"]
    files = code_import_closure(scratch.ROOT)
    assert list(files) == anchors["pack_closure"]["files"]
    assert len(files) == 79
    digest = hashlib.sha256(b"".join((scratch.ROOT / name).read_bytes() for name in files)).hexdigest()
    assert digest == anchors["pack_closure"]["sha256"]
    baseline = scratch.load_json(scratch.P2O / "acceptance-baseline.json")
    assert baseline["prompt_sha1"] == "e9d28ff61819a041610d763f5c1464ea860265be2d414f78c15a1f8aea1fdbcf"
    assert baseline["pack_closure_files"] == 79
    assert baseline["pack_closure_sha256"] == "5512395852ea7608a6075e2b6b45dca9bd85f112210af336cf6f8f77f5fd811f"
    assert baseline["trust_toml_sha256"] == hashlib.sha256(
        (scratch.ROOT / "targets/apple-m4-max/trust.toml").read_bytes()
    ).hexdigest()
    assert baseline["publication_allowed"] is False
    assert baseline["npu_simulated_only"] is True
