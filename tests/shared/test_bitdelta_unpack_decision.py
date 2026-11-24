"""CPU replay: bitdelta_gemm REG sources, the unpack probe and the negative decision."""

from __future__ import annotations

import hashlib

from tests.support import bitdelta_unpack as unpack


def _subjects() -> dict[str, dict]:
    return {
        "small": unpack.load_json(unpack.SMALL_CONFIRMATION),
        "paper": unpack.load_json(unpack.PAPER_CONFIRMATION),
    }


def test_reg_candidates_decode_in_tile_and_declare_no_scratch():
    """@brief 两个 REG 候选都是单 stage、零 scratch、tile 内解包；paper 仍与已登记 GEMM fixture 同源。"""
    unpack.assert_source_provenance()
    unpack.assert_no_shared_candidate()
    for size in unpack.SIZES:
        source = unpack.stored_source(size)
        body = unpack.launch_body(size)
        assert body.count("launch ") == 1
        assert "scratch" not in body
        assert "simdgroup_multiply_accumulate" in source
        assert "bitdelta_gemm_kernel" in source
        assert unpack.candidate_sha256(size) == (
            unpack.SMALL_SHA256 if size == "small" else unpack.PAPER_SHA256
        )
    # The decode mapping is the OpSpec formula: bit (k mod 32) of bits[k//32, j] is +1.
    assert "((w >> j) & 1u) != 0u) ? half(1.0f) : half(-1.0f)" in unpack.stored_source("small")
    assert "0x3C00u | ((((w >> j) & 1u) ^ 1u) << 15)" in unpack.stored_source("paper")


def test_probe_covers_every_decoded_element_and_audits_its_b_staging():
    """@brief probe 不是 candidate；三个展开臂逐元素覆盖，B staging 恰好一格，r1 的溢出被记录。"""
    document = unpack.load_json(unpack.PROBE)
    unpack.assert_probe(document)
    assert document["not_a_candidate"] is True
    assert document["publication_allowed"] is False
    assert unpack.R1_STAGING["over_read_ratio"] == 2.0
    assert unpack.R1_STAGING["threadgroup_overrun_bytes"] == 4592
    for size, one in document["sizes"].items():
        m, n, k = unpack.SHAPES[size]
        # Independent derivation from the candidate bytes: grid, tile constants and the
        # lane-per-word mapping. Never reuse the SHARED tiling for the REG candidate.
        reg = unpack.source_geometry(size)
        shared = unpack.shared_geometry(size)
        assert reg["grid_x"] * reg["tile_n"] == n and reg["grid_y"] * reg["tile_m"] == m
        assert (reg["tile_m"], reg["tile_n"]) == (32, 32) if size == "small" else (64, 64)
        assert reg["lanes_per_word"] == (4 if size == "small" else 2)
        assert one["packed_bytes"] == 4 * (k // 32) * n
        assert one["expand_ratio_fp16"] == 16.0
        assert one["traffic"]["reg_tiling"] == {
            key: reg[key]
            for key in ("tile_m", "tile_n", "row_blocks", "col_blocks", "output_threadgroups", "lanes_per_word")
        }, size
        assert one["traffic"]["shared_tiling"] == shared, size
        assert one["traffic"]["decode"]["reg_repeats"] == reg["row_blocks"]
        assert one["traffic"]["decode"]["reg_decoded_elements"] == reg["row_blocks"] * k * n
        assert one["traffic"]["decode"]["shared_decoded_elements"] == k * n
        assert one["traffic"]["reg"]["packed_read_bytes"] == reg["row_blocks"] * one["packed_bytes"]
        assert one["traffic"]["reg"]["packed_read_bytes_issued"] == (
            reg["lanes_per_word"] * reg["row_blocks"] * one["packed_bytes"]
        )
        assert one["traffic"]["reg"]["x_read_bytes"] == reg["col_blocks"] * 2 * m * k
        assert one["traffic"]["shared_fp16"]["x_read_bytes"] == shared["col_blocks"] * 2 * m * k
        assert one["traffic"]["reg"]["output_threadgroups"] == reg["output_threadgroups"]
        assert one["traffic"]["shared_fp16"]["output_threadgroups"] == shared["output_threadgroups"]
        audit = one["b_staging_audit"]
        assert audit["rows_written"] == audit["rows_declared"] == unpack.BLOCK_K
        assert audit["over_read_ratio"] == 1.0
        assert audit["threadgroup_overrun_bytes"] == 0
        assert audit["r1_defect_overrun_bytes"] == unpack.R1_STAGING["threadgroup_overrun_bytes"]
        for name in ("expand_fp32", "expand_fp16", "shared_packed16"):
            coverage = one["arms"][name]["coverage"]
            assert coverage["expected_elements"] == k * n
            assert coverage["read_back_elements"] == k * n
            assert coverage["mismatched_elements"] == 0
            assert coverage["ok"] is True
        assert one["output_agreement"]["packed_vs_shared_packed16_equal"] is True
        assert one["output_agreement"]["packed_vs_shared_fp32_equal"] is True
        assert one["output_agreement"]["packed_vs_shared_fp16_equal"] is True
    unpack.assert_superseded()


def test_paired_probe_measures_every_legal_shared_form_slower_than_reg():
    """@brief 同轮配对的 A/B：每个合法 SHARED 都更慢；paper 排序稳定，small 只看数量级。"""
    document = unpack.load_json(unpack.PROBE)
    unpack.assert_probe(document)
    for size, one in document["sizes"].items():
        ratios = one["interleaved"]["paired_ratio_over_reg"]
        assert one["interleaved"]["rounds"] == 12
        for name, ratio in ratios.items():
            assert ratio["median"] > 1.0, (size, name, ratio["median"])
            assert ratio["rounds_over_one"] >= 11, (size, name, ratio["rounds_over_one"])
        # At small the two cheapest legal forms are noise-comparable (7.94 vs 8.41, and they have
        # swapped places between runs); only paper's ordering is asserted.
        if size == "paper":
            assert ratios["shared_packed16"]["median"] < ratios["shared_fp32"]["median"]
            assert ratios["shared_fp32"]["median"] < ratios["shared_fp16"]["median"]
        else:
            cheapest = min(ratios["shared_packed16"]["median"], ratios["shared_fp32"]["median"])
            dearest = max(ratios["shared_packed16"]["median"], ratios["shared_fp32"]["median"])
            assert dearest / cheapest < 1.25
        no_fill = one["no_fill"]["shared_fp16"]["wall_us_median"]
        assert no_fill < one["arms"]["shared_fp16"]["wall_us_median"]
        assert one["no_fill"]["shared_packed16"]["wall_us_median"] < one["arms"]["shared_fp16"]["wall_us_median"]
    paper = document["sizes"]["paper"]
    # Magnitude bound only: the cheapest SHARED form is slightly (not dramatically) slower at paper.
    # This is not the harness gate; the registered-baseline gate is checked separately below.
    assert 1.0 < paper["interleaved"]["paired_ratio_over_reg"]["shared_packed16"]["median"] < 1.06
    small = document["sizes"]["small"]
    assert small["interleaved"]["paired_ratio_over_reg"]["shared_packed16"]["median"] > 4.0


def test_replay_corroborates_the_stable_comparisons_and_names_the_noise():
    """@brief 归档重放只做旁证：paper 排序复现；small 两个最省形态不可分辨，不作排序声明。"""
    wrapper = unpack.load_json(unpack.PROBE_REPLAY)
    unpack.assert_replay(wrapper)
    replay = wrapper["probe"]
    archived = unpack.load_json(unpack.PROBE)
    for size in unpack.SIZES:
        runs = {
            "archived": archived["sizes"][size]["interleaved"]["paired_ratio_over_reg"],
            "replay": replay["sizes"][size]["interleaved"]["paired_ratio_over_reg"],
        }
        for name, ratios in runs.items():
            assert ratios["shared_fp16"]["median"] > 1.0, (size, name)
        if size == "paper":
            for ratios in runs.values():
                assert ratios["shared_packed16"]["median"] < ratios["shared_fp32"]["median"]
                assert ratios["shared_packed16"]["median"] < 1.06  # magnitude bound, not the gate
        else:
            for ratios in runs.values():
                cheapest = min(ratios["shared_packed16"]["median"], ratios["shared_fp32"]["median"])
                dearest = max(ratios["shared_packed16"]["median"], ratios["shared_fp32"]["median"])
                assert cheapest > 4.0 and dearest / cheapest < 1.25


def test_decision_keeps_reg_because_every_legal_shared_form_is_slower():
    """@brief 决策由 fresh + probe 重算；合法 SHARED 全部更慢，可回收上限只有个位数。"""
    subjects = _subjects()
    probe = unpack.load_json(unpack.PROBE)
    document = unpack.load_json(unpack.DECISION)
    unpack.assert_decision(document, subjects, probe)
    assert document["adopted_route"] == "REG" and document["decision"] == "reg_only"
    assert document["implementation"]["reg_candidate_added"] is True
    assert document["implementation"]["shared_candidate_source_added"] is False
    assert document["implementation"]["same_algorithm_ab_run"] is False
    assert document["publication_allowed"] is False
    assert document["adoption_bar_ratio_over_reg"] == 1.0
    assert document["registered_baseline_gate_ratio"] == 1.10
    assert "not the harness gate" in document["adoption_bar"]
    for size, one in document["sizes"].items():
        assert one["shared_would_win"] is False
        assert one["meets_adoption_bar"] is False
        assert one["adoption_bar_ratio_over_reg"] == 1.0
        assert one["best_legal_shared_ratio_over_reg"] > 1.0
        assert one["best_legal_shared_estimate_us"] > one["reg_candidate_us"]
        assert one["probe_estimates_are_not_candidates"] is True
        assert one["reg_decode_repeats"] >= 2
        assert "best_legal_shared_measured_slower" in one["stop_reasons"]
        assert "every_measured_legal_shared_form_slower" in one["stop_reasons"]
        for name, arm in one["measured_legal_shared"].items():
            assert arm["paired_ratio_over_reg"] > 1.0, (size, name)
            assert arm["paired_rounds_slower_than_reg"] >= 11, (size, name)
    paper = document["sizes"]["paper"]
    assert paper["recoverable_decode_headroom_fraction"] < 0.10
    assert "recoverable_decode_headroom_single_digit" in paper["stop_reasons"]
    # The registered-baseline gate is context, never the reason: the paper SHARED estimate would
    # pass it (0.9685 of the fastest baseline), which is exactly why the decision cannot cite it.
    # The registered-baseline gate would NOT exclude a paper SHARED estimate: record that honestly
    # instead of turning an adoption bar into the harness gate.
    assert paper["best_legal_shared_estimate_over_best_baseline"] <= 1.10
    assert paper["best_legal_shared_would_pass_registered_baseline_gate"] is True
    assert document["sizes"]["small"]["best_legal_shared_would_pass_registered_baseline_gate"] is False
    assert paper["reg_over_roofline"] < 1.10
    assert paper["fp16_fill_marginal_us"] > 0
    assert paper["subtraction_error_us"] > 200.0
    assert paper["measured_no_fill_fp16_us"] > paper["roofline_lower_bound_us"]
    small = document["sizes"]["small"]
    assert small["reg_decoded_elements"] == 4 * small["shared_decoded_elements"]
    assert paper["reg_decoded_elements"] == 8 * paper["shared_decoded_elements"]
    # The small REG kernel is 32x32 (four row blocks); the SHARED consumer is 64x64 (two). The two
    # geometries must be recorded separately, which is exactly the accounting bug this pins.
    assert small["reg_decode_repeats"] == 4
    assert small["reg_tiling"]["tile_m"] == 32 and small["shared_tiling"]["tile_m"] == 64
    assert small["reg_tiling"]["output_threadgroups"] == 32
    assert small["shared_tiling"]["output_threadgroups"] == 8
    assert small["reg_x_read_bytes"] == 2 * small["shared_x_read_bytes"]
    assert paper["reg_tiling"]["output_threadgroups"] == 512 == paper["shared_tiling"]["output_threadgroups"]
    assert small["best_legal_shared_ratio_over_reg"] > 4.0
    assert small["two_stage_floor_over_reg"] > 1.0


def test_fresh_subjects_are_six_case_confirmed_and_not_published():
    """@brief 两档 fresh 都是六 case（含 IEEE）、identity match、非 7 的 fresh seed。"""
    for size, subject in _subjects().items():
        unpack.assert_subject(subject, size)
        assert subject["status"] == "confirmed"
        assert subject["confirmation_seed"] != subject["search_seed"]
        assert subject["confirmation_attempts"] == 1
        assert (subject["identity"] or {}).get("kind") == "match"
        assert "publication_allowed" not in subject
        cases = subject["confirmation_report"]["correctness"]["cases"]
        assert {item["case"] for item in cases} >= {"ieee", "tiny", "outlier"}


def test_prompt_pack_and_trust_boundaries_are_unchanged():
    """@brief live 锚点是当前 78 文件；P2-M 的 79 文件与 trust 哈希是历史锚点。"""
    from taic.trust import current_prompt_sha1
    from tcenv.taskcontract.pack import code_import_closure

    from tests.support.live_anchors import load_live_anchors
    anchors = load_live_anchors(unpack.ROOT)
    assert current_prompt_sha1() == anchors["prompt_sha1"]
    files = code_import_closure(unpack.ROOT)
    assert list(files) == anchors["pack_closure"]["files"]
    assert len(files) == 78
    digest = hashlib.sha256(b"".join((unpack.ROOT / name).read_bytes() for name in files)).hexdigest()
    assert digest == anchors["pack_closure"]["sha256"]
    baseline = unpack.load_json(unpack.P2P / "acceptance-baseline.json")
    assert baseline["prompt_sha1"] == "e9d28ff61819a041610d763f5c1464ea860265be2d414f78c15a1f8aea1fdbcf"
    assert baseline["pack_closure_files"] == 79
    assert baseline["pack_closure_sha256"] == "5512395852ea7608a6075e2b6b45dca9bd85f112210af336cf6f8f77f5fd811f"
    assert baseline["trust_toml_sha256"] == hashlib.sha256(
        (unpack.ROOT / "targets/apple-m4-max/trust.toml").read_bytes()
    ).hexdigest()
    assert baseline["publication_allowed"] is False
    assert baseline["npu_simulated_only"] is True
