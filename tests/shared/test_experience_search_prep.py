"""P2-S: verified experience becomes a frozen search input, not a live prompt edit."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from experiments.experience_catalog import (
    CONTEXT_RULE,
    DECISION_SIZE_ROWS,
    KDA_FRESH_ARMS,
    OVERLAY_MARK,
    _fmt_pct,
    _fmt_ratio,
    _fmt_us,
    _fresh,
    approval_document,
    build_catalog,
    dumps,
    prompt_boundary,
    require_seq_size_rows,
)
from experiments.search_yield_ab import (
    CONTROL_OVERRIDES,
    TREATMENT_OVERRIDES,
    build_experience_plan,
    build_plan,
    main,
)
from taic.experience_prompt import filter_experience_for_prompt
from taic.trust import current_prompt_sha1

ROOT = Path(__file__).resolve().parents[2]
P2S = ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2-S"
EVIDENCE = P2S / "evidence"
P2M_EPOCH = ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2-M/prompt-epoch.json"
EXPERIENCE = ROOT / "targets/apple-m4-max/experience.md"
FORBIDDEN = ("2.765", "1315", "unrunnable", "best_speedup", "search best", "publication_allowed=true")


def _load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_catalog_keeps_probe_and_search_best_out_of_the_taught_speedup():
    """Taught rules trace to fresh subjects. Probe floors and search best stay excluded."""
    catalog, overlay = build_catalog()
    assert json.loads(dumps(catalog)) == _load("catalog.json")
    assert (EVIDENCE / "experience-overlay.md").read_text(encoding="utf-8") == overlay
    assert catalog["teachable_ids"] == [
        "conv2d-paper-shared-u",
        "conv2d-small-single-stage",
        "gemm-silu-reg",
        "bitdelta-reg",
        "gemm-splitk-stop",
        "seq-serial-handoff",
    ]
    assert catalog["publication_allowed"] is False
    assert catalog["paid_search"] == "未授权/未执行"
    by_id = {item["id"]: item for item in catalog["lessons"]}
    assert by_id["conv2d-paper-scalar-ablation"]["teach"] is False
    assert min(arm["median_us"] for arm in by_id["conv2d-paper-scalar-ablation"]["arms"]) > 10000
    assert by_id["gemm-silu-occupied-retry"]["teach"] is False
    assert by_id["bitdelta-small-search-best"]["evidence_class"] == "search_best"
    assert by_id["bitdelta-small-search-best"]["teach"] is False
    assert by_id["p2r-worker-geometry"]["teach"] is False
    assert by_id["p2r-worker-geometry"]["evaluation_device_verdict"] is False
    assert by_id["seq-tail-decision-measurement"]["teach"] is False
    assert by_id["seq-tail-decision-measurement"]["evidence_class"] == "decision_measurement"
    assert by_id["seq-single-self-check"]["teach"] is False
    assert by_id["seq-single-self-check"]["evidence_class"] == "raw_self_check"
    assert by_id["seq-single-self-check"]["in_decision_sizes"] is False
    shared = by_id["conv2d-paper-shared-u"]
    assert shared["evidence_class"] == "fresh_subject_pair"
    assert shared["exact_ratio_recompute_over_global"] == pytest.approx(
        shared["recompute"]["median_us"] / shared["global_u"]["median_us"]
    )
    assert shared["global_u"]["confirmation_seed"] != 7
    small = by_id["conv2d-small-single-stage"]
    assert small["two_stage_floor"]["evidence_class"] == "probe_floor"
    assert small["two_stage_floor"]["not_a_candidate"] is True
    assert small["speedup_vs_best_baseline"] < 1
    gemm = by_id["gemm-silu-reg"]
    assert gemm["sizes"]["small"]["lower_bound_evidence_class"] == "probe_floor"
    assert gemm["sizes"]["paper"]["lower_bound_evidence_class"] == "traffic_bound"
    assert gemm["global_candidate_source_added"] is False
    assert by_id["gemm-splitk-stop"]["claim_kind"] == "do_not_implement"
    assert by_id["bitdelta-reg"]["sizes"]["paper"]["ratio_evidence_class"] == "paired_probe"
    for item in catalog["lessons"]:
        assert item["publication_allowed"] is False
        if item["teach"]:
            assert item["claim_kind"] != "confirmed_speedup"
    for token in FORBIDDEN:
        assert token not in overlay
    assert OVERLAY_MARK in overlay
    assert OVERLAY_MARK not in EXPERIENCE.read_text(encoding="utf-8")
    assert "不是 confirmed speedup" in overlay or "也不是 confirmed speedup" in overlay
    assert catalog["experiment"]["task"] == "gemm_silu"
    assert catalog["experiment"]["size"] == "small"
    assert catalog["experiment"]["reg_speedup_vs_best"] > 1.5
    assert catalog["overlay_context_rule"] == CONTEXT_RULE
    assert CONTEXT_RULE in overlay
    assert by_id["conv2d-paper-scalar-ablation"]["overlay_role"] == "counterexample_context"
    assert by_id["gemm-silu-occupied-retry"]["overlay_role"] == "catalog_only"
    assert by_id["bitdelta-small-search-best"]["overlay_role"] == "catalog_only"
    assert by_id["seq-tail-decision-measurement"]["overlay_role"] == "boundary_context"
    assert by_id["seq-single-self-check"]["overlay_role"] == "boundary_context"
    assert by_id["p2r-worker-geometry"]["overlay_role"] == "boundary_context"
    assert "同一份 device 源码" not in overlay
    assert "same device source" not in json.dumps(catalog)


def _build_files(report: dict) -> dict[str, str]:
    found: dict[str, str] = {}

    def take(path, digest) -> None:
        if isinstance(path, str) and isinstance(digest, str):
            found[path] = digest

    def walk(value) -> None:
        if isinstance(value, dict):
            take(value.get("path"), value.get("sha256"))
            take(value.get("entry_path"), value.get("entry_sha256"))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(report["build"])
    return found


def test_seq_controls_are_recomputed_from_artifacts_and_varlen_source():
    """KDA keeps one device file. Varlen changes the plan kernel, not a host program."""
    from tests.support.seq_handoff import VARLEN_PIECE_LINE, varlen_source

    confirmations = json.loads(
        (ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/evidence/confirmations.json").read_text(
            encoding="utf-8"
        )
    )
    subjects = confirmations["subjects"]
    kda_keys = [key for key in subjects if key.startswith("kda:")]
    kda_files = {key: _build_files(subjects[key]["confirmation_report"]) for key in kda_keys}
    device_shas = {files["device/host_piece_handoff.metal"] for files in kda_files.values()}
    assert device_shas == {"124c7485cbf2feb073e6a245042e09e7f7b9400ee1fd1b17c7543ced5e347b54"}
    three_piece_host = {kda_files[key]["host/tiling.c"] for key in kda_keys if key.endswith("/seq_three_piece")}
    single_piece_host = {kda_files[key]["host/tiling.c"] for key in kda_keys if "/single_piece_" in key}
    three_stage_geometry = {
        kda_files[key]["host/execution_geometry.json"]
        for key in kda_keys
        if not key.endswith("/single_piece_single_stage")
    }
    one_stage_geometry = {
        kda_files[key]["host/execution_geometry.json"]
        for key in kda_keys
        if key.endswith("/single_piece_single_stage")
    }
    assert len(three_piece_host) == len(single_piece_host) == 1
    assert three_piece_host != single_piece_host
    assert len(three_stage_geometry) == len(one_stage_geometry) == 1
    assert three_stage_geometry != one_stage_geometry
    three = varlen_source("pieces_three")
    one = varlen_source("pieces_one")
    added = "    uint pieces = 1u;"
    assert three.replace(VARLEN_PIECE_LINE, added, 1) == one
    assert three.count(VARLEN_PIECE_LINE) == 1
    plan_at = three.find("kernel void varlen_plan")
    line_at = three.find(VARLEN_PIECE_LINE)
    next_kernel = three.find("\nkernel void", plan_at + 1)
    assert plan_at < line_at < (next_kernel if next_kernel > 0 else len(three))
    three_sha = hashlib.sha256(three.encode("utf-8")).hexdigest()
    one_sha = hashlib.sha256(one.encode("utf-8")).hexdigest()
    assert _build_files(subjects["varlen_kda:mixed/pieces_three"]["confirmation_report"])["device/candidate.metal"] == three_sha
    assert _build_files(subjects["varlen_kda:mixed/pieces_one"]["confirmation_report"])["device/candidate.metal"] == one_sha
    assert "host/tiling.c" not in _build_files(subjects["varlen_kda:mixed/pieces_three"]["confirmation_report"])
    assert "host/tiling.c" not in _build_files(subjects["varlen_kda:mixed/pieces_one"]["confirmation_report"])
    catalog, overlay = build_catalog()
    controls = next(item for item in catalog["lessons"] if item["id"] == "seq-serial-handoff")["controls"]
    assert controls["kda"]["device_bytes_identical"] is True
    assert controls["kda"]["device_sha256"] == next(iter(device_shas))
    assert controls["kda"]["host_program_sha256_by_arm"] == {
        "seq_three_piece": next(iter(three_piece_host)),
        "single_piece": next(iter(single_piece_host)),
    }
    assert controls["kda"]["host_geometry_sha256_by_stage_count"] == {
        "3": next(iter(three_stage_geometry)),
        "1": next(iter(one_stage_geometry)),
    }
    assert controls["varlen"]["device_bytes_identical"] is False
    assert controls["varlen"]["host_program_present"] is False
    assert controls["varlen"]["pieces_three_sha256"] == three_sha
    assert controls["varlen"]["pieces_one_sha256"] == one_sha
    assert controls["varlen"]["diff"] == {
        "kernel": "varlen_plan",
        "removed_line": VARLEN_PIECE_LINE,
        "added_line": added,
        "only_text_change": True,
    }
    for digest in (next(iter(device_shas)), next(iter(three_piece_host)), three_sha, one_sha):
        assert digest[:8] in overlay
    assert "没有 Host Program" in overlay
    assert "这不是 Host Program 改切链" in overlay
    assert "uint pieces = min(MAX_SPLIT, len);" in overlay
    assert "uint pieces = 1u;" in overlay


def test_counterexample_context_stays_in_the_overlay_and_is_not_a_lesson():
    """Scalar arms and probe floors are visible context. Excluded lessons stay out."""
    ablation = json.loads(
        (ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2-I/evidence/conv2d-ablation.json").read_text(
            encoding="utf-8"
        )
    )
    floor = json.loads(
        (ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2-J/evidence/launch-floor.json").read_text(
            encoding="utf-8"
        )
    )
    catalog, overlay = build_catalog()
    assert catalog["overlay_context_rule"] in overlay
    labels = []
    for arm in ablation["arms"]:
        shown = f"{float(arm['confirmation']['median_us']):.2f}"
        assert arm["label"] in overlay
        assert shown in overlay
        labels.append(arm["label"])
    assert labels == ["recompute-f2", "scratch-f2", "scratch-f4"]
    wall = float(floor["arms"]["two_stage_no_fill"]["wall_us_median"])
    assert f"{wall:.4f}" in overlay
    assert "probe 下界" in overlay
    scalar = next(item for item in catalog["lessons"] if item["id"] == "conv2d-paper-scalar-ablation")
    assert scalar["teach"] is False
    assert scalar["overlay_role"] == "counterexample_context"
    assert "conv2d-paper-scalar-ablation" not in catalog["teachable_ids"]


def test_prompt_fingerprint_stays_on_p2m_and_the_overlay_is_only_a_counterfactual():
    """The live fingerprint and the P2-M render stay put. Treatment is a filtered copy."""
    _catalog, overlay = build_catalog()
    boundary = prompt_boundary(overlay=overlay)
    assert json.loads(dumps(boundary)) == _load("prompt-boundary.json")
    epoch = json.loads(P2M_EPOCH.read_text(encoding="utf-8"))
    assert boundary["prompt_changed"] is False
    from tests.support.live_anchors import load_live_anchors
    live = load_live_anchors(ROOT)["prompt_sha1"]
    assert boundary["previous_prompt_sha1"] == epoch["current_prompt_sha1"]
    assert boundary["current_prompt_sha1"] == live
    assert current_prompt_sha1() == live
    assert live != epoch["previous_prompt_sha1"]
    assert all(boundary["p2m_rendered_prompts_match"].values())
    assert boundary["control_filtered_experience_sha256"] != boundary["treatment_filtered_experience_sha256"]
    assert boundary["treatment_filtered_bytes"] > boundary["control_filtered_bytes"]
    assert boundary["experience_file_sha256"] == hashlib.sha256(EXPERIENCE.read_bytes()).hexdigest()
    assert boundary["system_prompt_identical_across_arms"] is True
    assert "experience.md" in boundary["session_visible_difference"]


def test_approval_stays_blank_and_experience_dry_run_has_one_difference():
    """The paired plan is not an approval, and the four search switches do not move."""
    approval = _load("approval.json")
    assert approval == approval_document()
    assert approval["delta"] is None
    assert approval["approved_by"] is None
    assert approval["executed"] is False
    assert approval["paid_search"] == "未授权/未执行"
    assert approval["execution_record"] is None
    overlay = EVIDENCE / "experience-overlay.md"
    digest = hashlib.sha256(overlay.read_bytes()).hexdigest()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    plan = build_experience_plan(
        overlay_sha256=digest,
        overlay_path="tests/support/golden/shared/shape-portfolio-阶段验收/P2-S/evidence/experience-overlay.md",
        git_head=head,
    )
    archived = _load("dry-run-plan.json")
    assert "injection_ready" not in archived
    live = json.loads(dumps(plan))
    assert live.pop("injection_ready") is True
    injection = live.pop("injection")
    assert injection["ready"] is True
    assert injection["global_experience_md"] is False
    assert injection["changes_current_prompt_sha1"] is False
    assert injection["control_copies"] == 0
    assert injection["treatment_copies"] == 1
    assert injection["authoritative_task_rules"] is False
    assert injection["live_search"] == "未授权/未执行"
    assert live == archived
    assert plan["control"] == plan["treatment"] == CONTROL_OVERRIDES
    assert plan["treatment"] != TREATMENT_OVERRIDES
    assert plan["factor"] == "experience-overlay"
    assert plan["tasks"] == ["gemm_silu"]
    assert plan["dtype"] == "fp16"
    assert plan["size"] == "small"
    assert plan["expected_pairs"] == 2
    assert plan["nominal_total_usd"] == 12.0
    assert plan["max_total_usd"] == 12.0
    assert plan["delta"] is None
    assert plan["paid_search"] == "未授权/未执行"
    assert plan["executed"] is False
    assert [cell["order"] for cell in plan["cells"][::2]] == ["AB", "BA"]
    for cell in plan["cells"]:
        assert cell["overrides"] == CONTROL_OVERRIDES
        assert cell["budget_usd"] == 3.0
        if cell["arm"] == "treatment":
            assert cell["experience_overlay_sha256"] == digest
        else:
            assert cell["experience_overlay_sha256"] is None
    switches = build_plan(git_head="abc")
    assert "factor" not in switches
    assert switches["treatment"] == TREATMENT_OVERRIDES


def test_experience_overlay_execute_refuses_even_when_the_four_gates_are_present(tmp_path, capsys, monkeypatch):
    """Missing gates still refuse. A filled gate reaches the fake runner and not run_one."""
    calls = []

    def runner(*args, **kwargs):
        calls.append((args, kwargs))

    def explode(*args, **kwargs):
        raise AssertionError("run_one was called")

    monkeypatch.setattr("experiments.common.run_one", explode)
    produced = 0
    out = ROOT / "runs" / f"_p2s_{hashlib.sha256(str(tmp_path).encode()).hexdigest()[:8]}"
    overlay = EVIDENCE / "experience-overlay.md"
    argv = [
        "--out", str(out),
        "--execute",
        "--allow-spend",
        "--allow-gpu",
        "--approved-by", "Ada Lovelace",
        "--delta", "0.05",
        "--factor", "experience-overlay",
        "--overlay", str(overlay),
        "--tasks", "gemm_silu",
        "--size", "small",
        "--pairs", "2",
        "--max-total-usd", "12",
    ]
    try:
        assert main([item for item in argv if item != "--allow-gpu"], runner=runner) == 2
        assert calls == []
        assert not out.exists()
        assert main(argv, runner=runner) == 0
        assert len(calls) == 4
        assert {kwargs["seed"] for _args, kwargs in calls} == {7}
        treated = [kwargs for _args, kwargs in calls if kwargs.get("experience_overlay_path")]
        assert len(treated) == 2
        assert all(kwargs["experience_overlay_sha256"] == hashlib.sha256(overlay.read_bytes()).hexdigest() for kwargs in treated)
        assert list(out.rglob("result.json")) == []
        produced = len(calls)
    finally:
        if out.exists():
            import shutil
            shutil.rmtree(out)
    dry = ROOT / "runs" / f"_p2s_dry_{hashlib.sha256(str(tmp_path).encode()).hexdigest()[:8]}"
    try:
        assert main([
            "--out", str(dry),
            "--dry-run",
            "--factor", "experience-overlay",
            "--overlay", str(overlay),
            "--tasks", "gemm_silu",
            "--size", "small",
            "--pairs", "2",
            "--budget-usd", "3",
            "--max-total-usd", "12",
        ], runner=runner) == 0
        assert len(calls) == produced
        written = json.loads((dry / "plan.json").read_text(encoding="utf-8"))
        assert written["paid_search"] == "未授权/未执行"
        assert written["injection_ready"] is True
        assert written["executed"] is False
        assert written["delta"] is None
        assert written["approved_by"] is None
        assert written["overlay_sha256"] == hashlib.sha256(overlay.read_bytes()).hexdigest()
    finally:
        import shutil
        shutil.rmtree(dry, ignore_errors=True)


Q_EVIDENCE = ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/evidence"
CONTRACT = ROOT / "docs/架构/verified-experience-search-prep.md"
# Same three checks as tests/shared/test_wiki_experience.py. Do not loosen that file.
_VERIFIED_SOURCE = re.compile(r"^来源:.*(`runs/|`docs/|`tcenv/|`taic/)", re.MULTILINE)
_VERIFIED_MEASURED = re.compile(r"^本机实测:", re.MULTILINE)
_VERIFIED_BOUNDARY = re.compile(r"^边界:", re.MULTILINE)


def _verified_gaps(body: str) -> list[str]:
    missing = []
    if _VERIFIED_SOURCE.search(body) is None:
        missing.append("source")
    if _VERIFIED_MEASURED.search(body) is None:
        missing.append("measurement")
    if _VERIFIED_BOUNDARY.search(body) is None:
        missing.append("boundary")
    return missing


def _verified_sections(text: str) -> list[tuple[str, str]]:
    chunks = text.split("\n## ")
    found = []
    for index, chunk in enumerate(chunks):
        if index == 0 and not text.startswith("## "):
            continue
        title, _, body = chunk.partition("\n")
        title = title.removeprefix("## ").strip()
        if re.search(r"^Confidence:\s*verified", body, re.MULTILINE):
            found.append((title, body))
    return found


def _section(text: str, title_prefix: str) -> str:
    for title, body in _verified_sections(text):
        if title.startswith(title_prefix):
            return body
    raise AssertionError(title_prefix)


def test_missing_decision_size_is_not_skipped():
    """A missing decision row raises. kda single is never treated as one of those rows."""
    decision = json.loads((Q_EVIDENCE / "decision.json").read_text(encoding="utf-8"))
    require_seq_size_rows(decision["sizes"])
    assert set(decision["sizes"]) == set(DECISION_SIZE_ROWS)
    assert "kda_single" not in decision["sizes"]
    dropped = dict(decision["sizes"])
    dropped.pop("kda_tail")
    with pytest.raises(ValueError, match="kda_tail"):
        require_seq_size_rows(dropped)
    extra = dict(decision["sizes"])
    extra["kda_single"] = {}
    with pytest.raises(ValueError, match="kda_single"):
        require_seq_size_rows(extra)
    catalog, overlay = build_catalog()
    by_id = {item["id"]: item for item in catalog["lessons"]}
    tail = by_id["seq-tail-decision-measurement"]
    single = by_id["seq-single-self-check"]
    assert tail["evidence_class"] == "decision_measurement"
    assert tail["in_decision_sizes"] is True
    assert tail["in_fresh_confirmation_set"] is False
    assert tail["seq_overhead_fraction_of_single_stage"] == decision["sizes"]["kda_tail"]["seq_overhead_fraction_of_single_stage"]
    assert single["evidence_class"] == "raw_self_check"
    assert single["in_decision_sizes"] is False
    assert single["pieces_per_chain"] == [1]
    assert "kda single 不在 decision.sizes" in overlay
    assert "kda tail 在 decision.sizes" in overlay
    assert "都在测量决策" not in overlay
    assert "出现在测量决策里" not in overlay


def test_kda_fresh_arms_split_handoff_from_stage_floor():
    """B is a fresh subject. A−C is total overhead across different confirmation seeds."""
    confirmations = json.loads((Q_EVIDENCE / "confirmations.json").read_text(encoding="utf-8"))
    decision = json.loads((Q_EVIDENCE / "decision.json").read_text(encoding="utf-8"))
    measurements = json.loads((Q_EVIDENCE / "seq-handoff-measurements.json").read_text(encoding="utf-8"))
    subjects = confirmations["subjects"]
    keys = [f"kda:{size}/{arm}" for size in ("small", "paper") for arm in KDA_FRESH_ARMS]
    keys.extend(("varlen_kda:mixed/pieces_three", "varlen_kda:mixed/pieces_one"))
    assert len(keys) == 8
    fresh = {key: _fresh(subjects[key], source=key) for key in keys}
    assert fresh["kda:small/seq_three_piece"]["median_us"] == pytest.approx(246.6388, abs=1e-3)
    assert fresh["kda:small/single_piece_three_stage"]["median_us"] == pytest.approx(224.8069, abs=1e-3)
    assert fresh["kda:small/single_piece_single_stage"]["median_us"] == pytest.approx(205.5133, abs=1e-3)
    assert fresh["kda:paper/seq_three_piece"]["median_us"] == pytest.approx(188422.0367, abs=1e-3)
    assert fresh["kda:paper/single_piece_three_stage"]["median_us"] == pytest.approx(187407.4533, abs=1e-3)
    assert fresh["kda:paper/single_piece_single_stage"]["median_us"] == pytest.approx(187275.1250, abs=1e-3)
    catalog, overlay = build_catalog()
    seq = next(item for item in catalog["lessons"] if item["id"] == "seq-serial-handoff")
    assert seq["fresh_subject_count"] == 8
    assert "not an isolated handoff" in seq["protocol"]
    rows = {row["name"]: row for row in seq["confirmed_rows"]}
    for size in ("small", "paper"):
        row = rows[f"kda_{size}"]
        arm_a = fresh[f"kda:{size}/seq_three_piece"]
        arm_b = fresh[f"kda:{size}/single_piece_three_stage"]
        arm_c = fresh[f"kda:{size}/single_piece_single_stage"]
        assert len({arm_a["confirmation_seed"], arm_b["confirmation_seed"], arm_c["confirmation_seed"]}) == 3
        assert row["fresh_a_us"] == arm_a["median_us"]
        assert row["fresh_b_us"] == arm_b["median_us"]
        assert row["fresh_c_us"] == arm_c["median_us"]
        assert row["ac_is_isolated_handoff"] is False
        assert row["same_input_paired_run"] is False
        assert row["fresh_ab_us"] == pytest.approx(arm_a["median_us"] - arm_b["median_us"])
        assert row["fresh_bc_us"] == pytest.approx(arm_b["median_us"] - arm_c["median_us"])
        assert row["fresh_ac_total_overhead_fraction"] == pytest.approx(
            (arm_a["median_us"] - arm_c["median_us"]) / arm_c["median_us"]
        )
        measured = decision["sizes"][f"kda_{size}"]
        assert measured["handoff_us"] == pytest.approx(measured["seq_three_piece_us"] - measured["single_piece_three_stage_us"])
        assert measured["stage_floor_us"] == pytest.approx(measured["single_piece_three_stage_us"] - measured["single_stage_us"])
        assert row["measurement_handoff_us"] == measured["handoff_us"]
        assert row["measurement_stage_floor_us"] == measured["stage_floor_us"]
        assert abs(row["fresh_a_us"] - row["measurement_seq_us"]) > 0.01
        for seed in (row["fresh_a_seed"], row["fresh_b_seed"], row["fresh_c_seed"]):
            assert str(seed) in overlay
        assert _fmt_us(row["fresh_b_us"]) in overlay
        assert _fmt_us(row["fresh_ab_us"]) in overlay
        assert _fmt_us(row["fresh_bc_us"]) in overlay
    body = _section(overlay, "因果状态")
    assert "不能解释为隔离的 handoff 成本" in body
    assert "不是同一次输入的 paired run" in body
    assert _fmt_pct(rows["kda_small"]["fresh_ac_total_overhead_fraction"]) in body
    assert _fmt_pct(rows["kda_paper"]["fresh_ac_total_overhead_fraction"]) in body
    single = measurements["measurements"]["kda:single"]
    recorded = next(item for item in catalog["lessons"] if item["id"] == "seq-single-self-check")
    for arm in KDA_FRESH_ARMS:
        assert recorded["medians_us"][arm] == single[arm]["median_us"]
    expected_ab = (
        single["seq_three_piece"]["median_us"] - single["single_piece_three_stage"]["median_us"]
    ) / single["single_piece_three_stage"]["median_us"]
    assert recorded["ab_fraction"] == pytest.approx(expected_ab)
    assert _fmt_pct(expected_ab) in body
    assert "这不是决策行" in body


def test_scope_units_and_sources_stay_attached_to_the_numbers():
    """Scope, the Split-K unit and every rendered number's file survive into the section source line."""
    catalog, overlay = build_catalog()
    by_id = {item["id"]: item for item in catalog["lessons"]}
    seq_body = _section(overlay, "因果状态")
    assert "Apple M4 Max" in seq_body
    assert "fp16" in seq_body
    assert "L1" in seq_body
    assert "不含 L2 capacity" in seq_body
    assert "不能外推成 SEQ 便宜" in seq_body
    seq_source = next(line for line in seq_body.splitlines() if line.startswith("来源:"))
    lesson = by_id["seq-serial-handoff"]
    assert lesson["decision_source"] in seq_source
    assert lesson["diff_source"] == "tests/support/seq_handoff.py"
    assert lesson["diff_source"] in seq_source
    assert "docs/架构/seq-state-handoff-headroom.md" in seq_source
    conv_body = _section(overlay, "conv2d paper")
    conv_source = next(line for line in conv_body.splitlines() if line.startswith("来源:"))
    assert by_id["conv2d-paper-scalar-ablation"]["source"] in conv_source
    assert "docs/架构/中间结果复用与Scratch Workspace.md" in conv_source
    bit = by_id["bitdelta-reg"]
    small_ratio = _fmt_ratio(bit["sizes"]["small"]["best_legal_shared_ratio_over_reg"])
    paper_ratio = _fmt_ratio(bit["sizes"]["paper"]["best_legal_shared_ratio_over_reg"])
    bit_body = _section(overlay, "bitdelta")
    assert bit["sizes"]["small"]["exact_ratio_taught"] is False
    assert bit["sizes"]["paper"]["exact_ratio_taught"] is True
    assert small_ratio not in bit_body
    assert small_ratio not in overlay
    assert paper_ratio in bit_body
    assert "大约一个数量级" in bit_body
    assert "不能教精确倍数或排序" in bit_body
    split = by_id["gemm-splitk-stop"]
    analysis = json.loads(
        (ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2-L/evidence/analysis.json").read_text(encoding="utf-8")
    )
    numer = analysis["paper"]["split2_partial_write_read_us_at_peak"]
    denom = analysis["paper"]["existing_candidate_us"]
    assert split["paper_split2_numerator_us"] == numer
    assert split["paper_split2_denominator_us"] == denom
    assert split["paper_split2_traffic_percent"] == pytest.approx(numer / denom * 100.0)
    assert split["paper_split2_is_byte_ratio"] is False
    split_body = _section(overlay, "GEMM Split-K")
    assert "不是字节流量比" in split_body
    assert "split2_partial_write_read_us_at_peak" in split_body
    assert "existing_candidate_us" in split_body
    assert "docs/架构/gemm-splitk-headroom.md" in split_body


def test_boundary_context_is_labelled_and_catalog_only_numbers_stay_out():
    """Tail, single and the worker fix stay visible as non-lessons. Occupied retry and search best do not."""
    catalog, overlay = build_catalog()
    by_id = {item["id"]: item for item in catalog["lessons"]}
    for name in ("seq-tail-decision-measurement", "seq-single-self-check", "p2r-worker-geometry"):
        item = by_id[name]
        assert item["overlay_role"] == "boundary_context"
        assert item["teach"] is False
    assert by_id["seq-tail-decision-measurement"]["numbers_adopted"] is False
    assert by_id["seq-single-self-check"]["numbers_adopted"] is False
    assert "数值不采用，不作为独立经验" in overlay
    assert "evaluation_device_verdict=false" in overlay
    occupied = by_id["gemm-silu-occupied-retry"]
    assert occupied["overlay_role"] == "catalog_only"
    assert f"{occupied['candidate_us']:.2f}" not in overlay
    assert "2.765" not in overlay
    prose = {
        "agents": (ROOT / "AGENTS.md").read_text(encoding="utf-8"),
        "contract": CONTRACT.read_text(encoding="utf-8"),
        "report": (P2S / "S.md").read_text(encoding="utf-8"),
        "module": (ROOT / "experiments/experience_catalog.py").read_text(encoding="utf-8"),
    }
    for label, text in prose.items():
        assert "kda single`/`tail` 和 P2-R 的 harness 修复不进入叠加层" not in text, label
        assert "`kda single` / `tail` 和 P2-R 的 harness 修复不进入叠加层" not in text, label
        assert "也不进入叠加层" not in text, label
    assert "数值不采用，不作为独立经验" in prose["agents"]
    assert "数值不采用，不作为独立经验" in prose["contract"]
    assert "数值不采用，不作为独立经验" in prose["report"]
    assert "overlay as boundary context" in prose["module"]
    assert "Occupied-retry and search-best figures stay out of the overlay." in prose["module"]
    assert "硬上限 $12" not in prose["agents"]
    assert "硬上限 $12" not in prose["contract"]
    assert "硬上限 $12" not in prose["report"]


def test_a_section_missing_its_measurement_fails_the_existing_verified_contract():
    """The counterfactual merge uses the unchanged wiki checks. A section without 本机实测 fails."""
    assert _verified_gaps(
        "Confidence: verified\n来源: `docs/架构/verified-experience-search-prep.md`\n边界: 只覆盖 fp16。\n"
    ) == ["measurement"]
    assert _verified_gaps("Confidence: verified\n本机实测: 1 µs\n边界: 只覆盖 fp16。\n") == ["source"]
    raw = EXPERIENCE.read_text(encoding="utf-8")
    _catalog, overlay = build_catalog()
    merged = filter_experience_for_prompt(raw.rstrip() + "\n\n" + overlay)
    overlay_sections = _verified_sections(overlay)
    assert len(overlay_sections) == 6
    merged_sections = dict(_verified_sections(merged))
    for title, body in overlay_sections:
        assert _verified_gaps(body) == [], title
        assert title in merged_sections
        assert _verified_gaps(merged_sections[title]) == [], title
    for title, body in _verified_sections(filter_experience_for_prompt(raw)):
        assert title in merged_sections
        assert _verified_gaps(merged_sections[title]) == [], title


def test_architecture_contract_matches_the_prompt_boundary():
    """The contract's treatment digest is the archived boundary, not a stale sentence."""
    boundary = _load("prompt-boundary.json")
    text = CONTRACT.read_text(encoding="utf-8")
    assert "5b23ac934eb5d5e1fe9a73902a8ebf31b8873bab18864d8c6d9dde76205293bf" not in text
    assert "20227 字节" not in text
    treatment = re.findall(r"过滤后的处理经验 \| (\d+) 字节，`([0-9a-f]{64})`", text)
    control = re.findall(r"过滤后的对照经验 \| (\d+) 字节，`([0-9a-f]{64})`", text)
    assert treatment == [(str(boundary["treatment_filtered_bytes"]), boundary["treatment_filtered_experience_sha256"])]
    assert control == [(str(boundary["control_filtered_bytes"]), boundary["control_filtered_experience_sha256"])]
    assert "是否启动后续 cell 的承诺上限" in text
    assert "可容忍超支仍未批准" in text
    assert "硬上限" not in text


def test_max_total_is_a_start_cap_not_a_bill_ceiling():
    """$12 decides whether another cell may start. It is not a ceiling the bill cannot pass."""
    plan = _load("dry-run-plan.json")
    meaning = plan["stop"]["hard_cap_usd_meaning"]
    assert "later cell" in meaning
    assert "not a ceiling the bill cannot pass" in meaning
    assert "Tolerable overrun is not approved." in meaning
    assert plan["stop"]["hard_cap_usd"] == plan["max_total_usd"] == 12.0
    assert plan["stop"]["per_cell_soft_budget_usd"] == 3.0
    assert plan["stop"]["in_flight_request_may_exceed_soft_budget"] is True
    assert plan["stop"]["tolerable_overrun_approved"] is None
    assert plan["stop"]["allowed_regression_approved"] is None
