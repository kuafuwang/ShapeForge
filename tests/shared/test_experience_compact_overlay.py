"""P2-V: task-scoped compact overlay, unpaid render, no new search.

These tests call ``prepare_session`` and the experience-overlay CLI. They do not
call a model, a GPU, ``taic run``, ``taic resume``, or ``track-explore``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from experiments.experience_compact import (
    COMPACT_REL,
    CONTROL_BYTES,
    CONTROL_SHA,
    FULL_OVERLAY_SHA,
    FULL_REL,
    FULL_TREATMENT_BYTES,
    FULL_TREATMENT_SHA,
    FLOOR_US,
    REG_US,
    SCOPE,
    CompactProjectionError,
    approval_document,
    assert_compact_execution_allowed,
    build_compact_plan,
    execute_compact_plan,
    main as compact_main,
    measure_render,
    parse_compact_scope,
    project_compact_overlay,
    projection_document,
    p2u_boundary,
    read_compact_overlay,
)
from experiments.search_yield_ab import (
    build_experience_plan,
    execute_plan,
    main,
    render_experience_plan,
)
from taic.experience_overlay import (
    OVERLAY_MARK,
    ExperienceOverlayBinding,
    ExperienceOverlayError,
    install_run_overlay,
    run_identity,
    seedless_kernel_brief,
)
from taic.experience_prompt import filter_experience_for_prompt
from taic.kernel_agent import prepare_session
from taic.llm import SearchConfig
from taic.task_binding import bind_task
from taic.trust import current_prompt_sha1

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收"
COMPACT = ROOT / COMPACT_REL
FULL = ROOT / FULL_REL
EXPERIENCE = ROOT / "targets/apple-m4-max/experience.md"
P2U_CONCLUSION = ROOT / "tests/support/golden/shared/acceptance-epochs/fixtures/p2u/conclusion.json"
P2M_EPOCH = ROOT / "tests/support/golden/shared/acceptance-epochs/fixtures/p2m/prompt-epoch.json"
DECISION = ROOT / "tests/support/golden/shared/acceptance-epochs/fixtures/p2o/evidence/decision.json"
PROBE = ROOT / "tests/support/golden/shared/acceptance-epochs/fixtures/p2o/evidence/scratch-probe.json"
SWITCHES = {"init_samples": 1, "task_ledger": False, "plateau_after": 0, "sweep_tool": False}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runs(tmp_path: Path, name: str) -> Path:
    dest = ROOT / "runs" / f"_p2v_{name}_{hashlib.sha256(str(tmp_path).encode()).hexdigest()[:8]}"
    if dest.exists():
        shutil.rmtree(dest)
    return dest


def _search() -> SearchConfig:
    return SearchConfig(**SWITCHES)


def _stage(run_dir: Path) -> None:
    guidance = run_dir / "agent_guidance"
    guidance.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(EXPERIENCE, guidance / "experience.md")


def _brief(run_dir: Path, work: Path, overlay):
    return seedless_kernel_brief(
        bind_task("gemm_silu", "fp16", "small"),
        _search(),
        repo=ROOT,
        run_dir=run_dir,
        work_dir=work,
        guidance_dir=run_dir / "agent_guidance",
        overlay=overlay,
    )


def _retarget(plan: dict, *, task: str, dtype: str, size: str) -> dict:
    cloned = json.loads(json.dumps(plan))
    cloned["tasks"] = [task]
    cloned["dtype"] = dtype
    cloned["size"] = size
    for cell in cloned["cells"]:
        cell["task"] = task
        cell["dtype"] = dtype
        cell["size"] = size
        cell["directory"] = f"{cell['arm']}/{task}_{size}_rep{cell['rep']}"
    return cloned


def test_projection_is_exact_and_rejects_other_scopes():
    assert project_compact_overlay() == read_compact_overlay()
    for kwargs in (
        {"task": "conv2d"},
        {"task": "bitdelta_gemm"},
        {"task": "kda"},
        {"task": "varlen_kda"},
        {"task": "gemm"},
        {"task": "GEMM_SILU"},
        {"size": "paper"},
        {"dtype": "fp32"},
        {"dtype": "bf16"},
        {"target_id": "qingwei"},
        {"target_id": "apple-m4"},
    ):
        fields = {**SCOPE, **kwargs}
        with pytest.raises(CompactProjectionError, match="no compact projection"):
            project_compact_overlay(**fields)
    full = FULL.read_text(encoding="utf-8")
    assert parse_compact_scope(full) is None
    assert parse_compact_scope(COMPACT.read_text(encoding="utf-8"))["size"] == "small"


def test_compact_text_keeps_only_the_gemm_silu_small_lesson():
    text = COMPACT.read_text(encoding="utf-8")
    encoded = text.encode("utf-8")
    assert 400 <= len(encoded) <= 1100
    assert text.count(OVERLAY_MARK) == 1
    assert "overlay_kind: compact" in text
    assert "## 任务合同" not in text and "这是权威任务规则" not in text
    for phrase in (
        "一次性 SiLU 中间结果留在寄存器",
        "不省 FLOPs",
        "fp32 写读",
        "barrier",
        "producer/consumer",
        "dispatch",
        REG_US,
        FLOOR_US,
        "not_a_candidate=true",
        "不是发布 speedup",
        "不指定候选 tiling",
        "fresh identity",
        "publication_allowed=false",
    ):
        assert phrase in text
    lowered = text.casefold()
    for banned in ("conv2d", "bitdelta", "kda", "varlen", "split-k", "splitk", "paper", "1160", "1.6721", "1.0071", "simdgroup", "steel", "bm=", "bk=", "grid="):
        assert banned not in lowered
    document = projection_document()
    assert document["lesson_id"] == "gemm-silu-reg"
    assert document["selector"] == "exact task dtype size target_id"
    assert document["omitted_size"] == "paper"
    assert "conv2d-paper-shared-u" in document["omitted_lesson_ids"]
    assert "bitdelta-reg" in document["omitted_lesson_ids"]
    assert "seq-serial-handoff" in document["omitted_lesson_ids"]
    assert "gemm-splitk-stop" in document["omitted_lesson_ids"]
    assert document["candidate_tiling_specified"] is False
    assert document["speedup_published"] is False
    assert document["overlay_bytes"] == len(encoded)
    assert document["overlay_sha256"] == hashlib.sha256(encoded).hexdigest()


def test_numbers_match_the_p2o_fresh_subject_and_probe():
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    small = decision["sizes"]["small"]
    arm = probe["sizes"]["small"]["arms"]["two_stage_floor"]
    document = projection_document()
    assert document["reg_fresh_us"] == pytest.approx(small["reg_candidate_us"])
    assert document["two_stage_floor_us"] == pytest.approx(arm["wall_us_median"])
    assert document["two_stage_floor_us"] == pytest.approx(small["two_stage_floor_us"])
    assert document["reg_fresh_us_text"] == REG_US == "1.8252"
    assert document["two_stage_floor_us_text"] == FLOOR_US == "8.7713"
    assert document["confirmation_seed"] == small["reg_confirmation_seed"] == 3775173618909035
    assert document["identity"] == "match" and document["search_seed"] == 7
    assert small["global_would_win"] is False
    assert "global_saves_no_flops" in small["stop_reasons"]
    assert decision["implementation"]["global_candidate_source_added"] is False
    assert probe["not_a_candidate"] is True
    assert arm["fill_bytes_per_iteration"] == 0
    assert arm["stages"] == 2
    assert arm["stage_kernels"] == ["probe_touch_producer", "probe_touch_consumer"]
    assert document["evidence_class_floor"] == "probe_floor"


def test_prepare_session_renders_one_compact_copy_and_a_clean_control(tmp_path):
    sizes = measure_render()
    treatment = tmp_path / "treatment"
    control = tmp_path / "control"
    _stage(treatment)
    _stage(control)
    record = install_run_overlay(treatment, COMPACT_REL, _digest(COMPACT))
    treatment_work = treatment / "agent_work" / "init"
    control_work = control / "agent_work" / "init"
    prepare_session(_brief(treatment, treatment_work, None), treatment_work)
    prepare_session(_brief(control, control_work, None), control_work)
    treated = (treatment_work / "experience.md").read_text(encoding="utf-8")
    untouched = (control_work / "experience.md").read_text(encoding="utf-8")
    assert treated.count(OVERLAY_MARK) == 1
    assert treated.count("overlay_kind: compact") == 1
    assert untouched.count(OVERLAY_MARK) == 0
    assert "overlay_kind: compact" not in untouched
    assert len(treated.encode()) == sizes["compact_treatment_bytes"] == 14389
    assert hashlib.sha256(treated.encode()).hexdigest() == sizes["compact_treatment_sha256"]
    assert len(untouched.encode()) == CONTROL_BYTES
    assert hashlib.sha256(untouched.encode()).hexdigest() == CONTROL_SHA
    sidecar = json.loads((treatment_work / "experience_overlay.json").read_text(encoding="utf-8"))
    assert sidecar["overlay_path"] == COMPACT_REL
    assert sidecar["overlay_sha256"] == record["overlay_sha256"] == _digest(COMPACT)
    assert sidecar["rendered_experience_sha256"] == hashlib.sha256(treated.encode()).hexdigest()
    assert sidecar["rendered_experience_bytes"] == len(treated.encode())
    assert sidecar["arm"] == "treatment" and sidecar["overlay_copies"] == 1
    assert sidecar["authoritative_task_rules"] is False and sidecar["not_a_task_contract"] is True
    assert sidecar["installed_in_global_experience"] is False
    assert not (control_work / "experience_overlay.json").exists()
    assert OVERLAY_MARK not in json.dumps(sidecar)
    def normalize(text: str, *paths: Path) -> str:
        ordered = sorted((path.resolve() for path in paths), key=lambda item: len(str(item)), reverse=True)
        for path in ordered:
            text = text.replace(str(path), "{path}")
        return text

    treated_task = normalize((treatment_work / "TASK.md").read_text(encoding="utf-8"), treatment, treatment_work)
    control_task = normalize((control_work / "TASK.md").read_text(encoding="utf-8"), control, control_work)
    assert treated_task == control_task
    assert "eval-v2" in treated_task
    assert "--validate" not in treated_task


def test_records_distinguish_compact_from_the_full_overlay(tmp_path):
    compact_run = tmp_path / "compact"
    full_run = tmp_path / "full"
    _stage(compact_run)
    _stage(full_run)
    install_run_overlay(compact_run, COMPACT_REL, _digest(COMPACT))
    install_run_overlay(full_run, FULL_REL, _digest(FULL))
    compact_work = compact_run / "agent_work" / "init"
    full_work = full_run / "agent_work" / "init"
    prepare_session(_brief(compact_run, compact_work, None), compact_work)
    prepare_session(_brief(full_run, full_work, None), full_work)
    compact_record = json.loads((compact_work / "experience_overlay.json").read_text(encoding="utf-8"))
    full_record = json.loads((full_work / "experience_overlay.json").read_text(encoding="utf-8"))
    assert compact_record["overlay_path"] != full_record["overlay_path"]
    assert compact_record["overlay_sha256"] != full_record["overlay_sha256"]
    assert compact_record["rendered_experience_sha256"] != full_record["rendered_experience_sha256"]
    assert full_record["rendered_experience_sha256"] == FULL_TREATMENT_SHA
    assert full_record["rendered_experience_bytes"] == FULL_TREATMENT_BYTES
    compact_text = (compact_work / "experience.md").read_text(encoding="utf-8")
    full_text = (full_work / "experience.md").read_text(encoding="utf-8")
    assert "overlay_kind: compact" in compact_text
    assert "overlay_kind: compact" not in full_text
    assert compact_text.count(OVERLAY_MARK) == full_text.count(OVERLAY_MARK) == 1
    assert len(compact_text.encode()) < len(full_text.encode())
    removed = (FULL_TREATMENT_BYTES - CONTROL_BYTES) - (len(compact_text.encode()) - CONTROL_BYTES)
    assert removed == 11080
    assert removed / (FULL_TREATMENT_BYTES - CONTROL_BYTES) > 0.9


def test_tamper_wrong_arm_and_cross_run_are_refused(tmp_path):
    run = tmp_path / "treatment"
    _stage(run)
    install_run_overlay(run, COMPACT_REL, _digest(COMPACT))
    original = (run / "experience_overlay.json").read_text(encoding="utf-8")
    forged = json.loads(original)
    forged["rendered_experience_sha256"] = "0" * 64
    (run / "experience_overlay.json").write_text(json.dumps(forged), encoding="utf-8")
    tampered = run / "agent_work" / "tamper"
    with pytest.raises(ExperienceOverlayError, match="rendered_experience_sha256"):
        prepare_session(_brief(run, tampered, None), tampered)
    assert not (tampered / "experience.md").exists()
    assert not (tampered / "device.md").exists()

    clean = tmp_path / "arm"
    _stage(clean)
    work = clean / "agent_work" / "init"
    wrong_arm = ExperienceOverlayBinding(COMPACT_REL, _digest(COMPACT), "control", run_identity(clean))
    with pytest.raises(ExperienceOverlayError, match="treatment"):
        prepare_session(_brief(clean, work, wrong_arm), work)
    assert not (work / "experience.md").exists()
    (run / "experience_overlay.json").write_text(original, encoding="utf-8")

    other = tmp_path / "other"
    _stage(other)
    shutil.copyfile(run / "experience_overlay.json", other / "experience_overlay.json")
    leaked = other / "agent_work" / "init"
    with pytest.raises(ExperienceOverlayError, match="different run"):
        prepare_session(_brief(other, leaked, None), leaked)
    assert not (leaked / "experience.md").exists()

    plan = build_compact_plan(git_head="abc")["experience_plan"]
    out = _runs(tmp_path, "wrong-task")
    try:
        with pytest.raises(CompactProjectionError, match="does not match"):
            render_experience_plan(out, _retarget(plan, task="conv2d", dtype="fp16", size="small"))
        with pytest.raises(CompactProjectionError, match="does not match"):
            render_experience_plan(out, _retarget(plan, task="gemm_silu", dtype="fp16", size="paper"))
        assert not out.exists()
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_default_search_and_prompt_stay_unchanged(tmp_path):
    epoch = json.loads(P2M_EPOCH.read_text(encoding="utf-8"))
    from tests.support.live_anchors import load_live_anchors
    assert current_prompt_sha1() == load_live_anchors(ROOT)["prompt_sha1"]
    assert epoch["current_prompt_sha1"] == "e9d28ff61819a041610d763f5c1464ea860265be2d414f78c15a1f8aea1fdbcf"
    raw = EXPERIENCE.read_text(encoding="utf-8")
    assert OVERLAY_MARK not in raw and "overlay_kind: compact" not in raw
    filtered = filter_experience_for_prompt(raw)
    assert len(filtered.encode()) == CONTROL_BYTES
    assert hashlib.sha256(filtered.encode()).hexdigest() == CONTROL_SHA
    run = tmp_path / "default"
    _stage(run)
    work = run / "agent_work" / "init"
    prepare_session(_brief(run, work, None), work)
    assert (work / "experience.md").read_text(encoding="utf-8") == filtered
    assert not (work / "experience_overlay.json").exists()


def test_dry_run_is_a_new_unapproved_experiment():
    prior = p2u_boundary()
    approval = approval_document()
    plan = build_compact_plan()
    assert prior["verdict"] == "no_gain" and prior["adopt_overlay"] is False and prior["reused"] is False
    assert prior["approved_by"] == "kuafu" and prior["delta"] == 0.01
    assert plan["new_experiment"] is True and plan["not_plan_4"] is True and plan["plan_4_started"] is False
    assert plan["inherits_p2u_approval"] is False
    assert plan["experiment_id"] == "p2v-compact-experience-overlay"
    assert plan["p2u_experiment_id"] == "p2u-experience-overlay-paid-20261114"
    assert plan["p2u_verdict"] == "no_gain" and plan["p2u_adopt_overlay"] is False
    assert plan["adopt_compact_overlay"] is False
    assert plan["approved_by"] is None and plan["delta"] is None and plan["tolerable_overrun"] is None
    assert plan["executed"] is False and plan["execution_record"] is None
    assert plan["paid_search"] == "未授权/未执行"
    assert plan["publication_allowed"] is False and plan["npu_simulated_only"] is True
    assert plan["overlay_kind"] == "compact"
    assert plan["compact_overlay_sha256"] != plan["full_overlay_sha256"] == FULL_OVERLAY_SHA
    inner = plan["experience_plan"]
    assert inner["approved_by"] is None and inner["delta"] is None and inner["executed"] is False
    assert inner["tasks"] == ["gemm_silu"] and inner["dtype"] == "fp16" and inner["size"] == "small"
    assert [cell["experience_overlay_sha256"] for cell in inner["cells"]] == [
        None,
        plan["compact_overlay_sha256"],
        plan["compact_overlay_sha256"],
        None,
    ]
    assert inner["control"] == inner["treatment"] == SWITCHES
    for key in ("approved_by", "delta", "tolerable_overrun", "execution_record"):
        assert approval[key] is None
    assert approval["executed"] is False and approval["inherits_p2u_approval"] is False
    conclusion = json.loads(P2U_CONCLUSION.read_text(encoding="utf-8"))
    assert conclusion["verdict"] == "no_gain" and conclusion["adopt_overlay"] is False


def test_p2u_approval_does_not_execute(tmp_path):
    calls = []

    def runner(*_args, **_kwargs):
        calls.append(True)

    out = _runs(tmp_path, "compact-execute")
    try:
        with pytest.raises(CompactProjectionError, match="does not authorize"):
            execute_compact_plan(
                runner=runner,
                approved_by="kuafu",
                delta=0.01,
                tolerable_overrun="raise the cap to 15",
                allow_spend=True,
                allow_gpu=True,
            )
        assert calls == []
        assert compact_main([
            "--out", str(out),
            "--execute",
            "--allow-spend",
            "--allow-gpu",
            "--approved-by", "kuafu",
            "--delta", "0.01",
            "--tolerable-overrun", "raise the cap to 15",
        ]) == 2
        assert not out.exists()
        assert calls == []
        with pytest.raises(CompactProjectionError, match="does not authorize"):
            execute_plan(out, build_compact_plan(git_head="abc")["experience_plan"], runner=runner)
        assert calls == []
        assert not out.exists()
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_shared_cli_rejects_the_wrong_task_and_compact_execute(tmp_path, monkeypatch):
    calls = []

    def runner(*_args, **kwargs):
        calls.append(kwargs)

    def explode(*_args, **_kwargs):
        raise AssertionError("run_one was called")

    monkeypatch.setattr("experiments.common.run_one", explode)
    wrong = _runs(tmp_path, "cli-wrong")
    paper = _runs(tmp_path, "cli-paper")
    executed = _runs(tmp_path, "cli-execute")
    rendered = _runs(tmp_path, "cli-render")
    try:
        assert main([
            "--out", str(wrong), "--dry-run", "--factor", "experience-overlay",
            "--overlay", str(COMPACT), "--tasks", "conv2d", "--size", "small",
        ], runner=runner) == 2
        assert not wrong.exists()
        assert main([
            "--out", str(paper), "--render-only", "--factor", "experience-overlay",
            "--overlay", str(COMPACT), "--tasks", "gemm_silu", "--dtype", "fp16", "--size", "paper",
        ], runner=runner) == 2
        assert not paper.exists()
        assert main([
            "--out", str(executed),
            "--execute", "--allow-spend", "--allow-gpu",
            "--approved-by", "kuafu", "--delta", "0.01",
            "--factor", "experience-overlay",
            "--overlay", str(COMPACT),
            "--tasks", "gemm_silu", "--dtype", "fp16", "--size", "small",
            "--pairs", "2", "--budget-usd", "3", "--max-total-usd", "12",
        ], runner=runner) == 2
        assert calls == []
        assert not executed.exists()
        assert compact_main(["--out", str(rendered), "--render-only"]) == 0
        manifest = json.loads((rendered / "render-only.json").read_text(encoding="utf-8"))
        assert manifest["produces_candidates"] is False
        assert manifest["calls_model"] is False
        assert manifest["is_search_result"] is False
        assert manifest["paid_search"] == "未授权/未执行"
        assert [cell["overlay_copies"] for cell in manifest["cells"]] == [0, 1, 1, 0]
        treatment = next(cell for cell in manifest["cells"] if cell["arm"] == "treatment")
        control = next(cell for cell in manifest["cells"] if cell["arm"] == "control")
        assert control["experience_bytes"] == CONTROL_BYTES
        assert control["experience_sha256"] == CONTROL_SHA
        assert treatment["experience_bytes"] == 14389
        assert treatment["experience_overlay"]["overlay_path"] == COMPACT_REL
        assert treatment["experience_overlay"]["overlay_sha256"] == _digest(COMPACT)
        session = rendered / "render-only" / "treatment" / "gemm_silu_small_rep1" / "agent_work" / "init" / "experience.md"
        assert session.read_text(encoding="utf-8").count("overlay_kind: compact") == 1
        control_session = rendered / "render-only" / "control" / "gemm_silu_small_rep1" / "agent_work" / "init" / "experience.md"
        assert "overlay_kind: compact" not in control_session.read_text(encoding="utf-8")
        assert list(rendered.rglob("result.json")) == []
        assert calls == []
    finally:
        for path in (wrong, paper, executed, rendered):
            shutil.rmtree(path, ignore_errors=True)


def test_full_overlay_gate_and_p2u_files_stay_put():
    digest = _digest(FULL)
    assert digest == FULL_OVERLAY_SHA
    plan = build_experience_plan(
        overlay_sha256=digest,
        overlay_path=FULL_REL,
        git_head="abc",
        pairs=1,
    )
    assert_compact_execution_allowed(plan)
    conclusion = json.loads(P2U_CONCLUSION.read_text(encoding="utf-8"))
    assert conclusion["verdict"] == "no_gain"
    assert conclusion["adopt_overlay"] is False
    assert conclusion["publication_allowed"] is False
    sizes = measure_render()
    assert sizes["full_treatment_bytes"] == FULL_TREATMENT_BYTES
    assert sizes["full_treatment_sha256"] == FULL_TREATMENT_SHA
    assert sizes["full_added_bytes"] == 12054
    assert sizes["compact_added_bytes"] == 974
    assert sizes["added_bytes_removed"] == 11080
