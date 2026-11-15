"""P2-T: treatment-only overlay injection through the real session render.

These tests call ``prepare_session`` and the experience-overlay CLI. They do not
call a model, a GPU, ``taic run``, ``taic resume``, or ``track-explore``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from experiments.experience_catalog import OVERLAY_MARK as CATALOG_MARK
from experiments.common import run_one
from experiments.search_yield_ab import (
    CONTROL_OVERRIDES,
    build_experience_plan,
    build_plan,
    execute_plan,
    main,
    render_experience_plan,
    summarize_experiment,
    write_plan,
)
from taic.experience_overlay import (
    OVERLAY_MARK,
    ExperienceOverlayBinding,
    ExperienceOverlayError,
    binding_for_run,
    install_run_overlay,
    run_identity,
    seedless_kernel_brief,
)
from taic.experience_prompt import filter_experience_for_prompt
from taic.kernel_agent import prepare_session
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator, confirmation_master_seed, eval_seed
from taic.trust import current_prompt_sha1
from tests.support.fakes import FakeClient
from tests.support.orchestrator import make_orch

ROOT = Path(__file__).resolve().parents[2]
P2S = ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2-S"
OVERLAY = P2S / "evidence" / "experience-overlay.md"
BOUNDARY = json.loads((P2S / "evidence" / "prompt-boundary.json").read_text(encoding="utf-8"))
EXPERIENCE = ROOT / "targets/apple-m4-max/experience.md"
GATES = ["--execute", "--allow-spend", "--allow-gpu", "--approved-by", "Ada Lovelace", "--delta", "0.05"]
DRIFT = {
    "experiments/search_yield_ab.py",
    "tests/shared/test_experience_search_prep.py",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runs(tmp_path: Path, name: str) -> Path:
    dest = ROOT / "runs" / f"_p2t_{name}_{hashlib.sha256(str(tmp_path).encode()).hexdigest()[:8]}"
    if dest.exists():
        shutil.rmtree(dest)
    return dest


def _llm() -> LLMConfig:
    return LLMConfig(
        base_url="http://127.0.0.1:9",
        api_key="sk-test-0123456789abcdef",
        model="fake",
        reasoning_effort="off",
    )


def _search() -> SearchConfig:
    return SearchConfig(**CONTROL_OVERRIDES)


def _binding(run_dir: Path) -> ExperienceOverlayBinding:
    return binding_for_run(str(OVERLAY.relative_to(ROOT)), _digest(OVERLAY), run_dir)


def _orch(run_dir: Path, *, resuming: bool = False) -> Orchestrator:
    return Orchestrator(
        "gemm_silu", "fp16", "small", _llm(), _search(),
        run_dir=run_dir, client=FakeClient([]), log=lambda _line: None, seed=1,
        resuming=resuming,
    )


def _install(run_dir: Path) -> dict:
    return install_run_overlay(run_dir, str(OVERLAY.relative_to(ROOT)), _digest(OVERLAY))


def _brief(run_dir: Path, work: Path, overlay):
    from taic.task_binding import bind_task

    guidance = run_dir / "agent_guidance"
    return seedless_kernel_brief(
        bind_task("gemm_silu", "fp16", "small"),
        _search(),
        repo=ROOT,
        run_dir=run_dir,
        work_dir=work,
        guidance_dir=guidance,
        overlay=overlay,
    )


def _normalize(text: str, *paths: Path) -> str:
    # Longer paths first so a work directory under the run is replaced intact.
    ordered = sorted((Path(path).resolve() for path in paths), key=lambda item: len(str(item)), reverse=True)
    for path in ordered:
        text = text.replace(str(path), "{path}")
    return text


def _silence_live_search(monkeypatch) -> None:
    def explode(*_args, **_kwargs):
        raise AssertionError("a live search entry was called")

    monkeypatch.setattr("experiments.common.run_one", explode)
    monkeypatch.setattr("taic.orchestrator.Orchestrator.run", explode)


def _experience_argv(out: Path, *extra: str) -> list[str]:
    return [
        "--out", str(out),
        "--factor", "experience-overlay",
        "--overlay", str(OVERLAY),
        "--tasks", "gemm_silu",
        "--size", "small",
        "--pairs", "2",
        "--budget-usd", "3",
        "--max-total-usd", "12",
        *extra,
    ]


def test_overlay_mark_matches_the_frozen_catalog_sentence():
    assert OVERLAY_MARK == CATALOG_MARK
    assert OVERLAY.read_text(encoding="utf-8").count(OVERLAY_MARK) == 1


def test_default_session_and_prompt_stay_byte_identical(tmp_path, monkeypatch):
    """No binding, no env var, and no run_config key. The fingerprint stays on P2-M."""
    monkeypatch.setenv("TAIC_EXPERIENCE_OVERLAY", str(OVERLAY))
    monkeypatch.setenv("EXPERIENCE_OVERLAY_SHA256", _digest(OVERLAY))
    from tests.support.live_anchors import load_live_anchors
    assert current_prompt_sha1() == load_live_anchors(ROOT)["prompt_sha1"]
    epoch = json.loads((P2S.parent / "P2-M/prompt-epoch.json").read_text(encoding="utf-8"))
    assert BOUNDARY["current_prompt_sha1"] == epoch["current_prompt_sha1"]
    assert _digest(EXPERIENCE) == BOUNDARY["experience_file_sha256"]
    assert OVERLAY_MARK not in EXPERIENCE.read_text(encoding="utf-8")
    run_dir = tmp_path / "default"
    work = run_dir / "agent_work" / "init"
    (run_dir / "agent_guidance").mkdir(parents=True)
    shutil.copyfile(EXPERIENCE, run_dir / "agent_guidance" / "experience.md")
    prepared = prepare_session(_brief(run_dir, work, None), work)
    experience = (work / "experience.md").read_text(encoding="utf-8")
    assert experience == filter_experience_for_prompt(EXPERIENCE.read_text(encoding="utf-8"))
    assert experience.count(OVERLAY_MARK) == 0
    assert hashlib.sha256(experience.encode("utf-8")).hexdigest() == BOUNDARY["control_filtered_experience_sha256"]
    assert not (work / "experience_overlay.json").exists()
    assert "experience_overlay.json" not in prepared.files
    assert (work / "TASK.md").read_text(encoding="utf-8")
    orch = _orch(tmp_path / "orch")
    config = json.loads((orch.run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert "experience_overlay" not in config
    assert OVERLAY_MARK not in (orch.run_dir / "system_prompt.txt").read_text(encoding="utf-8")
    system = (orch.run_dir / "system_prompt.txt").read_bytes()
    assert hashlib.sha256(system).hexdigest() == BOUNDARY["experiment_task_system_prompt"]["tracks"]["msl-single"]["sha256"]
    assert len(system) == BOUNDARY["experiment_task_system_prompt"]["tracks"]["msl-single"]["bytes"]
    legacy = make_orch(tmp_path / "legacy", [], [])
    legacy_config = json.loads((legacy.run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert "experience_overlay" not in legacy_config
    assert os.environ["TAIC_EXPERIENCE_OVERLAY"] == str(OVERLAY)


def test_prepare_session_injects_one_overlay_and_matches_the_orchestrator(tmp_path):
    """Treatment and control differ only in the staged experience file and its sidecar."""
    control_run = tmp_path / "control"
    treatment_run = tmp_path / "treatment"
    control = _orch(control_run)
    treatment = _orch(treatment_run)
    record = _install(treatment_run)
    assert record["arm"] == "treatment"
    assert record["run_id"] == run_identity(treatment_run)
    control_config = json.loads((control_run / "run_config.json").read_text(encoding="utf-8"))
    treatment_config = json.loads((treatment_run / "run_config.json").read_text(encoding="utf-8"))
    assert "experience_overlay" not in control_config
    record = treatment_config["experience_overlay"]
    assert record["overlay_path"] == OVERLAY.relative_to(ROOT).as_posix()
    assert record["overlay_sha256"] == _digest(OVERLAY)
    assert record["rendered_experience_sha256"] == BOUNDARY["treatment_filtered_experience_sha256"]
    assert record["rendered_experience_bytes"] == BOUNDARY["treatment_filtered_bytes"]
    assert record["arm"] == "treatment"
    assert record["run_id"] == run_identity(treatment_run)
    assert record["overlay_copies"] == 1
    assert record["authoritative_task_rules"] is False
    assert record["not_a_task_contract"] is True
    assert OVERLAY_MARK not in json.dumps(record)
    assert (control_run / "system_prompt.txt").read_bytes() == (treatment_run / "system_prompt.txt").read_bytes()
    assert OVERLAY_MARK not in (treatment_run / "system_prompt.txt").read_text(encoding="utf-8")
    assert OVERLAY_MARK not in (treatment_run / "agent_guidance" / "experience.md").read_text(encoding="utf-8")

    def staged(orch, work: Path):
        brief = orch._session_brief(True, "init", work_dir=work)
        other = seedless_kernel_brief(
            orch.binding, orch.search, repo=ROOT, run_dir=orch.run_dir, work_dir=work,
            guidance_dir=orch.run_dir / "agent_guidance", overlay=None,
        )
        left = asdict(brief)
        right = asdict(other)
        left["experience_overlay"] = None if brief.experience_overlay is None else brief.experience_overlay.identity()
        right["experience_overlay"] = None if other.experience_overlay is None else other.experience_overlay.identity()
        assert left == right
        prepared = prepare_session(brief, work)
        return prepared, (work / "experience.md").read_text(encoding="utf-8"), (work / "TASK.md").read_text(encoding="utf-8")

    control_prepared, control_experience, control_task = staged(control, control_run / "agent_work" / "init")
    treatment_prepared, treatment_experience, treatment_task = staged(treatment, treatment_run / "agent_work" / "init")
    assert control_experience.count(OVERLAY_MARK) == 0
    assert treatment_experience.count(OVERLAY_MARK) == 1
    assert hashlib.sha256(control_experience.encode()).hexdigest() == BOUNDARY["control_filtered_experience_sha256"]
    assert len(control_experience.encode()) == BOUNDARY["control_filtered_bytes"]
    assert hashlib.sha256(treatment_experience.encode()).hexdigest() == BOUNDARY["treatment_filtered_experience_sha256"]
    assert len(treatment_experience.encode()) == BOUNDARY["treatment_filtered_bytes"]
    assert "experience_overlay.json" not in control_prepared.files
    sidecar = json.loads((treatment_run / "agent_work" / "init" / "experience_overlay.json").read_text(encoding="utf-8"))
    assert sidecar["rendered_experience_sha256"] == record["rendered_experience_sha256"]
    assert sidecar["overlay_copies"] == 1
    assert OVERLAY_MARK not in json.dumps(sidecar)
    assert "experience_overlay.json" in treatment_prepared.files
    assert _normalize(control_task, control_run, control_run / "agent_work" / "init") == _normalize(
        treatment_task, treatment_run, treatment_run / "agent_work" / "init"
    )
    search_control = {key: control_config["search"][key] for key in CONTROL_OVERRIDES}
    search_treatment = {key: treatment_config["search"][key] for key in CONTROL_OVERRIDES}
    assert search_control == search_treatment == CONTROL_OVERRIDES


def test_bad_overlay_is_refused_before_a_session_file_is_written(tmp_path):
    run_dir = tmp_path / "bad"
    (run_dir / "agent_guidance").mkdir(parents=True)
    shutil.copyfile(EXPERIENCE, run_dir / "agent_guidance" / "experience.md")
    work = run_dir / "work"
    with pytest.raises(ExperienceOverlayError, match="does not match"):
        binding_for_run(str(OVERLAY.relative_to(ROOT)), "0" * 64, run_dir)
    with pytest.raises(ExperienceOverlayError, match="does not exist"):
        binding_for_run("runs/missing-overlay.md", _digest(OVERLAY), run_dir)
    outside = tmp_path / "outside.md"
    outside.write_text(OVERLAY_MARK + "\n", encoding="utf-8")
    with pytest.raises(ExperienceOverlayError, match="escapes"):
        binding_for_run(str(outside), hashlib.sha256(outside.read_bytes()).hexdigest(), run_dir)
    empty = ROOT / "runs" / f"_p2t_empty_{tmp_path.name}.md"
    twicemark = ROOT / "runs" / f"_p2t_twice_{tmp_path.name}.md"
    contract = ROOT / "runs" / f"_p2t_contract_{tmp_path.name}.md"
    try:
        empty.write_text("\n", encoding="utf-8")
        with pytest.raises(ExperienceOverlayError, match="empty"):
            binding_for_run(str(empty.relative_to(ROOT)), hashlib.sha256(empty.read_bytes()).hexdigest(), run_dir)
        twicemark.write_text(OVERLAY_MARK + "\n" + OVERLAY_MARK + "\n", encoding="utf-8")
        with pytest.raises(ExperienceOverlayError, match="exactly once"):
            binding_for_run(str(twicemark.relative_to(ROOT)), hashlib.sha256(twicemark.read_bytes()).hexdigest(), run_dir)
        contract.write_text(OVERLAY_MARK + "\n## 任务合同\n", encoding="utf-8")
        with pytest.raises(ExperienceOverlayError, match="task contract"):
            binding_for_run(str(contract.relative_to(ROOT)), hashlib.sha256(contract.read_bytes()).hexdigest(), run_dir)
    finally:
        for path in (empty, twicemark, contract):
            path.unlink(missing_ok=True)
    good = _binding(run_dir)
    wrong_arm = ExperienceOverlayBinding(good.overlay_path, good.overlay_sha256, "control", good.run_id)
    with pytest.raises(ExperienceOverlayError, match="treatment"):
        prepare_session(_brief(run_dir, work, wrong_arm), work)
    assert not (work / "experience.md").exists()
    mismatched = ExperienceOverlayBinding(good.overlay_path, good.overlay_sha256, "treatment", "runs/other-cell")
    brief = _brief(run_dir, work, mismatched)
    brief.run_id = run_identity(run_dir)
    with pytest.raises(ExperienceOverlayError, match="run_id"):
        prepare_session(brief, work)
    poisoned = run_dir / "agent_guidance" / "experience.md"
    poisoned.write_text(poisoned.read_text(encoding="utf-8") + "\n" + OVERLAY_MARK + "\n", encoding="utf-8")
    with pytest.raises(ExperienceOverlayError, match="already contains"):
        prepare_session(_brief(run_dir, work, good), work)
    other = tmp_path / "other"
    (other / "agent_guidance").mkdir(parents=True)
    shutil.copyfile(EXPERIENCE, other / "agent_guidance" / "experience.md")
    clean = _orch(tmp_path / "clean")
    forged = _install(clean.run_dir)
    (other / "experience_overlay.json").write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ExperienceOverlayError, match="different run"):
        prepare_session(_brief(other, other / "agent_work" / "init", None), other / "agent_work" / "init")


def test_overlay_does_not_leak_across_sessions_or_resume(tmp_path):
    treatment_run = tmp_path / "treatment"
    treatment = _orch(treatment_run)
    installed = _install(treatment_run)
    work = treatment_run / "agent_work" / "init"
    brief = treatment._session_brief(True, "init", work_dir=work)
    prepare_session(brief, work)
    prepare_session(brief, work)
    assert (work / "experience.md").read_text(encoding="utf-8").count(OVERLAY_MARK) == 1
    control_run = tmp_path / "control"
    control = _orch(control_run)
    control_work = control_run / "agent_work" / "init"
    (control_work / "experience_overlay.json").parent.mkdir(parents=True, exist_ok=True)
    (control_work / "experience_overlay.json").write_text('{"stale": true}\n', encoding="utf-8")
    prepare_session(control._session_brief(True, "init", work_dir=control_work), control_work)
    assert (control_work / "experience.md").read_text(encoding="utf-8").count(OVERLAY_MARK) == 0
    assert not (control_work / "experience_overlay.json").exists()
    with pytest.raises(ExperienceOverlayError, match="do not agree"):
        prepare_session(control._session_brief(True, "init", work_dir=work), work)
    resumed = _orch(treatment_run, resuming=True)
    resumed_work = treatment_run / "agent_work" / "resumed"
    prepare_session(resumed._session_brief(True, "init", work_dir=resumed_work), resumed_work)
    resumed_text = (resumed_work / "experience.md").read_text(encoding="utf-8")
    assert resumed_text.count(OVERLAY_MARK) == 1
    assert hashlib.sha256(resumed_text.encode("utf-8")).hexdigest() == installed["rendered_experience_sha256"]
    resumed_control = _orch(control_run, resuming=True)
    resumed_control_work = control_run / "agent_work" / "resumed"
    prepare_session(resumed_control._session_brief(True, "init", work_dir=resumed_control_work), resumed_control_work)
    assert (resumed_control_work / "experience.md").read_text(encoding="utf-8").count(OVERLAY_MARK) == 0
    copied = tmp_path / "copied"
    shutil.copytree(treatment_run / "agent_guidance", copied / "agent_guidance")
    shutil.copyfile(treatment_run / "experience_overlay.json", copied / "experience_overlay.json")
    with pytest.raises(ExperienceOverlayError, match="different run"):
        prepare_session(_brief(copied, copied / "agent_work" / "init", None), copied / "agent_work" / "init")


def test_render_only_and_execute_gates_do_not_start_a_search(tmp_path, monkeypatch):
    _silence_live_search(monkeypatch)
    calls = []

    def runner(*args, **kwargs):
        calls.append((args, kwargs))

    dry = _runs(tmp_path, "dry")
    rendered = _runs(tmp_path, "render")
    executed = _runs(tmp_path, "execute")
    blocked = _runs(tmp_path, "blocked")
    try:
        assert main(_experience_argv(blocked, "--execute"), runner=runner) == 2
        assert calls == []
        assert not blocked.exists()
        assert main(_experience_argv(dry, "--dry-run"), runner=runner) == 0
        plan = json.loads((dry / "plan.json").read_text(encoding="utf-8"))
        assert plan["injection_ready"] is True
        assert plan["executed"] is False
        assert plan["paid_search"] == "未授权/未执行"
        assert plan["delta"] is None and plan["approved_by"] is None
        assert plan["control"] == plan["treatment"] == CONTROL_OVERRIDES
        assert plan["search_eval_seed"] == 7
        assert calls == []
        assert main(_experience_argv(rendered, "--render-only"), runner=runner) == 0
        manifest = json.loads((rendered / "render-only.json").read_text(encoding="utf-8"))
        assert manifest["produces_candidates"] is False
        assert manifest["calls_model"] is False
        assert manifest["is_search_result"] is False
        assert manifest["paid_search"] == "未授权/未执行"
        assert manifest["injection_ready"] is True
        assert [cell["overlay_copies"] for cell in manifest["cells"]] == [0, 1, 1, 0]
        treatment_run = rendered / "render-only" / "treatment" / "gemm_silu_small_rep1"
        assert (treatment_run / "experience_overlay.json").is_file()
        assert not (treatment_run / "run_config.json").exists()
        assert len({cell["normalized_task_sha256"] for cell in manifest["cells"]}) == 1
        assert len({cell["system_prompt_sha256"] for cell in manifest["cells"]}) == 1
        treatment = next(cell for cell in manifest["cells"] if cell["arm"] == "treatment")
        control = next(cell for cell in manifest["cells"] if cell["arm"] == "control")
        assert treatment["experience_sha256"] == BOUNDARY["treatment_filtered_experience_sha256"]
        assert control["experience_sha256"] == BOUNDARY["control_filtered_experience_sha256"]
        assert treatment["experience_overlay"]["authoritative_task_rules"] is False
        assert list(rendered.rglob("result.json")) == []
        assert main(_experience_argv(rendered, "--render-only"), runner=runner) == 2
        assert main(_experience_argv(executed, *GATES), runner=runner) == 0
        assert len(calls) == 4
        assert {kwargs["seed"] for _args, kwargs in calls} == {7}
        treated = [kwargs for _args, kwargs in calls if kwargs.get("experience_overlay_path")]
        untouched = [kwargs for _args, kwargs in calls if "experience_overlay_path" not in kwargs]
        assert len(treated) == 2 and len(untouched) == 2
        assert {kwargs["experience_overlay_sha256"] for kwargs in treated} == {_digest(OVERLAY)}
        assert all("experience_overlay_sha256" not in kwargs for kwargs in untouched)
        assert list(executed.rglob("result.json")) == []
        assert calls and all(kwargs["init_samples"] == 1 and kwargs["sweep_tool"] is False for _args, kwargs in calls)
    finally:
        for path in (dry, rendered, executed, blocked):
            shutil.rmtree(path, ignore_errors=True)


def test_cli_rejects_a_bad_overlay_a_tampered_plan_and_the_wrong_arm(tmp_path, monkeypatch):
    _silence_live_search(monkeypatch)
    calls = []

    def runner(*_args, **_kwargs):
        calls.append(True)

    missing = _runs(tmp_path, "missing")
    empty_out = _runs(tmp_path, "empty")
    escape_out = _runs(tmp_path, "escape")
    tamper = _runs(tmp_path, "tamper")
    wrong = _runs(tmp_path, "wrong")
    empty = ROOT / "runs" / f"_p2t_empty_cli_{tmp_path.name}.md"
    outside = tmp_path / "outside.md"
    try:
        assert main([
            "--out", str(missing), "--dry-run", "--factor", "experience-overlay",
            "--overlay", str(ROOT / "runs" / "missing-overlay.md"),
            "--tasks", "gemm_silu", "--size", "small",
        ], runner=runner) == 2
        assert not missing.exists()
        empty.write_text(" \n", encoding="utf-8")
        outside.write_text(OVERLAY_MARK + "\n", encoding="utf-8")
        assert main([
            "--out", str(empty_out), "--dry-run", "--factor", "experience-overlay",
            "--overlay", str(empty), "--tasks", "gemm_silu", "--size", "small",
        ], runner=runner) == 2
        assert not empty_out.exists()
        assert main([
            "--out", str(escape_out), "--dry-run", "--factor", "experience-overlay",
            "--overlay", str(outside), "--tasks", "gemm_silu", "--size", "small",
        ], runner=runner) == 2
        assert not escape_out.exists()
        assert main(_experience_argv(tamper, "--dry-run"), runner=runner) == 0
        plan = json.loads((tamper / "plan.json").read_text(encoding="utf-8"))
        plan["overlay_sha256"] = "a" * 64
        (tamper / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        assert main(_experience_argv(tamper, *GATES), runner=runner) == 2
        plan = json.loads((tamper / "plan.json").read_text(encoding="utf-8"))
        plan["overlay_sha256"] = _digest(OVERLAY)
        plan["injection_ready"] = False
        (tamper / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        assert main(_experience_argv(tamper, *GATES), runner=runner) == 2
        with pytest.raises(ValueError, match="not injection_ready"):
            render_experience_plan(tamper, plan)
        assert main(_experience_argv(wrong, "--dry-run"), runner=runner) == 0
        plan = json.loads((wrong / "plan.json").read_text(encoding="utf-8"))
        assert plan["cells"][0]["arm"] == "control"
        plan["cells"][0]["experience_overlay_sha256"] = plan["overlay_sha256"]
        (wrong / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        assert main(_experience_argv(wrong, *GATES), runner=runner) == 2
        with pytest.raises(ValueError, match="control cell"):
            render_experience_plan(wrong, plan)
        assert calls == []
    finally:
        for path in (empty_out, escape_out, tamper, wrong):
            shutil.rmtree(path, ignore_errors=True)
        empty.unlink(missing_ok=True)


def test_prepare_session_rejects_a_tampered_rendered_sha_before_writing(tmp_path):
    """A forged rendered digest must not be replaced by a fresh sidecar."""
    run = tmp_path / "treatment"
    orch = _orch(run)
    _install(run)
    original = (run / "experience_overlay.json").read_text(encoding="utf-8")
    saved = json.loads(original)

    def refuse(name: str) -> None:
        work = run / "agent_work" / name
        with pytest.raises(ExperienceOverlayError, match="rendered_experience_sha256"):
            prepare_session(orch._session_brief(True, "init", work_dir=work), work)
        assert not (work / "device.md").exists()
        assert not (work / "experience.md").exists()
        assert not (work / "experience_overlay.json").exists()

    forged = dict(saved)
    forged["rendered_experience_sha256"] = "0" * 64
    (run / "experience_overlay.json").write_text(json.dumps(forged), encoding="utf-8")
    refuse("run-file")
    assert json.loads((run / "experience_overlay.json").read_text(encoding="utf-8"))["rendered_experience_sha256"] == "0" * 64

    (run / "experience_overlay.json").write_text(original, encoding="utf-8")
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    config["experience_overlay"] = dict(saved)
    config["experience_overlay"]["rendered_experience_sha256"] = "0" * 64
    (run / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    refuse("run-config")

    (run / "experience_overlay.json").write_text(json.dumps(forged), encoding="utf-8")
    refuse("both-records")


def _paired_result(eval_value: int, run_seed: int, speed: float) -> dict:
    return {
        "confirmed_status": "ok",
        "confirmed_speedup": speed,
        "confirmation_source_sha256": "ab" * 32,
        "best_speedup": speed,
        "spent_usd": 1.0,
        "run_seed": run_seed,
        "trace": [{
            "label": "init",
            "status": "ok",
            "eval_seed": eval_value,
            "eval_seed_from": "search",
        }],
    }


def test_experience_execute_binds_one_run_seed_without_changing_switch_runs(tmp_path, monkeypatch):
    """Both experience arms receive seed 7. Switch runs still omit it.

    ``run_seed=7`` is the shared run_config value. A v2 search request is the
    orchestrator's literal 7. ``eval_seed(7, ...)`` is a different number, and a
    trace that records only that derived value does not satisfy the summarizer.
    """
    calls = []

    def runner(*_args, **kwargs):
        calls.append(kwargs)

    switches = _runs(tmp_path, "switches")
    bound = _runs(tmp_path, "bound")
    summary = _runs(tmp_path, "summary")
    constructed = _runs(tmp_path, "constructed")
    try:
        plan = build_plan(tasks=("conv2d",), pairs=1, git_head="abc")
        execute_plan(switches, plan, runner=runner)
        assert len(calls) == 2
        assert all("seed" not in kwargs for kwargs in calls)
        calls.clear()

        experience = build_experience_plan(
            overlay_sha256=_digest(OVERLAY),
            overlay_path=OVERLAY.relative_to(ROOT).as_posix(),
            git_head="abc",
            pairs=1,
        )
        drifted = dict(experience)
        drifted["search_eval_seed"] = 8
        with pytest.raises(ValueError, match="search_eval_seed"):
            execute_plan(bound, drifted, runner=runner)
        assert calls == []
        execute_plan(bound, experience, runner=runner)
        assert len(calls) == 2
        assert {kwargs["seed"] for kwargs in calls} == {7}

        derived = eval_seed(7, "search", "init", "kernel void candidate() {}")
        assert derived != 7

        async def _do_not_search(self):
            raise RuntimeError("search must not start")

        def _agent(*_args, **_kwargs):
            raise AssertionError("an agent was requested")

        monkeypatch.setattr(Orchestrator, "run", _do_not_search)
        for arm, overlay in (("control", {}), ("treatment", {
            "experience_overlay_path": OVERLAY.relative_to(ROOT).as_posix(),
            "experience_overlay_sha256": _digest(OVERLAY),
        })):
            with pytest.raises(RuntimeError, match="search must not start"):
                run_one(
                    "gemm_silu", "fp16", "small", constructed / arm,
                    budget=3.0, seed=7, client=FakeClient([]),
                    agent_runner=_agent, log=lambda *_args, **_kwargs: None,
                    **CONTROL_OVERRIDES, **overlay,
                )
        recorded = []
        for arm in ("control", "treatment"):
            document = json.loads((constructed / arm / "run_config.json").read_text(encoding="utf-8"))
            assert document["task_mode"] == "v2"
            assert document["seed"] == document["run_seed"] == 7
            recorded.append(confirmation_master_seed(document["run_seed"]))
        assert recorded[0] == recorded[1] == confirmation_master_seed(7)
        control_doc = json.loads((constructed / "control" / "run_config.json").read_text(encoding="utf-8"))
        treatment_doc = json.loads((constructed / "treatment" / "run_config.json").read_text(encoding="utf-8"))
        assert "experience_overlay" not in control_doc
        assert treatment_doc["experience_overlay"]["overlay_sha256"] == _digest(OVERLAY)

        write_plan(summary, experience)
        cells = {cell["arm"]: cell for cell in experience["cells"]}
        for arm, speed in (("control", 1.0), ("treatment", 1.2)):
            folder = summary / cells[arm]["directory"]
            folder.mkdir(parents=True)
            (folder / "result.json").write_text(
                json.dumps(_paired_result(7, 7, speed)), encoding="utf-8",
            )
        assert summarize_experiment(summary, delta=0.05)["complete_pairs"] == 1
        treatment_path = summary / cells["treatment"]["directory"] / "result.json"
        treatment_path.write_text(json.dumps(_paired_result(7, 9, 1.2)), encoding="utf-8")
        mismatched = summarize_experiment(summary, delta=0.05)
        assert mismatched["complete_pairs"] == 0
        assert "run_seed" in mismatched["excluded_pairs"][0]["reason"]
        treatment_path.write_text(json.dumps(_paired_result(derived, 7, 1.2)), encoding="utf-8")
        derived_trace = summarize_experiment(summary, delta=0.05)
        assert derived_trace["complete_pairs"] == 0
        assert "evaluation seed" in derived_trace["excluded_pairs"][0]["reason"]

        plain = _runs(tmp_path, "plain-summary")
        try:
            switches_plan = build_plan(tasks=("conv2d",), pairs=1, git_head="abc")
            write_plan(plain, switches_plan)
            for cell, speed in zip(switches_plan["cells"], (1.0, 1.4), strict=True):
                folder = plain / cell["directory"]
                folder.mkdir(parents=True)
                body = _paired_result(7, 7, speed)
                del body["run_seed"]
                (folder / "result.json").write_text(json.dumps(body), encoding="utf-8")
            assert summarize_experiment(plain, delta=0.05)["complete_pairs"] == 1
        finally:
            shutil.rmtree(plain, ignore_errors=True)
    finally:
        for path in (switches, bound, summary, constructed):
            shutil.rmtree(path, ignore_errors=True)


def test_prepare_session_rejects_a_run_config_that_drops_the_overlay_key(tmp_path):
    """A run file without the matching run_config key must not stage guidance."""
    run = tmp_path / "treatment"
    orch = _orch(run)
    _install(run)
    config_path = run / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert "experience_overlay" in config
    del config["experience_overlay"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    work = run / "agent_work" / "dropped"
    with pytest.raises(ExperienceOverlayError, match="omits experience_overlay"):
        prepare_session(orch._session_brief(True, "init", work_dir=work), work)
    assert not (work / "device.md").exists()
    assert not (work / "experience.md").exists()
    assert not (work / "experience_overlay.json").exists()
    assert "experience_overlay" not in json.loads(config_path.read_text(encoding="utf-8"))
    assert (run / "experience_overlay.json").is_file()


def test_experience_plan_rejects_a_drifted_pair_before_calling_the_runner(tmp_path):
    """A cell that leaves the frozen matrix never reaches the runner."""
    calls = []

    def runner(*_args, **_kwargs):
        calls.append(True)

    plan = build_experience_plan(
        overlay_sha256=_digest(OVERLAY),
        overlay_path=OVERLAY.relative_to(ROOT).as_posix(),
        git_head="abc",
        pairs=1,
    )
    wider = build_experience_plan(
        overlay_sha256=_digest(OVERLAY),
        overlay_path=OVERLAY.relative_to(ROOT).as_posix(),
        git_head="abc",
        pairs=2,
    )
    out = _runs(tmp_path, "matrix")

    def refuse(forged: dict, match: str) -> None:
        calls.clear()
        with pytest.raises(ValueError, match=match):
            execute_plan(out, forged, runner=runner)
        assert calls == []

    def copy_of(source: dict) -> dict:
        return json.loads(json.dumps(source))

    task = copy_of(plan)
    task["cells"][1]["task"] = "conv2d"
    refuse(task, r"cell 1 task")
    dtype = copy_of(plan)
    dtype["cells"][1]["dtype"] = "fp32"
    refuse(dtype, r"cell 1 dtype")
    size = copy_of(plan)
    size["cells"][1]["size"] = "paper"
    refuse(size, r"cell 1 size")
    rep = copy_of(plan)
    rep["cells"][1]["rep"] = 2
    refuse(rep, r"cell 1 rep")
    order = copy_of(wider)
    order["cells"][2]["order"] = "AB"
    refuse(order, r"cell 2 order")
    dropped = copy_of(plan)
    dropped["cells"] = dropped["cells"][:1]
    refuse(dropped, "one control and one treatment")
    committees = copy_of(plan)
    committees["committees"][0]["model"] = "other-model"
    refuse(committees, "committees")
    execute_plan(out, plan, runner=runner)
    assert calls == [True, True]


def test_p2s_evidence_stays_byte_identical_except_the_two_injection_files():
    entries = {}
    for line in (P2S / "manifest.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    mismatched = sorted(name for name, digest in entries.items() if _digest(ROOT / name) != digest)
    assert mismatched == sorted(DRIFT)
    archived = json.loads((P2S / "evidence" / "dry-run-plan.json").read_text(encoding="utf-8"))
    assert "injection_ready" not in archived
    assert archived["paid_search"] == "未授权/未执行"
