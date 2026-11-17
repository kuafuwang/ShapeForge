"""Chip-level experience drafts. No GPU and no model calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taic.cli import main
from taic.experience_draft import DraftError, render_draft, set_verified, valid_host_pointer, validate_section
from taic.experience_prompt import filter_experience_for_prompt
from taic.harness import ProposalError, candidate_patch


_KW = dict(
    target_id="apple-m4-max",
    task="gemm_silu",
    title="寄存器里做完 SiLU",
    source="runs/example/report.json",
    measurement="fp16 small 1.8252 us",
    boundary="只覆盖 gemm_silu fp16 small apple-m4-max",
    counterexample="GLOBAL scratch 不省 FLOPs",
    observation="一次性 SiLU 中间结果留在寄存器。",
)


def test_reported_draft_has_the_practice_fields_and_stays_out_of_the_prompt():
    """@brief 草稿带题目、来源、实测、边界、反例，且 reported 进不了提示面。"""
    section = render_draft(**_KW)
    assert "Confidence: reported" in section
    for label in ("题目: gemm_silu", "来源:", "本机实测:", "边界:", "反例:", "落点: targets/apple-m4-max/experience.md"):
        assert label in section
    assert "backends/" not in section
    filtered = filter_experience_for_prompt("# 经验\n\n" + section)
    assert "寄存器里做完 SiLU" not in filtered
    assert "1.8252" not in filtered
    checked = validate_section(section)
    assert checked["confidence"] == "reported" and checked["prompt_eligible"] is False


def test_simulated_timing_and_theory_do_not_render():
    """@brief 模拟态计时和理论解释都不生成草稿。"""
    with pytest.raises(DraftError, match="simulated"):
        render_draft(**{**_KW, "simulated": True})
    with pytest.raises(DraftError, match="theory"):
        render_draft(**{**_KW, "theory": "所有经验注入都无效"})
    with pytest.raises(DraftError, match="chip"):
        render_draft(**{**_KW, "layer": "backend"})
    with pytest.raises(DraftError, match="boundary"):
        render_draft(**{**_KW, "boundary": ""})


def test_verified_requires_a_host_rerun_pointer_and_does_not_write_the_live_file(tmp_path: Path):
    """@brief 改成 verified 必须补宿主重跑指针，而且不改仓库里的经验文件。"""
    section = render_draft(**_KW)
    with pytest.raises(DraftError, match="host rerun"):
        set_verified(section, "")
    with pytest.raises(DraftError, match="host rerun"):
        set_verified(section, "1.8252 us")
    verified = set_verified(section, "runs/example/host_paired_comparisons.jsonl")
    assert "Confidence: verified" in verified
    assert "宿主重跑: runs/example/host_paired_comparisons.jsonl" in verified
    checked = validate_section(verified)
    assert checked["prompt_eligible"] is True
    bare = verified.replace("宿主重跑: runs/example/host_paired_comparisons.jsonl\n", "")
    with pytest.raises(DraftError, match="host rerun"):
        validate_section(bare)
    live = Path("targets/apple-m4-max/experience.md").read_text(encoding="utf-8")
    assert "寄存器里做完 SiLU" not in live


def test_changing_only_confidence_stays_out_of_the_prompt():
    """@brief 只改 Confidence、或再删掉题目/落点任一行，过滤器仍丢掉没有宿主重跑的新草稿。"""
    section = render_draft(**_KW)
    leaked = section.replace("Confidence: reported", "Confidence: verified")
    legacy = "## 旧条目\n\nConfidence: verified\n观察: 旧小节仍进入提示。\n"
    filtered = filter_experience_for_prompt("# 经验\n\n" + legacy + "\n" + leaked)
    assert "旧小节仍进入提示" in filtered
    assert "寄存器里做完 SiLU" not in filtered
    assert "1.8252" not in filtered
    assert "本机实测" not in filtered
    for label in ("题目:", "落点:"):
        stripped = "".join(line for line in leaked.splitlines(keepends=True) if not line.startswith(label))
        assert label not in stripped
        one_missing = filter_experience_for_prompt("# 经验\n\n" + stripped)
        assert "寄存器里做完 SiLU" not in one_missing
        assert "1.8252" not in one_missing
        assert "本机实测" not in one_missing
        assert "宿主重跑" not in stripped
    both = "".join(
        line for line in leaked.splitlines(keepends=True)
        if not line.startswith(("题目:", "落点:"))
    )
    assert "芯片:" in both and "未验证:" in both
    neither_label = filter_experience_for_prompt("# 经验\n\n" + both)
    assert "1.8252" not in neither_label
    assert "本机实测" not in neither_label
    with pytest.raises(DraftError, match="host rerun"):
        validate_section(leaked)
    verified = set_verified(section, "runs/example/host_paired_comparisons.jsonl")
    kept = filter_experience_for_prompt("# 经验\n\n" + verified)
    assert "寄存器里做完 SiLU" in kept
    assert "宿主重跑: runs/example/host_paired_comparisons.jsonl" in kept


def test_bare_bullets_are_not_generated(tmp_path: Path):
    """@brief 缺实践字段时不再追加裸 bullet。"""
    with pytest.raises(ProposalError, match="bare experience bullets"):
        candidate_patch(
            {"surface": "tip", "action": "先读 device.md 的线程组上限再选 tile。", "scope": {"task": "conv2d"}},
            target_id="apple-m4-max", proposal_key="k", root=tmp_path,
        )


def test_write_patch_refuses_a_simulated_outer_row(tmp_path: Path):
    """@brief 模拟标志在提案外层时，公开的 write_patch 也不生成草稿。"""
    from taic.harness import ProposalError, write_patch

    run = tmp_path / "run"
    run.mkdir()
    repo = tmp_path / "repo"
    experience = repo / "targets" / "apple-m4-max" / "experience.md"
    experience.parent.mkdir(parents=True)
    experience.write_text("# Experience\n", encoding="utf-8")
    row = {
        "status": "valid",
        "proposal_key": "simkey",
        "simulated": True,
        "measurement_kind": "simulated",
        "proposal": {
            "surface": "tip",
            "action": "Keep the one-shot SiLU intermediate in registers for this fp16 kernel.",
            "audience": "peer",
            "needs": "none",
            "scope": {"target_id": "apple-m4-max", "task": "gemm_silu"},
            "practice": {
                "task": "gemm_silu",
                "title": "寄存器里做完 SiLU",
                "source": "runs/example/report.json",
                "measurement": "simulated 1.2 us",
                "boundary": "只覆盖 fp16 small",
                "counterexample": "GLOBAL scratch 不省 FLOPs",
            },
        },
    }
    (run / "harness_proposals.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ProposalError, match="simulated"):
        write_patch(run, target_id="apple-m4-max", proposal_key="simkey", root=repo)
    assert not (run / "harness_promotions" / "simkey.patch").exists()
    assert "simulated" not in experience.read_text(encoding="utf-8")
    assert "Confidence: reported" not in experience.read_text(encoding="utf-8")


def test_cli_writes_only_the_requested_draft(tmp_path: Path, capsys):
    """@brief 命令把草稿写到 --out，模拟态拒绝，verified 要指针。"""
    out = tmp_path / "draft.md"
    code = main([
        "harness", "draft-experience",
        "--target", "apple-m4-max", "--task", "gemm_silu", "--title", "寄存器里做完 SiLU",
        "--source", "runs/example/report.json", "--measurement", "1.8252 us",
        "--boundary", "只覆盖 fp16 small", "--counterexample", "GLOBAL scratch 不省 FLOPs",
        "--out", str(out),
    ])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["confidence"] == "reported" and printed["prompt_eligible"] is False
    assert printed["layer"] == "chip"
    assert printed["path"] == "targets/apple-m4-max/experience.md"
    assert "Confidence: reported" in out.read_text(encoding="utf-8")
    refused = main([
        "harness", "draft-experience",
        "--target", "apple-m4-max", "--task", "gemm_silu", "--title", "t",
        "--source", "runs/example/report.json", "--measurement", "1 us",
        "--boundary", "fp16", "--counterexample", "paper 不沿用", "--simulated",
    ])
    assert refused == 5
    verified_out = tmp_path / "verified.md"
    promoted = main([
        "harness", "draft-experience", "--set-verified",
        "--draft", str(out), "--host-rerun", "runs/example/host_paired_comparisons.jsonl",
        "--target", "apple-m4-max", "--out", str(verified_out),
    ])
    assert promoted == 0
    assert "Confidence: verified" in verified_out.read_text(encoding="utf-8")
    assert "宿主重跑: runs/example/host_paired_comparisons.jsonl" in verified_out.read_text(encoding="utf-8")


def test_generated_section_passes_the_real_experience_wiki_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 生成小节并进真实经验文件的副本后，reported 和 verified 都过现有 wiki 合同。"""
    from tests.shared import test_wiki_experience as wiki

    original = Path(wiki.EXPERIENCE).read_text(encoding="utf-8")
    section = render_draft(**_KW)
    reported = tmp_path / "reported.md"
    reported.write_text(original.rstrip() + "\n\n" + section, encoding="utf-8")
    monkeypatch.setattr(wiki, "EXPERIENCE", reported)
    wiki.test_every_entry_states_its_confidence()
    wiki.test_a_reported_entry_is_never_wired_into_the_prompt()
    verified = set_verified(section, "runs/example/host_paired_comparisons.jsonl")
    promoted = tmp_path / "verified.md"
    promoted.write_text(original.rstrip() + "\n\n" + verified, encoding="utf-8")
    monkeypatch.setattr(wiki, "EXPERIENCE", promoted)
    wiki.test_every_entry_states_its_confidence()
    wiki.test_a_verified_entry_cites_a_run_or_a_document()
    assert "寄存器里做完 SiLU" not in Path("targets/apple-m4-max/experience.md").read_text(encoding="utf-8")


def test_set_verified_rejects_a_partial_or_escaping_section():
    """@brief 升 verified 只接受恰好一节完整草稿，指针不能逃出 runs/。"""
    section = render_draft(**_KW)
    pointer = "runs/example/host_paired_comparisons.jsonl"
    assert valid_host_pointer("runs/../outside.json") is False
    assert valid_host_pointer("runs/./inside.json") is False
    with pytest.raises(DraftError, match="exactly one"):
        set_verified(section + "\n" + section.replace("寄存器里做完 SiLU", "另一节"), pointer)
    with pytest.raises(DraftError, match="chip"):
        set_verified(section.replace("落点: targets/apple-m4-max/experience.md", "落点: backends/metal/experience.md"), pointer)
    with pytest.raises(DraftError, match="simulated"):
        set_verified(section.replace("本机实测: fp16 small 1.8252 us", "本机实测: simulated 1.2 us"), pointer)
    with pytest.raises(DraftError, match="host rerun"):
        set_verified(section, "runs/../outside.json")


def test_cli_refuses_set_verified_when_simulated(tmp_path: Path, capsys):
    """@brief --set-verified 和 --simulated 不能一起用。"""
    draft = tmp_path / "draft.md"
    draft.write_text(render_draft(**_KW), encoding="utf-8")
    code = main([
        "harness", "draft-experience", "--set-verified", "--simulated",
        "--draft", str(draft), "--host-rerun", "runs/example/host_paired_comparisons.jsonl",
        "--target", "apple-m4-max",
    ])
    assert code == 5
    assert "simulated" in capsys.readouterr().err


def _practice() -> dict:
    return {
        "task": "conv2d",
        "title": "线程组上限",
        "source": "runs/example/report.json",
        "measurement": "64 threads, 1.2 us on apple-m4-max",
        "boundary": "只覆盖 fp16 conv2d",
        "counterexample": "paper 档的大 tile 不沿用这条",
    }


def _convertible(tmp_path: Path, **extra):
    import time

    from taic.agent_feedback import parse_feedback

    run = tmp_path / "run"
    work = run / "agent_work" / "init"
    work.mkdir(parents=True)
    (work / "probe.py").write_text("print(1)\n", encoding="utf-8")
    text = "成绩单加印线程组上限。"
    parsed = parse_feedback(f"# 对系统的建议\n- {text}\n")
    item = next(row for row in parsed["items"] if row["section"] == "对系统的建议")
    now = time.time()
    record = {
        "seq": 1, "t": now, "task": "conv2d", "dtype": "fp16", "label": "init",
        "work_dir": "agent_work/init", "session_started": now - 60,
        "items": [item],
    }
    record.update(extra)
    (run / "state.json").write_text(json.dumps({"task": "conv2d", "phase": "done"}), encoding="utf-8")
    (run / "feedback.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    (run / "run_config.json").write_text(json.dumps({"task": "conv2d", "dtype": "fp16", "target": {"id": "apple-m4-max"}}), encoding="utf-8")
    return run, item["digest"]


def test_convert_practice_reaches_write_patch_and_a_swap_is_refused(tmp_path: Path, capsys):
    """@brief practice 从 convert、CLI 和看板进入提案；换掉测量后同一个 key 不能生成草稿。"""
    from observe.actions import convert_harness
    from taic.harness import convert_feedback, write_patch

    run, digest = _convertible(tmp_path)
    action = "先读 device.md 的线程组上限再选 tile。"
    with pytest.raises(ProposalError, match="practice.task"):
        convert_feedback(
            run, seq=1, digest=digest, target_id="apple-m4-max", task="conv2d", dtype="fp16",
            surface="tip", action=action, probe="probe.py", practice={**_practice(), "task": "gemm"},
            by="alice", confirm=True,
        )
    code = main([
        "harness", "convert", str(run), "--seq", "1", "--digest", digest, "--surface", "tip",
        "--action", action, "--probe", "probe.py", "--task", "conv2d", "--confirm",
        "--practice-json", json.dumps(_practice()),
    ])
    assert code == 0
    row = json.loads(capsys.readouterr().out)
    assert row["proposal"]["practice"]["task"] == "conv2d"
    assert row["proposal"]["practice"]["boundary"].startswith("只覆盖")
    repo = tmp_path / "repo"
    (repo / "targets" / "apple-m4-max").mkdir(parents=True)
    (repo / "targets" / "apple-m4-max" / "experience.md").write_text("# Experience\n", encoding="utf-8")
    meta = write_patch(run, target_id="apple-m4-max", proposal_key=row["proposal_key"], root=repo)
    patch = Path(meta["patch"]).read_text(encoding="utf-8")
    assert "Confidence: reported" in patch and "未验证" in patch and "`runs/example/report.json`" in patch
    stored = json.loads((run / "harness_proposals.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    stored["proposal"]["practice"]["measurement"] = "999 us on a different machine"
    (run / "harness_proposals.jsonl").write_text(json.dumps(stored, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ProposalError, match="practice"):
        write_patch(run, target_id="apple-m4-max", proposal_key=row["proposal_key"], root=repo)
    assert "999 us" not in Path(meta["patch"]).read_text(encoding="utf-8")
    from taic.acceptance import make_spec, read_spec
    from taic.review import build_review_prompt, evidence_bundle

    spec = make_spec(row)
    assert spec["practice"]["task"] == "conv2d"
    assert read_spec(spec)["practice"]["boundary"] == spec["practice"]["boundary"]
    bundle = evidence_bundle(run, row)
    assert bundle["practice"]["measurement"].startswith("64 threads")
    prompt = build_review_prompt(row["proposal"], run_id=row["run_id"], bundle=bundle)
    assert "practice:" in prompt and "64 threads" in prompt
    assert "practice:" not in build_review_prompt({"action": action}, run_id="run")

    other, other_digest = _convertible(tmp_path / "dash")
    from observe import query

    monkey_run = other
    original_resolve = query.resolve_run
    query.resolve_run = lambda run_id: monkey_run if run_id == "dash-run" else original_resolve(run_id)
    try:
        converted = convert_harness("dash-run", 1, other_digest, {
            "confirm": True, "surface": "tip", "action": action, "probe": "probe.py",
            "task": "conv2d", "target_id": "apple-m4-max", "practice": _practice(), "by": "dashboard",
        })
    finally:
        query.resolve_run = original_resolve
    assert converted["proposal"]["practice"]["source"] == "runs/example/report.json"
    page = Path("observe/static/index.html").read_text(encoding="utf-8")
    assert 'name="practice_source"' in page and "practicePayload" in page


def test_write_patch_refuses_simulated_evidence_level_and_provenance(tmp_path: Path):
    """@brief evidence_level、practice、provenance 里的模拟声明都不生成草稿。"""
    from taic.harness import write_patch

    action = "先读 device.md 的线程组上限再选 tile。"
    practice = _practice()
    cases = [
        {"evidence_level": "simulated", "simulated": False, "measurement_kind": "measured"},
        {"proposal_evidence_level": "simulated"},
        {"practice_evidence_level": "simulated"},
        {"provenance": {"evidence_level": "simulated", "measurement_type": "simulated"}},
    ]
    for index, case in enumerate(cases):
        run = tmp_path / f"run-{index}"
        run.mkdir()
        proposal = {
            "surface": "tip",
            "action": action,
            "audience": "peer",
            "needs": "none",
            "scope": {"target_id": "apple-m4-max", "task": "conv2d"},
            "practice": dict(practice),
        }
        if case.get("proposal_evidence_level"):
            proposal["evidence_level"] = case["proposal_evidence_level"]
        if case.get("practice_evidence_level"):
            proposal["practice"]["evidence_level"] = case["practice_evidence_level"]
        if case.get("provenance"):
            proposal["evidence"] = {"provenance": case["provenance"]}
        row = {
            "status": "valid",
            "proposal_key": f"sim{index}",
            "simulated": case.get("simulated", False),
            "measurement_kind": case.get("measurement_kind", "measured"),
            "proposal": proposal,
        }
        if case.get("evidence_level"):
            row["evidence_level"] = case["evidence_level"]
        (run / "harness_proposals.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(ProposalError, match="simulated"):
            write_patch(run, target_id="apple-m4-max", proposal_key=row["proposal_key"], root=tmp_path)
        assert not (run / "harness_promotions").exists()


def _convert_for_draft(run: Path, digest: str):
    from taic.harness import convert_feedback

    return convert_feedback(
        run, seq=1, digest=digest, target_id="apple-m4-max", task="conv2d", dtype="fp16",
        surface="tip", action="先读 device.md 的线程组上限再选 tile。", probe="probe.py",
        practice=_practice(), by="alice", confirm=True,
    )


def _assert_simulated_row(row: dict) -> None:
    assert row["simulated"] is True
    assert row["measurement_kind"] == "simulated"
    assert row["evidence_level"] == "simulated"
    assert row["evidence_provenance"]["level"] == "simulated"
    assert row["evidence_provenance"]["simulated"] is True


def test_convert_feedback_keeps_simulation_and_write_patch_refuses(tmp_path: Path):
    """@brief 公开转换不能把模拟声明洗成实测，write_patch 也不生成草稿。"""
    from taic.harness import ProposalError, convert_feedback, write_patch

    markers = [
        {"evidence_level": "simulated", "measurement_kind": "measured"},
        {"measurement_type": "simulated", "measurement_kind": "measured"},
        {"evidence_provenance": {"level": "simulated", "simulated": True}, "measurement_kind": "measured"},
    ]
    action = "先读 device.md 的线程组上限再选 tile。"
    for index, marker in enumerate(markers):
        run, digest = _convertible(tmp_path / f"sim-{index}", **marker)
        row = convert_feedback(
            run, seq=1, digest=digest, target_id="apple-m4-max", task="conv2d", dtype="fp16",
            surface="tip", action=action, probe="probe.py", practice=_practice(),
            by="alice", confirm=True,
        )
        _assert_simulated_row(row)
        stored = json.loads((run / "harness_proposals.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        _assert_simulated_row(stored)
        repo = tmp_path / f"repo-{index}"
        (repo / "targets" / "apple-m4-max").mkdir(parents=True)
        (repo / "targets" / "apple-m4-max" / "experience.md").write_text("# Experience\n", encoding="utf-8")
        with pytest.raises(ProposalError, match="simulated"):
            write_patch(run, target_id="apple-m4-max", proposal_key=row["proposal_key"], root=repo)
        assert not (run / "harness_promotions").exists()

    plain, digest = _convertible(tmp_path / "plain")
    row = _convert_for_draft(plain, digest)
    assert row["simulated"] is False
    assert row["measurement_kind"] == "measured"
    assert row["evidence_level"] == "model_script"
    assert row["evidence_provenance"]["simulated"] is False
    repo = tmp_path / "plain-repo"
    (repo / "targets" / "apple-m4-max").mkdir(parents=True)
    (repo / "targets" / "apple-m4-max" / "experience.md").write_text("# Experience\n", encoding="utf-8")
    meta = write_patch(plain, target_id="apple-m4-max", proposal_key=row["proposal_key"], root=repo)
    assert Path(meta["patch"]).is_file()


def test_record_proposals_keeps_simulation_declarations(tmp_path: Path):
    """@brief record_proposals 用同一套声明，不把模拟记录记成 measured / model_script。"""
    import time

    from taic import harness
    from taic.harness import ProposalError, write_patch

    doc = {
        "surface": "tip",
        "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"},
        "action": "先读 device.md 的线程组上限再选 tile。",
        "evidence": {"probe": "probe.py", "measurement": "64 threads"},
        "rationale": "test",
    }
    parsed = {"sections": {"对系统的建议": "```harness-proposal\n" + json.dumps(doc, ensure_ascii=False) + "\n```\n"}}
    markers = [
        {"evidence_level": "simulated", "measurement_kind": "measured"},
        {"measurement_type": "simulated", "measurement_kind": "measured"},
        {"evidence_provenance": {"level": "simulated", "simulated": True}, "measurement_kind": "measured"},
    ]
    for index, marker in enumerate(markers):
        run = tmp_path / f"rec-{index}"
        work = run / "agent_work" / "r1.p0"
        work.mkdir(parents=True)
        (work / "probe.py").write_text("print(1)\n", encoding="utf-8")
        record = {
            "seq": 1, "label": "r1.p0", "committee": "first", "provider": "deepseek",
            "model": "deepseek-flash", "effort": "high", "run_id": run.name,
            **marker,
        }
        rows = harness.record_proposals(
            run, record=record, parsed=parsed, target_id="apple-m4-max", task="conv2d",
            dtype="fp16", work=work, started=time.time() - 2,
        )
        assert rows and rows[0]["status"] == "valid"
        _assert_simulated_row(rows[0])
        with pytest.raises(ProposalError, match="simulated"):
            write_patch(run, target_id="apple-m4-max", proposal_key=rows[0]["proposal_key"], root=tmp_path)
        assert not (run / "harness_promotions").exists()

    run = tmp_path / "plain-record"
    work = run / "agent_work" / "r1.p0"
    work.mkdir(parents=True)
    (work / "probe.py").write_text("print(1)\n", encoding="utf-8")
    rows = harness.record_proposals(
        run, record={"seq": 1, "label": "r1.p0", "run_id": run.name}, parsed=parsed,
        target_id="apple-m4-max", task="conv2d", dtype="fp16", work=work, started=time.time() - 2,
    )
    assert rows[0]["status"] == "valid"
    assert rows[0]["simulated"] is False
    assert rows[0]["measurement_kind"] == "measured"
    assert rows[0]["evidence_level"] == "model_script"
    assert rows[0]["evidence_provenance"]["simulated"] is False


def _repo(tmp_path: Path, name: str) -> Path:
    repo = tmp_path / name
    (repo / "targets" / "apple-m4-max").mkdir(parents=True)
    (repo / "targets" / "apple-m4-max" / "experience.md").write_text("# Experience\n", encoding="utf-8")
    return repo


def test_practice_simulation_reaches_write_patch(tmp_path: Path):
    """@brief practice 上的模拟声明经过 convert_feedback 和 record_proposals 后仍被 write_patch 拒绝。"""
    import time

    from taic import harness
    from taic.harness import convert_feedback, write_patch

    action = "先读 device.md 的线程组上限再选 tile。"
    markers = [
        {"evidence_level": "simulated"},
        {"measurement_type": "simulated"},
        {"measurement_kind": "simulated"},
        {"measurement": "simulated 1.2 us"},
    ]
    for index, extra in enumerate(markers):
        run, digest = _convertible(tmp_path / f"practice-{index}")
        practice = _practice()
        practice.update(extra)
        row = convert_feedback(
            run, seq=1, digest=digest, target_id="apple-m4-max", task="conv2d", dtype="fp16",
            surface="tip", action=action, probe="probe.py", practice=practice,
            by="alice", confirm=True,
        )
        kept = row["proposal"]["practice"]
        for key, value in extra.items():
            assert kept[key] == value
        assert kept["simulated"] is True
        stored = json.loads((run / "harness_proposals.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        assert stored["proposal"]["practice"]["simulated"] is True
        with pytest.raises(ProposalError, match="simulated"):
            write_patch(run, target_id="apple-m4-max", proposal_key=row["proposal_key"], root=_repo(tmp_path, f"repo-{index}"))
        assert not (run / "harness_promotions").exists()

    doc_base = {
        "surface": "tip",
        "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"},
        "action": action,
        "evidence": {"probe": "probe.py", "measurement": "64 threads"},
        "rationale": "test",
    }
    for index, extra in enumerate(markers):
        practice = _practice()
        practice.update(extra)
        doc = {**doc_base, "practice": practice}
        parsed = {"sections": {"对系统的建议": "```harness-proposal\n" + json.dumps(doc, ensure_ascii=False) + "\n```\n"}}
        run = tmp_path / f"recorded-{index}"
        work = run / "agent_work" / "r1.p0"
        work.mkdir(parents=True)
        (work / "probe.py").write_text("print(1)\n", encoding="utf-8")
        rows = harness.record_proposals(
            run, record={"seq": 1, "label": "r1.p0", "run_id": run.name}, parsed=parsed,
            target_id="apple-m4-max", task="conv2d", dtype="fp16", work=work, started=time.time() - 2,
        )
        assert rows and rows[0]["status"] == "valid"
        kept = rows[0]["proposal"]["practice"]
        for key, value in extra.items():
            assert kept[key] == value
        assert kept["simulated"] is True
        with pytest.raises(ProposalError, match="simulated"):
            write_patch(run, target_id="apple-m4-max", proposal_key=rows[0]["proposal_key"], root=_repo(tmp_path, f"recorded-repo-{index}"))
        assert not (run / "harness_promotions").exists()


def test_candidate_patch_refuses_a_bare_simulated_measurement(tmp_path: Path):
    """@brief 测量正文写 simulated 1.2 us 时，candidate_patch 不生成 Confidence: reported。"""
    practice = _practice()
    practice["measurement"] = "simulated 1.2 us"
    proposal = {
        "surface": "tip",
        "action": "先读 device.md 的线程组上限再选 tile。",
        "scope": {"target_id": "apple-m4-max", "task": "conv2d"},
        "practice": practice,
    }
    with pytest.raises(ProposalError, match="simulated"):
        candidate_patch(proposal, target_id="apple-m4-max", root=_repo(tmp_path, "repo"))
