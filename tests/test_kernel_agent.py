"""The pi coding agent writes kernel.metal. A fake hmz stands in for the real CLI."""

import json
import stat
from pathlib import Path

import pytest

from taic.kernel_agent import (
    TASK_MAX_BYTES, AgentError, AgentResult, Prices, SessionBrief, _self_check_dir, agent_spec, dollars_for, kernel_text,
    output_tokens_for, parse_spent, prune_work, render_contract, run_kernel_agent, spend_from_stream,
    submission_name,
)
from taic.llm import SearchConfig


PRICES = Prices(input=0.3, output=1.2, cache_read=0.006, cache_write=0.0)


def _brief(tmp_path: Path, *, seedless: bool = True) -> SessionBrief:
    work = tmp_path / "agent_work" / "init"
    return SessionBrief(
        repo=tmp_path, task_name="conv2d", dtype="fp16", size="paper", kernel_name="conv2d_kernel",
        triton_source="def conv():\n    pass\n", description="a convolution", contract_text="grid 1,1,1",
        abi_stub="kernel void conv2d_kernel() {}", rtol=1e-3, atol_rel=1e-3, seedless=seedless,
        reps=15, min_improvement=1.005, noise_z=2.0, work_dir=str(work.resolve()),
    )


def test_writing_kernels_goes_through_hmz():
    assert SearchConfig().kernel_agent == "pi"


def test_pi_names_the_model_and_the_effort():
    assert agent_spec("moonshot", "kimi-k3") == "pi/moonshot/kimi-k3"
    assert agent_spec("deepseek", "deepseek-flash", "high") == "pi/deepseek/deepseek-flash:high"


def test_a_file_that_declares_the_kernel_is_the_source_without_a_fence():
    assert kernel_text("kernel void conv2d_kernel() {}\n", "conv2d_kernel").startswith("kernel void conv2d_kernel")
    assert kernel_text("kernel void other() {}\n", "conv2d_kernel") is None
    staged = "/* taic-launch\nlaunch other grid=1,1,1 threads=32,1,1\n*/\nkernel void other() {}\n"
    assert kernel_text(staged, "conv2d_kernel").startswith("/* taic-launch")


def test_the_submission_name_is_the_task_and_the_session():
    assert submission_name("conv2d", "fp16", "paper", "r16.p0") == "conv2d-fp16-paper-r16.p0.metal"
    assert submission_name("flash attention", "fp16", "64x32", "init") == "flash-attention-fp16-64x32-init.metal"
    assert submission_name("conv2d", "fp16", "paper", "r1.p0") != submission_name("conv2d", "fp16", "paper", "r2.p0")


def test_the_contract_names_both_commands_and_does_not_ask_for_a_fence(tmp_path):
    brief = _brief(tmp_path)
    text = render_contract(brief)
    submit = str(Path(brief.work_dir) / "conv2d-fp16-paper-init.metal")
    assert "--quick" in text and "--timed" in text
    assert submit in text
    commands = text.split("# 验证命令", 1)[1].split("# 晋升标准", 1)[0]
    assert "kernel.metal" not in commands
    assert "conv2d-fp16-paper-init.metal" in text
    assert "self_checks.jsonl" in text
    assert "NOTES.md" in text
    assert "```metal block" not in text
    assert "does not time" not in text
    assert "先把一颗核写进 `conv2d-fp16-paper-init.metal`" in text
    assert "本场的主产出始终是 `conv2d-fp16-paper-init.metal`" in text
    assert "外层不读它" in text
    assert "# 这个系统怎么工作" in text
    for title in ("# 需要帮助", "# 对系统的建议", "# 卡住的原因", "# 给下一棒的避坑提示", "# 被否决的前提"):
        assert title in text
    assert "必须写进「被否决的前提」节" in text


def test_a_seeded_contract_leads_with_the_outer_measurement(tmp_path):
    brief = _brief(tmp_path, seedless=False)
    brief.seed_source = "kernel void conv2d_kernel() {}\n"
    brief.notes_summary = "外层实测：100 µs，状态 ok，没换种子\n上一版自报 10 µs"
    brief.bound_line = "本算法运算量下界 932 µs，最强基线 torch-mps 790 µs。下界高于基线时，不减少运算量的写法不可能超过它。"
    text = render_contract(brief)
    assert text.index("外层实测") < text.index("上一版自报")
    report = str(Path(brief.work_dir) / "report.json")
    submit = str(Path(brief.work_dir) / "conv2d-fp16-paper-init.metal")
    assert f"--seed-report {report}" in text
    assert submit in text
    timed = text.split("# 评测命令", 1)[1].split("# 晋升标准", 1)[0]
    assert "kernel.metal" not in timed
    assert str(Path(brief.work_dir) / "kernel.metal") in text
    assert "不减少运算量" in text
    brief.peer_tips = "- [r1 first/fake, evaluated 1.100x] float8 是保留字"
    seeded = render_contract(brief)
    assert seeded.index("上一轮 NOTES") < seeded.index("本任务前序避坑提示")
    assert "一律按未验证处理" in seeded
    assert "float8 是保留字" in seeded
    seedless = _brief(tmp_path, seedless=True)
    seedless.peer_tips = "- [init first/fake, agent_failed] simd_any 要在分支外"
    assert "simd_any 要在分支外" in render_contract(seedless)


def test_spent_on_a_result_line_is_priced_by_kind_and_result_lines_add():
    stream = "\n".join([
        json.dumps({"kind": "text", "text": "working"}),
        json.dumps({"kind": "result", "spent": "Usage(input=1000000, output=500000, cache_read=2000000)"}),
        json.dumps({"kind": "result", "spent": {"input": 1000000, "output": 500000}}),
    ])
    spent, saw = spend_from_stream(stream)
    assert saw
    assert spent["input"] == pytest.approx(2_000_000)
    assert spent["output"] == pytest.approx(1_000_000)
    assert spent["cache_read"] == pytest.approx(2_000_000)
    # 1e6 input * 0.3 + 1e6 output * 1.2 + 2e6 cache_read * 0.006, per million
    assert dollars_for(spent, PRICES) == pytest.approx(0.3 * 2 + 1.2 * 1 + 0.006 * 2)
    assert spend_from_stream("not json") == ({}, False)
    assert parse_spent("Usage(input=3)") == {"input": 3.0}


def test_a_dollar_slice_becomes_output_tokens():
    assert output_tokens_for(1.2, 1.2) == 1_000_000
    assert output_tokens_for(0.0000001, 1.2) == 1


def _fake_hmz(path: Path, *, source: str, code: int = 0, spent: str = "Usage(output=1000)", write_submit: bool = True) -> None:
    line = json.dumps({"kind": "result", "spent": spent})
    write = f"pathlib.Path(name).write_text({source!r})\n" if write_submit else ""
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "name = 'missing.metal'\n"
        "for arg in sys.argv:\n"
        "    if arg.startswith('submit_name='):\n"
        "        name = arg.split('=', 1)[1]\n"
        f"{write}"
        "pathlib.Path('NOTES.md').write_text('kept this one\\n')\n"
        f"print({line!r})\n"
        f"sys.exit({code})\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_a_fake_hmz_leaves_the_kernel_and_a_computed_cost(tmp_path):
    hmz = tmp_path / "hmz"
    _fake_hmz(hmz, source="kernel void conv2d_kernel() {}\n")
    result = run_kernel_agent(
        _brief(tmp_path), work=tmp_path / "work", provider="deepseek", model="deepseek-flash",
        effort="high", prices=PRICES, turn_output_tokens=1000, turn_minutes=25, hmz=str(hmz),
    )
    assert isinstance(result, AgentResult)
    assert result.source.startswith("kernel void conv2d_kernel")
    assert result.cost_usd == pytest.approx(1.2 * 1000 / 1_000_000)
    assert result.cost_incomplete is False
    assert result.notes.startswith("kept")
    task = (tmp_path / "work" / "TASK.md").read_text()
    assert "assistant=" not in task
    assert "--timed" in task
    assert "turn_output_tokens" not in task


def test_a_missing_hmz_is_an_error_and_a_timeout_still_returns_the_file(tmp_path):
    with pytest.raises(AgentError, match="not on PATH"):
        run_kernel_agent(
            _brief(tmp_path), work=tmp_path / "work", provider="deepseek", model="deepseek-flash",
            effort=None, prices=PRICES, turn_output_tokens=10, turn_minutes=1, hmz=str(tmp_path / "missing"),
        )
    hmz = tmp_path / "hmz"
    hmz.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys, time\n"
        "name = 'missing.metal'\n"
        "for arg in sys.argv:\n"
        "    if arg.startswith('submit_name='):\n"
        "        name = arg.split('=', 1)[1]\n"
        "pathlib.Path(name).write_text('kernel void conv2d_kernel() { /* new */ }\\n')\n"
        "time.sleep(30)\n"
    )
    hmz.chmod(hmz.stat().st_mode | stat.S_IEXEC)
    result = run_kernel_agent(
        _brief(tmp_path), work=tmp_path / "slow", provider="deepseek", model="deepseek-flash",
        effort="high", prices=PRICES, turn_output_tokens=10, turn_minutes=1, hmz=str(hmz), timeout_s=2,
    )
    assert result.status == "timeout"
    assert result.cost_incomplete is True
    assert "new" in result.source


def test_a_seed_left_in_kernel_metal_is_not_a_submission(tmp_path):
    hmz = tmp_path / "hmz"
    _fake_hmz(hmz, source="kernel void conv2d_kernel() { /* new */ }\n", write_submit=False)
    brief = _brief(tmp_path, seedless=False)
    brief.seed_source = "kernel void conv2d_kernel() { /* seed */ }\n"
    work = tmp_path / "work"
    work.mkdir()
    (work / "conv2d-fp16-paper-init.metal").write_text("kernel void conv2d_kernel() { /* stale */ }\n")
    result = run_kernel_agent(
        brief, work=work, provider="deepseek", model="deepseek-flash",
        effort=None, prices=PRICES, turn_output_tokens=10, turn_minutes=1, hmz=str(hmz),
    )
    assert result.source is None
    # The seed kernel is archived after the session (scratch/<label>.tar.gz); it is never read as the submission.
    assert not (work / "conv2d-fp16-paper-init.metal").exists()


def test_a_nonzero_exit_with_no_kernel_is_a_failure(tmp_path):
    hmz = tmp_path / "hmz"
    _fake_hmz(hmz, source="// no kernel\n", code=2)
    result = run_kernel_agent(
        _brief(tmp_path), work=tmp_path / "empty", provider="deepseek", model="deepseek-flash",
        effort=None, prices=PRICES, turn_output_tokens=10, turn_minutes=1, hmz=str(hmz),
    )
    assert result.source is None and result.status == "failed"
    assert result.feedback == ""


def test_feedback_is_read_with_the_notes_and_capped(tmp_path):
    hmz = tmp_path / "hmz"
    _fake_hmz(hmz, source="kernel void conv2d_kernel() {}\n")
    script = hmz.read_text().replace(
        "pathlib.Path('NOTES.md').write_text('kept this one\\n')\n",
        "pathlib.Path('NOTES.md').write_text('kept this one\\n')\n"
        "pathlib.Path('FEEDBACK.md').write_text('kind: none\\n' + ('坑' * 20000))\n",
    )
    hmz.write_text(script)
    result = run_kernel_agent(
        _brief(tmp_path), work=tmp_path / "work", provider="deepseek", model="deepseek-flash",
        effort=None, prices=PRICES, turn_output_tokens=10, turn_minutes=1, hmz=str(hmz),
    )
    assert result.feedback.endswith("坑")
    assert not result.feedback.startswith("kind: none")
    assert len(result.feedback) == 16384
    assert result.notes.startswith("kept")


def test_task_md_keeps_its_dynamic_blocks_inside_the_budget(tmp_path):
    brief = _brief(tmp_path, seedless=False)
    brief.scorecard = "s" * 5000
    brief.notes_summary = "n" * 9000
    brief.peer_tips = "t" * 9000
    brief.history = "h" * 5000
    brief.failure = "f" * 5000
    brief.evidence = "e" * 5000
    text = render_contract(brief)
    assert len(text.encode("utf-8")) <= TASK_MAX_BYTES
    assert "# 本轮省略（上下文预算）" in text
    assert "NOTES.md" in text and "FEEDBACK.md" in text
    # the required contract, signature and both commands survive the trim
    assert "# 约束" in text and "kernel void conv2d_kernel()" in text
    assert "--quick" in text and "--timed" in text


def test_prune_archives_scratch_and_keeps_the_files_the_search_reads(tmp_path):
    work = tmp_path / "run" / "agent_work" / "r1"
    (work / "v").mkdir(parents=True)
    (work / "v" / "ablation.metal").write_text("x" * 2000)
    (work / "probe.py").write_text("print(1)\n")
    (work / "NOTES.md").write_text("notes\n")
    (work / "self_checks.jsonl").write_text("{}\n")
    out = prune_work(work, label="r1")
    assert not (work / "v").exists() and not (work / "probe.py").exists()
    assert (work / "NOTES.md").is_file() and (work / "self_checks.jsonl").is_file()
    archive = tmp_path / "run" / "scratch" / "r1.tar.gz"
    assert archive.is_file() and archive.stat().st_size > 0
    assert "v" in out["archived"] and "probe.py" in out["archived"]


def _real_brief(tmp_path, task_name: str) -> SessionBrief:
    from tasks import get_task

    task = get_task(task_name, "fp16", "paper")
    work = tmp_path / "agent_work" / "r1"
    return SessionBrief(
        repo=tmp_path, task_name=task_name, dtype="fp16", size="paper",
        kernel_name=task.contract.kernel_name, triton_source=task.triton_source,
        description=task.description, contract_text=task.contract.describe(),
        abi_stub=task.contract.abi_stub(), rtol=task.rtol, atol_rel=task.atol_rel,
        seedless=False, reps=15, min_improvement=1.005, noise_z=2.0, work_dir=str(work.resolve()),
    )


@pytest.mark.parametrize("task_name", ["conv2d", "flash_attention"])
def test_a_real_task_keeps_a_partial_scorecard_instead_of_dropping_it(tmp_path, task_name):
    brief = _real_brief(tmp_path, task_name)
    brief.scorecard = "s" * 5000
    brief.notes_summary = "n" * 9000
    brief.peer_tips = "t" * 9000
    brief.history = "h" * 5000
    brief.failure = "f" * 5000
    brief.evidence = "e" * 5000
    text = render_contract(brief)
    assert len(text.encode("utf-8")) <= TASK_MAX_BYTES
    assert "# 约束" in text and "--quick" in text and "--timed" in text
    assert "外层成绩单" in text  # the current-site block is trimmed, not dropped
    assert "本段已截断" in text
    assert "# 本轮省略（上下文预算）" in text
    assert "需要时读取" in text


def test_the_self_check_log_is_inside_the_writable_work_directory(tmp_path):
    brief = _brief(tmp_path)
    log = _self_check_dir(brief)
    work = Path(brief.work_dir)
    assert log == work / ".selfcheck"
    assert work in log.parents


def test_prune_keeps_a_probe_named_by_the_feedback(tmp_path):
    work = tmp_path / "run" / "agent_work" / "r1"
    work.mkdir(parents=True)
    (work / "FEEDBACK.md").write_text("[已实测: probe.py | x | 1.0 us]\n")
    (work / "probe.py").write_text("print(1)\n")
    (work / "other.py").write_text("print(2)\n")
    out = prune_work(work, label="r1")
    assert (work / "probe.py").is_file() and "probe.py" in out["kept"]
    assert not (work / "other.py").exists()
