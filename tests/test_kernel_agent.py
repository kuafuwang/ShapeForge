"""The pi coding agent writes kernel.metal. A fake hmz stands in for the real CLI."""

import json
import stat
from pathlib import Path

import pytest

from taic.kernel_agent import AgentError, agent_spec, cost_from_stream, kernel_text, run_kernel_agent, task_markdown
from taic.llm import SearchConfig


def test_the_chat_api_still_writes_kernels_unless_pi_is_asked_for():
    assert SearchConfig().kernel_agent == ""


def test_pi_names_the_model_as_provider_and_id():
    assert agent_spec("moonshot", "kimi-k3") == "pi/moonshot/kimi-k3"


def test_a_file_that_declares_the_kernel_is_the_source_without_a_fence():
    assert kernel_text("kernel void conv2d_kernel() {}\n", "conv2d_kernel").startswith("kernel void conv2d_kernel")
    assert kernel_text("kernel void other() {}\n", "conv2d_kernel") is None
    staged = "/* taic-launch\nlaunch other grid=1,1,1 threads=32,1,1\n*/\nkernel void other() {}\n"
    assert kernel_text(staged, "conv2d_kernel").startswith("/* taic-launch")


def test_the_task_tells_the_agent_to_read_notes_and_run_tcenv_not_our_tools(tmp_path):
    text = task_markdown(
        [{"role": "user", "content": "Reply with one ```metal block."}],
        repo=tmp_path, kernel_path=tmp_path / "kernel.metal", task_name="conv2d", dtype="fp16", size="paper", kernel_name="conv2d_kernel",
    )
    assert "tcenv.cli eval conv2d" in text and "--quick" in text
    assert "skill named taic-tools" in text
    assert "lookup_msl: read `msl_reference.md`" in text
    assert "quick_check: bash this command" in text
    assert "does not time the kernel" in text
    assert "recorded as unchanged" in text
    assert "Ignore that reply format" in text


def test_cost_from_the_stream_is_the_largest_cumulative_cost():
    stream = "\n".join([
        json.dumps({"event": "turn", "usage": {"cost": 0.1}}),
        json.dumps({"event": "finished", "usage": {"cost": 0.4}}),
    ])
    assert cost_from_stream(stream) == pytest.approx(0.4)
    assert cost_from_stream("not json") is None


def _fake_hmz(path: Path, *, source: str, code: int = 0, cost: float = 0.2) -> None:
    line = json.dumps({"event": "finished", "usage": {"cost": cost}})
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        f"pathlib.Path('kernel.metal').write_text({source!r})\n"
        f"print({line!r})\n"
        f"sys.exit({code})\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_a_fake_hmz_leaves_the_kernel_and_its_cost(tmp_path):
    hmz = tmp_path / "hmz"
    _fake_hmz(hmz, source="kernel void conv2d_kernel() {}\n", cost=0.2)
    source, cost, _tail = run_kernel_agent(
        [{"role": "system", "content": "contract"}, {"role": "user", "content": "write it"}],
        work=tmp_path / "work", repo=tmp_path, task_name="conv2d", dtype="fp16", size="paper",
        kernel_name="conv2d_kernel", provider="deepseek", model="deepseek-flash", budget_usd=3,
        hmz=str(hmz),
    )
    assert source.startswith("kernel void conv2d_kernel")
    assert cost == pytest.approx(0.2)
    task = (tmp_path / "work" / "TASK.md").read_text()
    assert "assistant=" not in task
    assert "--quick" in task


def test_a_missing_hmz_is_an_error_and_an_empty_file_is_no_kernel(tmp_path):
    with pytest.raises(AgentError):
        run_kernel_agent(
            [], work=tmp_path / "work", repo=tmp_path, task_name="conv2d", dtype="fp16", size="paper",
            kernel_name="conv2d_kernel", provider="deepseek", model="deepseek-flash", budget_usd=1,
            hmz=str(tmp_path / "missing"),
        )
    hmz = tmp_path / "hmz"
    _fake_hmz(hmz, source="// no kernel\n", code=0)
    source, _cost, _tail = run_kernel_agent(
        [], work=tmp_path / "empty", repo=tmp_path, task_name="conv2d", dtype="fp16", size="paper",
        kernel_name="conv2d_kernel", provider="deepseek", model="deepseek-flash", budget_usd=1,
        hmz=str(hmz),
    )
    assert source is None
