"""Failure handoff: a stable JSON block, cleared when the next run starts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

REPO = Path(__file__).resolve().parents[1]
BEGIN = "---BEGIN_AI_FAILURE_HANDOFF---"
END = "---END_AI_FAILURE_HANDOFF---"

COMPILE_CASE = """
from tcenv.evaluate import EvalReport

def test_bad_kernel():
    rep = EvalReport(
        task="relu",
        dtype="fp16",
        size="small",
        mode="quick",
        status="compile_error",
        stage="compile",
        message="compile failed",
        compile={"ok": False, "errors": "use of undeclared identifier 'bogus_symbol'"},
    )
    assert rep.status == "ok"
"""

TWO_STATUSES = """
from tcenv.evaluate import EvalReport

def test_compile():
    rep = EvalReport(
        task="relu", dtype="fp16", size="small", mode="quick",
        status="compile_error", stage="compile", message="compile failed",
        compile={"ok": False, "errors": "missing semicolon"},
    )
    assert rep.status == "ok"

def test_math():
    rep = EvalReport(
        task="relu", dtype="fp16", size="small", mode="quick",
        status="incorrect", stage="correct", message="out of tolerance",
    )
    assert rep.status == "ok"
"""


def _prepare(pytester: pytest.Pytester) -> None:
    pytester.makepyprojecttoml('[tool.pytest.ini_options]\naddopts = []\ntestpaths = ["."]\n')
    pytester.makeconftest(
        "import sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "pytest_plugins = ['tests.handoff']\n"
    )


def _run(pytester: pytest.Pytester, *args: str):
    path = pytester.path
    return pytester.runpytest_subprocess(
        "--rootdir",
        str(path),
        "--confcutdir",
        str(path),
        "-c",
        str(path / "pyproject.toml"),
        "-o",
        "addopts=",
        "-o",
        "testpaths=.",
        *args,
    )


def _triage(pytester: pytest.Pytester) -> Path:
    return pytester.path / "logs" / "pytest" / "triage"


def _handoffs(pytester: pytest.Pytester) -> list[Path]:
    triage = _triage(pytester)
    if not triage.is_dir():
        return []
    return sorted(path for path in triage.glob("*.md") if path.name != "triage_summary.md")


def _payload(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    start = text.index(BEGIN) + len(BEGIN)
    end = text.index(END)
    return json.loads(text[start:end])


def test_failure_records_the_eval_report(pytester: pytest.Pytester):
    _prepare(pytester)
    pytester.makepyfile(test_bad_kernel=COMPILE_CASE)
    result = _run(pytester, "test_bad_kernel.py")
    result.assert_outcomes(failed=1)
    files = _handoffs(pytester)
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert BEGIN in text and END in text
    payload = _payload(files[0])
    assert payload["schema"] == "ai_lowerling_failure_handoff/v1"
    assert payload["error_kind"] == "compile_error"
    assert "bogus_symbol" in payload["feedback"]
    assert "bogus_symbol" in text
    command = payload["minimal_repro_cmd"]
    assert "uv run pytest -q --tb=short" in command
    assert "'test_bad_kernel.py::test_bad_kernel'" in command
    assert "find logs/pytest/triage -name '*.md'" in str(result.stdout)
    summary = (_triage(pytester) / "triage_summary.md").read_text(encoding="utf-8")
    assert "test_bad_kernel.py::test_bad_kernel" in summary
    assert str(files[0].resolve()) in summary


def test_a_passing_run_writes_no_handoff(pytester: pytest.Pytester):
    _prepare(pytester)
    pytester.makepyfile(test_ok="def test_ok():\n    assert True\n")
    result = _run(pytester, "test_ok.py")
    result.assert_outcomes(passed=1)
    assert _handoffs(pytester) == []
    assert not (_triage(pytester) / "triage_summary.md").exists()


def test_a_later_run_drops_the_previous_failure(pytester: pytest.Pytester):
    _prepare(pytester)
    pytester.makepyfile(test_old='def test_old():\n    raise AssertionError("stale-token")\n')
    pytester.makepyfile(test_new='def test_new():\n    raise AssertionError("fresh-token")\n')
    first = _run(pytester, "test_old.py")
    first.assert_outcomes(failed=1)
    assert "test_old.py::test_old" in (_triage(pytester) / "triage_summary.md").read_text(encoding="utf-8")
    second = _run(pytester, "test_new.py")
    second.assert_outcomes(failed=1)
    summary = (_triage(pytester) / "triage_summary.md").read_text(encoding="utf-8")
    assert "test_new.py::test_new" in summary
    assert "test_old" not in summary
    assert "stale-token" not in summary
    payload = _payload(_handoffs(pytester)[0])
    assert payload["error_kind"] == "assertion"
    assert len(_handoffs(pytester)) == 1


def test_summary_groups_distinct_statuses(pytester: pytest.Pytester):
    _prepare(pytester)
    pytester.makepyfile(test_kinds=TWO_STATUSES)
    result = _run(pytester, "test_kinds.py")
    result.assert_outcomes(failed=2)
    summary = (_triage(pytester) / "triage_summary.md").read_text(encoding="utf-8")
    assert "## compile_error — 1 failure" in summary
    assert "## incorrect — 1 failure" in summary
    assert "test_kinds.py::test_compile" in summary
    assert "test_kinds.py::test_math" in summary
    files = _handoffs(pytester)
    assert len(files) == 2
    for path in files:
        assert str(path.resolve()) in summary
    kinds = {_payload(path)["error_kind"] for path in files}
    assert kinds == {"compile_error", "incorrect"}


def test_the_switch_writes_nothing(pytester: pytest.Pytester):
    _prepare(pytester)
    pytester.makepyfile(test_bad="def test_bad():\n    assert False\n")
    result = _run(pytester, "--no-failure-handoff", "test_bad.py")
    result.assert_outcomes(failed=1)
    assert not _triage(pytester).exists()
    assert "AI failure handoff" not in str(result.stdout)
