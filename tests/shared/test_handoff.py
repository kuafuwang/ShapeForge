"""Failure handoff: a stable JSON block, cleared when the next run starts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

REPO = Path(__file__).resolve().parents[2]
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
    """@brief 在 pytester 的临时目录里写入最小 pyproject 与 conftest，使子进程能加载 handoff 插件。

    conftest 把仓库根插到 ``sys.path``，并声明 ``pytest_plugins = ['tests.support.handoff']``。

    @param pytester pytest 的 pytester 夹具，提供临时工程目录与文件构造方法。
    """
    pytester.makepyprojecttoml('[tool.pytest.ini_options]\naddopts = []\ntestpaths = ["."]\n')
    pytester.makeconftest(
        "import sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "pytest_plugins = ['tests.support.handoff']\n"
    )


def _run(pytester: pytest.Pytester, *args: str):
    """@brief 在子进程里跑 pytest，显式固定 rootdir／confcutdir／配置文件并清空 addopts。

    @param pytester pytest 的 pytester 夹具，提供临时工程路径与 ``runpytest_subprocess``。
    @param args 追加给 pytest 的参数（例如测试文件名或 ``--no-failure-handoff``）。
    @return 子进程运行结果对象（含 stdout／stderr 与 outcomes 断言辅助）。
    """
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
    """@brief 返回临时工程里 triage 日志目录的路径（不保证已存在）。

    @param pytester pytest 的 pytester 夹具。
    @return ``logs/pytest/triage`` 的路径。
    """
    return pytester.path / "logs" / "pytest" / "triage"


def _handoffs(pytester: pytest.Pytester) -> list[Path]:
    """@brief 列出 triage 目录下的 handoff 文件（排除汇总文件 ``triage_summary.md``）。

    @param pytester pytest 的 pytester 夹具。
    @return 按路径排序的 handoff 文件列表；目录不存在时返回空列表。
    """
    triage = _triage(pytester)
    if not triage.is_dir():
        return []
    return sorted(path for path in triage.glob("*.md") if path.name != "triage_summary.md")


def _payload(path: Path) -> dict:
    """@brief 从 handoff 文件里取出 BEGIN／END 标记之间的 JSON 负载并解析。

    @param path 单个 handoff 文件路径。
    @return 解析后的 JSON 字典。
    """
    text = path.read_text(encoding="utf-8")
    start = text.index(BEGIN) + len(BEGIN)
    end = text.index(END)
    return json.loads(text[start:end])


def test_failure_records_the_eval_report(pytester: pytest.Pytester):
    """@brief 验证失败用例会写出带 schema 标记与编译错误的 handoff 文件，并在汇总与输出中可被找到。

    校验点：只有一个 handoff 文件、包含 BEGIN／END 标记、负载 schema 为
    ``ai_lowerling_failure_handoff/v1``、error_kind 为 compile_error、feedback 含具体符号名、
   ``minimal_repro_cmd`` 给出可复制的最小复现命令，stdout 与 ``triage_summary.md`` 指向该文件。

    @param pytester pytest 的 pytester 夹具，用来搭临时工程并跑子进程 pytest。
    """
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
    """@brief 验证全部通过的运行不产生 handoff，也不写 triage 汇总。

    @param pytester pytest 的 pytester 夹具，用来搭临时工程并跑子进程 pytest。
    """
    _prepare(pytester)
    pytester.makepyfile(test_ok="def test_ok():\n    assert True\n")
    result = _run(pytester, "test_ok.py")
    result.assert_outcomes(passed=1)
    assert _handoffs(pytester) == []
    assert not (_triage(pytester) / "triage_summary.md").exists()


def test_a_later_run_drops_the_previous_failure(pytester: pytest.Pytester):
    """@brief 验证新一次运行开始时清掉上一次的失败遗留：旧汇总与旧 handoff 内容不再出现。

    两次失败分别用 ``stale-token`` 与 ``fresh-token``，汇总只含新的失败，handoff 只保留一份，
    ``error_kind`` 为 assertion。

    @param pytester pytest 的 pytester 夹具，用来搭临时工程并跑子进程 pytest。
    """
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
    """@brief 验证汇总按 error_kind 分组统计：两个不同错误各自的标题带 1 failure 计数。

    两个失败用例分别产生 compile_error 与 incorrect，汇总里两种标题与两个用例名都在，
    每个 handoff 文件路径都列在汇总里，负载里的 error_kind 集合正好是这两种。

    @param pytester pytest 的 pytester 夹具，用来搭临时工程并跑子进程 pytest。
    """
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
    """@brief 验证 ``--no-failure-handoff`` 开关关掉 handoff：失败仍发生，但不建 triage 目录也不打印提示。

    @param pytester pytest 的 pytester 夹具，用来搭临时工程并跑子进程 pytest。
    """
    _prepare(pytester)
    pytester.makepyfile(test_bad="def test_bad():\n    assert False\n")
    result = _run(pytester, "--no-failure-handoff", "test_bad.py")
    result.assert_outcomes(failed=1)
    assert not _triage(pytester).exists()
    assert "AI failure handoff" not in str(result.stdout)
