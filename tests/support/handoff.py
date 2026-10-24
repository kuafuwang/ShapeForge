"""Write a short failure handoff an agent can read without the full traceback.

On a call-phase failure the plugin writes ``logs/pytest/triage/<worker>/<nodeid>.md``
in an xdist run (the controller also writes a root ``triage_summary.md``) with a
JSON block (schema ``ai_lowerling_failure_handoff/v1``) between
``---BEGIN_AI_FAILURE_HANDOFF---`` and ``---END_AI_FAILURE_HANDOFF---``.
That block is the ground truth: node id, error kind, a capped
``format_feedback`` of any ``EvalReport`` still in the failing frame, and a
single-process reproduce command. The session starts by deleting the triage
directory so a later run cannot be confused with an earlier one.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from . import run_logs

SCHEMA = "ai_lowerling_failure_handoff/v1"
BEGIN = "---BEGIN_AI_FAILURE_HANDOFF---"
END = "---END_AI_FAILURE_HANDOFF---"
FEEDBACK_MAX = 4000
MESSAGE_LINES_MAX = 24
PARAM_REPR_MAX = 200
ENV_KEYS = (
    "TCENV_ALLOW_GPU_HANG_TESTS",
    "TCENV_RUN_XCTRACE",
    "MTL_SHADER_VALIDATION",
    "TCENV_GPU_LOCK_FILE",
)
_SKIP_MARKERS = frozenset({"parametrize"})
_PREFERRED_REPORT_NAMES = ("rep", "report")

_GIT: pytest.StashKey[str] = pytest.StashKey()
_RECORDS: pytest.StashKey[list[dict[str, Any]]] = pytest.StashKey()


def pytest_addoption(parser: pytest.Parser) -> None:
    """@brief 注册 `--no-failure-handoff` 选项，用于关闭失败交接 Markdown 的写入。

    @param parser pytest 选项解析器。
    """
    parser.addoption(
        "--no-failure-handoff",
        action="store_true",
        default=False,
        help="Do not write logs/pytest/triage markdown when a test fails",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    """@brief 会话开始时记录 git 短哈希并清空记录列表；主进程会删掉旧的 triage 目录。

    @param session 当前 pytest 会话。
    """
    config = session.config
    config.stash[_GIT] = _git_short(Path(config.rootpath))
    config.stash[_RECORDS] = []
    if _disabled(config):
        return
    if _worker_id(config) == "master":
        triage = _triage_dir(config)
        if triage.exists():
            shutil.rmtree(triage)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    """@brief hookwrapper：仅在 call 阶段失败时为该用例写一份失败交接。

    @param item 当前测试项。
    @param call 当前测试的调用信息。
    """
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed:
        return
    if _disabled(item.config):
        return
    _record_failure(item, call, report)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """@brief 会话结束时汇总失败记录并写出 triage_summary.md。

    主进程优先从各 worker 的 manifest.jsonl 读取记录，xdist 下才能拿到全部失败。

    @param session 当前 pytest 会话。
    @param exitstatus pytest 退出码（本函数未使用）。
    """
    del exitstatus
    config = session.config
    if _disabled(config):
        return
    records = list(config.stash.get(_RECORDS, []))
    if _worker_id(config) == "master":
        records = _read_manifest_records(_triage_root(config)) or records
    if records:
        _write_summary(config, records)


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    """@brief 在终端摘要里列出每份失败交接的 Markdown 路径，并给出汇总文件位置。

    @param terminalreporter pytest 终端输出对象。
    @param exitstatus pytest 退出码（本函数未使用）。
    @param config pytest 配置对象。
    """
    del exitstatus
    if _disabled(config):
        return
    records = list(config.stash.get(_RECORDS, []))
    if _worker_id(config) == "master" and not records:
        records = _read_manifest_records(_triage_root(config))
    if not records:
        return
    terminalreporter.write_sep("=", "AI failure handoff")
    for rec in records:
        terminalreporter.write_line(f"{rec['nodeid']} -> {rec['abs_markdown']}")
    summary = _triage_dir(config) / "triage_summary.md"
    terminalreporter.write_line(f"Summary: {summary}")
    terminalreporter.write_line("Dump for an agent:")
    terminalreporter.write_line("  find logs/pytest/triage -name '*.md'")


def _disabled(config: pytest.Config) -> bool:
    """@brief 判断是否通过 `--no-failure-handoff` 关闭了失败交接。

    @param config pytest 配置对象。
    @return 关闭时返回 True。
    """
    return bool(config.getoption("--no-failure-handoff", default=False))


def _worker_id(config: pytest.Config) -> str:
    """@brief 取当前 xdist worker 名，串行运行时为 master。

    @param config pytest 配置对象。
    @return worker 标识字符串。
    """
    return run_logs.worker_id(config)


def _triage_root(config: pytest.Config) -> Path:
    """@brief 给出发交根目录 logs/pytest/triage 的路径。

    @param config pytest 配置对象。
    @return 仓库根下的 triage 目录路径。
    """
    return Path(config.rootpath) / "logs" / "pytest" / "triage"


def _triage_dir(config: pytest.Config) -> Path:
    """@brief 给出发交写入目录：主进程为 triage 根，worker 为其下的子目录。

    @param config pytest 配置对象。
    @return 当前进程应该写入的 triage 目录。
    """
    root = _triage_root(config)
    worker = _worker_id(config)
    return root if worker == "master" else root / worker


def _read_manifest_records(root: Path) -> list[dict[str, Any]]:
    """@brief 读取各 worker 的 manifest.jsonl，用于主进程汇总。

    无法读取的文件与无法解析的行会被跳过。

    @param root triage 根目录。
    @return 所有 manifest 里解析出的记录字典列表。
    """
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/manifest.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                records.append(row)
    return records


def _git_short(root: Path) -> str:
    """@brief 取仓库当前的短提交哈希，取不到时返回 unknown。

    @param root 仓库根目录。
    @return 短哈希，失败时为 "unknown"。
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=root,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _record_failure(item: pytest.Item, call: pytest.CallInfo[None], report: pytest.TestReport) -> None:
    """@brief 为一个失败的测试建 triage 目录、写 Markdown 与 manifest，并追加到报告分节。

    @param item 失败的测试项。
    @param call 调用阶段信息，提供异常信息。
    @param report 即将输出的测试报告，会追写一段交接路径。
    """
    excinfo = call.excinfo
    eval_report = _find_eval_report(excinfo)
    payload = _payload(item, excinfo, eval_report)
    triage = _triage_dir(item.config)
    try:
        triage.mkdir(parents=True, exist_ok=True)
        md_path = triage / f"{_sanitize_nodeid(item.nodeid)}.md"
        md_path.write_text(_render(item.nodeid, payload), encoding="utf-8")
        record = {
            "worker": payload["worker"],
            "nodeid": item.nodeid,
            "error_kind": payload["error_kind"],
            "exc_type": payload["exception"]["type"],
            "abs_markdown": str(md_path.resolve()),
            "source_file": payload["source"]["file"],
            "lineno": payload["source"]["lineno"],
        }
        with (triage / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        return
    records = item.config.stash.get(_RECORDS, None)
    if records is None:
        records = []
        item.config.stash[_RECORDS] = records
    records.append(record)
    report.sections.append(
        (
            "ai failure handoff",
            f"Markdown triage: {record['abs_markdown']}\n"
            f"(AI block inside Markdown under {BEGIN})\n",
        )
    )


def _payload(item: pytest.Item, excinfo: pytest.ExceptionInfo[BaseException] | None, eval_report: Any) -> dict[str, Any]:
    """@brief 组装交截 JSON 的全部字段：源码位置、标记、参数、反馈与复现命令。

    @param item 失败的测试项。
    @param excinfo 异常信息，可能为 None。
    @param eval_report 从失败帧里找到的 EvalReport，可能为 None。
    @return 可直接序列化的交截字典。
    """
    message = "" if excinfo is None else str(excinfo.value)
    lines = message.splitlines()
    rel, lineno = _source(item, excinfo)
    log_path = run_logs.current_test_log(item)
    log_text = run_logs.log_tail(log_path)
    try:
        log_rel = str(log_path.resolve().relative_to(Path(item.config.rootpath).resolve())) if log_path else ""
    except (OSError, ValueError):
        log_rel = str(log_path) if log_path else ""
    return {
        "schema": SCHEMA,
        "git_short": item.config.stash.get(_GIT, "unknown"),
        "worker": _worker_id(item.config),
        "nodeid": item.nodeid,
        "source": {"file": rel, "lineno": lineno},
        "markers": _markers(item),
        "error_kind": _error_kind(excinfo, eval_report),
        "pytest_short": lines[0] if lines else "",
        "exception": {
            "type": "" if excinfo is None else excinfo.typename,
            "message_lines": lines[:MESSAGE_LINES_MAX],
        },
        "params": _params(item),
        "feedback": _feedback(eval_report),
        "minimal_repro_cmd": _repro(item),
        "per_test_log": {"path": log_rel, "tail": log_text},
        "environment": {key: os.environ.get(key, "") for key in ENV_KEYS},
    }


def _find_eval_report(excinfo: pytest.ExceptionInfo[BaseException] | None) -> Any:
    """@brief 从异常栈帧里反向查找 EvalReport 对象，优先找 rep/report 变量名。

    @param excinfo 异常信息，可能为 None。
    @return 找到的 EvalReport，或 None。
    """
    if excinfo is None:
        return None
    try:
        from tcenv.evaluate import EvalReport
    except Exception:
        return None
    try:
        frames = list(excinfo.traceback)
    except Exception:
        return None
    for entry in reversed(frames):
        try:
            locals_ = entry.frame.f_locals
        except Exception:
            continue
        for name in _PREFERRED_REPORT_NAMES:
            value = locals_.get(name)
            if isinstance(value, EvalReport):
                return value
        for value in locals_.values():
            if isinstance(value, EvalReport):
                return value
    return None


def _error_kind(excinfo: pytest.ExceptionInfo[BaseException] | None, eval_report: Any) -> str:
    """@brief 定下交截的错误类别：优先用 EvalReport 状态，否则用异常类型名。

    @param excinfo 异常信息，可能为 None。
    @param eval_report 失败帧里的 EvalReport，可能为 None。
    @return 错误类别字符串，判定不出时为 "unknown"。
    """
    status = getattr(eval_report, "status", None)
    if isinstance(status, str) and status:
        return status
    if excinfo is None:
        return "unknown"
    if excinfo.typename == "AssertionError":
        return "assertion"
    return excinfo.typename


def _feedback(eval_report: Any) -> str:
    """@brief 用 format_feedback 把 EvalReport 渲染成文字，并截到 FEEDBACK_MAX。

    渲染失败时返回一段说明失败原因的文字，不抛异常。

    @param eval_report 失败帧里的 EvalReport，可能为 None。
    @return 反馈文本；没有报告时为空串。
    """
    if eval_report is None:
        return ""
    try:
        from tcenv.report import format_feedback

        text = format_feedback(eval_report)
    except Exception as exc:
        return f"(format_feedback failed: {type(exc).__name__}: {exc})"
    if len(text) > FEEDBACK_MAX:
        return text[:FEEDBACK_MAX]
    return text


def _source(item: pytest.Item, excinfo: pytest.ExceptionInfo[BaseException] | None) -> tuple[str, int]:
    """@brief 定位失败的源码位置：优先用异常栈里最深且属于测试文件的帧。

    @param item 失败的测试项。
    @param excinfo 异常信息，可能为 None。
    @return (相对仓库根的路径, 1 基行号) 二元组。
    """
    path = Path(item.path)
    root = Path(item.config.rootpath).resolve()
    try:
        rel = str(path.resolve().relative_to(root))
    except ValueError:
        rel = item.nodeid.split("::", 1)[0]
    lineno = 0
    function = getattr(item, "function", None)
    try:
        lineno = int(function.__code__.co_firstlineno)
    except (AttributeError, TypeError, ValueError):
        pass
    if excinfo is None:
        return rel, lineno
    try:
        test_file = str(path.resolve())
        for entry in reversed(list(excinfo.traceback)):
            frame_file = str(Path(entry.frame.raw.f_code.co_filename).resolve())
            if frame_file == test_file:
                return rel, int(entry.lineno) + 1
    except Exception:
        pass
    return rel, lineno


def _markers(item: pytest.Item) -> list[str]:
    """@brief 收集测试项上的标记名，去重并排除 parametrize。

    @param item 测试项。
    @return 排序后的标记名列表。
    """
    names: list[str] = []
    for mark in item.iter_markers():
        if mark.name in _SKIP_MARKERS or mark.name in names:
            continue
        names.append(mark.name)
    return sorted(names)


def _params(item: pytest.Item) -> dict[str, str]:
    """@brief 取出 parametrize 参数并转成受长度限制的字符串表示。

    @param item 测试项。
    @return 参数名到 repr 文本的字典；无参数化时为空。
    """
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return {}
    out: dict[str, str] = {}
    for key, value in callspec.params.items():
        text = repr(value)
        if len(text) > PARAM_REPR_MAX:
            text = text[:PARAM_REPR_MAX] + "…"
        out[str(key)] = text
    return out


def _repro(item: pytest.Item) -> str:
    """@brief 生成可单进程复现该用例的 shell 命令（先 cd 到仓库根）。

    @param item 测试项。
    @return 包含 cd 与 pytest 调用的多行命令。
    """
    root = shlex_quote(str(Path(item.config.rootpath).resolve()))
    nodeid = shlex_quote(item.nodeid)
    return f"cd {root}\nuv run pytest -q --tb=short {nodeid}"


def shlex_quote(text: str) -> str:
    """@brief 把文本包成单个 shell 单词，总是用单引号包裹并转义内部单引号。

    @param text 要引用的原始文本。
    @return 可直接拼进 shell 命令的引号字符串。
    """
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _sanitize_nodeid(nodeid: str) -> str:
    """@brief 把 node id 转成可用作文件名的字符串，并截断到 220 字符。

    @param nodeid pytest 测试节点标识。
    @return 释放了路径分隔符与非法字符的文件名。
    """
    sanitized = nodeid.replace("/", "_").replace("::", "__")
    for char in '[]:*?"<>|':
        sanitized = sanitized.replace(char, "_")
    return sanitized[:220]


def _render(nodeid: str, payload: dict[str, Any]) -> str:
    """@brief 渲染交截 Markdown：复现命令加一个包在四引号围栅里的 JSON 真值块。

    @param nodeid pytest 测试节点标识。
    @param payload 交截字典。
    @return 完整的 Markdown 文本。
    """
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        f"# Failure triage: `{nodeid}`\n\n"
        "## One-shot reproduce\n\n"
        "```bash\n"
        f"{payload['minimal_repro_cmd']}\n"
        "```\n\n"
        "## For AI assistants\n\n"
        "The fenced JSON is the ground truth for this failure.\n\n"
        "````\n"
        f"{BEGIN}\n"
        f"{body}\n"
        f"{END}\n"
        "````\n"
    )


def _write_summary(config: pytest.Config, records: list[dict[str, Any]]) -> Path:
    """@brief 按错误类别分组写出 triage_summary.md。

    @param config pytest 配置对象。
    @param records 全部失败记录。
    @return 写出的汇总文件路径。
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault(str(rec["error_kind"]), []).append(rec)
    lines = ["# Triage summary", "", f"Total failures: {len(records)}", ""]
    for kind, items in groups.items():
        noun = "failure" if len(items) == 1 else "failures"
        lines.append(f"## {kind} — {len(items)} {noun}")
        lines.append("")
        for rec in items:
            lines.append(f"- `{rec['nodeid']}`")
            lines.append(f"  {rec['abs_markdown']}")
        lines.append("")
    path = _triage_dir(config) / "triage_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
