"""Per-worker pytest logs, safe when pytest-xdist runs several processes.

Each worker writes its own session log and per-test log, so parallel tests never
append to the same file. A failed test gets its log excerpt attached to the
pytest report; ``tests.support.handoff`` also puts a capped tail in the JSON
block written for an AI assistant.

    uv run pytest -q                  # one log set per xdist worker
    uv run pytest -q -n 0             # serial: master logs
    uv run pytest -q --no-test-file-log

The logger is ``ai_lowerling.tests``. Application code that already logs through
stdlib logging is captured automatically; this plugin does not change Loguru.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Generator

import pytest

_BASE = logging.getLogger("ai_lowerling.tests")
_LOG_HANDLERS: dict[str, logging.Handler] = {}
_TEST_HANDLERS: dict[str, logging.Handler] = {}
_TEST_START: dict[str, float] = {}
_SESSION_HANDLER: pytest.StashKey[logging.Handler] = pytest.StashKey()
_MAX_TAIL = 8_000


def worker_id(config: pytest.Config) -> str:
    """@brief 返回本次会话的 xdist worker 名，串行会话返回 master。

    pytester 子进程会从外层运行继承 PYTEST_XDIST_WORKER，但这些子进程本身没加载 xdist，
    所以只有当前 config 真正启用 xdist 时才相信 worker 身份。

    @param config pytest 的 config 对象。
    @return worker 名，或 master。
    """
    if hasattr(config, "workerinput"):
        workerinput = getattr(config, "workerinput", None)
        if isinstance(workerinput, dict) and workerinput.get("workerid"):
            return str(workerinput["workerid"])
    return "master"


def pytest_addoption(parser: pytest.Parser) -> None:
    """@brief 注册日志相关命令行选项：日志根目录、按目录镜像、关闭单测日志文件。

    @param parser pytest 的选项解析器。
    """
    parser.addoption(
        "--log-dir",
        action="store",
        default="logs/pytest",
        help="Root for session and per-test logs (relative to pytest's cwd).",
    )
    parser.addoption(
        "--log-by-dir",
        action="store_true",
        default=False,
        help="Mirror the test node id under the log directory instead of one flat file per test.",
    )
    parser.addoption(
        "--no-test-file-log",
        action="store_true",
        default=False,
        help="Disable per-test .log files; session logs and failure excerpts remain enabled.",
    )


def _log_root(config: pytest.Config) -> Path:
    """@brief 把 --log-dir 解析成绝对路径，相对路径以 rootpath 为基准。

    @param config pytest 的 config 对象。
    @return 已解析的日志根目录。
    """
    path = Path(config.getoption("--log-dir"))
    return path.resolve() if path.is_absolute() else (Path(config.rootpath) / path).resolve()


def _safe_nodeid(nodeid: str) -> str:
    """@brief 把 node id 的路径分隔符与文件名非法字符换成下划线，并限长 220 字符。

    @param nodeid 测试的 node id。
    @return 可用作单层文件名的字符串。
    """
    value = nodeid.replace("/", "_").replace("::", "__")
    for char in '[]:*?"<>|':
        value = value.replace(char, "_")
    return value[:220]


def per_test_log_path(item: pytest.Item) -> Path:
    """@brief 计算当前 worker 下单测日志的路径，无论文件是否已存在。"""
    root = _log_root(item.config) / "tests" / worker_id(item.config)
    if item.config.getoption("--log-by-dir"):
        relative = item.nodeid.replace("::", "/")
        if relative.endswith(".py"):
            relative = relative[:-3]
        return root / f"{relative}.log"
    return root / f"{_safe_nodeid(item.nodeid)}.log"


def current_test_log(item: pytest.Item) -> Path | None:
    """@brief 返回该测试的单测日志路径，若单测文件日志被关闭则返回 None。

    pytester 子进程可能只加载 handoff 插件，因此那里可能拿不到日志选项。
    """
    try:
        if item.config.getoption("--no-test-file-log"):
            return None
    except ValueError:
        return None
    return per_test_log_path(item)


def log_tail(path: Path | None, limit: int = _MAX_TAIL) -> str:
    """@brief 读取日志尾部，长度受限，适合放进给 AI 的交接块。

    @param path 日志路径，None 表示没有日志。
    @param limit 最多保留的字符数。
    @return 尾部文本；读不到文件时返回空串。
    """
    if path is None:
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max(0, limit):]


def _handler(path: Path) -> logging.Handler:
    """@brief 按需建父目录并新建一个 DEBUG 级的 UTF-8 文件日志句柄。

    @param path 日志文件路径。
    @return 已配好格式的 FileHandler。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    return handler


def pytest_sessionstart(session: pytest.Session) -> None:
    """@brief 会话开始时建立日志根目录，并把会话日志句柄挂到测试 logger 上（仅一次）。

    @param session pytest 的 session 对象。
    """
    root = _log_root(session.config)
    root.mkdir(parents=True, exist_ok=True)
    if _SESSION_HANDLER in session.config.stash:
        return
    path = root / f"session-{worker_id(session.config)}.log"
    handler = _handler(path)
    _BASE.setLevel(logging.DEBUG)
    _BASE.addHandler(handler)
    session.config.stash[_SESSION_HANDLER] = handler
    _BASE.info("pytest session start (worker=%s, root=%s)", worker_id(session.config), root)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """@brief 每个测试开始时挂上单测日志句柄并记下开始时间；关闭文件日志时什么也不做。

    @param item 当前测试项。
    """
    if item.config.getoption("--no-test-file-log"):
        return
    path = per_test_log_path(item)
    handler = _handler(path)
    _BASE.addHandler(handler)
    _TEST_HANDLERS[item.nodeid] = handler
    _TEST_START[item.nodeid] = time.monotonic()
    _BASE.info("========== begin %s ==========", item.nodeid)


def pytest_runtest_teardown(item: pytest.Item) -> None:
    """@brief 测试结束时写下带耗时的结束分隔条，并摘下和关闭该测试的日志句柄。

    @param item 当前测试项。
    """
    handler = _TEST_HANDLERS.pop(item.nodeid, None)
    started = _TEST_START.pop(item.nodeid, None)
    if handler is not None:
        elapsed = "" if started is None else f" duration_s={time.monotonic() - started:.3f}"
        _BASE.info("========== end %s%s ==========", item.nodeid, elapsed)
        _BASE.removeHandler(handler)
        handler.close()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Generator[None, None, None]:
    """@brief 测试调用阶段失败时，把该测试日志的受限尾部附到报告 sections 里。

    @param item 当前测试项。
    @param call 当前阶段的调用信息。
    """
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed:
        return
    path = current_test_log(item)
    if path is None:
        return
    tail = log_tail(path)
    if tail:
        report.sections.append(("per-test log", tail))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """@brief 会话结束时记一行收尾日志，并摘下和关闭会话日志句柄。

    @param session pytest 的 session 对象。
    @param exitstatus pytest 的退出码。
    """
    _BASE.info("pytest session finish (worker=%s, exit=%s)", worker_id(session.config), exitstatus)
    handler = session.config.stash.get(_SESSION_HANDLER, None)
    if handler is not None:
        _BASE.removeHandler(handler)
        handler.close()
        del session.config.stash[_SESSION_HANDLER]
