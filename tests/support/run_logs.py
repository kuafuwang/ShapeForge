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
    """The xdist worker name, or ``master`` in a serial session.

    pytester subprocesses inherit ``PYTEST_XDIST_WORKER`` from the outer run even
    though they do not load xdist themselves. Trust the environment only when
    xdist is actually active in this config.
    """
    if hasattr(config, "workerinput"):
        workerinput = getattr(config, "workerinput", None)
        if isinstance(workerinput, dict) and workerinput.get("workerid"):
            return str(workerinput["workerid"])
    return "master"


def pytest_addoption(parser: pytest.Parser) -> None:
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
    path = Path(config.getoption("--log-dir"))
    return path.resolve() if path.is_absolute() else (Path(config.rootpath) / path).resolve()


def _safe_nodeid(nodeid: str) -> str:
    value = nodeid.replace("/", "_").replace("::", "__")
    for char in '[]:*?"<>|':
        value = value.replace(char, "_")
    return value[:220]


def test_log_path(item: pytest.Item) -> Path:
    """The per-test log path for this worker, whether or not the file exists yet."""
    root = _log_root(item.config) / "tests" / worker_id(item.config)
    if item.config.getoption("--log-by-dir"):
        relative = item.nodeid.replace("::", "/")
        if relative.endswith(".py"):
            relative = relative[:-3]
        return root / f"{relative}.log"
    return root / f"{_safe_nodeid(item.nodeid)}.log"


def current_test_log(item: pytest.Item) -> Path | None:
    """Path of the per-test log if file logging is enabled for this item.

    A pytester subprocess may load only the handoff plugin, so the logging
    option can be absent there.
    """
    try:
        if item.config.getoption("--no-test-file-log"):
            return None
    except ValueError:
        return None
    return test_log_path(item)


def log_tail(path: Path | None, limit: int = _MAX_TAIL) -> str:
    """A bounded UTF-8 tail suitable for an AI handoff block."""
    if path is None:
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max(0, limit):]


def _handler(path: Path) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    return handler


def pytest_sessionstart(session: pytest.Session) -> None:
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
    if item.config.getoption("--no-test-file-log"):
        return
    path = test_log_path(item)
    handler = _handler(path)
    _BASE.addHandler(handler)
    _TEST_HANDLERS[item.nodeid] = handler
    _TEST_START[item.nodeid] = time.monotonic()
    _BASE.info("========== begin %s ==========", item.nodeid)


def pytest_runtest_teardown(item: pytest.Item) -> None:
    handler = _TEST_HANDLERS.pop(item.nodeid, None)
    started = _TEST_START.pop(item.nodeid, None)
    if handler is not None:
        elapsed = "" if started is None else f" duration_s={time.monotonic() - started:.3f}"
        _BASE.info("========== end %s%s ==========", item.nodeid, elapsed)
        _BASE.removeHandler(handler)
        handler.close()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Generator[None, None, None]:
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
    _BASE.info("pytest session finish (worker=%s, exit=%s)", worker_id(session.config), exitstatus)
    handler = session.config.stash.get(_SESSION_HANDLER, None)
    if handler is not None:
        _BASE.removeHandler(handler)
        handler.close()
        del session.config.stash[_SESSION_HANDLER]
