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
    parser.addoption(
        "--no-failure-handoff",
        action="store_true",
        default=False,
        help="Do not write logs/pytest/triage markdown when a test fails",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
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
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed:
        return
    if _disabled(item.config):
        return
    _record_failure(item, call, report)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
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
    return bool(config.getoption("--no-failure-handoff", default=False))


def _worker_id(config: pytest.Config) -> str:
    return run_logs.worker_id(config)


def _triage_root(config: pytest.Config) -> Path:
    return Path(config.rootpath) / "logs" / "pytest" / "triage"


def _triage_dir(config: pytest.Config) -> Path:
    root = _triage_root(config)
    worker = _worker_id(config)
    return root if worker == "master" else root / worker


def _read_manifest_records(root: Path) -> list[dict[str, Any]]:
    """Read worker manifests for the controller's aggregate summary."""
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
    status = getattr(eval_report, "status", None)
    if isinstance(status, str) and status:
        return status
    if excinfo is None:
        return "unknown"
    if excinfo.typename == "AssertionError":
        return "assertion"
    return excinfo.typename


def _feedback(eval_report: Any) -> str:
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
    names: list[str] = []
    for mark in item.iter_markers():
        if mark.name in _SKIP_MARKERS or mark.name in names:
            continue
        names.append(mark.name)
    return sorted(names)


def _params(item: pytest.Item) -> dict[str, str]:
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
    root = shlex_quote(str(Path(item.config.rootpath).resolve()))
    nodeid = shlex_quote(item.nodeid)
    return f"cd {root}\nuv run pytest -q --tb=short {nodeid}"


def shlex_quote(text: str) -> str:
    """Always a single shell word, and always wrapped in single quotes."""
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _sanitize_nodeid(nodeid: str) -> str:
    sanitized = nodeid.replace("/", "_").replace("::", "__")
    for char in '[]:*?"<>|':
        sanitized = sanitized.replace(char, "_")
    return sanitized[:220]


def _render(nodeid: str, payload: dict[str, Any]) -> str:
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
