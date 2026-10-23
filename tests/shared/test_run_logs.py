"""Parallel-safe test logs and worker-specific failure triage."""

from __future__ import annotations

from types import SimpleNamespace

from tests.support import handoff, run_logs


def _config(tmp_path, worker: str = "master"):
    values = {"--log-dir": "logs/pytest", "--log-by-dir": False, "--no-test-file-log": False}
    return SimpleNamespace(
        rootpath=tmp_path,
        workerinput={"workerid": worker} if worker != "master" else {},
        getoption=lambda name, default=None: values.get(name, default),
    )


def test_worker_log_paths_and_triage_are_separated(tmp_path):
    config = _config(tmp_path, "gw3")
    assert run_logs.worker_id(config) == "gw3"
    assert handoff._triage_dir(config) == tmp_path / "logs" / "pytest" / "triage" / "gw3"
    item = SimpleNamespace(config=config, nodeid="tests/test_x.py::test_y")
    assert run_logs.per_test_log_path(item) == tmp_path / "logs" / "pytest" / "tests" / "gw3" / "tests_test_x.py__test_y.log"


def test_serial_triage_stays_at_the_summary_root(tmp_path):
    config = _config(tmp_path)
    assert handoff._triage_dir(config) == tmp_path / "logs" / "pytest" / "triage"
