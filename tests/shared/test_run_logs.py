"""Parallel-safe test logs and worker-specific failure triage."""

from __future__ import annotations

from types import SimpleNamespace

from tests.support import handoff, run_logs


def _config(tmp_path, worker: str = "master"):
    """@brief 造一个替代 pytest ``config`` 的命名空间，指定日志目录与 worker 名。

    @param tmp_path 用作 ``rootpath`` 的临时目录，日志路径都相对它拼出。
    @param worker worker 名；``master`` 表示串行运行，其余值放进 ``workerinput``。
    @return 只实现被测代码用到字段的 ``SimpleNamespace``。
    """
    values = {"--log-dir": "logs/pytest", "--log-by-dir": False, "--no-test-file-log": False}
    return SimpleNamespace(
        rootpath=tmp_path,
        workerinput={"workerid": worker} if worker != "master" else {},
        getoption=lambda name, default=None: values.get(name, default),
    )


def test_worker_log_paths_and_triage_are_separated(tmp_path):
    """@brief 验证并行运行时每个测试的日志与失败分诊目录都落在各自的 worker 子目录里。

    @param tmp_path 临时目录，作为 pytest 的 rootpath。
    """
    config = _config(tmp_path, "gw3")
    assert run_logs.worker_id(config) == "gw3"
    assert handoff._triage_dir(config) == tmp_path / "logs" / "pytest" / "triage" / "gw3"
    item = SimpleNamespace(config=config, nodeid="tests/test_x.py::test_y")
    assert run_logs.per_test_log_path(item) == tmp_path / "logs" / "pytest" / "tests" / "gw3" / "tests_test_x.py__test_y.log"


def test_serial_triage_stays_at_the_summary_root(tmp_path):
    """@brief 验证串行运行（没有 workerinput）时分诊目录停在汇总根，不多套一层 worker 目录。

    @param tmp_path 临时目录，作为 pytest 的 rootpath。
    """
    config = _config(tmp_path)
    assert handoff._triage_dir(config) == tmp_path / "logs" / "pytest" / "triage"
