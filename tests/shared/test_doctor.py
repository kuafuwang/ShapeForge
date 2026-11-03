"""Doctor is the one-command environment check: every failure carries the single next action."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

DOCTOR = Path(__file__).resolve().parents[2] / "scripts" / "doctor.py"
STATUSES = {"pass", "warn", "fail"}


def _load():
    """@brief 从 ``scripts/doctor.py`` 取出模块对象；``scripts/`` 不是包，只能按路径加载。

    @return 已初始化的 doctor 模块。
    """
    spec = importlib.util.spec_from_file_location("shapeforge_doctor", DOCTOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_check_reports_a_stable_id_and_an_actionable_next_step():
    """@brief 每条检查都有唯一 id 和一行实际观察；非 PASS 必须给且只给一条可执行下一步。

    这是 doctor 能被 agent 直接消费的前提：没有 ``next`` 的 FAIL 等于把问题丢回给人。
    """
    doctor = _load()
    results = doctor.run_checks()
    ids = [item["id"] for item in results]
    assert len(ids) == len(set(ids)), f"duplicate check ids: {ids}"
    for item in results:
        assert item["status"] in STATUSES, item
        assert item["detail"], f"{item['id']} 没有说明实际看到什么"
        if item["status"] == "pass":
            assert item["next"] == "", f"{item['id']} 通过时不该给下一步"
        else:
            assert item["next"], f"{item['id']} 是 {item['status']} 却没有下一步"
            assert "\n" not in item["next"], f"{item['id']} 的下一步应是一行命令"


def test_a_missing_checkout_fails_the_layout_check_without_raising(tmp_path, capsys):
    """@brief 指到一个空目录：布局检查 FAIL 并指向真正的修法，其余检查不抛异常。

    @param tmp_path pytest 提供的空目录。
    @param capsys 捕获 stdout，用来读 ``--json`` 输出。
    """
    doctor = _load()
    code = doctor.main(["--root", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1 and payload["ok"] is False
    layout = next(item for item in payload["results"] if item["id"] == "repo.layout")
    assert layout["status"] == "fail" and "missing" in layout["detail"]
    assert "repository root" in layout["next"]
    assert all(item["status"] in STATUSES for item in payload["results"])


def test_a_healthy_checkout_exits_zero_and_the_summary_counts_agree(capsys):
    """@brief 本仓库现状：没有 FAIL 时退出码为 0，人读摘要的计数与结果一致。

    @param capsys 捕获 stdout。
    """
    doctor = _load()
    results = doctor.run_checks(port=1)
    failed = [item for item in results if item["status"] == "fail"]
    warned = [item for item in results if item["status"] == "warn"]
    code = doctor.main(["--port", "1"])
    out = capsys.readouterr().out
    assert code == (1 if failed else 0)
    assert f"{len(results) - len(failed) - len(warned)} passed, {len(warned)} warnings, {len(failed)} failed" in out
    if failed or warned:
        assert "Next:" in out


def test_the_dashboard_probe_reports_the_configured_port_rather_than_hardcoding_it():
    """@brief 端口可配置：一个必然没人监听的端口只该 WARN，不该 FAIL，也不该假装成功。

    @return 无。
    """
    doctor = _load()
    closed = next(item for item in doctor.run_checks(port=1) if item["id"] == "dashboard.http")
    assert closed["status"] == "warn" and "127.0.0.1:1" in closed["detail"]
    assert "uv run python -m observe" in closed["next"]
