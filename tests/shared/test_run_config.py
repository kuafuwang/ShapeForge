"""Versioned run_config.json: hashes, no credentials, and a new run that cannot start does not call the model. No GPU."""

from __future__ import annotations

import hashlib
import json

import pytest

from experiments.common import run_one
from taic import cli, control
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator, RunStartError
from tasks import get_task

from ..support.fakes import FakeClient


def _cfg() -> tuple[LLMConfig, SearchConfig]:
    """@brief 构造离线测试用的 LLMConfig 和最小 SearchConfig。

    @return (LLMConfig, SearchConfig) 元组。
    """
    cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
    search = SearchConfig(budget_usd=1.0, branches=1, max_rounds=1, eval_reps=3, stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False)
    return cfg, search


def _start(tmp_path, client, **kw):
    """@brief 在 tmp_path/run 下用 fake client 构造一个 Orchestrator。

    @param tmp_path pytest 提供的临时目录。
    @param client fake LLM 客户端。
    @param kw 透传给 Orchestrator 的额外参数（例如 resuming）。
    @return 构造好的 Orchestrator。
    """
    cfg, search = _cfg()
    return Orchestrator("relu", "fp16", "small", cfg, search, run_dir=tmp_path / "run", client=client, log=lambda _s: None, **kw)


def test_v1_config_records_two_hashes_and_no_credentials(tmp_path):
    """@brief run_config.json（schema 2）记录契约和 Triton 两个哈希与任务指纹，不含密钥，也不发请求。

    @param tmp_path pytest 提供的临时目录。
    """
    client = FakeClient([])
    orch = _start(tmp_path, client)
    text = (orch.run_dir / "run_config.json").read_text()
    assert "sk-test" not in text and "api_key" not in text and "http://x" not in text
    doc = json.loads(text)
    task = get_task("relu", "fp16", "small")
    assert doc["schema_version"] == 2
    assert doc["target"]["id"] == "apple-m4-max"
    assert doc["target"]["backend_version"] == "1"
    assert doc["contract_description_sha256"] == hashlib.sha256(task.contract.describe().encode("utf-8")).hexdigest()
    assert doc["triton_source_sha256"] == hashlib.sha256(task.triton_source.encode("utf-8")).hexdigest()
    from tcenv.task import task_fingerprint

    assert "reference" in doc["hash_scope"] and "baseline" in doc["hash_scope"]
    assert doc["task_fingerprint"] == task_fingerprint(task)
    assert doc["cost_may_be_incomplete"] is True
    assert doc["git_commit"] == "unknown" or len(doc["git_commit"]) >= 7
    assert doc["git_dirty"] in (True, False, None)
    assert doc["settings_source"]
    assert client.calls == []
    assert list(orch.run_dir.glob("run_config.json")) == [orch.run_dir / "run_config.json"]


def test_a_failed_config_write_sends_no_request_and_leaves_no_result(tmp_path, monkeypatch):
    """@brief run_config 写盘失败时抛 RunStartError，不留下任何 run 文件也不发请求。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 补丁工具，用于让原子写入失败。
    """
    def boom(_path, _doc):
        raise OSError("disk full")

    monkeypatch.setattr(control, "write_json_atomic", boom)
    client = FakeClient([])
    with pytest.raises(RunStartError, match="run_config.json"):
        _start(tmp_path, client)
    run = tmp_path / "run"
    assert run.is_dir()
    assert not (run / "run_config.json").exists()
    assert not (run / "state.json").exists()
    assert not (run / "result.json").exists()
    assert not (run / "transcript.jsonl").exists()
    assert client.calls == []


def test_a_nonempty_directory_is_refused_without_touching_it(tmp_path):
    """@brief 非空目录被拒绝（提示 resume），原有文件不被修改。

    @param tmp_path pytest 提供的临时目录。
    """
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_config.json").write_text('{"keep": true, "schema_version": 0}')
    (run / "transcript.jsonl").write_text("old\n")
    client = FakeClient([])
    with pytest.raises(RunStartError, match="resume"):
        _start(tmp_path, client)
    assert json.loads((run / "run_config.json").read_text()) == {"keep": True, "schema_version": 0}
    assert (run / "transcript.jsonl").read_text() == "old\n"
    assert not (run / "state.json").exists()
    assert client.calls == []


def test_resuming_does_not_rewrite_the_existing_config(tmp_path):
    """@brief resuming=True 时不重写已有的 run_config.json。

    @param tmp_path pytest 提供的临时目录。
    """
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({"schema_version": 1, "note": "old"}))
    _start(tmp_path, FakeClient([]), resuming=True)
    assert json.loads((run / "run_config.json").read_text())["note"] == "old"
    assert not (run / "state.json").exists()


def test_taic_run_exits_5_when_branches_is_not_one(tmp_path, monkeypatch):
    """@brief branches 不为 1 时 taic run 返回退出码 5。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 补丁工具，用于注入固定配置。
    """
    cfg, search = _cfg()
    search.branches = 2
    monkeypatch.setattr("taic.llm.load_config", lambda *_a, **_k: (cfg, search))
    code = cli.main(["run", "relu", "--size", "small", "--preset", "base", "--branches", "2", "--run-dir", str(tmp_path / "run")])
    assert code == 5


def test_the_run_lock_alone_does_not_count_as_files_in_a_new_directory(tmp_path, monkeypatch):
    """@brief 目录里只有 run 锁文件时仍算新目录，orchestrator 可以启动。

    `taic run --run-dir X` 会在构造 orchestrator 之前先占住目录锁；
    把锁文件当成 run 内容会让每个显式指定的新目录拒绝自己。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 补丁工具，用于注入固定配置。
    """
    cfg, search = _cfg()
    monkeypatch.setattr("taic.llm.load_config", lambda *_a, **_k: (cfg, search))
    run = tmp_path / "claimed"
    control.acquire_run_lock(run, who="run")
    try:
        orch = Orchestrator("relu", "fp16", "small", cfg, search, run_dir=run, client=FakeClient([]), log=lambda _s: None)
        assert orch.run_dir == run and (run / "run_config.json").is_file()
    finally:
        control.release_run_lock(run)


def test_taic_run_exits_6_when_a_valid_kernel_was_found_but_the_confirmation_failed(tmp_path, monkeypatch):
    """@brief 找到有效 kernel 但确认失败时退出码为 6，而不是被视为完全成功。

    文件会正确写入，但发布数字是 null；只读退出码的脚本不能把它当作成功。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 补丁工具，用于替换配置和 Orchestrator。
    """
    from types import SimpleNamespace

    cfg, search = _cfg()
    monkeypatch.setattr("taic.llm.load_config", lambda *_a, **_k: (cfg, search))

    class Stub:
        def __init__(self, *a, **k):
            pass

        async def run(self):
            return SimpleNamespace(
                task="relu", dtype="fp16", model="fake", best_speedup=1.2, confirmed_speedup=None,
                spent_usd=0.1, stop_reason="confirm_error: SchemaError: boom", run_dir=str(tmp_path),
            )

    monkeypatch.setattr("taic.orchestrator.Orchestrator", Stub)
    code = cli.main(["run", "relu", "--size", "small", "--preset", "base", "--no-tools", "--run-dir", str(tmp_path / "run")])
    assert code == 6


def test_taic_run_exits_5_when_the_directory_already_holds_a_run(tmp_path, monkeypatch):
    """@brief 目录已包含 run 内容时退出码 5，原有文件保持不变。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 补丁工具，用于注入固定配置。
    """
    cfg, search = _cfg()
    monkeypatch.setattr("taic.llm.load_config", lambda *_a, **_k: (cfg, search))
    run = tmp_path / "taken"
    run.mkdir()
    (run / "keep.txt").write_text("stay")
    code = cli.main(["run", "relu", "--size", "small", "--preset", "base", "--no-tools", "--run-dir", str(run)])
    assert code == 5
    assert (run / "keep.txt").read_text() == "stay"
    assert not (run / "state.json").exists() and not (run / "result.json").exists()


def test_run_one_propagates_a_refused_directory_and_does_not_start(tmp_path, monkeypatch):
    """@brief run_one 把拒绝目录的 RunStartError 透传出去，不启动也不发请求。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 补丁工具，用于注入固定配置。
    """
    cfg, search = _cfg()
    monkeypatch.setattr("experiments.common.load_config", lambda *_a, **_k: (cfg, search))
    run = tmp_path / "run"
    run.mkdir()
    (run / "keep.txt").write_text("stay")
    client = FakeClient([])
    with pytest.raises(RunStartError):
        run_one("relu", "fp16", "small", run, client=client, log=lambda _s: None)
    assert client.calls == []
    assert (run / "keep.txt").read_text() == "stay"
    assert not (run / "result.json").exists()


def test_a_budget_event_that_fails_to_land_is_not_a_complete_record(tmp_path, monkeypatch):
    """@brief 预算事件写盘失败时标记 persisted=False，且不留下事件文件。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 补丁工具，用于让写入失败。
    """
    def boom(_path, _doc):
        raise OSError("nope")

    monkeypatch.setattr(control, "write_json_atomic", boom)
    event = control.record_event(tmp_path, "applied", 1.0, 2.0)
    assert event["persisted"] is False
    assert not (tmp_path / "budget_events.json").exists()


def test_taic_run_exits_4_when_another_process_holds_the_run_lock(tmp_path, monkeypatch):
    """@brief 其他进程持有 run 锁时退出码 4，目录里始终只有锁文件。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 补丁工具，用于注入固定配置。
    """
    import subprocess
    import time

    cfg, search = _cfg()
    monkeypatch.setattr("taic.llm.load_config", lambda *_a, **_k: (cfg, search))
    run = tmp_path / "run"
    run.mkdir()
    proc = subprocess.Popen(["sleep", "20"])
    try:
        (run / control.LOCK_FILE).write_text(json.dumps({"pid": proc.pid, "started_at": time.time(), "who": "run"}))
        assert cli.main(["run", "relu", "--size", "small", "--preset", "base", "--no-tools", "--run-dir", str(run)]) == 4
        assert [p.name for p in run.iterdir()] == [control.LOCK_FILE]
    finally:
        proc.kill()
        proc.wait()

