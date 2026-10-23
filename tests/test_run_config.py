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

from .support.fakes import FakeClient


def _cfg() -> tuple[LLMConfig, SearchConfig]:
    cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
    search = SearchConfig(budget_usd=1.0, branches=1, max_rounds=1, eval_reps=3, stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False)
    return cfg, search


def _start(tmp_path, client, **kw):
    cfg, search = _cfg()
    return Orchestrator("relu", "fp16", "small", cfg, search, run_dir=tmp_path / "run", client=client, log=lambda _s: None, **kw)


def test_v1_config_records_two_hashes_and_no_credentials(tmp_path):
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
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({"schema_version": 1, "note": "old"}))
    _start(tmp_path, FakeClient([]), resuming=True)
    assert json.loads((run / "run_config.json").read_text())["note"] == "old"
    assert not (run / "state.json").exists()


def test_taic_run_exits_5_when_branches_is_not_one(tmp_path, monkeypatch):
    cfg, search = _cfg()
    search.branches = 2
    monkeypatch.setattr("taic.llm.load_config", lambda *_a, **_k: (cfg, search))
    code = cli.main(["run", "relu", "--size", "small", "--preset", "base", "--branches", "2", "--run-dir", str(tmp_path / "run")])
    assert code == 5


def test_the_run_lock_alone_does_not_count_as_files_in_a_new_directory(tmp_path, monkeypatch):
    """`taic run --run-dir X` claims the directory lock before constructing the orchestrator.
    Counting that lock as run content made every explicit new directory refuse itself."""
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
    """The files are written (correct), but the published number is null. A script that only reads
    the exit status must not treat that as a full success."""
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
    def boom(_path, _doc):
        raise OSError("nope")

    monkeypatch.setattr(control, "write_json_atomic", boom)
    event = control.record_event(tmp_path, "applied", 1.0, 2.0)
    assert event["persisted"] is False
    assert not (tmp_path / "budget_events.json").exists()
