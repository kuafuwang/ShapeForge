"""A batch cell that was interrupted resumes instead of refusing the directory."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.common import run_one
from experiments.manifest import backfill, load

from .fakes import FakeClient
from .test_orchestrator import agent_from_script, code, fake_eval, llm_script

SECRET = "sk-test-0123456789abcdef"
ENDPOINT = "http://127.0.0.1:1"


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "taic.toml"
    path.write_text(
        '[llm]\ncredentials_file = "/no/such/models.json"\nprovider = "deepseek"\nmodel = "deepseek-flash"\n'
        'use_tools = false\nreasoning_effort = "off"\n'
        "[search]\nbudget_usd = 5\nbranches = 1\nmax_rounds = 0\neval_reps = 3\n"
        "stop_at_roofline = false\nabi_guard = false\ntarget_note = false\nrich_history = false\nstatus_lines = false\n"
        '[[committees]]\nname = "first"\n[[committees.members]]\nprovider = "deepseek"\nmodel = "model-a"\n'
        '[[committees]]\nname = "second"\n[[committees.members]]\nprovider = "deepseek"\nmodel = "model-b"\n',
        encoding="utf-8",
    )
    return path


def test_an_interrupted_cell_is_resumed_not_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("OPENAI_BASE_URL", ENDPOINT)
    config = _config(tmp_path)
    script = llm_script(code("speed=1.0"))
    run = tmp_path / "cell"
    run_one(
        "relu", "fp16", "small", run, config_path=str(config), preset="base",
        client=FakeClient(script), evaluate_fn=fake_eval([]), agent_runner=agent_from_script(script),
        log=lambda _s: None,
    )
    assert (run / "result.json").is_file()
    (run / "result.json").unlink()
    (run / "confirm_report.json").unlink(missing_ok=True)

    def must_not_start(*_a, **_k):
        raise AssertionError("a resumed run with no rounds left must not start a session")

    result = run_one(
        "relu", "fp16", "small", run, config_path=str(config), preset="base",
        client=FakeClient([]), evaluate_fn=fake_eval([]), agent_runner=must_not_start,
        log=lambda _s: None,
    )
    assert (run / "result.json").is_file()
    assert result.stop_reason == "max_rounds"


def test_backfill_records_arms_and_attaches_the_existing_directories(tmp_path):
    root = tmp_path / "p1"
    for name, task in (("a", "relu"), ("b", "gemm")):
        child = root / name
        child.mkdir(parents=True)
        (child / "run_config.json").write_text(json.dumps({"task": task, "dtype": "fp16", "size": "small"}))
    doc = backfill(root, kind="suite", driver="p1_runs")
    assert [arm["arm_id"] for arm in doc["arms"]] == ["a", "b"]
    assert [arm["runs"] for arm in doc["arms"]] == [["a"], ["b"]]
    assert load(root)["frozen"] == {"backfilled": True}
    assert doc["arms"][0]["delta"] == {"task": "relu", "dtype": "fp16", "size": "small"}
