"""Target profiles: an active chip loads, a reserved profile does not run. No GPU."""

from __future__ import annotations

import pytest

from tcenv.target import TargetError, calibration_file, load_target, require_active


def test_a_device_that_reports_other_limits_is_named_and_the_profile_stays():
    from tcenv.target import limit_notes

    profile = load_target("apple-m4-max")
    assert limit_notes(profile, {"max_threads_per_threadgroup": 1024, "max_threadgroup_memory_bytes": 32768}) == []
    notes = limit_notes(profile, {"max_threads_per_threadgroup": 512, "max_threadgroup_memory_bytes": 16384})
    assert len(notes) == 2
    assert "512" in notes[0] and "1024" in notes[0]
    assert "16384" in notes[1] and "32768" in notes[1]


def test_feedback_reads_calibration_peaks():
    from observe.feedback import _calibration

    peaks = _calibration()
    assert peaks is not None
    assert peaks["fp16"]
    assert peaks["read_gbps"]
    assert peaks["copy_gbps"]


def test_the_current_machine_loads_and_points_at_metal():
    profile = require_active("apple-m4-max")
    assert profile.backend == "metal"
    assert profile.limits.max_threadgroup_memory_bytes == 32768
    assert profile.device_doc.is_file()
    assert profile.experience.is_file()
    assert profile.skills.name == "skills"
    path, source = calibration_file("apple-m4-max")
    assert source == "target"
    assert path.name == "calibration.json"


def test_a_reserved_profile_loads_but_cannot_run():
    profile = load_target("_template")
    assert profile.status == "reserved"
    with pytest.raises(TargetError):
        require_active("_template")


def test_an_unknown_target_is_refused():
    with pytest.raises(TargetError):
        load_target("no-such-chip")


def test_device_match_rejects_a_different_chip():
    from tcenv.target import device_matches

    profile = load_target("apple-m4-max")
    assert device_matches(profile, {"name": "Apple M4 Max", "gpu_family": "Apple9"})
    assert not device_matches(profile, {"name": "Apple M3 Max", "gpu_family": "Apple8"})


def test_catalog_keeps_the_faster_kernel_and_records_the_slower_one(tmp_path, monkeypatch):
    import taic.kernel_catalog as catalog

    monkeypatch.setattr(catalog, "REPO", tmp_path)
    first = catalog.archive_confirmed(
        target_id="apple-m4-max", backend="metal", task="relu", dtype="fp16", size="small",
        source="kernel-a", speedup=1.2, run_id="run-a", label="r1",
    )
    second = catalog.archive_confirmed(
        target_id="apple-m4-max", backend="metal", task="relu", dtype="fp16", size="small",
        source="kernel-b", speedup=1.1, run_id="run-b", label="r2",
    )
    assert first == second
    assert first.read_text() == "kernel-a"
    history = (first.parent / "history.jsonl").read_text().strip().splitlines()
    assert len(history) == 2
    catalog.archive_confirmed(
        target_id="apple-m4-max", backend="metal", task="relu", dtype="fp16", size="small",
        source="kernel-c", speedup=1.4, run_id="run-c", label="r3",
    )
    assert first.read_text() == "kernel-c"


def test_migrate_moves_a_flat_run_and_stamps_the_target(tmp_path):
    from taic.migrate import migrate_runs

    run = tmp_path / "20260101_relu"
    run.mkdir()
    (run / "state.json").write_text('{"task": "relu"}', encoding="utf-8")
    moved = migrate_runs(tmp_path, "apple-m4-max")
    assert moved == ["20260101_relu"]
    stamped = tmp_path / "apple-m4-max" / "20260101_relu" / "state.json"
    assert '"target_id": "apple-m4-max"' in stamped.read_text()


def test_catalog_seed_is_read_back_and_a_foreign_run_is_refused(tmp_path, monkeypatch):
    import taic.kernel_catalog as catalog
    from taic.llm import LLMConfig, SearchConfig
    from taic.orchestrator import Orchestrator

    from .support.fakes import FakeClient

    monkeypatch.setattr(catalog, "REPO", tmp_path)
    folder = catalog.catalog_dir("apple-m4-max", "relu", "fp16", "small")
    folder.mkdir(parents=True)
    (folder / "best.metal").write_text("ARCHIVED", encoding="utf-8")
    cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
    search = SearchConfig(budget_usd=1.0, branches=1, max_rounds=1, eval_reps=3, stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False)
    found = Orchestrator(
        "relu", "fp16", "small", cfg, search, run_dir=tmp_path / "run", client=FakeClient([]),
        log=lambda _line: None, from_catalog=True, evaluate_fn=lambda *args, **kwargs: None,
    )
    assert found._prior_kernel_text() == "ARCHIVED"
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "best.metal").write_text("FOREIGN", encoding="utf-8")
    (foreign / "run_config.json").write_text('{"target": {"id": "other-chip"}}', encoding="utf-8")
    refused = Orchestrator(
        "relu", "fp16", "small", cfg, search, run_dir=tmp_path / "other-run", client=FakeClient([]),
        log=lambda _line: None, from_run=foreign, evaluate_fn=lambda *args, **kwargs: None,
    )
    assert refused._prior_kernel_text() is None

