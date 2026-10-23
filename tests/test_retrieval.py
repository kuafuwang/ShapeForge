"""The CLI query surface and the data/deleted TTL, without a browser or a GPU."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from taic import cli, gc, retrieval


def _run(runs: Path, name: str, task: str) -> None:
    directory = runs / name
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": task, "dtype": "fp16", "size": "small", "phase": "done", "started_at": 10, "updated_at": 10,
    }))
    (directory / "result.json").write_text(json.dumps({
        "task": task, "dtype": "fp16", "size": "small", "stop_reason": "max_rounds", "started_at": 10,
    }))


def _candidate(runs: Path, name: str, label: str, status: str) -> None:
    folder = runs / name / "candidates" / f"000_{label}"
    folder.mkdir(parents=True)
    (folder / "kernel.metal").write_text("kernel void k() {}\n")
    (folder / "report.json").write_text(json.dumps({"status": status, "stage": "abi", "speedup": None}))


def test_query_reads_the_index_and_prints_one_row_per_line(tmp_path, monkeypatch, capsys):
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "books.sqlite"))
    _run(runs, "relu1", "relu")
    _run(runs, "gemm1", "gemm")
    _candidate(runs, "relu1", "r1.p0", "abi_error")
    from observe import query

    query.refresh()
    assert retrieval.capabilities()["schema_version"] == "1"
    page = retrieval.runs(task="relu")
    assert [row["run_id"] for row in page["runs"]] == ["relu1"]
    assert retrieval.failures(status="abi_error")["failures"][0]["run_id"] == "relu1"

    assert cli.main(["query", "runs", "--task", "relu", "--jsonl"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["run_id"] for line in lines] == ["relu1"]


def test_gc_scans_and_applies_only_the_eligible_directories(tmp_path):
    root = tmp_path / "deleted"
    old = root / "old"
    new = root / "new"
    old.mkdir(parents=True)
    new.mkdir()
    (old / "x").write_text("x" * 10)
    now = time.time()
    os.utime(old, (now - 40 * 86400, now - 40 * 86400))
    os.utime(new, (now - 1 * 86400, now - 1 * 86400))
    rows = gc.scan(root, older_than_days=30, now=now)
    by_name = {Path(row["path"]).name: row for row in rows}
    assert by_name["old"]["eligible"] is True and by_name["new"]["eligible"] is False
    result = gc.apply(rows, log_path=root / "gc_log.jsonl", now=now)
    assert result["removed"] == [str(old)]
    assert not old.exists() and new.exists()
    assert json.loads((root / "gc_log.jsonl").read_text().splitlines()[0])["path"] == str(old)


def test_gc_cli_defaults_to_a_report(tmp_path, capsys):
    root = tmp_path / "deleted"
    (root / "old").mkdir(parents=True)
    os.utime(root / "old", (time.time() - 40 * 86400,) * 2)
    assert cli.main(["gc", "--root", str(root), "--older-than-days", "30"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "dry-run" and report["eligible"] == 1
    assert (root / "old").exists()


def test_gc_refuses_a_root_outside_the_archive_and_apply_without_confirm(tmp_path, capsys):
    repo_like = tmp_path / "repo"
    (repo_like / "taic").mkdir(parents=True)
    assert cli.main(["gc", "--root", str(repo_like), "--older-than-days", "30", "--apply", "--confirm"]) == 5
    assert cli.main(["gc", "--root", str(tmp_path / "deleted"), "--older-than-days", "0"]) == 5
    assert cli.main(["gc", "--root", str(tmp_path / "deleted"), "--older-than-days", "30", "--apply"]) == 5
    assert (repo_like / "taic").is_dir()
