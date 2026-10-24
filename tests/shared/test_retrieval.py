"""The CLI query surface and the data/deleted TTL, without a browser or a GPU."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from taic import cli, gc, retrieval


def _run(runs: Path, name: str, task: str) -> None:
    """@brief 在 runs 目录下造一个已结束的假 run，写出 state.json 与 result.json。

    @param runs runs 根目录。
    @param name run 目录名，即 run id。
    @param task 该 run 的任务名。
    """
    directory = runs / name
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": task, "dtype": "fp16", "size": "small", "phase": "done", "started_at": 10, "updated_at": 10,
    }))
    (directory / "result.json").write_text(json.dumps({
        "task": task, "dtype": "fp16", "size": "small", "stop_reason": "max_rounds", "started_at": 10,
    }))


def _candidate(runs: Path, name: str, label: str, status: str) -> None:
    """@brief 在指定 run 下造一个候选目录，含 kernel.metal 与带状态的 report.json。

    @param runs runs 根目录。
    @param name 候选所属的 run id。
    @param label 候选标签，决定目录名后缀。
    @param status 写进 report.json 的状态，例如 abi_error。
    """
    folder = runs / name / "candidates" / f"000_{label}"
    folder.mkdir(parents=True)
    (folder / "kernel.metal").write_text("kernel void k() {}\n")
    (folder / "report.json").write_text(json.dumps({"status": status, "stage": "abi", "speedup": None}))


def test_query_reads_the_index_and_prints_one_row_per_line(tmp_path, monkeypatch, capsys):
    """@brief 验证索引刷新后可按任务与失败状态检索，且 query CLI 的 --jsonl 每条结果占一行。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的 monkeypatch，用于把 runs 目录与各索引路径指向临时目录。
    @param capsys pytest 的输出捕获，用于读取 CLI 输出的 JSONL。
    """
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
    """@brief 验证扫描只把超过保留天数的目录标为可删，apply 只删它们并写下回收日志。

    @param tmp_path pytest 提供的临时目录。
    """
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
    """@brief 验证 gc CLI 默认走 dry-run，只报告可删数量而不删除目录。

    @param tmp_path pytest 提供的临时目录。
    @param capsys pytest 的输出捕获，用于解析 CLI 打印的报告。
    """
    root = tmp_path / "deleted"
    (root / "old").mkdir(parents=True)
    os.utime(root / "old", (time.time() - 40 * 86400,) * 2)
    assert cli.main(["gc", "--root", str(root), "--older-than-days", "30"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "dry-run" and report["eligible"] == 1
    assert (root / "old").exists()


def test_gc_refuses_a_root_outside_the_archive_and_apply_without_confirm(tmp_path, capsys):
    """@brief 验证 gc 对归档目录之外的 root、零天数以及缺 --confirm 的 --apply 都以非零退出码 5 拒绝。

    @param tmp_path pytest 提供的临时目录。
    @param capsys pytest 的输出捕获。
    """
    repo_like = tmp_path / "repo"
    (repo_like / "taic").mkdir(parents=True)
    assert cli.main(["gc", "--root", str(repo_like), "--older-than-days", "30", "--apply", "--confirm"]) == 5
    assert cli.main(["gc", "--root", str(tmp_path / "deleted"), "--older-than-days", "0"]) == 5
    assert cli.main(["gc", "--root", str(tmp_path / "deleted"), "--older-than-days", "30", "--apply"]) == 5
    assert (repo_like / "taic").is_dir()


def test_gc_apply_refuses_paths_outside_the_archive_and_symlinks(tmp_path):
    """@brief 验证扫描不跟随符号链接，且即使行被标为可删，指向归档根之外的路径也不会被删除。

    @param tmp_path pytest 提供的临时目录。
    """
    root = tmp_path / "deleted"
    root.mkdir()
    old = root / "old"
    old.mkdir()
    (old / "x").write_text("x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("keep")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    rows = gc.scan(root, older_than_days=0, now=time.time())
    assert "escape" not in {Path(row["path"]).name for row in rows}
    # a row that points outside the archive root is not deleted, even if it says eligible
    escaped = gc.apply([{"path": str(outside), "eligible": True, "files": 1, "bytes": 1, "age_days": 40}], log_path=root / "gc_log.jsonl", root=root)
    assert escaped["removed"] == [] and outside.is_dir()
    result = gc.apply(rows, log_path=root / "gc_log.jsonl", root=root)
    assert result["removed"] == [str(old)] and not old.exists() and outside.is_dir()

