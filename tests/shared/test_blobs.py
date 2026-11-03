"""§9.1 持久证据：确认写路径捕获内容寻址 blob，保留策略按文件 mtime，删库不碰 blob。

blob 与目录史在同一处写（``taic.kernel_catalog.archive_confirmed``）；写失败只记日志，确认照常成功。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from observe import publications
from observe.index import Index
from taic import blobs, kernel_catalog
from tests.support.v1_tasks import V1_TASK

_SOURCE = "kernel void blob_fixture() {}\n"


def _env(tmp_path, monkeypatch):
    """@brief 临时 catalog/blob/runs/索引根。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @return ``(catalog, blob 根, runs, Index)``。
    """
    catalog = tmp_path / "kernels"
    blob_root = tmp_path / "blobs"
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(catalog))
    monkeypatch.setenv("TAIC_BLOB_ROOT", str(blob_root))
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    monkeypatch.setenv("OBSERVE_REFRESH_ON_READ", "0")
    return catalog, blob_root, runs, Index(tmp_path / "observe.sqlite")


def _archive(source: str = _SOURCE, *, run_id: str = "a", speedup: float = 1.25,
             task: str = V1_TASK, size: str = "small") -> str:
    """@brief 走真实确认写路径；返回目录史记下的源码摘要。

    @param source 确认过的源码。
    @param run_id 来源目录名。
    @param speedup 确认加速比。
    @param task 任务名。
    @param size 尺寸档位。
    @return 源码摘要。
    """
    kernel_catalog.archive_confirmed(
        target_id="apple-m4-max", backend="metal", task=task, dtype="fp16", size=size,
        source=source, speedup=speedup, run_id=run_id, label="r1.p0",
        confirmed_values=[speedup], candidate_us=100.0, baseline_us={"mlx": 120.0},
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def test_the_confirmation_write_path_captures_a_matching_blob(tmp_path, monkeypatch):
    """@brief 验收①：新确认一次运行后 blob 存在，内容与 ``history.source_sha256`` 一致。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    catalog, blob_root, _runs, _index_ = _env(tmp_path, monkeypatch)
    digest = _archive()

    history = next(catalog.glob("*/*/*/*/history.jsonl"))
    record = json.loads(history.read_text().splitlines()[0])
    assert record["source_sha256"] == digest
    path = blob_root / digest[:2] / digest
    assert path.is_file() and path.read_text() == _SOURCE
    # 同一份源码再次确认不重复写，也不改内容。
    _archive(run_id="b", speedup=1.30)
    assert path.read_text() == _SOURCE
    assert len([item for item in blob_root.rglob("*") if item.is_file()]) == 1


def test_a_failed_blob_write_never_fails_the_confirmation_and_leaves_no_half_file(tmp_path, monkeypatch):
    """@brief 验收④：blob 写失败被吞掉，历史照写，正式名字下没有半截文件。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    catalog, blob_root, _runs, _index_ = _env(tmp_path, monkeypatch)
    from taic import control

    real = control.write_text_atomic

    def failing(path, text, **kwargs):
        if Path(path).parent.parent == blob_root:
            raise OSError("disk full")
        return real(path, text, **kwargs)

    monkeypatch.setattr(control, "write_text_atomic", failing)
    best = kernel_catalog.archive_confirmed(
        target_id="apple-m4-max", backend="metal", task=V1_TASK, dtype="fp16", size="small",
        source=_SOURCE, speedup=1.25, run_id="a", label="r1.p0",
    )
    assert best.is_file(), "确认本身必须成功"
    assert len(next(catalog.glob("*/*/*/*/history.jsonl")).read_text().splitlines()) == 1
    leftovers = [item for item in blob_root.rglob("*") if item.is_file()] if blob_root.exists() else []
    assert leftovers == [], "失败后不能留下半截 blob 或临时文件"


def test_publications_report_blob_availability_after_the_run_directory_moves(tmp_path, monkeypatch):
    """@brief 验收②：run 目录搬到归档后 ``source_available=0`` 但 ``blob_available=1``，端点仍给源码。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    import shutil

    catalog, blob_root, runs, index = _env(tmp_path, monkeypatch)
    digest = _archive(run_id="one")
    directory = runs / "conv2d" / "one"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": V1_TASK, "dtype": "fp16", "phase": "done", "started_at": 1.0,
    }), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "task": V1_TASK, "dtype": "fp16", "size": "small", "stop_reason": "max_rounds",
        "protocol_version": "3",
    }), encoding="utf-8")
    winner = directory / "candidates" / "000_r1.p0"
    winner.mkdir(parents=True)
    (winner / "kernel.metal").write_text(_SOURCE, encoding="utf-8")
    (winner / "report.json").write_text(json.dumps({"status": "ok", "stage": "bench"}), encoding="utf-8")
    index.sync_known()
    index._sync_kernels()
    before = publications.publications(task=V1_TASK)["rows"][0]
    assert before["source_available"] is True and before["blob_available"] is True
    refs = index.conn.execute("SELECT COUNT(*) FROM blob_refs WHERE sha256 = ?", (digest,)).fetchone()[0]
    assert refs == 1

    archived = tmp_path / "deleted" / "one"
    archived.parent.mkdir(parents=True)
    shutil.move(str(directory), str(archived))
    index.sync_known()
    after = publications.publications(task=V1_TASK)["rows"][0]
    assert after["source_available"] is False
    assert after["blob_available"] is True

    from starlette.testclient import TestClient

    from observe.app import app

    with TestClient(app) as client:
        response = client.get(f"/api/v2/publications/{digest}/source")
        assert response.status_code == 200 and response.text == _SOURCE
        assert client.get("/api/v2/publications/" + "0" * 64 + "/source").status_code == 404
        assert client.get("/api/v2/publications/not-a-sha/source").status_code == 404
    assert blob_root.joinpath(digest[:2], digest).is_file()


def test_prune_uses_the_blob_file_mtime_and_needs_no_reference_rows(tmp_path, monkeypatch):
    """@brief 验收③：没有 ``blob_refs`` 行也能按文件 mtime 列出；``--apply`` 只删到期项。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    blob_root = tmp_path / "blobs"
    monkeypatch.setenv("TAIC_BLOB_ROOT", str(blob_root))
    now = 1_800_000_000.0
    old = blob_root / ("aa" * 32)[:2] / ("aa" * 32)
    new = blob_root / ("bb" * 32)[:2] / ("bb" * 32)
    for path, age_days in ((old, 400.0), (new, 10.0)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("kernel\n", encoding="utf-8")
        stamp = now - age_days * 86400.0
        os.utime(path, (stamp, stamp))

    rows = blobs.scan(blob_root, older_than_days=365.0, now=now)
    assert [row["sha256"] for row in rows] == sorted([old.name, new.name])
    assert {row["sha256"]: row["eligible"] for row in rows} == {old.name: True, new.name: False}
    result = blobs.apply(rows)
    assert result["removed"] == [str(old)] and result["bytes"] == len("kernel\n")
    assert not old.exists() and new.is_file()
    assert not old.parent.exists(), "删空的分桶目录要一起收掉"
    assert blobs.read_source(new.name, blob_root) == "kernel\n"
    assert blobs.read_source("../../etc/passwd") is None


def test_the_blob_cli_defaults_to_a_report_and_requires_confirmation(tmp_path, monkeypatch, capsys):
    """@brief CLI：默认只报告；``--apply`` 缺 ``--prune``/``--confirm`` 时拒绝，给全才删。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @param capsys pytest 的 stdout/stderr 捕获。
    """
    from taic import cli

    blob_root = tmp_path / "blobs"
    monkeypatch.setenv("TAIC_BLOB_ROOT", str(blob_root))
    sha = "cc" * 32
    path = blob_root / sha[:2] / sha
    path.parent.mkdir(parents=True)
    path.write_text("kernel\n", encoding="utf-8")
    old = time.time() - 400 * 86400
    os.utime(path, (old, old))

    assert cli.main(["blobs", "--days", "365"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "report" and report["eligible"] == 1
    assert "rows" not in report

    assert cli.main(["blobs", "--prune", "--days", "365"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["mode"] == "dry-run" and dry["rows"] and dry["rows"][0]["eligible"] is True
    assert path.is_file(), "dry-run 不能删东西"

    assert cli.main(["blobs", "--apply", "--days", "365"]) == 5
    assert "refusing" in capsys.readouterr().err
    assert cli.main(["blobs", "--prune", "--apply", "--days", "365"]) == 5
    assert "refusing" in capsys.readouterr().err
    assert path.is_file()

    assert cli.main(["blobs", "--prune", "--apply", "--confirm", "--days", "365"]) == 0
    assert json.loads(capsys.readouterr().out)["bytes"] == len("kernel\n")
    assert not path.exists()


def test_discarding_the_index_keeps_the_blob_bytes(tmp_path, monkeypatch):
    """@brief 验收⑥：``Index._discard()`` 只删数据库，blob 集合逐字节不变。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _catalog, blob_root, _runs, index = _env(tmp_path, monkeypatch)
    digest = _archive()
    path = blob_root / digest[:2] / digest
    before = path.read_bytes()
    index._discard()
    assert not Path(str(index.path)).exists()
    assert path.read_bytes() == before
