"""PR23 便宜对账：缓存根 + 字符串相对路径 + ``scandir`` 计数。

这里断言的是机制（解析次数、路径归属、逐字段结果），不是 10k 的墙钟时间；计时由
``experiments.scale_probe`` 单独报告，避免把一台机器的抖动写进快套件。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observe.index import Index


def _run(root: Path, name: str, *, task: str = "relu", candidates: int = 0) -> Path:
    """@brief 造一个最小的已结束运行目录。

    @param root ``runs/`` 根。
    @param name 相对根目录的运行名。
    @param task 任务名。
    @param candidates 带 ``report.json``/``kernel.metal`` 的候选个数。
    @return 运行目录。
    """
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": task, "dtype": "fp16", "phase": "done", "started_at": 1, "updated_at": 1,
    }), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "task": task, "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.1, "budget_usd": 1,
    }), encoding="utf-8")
    if candidates:
        folder = directory / "candidates"
        folder.mkdir()
        for index in range(candidates):
            cand = folder / f"{index:03d}_r1.p0"
            cand.mkdir()
            (cand / "kernel.metal").write_text("kernel void k() {}\n", encoding="utf-8")
            (cand / "report.json").write_text(json.dumps({"status": "ok", "speedup": 1.0}), encoding="utf-8")
    return directory


def _env(tmp_path: Path, monkeypatch, count: int) -> tuple[Path, Index]:
    """@brief 指向临时 runs/ 与数据库，造 ``count`` 个运行，返回根与索引。

    @param tmp_path pytest 临时目录。
    @param monkeypatch 环境变量替换。
    @param count 运行个数。
    @return ``(runs 根, Index)``。
    """
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    runs.mkdir()
    for index in range(count):
        _run(runs, f"run_{index:02d}", candidates=1 if index % 5 == 0 else 0)
    return runs, Index(tmp_path / "observe.sqlite")


def _count_resolves(monkeypatch) -> dict:
    """@brief 包住 ``Path.resolve`` 计数，返回调用记录。

    @param monkeypatch pytest 替换工具。
    @return ``{"n": 次数, "paths": [...]}``。
    """
    calls = {"n": 0, "paths": []}
    real = Path.resolve

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        calls["paths"].append(str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counting)
    return calls


def test_the_root_is_resolved_once_and_a_hot_frame_never_resolves_per_run(tmp_path, monkeypatch):
    """@brief 冷帧只解析根一次；热帧 0 次根解析，逐 run 最多一次（``resolve_run`` 的越界检查）。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, index = _env(tmp_path, monkeypatch, 25)
    calls = _count_resolves(monkeypatch)
    index.sync_known()
    assert index.stats["root_resolves"] == 1, index.stats
    cold = calls["n"]
    assert cold <= 2 * 25, cold

    calls["n"] = 0
    calls["paths"].clear()
    index.sync_known()
    assert index.stats["root_resolves"] == 0, "the hot frame re-resolved the root"
    assert calls["n"] <= 2 * 25, calls["paths"]
    # 热帧用 `_reconcile_path`（不逐级 resolve），逐 run 的越界检查交给缓存的符号链接判断。
    assert calls["n"] == 0, calls["paths"]
    assert index.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 25

    # 结果的逐字段一致：另一份库从同一棵目录重建后，runs 行完全相同。
    fresh = Index(tmp_path / "fresh.sqlite")
    fresh.sync_known()
    columns = "run_id, status, task, dtype, source, binding, fp_state, fp_result, fp_config, fp_confirm, fp_extra"
    first = [dict(row) for row in index.conn.execute(f"SELECT {columns} FROM runs ORDER BY run_id")]
    second = [dict(row) for row in fresh.conn.execute(f"SELECT {columns} FROM runs ORDER BY run_id")]
    assert second == first


def test_rel_matches_the_resolve_based_relative_path_and_rejects_an_escaping_symlink(tmp_path, monkeypatch):
    """@brief 字符串相对路径与旧的 ``resolve().relative_to(root)`` 同值；越界符号链接仍被拒绝。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, index = _env(tmp_path, monkeypatch, 1)
    nested = runs / "search_ab" / "fa_on_1"
    nested.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "escaped").mkdir()
    (runs / "inside").mkdir()
    (runs / "inside" / "nested").mkdir()
    (runs / "link-in").symlink_to(runs / "inside", target_is_directory=True)
    (runs / "link-out").symlink_to(outside, target_is_directory=True)

    root = runs.resolve()
    for path in (runs, runs / "run_00", nested, runs / "inside", runs / "inside" / "nested"):
        resolved = path.resolve()
        expected = resolved.relative_to(root).as_posix() if resolved != root else "."
        assert index._rel(path) == expected, path
    # 根内别名保持旧 canonical run id（旧口径是 path.resolve().relative_to(root)）。
    assert index._rel(runs / "link-in") == "inside"
    assert index._rel(runs / "link-in" / "nested") == "inside/nested"
    # 根外路径退回词法目录名（旧口径的 path.name）。
    assert index._rel(tmp_path / "elsewhere") == "elsewhere"
    assert index._rel(runs / "link-out") == "link-out"
    assert index._rel(runs / "link-out" / "escaped") == "escaped"
    assert index._within_runs(runs) is True
    assert index._within_runs(runs / "run_00") is True
    assert index._within_runs(runs / "link-in") is True
    assert index._within_runs(runs / "link-in" / "nested") is True
    assert index._within_runs(runs / "link-out") is False
    # 祖先 symlink 指向根外：最后一段不是 symlink 也必须拒绝（旧实现会误判为 true）。
    assert index._within_runs(runs / "link-out" / "escaped") is False
    assert index._within_runs(tmp_path / "elsewhere") is False
    # 前缀相同但不是子目录（runs_evil）不能被当成根内。
    evil = tmp_path / "runs_evil"
    evil.mkdir()
    assert index._within_runs(evil) is False


def test_scandir_enumeration_finds_a_new_nested_run_without_listing_again(tmp_path, monkeypatch):
    """@brief 嵌套目录只改父 mtime：``scandir`` 计数后仍能发现新 run，且不重复摄取旧 run。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, index = _env(tmp_path, monkeypatch, 1)
    (runs / "conv2d").mkdir()
    _run(runs, "conv2d/one")
    index.sync_known()
    seen = index.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert seen == 2
    assert index.stats["dir_lists"] >= 1

    _run(runs, "conv2d/two")
    index.sync_known()
    assert index.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 3
    assert index.conn.execute("SELECT 1 FROM runs WHERE run_id = 'conv2d/two'").fetchone() is not None
    # 没有新目录的热帧不列目录。
    index.sync_known()
    assert index.stats["dir_lists"] == 0


def test_a_manifest_run_keeps_its_experiment_identity_without_resolving_paths(tmp_path, monkeypatch):
    """@brief 上行 ``experiment.json`` 的 arm 归属仍按相对路径算；含 manifest 时解析次数不涨。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, index = _env(tmp_path, monkeypatch, 0)
    experiment = runs / "search_ab"
    experiment.mkdir()
    (experiment / "experiment.json").write_text(json.dumps({
        "kind": "search_ab",
        "driver": "search_ab",
        "frozen": {"tasks": ["relu"]},
        "arms": [{"arm_id": "on", "delta": {}, "planned_n": 1, "runs": ["fa_on_1"]}],
    }), encoding="utf-8")
    _run(runs, "search_ab/fa_on_1")
    index.sync_known()
    row = index.conn.execute("SELECT source, experiment_id, arm_id, binding FROM runs WHERE run_id = 'search_ab/fa_on_1'").fetchone()
    assert row["source"] == "manifest"
    assert row["experiment_id"] == "search_ab"
    assert row["arm_id"] == "on"

    calls = _count_resolves(monkeypatch)
    index.sync_known()
    assert index.stats["root_resolves"] == 0
    assert calls["n"] <= 4, calls["paths"]


@pytest.mark.parametrize("kind", ["relative", "absolute"])
def test_ingest_rejects_a_directory_outside_runs(tmp_path, monkeypatch, kind):
    """@brief ``_ingest`` 在任何路径写法下都不读根外目录。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    @param kind ``relative`` 或 ``absolute`` 两种写法。
    """
    runs, index = _env(tmp_path, monkeypatch, 1)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "state.json").write_text(json.dumps({"task": "relu"}), encoding="utf-8")
    target = Path("../outside") if kind == "relative" else outside
    index._ingest(target)
    assert index.conn.execute("SELECT 1 FROM runs WHERE run_id = 'outside'").fetchone() is None


def test_ingest_and_payload_reject_an_ancestor_symlink_escape_without_reading_outside(tmp_path, monkeypatch):
    """@brief `link -> outside` 时 `link/escaped` 必须被 `_within_runs`/`_ingest`/payload 拒绝，且不读外部文件。

    旧实现只看最后一段：`link/escaped` 词法前缀在根内，索引会把根外目录写成 `link/escaped`。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    import observe.index as index_mod

    runs, index = _env(tmp_path, monkeypatch, 1)
    outside = tmp_path / "outside"
    _run(outside, "escaped")
    (runs / "link").symlink_to(outside, target_is_directory=True)
    assert index._within_runs(runs / "link" / "escaped") is False

    reads: list[str] = []
    real_load = index_mod.detail._load

    def recording_load(path):
        reads.append(str(path))
        return real_load(path)

    monkeypatch.setattr(index_mod.detail, "_load", recording_load)
    index._ingest(runs / "link" / "escaped")
    payload = index_mod._read_run_payload(str(runs / "link" / "escaped"))
    assert payload is None
    assert index.conn.execute("SELECT 1 FROM runs WHERE run_id = 'link/escaped'").fetchone() is None
    assert not any(str(outside) in path for path in reads), reads

    # 冷建/全量入口同样不摄取根外目录。
    index.sync_known(force=True)
    ids = [row[0] for row in index.conn.execute("SELECT run_id FROM runs ORDER BY run_id")]
    assert ids == ["run_00"], ids
    assert not any(str(outside) in path for path in reads), reads
    monkeypatch.setattr(index_mod.detail, "_load", real_load)


def test_an_in_root_symlink_alias_keeps_the_canonical_run_id_for_ingest_and_payload(tmp_path, monkeypatch):
    """@brief 根内别名 `alias -> real_run`：`_rel`、`_ingest`、payload 都落到 canonical id `real_run`。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    import observe.index as index_mod

    runs, index = _env(tmp_path, monkeypatch, 1)
    real = _run(runs, "real_run")
    (runs / "alias").symlink_to(real, target_is_directory=True)
    assert index._rel(runs / "alias") == "real_run"
    payload = index_mod._read_run_payload(str(runs / "alias"))
    assert payload is not None, "根内别名必须能读并落到 canonical 目录"
    index._prepare_frame()
    index._ingest(runs / "alias", payload=payload)
    index.sync_known(force=True)
    ids = [row[0] for row in index.conn.execute("SELECT run_id FROM runs ORDER BY run_id")]
    assert ids == ["real_run", "run_00"], ids
    assert index.conn.execute("SELECT 1 FROM runs WHERE run_id = 'alias'").fetchone() is None
