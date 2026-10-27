"""PR26 冷建并行：``sync_known`` 的 workers、并行/串行逐字段对拍与进程池回退。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import concurrent.futures

from observe.index import Index


def _env(tmp_path: Path, monkeypatch, name: str) -> Path:
    """@brief 指向独立的临时 runs/ 与数据库。

    @param tmp_path pytest 临时目录。
    @param monkeypatch 环境变量替换。
    @param name 子目录名（同一次测试要建两份独立的库）。
    @return ``runs/`` 根。
    """
    runs = tmp_path / name / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / name / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / name / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / name / "carry.json"))
    return runs


def _run(root: Path, name: str, *, candidates: int = 1) -> None:
    """@brief 造一个最小的已结束运行目录。

    @param root ``runs/`` 根。
    @param name 运行名。
    @param candidates 候选个数。
    """
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": "small", "phase": "done",
        "started_at": 1, "updated_at": 1,
    }), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.1, "budget_usd": 1,
    }), encoding="utf-8")
    folder = directory / "candidates"
    folder.mkdir()
    for index in range(candidates):
        cand = folder / f"{index:03d}_r1.p0"
        cand.mkdir()
        (cand / "kernel.metal").write_text("kernel void k() {}\n", encoding="utf-8")
        (cand / "report.json").write_text(json.dumps({"status": "ok", "speedup": 1.0}), encoding="utf-8")


def _tables(index: Index) -> dict[str, list[dict]]:
    """@brief 取所有业务表 + 两张 FTS 的全部行（排除 meta 与 FTS 影子表）。

    只比较真实列（`SELECT *`，不含 rowid）；按整行 JSON 排序。串行/并行必须全表逐字段一致。

    @param index 索引。
    @return 表名 -> 已排序行列表。
    """
    snapshot: dict[str, list[dict]] = {}
    tables = list(index.conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ))
    for table in tables:
        name = str(table["name"])
        if name == "meta" or "_fts_" in name:
            continue
        rows = [dict(row) for row in index.conn.execute(f'SELECT * FROM "{name}"')]
        snapshot[name] = sorted(rows, key=lambda row: json.dumps(row, sort_keys=True, default=str))
    return snapshot


def _make_confirmed(directory: Path) -> None:
    """@brief 把 `_run` 造出的目录改写成通过发布门的运行（含 best/confirm/最大候选）。

    @param directory 运行目录。
    """
    from tcenv.schema import PROTOCOL_VERSION

    kernel = "kernel void k() {}\n"
    digest = hashlib.sha256(kernel.encode("utf-8")).hexdigest()
    (directory / "best.metal").write_text(kernel, encoding="utf-8")
    candidate = directory / "candidates" / "000_r1.p0"
    (candidate / "kernel.metal").write_text(kernel, encoding="utf-8")
    (candidate / "report.json").write_text(json.dumps({"status": "ok", "speedup": 1.5}), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": "small", "stop_reason": "max_rounds",
        "started_at": 1, "spent_usd": 0.1, "budget_usd": 1, "target_id": "apple-m4-max",
        "confirmed_speedup": 1.5, "confirmed_status": "ok", "confirm_winner_label": "r1.p0",
        "confirmation_source_sha256": digest, "confirmed_values": [1.5],
        "confirmed_worst": 1.5, "confirmed_best": 1.5,
    }), encoding="utf-8")
    (directory / "confirm_report.json").write_text(json.dumps({
        "status": "ok", "speedup": 1.5,
        "benchmark": {
            "noisy": False, "candidate": {"cv": 0.01, "reps": 15},
            "baselines": {"mlx": {"cv": 0.01, "reps": 15}}, "best_baseline": "mlx",
            "confirmation": {"kernel_sha256": digest, "winner_label": "r1.p0",
                             "confirmed_values": [1.5], "speedup": 1.5, "candidate_us": None},
        },
        "protocol_version": PROTOCOL_VERSION, "target_id": "apple-m4-max",
    }), encoding="utf-8")


def _make_failing_candidate(directory: Path) -> None:
    """@brief 让一个候选报告失败，保证 `failure_members` 非空。

    @param directory 运行目录。
    """
    (directory / "candidates" / "000_r1.p0" / "report.json").write_text(json.dumps({
        "status": "error", "stage": "compile", "message": "abi_error: missing symbol",
    }), encoding="utf-8")


class _FakeExecutor:
    """@brief 在进程内按调用顺序跑 ``map`` 的假进程池。"""

    calls: list[int] = []

    def __init__(self, max_workers=None) -> None:
        self.max_workers = max_workers
        _FakeExecutor.calls.append(int(max_workers or 0))

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def map(self, function, iterable, chunksize=1):
        return [function(item) for item in iterable]


def _seed(runs: Path, count: int = 70) -> None:
    """@brief 造 ``count`` 场运行（超过 32 的并行阈值）。

    @param runs ``runs/`` 根。
    @param count 场数。
    """
    for number in range(count):
        _run(runs, f"run_{number:03d}", candidates=1 + number % 3)


def test_the_parallel_cold_build_uses_the_requested_workers_and_matches_the_serial_rows(tmp_path, monkeypatch):
    """@brief 70 场冷建：workers=2 走并行路径，与 workers=1 的串行结果逐字段一致。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch, "tree")
    _seed(runs)
    _make_confirmed(runs / "run_000")
    _make_failing_candidate(runs / "run_001")
    serial = Index(tmp_path / "tree" / "serial.sqlite")
    monkeypatch.setenv("OBSERVE_INDEX_WORKERS", "1")
    serial.sync_known()
    expected = _tables(serial)
    # 防止空表假绿：这些表必须在语料里真的有行。
    for name in ("failure_members", "confirmed_best", "run_rollup", "runs_fts", "cands_fts"):
        assert expected.get(name), f"{name} must be non-empty for the comparison to mean anything"

    # 同一棵树、另一份库：指纹与 mtime 相同，才能逐字段对拍。
    parallel = Index(tmp_path / "tree" / "parallel.sqlite")
    monkeypatch.setenv("OBSERVE_INDEX_WORKERS", "2")
    _FakeExecutor.calls = []
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _FakeExecutor)
    parallel.sync_known()
    assert _FakeExecutor.calls == [2], _FakeExecutor.calls
    assert _tables(parallel) == expected
    assert parallel.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 70


def test_a_real_process_pool_cold_build_matches_the_serial_rows(tmp_path, monkeypatch):
    """@brief 不替换进程池：真的用 spawn 子进程跑 70 场，结果与串行逐字段一致。

    前面的并行用例用 ``_FakeExecutor`` 在进程内调函数，没走过 payload 的序列化/Path 往返/
    子进程环境；这条是唯一会真正启动 ``ProcessPoolExecutor`` 的冷建测试。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch, "realpool")
    _seed(runs)
    _make_confirmed(runs / "run_000")
    _make_failing_candidate(runs / "run_001")
    monkeypatch.setenv("OBSERVE_INDEX_WORKERS", "1")
    serial = Index(tmp_path / "realpool" / "serial.sqlite")
    serial.sync_known()
    expected = _tables(serial)

    monkeypatch.setenv("OBSERVE_INDEX_WORKERS", "2")
    parallel = Index(tmp_path / "realpool" / "parallel.sqlite")
    parallel.sync_known()
    assert parallel.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 70
    assert _tables(parallel) == expected


def test_a_broken_process_pool_falls_back_to_the_serial_ingest(tmp_path, monkeypatch):
    """@brief 进程池建不起来时冷建仍完成，结果与串行一致。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """

    def broken(*_args, **_kwargs):
        raise OSError("no process pool here")

    runs = _env(tmp_path, monkeypatch, "broken")
    _seed(runs, 70)
    monkeypatch.setenv("OBSERVE_INDEX_WORKERS", "3")
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", broken)
    index = Index(tmp_path / "broken" / "observe.sqlite")
    index.sync_known()
    assert index.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 70
    assert index.conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] > 0


def test_the_worker_override_is_clamped_and_defaults_to_the_core_count(monkeypatch):
    """@brief ``OBSERVE_INDEX_WORKERS`` 覆盖：非法值退回 1，缺省是 ``min(8, cpu)``。

    @param monkeypatch pytest 替换工具。
    """
    index = Index.__new__(Index)
    monkeypatch.delenv("OBSERVE_INDEX_WORKERS", raising=False)
    import os

    assert index._index_workers() == min(8, os.cpu_count() or 1)
    monkeypatch.setenv("OBSERVE_INDEX_WORKERS", "4")
    assert index._index_workers() == 4
    monkeypatch.setenv("OBSERVE_INDEX_WORKERS", "0")
    assert index._index_workers() == 1
    monkeypatch.setenv("OBSERVE_INDEX_WORKERS", "not-a-number")
    assert index._index_workers() == 1


@pytest.mark.parametrize("count", [1, 63, 64])
def test_small_trees_stay_serial_and_never_start_a_pool(tmp_path, monkeypatch, count):
    """@brief 少于 32 场的冷建不启动进程池（避免小语料的固定开销）。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    @param count 场数。
    """
    runs = _env(tmp_path, monkeypatch, f"small{count}")
    _seed(runs, count)
    monkeypatch.setenv("OBSERVE_INDEX_WORKERS", "4")
    _FakeExecutor.calls = []
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _FakeExecutor)
    index = Index(tmp_path / f"small{count}" / "observe.sqlite")
    index.sync_known()
    assert (_FakeExecutor.calls == []) == (count < 64), _FakeExecutor.calls
    assert index.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == count


def test_a_payload_read_that_races_a_writer_retries_and_keeps_the_new_value(tmp_path, monkeypatch):
    """@brief payload 先读后算指纹的竞态：读取窗口变了就重试，不能返回旧内容配新指纹。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    import observe.index as index_mod

    runs = _env(tmp_path, monkeypatch, "race")
    _run(runs, "run_000", candidates=1)
    directory = runs / "run_000"
    real_read = index_mod._read_json_file
    raced = {"done": False}

    def racing_read(path):
        data = real_read(path)
        if path.name == "state.json" and not raced["done"]:
            raced["done"] = True
            payload = json.loads((directory / "state.json").read_text(encoding="utf-8"))
            payload["updated_at"] = 999_999_999
            (directory / "state.json").write_text(json.dumps(payload), encoding="utf-8")
        return data

    monkeypatch.setattr(index_mod, "_read_json_file", racing_read)
    payload = index_mod._read_run_payload(str(directory))
    monkeypatch.setattr(index_mod, "_read_json_file", real_read)
    assert payload is None or payload["state"]["updated_at"] == 999_999_999, payload


def test_a_raced_payload_does_not_leave_the_old_value_bound_to_the_new_fingerprint(tmp_path, monkeypatch):
    """@brief 冷建并行时写入的 state 不能被旧内容覆盖；下一帧也不能因指纹相同而停在旧值。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    import observe.index as index_mod

    runs = _env(tmp_path, monkeypatch, "race-index")
    _seed(runs, 70)
    directory = runs / "run_000"
    real_read = index_mod._read_json_file
    raced = {"done": False}

    def racing_read(path):
        data = real_read(path)
        if path == directory / "state.json" and not raced["done"]:
            raced["done"] = True
            payload = json.loads((directory / "state.json").read_text(encoding="utf-8"))
            payload["updated_at"] = 999_999_999
            (directory / "state.json").write_text(json.dumps(payload), encoding="utf-8")
        return data

    monkeypatch.setenv("OBSERVE_INDEX_WORKERS", "2")
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(index_mod, "_read_json_file", racing_read)
    index = Index(tmp_path / "race-index" / "observe.sqlite")
    index.sync_known()
    monkeypatch.setattr(index_mod, "_read_json_file", real_read)
    assert index.conn.execute("SELECT updated_at FROM runs WHERE run_id = 'run_000'").fetchone()[0] == 999_999_999
    index.sync_known()
    assert index.conn.execute("SELECT updated_at FROM runs WHERE run_id = 'run_000'").fetchone()[0] == 999_999_999


def test_a_stale_payload_is_rejected_and_the_serial_window_reads_the_new_state(tmp_path, monkeypatch):
    """@brief 并行 payload 在写入库前已过期：`_ingest` 必须回退串行读取，不能把旧 state 绑到新指纹。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    import observe.index as index_mod

    runs = _env(tmp_path, monkeypatch, "stale")
    _run(runs, "run_000", candidates=1)
    directory = runs / "run_000"
    payload = index_mod._read_run_payload(str(directory))
    assert payload is not None and payload["state"]["updated_at"] == 1
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    state["updated_at"] = 999_999_999
    (directory / "state.json").write_text(json.dumps(state), encoding="utf-8")
    index = Index(tmp_path / "stale" / "observe.sqlite")
    index._ingest(directory, payload=payload)
    row = index.conn.execute("SELECT updated_at FROM runs WHERE run_id = 'run_000'").fetchone()
    assert row[0] == 999_999_999
