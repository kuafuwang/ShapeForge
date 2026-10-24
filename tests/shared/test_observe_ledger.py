"""Spend totals live in the observe index. A later reading may raise a total. It never lowers one."""

from __future__ import annotations

import json
import sqlite3

import pytest

from observe.index import CarryoverError, Index, carryover_path, preserved_ledger_path
from observe.ledger import month_of, operator_name


def test_operator_name_keeps_a_tile_variant_with_its_operator():
    """@brief 验证算子名去掉 @64x32t256 这类 tile 变体后缀，空名字记为 unknown。
    """
    assert operator_name("flash_attention@64x32t256") == "flash_attention"
    assert operator_name("gemm") == "gemm"
    assert operator_name("") == "unknown"


def _run(root, name, *, task, spent, budget, started):
    """@brief 在索引根目录里手写一个已结束运行的 result.json 与 state.json。

    @param root 运行根目录。
    @param name 运行目录名。
    @param task 任务名。
    @param spent 已花金额。
    @param budget 预算上限。
    @param started 开始时间戳。
    @return 建好的运行目录路径。
    """
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(json.dumps({
        "task": task, "dtype": "fp16", "stop_reason": "max_rounds",
        "spent_usd": spent, "budget_usd": budget, "started_at": started,
    }))
    (directory / "state.json").write_text(json.dumps({
        "task": task, "dtype": "fp16", "phase": "done", "started_at": started, "updated_at": started,
    }))
    return directory


def test_spend_only_rises_and_a_rebuild_keeps_the_carryover_identity(tmp_path, monkeypatch):
    """@brief 验证花费总额只升不降：回写的更小金额只改当前值、不改最大值，重建索引后旧的账本 carryover 仍然接得上。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来把运行目录、索引库与账本都指到临时路径。
    """
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    books = tmp_path / "books.sqlite"
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(books))
    with sqlite3.connect(books) as conn:
        conn.execute(
            """
            CREATE TABLE search (
                run_id TEXT PRIMARY KEY, operator TEXT, task TEXT, dtype TEXT,
                started_at REAL, month TEXT, spent_usd REAL, budget_usd REAL
            )
            """
        )
        conn.execute("INSERT INTO search VALUES ('gone', 'gemm', 'gemm', 'fp16', 1, '2020-01', 10.0, 15.0)")
    index = Index(tmp_path / "observe.sqlite")
    runs = tmp_path / "runs"
    started = index.conn.execute("SELECT created_at FROM ledger_carryover").fetchone()[0] + 10
    directory = _run(runs, "fresh", task="relu", spent=2.0, budget=3.0, started=started)
    index.sync_known()
    assert index.total_usd() == pytest.approx(12.0)
    result = json.loads((directory / "result.json").read_text())
    result["spent_usd"] = 0.5
    (directory / "result.json").write_text(json.dumps(result))
    index.sync_known()
    got = index.conn.execute("SELECT spent_usd, spent_max_usd FROM runs WHERE run_id = 'fresh'").fetchone()
    assert (got["spent_usd"], got["spent_max_usd"]) == (0.5, 2.0)
    assert index.total_usd() == pytest.approx(12.0)
    result["spent_usd"] = 2.0
    (directory / "result.json").write_text(json.dumps(result))
    index.conn.close()
    (tmp_path / "observe.sqlite").unlink()
    for suffix in ("-wal", "-shm"):
        leftover = tmp_path / f"observe.sqlite{suffix}"
        if leftover.exists():
            leftover.unlink()
    rebuilt = Index(tmp_path / "observe.sqlite")
    rebuilt.sync_known()
    assert rebuilt.total_usd() == pytest.approx(12.0)
    assert json.loads(carryover_path().read_text())["rows"] == 1


def test_a_schema_rebuild_keeps_the_archived_spend(tmp_path, monkeypatch):
    """@brief 验证 schema 版本回退触发的重建仍保留已归档花费：删掉运行后账本行还在，重建后总额不变。

    drop_archived 保留账本行，删除因此不能压低已记录花费；而重建过去会直接删库，把这些行一起丢掉。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来把索引库与账本指到临时路径。
    """
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "missing.sqlite"))
    monkeypatch.delenv("OBSERVE_CARRYOVER", raising=False)
    index = Index(tmp_path / "observe.sqlite")
    runs = tmp_path / "runs"
    _run(runs, "gone", task="gemm", spent=3.5, budget=5.0, started=100.0)
    index.sync_known()
    assert index.total_usd() == pytest.approx(3.5)
    index.drop_archived(["gone"], [])
    assert index.total_usd() == pytest.approx(3.5)  # the ledger row survives the deletion
    index.conn.execute("UPDATE meta SET value = '0' WHERE key = 'index_schema_version'")
    index.conn.commit()
    index.conn.close()
    rebuilt = Index(tmp_path / "observe.sqlite")
    assert rebuilt.total_usd() == pytest.approx(3.5)
    assert not preserved_ledger_path().exists()


def test_an_empty_ledger_does_not_let_a_stale_sidecar_come_back(tmp_path, monkeypatch):
    """@brief 验证重建时数据库里没有花费行就必须清掉旧 sidecar，否则重放会把一个被人为清空的账本又装回来。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来把索引库指到临时路径。
    """
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "missing.sqlite"))
    monkeypatch.delenv("OBSERVE_CARRYOVER", raising=False)
    index = Index(tmp_path / "observe.sqlite")
    index.conn.execute("UPDATE meta SET value = '0' WHERE key = 'index_schema_version'")
    index.conn.commit()
    index.conn.close()
    preserved_ledger_path(index.path).write_text(json.dumps({
        "saved_at": 1.0, "rows": [["gone", "gemm", "2020-01", 100.0, 9.9, 9.9]],
    }))
    rebuilt = Index(tmp_path / "observe.sqlite")
    assert rebuilt.total_usd() == 0.0
    assert not preserved_ledger_path(rebuilt.path).exists()


def test_a_corrupt_sidecar_is_dropped_instead_of_retried(tmp_path, monkeypatch):
    """@brief 验证内容损坏的 sidecar 被直接丢弃而不是每次重建都重试，索引以空账本建立。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来把索引库指到临时路径。
    """
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "missing.sqlite"))
    monkeypatch.delenv("OBSERVE_CARRYOVER", raising=False)
    preserved_ledger_path(tmp_path / "observe.sqlite").write_text("{not json")
    index = Index(tmp_path / "observe.sqlite")
    assert index.total_usd() == 0.0
    assert not preserved_ledger_path(index.path).exists()


def test_a_carryover_checksum_mismatch_refuses_to_build(tmp_path, monkeypatch):
    """@brief 验证 carryover 的 sha256 对不上时索引拒绝建立，抛 CarryoverError。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来把 carryover 指到临时文件。
    """
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "missing.sqlite"))
    (tmp_path / "carry.json").write_text(json.dumps({
        "created_at": 1, "rows": 1, "total_usd": 3, "origin": "books", "sha256": "nope",
    }))
    with pytest.raises(CarryoverError):
        Index(tmp_path / "observe.sqlite")


def test_month_of_is_local_time():
    """@brief 验证 month_of 对空时间戳返回 unknown，对同一时间戳给出稳定结果。
    """
    assert month_of(None) == "unknown"
    assert month_of(1791315864.0) == month_of(1791315864.0)
