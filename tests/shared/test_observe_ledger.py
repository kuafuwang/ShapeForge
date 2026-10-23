"""Spend totals live in the observe index. A later reading may raise a total. It never lowers one."""

from __future__ import annotations

import json
import sqlite3

import pytest

from observe.index import CarryoverError, Index, carryover_path, preserved_ledger_path
from observe.ledger import month_of, operator_name


def test_operator_name_keeps_a_tile_variant_with_its_operator():
    assert operator_name("flash_attention@64x32t256") == "flash_attention"
    assert operator_name("gemm") == "gemm"
    assert operator_name("") == "unknown"


def _run(root, name, *, task, spent, budget, started):
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
    """drop_archived keeps ledger rows so a deletion cannot lower the recorded spend. A rebuild used
    to unlink the database and lose those rows with it."""
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
    """A rebuild whose database has no spend rows must clear an older sidecar: replaying it would
    restore a ledger that was deliberately emptied."""
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
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "missing.sqlite"))
    monkeypatch.delenv("OBSERVE_CARRYOVER", raising=False)
    preserved_ledger_path(tmp_path / "observe.sqlite").write_text("{not json")
    index = Index(tmp_path / "observe.sqlite")
    assert index.total_usd() == 0.0
    assert not preserved_ledger_path(index.path).exists()


def test_a_carryover_checksum_mismatch_refuses_to_build(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "missing.sqlite"))
    (tmp_path / "carry.json").write_text(json.dumps({
        "created_at": 1, "rows": 1, "total_usd": 3, "origin": "books", "sha256": "nope",
    }))
    with pytest.raises(CarryoverError):
        Index(tmp_path / "observe.sqlite")


def test_month_of_is_local_time():
    assert month_of(None) == "unknown"
    assert month_of(1791315864.0) == month_of(1791315864.0)
