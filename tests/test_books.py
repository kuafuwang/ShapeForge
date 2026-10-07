"""The spend book groups searches by operator and by the month they started. No GPU, no model calls."""

from __future__ import annotations

from observe.books import month_of, operator_name, sync


def test_operator_name_keeps_a_tile_variant_with_its_operator():
    assert operator_name("flash_attention@64x32t256") == "flash_attention"
    assert operator_name("gemm") == "gemm"
    assert operator_name("") == "unknown"


def test_spend_adds_up_by_operator_and_month_and_a_reset_does_not_change_the_spend(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "books.sqlite"))
    # October and November 2026 in UTC+8 are these epoch seconds (local month is what the book uses).
    october = month_of(1791315864.0)
    later = 1791315864.0 + 40 * 24 * 3600
    november = month_of(later)
    assert october != november
    first = sync([
        {"id": "p1/fa_flash", "task": "flash_attention", "dtype": "fp16", "started_at": 1791315864.0, "spent_usd": 3.12, "budget_usd": 3.0},
        {"id": "p1/fa_paper", "task": "flash_attention", "dtype": "fp16", "started_at": 1791315864.0, "spent_usd": 7.61, "budget_usd": 15.0},
        {"id": "p1/fa_wide", "task": "flash_attention@64x32t256", "dtype": "fp16", "started_at": later, "spent_usd": 3.04, "budget_usd": 3.0},
        {"id": "p1/gemm", "task": "gemm", "dtype": "fp16", "started_at": later, "spent_usd": 2.33, "budget_usd": 15.0},
    ])
    by_op = {row["operator"]: row for row in first["by_operator"]}
    assert by_op["flash_attention"]["runs"] == 3
    assert by_op["flash_attention"]["spent_usd"] == round(3.12 + 7.61 + 3.04, 6)
    assert by_op["gemm"]["spent_usd"] == 2.33
    by_month = {row["month"]: row for row in first["by_month"]}
    assert by_month[october]["spent_usd"] == round(3.12 + 7.61, 6)
    assert by_month[november]["spent_usd"] == round(3.04 + 2.33, 6)
    # a reset raises the allowance; the spent total of that search stays
    second = sync([
        {"id": "p1/fa_flash", "task": "flash_attention", "dtype": "fp16", "started_at": 1791315864.0, "spent_usd": 3.12, "budget_usd": 18.12},
        {"id": "p1/fa_paper", "task": "flash_attention", "dtype": "fp16", "started_at": 1791315864.0, "spent_usd": 7.61, "budget_usd": 15.0},
        {"id": "p1/fa_wide", "task": "flash_attention@64x32t256", "dtype": "fp16", "started_at": later, "spent_usd": 3.04, "budget_usd": 3.0},
        {"id": "p1/gemm", "task": "gemm", "dtype": "fp16", "started_at": later, "spent_usd": 2.33, "budget_usd": 15.0},
    ])
    again = {row["operator"]: row for row in second["by_operator"]}
    assert again["flash_attention"]["spent_usd"] == by_op["flash_attention"]["spent_usd"]
    assert again["flash_attention"]["budget_usd"] == round(18.12 + 15.0 + 3.0, 6)
    assert second["spent_usd"] == first["spent_usd"]
    # a later scan that does not mention gemm must not drop the money already recorded
    kept = sync([
        {"id": "p1/fa_flash", "task": "flash_attention", "dtype": "fp16", "started_at": 1791315864.0, "spent_usd": 3.12, "budget_usd": 18.12},
    ])
    assert {row["operator"] for row in kept["by_operator"]} >= {"flash_attention", "gemm"}
    assert kept["spent_usd"] == first["spent_usd"]


def test_an_unversioned_book_keeps_its_rows_and_becomes_version_1(tmp_path, monkeypatch):
    path = tmp_path / "books.sqlite"
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(path))
    import sqlite3
    from observe.books import SCHEMA_VERSION
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE search (
                run_id TEXT PRIMARY KEY,
                operator TEXT NOT NULL,
                task TEXT NOT NULL,
                dtype TEXT,
                started_at REAL,
                month TEXT NOT NULL,
                spent_usd REAL,
                budget_usd REAL
            )
            """
        )
        conn.execute(
            "INSERT INTO search VALUES ('old/run', 'gemm', 'gemm', 'fp16', 1791315864.0, '2026-10', 2.5, 15.0)"
        )
    report = sync([
        {"id": "new/run", "task": "relu", "dtype": "fp16", "started_at": 1791315864.0, "spent_usd": 0.2, "budget_usd": 1.0},
    ])
    assert report["schema_version"] == SCHEMA_VERSION == 1
    assert report["spent_usd"] == round(2.5 + 0.2, 6)
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        notes = conn.execute("SELECT version, note FROM schema_migration").fetchall()
        assert notes == [(1, "one row per search: operator, month, spent and current budget")]
        assert conn.execute("SELECT spent_usd FROM search WHERE run_id = 'old/run'").fetchone()[0] == 2.5


def test_a_newer_book_is_not_rewritten(tmp_path, monkeypatch):
    path = tmp_path / "books.sqlite"
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(path))
    import sqlite3
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version = 99")
        conn.execute(
            """
            CREATE TABLE search (
                run_id TEXT PRIMARY KEY,
                operator TEXT NOT NULL,
                task TEXT NOT NULL,
                dtype TEXT,
                started_at REAL,
                month TEXT NOT NULL,
                spent_usd REAL,
                budget_usd REAL
            )
            """
        )
        conn.execute("INSERT INTO search VALUES ('keep', 'gemm', 'gemm', 'fp16', NULL, 'unknown', 4.0, 15.0)")
    report = sync([
        {"id": "other", "task": "relu", "dtype": "fp16", "started_at": 1791315864.0, "spent_usd": 1.0, "budget_usd": 1.0},
    ])
    assert report["schema_version"] == 99
    assert report["spent_usd"] == 4.0
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 99
        assert conn.execute("SELECT COUNT(*) FROM search").fetchone()[0] == 1
        assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'schema_migration'").fetchone() is None


def test_the_book_is_not_stored_in_the_cache_directory(monkeypatch):
    monkeypatch.delenv("AI_LOWERLING_BOOKS", raising=False)
    from observe.books import REPO_ROOT, db_path
    assert db_path() == REPO_ROOT / "data" / "books.sqlite"
    assert ".cache" not in db_path().parts
