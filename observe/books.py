"""A small SQLite book of search spend.

The file lives in the repository ``data/`` directory, not in ``~/.cache``. Cache directories are cleared; this book is a record of money
already spent. Run directories remain the source of each new reading. A later reading may raise the current budget, and it may raise
the spent total, but it never lowers a spent total already stored and it never drops a search that this scan did not see.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCK = threading.Lock()
# Raise this integer when the book gains a column or a table. Each step is one function in ``_MIGRATIONS``.
# An older file is upgraded in order. A file newer than this code is left unchanged.
SCHEMA_VERSION = 1

_SEARCH_COLUMNS = (
    ("run_id", "TEXT"),
    ("operator", "TEXT NOT NULL DEFAULT ''"),
    ("task", "TEXT NOT NULL DEFAULT ''"),
    ("dtype", "TEXT"),
    ("started_at", "REAL"),
    ("month", "TEXT NOT NULL DEFAULT ''"),
    ("spent_usd", "REAL"),
    ("budget_usd", "REAL"),
)


def db_path() -> Path:
    """``data/books.sqlite`` inside the repository. ``AI_LOWERLING_BOOKS`` overrides the path in tests."""
    override = os.environ.get("AI_LOWERLING_BOOKS")
    if override:
        return Path(override)
    return REPO_ROOT / "data" / "books.sqlite"


def operator_name(task: str | None) -> str:
    """The operator a search belongs to. A tile variant (``flash_attention@64x32``) stays with its operator."""
    name = (task or "").strip()
    if not name:
        return "unknown"
    return name.split("@", 1)[0]


def month_of(started_at: float | None) -> str:
    """``YYYY-MM`` in local time. A missing start is ``unknown``, not the current month."""
    if started_at is None:
        return "unknown"
    try:
        return time.strftime("%Y-%m", time.localtime(float(started_at)))
    except (TypeError, ValueError, OSError):
        return "unknown"


def _num(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _add_column(conn: sqlite3.Connection, table: str, name: str, declaration: str) -> None:
    """Add a column when it is missing. Existing rows keep their values; SQLite fills the new column with NULL or its default."""
    if name not in _column_names(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


def _migrate_to_1(conn: sqlite3.Connection) -> None:
    """First published shape: one row per search. A file written before versions already has this table; its rows stay."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS search (
            run_id TEXT PRIMARY KEY,
            operator TEXT NOT NULL DEFAULT '',
            task TEXT NOT NULL DEFAULT '',
            dtype TEXT,
            started_at REAL,
            month TEXT NOT NULL DEFAULT '',
            spent_usd REAL,
            budget_usd REAL
        )
        """
    )
    for name, declaration in _SEARCH_COLUMNS:
        if name == "run_id":
            continue
        _add_column(conn, "search", name, declaration)


_MIGRATIONS = {
    1: ("one row per search: operator, month, spent and current budget", _migrate_to_1),
}


def _user_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0] if row else 0)


def _set_user_version(conn: sqlite3.Connection, version: int) -> None:
    # PRAGMA user_version does not accept a bound parameter. ``version`` is our own integer, not a value from the file.
    conn.execute(f"PRAGMA user_version = {int(version)}")


def _migrate(conn: sqlite3.Connection) -> int:
    """Apply every missing step. A book newer than this code is not rewritten."""
    version = _user_version(conn)
    if version > SCHEMA_VERSION:
        return version
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migration (
            version INTEGER PRIMARY KEY,
            applied_at REAL NOT NULL,
            note TEXT NOT NULL
        )
        """
    )
    for step in range(version + 1, SCHEMA_VERSION + 1):
        note, migrate = _MIGRATIONS[step]
        migrate(conn)
        conn.execute(
            "INSERT OR IGNORE INTO schema_migration (version, applied_at, note) VALUES (?, ?, ?)",
            (step, time.time(), note),
        )
        conn.commit()
        _set_user_version(conn, step)
    return SCHEMA_VERSION


def sync(rows: list[dict]) -> dict:
    """Upsert ``rows`` into the book and return the operator and month totals. Rows already stored but absent from ``rows`` stay."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, sqlite3.connect(path) as conn:
        version = _migrate(conn)
        if version <= SCHEMA_VERSION:
            conn.executemany(
            """
            INSERT INTO search (run_id, operator, task, dtype, started_at, month, spent_usd, budget_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                operator = excluded.operator,
                task = excluded.task,
                dtype = excluded.dtype,
                started_at = COALESCE(search.started_at, excluded.started_at),
                month = CASE WHEN search.started_at IS NULL THEN excluded.month ELSE search.month END,
                spent_usd = CASE
                    WHEN excluded.spent_usd IS NULL THEN search.spent_usd
                    WHEN search.spent_usd IS NULL THEN excluded.spent_usd
                    ELSE MAX(search.spent_usd, excluded.spent_usd)
                END,
                budget_usd = COALESCE(excluded.budget_usd, search.budget_usd)
            """,
            [_record(row) for row in rows if row.get("id")],
        )
        report = _report(conn)
    report["schema_version"] = version
    return report


def _record(row: dict) -> tuple:
    task = str(row.get("task") or "")
    return (
        str(row.get("id") or ""),
        operator_name(task),
        task,
        row.get("dtype") or "",
        _num(row.get("started_at")),
        month_of(_num(row.get("started_at"))),
        _num(row.get("spent_usd")),
        _num(row.get("budget_usd")),
    )


def _report(conn: sqlite3.Connection) -> dict:
    spent = conn.execute("SELECT COALESCE(SUM(spent_usd), 0), COALESCE(SUM(budget_usd), 0), COUNT(*) FROM search").fetchone()
    by_operator = [
        {"operator": op, "runs": runs, "spent_usd": round(money, 6), "budget_usd": round(allowance, 6)}
        for op, runs, money, allowance in conn.execute(
            "SELECT operator, COUNT(*), COALESCE(SUM(spent_usd), 0), COALESCE(SUM(budget_usd), 0) FROM search GROUP BY operator ORDER BY SUM(spent_usd) DESC, operator"
        )
    ]
    by_month = []
    for month, runs, money in conn.execute(
        "SELECT month, COUNT(*), COALESCE(SUM(spent_usd), 0) FROM search GROUP BY month ORDER BY month"
    ):
        operators = [
            {"operator": op, "runs": n, "spent_usd": round(part, 6)}
            for op, n, part in conn.execute(
                "SELECT operator, COUNT(*), COALESCE(SUM(spent_usd), 0) FROM search WHERE month = ? GROUP BY operator ORDER BY SUM(spent_usd) DESC, operator",
                (month,),
            )
        ]
        by_month.append({"month": month, "runs": runs, "spent_usd": round(money, 6), "operators": operators})
    return {
        "runs": spent[2],
        "spent_usd": round(spent[0], 6),
        "budget_usd": round(spent[1], 6),
        "by_operator": by_operator,
        "by_month": by_month,
        "schema_version": _user_version(conn),
    }
