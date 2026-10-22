"""Trust metrics are descriptive, with a small-sample lower bound. No GPU, no model calls."""

from __future__ import annotations

import sqlite3

from observe.trust import metrics, wilson_lower


def _db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, target_id TEXT);
        CREATE TABLE sessions (run_id TEXT, seq INTEGER, provider TEXT, model TEXT);
        CREATE TABLE feedback_items (run_id TEXT, seq INTEGER, task TEXT);
        CREATE TABLE feedback_claims (run_id TEXT, seq INTEGER, digest TEXT, premise TEXT, status TEXT, verification TEXT, audience TEXT);
        """
    )
    conn.execute("INSERT INTO runs VALUES ('r1', 'apple-m4-max')")
    conn.execute("INSERT INTO runs VALUES ('r2', 'apple-m4-max')")
    conn.execute("INSERT INTO sessions VALUES ('r1', 1, 'deepseek', 'deepseek-flash')")
    conn.execute("INSERT INTO sessions VALUES ('r2', 1, 'deepseek', 'deepseek-flash')")
    conn.execute("INSERT INTO feedback_items VALUES ('r1', 1, 'conv2d')")
    conn.execute("INSERT INTO feedback_items VALUES ('r2', 1, 'conv2d')")
    rows = [
        ("r1", 1, "a", "verified", "done", "confirmed", "peer"),
        ("r1", 1, "b", "unverified", "rejected", "refuted", "peer"),
        ("r2", 1, "c", "verified", "adopted", "confirmed", "peer"),
        ("r2", 1, "d", "downgraded", "pending", "unchecked", "peer"),
    ]
    conn.executemany("INSERT INTO feedback_claims VALUES (?,?,?,?,?,?,?)", rows)
    return conn


def test_metrics_count_reviews_and_measured_premises():
    conn = _db()
    rows = metrics(conn, target_id="apple-m4-max", task="conv2d")
    assert len(rows) == 1
    row = rows[0]
    assert row["sessions"] == 2 and row["claims"] == 4
    assert row["reviewed"] == 3 and row["reasonable"] == 2 and row["refuted"] == 1
    assert row["pending"] == 1 and row["premise"] == 4 and row["verified_premise"] == 2
    assert row["measured_rate"] == 0.5
    assert row["reasonable_rate"] == 2 / 3
    assert row["wilson_lower"] is not None and 0 < row["wilson_lower"] < row["reasonable_rate"]
    assert row["system_reviewed"] == 0 and row["system_wilson_lower"] is None


def test_an_empty_cell_does_not_claim_a_rate():
    assert wilson_lower(0, 0) is None
    assert metrics(_db(), target_id="another") == []
