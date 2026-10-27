"""Manifest, index, binding, comparison and the idle frame. No GPU and no model calls."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from experiments.manifest import ManifestError, attach, config_hash, ensure, extend_budget
from observe.index import INDEX_SCHEMA_VERSION, Index
from observe.legacy import infer
from observe.query import best_confirmed, compare, plan_outcomes, run_detail
from taic.agent_feedback import parse_feedback
from tcenv.schema import PROTOCOL_VERSION


def test_frozen_conditions_cannot_change_and_a_top_up_is_an_extension(tmp_path):
    """@brief 验证实验冻结条件不可修改，追加预算会记成 extension 而不改冻结值，重复 ensure 幂等保留已挂的 run。

    @param tmp_path pytest 提供的临时目录，用作实验目录。
    @raises ManifestError 改动 frozen（例如预算）后再次 ensure 时抛出。
    """
    frozen = {"tasks": ["relu"], "dtype": "fp16", "size": "small", "preset": "base", "model": None, "budget_usd": 3.0, "max_total_usd": None}
    arms = [
        {"arm_id": "off", "delta": {}, "planned_n": 1},
        {"arm_id": "on", "delta": {"hw_feedback": True}, "planned_n": 1},
    ]
    ensure(tmp_path, kind="search_ab", driver="search_ab", frozen=frozen, arms=arms)
    attach(tmp_path, "on", "fa_on_1")
    again = ensure(tmp_path, kind="search_ab", driver="search_ab", frozen=frozen, arms=arms)
    assert again["config_hash"] == config_hash(frozen)
    assert again["arms"][1]["runs"] == ["fa_on_1"]
    changed = dict(frozen)
    changed["budget_usd"] = 15.0
    with pytest.raises(ManifestError):
        ensure(tmp_path, kind="search_ab", driver="search_ab", frozen=changed, arms=arms)
    extend_budget(tmp_path, run="fa_on_1", new_cap=18.0, at=10)
    doc = json.loads((tmp_path / "experiment.json").read_text())
    assert doc["frozen"]["budget_usd"] == 3.0
    assert doc["budget_extensions"] == [{"run": "fa_on_1", "new_cap": 18.0, "at": 10}]
    assert doc["config_hash"] == config_hash(frozen)


def test_two_arms_share_a_config_hash_and_differ_by_the_switch_delta(tmp_path):
    """@brief 验证同一实验的两个 arm 共享一个 config_hash，差异只体现在各自的开关 delta 上。

    @param tmp_path pytest 提供的临时目录，用作实验目录。
    """
    frozen = {"tasks": ["flash_attention"], "dtype": "fp16", "budget_usd": 3.0}
    ensure(
        tmp_path,
        kind="search_ab",
        driver="search_ab",
        frozen=frozen,
        arms=[
            {"arm_id": "off", "delta": {}, "planned_n": 1},
            {"arm_id": "on", "delta": {"truncation_retry": True}, "planned_n": 1},
        ],
    )
    doc = json.loads((tmp_path / "experiment.json").read_text())
    assert doc["config_hash"] == config_hash(frozen)
    assert doc["arms"][0]["delta"] == {}
    assert doc["arms"][1]["delta"] == {"truncation_retry": True}


def test_legacy_names_and_a_truncated_result_do_not_fail_the_index(tmp_path, monkeypatch):
    """@brief 验证旧目录命名能被推断出 arm/trial/kind，而 result.json 截断时索引仍能完成同步。

    推断行标为 ``inferred`` 且 binding 为 none；损坏的 run 标为 missing；样本不足时 compare 不下判定。

    @param tmp_path pytest 提供的临时目录，用来放 runs 目录与索引库。
    @param monkeypatch pytest 的环境变量夹具，用于指向临时 runs 目录与临时数据库。
    """
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    parent = runs / "search_ab"
    child = parent / "fa_on_1"
    child.mkdir(parents=True)
    (child / "state.json").write_text(json.dumps({"task": "flash_attention", "dtype": "fp16", "phase": "done", "started_at": 10, "updated_at": 10}))
    guessed = infer(child, runs)
    assert guessed["arm_id"] == "on" and guessed["trial"] == 1 and guessed["kind"] == "search_ab"
    broken = runs / "broken"
    broken.mkdir()
    (broken / "result.json").write_text("{not json")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    row = index.conn.execute("SELECT source, binding, status FROM runs WHERE run_id = 'search_ab/fa_on_1'").fetchone()
    assert row["source"] == "inferred" and row["binding"] == "none" and row["status"] == "done"
    missing = index.conn.execute("SELECT status FROM runs WHERE run_id = 'broken'").fetchone()
    assert missing["status"] == "missing"
    compared = compare("search_ab")
    assert compared["verdict"] == "不下判定"


def test_schema_mismatch_rebuilds(tmp_path, monkeypatch):
    """@brief 验证索引库 schema 版本不匹配时会重建，并把版本写回当前 ``INDEX_SCHEMA_VERSION``。

    @param tmp_path pytest 提供的临时目录，用来放索引库。
    @param monkeypatch pytest 的环境变量夹具，用于指向空的配置路径。
    """
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    path = tmp_path / "observe.sqlite"
    import sqlite3
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO meta VALUES ('index_schema_version', '99')")
    index = Index(path)
    version = index.conn.execute("SELECT value FROM meta WHERE key = 'index_schema_version'").fetchone()[0]
    assert version == str(INDEX_SCHEMA_VERSION)


def _schema_11_with_a_ledger(path: Path) -> None:
    """@brief 造一个版本 11、带一条 2.5 美元账本的旧库。

    @param path 数据库路径。
    """
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO meta VALUES ('index_schema_version', '11')")
        conn.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, task TEXT, status TEXT)")
        conn.execute("INSERT INTO runs VALUES ('conv2d/old', 'conv2d', 'done')")
        conn.execute("""CREATE TABLE ledger (
            run_id TEXT PRIMARY KEY, operator TEXT, month TEXT, started_at REAL,
            spent_max_usd REAL, budget_usd REAL
        )""")
        conn.execute("INSERT INTO ledger VALUES ('conv2d/old', 'kuafu', '2026-10', 1.0, 2.5, 3.0)")


def test_a_ledger_write_failure_aborts_the_rebuild_and_keeps_the_old_database(tmp_path, monkeypatch):
    """@brief PR6：旁路账本写失败必须中止重建、保留旧库；修复后重开仍能恢复 2.5 美元。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    import observe.index as index_mod

    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    path = tmp_path / "observe.sqlite"
    _schema_11_with_a_ledger(path)
    original = path.read_bytes()
    real_replace = index_mod.os.replace

    def denied(*_args, **_kwargs):
        raise PermissionError("read-only volume")

    monkeypatch.setattr(index_mod.os, "replace", denied)
    with pytest.raises(index_mod.CarryoverError):
        index_mod.Index(path)
    assert path.exists() and path.read_bytes() == original, "保全失败时不能删旧库"
    assert not index_mod.preserved_ledger_path(path).exists()

    monkeypatch.setattr(index_mod.os, "replace", real_replace)
    index = index_mod.Index(path)
    assert index.total_usd() == pytest.approx(2.5)
    assert index.conn.execute("SELECT value FROM meta WHERE key = 'index_schema_version'").fetchone()[0] == "12"


def test_a_ledger_read_failure_aborts_the_rebuild_and_keeps_the_old_database(tmp_path, monkeypatch):
    """@brief PR6：旧库账本读不出来（锁/损坏）也必须中止重建，不能当作空账本丢掉。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    import observe.index as index_mod

    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    path = tmp_path / "observe.sqlite"
    _schema_11_with_a_ledger(path)
    original = path.read_bytes()
    real_connect = sqlite3.connect

    def locked(target, *args, **kwargs):
        if isinstance(target, str) and "mode=ro" in target:
            raise sqlite3.OperationalError("database is locked")
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", locked)
    with pytest.raises(index_mod.CarryoverError):
        index_mod.Index(path)
    assert path.exists() and path.read_bytes() == original
    monkeypatch.setattr(sqlite3, "connect", real_connect)
    assert index_mod.Index(path).total_usd() == pytest.approx(2.5)


def test_an_old_schema_database_is_discarded_and_rebuilt_with_its_ledger(tmp_path, monkeypatch):
    """@brief §12.1 老 schema 迁移：版本 11 的库删库重建，账本行从旁路恢复，schema 12 结构齐全。

    跨版本没有就地迁移：重建后旧 ``runs`` 行消失（由发现重新摄），但已发生的花费不能消失。

    @param tmp_path pytest 提供的临时目录，用来放旧库。
    @param monkeypatch pytest 的环境变量夹具，用于指向空的配置路径。
    """
    import sqlite3

    from observe.index import preserved_ledger_path

    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    path = tmp_path / "observe.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO meta VALUES ('index_schema_version', '11')")
        conn.execute("""CREATE TABLE runs (
            run_id TEXT PRIMARY KEY, task TEXT, status TEXT, publication TEXT, published_speedup REAL
        )""")
        conn.execute("INSERT INTO runs VALUES ('conv2d/old', 'conv2d', 'done', 'confirmed', 1.5)")
        conn.execute("""CREATE TABLE ledger (
            run_id TEXT PRIMARY KEY, operator TEXT, month TEXT, started_at REAL,
            spent_max_usd REAL, budget_usd REAL
        )""")
        conn.execute("INSERT INTO ledger VALUES ('conv2d/old', 'kuafu', '2026-10', 1.0, 2.5, 3.0)")

    index = Index(path)
    version = index.conn.execute("SELECT value FROM meta WHERE key = 'index_schema_version'").fetchone()[0]
    assert version == str(INDEX_SCHEMA_VERSION) == "12"
    assert index.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0, "旧行随旧库一起丢掉，由发现重建"
    assert index.total_usd() == pytest.approx(2.5), "重建不能丢掉已记录的花费"
    assert not preserved_ledger_path(path).exists(), "回放后旁路文件应该被删掉"

    columns = {row[1] for row in index.conn.execute("PRAGMA table_info(runs)")}
    assert {"search_best_speedup", "stop_code", "control_action", "control_reason",
            "topped_up", "initial_budget_usd"} <= columns
    turns = {row[1] for row in index.conn.execute("PRAGMA table_info(turns)")}
    assert "grp" in turns
    tables = {row[0] for row in index.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"run_pending_eval", "confirmed_best", "failure_members", "kernel_history", "run_rollup",
            "gpu_events", "lock_spans", "blob_refs", "runs_fts", "cands_fts"} <= tables
    indexes = {row[0] for row in index.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert {"candidates_sig", "sessions_sig", "turns_run_grp"} <= indexes


def test_session_diagnostic_columns_are_added_without_changing_the_index_epoch(tmp_path, monkeypatch):
    """@brief 验证旧 sessions 表缺诊断列时就地补齐，且不抬高 index schema 版本（不触发重建）。

    @param tmp_path pytest 提供的临时目录，用来放索引库。
    @param monkeypatch pytest 的环境变量夹具，用于指向空的配置路径。
    """
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    path = tmp_path / "observe.sqlite"
    import sqlite3
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO meta VALUES ('index_schema_version', ?)", (str(INDEX_SCHEMA_VERSION),))
        conn.execute("""CREATE TABLE sessions (
            run_id TEXT, seq INTEGER, label TEXT, committee TEXT, provider TEXT, model TEXT,
            outcome TEXT, cost_usd REAL, tokens_input INTEGER, tokens_output INTEGER,
            tokens_cache_read INTEGER, tokens_cache_write INTEGER, t REAL
        )""")
    index = Index(path)
    columns = {row[1] for row in index.conn.execute("PRAGMA table_info(sessions)")}
    assert {"candidate_label", "attempt", "session_id", "failed_turns", "last_kind", "failure_class", "error", "tool_calls", "diagnostics_path"} <= columns
    assert index.conn.execute("SELECT value FROM meta WHERE key = 'index_schema_version'").fetchone()[0] == str(INDEX_SCHEMA_VERSION)


def _schema_index(tmp_path, monkeypatch) -> Index:
    """@brief 建一个临时索引库（不扫 runs/）。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的环境变量夹具。
    @return 新建的 ``Index``。
    """
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    return Index(tmp_path / "observe.sqlite")


def test_schema_12_declares_every_new_column_index_and_table(tmp_path, monkeypatch):
    """@brief §12.1：schema 12 的新列、新索引与新表都在建库时出现。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的环境变量夹具。
    """
    index = _schema_index(tmp_path, monkeypatch)
    assert INDEX_SCHEMA_VERSION == 12
    assert index.conn.execute("SELECT value FROM meta WHERE key = 'index_schema_version'").fetchone()[0] == "12"

    def columns(table):
        return {row[1]: row for row in index.conn.execute(f"PRAGMA table_info({table})")}

    runs = columns("runs")
    assert {"search_best_speedup", "stop_code", "control_action", "control_reason", "topped_up",
            "initial_budget_usd"} <= set(runs)
    # PR4 不依赖新列：result-only 的目录 mtime 依赖折进 schema-11 就有的 fp_state。
    assert "dir_mtime_ns" not in runs
    assert {"message", "failure_sig", "cv", "reps", "cache_hit"} <= set(columns("candidates"))
    assert "failure_sig" in columns("sessions")
    turns = columns("turns")
    assert turns["grp"][4] == "0" and turns["grp"][3] == 1  # dflt_value / notnull

    expected_tables = {
        "failure_members": ["run_id", "idx", "sig", "status", "stage", "observed_at"],
        "kernel_history": ["target_id", "task", "dtype", "size", "seq", "run_id_raw", "run_id", "winner_label",
                           "source_sha256", "confirmed_speedup", "confirmed_values_json", "candidate_us",
                           "baseline_us_json", "task_fingerprint", "calibration_sha256", "updated_at",
                           "source_unknown"],
        "run_rollup": ["day", "task", "dtype", "size", "target_id", "protocol_epoch", "runs", "confirmed",
                       "spent_usd", "published_sum", "published_max"],
        "confirmed_best": ["run_id", "task", "dtype", "size", "target_id", "protocol_epoch",
                           "published_speedup", "kernel_sha256", "confirmed_at"],
        "run_pending_eval": ["run_id", "pending", "unavailable", "last_error_code", "retryable",
                             "requests_json", "updated_at"],
        "gpu_events": ["t", "pid", "kind", "mode", "what"],
        "gpu_waits": ["pid", "t", "mode", "what", "seq"],
        "lock_spans": ["pid", "start", "end", "closed_at", "mode", "what"],
        "blob_refs": ["sha256", "ref_kind", "ref_id", "created_at"],
        "run_parents": ["path"],
    }
    for table, expected in expected_tables.items():
        assert list(columns(table)) == expected, table
    for table, key in (
        ("failure_members", ["run_id", "idx"]),
        ("kernel_history", ["target_id", "task", "dtype", "size", "seq"]),
        ("run_rollup", ["day", "task", "dtype", "size", "target_id", "protocol_epoch"]),
        ("confirmed_best", ["run_id"]),
        ("run_pending_eval", ["run_id"]),
        ("gpu_waits", ["pid"]),
        ("blob_refs", ["sha256", "ref_kind", "ref_id"]),
        ("run_parents", ["path"]),
    ):
        pk = [(row[1], row[5]) for row in index.conn.execute(f"PRAGMA table_info({table})") if row[5]]
        assert [name for name, _rank in sorted(pk, key=lambda item: item[1])] == key, table

    indexes = {row[0] for row in index.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert {"candidates_sig", "sessions_sig", "turns_run_grp", "lock_spans_window", "gpu_events_pid",
            "runs_budget_cover"} <= indexes
    # 有界区间读的桶索引：span_id 是 lock_spans.rowid；超长 span 只进单独的小表。
    slots = columns("lock_span_slots")
    assert [name for name in slots] == ["bucket", "span_id"]
    assert [name for name in columns("lock_spans_long")] == ["span_id"]
    triggers = {row[0] for row in index.conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")}
    assert {"runs_fts_ai", "runs_fts_au", "runs_fts_ad", "cands_fts_ai", "cands_fts_au", "cands_fts_ad"} <= triggers
    # §10.3 的列形状：runs 投影 run_id/task/dtype/target_id/model/stop_reason；候选投影 label/message。
    assert index._fts_columns("runs_fts") == ("run_id", "task", "dtype", "target_id", "model", "stop_reason")
    assert index._fts_columns("cands_fts") == ("label", "message")


def test_the_runs_fts_projection_covers_the_six_documented_fields(tmp_path, monkeypatch):
    """@brief §10.3：``runs_fts`` 逐字段可检索（run_id/task/dtype/target_id/model/stop_reason），不再索引 operator。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的环境变量夹具。
    """
    index = _schema_index(tmp_path, monkeypatch)
    index.conn.execute(
        "INSERT INTO runs (run_id, task, dtype, target_id, model, stop_reason) "
        "VALUES ('conv2d/one', 'conv2d', 'fp16', 'apple-m4-max', 'deepseek-flash', 'max_rounds')"
    )
    for column, term in (
        ("run_id", "one"), ("task", "conv2d"), ("dtype", "fp16"),
        ("target_id", "apple"), ("model", "deepseek"), ("stop_reason", "rounds"),
    ):
        hits = index.conn.execute(
            "SELECT run_id FROM runs_fts WHERE runs_fts MATCH ?", (term,)
        ).fetchall()
        assert [row[0] for row in hits] == ["conv2d/one"], column
    # operator 不再进投影：搜它既命不中，也不代表运行内容。
    assert index.conn.execute("SELECT COUNT(*) FROM runs_fts WHERE runs_fts MATCH 'kuafu'").fetchone()[0] == 0


def test_a_wrong_or_partial_fts_projection_is_repaired_on_open(tmp_path, monkeypatch):
    """@brief 同版本的旧投影（operator 当列、触发器缺失）在下次打开时被修好并重建，已有行可检索。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的环境变量夹具。
    """
    import sqlite3

    from observe.index import db_path

    index = _schema_index(tmp_path, monkeypatch)
    index.conn.execute(
        "INSERT INTO runs (run_id, task, dtype, target_id, model, stop_reason, operator) "
        "VALUES ('conv2d/one', 'conv2d', 'fp16', 'apple-m4-max', 'deepseek-flash', 'max_rounds', 'kuafu')"
    )
    index.conn.commit()
    db = db_path()
    index.conn.close()

    # 造出上一版的错形状：runs_fts 索引 operator，触发器只建了一个旧列顺序的 INSERT。
    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE runs_fts")
    conn.execute(
        "CREATE VIRTUAL TABLE runs_fts USING fts5(run_id, task, operator, model, "
        "content='runs', content_rowid='rowid', tokenize='unicode61')"
    )
    conn.execute("DROP TRIGGER IF EXISTS runs_fts_ai")
    conn.execute("DROP TRIGGER IF EXISTS runs_fts_ad")
    conn.execute("DROP TRIGGER IF EXISTS runs_fts_au")
    conn.execute("INSERT INTO runs_fts(runs_fts) VALUES ('rebuild')")
    conn.execute(
        "CREATE TRIGGER runs_fts_ai AFTER INSERT ON runs BEGIN "
        "INSERT INTO runs_fts(rowid, run_id, task, operator, model) "
        "VALUES (new.rowid, new.run_id, new.task, new.operator, new.model); END"
    )
    conn.commit()
    conn.close()

    repaired = Index(db)
    assert repaired._fts_columns("runs_fts") == ("run_id", "task", "dtype", "target_id", "model", "stop_reason")
    triggers = {row[0] for row in repaired.conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")}
    assert {"runs_fts_ai", "runs_fts_au", "runs_fts_ad"} <= triggers
    # 修复时从内容表重建过：已有的那一行按新列可命中。
    for term in ("conv2d", "fp16", "apple", "rounds"):
        assert repaired.conn.execute("SELECT COUNT(*) FROM runs_fts WHERE runs_fts MATCH ?", (term,)).fetchone()[0] == 1
    assert repaired.conn.execute("SELECT COUNT(*) FROM runs_fts WHERE runs_fts MATCH 'kuafu'").fetchone()[0] == 0

    # 新触发器跟随写入：dtype/target/stop_reason 的改动都能检索到。
    repaired.conn.execute(
        "UPDATE runs SET dtype = 'bf16', target_id = 'apple-m5', stop_reason = 'budget_exhausted' WHERE run_id = 'conv2d/one'"
    )
    for term in ("bf16", "apple", "exhausted"):
        assert repaired.conn.execute("SELECT COUNT(*) FROM runs_fts WHERE runs_fts MATCH ?", (term,)).fetchone()[0] == 1
    assert repaired.conn.execute("SELECT COUNT(*) FROM runs_fts WHERE runs_fts MATCH 'fp16'").fetchone()[0] == 0


def _candidate_run(root: Path, name: str = "conv2d/fts") -> Path:
    """@brief 造一个带两颗同序号候选（``000_alpha`` / ``000_beta``）的运行，用来打候选 upsert 的冲突路径。

    @param root runs 根目录。
    @param name run id。
    @return 新建的 run 目录。
    """
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "phase": "done", "started_at": 10.0, "updated_at": 20.0,
    }), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "size": "paper", "stop_reason": "max_rounds",
        "started_at": 10.0, "spent_usd": 0.1, "best_speedup": 1.0,
    }), encoding="utf-8")
    for suffix, token in (("alpha", "oldtoken"), ("beta", "newtoken")):
        folder = directory / "candidates" / f"000_{suffix}"
        folder.mkdir(parents=True)
        (folder / "kernel.metal").write_text("kernel void k() {}\n", encoding="utf-8")
        (folder / "report.json").write_text(json.dumps({
            "status": "ok", "stage": "bench", "speedup": 1.0, "message": token,
        }), encoding="utf-8")
    return directory


def test_candidate_upsert_on_the_same_idx_keeps_the_fts_projection_consistent(tmp_path, monkeypatch):
    """@brief §PR6：同一 ``(run_id, idx)`` 冲突走 ON CONFLICT，不触发 REPLACE 的隐式删除；FTS 仍一致。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的环境变量夹具。
    """
    import sqlite3

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    index = Index(tmp_path / "observe.sqlite")
    directory = _candidate_run(runs)
    index.sync_known()
    # 两颗候选的 idx 都是 0：后一颗必须更新前一颗的行，而不是留下一条没有内容行的 FTS 记录。
    rows = [dict(row) for row in index.conn.execute("SELECT idx, label FROM candidates")]
    assert len(rows) == 1 and rows[0]["label"] == "beta"
    index.conn.execute("INSERT INTO cands_fts(cands_fts) VALUES ('integrity-check')")

    # 再摄取一次（报告内容不变）也不能把投影写坏。
    index.sync_known()
    index.conn.execute("INSERT INTO cands_fts(cands_fts) VALUES ('integrity-check')")
    assert index.conn.execute("SELECT COUNT(*) FROM cands_fts WHERE cands_fts MATCH 'beta'").fetchone()[0] == 1


def test_drop_archived_removes_candidates_and_their_fts_rows(tmp_path, monkeypatch):
    """@brief §PR6：归档删除要同时清掉 candidates 与 cands_fts，之后还能继续写。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的环境变量夹具。
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    index = Index(tmp_path / "observe.sqlite")
    directory = _candidate_run(runs, "conv2d/gone")
    index.sync_known()
    assert index.conn.execute("SELECT COUNT(*) FROM cands_fts WHERE cands_fts MATCH 'beta'").fetchone()[0] == 1

    index.drop_archived(["conv2d/gone"], [])
    assert index.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert index.conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0
    assert index.conn.execute("SELECT COUNT(*) FROM cands_fts WHERE cands_fts MATCH 'beta'").fetchone()[0] == 0
    index.conn.execute("INSERT INTO cands_fts(cands_fts) VALUES ('integrity-check')")

    # 删除之后再导入同一目录：投影要能重新接收新行。
    index._ingest(directory, force=True)
    assert index.conn.execute("SELECT COUNT(*) FROM cands_fts WHERE cands_fts MATCH 'beta'").fetchone()[0] == 1
    index.conn.execute("INSERT INTO cands_fts(cands_fts) VALUES ('integrity-check')")


def test_a_same_column_but_contentless_or_corrupt_fts_projection_is_repaired(tmp_path, monkeypatch):
    """@brief §PR6：列名相同但 ``content`` 声明错误/内容损坏的投影，也要在打开时被修好并重建。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的环境变量夹具。
    """
    import sqlite3

    from observe.index import db_path

    index = _schema_index(tmp_path, monkeypatch)
    index.conn.execute(
        "INSERT INTO runs (run_id, task, dtype, target_id, model, stop_reason) "
        "VALUES ('conv2d/one', 'conv2d', 'fp16', 'apple-m4-max', 'deepseek-flash', 'max_rounds')"
    )
    index.conn.commit()
    db = db_path()
    index.conn.close()

    # 列名相同，但声明成 contentless（没有 content）；旧代码只看列名，会永远跳过 rebuild。
    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE runs_fts")
    conn.execute(
        "CREATE VIRTUAL TABLE runs_fts USING fts5(run_id, task, dtype, target_id, model, stop_reason, tokenize='unicode61')"
    )
    conn.execute("INSERT INTO runs_fts(run_id, task, dtype, target_id, model, stop_reason) "
                 "VALUES ('conv2d/one', 'conv2d', 'fp16', 'apple-m4-max', 'deepseek-flash', 'max_rounds')")
    conn.commit()
    conn.close()

    repaired = Index(db)
    assert repaired._fts_columns("runs_fts") == ("run_id", "task", "dtype", "target_id", "model", "stop_reason")
    assert repaired.conn.execute("SELECT COUNT(*) FROM runs_fts WHERE runs_fts MATCH 'fp16'").fetchone()[0] == 1
    repaired.conn.execute("INSERT INTO runs_fts(runs_fts) VALUES ('integrity-check')")

    # 内容损坏：往 external-content 投影里塞一条没有内容行的 phantom row，重开时必须被修掉。
    repaired.conn.execute(
        "INSERT INTO runs_fts(rowid, run_id, task, dtype, target_id, model, stop_reason) "
        "VALUES (99999, 'phantom', 'conv2d', 'fp16', 'apple-m4-max', 'm', 'max_rounds')"
    )
    repaired.conn.commit()
    repaired.conn.close()

    again = Index(db)
    assert again.conn.execute("SELECT COUNT(*) FROM runs_fts WHERE runs_fts MATCH 'phantom'").fetchone()[0] == 0
    assert again.conn.execute("SELECT COUNT(*) FROM runs_fts WHERE runs_fts MATCH 'fp16'").fetchone()[0] == 1
    again.conn.execute("INSERT INTO runs_fts(runs_fts) VALUES ('integrity-check')")

    # 同 rowid、错 token：把 rowid 1 的索引项换成另一个 token（行数、rowid 都不变）。
    again.conn.execute("INSERT INTO runs_fts(runs_fts, rowid, run_id, task, dtype, target_id, model, stop_reason) "
                       "VALUES ('delete', 1, 'conv2d/one', 'conv2d', 'fp16', 'apple-m4-max', 'deepseek-flash', 'max_rounds')")
    again.conn.execute("INSERT INTO runs_fts(rowid, run_id, task, dtype, target_id, model, stop_reason) "
                       "VALUES (1, 'conv2d/one', 'conv2d', 'wrongtoken', 'apple-m4-max', 'deepseek-flash', 'max_rounds')")
    again.conn.commit()
    again.conn.close()

    repaired_swap = Index(db)
    assert repaired_swap.conn.execute(
        "SELECT COUNT(*) FROM runs_fts WHERE runs_fts MATCH 'wrongtoken'"
    ).fetchone()[0] == 0, "指向错误 token 的索引必须在打开时被重建掉"
    rows = sorted(row[0] for row in repaired_swap.conn.execute(
        "SELECT rowid FROM runs_fts WHERE runs_fts MATCH 'fp16'"
    ))
    assert rows == [1], f"真实 token 必须恢复且只指向 rowid 1，实际 {rows}"
    content_rows = sorted(row[0] for row in repaired_swap.conn.execute("SELECT rowid FROM runs WHERE rowid IN (1, 2)"))
    index_rows = sorted(row[0] for row in repaired_swap.conn.execute(
        "SELECT rowid FROM runs_fts WHERE runs_fts MATCH 'conv2d'"
    ))
    assert index_rows == content_rows, "源表与投影的 rowid 集合必须精确一致"


def _confirmed_schema_run(root: Path) -> Path:
    """@brief 造一个能通过发布门的 run，让重建对拍里的 ``confirmed_best`` 非空。

    @param root runs 根目录。
    @return run 目录。
    """
    import hashlib

    from tcenv.schema import PROTOCOL_VERSION

    source = "kernel void k() {}\n"
    digest = hashlib.sha256(source.encode()).hexdigest()
    directory = root / "conv2d" / "confirmed"
    (directory / "candidates" / "000_r1.p0").mkdir(parents=True)
    (directory / "best.metal").write_text(source, encoding="utf-8")
    (directory / "candidates" / "000_r1.p0" / "kernel.metal").write_text(source, encoding="utf-8")
    (directory / "candidates" / "000_r1.p0" / "report.json").write_text(
        json.dumps({"status": "ok", "stage": "bench", "speedup": 1.1}), encoding="utf-8"
    )
    (directory / "state.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "phase": "done", "started_at": 10.0, "updated_at": 20.0,
    }), encoding="utf-8")
    (directory / "run_config.json").write_text(json.dumps({
        "target": {"id": "apple-m4-max", "backend": "metal", "backend_version": "v1"},
    }), encoding="utf-8")
    values = [1.1]
    (directory / "result.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "size": "paper", "stop_reason": "max_rounds",
        "started_at": 10.0, "spent_usd": 0.2, "target_id": "apple-m4-max",
        "confirmed_speedup": 1.1, "confirmed_status": "ok", "confirm_winner_label": "r1.p0",
        "confirmation_source_sha256": digest, "confirmed_values": values,
        "confirmed_worst": 1.1, "confirmed_best": 1.1,
    }), encoding="utf-8")
    (directory / "confirm_report.json").write_text(json.dumps({
        "status": "ok", "speedup": 1.1,
        "benchmark": {"noisy": False, "candidate": {"cv": 0.01, "reps": 15},
                      "baselines": {"mlx": {"cv": 0.01, "reps": 15}}, "best_baseline": "mlx",
                      "confirmation": {"kernel_sha256": digest, "winner_label": "r1.p0",
                                       "confirmed_values": values, "speedup": 1.1, "candidate_us": None}},
        "protocol_version": PROTOCOL_VERSION, "target_id": "apple-m4-max",
    }), encoding="utf-8")
    return directory


def _schema_run(root: Path) -> Path:
    """@brief 造一个带候选与转录的已结束 run，供 schema 重建对拍。

    @param root runs 根目录。
    @return 新建的 run 目录。
    """
    directory = root / "conv2d" / "schema"
    folder = directory / "candidates" / "000_r1.p0"
    folder.mkdir(parents=True)
    (folder / "kernel.metal").write_text("kernel void k() {}\n", encoding="utf-8")
    (folder / "report.json").write_text(json.dumps({
        "status": "ok", "stage": "bench", "speedup": 1.2, "message": "coalesced epilogue stores",
        "benchmark": {"noise_cv": 0.01, "attempts": 3}, "candidate_us": 700.0,
    }), encoding="utf-8")
    (directory / "state.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "phase": "done", "started_at": 10.0, "updated_at": 20.0,
        "spent_usd": 0.4, "budget_usd": 2.0, "operator": "kuafu",
    }), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "size": "paper", "stop_reason": "max_rounds", "started_at": 10.0,
        "spent_usd": 0.5, "budget_usd": 2.0,
    }), encoding="utf-8")
    (directory / "transcript.jsonl").write_text(json.dumps({
        "event": "agent", "label": "r1.p0", "candidate_label": "r1.p0", "provider": "deepseek",
        "model": "deepseek-flash", "committee": "first", "outcome": "evaluated", "cost_usd": 0.25,
        "tokens": {"input": 4, "output": 8},
    }) + "\n", encoding="utf-8")
    parsed = parse_feedback("kind: tool\n\n# 需要帮助\n- 缺一把探针\n")
    (directory / "feedback.jsonl").write_text(json.dumps({
        "format": "agent_feedback", "v": 1, "seq": 1, "t": 11.0, "task": "conv2d", "dtype": "fp16",
        "size": "paper", "label": "r1.p0", "round": 1, "committee": "first", "provider": "deepseek",
        "model": "deepseek-flash", "outcome": "evaluated", "kinds": parsed["kinds"],
        "sections": parsed["sections"], "items": parsed["items"],
        "unverified_premises": parsed["unverified_premises"], "raw": parsed["raw"],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    return directory


# 这些表由 SQLite 自己维护（FTS5 影子表），重建对拍时比它们没有意义。
_SHADOW_PREFIXES = ("runs_fts_", "cands_fts_")


def _snapshot(index: Index) -> dict:
    """@brief 对拍用的逐字段快照：``sqlite_master`` 里每张业务表按 rowid 排序后的全部列。

    包含 schema 12 新建的表（空表也要出现在字典里），但不比 FTS5 的影子表。

    @param index 索引实例。
    @return 表名到行列表的字典。
    """
    tables = [
        row[0] for row in index.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    out = {}
    for table in tables:
        # meta 里有 built_at/偏移这类每次建库都会变的状态；它不承载 runs/ 的事实。
        if table == "meta" or table.startswith(_SHADOW_PREFIXES) or table in ("runs_fts", "cands_fts"):
            continue
        # ``lock_span_slots`` 是 WITHOUT ROWID，不能统一按 rowid 排；下面的 sorted 已经给出稳定顺序。
        out[table] = sorted(
            (dict(row) for row in index.conn.execute(f"SELECT * FROM {table}")),
            key=lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False),
        )
    return out


def test_dropping_the_database_and_rebuilding_gives_the_same_rows(tmp_path, monkeypatch):
    """@brief §12.1：删库重建后逐字段一致，包括新列与被触发器维护的 FTS 投影。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的环境变量夹具。
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_TICK_INTERVAL", "0")
    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "3600")
    directory = _schema_run(runs)
    _confirmed_schema_run(runs)
    db = tmp_path / "observe.sqlite"
    first = Index(db)
    first.sync_known()
    first.ensure_turns(directory)  # 已结束的 run 按需投影轮次；这里要先有行才能对拍 grp。
    before = _snapshot(first)
    assert len(before["confirmed_best"]) == 1, "重建对拍必须覆盖非空的 confirmed_best"
    assert before["runs"][0]["run_id"] == "conv2d/schema"
    # PR9 给所有候选写脱敏 message；只有失败候选有 failure_sig。
    by_run = {row["run_id"]: row for row in before["candidates"]}
    assert by_run["conv2d/schema"]["message"] == "coalesced epilogue stores"
    assert all(row["failure_sig"] is None for row in before["candidates"])
    assert by_run["conv2d/schema"]["cv"] is None  # 夹具的 benchmark 没有 candidate.cv
    assert before["turns"][0]["grp"] == 1, "PR8 投影从 1 开始；相邻同标签共组"
    fts_before = sorted(tuple(row) for row in first.conn.execute(
        "SELECT run_id FROM runs_fts WHERE runs_fts MATCH 'conv2d'"
    ))
    cands_before = sorted(tuple(row) for row in first.conn.execute(
        "SELECT label FROM cands_fts WHERE cands_fts MATCH 'p0'"
    ))
    assert fts_before == [("conv2d/confirmed",), ("conv2d/schema",)]
    assert cands_before == [("r1.p0",), ("r1.p0",)]
    first.conn.close()

    for suffix in ("", "-wal", "-shm"):
        path = Path(str(db) + suffix) if suffix else db
        if path.exists():
            path.unlink()
    second = Index(db)
    second.sync_known()
    second.ensure_turns(directory)
    after = _snapshot(second)
    assert after == before
    assert sorted(tuple(row) for row in second.conn.execute(
        "SELECT run_id FROM runs_fts WHERE runs_fts MATCH 'conv2d'"
    )) == fts_before
    assert sorted(tuple(row) for row in second.conn.execute(
        "SELECT label FROM cands_fts WHERE cands_fts MATCH 'p0'"
    )) == cands_before


def test_missing_cleanup_clears_candidates_fts_and_confirmed_best(tmp_path, monkeypatch):
    """@brief 物理删除、running→missing、broken run 都走同一套清理：发布字段、候选、FTS 与物化表。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的环境变量夹具。
    """
    import os
    import shutil

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_TICK_INTERVAL", "0")
    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "3600")
    index = Index(tmp_path / "observe.sqlite")

    confirmed = _confirmed_schema_run(runs)
    cands = _candidate_run(runs, "conv2d/cands")
    index.sync_known()
    assert index.conn.execute("SELECT COUNT(*) FROM confirmed_best").fetchone()[0] == 1
    assert index.conn.execute("SELECT COUNT(*) FROM cands_fts WHERE cands_fts MATCH 'beta'").fetchone()[0] == 1

    # 物理删除。
    shutil.rmtree(confirmed)
    shutil.rmtree(cands)
    index.sync_known()
    assert index.conn.execute("SELECT COUNT(*) FROM confirmed_best").fetchone()[0] == 0
    assert index.conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0
    assert index.conn.execute("SELECT COUNT(*) FROM cands_fts WHERE cands_fts MATCH 'beta'").fetchone()[0] == 0

    # running → missing。
    live = runs / "conv2d" / "live"
    (live / "candidates" / "000_r1").mkdir(parents=True)
    (live / "state.json").write_text(json.dumps({
        "pid": os.getpid(), "phase": "plan", "started_at": 10.0, "updated_at": 20.0,
        "task": "conv2d", "dtype": "fp16",
    }), encoding="utf-8")
    index.sync_known()
    assert index.conn.execute("SELECT status FROM runs WHERE run_id = 'conv2d/live'").fetchone()[0] == "running"
    shutil.rmtree(live)
    index.tick()
    assert index.conn.execute("SELECT COUNT(*) FROM candidates WHERE run_id = 'conv2d/live'").fetchone()[0] == 0
    row = index.conn.execute("SELECT status, publication FROM runs WHERE run_id = 'conv2d/live'").fetchone()
    assert (row["status"], row["publication"]) == ("missing", None)

    # broken run（两份快照都读不出来）。
    broken = runs / "conv2d" / "broken"
    (broken / "candidates" / "000_r1").mkdir(parents=True)
    (broken / "result.json").write_text('{"task": "conv2d", ', encoding="utf-8")
    index.sync_known()
    row = index.conn.execute("SELECT status, publication FROM runs WHERE run_id = 'conv2d/broken'").fetchone()
    assert (row["status"], row["publication"]) == ("missing", None)
    assert index.conn.execute("SELECT COUNT(*) FROM candidates WHERE run_id = 'conv2d/broken'").fetchone()[0] == 0


def test_the_fts_triggers_follow_insert_update_and_delete(tmp_path, monkeypatch):
    """@brief FTS 触发器覆盖插入、就地改写与删除；先旧后新的值都要从投影里消失。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的环境变量夹具。
    """
    index = _schema_index(tmp_path, monkeypatch)
    run = {
        "run_id": "r1", "status": "done", "task": "conv2d", "operator": "kuafu", "model": "m1",
    }
    columns = ", ".join(run)
    marks = ", ".join("?" for _ in run)
    index.conn.execute(f"INSERT INTO runs ({columns}) VALUES ({marks})", tuple(run.values()))
    assert index.conn.execute("SELECT COUNT(*) FROM runs_fts WHERE runs_fts MATCH 'conv2d'").fetchone()[0] == 1
    index.conn.execute("UPDATE runs SET task = 'relu', operator = 'other' WHERE run_id = 'r1'")
    assert index.conn.execute("SELECT COUNT(*) FROM runs_fts WHERE runs_fts MATCH 'conv2d'").fetchone()[0] == 0
    assert index.conn.execute("SELECT COUNT(*) FROM runs_fts WHERE runs_fts MATCH 'relu'").fetchone()[0] == 1
    index.conn.execute("DELETE FROM runs WHERE run_id = 'r1'")
    assert index.conn.execute("SELECT COUNT(*) FROM runs_fts WHERE runs_fts MATCH 'relu'").fetchone()[0] == 0

    index.conn.execute(
        "INSERT INTO candidates (run_id, idx, label, message) VALUES ('r1', 0, 'r1.p0', 'coalesced stores')"
    )
    assert index.conn.execute("SELECT COUNT(*) FROM cands_fts WHERE cands_fts MATCH 'coalesced'").fetchone()[0] == 1
    index.conn.execute("UPDATE candidates SET message = 'vectorized loads' WHERE run_id = 'r1' AND idx = 0")
    assert index.conn.execute("SELECT COUNT(*) FROM cands_fts WHERE cands_fts MATCH 'coalesced'").fetchone()[0] == 0
    assert index.conn.execute("SELECT COUNT(*) FROM cands_fts WHERE cands_fts MATCH 'vectorized'").fetchone()[0] == 1
    index.conn.execute("DELETE FROM candidates WHERE run_id = 'r1'")
    assert index.conn.execute("SELECT COUNT(*) FROM cands_fts WHERE cands_fts MATCH 'vectorized'").fetchone()[0] == 0


def test_binding_drift_is_left_out_of_the_arm(tmp_path, monkeypatch):
    """@brief 验证指纹与锚不一致的 trial 绑定为 drift，既不计入 arm 统计也单独列在 drift 里。

    @param tmp_path pytest 提供的临时目录，用来放实验目录与索引库。
    @param monkeypatch pytest 的环境变量夹具，用于指向临时 runs 目录与临时数据库。
    """
    runs = tmp_path / "runs" / "exp"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    frozen = {"tasks": ["relu"], "dtype": "fp16", "size": "small", "budget_usd": 1.0}
    ensure(runs, kind="repeatability", driver="repeatability", frozen=frozen, arms=[{"arm_id": "relu", "delta": {}, "planned_n": 2}])
    attach(runs, "relu", "relu/trial_1")
    first = runs / "relu" / "trial_1"
    _finished(first, fingerprint="aaa", preset_budget=1.0)
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    assert index.conn.execute("SELECT binding FROM runs WHERE run_id = 'exp/relu/trial_1'").fetchone()["binding"] == "verified"
    attach(runs, "relu", "relu/trial_2")
    second = runs / "relu" / "trial_2"
    _finished(second, fingerprint="bbb", preset_budget=1.0)
    index.sync_known()
    assert index.conn.execute("SELECT binding FROM runs WHERE run_id = 'exp/relu/trial_2'").fetchone()["binding"] == "drift"
    doc = compare("exp")
    assert "exp/relu/trial_2" in doc["drift"]
    assert doc["arms"][0]["n"] == 1


def test_plan_outcomes_stay_empty_until_the_switch_is_on(tmp_path):
    """@brief 验证 plan_ledger 关闭时 ``plan_outcomes`` 为空，打开后产出带已被否证结论的节点且不带优先/置信度字段。

    @param tmp_path pytest 提供的临时目录，用作 run 目录。
    """
    directory = tmp_path / "run"
    directory.mkdir()
    state = {"history": ["round 1: try vectors -> not better"], "trace": [{"label": "r1.p0", "status": "ok", "speedup": 1.0, "best_speedup": 1.2, "improved": False}]}
    assert plan_outcomes(directory, {"search": {"plan_ledger": False}}, state, None) == []
    nodes = plan_outcomes(directory, {"search": {"plan_ledger": True}}, state, None)
    assert nodes and nodes[0]["outcome"] == "falsified"
    assert "priority" not in nodes[0] and "confidence" not in nodes[0]


def test_an_idle_frame_does_not_grow_with_finished_runs(tmp_path, monkeypatch):
    """@brief 验证空闲 tick 的代价不随历史 run 数量增长：两次 tick 都不读 JSON、不列目录。

    @param tmp_path pytest 提供的临时目录，用来放 runs 目录与索引库。
    @param monkeypatch pytest 的环境变量夹具，用于指向临时 runs 目录与临时数据库。
    """
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    _populate(runs, 100)
    index = Index(tmp_path / "observe.sqlite")
    index.tick()
    index.tick()
    idle_small = dict(index.stats)
    _populate(runs, 1000)
    index.tick()
    index.tick()
    idle_large = dict(index.stats)
    assert idle_large["json_reads"] == idle_small["json_reads"] == 0
    assert idle_large["dir_lists"] == idle_small["dir_lists"] == 0


def test_feature_grid_keeps_the_fastest_in_each_cell(tmp_path, monkeypatch):
    """@brief 验证 feature grid 每个格子只留最快的候选，且未通过确认的候选不会标记为已发布。

    @param tmp_path pytest 提供的临时目录，用来放 runs 目录与索引库。
    @param monkeypatch pytest 的环境变量夹具，用于指向临时 runs 目录与临时数据库。
    """
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    directory = runs / "gemm"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({"task": "gemm", "dtype": "fp16", "phase": "done", "started_at": 1, "updated_at": 1}))
    (directory / "result.json").write_text(json.dumps({
        "task": "gemm", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.1, "budget_usd": 1,
        "confirm_winner_label": "r1.p0",
    }))
    slow = directory / "candidates" / "000_init"
    fast = directory / "candidates" / "001_r1.p0"
    for path, speed in ((slow, 0.4), (fast, 1.2)):
        path.mkdir(parents=True)
        (path / "kernel.metal").write_text("kernel void k() {}\n")
        (path / "report.json").write_text(json.dumps({"speedup": speed, "status": "ok"}))
    detail = run_detail("gemm")
    assert len(detail["feature_grid"]) == 1
    assert detail["feature_grid"][0]["label"] == "r1.p0"
    assert detail["feature_grid"][0]["speedup"] == 1.2
    assert detail["feature_grid"][0]["published"] is False
    assert detail["plan_outcomes"] == []


def _finished(directory: Path, *, fingerprint: str, preset_budget: float) -> None:
    """@brief 在指定目录写下一套已结束 run 的记录文件（state.json、result.json、run_config.json）。

    @param directory 目标 run 目录，不存在时创建。
    @param fingerprint 写进 ``run_config.json`` 的 task 指纹，用于制造 binding drift。
    @param preset_budget 写进 result 与 ``search.budget_usd`` 的预算值。
    """
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "phase": "done", "started_at": time.time(), "updated_at": time.time(),
    }))
    (directory / "result.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.1, "budget_usd": preset_budget,
    }))
    (directory / "run_config.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": "small", "task_fingerprint": fingerprint,
        "search": {"budget_usd": preset_budget},
    }))


def test_a_child_attached_later_is_rebound(tmp_path, monkeypatch):
    """@brief 验证先于实验存在、后来才 attach 的子 run 在下次 sync 后由 drift 重绑为 verified。

    @param tmp_path pytest 提供的临时目录，用来放实验目录与索引库。
    @param monkeypatch pytest 的环境变量夹具，用于指向临时 runs 目录与临时数据库。
    """
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    frozen = {"task": "relu", "dtype": "fp16", "size": "small", "budget_usd": 1.0}
    ensure(runs / "exp", kind="repeatability", driver="repeatability", frozen=frozen, arms=[{"arm_id": "relu", "delta": {}, "planned_n": 1}])
    first = runs / "exp" / "relu" / "trial_1"
    _finished(first, fingerprint="aaa", preset_budget=1.0)
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    assert index.conn.execute("SELECT binding FROM runs WHERE run_id = 'exp/relu/trial_1'").fetchone()["binding"] == "drift"
    attach(runs / "exp", "relu", "relu/trial_1")
    index.sync_known()
    assert index.conn.execute("SELECT binding FROM runs WHERE run_id = 'exp/relu/trial_1'").fetchone()["binding"] == "verified"


def test_the_lowest_trial_sets_the_fingerprint_anchor(tmp_path, monkeypatch):
    """@brief 验证编号最小的 trial 决定 arm 的指纹锚：先登记 trial_2 再登记 trial_1，锚取 trial_1，trial_2 转为 drift。

    @param tmp_path pytest 提供的临时目录，用来放实验目录与索引库。
    @param monkeypatch pytest 的环境变量夹具，用于指向临时 runs 目录与临时数据库。
    """
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    frozen = {"task": "relu", "dtype": "fp16", "size": "small", "budget_usd": 1.0}
    ensure(runs / "exp", kind="repeatability", driver="repeatability", frozen=frozen, arms=[{"arm_id": "relu", "delta": {}, "planned_n": 2}])
    later = runs / "exp" / "relu" / "trial_2"
    _finished(later, fingerprint="bbb", preset_budget=1.0)
    attach(runs / "exp", "relu", "relu/trial_2")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    earlier = runs / "exp" / "relu" / "trial_1"
    _finished(earlier, fingerprint="aaa", preset_budget=1.0)
    attach(runs / "exp", "relu", "relu/trial_1")
    index.sync_known()
    anchor = index.conn.execute("SELECT task_fingerprint FROM arms WHERE arm_id = 'relu'").fetchone()["task_fingerprint"]
    assert anchor == "aaa"
    assert index.conn.execute("SELECT binding FROM runs WHERE run_id = 'exp/relu/trial_1'").fetchone()["binding"] == "verified"
    assert index.conn.execute("SELECT binding FROM runs WHERE run_id = 'exp/relu/trial_2'").fetchone()["binding"] == "drift"


def test_editing_the_winner_kernel_drops_the_confirmed_speedup(tmp_path, monkeypatch):
    """@brief 验证 winner kernel 被改后哈希不再对应确认源，已确认的发布状态与加速比会被撤下。

    @param tmp_path pytest 提供的临时目录，用来放 runs 目录与索引库。
    @param monkeypatch pytest 的环境变量夹具，用于指向临时 runs 目录与临时数据库。
    """
    import hashlib

    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    directory = runs / "gemm"
    metal = "kernel void k() {}\n"
    digest = hashlib.sha256(metal.encode()).hexdigest()
    cand = directory / "candidates" / "000_r1.p0"
    cand.mkdir(parents=True)
    (cand / "kernel.metal").write_text(metal)
    (directory / "best.metal").write_text(metal)
    (directory / "confirm_report.json").write_text(json.dumps({
        "status": "ok", "speedup": 1.4,
        "benchmark": {"noisy": False, "candidate": {"cv": 0.01, "reps": 15}, "baselines": {"mlx": {"cv": 0.01, "reps": 15}}, "best_baseline": "mlx", "confirmation": {"kernel_sha256": digest, "winner_label": "r1.p0", "confirmed_values": [1.4], "speedup": 1.4, "candidate_us": None}},
        "protocol_version": PROTOCOL_VERSION, "target_id": "apple-m4-max",
    }))
    (directory / "state.json").write_text(json.dumps({"task": "gemm", "dtype": "fp16", "phase": "done", "started_at": 1, "updated_at": 1}))
    (directory / "result.json").write_text(json.dumps({
        "task": "gemm", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.1, "budget_usd": 1,
        "confirmed_status": "ok", "confirmed_speedup": 1.4, "confirm_winner_label": "r1.p0",
        "confirmation_source_sha256": digest, "confirmed_values": [1.4], "target_id": "apple-m4-max",
    }))
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    got = index.conn.execute("SELECT publication, published_speedup FROM runs").fetchone()
    assert (got["publication"], got["published_speedup"]) == ("confirmed", 1.4)
    (cand / "kernel.metal").write_text("kernel void k() { int x = 1; }\n")
    index.sync_known()
    got = index.conn.execute("SELECT publication, published_speedup FROM runs").fetchone()
    assert got["publication"] != "confirmed" and got["published_speedup"] is None
    assert best_confirmed()["best"] == []


def test_drift_does_not_delete_the_ledger(tmp_path, monkeypatch):
    """@brief 绑定漂移只影响可比性，不得删除已经发生的实际支出。"""
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    index = Index(tmp_path / "observe.sqlite")
    row = {"task": "relu", "started_at": 1.0, "disk_budget_usd": 1.0}
    index._ledger_row("run", row, 2.0, "manifest", "verified")
    assert index.total_usd() == 2.0
    index._ledger_row("run", row, 2.0, "inferred", "drift")
    assert index.total_usd() == 2.0


def test_a_dead_run_leaves_the_near_budget_banner(tmp_path, monkeypatch):
    """@brief 验证进程已死的 run 在超时后变为 interrupted，且不再显示“接近预算”的提示。

    @param tmp_path pytest 提供的临时目录，用来放 runs 目录与索引库。
    @param monkeypatch pytest 的环境变量与替换夹具，用于把 detail 模块的时间推后 100 秒。
    """
    from observe import detail
    from observe.query import history

    runs = tmp_path / "runs" / "live"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    runs.mkdir(parents=True)
    (runs / "state.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "phase": "evaluate", "pid": 999999,
        "started_at": 1, "updated_at": time.time(), "spent_usd": 0.9, "budget_usd": 1.0,
    }))
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    assert history()[0]["status"] == "running"
    original = time.time
    monkeypatch.setattr(detail.time, "time", lambda: original() + 100)
    index.tick()
    row = history()[0]
    assert row["status"] == "interrupted" and row["budget"]["level"] != "near"


def test_llm_switches_are_visible_to_binding():
    """@brief 验证开关查找会从 search 段回落到 llm 段，因此 llm 开关对绑定计算可见。"""
    from observe.index import _lookup

    config = {"search": {"stop_at_roofline": True}, "llm": {"truncation_retry": True, "model": "m"}}
    assert _lookup(config, config["search"], "truncation_retry") is True
    assert _lookup(config, config["search"], "model") == "m"


def test_the_connection_can_be_read_from_another_thread(tmp_path, monkeypatch):
    """@brief 验证索引连接可在其他线程里执行查询而不报错（线程安全的连接配置）。

    @param tmp_path pytest 提供的临时目录，用来放索引库。
    @param monkeypatch pytest 的环境变量夹具，用于指向临时 runs 目录与空的 books 库。
    """
    import threading

    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    index = Index(tmp_path / "observe.sqlite")
    errors = []

    def read():
        try:
            index.conn.execute("SELECT 1").fetchone()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=read)
    thread.start()
    thread.join()
    assert errors == []


def _populate(root: Path, count: int) -> None:
    """@brief 批量生成已结束的 run 目录，每个带 50 个候选占位文件；已存在的目录跳过。

    @param root 生成 run 的根目录，不存在时创建。
    @param count 目标 run 数量，目录名按 ``run_0000`` 递增编号。
    """
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        directory = root / f"run_{index:04d}"
        if directory.exists():
            continue
        directory.mkdir()
        (directory / "state.json").write_text(json.dumps({
            "task": "relu", "dtype": "fp16", "phase": "done", "pid": 1, "started_at": index, "updated_at": index,
        }))
        (directory / "result.json").write_text(json.dumps({
            "task": "relu", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.01, "budget_usd": 1,
        }))
        candidates = directory / "candidates"
        candidates.mkdir()
        for cand in range(50):
            (candidates / f"{cand}.json").write_text("{}")


def test_candidates_are_indexed_and_a_list_page_carries_the_next_cursor(tmp_path, monkeypatch):
    """@brief 验证候选入库、列表分页带 next 游标且两页不重叠，失败与最佳确认查询各自返回预期结果。

    @param tmp_path pytest 提供的临时目录，用来放 runs 目录与索引库。
    @param monkeypatch pytest 的环境变量夹具，用于指向临时 runs 目录与空的 books 库。
    """
    from observe import query as scan

    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    trace = [{"label": "init", "status": "ok", "speedup": 1.0, "within_noise": True},
             {"label": "r1.p0", "status": "abi_error", "speedup": None, "within_noise": False}]
    for index, (name, task) in enumerate((("run0", "gemm"), ("run1", "relu"), ("run2", "gemm"))):
        directory = runs / name
        directory.mkdir(parents=True)
        (directory / "state.json").write_text(json.dumps({
            "task": task, "dtype": "fp16", "phase": "done", "started_at": 10 + index, "updated_at": 10 + index,
            "model": "model-a", "trace": trace,
        }))
        (directory / "result.json").write_text(json.dumps({
            "task": task, "dtype": "fp16", "started_at": 10 + index, "spent_usd": 1.0,
            "protocol_version": "2", "trace": trace,
        }))
        (directory / "run_config.json").write_text(json.dumps({
            "protocol_version": "2",
            "committees": [{"name": "first", "members": [{"provider": "a", "model": "model-a"}]}],
        }))
        for idx, label in enumerate(("init", "r1.p0")):
            folder = directory / "candidates" / f"{idx:03d}_{label}"
            folder.mkdir(parents=True)
            (folder / "kernel.metal").write_text(f"kernel void {task}_{idx}() {{}}\n")
            status = "ok" if label == "init" else "abi_error"
            (folder / "report.json").write_text(json.dumps({
                "status": status, "stage": "bench" if status == "ok" else "abi",
                "speedup": 1.0 if status == "ok" else None, "eligible": status == "ok",
            }))
    scan.refresh()
    page = scan.runs_page(limit=2)
    assert len(page["runs"]) == 2 and page["next"]
    assert page["runs"][0]["protocol_version"] == "2"
    assert page["runs"][0]["committees_json"]
    page2 = scan.runs_page(limit=2, cursor=page["next"])
    assert set(row["run_id"] for row in page["runs"]).isdisjoint(row["run_id"] for row in page2["runs"])
    assert len(scan.candidates_page("run0")["candidates"]) == 2
    failed = scan.failures(status="abi_error")
    assert {row["run_id"] for row in failed["failures"]} == {"run0", "run1", "run2"}
    assert scan.best_confirmed(task="gemm")["best"] == []


def test_a_legacy_flame_chase_run_is_not_labelled_chat(tmp_path, monkeypatch):
    """@brief 验证老 run 凭 loop_protocol 决定协议时代：flame_chase_1 归为 epoch 2，缺失则标 legacy 而不误判成 chat。

    @param tmp_path pytest 提供的临时目录，用来放 runs 目录与索引库。
    @param monkeypatch pytest 的环境变量夹具，用于指向临时 runs 目录与空的 books 库。
    """
    from observe import query as scan

    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    for name, protocol in (("old_flame", "flame_chase_1"), ("old_chat", None)):
        directory = runs / name
        directory.mkdir(parents=True)
        (directory / "state.json").write_text(json.dumps({"task": "relu", "dtype": "fp16", "phase": "done"}))
        (directory / "result.json").write_text(json.dumps({"task": "relu", "dtype": "fp16"}))
        config = {"committees": []}
        if protocol:
            config["loop_protocol"] = protocol
        (directory / "run_config.json").write_text(json.dumps(config))
    scan.refresh()
    page = scan.runs_page(limit=10)
    by_id = {row["run_id"]: row["protocol_version"] for row in page["runs"]}
    assert by_id["old_flame"] == "flame_chase_1"
    assert by_id["old_chat"] == "legacy"
    epochs = {row["run_id"]: row["protocol_epoch"] for row in page["runs"]}
    assert epochs["old_flame"] == "2" and epochs["old_chat"] == "legacy"


def test_sessions_and_model_totals_come_from_agent_rows(tmp_path, monkeypatch):
    """@brief 验证会话与模型统计只来自 transcript 的 agent 行（chat 行不计），并带上诊断列与 token 数。

    @param tmp_path pytest 提供的临时目录，用来放 runs 目录与索引库。
    @param monkeypatch pytest 的环境变量夹具，用于指向临时 runs 目录与空的 books 库。
    """
    from observe import query as scan

    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    directory = runs / "hmz"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({"task": "relu", "dtype": "fp16", "phase": "done"}))
    (directory / "result.json").write_text(json.dumps({"task": "relu", "dtype": "fp16"}))
    (directory / "run_config.json").write_text(json.dumps({"loop_protocol": "flame_chase_1"}))
    (directory / "transcript.jsonl").write_text("\n".join(json.dumps(row) for row in [
        {"event": "agent", "label": "init", "committee": "first", "provider": "deepseek", "model": "deepseek-flash",
         "outcome": "evaluated", "cost_usd": 1.0, "tokens": {"input": 10, "output": 20, "cache_read": 30, "cache_write": 40}},
        {"event": "agent", "label": "r1.p0.retry1", "candidate_label": "r1.p0", "attempt": 1,
         "committee": "second", "provider": "moonshot", "model": "kimi-k3",
         "outcome": "agent_failed", "error": "Connection error.", "cost_usd": 2.0,
         "runtime": {"session_ids": ["session-r1"], "failed_turns": 4, "last_terminal_kind": "failed",
                     "failure_class": "turn_failed", "tool_calls": 7},
         "diagnostics_path": "agent_diagnostics/r1.json",
         "tokens": {"input": 10, "output": 20, "cache_read": 30, "cache_write": 40}},
        {"event": "chat", "messages": [{"role": "user", "content": "not a session"}]},
    ]) + "\n")
    scan.refresh()
    sessions = scan.sessions_page("hmz")["sessions"]
    assert [row["model"] for row in sessions] == ["deepseek-flash", "kimi-k3"]
    assert sessions[0]["tokens_cache_write"] == 40
    assert sessions[0]["candidate_label"] == "init" and sessions[0]["attempt"] == 0
    assert sessions[1]["candidate_label"] == "r1.p0" and sessions[1]["attempt"] == 1
    assert sessions[1]["session_id"] == "session-r1" and sessions[1]["failed_turns"] == 4
    assert sessions[1]["last_kind"] == "failed" and sessions[1]["failure_class"] == "turn_failed"
    assert sessions[1]["error"] == "Connection error." and sessions[1]["tool_calls"] == 7
    assert sessions[1]["diagnostics_path"] == "agent_diagnostics/r1.json"
    totals = scan.model_stats()["models"]
    assert {row["model"]: row["sessions"] for row in totals} == {"deepseek-flash": 1, "kimi-k3": 1}
    assert {row["model"]: row["provider"] for row in totals} == {"deepseek-flash": "deepseek", "kimi-k3": "moonshot"}
    assert scan.model_stats(run_id="hmz")["models"] == totals
    assert scan.model_stats(run_id="missing")["models"] == []


def test_migrating_a_run_does_not_create_a_second_index_row(tmp_path, monkeypatch):
    """@brief 验证迁移一个 run 不会在索引里产生第二行，否则旧账本行会让花费被重复计入。

    改名不是一个新的 run：迁移后只剩迁移命名的那一行，总额不变。

    @param tmp_path pytest 提供的临时目录，用来放 runs 目录与索引库。
    @param monkeypatch pytest 的环境变量夹具，用于指向临时 runs 目录与临时数据库。
    """
    from taic.migrate import migrate_runs

    runs = tmp_path / "runs"
    db = tmp_path / "observe.sqlite"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(db))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "missing.sqlite"))
    monkeypatch.delenv("OBSERVE_CARRYOVER", raising=False)
    directory = runs / "one"
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(json.dumps({
        "schema_version": 1, "task": "relu", "dtype": "fp16", "size": "small",
        "stop_reason": "max_rounds", "spent_usd": 1.0, "budget_usd": 2.0, "started_at": 100.0,
    }))
    (directory / "state.json").write_text(json.dumps({
        "schema_version": 1, "task": "relu", "dtype": "fp16", "phase": "done",
        "started_at": 100.0, "updated_at": 110.0, "pid": 0,
    }))
    index = Index(db)
    index.sync_known()
    assert index.total_usd() == pytest.approx(1.0)
    assert migrate_runs(runs, "apple-m4-max") == ["one"]
    index.sync_known()
    rows = [row["run_id"] for row in index.conn.execute("SELECT run_id FROM runs")]
    assert rows == ["apple-m4-max/one"]
    assert index.total_usd() == pytest.approx(1.0)

