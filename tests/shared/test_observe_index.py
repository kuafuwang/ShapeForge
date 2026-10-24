"""Manifest, index, binding, comparison and the idle frame. No GPU and no model calls."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from experiments.manifest import ManifestError, attach, config_hash, ensure, extend_budget
from observe.index import INDEX_SCHEMA_VERSION, Index
from observe.legacy import infer
from observe.query import best_confirmed, compare, plan_outcomes, run_detail
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

