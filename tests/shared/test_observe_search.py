"""§10.3 搜索：runs/candidates 走 FTS5 投影，失败簇直接匹配小表；删除后不再命中。"""

from __future__ import annotations

import time

from observe import query
from observe.index import get_index


def _index(tmp_path, monkeypatch):
    """@brief 临时空 runs 根 + 索引。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @return ``(runs 根, Index)``。
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    monkeypatch.setenv("OBSERVE_REFRESH_ON_READ", "0")
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(tmp_path / "kernels"))
    monkeypatch.setenv("OBSERVE_TICK_INTERVAL", "3600")
    index = get_index()
    # 这些行是直接插进索引的（没有 run 目录）；别让 ``failure_clusters`` 的 tick 把它们标成 missing。
    query._TICK["at"] = time.monotonic()
    return runs, index


def _run(index, run_id: str, *, task="relu", dtype="fp16", model="deepseek-flash",
         stop="max_rounds", status="done") -> None:
    """@brief 直接插一行运行（FTS 触发器同步）。

    @param index 索引。
    @param run_id 运行 id。
    @param task 任务名。
    @param dtype 数据类型。
    @param model 模型。
    @param stop 停止原因。
    @param status 状态。
    """
    index.conn.execute(
        "INSERT INTO runs (run_id, task, dtype, target_id, model, stop_reason, status, protocol_version)"
        " VALUES (?, ?, ?, 'apple-m4-max', ?, ?, ?, '3')",
        (run_id, task, dtype, model, stop, status),
    )


def _candidate(index, run_id: str, label: str, *, status="abi_error", stage="abi",
               message="ABI mismatch: missing function signature for relu") -> None:
    """@brief 直接插一颗候选（FTS 触发器同步）。

    @param index 索引。
    @param run_id 运行 id。
    @param label 标签。
    @param status 状态。
    @param stage 阶段。
    @param message 报告信息。
    """
    index.conn.execute(
        "INSERT INTO candidates (run_id, idx, label, status, stage, message) VALUES (?, 0, ?, ?, ?, ?)",
        (run_id, label, status, stage, message),
    )


def test_runs_search_matches_metadata_and_drives_the_fts_index(tmp_path, monkeypatch):
    """@brief 验收：runs 六个字段可检索；``EXPLAIN QUERY PLAN`` 走 FTS 而非 ``SCAN runs``。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, index = _index(tmp_path, monkeypatch)
    _run(index, "relu/one", model="deepseek-flash", stop="max_rounds")
    _run(index, "relu/two", model="kimi-k3", stop="budget_exhausted")
    _run(index, "gemm/one", task="gemm")
    index.conn.commit()

    document = query.search(q="relu", kind="runs")
    assert document["total"] == 2
    assert {row["run_id"] for row in document["rows"]} == {"relu/one", "relu/two"}
    assert query.search(q="kimi", kind="runs")["rows"][0]["run_id"] == "relu/two"
    assert query.search(q="budget_exhausted", kind="runs")["total"] == 1
    assert query.search(q="不存在", kind="runs")["rows"] == []

    plan = [tuple(row) for row in index.conn.execute(
        "EXPLAIN QUERY PLAN SELECT r.run_id FROM runs_fts JOIN runs r ON r.rowid = runs_fts.rowid"
        " WHERE runs_fts MATCH 'relu' AND r.status != 'missing'"
    )]
    text = " ".join(str(row[-1]) for row in plan)
    assert "runs_fts" in text and "VIRTUAL TABLE INDEX" in text
    assert "SCAN runs " not in text and "SCAN runs\n" not in text


def test_candidate_search_finds_message_substrings(tmp_path, monkeypatch):
    """@brief 验收：``kind=candidates`` 按失败 message 命中（真实 abi_error 文本）。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, index = _index(tmp_path, monkeypatch)
    _run(index, "relu/one")
    _candidate(index, "relu/one", "r1.p0")
    _run(index, "relu/two")
    _candidate(index, "relu/two", "r9.p0", status="incorrect", stage="correctness",
               message="elements out of tolerance: max abs error 2.5")
    index.conn.commit()

    document = query.search(q="missing signature", kind="candidates")
    assert document["total"] == 1
    row = document["rows"][0]
    assert row["run_id"] == "relu/one" and row["label"] == "r1.p0" and "ABI mismatch" in row["message"]
    assert query.search(q="tolerance", kind="candidates")["rows"][0]["label"] == "r9.p0"
    assert query.search(q="tolerance", kind="runs")["rows"] == []


def test_a_deleted_or_missing_run_no_longer_matches(tmp_path, monkeypatch):
    """@brief 验收：删掉一场 run 后 FTS 不再命中；``status='missing'`` 也不算命中。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, index = _index(tmp_path, monkeypatch)
    _run(index, "relu/one")
    _run(index, "relu/two")
    index.conn.commit()
    assert query.search(q="relu", kind="runs")["total"] == 2

    index.conn.execute("DELETE FROM runs WHERE run_id = 'relu/one'")
    index.conn.commit()
    assert query.search(q="relu", kind="runs")["rows"][0]["run_id"] == "relu/two"

    index.conn.execute("UPDATE runs SET status = 'missing' WHERE run_id = 'relu/two'")
    index.conn.commit()
    assert query.search(q="relu", kind="runs")["rows"] == []


def test_cluster_search_matches_the_small_derived_table(tmp_path, monkeypatch):
    """@brief 验收：``kind=clusters`` 按簇名/签名匹配；空查询不炸。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, index = _index(tmp_path, monkeypatch)
    _run(index, "relu/one")
    _candidate(index, "relu/one", "r1.p0")
    index.conn.execute(
        "INSERT INTO failure_members (run_id, idx, sig, status, stage, observed_at)"
        " VALUES ('relu/one', 0, 'sig-abi-1', 'abi_error', 'abi', 10.0)"
    )
    index.conn.commit()

    document = query.search(q="sig-abi", kind="clusters")
    assert document["total"] == 1
    assert document["rows"][0]["sig"] == "sig-abi-1" and document["rows"][0]["n"] == 1
    assert query.search(q="", kind="clusters") == {
        "schema_version": "1", "kind": "clusters", "q": "", "rows": [], "total": 0,
    }


def test_search_caps_the_rows_but_reports_the_full_total(tmp_path, monkeypatch):
    """@brief 验收：limit 只限制展示，``total`` 是完整命中数。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, index = _index(tmp_path, monkeypatch)
    for number in range(25):
        _run(index, f"relu/{number:02d}")
    index.conn.commit()
    document = query.search(q="relu", kind="runs", limit=5)
    assert len(document["rows"]) == 5 and document["total"] == 25


def test_the_search_http_endpoint_is_reachable(tmp_path, monkeypatch):
    """@brief HTTP：``/api/v2/search`` 三种 kind 都可达，空查询返回空行。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from starlette.testclient import TestClient

    from observe.app import app

    _runs, index = _index(tmp_path, monkeypatch)
    _run(index, "relu/one")
    _candidate(index, "relu/one", "r1.p0")
    index.conn.execute(
        "INSERT INTO failure_members (run_id, idx, sig, status, stage, observed_at)"
        " VALUES ('relu/one', 0, 'sig-abi-1', 'abi_error', 'abi', 10.0)"
    )
    index.conn.commit()
    with TestClient(app) as client:
        runs = client.get("/api/v2/search?q=relu&kind=runs")
        assert runs.status_code == 200 and runs.json()["rows"][0]["run_id"] == "relu/one"
        candidates = client.get("/api/v2/search?q=missing&kind=candidates")
        assert candidates.status_code == 200 and candidates.json()["rows"][0]["label"] == "r1.p0"
        clusters = client.get("/api/v2/search?q=sig-abi&kind=clusters")
        assert clusters.status_code == 200 and clusters.json()["rows"][0]["n"] == 1
        empty = client.get("/api/v2/search?q=")
        assert empty.status_code == 200 and empty.json()["rows"] == []
