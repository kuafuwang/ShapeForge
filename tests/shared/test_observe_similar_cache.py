"""句向量与相似结论的持久缓存、语料失效与后台预热。

无 GPU、无模型请求：句向量与相似度都用假实现代替，只验证缓存与失效语义。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observe import feedback, similarity
from taic.agent_feedback import parse_feedback


def _open(tmp_path, monkeypatch):
    """@brief 建临时 runs 与索引环境；缓存库随 OBSERVE_DB 一起落到临时目录。

    @param tmp_path pytest 临时目录。
    @param monkeypatch 环境变量夹具。
    @return ``(runs 目录, 索引实例)``。
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("TAIC_REVIEW_EVENTS", str(tmp_path / "review_events.jsonl"))
    import observe.index as index_mod

    index_mod._SINGLE = None
    return runs, index_mod.get_index()


def _run(root: Path, name: str, text: str, *, t: float) -> Path:
    """@brief 造一场只有一条系统建议的已结束 run。

    @param root runs 根目录。
    @param name run 目录名。
    @param text 建议文本。
    @param t 记录时间戳（决定候选的早晚关系）。
    @return run 目录。
    """
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "phase": "done", "started_at": t, "updated_at": t,
    }))
    parsed = parse_feedback("# 对系统的建议\n- " + text + "\n")
    record = {
        "format": "agent_feedback", "v": 1, "seq": 1, "t": t, "task": "conv2d", "dtype": "fp16", "size": "paper",
        "label": "r1.p0", "round": 1, "committee": "first", "provider": "deepseek", "model": "m",
        "effort": "high", "outcome": "evaluated", "exit": "normal", "best_speedup": 1.0,
        "kinds": parsed["kinds"], "work_dir": "agent_work/r1.p0", "sections": parsed["sections"],
        "items": parsed["items"], "unverified_premises": parsed["unverified_premises"], "raw": parsed["raw"],
    }
    (directory / "feedback.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n")
    return directory


@pytest.fixture
def fake_similar(monkeypatch):
    """@brief 用可计数的假 find_similar 替换句向量检索，并令 available 恒真。

    @param monkeypatch 补丁夹具。
    @return 记录每次查询文本的列表。
    """
    calls = []

    def fake(text, candidates, threshold=None, limit=3):
        calls.append(text)
        return [{**candidates[0], "score": 0.99}] if candidates else []

    monkeypatch.setattr(similarity, "find_similar", fake)
    monkeypatch.setattr(similarity, "available", lambda: True)
    return calls


def _claim(conn, run_id: str) -> dict:
    """@brief 取一场 run 的 claim 行并补齐文本，作为 similar_claims 的输入。

    @param conn 索引连接。
    @param run_id 运行 id。
    @return 带 text 的 claim 行。
    """
    rows = [dict(row) for row in conn.execute(feedback._CLAIM_LIST + " WHERE c.run_id = ?", (run_id,))]
    feedback._attach_claim_text(rows)
    return rows[0]


def _similar_for(conn, row: dict) -> list[dict]:
    """@brief 按行内容调用 similar_claims。

    @param conn 索引连接。
    @param row claim 行。
    @return 近邻列表。
    """
    return feedback.similar_claims(
        conn,
        digest=str(row["digest"]), task=str(row.get("task") or ""), target_id=str(row.get("target_id") or ""),
        text=str(row.get("text") or ""), before_t=row.get("t"),
        exclude_run=str(row["run_id"]), exclude_seq=row["seq"],
    )


def test_similar_results_are_reused_without_recomputing(tmp_path, monkeypatch, fake_similar):
    """@brief 同一 claim 的近邻只算一次；清掉进程内缓存后仍从持久缓存命中。

    @param tmp_path 临时目录。
    @param monkeypatch 补丁夹具。
    @param fake_similar 计数用的假检索。
    """
    runs, index = _open(tmp_path, monkeypatch)
    _run(runs, "conv", "矩阵单元和 SIMT ALU 吞吐相近。", t=10)
    _run(runs, "conv2", "矩阵单元和 SIMT ALU 吞吐差不多。", t=20)
    index.sync_known()
    row = _claim(index.conn, "conv2")
    first = _similar_for(index.conn, row)
    assert len(fake_similar) == 1 and first and first[0]["score"] == 0.99
    again = _similar_for(index.conn, row)
    assert again == first and len(fake_similar) == 1
    similarity.clear_cache()
    third = _similar_for(index.conn, row)
    assert third == first and len(fake_similar) == 1, "持久缓存必须扛住进程内缓存清空"


def test_a_new_earlier_claim_invalidates_the_cached_neighbours(tmp_path, monkeypatch, fake_similar):
    """@brief 出现新的更早候选后语料键变化，旧近邻不能继续冒充新鲜结果。

    @param tmp_path 临时目录。
    @param monkeypatch 补丁夹具。
    @param fake_similar 计数用的假检索。
    """
    runs, index = _open(tmp_path, monkeypatch)
    _run(runs, "conv", "矩阵单元和 SIMT ALU 吞吐相近。", t=10)
    _run(runs, "conv2", "矩阵单元和 SIMT ALU 吞吐差不多。", t=20)
    index.sync_known()
    row = _claim(index.conn, "conv2")
    _similar_for(index.conn, row)
    _similar_for(index.conn, row)
    assert len(fake_similar) == 1
    _run(runs, "older", "更早的一场也谈过吞吐。", t=5)
    index.sync_known()
    _similar_for(index.conn, row)
    assert len(fake_similar) == 2, "候选集变了必须重算"


def test_the_warmer_precomputes_so_the_read_path_never_searches(tmp_path, monkeypatch, fake_similar):
    """@brief 预热把所有待审 claim 算完；之后读取路径调 find_similar 就是错误。

    @param tmp_path 临时目录。
    @param monkeypatch 补丁夹具。
    @param fake_similar 计数用的假检索。
    """
    runs, index = _open(tmp_path, monkeypatch)
    _run(runs, "conv", "矩阵单元和 SIMT ALU 吞吐相近。", t=10)
    _run(runs, "conv2", "矩阵单元和 SIMT ALU 吞吐差不多。", t=20)
    index.sync_known()
    for _ in range(10):
        if not feedback.warm_similar_once(index.conn):
            break
    else:
        pytest.fail("预热没有收敛")
    assert fake_similar, "预热应当真的做过检索"

    def forbidden(*args, **kwargs):
        raise AssertionError("预热之后读路径不应再算相似")

    monkeypatch.setattr(similarity, "find_similar", forbidden)
    rows = [dict(row) for row in index.conn.execute(feedback._CLAIM_LIST)]
    feedback._attach_claim_text(rows)
    feedback.attach_similar_claims(index.conn, rows)
    assert all("similar" in row for row in rows)


class _FakeModel:
    """@brief 记录调用次数的假句向量模型，向量由文本长度决定。"""

    def __init__(self) -> None:
        self.calls = 0

    def vectorForString_(self, text):  # noqa: N802 -- 与 PyObjC 的命名一致
        self.calls += 1
        return [1.0, float(len(text) % 5), 0.5]


def test_an_idle_warm_pass_costs_a_handful_of_queries(tmp_path, monkeypatch, fake_similar):
    """@brief 预热收敛后的空闲轮只发三条聚合查询，不再逐条待审 claim 算语料键。

    旧实现每次空闲轮都跑一遍 ``_CLAIM_LIST`` 并对每条待审 claim 发两次语料键查询（实测 90 条
    待审 claim = 181 条 SQL，每 2 秒一次）。

    @param tmp_path 临时目录。
    @param monkeypatch 补丁夹具。
    @param fake_similar 计数用的假检索。
    """
    runs, index = _open(tmp_path, monkeypatch)
    for number in range(6):
        _run(runs, f"conv{number}", f"矩阵单元和 SIMT ALU 吞吐相近（第 {number} 场）。", t=10 + number)
    index.sync_known()
    for _ in range(30):
        if not feedback.warm_similar_once(index.conn):
            break
    else:
        pytest.fail("预热没有收敛")
    warmed = len(fake_similar)

    statements = []
    index.conn.set_trace_callback(statements.append)
    try:
        assert feedback.warm_similar_once(index.conn) is False
    finally:
        index.conn.set_trace_callback(None)
    assert len(fake_similar) == warmed, "收敛后的空闲轮不应再检索"
    assert len(statements) <= 4, f"空闲轮 SQL 数应只有几条聚合，实际 {len(statements)}：{statements}"


def test_the_warm_pass_does_not_hold_the_index_write_lock(tmp_path, monkeypatch):
    """@brief 预热跑重计算时，请求线程借写连接不能被它挡住（只读连接 + 不持锁）。

    @param tmp_path 临时目录。
    @param monkeypatch 补丁夹具。
    """
    import threading
    import time

    from observe import index as index_mod

    runs, index = _open(tmp_path, monkeypatch)
    _run(runs, "conv", "矩阵单元和 SIMT ALU 吞吐相近。", t=10)
    _run(runs, "conv2", "矩阵单元和 SIMT ALU 吞吐差不多。", t=20)
    index.sync_known()
    started = threading.Event()
    release = threading.Event()

    def slow(text, candidates, threshold=None, limit=3):
        started.set()
        release.wait(5)
        return []

    monkeypatch.setattr(similarity, "find_similar", slow)
    monkeypatch.setattr(similarity, "available", lambda: True)
    worker = threading.Thread(target=feedback.warm_similar_once, kwargs={"conn": None}, daemon=True)
    worker.start()
    assert started.wait(5), "预热没有开始"
    begin = time.perf_counter()
    with index_mod.borrow() as borrowed:
        borrowed.conn.execute("SELECT 1").fetchone()
    waited = time.perf_counter() - begin
    release.set()
    worker.join(5)
    assert waited < 0.2, f"预热占着写锁：borrow() 等了 {waited * 1000:.0f}ms"


def test_prune_keeps_the_cache_under_its_row_caps(tmp_path, monkeypatch):
    """@brief 缓存库有行数上限：prune 从最旧的行开始删，新行保留。

    @param tmp_path 临时目录。
    @param monkeypatch 补丁夹具。
    """
    from observe import simcache

    _open(tmp_path, monkeypatch)
    simcache._conn().executemany(
        "INSERT INTO similar(run_id, seq, digest, corpus_count, corpus_rev, payload, at)"
        " VALUES (?, ?, ?, 0, 0, '[]', ?)",
        [(f"run{number}", number, f"digest{number}", float(number)) for number in range(50)],
    )
    simcache._conn().executemany(
        "INSERT INTO embeddings(text_sha, language, dim, vector, at) VALUES (?, 'en', 1, 'x', ?)",
        [(f"sha{number}", float(number)) for number in range(50)],
    )
    simcache._conn().commit()
    removed = simcache.prune(max_similar=10, max_embeddings=5)
    assert removed == {"similar": 40, "embeddings": 45}
    kept = [row[0] for row in simcache._conn().execute("SELECT run_id FROM similar ORDER BY at")]
    assert len(kept) == 10 and kept[-1] == "run49" and "run0" not in kept, kept
    stats = simcache.stats()
    assert stats["similar"] == 10 and stats["embeddings"] == 5
    assert simcache.prune(max_similar=10, max_embeddings=5) == {"similar": 0, "embeddings": 0}


def test_a_converged_warm_pass_prunes_the_cache(tmp_path, monkeypatch, fake_similar):
    """@brief 预热收敛时顺手调用 prune：库不会只增不减。

    @param tmp_path 临时目录。
    @param monkeypatch 补丁夹具。
    @param fake_similar 计数用的假检索。
    """
    from observe import simcache

    runs, index = _open(tmp_path, monkeypatch)
    _run(runs, "conv", "矩阵单元和 SIMT ALU 吞吐相近。", t=10)
    _run(runs, "conv2", "矩阵单元和 SIMT ALU 吞吐差不多。", t=20)
    index.sync_known()
    assert feedback.warm_similar_once(index.conn) is True
    for _ in range(10):
        if not feedback.warm_similar_once(index.conn):
            break
    else:
        pytest.fail("预热没有收敛")
    monkeypatch.setattr(simcache, "SIMILAR_MAX_ROWS", 0)
    simcache._conn().execute("DELETE FROM warm_state")
    simcache._conn().commit()
    assert feedback.warm_similar_once(index.conn) is False
    assert simcache.stats()["similar"] == 0, "收敛轮应当把超限的行压回上限内"


def test_embeddings_survive_an_in_memory_cache_clear(tmp_path, monkeypatch):
    """@brief 句向量写进持久缓存：进程内 LRU 清空后不再调用系统模型。

    @param tmp_path 临时目录。
    @param monkeypatch 用假模型替换 _model。
    """
    _open(tmp_path, monkeypatch)
    model = _FakeModel()
    monkeypatch.setattr(similarity, "_model", lambda language: model)
    first = similarity.embedding("矩阵单元 吞吐")
    assert first is not None and model.calls == 1
    similarity.clear_cache()
    again = similarity.embedding("矩阵单元 吞吐")
    assert again == first and model.calls == 1, "句向量不应随进程内缓存清空而重算"
    other = similarity.embedding("另一段文本")
    assert other != first and model.calls == 2
