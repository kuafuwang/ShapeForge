"""Failure fingerprints and the derived ``failure_members`` table (§7).

The signature module is deterministic and the member table counts are derived, so a repeat
ingest cannot inflate a cluster and deleting/moving/missing a run must remove its members.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observe import query
from observe.failure_sig import MESSAGE_CHARS, normalize, session_signature, signature
from observe.index import Index

T0 = 1_700_000_000.0


def _prepare(tmp_path, monkeypatch):
    """@brief Point the runs root, index and books at a temporary directory.

    @param tmp_path pytest temporary directory.
    @param monkeypatch pytest patcher.
    @return ``(runs root, index)``.
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carryover.json"))
    monkeypatch.setenv("OBSERVE_TICK_INTERVAL", "0")
    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "0")
    return runs, Index(tmp_path / "observe.sqlite")


def _run(root: Path, name: str, *, started_at: float = T0) -> Path:
    """@brief Write a finished run directory with a state and a result.

    @param root runs root.
    @param name run id.
    @param started_at source start time.
    @return The run directory.
    """
    directory = root / name
    directory.mkdir(parents=True)
    common = {"task": "conv2d", "dtype": "fp16", "started_at": started_at, "spent_usd": 0.25, "budget_usd": 2.0}
    (directory / "state.json").write_text(json.dumps({**common, "phase": "done", "updated_at": started_at + 1}), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({**common, "stop_reason": "max_rounds"}), encoding="utf-8")
    return directory


def _transcript(directory: Path, *, failing: bool) -> None:
    """@brief Write an ``event=agent`` transcript: one failing session or one evaluated one.

    @param directory run directory.
    @param failing whether the session failed with the provider error used by the sig fixtures.
    """
    event = {
        "event": "agent", "label": "r1.p0", "provider": "deepseek", "model": "deepseek-flash",
        "committee": "first", "outcome": "agent_failed" if failing else "evaluated",
    }
    if failing:
        event["error"] = "provider returned 402"
        event["runtime"] = {"failure_class": "turn_failed", "failed_turns": 0}
    (directory / "transcript.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")


def _candidate(directory: Path, index: int, label: str, report: dict) -> Path:
    """@brief Write one candidate folder with a kernel and a report.

    @param directory run directory.
    @param index three-digit candidate index.
    @param label candidate label.
    @param report report document.
    @return The candidate directory.
    """
    folder = directory / "candidates" / f"{index:03d}_{label}"
    folder.mkdir(parents=True)
    (folder / "kernel.metal").write_text("kernel void k() {}\n", encoding="utf-8")
    # A candidate report is current evidence only at schema 4 (plan §8); fixtures say so.
    (folder / "report.json").write_text(json.dumps({"schema_version": 4, **report}), encoding="utf-8")
    return folder


def test_one_normalized_signature_covers_numbers_labels_and_paths_but_not_the_sign():
    """@brief §7.1：数字/标签/路径/引号被归一化；正负号刻意分开（两类不同修法）。"""
    first = signature("incorrect", "correctness", (
        "case 'a': 1/8 elements out of tolerance (worst scaled error 1.5); "
        "worst: o[0, 1, 10] got -0.0022 want -0.0031 (see /Users/me/runs/x/candidates/000_r1.p0)"
    ))
    second = signature("incorrect", "correctness", (
        "case \"b\": 7/1048576 elements out of tolerance (worst scaled error 9); "
        "worst: p[2, 3, 4] got -12 want -0.5 (see /tmp/other/candidates/012_r9.p1)"
    ))
    assert first == second
    positive = signature("incorrect", "correctness", "worst: p[2, 3, 4] got 12 want 0.5")
    negative = signature("incorrect", "correctness", "worst: p[2, 3, 4] got -12 want 0.5")
    assert positive != negative, "符号差异必须保留为两个簇"
    # 状态与阶段也在指纹里。
    assert signature("abi_error", "abi", "x") != signature("abi_error", "compile", "x")
    # 脱敏在归一化之前：sk- 形态的密钥不会留在签名输入里。
    assert "sk-" not in normalize("failed with sk-abcdefghijklmnop and /Users/me/x")
    # 会话签名带前缀，不会与候选签名重合。
    assert session_signature("turn_failed", "x") != signature("turn_failed", "turn_failed", "x")
    assert session_signature("", "") is None


def _cluster_fixture(root: Path) -> Path:
    """@brief One run with two failing candidates that share a normalized message and one ok.

    @param root runs root.
    @return The run directory.
    """
    directory = _run(root, "conv2d/cluster", started_at=T0 + 5)
    message = "case 'outlier': {n}/100 elements out of tolerance; worst: o[0, 1, {k}] got {got} want {want}"
    _candidate(directory, 0, "r1.p0", {
        "status": "incorrect", "stage": "correctness", "speedup": 0.1,
        "message": message.format(n=1, k=10, got="-0.002", want="-0.003"),
        "benchmark": {"candidate": {"cv": 0.02, "reps": 15}}, "cache_hit": False,
    })
    _candidate(directory, 1, "r1.p1", {
        "status": "incorrect", "stage": "correctness", "speedup": 0.2,
        "message": message.format(n=7, k=99, got="-12", want="-0.5"),
        "benchmark": {"candidate": {"cv": 0.01, "reps": 20}},
    })
    _candidate(directory, 2, "r1.p2", {
        "status": "ok", "stage": "done", "speedup": 1.4, "message": "compiled, ABI ok, safe, correct and benchmarked",
    })
    return directory


def test_the_member_table_is_derived_and_a_repeat_ingest_does_not_inflate_it(tmp_path, monkeypatch):
    """@brief §7.2 验收：同一事务先清后插；两次摄取后每个簇的 ``n`` 不变，``observed_at`` 取运行开始时间。"""
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = _cluster_fixture(runs)
    index.sync_known()
    rows = index.conn.execute("SELECT * FROM failure_members ORDER BY idx").fetchall()
    assert len(rows) == 2
    assert rows[0]["sig"] == rows[1]["sig"]
    assert [row["idx"] for row in rows] == [0, 1]
    assert all(row["observed_at"] == pytest.approx(T0 + 5) for row in rows), "observed_at 必须是源数据时间"
    stored = dict(index.conn.execute(
        "SELECT run_id, started_at FROM runs WHERE run_id = 'conv2d/cluster'"
    ).fetchone())
    assert stored["started_at"] == pytest.approx(T0 + 5)
    before = index.conn.execute("SELECT sig, COUNT(*) AS n FROM failure_members GROUP BY sig").fetchall()
    # 两次真正重跑 delete+insert，而不是让指纹短路掉第二次摄取：改 state.json 的指纹，
    # 每次都走完整的 _upsert_candidates。
    state_path = directory / "state.json"
    original = state_path.read_text(encoding="utf-8")
    state_path.write_text(original + " ", encoding="utf-8")
    index.sync_known()
    state_path.write_text(original, encoding="utf-8")
    index.sync_known()
    after = index.conn.execute("SELECT sig, COUNT(*) AS n FROM failure_members GROUP BY sig").fetchall()
    assert [(row["sig"], row["n"]) for row in after] == [(row["sig"], row["n"]) for row in before]
    assert len(after) == 1 and after[0]["n"] == 2

    clusters = query.failure_clusters()
    assert clusters["total_failures"] == 2
    assert len(clusters["clusters"]) == 1
    cluster = clusters["clusters"][0]
    assert cluster["n"] == 2 and cluster["run_id"] == "conv2d/cluster" and cluster["label"]
    # 示例坐标必须来自同一条候选行，不能把不同行的 run_id/idx/label 拼起来。
    match = index.conn.execute(
        "SELECT label FROM candidates WHERE run_id = ? AND idx = ?",
        (cluster["run_id"], cluster["idx"]),
    ).fetchone()
    assert match is not None and match[0] == cluster["label"]
    # cache_hit 可空：面板报覆盖率而不是比例。
    assert clusters["cache_hit_known"] == 1 and clusters["cache_hit_total"] == 2
    detail_doc = query.failure_cluster(cluster["sig"])
    assert detail_doc is not None and detail_doc["n"] == 2
    assert len(detail_doc["examples"]) == 2
    example = detail_doc["examples"][0]
    assert example["report_path"].endswith("/report.json")
    assert example["feedback_path"].endswith("/feedback.txt")
    assert "candidates/00" in example["report_path"]
    # 消息与 benchmark 字段来自报告本身。
    candidate = dict(index.conn.execute(
        "SELECT message, failure_sig, cv, reps, cache_hit FROM candidates WHERE run_id = 'conv2d/cluster' AND idx = 0"
    ).fetchone())
    assert candidate["failure_sig"] == cluster["sig"]
    assert candidate["message"] and len(candidate["message"]) <= MESSAGE_CHARS
    assert candidate["cv"] == pytest.approx(0.02) and candidate["reps"] == 15 and candidate["cache_hit"] == 0
    assert index.conn.execute(
        "SELECT cache_hit FROM candidates WHERE run_id = 'conv2d/cluster' AND idx = 1"
    ).fetchone()[0] is None
    assert query.failure_clusters(task="relu")["clusters"] == []
    assert query.failure_clusters(stage="abi")["clusters"] == []
    rows = query.failures(sig=cluster["sig"])["failures"]
    assert len(rows) == 2
    assert all(row["message"] for row in rows)
    assert all(row["report_path"].endswith("/report.json") for row in rows)
    assert query.failures(sig="0" * 16)["failures"] == []


def test_an_in_place_candidate_rewrite_refreshes_message_and_members_on_sync(tmp_path, monkeypatch):
    """@brief PR9 新鲜度：候选报告就地改写后，普通 ``sync_known`` 必须更新 message/failure_sig/成员。

    目录 mtime 不会因为改写一个文件而变；旧实现只指纹 ``candidates/`` 目录，于是永远保留旧签名。
    这里不调 ``resolve``、不加 ``force``。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = _cluster_fixture(runs)
    index.sync_known()
    first = index.conn.execute(
        "SELECT failure_sig FROM candidates WHERE run_id = 'conv2d/cluster' AND idx = 0"
    ).fetchone()[0]
    before = index.conn.execute(
        "SELECT COUNT(*) FROM failure_members WHERE run_id = 'conv2d/cluster'"
    ).fetchone()[0]
    assert before == 2
    folder = directory / "candidates" / "000_r1.p0"
    directory_mtime = directory.stat().st_mtime_ns
    (folder / "report.json").write_text(json.dumps({
        "schema_version": 4, "status": "ok", "stage": "done", "speedup": 1.1, "message": "fixed and benchmarked",
    }), encoding="utf-8")
    # 改写文件不会改父目录 mtime——这正是本测试要守住的前提，所以必须分别取改写前后两次。
    assert directory.stat().st_mtime_ns == directory_mtime
    index.sync_known()
    row = index.conn.execute(
        "SELECT status, message, failure_sig FROM candidates WHERE run_id = 'conv2d/cluster' AND idx = 0"
    ).fetchone()
    assert row["status"] == "ok" and row["message"] == "fixed and benchmarked"
    assert row["failure_sig"] is None
    assert index.conn.execute(
        "SELECT COUNT(*) FROM failure_members WHERE run_id = 'conv2d/cluster'"
    ).fetchone()[0] == 1
    assert index.conn.execute(
        "SELECT COUNT(*) FROM failure_members WHERE run_id = 'conv2d/cluster' AND idx = 0"
    ).fetchone()[0] == 0
    remaining = index.conn.execute(
        "SELECT COUNT(*) FROM failure_members WHERE run_id = 'conv2d/cluster'"
    ).fetchone()[0]
    assert remaining == 1 and first is not None


def test_removing_the_whole_candidates_directory_clears_the_members(tmp_path, monkeypatch):
    """@brief PR9：``candidates/`` 整目录消失时，早退不能留下成员行。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    import shutil

    runs, index = _prepare(tmp_path, monkeypatch)
    directory = _cluster_fixture(runs)
    index.sync_known()
    assert index.conn.execute("SELECT COUNT(*) FROM failure_members").fetchone()[0] == 2
    shutil.rmtree(directory / "candidates")
    index.tick()
    assert index.conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0
    assert index.conn.execute("SELECT COUNT(*) FROM failure_members").fetchone()[0] == 0
    assert query.failure_clusters()["total_failures"] == 0


def test_ingest_reads_each_candidate_report_once(tmp_path, monkeypatch):
    """@brief 验收③：写指纹不增加 I/O；每份 ``report.json`` 只被读一次。"""
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = _cluster_fixture(runs)
    reads = []
    real = Index._read_json

    def counting(self, path):
        reads.append(str(path))
        return real(self, path)

    monkeypatch.setattr(Index, "_read_json", counting)
    index.sync_known()
    reports = [path for path in reads if path.endswith("report.json") and "candidates" in path]
    assert sorted(reports) == sorted(str(path) for path in (directory / "candidates").glob("*/report.json"))
    assert len(reports) == len(set(reports)) == 3
    monkeypatch.setattr(Index, "_read_json", real)


def test_the_three_deletion_paths_remove_the_members(tmp_path, monkeypatch):
    """@brief 验收④：``drop_archived`` / ``_upsert_missing`` / ``forget_moved`` 都清掉成员行。"""
    runs, index = _prepare(tmp_path, monkeypatch)
    _cluster_fixture(runs)
    other = _run(runs, "conv2d/missing", started_at=T0 + 6)
    _candidate(other, 0, "r2.p0", {"status": "abi_error", "stage": "abi", "message": "bad ABI"})
    moved = _run(runs, "conv2d/moved", started_at=T0 + 7)
    _candidate(moved, 0, "r3.p0", {"status": "compile_error", "stage": "compile", "message": "Metal compiler rejected the kernel"})
    index.sync_known()
    assert index.conn.execute("SELECT COUNT(*) FROM failure_members").fetchone()[0] == 4

    # _upsert_missing：state 删掉、result 写成坏 JSON → broken run。
    (other / "state.json").unlink()
    (other / "result.json").write_text("{broken", encoding="utf-8")
    index.tick()
    assert index.conn.execute(
        "SELECT COUNT(*) FROM failure_members WHERE run_id = 'conv2d/missing'"
    ).fetchone()[0] == 0
    assert index.conn.execute(
        "SELECT status FROM runs WHERE run_id = 'conv2d/missing'"
    ).fetchone()[0] == "missing"

    index.forget_moved("conv2d/moved")
    assert index.conn.execute(
        "SELECT COUNT(*) FROM failure_members WHERE run_id = 'conv2d/moved'"
    ).fetchone()[0] == 0

    index.drop_archived(["conv2d/cluster"], [])
    assert index.conn.execute("SELECT COUNT(*) FROM failure_members").fetchone()[0] == 0
    assert query.failure_clusters()["total_failures"] == 0


def test_a_rebuilt_database_has_the_same_members(tmp_path, monkeypatch):
    """@brief 验收⑤：删库重建后 ``failure_members`` 的 ``(sig, run_id, idx)`` 与重建前逐行一致。"""
    runs, index = _prepare(tmp_path, monkeypatch)
    _cluster_fixture(runs)
    index.sync_known()
    before = sorted(tuple(row) for row in index.conn.execute("SELECT sig, run_id, idx, status, stage, observed_at FROM failure_members"))
    assert before
    index.conn.close()
    import observe.index as index_mod

    index_mod._SINGLE = None
    db = Path(tmp_path / "observe.sqlite")
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(db) + suffix) if suffix else db
        path.unlink(missing_ok=True)
    rebuilt = Index(db)
    rebuilt.sync_known()
    after = sorted(tuple(row) for row in rebuilt.conn.execute("SELECT sig, run_id, idx, status, stage, observed_at FROM failure_members"))
    assert after == before


def test_the_sig_lookup_uses_the_candidates_sig_index(tmp_path, monkeypatch):
    """@brief 验收⑥：按签名查证据走 ``candidates_sig``，不扫全表。"""
    runs, index = _prepare(tmp_path, monkeypatch)
    _cluster_fixture(runs)
    index.sync_known()
    conn = index.read_only()
    try:
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT c.run_id, c.idx FROM candidates c JOIN runs r ON r.run_id = c.run_id"
            " WHERE c.failure_sig = ? AND r.status != 'missing' ORDER BY COALESCE(r.started_at, 0) DESC LIMIT ?",
            ("deadbeefdeadbeef", 20),
        ).fetchall()
    finally:
        conn.close()
    text = " | ".join(str(row["detail"]) for row in plan)
    assert "candidates_sig" in text, text
    assert "SCAN candidates" not in text, text


def test_a_candidate_signature_does_not_project_sessions(tmp_path, monkeypatch):
    """@brief 回归：候选指纹命中时不能走会话懒投影路径（否则普通 --sig 会白读 transcript）。

    候选与会话的指纹前缀不同（§7.2），所以候选有命中时跳过会话分支；会话分支只在候选没有
    命中时才做懒投影。这里用一场带失败候选与 transcript 的冷索引验证：候选 sig 直接命中、
    sessions 仍为 0 行、也没有写 sessions_fp。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = _run(runs, "conv2d/cand")
    _candidate(directory, 1, "r1.p0", {"status": "abi_error", "stage": "abi", "message": "kernel does not match the required ABI"})
    (directory / "transcript.jsonl").write_text(json.dumps({
        "event": "agent", "label": "r1.p0", "provider": "deepseek", "model": "deepseek-flash",
        "committee": "first", "outcome": "agent_failed", "error": "provider returned 402",
    }) + "\n", encoding="utf-8")
    index.sync_known()
    assert index.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    sig = signature("abi_error", "abi", "kernel does not match the required ABI")
    rows = query.failures(sig=sig)["failures"]
    assert len(rows) == 1 and rows[0]["failure_sig"] == sig
    assert rows[0]["report_path"] and rows[0]["report_path"].endswith("report.json")
    assert index.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0, "候选命中不应投影 sessions"
    assert index.conn.execute(
        "SELECT COUNT(*) FROM meta WHERE key LIKE 'sessions_fp:%'"
    ).fetchone()[0] == 0


def test_a_session_signature_outside_one_lazy_batch_says_it_did_not_finish(tmp_path, monkeypatch):
    """@brief 回归：冷索引上会话懒投影一次只推进一批，没查完必须带 ``sessions_unprojected``。

    旧实现算了 ``unprojected`` 却从不赋值（恒为 0，键永远不出现）：超过一批待投影 run 时，
    目标 run 落在下一批，第一次查询返回空列表而且没有任何“可能没查全”的标记，调用方会把
    空列表读成“没有这个失败”。这里把每批压到 2 个 run，验证第一批给出标记、第二批真找到。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, index = _prepare(tmp_path, monkeypatch)
    failing = _run(runs, "conv2d/old-failure", started_at=T0)
    (failing / "transcript.jsonl").write_text(json.dumps({
        "event": "agent", "label": "r1.p0", "provider": "deepseek", "model": "deepseek-flash",
        "committee": "first", "outcome": "agent_failed", "error": "provider returned 402",
        "runtime": {"failure_class": "turn_failed", "failed_turns": 0},
    }) + "\n", encoding="utf-8")
    for n in range(3):
        newer = _run(runs, f"conv2d/newer{n}", started_at=T0 + 10 + n)
        (newer / "transcript.jsonl").write_text(json.dumps({
            "event": "agent", "label": "r1.p0", "outcome": "evaluated",
        }) + "\n", encoding="utf-8")
    index.sync_known()
    assert index.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    monkeypatch.setattr(query, "SESSIONS_SIG_LAZY_RUNS", 2)
    sig = session_signature("turn_failed", "provider returned 402")
    first = query.failures(sig=sig)
    assert first["failures"] == [], first
    # 批内只投影了 2 个新 run；目标 run 与另一个新 run 还没投影，数量必须说出来。
    assert first.get("sessions_unprojected") == 2, first
    second = query.failures(sig=sig)
    assert [row["run_id"] for row in second["failures"]] == ["conv2d/old-failure"], second
    assert "sessions_unprojected" not in second, second


def test_known_transcriptless_runs_do_not_take_the_lazy_projection_batch(tmp_path, monkeypatch):
    """@brief 回归：冷建已写 ``transcript_bytes=0``，pending 选择不能拿这些 run 填批次。

    旧实现只排除已有 ``sessions_fp`` 的 run：一批无 transcript 的新 run 会反复占满配额，
    目标（更旧的失败会话）永远到不了。这里两个较新的 run 没有 transcript，批大小为 2：
    pending 应直接跳过它们，第一次查询就命中较旧的目标，且不为空目录写投影。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, index = _prepare(tmp_path, monkeypatch)
    target = _run(runs, "conv2d/target", started_at=T0)
    _transcript(target, failing=True)
    for offset in (3, 4):
        _run(runs, f"conv2d/no-transcript-{offset}", started_at=T0 + offset)
    index.sync_known()
    assert index.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    monkeypatch.setattr(query, "SESSIONS_SIG_LAZY_RUNS", 2)
    sig = session_signature("turn_failed", "provider returned 402")
    first = query.failures(sig=sig)
    assert [row["run_id"] for row in first["failures"]] == ["conv2d/target"], first
    keys = [row[0] for row in index.conn.execute(
        "SELECT key FROM meta WHERE key LIKE 'sessions_fp:%' ORDER BY key"
    )]
    assert keys == ["sessions_fp:conv2d/target"], keys


def test_a_transcriptless_run_gets_a_marker_so_the_next_batch_advances(tmp_path, monkeypatch):
    """@brief 回归：老索引没有 ``transcript_bytes`` 时，跳过的 run 要写游标，否则永不前进。

    模拟本功能上线前的库：删掉 ``transcript_bytes``。两个较新 run 没有 transcript，批大小 2，
    第一次只能推进它们（写入 ``0:0`` 游标）；第二次必须命中较旧的目标，第三次仍命中。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, index = _prepare(tmp_path, monkeypatch)
    target = _run(runs, "conv2d/target", started_at=T0)
    _transcript(target, failing=True)
    for offset in (3, 4):
        _run(runs, f"conv2d/no-transcript-{offset}", started_at=T0 + offset)
    index.sync_known()
    index.conn.execute("DELETE FROM meta WHERE key LIKE 'transcript_bytes:%'")
    index.conn.commit()
    # 老库的这些 run 不会再被摄取（指纹没变），所以 meta 不会自愈；关掉读时 tick 才能测到标记路径。
    monkeypatch.setattr(query, "tick", lambda *_a, **_k: (index.revision(), []))
    monkeypatch.setattr(query, "SESSIONS_SIG_LAZY_RUNS", 2)
    sig = session_signature("turn_failed", "provider returned 402")
    first = query.failures(sig=sig)
    assert first["failures"] == [], first
    assert first.get("sessions_unprojected") == 1, first
    markers = {
        row[0]: row[1] for row in index.conn.execute(
            "SELECT key, value FROM meta WHERE key LIKE 'sessions_fp:conv2d/no-transcript-%' ORDER BY key"
        )
    }
    assert markers == {
        "sessions_fp:conv2d/no-transcript-3": "0:0",
        "sessions_fp:conv2d/no-transcript-4": "0:0",
    }, markers
    second = query.failures(sig=sig)
    assert [row["run_id"] for row in second["failures"]] == ["conv2d/target"], second
    third = query.failures(sig=sig)
    assert [row["run_id"] for row in third["failures"]] == ["conv2d/target"], third


def test_the_failures_session_branch_applies_the_exact_status_filter(tmp_path, monkeypatch):
    """@brief 回归：``failures(status=...)`` 对懒投影与已投影两条会话路径都要精确匹配。

    旧实现只给候选查询加 ``c.status``，会话分支漏了 ``s.outcome``：查 ``provider_failed``
    也会返回 ``agent_failed`` 的会话行。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = _run(runs, "conv2d/no-kernel", started_at=T0)
    _transcript(directory, failing=True)
    index.sync_known()
    assert index.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    sig = session_signature("turn_failed", "provider returned 402")
    # 冷索引：先懒投影，再按 outcome 过滤。
    assert query.failures(sig=sig, status="provider_failed")["failures"] == []
    found = query.failures(sig=sig, status="agent_failed")["failures"]
    assert [row["run_id"] for row in found] == ["conv2d/no-kernel"], found
    # 已投影：会话行已在表里，第一个查询直接命中，同样要过滤。
    index.ensure_sessions(directory)
    assert index.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
    assert query.failures(sig=sig, status="provider_failed")["failures"] == []
    found = query.failures(sig=sig, status="agent_failed")["failures"]
    assert [row["run_id"] for row in found] == ["conv2d/no-kernel"], found


def test_the_session_signature_lands_in_its_own_column(tmp_path, monkeypatch):
    """@brief §7.2：会话失败写 ``sessions.failure_sig``；非失败会话为 NULL，且与候选簇不同源。"""
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = _run(runs, "conv2d/sessions")
    (directory / "transcript.jsonl").write_text("".join(json.dumps(row) + "\n" for row in (
        {"event": "agent", "label": "r1.p0", "provider": "deepseek", "model": "deepseek-flash",
         "committee": "first", "outcome": "agent_failed", "error": "Connection error.",
         "runtime": {"failure_class": "turn_failed", "failed_turns": 2}},
        {"event": "agent", "label": "r2.p0", "provider": "deepseek", "model": "deepseek-flash",
         "committee": "second", "outcome": "evaluated"},
    )), encoding="utf-8")
    index.sync_known()
    index.ensure_sessions(directory)
    rows = index.conn.execute(
        "SELECT outcome, failure_sig FROM sessions WHERE run_id = 'conv2d/sessions' ORDER BY seq"
    ).fetchall()
    assert rows[0]["failure_sig"] == session_signature("turn_failed", "Connection error.")
    assert rows[1]["failure_sig"] is None
    # 同一条消息在候选簇里是另一个签名。
    assert rows[0]["failure_sig"] != signature("agent_failed", "agent_failed", "Connection error.")


def test_the_cluster_list_stays_under_the_target_size(tmp_path, monkeypatch):
    """@brief §7.3：20 个簇的列表 ≤4KB（C 类校准目标，测试只记录实测）。"""
    runs, index = _prepare(tmp_path, monkeypatch)
    for number in range(20):
        directory = _run(runs, f"conv2d/c{number:02d}", started_at=T0 + number)
        _candidate(directory, 0, f"r{number}.p0", {
            "status": "incorrect", "stage": "correctness",
            "message": f"failure kind {chr(97 + number)}: {number}/100 elements out of tolerance; worst: o[0, 1, {number}] got -1 want -2",
        })
    index.sync_known()
    document = query.failure_clusters(limit=20)
    assert len(document["clusters"]) == 20
    size = len(json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    assert size <= 4096, f"20 个簇的列表 {size}B 超过 4KB"
    assert document["clusters"][0]["name"]


def test_the_cluster_endpoints_serve_the_list_and_one_cluster(tmp_path, monkeypatch):
    """@brief §7.3：两个端点从索引读；非法 ``limit`` 返回 400，未知签名返回 404。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from starlette.testclient import TestClient

    from observe.app import app

    runs, index = _prepare(tmp_path, monkeypatch)
    _cluster_fixture(runs)
    index.sync_known()
    with TestClient(app) as client:
        document = client.get("/api/v2/failure-clusters").json()
        assert document["total_failures"] == 2 and len(document["clusters"]) == 1
        sig = document["clusters"][0]["sig"]
        one = client.get(f"/api/v2/failure-clusters/{sig}").json()
        assert one["sig"] == sig and one["n"] == 2 and one["examples"]
        assert one["examples"][0]["report_path"].endswith("/report.json")
        assert client.get("/api/v2/failure-clusters?limit=x").status_code == 400
        assert client.get(f"/api/v2/failure-clusters/{sig}?limit=x").status_code == 400
        assert client.get("/api/v2/failure-clusters/" + "0" * 16).status_code == 404


def test_two_hundred_synthetic_runs_keep_one_signature_and_an_additive_count(tmp_path, monkeypatch):
    """@brief PR9 不变量：200 场同 message 不同数字/标签/路径的 run 只产生一个簇，计数只增对应量。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _prepare(tmp_path, monkeypatch)
    for number in range(200):
        directory = _run(runs, f"conv2d/batch_{number:03d}", started_at=T0 + number)
        _candidate(directory, 0, f"r{number}.p0", {
            "status": "incorrect", "stage": "correctness",
            "message": (f"case 'outlier': {number}/100 elements out of tolerance; worst: "
                        f"o[0, 1, {number}] got -0.002 want -0.003 (see /tmp/run{number})"),
        })
    index.sync_known()
    clusters = query.failure_clusters()
    assert len(clusters["clusters"]) == 1 and clusters["clusters"][0]["n"] == 200
    first_sig = clusters["clusters"][0]["sig"]
    index.sync_known()
    assert query.failure_clusters()["clusters"][0]["n"] == 200
    for number in range(200, 240):
        directory = _run(runs, f"conv2d/more_{number:03d}", started_at=T0 + number)
        _candidate(directory, 0, f"r{number}.p0", {
            "status": "incorrect", "stage": "correctness",
            "message": f"case 'other': {number}/50 elements out of tolerance; worst: p[0, 0, 0] got -9 want 0",
        })
    index.sync_known()
    clusters = query.failure_clusters()
    by_sig = {item["sig"]: item["n"] for item in clusters["clusters"]}
    assert by_sig[first_sig] == 200
    assert sorted(by_sig.values()) == [40, 200]


def test_a_residual_error_on_a_successful_session_is_not_a_failure_signature(tmp_path, monkeypatch):
    """@brief PR9：``evaluated``/``unchanged`` 即便残留 error 也不能得到 ``sessions.failure_sig``。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = _run(runs, "conv2d/session_gate")
    (directory / "transcript.jsonl").write_text("".join(json.dumps(row) + "\n" for row in (
        {"event": "agent", "label": "r1.p0", "outcome": "evaluated", "error": "stale connection error",
         "runtime": {"failure_class": "turn_failed"}},
        {"event": "agent", "label": "r2.p0", "outcome": "unchanged", "error": "stale error"},
        {"event": "agent", "label": "r3.p0", "outcome": "agent_failed", "error": "provider failed",
         "runtime": {"failure_class": "turn_failed"}},
    )), encoding="utf-8")
    index.sync_known()
    index.ensure_sessions(directory)
    rows = index.conn.execute(
        "SELECT outcome, failure_sig FROM sessions WHERE run_id = 'conv2d/session_gate' ORDER BY seq"
    ).fetchall()
    assert [row["failure_sig"] for row in rows[:2]] == [None, None]
    assert rows[2]["failure_sig"] is not None
