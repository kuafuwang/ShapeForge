"""PR27 1M 契约：六条可测不变量（不承诺 1M 延迟）。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from observe import detail, query
from observe.index import Index


def _env(tmp_path: Path, monkeypatch) -> Path:
    """@brief 指向临时 runs/ 与数据库。

    @param tmp_path pytest 临时目录。
    @param monkeypatch 环境变量替换。
    @return ``runs/`` 根。
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    monkeypatch.setenv("OBSERVE_TICK_INTERVAL", "0")
    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "0")
    return runs


def _run(root: Path, name: str, *, started_at: float, candidates: int = 1,
         confirmed: bool = False, transcript: str = "") -> Path:
    """@brief 造一个已结束运行。

    @param root ``runs/`` 根。
    @param name 运行名。
    @param started_at unix 秒（用于日桶）。
    @param candidates 候选个数。
    @param confirmed 是否写通过发布门的确认材料。
    @param transcript transcript 文本。
    @return 运行目录。
    """
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": "small", "phase": "done",
        "started_at": started_at, "updated_at": started_at,
    }), encoding="utf-8")
    (directory / "run_config.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": "small", "target": {"id": "apple-m4-max"},
    }), encoding="utf-8")
    folder = directory / "candidates"
    folder.mkdir()
    for index in range(candidates):
        cand = folder / f"{index:03d}_r1.p0"
        cand.mkdir()
        (cand / "kernel.metal").write_text("kernel void k() {}\n", encoding="utf-8")
        # 派生表只从当前（schema-4）报告重建，夹具按真实运行写。
        report = {"schema_version": 4, "status": "ok", "speedup": 1.0 + index / 10}
        if index == 0 and not confirmed:
            report = {"schema_version": 4, "status": "error", "stage": "compile", "message": f"abi_error: missing {name}"}
        (cand / "report.json").write_text(json.dumps(report), encoding="utf-8")
    result = {
        "task": "relu", "dtype": "fp16", "size": "small", "stop_reason": "max_rounds",
        "started_at": started_at, "spent_usd": 0.5, "budget_usd": 2.0, "target_id": "apple-m4-max",
    }
    if confirmed:
        kernel = "kernel void k() {}\n"
        digest = hashlib.sha256(kernel.encode("utf-8")).hexdigest()
        (directory / "best.metal").write_text(kernel, encoding="utf-8")
        from experiments.common import synthetic_confirmation_documents

        confirmed_result, report = synthetic_confirmation_documents(
            task="relu", dtype="fp16", size="small", target_id="apple-m4-max",
            speedup=1.5, label="r1.p0", kernel_sha256=digest,
        )
        result.update({
            key: value for key, value in confirmed_result.items()
            if key.startswith("confirm")
        })
        (directory / "confirm_report.json").write_text(json.dumps(report), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if transcript:
        (directory / "transcript.jsonl").write_text(transcript, encoding="utf-8")
    return directory


def _corpus(root: Path) -> None:
    """@brief 8 场跨两天的语料（1 场确认、1 场失败候选、1 场 transcript）。

    @param root ``runs/`` 根。
    """
    base = 1_700_000_000
    _run(root, "day1/a", started_at=base)
    _run(root, "day1/b", started_at=base + 60, confirmed=True)
    _run(root, "day1/c", started_at=base + 120, transcript=json.dumps({"event": "chat", "reply": "hi"}) + "\n")
    _run(root, "day2/d", started_at=base + 86_400)
    _run(root, "day2/e", started_at=base + 86_400 + 60)
    _run(root, "day2/f", started_at=base + 86_400 + 120)
    _run(root, "day3/g", started_at=base + 2 * 86_400)
    _run(root, "day3/h", started_at=base + 2 * 86_400 + 60)


def _bucket_sql(index: Index) -> list[dict]:
    """@brief 从 ``runs`` 逐行 GROUP BY 出的汇总，用于与 ``run_rollup`` 对拍。

    @param index 索引。
    @return 每个桶一个字典。
    """
    rows = index.conn.execute(
        """
        SELECT CASE WHEN started_at >= 1000000000
                    THEN strftime('%Y-%m-%d', started_at, 'unixepoch') ELSE '' END AS day,
               COALESCE(task, '') AS task, COALESCE(dtype, '') AS dtype, COALESCE(size, '') AS size,
               COALESCE(target_id, '') AS target_id,
               CASE WHEN protocol_version = 'flame_chase_1' THEN '2' ELSE COALESCE(protocol_version, '') END AS protocol_epoch,
               COUNT(*) AS runs,
               SUM(CASE WHEN publication = 'confirmed' THEN 1 ELSE 0 END) AS confirmed,
               SUM(COALESCE(spent_max_usd, 0)) AS spent_usd,
               SUM(COALESCE(published_speedup, 0)) AS published_sum,
               MAX(published_speedup) AS published_max
        FROM runs WHERE status != 'missing'
        GROUP BY day, task, dtype, size, target_id, protocol_epoch
        ORDER BY day, task, dtype, size, target_id, protocol_epoch
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _rollup_rows(index: Index) -> list[dict]:
    """@brief ``run_rollup`` 的全部行，按桶键排序。

    @param index 索引。
    @return 行列表。
    """
    return [dict(row) for row in index.conn.execute(
        "SELECT * FROM run_rollup ORDER BY day, task, dtype, size, target_id, protocol_epoch"
    )]


def test_run_rollup_equals_the_row_group_by_and_a_deletion_moves_one_bucket(tmp_path, monkeypatch):
    """@brief ``run_rollup`` 与逐行 GROUP BY 逐字段一致；删一场只让它的桶少 1。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    _corpus(runs)
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    expected = _bucket_sql(index)
    actual = _rollup_rows(index)
    assert actual == expected, (actual, expected)
    assert sum(row["runs"] for row in actual) == 8
    assert any(row["confirmed"] for row in actual)

    before = {row["day"]: row["runs"] for row in actual}
    index.drop_archived(["day1/a"], [])
    after_rows = _rollup_rows(index)
    after = {row["day"]: row["runs"] for row in after_rows}
    assert after["2023-11-14"] == before["2023-11-14"] - 1
    assert sum(row["runs"] for row in after_rows) == 7
    assert after_rows == _bucket_sql(index)


def test_the_rollup_read_model_matches_ten_thousand_rows_and_never_mixes_keys(tmp_path, monkeypatch):
    """@brief §11.2：10k 行上 ``query.rollup()`` 与逐行 GROUP BY 逐桶一致，分组键不可混算。

    走生产的写入助手（``_rollup_adjust`` + ``_rollup_fields``）而不是测试自算的加总，
    再让读模型（``query.rollup``）与 SQL GROUP BY 对拍。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    from observe import index as index_mod

    _env(tmp_path, monkeypatch)
    index = Index(tmp_path / "observe.sqlite")
    rows = []
    for number in range(10_000):
        rows.append((
            f"conv2d/{number:05d}", "relu" if number % 2 else "conv2d",
            "fp16" if number % 3 else "bf16", "paper" if number % 5 else "small",
            "apple-m4-max", str(3 + number % 2),
            "confirmed" if number % 500 == 0 else "in_progress",
            1.0 + (number % 7) / 10 if number % 500 == 0 else None,
            1_704_067_200 + number * 60,
        ))
    index.conn.executemany(
        "INSERT INTO runs (run_id, task, dtype, size, target_id, protocol_version, publication,"
        " published_speedup, started_at, status, spent_max_usd) VALUES (?,?,?,?,?,?,?,?,?, 'done', 0.5)",
        rows,
    )
    for row in rows:
        fields = index_mod._rollup_fields({
            "started_at": row[8], "task": row[1], "dtype": row[2], "size": row[3],
            "target_id": row[4], "protocol_version": row[5],
        })
        index._rollup_adjust(
            fields, runs=1, confirmed=1 if row[6] == "confirmed" else 0,
            spent=0.5, published=row[7],
        )
    index.conn.commit()
    # 这些 runs 行是直接写进索引的合成行（没有目录），读模型的正确性与对账正交：
    # 读时 tick 会把没有目录的行当“缺失”删掉，这里只量 query.rollup 的聚合。
    monkeypatch.setattr(query, "tick", lambda *_args, **_kwargs: (index.revision(), []))

    expected = _bucket_sql(index)
    actual = _rollup_rows(index)
    assert actual == expected, "10k 行上 run_rollup 与 GROUP BY 必须逐桶一致"
    assert len(actual) == len(expected) and sum(row["runs"] for row in actual) == 10_000

    read = query.rollup(merge_after_days=0)
    assert read["keys"] == ["task", "dtype", "size", "target_id", "protocol_epoch"]
    assert read["merged"] == 0
    assert [row["day"] for row in read["rows"]] == [row["day"] for row in actual]
    for row, source in zip(read["rows"], actual):
        for key in ("task", "dtype", "size", "target_id", "protocol_epoch"):
            assert row[key] == source[key], (key, row, source)
        assert (row["runs"], row["confirmed"]) == (source["runs"], source["confirmed"])
        assert row["spent_usd"] == pytest.approx(source["spent_usd"])
    # 逐桶的分组键唯一：没有任何一行把两个 dtype/纪元加在一起。
    keys = [(row["day"], row["task"], row["dtype"], row["size"], row["target_id"], row["protocol_epoch"]) for row in read["rows"]]
    assert len(keys) == len(set(keys))
    for row in read["rows"]:
        assert row["dtype"] in {"fp16", "bf16"} and row["protocol_epoch"] in {"3", "4"}


def test_the_rollup_read_model_merges_old_days_into_months_without_touching_storage(tmp_path, monkeypatch):
    """@brief §11.2 分层：读时把 >400 天的桶并成月，存储里仍是按天的原桶。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    _corpus(runs)
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    stored = _rollup_rows(index)
    assert all(len(row["day"]) == 10 for row in stored), "存储必须仍是按天"

    # 语料是 2023-11；把“现在”推到两年后，所有桶都应并成 2023-11 一个月。
    later = 1_700_000_000 + 730 * 86_400
    merged = query.rollup(merge_after_days=400, now=later)
    assert merged["rows"], merged
    assert {row["day"] for row in merged["rows"]} == {"2023-11"}
    assert all(row["month"] is True for row in merged["rows"])
    assert merged["merged"] == len(stored)
    assert sum(row["runs"] for row in merged["rows"]) == sum(row["runs"] for row in stored)
    assert sum(row["confirmed"] for row in merged["rows"]) == sum(row["confirmed"] for row in stored)
    # 读时的合并不能改存储。
    assert _rollup_rows(index) == stored


def test_every_derived_table_rebuilds_from_the_same_sources(tmp_path, monkeypatch):
    """@brief 删库重建后 failure_members / run_rollup / confirmed_best / FTS 投影逐字段一致。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    _corpus(runs)
    # 目录史：一条真实可解析的确认记录，让 kernel_history 非空。
    from taic.kernel_catalog import catalog_root

    folder = catalog_root() / "apple-m4-max" / "relu" / "fp16" / "small"
    folder.mkdir(parents=True)
    digest = hashlib.sha256("kernel void k() {}\n".encode("utf-8")).hexdigest()
    (folder / "history.jsonl").write_text(json.dumps({
        "target_id": "apple-m4-max", "task": "relu", "dtype": "fp16", "size": "small",
        "run_id": "day1/b", "winner_label": "r1.p0", "source_sha256": digest,
        "confirmed_speedup": 1.5, "confirmed_values": [1.5], "baseline_us": {},
        "task_fingerprint": "tf", "calibration_sha256": "cal", "updated_at": 1_700_000_000,
    }) + "\n", encoding="utf-8")
    # GPU 事件：一条已配对区间 + 一条未配对 wait，覆盖 gpu_events/gpu_waits/lock_spans。
    events_path = detail.gpu_events_path()
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("".join(json.dumps(row) + "\n" for row in (
        {"t": 1_700_000_000.0, "pid": 123, "kind": "wait", "mode": "shared", "what": "bench"},
        {"t": 1_700_000_001.0, "pid": 123, "kind": "acquire", "mode": "shared", "what": "bench"},
        {"t": 1_700_000_002.0, "pid": 456, "kind": "wait", "mode": "exclusive", "what": "eval"},
    )), encoding="utf-8")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    index.ensure_gpu_events()
    index._sync_kernels()

    def snapshot(instance: Index) -> dict:
        return {
            "failure_members": [dict(row) for row in instance.conn.execute(
                "SELECT * FROM failure_members ORDER BY run_id, idx")],
            "run_rollup": _rollup_rows(instance),
            "confirmed_best": [dict(row) for row in instance.conn.execute(
                "SELECT * FROM confirmed_best ORDER BY run_id")],
            "kernel_history": [dict(row) for row in instance.conn.execute(
                "SELECT * FROM kernel_history ORDER BY target_id, task, dtype, size, seq")],
            "gpu_events": [dict(row) for row in instance.conn.execute(
                "SELECT rowid, t, pid, kind, mode, what FROM gpu_events ORDER BY rowid")],
            "gpu_waits": [dict(row) for row in instance.conn.execute(
                "SELECT * FROM gpu_waits ORDER BY pid, seq")],
            "lock_spans": [dict(row) for row in instance.conn.execute(
                "SELECT pid, start, end, mode, what FROM lock_spans ORDER BY start, pid")],
            "runs_fts": [dict(row) for row in instance.conn.execute(
                "SELECT rowid, run_id, task, dtype, target_id, model, stop_reason FROM runs_fts ORDER BY rowid")],
            "cands_fts": [dict(row) for row in instance.conn.execute(
                "SELECT rowid, label, message FROM cands_fts ORDER BY rowid")],
        }

    before = snapshot(index)
    assert before["failure_members"] and before["run_rollup"] and before["confirmed_best"]
    assert before["kernel_history"], "kernel_history 必须有可重建的非空行"
    assert before["kernel_history"][0]["run_id"] == "day1/b"
    assert before["gpu_events"] and before["gpu_waits"] and before["lock_spans"]
    assert before["runs_fts"] and before["cands_fts"]
    index.conn.close()
    index._discard()
    fresh = Index(tmp_path / "observe.sqlite")
    fresh.sync_known(force=True)
    fresh.ensure_gpu_events()
    fresh._sync_kernels()
    assert snapshot(fresh) == before


def test_discarding_the_index_keeps_the_blob_store(tmp_path, monkeypatch):
    """@brief 删库只删 SQLite 文件，不碰 blob 目录（证据不静默消失）。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    _env(tmp_path, monkeypatch)
    index = Index(tmp_path / "observe.sqlite")
    blob = tmp_path / "data" / "blobs" / "ab" / "abc"
    blob.parent.mkdir(parents=True)
    blob.write_text("kernel void k() {}\n", encoding="utf-8")
    index.conn.close()
    index._discard()
    assert blob.read_text(encoding="utf-8") == "kernel void k() {}\n"
    assert not (tmp_path / "observe.sqlite").exists()


def test_read_paths_decode_no_row_json_and_touch_no_run_directory(tmp_path, monkeypatch):
    """@brief 首页读模型不解码 ``row_json``、不列运行目录（热路径不读目录）。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    _corpus(runs)
    from observe.index import get_index

    index = get_index()
    index.sync_known(force=True)
    with index.conn:
        index.conn.execute("UPDATE runs SET row_json = '{broken'")
    index._prepare_frame()
    page = query.runs_page(limit=50)
    query.patrol()
    query.budget_report()
    query.overview()
    query.live_overview()
    query.failure_clusters()
    query.best_confirmed()
    assert page["runs"] and page["total"] == 8
    assert index.stats["dir_lists"] == 0, index.stats
    assert index.stats["json_reads"] == 0, index.stats


def test_every_list_endpoint_has_a_cursor_and_a_declared_cap(tmp_path, monkeypatch):
    """@brief 列表读模型都带游标/上限字段，并且不超过声明的上限。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    _corpus(runs)
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    page = query.runs_page(limit=3)
    assert set(page) >= {"runs", "total", "next", "limit"} and len(page["runs"]) <= 3 and page["next"]
    board = query.feedback_board(limit=5)
    assert "next_offset" in board and len(board["upserts"]) <= 5
    experiments = query.experiments(limit=2)
    assert set(experiments) >= {"experiments", "total", "next"} and len(experiments["experiments"]) <= 2
    from observe.deletions import list_deletions

    listed = list_deletions(limit=2)
    assert set(listed) >= {"deletions", "next", "truncated"} and len(listed["deletions"]) <= 2
    candidates = query.candidates_page("day1/a", limit=1)
    assert candidates["limit"] == 1 and len(candidates["candidates"]) <= 1


def test_the_full_feedback_export_is_byte_capped_and_says_so(tmp_path, monkeypatch):
    """@brief §11.5：``?full=1`` 的全量导出有 512KB 字节上限，超了截断并带 ``truncated``。

    这条不走 HTTP：直接把足够大的 claim 行写进索引，读模型自己必须截断。
    `full=True` 的路径此前没有任何测试（只有 ``limit=100`` 的探针），截断分支是死代码。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    from observe import feedback

    _env(tmp_path, monkeypatch)
    index = Index(tmp_path / "observe.sqlite")
    text = "x" * 2000
    rows = []
    claims = []
    for number in range(400):
        run_id = f"conv2d/full{number:03d}"
        rows.append((run_id, 1, float(number), "conv2d", "fp16", "paper", "r1.p0"))
        claims.append((run_id, 1, f"digest-{number}", "对系统的建议", "system", f"前提 {number}", "pending"))
    index.conn.executemany(
        "INSERT INTO feedback_items (run_id, seq, t, task, dtype, size, label)"
        " VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    index.conn.executemany(
        "INSERT OR REPLACE INTO feedback_claims (run_id, seq, digest, section, audience, premise, status)"
        " VALUES (?,?,?,?,?,?,?)",
        claims,
    )
    index.conn.execute(
        "UPDATE feedback_claims SET digest = digest || ?", ("y" * 2000,),
    )
    index.conn.commit()
    board = query.feedback_board(full=True)
    assert board.get("truncated") is True, sorted(board)
    encoded = json.dumps(board, ensure_ascii=False, default=str).encode("utf-8")
    assert len(encoded) <= feedback.FEEDBACK_FULL_BYTES, len(encoded)
    assert 0 < len(board["upserts"]) < 400
    page = query.feedback_board(limit=20)
    assert page.get("truncated") is not True and len(page["upserts"]) == 20


def test_the_full_export_cap_includes_the_envelope_not_only_the_rows(tmp_path, monkeypatch):
    """@brief 边界：单行几乎占满 512KB 时，最终 body（含 envelope/calibration/truncated）也不能超。

    旧实现只累加行字节：第一行比上限少 20B，于是保留它，加上外壳后响应超限 366B。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    from observe import feedback

    _env(tmp_path, monkeypatch)
    index = Index(tmp_path / "observe.sqlite")
    index.conn.execute(
        "INSERT INTO feedback_items (run_id, seq, t, task, dtype, size, label)"
        " VALUES ('conv2d/big', 1, 1.0, 'relu', 'fp16', 'paper', 'r1')",
    )
    index.conn.execute(
        "INSERT INTO feedback_claims (run_id, seq, digest, section, audience, premise, status, rev)"
        " VALUES ('conv2d/big', 1, 'd1', '对系统的建议', 'system', 'p', 'pending', 1)",
    )
    index.conn.execute(
        "INSERT INTO feedback_items (run_id, seq, t, task, dtype, size, label)"
        " VALUES ('conv2d/small', 1, 0.5, 'relu', 'fp16', 'paper', 'r1')",
    )
    index.conn.execute(
        "INSERT INTO feedback_claims (run_id, seq, digest, section, audience, premise, status, rev)"
        " VALUES ('conv2d/small', 1, 'd2', '对系统的建议', 'system', 'p', 'pending', 1)",
    )
    index.conn.commit()
    # 用生产投影量一行单独出现时的真实字节数，再把这一行撑到“比上限少 20B”：外壳会把它顶出去。
    base_row = len(json.dumps(
        feedback.review_feed(index.conn, since=-1)["upserts"][0], ensure_ascii=False, default=str,
    ).encode("utf-8"))
    extra = max(0, feedback.FEEDBACK_FULL_BYTES - 20 - base_row)
    index.conn.execute("UPDATE feedback_claims SET digest = digest || ? WHERE run_id = 'conv2d/big'", ("x" * extra,))
    index.conn.commit()
    assert feedback.FEEDBACK_FULL_BYTES - 20 - base_row < len(
        json.dumps(feedback.review_feed(index.conn, since=-1)["upserts"][0], ensure_ascii=False, default=str).encode("utf-8"),
    )

    board = query.feedback_board(full=True)
    encoded = len(json.dumps(board, ensure_ascii=False, default=str).encode("utf-8"))
    assert board.get("truncated") is True
    assert encoded <= feedback.FEEDBACK_FULL_BYTES, (encoded, feedback.FEEDBACK_FULL_BYTES)
    assert len(board["upserts"]) < 2, "the near-cap row must be dropped when the envelope no longer fits"


class _CountingHandle:
    """@brief 包住真实文件句柄，统计读出的字节数。"""

    def __init__(self, handle) -> None:
        self.handle = handle
        self.bytes = 0

    def __enter__(self):
        self.handle.__enter__()
        return self

    def __exit__(self, *exc):
        return self.handle.__exit__(*exc)

    def __iter__(self):
        for line in self.handle:
            self.bytes += len(line.encode("utf-8")) if isinstance(line, str) else len(line)
            yield line

    def read(self, *args):
        data = self.handle.read(*args)
        self.bytes += len(data.encode("utf-8")) if isinstance(data, str) else len(data)
        return data

    def __getattr__(self, name):
        return getattr(self.handle, name)


def test_cold_build_reads_zero_transcript_bytes_and_the_flame_cap_applies(tmp_path, monkeypatch):
    """@brief 2KB 与 1MB transcript 的冷建都不读 transcript；>1MB 的 flame_chase 也不再全读。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    small = json.dumps({"event": "agent", "label": "s1", "tokens": {"output": 1}}) + "\n" + json.dumps({"event": "chat", "reply": "x" * 2000}) + "\n"
    big = json.dumps({"event": "agent", "label": "big", "tokens": {"output": 1}}) + "\n" + json.dumps({"event": "chat", "reply": "x" * 1_100_000}) + "\n"
    _run(runs, "small", started_at=1_700_000_000, transcript=small)
    _run(runs, "big", started_at=1_700_000_100, transcript=big)
    (runs / "big" / "run_config.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": "small", "loop_protocol": "flame_chase_1",
        "target": {"id": "apple-m4-max"},
    }), encoding="utf-8")

    real_open = Path.open
    total = {"bytes": 0, "handles": []}

    def counting_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        if self.name != "transcript.jsonl":
            return handle
        wrapped = _CountingHandle(handle)
        total["handles"].append(wrapped)
        return wrapped

    monkeypatch.setattr(Path, "open", counting_open)
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    assert index.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
    assert total["bytes"] == 0, total["bytes"]
    assert index.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0

    # 首次读 sessions 才懒投影；>1MB 的 flame_chase_1 也按上限跳过。
    small_rows = query.sessions_page("small")["sessions"]
    assert small_rows and len(small_rows) == 1
    assert query.sessions_page("big")["sessions"] == []


def test_deleting_the_only_confirmed_run_subtracts_confirmed_and_published(tmp_path, monkeypatch):
    """@brief 删除唯一 confirmed run：confirmed 减 1、published_sum 减 1.5、max 归空，且与逐行 GROUP BY 一致。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    _corpus(runs)
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    before = {row["day"]: row for row in _rollup_rows(index)}
    assert before["2023-11-14"]["confirmed"] == 1
    assert before["2023-11-14"]["published_sum"] == 1.5
    index.drop_archived(["day1/b"], [])
    after = {row["day"]: row for row in _rollup_rows(index)}
    assert after["2023-11-14"]["confirmed"] == 0, after
    assert after["2023-11-14"]["published_sum"] == 0, after
    assert after["2023-11-14"]["published_max"] is None, after
    assert _rollup_rows(index) == _bucket_sql(index)


def test_a_confirmed_run_turning_unconfirmed_moves_the_bucket(tmp_path, monkeypatch):
    """@brief confirmed → 未确认：旧桶减 confirmed/published，新桶不计，两表逐字段一致。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    _corpus(runs)
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    (runs / "day1" / "b" / "confirm_report.json").write_text(
        json.dumps({"status": "failed", "speedup": None}), encoding="utf-8")
    index.sync_known(force=True)
    row = {item["day"]: item for item in _rollup_rows(index)}["2023-11-14"]
    assert row["confirmed"] == 0 and row["published_sum"] == 0 and row["published_max"] is None, row
    assert _rollup_rows(index) == _bucket_sql(index)
    assert index.conn.execute(
        "SELECT publication FROM runs WHERE run_id = 'day1/b'"
    ).fetchone()[0] != "confirmed"
