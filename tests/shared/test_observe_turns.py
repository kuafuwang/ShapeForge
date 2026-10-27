"""The turn table is a projection of transcript.jsonl. It does not redraw the detail page."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from observe.detail import turns_feed
from observe.index import get_index
from observe.query import feature_grid, run_detail, run_summary


def _prepare(tmp_path, monkeypatch) -> tuple[Path, object]:
    """@brief 把运行目录、索引库、账本与 carryover 都指到临时目录，并取回索引单例。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来设置环境变量。
    @return (运行根目录, 索引对象) 二元组。
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "missing-carryover.json"))
    return runs, get_index()


def _running(directory: Path) -> None:
    """@brief 在目录里写一份「正在运行」的 state.json，pid 用当前进程。

    @param directory 运行目录。
    """
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "phase": "llm", "pid": os.getpid(),
        "started_at": time.time(), "updated_at": time.time(), "round": 1,
    }))


def _line(event: str, label: str, **fields) -> str:
    """@brief 拼一行 transcript.jsonl。

    @param event 事件名。
    @param label 该行对应的候选标签。
    @param fields 追加到行里的其它字段。
    @return 以换行结尾的 JSON 文本。
    """
    return json.dumps({"event": event, "label": label, **fields}, ensure_ascii=False) + "\n"


def _rows(index, run_id: str) -> list[dict]:
    """@brief 读出索引里某次运行已投影的回合行，并把 seq 合进行的正文。

    @param index 观察索引。
    @param run_id 运行 id。
    @return 按 seq 排序的回合字典列表。
    """
    out = []
    for item in index.conn.execute("SELECT seq, kind, label, row_json FROM turns WHERE run_id = ? ORDER BY seq", (run_id,)):
        body = json.loads(item["row_json"])
        body["seq"] = item["seq"]
        out.append(body)
    return out


def test_a_running_search_catches_up_without_redrawing_the_detail(tmp_path, monkeypatch):
    """@brief 验证运行中的搜索只增量读入新增的 transcript 字节、不重绘详情页，transcript 被缩短时重建回合表，索引库重建后回合仍能复原。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来把索引与账本指到临时路径。
    """
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = runs / "conv"
    _running(directory)
    transcript = directory / "transcript.jsonl"
    first = _line("chat", "init", model="kimi-k3", seconds=1.5, reply={"content": "hello"})
    transcript.write_bytes(first.encode() + b'{"event":"chat","label":"init2"')
    revision, _changed = index.tick()
    assert index.stats["turn_parses"] == 1
    assert index.conn.execute("SELECT turn_bytes FROM runs WHERE run_id = 'conv'").fetchone()["turn_bytes"] == len(first.encode())
    stamp = run_detail("conv")["detail_stamp"]
    summary_stamp = run_summary("conv")["detail_stamp"]
    turns = index.turns_revision()
    with transcript.open("ab") as handle:
        handle.write(b',"model":"kimi-k3","reply":{"content":"second"}}\n')
    index.tick()
    assert index.stats["turn_parses"] == 1
    assert index.stats["turn_seek"] == len(first.encode())
    revision_after, changed = index.tick()
    # the line was already consumed by the tick above; this frame finds nothing new
    assert revision_after == revision
    assert changed == []
    assert index.turns_revision() == turns + 1
    # 追加对话只前移回合版本，不动摘要戳（§5.2）；但完整详情含 timeline（问模型泳道读 transcript），
    # 它自己的戳必须前移。共用摘要戳会让完整详情的旧负载一直命中 same（同一类陈旧响应 bug）。
    assert run_summary("conv")["detail_stamp"] == summary_stamp
    moved = run_detail("conv")["detail_stamp"]
    assert moved != stamp
    assert run_detail("conv", stamp=stamp).get("same") is not True
    assert run_detail("conv", stamp=moved) == {"same": True, "id": "conv", "detail_stamp": moved}
    rows = _rows(index, "conv")
    assert [row["seq"] for row in rows] == [1, 2]
    assert rows[1]["reply"] == "second"

    transcript.write_text(_line("chat", "only", reply={"content": "kept"}))
    index.tick()
    shrunk = _rows(index, "conv")
    assert [row["label"] for row in shrunk] == ["only"]
    assert index.conn.execute("SELECT turn_bytes FROM runs WHERE run_id = 'conv'").fetchone()["turn_bytes"] == transcript.stat().st_size

    before = [(row["seq"], row["label"], row["reply"]) for row in shrunk]
    index.conn.close()
    import observe.index as index_mod

    index_mod._SINGLE = None
    db = Path(os.environ["OBSERVE_DB"])
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True) if suffix else db.unlink(missing_ok=True)
    rebuilt = get_index()
    rebuilt.tick()
    assert [(row["seq"], row["label"], row["reply"]) for row in _rows(rebuilt, "conv")] == before


def test_turns_hide_the_system_prompt_the_kernel_and_secrets(tmp_path, monkeypatch):
    """@brief 验证回合投影隐去系统提示词、密钥、工具参数与内核全文，并把回复与思考截断到上限。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来把索引与账本指到临时路径。
    """
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = runs / "conv"
    _running(directory)
    kernel = "K" * 2000
    thinking = "T" * 2000
    (directory / "transcript.jsonl").write_text(_line(
        "chat", "init", model="kimi-k3",
        messages=[
            {"role": "system", "content": "SYSTEM_ONLY_MARKER"},
            {"role": "user", "content": "please sk-SUPERSECRETTOKEN"},
        ],
        reply={
            "content": kernel,
            "reasoning_content": thinking,
            "tool_calls": [{"function": {"name": "quick_check", "arguments": "KERNEL_ARGUMENT_MARKER" + ("Z" * 3000)}}],
        },
        usage={"prompt_tokens": 3, "completion_tokens": 4},
        cost_usd=1.25,
    ))
    index.tick()
    blob = json.dumps(_rows(index, "conv"), ensure_ascii=False)
    assert "SYSTEM_ONLY_MARKER" not in blob
    assert "SUPERSECRETTOKEN" not in blob
    assert "KERNEL_ARGUMENT_MARKER" not in blob
    assert kernel not in blob
    assert thinking not in blob
    row = _rows(index, "conv")[0]
    assert len(row["reply"]) <= 800
    assert len(row["thinking"]) <= 300
    assert row["tools"] == ["quick_check"]
    assert row["sent"] == "please <redacted>"


def test_same_label_merges_until_the_label_comes_back_later(tmp_path, monkeypatch):
    """@brief 验证同一标签的多行（含重试）合并成一个回合，标签再次出现时另起一段，分页游标能逐段往回走。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来把索引与账本指到临时路径。
    """
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = runs / "conv"
    _running(directory)
    (directory / "transcript.jsonl").write_text("".join([
        _line("chat", "A", reply={"content": "one"}),
        _line("retry", "B", error="timeout talking"),
        _line("chat", "B", reply={"content": "body"}),
        _line("chat", "A", reply={"content": "two"}),
    ]))
    index.tick()
    tail = turns_feed("conv", limit=1)
    assert tail["rows"][0]["label"] == "A" and tail["rows"][0]["min_seq"] == 4 and tail["rows"][0]["reply"] == "two"
    assert tail["cursor"] == 4
    middle = turns_feed("conv", before=tail["cursor"], limit=1)
    assert middle["rows"][0]["label"] == "B" and middle["rows"][0]["min_seq"] == 2
    assert middle["rows"][0]["reply"] == "body"
    assert any("timeout" in item for item in middle["rows"][0]["retries"])
    assert middle["cursor"] == 2
    oldest = turns_feed("conv", before=middle["cursor"], limit=1)
    assert oldest["rows"][0]["label"] == "A" and oldest["rows"][0]["min_seq"] == 1 and oldest["rows"][0]["reply"] == "one"
    assert oldest["cursor"] is None
    assert turns_feed("missing") is None
    assert turns_feed("a/../../etc") is None


def test_an_unchanged_stamp_does_not_read_the_turns_table(tmp_path, monkeypatch):
    """@brief 验证 stamp 未变时获取回合不再读 turns 表，直接返回 same。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来把索引与账本指到临时路径。
    """
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = runs / "conv"
    _running(directory)
    (directory / "transcript.jsonl").write_text(_line("chat", "init", reply={"content": "hello"}))
    index.tick()
    feed = turns_feed("conv")
    index.stats["turn_reads"] = 0
    again = turns_feed("conv", stamp=feed["stamp"])
    assert again == {"same": True, "stamp": feed["stamp"]}
    assert index.stats["turn_reads"] == 0


def _finished(directory: Path) -> None:
    """@brief 在目录里写一份已结束运行的 state.json 与 result.json。

    @param directory 运行目录。
    """
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "phase": "done", "started_at": 1, "updated_at": 1,
    }))
    (directory / "result.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.1, "budget_usd": 1,
    }))


def test_a_finished_run_is_projected_when_its_turn_list_opens(tmp_path, monkeypatch):
    """@brief 验证已结束的运行不会在后台索引里预投影回合，直到有人打开它的回合列表才读入，之后 stamp 未变也不再解析。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来把索引与账本指到临时路径。
    """
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = runs / "conv"
    _finished(directory)
    (directory / "transcript.jsonl").write_text(_line("chat", "init", reply={"content": "kept"}))
    index.tick()
    assert index.conn.execute("SELECT COUNT(*) FROM turns WHERE run_id = 'conv'").fetchone()[0] == 0
    assert index.conn.execute("SELECT turn_bytes FROM runs WHERE run_id = 'conv'").fetchone()[0] == 0
    feed = turns_feed("conv")
    assert [row["reply"] for row in feed["rows"]] == ["kept"]
    assert index.conn.execute("SELECT turn_bytes FROM runs WHERE run_id = 'conv'").fetchone()[0] > 0
    index.stats["turn_parses"] = 0
    again = turns_feed("conv", stamp=feed["stamp"])
    assert again["same"] is True
    assert index.stats["turn_parses"] == 0


def test_a_run_that_already_has_turns_still_catches_the_line_written_as_it_finishes(tmp_path, monkeypatch):
    """@brief 验证已有回合的运行在写出 result.json 时，仍能补上它结束时新写的那一行。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来把索引与账本指到临时路径。
    """
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = runs / "conv"
    _running(directory)
    transcript = directory / "transcript.jsonl"
    transcript.write_text(_line("chat", "init", reply={"content": "one"}))
    index.tick()
    (directory / "result.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.1, "budget_usd": 1,
    }))
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(_line("chat", "init2", reply={"content": "two"}))
    index.tick()
    assert [row["reply"] for row in _rows(index, "conv")] == ["one", "two"]


def test_feature_grid_signs_a_source_once_and_then_uses_the_cache(tmp_path, monkeypatch):
    """@brief 验证 feature_grid 对每份不同源码只算一次特征签名（三份文件两份源码就是两次），之后命中缓存，即使 state.json 变了也不再重算。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来包一层能计数的 feature_signature。
    """
    runs, _index = _prepare(tmp_path, monkeypatch)
    directory = runs / "gemm"
    directory.mkdir()
    state = {"task": "gemm", "dtype": "fp16", "phase": "done", "started_at": 1, "updated_at": 1}
    (directory / "state.json").write_text(json.dumps(state))
    (directory / "result.json").write_text(json.dumps({"task": "gemm", "dtype": "fp16", "stop_reason": "max_rounds"}))
    same = "kernel void same() {}\n"
    other = "kernel void other() {}\n"
    for name, source, speed in (("000_slow", same, 0.2), ("001_fast", same, 1.5), ("002_other", other, 0.9)):
        path = directory / "candidates" / name
        path.mkdir(parents=True)
        (path / "kernel.metal").write_text(source)
        (path / "report.json").write_text(json.dumps({"speedup": speed, "status": "ok"}))
    import observe.query as query

    calls = {"n": 0}
    real = query.feature_signature

    def wrapped(source, *, backend_id):
        calls["n"] += 1
        return real(source, backend_id=backend_id or "metal")

    monkeypatch.setattr(query, "feature_signature", wrapped)
    query._GRID_CACHE.clear()
    grid = feature_grid(directory, {"confirm_winner_label": "fast"})
    assert calls["n"] == 2  # three files, two distinct sources
    assert [row["label"] for row in grid] == ["fast"]
    assert grid[0]["speedup"] == 1.5
    feature_grid(directory, {"confirm_winner_label": "fast"})
    assert calls["n"] == 2
    state["updated_at"] = 2
    (directory / "state.json").write_text(json.dumps(state))
    feature_grid(directory, {"confirm_winner_label": "fast"})
    # The grid key changes with state.json, but the source signatures are cached by source sha.
    assert calls["n"] == 2


def test_fa_paper_signs_each_distinct_source_once(tmp_path, monkeypatch):
    """@brief 用一个真实运行目录验证特征签名按源码摘要去重：每份不同源码只算一次，stamp 未变时不再重算。

    仓库里没有该运行时直接跳过。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来设置环境变量并包一层能计数的 feature_signature。
    """
    directory = Path("runs/apple-m4-max/p1/fa_paper")
    candidates = directory / "candidates"
    if not candidates.is_dir():
        pytest.skip("fa_paper is not in this checkout")
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(Path("runs").resolve()))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "missing-carryover.json"))
    distinct = set()
    for path in candidates.iterdir():
        source = path / "kernel.metal"
        if source.is_file():
            distinct.add(hashlib.sha1(source.read_bytes()).hexdigest())
    import observe.query as query

    calls = {"n": 0}
    real = query.feature_signature

    def wrapped(source, *, backend_id):
        calls["n"] += 1
        return real(source, backend_id=backend_id or "metal")

    monkeypatch.setattr(query, "feature_signature", wrapped)
    query._GRID_CACHE.clear()
    # 详情读路径不再摄取索引；要拿到 detail_stamp 就先让索引收录这一场（页面由列表页完成）。
    get_index().resolve("apple-m4-max/p1/fa_paper")
    detail = run_detail("apple-m4-max/p1/fa_paper")
    assert calls["n"] == len(distinct)
    assert detail["detail_stamp"]
    run_detail("apple-m4-max/p1/fa_paper")
    assert calls["n"] == len(distinct)
    assert run_detail("apple-m4-max/p1/fa_paper", stamp=detail["detail_stamp"])["same"] is True
    assert calls["n"] == len(distinct)


def _reference_groups(index, run_id: str) -> list[dict]:
    """@brief 独立的逐组期望值：把整份投影行按相邻同标签分组（旧 ``_turn_groups`` 语义）。

    @param index 观察索引。
    @param run_id 运行 id。
    @return 每组含 label/min_seq/max_seq/事件数。
    """
    from observe.detail import _turn_groups

    events = _rows(index, run_id)
    return [
        {"label": group["label"], "min_seq": group["min_seq"], "max_seq": group["max_seq"],
         "count": len(group["events"])}
        for group in _turn_groups(events)
    ]


def test_the_grp_projection_matches_the_adjacent_label_reference(tmp_path, monkeypatch):
    """@brief §5.3 验收①/②：三份夹具逐组等价，且跨多次摄入的分组边界与整段重算一致。

    ① 同标签重试共组；② 续跑时同名标签再次出现另起一组（分两次摄入）；③ 单组 500 事件。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = runs / "conv"
    _running(directory)
    transcript = directory / "transcript.jsonl"
    lines = [
        _line("chat", "A", reply={"content": "first"}),
        _line("retry", "A", error="timeout talking"),
        _line("chat", "A", reply={"content": "still A"}),
        _line("chat", "B", reply={"content": "B"}),
    ]
    transcript.write_text("".join(lines))
    index.tick()
    # 续跑：同名 A 在 B 之后再出现，必须另起一组；再追加一条 B 又一组。
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(_line("chat", "A", reply={"content": "A again"}))
    index.tick()
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(_line("chat", "B", reply={"content": "B again"}))
    index.tick()
    assert _reference_groups(index, "conv") == [
        {"label": "A", "min_seq": 1, "max_seq": 3, "count": 3},
        {"label": "B", "min_seq": 4, "max_seq": 4, "count": 1},
        {"label": "A", "min_seq": 5, "max_seq": 5, "count": 1},
        {"label": "B", "min_seq": 6, "max_seq": 6, "count": 1},
    ]
    feed = turns_feed("conv", limit=2)
    assert [(row["label"], row["min_seq"], row["max_seq"]) for row in feed["rows"]] == [("A", 5, 5), ("B", 6, 6)]
    older = turns_feed("conv", before=feed["cursor"], limit=2)
    assert [(row["label"], row["min_seq"], row["max_seq"]) for row in older["rows"]] == [("A", 1, 3), ("B", 4, 4)]
    assert older["cursor"] is None
    # 空标签每行一组。
    _running(runs / "blank")
    (runs / "blank" / "transcript.jsonl").write_text("".join([
        _line("chat", "", reply={"content": "x"}),
        _line("chat", "", reply={"content": "y"}),
    ]))
    index.tick()
    assert [row["count"] for row in _reference_groups(index, "blank")] == [1, 1]


def _many_groups(directory: Path, count: int) -> None:
    """@brief 写一场 ``count`` 组的 transcript：每组一行，标签 c0,c1,...

    @param directory 运行目录。
    @param count 组数。
    """
    _running(directory)
    with (directory / "transcript.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(_line("chat", f"c{index}", reply={"content": "r"}))


def test_a_page_reads_only_its_own_groups_and_does_not_grow_with_the_run(tmp_path, monkeypatch):
    """@brief §5.3 验收③：一页只读本页组的事件总数，100 组与 10,000 组的 run 读行数相同。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from observe import detail

    runs, index = _prepare(tmp_path, monkeypatch)
    _many_groups(runs / "small", 100)
    _many_groups(runs / "huge", 10_000)
    index.tick()

    parsed: list[int] = []
    real = detail._parse_turn_rows

    def counting(rows):
        parsed.append(len(rows))
        return real(rows)

    monkeypatch.setattr(detail, "_parse_turn_rows", counting)
    small = turns_feed("small", limit=40)
    small_rows = sum(parsed)
    parsed.clear()
    huge = turns_feed("huge", limit=40)
    huge_rows = sum(parsed)
    assert len(small["rows"]) == 40 and len(huge["rows"]) == 40
    assert small_rows == huge_rows == 40, (small_rows, huge_rows)
    assert small["cursor"] is not None and huge["cursor"] is not None
    monkeypatch.setattr(detail, "_parse_turn_rows", real)


def test_a_500_event_group_is_truncated_in_sql_and_all_1_returns_it(tmp_path, monkeypatch):
    """@brief §5.3 验收④：单组 500 事件只读最新 400 条并标 ``truncated_events``；``all=1`` 读回全量。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from observe import detail

    runs, index = _prepare(tmp_path, monkeypatch)
    directory = runs / "big"
    _running(directory)
    with (directory / "transcript.jsonl").open("w", encoding="utf-8") as handle:
        for _ in range(500):
            handle.write(_line("retry", "A", error="timeout talking"))
    index.tick()
    assert _reference_groups(index, "big") == [{"label": "A", "min_seq": 1, "max_seq": 500, "count": 500}]
    grp = index.conn.execute("SELECT DISTINCT grp FROM turns WHERE run_id = 'big'").fetchone()[0]

    parsed: list[int] = []
    real = detail._parse_turn_rows

    def counting(rows):
        parsed.append(len(rows))
        return real(rows)

    monkeypatch.setattr(detail, "_parse_turn_rows", counting)
    feed = turns_feed("big", limit=10)
    assert sum(parsed) == detail.TURN_GROUP_EVENTS
    monkeypatch.setattr(detail, "_parse_turn_rows", real)
    assert len(feed["rows"]) == 1
    row = feed["rows"][0]
    assert row["truncated_events"] == 100 and len(row["retries"]) == 400
    assert (row["min_seq"], row["max_seq"]) == (1, 500)

    full = turns_feed("big", grp=grp, everything=True)
    assert full["all"] is True and full["grp"] == grp
    assert len(full["rows"]) == 1 and len(full["rows"][0]["retries"]) == 500
    assert "truncated_events" not in full["rows"][0]


def test_the_two_group_queries_and_the_rows_query_use_the_group_index(tmp_path, monkeypatch):
    """@brief §5.3 验收⑤：实现真正发出的三条 ``FROM turns`` 语句都走 ``turns_run_grp``，不 ``SCAN turns``。

    这里 EXPLAIN 的是 trace 抓到的真实语句，不是测试里手抄的 SQL——手抄件会和实现一起漂移。
    跨组 ``ORDER BY seq`` 可能出现 ``USE TEMP B-TREE``，这是正常的。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from observe.index import Index

    runs, index = _prepare(tmp_path, monkeypatch)
    _many_groups(runs / "conv", 60)
    index.tick()
    turns_feed("conv", limit=40)  # 预热投影
    real = Index.read_only
    statements: list[str] = []

    def instrumented(self):
        conn = real(self)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(Index, "read_only", instrumented)
    statements.clear()
    turns_feed("conv", limit=40)
    monkeypatch.setattr(Index, "read_only", real)

    turn_sql = [sql for sql in statements if "FROM turns" in sql]
    assert len(turn_sql) >= 3, turn_sql
    assert any("GROUP BY grp" in sql and "MIN(seq)" in sql for sql in turn_sql), turn_sql
    assert any("COUNT(*)" in sql for sql in turn_sql), turn_sql
    assert any("row_json" in sql and "ORDER BY seq" in sql for sql in turn_sql), turn_sql
    conn = real(index)
    try:
        for sql in turn_sql:
            plan = conn.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
            text = " | ".join(str(row["detail"]) for row in plan)
            assert "turns_run_grp" in text, (sql, text)
            assert "SCAN turns" not in text, (sql, text)
    finally:
        conn.close()


def test_rows_projected_before_grp_existed_are_regrouped_on_the_next_touch(tmp_path, monkeypatch):
    """@brief 升级修复：``turns`` 已有行但 ``grp`` 全是 0（旧代码没写过组号）时，下一次追平整份重投影。

    这是 schema 12 落地后才加投影规则留下的唯一不可恢复状态；``turn_bytes`` 已到文件末尾，
    所以修复必须先归零偏移再重读，否则会一直跳过重建。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _prepare(tmp_path, monkeypatch)
    directory = runs / "conv"
    _running(directory)
    (directory / "transcript.jsonl").write_text("".join([
        _line("chat", "A", reply={"content": "one"}),
        _line("retry", "A", error="timeout talking"),
        _line("chat", "B", reply={"content": "two"}),
    ]))
    index.tick()
    expected = _reference_groups(index, "conv")
    assert expected == [{"label": "A", "min_seq": 1, "max_seq": 2, "count": 2},
                        {"label": "B", "min_seq": 3, "max_seq": 3, "count": 1}]
    # 模拟升级前的库：行都在、偏移也到末尾，但没有组号。
    stale = turns_feed("conv")
    index.conn.execute("UPDATE turns SET grp = 0 WHERE run_id = 'conv'")
    index.conn.commit()
    index.stats["turn_parses"] = 0
    index.tick()
    assert turns_feed("conv", stamp=stale["stamp"]) != {"same": True, "stamp": stale["stamp"]}, (
        "重投影没有改 turn_bytes，必须靠 epoch 让旧 stamp 失效"
    )
    assert index.stats["turn_parses"] == 3, "整份重投影一次"
    assert index.conn.execute("SELECT MIN(grp) FROM turns WHERE run_id = 'conv'").fetchone()[0] >= 1
    assert _reference_groups(index, "conv") == expected
    # 第二次不再重投影。
    index.stats["turn_parses"] = 0
    index.tick()
    assert index.stats["turn_parses"] == 0


def test_a_turn_page_is_bounded_at_the_sqlite_layer(tmp_path, monkeypatch):
    """@brief PR8：在 SQLite 层证明翻页只读本页的组/事件，并 EXPLAIN 实际执行的组查询。

    只看 ``_parse_turn_rows`` 的计数位于 fetch 之后，证明不了没有先 ``fetchall()`` 再切片。
    这里用 trace/progress handler 抓真实 SQL 与 VDBE 步数：5000 组的 run 与 60 组的 run
    读一页的步数相当，且每条 ``FROM turns`` 的语句都带 run_id 与 LIMIT/BETWEEN/IN 边界。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from observe.index import Index

    runs, index = _prepare(tmp_path, monkeypatch)
    _many_groups(runs / "small", 60)
    _many_groups(runs / "huge", 5000)
    index.tick()
    turns_feed("huge", limit=40)  # 预热投影，后面的计数只量读页
    real = Index.read_only
    statements: list[str] = []
    ops = {"n": 0}

    def instrumented(self):
        conn = real(self)
        conn.set_trace_callback(statements.append)

        def handler() -> int:
            ops["n"] += 1
            return 0

        conn.set_progress_handler(handler, 1)
        return conn

    monkeypatch.setattr(Index, "read_only", instrumented)

    statements.clear()
    ops["n"] = 0
    turns_feed("small", limit=40)
    small_ops = ops["n"]
    statements.clear()
    ops["n"] = 0
    huge = turns_feed("huge", limit=40)
    huge_ops = ops["n"]
    assert len(huge["rows"]) == 40
    assert huge_ops < small_ops * 3 + 5000, (small_ops, huge_ops)

    turn_sql = [sql for sql in statements if "FROM turns" in sql]
    assert turn_sql, statements
    for sql in turn_sql:
        assert "run_id" in sql, sql
        assert "LIMIT" in sql or "BETWEEN" in sql or "IN (" in sql, sql
    monkeypatch.setattr(Index, "read_only", real)
