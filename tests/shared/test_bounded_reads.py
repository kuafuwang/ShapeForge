"""Bounded tail reads, a head-only kernel excerpt, and one streaming pass for全流聚合.

The three transcript aggregators (``modelcost.from_path``, ``_model_notes``, ``_chat_events``)
may each walk the whole file once; none of them may build a full event list, and the
tail-shaped views may not read a history file end to end.
"""

from __future__ import annotations

import io
import json
import os
import tracemalloc
from contextlib import contextmanager
from pathlib import Path

import pytest

from observe import detail
from taic import modelcost
from taic.reads import TAIL_BYTES, parse_jsonl, tail_lines, tail_rows, tail_text
from tests.support import run_logs


class _Metered:
    """@brief 包住文件对象，统计 ``read``/``readline`` 实际读入的字节数。"""

    def __init__(self, handle, read_bytes: list[int]) -> None:
        self._handle = handle
        self._read_bytes = read_bytes

    def read(self, *args):
        data = self._handle.read(*args)
        self._read_bytes[0] += len(data) if isinstance(data, bytes) else len(data.encode("utf-8"))
        return data

    def readline(self, *args):
        data = self._handle.readline(*args)
        self._read_bytes[0] += len(data) if isinstance(data, bytes) else len(data.encode("utf-8"))
        return data

    def __iter__(self):
        for line in self._handle:
            # 文本模式的 read 返回 str；按 UTF-8 编码后的字节数计，非 ASCII 才不会被低估。
            self._read_bytes[0] += len(line) if isinstance(line, bytes) else len(line.encode("utf-8"))
            yield line

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._handle.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._handle, name)


@contextmanager
def _meter(monkeypatch, path: Path):
    """@brief 只统计打开指定文件时读入的字节数；其它文件原样放行。

    @param monkeypatch pytest 的替换工具。
    @param path 要计量的文件。
    @return 上下文管理器，产出一个单元素列表（累计字节数）。
    """
    read_bytes = [0]
    real = io.open
    target = str(path)

    def counting(file, mode="r", *args, **kwargs):
        handle = real(file, mode, *args, **kwargs)
        if str(file) == target and "r" in mode:
            return _Metered(handle, read_bytes)
        return handle

    monkeypatch.setattr(io, "open", counting)
    try:
        yield read_bytes
    finally:
        monkeypatch.setattr(io, "open", real)


def _events_file(path: Path, *, count: int = 4000, extra: int = 0) -> None:
    """@brief 写一份 GPU 事件式的 JSONL，末尾带一条半行，模拟进程被杀。

    @param path 文件路径。
    @param count 完整事件条数。
    @param extra 额外填充的字节数（每条事件的 ``what`` 长度）。
    """
    filler = "x" * extra
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(json.dumps({"t": 1000.0 + index, "pid": 42, "kind": "wait" if index % 2 else "acquire",
                                     "mode": "ex", "what": f"benchmark of gemm fp16 paper {index} {filler}"}) + "\n")
        handle.write('{"t": 99999, "pid": 42, "kind": "wait"')


def test_the_event_tail_reads_a_bounded_slice_and_matches_a_full_scan(tmp_path, monkeypatch):
    """@brief 尾读 GPU 事件：结果与全量扫描的最后 40 条一致，读入字节远小于文件大小。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    monkeypatch.setenv("AI_LOWERLING_OBSERVE_DIR", str(tmp_path / "cache"))
    path = tmp_path / "cache" / "observe" / "gpu_events.jsonl"
    path.parent.mkdir(parents=True)
    _events_file(path, count=6000, extra=600)
    full = parse_jsonl(path.read_text(encoding="utf-8"))
    assert len(full) == 6000 and path.stat().st_size > TAIL_BYTES * 8

    # 旧语义：先取最后 40 个物理行，再解析；末尾那条半行占一个名额，所以只会得到 39 条事件。
    reference = parse_jsonl("\n".join(path.read_text(encoding="utf-8").splitlines()[-40:]))
    with _meter(monkeypatch, path) as read_bytes:
        shown = detail._tail_events()
    assert shown == reference
    assert len(shown) == len(full[-39:]) == 39
    assert read_bytes[0] <= TAIL_BYTES + 4
    assert read_bytes[0] < path.stat().st_size // 10, "尾读不该把整份历史读一遍"


T0 = 1_700_000_000.0


def _old_lock_spans(events, *, started: float, ended: float, pid: int = 42,
                    task: str = "conv2d", dtype: str = "fp16") -> list[dict]:
    """@brief 旧实现的全量重放（作为对照的 oracle）：排序事件、按 pid 配对、零长与 clamp 规则。

    @param events 全部 GPU 事件。
    @param started 运行开始时间。
    @param ended 运行结束时间。
    @param pid 运行进程号。
    @param task 任务名。
    @param dtype 数据类型。
    @return 区间列表。
    """
    lo, hi = started - 2.0, ended + 2.0
    pending: dict = {}
    spans: list[dict] = []

    def emit(wait: dict, stop: float) -> None:
        begin = float(wait["t"])
        if stop < begin:
            stop = begin
        if stop < lo or begin > hi:
            return
        if not detail._wait_belongs(wait, task, dtype, pid):
            return
        if stop > hi:
            stop = hi
        spans.append({"start": begin, "end": stop})

    for event in sorted(events, key=lambda item: float(item["t"])):
        kind = event.get("kind")
        epid = event.get("pid")
        if kind == "wait":
            previous = pending.get(epid)
            if previous is not None:
                emit(previous, float(previous["t"]))
            pending[epid] = event
            continue
        if kind in ("acquire", "give_up") and epid in pending:
            emit(pending.pop(epid), float(event["t"]))
    for wait in pending.values():
        emit(wait, float(wait["t"]))
    return spans


def _gpu_events(path: Path, events: list[dict], *, mode: str = "w") -> None:
    """@brief 把事件列表写成 GPU 事件日志。

    @param path 日志路径。
    @param events 事件列表。
    @param mode 写入模式（``w`` 或 ``a``）。
    """
    with path.open(mode, encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def _timeline_run(tmp_path: Path, monkeypatch, *, started: float = T0, ended: float = T0 + 100):
    """@brief 造一份能算 ``[start-2, end+2]`` 窗口的已结束运行，返回 ``(state, result, status, 日志路径)``。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @param started 运行开始时间。
    @param ended 运行结束时间。
    @return ``(state, result, status, gpu_events 路径)``。
    """
    monkeypatch.setenv("AI_LOWERLING_OBSERVE_DIR", str(tmp_path / "cache"))
    path = tmp_path / "cache" / "observe" / "gpu_events.jsonl"
    path.parent.mkdir(parents=True)
    state = {"pid": 42, "task": "conv2d", "dtype": "fp16", "started_at": started, "updated_at": ended}
    result = {"task": "conv2d", "dtype": "fp16", "started_at": started, "updated_at": ended}
    return state, result, "done", path


def _gpu(kind: str, when: float, pid: int = 42, what: str = "benchmark of conv2d fp16") -> dict:
    """@brief 造一条 GPU 锁事件。

    @param kind ``wait`` / ``acquire`` / ``give_up``。
    @param when 事件时间。
    @param pid 进程号。
    @param what 事件描述。
    @return 事件字典。
    """
    return {"t": when, "pid": pid, "kind": kind, "mode": "ex", "what": what}


def test_lock_spans_match_the_old_replay_for_the_six_pairing_rules(tmp_path, monkeypatch):
    """@brief §5.5.2：六种 wait/acquire 配对（窗口外、未配对、连续 wait、窗口内、clamp、pid 复用）逐个与旧重放一致。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    state, result, status, path = _timeline_run(tmp_path, monkeypatch)
    import observe.index as index_mod

    index = index_mod.get_index()
    cases = {
        # wait<lo 且 acquire>hi → 画到 hi（不能当未结束漏掉）
        "wait_before_window_acquire_after": [
            _gpu("wait", T0 - 100), _gpu("acquire", T0 + 200),
        ],
        # wait<lo 永不 acquire → 不出现（零长在窗口前被过滤）
        "wait_before_window_never_acquired": [
            _gpu("wait", T0 - 100),
        ],
        # wait1<wait2<lo：只有 wait2 有效，wait1 已被零长关闭并过滤
        "two_waits_before_window": [
            _gpu("wait", T0 - 100), _gpu("wait", T0 - 50), _gpu("acquire", T0),
        ],
        # 窗口内 wait+acquire → 正常 span
        "inside_window": [
            _gpu("wait", T0 + 20), _gpu("acquire", T0 + 50),
        ],
        # 窗口内 wait、acquire>hi → 截到 hi
        "acquire_after_window": [
            _gpu("wait", T0 + 20), _gpu("acquire", T0 + 200),
        ],
        # 运行结束后同 pid 又被 acquire：第一段截到 hi，后一段整段在窗口外
        "pid_reused_after_run": [
            _gpu("wait", T0 + 20), _gpu("acquire", T0 + 300),
            _gpu("wait", T0 + 350), _gpu("acquire", T0 + 400),
        ],
        # 窗口内的 wait 被新的 wait 覆盖 → 旧 wait 记零长，仍在窗口内
        "overwritten_wait_inside_window": [
            _gpu("wait", T0 + 20), _gpu("wait", T0 + 30), _gpu("acquire", T0 + 40),
        ],
        # 别的算子的等待不算进来
        "other_task": [
            _gpu("wait", T0 + 20, pid=7, what="benchmark of relu fp16"),
            _gpu("acquire", T0 + 50, pid=7, what="benchmark of relu fp16"),
        ],
        # 两个 pid 都还没配对：旧实现最后按 pending 的进入顺序输出（pid1 先进入，虽然它后来被覆盖成 t=30）
        "two_pending_in_insertion_order": [
            _gpu("wait", T0 + 10, pid=1), _gpu("wait", T0 + 20, pid=2),
            _gpu("wait", T0 + 30, pid=1),
        ],
        # 不同 pid、各自没有历史、全局倒序追加：必须按全局 t 重排，不能只看同 pid 回退。
        "global_reverse_across_pids": [
            _gpu("wait", T0 + 20, pid=1), _gpu("wait", T0 + 10, pid=2),
        ],
    }
    for name, events in cases.items():
        path.unlink(missing_ok=True)
        _gpu_events(path, events)
        index.ensure_gpu_events()
        shown = detail._lock_spans(state, result, None, status)
        assert shown == _old_lock_spans(events, started=T0, ended=T0 + 100), name

    # 追加顺序乱掉：acquire(20), wait(10), acquire(30) —— 旧实现按 t 排序后是 [10,20]。
    late = [_gpu("acquire", T0 + 20), _gpu("wait", T0 + 10), _gpu("acquire", T0 + 30)]
    path.unlink(missing_ok=True)
    _gpu_events(path, late)
    index.ensure_gpu_events()
    assert detail._lock_spans(state, result, None, status) == [{"start": T0 + 10, "end": T0 + 20}]
    assert detail._lock_spans(state, result, None, status) == _old_lock_spans(late, started=T0, ended=T0 + 100)

    # 混合多个 pid，且按时间乱序写入：输出顺序要跟旧重放一样（按关闭时刻，最后才是 pending）。
    mixed = [event for events in cases.values() for event in events]
    mixed = list(reversed(sorted(mixed, key=lambda item: item["t"])))
    path.unlink(missing_ok=True)
    _gpu_events(path, mixed)
    index.ensure_gpu_events()
    assert detail._lock_spans(state, result, None, status) == _old_lock_spans(mixed, started=T0, ended=T0 + 100)

    # 多 pid 的关闭顺序：B 先关闭，A 后关闭；旧实现先出 B，再出 A。
    order_case = [
        _gpu("wait", T0 + 10, pid=1), _gpu("wait", T0 + 20, pid=2),
        _gpu("acquire", T0 + 30, pid=2), _gpu("acquire", T0 + 100, pid=1),
    ]
    path.unlink(missing_ok=True)
    _gpu_events(path, order_case)
    index.ensure_gpu_events()
    shown = detail._lock_spans(state, result, None, status)
    assert shown == [{"start": T0 + 20, "end": T0 + 30}, {"start": T0 + 10, "end": T0 + 100}]
    assert shown == _old_lock_spans(order_case, started=T0, ended=T0 + 100)


def test_the_gpu_history_is_read_incrementally_and_survives_rotation(tmp_path, monkeypatch):
    """@brief §5.5.2：只读新字节；轮转产生的分片能被重放，重建结果与轮转前逐行一致；历史被删时标截断。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    state, result, status, path = _timeline_run(tmp_path, monkeypatch)
    import observe.index as index_mod

    index = index_mod.get_index()
    first = [_gpu("wait", T0), _gpu("acquire", T0 + 10)]
    _gpu_events(path, first)
    index.ensure_gpu_events()
    before = [tuple(row) for row in index.conn.execute("SELECT t, pid, kind FROM gpu_events ORDER BY rowid")]
    spans_before = [tuple(row) for row in index.conn.execute("SELECT pid, start, end FROM lock_spans ORDER BY start, end")]
    assert len(before) == 2 and len(spans_before) == 1

    # 没有新字节时只做有界的已消费前缀校验（头尾各 4KB），不重读历史。
    with _meter(monkeypatch, path) as read_bytes:
        index.ensure_gpu_events()
    assert read_bytes[0] <= 8192

    # 轮转：旧文件改名保留成分片，新文件继续追加；inode 变化后按分片重放。
    shard = path.with_name(path.name + ".1")
    path.rename(shard)
    second = [_gpu("wait", T0 + 20), _gpu("acquire", T0 + 30)]
    _gpu_events(path, second)
    index.ensure_gpu_events()
    after = [tuple(row) for row in index.conn.execute("SELECT t, pid, kind FROM gpu_events ORDER BY t, kind")]
    expected = sorted(before + [(row["t"], 42, row["kind"]) for row in second], key=lambda item: (item[0], item[2]))
    assert after == expected
    assert detail.gpu_history_truncated() is False
    assert detail._lock_spans(state, result, None, status) == _old_lock_spans(
        first + second, started=T0, ended=T0 + 100
    )

    # 分片被删且当前文件被替换：历史不再完整，必须显式标截断。
    shard.unlink()
    path.unlink()
    _gpu_events(path, [_gpu("wait", T0 + 60)])
    index.ensure_gpu_events()
    assert detail.gpu_history_truncated() is True

    # 源日志被原地截短：清空重建，同样标截断。
    _gpu_events(path, [_gpu("wait", T0 + 70)])
    index.ensure_gpu_events()
    path.write_text("", encoding="utf-8")
    index.ensure_gpu_events()
    assert index.conn.execute("SELECT COUNT(*) FROM gpu_events").fetchone()[0] == 0
    assert detail.gpu_history_truncated() is True


def test_a_100k_event_log_is_not_reread_to_draw_a_timeline_window(tmp_path, monkeypatch):
    """@brief §5.5.2：10 万条事件入库后，时间线只查相交区间，不再读日志（读入 0 字节）。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    state, result, status, path = _timeline_run(tmp_path, monkeypatch, ended=T0 + 10)
    events = []
    for index in range(50_000):
        when = T0 + index * 0.02  # 共 1000 秒，运行窗口只有 12 秒
        events.append(_gpu("wait", when))
        events.append(_gpu("acquire", when + 0.001))
    _gpu_events(path, events)
    import observe.index as index_mod

    index = index_mod.get_index()
    index.ensure_gpu_events()
    assert index.conn.execute("SELECT COUNT(*) FROM gpu_events").fetchone()[0] == 100_000

    with _meter(monkeypatch, path) as read_bytes:
        shown = detail._lock_spans(state, result, None, status)
    # 时间线只查相交区间；唯一读盘是已消费前缀的 8KB 校验。
    assert read_bytes[0] <= 8192
    assert shown == _old_lock_spans(events, started=T0, ended=T0 + 10)
    assert len(shown) == 601  # 窗口 [T0-2, T0+12] 只覆盖前 601 对，其余 99,398 条事件不参与查询


def test_the_epic_feed_reads_only_the_tail(tmp_path, monkeypatch):
    """@brief epic 日志只读尾部；展示行与全量扫描的最后 80 个事件一致。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    path = tmp_path / "epic.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(20000):
            event = {"event": "usage", "cost": 0.001, "output_tokens": index, "seconds": 1}
            handle.write(json.dumps(event) + "\n")
        handle.write(json.dumps({"event": "ended", "how": "done"}) + "\n")
    assert path.stat().st_size > TAIL_BYTES
    with _meter(monkeypatch, path) as read_bytes:
        rows = detail._epic_lines_in(path)
    assert rows[-1] == {"who": "结束", "text": "完成"}
    assert len(rows) == 80
    assert read_bytes[0] <= TAIL_BYTES + 4
    assert read_bytes[0] < path.stat().st_size


def test_the_test_log_tail_keeps_the_last_characters_without_reading_the_whole_file(tmp_path, monkeypatch):
    """@brief 失败日志的 ``log_tail`` 仍返回最后 8,000 字符，但只读尾部。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    path = tmp_path / "test.log"
    text = "".join(f"line {index:06d} 有中文也有 ASCII\n" for index in range(20000))
    path.write_text(text, encoding="utf-8")
    assert len(text) > 400_000

    with _meter(monkeypatch, path) as read_bytes:
        got = run_logs.log_tail(path, limit=8000)
    assert got == text[-8000:]
    assert read_bytes[0] < path.stat().st_size // 5


def test_the_kernel_excerpt_reads_only_the_head_and_keeps_multibyte_text(tmp_path, monkeypatch):
    """@brief 内核片段只读开头 2,048 字节；中文与 1MB 的后半段都不影响前 400 字。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    path = tmp_path / "kernel.metal"
    head = "#include <metal_stdlib>\n// 中文注释：合并写入\n" * 20
    text = head + "x" * 1_000_000
    path.write_text(text, encoding="utf-8")

    with _meter(monkeypatch, path) as read_bytes:
        got = detail._excerpt(path)
    assert got == text[:400]
    assert read_bytes[0] <= 2048 + 4
    assert "中文注释：合并写入" in got


def _transcript(path: Path, *, sessions: int = 3, filler: int = 4096, tail_rows_extra: int = 0) -> list[dict]:
    """@brief 写一份含 chat/agent/retry/warning/坏行的 transcript，返回完整解析结果。

    @param path 文件路径。
    @param sessions agent 行数（每行带一份填充正文，模拟真实体积）。
    @param filler 每条 agent 行 ``reply`` 的填充长度。
    @param tail_rows_extra 额外追加的 chat 行数。
    @return 完整解析后的事件列表。
    """
    rows = []
    for index in range(sessions):
        rows.append({
            "event": "agent", "label": f"r{index}.p0", "candidate_label": f"r{index}.p0",
            "provider": "deepseek", "model": "deepseek-flash", "committee": "first",
            "outcome": "evaluated", "cost_usd": 0.25, "seconds": 2.0,
            "tokens": {"input": 10, "output": 20, "cache_read": 30, "cache_write": 0},
            "reply": {"content": "y" * filler},
        })
    rows.append({"event": "chat", "label": "init#t0", "model": "deepseek-flash", "seconds": 1.5,
                 "t": 11.0, "messages": [{"role": "user", "content": "prompt"}],
                 "reply": {"content": "answer"}})
    rows.append({"event": "retry", "label": "r0.p0", "model": "deepseek-flash", "error": "read timeout"})
    rows.append({"event": "error", "label": "r0.p0", "model": "deepseek-flash", "status": 402, "error": "payment required"})
    rows.append({"event": "warning", "label": "r1.p0", "model": "deepseek-flash", "message": "reply truncated by max_tokens"})
    rows.append({"event": "chat", "label": "r1.plan", "model": "deepseek-flash", "finish_reason": "tool_calls",
                 "reply": {"content": "", "tool_calls": [{"function": {"name": "read_file"}}]}})
    for index in range(tail_rows_extra):
        rows.append({"event": "chat", "label": f"r2.t{index}", "model": "m", "seconds": 0.5, "t": 20.0 + index})
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.write('{"event": "chat", "label": "half')
    return rows


def test_the_three_aggregators_each_stream_the_transcript_once(tmp_path, monkeypatch):
    """@brief ``modelcost.from_path``/``_model_notes``/``_chat_events`` 各读一遍，输出仍逐字段一致。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    path = tmp_path / "transcript.jsonl"
    rows = _transcript(path, sessions=4, tail_rows_extra=3)
    size = path.stat().st_size
    assert size > 16_000

    with _meter(monkeypatch, path) as read_bytes:
        totals = modelcost.from_path(path)
    assert read_bytes[0] <= size + 4
    pure = modelcost.from_events(rows)
    assert totals == pure
    assert totals[0]["provider"] == "deepseek" and totals[0]["sessions"] == 4
    assert totals[0]["cost_usd"] == pytest.approx(1.0)

    with _meter(monkeypatch, path) as read_bytes:
        notes = detail._model_notes(tmp_path)
    assert read_bytes[0] <= size + 4
    kinds = [(note["label"], note["kind"]) for note in notes]
    assert kinds[:4] == [("r0.p0", "agent_empty"), ("r1.p0", "agent_empty"),
                         ("r2.p0", "agent_empty"), ("r3.p0", "agent_empty")]
    assert kinds[4:] == [
        ("init#t0", "reply"), ("r0.p0", "timeout"), ("r0.p0", "http"),
        ("r1.p0", "truncated"), ("r1.plan", "tool"),
        ("r2.t0", "empty"), ("r2.t1", "empty"), ("r2.t2", "empty"),
    ]

    with _meter(monkeypatch, path) as read_bytes:
        chat = detail._chat_events(tmp_path)
    assert read_bytes[0] <= size + 4
    assert [row["label"] for row in chat][:2] == ["init#t0", "r1.plan"]
    assert chat[0]["seconds"] == pytest.approx(1.5) and chat[0]["t"] == pytest.approx(11.0)


def _brief_run(root: Path) -> Path:
    """@brief 造一个 ``taic brief`` 能读的瘦 run 目录。

    @param root 临时根目录。
    @return run 目录。
    """
    run = root / "run"
    run.mkdir()
    (run / "state.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "phase": "done", "started_at": T0, "updated_at": T0 + 10,
    }), encoding="utf-8")
    (run / "result.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.1, "best_speedup": 1.1,
    }), encoding="utf-8")
    return run


def test_brief_summarize_run_reads_the_transcript_once(tmp_path, monkeypatch):
    """@brief §5.5.1：``summarize_run`` 的完整调用只遍历 transcript 一次，输出与流式聚合一致。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from taic import brief

    run = _brief_run(tmp_path)
    path = run / "transcript.jsonl"
    rows = _transcript(path, sessions=5, filler=3000, tail_rows_extra=2)
    size = path.stat().st_size
    with _meter(monkeypatch, path) as read_bytes:
        # §6.1：``model_totals`` 是重面板，默认 L1 不含它，要靠 ``expand`` 明确索取。
        doc = brief.summarize_run(run, expand={"models"})
    assert read_bytes[0] <= size + 4, f"brief 读了 {read_bytes[0]}B，文件只有 {size}B"
    assert doc["counts"]["chat"] == 4
    assert doc["counts"]["retry"] == 1
    assert doc["counts"]["truncation_retries"] == 1
    assert doc["model_totals"] == modelcost.from_events(rows)
    assert doc["model_totals"][0]["model"] == "deepseek-flash"


def test_brief_and_detail_consumers_stay_bounded_on_a_92mb_transcript(tmp_path, monkeypatch):
    """@brief §5.5.1：92.8MB transcript 上 ``summarize_run`` 一次 pass 且峰值 ≤8MB；详情聚合同样有界。

    顶层调用必须不超过文件字节一次；只分别测单个 consumer 不能证明这一点。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from taic import brief

    run = _brief_run(tmp_path)
    path = run / "transcript.jsonl"
    filler = "y" * 40_000
    target = 92_815_056
    written = 0
    index = 0
    chats = 0
    with path.open("w", encoding="utf-8") as handle:
        while written < target:
            if index % 200 == 0:
                row = {"event": "chat", "label": f"c{index}", "model": "m", "seconds": 1.0,
                       "t": T0 + index, "reply": {"content": filler}}
                chats += 1
            else:
                row = {
                    "event": "agent", "label": f"r{index}.p0", "candidate_label": f"r{index}.p0",
                    "provider": "deepseek", "model": "deepseek-flash", "committee": "first",
                    "outcome": "evaluated", "cost_usd": 0.001, "seconds": 1.0, "t": T0 + index,
                    "tokens": {"input": 10, "output": 20, "cache_read": 30, "cache_write": 0},
                    "reply": {"content": filler},
                }
            line = json.dumps(row, ensure_ascii=False) + "\n"
            handle.write(line)
            written += len(line.encode("utf-8"))
            index += 1
    size = path.stat().st_size
    assert size >= 92_000_000

    tracemalloc.start()
    try:
        with _meter(monkeypatch, path) as read_bytes:
            doc = brief.summarize_run(run, expand={"models", "failures"})
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert read_bytes[0] <= size + 4, f"summarize_run 读了 {read_bytes[0]}B，文件 {size}B"
    assert peak <= 8 * 1024 * 1024, f"summarize_run 峰值 {peak / 1e6:.1f}MB 超过 8MB"
    assert doc["counts"]["chat"] == chats

    for name, consumer in (
        ("modelcost.from_path", lambda: modelcost.from_path(path)),
        ("detail._model_notes", lambda: detail._model_notes(run)),
        ("detail._chat_events", lambda: detail._chat_events(run)),
    ):
        tracemalloc.start()
        try:
            consumer()
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak <= 8 * 1024 * 1024, f"{name} 峰值 {peak / 1e6:.1f}MB 超过 8MB"


def test_tail_rows_keeps_the_old_physical_line_semantics(tmp_path):
    """@brief 尾读解析器与旧的 ``splitlines()[-keep:]`` 同义：半行/坏行占名额，长行不被截断。

    @param tmp_path pytest 提供的临时目录。
    """
    path = tmp_path / "events.jsonl"
    _events_file(path, count=10)
    # 半行占一个名额：最后 3 行 = 事件 8、事件 9、半行 → 只有 2 条事件。
    rows = tail_rows(path, keep=3)
    assert [row["t"] for row in rows] == [1008.0, 1009.0]

    # 坏行也占名额。
    path.write_text(
        json.dumps({"t": 1.0}) + "\n" + "not json\n" + json.dumps({"t": 3.0}) + "\n",
        encoding="utf-8",
    )
    assert [row["t"] for row in tail_rows(path, keep=3)] == [1.0, 3.0]

    # 单行比回读块还长：必须读完整行，不能截断，也不能拆成多行。
    long_line = json.dumps({"t": 9.0, "what": "x" * 5000})
    path.write_text(json.dumps({"t": 1.0}) + "\n" + long_line + "\n", encoding="utf-8")
    rows = tail_rows(path, keep=1, chunk=64)
    assert len(rows) == 1 and len(rows[0]["what"]) == 5000

    # 精确边界：文件以换行结尾时，末尾空段不算一行。
    path.write_text("".join(json.dumps({"t": float(i)}) + "\n" for i in range(4)), encoding="utf-8")
    assert [row["t"] for row in tail_rows(path, keep=2, chunk=5)] == [2.0, 3.0]
    assert tail_rows(tmp_path / "missing.jsonl", keep=3) == []


def test_tail_text_never_reads_more_than_the_limit(tmp_path, monkeypatch):
    """@brief ``tail_text`` 的 ``read(limit)`` 是硬上限：stat 之后继续追加也读不出超限字节。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    path = tmp_path / "events.jsonl"
    path.write_text("a" * 4096 + "\n", encoding="utf-8")
    calls = []

    class _Metered:
        def __init__(self, handle):
            self._handle = handle

        def read(self, *args):
            calls.append(args)
            return self._handle.read(*args)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

        def __getattr__(self, name):
            return getattr(self._handle, name)

    real = io.open
    target = str(path)

    def counting(file, mode="r", *args, **kwargs):
        handle = real(file, mode, *args, **kwargs)
        return _Metered(handle) if str(file) == target and "r" in mode else handle

    monkeypatch.setattr(io, "open", counting)
    got = tail_text(path, limit=100, trim_partial=False)
    monkeypatch.setattr(io, "open", real)
    assert calls and calls[0] == (100,)
    assert len(got.encode("utf-8")) == 100


def test_a_100k_off_window_pending_waits_never_leave_sqlite(tmp_path, monkeypatch):
    """@brief §5.5.2：10 万个窗口外的未配对 wait 留在表里，但时间线一行都不取。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    state, result, status, path = _timeline_run(tmp_path, monkeypatch)
    events = [_gpu("wait", T0 - 5000.0 - index, pid=index + 1) for index in range(100_000)]
    _gpu_events(path, events)
    import observe.index as index_mod

    index = index_mod.get_index()
    index.ensure_gpu_events()
    assert index.conn.execute("SELECT COUNT(*) FROM gpu_waits").fetchone()[0] == 100_000
    closed, waits = detail._lock_rows(index.conn, T0 - 2.0, T0 + 102.0)
    assert closed == [] and waits == []
    assert detail._lock_spans(state, result, None, status) == []


def test_truncated_history_is_sticky_and_a_lost_shard_changes_the_panel_stamp(tmp_path, monkeypatch):
    """@brief §5.5.2：分片被删要标截断且不自行清回 0；分片单独变化也要让时间线戳变化。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from observe import query

    state, result, _status, path = _timeline_run(tmp_path, monkeypatch)
    directory = tmp_path / "run"
    directory.mkdir()
    for name in ("state.json", "result.json", "eval_progress.json"):
        (directory / name).write_text("{}", encoding="utf-8")
    import observe.index as index_mod

    index = index_mod.get_index()
    _gpu_events(path, [_gpu("wait", T0), _gpu("acquire", T0 + 10)])
    index.ensure_gpu_events()
    before = query._timeline_panel_stamp(directory)
    assert detail.gpu_history_truncated() is False

    shard = path.with_name(path.name + ".1")
    path.rename(shard)
    _gpu_events(path, [_gpu("wait", T0 + 20), _gpu("acquire", T0 + 30)])
    index.ensure_gpu_events()
    assert detail.gpu_history_truncated() is False
    with_shard = query._timeline_panel_stamp(directory)
    assert with_shard != before

    # 活动文件完全没变，只是分片被删：必须发现并让面板戳变化。
    shard.unlink()
    index.ensure_gpu_events()
    assert detail.gpu_history_truncated() is True
    assert query._timeline_panel_stamp(directory) != with_shard

    # 之后再加一个更大的分片也不能把“历史已丢”清回 0。
    path.with_name(path.name + ".2").write_text("x" * 10_000, encoding="utf-8")
    index.ensure_gpu_events()
    assert detail.gpu_history_truncated() is True


def test_the_real_writer_rotates_and_the_index_replays_every_shard(tmp_path, monkeypatch):
    """@brief §5.5.2：真实写者按阈值轮转（改名保留分片），索引重放分片+活动文件后与旧重放一致。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    monkeypatch.setenv("AI_LOWERLING_OBSERVE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TCENV_GPU_EVENTS_MAX_BYTES", "200")
    from tcenv import events

    # 超过上一版默认保留数（8）的轮转次数：写者不得删除任何分片。
    for index in range(30):
        events._record_gpu("wait", "ex", f"benchmark of conv2d fp16 {index}")
        events._record_gpu("acquire", "ex", f"benchmark of conv2d fp16 {index}")

    path = tmp_path / "cache" / "observe" / "gpu_events.jsonl"
    shards = sorted(
        (item for item in path.parent.glob(path.name + ".*") if item.is_file()),
        key=lambda item: (item.stat().st_mtime_ns, item.name),
    )
    assert len(shards) > 8, f"写者轮转了 {len(shards)} 次，分片不应被自动删除"

    def rows(item: Path) -> list[dict]:
        return [json.loads(line) for line in item.read_text(encoding="utf-8").splitlines() if line.strip()]

    ordered: list[dict] = []
    for shard in shards:
        ordered.extend(rows(shard))
    ordered.extend(rows(path))
    started = float(ordered[0]["t"]) - 1.0
    ended = float(ordered[-1]["t"]) + 1.0
    state = {"pid": os.getpid(), "task": "conv2d", "dtype": "fp16",
             "started_at": started, "updated_at": ended}
    result = {"task": "conv2d", "dtype": "fp16", "started_at": started, "updated_at": ended}

    import observe.index as index_mod

    index = index_mod.get_index()
    index.ensure_gpu_events()
    assert detail.gpu_history_truncated() is False
    assert detail._lock_spans(state, result, None, "done") == _old_lock_spans(
        ordered, started=started, ended=ended, pid=os.getpid()
    )

    # PR6 重建验收：删掉派生库后，从保留的分片 + 活动文件重建出逐行一样的事件与区间。
    events_before = [
        tuple(row) for row in index.conn.execute(
            "SELECT t, pid, kind, mode, what FROM gpu_events ORDER BY rowid"
        )
    ]
    spans_before = sorted(
        tuple(row) for row in index.conn.execute("SELECT pid, start, end, closed_at FROM lock_spans")
    )
    db = index.path
    index.conn.close()
    for suffix in ("", "-wal", "-shm"):
        path_to_db = db.with_name(db.name + suffix) if suffix else db
        path_to_db.unlink(missing_ok=True)
    index_mod._SINGLE = None

    rebuilt = index_mod.get_index()
    rebuilt.ensure_gpu_events()
    events_after = [
        tuple(row) for row in rebuilt.conn.execute(
            "SELECT t, pid, kind, mode, what FROM gpu_events ORDER BY rowid"
        )
    ]
    spans_after = sorted(
        tuple(row) for row in rebuilt.conn.execute("SELECT pid, start, end, closed_at FROM lock_spans")
    )
    assert events_after == events_before
    assert spans_after == spans_before
    assert detail.gpu_history_truncated() is False


def test_events_with_non_finite_or_string_timestamps_are_dropped(tmp_path, monkeypatch):
    """@brief §5.5.2：``t`` 不是有限实数（字符串 / NaN / Inf / null）的事件整条丢掉，与旧读者一致。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    state, result, status, path = _timeline_run(tmp_path, monkeypatch)
    good = [_gpu("wait", T0), _gpu("acquire", T0 + 10)]
    lines = [
        '{"t": "10", "pid": 42, "kind": "wait", "mode": "ex", "what": "benchmark of conv2d fp16"}',
        '{"t": NaN, "pid": 42, "kind": "wait", "mode": "ex", "what": "benchmark of conv2d fp16"}',
        '{"t": Infinity, "pid": 42, "kind": "wait", "mode": "ex", "what": "benchmark of conv2d fp16"}',
        '{"t": null, "pid": 42, "kind": "wait", "mode": "ex", "what": "benchmark of conv2d fp16"}',
        json.dumps(good[0]),
        json.dumps(good[1]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    import observe.index as index_mod

    index = index_mod.get_index()
    index.ensure_gpu_events()
    assert index.conn.execute("SELECT COUNT(*) FROM gpu_events").fetchone()[0] == 2
    assert detail._lock_spans(state, result, None, status) == _old_lock_spans(good, started=T0, ended=T0 + 100)


def test_tail_rows_handles_cr_and_crlf_separators(tmp_path):
    """@brief 尾读与 ``splitlines()`` 同义：只用 ``\\r`` 分隔的日志不能整份读回。

    @param tmp_path pytest 提供的临时目录。
    """
    rows = [json.dumps({"t": float(index)}) for index in range(60)]
    path = tmp_path / "cr.jsonl"
    path.write_bytes("\r".join(rows).encode("utf-8"))
    assert [row["t"] for row in tail_rows(path, keep=3)] == [57.0, 58.0, 59.0]
    path.write_bytes("\r\n".join(rows).encode("utf-8"))
    assert [row["t"] for row in tail_rows(path, keep=2)] == [58.0, 59.0]


def test_tail_lines_has_a_hard_byte_budget_for_a_huge_last_line(tmp_path, monkeypatch):
    """@brief 单个超长物理行不能把尾读退化成整读：预算用完就丢半行并返回空。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    path = tmp_path / "huge.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(3):
            handle.write(json.dumps({"t": float(index)}) + "\n")
        handle.write(json.dumps({"t": 9.0, "what": "x" * 1_000_000}) + "\n")
    assert path.stat().st_size > 1_000_000

    with _meter(monkeypatch, path) as read_bytes:
        rows = tail_rows(path, keep=2, chunk=4096, max_bytes=256)
    assert rows == [], "预算内装不下任何完整行时返回空，而不是整读文件"
    assert read_bytes[0] <= 256 + 4


def test_the_event_tail_cannot_slurp_a_ten_megabyte_single_line(tmp_path, monkeypatch):
    """@brief 10MB 单行日志上 ``_tail_events`` 的读入不超过默认预算（256KB）。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    monkeypatch.setenv("AI_LOWERLING_OBSERVE_DIR", str(tmp_path / "cache"))
    path = tmp_path / "cache" / "observe" / "gpu_events.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("x" * 10_000_000 + "\n", encoding="utf-8")

    with _meter(monkeypatch, path) as read_bytes:
        rows = detail._tail_events()
    assert rows == []
    assert read_bytes[0] <= TAIL_BYTES + 4
    assert read_bytes[0] < path.stat().st_size // 10


def test_deleting_an_older_shard_during_a_rotation_marks_history_truncated(tmp_path, monkeypatch):
    """@brief §5.5.2：删掉较老分片与 inode 变化同时发生时，必须逐分片核对并标截断。

    只看“剩余分片总字节数”会被同一次轮转出来的更大新分片掩盖，历史静默丢失但 truncated 仍为 0。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    state, result, status, path = _timeline_run(tmp_path, monkeypatch)
    import observe.index as index_mod

    index = index_mod.get_index()

    def rotate(suffix: str) -> None:
        path.rename(path.with_name(path.name + suffix))

    batches = [
        [_gpu("wait", T0 + 1), _gpu("acquire", T0 + 2)],
        [_gpu("wait", T0 + 3), _gpu("acquire", T0 + 4)],
        [_gpu("wait", T0 + 5), _gpu("acquire", T0 + 6)],
        [_gpu("wait", T0 + 7), _gpu("acquire", T0 + 8)],
    ]
    _gpu_events(path, batches[0])
    index.ensure_gpu_events()
    rotate(".1")
    _gpu_events(path, batches[1])
    index.ensure_gpu_events()
    rotate(".2")
    _gpu_events(path, batches[2])
    index.ensure_gpu_events()

    # 删除最老分片，同时把当前文件轮转成新分片：两种变化在同一帧发生。
    (path.parent / (path.name + ".1")).unlink()
    rotate(".3")
    _gpu_events(path, batches[3])
    index.ensure_gpu_events()

    assert detail.gpu_history_truncated() is True
    kept = [row["t"] for row in index.conn.execute("SELECT t FROM gpu_events ORDER BY rowid")]
    assert kept == [T0 + 3, T0 + 4, T0 + 5, T0 + 6, T0 + 7, T0 + 8]
    retained = [event for batch in batches[1:] for event in batch]
    assert detail._lock_spans(state, result, None, status) == _old_lock_spans(
        retained, started=T0, ended=T0 + 100
    )
    # 同 inode 的后续帧不能把截断标记清回 0。
    index.ensure_gpu_events()
    assert detail.gpu_history_truncated() is True


def test_a_missing_active_file_replays_shards_without_marking_truncated(tmp_path, monkeypatch):
    """@brief 写者已把活动文件改名成新分片、新文件还没建：历史仍完整，不能标截断，也不能丢事件。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _state, _result, _status, path = _timeline_run(tmp_path, monkeypatch)
    import observe.index as index_mod

    index = index_mod.get_index()
    events = [_gpu("wait", T0 + 1), _gpu("acquire", T0 + 2)]
    _gpu_events(path, events)
    index.ensure_gpu_events()
    path.rename(path.with_name(path.name + ".1"))

    index.ensure_gpu_events()
    assert detail.gpu_history_truncated() is False
    assert index.conn.execute("SELECT COUNT(*) FROM gpu_events").fetchone()[0] == 2

    # 冷库只剩分片：删库重建也必须重放分片，而不是返回空并保持“未截断”。
    db = index.path
    index.conn.close()
    for suffix in ("", "-wal", "-shm"):
        (db.with_name(db.name + suffix) if suffix else db).unlink(missing_ok=True)
    index_mod._SINGLE = None
    fresh = index_mod.get_index()
    fresh.ensure_gpu_events()
    assert fresh.conn.execute("SELECT COUNT(*) FROM gpu_events").fetchone()[0] == 2
    assert detail.gpu_history_truncated() is False


def test_a_same_inode_rewrite_is_detected_by_the_prefix_checkpoint(tmp_path, monkeypatch):
    """@brief 同 inode 的等长改写、以及缩短后再长回来，必须按前缀校验发现并重建 + 标截断。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _state, _result, _status, path = _timeline_run(tmp_path, monkeypatch)
    import observe.index as index_mod

    index = index_mod.get_index()
    original = [{"t": T0 + 1, "pid": 1, "kind": "wait", "mode": "ex", "what": "benchmark of conv2d fp16"}]
    _gpu_events(path, original)
    index.ensure_gpu_events()

    # 等长改写：pid 1 → 2，字节数相同，只看 size 发现不了。
    rewritten = [{"t": T0 + 1, "pid": 2, "kind": "wait", "mode": "ex", "what": "benchmark of conv2d fp16"}]
    assert len(json.dumps(rewritten[0])) == len(json.dumps(original[0]))
    _gpu_events(path, rewritten)
    index.ensure_gpu_events()
    assert detail.gpu_history_truncated() is True
    rows = [row["pid"] for row in index.conn.execute("SELECT pid FROM gpu_events ORDER BY rowid")]
    assert rows == [2], "同长度改写必须重读，而不是沿用旧内容"

    # 缩短后再长回来：先写成更短的不同内容，再写回更长的内容。
    shorter = [{"t": T0 + 5, "pid": 3, "kind": "wait", "mode": "ex", "what": "x"}]
    _gpu_events(path, shorter)
    index.ensure_gpu_events()
    longer = [{"t": T0 + 8, "pid": 4, "kind": "wait", "mode": "ex", "what": "benchmark of conv2d fp16"}]
    _gpu_events(path, longer)
    index.ensure_gpu_events()
    rows = [row["pid"] for row in index.conn.execute("SELECT pid FROM gpu_events ORDER BY rowid")]
    assert rows == [4], "缩短后再长回来不能从旧 offset 接着读"
    assert detail.gpu_history_truncated() is True


def test_a_shrunk_active_file_keeps_the_shard_history(tmp_path, monkeypatch):
    """@brief 同 inode 变短时走分片感知重建：已有分片历史不能被活动文件覆盖掉。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    state, result, status, path = _timeline_run(tmp_path, monkeypatch)
    import observe.index as index_mod

    index = index_mod.get_index()
    first = [_gpu("wait", T0 + 1)]
    _gpu_events(path, first)
    index.ensure_gpu_events()
    path.rename(path.with_name(path.name + ".1"))
    _gpu_events(path, [{"t": T0 + 2, "pid": 42, "kind": "acquire", "mode": "ex", "what": "benchmark of conv2d fp16"}])
    index.ensure_gpu_events()
    assert detail.gpu_history_truncated() is False

    # 活动文件原地变短（同 inode）：分片里的 wait(t=T0+1) 仍必须保留。
    _gpu_events(path, [{"t": T0 + 3, "pid": 42, "kind": "acquire", "mode": "ex", "what": "x"}])
    index.ensure_gpu_events()
    kept = [row["t"] for row in index.conn.execute("SELECT t FROM gpu_events ORDER BY rowid")]
    assert kept == [T0 + 1, T0 + 3]
    assert detail.gpu_history_truncated() is True
    assert len(detail._lock_spans(state, result, None, status)) == 1


def test_brief_memory_stays_bounded_with_many_agent_failures(tmp_path):
    """@brief §5.5.1：1 万条 ``agent_failed`` 行也不能让 brief 的表/来源随全部事件增长。

    @param tmp_path pytest 提供的临时目录。
    """
    from taic import brief

    run = _brief_run(tmp_path)
    path = run / "transcript.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(10_000):
            handle.write(json.dumps({
                "event": "agent", "label": f"r{index}.p0", "candidate_label": f"r{index}.p0",
                "provider": "deepseek", "model": "deepseek-flash", "committee": "first",
                "outcome": "agent_failed", "cost_usd": 0.0001,
                "runtime": {"failure_class": "turn_failed", "failed_turns": 1,
                            "last_terminal_kind": "failed"},
            }) + "\n")
    size = path.stat().st_size
    assert size > 1_000_000

    tracemalloc.start()
    try:
        doc = brief.summarize_run(run, expand={"models", "failures"})
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= 4 * 1024 * 1024, f"failure-heavy brief 峰值 {peak / 1e6:.1f}MB 超过 4MB"
    assert len(doc["agent_failures"]) == 4, "失败只保留最后 4 条"
    assert doc["model_totals"][0]["sessions"] == 10_000, "聚合仍要覆盖全部 agent 行"


def test_the_timeline_queries_use_window_indexes(tmp_path, monkeypatch):
    """@brief 时间线的 closed span / pending wait 查询都走窗口索引，不做全表扫描。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    state, result, status, path = _timeline_run(tmp_path, monkeypatch)
    import observe.index as index_mod

    index = index_mod.get_index()
    historical = []
    for step in range(5_000):
        when = T0 - 10_000.0 + step
        historical.append(_gpu("wait", when))
        historical.append(_gpu("acquire", when + 0.5))
    _gpu_events(path, historical + [_gpu("wait", T0 + 10), _gpu("acquire", T0 + 20)])
    index.ensure_gpu_events()

    for sql, args, expected in (
        (detail.LOCK_SPAN_SQL, (T0 - 2.0, T0 + 102.0), "lock_spans_end"),
        (detail.PENDING_WAIT_SQL, (T0 - 2.0, T0 + 102.0), "gpu_waits_t"),
    ):
        plan = " ".join(str(row[3]) for row in index.conn.execute("EXPLAIN QUERY PLAN " + sql, args))
        assert expected in plan, plan
        assert f"SCAN {expected.split('_')[0]}" not in plan, plan

    # 窗口条件必须下推到 SQL：SQLite 交给 Python 的行数就是窗口内的行数，而不是全部未来 span。
    yielded = index.conn.execute(detail.LOCK_SPAN_SQL, (T0 - 2.0, T0 + 102.0)).fetchall()
    assert len(yielded) == 1, f"锁窗口查询从 SQLite 取出了 {len(yielded)} 行"
    closed, waits = detail._lock_rows(index.conn, T0 - 2.0, T0 + 102.0)
    assert len(closed) == 1 and waits == []
    assert detail._lock_spans(state, result, None, status) == [{"start": T0 + 10, "end": T0 + 20}]


def test_a_lost_shard_stays_truncated_after_dropping_the_derived_database(tmp_path, monkeypatch):
    """@brief 截断标记写在源侧：分片丢失后删掉派生库重建，仍然显式记住“历史不完整”。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _state, _result, _status, path = _timeline_run(tmp_path, monkeypatch)
    import observe.index as index_mod

    index = index_mod.get_index()
    _gpu_events(path, [_gpu("wait", T0 + 1), _gpu("acquire", T0 + 2)])
    index.ensure_gpu_events()
    shard = path.with_name(path.name + ".1")
    path.rename(shard)
    _gpu_events(path, [_gpu("wait", T0 + 3), _gpu("acquire", T0 + 4)])
    index.ensure_gpu_events()
    assert detail.gpu_history_truncated() is False

    shard.unlink()
    index.ensure_gpu_events()
    assert detail.gpu_history_truncated() is True
    marker = path.with_suffix(".truncated")
    assert marker.is_file(), "截断事实必须落在源侧标记文件上"

    db = index.path
    index.conn.close()
    for suffix in ("", "-wal", "-shm"):
        (db.with_name(db.name + suffix) if suffix else db).unlink(missing_ok=True)
    index_mod._SINGLE = None
    fresh = index_mod.get_index()
    assert detail.gpu_history_truncated() is True, "删库重建后不能把截断清成 0"
    fresh.ensure_gpu_events()
    assert detail.gpu_history_truncated() is True


def test_tail_lines_keeps_a_complete_line_when_the_budget_lands_on_a_boundary(tmp_path):
    """@brief 预算恰好落在行边界上时，第一段是完整行，不能丢。

    @param tmp_path pytest 提供的临时目录。
    """
    path = tmp_path / "edge.jsonl"
    path.write_text("1\n2\n3\n", encoding="utf-8")
    assert tail_lines(path, keep=2, max_bytes=4) == ["2", "3"]
    assert tail_lines(path, keep=3, max_bytes=4) == ["2", "3"]


def test_tail_text_treats_a_directory_as_unreadable(tmp_path):
    """@brief ``tail_text`` 对目录等不可读路径返回空串，不抛 ``IsADirectoryError``。

    @param tmp_path pytest 提供的临时目录。
    """
    assert tail_text(tmp_path, limit=10) == ""


def test_rotated_gpu_events_stay_in_both_tail_views(tmp_path, monkeypatch):
    """@brief 轮转后 overview/detail 的尾读与 ``brief --gpu`` 都返回跨分片的逻辑尾。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    monkeypatch.setenv("AI_LOWERLING_OBSERVE_DIR", str(tmp_path / "cache"))
    path = tmp_path / "cache" / "observe" / "gpu_events.jsonl"
    path.parent.mkdir(parents=True)
    _gpu_events(path, [_gpu("wait", T0 + 1), _gpu("acquire", T0 + 2), _gpu("wait", T0 + 3)])
    path.rename(path.with_name(path.name + ".1"))
    _gpu_events(path, [_gpu("acquire", T0 + 4)])

    from taic import brief

    assert [row["t"] for row in detail._tail_events()] == [T0 + 1, T0 + 2, T0 + 3, T0 + 4]
    assert [row["t"] for row in brief._gpu_events()] == [T0 + 1, T0 + 2, T0 + 3, T0 + 4]

    # 活动文件暂缺（轮转间隙）：仍然要读分片的逻辑尾。
    path.unlink()
    assert [row["t"] for row in detail._tail_events()] == [T0 + 1, T0 + 2, T0 + 3]
    assert [row["t"] for row in brief._gpu_events()] == [T0 + 1, T0 + 2, T0 + 3]


def test_tail_text_keeps_a_complete_line_when_the_cut_lands_on_a_boundary(tmp_path):
    """@brief 切点正好在行边界时 ``tail_text`` 不能丢掉完整的第一行。

    @param tmp_path pytest 提供的临时目录。
    """
    path = tmp_path / "edge.txt"
    path.write_text("1\n2\n3\n", encoding="utf-8")
    assert tail_text(path, limit=4) == "2\n3\n"


def test_brief_gpu_tail_keeps_valid_events_across_shards(tmp_path, monkeypatch):
    """@brief ``brief --gpu`` 取“最后 8 条有效事件”，坏行不占名额；轮转分片也要看。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    monkeypatch.setenv("AI_LOWERLING_OBSERVE_DIR", str(tmp_path / "cache"))
    path = tmp_path / "cache" / "observe" / "gpu_events.jsonl"
    path.parent.mkdir(parents=True)
    valid = [_gpu("wait", T0 + index) for index in range(1, 10)]
    _gpu_events(path, valid[:5])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
    _gpu_events(path, valid[5:], mode="a")
    path.rename(path.with_name(path.name + ".1"))
    _gpu_events(path, [_gpu("acquire", T0 + 100)])

    from taic import brief

    shown = [row["t"] for row in brief._gpu_events()]
    assert shown == [T0 + index for index in range(3, 10)] + [T0 + 100], shown
    assert len(shown) == 8, "坏行不能占掉一条有效事件的名额"


def test_consecutive_rotation_gaps_still_pick_up_new_shards(tmp_path, monkeypatch):
    """@brief 活动文件连续缺失期间又轮转出新分片：不能因为“旧分片没丢”就提前返回。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _state, _result, _status, path = _timeline_run(tmp_path, monkeypatch)
    import observe.index as index_mod

    index = index_mod.get_index()
    _gpu_events(path, [_gpu("wait", T0 + 1)])
    index.ensure_gpu_events()
    path.rename(path.with_name(path.name + ".1"))
    index.ensure_gpu_events()
    assert index.conn.execute("SELECT COUNT(*) FROM gpu_events").fetchone()[0] == 1

    _gpu_events(path, [_gpu("wait", T0 + 2)])
    path.rename(path.with_name(path.name + ".2"))
    index.ensure_gpu_events()
    rows = [row["t"] for row in index.conn.execute("SELECT t FROM gpu_events ORDER BY t")]
    assert rows == [T0 + 1, T0 + 2]
    assert detail.gpu_history_truncated() is False


def test_a_middle_rewrite_of_a_large_active_file_is_detected(tmp_path, monkeypatch):
    """@brief 大于 8KB 的文件只改中段：必须做完整前缀对拍，不能靠两个采样点。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _state, _result, _status, path = _timeline_run(tmp_path, monkeypatch)
    import observe.index as index_mod

    index = index_mod.get_index()
    events = [_gpu("wait", T0 + 1, what="x" * 120) for _ in range(150)]
    _gpu_events(path, events)
    assert path.stat().st_size > 16_000
    index.ensure_gpu_events()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[75]["what"] = "y" * 120
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    index.ensure_gpu_events()
    assert detail.gpu_history_truncated() is True
    stored = [row["what"] for row in index.conn.execute("SELECT what FROM gpu_events ORDER BY rowid")]
    assert set(stored) == {"x" * 120, "y" * 120}, "中段改写必须重读"


def test_an_empty_active_file_after_rotation_is_not_truncated(tmp_path, monkeypatch):
    """@brief 写者 create-before-append：轮转后先建空活动文件，再写第一条，不能产生假截断。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _state, _result, _status, path = _timeline_run(tmp_path, monkeypatch)
    import observe.index as index_mod

    index = index_mod.get_index()
    _gpu_events(path, [_gpu("wait", T0 + 1)])
    index.ensure_gpu_events()
    path.rename(path.with_name(path.name + ".1"))
    path.write_text("", encoding="utf-8")
    index.ensure_gpu_events()
    index.ensure_gpu_events()
    assert detail.gpu_history_truncated() is False
    assert index.conn.execute("SELECT COUNT(*) FROM gpu_events").fetchone()[0] == 1


def test_a_same_size_shard_replacement_is_marked_truncated(tmp_path, monkeypatch):
    """@brief 不可变分片被等长改写也要发现：记录并核对 inode/mtime/size。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _state, _result, _status, path = _timeline_run(tmp_path, monkeypatch)
    import observe.index as index_mod

    index = index_mod.get_index()
    _gpu_events(path, [_gpu("wait", T0 + 1, what="x" * 120)])
    size = path.stat().st_size
    path.rename(path.with_name(path.name + ".1"))
    _gpu_events(path, [_gpu("wait", T0 + 2)])
    index.ensure_gpu_events()
    shard = path.with_name(path.name + ".1")
    shard.write_text(json.dumps(_gpu("wait", T0 + 1, what="y" * 120)) + "\n", encoding="utf-8")
    assert shard.stat().st_size == size
    index.ensure_gpu_events()
    assert detail.gpu_history_truncated() is True


def test_a_large_history_late_append_replays_with_bounded_memory(tmp_path, monkeypatch):
    """@brief 一条乱序追加触发的重放必须用流式游标，内存不随全历史增长。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _state, _result, _status, path = _timeline_run(tmp_path, monkeypatch)
    import observe.index as index_mod

    index = index_mod.get_index()
    events = []
    for step in range(20_000):
        events.append(_gpu("wait", T0 + 1 + step * 0.001))
        events.append(_gpu("acquire", T0 + 1.0005 + step * 0.001))
    _gpu_events(path, events)
    index.ensure_gpu_events()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_gpu("wait", T0 - 100)) + "\n")

    tracemalloc.start()
    try:
        index.ensure_gpu_events()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= 8 * 1024 * 1024, f"late-append 重放峰值 {peak / 1e6:.1f}MB 超过 8MB"


def test_a_cross_file_tail_keeps_a_complete_first_line_at_the_budget_boundary(tmp_path):
    """@brief PR5 回归：预算恰好落在行首时不能丢掉那条完整行（含跨分片与完整 JSON 事件）。

    @param tmp_path pytest 提供的临时目录。
    """
    from taic.reads import tail_text, tail_text_across, tail_valid_rows_across

    plain = tmp_path / "plain.log"
    plain.write_text("1\n2\n3\n", encoding="utf-8")
    assert tail_text(plain, limit=4) == "2\n3\n"
    assert tail_text_across([plain], limit=4) == "2\n3\n"
    # 切点落在行中间时仍丢半行。
    assert tail_text_across([plain], limit=3) == "3\n"

    # 跨分片：预算从新往旧花，最旧的那份恰好从行首开始。
    older = tmp_path / "events.log.1"
    newer = tmp_path / "events.log"
    older.write_text("1\n2\n", encoding="utf-8")
    newer.write_text("3\n", encoding="utf-8")
    assert tail_text_across([older, newer], limit=4) == "2\n3\n"
    assert tail_text_across([older, newer], limit=5) == "2\n3\n"
    assert tail_text_across([older, newer], limit=10) == "1\n2\n3\n"

    # 恰好覆盖一个完整 JSON 事件时不能把它当成半行丢掉。
    event = json.dumps({"event": "usage", "cost": 0.5})
    log = tmp_path / "events.jsonl"
    log.write_text("older\n" + event + "\n", encoding="utf-8")
    rows = tail_valid_rows_across([log], keep=1, max_bytes=len(event) + 1)
    assert rows == [{"event": "usage", "cost": 0.5}]


def test_the_span_query_is_bounded_at_the_sqlite_layer(tmp_path, monkeypatch):
    """@brief PR5：100k 条未来 span 不得被区间查询扫过；用 SQLite 的 VM 步数与执行计划证明。

    旧查询 ``end >= lo AND start <= hi`` 会沿 ``lock_spans_end`` 扫掉全部未来行。这里直接对
    实现使用的 SQL 做 ``EXPLAIN QUERY PLAN``，并用 ``set_progress_handler`` 在 SQLite 层计步：
    窗口查询的步数必须远小于一次全表扫描，且 R*Tree 与源表行数一致。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    state, result, status, path = _timeline_run(tmp_path, monkeypatch, ended=T0 + 10)
    events = []
    for index in range(50_000):
        when = T0 + index * 0.02
        events.append(_gpu("wait", when))
        events.append(_gpu("acquire", when + 0.001))
    _gpu_events(path, events)
    import observe.index as index_mod

    index = index_mod.get_index()
    index.ensure_gpu_events()
    conn = index.read_only()
    try:
        assert conn.execute("SELECT COUNT(*) FROM lock_spans").fetchone()[0] == 50_000
        lo, hi = T0 - 2.0, T0 + 12.0
        first = int(lo // detail.LOCK_SPAN_BUCKET_SECONDS)
        last = int(hi // detail.LOCK_SPAN_BUCKET_SECONDS)
        buckets = conn.execute(
            "SELECT COUNT(DISTINCT bucket), COUNT(*) FROM lock_span_slots WHERE bucket BETWEEN ? AND ?",
            (first, last),
        ).fetchone()
        assert buckets[0] <= 4, "窗口只覆盖极少几个桶"
        assert buckets[1] < 5000, "桶内候选集随窗口秒数有界，不随总历史增长"
        plan = conn.execute(
            "EXPLAIN QUERY PLAN " + detail.LOCK_SPAN_SLOT_SQL, (first, last, hi, lo)
        ).fetchall()
        text = " | ".join(str(row["detail"]) for row in plan)
        assert "lock_span_slots" in text and "USING PRIMARY KEY" in text, text
        assert "SCAN l" not in text and "SCAN lock_spans" not in text, text
        assert "SCAN l" not in text and "SCAN lock_spans" not in text, text

        steps = {"n": 0}

        def handler() -> int:
            steps["n"] += 1
            return 0

        conn.set_progress_handler(handler, 1)
        try:
            rows = conn.execute(detail.LOCK_SPAN_SLOT_SQL, (first, last, hi, lo)).fetchall()
        finally:
            conn.set_progress_handler(None, 1)
        window_steps = steps["n"]
        assert len(rows) == 601

        steps["n"] = 0
        conn.set_progress_handler(handler, 1)
        try:
            scanned = conn.execute("SELECT pid, start, end FROM lock_spans").fetchall()
        finally:
            conn.set_progress_handler(None, 1)
        full_steps = steps["n"]
        assert len(scanned) == 50_000
        assert window_steps < full_steps / 8, (window_steps, full_steps)
    finally:
        conn.close()
    assert detail._lock_spans(state, result, None, status) == _old_lock_spans(
        events, started=T0, ended=T0 + 10
    )
