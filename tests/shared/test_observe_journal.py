"""PR24 写者日志：白名单、分片、只摄取点名的 run、轮转回退与 tick 新鲜度。

10k 的墙钟数字由 ``experiments.scale_probe`` 报告；这里断言机制与逐字段结果。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import taic.control as control
from experiments.manifest import ensure
from observe.index import Index, get_index
from tcenv import index_journal
from tcenv.schema import PROTOCOL_VERSION


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


def _run(root: Path, name: str, *, candidates: int = 0, transcript: bool = False) -> Path:
    """@brief 造一个最小的已结束运行目录（用普通 write，不触发日志）。

    @param root ``runs/`` 根。
    @param name 相对运行名。
    @param candidates 候选目录个数。
    @param transcript 是否写一份 transcript。
    @return 运行目录。
    """
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "phase": "done", "started_at": 1, "updated_at": 1,
    }), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.1, "budget_usd": 1,
    }), encoding="utf-8")
    if candidates:
        folder = directory / "candidates"
        folder.mkdir()
        for index in range(candidates):
            cand = folder / f"{index:03d}_r1.p0"
            cand.mkdir()
            (cand / "kernel.metal").write_text("kernel void k() {}\n", encoding="utf-8")
            (cand / "report.json").write_text(json.dumps({"status": "ok", "speedup": 1.0}), encoding="utf-8")
    if transcript:
        (directory / "transcript.jsonl").write_text(
            json.dumps({"event": "chat", "reply": "hi"}) + "\n", encoding="utf-8")
    return directory


def _lines(runs: Path) -> list[dict]:
    """@brief 读出本进程分片里的每一行。

    @param runs ``runs/`` 根。
    @return 解析后的字典列表。
    """
    shard = index_journal.journal_dir(runs) / f"{os.getpid()}.jsonl"
    if not shard.is_file():
        return []
    rows = []
    for line in shard.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _rows(index: Index) -> list[dict]:
    """@brief 取 ``runs`` 表的关键字段（用于逐字段对拍）。

    @param index 索引。
    @return 按 run_id 排序的行。
    """
    return [dict(row) for row in index.conn.execute("SELECT * FROM runs ORDER BY run_id")]


def _clean_rebuild(tmp_path: Path) -> list[dict]:
    """@brief 从同一棵目录干净重建一份索引并取行。

    @param tmp_path pytest 临时目录。
    @return 按 run_id 排序的行。
    """
    fresh = Index(tmp_path / "fresh.sqlite")
    fresh.sync_known()
    return _rows(fresh)


def test_only_the_whitelisted_files_reach_the_journal_and_the_shard_is_per_pid(tmp_path, monkeypatch):
    """@brief state/result/confirm/config/best/候选/反馈/清单会记一行；transcript/progress 不记。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    directory = _run(runs, "run", candidates=1)
    control.write_json_atomic(directory / "confirm_report.json", {"status": "ok"})
    control.write_json_atomic(directory / "run_config.json", {"task": "relu"})
    control.write_text_atomic(directory / "best.metal", "kernel void k() {}\n")
    control.write_json_atomic(directory / "candidates" / "000_r1.p0" / "report.json", {"status": "ok"})
    control.write_text_atomic(directory / "candidates" / "000_r1.p0" / "kernel.metal", "kernel void k() {}\n")
    control.append_jsonl(directory / "feedback.jsonl", {"seq": 1})
    control.write_json_atomic(directory / "ledger.json", {"summary": {}})
    control.record_event(directory, "applied", 0.0, 1.0)
    shard = index_journal.journal_dir(runs) / f"{os.getpid()}.jsonl"
    assert shard.is_file()
    paths = [row["path"] for row in _lines(runs)]
    for name in ("confirm_report.json", "run_config.json", "best.metal", "report.json", "kernel.metal",
                 "feedback.jsonl", "ledger.json", "budget_events.json"):
        assert name in paths, (name, paths)
    assert all(row["run"] == "run" for row in _lines(runs))
    # 文件名对但不在 candidates/ 下的 report.json 不记。
    before = len(_lines(runs))
    control.write_text_atomic(directory / "report.json", "{}")
    assert len(_lines(runs)) == before
    # transcript / eval_progress 不是索引输入，不记。
    control.write_text_atomic(directory / "transcript.jsonl", '{"event": "chat"}\n')
    control.write_json_atomic(directory / "eval_progress.json", {"stage": "run"})
    assert len(_lines(runs)) == before
    # runs/ 外的写入不记。
    control.write_json_atomic(tmp_path / "elsewhere" / "state.json", {})
    control.write_json_atomic(tmp_path / "state.json", {})
    assert len(_lines(runs)) == before

    # experiment.json 记实验根，而不是某个 run。
    experiment = runs / "exp"
    ensure(experiment, kind="search_ab", driver="search_ab", frozen={"tasks": ["relu"]},
           arms=[{"arm_id": "on", "delta": {}, "planned_n": 1}])
    rows = [row for row in _lines(runs) if row["path"] == "experiment.json"]
    assert rows and rows[-1]["run"] == "exp", rows


def test_a_journal_write_failure_never_raises(tmp_path, monkeypatch):
    """@brief 写者日志失败只吞掉，不影响状态发布与反馈追加。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    directory = _run(runs, "run")

    def broken(*_args, **_kwargs):
        raise OSError("journal is gone")

    monkeypatch.setattr(index_journal, "_append", broken)
    index_journal.note(directory / "state.json")
    control.write_json_atomic(directory / "state.json", {"task": "relu", "phase": "done"})
    control.append_jsonl(directory / "feedback.jsonl", {"seq": 1})
    assert json.loads((directory / "state.json").read_text(encoding="utf-8"))["phase"] == "done"
    assert (directory / "feedback.jsonl").read_text(encoding="utf-8").strip()


def test_a_journal_line_ingests_only_the_run_it_names(tmp_path, monkeypatch):
    """@brief 40 场里改 1 场：热 sync_known 只摄取它；伪造不存在与真实 run 的行都不报错。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    for index in range(40):
        _run(runs, f"run_{index:02d}", candidates=1)
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    before = {row["run_id"]: row["fp_state"] for row in _rows(index)}

    target = runs / "run_07"
    state = json.loads((target / "state.json").read_text(encoding="utf-8"))
    state["updated_at"] = 999
    control.write_json_atomic(target / "state.json", state)
    # 伪造两条：一条不存在的 run，一条真实但没变的 run。
    shard = index_journal.journal_dir(runs) / f"{os.getpid()}.jsonl"
    with shard.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"run": "nope/missing", "path": "state.json", "at": 1}) + "\n")
        handle.write(json.dumps({"run": "run_08", "path": "state.json", "at": 1}) + "\n")

    calls = []
    original = Index._ingest

    def counting(self, directory, *, force=False, once=True):
        calls.append(str(directory))
        return original(self, directory, force=force, once=once)

    monkeypatch.setattr(Index, "_ingest", counting)
    index.sync_known()
    # 伪造的行只导致“stat 它”：run_07 变了，run_08 被点名但指纹未变；不存在的 run 被跳过。
    assert sorted(Path(item).name for item in calls) == ["run_07", "run_08"], calls
    after = {row["run_id"]: row["fp_state"] for row in _rows(index)}
    changed = [name for name in before if before[name] != after[name]]
    assert changed == ["run_07"], changed
    row = index.conn.execute("SELECT updated_at FROM runs WHERE run_id = 'run_07'").fetchone()
    assert row[0] == 999


def test_deleting_the_journal_falls_back_to_the_same_rows(tmp_path, monkeypatch):
    """@brief 删掉日志后退回逐 run 对账：就地改 state 仍被发现，结果与干净重建逐字段一致。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    for index in range(6):
        _run(runs, f"run_{index}")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    control.write_json_atomic(runs / "run_3" / "state.json", {"task": "relu", "dtype": "fp16", "phase": "done", "updated_at": 42})
    for item in index_journal.journal_dir(runs).iterdir():
        item.unlink()
    index.sync_known()
    assert index.conn.execute("SELECT updated_at FROM runs WHERE run_id = 'run_3'").fetchone()[0] == 42
    assert _rows(index) == _clean_rebuild(tmp_path)


def test_a_truncated_shard_forces_one_full_reconcile(tmp_path, monkeypatch):
    """@brief 分片被截断（size < offset）→ 从头重读并补一次全量对账，结果与干净重建一致。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    for index in range(8):
        _run(runs, f"run_{index}")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    shard = index_journal.journal_dir(runs) / f"{os.getpid()}.jsonl"
    # 写者日志里有真实内容，先把偏移推大。
    for step in range(30):
        control.write_json_atomic(runs / "run_0" / "state.json", {"task": "relu", "phase": "done", "updated_at": step})
    index.sync_known()
    consumed = int(index.conn.execute(
        f"SELECT value FROM meta WHERE key = 'journal_offset:{shard.name}'"
    ).fetchone()[0])
    assert consumed > 100
    shard.write_bytes(shard.read_bytes()[:100])
    calls = []
    original = Index._ingest

    def counting(self, directory, *, force=False, once=True):
        calls.append(str(directory))
        return original(self, directory, force=force, once=once)

    monkeypatch.setattr(Index, "_ingest", counting)
    index.sync_known()
    assert len(calls) >= 8, calls
    assert _rows(index) == _clean_rebuild(tmp_path)


def _confirmed_run(root: Path, name: str = "confirmed") -> Path:
    """@brief 造一场通过发布门的运行（best/候选/确认报告摘要一致）。

    @param root ``runs/`` 根。
    @param name 运行名。
    @return 运行目录。
    """
    directory = _run(root, name)
    kernel = "kernel void k() {}\n"
    digest = hashlib.sha256(kernel.encode("utf-8")).hexdigest()
    (directory / "best.metal").write_text(kernel, encoding="utf-8")
    candidate = directory / "candidates" / "000_r1.p0"
    candidate.mkdir(parents=True)
    (candidate / "kernel.metal").write_text(kernel, encoding="utf-8")
    (candidate / "report.json").write_text(json.dumps({"status": "ok", "speedup": 1.5}), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": "paper", "stop_reason": "max_rounds",
        "started_at": 1, "spent_usd": 0.1, "target_id": "apple-m4-max",
        "confirmed_speedup": 1.5, "confirmed_status": "ok", "confirm_winner_label": "r1.p0",
        "confirmation_source_sha256": digest, "confirmed_values": [1.5],
        "confirmed_worst": 1.5, "confirmed_best": 1.5,
    }), encoding="utf-8")
    (directory / "confirm_report.json").write_text(json.dumps({
        "status": "ok", "speedup": 1.5,
        "benchmark": {
            "noisy": False,
            "candidate": {"cv": 0.01, "reps": 15},
            "baselines": {"mlx": {"cv": 0.01, "reps": 15}},
            "best_baseline": "mlx",
            "confirmation": {
                "kernel_sha256": digest, "winner_label": "r1.p0",
                "confirmed_values": [1.5], "speedup": 1.5, "candidate_us": None,
            },
        },
        "protocol_version": PROTOCOL_VERSION,
        "target_id": "apple-m4-max",
    }), encoding="utf-8")
    return directory


def _remove_journal(runs: Path) -> None:
    """@brief 清掉写者日志目录，模拟“没有 journal”的兜底路径。

    @param runs ``runs/`` 根。
    """
    folder = index_journal.journal_dir(runs)
    if folder.is_dir():
        for item in folder.iterdir():
            item.unlink()
        folder.rmdir()


def test_one_tick_sees_an_in_place_confirm_edit_with_and_without_the_journal(tmp_path, monkeypatch):
    """@brief 不动目录 mtime 改写 confirm_report.json → 一次 tick 后发布结论必须变。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    directory = _confirmed_run(runs)
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    first = index.conn.execute("SELECT publication, published_speedup FROM runs WHERE run_id = 'confirmed'").fetchone()
    assert first["publication"] == "confirmed" and first["published_speedup"] == 1.5
    original = (directory / "confirm_report.json").read_text(encoding="utf-8")
    for mode in ("journal", "no-journal"):
        if mode == "no-journal":
            _remove_journal(runs)
        (directory / "confirm_report.json").write_text(json.dumps({"status": "failed", "speedup": None}), encoding="utf-8")
        index.tick()
        row = index.conn.execute("SELECT publication, published_speedup FROM runs WHERE run_id = 'confirmed'").fetchone()
        assert row["publication"] != "confirmed", (mode, dict(row))
        assert row["published_speedup"] is None, (mode, dict(row))
        # 恢复成通过，下一轮再验证一次（两个方向都变）。
        (directory / "confirm_report.json").write_text(original, encoding="utf-8")
        index.tick()
        again = index.conn.execute("SELECT publication, published_speedup FROM runs WHERE run_id = 'confirmed'").fetchone()
        assert again["publication"] == "confirmed" and again["published_speedup"] == 1.5, (mode, dict(again))


def test_one_tick_sees_a_non_winner_candidate_edit_with_and_without_the_journal(tmp_path, monkeypatch):
    """@brief 非 winner 候选原地改写 → 一次 tick 后 failure_sig 与 failure_members 必须变。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    directory = _run(runs, "mixed", candidates=1)
    winning = directory / "candidates" / "001_r1.p0"
    winning.mkdir()
    (winning / "kernel.metal").write_text("kernel void w() {}\n", encoding="utf-8")
    (winning / "report.json").write_text(json.dumps({"status": "ok", "speedup": 2.0}), encoding="utf-8")
    failing = directory / "candidates" / "000_r1.p0"
    (failing / "report.json").write_text(json.dumps({
        "status": "error", "message": "abi_error: undefined symbol a", "stage": "compile",
    }), encoding="utf-8")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)

    def failure_rows():
        return [dict(row) for row in index.conn.execute(
            "SELECT run_id, idx, sig, status, stage FROM failure_members ORDER BY idx")]

    before = failure_rows()
    assert before and before[0]["sig"], before
    for mode in ("journal", "no-journal"):
        if mode == "no-journal":
            _remove_journal(runs)
        (failing / "report.json").write_text(json.dumps({
            "status": "error", "message": "abi_error: undefined symbol b", "stage": "link",
        }), encoding="utf-8")
        index.tick()
        after = failure_rows()
        assert after[0]["sig"] != before[0]["sig"], (mode, after)
        # 恢复原内容，下一轮再变回来。
        (failing / "report.json").write_text(json.dumps({
            "status": "error", "message": "abi_error: undefined symbol a", "stage": "compile",
        }), encoding="utf-8")
        index.tick()
        assert failure_rows()[0]["sig"] == before[0]["sig"], mode


def test_one_tick_sees_an_outbox_rewrite_in_place(tmp_path, monkeypatch):
    """@brief 原地把 outbox 请求 pending → unavailable → run_pending_eval 必须变。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    directory = _run(runs, "outbox")
    outbox = directory / "evaluation_outbox"
    outbox.mkdir()
    request = outbox / "req_001.json"
    request.write_text(json.dumps({
        "request_id": "req_001", "status": "pending", "error": None,
    }), encoding="utf-8")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)

    def projection():
        return dict(index.conn.execute(
            "SELECT pending, unavailable, last_error_code FROM run_pending_eval WHERE run_id = 'outbox'"
        ).fetchone())

    assert projection()["pending"] == 1
    request.write_text(json.dumps({
        "request_id": "req_001", "status": "unavailable",
        "error": {"code": "site_unreachable", "retryable": False},
    }), encoding="utf-8")
    index.tick()
    row = projection()
    assert row["pending"] == 0 and row["unavailable"] == 1 and row["last_error_code"] == "site_unreachable", row


def test_a_new_run_that_only_writes_state_is_still_discovered(tmp_path, monkeypatch):
    """@brief 新 run 的第一份 state.json 经日志到达，不需要等对账周期。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    for index in range(5):
        _run(runs, f"run_{index}")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    fresh = runs / "nested" / "deep" / "new_run"
    fresh.mkdir(parents=True)
    control.write_json_atomic(fresh / "state.json", {"task": "relu", "dtype": "fp16", "phase": "running", "updated_at": 5})
    index.tick()
    assert index.conn.execute("SELECT 1 FROM runs WHERE run_id = 'nested/deep/new_run'").fetchone() is not None
    assert get_index() is not None


def _healthy_frame_stats(index: Index, runs: Path, monkeypatch) -> list[str]:
    """@brief 在一个已建好的索引上跑健康日志帧，返回它触发的 ``Path.stat`` 路径。

    @param index 已建好的同一份索引（不能新建，否会走冷建）。
    @param runs 已在用的 ``runs/`` 根。
    @param monkeypatch pytest 替换工具。
    @return 被 stat 的路径文本列表。
    """
    calls: list[str] = []
    real_stat = Path.stat

    def counting_stat(self, *args, **kwargs):
        calls.append(str(self))
        return real_stat(self, *args, **kwargs)

    shard = index_journal.journal_dir(runs) / f"{os.getpid()}.jsonl"
    with shard.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"run": "run_08", "path": "state.json", "at": 1}) + "\n")
    monkeypatch.setattr(Path, "stat", counting_stat)
    try:
        index.sync_known()
    finally:
        monkeypatch.setattr(Path, "stat", real_stat)
    return calls


def test_a_healthy_journal_frame_does_not_stat_every_watched_directory(tmp_path, monkeypatch):
    """@brief 40/80 场的健康日志帧 stat 次数几乎不变：不再随 run 总数线性增长。

    旧实现先遍历 ``watched`` 逐目录 stat，再消费 journal；健康日志帧只应 stat 根/父目录与点名的 run。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    for index in range(40):
        _run(runs, f"run_{index:02d}", candidates=1)
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    # 第一条写者日志：创建 .index_journal 会改 runs/ 的 mtime，这一帧允许一次根扫描。
    control.write_json_atomic(runs / "run_07" / "state.json", {"task": "relu", "phase": "done", "updated_at": 7})
    index.sync_known()
    small = _healthy_frame_stats(index, runs, monkeypatch)

    runs80 = tmp_path / "tree80" / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs80))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "tree80" / "observe.sqlite"))
    runs80.mkdir(parents=True)
    for number in range(80):
        _run(runs80, f"run_{number:02d}", candidates=1)
    big = Index(tmp_path / "tree80" / "observe.sqlite")
    big.sync_known(force=True)
    control.write_json_atomic(runs80 / "run_07" / "state.json", {"task": "relu", "phase": "done", "updated_at": 7})
    big.sync_known()
    large = _healthy_frame_stats(big, runs80, monkeypatch)

    assert len(large) <= len(small) + 5, (len(small), len(large))
    assert len(small) < 90, small
    assert not any(item.endswith("/runs/run_00") for item in small), "unchanged run was statted"
    assert not any(item.endswith("/runs/run_39") for item in small), "unchanged run was statted"
    assert any(item.endswith("/runs/run_08") or item.endswith("/runs/run_08/state.json") for item in small), small


def test_a_healthy_journal_frame_does_not_realpath_every_run(tmp_path, monkeypatch):
    """@brief 健康日志帧只对点名的 run 做一次 `realpath`；普通 run 不逐个解析。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = _env(tmp_path, monkeypatch)
    for index in range(40):
        _run(runs, f"run_{index:02d}", candidates=1)
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    control.write_json_atomic(runs / "run_07" / "state.json", {"task": "relu", "phase": "done", "updated_at": 7})
    index.sync_known()
    calls: list[str] = []
    real_realpath = os.path.realpath

    def counting_realpath(path, *args, **kwargs):
        calls.append(str(path))
        return real_realpath(path, *args, **kwargs)

    monkeypatch.setattr(os.path, "realpath", counting_realpath)
    shard = index_journal.journal_dir(runs) / f"{os.getpid()}.jsonl"
    with shard.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"run": "run_08", "path": "state.json", "at": 1}) + "\n")
    index.sync_known()
    monkeypatch.setattr(os.path, "realpath", real_realpath)
    assert len(calls) <= 10, calls
    assert all("run_0" not in path or "run_08" in path for path in calls), calls
