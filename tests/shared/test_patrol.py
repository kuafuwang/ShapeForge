"""§6.4：``taic query patrol`` 是一屏有界判定——不摄取索引、不解码 ``row_json``、≤1.5KB。

数据只来自窄列与派生表（``runs``/``confirmed_best``/``failure_members``/``sessions``/``run_pending_eval``）。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from observe import detail, query
from observe.index import Index
from taic import cli, retrieval

PATROL_KEYS = {"schema_version", "at", "index_revision", "protocol_epochs", "alerts", "live", "failing", "best", "truncated"}


def _bytes(value) -> int:
    """@brief plan §6.4 的序列化口径。

    @param value 可 JSON 序列化的对象。
    @return UTF-8 字节数。
    """
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _index(tmp_path, monkeypatch):
    """@brief 临时 runs/ + 索引，关掉读时 tick。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @return ``(runs, Index)``。
    """
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "books.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    monkeypatch.setenv("OBSERVE_REFRESH_ON_READ", "0")
    # 巡检的 best 读目录史；测试必须自带一个空的 catalog 根，不能读仓库里真实的 kernels/。
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(tmp_path / "kernels"))
    # archive_confirmed 现在同时写 blob；测试的 blob 也必须落在临时目录里。
    monkeypatch.setenv("TAIC_BLOB_ROOT", str(tmp_path / "blobs"))
    return runs, Index(tmp_path / "observe.sqlite")


def _run_row(index, name: str, *, spent: float = 0.1, granted: float = 1.0, stop: str = "max_rounds",
             status: str = "done") -> None:
    """@brief 直接插一行 ``runs``（窄列足够，``row_json`` 只是占位）。

    @param index 打开的索引。
    @param name run id。
    @param spent 已花费。
    @param granted 告警预算。
    @param stop 停止原因。
    @param status 运行状态。
    """
    index.conn.execute(
        "INSERT INTO runs (run_id, task, dtype, status, stop_reason, stop_code, spent_usd, budget_usd,"
        " granted_budget_usd, publication, published_speedup, started_at, updated_at, row_json)"
        " VALUES (?, 'relu', 'fp16', ?, ?, ?, ?, ?, ?, 'in_progress', NULL, 1.0, 2.0, '{}')",
        (name, status, stop, stop.split(":", 1)[0][:24], spent, 1.0, granted),
    )


def test_patrol_is_a_bounded_fixed_shape_and_truncates_by_item_and_bytes(tmp_path, monkeypatch):
    """@brief 22 簇/20 个 best/12 条超预算时，巡检仍 ≤1,536B 并逐项报 ``truncated``。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, index = _index(tmp_path, monkeypatch)
    for number in range(22):
        sig = f"{number:016x}"
        index.conn.execute(
            "INSERT INTO failure_members (run_id, idx, sig, status, stage, observed_at)"
            " VALUES (?, 0, ?, 'incorrect', 'correctness', ?)",
            (f"cluster/{number}", sig, 100.0 + number),
        )
        index.conn.execute(
            "INSERT INTO candidates (run_id, idx, label, status, stage, message)"
            " VALUES (?, 0, 'r1.p0', 'incorrect', 'correctness', ?)",
            (f"cluster/{number}", f"worst: out[{number}] got -1.5"),
        )
    catalog = tmp_path / "kernels"
    for number in range(20):
        folder = catalog / "apple-m4-max" / f"task{number}" / "fp16" / "paper"
        folder.mkdir(parents=True)
        (folder / "history.jsonl").write_text(json.dumps({
            "target_id": "apple-m4-max", "task": f"task{number}", "dtype": "fp16", "size": "paper",
            "run_id": f"best/{number}", "winner_label": "r1.p0", "source_sha256": "a" * 64,
            "confirmed_speedup": 1.0 + number / 100, "confirmed_values": [1.0],
            "candidate_us": 100.0, "baseline_us": {"mlx": 120.0}, "updated_at": 1_700_000_000.0 + number,
        }) + "\n", encoding="utf-8")
    for number in range(12):
        _run_row(index, f"over/{number}", spent=5.0 + number, granted=1.0)
    index.conn.commit()

    doc = query.patrol()
    assert set(doc) == PATROL_KEYS
    assert set(doc["live"]) == {"runs", "evals", "gpu"}
    assert set(doc["live"]["gpu"]) == {"holders", "waiters"}
    assert set(doc["failing"]) == {"total", "clusters"}
    assert set(doc["truncated"]) == {"clusters", "best", "alerts", "epochs"}
    assert _bytes(doc) <= 1536, f"patrol {_bytes(doc)}B 超过 1,536B"
    assert doc["failing"]["total"] == 22
    assert len(doc["failing"]["clusters"]) < 22 and doc["truncated"]["clusters"] is True
    assert len(doc["best"]) < 20 and doc["truncated"]["best"] is True
    assert len(doc["alerts"]) <= 10 and doc["truncated"]["alerts"] is True
    # 三类列表都至少留一条代表，不能靠“清空某一栏”凑预算。
    assert doc["failing"]["clusters"] and doc["best"] and doc["alerts"]
    cluster = doc["failing"]["clusters"][0]
    assert set(cluster) == {"sig", "name", "n", "first", "last", "example"}
    assert set(cluster["example"]) == {"run", "label"}
    assert cluster["name"] and cluster["n"] == 1
    assert set(doc["best"][0]) == {"task", "target", "published_speedup", "confirmed_at", "stale_days"}
    for alert in doc["alerts"]:
        assert "run" in alert
        if alert["kind"].startswith("budget_"):
            assert "label" not in alert and "sig" not in alert


def test_patrol_bounds_the_protocol_epochs_metadata_too(tmp_path, monkeypatch):
    """@brief 评审回归：固定元数据也必须受 1,536B 约束，不能在三个列表之外无限增长。

    ``protocol_version`` 是 ``runs`` 的普通文本列：库里可以有任意多个、任意长的取值。
    旧实现把它们全部塞进 ``protocol_epochs``，于是 200 个普通长度就有 10.9KB、单个 2000 字符也有 2.2KB，
    而“逐项丢弃”只弹 best/簇/告警，救不回来，``truncated`` 还全是 false。现在纪元逐项 ≤24 字符、
    至多 8 项，超了就丢并置 ``truncated.epochs``；仍超预算时连纪元也会被逐项丢弃。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    for number in range(200):
        _run_row(index, f"epoch/{number}", spent=0.1, granted=1.0)
        index.conn.execute(
            "UPDATE runs SET protocol_version = ? WHERE run_id = ?",
            (f"version-{number:04d}-" + "x" * 32, f"epoch/{number}"),
        )
    index.conn.commit()

    doc = query.patrol()
    assert _bytes(doc) <= 1536, f"200 个纪元时 patrol {_bytes(doc)}B 超过 1,536B"
    assert len(doc["protocol_epochs"]) <= 8
    assert all(len(epoch) <= 24 for epoch in doc["protocol_epochs"])
    assert doc["truncated"]["epochs"] is True

    # 单个超长版本：截到 24 字符，仍要 ≤1.5KB 并如实报截断。
    runs2, index2 = _index(tmp_path / "long", monkeypatch)
    _run_row(index2, "epoch/long", spent=0.1, granted=1.0)
    index2.conn.execute("UPDATE runs SET protocol_version = ? WHERE run_id = 'epoch/long'", ("v" * 2000,))
    index2.conn.commit()
    long_doc = query.patrol()
    assert _bytes(long_doc) <= 1536
    assert long_doc["protocol_epochs"] == ["v" * 24]
    assert long_doc["truncated"]["epochs"] is True


def _archive(monkeypatch, catalog, *, task, dtype, size, speedup, run_id, confirmed_at, target="apple-m4-max"):
    """@brief 走真实写路径 ``kernel_catalog.archive_confirmed``，并把 ``time.time`` 固定成确认时刻。

    @param monkeypatch pytest 的替换工具。
    @param catalog catalog 根目录（`TAIC_CATALOG_ROOT`）。
    @param task 任务名。
    @param dtype 数据类型。
    @param size 尺寸标签。
    @param speedup 确认加速比。
    @param run_id 来源 run 的目录名。
    @param confirmed_at 确认时刻（Unix 秒）。
    @param target 目标 id。
    """
    import types

    from taic import kernel_catalog

    monkeypatch.setattr(kernel_catalog, "time", types.SimpleNamespace(time=lambda: confirmed_at))
    kernel_catalog.archive_confirmed(
        target_id=target, backend="metal", task=task, dtype=dtype, size=size,
        source=f"kernel void k_{speedup}() {{}}\n", speedup=speedup, run_id=run_id,
        label="r1.p0", confirmed_values=[speedup], candidate_us=100.0,
        baseline_us={"mlx": 120.0, "torch_mps": 130.0},
    )


def test_patrol_best_comes_from_the_catalog_history_through_the_real_write_path(tmp_path, monkeypatch):
    """@brief 评审回归：``best`` 必须走 ``kernel_history``（目录史的确认时刻），不是手造表行。

    用真实的 ``kernel_catalog.archive_confirmed`` 写三条确认史（同一 task/target）+ 别的任务，
    再把时钟固定成确认时刻：patrol 必须给出每个 (task, target) 一行、当前最佳取最大、
    ``confirmed_at`` 取目录史的 ``updated_at``（确认时刻），而不是 run 的 ``started_at``。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    catalog = tmp_path / "kernels"
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(catalog))
    _archive(monkeypatch, catalog, task="conv2d", dtype="fp16", size="paper", speedup=1.20,
             run_id="conv2d-early", confirmed_at=1000.0)
    _archive(monkeypatch, catalog, task="conv2d", dtype="fp16", size="paper", speedup=1.60,
             run_id="conv2d-best", confirmed_at=2000.0)
    _archive(monkeypatch, catalog, task="conv2d", dtype="fp16", size="paper", speedup=1.10,
             run_id="conv2d-late", confirmed_at=3000.0)
    _archive(monkeypatch, catalog, task="flash_attention", dtype="fp16", size="paper", speedup=1.03,
             run_id="flash-best", confirmed_at=1500.0)
    _runs, index = _index(tmp_path, monkeypatch)
    index.ensure_kernel_history()

    assert index.conn.execute("SELECT COUNT(*) FROM kernel_history").fetchone()[0] == 4
    doc = query.patrol()
    rows = {(row["task"], row["target"]): row for row in doc["best"]}
    assert set(rows) == {("conv2d", "apple-m4-max"), ("flash_attention", "apple-m4-max")}
    best_conv2d = rows[("conv2d", "apple-m4-max")]
    assert best_conv2d["published_speedup"] == pytest.approx(1.6), "当前最佳取该组最大值"
    assert best_conv2d["confirmed_at"] == pytest.approx(3000.0), "最后确认取目录史的确认时刻"
    assert rows[("flash_attention", "apple-m4-max")]["published_speedup"] == pytest.approx(1.03)
    assert [row["published_speedup"] for row in doc["best"]] == pytest.approx([1.6, 1.03])


def test_patrol_best_is_durable_when_the_publication_disappears(tmp_path, monkeypatch):
    """@brief 评审回归：发布丢失会让 ``confirmed_best`` 删行，但目录史仍在，patrol 的 best 不该消失。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    catalog = tmp_path / "kernels"
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(catalog))
    _archive(monkeypatch, catalog, task="conv2d", dtype="fp16", size="paper", speedup=1.4,
             run_id="conv2d-x", confirmed_at=5000.0)
    _runs, index = _index(tmp_path, monkeypatch)
    index.conn.execute(
        "INSERT INTO confirmed_best (run_id, task, dtype, size, target_id, protocol_epoch,"
        " published_speedup, kernel_sha256, confirmed_at) VALUES"
        " ('gone/run', 'conv2d', 'fp16', 'paper', 'apple-m4-max', '3', 1.4, ?, 10.0)",
        ("a" * 64,),
    )
    index.conn.commit()
    assert len(query.patrol()["best"]) == 1

    # 发布丢失：confirmed_best 行被删（例如 run 目录消失），目录史不受影响。
    index.conn.execute("DELETE FROM confirmed_best")
    index.conn.commit()
    doc = query.patrol()
    assert len(doc["best"]) == 1 and doc["best"][0]["published_speedup"] == pytest.approx(1.4)


def test_kernel_history_is_rebuildable_from_the_catalog(tmp_path, monkeypatch):
    """@brief 派生表不变量：删掉 ``kernel_history`` 行和 meta 后重投影，逐行一致。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    catalog = tmp_path / "kernels"
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(catalog))
    _archive(monkeypatch, catalog, task="conv2d", dtype="fp16", size="paper", speedup=1.25,
             run_id="one", confirmed_at=10.0)
    _archive(monkeypatch, catalog, task="relu", dtype="fp16", size="small", speedup=1.5,
             run_id="two", confirmed_at=20.0)
    _runs, index = _index(tmp_path, monkeypatch)
    index.ensure_kernel_history()
    query_text = (
        "SELECT target_id, task, dtype, size, seq, run_id_raw, winner_label, source_sha256,"
        " confirmed_speedup, candidate_us, updated_at FROM kernel_history"
        " ORDER BY task, dtype, size, seq"
    )
    first = [tuple(row) for row in index.conn.execute(query_text)]
    index.conn.execute("DELETE FROM kernel_history")
    index.conn.execute("DELETE FROM meta WHERE key = 'kernel_history_fp_v2'")
    index.conn.commit()
    index.ensure_kernel_history()
    assert [tuple(row) for row in index.conn.execute(query_text)] == first


def _bad_state_runs(tmp_path, monkeypatch):
    """@brief 造十种坏状态：4 超预算、4 会话失败（带 label）、2 待评测不可用；外加 3 个在跑。

    坏状态的 run 都有真实目录（摄取后不再手工插 ``runs`` 行），否则后续一次 ``tick()`` 会把它们
    标成 missing；在跑的 3 个用活 pid + 新鲜心跳，所以 ``live.runs`` 是精确的 3。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @return ``(runs, index)``。
    """
    runs, index = _index(tmp_path, monkeypatch)
    # 10 个坏状态：3 超预算、3 会话失败、2 待评测不可用、2 发布类（回退 + 丢失）。
    for kind, count in (("over", 3), ("session", 3), ("eval", 2), ("publication", 2)):
        for number in range(count):
            name = f"bad/{kind}{number}"
            directory = runs / "bad" / f"{kind}{number}"
            directory.mkdir(parents=True)
            common = {"task": "relu", "dtype": "fp16", "started_at": 1.0}
            spent = 2.0 + number if kind == "over" else 0.1
            (directory / "state.json").write_text(json.dumps({
                **common, "phase": "done", "updated_at": 2.0, "spent_usd": spent, "budget_usd": 1.0,
            }))
            (directory / "result.json").write_text(json.dumps({
                **common, "stop_reason": "max_rounds", "spent_usd": spent, "budget_usd": 1.0,
                "protocol_version": "3",
            }))
            if kind == "publication":
                # 一条目录史回退、一条发布丢失；两者都在真实摄取的 run 上。
                if number == 0:
                    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
            if kind == "session":
                # 真正的 agent_failed 会话经常没有候选：这里刻意不放 candidates/，只留一条转录事件，
                # 让会话失败来自真实摄取路径（旧夹具为了用 failures API 定位而注入了无关候选）。
                (directory / "transcript.jsonl").write_text(json.dumps({
                    "event": "agent", "label": "r1.p0", "candidate_label": "r1.p0", "attempt": 0,
                    "committee": "first", "provider": "deepseek", "model": "deepseek-flash",
                    "outcome": "agent_failed", "cost_usd": 0.1,
                    "runtime": {"failure_class": "turn_failed", "failed_turns": 2,
                                "last_terminal_kind": "failed", "last_error": "boom"},
                    "diagnostics_path": "agent_diagnostics/r1.json",
                }) + "\n", encoding="utf-8")
    for number in range(3):
        directory = runs / "live" / str(number)
        directory.mkdir(parents=True)
        (directory / "state.json").write_text(json.dumps({
            "task": "relu", "dtype": "fp16", "phase": "search", "pid": os.getpid(),
            "started_at": time.time() - 5, "updated_at": time.time(),
        }))
    # publication0 是回退的起点：把它做成真的确认过，夹具才只有「1 回退 + 1 丢失」两条发布告警。
    import hashlib

    directory = runs / "bad" / "publication0"
    source = b"kernel void k() {}\n"
    digest = hashlib.sha256(source).hexdigest()
    folder = directory / "candidates" / "000_r1.p0"
    folder.mkdir(parents=True)
    (folder / "kernel.metal").write_bytes(source)
    (folder / "report.json").write_text(json.dumps({"status": "ok", "stage": "bench", "speedup": 1.30}))
    (directory / "best.metal").write_bytes(source)
    (directory / "result.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": "paper", "target_id": "apple-m4-max",
        "protocol_version": "3", "stop_reason": "max_rounds", "spent_usd": 0.1, "budget_usd": 1.0,
        "confirmation_source_sha256": digest, "confirm_winner_label": "r1.p0", "confirmed_status": "ok",
        "confirmed_values": [1.30], "confirmed_speedup": 1.30, "candidate_us": None,
    }))
    (directory / "confirm_report.json").write_text(json.dumps({
        "status": "ok", "speedup": 1.30, "protocol_version": "3", "target_id": "apple-m4-max",
        "benchmark": {
            "noisy": False, "candidate": {"cv": 0.01, "reps": 15},
            "baselines": {"mlx": {"cv": 0.01, "reps": 15}}, "best_baseline": "mlx",
            "confirmation": {
                "kernel_sha256": digest, "winner_label": "r1.p0", "confirmed_values": [1.30],
                "speedup": 1.30, "candidate_us": None,
            },
        },
    }))
    index.sync_known()
    # 待评测不可用只有索引侧的状态（outbox 计数），摄取之后补上。
    for number in range(2):
        index.conn.execute(
            "INSERT INTO run_pending_eval (run_id, pending, unavailable, requests_json)"
            " VALUES (?, 0, 1, '[]')"
            " ON CONFLICT(run_id) DO UPDATE SET unavailable = 1, pending = 0, requests_json = '[]'",
            (f"bad/eval{number}",),
        )
    index.conn.commit()
    # 发布类坏状态：目录史里 bad/publication0 → bad/publication1 是回退；
    # bad/publication1 的确认报告没有 confirmation 块 → 当前不发布（publication_lost）。
    catalog = tmp_path / "kernels" / "apple-m4-max" / "relu" / "fp16" / "paper"
    catalog.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(b"kernel void k() {}\n").hexdigest()
    for number, speedup, confirmed_at in ((0, 1.30, 100.0), (1, 1.10, 200.0)):
        with (catalog / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "target_id": "apple-m4-max", "task": "relu", "dtype": "fp16", "size": "paper",
                "task_fingerprint": "fp", "calibration_sha256": "cal",
                "run_id": f"publication{number}", "winner_label": "r1.p0", "source_sha256": digest,
                "confirmed_speedup": speedup, "confirmed_values": [speedup],
                "candidate_us": 700.0, "baseline_us": {"mlx": 800.0}, "updated_at": confirmed_at,
            }) + "\n")
    index._sync_kernels()
    return runs, index


def test_ten_bad_states_are_locatable_in_three_calls_and_live_counts_are_exact(tmp_path, monkeypatch):
    """@brief 验收（plan §6.4 第 3 条）：注入 10 个坏状态，每一项都能在 ≤3 次调用内定位到 run。

    同时给出精确的 live 计数（runs 来自索引、evals/gpu 直接反映底层函数的返回值），以及
    告警类型的 label 约定：会话失败类带候选 label，预算类没有（不编造）。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _bad_state_runs(tmp_path, monkeypatch)
    monkeypatch.setattr(detail, "live_evals", lambda: [{"pid": 1}, {"pid": 2}])
    monkeypatch.setattr(detail, "gpu_lock_counts", lambda: {"holders": 2, "waiters": 1})

    doc = query.patrol()  # 第 1 次调用
    assert doc["live"] == {"runs": 3, "evals": 2, "gpu": {"holders": 2, "waiters": 1}}
    kinds = [alert["kind"] for alert in doc["alerts"]]
    assert kinds.count("budget_over") == 3
    assert kinds.count("session_failure") == 3
    assert kinds.count("eval_unavailable") == 2
    assert kinds.count("confirmation_regressed") == 1
    assert kinds.count("publication_lost") == 1
    assert len(doc["alerts"]) == 10
    assert doc["truncated"]["alerts"] is False

    def locate(alert: dict) -> str:
        """@brief 用最多 2 次公开调用把告警定位回它的 run。

        @param alert patrol 给的一条告警。
        @return 定位到的 run id。
        """
        if alert["kind"].startswith("budget_"):
            flagged = query.budget_report()["flagged"]  # 第 2 次调用
            match = [row for row in flagged if row["id"] == alert["run"]]
            assert match, f"预算报告里找不到 {alert['run']}"
            assert query.run_summary(alert["run"]) is not None  # 第 3 次调用
            assert "label" not in alert and "sig" not in alert
            return alert["run"]
        if alert["kind"] == "session_failure":
            # 会话失败要走 sessions 读模型：这类 run 往往根本没有候选，failures 会是空的。
            assert retrieval.failures(run=alert["run"])["failures"] == [], "夹具必须证明没有候选"
            sessions = retrieval.sessions(alert["run"])["sessions"]  # 第 2 次调用
            failing = [
                row for row in sessions
                if row["outcome"] in {"agent_failed", "provider_failed", "agent_no_submission"}
                and row["candidate_label"] == alert["label"]
            ]
            assert failing, f"{alert['run']}/{alert['label']} 在 sessions 里找不到失败会话"
            assert failing[0]["failure_class"] == "turn_failed"  # 第 3 次调用可读的细节
            assert failing[0]["diagnostics_path"].endswith("agent_diagnostics/r1.json")
            return alert["run"]
        if alert["kind"] == "eval_unavailable":
            summary = query.run_summary(alert["run"])  # 第 2 次调用
            assert summary["pending_evaluation"]["count"] >= 1
            assert "label" not in alert
            return alert["run"]
        assert alert["kind"] in {"confirmation_regressed", "publication_lost", "catalog_mismatch"}
        from observe import publications

        rows = publications.publications(task="relu")["rows"]  # 第 2 次调用
        match = [row for row in rows if row["run_id"] == alert["run"]]
        assert match, f"发布史里找不到 {alert['run']}"
        row = match[0]
        if alert["kind"] == "confirmation_regressed":
            assert row["delta_vs_prev"] is not None and row["delta_vs_prev"] < 0
        elif alert["kind"] == "publication_lost":
            assert row["current_publication"] != "confirmed"
            assert row["publication_reason"] == alert["reason"]  # 第 3 次调用可读到的原因码
        return alert["run"]

    for alert in doc["alerts"]:
        assert locate(alert) == alert["run"]
    assert {alert["run"] for alert in doc["alerts"]} == {
        *(f"bad/over{number}" for number in range(3)),
        *(f"bad/session{number}" for number in range(3)),
        *(f"bad/eval{number}" for number in range(2)),
        # 回退与丢失都在 bad/publication1 上；publication0 是回退的起点且已确认。
        "bad/publication1",
    }


def test_patrol_stays_under_the_budget_with_a_two_thousand_character_run_id(tmp_path, monkeypatch):
    """@brief 长 id 不能绕过 1.5KB：单条告警的 run id 就有 2000 字符时，丢弃它并如实置 truncated。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, index = _index(tmp_path, monkeypatch)
    long_id = "bad/" + "x" * 2000
    _run_row(index, long_id, spent=5.0, granted=1.0)
    index.conn.commit()

    doc = query.patrol()
    assert _bytes(doc) <= 1536, f"长 id 时 patrol {_bytes(doc)}B 超过 1,536B"
    assert doc["alerts"] == [], "装不下时只能丢掉这条告警"
    assert doc["truncated"]["alerts"] is True


def _failing_session_run(runs, name: str, *, transcript: str) -> Path:
    """@brief 造一个只有转录的 run（没有候选、没有确认），用于会话失败与删除回归。

    @param runs runs 根目录。
    @param name run id。
    @param transcript transcript 内容。
    @return run 目录。
    """
    directory = runs / name
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "phase": "done", "started_at": 1.0, "updated_at": 2.0,
    }))
    (directory / "result.json").write_text(json.dumps({"task": "relu", "stop_reason": "agent_failure"}))
    (directory / "transcript.jsonl").write_text(transcript, encoding="utf-8")
    return directory


def _agent_failure_line() -> str:
    """@brief 一条 agent_failed 转录事件（带 diagnostics 与失败类别）。

    @return JSONL 行。
    """
    return json.dumps({
        "event": "agent", "label": "r1.p0", "candidate_label": "r1.p0", "attempt": 0,
        "committee": "first", "provider": "deepseek", "model": "deepseek-flash",
        "outcome": "agent_failed", "cost_usd": 0.1,
        "runtime": {"failure_class": "turn_failed", "failed_turns": 2,
                    "last_terminal_kind": "failed", "last_error": "boom"},
        "diagnostics_path": "agent_diagnostics/r1.json",
    }) + "\n"


def test_a_physically_deleted_run_leaves_no_session_alert_and_no_projection_meta(tmp_path, monkeypatch):
    """@brief 评审回归：物理删除后 sessions 行还在，但 patrol 不能再为它发告警，per-run meta 也要清掉。

    旧实现直接查全部 ``sessions``，已删除的 run 会得到一条无法定位的 ``session_failure``（还会挤掉
    上限）；``_mark_missing`` 也没有清理 ``kw_src``/``reconcile_fp``。这里覆盖物理删除与同 ID 重建。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    import shutil

    runs, index = _index(tmp_path, monkeypatch)
    directory = _failing_session_run(runs, "gone/run", transcript=_agent_failure_line())
    index.sync_known()
    assert [alert["run"] for alert in query.patrol()["alerts"] if alert["kind"] == "session_failure"] == ["gone/run"]
    assert index.conn.execute("SELECT COUNT(*) FROM meta WHERE key = 'kw_src:gone/run'").fetchone()[0] == 1
    assert index.conn.execute("SELECT COUNT(*) FROM meta WHERE key = 'sessions_fp:gone/run'").fetchone()[0] == 1

    shutil.rmtree(directory)
    index.sync_known()
    assert index.conn.execute("SELECT status FROM runs WHERE run_id = 'gone/run'").fetchone()[0] == "missing"
    assert query.run_summary("gone/run") is None
    assert all(alert["run"] != "gone/run" for alert in query.patrol()["alerts"])
    for key in ("kw_src:gone/run", "sessions_fp:gone/run", "sessions_bytes:gone/run"):
        assert index.conn.execute("SELECT COUNT(*) FROM meta WHERE key = ?", (key,)).fetchone()[0] == 0, key

    # 同 ID 重建：新的纯 chat 转录，窄投影不能沿用旧的会话/来源状态。
    rebuilt = _failing_session_run(runs, "gone/run", transcript=json.dumps({"event": "chat", "label": "r1.p0"}) + "\n")
    index.sync_known()
    assert query.run_summary("gone/run")["kernel_writer"] == "chat"
    assert index.conn.execute("SELECT COUNT(*) FROM meta WHERE key = 'kw_src:gone/run'").fetchone()[0] == 1
    assert query.patrol()["failing"]["total"] == 0
    assert rebuilt.is_dir()


def test_a_failure_session_without_class_or_error_still_alerts(tmp_path, monkeypatch):
    """@brief 评审回归：``agent_failed`` 但 ``failure_class``/``error`` 都为空时也必须告警。

    ``session_failure_signature`` 在这种输入上返回 NULL（签名只用于聚类），所以告警的存在条件必须
    看 outcome。真实语料里 6 条 ``agent_failed`` 的 ``failure_sig`` 全是 NULL；旧谓词
    ``failure_sig IS NOT NULL`` 会把它们整类漏掉。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    _failing_session_run(runs, "bare/run", transcript=json.dumps({
        "event": "agent", "outcome": "agent_failed", "label": "r1.p0",
    }) + "\n")
    index.sync_known()
    # §11.4 选 (a)：冷建不投影 sessions；这里是“第一次读会话”的懒投影。
    index.ensure_sessions(runs / "bare/run")

    stored = [
        dict(row) for row in index.conn.execute(
            "SELECT outcome, failure_sig, candidate_label FROM sessions WHERE run_id = 'bare/run'"
        )
    ]
    assert len(stored) == 1, "会话必须来自真实摄取"
    assert stored[0]["outcome"] == "agent_failed" and stored[0]["failure_sig"] is None
    assert stored[0]["candidate_label"] == "r1.p0"

    doc = query.patrol()
    alerts = [alert for alert in doc["alerts"] if alert["run"] == "bare/run"]
    assert alerts, "没有签名的失败会话也必须告警"
    assert alerts[0]["kind"] == "session_failure" and alerts[0]["label"] == "r1.p0"
    assert alerts[0]["n"] == 1

    # ≤3 次调用定位：patrol → sessions（这类 run 没有候选，failures 是空的）。
    assert retrieval.failures(run="bare/run")["failures"] == []
    sessions = retrieval.sessions("bare/run")["sessions"]
    failing = [row for row in sessions if row["outcome"] == "agent_failed"]
    assert failing and failing[0]["candidate_label"] == "r1.p0"


def test_drop_archived_removes_the_session_alert(tmp_path, monkeypatch):
    """@brief 评审回归：``drop_archived`` 之后也不能再为那条会话发告警。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    _failing_session_run(runs, "archived/run", transcript=_agent_failure_line())
    index.sync_known()
    assert any(alert["run"] == "archived/run" for alert in query.patrol()["alerts"])

    index.drop_archived(["archived/run"], [])
    # drop_archived 只删索引行（目录由 gc 删）：索引里没有这条 run 了，告警也必须随之消失。
    assert index.conn.execute("SELECT COUNT(*) FROM runs WHERE run_id = 'archived/run'").fetchone()[0] == 0
    assert all(alert["run"] != "archived/run" for alert in query.patrol()["alerts"])
    assert index.conn.execute("SELECT COUNT(*) FROM meta WHERE key = 'kw_src:archived/run'").fetchone()[0] == 0


def test_patrol_never_ingests_and_never_decodes_row_json(tmp_path, monkeypatch):
    """@brief 计数器：``_ingest`` 0 次、``row_json`` 解码 0 次、``pending_evaluation()`` 0 次。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    directory = runs / "relu" / "live"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({"task": "relu", "status": "running"}))
    (directory / "result.json").write_text(json.dumps({"task": "relu", "stop_reason": "max_rounds"}))
    index.sync_known()

    def forbidden_ingest(self, *args, **kwargs):
        raise AssertionError("patrol must not ingest")

    monkeypatch.setattr(Index, "_ingest", forbidden_ingest)
    def forbidden_pending(path):
        raise AssertionError("patrol must not call pending_evaluation()")

    monkeypatch.setattr(detail, "pending_evaluation", forbidden_pending)

    counted = {"loads": 0}
    real_json = json

    class Counting:
        """最小的 ``json`` 替身：只数 ``loads``，其余原样转发。"""

        @staticmethod
        def loads(*args, **kwargs):
            counted["loads"] += 1
            return real_json.loads(*args, **kwargs)

        @staticmethod
        def dumps(*args, **kwargs):
            return real_json.dumps(*args, **kwargs)

    monkeypatch.setattr(query, "json", Counting)

    doc = query.patrol()
    assert counted["loads"] == 0, "patrol 不该解码任何 row_json"
    # 精确计数：live.runs 就是索引里 status='running' 的行数（不做 0/1 的模糊断言）。
    running = index.conn.execute("SELECT COUNT(*) FROM runs WHERE status = 'running'").fetchone()[0]
    assert doc["live"]["runs"] == running
    assert doc["index_revision"] >= 0


def test_every_alert_locates_its_run_and_failure_alerts_carry_the_label(tmp_path, monkeypatch):
    """@brief 每条告警自带 run；会话失败类还带 label，可用 ``query failures --run/--label`` 复核。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    directory = runs / "relu" / "failed"
    directory.mkdir(parents=True)
    common = {"task": "relu", "dtype": "fp16", "started_at": 1.0}
    (directory / "state.json").write_text(json.dumps({**common, "phase": "done", "updated_at": 2.0}))
    (directory / "result.json").write_text(json.dumps({**common, "stop_reason": "agent_failure"}))
    folder = directory / "candidates" / "000_r1.p0"
    folder.mkdir(parents=True)
    (folder / "kernel.metal").write_text("kernel void k() {}\n")
    (folder / "report.json").write_text(json.dumps({
        "status": "compile_error", "stage": "compile", "message": "Metal compiler rejected the kernel",
    }))
    index.sync_known()
    run_id = "relu/failed"
    index.conn.execute(
        "INSERT INTO sessions (run_id, seq, label, candidate_label, attempt, committee, provider, model,"
        " outcome, failure_class, error, failure_sig) VALUES (?, 1, 'r1.p0', 'r1.p0', 0, 'first',"
        " 'deepseek', 'deepseek-flash', 'agent_failed', 'turn_failed', 'boom', '0123456789abcdef')",
        (run_id,),
    )
    index.conn.execute(
        "UPDATE runs SET spent_usd = 5.0, budget_usd = 1.0, granted_budget_usd = 1.0 WHERE run_id = ?",
        (run_id,),
    )
    index.conn.commit()

    doc = query.patrol()
    kinds = {alert["kind"] for alert in doc["alerts"]}
    assert "session_failure" in kinds and "budget_over" in kinds
    for alert in doc["alerts"]:
        assert alert["run"]
        if alert["kind"] == "session_failure":
            assert alert["label"] == "r1.p0"
            # ≤3 次调用定位：patrol → failures --run/--label → 报告路径。
            found = retrieval.failures(run=alert["run"], label=alert["label"])
            assert found["failures"] and found["failures"][0]["report_path"].endswith("report.json")
        else:
            assert "label" not in alert

    assert cli.main(["query", "patrol"]) == 0


def test_patrol_matches_the_budget_banner_and_reports_real_totals(tmp_path, monkeypatch, capsys):
    """@brief 预算告警与横幅同源（含“已加预算撤销耗尽”），``failing.total`` 来自成员表。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @param capsys pytest 的输出捕获器。
    """
    _runs, index = _index(tmp_path, monkeypatch)
    _run_row(index, "over/one", spent=3.0, granted=1.0)
    _run_row(index, "revoked/one", spent=0.5, granted=11.0, stop="budget_exhausted")
    _run_row(index, "exhausted/one", spent=0.9, granted=1.0, stop="  budget_exhausted\n")
    index.conn.commit()

    doc = query.patrol()
    assert doc["truncated"]["best"] is False and doc["best"] == []
    kinds = [(alert["kind"], alert["run"]) for alert in doc["alerts"]]
    assert ("budget_over", "over/one") in kinds
    # 已记录的加预算尚未被运行采纳：耗尽标记撤销，不再报预算告警。
    assert all(run != "revoked/one" for _kind, run in kinds)
    # 带前后空白的停止原因与 ``_grade`` 的 strip 语义一致：算耗尽。
    assert ("budget_exhausted", "exhausted/one") in kinds

    assert cli.main(["query", "patrol"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == PATROL_KEYS and _bytes(printed) <= 1536
