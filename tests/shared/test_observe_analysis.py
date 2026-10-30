"""§8.4 分析面板：落差 / 每美元 / 委员会有效性 / 噪声。

数据只来自索引窄列与已确认运行的 ``confirm_report.json``；``published`` 必须等于
``conclusion_of_run`` 的结果（不能用 ``result.confirmed_speedup`` 顶替），跨协议纪元要拒绝。
"""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import pytest

from experiments.common import conclusion_of_run
from experiments.manifest import attach, ensure
from observe import analysis, query
from observe.index import Index

_SOURCE = "kernel void analysis_fixture() {}\n"


def _index(tmp_path, monkeypatch):
    """@brief 临时 runs 根 + 索引（关掉读时 tick，catalog 指向空目录）。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @return ``(runs 根, Index)``。
    """
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    monkeypatch.setenv("OBSERVE_REFRESH_ON_READ", "0")
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(tmp_path / "kernels"))
    return runs, Index(tmp_path / "observe.sqlite")


def _published_run(runs: Path, name: str, *, task: str = "conv2d", speedup: float = 1.25,
                   search_best: float = 1.5, spent: float = 2.0, protocol: str = "3",
                   dtype: str = "fp16", size: str = "paper") -> Path:
    """@brief 造一个真的会发布的已确认 run（确认门绑定内核、各批一致）。

    @param runs runs 根目录。
    @param name run id。
    @param task 任务名。
    @param speedup 确认加速比（各批中位数）。
    @param search_best 搜索最优（写进 result.best_speedup，索引会投影成 search_best_speedup）。
    @param spent 已花费。
    @param protocol 协议版本。
    @param dtype 数据类型。
    @param size 尺寸标签。
    @return run 目录。
    """
    from tcenv.schema import PROTOCOL_VERSION

    directory = runs / name
    directory.mkdir(parents=True)
    digest = hashlib.sha256(_SOURCE.encode("utf-8")).hexdigest()
    values = [speedup]
    (directory / "best.metal").write_text(_SOURCE, encoding="utf-8")
    folder = directory / "candidates" / "000_r1.p0"
    folder.mkdir(parents=True)
    (folder / "kernel.metal").write_text(_SOURCE, encoding="utf-8")
    (folder / "report.json").write_text(json.dumps({
        "status": "ok", "stage": "bench", "speedup": speedup, "candidate_us": 700.0,
        "benchmark": {"noisy": False, "candidate": {"cv": 0.01, "reps": 15}},
    }), encoding="utf-8")
    (directory / "state.json").write_text(json.dumps({
        "task": task, "dtype": dtype, "phase": "done", "started_at": 10.0, "updated_at": 20.0,
        "spent_usd": spent, "budget_usd": 10.0, "best_speedup": search_best,
    }), encoding="utf-8")
    from experiments.common import synthetic_confirmation_documents

    # Both sides of the fixture say the same protocol, so the frozen expected and the confirm
    # report agree: the "epoch" these tests group by is a real published epoch, not a mismatch.
    result, report = synthetic_confirmation_documents(
        task=task, dtype=dtype, size=size, target_id="apple-m4-max",
        speedup=speedup, label="r1.p0", kernel_sha256=digest, values=values,
        protocol_version=PROTOCOL_VERSION if protocol == "flame_chase_1" else protocol,
    )
    result.update({
        "stop_reason": "max_rounds", "spent_usd": spent, "budget_usd": 10.0,
        "best_speedup": search_best, "protocol_version": protocol,
    })
    (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (directory / "confirm_report.json").write_text(json.dumps(report), encoding="utf-8")
    (directory / "run_config.json").write_text(json.dumps({
        "task": task, "dtype": dtype, "size": size,
        "target": {"id": "apple-m4-max", "backend": "metal", "backend_version": "v1"},
        "search": {"budget_usd": 10.0},
    }), encoding="utf-8")
    return directory


def _unpublished_run(runs: Path, name: str, *, task: str = "conv2d", speedup: float = 9.9,
                     search_best: float = 9.9) -> Path:
    """@brief 造一个确认报告是旧格式（没有 confirmation 块）的 run：有 confirmed_speedup 但不发布。

    @param runs runs 根目录。
    @param name run id。
    @param task 任务名。
    @param speedup 写进 result.confirmed_speedup（不该被落差面板使用）。
    @param search_best 搜索最优。
    @return run 目录。
    """
    directory = runs / name
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": task, "dtype": "fp16", "phase": "done", "started_at": 10.0, "updated_at": 20.0,
        "spent_usd": 1.0, "budget_usd": 10.0, "best_speedup": search_best,
    }), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "task": task, "dtype": "fp16", "size": "paper", "stop_reason": "max_rounds", "spent_usd": 1.0,
        "budget_usd": 10.0, "best_speedup": search_best, "target_id": "apple-m4-max",
        "protocol_version": "3", "confirmed_speedup": speedup, "confirmed_values": [speedup],
        "confirmed_status": "ok", "confirmation_source_sha256": "a" * 64, "confirm_winner_label": "r1.p0",
    }), encoding="utf-8")
    (directory / "confirm_report.json").write_text(json.dumps({"status": "ok", "speedup": speedup}))
    return directory


def test_gap_matches_the_publication_gate_and_counts_unpublished_separately(tmp_path, monkeypatch):
    """@brief 落差来自 ``conclusion_of_run``；有 confirmed_speedup 但当前不发布的 run 只进计数。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    published = _published_run(runs, "conv2d/ok", speedup=1.648635, search_best=1.695914)
    hidden = _unpublished_run(runs, "conv2d/stale", speedup=1.738183, search_best=1.738183)
    index.sync_known()

    result = json.loads((published / "result.json").read_text())
    expected = conclusion_of_run(published, result)
    assert expected["publication"] == "confirmed" and expected["speedup"] == pytest.approx(1.648635)
    assert json.loads((hidden / "result.json").read_text())["confirmed_speedup"] == pytest.approx(1.738183)

    doc = analysis.gap(protocol_epoch="3")
    assert len(doc["rows"]) == 1, "只有当前发布的那一场进落差"
    row = doc["rows"][0]
    assert row["published"] == pytest.approx(expected["speedup"])
    assert row["search_best"] == pytest.approx(1.695914)
    assert row["signed_delta_pct"] == pytest.approx(-2.7878, abs=1e-3)
    assert row["n_batches"] == 1 and row["cv"] == pytest.approx(0.01)
    assert doc["unpublished"] == 1, "旧格式确认报告只进未发布计数"
    assert all(item["run_id"] != "conv2d/stale" for item in doc["rows"])
    assert doc["groups"][0]["insufficient"] is True and doc["groups"][0]["median_signed_delta_pct"] is None


def test_gap_refuses_to_mix_protocol_epochs_and_stamp_short_circuits(tmp_path, monkeypatch):
    """@brief 跨纪元必须显式选择（400）；同 stamp 只回 ``same``。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    _published_run(runs, "conv2d/a", speedup=1.2, search_best=1.3, protocol="3")
    _published_run(runs, "conv2d/b", speedup=1.1, search_best=1.2, protocol="flame_chase_1")
    index.sync_known()

    with pytest.raises(analysis.EpochRequired) as caught:
        analysis.gap()
    assert sorted(caught.value.epochs) == ["2", "3"]

    doc = analysis.gap(protocol_epoch="2")
    assert [row["run_id"] for row in doc["rows"]] == ["conv2d/b"]
    assert doc["protocol_epoch"] == "2" and doc["epochs"] == ["2", "3"]
    again = analysis.gap(protocol_epoch="2", stamp=doc["stamp"])
    assert again == {"same": True, "stamp": doc["stamp"]}


def test_usd_metrics_exclude_published_at_or_below_one_and_use_spent_max(tmp_path, monkeypatch):
    """@brief 每美元替代口径：published ≤ 1 → null + excluded；分母用单调的 spent_max。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    good = _published_run(runs, "conv2d/good", speedup=1.5, search_best=1.6, spent=3.0)
    flat = _published_run(runs, "conv2d/flat", speedup=1.0, search_best=1.0, spent=4.0)
    index.sync_known()
    doc = analysis.gap(protocol_epoch="3")
    by_id = {row["run_id"]: row for row in doc["rows"]}
    assert by_id["conv2d/good"]["usd_per_100pct_gain"] == pytest.approx(6.0)
    assert by_id["conv2d/good"]["usd_per_confirmed"] == pytest.approx(3.0)
    assert by_id["conv2d/flat"]["usd_per_100pct_gain"] is None
    assert doc["excluded"] == 1

    # spent_max 只升不降：把 state/result 的花费改成更低值，分母仍是历史最大。
    for name in ("state.json", "result.json"):
        path = good / name
        body = json.loads(path.read_text())
        body["spent_usd"] = 1.0
        path.write_text(json.dumps(body), encoding="utf-8")
    index.sync_known()
    row = next(item for item in analysis.gap(protocol_epoch="3")["rows"] if item["run_id"] == "conv2d/good")
    assert row["usd_per_confirmed"] == pytest.approx(3.0), "分母必须是 spent_max_usd"
    assert row["usd_per_100pct_gain"] == pytest.approx(6.0)


def test_gap_groups_only_give_medians_with_five_samples(tmp_path, monkeypatch):
    """@brief ``n<5`` 只给点；第 5 场之后才给中位数。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    for number in range(5):
        _published_run(runs, f"conv2d/{number}", speedup=1.1 + number * 0.1,
                       search_best=1.5 + number * 0.1, spent=1.0)
    index.sync_known()
    doc = analysis.gap(protocol_epoch="3")
    group = doc["groups"][0]
    assert group["n"] == 5 and group["insufficient"] is False
    assert group["median_signed_delta_pct"] is not None
    assert group["median_usd_per_100pct_gain"] is not None

    # 删掉一场后回到 4 个样本：中位数必须消失。
    analysis_index = index
    analysis_index.conn.execute("UPDATE runs SET status = 'missing' WHERE run_id = 'conv2d/4'")
    analysis_index.conn.commit()
    doc = analysis.gap(protocol_epoch="3")
    group = doc["groups"][0]
    assert group["n"] == 4 and group["insufficient"] is True
    assert group["median_signed_delta_pct"] is None and group["median_usd_per_100pct_gain"] is None


def _session(index, run_id: str, label: str, *, committee: str = "first", provider: str = "deepseek",
             model: str = "deepseek-flash", cost: float | None = 0.5, outcome: str = "evaluated") -> None:
    """@brief 插一条 sessions 行。

    @param index 索引。
    @param run_id 运行 id。
    @param label 候选标签。
    @param committee 委员会。
    @param provider provider。
    @param model 模型。
    @param cost 花费；None 表示没有成本。
    @param outcome 结局。
    """
    index.conn.execute(
        "INSERT INTO sessions (run_id, seq, label, candidate_label, attempt, committee, provider, model,"
        " outcome, cost_usd) VALUES (?, (SELECT COALESCE(MAX(seq), 0) + 1 FROM sessions), ?, ?, 0, ?, ?, ?, ?, ?)",
        (run_id, label, label, committee, provider, model, outcome, cost),
    )


def _candidate(index, run_id: str, label: str, *, status: str = "ok", speedup: float = 1.2) -> None:
    """@brief 插一条 candidates 行（committee 面板按 label join）。

    @param index 索引。
    @param run_id 运行 id。
    @param label 候选标签。
    @param status 状态。
    @param speedup 加速比。
    """
    index.conn.execute(
        "INSERT OR REPLACE INTO candidates (run_id, idx, label, status, stage, speedup)"
        " VALUES (?, (SELECT COALESCE(MAX(idx), -1) + 1 FROM candidates WHERE run_id = ?), ?, ?, 'bench', ?)",
        (run_id, run_id, label, status, speedup),
    )


def test_committees_count_sessions_not_runs_and_keep_an_empty_bucket(tmp_path, monkeypatch):
    """@brief 委员会面板：单位是 session；按 task+纪元分桶；空委员会单列；成本是 SUM 再除的下界。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    for name, task in (("a/x", "relu"), ("b/y", "gemm")):
        directory = runs / name
        directory.mkdir(parents=True)
        (directory / "state.json").write_text(json.dumps({"task": task, "dtype": "fp16"}))
    index.sync_known()
    _candidate(index, "a/x", "r1.p0", speedup=1.2)
    _candidate(index, "a/x", "r1.p1", status="incorrect", speedup=None)
    for number in range(3):
        _session(index, "a/x", f"r1.p{number}", cost=1.0 if number < 2 else None)
    _session(index, "a/x", "r1.p0", committee="", cost=None)
    _session(index, "b/y", "r1.p0", committee="second", provider="moonshot", model="kimi-k3", cost=2.0)
    index.conn.execute("UPDATE candidates SET speedup = 1.2 WHERE run_id = 'a/x' AND label = 'r1.p0'")
    index.conn.commit()

    doc = analysis.committees(protocol_epoch="legacy")
    buckets = {(row["task"], row["protocol_epoch"], row["committee"]): row for row in doc["buckets"]}
    first = buckets[("relu", "legacy", "first")]
    assert first["sessions"] == 3 and first["runs"] == 1, "单位是 session，不是场"
    assert first["cost_usd"] == pytest.approx(2.0)
    assert first["sessions_with_cost"] == 2 and first["cost_is_lower_bound"] is True
    # 与 model_stats() 同一口径：SUM(cost) / 全部 session（不是除以“有成本的会话数”）。
    assert first["cost_per_session_usd"] == pytest.approx(2.0 / 3.0)
    models = query.model_stats()["models"]
    assert models and models[0]["cost_per_session_usd"] == pytest.approx(first["cost_per_session_usd"])
    # 只有 r1.p0/r1.p1 有对应候选（r1.p2 没有），join 率本身就是有效性的一部分。
    assert first["joined_sessions"] == 2 and first["ok_share"] == pytest.approx(0.5)
    assert first["joined_sessions"] < first["sessions"]
    assert first["insufficient"] is True and first["sessions"] < analysis.SMALL_SAMPLE

    empty = buckets[("relu", "legacy", "")]
    assert empty["sessions"] == 1 and empty["cost_usd"] == 0.0 and empty["joined_sessions"] == 1
    assert doc["totals"]["empty_committee_sessions"] == 1
    second = buckets[("gemm", "legacy", "second")]
    assert second["sessions"] == 1 and second["runs"] == 1
    # 不同任务绝不混进同一个桶（禁止跨题平均）。
    assert {row["task"] for row in doc["buckets"]} == {"relu", "gemm"}
    # 逐桶判定：全局 insufficient 只是“有没有不足桶”，并点名它们（混合桶时不会消失）。
    assert doc["insufficient"] is True
    assert set(doc["insufficient_buckets"]) == {row["id"] for row in doc["buckets"]}


def _candidate_file(directory: Path, index: int, *, cv, reps, cache_hit) -> None:
    """@brief 写一颗候选报告（噪声面板读索引列，这里经真实摄取写入）。

    @param directory 运行目录。
    @param index 候选序号。
    @param cv 噪声 cv；None 表示没有。
    @param reps 重复次数；None 表示没有。
    @param cache_hit 缓存命中；None 表示缺失。
    """
    # index=0 覆盖确认胜者的报告（label r1.p0）；其余用独立 label，避免“同名 win 者不唯一”。
    label = "r1.p0" if index == 0 else f"r{index}.p0"
    folder = directory / "candidates" / f"{index:03d}_{label}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "kernel.metal").write_text(_SOURCE, encoding="utf-8")
    # 噪声面板只从当前（schema-4）报告取 cv/reps/cache_hit。
    report = {"schema_version": 4, "status": "ok", "stage": "bench", "speedup": 1.0}
    if cv is not None or reps is not None:
        report["benchmark"] = {"candidate": {key: value for key, value in (("cv", cv), ("reps", reps)) if value is not None}}
    if cache_hit is not None:
        report["cache_hit"] = cache_hit
    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_noise_quantiles_match_a_direct_scan_and_cache_hit_coverage_is_explicit(tmp_path, monkeypatch):
    """@brief 噪声面板：p10/p50/p90 与直接扫候选报告一致；cache_hit 缺失不算 false。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    directory = runs / "relu" / "noisy"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({"task": "relu", "dtype": "fp16", "phase": "done"}))
    (directory / "result.json").write_text(json.dumps({"task": "relu", "stop_reason": "max_rounds"}))
    cvs = [round(0.001 * number, 6) for number in range(1, 11)]
    for number, cv in enumerate(cvs):
        _candidate_file(directory, number, cv=cv, reps=15, cache_hit=False if number < 3 else None)
    _candidate_file(directory, 10, cv=None, reps=None, cache_hit=True)
    index.sync_known()

    doc = analysis.noise(task="relu", protocol_epoch="legacy")
    ordered = sorted(cvs)
    assert doc["cv"]["n"] == len(cvs)
    assert doc["cv"]["p10"] == pytest.approx(ordered[0])
    assert doc["cv"]["p50"] == pytest.approx(ordered[4])
    assert doc["cv"]["p90"] == pytest.approx(ordered[8])
    assert doc["cv"]["min"] == pytest.approx(ordered[0]) and doc["cv"]["max"] == pytest.approx(ordered[-1])
    assert doc["cv"]["coverage"] == round(len(cvs) / 11, 4)
    assert doc["reps"] == [{"reps": 15, "n": len(cvs)}]
    assert doc["cache_hit"] == {
        "total": 11, "known": 4, "true": 1, "false": 3, "coverage": round(4 / 11, 4),
    }, "缺失的 cache_hit 不能当成 false"
    assert doc["insufficient"] is False


def test_rank_correlation_waits_for_twenty_samples(tmp_path, monkeypatch):
    """@brief 落差与噪声的秩相关只在 n≥20 时给。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    for number in range(20):
        cv = 0.001 * (number + 1)
        _published_run(runs, f"conv2d/{number}", speedup=1.1 + number * 0.01,
                       search_best=1.5 + number * 0.01, spent=1.0)
        # 覆盖确认报告里的 cv，让落差行带上与 published 同向的噪声。
        report = runs / f"conv2d/{number}" / "confirm_report.json"
        body = json.loads(report.read_text())
        body["benchmark"]["candidate"]["cv"] = cv
        report.write_text(json.dumps(body), encoding="utf-8")
    index.sync_known()
    doc = analysis.gap(protocol_epoch="3")
    assert doc["rank_correlation"]["n"] == 20 and doc["rank_correlation"]["insufficient"] is False
    assert doc["rank_correlation"]["spearman"] == pytest.approx(1.0), "cv 与 published 同向时秩相关为 1"

    index.conn.execute("UPDATE runs SET status = 'missing' WHERE run_id = 'conv2d/19'")
    index.conn.commit()
    doc = analysis.gap(protocol_epoch="3")
    assert doc["rank_correlation"]["n"] == 19 and doc["rank_correlation"]["insufficient"] is True
    assert doc["rank_correlation"]["spearman"] is None


def test_every_analysis_panel_caps_its_rows_and_stays_under_eight_kilobytes(tmp_path, monkeypatch):
    """@brief 验收：行数超过上限时按 ``truncated`` 截断，三个面板仍各 ≤8KB。

    面板 ≤8KB 靠的是字节预算（先截展示文本，再从尾部丢行/桶）与统计口径分离：行/桶上限只限制
    展示，``published_total`` 等统计仍基于完整过滤集。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    for number in range(40):
        _published_run(runs, f"conv2d/{number:02d}", speedup=1.05 + number * 0.01,
                       search_best=1.2 + number * 0.01, spent=1.0)
    index.sync_known()
    for number in range(50):
        _session(index, f"conv2d/{number % 40:02d}", "r1.p0", model=f"model-{number:02d}")
    for number in range(60):
        _candidate_file(runs / "conv2d" / "00", 100 + number, cv=0.001 * (number + 1), reps=15,
                        cache_hit=False if number % 3 else None)
    index.conn.commit()
    index.sync_known()

    documents = {
        "gap": analysis.gap(protocol_epoch="3"),
        "committees": analysis.committees(protocol_epoch="3"),
        "noise": analysis.noise(protocol_epoch="3"),
    }
    for name, document in documents.items():
        size = len(json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        assert size <= analysis.PANEL_BYTES, f"{name} 面板 {size}B 超过 8KB"
    assert documents["gap"]["published_total"] == 40, "统计基于完整过滤集，不是展示行数"
    assert len(documents["gap"]["rows"]) <= analysis.PANEL_ROWS
    assert documents["gap"]["truncated"]["rows"] is True
    assert len(documents["committees"]["buckets"]) <= analysis.PANEL_COLUMNS
    assert documents["committees"]["truncated"]["buckets"] is True


def test_the_analysis_view_renders_trends_and_the_small_sample_notice_in_node():
    """@brief UI 验收：``renderTrend`` 在 n<5 时出现“样本不足以判定”且不画线，≥5 时 polyline ≤200 点。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = Path("observe/static/index.html").read_text()

    def extract(name):
        start = page.index(f"function {name}(")
        # 参数列表里可能有 ``options = {}``：先配平圆括号，再找函数体的第一个 ``{``。
        paren = page.index("(", start)
        depth = 0
        cursor = paren
        while cursor < len(page):
            if page[cursor] == "(":
                depth += 1
            elif page[cursor] == ")":
                depth -= 1
                if depth == 0:
                    break
            cursor += 1
        brace = page.index("{", cursor)
        depth = 0
        for index in range(brace, len(page)):
            if page[index] == "{":
                depth += 1
            elif page[index] == "}":
                depth -= 1
                if depth == 0:
                    return page[start:index + 1]
        raise AssertionError(f"unclosed function {name}")

    script = "\n".join(extract(name) for name in ("esc", "preferredEpoch", "renderTrend")) + r'''
const four = renderTrend([{x: "a", y: -1}, {x: "b", y: -2}, {x: "c", y: null}, {x: "d", y: -3}], {label: "t", insufficient: true});
if (!four.includes("样本不足以判定")) throw new Error("small sample notice missing: " + four);
if (four.includes("<polyline")) throw new Error("n<5 must only draw points");
if ((four.match(/<circle/g) || []).length !== 3) throw new Error("expected 3 points");
const many = renderTrend(Array.from({length: 300}, (_, i) => ({x: "r" + i, y: i % 7})), {label: "t"});
if (!many.includes("<polyline")) throw new Error("large sample must draw a line");
const pairs = (many.match(/points="([^"]+)"/) || [, ""])[1].trim().split(/\s+/).filter(Boolean);
if (pairs.length !== 200) throw new Error("polyline must cap at 200 points, got " + pairs.length);
if (preferredEpoch(["legacy", "2", "10", "3"]) !== "10") throw new Error("preferredEpoch must pick the largest numeric epoch");
if (preferredEpoch([]) !== "") throw new Error("empty epochs must stay empty");
if (preferredEpoch(["legacy", "unknown"]) !== "legacy") throw new Error("no numeric epoch must fall back to the first");
console.log("ok");
'''
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_the_analysis_view_makes_at_most_four_requests_and_reuses_stamps_in_node():
    """@brief UI 验收：首次打开分析视图 ≤4 个请求（gap 混纪元 400 + 重试、committees、noise），
    之后再打开带 stamp 短路（3 个请求），n<5 的组显示“样本不足以判定”。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = Path("observe/static/index.html").read_text()

    def extract(name):
        marker = f"async function {name}(" if f"async function {name}(" in page else f"function {name}("
        start = page.index(marker)
        paren = page.index("(", start)
        depth = 0
        cursor = paren
        while cursor < len(page):
            if page[cursor] == "(":
                depth += 1
            elif page[cursor] == ")":
                depth -= 1
                if depth == 0:
                    break
            cursor += 1
        brace = page.index("{", cursor)
        depth = 0
        for index in range(brace, len(page)):
            if page[index] == "{":
                depth += 1
            elif page[index] == "}":
                depth -= 1
                if depth == 0:
                    return page[start:index + 1]
        raise AssertionError(f"unclosed function {name}")

    names = ("esc", "num", "pctText", "idText", "preferredEpoch", "renderTrend", "analysisFetch", "loadAnalysis")
    state_line = next(
        line.strip() for line in page.splitlines() if line.strip().startswith("const analysisState")
    )
    assert state_line.endswith("};"), f"unexpected analysisState declaration: {state_line}"
    script = state_line + "\n" + "\n".join(extract(name) for name in names) + r'''
const applied = {task: ""};
const hosts = {};
globalThis.document = { getElementById: (id) => (hosts[id] = hosts[id] || { innerHTML: "" }) };
const calls = [];
const gapDoc = { same: false, stamp: "g1", protocol_epoch: "3", epochs: ["2", "3"], published_total: 25,
  unpublished: 2, excluded: 1, truncated: { rows: true },
  rank_correlation: { n: 25, insufficient: false, spearman: 0.42 },
  rows: [{ task: "relu", run_id: "conv2d/one", dtype: "fp16", size: "paper", target_id: "apple-m4-max",
           protocol_epoch: "3", arm_id: "", search_best: 1.5, published: 1.2,
           signed_delta_pct: -20.0, cv: 0.0123, n_batches: 3, usd_per_100pct_gain: 5.0 }],
  groups: [{ task: "relu", dtype: "fp16", size: "paper", target_id: "apple-m4-max",
             protocol_epoch: "3", arm_id: "", insufficient: true }] };
const committeeDoc = { same: false, stamp: "c1", insufficient: true, protocol_epoch: "3",
  insufficient_buckets: ["relu/3/second/kimi-k3"],
  buckets: [
    { id: "relu/3/first/m", task: "relu", dtype: "fp16", protocol_epoch: "3", committee: "first", model: "m",
      sessions: 5, runs: 1, cost_usd: 1.0, cost_is_lower_bound: false, cost_per_session_usd: 0.2,
      ok_share: 0.5, insufficient: false },
    { id: "relu/3/second/kimi-k3", task: "relu", dtype: "fp16", protocol_epoch: "3", committee: "second",
      model: "kimi-k3", sessions: 1, runs: 1, cost_usd: 0.5, cost_is_lower_bound: false,
      cost_per_session_usd: 0.5, ok_share: null, insufficient: true }],
  totals: { sessions: 6, runs: 2, cost_usd: 1.5, empty_committee_sessions: 0 } };
const noiseDoc = { same: false, stamp: "n1", insufficient: false, protocol_epoch: "3",
  cv: { n: 6, p10: 0.1, p50: 0.2, p90: 0.3, coverage: 0.5 }, reps: [{ reps: 15, n: 6 }],
  cache_hit: { total: 6, known: 3, coverage: 0.5 } };
let sameMode = false;
globalThis.fetch = async (url) => {
  calls.push(url);
  if (url.startsWith("/api/v2/gap")) {
    if (!url.includes("protocol_epoch=")) {
      return { ok: false, json: async () => ({ error: "protocol_epoch required", epochs: ["2", "3"] }) };
    }
    return { ok: true, json: async () => (sameMode ? { same: true, stamp: "g1" } : gapDoc) };
  }
  if (url.startsWith("/api/v2/committees")) {
    return { ok: true, json: async () => (sameMode ? { same: true, stamp: "c1" } : committeeDoc) };
  }
  if (url.startsWith("/api/v2/noise")) {
    return { ok: true, json: async () => (sameMode ? { same: true, stamp: "n1" } : noiseDoc) };
  }
  throw new Error("unexpected url " + url);
};
loadAnalysis().then(() => {
  if (calls.length !== 4) throw new Error("first load must be <=4 requests, got " + calls.length + ": " + calls.join(" "));
  if (!calls[0].startsWith("/api/v2/gap")) throw new Error("gap must be first");
  if (!calls[1].includes("protocol_epoch=3")) throw new Error("retry must name the epoch: " + calls[1]);
  const gapHtml = hosts["analysis-gap"].innerHTML;
  if (!gapHtml.includes("样本不足以判定")) throw new Error("small-sample notice missing");
  if (!gapHtml.includes("替代口径")) throw new Error("usd alternative note missing");
  // 聚合口径：已发布总数是完整过滤集的 25，不是展示的 1 行；截断要明示。
  if (!gapHtml.includes("已发布 25 场")) throw new Error("published_total not shown: " + gapHtml);
  if (!gapHtml.includes("表格只列前 20 场")) throw new Error("row truncation not shown");
  if (!gapHtml.includes("Spearman 0.42")) throw new Error("rank correlation not shown");
  if (!gapHtml.includes("0.0123") || !gapHtml.includes('<td class="numeric">3</td>')) throw new Error("cv / batches not shown");
  const committeeHtml = hosts["analysis-committees"].innerHTML;
  if (!committeeHtml.includes("单位是 session")) throw new Error("committee header missing");
  // 混合桶：5-session 桶够、1-session 桶不够；全局提示必须点名不足桶，不能消失。
  if (!committeeHtml.includes("样本不足以判定：relu/3/second/kimi-k3")) {
    throw new Error("insufficient bucket not named: " + committeeHtml);
  }
  if (!committeeHtml.includes('<td class="numeric">0.2</td>') || !committeeHtml.includes('<td class="numeric">0.5</td>')) {
    throw new Error("cost_per_session_usd not shown");
  }
  if (!hosts["analysis-noise"].innerHTML.includes("缺失不等于 false")) throw new Error("cache_hit coverage note missing");
  if (!hosts["analysis-noise"].innerHTML.includes("reps 分布：15×6")) throw new Error("reps distribution missing");
  sameMode = true;
  calls.length = 0;
  return loadAnalysis();
}).then(() => {
  if (calls.length !== 3) throw new Error("second load must be 3 stamped requests, got " + calls.join(" "));
  if (!calls.some((url) => url.includes("stamp=g1"))) throw new Error("gap stamp missing");
  if (!calls.some((url) => url.includes("stamp=c1"))) throw new Error("committee stamp missing");
  if (!calls.some((url) => url.includes("stamp=n1"))) throw new Error("noise stamp missing");
  console.log("ok");
}).catch((err) => { console.error(err); process.exit(1); });
'''
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_gap_aggregates_cover_the_full_set_not_only_the_display_rows(tmp_path, monkeypatch):
    """@brief 评审回归：25 场同配置已发布运行时，总数/组中位数/秩相关基于 25 场，行只展示 20。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    for number in range(25):
        _published_run(runs, f"conv2d/{number:02d}", speedup=1.1 + number * 0.01,
                       search_best=1.4 + number * 0.01, spent=1.0)
        report = runs / f"conv2d/{number:02d}" / "confirm_report.json"
        body = json.loads(report.read_text())
        body["benchmark"]["candidate"]["cv"] = 0.001 * (number + 1)
        report.write_text(json.dumps(body), encoding="utf-8")
    index.sync_known()

    doc = analysis.gap(protocol_epoch="3")
    assert doc["published_total"] == 25, "已发布总数必须是完整过滤集"
    assert doc["rows"] and len(doc["rows"]) == analysis.PANEL_ROWS
    assert doc["truncated"]["rows"] is True
    assert len(doc["groups"]) == 1
    group = doc["groups"][0]
    assert group["n"] == 25 and group["insufficient"] is False
    assert group["median_signed_delta_pct"] is not None
    assert doc["rank_correlation"]["n"] == 25 and doc["rank_correlation"]["insufficient"] is False


def test_gap_groups_by_the_unified_comparability_key_not_by_task_only(tmp_path, monkeypatch):
    """@brief 评审回归：同 task/纪元、五种 dtype 的五场运行必须分成五组，不能给出跨 dtype 的中位数。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    for dtype in ("fp16", "bf16", "fp32", "int8", "int4"):
        _published_run(runs, f"conv2d/{dtype}", speedup=1.2, search_best=1.3, dtype=dtype)
    index.sync_known()

    doc = analysis.gap(protocol_epoch="3")
    assert doc["published_total"] == 5
    groups = {(group["task"], group["dtype"]): group for group in doc["groups"]}
    assert set(groups) == {("conv2d", dtype) for dtype in ("fp16", "bf16", "fp32", "int8", "int4")}
    for group in groups.values():
        assert group["n"] == 1 and group["insufficient"] is True
        assert group["median_signed_delta_pct"] is None and group["median_usd_per_100pct_gain"] is None


def test_committees_and_noise_follow_the_chosen_epoch_and_the_config_key(tmp_path, monkeypatch):
    """@brief 评审回归：委员会按统一可比性键 + 纪元分桶；噪声只统计所选纪元，不跨纪元暗中汇总。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    first = _published_run(runs, "conv2d/new", speedup=1.2, search_best=1.3, protocol="3", dtype="fp16")
    second = _published_run(runs, "conv2d/old", speedup=1.1, search_best=1.2, protocol="flame_chase_1", dtype="bf16")
    _candidate_file(first, 0, cv=0.01, reps=15, cache_hit=False)
    _candidate_file(second, 0, cv=0.05, reps=15, cache_hit=None)
    index.sync_known()
    _session(index, "conv2d/new", "r1.p0", model="m-new", cost=1.0)
    _session(index, "conv2d/old", "r1.p0", model="m-old", cost=1.0)
    index.conn.commit()

    with pytest.raises(analysis.EpochRequired):
        analysis.committees()
    with pytest.raises(analysis.EpochRequired):
        analysis.noise()

    new_buckets = analysis.committees(protocol_epoch="3")["buckets"]
    old_buckets = analysis.committees(protocol_epoch="2")["buckets"]
    assert [(row["dtype"], row["model"], row["protocol_epoch"]) for row in new_buckets] == [("fp16", "m-new", "3")]
    assert [(row["dtype"], row["model"], row["protocol_epoch"]) for row in old_buckets] == [("bf16", "m-old", "2")]

    new_noise = analysis.noise(protocol_epoch="3")
    old_noise = analysis.noise(protocol_epoch="2")
    assert new_noise["protocol_epoch"] == "3" and new_noise["cv"]["n"] == 1
    assert new_noise["cv"]["p50"] == pytest.approx(0.01) and new_noise["cv"]["max"] == pytest.approx(0.01)
    assert old_noise["protocol_epoch"] == "2" and old_noise["cv"]["p50"] == pytest.approx(0.05)


def test_the_panels_keep_the_byte_budget_with_long_text_and_many_epochs(tmp_path, monkeypatch):
    """@brief 评审回归：500 字符 task、长 model 与 30 个纪元时，三个面板仍 ≤8KB 且置 truncated。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    long_task = "task-" + "x" * 494
    for number in range(25):
        _published_run(runs, f"conv2d/{number:02d}", task=long_task, speedup=1.1 + number * 0.01,
                       search_best=1.4 + number * 0.01, spent=1.0)
    # 另外 30 个纪元（不成已发布），把 epochs 列表撑到上限之外。
    for number in range(30):
        _published_run(runs, f"conv2d/epoch{number:02d}", task=long_task, speedup=1.05,
                       search_best=1.1, spent=1.0, protocol=f"w{number:03d}")
    index.sync_known()
    for number in range(25):
        _session(index, f"conv2d/{number:02d}", "r1.p0", model="model-" + "y" * 200, cost=0.5)
    # 一条超长 run_id 的已发布行（没有目录）：面板不能把整段 id 塞进 20 行。
    index.conn.execute(
        "INSERT INTO runs (run_id, task, dtype, size, status, publication, published_speedup,"
        " search_best_speedup, spent_max_usd, protocol_version, row_json)"
        " VALUES (?, ?, 'fp16', 'paper', 'done', 'confirmed', 1.5, 1.6, 1.0, '3', '{}')",
        ("conv2d/" + "z" * 1000, long_task),
    )
    index.conn.commit()

    documents = {
        "gap": analysis.gap(protocol_epoch="3"),
        "committees": analysis.committees(protocol_epoch="3"),
        "noise": analysis.noise(protocol_epoch="3"),
    }
    for name, document in documents.items():
        size = len(json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        assert size <= analysis.PANEL_BYTES, f"{name} 面板 {size}B 超过 8KB"
    assert documents["gap"]["published_total"] == 26, "超长 id 也计入真实总数"
    assert documents["gap"]["truncated"]["rows"] is True
    assert documents["gap"]["truncated"]["epochs"] is True, "30 个纪元要按 8 项上限截断"
    assert len(documents["gap"]["epochs"]) == analysis.PANEL_EPOCHS
    assert all(len(epoch) <= analysis.EPOCH_CHARS for epoch in documents["gap"]["epochs"])
    # 长 task / 长 model 必须在展示文本上被真正截断（不是靠短字段夹具通过预算）。
    assert documents["gap"]["truncated"]["text"] is True
    assert documents["committees"]["truncated"]["text"] is True
    assert len(documents["committees"]["buckets"]) <= analysis.PANEL_COLUMNS


def test_the_usd_metric_divides_tiny_positive_gains_instead_of_dropping_them():
    """@brief 评审回归：只有 published ≤ 1 才算 excluded；略高于 1 的增益按 ε 下限除。

    @return None
    """
    assert analysis.usd_per_100pct_gain(2.0, 0.9) is None
    assert analysis.usd_per_100pct_gain(2.0, 1.0) is None
    assert analysis.usd_per_100pct_gain(2.0, 1.5) == pytest.approx(4.0)
    assert analysis.usd_per_100pct_gain(2.0, 1 + 5e-10) == pytest.approx(2.0 / analysis.USD_EPS)


def test_the_analysis_trends_filter_by_the_full_comparability_key_in_node():
    """@brief 评审回归：同 task 不同 dtype 的两个组各自只拿自己的点。

    只按 ``row.task === group.task`` 过滤会让两组拿到同一批混合点：图看起来有数据，配置已混。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = Path("observe/static/index.html").read_text()

    def extract(name):
        marker = f"async function {name}(" if f"async function {name}(" in page else f"function {name}("
        start = page.index(marker)
        paren = page.index("(", start)
        depth = 0
        cursor = paren
        while cursor < len(page):
            if page[cursor] == "(":
                depth += 1
            elif page[cursor] == ")":
                depth -= 1
                if depth == 0:
                    break
            cursor += 1
        brace = page.index("{", cursor)
        depth = 0
        for index in range(brace, len(page)):
            if page[index] == "{":
                depth += 1
            elif page[index] == "}":
                depth -= 1
                if depth == 0:
                    return page[start:index + 1]
        raise AssertionError(f"unclosed function {name}")

    names = ("esc", "num", "pctText", "idText", "preferredEpoch", "renderTrend", "analysisFetch", "loadAnalysis")
    state_line = next(
        line.strip() for line in page.splitlines() if line.strip().startswith("const analysisState")
    )
    script = state_line + "\n" + "\n".join(extract(name) for name in names) + r'''
const applied = {task: ""};
const hosts = {};
globalThis.document = { getElementById: (id) => (hosts[id] = hosts[id] || { innerHTML: "" }) };
const base = {size: "paper", target_id: "gpu", protocol_epoch: "3", arm_id: ""};
const gapDoc = { same: false, stamp: "g1", protocol_epoch: "3", epochs: ["3"], published_total: 2,
  unpublished: 0, excluded: 0, truncated: { rows: false },
  rank_correlation: { n: 0, insufficient: true, spearman: null },
  rows: [
    {...base, task: "relu", dtype: "fp16", run_id: "relu/a", search_best: 1.5, published: 1.2,
     signed_delta_pct: -20.0, cv: 0.01, n_batches: 1, usd_per_100pct_gain: 5},
    {...base, task: "relu", dtype: "fp32", run_id: "relu/b", search_best: 1.6, published: 1.4,
     signed_delta_pct: -12.5, cv: 0.02, n_batches: 1, usd_per_100pct_gain: 6}],
  groups: [
    {...base, task: "relu", dtype: "fp16", n: 1, insufficient: true},
    {...base, task: "relu", dtype: "fp32", n: 1, insufficient: true}] };
globalThis.fetch = async (url) => {
  if (url.startsWith("/api/v2/gap")) return { ok: true, json: async () => gapDoc };
  if (url.startsWith("/api/v2/committees") || url.startsWith("/api/v2/noise")) {
    return { ok: true, json: async () => ({ same: true, stamp: "x" }) };
  }
  throw new Error("unexpected url " + url);
};
loadAnalysis().then(() => {
  const html = hosts["analysis-gap"].innerHTML;
  const figures = html.split("<figure").slice(1);
  if (figures.length !== 2) throw new Error("expected two trends, got " + figures.length);
  const fp16 = figures.find((chunk) => chunk.includes("relu · fp16"));
  const fp32 = figures.find((chunk) => chunk.includes("relu · fp32"));
  if (!fp16 || !fp32) throw new Error("trend label lost the config key: " + html);
  const count = (chunk) => (chunk.match(/<circle/g) || []).length;
  if (count(fp16) !== 1) throw new Error("fp16 trend mixed other configs: " + fp16);
  if (count(fp32) !== 1) throw new Error("fp32 trend mixed other configs: " + fp32);
  console.log("ok");
}).catch((err) => { console.error(err); process.exit(1); });
'''
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_the_committees_panel_budgets_the_insufficient_names_too(tmp_path, monkeypatch):
    """@brief 评审回归：辅助列表 ``insufficient_buckets`` 也必须受 8KB 预算约束。

    50 个单-session 桶、model 约 200 字符时，只弹 ``buckets`` 会在桶已空后仍超预算；辅助列表
    还要有截断标志，且完整计数必须保留（``insufficient_count``），不能只留展示项。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    for number in range(50):
        _published_run(runs, f"conv2d/{number:02d}", speedup=1.2, search_best=1.3)
    index.sync_known()
    for number in range(50):
        _session(index, f"conv2d/{number:02d}", "r1.p0",
                 model=f"model-{number:02d}-" + "y" * 200, cost=0.5)
    index.conn.commit()

    doc = analysis.committees(protocol_epoch="3")
    size = len(json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    assert size <= analysis.PANEL_BYTES
    assert doc["insufficient"] is True
    assert doc["insufficient_count"] == 50, "完整集统计不能跟着展示列表一起缩水"
    assert len(doc["insufficient_buckets"]) < 50
    assert doc["truncated"]["insufficient_buckets"] is True
    assert doc["truncated"]["buckets"] is True


def test_noise_reports_more_reps_than_it_shows(tmp_path, monkeypatch):
    """@brief 评审回归：21 种不同 reps 时返回 20 项并置 ``truncated.reps=True``。

    固定 ``LIMIT 20`` 且恒置 False 会让第 21 种 reps 静默消失，读者以为分布完整。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    _published_run(runs, "relu/noise", task="relu")
    index.sync_known()
    index.conn.execute("DELETE FROM candidates WHERE run_id = ?", ("relu/noise",))
    for number in range(21):
        index.conn.execute(
            "INSERT INTO candidates(run_id, idx, label, status, stage, reps) VALUES (?, ?, ?, 'ok', 'bench', ?)",
            ("relu/noise", 100 + number, f"r{number}.p0", number + 1),
        )
    index.conn.commit()

    doc = analysis.noise(task="relu", protocol_epoch="3")
    assert len(doc["reps"]) == analysis.PANEL_COLUMNS
    assert doc["reps_total"] == 21
    assert doc["truncated"]["reps"] is True
    assert sum(row["n"] for row in doc["reps"]) == analysis.PANEL_COLUMNS


def test_an_old_format_confirmation_supplies_no_batch_count(tmp_path):
    """schema-3 确认报告不是当前证据：批次数不可判，不能从 ``result.confirmed_values`` 回填。

    The reviewer's repro: a schema-3 confirm report plus two ``confirmed_values`` used to yield
    ``(None, 2)``.  That 2 is the run's own memory of a confirmation whose records this reader
    cannot check, so the panel must show ``null`` rather than a remembered number.
    """
    run = _unpublished_run(tmp_path, "conv2d/old", speedup=1.2)
    result = json.loads((run / "result.json").read_text())
    result["confirmed_values"] = [1.2, 1.21]
    (run / "result.json").write_text(json.dumps(result))

    cv, batches = analysis._confirm_stats(run, "conv2d/old")
    assert batches is None, "旧格式确认报告的批次数不得由 result.json 回退"
    assert cv is None or isinstance(cv, float)

    # And a current (schema-4) report still supplies the count: the gate is about the format, not
    # about hiding numbers.
    from experiments.common import synthetic_confirmation_documents

    _, report = synthetic_confirmation_documents(
        task="conv2d", dtype="fp16", size="paper", target_id="apple-m4-max",
        speedup=1.2, label="r1.p0", kernel_sha256="a" * 64, values=[1.2, 1.21],
    )
    (run / "confirm_report.json").write_text(json.dumps(report))
    _cv, batches = analysis._confirm_stats(run, "conv2d/old")
    assert batches == 2
