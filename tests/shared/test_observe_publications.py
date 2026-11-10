"""§8.1-8.2 发布史：目录史、来源解析、当前发布判定的原因码与三类告警。

两个数分开命名：``catalog_confirmed_speedup`` 是历史事实，``current_publication`` 是当前发布门。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.common import conclusion_of_run, publication_reason
from observe import publications, query
from observe.index import Index

_SOURCE = "kernel void publication_fixture() {}\n"


def _index(tmp_path, monkeypatch):
    """@brief 临时 runs 根 + 索引 + 空 catalog 根。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @return ``(runs 根, catalog 根, Index)``。
    """
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    catalog = tmp_path / "kernels"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    monkeypatch.setenv("OBSERVE_REFRESH_ON_READ", "0")
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(catalog))
    return runs, catalog, Index(tmp_path / "observe.sqlite")


def _run(runs: Path, name: str, *, task: str = "conv2d", dtype: str = "fp16", size: str = "paper",
         speedup: float = 1.25, values: list | None = None, protocol: str = "3",
         target: str = "apple-m4-max", run_confirmed: float | None = 1.25) -> Path:
    """@brief 造一个会发布的已确认 run；``run_confirmed`` 控制写进 result 的 confirmed_speedup。

    @param runs runs 根目录。
    @param name run id。
    @param task 任务名。
    @param dtype 数据类型。
    @param size 尺寸标签。
    @param speedup 确认报告与各批的加速比。
    @param values 各批测量值（默认 ``[speedup]``）。
    @param protocol 协议版本。
    @param target 目标 id。
    @param run_confirmed 写进 result 的 ``confirmed_speedup``（None 表示不写）。
    @return run 目录。
    """
    from tcenv.schema import PROTOCOL_VERSION

    directory = runs / name
    directory.mkdir(parents=True)
    digest = hashlib.sha256(_SOURCE.encode("utf-8")).hexdigest()
    values = list(values if values is not None else [speedup])
    (directory / "best.metal").write_text(_SOURCE, encoding="utf-8")
    folder = directory / "candidates" / "000_r1.p0"
    folder.mkdir(parents=True)
    (folder / "kernel.metal").write_text(_SOURCE, encoding="utf-8")
    (folder / "report.json").write_text(json.dumps({
        "status": "ok", "stage": "bench", "speedup": speedup,
    }), encoding="utf-8")
    (directory / "state.json").write_text(json.dumps({
        "task": task, "dtype": dtype, "size": size, "phase": "done", "started_at": 10.0,
        "updated_at": 20.0, "spent_usd": 0.1, "budget_usd": 1.0, "best_speedup": speedup,
    }), encoding="utf-8")
    from experiments.common import synthetic_confirmation_documents

    result, report = synthetic_confirmation_documents(
        task=task, dtype=dtype, size=size, target_id=target,
        speedup=speedup, label="r1.p0", kernel_sha256=digest, values=values,
        protocol_version=PROTOCOL_VERSION if protocol == "flame_chase_1" else protocol,
    )
    result.update({
        "stop_reason": "max_rounds", "spent_usd": 0.1, "budget_usd": 1.0,
        "best_speedup": speedup, "protocol_version": protocol,
    })
    if run_confirmed is None:
        result.pop("confirmed_speedup", None)
    else:
        result["confirmed_speedup"] = run_confirmed
    (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (directory / "confirm_report.json").write_text(json.dumps(report), encoding="utf-8")
    (directory / "run_config.json").write_text(json.dumps({
        "task": task, "dtype": dtype, "size": size, "protocol_version": protocol,
        "target": {"id": target, "backend": "metal", "backend_version": "v1"},
        "search": {"budget_usd": 1.0},
    }), encoding="utf-8")
    return directory


def _archive(catalog: Path, *, task: str = "conv2d", speedup: float, run_id: str,
             confirmed_at: float, dtype: str = "fp16", size: str = "paper",
             label: str = "r1.p0", sha: str, target: str = "apple-m4-max") -> None:
    """@brief 写一行目录史（真实 schema：``updated_at`` 是确认时刻）。

    @param catalog catalog 根目录。
    @param task 任务名。
    @param speedup 目录史里的确认加速比。
    @param run_id 来源目录名。
    @param confirmed_at 确认时刻。
    @param dtype 数据类型。
    @param size 尺寸标签。
    @param label 胜出标签。
    @param sha 内核摘要。
    @param target 目标 id。
    """
    folder = catalog / target / task / dtype / size
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "target_id": target, "task": task, "dtype": dtype, "size": size,
            "task_fingerprint": "fp", "calibration_sha256": "cal",
            "run_id": run_id, "winner_label": label, "source_sha256": sha,
            "confirmed_speedup": speedup, "confirmed_values": [speedup],
            "candidate_us": 700.0, "baseline_us": {"mlx": 800.0}, "updated_at": confirmed_at,
        }) + "\n")


def test_the_real_catalog_keeps_the_points_and_refuses_every_old_format_confirmation(tmp_path, monkeypatch):
    """@brief 真实语料验收（§8.1）：conv2d 三点三值与环比不变；旧格式确认报告一律不可判。

    The catalog history keeps the numbers it published.  What changed is ``current_publication``:
    every real ``confirm_report.json`` is schema 3, and R6 refuses an old-format confirmation
    outright instead of treating its ``status`` as current evidence.  So every such run is a
    publication the current gate cannot reproduce — ``not_confirmed``, counted in
    ``publication_lost``, never backfilled.

    只读真实 ``runs/`` 与 ``kernels/``；索引复制到临时目录再投影，避免并行测试写同一份派生库。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    import shutil

    from observe.index import get_index

    repo = Path(__file__).resolve().parents[2]
    real_db = repo / "data" / "observe.sqlite"
    if not (repo / "kernels").is_dir() or not real_db.exists():
        pytest.skip("仓库里没有真实 catalog/索引")
    copied = tmp_path / "observe.sqlite"
    shutil.copyfile(real_db, copied)
    # 把生产 catalog 复制成临时夹具并只保留本测试断言的历史题：本测试验证的是**历史** conv2d/
    # flash_attention 的旧格式确认与丢失判定，gemm 的新 confirmed 由 test_search_v2_gpu 单独覆盖。
    # 同一理由适用于之后才进入 catalog 的题（affine_ramp 的三 Track 探索、relu 的 P3 分片）：
    # 每跑一次本地搜索就改这份断言只会让这份历史证据随机器漂移，所以用允许列表而不是逐个删。
    catalog = tmp_path / "kernels"
    shutil.copytree(repo / "kernels", catalog)
    keep = {"conv2d", "flash_attention", "relu"}
    for extra in sorted((catalog / "apple-m4-max").iterdir()):
        if extra.is_dir() and extra.name not in keep:
            shutil.rmtree(extra, ignore_errors=True)
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(repo / "runs"))
    monkeypatch.setenv("OBSERVE_DB", str(copied))
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(catalog))
    monkeypatch.setenv("OBSERVE_REFRESH_ON_READ", "0")
    index = get_index()
    index._sync_kernels()
    # 先按真实 runs/ 重新摄取：发布判定与原因码是摄入时投影的列，不追平就会读到旧库的空列。
    index.sync_known()
    doc = publications.publications()
    conv2d = [row for row in doc["rows"] if row["task"] == "conv2d"]
    assert [row["catalog_confirmed_speedup"] for row in conv2d] == pytest.approx(
        [1.445959, 1.738183, 1.648635], abs=1e-6,
    ), "conv2d 三点三值必须落在 §8.1 的 ±1e-6 上"
    assert [row["delta_vs_prev"] for row in conv2d] == [
        None, pytest.approx(20.2097, abs=1e-3), pytest.approx(-5.1518, abs=1e-3),
    ]
    assert [row["is_best_so_far"] for row in conv2d] == [True, True, False]
    # 真实语料里的 confirm_report.json 全部是 schema 3。R6 之后旧格式确认报告是**不可判**证据，
    # 不迁移、不猜字段：`current_publication` 只能是 `未确认`，目录史里的历史数字原样保留。
    assert [row["current_publication"] for row in conv2d] == ["in_progress", "未确认", "未确认"]
    lost = [row for row in doc["rows"] if row["current_publication"] == "未确认" and row["catalog_confirmed_speedup"]]
    assert len(lost) == 4, "每一份旧格式确认报告都是 publishing_lost，而不是被当成当前证据"
    assert {row["publication_reason"] for row in lost} == {"not_confirmed"}
    unknown = [row for row in doc["rows"] if row["source_unknown"]]
    assert len(unknown) == 4 and all(row["delta_vs_prev"] is None for row in unknown)
    assert len(doc["rows"]) == 9
    parseable = [row for row in doc["rows"] if not row["source_unknown"]]
    assert len(parseable) == 5 and all(row["source_available"] and not row["source_ambiguous"] for row in parseable)
    # 计数必须来自完整过滤集：limit 只是每类样本上限，不能用“样本条数”冒充“丢失条数”。
    report = publications.publication_alert_report(limit=2)
    assert report["counts"] == {
        "confirmation_regressed": 1, "publication_lost": 4, "catalog_mismatch": 0,
    }, report["counts"]
    # 4 条 publication_lost 超过 `limit=2` 的样本上限，但 `counts` 仍是完整过滤集。
    assert report["truncated"] is True and report["total"] == 5, (report["truncated"], report["total"])
    lost_alerts = [alert for alert in report["alerts"] if alert["kind"] == "publication_lost"]
    # `limit=2` 是每类**样本**上限：完整条数在 `counts` 里，样本必须先落在最近的 conv2d 上。
    assert {alert["run"] for alert in lost_alerts} == {
        "apple-m4-max/20261024-065432_conv2d_fp16_deepseek-flash",
        "apple-m4-max/20261025-070409_conv2d_fp16_deepseek-flash_g5yab2vc",
    }, "publication_lost 的样本要点名那两条旧格式确认报告"
    assert {alert["reason"] for alert in lost_alerts} == {"not_confirmed"}
    assert [alert["kind"] for alert in report["alerts"]].count("confirmation_regressed") == 1
    assert all(alert["reason"] for alert in report["alerts"])


def test_a_synthetic_catalog_reports_regressions_losses_and_mismatches(tmp_path, monkeypatch):
    """@brief 合成语料：三点环比、发布丢失、目录史与 result 不一致各触发一条告警。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, catalog, index = _index(tmp_path, monkeypatch)
    first = _run(runs, "conv2d/a", speedup=1.40, run_confirmed=1.40)
    second = _run(runs, "conv2d/b", speedup=1.50, run_confirmed=1.50)
    third = _run(runs, "conv2d/c", speedup=1.20, run_confirmed=1.20)
    lost = _run(runs, "conv2d/lost", speedup=1.10, run_confirmed=1.10)
    report = json.loads((lost / "confirm_report.json").read_text())
    report["benchmark"].pop("confirmation")
    (lost / "confirm_report.json").write_text(json.dumps(report), encoding="utf-8")
    index.sync_known()
    digest = hashlib.sha256(_SOURCE.encode("utf-8")).hexdigest()
    _archive(catalog, task="conv2d", speedup=1.40, run_id="a", confirmed_at=100.0, sha=digest)
    _archive(catalog, task="conv2d", speedup=1.50, run_id="b", confirmed_at=200.0, sha=digest)
    _archive(catalog, task="conv2d", speedup=1.20, run_id="c", confirmed_at=300.0, sha=digest)
    _archive(catalog, task="conv2d", speedup=1.10, run_id="lost", confirmed_at=400.0, sha=digest)
    index._sync_kernels()

    doc = publications.publications(task="conv2d")
    rows = {row["run_id_raw"]: row for row in doc["rows"]}
    assert [rows["a"]["delta_vs_prev"], rows["b"]["delta_vs_prev"], rows["c"]["delta_vs_prev"]] == [
        None, pytest.approx(7.1429, abs=1e-3), pytest.approx(-20.0, abs=1e-3),
    ]
    assert [rows[key]["is_best_so_far"] for key in ("a", "b", "c")] == [True, True, False]
    assert rows["lost"]["current_publication"] != "confirmed"
    assert rows["lost"]["publication_reason"] == "missing_confirmation_block"

    alerts = publications.publication_alerts(limit=3)
    kinds = [alert["kind"] for alert in alerts]
    assert "confirmation_regressed" in kinds
    assert sum(1 for alert in alerts if alert["kind"] == "publication_lost") == 1
    regression = next(alert for alert in alerts if alert["kind"] == "confirmation_regressed")
    assert regression["run"] == "conv2d/c" and regression["delta_pct"] == pytest.approx(-20.0)

    # 手工改 result.confirmed_speedup → catalog_mismatch。告警读的是索引投影，所以先让摄入
    # 看到这次就地改写（巡检路径不打开 run 目录，见 test_patrol_reads_publication_alerts_from_the_index_only）。
    body = json.loads((third / "result.json").read_text())
    body["confirmed_speedup"] = 1.25
    (third / "result.json").write_text(json.dumps(body), encoding="utf-8")
    index.sync_known()
    index._sync_kernels()
    alerts = publications.publication_alerts(limit=3)
    assert any(alert["kind"] == "catalog_mismatch" and alert["run"] == "conv2d/c" for alert in alerts)
    assert first.is_dir() and second.is_dir()


def test_source_disambiguation_marks_an_ambiguous_match_instead_of_picking_one(tmp_path, monkeypatch):
    """@brief §8.2：同 label 的多份候选里 sha 命中 3 份时 ``source_ambiguous=1``，不随便挑一个。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, catalog, index = _index(tmp_path, monkeypatch)
    directory = _run(runs, "conv2d/sha", speedup=1.3)
    for number in (1, 2):
        folder = directory / "candidates" / f"00{number}_r1.p0"
        folder.mkdir(parents=True)
        (folder / "kernel.metal").write_text(_SOURCE, encoding="utf-8")
        (folder / "report.json").write_text(json.dumps({"status": "ok", "stage": "bench"}))
    index.sync_known()
    digest = hashlib.sha256(_SOURCE.encode("utf-8")).hexdigest()
    _archive(catalog, task="conv2d", speedup=1.3, run_id="sha", confirmed_at=100.0, sha=digest)
    index._sync_kernels()

    row = publications.publications(task="conv2d")["rows"][0]
    assert row["run_id"] == "conv2d/sha"
    assert row["source_available"] is False and row["source_ambiguous"] is True

    # 删掉多余的候选后唯一命中 → 重新投影后 available。
    for number in (1, 2):
        folder = directory / "candidates" / f"00{number}_r1.p0"
        (folder / "kernel.metal").unlink()
        (folder / "report.json").unlink()
        folder.rmdir()
    index._sync_kernels()
    row = publications.publications(task="conv2d")["rows"][0]
    assert row["source_available"] is True and row["source_ambiguous"] is False


def test_a_missing_catalog_root_yields_no_rows_and_no_extra_json_reads(tmp_path, monkeypatch):
    """@brief §8.1 验收：``kernels/`` 缺失时端点空数组，``tick()`` 的 ``json_reads`` 不增加。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, catalog, index = _index(tmp_path, monkeypatch)
    assert not catalog.exists()
    index._last_full = 0
    index.tick()
    reads = index.stats["json_reads"]
    index._last_full = 0
    index.tick()
    assert index.stats["json_reads"] == reads, "没有 catalog 时不该多读 JSON"
    doc = publications.publications()
    assert doc["rows"] == [] and doc["total"] == 0
    assert publications.publication_alerts() == []


def _reason_fixture(tmp_path, monkeypatch):
    """@brief 一个可发布的 run + 索引，用于逐条构造 ``publication_reason`` 的 reason 码。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @return ``(directory, report_doc, result_doc)``。
    """
    runs, _catalog, index = _index(tmp_path, monkeypatch)
    directory = _run(runs, "conv2d/reason", speedup=1.25)
    index.sync_known()
    return directory, json.loads((directory / "confirm_report.json").read_text()), json.loads(
        (directory / "result.json").read_text())


def _write_reason(directory: Path, report: dict, result: dict | None) -> dict:
    """@brief 写回夹具并返回 ``publication_reason``。

    @param directory run 目录。
    @param report 确认报告。
    @param result 结果字典；None 表示删掉 result.json。
    @return ``publication_reason`` 的输出。
    """
    (directory / "confirm_report.json").write_text(json.dumps(report), encoding="utf-8")
    if result is None:
        (directory / "result.json").unlink(missing_ok=True)
        return publication_reason(directory, None)
    (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return publication_reason(directory, result)


def test_publication_reason_covers_every_branch(tmp_path, monkeypatch):
    """@brief 验收：14 个 reason 码逐个命中；``conclusion_of_run`` 仍只返回两个键。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory, report, result = _reason_fixture(tmp_path, monkeypatch)
    base = _write_reason(directory, json.loads(json.dumps(report)), dict(result))
    assert base == {"publication": "confirmed", "reason": "confirmed"}
    assert conclusion_of_run(directory, result) == {"speedup": 1.25, "publication": "confirmed"}

    assert _write_reason(directory, json.loads(json.dumps(report)), None)["reason"] == "no_result_json"

    unbound = {key: value for key, value in result.items() if key != "confirmation_source_sha256"}
    assert _write_reason(directory, json.loads(json.dumps(report)), unbound)["reason"] == "not_confirmed"

    bad_status = {**result, "confirmed_status": "incorrect"}
    assert _write_reason(directory, json.loads(json.dumps(report)), bad_status)["reason"] == "not_confirmed"

    bad_sha = {**result, "confirmation_source_sha256": "b" * 64}
    assert _write_reason(directory, json.loads(json.dumps(report)), bad_sha)["reason"] == "sha_mismatch"

    bad_speed = {**result, "confirmed_speedup": 1.3}
    assert _write_reason(directory, json.loads(json.dumps(report)), bad_speed)["reason"] == "speedup_mismatch"

    bad_values = {**result, "confirmed_values": []}
    assert _write_reason(directory, json.loads(json.dumps(report)), bad_values)["reason"] == "values_mismatch"

    # 以下分支都在确认报告的 benchmark 里改，改完要写回。
    def mutated(change):
        doc = json.loads(json.dumps(report))
        change(doc)
        return doc

    missing_block = mutated(lambda doc: doc["benchmark"].pop("confirmation"))
    assert _write_reason(directory, missing_block, dict(result))["reason"] == "missing_confirmation_block"

    label = mutated(lambda doc: doc["benchmark"]["confirmation"].update({"winner_label": "r9.p0"}))
    assert _write_reason(directory, label, dict(result))["reason"] == "label_mismatch"

    values = mutated(lambda doc: doc["benchmark"]["confirmation"].update({"confirmed_values": [1.3]}))
    assert _write_reason(directory, values, dict(result))["reason"] == "values_mismatch"

    report_speed = mutated(lambda doc: (doc.__setitem__("speedup", 1.3),
                                        doc["benchmark"]["confirmation"].update({"speedup": 1.3})))
    assert _write_reason(directory, report_speed, dict(result))["reason"] == "speedup_mismatch"

    candidate = mutated(lambda doc: doc["benchmark"]["confirmation"].update({"candidate_us": 900.0}))
    with_candidate = {**result, "candidate_us": 700.0}
    assert _write_reason(directory, candidate, with_candidate)["reason"] == "candidate_us_mismatch"

    protocol = mutated(lambda doc: doc.__setitem__("protocol_version", "999"))
    assert _write_reason(directory, protocol, dict(result))["reason"] == "protocol_mismatch"

    target = mutated(lambda doc: doc.__setitem__("target_id", "other-gpu"))
    assert _write_reason(directory, target, dict(result))["reason"] == "target_mismatch"

    noisy = mutated(lambda doc: doc["benchmark"]["candidate"].update({"cv": 0.5}))
    assert _write_reason(directory, noisy, dict(result))["reason"] == "timing_uncertain"

    def with_batches(change):
        """改确认值时同步补出**逐值一条**的批次记录：谓词要求两者一一对应（§7.2）。"""
        doc = json.loads(json.dumps(report))
        change(doc)
        confirmation = doc["benchmark"]["confirmation"]
        confirmation["batches"] = [json.loads(json.dumps(confirmation["batches"][0]))
                                   for _value in confirmation["confirmed_values"]]
        return doc

    disagree = with_batches(lambda doc: (doc["benchmark"]["confirmation"].update(
        {"confirmed_values": [1.0, 1.1], "speedup": 1.05}), doc.__setitem__("speedup", 1.05)))
    disagree_result = {**result, "confirmed_values": [1.0, 1.1], "confirmed_speedup": 1.05}
    assert _write_reason(directory, disagree, disagree_result)["reason"] == "batches_disagree"

    # 两个测量值、只有一条身份记录：批次记录不全，不发布（不是“两处不一致”，是缺证据）。
    incomplete = with_batches(lambda doc: (doc["benchmark"]["confirmation"].update(
        {"confirmed_values": [1.0, 1.01], "speedup": 1.005}), doc.__setitem__("speedup", 1.005)))
    incomplete["benchmark"]["confirmation"]["batches"].pop()
    incomplete_result = {**result, "confirmed_values": [1.0, 1.01], "confirmed_speedup": 1.005}
    assert _write_reason(directory, incomplete, incomplete_result)["reason"] == "not_confirmed"

    cached = mutated(lambda doc: doc.__setitem__("cache_hit", True))
    assert _write_reason(directory, cached, dict(result))["reason"] == "cache_hit"


def test_patrol_carries_publication_alerts_with_reason_codes(tmp_path, monkeypatch):
    """@brief 巡检包含发布类告警（目录史回退 / 发布丢失），每条都带 reason 代码且文档仍 ≤1.5KB。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, catalog, index = _index(tmp_path, monkeypatch)
    good = _run(runs, "conv2d/best", speedup=1.30)
    worse = _run(runs, "conv2d/worse", speedup=1.10)
    lost = _run(runs, "conv2d/lost", speedup=1.00, run_confirmed=1.00)
    report = json.loads((lost / "confirm_report.json").read_text())
    report["benchmark"].pop("confirmation")
    (lost / "confirm_report.json").write_text(json.dumps(report), encoding="utf-8")
    index.sync_known()
    digest = hashlib.sha256(_SOURCE.encode("utf-8")).hexdigest()
    # 目录史顺序：先 1.00（丢失），再 1.30，最后 1.10 —— 只有一条环比回退，另一条是发布丢失。
    _archive(catalog, speedup=1.00, run_id="lost", confirmed_at=100.0, sha=digest)
    _archive(catalog, speedup=1.30, run_id="best", confirmed_at=200.0, sha=digest)
    _archive(catalog, speedup=1.10, run_id="worse", confirmed_at=300.0, sha=digest)
    index._sync_kernels()

    doc = query.patrol()
    kinds = [alert["kind"] for alert in doc["alerts"]]
    assert "confirmation_regressed" in kinds and "publication_lost" in kinds
    assert all(alert.get("reason") for alert in doc["alerts"] if alert["kind"] in {
        "confirmation_regressed", "publication_lost", "catalog_mismatch"})
    assert len(json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) <= 1536
    assert good.is_dir() and worse.is_dir()


def test_the_publication_stamp_covers_every_file_the_panel_reads(tmp_path, monkeypatch):
    """@brief 评审回归：戳必须覆盖目录史、确认报告、候选源码、best 与纪元所读的 run 文件。

    只盯全局 revision 与 ``result.json`` 的 stat 会让“目录史 1.5→1.6”和“确认报告就地改写”都返回
    ``same=True``：面板显示的是新值，客户端却继续用旧缓存。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, catalog, index = _index(tmp_path, monkeypatch)
    directory = _run(runs, "conv2d/a", speedup=1.4, run_confirmed=1.4)
    index.sync_known()
    digest = hashlib.sha256(_SOURCE.encode("utf-8")).hexdigest()
    _archive(catalog, speedup=1.4, run_id="a", confirmed_at=1, sha=digest)
    index._sync_kernels()
    history = catalog / "apple-m4-max/conv2d/fp16/paper/history.jsonl"

    def stale(label: str, mutate) -> dict:
        stamp = publications.publications(task="conv2d")["stamp"]
        mutate()
        answer = publications.publications(task="conv2d", stamp=stamp)
        assert answer.get("same") is not True, f"{label} 改了，旧戳却仍然 same=True"
        return publications.publications(task="conv2d")

    def rewrite_history() -> None:
        lines = [json.loads(line) for line in history.read_text().splitlines()]
        lines[0]["confirmed_speedup"] = 1.6
        history.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")

    document = stale("目录史就地改写", rewrite_history)
    assert document["rows"][0]["catalog_confirmed_speedup"] == pytest.approx(1.6)
    assert index.catalog_revision() > 0

    def append_history() -> None:
        _archive(catalog, speedup=1.2, run_id="a", confirmed_at=2, sha=digest)

    document = stale("目录史追加", append_history)
    assert document["total"] == 2

    def rewrite_confirm_report() -> None:
        (directory / "confirm_report.json").write_text(
            json.dumps({"status": "ok", "note": "in-place"}), encoding="utf-8")

    document = stale("确认报告就地改写", rewrite_confirm_report)
    assert document["rows"][0]["current_publication"] != "confirmed"

    def rewrite_candidate() -> None:
        kernel = directory / "candidates" / "000_r1.p0" / "kernel.metal"
        kernel.write_text(_SOURCE + "// edited in place\n", encoding="utf-8")

    document = stale("候选来源变化", rewrite_candidate)
    assert document["rows"][0]["source_available"] is False

    def rewrite_best() -> None:
        (directory / "best.metal").write_text(_SOURCE + "// rewritten best\n", encoding="utf-8")

    document = stale("best 就地改写", rewrite_best)
    assert document["rows"][0]["current_publication"] != "confirmed"

    def rewrite_epoch_files() -> None:
        body = json.loads((directory / "result.json").read_text())
        body.pop("protocol_version", None)
        (directory / "result.json").write_text(json.dumps(body), encoding="utf-8")
        config = json.loads((directory / "run_config.json").read_text())
        config.pop("protocol_version", None)
        (directory / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
        state = json.loads((directory / "state.json").read_text())
        state["protocol_version"] = "flame_chase_1"
        (directory / "state.json").write_text(json.dumps(state), encoding="utf-8")

    document = stale("纪元所读的 run_config/state", rewrite_epoch_files)
    assert document["rows"][0]["comparability_key"][-1] == "2"


def test_a_catalog_row_re_resolves_after_its_run_appears(tmp_path, monkeypatch):
    """@brief 评审回归：catalog 先到、run 后到时，下一次 tick 必须重解析而不是永久 ``source_unknown=1``。

    解析依赖 ``runs``/``candidates``，所以失效条件除了目录史文件指纹还要含 runs 代际。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, catalog, index = _index(tmp_path, monkeypatch)
    digest = hashlib.sha256(_SOURCE.encode("utf-8")).hexdigest()
    _archive(catalog, speedup=1.25, run_id="late", confirmed_at=1, sha=digest)
    index._sync_kernels()
    before = publications.publications(task="conv2d")["rows"]
    assert [row["run_id"] for row in before] == [""]
    assert before[0]["source_unknown"] == 1

    _run(runs, "conv2d/late", speedup=1.25)
    index.sync_known()
    after = publications.publications(task="conv2d")["rows"]
    assert after[0]["run_id"] == "conv2d/late"
    assert after[0]["source_unknown"] == 0
    assert after[0]["current_publication"] == "confirmed"
    assert after[0]["source_available"] is True
    # 没有新变化时不再重建目录史投影（否则每个请求都白写一次表）。
    revision = index.catalog_revision()
    publications.publications(task="conv2d")
    assert index.catalog_revision() == revision


def test_the_regression_alert_uses_the_unrounded_catalog_values(tmp_path, monkeypatch):
    """@brief 评审回归：1.0 → 0.999999998 的下降必须触发 ``confirmation_regressed``。

    展示百分比先换成百分比再 round(4) 会得到 ``-0.0``；判定必须用未舍入的相邻目录史值。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, catalog, index = _index(tmp_path, monkeypatch)
    _run(runs, "conv2d/a", speedup=1.0, run_confirmed=1.0)
    _run(runs, "conv2d/b", speedup=0.999999998, run_confirmed=0.999999998)
    index.sync_known()
    digest = hashlib.sha256(_SOURCE.encode("utf-8")).hexdigest()
    _archive(catalog, speedup=1.0, run_id="a", confirmed_at=1, sha=digest)
    _archive(catalog, speedup=0.999999998, run_id="b", confirmed_at=2, sha=digest)
    index._sync_kernels()

    rows = publications.publications(task="conv2d")["rows"]
    assert rows[0]["catalog_confirmed_speedup"] - rows[1]["catalog_confirmed_speedup"] > 1e-9
    assert rows[1]["regressed"] is True
    assert rows[1]["delta_vs_prev"] == pytest.approx(0.0, abs=1e-6)
    alerts = publications.publication_alerts(limit=3)
    assert any(alert["kind"] == "confirmation_regressed" and alert["run"] == "conv2d/b" for alert in alerts)

    # 低于 1e-9 的下降不是回退：公式的 epsilon 不能被“四舍五入后为 0”替代。
    runs2, catalog2, index2 = _index(tmp_path / "tiny", monkeypatch)
    _run(runs2, "conv2d/a", speedup=1.0, run_confirmed=1.0)
    _run(runs2, "conv2d/b", speedup=0.9999999995, run_confirmed=0.9999999995)
    index2.sync_known()
    _archive(catalog2, speedup=1.0, run_id="a", confirmed_at=1, sha=digest)
    _archive(catalog2, speedup=0.9999999995, run_id="b", confirmed_at=2, sha=digest)
    index2._sync_kernels()
    tiny_rows = publications.publications(task="conv2d")["rows"]
    assert tiny_rows[1]["regressed"] is False
    assert not [alert for alert in publications.publication_alerts(limit=3)
                if alert["kind"] == "confirmation_regressed"]


def _five_publication_alerts(tmp_path, monkeypatch):
    """@brief 恰好 2 条回退、2 条发布丢失、1 条目录不一致的夹具。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @return ``(runs 根, catalog 根, Index)``。
    """
    runs, catalog, index = _index(tmp_path, monkeypatch)
    _run(runs, "conv2d/first", speedup=1.60, run_confirmed=1.60)
    _run(runs, "conv2d/best", speedup=1.80, run_confirmed=1.80)
    lost_one = _run(runs, "conv2d/lost1", speedup=1.30, run_confirmed=1.30)
    lost_two = _run(runs, "conv2d/lost2", speedup=1.20, run_confirmed=1.20)
    _run(runs, "conv2d/bad", speedup=1.10, run_confirmed=1.10)
    for directory in (lost_one, lost_two):
        report = json.loads((directory / "confirm_report.json").read_text())
        report["benchmark"].pop("confirmation")
        (directory / "confirm_report.json").write_text(json.dumps(report), encoding="utf-8")
    index.sync_known()
    digest = hashlib.sha256(_SOURCE.encode("utf-8")).hexdigest()
    _archive(catalog, speedup=1.60, run_id="first", confirmed_at=1, sha=digest)
    _archive(catalog, speedup=1.80, run_id="best", confirmed_at=2, sha=digest)
    _archive(catalog, speedup=1.30, run_id="lost1", confirmed_at=3, sha=digest)
    _archive(catalog, speedup=1.20, run_id="lost2", confirmed_at=4, sha=digest)
    # 目录史写得比 result.confirmed_speedup（1.10）高：只触发 catalog_mismatch，不再多一条回退。
    _archive(catalog, speedup=1.85, run_id="bad", confirmed_at=5, sha=digest)
    index._sync_kernels()
    return runs, catalog, index


def test_patrol_keeps_the_full_publication_set_with_budget_alerts(tmp_path, monkeypatch):
    """@brief 评审回归：5 条发布 + 3 条预算（约 1.1KB）必须全部保留，不能按类预切。

    发布告警（含丢失/不一致）排在预算前面；全局配额是 10 条/1,536B，8 条远未超额。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, _catalog, index = _five_publication_alerts(tmp_path, monkeypatch)
    for number in range(3):
        index.conn.execute(
            "INSERT INTO runs (run_id, task, dtype, status, spent_usd, budget_usd,"
            " granted_budget_usd, stop_reason) VALUES (?, 'relu', 'fp16', 'done', 2.0, 1.0, 1.0, 'max_rounds')",
            (f"budget/{number}",),
        )
    index.conn.commit()
    report = publications.publication_alert_report()
    assert report["counts"] == {
        "confirmation_regressed": 2, "publication_lost": 2, "catalog_mismatch": 1,
    }
    doc = query.patrol()
    kinds = [alert["kind"] for alert in doc["alerts"]]
    assert kinds[:5] == [
        "confirmation_regressed", "confirmation_regressed",
        "publication_lost", "publication_lost", "catalog_mismatch",
    ]
    assert kinds[5:] == ["budget_over", "budget_over", "budget_over"]
    assert len(doc["alerts"]) == 8
    assert doc["truncated"]["alerts"] is False
    assert len(json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) <= 1536


def test_publication_alerts_count_the_full_catalog_not_only_the_200_panel_rows(tmp_path, monkeypatch):
    """@brief 评审回归：告警统计基于完整过滤集，不能从面板的 200 行派生。

    第 201 行以后的回退也要计数（``counts``），面板仍只展示 200 行并置 ``truncated.rows``。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, catalog, index = _index(tmp_path, monkeypatch)
    _run(runs, "conv2d/many", speedup=1.0, run_confirmed=1.0)
    index.sync_known()
    digest = hashlib.sha256(_SOURCE.encode("utf-8")).hexdigest()
    total = 205
    for number in range(total):
        _archive(catalog, speedup=2.10 - number * 0.005, run_id="many", confirmed_at=float(number + 1), sha=digest)
    index._sync_kernels()

    document = publications.publications(task="conv2d")
    assert document["total"] == total
    assert len(document["rows"]) == publications.PUBLICATION_ROWS
    assert document["truncated"]["rows"] is True
    report = publications.publication_alert_report()
    assert report["counts"]["confirmation_regressed"] == total - 1


def test_patrol_reads_publication_alerts_from_the_index_only(tmp_path, monkeypatch):
    """@brief 评审回归：``patrol()`` 的发布告警只读索引投影，不打开任何 run 目录。

    统计 ``detail.resolve_run`` 与 ``detail._load``：凡是指向 runs 根的读取都算违约。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from observe import detail

    runs, _catalog, index = _five_publication_alerts(tmp_path, monkeypatch)
    root = runs.resolve()
    touches: list[str] = []
    real_resolve = detail.resolve_run
    real_load = detail._load

    def counting_resolve(run_id):
        touches.append(f"resolve:{run_id}")
        return real_resolve(run_id)

    def counting_load(path):
        resolved = Path(path).resolve()
        if resolved == root or root in resolved.parents:
            touches.append(f"load:{resolved}")
        return real_load(path)

    monkeypatch.setattr(detail, "resolve_run", counting_resolve)
    monkeypatch.setattr(detail, "_load", counting_load)
    doc = query.patrol()
    assert len(doc["alerts"]) == 5
    assert touches == []
    assert index.catalog_revision() > 0


def test_the_publications_view_shows_unresolvable_sources_and_reuses_its_stamp_in_node():
    """@brief §8.1 验收 3 的 UI 部分：发布史面板显示 ``source_unknown`` 并复用 stamp。

    发布史按需加载（按钮触发），不占首屏与分析视图的请求预算；第二次调用只发一个带 stamp 的
    请求且不重绘。来源不可解析与来源有歧义必须各有明确文案。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = Path("observe/static/index.html").read_text()
    assert 'id="analysis-publications-load"' in page
    assert 'analysisFetch("/api/v2/publications"' in page

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

    names = ("esc", "num", "pctText", "idText", "analysisFetch", "loadPublications")
    state_line = next(
        line.strip() for line in page.splitlines() if line.strip().startswith("const analysisState")
    )
    script = state_line + "\n" + "\n".join(extract(name) for name in names) + r'''
const hosts = {};
globalThis.document = { getElementById: (id) => (hosts[id] = hosts[id] || { innerHTML: "" }) };
const doc = { same: false, stamp: "p1", total: 202, truncated: { rows: true }, rows: [
  { task: "relu", dtype: "fp16", size: "paper", catalog_confirmed_speedup: 1.2, delta_vs_prev: null,
    regressed: false, current_publication: "confirmed", publication_reason: "confirmed",
    run_id: "relu/a", run_id_raw: "a", source_unknown: 0, source_available: true, source_ambiguous: false },
  { task: "relu", dtype: "fp16", size: "paper", catalog_confirmed_speedup: 1.1, delta_vs_prev: -8.3333,
    regressed: true, current_publication: "未确认", publication_reason: "missing_confirmation_block",
    run_id: "", run_id_raw: "run", source_unknown: 1, source_available: false, source_ambiguous: false },
  { task: "relu", dtype: "fp16", size: "paper", catalog_confirmed_speedup: 1.05, delta_vs_prev: -4.5455,
    regressed: true, current_publication: "confirmed", publication_reason: "confirmed",
    run_id: "relu/c", run_id_raw: "c", source_unknown: 0, source_available: false, source_ambiguous: true } ] };
const calls = [];
globalThis.fetch = async (url) => {
  calls.push(url);
  const body = calls.length === 1 ? doc : { same: true, stamp: "p1" };
  return { ok: true, json: async () => body };
};
loadPublications().then(() => {
  if (calls.length !== 1) throw new Error("first load must be one request, got " + calls.join(" "));
  if (!calls[0].startsWith("/api/v2/publications")) throw new Error("wrong endpoint: " + calls[0]);
  const html = hosts["analysis-publications"].innerHTML;
  if (!html.includes("来源不可解析")) throw new Error("source_unknown not shown: " + html);
  if (!html.includes("来源有歧义")) throw new Error("source_ambiguous not shown: " + html);
  if (!html.includes("目录史 202 行")) throw new Error("total not shown: " + html);
  if (!html.includes("只列前 200 行")) throw new Error("row truncation not shown: " + html);
  if (!html.includes("missing_confirmation_block")) throw new Error("reason code not shown: " + html);
  if (!html.includes("目录史（历史事实）")) throw new Error("catalog-vs-current note missing");
  return loadPublications();
}).then(() => {
  if (calls.length !== 2) throw new Error("second load must be one stamped request, got " + calls.join(" "));
  if (!calls[1].includes("stamp=p1")) throw new Error("stamp not reused: " + calls[1]);
  console.log("ok");
}).catch((err) => { console.error(err); process.exit(1); });
'''
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_publications_serialize_on_the_write_lock_and_reenter_from_tick(tmp_path, monkeypatch):
    """@brief 评审回归：公开写入口与 tick 串行化，且 tick 已持锁时同线程重入不死锁。

    发布史端点会经 ``ensure_kernel_history()`` 写库；HTTP 在线程里调用、连接开了
    ``check_same_thread=False``。它必须和 10 秒全扫互斥，但 ``tick()`` 自己持锁再调时不能锁死。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    import threading
    import time

    import observe.index as index_mod

    runs, catalog, index = _index(tmp_path, monkeypatch)
    _run(runs, "conv2d/a", speedup=1.25)
    index.sync_known()
    digest = hashlib.sha256(_SOURCE.encode("utf-8")).hexdigest()
    _archive(catalog, speedup=1.25, run_id="a", confirmed_at=1, sha=digest)

    held: list[bool] = []
    real = index_mod.Index._resolve_history_run

    def spy(self, *arguments):
        held.append(index_mod._WRITE.locked())
        return real(self, *arguments)

    monkeypatch.setattr(index_mod.Index, "_resolve_history_run", spy)
    index._sync_kernels()
    assert held and all(held), "目录史投影必须在写锁内重建"

    # 同线程重入：模拟 tick() 持锁时顺手调 _sync_kernels() 与发布史。
    done: list[bool] = []

    def reenter() -> None:
        with index_mod._WRITE:
            index._sync_kernels()
            publications.publications(task="conv2d")
            done.append(True)

    thread = threading.Thread(target=reenter, daemon=True)
    thread.start()
    thread.join(timeout=10)
    assert done == [True], "同线程重入写锁死锁"

    # 跨线程串行：另一条线程持锁时，发布史必须等它放锁，不能并发写同一个连接。
    started = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def holder() -> None:
        with index_mod._WRITE:
            started.set()
            release.wait(10)

    def requester() -> None:
        publications.publications(task="conv2d")
        order.append("after")

    owner = threading.Thread(target=holder, daemon=True)
    owner.start()
    assert started.wait(10)
    waiter = threading.Thread(target=requester, daemon=True)
    waiter.start()
    time.sleep(0.2)
    assert order == [], "发布史在写锁被占用时仍然进了临界区"
    release.set()
    owner.join(timeout=10)
    waiter.join(timeout=10)
    assert order == ["after"]


def test_patrol_flags_a_true_global_alert_overflow_without_losing_publication_kinds(tmp_path, monkeypatch):
    """@brief 真超额用例：5 条发布 + 8 条预算 = 13 条候选，只留全局 10 条并置 ``truncated.alerts``。

    发布类必须优先保留（含丢失与不一致），被丢的是预算尾部。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, _catalog, index = _five_publication_alerts(tmp_path, monkeypatch)
    for number in range(8):
        index.conn.execute(
            "INSERT INTO runs (run_id, task, dtype, status, spent_usd, budget_usd,"
            " granted_budget_usd, stop_reason) VALUES (?, 'relu', 'fp16', 'done', 2.0, 1.0, 1.0, 'max_rounds')",
            (f"budget/{number}",),
        )
    index.conn.commit()

    doc = query.patrol()
    alerts = doc["alerts"]
    publication_kinds = {"confirmation_regressed", "publication_lost", "catalog_mismatch"}
    assert len(alerts) == 10
    assert sum(1 for alert in alerts if alert["kind"] in publication_kinds) == 5
    assert sum(1 for alert in alerts if alert["kind"].startswith("budget_")) == 5
    assert doc["truncated"]["alerts"] is True
    assert len(json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) <= 1536
