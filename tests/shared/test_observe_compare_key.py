"""§8.0（PR12）：comparability key 取代 ``row_hash``，``compare`` 按 (arm_id, config_key) 分组。

一个臂里出现多个 ``config_key`` 时不再静默丢弃运行，而是并列进 ``mixed`` 并点名 run_id。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.manifest import attach, ensure
from observe import query
from observe.compare_key import FROZEN_FIELDS, comparison_group, config_key, protocol_epoch
from observe.index import Index

ROOT = Path(__file__).resolve().parents[2]


def _config(*, task="relu", dtype="fp16", size="small", budget=1.0, protocol="3", **extra) -> dict:
    """@brief 造一条最小运行行（``config_key`` 只需要这些列）。

    @param task 任务名。
    @param dtype 数据类型。
    @param size 规模标签。
    @param budget 预算。
    @param protocol 协议版本（``protocol_epoch`` 的输入）。
    @param extra 额外列。
    @return 行字典。
    """
    row = {"task": task, "dtype": dtype, "size": size, "budget_usd": budget, "protocol_version": protocol}
    row.update(extra)
    return row


def test_the_config_key_covers_the_comparable_fields_and_ignores_the_rest():
    """@brief 配置键只由题目/规模/预算/协议纪元与冻结条件决定，不看时间、花费与 delta。

    @return None
    """
    frozen = {"tasks": ["relu"], "dtype": "fp16", "budget_usd": 1.0}
    base = _config()
    key = config_key(base, frozen)
    assert len(key) == 12 and all(char in "0123456789abcdef" for char in key)
    assert config_key(dict(base), dict(frozen)) == key, "同一配置必须稳定"
    # 不参与：运行时刻、花费明细、发布数字、run id、臂的 delta（delta 由 binding=drift 判定）。
    for noise in ({"started_at": 1.0}, {"spent_usd": 9.9}, {"published_speedup": 1.5},
                  {"run_id": "run/x"}, {"delta": {"search.rounds": 3}}):
        assert config_key({**base, **noise}, frozen) == key, f"{sorted(noise)} 不该进配置键"
    # 参与：题目/规模/预算/协议纪元、冻结条件里的题目/规模/预算。
    assert config_key(_config(task="gemm"), frozen) != key
    assert config_key(_config(size="paper"), frozen) != key
    assert config_key(_config(budget=2.0), frozen) != key
    assert config_key(_config(protocol="2"), frozen) != key
    assert config_key(base, {**frozen, "budget_usd": 2.0}) != key
    assert config_key(base, {**frozen, "size": "paper"}) != key
    assert config_key(base, {**frozen, "tasks": ["gemm"]}) != key
    assert set(FROZEN_FIELDS) == {"task", "tasks", "dtype", "size", "budget_usd"}


def test_the_protocol_epoch_matches_the_index_and_the_grouping_key_is_fixed():
    """@brief ``flame_chase_1`` 记成 ``2``；跨运行分组键是 (task, dtype, size, target, epoch, arm)。

    @return None
    """
    assert protocol_epoch({"protocol_version": "flame_chase_1"}) == "2"
    assert protocol_epoch({"protocol_version": "3"}) == "3"
    assert protocol_epoch({"protocol_epoch": "2", "protocol_version": "flame_chase_1"}) == "2"
    row = {"task": "conv2d", "dtype": "fp16", "size": "paper", "target_id": "apple-m4-max",
           "protocol_version": "3", "arm_id": None}
    assert comparison_group(row) == ("conv2d", "fp16", "paper", "apple-m4-max", "3", "")
    assert comparison_group({**row, "arm_id": "relu"})[-1] == "relu"


def _experiment(tmp_path, monkeypatch, *, frozen, arms, kind="repeatability"):
    """@brief 临时 runs 根 + 实验清单。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @param frozen 冻结条件。
    @param arms arm 列表。
    @param kind 实验类型（``contract_menu`` 会走 judge）。
    @return ``(runs 根, 实验目录)``。
    """
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    monkeypatch.delenv("OBSERVE_REFRESH_ON_READ", raising=False)
    experiment = runs / "exp"
    ensure(experiment, kind=kind, driver=kind, frozen=frozen, arms=arms)
    return runs, experiment


def _run(experiment: Path, arm: str, relative: str, *, size: str = "small", speedup=None,
         budget: float = 1.0, fingerprint: str = "fp") -> None:
    """@brief 写一个已结束、与实验绑定的 run，并登记到 arm。

    @param experiment 实验目录。
    @param arm 臂 id。
    @param relative 相对实验目录的 run 路径。
    @param size 规模标签。
    @param speedup 搜索最优（不构成发布数字）。
    @param budget 预算。
    @param fingerprint 任务指纹（同一臂内必须一致才是 verified）。
    """
    directory = experiment / relative
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "phase": "done", "started_at": 10.0, "updated_at": 20.0,
        "spent_usd": 0.1, "best_speedup": speedup,
    }))
    (directory / "result.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.1,
        "budget_usd": budget, "best_speedup": speedup,
    }))
    (directory / "run_config.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": size, "task_fingerprint": fingerprint,
        "search": {"budget_usd": budget},
    }))
    attach(experiment, arm, relative)


PURE_ARM_FIELDS = {
    "arm_id", "n", "published_n", "median", "min", "max", "confirmed_n", "fast", "spent_usd", "gain_per_usd",
}


def test_a_single_config_arm_keeps_the_old_fields_and_has_no_mixed_rows(tmp_path, monkeypatch):
    """@brief 纯单配置臂：``mixed == []``，臂行与旧实现逐字段一致（不因为改动多出键或改数）。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    frozen = {"tasks": ["relu"], "dtype": "fp16", "budget_usd": 1.0}
    _runs, experiment = _experiment(tmp_path, monkeypatch, frozen=frozen,
                                    arms=[{"arm_id": "relu", "delta": {}, "planned_n": 2}])
    _run(experiment, "relu", "relu/trial_1", speedup=1.1)
    _run(experiment, "relu", "relu/trial_2", speedup=1.2)
    Index(tmp_path / "observe.sqlite").sync_known()

    doc = query.compare("exp")
    assert doc["mixed"] == []
    assert len(doc["arms"]) == 1
    arm = doc["arms"][0]
    assert set(arm) == PURE_ARM_FIELDS
    assert arm["arm_id"] == "relu" and arm["n"] == 2 and arm["published_n"] == 0
    assert arm["median"] is None and arm["min"] is None and arm["max"] is None
    assert arm["confirmed_n"] == 0 and arm["spent_usd"] == pytest.approx(0.2)
    assert arm["gain_per_usd"] is None
    assert set(arm["fast"]) == {"0.95", "1", "1.005"}
    assert doc["verdict"] == "样本不足以判定" and doc["reason"] == "发布数字少于 2 个"


def test_a_mixed_arm_is_reported_with_its_run_ids_instead_of_dropped(tmp_path, monkeypatch):
    """@brief 一个臂混了两种可比配置时，``mixed`` 非空并点名 run_id，两组各自算指标、一条都不丢。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    # 冻结条件里没有 size：两种 size 都能 verified，但它们的 config_key 不同 → 混合臂。
    frozen = {"tasks": ["relu"], "dtype": "fp16", "budget_usd": 1.0}
    _runs, experiment = _experiment(tmp_path, monkeypatch, frozen=frozen,
                                    arms=[{"arm_id": "relu", "delta": {}, "planned_n": 2}])
    _run(experiment, "relu", "relu/trial_1", size="small", speedup=1.1)
    _run(experiment, "relu", "relu/trial_2", size="paper", speedup=1.2)
    Index(tmp_path / "observe.sqlite").sync_known()

    doc = query.compare("exp")
    assert doc["mixed"], "混合臂必须报出来，不能静默丢弃"
    assert {tuple(sorted(entry["runs"])) for entry in doc["mixed"]} == {
        ("exp/relu/trial_1",), ("exp/relu/trial_2",),
    }
    assert all(entry["arm_id"] == "relu" for entry in doc["mixed"])
    keys = {entry["config_key"] for entry in doc["mixed"]}
    assert len(keys) == 2 and all(len(key) == 12 for key in keys)
    assert len(doc["arms"]) == 2, "一个臂、两种配置 → 两行，各自算指标"
    assert sum(row["n"] for row in doc["arms"]) == 2, "没有任何运行被丢掉"
    assert all(row["arm_id"] == "relu" and row["config_key"] in keys for row in doc["arms"])
    assert all(set(row) == PURE_ARM_FIELDS | {"config_key"} for row in doc["arms"])
    # 混合臂不给跨配置的 winner：判定身份按 config_key 分桶，且整臂胜利会被误导。
    assert doc["verdict"] == "样本不足以判定" and doc["winner"] is None
    assert "混了多种可比配置" in doc["reason"]


_SOURCE = "kernel void confirmed() {}\n"


def _confirmed_run(experiment: Path, arm: str, relative: str, *, speedup: float, size: str,
                   budget: float = 1.0, fingerprint: str = "fp") -> None:
    """@brief 写一个真的会发布数字的已确认 run（确认门绑定 best/候选内核 + 各批一致）。

    @param experiment 实验目录。
    @param arm 臂 id。
    @param relative 相对实验目录的 run 路径。
    @param speedup 确认加速比（同时也是各批中位数）。
    @param size 规模标签（决定 config_key）。
    @param budget 预算。
    @param fingerprint 任务指纹（同臂一致才是 verified）。
    """
    import hashlib

    from tcenv.schema import PROTOCOL_VERSION

    directory = experiment / relative
    directory.mkdir(parents=True)
    digest = hashlib.sha256(_SOURCE.encode("utf-8")).hexdigest()
    values = [speedup]
    (directory / "best.metal").write_text(_SOURCE, encoding="utf-8")
    folder = directory / "candidates" / "000_r1.p0"
    folder.mkdir(parents=True)
    (folder / "kernel.metal").write_text(_SOURCE, encoding="utf-8")
    (folder / "report.json").write_text(json.dumps({
        "status": "ok", "stage": "bench", "speedup": speedup,
    }), encoding="utf-8")
    (directory / "state.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "phase": "done", "started_at": 10.0, "updated_at": 20.0,
        "spent_usd": 0.1, "best_speedup": speedup,
    }), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": size, "stop_reason": "max_rounds",
        "spent_usd": 0.1, "budget_usd": budget, "best_speedup": speedup, "target_id": "apple-m4-max",
        "confirmation_source_sha256": digest, "confirm_winner_label": "r1.p0",
        "confirmed_status": "ok", "confirmed_speedup": speedup, "confirmed_values": values,
        "candidate_us": None,
    }), encoding="utf-8")
    (directory / "confirm_report.json").write_text(json.dumps({
        "status": "ok", "speedup": speedup, "protocol_version": PROTOCOL_VERSION,
        "target_id": "apple-m4-max",
        "benchmark": {
            "noisy": False, "candidate": {"cv": 0.01, "reps": 15},
            "baselines": {"mlx": {"cv": 0.01, "reps": 15}}, "best_baseline": "mlx",
            "confirmation": {
                "kernel_sha256": digest, "winner_label": "r1.p0", "confirmed_values": values,
                "speedup": speedup, "candidate_us": None,
            },
        },
    }), encoding="utf-8")
    (directory / "run_config.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": size,
        "target": {"id": "apple-m4-max", "backend": "metal", "backend_version": "v1"},
        "task_fingerprint": fingerprint, "search": {"budget_usd": budget},
    }), encoding="utf-8")
    attach(experiment, arm, relative)


def test_a_mixed_contract_menu_arm_cannot_win_with_two_single_sample_configs(tmp_path, monkeypatch):
    """@brief 评审回归：混合臂的两种单样本配置不能凑成“两次发布”并赢下 contract_menu。

    旧实现只在展示时按 config_key 分组，判定仍用 arm_id 作 slug：a 臂两条（size=small 2.0、
    size=paper 2.1）会被合成一个 2 样本的契约，b 臂 1.0/1.1 被它压过 → verdict 胜出、winner=a，
    与对照视图里的“这些运行不可直接对比”自相矛盾。现在判定身份带 config_key，且混合臂不给 winner。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    frozen = {"tasks": ["relu"], "dtype": "fp16", "budget_usd": 1.0}
    _runs, experiment = _experiment(
        tmp_path, monkeypatch, frozen=frozen, kind="contract_menu",
        arms=[{"arm_id": "a", "delta": {}, "planned_n": 2}, {"arm_id": "b", "delta": {}, "planned_n": 2}],
    )
    # 冻结条件不含 size：两种 size 都能 verified，但 config_key 不同。
    _confirmed_run(experiment, "a", "a/trial_1", speedup=2.0, size="small")
    _confirmed_run(experiment, "a", "a/trial_2", speedup=2.1, size="paper")
    _confirmed_run(experiment, "b", "b/trial_1", speedup=1.0, size="small")
    _confirmed_run(experiment, "b", "b/trial_2", speedup=1.1, size="small")
    Index(tmp_path / "observe.sqlite").sync_known()

    doc = query.compare("exp")
    assert doc["kind"] == "contract_menu"
    assert doc["mixed"] and {entry["arm_id"] for entry in doc["mixed"]} == {"a"}
    assert len({entry["config_key"] for entry in doc["mixed"]}) == 2
    # 两种配置各只有一个发布样本；它们不能被合并成一个 2 样本契约。
    by_config = {(row["arm_id"], row.get("config_key")): row for row in doc["arms"]}
    for entry in doc["mixed"]:
        assert by_config[("a", entry["config_key"])]["published_n"] == 1
    assert next(row for row in doc["arms"] if row["arm_id"] == "b")["published_n"] == 2
    assert doc["verdict"] != "胜出" and doc["winner"] is None, "混合配置不得产生 winner"
    assert "混了多种可比配置" in doc["reason"]
    # 对照：若按 arm_id 合并（旧语义），这两组单样本会凑成 2 样本并真的判 a 胜出——说明本夹具是有区分力的。
    from experiments.contract_menu import judge

    merged = judge([
        {"slug": "a", "published_speedup": 2.0}, {"slug": "a", "published_speedup": 2.1},
        {"slug": "b", "published_speedup": 1.0}, {"slug": "b", "published_speedup": 1.1},
    ], ["a", "b"])
    assert merged["winner"] == "a", "夹具必须能暴露旧语义的错误"


def test_row_hash_is_gone_from_the_source():
    """@brief 验收：``row_hash`` 已删除（源码里不再出现），分组只走 ``config_key``。

    @return None
    """
    needle = "row" + "_hash"
    offenders = []
    for base in (ROOT / "observe", ROOT / "taic", ROOT / "tests"):
        for path in sorted(base.rglob("*.py")):
            if path.resolve() == Path(__file__).resolve():
                continue
            if needle in path.read_text(encoding="utf-8", errors="replace"):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], f"还有文件引用 {needle}：{offenders}"
