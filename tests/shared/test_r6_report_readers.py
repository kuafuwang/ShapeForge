"""R6: evidence readers accept **only** the current report schema (4) and reject damage strictly.

CPU only.  Plan §8: a run's schema-3 report is not "a slightly older current report" — it has no
batch identity and no frozen expected, so it is undecidable evidence: refused, never migrated and
never silently treated as the current format.
"""

from __future__ import annotations

import json

import pytest

from experiments import common
from taic.artifacts import NeedsInspection, read_versioned, read_versioned_exact
from taic.experience import build_experience_index
from tcenv.evaluate import EvalReport
from tcenv.schema import PROTOCOL_VERSION, REPORT_SCHEMA_VERSION

from ..support.orchestrator import stamp_report
from tests.support.v1_tasks import V1_TASK

KERNEL_FILE = "kernel.metal"


def _report(*, schema: int = REPORT_SCHEMA_VERSION) -> dict:
    report = EvalReport(
        task=V1_TASK, dtype="fp16", size="small", mode="full", seed=1, status="ok", stage="done",
        speedup=1.2, target_id="apple-m4-max", protocol_version=PROTOCOL_VERSION,
        backend="metal", backend_version="1",
    )
    report.speedup, report.speedups, report.candidate_us, report.baseline_us = 1.2, {"mlx": 1.2}, 100.0, {"mlx": 120.0}
    stamp_report(report, V1_TASK, "fp16", "small", "kernel void k() {}\n", build_kind="native")
    document = report.to_json()
    document["schema_version"] = schema
    return document


def test_the_confirm_report_reader_accepts_exactly_the_current_schema(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    assert common.read_confirm(run) is None
    document = _report()
    (run / "confirm_report.json").write_text(json.dumps(document), encoding="utf-8")
    assert common.read_confirm(run)["schema_version"] == REPORT_SCHEMA_VERSION


@pytest.mark.parametrize("schema", [0, 3, 5, "4", 4.0, True])
def test_an_old_or_unknown_confirm_report_is_refused_not_upgraded(tmp_path, schema):
    """A non-4 confirm report is undecidable evidence: NeedsInspection, not ``None``."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "confirm_report.json").write_text(json.dumps(_report(schema=schema)), encoding="utf-8")
    with pytest.raises(NeedsInspection):
        common.read_confirm(run)


def test_a_duplicate_key_in_a_confirm_report_is_damage(tmp_path):
    """``json.loads`` would keep the last value; the strict loader refuses the document instead."""
    run = tmp_path / "run"
    run.mkdir()
    text = json.dumps(_report())
    (run / "confirm_report.json").write_text(text[:-1] + ', "speedup": 9.0}', encoding="utf-8")
    with pytest.raises(NeedsInspection):
        common.read_confirm(run)


def test_the_disk_publication_reader_treats_a_schema3_report_as_undecidable(tmp_path):
    """The published-conclusion path must not read an old report as current evidence."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "confirm_report.json").write_text(json.dumps(_report(schema=3)), encoding="utf-8")
    assert common._read_confirm_report(run) is None


def test_the_checkpoint_reader_still_accepts_older_versions_but_not_duplicate_keys(tmp_path):
    """``read_versioned`` keeps its max-version contract; only the strictness about keys changed."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"schema_version": 0, "a": 1}), encoding="utf-8")
    assert read_versioned(path, 1) == {"schema_version": 0, "a": 1}
    path.write_text('{"schema_version": 0, "a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(NeedsInspection):
        read_versioned(path, 1)


def test_read_versioned_exact_requires_the_version_and_reports_absence_as_none(tmp_path):
    path = tmp_path / "report.json"
    assert read_versioned_exact(path, version=REPORT_SCHEMA_VERSION) is None
    path.write_text(json.dumps(_report()), encoding="utf-8")
    assert read_versioned_exact(path, version=REPORT_SCHEMA_VERSION) is not None
    with pytest.raises(NeedsInspection):
        read_versioned_exact(path, version=REPORT_SCHEMA_VERSION + 1)


def test_experience_mining_ignores_an_old_report(tmp_path):
    """Experience is written to an agent's prompt: an old-format report is not evidence for it."""
    folder = tmp_path / "run" / "candidates" / "000_init"
    folder.mkdir(parents=True)
    (folder / KERNEL_FILE).write_text("kernel void k() {}\n", encoding="utf-8")
    (folder / "report.json").write_text(json.dumps(_report(schema=3)), encoding="utf-8")
    got = build_experience_index([tmp_path / "run"])
    assert got["wins"] == 0 and got["entries"] == []

    (folder / "report.json").write_text(json.dumps(_report()), encoding="utf-8")
    folder2 = tmp_path / "run" / "candidates" / "001_more"
    folder2.mkdir()
    (folder2 / KERNEL_FILE).write_text("kernel void k() {}\n// faster\n", encoding="utf-8")
    faster = _report()
    faster["speedup"] = 1.6
    (folder2 / "report.json").write_text(json.dumps(faster), encoding="utf-8")
    assert build_experience_index([tmp_path / "run"])["wins"] == 1


def test_two_values_with_one_batch_identity_are_not_published(tmp_path):
    """两条测量值、一条身份记录：批次记录不全，读盘侧也不发布（谓词 §7.2 一一对应）。

    这是合成夹具曾依赖过的漏洞：旧谓词只遍历现有批次，不看它是否覆盖全部 confirmed value。
    修复后纯函数与读盘侧都必须是 ``not_confirmed``，而不是把缺的那条当成「一致」。
    """
    run = tmp_path / "run"
    (run / "candidates" / "000_r1.p0").mkdir(parents=True)
    (run / "candidates" / "000_r1.p0" / KERNEL_FILE).write_text("kernel void k() {}\n")
    (run / "best.metal").write_text("kernel void k() {}\n")
    digest = __import__("hashlib").sha256(b"kernel void k() {}\n").hexdigest()
    result, confirm = common.synthetic_confirmation_documents(
        task=V1_TASK, dtype="fp16", size="small", target_id="apple-m4-max",
        speedup=1.2, label="r1.p0", kernel_sha256=digest, values=[1.2, 1.21],
    )
    assert len(confirm["benchmark"]["confirmation"]["batches"]) == 2
    # 只有一条身份记录，却宣称两个测量值：不发布。
    confirm["benchmark"]["confirmation"]["batches"].pop()
    (run / "confirm_report.json").write_text(json.dumps(confirm), encoding="utf-8")
    assert common.publication_reason(run, result) == {"publication": "未确认", "reason": "not_confirmed"}
    assert common.conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}


def _bound_run(tmp_path, name: str, *, values: list[float] | None = None):
    """@brief 一个通过确认门的合法绑定（best/候选内核同哈希），返回 ``(run, result)``。"""
    run = tmp_path / name
    folder = run / "candidates" / "000_r1.p0"
    folder.mkdir(parents=True)
    source = "kernel void k() {}\n"
    (folder / KERNEL_FILE).write_text(source)
    (run / "best.metal").write_text(source)
    digest = __import__("hashlib").sha256(source.encode()).hexdigest()
    result, report = common.synthetic_confirmation_documents(
        task=V1_TASK, dtype="fp16", size="small", target_id="apple-m4-max",
        speedup=1.2, label="r1.p0", kernel_sha256=digest,
        **({"values": values} if values else {}),
    )
    return run, result, report


@pytest.mark.parametrize("version", [4.0, True, 3, 5, "4", None])
def test_only_an_integer_four_confirm_report_can_publish(tmp_path, version):
    """确认报告的版本必须是非 bool 的整数 4：``4.0 == 4`` 不能让浮点版本文档过门。

    The reviewer's repro: a fully bound, otherwise valid confirmation whose ``schema_version`` is
    ``4.0`` still returned ``{'speedup': 1.2, 'publication': 'confirmed'}`` because the reader used
    ``!=``.  The gate is published evidence, so the version test is the shared
    ``document_state`` (type-strict) rather than a float-comparing equality.
    """
    run, result, report = _bound_run(tmp_path, f"run{str(version).replace('.', '_')}")
    report["schema_version"] = version
    (run / "confirm_report.json").write_text(json.dumps(report), encoding="utf-8")
    assert common.conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}
    assert common.publication_reason(run, result)["reason"] == "not_confirmed"
    # The strict reader refuses it outright (``NeedsInspection``), it does not pretend it is absent.
    with pytest.raises(NeedsInspection):
        common.read_confirm(run)


def test_an_integer_four_confirm_report_still_publishes(tmp_path):
    """正例：同一个绑定、版本是整数 4 时照常发布（门是类型严格，不是把 4 也拒掉）。"""
    run, result, report = _bound_run(tmp_path, "run_ok")
    assert report["schema_version"] == 4 and not isinstance(report["schema_version"], bool)
    (run / "confirm_report.json").write_text(json.dumps(report), encoding="utf-8")
    assert common.conclusion_of_run(run, result)["publication"] == "confirmed"
    assert common.read_confirm(run)["schema_version"] == 4


def test_a_float_version_does_not_pass_any_candidate_or_confirm_reader(tmp_path):
    """同一个类型严格判定覆盖每个读入口：``4.0`` 不得被当成 4。

    The reviewer's repro set ``schema_version=4.0`` with ``status=ok, speedup=9.9`` and both
    ``observe.feedback._candidate_report`` and ``taic.review._report_view`` returned the numbers,
    because each compared with ``!=`` against the integer constant.  Every reader now asks the one
    ``taic.artifacts.document_state``; these four are the entry points that had their own copy.
    """
    from observe import analysis, query
    from observe.feedback import _candidate_report
    from taic import brief, review

    run = tmp_path / "run"
    candidate = run / "candidates" / "000_r1.p0"
    candidate.mkdir(parents=True)
    (candidate / KERNEL_FILE).write_text("kernel void k() {}\n")
    float_report = {"schema_version": 4.0, "status": "ok", "speedup": 9.9,
                    "candidate_us": 1.0, "benchmark": {"candidate": {"cv": 0.01, "reps": 15}}}
    (candidate / "report.json").write_text(json.dumps(float_report), encoding="utf-8")
    (run / "confirm_report.json").write_text(json.dumps(float_report), encoding="utf-8")
    (run / "result.json").write_text(json.dumps({
        "task": V1_TASK, "dtype": "fp16", "size": "small", "stop_reason": "max_rounds",
        "confirmed_values": [9.9, 9.8], "confirm_winner_label": "r1.p0",
    }), encoding="utf-8")
    (run / "state.json").write_text(json.dumps({"task": V1_TASK, "phase": "done"}), encoding="utf-8")

    document, path = _candidate_report(run, "r1.p0")
    assert document is None and path.endswith("report.json")
    assert review._report_view(run, "r1.p0") == {}
    assert brief._cand_from_dir(candidate)["speedup"] is None
    cv, batches = analysis._confirm_stats(run, "run")
    assert cv is None and batches is None

    # Integer 4 in the same files is still read (the gate is type-strict, not "reject floats").
    integer_report = {**float_report, "schema_version": 4}
    (candidate / "report.json").write_text(json.dumps(integer_report), encoding="utf-8")
    (run / "confirm_report.json").write_text(json.dumps(integer_report), encoding="utf-8")
    document, _path = _candidate_report(run, "r1.p0")
    assert document is not None and document["speedup"] == pytest.approx(9.9)
    assert review._report_view(run, "r1.p0")["speedup"] == pytest.approx(9.9)
    assert brief._cand_from_dir(candidate)["speedup"] == pytest.approx(9.9)
    assert analysis._confirm_stats(run, "run")[0] == pytest.approx(0.01)


@pytest.mark.parametrize("change", ["absent", "null", "text"])
def test_a_batch_without_a_boolean_timing_flag_does_not_publish(tmp_path, change):
    """读盘侧同一负例：批次记录里缺 `timing_uncertain`（或为 null/文本）就不发布。

    The reviewer's repro used ``synthetic_confirmation_documents(values=[1.2, 1.19])`` (a legal
    binding) and then deleted ``benchmark.confirmation.batches[*]["timing_uncertain"]`` — the gate
    still said ``confirmed``, because only an explicit ``True`` was refused.
    """
    run, result, report = _bound_run(tmp_path, f"run_{change}", values=[1.2, 1.19])
    batches = report["benchmark"]["confirmation"]["batches"]
    assert len(batches) == 2
    for batch in batches:
        if change == "absent":
            batch.pop("timing_uncertain", None)
        elif change == "null":
            batch["timing_uncertain"] = None
        else:
            batch["timing_uncertain"] = "false"
    (run / "confirm_report.json").write_text(json.dumps(report), encoding="utf-8")
    assert common.conclusion_verdict(run, result) == (
        {"speedup": None, "publication": "未确认"}, "not_confirmed",
    )
    assert common.conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}


def test_an_explicit_false_timing_flag_still_publishes(tmp_path):
    """正例：两条批次都显式 `timing_uncertain: false` 时照常发布。"""
    run, result, report = _bound_run(tmp_path, "run_stable", values=[1.2, 1.19])
    batches = report["benchmark"]["confirmation"]["batches"]
    assert len(batches) == 2 and all(batch["timing_uncertain"] is False for batch in batches)
    (run / "confirm_report.json").write_text(json.dumps(report), encoding="utf-8")
    assert common.conclusion_verdict(run, result)[0]["publication"] == "confirmed"


@pytest.mark.parametrize("change", ["absent", "null", "text", "zero", "one"])
def test_a_confirm_report_without_a_boolean_cache_hit_does_not_publish(tmp_path, change):
    """读盘侧同一负例：顶层 `cache_hit` 缺失/null/文本/数值 → 不发布（逐批 False 不能替它）。

    The reviewer's repro: with a legal ``synthetic_confirmation_documents(values=[1.2, 1.19])``
    binding, only mutating ``confirm_report["cache_hit"]`` gave ``confirmed`` for
    absent/null/``"false"``/``0``/``1``.  §7.2 requires the confirm report itself to state
    ``cache_hit: false``; a required fact cannot be inferred from the per-batch records.
    """
    run, result, report = _bound_run(tmp_path, f"run_cache_{change}", values=[1.2, 1.19])
    if change == "absent":
        report.pop("cache_hit", None)
    elif change == "null":
        report["cache_hit"] = None
    elif change == "text":
        report["cache_hit"] = "false"
    elif change == "zero":
        report["cache_hit"] = 0
    else:
        report["cache_hit"] = 1
    (run / "confirm_report.json").write_text(json.dumps(report), encoding="utf-8")
    assert common.conclusion_verdict(run, result) == (
        {"speedup": None, "publication": "未确认"}, "not_confirmed",
    )
    assert common.conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}


def test_an_explicit_false_cache_hit_still_publishes(tmp_path):
    """正例：顶层显式 `cache_hit: false`（逐批也 `false`）时照常发布。"""
    run, result, report = _bound_run(tmp_path, "run_cache_ok", values=[1.2, 1.19])
    assert report["cache_hit"] is False
    assert all(batch["cache_hit"] is False for batch in report["benchmark"]["confirmation"]["batches"])
    (run / "confirm_report.json").write_text(json.dumps(report), encoding="utf-8")
    assert common.conclusion_verdict(run, result)[0]["publication"] == "confirmed"
