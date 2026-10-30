"""A confirmed speedup is shown only when it is bound to the kernel that was remeasured. No GPU."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from experiments.common import (
    _read_confirm_report, conclusion_of_run, publication_reason, published_conclusion, summarize,
)
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator, RunResult
from tcenv.evaluate import EvalReport
from tcenv.schema import PROTOCOL_VERSION

from ..support.fakes import FakeClient
from ..support.report_build import confirmation_report

QUIET_BENCH = {"noisy": False, "candidate": {"cv": 0.01, "reps": 15}, "baselines": {"mlx": {"cv": 0.01, "reps": 15}}, "best_baseline": "mlx"}

SOURCE = "kernel void relu_kernel() {}\n"

# The single publication predicate also needs the frozen expected and the per-batch identity
# record; these fixtures are a complete, matching confirmation.
_BATCH = {
    "label": "r1.p0", "shape": "full", "build_kind": "native", "measurement_kind": "measured",
    "cache_hit": False, "timing_uncertain": False,
    "identity": {"kind": "match", "differences": [], "shape": "full", "build_kind": "native",
                 "measurement_kind": "measured", "cache_hit": False, "timing_uncertain": False},
}
#: The identity a publishable confirmation must carry on both sides (§7.2).  A frozen expected
#: that omits any required field is ``*_identity_unknown`` and cannot publish, so the fixtures
#: start from a complete document.
_IDENTITY = {
    "evaluation_fingerprint": "1" * 64, "baseline_implementation_digest": "2" * 64,
    "baseline_environment_digest": "3" * 64, "backend": "metal", "backend_version": "1",
    # Required identity on both sides and in the ``evaluation`` projection (§7.2).
    "protocol_version": PROTOCOL_VERSION, "target_id": "apple-m4-max",
}
_BUILD_IDENTITY = {"shape": "full", "build_kind": "native",
                   "opspec_digest": "4" * 64, "lowering_profile_digest": "5" * 64}


def _expected(report: dict) -> dict:
    """The complete frozen expected for ``report``: its identity, minus outcome-only digests."""
    build = report["build"]
    return {
        "protocol_version": report["protocol_version"], "target_id": report["target_id"],
        "build": {
            "candidate_sha256": build["candidate_sha256"],
            "opspec_digest": build["opspec_digest"],
            "lowering_profile_digest": build["lowering_profile_digest"],
            "provenance": {
                key: value for key, value in build["provenance"].items() if key != "artifact_set_digest"
            },
        },
        "evaluation": {field: report[field] for field in _IDENTITY},
    }


def _digest(text: str) -> str:
    """@brief 算出给定内核源码的 sha256，用于把确认结果绑定到被重测的那份源码。

    @param text 内核源码。
    @return sha256 十六进制摘要。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _result(**extra):
    """@brief 造一份 result.json 的内容：搜索最好 1.4、一次状态为 incorrect 的确认 0.9，可用关键字覆盖任意字段。

    @param extra 要覆盖的字段。
    @return 结果字典。
    """
    base = {
        "task": "relu", "dtype": "fp16", "model": "fake", "stop_reason": "max_rounds", "rounds": 1,
        "evaluations": 1, "spent_usd": 0.1, "budget_usd": 1.0, "best_speedup": 1.4, "confirmed_speedup": 0.9,
        "confirmed_status": "incorrect", "confirm_winner_label": "r1.p0", "confirmation_source_sha256": _digest(SOURCE),
        "confirmed_values": [], "target_id": "apple-m4-max",
        "confirmations": [dict(_BATCH, status="ok", failed_batches=0)],
    }
    base.update(extra)
    return base


def test_an_incorrect_confirmation_is_not_published_as_the_search_best():
    """@brief 验证确认状态为 incorrect 时不给发布数字（publication 为未确认），但搜索最好值仍在汇总里单独出现。
    """
    got = published_conclusion(_result(), best_metal_sha256=_digest(SOURCE), confirm_status="incorrect", winner_metal_sha256=_digest(SOURCE))
    assert got == {"speedup": None, "publication": "未确认"}
    row = summarize(_result(), {"status": "incorrect"})
    assert row["confirmed_speedup"] == 0.9 and row["best_speedup"] == 1.4
    assert row["published_speedup"] is None and row["publication"] == "未确认"


def test_a_result_without_the_hash_field_is_an_old_file(tmp_path):
    """@brief 验证没有源码摘要字段的 result.json 被当作旧文件，发布状态是旧版未绑定源码且不给速度。

    @param tmp_path pytest 提供的临时目录。
    """
    old = {"task": "relu", "dtype": "fp16", "model": "fake", "stop_reason": "max_rounds", "rounds": 1, "evaluations": 1,
           "spent_usd": 0.1, "budget_usd": 1.0, "best_speedup": 1.4, "confirmed_speedup": 1.1, "confirmed_status": "ok"}
    assert published_conclusion(old)["publication"] == "旧版未绑定源码"
    assert published_conclusion(old)["speedup"] is None


def _bind_report(report, digest, values, speedup):
    """@brief 造一份**真实的 schema-4** 确认报告：native build + 确认块 + 冻结 expected。

    The publication read boundary runs the whole schema closure (``EvalReport.from_json``), so a
    fixture can no longer be a hand-built subset: this one carries a real native build record
    (``candidate_sha256`` = ``digest``), the top-level evaluation identity and the confirmation
    block with its frozen expected and per-batch records.
    """
    from ..support.report_build import ok_report_dict

    document = ok_report_dict("relu", "fp16", "small", source=SOURCE.encode(), speedup=speedup, mode="full")
    document.update({key: value for key, value in report.items() if key != "benchmark"})
    # A real confirm report carries the top-level ``cache_hit`` fact (§7.2); the explicit-``True``
    # case must survive, so only default it.
    document.setdefault("cache_hit", False)
    confirmation = {
        "kernel_sha256": digest,
        "winner_label": "r1.p0",
        "confirmed_values": list(values),
        "speedup": speedup,
        "candidate_us": None,
        # 谓词要求每条 confirmed value 都有一条批次记录（§7.2）：三个值就得三条。
        "batches": [json.loads(json.dumps(_BATCH)) for _value in values],
    }
    document["benchmark"] = {**QUIET_BENCH, "confirmation": confirmation}
    confirmation["expected"] = _expected(document)
    return document


def test_a_bound_confirmation_shows_the_remeasured_number(tmp_path):
    """@brief 验证摘要绑定成功时用重测值发布，而 best.metal 一变就不再继承搜索数字。

    @param tmp_path pytest 提供的临时目录。
    """
    digest = _digest(SOURCE)
    run = tmp_path / "run"
    (run / "candidates" / "000_r1.p0").mkdir(parents=True)
    (run / "best.metal").write_text(SOURCE)
    (run / "candidates" / "000_r1.p0" / "kernel.metal").write_text(SOURCE)
    (run / "confirm_report.json").write_text(json.dumps(_bind_report({
        "status": "ok", "speedup": 1.25, "benchmark": QUIET_BENCH,
        "protocol_version": PROTOCOL_VERSION, "target_id": "apple-m4-max",
    }, digest, [1.25], 1.25)))
    result = _result(confirmed_status="ok", confirmed_speedup=1.25, confirmation_source_sha256=digest, confirmed_values=[1.25])
    # Without the confirm report there is nothing to check identity against: the binding-only
    # helper must not claim a publication by itself.
    got = published_conclusion(result, best_metal_sha256=digest, confirm_status="ok", winner_metal_sha256=digest)
    assert got == {"speedup": None, "publication": "未确认"}
    row = summarize(result, None, run_dir=run)
    assert row["published_speedup"] == 1.25 and row["publication"] == "confirmed"
    # a different best.metal does not inherit the search number
    (run / "best.metal").write_text(SOURCE + " ")
    row = summarize(result, None, run_dir=run)
    assert row["published_speedup"] is None and row["publication"] == "未确认"


def test_finish_writes_the_winner_hash_and_leaves_it_null_when_confirmation_failed(tmp_path):
    """@brief 验证 _finish 在有确认胜者时把源码摘要写进 result.json，确认失败时该字段仍存在但为 null，搜索值与确认值分开记录。

    @param tmp_path pytest 提供的临时目录。
    """
    cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
    search = SearchConfig(budget_usd=1.0, branches=1, max_rounds=0, eval_reps=3, stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False)

    def ev(*_a, **_k):
        return EvalReport(task="relu", dtype="fp16", size="small", mode="full", status="ok")

    orch = Orchestrator("relu", "fp16", "small", cfg, search, run_dir=tmp_path / "run", client=FakeClient([]), evaluate_fn=ev, log=lambda _s: None, seed=1)
    orch._confirm_winner = SimpleNamespace(label="r1.p0", source=SOURCE, speedup=1.25)
    orch.best = orch._confirm_winner
    res = RunResult(run_dir=str(orch.run_dir), task="relu", dtype="fp16", size="small", model="fake")
    res.confirmed_status, res.confirmed_speedup = "ok", 1.25
    orch._finish(res)
    doc = json.loads((orch.run_dir / "result.json").read_text())
    assert doc["confirmation_source_sha256"] == _digest(SOURCE)
    assert (orch.run_dir / "best.metal").read_text() == SOURCE

    orch._confirm_winner = None
    orch.best = SimpleNamespace(label="init", source=SOURCE, speedup=1.4)
    res2 = RunResult(run_dir=str(orch.run_dir), task="relu", dtype="fp16", size="small", model="fake")
    res2.confirmed_status = "incorrect"
    orch._finish(res2)
    doc2 = json.loads((orch.run_dir / "result.json").read_text())
    assert "confirmation_source_sha256" in doc2 and doc2["confirmation_source_sha256"] is None
    assert doc2["best_speedup"] == 1.4 and doc2["confirmed_speedup"] is None


def test_a_confirmation_served_from_the_evaluation_cache_is_not_published(tmp_path):
    """@brief 验证来自评测缓存的确认报告（cache_hit）不发布数字，发布状态记为不确定。

    cache_hit 表示报告是回放而不是重测，闸门不能把它当成已确认。

    @param tmp_path pytest 提供的临时目录。
    """
    digest = _digest(SOURCE)
    run = tmp_path / "run"
    (run / "candidates" / "000_r1.p0").mkdir(parents=True)
    (run / "best.metal").write_text(SOURCE)
    (run / "candidates" / "000_r1.p0" / "kernel.metal").write_text(SOURCE)
    (run / "confirm_report.json").write_text(json.dumps(_bind_report({
        "status": "ok", "speedup": 1.25, "cache_hit": True, "benchmark": QUIET_BENCH,
        "protocol_version": PROTOCOL_VERSION, "target_id": "apple-m4-max",
    }, digest, [1.25, 1.25, 1.25], 1.25)))
    result = _result(confirmed_status="ok", confirmed_speedup=1.25, confirmation_source_sha256=digest, confirmed_values=[1.25, 1.25, 1.25])
    assert conclusion_of_run(run, result) == {"speedup": None, "publication": "不确定"}


def _bound_run(tmp_path, **report_extra):
    """@brief 造一个源码摘要与确认报告互相绑定的运行目录，供发布闸门的用例复用。

    @param tmp_path pytest 提供的临时目录。
    @param report_extra 追加或覆盖到 confirm_report.json 的字段。
    @return (运行目录, 源码摘要) 二元组。
    """
    digest = _digest(SOURCE)
    run = tmp_path / "run"
    (run / "candidates" / "000_r1.p0").mkdir(parents=True, exist_ok=True)
    (run / "best.metal").write_text(SOURCE)
    (run / "candidates" / "000_r1.p0" / "kernel.metal").write_text(SOURCE)
    report = _bind_report({"status": "ok", "speedup": 1.25, "benchmark": QUIET_BENCH, "protocol_version": PROTOCOL_VERSION, "target_id": "apple-m4-max"}, digest, [1.25], 1.25)
    report.update(report_extra)
    (run / "confirm_report.json").write_text(json.dumps(report))
    return run, digest


def test_a_failed_winner_entry_cannot_be_overruled_by_the_top_level_status(tmp_path):
    """读盘也不能把胜者自己的失败记录改成顶层 ok 后发布。"""
    run, digest = _bound_run(tmp_path)
    result = _result(
        confirmed_status="ok", confirmed_speedup=1.25,
        confirmation_source_sha256=digest, confirmed_values=[1.25],
    )
    result["confirmations"][0].update(status="build_identity_mismatch", failed_batches=1)
    assert conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run, result)["reason"] == "build_identity_mismatch"

    result["confirmations"][0].update(status="ok", failed_batches=1)
    assert conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run, result)["reason"] == "not_confirmed"

    result["confirmations"][0]["failed_batches"] = 0
    assert conclusion_of_run(run, result) == {"speedup": 1.25, "publication": "confirmed"}


def test_a_frozen_expected_that_contradicts_the_report_does_not_publish(tmp_path):
    """冻结 expected 与报告矛盾或不完整时都不发布（读盘路径，§7.2）。

    Two documents are refused here: a frozen expected whose identity drifted away from the report
    (``*_identity_mismatch``), and one that simply does not state a required identity
    (``*_identity_unknown``) while the batch record claims ``match``.  Making the expected complete
    and consistent again restores publication, so the gate rejects the bad document rather than the
    presence of an expected.
    """
    digest = _digest(SOURCE)
    run = tmp_path / "run"
    (run / "candidates" / "000_r1.p0").mkdir(parents=True)
    (run / "best.metal").write_text(SOURCE)
    (run / "candidates" / "000_r1.p0" / "kernel.metal").write_text(SOURCE)
    report = _bind_report({
        "status": "ok", "speedup": 1.25, "benchmark": QUIET_BENCH,
        "protocol_version": PROTOCOL_VERSION, "target_id": "apple-m4-max",
    }, digest, [1.25], 1.25)
    expected = report["benchmark"]["confirmation"]["expected"]
    result = _result(confirmed_status="ok", confirmed_speedup=1.25, confirmation_source_sha256=digest, confirmed_values=[1.25])

    def write():
        (run / "confirm_report.json").write_text(json.dumps(report))

    # A drifted evaluation identity.
    expected["evaluation"]["evaluation_fingerprint"] = "f" * 64
    write()
    assert conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run, result)["reason"] == "evaluation_identity_mismatch"

    # The reviewer's repro: required identity missing/None in the frozen expected still published
    # because the batch record says match.  It is ``evaluation_identity_unknown`` now.
    for field in ("evaluation_fingerprint", "baseline_environment_digest"):
        expected["evaluation"][field] = None
    write()
    assert conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run, result)["reason"] == "evaluation_identity_unknown"

    # A missing build identity is the build-side unknown.
    expected["evaluation"] = _expected(report)["evaluation"]
    expected["build"].pop("opspec_digest")
    write()
    assert conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run, result)["reason"] == "build_identity_unknown"

    # Complete and consistent again → published.
    report["benchmark"]["confirmation"]["expected"] = _expected(report)
    expected = report["benchmark"]["confirmation"]["expected"]
    write()
    assert conclusion_of_run(run, result) == {"speedup": 1.25, "publication": "confirmed"}


def test_a_frozen_expected_without_protocol_or_target_does_not_publish(tmp_path):
    """评审复现：报告与冻结 expected 同时去掉 protocol/target 时，读盘仍必须拒绝。

    The old check was ``confirm_report.get(field) != expected.get(field)``, so removing the pair
    from *both* documents compared ``None == None`` and published ``speedup=1.2`` through
    ``conclusion_verdict``.  Missing/None is ``evaluation_identity_unknown`` now, and restoring the
    same values on both sides publishes again.
    """
    digest = _digest(SOURCE)
    run = tmp_path / "run"
    (run / "candidates" / "000_r1.p0").mkdir(parents=True)
    (run / "best.metal").write_text(SOURCE)
    (run / "candidates" / "000_r1.p0" / "kernel.metal").write_text(SOURCE)
    report = _bind_report({
        "status": "ok", "speedup": 1.25, "benchmark": QUIET_BENCH,
        "protocol_version": PROTOCOL_VERSION, "target_id": "apple-m4-max",
    }, digest, [1.25], 1.25)
    expected = report["benchmark"]["confirmation"]["expected"]
    result = _result(confirmed_status="ok", confirmed_speedup=1.25, confirmation_source_sha256=digest, confirmed_values=[1.25])

    def write():
        (run / "confirm_report.json").write_text(json.dumps(report))

    # A frozen expected that no longer states the pair is an identity unknown...
    for field in ("protocol_version", "target_id"):
        expected.pop(field)
        expected["evaluation"].pop(field)
    write()
    assert conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run, result)["reason"] == "evaluation_identity_unknown"

    # ...and a *report* that drops a required top-level field is refused by the schema-4 read
    # boundary itself (``EvalReport.from_json``): damaged evidence is "not confirmed", never a
    # partial trust of the fields that happen to be left.
    for field in ("protocol_version", "target_id"):
        report.pop(field)
    write()
    assert conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run, result)["reason"] == "not_confirmed"

    # Explicit None on both sides is not agreement either (the schema allows a null there, so this
    # one reaches the predicate and is refused as an unknown identity).
    for field in ("protocol_version", "target_id"):
        report[field] = None
        expected[field] = None
        expected["evaluation"][field] = None
    write()
    assert conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run, result)["reason"] == "evaluation_identity_unknown"

    # The same values on both sides publish again.
    report["protocol_version"] = expected["protocol_version"] = PROTOCOL_VERSION
    report["target_id"] = expected["target_id"] = "apple-m4-max"
    expected["evaluation"]["protocol_version"] = PROTOCOL_VERSION
    expected["evaluation"]["target_id"] = "apple-m4-max"
    write()
    assert conclusion_of_run(run, result) == {"speedup": 1.25, "publication": "confirmed"}


def test_a_report_that_measured_another_candidate_does_not_publish(tmp_path):
    """§7.2：报告本体必须描述被记录的那个候选（``build.candidate_sha256`` 绑定源码摘要）。

    A whole, schema-valid report for a *different* source is bound to this run: the confirmation
    block still carries the winner's hash, so the old checks passed.  The report's own build must
    name the recorded source, otherwise it is a measurement of another kernel.
    """
    digest = _digest(SOURCE)
    run = tmp_path / "run"
    (run / "candidates" / "000_r1.p0").mkdir(parents=True)
    (run / "best.metal").write_text(SOURCE)
    (run / "candidates" / "000_r1.p0" / "kernel.metal").write_text(SOURCE)
    other = "kernel void other_kernel() {}\n"
    report = confirmation_report(
        "relu", "fp16", "small", source=other.encode(), label="r1.p0", values=[1.25], speedup=1.25,
    )
    report["benchmark"]["confirmation"]["kernel_sha256"] = digest
    (run / "confirm_report.json").write_text(json.dumps(report))
    result = _result(confirmed_status="ok", confirmed_speedup=1.25, confirmation_source_sha256=digest, confirmed_values=[1.25])
    assert conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run, result)["reason"] == "sha_mismatch"

    # The same report for the *measured* source publishes: the gate rejects the wrong candidate,
    # not the presence of a build record.
    report = confirmation_report(
        "relu", "fp16", "small", source=SOURCE.encode(), label="r1.p0", values=[1.25], speedup=1.25,
    )
    (run / "confirm_report.json").write_text(json.dumps(report))
    assert conclusion_of_run(run, result) == {"speedup": 1.25, "publication": "confirmed"}


def test_the_report_body_must_be_a_full_native_measurement(tmp_path):
    """§7.2：schema 允许 ``simulator``，但确认报告本体必须是 native/full；partial 直接是非法报告。"""
    run, digest = _bound_run(tmp_path)
    report = json.loads((run / "confirm_report.json").read_text(encoding="utf-8"))
    result = _result(confirmed_status="ok", confirmed_speedup=1.25, confirmation_source_sha256=digest, confirmed_values=[1.25])

    report["build"]["build_kind"] = "simulator"
    (run / "confirm_report.json").write_text(json.dumps(report))
    assert conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run, result)["reason"] == "confirmation_ineligible"

    # ``partial`` is not a legal shape for an ``ok`` report, so the read boundary refuses the whole
    # document before the predicate sees it.
    report["build"]["build_kind"] = "native"
    report["build"]["shape"] = "partial"
    (run / "confirm_report.json").write_text(json.dumps(report))
    assert conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run, result)["reason"] == "not_confirmed"


def test_a_result_target_that_differs_from_the_frozen_identity_does_not_publish(tmp_path):
    """§7.2：``result.target_id`` 与冻结 expected/报告不同或缺字段都不发布（读盘回归）。"""
    run, digest = _bound_run(tmp_path)
    base = dict(confirmed_status="ok", confirmed_speedup=1.25, confirmation_source_sha256=digest, confirmed_values=[1.25])
    assert conclusion_of_run(run, _result(**base)) == {"speedup": 1.25, "publication": "confirmed"}

    got = conclusion_of_run(run, _result(**{**base, "target_id": "other"}))
    assert got == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run, _result(**{**base, "target_id": "other"}))["reason"] == "target_mismatch"
    for missing in (None, ""):
        result = _result(**{**base, "target_id": missing})
        assert conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}
        assert publication_reason(run, result)["reason"] == "evaluation_identity_unknown"


def test_an_illegal_report_field_type_is_refused_at_the_read_boundary(tmp_path):
    """§7.2/评审复现：报告与冻结 expected 的 ``evaluation_fingerprint`` 同时是整数 0 时不得发布。

    ``EvalReport.from_json`` refuses that document; the publication reader used to accept it because
    it only compared the two values with each other.  The reader now runs the same schema closure, so
    the run is simply "not confirmed" (damaged evidence), never a partial trust.
    """
    from tcenv.evaluate import EvalReport
    from tcenv.schema import SchemaError

    run, digest = _bound_run(tmp_path)
    report = json.loads((run / "confirm_report.json").read_text(encoding="utf-8"))
    expected = report["benchmark"]["confirmation"]["expected"]
    report["evaluation_fingerprint"] = 0
    expected["evaluation"]["evaluation_fingerprint"] = 0
    (run / "confirm_report.json").write_text(json.dumps(report))
    result = _result(confirmed_status="ok", confirmed_speedup=1.25, confirmation_source_sha256=digest, confirmed_values=[1.25])
    with pytest.raises(SchemaError):
        EvalReport.from_json(report)
    assert _read_confirm_report(run) is None
    assert conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run, result)["reason"] == "not_confirmed"


def test_a_frozen_expected_without_provenance_does_not_publish(tmp_path):
    """§7.2：冻结 expected 必须给出完整 BuildProvenance；删掉它不是「没有可比的字段」。"""
    from tcenv.evaluate import EvalReport

    run, digest = _bound_run(tmp_path)
    report = json.loads((run / "confirm_report.json").read_text(encoding="utf-8"))
    expected = report["benchmark"]["confirmation"]["expected"]
    expected["build"].pop("provenance")
    (run / "confirm_report.json").write_text(json.dumps(report))
    result = _result(confirmed_status="ok", confirmed_speedup=1.25, confirmation_source_sha256=digest, confirmed_values=[1.25])
    # The report stays a legal document: its own provenance is untouched.
    EvalReport.from_json(report)
    assert conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run, result)["reason"] == "build_identity_unknown"


def test_the_published_number_must_be_the_median_of_the_recorded_batches(tmp_path):
    """@brief 验证发布数字必须是已记录批次的结论：result.json 里与批次对不上的加速比不会被发布。

    手改过或合并错的 result.json 不能发布一个批次支撑不起的数字。

    @param tmp_path pytest 提供的临时目录。
    """
    run, digest = _bound_run(tmp_path)
    report = json.loads((run / "confirm_report.json").read_text())
    report["speedup"] = 1.0
    report["benchmark"]["confirmation"]["speedup"] = 1.0
    report["benchmark"]["confirmation"]["confirmed_values"] = [1.0, 1.0, 1.0]
    # 三个值必须配三条批次记录，否则测到的是「记录不全」而不是「中位数不对」。
    report["benchmark"]["confirmation"]["batches"] = [
        json.loads(json.dumps(_BATCH)) for _value in (1.0, 1.0, 1.0)
    ]
    (run / "confirm_report.json").write_text(json.dumps(report))
    result = _result(confirmed_status="ok", confirmed_speedup=1.9, confirmation_source_sha256=digest,
                     confirmed_values=[1.0, 1.0, 1.0])
    assert conclusion_of_run(run, result)["speedup"] is None
    result["confirmed_speedup"] = 1.0
    assert conclusion_of_run(run, result) == {"speedup": 1.0, "publication": "confirmed"}


def test_a_confirmation_report_with_another_speedup_is_not_published(tmp_path):
    """@brief 报告里的 speedup 与 result 不一致时，不能用 result 的数字冒充该报告的复测。"""
    run, digest = _bound_run(tmp_path)
    report = json.loads((run / "confirm_report.json").read_text())
    report["speedup"] = 99
    (run / "confirm_report.json").write_text(json.dumps(report))
    result = _result(confirmed_status="ok", confirmed_speedup=1.25, confirmation_source_sha256=digest, confirmed_values=[1.25])
    assert conclusion_of_run(run, result)["speedup"] is None


def test_a_non_finite_or_textual_confirmed_speedup_is_not_published(tmp_path):
    """@brief 验证确认加速比是 Inf、NaN、字符串或布尔值时都不发布。

    @param tmp_path pytest 提供的临时目录。
    """
    run, digest = _bound_run(tmp_path)
    for bad in (float("inf"), float("nan"), "1.25", True):
        result = _result(confirmed_status="ok", confirmed_speedup=bad, confirmation_source_sha256=digest,
                         confirmed_values=[1.0])
        assert conclusion_of_run(run, result)["speedup"] is None, bad


def test_a_confirm_report_from_another_epoch_or_target_is_not_published(tmp_path):
    """@brief 验证协议版本或目标芯片对不上的 confirm_report 不发布，状态记为未确认。

    @param tmp_path pytest 提供的临时目录。
    """
    for extra in ({"protocol_version": "0"}, {"target_id": "another-mac"}):
        run, digest = _bound_run(tmp_path / str(extra), **extra)
        result = _result(confirmed_status="ok", confirmed_speedup=1.25, confirmation_source_sha256=digest,
                         confirmed_values=[1.25])
        assert conclusion_of_run(run, result) == {"speedup": None, "publication": "未确认"}, extra

