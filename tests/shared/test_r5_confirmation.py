"""R5: the unified confirmation gate (local path).  CPU only, no device."""

from __future__ import annotations

import json

import pytest

from experiments.common import conclusion_of_run
from tcenv.evaluate import EvalReport
from tcenv.evaluation import LocalEvaluator
from tcenv.schema import PROTOCOL_VERSION

from ..support.orchestrator import code, llm_script, make_orch, run, stamp_report
from tests.support.v1_tasks import V1_TASK


def _eval(*, build_kind: str = "native", mutate=None):
    """An evaluator that stamps a native build, optionally mutating the report per call."""

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **kw):
        report = EvalReport(
            task=task, dtype=dtype, size=size, mode=mode, seed=seed or 1, stage="done",
            status="ok", message="ok", target_id="apple-m4-max", protocol_version=PROTOCOL_VERSION,
            backend="metal", backend_version="1",
        )
        report.speedup, report.speedups, report.candidate_us, report.baseline_us = 1.2, {"mlx": 1.2}, 100.0, {"mlx": 120.0}
        report.benchmark = {
            "candidate": {"median_us": 100.0, "cv": 0.01, "reps": reps},
            "baselines": {"mlx": {"median_us": 120.0, "cv": 0.01, "reps": reps}},
            "best_baseline": "mlx",
        }
        stamp_report(report, task, dtype, size, source, build_kind=build_kind)
        if mutate is not None:
            mutate(report, reps)
        return report

    return ev


def _run(tmp_path, evaluate_fn, *, batches=1):
    orch = make_orch(
        tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0, evaluate_fn=evaluate_fn,
        confirm_batches=batches,
    )
    return orch, run(orch)


def test_a_quiet_native_confirmation_records_identity_and_publishes(tmp_path):
    """合格确认写 expected/逐批身份，唯一谓词允许发布，读盘路径给出同一结论。"""
    orch, res = _run(tmp_path, _eval())
    assert res.confirmed_status == "ok"
    document = json.loads((orch.run_dir / "confirm_report.json").read_text())
    assert document["schema_version"] == 4
    confirmation = document["benchmark"]["confirmation"]
    assert confirmation["expected"]["build"]["candidate_sha256"] == res.confirmation_source_sha256
    assert confirmation["expected"]["protocol_version"] == PROTOCOL_VERSION
    assert confirmation["batches"][0]["identity"]["kind"] == "match"
    assert res.confirmations[0]["identity"]["kind"] == "match"
    assert res.confirmations[0]["identity"] == confirmation["batches"][0]["identity"]
    stored = json.loads((orch.run_dir / "result.json").read_text())
    assert conclusion_of_run(orch.run_dir, stored) == {"speedup": pytest.approx(1.2), "publication": "confirmed"}


def test_two_agreeing_batches_publish_with_two_identity_records(tmp_path):
    """两批都 match 时写两条身份记录并发布：一一对应不是「只准一批」。"""
    orch, res = _run(tmp_path, _eval(), batches=2)
    assert res.confirmed_status == "ok"
    assert len(res.confirmed_values) == 2
    document = json.loads((orch.run_dir / "confirm_report.json").read_text())
    confirmation = document["benchmark"]["confirmation"]
    assert len(confirmation["batches"]) == 2
    assert all(batch["identity"]["kind"] == "match" for batch in confirmation["batches"])
    assert conclusion_of_run(orch.run_dir, json.loads((orch.run_dir / "result.json").read_text()))["publication"] == "confirmed"


def test_a_simulator_confirmation_is_ineligible_and_writes_no_report(tmp_path):
    """simulator 的 full 报告也不能确认：记 confirmation_ineligible，不 append、不写 confirm_report。"""
    orch, res = _run(tmp_path, _eval(build_kind="simulator"))
    assert res.confirmed_status == "confirmation_ineligible"
    assert res.confirmed_values == [] and res.confirmation_source_sha256 is None
    assert not (orch.run_dir / "confirm_report.json").exists()
    entry = res.confirmations[0]
    assert entry["status"] == "confirmation_ineligible" and entry["failed_batches"] == 1
    assert entry["identity"]["build_kind"] == "simulator" and entry["identity"]["measurement_kind"] == "measured"
    recheck = json.loads((orch.run_dir / "rechecks" / f"confirm_{entry['label']}.json").read_text())
    # Eligibility is decided before identity, so the status is ``confirmation_ineligible``; the
    # record still keeps the full identity differences (a simulator build is not this executor's
    # build, so its provenance does not match the current expected).
    assert entry["identity"]["build_kind"] == "simulator"
    assert recheck["identity"]["kind"] == "build_identity_mismatch"


def test_a_cache_served_confirmation_is_ineligible(tmp_path):
    """确认期缓存命中不能当复测：confirmation_ineligible，不发布。"""
    def mutate(report, reps):
        report.cache_hit = True

    orch, res = _run(tmp_path, _eval(mutate=mutate))
    assert res.confirmed_status == "confirmation_ineligible"
    assert not (orch.run_dir / "confirm_report.json").exists()


def test_a_missing_evaluation_identity_is_unknown_not_a_rounding_issue(tmp_path):
    """评测身份为空时记 evaluation_identity_unknown，绝不当作相等。"""
    def mutate(report, reps):
        report.evaluation_fingerprint = None

    orch, res = _run(tmp_path, _eval(mutate=mutate))
    assert res.confirmed_status == "evaluation_identity_unknown"
    assert res.confirmations[0]["identity"]["differences"]
    assert not (orch.run_dir / "confirm_report.json").exists()


def test_a_noisy_confirmation_is_uncertain_and_publishes_nothing(tmp_path):
    """计时不稳单独定状态 timing_uncertain；候选不进 confirmed，也不写 confirm_report。"""
    def mutate(report, reps):
        if reps > 3:  # the confirmation pass uses eval_reps * 2
            report.benchmark["candidate"]["cv"] = 0.5

    orch, res = _run(tmp_path, _eval(mutate=mutate))
    assert res.confirmed_status == "timing_uncertain"
    assert res.confirmed_values == []
    assert not (orch.run_dir / "confirm_report.json").exists()
    assert res.confirmations[0]["identity"]["timing_uncertain"] is True


def test_a_different_search_report_is_a_mismatch_and_stops_remaining_batches(tmp_path):
    """搜索报告与确认报告的候选不同是 build_identity_mismatch；该候选剩余批次与后续候选停止。"""
    seen = []

    def mutate(report, reps):
        if reps > 3:
            report.build["assembled_sha256"] = "f" * 64

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **kw):
        seen.append(reps)
        return _eval(mutate=mutate)(task, dtype, size, source, mode=mode, seed=seed, reps=reps, **kw)

    orch, res = _run(tmp_path, ev, batches=3)
    assert res.confirmed_status == "build_identity_mismatch"
    assert len([r for r in seen if r > 3]) == 1  # only the first batch ran
    assert res.confirmations[0]["failed_batches"] == 3
    assert not (orch.run_dir / "confirm_report.json").exists()


def test_an_already_confirmed_candidate_still_wins_after_a_later_one_fails(tmp_path):
    """先完整确认的候选仍是 winner：后一颗候选身份失败只作废它自己，不推翻已确认的数字。

    复测池先放当前最优，再放噪声范围内的并列。这里最优（``init``）三批全部 match，并列那颗
    在确认报告里给出不同的 ``assembled_sha256``（``build_identity_mismatch``）；``stop=True`` 只停掉
    失败候选与它后面的候选，``confirmed`` 里已经完成的最优仍然可以被发布。
    """

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **kw):
        # ``init`` stays the best (the patch is 0.4% faster at most: below ``min_improvement``), and
        # the two are close enough to be remeasured together as a near-tie.
        speed = 1.20 if "speed=1.20" in source else 1.195
        report = EvalReport(
            task=task, dtype=dtype, size=size, mode=mode, seed=seed or 1, stage="done",
            status="ok", message="ok", target_id="apple-m4-max", protocol_version=PROTOCOL_VERSION,
            backend="metal", backend_version="1",
        )
        report.speedup, report.speedups = speed, {"mlx": speed}
        report.candidate_us, report.baseline_us = 100.0, {"mlx": 100.0 * speed}
        report.benchmark = {
            "candidate": {"median_us": 100.0, "cv": 0.01, "reps": reps},
            "baselines": {"mlx": {"median_us": 100.0, "cv": 0.01, "reps": reps}},
            "best_baseline": "mlx",
        }
        stamp_report(report, task, dtype, size, source)
        if reps > 3 and "speed=1.195" in source:
            # Only the confirmation batch of the near-tie disagrees with its own search report.
            report.build["assembled_sha256"] = "f" * 64
        return report

    orch = make_orch(
        tmp_path, llm_script(code("speed=1.20"), patches={"A": code("speed=1.195")}), [],
        branches=1, max_rounds=1, evaluate_fn=ev,
    )
    res = run(orch)
    assert [entry["label"] for entry in res.confirmations] == ["init", "r1.p0"]
    assert res.confirmations[0]["status"] == "ok"
    assert res.confirmations[0]["failed_batches"] == 0
    assert res.confirmations[0]["identity"]["kind"] == "match"
    assert res.confirmations[1]["status"] == "build_identity_mismatch"
    assert res.confirmations[1]["failed_batches"] == 3
    # The failed near-tie is not the winner: the candidate that finished its batches is.
    assert res.confirmed_status == "ok"
    assert res.confirm_winner_label == res.best_label == "init"
    document = json.loads((orch.run_dir / "confirm_report.json").read_text())
    assert document["benchmark"]["confirmation"]["winner_label"] == "init"
    assert conclusion_of_run(orch.run_dir, json.loads((orch.run_dir / "result.json").read_text()))["publication"] == "confirmed"


def test_a_drifted_expected_at_finalize_is_not_published(tmp_path):
    """批次用第一份 expected 比较；finalize 时重算若漂移，整颗候选不得发布。

    评审复现的路径是：每批比较前算一次 expected，选出 winner 后**再**算一次并直接冻结进
    `confirm_report`。第二次调用漂移时，旧批次仍记录 `match`（它们比的是第一份），报告却会以
    一份与自身身份矛盾的 expected 发布，读盘也继续确认。现在每颗候选只在第一批冻结一次，
    finalize 的重算只作检查：任何漂移都走同一组身份失败出口。
    """
    from tests.support.report_build import structural_expected

    calls = {"n": 0}

    def expected(*, task, dtype, size, target_id, source):
        calls["n"] += 1
        document = structural_expected(task=task, dtype=dtype, size=size, target_id=target_id, source=source)
        if calls["n"] > 1:
            document["evaluation"]["evaluation_fingerprint"] = "f" * 64
        return document

    orch = make_orch(
        tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0, evaluate_fn=_eval(),
        evaluator=LocalEvaluator(_eval(), expected_identity=expected),
    )
    res = run(orch)
    # One call froze the comparison expected, one call checked it at finalization.
    assert calls["n"] == 2
    assert res.confirmed_status == "evaluation_identity_mismatch"
    assert res.confirmations[0]["status"] == "evaluation_identity_mismatch"
    # The batch really did compare against the first (structural) expected, so its own record is a
    # match; what fails is freezing a *different* document next to it.
    assert res.confirmations[0]["identity"]["kind"] == "match"
    assert not (orch.run_dir / "confirm_report.json").exists()
    stored = json.loads((orch.run_dir / "result.json").read_text())
    assert stored["confirmed_status"] == "evaluation_identity_mismatch"
    assert stored["confirmations"][0]["identity"]["kind"] == "match"
    # The read-disk predicate must not publish it either.
    assert conclusion_of_run(orch.run_dir, stored) == {"speedup": None, "publication": "未确认"}


def test_a_failed_batch_is_not_blamed_on_the_kernel(tmp_path):
    """身份/资格/计时失败不归责候选：recheck 记录 kernel_error 为 False，run 不以 confirm_error 收尾。"""
    orch, res = _run(tmp_path, _eval(build_kind="simulator"))
    entry = res.confirmations[0]
    recheck = json.loads((orch.run_dir / "rechecks" / f"confirm_{entry['label']}.json").read_text())
    assert recheck["counts_as_kernel_error"] is False
    assert res.stop_reason == "max_rounds"


def test_an_unclassified_confirmation_failure_is_a_confirm_error(tmp_path):
    """§7.2：确认阶段的未分类异常必须走到 ``run()`` 的 ``confirm_error`` 收尾，不能变成自由文本批次状态。

    Before the fix the batch's ``except Exception`` wrote ``error: RuntimeError: ...`` into the entry
    status and stopped quietly, so the run finished as ``max_rounds`` and the CLI exited 0.  The
    exception now propagates: ``run()`` records ``confirm_error: <type>: <text>``, ``_finish`` writes
    result.json but no ``confirm_report.json``, and ``_result_code`` is 6.
    """
    from taic.cli import _result_code

    search_calls = {"full": 0}
    base = _eval()

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **kw):
        if mode == "full":
            search_calls["full"] += 1
            if search_calls["full"] > 1:  # the first full call is the search seed; this is the confirmation
                raise RuntimeError("device vanished during confirmation")
        return base(task, dtype, size, source, mode=mode, seed=seed, reps=reps, **kw)

    orch, res = _run(tmp_path, ev)
    assert res.stop_reason.startswith("confirm_error: RuntimeError:")
    assert res.confirmed_status is None
    stored = json.loads((orch.run_dir / "result.json").read_text())
    assert stored["stop_reason"] == res.stop_reason
    assert not (orch.run_dir / "confirm_report.json").exists()
    # No free-text batch status was invented for the failure.
    assert all(not str(entry.get("status") or "").startswith("error:") for entry in res.confirmations)
    assert _result_code(res) == 6


# -- the current expected is not the search report's self-report -----------------------------
def _changed_expected(change):
    from tests.support.report_build import structural_expected

    def expected(*, task, dtype, size, target_id, source):
        document = structural_expected(task=task, dtype=dtype, size=size, target_id=target_id, source=source)
        change(document)
        return document

    return expected


def test_a_registry_change_since_the_search_invalidates_two_identical_old_reports(tmp_path):
    """两份报告带着同一份旧 identity 时，当前 registry 已变化仍必须 mismatch（expected 不取自报告）。"""
    def change(document):
        document["evaluation"]["evaluation_fingerprint"] = "f" * 64

    orch = make_orch(
        tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0, evaluate_fn=_eval(),
        evaluator=LocalEvaluator(_eval(), expected_identity=_changed_expected(change)),
    )
    res = run(orch)
    assert res.confirmed_status == "evaluation_identity_mismatch"
    entry = res.confirmations[0]
    sides = {d["side"] for d in entry["identity"]["differences"] if d["field"] == "evaluation_fingerprint"}
    assert sides == {"search", "confirmation"}
    assert not (orch.run_dir / "confirm_report.json").exists()


def test_an_executor_profile_change_since_the_search_is_a_build_mismatch(tmp_path):
    """执行器 profile（toolchain）变化时，两份旧报告一致也算 build mismatch。"""
    def change(document):
        document["build"]["provenance"]["toolchain"]["version"] = "changed-toolchain"

    orch = make_orch(
        tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0, evaluate_fn=_eval(),
        evaluator=LocalEvaluator(_eval(), expected_identity=_changed_expected(change)),
    )
    res = run(orch)
    assert res.confirmed_status == "build_identity_mismatch"
    entry = res.confirmations[0]
    assert {d["side"] for d in entry["identity"]["differences"] if d["field"] == "provenance.toolchain.version"} == {"search", "confirmation"}


def test_the_default_local_executor_without_a_current_profile_is_unknown(tmp_path):
    """默认 LocalEvaluator（自定义 evaluate_fn，不探测）不提供当前 BuildProvenance：即使两份报告相同也只能 unknown。

    这正是生产默认路径的负例：expected 不完整时按 unknown 处理，绝不退化成“两份报告互相相等”。
    """
    from tests.support.orchestrator import fake_eval

    # A plain LocalEvaluator over a fake function: it cannot state today's toolchain profile.
    orch = make_orch(
        tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0,
        evaluate_fn=fake_eval([]), evaluator=LocalEvaluator(fake_eval([])),
    )
    res = run(orch)
    assert res.confirmed_status == "build_identity_unknown"
    fields = {d["field"] for d in res.confirmations[0]["identity"]["differences"] if d["domain"] == "build"}
    assert "provenance.builder_id" in fields and "provenance.toolchain.version" in fields
    assert not (orch.run_dir / "confirm_report.json").exists()


def _probe_profile(math: str):
    from tcenv.remote.artifact_builder import BuildProfile

    return BuildProfile(
        build_profile="r5-probe-fixture", builder_id="r5-probe-fixture", builder_version="0",
        build_host="control_host", toolchain_id="apple-metal", toolchain_version="4.0",
        sdk_version="macos-test",
        compile_profile={"math": math, "device": "Fixture GPU", "gpu_family": "FixtureFamily"},
        build_kind="native",
    )


def _safe_launch_source() -> str:
    return (
        "/* taic-launch\nmath safe\nlaunch relu_kernel grid=1,1,1 threads=32,1,1\n*/\n"
        "kernel void relu_kernel() {}\n"
    )


def test_the_production_expected_uses_launch_math_and_available_probe_device(monkeypatch):
    """真实默认路径的 expected 必须含同一 safe math 与可用 probe 的设备身份。"""
    from tcenv.evaluate import confirmation_expected_document
    from tcenv.remote.artifact_builder import BuildCapabilities

    calls = []

    class Builder:
        def probe(self):
            return BuildCapabilities(profile=_probe_profile("safe"), available=True)

    backend = __import__("tcenv.backends.registry", fromlist=["load_backend"]).load_backend("metal")

    def factory(**kwargs):
        calls.append(kwargs)
        return Builder()

    monkeypatch.setattr(backend, "artifact_builder", factory)
    monkeypatch.setattr("tcenv.backends.registry.load_backend", lambda name: backend)
    expected = confirmation_expected_document(
        V1_TASK, "fp16", "small", "apple-m4-max", _safe_launch_source(), probe=True,
    )
    assert calls[0]["math"] == "safe"
    assert expected["build"]["provenance"]["compile_profile"]["math"] == "safe"
    assert expected["evaluation"]["device_name"] == "Fixture GPU"
    assert expected["evaluation"]["gpu_family"] == "FixtureFamily"


def test_an_unavailable_probe_leaves_provenance_and_device_identity_unknown(monkeypatch):
    """probe 失败的 fallback 不得冒充当前工具链或设备身份。"""
    from tcenv.evaluate import confirmation_expected_document
    from tcenv.remote.artifact_builder import BuildCapabilities

    class Builder:
        def probe(self):
            return BuildCapabilities(
                profile=_probe_profile("safe"), available=False, detail="no device",
            )

    backend = __import__("tcenv.backends.registry", fromlist=["load_backend"]).load_backend("metal")
    monkeypatch.setattr(backend, "artifact_builder", lambda **kwargs: Builder())
    monkeypatch.setattr("tcenv.backends.registry.load_backend", lambda name: backend)
    expected = confirmation_expected_document(
        V1_TASK, "fp16", "small", "apple-m4-max", _safe_launch_source(), probe=True,
    )
    assert expected["build"]["provenance"] is None
    assert expected["evaluation"]["device_name"] is None
    assert expected["evaluation"]["gpu_family"] is None


# -- a confirmation that cannot be evaluated is its own outcome, not free text ----------------
def test_an_unavailable_confirmation_batch_is_not_free_text_and_stops_the_rest(tmp_path):
    """`EvaluationUnavailable` 归一为 `evaluation_unavailable`：不写 `error: ...`、不 append、不写报告。

    §7.2: the batch status is the closed-set outcome, the remaining batches are not measured, the
    kernel is not blamed (no recheck record, ``kernel_error`` false) and nothing is published.
    """
    from tcenv.evaluation import EvaluationUnavailable

    base = _eval()
    full_calls = []

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=3, **kw):
        # The confirmation batches double the repetitions; the search evaluation does not, so this
        # raises only once the confirmation phase starts.
        if reps == 6:
            full_calls.append(mode)
            raise EvaluationUnavailable("result_invalid", "damaged confirmation result")
        return base(task, dtype, size, source, mode=mode, seed=seed, reps=reps, **kw)

    orch, res = _run(tmp_path, ev, batches=2)
    entry = res.confirmations[0]
    assert entry["status"] == "evaluation_unavailable"
    assert not entry["status"].startswith("error:"), "the closed-set outcome is not free text"
    # Both batches were still owed; neither the second one nor anything else was measured.
    assert entry["failed_batches"] == 2 and len(full_calls) == 1
    assert res.confirmed_status == "evaluation_unavailable"
    assert res.confirmed_values == [] and res.confirmation_source_sha256 is None
    assert not (orch.run_dir / "confirm_report.json").exists()
    assert not (orch.run_dir / "rechecks").exists(), "an unavailable evaluation is not a kernel failure"


@pytest.mark.parametrize("error", ["SchemaError", "ContractError"])
def test_structural_damage_during_confirmation_is_result_invalid(tmp_path, error):
    """结构损坏（报告/回执/HMAC 解析失败）包成 `EvaluationUnavailable("result_invalid")`，同一条出口。"""
    from tcenv.remote.contracts import ContractError
    from tcenv.schema import SchemaError

    base = _eval()
    raised = SchemaError if error == "SchemaError" else ContractError

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=3, **kw):
        if reps == 6:
            raise raised("damaged confirmation result")
        return base(task, dtype, size, source, mode=mode, seed=seed, reps=reps, **kw)

    orch, res = _run(tmp_path, ev, batches=1)
    entry = res.confirmations[0]
    assert entry["status"] == "evaluation_unavailable"
    assert res.confirmed_status == "evaluation_unavailable"
    assert not (orch.run_dir / "confirm_report.json").exists()


def test_a_confirmation_context_that_needs_inspection_ends_the_run_as_confirm_error(tmp_path):
    """`NeedsInspection` 继续上抛：run() 记 `confirm_error: needs_inspection`，写 result、不写报告。"""
    from taic.artifacts import NeedsInspection

    base = _eval()

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=3, **kw):
        if reps == 6:
            raise NeedsInspection("evaluation_context disagrees with the persisted row")
        return base(task, dtype, size, source, mode=mode, seed=seed, reps=reps, **kw)

    orch, res = _run(tmp_path, ev, batches=1)
    assert res.stop_reason.startswith("confirm_error") and "needs_inspection" in res.stop_reason
    assert res.confirmed_status == "needs_inspection"
    assert res.best_speedup == pytest.approx(1.2)
    assert (orch.run_dir / "result.json").is_file()
    assert not (orch.run_dir / "confirm_report.json").exists()
    from taic.cli import _result_code

    assert _result_code(res) == 6
