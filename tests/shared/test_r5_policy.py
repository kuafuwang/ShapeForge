"""R5: the pure confirmation-identity comparator and the single publication predicate."""

from __future__ import annotations

import pytest

from taic.policy import (
    PUBLICATION_REASONS, build_identity_view, compare_confirmation_identity,
    confirmation_publication, evaluation_identity_view,
)
from tests.support.v1_tasks import V1_TASK


def _view(*, candidate="a" * 64, provenance=None, assembled="b" * 64, native="d" * 64, **build):
    from tests.support.report_build import report_for_request

    report = report_for_request(
        {"task": V1_TASK, "dtype": "fp16", "size": "small", "mode": "full", "seed": 1},
        build_kind="native",
    )
    report["build"].update(build)
    report["build"]["candidate_sha256"] = candidate
    report["build"]["assembled_sha256"] = assembled
    if provenance is not None:
        report["build"]["provenance"] = provenance
    if native is None:
        report["build"].pop("native_digest", None)
        report["build"]["native_absent_reason"] = "portable_artifact_set"
    return report


def test_a_matching_pair_is_a_match():
    """搜索/确认报告的 build 与 evaluation 身份相同且等于 expected 时是 match。"""
    search = _view()
    confirmation = _view()
    got = compare_confirmation_identity(scope="local", candidate_sha256="a" * 64, expected={
        "build": build_identity_view(search), "evaluation": evaluation_identity_view(search),
    }, search=search, confirmation=confirmation)
    assert got["kind"] == "match"
    assert got["differences"] == []


def test_a_different_candidate_is_a_build_mismatch_attributed_to_the_report():
    """候选不同是 mismatch，并指出是哪一侧（不把 expected 当 unknown）。"""
    search = _view()
    confirmation = _view(candidate="c" * 64)
    got = compare_confirmation_identity(scope="local", candidate_sha256="a" * 64, expected={
        "build": build_identity_view(search), "evaluation": evaluation_identity_view(search),
    }, search=search, confirmation=confirmation)
    assert got["kind"] == "build_identity_mismatch"
    hit = [d for d in got["differences"] if d["field"] == "candidate_sha256" and d["kind"] == "mismatch"]
    assert hit and hit[0]["side"] == "confirmation"


def test_a_missing_reported_field_is_unknown_not_equal():
    """一边有、一边没有是 unknown：不得当作相等，也不得当作明确不一致。"""
    search = _view()
    confirmation = _view(native=None)
    got = compare_confirmation_identity(scope="local", candidate_sha256="a" * 64, expected={
        "build": build_identity_view(search), "evaluation": evaluation_identity_view(search),
    }, search=search, confirmation=confirmation)
    assert got["kind"] == "build_identity_unknown"
    assert any(d["field"] == "native_digest" and d["kind"] == "unknown" for d in got["differences"])


def test_unknown_expected_is_unknown_even_when_both_reports_agree():
    """可重算的 expected 自己不完整时，即使两边报告一致也是 unknown。"""
    search = _view()
    got = compare_confirmation_identity(scope="local", candidate_sha256="a" * 64, expected={
        "build": build_identity_view(search),
        "evaluation": evaluation_identity_view(search) | {"evaluation_fingerprint": None},
    }, search=search, confirmation=_view())
    assert got["kind"] == "evaluation_identity_unknown"


def test_outcome_only_build_fields_are_compared_between_the_two_periods():
    """assembled/artifact/native 摘要没有当前预期：两份报告互相比较，不拿 search 值当预期。"""
    search = _view()
    confirmation = _view(assembled="e" * 64)
    got = compare_confirmation_identity(scope="local", candidate_sha256="a" * 64, expected={
        "build": build_identity_view(search),
        "evaluation": evaluation_identity_view(search),
    }, search=search, confirmation=confirmation)
    assert got["kind"] == "build_identity_mismatch"
    hits = [d for d in got["differences"] if d["field"] == "assembled_sha256"]
    assert {d["side"] for d in hits} == {"search", "confirmation"}


def test_mismatch_outranks_unknown_and_build_outranks_evaluation():
    """主状态按 mismatch 先于 unknown、build 先于 evaluation 确定，differences 全部保留。"""
    search = _view()
    broken = _view(candidate="c" * 64)
    broken["evaluation_fingerprint"] = "e" * 64
    got = compare_confirmation_identity(scope="local", candidate_sha256="a" * 64, expected={
        "build": build_identity_view(search), "evaluation": evaluation_identity_view(search),
    }, search=search, confirmation=broken)
    assert got["kind"] == "build_identity_mismatch"
    assert {d["domain"] for d in got["differences"]} == {"build", "evaluation"}
    assert got["differences"] == sorted(got["differences"], key=lambda d: (d["domain"], d["field"], d["kind"], d["side"]))


def _batch(**over):
    batch = {
        "label": "w", "shape": "full", "build_kind": "native", "measurement_kind": "measured",
        "cache_hit": False, "timing_uncertain": False, "identity": {"kind": "match", "differences": []},
    }
    batch.update(over)
    return batch


def _batches(**over):
    """两个 confirmed value 就对应两条批次记录：谓词按 §7.2 要求两者一一对应。"""
    return [dict(_batch(**over), identity=dict(_batch(**over)["identity"])) for _value in (1.2, 1.19)]


def _result(**over):
    value = {
        "confirmed_status": "ok", "confirmation_source_sha256": "a" * 64,
        "confirm_winner_label": "w", "confirmed_speedup": 1.195, "confirmed_values": [1.2, 1.19],
        "candidate_us": 10.0, "target_id": "apple",
        # ``confirmations`` 是**每个候选一条**冻结记录，不是每批一条：三条批记录仍只有一条。
        "confirmations": [dict(_batch(), status="ok", failed_batches=0)],
    }
    value.update(over)
    return value


#: The identity a publishable confirmation must carry on both sides (§7.2): the report states it
#: and the frozen expected repeats it.  ``candidate_sha256`` is the winner's source hash.
_IDENTITY = {
    "evaluation_fingerprint": "1" * 64,
    "baseline_implementation_digest": "2" * 64,
    "baseline_environment_digest": "3" * 64,
    "backend": "metal",
    "backend_version": "1",
    # Required identity on both sides (§7.2): the top-level pair and the ``evaluation``
    # projection must both state them, so ``None == None`` can never count as agreement.
    "protocol_version": "3", "target_id": "apple",
}
#: A complete ``BuildProvenance`` (§7.2): the report states it in full, the frozen expected repeats
#: every recomputable path (``artifact_set_digest`` is outcome-only and never required from it).
def _provenance(**over) -> dict:
    """A fresh complete ``BuildProvenance``; never share the nested objects between documents."""
    value = {
        "schema_id": "npu.build.provenance.v1",
        "builder_id": "fixture-builder", "builder_version": "1", "build_host": "control_host",
        "build_profile": "fixture", "toolchain": {"id": "fixture", "version": "1", "sdk_version": ""},
        "compile_profile": {"compiled": False}, "source_sha256": "a" * 64,
    }
    value.update(over)
    return value


_BUILD_IDENTITY = {
    "shape": "full", "build_kind": "native", "candidate_sha256": "a" * 64,
    "opspec_digest": "4" * 64, "lowering_profile_digest": "5" * 64,
    "provenance": _provenance(artifact_set_digest="6" * 64),
}


def _confirm(**over):
    value = {
        # A real confirm report states ``cache_hit``: the top-level field is required by §7.2.
        "cache_hit": False, "status": "ok", "speedup": 1.195, "protocol_version": "3", "target_id": "apple",
        **_IDENTITY,
        "build": dict(_BUILD_IDENTITY),
        "benchmark": {"candidate": {"cv": 0.01, "reps": 5}, "best_baseline": "x",
                      "baselines": {"x": {"cv": 0.01, "reps": 5}}, "confirmation": {
            "kernel_sha256": "a" * 64, "winner_label": "w", "confirmed_values": [1.2, 1.19],
            "speedup": 1.195, "candidate_us": 10.0,
        }},
    }
    value.update(over)
    return value


def _expected():
    """A complete frozen expected: the same identity the report states.

    An expected that omits a required identity is ``unknown`` at comparison time, so a fixture that
    wants to exercise *other* branches must start from a complete document.
    """
    return {
        "protocol_version": "3", "target_id": "apple",
        "build": {
            "candidate_sha256": "a" * 64, "opspec_digest": "4" * 64, "lowering_profile_digest": "5" * 64,
            "provenance": _provenance(),
        },
        "evaluation": dict(_IDENTITY),
    }


def _bindings(**over):
    value = {"confirmation_source_sha256": "a" * 64, "confirm_report_status": "ok",
             "best_source_sha256": "a" * 64, "winner_source_sha256": "a" * 64}
    value.update(over)
    return value


def test_a_complete_confirmation_is_published():
    """全部批次 native+measured+fresh 且身份 match 时唯一谓词允许发布。"""
    got = confirmation_publication(
        result=_result(), confirm_report=_confirm(), batches=_batches(), expected=_expected(), bindings=_bindings(),
    )
    assert got == {"allowed": True, "status": "confirmed", "reason": "confirmed"}


def test_a_failed_winner_entry_cannot_be_overruled_by_the_top_level_status():
    """顶层 ok 不能推翻胜者自己的失败记录；批次 identity 相同也不能掩盖矛盾。"""
    result = _result()
    result["confirmations"][0].update(status="build_identity_mismatch", failed_batches=2)
    got = confirmation_publication(
        result=result, confirm_report=_confirm(), batches=_batches(), expected=_expected(), bindings=_bindings(),
    )
    assert got == {"allowed": False, "status": "未确认", "reason": "build_identity_mismatch"}

    result["confirmations"][0].update(status="ok", failed_batches=1)
    got = confirmation_publication(
        result=result, confirm_report=_confirm(), batches=_batches(), expected=_expected(), bindings=_bindings(),
    )
    assert got == {"allowed": False, "status": "未确认", "reason": "not_confirmed"}

    result["confirmations"][0]["failed_batches"] = 0
    assert confirmation_publication(
        result=result, confirm_report=_confirm(), batches=_batches(), expected=_expected(), bindings=_bindings(),
    )["allowed"] is True


@pytest.mark.parametrize("field,value,reason,status", [
    ("build_kind", "simulator", "confirmation_ineligible", "未确认"),
    ("measurement_kind", "simulated", "confirmation_ineligible", "未确认"),
    ("cache_hit", True, "confirmation_ineligible", "未确认"),
    ("shape", "partial", "build_identity_unknown", "未确认"),
    ("timing_uncertain", True, "timing_uncertain", "不确定"),
])
def test_a_batch_must_be_a_fresh_native_measurement(field, value, reason, status):
    """非 native、非 measured、缓存命中、partial、计时不稳都不发布。"""
    got = confirmation_publication(
        result=_result(), confirm_report=_confirm(), batches=_batches(**{field: value}),
        expected=_expected(), bindings=_bindings(),
    )
    assert (got["allowed"], got["reason"], got["status"]) == (False, reason, status)


@pytest.mark.parametrize("kind", ["build_identity_mismatch", "build_identity_unknown",
                                  "evaluation_identity_mismatch", "evaluation_identity_unknown"])
def test_a_batch_identity_failure_blocks_publication(kind):
    """任一批次身份不是 match 就不发布，原因码就是该批主状态。"""
    batches = _batches(identity={"kind": kind, "differences": [{"domain": "build", "field": "candidate_sha256", "kind": "mismatch", "side": "search"}]})
    result = _result(confirmations=[dict(batches[0], status="ok", failed_batches=0)])
    got = confirmation_publication(
        result=result, confirm_report=_confirm(), batches=batches, expected=_expected(), bindings=_bindings(),
    )
    assert got["allowed"] is False and got["reason"] == kind


@pytest.mark.parametrize("missing", ["absent", None, "false", 0, 1])
def test_every_batch_must_record_timing_uncertain_as_a_boolean(missing):
    """每批的 ``timing_uncertain`` 必须严格是布尔 ``False``：缺字段/None/非布尔都算记录不全。

    The reviewer's hole: only an explicit ``True`` was refused, so a batch with the required field
    missing (or ``null``, or a string) still published.  ``False`` is the only value that says "the
    timing was stable".
    """
    batches = _batches()
    for batch in batches:
        if missing == "absent":
            batch.pop("timing_uncertain", None)
        else:
            batch["timing_uncertain"] = missing
    got = confirmation_publication(
        result=_result(), confirm_report=_confirm(), batches=batches,
        expected=_expected(), bindings=_bindings(),
    )
    assert got == {"allowed": False, "status": "未确认", "reason": "not_confirmed"}


@pytest.mark.parametrize("missing", ["absent", None, "false", 0, 1])
def test_the_confirm_report_must_state_cache_hit_as_a_boolean(missing):
    """顶层 ``cache_hit`` 必须严格是布尔 ``False``：缺失/null/字符串/数值都算记录不全。

    The reviewer's hole: only an explicit ``True`` was refused, so a confirm report without the
    field (or with ``None``/``"false"``/``0``/``1``) still published.  A per-batch ``False`` does not
    fill in this required top-level fact.
    """
    report = _confirm()
    if missing == "absent":
        report.pop("cache_hit", None)
    else:
        report["cache_hit"] = missing
    got = confirmation_publication(
        result=_result(), confirm_report=report, batches=_batches(),
        expected=_expected(), bindings=_bindings(),
    )
    assert got == {"allowed": False, "status": "未确认", "reason": "not_confirmed"}


def test_a_cached_confirm_report_is_uncertain():
    """显式 ``True`` 仍是「不确定/cache_hit」。"""
    got = confirmation_publication(
        result=_result(), confirm_report=_confirm(cache_hit=True), batches=_batches(),
        expected=_expected(), bindings=_bindings(),
    )
    assert got == {"allowed": False, "status": "不确定", "reason": "cache_hit"}


def test_an_explicitly_unstable_batch_stays_uncertain():
    """显式 ``True`` 仍是「不确定/timing_uncertain」，不降级成「记录不全」。"""
    got = confirmation_publication(
        result=_result(), confirm_report=_confirm(), batches=_batches(timing_uncertain=True),
        expected=_expected(), bindings=_bindings(),
    )
    assert got == {"allowed": False, "status": "不确定", "reason": "timing_uncertain"}


def test_two_confirmed_values_need_two_identity_records():
    """一条身份记录不能为两个测量值背书：批次数与 confirmed values 必须一一对应。"""
    result = _result()
    got = confirmation_publication(
        result=result, confirm_report=_confirm(), batches=_batches()[:1],
        expected=_expected(), bindings=_bindings(),
    )
    assert got == {"allowed": False, "status": "未确认", "reason": "not_confirmed"}


def test_a_foreign_batch_cannot_stand_in_for_a_missing_winner_batch():
    """条数够了但有一条属于别的候选：winner 的批次记录仍然缺，不发布。"""
    batches = [_batches()[0], _batch(label="other")]
    result = _result(confirmations=[dict(batch, status="ok", failed_batches=0) for batch in batches])
    got = confirmation_publication(
        result=result, confirm_report=_confirm(), batches=batches, expected=_expected(), bindings=_bindings(),
    )
    assert got["allowed"] is False and got["reason"] == "not_confirmed"


def test_a_duplicated_frozen_winner_record_is_refused():
    """result.confirmations 里胜者出现两次时不能靠「最后一条覆盖前一条」混过去。"""
    frozen = _result()["confirmations"]
    got = confirmation_publication(
        result=_result(confirmations=frozen + [dict(frozen[0])]), confirm_report=_confirm(),
        batches=_batches(), expected=_expected(), bindings=_bindings(),
    )
    assert got["allowed"] is False and got["reason"] == "not_confirmed"


def test_the_two_batch_records_must_agree():
    """result.confirmations 与传入批次记录不一致时不发布（两处记录必须相同）。"""
    got = confirmation_publication(
        result=_result(confirmations=[]), confirm_report=_confirm(), batches=_batches(),
        expected=_expected(), bindings=_bindings(),
    )
    assert got["allowed"] is False and got["reason"] == "not_confirmed"


def test_a_missing_frozen_expected_cannot_be_replaced_by_the_current_environment():
    """冻结 expected 缺失时不发布，也不拿当前环境补值。"""
    got = confirmation_publication(
        result=_result(), confirm_report=_confirm(), batches=_batches(), expected=None, bindings=_bindings(),
    )
    assert got["allowed"] is False and got["reason"] == "not_confirmed"


def test_protocol_and_target_compare_against_the_frozen_expected():
    """protocol/target 只与冻结 expected 比较，不读当前常量。"""
    expected = _expected() | {"protocol_version": "2"}
    got = confirmation_publication(
        result=_result(), confirm_report=_confirm(), batches=_batches(), expected=expected, bindings=_bindings(),
    )
    assert got["reason"] == "protocol_mismatch"


def test_the_frozen_expected_must_describe_the_report_it_bound():
    """§7.2：冻结 expected 必须与报告一致**且完整**；缺必需身份是 unknown，不是可发布。

    Two bypasses are pinned here:

    * a *contradiction* — the frozen copy drifted away from the report it claims to bind while the
      batch records still say ``match`` (``*_identity_mismatch``);
    * an *incomplete* expected — ``evaluation_fingerprint``/baseline digests ``None`` (or a required
      build field absent).  The comparator already calls that ``*_identity_unknown``; the predicate
      must not accept a batch record claiming ``match`` against a document that never stated it.
      Missing fields are never filled from today's environment.

    Optional fields keep their semantics: ``device_name``/``gpu_family`` may stay absent, and the
    outcome-only build digests are never required from an expected.
    """

    def decide(expected):
        return confirmation_publication(
            result=_result(), confirm_report=_confirm(), batches=_batches(),
            expected=expected, bindings=_bindings(),
        )

    # Complete + consistent → publishable.
    assert decide(_expected()) == {"allowed": True, "status": "confirmed", "reason": "confirmed"}

    # A drifted evaluation identity is a mismatch, and the build one uses the build code.
    drifted = _expected()
    drifted["evaluation"]["evaluation_fingerprint"] = "f" * 64
    assert decide(drifted) == {"allowed": False, "status": "未确认", "reason": "evaluation_identity_mismatch"}
    drifted_build = _expected()
    drifted_build["build"]["opspec_digest"] = "f" * 64
    assert decide(drifted_build) == {"allowed": False, "status": "未确认", "reason": "build_identity_mismatch"}

    # The reviewer's repro: a frozen expected without the evaluation identity still published
    # because the batch record said ``match``.  It is ``evaluation_identity_unknown`` now.
    for field in ("evaluation_fingerprint", "baseline_implementation_digest", "baseline_environment_digest"):
        incomplete = _expected()
        incomplete["evaluation"][field] = None
        assert decide(incomplete) == {"allowed": False, "status": "未确认", "reason": "evaluation_identity_unknown"}
        missing = _expected()
        missing["evaluation"].pop(field)
        assert decide(missing) == {"allowed": False, "status": "未确认", "reason": "evaluation_identity_unknown"}
    both_missing = _expected()
    both_missing["evaluation"]["evaluation_fingerprint"] = None
    both_missing["evaluation"]["baseline_environment_digest"] = None
    assert decide(both_missing) == {"allowed": False, "status": "未确认", "reason": "evaluation_identity_unknown"}

    # A required build identity the expected does not state is a build unknown.
    for field in ("candidate_sha256", "opspec_digest", "lowering_profile_digest"):
        incomplete = _expected()
        incomplete["build"][field] = None
        assert decide(incomplete) == {"allowed": False, "status": "未确认", "reason": "build_identity_unknown"}
        missing = _expected()
        missing["build"].pop(field)
        assert decide(missing) == {"allowed": False, "status": "未确认", "reason": "build_identity_unknown"}

    # Provenance the report states must be stated by the expected too; the outcome-only artifact
    # digest is exempt (it is compared between the two periods, never required from an expected).
    report = _confirm()
    for path in ("builder_id", "toolchain.sdk_version"):
        incomplete = _expected()
        head, _, leaf = path.partition(".")
        if leaf:
            incomplete["build"]["provenance"][head].pop(leaf)
        else:
            incomplete["build"]["provenance"].pop(head)
        assert confirmation_publication(
            result=_result(), confirm_report=report, batches=_batches(),
            expected=incomplete, bindings=_bindings(),
        ) == {"allowed": False, "status": "未确认", "reason": "build_identity_unknown"}

    # The reviewer's repro: deleting the provenance from *both* documents left the union empty, so
    # nothing was compared.  An absent report provenance is a build unknown, never agreement.
    bare_report = _confirm(build={key: value for key, value in _BUILD_IDENTITY.items() if key != "provenance"})
    bare_expected = _expected()
    bare_expected["build"].pop("provenance", None)
    assert confirmation_publication(
        result=_result(), confirm_report=bare_report, batches=_batches(),
        expected=bare_expected, bindings=_bindings(),
    ) == {"allowed": False, "status": "未确认", "reason": "build_identity_unknown"}
    assert confirmation_publication(
        result=_result(), confirm_report=_confirm(), batches=_batches(),
        expected=bare_expected, bindings=_bindings(),
    ) == {"allowed": False, "status": "未确认", "reason": "build_identity_unknown"}

    # Optional evaluation fields stay optional.
    optional = _expected()
    optional["evaluation"]["device_name"] = None
    optional["evaluation"].pop("gpu_family", None)
    assert decide(optional)["allowed"] is True


def test_the_report_must_be_the_full_native_measurement_of_the_winner():
    """§7.2：确认报告本体必须是**胜者**的 full/native 测量，不能只靠批次记录自称 native。

    The reviewer rewrote ``build.candidate_sha256`` in the report *and* the frozen expected to
    ``"b"*64`` while ``best.metal``/the winner file stayed the measured source; the predicate only
    compared the confirmation block's hash, so it still published.  The report's own build must name
    the recorded source, and a partial or simulator build is ineligible whatever the batches say.
    """

    def decide(report, extra=None):
        expected = _expected()
        if extra:
            expected["build"].update(extra)
        return confirmation_publication(
            result=_result(), confirm_report=report, batches=_batches(),
            expected=expected, bindings=_bindings(),
        )

    # Another candidate, stated consistently on both sides, is still not the measured winner.
    both = {"candidate_sha256": "b" * 64}
    assert decide(_confirm(build={**_BUILD_IDENTITY, **both}), both) == {
        "allowed": False, "status": "未确认", "reason": "sha_mismatch",
    }
    # The report body must be a fresh, full, native measurement of its own.
    for build in (
        {**_BUILD_IDENTITY, "shape": "partial"},
        {**_BUILD_IDENTITY, "build_kind": "simulator"},
        {**_BUILD_IDENTITY, "build_kind": "source_only"},
    ):
        assert decide(_confirm(build=build))["reason"] == "confirmation_ineligible"
    assert decide(_confirm(status="incorrect"))["reason"] == "confirmation_ineligible"


def test_the_results_target_must_match_the_frozen_identity():
    """§7.2：``result.target_id`` 也是发布身份；与冻结 expected/报告不同或缺字段都不发布。"""
    for target in ("other", None, ""):
        got = confirmation_publication(
            result=_result(target_id=target), confirm_report=_confirm(), batches=_batches(),
            expected=_expected(), bindings=_bindings(),
        )
        assert got["allowed"] is False
        assert got["reason"] == ("target_mismatch" if isinstance(target, str) and target else "evaluation_identity_unknown")


def test_protocol_and_target_must_be_stated_on_both_sides():
    """§7.2/F-28：protocol/target 是必需身份，两边同时缺失或为 ``None`` 不得算一致。

    The predicate used a plain ``!=`` comparison, so ``None == None`` published a document whose
    report *and* frozen expected had both dropped the top-level pair.  Missing/None on either side
    (or in the ``evaluation`` projection) is ``evaluation_identity_unknown``; only two stated,
    different values is ``protocol_mismatch``/``target_mismatch``.
    """

    def decide(report, expected):
        return confirmation_publication(
            result=_result(), confirm_report=report, batches=_batches(),
            expected=expected, bindings=_bindings(),
        )

    assert decide(_confirm(), _expected())["allowed"] is True

    # The reviewer's repro: remove the pair from the report and from both expected projections.
    report, expected = _confirm(), _expected()
    for field in ("protocol_version", "target_id"):
        report.pop(field)
        expected.pop(field)
        expected["evaluation"].pop(field)
    assert decide(report, expected) == {
        "allowed": False, "status": "未确认", "reason": "evaluation_identity_unknown",
    }

    # Explicit None on both sides is not agreement either.
    report, expected = _confirm(protocol_version=None, target_id=None), _expected()
    for field in ("protocol_version", "target_id"):
        expected[field] = None
        expected["evaluation"][field] = None
    assert decide(report, expected) == {
        "allowed": False, "status": "未确认", "reason": "evaluation_identity_unknown",
    }

    # One-sided presence is unknown in both directions...
    for field in ("protocol_version", "target_id"):
        expected = _expected()
        expected.pop(field)
        assert decide(_confirm(), expected) == {
            "allowed": False, "status": "未确认", "reason": "evaluation_identity_unknown",
        }
        report = _confirm()
        report.pop(field)
        assert decide(report, _expected()) == {
            "allowed": False, "status": "未确认", "reason": "evaluation_identity_unknown",
        }
        # ...and so is an ``evaluation`` projection that omits one of them.
        expected = _expected()
        expected["evaluation"].pop(field)
        assert decide(_confirm(), expected) == {
            "allowed": False, "status": "未确认", "reason": "evaluation_identity_unknown",
        }

    # Two stated, different values keep their dedicated reason codes.
    expected = _expected()
    expected["protocol_version"] = "4"
    assert decide(_confirm(), expected)["reason"] == "protocol_mismatch"
    expected = _expected()
    expected["target_id"] = "other"
    assert decide(_confirm(), expected)["reason"] == "target_mismatch"


def test_every_reason_is_in_the_documented_closed_set():
    """原因码闭集完整；谓词从不返回自由文本。"""
    assert "confirmed" in PUBLICATION_REASONS
    assert set(PUBLICATION_REASONS) >= {
        "cache_hit", "timing_uncertain", "batches_disagree", "protocol_mismatch",
        "target_mismatch", "confirmation_ineligible",
    }


def test_an_absent_current_provenance_is_unknown_not_a_two_period_match():
    """expected 没有当前 provenance 时，两份完全相同的报告仍是 build_identity_unknown。"""
    search = _view()
    expected = {
        "build": dict(build_identity_view(search), provenance=None),
        "evaluation": evaluation_identity_view(search),
    }
    report = {**search, **evaluation_identity_view(search)}
    got = compare_confirmation_identity(
        scope="local", candidate_sha256="a" * 64, expected=expected, search=report, confirmation=report,
    )
    assert got["kind"] == "build_identity_unknown"
    assert [d for d in got["differences"] if d["domain"] == "build" and d["kind"] == "unknown"]


def test_only_the_three_outcome_digests_use_a_two_period_comparison():
    """assembled/artifact/native 是 outcome-only；provenance 不是（缺当前预期必须 unknown）。"""
    from taic.policy import OUTCOME_ONLY_BUILD_FIELDS

    assert OUTCOME_ONLY_BUILD_FIELDS == frozenset({"assembled_sha256", "artifact_set_digest", "native_digest"})
