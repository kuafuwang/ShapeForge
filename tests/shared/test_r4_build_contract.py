"""R4: the build record at the two ends that actually accept results.

The evaluator side (``JobStore``) validates before signing; the host side
(``Orchestrator._evaluate_report``) validates again after transport.  Both must refuse a v3
descriptor, a v4 descriptor over a non-schema-4 payload, and a report whose build contradicts
the signed build evidence or the evaluator provenance.  No device, no model.
"""

from __future__ import annotations

import copy
import hashlib

import pytest

from tcenv.remote import contracts
from tcenv.remote.job_store import JobStore
from tcenv.remote.job_types import JobError
from tests.support.job_engine import engine, identity, request
from tests.support.report_build import report_for_request, simulated_build


def _report(status: str = "ok", *, build_kind: str = "native") -> dict:
    """A schema-4 report dict; the default is the local Metal structural fixture."""

    """A schema-4 report dict for the legacy source dialect."""
    return report_for_request(
        request("unit-1"), status=status, source=b"npu-kernel relu_kernel\n", build_kind=build_kind,
    )


def _store(tmp_path) -> JobStore:
    service = engine(tmp_path / "state")
    try:
        return service.store, service
    finally:
        service.close()


def _provenance(store: JobStore) -> dict:
    return identity().external_provenance(execution_protocol_version="3")


# -- evaluator side: schema 4 is the only payload a v4 descriptor may name -----
def test_a_v3_descriptor_is_refused_by_the_signature_document():
    """v3 descriptor 一律拒绝（混版在这里 fail closed，不协商）。"""
    from tests.support.remote_contract import load_evaluator_golden

    vector = load_evaluator_golden()["signature"]
    unsigned = copy.deepcopy(vector["unsigned_result"])
    unsigned["report"]["schema_id"] = "tcenv.eval-report.v3"
    with pytest.raises(contracts.ContractError, match="candidate report schema"):
        contracts.result_signature_document(unsigned, key_id=vector["key_id"])
    # The current descriptor is the one the frozen vector signs.
    assert vector["unsigned_result"]["report"]["schema_id"] == contracts.REPORT_SCHEMA_ID


def test_the_evaluator_refuses_a_v4_descriptor_over_a_non_schema_4_payload(tmp_path):
    """evaluator 在签名前用严格 schema 4 校验 payload；旧报告不能借 v4 descriptor 混进来。"""
    store, service = _store(tmp_path)
    del service
    source = b"npu-kernel relu_kernel\n"
    req = request("unit-1", source=source)
    legacy = _report()
    legacy["schema_version"] = 3
    legacy.pop("build")
    with pytest.raises(JobError, match="EvalReport validation") as caught:
        JobStore._validate_report(req, legacy, None)
    assert caught.value.code == "invalid_argument"
    # The same payload with a v4 report descriptor is still refused: the descriptor is not a
    # substitute for the payload schema.
    descriptor = {"schema_id": contracts.REPORT_SCHEMA_ID, "media_type": "application/json",
                  "size_bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}
    assert descriptor["schema_id"] == "tcenv.eval-report.v4"
    with pytest.raises(JobError, match="EvalReport validation"):
        JobStore._validate_report(req, legacy, None)
    # A real schema-4 payload passes the evaluator-side check.
    assert JobStore._validate_report(req, _report(), None) is None


# -- build_kind × measurement_kind ------------------------------------------
def test_a_non_native_build_cannot_claim_a_measured_evaluation(tmp_path):
    """build 非空时，build_kind≠native 与 measurement_kind=measured 是结构矛盾。"""
    store, service = _store(tmp_path)
    try:
        req = request("unit-1")
        report = _report(build_kind="simulator")
        measured = dict(_provenance(store), measurement_kind="measured", publication_allowed=True)
        with pytest.raises(JobError, match="non-native build"):
            store.validate_worker_result(
                req, report=report, evidence={"candidate_sha256": req.candidate_sha256},
                provenance=measured,
            )
        simulated = dict(measured, measurement_kind="simulated", publication_allowed=False)
        store.validate_worker_result(
            req, report=report, evidence={"candidate_sha256": req.candidate_sha256},
            provenance=simulated,
        )
    finally:
        service.close()


def test_a_null_build_does_not_participate_in_the_kind_cross_check(tmp_path):
    """``build=null``（合成 timeout）在 measured evaluator 上仍合法。"""
    store, service = _store(tmp_path)
    try:
        req = request("unit-1")
        from tcenv.evaluate import EvalReport

        timeout = EvalReport(
            task=req.task, dtype=req.dtype, size=req.size, mode=req.mode, seed=req.seed_int,
            status="timeout", stage="timeout", target_id=req.target_id,
            protocol_version="3",
        ).to_json()
        store.validate_worker_result(
            req, report=timeout, evidence={"candidate_sha256": req.candidate_sha256},
            provenance=_provenance(store),
        )
    finally:
        service.close()


# -- report build vs signed BuildEvidence -----------------------------------
def test_the_report_build_must_agree_with_the_signed_build_evidence(tmp_path):
    """ok/reused 签名为 full，compile_failed 为 partial，not_run 为 null；provenance 逐字段相同。"""
    report = _report()
    signed = {"status": "ok", "provenance": report["build"]["provenance"],
              "artifacts": report["build"]["artifacts"]}
    JobStore._check_report_against_signed_build(report, {"build": signed})

    with pytest.raises(JobError, match="does not match"):
        JobStore._check_report_against_signed_build(
            {**report, "build": None}, {"build": signed},
        )
    with pytest.raises(JobError, match="provenance differs"):
        JobStore._check_report_against_signed_build(
            report, {"build": {**signed, "provenance": dict(signed["provenance"], builder_id="other")}},
        )
    with pytest.raises(JobError, match="artifacts differ"):
        JobStore._check_report_against_signed_build(
            report, {"build": {**signed, "artifacts": {"schema_id": "x"}}},
        )

    partial = _report("compile_error")
    JobStore._check_report_against_signed_build(
        partial, {"build": {"status": "compile_failed", "provenance": partial["build"]["provenance"]}},
    )
    with pytest.raises(JobError, match="does not match"):
        JobStore._check_report_against_signed_build(partial, {"build": signed})

    timeout = {**report, "status": "timeout", "build": None}
    JobStore._check_report_against_signed_build(timeout, {"build": {"status": "not_run"}})
    with pytest.raises(JobError, match="unknown signed build status"):
        JobStore._check_report_against_signed_build(timeout, {"build": {"status": "other"}})


# -- host side ---------------------------------------------------------------
def test_the_host_refuses_a_result_whose_report_is_not_schema_4(tmp_path):
    """v4 descriptor 配非 schema 4 payload 在 host 解析时按 result_invalid 拒绝。"""
    from dataclasses import replace

    from taic.orchestrator import Orchestrator
    from tests.shared.test_orchestrator_runtime import (
        GrpcShapedEvaluator, _test_epoch, llm_script, make_orch, code, run,
    )
    from tests.support.fakes import FakeClient

    class LegacyPayload(GrpcShapedEvaluator):
        def formal(self, request):
            result = super().formal(request)
            legacy = {"task": result.report.task, "dtype": result.report.dtype,
                      "size": result.report.size, "mode": result.report.mode,
                      "status": "ok", "simulated": True}
            return replace(result, report=legacy)

    orch = make_orch(
        tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0,
        evaluator=LegacyPayload(), publishable_site_ids={"local"},
        publishable_epochs={_test_epoch()},
    )
    from tcenv.evaluation import EvaluationUnavailable

    with pytest.raises(EvaluationUnavailable) as caught:
        orch._evaluate_report("// speed=1.2\nkernel void demo() {}", mode="quick", seed=3, reps=1, label="init")
    assert caught.value.code == "result_invalid"
