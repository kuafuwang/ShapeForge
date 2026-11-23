"""R5: the search-period ``evaluation_context.json`` projection and this batch's context (D43, F-36).

CPU only, no device, no model.  The "remote" evaluator is a scripted stand-in that returns a
schema-4 structural native report; the outbox rows are written by the real ``EvaluationOutbox``.
The NPU cases use the vendor-sim harness (a fake evaluator, never a real device).

What this pins:

* a remote search writes the fixed 8-key projection next to the candidate, and every field agrees
  with the completed outbox row / accepted report (the local path writes nothing);
* the projection/report reconciliation refuses *any* rewritten field — damage is never an unknown
  context;
* the confirmation gate compares **this batch's** validated context, not the search-period one, so
  a batch measured under a different evaluator epoch cannot publish.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.common import conclusion_of_run
from taic.artifacts import (
    EVALUATION_CONTEXT_FIELDS,
    EVALUATION_CONTEXT_FILE_NAME,
    reconcile_evaluation_context,
    validate_evaluation_context,
)
from tests.support.v2_specimen import V2_TASK
from taic.evaluation_outbox import EvaluationOutbox
from taic.orchestrator import Candidate, EvaluatedReport
from tcenv.evaluate import EvalReport
from tcenv.evaluation import EvaluationRequest, EvaluationResult, local_provenance, receipt_id_for
from tcenv.remote import contracts
from tcenv.schema import SchemaError

from ..support.job_engine import NPU_TARGET, NPU_TASK_FINGERPRINT, npu_request
from ..support.orchestrator import KERNEL, code, fake_eval, llm_script, make_orch, run, stamp_report
from ..support.vendor_sim.harness import (
    NPU_HOST_SIGNATURE_KEY,
    ModuleBuildGrpcHarness,
    NpuGrpcHarness,
)


class RemoteEvaluator:
    """A remote-shaped executor: one finished native result, no device and no network."""

    transport_kind = "grpc"

    def __init__(self) -> None:
        self.requests: list[EvaluationRequest] = []
        self._provenance = None

    @property
    def provenance(self):
        """The last observed provenance; ``_observe_evaluator`` reads epoch/site/backend from it."""
        return self._provenance

    def execution_metadata(self) -> dict:
        """The gRPC contract a resumed run must record; the legacy dialect is not an NPU contract."""
        return {
            "transport": "grpc",
            "api_contract_id": contracts.LEGACY_DIALECT.api_contract_id,
            "evidence_contract_id": contracts.LEGACY_DIALECT.evidence_contract_id,
        }

    def formal(self, request: EvaluationRequest) -> EvaluationResult:
        self.requests.append(request)
        report = EvalReport(
            task=request.task, dtype=request.dtype, size=request.size, mode=request.mode,
            seed=request.seed or 1, stage="done", status="ok", message="ok", speedup=1.2,
            target_id=request.target_id or "apple-m4-max",
        )
        report.speedup, report.speedups, report.candidate_us, report.baseline_us = 1.2, {"mlx": 1.2}, 100.0, {"mlx": 120.0}
        report.benchmark = {
            "candidate": {"median_us": 100.0, "cv": 0.01, "reps": request.reps},
            "baselines": {"mlx": {"median_us": 120.0, "cv": 0.01, "reps": request.reps}},
            "best_baseline": "mlx",
        }
        stamp_report(report, request.task, request.dtype, request.size, request.source, build_kind="native")
        provenance = local_provenance(request)
        self._provenance = provenance
        return EvaluationResult(
            report=report, provenance=provenance,
            evidence={"candidate_sha256": request.candidate_sha256},
            receipt={
                "receipt_id": receipt_id_for(request), "request_id": request.request_id,
                "request_digest": request.request_digest, "measurement_kind": "measured",
            },
        )


def _epoch() -> str:
    return local_provenance(EvaluationRequest(
        task=V2_TASK, dtype="fp16", size="small", source="kernel void demo() {}", target_id="apple-m4-max",
    )).epoch


def _remote_run(tmp_path, **search_kw):
    evaluator = RemoteEvaluator()
    orch = make_orch(
        tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0, evaluator=evaluator,
        publishable_site_ids={"local"}, publishable_epochs={_epoch()}, **search_kw,
    )
    return orch, run(orch), evaluator


def _candidate_dirs(run_dir: Path) -> list[Path]:
    return sorted(path for path in (run_dir / "candidates").iterdir() if path.is_dir())


def _rows(run_dir: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (run_dir / "evaluation_outbox").glob("*.json")
        if not path.name.endswith(".result.json") and path.name != "receipts.json"
    ]


def _completed_row(run_dir: Path, request_id: str) -> dict:
    """The completed row a projection points at (the confirmation batch has its own row too)."""
    rows = [
        row for row in _rows(run_dir)
        if row.get("status") == "completed" and row.get("request_id") == request_id
    ]
    assert len(rows) == 1, rows
    return rows[0]




def test_a_local_search_writes_no_projection(tmp_path):
    """A local candidate has no remote context: the file is absent, not an empty document."""
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0, evaluate_fn=fake_eval([]))
    run(orch)
    folders = _candidate_dirs(orch.run_dir)
    assert folders
    assert not any((folder / EVALUATION_CONTEXT_FILE_NAME).exists() for folder in folders)
















def report_source_bytes(orch, folder: Path) -> bytes:
    return (folder / orch.kernel_name).read_bytes()


# ------------------------------------------------------------------------- write -> read -> restore


def _resuming(run_dir: Path):
    """A resuming orchestrator with the same remote evaluator identity as the original run."""
    from taic import resume as rs  # noqa: F401  (the snapshot type lives here)

    from taic.llm import LLMConfig, SearchConfig
    from taic.orchestrator import Orchestrator

    from ..support.fakes import FakeClient
    from ..support.orchestrator import fake_eval

    llm_cfg = LLMConfig(
        base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake",
        reasoning_effort="off", use_tools=False, truncation_retry=False,
    )
    search = SearchConfig(budget_usd=5.0, branches=1, max_rounds=1, max_repair_tries=2, eval_reps=3)
    return Orchestrator(
        V2_TASK, "fp16", "small", llm_cfg, search, run_dir=run_dir, client=FakeClient([]),
        evaluate_fn=fake_eval([]), evaluator=RemoteEvaluator(), resuming=True, log=lambda _s: None,
        publishable_site_ids={"local"}, publishable_epochs={_epoch()},
    )


def _snapshot(run_dir: Path):
    from taic import resume as rs

    return rs.Snapshot(
        run_dir=str(run_dir), task=V2_TASK, dtype="fp16", size="small", model="fake",
        requests=[], candidates=rs._read_candidates(run_dir, []),
        trace=[{"idx": 0, "label": "init", "status": "ok", "improved": True}],
        history=[], best_label="init", rounds_done=1, next_round=2, stagnant=0, partial=None,
    )






# --------------------------------------------------------------------------------- NPU dialects


def _npu_request(request_id: str, source: bytes, *, target_id: str = NPU_TARGET) -> EvaluationRequest:
    return EvaluationRequest(
        kind="formal_eval", request_id=request_id, run_id="npu-run-1",
        session_id="npu-session-1", task=V2_TASK, dtype="fp16", size="small",
        target_id=target_id, task_fingerprint=NPU_TASK_FINGERPRINT,
        source=source.decode("utf-8"), mode="quick", seed=3, reps=2,
        queue_timeout_s=10.0, candidate_timeout_s=10.0,
    )


def _npu_outbox(root: Path, *, build: bool = False) -> EvaluationOutbox:
    api = contracts.NPU_BUILD_API_CONTRACT_ID if build else contracts.NPU_SOURCE_API_CONTRACT_ID
    evidence = (
        contracts.NPU_BUILD_EVIDENCE_CONTRACT_ID if build else contracts.NPU_SOURCE_EVIDENCE_CONTRACT_ID
    )
    return EvaluationOutbox(
        root, signature_key=NPU_HOST_SIGNATURE_KEY, transport="grpc",
        api_contract_id=api, evidence_contract_id=evidence,
    )


def _projection_for_npu(tmp_path: Path, *, request_id: str, source: bytes, build: bool = False):
    """Write one completed NPU row by the real outbox and project it back through a fresh one."""
    from dataclasses import replace

    from taic.evaluation_outbox import bind_wire_digest

    root = tmp_path / "run"
    if build:
        with ModuleBuildGrpcHarness(tmp_path / "prebuilt") as harness:
            outcome = harness.host_build(source)
            harness.upload_module(outcome)
            harness.set_prebuilt(outcome, outcome.bundle, source)
            host = _npu_request(request_id, b"", target_id=harness.engine.identity.target_id)
            host = replace(host, source="")
            request = bind_wire_digest(host, harness.factory._for_request(host), npu_contract=True)
            outbox = _npu_outbox(root, build=True)
            outbox.begin(request, epoch=harness.factory.epoch)
            outbox.complete(request, harness.factory.formal(request))
            context = _npu_outbox(root, build=True).load_completed_context(request.request_id, request.request_digest)
    else:
        with NpuGrpcHarness(tmp_path) as harness:
            external = npu_request(request_id, source=source)
            host = _npu_request(request_id, source, target_id=harness.engine.identity.target_id)
            request = bind_wire_digest(
                replace(host, contract_request_digest=external.request_digest),
                harness.factory._for_request(host), npu_contract=True,
            )
            outbox = _npu_outbox(root)
            outbox.begin(request, epoch=harness.factory.epoch)
            outbox.complete(request, harness.factory.formal(request))
            context = _npu_outbox(root).load_completed_context(
                request.request_id, request.request_digest, source_bytes=source,
            )
    assert context is not None
    return root, context


def test_an_npu_projection_keeps_the_candidate_digest_and_device_group(tmp_path):
    """The NPU projection is the row's CandidateDigest and device group, not a host-side guess."""
    root, context = _projection_for_npu(
        tmp_path, request_id="proj-npu-1", source=b"npu-kernel relu_kernel\n",
    )
    document = _npu_outbox(root).projection_document(context)
    assert list(document) == list(EVALUATION_CONTEXT_FIELDS)
    assert document["candidate"] == context.row["candidate_digest"]
    assert set(document["candidate"]) == {"algorithm", "value"}
    assert document["device_group"] == context.row["device_group"]
    # Only the build dialect carries a request-level build document.
    assert document["build_provenance"] is None
    report = context.result.report
    report = report.to_json() if hasattr(report, "to_json") else report
    assert reconcile_evaluation_context(document, row=context.row, report=report) == document


def test_a_build_dialect_projection_keeps_the_build_identity(tmp_path):
    """The build request identity is its own projection field, reconciled against the row."""
    root, context = _projection_for_npu(
        tmp_path, request_id="proj-npu-build-1",
        source=b"npu-kernel relu_kernel\n  tx_thread_id\n", build=True,
    )
    document = _npu_outbox(root, build=True).projection_document(context)
    assert context.row["build_provenance"] is not None
    assert document["build_provenance"] == context.row["build_provenance"]
    report = context.result.report
    report = report.to_json() if hasattr(report, "to_json") else report
    assert reconcile_evaluation_context(document, row=context.row, report=report) == document
    # A rewritten build identity is damage when the projection is reconciled with its row.
    tampered = {**document, "build_provenance": {"mode": "prebuilt", "provenance": {"build_host": "control_host"}}}
    with pytest.raises(SchemaError, match="build_provenance"):
        reconcile_evaluation_context(tampered, row=context.row, report=report)
