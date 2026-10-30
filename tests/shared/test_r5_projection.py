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
        task="relu", dtype="fp16", size="small", source="kernel void demo() {}", target_id="apple-m4-max",
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


def test_a_remote_search_writes_the_fixed_schema_projection_next_to_the_candidate(tmp_path):
    """Every projection field agrees with the completed row / accepted report; nothing is invented."""
    orch, _res, _evaluator = _remote_run(tmp_path)
    folders = _candidate_dirs(orch.run_dir)
    assert len(folders) == 1
    path = folders[0] / EVALUATION_CONTEXT_FILE_NAME
    document = json.loads(path.read_text(encoding="utf-8"))
    # Fixed schema, fixed key order: no extra key, no missing key, no second vocabulary.
    assert list(document) == list(EVALUATION_CONTEXT_FIELDS)
    assert validate_evaluation_context(document) == document

    row = _completed_row(orch.run_dir, document["request_id"])
    report = json.loads((folders[0] / "report.json").read_text(encoding="utf-8"))
    assert document["request_id"] == row["request_id"]
    assert document["request_digest"] == row["request_digest"]
    assert document["candidate"] == report["build"]["candidate_sha256"]
    assert document["candidate"] == row["candidate_sha256"]
    assert document["evaluator_epoch"] == row["evaluator_epoch"]
    # The generic (Metal/source) dialect persists neither a device group nor a request-level build
    # identity, so the projection says ``null`` instead of guessing one.
    assert document["device_group"] is None
    assert document["build_provenance"] is None
    # The frozen environment component has no column in the row; it is the accepted report's field.
    assert document["baseline_environment_digest"] == report["baseline_environment_digest"]


def test_a_local_search_writes_no_projection(tmp_path):
    """A local candidate has no remote context: the file is absent, not an empty document."""
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0, evaluate_fn=fake_eval([]))
    run(orch)
    folders = _candidate_dirs(orch.run_dir)
    assert folders
    assert not any((folder / EVALUATION_CONTEXT_FILE_NAME).exists() for folder in folders)


def test_the_search_evaluation_returns_its_own_validated_context(tmp_path):
    """``_evaluate_report`` hands back the row/result it just wrote, with the source rehashed."""
    orch, _res, _evaluator = _remote_run(tmp_path)
    folder = _candidate_dirs(orch.run_dir)[0]
    source = (folder / orch.kernel_name).read_text()
    evaluated = orch._evaluate_report(source, mode="quick", seed=3, reps=1, label="init")
    assert isinstance(evaluated, EvaluatedReport)
    context = evaluated.context
    assert context is not None
    assert context.source_hash == "matched"
    assert context.result.provenance.request_digest == context.request_digest


def test_a_retained_context_before_the_projection_is_refused_after_row_tampering(tmp_path, monkeypatch):
    """§23 路径：写 ``evaluation_context.json`` 之前的读回也会拒绝被改写的行身份。

    ``_retain_evaluation_context`` is the only reader between ``complete`` and the projection, and it
    goes through ``load_completed_context``.  A row whose epoch or candidate contradicts its retained
    result must stop the whole evaluation (``evaluation_context_invalid``), not be projected.
    """
    from tcenv.evaluation import EvaluationUnavailable

    orch, _res, _evaluator = _remote_run(tmp_path)
    folder = _candidate_dirs(orch.run_dir)[0]
    source = (folder / orch.kernel_name).read_text()
    seen = []
    original_begin = orch._outbox.begin

    def recording_begin(request, *, epoch=""):
        seen.append(request)
        return original_begin(request, epoch=epoch)

    monkeypatch.setattr(orch._outbox, "begin", recording_begin)
    evaluated = orch._evaluate_report(source, mode="quick", seed=3, reps=1, label="init")
    request = seen[-1]
    row_path = orch._outbox.row_path(orch.run_dir, request.request_id)
    original = json.loads(row_path.read_text(encoding="utf-8"))
    for field, value in (("evaluator_epoch", "f" * 64), ("candidate_sha256", "f" * 64)):
        tampered = dict(original, **{field: value})
        row_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(EvaluationUnavailable, match="result_invalid"):
            orch._retain_evaluation_context(request)
    row_path.write_text(json.dumps(original), encoding="utf-8")
    assert orch._retain_evaluation_context(request) is not None
    assert evaluated.context is not None


def test_the_projection_reconciles_with_its_row_and_report(tmp_path):
    """The writer's own output must pass the read-side reconciliation (F-36)."""
    orch, _res, _evaluator = _remote_run(tmp_path)
    folder = _candidate_dirs(orch.run_dir)[0]
    projection = json.loads((folder / EVALUATION_CONTEXT_FILE_NAME).read_text(encoding="utf-8"))
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    row = _completed_row(orch.run_dir, projection["request_id"])
    assert reconcile_evaluation_context(projection, row=row, report=report) == projection


@pytest.mark.parametrize(
    "field, value",
    [
        ("request_id", "ev-rewritten"),
        ("request_digest", "ff" * 32),
        ("candidate", "9" * 64),
        ("evaluator_epoch", "e" * 64),
        ("device_group", {"device_group_id": "g", "vendor_id": "v", "profile_digest": "p",
                          "topology_digest": "t", "resource_generation": 1}),
        ("build_provenance", {"mode": "prebuilt", "provenance": {"build_host": "control_host"}}),
        ("baseline_environment_digest", "0" * 64),
    ],
)
def test_any_rewritten_projection_field_is_damage_not_unknown(tmp_path, field, value):
    """A projection that disagrees with its row/report is damage, never a silently unknown context."""
    orch, _res, _evaluator = _remote_run(tmp_path)
    folder = _candidate_dirs(orch.run_dir)[0]
    projection = json.loads((folder / EVALUATION_CONTEXT_FILE_NAME).read_text(encoding="utf-8"))
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    row = _completed_row(orch.run_dir, projection["request_id"])
    tampered = dict(projection)
    tampered[field] = value
    with pytest.raises(SchemaError, match=field):
        reconcile_evaluation_context(tampered, row=row, report=report)
    # The untouched projection still reconciles: the test above is about the one field.
    assert reconcile_evaluation_context(projection, row=row, report=report) == projection


def test_a_remote_confirmation_uses_this_batchs_context_for_every_remote_field(tmp_path):
    """The batch's own validated row answers the remote fields; only the unpersisted one is unknown.

    The generic (Metal/source) dialect persists no request-level build document, so the host has no
    independent BuildProvenance to compare against and the batch cannot publish (plan §7.2: an
    expected that cannot be recomputed is unknown).  What it *can* — and now does — answer from the
    batch's context is the CandidateDigest/source binding, the device group and the evaluator epoch:
    none of those is ``unknown`` any more, so the run does not fail for "no context at all".
    """
    orch, res, _evaluator = _remote_run(tmp_path)
    entry = res.confirmations[0]
    identity = entry["identity"]
    unknown = {item["field"] for item in identity["differences"] if item["kind"] == "unknown"}
    assert "candidate_digest" not in unknown
    assert "evaluator_epoch" not in unknown
    assert not any(field.startswith("device_group.") for field in unknown)
    assert "build_provenance" not in unknown
    assert unknown and all(field.startswith("provenance.") for field in unknown)
    assert entry["status"] == "build_identity_unknown" == identity["kind"]
    assert not (orch.run_dir / "confirm_report.json").exists()
    # The read-disk predicate sees the same batch record: it does not publish either.
    stored = json.loads((orch.run_dir / "result.json").read_text(encoding="utf-8"))
    assert conclusion_of_run(orch.run_dir, stored)["publication"] != "confirmed"


def test_the_remote_expected_never_asks_the_host_for_library_versions(tmp_path, monkeypatch):
    """The frozen search-period environment comes from the projection; the host's libraries never do."""
    orch, _res, _evaluator = _remote_run(tmp_path)
    folder = _candidate_dirs(orch.run_dir)[0]
    projection = json.loads((folder / EVALUATION_CONTEXT_FILE_NAME).read_text(encoding="utf-8"))
    report = EvalReport.from_json(json.loads((folder / "report.json").read_text(encoding="utf-8")))
    cand = Candidate(idx=0, label="init", source=(folder / orch.kernel_name).read_text(), report=report)
    row = _completed_row(orch.run_dir, projection["request_id"])
    cand.evaluation_context = orch._outbox.load_completed_context(
        row["request_id"], row["request_digest"], source_bytes=report_source_bytes(orch, folder),
    )
    cand.evaluation_context_projection = projection

    def explode(*_args, **_kwargs):
        raise AssertionError("a remote expected must not recompute the host's library versions")

    monkeypatch.setattr("tcenv.evaluate.baseline_environment_digest", explode)
    expected = orch._confirmation_expected(cand)
    assert expected["evaluation"]["baseline_environment_digest"] == projection["baseline_environment_digest"]
    assert expected["evaluation"]["evaluation_fingerprint"]


def test_a_confirmation_batch_measured_under_another_epoch_is_a_mismatch(tmp_path):
    """The gate reads the batch's context: keeping the search one would hide an epoch change."""
    from dataclasses import replace

    from taic.evaluation_outbox import CompletedEvaluationContext

    orch, _res, _evaluator = _remote_run(tmp_path)
    folder = _candidate_dirs(orch.run_dir)[0]
    report = EvalReport.from_json(json.loads((folder / "report.json").read_text(encoding="utf-8")))
    row = _completed_row(orch.run_dir, json.loads((folder / EVALUATION_CONTEXT_FILE_NAME).read_text(encoding="utf-8"))["request_id"])
    context = orch._outbox.load_completed_context(
        row["request_id"], row["request_digest"], source_bytes=report_source_bytes(orch, folder),
    )
    assert isinstance(context, CompletedEvaluationContext)
    cand = Candidate(idx=0, label="init", source=report_source_bytes(orch, folder).decode(), report=report)
    cand.evaluation_context = context
    cand.evaluation_context_projection = json.loads(
        (folder / EVALUATION_CONTEXT_FILE_NAME).read_text(encoding="utf-8")
    )
    other = replace(context, row={**context.row, "evaluator_epoch": "f" * 64})
    expected = orch._confirmation_expected(cand)
    batch = orch._confirmation_batch(cand, EvaluatedReport(report=report, context=other), expected)
    assert batch["identity"]["kind"] == "build_identity_mismatch"
    assert any(
        item["field"] == "evaluator_epoch" and item["side"] == "confirmation"
        for item in batch["identity"]["differences"]
    )
    # The same candidate with its own (unchanged) context has no epoch difference at all: the
    # remaining unknown is the build provenance the generic dialect does not persist.
    own = orch._confirmation_batch(cand, EvaluatedReport(report=report, context=context), expected)
    own_fields = {(item["field"], item["kind"]) for item in own["identity"]["differences"]}
    assert ("evaluator_epoch", "mismatch") not in own_fields
    assert ("evaluator_epoch", "unknown") not in own_fields
    assert own_fields and all(field.startswith("provenance.") for field, _kind in own_fields)


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
        "relu", "fp16", "small", llm_cfg, search, run_dir=run_dir, client=FakeClient([]),
        evaluate_fn=fake_eval([]), evaluator=RemoteEvaluator(), resuming=True, log=lambda _s: None,
        publishable_site_ids={"local"}, publishable_epochs={_epoch()},
    )


def _snapshot(run_dir: Path):
    from taic import resume as rs

    return rs.Snapshot(
        run_dir=str(run_dir), task="relu", dtype="fp16", size="small", model="fake",
        requests=[], candidates=rs._read_candidates(run_dir, []),
        trace=[{"idx": 0, "label": "init", "status": "ok", "improved": True}],
        history=[], best_label="init", rounds_done=1, next_round=2, stagnant=0, partial=None,
    )


def test_a_written_projection_is_read_back_and_mounted_by_a_resume(tmp_path):
    """The file the search wrote is the file the resume reads: projection -> row/result context."""
    orch, _res, _evaluator = _remote_run(tmp_path)
    notes: list[str] = []
    from taic import resume as rs

    candidates = rs._read_candidates(orch.run_dir, notes)
    assert notes == []
    assert [c["evaluation_context_state"] for c in candidates] == ["completed"]
    resumed = _resuming(orch.run_dir)
    resumed.restore(_snapshot(orch.run_dir), dry_run=True)
    context = resumed.candidates[0].evaluation_context
    assert context is not None and context.source_hash == "matched"
    projection = resumed.candidates[0].evaluation_context_projection
    assert projection == json.loads((_candidate_dirs(orch.run_dir)[0] / EVALUATION_CONTEXT_FILE_NAME).read_text())


def test_a_tampered_projection_is_refused_at_restore(tmp_path):
    """A projection that disagrees with its completed row is damage, never a silent unknown."""
    from taic.artifacts import NeedsInspection

    orch, _res, _evaluator = _remote_run(tmp_path)
    path = _candidate_dirs(orch.run_dir)[0] / EVALUATION_CONTEXT_FILE_NAME
    projection = json.loads(path.read_text(encoding="utf-8"))
    projection["evaluator_epoch"] = "f" * 64
    path.write_text(json.dumps(projection), encoding="utf-8")
    resumed = _resuming(orch.run_dir)
    with pytest.raises(NeedsInspection, match="disagrees"):
        resumed.restore(_snapshot(orch.run_dir), dry_run=True)
    assert resumed.candidates == []


# --------------------------------------------------------------------------------- NPU dialects


def _npu_request(request_id: str, source: bytes, *, target_id: str = NPU_TARGET) -> EvaluationRequest:
    return EvaluationRequest(
        kind="formal_eval", request_id=request_id, run_id="npu-run-1",
        session_id="npu-session-1", task="relu", dtype="fp16", size="small",
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


def test_a_remote_confirmation_batch_with_a_damaged_row_is_unavailable_not_a_kernel_error(tmp_path):
    """远端确认批次读回自己的行/result 失败时，出口是 `evaluation_unavailable`（不是 free text）。

    §7.2: the confirmation-side context is validated per batch.  A row whose digest no longer matches
    is structural damage, so the batch is ``evaluation_unavailable`` — not ``error: ...`` free text,
    not an identity unknown, not a kernel failure, and nothing is published.
    """
    import asyncio

    orch, res, _evaluator = _remote_run(tmp_path)
    # The confirmation request is deterministic, so re-running it replays its own durable row.
    # Damage every row on disk: whichever one is replayed, the read-back must fail closed.
    damaged: list[Path] = []
    for path in (orch.run_dir / "evaluation_outbox").glob("*.json"):
        if path.name.endswith(".result.json") or path.name == "receipts.json":
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        document["request_digest"] = "f" * 64
        path.write_text(json.dumps(document), encoding="utf-8")
        damaged.append(path)
    assert damaged
    res.confirmations.clear()
    assert asyncio.run(orch._confirm(res)) is None
    entry = res.confirmations[-1]
    assert entry["status"] == "evaluation_unavailable"
    assert not entry["status"].startswith("error:"), "the closed-set outcome is not free text"
    assert res.confirmed_status == "evaluation_unavailable" and res.confirmed_values == []
    assert not (orch.run_dir / "confirm_report.json").exists()
