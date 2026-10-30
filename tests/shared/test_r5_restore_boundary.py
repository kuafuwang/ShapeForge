"""R5: the resume boundary around a remote evaluation context (D43, F-30/F-36).

CPU only, no device, no model: the "remote" evaluator is a scripted stand-in and the outbox rows are
written by the real ``EvaluationOutbox``.  These tests pin the four recoverable outcomes:

* a completed row mounts a *validated* context on the candidate (projection -> row/result);
* a damaged context is ``NeedsInspection`` at ``restore``, and a refused resume at the CLI/experiments
  boundary (``ResumeRefused``/``RunStartError``), never a traceback;
* the second (non-dry-run) restore reuses the context the first one already validated;
* a refused second restore puts the earlier results back.
"""

from __future__ import annotations

import json

import pytest

from experiments import common as exp_common
from taic import cli, resume as rs
from taic.artifacts import EVALUATION_CONTEXT_FILE_NAME, NeedsInspection, evaluation_context_document
from taic.evaluation_outbox import CompletedEvaluationContext, EvaluationOutbox
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator, RunStartError
from tcenv.evaluate import EvalReport
from tcenv.evaluation import EvaluationRequest, EvaluationResult, local_provenance, receipt_id_for

from ..support.fakes import FakeClient
from ..support.orchestrator import KERNEL, code, fake_eval, llm_script, make_orch, run, stamp_report
from ..support.resume import mark_dead, ns

KERNEL_FILE = "kernel.metal"
SOURCE = "kernel void k() {}\n"


class FakeRemoteEvaluator:
    """A remote-shaped executor: one finished result, no device and no network."""

    transport_kind = "grpc"
    # The confirmation batch reads this to decide ``measurement_kind``.
    provenance = {"measurement_kind": "measured"}

    def __init__(self) -> None:
        self.requests: list[EvaluationRequest] = []

    def formal(self, request: EvaluationRequest) -> EvaluationResult:
        self.requests.append(request)
        report = EvalReport(
            task=request.task, dtype=request.dtype, size=request.size, mode=request.mode,
            seed=request.seed or 1, status="ok", message="ok", speedup=1.2,
            target_id=request.target_id or "apple-m4-max",
        )
        report.speedup, report.speedups, report.candidate_us, report.baseline_us = 1.2, {"mlx": 1.2}, 100.0, {"mlx": 120.0}
        stamp_report(report, request.task, request.dtype, request.size, request.source, build_kind="native")
        return EvaluationResult(
            report=report, provenance=local_provenance(request),
            evidence={"candidate_sha256": request.candidate_sha256},
            receipt={
                "receipt_id": receipt_id_for(request), "request_id": request.request_id,
                "request_digest": request.request_digest, "measurement_kind": "measured",
            },
        )


def _config() -> tuple[LLMConfig, SearchConfig]:
    llm_cfg = LLMConfig(
        base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake",
        reasoning_effort="off", use_tools=False, truncation_retry=False,
    )
    search = SearchConfig(budget_usd=5.0, branches=1, max_rounds=1, max_repair_tries=2, eval_reps=3)
    return llm_cfg, search


def _epoch() -> str:
    return local_provenance(EvaluationRequest(
        task="relu", dtype="fp16", size="small", source="kernel void demo() {}", target_id="apple-m4-max",
    )).epoch


def _bare_run_dir(tmp_path) -> "object":
    """A run directory a resume can read: ``run_config.json`` only, no placement record.

    ``execution.json`` is deliberately absent: this test is about the *context* boundary, not about
    the transport placement check (a gRPC placement demands NPU contract ids, which a hand-built
    legacy row does not have).  The legacy outbox is therefore injected below; the code under test
    (``load_completed_context`` + ``Candidate`` mounting) does not depend on the dialect.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps({
        "loop_protocol": "flame_chase_1", "run_seed": 7, "report_schema_epoch": 4,
        "target": {"id": "apple-m4-max"},
    }), encoding="utf-8")
    return run_dir


def _resuming(run_dir, evaluator) -> Orchestrator:
    llm_cfg, search = _config()
    return Orchestrator(
        "relu", "fp16", "small", llm_cfg, search, run_dir=run_dir, client=FakeClient([]),
        evaluate_fn=fake_eval([]), evaluator=evaluator, resuming=True, log=lambda _s: None,
        publishable_site_ids={"local"}, publishable_epochs={_epoch()},
    )


def _write_candidate(run_dir, source: str = SOURCE):
    folder = run_dir / "candidates" / "000_init"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / KERNEL_FILE).write_text(source, encoding="utf-8")
    report = EvalReport(
        task="relu", dtype="fp16", size="small", mode="full", seed=1, status="ok", stage="done",
        speedup=1.2, target_id="apple-m4-max",
    )
    report.speedup, report.speedups, report.candidate_us, report.baseline_us = 1.2, {"mlx": 1.2}, 100.0, {"mlx": 120.0}
    stamp_report(report, "relu", "fp16", "small", source, build_kind="native")
    (folder / "report.json").write_text(json.dumps(report.to_json()), encoding="utf-8")
    return folder


def _completed_request(run_dir, evaluator: FakeRemoteEvaluator, source: str = SOURCE):
    """A completed outbox row, written by the real outbox (no hand-written row bytes)."""
    request = EvaluationRequest(
        task="relu", dtype="fp16", size="small", source=source, mode="full", seed=1, reps=3,
        run_id=run_dir.name, session_id="init", task_fingerprint="a" * 64, target_id="apple-m4-max",
    )
    outbox = EvaluationOutbox(run_dir, transport="grpc")
    outbox.begin(request)
    result = evaluator.formal(request)
    outbox.complete(request, result)
    return request, result


def _projection(request: EvaluationRequest, result: EvaluationResult) -> dict:
    """The fixed-schema projection a search-period write would leave next to the candidate.

    The environment component has no column in the outbox row, so it is the accepted report's own
    top-level field — that is exactly what the projection/report reconciliation compares.
    """
    report = result.report.to_json() if hasattr(result.report, "to_json") else dict(result.report or {})
    return evaluation_context_document(
        request_id=request.request_id, request_digest=request.request_digest,
        candidate=request.candidate_sha256, evaluator_epoch=result.provenance.epoch,
        baseline_environment_digest=report.get("baseline_environment_digest"),
    )


def _with_completed_context(tmp_path, *, tamper_result: bool = False, tamper_report: bool = False):
    """A run directory with one candidate whose completed row has a projection next to it."""
    run_dir = _bare_run_dir(tmp_path)
    _write_candidate(run_dir)
    evaluator = FakeRemoteEvaluator()
    request, result = _completed_request(run_dir, evaluator)
    (run_dir / "candidates" / "000_init" / EVALUATION_CONTEXT_FILE_NAME).write_text(
        json.dumps(_projection(request, result)), encoding="utf-8",
    )
    if tamper_report:
        # The retained report is the search result's, not a candidate report: rewriting its schema
        # version leaves a document no schema-4 reader accepts, whatever the HMAC says.
        path = EvaluationOutbox.result_path(run_dir, request.request_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["report"]["schema_version"] = 3
        path.write_text(json.dumps(payload), encoding="utf-8")
    if tamper_result:
        # A rewritten result must disagree with the row it is filed under: the persisted
        # provenance digest is what binds the envelope to the request.
        path = EvaluationOutbox.result_path(run_dir, request.request_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["provenance"]["request_digest"] = "ff" * 32
        path.write_text(json.dumps(payload), encoding="utf-8")
    return run_dir, request, result


def _snapshot(run_dir):
    cands = rs._read_candidates(run_dir, [])
    return rs.Snapshot(
        run_dir=str(run_dir), task="relu", dtype="fp16", size="small", model="fake",
        requests=[], candidates=cands,
        trace=[{"idx": 0, "label": "init", "status": "ok", "improved": True}],
        history=[], best_label="init", rounds_done=1, next_round=2, stagnant=0, partial=None,
    )


def test_a_completed_row_mounts_a_validated_context_on_the_candidate(tmp_path):
    """restore turns the projection into a validated row/result; the source is rehashed."""
    run_dir, request, _result = _with_completed_context(tmp_path)
    resumed = _resuming(run_dir, FakeRemoteEvaluator())
    resumed.restore(_snapshot(run_dir), dry_run=True)
    context = resumed.candidates[0].evaluation_context
    assert isinstance(context, CompletedEvaluationContext)
    assert context.request_id == request.request_id
    assert context.row["status"] == "completed"
    assert context.result.provenance.request_digest == request.request_digest
    assert context.source_hash == "matched"


def test_a_damaged_context_is_needs_inspection_at_restore(tmp_path):
    """A tampered result envelope is outbox damage: the resume stops instead of using it."""
    run_dir, _request, _result = _with_completed_context(tmp_path, tamper_result=True)
    resumed = _resuming(run_dir, FakeRemoteEvaluator())
    with pytest.raises(NeedsInspection, match="result_invalid"):
        resumed.restore(_snapshot(run_dir), dry_run=True)
    assert resumed.candidates == []


def test_a_retained_report_that_is_not_schema_4_is_refused_before_any_snapshot_move(tmp_path):
    """评审复现：restore 在装载阶段就拒绝 schema-3 保留报告，且没移动快照、没改写运行目录。

    ``Orchestrator.restore(dry_run=True)`` used to print ``RESTORE_ACCEPTED 3 EvalReport`` for a run
    whose retained result report had been rewritten to schema 3.  The context read now runs the
    schema-4 closure, so the resume stops at ``NeedsInspection`` with the run directory byte-for-byte
    (and mtime-for-mtime) unchanged.
    """
    run_dir, _request, _result = _with_completed_context(tmp_path, tamper_report=True)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in run_dir.rglob("*") if path.is_file()
    }
    resumed = _resuming(run_dir, FakeRemoteEvaluator())
    with pytest.raises(NeedsInspection, match="result_invalid"):
        resumed.restore(_snapshot(run_dir), dry_run=True)
    assert resumed.candidates == []
    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in run_dir.rglob("*") if path.is_file()
    }
    assert after == before


def test_the_second_restore_reuses_the_context_the_first_one_validated(tmp_path, monkeypatch):
    """A dry-run restore validates; the real one must not re-read (or rewrite) the outbox."""
    run_dir, request, _result = _with_completed_context(tmp_path)
    calls: list[str] = []
    real = EvaluationOutbox.load_completed_context

    def spy(self, request_id, request_digest, *, source_bytes=None):
        calls.append(request_id)
        return real(self, request_id, request_digest, source_bytes=source_bytes)

    monkeypatch.setattr(EvaluationOutbox, "load_completed_context", spy)
    resumed = _resuming(run_dir, FakeRemoteEvaluator())
    snap = _snapshot(run_dir)
    resumed.restore(snap, dry_run=True)
    resumed.restore(snap, dry_run=True)
    assert calls == [request.request_id]


def test_prepare_resume_converts_a_needs_inspection_restore_into_a_refusal(tmp_path, monkeypatch):
    """``cli.prepare_resume`` reports ResumeRefused (exit 5); it does not leak the exception."""
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=1)
    run(orch)
    run_dir = tmp_path / "run"
    # A killed run: no result.json, no "done" phase, the recorded pid is dead.
    (run_dir / "result.json").unlink(missing_ok=True)
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    state["phase"] = "search"
    state.pop("stop_reason", None)
    (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    mark_dead(run_dir)
    before = {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}

    def refuse(self, snap, *, dry_run=False):
        raise NeedsInspection("the evaluation context is damaged")

    monkeypatch.setattr(Orchestrator, "restore", refuse)
    with pytest.raises(cli.ResumeRefused) as caught:
        cli.prepare_resume(
            ns(run_dir, add_budget=5.0), client=FakeClient([]), evaluate_fn=fake_eval([]),
            log=lambda _s: None,
        )
    assert caught.value.code == 5
    assert "damaged" in str(caught.value)
    after = {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
    assert after == before


def test_a_failing_second_restore_puts_the_earlier_results_back(tmp_path, monkeypatch):
    """A refused non-dry-run restore must not leave the moved results behind."""
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=1)
    run(orch)
    run_dir = tmp_path / "run"
    snap = rs.load_snapshot(run_dir, orch.search, KERNEL)
    llm_cfg, search = _config()
    resumed = Orchestrator(
        "relu", "fp16", "small", llm_cfg, search, run_dir=run_dir, client=FakeClient([]),
        evaluate_fn=fake_eval([]), resuming=True, log=lambda _s: None,
    )
    info = {"kind": "resuming", "budget_usd": 5.0}

    def refuse(_snap, *, dry_run=False):
        if not dry_run:
            raise NeedsInspection("the evaluation context is damaged")

    monkeypatch.setattr(resumed, "restore", refuse)
    with pytest.raises(cli.ResumeRefused):
        cli.run_prepared(ns(run_dir), resumed, snap, info, 10.0)
    assert (run_dir / "result.json").is_file()
    assert not list(run_dir.glob("result.before_topup_*.json"))


def test_the_experiments_resume_entry_uses_the_same_boundary(tmp_path):
    """``experiments.common._resume_one`` reports a run start error, not a traceback."""
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=1)
    run(orch)
    run_dir = tmp_path / "run"
    mark_dead(run_dir)
    folder = sorted((run_dir / "candidates").iterdir())[0]
    (folder / EVALUATION_CONTEXT_FILE_NAME).write_text(
        json.dumps({"schema_version": 1, "request_id": "rewritten"}), encoding="utf-8",
    )
    llm_cfg, search = _config()
    with pytest.raises(RunStartError, match="not resumable"):
        exp_common._resume_one(
            "relu", "fp16", "small", run_dir, llm_cfg, search, {"llm": {}}, None,
            FakeClient([]), fake_eval([]), None, lambda _s: None,
        )


def test_a_remote_batch_without_a_validated_context_is_unknown_not_published(tmp_path):
    """A remote batch compares report, expected and context; missing context cannot match."""
    from taic.orchestrator import Candidate, EvaluatedReport

    run_dir = _bare_run_dir(tmp_path)
    _write_candidate(run_dir)
    resumed = _resuming(run_dir, FakeRemoteEvaluator())
    report = EvalReport(
        task="relu", dtype="fp16", size="small", mode="full", seed=1, status="ok", stage="done",
        speedup=1.2, target_id="apple-m4-max",
    )
    report.speedup, report.speedups, report.candidate_us, report.baseline_us = 1.2, {"mlx": 1.2}, 100.0, {"mlx": 120.0}
    stamp_report(report, "relu", "fp16", "small", SOURCE, build_kind="native")
    cand = Candidate(idx=0, label="init", source=SOURCE, report=report)
    expected = resumed._confirmation_expected(cand)
    # No validated context: the persisted capability components (CandidateDigest, epoch, device
    # group, build request) are unknown and the evaluator's pack is unknown too, so no fingerprint
    # can be recomputed.  The environment component is the accepted report's frozen value — the one
    # component the plan allows to come from the search report.
    assert expected["build"]["candidate_digest"] is None
    assert expected["evaluation"]["evaluator_epoch"] is None
    assert expected["evaluation"]["baseline_environment_digest"] == report.to_json()["baseline_environment_digest"]
    assert expected["evaluation"]["evaluation_fingerprint"] is None
    batch = resumed._confirmation_batch(cand, EvaluatedReport(report=report), expected)
    assert batch["identity"]["kind"] == "build_identity_unknown"
    # Eligibility is decided first: without a validated context there is no measurement provenance
    # for this batch, so ``measurement_kind`` is ``unknown`` and the batch is ineligible before the
    # identity status is read.  Either way it cannot publish.
    assert batch["measurement_kind"] == "unknown"
    assert resumed._confirmation_batch_status(batch) == "confirmation_ineligible"
