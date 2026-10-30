"""R5: the frozen remote evaluation-context projection and its read-only resume path.

CPU only: no device, no model, no outbox HMAC.  These tests pin the *format* (fixed schema v1),
the one-authority path helper, and the three-way read outcome the plan requires (§7.2, D43):
a missing or pending row is an **unknown** context, a damaged projection or row is
``NeedsInspection``, and a local candidate simply has no projection.
"""

from __future__ import annotations

import json

import pytest

from taic import resume as rs
from taic.artifacts import (
    EVALUATION_CONTEXT_FILE_NAME,
    EVALUATION_CONTEXT_SCHEMA_VERSION,
    NeedsInspection,
    evaluation_context_document,
    read_evaluation_context,
    validate_evaluation_context,
)
from taic.evaluation_outbox import EvaluationOutbox, row_state
from tcenv.evaluate import EvalReport
from tcenv.schema import PROTOCOL_VERSION, SchemaError

from ..support.orchestrator import code, llm_script, make_orch, stamp_report

KERNEL_FILE = "kernel.metal"  # ``taic.resume._kernel_name`` fallback
REQUEST_ID = "req-0123456789abcdef"
REQUEST_DIGEST = "ab" * 32
SOURCE_SHA = "cd" * 32
DEVICE_GROUP = {
    "device_group_id": "npu-0",
    "vendor_id": "ascend",
    "profile_digest": "11" * 32,
    "topology_digest": "22" * 32,
    "resource_generation": "3",
}


def _generic_projection(**over) -> dict:
    document = evaluation_context_document(
        request_id=REQUEST_ID,
        request_digest=REQUEST_DIGEST,
        candidate=SOURCE_SHA,
        evaluator_epoch="epoch-1",
        baseline_environment_digest="ee" * 32,
    )
    document.update(over)
    return document


def _npu_projection() -> dict:
    return evaluation_context_document(
        request_id=REQUEST_ID,
        request_digest=REQUEST_DIGEST,
        candidate={"algorithm": "sha256", "value": SOURCE_SHA},
        evaluator_epoch="epoch-1",
        device_group=DEVICE_GROUP,
        build_provenance={"schema_id": "npu.build.request/3", "mode": "evaluator", "provenance": {}},
        baseline_environment_digest=None,
    )


def _ok_report(speed: float) -> EvalReport:
    report = EvalReport(
        task="relu", dtype="fp16", size="small", mode="full", seed=1, status="ok", stage="done",
        speedup=speed, target_id="apple-m4-max", protocol_version=PROTOCOL_VERSION,
        backend="metal", backend_version="1",
    )
    report.speedup, report.speedups, report.candidate_us, report.baseline_us = speed, {"mlx": speed}, 100.0, {"mlx": 120.0}
    stamp_report(report, "relu", "fp16", "small", "kernel void k() {}\n", build_kind="native")
    return report


def _candidate_dir(run, label: str = "init") -> "object":
    folder = run / "candidates" / f"000_{label}"
    folder.mkdir(parents=True)
    (folder / KERNEL_FILE).write_text("kernel void k() {}\n")
    (folder / "report.json").write_text(json.dumps(_ok_report(1.2).to_json()))
    return folder


def _write_row(run, request_id: str, *, status: str, **extra) -> None:
    path = EvaluationOutbox.row_path(run, request_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"request_id": request_id, "status": status, **extra}), encoding="utf-8")


# ------------------------------------------------------------------------------------- the frozen schema


def test_the_projection_is_a_fixed_eight_key_schema_v1_document():
    """The projection's key set and version are frozen: a reader never guesses a missing field."""
    document = _generic_projection()
    assert list(document) == [
        "schema_version", "request_id", "request_digest", "candidate",
        "evaluator_epoch", "device_group", "build_provenance", "baseline_environment_digest",
    ]
    assert document["schema_version"] == EVALUATION_CONTEXT_SCHEMA_VERSION == 1
    assert validate_evaluation_context(document) == document


def test_an_npu_projection_keeps_the_candidate_digest_device_group_and_build_identity():
    """The NPU dialect's CandidateDigest, device group and build document are carried verbatim."""
    document = _npu_projection()
    assert document["candidate"] == {"algorithm": "sha256", "value": SOURCE_SHA}
    assert document["device_group"] == DEVICE_GROUP
    assert document["build_provenance"]["mode"] == "evaluator"
    assert document["baseline_environment_digest"] is None
    assert validate_evaluation_context(document) == document


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(lambda doc: doc.pop("evaluator_epoch"), id="missing-key"),
        pytest.param(lambda doc: doc.update(extra=1), id="unknown-key"),
        pytest.param(lambda doc: doc.update(schema_version=2), id="newer-version"),
        pytest.param(lambda doc: doc.update(schema_version="1"), id="version-not-an-int"),
        pytest.param(lambda doc: doc.update(schema_version=True), id="version-is-a-bool"),
        pytest.param(lambda doc: doc.update(request_id=""), id="empty-request-id"),
        pytest.param(lambda doc: doc.update(request_digest="not-hex"), id="digest-not-hex"),
        pytest.param(lambda doc: doc.update(candidate={"value": SOURCE_SHA}), id="candidate-missing-algorithm"),
        pytest.param(lambda doc: doc.update(candidate={"algorithm": "sha256", "value": SOURCE_SHA, "x": 1}), id="candidate-extra-key"),
        pytest.param(lambda doc: doc.update(device_group=[]), id="device-group-not-an-object"),
        pytest.param(lambda doc: doc.update(build_provenance="x"), id="build-provenance-not-an-object"),
        pytest.param(lambda doc: doc.update(baseline_environment_digest="nope"), id="environment-not-hex"),
    ],
)
def test_a_projection_that_is_not_the_fixed_schema_is_damaged_not_unknown(mutation):
    """A rewritten projection is damaged: unknown would let a tampered pointer look like "never evaluated"."""
    document = _generic_projection()
    mutation(document)
    with pytest.raises(SchemaError):
        validate_evaluation_context(document, what="evaluation_context.json")


def test_a_projection_with_a_duplicate_key_is_damaged(tmp_path):
    """A duplicated key must not be silently collapsed by ``json.loads`` before the schema check."""
    path = tmp_path / EVALUATION_CONTEXT_FILE_NAME
    text = json.dumps(_generic_projection())
    path.write_text(text[:-1] + ', "evaluator_epoch": "epoch-2"}', encoding="utf-8")
    with pytest.raises(NeedsInspection):
        read_evaluation_context(path)


def test_a_well_formed_projection_round_trips_and_an_absent_one_is_none(tmp_path):
    path = tmp_path / EVALUATION_CONTEXT_FILE_NAME
    assert read_evaluation_context(path) is None
    document = _npu_projection()
    path.write_text(json.dumps(document), encoding="utf-8")
    assert read_evaluation_context(path) == document


# --------------------------------------------------------------------------- the one-authority path helper


def test_the_public_path_helper_is_the_path_the_outbox_writes(tmp_path):
    """``row_state`` must probe the very file ``EvaluationOutbox`` writes, without constructing one."""
    outbox = EvaluationOutbox(tmp_path / "run")
    assert EvaluationOutbox.row_path(tmp_path / "run", REQUEST_ID) == outbox._state_path(REQUEST_ID)
    assert EvaluationOutbox.result_path(tmp_path / "run", REQUEST_ID) == outbox._result_path(REQUEST_ID)
    assert row_state(tmp_path / "run", REQUEST_ID) == "missing"


@pytest.mark.parametrize("status,expected", [("completed", "completed"), ("pending", "pending"), ("unavailable", "pending")])
def test_the_row_state_probe_maps_failures_to_unknown_not_to_damage(tmp_path, status, expected):
    """Only a completed row is a usable context; pending/unavailable are an unknown context."""
    _write_row(tmp_path, REQUEST_ID, status=status)
    assert row_state(tmp_path, REQUEST_ID) == expected


@pytest.mark.parametrize(
    "text,request_id",
    [
        pytest.param("{not json", REQUEST_ID, id="invalid-json"),
        pytest.param("[1, 2]", REQUEST_ID, id="not-an-object"),
        pytest.param(json.dumps({"request_id": "someone-else", "status": "completed"}), REQUEST_ID, id="wrong-request-id"),
        pytest.param(json.dumps({"request_id": REQUEST_ID, "status": "finished"}), REQUEST_ID, id="unknown-status"),
        pytest.param(json.dumps({"request_id": REQUEST_ID}), REQUEST_ID, id="no-status"),
    ],
)
def test_a_damaged_row_is_never_reported_as_a_missing_row(tmp_path, text, request_id):
    """A row that cannot be read is a tampered identity, not evidence that nothing was evaluated."""
    path = EvaluationOutbox.row_path(tmp_path, request_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(NeedsInspection):
        row_state(tmp_path, request_id)


# ------------------------------------------------------------------------------- _read_candidates wiring


def test_a_completed_row_makes_the_projection_a_completed_context(tmp_path):
    """A remote candidate with a completed row is readable and mounts the projection unchanged."""
    folder = _candidate_dir(tmp_path)
    document = _generic_projection()
    (folder / EVALUATION_CONTEXT_FILE_NAME).write_text(json.dumps(document), encoding="utf-8")
    _write_row(tmp_path, REQUEST_ID, status="completed")

    notes: list[str] = []
    cands = rs._read_candidates(tmp_path, notes)
    assert [c["label"] for c in cands] == ["init"]
    assert cands[0]["evaluation_context"] == document
    assert cands[0]["evaluation_context_state"] == "completed"
    assert notes == []


@pytest.mark.parametrize("row", ["missing", "pending"])
def test_a_missing_or_pending_row_leaves_the_context_unknown(tmp_path, row):
    """A pointer without a completed row is unknown: the candidate is still readable, never confirmable."""
    folder = _candidate_dir(tmp_path)
    (folder / EVALUATION_CONTEXT_FILE_NAME).write_text(json.dumps(_generic_projection()), encoding="utf-8")
    if row == "pending":
        _write_row(tmp_path, REQUEST_ID, status="pending")

    notes: list[str] = []
    cands = rs._read_candidates(tmp_path, notes)
    assert cands[0]["evaluation_context"] is None
    assert cands[0]["evaluation_context_state"] == "unknown"
    assert any(row in note for note in notes)


def test_a_local_candidate_has_no_remote_context(tmp_path):
    """A local run writes no projection; that is the normal case, not an unknown one."""
    _candidate_dir(tmp_path)
    notes: list[str] = []
    cands = rs._read_candidates(tmp_path, notes)
    assert cands[0]["evaluation_context"] is None
    assert cands[0]["evaluation_context_state"] == "local"
    assert notes == []


def test_a_damaged_projection_stops_the_resume_instead_of_becoming_unknown(tmp_path):
    """A projection nobody can read needs a person: it must not degrade into "no remote evaluation"."""
    folder = _candidate_dir(tmp_path)
    (folder / EVALUATION_CONTEXT_FILE_NAME).write_text(json.dumps(_generic_projection(schema_version=7)), encoding="utf-8")
    with pytest.raises(NeedsInspection):
        rs._read_candidates(tmp_path, [])


def test_a_damaged_row_stops_the_resume_instead_of_becoming_unknown(tmp_path):
    """The row is part of the pointer's evidence; damaging it is not the same as never having it."""
    folder = _candidate_dir(tmp_path)
    (folder / EVALUATION_CONTEXT_FILE_NAME).write_text(json.dumps(_generic_projection()), encoding="utf-8")
    path = EvaluationOutbox.row_path(tmp_path, REQUEST_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(NeedsInspection):
        rs._read_candidates(tmp_path, [])


# ----------------------------------------------------------------------------------- Candidate mounting


def test_a_projection_without_an_outbox_is_refused_not_silently_skipped(tmp_path):
    """A remote context in a run resumed as local must be refused: skipping it would lose identity."""
    from taic.artifacts import NeedsInspection

    folder = _candidate_dir(tmp_path)
    document = _generic_projection()
    (folder / EVALUATION_CONTEXT_FILE_NAME).write_text(json.dumps(document), encoding="utf-8")
    _write_row(tmp_path, REQUEST_ID, status="completed")
    cands = rs._read_candidates(tmp_path, [])
    snap = rs.Snapshot(
        run_dir=str(tmp_path), task="relu", dtype="fp16", size="small", model="fake",
        requests=[], candidates=cands, trace=[{"idx": 0, "label": "init", "status": "ok", "improved": True}],
        history=[], best_label="init", rounds_done=1, next_round=2, stagnant=0, partial=None,
    )
    resumed = make_orch(tmp_path / "resumed", llm_script(code("speed=1.2")), [], max_rounds=1)
    assert resumed._outbox is None
    with pytest.raises(NeedsInspection, match="no evaluation outbox"):
        resumed.restore(snap, dry_run=True)
