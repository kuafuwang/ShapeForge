"""G21：``msl-bundle`` 交卷的字节保真，以及非法 UTF-8 的闭合路径。

This is the session boundary, not the request boundary: ``collect_session`` runs *inside* a billed
session, so it must never raise (that would lose the cost and be recorded as ``agent_failed``).  A
submission whose bytes are not text becomes ``source=None`` plus ``diagnostics.failure_class ==
"candidate_delivery"``; the typed rejection is recorded later.  The single-file branch keeps its
historical ``read_text(errors="replace")`` behaviour byte for byte.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taic.kernel_agent import HmzInvocation, SessionBrief, collect_session, submission_name
from taic.session_supervisor import classify_result
from tcenv.backends.metal.search_candidate import domain_for, preflight_submission
from tcenv.taskcontract.locator import REPO_ROOT, discover_standard_tasks
from tcenv.target import load_target
from tcenv.track.envelope import decode_envelope, encode_envelope
from tcenv.track.plane import CandidateArtifact, CandidateEntry
from tests.support.shape_portfolio_metal import (
    DEVICE_A,
    HOST_TILING_A,
    geometry_bytes,
    geometry_document,
    write_p1_task,
)

CRLF_DEVICE = DEVICE_A.replace(b"\n", b"\r\n")


def _brief(tmp_path: Path, *, track_id: str = "msl-bundle", v2: bool = True) -> SessionBrief:
    work = tmp_path / "agent_work" / "init"
    work.mkdir(parents=True, exist_ok=True)
    return SessionBrief(
        repo=tmp_path,
        task_name="shape_portfolio_pointwise",
        dtype="fp16",
        size="threshold",
        kernel_name="variant_a",
        triton_source="",
        description="a pointwise task",
        contract_text="",
        abi_stub="",
        rtol=1e-3,
        atol_rel=1e-3,
        seedless=True,
        v2=v2,
        track_id=track_id,
        work_dir=str(work.resolve()),
        label="init",
    )


def _invocation(*, status: str = "ok", saw_tokens: bool = True, cost_usd: float | None = 1.25) -> HmzInvocation:
    return HmzInvocation(
        status=status,
        returncode=0 if status == "ok" else 1,
        stdout="",
        stderr="",
        error="",
        spent={"input": 10, "output": 10, "cache_read": 0, "cache_write": 0},
        saw_tokens=saw_tokens,
        cost_usd=cost_usd,
        elapsed_s=1.0,
        argv=("hmz",),
    )


def _bundle_text(device: bytes = DEVICE_A) -> str:
    return encode_envelope(
        CandidateArtifact(
            track_id="msl-bundle",
            entries=(
                CandidateEntry(path="device/variant_a.metal", role="device_kernel", media_type="text/plain", data=device),
                CandidateEntry(path="host/tiling.json", role="host_tiling", media_type="application/json", data=HOST_TILING_A),
                CandidateEntry(
                    path="host/execution_geometry.json",
                    role="schedule",
                    media_type="application/json",
                    data=geometry_bytes(geometry_document("variant_a")),
                ),
            ),
        )
    )


def _write_submission(brief: SessionBrief, text: bytes) -> Path:
    work = Path(brief.work_dir)
    path = work / submission_name(brief.task_name, brief.dtype, brief.size, brief.label, brief.file_ext)
    path.write_bytes(text)
    return path


def test_a_crlf_envelope_survives_collect_and_decodes_byte_identically(tmp_path):
    brief = _brief(tmp_path)
    text = _bundle_text(CRLF_DEVICE)
    # Today's ``read_text`` normalizes CRLF and the body/marker arithmetic then breaks, so this test
    # fails without the byte-exact bundle branch.
    path = _write_submission(brief, text.encode("utf-8"))
    raw = path.read_bytes()
    result = collect_session(brief, Path(brief.work_dir), _invocation())
    assert result.status == "ok"
    assert result.diagnostics["failure_class"] == ""
    assert result.source == text
    candidate = decode_envelope(result.source)
    entry = next(item for item in candidate.entries if item.path == "device/variant_a.metal")
    assert entry.data == CRLF_DEVICE
    assert candidate.candidate_sha256 == decode_envelope(text).candidate_sha256
    import hashlib

    assert hashlib.sha256(raw).hexdigest() == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_the_collected_bundle_passes_the_real_preflight(tmp_path):
    root = tmp_path / "tasks"
    write_p1_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    brief = _brief(tmp_path)
    _write_submission(brief, _bundle_text(CRLF_DEVICE).encode("utf-8"))
    result = collect_session(brief, Path(brief.work_dir), _invocation())
    outcome = preflight_submission(
        locator.load(), result.source, target=load_target("apple-m4-max"), track_id="msl-bundle"
    )
    assert outcome.ok, outcome.rejection
    assert domain_for(locator.load(), load_target("apple-m4-max"))[0].digest  # the plan is the same one


def test_invalid_utf8_with_a_known_cost_keeps_the_bill_and_the_ok_status(tmp_path):
    brief = _brief(tmp_path)
    _write_submission(brief, b"\xff\xfe not utf-8")
    result = collect_session(brief, Path(brief.work_dir), _invocation(saw_tokens=True, cost_usd=1.25))
    assert result.source is None
    # The session completed normally; only its bytes are not text.
    assert result.status == "ok"
    assert result.cost_incomplete is False
    assert result.cost_usd == 1.25
    assert result.diagnostics["failure_class"] == "candidate_delivery"
    assert result.diagnostics["delivery"]["reason"] == "invalid_utf8"
    assert result.diagnostics["delivery"]["stage"] == "delivery"
    assert classify_result(result, None) == "delivery_rejected"


def test_invalid_utf8_with_an_unknown_cost_stays_incomplete(tmp_path):
    brief = _brief(tmp_path)
    _write_submission(brief, b"\xff\xfe not utf-8")
    result = collect_session(brief, Path(brief.work_dir), _invocation(saw_tokens=False, cost_usd=None))
    # The cost completeness rule must not be rewritten: an unknown bill still stops the ledger.
    assert result.cost_incomplete is True
    assert result.cost_usd is None
    assert result.diagnostics["failure_class"] == "candidate_delivery"


@pytest.mark.parametrize("status", ["timeout", "failed"])
def test_a_timeout_or_process_failure_is_never_relabelled_as_a_delivery_rejection(tmp_path, status):
    brief = _brief(tmp_path)
    _write_submission(brief, b"\xff\xfe not utf-8")
    result = collect_session(brief, Path(brief.work_dir), _invocation(status=status))
    assert result.status == status
    assert result.diagnostics["failure_class"] != "candidate_delivery"
    assert "delivery" not in result.diagnostics
    assert classify_result(result, None) == "agent_failed"


def test_a_missing_bundle_submission_is_still_a_plain_no_submission(tmp_path):
    brief = _brief(tmp_path)
    result = collect_session(brief, Path(brief.work_dir), _invocation())
    assert result.source is None
    assert result.diagnostics["failure_class"] == "no_submission"
    assert classify_result(result, None) == "agent_failed"


def test_the_single_file_branch_keeps_its_replacement_character_behaviour(tmp_path):
    brief = _brief(tmp_path, track_id="msl-single")
    good = "kernel void variant_a() {}\n// \xe9\n"
    _write_submission(brief, good.encode("utf-8").replace(b"\n", b"\r\n"))
    result = collect_session(brief, Path(brief.work_dir), _invocation())
    # The historical single-file path normalizes CRLF via read_text and keeps the kernel text.
    assert result.source is not None
    assert "\r" not in result.source
    assert result.diagnostics["failure_class"] == ""
    # An invalid byte degrades to U+FFFD instead of becoming a delivery rejection.
    _write_submission(brief, b"kernel void variant_a() {}\n// \xff\n")
    second = collect_session(brief, Path(brief.work_dir), _invocation())
    assert second.diagnostics["failure_class"] == ""
    assert second.source is not None and "\ufffd" in second.source
