"""B2：``msl-bundle`` 的 track 接线——运行对象、记录、命令、resume 与交卷闭环。

The assertions are on the objects that actually run (the orchestrator's binding, the real
evaluator's binding and the ``SessionBrief`` the session receives), not on a local variable: the
v4/v7 failures of the plan were exactly "the track was set on a copy nothing consumes".
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from taic.cli import _recorded_track, _requested_track
from taic.kernel_agent import HmzInvocation, SessionBrief, collect_session, submission_name
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator, RunStartError
from taic.task_binding import bind_task
from taic.v2_evaluation import V2LocalEvaluator
from tcenv.taskcontract.locator import REPO_ROOT, discover_standard_tasks
from tcenv.track.envelope import encode_envelope
from tcenv.track.plane import CandidateArtifact, CandidateEntry
from tests.support.shape_portfolio_metal import (
    DEVICE_A,
    HOST_TILING_A,
    geometry_bytes,
    geometry_document,
    write_p1_task,
)
from tests.support.fakes import FakeClient, make_response

OP = "shape_portfolio_pointwise"


def _binding(tmp_path: Path, *, track_id: str = "msl-bundle", workspace: bool = True):
    root = tmp_path / "tasks"
    write_p1_task(root, workspace=workspace)
    return replace(bind_task(OP, "fp16", "threshold", tasks_root=root), track_id=track_id)


def _envelope(device: bytes = DEVICE_A) -> str:
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


def _llm() -> LLMConfig:
    return LLMConfig(
        base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake",
        reasoning_effort="off", use_tools=False, truncation_retry=False,
    )


def _search(**overrides) -> SearchConfig:
    off = dict(
        stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False,
        status_lines=False, plan_ledger=False,
    )
    values = dict(budget_usd=5.0, branches=1, max_rounds=0, eval_reps=3, kernel_agent="pi", correctness_only=True)
    values.update(off)
    values.update(overrides)
    return SearchConfig(**values)


class _FakeCandidate:
    """The three fields ``_failure_record`` reads from a candidate."""

    def __init__(self, *, label: str, source: str):
        self.idx = 0
        self.label = label
        self.source = source
        self.report = None


class _NeverEvaluates(V2LocalEvaluator):
    """A v2 evaluator that records requests and never touches the GPU."""

    def __init__(self, binding, calls):
        super().__init__(binding)
        self.calls = calls

    def _evaluate(self, request):  # pragma: no cover - a bundle session must never reach this
        self.calls.append(request)
        raise AssertionError("a delivery-rejected session must not reach the evaluator")


def _orchestrator(tmp_path: Path, *, binding=None, evaluator=None, agent_runner=None, track_id="msl-bundle"):
    binding = binding or _binding(tmp_path)
    return Orchestrator(
        OP, "fp16", "threshold", _llm(), _search(),
        run_dir=tmp_path / "run",
        evaluator=evaluator,
        client=FakeClient(lambda messages, **kw: make_response("no submission")),
        agent_runner=agent_runner,
        log=lambda s: None,
        track_id=track_id,
        binding=binding,
    )


def test_the_three_running_objects_carry_the_bundle_track(tmp_path):
    calls: list = []
    binding = _binding(tmp_path)
    orch = _orchestrator(tmp_path, binding=binding, evaluator=_NeverEvaluates(binding, calls))
    assert orch.binding.track_id == "msl-bundle"
    assert orch.track_id == "msl-bundle"
    assert orch.track.delivery == "candidate_bundle"
    # The real evaluator is the one the search loop calls.
    assert orch.evaluator.binding.track_id == "msl-bundle"
    brief = orch._session_brief(seedless=True, label="init", work_dir=tmp_path / "w")
    assert brief.track_id == "msl-bundle"
    assert brief.abi_stub  # derived from the same binding, not a stale single-file stub


def test_the_single_default_is_unchanged(tmp_path):
    binding = _binding(tmp_path, track_id="msl-single")
    calls: list = []
    orch = _orchestrator(tmp_path, binding=binding, evaluator=_NeverEvaluates(binding, calls), track_id="msl-single")
    assert orch.binding.track_id == "msl-single"
    assert orch.evaluator.binding.track_id == "msl-single"
    assert orch._session_brief(seedless=True, label="init", work_dir=tmp_path / "w").track_id == "msl-single"


def test_a_bundle_run_on_a_task_without_a_workspace_is_refused_at_startup(tmp_path):
    binding = _binding(tmp_path, workspace=False)
    with pytest.raises(RunStartError, match="workspace"):
        _orchestrator(tmp_path, binding=binding)


def test_an_injected_evaluator_must_deliver_the_same_mechanism(tmp_path):
    single = _binding(tmp_path, track_id="msl-single")
    with pytest.raises(RunStartError, match="delivery mechanism"):
        _orchestrator(tmp_path, binding=_binding(tmp_path), evaluator=_NeverEvaluates(single, []))


def test_an_unknown_track_is_a_startup_refusal(tmp_path):
    with pytest.raises(RunStartError, match="cannot be selected"):
        _orchestrator(tmp_path, track_id="no-such-track")


def test_the_check_command_carries_the_track_only_for_a_bundle(tmp_path):
    from taic.kernel_agent import _check_command

    bundle = _orchestrator(tmp_path, evaluator=_NeverEvaluates(_binding(tmp_path), []))
    brief = bundle._session_brief(seedless=True, label="init", work_dir=tmp_path / "w")
    assert "--track msl-bundle" in _check_command(brief, "quick")
    single_binding = _binding(tmp_path, track_id="msl-single")
    single = _orchestrator(
        tmp_path / "s", binding=single_binding, evaluator=_NeverEvaluates(single_binding, []), track_id="msl-single"
    )
    single_brief = single._session_brief(seedless=True, label="init", work_dir=tmp_path / "ws")
    command = _check_command(single_brief, "quick")
    assert "--track" not in command
    assert command.startswith("uv run --directory")


def test_the_retest_command_carries_the_track_only_for_a_bundle(tmp_path):
    from tcenv.evaluate import EvalReport

    bundle = _orchestrator(tmp_path, evaluator=_NeverEvaluates(_binding(tmp_path), []))
    bundle.best = None
    cand = _FakeCandidate(label="init", source=_envelope())
    report = EvalReport(task=OP, dtype="fp16", size="threshold", mode="full", status="incorrect", stage="correctness")
    record = bundle._failure_record(cand, report, 7, "new_best", True)
    assert record["retest_command"].endswith("--track msl-bundle")
    single_binding = _binding(tmp_path, track_id="msl-single")
    single = _orchestrator(
        tmp_path / "s", binding=single_binding, evaluator=_NeverEvaluates(single_binding, []), track_id="msl-single"
    )
    record = single._failure_record(cand, report, 7, "new_best", True)
    assert "--track" not in record["retest_command"]


# --------------------------------------------------------------- run record + resume rules


def test_the_run_and_result_track_blocks_come_from_the_registry(tmp_path):
    orch = _orchestrator(tmp_path, evaluator=_NeverEvaluates(_binding(tmp_path), []))
    document = json.loads((tmp_path / "run" / "run_config.json").read_text())
    assert document["track"] == {"track_id": "msl-bundle", "track_version": 1, "delivery": "candidate_bundle"}


def _run_dir(tmp_path: Path, doc: dict) -> Path:
    folder = tmp_path / "recorded"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run_config.json").write_text(json.dumps(doc))
    return folder


def test_resume_reads_the_recorded_track(tmp_path):
    folder = _run_dir(tmp_path, {"task_mode": "v2", "track": {"track_id": "msl-bundle", "track_version": 1, "delivery": "candidate_bundle"}})
    assert _recorded_track(folder) == "msl-bundle"
    # The caller may restate it, but not change it.
    assert _recorded_track(folder, requested="msl-bundle") == "msl-bundle"
    with pytest.raises(Exception, match="does not match"):
        _recorded_track(folder, requested="msl-single")


def test_resume_tolerates_a_legacy_v2_record_without_a_track_block(tmp_path):
    folder = _run_dir(tmp_path, {"task_mode": "v2"})
    assert _recorded_track(folder) == "msl-single"


def test_resume_keeps_a_v1_run_at_none(tmp_path):
    folder = _run_dir(tmp_path, {"task_mode": "v1", "track": None})
    assert _recorded_track(folder) is None
    with pytest.raises(Exception, match="msl-single"):
        _recorded_track(folder, requested="msl-bundle")


@pytest.mark.parametrize(
    "block,match",
    [
        ("not-a-dict", "malformed"),
        ({"track_id": "msl-bundle", "track_version": 1}, "missing"),
        ({"track_id": "no-such-track", "track_version": 1, "delivery": "single_source"}, "unknown track"),
        ({"track_id": "msl-bundle", "track_version": 9, "delivery": "candidate_bundle"}, "registered at version"),
        ({"track_id": "msl-bundle", "track_version": 1, "delivery": "single_source"}, "registry says"),
    ],
)
def test_a_malformed_or_contradictory_recorded_track_is_refused(tmp_path, block, match):
    folder = _run_dir(tmp_path, {"task_mode": "v2", "track": block})
    with pytest.raises(Exception, match=match):
        _recorded_track(folder)


def test_the_requested_track_precedence(tmp_path):
    class Args:
        def __init__(self, track):
            self.track = track

    assert _requested_track(Args(None), {}) is None
    assert _requested_track(Args(None), {"search": {"track": "msl-bundle"}}) == "msl-bundle"
    assert _requested_track(Args("msl-single"), {"search": {"track": "msl-bundle"}}) == "msl-single"
    # The parser advertises the flag and defaults to the omission sentinel, not to "msl-single".
    import subprocess
    import sys as _sys

    out = subprocess.run(
        [_sys.executable, "-m", "taic.cli", "run", "--help"], capture_output=True, text=True, timeout=120,
    )
    assert "--track {msl-single,msl-bundle,tilelang-pipeline}" in out.stdout
    resume_help = subprocess.run(
        [_sys.executable, "-m", "taic.cli", "resume", "--help"], capture_output=True, text=True, timeout=120,
    )
    # Resume parses every registered id (MT1b registered the interface-only TileLang Track); a new
    # run still only offers the available MSL Tracks above.
    assert "--track {msl-single,msl-bundle,tilelang-pipeline}" in resume_help.stdout


# ------------------------------------------------------- G21 end to end through one attempt


class _Session:
    """An agent runner that writes a submission and then collects it like a real session."""

    def __init__(self, payload: bytes, *, saw_tokens: bool = True, status: str = "ok"):
        self.payload = payload
        self.saw_tokens = saw_tokens
        self.status = status
        self.calls = 0

    def __call__(self, brief: SessionBrief, tag: str):
        self.calls += 1
        work = Path(brief.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        name = submission_name(brief.task_name, brief.dtype, brief.size, brief.label, brief.file_ext)
        (work / name).write_bytes(self.payload)
        invocation = HmzInvocation(
            status=self.status, returncode=0, stdout="", stderr="", error="",
            spent={"input": 10, "output": 10, "cache_read": 0, "cache_write": 0},
            saw_tokens=self.saw_tokens,
            cost_usd=1.25 if self.saw_tokens else None,
            elapsed_s=1.0, argv=("hmz",),
        )
        return collect_session(brief, work, invocation)


def _attempt(orch, *, seedless=True):
    return asyncio.run(orch._one_session_attempt("t0", seedless=seedless, candidate_label="t0", attempt=0))


def test_a_crlf_bundle_reaches_the_evaluator_byte_identically(tmp_path):
    payload = _envelope(DEVICE_A.replace(b"\n", b"\r\n")).encode("utf-8")
    calls: list = []

    class _Record(V2LocalEvaluator):
        def _evaluate(self, request):
            calls.append(request)
            from tcenv.evaluate import EvalReport
            from tcenv.evaluation import EvaluationResult

            report = EvalReport(
                task=request.task, dtype=request.dtype, size=request.size, mode=request.mode,
                status="ok", stage="done", target_id=request.target_id,
            )
            return EvaluationResult(report=report, provenance=self._provenance(request))

    binding = _binding(tmp_path)
    orch = _orchestrator(tmp_path, binding=binding, evaluator=_Record(binding), agent_runner=_Session(payload))
    outcome = _attempt(orch)
    assert outcome == "evaluated"
    assert len(calls) == 1
    from tcenv.track.envelope import decode_envelope

    assert decode_envelope(calls[0].source).entries[0].data.endswith(b"\r\n")


def test_invalid_utf8_with_a_known_cost_is_a_typed_rejection(tmp_path):
    payload = b"\xff\xfe not utf-8"
    binding = _binding(tmp_path)
    orch = _orchestrator(tmp_path, binding=binding, evaluator=_NeverEvaluates(binding, []), agent_runner=_Session(payload))
    outcome = _attempt(orch)
    assert outcome == "delivery_rejected"
    # The billed cost is still charged, and the candidate row exists with an empty source.
    assert orch.ledger.total_usd == pytest.approx(1.25)
    assert orch.candidates and orch.candidates[-1].source == ""
    report = orch.candidates[-1].report
    assert report is not None and report.status == "rejected" and report.stage == "delivery"
    assert orch.trace[-1]["status"] == "rejected"
    assert orch._failure_feedback
    # No evaluation happened.
    assert orch.evaluations == 0
    assert orch.best is None


def test_invalid_utf8_with_an_unknown_cost_still_stops_the_ledger(tmp_path):
    from taic.orchestrator import StopRun

    payload = b"\xff\xfe not utf-8"
    binding = _binding(tmp_path)
    orch = _orchestrator(
        tmp_path, binding=binding, evaluator=_NeverEvaluates(binding, []),
        agent_runner=_Session(payload, saw_tokens=False),
    )
    with pytest.raises(StopRun, match="cost_incomplete"):
        _attempt(orch)
    assert orch.ledger.total_usd == 0.0


def test_the_rejection_left_a_placeholder_kernel_that_resume_can_read(tmp_path):
    payload = b"\xff\xfe not utf-8"
    binding = _binding(tmp_path)
    orch = _orchestrator(tmp_path, binding=binding, evaluator=_NeverEvaluates(binding, []), agent_runner=_Session(payload))
    _attempt(orch)
    cand = orch.candidates[-1]
    folder = orch.run_dir / "candidates" / f"{cand.idx:03d}_{cand.label}"
    assert (folder / orch.kernel_name).read_bytes() == b""
    assert (folder / "report.json").is_file()
    # An empty source is never a trusted cache entry.
    assert "" not in {key for key in orch.cache}
    assert orch._cache_this("") is False
