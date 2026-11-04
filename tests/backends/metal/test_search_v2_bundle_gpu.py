"""B5-a：``msl-bundle`` 的搜索垂直切片（真实 Metal，需 GPU 许可）。

Everything here goes through the **search** entry points (`evaluate_candidate`, `V2LocalEvaluator`,
`confirm_candidate`), not through a hand-built portfolio: the question is whether the search wrapper
produces the same evidence the shipped A4 slice does for host tiling, ``inside_timing``, multi-stage
and multi-source bundles.

Run with::

    uv run pytest -q -n 0 -m gpu tests/backends/metal/test_search_v2_bundle_gpu.py

Dynamic ``[[threadgroup(k)]]`` parameters are **not supported in this delivery** (the ABI refuses
them), so that case is a negative here, not a positive.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from taic.task_binding import bind_task
from taic.v2_evaluation import V2LocalEvaluator
from tcenv.backends.metal.search_candidate import (
    confirm_candidate,
    confirmation_input,
    domain_for,
    evaluate_candidate,
    request_for,
    usage_for,
)
from tcenv.contract_errors import TaskContractError
from tcenv.evaluation import EvaluationRequest
from tcenv.taskcontract.locator import REPO_ROOT
from tcenv.target import load_target
from tcenv.track.envelope import encode_envelope
from tcenv.track.plane import CandidateArtifact, CandidateEntry
pytestmark = pytest.mark.gpu

TASK = "affine_ramp"
SUPPORT = REPO_ROOT / "tests" / "support" / "msl_bundle_affine_ramp" / "host"


def _known_good() -> bytes:
    return (REPO_ROOT / "tasks" / TASK / "known_good.metal").read_bytes()


def _tiling() -> bytes:
    return (SUPPORT / "tiling.json").read_bytes()


def _geometry(*, zero: bool = False, stages: list | None = None) -> bytes:
    if stages is not None:
        return json.dumps(
            {"schema_version": 1, "source": "candidate", "stages": stages}, indent=2, sort_keys=True
        ).encode() + b"\n"
    name = "execution_geometry_zero.json" if zero else "execution_geometry.json"
    return (SUPPORT / name).read_bytes()


def _bundle(
    *,
    zero: bool = False,
    device: list[tuple[str, bytes]] | None = None,
    tiling: bytes | None = None,
    stages: list | None = None,
) -> str:
    entries = [
        CandidateEntry(path=path, role="device_kernel", media_type="text/plain", data=data)
        for path, data in (device or [("device/affine_ramp.metal", _known_good())])
    ]
    entries.append(
        CandidateEntry(
            path="host/tiling.json", role="host_tiling", media_type="application/json", data=tiling or _tiling()
        )
    )
    entries.append(
        CandidateEntry(
            path="host/execution_geometry.json",
            role="schedule",
            media_type="application/json",
            data=_geometry(zero=zero, stages=stages),
        )
    )
    return encode_envelope(CandidateArtifact(track_id="msl-bundle", entries=tuple(entries)))


def _locator(tmp_path: Path, size: str = "small"):
    """The search's real locator: it is the one ``bind_task`` builds for this exact instance."""
    if not (tmp_path / "tasks" / TASK).is_dir():
        import shutil

        shutil.copytree(REPO_ROOT / "tasks" / TASK, tmp_path / "tasks" / TASK)
    return bind_task(TASK, "fp16", size, tasks_root=tmp_path / "tasks").locator


def _binding(tmp_path: Path, size: str = "small"):
    _locator(tmp_path, size)  # materialize the task tree first
    return replace(bind_task(TASK, "fp16", size, tasks_root=tmp_path / "tasks"), track_id="msl-bundle")


def _evaluate(tmp_path: Path, submission: str, *, size: str = "small", mode: str = "full", reps: int = 3):
    locator = _locator(tmp_path, size)
    return evaluate_candidate(
        locator, submission, target=load_target("apple-m4-max"), mode=mode, reps=reps, track_id="msl-bundle"
    )


# ------------------------------------------------------------------ 1. host tiling really runs


def test_the_search_path_executes_the_host_tiling_plan(tmp_path):
    run = _evaluate(tmp_path, _bundle(), mode="quick")
    assert run.projection.track_id == "msl-bundle"
    evaluation = run.evaluation
    assert evaluation["host_runs"], "the Host Tiling plan was never executed"
    assert evaluation["host_runs"][-1]["writes"] == 4 * 16  # NS = ceil(1000/64) = 16
    assert evaluation["workspace_bytes"] == {"scale_tab": 4 * 16}  # the ABI span, not "the geometry"
    assert evaluation["workspace_zero"] == {"scale_tab": False}
    assert evaluation["lifecycle"]["prepare"][0]["role"] == "host_tiling"
    assert evaluation["lifecycle"]["prepare"][0]["inside_timing"] is False
    assert evaluation["dispatch"] == evaluation["expected_dispatch"]
    assert evaluation["dispatch_stages"] == evaluation["expected_dispatch_stages"]


def test_the_workspace_span_equals_the_abi_not_the_geometry(tmp_path):
    run = _evaluate(tmp_path, _bundle(), mode="quick")
    span = 4 * int(_locator(tmp_path).load().instance.symbols["NS"])
    assert run.evaluation["workspace_bytes"] == {"scale_tab": span}


# ------------------------------------------------------- 2. derived lifecycle inside the timer


def test_a_declared_zero_workspace_moves_the_lifecycle_inside_timing(tmp_path):
    run = _evaluate(tmp_path, _bundle(zero=True), mode="full", reps=3)
    evaluation = run.evaluation
    assert evaluation["workspace_zero"] == {"scale_tab": True}
    assert evaluation["lifecycle"]["prepare"][0]["inside_timing"] is True
    assert run.result is not None
    assert run.result.execution["workspace_fills"], "no workspace fill was recorded"
    assert run.result.report["benchmark"]["candidate"]["median_us"] > 0


# ----------------------------------------------------- 3. multi-stage (+ the dynamic negative)


MULTI_STAGES = [
    {
        "kernel": "affine_ramp", "grid": ["N", 1, 1], "threads": [32, 1, 1],
        "threadgroup_bytes": 0, "workspace": [{"name": "scale_tab", "zero": False}],
        "when": None, "derived_scalars": [],
    },
    {
        "kernel": "affine_ramp_tail", "grid": ["N", 1, 1], "threads": [32, 1, 1],
        "threadgroup_bytes": 0, "workspace": [{"name": "scale_tab", "zero": False}],
        "when": None, "derived_scalars": [],
    },
]

TAIL_DEVICE = b"""\
#include <metal_stdlib>
using namespace metal;
// The last element is recomputed by an independent second stage.  It is the same value, so the
// result is unchanged; what the test proves is that two ordered stages really dispatch.
kernel void affine_ramp_tail(device const half* x   [[buffer(0)]],
                             device half* y         [[buffer(1)]],
                             device float* scale_tab [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
  if (i + 1 == N) {
    y[i] = half(scale_tab[i / SEG] * float(x[i]));
  }
}
"""


def test_a_two_stage_two_source_bundle_runs_both_stages(tmp_path):
    submission = _bundle(
        device=[
            ("device/affine_ramp.metal", _known_good()),
            ("device/affine_ramp_tail.metal", TAIL_DEVICE),
        ],
        stages=MULTI_STAGES,
    )
    run = _evaluate(tmp_path, submission, mode="quick")
    evaluation = run.evaluation
    assert evaluation["dispatch_stages"] == [0, 1]
    assert evaluation["dispatch"] == evaluation["expected_dispatch"]
    assert [item["kernel"] for item in evaluation["dispatch"]] == ["affine_ramp", "affine_ramp_tail"]
    assert all(item["threadgroup_bytes"] == 0 for item in evaluation["dispatch"])
    assert run.result is not None and run.result.report["status"] in ("ok", "incorrect")
    # The second source has no launch block of its own; the join is one compilation unit.
    assert run.projection.candidate.entries[-1].role != "launch"


def test_a_dynamic_threadgroup_parameter_is_refused(tmp_path):
    bad = b"""\
#include <metal_stdlib>
using namespace metal;
kernel void affine_ramp(device const half* x [[buffer(0)]],
                        device half* y [[buffer(1)]],
                        device float* scale_tab [[buffer(2)]],
                        threadgroup float scratch[64] [[threadgroup(0)]],
                        uint i [[thread_position_in_grid]]) {
  y[i] = half(scale_tab[i / SEG] * float(x[i]));
}
/* taic-launch
math safe
workspace scale_tab NS*4
launch affine_ramp grid=N,1,1 threads=32,1,1 workspace=scale_tab
*/
"""
    with pytest.raises(TaskContractError):
        _evaluate(tmp_path, _bundle(device=[("device/affine_ramp.metal", bad)]), mode="quick")


# ------------------------------------------------------------------ 4. the evaluator path


def test_the_v2_evaluator_accepts_the_bundle_and_reports_dispatch(tmp_path):
    binding = _binding(tmp_path)
    evaluator = V2LocalEvaluator(binding)
    request = EvaluationRequest(
        task=TASK, dtype="fp16", size="small", source=_bundle(), mode="quick", seed=7, reps=3,
        target_id="apple-m4-max",
    )
    result = evaluator.formal(request)
    report = result.report if hasattr(result.report, "status") else type("R", (), result.report)
    if hasattr(report, "status"):
        assert report.status in ("ok", "incorrect"), report.message
        assert report.stage != "delivery"
    else:  # a schema-4 dict report
        assert report["status"] in ("ok", "incorrect"), report.get("message")
        assert report["stage"] != "delivery"


def test_a_bundle_session_that_submits_one_file_is_a_typed_rejection(tmp_path):
    binding = _binding(tmp_path)
    evaluator = V2LocalEvaluator(binding)
    request = EvaluationRequest(
        task=TASK, dtype="fp16", size="small", source=_known_good().decode(), mode="quick", seed=7, reps=3,
        target_id="apple-m4-max",
    )
    result = evaluator.formal(request)
    report = result.report
    status = report.status if hasattr(report, "status") else report["status"]
    stage = report.stage if hasattr(report, "stage") else report["stage"]
    assert status == "rejected"
    assert stage == "delivery"


# -------------------------------------------------------- 5. confirmation (freeze + dispatch)


def test_the_confirmation_freezes_the_bundle_subject_and_replays_the_same_dispatch(tmp_path):
    from tcenv.backends.metal import portfolio_confirmation

    locator = _locator(tmp_path, "small")
    contract = locator.load()
    target = load_target("apple-m4-max")
    plan, _authority = domain_for(contract, target)
    frozen = confirmation_input(
        locator, _bundle(), target=target, guidance_digest="0" * 64, master_seed=b"0123456789abcdef",
        track_id="msl-bundle",
    )
    assert frozen.freeze.protocol_epoch == 2
    assert frozen.freeze.portfolio().planner.planner_id == "taic.search.bundle"
    assert usage_for(contract) == "benchmark"

    captured: dict[str, list] = {"search": [], "confirmations": []}
    real = portfolio_confirmation._evaluate

    def spy(*args, **kwargs):
        run = real(*args, **kwargs)
        seed = kwargs.get("seed") if "seed" in kwargs else None
        bucket = "search" if seed == 7 else "confirmations"
        captured[bucket].append(run)
        return run

    portfolio_confirmation._evaluate = spy
    try:
        document = confirm_candidate(
            locator, _bundle(), target=target, guidance_digest="0" * 64, master_seed=b"0123456789abcdef",
            track_id="msl-bundle",
        )
    finally:
        portfolio_confirmation._evaluate = real

    # A tiny instance can legitimately come out ``timing_uncertain`` on a busy machine; that is a
    # measurement verdict, not a delivery failure.  What must hold either way is the identity of the
    # evaluated subject and the per-run dispatch evidence asserted below.
    assert document["status"] == "confirmed" or document.get("reason") == "timing_uncertain", (
        document.get("reason"), document.get("confirmation_report", {}).get("message")
    )
    assert document["identity"] is not None
    assert (document.get("search_report") or {}).get("status") == "ok"
    assert (document.get("confirmation_report") or {}).get("status") == "ok"
    assert captured["search"], "the confirmation never ran its search evaluation"
    assert len(captured["confirmations"]) == int(document["confirmation_attempts"])
    search_run = captured["search"][0]
    for run in captured["search"] + captured["confirmations"]:
        assert run.evaluation["dispatch"] == run.evaluation["expected_dispatch"]
        assert run.evaluation["dispatch_stages"] == run.evaluation["expected_dispatch_stages"]
        assert run.evaluation["host_runs"]
    for run in captured["confirmations"]:
        assert run.evaluation["dispatch_stages"] == search_run.evaluation["dispatch_stages"]
        assert run.record.variant_id == search_run.record.variant_id
