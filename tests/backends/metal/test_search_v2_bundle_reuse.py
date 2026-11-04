"""B1：投影与 lifecycle 复用证明（只读证明，不新增机制）。

An ``msl-bundle`` search candidate reuses the shipped projection, `HostTilingPlan`, `CandidateLifecycle`
and geometry authority.  These tests prove that reuse from the outside: the projection's track, the
parsed host plan, the lifecycle's prepare steps, the derived ``inside_timing`` and the authority rule
that dispatch comes from the geometry entry rather than from the source's launch block.
"""

from __future__ import annotations

import json

import pytest

from tcenv.backends.metal.search_candidate import domain_for, portfolio_for_submission
from tcenv.backends.metal.track_adapter import combined_candidate_source
from tcenv.backends.metal.v2_runtime import HostTilingPlan
from tcenv.contract_errors import GeometryError, TaskContractError
from tcenv.portfolio.candidate import project_provider
from tcenv.taskcontract.locator import REPO_ROOT, discover_standard_tasks
from tcenv.taskcontract.portfolio_runner import _lifecycle_factory
from tcenv.target import load_target
from tcenv.track.delivery import candidate_geometry
from tcenv.track.envelope import decode_envelope, encode_envelope
from tcenv.track.geometry import resolve_geometry
from tcenv.track.plane import CandidateArtifact, CandidateEntry
from tests.support.shape_portfolio_metal import (
    DEVICE_A,
    HOST_TILING_A,
    geometry_bytes,
    geometry_document,
    write_p1_task,
)

TARGET = load_target("apple-m4-max")


def _contract(tmp_path):
    root = tmp_path / "tasks"
    write_p1_task(root)
    return discover_standard_tasks(root, code_root=REPO_ROOT)[0].load()


def _submission(*, zero: bool = False, geometry: dict | None = None) -> str:
    document = geometry or geometry_document("variant_a", workspace_zero=zero)
    entries = (
        CandidateEntry(path="device/variant_a.metal", role="device_kernel", media_type="text/plain", data=DEVICE_A),
        CandidateEntry(path="host/tiling.json", role="host_tiling", media_type="application/json", data=HOST_TILING_A),
        CandidateEntry(
            path="host/execution_geometry.json",
            role="schedule",
            media_type="application/json",
            data=geometry_bytes(document),
        ),
    )
    return encode_envelope(CandidateArtifact(track_id="msl-bundle", entries=entries))


def _projection(contract, submission: str):
    plan, authority = domain_for(contract, TARGET)
    _candidate, portfolio = portfolio_for_submission(
        submission, plan, authority=authority, track_id="msl-bundle"
    )
    return project_provider(portfolio, portfolio.variants[0], authority=authority)


def _resolved(contract, projection, submission: str):
    return resolve_geometry(
        candidate_geometry(decode_envelope(submission)), instance=contract.instance,
        contract=__import__("tcenv.track.abi", fromlist=["derive_tensor_memory_contract"]).derive_tensor_memory_contract(
            contract.instance, target=TARGET
        ),
        limits=__import__("tcenv.track.abi", fromlist=["abi_limits_from_target"]).abi_limits_from_target(TARGET),
    )


def test_the_projection_is_the_bundle_track_and_parses_the_host_plan(tmp_path):
    contract = _contract(tmp_path)
    submission = _submission()
    projection = _projection(contract, submission)
    assert projection.track_id == "msl-bundle"
    assert {entry.path: entry.role for entry in projection.candidate.entries} == {
        "device/variant_a.metal": "device_kernel",
        "host/tiling.json": "host_tiling",
        "host/execution_geometry.json": "schedule",
    }
    host_plan = HostTilingPlan.from_candidate(projection.candidate, contract)
    assert host_plan is not None
    assert [write.workspace for write in host_plan.writes] == ["scratch"]
    assert host_plan.writes[0].kind == "linear"


def test_the_lifecycle_host_steps_are_host_tiling_then_schedule(tmp_path):
    contract = _contract(tmp_path)
    submission = _submission()
    projection = _projection(contract, submission)
    resolved = _resolved(contract, projection, submission)
    factory = _lifecycle_factory(
        projection=projection, resolved=resolved, contract=contract, target_id=TARGET.id
    )
    assert factory is not None
    lifecycle = factory(resolved)
    steps = tuple(step.declaration.name for step in lifecycle.host_steps)
    assert steps == ("host_tiling", "schedule")
    assert lifecycle.track_id == "msl-bundle"
    assert lifecycle.candidate_sha256 == projection.candidate.candidate_sha256


@pytest.mark.parametrize("zero,inside_timing", [(False, False), (True, True)])
def test_inside_timing_is_derived_from_the_declared_zero_workspace(tmp_path, zero, inside_timing):
    contract = _contract(tmp_path)
    submission = _submission(zero=zero)
    projection = _projection(contract, submission)
    resolved = _resolved(contract, projection, submission)
    factory = _lifecycle_factory(
        projection=projection, resolved=resolved, contract=contract, target_id=TARGET.id
    )
    lifecycle = factory(resolved)
    host_tiling = next(step for step in lifecycle.host_steps if step.declaration.name == "host_tiling")
    assert host_tiling.declaration.inside_timing is inside_timing
    assert host_tiling.declaration.input_dependent is False
    schedule = next(step for step in lifecycle.host_steps if step.declaration.name == "schedule")
    assert schedule.declaration.inside_timing is False


def test_dispatch_comes_from_the_geometry_entry_not_from_the_launch_block(tmp_path):
    contract = _contract(tmp_path)
    # The device source says grid=N,1,1 threads=1,1,1; the geometry says threads=32.
    submission = _submission(geometry=geometry_document("variant_a", threads=32))
    projection = _projection(contract, submission)
    resolved = _resolved(contract, projection, submission)
    stage = resolved.stages[0]
    assert stage.threads == (32, 1, 1)
    assert "threads=1,1,1" in combined_candidate_source(projection.candidate)


def test_a_geometry_that_disagrees_with_the_delivery_is_refused(tmp_path):
    contract = _contract(tmp_path)
    # ``source`` must be "candidate": a fallback geometry cannot arrive inside a candidate.
    document = dict(geometry_document("variant_a"))
    document["source"] = "fallback"
    submission = _submission(geometry=document)
    # The closed geometry parser refuses it while the parent portfolio is being read.
    with pytest.raises(GeometryError, match="source 'candidate'"):
        _projection(contract, submission)


def test_a_geometry_workspace_the_launch_does_not_declare_is_a_problem(tmp_path):
    contract = _contract(tmp_path)
    from tcenv.backends.metal.launch import parse_launch_spec
    from tcenv.backends.metal.track_adapter import launch_constants
    from tcenv.backends.metal.v2_runtime import workspace_size_problems
    from tcenv.track.abi import derive_tensor_memory_contract

    submission = _submission()  # the geometry uses ``scratch``; the source only declares math.
    projection = _projection(contract, submission)
    resolved = _resolved(contract, projection, submission)
    source = combined_candidate_source(projection.candidate).replace(
        "workspace scratch N*4\n", ""
    ).replace(" workspace=scratch", "")
    spec = parse_launch_spec(source, launch_constants(contract.instance))
    problems = workspace_size_problems(derive_tensor_memory_contract(contract.instance, target=TARGET), resolved, spec)
    assert any("does not declare" in problem for problem in problems)


def test_the_host_plan_must_write_a_declared_workspace(tmp_path):
    contract = _contract(tmp_path)
    bad = json.dumps(
        {
            "schema_id": "tcenv.metal.host-tiling.1",
            "schema_version": 1,
            "writes": [
                {"workspace": "not_a_workspace", "kind": "linear", "start": 0, "step": 0,
                 "count": 4, "stride": 1, "offset": 0, "dtype": "fp32"}
            ],
        }
    ).encode()
    entries = (
        CandidateEntry(path="device/variant_a.metal", role="device_kernel", media_type="text/plain", data=DEVICE_A),
        CandidateEntry(path="host/tiling.json", role="host_tiling", media_type="application/json", data=bad),
        CandidateEntry(
            path="host/execution_geometry.json",
            role="schedule",
            media_type="application/json",
            data=geometry_bytes(geometry_document("variant_a")),
        ),
    )
    submission = encode_envelope(CandidateArtifact(track_id="msl-bundle", entries=entries))
    with pytest.raises(TaskContractError, match="unknown workspace"):
        _projection(contract, submission)


def test_the_bundle_confirmation_freezes_the_bundle_generation(tmp_path):
    from tcenv.backends.metal.search_candidate import confirmation_input

    contract = _contract(tmp_path)
    submission = _submission()
    root = tmp_path / "tasks"
    from tcenv.taskcontract.locator import discover_standard_tasks

    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    frozen = confirmation_input(
        locator, submission, target=TARGET, guidance_digest="0" * 64, master_seed=b"0123456789abcdef",
        track_id="msl-bundle",
    )
    # Bundle evidence is a later generation of the same frozen subject.
    assert frozen.freeze.protocol_epoch == 2
    assert frozen.freeze.portfolio().planner.planner_id == "taic.search.bundle"
    # The single-file generation is unchanged.
    single = confirmation_input(
        locator, "kernel void variant_b() {}\n/* taic-launch\nmath safe\nlaunch variant_b grid=N,1,1 threads=32,1,1\n*/\n",
        target=TARGET, guidance_digest="0" * 64, master_seed=b"0123456789abcdef",
    )
    assert single.freeze.protocol_epoch == 1
    assert single.freeze.portfolio().planner.planner_id == "taic.search.single"
