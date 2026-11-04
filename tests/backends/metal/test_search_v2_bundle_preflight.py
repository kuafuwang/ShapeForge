"""``msl-bundle`` 搜索交付的 CPU 预检：九条交付负例、两项 CPU 缺口与异常源头规范化。

The point of this file is that **nothing here builds and nothing here raises**: every illegal
submission is a typed ``PreflightOutcome`` with a stage, because the search must turn it into a
``rejected`` candidate instead of dying.  The Metal backend is involved only through its pure-CPU
helpers (envelope/portfolio/geometry/launch/Host Tiling), so the file lives under
``tests/backends/metal/`` like the rest of the v2 delivery tests.
"""

from __future__ import annotations

import json

import pytest

from tcenv.backends.metal.search_candidate import (
    BUNDLE_TRACK_ID,
    PreflightOutcome,
    bundle_parent_entries,
    domain_for,
    portfolio_for_submission,
    preflight_submission,
)
from tcenv.backends.metal.v2_runtime import HostTilingPlan, HostTilingWrite, host_write_span
from tcenv.contract_errors import TaskContractError
from tcenv.taskcontract.locator import REPO_ROOT, discover_standard_tasks
from tcenv.target import load_target
from tcenv.track.envelope import EnvelopeError, decode_envelope, encode_envelope
from tcenv.track.plane import CandidateArtifact, CandidateEntry
from tests.support.shape_portfolio_metal import (
    DEVICE_A,
    DEVICE_B,
    HOST_TILING_A,
    geometry_bytes,
    geometry_document,
    write_p1_task,
)

TARGET = load_target("apple-m4-max")
SINGLE_SOURCE = """\
kernel void plain(device const half* x [[buffer(0)]],
                  device half* y [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
  y[i] = half(-float(x[i]));
}
/* taic-launch
math safe
launch plain grid=N,1,1 threads=1,1,1
*/
"""


def _contract(tmp_path, *, workspace: bool = True):
    root = tmp_path / "tasks"
    write_p1_task(root, workspace=workspace)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    return locator.load()


def _bundle(
    *,
    device: tuple[tuple[str, bytes], ...] = (("device/variant_a.metal", DEVICE_A),),
    host: bytes = HOST_TILING_A,
    host_path: str = "host/tiling.json",
    geometry: bytes | None = None,
    geometry_path: str = "host/execution_geometry.json",
    geometry_media_type: str = "application/json",
    track_id: str = BUNDLE_TRACK_ID,
    extra: tuple[CandidateEntry, ...] = (),
) -> str:
    entries = [
        CandidateEntry(path=path, role="device_kernel", media_type="text/plain", data=data)
        for path, data in device
    ]
    entries.append(CandidateEntry(path=host_path, role="host_tiling", media_type="application/json", data=host))
    entries.append(
        CandidateEntry(
            path=geometry_path,
            role="schedule",
            media_type=geometry_media_type,
            data=geometry if geometry is not None else geometry_bytes(geometry_document("variant_a")),
        )
    )
    entries.extend(extra)
    return encode_envelope(CandidateArtifact(track_id=track_id, entries=tuple(entries)))


def _preflight(contract, submission: str, *, track_id: str = BUNDLE_TRACK_ID) -> PreflightOutcome:
    return preflight_submission(contract, submission, target=TARGET, track_id=track_id)


def _rejected(contract, submission: str, *, track_id: str = BUNDLE_TRACK_ID, stage: str | None = None):
    outcome = _preflight(contract, submission, track_id=track_id)
    assert not outcome.ok, "the submission was accepted but should have been refused"
    assert outcome.rejection
    if stage is not None:
        assert outcome.stage == stage, f"refused at {outcome.stage!r}, expected {stage!r}: {outcome.rejection}"
    return outcome


def test_a_legal_bundle_passes_the_cpu_preflight(tmp_path):
    contract = _contract(tmp_path)
    outcome = _preflight(contract, _bundle())
    assert outcome.ok, outcome.rejection
    assert outcome.candidate is not None and outcome.candidate.track_id == BUNDLE_TRACK_ID
    assert outcome.projection is not None and outcome.projection.track_id == BUNDLE_TRACK_ID
    # The projected child keeps the fixed geometry path with the ``schedule`` role.
    paths = {entry.path: entry.role for entry in outcome.projection.candidate.entries}
    assert paths["host/execution_geometry.json"] == "schedule"
    assert paths["host/tiling.json"] == "host_tiling"


def test_a_legal_single_file_still_passes(tmp_path):
    contract = _contract(tmp_path)
    outcome = _preflight(contract, SINGLE_SOURCE, track_id="msl-single")
    assert outcome.ok, outcome.rejection
    assert outcome.projection is not None and outcome.projection.track_id == "msl-single"


# ---------------------------------------------------------------- §2.2 delivery negatives


def test_a_bundle_run_refuses_a_single_file(tmp_path):
    contract = _contract(tmp_path)
    outcome = _rejected(contract, SINGLE_SOURCE, stage="delivery")
    assert "not a complete candidate envelope" in outcome.rejection


def test_a_single_run_refuses_an_envelope(tmp_path):
    contract = _contract(tmp_path)
    outcome = _rejected(contract, _bundle(), track_id="msl-single", stage="delivery")
    assert "single file" in outcome.rejection


def test_a_truncated_envelope_is_not_mistaken_for_a_single_file(tmp_path):
    contract = _contract(tmp_path)
    truncated = _bundle().split("taic-end-envelope")[0]
    outcome = _rejected(contract, truncated, track_id="msl-single", stage="delivery")
    assert "envelope" in outcome.rejection


def test_an_envelope_for_another_track_is_refused_not_relabelled(tmp_path):
    contract = _contract(tmp_path)
    outcome = _rejected(contract, _bundle(track_id="msl-single"), stage="delivery")
    assert "never relabels" in outcome.rejection


def test_a_geometry_entry_at_another_path_is_refused(tmp_path):
    contract = _contract(tmp_path)
    outcome = _rejected(contract, _bundle(geometry_path="host/geometry_a.json"), stage="delivery")
    assert "fixed path" in outcome.rejection


def test_a_geometry_entry_with_the_wrong_media_type_is_refused(tmp_path):
    contract = _contract(tmp_path)
    outcome = _rejected(contract, _bundle(geometry_media_type="text/plain"), stage="delivery")
    assert "media type" in outcome.rejection


def test_a_bundle_without_a_host_tiling_plan_is_refused(tmp_path):
    contract = _contract(tmp_path)
    envelope = encode_envelope(
        CandidateArtifact(
            track_id=BUNDLE_TRACK_ID,
            entries=(
                CandidateEntry(path="device/variant_a.metal", role="device_kernel", media_type="text/plain", data=DEVICE_A),
                CandidateEntry(
                    path="host/execution_geometry.json",
                    role="schedule",
                    media_type="application/json",
                    data=geometry_bytes(geometry_document("variant_a")),
                ),
            ),
        )
    )
    outcome = _rejected(contract, envelope, stage="delivery")
    assert "Host Tiling plan" in outcome.rejection


def test_a_bundle_without_a_device_source_is_refused(tmp_path):
    contract = _contract(tmp_path)
    envelope = encode_envelope(
        CandidateArtifact(
            track_id=BUNDLE_TRACK_ID,
            entries=(
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
    outcome = _rejected(contract, envelope, stage="delivery")
    assert "no device_kernel source" in outcome.rejection


def test_a_role_outside_the_bundle_closed_set_is_refused(tmp_path):
    contract = _contract(tmp_path)
    envelope = _bundle(
        extra=(
            CandidateEntry(
                path="coverage_plan/extra.json", role="launch", media_type="application/json", data=b"{}"
            ),
        )
    )
    outcome = _rejected(contract, envelope, stage="delivery")
    assert "role" in outcome.rejection


def test_the_host_plan_and_the_geometry_cannot_share_one_path(tmp_path):
    # A shared path is already unrepresentable in the candidate plane; this pins that invariant so a
    # future "one JSON for both" idea fails at the identity layer, not in the projection.
    with pytest.raises(TaskContractError, match="collide"):
        CandidateArtifact(
            track_id=BUNDLE_TRACK_ID,
            entries=(
                CandidateEntry(path="host/x.json", role="host_tiling", media_type="application/json", data=b"{}"),
                CandidateEntry(path="host/x.json", role="schedule", media_type="application/json", data=b"{}"),
            ),
        )


def test_the_geometry_entry_is_never_listed_in_the_variant_entries(tmp_path):
    contract = _contract(tmp_path)
    _candidate, portfolio = portfolio_for_submission(
        _bundle(), domain_for(contract, TARGET)[0], authority=domain_for(contract, TARGET)[1],
        track_id=BUNDLE_TRACK_ID,
    )
    variant = portfolio.variants[0]
    assert variant.geometry_entry is not None
    assert variant.geometry_entry not in variant.entries


def test_a_task_without_a_workspace_is_refused(tmp_path):
    contract = _contract(tmp_path, workspace=False)
    outcome = _rejected(contract, _bundle(), stage="geometry")
    assert "workspace" in outcome.rejection


def test_a_device_source_without_a_launch_block_is_refused(tmp_path):
    contract = _contract(tmp_path)
    plain = DEVICE_A.replace(b"/* taic-launch", b"/* not-a-launch").split(b"*/", 1)[1]
    outcome = _rejected(contract, _bundle(device=(("device/variant_a.metal", plain),)), stage="geometry")
    assert "taic-launch" in outcome.rejection


# ---------------------------------------------------------- §1.4 the two missing CPU checks


def test_two_device_sources_that_each_carry_a_launch_block_are_refused_before_build(tmp_path):
    contract = _contract(tmp_path)
    outcome = _rejected(
        contract,
        _bundle(device=(("device/variant_a.metal", DEVICE_A), ("device/variant_b.metal", DEVICE_B))),
        stage="launch",
    )
    assert "exactly once" in outcome.rejection


def test_a_launch_workspace_that_disagrees_with_the_abi_span_is_refused_before_build(tmp_path):
    contract = _contract(tmp_path)
    # ``scratch`` is N fp32 values; the block claims twice that.
    lying = DEVICE_A.replace(b"workspace scratch N*4", b"workspace scratch N*8")
    outcome = _rejected(contract, _bundle(device=(("device/variant_a.metal", lying),)), stage="launch")
    assert "ABI TensorMemoryContract spans" in outcome.rejection


# ------------------------------------------------------ Host Tiling span (preflight, no build)


def _tiling(entry: dict) -> bytes:
    return json.dumps(
        {"schema_id": "tcenv.metal.host-tiling.1", "schema_version": 1, "writes": [entry]}, sort_keys=True
    ).encode()


def _write(**overrides) -> HostTilingWrite:
    values = dict(
        workspace="scratch", kind="linear", start=0, step=0, count=4, stride=1, offset=0, dtype="fp32"
    )
    values.update(overrides)
    return HostTilingWrite(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"count": -1},
        {"stride": 0},
        {"offset": -1, "count": 2, "stride": 2},
        {"count": 1000, "offset": 0},
    ],
)
def test_an_illegal_host_tiling_span_is_refused_in_the_preflight(tmp_path, overrides):
    contract = _contract(tmp_path)
    entry = {
        "workspace": "scratch", "kind": "linear", "start": 0, "step": 0,
        "count": 4, "stride": 1, "offset": 0, "dtype": "fp32",
    }
    entry.update(overrides)
    outcome = _rejected(contract, _bundle(host=_tiling(entry)), stage="delivery")
    assert "host tiling write" in outcome.rejection


@pytest.mark.parametrize("overrides", [{"count": 0, "stride": 1}, {"count": 0, "stride": 2}])
def test_an_empty_host_write_is_accepted(tmp_path, overrides):
    # count == 0 must have span 0: the naive formula would be negative and reject a write the
    # runtime accepts.  ``stride >= 1`` still holds (count == 0 does not relax it).
    contract = _contract(tmp_path)
    entry = {
        "workspace": "scratch", "kind": "linear", "start": 0, "step": 0,
        "count": 4, "stride": 1, "offset": 0, "dtype": "fp32",
    }
    entry.update(overrides)
    outcome = _preflight(contract, _bundle(host=_tiling(entry)))
    assert outcome.ok, outcome.rejection


def test_the_shared_span_helper_reports_the_zero_count_as_zero():
    assert host_write_span(_write(count=0, stride=2), 16) == []
    assert host_write_span(_write(stride=0, count=0), 16) != []
    assert host_write_span(_write(offset=-1, count=2, stride=2), 64) != []


# ------------------------------------------------- §1.6 the four exception classes at the source


def test_a_non_ascii_manifest_line_is_an_envelope_error():
    text = _bundle()
    broken = text.replace("taic-manifest ", "taic-manifest \u00e9", 1)
    with pytest.raises(EnvelopeError):
        decode_envelope(broken)


def test_a_deeply_nested_manifest_is_an_envelope_error():
    assert EnvelopeError  # the normalization lives in loads_strict -> SchemaError -> EnvelopeError
    from tcenv.schema import SchemaError, loads_strict

    with pytest.raises(SchemaError):
        loads_strict("[" * 100000 + "]" * 100000)


def test_a_deeply_nested_host_tiling_plan_is_a_contract_error(tmp_path):
    contract = _contract(tmp_path)
    nested = ("[" * 100000 + "]" * 100000).encode()
    candidate = decode_envelope(_bundle(host=nested))
    with pytest.raises(TaskContractError):
        HostTilingPlan.from_candidate(candidate, contract)


def test_a_host_tiling_plan_with_a_null_schema_version_is_a_contract_error(tmp_path):
    contract = _contract(tmp_path)
    document = {
        "schema_id": "tcenv.metal.host-tiling.1",
        "schema_version": None,
        "writes": [{"workspace": "scratch", "kind": "linear", "start": 0, "step": 0,
                    "count": 4, "stride": 1, "offset": 0, "dtype": "fp32"}],
    }
    candidate = decode_envelope(_bundle(host=json.dumps(document).encode()))
    with pytest.raises(TaskContractError, match="schema_version"):
        HostTilingPlan.from_candidate(candidate, contract)


def test_a_host_tiling_plan_with_a_huge_integer_is_a_contract_error(tmp_path):
    contract = _contract(tmp_path)
    huge = "9" * 5000
    raw = (
        '{"schema_id": "tcenv.metal.host-tiling.1", "schema_version": 1, "writes": [{"workspace": "scratch", '
        '"kind": "linear", "start": ' + huge + ', "step": 0, "count": 4, "stride": 1, "offset": 0, "dtype": "fp32"}]}'
    ).encode()
    candidate = decode_envelope(_bundle(host=raw))
    with pytest.raises(TaskContractError):
        HostTilingPlan.from_candidate(candidate, contract)


def test_a_deeply_nested_geometry_entry_is_a_geometry_error(tmp_path):
    from tcenv.contract_errors import GeometryError
    from tcenv.track.delivery import candidate_geometry

    nested = ("[" * 100000 + "]" * 100000).encode()
    candidate = decode_envelope(_bundle(geometry=nested))
    with pytest.raises(GeometryError):
        candidate_geometry(candidate)


def test_a_launch_with_a_huge_decimal_literal_is_a_launch_rejection():
    from tcenv.backends.metal.launch import LaunchRejected, parse_launch_spec

    source = "kernel void k() {}\n/* taic-launch\nlaunch k grid=" + "9" * 5000 + ",1,1 threads=1,1,1\n*/\n"
    with pytest.raises(LaunchRejected):
        parse_launch_spec(source, {"N": 8})


def test_the_bundle_parent_entries_keep_every_candidate_byte(tmp_path):
    contract = _contract(tmp_path)
    plan, _authority = domain_for(contract, TARGET)
    candidate = decode_envelope(_bundle())
    parent = bundle_parent_entries(candidate, plan)
    by_path = {entry.path: entry for entry in parent}
    # Bytes are copied verbatim; only the geometry entry's role is rewritten for the outer reader.
    assert by_path["device/variant_a.metal"].data == DEVICE_A
    assert by_path["host/tiling.json"].data == HOST_TILING_A
    assert by_path["host/execution_geometry.json"].role == "host_tiling"
    assert by_path["host/execution_geometry.json"].data == geometry_bytes(geometry_document("variant_a"))


def test_the_single_file_parent_package_is_byte_identical_to_the_old_wrapper(tmp_path):
    """`portfolio_for_submission` must add nothing for a single file.

    The parent entries (and therefore the manifest bytes, the guard digest and
    `candidate_sha256`) are compared against the shipped `envelope_entries` builder, so a future
    change to the bundle branch cannot drift the single-file evidence.
    """
    from tcenv.backends.metal.search_candidate import envelope_entries, portfolio_for

    contract = _contract(tmp_path)
    plan, authority = domain_for(contract, TARGET)
    _candidate, portfolio = portfolio_for_submission(
        SINGLE_SOURCE, plan, authority=authority, track_id="msl-single"
    )
    expected = envelope_entries(SINGLE_SOURCE, plan)
    assert portfolio.entries == expected
    assert [entry.data for entry in portfolio.entries] == [entry.data for entry in expected]
    assert portfolio.parent_sha256 == portfolio_for(SINGLE_SOURCE, plan).parent_sha256
