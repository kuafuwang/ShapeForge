"""P2-Q: the SEQ route decision, the cost algebra and the untouched-mechanism checks (CPU only).

No backend and no GPU here: this file proves that the archived decision follows from the archived
measurements, that the arms differ only in where the chain is cut, and that this phase added no
mechanism (no ScratchSpec, no task edit, and the live prompt and pack closure stay on the current anchors).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from taic.trust import current_prompt_sha1
from tcenv.taskcontract.pack import code_import_closure

from ..support import seq_handoff as sh

ROOT = Path(__file__).resolve().parents[2]


def _require(condition: bool, message: str) -> None:
    assert condition, message


def test_the_measurements_prove_the_arms_are_apart_from_the_cut():
    """Every arm evaluates the same device bytes; only the Host Program's piece count changes."""
    shipped_planner = sh.KDA_PLANNER.read_text(encoding="utf-8")
    shipped_device = sh.KDA_DEVICE.read_text(encoding="utf-8")
    for arm in sh.KDA_ARMS:
        planner = sh.kda_planner_source(arm)
        changed = [
            (left, right)
            for left, right in zip(shipped_planner.splitlines(), planner.splitlines())
            if left != right
        ]
        if arm == "seq_three_piece":
            assert planner == shipped_planner
        else:
            assert changed == [(sh.KDA_PIECE_LINE, "    const uint32_t pieces = 1u;")], changed
            assert len(planner.splitlines()) == len(shipped_planner.splitlines())
        # The device source is never mutated by an arm: it is read straight from the fixture.
        assert sh.KDA_DEVICE.read_text(encoding="utf-8") == shipped_device

    shipped_varlen = sh.VARLEN_DEVICE.read_text(encoding="utf-8")
    assert sh.varlen_source("pieces_three") == shipped_varlen
    one = sh.varlen_source("pieces_one")
    expected = shipped_varlen.replace(sh.VARLEN_PIECE_LINE, "    uint pieces = 1u;")
    assert one == expected
    changed = [
        (left, right)
        for left, right in zip(shipped_varlen.splitlines(), one.splitlines())
        if left != right
    ]
    assert changed == [
        (
            "        uint pieces = min(MAX_SPLIT, len);",
            "        uint pieces = 1u;",
        )
    ], changed


def test_the_geometry_arms_are_prefixes_of_the_shipped_stage_list():
    """Truncating a candidate's own stage list is a candidate decision, not a harness shortcut."""
    from tests.support.kda_fixed_bundle import geometry_document

    shipped = geometry_document()["stages"]
    for arm in sh.KDA_ARMS:
        stages = sh.kda_geometry(arm)["stages"]
        assert stages == shipped[: len(stages)]
        assert len(stages) == sh.kda_stage_count(arm)
        # The measured device source keeps its reflected ``handoff_states`` parameter in every arm,
        # so the declared workspace stays bound even when no stage uses it.
        assert all(
            "handoff_states" in [item["name"] for item in stage["workspace"]] for stage in stages
        )


def test_the_state_algebra_is_read_off_the_declared_abi():
    """Handoff capacity comes from the OpSpec workspace, so the numbers are contract facts."""
    algebra = sh.state_algebra()
    assert algebra["kda:small"]["declared_shape"] == [2, 2, 8, 8]
    assert algebra["kda:small"]["capacity_bytes"] == 2 * 2 * 8 * 8 * 4
    assert algebra["kda:paper"]["capacity_bytes"] == 2 * 8 * 64 * 64 * 4
    assert algebra["kda:paper"]["state_bytes_per_chain"] == 64 * 64 * 4
    assert algebra["varlen_kda:mixed"]["declared_shape"] == [12, 8, 8]
    assert algebra["varlen_kda:mixed"]["capacity_bytes"] == 12 * 8 * 8 * 4
    # A single-token fixed chain has one piece, so *no* handoff slot is ever written: the sparse-use
    # fact the scratch route cannot satisfy with an uninitialized_poison buffer.
    assert algebra["kda:single"]["boundaries_used"] == 0
    assert algebra["kda:single"]["unused_states"] == algebra["kda:single"]["capacity_states"]
    # A short chain inside a packed batch leaves a slot unused even when the batch is full.
    assert algebra["varlen_kda:mixed"]["unused_states"] == 2
    for one in algebra.values():
        assert one["declared_workspace"] is True
        assert one["bytes_written"] == one["boundaries_used"] * one["state_bytes_per_chain"]


def test_the_archived_decision_follows_from_the_archived_measurements():
    measurements = sh.load_json(sh.MEASUREMENTS)
    probe = sh.load_json(sh.PROBE)
    scratch = sh.load_json(sh.SCRATCH_ATTEMPT)
    decision = sh.load_json(sh.DECISION)
    sh.assert_measurements(measurements)
    sh.assert_scratch_attempt(scratch)
    sh.assert_decision(decision)
    assert decision == sh.decision_document(measurements, probe, scratch)
    assert decision["adopted_route"] == "existing_abi_workspace_serial_stages"
    assert decision["scratch_spec_added"] is False
    assert decision["implementation_changed"] is False
    assert decision["works_concurrency_gain_available"] is False
    assert "cross-threadgroup spin" in decision["concurrency_note"]
    sizes = decision["sizes"]
    # Fixed length: the cut is pure overhead, and it is a fixed cost rather than a share.
    assert sizes["kda_small"]["seq_overhead_fraction_of_single_stage"] > 0.15
    assert sizes["kda_tail"]["seq_overhead_fraction_of_single_stage"] > 0.10
    assert sizes["kda_paper"]["seq_overhead_fraction_of_single_stage"] < 0.01
    # Packed varlen: the same four stages, so the whole difference is the handoff and the flags.
    assert sizes["varlen_mixed"]["seq_overhead_fraction_of_single_stage"] > 0.40


def test_the_archived_confirmations_are_fresh_and_bound_to_the_measured_sources():
    subjects = sh.load_json(sh.CONFIRMATIONS)
    sh.assert_confirmations(subjects)
    digests = sh.source_digests()
    for key, subject in subjects["subjects"].items():
        arm, arm_value = key.split("/")
        task, size = arm.split(":")
        source = sh.kda_source(arm_value) if task == "kda" else sh.varlen_source(arm_value)
        expected = digests[f"{'varlen' if task == 'varlen_kda' else task}:{arm_value}"]
        assert sh.sha256_bytes(source.encode("utf-8")) == expected, key
        assert subject["candidate_sha256"] == sh.candidate_digest(task, source), key
        assert int(subject["confirmation_seed"]) != 7, key


def test_the_scratch_route_is_recorded_as_it_behaves_including_the_legal_counterexample():
    """The phase first claimed poison could not hold a recurrent state.  That was wrong; pin both.

    A single buffer is refused, but a two-buffer ping-pong is accepted by the lifecycle, so the
    recorded evidence must show both and must not claim the protocol forbids the route.
    """
    scratch = sh.load_json(sh.SCRATCH_ATTEMPT)
    sh.assert_scratch_attempt(scratch)
    poison = scratch["attempts"]["uninitialized_poison"]
    assert poison["accepted"] is False
    assert poison["stage"] == "geometry"
    assert "write-only" in poison["detail"]
    pingpong = scratch["attempts"]["two_buffer_pingpong"]
    assert pingpong["accepted_by_lifecycle"] is True
    assert pingpong["init_policy"] == "uninitialized_poison"
    # The recorded uses must be the geometry's own uses, and the ping-pong must not claim a storage
    # penalty: two buffers of H*D*D fp32 equal the declared 2*H*D*D capacity.
    assert pingpong["uses"] == sh.pingpong_uses()
    assert pingpong["uses"]["stage0"] == ["state_a write", "state_b write"]
    assert pingpong["scratch_bytes"] == pingpong["workspace_bytes_at_the_same_capacity"]
    assert pingpong["storage_penalty"] is False
    assert scratch["scratch_candidate_added"] is False
    assert scratch["attempts"]["device_zero"]["outcome"] in ("no_benefit", "blocked")
    # The adopted route must not be justified by the retracted claim.
    decision = sh.load_json(sh.DECISION)
    rationale = " ".join(decision["route_rationale"])
    assert "inherently" not in rationale
    assert "read_write" in rationale
    assert "doubles" not in rationale and "double the storage" not in rationale


def test_the_probe_arms_are_measurements_and_not_candidates():
    probe = sh.load_json(sh.PROBE)
    assert probe["schema_id"] == sh.SCHEMA_PROBE
    assert probe["not_a_candidate"] is True
    assert probe["publication_allowed"] is False
    assert probe["device_name"]
    assert probe["iters"] >= 100
    assert len(probe["arms"]) >= 12
    # The handoff copy is width-bound: the same bytes at a realistic thread count are not a cost.
    for dim in (8, 64):
        narrow = probe["handoff"][f"handoff_state_fp32_D{dim}"]
        wide = probe["handoff"][f"handoff_state_fp32_D{dim}_wide"]
        assert narrow["state_bytes"] == wide["state_bytes"]
        assert narrow["threads"] < wide["threads"]
    assert probe["handoff"]["handoff_state_fp32_D64"]["handoff_over_control_us"] > 10.0
    assert probe["handoff"]["handoff_state_fp32_D64_wide"]["handoff_over_control_us"] < 5.0
    # The stage floor across the shipped fixture's own pipeline is positive at D=8.
    small = probe["kda_stage_cost"]["small"]
    assert small["pipeline_stage_floor_us"] > 0.0
    assert small["pipeline_ratio"] > 1.0
    # The synthetic empty-stage slope is not resolvable at this size; the probe says so instead of
    # inventing a floor, and the stage cost comes from the same-work one-vs-three-stage arms.
    assert abs(probe["stage_floor"]["per_stage_wall_us"]) < 5.0
    assert "encoder" in probe["stage_floor"]["note"]


def test_the_decision_does_not_publish_and_keeps_the_npu_simulated():
    decision = sh.load_json(sh.DECISION)
    assert decision["publication_allowed"] is False
    assert decision["npu_simulated_only"] is True
    measurements = sh.load_json(sh.MEASUREMENTS)
    assert measurements["publication_allowed"] is False
    assert measurements["npu_simulated_only"] is True


def test_no_mechanism_was_added_for_this_phase():
    """The scratch field set, the pack closure and the prompt/trust anchors are unchanged."""
    from tcenv.scratch import INIT_POLICIES, ScratchSpec

    assert INIT_POLICIES == {"uninitialized_poison", "device_zero", "host_written"}
    assert set(ScratchSpec.__dataclass_fields__) == {"name", "dtype", "size_terms", "init_policy"}
    files = code_import_closure(ROOT)
    digest = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in files)).hexdigest()
    from tests.support.live_anchors import load_live_anchors
    anchors = load_live_anchors(ROOT)
    assert list(files) == anchors["pack_closure"]["files"]
    assert len(files) == 78
    assert digest == anchors["pack_closure"]["sha256"]
    assert current_prompt_sha1() == anchors["prompt_sha1"]


def test_the_operator_has_no_untracked_task_edit_behind_this_phase():
    """``tasks/kda`` and ``tasks/varlen_kda`` are read, never written, by this phase."""
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--", "tasks/kda", "tasks/varlen_kda"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert completed.stdout.strip() == "", completed.stdout
