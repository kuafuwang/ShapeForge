"""P2-Q: the existing SEQ state-handoff path on real Metal, and every way it must refuse.

This file answers the safety half of Plan 2C.  It does **not** add a mechanism: the same
``host_piece_handoff.metal`` / ``kda_fixed_planner.c`` / ``piece_handoff.metal`` bytes that ship in
the repository run here, and the only things that change between arms are where the candidate cuts
the chain (the Host Program's piece count) and how many stages its own geometry dispatches.

Arms (see ``tests/support/seq_handoff.py``):

* ``seq_three_piece`` — the shipped SEQ path, up to three pieces over three serial stages.
* ``single_piece_three_stage`` — one piece, the same three dispatched stages: the handoff is gone,
  the dispatches are not.
* ``single_piece_single_stage`` — one piece, one dispatched stage.

The refusal cases cover the state source, the state store, the stage order, the work-list span and
the result writeback, in the order the harness checks them.  Nothing here spins on a flag from
another threadgroup: ordering is the serial encoder plus a buffer barrier.
"""

from __future__ import annotations

import pytest

from taic.task_binding import bind_task
from tcenv.backends.metal.search_candidate import evaluate_candidate, preflight_submission
from tcenv.target import load_target

from ...support import seq_handoff as sh
from ...support.kda_fixed_bundle import DEVICE_SOURCE, bundle_source, geometry_document

pytestmark = pytest.mark.gpu

TARGET = load_target("apple-m4-max")
#: Timing repeats for the replay node; the archive used the production default of 15.
PERF_REPS = 15


def _why(report: dict) -> str:
    """One readable line from a schema-4 report, never raising on a missing key."""
    parts = [f"status={report.get('status')!r}"]
    if report.get("stage"):
        parts.append(f"stage={report['stage']!r}")
    if report.get("message"):
        parts.append(str(report["message"]))
    errors = (report.get("compile") or {}).get("errors")
    if errors:
        parts.append(f"compile: {errors[:600]}")
    violations = (report.get("safety") or {}).get("violations") or []
    if violations:
        parts.append("safety: " + "; ".join(str(item) for item in violations[:3]))
    return " | ".join(parts)


def _evaluate(source: str, *, task: str = "kda", size: str = "small", cases: tuple[str, ...] | None = None):
    track = "msl-bundle" if task == "kda" else "msl-single"
    return evaluate_candidate(
        bind_task(task, "fp16", size).locator,
        source,
        target=TARGET,
        mode="quick",
        reps=3,
        cases=cases,
        track_id=track,
    )


def _refused(run) -> dict:
    report = run.result.report
    assert report["status"] in ("incorrect", "unsafe"), _why(report)
    return report


def _replace(source: str, needle: str, replacement: str) -> str:
    assert source.count(needle) == 1, f"the mutation anchor is not unique: {needle!r}"
    return source.replace(needle, replacement)


# --- the measured arms ----------------------------------------------------------------------


@pytest.mark.parametrize("size", sh.KDA_SIZES)
@pytest.mark.parametrize("arm", sh.KDA_ARMS)
def test_fixed_kda_seq_arm_passes_every_declared_case(arm: str, size: str):
    """Every arm is correct and safe on all three declared ``tasks/kda`` cases.

    ``tasks/kda`` declares ``normal``, ``wide_range`` and ``outlier`` — it declares **no** ``ieee``
    case, so "every declared case" is those three.  Adding one would move the task contract, which
    this phase does not do; the harness still runs its three initial-value trials (out1/out3
    poisoned, out2 zeroed) for each case, so a result that depends on stale state is refused.
    """
    run = _evaluate(sh.kda_source(arm), size=size)
    report = run.result.report
    assert report["status"] == "ok", _why(report)
    assert [case["case"] for case in report["correctness"]["cases"]] == list(sh.KDA_CASES)
    assert all(case["ok"] for case in report["correctness"]["cases"])
    assert report["safety"]["ok"] is True
    assert (report["safety"].get("nondeterminism") or []) == []
    # Dispatch is the candidate's own geometry, not a harness rule.
    assert [item["kernel"] for item in run.evaluation["dispatch"]] == [
        f"kda_piece{index}" for index in range(sh.kda_stage_count(arm))
    ]


@pytest.mark.parametrize("arm", sh.VARLEN_ARMS)
def test_packed_varlen_arm_passes_every_declared_case(arm: str):
    """Both cuts of the packed-varlen chain pass all nine declared L1 cases."""
    run = _evaluate(sh.varlen_source(arm), task="varlen_kda", size="mixed")
    report = run.result.report
    assert report["status"] == "ok", _why(report)
    assert {case["case"] for case in report["correctness"]["cases"]} == set(sh.VARLEN_CASES)
    assert all(case["ok"] for case in report["correctness"]["cases"])
    assert report["safety"]["ok"] is True
    assert [item["kernel"] for item in run.evaluation["dispatch"]] == [
        "varlen_plan",
        "varlen_piece0",
        "varlen_piece1",
        "varlen_piece2",
    ]


def test_the_handoff_workspace_capacity_is_the_declared_abi_capacity():
    """The allocated bytes are the OpSpec formula, and the one-piece arm allocates but never uses them."""
    three = _evaluate(sh.kda_source("seq_three_piece"), size="small")
    one = _evaluate(sh.kda_source("single_piece_single_stage"), size="small")
    algebra = sh.state_algebra()["kda:small"]
    for run in (three, one):
        assert run.evaluation["workspace_bytes"]["handoff_states"] == algebra["capacity_bytes"]
        assert run.evaluation["workspace_zero"]["handoff_states"] is False
    # The one-stage arm keeps the reflected binding (the device source is unchanged) but no stage
    # reads or writes a handoff byte, so the timing cannot include a handoff store.
    assert one.evaluation["workspace_bytes"]["handoff_states"] > 0


# --- refused: the state source ---------------------------------------------------------------


def test_a_continuation_that_starts_from_the_initial_state_is_refused():
    """Piece 1 and 2 must continue from the handoff, never restart from ``initial_state``."""
    source = _replace(
        DEVICE_SOURCE,
        "                : handoff_states[(STAGE - 1) * H * D * D + index];",
        "                : initial_state[index];",
    )
    report = _refused(_evaluate(bundle_source(device_source=source), size="small"))
    assert report["status"] == "incorrect", _why(report)


def test_a_continuation_that_starts_from_zero_is_refused():
    """A zeroed continuation is the classic "looks fast, is wrong" SEQ kernel."""
    source = _replace(
        DEVICE_SOURCE,
        "                : handoff_states[(STAGE - 1) * H * D * D + index];",
        "                : 0.0f;",
    )
    report = _refused(_evaluate(bundle_source(device_source=source), size="small"))
    assert report["status"] == "incorrect", _why(report)


# --- refused: the state store ----------------------------------------------------------------


def test_a_partial_handoff_write_is_refused():
    """Writing all but the last state row must not pass: the consumer reads the poisoned row.

    Correctness poisons the workspace on the first and third trial and zeroes it on the second, so a
    partially written state changes the result between trials instead of hiding behind one initial
    value.
    """
    source = _replace(
        DEVICE_SOURCE,
        "            } else {\n                handoff_states[STAGE * H * D * D + index] = state[vi * D + ki];",
        "            } else {\n                if (vi + 1u < D) { handoff_states[STAGE * H * D * D + index] = state[vi * D + ki]; }",
    )
    report = _refused(_evaluate(bundle_source(device_source=source), size="small"))
    assert report["status"] in ("unsafe", "incorrect"), _why(report)


#: A previous round's terminal state, written into the slot a *later* piece will consume.  The value
#: is produced by the fixture itself (``initial_state + offset``), so the experiment does not depend
#: on the harness's poison/zero policy for the workspace.  It is a *prefix* block: it is inserted
#: before the terminal-state store loop and closes only itself, so the surrounding braces stay
#: balanced (an unbalanced insertion fails to compile and the harness reports an identity error
#: instead of a correctness verdict).
_STALE_SEED = """    if (STAGE == 0u && published == 3) {
        // Negative-fixture only: seed the slot that piece 2 consumes with "the previous round's"
        // terminal state.  A normal piece 1 overwrites it; a piece 1 that fails to publish does not.
        for (uint vi = 0; vi < D; ++vi) {
            for (uint ki = 0; ki < D; ++ki) {
                handoff_states[1 * H * D * D + head * D * D + vi * D + ki] =
                    initial_state[head * D * D + vi * D + ki] + OFFSET;
            }
        }
    }
"""
#: The terminal-state store loop, so this negative can seed before it or skip it for one stage.
#: The trailing ``if (int(STAGE) + 1 == published) {`` is what makes the anchor unique: the state
#: *prologue* has the same two loops and the same ``index`` line.
_SKIP_STORE_ANCHOR = (
    "    for (uint vi = 0; vi < D; ++vi) {\n"
    "        for (uint ki = 0; ki < D; ++ki) {\n"
    "            const uint index = head * D * D + vi * D + ki;\n"
    "            if (int(STAGE) + 1 == published) {"
)


def _seeded(offset: str) -> str:
    """The fixture plus a "previous round" seed in slot 1 (the only entry point for this negative)."""
    block = _STALE_SEED.replace("OFFSET", f"({offset})")
    return _replace(DEVICE_SOURCE, _SKIP_STORE_ANCHOR, block + _SKIP_STORE_ANCHOR)


def _without_piece1_publish(source: str) -> str:
    """Skip the whole store block for stage 1: the piece never publishes its terminal state."""
    return _replace(source, _SKIP_STORE_ANCHOR, "    if (STAGE == 1u) { return; }\n" + _SKIP_STORE_ANCHOR)


def test_a_stale_handoff_state_is_refused():
    """A consumer that reads the previous round's terminal state must be refused, with controls.

    A three-piece chain hands state through slot 0 and then slot 1.  Piece 0 seeds slot 1 with a
    fabricated "previous round" state; a normal piece 1 overwrites it, so the seeded control must be
    ``ok``.  When piece 1 does not publish at all, piece 2 consumes the seeded bytes: the refusal is
    a finite, moderate numeric error, and changing the seed's offset changes that error, which is
    what proves piece 2 really read those bytes.

    A *cross-round* stale read is structurally impossible here: correctness poisons the workspace on
    the first and third trial and zeroes it on the second, so any candidate that reuses last round's
    state without rewriting it fails on the trial protocol, not only on this fixture.
    """
    control = _evaluate(bundle_source(device_source=_seeded("0.5f")), size="small")
    assert control.result.report["status"] == "ok", _why(control.result.report)

    negative = _without_piece1_publish(_seeded("0.5f"))
    report = _refused(_evaluate(bundle_source(device_source=negative), size="small"))
    assert report["status"] in ("incorrect", "unsafe"), _why(report)
    first = (report.get("correctness") or {}).get("cases")[0]
    assert first["ok"] is False, first
    stale_err = first["scaled_err"]
    assert stale_err != "inf", first["message"]
    assert float(stale_err) > 1.0, first["message"]

    shifted = _without_piece1_publish(_seeded("0.25f"))
    shifted_report = _refused(_evaluate(bundle_source(device_source=shifted), size="small"))
    shifted_first = (shifted_report.get("correctness") or {}).get("cases")[0]
    assert shifted_first["ok"] is False, shifted_first
    assert shifted_first["scaled_err"] != "inf", shifted_first["message"]
    assert float(shifted_first["scaled_err"]) != float(stale_err), (stale_err, shifted_first["scaled_err"])


# --- refused: the stage order and the stage count ---------------------------------------------


def test_a_consumer_dispatched_before_its_producer_is_refused():
    """Swapping ``kda_piece0`` and ``kda_piece1`` in the candidate geometry must be refused."""
    document = geometry_document()
    stages = document["stages"]
    document["stages"] = [stages[1], stages[0], stages[2]]
    report = _refused(_evaluate(bundle_source(geometry=document), size="small"))
    assert report["status"] in ("incorrect", "unsafe"), _why(report)


def test_a_dropped_stage_while_pieces_remain_is_refused():
    """Two stages cannot carry a three-piece cut: ``final_state`` is never written."""
    report = _refused(_evaluate(_two_stage_cut(), size="small"))
    assert report["status"] == "unsafe", _why(report)


def _two_stage_cut() -> str:
    """The shipped three-piece planner with only two dispatched stages."""
    from tests.support.kda_fixed_bundle import PLANNER_SOURCE

    document = geometry_document()
    document["stages"] = document["stages"][:2]
    return bundle_source(planner_source=PLANNER_SOURCE, geometry=document)


def test_publishing_more_pieces_than_the_dispatched_stages_is_refused():
    """Publishing ``item_counts = 4`` while only three stages exist must be refused.

    The fixture's own ``published > 3`` guard makes every stage return, so nothing writes
    ``final_state`` and the trial's initial write contents reach the comparison: the harness reports
    ``unsafe``.  The point of the node is that an inconsistent publish count cannot pass silently,
    not which of the two verdicts it earns.
    """
    from tests.support.kda_fixed_bundle import PLANNER_SOURCE

    planner = _replace(
        PLANNER_SOURCE,
        "    workspaces->item_counts[0] = (int32_t)pieces;",
        "    workspaces->item_counts[0] = 4;",
    )
    report = _refused(_evaluate(bundle_source(planner_source=planner), size="small"))
    assert report["status"] == "unsafe", _why(report)
    assert "initial write-output contents" in _why(report), _why(report)


# --- refused: the work list span and the result writeback ------------------------------------


def test_a_work_list_write_beyond_the_declared_capacity_is_refused():
    """A planner that publishes four pieces writes past the declared six-word span."""
    from tests.support.kda_fixed_bundle import PLANNER_SOURCE

    planner = _replace(
        PLANNER_SOURCE,
        "    const uint32_t pieces = tokens < 3u ? tokens : 3u;",
        "    const uint32_t pieces = 4u;",
    )
    run = _evaluate(bundle_source(planner_source=planner), size="small")
    report = run.result.report
    assert report["status"] != "ok", _why(report)
    detail = " ".join(_why(report).split())
    assert any(word in detail for word in ("work_items", "capacity", "span", "guard sentinel")), detail
    assert "host program failed" in detail or "guard" in detail, detail


def test_a_partial_final_state_write_is_unsafe():
    """Writing all but the last ``final_state`` row depends on the trial's initial write contents.

    The guard sits *inside* the final branch.  Putting ``vi + 1u < D`` in the branch condition
    instead would also send the last row into the handoff store, i.e. it would test an out-of-bounds
    handoff write rather than a partial result write.
    """
    source = _replace(
        DEVICE_SOURCE,
        "            if (int(STAGE) + 1 == published) {\n                final_state[index] = state[vi * D + ki];",
        "            if (int(STAGE) + 1 == published) {\n                if (vi + 1u < D) { final_state[index] = state[vi * D + ki]; }",
    )
    report = _refused(_evaluate(bundle_source(device_source=source), size="small"))
    assert report["status"] == "unsafe", _why(report)
    assert "initial write-output contents" in _why(report), _why(report)


# --- the scratch route -----------------------------------------------------------------------


def test_a_single_read_write_buffer_cannot_be_a_poison_handoff_state():
    """One scratch buffer that both receives and hands on the state is refused, before any compile.

    ``uninitialized_poison`` requires the first active stage to bind the scratch ``write``-only, so a
    later stage cannot read an unwritten value.  A *single* buffer used by every stage is
    ``read_write`` by construction and is refused at geometry resolution.
    """
    outcome = preflight_submission(
        bind_task("kda", "fp16", "small").contract,
        sh.scratch_sources()["uninitialized_poison"],
        target=TARGET,
        track_id="msl-bundle",
    )
    assert not outcome.ok
    assert outcome.stage == "geometry"
    assert "first used as read_write" in outcome.rejection
    assert "write-only" in outcome.rejection


def test_a_two_buffer_pingpong_satisfies_the_scratch_lifecycle():
    """The protocol *can* express a recurrent state: a second buffer makes each first use a write.

    This node exists because an earlier revision of this phase claimed the opposite.  The ping-pong
    is accepted by the lifecycle and then hits the v2 worker's geometry identity check, which drops
    ``scratch`` when it rebuilds the lifecycle geometry.  Either outcome is recorded: the point of the
    node is that the *lifecycle* is not what forbids the scratch route.
    """
    recorded = sh.load_json(sh.SCRATCH_ATTEMPT)["attempts"]["two_buffer_pingpong"]
    outcome = preflight_submission(
        bind_task("kda", "fp16", "small").contract,
        sh.pingpong_source(),
        target=TARGET,
        track_id="msl-bundle",
    )
    assert outcome.ok, outcome.rejection
    assert recorded["accepted_by_lifecycle"] is True
    run = evaluate_candidate(
        bind_task("kda", "fp16", "small").locator,
        sh.pingpong_source(),
        target=TARGET,
        mode="quick",
        reps=3,
        track_id="msl-bundle",
    )
    report = run.result.report
    if report["status"] == "ok":
        # A later phase fixed the worker: the ping-pong now runs end to end.
        assert all(case["ok"] for case in report["correctness"]["cases"]), _why(report)
    else:
        assert report["status"] == "abi_error", _why(report)
        assert any("reflects readwrite" in violation for violation in report["abi"]["violations"]), report["abi"]


# --- replay of the archived comparison -------------------------------------------------------


def test_the_archived_ordering_replays_on_this_machine():
    """Re-run the ``small`` arms and require the archived identity *and* the archived ordering.

    The archived medians are microseconds on one Mac; comparing them value-for-value would fail on
    any load change.  What must hold is the identity (source bytes, dispatch, workspace bytes, cases)
    plus the direction of the comparison: cutting the chain costs more than not cutting it, and one
    stage costs less than three.
    """
    measurements = sh.load_json(sh.MEASUREMENTS)
    observed: dict[str, float] = {}
    identities: dict[str, tuple] = {}
    for arm in sh.KDA_ARMS:
        source = sh.kda_source(arm)
        assert sh.sha256_bytes(source.encode("utf-8")) == measurements["measurements"]["kda:small"][arm]["source_sha256"]
        run = evaluate_candidate(
            bind_task("kda", "fp16", "small").locator,
            source,
            target=TARGET,
            mode="full",
            seed=7,
            reps=PERF_REPS,
            track_id="msl-bundle",
        )
        report = run.result.report
        assert report["status"] == "ok", _why(report)
        observed[arm] = float(report["candidate_us"])
        identities[arm] = (
            [item["kernel"] for item in run.evaluation["dispatch"]],
            dict(run.evaluation["workspace_bytes"]),
            [case["case"] for case in report["correctness"]["cases"]],
        )
    assert identities["seq_three_piece"][0] == ["kda_piece0", "kda_piece1", "kda_piece2"]
    assert identities["single_piece_three_stage"][0] == ["kda_piece0", "kda_piece1", "kda_piece2"]
    assert identities["single_piece_single_stage"][0] == ["kda_piece0"]
    # The one-stage arm keeps the same declared handoff capacity: only the use disappears.
    assert identities["seq_three_piece"][1] == identities["single_piece_single_stage"][1]
    assert identities["seq_three_piece"][2] == list(sh.KDA_CASES)

    assert observed["seq_three_piece"] > observed["single_piece_three_stage"] * 1.01, observed
    assert observed["single_piece_three_stage"] > observed["single_piece_single_stage"] * 1.01, observed
