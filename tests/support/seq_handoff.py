"""P2-Q helpers: SEQ state-handoff cost algebra, the measured arms and the route decision.

Plan 2C asks one question: when a *causal* recurrent state has to cross a stage boundary, is the
existing protocol safe, and is the handoff write/read plus stage scheduling cost ever worth paying?

The arms below keep the device algorithm and the ABI fixed and only change **where the chain is
cut**:

* ``seq_three_piece`` — the shipped SEQ path: the candidate Host Program cuts a chain into up to
  three pieces and the state crosses two ``handoff_states`` boundaries between serial stages.
* ``single_piece_three_stage`` — the same device source, the same three dispatched stages, but the
  planner publishes one piece per chain.  The handoff traffic disappears; the dispatches stay.
  ``seq_three_piece - single_piece_three_stage`` is therefore the handoff (plus flag) cost.
* ``single_piece_single_stage`` — one piece and one dispatched stage.
  ``single_piece_three_stage - single_piece_single_stage`` is therefore the cost of the two extra
  stage dispatches and their buffer barriers.

``kda`` is the fixed-length task (no ``cu_seqlens``); ``varlen_kda`` is the L1 packed-varlen task
where the plan stage reads the *actual* ``cu_seqlens``.  Both are measured with the same device
source in every arm, so the difference is never an algorithm change.

This module is pure Python.  It reads registered OpSpec instances and builds candidate sources, so
the shared CPU test can assert the decision without importing a backend or Metal.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "tasks"
P2Q = ROOT / "tests" / "support" / "golden" / "shared" / "acceptance-epochs" / "fixtures" / "p2q"
EVIDENCE = P2Q / "evidence"
FIXTURES = ROOT / "tests" / "support" / "golden" / "metal"

KDA_DEVICE = FIXTURES / "kda" / "host_piece_handoff.metal"
KDA_PLANNER = ROOT / "tests" / "support" / "golden" / "shared" / "hostprogram" / "kda_fixed_planner.c"
VARLEN_DEVICE = FIXTURES / "varlen" / "piece_handoff.metal"

SCHEMA_MEASUREMENTS = "shapeforge.seq-handoff-measurements.1"
SCHEMA_PROBE = "shapeforge.seq-handoff-probe.2"
SCHEMA_DECISION = "shapeforge.seq-handoff-decision.2"
SCHEMA_SCRATCH = "shapeforge.seq-handoff-scratch-attempt.1"

#: Measured arms, in report order.
KDA_ARMS = ("seq_three_piece", "single_piece_three_stage", "single_piece_single_stage")
VARLEN_ARMS = ("pieces_three", "pieces_one")
KDA_SIZES = ("single", "small", "tail", "paper")
VARLEN_SIZES = ("mixed",)
#: Sizes that carry a real multi-piece split, so the handoff cost is observable at all.
MULTI_PIECE_SIZES = ("small", "tail", "paper")

MEASUREMENTS = EVIDENCE / "seq-handoff-measurements.json"
PROBE = EVIDENCE / "seq-handoff-probe.json"
SCRATCH_ATTEMPT = EVIDENCE / "seq-handoff-scratch-attempt.json"
DECISION = EVIDENCE / "decision.json"
CONFIRMATIONS = EVIDENCE / "confirmations.json"

#: ``tasks/kda`` declares exactly these cases.  It declares no ``ieee`` case, and this phase does
#: not add one: adding a case would move the task contract identity in the middle of an epoch.
#: ``tasks/varlen_kda`` declares the nine packed-varlen cases below.
KDA_CASES = ("normal", "wide_range", "outlier")
VARLEN_CASES = (
    "canonical",
    "permuted",
    "skewed",
    "all_unaligned_primes",
    "max_handoff_stress",
    "chunk_boundary",
    "zero_length",
    "single_sequence",
    "fresh_distribution",
)

#: The planner line that decides how many pieces a fixed-length chain is cut into.
KDA_PIECE_LINE = "    const uint32_t pieces = tokens < 3u ? tokens : 3u;"
#: The plan-stage line that decides how many pieces a packed-varlen chain is cut into.
VARLEN_PIECE_LINE = "    uint pieces = min(MAX_SPLIT, len);"

GUIDANCE = (
    "P2-Q SEQ state handoff: the candidate owns the chain cut, the handoff workspace is the task ABI"
)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def load_json(path: Path) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise AssertionError(f"{path} is not a JSON object")
    return document


def dump_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _replace_once(source: str, needle: str, replacement: str) -> str:
    if source.count(needle) != 1:
        raise AssertionError(f"the needle must appear exactly once: {needle!r}")
    return source.replace(needle, replacement)


# --- arm construction -----------------------------------------------------------------------


def kda_planner_source(arm: str) -> str:
    """The candidate planner for one arm.  Only the piece count changes."""
    source = KDA_PLANNER.read_text(encoding="utf-8")
    if arm == "seq_three_piece":
        return source
    if arm in ("single_piece_three_stage", "single_piece_single_stage"):
        return _replace_once(source, KDA_PIECE_LINE, "    const uint32_t pieces = 1u;")
    raise AssertionError(f"unknown kda arm {arm!r}")


def kda_stage_count(arm: str) -> int:
    if arm == "seq_three_piece" or arm == "single_piece_three_stage":
        return 3
    if arm == "single_piece_single_stage":
        return 1
    raise AssertionError(f"unknown kda arm {arm!r}")


def kda_geometry(arm: str, *, size: str = "small") -> dict:
    """Bundle geometry for one arm: the shipped document truncated to the arm's stage count.

    The stage bodies, the tensor list, the workspace list and the ``grid`` expression are the
    shipped ones.  Truncating stages is exactly what a candidate may do; it is not a harness
    shortcut.
    """
    from tests.support.kda_fixed_bundle import geometry_document

    document = json.loads(json.dumps(geometry_document()))
    # The device source is unchanged in every arm, so every stage keeps binding the declared
    # ``handoff_states`` workspace: the reflected parameter must resolve to a contract tensor.  In
    # the one-stage arm no stage writes or reads a byte of it (``zero`` is false, so the runtime
    # does not clear it either); it is an allocated binding, not a used handoff.
    document["stages"] = document["stages"][: kda_stage_count(arm)]
    del size
    return document


def kda_source(arm: str) -> str:
    """The candidate envelope for one arm (same device bytes in every arm)."""
    from tests.support.kda_fixed_bundle import bundle_source

    return bundle_source(planner_source=kda_planner_source(arm), geometry=kda_geometry(arm))


def varlen_source(arm: str) -> str:
    """The packed-varlen fixture with one or three pieces per chain, one file either way."""
    source = VARLEN_DEVICE.read_text(encoding="utf-8")
    if arm == "pieces_three":
        return source
    if arm == "pieces_one":
        return _replace_once(source, VARLEN_PIECE_LINE, "    uint pieces = 1u;")
    raise AssertionError(f"unknown varlen arm {arm!r}")


def scratch_sources() -> dict[str, str]:
    """Two attempts to move the handoff state from the OpSpec workspace onto candidate scratch.

    Both keep the device arithmetic and change only *where* the state lives: the pointer parameter
    is renamed to the scratch name and the stage binding moves from ``workspace`` to ``scratch``.

    * ``uninitialized_poison`` — the only policy that gets the per-iteration rewrite probes.
    * ``device_zero`` — a buffer, so it must be cleared inside the timing window.
    """
    from tests.support.kda_fixed_bundle import bundle_source

    device = KDA_DEVICE.read_text(encoding="utf-8")
    # Two reflected signatures carry the pointer (the shared body and the macro-generated kernel),
    # plus the body's two accesses.  The launch block keeps the OpSpec workspace name: it is the
    # executor cross-audit and must keep resolving against the task ABI.
    renamed = device.replace("device float* handoff_states", "device float* chain_state")
    renamed = renamed.replace("handoff_states[", "chain_state[")
    if renamed.count("chain_state") != 4:
        raise AssertionError("the handoff pointer is not renamed exactly where expected")
    out = {}
    for policy in ("uninitialized_poison", "device_zero"):
        document = json.loads(json.dumps(kda_geometry("seq_three_piece")))
        for stage in document["stages"]:
            stage["workspace"] = [item for item in stage["workspace"] if item["name"] != "handoff_states"]
            stage["scratch"] = [{"name": "chain_state", "access": "read_write"}]
        document["scratch"] = [
            {"name": "chain_state", "dtype": "fp32", "size_terms": ["2*H*D*D"], "init_policy": policy}
        ]
        out[policy] = bundle_source(device_source=renamed, geometry=document)
    return out


def candidate_digest(task: str, source: str) -> str:
    """The candidate digest the harness will register for this submission.

    A ``msl-bundle`` submission is an envelope, so its candidate digest is the envelope's canonical
    candidate digest and *not* the sha256 of the text.  An ``msl-single`` submission is one file and
    the two coincide.  Both helpers live in the backend-neutral ``tcenv.track`` layer, so a shared
    CPU test can recompute the digest without importing a backend.
    """
    from tcenv.track import candidate_from_single_source
    from tcenv.track.envelope import decode_envelope

    if task == "kda":
        return decode_envelope(source).candidate_sha256
    return candidate_from_single_source(source).candidate_sha256


def pingpong_uses() -> dict[str, list[str]]:
    """The per-stage scratch uses the ping-pong geometry actually declares (read off the document)."""
    uses: dict[str, list[str]] = {}
    for index, stage in enumerate(pingpong_geometry()["stages"]):
        uses[f"stage{index}"] = [f"{item['name']} {item['access']}" for item in stage["scratch"]]
    return uses


def pingpong_geometry() -> dict:
    """Geometry for the two-buffer ping-pong: each buffer's first use is a write.

    The total scratch bytes equal the declared ``handoff_states`` capacity (``2*H*D*D`` fp32): the
    ping-pong splits the same bytes into two buffers instead of indexing one, so there is **no**
    storage penalty and no traffic saving.  The extra cost is one more kernel argument and a
    different slot layout.
    """
    document = json.loads(json.dumps(kda_geometry("seq_three_piece")))
    uses = (
        (("state_a", "write"), ("state_b", "write")),
        (("state_a", "read"), ("state_b", "write")),
        (("state_b", "read"),),
    )
    for stage, spec in zip(document["stages"], uses):
        stage["workspace"] = [item for item in stage["workspace"] if item["name"] != "handoff_states"]
        stage["scratch"] = [{"name": name, "access": access} for name, access in spec]
    document["scratch"] = [
        {"name": name, "dtype": "fp32", "size_terms": ["H*D*D"], "init_policy": "uninitialized_poison"}
        for name in ("state_a", "state_b")
    ]
    return document


def pingpong_source() -> str:
    """A two-buffer ping-pong for the same recurrent state, both ``uninitialized_poison``.

    Each buffer's first use is a *write* (stage 0 writes ``state_a``; stage 1 reads it and writes
    ``state_b``; stage 2 reads ``state_b``), which is the one legal way to give a recurrent state a
    poison scratch.  It needs a 15th buffer argument and a different slot layout, so it is recorded
    as an attempt, never adopted.
    """
    from tests.support.kda_fixed_bundle import bundle_source

    device = KDA_DEVICE.read_text(encoding="utf-8")
    device = _replace_once(
        device,
        "                           device float* handoff_states [[buffer(13)]],            \\",
        "                           device float* state_a [[buffer(13)]],                   \\\n"
        "                           device float* state_b [[buffer(14)]],                   \\",
    )
    device = _replace_once(
        device,
        "                          output, final_state, work_items, item_counts,              \\\n"
        "                          handoff_states, head);                                     \\",
        "                          output, final_state, work_items, item_counts,              \\\n"
        "                          state_a, state_b, head);                                   \\",
    )
    device = _replace_once(
        device,
        "                           device float* handoff_states,\n",
        "                           device float* state_a,\n                           device float* state_b,\n",
    )
    device = _replace_once(
        device,
        "                : handoff_states[(STAGE - 1) * H * D * D + index];",
        "                : (STAGE == 1 ? state_a[index] : state_b[index]);",
    )
    device = _replace_once(
        device,
        "            } else {\n                handoff_states[STAGE * H * D * D + index] = state[vi * D + ki];",
        "            } else if (STAGE == 0) {\n                state_a[index] = state[vi * D + ki];\n"
        "            } else {\n                state_b[index] = state[vi * D + ki];",
    )
    return bundle_source(device_source=device, geometry=pingpong_geometry())


def source_digests() -> dict[str, str]:
    """Every measured source, by arm, so a replay can prove it evaluated the archived bytes."""
    digests = {f"kda:{arm}": sha256_bytes(kda_source(arm).encode("utf-8")) for arm in KDA_ARMS}
    digests.update({f"varlen:{arm}": sha256_bytes(varlen_source(arm).encode("utf-8")) for arm in VARLEN_ARMS})
    return digests


# --- state-size algebra ---------------------------------------------------------------------


def _instance(op: str, size: str):
    from tcenv.taskcontract import load_task_contract

    return load_task_contract(TASKS / op, dtype="fp16", size=size).instance


def _balanced_pieces(length: int, maximum: int = 3) -> int:
    """The shipped cut: ``min(maximum, length)`` non-empty pieces."""
    return max(1, min(int(maximum), int(length)))


def state_algebra() -> dict[str, dict]:
    """Handoff storage and traffic per registered instance, read off the declared ABI.

    ``handoff_states`` is an OpSpec workspace in both tasks, so its capacity is a contract fact,
    not a candidate choice.  ``written`` is what a *fully used* split actually touches.
    """
    out: dict[str, dict] = {}
    for op, sizes in (
        ("kda", ("single", "small", "tail", "paper")),
        ("varlen_kda", ("fixed", "mixed", "uniform", "capacity")),
    ):
        for size in sizes:
            instance = _instance(op, size)
            tensor = instance.tensor("handoff_states")
            dim = int(instance.symbols["D"])
            heads = int(instance.symbols["H"])
            state_bytes = dim * dim * 4
            if op == "kda":
                tokens = int(instance.symbols["N"])
                segments = [tokens] * heads
            else:
                declared = instance.canonical_seq_lens
                if declared is None and instance.workload_profiles:
                    declared = tuple(int(value) for value in instance.workload_profiles[0].seq_lens)
                segments = [length for length in (declared or ()) for _ in range(heads)]
            chains = len(segments)
            capacity_states = int(tensor.numel) // (dim * dim)
            pieces = [_balanced_pieces(length) for length in segments]
            boundaries = sum(one - 1 for one in pieces)
            out[f"{op}:{size}"] = {
                "task": op,
                "size": size,
                "tokens": sum(segments),
                "chains": chains,
                "heads": heads,
                "state_bytes_per_chain": state_bytes,
                "capacity_states": capacity_states,
                "capacity_bytes": int(tensor.nbytes),
                "declared_workspace": True,
                "declared_shape": list(tensor.shape),
                "pieces_per_chain": sorted(set(pieces)),
                "boundaries_used": boundaries,
                "bytes_written": boundaries * state_bytes,
                "bytes_written_plus_read": 2 * boundaries * state_bytes,
                "unused_states": capacity_states - boundaries,
            }
    return out


# --- evidence assertions --------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _measured(document: Mapping[str, Any], task: str, size: str, arm: str) -> dict:
    try:
        return document["measurements"][f"{task}:{size}"][arm]
    except KeyError as exc:  # pragma: no cover - only on a malformed archive
        raise AssertionError(f"{task}:{size}/{arm} is missing from the measurements") from exc


def assert_measurements(document: Mapping[str, Any]) -> None:
    """Every arm must be ``ok``, stable and archived with its own source digest."""
    _require(document["schema_id"] == SCHEMA_MEASUREMENTS, "wrong measurements schema")
    _require(document["publication_allowed"] is False, "measurements must not publish")
    _require(document["search_seed"] == 7, "the comparison must stay on the search seed")
    digests = source_digests()
    for key, digest in document["source_digests"].items():
        _require(digests[key] == digest, f"{key}: the measured source is not the archived source")
    for size in KDA_SIZES:
        for arm in KDA_ARMS:
            entry = _measured(document, "kda", size, arm)
            _require(entry["status"] == "ok", f"kda:{size}/{arm} is not ok: {entry['status']}")
            _require(entry["cases_ok"] == list(KDA_CASES), f"kda:{size}/{arm} did not run every declared case")
            _require(entry["repeats"] >= 3, f"kda:{size}/{arm} has fewer than three repeats")
            _require(entry["cv"] is not None and entry["cv"] < 0.06, f"kda:{size}/{arm} is too noisy")
    for size in VARLEN_SIZES:
        for arm in VARLEN_ARMS:
            entry = _measured(document, "varlen_kda", size, arm)
            _require(entry["status"] == "ok", f"varlen_kda:{size}/{arm} is not ok: {entry['status']}")
            _require(entry["cases_ok"] == list(VARLEN_CASES), f"varlen_kda:{size}/{arm} skipped a case")
            _require(entry["repeats"] >= 3, f"varlen_kda:{size}/{arm} has fewer than three repeats")
            _require(entry["cv"] is not None and entry["cv"] < 0.12, f"varlen_kda:{size}/{arm} is too noisy")


def assert_confirmations(document: Mapping[str, Any]) -> None:
    """Each measured arm passed a fresh confirmation with its own seed, never the search seed."""
    _require(document["schema_id"] == "shapeforge.seq-handoff-confirmations.1", "wrong confirmation schema")
    for key, subject in document["subjects"].items():
        _require(subject["status"] == "confirmed", f"{key}: fresh confirmation is {subject['status']}")
        _require(subject["identity"]["kind"] == "match", f"{key}: identity is {subject['identity']['kind']}")
        _require(int(subject["confirmation_seed"]) != 7, f"{key}: the fresh seed is the search seed")
        _require(subject.get("publication_allowed", False) is False, f"{key}: a confirmation must not publish")
    covered = set()
    for key in document["subjects"]:
        task, rest = key.split(":", 1)
        covered.add(f"{'varlen' if task == 'varlen_kda' else task}:{rest.split('/', 1)[1]}")
    expected = {f"kda:{arm}" for arm in KDA_ARMS} | {f"varlen:{arm}" for arm in VARLEN_ARMS}
    _require(
        covered == expected,
        f"fresh confirmations do not cover the measured arms: {sorted(expected ^ covered)}",
    )


def assert_scratch_attempt(document: Mapping[str, Any]) -> None:
    """The scratch route must be recorded as it actually behaves, including the legal counterexample."""
    _require(document["schema_id"] == SCHEMA_SCRATCH, "wrong scratch-attempt schema")
    _require(document["scratch_candidate_added"] is False, "a SEQ scratch candidate was added")
    poison = document["attempts"]["uninitialized_poison"]
    _require(poison["accepted"] is False, "a single read_write poison buffer was accepted as a state handoff")
    _require("first used as read_write" in poison["detail"], poison["detail"])
    # One buffer that both receives and hands on the state cannot be poison, but a two-buffer
    # ping-pong can: each buffer's first use is a write.  The record must not pretend otherwise.
    pingpong = document["attempts"]["two_buffer_pingpong"]
    _require(pingpong["accepted_by_lifecycle"] is True, "the two-buffer ping-pong is no longer accepted")
    _require(
        pingpong["uses"] == pingpong_uses(),
        f"the recorded ping-pong uses are not the geometry's own uses: {pingpong['uses']}",
    )
    _require(
        pingpong["scratch_bytes"] == pingpong["workspace_bytes_at_the_same_capacity"],
        "the ping-pong no longer moves the same total bytes as the declared workspace",
    )
    _require(
        pingpong["evaluation_status"] in ("crash", "ok"),
        f"unexpected ping-pong evaluation status {pingpong['evaluation_status']!r}",
    )
    if pingpong["evaluation_status"] == "crash":
        _require(
            "lifecycle was registered for a different resolved geometry" in pingpong["evaluation_message"],
            pingpong["evaluation_message"],
        )
    zero = document["attempts"]["device_zero"]
    _require(
        zero["outcome"] in ("no_benefit", "blocked"),
        f"device_zero scratch outcome {zero['outcome']!r} is not a measured non-benefit",
    )
    _require(
        document["reason"] == "no_measured_benefit_plus_bundle_scratch_blocker",
        f"unexpected scratch reason {document['reason']!r}",
    )


def assert_decision(document: Mapping[str, Any]) -> None:
    """The route decision must follow from the measurements, not from a preference."""
    _require(document["schema_id"] == SCHEMA_DECISION, "wrong decision schema")
    _require(document["adopted_route"] == "existing_abi_workspace_serial_stages", "the route moved")
    _require(document["scratch_spec_added"] is False, "decision adds a ScratchSpec")
    _require(document["publication_allowed"] is False, "decision must not publish")
    _require(document["npu_simulated_only"] is True, "NPU must stay simulation-only")
    for name, one in document["sizes"].items():
        _require(
            one["single_stage_us"] < one["seq_three_piece_us"],
            f"{name}: the SEQ path is not slower than the fused single-stage path",
        )
        if one["stages_differ_between_arms"]:
            _require(
                one["single_stage_us"] < one["single_piece_three_stage_us"],
                f"{name}: the stage floor is not positive",
            )
        _require(one["handoff_us"] > 0.0, f"{name}: the handoff cost is not positive")
        _require(one["handoff_fraction_of_single_stage"] > 0.0, f"{name}: the handoff fraction is not positive")
        if one["stages_differ_between_arms"]:
            _require(one["stage_floor_us"] > 0.0, f"{name}: the stage floor is not positive")
        else:
            _require(one["stage_floor_us"] == 0.0, f"{name}: the arms do not share a stage count")
        _require(
            one["seq_overhead_us"] == one["handoff_us"] + one["stage_floor_us"],
            f"{name}: the overhead decomposition does not close",
        )
    small = document["sizes"]["kda_small"]
    _require(small["handoff_fraction_of_single_stage"] > 0.05, "small: the handoff fraction is not material")
    paper = document["sizes"]["kda_paper"]
    _require(paper["handoff_fraction_of_single_stage"] < 0.02, "paper: the handoff is not a fixed cost")
    varlen = document["sizes"]["varlen_mixed"]
    _require(
        varlen["handoff_fraction_of_single_stage"] > 0.20,
        "varlen mixed: the packed handoff is not the dominant extra cost",
    )


def decision_document(measurements: Mapping[str, Any], probe: Mapping[str, Any], scratch: Mapping[str, Any]) -> dict:
    """Recompute the route decision from the archived measurements and the probe."""
    algebra = state_algebra()
    sizes: dict[str, dict] = {}
    for size in MULTI_PIECE_SIZES:
        seq = _measured(measurements, "kda", size, "seq_three_piece")
        three = _measured(measurements, "kda", size, "single_piece_three_stage")
        one = _measured(measurements, "kda", size, "single_piece_single_stage")
        handoff = seq["median_us"] - three["median_us"]
        floor = three["median_us"] - one["median_us"]
        sizes[f"kda_{size}"] = {
            "task": "kda",
            "size": size,
            "seq_three_piece_us": seq["median_us"],
            "single_piece_three_stage_us": three["median_us"],
            "single_stage_us": one["median_us"],
            "handoff_us": handoff,
            "stage_floor_us": floor,
            "seq_overhead_us": seq["median_us"] - one["median_us"],
            "handoff_fraction_of_single_stage": handoff / one["median_us"],
            "seq_overhead_fraction_of_single_stage": (seq["median_us"] - one["median_us"]) / one["median_us"],
            "state_bytes_per_chain": algebra[f"kda:{size}"]["state_bytes_per_chain"],
            "boundaries_used": algebra[f"kda:{size}"]["boundaries_used"],
            "stages_differ_between_arms": True,
        }
    varlen_three = _measured(measurements, "varlen_kda", "mixed", "pieces_three")
    varlen_one = _measured(measurements, "varlen_kda", "mixed", "pieces_one")
    sizes["varlen_mixed"] = {
        "task": "varlen_kda",
        "size": "mixed",
        "seq_three_piece_us": varlen_three["median_us"],
        "single_piece_three_stage_us": varlen_one["median_us"],
        "single_stage_us": varlen_one["median_us"],
        "handoff_us": varlen_three["median_us"] - varlen_one["median_us"],
        "stage_floor_us": 0.0,
        "seq_overhead_us": varlen_three["median_us"] - varlen_one["median_us"],
        "handoff_fraction_of_single_stage": (varlen_three["median_us"] - varlen_one["median_us"])
        / varlen_one["median_us"],
        "seq_overhead_fraction_of_single_stage": (varlen_three["median_us"] - varlen_one["median_us"])
        / varlen_one["median_us"],
        "state_bytes_per_chain": algebra["varlen_kda:mixed"]["state_bytes_per_chain"],
        "boundaries_used": algebra["varlen_kda:mixed"]["boundaries_used"],
        "stages_differ_between_arms": False,
        "note": (
            "the packed-varlen arms keep the same four dispatched stages (plan + 3 pieces), so the "
            "difference is the handoff and the flag protocol, not the stage floor"
        ),
    }
    return {
        "schema_id": SCHEMA_DECISION,
        "epoch": "p2-q-seq-state-handoff-1",
        "parent_head": "a9c1dccce636107c2852995578b05611bf19865b",
        "question": (
            "when does a causal state have to cross a stage boundary, and is the existing protocol safe "
            "and worth its handoff plus scheduling cost?"
        ),
        "adopted_route": "existing_abi_workspace_serial_stages",
        "route_rationale": [
            "the OpSpec workspace declares the handoff capacity in the task ABI, so the harness can reason "
            "about final_state and read_write state without trusting a candidate-owned size",
            "the correctness protocol already poisons that workspace on out1/out3 and zeroes it on out2, "
            "so a stale, partial or unpublished handoff changes the result between trials and is refused",
            "moving the same bytes to candidate scratch buys no dispatch and no traffic saving; the measured "
            "handoff cost is a property of the cut and of how many threads move the state, not of where the "
            "buffer is declared",
            "the only legal scratch shape is a two-buffer ping-pong: a single buffer that both receives and hands "
            "on the state is refused as uninitialized_poison (it is read_write), and the ping-pong moves exactly "
            "the same bytes as the declared workspace while adding one kernel argument and a different slot "
            "layout -- identity churn with no measured gain.  It is also unrunnable today: every "
            "bundle + Host Program + scratch candidate is blocked by the v2 worker geometry identity",
        ],
        "scratch_spec_added": False,
        "scratch_attempt": scratch,
        "implementation_changed": False,
        "sizes": sizes,
        "state_algebra": algebra,
        "probe_arm_count": len(probe["arms"]),
        "works_concurrency_gain_available": False,
        "concurrency_note": (
            "the permitted Metal synchronization model is a serial encoder plus a buffer barrier between "
            "stages, so cutting one chain into pieces adds stages and threadgroups but never runs two "
            "pieces of the same chain at the same time; a cross-threadgroup spin would be a deadlock risk "
            "and is not accepted as evidence"
        ),
        "publication_allowed": False,
        "npu_simulated_only": True,
    }


__all__ = [
    "CONFIRMATIONS",
    "DECISION",
    "EVIDENCE",
    "GUIDANCE",
    "KDA_ARMS",
    "KDA_CASES",
    "KDA_PIECE_LINE",
    "KDA_SIZES",
    "MEASUREMENTS",
    "MULTI_PIECE_SIZES",
    "P2Q",
    "PROBE",
    "SCRATCH_ATTEMPT",
    "VARLEN_ARMS",
    "VARLEN_CASES",
    "VARLEN_PIECE_LINE",
    "VARLEN_SIZES",
    "assert_confirmations",
    "assert_decision",
    "assert_measurements",
    "assert_scratch_attempt",
    "decision_document",
    "dump_json",
    "kda_geometry",
    "kda_planner_source",
    "kda_source",
    "kda_stage_count",
    "load_json",
    "pingpong_geometry",
    "pingpong_uses",
    "pingpong_source",
    "scratch_sources",
    "sha256",
    "sha256_bytes",
    "source_digests",
    "candidate_digest",
    "state_algebra",
    "varlen_source",
]
