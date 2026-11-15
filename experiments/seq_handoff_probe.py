"""Re-runnable, non-publishable cost probe for SEQ state handoff (Plan 2C / P2-Q).

Plan 2C asks what the *fixed* costs of the existing SEQ delivery actually are, so a decision about
candidate scratch can rest on measured microseconds instead of an argument:

* ``stage_floor_n{1..4}`` — an empty pipeline of one to four stages, dispatched exactly the way the
  harness dispatches a multi-stage candidate (serial encoder, buffer barrier between stages, the
  same rotating buffer sets).  The slope is the per-stage floor: one more dispatch plus one more
  barrier.
* ``handoff_control`` / ``handoff_state_fp32_D8`` / ``handoff_state_fp32_D64`` — a two-stage
  producer/consumer over the *same* ``H*D*D`` fp32 state bytes the KDA handoff moves.  The control
  writes and reads one word, so the difference is the state write plus read, not the stage floor.
* ``kda_full_1_piece`` / ``kda_half_piece_stage1`` / ``kda_half_piece_stage2`` — the shipped
  ``host_piece_handoff.metal`` compiled as-is, one stage at a time, so the per-stage device cost of a
  full chain versus a piece is visible.  This uses the fixture's own bytes; only the injected
  ``N``/``H``/``D`` constants and the work list are supplied here.

Nothing in this file is a candidate, a speedup or a publication.  The kernels are the shipped
fixture or deliberately incomplete probes; ``not_a_candidate=true``.
"""
from __future__ import annotations

import argparse
import json
import statistics
import struct
from pathlib import Path

from tcenv.backends.metal.runtime import BufferArg, get_context
from tcenv.gpulock import gpu_lock

ROOT = Path(__file__).resolve().parents[1]
KDA_DEVICE = ROOT / "tests" / "support" / "golden" / "metal" / "kda" / "host_piece_handoff.metal"
OUT = (
    ROOT / "tests" / "support" / "golden" / "shared" / "acceptance-epochs" / "fixtures" / "p2q" / "evidence"
    / "seq-handoff-probe.json"
)
SCHEMA_ID = "shapeforge.seq-handoff-probe.2"
TARGET_ID = "apple-m4-max"
ITERS = 100
#: The empty-stage floor is a few microseconds per dispatch.  A *longer* buffer is worse here:
#: the command buffer becomes encoder-bound and the GPU timestamps stop reflecting the stage cost.
FLOOR_ITERS = 100
FLOOR_REPEATS = 9
NSETS = 16
REPEATS = 5
#: The two state widths the registered tasks actually use: ``tasks/kda`` small/tail (D=8) and
#: paper (D=64), plus ``tasks/varlen_kda`` L1 (D=8).
STATE_SHAPES = {8: 2, 64: 8}

_EMPTY_KERNEL = """
kernel void probe_empty(device float* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    if (gid == 0xFFFFFFFFu) { out[0] = 1.0f; }
}
"""


def _handoff_source(per_thread: int, threads: int) -> str:
    """Two-stage state round trip: producer writes ``per_thread * threads`` floats, consumer reads.

    ``per_thread=1, threads=1`` is the control: one word written and read, so the arm measures the
    stage floor only.  Every arm moves the *same total bytes*; only the thread count changes, which
    separates the latency-bound copy the fixture actually performs from the bandwidth of the same
    bytes at a realistic width.
    """
    return f"""
constant constexpr uint PT = {per_thread}u;
constant constexpr uint THREADS = {threads}u;

kernel void probe_producer(device float* state [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {{
    for (uint i = 0u; i < PT; i++) {{
        state[gid * PT + i] = float(i) * 0.5f + 1.0f;
    }}
    out[gid] = state[gid * PT];
}}

kernel void probe_consumer(device float* state [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {{
    float acc = 0.0f;
    for (uint i = 0u; i < PT; i++) {{
        acc += state[gid * PT + i];
    }}
    out[THREADS + gid] = acc;
}}
"""


def _kda_standalone_source(device: str, symbols: dict[str, int], lower_bound: int) -> str:
    """The shipped fixture plus the integers the harness assembler injects."""
    preamble = (
        "// Probe-only preamble: the harness assembler injects the same integers from the task\n"
        "// instance.  The kernel bodies below are the shipped fixture's own bytes.\n"
        f"constant constexpr uint N = {symbols['N']}u;\n"
        f"constant constexpr uint H = {symbols['H']}u;\n"
        f"constant constexpr uint D = {symbols['D']}u;\n"
        f"constant constexpr int LOWER_BOUND = {lower_bound};\n"
    )
    marker = "using namespace metal;\n"
    assert device.count(marker) == 1
    return device.replace(marker, marker + preamble, 1)


def _timing(ctx, stages, grids, args_by_set, *, iters=ITERS) -> dict:
    """Run one command buffer of ``iters`` pipeline iterations and report per-iteration times."""
    timing = ctx.run_pipeline(
        stages=stages,
        grids=grids,
        threads=[(1, 1, 1)] * len(stages),
        threadgroup_bytes=[0] * len(stages),
        arg_sets=args_by_set,
        fills=[[] for _ in args_by_set],
        iters=iters,
        dispatch="serial",
    )
    return {
        "wall_per_iter_us": timing.wall_per_iter_s * 1e6,
        "gpu_per_iter_us": timing.gpu_per_iter_s * 1e6,
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _repeat(make, *, repeats=REPEATS) -> dict:
    walls, gpus = [], []
    for _ in range(repeats):
        sample = make()
        walls.append(sample["wall_per_iter_us"])
        gpus.append(sample["gpu_per_iter_us"])
    return {
        "wall_per_iter_us": _median(walls),
        "gpu_per_iter_us": _median(gpus),
        "wall_samples_us": walls,
        "gpu_samples_us": gpus,
        "repeats": repeats,
    }


def _buffer(ctx, nbytes: int) -> object:
    return ctx.new_buffer(max(int(nbytes), 16))


def run_stage_floor(ctx) -> dict:
    """Per-stage floor: ``n`` empty dispatches with a buffer barrier between each pair."""
    arms = {}
    out = _buffer(ctx, 16)
    kernel = ctx.compile(_EMPTY_KERNEL, "probe_empty", math="fast")
    for stages_count in (1, 2, 3, 4):
        args = [[[(0, BufferArg(out))] for _ in range(stages_count)]]
        arms[f"stage_floor_n{stages_count}"] = _repeat(
            lambda args=args, count=stages_count: _timing(
                ctx, [kernel] * count, [(8, 1, 1)] * count, args, iters=FLOOR_ITERS
            ),
            repeats=FLOOR_REPEATS,
        )
    # Least-squares slope over 1..4 stages: the marginal cost of one more dispatch plus barrier.
    counts = [1, 2, 3, 4]
    for key, name in (("wall_per_iter_us", "per_stage_wall_us"), ("gpu_per_iter_us", "per_stage_gpu_us")):
        values = [arms[f"stage_floor_n{count}"][key] for count in counts]
        mean_x = sum(counts) / 4.0
        mean_y = sum(values) / 4.0
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(counts, values)) / sum(
            (x - mean_x) ** 2 for x in counts
        )
        arms[name] = slope
        arms[f"{name}_intercept"] = mean_y - slope * mean_x
    arms["note"] = (
        "synthetic floor: an empty stage at grid 8x1x1 costs far less than one command-buffer "
        "submission, so the GPU timestamps of a near-empty buffer partly measure the encoder gaps "
        "rather than the barrier. Treat this as a lower bound and use the kda pipeline arms "
        "(same work, one stage vs three) for the authoritative per-stage cost."
    )
    return arms


def run_handoff(ctx) -> dict:
    """Write plus read of the real state bytes, at the fixture's width and at a full width."""
    out = _buffer(ctx, 1 << 20)
    arms: dict = {}

    def arm(per_thread: int, threads: int, state_bytes: int) -> dict:
        state = _buffer(ctx, max(state_bytes, 16))
        pair = ctx.compile_all(_handoff_source(per_thread, threads), ["probe_producer", "probe_consumer"], math="fast")
        sample = _repeat(
            lambda pair=pair, state=state, threads=threads: _timing(
                ctx,
                pair,
                [(threads, 1, 1), (threads, 1, 1)],
                [[[(0, BufferArg(state)), (1, BufferArg(out))]] * 2],
            )
        )
        sample["state_bytes"] = state_bytes
        sample["threads"] = threads
        sample["floats_per_thread"] = per_thread
        return sample

    arms["handoff_control"] = arm(1, 1, 4)
    for dim, heads in STATE_SHAPES.items():
        total_floats = heads * dim * dim
        # The fixture's own width: one threadgroup per head, one thread per threadgroup.
        arms[f"handoff_state_fp32_D{dim}"] = arm(total_floats // heads, heads, total_floats * 4)
        # The same bytes at a realistic thread count (16 floats per thread).
        wide = total_floats // 16
        arms[f"handoff_state_fp32_D{dim}_wide"] = arm(16, wide, total_floats * 4)
        for key in (f"handoff_state_fp32_D{dim}", f"handoff_state_fp32_D{dim}_wide"):
            arms[key]["handoff_over_control_us"] = (
                arms[key]["gpu_per_iter_us"] - arms["handoff_control"]["gpu_per_iter_us"]
            )
            written = 2 * arms[key]["state_bytes"]
            arms[key]["round_trip_gbps"] = written / (arms[key]["handoff_over_control_us"] * 1e-6) / 1e9
    return arms


def _kda_buffers(ctx, symbols: dict[str, int]) -> dict[str, object]:
    tokens, heads, dim = symbols["N"], symbols["H"], symbols["D"]
    half = lambda count: _buffer(ctx, count * 2)  # noqa: E731 - a probe, not production code
    buffers = {
        "q": half(tokens * heads * dim),
        "k": half(tokens * heads * dim),
        "v": half(tokens * heads * dim),
        "g": half(tokens * heads * dim),
        "beta": half(tokens * heads),
        "A_log": _buffer(ctx, heads * 4),
        "dt_bias": _buffer(ctx, heads * dim * 4),
        "initial_state": _buffer(ctx, heads * dim * dim * 4),
        "scale": _buffer(ctx, 4),
        "output": half(tokens * heads * dim),
        "final_state": _buffer(ctx, heads * dim * dim * 4),
        "work_items": _buffer(ctx, 3 * 2 * 4),
        "item_counts": _buffer(ctx, 4),
        "handoff_states": _buffer(ctx, 2 * heads * dim * dim * 4),
    }
    # Non-degenerate inputs: the recurrence must do the same work in every arm, and the probe
    # writes real half bit patterns (0.25) instead of leaving the buffers at their allocation value.
    for name in ("q", "k", "v", "g", "beta"):
        view = ctx.view(buffers[name])
        view[:] = struct.pack(f"<{len(view) // 2}e", *([0.25] * (len(view) // 2)))
    view = ctx.view(buffers["A_log"])
    view[:] = struct.pack(f"<{len(view) // 4}f", *([0.5] * (len(view) // 4)))
    view = ctx.view(buffers["dt_bias"])
    view[:] = struct.pack(f"<{len(view) // 4}f", *([0.1] * (len(view) // 4)))
    view = ctx.view(buffers["scale"])
    view[:4] = struct.pack("<f", 0.1)
    return buffers


def _publish(ctx, buffers, spans: list[tuple[int, int]], published: int) -> None:
    view = ctx.view(buffers["work_items"])
    view[:] = struct.pack(f"<{len(view) // 4}I", *([0] * (len(view) // 4)))
    words = []
    for begin, count in spans:
        words.extend([begin, count])
    view[: len(words) * 4] = struct.pack(f"<{len(words)}I", *words)
    counts = ctx.view(buffers["item_counts"])
    counts[:4] = struct.pack("<i", published)


def run_kda_stages(ctx, symbols: dict[str, int], lower_bound: int) -> dict:
    """One command buffer of the shipped fixture: a whole chain versus a cut chain.

    ``pipeline_*`` arms keep the three shipped stages in one command buffer, so the comparison is
    the same submission shape the harness uses.  The ``stage`` arms run one stage alone in its own
    command buffer and are a *diagnostic* of where the device time goes, not a pipeline total.
    """
    device = KDA_DEVICE.read_text(encoding="utf-8")
    source = _kda_standalone_source(device, symbols, lower_bound)
    kernels = ctx.compile_all(source, ["kda_piece0", "kda_piece1", "kda_piece2"], math="safe")
    heads = symbols["H"]
    tokens = symbols["N"]
    pieces = min(3, tokens)
    base, rem = divmod(tokens, pieces)
    spans, cursor = [], 0
    for piece in range(pieces):
        count = base + (1 if piece < rem else 0)
        spans.append((cursor, count))
        cursor += count
    order = (
        "q", "k", "v", "g", "beta", "A_log", "dt_bias", "initial_state", "scale",
        "output", "final_state", "work_items", "item_counts", "handoff_states",
    )

    def bind(table: list[tuple[int, int]], published: int):
        buffers = _kda_buffers(ctx, symbols)
        _publish(ctx, buffers, table, published)
        pairs = [(index, BufferArg(buffers[name])) for index, name in enumerate(order)]
        return pairs

    def pipeline(table: list[tuple[int, int]], published: int, stages: int) -> dict:
        pairs = bind(table, published)
        return _timing(ctx, kernels[:stages], [(heads, 1, 1)] * stages, [[pairs] * stages])

    def one(stage_index: int, table: list[tuple[int, int]], published: int) -> dict:
        pairs = bind(table, published)
        return _timing(ctx, [kernels[stage_index]], [(heads, 1, 1)], [[pairs]])

    arms = {
        "kda_pipeline_1_piece": _repeat(lambda: pipeline([(0, tokens)], 1, 3)),
        "kda_pipeline_3_pieces": _repeat(lambda: pipeline(spans, pieces, 3)),
        "kda_pipeline_1_piece_1_stage": _repeat(lambda: pipeline([(0, tokens)], 1, 1)),
        "kda_stage0_alone_piece": _repeat(lambda: one(0, spans, pieces)),
        "kda_stage1_alone_piece": _repeat(lambda: one(1, spans, pieces)),
        "kda_stage2_alone_piece": _repeat(lambda: one(2, spans, pieces)),
    }
    arms["symbols"] = {"N": tokens, "H": heads, "D": symbols["D"]}
    arms["spans"] = [list(span) for span in spans]
    arms["pipeline_handoff_plus_extra_stage_us"] = (
        arms["kda_pipeline_3_pieces"]["wall_per_iter_us"] - arms["kda_pipeline_1_piece"]["wall_per_iter_us"]
    )
    arms["pipeline_stage_floor_us"] = (
        arms["kda_pipeline_1_piece"]["wall_per_iter_us"] - arms["kda_pipeline_1_piece_1_stage"]["wall_per_iter_us"]
    )
    arms["pipeline_ratio"] = (
        arms["kda_pipeline_3_pieces"]["wall_per_iter_us"] / arms["kda_pipeline_1_piece_1_stage"]["wall_per_iter_us"]
    )
    return arms


def run_probe(path: Path = OUT) -> dict:
    """Run every probe arm and write the evidence document.

    @param path 输出 JSON 路径。
    @return 证据文档。
    """
    with gpu_lock("P2-Q SEQ state handoff probe"):
        ctx = get_context()
        info = ctx.info()
        document = {
            "schema_id": SCHEMA_ID,
            "not_a_candidate": True,
            "publication_allowed": False,
            "target_id": TARGET_ID,
            "device_name": info.get("name"),
            "gpu_family": info.get("gpu_family"),
            "iters": ITERS,
            "rotation_sets": 1,
            "harness_rotation_sets": NSETS,
            "rotation_note": (
                "the probe reuses one buffer set so the arms differ only in where the chain is cut; "
                "the harness rotates 16 sets, which is why its per-stage cost is higher than the probe's"
            ),
            "repeats": REPEATS,
            "stage_floor": run_stage_floor(ctx),
            "handoff": run_handoff(ctx),
            "kda_stage_cost": {
                "small": run_kda_stages(ctx, {"N": 14, "H": 2, "D": 8}, -5),
                "paper": run_kda_stages(ctx, {"N": 256, "H": 8, "D": 64}, -5),
            },
        }
    document["arms"] = (
        [f"stage_floor/{name}" for name in document["stage_floor"] if name.startswith("stage_floor")]
        + [f"handoff/{name}" for name in document["handoff"]]
        + [
            f"kda_stage_cost/{size}/{name}"
            for size in ("small", "paper")
            for name in document["kda_stage_cost"][size]
            if name.startswith("kda_")
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    document = run_probe(args.out)
    floor = document["stage_floor"]
    print(
        f"OK: stage floor {floor['stage_floor_n1']['wall_per_iter_us']:.2f} us -> "
        f"{floor['stage_floor_n4']['wall_per_iter_us']:.2f} us, "
        f"per stage {floor['per_stage_wall_us']:.2f} us"
    )
    for name, arm in document["handoff"].items():
        extra = arm.get("handoff_over_control_us")
        suffix = "" if extra is None else f" (+{extra:.2f} over control)"
        print(f"OK: {name} wall {arm['wall_per_iter_us']:.2f} us gpu {arm['gpu_per_iter_us']:.2f} us{suffix}")
    for size, block in document["kda_stage_cost"].items():
        print(
            f"OK: kda {size} pipeline 1-part {block['kda_pipeline_1_piece']['gpu_per_iter_us']:.2f} us, "
            f"3-part {block['kda_pipeline_3_pieces']['gpu_per_iter_us']:.2f} us, "
            f"1-stage {block['kda_pipeline_1_piece_1_stage']['gpu_per_iter_us']:.2f} us (gpu per iter)"
        )
    print(f"OK: wrote {args.out}")


if __name__ == "__main__":
    main()
