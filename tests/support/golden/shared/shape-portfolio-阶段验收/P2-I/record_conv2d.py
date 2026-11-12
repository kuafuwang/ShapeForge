#!/usr/bin/env python3
"""Quiet-window conv2d ablation: one fresh confirmation per arm, then isolated stage times.

The confirmation median, cv and baseline ratios come from ``confirm_candidate``.
Stage times are a second measurement: each kernel in its own command buffer after
one fused warmup.  They are not the confirmation median and they are not a
publication.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcenv.backends.metal.runtime import BufferArg, TextureArg, get_context
from tcenv.backends.metal.sanitize import GUARD_BYTES
from tcenv.backends.metal.scratch_texture import make_scratch_texture, scratch_texture_layout
from tcenv.backends.metal.search_candidate import SEARCH_SEED, confirm_candidate, evaluate_candidate, guidance_fingerprint
from tcenv.backends.metal.v2_runtime import benchmark_safety_identity
from tcenv.taskcontract.locator import discover_standard_tasks
from tcenv.taskcontract.portfolio_runner import locator_for_instance
from tcenv.target import load_target
from tests.backends.metal.conv2d_scratch_kernel import launch_counts, metal_source, metal_source_recompute


def _master_key() -> bytes:
    import importlib.util

    path = ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2-I/verify.py"
    spec = importlib.util.spec_from_file_location("p2i_acceptance_key", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.P2H_MASTER_KEY


def _locator():
    found = next(
        item
        for item in discover_standard_tasks(ROOT / "tasks", code_root=ROOT)
        if item.op_name == "conv2d"
    )
    return locator_for_instance(found, "fp16", "paper")


def _symbols(locator) -> dict[str, int]:
    return {str(name): int(value) for name, value in locator.load().instance.symbols.items()}


def _why(report: dict) -> str:
    parts = [f"status={report.get('status')!r}"]
    if report.get("message"):
        parts.append(str(report["message"]))
    errors = (report.get("compile") or {}).get("errors")
    if errors:
        parts.append(f"compile: {errors[:800]}")
    for key in ("abi", "safety", "correctness"):
        block = report.get(key) or {}
        violations = block.get("violations") or []
        if violations:
            parts.append(key + ": " + "; ".join(str(item) for item in violations[:3]))
    return " | ".join(parts)


def _bench(report: dict) -> dict:
    bench = report.get("benchmark") or {}
    candidate = bench.get("candidate") or {}
    return {
        "median_us": candidate.get("median_us"),
        "cv": candidate.get("cv"),
        "reps": candidate.get("reps"),
        "iters": candidate.get("iters"),
        "candidate_gpu_us_median": bench.get("candidate_gpu_us_median"),
        "baseline_us": report.get("baseline_us") or {},
        "speedup": report.get("speedup"),
        "speedups": report.get("speedups") or {},
        "speedup_withheld": bench.get("speedup_withheld"),
    }


def _confirm(locator, source: str, *, label: str) -> dict:
    print(f"confirm {label}", flush=True)
    document = confirm_candidate(
        locator,
        source,
        target=load_target("apple-m4-max"),
        guidance_digest=guidance_fingerprint("conv2d scratch ablation", label, source),
        master_seed=_master_key(),
    )
    seed = int(document["confirmation_seed"])
    if seed == SEARCH_SEED:
        raise SystemExit(f"{label}: fresh seed collapsed to the search seed {SEARCH_SEED}")
    report = document["confirmation_report"]
    print(f"  status={document['status']} reason={document.get('reason')} seed={seed} {_why(report)}", flush=True)
    return {
        "label": label,
        "status": document["status"],
        "reason": document.get("reason"),
        "candidate_sha256": document["candidate_sha256"],
        "search_seed": document["search_seed"],
        "confirmation_seed": seed,
        "confirmation_attempts": document["confirmation_attempts"],
        "identity": document["identity"].get("kind"),
        "publication_allowed": False,
        "search": _bench(document["search_report"]),
        "confirmation": _bench(report),
        "subject": document,
    }


def _buffer(ctx, nbytes: int, payload: bytes | None = None):
    buf = ctx.device.newBufferWithLength_options_(nbytes, 0)
    if payload is not None:
        buf.contents().as_buffer(nbytes)[:] = payload
    return buf


def _time_kernel(ctx, kernel, grid, args) -> float:
    samples = []
    for _ in range(5):
        timing = ctx.run_pipeline([kernel], [grid], [(64, 1, 1)], [0], [[args]], [[]], iters=8)
        samples.append(float(timing.gpu_s) / 8.0 * 1e6)
    return float(np.median(samples))


def _stage_times(source: str, symbols: dict[str, int], tile: int, *, stages: int) -> dict:
    header = "\n".join(f"constant constexpr uint {name} = {int(value)}u;" for name, value in symbols.items())
    text = source.replace("using namespace metal;\n", "using namespace metal;\n" + header + "\n", 1)
    ctx = get_context()
    n, h, w, c = (symbols[name] for name in ("N", "H", "W", "C"))
    oc, oh, ow = (symbols[name] for name in ("OC", "OH", "OW"))
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n * h * w * c).astype(np.float16)
    weight = rng.standard_normal(oc * 9 * c).astype(np.float16)
    xb = _buffer(ctx, x.nbytes, x.tobytes())
    wb = _buffer(ctx, weight.nbytes, weight.tobytes())
    yb = _buffer(ctx, n * oh * ow * oc * 2)
    if stages == 1:
        kernel = ctx.compile(text, "conv2d_kernel", math="safe")
        _stage0, stage1 = launch_counts(symbols, tile)
        args = [(0, BufferArg(xb)), (1, BufferArg(wb)), (2, BufferArg(yb))]
        ctx.run_pipeline([kernel], [(stage1, 1, 1)], [(64, 1, 1)], [0], [[args]], [[]], iters=1)
        return {
            "stage0_gpu_us": None,
            "stage1_gpu_us": _time_kernel(ctx, kernel, (stage1, 1, 1), args),
            "protocol": "one kernel, own command buffer, 5 medians of gpu_s/8 after one warmup; not the confirmation median",
        }
    elements = 16 if tile == 2 else 36
    count = elements * c * oc
    _width, _height, row = scratch_texture_layout(count, guard_bytes=GUARD_BYTES)
    ub = _buffer(ctx, count * 4 + row)
    texture = make_scratch_texture(ub, offset=0, count=count, guard_bytes=GUARD_BYTES)
    stage0, stage1 = launch_counts(symbols, tile)
    first = ctx.compile(text, "conv2d_wgrid", math="safe")
    second = ctx.compile(text, "conv2d_kernel", math="safe")
    # Buffer slot 0 and texture slot 0 are different binding spaces.
    producer = [(0, BufferArg(wb)), (0, TextureArg(texture))]
    consumer = [(0, BufferArg(xb)), (0, TextureArg(texture)), (1, BufferArg(yb)), (2, BufferArg(wb))]
    ctx.run_pipeline(
        [first, second],
        [(stage0, 1, 1), (stage1, 1, 1)],
        [(64, 1, 1), (64, 1, 1)],
        [0, 0],
        [[producer, consumer]],
        [[]],
        iters=1,
    )
    return {
        "stage0_gpu_us": _time_kernel(ctx, first, (stage0, 1, 1), producer),
        "stage1_gpu_us": _time_kernel(ctx, second, (stage1, 1, 1), consumer),
        "protocol": "each stage in its own command buffer after one fused warmup; not the confirmation median",
    }


def main() -> None:
    locator = _locator()
    symbols = _symbols(locator)
    recompute = metal_source_recompute(symbols, 2)
    scratch_f2 = metal_source(symbols, 2)
    scratch_f4 = metal_source(symbols, 4)
    print("quick recompute F(2x2)", flush=True)
    quick = evaluate_candidate(
        locator, recompute, target=load_target("apple-m4-max"), mode="quick", reps=1,
    ).result.report
    if quick.get("status") != "ok":
        raise SystemExit("recompute quick failed: " + _why(quick))
    print("  ok", flush=True)
    arms = [
        ("recompute-f2", recompute, 2, 1),
        ("scratch-f2", scratch_f2, 2, 2),
        ("scratch-f4", scratch_f4, 4, 2),
    ]
    out = Path(__file__).resolve().parent / "evidence" / "conv2d-ablation.json"
    records = []
    for label, source, tile, stages in arms:
        record = _confirm(locator, source, label=label)
        print(f"stage split {label}", flush=True)
        record["isolated_stages"] = _stage_times(source, symbols, tile, stages=stages)
        records.append(record)
        payload = {
            "schema_id": "shapeforge.p2i-conv2d-ablation.1",
            "publication_allowed": False,
            "search_seed": SEARCH_SEED,
            "benchmark_safety_identity": benchmark_safety_identity(),
            "arms": records,
        }
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"OK: wrote {out} ({len(records)} arms)", flush=True)


if __name__ == "__main__":
    main()
