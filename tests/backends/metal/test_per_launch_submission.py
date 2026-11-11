"""P2-G: explicit per-launch submission with per-set SharedEvent dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
from pathlib import Path

import numpy as np
import pytest

from tcenv.backends.metal.runtime import (
    BufferArg,
    MetalContext,
    ScalarArg,
    get_context,
    per_launch_eligible,
)
from tcenv.identity import fingerprint
from tcenv.track import REPO_ROOT


class _Stage:
    def __init__(self, name="candidate"):
        self.name = name
        self.pso = name
        self.bindings = ()


def test_per_launch_eligibility_is_narrower_than_the_native_batch():
    stage = _Stage()
    buf = BufferArg(object())
    args = [[[(0, buf)]], [[(0, buf)]], [[(0, buf)]]]
    assert per_launch_eligible(stages=[stage], threadgroup_bytes=[0], arg_sets=args, fills=[[], [], []])
    assert not per_launch_eligible(stages=[stage, _Stage("second")], threadgroup_bytes=[0, 0], arg_sets=args, fills=[[], [], []])
    assert not per_launch_eligible(stages=[stage], threadgroup_bytes=[16], arg_sets=args, fills=[[], [], []])
    assert not per_launch_eligible(stages=[stage], threadgroup_bytes=[0], arg_sets=args, fills=[[(object(), 0, 4)], [], []])
    scalar = [[[(0, ScalarArg(b"x"))]] for _ in range(3)]
    assert not per_launch_eligible(stages=[stage], threadgroup_bytes=[0], arg_sets=scalar, fills=[[], [], []])


def test_evaluator_requires_an_explicit_known_submission():
    from tcenv.backends.metal.kernel_pipeline import MetalKernelPipelineEvaluator

    assert MetalKernelPipelineEvaluator().batch_submission == "single_buffer"
    assert MetalKernelPipelineEvaluator(batch_submission="per_launch_set_event").batch_submission == "per_launch_set_event"
    with pytest.raises(ValueError, match="unknown batch submission"):
        MetalKernelPipelineEvaluator(batch_submission="unsafe_no_wait")


def test_per_launch_confirmation_uses_a_new_epoch(tmp_path):
    from taic.task_binding import bind_task
    from tcenv.backends.metal.search_candidate import confirmation_input
    from tcenv.target import load_target

    source = (REPO_ROOT / "kernels/apple-m4-max/softmax/fp16/paper/best.metal").read_text()
    binding = bind_task("softmax", "fp16", "paper", target_id="apple-m4-max")
    kwargs = dict(
        target=load_target("apple-m4-max"),
        guidance_digest=fingerprint({"guidance": "P2-G per-launch epoch test"}),
        master_seed=7,
    )
    default = confirmation_input(binding.locator, source, **kwargs)
    experimental = confirmation_input(binding.locator, source, batch_submission="per_launch_set_event", **kwargs)
    assert default.freeze.protocol_epoch == 1
    assert experimental.freeze.protocol_epoch == 4
    assert experimental.freeze.digest != default.freeze.digest
    assert experimental.validation.digest != default.validation.digest
    assert experimental.freeze.task_pack_digest == default.freeze.task_pack_digest


def test_taic_config_maps_the_explicit_submission_to_the_protocol_epoch():
    from types import SimpleNamespace

    from taic.cli import _apply_output_switches
    from taic.llm import SearchConfig
    from taic.v2_evaluation import V2LocalEvaluator

    search = SearchConfig()
    assert search.batch_submission == "single_buffer"
    args = SimpleNamespace(batch_submission="per_launch_set_event", plateau_after=None, task_ledger=None, sweep_tool=None)
    _apply_output_switches(args, search)
    assert search.batch_submission == "per_launch_set_event"
    evaluator = V2LocalEvaluator(SimpleNamespace(is_v2=True), batch_submission=search.batch_submission)
    assert evaluator.protocol_version == "3+per_launch_set_event"
    assert V2LocalEvaluator(SimpleNamespace(is_v2=True)).protocol_version == "3"


def test_per_launch_emits_one_wait_and_signal_per_set_generation():
    import Metal

    class Encoder:
        def __init__(self, cb):
            self.cb = cb

        def setComputePipelineState_(self, pso):
            self.cb.pso = pso

        def setBuffer_offset_atIndex_(self, buf, offset, index):
            self.cb.bound = (buf, offset, index)

        def dispatchThreadgroups_threadsPerThreadgroup_(self, grid, threads):
            self.cb.dispatched = (grid, threads)

        def endEncoding(self):
            pass

    class CommandBuffer:
        def __init__(self, ordinal):
            self.ordinal = ordinal
            self.encoder = Encoder(self)
            self.waits = []
            self.signals = []
            self.dispatched = None

        def computeCommandEncoderWithDispatchType_(self, kind):
            return self.encoder

        def encodeWaitForEvent_value_(self, event, value):
            self.waits.append((event, value))

        def encodeSignalEvent_value_(self, event, value):
            self.signals.append((event, value))

        def commit(self):
            pass

        def waitUntilCompleted(self):
            pass

        def status(self):
            return Metal.MTLCommandBufferStatusCompleted

        def GPUStartTime(self):
            return float(self.ordinal)

        def GPUEndTime(self):
            return float(self.ordinal) + 0.5

    class Queue:
        def __init__(self):
            self.cbs = []

        def commandBuffer(self):
            cb = CommandBuffer(len(self.cbs))
            self.cbs.append(cb)
            return cb

    class Device:
        def __init__(self):
            self.events = []

        def newSharedEvent(self):
            event = object()
            self.events.append(event)
            return event

    ctx = object.__new__(MetalContext)
    ctx.queue = Queue()
    ctx.device = Device()
    ctx.command_buffers = 0
    stage = _Stage()
    args = [[[(0, BufferArg(object()))]] for _ in range(3)]
    recorded = []
    timing = ctx.run_pipeline_per_launch(
        [stage], [(1, 1, 1)], [(32, 1, 1)], [0], args, [[], [], []],
        iters=7,
        dispatch_recorder=lambda *row: recorded.append(row),
        dispatch_workspace=[("scratch",)],
    )
    assert len(ctx.queue.cbs) == 7
    assert ctx.command_buffers == 7
    assert timing.iters == 7 and timing.gpu_s == 6.5
    for i, cb in enumerate(ctx.queue.cbs):
        generation, s = divmod(i, 3)
        expected_wait = [] if generation == 0 else [(ctx.device.events[s], generation)]
        assert cb.waits == expected_wait
        assert cb.signals == [(ctx.device.events[s], generation + 1)]
        assert cb.dispatched is not None
    assert len(recorded) == 7
    assert recorded[0][0] == 0 and recorded[0][-1] == ("scratch",)


@pytest.mark.gpu
def test_per_launch_preserves_read_write_generation_dependencies():
    """Six increments over three sets must leave two updates in each set, never a lost update."""
    from tcenv.backends.registry import load_backend

    ctx = get_context()
    load_backend("metal").prepare_context(ctx)
    kernel = ctx.compile(
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k(device uint* y [[buffer(0)]], uint i [[thread_position_in_grid]]) { y[i] += 1u; }",
        "k",
    )
    count = 1024
    buffers = [ctx.device.newBufferWithLength_options_(count * 4, 0) for _ in range(3)]
    for buf in buffers:
        buf.contents().as_buffer(count * 4)[:] = b"\0" * (count * 4)
    args = [[[(0, BufferArg(buf))]] for buf in buffers]
    before = int(getattr(ctx, "command_buffers", 0))
    timing = ctx.run_pipeline_per_launch(
        [kernel], [(1, 1, 1)], [(count, 1, 1)], [0], args, [[], [], []], iters=6,
    )
    assert int(getattr(ctx, "command_buffers", 0)) == before + 6
    assert timing.wall_s > 0 and timing.gpu_s >= 0
    for buf in buffers:
        out = np.frombuffer(buf.contents().as_buffer(count * 4), dtype=np.uint32)
        assert np.all(out == 2)


@pytest.mark.gpu
def test_per_launch_rejects_a_fill_instead_of_dropping_workspace_zeroing():
    ctx = get_context()
    stage = _Stage()
    buf = ctx.device.newBufferWithLength_options_(16, 0)
    with pytest.raises(ValueError, match="no fill"):
        ctx.run_pipeline_per_launch(
            [stage], [(1, 1, 1)], [(32, 1, 1)], [0],
            [[[(0, BufferArg(buf))]]], [[(buf, 0, 4)]], iters=1,
        )


def _record(name: str, payload: dict) -> None:
    directory = os.environ.get("SHAPE_PORTFOLIO_P2G_EVIDENCE_DIR")
    if not directory:
        return
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    document = {
        "host": {"chip": platform.processor(), "machine": platform.machine(), "system": platform.system()},
        **payload,
    }
    (target / name).write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _softmax():
    from taic.task_binding import bind_task
    from tcenv.target import load_target

    source = (REPO_ROOT / "kernels/apple-m4-max/softmax/fp16/paper/best.metal").read_text()
    binding = bind_task("softmax", "fp16", "paper", target_id="apple-m4-max")
    return binding, source, load_target("apple-m4-max")


@pytest.mark.gpu
def test_p2_g_softmax_per_launch_fresh_confirmation():
    from taic.portfolio_manifest import ActivePortfolioPointer, DispatchManifest
    from tcenv.backends.metal.portfolio_confirmation import activation_expectation, confirm_subject
    from tcenv.backends.metal.search_candidate import confirmation_input
    from tcenv.portfolio.dispatch import DispatchRequest

    binding, source, target = _softmax()
    contract = binding.locator.load()
    frozen = confirmation_input(
        binding.locator,
        source,
        target=target,
        guidance_digest=fingerprint({"guidance": "P2-G per-launch softmax", "source_sha256": hashlib.sha256(source.encode()).hexdigest()}),
        master_seed=7,
        batch_submission="per_launch_set_event",
    )
    assert frozen.freeze.protocol_epoch == 4
    golden_point = frozen.freeze.domain_plan.find("fp16", "paper", "golden")
    assert golden_point is not None
    golden_request = DispatchRequest.from_instance(
        contract.instance, target_abi_digest=golden_point.target_abi_digest, usage="golden",
    )
    documents = []
    for request in (golden_request, frozen.request):
        document = confirm_subject(
            locator=binding.locator,
            domain_plan=frozen.freeze.domain_plan,
            portfolio=frozen.freeze.portfolio(),
            request=request,
            evidence=frozen.evidence,
            validation=frozen.validation,
            freeze=frozen.freeze,
            batch_submission="per_launch_set_event",
        )
        assert document["status"] == "confirmed", document.get("reason")
        assert document["identity"]["kind"] == "match"
        assert document["confirmation_report"]["protocol_version"] == "3+per_launch_set_event"
        assert document["expected"]["protocol_version"] == "3+per_launch_set_event"
        assert document["expected"]["evaluation"]["protocol_version"] == "3+per_launch_set_event"
        if request.usage == "benchmark":
            assert document["confirmation_report"]["benchmark"]["batch_submission"] == "per_launch_set_event"
        documents.append(document)
    expectations = {
        document["instance_key"]: activation_expectation(
            locator=binding.locator, freeze=frozen.freeze, validation=frozen.validation, document=document,
        )
        for document in documents
    }
    active = DispatchManifest.from_confirmations(
        frozen.freeze, frozen.validation, documents, expectations, locator=binding.locator,
    ).activate(frozen.freeze, frozen.validation)
    assert len(active.subjects) == 2 and active.activation_generation == 1
    pointer = ActivePortfolioPointer(active.digest, active.activation_generation)
    _record("per-launch-softmax-confirmation.json", {
        "schema_id": "shapeforge.p2g-per-launch-confirmation.1",
        "publication_allowed": False,
        "portfolio_freeze_digest": frozen.freeze.digest,
        "validation_plan_digest": frozen.validation.digest,
        "documents": documents,
        "dispatch_manifest": active.document(),
        "active_pointer": pointer.document(),
    })


@pytest.mark.gpu
def test_p2_g_softmax_paired_fresh_evaluations():
    from tcenv.backends.metal.search_candidate import evaluate_candidate

    binding, source, target = _softmax()
    pairs = []
    for seed in (2026111023, 2026111024, 2026111025):
        row = {"seed": seed}
        for label, submission in (("single_buffer", "single_buffer"), ("per_launch_set_event", "per_launch_set_event")):
            run = evaluate_candidate(
                binding.locator, source,
                target=target, mode="full", seed=seed, reps=15,
                batch_submission=submission,
            )
            report = run.result.report
            assert report["status"] == "ok", report.get("message")
            assert report["correctness"]["ok"] and report["safety"]["ok"]
            if label == "single_buffer":
                assert report["protocol_version"] == "3"
                assert "batch_submission" not in report["benchmark"]
            else:
                assert report["protocol_version"] == "3+per_launch_set_event"
                assert report["benchmark"]["batch_submission"] == "per_launch_set_event"
            row[label] = {
                "candidate_us": report["candidate_us"],
                "mlx_us": report["baseline_us"]["mlx"],
                "speedup": report["speedup"],
                "evaluation_identity": run.result.identity,
                "report": report,
            }
        pairs.append(row)
    old_median = statistics.median(row["single_buffer"]["speedup"] for row in pairs)
    new_median = statistics.median(row["per_launch_set_event"]["speedup"] for row in pairs)
    assert new_median > 1.0, new_median
    assert new_median > old_median + 0.03, (old_median, new_median)
    _record("per-launch-softmax-pairs.json", {
        "schema_id": "shapeforge.p2g-per-launch-pairs.1",
        "publication_allowed": False,
        "candidate_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "pairs": pairs,
        "median_speedup": {"single_buffer": old_median, "per_launch_set_event": new_median},
    })
