"""M1: Metal scratch is typed, per-rotation, and recomputed on every timed iteration.

These checks stay on the CPU.  They do not launch a GPU kernel and they do not
call a model.  A real conv2d device loop is the later GPU smoke.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from taic.trust import current_prompt_sha1
from tcenv.backends.metal.host_program import MetalHostProgram
from tcenv.backends.metal.launch import parse_launch_spec
from tcenv.backends.metal.runtime import Binding, pipeline_ops
from tcenv.backends.metal.sanitize import GUARD_BYTES
from tcenv.backends.metal.stage_binding import ReflectedArgument, bind_pipeline_arguments
from tcenv.backends.metal.track_adapter import MetalV2Executor, launch_constants, msl_single_geometry_resolver
from tcenv.backends.metal.v2_runtime import MetalV2Runtime, V2ArgSet, _poison_word
from tcenv.build_projection import project_full
from tcenv.contract_errors import GeometryError, TaskContractError
from tcenv.evaluate import EvalReport
from tcenv.hostprogram.contract import host_contract_spec, host_workspace_view
from tcenv.hostprogram.native_runtime import NativeHostRuntime
from tcenv.hostprogram.native_toolchain import compile_native_host_program, discover_native_toolchain
from tcenv.portfolio.candidate import _require_bundle_shape, parse_portfolio
from tcenv.remote.artifact_builder import (
    BuildArtifact,
    BuildEvidence,
    BuildOutcome,
    BuildProfile,
    BuildProvenance,
)
from tcenv.remote.artifact_bundle import ArtifactBundle, BundleEntry, NativeExecutable
from tcenv.schema import PROTOCOL_VERSION
from tcenv.scratch import bundle_delivery_allowed
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.taskcontract.loader import load_task_contract
from tcenv.taskcontract.locator import REPO_ROOT, discover_standard_tasks
from tcenv.taskcontract.portfolio_runner import _preflight_hook, locator_for_instance, run_portfolio_task
from tcenv.taskcontract.runner import evaluate_task
from tcenv.track import (
    CandidateArtifact,
    CandidateEntry,
    CandidateLifecycle,
    HostStep,
    PrepareDeclaration,
    candidate_from_single_source,
    geometry_pair,
    select_geometry,
)
from tcenv.track.abi import abi_limits_from_target, derive_tensor_memory_contract
from tcenv.track.delivery import parse_candidate_geometry
from tcenv.track.geometry import resolve_geometry
from tcenv.track.protocol import TimingDiscipline, benchmark_launch_plan, correctness_launch_plan
from tests.shared.test_shape_portfolio_runner import RecordingBuilder, RecordingEvaluator, _authority, _request
from tests.support.shape_portfolio_metal import (
    evidence_all_valid,
    geometry_bytes,
    geometry_document,
    load_p1_task,
    p1_instances,
    parent_entries,
    portfolio_document,
)
from tcenv.portfolio.domain import normalize_semantic_domain_plan

PROMPT_EPOCH = json.loads((REPO_ROOT / "tests/support/golden/shared/acceptance-epochs/fixtures/p2m/prompt-epoch.json").read_text())
PROMPT_SHA1 = "e9d28ff61819a041610d763f5c1464ea860265be2d414f78c15a1f8aea1fdbcf"
OLD_LAUNCH = """\
/* taic-launch
math safe
workspace tmp 16 zero
launch k grid=1,1,1 threads=8,1,1 workspace=tmp
*/
"""
OLD_DIGEST = (
    "math=safe\n"
    "workspace tmp 16 zero\n"
    "launch k grid=1,1,1 threads=8,1,1 threadgroup=0 workspace=tmp"
)


def _conv2d():
    # Load the v2 directory directly.  The test session routes bind_task("conv2d")
    # through the archived v1 specimen, which has no OpSpec instance.
    root = Path(__file__).resolve().parents[3]
    contract = load_task_contract(root / "tasks" / "conv2d", dtype="fp16", size="small")
    target = load_target(DEFAULT_TARGET_ID)
    abi = derive_tensor_memory_contract(contract.instance, target=target)
    limits = abi_limits_from_target(target)
    return contract, abi, limits


def _source(*, terms: str = "36*C*OC", flag: str = "", first: str = "write", name: str = "uwin") -> str:
    suffix = f" {flag}" if flag else ""
    return f"""\
/* taic-launch
math safe
scratch {name} fp32 {terms}{suffix}
launch conv2d_wgrid grid=1,1,1 threads=32,1,1 scratch={name}:{first}
launch conv2d_kernel grid=2,1,1 threads=64,1,1 scratch={name}:read
*/
"""


class _Ctx:
    def new_buffer(self, nbytes: int) -> bytearray:
        return bytearray(nbytes)

    def view(self, buf: bytearray) -> bytearray:
        return buf


def _resolved(source: str, contract, abi, limits):
    geometry = msl_single_geometry_resolver(contract)(candidate_from_single_source(source))
    return resolve_geometry(geometry, instance=contract.instance, contract=abi, limits=limits)


def _reflect(
    dtype: str, elem: int, name: str, index: int, access: str, *, kind: str = "buffer"
) -> ReflectedArgument:
    return ReflectedArgument(
        index=index,
        name=name,
        kind=kind,
        access=access,
        scalar_type=None if kind != "buffer" else dtype,
        elem_size=None if kind != "buffer" else elem,
    )


def test_prompt_epoch_and_old_launch_digest_stay_put():
    """@brief 提示面登记到 P2-M；没有 scratch 的 launch digest 与旧文本相同。"""
    from tests.support.live_anchors import load_live_anchors
    live = load_live_anchors(REPO_ROOT)["prompt_sha1"]
    assert current_prompt_sha1() == live
    assert PROMPT_SHA1 == PROMPT_EPOCH["current_prompt_sha1"]
    assert live != PROMPT_EPOCH["previous_prompt_sha1"]
    assert parse_launch_spec(OLD_LAUNCH, {}).digest() == OLD_DIGEST
    assert parse_launch_spec(OLD_LAUNCH, {}).scratches == ()


def test_conv2d_single_and_bundle_scratch_precheck_and_binding():
    """@brief F(4x4)/F(2x2) 预检、槽位绑定、空 bundle 仍拒绝、非空 scratch 放行。"""
    contract, abi, limits = _conv2d()
    symbols = contract.instance.symbols
    wide = _resolved(_source(), contract, abi, limits)
    narrow = _resolved(_source(terms="16*C*OC"), contract, abi, limits)
    assert wide.scratch[0].nbytes == 36 * int(symbols["C"]) * int(symbols["OC"]) * 4
    assert narrow.scratch[0].nbytes == 16 * int(symbols["C"]) * int(symbols["OC"]) * 4
    assert wide.stages[0].scratch[0].access == "write"
    assert wide.stages[1].scratch[0].access == "read"
    half = _reflect("half", 2, "x", 0, "read")
    weight = _reflect("half", 2, "w", 0, "read")
    # 合法 producer 是写纹理，槽位在纹理空间。buffer 的 access=write 反射上不了这台机器。
    uwin_w = _reflect("float", 4, "uwin", 0, "write", kind="texture")
    uwin_r = _reflect("float", 4, "uwin", 0, "read", kind="texture")
    out = _reflect("half", 2, "y", 1, "write")
    weight_out = _reflect("half", 2, "w", 2, "read")
    bindings = bind_pipeline_arguments(
        wide,
        contract=abi,
        reflection={
            "conv2d_wgrid": (weight, uwin_w),
            "conv2d_kernel": (half, uwin_r, out, weight_out),
        },
        limits=limits,
    )
    assert [argument.name for argument in bindings[0].arguments] == ["w", "uwin"]
    assert [(argument.slot, argument.binding_kind) for argument in bindings[0].arguments] == [
        (0, "buffer"),
        (0, "texture"),
    ]
    assert bindings[0].arguments[1].role == "scratch"
    # buffer 槽和纹理槽是两套下标。consumer 的 x/y/w 占 buffer 0..2，uwin 占纹理 0。
    assert [argument.name for argument in bindings[1].arguments] == ["x", "y", "w", "uwin"]
    assert [(argument.slot, argument.binding_kind) for argument in bindings[1].arguments] == [
        (0, "buffer"),
        (1, "buffer"),
        (2, "buffer"),
        (0, "texture"),
    ]
    ctx = _Ctx()
    first = V2ArgSet(ctx, abi, wide)
    second = V2ArgSet(ctx, abi, wide)
    assert first.workspace_bufs["uwin"] is not second.workspace_bufs["uwin"]
    assert first.workspace_bufs["uwin"].mem is not second.workspace_bufs["uwin"].mem
    assert first.workspace_fills_declared_zero() == []
    legacy = SimpleNamespace(stages=wide.stages)
    allocated = V2ArgSet(ctx, abi, legacy)
    assert "uwin" not in allocated.workspace_bufs
    assert "x" in allocated.buffers
    entries = (
        CandidateEntry(path="device/k.metal", role="device_kernel", media_type="text/plain", data=b"kernel\n"),
        CandidateEntry(path="host/tiling.c", role="host_tiling", media_type="text/plain", data=b"int host_tiling(void){return 0;}\n"),
    )
    with pytest.raises(TaskContractError, match="declares no workspace tensor"):
        _require_bundle_shape(entries, where="conv2d", workspace_declared=False, scratch_declared=False)
    _require_bundle_shape(entries, where="conv2d", workspace_declared=False, scratch_declared=True)
    assert bundle_delivery_allowed(workspace_declared=False, scratch_declared=False) is False
    document = {
        "schema_version": 1,
        "source": "candidate",
        "scratch": [{"name": "uwin", "dtype": "fp32", "size_terms": ["16", "C", "OC"]}],
        "stages": [
            {
                "kernel": "conv2d_wgrid",
                "grid": [1, 1, 1],
                "threads": [32, 1, 1],
                "scratch": [{"name": "uwin", "access": "write"}],
            },
            {
                "kernel": "conv2d_kernel",
                "grid": [2, 1, 1],
                "threads": [64, 1, 1],
                "scratch": [{"name": "uwin", "access": "read"}],
            },
        ],
    }
    bundle_geometry = parse_candidate_geometry(document)
    resolved_bundle = resolve_geometry(
        bundle_geometry, instance=contract.instance, contract=abi, limits=limits
    )
    assert resolved_bundle.scratch[0].nbytes == narrow.scratch[0].nbytes


def test_every_timed_iteration_dispatches_both_stages_and_identity_misses():
    """@brief 每一轮都发射 stage 0 和 stage 1；改 scratch 后旧 digest 不得命中。"""
    contract, abi, limits = _conv2d()
    ops = pipeline_ops(4, 2, 2, concurrent=False, filled=[False, False])
    dispatches = [item for item in ops if item[0] == "dispatch"]
    assert dispatches == [
        ("dispatch", 0, 0),
        ("dispatch", 0, 1),
        ("dispatch", 1, 0),
        ("dispatch", 1, 1),
        ("dispatch", 0, 0),
        ("dispatch", 0, 1),
        ("dispatch", 1, 0),
        ("dispatch", 1, 1),
    ]
    assert not any(item[0] == "fill" for item in ops)
    zeroed = pipeline_ops(2, 1, 2, concurrent=False, filled=[True])
    assert zeroed[0] == ("fill", 0)
    assert ("dispatch", 0, 0) in zeroed and ("dispatch", 0, 1) in zeroed
    left = msl_single_geometry_resolver(contract)(candidate_from_single_source(_source()))
    right = msl_single_geometry_resolver(contract)(candidate_from_single_source(_source()))
    pair_a = geometry_pair(left, instance=contract.instance, contract=abi, limits=limits)
    pair_b = geometry_pair(right, instance=contract.instance, contract=abi, limits=limits)
    assert pair_a.identity == pair_b.identity
    changed = msl_single_geometry_resolver(contract)(candidate_from_single_source(_source(terms="16*C*OC")))
    pair_c = geometry_pair(changed, instance=contract.instance, contract=abi, limits=limits)
    cache = {pair_a.identity: "old"}
    assert pair_c.identity not in cache
    assert pair_c.identity != pair_a.identity


def test_scratch_lifecycle_negatives_and_device_zero_stay_inside_timing():
    """@brief 借 y、未写先读、重复计费被拒；device_zero 不在 prepare 计划里清零。"""
    contract, abi, limits = _conv2d()
    with pytest.raises(GeometryError, match="collides"):
        _resolved(_source(name="y"), contract, abi, limits)
    with pytest.raises(GeometryError, match="first used as read"):
        _resolved(_source(first="read"), contract, abi, limits)
    billed = """\
/* taic-launch
math safe
workspace uwin 64
scratch uwin fp32 16
launch k grid=1,1,1 threads=8,1,1 workspace=uwin scratch=uwin:write
*/
"""
    with pytest.raises(Exception, match="workspace"):
        parse_launch_spec(billed, {"C": 1})
    zeroed = _resolved(_source(flag="zero"), contract, abi, limits)
    assert zeroed.scratch[0].init_policy == "device_zero"
    ctx = _Ctx()
    argset = V2ArgSet(ctx, abi, zeroed)
    assert argset.workspace_zero["uwin"] is True
    assert argset.workspace_fills_declared_zero()
    argset.workspace_bufs["uwin"].poison()
    before = argset.workspace_bufs["uwin"].read()
    assert before == bytes([0xFF]) * len(before)
    argset.workspace_bufs["uwin"].write(b"\x00" * len(before))
    assert argset.workspace_bufs["uwin"].read() != before
    discipline = SimpleNamespace(mode="fresh_rotation")
    bench = benchmark_launch_plan(abi, zeroed, discipline=discipline)
    assert bench.action("uwin") == "untouched"
    out1 = correctness_launch_plan(abi, zeroed, "out1")
    out2 = correctness_launch_plan(abi, zeroed, "out2")
    assert out1.action("uwin") == "poison"
    assert out2.action("uwin") == "zero"
    with pytest.raises(GeometryError, match="host_written requires"):
        _resolved(_source(flag="host"), contract, abi, limits)
    hosted = resolve_geometry(
        msl_single_geometry_resolver(contract)(candidate_from_single_source(_source(flag="host", terms="4"))),
        instance=contract.instance,
        contract=abi,
        limits=limits,
        host_program_present=True,
    )
    assert hosted.scratch[0].init_policy == "host_written"
    assert hosted.scratch[0].nbytes == 16


def test_host_written_scratch_really_runs_and_a_nonzero_return_fails():
    """@brief host_written 会编译并执行 Host Program；返回 0 才提交，返回 1 失败。"""
    contract, abi, _limits = _conv2d()
    view = host_workspace_view("uwin", "fp32", 4)
    spec = host_contract_spec(instance=contract.instance, tensors=abi.tensors, mapped_limit=0, extra_workspaces=(view,))
    assert spec.workspace("uwin").capacity == 4
    base = spec.document()
    plain = host_contract_spec(instance=contract.instance, tensors=abi.tensors, mapped_limit=0)
    assert "uwin" not in {item["name"] for item in plain.document()["workspaces"]}
    assert base != plain.document()
    ok_source = r"""
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)inputs; (void)shapes; (void)target; (void)resolved;
    workspaces->uwin[0] = 1.0f;
    workspaces->uwin[1] = 1.0f;
    workspaces->uwin[2] = 1.0f;
    workspaces->uwin[3] = 1.0f;
    workspaces->uwin_offset = 0u;
    workspaces->uwin_count = 4u;
    return 0;
}
"""
    bad_source = ok_source.replace("return 0;", "return 1;")
    toolchain = discover_native_toolchain()
    target = load_target(DEFAULT_TARGET_ID)
    values = {
        "max_buffer_index": int(target.raw_limits["max_buffer_index"]),
        "max_threads_per_threadgroup": int(target.raw_limits["max_threads_per_threadgroup"]),
        "max_threadgroup_memory_bytes": int(target.raw_limits["max_threadgroup_memory_bytes"]),
    }

    def _run(source: str):
        build = compile_native_host_program(source, spec=spec, entry_path="host/tiling.c", toolchain=toolchain)
        runtime = NativeHostRuntime(build, spec)
        runtime.set_target(values)
        return runtime.execute(lambda *_args: b"", values=None, audit=True)

    good = _run(ok_source)
    assert good.ok, good.error_detail
    written = next(span for span in good.spans if span.name == "uwin")
    assert written.data == struct.pack("<4f", 1.0, 1.0, 1.0, 1.0)
    bad = _run(bad_source)
    assert bad.ok is False
    assert not bad.spans


def test_first_poison_producer_reflected_readwrite_is_refused():
    """@brief 首个 uninitialized_poison 使用即使声明 write，反射 readwrite 也拒绝。"""
    contract, abi, limits = _conv2d()
    wide = _resolved(_source(), contract, abi, limits)
    half = _reflect("half", 2, "x", 0, "read")
    weight = _reflect("half", 2, "w", 0, "read")
    uwin_rw = _reflect("float", 4, "uwin", 1, "readwrite")
    uwin_r = _reflect("float", 4, "uwin", 1, "read")
    out = _reflect("half", 2, "y", 2, "write")
    with pytest.raises(GeometryError, match="readwrite"):
        bind_pipeline_arguments(
            wide,
            contract=abi,
            reflection={"conv2d_wgrid": (weight, uwin_rw), "conv2d_kernel": (half, uwin_r, out)},
            limits=limits,
        )
    zeroed = _resolved(_source(flag="zero"), contract, abi, limits)
    accepted = bind_pipeline_arguments(
        zeroed,
        contract=abi,
        reflection={"conv2d_wgrid": (weight, uwin_rw), "conv2d_kernel": (half, uwin_r, out)},
        limits=limits,
    )
    assert accepted[0].arguments[1].name == "uwin"
    texture_w = ReflectedArgument(index=0, name="uwin", kind="texture", access="write")
    texture_r = ReflectedArgument(index=0, name="uwin", kind="texture", access="read")
    bound = bind_pipeline_arguments(
        wide,
        contract=abi,
        reflection={
            "conv2d_wgrid": (_reflect("half", 2, "w", 0, "read"), texture_w),
            "conv2d_kernel": (
                _reflect("half", 2, "x", 0, "read"),
                texture_r,
                _reflect("half", 2, "y", 1, "write"),
            ),
        },
        limits=limits,
    )
    produced = next(argument for argument in bound[0].arguments if argument.name == "uwin")
    consumed = next(argument for argument in bound[1].arguments if argument.name == "uwin")
    assert produced.binding_kind == "texture" and produced.access == "write" and produced.slot == 0
    assert consumed.binding_kind == "texture" and consumed.access == "read" and consumed.slot == 0


def test_scratch_texture_layout_puts_the_extra_row_in_the_guard():
    """@brief fp32 scratch 的纹理多一行，且这一行不超过尾保护带。"""
    from tcenv.backends.metal.sanitize import GUARD_BYTES
    from tcenv.backends.metal.scratch_texture import scratch_texture_layout

    width, height, row = scratch_texture_layout(4, guard_bytes=GUARD_BYTES)
    assert (width, height, row) == (4, 2, 16)
    for count in (16 * 32 * 64, 36 * 32 * 64, 16 * 64 * 128, 36 * 64 * 128):
        width, height, row = scratch_texture_layout(count, guard_bytes=GUARD_BYTES)
        assert count % width == 0
        assert row == width * 4 and row <= GUARD_BYTES and row % 16 == 0
        assert height == count // width + 1
        assert height <= 16384


_SENTINEL = (0x5A5A5A5A).to_bytes(4, "little")
_SCRATCH_TOTAL = GUARD_BYTES * 2 + 256
_BUILD_PROFILE = BuildProfile(
    build_profile="metal-msl-control-host-v1",
    builder_id="metal-source-builder",
    builder_version="1",
    build_host="control_host",
    toolchain_id="apple-metal",
    toolchain_version="unknown",
    sdk_version="unknown",
    compile_profile={"language": "msl", "math": "safe", "device": "test", "gpu_family": "test"},
)


def _is_probe_word(word: bytes) -> bool:
    """@brief 覆盖探针涂上的 signaling NaN。``0xFF`` 初值和普通写入都不是这种字。"""
    value = int.from_bytes(word, "little")
    return (value & 0x7FC00000) == 0x7F800000 and (value & 0x003FFFFF) != 0


def _payload_is_coverage_probe(payload: bytes) -> bool:
    return any(_is_probe_word(payload[index : index + 4]) for index in range(0, len(payload) - 3, 4))


class _ScratchCtx:
    """@brief 不碰 GPU 的缓冲与流水线替身。``lazy`` 在看到哨兵后不再改 scratch。"""

    def __init__(
        self,
        *,
        lazy: bool = False,
        capture: bool = False,
        probe_oob: bool = False,
        corrupt_inputs: bool = False,
    ) -> None:
        self.lazy = lazy
        self.capture = capture
        self.probe_oob = probe_oob
        self.corrupt_inputs = corrupt_inputs
        self.pipeline_calls = 0
        self.probe_oob_writes = 0
        self.scratch_at_dispatch: bytes | None = None
        self.timed_input_writes = 0
        self.probe_input_writes = 0
        self.probe_dispatches = 0
        self.mutated_inputs: dict[int, tuple[bytearray, int, int]] = {}
        self.bytes_at_probe: dict[int, int] = {}

    def new_buffer(self, nbytes: int) -> bytearray:
        return bytearray(nbytes)

    def view(self, buf: bytearray) -> bytearray:
        return buf

    def _apply(self, stage_args) -> None:
        if self.corrupt_inputs:
            self._apply_corrupt_inputs(stage_args)
            return
        for _slot, arg in stage_args:
            buf = getattr(arg, "buf", None)
            if buf is None or len(buf) != _SCRATCH_TOTAL:
                continue
            offset = int(arg.offset)
            if self.capture and self.scratch_at_dispatch is None:
                self.scratch_at_dispatch = bytes(buf[offset : offset + 16])
            if self.lazy and bytes(buf[offset : offset + 4]) == _SENTINEL:
                return
            # The residual probe leaves word 0 and paints word 1 with ``_poison_word(1, 0)``.
            # A stage that stores past the logical scratch only then is invisible to a safety
            # check that ran before the probe.
            probe_tail = self.probe_oob and bytes(buf[offset + 4 : offset + 8]) == _poison_word(1, 0)
            if self.lazy or not self.capture:
                payload = _SENTINEL + b"\x11" * 12 if self.lazy else b"\x22" * 16
                buf[offset : offset + 16] = payload
            if probe_tail:
                buf[offset + 16] = 0
                self.probe_oob_writes += 1

    def _apply_corrupt_inputs(self, stage_args) -> None:
        """@brief 普通 launch 改 x/w 并写满 scratch；覆盖探针只写满 scratch。"""
        scratch = None
        inputs = []
        for slot, arg in stage_args:
            buf = getattr(arg, "buf", None)
            if buf is None:
                continue
            if len(buf) == _SCRATCH_TOTAL:
                scratch = arg
            elif int(slot) in (0, 1):
                inputs.append(arg)
        if scratch is None:
            return
        offset = int(scratch.offset)
        payload = bytes(scratch.buf[offset : offset + 16])
        probe = _payload_is_coverage_probe(payload)
        scratch.buf[offset : offset + 16] = b"\x44" * 16
        if probe:
            self.probe_dispatches += 1
            for arg in inputs:
                buf = arg.buf
                at = int(arg.offset)
                key = id(buf)
                current = int(buf[at])
                if key not in self.bytes_at_probe:
                    self.bytes_at_probe[key] = current
                elif self.bytes_at_probe[key] != current:
                    self.probe_input_writes += 1
            return
        for arg in inputs:
            buf = arg.buf
            at = int(arg.offset)
            key = id(buf)
            if key not in self.mutated_inputs:
                self.mutated_inputs[key] = (buf, at, int(buf[at]))
            buf[at] = (int(buf[at]) + 1) & 0xFF
            self.timed_input_writes += 1

    def run_pipeline(
        self,
        kernels,
        grids,
        threads,
        tg,
        arg_sets,
        fills,
        iters=1,
        dispatch="serial",
        dispatch_recorder=None,
        dispatch_workspace=None,
    ):
        self.pipeline_calls += 1
        for i in range(int(iters)):
            group = arg_sets[i % len(arg_sets)]
            for index, stage_args in enumerate(group):
                self._apply(stage_args)
                if dispatch_recorder is not None:
                    workspace = () if dispatch_workspace is None else tuple(dispatch_workspace[index])
                    dispatch_recorder(index, kernels[index], grids[index], threads[index], int(tg[index]), workspace)
        return SimpleNamespace(wall_s=0.02 * int(iters), gpu_s=0.0, iters=int(iters))


def _dtype_codes() -> tuple[int, int]:
    import Metal

    return int(Metal.MTLDataTypeHalf), int(Metal.MTLDataTypeFloat)


def _coverage_runtime(ctx: _ScratchCtx, *, bind_inputs: bool = False) -> MetalV2Runtime:
    """@brief 4 个 fp32 的 poison scratch。假核可以整段重写，也可以看见哨兵后停手。"""
    contract, abi, limits = _conv2d()
    source = """\
/* taic-launch
math safe
scratch uwin fp32 4
launch k grid=1,1,1 threads=8,1,1 scratch=uwin:write
*/
"""
    geometry = msl_single_geometry_resolver(contract)(candidate_from_single_source(source))
    selection = select_geometry(geometry_pair(geometry, instance=contract.instance, contract=abi, limits=limits))
    half, flt = _dtype_codes()
    bindings = [
        Binding("y", 0, "buffer", half, "write", 2, True),
        Binding("uwin", 1, "buffer", flt, "write", 4, True),
    ]
    if bind_inputs:
        # Slots 0 and 1 are the read-only inputs.  The fake launch writes those only
        # when the scratch is not a coverage-probe paint.
        bindings = [
            Binding("x", 0, "buffer", half, "read", 2, True),
            Binding("w", 1, "buffer", half, "read", 2, True),
            Binding("y", 2, "buffer", half, "write", 2, True),
            Binding("uwin", 3, "buffer", flt, "write", 4, True),
        ]
    kernel = SimpleNamespace(
        name="k",
        bindings=bindings,
    )
    runtime = MetalV2Runtime(
        ctx,
        contract=contract,
        abi=abi,
        geometry=selection.geometry,
        kernels=[kernel],
        launch_spec=parse_launch_spec(source, launch_constants(contract.instance)),
        host_plan=None,
        host_inside_timing=False,
        limits=limits,
        selection=selection,
    )
    scratch = runtime.argset.workspace_bufs["uwin"]
    assert scratch.nbytes == 16
    assert len(scratch.buf) == _SCRATCH_TOTAL
    return runtime


def test_sentinel_skip_fails_full_coverage_on_both_paths(monkeypatch):
    """@brief 首轮写哨兵、之后跳过的 producer 过不了 correctness 和 benchmark 的整段复验。"""
    monkeypatch.setattr(
        "tcenv.backends.metal.v2_runtime._check_v2_case",
        lambda *args, **kwargs: SimpleNamespace(ok=True, safety=[]),
    )
    lazy = _coverage_runtime(_ScratchCtx(lazy=True))
    results, safety, nondet, _dispatch, _host = lazy.correctness(cases=("normal",), seed=1, determinism=False)
    assert nondet == []
    assert results[-1].ok is False
    assert any("scratch coverage" in item and "`uwin`" in item and "harness poison" in item for item in safety)

    measured = _coverage_runtime(_ScratchCtx(lazy=True)).benchmark(
        seed=1, reps=1, discipline=TimingDiscipline.fresh_rotation(2)
    )
    assert any("scratch coverage" in item and "`uwin`" in item and "harness poison" in item for item in measured.guard_violations)

    honest_results, honest_safety, honest_nondet, _, _ = _coverage_runtime(_ScratchCtx(lazy=False)).correctness(
        cases=("normal",), seed=1, determinism=False
    )
    assert honest_nondet == [] and honest_safety == [] and honest_results[-1].ok
    honest = _coverage_runtime(_ScratchCtx(lazy=False)).benchmark(
        seed=1, reps=1, discipline=TimingDiscipline.fresh_rotation(2)
    )
    assert honest.guard_violations == []


def test_a_guard_store_only_during_the_coverage_probe_is_unsafe(monkeypatch):
    """@brief 平时写满 16 字节，只在残留探针的第 1 个字出现时写穿尾保护带。

    ``_check_v2_case`` 的替身只让数值门通过，探针才会执行。``safety_violations`` 和
    ``guard_violations`` 不替换：违例文本必须来自探针写过的真实保护字节。
    """
    monkeypatch.setattr(
        "tcenv.backends.metal.v2_runtime._check_v2_case",
        lambda *args, **kwargs: SimpleNamespace(ok=True, safety=[]),
    )
    ctx = _ScratchCtx(probe_oob=True)
    runtime = _coverage_runtime(ctx)
    results, safety, nondet, _, _ = runtime.correctness(cases=("normal",), seed=1, determinism=False)
    scratch = runtime.argset.workspace_bufs["uwin"]
    assert scratch.buf[scratch.guard + scratch.nbytes] == 0
    assert ctx.probe_oob_writes >= 1
    assert nondet == []
    assert results[-1].ok is False
    assert any("buffer `uwin`" in item and "write past the end" in item for item in safety)
    assert not any("scratch coverage" in item for item in safety)

    bench_ctx = _ScratchCtx(probe_oob=True)
    measured = _coverage_runtime(bench_ctx).benchmark(
        seed=1, reps=1, discipline=TimingDiscipline.fresh_rotation(2)
    )
    assert bench_ctx.probe_oob_writes >= 1
    assert any("buffer `uwin`" in item and "write past the end" in item for item in measured.guard_violations)
    assert not any("scratch coverage" in item for item in measured.guard_violations)


def test_timed_launch_input_mutation_is_reported_when_the_coverage_probe_does_not_write():
    """@brief 普通计时 launch 改 x/w，覆盖探针把 scratch 写满但不碰这两个输入。

    探针前后的字节相同，所以只做那一次差分会漏掉。终检必须调用未被替换的
    ``safety_violations()``，对照 ``load_initial`` 的 ``_read_snapshot``。两个轮转组都要报。
    """
    ctx = _ScratchCtx(corrupt_inputs=True)
    runtime = _coverage_runtime(ctx, bind_inputs=True)
    measured = runtime.benchmark(seed=1, reps=1, discipline=TimingDiscipline.fresh_rotation(2))
    assert measured.rotation_sets == 2
    assert ctx.timed_input_writes >= 4
    assert ctx.probe_dispatches >= 6
    assert ctx.probe_input_writes == 0
    assert len(ctx.mutated_inputs) == 4
    assert len(ctx.bytes_at_probe) == 4
    for buf, at, original in ctx.mutated_inputs.values():
        assert int(buf[at]) != original
        assert int(buf[at]) == ctx.bytes_at_probe[id(buf)]
    assert measured.guard_violations.count("input `x` was modified by the kernel") == 2
    assert measured.guard_violations.count("input `w` was modified by the kernel") == 2
    assert not any("scratch coverage" in item for item in measured.guard_violations)
    assert not any("write past the end" in item for item in measured.guard_violations)


def test_host_written_scratch_is_committed_through_evaluate_task():
    """@brief bundle/evaluate_task 上，Host Program 返回 0 才写入 scratch 并 dispatch；返回 1 两者都不发生。"""
    found = next(item for item in discover_standard_tasks(REPO_ROOT / "tasks", code_root=REPO_ROOT) if item.op_name == "relu")
    locator = locator_for_instance(found, "fp16", "small")
    contract = locator.load()
    device = """\
#include <metal_stdlib>
using namespace metal;
kernel void relu_kernel(
    device const half* x [[buffer(0)]],
    device half* y [[buffer(1)]],
    device float* uwin [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  (void)x; (void)y; (void)uwin; (void)tid;
}
/* taic-launch
math safe
scratch uwin fp32 4 host
launch relu_kernel grid=1,1,1 threads=64,1,1 scratch=uwin:write
*/
"""
    geometry = {
        "schema_version": 1,
        "source": "candidate",
        "scratch": [{"name": "uwin", "dtype": "fp32", "size_terms": ["4"], "init_policy": "host_written"}],
        "stages": [
            {
                "kernel": "relu_kernel",
                "grid": [1, 1, 1],
                "threads": [64, 1, 1],
                "scratch": [{"name": "uwin", "access": "write"}],
            }
        ],
    }
    host_ok = """\
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)inputs; (void)shapes; (void)target; (void)resolved;
    workspaces->uwin[0] = 1.0f;
    workspaces->uwin[1] = 1.0f;
    workspaces->uwin[2] = 1.0f;
    workspaces->uwin[3] = 1.0f;
    workspaces->uwin_offset = 0u;
    workspaces->uwin_count = 4u;
    return 0;
}
"""

    def _once(host_source: str):
        candidate = CandidateArtifact(
            track_id="msl-bundle",
            entries=(
                CandidateEntry(path="device/relu.metal", role="device_kernel", media_type="text/plain", data=device.encode()),
                CandidateEntry(path="host/tiling.c", role="host_tiling", media_type="text/plain", data=host_source.encode()),
                CandidateEntry(
                    path="host/execution_geometry.json",
                    role="schedule",
                    media_type="application/json",
                    data=json.dumps(geometry).encode(),
                ),
            ),
        )

        class _Builder:
            def build(self, request):
                host = MetalHostProgram.build_for(
                    request.candidate,
                    instance=request.contract.instance,
                    tensors=request.abi.tensors,
                    target=load_target(request.target_id),
                    target_id=request.target_id,
                    native_digest=None,
                    inside_timing=True,
                    input_dependent=True,
                    preprocesses_inputs=True,
                    geometry=request.selection.geometry,
                )
                device_entry = next(entry for entry in request.candidate.entries if entry.role == "device_kernel")
                entries = tuple(
                    BundleEntry(
                        path=entry.path,
                        role="AUXILIARY",
                        media_type=entry.media_type,
                        size_bytes=len(entry.data),
                        sha256=entry.sha256,
                        data=bytes(entry.data),
                    )
                    for entry in request.candidate.entries
                )
                native = NativeExecutable(
                    backend_id="metal",
                    kind="compute-pipeline",
                    entry_path=device_entry.path,
                    entry_sha256=device_entry.sha256,
                    descriptor={"backend": "metal", "kernel_name": "relu_kernel", "host_program": host.document()},
                    handle=object(),
                )
                bundle = ArtifactBundle(entries=entries, native=native)
                provenance = BuildProvenance(
                    profile=_BUILD_PROFILE,
                    source_sha256=request.candidate.candidate_sha256,
                    artifact_set_digest=bundle.artifact_set_digest,
                )
                evidence = BuildEvidence(
                    mode="prebuilt",
                    status="ok",
                    provenance=provenance,
                    artifacts=tuple(BuildArtifact.from_entry(entry) for entry in bundle.entries),
                    placement="control_host",
                    native=native,
                )
                return BuildOutcome(bundle=bundle, provenance=provenance, evidence=evidence)

        class _Evaluator:
            def __init__(self) -> None:
                self.ctx = _ScratchCtx(capture=True)
                self.scratch_after = b""
                self.pipeline_calls = 0
                self.dispatch_count = 0
                self.message = ""

            def evaluate(self, request):
                half, flt = _dtype_codes()
                kernel = SimpleNamespace(
                    name="relu_kernel",
                    bindings=[
                        Binding("x", 0, "buffer", half, "read", 2, True),
                        Binding("y", 1, "buffer", half, "write", 2, True),
                        Binding("uwin", 2, "buffer", flt, "write", 4, True),
                    ],
                )
                executor = MetalV2Executor(request)
                runtime = executor.bind(self.ctx, [kernel])
                results, _safety, _nondet, dispatch, _host = runtime.correctness(
                    cases=("normal",), seed=1, determinism=False
                )
                self.pipeline_calls = self.ctx.pipeline_calls
                self.scratch_after = runtime.argset.workspace_bufs["uwin"].read()
                self.dispatch_count = len(dispatch)
                self.message = str(getattr(results[0], "message", ""))
                build = project_full(
                    request.build,
                    build_kind="native",
                    identity={"opspec_digest": "a" * 64, "lowering_profile_digest": "b" * 64},
                )
                report = EvalReport(
                    task=request.build_request.locator.op_name,
                    dtype=request.build_request.locator.dtype,
                    size=request.build_request.locator.size,
                    mode="full",
                    status="ok",
                    stage="bench",
                    seed=1,
                    build=build,
                    correctness={"passed": True},
                    protocol_version=PROTOCOL_VERSION,
                    target_id=DEFAULT_TARGET_ID,
                    backend="metal",
                    backend_version="1",
                )
                return report.to_json()

        def factory(resolved):
            return CandidateLifecycle(
                track_id="msl-bundle",
                geometry=resolved,
                host_steps=(
                    HostStep(
                        PrepareDeclaration(
                            name="host_tiling",
                            input_dependent=True,
                            preprocesses_inputs=True,
                            inside_timing=True,
                        ),
                        role="host_tiling",
                    ),
                ),
                candidate_sha256=candidate.candidate_sha256,
                target_id=DEFAULT_TARGET_ID,
                instance_key=contract.fingerprint(),
            )

        evaluator = _Evaluator()
        evaluate_task(
            locator,
            track_id="msl-bundle",
            candidate=candidate,
            builder=_Builder(),
            evaluator=evaluator,
            lifecycle_factory=factory,
        )
        return evaluator

    committed = _once(host_ok)
    assert committed.pipeline_calls == 1
    assert committed.dispatch_count == 1
    assert committed.ctx.scratch_at_dispatch == struct.pack("<4f", 1.0, 1.0, 1.0, 1.0)
    refused = _once(host_ok.replace("return 0;", "return 1;"))
    assert refused.pipeline_calls == 0
    assert refused.dispatch_count == 0
    assert refused.scratch_after == b"\xff" * 16
    assert "host program failed" in refused.message


def test_portfolio_host_written_scratch_passes_preflight_and_evaluate(tmp_path):
    """@brief portfolio 预检和随后的 geometry_pair 都看到 Host Plan，合法 host_written 能进入 evaluate。"""
    from tests.shared.test_shape_portfolio_runner import single_source_resolver

    root = tmp_path / "tasks"
    load_p1_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    contract = locator.load()
    instances = p1_instances(contract)
    target = load_target(DEFAULT_TARGET_ID)
    plan = normalize_semantic_domain_plan(contract.spec, instances, target=target)
    document = geometry_document("variant_a")
    document["scratch"] = [{"name": "uwin", "dtype": "fp32", "size_terms": ["4"], "init_policy": "host_written"}]
    document["stages"][0]["scratch"] = [{"name": "uwin", "access": "write"}]
    portfolio = parse_portfolio(
        parent_entries(
            portfolio_document(plan.digest),
            extra_sources={
                "host/geometry_a.json": ("host_tiling", "application/json", geometry_bytes(document)),
            },
        ),
        domain_plan=plan,
    )
    variant = next(item for item in portfolio.variants if item.variant_key == "variant-a-bundle")
    instance = next(item for item in instances if item.size == "threshold")
    hook = _preflight_hook(
        instance,
        target=target,
        max_buffer_length=None,
        portfolio=portfolio,
        authority=_authority(plan, contract, instances),
        resolver=single_source_resolver,
    )
    preflight = hook(variant.geometry, provider=variant.variant_key)
    assert preflight.status == "ok", preflight.detail
    assert any(spec.name == "uwin" and spec.init_policy == "host_written" for spec in preflight.resolved.scratch)

    class _Eval(RecordingEvaluator):
        def evaluate(self, request):
            report = super().evaluate(request)
            execution = self.last_execution
            if execution is not None:
                for item, stage in zip(execution["dispatch"], request.geometry.active_stages, strict=True):
                    item["workspace"] = [use.name for use in stage.workspace] + [
                        use.name for use in stage.scratch
                    ]
            return report

    evaluator = _Eval()
    result = run_portfolio_task(
        locator=locator,
        domain_plan=plan,
        portfolio=portfolio,
        request=_request(plan, contract, "threshold"),
        evidence=evidence_all_valid(portfolio),
        builder=RecordingBuilder(),
        evaluator=evaluator,
        geometry_resolver_factory=lambda _contract: single_source_resolver,
    )
    assert evaluator.requests
    assert result.result.report["status"] == "ok"
    assert any(
        spec["name"] == "uwin" and spec["init_policy"] == "host_written"
        for spec in result.evaluation["geometry"]["scratch"]
    )


def test_archived_p2h_p3_and_p4_geometries_replay_with_empty_scratch(tmp_path):
    """@brief 重放 P2-H、P4 和复制出来的性能归档几何。scratch=() 时 canonical 字节和身份投影与归档一致。

    P2-H 只保存几何文件的 sha256。这里用归档当时的候选字节重新解析，再比对冻结几何
    文件的哈希。性能归档的 ``geometry_identity`` 能由归档 dispatch 重新求值的，直接比对；
    符号网格要回到候选字节仍在树里的那份几何。P4 没有几何文档，重放只确认它没有
    长出 scratch 通道。``host_program_present`` 保持归档时的默认 False。
    """
    import hashlib
    import json

    from tcenv.opspec.v2 import instantiate_v2
    from tcenv.portfolio.candidate import parse_portfolio, project_provider
    from tcenv.portfolio.domain import DomainAuthority, normalize_semantic_domain_plan
    from tcenv.remote.execution_geometry import canonical_geometry_bytes
    from tcenv.taskcontract.portfolio_runner import _lifecycle_factory
    from tcenv.track import CanonicalBundle
    from tcenv.track.delivery import candidate_geometry
    from tests.backends.metal.test_shape_portfolio_performance_gpu import _fixture
    from tests.support.shape_portfolio_metal import (
        p1_instances,
        parent_entries,
        portfolio_document,
        write_p1_task,
    )

    epoch = REPO_ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收"
    target = load_target(DEFAULT_TARGET_ID)
    limits = abi_limits_from_target(target)

    def assert_empty(geometry, pair, data: bytes) -> None:
        assert geometry.scratch == ()
        assert "scratch" not in geometry.document()
        assert pair.candidate is not None and pair.candidate.scratch == ()
        assert "scratch" not in pair.candidate.document()
        parsed = json.loads(data)
        assert "scratch" not in parsed["geometry"]["candidate"]
        resolution = parsed["geometry"]["candidate_resolution"]
        if resolution is not None:
            assert "scratch" not in resolution

    def freeze(candidate, geometry, instance, contract, projection=None):
        abi = derive_tensor_memory_contract(instance, target=target)
        pair = geometry_pair(
            geometry,
            instance=instance,
            contract=abi,
            limits=limits,
            host_program_present=False,
        )
        resolved = select_geometry(pair).geometry
        lifecycle = None
        if projection is not None:
            factory = _lifecycle_factory(
                projection=projection, resolved=resolved, contract=contract, target_id=target.id
            )
            lifecycle = None if factory is None else factory(resolved)
        bundle = CanonicalBundle(candidate=candidate, pair=pair, lifecycle=lifecycle)
        data = canonical_geometry_bytes(bundle)
        assert_empty(geometry, pair, data)
        return pair.identity, hashlib.sha256(data).hexdigest()

    produced: set[tuple[str, str]] = set()
    live_identity: dict[tuple[str, str], str] = {}

    task_dir = write_p1_task(tmp_path / "p1")
    base = load_task_contract(task_dir)
    instances = p1_instances(base)
    plan = normalize_semantic_domain_plan(base.spec, instances, target=target)
    authority = DomainAuthority(
        plan=plan, spec=base.spec, target=target, instances=tuple(instances), contract=base
    )
    portfolio = parse_portfolio(parent_entries(portfolio_document(authority.plan.digest)), domain_plan=authority.plan)
    providers = list(portfolio.variants)
    if portfolio.fallback is not None:
        providers.append(portfolio.fallback)
    for provider in providers:
        projection = project_provider(portfolio, provider, authority=authority)
        candidate = projection.candidate
        for entry in candidate.entries:
            if str(entry.path).endswith("execution_geometry.json"):
                produced.add((str(entry.path), hashlib.sha256(bytes(entry.data)).hexdigest()))
                assert parse_candidate_geometry(json.loads(bytes(entry.data))).scratch == ()
        for instance in instances:
            sized = load_task_contract(task_dir, dtype=instance.dtype, size=instance.size)
            if candidate.track_id == "msl-single":
                geometry = msl_single_geometry_resolver(sized)(candidate)
            else:
                geometry = candidate_geometry(candidate)
            identity, digest = freeze(candidate, geometry, instance, sized, projection)
            produced.add(("execution_geometry.json", digest))
            live_identity[(candidate.candidate_sha256, str(instance.size))] = identity

    source = (REPO_ROOT / "kernels/apple-m4-max/softmax/fp16/paper/best.metal").read_text(encoding="utf-8")
    softmax = load_task_contract(REPO_ROOT / "tasks/softmax", dtype="fp16", size="paper")
    softmax_candidate = candidate_from_single_source(source, path="device/candidate.metal")
    softmax_geometry = msl_single_geometry_resolver(softmax)(softmax_candidate)
    _identity, digest = freeze(softmax_candidate, softmax_geometry, softmax.instance, softmax)
    produced.add(("execution_geometry.json", digest))

    archived: set[tuple[str, str]] = set()
    archived_p2h = REPO_ROOT / "tests/support/golden/shared/acceptance-epochs/fixtures/p2h"
    for path in archived_p2h.rglob("*.json"):
        def walk(value: object) -> None:
            if isinstance(value, dict):
                name = value.get("path")
                if name in {"execution_geometry.json", "host/execution_geometry.json"} and "sha256" in value:
                    archived.add((str(name), str(value["sha256"])))
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(json.loads(path.read_text(encoding="utf-8")))
    assert len(archived) == 9
    assert archived <= produced

    locator, gemm_plan, gemm_portfolio = _fixture(tmp_path / "gemm")
    gemm = locator.load()
    gemm_instances = tuple(instantiate_v2(gemm.spec, "fp16", size) for size in ("small", "paper"))
    gemm_authority = DomainAuthority(
        plan=gemm_plan, spec=gemm.spec, target=target, instances=gemm_instances, contract=gemm
    )
    for provider in gemm_portfolio.variants:
        projection = project_provider(gemm_portfolio, provider, authority=gemm_authority)
        for instance in gemm_instances:
            if not provider.variant_key.endswith(str(instance.size)):
                continue
            sized = load_task_contract(locator.task_dir, dtype="fp16", size=instance.size)
            geometry = msl_single_geometry_resolver(sized)(projection.candidate)
            identity, _digest = freeze(projection.candidate, geometry, instance, sized, projection)
            live_identity[(projection.candidate.candidate_sha256, str(instance.size))] = identity

    from_dispatch = 0
    from_source = 0
    checked = 0
    for path in sorted((REPO_ROOT / "tests/support/golden/shared/acceptance-epochs/fixtures/p3").rglob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        rows = document["measurements"] if isinstance(document.get("measurements"), list) else [document]
        for row in rows:
            if not isinstance(row, dict) or "geometry_identity" not in row or "dispatch" not in row:
                continue
            dispatch = row["dispatch"]
            if not isinstance(dispatch, list) or not dispatch:
                continue
            size = str(row["instance_key"]).split("/")[1]
            if size in {"small", "paper"}:
                instance = next(item for item in gemm_instances if item.size == size)
                shaped = load_task_contract(locator.task_dir, dtype="fp16", size=size)
            else:
                instance = next(item for item in instances if item.size == size)
                shaped = load_task_contract(task_dir, dtype="fp16", size=size)
            rebuilt = parse_candidate_geometry(
                {
                    "schema_version": 1,
                    "source": "candidate",
                    "stages": [
                        {
                            "kernel": str(stage["kernel"]),
                            "grid": list(stage["grid"]),
                            "threads": list(stage["threads"]),
                            "threadgroup_bytes": int(stage.get("threadgroup_bytes") or 0),
                            "workspace": list(stage.get("workspace") or []),
                        }
                        for stage in dispatch
                    ],
                }
            )
            identity, _digest = freeze(softmax_candidate, rebuilt, instance, shaped)
            checked += 1
            if identity == row["geometry_identity"]:
                from_dispatch += 1
                continue
            source_identity = live_identity.get((str(row["candidate_sha256"]), size))
            assert source_identity == row["geometry_identity"]
            from_source += 1
    assert checked >= 9
    assert from_dispatch >= 6
    assert from_source >= 2

    p4_geometry = 0
    for path in (epoch / "P4").rglob("*.json"):
        def walk_p4(value: object) -> None:
            nonlocal p4_geometry
            if isinstance(value, dict):
                if value.get("path") in {"execution_geometry.json", "host/execution_geometry.json"}:
                    p4_geometry += 1
                for child in value.values():
                    walk_p4(child)
            elif isinstance(value, list):
                for child in value:
                    walk_p4(child)

        walk_p4(json.loads(path.read_text(encoding="utf-8")))
    assert p4_geometry == 0
