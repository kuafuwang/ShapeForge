"""Metal paired-comparison capability. CPU only: prepared objects stand in for two bound runtimes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from taic.paired_compare import PairedCompareError
from tcenv.backends.metal.paired_compare import bind_metal_runners, compare_for_backend, runners_from_prepared


def test_an_unimplemented_backend_does_not_call_runners():
    """@brief 非 Metal 后端不调用 runner。"""

    def boom(_k: int) -> float:
        raise AssertionError("runner must not be called")

    refused = compare_for_backend("qingwei", boom, boom, sha_a="ab" * 32, sha_b="cd" * 32)
    assert refused["supported"] is False
    assert refused["applied"] is False
    unbound = compare_for_backend("metal", None, None, sha_a="ab" * 32, sha_b="cd" * 32)
    assert unbound["supported"] is True and unbound["applied"] is False and unbound["gpu"] is False


def test_metal_compares_two_in_process_runners():
    """@brief 两颗进程内 runner 可以交替计时，同一颗核的区间包含 1。"""
    document = compare_for_backend(
        "metal", lambda k: 1e-4 * k, lambda k: 1e-4 * k, sha_a="ee" * 32, sha_b="ee" * 32, reps=4, seed=7,
    )
    assert document["applied"] is True
    assert document["backend"] == "metal"
    assert document["conclusion"] == "no_diff"
    assert document["interval"][0] <= 1.0 <= document["interval"][1]
    assert document["changes_seed"] is False
    assert document["elapsed_s"] > 0


class _Stage:
    def __init__(self, grid, threads):
        self.grid = grid
        self.threads = threads


class _Prepared:
    def __init__(self, *, name: str, grid, threads):
        self.name = name
        self.task = "relu"
        self.dtype = "fp16"
        self.size = "small"
        self.contract_digest = "c" * 64
        self.input_digest = "i" * 64
        self.geometry = SimpleNamespace(active_stages=(_Stage(grid, threads),))
        self.bindings = (SimpleNamespace(name="x", index=0, kind="buffer"),)
        self.shapes = {"x": (8, 16), "y": (8, 16)}
        self.inputs = {"x": bytes(range(32)), "y": b"\x00" * 16}
        self.seen: list[dict] = []

    def launch(self, k, *, grids, threads, bindings, shapes, inputs):
        self.seen.append({
            "k": k, "grids": grids, "threads": threads, "bindings": bindings, "shapes": shapes, "inputs": inputs,
        })
        return 0.001 * int(k)


def test_prepared_runners_keep_each_geometry_and_the_shared_contract_inputs():
    """@brief 假的 v2 对象带着形状、绑定和几何。两颗核不共用一个网格，输入是同一份。"""
    left = _Prepared(name="left", grid=(4, 1, 1), threads=(32, 1, 1))
    right = _Prepared(name="right", grid=(8, 2, 1), threads=(64, 1, 1))
    runner_a, runner_b = bind_metal_runners(None, None, prepared=(left, right))
    assert runner_a(3) == pytest.approx(0.003)
    assert runner_b(2) == pytest.approx(0.002)
    assert left.seen[0]["grids"] == ((4, 1, 1),)
    assert right.seen[0]["grids"] == ((8, 2, 1),)
    assert left.seen[0]["threads"] == ((32, 1, 1),)
    assert right.seen[0]["threads"] == ((64, 1, 1),)
    assert left.seen[0]["inputs"] == right.seen[0]["inputs"] == {"x": bytes(range(32)), "y": b"\x00" * 16}
    assert left.seen[0]["shapes"]["x"] == (8, 16)
    assert left.seen[0]["bindings"][0].name == "x"
    assert len(left.seen[0]["inputs"]["x"]) != 1024
    other = _Prepared(name="other", grid=(1, 1, 1), threads=(1, 1, 1))
    other.contract_digest = "d" * 64
    with pytest.raises(PairedCompareError, match="one task contract"):
        runners_from_prepared(left, other)
    bare = _Prepared(name="bare", grid=(1, 1, 1), threads=(1, 1, 1))
    bare.geometry = SimpleNamespace(active_stages=())
    with pytest.raises(PairedCompareError, match="execution geometry"):
        runners_from_prepared(left, bare)


def test_the_parent_uses_the_bounded_worker_instead_of_opening_a_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 没有注入的 runner 时，父进程把任务合同交给有界 worker，不创建 MetalContext。"""
    left = tmp_path / "left.metal"
    right = tmp_path / "right.metal"
    left.write_text("kernel void left() {}\n", encoding="utf-8")
    right.write_text("kernel void right() {}\n", encoding="utf-8")
    calls = {}

    def fake_run(payload, **kwargs):
        calls["payload"] = dict(payload)
        calls["kwargs"] = dict(kwargs)
        calls["module"] = kwargs.get("worker_module")
        return {"paired": {
            "applied": True, "supported": True, "backend": "metal", "gpu": True, "changes_seed": False,
            "elapsed_s": 0.01, "sha_a": "ab" * 32, "sha_b": "cd" * 32, "median_a_s": 1.0, "median_b_s": 1.0,
            "ratios": [1.0], "interval": [0.9, 1.1], "rounds": 2, "cv": 0.0, "conclusion": "no_diff",
        }}

    monkeypatch.setattr("tcenv.v2_worker.run_bounded_v2", fake_run)

    def refuse_device():
        raise AssertionError("the parent must not construct a MetalContext")

    monkeypatch.setattr("tcenv.backends.metal.runtime.MetalContext", refuse_device)
    document = compare_for_backend(
        "metal", path_a=left, path_b=right, sha_a="ab" * 32, sha_b="cd" * 32,
        task="relu", dtype="fp16", size="small", track_id="msl-single", seed=7,
    )
    assert calls["module"] == "tcenv.backends.metal.paired_compare"
    assert calls["payload"]["task"] == "relu"
    assert calls["kwargs"]["timeout"] == 420.0
    assert calls["kwargs"]["launch_deadline_s"] == 8.0
    assert calls["kwargs"]["worker_module"] == "tcenv.backends.metal.paired_compare"
    assert calls["payload"]["task"] == "relu"
    assert calls["payload"]["dtype"] == "fp16"
    assert calls["payload"]["size"] == "small"
    assert calls["payload"]["path_a"] == str(left)
    assert document["applied"] is True and document["changes_seed"] is False
    with pytest.raises(PairedCompareError, match="does not invent an ABI"):
        compare_for_backend("metal", path_a=left, path_b=right, sha_a="ab" * 32, sha_b="cd" * 32)


def test_a_bundle_with_host_tiling_assembles_through_evaluate_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief bundle 的 lifecycle 来自 evaluate_task，组装过程不创建设备。"""
    from tcenv.taskcontract import discover_standard_tasks
    from tcenv.track import REPO_ROOT
    from tests.shared.test_evaluate_task import _bundle_candidate, _geometry_document, _tracks
    from tests.support.v2_tasks import write_task
    from tcenv.backends.metal.paired_compare import assemble_candidate_request

    def refuse_device(*_args, **_kwargs):
        raise AssertionError("assembly must not construct a MetalContext")

    monkeypatch.setattr("tcenv.backends.metal.runtime.MetalContext", refuse_device)
    root = tmp_path / "tasks"
    write_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    candidate = _bundle_candidate(_geometry_document())

    class _NoDevice:
        def prepare(self, request):
            self.request = request

        def build(self, request):
            raise AssertionError("assembly must not compile")

    request = assemble_candidate_request(
        candidate=candidate,
        locator=locator,
        track_id="msl-bundle",
        track_registry_path=_tracks(tmp_path),
        builder=_NoDevice(),
    )
    lifecycle = request.canonical_bundle.lifecycle
    assert lifecycle is not None
    assert any(step.role == "host_tiling" for step in lifecycle.host_steps)
    assert request.build is None
    assert request.geometry.active_stages
    assert request.build_request.contract.op_name == locator.op_name


def test_the_worker_installs_the_v2_heartbeat_and_takes_the_machine_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 真实 worker 入口调用 _install_heartbeat，并在 exclusive gpu_lock 里计时。"""
    from contextlib import contextmanager

    from tcenv.backends.metal.paired_compare import _worker_main

    seen = {}

    def heartbeat(path, target_id):
        seen["heartbeat"] = (path, target_id)

    @contextmanager
    def lock(what, *, exclusive=True, **_kwargs):
        seen["lock"] = (what, exclusive)
        seen["inside"] = True
        yield 0.0
        seen["inside"] = False

    def context():
        seen["context"] = True
        return object()

    def assemble(*_args, **_kwargs):
        assert seen.get("inside") is True
        left = _Prepared(name="left", grid=(4, 1, 1), threads=(32, 1, 1))
        right = _Prepared(name="right", grid=(8, 2, 1), threads=(64, 1, 1))
        return left, right

    monkeypatch.setattr("tcenv.v2_worker._install_heartbeat", heartbeat)
    monkeypatch.setattr("tcenv.gpulock.gpu_lock", lock)
    monkeypatch.setattr("tcenv.backends.metal.runtime.get_context", context)
    monkeypatch.setattr("tcenv.backends.metal.paired_compare.assemble_v2_pair", assemble)
    heartbeat_path = tmp_path / "heartbeat"
    out = tmp_path / "out.json"
    envelope = _worker_main({
        "out": str(out),
        "heartbeat": str(heartbeat_path),
        "target_id": "apple-m4-max",
        "payload": {
            "task": "relu", "dtype": "fp16", "size": "small", "track_id": "msl-single",
            "path_a": "a.metal", "path_b": "b.metal", "sha_a": "ab" * 32, "sha_b": "cd" * 32,
            "seed": 7, "reps": 2,
        },
    })
    assert seen["heartbeat"] == (str(heartbeat_path), "apple-m4-max")
    assert seen["lock"][1] is True
    assert seen["context"] is True
    assert envelope["paired"]["applied"] is True
    assert not heartbeat_path.exists()


def test_the_cli_reaches_the_same_supervised_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 独立 CLI 与进程内比较都把同一套期限交给 v2 worker。"""
    from tcenv.cli import main

    seen = []

    def fake_run(payload, **kwargs):
        seen.append(dict(kwargs))
        return {"paired": {
            "applied": True, "supported": True, "backend": "metal", "gpu": True,
            "changes_seed": False, "elapsed_s": 0.01, "sha_a": "ab" * 32, "sha_b": "cd" * 32,
        }}

    monkeypatch.setattr("tcenv.v2_worker.run_bounded_v2", fake_run)
    left = tmp_path / "left.metal"
    right = tmp_path / "right.metal"
    left.write_text("kernel void left() {}\n", encoding="utf-8")
    right.write_text("kernel void right() {}\n", encoding="utf-8")
    compare_for_backend(
        "metal", path_a=left, path_b=right, sha_a="ab" * 32, sha_b="cd" * 32,
        task="relu", dtype="fp16", size="small",
    )
    assert main([
        "paired-compare", str(left), str(right), "--task", "relu", "--dtype", "fp16", "--size", "small",
        "--self-check-dir", str(tmp_path / ".selfcheck"),
    ]) == 0
    assert len(seen) == 2
    assert {item["worker_module"] for item in seen} == {"tcenv.backends.metal.paired_compare"}
    assert {item["timeout"] for item in seen} == {420.0}
    assert {item["launch_deadline_s"] for item in seen} == {8.0}


def test_outside_host_prepare_stays_outside_the_timed_loop(monkeypatch):
    """@brief outside-timing 的 host prepare 在计时前完成；inside 声明留在 run_once 里。

    outside 的评测边界仍是一块融合命令缓冲（评测的 ``_batched_run``），所以这里看到的是
    ``prepare`` 之后接一次 ``run_pipeline(iters=k)``，不是 k 次 ``run_once``；inside 声明
    的 Host Tiling 每次 launch 都是 CPU 工作，只能逐 launch。
    """
    from tcenv.backends.metal import paired_compare as metal_pairs
    from tcenv.backends.metal.runtime import RunTiming

    class _Tensor:
        def __init__(self, name: str) -> None:
            self.name = name
            self.shape = (4,)
            self.strides = (1,)
            self.dtype = "fp16"
            self.space = "global"
            self.access = "read"

    class _Stage:
        def __init__(self) -> None:
            self.grid = (1, 1, 1)
            self.threads = (32, 1, 1)

    class _ArgSet:
        def load_initial(self, _inputs) -> None:
            return None

        def apply_fill_plan(self, _plan, **_kwargs) -> None:
            return None

        def stage_args(self, _binding):
            return ()

        def workspace_fills_declared_zero(self, exclude):
            return ()

        def guard_violations(self):
            return []

        def safety_violations(self):
            return []

    class _Ctx:
        def run_pipeline(self, _kernels, _grids, _threads, _tg, _arg_sets, _fills, *, iters=1, dispatch="serial", **_kwargs):
            events.append(("run_pipeline", int(iters), dispatch))
            return RunTiming(wall_s=0.001, gpu_s=1e-6 * int(iters), iters=int(iters))

    class _HostPlan:
        def __init__(self, sink) -> None:
            self.sink = sink

        def execute(self, one):
            self.sink.append(("prepare", len(self.sink), one))
            return {"prepared": True}

    class _Runtime:
        def __init__(self, *, inside: bool) -> None:
            self.events: list[tuple] = []
            self.host_plan = _HostPlan(self.events)
            self.host_inside_timing = inside
            self.host_program = None
            self.batch_submission = "single_buffer"
            self.rotation_sets = 1
            self.ctx = _Ctx()
            self.geometry = SimpleNamespace(active_stages=(_Stage(),), workspace_uses=())
            self.bindings = ()
            self.abi = SimpleNamespace(tensors=(_Tensor("x"),))
            self.argset = _ArgSet()

        def new_argset(self):
            return _ArgSet()

        def _stage_sets(self):
            return [], [(1, 1, 1)], [(32, 1, 1)], [0]

        def run_once(self, **kwargs):
            self.events.append(("run_once", kwargs.get("host_outside_done"), len(self.events)))
            return []

    events: list[tuple | str] = []

    def clock():
        events.append("clock")
        return float(len(events))

    monkeypatch.setattr(metal_pairs.time, "perf_counter", clock)
    inputs = {"x": b"abcd"}

    outside = _Runtime(inside=False)
    bound = metal_pairs._BoundV2(
        outside, task="relu", dtype="fp16", size="small", contract_digest="c" * 64, inputs=inputs,
    )
    bound.launch(1, grids=((1, 1, 1),), threads=((32, 1, 1),), bindings=(), shapes={"x": (4,)}, inputs=inputs)
    # Host Tiling 在计时外：每个轮转组先 prepare，再是一块融合命令缓冲，不是逐 launch。
    assert len(outside.events) == 1
    kind, _, prepared = outside.events[0]
    assert kind == "prepare" and prepared is bound._arg_sets[0]
    assert [item for item in events if isinstance(item, tuple)] == [("run_pipeline", 1, "concurrent")]

    events.clear()
    inside = _Runtime(inside=True)
    bound = metal_pairs._BoundV2(
        inside, task="relu", dtype="fp16", size="small", contract_digest="c" * 64, inputs=inputs,
    )
    bound.launch(1, grids=((1, 1, 1),), threads=((32, 1, 1),), bindings=(), shapes={"x": (4,)}, inputs=inputs)
    assert inside.events == [("run_once", False, 0)]
    assert events[0] == "clock"


def test_outside_host_prepare_runs_on_every_rotation_set():
    """@brief outside Host Tiling 必须逐组 execute；只准备 runtime 自己那组会让其余组未初始化。"""
    runtime = _GpuRuntime(gpu_s=3e-6)
    plan = SimpleNamespace(executed=[], execute=lambda one: plan.executed.append(one))
    runtime.host_plan = plan
    bound = _bound(runtime)
    assert len(bound._arg_sets) == runtime.rotation_sets == 3
    _launch(bound, 4)
    assert plan.executed == bound._arg_sets
    assert len(runtime.ctx.calls) == 1
    # 评测每批计时前都会重做 ``_apply_bench_plans``；ab 每批也逐组重做，而不是只做一次。
    _launch(bound, 4)
    assert plan.executed == bound._arg_sets * 2


class _FusedCtx:
    """假的 MetalContext：融合路径一次 ``run_pipeline`` 编 k 轮。"""

    def __init__(self, gpu_s: float | None) -> None:
        self.gpu_s = gpu_s
        self.calls: list[dict] = []
        self.per_launch_calls: list[int] = []

    def _timing(self, iters: int):
        from tcenv.backends.metal.runtime import RunTiming

        gpu = 0.0 if self.gpu_s is None else float(self.gpu_s) * int(iters)
        return RunTiming(wall_s=float(int(iters)), gpu_s=gpu, iters=int(iters))

    def run_pipeline(self, _kernels, _grids, _threads, _tg, arg_sets, fills, *, iters=1, dispatch="serial", **_kwargs):
        self.calls.append({"iters": int(iters), "dispatch": dispatch, "sets": len(arg_sets), "fills": len(fills)})
        return self._timing(int(iters))

    def run_pipeline_per_launch(self, _kernels, _grids, _threads, _tg, arg_sets, fills, *, iters=1, **_kwargs):
        self.per_launch_calls.append(int(iters))
        return self._timing(int(iters))


class _FakeArgSet:
    """假的 V2ArgSet：只提供 launch 需要的那几个入口，并模拟 guard 被写穿。"""

    def __init__(self, stage_arg=()) -> None:
        self.bad = False
        self._stage_arg = tuple(stage_arg)

    def load_initial(self, _inputs) -> None:
        return None

    def apply_fill_plan(self, _plan, **_kwargs) -> None:
        return None

    def stage_args(self, _binding):
        return list(self._stage_arg)

    def workspace_fills_declared_zero(self, exclude):
        return ()

    def guard_violations(self):
        return ["write past the end of its allocation"] if self.bad else []

    def safety_violations(self):
        return ["buffer `y`: write past the end of its allocation"] if self.bad else []


class _GpuRuntime:
    """假的 v2 runtime：融合路径一次命令缓冲；Host 路径一次 run_once 一条 GPU 秒。"""

    def __init__(self, *, gpu_s: float | None):
        self.gpu_s = gpu_s
        self.grids = [(4, 1, 1)]
        self.threads = [(32, 1, 1)]
        self.argset = _FakeArgSet()
        self.argsets: list[_FakeArgSet] = []
        self.stage_arg: tuple = ()
        self.rotation_sets = 3
        self.ctx = _FusedCtx(gpu_s)
        self.host_plan = None
        self.host_inside_timing = False
        self.host_program = None
        self.batch_submission = "single_buffer"
        self._attempt_gpu: list[float] = []
        self._gpu_collect = False
        self.calls = 0
        self.geometry = SimpleNamespace(active_stages=(_Stage((4, 1, 1), (32, 1, 1)),), workspace_uses=())
        self.bindings = (SimpleNamespace(name="x", index=0, kind="buffer"),)
        self.abi = SimpleNamespace(tensors=(SimpleNamespace(
            name="x", shape=(8, 16), strides=(16, 1), dtype="fp16", space="global", access="read",
        ),))

    def new_argset(self):
        one = _FakeArgSet(self.stage_arg)
        self.argsets.append(one)
        return one

    def _stage_sets(self):
        return (("k",), self.grids, self.threads, (0,))

    def run_once(self, **_kwargs):
        self.calls += 1
        if self.gpu_s is not None and self._gpu_collect:
            self._attempt_gpu.append(float(self.gpu_s))
        return []


def _bound(runtime: _GpuRuntime):
    from tcenv.backends.metal.paired_compare import _BoundV2

    return _BoundV2(
        runtime, task="relu", dtype="fp16", size="small",
        contract_digest="c" * 64, inputs={"x": b"\x01\x02"},
    )


def _launch(bound, k: int) -> float:
    return bound.launch(
        k, grids=((4, 1, 1),), threads=((32, 1, 1),),
        bindings=(SimpleNamespace(name="x", index=0, kind="buffer"),),
        shapes={"x": (8, 16), "y": (8, 16)}, inputs={"x": b"\x01\x02"},
    )


def test_a_paired_launch_reuses_the_evaluators_fused_command_buffer():
    """@brief 非 Host 路径一块融合命令缓冲装 k 次 dispatch，和评测的 candidate_gpu_us_median 同一边界。

    旧实现对同一块缓冲串行调 k 次 ``run_once``（k 个命令缓冲）再求和；评测看的是一块
    融合缓冲的 ``(GPUEnd-GPUStart)/k``。同一份字节、同一个核，两者不是同一个物理量。
    """
    from tcenv.backends.metal.paired_compare import FUSED_SCOPE, GPU_TIMESTAMP

    runtime = _GpuRuntime(gpu_s=3e-6)
    bound = _bound(runtime)
    total = _launch(bound, 5)
    # 一次融合提交，不是 5 次逐 launch；参数组是评测的轮转组，不是单组。
    assert [call["iters"] for call in runtime.ctx.calls] == [5]
    assert runtime.ctx.calls[0]["dispatch"] == "concurrent"
    assert runtime.ctx.calls[0]["sets"] == runtime.rotation_sets == 3
    assert runtime.ctx.calls[0]["fills"] == 3
    assert runtime.calls == 0
    # 5 次 dispatch 的整批 GPU 时间 -> interleaved 除以 k 得到每迭代时间。
    assert total == pytest.approx(15e-6)
    assert bound.clock == GPU_TIMESTAMP
    assert bound.measurement_scope == FUSED_SCOPE


def test_a_fused_batch_without_gpu_samples_reports_wall_clock_and_says_so():
    """@brief 融合缓冲没有 GPU 时间戳时退回墙钟，并把这件事记在文档里，不能默默混用。"""
    from tcenv.backends.metal.paired_compare import FUSED_SCOPE, WALL_CLOCK

    runtime = _GpuRuntime(gpu_s=None)
    bound = _bound(runtime)
    total = _launch(bound, 3)
    assert total > 0.0
    assert bound.clock == WALL_CLOCK
    assert bound.measurement_scope == FUSED_SCOPE
    assert len(runtime.ctx.calls) == 1
    assert runtime.calls == 0


def test_a_host_program_batch_keeps_the_per_launch_boundary():
    """@brief 评测的 Host Program 路径逐 launch 记一条 GPU 样本，这里也逐 launch。"""
    from tcenv.backends.metal.paired_compare import GPU_TIMESTAMP, PER_LAUNCH_SCOPE

    runtime = _GpuRuntime(gpu_s=3e-6)
    runtime.host_program = SimpleNamespace(written_workspaces=frozenset())
    bound = _bound(runtime)
    total = _launch(bound, 4)
    assert runtime.ctx.calls == []
    assert runtime.calls == 4
    assert total == pytest.approx(12e-6)
    assert bound.clock == GPU_TIMESTAMP
    assert bound.measurement_scope == PER_LAUNCH_SCOPE


def test_host_tiling_inside_timing_keeps_the_per_launch_boundary():
    """@brief 计时内的 Host Tiling 每次 launch 都有 CPU 工作，不能编进一块融合缓冲。"""
    from tcenv.backends.metal.paired_compare import PER_LAUNCH_SCOPE

    runtime = _GpuRuntime(gpu_s=2e-6)
    runtime.host_plan = object()
    runtime.host_inside_timing = True
    bound = _bound(runtime)
    total = _launch(bound, 3)
    assert runtime.ctx.calls == []
    assert runtime.calls == 3
    assert total == pytest.approx(6e-6)
    assert bound.measurement_scope == PER_LAUNCH_SCOPE


def test_a_per_launch_submission_candidate_keeps_the_evaluators_busy_sum():
    """@brief 候选自己选 per_launch_set_event 时，评测读各缓冲自身时长之和，ab 也走同一条。"""
    from tcenv.backends.metal.paired_compare import PER_LAUNCH_SCOPE
    from tcenv.backends.metal.runtime import BufferArg

    runtime = _GpuRuntime(gpu_s=3e-6)
    runtime.batch_submission = "per_launch_set_event"
    from tcenv.backends.metal.runtime import BufferArg

    runtime.stage_arg = ((0, BufferArg(object())),)
    bound = _bound(runtime)
    total = _launch(bound, 3)
    assert runtime.ctx.calls == []
    assert runtime.ctx.per_launch_calls == [3]
    assert runtime.calls == 0
    assert total == pytest.approx(9e-6)
    assert bound.measurement_scope == PER_LAUNCH_SCOPE


def test_the_fused_batch_keeps_the_evaluators_rotation_and_barrier_boundary():
    """@brief 一批转过大轮转组，barrier 落在评测同一个 pass 边界，不是每轮都串行。"""
    from tcenv.backends.metal.runtime import pipeline_ops

    runtime = _GpuRuntime(gpu_s=3e-6)
    runtime.rotation_sets = 4
    bound = _bound(runtime)
    assert bound.rotation_sets == 4 and bound.rotation_reused is True
    _launch(bound, 9)
    assert runtime.ctx.calls[0]["iters"] == 9
    assert runtime.ctx.calls[0]["sets"] == 4
    # 同一组数下，评测的编码顺序每转完 4 次 dispatch 才插一个 rotate barrier。
    ops = pipeline_ops(iters=9, nsets=4, nstages=1, concurrent=True, filled=[False] * 4)
    rotates = [index for index, op in enumerate(ops) if op[0] == "rotate_barrier"]
    dispatched_before = [sum(1 for op in ops[:index] if op[0] == "dispatch") for index in rotates]
    assert dispatched_before == [4, 8]


def test_a_capacity_candidate_does_not_claim_the_evaluator_rotation():
    """@brief L2 容量题只准备了第一份输入：一组参数就必须明说不是评测轮转。"""
    from tcenv.backends.metal.paired_compare import SINGLE_ARGSET_SCOPE

    runtime = _GpuRuntime(gpu_s=3e-6)
    runtime.contract = SimpleNamespace(instance=SimpleNamespace(is_capacity=True))
    bound = _bound(runtime)
    assert bound.rotation_sets == 1 and bound.rotation_reused is False
    _launch(bound, 3)
    assert runtime.ctx.calls[0]["sets"] == 1
    assert bound.measurement_scope == SINGLE_ARGSET_SCOPE


def test_the_fused_batch_still_tracks_the_argsets_for_the_post_timing_check():
    """@brief 融合路径仍把实际用过的轮转组交给 guard / 只读输入终检，且检查本身不再启动。"""
    runtime = _GpuRuntime(gpu_s=3e-6)
    bound = _bound(runtime)
    _launch(bound, 2)
    assert bound.post_timing_safety() == []
    assert len(runtime.ctx.calls) == 1
    # 只检查这一批实际转过的组（k=2 -> 前两组），不是 runtime 自己那一组。
    assert runtime.argsets[0] in bound._used_argsets
    assert runtime.argsets[2] not in bound._used_argsets
    runtime.argsets[0].bad = True
    found = bound.post_timing_safety()
    assert any("write past the end" in item for item in found)
    # 终检只读，不再发射一次 kernel。
    assert len(runtime.ctx.calls) == 1


def _history(clocks, scope="fused_command_buffer_per_iteration"):
    """一颗只带逐批时钟历史的假 runner。"""

    def run_k(_k: int) -> float:
        return 1e-6

    run_k.clock_history = [("sample", clock, scope) for clock in clocks]
    return run_k


def test_a_document_is_wall_clock_when_the_two_sides_disagree():
    """@brief 一侧有 GPU 时间戳、另一侧没有时，比值只能标注成墙钟。

    只有 ``clock_of`` 没有逐批历史时，连“两侧都是 GPU”也不能自证：最后一批读数不是历史。
    """
    from tcenv.backends.metal.paired_compare import GPU_TIMESTAMP, WALL_CLOCK, _clock_of

    def gpu_side(_k: int) -> float:
        return 1e-6

    def wall_side(_k: int) -> float:
        return 1e-6

    gpu_side.clock_of = lambda: GPU_TIMESTAMP
    wall_side.clock_of = lambda: "wall_clock"
    assert _clock_of(gpu_side, wall_side) == WALL_CLOCK
    # 没有任何真实调用历史：不能把两次最后一次读数当成整次配对。
    assert _clock_of(gpu_side, gpu_side) == WALL_CLOCK


def _clock_sequence_runner(per_k_s: float, sample_clocks: list[str]):
    """旧式 runner：只有 ``clock_of``，每次真实调用之后才更新。"""
    from tcenv.bench import current_bench_call

    state = {"clock": "gpu_timestamp", "sample": 0}

    def run_k(k: int) -> float:
        call = current_bench_call()
        if call is not None and call.phase == "sample":
            index = state["sample"]
            state["clock"] = sample_clocks[min(index, len(sample_clocks) - 1)]
            state["sample"] = index + 1
        else:
            state["clock"] = "gpu_timestamp"
        return per_k_s * int(k)

    run_k.clock_of = lambda: state["clock"]  # type: ignore[attr-defined]
    run_k.state = state  # type: ignore[attr-defined]
    return run_k


def test_compare_runners_records_the_clock_of_each_call_not_the_last_one():
    """@brief 旧式 runner 的动态调用：中间退回墙钟、最后恢复 GPU 不能洗白整次配对。"""
    from tcenv.backends.metal.paired_compare import GPU_TIMESTAMP, WALL_CLOCK, compare_runners

    recovered = _clock_sequence_runner(3e-6, [WALL_CLOCK, GPU_TIMESTAMP, GPU_TIMESTAMP])
    steady = _clock_sequence_runner(3e-6, [GPU_TIMESTAMP] * 3)
    document = compare_runners(
        recovered, steady,
        sha_a="aa" * 32, sha_b="bb" * 32, seed=7, reps=3, warmup_reps=0, kmin=1,
    )
    # 最后一批真的是 GPU；旧实现只看这里就会写错。
    assert recovered.state["clock"] == GPU_TIMESTAMP
    assert document["timed_by"] == WALL_CLOCK

    missing_last = _clock_sequence_runner(3e-6, [GPU_TIMESTAMP, GPU_TIMESTAMP, WALL_CLOCK])
    again = compare_runners(
        missing_last, _clock_sequence_runner(3e-6, [GPU_TIMESTAMP] * 3),
        sha_a="cc" * 32, sha_b="dd" * 32, seed=7, reps=3, warmup_reps=0, kmin=1,
    )
    assert missing_last.state["clock"] == WALL_CLOCK
    assert again["timed_by"] == WALL_CLOCK

    quiet = compare_runners(
        _clock_sequence_runner(3e-6, [GPU_TIMESTAMP] * 3),
        _clock_sequence_runner(3e-6, [GPU_TIMESTAMP] * 3),
        sha_a="ee" * 32, sha_b="ff" * 32, seed=7, reps=3, warmup_reps=0, kmin=1,
    )
    assert quiet["timed_by"] == GPU_TIMESTAMP


def test_one_wall_clock_batch_keeps_the_whole_paired_document_wall_clock():
    """@brief 缺失 -> 恢复 -> 缺失等序列：任一批墙钟就够了，不用最后一批覆盖历史。"""
    from tcenv.backends.metal.paired_compare import (
        GPU_TIMESTAMP,
        MIXED_SCOPE,
        PER_LAUNCH_SCOPE,
        UNSPECIFIED_SCOPE,
        WALL_CLOCK,
        _clock_of,
        timing_identity,
    )

    gpu, wall = GPU_TIMESTAMP, WALL_CLOCK
    for sequence in (
        [wall, gpu],
        [gpu, wall],
        [wall, gpu, gpu],
        [gpu, gpu, wall],
        [gpu, wall, gpu],
        [wall, gpu, wall],
    ):
        assert _clock_of(_history(sequence), _history([gpu] * len(sequence))) == WALL_CLOCK, sequence
    assert _clock_of(_history([gpu, gpu]), _history([gpu, gpu])) == GPU_TIMESTAMP
    # 没有计入统计的批次不是 GPU 文档，也不是墙钟文档。
    empty = timing_identity({"a": _history([]), "b": _history([])})
    assert empty == {"timed_by": None, "measurement_scope": None}
    # 校准 / 预热批次不进统计，所以不能污染 timed_by。
    warm = _history([gpu])
    warm.clock_history.insert(0, ("warmup", wall, "fused_command_buffer_per_iteration"))
    assert _clock_of(warm, _history([gpu])) == GPU_TIMESTAMP
    # 两侧命令缓冲边界不同或根本没报告时，文档不能说它们是同一个量。
    mixed = timing_identity(
        {"a": _history([gpu]), "b": _history([gpu], scope=PER_LAUNCH_SCOPE)},
    )
    assert mixed == {"timed_by": GPU_TIMESTAMP, "measurement_scope": MIXED_SCOPE}
    unknown = timing_identity({"a": _history([gpu], scope=None), "b": _history([gpu])})
    assert unknown == {"timed_by": GPU_TIMESTAMP, "measurement_scope": UNSPECIFIED_SCOPE}


def test_the_production_call_discards_warmup_and_rotates_every_buffer_set():
    """@brief 真实设备路径传的是生产预热与批量下限，不是假 runner 测试用的 0/1。"""
    from taic.paired_compare import PRODUCTION_KMIN, PRODUCTION_WARMUP_REPS
    from tcenv.backends.metal.paired_compare import _production_call

    assert PRODUCTION_WARMUP_REPS > 0
    assert PRODUCTION_KMIN >= 4
    seen: list[int] = []

    def side(_k: int) -> float:
        seen.append(_k)
        return 1e-5

    reps = 4
    document = _production_call(side, side, sha_a="ee" * 32, sha_b="ee" * 32, seed=7, reps=reps)
    assert document["timed_by"] == "wall_clock"
    assert document["conclusion"] == "no_diff"
    # choose_iters 会先用 k=1 探路，这是它的探针，不是最终批量。被计入统计的批量必须
    # 不低于 kmin，否则一批只转不过一组缓冲区。
    chosen = max(seen)
    assert chosen >= PRODUCTION_KMIN
    # 两颗 runner × (计入轮数 + 丢弃的预热轮数)。
    assert seen.count(chosen) == 2 * (reps + PRODUCTION_WARMUP_REPS)
