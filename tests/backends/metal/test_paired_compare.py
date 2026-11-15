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
    """@brief outside-timing 的 host prepare 在计时前完成；inside 声明留在 run_once 里。"""
    from tcenv.backends.metal import paired_compare as metal_pairs

    class _Tensor:
        def __init__(self, name: str) -> None:
            self.name = name
            self.shape = (4,)

    class _Stage:
        def __init__(self) -> None:
            self.grid = (1, 1, 1)
            self.threads = (32, 1, 1)

    class _Runtime:
        def __init__(self, *, inside: bool) -> None:
            self.host_plan = object()
            self.host_inside_timing = inside
            self.geometry = SimpleNamespace(active_stages=(_Stage(),))
            self.bindings = ()
            self.abi = SimpleNamespace(tensors=(_Tensor("x"),))
            self.argset = SimpleNamespace(load_initial=lambda _inputs: None)
            self.events: list[tuple] = []

        def _stage_sets(self):
            return [], [(1, 1, 1)], [(32, 1, 1)], [0]

        def host_prepare(self):
            self.events.append(("prepare", len(events)))
            return {"prepared": True}

        def run_once(self, **kwargs):
            self.events.append(("run_once", kwargs.get("host_outside_done"), len(events)))
            return []

    events: list[str] = []

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
    assert outside.events[0] == ("prepare", 0)
    assert outside.events[1] == ("run_once", True, 1)

    events.clear()
    inside = _Runtime(inside=True)
    bound = metal_pairs._BoundV2(
        inside, task="relu", dtype="fp16", size="small", contract_digest="c" * 64, inputs=inputs,
    )
    bound.launch(1, grids=((1, 1, 1),), threads=((32, 1, 1),), bindings=(), shapes={"x": (4,)}, inputs=inputs)
    assert inside.events == [("run_once", False, 1)]
    assert events[0] == "clock"
