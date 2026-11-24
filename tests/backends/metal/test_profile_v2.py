"""``tcenv profile`` on the v2 path. CPU only: the bound runtime is a stand-in.

Two things are asserted that a device test cannot check cheaply: that the
recording worker writes the *actually launched* kernel names (a v2 task has no
single ``contract.kernel_name``), and that the task's own quick evaluation stays
the only correctness gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


class _Stage:
    def __init__(self, kernel, grid=(4, 1, 1), threads=(32, 1, 1)):
        self.kernel = kernel
        self.grid = grid
        self.threads = threads
        self.threadgroup_bytes = 0


class _FakeRuntime:
    """记下每轮启动，并假装有一个 argset 和一个 geometry。"""

    def __init__(self, stages):
        self.geometry = SimpleNamespace(active_stages=tuple(stages))
        self.argset = SimpleNamespace(load_initial=lambda _inputs: None)
        self.host_plan = None
        self.host_inside_timing = False
        self.launches = 0
        self.prepared = False

    def host_prepare(self):
        self.prepared = True

    def run_once(self, **_kwargs):
        self.launches += 1
        return []


def _install_fakes(monkeypatch, *, stages, calls):
    import tcenv.xcworker as worker

    runtime = _FakeRuntime(stages)

    monkeypatch.setattr(
        "tcenv.backends.metal.paired_compare.assemble_candidate_request",
        lambda source, **kwargs: (calls.append(("assemble", source[:20])), _evaluation(kwargs))[1],
    )
    monkeypatch.setattr(
        "tcenv.backends.metal.track_adapter.MetalTrackBuilder",
        lambda **_kwargs: SimpleNamespace(
            build=lambda _request: SimpleNamespace(executable_handles=lambda: [object()])
        ),
    )
    monkeypatch.setattr(
        "tcenv.remote.execution_geometry.attach_execution_geometry",
        lambda compiled, **_kwargs: compiled,
    )
    monkeypatch.setattr(
        "tcenv.backends.metal.track_adapter.MetalV2Executor",
        lambda _built: SimpleNamespace(bind=lambda _ctx, _handles: runtime),
    )
    monkeypatch.setattr(
        "tcenv.backends.metal.v2_runtime.prepare_v2_case",
        lambda _contract, _case, _seed: SimpleNamespace(initial={"x": b"\x01\x02"}),
    )
    monkeypatch.setattr("tcenv.backends.metal.runtime.get_context", lambda: object())
    monkeypatch.setattr("tcenv.backends.registry.load_backend", lambda _name: SimpleNamespace(
        prepare_context=lambda _ctx: None,
    ))
    monkeypatch.setattr("tcenv.worker._install_heartbeat", lambda *_a, **_k: None)
    return runtime


def _evaluation(kwargs):
    contract = SimpleNamespace(
        instance=SimpleNamespace(case_domain=SimpleNamespace(cases=["canonical"])),
        op_name=kwargs.get("task", ""), dtype=kwargs.get("dtype", ""), size=kwargs.get("size", ""),
    )
    return SimpleNamespace(
        build_request=SimpleNamespace(contract=contract),
        canonical_bundle=SimpleNamespace(),
        geometry=SimpleNamespace(active_stages=()),
    )


def _request(tmp_path: Path, *, duration_s: float = 0.0, batch: int = 5) -> dict:
    kernel = tmp_path / "kernel.metal"
    kernel.write_text("kernel void two_stage() {}\n", encoding="utf-8")
    return {
        "task": "conv2d", "dtype": "fp16", "size": "small",
        "kernel_path": str(kernel), "out": str(tmp_path / "out.json"),
        "heartbeat": str(tmp_path / "heartbeat"),
        "duration_s": duration_s, "batch": batch,
        "track_id": "msl-single", "target_id": "apple-m4-max",
    }


def test_the_v2_recording_writes_the_kernels_it_actually_launched(monkeypatch, tmp_path):
    """@brief v2 题没有 contract.kernel_name，录制者必须把真实核名写进 out.json。"""
    import tcenv.xcworker as worker

    calls: list[tuple[str, str]] = []
    runtime = _install_fakes(
        monkeypatch,
        stages=[_Stage("stage_a"), _Stage("stage_b"), _Stage("stage_c")],
        calls=calls,
    )
    monkeypatch.setattr(worker, "is_standard_v2_task", lambda _task: True, raising=False)

    code = worker._main_v2(_request(tmp_path, duration_s=0.0, batch=5))
    assert code == 0
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["kernels"] == ["stage_a", "stage_b", "stage_c"]
    # 一轮迭代连着启动三个 stage；analyze 靠这个数把区间求和成一次迭代。
    assert out["dispatch_group"] == 3
    # 热机一次 + 至少一批。
    assert runtime.launches >= 1 + out["launches"]
    assert calls and calls[0][0] == "assemble"


def test_the_v2_recording_reuses_the_one_host_prepare(monkeypatch, tmp_path):
    """@brief host prepare 在计时之外只做一次，之后每轮只付启动。"""
    import tcenv.xcworker as worker

    runtime = _install_fakes(monkeypatch, stages=[_Stage("only")], calls=[])
    runtime.host_plan = object()          # 有 Host Program 才需要 outside-timing 的准备
    worker._main_v2(_request(tmp_path, duration_s=0.0, batch=3))
    assert runtime.prepared is True


def test_main_dispatches_a_v2_task_to_the_v2_route(monkeypatch, tmp_path):
    """@brief 主入口按 spec_version 分流，不去碰会 fail closed 的 v1 装配。"""
    import tcenv.xcworker as worker

    seen: list[str] = []
    monkeypatch.setattr(
        "tcenv.backends.metal.xctrace.is_standard_v2_task", lambda task: task == "conv2d",
    )
    monkeypatch.setattr(worker, "_main_v2", lambda req: (seen.append(req["task"]), 0)[1])

    request = _request(tmp_path)
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["xcworker", str(request_file)])
    assert worker.main() == 0
    assert seen == ["conv2d"]


def test_a_delivery_rejection_becomes_a_refusal_not_a_traceback(monkeypatch):
    """@brief 交付形态不合规是候选自己的事：quick 门报 rejected，不是抛异常。"""
    from tcenv.cli_common import _v2_quick_verdict
    from tcenv.contract_errors import TaskContractError

    def refuse(*_args, **_kwargs):
        raise TaskContractError("candidate: the msl-single source has no 'taic-launch' block")

    monkeypatch.setattr("taic.task_binding.bind_task", lambda *a, **k: SimpleNamespace(
        locator=object(), track_id="msl-single", is_v2=True))
    monkeypatch.setattr("tcenv.backends.metal.search_candidate.evaluate_candidate", refuse)
    monkeypatch.setattr("tcenv.target.load_target", lambda _id: object())

    verdict = _v2_quick_verdict("conv2d", "fp16", "small", "kernel void k() {}\n", target_id="apple-m4-max")
    assert verdict["status"] == "rejected"
    assert "taic-launch" in verdict["message"]


def test_ab_maps_a_delivery_rejection_to_a_rejected_row(monkeypatch):
    """@brief 同一份拒绝在 ab 里只让那一份文件变成 rejected，不让整次排序崩掉。"""
    from tcenv.backends.metal.ab_bench import _quick_verdict
    from tcenv.contract_errors import TaskContractError

    def refuse(*_args, **_kwargs):
        raise TaskContractError("candidate: the msl-single source has no 'taic-launch' block")

    monkeypatch.setattr("taic.task_binding.bind_task", lambda *a, **k: SimpleNamespace(
        locator=object(), track_id="msl-single", is_v2=True))
    monkeypatch.setattr("tcenv.backends.metal.search_candidate.evaluate_candidate", refuse)
    monkeypatch.setattr("tcenv.target.load_target", lambda _id: object())

    verdict = _quick_verdict("kernel void k() {}\n", task="conv2d", dtype="fp16", size="small",
                             target_id="apple-m4-max", seed=7)
    assert verdict["status"] == "rejected"
    assert verdict["ok"] is False


def test_the_kernel_name_lookup_uses_the_current_v2_discovery():
    """Actual candidate kernel names come from the recording, not a task-name registry."""
    from tcenv.backends.metal.xctrace import is_standard_v2_task

    assert is_standard_v2_task("conv2d") is True
    assert is_standard_v2_task("definitely-not-a-task") is False


def test_profile_refuses_before_recording_when_the_v2_gate_fails(monkeypatch, tmp_path, capsys):
    """@brief 门在录制之前：quick 不过就不启动 xctrace。"""
    from tcenv.cli import main

    kernel = tmp_path / "k.metal"
    kernel.write_text("kernel void k() {}\n", encoding="utf-8")
    recorded: list[object] = []
    monkeypatch.setattr(
        "tcenv.backends.metal.xctrace.profile_kernel",
        lambda *a, **k: recorded.append((a, k)),
    )
    monkeypatch.setattr(
        "tcenv.cli_common._v2_quick_verdict",
        lambda *_a, **_k: {"status": "rejected", "message": "no 'taic-launch' block"},
    )
    code = main(["profile", "conv2d", str(kernel), "--dtype", "fp16", "--size", "small"])
    assert code == 1
    assert recorded == []
    assert "refusing to profile" in capsys.readouterr().out
