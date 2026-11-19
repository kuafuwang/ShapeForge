"""Process launched *under* ``xctrace record``: loads an already-validated kernel and launches it repeatedly.

Only kernels that passed the full evaluation are ever profiled this way. Every launch is bounded and the
heartbeat protocol of the normal worker is kept, so the parent can kill a runaway process quickly.

Two composition roots live here, chosen by the task's ``spec_version``:

* v2 standard entry -- the live path. Assembly, compile and bind go through the same
  ``evaluate_task`` / track-builder / ``MetalV2Executor.bind`` sequence the evaluator and
  ``tcenv ab`` use, then the bound runtime is launched in a loop. The kernel names are
  written to ``out.json`` because a v2 task has no single ``contract.kernel_name``.
* v1 -- kept verbatim for the preserved v1 fixtures (``TAIC_BIND_TASK_ROUTE=specimen``).
  It is the only route that still builds through ``backend.artifact_builder``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _assemble_task(name: str, dtype: str, size: str):
    """@brief 取本 worker 要跑的任务对象。

    v1 装配在 B4 之后没有任何活题；``TAIC_BIND_TASK_ROUTE=specimen``（只由测试设置，并通过
    环境变量传给本子进程）改用归档的 v1 specimen。生产不设置该变量，仍走 ``tasks.get_task``。

    @param name 题名。
    @param dtype 元素类型。
    @param size 尺寸档位。
    @return ``tcenv.task.Task``。
    """
    import os

    if os.environ.get("TAIC_BIND_TASK_ROUTE") == "specimen":
        from tests.support.v1_tasks import v1_task

        return v1_task(name, dtype, size)
    from tasks import get_task

    return get_task(name, dtype, size)


def _main_v2(req: dict) -> int:
    """@brief v2 标准目录的录制路径：装配、绑定，然后把已绑定的 runtime 反复启动。

    装配不在这里发明：``assemble_candidate_request`` 就是评测器和 ``tcenv ab`` 用的那一段，
    编译交给 Track builder，绑定交给 ``MetalV2Executor.bind``。启动次数、dispatch group
    与**实际启动的核名**写回 ``out.json``；v2 没有单个 ``contract.kernel_name``，所以核名
    必须由这里给出，:func:`tcenv.backends.metal.xctrace.analyze` 优先读它。

    @param req 请求文档。
    @return 成功为 0。
    """
    from .backends.metal.paired_compare import assemble_candidate_request
    from .backends.metal.runtime import get_context
    from .backends.metal.track_adapter import MetalTrackBuilder, MetalV2Executor
    from .backends.metal.v2_runtime import prepare_v2_case
    from .backends.registry import load_backend
    from .remote.execution_geometry import attach_execution_geometry
    from .taskcontract.runner import CandidateEvaluationRequest
    from .worker import _install_heartbeat

    target_id = str(req.get("target_id") or "apple-m4-max")
    track_id = str(req.get("track_id") or "msl-single")
    backend = load_backend("metal")
    _install_heartbeat(req.get("heartbeat"), backend)
    ctx = get_context()
    # Composition root: inject the internal zero-fill kernel built through the factory. The v1
    # branch below does the same; without it any geometry that zero-fills a buffer aborts with
    # "the internal zero-fill kernel was not injected".
    prepare = getattr(backend, "prepare_context", None)
    if callable(prepare):
        prepare(ctx)

    candidate = Path(req["kernel_path"]).read_text()
    evaluation = assemble_candidate_request(
        candidate, task=req["task"], dtype=req["dtype"], size=req["size"],
        track_id=track_id, target_id=target_id,
    )
    compiled = MetalTrackBuilder(target_id=target_id).build(evaluation.build_request)
    attached = attach_execution_geometry(compiled, canonical_bundle=evaluation.canonical_bundle)
    built = CandidateEvaluationRequest(
        build=attached,
        build_request=evaluation.build_request,
        canonical_bundle=evaluation.canonical_bundle,
        geometry=evaluation.geometry,
    )
    handles = list(compiled.executable_handles())
    if not handles:
        raise RuntimeError("the track builder returned no executable handle")
    runtime = MetalV2Executor(built).bind(ctx, handles)
    contract = evaluation.build_request.contract
    prepared_case = prepare_v2_case(contract, contract.instance.case_domain.cases[0], 4242)
    initial = {name: bytes(data) for name, data in prepared_case.initial.items()}
    if not initial:
        raise RuntimeError("prepare_v2_case returned no inputs for this contract")
    runtime.argset.load_initial(initial)
    # outside-timing 的 host prepare 只做一次，之后每轮只走启动。
    host_outside_done = False
    if getattr(runtime, "host_plan", None) is not None and not getattr(runtime, "host_inside_timing", False):
        runtime.host_prepare()
        host_outside_done = True
    # 第一次启动是热机，它的 dispatch 不进入统计。
    runtime.run_once(benchmark=True, record_dispatch=False, host_outside_done=host_outside_done)
    batch = max(1, int(req.get("batch", 20)))
    n = 0
    t_end = time.time() + float(req.get("duration_s", 3.0))
    while time.time() < t_end:
        for _ in range(batch):
            runtime.run_once(benchmark=True, record_dispatch=False, host_outside_done=host_outside_done)
        n += batch
    stages = tuple(runtime.geometry.active_stages)
    Path(req["out"]).write_text(json.dumps({
        "launches": n,
        # 一轮迭代里连着启动几个 stage；analyze 用它把多个区间求和成一次迭代。
        "dispatch_group": len(stages),
        "kernels": [str(stage.kernel) for stage in stages],
    }))
    return 0



def main() -> int:
    """@brief 在 ``xctrace record`` 下反复启动一个已通过完整评测的内核。

    从请求文件读取任务、内核与时长，按单核或 taic-launch 流水线两种方式循环启动，
    并把启动次数与 dispatch group 写回请求指定的输出文件。

    @return 成功时为 0；启动规格被拒绝时为 1。
    """
    req = json.loads(Path(sys.argv[1]).read_text())
    from .backends.metal.xctrace import is_standard_v2_task

    if is_standard_v2_task(req["task"]):
        return _main_v2(req)
    task = _assemble_task(req["task"], req["dtype"], req["size"])

    from .backends.metal.runtime import get_context
    from .backends.metal.sanitize import ArgSet
    from .backends.metal.testing import prepare_case
    from .backends.registry import load_backend
    from .worker import _install_heartbeat

    # xcworker is explicitly a Metal worker; resolve the backend instead of
    # relying on the removed implicit Metal fallback in _install_heartbeat.
    backend = load_backend("metal")
    _install_heartbeat(req.get("heartbeat"), backend)
    ctx = get_context()
    # Composition root: inject the internal zero-fill kernel built through the factory.
    prepare = getattr(backend, "prepare_context", None)
    if callable(prepare):
        prepare(ctx)
    candidate = Path(req["kernel_path"]).read_text()
    from .backends.metal.launch import LaunchRejected, parse_launch_spec

    constants = {name: int(value) for name, _typ, value in task.contract.constants if isinstance(value, int) and not isinstance(value, bool)}
    try:
        spec = parse_launch_spec(candidate, constants)
    except LaunchRejected as e:
        print(f"taic-launch rejected: {e}", file=sys.stderr)
        return 1
    data = prepare_case(task, task.cases[0], 4242)
    # Profiling launches an already-validated kernel through the same registered
    # ArtifactBuilder the evaluator used, not through a second compile path.
    from .remote.artifact_builder import SourceBundle

    names = None if spec is None else list(dict.fromkeys(stage.kernel for stage in spec.stages))
    builder = backend.artifact_builder(
        kernel_contract=task.contract,
        specialization=task.specialization,
        math=("fast" if spec is None else spec.math),
        kernel_names=names,
    )
    outcome = builder.build(
        SourceBundle(candidate.encode("utf-8")), task_contract=task.contract,
    )
    compiled = list(outcome.executable_handles())
    if not compiled:
        raise RuntimeError("the backend builder returned no executable handle")
    if spec is None:
        kernel = compiled[0]
        argset = ArgSet(ctx, task.contract)
        argset.load_inputs(data.raw)
        argset.poison_outputs()
        args = [argset.args()]
        ctx.run(kernel, args, task.contract.grid, task.contract.threads)
        launch = lambda iters: ctx.run(kernel, args, task.contract.grid, task.contract.threads, iters=iters)
        group, kernels_launched = 1, [task.contract.kernel_name]
    else:
        by_name = {k.name: k for k in compiled}
        ordered = [by_name[stage.kernel] for stage in spec.stages]
        argset = ArgSet(ctx, task.contract, workspaces=spec.workspaces)
        argset.load_inputs(data.raw)
        argset.poison_outputs()
        stage_args = [[argset.stage_args(k) for k in ordered]]
        fills = [argset.workspace_fills()]
        grids = [stage.grid for stage in spec.stages]
        threads = [stage.threads for stage in spec.stages]
        tg = [stage.threadgroup_bytes for stage in spec.stages]

        def launch(iters, _ordered=ordered, _grids=grids, _threads=threads, _tg=tg, _args=stage_args, _fills=fills):
            return ctx.run_pipeline(_ordered, _grids, _threads, _tg, _args, _fills, iters=iters, dispatch="concurrent")

        launch(1)
        group = len(spec.stages) + (1 if spec.workspaces else 0)
        kernels_launched = names
    t_end = time.time() + float(req.get("duration_s", 3.0))
    n = 0
    batch = int(req.get("batch", 20))
    while time.time() < t_end:
        launch(batch)
        n += batch
    Path(req["out"]).write_text(json.dumps({"launches": n, "dispatch_group": group, "kernels": kernels_launched}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
