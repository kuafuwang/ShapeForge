"""Process launched *under* ``xctrace record``: loads an already-validated kernel and launches it repeatedly.

Only kernels that passed the full evaluation are ever profiled this way. Every launch is bounded and the
heartbeat protocol of the normal worker is kept, so the parent can kill a runaway process quickly.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def main() -> int:
    """@brief 在 ``xctrace record`` 下反复启动一个已通过完整评测的内核。

    从请求文件读取任务、内核与时长，按单核或 taic-launch 流水线两种方式循环启动，
    并把启动次数与 dispatch group 写回请求指定的输出文件。

    @return 成功时为 0；启动规格被拒绝时为 1。
    """
    req = json.loads(Path(sys.argv[1]).read_text())
    from tasks import get_task

    from .backends.metal.runtime import get_context
    from .backends.metal.sanitize import ArgSet
    from .backends.metal.testing import prepare_case
    from .backends.registry import load_backend
    from .worker import _install_heartbeat

    # xcworker is explicitly a Metal worker; resolve the backend instead of
    # relying on the removed implicit Metal fallback in _install_heartbeat.
    backend = load_backend("metal")
    _install_heartbeat(req.get("heartbeat"), backend)
    task = get_task(req["task"], req["dtype"], req["size"])
    ctx = get_context()
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
