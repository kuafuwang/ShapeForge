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
    req = json.loads(Path(sys.argv[1]).read_text())
    from tasks import get_task

    from .metal_runtime import get_context
    from .sanitize import ArgSet
    from .testing import prepare_case
    from .worker import _install_heartbeat

    _install_heartbeat(req.get("heartbeat"))
    task = get_task(req["task"], req["dtype"], req["size"])
    ctx = get_context()
    src, _ = task.contract.assemble(Path(req["kernel_path"]).read_text())
    kernel = ctx.compile(src, task.contract.kernel_name)
    data = prepare_case(task, task.cases[0], 4242)
    argset = ArgSet(ctx, task.contract)
    argset.load_inputs(data.raw)
    argset.poison_outputs()
    args = [argset.args()]
    # one launch to warm up, then batches of launches for ~duration seconds
    ctx.run(kernel, args, task.contract.grid, task.contract.threads)
    t_end = time.time() + float(req.get("duration_s", 3.0))
    n = 0
    while time.time() < t_end:
        ctx.run(kernel, args, task.contract.grid, task.contract.threads, iters=int(req.get("batch", 20)))
        n += int(req.get("batch", 20))
    Path(req["out"]).write_text(json.dumps({"launches": n}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
