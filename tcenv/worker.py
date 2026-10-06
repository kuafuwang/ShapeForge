"""Subprocess entry point: ``python -m tcenv.worker request.json``.

Runs one evaluation and writes the JSON report to the path in the request. Isolating each
candidate in its own process means a GPU hang, abort or segfault only loses that candidate.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def _install_heartbeat(path: str | None) -> None:
    """Publish "B<time>" before and "E<time>" after every GPU launch, and "W<time>" while waiting for the GPU lock
    (fixed width, overwritten in place)."""
    if not path:
        return
    import os
    import time

    from .gpulock import set_wait_hooks
    from .metal_runtime import set_launch_hooks

    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    set_launch_hooks(
        lambda: os.pwrite(fd, f"B{time.time():020.6f}\n".encode(), 0),
        lambda: os.pwrite(fd, f"E{time.time():020.6f}\n".encode(), 0),
    )
    set_wait_hooks(
        lambda: os.pwrite(fd, f"W{time.time():020.6f}\n".encode(), 0),
        lambda: os.pwrite(fd, f"E{time.time():020.6f}\n".encode(), 0),
    )


def main() -> int:
    req = json.loads(Path(sys.argv[1]).read_text())
    out = Path(req["out"])
    try:
        _install_heartbeat(req.get("heartbeat"))
        if req.get("stage"):
            from observe.progress import bind

            bind(req["stage"])
        from .evaluate import evaluate_in_process

        source = Path(req["kernel_path"]).read_text()
        rep = evaluate_in_process(
            req["task"],
            req["dtype"],
            req["size"],
            source,
            mode=req["mode"],
            seed=req["seed"],
            reps=req["reps"],
            cases=tuple(req["cases"]) if req.get("cases") else None,
        )
        out.write_text(json.dumps(rep.to_json()))
        return 0
    except Exception as e:  # noqa: BLE001
        from .evaluate import EvalReport

        rep = EvalReport(task=req["task"], dtype=req["dtype"], size=req["size"], mode=req["mode"], seed=req["seed"])
        rep.status = "crash"
        rep.message = f"harness error: {type(e).__name__}: {e}\n" + "".join(traceback.format_exc().splitlines(True)[-8:])
        out.write_text(json.dumps(rep.to_json()))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
