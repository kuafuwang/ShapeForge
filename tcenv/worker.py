"""Subprocess entry point: ``python -m tcenv.worker request.json``.

Runs one evaluation and writes the JSON report to the path in the request. Isolating each
candidate in its own process means a GPU hang, abort or segfault only loses that candidate.
"""
# 判卷流水线的子进程入口，由 evaluate._run_worker 用 `python -m tcenv.worker` 拉起。
# 本进程只写心跳并调用 evaluate_in_process；看见心跳、决定 SIGKILL 的是父进程。
# 等 GPU 锁时写 W。父进程把标着 W 的时间从超时里扣掉：排队时核还没跑，这段时间不属于这份候选。

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


# 心跳和监督的分工在这里：本进程只广播状态，不自己决定死亡。
# B/E 包住每一次 command buffer 的提交和完成，父进程用 B 的墙上时间判断单次启动是否该杀。
# W/E 包住等 GPU 锁。W 期间核没有在跑，父进程不计这段时间。
def _install_heartbeat(path: str | None) -> None:
    """Publish "B<time>" before and "E<time>" after every GPU launch, and "W<time>" while waiting for the GPU lock
    (fixed width, overwritten in place)."""
    if not path:
        return
    import os
    import time

    from .gpulock import set_wait_hooks
    from .metal_runtime import set_compile_hooks, set_launch_hooks

    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    # 定宽、从文件头原地覆盖。父进程按 22 字节来读，写一半也不会变成另一条更长的记录。

    set_compile_hooks(
        lambda: os.pwrite(fd, f"C{time.time():020.6f}\n".encode(), 0),
        lambda: os.pwrite(fd, f"E{time.time():020.6f}\n".encode(), 0),
    )
    set_launch_hooks(
        lambda: os.pwrite(fd, f"B{time.time():020.6f}\n".encode(), 0),
        lambda: os.pwrite(fd, f"E{time.time():020.6f}\n".encode(), 0),
    )
    set_wait_hooks(
        lambda: os.pwrite(fd, f"W{time.time():020.6f}\n".encode(), 0),
        lambda: os.pwrite(fd, f"E{time.time():020.6f}\n".encode(), 0),
    )


# 无论评测正常返回还是框架自己抛异常，都要留下 JSON。父进程把「没有报告」单独当成 crash。
def _refuse(raw: object, err: BaseException) -> None:
    """Leave an explicit refusal when the request itself cannot be read. Do not start an evaluation."""
    from .evaluate import EvalReport
    from .schema import write_json_atomic

    message = str(err)
    if not isinstance(raw, dict) or not raw.get("out"):
        sys.stderr.write(message + "\n")
        return
    rep = EvalReport(
        task=str(raw.get("task") or ""), dtype=str(raw.get("dtype") or ""), size=str(raw.get("size") or ""),
        mode=str(raw.get("mode") or ""), seed=int(raw["seed"]) if isinstance(raw.get("seed"), int) and not isinstance(raw.get("seed"), bool) else 0,
    )
    rep.status = "crash"
    rep.message = message
    write_json_atomic(Path(raw["out"]), rep.to_json())


def main() -> int:
    raw = json.loads(Path(sys.argv[1]).read_text())
    from .schema import SchemaError, read_request

    try:
        req = read_request(raw)
    except SchemaError as e:
        _refuse(raw, e)
        return 2
    out = Path(req["out"])
    try:
        _install_heartbeat(req.get("heartbeat"))
        if req.get("stage"):
            from tcenv.events import bind_progress

            bind_progress(req["stage"])
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
            fast_bench=bool(req.get("fast_bench")),
            require_contract_launch=bool(req.get("require_contract_launch")),
            target_id=str(req["target_id"]),
        )
        from .schema import write_json_atomic

        write_json_atomic(out, rep.to_json())
        return 0
    except Exception as e:  # noqa: BLE001
        # 框架异常也写成 status crash，并带上回溯的尾部。否则父进程只能看到进程退出、没有报告。
        from .evaluate import EvalReport
        from .schema import write_json_atomic

        rep = EvalReport(task=req["task"], dtype=req["dtype"], size=req["size"], mode=req["mode"], seed=req["seed"])
        rep.status = "crash"
        rep.message = f"harness error: {type(e).__name__}: {e}\n" + "".join(traceback.format_exc().splitlines(True)[-8:])
        write_json_atomic(out, rep.to_json())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
