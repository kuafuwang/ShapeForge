"""Shared worker heartbeat hooks used by the current v2 xctrace worker.

There is no task-registry evaluation entry point here.  Bounded evaluation lives in
``tcenv.v2_worker``; the remote job bridge owns its own worker protocol.
"""

from __future__ import annotations


def _install_heartbeat(path: str | None, backend) -> None:
    """Install compile, launch and GPU-lock heartbeat writes for a resolved backend."""
    if not path:
        return
    import os
    import time

    from .gpulock import set_wait_hooks

    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    compile_hooks = (
        lambda: os.pwrite(fd, f"C{time.time():020.6f}\n".encode(), 0),
        lambda: os.pwrite(fd, f"E{time.time():020.6f}\n".encode(), 0),
    )
    launch_hooks = (
        lambda: os.pwrite(fd, f"B{time.time():020.6f}\n".encode(), 0),
        lambda: os.pwrite(fd, f"E{time.time():020.6f}\n".encode(), 0),
    )
    backend.set_heartbeat_hooks(on_compile=compile_hooks, on_launch=launch_hooks)
    set_wait_hooks(
        lambda: os.pwrite(fd, f"W{time.time():020.6f}\n".encode(), 0),
        lambda: os.pwrite(fd, f"E{time.time():020.6f}\n".encode(), 0),
    )
