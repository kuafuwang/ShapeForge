"""Machine-wide GPU lock (tcenv.gpulock) and the supervisor's handling of lock waits. No GPU work."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tcenv import gpulock

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def lock_file(tmp_path, monkeypatch):
    path = tmp_path / "gpu.lock"
    monkeypatch.setenv("TCENV_GPU_LOCK_FILE", str(path))
    monkeypatch.setattr(gpulock, "_wait_hooks", None)
    return path


def hold(mode: str, seconds: float) -> subprocess.Popen:
    """Another process that holds the lock (``mode`` "ex" or "sh") for ``seconds``; prints "held" once it has it."""
    code = textwrap.dedent(
        f"""
        import sys, time
        from tcenv.gpulock import gpu_lock
        with gpu_lock("test holder", exclusive={mode == 'ex'}):
            print("held", flush=True)
            time.sleep({seconds})
        """
    )
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, cwd=str(REPO))
    assert p.stdout.readline().strip() == "held"
    return p


def try_lock(exclusive: bool, wait: float = 0.3) -> bool:
    try:
        with gpulock.gpu_lock("probe", exclusive=exclusive, max_wait_s=wait):
            return True
    except gpulock.GpuLockTimeout:
        return False


def test_exclusive_excludes_everyone_and_shared_holders_coexist(lock_file):
    p = hold("ex", 30)
    try:
        assert "test holder" in gpulock.holder()
        assert not try_lock(exclusive=False) and not try_lock(exclusive=True)
    finally:
        p.kill(), p.wait()
    assert try_lock(exclusive=True)  # the kernel released the lock of the killed holder

    p = hold("sh", 30)
    try:
        assert try_lock(exclusive=False)
        assert not try_lock(exclusive=True)
    finally:
        p.kill(), p.wait()


def test_nested_requests_upgrade_and_return_to_shared(lock_file):
    with gpulock.gpu_lock("outer", exclusive=False):
        with gpulock.gpu_lock("inner shared", exclusive=False) as waited:
            assert waited == 0.0 and gpulock._mode == "sh"
        with gpulock.gpu_lock("timing"):
            assert gpulock._mode == "ex" and "timing" in gpulock.holder()
            with gpulock.gpu_lock("calibration inside timing"):
                assert gpulock._mode == "ex"
            p = subprocess.run([sys.executable, "-c", "from tcenv.gpulock import gpu_lock, GpuLockTimeout\ntry:\n    with gpu_lock('x', exclusive=False, max_wait_s=0.3): print('got')\nexcept GpuLockTimeout: print('busy')"], capture_output=True, text=True, cwd=str(REPO))
            assert p.stdout.strip() == "busy"
        assert gpulock._mode == "sh" and gpulock.holder() == ""
    assert gpulock._mode is None and gpulock._fd is None


def test_waiting_runs_the_hooks_and_gives_up_after_max_wait(lock_file):
    events = []
    gpulock.set_wait_hooks(lambda: events.append("wait"), lambda: events.append("acquired"))
    p = hold("ex", 1.0)
    try:
        with gpulock.gpu_lock("waiter") as waited:
            assert waited > 0.3
    finally:
        p.wait()
    assert events == ["wait", "acquired"]

    p = hold("ex", 30)
    try:
        t0 = time.monotonic()
        with pytest.raises(gpulock.GpuLockTimeout, match="test holder"):
            with gpulock.gpu_lock("impatient", max_wait_s=0.5):
                pass
        assert time.monotonic() - t0 < 5
    finally:
        p.kill(), p.wait()
    assert gpulock._fd is None


def test_time_spent_waiting_for_the_lock_does_not_count_toward_the_timeout(lock_file, monkeypatch):
    """The real worker, blocked on the lock held by another process for longer than the evaluation timeout."""
    from tcenv import evaluate as ev

    monkeypatch.setattr(ev, "POST_KILL_COOLDOWN_S", 0.0)
    kernel = lock_file.parent / "k.metal"
    kernel.write_text("kernel void relu_kernel() { while (true) {} }")  # refused by the static screen: nothing reaches the GPU
    p = hold("ex", 14.0)
    t0 = time.perf_counter()
    try:
        rep = ev._run_worker("relu", "fp16", "small", kernel, "quick", 1, 3, 8.0, validation=False, cases=None)
    finally:
        p.wait()
    assert rep.status == "rejected", rep.message
    assert time.perf_counter() - t0 > 12  # it really waited, beyond the 8 s timeout


def test_the_timeout_clock_resumes_after_the_wait(lock_file, monkeypatch):
    from tcenv import evaluate as ev

    monkeypatch.setattr(ev, "POST_KILL_COOLDOWN_S", 0.0)
    t0 = time.perf_counter()
    rep = ev._run_worker("relu", "fp16", "small", Path("/dev/null"), "quick", 1, 3, 2.0, validation=False, cases=None, worker_module="tests.support.fake_worker_lockwait")
    assert rep.status == "timeout" and "exceeded" in rep.message
    assert 4.0 < time.perf_counter() - t0 < 10.0  # 3 s waiting (not counted) + the 2 s timeout


def other_process_can_lock(exclusive: bool, wait: float = 0.3) -> bool:
    code = (
        "from tcenv.gpulock import gpu_lock, GpuLockTimeout\n"
        "try:\n"
        f"    with gpu_lock('probe', exclusive={exclusive}, max_wait_s={wait}): print('yes')\n"
        "except GpuLockTimeout: print('no')\n"
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO)).stdout.strip() == "yes"


def test_released_gives_up_the_hold_for_the_block_and_takes_it_back(lock_file):
    with gpulock.gpu_lock("outer", exclusive=False):
        assert not other_process_can_lock(exclusive=True)
        with gpulock.released():
            assert gpulock._mode is None
            assert other_process_can_lock(exclusive=True)  # a worker could take it exclusively now
        assert gpulock._mode == "sh"
        assert not other_process_can_lock(exclusive=True) and other_process_can_lock(exclusive=False)
    with gpulock.gpu_lock("timing", exclusive=True):
        assert not other_process_can_lock(exclusive=False)
        with gpulock.released():
            assert other_process_can_lock(exclusive=False)
        assert gpulock._mode == "ex" and not other_process_can_lock(exclusive=False)
    assert gpulock._mode is None and gpulock._fd is None
    assert other_process_can_lock(exclusive=True)


def test_a_process_holding_the_shared_lock_does_not_block_its_own_worker(lock_file, monkeypatch):
    """The pytest-level shared hold used to deadlock a worker that needs the lock exclusively."""
    from tcenv import evaluate as ev

    monkeypatch.setattr(ev, "POST_KILL_COOLDOWN_S", 0.0)
    with gpulock.gpu_lock("pytest-like parent", exclusive=False):
        rep = ev._run_worker("relu", "fp16", "small", Path("/dev/null"), "quick", 1, 3, 60.0, validation=False, cases=None, worker_module="tests.support.fake_worker_exclusive")
    assert rep.status == "ok", rep.message


def test_a_read_only_open_still_takes_the_exclusive_lock(lock_file):
    """A coding agent can flock the machine lock without write permission on the file."""
    lock_file.write_text("pid 1: outer\n")
    lock_file.chmod(0o444)
    gpulock._fd = None
    gpulock._mode = None
    gpulock._what = ""
    gpulock._writable = True
    with gpulock.gpu_lock("agent self-check", exclusive=True):
        assert gpulock._writable is False
        assert lock_file.read_text() == "pid 1: outer\n"
        assert not other_process_can_lock(exclusive=True)
    assert gpulock._fd is None
    lock_file.chmod(0o644)
