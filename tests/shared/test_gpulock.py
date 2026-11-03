"""Machine-wide GPU lock (tcenv.gpulock) and the supervisor's handling of lock waits. No GPU work."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tcenv import gpulock
from tests.support.v1_tasks import V1_TASK

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def lock_file(tmp_path, monkeypatch):
    """@brief 把机器锁文件指到临时目录并复位等待钩子，避免用例之间互相影响。

    @param tmp_path 临时目录，锁文件放在这里。
    @param monkeypatch 用于设置 ``TCENV_GPU_LOCK_FILE`` 与复位模块状态。
    @return 锁文件路径。
    """
    path = tmp_path / "gpu.lock"
    monkeypatch.setenv("TCENV_GPU_LOCK_FILE", str(path))
    monkeypatch.setattr(gpulock, "_wait_hooks", None)
    return path


def hold(mode: str, seconds: float) -> subprocess.Popen:
    """@brief 另起一个进程持有锁（``mode`` 为 ``ex`` 或 ``sh``）``seconds`` 秒；拿到锁后先打印 "held"。

    @param mode ``ex`` 为独占，其他值为共享。
    @param seconds 持锁时长。
    @return 已确认拿到锁的子进程句柄。
    """
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
    """@brief 在当前进程里试锁，超时就算失败。

    @param exclusive 是否请求独占。
    @param wait 最长等待秒数。
    @return 拿到锁则为 True，抛 ``GpuLockTimeout`` 则为 False。
    """
    try:
        with gpulock.gpu_lock("probe", exclusive=exclusive, max_wait_s=wait):
            return True
    except gpulock.GpuLockTimeout:
        return False


def test_exclusive_excludes_everyone_and_shared_holders_coexist(lock_file):
    """@brief 验证独占锁排斥其他所有请求、共享锁之间可共存，且持有者被杀后锁能被立刻接手。

    @param lock_file 本用例使用的锁文件。
    """
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
    """@brief 验证同进程嵌套请求只会升级到独占、退出后回落到共享，且升级期间别的进程拿不到共享锁。

    @param lock_file 本用例使用的锁文件。
    """
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
    """@brief 验证等待时依次调用等待与获得钩子并返回真实等待时长；超过 ``max_wait_s`` 则以带持有者信息的超时异常放弃。

    @param lock_file 本用例使用的锁文件。
    """
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
    """@brief 跑真实 worker：被别的进程压住的时间超过评测超时，仍然不判成超时。

    被拒绝的是内核本身（静态筛选拦下死循环），所以结论是 rejected 而不是 timeout——等锁的时间没算进 8 秒超时。

    @param lock_file 本用例使用的锁文件。
    @param monkeypatch 把评测后的冷却时间置零，避免额外等待。
    """
    from tcenv import evaluate as ev

    monkeypatch.setattr(ev, "POST_KILL_COOLDOWN_S", 0.0)
    kernel = lock_file.parent / "k.metal"
    kernel.write_text("kernel void relu_kernel() { while (true) {} }")  # refused by the static screen: nothing reaches the GPU
    p = hold("ex", 14.0)
    t0 = time.perf_counter()
    try:
        rep = ev._run_worker(V1_TASK, "fp16", "small", kernel, "quick", 1, 3, 8.0, validation=False, cases=None)
    finally:
        p.wait()
    assert rep.status == "rejected", rep.message
    assert time.perf_counter() - t0 > 12  # it really waited, beyond the 8 s timeout


def test_the_timeout_clock_resumes_after_the_wait(lock_file, monkeypatch):
    """@brief 验证超时计时在等锁结束后才继续：3 秒等锁不计入，真正的 2 秒超时仍然触发。

    @param lock_file 本用例使用的锁文件。
    @param monkeypatch 把评测后的冷却时间置零。
    """
    from tcenv import evaluate as ev

    monkeypatch.setattr(ev, "POST_KILL_COOLDOWN_S", 0.0)
    t0 = time.perf_counter()
    rep = ev._run_worker(V1_TASK, "fp16", "small", Path("/dev/null"), "quick", 1, 3, 2.0, validation=False, cases=None, worker_module="tests.support.fake_worker_lockwait")
    assert rep.status == "timeout" and "exceeded" in rep.message
    assert 4.0 < time.perf_counter() - t0 < 10.0  # 3 s waiting (not counted) + the 2 s timeout


def other_process_can_lock(exclusive: bool, wait: float = 0.3) -> bool:
    """@brief 另起一个进程试锁，用它的结果判断锁是否真的被让出。

    @param exclusive 是否请求独占。
    @param wait 最长等待秒数。
    @return 那个子进程拿到锁则为 True。
    """
    code = (
        "from tcenv.gpulock import gpu_lock, GpuLockTimeout\n"
        "try:\n"
        f"    with gpu_lock('probe', exclusive={exclusive}, max_wait_s={wait}): print('yes')\n"
        "except GpuLockTimeout: print('no')\n"
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO)).stdout.strip() == "yes"


def test_released_gives_up_the_hold_for_the_block_and_takes_it_back(lock_file):
    """@brief 验证 ``released()`` 在块内让出锁、块结束后原样拿回，共享与独占两种持有都成立。

    @param lock_file 本用例使用的锁文件。
    """
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
    """@brief 父进程持有共享锁时，需要独占锁的 worker 不再死锁，能正常返回 ok。

    历史上 pytest 层的共享持有会把要独占的 worker 卡死，这里是那条回归的守护。

    @param lock_file 本用例使用的锁文件。
    @param monkeypatch 把评测后的冷却时间置零。
    """
    from tcenv import evaluate as ev

    monkeypatch.setattr(ev, "POST_KILL_COOLDOWN_S", 0.0)
    with gpulock.gpu_lock("pytest-like parent", exclusive=False):
        rep = ev._run_worker(V1_TASK, "fp16", "small", Path("/dev/null"), "quick", 1, 3, 60.0, validation=False, cases=None, worker_module="tests.support.fake_worker_exclusive")
    assert rep.status == "ok", rep.message


def test_a_v2_parent_holding_the_shared_lock_releases_it_for_its_worker(lock_file, monkeypatch):
    """@brief pytest 的共享 GPU 锁不能卡住 v2 worker 的独占锁。

    A4 的真实切片在 gpu 夹具里整段持有共享锁，再由 ``run_bounded_v2`` 拉起要独占锁的
    worker。父进程等待时必须放开自己的锁，否则两边互相等。

    @param lock_file 本用例使用的锁文件。
    @param monkeypatch 把评测后的冷却时间置零。
    """
    from tcenv import evaluate as ev
    from tcenv.v2_worker import run_bounded_v2

    monkeypatch.setattr(ev, "POST_KILL_COOLDOWN_S", 0.0)
    with gpulock.gpu_lock("pytest-like parent", exclusive=False):
        result = run_bounded_v2(
            {"target_id": "apple-m4-max"},
            timeout=20.0,
            launch_deadline_s=8.0,
            worker_module="tests.support.fake_v2_worker_exclusive",
        )
    assert result.get("report", {}).get("status") == "ok", result


def test_a_read_only_open_still_takes_the_exclusive_lock(lock_file):
    """@brief 锁文件只读时仍能取到独占锁，且不覆写锁文件内容。

    编码代理可能没有锁文件的写权限，``flock`` 在这种情况下必须照常工作。

    @param lock_file 本用例使用的锁文件，会先改成 0444。
    """
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
