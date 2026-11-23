"""posix_spawn worker creation: pipes, cwd, process groups and gRPC-after-init safety."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys

from tcenv.children import forget, popen_group_posix


def _reap(proc) -> None:
    try:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
    finally:
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
        forget(proc)


def test_posix_spawn_supports_pipes_cwd_and_a_new_session(tmp_path):
    """@brief 子进程读取 cwd、回传 stdout/stderr，并成为自己的进程组 leader。"""
    script = (
        "import os,sys; print(os.getcwd()); print(os.getpid(),os.getpgrp()); "
        "print('err-ok',file=sys.stderr)"
    )
    proc = popen_group_posix(
        [sys.executable, "-c", script], stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(tmp_path),
    )
    try:
        lines = proc.stdout.read().decode().splitlines()
        errors = proc.stderr.read().decode()
        assert proc.wait(timeout=5) == 0
        assert lines[0] == str(tmp_path)
        assert lines[1] == f"{proc.pid} {proc.pid}"
        assert errors.strip() == "err-ok"
    finally:
        _reap(proc)


def test_posix_spawn_after_grpc_init_does_not_inherit_the_poll_failure():
    """@brief gRPC 已启动后，新 worker 不再报 FD from fork parent still in poll list。"""
    import concurrent.futures
    import grpc

    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=1))
    server.start()
    proc = popen_group_posix(
        [sys.executable, "-c", "print('spawn-ok')"],
        stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        output = proc.stdout.read().decode().strip()
        errors = proc.stderr.read().decode()
        assert proc.wait(timeout=5) == 0
        assert output == "spawn-ok"
        assert "FD from fork parent still in poll list" not in errors
        assert errors == ""
    finally:
        _reap(proc)
        server.stop(0)


def test_posix_spawn_closes_grpc_and_other_inheritable_parent_fds():
    """@brief gRPC poll pipe 与无关 FD 都不得以同一 dev/inode 进入 worker。"""
    import concurrent.futures
    import grpc

    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=1))
    server.add_insecure_port("127.0.0.1:0")
    server.start()
    parent = {}
    for name in os.listdir("/dev/fd"):
        try:
            fd = int(name)
            if fd > 2 and os.get_inheritable(fd):
                info = os.fstat(fd)
                parent[fd] = (info.st_dev, info.st_ino)
        except OSError:
            pass
    code = (
        "import json,os,sys; parent={int(k):tuple(v) for k,v in json.loads(sys.argv[1]).items()}; "
        "found={}; "
        "exec('for fd, expected in parent.items():\\n"
        "    try:\\n"
        "        info=os.fstat(fd)\\n"
        "        actual=(info.st_dev,info.st_ino)\\n"
        "        if actual == expected: found[str(fd)]=actual\\n"
        "    except OSError: pass'); "
        "print(json.dumps(found))"
    )
    proc = popen_group_posix(
        [sys.executable, "-c", code, json.dumps(parent)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        proc.stdin.close()
        inherited = json.loads(proc.stdout.read().decode())
        assert proc.wait(timeout=5) == 0
        assert inherited == {}, (parent, inherited)
    finally:
        _reap(proc)
        server.stop(0)


def test_a_closed_parent_stdin_is_recreated_as_a_pipe_not_closed_again():
    """@brief pipe 分到 fd 0 时，DUP2(0,0) 后不得再 CLOSE(0)。"""
    outer = r'''
import os, sys
from tcenv.children import forget, popen_group_posix
os.close(0)
p = popen_group_posix(
    [sys.executable, "-c", "import sys; print(len(sys.stdin.buffer.read()))"],
    stdin=-1, stdout=-1, stderr=-1,
)
try:
    p.stdin.write(b"abc")
    p.stdin.close()
    print("OUT", p.stdout.read().decode().strip())
    print("ERR", p.stderr.read().decode().strip())
    print("RC", p.wait(timeout=5))
finally:
    for stream in (p.stdin, p.stdout, p.stderr):
        if stream is not None and not stream.closed:
            stream.close()
    if p.poll() is None:
        p.kill(); p.wait(timeout=5)
    forget(p)
'''
    completed = subprocess.run(
        [sys.executable, "-c", outer], text=True, capture_output=True, timeout=15, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "OUT 3" in completed.stdout and "RC 0" in completed.stdout
    assert "BrokenPipeError" not in completed.stdout + completed.stderr


def test_closed_parent_stdout_and_stderr_are_recreated_as_pipes(tmp_path):
    """@brief DUP2(child,1/2) 后不得因父端编号碰撞再关掉新 stdout/stderr。"""
    for target in (1, 2):
        outer = r'''
import os, sys
from tcenv.children import forget, popen_group_posix
saved = os.dup(int(sys.argv[1]))
os.close(int(sys.argv[1]))
child = "import os,sys; os.write(int(sys.argv[1]), b'ok')"
p = popen_group_posix(
    [sys.executable, "-c", child, sys.argv[1]],
    stdin=-1, stdout=-1, stderr=-1,
)
try:
    p.stdin.close()
    out = p.stdout.read() if sys.argv[1] == "1" else p.stderr.read()
    code = p.wait(timeout=5)
    os.write(saved, f"OUT {out.decode().strip()} RC {code}\n".encode())
finally:
    for stream in (p.stdin, p.stdout, p.stderr):
        if stream is not None and not stream.closed:
            stream.close()
    if p.poll() is None:
        p.kill(); p.wait(timeout=5)
    forget(p)
'''
        completed = subprocess.run(
            [sys.executable, "-c", outer, str(target)], text=True, capture_output=True, timeout=15, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "OUT ok RC 0" in completed.stdout + completed.stderr


def test_the_spawned_process_group_can_be_killed_and_reaped():
    """@brief 新 session 的 pgid 等于 pid，整组 SIGKILL 后可 wait 回收。"""
    proc = popen_group_posix(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        assert proc.poll() is None
        assert os.getpgid(proc.pid) == proc.pid
        os.killpg(proc.pid, signal.SIGKILL)
        assert proc.wait(timeout=5) != 0
    finally:
        _reap(proc)
