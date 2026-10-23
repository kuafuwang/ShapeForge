"""Compiler baseline: the task's Triton specification compiled to Metal by a community Triton backend (triton-msl).

The backend lives in an isolated virtual environment (``.venvs/triton-msl``) and is only ever used through a supervised
subprocess (``tcenv.triton_runner``); the main environment does not import Triton. A measurement stores the ratios
``triton / mlx`` and ``triton / torch_mps`` taken in one interleaved run. During a normal evaluation the main environment
multiplies the ratio with its own live MLX time, which keeps drift out of the comparison without running Triton there.

Important when interpreting results: triton-msl is a young alpha-stage backend, so "faster than triton-msl" is a much weaker
statement than the paper's "faster than NVIDIA Triton".
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PY = REPO_ROOT / ".venvs" / "triton-msl" / "bin" / "python"
CACHE_DIR = REPO_ROOT / "runs" / "triton_baselines"
LAUNCH_DEADLINE_S = 5.0  # one timed batch; a normal one takes milliseconds
COOLDOWN_S = 3.0


def available() -> bool:
    return VENV_PY.exists()


def metal_toolchain_ok() -> bool:
    """triton-msl compiles through Apple's Metal toolchain (``xcrun metallib``), an optional Xcode component."""
    r = subprocess.run(["xcrun", "-sdk", "macosx", "--find", "metallib"], capture_output=True, text=True)
    return r.returncode == 0


METAL_TOOLCHAIN_HINT = "the Metal toolchain is not installed (xcrun cannot find 'metallib'); install it once with:  xcodebuild -downloadComponent MetalToolchain"


def cache_path(task: str, dtype: str, size: str) -> Path:
    return CACHE_DIR / f"{task}_{dtype}_{size}.json"


def load_cached(task: str, dtype: str, size: str) -> dict | None:
    """Only trustworthy measurements are used: status ok, internally consistent, not launch-bound."""
    p = cache_path(task, dtype, size)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
    if d.get("status") != "ok" or d.get("consistent") is not True or d.get("warning"):
        return None
    try:  # a measurement of another spec / size / contract / case list is stale
        from tasks import get_task

        from ...triton_runner import spec_fingerprint

        if d.get("spec_fingerprint") != spec_fingerprint(get_task(task, dtype, size)):
            return None
    except Exception:  # noqa: BLE001 - cannot verify what was measured: do not use it
        return None
    return d


def estimate(cached: dict, live_us: dict[str, float]) -> dict | None:
    """Scale the measured ratio with the live vendor time of *this* evaluation."""
    for ref in ("mlx", "torch_mps"):
        if ref in cached.get("ratio_to", {}) and ref in live_us:
            return {
                "name": "triton_msl",
                "estimated_us": cached["ratio_to"][ref] * live_us[ref],
                "basis": f"ratio to {ref} ({cached['ratio_to'][ref]:.3f}) from an isolated interleaved run x this run's {ref} time",
                "best_num_warps": cached.get("best_num_warps"),
                "versions": cached.get("versions"),
            }
    return None


def busy_processes() -> list[str]:
    """Other running jobs (not stopped ones) that use GPU/CPU heavily; this backend's Python dispatch is sensitive to CPU contention."""
    import re

    r = subprocess.run(["ps", "-axo", "pid=,stat=,command="], capture_output=True, text=True)
    pat = re.compile(r"tcenv\.worker|tcenv\.xcworker|experiments\.(run_suite|repeatability|semantic)|taic\.cli|pytest")
    me = str(os.getpid())
    out = []
    for ln in r.stdout.splitlines():
        pid, stat, cmd = ln.strip().split(None, 2)
        if pid != me and not stat.startswith("T") and pat.search(cmd) and "ps -axo" not in cmd and not cmd.startswith(("/bin/zsh -c", "/bin/bash -c")):
            out.append(f"{pid} {cmd[:80]}")
    return out


def wait_quiet(max_wait_s: float, *, max_load: float = 6.0, poll_s: float = 10.0, log=print) -> dict:
    """Block until the machine is quiet (1-minute load average <= ``max_load`` and no job from ``busy_processes``), at most ``max_wait_s``.

    This backend dispatches through Python, so competing CPU work inflates its time much more than MLX's or torch's. Returns what was
    observed so that the measurement can record it."""
    t0 = time.time()
    announced = False
    while True:
        load, busy = os.getloadavg()[0], busy_processes()
        if load <= max_load and not busy:
            return {"waited_s": time.time() - t0, "quiet": True, "load_avg_1m": load}
        if time.time() - t0 >= max_wait_s:
            return {"waited_s": time.time() - t0, "quiet": False, "load_avg_1m": load, "busy": busy[:3]}
        if not announced:
            log(f"  waiting for a quiet machine (load {load:.1f} > {max_load}, busy: {busy[:1]}) for up to {max_wait_s:.0f}s")
            announced = True
        time.sleep(poll_s)


def measure(task: str, dtype: str, size: str, *, reps: int = 15, timeout: float = 900.0, log=print, wait_quiet_s: float = 0.0) -> dict:
    """Run the Triton spec of ``task`` through triton-msl (supervised) and cache the result. ``wait_quiet_s`` > 0: wait that long for a quiet machine first."""
    if not available():
        raise RuntimeError("no triton-msl environment: expected .venvs/triton-msl (create it with scripts/install_triton_msl.sh)")
    if not metal_toolchain_ok():
        raise RuntimeError(METAL_TOOLCHAIN_HINT)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    quiet = wait_quiet(wait_quiet_s, log=log) if wait_quiet_s > 0 else None
    busy = busy_processes()
    if busy:
        log(f"  WARNING: other jobs are running ({len(busy)}); the measurement will be repeated if it looks inconsistent, but run this on a quiet machine: {busy[0]}")
    with tempfile.TemporaryDirectory(prefix="tcenv_triton_") as td:
        tdp = Path(td)
        req = {"task": task, "dtype": dtype, "size": size, "out": str(tdp / "out.json"), "heartbeat": str(tdp / "hb"), "workdir": str(tdp / "spec"), "reps": reps}
        (tdp / "req.json").write_text(json.dumps(req))
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        killed = ""
        with open(tdp / "stderr.txt", "w") as ef, open(tdp / "stdout.txt", "w") as of:
            from ...children import forget, kill_group, popen_group

            proc = popen_group([str(VENV_PY), "-m", "tcenv.triton_runner", str(tdp / "req.json")], stdout=of, stderr=ef, env=env, cwd=str(REPO_ROOT))
            from ...evaluate import supervise

            from ...gpulock import released

            try:
                with released():
                    reason = supervise(proc, tdp / "hb", launch_deadline_s=LAUNCH_DEADLINE_S, timeout=timeout, poll_s=0.2)
            finally:
                kill_group(proc)
                forget(proc)
            if reason:
                killed = f"a timed batch of launches ran for more than {LAUNCH_DEADLINE_S:.0f}s" if reason == "launch" else f"exceeded {timeout:.0f}s"
                time.sleep(COOLDOWN_S)
        if killed:
            result = {"status": "timeout", "task": task, "dtype": dtype, "size": size, "error": killed}
        elif (tdp / "out.json").exists():
            result = json.loads((tdp / "out.json").read_text())
        else:
            tail = "\n".join((tdp / "stderr.txt").read_text(errors="replace").strip().splitlines()[-8:])
            result = {"status": "crash", "task": task, "dtype": dtype, "size": size, "error": f"runner exited with {proc.returncode}: {tail}"}
    result["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    result["busy_processes_at_start"] = busy
    if quiet is not None:
        result["quiet_wait"] = quiet
    cache_path(task, dtype, size).write_text(json.dumps(result, indent=1))
    log(f"  triton_msl {task} {dtype} {size}: {result['status']}" + (f" ratio_to={ {k: round(v, 3) for k, v in result.get('ratio_to', {}).items()} }" if result["status"] == "ok" else f" - {result.get('error', '')[:160]}"))
    return result
