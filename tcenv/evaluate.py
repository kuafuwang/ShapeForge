"""End-to-end evaluation of one candidate kernel (the TCEnv "execute, measure, select" loop).

Stages (Figure 2, steps 2-4 of the paper):
  compile  -> assemble into the ABI template, compile with the Metal compiler, create the pipeline
  abi      -> reflection check against the contract
  safety   -> guard bands, NaN-poisoned outputs, input immutability, determinism
  correct  -> randomized adversarial differential tests vs a CPU reference
  bench    -> interleaved benchmark against MLX / PyTorch-MPS baselines, roofline
  validate -> (separate subprocess) the same launch under Metal shader validation

``evaluate()`` drives workers in subprocesses so that a GPU hang or crash never takes the
orchestrator down. ``evaluate_in_process()`` is what the workers call.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

STATUS_OK = "ok"
NOISY_CV = 0.05  # quiet measurements here have a per-repetition CV of 0.5-1.5%; above this something else was using the machine
BENCH_ATTEMPTS = 3  # noisy benchmarks are repeated; the quietest attempt is kept
OWN_RUNNERS = ("candidate", "candidate_serial")


def _note_stage(**fields) -> None:
    try:
        from observe.progress import update

        update(**fields)
    except Exception:
        return


@dataclass
class EvalReport:
    task: str
    dtype: str
    size: str
    mode: str
    status: str = "pending"  # ok | rejected | compile_error | abi_error | unsafe | incorrect | runtime_error | timeout | crash
    stage: str = "start"  # last stage reached
    message: str = ""
    seed: int = 0
    compile: dict = field(default_factory=dict)
    abi: dict = field(default_factory=dict)
    pipeline: dict = field(default_factory=dict)
    safety: dict = field(default_factory=dict)
    correctness: dict = field(default_factory=dict)
    benchmark: dict = field(default_factory=dict)
    roofline: dict = field(default_factory=dict)
    speedup: float | None = None  # best baseline time / candidate time (>1 means faster than the baseline)
    speedups: dict = field(default_factory=dict)
    candidate_us: float | None = None
    baseline_us: dict = field(default_factory=dict)
    compiler_baseline: dict = field(default_factory=dict)  # optional triton-msl comparison (see tcenv.triton_baseline)
    verification: dict = field(default_factory=dict)  # formal equivalence check (tcenv.verify); always 'unsupported' for now
    elapsed_s: float = 0.0

    @property
    def eligible(self) -> bool:
        """Safe and numerically correct (the first two eligibility conditions of the paper)."""
        return self.status == STATUS_OK

    def to_json(self) -> dict:
        d = asdict(self)
        d["eligible"] = self.eligible
        return d

    @staticmethod
    def from_json(d: dict) -> "EvalReport":
        d = dict(d)
        d.pop("eligible", None)
        known = {f.name for f in fields(EvalReport)}
        return EvalReport(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# in-process evaluation (runs inside the worker)
# ---------------------------------------------------------------------------


def evaluate_in_process(
    task_name: str,
    dtype: str,
    size: str,
    kernel_source: str,
    *,
    mode: str = "full",  # "validate" | "quick" | "full"
    seed: int | None = None,
    reps: int = 15,
    cases: tuple[str, ...] | None = None,
) -> EvalReport:
    """Untimed stages hold the machine-wide GPU lock shared, the benchmark holds it exclusively (tcenv.gpulock)."""
    from .gpulock import gpu_lock

    with gpu_lock(f"{mode} evaluation of {task_name} {dtype} {size}", exclusive=False):
        return _evaluate_in_process(task_name, dtype, size, kernel_source, mode=mode, seed=seed, reps=reps, cases=cases)


def _evaluate_in_process(
    task_name: str,
    dtype: str,
    size: str,
    kernel_source: str,
    *,
    mode: str,
    seed: int | None,
    reps: int,
    cases: tuple[str, ...] | None,
) -> EvalReport:
    import torch  # noqa: F401  (imported early so failures are reported as crashes of the worker)

    from tasks import get_task

    from . import abi as abi_mod
    from .baselines import BaselineRunner, build_runners, rotation_count
    from .bench import interleaved
    from .gpulock import gpu_lock
    from .metal_runtime import BufferArg, MetalCompileError, MetalRuntimeError, get_context
    from .profile import get_calibration, pipeline_stats, roofline
    from .sanitize import ArgSet
    from .testing import prepare_case, check_case, run_correctness

    t_start = time.perf_counter()
    task = get_task(task_name, dtype, size)
    seed = int(seed if seed is not None else secrets.randbits(48))
    rep = EvalReport(task=task_name, dtype=dtype, size=size, mode=mode, seed=seed)
    from .verify import verify

    rep.verification = verify(task, kernel_source).to_json()  # formal equivalence: unsupported for MSL (tcenv.verify)
    contract = task.contract
    ctx = get_context()
    dev_info = ctx.info()

    def finish(status: str, message: str = "") -> EvalReport:
        rep.status = status
        if message:
            rep.message = message
        rep.elapsed_s = time.perf_counter() - t_start
        _note_stage(stage=rep.stage, status=status, task=task_name, dtype=dtype, size=size, mode=mode)
        return rep

    # -- static screen (before anything touches the GPU) ---------------------------
    from .screen import screen_source

    rep.stage = "screen"
    _note_stage(stage="screen", status="running", task=task_name, dtype=dtype, size=size, mode=mode)
    refused = screen_source(kernel_source)
    if refused:
        rep.abi = {"ok": False, "violations": refused}
        return finish("rejected", "refused before compilation: " + refused[0])

    # -- compile -------------------------------------------------------------
    rep.stage = "compile"
    _note_stage(stage="compile", status="running", task=task_name, dtype=dtype, size=size, mode=mode)
    full_src, header_lines = contract.assemble(kernel_source)
    try:
        kernel = ctx.compile(full_src, contract.kernel_name)
    except MetalCompileError as e:
        msg = contract.rewrite_compiler_lines(str(e), header_lines)
        rep.compile = {"ok": False, "errors": msg}
        return finish("compile_error", "Metal compiler rejected the kernel")
    rep.compile = {
        "ok": True,
        "compile_s": kernel.compile_s,
        "pipeline_s": kernel.pipeline_s,
        "language_version": kernel.language_version,
    }

    # -- abi -------------------------------------------------------------------
    rep.stage = "abi"
    _note_stage(stage="abi", status="running", task=task_name, dtype=dtype, size=size, mode=mode)
    problems = abi_mod.check_abi(contract, kernel, max_tg_memory=dev_info["max_threadgroup_memory_bytes"])
    rep.abi = {"ok": not problems, "violations": problems}
    rep.pipeline = pipeline_stats(kernel, contract, dev_info)
    if problems:
        return finish("abi_error", "kernel does not match the required ABI / launch contract")

    # -- safety + correctness ------------------------------------------------------
    rep.stage = "correctness"
    _note_stage(stage="correctness", status="running", task=task_name, dtype=dtype, size=size, mode=mode)
    tight = mode == "validate"
    argset = ArgSet(ctx, contract, tight=tight)

    baseline_cache: dict[str, Any] = {}

    def baseline_fn(data):
        if mode == "validate":
            return None
        outs: dict[str, dict] = {}
        for impl in task.baselines:
            try:
                r = BaselineRunner(impl, task, data.rounded, rotate=1)
                outs[impl.name] = r.outputs()
            except Exception as e:  # noqa: BLE001
                baseline_cache.setdefault("errors", []).append(f"{impl.name}: {type(e).__name__}: {e}")
        return outs or None

    use_cases = cases or (task.cases[:1] if mode == "validate" else task.cases)
    try:
        results, safety, nondet = run_correctness(
            ctx, kernel, task, argset, seed, cases=use_cases, baseline_fns=baseline_fn, determinism=(mode != "validate")
        )
    except MetalRuntimeError as e:
        rep.safety = {"ok": False, "violations": [f"GPU execution error: {e}"], "nondeterminism": [], "validation": []}
        return finish("runtime_error", f"the GPU reported an error while running the kernel: {str(e)[:300]}")

    rep.correctness = {
        "ok": all(r.ok for r in results),
        "cases": [r.to_json() for r in results],
        "tolerance": {"rtol": task.rtol, "atol_rel": task.atol_rel, "exact": task.exact},
        "baseline_errors": baseline_cache.get("errors", []),
    }
    rep.safety = {"ok": not safety and not nondet, "violations": safety, "nondeterminism": nondet, "validation": []}
    if safety or nondet:
        return finish("unsafe", (safety + nondet)[0])
    if not rep.correctness["ok"]:
        first_bad = next(r for r in results if not r.ok)
        return finish("incorrect", f"case '{first_bad.case}': {first_bad.message}")
    if mode in ("validate", "quick"):
        return finish(STATUS_OK, "compiled, ABI ok, safe and correct")

    # -- benchmark ---------------------------------------------------------------
    rep.stage = "bench"
    _note_stage(stage="bench", status="running", task=task_name, dtype=dtype, size=size, mode=mode, attempt=1, attempts_max=BENCH_ATTEMPTS)
    data0 = prepare_case(task, task.cases[0], seed + 1)
    R = rotation_count(task)
    sets = []
    for _ in range(R):
        s = ArgSet(ctx, contract)
        s.load_inputs(data0.raw)
        s.poison_outputs()
        sets.append(s)
    arg_sets = [s.args() for s in sets]
    gpu_samples: list[float] = []
    launched: set[int] = set()  # buffer sets that were actually used (a batch smaller than R leaves the rest untouched)

    # The candidate is dispatched the way MLX dispatches the independent calls of a graph (concurrent encoder, buffer barrier between
    # passes over the rotating sets), otherwise the tail of every launch would be serialized against the next one and the candidate
    # would be penalized by up to ~40% (experiments/dispatch_ab.py). The strictly serial time is reported alongside, for information.
    def cand_run(k: int) -> float:
        tm = ctx.run(kernel, arg_sets, contract.grid, contract.threads, iters=k, dispatch="concurrent")
        gpu_samples.append(tm.gpu_per_iter_s)
        launched.update(range(min(k, len(arg_sets))))
        return tm.wall_s

    def cand_serial_run(k: int) -> float:
        return ctx.run(kernel, arg_sets, contract.grid, contract.threads, iters=k, dispatch="serial").wall_s

    attempts: list[tuple[float, dict, list[float]]] = []  # (noise, stats, candidate GPU samples) per attempt
    with gpu_lock(f"benchmark of {task_name} {dtype} {size}") as lock_wait_s:
        calib = get_calibration()
        runners: dict[str, Any] = {"candidate": cand_run, "candidate_serial": cand_serial_run}
        base_runners = build_runners(task, data0.rounded)
        for n, r in base_runners.items():
            runners[n] = r.run_k
        try:
            for attempt in range(BENCH_ATTEMPTS):
                _note_stage(stage="bench", status="running", attempt=attempt + 1, attempts_max=BENCH_ATTEMPTS, noisy=False)
                first = len(gpu_samples)
                st = interleaved(runners, reps=reps, seed=seed + attempt, kmin=R)
                attempts.append((max(s.cv for n, s in st.items() if n != "candidate_serial"), st, gpu_samples[first:]))
                noisy = attempts[-1][0] > NOISY_CV
                _note_stage(stage="bench", status="running", attempt=attempt + 1, attempts_max=BENCH_ATTEMPTS, noisy=noisy)
                if not noisy:
                    break
        except MetalRuntimeError as e:
            return finish("runtime_error", f"GPU error during benchmarking: {str(e)[:300]}")
    noise, stats, cand_gpu = min(attempts, key=lambda a: a[0])

    # spot-check the outputs that were produced *during* the timed runs
    bad_after: list[str] = []
    checked = sorted(launched)
    for i, s in enumerate(sets):
        if i not in launched:
            continue
        viol = s.safety_violations()
        if viol:
            bad_after.extend(f"[benchmark set {i}] {v}" for v in viol)
        if i in (checked[0], checked[-1]):
            res = check_case(task, data0, s.read_outputs(), None)
            if not res.ok:
                bad_after.append(f"[benchmark set {i}] output wrong during timed runs: {res.message}")
    if bad_after:
        rep.safety = {"ok": False, "violations": bad_after, "nondeterminism": [], "validation": []}
        return finish("unsafe", bad_after[0])

    cand = stats["candidate"]
    rep.candidate_us = cand.median_us
    rep.benchmark = {
        "candidate": cand.to_json(),
        "candidate_gpu_us_median": (sorted(cand_gpu)[len(cand_gpu) // 2] * 1e6) if cand_gpu else None,
        "baselines": {n: stats[n].to_json() for n in stats if n not in OWN_RUNNERS},
        "candidate_serial_us": stats["candidate_serial"].median_us,
        "dispatch": "concurrent (as MLX); candidate_serial_us is the strictly serial time of the same kernel",
        "rotation_sets": R,
        "noise_cv": noise,  # largest per-repetition CV among the runners of the kept attempt
        "attempts": len(attempts),
        "noisy": noise > NOISY_CV,
        "gpu_lock_wait_s": lock_wait_s,
        "protocol": f"interleaved, {reps} reps, median per-iteration wall time of back-to-back launches over {R} rotating buffer sets, concurrent dispatch for the candidate as for MLX",
    }
    rep.baseline_us = {n: stats[n].median_us for n in stats if n not in OWN_RUNNERS}
    rep.speedups = {n: stats[n].median_s / cand.median_s for n in stats if n not in OWN_RUNNERS}
    if rep.speedups:
        best = min(rep.baseline_us, key=lambda n: rep.baseline_us[n])
        rep.speedup = rep.speedups[best]
        rep.benchmark["best_baseline"] = best
    rep.roofline = roofline(task, cand.median_s, calib)
    try:  # compiler baseline (Triton -> Metal through triton-msl), when it was measured for this task
        from .triton_baseline import estimate, load_cached

        cached = load_cached(task_name, dtype, size)
        est = estimate(cached, rep.baseline_us) if cached else None
        if est:
            est["speedup"] = est["estimated_us"] / cand.median_us
            rep.compiler_baseline = est
    except Exception:  # noqa: BLE001 - optional information must never fail an evaluation
        pass
    rep.stage = "done"
    return finish(STATUS_OK, "compiled, ABI ok, safe, correct and benchmarked")


# ---------------------------------------------------------------------------
# subprocess driver
# ---------------------------------------------------------------------------


LAUNCH_DEADLINE_S = 2.0  # a single GPU launch of a legitimate candidate takes milliseconds
POST_KILL_COOLDOWN_S = 3.0  # give the GPU time to recover after a runaway kernel was killed


def _read_heartbeat(path: Path) -> tuple[str, float] | None:
    try:
        raw = path.read_bytes()[:22].decode()
        return raw[0], float(raw[1:21])
    except Exception:  # noqa: BLE001 - file not created yet / partially written
        return None


def _sync_progress(stage_path: Path, hb_path: Path, dest: Path, seen: list) -> None:
    """Copy the worker's stage file and the heartbeat letter into ``dest`` when either changes.

    The heartbeat file itself stays a fixed-width ``B`` / ``E`` / ``W`` plus a timestamp. Age is computed by the
    dashboard from ``heartbeat_at``, so this does not rewrite the snapshot on every poll.
    """
    try:
        hb = _read_heartbeat(hb_path)
        mtime = stage_path.stat().st_mtime_ns if stage_path.exists() else 0
        sig = (mtime, hb)
        if sig == seen[0]:
            return
        seen[0] = sig
        doc: dict = {}
        if stage_path.exists():
            doc = json.loads(stage_path.read_text())
        if hb:
            doc["heartbeat"] = hb[0]
            doc["heartbeat_at"] = hb[1]
        if not doc:
            return
        from observe.publish import publish

        publish(dest, doc)
    except Exception:
        return


def supervise(
    proc: subprocess.Popen,
    hb_path: Path,
    *,
    launch_deadline_s: float,
    timeout: float,
    poll_s: float = 0.1,
    stage_path: Path | None = None,
    progress_path: Path | None = None,
) -> str:
    """Wait for ``proc``; SIGKILL it when one GPU launch has been in flight (heartbeat "B") for longer than
    ``launch_deadline_s`` ("launch") or the run took longer than ``timeout`` ("timeout"). Time spent waiting for the
    GPU lock (heartbeat "W", see tcenv.gpulock) does not count toward ``timeout``. Returns "" if it exited by itself."""
    t0 = last = time.perf_counter()
    paused = 0.0
    seen: list = [None]
    while proc.poll() is None:
        time.sleep(poll_s)
        now = time.perf_counter()
        hb = _read_heartbeat(hb_path)
        if progress_path is not None and stage_path is not None:
            _sync_progress(stage_path, hb_path, progress_path, seen)
        if hb and hb[0] == "W":
            paused += now - last
        last = now
        reason = ""
        if hb and hb[0] == "B" and time.time() - hb[1] > launch_deadline_s:
            reason = "launch"
        elif now - t0 - paused > timeout:
            reason = "timeout"
        if reason:
            proc.kill()
            proc.wait()
            if progress_path is not None and stage_path is not None:
                _sync_progress(stage_path, hb_path, progress_path, [None])
            return reason
    if progress_path is not None and stage_path is not None:
        _sync_progress(stage_path, hb_path, progress_path, [None])
    return ""


def _run_worker(
    task: str,
    dtype: str,
    size: str,
    kernel_path: Path,
    mode: str,
    seed: int,
    reps: int,
    timeout: float,
    *,
    validation: bool,
    cases: list[str] | None,
    launch_deadline_s: float = LAUNCH_DEADLINE_S,
    worker_module: str = "tcenv.worker",
    progress_path: Path | None = None,
) -> EvalReport:
    """Run one worker under supervision.

    The parent kills the worker (SIGKILL) as soon as a single GPU launch has been in flight for longer than
    ``launch_deadline_s`` (heartbeat file written by the worker around every launch) or the overall
    ``timeout`` elapsed. Nothing can preempt a running Metal kernel from inside, so this bounds the damage of a
    runaway kernel, which would otherwise starve the display of the GPU.
    """
    from .sanitize import env_with_validation, validation_problems

    with tempfile.TemporaryDirectory(prefix="tcenv_") as td:
        tdp = Path(td)
        out_path, hb_path = tdp / "report.json", tdp / "heartbeat"
        stage_path = tdp / "stage.json"
        err_path = tdp / "stderr.txt"
        req = {
            "task": task, "dtype": dtype, "size": size, "kernel_path": str(kernel_path), "mode": mode, "seed": seed,
            "reps": reps, "cases": cases, "out": str(out_path), "heartbeat": str(hb_path), "stage": str(stage_path),
        }
        req_path = tdp / "request.json"
        req_path.write_text(json.dumps(req))
        env = env_with_validation() if validation else dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")  # no compile-worker pool that could outlive a SIGKILLed worker
        base = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=seed)
        t0 = time.perf_counter()
        killed = ""
        with open(err_path, "w") as err_f, open(tdp / "stdout.txt", "w") as out_f:
            proc = subprocess.Popen(
                [sys.executable, "-m", worker_module, str(req_path)], stdout=out_f, stderr=err_f, env=env, cwd=str(REPO_ROOT)
            )
            from .gpulock import released

            with released():  # this process only waits; a worker needs the lock itself
                reason = supervise(
                    proc, hb_path, launch_deadline_s=launch_deadline_s, timeout=timeout,
                    stage_path=stage_path, progress_path=progress_path,
                )
            if reason == "launch":
                killed = (
                    f"a single GPU launch ran for more than {launch_deadline_s:.0f}s (runaway or non-terminating kernel); "
                    "the worker was killed immediately"
                )
            elif reason == "timeout":
                killed = f"evaluation exceeded {timeout:.0f}s; the worker was killed"
        if killed:
            time.sleep(POST_KILL_COOLDOWN_S)
            base.status, base.stage, base.message = "timeout", "timeout", killed
            base.elapsed_s = time.perf_counter() - t0
            return base
        stderr = err_path.read_text(errors="replace")
        if out_path.exists():
            rep = EvalReport.from_json(json.loads(out_path.read_text()))
        else:
            base.status = "crash"
            tail = "\n".join(stderr.strip().splitlines()[-12:])
            base.message = f"worker exited with code {proc.returncode} without a report (GPU fault or process crash). stderr tail:\n{tail}"
            base.elapsed_s = time.perf_counter() - t0
            return base
        if validation:
            probs = validation_problems(stderr)
            rep.safety = dict(rep.safety or {})
            rep.safety["validation"] = probs
            if probs and rep.status in (STATUS_OK, "incorrect", "unsafe", "runtime_error"):
                rep.safety["ok"] = False
                rep.status = "unsafe"
                rep.message = "Metal shader validation: " + probs[0]
        return rep


def evaluate(
    task: str,
    dtype: str,
    size: str,
    kernel_source: str,
    *,
    mode: str = "full",
    seed: int | None = None,
    reps: int = 15,
    timeout: float | None = None,
    cases: list[str] | None = None,
    workdir: Path | None = None,
    progress_path: Path | None = None,
) -> EvalReport:
    """Evaluate ``kernel_source`` on ``task``. ``mode``: "quick" (compile+ABI+safety+correctness)
    or "full" (additionally shader validation and the benchmark)."""
    seed = int(seed if seed is not None else secrets.randbits(48))
    t0 = time.perf_counter()
    from .screen import screen_source

    def _live(rep: EvalReport) -> EvalReport:
        if progress_path is not None:
            try:
                from observe.publish import publish

                publish(progress_path, {
                    "task": rep.task, "dtype": rep.dtype, "size": rep.size, "mode": rep.mode,
                    "stage": rep.stage, "status": rep.status, "message": (rep.message or "")[:300],
                    "finished": rep.status != "running", "updated_at": time.time(),
                })
            except Exception:
                pass
        return rep

    refused = screen_source(kernel_source)
    if refused:  # never reaches a worker, let alone the GPU
        rep = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=seed, status="rejected", stage="screen")
        rep.message = "refused before compilation: " + refused[0]
        rep.abi = {"ok": False, "violations": refused}
        return _live(rep)
    if progress_path is not None:
        try:
            from observe.publish import publish

            publish(progress_path, {"task": task, "dtype": dtype, "size": size, "mode": mode, "stage": "screen", "status": "running", "updated_at": time.time()})
        except Exception:
            pass
    with tempfile.TemporaryDirectory(prefix="tcenv_src_") as td:
        kp = Path(workdir or td) / "kernel.metal"
        kp.parent.mkdir(parents=True, exist_ok=True)
        kp.write_text(kernel_source)
        if mode == "full":
            # 1) shader-validation pass: tight allocations, one case. Also yields fast compile/ABI feedback.
            v = _run_worker(task, dtype, size, kp, "validate", seed, reps, timeout or 120.0, validation=True, cases=cases, launch_deadline_s=2 * LAUNCH_DEADLINE_S, progress_path=progress_path)
            if v.status != STATUS_OK:
                v.mode = "full"
                v.elapsed_s = time.perf_counter() - t0
                return _live(v)
            rep = _run_worker(task, dtype, size, kp, "full", seed, reps, timeout or 420.0, validation=False, cases=cases, progress_path=progress_path)
            rep.safety = dict(rep.safety or {})
            rep.safety["validation"] = []
            rep.safety["shader_validation"] = "clean"
        else:
            rep = _run_worker(task, dtype, size, kp, "quick", seed, reps, timeout or 240.0, validation=False, cases=cases, progress_path=progress_path)
        rep.elapsed_s = time.perf_counter() - t0
        return _live(rep)
