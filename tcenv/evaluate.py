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
# 判卷总控：一份 Metal 候选在这里被决定留下还是判死。
# 父进程先做 screen。子进程内部是 compile → abi → safety → correct，只有 full 那次才进入 bench。
# safety 与 correct 共用一次正确性运行：越界、改输入或非确定记 unsafe，安全通过后数值不对才记 incorrect。
# validate 是另一个子进程，排在 bench 之前。full 先在 Metal shader validation 下做紧分配、单用例检查，失败立刻返回，通过后才测速。
# 状态码各自对应哪一关，写在 EvalReport 上面。

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

from .schema import REPORT_SCHEMA_VERSION, REQUEST_SCHEMA_VERSION, SchemaError, migrate_report, write_json_atomic

REPO_ROOT = Path(__file__).resolve().parent.parent

STATUS_OK = "ok"
# 安静机器上单次重复的 CV 远低于这个值（见行尾英文）。超过它说明测速时还有别的负载，bench 会重测并留下最安静的一次。
NOISY_CV = 0.05  # quiet measurements here have a per-repetition CV of 0.5-1.5%; above this something else was using the machine
# 吵的测速最多做这么多次；留下的是 CV 最小的那一次，三次都吵也仍然留下最安静的。
BENCH_ATTEMPTS = 3  # noisy benchmarks are repeated; the quietest attempt is kept
OWN_RUNNERS = ("candidate", "candidate_serial")


# 进度写不出去也不能改判卷结果，所以上报失败就当没这回事。
def _note_stage(**fields) -> None:
    try:
        from tcenv.events import update_progress

        update_progress(**fields)
    except Exception:
        return


# 一份候选的判决书。status 是生死，stage 是停在哪一关。
# ok：本模式该过的关都过了。quick / validate 停在 correct，full 还包含 bench。eligible 只认 ok。
# rejected：screen 在编译前拒绝，子进程和 GPU 都没碰。
# compile_error：Metal 编译器拒绝了拼上头之后的源码。
# abi_error：编译通过了，但函数名、参数槽、读写或线程规模对不上契约。
# unsafe：护栏被写穿、输入被改，或结果随输出初值 / 两次启动而变。测速抽查时输出又不对，也记在这里。父进程若在 stderr 里看到 shader validation 违例，会把已经启动过的候选改判成这个。
# incorrect：安全关过了，和 CPU 参考的差分没过。输出还留着 NaN 毒化图案（没写满）也算这一关。
# runtime_error：correct 或 bench 途中 GPU 报错。
# timeout：父进程把子进程 SIGKILL 了。单次启动的心跳 B 过久，或整段时限到了；等锁的心跳 W 不计。
# crash：子进程没留下报告，或评测框架自己抛了异常。
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

    # 实现上就是 status 为 ok。形式验证恒为 unsupported，不参与这一条。
    @property
    def eligible(self) -> bool:
        """Safe and numerically correct (the first two eligibility conditions of the paper)."""
        return self.status == STATUS_OK

    def to_json(self) -> dict:
        d = asdict(self)
        d["eligible"] = self.eligible  # derived; older readers still look for it
        d["schema_version"] = REPORT_SCHEMA_VERSION
        return d

    # 读回子进程写的 JSON。没有 schema_version 的旧文件按版本 0 迁移。更新的版本或本版本不认识的字段直接拒绝，不再丢掉后照常构造。
    @staticmethod
    def from_json(d: dict) -> "EvalReport":
        known = {f.name for f in fields(EvalReport)}
        return EvalReport(**migrate_report(d, known))


# ---------------------------------------------------------------------------
# in-process evaluation (runs inside the worker)
# ---------------------------------------------------------------------------


# 子进程里真正开始评的入口。先拿共享 GPU 锁：不计时的关卡可以和别的不计时工作重叠，bench 段再升级成独占，避免两份墙钟叠在一起把 CV 抬高。
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


# 关卡都在这个函数里。开头的 verify 只记账，失败与否都不看它。
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

    # 写上这一关的 status 后返回。失败时 stage 停在调用前设好的那一关，后面的关不再跑。
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
    # 子进程里再过一遍同一条筛，拒绝理由放进 abi.violations，反馈才会印出原因。父进程那一次是为了连子进程都不起。
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
        # 编译器诊断的是「契约头 + 候选正文」。行号减掉文件头，模型才对得上自己写的那几行。
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
    # validate 用紧分配：缓冲两侧没有护栏，越界才会被 Metal shader validation 看见。quick / full 靠护栏图案抓越界。
    tight = mode == "validate"
    argset = ArgSet(ctx, contract, tight=tight)

    baseline_cache: dict[str, Any] = {}

    def baseline_fn(data):
        # validate 只对 CPU 参考使用名义容差，不用 MLX / PyTorch 基线自身的误差来放宽门槛。
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

    # validate 只跑一个用例，并跳过「换输出初值再跑」的确定性检查。这一次要尽快让 shader validation 看见访存；全用例和确定性留在 quick / full。
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
    import platform

    from .domain import coverage_for_report

    # 评测入口编译时不传 math，metal_runtime.compile 的默认是 fast。证据范围把这一点写明，换工具链后旧记录作废。
    rep.correctness["coverage"] = coverage_for_report(task, results, kernel_source, {
        "device": dev_info.get("name"),
        "macos": platform.mac_ver()[0],
        "msl": dev_info.get("language_version"),
        "math_mode": "fast",
    })
    rep.safety = {"ok": not safety and not nondet, "violations": safety, "nondeterminism": nondet, "validation": []}
    # 同一次运行同时带回安全违例和数值结果。有违例或非确定就定 unsafe；安全干净之后，数值不过才定 incorrect。
    if safety or nondet:
        return finish("unsafe", (safety + nondet)[0])
    if not rep.correctness["ok"]:
        first_bad = next(r for r in results if not r.ok)
        return finish("incorrect", f"case '{first_bad.case}': {first_bad.message}")
    # quick / validate 到正确性为止。bench 只属于后面那个 full 子进程。
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
    # 测速升级成独占锁。共享锁下可以并行做正确性检查；两份墙钟叠在一起会把 CV 抬过 NOISY_CV。
    with gpu_lock(f"benchmark of {task_name} {dtype} {size}") as lock_wait_s:
        calib = get_calibration()
        runners: dict[str, Any] = {"candidate": cand_run, "candidate_serial": cand_serial_run}
        base_runners = build_runners(task, data0.rounded)
        for n, r in base_runners.items():
            runners[n] = r.run_k
        try:
            # 噪声是本轮除 candidate_serial 以外各计时器 CV 的最大值。serial 只作对照，不参与要不要重测。
            # 超过 NOISY_CV 就再测，因为这次墙钟里混进了别人的负载，不能用来比快慢。
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
    # 留下 CV 最小的一次。三次都超过阈值也留最安静的那次，并在报告里标 noisy；吵本身不判死。
    noise, stats, cand_gpu = min(attempts, key=lambda a: a[0])

    # 正确性关是单套 buffer、单次启动。测速换成多套轮转和 concurrent 派发，所以计时产出还要再查。
    # 每个真正发射过的集合都查护栏和输入；和参考对数值只抽最先、最后两套。
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
        # 拿不到 Triton→Metal 的对照就跳过。它是附加信息，失败不能把已经通过的候选改判。
        pass
    rep.stage = "done"
    return finish(STATUS_OK, "compiled, ABI ok, safe, correct and benchmarked")


# ---------------------------------------------------------------------------
# subprocess driver
# ---------------------------------------------------------------------------


# 一次 command buffer 从提交到完成，心跳 B 最多停这么久。再长父进程就 SIGKILL。合法启动是毫秒级。
LAUNCH_DEADLINE_S = 2.0  # a single GPU launch of a legitimate candidate takes milliseconds
# 杀掉失控核之后先歇一会儿再评下一份，给 GPU 从这次提交里恢复的时间。
POST_KILL_COOLDOWN_S = 3.0  # give the GPU time to recover after a runaway kernel was killed


# 心跳是固定 22 字节：一个字母加 20 位墙上时间。还没写好或写了一半就当这一拍没有心跳，避免误杀。
def _read_heartbeat(path: Path) -> tuple[str, float] | None:
    try:
        raw = path.read_bytes()[:22].decode()
        return raw[0], float(raw[1:21])
    except Exception:  # noqa: BLE001 - file not created yet / partially written
        return None


# 看板要看到阶段和心跳字母。只有阶段文件或心跳变了才发布，避免每 0.1 秒重写一份快照。
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
        from tcenv.events import publish

        publish(dest, doc)
    except Exception:
        return


# 父进程的监督。子进程只把状态写进心跳文件，杀不杀由这里决定。
# 心跳 B 表示一次 command buffer 已提交还没完成。超过 launch_deadline_s 就 SIGKILL：Metal 核跑起来之后，进程内部抢占不了它。
# 心跳 W 表示在等 GPU 锁，核还没跑。这段墙钟累进 paused，和 timeout 比较时扣掉。
# 子进程自己退出则返回空串。
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
        # 这一拍还标着 W，就把刚过去的轮询间隔从超时预算里拿掉。排队不是这份候选自己的运行时间。
        if hb and hb[0] == "W":
            paused += now - last
        last = now
        reason = ""
        # B 的期限对的是心跳里的墙上时间，并且先于整段 timeout。一次启动卡死要单独记成 launch。
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


# 图形芯片上的编译和启动都在这个子进程里。核卡死、驱动中止或段错误只丢掉这一份候选，搜索进程还在。
# SIGKILL 之后内核会放开子进程持有的 flock；冷却时间用来等 GPU 从这次提交里恢复。
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
            "schema_version": REQUEST_SCHEMA_VERSION,
            "task": task, "dtype": dtype, "size": size, "kernel_path": str(kernel_path), "mode": mode, "seed": seed,
            "reps": reps, "cases": cases, "out": str(out_path), "heartbeat": str(hb_path), "stage": str(stage_path),
        }
        req_path = tdp / "request.json"
        write_json_atomic(req_path, req)
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

            # 父进程等待时先放下自己的 GPU 锁，否则子进程的独占 bench 永远拿不到。
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
        # 单次启动过久和整段超时都记成 status timeout，具体是哪一种留在 message 里。
        if killed:
            time.sleep(POST_KILL_COOLDOWN_S)
            base.status, base.stage, base.message = "timeout", "timeout", killed
            base.elapsed_s = time.perf_counter() - t0
            return base
        stderr = err_path.read_text(errors="replace")
        if out_path.exists():
            try:
                rep = EvalReport.from_json(json.loads(out_path.read_text()))
            except SchemaError as e:
                # 比这份代码新、或不是对象的报告留在返回值里给人看。再抛出去会连临时目录一起删掉，续跑会当成这次评测还没做过。
                base.status = "crash"
                base.message = f"worker report needs human inspection: {e}"
                base.elapsed_s = time.perf_counter() - t0
                return base
            except (OSError, ValueError) as e:
                base.status = "crash"
                base.message = f"worker left a damaged report and needs human inspection: {e}"
                base.elapsed_s = time.perf_counter() - t0
                return base
        else:
            # 监督没有杀它，进程自己退出了却没有 report.json：GPU fault 或段错误，框架来不及写判决。
            base.status = "crash"
            tail = "\n".join(stderr.strip().splitlines()[-12:])
            base.message = f"worker exited with code {proc.returncode} without a report (GPU fault or process crash). stderr tail:\n{tail}"
            base.elapsed_s = time.perf_counter() - t0
            return base
        if validation:
            # 校验违例在子进程 stderr 里，不在 JSON。status 已是 ok、incorrect、unsafe 或 runtime_error，说明核启动过，改判 unsafe。
            probs = validation_problems(stderr)
            rep.safety = dict(rep.safety or {})
            rep.safety["validation"] = probs
            if probs and rep.status in (STATUS_OK, "incorrect", "unsafe", "runtime_error"):
                rep.safety["ok"] = False
                rep.status = "unsafe"
                rep.message = "Metal shader validation: " + probs[0]
        return rep


# 搜索进程调用的入口。Metal 编译和启动发生在子进程里，图形芯片卡死不会拖垮还在搜下一批候选的进程。
# 单次启动超时由 supervise 看见心跳 B 后 SIGKILL；整段时限另算，等锁的 W 不计。
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
                from tcenv.events import publish

                publish(progress_path, {
                    "task": rep.task, "dtype": rep.dtype, "size": rep.size, "mode": rep.mode,
                    "stage": rep.stage, "status": rep.status, "message": (rep.message or "")[:300],
                    "finished": rep.status != "running", "updated_at": time.time(),
                })
            except Exception:
                pass
        return rep

    # 明显的死循环和非法 include 在这里结束，不起子进程，也不把核提交到 GPU。
    refused = screen_source(kernel_source)
    if refused:  # never reaches a worker, let alone the GPU
        rep = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=seed, status="rejected", stage="screen")
        rep.message = "refused before compilation: " + refused[0]
        rep.abi = {"ok": False, "violations": refused}
        return _live(rep)
    if progress_path is not None:
        try:
            from tcenv.events import publish

            publish(progress_path, {"task": task, "dtype": dtype, "size": size, "mode": mode, "stage": "screen", "status": "running", "updated_at": time.time()})
        except Exception:
            pass
    with tempfile.TemporaryDirectory(prefix="tcenv_src_") as td:
        kp = Path(workdir or td) / "kernel.metal"
        kp.parent.mkdir(parents=True, exist_ok=True)
        kp.write_text(kernel_source)
        if mode == "full":
            # 1) shader-validation pass: tight allocations, one case. Also yields fast compile/ABI feedback.
            # 校验子进程在 bench 之前：打开会插桩访存和 barrier 的 shader validation，单次启动期限用两倍 LAUNCH_DEADLINE_S。
            # 这次不过关就带着它的状态码返回。通过后第二个子进程再做完整评测并进入 bench，且不再设置校验环境。
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
