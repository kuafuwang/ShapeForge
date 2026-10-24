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

import hashlib
import json
import os
import re
import secrets
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .schema import PROTOCOL_VERSION, REPORT_SCHEMA_VERSION, REQUEST_SCHEMA_VERSION, SchemaError, migrate_report, write_json_atomic

REPO_ROOT = Path(__file__).resolve().parent.parent

STATUS_OK = "ok"
# 安静机器上单次重复的 CV 远低于这个值（见行尾英文）。超过它说明测速时还有别的负载，bench 会重测并留下最安静的一次。
NOISY_CV = 0.05  # quiet measurements here have a per-repetition CV of 0.5-1.5%; above this something else was using the machine
_CORRECTNESS_CACHE: dict[tuple, dict] = {}
# 吵的测速最多做这么多次；留下的是 CV 最小的那一次，三次都吵也仍然留下最安静的。
BENCH_ATTEMPTS = 3  # noisy benchmarks are repeated; the quietest attempt is kept
OWN_RUNNERS = ("candidate", "candidate_serial")


# 进度写不出去也不能改判卷结果，所以上报失败就当没这回事。
def _note_stage(**fields) -> None:
    """@brief 上报进度阶段，失败时静默忽略。

    @param fields 要上报的进度字段。
    """
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
    """@brief 一份候选的判决书：生死状态、停在哪一关以及各关证据。

    ``status`` 为 ok 时，本模式该过的关都过了：quick / validate 停在 correct，full 还包含 bench。
    其余状态包括 rejected（编译前被筛掉）、compile_error、abi_error、unsafe、incorrect、
    runtime_error、timeout 与 crash，细节留在 ``message`` 与对应关卡的字典里。
    """

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
    compiler_baseline: dict = field(default_factory=dict)  # optional triton-msl comparison (see tcenv.backends.metal.triton_baseline)
    verification: dict = field(default_factory=dict)  # formal equivalence check (tcenv.verify); always 'unsupported' for now
    protocol_version: str = ""
    target_id: str = ""
    backend: str = ""
    backend_version: str = ""
    device_name: str = ""
    gpu_family: str = ""
    device_warnings: list = field(default_factory=list)
    timeline: list = field(default_factory=list)
    cache_hit: bool = False
    elapsed_s: float = 0.0

    # 实现上就是 status 为 ok。形式验证恒为 unsupported，不参与这一条。
    @property
    def eligible(self) -> bool:
        """@brief 候选是否安全且数值正确。

        Safe and numerically correct (the first two eligibility conditions of the paper).

        @return 状态为 ok 时为 True；形式验证恒为 unsupported，不参与这一条。
        """
        return self.status == STATUS_OK

    def to_json(self) -> dict:
        """@brief 把报告转换成可写入文件的字典。

        @return 字段字典，并补上派生的 ``eligible`` 与 ``schema_version``。
        """
        d = asdict(self)
        d["eligible"] = self.eligible  # derived; older readers still look for it
        d["schema_version"] = REPORT_SCHEMA_VERSION
        return d

    # 读回子进程写的 JSON。没有 schema_version 的旧文件按版本 0 迁移。更新的版本或本版本不认识的字段直接拒绝，不再丢掉后照常构造。
    @staticmethod
    def from_json(d: dict) -> "EvalReport":
        """@brief 从 JSON 字典构造报告，必要时做版本迁移。

        @param d 已反序列化的报告字典。
        @return 构造出的 ``EvalReport``。
        @raises SchemaError 版本过新、含未知字段或缺少必要字段时抛出。
        """
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
    fast_bench: bool = False,
    require_contract_launch: bool = False,
    validate: bool = False,
    measure_serial: bool = True,
    target_id: str = "apple-m4-max",
) -> EvalReport:
    """@brief 子进程内开始评测的入口：共享锁下跑完整流水线。

    Untimed stages hold the machine-wide GPU lock shared, the benchmark holds it exclusively (tcenv.gpulock).

    ``validate`` asks for the instrumented shader-validation pass before the timed one. ``mode="full"`` always
    runs it; ``mode="timed"`` runs it only when asked, so a self-check can pay for it once on the final version.

    @param task_name 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @param kernel_source 候选 MSL 源码。
    @param mode "validate" | "quick" | "full" | "timed"。
    @param seed 随机种子；为 None 时随机生成。
    @param reps 计时重复次数。
    @param cases 要跑的正确性用例；为 None 时用任务默认用例。
    @param fast_bench 是否走搜索阶段的快速计时（只比记住的最快基线）。
    @param require_contract_launch 是否禁止候选自带的 taic-launch 块。
    @param validate 是否先跑着色器校验子进程。
    @param measure_serial 是否额外测严格串行 dispatch。
    @param target_id 目标设备 id。
    @return 本次评测的报告。
    """
    from .gpulock import gpu_lock

    with gpu_lock(f"{mode} evaluation of {task_name} {dtype} {size}", exclusive=False):
        return _evaluate_in_process(
            task_name, dtype, size, kernel_source, mode=mode, seed=seed, reps=reps, cases=cases,
            fast_bench=fast_bench, require_contract_launch=require_contract_launch, validate=validate,
            measure_serial=measure_serial, target_id=target_id,
        )


# 关卡都在这个函数里。开头的 verify 只记账，失败与否都不看它。
def _fast_bench_path(task_name: str, dtype: str, size: str) -> Path:
    """@brief 快速计时记住比基线所用的缓存文件路径。

    @param task_name 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @return 缓存 JSON 文件路径，目录可由 ``TCENV_FAST_BENCH_DIR`` 改写。
    """
    root = Path(os.environ.get("TCENV_FAST_BENCH_DIR", Path.home() / ".cache" / "ai_lowerling" / "fast_bench"))
    safe = re.sub(r"[^A-Za-z0-9_.@-]+", "_", f"{task_name}_{dtype}_{size}")
    return root / f"{safe}.json"


def _remembered_baseline(task_name: str, dtype: str, size: str, target_id: str | None = None) -> str | None:
    """@brief 读取上次记住的最快基线名。

    @param task_name 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @param target_id 目标 id；与缓存记录不一致时不采用。
    @return 基线名；无可用记录时为 None。
    """
    path = _fast_bench_path(task_name, dtype, size)
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    recorded = doc.get("target_id")
    if target_id and isinstance(recorded, str) and recorded != target_id:
        return None
    name = doc.get("best_baseline")
    return name if isinstance(name, str) and name else None


def selected_bench_names(baseline_names: list[str], *, fast_bench: bool, remembered: str | None, measure_serial: bool = True) -> list[str]:
    """@brief 选出本次计时实际要测的 runner 名称。

    Search-stage ``fast_bench`` drops the serial dispatch and every baseline except the remembered fastest one.

    @param baseline_names 全部可用基线名。
    @param fast_bench 是否为搜索阶段快速计时。
    @param remembered 上次记住的最快基线名。
    @param measure_serial 是否保留严格串行 dispatch。
    @return 选中的 runner 名列表，第一项恒为 candidate。
    """
    names = ["candidate"]
    if measure_serial and not fast_bench:
        names.append("candidate_serial")
    if fast_bench and remembered in baseline_names:
        names.append(remembered)
    else:
        names.extend(baseline_names)
    return names


def _remember_baseline(task_name: str, dtype: str, size: str, name: str, *, target_id: str) -> None:
    """@brief 记住本次最快基线，供之后的快速计时复用。

    @param task_name 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @param name 基线名。
    @param target_id 目标 id，一并写入缓存。
    """
    path = _fast_bench_path(task_name, dtype, size)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"best_baseline": name, "target_id": target_id}))


def _evaluate_in_process(
    task_name: str,
    dtype: str,
    size: str,
    kernel_source: str,
    *,
    mode: str = "full",
    seed: int | None = None,
    reps: int = 15,
    cases: tuple[str, ...] | None = None,
    fast_bench: bool = False,
    require_contract_launch: bool = False,
    validate: bool = False,
    measure_serial: bool = False,
    target_id: str = "apple-m4-max",
) -> EvalReport:
    """@brief 在子进程内部依次跑过 device_check、screen、compile、abi、correct 与 bench。

    只由 ``tcenv.worker`` 调用。quick / validate 在正确性后返回，full 继续测速；
    任一类失败都会停在对应关卡并用 ``finish`` 记录状态与耗时。

    @param task_name 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @param kernel_source 候选 MSL 源码。
    @param mode 评测模式，决定跑到哪一关。
    @param seed 随机种子。
    @param reps 计时重复次数。
    @param cases 要跑的正确性用例。
    @param fast_bench 是否只测记住的最快基线。
    @param require_contract_launch 是否禁止 taic-launch 块。
    @param validate 是否为着色器校验那一趟。
    @param measure_serial 是否额外测串行 dispatch。
    @param target_id 目标设备 id。
    @return 本次评测的报告。
    """
    import torch  # noqa: F401  (imported early so failures are reported as crashes of the worker)

    from tasks import get_task

    from .bench import interleaved
    from .gpulock import gpu_lock

    t_start = time.perf_counter()
    timeline: list[dict] = []
    stage_name = "device_check"
    stage_started = t_start
    task = get_task(task_name, dtype, size)
    seed = int(seed if seed is not None else secrets.randbits(48))
    from .schema import PROTOCOL_VERSION

    rep = EvalReport(task=task_name, dtype=dtype, size=size, mode=mode, seed=seed, target_id=target_id, protocol_version=PROTOCOL_VERSION)
    from .verify import verify

    rep.verification = verify(task, kernel_source).to_json()  # formal equivalence: unsupported for MSL (tcenv.verify)
    from .target import TargetError, device_matches, limit_notes, require_active

    try:
        profile = require_active(target_id)
    except TargetError as exc:
        rep.stage = "device_check"
        rep.status = "rejected"
        rep.message = str(exc)
        rep.elapsed_s = time.perf_counter() - t_start
        rep.timeline = [{"stage": "device_check", "status": "rejected", "duration_s": rep.elapsed_s}]
        return rep
    contract = task.contract
    from .backends.registry import load_backend

    backend = load_backend(profile.backend)
    ctx = backend.context()
    dev_info = backend.probe(ctx, profile).raw
    rep.backend = profile.backend
    rep.backend_version = profile.backend_version
    rep.device_name = str(dev_info.get("name") or "")
    rep.gpu_family = str(dev_info.get("gpu_family") or "")
    if not device_matches(profile, dev_info):
        rep.stage = "device_check"
        rep.status = "rejected"
        rep.message = (
            f"device {rep.device_name} ({rep.gpu_family}) does not match target {profile.id} "
            f"({profile.match_name_contains}, {profile.match_family})"
        )
        rep.elapsed_s = time.perf_counter() - t_start
        rep.timeline = [{"stage": "device_check", "status": "rejected", "duration_s": rep.elapsed_s}]
        return rep
    rep.device_warnings = limit_notes(profile, dev_info)

    def _enter(stage: str) -> None:
        nonlocal stage_name, stage_started
        now = time.perf_counter()
        if stage != stage_name:
            timeline.append({"stage": stage_name, "status": "ok", "duration_s": now - stage_started})
            stage_name = stage
            stage_started = now

    # 写上这一关的 status 后返回。失败时 stage 停在调用前设好的那一关，后面的关不再跑。
    def finish(status: str, message: str = "") -> EvalReport:
        now = time.perf_counter()
        rep.status = status
        if message:
            rep.message = message
        rep.elapsed_s = now - t_start
        timeline.append({"stage": stage_name, "status": status, "duration_s": now - stage_started})
        rep.timeline = list(timeline)
        _note_stage(stage=rep.stage, status=status, task=task_name, dtype=dtype, size=size, mode=mode)
        return rep

    # -- static screen (before anything touches the GPU) ---------------------------
    from .backends.metal.screen import screen_source

    rep.stage = "screen"
    _enter("screen")
    _note_stage(stage="screen", status="running", task=task_name, dtype=dtype, size=size, mode=mode)
    # 子进程里再过一遍同一条筛，拒绝理由放进 abi.violations，反馈才会印出原因。父进程那一次是为了连子进程都不起。
    refused = backend.screen(kernel_source, task)
    if refused:
        rep.abi = {"ok": False, "violations": refused}
        return finish("rejected", "refused before compilation: " + refused[0])

    from .backends.metal.launch import LaunchRejected, check_launch_limits

    try:
        spec = backend.parse_launch(kernel_source, contract)
        if spec is not None and require_contract_launch:
            raise LaunchRejected("a contract experiment requires the task's own launch; taic-launch is not allowed here")
        if spec is not None:
            mtp = dev_info.get("max_threads_per_threadgroup_xyz") or (dev_info["max_threads_per_threadgroup"],) * 3
            check_launch_limits(
                spec,
                contract_names={a.name for a in contract.args},
                max_threads_xyz=tuple(mtp),
                max_threads_product=int(dev_info["max_threads_per_threadgroup"]),
                max_threadgroup_memory_bytes=int(dev_info["max_threadgroup_memory_bytes"]),
                recommended_working_set_bytes=int(dev_info.get("recommended_working_set_bytes") or 0),
                max_buffer_length=int(dev_info.get("max_buffer_length") or 2**31),
                rotation_sets=backend.rotation_sets(task),
            )
    except LaunchRejected as e:
        rep.abi = {"ok": False, "violations": [str(e)]}
        return finish("rejected", "refused before compilation: " + str(e))
    except MemoryError as e:
        return finish("rejected", f"refused before compilation: could not size the launch ({e})")

    # -- compile -------------------------------------------------------------
    rep.stage = "compile"
    _enter("compile")
    _note_stage(stage="compile", status="running", task=task_name, dtype=dtype, size=size, mode=mode)
    full_src, header_lines = backend.assemble(contract, kernel_source)
    try:
        if spec is None:
            kernels = [backend.compile(ctx, full_src, contract)]
        else:
            names = list(dict.fromkeys(stage.kernel for stage in spec.stages))
            kernels = backend.compile_all(ctx, full_src, contract, names=names, math=spec.math)
    except backend.compile_error() as e:
        # 编译器诊断的是「契约头 + 候选正文」。行号减掉文件头，模型才对得上自己写的那几行。
        msg = backend.rewrite_compiler_lines(contract, str(e), header_lines)
        rep.compile = {"ok": False, "errors": msg}
        return finish("compile_error", "Metal compiler rejected the kernel")
    kernel = kernels[0]
    rep.compile = {
        "ok": True,
        "compile_s": kernel.compile_s,
        "pipeline_s": sum(k.pipeline_s for k in kernels),
        "language_version": kernel.language_version,
        "math_mode": spec.math if spec is not None else "fast",
        "launch": None if spec is None else spec.digest(),
    }

    # -- abi -------------------------------------------------------------------
    rep.stage = "abi"
    _enter("abi")
    _note_stage(stage="abi", status="running", task=task_name, dtype=dtype, size=size, mode=mode)
    problems = backend.check_abi(
        contract, kernels,
        max_tg_memory=dev_info["max_threadgroup_memory_bytes"],
        max_buffer_index=profile.limits.max_buffer_index,
        target_id=target_id,
        spec=spec,
    )
    # 插桩会把静态线程组内存报成大约两倍。超过设备上限后再启动，Metal 在
    # validateComputeFunctionArguments 里断言并 SIGABRT，子进程变成 crash。
    # 这种只回报那一条、不启动。父进程改用不插桩的完整评测去比目标档案里的线程组内存上限。
    if mode == "validate":
        problems = backend.validation_problems(
            problems, [k.static_tg_bytes for k in kernels], dev_info["max_threadgroup_memory_bytes"],
        )
    rep.abi = {"ok": not problems, "violations": problems}
    if spec is None:
        rep.pipeline = backend.pipeline_stats(kernel, contract, dev_info, kernel_source=kernel_source)
    else:
        by_name = {k.name: k for k in kernels}
        rep.pipeline = backend.pipeline_stats_many([(by_name[s.kernel], s) for s in spec.stages], dev_info, kernel_source=kernel_source)
    if problems:
        return finish("abi_error", "kernel does not match the required ABI / launch contract")

    # -- safety + correctness ------------------------------------------------------
    rep.stage = "correctness"
    _enter("correctness")
    _note_stage(stage="correctness", status="running", task=task_name, dtype=dtype, size=size, mode=mode)
    # validate 用紧分配：缓冲两侧没有护栏，越界才会被 Metal shader validation 看见。quick / full 靠护栏图案抓越界。
    tight = mode == "validate"
    try:
        argset = backend.prepare_correctness(ctx, contract, tight=tight, spec=spec)
    except MemoryError as e:
        return finish("rejected", f"refused before launch: could not allocate the workspace ({e})")

    baseline_cache: dict[str, Any] = {}
    correctness_key = (task_name, dtype, size, seed)

    def baseline_fn(data):
        # 每个用例单独缓存：同一份 (task,dtype,size,seed) 的不同 case 参考输出不同，用 case i 的基线
        # 去定 case j 的误差容限，会把 allow 推到 ALLOW_CAP。
        remembered = _CORRECTNESS_CACHE.get((*correctness_key, data.case))
        # validate 只对 CPU 参考使用名义容差，不用 MLX / PyTorch 基线自身的误差来放宽门槛。
        if mode == "validate":
            return None
        if fast_bench and remembered is not None:
            return remembered
        outs: dict[str, dict] = {}
        for impl in task.baselines:
            try:
                r = backend.baseline_runner(impl, task, data)
                outs[impl.name] = r.outputs()
            except Exception as e:  # noqa: BLE001
                baseline_cache.setdefault("errors", []).append(f"{impl.name}: {type(e).__name__}: {e}")
        if outs and fast_bench:
            _CORRECTNESS_CACHE[(*correctness_key, data.case)] = outs
        return outs or None

    # validate 只跑一个用例，并跳过「换输出初值再跑」的确定性检查。这一次要尽快让 shader validation 看见访存；全用例和确定性留在 quick / full。
    use_cases = cases or (task.cases[:1] if mode == "validate" else task.cases)
    launcher = None
    if spec is not None:
        by_name = {k.name: k for k in kernels}
        ordered = [by_name[stage.kernel] for stage in spec.stages]
        grids = [stage.grid for stage in spec.stages]
        threadss = [stage.threads for stage in spec.stages]
        tg_bytes = [stage.threadgroup_bytes for stage in spec.stages]

        def launcher(one, *, poison, fill_zero, _ordered=ordered, _grids=grids, _threads=threadss, _tg=tg_bytes):
            if fill_zero:
                for a in contract.outputs:
                    gb = one.buffers[a.name]
                    gb.mem[gb.guard : gb.guard + a.nbytes] = bytes(a.nbytes)
            elif poison:
                one.poison_outputs()
            one.prepare_workspaces(poison=poison, fill_zero=fill_zero)
            ctx.run_pipeline(_ordered, _grids, _threads, _tg, [[one.stage_args(k) for k in _ordered]], [[]], iters=1, dispatch="serial")
            return one.read_outputs()

    try:
        results, safety, nondet = backend.run_correctness(
            ctx, kernel, task, argset, seed, cases=use_cases, baseline_fns=baseline_fn,
            determinism=(mode != "validate"), launcher=launcher,
        )
    except backend.runtime_error() as e:
        rep.safety = {"ok": False, "violations": [f"GPU execution error: {e}"], "nondeterminism": [], "validation": []}
        return finish("runtime_error", f"the GPU reported an error while running the kernel: {str(e)[:300]}")

    rep.correctness = {
        "ok": all(r.ok for r in results),
        "cases": [r.to_json() for r in results],
        "tolerance": {"rtol": task.rtol, "atol_rel": task.atol_rel, "exact": task.exact},
        "baseline_errors": baseline_cache.get("errors", []),
    }
    import platform

    # 评测入口编译时不传 math，metal_runtime.compile 的默认是 fast。证据范围把这一点写明，换工具链后旧记录作废。
    rep.correctness["coverage"] = backend.coverage(task, results, kernel_source, {
        "device": dev_info.get("name"),
        "macos": platform.mac_ver()[0],
        "msl": dev_info.get("language_version"),
        "math_mode": spec.math if spec is not None else "fast",
    })
    if isinstance(rep.correctness.get("coverage"), dict):
        # sampling_coverage drops the toolchain dict, so the mode is written again here.
        rep.correctness["coverage"]["math_mode"] = spec.math if spec is not None else "fast"
        if spec is not None:
            rep.correctness["coverage"]["launch"] = spec.digest()
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
    _enter("bench")
    _note_stage(stage="bench", status="running", task=task_name, dtype=dtype, size=size, mode=mode, attempt=1, attempts_max=BENCH_ATTEMPTS)
    data0 = backend.prepare_case(task, seed + 1)
    try:
        sets = backend.prepare_bench_sets(ctx, contract, task, spec=spec, data0=data0)
    except MemoryError as e:
        return finish("rejected", f"refused before launch: could not allocate the workspace ({e})")

    attempts: list[tuple[float, dict, list[float]]] = []  # (noise, stats, candidate GPU samples) per attempt
    # 测速升级成独占锁。共享锁下可以并行做正确性检查；两份墙钟叠在一起会把 CV 抬过 NOISY_CV。
    with gpu_lock(f"benchmark of {task_name} {dtype} {size}") as lock_wait_s:
        calib = backend.calibration(target_id)
        base_runners = backend.baseline_runners(task, data0)
        only = _remembered_baseline(task_name, dtype, size, target_id) if fast_bench else None
        selected = selected_bench_names(list(base_runners), fast_bench=fast_bench, remembered=only, measure_serial=measure_serial)
        try:
            harness = backend.prepare_bench(
                ctx, kernels, contract, task,
                spec=spec, data0=data0, reps=reps, measure_serial=measure_serial,
                sets=sets, base_runners=base_runners, selected=selected,
            )
        except MemoryError as e:
            return finish("rejected", f"refused before launch: could not allocate the workspace ({e})")
        R = harness.rotation_sets
        gpu_samples = harness.gpu_samples
        runners = {"candidate": harness.cand_run, **dict(harness.baseline_runners)}
        if harness.cand_serial_run is not None:
            runners["candidate_serial"] = harness.cand_serial_run
        try:
            # 噪声是本轮除 candidate_serial 以外各计时器 CV 的最大值。serial 只作对照，不参与要不要重测。
            # 超过 NOISY_CV 就再测，因为这次墙钟里混进了别人的负载，不能用来比快慢。
            for attempt in range(BENCH_ATTEMPTS):
                _note_stage(stage="bench", status="running", attempt=attempt + 1, attempts_max=BENCH_ATTEMPTS, noisy=False)
                first = len(gpu_samples)
                st = interleaved(runners, reps=reps, seed=seed + attempt, kmin=R)
                attempts.append((max(s.cv for n, s in st.items() if n != "candidate_serial" or "candidate_serial" not in st), st, gpu_samples[first:]))
                noisy = attempts[-1][0] > NOISY_CV
                _note_stage(stage="bench", status="running", attempt=attempt + 1, attempts_max=BENCH_ATTEMPTS, noisy=noisy)
                if not noisy:
                    break
        except backend.runtime_error() as e:
            return finish("runtime_error", f"GPU error during benchmarking: {str(e)[:300]}")
    # 留下第一次安静的尝试。在最多三次里挑 CV 最小的一次，是对估计量按它自己的离散度做选择，
    # 中位数会系统性偏向更安静的那一次；三次都吵时才退回到最安静的一次并标 noisy。
    quiet = [a for a in attempts if a[0] <= NOISY_CV]
    noise, stats, cand_gpu = quiet[0] if quiet else min(attempts, key=lambda a: a[0])

    # Correctness used one buffer set. The benchmark rotates and dispatches concurrently,
    # so ask the backend to re-check the sets that actually ran during the timed attempts.
    bad_after = harness.spot_check()
    if bad_after:
        rep.safety = {"ok": False, "violations": bad_after, "nondeterminism": [], "validation": []}
        return finish("unsafe", bad_after[0])

    cand = stats["candidate"]
    rep.candidate_us = cand.median_us
    rep.benchmark = {
        "candidate": cand.to_json(),
        "candidate_gpu_us_median": (statistics.median(cand_gpu) * 1e6) if cand_gpu else None,
        "baselines": {n: stats[n].to_json() for n in stats if n not in OWN_RUNNERS},
        "candidate_serial_us": stats["candidate_serial"].median_us if "candidate_serial" in stats else None,
        "measure_serial": measure_serial,
        "dispatch": harness.dispatch_note,
        "rotation_sets": R,
        "noise_cv": noise,  # largest per-repetition CV among the runners of the kept attempt
        "attempts": len(attempts),
        "noisy": noise > NOISY_CV,
        "gpu_lock_wait_s": lock_wait_s,
        "protocol": harness.protocol_note,
    }
    rep.baseline_us = {n: stats[n].median_us for n in stats if n not in OWN_RUNNERS}
    rep.speedups = {n: stats[n].median_s / cand.median_s for n in stats if n not in OWN_RUNNERS}
    if rep.speedups:
        best = min(rep.baseline_us, key=lambda n: rep.baseline_us[n])
        rep.speedup = rep.speedups[best]
        rep.benchmark["best_baseline"] = best
        if fast_bench:
            _remember_baseline(task_name, dtype, size, best, target_id=target_id)
    rep.roofline = backend.roofline(task, cand.median_s, calib)
    try:  # compiler baseline (Triton -> Metal through triton-msl), when it was measured for this task
        from .backends.metal.triton_baseline import estimate, load_cached

        cached = load_cached(task_name, dtype, size)
        est = estimate(cached, rep.baseline_us) if cached else None
        if est:
            est["speedup"] = est["estimated_us"] / cand.median_us
            rep.compiler_baseline = est
    except Exception:  # noqa: BLE001 - optional information must never fail an evaluation
        # 拿不到 Triton→Metal 的对照就跳过。它是附加信息，失败不能把已经通过的候选改判。
        pass
    rep.stage = "done"
    _enter("done")
    return finish(STATUS_OK, "compiled, ABI ok, safe, correct and benchmarked")


# ---------------------------------------------------------------------------
# subprocess driver
# ---------------------------------------------------------------------------


# 一次 command buffer 从提交到完成，心跳 B 最多停这么久。再长父进程就 SIGKILL。合法单核是毫秒级。
LAUNCH_DEADLINE_S = 2.0  # a single GPU launch of a legitimate candidate takes milliseconds
COMPILE_DEADLINE_S = 90.0  # a stuck Metal frontend must not hold the shared GPU lock for the whole worker timeout
# 流水线仍是一条 command buffer，里面每一段各派发一次。每段给单核同样的 2 秒，总和封顶，避免 32 段把显示器占满一分钟。
# 超过这个数仍然 SIGKILL，并且仍然计入 max_hangs：杀了之后分不出「还会自己结束」和「死循环」。
PIPELINE_DEADLINE_CAP_S = 8.0


def command_buffer_deadline_s(n_stages: int) -> float:
    """@brief 给出一次在飞 command buffer 的父进程等待时长。

    How long the parent waits for one in-flight command buffer.

    One kernel stays at 2 seconds. A pipeline is one buffer with one dispatch per stage,
    so the wait is 2 seconds times the stage count, and never more than
    ``PIPELINE_DEADLINE_CAP_S``.

    @param n_stages 流水线阶段数。
    @return 等待秒数。
    """
    stages = max(1, int(n_stages))
    if stages == 1:
        return LAUNCH_DEADLINE_S
    return min(PIPELINE_DEADLINE_CAP_S, LAUNCH_DEADLINE_S * stages)


def deadline_for_source(task: str, dtype: str, size: str, source: str) -> float:
    """@brief 按候选源码决定本次 command buffer 的等待时限。

    The command-buffer wait for this source. An unreadable launch block stays at 2 seconds.

    The worker rejects that block before the GPU. A parsed pipeline uses one wait for every stage.

    @param task 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @param source 候选 MSL 源码。
    @return 等待秒数。
    """
    from tasks import get_task

    from .backends.metal.launch import LaunchRejected, parse_launch_spec

    try:
        contract = get_task(task, dtype, size).contract
    except Exception:  # noqa: BLE001 - an unknown task still has the single-kernel wait
        return LAUNCH_DEADLINE_S
    constants = {
        name: int(value)
        for name, _typ, value in contract.constants
        if isinstance(value, int) and not isinstance(value, bool)
    }
    try:
        spec = parse_launch_spec(source, constants)
    except LaunchRejected:
        return LAUNCH_DEADLINE_S
    if spec is None:
        return LAUNCH_DEADLINE_S
    return command_buffer_deadline_s(len(spec.stages))
# 杀掉失控核之后先歇一会儿再评下一份，给 GPU 从这次提交里恢复的时间。
POST_KILL_COOLDOWN_S = 3.0  # give the GPU time to recover after a runaway kernel was killed


# 心跳是固定 22 字节：一个字母加 20 位墙上时间。还没写好或写了一半就当这一拍没有心跳，避免误杀。
def _read_heartbeat(path: Path) -> tuple[str, float] | None:
    """@brief 读取定宽心跳：字母加 20 位墙上时间。

    @param path 心跳文件路径。
    @return ``(状态字母, 墙上时间)``；文件缺失或未写完时为 None。
    """
    try:
        raw = path.read_bytes()[:22].decode()
        return raw[0], float(raw[1:21])
    except Exception:  # noqa: BLE001 - file not created yet / partially written
        return None


# 看板要看到阶段和心跳字母。只有阶段文件或心跳变了才发布，避免每 0.1 秒重写一份快照。
def _sync_progress(stage_path: Path, hb_path: Path, dest: Path, seen: list) -> None:
    """@brief 把 worker 的阶段文件与心跳字母同步到看板快照。

    Copy the worker's stage file and the heartbeat letter into ``dest`` when either changes.

    The heartbeat file itself stays a fixed-width ``B`` / ``E`` / ``W`` plus a timestamp. Age is computed by the
    dashboard from ``heartbeat_at``, so this does not rewrite the snapshot on every poll.

    @param stage_path worker 写的阶段文件。
    @param hb_path 心跳文件。
    @param dest 看板要读的快照路径。
    @param seen 单元素列表，保存上次见过的签名以避免重复发布。
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
    compile_deadline_s: float = COMPILE_DEADLINE_S,
) -> str:
    """@brief 监督一个 worker 子进程，必要时 SIGKILL 它的进程组。

    Wait for ``proc``; SIGKILL its process group when compile (heartbeat "C") exceeds ``compile_deadline_s``,
    one GPU launch (heartbeat "B") exceeds ``launch_deadline_s``, or the run exceeds ``timeout``. Time spent waiting
    for the GPU lock (heartbeat "W") does not count toward ``timeout``. Returns "" if it exited by itself.

    @param proc worker 子进程。
    @param hb_path 心跳文件路径。
    @param launch_deadline_s 单次 command buffer 的最长秒数。
    @param timeout 整段评测的最长秒数，等锁时间不计入。
    @param poll_s 轮询间隔秒数。
    @param stage_path worker 的阶段文件，用于同步看板。
    @param progress_path 看板快照路径。
    @param compile_deadline_s 编译阶段的最长秒数。
    @return 空串表示子进程自行退出；否则返回 "compile" / "launch" / "timeout"。
    """
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
        # C 是编译。B 是一次 command buffer。都按心跳里的墙上时间，并且先于整段 timeout。
        if hb and hb[0] == "C" and time.time() - hb[1] > compile_deadline_s:
            reason = "compile"
        elif hb and hb[0] == "B" and time.time() - hb[1] > launch_deadline_s:
            reason = "launch"
        elif now - t0 - paused > timeout:
            reason = "timeout"
        if reason:
            from .children import kill_group

            kill_group(proc)
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
    fast_bench: bool = False,
    require_contract_launch: bool = False,
    measure_serial: bool = True,
    target_id: str = "apple-m4-max",
) -> EvalReport:
    """@brief 在一份新临时目录里拉起一个 worker 并回收它的报告。

    Run one worker under supervision.

    The parent kills the worker (SIGKILL) as soon as a single GPU launch has been in flight for longer than
    ``launch_deadline_s`` (heartbeat file written by the worker around every launch) or the overall
    ``timeout`` elapsed. Nothing can preempt a running Metal kernel from inside, so this bounds the damage of a
    runaway kernel, which would otherwise starve the display of the GPU.

    @param task 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @param kernel_path 已写盘的候选源码路径。
    @param mode worker 要跑的模式。
    @param seed 随机种子。
    @param reps 计时重复次数。
    @param timeout 整段评测的最长秒数。
    @param validation 是否在着色器校验环境下运行。
    @param cases 要跑的正确性用例。
    @param launch_deadline_s 单次 command buffer 的最长秒数。
    @param worker_module 要拉起的 Python 模块名。
    @param progress_path 看板快照路径。
    @param fast_bench 是否只测记住的最快基线。
    @param require_contract_launch 是否禁止 taic-launch 块。
    @param measure_serial 是否额外测串行 dispatch。
    @param target_id 目标设备 id。
    @return worker 报告；被杀或崩溃时返回对应的 timeout / crash 报告。
    """
    from .backends.metal.sanitize import VALIDATION_ENV, env_with_validation, validation_problems

    with tempfile.TemporaryDirectory(prefix="tcenv_") as td:
        tdp = Path(td)
        out_path, hb_path = tdp / "report.json", tdp / "heartbeat"
        stage_path = tdp / "stage.json"
        err_path = tdp / "stderr.txt"
        req = {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "task": task, "dtype": dtype, "size": size, "kernel_path": str(kernel_path), "mode": mode, "seed": seed,
            "reps": reps, "cases": cases, "out": str(out_path), "heartbeat": str(hb_path), "stage": str(stage_path),
            "target_id": target_id, "measure_serial": measure_serial,
        }
        if fast_bench:
            req["fast_bench"] = True
        if require_contract_launch:
            req["require_contract_launch"] = True
        req_path = tdp / "request.json"
        write_json_atomic(req_path, req)
        from .children import forget, kill_group, popen_group, scrubbed_env

        env = scrubbed_env()
        if validation:
            env = env_with_validation(env)
        else:
            # A shell that exported MTL_SHADER_VALIDATION would instrument the timed build as well:
            # the candidate pays for validation while the MLX/torch baselines do not, and no field
            # in the report would show it. The timed worker gets an uninstrumented environment.
            for key in VALIDATION_ENV:
                env.pop(key, None)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")  # no compile-worker pool that could outlive a SIGKILLed worker
        base = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=seed)
        t0 = time.perf_counter()
        killed = ""
        with open(err_path, "w") as err_f, open(tdp / "stdout.txt", "w") as out_f:
            proc = popen_group(
                [sys.executable, "-m", worker_module, str(req_path)], stdout=out_f, stderr=err_f, env=env, cwd=str(REPO_ROOT)
            )
            from .gpulock import released

            try:
                # 父进程等待时先放下自己的 GPU 锁，否则子进程的独占 bench 永远拿不到。
                with released():  # this process only waits; a worker needs the lock itself
                    reason = supervise(
                        proc, hb_path, launch_deadline_s=launch_deadline_s, timeout=timeout,
                        stage_path=stage_path, progress_path=progress_path,
                    )
            finally:
                kill_group(proc)
                forget(proc)
            if reason == "compile":
                killed = f"Metal compilation ran for more than {COMPILE_DEADLINE_S:.0f}s; the worker was killed"
            elif reason == "launch":
                if launch_deadline_s > LAUNCH_DEADLINE_S:
                    killed = (
                        f"a pipeline command buffer ran for more than {launch_deadline_s:.0f}s "
                        "(runaway or non-terminating kernel); the worker was killed immediately"
                    )
                else:
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



EVAL_CACHE_SCHEMA = 2


def _eval_cache_dir() -> Path:
    """@brief 评测缓存目录。

    @return ``TCENV_EVAL_CACHE_DIR`` 指定的目录，默认为用户缓存下的 ``evals``。
    """
    return Path(os.environ.get("TCENV_EVAL_CACHE_DIR", Path.home() / ".cache" / "ai_lowerling" / "evals"))


def _callable_identity(fn) -> str:
    """@brief 为缓存键提取可执行函数的源码、闭包和默认值身份。"""
    if fn is None:
        return ""
    import functools
    import inspect

    if isinstance(fn, functools.partial):
        return repr({"partial": _callable_identity(fn.func), "args": fn.args, "keywords": fn.keywords})
    body = {
        "module": getattr(fn, "__module__", ""),
        "name": getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn))),
        "defaults": repr(getattr(fn, "__defaults__", None)),
        "kwdefaults": repr(getattr(fn, "__kwdefaults__", None)),
    }
    try:
        body["source"] = inspect.getsource(fn)
    except (OSError, TypeError):
        code = getattr(fn, "__code__", None)
        body["source"] = repr(code.co_code) if code is not None else ""
    closure = getattr(fn, "__closure__", None)
    if closure:
        body["closure"] = repr([cell.cell_contents for cell in closure])
    return repr(body)


def _evaluation_fingerprint(task) -> str:
    """@brief 计算影响评测结果的可执行语义指纹，而不只任务规格文本。"""
    from .task import task_fingerprint

    body = {
        "task": task_fingerprint(task),
        "gen_inputs": _callable_identity(task.gen_inputs),
        "reference": _callable_identity(task.reference),
        "error_scale": _callable_identity(task.error_scale),
        "baselines": [
            {"name": impl.name, "framework": impl.framework, "fn": _callable_identity(impl.fn)}
            for impl in task.baselines
        ],
    }
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, default=repr, separators=(",", ":")).encode("utf-8")).hexdigest()


def _eval_cache_key(
    *, task: str, dtype: str, size: str, mode: str, seed: int, reps: int,
    cases: tuple[str, ...] | None, fast_bench: bool, require_contract_launch: bool,
    validate: bool, measure_serial: bool, target_id: str, kernel_source: str,
    backend_id: str = "", backend_version: str = "", evaluation_fingerprint: str | None = None,
) -> str:
    """@brief 计算一次评测的缓存键。

    键覆盖文件 schema、测量协议、任务指纹、内核哈希与全部影响结果的评测参数。

    @param task 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @param mode 评测模式。
    @param seed 随机种子。
    @param reps 计时重复次数。
    @param cases 正确性用例。
    @param fast_bench 是否快速计时。
    @param require_contract_launch 是否禁止 taic-launch 块。
    @param validate 是否含着色器校验。
    @param measure_serial 是否测串行 dispatch。
    @param target_id 目标设备 id。
    @param kernel_source 候选源码。
    @param backend_id 后端身份；后端变化不能复用旧报告。
    @param backend_version 后端版本；版本变化不能复用旧报告。
    @param evaluation_fingerprint 可覆盖的评测语义指纹；默认从任务的可执行输入、参考和基线计算。
    @return 十六进制 sha256 缓存键。
    """
    from .schema import PROTOCOL_VERSION
    from .task import task_fingerprint
    from tasks import get_task

    body = {
        "schema": EVAL_CACHE_SCHEMA,
        "protocol": PROTOCOL_VERSION,
        "task": task, "dtype": dtype, "size": size, "mode": mode,
        "task_fingerprint": task_fingerprint(get_task(task, dtype, size)),
        "evaluation_fingerprint": _evaluation_fingerprint(get_task(task, dtype, size)) if evaluation_fingerprint is None else evaluation_fingerprint,
        "backend": backend_id,
        "backend_version": backend_version,
        "kernel_sha256": hashlib.sha256(kernel_source.encode("utf-8")).hexdigest(),
        "seed": int(seed), "reps": int(reps), "cases": list(cases or ()),
        "fast_bench": bool(fast_bench), "require_contract_launch": bool(require_contract_launch),
        "validate": bool(validate), "measure_serial": bool(measure_serial), "target_id": target_id,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _cache_read(key: str) -> "EvalReport | None":
    """@brief 按缓存键读取一份报告。

    @param key 缓存键。
    @return 报告；未命中或文件损坏时为 None。
    """
    path = _eval_cache_dir() / f"{key}.json"
    if not path.is_file():
        return None
    try:
        from .schema import SchemaError

        doc = json.loads(path.read_text(encoding="utf-8"))
        return EvalReport.from_json(doc)
    except (OSError, ValueError, SchemaError):
        return None


def _cache_write(key: str, report: "EvalReport") -> None:
    """@brief 缓存一份状态为 ok 的报告。

    @param key 缓存键。
    @param report 要缓存的报告；非 ok 状态不会写入。
    """
    if report.status != "ok":
        return
    path = _eval_cache_dir() / f"{key}.json"
    try:
        from .schema import write_json_atomic

        write_json_atomic(path, report.to_json())
    except OSError:
        return


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
    fast_bench: bool = False,
    require_contract_launch: bool = False,
    self_check_dir: Path | None = None,
    validate: bool = False,
    measure_serial: bool = False,
    eval_cache: bool = False,
    eval_cache_refresh: bool = False,
    target_id: str = "apple-m4-max",
) -> EvalReport:
    """@brief 在受监督的子进程里评测一份候选内核。

    Evaluate ``kernel_source`` on ``task``. ``mode``: "quick" (compile+ABI+safety+correctness),
    "timed" (those, plus the benchmark, without shader validation), or "full" (validation and the benchmark).

    ``self_check_dir`` is set for a coding agent's ``--timed`` run. A previous timeout or crash in that
    directory is refused before any GPU work, and every timed result is appended to ``self_checks.jsonl``.

    ``validate`` asks a ``timed`` run for the instrumented shader-validation pass first, so a self-check can pay
    for it once on the version it intends to hand back. ``full`` always validates; ``quick`` never does.

    @param task 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @param kernel_source 候选 MSL 源码。
    @param mode 评测模式。
    @param seed 随机种子；为 None 时随机生成。
    @param reps 计时重复次数。
    @param timeout 整段评测的最长秒数；为 None 时按模式取默认值。
    @param cases 要跑的正确性用例。
    @param workdir 候选源码的写盘目录。
    @param progress_path 看板快照路径。
    @param fast_bench 是否只测记住的最快基线。
    @param require_contract_launch 是否禁止 taic-launch 块。
    @param self_check_dir 自测记录目录；只在 timed 模式使用。
    @param validate timed 模式下是否先跑着色器校验。
    @param measure_serial 是否额外测串行 dispatch。
    @param eval_cache 是否复用相同参数的缓存报告。
    @param eval_cache_refresh 是否跳过读缓存、仍写回新报告。
    @param target_id 目标设备 id。
    @return 本次评测的报告。
    """
    seed = int(seed if seed is not None else secrets.randbits(48))
    t0 = time.perf_counter()
    from .backends.registry import load_backend
    from .backends.metal.screen import screen_source
    from .target import TargetError, load_target as _load_target

    try:
        backend = load_backend(_load_target(target_id).backend)
    except TargetError:
        backend = None

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

    def _checked(rep: EvalReport) -> EvalReport:
        if self_check_dir is not None and mode == "timed" and rep.status != "refused_hang":
            from .selfcheck import append

            bench = rep.benchmark if isinstance(rep.benchmark, dict) else {}
            cand = bench.get("candidate") if isinstance(bench.get("candidate"), dict) else {}
            append(self_check_dir, {
                "status": rep.status,
                "candidate_us": rep.candidate_us,
                "cv": cand.get("cv"),
                "speedup": rep.speedup,
                "elapsed_s": rep.elapsed_s,
                # A later trust check must know which harness/device produced this row.
                "task": task, "dtype": dtype, "size": size,
                "target_id": target_id, "protocol_version": PROTOCOL_VERSION,
                "kernel_sha256": hashlib.sha256(kernel_source.encode("utf-8")).hexdigest(),
            })
        return _live(rep)

    cache_key = None
    if eval_cache and not fast_bench and seed is not None:
        cache_key = _eval_cache_key(
            task=task, dtype=dtype, size=size, mode=mode, seed=seed, reps=reps, cases=cases,
            fast_bench=fast_bench, require_contract_launch=require_contract_launch, validate=validate,
            measure_serial=measure_serial, target_id=target_id, kernel_source=kernel_source,
            backend_id=getattr(backend, "id", ""), backend_version=getattr(backend, "version", ""),
        )
        cached = None if eval_cache_refresh else _cache_read(cache_key)
        if cached is not None:
            cached.cache_hit = True
            if self_check_dir is not None and mode == "timed" and cached.status == "ok":
                from .selfcheck import append

                bench = cached.benchmark if isinstance(cached.benchmark, dict) else {}
                cand = bench.get("candidate") if isinstance(bench.get("candidate"), dict) else {}
                append(self_check_dir, {
                    "status": "ok", "candidate_us": cached.candidate_us, "cv": cand.get("cv"),
                    "speedup": cached.speedup, "elapsed_s": 0.0, "cached": True,
                    "task": task, "dtype": dtype, "size": size,
                    "target_id": target_id, "protocol_version": PROTOCOL_VERSION,
                    "kernel_sha256": hashlib.sha256(kernel_source.encode("utf-8")).hexdigest(),
                })
            return _live(cached)

    if mode == "timed" and self_check_dir is not None:
        from .selfcheck import budget_reason, earlier_hang

        blocked = earlier_hang(self_check_dir) or budget_reason(self_check_dir)
        if blocked:
            rep = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=seed, status="rejected", stage="screen")
            rep.message = blocked
            rep.elapsed_s = time.perf_counter() - t0
            return _live(rep)

    # 明显的死循环和非法 include 在这里结束，不起子进程，也不把核提交到 GPU。
    refused = backend.screen(kernel_source) if backend is not None else screen_source(kernel_source)
    if refused:  # never reaches a worker, let alone the GPU
        rep = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=seed, status="rejected", stage="screen")
        rep.message = "refused before compilation: " + refused[0]
        rep.abi = {"ok": False, "violations": refused}
        rep.elapsed_s = time.perf_counter() - t0
        rep.timeline = [{"stage": "screen", "status": "rejected", "duration_s": rep.elapsed_s}]
        if cache_key is not None:
            _cache_write(cache_key, rep)
        return _checked(rep)
    if progress_path is not None:
        try:
            from tcenv.events import publish

            publish(progress_path, {"task": task, "dtype": dtype, "size": size, "mode": mode, "stage": "screen", "status": "running", "updated_at": time.time()})
        except Exception:
            pass
    with tempfile.TemporaryDirectory(prefix="tcenv_src_") as td:
        kp = Path(workdir or td) / f"kernel{backend.file_ext if backend is not None else '.metal'}"
        kp.parent.mkdir(parents=True, exist_ok=True)
        kp.write_text(kernel_source)
        deadline = deadline_for_source(task, dtype, size, kernel_source)
        def _validation_pass() -> tuple[EvalReport | None, bool]:
            """Run the instrumented pass. Returns (a failure to stop on, whether the only complaint was the static limit)."""
            # 校验子进程在 bench 之前：打开会插桩访存和 barrier 的 shader validation，期限是这次 command buffer 时限的两倍。
            # 单核仍是 4 秒。流水线是「每段 2 秒、封顶 8 秒」的两倍。这次不过关就带着它的状态码返回。
            v = _run_worker(task, dtype, size, kp, "validate", seed, reps, timeout or 120.0, validation=True, cases=cases, launch_deadline_s=2 * deadline, progress_path=progress_path, require_contract_launch=require_contract_launch, measure_serial=measure_serial, target_id=target_id)
            # 校验插桩会把静态线程组内存报成大约两倍。只有这一条时，不拿它否决内核，改由下面不插桩的编译去比目标档案里的上限。
            from .backends.metal.abi import only_static_threadgroup_limit

            size_only = v.status == "abi_error" and only_static_threadgroup_limit((v.abi or {}).get("violations") or [])
            return (None if (v.status == STATUS_OK or size_only) else v), size_only

        from .target import load_target

        tg_limit = load_target(target_id).limits.max_threadgroup_memory_bytes
        size_only = False
        if mode == "full" or (mode == "timed" and validate):
            failed, size_only = _validation_pass()
            if failed is not None:
                failed.mode = "full" if mode == "full" else "timed"
                failed.elapsed_s = time.perf_counter() - t0
                return _checked(failed)
        if mode == "full":
            rep = _run_worker(task, dtype, size, kp, "full", seed, reps, timeout or 420.0, validation=False, cases=cases, progress_path=progress_path, fast_bench=fast_bench, launch_deadline_s=deadline, require_contract_launch=require_contract_launch, measure_serial=measure_serial, target_id=target_id)
            rep.safety = dict(rep.safety or {})
            rep.safety["validation"] = []
            rep.safety["shader_validation"] = (
                f"skipped: the validation pipeline's threadgroup size is the instrumented figure; the uninstrumented pipeline checked the {tg_limit} B limit"
                if size_only else "clean"
            )
        elif mode == "timed":
            # 和 full 同一套计时。校验只在调用方要求时跑（--timed --validate）：它慢，而且插桩会把线程组内存报成大约两倍。
            rep = _run_worker(task, dtype, size, kp, "full", seed, reps, timeout or 420.0, validation=False, cases=cases, progress_path=progress_path, fast_bench=fast_bench, launch_deadline_s=deadline, require_contract_launch=require_contract_launch, measure_serial=measure_serial, target_id=target_id)
            rep.mode = "timed"
            if validate:
                rep.safety = dict(rep.safety or {})
                rep.safety["validation"] = []
                rep.safety["shader_validation"] = (
                    f"skipped: the validation pipeline's threadgroup size is the instrumented figure; the uninstrumented pipeline checked the {tg_limit} B limit"
                    if size_only else "clean"
                )
        else:
            rep = _run_worker(task, dtype, size, kp, "quick", seed, reps, timeout or 240.0, validation=False, cases=cases, progress_path=progress_path, launch_deadline_s=deadline, require_contract_launch=require_contract_launch, measure_serial=measure_serial, target_id=target_id)
        rep.elapsed_s = time.perf_counter() - t0
        if cache_key is not None:
            _cache_write(cache_key, rep)
        return _checked(rep)
