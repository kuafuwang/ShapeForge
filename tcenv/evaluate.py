"""End-to-end evaluation of one candidate kernel (the TCEnv "execute, measure, select" loop).

Stages (Figure 2, steps 2-4 of the paper):
  compile  -> assemble into the ABI template, compile with the Metal compiler, create the pipeline
  abi      -> reflection check against the contract
  safety   -> guard bands, NaN-poisoned outputs, input immutability, determinism
  correct  -> randomized adversarial differential tests vs a CPU reference
  bench    -> interleaved benchmark against MLX / PyTorch-MPS baselines, roofline
  validate -> (separate subprocess) the same launch under Metal shader validation

Current v2 workers inject the projected task and execution into ``_evaluate_in_process``;
bounded worker supervision owns process cleanup and watchdog behavior.
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
from typing import Any, Mapping

from .bench import BENCH_ATTEMPTS, NOISY_CV
from .contract_errors import UnsupportedContractCapability
from .schema import (
    PROTOCOL_VERSION, REPORT_SCHEMA_VERSION, REQUEST_SCHEMA_VERSION, SchemaError,
    document_version, read_report_json, write_json_atomic,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

STATUS_OK = "ok"
# 安静机器上单次重复的 CV 远低于 ``NOISY_CV``（单次 0.5-1.5%）。超过它说明测速时还有别的负载，
# bench 会重测并保留最安静的一次。这两个常数只有一份定义（``tcenv.bench``）：v2 runtime
# 用同一套重试与选择规则，所以同一次 v1/v2 对照的中位数是同一个规则选出来的。
def _gpu_median_us(measured: Any) -> float | None:
    """@brief 被选中的那次 v2 计时里，候选臂计入统计的 GPU 时间中位数（微秒/迭代）。

    融合路径是该轮命令缓冲的 GPU 时长除以 k。Host 路径保留每一次 launch，不是最后
    一批的尾巴。逐 launch 提交用各缓冲自身时长之和再除以 k。校准、预热、对照臂、
    其它 attempt、覆盖探针和事后单独测量的 Torch 都不在这个元组里。没有样本时保持
    ``None``。缺失的时间戳不补 0。
    """
    samples = tuple(getattr(measured, "gpu_per_iter_s", ()) or ())
    if not samples:
        return None
    return float(statistics.median(samples) * 1e6)


def case_coverage(results: Any) -> dict[str, Any] | None:
    """@brief 由 correctness 结果聚合 L1 覆盖统计（固定 shape 实例返回 ``None``）。

    Only a varlen ``CaseResult`` carries ``case_realization``.  A case that is ``partial``,
    ``not_constructible``, ``deviates`` or ``unclassified`` is listed under its own kind instead
    of being counted as full coverage, so a successful report can tell a realised case from a
    replay of the canonical profile.

    @param results ``CaseResult`` 序列。
    @return ``{"cases": n, "exact": [...], ...}``；没有任何 realization 时为 ``None``。
    """
    records = [result.case_realization for result in results if result.case_realization is not None]
    if not records:
        return None
    return {
        "cases": len(records),
        **{
            kind: [record["case"] for record in records if record["kind"] == kind]
            for kind in ("exact", "partial", "not_constructible", "deviates", "unclassified")
        },
    }


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
    #: The one build record (§6.2): ``None`` or a closed ``partial``/``full`` object.
    #: Projected from ``BuildOutcome``/``BuildError`` in the process that holds them.
    build: dict | None = None
    #: Top-level evaluation identity (§7.2).  Required keys, hex or ``null``; never part of
    #: ``build`` and never folded into ``artifact_set_digest``.
    evaluation_fingerprint: str | None = None
    baseline_implementation_digest: str | None = None
    baseline_environment_digest: str | None = None
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

    # 读回子进程写的 JSON。只接受 schema 4；旧版本（含没有 schema_version 的版本 0）一律拒绝，不迁移、
    # 不补默认值。重复键必须在读取层先拒绝：dict 已经不记得某个键出现过两次。
    @staticmethod
    def from_json(d: dict) -> "EvalReport":
        """@brief 从 JSON 字典构造报告（只读 schema 4，闭集校验）。

        Only schema 4 is read.  There is no migration and no default status: a v0–v3 report,
        a missing/duplicate ``build``, a top-level key this version does not know (including the
        retired ``simulated``) or a status/shape combination outside the closed table is refused.

        @param d 已由拒绝重复键的 loader 解析出的报告字典。
        @return 构造出的 ``EvalReport``。
        @raises SchemaError 版本、字段闭集、状态或 ``build`` 非法时抛出。
        """
        from .build_projection import validate_for_status
        from .remote.job_types import CANDIDATE_STATUSES

        if not isinstance(d, dict):
            raise SchemaError("report is not a JSON object")
        version = document_version(d, max_version=REPORT_SCHEMA_VERSION, what="report")
        if version != REPORT_SCHEMA_VERSION:
            raise SchemaError(
                f"report schema_version {version} is not the schema this process reads "
                f"({REPORT_SCHEMA_VERSION}); old reports are not migrated"
            )
        known = {f.name for f in fields(EvalReport)}
        unknown = set(d) - known - {"schema_version", "eligible"}
        if unknown:
            raise SchemaError(f"report has fields this process does not know ({', '.join(sorted(unknown))})")
        missing = known - set(d)
        if missing:
            raise SchemaError(f"report is missing {sorted(missing)}")
        status = d["status"]
        if status not in CANDIDATE_STATUSES:
            raise SchemaError(f"report status {status!r} is not a persisted candidate status")
        if "eligible" in d and (
            not isinstance(d["eligible"], bool) or d["eligible"] != (status == STATUS_OK)
        ):
            raise SchemaError("report eligible must be a boolean equal to status == 'ok'")
        for name in ("evaluation_fingerprint", "baseline_implementation_digest", "baseline_environment_digest"):
            value = d[name]
            if value is not None and (
                not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise SchemaError(f"report {name} must be 64 lowercase hex characters or null")
        body = {key: value for key, value in d.items() if key not in ("schema_version", "eligible")}
        body["build"] = validate_for_status(d["build"], status=status)
        return EvalReport(**body)


def rejected_report(
    task: str,
    dtype: str,
    size: str,
    mode: str,
    *,
    stage: str,
    message: str,
    seed: int = 0,
    target_id: str = "",
    protocol_version: str = "",
) -> EvalReport:
    """@brief The one factory for a ``rejected`` report (no build, no GPU).

    Two entry points produce a rejected outcome for the same reason -- a delivery that cannot be a
    candidate: the request boundary (an ``EvaluationRequest`` exists and the preflight refused it)
    and the session boundary (the submitted bytes are not even decodable, so no request exists).
    Both must shape the record identically, so neither builds one by hand.

    @param task task name.
    @param dtype element type.
    @param size size label.
    @param mode evaluation mode the caller asked for.
    @param stage the stage that refused the candidate (``delivery`` for a submission defect).
    @param message actionable diagnostic for the model.
    @param seed input seed used for the request (0 when there was no request).
    @param target_id target profile id.
    @param protocol_version §7.2 protocol version tag.
    @return a ``rejected`` report with an empty build slot.
    """
    return EvalReport(
        task=task,
        dtype=dtype,
        size=size,
        mode=mode,
        seed=int(seed),
        stage=str(stage),
        status="rejected",
        message=str(message)[:2000],
        target_id=target_id,
        protocol_version=protocol_version,
    )


# ---------------------------------------------------------------------------
# in-process evaluation (runs inside the worker)
# ---------------------------------------------------------------------------


# 子进程里真正开始评的入口。先拿共享 GPU 锁：不计时的关卡可以和别的不计时工作重叠，bench 段再升级成独占，避免两份墙钟叠在一起把 CV 抬高。


# 关卡都在这个函数里。开头的 verify 只记账，失败与否都不看它。








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
    validate: bool = False,
    measure_serial: bool = False,
    target_id: str = "apple-m4-max",
    task_obj: Any | None = None,
    build_identity: dict[str, str] | None = None,
    execution: Any | None = None,
    build_outcome: Any | None = None,
    batch_submission: str = "single_buffer",
    control: bool = False,
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
    @param validate 是否为着色器校验那一趟。
    @param measure_serial 是否额外测串行 dispatch。
    @param target_id 目标设备 id。
    @param task_obj 当前 v2 调用方注入的评测 Task；不允许在 worker 内重新装题。
    @param build_identity v2 的 opspec/lowering 构建身份；省略时取 builder 的 Specialization。
    @param execution A4 的 v2 runtime executor；提供真实反射绑定、read_write/workspace 协议和 dispatch 记录。
    @param build_outcome A4 已构建并附加冻结几何的 BuildOutcome；提供时它才是本进程实际执行的
        构建，报告必须投影这一对象，不能用另一个 build 覆盖。
    @param batch_submission 候选批量提交方式；非默认实验路径会进入 protocol_version 与 benchmark 证据。
    @return 本次评测的报告。
    """
    import torch  # noqa: F401  (imported early so failures are reported as crashes of the worker)

    from .bench import attempt_noise, interleaved, select_attempt
    from .gpulock import gpu_lock

    t_start = time.perf_counter()
    timeline: list[dict] = []
    stage_name = "device_check"
    stage_started = t_start
    if task_obj is None or execution is None:
        raise UnsupportedContractCapability(
            "v2 in-process evaluation requires the caller's injected task_obj and execution; "
            "there is no task-registry fallback inside the worker"
        )
    task = task_obj
    seed = int(seed if seed is not None else secrets.randbits(48))
    from .schema import PROTOCOL_VERSION

    if batch_submission not in ("single_buffer", "per_launch_set_event"):
        raise ValueError(f"unknown batch submission {batch_submission!r}")
    protocol_version = PROTOCOL_VERSION if batch_submission == "single_buffer" else f"{PROTOCOL_VERSION}+{batch_submission}"
    rep = EvalReport(task=task_name, dtype=dtype, size=size, mode=mode, seed=seed, target_id=target_id, protocol_version=protocol_version)
    from .verify import verify

    rep.verification = verify(task, kernel_source).to_json()  # formal equivalence: unsupported for MSL (tcenv.verify)
    from .target import TargetError, require_active

    def reject_device(message: str) -> EvalReport:
        rep.stage = "device_check"
        rep.status = "rejected"
        rep.message = message
        rep.elapsed_s = time.perf_counter() - t_start
        rep.timeline = [{"stage": "device_check", "status": "rejected", "duration_s": rep.elapsed_s}]
        return rep

    try:
        profile = require_active(target_id)
    except TargetError as exc:
        return reject_device(str(exc))
    contract = task.contract
    from .backends.registry import load_backend

    backend = load_backend(profile.backend)
    ctx = backend.context()
    # Composition root: build and inject any internal kernel this backend's runtime needs
    # (Metal's zero-fill) through the registered factory, before the runtime is used.
    prepare = getattr(backend, "prepare_context", None)
    if callable(prepare):
        prepare(ctx)
    try:
        device = backend.probe(ctx, profile)
    except TargetError as exc:
        return reject_device(str(exc))
    dev_info = device.raw
    # §7.2: the three top-level evaluation identities are computed here, in the measuring
    # process.  The parent computes the same functions for the cache key/expected; a null
    # environment component makes the evaluation fingerprint null (unknown, never equality).
    rep.backend = profile.backend
    rep.evaluation_fingerprint = _evaluation_fingerprint(task, profile.backend, target_id)
    rep.baseline_implementation_digest = baseline_implementation_digest(task, profile.backend)
    rep.baseline_environment_digest = baseline_environment_digest(task, profile.backend, target_id)
    rep.backend_version = profile.backend_version
    rep.device_name = str(dev_info.get("name") or "")
    rep.gpu_family = str(dev_info.get("gpu_family") or "")
    if not device.matches:
        return reject_device(device.match_error or f"device {rep.device_name} does not match target {profile.id}")
    rep.device_warnings = list(device.warnings)

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

    # -- static screen (before anything touches the device) ------------------------
    rep.stage = "screen"
    _enter("screen")
    _note_stage(stage="screen", status="running", task=task_name, dtype=dtype, size=size, mode=mode)
    # The v2 geometry was already resolved by evaluate_task; the adapter's launch block is only
    # the source-side view used to cross-check workspace sizes, not the geometry authority.
    refused = execution.screen() if hasattr(execution, "screen") else []
    if refused:
        rep.abi = {"ok": False, "violations": list(refused)}
        return finish("rejected", "refused before compilation: " + refused[0])
    spec = execution.launch_spec

    # -- compile -------------------------------------------------------------
    rep.stage = "compile"
    _enter("compile")
    _note_stage(stage="compile", status="running", task=task_name, dtype=dtype, size=size, mode=mode)
    # One build path: the evaluator asks the backend for its registered
    # ArtifactBuilder and calls the neutral build entry point.  Metal's builder
    # assembles the contract header and calls ``MetalContext.compile`` itself;
    # no production code here (or elsewhere) calls the backend/runtime compile
    # primitive directly.  The candidate bytes are the SOURCE artifact identity,
    # so ``candidate_sha256`` keeps describing exactly the text the model wrote.
    from .remote.artifact_builder import BuildError, SourceBundle

    kernel_names = execution.kernel_names
    builder = backend.artifact_builder(
        kernel_contract=contract,
        specialization=task.specialization,
        math=execution.math,
        kernel_names=kernel_names,
    )
    # The builder registers what kind of build it makes; the projector takes it from there, never
    # from the request or the model.  The lowering identity comes from the same Specialization the
    # builder was constructed with (§6.2/§6.3).
    from .build_projection import project_full, project_partial

    build_kind = getattr(builder, "build_kind", None)
    identity = build_identity if build_identity is not None else getattr(builder, "lowering_identity", None)
    if build_kind not in ("native", "simulator", "source_only"):
        raise BuildError("toolchain_profile", "the builder does not declare a registered build_kind")
    if build_outcome is not None:
        # The adapter already compiled this exact candidate and attached the frozen geometry.
        # Use that object as the actual build; do not compile a second, different outcome and
        # then overwrite its report identity later.
        outcome = build_outcome
    else:
        try:
            outcome = builder.build(
                SourceBundle(kernel_source.encode("utf-8")), task_contract=contract,
            )
        except BuildError as e:
            if e.error_class == "compile":
                # §5.1: only a *projectable* compile failure is the candidate's.  A builder that did
                # not carry the typed provenance/assembled digest is a build implementation or
                # infrastructure fault; blaming the candidate would fabricate a partial projection.
                from .remote.artifact_builder import require_build_error_projection

                try:
                    # ``require_assembled=True``: this path encodes a Python ``str`` itself, so a
                    # build that reached the compiler always has assembled bytes; a null digest here
                    # is a builder defect, not a candidate failure (§5.1).
                    require_build_error_projection(e, require_assembled=True)
                except ValueError as exc:
                    raise BuildError(
                        "toolchain_profile",
                        f"the builder's compile failure is not projectable: {exc}",
                    ) from e
            if e.error_class == "deadline":
                # A build that ran out of time is a timeout, not a crash and not a candidate compile
                # error (plan §5.3-3).  Local callers pass ``deadline_monotonic=None``, so this only
                # fires for a caller that asked for a deadline; the mapping must exist *before* one
                # does, otherwise the worker's exception handler would record it as ``crash``.
                return finish("timeout", f"the build did not finish before the deadline: {e.detail}"[:400])
            if e.error_class != "compile":
                # Toolchain/profile failures are environment failures; the legacy
                # evaluator let those propagate rather than blaming the candidate.
                raise
            # The builder already rewrote compiler line numbers against the contract
            # header.  The report keeps the legacy text shape: the rewritten
            # diagnostic only, not the exception class prefix.
            rep.compile = {"ok": False, "errors": e.report_detail}
            # Only the attachments above are projected; the diagnostic text stays in ``compile``.
            # ``project_partial`` re-checks the attachments and the shape, so a partial can never
            # claim an artifact set, evidence or a native executable.
            rep.build = project_partial(e, build_kind=build_kind, identity=identity)
            return finish("compile_error", "Metal compiler rejected the kernel")

    # The one successful build record: projected from the neutral outcome while this process
    # still holds it.  A later abi/correctness/runtime failure keeps it (the build really ran).
    rep.build = project_full(outcome, build_kind=build_kind, identity=identity)
    kernels = list(outcome.executable_handles())
    if not kernels:
        raise RuntimeError("the backend builder returned no executable handle")
    kernel = kernels[0]
    rep.compile = {
        "ok": True,
        "compile_s": kernel.compile_s,
        "pipeline_s": sum(k.pipeline_s for k in kernels),
        "language_version": kernel.language_version,
        "math_mode": spec.math if spec is not None else "fast",
        "launch": None if spec is None else spec.digest(),
    }

    if execution is not None:
        # The v2 branch is the A4 seam: reflection-derived per-stage bindings, real runtime
        # dispatch recording and the A2 read_write/workspace protocols replace the legacy
        # Contract/ArgSet checks for this candidate.
        from .contract_errors import GeometryError

        rep.stage = "abi"
        _enter("abi")
        try:
            runtime = execution.bind(ctx, kernels)
        except GeometryError as exc:
            rep.abi = {"ok": False, "violations": [str(exc)]}
            return finish("abi_error", "kernel does not match the required v2 ABI / launch contract")
        rep.abi = {"ok": True, "violations": []}
        rep.pipeline = {"kind": "v2", "stages": len(execution.geometry.active_stages)}
        rep.stage = "correctness"
        _enter("correctness")
        try:
            results, safety, nondet, dispatch, host = execution.correctness(
                ctx, runtime, seed=seed, cases=cases, determinism=(mode != "validate"),
            )
        except (execution.runtime_error(), GeometryError) as exc:
            rep.safety = {"ok": False, "violations": [f"GPU/v2 execution error: {exc}"], "nondeterminism": [], "validation": []}
            return finish("runtime_error", f"the v2 runtime reported an error while running the kernel: {str(exc)[:300]}")
        rep.correctness = {
            "ok": all(r.ok for r in results),
            "cases": [r.to_json() for r in results],
            "tolerance": {"rtol": task.rtol, "atol_rel": task.atol_rel, "exact": task.exact},
            "baseline_errors": [],
            "coverage": {"kind": "v2-runtime", "scope": "real Metal reflection/read_write/workspace path"},
        }
        # L1 覆盖统计：把每个 case 的 realization 按种类聚合。一个 partial/not_constructible
        # 的 case 不算该 case 的完整覆盖；deviates 是 task planner 的缺陷信号。固定 shape 实例
        # 没有 realization，这个键不出现，旧证据形状不变。
        coverage = case_coverage(results)
        if coverage is not None:
            rep.correctness["coverage"]["case_coverage"] = coverage
        rep.safety = {"ok": not safety and not nondet, "violations": list(safety), "nondeterminism": list(nondet), "validation": []}
        execution.record(dispatch=dispatch, host=host)
        if safety or nondet:
            return finish("unsafe", (list(safety) + list(nondet))[0])
        if not rep.correctness["ok"]:
            first_bad = next(r for r in results if not r.ok)
            return finish("incorrect", f"case '{first_bad.case}': {first_bad.message}")
        if mode in ("validate", "quick"):
            return finish(STATUS_OK, "compiled, ABI ok, safe and correct")
        rep.stage = "bench"
        _enter("bench")
        # The v2 measurement uses the same boundary as v1: the task's registered baselines are
        # built by the same backend factory and measured inside the same interleaved call, so the
        # candidate median is taken under the same GPU state and chosen by the same attempt rule.
        base_runners = {
            name: runner.run_k
            for name, runner in backend.baseline_runners(task, backend.prepare_case(task, seed + 1)).items()
        }
        try:
            measured = execution.benchmark(
                ctx, runtime, seed=seed, reps=reps, baseline_runners=base_runners,
                control=bool(control),
            )
        except (execution.runtime_error(), GeometryError) as exc:
            return finish("runtime_error", f"the v2 runtime reported an error while benchmarking: {str(exc)[:300]}")
        execution.record(dispatch=measured.dispatch, host=measured.host)
        if measured.guard_violations:
            rep.safety = {"ok": False, "violations": list(measured.guard_violations), "nondeterminism": [], "validation": []}
            return finish("unsafe", list(measured.guard_violations)[0])
        stats = measured.stats
        rep.candidate_us = stats.median_us
        rep.benchmark = {
            "candidate": stats.to_json(),
            "candidate_gpu_us_median": _gpu_median_us(measured),
            # 同窗口 A/A 对照：同一份候选字节在同一批交错里再启动一次。未启用时为 None，
            # 报告因此能说明这次数字是不是配对的，而不是让人以为它和冻结种子可比。
            "control": getattr(measured, "control", None),
            "baselines": {name: value.to_json() for name, value in measured.baselines.items()},
            "candidate_serial_us": None,
            "measure_serial": False,
            "dispatch": "v2 runtime recorder, concurrent encoder as MLX",
            "rotation_sets": measured.rotation_sets,
            "noise_cv": measured.noise_cv,
            "attempts": measured.attempts,
            "noisy": measured.noise_cv > NOISY_CV,
            "gpu_lock_wait_s": 0.0,
            "protocol": (
                "v2 real runtime, interleaved with the registered baselines; read_write restored per trial, "
                "only declared-zero workspace cleared in timing"
            ),
        }
        extended_attempts = getattr(measured, "extended_attempts", None)
        if extended_attempts:
            # 没有额外尝试时不写这两个键，旧报告形状不变。attempts 本身已是实际次数。
            rep.benchmark["base_attempts"] = int(measured.base_attempts)
            rep.benchmark["extended_attempts"] = int(extended_attempts)
        if batch_submission != "single_buffer":
            actual_submission = str(getattr(measured, "batch_submission", "single_buffer_fallback"))
            rep.benchmark["batch_submission_requested"] = batch_submission
            rep.benchmark["batch_submission"] = actual_submission
            if actual_submission == "per_launch_set_event":
                rep.benchmark["dispatch"] = "v2 runtime recorder, one command buffer per launch, per-set SharedEvent dependencies"
                rep.benchmark["protocol"] += "; per-launch command buffers with per-set SharedEvent dependencies"
            else:
                rep.benchmark["protocol"] += "; per-launch submission was requested but the candidate used the single-buffer fallback"
        rep.baseline_us = {name: value.median_us for name, value in measured.baselines.items()}
        scope = str(getattr(measured, "measurement_scope", "single_shape"))
        batches = list(getattr(measured, "batches", ()) or ())
        if batches:
            # The ordered resolved batches the timed mixture actually ran (L2).  The report is where a
            # reviewer looks, so the sequence and the per-batch input identity travel with it.  A
            # declaration whose batches all resolve to one shape *is* comparable to the single-shape
            # baseline (that is the case ``speedup_policy`` publishes a ratio for), so the flag follows
            # the scope instead of being fixed to False.
            mixture = scope != "single_shape"
            rep.benchmark["continuous_batching"] = {
                "mixture": mixture,
                "measurement_scope": scope,
                "baseline_scope": "single_shape",
                "comparable_single_shape_measurement": not mixture,
                "batches": [
                    {
                        "ordinal": int(batch.get("ordinal", index)),
                        "case": str(batch.get("case")),
                        "provider": str(batch.get("provider")),
                        "resolved": {
                            "total_tokens": int((batch.get("resolved_shape") or {}).get("total_tokens", 0)),
                            "num_seqs": int((batch.get("resolved_shape") or {}).get("num_seqs", 0)),
                            "full_capacity": bool((batch.get("resolved_shape") or {}).get("full_capacity")),
                        },
                        "evaluation_input_digest": batch.get("evaluation_input_digest"),
                    }
                    for index, batch in enumerate(batches)
                ],
            }
        policy = speedup_policy(
            {name: value.median_s / stats.median_s for name, value in measured.baselines.items()},
            rep.baseline_us,
            scope=scope,
        )
        if policy["withheld"] is not None:
            # A mixture median covers several resolved batches while the registered baselines are measured
            # on one static shape.  Publishing their ratio would credit the candidate with the cheaper
            # small batches; the numbers stay in the report, the *ratio* is not published.
            rep.benchmark["measurement_scope"] = scope
            rep.benchmark["baseline_scope"] = "single_shape"
            rep.benchmark["speedup_withheld"] = policy["withheld"]
        else:
            rep.speedups = dict(policy["speedups"])
            if policy["speedup"] is not None:
                rep.speedup = float(policy["speedup"])
                rep.benchmark["best_baseline"] = policy["best_baseline"]
        rep.roofline = backend.roofline(task, stats.median_s, backend.calibration(target_id))
        return finish(STATUS_OK, "compiled, ABI ok, safe, correct and measured")


LAUNCH_DEADLINE_S = 2.0  # a legitimate single launch is milliseconds; the watchdog is fail-closed
COMPILE_DEADLINE_S = 90.0  # a stuck compiler must not hold the worker indefinitely
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


def deadline_for_source(task: str, dtype: str, size: str, source: str, *, backend: Any = None) -> float:
    """@brief 按候选源码决定本次 command buffer 的等待时限。

    The command-buffer wait for this source. An unreadable launch block stays at 2 seconds.

    The worker rejects that block before the GPU. A parsed pipeline uses one wait for every stage.

    @param task 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @param source 候选 MSL 源码。
    @return 等待秒数。
    """
    from types import SimpleNamespace

    from taic.task_binding import bind_task

    from .backends.base import LaunchRejected

    try:
        binding = bind_task(task, dtype, size)
        instance = binding.contract.instance
        constants = [(str(name), "uint", int(value)) for name, value in instance.symbols.items()]
        constants += [
            (str(name), "uint", int(value)) for name, value in instance.resolved_attrs.items()
            if isinstance(value, int) and not isinstance(value, bool)
        ]
        contract = SimpleNamespace(constants=constants)
    except Exception:  # noqa: BLE001 - an unknown task still has the single-launch wait
        return LAUNCH_DEADLINE_S
    if backend is None:
        # Compatibility for Metal profiling callers.  The generic evaluator
        # always supplies its already-resolved backend.
        from .backends.registry import load_backend

        backend = load_backend("metal")
    try:
        return command_buffer_deadline_s(backend.launch_stage_count(source, contract))
    except LaunchRejected:
        return LAUNCH_DEADLINE_S
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



EVAL_CACHE_SCHEMA = 3




def _registered_identity(entry, role: str) -> dict:
    """@brief 取一个角色的**登记**实现身份（缺登记或字段非法立即失败）。

    The evaluation identity is ``{implementation_id, implementation_version, params}``: a
    closure's ``repr`` (its source text, defaults, closed-over values) is not an identity — it
    changes with the wording of the code, and two structurally different callables can print
    the same text.  A task without a registration cannot be evaluated into a cache entry.

    @param entry track 登记块。
    @param role 角色名（诊断用）。
    @return 可 canonical 化的身份字典。
    @raises EvaluationError 未登记、版本非法或 params 不可 canonical 化。
    """
    from .evaluation import EvaluationError

    if not isinstance(entry, dict):
        raise EvaluationError(f"evaluation identity: {role} is not registered")
    implementation_id = str(entry.get("implementation_id") or "").strip()
    version = entry.get("implementation_version")
    params = entry.get("params")
    if not implementation_id:
        raise EvaluationError(f"evaluation identity: {role} has no implementation_id")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise EvaluationError(f"evaluation identity: {role} has an illegal implementation_version {version!r}")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise EvaluationError(f"evaluation identity: {role} params must be an object")
    try:
        # Round-trip through the canonical encoder so a non-JSON value is refused here rather
        # than silently stringified into the key.
        encoded = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"evaluation identity: {role} params are not canonical: {exc}") from exc
    return {"implementation_id": implementation_id, "implementation_version": version, "params": json.loads(encoded)}


#: The framework registry: for each baseline framework, the module to probe and the
#: distribution whose version is declared.  It is part of the evaluator, not the task, so a
#: task cannot silently declare a version of something the host never checks.
FRAMEWORK_LIBRARIES: dict[str, tuple[str, str]] = {
    "mlx": ("mlx", "mlx"),
    "torch": ("torch", "torch"),
}

#: Version of the baseline registration itself; it enters the implementation digest.
BASELINE_REGISTRY_VERSION = 1


def _baseline_registrations(task) -> list[dict]:
    """@brief 取该任务**完整**的 baseline 登记表（不因 ``fast_bench`` 缩水）。

    The complete ``Task.baselines`` registry, sorted by ``(name, framework, kind)`` so both
    sides of a comparison hash the same list.

    @param task 任务。
    @return 排序后的登记行。
    @raises EvaluationError 没有 baseline 登记。
    """
    from .evaluation import EvaluationError

    registrations = dict(task.evaluation_implementations or {})
    baselines = registrations.get("baselines") or []
    if not isinstance(baselines, list):
        raise EvaluationError(f"task {task.name}: baseline identities must be a list")
    # An empty registry is a real identity (the semantic test family has no baselines); the
    # digest is over the empty list rather than an error or a fabricated row.
    rows = []
    for entry in baselines:
        name = str(entry.get("name"))
        framework = str(entry.get("framework"))
        if framework not in FRAMEWORK_LIBRARIES:
            raise EvaluationError(
                f"task {task.name}: baseline {name!r} declares unregistered framework {framework!r}"
            )
        rows.append({
            "name": name,
            "framework": framework,
            "kind": "task",
            "implementation": _registered_identity(entry, f"baseline {name}"),
        })
    rows.sort(key=lambda row: (row["name"], row["framework"], row["kind"]))
    return rows


def _canonical_digest(document) -> str:
    """@brief 按 §3.1 的 canonical 值编码哈希一个身份文档。

    The one §3.1 encoder (``tcenv.identity.fingerprint``) normalises floats to their shortest
    round-trip decimal text first; a bare ``canonical_digest_document`` rejects them.

    @param document 可 canonical 化的 JSON 文档。
    @return 十六进制 sha256。
    """
    from . import identity

    return identity.fingerprint(document)


def baseline_implementation_digest(task, backend_id: str = "metal") -> str:
    """@brief baseline **实现**摘要：登记身份，不看宿主是否安装。

    The implementation identity is what the task pack registers; whether the library is
    installed on this host belongs to the environment digest and never changes this one.

    @param task 任务。
    @param backend_id 后端 id。
    @return 十六进制 sha256。
    @raises EvaluationError 登记缺失或不可 canonical 化。
    """
    return _canonical_digest({
        "fingerprint_kind": "baseline_implementation",
        "fingerprint_version": 1,
        "registry_version": BASELINE_REGISTRY_VERSION,
        "backend_id": backend_id,
        "baselines": _baseline_registrations(task),
    })


def baseline_environment_digest(task, backend_id: str = "metal", target_id: str = "apple-m4-max") -> str | None:
    """@brief baseline **环境**摘要：宿主上的可用性与库版本。

    Availability uses ``importlib.util.find_spec`` and the version comes from
    ``importlib.metadata.version``; no framework is imported and no device is opened.  An
    installed framework with an unreadable version makes this ``None`` (unknown), which also
    makes the evaluation fingerprint ``None``: it is not evidence of equality.

    @param task 任务。
    @param backend_id 后端 id。
    @param target_id 目标设备 id。
    @return 十六进制 sha256；版本不可识别时为 ``None``。
    @raises EvaluationError 登记缺失。
    """
    import importlib.metadata
    import importlib.util

    rows = []
    for row in _baseline_registrations(task):
        module_name, distribution = FRAMEWORK_LIBRARIES[row["framework"]]
        try:
            available = importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):
            available = False
        library = None
        if available:
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                return None
            except Exception:  # noqa: BLE001 - an unreadable version is unknown, not equality
                return None
            if not isinstance(version, str) or not version:
                return None
            library = {"name": distribution, "version": version}
        rows.append({
            "name": row["name"],
            "framework": row["framework"],
            "kind": row["kind"],
            "available": available,
            "library": library,
        })
    return _canonical_digest({
        "fingerprint_kind": "baseline_environment",
        "fingerprint_version": 1,
        "backend_id": backend_id,
        "target_id": target_id,
        "baselines": rows,
    })


#: "The caller did not override this component" — distinct from an explicit ``None``, which means
#: "this component is unknown".  Conflating them would let a remote expected silently fall back to
#: the host's own library versions, which is exactly what plan §7.2 forbids.
_UNSET = object()


def speedup_policy(
    speedups: Mapping[str, float],
    baseline_us: Mapping[str, float],
    *,
    scope: str,
) -> dict:
    """@brief 决定一次测量能否发布 speedup（mixture 与单形状基线不可比）。

    The v1/v2 protocol divides the candidate median by a registered baseline median.  That is a
    like-for-like ratio only when both ran the same workload.  An L2 continuous-batching mixture runs
    several resolved batches while the registered baselines are single-shape, so dividing them would
    credit the candidate with the cheaper small batches.  The numbers stay in the report (``baseline_us``
    and the candidate median) for diagnosis; the *ratio* is withheld with a machine-readable reason.

    @param speedups 各基线名 → 基线中位数 / 候选中位数。
    @param baseline_us 各基线名 → 基线中位微秒（用于选最快基线）。
    @param scope ``measured.measurement_scope``（``single_shape`` 或
        ``continuous_batching_mixture``）。
    @return ``{"speedups", "speedup", "best_baseline", "withheld"}``。
    """
    if not speedups:
        return {"speedups": {}, "speedup": None, "best_baseline": None, "withheld": None}
    if scope != "single_shape":
        return {
            "speedups": {},
            "speedup": None,
            "best_baseline": None,
            "withheld": (
                "the timed mixture runs several resolved batches (continuous batching); the registered "
                "baselines are single-shape, so a mixture/baseline ratio is not like-for-like. Re-measure a "
                "like-for-like baseline before publishing a speedup."
            ),
        }
    best = min(baseline_us, key=lambda name: baseline_us[name])
    return {
        "speedups": dict(speedups),
        "speedup": float(speedups[best]),
        "best_baseline": best,
        "withheld": None,
    }


def _evaluation_fingerprint(
    task, backend_id: str = "metal", target_id: str = "apple-m4-max", *,
    task_pack_digest=_UNSET, environment_digest=_UNSET,
) -> str | None:
    """@brief 计算影响评测结果的可执行语义指纹：登记身份 + task-pack 摘要。

    Registered identities only (no closure ``repr``): the task fingerprint, the task-pack
    digest, the registered gen/reference/error-scale implementations and both baseline
    digests.  ``None`` means the host side of the identity could not be established, which is
    ``evaluation_identity_unknown`` at confirmation time, never a match.

    The two keyword overrides exist for the *remote* confirmation expected (plan §7.2): the pack
    component then comes from the validated evaluator provenance (the deployed pack the measurement
    ran with) and the environment component from the **frozen** search-period value, so the host
    never substitutes its own library versions.  Both default to the locally recomputed values.

    @param task 任务。
    @param backend_id 后端 id。
    @param target_id 目标设备 id。
    @param task_pack_digest 覆盖 pack 分量；``None`` 时用本机登记的 task-pack 摘要。
    @param environment_digest 覆盖环境分量；``None`` 时用本机安装的库版本重算。
    @return 十六进制 sha256 或 ``None``（环境分量不可识别）。
    @raises EvaluationError 任务没有登记评测实现或 task-pack 身份。
    """
    from .evaluation import EvaluationError
    from .task import task_fingerprint

    registrations = dict(task.evaluation_implementations or {})
    if not registrations:
        raise EvaluationError(
            f"task {task.name}: no registered evaluation implementations; a closure repr is not an identity"
        )
    pack = registrations.get("task_pack")
    if not isinstance(pack, dict) or not str(pack.get("digest") or ""):
        raise EvaluationError(f"task {task.name}: the task-pack digest is not registered")
    implementation = baseline_implementation_digest(task, backend_id)
    environment = (
        baseline_environment_digest(task, backend_id, target_id)
        if environment_digest is _UNSET else environment_digest
    )
    if environment is None:
        return None
    resolved_pack = str(pack["digest"]) if task_pack_digest is _UNSET else task_pack_digest
    if not resolved_pack:
        # An explicitly unknown pack digest is not a licence to use the host's registered pack.
        return None
    return _canonical_digest({
        "fingerprint_kind": "evaluation",
        "fingerprint_version": 1,
        "task_fingerprint": task_fingerprint(task, backend_id),
        "task_pack_digest": resolved_pack,
        "gen_inputs": _registered_identity(registrations.get("gen_inputs"), "gen_inputs"),
        "reference": _registered_identity(registrations.get("reference"), "reference"),
        "error_scale": (
            None
            if registrations.get("error_scale") is None
            else _registered_identity(registrations.get("error_scale"), "error_scale")
        ),
        "baseline_implementation_digest": implementation,
        "baseline_environment_digest": environment,
    })


















# 搜索进程调用的入口。Metal 编译和启动发生在子进程里，图形芯片卡死不会拖垮还在搜下一批候选的进程。
# 单次启动超时由 supervise 看见心跳 B 后 SIGKILL；整段时限另算，等锁的 W 不计。
