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

from .bench import BENCH_ATTEMPTS, NOISY_CV
from .schema import (
    PROTOCOL_VERSION, REPORT_SCHEMA_VERSION, REQUEST_SCHEMA_VERSION, SchemaError,
    document_version, read_report_json, write_json_atomic,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

STATUS_OK = "ok"
# 安静机器上单次重复的 CV 远低于 ``NOISY_CV``（单次 0.5-1.5%）。超过它说明测速时还有别的负载，
# bench 会重测并保留最安静的一次。这两个常数只有一份定义（``tcenv.bench``）：v2 runtime
# 用同一套重试与选择规则，所以同一次 v1/v2 对照的中位数是同一个规则选出来的。
_CORRECTNESS_CACHE: dict[tuple, dict] = {}
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
    task_obj: Any | None = None,
    build_identity: dict[str, str] | None = None,
    execution: Any | None = None,
    build_outcome: Any | None = None,
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
    @param task_obj 已装配的 Task（v2 评测注入）；省略时按名字走 v1 registry。
    @param build_identity v2 的 opspec/lowering 构建身份；省略时取 builder 的 Specialization。
    @param execution A4 的 v2 runtime executor；提供真实反射绑定、read_write/workspace 协议和 dispatch 记录。
    @param build_outcome A4 已构建并附加冻结几何的 BuildOutcome；提供时它才是本进程实际执行的
        构建，报告必须投影这一对象，不能用另一个 build 覆盖。
    @return 本次评测的报告。
    """
    import torch  # noqa: F401  (imported early so failures are reported as crashes of the worker)

    from tasks import get_task

    from .bench import attempt_noise, interleaved, select_attempt
    from .gpulock import gpu_lock

    t_start = time.perf_counter()
    timeline: list[dict] = []
    stage_name = "device_check"
    stage_started = t_start
    task = get_task(task_name, dtype, size) if task_obj is None else task_obj
    seed = int(seed if seed is not None else secrets.randbits(48))
    from .schema import PROTOCOL_VERSION

    rep = EvalReport(task=task_name, dtype=dtype, size=size, mode=mode, seed=seed, target_id=target_id, protocol_version=PROTOCOL_VERSION)
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
    if execution is None:
        # 子进程里再过一遍同一条筛，拒绝理由放进 abi.violations，反馈才会印出原因。父进程那一次是为了连子进程都不起。
        refused = backend.screen(kernel_source, task)
        if refused:
            rep.abi = {"ok": False, "violations": refused}
            return finish("rejected", "refused before compilation: " + refused[0])

        from .backends.base import LaunchRejected

        try:
            spec = backend.parse_launch(kernel_source, contract)
            backend.validate_launch(
                spec,
                source=kernel_source,
                contract=contract,
                task=task,
                target=profile,
                device_info=dev_info,
                require_contract_launch=require_contract_launch,
            )
        except LaunchRejected as e:
            rep.abi = {"ok": False, "violations": [str(e)]}
            return finish("rejected", "refused before compilation: " + str(e))
        except MemoryError as e:
            return finish("rejected", f"refused before compilation: could not size the launch ({e})")
    else:
        # The v2 geometry was already resolved by evaluate_task; the adapter's launch block is
        # only the source-side view used to cross-check workspace sizes.  It is not the geometry
        # authority.  The v2 boundary is deliberately narrower than v1's loop/goto screen: v2
        # candidates own their control flow, while the worker owns runaway execution.
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

    kernel_names = (
        execution.kernel_names
        if execution is not None
        else (None if spec is None else list(dict.fromkeys(stage.kernel for stage in spec.stages)))
    )
    builder = backend.artifact_builder(
        kernel_contract=contract,
        specialization=task.specialization,
        math=(execution.math if execution is not None else ("fast" if spec is None else spec.math)),
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
            "candidate_gpu_us_median": None,
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
        rep.baseline_us = {name: value.median_us for name, value in measured.baselines.items()}
        rep.speedups = {name: value.median_s / stats.median_s for name, value in measured.baselines.items()}
        if rep.speedups:
            best = min(rep.baseline_us, key=lambda name: rep.baseline_us[name])
            rep.speedup = rep.speedups[best]
            rep.benchmark["best_baseline"] = best
        rep.roofline = backend.roofline(task, stats.median_s, backend.calibration(target_id))
        return finish(STATUS_OK, "compiled, ABI ok, safe, correct and measured")

    # -- abi -------------------------------------------------------------------
    rep.stage = "abi"
    _enter("abi")
    _note_stage(stage="abi", status="running", task=task_name, dtype=dtype, size=size, mode=mode)
    problems = backend.check_abi(
        contract, kernels,
        target=profile,
        device_info=dev_info,
        spec=spec,
        validation=(mode == "validate"),
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
                attempts.append((attempt_noise(st), st, gpu_samples[first:]))
                noisy = attempts[-1][0] > NOISY_CV
                _note_stage(stage="bench", status="running", attempt=attempt + 1, attempts_max=BENCH_ATTEMPTS, noisy=noisy)
                if not noisy:
                    break
        except backend.runtime_error() as e:
            return finish("runtime_error", f"GPU error during benchmarking: {str(e)[:300]}")
    # 留下第一次安静的尝试。在最多三次里挑 CV 最小的一次，是对估计量按它自己的离散度做选择，
    # 中位数会系统性偏向更安静的那一次；三次都吵时才退回到最安静的一次并标 noisy。
    # 选择规则在 ``tcenv.bench`` 里只有一份（v2 runtime 用的是同一个函数）。
    noise, stats, cand_gpu = select_attempt(attempts, noisy_cv=NOISY_CV)

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
    try:  # optional backend-owned compiler baseline (Triton→MSL on Metal)
        rep.compiler_baseline = backend.compiler_baseline(task_name, dtype, size, rep.baseline_us, cand.median_us)
    except Exception:  # noqa: BLE001 - optional information must never fail an evaluation
        # 拿不到后端编译器对照就跳过。它是附加信息，失败不能把已经通过的候选改判。
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
    from tasks import get_task

    from .backends.base import LaunchRejected

    try:
        contract = get_task(task, dtype, size).contract
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
    backend: Any = None,
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
    if backend is None:
        # Direct Metal-era callers remain valid, but target selection goes
        # through the registry rather than importing a Metal runtime here.
        from .backends.registry import load_backend
        from .target import load_target

        backend = load_backend(load_target(target_id).backend)

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
        # Metal removes inherited MTL_* instrumentation for timed workers and
        # enables it for validation workers; other backends own their own env.
        env = backend.worker_env(env, validation=validation)
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
                rep = EvalReport.from_json(read_report_json(out_path.read_text(encoding="utf-8"), what="worker report"))
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
            probs = backend.parse_validation_stderr(stderr)
            rep.safety = dict(rep.safety or {})
            rep.safety["validation"] = probs
            if probs and rep.status in (STATUS_OK, "incorrect", "unsafe", "runtime_error"):
                rep.safety["ok"] = False
                rep.status = "unsafe"
                rep.message = "Metal shader validation: " + probs[0]
        return rep



EVAL_CACHE_SCHEMA = 3


def _eval_cache_dir() -> Path:
    """@brief 评测缓存目录。

    @return ``TCENV_EVAL_CACHE_DIR`` 指定的目录，默认为用户缓存下的 ``evals``。
    """
    return Path(os.environ.get("TCENV_EVAL_CACHE_DIR", Path.home() / ".cache" / "ai_lowerling" / "evals"))


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


def remote_confirmation_expected(
    task: str, dtype: str, size: str, target_id: str, source: str, *,
    task_pack_digest: str | None,
    baseline_environment_digest: str | None,
    build_provenance: dict | None = None,
    device_group: dict | None = None,
    evaluator_epoch: str | None = None,
    candidate_digest: dict | None = None,
) -> dict:
    """@brief 远端确认 expected：host 重算实现分量，pack/环境分量取冻结值，绝不使用 host 的库版本。

    Plan §7.2: the host may recompute the task/implementation components (task fingerprint,
    registered gen/reference/error-scale identities, baseline *implementation* digest) because they
    come from the registered task pack.  It must not recompute the **environment** component — an
    evaluator that upgraded a baseline library between search and confirmation then looks equal to
    a host that never had that library.  So:

    * ``task_pack_digest`` comes from the validated evaluator provenance (the deployed pack);
    * ``baseline_environment_digest`` is the **frozen** search-period value (an evaluator that
      reports ``null`` makes the whole document unable to publish, which is ``unknown``);
    * ``build_provenance``/``device_group``/``evaluator_epoch``/``candidate_digest`` come from the
      current validated capability, never from the search report's self-reported fields.

    A missing host-recomputable component is a deployment error (``EvaluationError``), not a
    publishable unknown.

    @param task 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @param target_id 目标设备 id。
    @param source 当前候选源码。
    @param task_pack_digest 已校验 evaluator provenance 里的部署 pack 摘要；``None`` 表示未知。
    @param baseline_environment_digest 搜索期冻结的 baseline 环境摘要；``None`` 表示未知。
    @param build_provenance 当前 evaluator 的 BuildProvenance（或请求级 build 文档）；可为 ``None``。
    @param device_group 当前 evaluator 能力声明的 device group；可为 ``None``。
    @param evaluator_epoch 当前 evaluator epoch；可为 ``None``。
    @param candidate_digest 当前 evaluator 绑定的 CandidateDigest；可为 ``None``。
    @return 与 :func:`confirmation_expected_document` 同形状的 expected，另含远端分量。
    @raises EvaluationError 任务没有登记评测实现、gen/reference 身份或 task-pack 身份时。
    """
    import hashlib

    from tasks import get_task

    from .schema import PROTOCOL_VERSION

    task_obj = get_task(task, dtype, size)
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    registrations = dict(task_obj.evaluation_implementations or {})
    if not registrations:
        from .evaluation import EvaluationError

        raise EvaluationError(
            f"task {task}: no registered evaluation implementations; a closure repr is not an identity"
        )
    backend_id = "metal"
    backend_version = None
    try:
        from .backends.registry import load_backend
        from .target import load_target

        target = load_target(target_id)
        backend = load_backend(target.backend)
        backend_id = str(getattr(backend, "id", "") or backend_id)
        backend_version = str(getattr(backend, "version", "") or "") or None
    except Exception:
        backend_id, backend_version = "metal", None
    specialization = getattr(task_obj, "specialization", None)
    build: dict = {"candidate_sha256": source_sha256}
    if specialization is not None:
        build["opspec_digest"] = specialization.opspec_digest
        build["lowering_profile_digest"] = specialization.lowering_profile_digest
    # A request-level build document wraps the provenance; the report's ``build.provenance`` is
    # the inner document.  Both are accepted, and neither is copied out of the search report.
    inner = build_provenance.get("provenance") if isinstance(build_provenance, dict) else None
    build["provenance"] = dict(inner) if isinstance(inner, dict) else (
        dict(build_provenance) if isinstance(build_provenance, dict) else None
    )
    build["candidate_digest"] = dict(candidate_digest) if isinstance(candidate_digest, dict) else None
    build["device_group"] = dict(device_group) if isinstance(device_group, dict) else None
    build["build_provenance"] = dict(build_provenance) if isinstance(build_provenance, dict) else None
    evaluation = {
        "evaluation_fingerprint": _evaluation_fingerprint(
            task_obj, backend_id, target_id,
            task_pack_digest=task_pack_digest,
            environment_digest=baseline_environment_digest,
        ),
        "baseline_implementation_digest": baseline_implementation_digest(task_obj, backend_id),
        "baseline_environment_digest": baseline_environment_digest,
        "task_pack_digest": task_pack_digest,
        "protocol_version": PROTOCOL_VERSION,
        "target_id": target_id,
        "backend": backend_id,
        "backend_version": backend_version,
        "evaluator_epoch": evaluator_epoch,
        "device_group": dict(device_group) if isinstance(device_group, dict) else None,
    }
    return {
        "build": build,
        "evaluation": evaluation,
        "protocol_version": PROTOCOL_VERSION,
        "target_id": target_id,
    }


def confirmation_expected_document(
    task: str, dtype: str, size: str, target_id: str, source: str, *, probe: bool = False,
) -> dict:
    """@brief 当前任务、注册表与执行器能重算的确认 expected（不读任何报告）。

    The document never contains a value copied out of a report.  Build fields only a build can
    produce (``assembled_sha256``, ``artifact_set_digest``, ``native_digest``) are *omitted*.
    With ``probe=False`` the current BuildProvenance and device fields stay ``None``; with
    ``probe=True`` they come only from an *available* executor probe.  Outcome-only fields are
    compared between the search and confirmation reports, never filled from either report.
    The candidate's launch math is parsed from the same source before probing, so a safe-math
    candidate cannot accidentally get a fast-math expected profile.

    @param task 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @param target_id 目标设备 id。
    @param source 当前候选源码。
    @param probe 是否向执行器探测当前工具链 profile（真实运行使用；CPU 侧默认不探测）。
    @return ``{"build": ..., "evaluation": ..., "protocol_version": ..., "target_id": ...}``。
    """
    import hashlib

    from tasks import get_task

    from .schema import PROTOCOL_VERSION

    task_obj = get_task(task, dtype, size)
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    specialization = getattr(task_obj, "specialization", None)
    build: dict = {"candidate_sha256": source_sha256}
    if specialization is not None:
        build["opspec_digest"] = specialization.opspec_digest
        build["lowering_profile_digest"] = specialization.lowering_profile_digest
    build["provenance"] = None
    backend_id = "metal"
    backend_version = None
    device_name = None
    gpu_family = None
    evaluation = {}
    try:
        # Registry facts only (module import + target profile).  A device probe happens only when
        # the caller explicitly asks for it: the real executor does (it is about to evaluate), a
        # CPU-side caller does not.
        from .backends.registry import load_backend
        from .target import load_target

        target = load_target(target_id)
        backend = load_backend(target.backend)
        backend_id = str(getattr(backend, "id", "") or backend_id)
        backend_version = str(getattr(backend, "version", "") or "") or None
        if probe:
            launch = backend.parse_launch(source, task_obj.contract)
            math = "fast" if launch is None else launch.math
            builder = backend.artifact_builder(
                kernel_contract=getattr(task_obj, "contract", None),
                specialization=specialization,
                math=math,
            )
            capabilities = builder.probe()
            # A fallback profile returned for an unavailable toolchain is diagnostic, never an
            # authoritative current identity.  Only a live probe may fill these expected fields.
            if capabilities.available:
                build["provenance"] = capabilities.profile.provenance_document(
                    source_sha256=source_sha256, artifact_set_digest=None,
                )
                compile_profile = capabilities.profile.compile_profile
                device_name = str(compile_profile.get("device") or "") or None
                gpu_family = str(compile_profile.get("gpu_family") or "") or None
    except Exception:
        build["provenance"] = None
        device_name = None
        gpu_family = None
    evaluation.update(
        evaluation_fingerprint=_evaluation_fingerprint(task_obj, backend_id, target_id),
        baseline_implementation_digest=baseline_implementation_digest(task_obj, backend_id),
        baseline_environment_digest=baseline_environment_digest(task_obj, backend_id, target_id),
        protocol_version=PROTOCOL_VERSION,
        target_id=target_id,
        backend=backend_id,
        backend_version=backend_version or None,
        device_name=device_name,
        gpu_family=gpu_family,
    )
    return {
        "build": build,
        "evaluation": evaluation,
        "protocol_version": PROTOCOL_VERSION,
        "target_id": target_id,
    }


def _host_sdk_version(backend_id: str) -> str:
    """@brief 取缓存键里的宿主 SDK 版本（Metal 为 ``platform.mac_ver()``）。

    The value has the same form the Metal ``BuildProfile`` records (``macos-<release>``), so
    the cache key and the cached report's toolchain identity are directly comparable.

    @param backend_id 后端 id。
    @return 版本字符串；非 Metal 或不可读时为空串（空串使 Metal 缓存不可用）。
    """
    if backend_id != "metal":
        return ""
    import platform

    release = str(platform.mac_ver()[0] or "")
    return f"macos-{release}" if release else ""


def _eval_cache_identity(
    *, task: str, dtype: str, size: str, mode: str, seed: int, reps: int,
    cases: tuple[str, ...] | None, fast_bench: bool, require_contract_launch: bool,
    validate: bool, measure_serial: bool, target_id: str, kernel_source: str,
    backend_id: str = "", backend_version: str = "", evaluation_fingerprint: str | None = None,
) -> dict | None:
    """@brief 组装评测缓存的键与期望身份；任一身份无法生成时返回 ``None``（禁用缓存）。

    The key and the expected report identity are the same document: the parent computes it
    before spawning the worker, and a cached value may only be reused when its report fields
    equal it.  A ``null``/``"unknown"`` component is never hashed into a reusable key (§7.1).

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
    @param backend_id 后端身份。
    @param backend_version 后端版本。
    @param evaluation_fingerprint 可覆盖的评测语义指纹。
    @return 键文档；任一身份缺失时为 ``None``。
    """
    from .schema import PROTOCOL_VERSION
    from .task import task_fingerprint
    from tasks import get_task

    task_obj = get_task(task, dtype, size)
    fingerprint_backend = backend_id or "metal"
    host_sdk = _host_sdk_version(fingerprint_backend)
    if fingerprint_backend == "metal" and not host_sdk:
        return None
    implementation = baseline_implementation_digest(task_obj, fingerprint_backend)
    environment = baseline_environment_digest(task_obj, fingerprint_backend, target_id)
    fingerprint = (
        _evaluation_fingerprint(task_obj, fingerprint_backend, target_id)
        if evaluation_fingerprint is None else evaluation_fingerprint
    )
    if not environment or not fingerprint:
        return None
    return {
        "schema": EVAL_CACHE_SCHEMA,
        "protocol": PROTOCOL_VERSION,
        "task": task, "dtype": dtype, "size": size, "mode": mode,
        "task_fingerprint": task_fingerprint(task_obj, fingerprint_backend),
        "evaluation_fingerprint": fingerprint,
        "baseline_implementation_digest": implementation,
        "baseline_environment_digest": environment,
        "host_sdk_version": host_sdk,
        "backend": backend_id,
        "backend_version": backend_version,
        "kernel_sha256": hashlib.sha256(kernel_source.encode("utf-8")).hexdigest(),
        "seed": int(seed), "reps": int(reps), "cases": list(cases or ()),
        "fast_bench": bool(fast_bench), "require_contract_launch": bool(require_contract_launch),
        "validate": bool(validate), "measure_serial": bool(measure_serial), "target_id": target_id,
    }


def _eval_cache_key(
    *, task: str, dtype: str, size: str, mode: str, seed: int, reps: int,
    cases: tuple[str, ...] | None, fast_bench: bool, require_contract_launch: bool,
    validate: bool, measure_serial: bool, target_id: str, kernel_source: str,
    backend_id: str = "", backend_version: str = "", evaluation_fingerprint: str | None = None,
) -> str | None:
    """@brief 计算一次评测的缓存键；身份不完整时返回 ``None``。

    @param 同 :func:`_eval_cache_identity`。
    @return 十六进制 sha256；身份不完整（禁用缓存）时为 ``None``。
    """
    body = _eval_cache_identity(
        task=task, dtype=dtype, size=size, mode=mode, seed=seed, reps=reps, cases=cases,
        fast_bench=fast_bench, require_contract_launch=require_contract_launch,
        validate=validate, measure_serial=measure_serial, target_id=target_id,
        kernel_source=kernel_source, backend_id=backend_id, backend_version=backend_version,
        evaluation_fingerprint=evaluation_fingerprint,
    )
    if body is None:
        return None
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _cache_value_matches(report: "EvalReport", expected: dict, *, backend_id: str) -> bool:
    """@brief 缓存值是否仍描述本次评测（schema 与身份都必须相符）。

    A hit is a *reuse of the same measurement context*: the cached report must be schema 4,
    ``ok``, carry a full build whose candidate/opspec/lowering/sdk identity equals the key,
    and (for the local Metal evaluator) be a native build with a native digest.  Anything else
    is a miss and is recompiled/re-measured.

    @param report 读回的报告。
    @param expected 键文档。
    @param backend_id 后端 id。
    @return 可复用时为 True。
    """
    build = report.build
    if report.status != STATUS_OK or not isinstance(build, dict) or build.get("shape") != "full":
        return False
    if report.evaluation_fingerprint != expected["evaluation_fingerprint"]:
        return False
    if report.baseline_implementation_digest != expected["baseline_implementation_digest"]:
        return False
    if report.baseline_environment_digest != expected["baseline_environment_digest"]:
        return False
    if build.get("candidate_sha256") != expected["kernel_sha256"]:
        return False
    from tasks import get_task

    specialization = get_task(expected["task"], expected["dtype"], expected["size"]).specialization
    if specialization is not None:
        if build.get("opspec_digest") != specialization.opspec_digest:
            return False
        if build.get("lowering_profile_digest") != specialization.lowering_profile_digest:
            return False
    if backend_id == "metal":
        toolchain = (build.get("provenance") or {}).get("toolchain") or {}
        if toolchain.get("sdk_version") != expected["host_sdk_version"]:
            return False
        # A simulator/source_only value is never a local Metal measurement.
        if build.get("build_kind") != "native" or not build.get("native_digest"):
            return False
    return True


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

        doc = read_report_json(path.read_text(encoding="utf-8"), what=f"cached report {path}")
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
    from .target import TargetError, load_target

    profile = None
    target_error = None
    try:
        profile = load_target(target_id)
        backend = load_backend(profile.backend)
    except TargetError as exc:
        backend = None
        target_error = exc

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

    if backend is None or profile is None:
        rep = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=seed, status="rejected", stage="device_check")
        rep.message = str(target_error or f"target {target_id} has no usable backend")
        rep.elapsed_s = time.perf_counter() - t0
        rep.timeline = [{"stage": "device_check", "status": "rejected", "duration_s": rep.elapsed_s}]
        return _checked(rep)

    cache_key = None
    expected = None
    if eval_cache and not fast_bench and seed is not None:
        expected = _eval_cache_identity(
            task=task, dtype=dtype, size=size, mode=mode, seed=seed, reps=reps, cases=cases,
            fast_bench=fast_bench, require_contract_launch=require_contract_launch, validate=validate,
            measure_serial=measure_serial, target_id=target_id, kernel_source=kernel_source,
            backend_id=getattr(backend, "id", ""), backend_version=getattr(backend, "version", ""),
        )
        # An incomplete evaluation identity disables the cache; a null/"unknown" component is
        # never hashed into a reusable key (§7.1).
        cache_key = None if expected is None else hashlib.sha256(
            json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        cached = None if (cache_key is None or eval_cache_refresh) else _cache_read(cache_key)
        if cached is not None and not _cache_value_matches(
            cached, expected, backend_id=getattr(backend, "id", ""),
        ):
            cached = None
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

    # A segmented envelope is a v2 candidate-bundle submission.  The v1 evaluator compiles one
    # MSL file; writing the envelope as kernel.metal would split/lose the file segments, so refuse
    # it intact before any self-check write or worker launch.
    segmented = segmented_source_refusal(kernel_source)
    if segmented:
        rep = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=seed, status="rejected", stage="screen")
        rep.message = segmented
        rep.abi = {"ok": False, "violations": [segmented]}
        rep.elapsed_s = time.perf_counter() - t0
        rep.timeline = [{"stage": "screen", "status": "rejected", "duration_s": rep.elapsed_s}]
        return _checked(rep)

    # Backend-specific static rejection happens before a worker or device launch.
    refused = backend.screen(kernel_source)
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
        kp = Path(workdir or td) / f"kernel{profile.file_ext}"
        kp.parent.mkdir(parents=True, exist_ok=True)
        kp.write_text(kernel_source)
        deadline = deadline_for_source(task, dtype, size, kernel_source, backend=backend)

        def _validation_pass() -> tuple[EvalReport | None, bool]:
            """Run the backend instrumented pass and classify its only-limit exemption."""
            v = _run_worker(task, dtype, size, kp, "validate", seed, reps, timeout or 120.0, validation=True, cases=cases, launch_deadline_s=2 * deadline, progress_path=progress_path, require_contract_launch=require_contract_launch, measure_serial=measure_serial, target_id=target_id, backend=backend)
            size_only = v.status == "abi_error" and backend.is_instrumented_size_only((v.abi or {}).get("violations") or [])
            return (None if (v.status == STATUS_OK or size_only) else v), size_only

        def validation_note(size_only: bool) -> str:
            if not backend.capabilities.supports_validation:
                return "unsupported"
            return backend.validation_summary(size_only=size_only, target=profile)

        size_only = False
        if backend.capabilities.supports_validation and (mode == "full" or (mode == "timed" and validate)):
            failed, size_only = _validation_pass()
            if failed is not None:
                failed.mode = "full" if mode == "full" else "timed"
                failed.elapsed_s = time.perf_counter() - t0
                return _checked(failed)
        if mode == "full":
            rep = _run_worker(task, dtype, size, kp, "full", seed, reps, timeout or 420.0, validation=False, cases=cases, progress_path=progress_path, fast_bench=fast_bench, launch_deadline_s=deadline, require_contract_launch=require_contract_launch, measure_serial=measure_serial, target_id=target_id, backend=backend)
            rep.safety = dict(rep.safety or {})
            rep.safety["validation"] = []
            rep.safety["shader_validation"] = validation_note(size_only)
        elif mode == "timed":
            # 和 full 同一套计时。校验只在后端支持且调用方要求时跑。
            rep = _run_worker(task, dtype, size, kp, "full", seed, reps, timeout or 420.0, validation=False, cases=cases, progress_path=progress_path, fast_bench=fast_bench, launch_deadline_s=deadline, require_contract_launch=require_contract_launch, measure_serial=measure_serial, target_id=target_id, backend=backend)
            rep.mode = "timed"
            if validate:
                rep.safety = dict(rep.safety or {})
                rep.safety["validation"] = []
                rep.safety["shader_validation"] = validation_note(size_only)
        else:
            rep = _run_worker(task, dtype, size, kp, "quick", seed, reps, timeout or 240.0, validation=False, cases=cases, progress_path=progress_path, launch_deadline_s=deadline, require_contract_launch=require_contract_launch, measure_serial=measure_serial, target_id=target_id, backend=backend)
        rep.elapsed_s = time.perf_counter() - t0
        if cache_key is not None:
            _cache_write(cache_key, rep)
        return _checked(rep)


def segmented_source_refusal(kernel_source: str) -> str | None:
    """@brief 自检/单文件编译器是否必须拒绝一份分段候选。

    The envelope carries several files; the v1 evaluator has one ``kernel.metal`` path and would
    otherwise write the whole envelope as source (a confusing compile failure) or, worse, a
    caller could truncate it to one segment.  The v2 path decodes it in ``evaluate_task``.

    @param kernel_source 候选提交文本。
    @return 拒绝原因；不是分段 envelope 时为 None。
    """
    from .track.envelope import is_candidate_envelope

    if is_candidate_envelope(kernel_source):
        return (
            "a segmented candidate envelope is a v2 candidate-bundle submission; the v1 evaluator compiles "
            "one MSL file and must not split or drop its file segments"
        )
    return None
