"""Shared Metal kernel-pipeline build/evaluation substrate.

This module is deliberately independent of any candidate DSL or MSL Track adapter.  It owns the
v2 task/ABI projection, the bounded-worker evaluator and the schema-4 compile-failure projection.
A Track supplies only an executor adapter with the established ``screen/bind/correctness/
benchmark/record`` interface.  Actual buffers, guards, fresh rotation, dispatch parity,
CommandBuffer commit/wait and timing remain in ``MetalV2Runtime`` and ``_evaluate_in_process``.

MSL keeps its candidate parser in ``track_adapter.py``.  TileLang supplies generated source and
its already-frozen geometry in ``tilelang_runtime.py``.  Neither route calls the other's candidate
contract, HostTilingPlan or Track adapter.
"""

from __future__ import annotations

import math
import os
from typing import Any, Mapping

from ...contract import AbiArg, Contract
from ...contract_errors import GeometryError, TaskContractError
from ...evaluate import _evaluate_in_process
from ...hostprogram.errors import HostCapabilityUnavailable
from ...lowering import Specialization
from ...target import DEFAULT_TARGET_ID, load_target
from ...task import Task
from ...taskcontract.loader import TaskContract
from ...taskcontract.runner import CandidateBuildRequest, CandidateEvaluationRequest
from ...track import ResolvedGeometry
from ...track.expr import INT, evaluate_term


def _resolved_names(instance: Any) -> dict[str, int]:
    """@brief ABI/几何求值可用的整数名字（symbols + 整数 attrs）。"""
    names = {str(name): int(value) for name, value in dict(instance.symbols).items()}
    names.update(
        {str(name): int(value) for name, value in dict(instance.resolved_attrs).items() if isinstance(value, int) and not isinstance(value, bool)}
    )
    return names


def _resolved_shape(instance: Any, tensor: Any) -> tuple[int, ...]:
    """@brief 把 ABI 张量的符号 shape 求值为具体维度。"""
    names = _resolved_names(instance)
    return tuple(
        int(evaluate_term(term, names, where=f"{tensor.name}.shape", expect=INT)) for term in tensor.shape
    )


def metal_contract(
    *,
    task_contract: TaskContract,
    abi: Any,
    geometry: ResolvedGeometry,
    target_id: str = DEFAULT_TARGET_ID,
) -> Contract:
    """@brief 由 v2 ABI + 已选几何合成 Metal 编译契约（不含任何 DSL 候选规则）。"""
    active = geometry.active_stages
    if not active:
        raise GeometryError("the candidate geometry has no active stage")
    instance = task_contract.instance
    args: list[AbiArg] = []
    for position, tensor in enumerate(abi.tensors):
        if tensor.space == "workspace":
            continue
        role = "input" if tensor.role == "input" else "output"
        args.append(
            AbiArg(
                name=tensor.name,
                index=int(position),
                role=role,
                dtype=tensor.dtype,
                shape=_resolved_shape(instance, tensor),
            )
        )
    constants: list[tuple[str, str, Any]] = []
    for name, value in dict(instance.symbols).items():
        constants.append((str(name), "uint", int(value)))
    for name, value in dict(instance.resolved_attrs).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            constants.append((str(name), "uint", int(value)))
        elif isinstance(value, float):
            constants.append((str(name), "float", float(value)))
    target = load_target(target_id)
    first = active[0]
    return Contract(
        kernel_name=first.kernel,
        grid=first.grid,
        threads=first.threads,
        args=tuple(args),
        constants=tuple(constants),
        gpu=target.display_name,
    )


def launch_constants(instance: Any) -> dict[str, int]:
    """@brief 候选几何表达式可用的整数常量。"""
    return _resolved_names(instance)


def _implementation_identity(contract: TaskContract) -> dict[str, Any]:
    """@brief v2 的评测实现身份（冻结、无闭包 repr）。"""
    return {
        "task_pack": {"digest": ""},
        "gen_inputs": {"implementation_id": f"{contract.op_name}.gen_inputs", "implementation_version": 1},
        "reference": {
            "implementation_id": contract.reference.implementation_id,
            "implementation_version": int(contract.reference.reference_version),
        },
        "error_scale": None if contract.reference.error_scale_kind in ("none", "") else {
            "implementation_id": f"{contract.op_name}.error_scale",
            "implementation_version": int(contract.reference.reference_version),
        },
        "baselines": [entry.document() for entry in contract.baselines],
    }


def v2_evaluation_task(request: CandidateBuildRequest, *, lowering_profile_digest: str | None = None) -> Task:
    """@brief 把 v2 合同投影成共享 Metal evaluator 的 ``Task``（不注册全局名字）。"""
    contract = request.contract
    instance = contract.instance
    metal = metal_contract(
        task_contract=contract, abi=request.abi, geometry=request.selection.geometry, target_id=request.target_id
    )
    specialization = Specialization(
        opspec_digest=contract.spec.digest,
        lowering_profile_digest=lowering_profile_digest or contract.fingerprint(),
        spec_params=instance.spec_params,
        backend_params={},
        contract=metal,
    )
    implementations = _implementation_identity(contract)
    implementations["task_pack"] = {"digest": request.locator.deployment_pack_digest}
    error_scale = None
    if contract.reference.error_scale_kind not in ("none", ""):
        error_scale = lambda inputs, refs: contract.reference.error_scale(instance, inputs, refs)
    return Task(
        name=contract.op_name,
        dtype=instance.dtype,
        size=instance.size,
        contract=metal,
        triton_source=contract.statement,
        description=contract.statement,
        cases=tuple(instance.case_domain.cases),
        gen_inputs=lambda case, seed: contract.reference.generate(instance, case, seed),
        reference=lambda inputs: contract.reference.execute(instance, inputs),
        baselines=contract.task_baselines(),
        flops=instance.work_accounting.flops,
        min_bytes=instance.work_accounting.min_bytes,
        rtol=instance.tolerances.rtol,
        atol_rel=instance.tolerances.atol_rel,
        exact=instance.tolerances.exact,
        error_scale=error_scale,
        semantic_instance=instance,
        specialization=specialization,
        meta=instance.evaluation_meta(),
        evaluation_implementations=implementations,
    )


def _build_identity_view(build: Mapping[str, Any]) -> dict[str, Any]:
    """@brief 跨 worker 比较的 build 身份视图：排除耗时/日志，只保留可复现身份。"""
    fields = (
        "shape", "build_kind", "candidate_sha256", "assembled_sha256", "opspec_digest",
        "lowering_profile_digest", "provenance", "placement", "artifacts", "native",
        "native_digest", "native_absent_reason",
    )
    return {key: build.get(key) for key in fields}


def metal_compile_failure_report(
    *,
    contract: TaskContract,
    target_id: str,
    mode: str,
    seed: int,
    error: Any,
    identity: Mapping[str, str],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """@brief 真实编译失败 → schema-4 ``compile_error`` + ``partial`` 构建记录。"""
    from ...build_projection import project_partial
    from ...evaluate import EvalReport
    from ...remote.artifact_builder import require_build_error_projection
    from ...schema import PROTOCOL_VERSION
    from ...target import load_target

    error_class = getattr(error, "error_class", None)
    if error_class != "compile":
        raise TaskContractError(
            f"only a projectable compile failure can be reported here, got {error_class!r}"
        )
    require_build_error_projection(error, require_assembled=True)
    partial = project_partial(error, build_kind="native", identity=identity)
    profile = load_target(target_id)
    report = EvalReport(
        task=contract.op_name,
        dtype=contract.instance.dtype,
        size=contract.instance.size,
        mode=mode,
        status="compile_error",
        stage="compile",
        seed=int(seed),
        compile={"ok": False, "errors": error.report_detail},
        build=partial,
        target_id=str(target_id),
        backend=profile.backend,
        backend_version=profile.backend_version,
        protocol_version=PROTOCOL_VERSION,
    )
    return report.to_json(), partial


def configured_compile_deadline(*, worker_timeout: float) -> float | None:
    """Read the optional per-process build timeout without changing the default path."""
    raw = os.environ.get("TAIC_TRACK_BUILD_TIMEOUT_SEC")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise TaskContractError("TAIC_TRACK_BUILD_TIMEOUT_SEC must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0 or value > float(worker_timeout):
        raise TaskContractError(
            "TAIC_TRACK_BUILD_TIMEOUT_SEC must be finite, positive, and no greater than the worker timeout"
        )
    return value


class MetalKernelPipelineEvaluator:
    """@brief DSL 无关的真实 Metal evaluator：bounded worker + schema-4 + execution evidence。

    ``executor_type`` is the one Track-specific injection.  It adapts candidate/build artifacts to
    the shared runtime interface; it is not a second evaluator and cannot change timing or safety.
    """

    executor_type: Any = None
    task_factory = staticmethod(v2_evaluation_task)

    def __init__(
        self,
        *,
        target_id: str = DEFAULT_TARGET_ID,
        mode: str = "full",
        seed: int = 0,
        reps: int = 5,
        cases: tuple[str, ...] | None = ("normal", "wide_range", "outlier"),
        worker_child: bool = False,
        worker_timeout: float = 420.0,
        worker_launch_deadline_s: float = 8.0,
        worker_compile_deadline_s: float | None = None,
        worker_module: str = "tcenv.v2_worker",
        frozen_identity: Mapping[str, str] | None = None,
    ) -> None:
        self.target_id = target_id
        self.mode = mode
        self.seed = seed
        self.reps = reps
        self.cases = cases
        self.worker_child = worker_child
        self.worker_timeout = float(worker_timeout)
        self.worker_launch_deadline_s = float(worker_launch_deadline_s)
        self.worker_compile_deadline_s = (
            None if worker_compile_deadline_s is None else float(worker_compile_deadline_s)
        )
        self.worker_module = str(worker_module)
        self.frozen_identity = None if frozen_identity is None else dict(frozen_identity)
        self.last_execution: dict[str, Any] | None = None
        self.evaluated_build: Mapping[str, Any] | None = None

    @property
    def owns_build(self) -> bool:
        """@brief 父进程包装器拥有首次构建并交给 bounded worker；worker 内不再递归。"""
        return not self.worker_child

    def _report_identity(self, build_request: CandidateBuildRequest) -> dict[str, str]:
        """@brief 报告使用控制机冻结的构建身份，不采用 staged 路径拼写。"""
        staged = build_request.contract.spec.digest
        if self.frozen_identity is not None:
            if staged != self.frozen_identity["opspec_digest"]:
                raise TaskContractError(
                    "the staged task opspec digest does not match the control-host task the worker was given"
                )
            return dict(self.frozen_identity)
        return {
            "opspec_digest": staged,
            "lowering_profile_digest": build_request.contract.fingerprint(),
        }

    def evaluate(self, request: CandidateEvaluationRequest) -> Mapping[str, Any]:
        """@brief 在外部有界 worker 内构建并执行一次 kernel-pipeline 候选。"""
        self.evaluated_build = None
        if self.worker_child:
            return self._evaluate_direct(request)
        from ...v2_worker import encode_request, run_bounded_v2

        compile_deadline = self.worker_compile_deadline_s
        if compile_deadline is None:
            compile_deadline = configured_compile_deadline(worker_timeout=self.worker_timeout)
        payload = encode_request(
            request,
            mode=self.mode,
            seed=self.seed,
            reps=self.reps,
            cases=self.cases,
            target_id=self.target_id,
            timeout=self.worker_timeout,
            launch_deadline_s=self.worker_launch_deadline_s,
        )
        result = run_bounded_v2(
            payload,
            timeout=self.worker_timeout,
            launch_deadline_s=self.worker_launch_deadline_s,
            compile_deadline_s=compile_deadline,
            worker_module=self.worker_module,
        )
        if result.get("status") in ("timeout", "crash"):
            self.last_execution = {
                "provider": "bounded-v2-worker",
                "status": result.get("status"),
                "message": result.get("message"),
                "dispatch": [],
            }
            return self._synthetic_failure(request, str(result.get("status")), str(result.get("message") or ""))
        report = result.get("report")
        if not isinstance(report, Mapping):
            self.last_execution = {"provider": "bounded-v2-worker", "status": "crash", "dispatch": []}
            return self._synthetic_failure(
                request, "crash", "the bounded v2 worker returned no schema-4 report"
            )
        execution = result.get("execution")
        self.last_execution = dict(execution) if isinstance(execution, Mapping) else {
            "provider": "bounded-v2-worker",
            "dispatch": [],
        }
        worker_track = result.get("track")
        if isinstance(worker_track, Mapping):
            self.last_execution["worker"] = {
                "track": dict(worker_track),
                "track_id": worker_track.get("track_id"),
                "track_version": worker_track.get("track_version"),
                "track_digest": worker_track.get("track_digest"),
                "registry_digest": worker_track.get("registry_digest"),
                "presentation_sha256": result.get("presentation_sha256"),
            }
        build = result.get("build")
        self.evaluated_build = dict(build) if isinstance(build, Mapping) else None
        if report.get("status") in ("crash", "timeout") and report.get("build") is None:
            self.evaluated_build = None
            return self._synthetic_failure(
                request, str(report["status"]), str(report.get("message") or "the bounded v2 worker failed")
            )
        return report

    def _run_in_process(self, *args: Any, **kwargs: Any) -> Any:
        return _evaluate_in_process(*args, **kwargs)

    def _evaluate_direct(self, request: CandidateEvaluationRequest) -> Mapping[str, Any]:
        """@brief worker 内的真实执行路径（不再递归启动 worker）。"""
        if self.executor_type is None:
            raise TaskContractError("a kernel-pipeline evaluator needs its Track executor adapter")
        build_request = request.build_request
        identity = self._report_identity(build_request)
        task = self.task_factory(build_request, lowering_profile_digest=identity["lowering_profile_digest"])
        try:
            executor = self.executor_type(request)
        except HostCapabilityUnavailable as exc:
            return self._capability_report(request, identity, exc)
        report = self._run_in_process(
            task.name,
            task.dtype,
            task.size,
            executor.text,
            mode=self.mode,
            seed=self.seed,
            reps=self.reps,
            cases=self.cases,
            task_obj=task,
            target_id=self.target_id,
            build_identity=identity,
            execution=executor,
            build_outcome=request.build,
        )
        self.last_execution = executor.evidence
        self._check_full_build(report, request, identity)
        return report.to_json()

    def _capability_report(
        self,
        request: CandidateEvaluationRequest,
        identity: Mapping[str, Any],
        exc: Exception,
    ) -> Mapping[str, Any]:
        """@brief 执行期缺工具链：``rejected`` + 已构建的完整 build 记录。"""
        from ...build_projection import project_full
        from ...evaluate import EvalReport
        from ...schema import PROTOCOL_VERSION

        build_request = request.build_request
        report = EvalReport(
            task=build_request.contract.op_name,
            dtype=build_request.contract.instance.dtype,
            size=build_request.contract.instance.size,
            mode=self.mode,
            seed=self.seed,
            status="rejected",
            stage="host_toolchain",
            message=f"backend_capability_unavailable: {exc}"[:2000],
            target_id=self.target_id,
            protocol_version=PROTOCOL_VERSION,
        )
        if request.build is not None:
            report.build = project_full(request.build, build_kind="native", identity=identity)
        self.last_execution = {
            "provider": "bounded-v2-worker",
            "stage": "host_toolchain",
            "dispatch": [],
        }
        return report.to_json()

    def _check_full_build(self, report: Any, request: CandidateEvaluationRequest, identity: Mapping[str, Any]) -> None:
        """@brief 报告 full build 必须与 attached build 完全一致，绝不覆盖。"""
        build = report.build if hasattr(report, "build") else report.get("build") if isinstance(report, Mapping) else None
        if not isinstance(build, Mapping) or build.get("shape") != "full":
            return
        from ...build_projection import project_full

        expected = project_full(request.build, build_kind="native", identity=identity)
        if _build_identity_view(build) != _build_identity_view(expected):
            raise TaskContractError(
                "the Metal evaluator's full build does not match the attached build: "
                "assembled/native/artifact identity drifted instead of being bound to the executed geometry"
            )

    def _synthetic_failure(self, request: CandidateEvaluationRequest, status: str, message: str) -> Mapping[str, Any]:
        """@brief 外部 worker 超时/崩溃时仍返回无伪造 build 的 schema-4 报告。"""
        from ...evaluate import EvalReport

        build_request = request.build_request
        rep = EvalReport(
            task=build_request.contract.op_name,
            dtype=build_request.contract.instance.dtype,
            size=build_request.contract.instance.size,
            mode=self.mode,
            seed=self.seed,
        )
        rep.status = status
        rep.stage = status
        rep.message = message[:1000]
        rep.build = None
        return rep.to_json()


__all__ = [
    "MetalKernelPipelineEvaluator",
    "configured_compile_deadline",
    "launch_constants",
    "metal_compile_failure_report",
    "metal_contract",
    "v2_evaluation_task",
]
