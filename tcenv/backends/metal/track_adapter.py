"""A4: the real Metal adapters for the v2 track/evaluate_task seam.

A1/A2/A3 froze the hardware-neutral candidate, ABI, geometry, bundle and evidence planes.
This module is the *backend* side of that seam:

* ``MSLSingleGeometryResolver`` parses the candidate's one ``taic-launch`` block into the typed
  ``ExecutionGeometry``; it does not invent a launch when the block is absent.
* ``MetalTrackBuilder`` compiles the candidate with the existing Metal runtime.  A bundle is
  compiled as one unit (Host Tiling kernels first, then device kernels), while its candidate
  entries remain the AUXILIARY/SOURCE artifact set; the build's source identity is fixed before
  compilation.
* ``MetalTrackEvaluator`` executes the built artifact through the existing Metal evaluator
  (correctness, safety, ABI, timing) and binds the schema-4 build record to the geometry-bearing
  artifact set produced by A3's ``attach_execution_geometry``.

No op name, shape, tile or grid appears here.  The only algorithm facts come from the candidate
and its declared geometry; the track is a delivery mechanism.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Mapping

from ...contract_errors import GeometryError, TaskContractError
from ...contract import AbiArg, Contract
from ...evaluate import _evaluate_in_process
from ...lowering import Specialization
from ...remote.artifact_builder import (
    BuildArtifact,
    BuildError,
    BuildEvidence,
    BuildOutcome,
    BuildProvenance,
    SourceBundle,
)
from ...remote.artifact_bundle import ArtifactBundle, BundleEntry, NativeExecutable
from ...target import DEFAULT_TARGET_ID, load_target
from ...task import Task
from ...taskcontract.runner import CandidateBuildRequest, CandidateEvaluationRequest
from .runtime import MetalRuntimeError
from ...taskcontract.loader import TaskContract
from ...track import (
    CandidateArtifact,
    ExecutionGeometry,
    GeometryStage,
    ResolvedGeometry,
)
from ...track.expr import INT, evaluate_term
from ...hostprogram.contract import host_contract_spec, referenced_workspaces
from ...hostprogram.delivery import host_tiling_entry, preflight_host_program
from ...hostprogram.errors import HostCapabilityUnavailable, HostProgramError
from ...hostprogram.native_abi import native_assembled_sha256, render_native_support
from ...hostprogram.native_toolchain import discover_native_toolchain
from .host_program import MetalHostProgram
from .launch import launch_math, parse_launch_spec
from .screen import screen_v2_source
from .source_builder import MetalSourceBuilder
from . import v2_runtime
from .v2_runtime import HostTilingPlan, MetalV2Runtime, reflection_for_kernel


class HostCapabilityBuildError(BuildError):
    """@brief 本机缺 native clang；是环境能力缺失，不是候选源码错误。

    ``BuildError('toolchain_profile')`` 在 v2 worker 里会被当成基础设施异常重抛（进程崩溃）。
    Host Program 的缺依赖必须是一个稳定的 typed outcome：构建边界保留
    ``backend_capability_unavailable``，worker 把它投影成 ``rejected`` 报告而不是 crash。
    """

    error_code = "backend_capability_unavailable"

    def __init__(self, detail: str) -> None:
        super().__init__("toolchain_profile", detail[:400])

TRACK_ADAPTER_SCHEMA_VERSION = 1


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
    """@brief 由 v2 ABI + 已选几何合成 Metal 编译契约。

    Workspace tensors are deliberately excluded from ``Contract.args``: the existing Metal
    runtime allocates them from the ``taic-launch`` workspace declaration, and a workspace is
    not a reference output (``check_case`` compares contract outputs only).  The v2 ABI keeps
    the workspace binding for the neutral evidence plane; the Metal executor binds it by name.

    @param task_contract v2 任务合同。
    @param abi TensorMemoryContract。
    @param geometry 已选 typed 几何。
    @param target_id 目标。
    @return v1 Metal Contract。
    @raises GeometryError 没有激活 stage 或 ABI 形状不可求值。
    """
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
    # The compiler header carries only the instance's named integers/floats.  A candidate may
    # use them; the harness does not choose a tile or grid.
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
    """@brief ``taic-launch`` 表达式可用的整数常量。"""
    return _resolved_names(instance)


def msl_single_geometry_resolver(task_contract: TaskContract):
    """@brief msl-single 的唯一几何解析器：候选的 ``taic-launch`` 块。

    The block is optional in the A1 interface but A4's real single-file track needs it: the
    harness may not choose a grid/threads from a task name or shape.  Absence is refused.

    @param task_contract v2 任务合同。
    @return ``CandidateArtifact -> ExecutionGeometry``。
    """

    def resolve(candidate: CandidateArtifact) -> ExecutionGeometry:
        if not candidate.single_source:
            raise GeometryError("the msl-single resolver accepts exactly one device_kernel entry")
        try:
            text = candidate.entries[0].data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GeometryError(f"the msl-single candidate is not UTF-8 text: {exc}") from exc
        from .launch import LaunchRejected, parse_launch_spec

        try:
            spec = parse_launch_spec(text, launch_constants(task_contract.instance))
        except LaunchRejected as exc:
            raise GeometryError(f"the msl-single taic-launch block is invalid: {exc}") from exc
        if spec is None:
            raise GeometryError(
                "msl-single needs one /* taic-launch */ block; the harness does not invent a grid or threads"
            )
        workspaces = {workspace.name: bool(workspace.zero) for workspace in spec.workspaces}
        stages: list[GeometryStage] = []
        for stage in spec.stages:
            unknown = [name for name in stage.workspaces if name not in workspaces]
            if unknown:
                raise GeometryError(f"the msl-single launch uses undeclared workspace(s) {unknown}")
            stages.append(
                GeometryStage(
                    kernel=stage.kernel,
                    grid=tuple(stage.grid),
                    threads=tuple(stage.threads),
                    threadgroup_bytes=int(stage.threadgroup_bytes),
                    workspace=tuple(
                        {"name": name, "zero": bool(workspaces[name])} for name in stage.workspaces
                    ),
                )
            )
        return ExecutionGeometry(source="candidate", stages=tuple(stages))

    return resolve


def combined_candidate_source(candidate: CandidateArtifact) -> str:
    """@brief 返回 bundle 的 device MSL 编译单元（Host Tiling 是 host plan，不是 MSL）。

    The host_tiling entry is a declarative host plan consumed by the A4 host executor; it is
    not silently renamed into a device kernel and is not concatenated into the MSL library.
    """
    if candidate.single_source:
        return candidate.entries[0].data.decode("utf-8")
    # ``CandidateEntry`` is not orderable: the join order must come from the path, not from the
    # dataclass.  Without the key a two-device bundle raises ``TypeError`` at the first build.
    host = sorted((e for e in candidate.entries if e.role == "host_tiling"), key=lambda entry: entry.path)
    device = sorted(
        (e for e in candidate.entries if e.role in ("device_kernel", "launch")), key=lambda entry: entry.path
    )
    if len(host) != 1:
        raise TaskContractError(f"an A4 msl-bundle needs exactly one host_tiling entry, got {len(host)}")
    if not device:
        raise TaskContractError("an A4 msl-bundle needs at least one device kernel entry")
    parts: list[str] = []
    for entry in device:
        try:
            parts.append(entry.data.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise TaskContractError(f"candidate entry {entry.path!r} is not UTF-8 text: {exc}") from exc
    return "\n".join(parts)


def _candidate_entries(candidate: CandidateArtifact) -> tuple[BundleEntry, ...]:
    """@brief 把候选 entries 变成构建产物的 AUXILIARY 字节。"""
    return tuple(
        BundleEntry(
            path=entry.path,
            role="AUXILIARY",
            media_type=entry.media_type,
            size_bytes=len(entry.data),
            sha256=entry.sha256,
            data=bytes(entry.data),
        )
        for entry in candidate.entries
    )


class MetalTrackBuilder:
    """@brief 真实 Metal ``CandidateBuilder``：编译 + native pipeline，不产生第二身份。

    ``prepare`` runs every pre-compile check (delivery shape, v2 source boundary, typed Metal
    contract, candidate Host Tiling plan) in whichever process calls it.  ``build`` compiles once.
    The control host calls ``prepare`` only when its evaluator owns the first real build; the
    bounded v2 worker calls ``build`` inside its own supervised process.
    """

    build_kind = "native"

    def __init__(self, *, target_id: str = DEFAULT_TARGET_ID, math: str | None = None) -> None:
        self.target_id = target_id
        self.math = math

    def _prepare(self, request: CandidateBuildRequest) -> tuple[Contract, str, str, tuple[str, ...]]:
        """@brief 首次编译前的全部交付/边界检查。

        @param request A3 构建请求。
        @return ``(metal contract, source text, math, kernel names)``。
        @raises TaskContractError/GeometryError 交付/几何/边界不合法。
        """
        candidate = request.candidate
        geometry = request.selection.geometry
        contract = metal_contract(
            task_contract=request.contract, abi=request.abi, geometry=geometry, target_id=self.target_id
        )
        text = combined_candidate_source(candidate)
        # The v2 boundary runs before the first compiler call.  v2 deliberately allows loops and
        # goto; only the host-path/include boundary belongs to the static screen.
        refused = screen_v2_source(text)
        if refused:
            raise TaskContractError("v2 candidate refused before compilation: " + refused[0])
        # Parse/预检 the candidate-owned host work even at build time: a malformed plan or a
        # refused C99 source must not survive until after a GPU dispatch.  The real Clang call
        # happens only in ``build()`` inside the bounded worker.
        entry = host_tiling_entry(candidate)
        if entry is not None and entry.kind == "program":
            spec = host_contract_spec(instance=request.contract.instance, tensors=request.abi.tensors)
            declared = {tensor.name for tensor in request.abi.tensors if tensor.space == "workspace"}
            problems = preflight_host_program(
                entry.source, spec=spec, declared_workspaces=declared, where=entry.path
            )
            if problems:
                raise TaskContractError("host program refused before compilation: " + "; ".join(problems[:4]))
            referenced = set(referenced_workspaces(entry.source, spec))
            possible = {use.name for stage in request.selection.geometry.stages for use in stage.workspace}
            unbound = sorted(referenced - possible)
            if unbound:
                raise TaskContractError(
                    f"{entry.path}: writes workspace(s) {unbound} that no possible stage binds; the possible-stage "
                    "union is the allocation scope"
                )
            zeroed = {use.name for stage in request.selection.geometry.stages for use in stage.workspace if use.zero}
            conflict = sorted(referenced & zeroed)
            if conflict:
                raise TaskContractError(
                    f"{entry.path}: writes {conflict}, but the geometry declares them device-zero; host-written and "
                    "device-zeroed initialization of one block are mutually exclusive"
                )
        else:
            HostTilingPlan.from_candidate(candidate, request.contract)
        math = self.math or launch_math(text) or "safe"
        kernel_names = tuple(dict.fromkeys(stage.kernel for stage in geometry.active_stages))
        return contract, text, math, kernel_names

    def prepare(self, request: CandidateBuildRequest) -> None:
        """@brief 在执行方拥有首次真实编译时，由控制机运行的全部预编译检查。

        No compiler, device or pipeline is touched here, so a stuck Metal frontend stays inside
        the evaluator's externally supervised worker instead of this process.

        @param request A3 构建请求。
        @raises TaskContractError/GeometryError 交付/几何/边界不合法。
        """
        self._prepare(request)

    def build(self, request: CandidateBuildRequest) -> BuildOutcome:
        """@brief 编译候选并返回附几何前的构建产物。

        @param request A3 构建请求。
        @return BuildOutcome（bundle 的 SOURCE 由 A3 attach 时加入）。
        @raises TaskContractError/GeometryError 交付/几何不合法。
        """
        candidate = request.candidate
        contract, text, math, kernel_names = self._prepare(request)
        host_entry = host_tiling_entry(candidate)
        host_toolchain = None
        if host_entry is not None and host_entry.kind == "program":
            # Resolve the Host Program toolchain before compiling MSL: missing native clang is
            # an environment capability refusal with a stable code, not a candidate
            # compile failure and not a worker crash.
            try:
                host_toolchain = discover_native_toolchain()
            except HostCapabilityUnavailable as exc:
                raise HostCapabilityBuildError(str(exc)) from exc
        builder = MetalSourceBuilder(
            kernel_contract=contract, math=math, kernel_names=kernel_names
        )
        if candidate.single_source:
            candidate_entry = candidate.entries[0]
            source_bundle = SourceBundle(
                text.encode("utf-8"),
                media_type=candidate_entry.media_type,
                path=candidate_entry.path,
            )
        else:
            source_bundle = SourceBundle(text.encode("utf-8"))
        compiled = builder.build(source_bundle, task_contract=contract)
        host_program = None
        if host_entry is not None and host_entry.kind == "program":
            lifecycle = request.canonical_bundle.lifecycle
            host_steps = [] if lifecycle is None else [s for s in lifecycle.host_steps if s.role == "host_tiling"]
            if len(host_steps) != 1:
                raise GeometryError(
                    "the candidate ships a C99 Host Program without exactly one host_tiling prepare step"
                )
            declaration = host_steps[0].declaration
            started = time.monotonic()
            try:
                host_program = MetalHostProgram.build_for(
                    candidate,
                    instance=request.contract.instance,
                    tensors=request.abi.tensors,
                    target=load_target(self.target_id),
                    target_id=self.target_id,
                    native_digest=None,
                    inside_timing=bool(declaration.inside_timing),
                    input_dependent=bool(declaration.input_dependent),
                    preprocesses_inputs=bool(declaration.preprocesses_inputs),
                    geometry=request.selection.geometry,
                    toolchain=host_toolchain,
                )
            except HostCapabilityUnavailable as exc:
                raise HostCapabilityBuildError(
                    f"the restricted C99 Host Program toolchain is unavailable on this host: {exc}"
                ) from exc
            except HostProgramError as exc:
                spec = host_contract_spec(instance=request.contract.instance, tensors=request.abi.tensors)
                used = referenced_workspaces(host_entry.source, spec)
                header = render_native_support(spec, used)
                digest = native_assembled_sha256(header, host_entry.source)
                message = str(exc)
                raise BuildError(
                    "compile",
                    f"HostProgramError: {message}"[:500],
                    report_detail=message[:8000],
                    provenance=BuildProvenance(
                        profile=compiled.provenance.profile,
                        source_sha256=candidate.candidate_sha256,
                        artifact_set_digest=None,
                    ),
                    assembled_sha256=digest,
                    log=(
                        f"hostprogram: entry={host_entry.path} assembled_sha256={digest}\n"
                        + message
                    )[:8000].encode("utf-8"),
                    duration_ms=max(1, int(round((time.monotonic() - started) * 1000))),
                ) from exc
        if candidate.single_source:
            if host_program is not None:  # pragma: no cover - single-source cannot carry host work
                raise TaskContractError("a single-source candidate cannot ship a Host Program")
            # The compiled source is the candidate file; SOURCE identity is already correct.
            return compiled
        entries = _candidate_entries(candidate)
        device = next(entry for entry in candidate.entries if entry.role == "device_kernel")
        native = compiled.bundle.native
        if native is None:
            raise TaskContractError("the Metal builder returned no native executable")
        descriptor = dict(native.descriptor)
        if host_program is not None:
            # The native digest enters ``native_digest`` through the descriptor; dylib bytes stay
            # out of Schema-4 entries and out of BuildOutcome.payloads.
            descriptor["host_program"] = host_program.document()
        new_native = NativeExecutable(
            backend_id=native.backend_id,
            kind=native.kind,
            entry_path=device.path,
            entry_sha256=device.sha256,
            descriptor=descriptor,
            handle=native.handle,
        )
        bundle = ArtifactBundle(entries=entries, native=new_native)
        provenance = BuildProvenance(
            profile=compiled.provenance.profile,
            source_sha256=candidate.candidate_sha256,
            artifact_set_digest=bundle.artifact_set_digest,
        )
        evidence = BuildEvidence(
            mode="prebuilt",
            status="ok",
            provenance=provenance,
            artifacts=tuple(BuildArtifact.from_entry(entry) for entry in bundle.entries),
            duration_ms=compiled.evidence.duration_ms,
            log=compiled.evidence.log,
            native=new_native,
            placement="control_host",
        )
        return BuildOutcome(
            bundle=bundle,
            provenance=provenance,
            evidence=evidence,
            assembled_sha256=compiled.assembled_sha256,
        )


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
    """@brief 把 v2 合同投影成现有 Metal evaluator 的 ``Task``（不注册全局名字）。

    ``lowering_profile_digest`` 给控制机冻结的合同指纹：staged 副本的 pack 路径拼写不同，
    直接 ``contract.fingerprint()`` 会让 evaluation_fingerprint 与 build 身份/控制机 expected 不一致。
    缺省保持原行为（调用方就是控制机时使用本机合同指纹）。
    """
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
        # Task.__post_init__ 要求派生字段与 SemanticInstance 一致；sem_int 是首个 exact=True 的
        # v2 任务，漏掉它会让 Task(exact=False) 与实例的 exact=True 分叉并被拒绝。
        exact=instance.tolerances.exact,
        error_scale=error_scale,
        semantic_instance=instance,
        specialization=specialization,
        meta=instance.evaluation_meta(),
        evaluation_implementations=implementations,
    )


class MetalV2Executor:
    """@brief A4 的 v2 runtime 执行器：反射绑定、协议、host tiling 与真实 dispatch 记录。"""

    def __init__(self, request: CandidateEvaluationRequest) -> None:
        self.request = request
        build_request = request.build_request
        self.candidate = build_request.candidate
        self.contract = build_request.contract
        self.abi = build_request.abi
        self.geometry = request.geometry
        self.text = combined_candidate_source(self.candidate)
        self.launch_spec = parse_launch_spec(self.text, launch_constants(self.contract.instance))
        if self.launch_spec is None:
            raise GeometryError("the candidate has no taic-launch block; there is no source-side launch to bind")
        self.host_entry = host_tiling_entry(self.candidate)
        self.host_program = None
        if self.host_entry is not None and self.host_entry.kind == "program":
            self.host_plan = None
        else:
            self.host_plan = HostTilingPlan.from_candidate(self.candidate, self.contract)
        lifecycle = request.canonical_bundle.lifecycle
        host_steps = [] if lifecycle is None else [step for step in lifecycle.host_steps if step.role == "host_tiling"]
        if self.host_entry is not None and len(host_steps) != 1:
            raise GeometryError("the candidate ships a host_tiling entry without exactly one host_tiling prepare step")
        if self.host_entry is None and host_steps:
            raise GeometryError("the lifecycle declares host_tiling work but the candidate has no host_tiling plan")
        self.host_inside_timing = bool(host_steps[0].declaration.inside_timing) if host_steps else False
        if self.host_entry is not None and self.host_entry.kind == "program":
            declaration = host_steps[0].declaration
            native = None if request.build is None else request.build.bundle.native
            descriptor = {} if native is None else dict(native.descriptor.get("host_program") or {})
            self.host_program = MetalHostProgram.from_descriptor(
                self.candidate,
                instance=self.contract.instance,
                tensors=build_request.abi.tensors,
                target=load_target(build_request.target_id),
                target_id=build_request.target_id,
                descriptor=descriptor,
                native_digest=None if native is None else native.digest,
                inside_timing=bool(declaration.inside_timing),
                input_dependent=bool(declaration.input_dependent),
                preprocesses_inputs=bool(declaration.preprocesses_inputs),
            )
        self.kernel_names = tuple(dict.fromkeys(stage.kernel for stage in self.geometry.active_stages))
        self.math = launch_math(self.text) or "safe"
        self.evidence: dict[str, Any] | None = None
        self.runtime: MetalV2Runtime | None = None

    @staticmethod
    def runtime_error():
        return MetalRuntimeError

    def screen(self) -> list[str]:
        """@brief 首次编译前的 v2 边界检查（只查 include/宿主路径，不查循环/goto）。"""
        return screen_v2_source(self.text)

    def bind(self, ctx: Any, kernels: list[Any]) -> MetalV2Runtime:
        self.runtime = MetalV2Runtime(
            ctx,
            contract=self.contract,
            abi=self.abi,
            geometry=self.geometry,
            kernels=kernels,
            launch_spec=self.launch_spec,
            host_plan=self.host_plan,
            host_inside_timing=self.host_inside_timing,
            limits=self.request.build_request.limits,
            selection=self.request.build_request.selection,
            host_program=self.host_program,
        )
        self.evidence = {
            "provider": self.runtime.provider,
            "geometry_identity": self.request.canonical_bundle.geometry_identity,
            "bindings": [binding.document() for binding in self.runtime.bindings],
            "workspace_bytes": self.runtime.argset.actual_buffers()["workspace"],
            "workspace_zero": dict(self.runtime.argset.workspace_zero),
            "workspace_fills": [
                use.name for use in self.geometry.workspace_uses if use.zero
            ],
            "host_tiling": None if self.host_plan is None else self.host_plan.document(),
            "host_program": None if self.host_program is None else self.host_program.document(),
            "dispatch": [],
            "host_runs": [],
        }
        return self.runtime

    def correctness(self, ctx: Any, runtime: MetalV2Runtime, *, seed: int, cases: tuple[str, ...] | None, determinism: bool):
        results, safety, nondet, dispatch, host = runtime.correctness(cases=cases, seed=seed, determinism=determinism)
        return results, safety, nondet, dispatch, host

    def benchmark(self, ctx: Any, runtime: MetalV2Runtime, *, seed: int, reps: int, baseline_runners: Mapping[str, Any] | None = None):
        """@brief 用 v1 Metal 路径同一套轮转规模与计时边界计时候选。

        The batch protocol (``tcenv.bench``), the rotation policy and the interleaved baselines
        belong to the harness, not to the task.  ``rotation_count`` in
        ``tcenv.backends.metal.baselines`` sizes the ring so the working set exceeds the caches
        (192 MiB target, at most 16 sets); the v1 path uses it for every task.  The v2 path
        derives the same number from its ABI spans, so a v1/v2 timing comparison at the same task
        is not distorted by a different memory footprint.

        ``baseline_runners`` is the task's registered baseline timers.  They are measured inside
        the *same* ``interleaved`` call as the candidate, exactly like the v1 path: without them
        the candidate would be the only runner, the GPU would spend the measurement ramping its
        clock, and the ``small`` instance would report a 30-70% regression that is a measurement
        artifact rather than a slower kernel.

        @param ctx Metal 上下文。
        @param runtime v2 runtime。
        @param seed 计时种子。
        @param reps 交错重复次数。
        @param baseline_runners 登记的基线计时器（与 v1 同一次交错）。
        @return runtime 的 ``V2Benchmark``。
        """
        from ...track import TimingDiscipline
        from ...track.abi import tensor_span_bytes
        from ...dtypes import get_dtype
        from .baselines import RING_MAX, RING_TARGET_BYTES

        total = 0
        for tensor in runtime.abi.tensors:
            total += tensor_span_bytes(
                tuple(int(v) for v in tensor.shape),
                tuple(int(v) for v in tensor.strides),
                get_dtype(tensor.dtype).nbytes,
            )
        rotation = max(1, min(RING_MAX, -(-RING_TARGET_BYTES // max(total, 1))))
        native_before = getattr(getattr(ctx, "_native_batch", None), "calls", 0)
        result = runtime.benchmark(
            seed=seed,
            reps=reps,
            discipline=TimingDiscipline.fresh_rotation(rotation),
            baseline_runners=baseline_runners,
        )
        if self.evidence is not None:
            self.evidence["workspace_probe"] = dict(runtime.workspace_probe)
            self.evidence["benchmark_guard_violations"] = list(result.guard_violations)
            self.evidence["benchmark_protocol"] = {
                "rotation_sets": int(result.rotation_sets),
                "attempts": int(result.attempts),
                "noise_cv": float(result.noise_cv),
                "baselines": sorted(result.baselines),
                "native_batch_calls": int(getattr(getattr(ctx, "_native_batch", None), "calls", 0) - native_before),
                **dict(runtime.benchmark_diagnostics),
            }
            if runtime.contract.instance.is_capacity:
                # The timed mixture, in declared order.  It is the L2 measurement boundary, so it travels
                # with the evidence instead of being inferable only from a case name.
                self.evidence["benchmark_batches"] = [dict(item) for item in result.batches]
                notes = dict(runtime.benchmark_mixture_notes)
                mixture = bool(notes.get("mixture"))
                self.evidence["continuous_batching"] = {
                    **notes,
                    "mixture_vs_single_shape_baseline": mixture,
                    "comparable_single_shape_measurement": not mixture,
                }
        return result

    def record(self, *, dispatch, host) -> None:
        """@brief 把一次执行的 dispatch/host/launch 证据并入评测记录。

        ``host`` is a sequence (one document per ordered call) for the v2 runtime; a single mapping is
        still accepted.  ``launch_runs`` is pulled from the runtime: one entry per real call, kept in
        call order and **not** deduplicated by the static dispatch signature (a frozen Dyn-0 grid is
        identical across batches, so signature dedup would keep only the first call of a
        continuous-batching sweep).
        """
        if self.evidence is None:
            self.evidence = {"dispatch": [], "host_runs": []}
        seen = {
            (item["stage"], item["kernel"], tuple(item["grid"]), tuple(item["threads"]), item["threadgroup_bytes"], tuple(item["workspace"]), item["provider"])
            for item in self.evidence["dispatch"]
        }
        for record in dispatch:
            item = record.document()
            key = (item["stage"], item["kernel"], tuple(item["grid"]), tuple(item["threads"]), item["threadgroup_bytes"], tuple(item["workspace"]), item["provider"])
            if key not in seen:
                seen.add(key)
                self.evidence["dispatch"].append(item)
        if host is not None:
            recorded_hosts = {v2_runtime.host_run_identity(item) for item in self.evidence["host_runs"]}
            for item in ([host] if isinstance(host, Mapping) else list(host)):
                key = v2_runtime.host_run_identity(item)
                if key not in recorded_hosts:
                    recorded_hosts.add(key)
                    self.evidence["host_runs"].append(item)
        runtime = getattr(self, "runtime", None)
        launches = getattr(runtime, "launch_runs", None) if runtime is not None else None
        if launches:
            recorded = {
                (item.get("phase"), int(item.get("ordinal", -1))) for item in self.evidence.get("launch_runs", [])
            }
            for item in launches:
                key = (item.get("phase"), int(item.get("ordinal", -1)))
                if key not in recorded:
                    recorded.add(key)
                    self.evidence.setdefault("launch_runs", []).append(dict(item))


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
    """@brief 真实编译失败 → schema-4 ``compile_error`` + ``partial`` 构建记录。

    The v2 worker owns the first real compile, so it also owns the *projection* of a compile
    failure: the report carries the typed ``BuildError`` attachments (real assembled digest, real
    duration/log) and never a ``full`` build or an artifact set.  The control host rejects an
    unprojectable failure as infrastructure instead of blaming the candidate.

    @param contract v2 任务合同。
    @param target_id 目标。
    @param mode 评测模式。
    @param seed 本次种子。
    @param error ``BuildError``（``error_class == "compile"``）。
    @param identity 控制机冻结的 opspec/lowering 身份。
    @return ``(schema-4 report, partial build)``。
    @raises TaskContractError 不是可投影的候选编译失败。
    """
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
    partial = project_partial(error, build_kind=MetalTrackBuilder.build_kind, identity=identity)
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


class MetalTrackEvaluator:
    """@brief 真实 Metal ``CandidateEvaluator``：correctness/safety/timing + schema-4 报告。"""

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
        #: The schema-4 build record the evaluator actually produced and executed.  In the parent
        #: this is the bounded worker's real build, never a locally synthesised one.
        self.evaluated_build: Mapping[str, Any] | None = None

    @property
    def owns_build(self) -> bool:
        """@brief 该 evaluator 是否自行完成首次真实编译（父进程不得先编译）。

        @return 父进程包装器为 True；worker 内的直接执行器为 False。
        """
        return not self.worker_child

    def _report_identity(self, build_request: CandidateBuildRequest) -> dict[str, str]:
        """@brief 报告使用的构建身份。

        控制机上的合同指纹在暂存后会因 pack 路径拼写不同而变。worker 发布控制机冻结的那一份，
        并核对暂存副本的 opspec 摘要没有换题。
        """
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
        """@brief 在外部有界 worker 内用真实 v2 runtime 跑一次评测。

        The evaluator owns the candidate's first real compile: ``evaluate_task`` only ran the
        builder's pre-compile checks here.  This method sends a staged task pack, the exact
        candidate entries and the control host's frozen TrackSpec/registry/rendering identity to
        ``tcenv.v2_worker``.  The worker builds once and executes; the parent only waits and
        SIGKILLs its process group on a stuck compile or launch.  The worker's real build record
        and the track/rendering identity it actually used are published back and cross-audited by
        ``evaluate_task``; nothing is synthesised here.
        """
        self.evaluated_build = None
        if self.worker_child:
            return self._evaluate_direct(request)
        from ...v2_worker import encode_request, run_bounded_v2

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
            compile_deadline_s=self.worker_compile_deadline_s,
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

    def _evaluate_direct(self, request: CandidateEvaluationRequest) -> Mapping[str, Any]:
        """@brief worker 内的真实执行路径（不再递归启动 worker）。"""
        build_request = request.build_request
        identity = self._report_identity(build_request)
        # 与 build 身份同源：控制机冻结的合同指纹，而不是 staged 副本重算的指纹。
        task = v2_evaluation_task(build_request, lowering_profile_digest=identity["lowering_profile_digest"])
        try:
            executor = MetalV2Executor(request)
        except HostCapabilityUnavailable as exc:
            # A different-process executor that must rebuild the module (cache miss) can find no
            # clang/lld/engine.  That is an environment capability refusal with a stable code,
            # not an infrastructure crash and not the candidate's fault.
            return self._capability_report(request, identity, exc)
        report = _evaluate_in_process(
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
        """@brief 执行期缺 Host Program 工具链：``rejected`` + 已构建的完整 build 记录。

        @param request 评测请求。
        @param identity 冻结的构建身份。
        @param exc ``HostCapabilityUnavailable``。
        @return schema-4 报告 JSON。
        """
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
        """@brief 外部 worker 超时/崩溃时仍返回绑定 attached build 的 schema-4 报告。"""
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
        # timeout/crash intentionally carry no build: the build did not produce a candidate
        # report, and synthesising a full one would fabricate evidence.
        rep.build = None
        return rep.to_json()


__all__ = [
    "MetalTrackBuilder",
    "MetalTrackEvaluator",
    "TRACK_ADAPTER_SCHEMA_VERSION",
    "combined_candidate_source",
    "launch_constants",
    "metal_compile_failure_report",
    "metal_contract",
    "msl_single_geometry_resolver",
    "v2_evaluation_task",
]
