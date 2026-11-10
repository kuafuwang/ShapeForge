"""Metal implementation adapter.

This file is deliberately lazy: importing it in the parent process must not
import PyObjC, Metal, or allocate a device.  Each method imports the existing
implementation only when called.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from ..base import BenchHarness, Capabilities, DeviceInfo, LaunchRejected


class MetalBackend:
    id = "metal"
    version = "1"
    language = "msl"
    file_ext = ".metal"
    display_name = "Metal"
    source_fence = "metal"
    reference_skill = "msl_reference"
    skill_names = ("apple_gpu_optimization", "metal_profiling", "msl_reference", "profiler_evidence", "hardware_feedback")
    skill_optimization = "apple_gpu_optimization"
    skill_profiling = "metal_profiling"
    skill_profiler = "profiler_evidence"
    skill_hardware = "hardware_feedback"
    runtime_headers = "metal_stdlib, metal_simdgroup and metal_simdgroup_matrix"
    capabilities = Capabilities(
        supports_launch_spec=True,
        supports_validation=True,
        supports_device_timestamps=True,
        supports_profiler=True,
        supports_hw_features=True,
        supports_feature_signature=True,
    )

    def track_adapter(self, track):
        """Construct the adapter for an already-selected TrackSpec, lazily and fail-closed."""
        from .track_factory import adapter_for_track

        return adapter_for_track(track)

    def screen(self, source: str, task: Any = None) -> list[str]:
        """@brief 静态预筛候选源码，在编译和提交 GPU 之前拦下危险构造。

        @param source MSL 源码文本。
        @param task 任务对象；当前实现未使用，保留接口一致。
        @return 拒绝理由列表，空列表表示可以继续编译。
        """
        from .screen import screen_source

        return screen_source(source)

    def parse_launch(self, source: str, contract: Any):
        """@brief 解析源码里的 ``taic-launch`` 块，并注入契约里的整数常量。

        @param source MSL 源码文本。
        @param contract 任务契约，提供可用的整数常量。
        @return 解析出的启动规格；源码没有启动块时为 None。
        """
        from .launch import parse_launch_spec

        constants = {
            name: int(value)
            for name, _type, value in contract.constants
            if isinstance(value, int) and not isinstance(value, bool)
        }
        return parse_launch_spec(source, constants)

    def source_name_pattern(self, kernel_name: str) -> str:
        """Return the legacy MSL declaration regex used by source extraction."""
        return r"\bkernel\s+void\s+" + re.escape(kernel_name) + r"\b"

    def launch_stage_count(self, source: str, contract: Any) -> int:
        """Parse the Metal launch block only to size the generic watchdog."""
        spec = self.parse_launch(source, contract)
        return 1 if spec is None else len(spec.stages)

    def validate_launch(
        self,
        spec: Any,
        *,
        source: str,
        contract: Any,
        task: Any,
        target: Any,
        device_info: Mapping[str, Any],
        require_contract_launch: bool,
    ) -> None:
        """Apply Metal grid/threadgroup/workspace limits after generic parsing."""
        if spec is not None and require_contract_launch:
            raise LaunchRejected("a contract experiment requires the task's own launch; taic-launch is not allowed here")
        if spec is None:
            return
        from .launch import check_launch_limits

        mtp = device_info.get("max_threads_per_threadgroup_xyz") or (device_info["max_threads_per_threadgroup"],) * 3
        check_launch_limits(
            spec,
            contract_names={a.name for a in contract.args},
            max_threads_xyz=tuple(mtp),
            max_threads_product=int(device_info["max_threads_per_threadgroup"]),
            max_threadgroup_memory_bytes=int(device_info["max_threadgroup_memory_bytes"]),
            recommended_working_set_bytes=int(device_info.get("recommended_working_set_bytes") or 0),
            max_buffer_length=int(device_info.get("max_buffer_length") or 2**31),
            rotation_sets=self.rotation_sets(task),
        )

    def validation_env(self, base_env: Mapping[str, str]) -> dict[str, str]:
        """@brief 在给定环境变量上叠加 Metal 着色器校验开关。

        @param base_env 基础环境变量映射。
        @return 打开校验开关后的新字典。
        """
        from .sanitize import env_with_validation

        return env_with_validation(dict(base_env))

    def worker_env(self, base_env: Mapping[str, str], *, validation: bool) -> dict[str, str]:
        """Enable Metal validation, or remove inherited instrumentation for timed runs."""
        if validation:
            return self.validation_env(base_env)
        from .sanitize import VALIDATION_ENV

        env = dict(base_env)
        for key in VALIDATION_ENV:
            env.pop(key, None)
        return env

    def parse_validation_stderr(self, stderr: str) -> list[str]:
        """@brief 从校验子进程的 stderr 中抽取违例行。

        @param stderr 校验子进程的标准错误文本。
        @return 违例行列表（单条截断，最多 12 条）。
        """
        from .sanitize import validation_problems

        return validation_problems(str(stderr or ""))

    def is_instrumented_size_only(self, violations: list[str]) -> bool:
        """@brief 判断 ABI 问题是否只有插桩后放大的 threadgroup 内存超限。

        @param violations ABI 阶段报出的问题列表。
        @return 仅剩静态上限文案时为 True。
        """
        from .abi import only_static_threadgroup_limit

        return only_static_threadgroup_limit(violations)

    def validation_problems(self, violations: list[str], static_bytes: list[int], max_tg_memory: int) -> list[str]:
        """@brief 过滤校验问题，避免插桩放大的内存数字掩盖签名错误。

        @param violations 校验阶段的问题列表。
        @param static_bytes 各 pipeline 报告的静态 threadgroup 内存字节数。
        @param max_tg_memory 设备允许的 threadgroup 内存字节数。
        @return 过滤后的问题列表。
        """
        from .abi import validation_problems

        return validation_problems(violations, static_bytes, max_tg_memory)

    def validation_summary(self, *, size_only: bool, target: Any) -> str:
        """Preserve the existing Metal shader-validation report text."""
        if not size_only:
            return "clean"
        limit = target.limits.max_threadgroup_memory_bytes
        return (
            "skipped: the validation pipeline's threadgroup size is the instrumented figure; "
            f"the uninstrumented pipeline checked the {limit} B limit"
        )

    def compiler_baseline(self, task_name: str, dtype: str, size: str, baseline_us: Mapping[str, float], candidate_us: float) -> dict:
        """Load the optional Triton→Metal comparison owned by this backend."""
        from .triton_baseline import estimate, load_cached

        cached = load_cached(task_name, dtype, size)
        est = estimate(cached, baseline_us) if cached else None
        if est:
            est["speedup"] = est["estimated_us"] / candidate_us
        return est or {}

    def device_matches(self, target: Any, info: Mapping[str, Any]) -> bool:
        """Match the Apple device name and Metal GPU family recorded by the target."""
        name = str(info.get("name") or "")
        family = str(info.get("gpu_family") or "")
        return target.match_name_contains in name and family == target.match_family

    def limit_notes(self, target: Any, info: Mapping[str, Any]) -> list[str]:
        """Preserve the existing Metal profile-vs-device limit warnings."""
        notes = []
        live_threads = info.get("max_threads_per_threadgroup")
        live_memory = info.get("max_threadgroup_memory_bytes")
        if live_threads is not None and int(live_threads) != target.limits.max_threads_per_threadgroup:
            notes.append(
                f"target {target.id} records {target.limits.max_threads_per_threadgroup} threads per threadgroup; "
                f"this device reports {int(live_threads)}. The profile limit stays in force."
            )
        if live_memory is not None and int(live_memory) != target.limits.max_threadgroup_memory_bytes:
            notes.append(
                f"target {target.id} records {target.limits.max_threadgroup_memory_bytes} B of threadgroup memory; "
                f"this device reports {int(live_memory)} B. The profile limit stays in force."
            )
        return notes

    def set_heartbeat_hooks(self, *, on_compile=None, on_launch=None) -> None:
        """@brief 安装编译与启动前后的心跳回调。

        @param on_compile (begin, end) 二元组；省略则不改编译钩子。
        @param on_launch (begin, end) 二元组；省略则不改启动钩子。
        """
        from .runtime import set_compile_hooks, set_launch_hooks

        if on_compile is not None:
            set_compile_hooks(*on_compile)
        if on_launch is not None:
            set_launch_hooks(*on_launch)

    def context(self):
        """@brief 返回当前进程的 Metal 上下文（设备与命令队列）。

        @return 惰性创建的 MetalContext 单例。
        """
        from .runtime import get_context

        return get_context()

    def compile_error(self):
        """@brief 返回编译失败的异常类型，供上层分类。

        @return MetalCompileError 类。
        """
        from .runtime import MetalCompileError

        return MetalCompileError

    def runtime_error(self):
        """@brief 返回 GPU 执行失败的异常类型，供上层分类。

        @return MetalRuntimeError 类。
        """
        from .runtime import MetalRuntimeError

        return MetalRuntimeError

    def prepare_case(self, task: Any, seed: int):
        """@brief 生成正确性检查用的算例数据（输入、参考输出与误差尺度）。

        @param task 任务对象。
        @param seed 随机种子。
        @return CaseData 实例。
        """
        from .testing import prepare_case

        return prepare_case(task, task.cases[0], seed)

    def baseline_runners(self, task: Any, data0: Any) -> Mapping[str, Any]:
        """@brief 构造该任务的全部框架基线计时器。

        @param task 任务对象。
        @param data0 第一个算例的数据，取其已舍入输入。
        @return 基线名到 BaselineRunner 的映射。
        """
        from .baselines import build_runners

        return build_runners(task, data0.rounded)

    def baseline_runner(self, impl: Any, task: Any, data: Any):
        """@brief 为单个基线实现构造一个计时器。

        @param impl 任务声明的基线实现。
        @param task 任务对象。
        @param data 算例数据，取其 ``rounded`` 输入。
        @return 使用一轮轮转缓冲的 BaselineRunner。
        """
        from .baselines import BaselineRunner

        return BaselineRunner(impl, task, data.rounded, rotate=1)

    def rotation_sets(self, task: Any) -> int:
        """@brief 计算基线计时需要的轮转缓冲组数。

        @param task 任务对象。
        @return 轮转组数。
        """
        from .baselines import rotation_count

        return rotation_count(task)

    def coverage(self, task: Any, results: list, source: str, toolchain: dict) -> dict:
        """@brief 汇总本次评测的输入覆盖度记录。

        @param task 任务对象。
        @param results 各算例的结果列表。
        @param source 候选核源码。
        @param toolchain 工具链信息。
        @return 覆盖度字典（domain_sweep 或 sampling）。
        """
        from .domain import coverage_for_report

        return coverage_for_report(task, results, source, toolchain)

    def calibration(self, target_id: str) -> dict:
        """@brief 读取或生成本机的带宽与算力标定。

        @param target_id 目标机型标识。
        @return 标定数据字典。
        """
        from .profile import get_calibration

        return get_calibration(target_id=target_id)

    def roofline(self, task: Any, time_s: float, calibration: dict) -> dict:
        """@brief 按标定结果给出该任务这次测量的 roofline 位置。

        @param task 任务对象。
        @param time_s 实测耗时（秒）。
        @param calibration 标定数据。
        @return roofline 指标字典。
        """
        from .profile import roofline

        return roofline(task, time_s, calibration)

    def assemble(self, contract: Any, source: str) -> tuple[str, int]:
        """@brief 把契约头与候选源码拼成可编译的完整源码。

        @param contract 任务契约。
        @param source 候选核源码。
        @return (完整源码, 头部行数)。
        """
        return contract.assemble(source)

    def rewrite_compiler_lines(self, contract: Any, message: str, header_lines: int) -> str:
        """@brief 把编译器诊断里的行号映射回用户源码。

        @param contract 任务契约。
        @param message 编译器原始诊断文本。
        @param header_lines 头部占用的行数。
        @return 改写后的诊断文本。
        """
        return contract.rewrite_compiler_lines(message, header_lines)

    def describe(self, contract: Any) -> str:
        """@brief 生成契约的人类可读描述。

        @param contract 任务契约。
        @return 描述文本。
        """
        return contract.describe()

    def abi_stub(self, contract: Any) -> str:
        """@brief 生成契约对应的 ABI 声明桩。

        @param contract 任务契约。
        @return ABI 桩文本。
        """
        return contract.abi_stub()

    def probe(self, ctx: Any, target: Any = None) -> DeviceInfo:
        """@brief 读取设备的名称、家族与各项上限。

        @param ctx Metal 上下文。
        @param target 目标档案；在这里校验 Metal limits 与设备身份。
        @return 填充好的 DeviceInfo。
        """
        info = ctx.info()
        matches = True
        match_error = ""
        warnings: tuple[str, ...] = ()
        if target is not None:
            # Force validation of all four Metal limits here, at device_check,
            # rather than making them generic target.toml schema keys.
            _ = target.limits
            matches = self.device_matches(target, info)
            if not matches:
                match_error = (
                    f"device {info.get('name') or ''} ({info.get('gpu_family') or ''}) does not match target {target.id} "
                    f"({target.match_name_contains}, {target.match_family})"
                )
            else:
                warnings = tuple(self.limit_notes(target, info))
        return DeviceInfo(
            backend=self.id,
            name=str(info.get("name") or ""),
            family=str(info.get("gpu_family") or ""),
            limits=dict(info),
            raw=dict(info),
            matches=matches,
            match_error=match_error,
            warnings=warnings,
        )

    def prepare_context(self, context: Any) -> None:
        """@brief 组合根钩子：经 backend 工厂构建并注入 runtime 需要的内部核（清零核）。

        The runtime must not compile on its own (plan §5.2): the caller that owns the context
        builds the fixed internal fill kernel through ``artifact_builder`` with the registered
        ``metal.internal-fill`` diagnostic profile (raw source, no task header) and injects the
        pipeline.  A backend without internal kernels simply does not implement this hook.

        @param context 该 backend 的 runtime 上下文。
        """
        from tcenv.remote.artifact_builder import SourceBundle

        from .native_batch import prepare_native_batch
        from .source_builder import DiagnosticContract

        # The fixed native batch encoder is a runtime optimization, not a candidate build. Prepare
        # it outside all benchmark windows; unsupported contexts keep the PyObjC path.
        prepare_native_batch(context)

        # A context that cannot accept an injection is a test double (it has no internal kernels
        # and no runtime of its own); the real ``MetalContext`` always exposes the hook.
        if not hasattr(context, "inject_fill_kernel"):
            return
        if getattr(context, "_fill", None) is not None:
            return
        contract = DiagnosticContract(kernel_name=context.FILL_KERNEL_NAME)
        builder = self.artifact_builder(
            kernel_contract=contract, diagnostic_profile="metal.internal-fill"
        )
        outcome = builder.build(
            SourceBundle(source=context.FILL_KERNEL_SOURCE.encode("utf-8")),
            task_contract=contract,
            deadline_monotonic=None,
        )
        handles = outcome.executable_handles()
        if len(handles) != 1:
            raise ValueError("the internal fill build must produce exactly one pipeline")
        context.inject_fill_kernel(handles[0])

    def artifact_builder(
        self, *, kernel_contract: Any = None, math: str = "fast", kernel_names: Any = None,
        diagnostic_profile: str | None = None, specialization: Any = None,
    ):
        """Return the one Metal ``ArtifactBuilder`` this backend compiles through.

        There is deliberately no ``compile``/``compile_all`` hook on the backend:
        the evaluator gets this builder from the backend and calls the unified
        ``ArtifactBuilder.build``.  The builder holds the contract/math/declared
        kernel names as configuration; a caller cannot hand it a command or a
        path.  ``diagnostic_profile`` selects a *registered* diagnostic profile
        (raw-source assemble, no task header) for probes/calibration, which have
        no task contract.  A candidate build passes ``specialization``: the
        contract and the lowering identity then come from that one object
        (plan §5.1/§5.3-2).
        """

        from .source_builder import MetalSourceBuilder

        return MetalSourceBuilder(
            backend=self, kernel_contract=kernel_contract, math=math,
            kernel_names=kernel_names, diagnostic_profile=diagnostic_profile,
            specialization=specialization,
        )

    def check_abi(
        self,
        contract: Any,
        kernels: list[Any],
        *,
        target: Any = None,
        device_info: Mapping[str, Any] | None = None,
        spec: Any = None,
        validation: bool = False,
        max_tg_memory: int | None = None,
        max_buffer_index: int | None = None,
        target_id: str = "",
    ) -> list[str]:
        """Check Metal ABI using target-owned limits; retain the old call shape.

        ``target``/``device_info`` are the generic evaluator path.  The three
        legacy keyword arguments remain for existing Metal tooling and tests.
        """
        from . import abi as abi_mod

        if target is not None:
            target_id = target.id
            # The legacy evaluator checks live device's threadgroup-memory
            # report, while buffer-slot policy comes from the target profile.
            max_tg_memory = (
                device_info["max_threadgroup_memory_bytes"]
                if device_info is not None else target.limits.max_threadgroup_memory_bytes
            )
            max_buffer_index = target.limits.max_buffer_index
        if spec is None:
            problems = abi_mod.check_abi(contract, kernels[0], max_tg_memory=max_tg_memory, target_id=target_id or None)
        else:
            by_name = {kernel.name: kernel for kernel in kernels}
            problems = abi_mod.check_pipeline(
                contract, by_name, spec,
                max_tg_memory=max_tg_memory, max_buffer_index=max_buffer_index, target_id=target_id or None,
            )
        if validation:
            if max_tg_memory is None:
                raise ValueError("Metal validation ABI checks require a threadgroup memory limit")
            problems = self.validation_problems(problems, [k.static_tg_bytes for k in kernels], max_tg_memory)
        return problems

    def pipeline_stats(self, kernel: Any, contract: Any, device_info: Mapping[str, Any], *, kernel_source: str | None = None) -> dict:
        """@brief 汇总单个 pipeline 的设备统计与源码级压力提示。

        @param kernel 已编译的核。
        @param contract 任务契约。
        @param device_info 设备信息映射。
        @param kernel_source 候选源码，用于统计 fragment 声明。
        @return 统计字典。
        """
        from .profile import pipeline_stats

        return pipeline_stats(kernel, contract, dict(device_info), kernel_source=kernel_source)

    def pipeline_stats_many(self, pairs: list, device_info: Mapping[str, Any], *, kernel_source: str | None = None) -> dict:
        """@brief 汇总多段流水线的设备统计。

        @param pairs (已编译核, 段声明) 的序列。
        @param device_info 设备信息映射。
        @param kernel_source 候选源码，用于统计 fragment 声明。
        @return 顶层沿用单核形状、``stages`` 为逐段列表的统计字典。
        """
        from .profile import pipeline_stats_many

        return pipeline_stats_many(pairs, dict(device_info), kernel_source=kernel_source)

    def prepare_correctness(self, ctx: Any, contract: Any, *, tight: bool, spec: Any = None):
        """@brief 准备正确性检查用的一组绑定参数。

        @param ctx Metal 上下文。
        @param contract 任务契约。
        @param tight 是否使用无保护带的紧分配（着色器校验用）。
        @param spec 流水线启动规格，提供 workspace；省略则无 workspace。
        @return ArgSet 实例。
        """
        from .sanitize import ArgSet

        return ArgSet(ctx, contract, tight=tight, workspaces=() if spec is None else spec.workspaces)

    def run_correctness(self, ctx: Any, kernel: Any, task: Any, argset: Any, seed: int, *, cases=None, baseline_fns=None, determinism: bool, launcher=None):
        """@brief 跑随机差分测试与安全、确定性检查。

        @param ctx Metal 上下文。
        @param kernel 已编译的候选核。
        @param task 任务对象。
        @param argset 绑定参数集合。
        @param seed 随机种子。
        @param cases 要跑的算例名；省略时用任务自带的算例。
        @param baseline_fns 生成基线输出的回调。
        @param determinism 是否检查多次启动的一致性。
        @param launcher 自定义单次启动函数；省略时用内置 launch_once。
        @return (算例结果, 安全问题, 非确定性问题) 三元组。
        """
        from .testing import run_correctness

        return run_correctness(
            ctx, kernel, task, argset, seed, cases=cases,
            baseline_fns=baseline_fns, determinism=determinism, launcher=launcher,
        )

    def prepare_bench_sets(self, ctx: Any, contract: Any, task: Any, *, spec: Any, data0: Any) -> list[Any]:
        """@brief 构造计时用的多组轮转缓冲。

        @param ctx Metal 上下文。
        @param contract 任务契约。
        @param task 任务对象，提供轮转组数。
        @param spec 流水线启动规格，提供 workspace。
        @param data0 第一个算例的数据。
        @return ArgSet 列表，每组都已装载输入并毒化输出。
        """
        from .baselines import rotation_count
        from .sanitize import ArgSet

        sets = []
        for _ in range(rotation_count(task)):
            one = ArgSet(ctx, contract, workspaces=() if spec is None else spec.workspaces)
            one.load_inputs(data0.raw)
            one.poison_outputs()
            sets.append(one)
        return sets

    def prepare_bench(
        self,
        ctx: Any,
        kernels: list[Any],
        contract: Any,
        task: Any,
        *,
        spec: Any,
        data0: Any,
        reps: int,
        measure_serial: bool,
        sets: list[Any],
        base_runners: Mapping[str, Any],
        selected: list[str],
    ) -> BenchHarness:
        """@brief 组装计时协议：候选、串行候选与各基线的运行函数。

        @param ctx Metal 上下文。
        @param kernels 已编译的核列表。
        @param contract 任务契约。
        @param task 任务对象。
        @param spec 流水线启动规格；省略时按单核计时。
        @param data0 第一个算例的数据，用于计时中抽查输出。
        @param reps 交错计时的重复次数。
        @param measure_serial 是否额外计时严格串行版本。
        @param sets 轮转缓冲组。
        @param base_runners 基线名到计时器的映射。
        @param selected 本次要参与计时的 runner 名字。
        @return 填好各回调的 BenchHarness。
        """
        from .testing import check_case

        rotation_sets = len(sets)
        gpu_samples: list[float] = []
        launched: set[int] = set()
        kernel = kernels[0]
        if spec is None:
            arg_sets = [one.args() for one in sets]

            def cand_run(k: int) -> float:
                tm = ctx.run(kernel, arg_sets, contract.grid, contract.threads, iters=k, dispatch="concurrent")
                gpu_samples.append(tm.gpu_per_iter_s)
                launched.update(range(min(k, len(arg_sets))))
                return tm.wall_s

            def cand_serial_run(k: int) -> float:
                return ctx.run(kernel, arg_sets, contract.grid, contract.threads, iters=k, dispatch="serial").wall_s

            dispatch_note = "concurrent (as MLX)" + ("; candidate_serial_us is the strictly serial time of the same kernel" if measure_serial else "")
            protocol = f"interleaved, {reps} reps, median per-iteration wall time of back-to-back launches over {rotation_sets} rotating buffer sets, concurrent dispatch for the candidate as for MLX"
        else:
            by_name = {item.name: item for item in kernels}
            ordered = [by_name[stage.kernel] for stage in spec.stages]
            grids = [stage.grid for stage in spec.stages]
            threadss = [stage.threads for stage in spec.stages]
            tg_bytes = [stage.threadgroup_bytes for stage in spec.stages]
            stage_sets = [[one.stage_args(k) for k in ordered] for one in sets]
            fills = [one.workspace_fills(declared_zero_only=True) for one in sets]

            def cand_run(k: int) -> float:
                tm = ctx.run_pipeline(ordered, grids, threadss, tg_bytes, stage_sets, fills, iters=k, dispatch="concurrent")
                gpu_samples.append(tm.gpu_per_iter_s)
                launched.update(range(min(k, len(stage_sets))))
                return tm.wall_s

            def cand_serial_run(k: int) -> float:
                return ctx.run_pipeline(ordered, grids, threadss, tg_bytes, stage_sets, fills, iters=k, dispatch="serial").wall_s

            dispatch_note = "pipeline, concurrent encoder with a buffer barrier between stages" + ("; candidate_serial_us is the same stages in a serial encoder" if measure_serial else "")
            protocol = f"interleaved, {reps} reps, median per-iteration wall time of a {len(spec.stages)}-stage pipeline over {rotation_sets} rotating buffer sets; workspace is zeroed inside the timed buffer"

        runners = {}
        for name in selected:
            if name == "candidate":
                runners[name] = cand_run
            elif name == "candidate_serial":
                runners[name] = cand_serial_run
            elif name in base_runners:
                runners[name] = base_runners[name].run_k

        def spot_check() -> list[str]:
            bad: list[str] = []
            checked = sorted(launched)
            for index, one in enumerate(sets):
                if index not in launched:
                    continue
                bad.extend(f"[benchmark set {index}] {item}" for item in one.safety_violations())
                if index in (checked[0], checked[-1]):
                    result = check_case(task, data0, one.read_outputs(), None)
                    if not result.ok:
                        bad.append(f"[benchmark set {index}] output wrong during timed runs: {result.message}")
            return bad

        return BenchHarness(
            rotation_sets=rotation_sets,
            dispatch_note=dispatch_note,
            protocol_note=protocol,
            cand_run=cand_run,
            cand_serial_run=cand_serial_run if measure_serial else None,
            baseline_runners={name: runner for name, runner in runners.items() if name not in ("candidate", "candidate_serial")},
            gpu_samples=gpu_samples,
            spot_check=spot_check,
        )
