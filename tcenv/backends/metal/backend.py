"""Metal implementation adapter.

This file is deliberately lazy: importing it in the parent process must not
import PyObjC, Metal, or allocate a device.  Each method imports the existing
implementation only when called.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..base import BenchHarness, Capabilities, DeviceInfo


class MetalBackend:
    id = "metal"
    version = "1"
    language = "msl"
    file_ext = ".metal"
    capabilities = Capabilities(
        supports_launch_spec=True,
        supports_validation=True,
        supports_device_timestamps=True,
        supports_profiler=True,
    )

    def screen(self, source: str, task: Any = None) -> list[str]:
        from ...screen import screen_source

        return screen_source(source)

    def parse_launch(self, source: str, contract: Any):
        from ...launch import parse_launch_spec

        constants = {
            name: int(value)
            for name, _type, value in contract.constants
            if isinstance(value, int) and not isinstance(value, bool)
        }
        return parse_launch_spec(source, constants)

    def validation_env(self, base_env: Mapping[str, str]) -> dict[str, str]:
        from ...sanitize import env_with_validation

        return env_with_validation(dict(base_env))

    def parse_validation_stderr(self, stderr: str) -> list[str]:
        from ...sanitize import validation_problems

        return validation_problems(str(stderr or ""))

    def is_instrumented_size_only(self, violations: list[str]) -> bool:
        from ...abi import only_static_threadgroup_limit

        return only_static_threadgroup_limit(violations)

    def validation_problems(self, violations: list[str], static_bytes: list[int], max_tg_memory: int) -> list[str]:
        from ...abi import validation_problems

        return validation_problems(violations, static_bytes, max_tg_memory)

    def set_heartbeat_hooks(self, *, on_compile=None, on_launch=None) -> None:
        from ...metal_runtime import set_compile_hooks, set_launch_hooks

        if on_compile is not None:
            set_compile_hooks(*on_compile)
        if on_launch is not None:
            set_launch_hooks(*on_launch)

    def context(self):
        from ...metal_runtime import get_context

        return get_context()

    def compile_error(self):
        from ...metal_runtime import MetalCompileError

        return MetalCompileError

    def runtime_error(self):
        from ...metal_runtime import MetalRuntimeError

        return MetalRuntimeError

    def prepare_case(self, task: Any, seed: int):
        from ...testing import prepare_case

        return prepare_case(task, task.cases[0], seed)

    def baseline_runners(self, task: Any, data0: Any) -> Mapping[str, Any]:
        from ...baselines import build_runners

        return build_runners(task, data0.rounded)

    def calibration(self, target_id: str) -> dict:
        from ...profile import get_calibration

        return get_calibration(target_id=target_id)

    def roofline(self, task: Any, time_s: float, calibration: dict) -> dict:
        from ...profile import roofline

        return roofline(task, time_s, calibration)

    def assemble(self, contract: Any, source: str) -> tuple[str, int]:
        return contract.assemble(source)

    def rewrite_compiler_lines(self, contract: Any, message: str, header_lines: int) -> str:
        return contract.rewrite_compiler_lines(message, header_lines)

    def describe(self, contract: Any) -> str:
        return contract.describe()

    def abi_stub(self, contract: Any) -> str:
        return contract.abi_stub()

    def probe(self, ctx: Any, target: Any = None) -> DeviceInfo:
        info = ctx.info()
        return DeviceInfo(
            backend=self.id,
            name=str(info.get("name") or ""),
            family=str(info.get("gpu_family") or ""),
            limits=dict(info),
            raw=dict(info),
        )

    def compile(self, ctx: Any, source: str, contract: Any, *, math: str = "fast"):
        return ctx.compile(source, contract.kernel_name, math=math)

    def compile_all(self, ctx: Any, source: str, contract: Any, *, names: list[str] | None = None, math: str = "fast"):
        return ctx.compile_all(source, names or [contract.kernel_name], math=math)

    def check_abi(
        self,
        contract: Any,
        kernels: list[Any],
        *,
        max_tg_memory: int,
        max_buffer_index: int,
        target_id: str,
        spec: Any = None,
    ) -> list[str]:
        from ... import abi as abi_mod

        if spec is None:
            return abi_mod.check_abi(contract, kernels[0], max_tg_memory=max_tg_memory, target_id=target_id)
        by_name = {kernel.name: kernel for kernel in kernels}
        return abi_mod.check_pipeline(
            contract, by_name, spec,
            max_tg_memory=max_tg_memory, max_buffer_index=max_buffer_index, target_id=target_id,
        )

    def pipeline_stats(self, kernel: Any, contract: Any, device_info: Mapping[str, Any], *, kernel_source: str | None = None) -> dict:
        from ...profile import pipeline_stats

        return pipeline_stats(kernel, contract, dict(device_info), kernel_source=kernel_source)

    def pipeline_stats_many(self, pairs: list, device_info: Mapping[str, Any], *, kernel_source: str | None = None) -> dict:
        from ...profile import pipeline_stats_many

        return pipeline_stats_many(pairs, dict(device_info), kernel_source=kernel_source)

    def prepare_correctness(self, ctx: Any, contract: Any, *, tight: bool, spec: Any = None):
        from ...sanitize import ArgSet

        return ArgSet(ctx, contract, tight=tight, workspaces=() if spec is None else spec.workspaces)

    def run_correctness(self, ctx: Any, kernel: Any, task: Any, argset: Any, seed: int, *, cases=None, baseline_fns=None, determinism: bool, launcher=None):
        from ...testing import run_correctness

        return run_correctness(
            ctx, kernel, task, argset, seed, cases=cases,
            baseline_fns=baseline_fns, determinism=determinism, launcher=launcher,
        )

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
        base_runners: Mapping[str, Any],
        selected: list[str],
    ) -> BenchHarness:
        from ...baselines import rotation_count
        from ...sanitize import ArgSet
        from ...testing import check_case

        rotation_sets = rotation_count(task)
        sets = []
        for _ in range(rotation_sets):
            one = ArgSet(ctx, contract, workspaces=() if spec is None else spec.workspaces)
            one.load_inputs(data0.raw)
            one.poison_outputs()
            sets.append(one)
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
            fills = [one.workspace_fills() for one in sets]

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
