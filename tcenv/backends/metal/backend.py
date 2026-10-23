"""Metal implementation adapter.

This file is deliberately lazy: importing it in the parent process must not
import PyObjC, Metal, or allocate a device.  Each method imports the existing
implementation only when called.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..base import Capabilities, DeviceInfo


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

    def set_heartbeat_hooks(self, *, on_compile=None, on_launch=None) -> None:
        from ...metal_runtime import set_compile_hooks, set_launch_hooks

        if on_compile is not None:
            set_compile_hooks(*on_compile)
        if on_launch is not None:
            set_launch_hooks(*on_launch)

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
