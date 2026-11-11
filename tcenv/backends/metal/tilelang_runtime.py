"""TileLang /2 executor adapter over the shared Metal kernel-pipeline runtime.

Only artifact/geometry adaptation lives here.  Correctness, guards, buffer lifetime, reflection
binding, dispatch parity, fresh rotation, interleaved baselines and CommandBuffer timing are the
same ``MetalV2Runtime``/``_evaluate_in_process`` used by the existing Metal Track.  No MSL
candidate contract, ``taic-launch`` parser or HostTilingPlan is imported.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...contract_errors import GeometryError, TaskContractError
from ...dtypes import get_dtype
from ...track import TimingDiscipline
from ...track.abi import tensor_span_bytes
from .kernel_pipeline import MetalKernelPipelineEvaluator
from .launch import LaunchSpec, StageDecl, WorkspaceDecl, launch_math
from .runtime import MetalRuntimeError
from .screen import screen_v2_source
from . import v2_runtime
from .v2_runtime import MetalV2Runtime


def _configure_native_batch(ctx: Any, track_version: int) -> None:
    """Select the Track-owned native encode strategy without replacing the runtime.

    /3 keeps the fast native encoder but bounds concurrent width with a short barrier interval.
    /2 and MSL retain the existing rotation-level barriers.
    CommandBuffer commit/wait, guards, rotation and dispatch evidence stay in the shared runtime.
    """
    ctx.tilelang_barrier_interval = 2 if int(track_version) == 3 else 0


def launch_spec_from_geometry(geometry: Any, abi: Any) -> LaunchSpec:
    """@brief 将已冻结的 ResolvedGeometry/ABI 投影成 runtime 的 LaunchSpec（不发明调度）。"""
    by_name = {tensor.name: tensor for tensor in abi.tensors}
    workspaces: list[WorkspaceDecl] = []
    for use in geometry.workspace_uses:
        tensor = by_name.get(use.name)
        if tensor is None or tensor.space != "workspace":
            raise GeometryError(f"geometry workspace {use.name!r} is not an ABI workspace")
        nbytes = tensor_span_bytes(
            tuple(int(value) for value in tensor.shape),
            tuple(int(value) for value in tensor.strides),
            get_dtype(tensor.dtype).nbytes,
        )
        workspaces.append(WorkspaceDecl(use.name, int(nbytes), bool(use.zero)))
    stages = tuple(
        StageDecl(
            kernel=stage.kernel,
            grid=tuple(int(value) for value in stage.grid),
            threads=tuple(int(value) for value in stage.threads),
            threadgroup_bytes=int(stage.threadgroup_bytes),
            workspaces=tuple(use.name for use in stage.workspace),
        )
        for stage in geometry.active_stages
    )
    if not stages:
        raise GeometryError("the TileLang geometry has no active stage")
    return LaunchSpec(math="safe", workspaces=tuple(workspaces), stages=stages)


class TileLangMetalExecutor:
    """@brief 将已构建的 TileLang artifact 接入共享 MetalV2Runtime 的证据适配器。"""

    def __init__(self, request: Any) -> None:
        self.request = request
        build_request = request.build_request
        track_selection = getattr(build_request, "track_selection", None)
        track = getattr(track_selection, "track", None)
        self.track_version = int(getattr(track, "track_version", 2))
        self.candidate = build_request.candidate
        self.contract = build_request.contract
        self.abi = build_request.abi
        self.geometry = request.geometry
        if request.build is None:
            raise TaskContractError("the TileLang worker needs the real attached Metal build before execution")
        entry = request.build.bundle.entry("kernel.metal")
        if entry.data is None:
            raise TaskContractError("the TileLang build has no generated kernel.metal bytes")
        try:
            self.text = bytes(entry.data).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TaskContractError(f"the TileLang generated MSL is not UTF-8: {exc}") from exc
        self.launch_spec = launch_spec_from_geometry(self.geometry, self.abi)
        lifecycle = request.canonical_bundle.lifecycle
        host_steps = () if lifecycle is None else lifecycle.host_steps
        if host_steps:
            raise GeometryError("TileLang /2 declares no host_tiling lifecycle; its geometry is not a host prepare step")
        self.host_entry = None
        self.host_plan = None
        self.host_program = None
        self.host_inside_timing = False
        self.kernel_names = tuple(dict.fromkeys(stage.kernel for stage in self.geometry.active_stages))
        self.math = launch_math(self.text) or "safe"
        self.evidence: dict[str, Any] | None = None
        self.runtime: MetalV2Runtime | None = None
        native = request.build.bundle.native
        self.tilelang_build = (
            {} if native is None else dict(native.descriptor.get("tilelang") or {})
        )
        if not self.tilelang_build:
            raise TaskContractError("the attached TileLang Metal build has no frontend/build identity")

    @staticmethod
    def runtime_error():
        return MetalRuntimeError

    def screen(self) -> list[str]:
        """@brief 对 frontend 生成的 MSL 再跑同一边界检查（不允许宿主 include/路径逃逸）。"""
        return screen_v2_source(self.text)

    def bind(self, ctx: Any, kernels: list[Any]) -> MetalV2Runtime:
        _configure_native_batch(ctx, self.track_version)
        self.runtime = MetalV2Runtime(
            ctx,
            contract=self.contract,
            abi=self.abi,
            geometry=self.geometry,
            kernels=kernels,
            launch_spec=self.launch_spec,
            host_plan=None,
            host_inside_timing=False,
            limits=self.request.build_request.limits,
            selection=self.request.build_request.selection,
            host_program=None,
            batch_submission=getattr(self, "batch_submission", "single_buffer"),
        )
        self.evidence = {
            "provider": self.runtime.provider,
            "geometry_identity": self.request.canonical_bundle.geometry_identity,
            "tilelang_build": dict(self.tilelang_build),
            "bindings": [binding.document() for binding in self.runtime.bindings],
            "workspace_bytes": self.runtime.argset.actual_buffers()["workspace"],
            "workspace_zero": dict(self.runtime.argset.workspace_zero),
            "workspace_fills": [use.name for use in self.geometry.workspace_uses if use.zero],
            "host_tiling": None,
            "host_program": None,
            "dispatch": [],
            "host_runs": [],
        }
        return self.runtime

    def correctness(self, ctx: Any, runtime: MetalV2Runtime, *, seed: int, cases: tuple[str, ...] | None, determinism: bool):
        return runtime.correctness(cases=cases, seed=seed, determinism=determinism)

    def benchmark(self, ctx: Any, runtime: MetalV2Runtime, *, seed: int, reps: int, baseline_runners: Mapping[str, Any] | None = None):
        """@brief 与现有 Metal kernel-pipeline 相同的 fresh-rotation/interleaved 计时边界。"""
        from .baselines import RING_MAX, RING_TARGET_BYTES

        total = 0
        for tensor in runtime.abi.tensors:
            total += tensor_span_bytes(
                tuple(int(value) for value in tensor.shape),
                tuple(int(value) for value in tensor.strides),
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
        """@brief 记录 runtime 的真实 dispatch/host/launch，而不是 DSL 自报 launch/time。"""
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


class TileLangTrackEvaluator(MetalKernelPipelineEvaluator):
    """@brief TileLang /2 的 bounded evaluator；所有测量与安全逻辑来自共享底座。"""

    executor_type = TileLangMetalExecutor


__all__ = ["TileLangMetalExecutor", "TileLangTrackEvaluator", "launch_spec_from_geometry"]
