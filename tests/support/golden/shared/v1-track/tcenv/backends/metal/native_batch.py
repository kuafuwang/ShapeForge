"""Optional native encoder for single-stage, buffer-only, no-fill Metal batches.

This is a runtime execution optimization, not a candidate build path. The fixed harness-owned
Objective-C source is compiled once during backend context preparation, outside benchmark timing.
Every unsupported pipeline keeps using the existing PyObjC implementation. The outer wall clock
starts before plan preparation and ends after native ``waitUntilCompleted`` so no timed work is
moved out of the measurement boundary.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import objc

_SOURCE = Path(__file__).with_name("native_batch.m")


class _Size(ctypes.Structure):
    _fields_ = [("width", ctypes.c_uint64), ("height", ctypes.c_uint64), ("depth", ctypes.c_uint64)]


class NativeBatchExecutor:
    """A cached native library plus per-argument-set pointer plans."""

    def __init__(self, context: Any, library: Any) -> None:
        self.context = context
        self.library = library
        self.calls = 0
        self._plans: dict[tuple, tuple[Any, Any, Any, Any]] = {}
        self._run = library.taic_metal_run_batch
        self._run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            _Size,
            _Size,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_char_p,
            ctypes.c_uint64,
        ]
        self._run.restype = ctypes.c_int
        self._run_pipeline = library.taic_metal_run_pipeline
        self._run_pipeline.argtypes = [
            ctypes.c_void_p,                 # queue
            ctypes.POINTER(ctypes.c_void_p), # pso_ptrs
            ctypes.c_uint64,                 # nstages
            ctypes.c_void_p,                 # fill_pso_ptr
            ctypes.POINTER(ctypes.c_void_p), # buffers
            ctypes.POINTER(ctypes.c_uint64), # offsets
            ctypes.POINTER(ctypes.c_uint64), # slots
            ctypes.POINTER(ctypes.c_uint64), # arg_counts
            ctypes.c_uint64,                 # nsets
            ctypes.c_uint64,                 # iters
            ctypes.POINTER(_Size),           # grids
            ctypes.POINTER(_Size),           # threads
            ctypes.c_int,                    # concurrent
            ctypes.POINTER(ctypes.c_void_p), # fill_buffers
            ctypes.POINTER(ctypes.c_uint64), # fill_offsets
            ctypes.POINTER(ctypes.c_uint64), # fill_sizes
            ctypes.POINTER(ctypes.c_double), # gpu_s
            ctypes.c_char_p,                 # error
            ctypes.c_uint64,                 # error_capacity
        ]
        self._run_pipeline.restype = ctypes.c_int

    def run(
        self,
        *,
        started: float,
        stages: list[Any],
        grids: list[tuple[int, int, int]],
        threads: list[tuple[int, int, int]],
        threadgroup_bytes: list[int],
        arg_sets: list[list[list[tuple[int, Any]]]],
        iters: int,
        dispatch: str,
        dispatch_recorder: Any,
        dispatch_workspace: list[tuple[str, ...]] | None,
    ) -> tuple[float, float]:
        """Execute an eligible batch and return ``(outer_wall_s, gpu_s)``."""
        import time

        from .runtime import BufferArg, MetalRuntimeError

        stage = stages[0]
        key = (id(stages), id(arg_sets), tuple(grids), tuple(threads))
        plan = self._plans.get(key)
        if plan is None:
            arguments = arg_sets[0][0]
            pointers = (ctypes.c_void_p * (len(arguments) * len(arg_sets)))()
            offsets = (ctypes.c_uint64 * (len(arguments) * len(arg_sets)))()
            slots = (ctypes.c_uint64 * len(arguments))()
            retained = []
            flat_index = 0
            for arg_index, (slot, _arg) in enumerate(arguments):
                slots[arg_index] = slot
            for one_set in arg_sets:
                for _slot, arg in one_set[0]:
                    retained.append(arg.buf)
                    pointers[flat_index] = objc.pyobjc_id(arg.buf)
                    offsets[flat_index] = arg.offset
                    flat_index += 1
            plan = (pointers, offsets, slots, retained, stages, arg_sets)
            self._plans[key] = plan
        pointers, offsets, slots, _retained, _stages, _arg_sets = plan
        if dispatch_recorder is not None:
            for _ in range(iters):
                dispatch_recorder(
                    0,
                    stage,
                    grids[0],
                    threads[0],
                    int(threadgroup_bytes[0]),
                    tuple(dispatch_workspace[0]) if dispatch_workspace is not None else (),
                )
        dynamic = next((binding for binding in stage.bindings if binding.kind == "threadgroup"), None)
        gpu = ctypes.c_double()
        error = ctypes.create_string_buffer(512)
        from . import runtime as runtime_module

        if runtime_module._LAUNCH_HOOKS:
            runtime_module._LAUNCH_HOOKS[0]()
        try:
            result = self._run(
                objc.pyobjc_id(self.context.queue),
                objc.pyobjc_id(stage.pso),
                pointers,
                offsets,
                slots,
                len(slots),
                len(arg_sets),
                iters,
                _Size(*grids[0]),
                _Size(*threads[0]),
                int(dispatch == "concurrent"),
                -1 if dynamic is None else int(dynamic.index),
                int(threadgroup_bytes[0]),
                ctypes.byref(gpu),
                error,
                len(error),
            )
        finally:
            if runtime_module._LAUNCH_HOOKS:
                runtime_module._LAUNCH_HOOKS[1]()
        if result != 0:
            raise MetalRuntimeError(error.value.decode(errors="replace") or f"native batch failed rc={result}")
        self.calls += 1
        return time.perf_counter() - started, float(gpu.value)


    def run_pipeline(
        self,
        *,
        started: float,
        stages: list[Any],
        grids: list[tuple[int, int, int]],
        threads: list[tuple[int, int, int]],
        threadgroup_bytes: list[int],
        arg_sets: list[list[list[tuple[int, Any]]]],
        iters: int,
        dispatch: str,
        dispatch_recorder: Any,
        dispatch_workspace: list[tuple[str, ...]] | None,
    ) -> tuple[float, float]:
        """@brief 多阶段、纯 buffer、无 fill 的整批 pipeline（与 PyObjC run_pipeline 同语义）。

        @param started Python 侧计时起点。
        @param stages 每段的 CompiledKernel。
        @param grids 每段 grid。
        @param threads 每段 threads。
        @param threadgroup_bytes 每段动态 threadgroup 字节（本路径不支持，需为 0）。
        @param arg_sets ``[set][stage][(slot, BufferArg)]``。
        @param iters 轮数。
        @param dispatch ``serial``/``concurrent``。
        @param dispatch_recorder dispatch 审计回调（在实际 dispatch 前逐次调用）。
        @param dispatch_workspace 各段 workspace 名（审计用）。
        @return ``(outer_wall_s, gpu_s)``。
        """
        import time

        from .runtime import MetalRuntimeError

        nstages = len(stages)
        arg_counts = (ctypes.c_uint64 * nstages)(*[len(one) for one in arg_sets[0]])
        total_args = sum(arg_counts)
        pso_ptrs = (ctypes.c_void_p * nstages)(*[objc.pyobjc_id(stage.pso) for stage in stages])
        slots = (ctypes.c_uint64 * total_args)()
        for si, one in enumerate(arg_sets[0]):
            base = sum(arg_counts[:si])
            for a, (slot, _arg) in enumerate(one):
                slots[base + a] = slot
        buffers = (ctypes.c_void_p * (len(arg_sets) * total_args))()
        offsets = (ctypes.c_uint64 * (len(arg_sets) * total_args))()
        retained = []
        for s, one_set in enumerate(arg_sets):
            arg_base = 0
            for si, one in enumerate(one_set):
                for a, (_slot, arg) in enumerate(one):
                    index = s * total_args + arg_base + a
                    retained.append(arg.buf)
                    buffers[index] = objc.pyobjc_id(arg.buf)
                    offsets[index] = arg.offset
                arg_base += len(one)
        grid_arr = (_Size * nstages)(*[_Size(*g) for g in grids])
        thread_arr = (_Size * nstages)(*[_Size(*t) for t in threads])
        if dispatch_recorder is not None:
            for _ in range(iters):
                for si, stage in enumerate(stages):
                    dispatch_recorder(
                        si, stage, grids[si], threads[si], int(threadgroup_bytes[si]),
                        tuple(dispatch_workspace[si]) if dispatch_workspace is not None else (),
                    )
        gpu = ctypes.c_double()
        error = ctypes.create_string_buffer(512)
        from . import runtime as runtime_module

        if runtime_module._LAUNCH_HOOKS:
            runtime_module._LAUNCH_HOOKS[0]()
        try:
            result = self._run_pipeline(
                objc.pyobjc_id(self.context.queue),
                pso_ptrs,
                nstages,
                None,
                buffers,
                offsets,
                slots,
                arg_counts,
                len(arg_sets),
                iters,
                grid_arr,
                thread_arr,
                int(dispatch == "concurrent"),
                None,
                None,
                None,
                ctypes.byref(gpu),
                error,
                len(error),
            )
        finally:
            if runtime_module._LAUNCH_HOOKS:
                runtime_module._LAUNCH_HOOKS[1]()
        if result != 0:
            raise MetalRuntimeError(error.value.decode(errors="replace") or f"native pipeline failed rc={result}")
        self.calls += 1
        return time.perf_counter() - started, float(gpu.value)


def eligible(
    *,
    stages: list[Any],
    arg_sets: list[list[list[tuple[int, Any]]]],
    fills: list[list[tuple]],
) -> bool:
    """@brief 多阶段、纯 buffer、无 fill 的 pipeline 可走原生编码器。"""
    from .runtime import BufferArg

    if not stages or any(fills) or not arg_sets:
        return False
    # 动态 threadgroup 内存的原生路径尚未支持：有这些段的批次退回 PyObjC。
    if any(any(binding.kind == "threadgroup" for binding in getattr(stage, "bindings", ())) for stage in stages):
        return False
    nstages = len(stages)
    return all(
        one_set
        and len(one_set) == nstages
        and all(
            0 < len(stage_args) <= 31
            and all(isinstance(arg, BufferArg) and 0 <= slot <= 30 for slot, arg in stage_args)
            for stage_args in one_set
        )
        for one_set in arg_sets
    )


def prepare_native_batch(context: Any) -> None:
    """@brief 在 backend 组合根预编译固定原生执行器；失败时保留 Python 路径。"""
    if getattr(context, "_native_batch", None) is not None:
        return
    if not hasattr(context, "queue") or not hasattr(context, "device"):
        return
    try:
        source = _SOURCE.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        cache = Path(os.environ.get("TMPDIR", "/tmp")) / f"taic-native-batch-{os.getuid()}"
        cache.mkdir(parents=True, exist_ok=True)
        # Never trust a pre-existing dylib merely because it has the source digest in its name.
        # Rebuild the fixed harness source per process and load only this process's output.
        output = cache / f"{digest}.{os.getpid()}.dylib"
        output.unlink(missing_ok=True)
        try:
            subprocess.run(
                [
                    "clang",
                    "-dynamiclib",
                    "-fobjc-arc",
                    "-framework",
                    "Metal",
                    "-framework",
                    "Foundation",
                    str(_SOURCE),
                    "-o",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            library = ctypes.CDLL(str(output))
        finally:
            output.unlink(missing_ok=True)
        context._native_batch = NativeBatchExecutor(context, library)
    except (OSError, subprocess.SubprocessError, AttributeError):
        # The optimization is optional; unsupported systems retain the existing implementation.
        context._native_batch = None


__all__ = ["NativeBatchExecutor", "eligible", "prepare_native_batch"]
