"""Two v2 candidates, one task contract, one prepared input.

The parent process does not build a ``MetalContext``. A CPU test passes two
prepared objects that already carry the task shape, bindings and execution
geometry. The production path asks the existing bounded v2 worker to build
both runners: one contract, one ``prepare_v2_case``, and two
``MetalV2Executor.bind`` results. Each runner then launches with that
candidate's own geometry. This module does not invent a grid, a thread count
or a buffer payload. Each batch reuses the evaluator's rotation ring: a fixed/L1
candidate gets the same number of independent buffer sets the evaluator times,
so the rotate barrier and cache footprint sit on the same pass boundary. An L2
capacity instance only has its first prepared case here, so it keeps one set and
labels the document ``single_argset_non_evaluator_rotation`` instead of claiming
the evaluator's mixture.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from taic.paired_compare import (
    PRODUCTION_KMIN,
    PRODUCTION_REPS,
    PRODUCTION_WARMUP_REPS,
    PairedCompareError,
    backend_capability,
    compare_callables,
)

Runner = Callable[[int], float]

#: 计时读的是融合命令缓冲的 GPU 时间戳，和 ``eval-v2 --timed`` 的
#: ``candidate_gpu_us_median`` 同一个物理量。
GPU_TIMESTAMP = "gpu_timestamp"
#: 没有 GPU 样本时的退路。墙钟含编码/提交/等待，不能和上面那个数放在一起比。
WALL_CLOCK = "wall_clock"

#: 测量边界：一整块融合命令缓冲的每迭代 GPU 时间。评测的非 Host 路径就是这一种，
#: ``candidate_gpu_us_median`` 取自 ``run_pipeline(iters=k, dispatch="concurrent")``。
#: ab/paired 复用评测的轮转组数与相互独立的缓冲组，所以 barrier 也在同一个 pass 边界。
FUSED_SCOPE = "fused_command_buffer_per_iteration"
#: 测量边界：每次 launch 一个命令缓冲，各缓冲自身 GPU 时长之和除以 k。评测的
#: Host Program 路径和计时内 Host Tiling 路径每次 run_once 一条样本，就是这一种。
PER_LAUNCH_SCOPE = "per_launch_command_buffers_per_iteration"
#: 只有一组参数、没有复用评测的轮转 mixture。L2 容量题的评测轮转覆盖声明的多个
#: ``(total_tokens, num_seqs)``，ab 只准备了第一份输入，不能把两者说成同一边界。
SINGLE_ARGSET_SCOPE = "single_argset_non_evaluator_rotation"
#: 两侧边界不同（或路径不报告边界）时，文档不能自称它们同量。
MIXED_SCOPE = "mixed"
UNSPECIFIED_SCOPE = "unspecified"


def capability() -> dict:
    """@brief Metal 提供配对比较。导入本模块不创建设备。"""
    return backend_capability("metal")


def _as_tuple(value: Any, what: str) -> tuple[int, ...]:
    try:
        numbers = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise PairedCompareError(f"a paired {what} is not a launch geometry") from exc
    if not numbers:
        raise PairedCompareError(f"a paired {what} is empty")
    return numbers


def _require_prepared(item: Any) -> None:
    """@brief 一颗 runner 必须带着合同形状、绑定和自己的几何。"""
    stages = tuple(getattr(getattr(item, "geometry", None), "active_stages", ()) or ())
    if not stages:
        raise PairedCompareError("a paired runner has no execution geometry")
    for stage in stages:
        _as_tuple(getattr(stage, "grid", None), "grid")
        _as_tuple(getattr(stage, "threads", None), "threads")
    if getattr(item, "bindings", None) is None:
        raise PairedCompareError("a paired runner has no bindings")
    shapes = getattr(item, "shapes", None)
    if not isinstance(shapes, dict) or not shapes:
        raise PairedCompareError("a paired runner has no tensor shapes")
    inputs = getattr(item, "inputs", None)
    if not isinstance(inputs, dict) or not inputs:
        raise PairedCompareError("a paired runner has no prepared inputs")
    for name in ("task", "dtype", "size", "contract_digest", "input_digest"):
        if not str(getattr(item, name, "") or "").strip():
            raise PairedCompareError(f"a paired runner has no {name}")


def _same_contract(left: Any, right: Any) -> None:
    keys = ("task", "dtype", "size", "contract_digest", "input_digest")
    if tuple(getattr(left, key) for key in keys) != tuple(getattr(right, key) for key in keys):
        raise PairedCompareError("paired runners must share one task contract and one prepared input")
    if dict(left.shapes) != dict(right.shapes) or dict(left.inputs) != dict(right.inputs):
        raise PairedCompareError("paired runners must share the prepared tensor bytes of one contract")


def runners_from_prepared(left: Any, right: Any) -> tuple[Runner, Runner]:
    """@brief 用两颗已经备好的 v2 runner 生成交错计时的 ``run_k``。

    启动网格和线程来自各自的几何，输入字节来自同一份准备结果。这里不分配
    新的缓冲，也不改网格。

    @param left 第一颗已绑定的候选。
    @param right 第二颗已绑定的候选。
    @return ``(run_k, run_k)``。
    @raises PairedCompareError 几何、绑定、形状或合同不一致时。
    """
    _require_prepared(left)
    _require_prepared(right)
    _same_contract(left, right)
    return runner_for_prepared(left), runner_for_prepared(right)


def runner_for_prepared(prepared: Any) -> Runner:
    """@brief 一颗已备好的候选 -> ``run_k(k) -> 整批秒数``。

    网格、线程、绑定和输入字节都来自这颗候选自己的准备结果；这里不分配新缓冲。

    @param prepared 已经 ``bind`` 过的 ``_BoundV2``。
    @return 可直接交给 ``tcenv.bench.interleaved`` 的 runner。
    """
    _require_prepared(prepared)
    grids = tuple(_as_tuple(stage.grid, "grid") for stage in prepared.geometry.active_stages)
    threads = tuple(_as_tuple(stage.threads, "threads") for stage in prepared.geometry.active_stages)
    bindings = tuple(prepared.bindings)
    shapes = {name: _as_tuple(shape, "shape") for name, shape in prepared.shapes.items()}
    inputs = {name: bytes(data) for name, data in prepared.inputs.items()}

    def run_k(k: int) -> float:
        total = float(prepared.launch(
            int(k), grids=grids, threads=threads, bindings=bindings, shapes=shapes, inputs=inputs,
        ))
        # 按调用顺序记下这一批的时钟与测量边界。``interleaved`` 只会把 sample 阶段
        # 的批次算进中位数，所以文档的 timed_by 只能由这些批次决定（见
        # ``tcenv.bench.counted_batch_timing``）。
        from tcenv.bench import SAMPLE_PHASE, current_bench_call

        call = current_bench_call()
        phase = SAMPLE_PHASE if call is not None and call.phase == SAMPLE_PHASE else (
            call.phase if call is not None else "direct"
        )
        run_k.clock_history.append((  # type: ignore[attr-defined]
            phase,
            str(getattr(prepared, "clock", WALL_CLOCK)),
            getattr(prepared, "measurement_scope", None),
        ))
        return total

    run_k.clock_history = []  # type: ignore[attr-defined]
    # 让下游能读出这颗 runner 上一次样本用的是哪种时钟。
    run_k.clock_of = lambda: str(getattr(prepared, "clock", WALL_CLOCK))
    # 这一批实际用的轮转组数；ab 据此让每批至少转过全部缓冲组，和评测的 kmin 一致。
    run_k.rotation_sets = int(getattr(prepared, "rotation_sets", 1) or 1)  # type: ignore[attr-defined]
    # 计时之后的 guard / 只读输入。ab 读这个钩子，发现违规就停，不再启动。
    run_k.safety_violations = lambda: list(prepared.post_timing_safety())  # type: ignore[attr-defined]
    return run_k


def recorded_runner(runner: Runner) -> Runner:
    """@brief 给只有 ``clock_of`` 的旧 runner 留下逐批时钟，不再只看最后一次读数。

    每次真正调用之后读一次 ``clock_of``，按 ``tcenv.bench`` 当前的阶段记录。缺失 -> 恢复
    这类序列因此保留历史：最后一批恢复成 GPU 不能把前面退回墙钟的样本洗白。已经自己记录
    逐批历史的 runner（``runner_for_prepared``）原样返回。

    @param runner ``run_k(k) -> 整批秒数``，可以只提供 ``clock_of``。
    @return 带 ``clock_history`` 的 runner；安全/轮转钩子原样透传。
    """
    if isinstance(getattr(runner, "clock_history", None), list):
        return runner
    from tcenv.bench import SAMPLE_PHASE, current_bench_call

    history: list[tuple[str, str, Any]] = []
    reader = getattr(runner, "clock_of", None)

    def run_k(k: int) -> float:
        seconds = float(runner(int(k)))
        call = current_bench_call()
        phase = SAMPLE_PHASE if call is not None and call.phase == SAMPLE_PHASE else (
            call.phase if call is not None else "direct"
        )
        clock = str(reader()) if callable(reader) else WALL_CLOCK
        history.append((phase, clock, getattr(runner, "measurement_scope", None)))
        return seconds

    run_k.clock_history = history  # type: ignore[attr-defined]
    for name in ("clock_of", "safety_violations", "rotation_sets"):
        value = getattr(runner, name, None)
        if value is not None:
            setattr(run_k, name, value)
    return run_k


def _clock_of(*runners: Any) -> str | None:
    """@brief 一次配对里两侧用的是同一种时钟才叫 GPU 时间戳。

    两侧不一致时只能退成墙钟：把 GPU 秒和墙钟秒放进同一个比值里没有意义。任一侧没有
    计入统计的批次时返回 ``None``（不可用）；只有 ``clock_of``、没有逐批历史时不能自证
    GPU，只能说墙钟。
    """
    return timing_identity({str(index): runner for index, runner in enumerate(runners)})["timed_by"]


def timing_identity(runners: Mapping[str, Any]) -> dict[str, Any]:
    """@brief 一批交错计时的 runner 的时钟与测量边界。

    只看``interleaved`` 中真正计入中位数的批次（``tcenv.bench.counted_batch_timing``）。
    一批退回墙钟就整份标成 ``wall_clock``；没有计入统计的批次时两者都是 ``None``。
    ``measurement_scope`` 是这些数字的命令缓冲边界；各路径不同时是 ``mixed``，
    runner 不报告边界时是 ``unspecified``。

    @param runners 标签到 ``run_k`` 的映射。
    @return ``{"timed_by", "measurement_scope"}``。
    """
    from tcenv.bench import counted_batch_timing

    clocks: set[str] = set()
    scopes: set[str] = set()
    unspecified = False
    for runner in runners.values():
        counted = counted_batch_timing(runner)
        if counted is None:
            # 没有逐批历史的 runner 无法证明每一批都是 GPU 时间戳：最后一批读数不能代表
            # 整份文档，退回墙钟并把测量边界标成未知。
            clocks.add(WALL_CLOCK)
            unspecified = True
            continue
        if not counted:
            return {"timed_by": None, "measurement_scope": None}
        for clock, scope in counted:
            clocks.add(clock)
            if scope is None:
                unspecified = True
            else:
                scopes.add(scope)
    if not clocks:
        return {"timed_by": None, "measurement_scope": None}
    timed_by = GPU_TIMESTAMP if clocks == {GPU_TIMESTAMP} else WALL_CLOCK
    if unspecified:
        scope: str | None = UNSPECIFIED_SCOPE
    elif len(scopes) == 1:
        scope = next(iter(scopes))
    elif len(scopes) > 1:
        scope = MIXED_SCOPE
    else:
        scope = None
    return {"timed_by": timed_by, "measurement_scope": scope}


def _digest_inputs(initial: dict[str, bytes]) -> str:
    blob = b"".join(name.encode("utf-8") + b"\0" + bytes(initial[name]) for name in sorted(initial))
    return hashlib.sha256(blob).hexdigest()


def _benchmark_plan(runtime: Any, rotation_sets: int):
    """@brief 与评测同一纪律的 benchmark fill plan（fresh_rotation）。"""
    from tcenv.track import TimingDiscipline, benchmark_launch_plan

    return benchmark_launch_plan(
        runtime.abi, runtime.geometry,
        discipline=TimingDiscipline.fresh_rotation(max(1, int(rotation_sets))),
    )


def _rotation_plan(runtime: Any) -> tuple[int, bool]:
    """@brief ab 这批 launch 的轮转组数与是否真的复用了评测的 mixture。

    固定/L1 题的评测轮转是同一个 case 的 R 份独立缓冲组，``_BoundV2`` 可以用自己的
    ``V2ArgSet`` 重建。L2 容量题的轮转覆盖声明的多个 ``(total_tokens, num_seqs)``，
    而 ab 只准备了第一份输入：这里退回一组，并把边界标成非评测轮转。

    @param runtime 已绑定的 v2 runtime。
    @return ``(组数, 是否复用评测轮转)``。
    """
    instance = getattr(getattr(runtime, "contract", None), "instance", None)
    if instance is not None and bool(getattr(instance, "is_capacity", False)):
        return 1, False
    value = getattr(runtime, "rotation_sets", None)
    if value is None:
        from .v2_runtime import benchmark_rotation

        value = benchmark_rotation(runtime.abi)
    return max(1, int(value)), True


class _BoundV2:
    """@brief 一颗已经 ``bind`` 过的 v2 runtime，加上同一份准备好的输入。"""

    def __init__(self, runtime: Any, *, task: str, dtype: str, size: str, contract_digest: str, inputs: dict[str, bytes]) -> None:
        self.runtime = runtime
        self.task = task
        self.dtype = dtype
        self.size = size
        self.contract_digest = contract_digest
        self.geometry = runtime.geometry
        self.bindings = tuple(runtime.bindings)
        self.shapes = {tensor.name: tuple(int(value) for value in tensor.shape) for tensor in runtime.abi.tensors}
        self.inputs = {name: bytes(data) for name, data in inputs.items()}
        self.input_digest = _digest_inputs(self.inputs)
        self.clock = WALL_CLOCK
        self.measurement_scope: str | None = None
        self._used_argsets: list[Any] = []
        runtime.argset.load_initial(self.inputs)
        self.rotation_sets, self.rotation_reused = _rotation_plan(runtime)
        self._arg_sets: list[Any] = []
        self._stage_args: list[list[Any]] = []
        self._fills: list[list[tuple]] = []
        self._repeat_plan: Any | None = None
        if self._uses_fused_pipeline():
            plan = _benchmark_plan(runtime, self.rotation_sets)
            # 评测在自己的计时窗外建好全部轮转组（输入、零填充计划）。这里做同一件事：
            # 每组是独立缓冲，run_pipeline 才会在转完一整轮时插 barrier。逐 launch 的
            # Host 路径只用 runtime 自己的 argset，不另分配一批用不上的缓冲。
            self._arg_sets = [self._new_argset(runtime, plan) for _ in range(self.rotation_sets)]
            self._stage_args = [
                [one.stage_args(binding) for binding in self.bindings] for one in self._arg_sets
            ]
            host_written = (
                runtime.host_program.written_workspaces
                if getattr(runtime, "host_program", None) is not None
                else frozenset()
            )
            self._fills = [
                one.workspace_fills_declared_zero(exclude=host_written) for one in self._arg_sets
            ]
            # read_write 状态每个计时批次都恢复初始值，和评测的 ``_apply_bench_plans`` 一致。
            self._repeat_plan = self._read_write_plan(runtime)

    def _new_argset(self, runtime: Any, plan: Any):
        """@brief 一组独立的轮转参数；只有 runtime 提供工厂时才走工厂。"""
        factory = getattr(runtime, "new_argset", None)
        one = factory() if callable(factory) else None
        if one is None:
            from .v2_runtime import V2ArgSet

            one = V2ArgSet(runtime.ctx, runtime.abi, runtime.geometry)
        one.load_initial(self.inputs)
        one.apply_fill_plan(plan)
        return one

    def _read_write_plan(self, runtime: Any) -> Any | None:
        """@brief 只恢复 read_write 状态的批次准备计划；没有该状态时是 ``None``。"""
        from tcenv.track import RESTORE_INITIAL

        plan = _benchmark_plan(runtime, self.rotation_sets)
        access = {tensor.name: tensor.access for tensor in runtime.abi.tensors if tensor.space != "workspace"}
        tensors = tuple(
            fill for fill in plan.tensors
            if fill.action == RESTORE_INITIAL and access.get(fill.name) == "read_write"
        )
        if not tensors:
            return None
        return type(plan)(
            phase=plan.phase, tensors=tensors,
            guard_pattern=plan.guard_pattern, guard_bytes=plan.guard_bytes,
        )

    def _track_argset(self, sets=None) -> None:
        """@brief 记下这一批实际用过的轮转组，计时后的安全检查靠它。"""
        for one in (self._arg_sets if sets is None else sets):
            if one is not None and all(one is not seen for seen in self._used_argsets):
                self._used_argsets.append(one)

    def _prepare_batch(self) -> None:
        """@brief 计时窗前准备每个轮转组：read_write 恢复初始值，outside 的 Host Tiling 逐组执行。

        评测在自己的 ``_apply_bench_plans`` 里对每个 set 做同样的事。只执行 runtime 自己
        那一组会让其余轮转组的 host-written workspace 停在未初始化状态。
        """
        host_plan = getattr(self.runtime, "host_plan", None)
        prepare_host = (
            host_plan is not None
            and not getattr(self.runtime, "host_inside_timing", False)
            and callable(getattr(host_plan, "execute", None))
        )
        for one in self._arg_sets:
            if self._repeat_plan is not None:
                one.apply_fill_plan(self._repeat_plan, reset_guards=False)
            if prepare_host:
                host_plan.execute(one)

    def _uses_fused_pipeline(self) -> bool:
        """@brief 评测的非 Host 路径就是一块融合命令缓冲。

        ``MetalV2Runtime.benchmark`` 的 ``cand_run`` 只在两种情况下逐 launch：
        候选带 Host Program（一次 native 调用 + 一个命令缓冲），或声明 Host Tiling
        在计时内。其余路径都走 ``_batched_run`` 的 ``run_pipeline(iters=k)``。
        """
        runtime = self.runtime
        if getattr(runtime, "host_program", None) is not None:
            return False
        if getattr(runtime, "host_plan", None) is not None and getattr(runtime, "host_inside_timing", False):
            return False
        return True

    def launch(self, k: int, *, grids, threads, bindings, shapes, inputs) -> float:
        """@brief 计时 ``k`` 次 launch，返回整批的秒数。

        非 Host 路径复用评测的融合 ``run_pipeline`` 与轮转组：一块命令缓冲装 ``k`` 次
        dispatch，读 ``GPUEnd-GPUStart``。评测的 ``candidate_gpu_us_median`` 就是这块
        缓冲的每迭代 GPU 时间，所以这里返回整批 GPU 秒，``interleaved`` 除以 ``k``。
        Host Program 与计时内 Host Tiling 的评测本身逐 launch，这里也逐 launch，
        保持同一个物理量。没有 GPU 样本时退回墙钟，并在 ``self.clock`` 里标明，
        不让下游把两件事混为一谈。
        """
        _kernels, own_grids, own_threads, _group = self.runtime._stage_sets()
        if tuple(tuple(int(value) for value in item) for item in own_grids) != tuple(grids):
            raise PairedCompareError("the timed launch does not match this candidate's execution geometry")
        if tuple(tuple(int(value) for value in item) for item in own_threads) != tuple(threads):
            raise PairedCompareError("the timed launch does not match this candidate's execution geometry")
        if dict(inputs) != self.inputs:
            raise PairedCompareError("the timed launch does not use the prepared task inputs")
        del bindings, shapes
        self.runtime.argset.load_initial(self.inputs)
        if self._uses_fused_pipeline():
            return self._launch_fused(int(k))
        # 逐 launch 的 Host 路径自己执行 host 工作（Host Program 或计时内 Host Tiling）。
        return self._launch_per_launch(int(k))

    def _accept_batch_gpu(self, timing: Any, k: int) -> float | None:
        """@brief 整批 GPU 秒；缺失或迭代数对不上时返回 ``None``，由调用方退回墙钟。"""
        from tcenv.bench import positive_gpu_sample

        iters = int(getattr(timing, "iters", 0) or 0)
        gpu_s = float(getattr(timing, "gpu_s", 0.0) or 0.0)
        if iters != int(k) or not positive_gpu_sample(gpu_s):
            return None
        return gpu_s

    def _launch_fused(self, k: int) -> float:
        """@brief 一块融合命令缓冲装 k 次 dispatch，复用评测的轮转组与 barrier 边界。

        和评测的 ``_batched_run`` 一样传入全部轮转组的 stage args / zero fills：评测的
        ``run_pipeline`` 每转完一整轮（``i % nsets == 0``）插 rotate barrier，单组会把
        每一轮都串行化，还让同一组缓冲反复命中缓存。不发明新网格或 ABI，只复用 runtime
        已有的 ``run_pipeline`` / ``run_pipeline_per_launch``。L2 容量题只准备了一组时，
        ``measurement_scope`` 明确写成非评测轮转，不冒充同一边界。
        """
        from .runtime import PER_LAUNCH_SET_EVENT, SINGLE_BUFFER, per_launch_eligible

        runtime = self.runtime
        kernels, grids, threads, tg = runtime._stage_sets()
        self._prepare_batch()
        self._track_argset(self._arg_sets[: min(int(k), self.rotation_sets)])
        per_launch = (
            getattr(runtime, "batch_submission", SINGLE_BUFFER) == PER_LAUNCH_SET_EVENT
            and per_launch_eligible(
                stages=kernels, threadgroup_bytes=tg, arg_sets=self._stage_args, fills=self._fills,
            )
        )
        started = time.perf_counter()
        if per_launch:
            timing = runtime.ctx.run_pipeline_per_launch(
                kernels, grids, threads, tg, self._stage_args, self._fills, iters=int(k),
            )
            scope = PER_LAUNCH_SCOPE
        else:
            timing = runtime.ctx.run_pipeline(
                kernels, grids, threads, tg, self._stage_args, self._fills,
                iters=int(k), dispatch="concurrent",
            )
            scope = FUSED_SCOPE
        if not self.rotation_reused:
            scope = SINGLE_ARGSET_SCOPE
        wall = time.perf_counter() - started
        self.measurement_scope = scope
        gpu_s = self._accept_batch_gpu(timing, int(k))
        if gpu_s is not None:
            self.clock = GPU_TIMESTAMP
            return float(gpu_s)
        self.clock = WALL_CLOCK
        return wall

    def _launch_per_launch(self, k: int) -> float:
        """@brief 逐 launch 的评测边界：k 次 ``run_once``，各自一个命令缓冲。

        Host Program 和计时内 Host Tiling 的评测每次 ``run_once`` 记一条 GPU 样本，
        这里把整批的样本相加；``interleaved`` 再除以 k，得到同一个每迭代物理量。
        """
        runtime = self.runtime
        # 打开 runtime 已有的 GPU 采样口。一次 run_once 就是一次 dispatch，所以每条样本
        # 已经是「这一次 launch 的 GPU 秒」，整批求和就是 interleaved 要的批次秒数。
        runtime._attempt_gpu = []
        runtime._gpu_collect = True
        try:
            started = time.perf_counter()
            for _ in range(int(k)):
                runtime.run_once(benchmark=True, record_dispatch=False, host_outside_done=False)
                # 逐 launch 的评测用 runtime 自己的 argset；终检也读这一组，不读融合路径的缓冲组。
                self._track_argset([getattr(runtime, "argset", None)])
            wall = time.perf_counter() - started
        finally:
            runtime._gpu_collect = False
        from tcenv.bench import positive_gpu_sample

        samples = tuple(runtime._attempt_gpu)
        self.measurement_scope = PER_LAUNCH_SCOPE
        if len(samples) == int(k) and samples and all(positive_gpu_sample(value) for value in samples):
            self.clock = GPU_TIMESTAMP
            return float(sum(samples))
        self.clock = WALL_CLOCK
        return wall

    def post_timing_safety(self) -> list[str]:
        """@brief 计时循环已经用过的轮转组上的 guard / 只读输入检查。

        不重新启动。还没计时过时没有已用轮转组，返回空列表。
        """
        return rotation_set_safety(self._used_argsets)


def rotation_set_safety(sets) -> list[str]:
    """@brief 用 benchmark 终检原语检查这些实际用过的轮转组。

    ``collect_benchmark_safety`` 只读 guard 和 ``load_initial`` 的只读快照。
    这里不传覆盖探针结论，也不再发射一次 kernel。
    """
    from tcenv.backends.metal.v2_runtime import collect_benchmark_safety

    usable = [one for one in sets if one is not None]
    if not usable:
        return []
    return [str(item) for item in collect_benchmark_safety(usable, ())]


# 与 MetalKernelPipelineEvaluator 的默认外部期限相同：整段 420 秒，单次启动 8 秒。
# 编译期限沿用 v2 worker 的默认（或 TAIC_TRACK_BUILD_TIMEOUT_SEC）。
WORKER_TIMEOUT_S = 420.0
WORKER_LAUNCH_DEADLINE_S = 8.0


def _locate_task(task: str, dtype: str, size: str):
    """@brief 找到同一道 v2 题的 locator。合同由 ``evaluate_task`` 自己装。"""
    from tcenv.evaluate import REPO_ROOT
    from tcenv.taskcontract.loader import load_task_contract
    from tcenv.taskcontract.locator import TaskLocator, discover_standard_tasks
    from tcenv.taskcontract.pack import task_pack_closure

    found = None
    for locator in discover_standard_tasks(REPO_ROOT / "tasks", code_root=REPO_ROOT):
        if locator.op_name == task:
            found = locator
            break
    if found is None:
        raise PairedCompareError(f"no v2 task named {task!r}")
    contract = load_task_contract(found.task_dir, dtype=dtype, size=size)
    closure = task_pack_closure(found.task_dir, root=REPO_ROOT)
    located = TaskLocator(
        task_id=found.task_id,
        op_name=contract.op_name,
        dtype=contract.dtype,
        size=contract.size,
        pack_digest=contract.pack_digest,
        deployment_pack_digest=closure.digest,
        task_dir=found.task_dir,
        code_root=REPO_ROOT,
    )
    return located


def _decode_candidate(source: str, track):
    """@brief 用 ``evaluate_task`` 同一套入口把源码解成候选。"""
    from tcenv.track import candidate_from_single_source, decode_envelope, is_candidate_envelope

    if is_candidate_envelope(source):
        return decode_envelope(source)
    if track.delivery == "single_source":
        return candidate_from_single_source(source)
    raise PairedCompareError(
        f"track {track.track_id} delivers a candidate bundle; submit the segmented envelope, not one file"
    )


def _lifecycle_factory_for(candidate, contract, target_id: str):
    """@brief 有 Host Tiling 时用 portfolio 的同一工厂。没有 host 工作就返回 None。"""
    from types import SimpleNamespace

    from tcenv.taskcontract.portfolio_runner import _host_work_roles, _lifecycle_factory

    if not _host_work_roles(candidate):
        return None
    projection = SimpleNamespace(candidate=candidate, track_id=candidate.track_id, label="paired")

    def factory(chosen):
        made = _lifecycle_factory(
            projection=projection, resolved=chosen, contract=contract, target_id=target_id,
        )
        if made is None:
            raise PairedCompareError("host work was not registered in a CandidateLifecycle")
        return made(chosen)

    return factory


def assemble_candidate_request(
    source: str | None = None,
    *,
    candidate=None,
    task: str | None = None,
    dtype: str | None = None,
    size: str | None = None,
    track_id: str = "msl-single",
    target_id: str = "apple-m4-max",
    locator=None,
    track_registry_path=None,
    builder=None,
):
    """@brief 用 ``evaluate_task`` 组装一颗候选，停在编译之前。

    合同、track、冻结几何和 lifecycle 都是 ``evaluate_task`` 写进去的。
    带 Host Tiling 的 bundle 得到恰好含 ``host_tiling`` 的 lifecycle。
    这个函数不创建 ``MetalContext``，也不调用 ``builder.build``。

    @return ``CandidateEvaluationRequest``。``build`` 为 None。
    """
    from tcenv.evaluate import EvalReport
    from tcenv.schema import PROTOCOL_VERSION
    from tcenv.taskcontract.runner import evaluate_task, select_track

    if (source is None) == (candidate is None):
        raise PairedCompareError("pass exactly one of source or candidate")
    if locator is None:
        if not task or not dtype or not size:
            raise PairedCompareError("paired compare needs one task, dtype and size; it does not invent an ABI")
        locator = _locate_task(task, dtype, size)
    contract = locator.load()
    selection = select_track(track_id, registry_path=track_registry_path)
    if candidate is None:
        assert source is not None
        candidate = _decode_candidate(source, selection.track)
    resolver = None
    if selection.track.delivery == "single_source":
        from tcenv.backends.metal.track_adapter import msl_single_geometry_resolver

        resolver = msl_single_geometry_resolver(contract)
    if builder is None:
        from tcenv.backends.metal.track_adapter import MetalTrackBuilder

        builder = MetalTrackBuilder(target_id=target_id)

    class _Capture:
        owns_build = True

        def __init__(self) -> None:
            self.request = None

        def evaluate(self, request):
            self.request = request
            return EvalReport(
                task=contract.op_name,
                dtype=contract.dtype,
                size=contract.size,
                mode="quick",
                seed=0,
                status="rejected",
                stage="assembly",
                message="paired assembly stops before a device build",
                target_id=target_id,
                protocol_version=PROTOCOL_VERSION,
            )

    captured = _Capture()
    evaluate_task(
        locator,
        track_id=track_id,
        candidate=candidate,
        builder=builder,
        evaluator=captured,
        geometry_resolver=resolver,
        lifecycle_factory=_lifecycle_factory_for(candidate, contract, target_id),
        track_registry_path=track_registry_path,
        target_id=target_id,
    )
    if captured.request is None:
        raise PairedCompareError("evaluate_task did not assemble a candidate request")
    return captured.request


def assemble_v2_pair(
    path_a, path_b, *, task: str, dtype: str, size: str, seed: int, track_id: str, context: Any,
    target_id: str = "apple-m4-max",
):
    """@brief 在已有的 context 上绑定两颗由 ``evaluate_task`` 组装好的 runner。

    合同和 ``prepare_v2_case`` 只做一次。每颗候选带着自己的几何、lifecycle 和
    ``MetalV2Executor.bind``。调用方必须已经处在有外部期限和机器锁的 worker 里。

    @return 两颗 ``_BoundV2``。
    """
    left, right = assemble_v2_group(
        (path_a, path_b), task=task, dtype=dtype, size=size, seed=seed,
        track_id=track_id, context=context, target_id=target_id,
    )
    return left, right


def assemble_v2_group(
    paths, *, task: str, dtype: str, size: str, seed: int, track_id: str, context: Any,
    target_id: str = "apple-m4-max",
):
    """@brief 同一个合同下绑定任意个候选，用于 N 路交错计时。

    和两路版本走完全相同的组装路径：一份合同、一次 ``prepare_v2_case``、每颗候选
    自己的几何与 ``MetalV2Executor.bind``。不发明网格，也不分配新缓冲。

    @param paths 候选文件路径序列，至少两个。
    @return 按输入顺序排列的 ``_BoundV2`` 列表。
    """
    ordered = tuple(paths)
    if len(ordered) < 2:
        raise PairedCompareError("paired timing needs at least two candidate files")
    if len(set(str(item) for item in ordered)) != len(ordered):
        raise PairedCompareError("the same candidate file appears twice in one paired timing")
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalV2Executor
    from tcenv.backends.metal.v2_runtime import prepare_v2_case
    from tcenv.remote.execution_geometry import attach_execution_geometry
    from tcenv.taskcontract.runner import CandidateEvaluationRequest

    bound = []
    initial = None
    contract_digest = None
    task_name = task
    task_dtype = dtype
    task_size = size
    for path in ordered:
        source = Path(path).read_text(encoding="utf-8")
        evaluation = assemble_candidate_request(
            source, task=task, dtype=dtype, size=size, track_id=track_id, target_id=target_id,
        )
        contract = evaluation.build_request.contract
        if initial is None:
            case_name = contract.instance.case_domain.cases[0]
            prepared_case = prepare_v2_case(contract, case_name, int(seed))
            initial = {name: bytes(data) for name, data in prepared_case.initial.items()}
            if not initial:
                raise PairedCompareError("prepare_v2_case returned no inputs for this contract")
            contract_digest = contract.fingerprint()
            task_name = contract.op_name
            task_dtype = contract.dtype
            task_size = contract.size
        compiled = MetalTrackBuilder(target_id=target_id).build(evaluation.build_request)
        attached = attach_execution_geometry(compiled, canonical_bundle=evaluation.canonical_bundle)
        built = CandidateEvaluationRequest(
            build=attached,
            build_request=evaluation.build_request,
            canonical_bundle=evaluation.canonical_bundle,
            geometry=evaluation.geometry,
        )
        runtime = MetalV2Executor(built).bind(context, list(compiled.executable_handles()))
        bound.append(_BoundV2(
            runtime, task=task_name, dtype=task_dtype, size=task_size,
            contract_digest=contract_digest, inputs=initial,
        ))
    return bound


def _bounded_document(path_a, path_b, *, task: str, dtype: str, size: str, track_id: str, seed: int, sha_a: str, sha_b: str, reps: int = PRODUCTION_REPS) -> dict:
    """@brief 让现有的有界 v2 worker 构建两颗 runner 并交错计时。

    父进程不创建 ``MetalContext``。worker 被外部期限监督。
    """
    from tcenv.backends.metal.kernel_pipeline import configured_compile_deadline
    from tcenv.v2_worker import run_bounded_v2

    payload = {
        "task": task,
        "dtype": dtype,
        "size": size,
        "track_id": track_id,
        "target_id": "apple-m4-max",
        "seed": int(seed),
        "reps": int(reps),
        "path_a": str(path_a),
        "path_b": str(path_b),
        "sha_a": sha_a,
        "sha_b": sha_b,
    }
    result = run_bounded_v2(
        payload,
        timeout=WORKER_TIMEOUT_S,
        launch_deadline_s=WORKER_LAUNCH_DEADLINE_S,
        compile_deadline_s=configured_compile_deadline(worker_timeout=WORKER_TIMEOUT_S),
        worker_module="tcenv.backends.metal.paired_compare",
    )
    paired = result.get("paired") if isinstance(result, dict) else None
    if isinstance(paired, dict):
        report = result.get("report") if isinstance(result.get("report"), dict) else {}
        if not paired.get("status") and report.get("status"):
            paired = dict(paired)
            paired["status"] = report.get("status")
        return paired
    message = ""
    status = "crash"
    if isinstance(result, dict):
        message = str(result.get("message") or "")
        status = str(result.get("status") or status)
    failed = {
        "applied": False,
        "supported": True,
        "backend": "metal",
        "gpu": False,
        "changes_seed": False,
        "status": status,
        "timed_by": None,
        "measurement_scope": None,
        "reason": (message or "the bounded paired worker returned no measurement")[:500],
    }
    if isinstance(result, dict) and "elapsed_s" in result:
        failed["elapsed_s"] = result.get("elapsed_s")
    return failed


def bind_metal_runners(
    path_a, path_b, *, seed: int = 7, context=None, task: str | None = None,
    dtype: str | None = None, size: str | None = None, track_id: str = "msl-single",
    prepared=None, sha_a: str | None = None, sha_b: str | None = None, reps: int = PRODUCTION_REPS,
):
    """@brief 得到两颗 runner，或在父进程里改请有界 worker 直接给出测量。

    ``prepared`` 是测试注入的两颗带形状、绑定和几何的对象。``context`` 只在
    已经受监督的 worker 里传入。两者都没有时，父进程不创建设备。

    @return ``(run_k, run_k)``，或 worker 已经完成的测量文档。
    """
    if prepared is not None:
        left, right = prepared
        return runners_from_prepared(left, right)
    if context is not None:
        if not task or not dtype or not size:
            raise PairedCompareError("paired compare needs one task, dtype and size; it does not invent an ABI")
        left, right = assemble_v2_pair(
            path_a, path_b, task=task, dtype=dtype, size=size, seed=seed, track_id=track_id, context=context,
        )
        return runners_from_prepared(left, right)
    if not task or not dtype or not size:
        raise PairedCompareError("paired compare needs one task, dtype and size; it does not invent an ABI")
    return _bounded_document(
        path_a, path_b, task=task, dtype=dtype, size=size, track_id=track_id, seed=seed,
        sha_a=str(sha_a or ""), sha_b=str(sha_b or ""), reps=reps,
    )


def _rotation_of(runner: Any) -> int:
    """@brief 这颗 runner 一批要转过的轮转组数（旧 runner 没有这个属性时是 1）。"""
    try:
        return max(1, int(getattr(runner, "rotation_sets", 1) or 1))
    except (TypeError, ValueError):
        return 1


def compare_runners(runner_a: Runner, runner_b: Runner, **kwargs: Any) -> dict:
    """@brief 交错两颗已经在本进程里的 runner，并记下这次比较的墙钟。

    只有 ``clock_of`` 的旧 runner 也会在每次真实调用后记下时钟；一批至少转过评测的
    全部轮转组，否则 rotate barrier 和缓存足迹和评测不是同一边界。
    """
    started = time.perf_counter()
    recorded_a = recorded_runner(runner_a)
    recorded_b = recorded_runner(runner_b)
    rotation = max(_rotation_of(recorded_a), _rotation_of(recorded_b))
    options = dict(kwargs)
    options["kmin"] = max(int(options.get("kmin", 1) or 1), rotation)
    document = compare_callables(recorded_a, recorded_b, **options)
    document["backend"] = "metal"
    document["applied"] = True
    document["supported"] = True
    document["gpu"] = True
    document["elapsed_s"] = time.perf_counter() - started
    document["changes_seed"] = False
    document.update(timing_identity({"a": recorded_a, "b": recorded_b}))
    return document


def _production_call(runner_a: Runner, runner_b: Runner, *, sha_a: str, sha_b: str, seed: int, reps: int) -> dict:
    """@brief 真实设备上的配对测量：GPU 时间戳、丢预热、一批转过全部缓冲区组。

    没有这一步，这颗 3 µs 量级的核会报成约 150 µs 墙钟，A/A 抖动 4%，
    而晋级门要求分辨 1%。参数来自现场反馈里逐条写明的复现与期望值。
    """
    return compare_runners(
        runner_a, runner_b,
        sha_a=sha_a, sha_b=sha_b, seed=seed, reps=reps,
        warmup_reps=PRODUCTION_WARMUP_REPS, kmin=PRODUCTION_KMIN,
    )


def compare_paths(
    path_a, path_b, *, sha_a: str, sha_b: str, seed: int = 7, reps: int = PRODUCTION_REPS,
    binder=None, context=None, task: str | None = None, dtype: str | None = None,
    size: str | None = None, track_id: str = "msl-single", prepared=None,
) -> dict:
    """@brief 从两份源码得到一次配对测量。

    测试可以传入 ``binder`` 或 ``prepared``。没有注入时走有界 worker。
    """
    if binder is not None:
        runner_a, runner_b = binder(path_a, path_b)
        return _production_call(runner_a, runner_b, sha_a=sha_a, sha_b=sha_b, seed=seed, reps=reps)
    return compare_for_backend(
        "metal", path_a=path_a, path_b=path_b, sha_a=sha_a, sha_b=sha_b, seed=seed, reps=reps,
        context=context, task=task, dtype=dtype, size=size, track_id=track_id, prepared=prepared,
    )


def compare_for_backend(backend_id: str, runner_a: Runner | None = None, runner_b: Runner | None = None, **kwargs: Any) -> dict:
    """@brief 按后端分发。Metal 在没有现成 runner 时绑定或交给有界 worker。"""
    info = backend_capability(backend_id)
    if not info["supported"]:
        return {"applied": False, "supported": False, "backend": info["backend"], "reason": info["reason"], "changes_seed": False}
    path_a = kwargs.pop("path_a", None)
    path_b = kwargs.pop("path_b", None)
    binder = kwargs.pop("binder", None)
    context = kwargs.pop("context", None)
    prepared = kwargs.pop("prepared", None)
    task = kwargs.pop("task", None)
    dtype = kwargs.pop("dtype", None)
    size = kwargs.pop("size", None)
    track_id = kwargs.pop("track_id", "msl-single")
    if runner_a is None or runner_b is None:
        if binder is not None:
            runner_a, runner_b = binder(path_a, path_b)
        elif not path_a or not path_b:
            return {
                "applied": False,
                "supported": True,
                "backend": "metal",
                "reason": "Metal can compare two in-process runners; none are bound for these sources",
                "changes_seed": False,
                "gpu": False,
            }
        else:
            produced = bind_metal_runners(
                path_a, path_b, seed=int(kwargs.get("seed", 7)), context=context, task=task,
                dtype=dtype, size=size, track_id=str(track_id or "msl-single"), prepared=prepared,
                sha_a=kwargs.get("sha_a"), sha_b=kwargs.get("sha_b"), reps=int(kwargs.get("reps", PRODUCTION_REPS)),
            )
            if isinstance(produced, dict):
                return produced
            runner_a, runner_b = produced
    if runner_a is not None and runner_b is not None:
        return _production_call(
            runner_a, runner_b,
            sha_a=str(kwargs.get("sha_a") or ""), sha_b=str(kwargs.get("sha_b") or ""),
            seed=int(kwargs.get("seed", 7)), reps=int(kwargs.get("reps", PRODUCTION_REPS)),
        )
    return compare_runners(runner_a, runner_b, **kwargs)


def _worker_main(request: dict) -> dict:
    """@brief 安装 v2 心跳，独占机器 GPU 锁，再绑定两颗 runner 并交错计时。

    心跳和锁与 ``tcenv.v2_worker`` 的评测 worker 相同：``_install_heartbeat``
    提供编译 ``C`` 和启动 ``B``，``gpu_lock(..., exclusive=True)`` 在等锁时写 ``W``。
    父进程已经放开自己的文件锁，子进程必须自己取得这把锁。
    """
    from tcenv.backends.metal.runtime import get_context
    from tcenv.gpulock import gpu_lock
    from tcenv.v2_worker import RESULT_SCHEMA, _install_heartbeat

    payload = request["payload"]
    target_id = str(request.get("target_id") or payload.get("target_id") or "apple-m4-max")
    out = Path(request["out"])
    try:
        heartbeat = request.get("heartbeat")
        if not heartbeat:
            raise PairedCompareError("the paired worker needs the supervisor heartbeat path")
        _install_heartbeat(str(heartbeat), target_id)
        with gpu_lock(
            f"paired compare: {payload.get('task')} {payload.get('dtype')} {payload.get('size')}",
            exclusive=True,
        ):
            context = get_context()
            left, right = assemble_v2_pair(
                payload["path_a"], payload["path_b"], task=payload["task"], dtype=payload["dtype"],
                size=payload["size"], seed=int(payload.get("seed", 7)),
                track_id=str(payload.get("track_id") or "msl-single"), context=context,
                target_id=target_id,
            )
            runner_a, runner_b = runners_from_prepared(left, right)
            document = _production_call(
                runner_a, runner_b, sha_a=payload.get("sha_a") or "", sha_b=payload.get("sha_b") or "",
                seed=int(payload.get("seed", 7)), reps=int(payload.get("reps", PRODUCTION_REPS)),
            )
        envelope = {
            "schema_version": RESULT_SCHEMA,
            "report": {"status": "ok", "stage": "paired", "message": ""},
            "execution": {"provider": "bounded-v2-worker", "paired": True},
            "paired": document,
        }
    except Exception as exc:  # noqa: BLE001 - the parent reads this envelope instead of a traceback
        envelope = {
            "schema_version": RESULT_SCHEMA,
            "report": {"status": "crash", "stage": "paired", "message": f"{type(exc).__name__}: {exc}"[:500]},
            "execution": None,
            "paired": {
                "applied": False,
                "supported": True,
                "backend": "metal",
                "gpu": False,
                "changes_seed": False,
                "status": "crash",
                "timed_by": None,
                "measurement_scope": None,
                "reason": f"{type(exc).__name__}: {exc}"[:500],
            },
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(envelope, sort_keys=True), encoding="utf-8")
    return envelope


def main(argv: list[str] | None = None) -> int:
    """@brief ``python -m tcenv.backends.metal.paired_compare REQUEST.json``。"""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("usage: python -m tcenv.backends.metal.paired_compare REQUEST.json\n")
        return 2
    request = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    _worker_main(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
