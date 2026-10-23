"""Backend protocol for the evaluation layer.

Phase 1 is intentionally small: the Metal adapter delegates to today's functions
and no component is moved.  The protocol draws the boundary that later phases can
migrate behind without changing the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol


@dataclass(frozen=True)
class DeviceInfo:
    """@brief 一次设备探测的结果：后端名、设备名、family 与原始信息。"""

    backend: str
    name: str
    family: str
    limits: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Capabilities:
    """@brief 后端支持的功能开关集合。"""

    supports_launch_spec: bool = False
    supports_validation: bool = False
    supports_device_timestamps: bool = False
    supports_profiler: bool = False
    supports_hw_features: bool = False
    supports_feature_signature: bool = False


@dataclass
class BenchHarness:
    """@brief 计时所需的启动器、基线与抽查函数集合。"""

    rotation_sets: int
    dispatch_note: str
    protocol_note: str
    cand_run: Callable[[int], float]
    cand_serial_run: Callable[[int], float] | None
    baseline_runners: Mapping[str, Callable[[int], float]]
    gpu_samples: list[float]
    spot_check: Callable[[], list[str]]


class Backend(Protocol):
    """@brief 评测层与具体加速器之间的边界协议。

    Phase 1 is intentionally small: the Metal adapter delegates to today's functions
    and no component is moved.  The protocol draws the boundary that later phases can
    migrate behind without changing the orchestrator.
    """

    id: str
    version: str
    language: str
    file_ext: str
    display_name: str
    source_fence: str
    reference_skill: str
    skill_names: tuple[str, ...]
    skill_optimization: str
    skill_profiling: str
    skill_profiler: str
    skill_hardware: str
    runtime_headers: str
    capabilities: Capabilities

    # Parent-process methods must be CPU/text only.  They may import the
    # backend module, but must not initialise the device runtime.
    def screen(self, source: str, task: Any = None) -> list[str]:
        """@brief 静态筛查候选源码。

        @param source 候选源码。
        @param task 目标任务，可提供额外约束。
        @return 拒绝理由列表；为空表示通过。
        """
        ...

    def parse_launch(self, source: str, contract: Any) -> Any | None:
        """@brief 解析候选自带的 taic-launch 块。

        @param source 候选源码。
        @param contract 编译契约。
        @return 启动规格；没有该块时为 None。
        """
        ...

    def validation_env(self, base_env: Mapping[str, str]) -> dict[str, str]:
        """@brief 打开着色器校验所需的环境变量。

        @param base_env 基础环境。
        @return 新的环境字典。
        """
        ...

    def parse_validation_stderr(self, stderr: str) -> list[str]:
        """@brief 从校验子进程的 stderr 中提取违例。

        @param stderr 子进程错误输出。
        @return 违例说明列表。
        """
        ...

    def is_instrumented_size_only(self, violations: list[str]) -> bool:
        """@brief 违例是否只有插桩后的静态线程组内存上限一条。

        @param violations 违例列表。
        @return 仅为该条时为 True。
        """
        ...

    def validation_problems(self, violations: list[str], static_bytes: list[int], max_tg_memory: int) -> list[str]:
        """@brief 校验趟里的 ABI 违例过滤。

        @param violations 原始违例列表。
        @param static_bytes 各内核的静态线程组内存字节数。
        @param max_tg_memory 设备线程组内存上限。
        @return 过滤后的违例列表。
        """
        ...

    # Child-process methods.  The caller supplies the runtime context.
    def set_heartbeat_hooks(self, *, on_compile=None, on_launch=None) -> None:
        """@brief 注册编译与启动前后的心跳回调。

        @param on_compile ``(before, after)`` 两个编译回调。
        @param on_launch ``(before, after)`` 两个启动回调。
        """
        ...

    def context(self):
        """@brief 取得设备运行时上下文。

        @return 后端自己的上下文对象。
        """
        ...

    def compile_error(self):
        """@brief 该后端的编译错误类型。

        @return 异常类，供 ``except`` 使用。
        """
        ...

    def runtime_error(self):
        """@brief 该后端的运行时错误类型。

        @return 异常类，供 ``except`` 使用。
        """
        ...

    def probe(self, ctx: Any, target: Any = None) -> DeviceInfo:
        """@brief 探测当前设备信息。

        @param ctx 运行时上下文。
        @param target 可选的目标档案。
        @return 设备信息。
        """
        ...

    def prepare_case(self, task: Any, seed: int):
        """@brief 生成一个带参考输出的测试用例。

        @param task 目标任务。
        @param seed 随机种子。
        @return 用例数据对象。
        """
        ...

    def baseline_runners(self, task: Any, data0: Any) -> Mapping[str, Any]:
        """@brief 构建全部厂商基线 runner。

        @param task 目标任务。
        @param data0 首个用例数据。
        @return 基线名到 runner 的映射。
        """
        ...

    def baseline_runner(self, impl: Any, task: Any, data: Any):
        """@brief 构建单个基线 runner。

        @param impl 基线实现描述。
        @param task 目标任务。
        @param data 用例数据。
        @return 该基线的 runner。
        """
        ...

    def rotation_sets(self, task: Any) -> int:
        """@brief 该任务计时时轮转的缓冲组数。

        @param task 目标任务。
        @return 缓冲组数。
        """
        ...

    def coverage(self, task: Any, results: list, source: str, toolchain: dict) -> dict:
        """@brief 描述本次正确性证据的覆盖范围。

        @param task 目标任务。
        @param results 各用例的检查结果。
        @param source 候选源码。
        @param toolchain 工具链与设备信息。
        @return 覆盖范围字典。
        """
        ...

    def calibration(self, target_id: str) -> dict:
        """@brief 取该目标的峰值带宽与算力标定。

        @param target_id 目标 id。
        @return 标定字典。
        """
        ...

    def roofline(self, task: Any, time_s: float, calibration: dict) -> dict:
        """@brief 计算本次耗时相对硬件下界的位置。

        @param task 目标任务。
        @param time_s 候选耗时秒数。
        @param calibration 峰值标定字典。
        @return roofline 字典。
        """
        ...

    def assemble(self, contract: Any, source: str) -> tuple[str, int]:
        """@brief 把契约头与候选源码拼成编译单元。

        @param contract 编译契约。
        @param source 候选源码。
        @return ``(完整源码, 头行数)``。
        """
        ...

    def rewrite_compiler_lines(self, contract: Any, message: str, header_lines: int) -> str:
        """@brief 把编译器行号映回候选源码。

        @param contract 编译契约。
        @param message 编译器诊断。
        @param header_lines 头占用的行数。
        @return 重写后的诊断文本。
        """
        ...

    def describe(self, contract: Any) -> str:
        """@brief 渲染契约摘要。

        @param contract 编译契约。
        @return 多行摘要文本。
        """
        ...

    def abi_stub(self, contract: Any) -> str:
        """@brief 渲染给模型的 ABI 签名骨架。

        @param contract 编译契约。
        @return 签名骨架文本。
        """
        ...

    def compile(self, ctx: Any, source: str, contract: Any, *, math: str = "fast"):
        """@brief 编译单个内核。

        @param ctx 运行时上下文。
        @param source 完整源码。
        @param contract 编译契约。
        @param math 数学模式。
        @return 编译后的内核对象。
        """
        ...

    def compile_all(self, ctx: Any, source: str, contract: Any, *, names: list[str] | None = None, math: str = "fast"):
        """@brief 编译流水线中的多个内核。

        @param ctx 运行时上下文。
        @param source 完整源码。
        @param contract 编译契约。
        @param names 要编译的内核名；为 None 时全部。
        @param math 数学模式。
        @return 内核对象列表。
        """
        ...
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
        """@brief 反射检查编译结果是否匹配契约与启动规格。

        @param contract 编译契约。
        @param kernels 编译出的内核。
        @param max_tg_memory 设备线程组内存上限。
        @param max_buffer_index 最大 buffer 槽位。
        @param target_id 目标 id。
        @param spec 可选启动规格。
        @return 违例列表；为空表示通过。
        """
        ...
    def pipeline_stats(self, kernel: Any, contract: Any, device_info: Mapping[str, Any], *, kernel_source: str | None = None) -> dict:
        """@brief 单核的流水线统计。

        @param kernel 编译后的内核。
        @param contract 编译契约。
        @param device_info 设备信息。
        @param kernel_source 候选源码，用于源码级启发式。
        @return 流水线统计字典。
        """
        ...

    def pipeline_stats_many(self, pairs: list, device_info: Mapping[str, Any], *, kernel_source: str | None = None) -> dict:
        """@brief 多阶段流水线的统计。

        @param pairs ``(内核, 阶段规格)`` 列表。
        @param device_info 设备信息。
        @param kernel_source 候选源码。
        @return 流水线统计字典。
        """
        ...

    def prepare_correctness(self, ctx: Any, contract: Any, *, tight: bool, spec: Any = None):
        """@brief 准备正确性检查所需的缓冲区。

        @param ctx 运行时上下文。
        @param contract 编译契约。
        @param tight 是否使用紧分配（无护栏）。
        @param spec 可选启动规格。
        @return 参数集合对象。
        """
        ...

    def run_correctness(self, ctx: Any, kernel: Any, task: Any, argset: Any, seed: int, *, cases=None, baseline_fns=None, determinism: bool, launcher=None):
        """@brief 跑安全、数值正确与确定性检查。

        @param ctx 运行时上下文。
        @param kernel 编译后的内核。
        @param task 目标任务。
        @param argset 输入输出参数集合。
        @param seed 随机种子。
        @param cases 要跑的用例。
        @param baseline_fns 每个用例的基线输出提供者。
        @param determinism 是否检查确定性。
        @param launcher 可选的启动器。
        @return ``(用例结果, 安全违例, 非确定违例)``。
        """
        ...

    def prepare_bench_sets(self, ctx: Any, contract: Any, task: Any, *, spec: Any, data0: Any) -> list[Any]:
        """@brief 准备轮转使用的多组缓冲区。

        @param ctx 运行时上下文。
        @param contract 编译契约。
        @param task 目标任务。
        @param spec 启动规格。
        @param data0 首个用例数据。
        @return 缓冲组列表。
        """
        ...
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
        """@brief 构建计时 harness：候选与基线 runner、轮转组数与抽查函数。

        @param ctx 运行时上下文。
        @param kernels 编译出的内核。
        @param contract 编译契约。
        @param task 目标任务。
        @param spec 启动规格。
        @param data0 首个用例数据。
        @param reps 计时重复次数。
        @param measure_serial 是否额外测串行 dispatch。
        @param sets 轮转缓冲组。
        @param base_runners 基线 runner 映射。
        @param selected 本次要测的 runner 名。
        @return 计时 harness。
        """
        ...
