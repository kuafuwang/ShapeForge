# Backend 协议与 GPU 实现计划

先抽协议，Metal GPU 行为不变；本轮只做 Metal 一个实现，NPU 只留注册位。

状态：Phase 1、Phase 2 已实现；Phase 2 的 Metal 模块已迁入 `tcenv/backends/metal/`，旧路径保留兼容 shim；Phase 3–4 未实施（2026-10-23 审核修订）
审核：对照当前代码补入 `source_pressure` 的 Metal 归属、嵌套报告 payload、`measure_serial` 默认值、`prompt_sha1` 的实际 SHA-256 和完整 skills 列表。
Phase 1/2 实施：`tcenv/backends/{base,registry}.py`、`tcenv/backends/metal/{backend,contract,screen,launch,abi,sanitize,testing,baselines,profile,pressure,runtime,domain,verify,xctrace,triton_baseline}.py`；`tcenv/*.py` 旧路径是兼容 shim。`evaluate`/`worker` 经 adapter，报告 schema 3、backend_version/timeline、索引 v11 和 `runs.backend` 已落地；Phase 3 提示/文件扩展和 Phase 4 shim 清理仍未完成。GPU 基线尚未采。
目标读者：维护 `taic` / `tcenv` 的工程师
范围：先抽协议，Metal GPU 作为第一个实现；NPU 不实现，只预留注册位和接口。
阶段编号全文只有一套，见 §11。§6–§10 是工作包（WP-A 至 WP-D），不自带 Phase 编号，所属阶段在 §11 的表里对齐。

## 0. 一句话

把 `tcenv` 从“Metal 专用评测器”改成“评测流程 + 可插拔 backend”。当前 GPU 行为保持不变；未来新增同为 kernel 级的 accelerator 时，只增加一个 backend、对应任务和 skills，不改搜索、反馈、信任和审计主流程。只能跑图级程序的设备（如部分 NPU）不在这句承诺内，见 §2。

### 0.1 术语（首次出现以此为准）

| 词 | 白话 |
|---|---|
| `backend` | 一套语言加运行时的实现（如 `metal`），回答“源码怎么筛、怎么编、怎么跑” |
| `target` | 一台具体设备的档案（如 `apple-m4-max`），回答“在哪台机器上、限制是多少”；一个 target 只指向一个 backend |
| `protocol_version` | 计量协议的代次（现为 `"3"`）。计时窗口或确认批次一变就升代，不同代的结果不能混算 |
| `EvalReport` | 一份候选的判决书：状态、停在哪一关、正确性、计时都在里面 |
| `roofline` | 用峰值带宽/算力估出的理论下界，用来看候选离硬件上限还有多远 |

## 1. 目标

1. 定义稳定的 backend 协议，覆盖设备探测、源码筛选、启动规格、编译、ABI 检查、执行、正确性、计时原语、profile 原语和标定量。
2. 把现有 Metal 代码收进 `MetalBackend`，作为协议的第一个完整实现。
3. `taic` 的搜索、委员会、预算、反馈、review、trust、overlay、P2、observe 数据层不依赖 Metal 细节。
4. 新增 backend 时可以复用 `EvalReport`、`Task` 的通用部分、benchmark 协议、run 目录和 SQLite 数据层。
5. 为 kernel 级 NPU 保留清晰的扩展点，但本轮不新增 NPU target、不写 NPU runtime、不伪造 NPU 评测。
6. 从协议落地起，结果能回答“哪个 backend、哪台设备、哪个阶段失败”，且不跨 backend 混算。可观测性字段分阶段加，见 §10。

## 2. 非目标

- 不实现 NPU、ANE、CUDA、ROCm 或任何第二个 backend。
- 不改变当前搜索策略、委员会轮换、预算规则、promotion 语义。
- 不改变 `feedback.jsonl`、`feedback_review.json`、trust 文件格式。
- 不把 Metal 的线程组/`simdgroup` 概念硬塞进通用接口。
- 不在本阶段重写 `tcenv/evaluate.py` 的全部内部逻辑；先加适配层，再分阶段迁移。
- 图级 NPU（只能跑图/IR、没有 kernel 启动模型）不在本计划的扩展承诺内。它的搜索层级要从 kernel 改成 graph/operator，需要另立计划，不能硬套本协议和 `Task`（见 §12）。

## 3. 现状与耦合点

### 3.1 已经通用的部分

- `taic/orchestrator.py`：搜索、轮次、预算、委员会、候选保存、确认复测。
- `taic/agent_feedback.py`、`taic/harness.py`、`taic/review.py`：反馈解析、结构提案、trust、overlay、P2、review。
- `observe/`：SQLite 索引、run 目录、dashboard、suggestions、review queue。
- `tcenv/bench.py::interleaved`：交替计时和 CV 统计本身不依赖 Metal。
- `tcenv/evaluate.py::EvalReport`：状态、阶段、正确性、benchmark、roofline 的序列化结构，已有 `backend`、`device_name`、`gpu_family`、`device_warnings`、`elapsed_s` 字段。
- `TargetProfile` 已有 `backend`、`language`、`file_ext` 字段（后两者读自 `backends/<name>/backend.toml`）；`targets/_template` 和 `backends/_template` 已存在。
- `run_config.json` / `result.json` 已记录 `target_id` / `backend`；SQLite `runs` 表没有 `backend` 列，只有 `target_id`，backend 只出现在行的 `row_json` 里，不可直接过滤（更正旧稿）。

### 3.2 仍然是 Metal 专用的部分

| 位置 | 现状 |
|---|---|
| `tcenv/target.py` | `_BACKENDS = {"metal"}`；`Limits` 四个键为 `max_buffer_index`、`max_threads_per_threadgroup`、`max_threadgroup_memory_bytes`、`counter_sample_us`；`device_matches` 检查 `gpu_family` |
| `tcenv/metal_runtime.py` | PyObjC/Metal 编译、pipeline、buffer、命令缓冲、GPU 时间戳 |
| `tcenv/evaluate.py` | 主流程直接使用 `get_context()`、`MetalCompileError`、MSL contract、`ArgSet`、Metal launcher；父/子进程、心跳期限、GPU 锁、临时 `kernel.metal` 都在这一层 |
| `tcenv/worker.py` | 子进程入口直接 `from .metal_runtime import set_compile_hooks, set_launch_hooks` 绑定 `"C"`/`"B"`/`"E"` 心跳 |
| `tcenv/contract.py` | `AbiArg.msl_declaration`、`Contract.render_header/assemble/abi_stub` 生成 MSL；`Contract` 本身带 `grid/threads` 与 buffer 下标，是 MSL 启动契约 |
| `tcenv/screen.py` | MSL 源码规则；父进程和子进程各跑一次 |
| `tcenv/launch.py` | `/* taic-launch */` MSL 启动规格：父进程用语法解析算看门狗期限，子进程再结合 `dev_info` 查硬件上限，失败记 `rejected` |
| `tcenv/sanitize.py` | `ArgSet` 与护栏 buffer（旧稿误写在 `testing.py`，此处更正），以及 `env_with_validation`、`validation_problems` |
| `tcenv/testing.py` | `prepare_case`、`check_case`、`launch_once`、`run_correctness`；安全与正确性同一次运行 |
| `tcenv/gpulock.py` | 机器级 GPU 锁：不计时阶段共享，测速段独占；等锁时间已记在 `benchmark.gpu_lock_wait_s` |
| `tcenv/abi.py` | `CompiledKernel`、Metal pipeline 反射、`max_total_threads`、threadgroup 上限、`only_static_threadgroup_limit`（插桩翻倍豁免） |
| `tcenv/profile.py` | Metal pipeline 统计、Metal 峰值带宽/算力与标定；当前还会把 `source_pressure` 拼进 pipeline 统计 |
| `tcenv/pressure.py` | 从 MSL 源码统计 `simdgroup_float8x8` 槽数/最大数组，作为 report 的源码级提示；不是通用 backend 指标 |
| `tcenv/baselines.py` | MLX / PyTorch-MPS runner 与 `rotation_count(task)`，Metal 上执行 |
| `tcenv/triton_baseline.py` | Triton→MSL 的编译器对照基线，挂在 bench 之后 |
| `tcenv/report.py` | `format_feedback` 写死 `"Metal compiler errors"`、`threadgroup memory`、`maxTotalThreadsPerThreadgroup`、`SIMD width`、`source pressure`、`triton_msl` |
| `tcenv/xctrace.py` | Instruments Metal System Trace 录制与计数器解析（供 `tcenv profile` 与 `orchestrator` 慢核证据使用） |
| `tcenv/domain.py` & `tcenv/verify.py` | `domain` 覆盖范围文案拼入 `macos/MSL/math_mode`；`verify` 写死无 MSL/AIR 形式等价工具 |
| `tasks/*/task.py` | `naive_msl`、MSL `Contract`、`grid/threads` |
| `taic/kernel_agent.py` | 模型真正读的 `TASK.md` 在这里生成：写死“写成 Metal”、```metal 骨架、`stage_guidance` 拷 `msl_reference.md`；种子与交卷文件名固定 `.metal` |
| `taic/prompts.py` | 系统提示固定“Triton → MSL”；技能名写死，技能文件则已按 backend 目录选（`taic/target.py::skill_path`） |
| `taic/parse.py` | `extract_metal`：taic-launch 块优先、同名多块取最后，另有整文件兜底 |
| `taic/hwfeatures.py` | 解析 MSL/AIR、Apple GPU 特性；结果经 `hwfeedback` 回灌进反馈与 `TASK.md` 证据 |
| `taic/policy.py` | `feature_signature(source)` 调 `hwfeatures.analyze_source` 与 `tcenv.launch.declared_stage_kernels` 为父代多样性归档打 7 维 MSL 特征 |
| `taic/feedback_extra.py` | `threadgroup_limit_note` 正则匹配静态 threadgroup 内存超限并注入修复提示 |
| `taic/kernel_catalog.py` | `archive_confirmed` 记录 `gpu_family`，默认写 `best{file_ext}` 并回退查找 `best.metal` |
| `tcenv/cli.py` | `eval/naive/profile` 的 MSL 语义；`triton-msl` baseline；xctrace |
| `backends/metal/` | 只有 `backend.toml`、skills/wiki，没有 backend 代码入口 |

### 3.3 三处容易误读的事实（实施前先认清）

1. `EvalReport` 的顶层是封闭 schema：`REPORT_SCHEMA_VERSION = 2`（`tcenv/schema.py`），不认识的顶层字段一律拒读。给它加任何顶层字段都必须先升 schema 版本，不是“加个可选字段”就行。**嵌套字典不受这条校验**：`pipeline`、`benchmark` 等字典新增键不会触发 schema 失败；`pipeline.source_pressure` 就是这样进入报告的。Phase 0 必须快照嵌套 payload，Phase 1 必须原样保留，不能把嵌套键变化当作“没有报告变化”。
2. `prompt_sha1` 是历史字段名；`taic/trust.py::current_prompt_sha1()` 实际对提示源文件逐字节算 SHA-256，不是对输出算的。它覆盖 `taic/prompts.py`、`taic/kernel_agent.py`、`taic/harness.py` 三个文件。改这三个文件里任何一行，哪怕生成的提示词一字不变，旧 grant 也会失效。
3. SQLite 索引没有在线迁移：`INDEX_SCHEMA_VERSION = 10`（`observe/index.py`），版本对不上就整库丢弃重建。10 万 run 冷建约 157.6 秒（`docs/架构/数据层.md`）。给索引加列等于安排一次全量重建，不是顺手的事。

## 4. 设计原则

1. **GPU 零行为变化**，口径全文统一如下，不再有第二种说法：已有的字段名、语义、状态码、阶段名、prompt 输出字节保持不变；只允许新增可选字段，且每个 Phase 必须列明本阶段新增哪几个字段。Phase 1 不新增任何**顶层**报告字段，也不改变现有嵌套 payload；当前 `pipeline.source_pressure` / `hints` 必须原样保留（见 §3.3、§11）。
2. **协议只描述阶段，不描述 Metal 术语**：通用接口里不出现 `simdgroup`、`threadgroup`、`gpu_family`。
3. **backend 拥有语言细节**：语言名、扩展名、源码筛选、编译诊断、ABI、启动规格、profile 原语都由 backend 提供。语言与扩展名的唯一权威是 `backend.toml`，`Capabilities` 从它读，不另设一份会漂移的副本。
4. **任务只声明语义和参考**：任务不假设 MSL；参考源码按 backend id 存放（见 §8）。
5. **失败可定位**：每个阶段仍写 `stage/status/message`，不把所有错误压成一个 backend 异常。失败通道见 §6.3。
6. **不跨 backend 共享证据**：可比性键只有一张表，见 §9.1。全文其他章节只引用这张表，不再重述。
7. **NPU 只留扩展位**：没有实现、没有 target、没有默认 skills；未知 backend 仍明确拒绝。图级设备是明确的例外，不留位（见 §2、§12）。
8. **计时协议不进 backend**：交替计时、CV、重测、噪声门、确认批次属于测量协议，由编排层掌握。backend 只提供“跑一次/跑一批”的原语，否则每个 backend 各写一套计时，结果不可比且无人察觉。

## 5. 目标架构

下图是 Phase 2 完成后的终态，不是 Phase 1 的形态。Phase 1 只新增 adapter 文件，不搬动任何现有文件，形态以 §11 的 Phase 1 一格为准。

```text
taic/
  orchestrator / committees / policy / budget
  feedback / harness / review / trust
        |
        |  evaluate_fn(...) -> EvalReport
        v
tcenv/
  evaluate.py             # 只做阶段编排、状态和报告
  backends/
    base.py               # Protocol + DeviceInfo + Capabilities + 请求/结果
    registry.py           # backend.toml -> entrypoint 加载
    metal/
      backend.py          # MetalBackend 实现协议
      contract.py         # MSL render/assemble/abi_stub（从 tcenv/contract.py 迁入或委托）
      runtime.py          # 包装 metal_runtime
      screen.py           # 包装 screen
      abi.py              # 包装 abi
      profile.py          # 包装 profile
      baselines.py        # 包装 baselines
  target.py               # 通用 target 加载；backend 由注册表验证
tasks/
  task.py                 # 通用 Task：语义、cases、reference、tolerances
  <task>/task.py          # 通用元数据 + backend 源码映射
```

## 6. 工作包 A：Backend 协议

文件建议：`tcenv/backends/base.py`。

### 6.1 阶段以现有评测为准

协议的阶段划分对齐 `tcenv/evaluate.py` 现在的停点，不另发明一套。现有流程不是一条直线，有三个容易漏掉的结构，协议必须如实保留：

| 阶段（`rep.stage`） | 谁做 | 说明 |
|---|---|---|
| `device_check` | backend.probe + 编排判决 | 设备不匹配记 `rejected`，在碰 GPU 前返回 |
| `screen` | backend | 源码筛。父进程先跑一次（拒绝就不起子进程），子进程再跑一次（拒绝理由要进反馈） |
| （启动规格解析） | backend | `/* taic-launch */` 分两步：父进程先做纯语法解析算流水线看门狗期限；子进程编译前再结合 `dev_info` 与轮转集数查上限，失败记 `rejected` |
| `compile` | backend | 契约头拼装与编译（入参含已解析的 `launch_spec`） |
| `abi` | backend | ABI 反射检查；pipeline 统计也写在这一关，没有独立的 profile 阶段 |
| `correctness` | backend 原语 + 编排判决 | 安全（护栏、改输入、非确定）与数值正确性是同一次 GPU 运行：有安全违例先记 `unsafe`，安全干净而数值不对才记 `incorrect`。两者不能拆成协议里互不相干的两个方法 |
| `bench` | 编排（计时协议）+ backend 会话原语 | 先预分配 `R` 套轮转护栏 buffer 并构造候选/基线计时闭包，再由编排层跑 `interleaved`、CV 与最多 3 次重测，测完后对发射过的 buffer 做护栏与首尾数值抽查（失败改判 `unsafe`） |
| `done` | 编排 | roofline 在 bench 末尾由编排层按公式算，backend 只提供峰值/标定量；可选附加 `compiler_baseline` |

另外三个不在阶段序列里、但协议必须显式对齐的机制：

- **对外 `mode` 与子进程 `mode` 的映射**：`evaluate()` 对外接收 `"quick" | "timed" | "full"`；子进程 `evaluate_in_process()` 接收 `"validate" | "quick" | "full"`。其中 `"timed"` 在父进程按 `validate` 标志可选先跑一次 `"validate"` 子进程，再跑一次不插桩的 `"full"` 子进程，最后把 `rep.mode` 写回 `"timed"` 并追加 `self_checks.jsonl`。
- **validation 子进程三件套**：`full`（及带 `validate=True` 的 `timed`）在 bench 前起一个插桩子进程，期限翻倍。backend 需向父进程提供三项原语：注入插桩环境变量（`validation_env`）、解析子进程 `stderr` 中的校验违例（`parse_validation_stderr`）、以及判断 ABI 失败是否仅因插桩使静态线程组内存翻倍（`is_instrumented_size_only`，若是则放行给后面的非插桩子进程去比目标档案上限）。
- **父/子进程切分与惰性导入硬约束**：
  - **父进程只做纯 CPU/文本动作**：`screen`、`parse_launch`（算 `deadline_for_source`）、构造 `validation_env`、解析 `stderr`、用 `target.file_ext` 落临时源文件路径。**硬约束**：父进程调用 `load_backend("metal")` 时，模块顶层不得 `import Metal` / PyObjC / `torch.mps`；设备上下文只能在子进程内惰性初始化。
  - **子进程做设备动作与心跳挂接**：`tcenv/worker.py` 启动后通过 `backend.set_heartbeat_hooks(on_compile=..., on_launch=...)` 绑定 `"C"`/`"B"`/`"E"` 心跳，随后执行 `probe`、`check_launch_limits`、`compile`、`check_abi`、`run_correctness`、`prepare_bench`。父进程的 `supervise()` 和 `gpulock`（共享→独占升级）完全留在编排层。

### 6.2 协议草案（按父/子进程与会话原语拆，不收成一个 `benchmark()`）

```python
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol


@dataclass(frozen=True)
class DeviceInfo:
    backend: str
    name: str
    family: str
    limits: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Capabilities:
    # language / file_ext 不在这里另存，从 backend.toml 读，避免两处漂移。
    supports_launch_spec: bool
    supports_validation: bool
    supports_device_timestamps: bool
    supports_profiler: bool


@dataclass
class CompileRequest:
    task: Any
    source: str
    mode: str                      # worker 内部模式: validate | quick | full
    contract: Any                  # 通用契约；Metal 实现在内部转成 MslContract
    target: Any
    launch_spec: Any = None        # parse_launch 的产物，compile/check_abi/pipeline_stats 共用
    device_info: DeviceInfo | None = None


@dataclass
class Compiled:
    artifact: Any                  # 唯一允许不透明的对象，由 backend 解释
    diagnostics: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchHarness:
    rotation_sets: int
    dispatch_note: str
    protocol_note: str
    cand_run: Callable[[int], float]
    cand_serial_run: Callable[[int], float] | None
    baseline_runners: Mapping[str, Callable[[int], float]]
    gpu_samples: list[float]       # 每次 cand_run 追加设备时间戳样本（秒）
    spot_check: Callable[[], list[str]]  # 测速后查护栏与首尾套数值，非空即判 unsafe


class Backend(Protocol):
    id: str                        # 必须与 backend.toml 的 id、目录名、target.backend 四者一致
    version: str                   # backend 代码版本；与 backend.toml 的 version 一致
    capabilities: Capabilities

    # --- 父进程可调用（纯文本/环境，不得触发 GPU 初始化） ---
    def screen(self, source: str, task) -> list[str]: ...
    def parse_launch(self, source: str, contract) -> Any | None: ...
    def validation_env(self, base_env: Mapping[str, str]) -> dict[str, str]: ...
    def parse_validation_stderr(self, stderr: str) -> list[str]: ...
    def is_instrumented_size_only(self, abi_violations: list[str]) -> bool: ...

    # --- 子进程内调用（worker 进程） ---
    def set_heartbeat_hooks(
        self,
        *,
        on_compile: tuple[Callable[[], None], Callable[[], None]] | None = None,
        on_launch: tuple[Callable[[], None], Callable[[], None]] | None = None,
    ) -> None: ...
    def probe(self, target) -> DeviceInfo: ...
    def check_launch_limits(self, launch_spec: Any, task, device_info: DeviceInfo) -> None: ...
    def compile(self, request: CompileRequest) -> Compiled: ...
    def check_abi(self, compiled: Compiled, task, request: CompileRequest) -> list[str]: ...
    def pipeline_stats(self, compiled: Compiled, task, request: CompileRequest) -> dict: ...
    def run_correctness(self, compiled: Compiled, task, cases, seed: int, *, request: CompileRequest) -> Any: ...
    def prepare_bench(
        self,
        compiled: Compiled,
        task,
        case_data,
        *,
        request: CompileRequest,
        measure_serial: bool = False,
    ) -> BenchHarness: ...
    def calibration_peaks(self, target_id: str) -> dict: ...
```

约定：

- 只有 `Compiled.artifact`（及 `launch_spec` 内部阶段表）可以不透明。任务、契约、违例列表、`BenchHarness` 必须有明确形状。
- `benchmark()` / `profile()` / `roofline()` 不作为协议方法出现：交替计时循环、CV 重测和 roofline 公式都在编排层；backend 只给 `prepare_bench`、`pipeline_stats`、`calibration_peaks` 三个原语。`prepare_bench` 把预分配轮转 buffer、候选与基线 runner、GPU 时间戳收集和测速后 `spot_check()` 封成一次会话，避免无状态 `launch()` 漏掉测速后安全抽查。`measure_serial` 的默认值与当前 `evaluate` 一样是 `False`；adapter 必须把调用方显式传入的值原样传给 backend，不能靠协议默认值改变计时字段。
- `pipeline_stats` 可以返回空字典；不要求每个 backend 都有 Instruments 级证据。`triton_baseline` 保持为 Metal 侧可选附加信息，失败不改判。
- `Capabilities` 决定 `tcenv eval` 是否允许 `--validate`、是否有设备时间戳、是否有 profile。

### 6.3 失败通道（照现有状态码映射，不新发明）

| backend 行为 | 编排层落点 |
|---|---|
| `screen` / `parse_launch` 返回违例 | `status=rejected`，`stage=screen`，违例进反馈 |
| `check_abi` 返回问题列表 | `status=abi_error`，`stage=abi` |
| `compile` 抛 backend 编译异常 | `status=compile_error`，`stage=compile`，诊断按契约头行数回译 |
| 正确性运行返回安全违例 | `status=unsafe`（优先于数值错误） |
| 正确性运行数值不对 | `status=incorrect`，定位到第一个失败 case |
| 启动/运行抛 backend 运行异常 | `status=runtime_error` |
| 子进程超时或被杀 | `status=timeout` / `crash`，由父进程判决，与 backend 无关 |

新增诊断细节（如 `error_kind`）只能挂在已有字段旁边，不替代 `status/stage/message`。这一节属于哪个 Phase 落地，见 §11；失败通道本身在 Phase 1 就要保持不变。

## 7. 工作包 B：注册与配置

### 7.1 `backend.toml`

现有 `backends/metal/backend.toml` 已有字段：

```toml
schema_version = 1
id = "metal"
language = "msl"
file_ext = ".metal"
runtime = "metal"
```

本计划给它增加：

```toml
version = "1"
entrypoint = "tcenv.backends.metal.backend:MetalBackend"
skills = ["apple_gpu_optimization", "metal_profiling", "msl_reference", "profiler_evidence", "hardware_feedback"]
```

已决定，不再列为开放问题：

- `entrypoint` 只允许仓库内的 `tcenv.backends.*`。注册表加载时校验这个前缀，拒绝任意包路径，避免“写个 toml 就能让评测器 import 任意代码”。
- 加载时强制 `backend.toml:id` == 目录名 == `target.toml` 的 `backend` 字段，三者不一致直接拒绝。现在的 `load_target` 不校验 backend.toml 的 `id`，新注册表必须补上，否则后面参考源码的 key 会出现两种写法。
- `schema_version` 是 toml 结构版本，`version` 是 backend 代码版本，两者不是一回事，文档和代码里都不得混用。

`tcenv/target.py` 不再写死 `_BACKENDS = {"metal"}`，改成注册表：

```python
# tcenv/backends/registry.py
def load_backend(name: str) -> Backend:
    doc = tomllib.loads((BACKENDS / name / "backend.toml").read_text())
    module, cls = doc["entrypoint"].split(":")
    ...
```

- 未注册 backend 仍抛 `TargetError`。
- `backends/_template/backend.toml` 补上 `entrypoint`、`skills` 的注释，说明未来 backend 需要实现什么。
- 不新增 `targets/npu-*`，不把 `_template` 设为 active。

### 7.2 `target.toml` 与 `Limits`

现在的矛盾先写明：四个 GPU 键既是必填、又想改成可选，不能同时成立。结论如下：

- 通用 `TargetProfile` 只保存 `backend` 和原始 `[limits]` 表（`raw_limits: Mapping[str, Any]`），不再在通用层要求那四个键。
- Metal backend 自己校验它需要的四个键，缺了就拒绝，拒绝的落点仍是 `stage=device_check`、`status=rejected`。
- `device_matches` / `limit_notes` 是 Metal 语义，迁入 `MetalBackend.probe`。注意它现在的语义是“设备报告值与档案不一致时，档案限制继续生效，只给一条提示”，迁移后保持不变。
- 校准、经验、skills/wiki 的目录约定不变。

## 8. 工作包 C：任务、提示与 agent

这一包是最容易改错的一包，因为它同时牵动任务指纹、信任和模型真正读到的文本。逐项写清：

### 8.1 参考源码：key 用 backend id

已决定，不再列为开放问题：key 是 **backend id**（如 `"metal"`），不是语言名（`"msl"`）。理由：两个 backend 可以同语言不同 ABI。`backend.toml` 里 `id` 与 `language` 是两个命名空间，混用迟早撞上。

- `Task.naive_msl` 改为 `reference_sources: dict[str, str]`，按 backend id 取。
- 但 `get_task(name, dtype, size)` 现在没有 backend/target 入参，源码在任务构造时就以字符串灌好了。所以这不只是改个字段名：要给任务构造或评测入口一条“按 `target.backend` 选源”的路径。
- 过渡期保留 `naive_msl` 属性，返回 `reference_sources.get("metal", "")`，让旧调用点先不动。

### 8.2 `naive_msl` 不只是参考源码

【实施前必须认清】`naive_msl` 现在身兼三职，改名或多值化都会动到它们，本计划逐项给出处理方式，不留给实施者猜：

1. **评测自检核**：`tcenv naive` 把它走一遍完整 `evaluate()`，验证评测架本身没坏。某个 backend 没有参考源码时，自检行为要明确定义（跳过并报“无参考”，不能静默通过）。
2. **任务指纹成分**：`task_fingerprint` 直接哈希 `task.naive_msl`（`tcenv/task.py`）。改成多值后，指纹按“当前 backend 的那份源码”取值，并写明旧 run 的指纹因此不可比、要靠 `git_commit` 追溯。
3. **硬件特性天花板基准**：`hwfeatures` 以任务的朴素标量核为“无特性天花板”。参考源码缺失的 backend，这条天花板没有定义，要明确关掉而不是拿别 backend 的源码顶替。

### 8.3 `Contract` 的拆分有一个未决点，先决议再写代码

`Contract` 不是“把 `render_header/assemble` 拆出去”就能通用化的：它本身就是 MSL 启动契约，`grid/threads`、`(name, msl_type, value)` 常量、`[[buffer(n)]]` 都在通用类里，每个 `Task` 还直接持有它（`tcenv/task.py`），各题把 grid/threads/constants 钉死在构造里。

决议（Phase 2 已落）：`grid/threads`、buffer 下标和 `msl_type` 继续留在通用 `Contract`，因为它们同时描述启动几何和 ABI；MSL 文本渲染（`assemble`/`describe`/`abi_stub`/`rewrite_compiler_lines`）由 `MetalBackend` 提供，Phase 2 仍委托现有 `tcenv/contract.py` 实现，保证描述文本和指纹逐字节不变。文件迁移/删除留到 Phase 4，不在协议迁移中改变 `contract.describe()`。还有一条容易漏的稳定性约束：`Contract.describe()` 的文本被哈希进 `run_config.json`（`contract_description_sha256`），拆分类或改渲染顺序若动了描述文本，Phase 2 的“报告不变”验收测不出来。Phase 2 的验收必须加一条：`contract.describe()` 与 `task_fingerprint` 逐字节快照对比。

### 8.4 提示面是 `TASK.md`，不是系统提示

【实施前必须认清】模型写核真正读的是每场会话目录里的 `TASK.md`，由 `taic/kernel_agent.py::render_contract` 生成，`hmz` 只读它做事。系统提示（`taic/prompts.py`）只是其中一层。若只把系统提示参数化，Phase 3 的验收过了也等于没解耦，因为 `TASK.md` 里还有一整层写死的 Metal 文案：

- “把 Triton 算子写成 Metal”的题面句；
- ```metal 骨架与种子文件语义；
- `stage_guidance` 固定拷的 `msl_reference.md`；
- `SessionBrief` 里塞入的 `contract.describe()` / `abi_stub()`（经 `taic/orchestrator.py`）。

Phase 3 的提示面清单按这四层列，不按文件列。通用层从 `backend.toml` 声明的 `skills` 名单按名装载技能文件（文件仍从 `backends/<backend>/skills` 取，现已如此），不再由 `language/file_ext` 推导技能名。

### 8.5 文件名表面比 `.metal` 大得多

输出扩展名改由 backend 的 `file_ext` 决定时，受影响的不止 `kernel_agent` 的输出/提交名。评测临时文件、归档与续跑路径同样按文件名找内核，必须一并列入 Phase 3 的验收：

- `tcenv/evaluate.py` 父进程落盘的临时源文件 `kernel.metal`（改为按 `target.file_ext` 命名）；
- `taic/orchestrator.py` 的 `kernel.metal` / `best.metal` 写读与 `--from` 查找；
- `taic/resume.py` 按 `kernel.metal` 重建候选；
- `taic/kernel_catalog.py` 的 `best{file_ext}` 归档与 `best.metal` 回退查找；
- `taic/kernel_agent.py` 的种子文件名与交卷名后缀。

验收口径：改名后，“种子只读、交卷文件缺失记 `agent_failed`”等既有语义逐项不变。AGENTS.md 规定的委员会与 `agent_failed` 交接规则不在本计划改动范围内。

### 8.6 `extract`、`hwfeatures`、`policy` 与反馈文案的归属

- `taic/parse.py::extract_metal` 不是一个 `extract_source()` 签名能无损搬走的：它带“taic-launch 块优先、同名多块取最后”的语义，`kernel_agent` 另有整文件兜底。Phase 3 必须先把这两个方法（或其替代）正式补进 §6 的协议，或明确删掉 Phase 3 对它们的引用，禁止协议外调用。
- `taic/hwfeatures.py` 与 `taic/policy.py::feature_signature` 不能被“通用流程只调 `backend.profile()`”一句带过：`hwfeatures` 是源码/AIR 静态分析加按题名查期望表，经 `hwfeedback` 回灌进反馈和 `TASK.md` 证据（`taic/orchestrator.py`）；同时 `taic/policy.py::select_diverse_parent` 直接调用 `hwfeatures.analyze_source` 和 `tcenv.launch.declared_stage_kernels` 计算 7 维特征来做父代多样性归档。归属二选一并在本节写死：随 Metal backend 迁走（非 Metal backend 下 `feature_signature` 返回空元组退化为按 speedup 选），或明确冻结在原地。无论哪种，关闭 `hw_feedback` 时 `TASK.md` 的字节必须不变，要有一条回归验收盯住。
- `tcenv/report.py`、`taic/feedback_extra.py`、`tcenv/domain.py`、`tcenv/verify.py`、`tcenv/xctrace.py` 中的 Metal 专属文案与工具（如 `"Metal compiler errors"`、`threadgroup_limit_note`、`source pressure`、Instruments 采样）：Phase 1–2 保持现有文本以确保 golden 与反馈字节不变；Phase 3–4 将编译器标题、限额提示、源码压力提示和外部 profiler 入口收归 backend（无对应能力的 backend 返回空提示）。
- `tcenv/pressure.py` 是 **Metal-specific** 的源码启发式：它只认 `simdgroup_float8x8`，不是通用协议指标。已决定随 `profile.py` 一起迁到 `backends/metal/`，由 `MetalBackend.pipeline_stats` 调用；通用 `Backend` 协议不出现 `fragment_slots` / `simdgroup`。Phase 1 先保持当前位置和输出，不改变报告。

## 9. 工作包 D：数据层、信任与可比性

### 9.1 唯一可比性键表（全文只此一处，其他章节引用它）

| 用途 | 键 | 说明 |
|---|---|---|
| trust 条目 | `target_id + task + provider/model + effort` | 当前 `taic/trust.py` 的 grant id 包含 effort；`target_id` 已隐含 backend，不再加全局 backend 键 |
| overlay | `(target_id, task, protocol_version)` | target 不同即不共享，backend 自然分开 |
| benchmark / roofline 聚合 | `backend + protocol_version` | 两者任一不同都不能算进同一个中位数或发布结论 |
| 提示词信任 | `prompt_sha1`（历史字段名；实际 SHA-256，见 §3.3） | 与上表各键独立；源文件一变即失效，处理方式见 §11 Phase 3 |

### 9.2 数据层现状（更正旧稿）

- `run_config.json`、`result.json`、`EvalReport` 已记录 `backend`。
- SQLite `runs` 表只有 `target_id`，没有 `backend` 列；backend 只在 `row_json` 里。因此“按 backend 过滤/分组”需要先补列，而补列要动索引版本（见 §3.3 第 3 条）。
- 给 `EvalReport` 加字段必须升 `REPORT_SCHEMA_VERSION`（现为 2），并处理旧进程读新文件的拒绝路径。

### 9.3 backend 版本怎么定义

`backend_version` 若只是 `backend.toml` 里手写的 `"1"`，代码改了数字没改，审计就对不上。本计划定为：版本号跟着 backend 代码一起变更，并有一条测试卡住“代码目录有改动而版本号未动”；或指纹直接覆盖该 backend 的 Python 文件。只哈希 toml 不够。仓库里已有 `git_commit` / `git_dirty`，两者一起看才可追溯。

设备限制要记两份，不能只记一份 `device_limits`：档案里强制执行的 limits，和 `probe` 读到的现场值。现在的语义是档案值生效、现场值只做提示（见 §7.2），只存一份就分不清当时用的是哪套。

### 9.4 “不混算”先是设计约束，暂不是验收项

本计划不实现第二个 backend，reserved 的 backend 又不能运行，所以“混合 backend 不进同一排名”现在无法用真实数据验收。它在本计划里是设计约束：查询与页面在实现上按 §9.1 的键分组，不得写跨组聚合的代码。等有假 backend 或历史 run 样本能构造混合数据时，再把它升级为验收项并写明样本来源。在此之前，任何 Phase 的验收都不写这一条。

### 9.5 NPU 预留

- 本轮只改 `backends/_template` 的说明文字，不新建任何 `backends/<npu>/` 目录，文档示例里也不出现假想的 NPU 名字。
- 未来的 backend 目录可以先建 toml、`status=reserved`、没有可用的 `entrypoint`；这时 `load_backend` 明确报“not implemented”，`taic run` / `tcenv eval` 选不中它。注意 `reserved` 现在是 target 的状态词，backend 借用同一个词时必须写明是“toml 在、入口不在”，不能和 target 的 `reserved` 混讲。
- 未来的 NPU backend 必须实现同一协议，不能绕过 `EvalReport`/trust/审计。

## 10. 可观测性（Phase 2 之后，不进 Phase 1）

本节整体降级：它不属于 Phase 1。先写明已经有的，避免重做：

- `EvalReport.elapsed_s` 已有总耗时；
- `benchmark` 里已有 `attempts`、`noise_cv`、`gpu_lock_wait_s`，bench 的进度事件已带 `attempt`；
- `tcenv/events.py` 已在写阶段进度。

因此本计划不设计“编译重试”这类现有流程里不存在的东西，阶段名只用现有枚举（`bench`，不是 `benchmark`）。真正要补的只有两件，且都在 Phase 2 及以后：

1. **阶段时间线**：在 `stage/status/message` 之外补一份每阶段的起止与耗时，作为审计视图，不替代状态。打点在阶段边界，不进 benchmark 的计时窗口，不额外同步 GPU。字段集在动手前先定一版最小集（`stage/status/duration_s` 起步），`attempt`、`error_kind` 等后续再加，不在第一版里全塞。
2. **backend 维度**：`EvalReport` 加 `backend_version` 与 timeline 时同步升 `REPORT_SCHEMA_VERSION` 到 3（见 §9.2）；SQLite 补 `backend` / `backend_version` 两列时同步把 `INDEX_SCHEMA_VERSION` 从 10 升到 11，并把“触发一次全量重建、10 万 run 约 157.6 秒”写进该 Phase 的验收与回滚。历史 run 没有这些字段时显示“未知版本”，不回写、不推断。

以下三项本计划只记名，不排期，不写进任何 Phase 的任务与完成定义，等真正动手时各自立项再定名字与归属：只读的 backend 能力清单接口、设备/工具链健康检查命令、dashboard 的 backend 徽标与筛选。安全约束先写在这里备查：报告与索引不写 API key、token、完整环境变量；供应商 profiler 大输出只存摘要和路径；错误文本截断后入索引。

## 11. 阶段表（全文唯一的阶段编号）

每格四栏固定：做什么、做完看到什么、本阶段不做什么、验收与回滚。§6–§10 的工作包在表头标注所属阶段。

| Phase | 工作包 | 做什么 | 做完看到什么 | 本阶段不做 | 验收与回滚 |
|---|---|---|---|---|---|
| 0 | — | 冻结基线：记录现有离线测试结果、golden 清单、三个提示源文件的 `prompt_sha1`（实际 SHA-256）与一版 `system_prompt` 输出快照、`EvalReport.to_json()` 顶层字段和嵌套 payload（含 `pipeline.source_pressure` / `hints`）、`Contract.describe()` 与 `task_fingerprint` 快照；另采一版 GPU 基线（比对字段、样本量、允许偏差写明） | 一份基线记录文件，路径在本阶段第一个 PR 里定死并写回本表 | 不改任何代码 | 验收：基线文件落盘且可复跑比对。回滚：无代码变更，无需回滚 |
| 1（已完成） | WP-A（仅 adapter 与注册表）、WP-B | 新增 `base.py`/`registry.py`/`metal/backend.py`；`target.py` 注册表化并移除 `_BACKENDS`；`evaluate()` 与 `worker.py` 经 adapter 调到现有函数；`backend.toml` 补 `version`/`entrypoint`/`skills` | `load_backend("metal")` 可用且父进程导入不触发 Metal/PyObjC 初始化；未知名字明确拒绝；既有报告字段、状态与计时数字与基线一致 | 不搬文件；不升 `REPORT_SCHEMA_VERSION`；不加 timeline、不动 SQLite、不改 prompt 与 `TASK.md`；`probe` 仍在子进程内 | 验收：未知 backend 抛 `TargetError`；父进程 `load_backend("metal")` 后 `sys.modules` 无 `Metal`；同一候选的 `status/stage/candidate_us/speedup` 与基线一致。回滚：adapter 一个提交可整体摘除，`evaluate` 回到直调路径，不动任何已有 run 数据 |
| 2 | WP-A（迁移）、WP-B（Limits）、WP-D | 先落 §8.3 的 Contract 决议，再把 screen/compile/ABI/正确性/计时的 Metal 实现迁入 backend，旧模块留薄包装；`evaluate.py` 只依赖协议；给 `EvalReport` 加 `backend_version` 与最小 timeline 并升报告 schema 到 3；SQLite 补两列并升索引版本到 11 | 通用 `evaluate` 不再 import `MetalContext`；阶段名仍是现有枚举；dashboard 能看到阶段耗时 | 不改 `Task.naive_msl` 与提示面；不删旧 import 的薄包装 | 验收：全量测试与 golden 通过；`contract.describe()`、`task_fingerprint` 与基线逐字节一致；索引重建后旧 run 可读且历史行显示“未知版本”。回滚：薄包装保留期间可切回旧路径；索引回滚等于再触发一次重建，要在发布说明里预警，不与 Phase 1 的回滚混讲 |
| 3 | WP-C | `reference_sources` 按 backend id 落地并打通按 target 选源；提示面四层参数化；`extract` 与 `hwfeatures` 按 §8.6 的归属处理 | Metal 的 `TASK.md` 与系统提示输出与基线逐字节一致；或不一致时已有重录与重新授权的记录 | 不删旧模块；不改委员会与 `agent_failed` 语义 | 验收：输出字节不变，或变化时同时满足“golden 已重录、`prompt_sha1` 新旧对照表已留、trust 已显式重新 grant”三件，缺一不可。回滚：保留旧提示生成路径一个 Phase，双路径期内可切回 |
| 4 | — | 删临时 shim；更新架构文档、`docs/代理反馈.md`、backend README、target template | 通用协议签名里没有 Metal 术语；没有未注册 backend 可运行 | 不在此阶段新增任何 backend | 验收：旧模块薄包装删除后全量测试仍过。回滚：shim 删除单独成提交，可单独 revert；删除前双路径至少保留一个 Phase |

排期归属（负责人与工期）在本计划之外另行填报；本表只定顺序、边界和每步的交付物。第一周的动作就是 Phase 0 加 Phase 1 的第一个 PR（基线文件 + 注册表单测）。

## 12. NPU 未来接入清单（本轮不做）

仅适用于 kernel 级设备：

1. `backends/<npu>/backend.toml` + `Backend` 实现 + skills/wiki。
2. `targets/<npu>/target.toml`（reserved/active）+ device/experience + calibration。
3. 任务参考源与算子约束。注意 `Task` 通体假设“单个具名 kernel + 启动几何 + 命名 buffer 的张量进出 + 逐元素容差”，接入前先逐条核对这些假设在新设备上是否成立。
4. 编译/ABI/安全/正确性/计时/profile 实现。安全语义若在新设备上没有护栏等价物，要单独定义，不能默认沿用。
5. 独立的 protocol/roofline/benchmark 标定；不与 Metal 结果混算（键见 §9.1）。
6. 独立的 prompt/skills/输出文件扩展名。

图级设备（只能跑图/IR）不在此清单内：它的 `Task` 需要 `GraphTask` 一类的兄弟抽象，搜索层级也要改，属于另一份计划，本计划只在此声明边界，不预留字段。

## 13. 风险与回滚

| 风险 | 处理 |
|---|---|
| 抽象过度，GPU 行为变化 | Phase 1 只做 adapter，不改内部；与基线逐字段比对（比对字段与偏差在 Phase 0 定死） |
| 提示源文件改动导致 trust 失效 | 这是必然结果而非意外：Phase 3 把重新 grant 写成前置步骤，见 §3.3、§11 |
| `EvalReport` 字段被 backend 污染 | 通用字段保持；backend 细节放 `metadata`；加字段必须先升报告 schema（§9.2） |
| 错误状态被压扁 | 每个阶段仍映射原 `stage/status/message`；新增 `error_kind` 不替代状态（§6.3） |
| backend 结果被混算 | 可比性键只有 §9.1 一张表；本计划期内是设计约束，不写成验收（§9.4） |
| timeline 拖慢计时 | 阶段边界打点，benchmark 内部不额外同步；只索引摘要 |
| 供应商日志泄露凭据 | 统一截断与脱敏；大 profiler 文件只存路径/摘要（§10） |
| NPU 需求提前污染接口 | 通用接口不含 `simdgroup/threadgroup`；NPU 用自己的 limits/metadata；图级需求不进本协议（§12） |
| 索引加列触发全量重建被误当成“顺手改动” | 明确写进 Phase 2 的验收与发布预警：10 万 run 约 157.6 秒（§10） |
| 旧模块导入破坏 | 迁移期保留 `tcenv/screen.py` 等薄包装，Phase 4 再删（§11） |

总体回滚原则：每个 Phase 独立提交；Phase 1 的 adapter 可整体摘除回到旧 `evaluate` 路径；不修改现有 run 数据。涉及 schema 与索引版本的 Phase 2，其回滚不是免费的，必须按 §11 该格的写法单独预警。

## 14. 文件级清单（按 Phase 分组，不再平铺）

Phase 1 新增：

- `tcenv/backends/base.py`
- `tcenv/backends/registry.py`
- `tcenv/backends/metal/backend.py`
- `tests/test_backend_registry.py`
- `tests/test_backend_metal_adapter.py`

Phase 1 修改：

- `tcenv/target.py`
- `tcenv/evaluate.py`（只加 adapter 调用点）
- `tcenv/worker.py`（心跳钩子经 `backend.set_heartbeat_hooks` 挂接）
- `backends/metal/backend.toml`
- `backends/_template/backend.toml`

Phase 2 及以后（各自在对应 Phase 的 PR 里再确认）：

- `tcenv/backends/metal/{contract,screen,abi,profile,pressure,baselines,xctrace}.py`
- `tcenv/{contract,screen,abi,profile,pressure,baselines,sanitize,testing,launch,metal_runtime,gpulock,triton_baseline,report,domain,verify,xctrace,cli}.py`（迁移或留薄包装）
- `tcenv/schema.py`（报告 schema 升到 3）
- `observe/index.py`（索引版本升到 11，补两列）、`observe/query.py`、`observe/app.py`
- `tasks/__init__.py`、`tcenv/task.py`、`tasks/*/task.py`（Phase 3）
- `taic/{prompts,kernel_agent,parse,hwfeatures,policy,feedback_extra,kernel_catalog,orchestrator,resume}.py`（Phase 3，清单见 §8）
- `docs/架构/数据层.md`、`docs/架构/可观测.md`（对应 Phase 落地时更新）

## 15. 已决定与开放问题

已决定（从开放问题中移出，正文已有落点，不再讨论）：

1. 参考源码的 key 用 backend id，不用语言名（§8.1）。
2. `entrypoint` 只允许仓库内 `tcenv.backends.*`，加载时校验前缀与 id 一致性（§7.1）。
3. `protocol_version` 保持数字代次，另带 `backend` 字段，聚合按两者分组（§9.1）。不在版本号里塞 backend 前缀。
4. `tcenv/pressure.py` 的 `simdgroup_float8x8` 源码压力提示归 Metal backend；Phase 2 随 `profile.py` 迁入 `backends/metal/`，通用协议不暴露该指标（§8.6）。

开放问题（每条标注阻塞哪个 Phase）：

1. `Contract` 的通用化边界（§8.3 的决议问题）。**阻塞 Phase 2**，动拆分代码前必须定并写回 §8.3。
2. `backend_version` 的具体算法：人工版本号加测试卡住，或覆盖代码文件的指纹（§9.3）。**阻塞 Phase 2** 的可观测部分，不阻塞协议与 adapter。
3. 图级设备是否立 `GraphTask`。**不阻塞本计划任何 Phase**，属于另一份计划的立项问题，本计划只声明边界（§12）。

## 16. Definition of Done（本计划实施完成）

- `tcenv/target.py` 不再有硬编码 `_BACKENDS = {"metal"}`，且注册表校验 id 一致性与 entrypoint 前缀。
- `tcenv/backends/base.py` 定义协议，`MetalBackend` 完整实现，阶段与失败映射符合 §6.1、§6.3。
- 通用 `evaluate` 只通过协议调用 backend；Metal 行为、已有报告字段、CLI 输出与 Phase 0 基线一致（新增字段以各 Phase 清单为限）。
- 现有全量测试与 golden 通过；prompt 与 `TASK.md` 输出与基线逐字节一致，或按 Phase 3 的三件套（重录、对照表、重新 grant）留有完整记录。
- 新增 registry/adapter 测试；未知 backend 明确拒绝；reserved（无入口）的 backend 不可运行。
- `run_config`/`result`/`EvalReport`/SQLite 都记录 backend 与 version（SQLite 一项以 Phase 2 的索引版本 11 为准）；旧 run 可读，历史缺失显示“未知版本”。
- 设计约束层面无跨 backend 聚合代码路径（§9.4）；该条在本计划期内是代码评审项，不是数据验收项。
- `backends/_template` 文档说明未来 kernel 级 backend 需要实现哪些接口。
- `source_pressure` 的 `simdgroup_float8x8` 解析归 Metal backend；通用协议和 `TASK.md` 不出现 fragment 术语，报告中的现有提示保持可用。
- 没有新增 NPU 代码、target 或默认任务；图级设备未被硬塞进 `Task`。

