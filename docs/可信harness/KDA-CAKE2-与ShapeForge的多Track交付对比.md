# 一道算子可以怎样交付：KDA、昇腾 CAKE2 与 ShapeForge 的多 Track 对比

> **文首「当前事实」按 registry 更新。正文仍是计划完成态预览，占位符不是已选技术。**
>
> 目的是把[多 DSL、平等 Track 与轻量批量探索实施计划](../架构/多DSL-Track实施计划.md)
> 里已经验收的用法和仍未做的目标分开。
>
> - **当前事实**：`msl-single/1`、`msl-bundle/1`、`tilelang-pipeline/2` 与 `/3` 是 available 的 Metal Track。`/1` 仍是 `interface_only`。新 run 在同一 track id 上取最高 available 版本，所以 TileLang 选到 `/3`。没有第三条 DSL，MT3 Graph 未启动。批量探索在 `taic/track_explore.py`；r15 是 exploration winner，不是 speedup 发布。`publication_allowed=false`。
> - **下文的目标完成态**：假设更后面的 Graph 与同输入复测也已验收。`dsl-x`、`dsl-y` 只是占位符。
> - **Graph/JIT、真正同输入复测、跨 Track shape 拼盘、清微真实 NPU Track**不在已完成范围内。
>
> 如果这份预览里的目标不是你想要的，应先改计划，不应先写实现。

---

## 一、完成后，我们到底想拥有什么

一句话：

> **同一道算子，可以让人或 LLM 分别用多种 DSL 编写候选；每条 DSL 是平等的一等 Track，拥有自己的调度与构建执行方式；各 Track 独立跑完后，用一个轻量脚本汇总结果，选择一条获胜 Track。**

目标完成态如下：

```text
                         同一道算子
              OpSpec + reference + 统一验收规则
                             │
       ┌───────────────┬─────┴──────┬───────────────┐
       │               │            │               │
 msl-single       msl-bundle      dsl-x           dsl-y
 MSL 单文件       MSL Host+Bundle  DSL X 候选      DSL Y 候选
       │               │            │               │
 MSL 启动几何     MSL + C99 Host   X frontend      Y frontend
       │               │            │               │
 MSL 编译执行     候选 Host 调度   X 调度与构建    Y 调度与构建
       │               │            │               │
       └───────────────┴─────┬──────┴───────────────┘
                             │
                共享 Metal 执行与可信评测底座
                             │
                  各 Track 独立 run_dir/结果
                             │
                 轻量批量脚本汇总、比较、选优
```

这张图里有三条最重要的边界。

### 1. `msl-bundle` 不是所有 DSL 的必经下游

DSL X 不需要先变成 `msl-bundle`，DSL Y 也不需要借用它的 `host/tiling.c`。

它们可以各自拥有：

- 候选源文件形式；
- frontend/compiler；
- Host planner 或符合自身形态的调度机制；
- work list、handoff 和内部生命周期；
- 构建参数、缓存和构建身份。

后端在 Apple GPU 上生成 MSL、AIR 或 `.metallib` 并不自动取消独立性。真正需要区分的是：

```text
是否拥有独立候选级 frontend / 调度语义 / 交付合同 / 构建身份
```

而不是“最终有没有出现 MSL 字节”。

### 2. 路线独立，不等于各自带一个黑盒裁判

所有 Apple GPU Track 仍必须共享：

- Metal buffer 生命周期与只读输入保护；
- output/workspace guard、poison；
- bounded worker；
- fresh-rotation 与 baseline/candidate 交错计时；
- CommandBuffer commit/wait 和硬件状态；
- correctness、build/evaluation identity、fresh confirmation 和发布规则。

**平等的是生成与交付路线，统一的是执行审计与裁判。**共享底座不能被扩大解释成共享
DSL、共享 Host 调度、共享 ExecutionGeometry 或共享编译路线。

### 3. 最后比较是一个轻量工具，不是新中央调度系统

各 Track 各跑各的，不写统一 Host Scheduler，不把 DSL 合并成一条 IR 流水线，也不让
TrackRegistry 暗中换路。

```text
Track A 独立跑 → result A ┐
Track B 独立跑 → result B ├→ 轻量汇总脚本 → exploration winner
Track C 独立跑 → result C ┘
```

---

## 二、当前事实与目标完成态

| 能力 | 当前事实 | 假设计划全部完成后 |
| --- | --- | --- |
| MSL 单文件 Track | 已有 `msl-single` | 保留，行为与证据不串用 |
| MSL Host+Device Track | 已有 `msl-bundle` | 保留，只是平等路线之一 |
| 非 MSL DSL Track | `tilelang-pipeline/1` 已登记为 `interface_only`（不可被新 run 选择；source-only codegen 可执行但 0 dispatch，无 Metal build/dispatch，不可确认/发布）；TileLang 已完成受信安装、身份核对、外层 source-only 隔离 probe（6/6 产物、0 dispatch、4 项真实负例，r1–r4 失败保留）与 MT1b 合同/身份/adapter，但仍没有可执行/可计时的非 MSL Track | 两条真实 DSL Track，具有独立 frontend、调度和 Metal 证据 |
| DSL 选型 | 首轮只选 TileLang `0.1.15`（commit `a35f8dd…`，wheel `dcffc9aa…`，安装 manifest `06f3b0ad…`）；Halide 不安装/不构建，第二条 DSL 以后评估；MT0 仍 blocked，MT1b 已完成 TileLang-first 的合同/身份/adapter（`interface_only`） | MT0 已选定具体名称、版本、来源和工具链；本文占位符已替换 |
| 首期 ExecutionKind | 现有 kernel pipeline | 两条新 DSL 都是 `kernel_pipeline`，不伪造 graph 几何 |
| Host 调度 | 候选自己拥有 | 各 DSL/候选继续自己拥有，不建立中央 planner |
| 嵌入式 DSL 隔离 | 已选定不改 hmz 的 lower-level Profile 路线；适配/静态与普通 CPU 验证完成，无 mds 写根；r1 失败与 r2 真实外层 observed 均保留并完成独立复核。TileLang 外层 source-only probe 已真实完成（Profile 应用、`online=false`、空 host allowlist、无 mds、0 dispatch、4 项真实负例），r1–r4 失败与修正链逐字节保留；MT0/frontend 生产化/Metal 仍未完成 | 由 sandbox 外控制机复用经审计的 hmz OS fence；ShapeForge 只管 attempt/artifact/evidence，实证后才可进入后续能力验收 |
| 多 Track 批量工具 | 没有 | 有轻量 `track-explore` 工具，逐条运行并汇总 |
| 多实例选优 | 没有现成探索工具 | 每条 Track 整包比较，最终只选一条 Track |
| Graph/JIT Track | 没有 | **仍属后续扩展**，不计入首期完成条件 |
| 跨 Track shape 拼盘 | 没有 | **仍不做**，不把不同 Track 的实例赢家混成 runtime portfolio |
| 清微 NPU | 只有模拟 | 仍然只有模拟，`simulated=true/publication_allowed=false` |

**完成计划不是“做成一个什么都能跑的通用 DSL 平台”，而是交付上述明确、可验证的最小闭环。**

---

## 三、Track 不是 DSL：完成后这个区别仍然成立

### 3.1 最简单的定义

| 概念 | 含义 | 例子 |
| --- | --- | --- |
| DSL / IR | 候选用什么语言或表示来写 | MSL、DSL X、DSL Y |
| Track | 候选怎样交付、校验、构建、执行和留证 | `msl-single`、`msl-bundle`、`dsl-x`、`dsl-y` |
| Candidate | 某条 Track 上的一份具体实现 | DSL X 写的一个 RMSNorm 候选 |
| Variant | 一个候选内部针对不同输入域的实现 | small/large 两种实现 |
| Agent | 谁生成或修改候选 | 人或 LLM |
| Backend / Target | 在哪类设备上执行 | Apple Metal、清微模拟 |
| Harness | 独立裁判与证据拥有者 | correctness、timing、confirmation、publication |

### 3.2 同一种 DSL 可以走不同 Track

当前 MSL 已经证明这一点：

```text
MSL ─→ msl-single：单文件 + 声明式启动几何
MSL ─→ msl-bundle：MSL + C99 Host Program + Geometry Bundle
```

所以 Track 不是“语言的另一个名字”。

### 3.3 不同 DSL 可以拥有平等 Track

计划完成后的结构是：

```text
MSL   ─→ msl-single / msl-bundle
DSL X ─→ dsl-x Track
DSL Y ─→ dsl-y Track
```

DSL X/Y 不经过现有 MSL Track 的合同或 adapter。它们可以复用通用
CandidateArtifact/ArtifactBundle 的打包与身份思想，但不需要采用 `msl-bundle` 的具体文件格式。

### 3.4 `available` 不等于“整个 DSL 全都支持”

每条 Track 声明的 capability 都必须有真实证据。

未覆盖的 DSL 语法、资源或 ExecutionKind 必须由 validator 返回 typed
`unsupported_contract_capability`，不能等编译器报错、进程崩溃或让 LLM 不断烧轮次试错。

扩大能力范围需要升 `track_version`。Registry 支持：

```text
(track_id, track_version)
```

多版本共存：新运行选可用版本，旧版本保留给 frozen reconstruction/resume/replay。

### 3.5 候选表达力优先于共享抽象

多 Track 最大的风险不是“路线太多”，而是为了复用抽象，把所有路线压成同一种写法：

```text
所有 DSL → msl-bundle 的 HostTilingPlan
所有候选 → 现有 GeometryPair
所有执行 → 固定 stage / grid / threads
```

如果这样做，虽然形式上有多条 Track，实际上却可能抹掉 DSL 独特的图编译、自动 tiling、布局
推导、Device 调度、persistent 状态机或内部生命周期，反而更难产生优秀内核。

因此责任分界必须是：

```text
Track 自己拥有：
  Candidate Source
  Frontend / Compiler
  Host Scheduler
  Internal ExecutionPlan
  Build / Encode Strategy

Harness 统一要求：
  Frozen Execution Identity
  Declared Buffers / Workspace
  Observable Actual Execution
  Correctness
  Timing Boundary
  Fresh Evidence
```

**现有 GeometryPair 是 `kernel_pipeline` 的一种实现，不是所有 Track 的统一上限。**执行身份应
采用带标签、可扩展的 ExecutionPlan：与现有几何相符的 kernel 路线可以复用 GeometryPair；未来
其他路线使用自己真实的执行计划，不得为了通过接口而伪造 `stage=1`、`grid=(1,1,1)` 或套用
`msl-bundle` 的 HostTilingPlan。

| 约束 | 是否必要 | 原因 |
| --- | --- | --- |
| 输入输出、dtype、读写权限 | 必要 | 否则无法定义与验证正确性 |
| workspace 声明、guard/span | 必要 | 否则无法保护内存与状态 |
| 输入相关 Host/JIT 工作计时 | 必要 | 否则不同路线测的不是同一件事 |
| bounded worker、超时回收 | 必要 | 控制失控执行，不是限制算法风格 |
| 实际执行可审计 | 必要 | 自报 JSON 不能冒充硬件事实 |
| 必须使用 `msl-bundle` | **不必要且有害** | 会压平 DSL 差异 |
| 必须使用现有 HostTilingPlan | **不必要且有害** | 会限制 Host 调度自由 |
| 必须使用现有 GeometryPair | **仅适合与其相符的 kernel 路线** | 不能成为所有 Track 的永久统一格式 |
| 固定 tiling/chunk/stage、共用 Host Scheduler | **不必要且有害** | 直接限制优秀内核的探索空间 |

首期只实现 kernel DSL 是为了控制工程范围，**不是宣布其他执行模型低一等或永远不支持**。
真正的原则是：Track 自由决定怎样做出好内核，Harness 只决定结果是否可信、能否公平比较。

---

## 四、目标完成态中的四条 Track

| Track | 候选作者交付什么 | 内部路线归谁 | 共享什么 |
| --- | --- | --- | --- |
| `msl-single` | 一个 MSL 文件和声明式启动几何 | 候选算法、stage、grid/threads | 任务、正确性、安全、计时、证据 |
| `msl-bundle` | MSL、受限 C99 Host Program、ExecutionGeometry、workspace | 候选 Host planner、work list、handoff、几何 | 同上 |
| `dsl-x`（占位） | DSL X 源文件、自己的调度/构建声明 | DSL X frontend、候选调度、编译参数和生命周期 | 同上 |
| `dsl-y`（占位） | DSL Y 源文件、自己的调度/构建声明 | DSL Y frontend、候选调度、编译参数和生命周期 | 同上 |

### 4.1 两条 DSL 不能只是换皮

以下不能算两条独立 DSL Track：

- 同一个 frontend 改个名字；
- 一个 JSON/S-expression 文件加简单字符串拼接，吐出 `.metal` 就自称新 DSL；
- 绕开现有 Python 类，但仍完整复用 `msl-bundle` 合同和 adapter；
- 把 preserved Triton v1 重新包装成生产 Track；
- 用 CUDA/CANN 路线凑数；
- 一条真实 Metal DSL 加一条清微模拟 DSL，宣称两条真实 Metal Track 已完成。

后端可以生成 MSL，但候选级抽象、优化/调度语义、构建身份和交付方式必须有实质区别。

### 4.2 Host 调度不强制共用

公开 KDA 也是类似分工：

```text
共享：任务入口、输入输出、benchmark、correctness、晋级规则
独立：各路线 Host planner、缓存、work-list 格式、handoff 和 launch 生命周期
```

ShapeForge 目标完成态保留同样的自由：

```text
msl-single：候选自己的 MSL 与启动几何
msl-bundle：候选自己的 C99 Host Program
DSL X：自己的 frontend 与调度方式
DSL Y：自己的 frontend 与调度方式
```

Track 可以提供通用构建与执行适配，但不替候选决定 LPT、range-cut、chunk、tile 或 handoff。

### 4.3 嵌入式宿主代码不是免费获得的自由

MT0 会区分：

- `declarative`：候选字节由受信 frontend 解析/编译；
- `embedded_host_code`：候选会被 import/执行为宿主 Python/C++ 等代码。

后者如果没有经实证的文件写集、网络、凭据和子进程隔离，最高只能 `interface_only`，不能标
`available`。当前 bounded worker 不是完整 OS 沙箱，import/ctypes 黑名单也不能替代沙箱。

隔离实现路线最终选择 **不修改 hmz，复用其 lower-level macOS Seatbelt Profile/离线 Proxy**，不是再维护
一套 ShapeForge Seatbelt renderer，也不是退回到削弱表达力的声明式子集。受信 driver 由 hmz uv-tool
自己的解释器在 sandbox 外运行，项目 Python 不跨环境 import hmz；它只接受严格 attempt/argv，启动
`Proxy(())`，把实际 proxy port 交给 `Profile(port=...)`，再经 `Profile.command()` 启动 rlimit shim
和候选。候选构建固定 `online=false`，受信依赖安装/准备在另一阶段完成。`port=None` 不是离线，
loopback/Unix socket/Mach service 边界也不能被写成“所有本地 IPC 已消失”。

旧 internal-fence `wrap._seatbelted()` 会额外授予 per-user `C/mds`，新路径完全不调用该包装，因此持久
写集只含 attempt/build 与非持久 OS devices，**不需要 mds，也不保留 internal/lower-level 双路径**。
ShapeForge 保留自己的 attempt 环境、临时 HOME/cache、只读输入/工具链、rlimit、进程组 watchdog、
artifact/evidence 合同；受控 Python/Clang 及其后代仍在同一个 Profile 内。缺 hmz、解释器/布局/版本/
源码/driver/protocol 漂移或 Seatbelt 不可用都 typed fail-closed，没有无隔离 fallback。

MT0-isolation-profile 已先完成适配、静态/CPU mock 与普通 shim/watchdog，随后在新 epoch 保留第一次
`outer_probe_failed` 和第二次 `outer_probe_observed`，并对 raw、探针修正、Profile/Proxy、negative controls
与身份做了不重跑 probe 的独立复核。r2 raw `checks` 实际为 18/18；这不是旧文字中的 17 项，也不是正式
production pass。真实文件/网络/凭据/普通与 `setsid` 后代及 artifact 的上述观测只覆盖所测攻击面；
loopback/Unix socket/Mach service 不宣称全部消失，`setsid` 脱离进程组缺口仍在。MT0 整体仍 blocked，
因为第二条 DSL 选型与真实 Metal 验收未完成（MT1b 的合同/身份/adapter 已单独完成，不改变这一点）。旧
[MT0-isolation](../../tests/support/golden/shared/多DSL-Track-阶段验收/MT0-isolation/acceptance.md) 仅作 internal-fence 历史；
当前证据见 [MT0-isolation-profile](../../tests/support/golden/shared/多DSL-Track-阶段验收/MT0-isolation-profile/acceptance.md)。

### 4.4 首轮 DSL 决定：TileLang 先行，source-only probe 已完成到独立复核范围

用户已决定首轮只选择 TileLang，不在本轮安装或构建 Halide；第二条非 MSL DSL 以后单独评估。
受信准备已完成：

- 固定 `tilelang==0.1.15`（上游 commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`，TVM 子模块
  `4211874e9e4b8770fd20ad0d98ab80afd1df5028`）的 macOS arm64 wheel 与 19 个依赖 wheel 安装到
  受控、可审计的 staging；每个 wheel 与 PyPI 发布 sha256 一致，license/依赖/实际安装身份
  （RECORD 21,199/21,199）已记录；
- tag 与 TVM 子模块的 Python 源与 wheel 内副本逐字节一致；native 库没有重建，不能宣称完整
  二进制可复现。

source-only frontend probe 已在外层控制机真实完成（`probe-result-r5.json`）：最低 Seatbelt 能力
探测通过，lower-level `Proxy(()) + Profile` 实际应用，`online=false`、空 host allowlist、无 mds
写根；六个构建产物按原字节归档在包内 `evidence/`（`artifact_set_sha256=bc6a7300…`），
`full_build=true`、`probe_executed=true`、`device_dispatch_calls=0`——没有编译 Metal library、
没有创建 pipeline/buffer、没有 dispatch，`host_module.txt` 只是未编译的生成 C source。前四轮
失败（r1–r3 候选 `@prim_func` parser/诊断渲染，r4 缺 Target context）与候选/entry 输入、日志、
修复链全部保留；修正仅为 `@T.prim_func` 与 `with target:`，算法/grid/threads/options 不变。
4 项真实负例全部 matched，另 5 项由冻结 adapter 合同与固定节点覆盖。

MT0 因此仍 blocked，唯一剩余阻断是第二条非 MSL DSL：用户决定本轮只做 TileLang。MT1b 已完成
task-agnostic 的 `kernel_pipeline` Track 合同、`(track_id, track_version)` 多版本 registry、身份/adapter
接缝与 CPU/协议验收（`interface_only`，42+ 固定节点、verifier `--check`）；真实 Metal
correctness/performance、生产 DSL 可用性与发布仍未开始，属 MT2 及后续阶段（尚未授权）。

证据与边界见 [MT0-tilelang-probe](../../tests/support/golden/shared/多DSL-Track-阶段验收/MT0-tilelang-probe/acceptance.md)。

---

## 五、完成后，一个人怎样实际使用

### 5.1 新增普通算子，不需要新增 Track

任务作者仍然只提供：

```text
tasks/<op>/opspec.json
tasks/<op>/reference.py
tasks/<op>/baselines.*   # 有已登记性能比较对象时
```

不需要增加：

- 逐题 pattern；
- 中央 tiling 注册表；
- 按 op/shape 划分的 Track；
- StagePartitioner 或 TileConnector。

候选作者才决定使用哪种 DSL、算法、Tiling、Host Tiling 和 ExecutionGeometry。

### 5.2 分别探索各 Track

以下是**目标接口示意，不是当前已经存在的命令**：

```bash
# 每条 Track 是独立运行，独立 run_dir、cache、build 和 confirmation
python -m taic.cli track-explore --config track_explore.json
```

配置表达的核心是：

```text
task / target
participating_tracks = [msl-single, msl-bundle, dsl-x, dsl-y]
benchmarks = [{dtype, size, usage=benchmark}, ...]
run_seed / master_seed / 输入生成器与派生算法版本
每 track×instance 的生成轮数、构建超时与费用上限
独立 run_dir / 输出 summary
```

脚本逐条调用现有单 Track 流程，不重写内部搜索，也不制造另一套 evaluator。

### 5.3 每条 Track 自己产生最佳有效结果

每条 Track 内部仍然执行：

```text
人或 LLM 生成候选
→ Track 自己的校验与构建
→ correctness / safety
→ 真实执行与性能测量
→ fresh confirmation
→ 留下有效结果或诚实失败
```

Track A 的 correctness 或确认结果不能给 Track B 背书；build、cache、confirmation、speedup
不能串用。

---

## 六、最后怎样比较：简单，但不能假装相同输入

### 6.1 单实例探索

`benchmarks` 只有一个 `(dtype, size)` 时：

1. 各 Track 独立跑完；
2. 淘汰 incorrect、unsafe、identity 不符、confirmation 失败或 timing uncertain 的结果；
3. 按确认的 `latency_ratio = candidate / baseline` 升序；
4. 选出 exploration winner 及其 Track。

若没有合格结果，输出 `no_winner`，不能降低门限制造冠军。

如果以后要额外要求 `speedup > 1.0`，需要单独批准为更严格的探索准入门，不能声称当前
subject confirmation 已天然包含它。

### 6.2 多实例探索：每条 Track 整包竞争

例如有 `small`、`paper` 两个实例：

```text
msl-single → 自己的 small 最佳候选 + paper 最佳候选 → 向量 M = [M_small, M_paper]
msl-bundle → 自己的 small 最佳候选 + paper 最佳候选 → 向量 B = [B_small, B_paper]
dsl-x      → 自己的 small 最佳候选 + paper 最佳候选 → 向量 X = [X_small, X_paper]
dsl-y      → 自己的 small 最佳候选 + paper 最佳候选 → 向量 Y = [Y_small, Y_paper]
```

每条向量叫 **per-track per-instance best bundle**。

比较的是：

```text
M vs B vs X vs Y → 唯一获胜 Track
```

不是：

```text
small 从 Track A 拿一个，paper 从 Track B 拿一个 → 跨 Track 拼盘
```

后者属于跨 Track runtime portfolio，首期不做。

### 6.3 多实例排序

先逐实例过硬门：

- correctness、identity、fresh confirmation、quiet timing；
- 每实例 effective/specialized `latency_ratio <= 1.10`；
- 不缺实例，不混入 fallback 成绩。

过门后默认按：

```text
geomean(per-instance latency_ratio)
```

等权升序，避免 paper 的长耗时垄断结果。Geomean 只在全部实例过门后排序，不能掩盖局部失败。

保持：

```text
latency_ratio = candidate / baseline
speedup = baseline / candidate
```

`latency_ratio <= 1.10` 是回归容忍门，不是 `<= 1/1.10`，也不是至少 1.10x 加速。

### 6.4 为什么首轮只叫 exploration winner

现有 fresh seed 派生包含 `portfolio_freeze_digest`，而 freeze 绑定候选字节：

```text
相同 run/master seed
+ 不同 Track/候选
→ 不同 freeze digest
→ 不同 confirmation_seed
→ 可能不同的实际 fresh 输入
```

所以首轮采用方案 A：

- 相同 run/master seed、派生算法、任务、实例、输入生成器；
- 各 Track 走自己现有的 fresh confirmation；
- 比较的是**同分布、各自确认的探索结果**；
- 不宣称“同一 fresh 输入”；
- 输出 exploration winner，正式发表仍回现有发布链。

真正同输入比较需要后续方案 B：候选无关的确认种子派生、新确认 epoch 和同输入复测。
它不是现有唯一发布谓词，也不包含在首期完成条件里。

### 6.5 `comparability_digest` 防止不同口径硬比

各 Track 必须共享并记录：

- `opspec_digest`；
- 实例 `(dtype, size, usage)`；
- reference implementation digest；
- baseline implementation/environment digest；
- target/device；
- protocol_version；
- timing discipline；
- 输入生成器与 seed 派生算法版本。

这些共同字段形成 `comparability_digest`。不一致时只展示结果，不参与选优。

每条 Track 的完整 build/evaluation identity 仍分别保留，不能因为 comparability 相同就省略。

---

## 七、完成后会留下什么结果

批量工具写出 `track-explore-summary.json`，至少包含：

- 参与的 Tracks 和 benchmark 实例；
- 每条 Track 的独立 run_dir；
- 每实例 candidate/build/evaluation identity；
- `comparability_digest`；
- run/master seed 与派生算法版本；
- 各 Track 实际 confirmation_seed 和 fresh 输入身份；
- correctness、confirmation、timing 状态；
- 每实例 latency_ratio 与淘汰原因；
- agent turns、构建耗时、搜索 wall/cost；
- selection policy digest；
- exploration winner 或 `no_winner`。

多实例 winner 记录为：

```text
(
  winning_track,
  per-instance candidate identity vector
)
```

向量按 `instance_key` 排序，每项包含：

```text
(instance_key, candidate_sha256, build/evaluation identity)
```

其 canonical digest 是 `exploration_bundle_digest`。它不是 runtime PortfolioCandidate，不启用跨
Track dispatch，也不自动写入 catalog 或历史 best。

---

## 八、KDA 的多种实现与这个目标有什么关系

### 8.1 KDA 的三条路线是平等的具体实现

固定审查版本中有：

| 路线 | 主要形式 | Host 调度与生命周期 |
| --- | --- | --- |
| CuTe | CuTe DSL / Python | `impl/forward.py::_plan()` + `impl/planner.py`，自己的 plan/cache/schedule |
| TIRx | TIRx / TVM | `prepare()` + `setup()`、`_host_item_table()`，自己的 work list 和 CUDA Graph |
| static PTX | static PTX + TVM FFI | `KDAPiecesLaunch` 和 `kda_ptx/scheduler.py`，自己的 descriptor/fallback/handoff |

它们共享外部任务 ABI、benchmark 和 correctness，**不是共享一个统一 Host Scheduler**。

这正是目标 ShapeForge 要学习的路线关系：各自优化内部，外部公平验收。

### 8.2 KDA 公开 `bench.py` 不自动跑完三条路线选冠军

实际调用是：

```bash
python bench.py cute
python bench.py tirx
python bench.py ptx
```

`KERNELS` 字典只负责选择本次模块。程序分别评测 workloads 并打印 speedup/geomean，不自动循环
三条实现、不自动选全局 winner、不自动按 shape 组合。

人可以分别跑完再比较。这与 ShapeForge 的轻量批量脚本目标相同：**外部简单汇总，不建复杂中央
Race Orchestrator。**

### 8.3 不把 KDA 的技术栈直接搬来

CuTe、TIRx、PTX、CUDA 调度与同步规则不能进入 ShapeForge 生产、pack 或 `agent_guidance`。
我们学习的是：

- 多路线平等探索；
- 各自 Host 调度；
- prepare/run 生命周期；
- 独立 fallback；
- 正确性与速度一起验收。

---

## 九、CAKE2 的 Host+Device 和 Evo variants 又是什么

### 9.1 CAKE2 的主体是昇腾原生工程流水线

典型结构是：

```text
op_desc / reference
→ DSL 与 lowering
→ op_host Host Tiling + op_kernel Device Kernel
→ 编译、安装、绑定
→ 多 case correctness / performance
```

ShapeForge 学习的是 Host+Device 协同交付，不复制 CANN/AscendC、TilingData/TilingKey、
msopgen、ACL/PyBind 或 msprof。

### 9.2 CAKE2 Evo variants 更接近自动选 best，但不是多 DSL Track

`cake-evo` 产生 `round_*/parallel_*`，分别评测并选择较好变体。它主要是在同一 AscendC 工程
路线内进化优化方案，不是多种 DSL 作为平等 Track 的竞速。

`TilingKey` 是同一工程内选择 tiling/特化实现，也不是 TrackRegistry 选择。

---

## 十、三者放在一起比较

| 比较项 | KDA | 昇腾 CAKE2 | ShapeForge 计划完成态 |
| --- | --- | --- | --- |
| 多路线主体 | CuTe/TIRx/PTX 等具体实现 | AscendC 工程与 Evo variants | MSL + 两种独立 DSL Track |
| 路线关系 | 外部任务相同，内部各自实现 | 同一工程内多变体进化 | 一等、平等，无 `msl-bundle` 主从关系 |
| Host 调度 | 各路线自己实现 | 每个算子/变体自己的 Host Tiling | 各 DSL/候选自己拥有 |
| 后端生成 MSL | 不适用于其 NVIDIA 路线 | 不适用于其 Ascend 路线 | 不自动否决，独立与否看 frontend/调度/合同/身份 |
| 执行审计 | 具体 benchmark/judge | CAKE2 工程评测链 | 共享 Metal buffer/dispatch/commit/wait/timing 根 |
| 自动跑完选最佳 | 公开 `bench.py` 没有，一次一个 kernel | Evo 有变体选择与 best，但非多 DSL Track | 有轻量批量脚本，各 Track 独立跑后汇总 |
| 多实例结果 | 外部自行比较 | Evo 最优变体 | per-track 实例向量整包比较，只选一条 Track |
| 比较证据 | 任务工作区与判卷 | 工程与 Evo 报告 | 各自 fresh confirmation + comparability digest |
| 首轮 winner 含义 | 由外部比较决定 | 工程内 best | exploration winner，不冒充同输入或正式发布 |
| 真实硬件边界 | NVIDIA | 昇腾 NPU | Metal 真机；清微只模拟 |

### 10.1 “所有 Apple GPU Track 共享执行底座”是上游已有机制吗？

**有共享任务与平台运行时的相似之处，但固定审查版本没有显示 KDA/CAKE2 也具有
ShapeForge 这种强制的、Harness 拥有的统一执行审计底座。**需要把三个“共享”分开：

| 共享层 | KDA | 昇腾 CAKE2 | ShapeForge |
| --- | --- | --- | --- |
| 任务与裁判 | 统一任务 ABI、输入、FlashKDA、correctness、benchmark | reference、ModelNew、test cases、golden、evaluate.py | OpSpec、reference、baseline、correctness、timing、confirmation |
| 平台运行时 | CUDA、Torch、TVM、CUPTI 等 | CANN、AscendC runtime、构建安装与 PyBind | Metal、MTLBuffer、MTLCommandBuffer、bounded worker |
| Harness 执行审计底座 | 未见统一的跨路线抽象 | 固定审查版本未见等价机制 | 明确要求所有 Apple GPU Track 汇入同一执行与计时信任根 |

**KDA 是“共享 benchmark，内部各管各的”。**公开 `bench.py` 统一生成输入、运行 FlashKDA、检查
output/final_state 并计时；但 CuTe 自己管理 plan/cache 和 schedule，TIRx 自己生成 work list、
准备 tensor maps 和 CUDA Graph，PTX 自己管理 scheduler、descriptor table、handoff 和 fallback。
它们都使用 CUDA，并不等于共享 Harness 的 buffer guard、实际 dispatch parity、workspace poison
或统一 Host 计时根。

**CAKE2 是“共享 CANN 工程与评测流水线”。**它统一 op_desc/reference、Model/ModelNew、test
cases、CPU fp64 golden、构建安装、PyBind 和 evaluate.py；但每个算子或 Evo variant 仍有自己的
Host Tiling、TilingData/TilingKey、workspace、Device Kernel、内部同步和构建产物。固定审查版本
未显示它要求所有路线都经过 ShapeForge 式的统一 buffer/dispatch/timing 审计接口，很多底层保证
由 CANN 平台和具体工程提供。

ShapeForge 额外要求共享执行底座，是因为它接收的是人或 LLM 自由提交的未知 DSL 与内部路径。
如果执行和计时也完全各管各的，就可能出现：

```text
Track A：把 Host、workspace 初始化与同步都计入
Track B：只测 Device kernel
Track C：在 warmup 中提前完成输入相关 JIT
Track D：自报 dispatch 和耗时，Harness 无法独立核对
```

因此本地设计是：

```text
独立：DSL frontend、Host planner、work list、编译和内部算法
共享：buffer 生命周期、输入输出保护、guard/poison、实际执行记录、
      commit/wait、计时根、fresh rotation、correctness 和证据身份
```

共享的是**安全与测量信任根**，不是强迫所有 DSL 都长成 `msl-bundle`。这也是后续 Graph 路线需要
独立 ExecutionKind 的原因：kernel 审计 grid/threads/dispatch，graph 审计 executable/binding/
encode，二者可以共享底层安全与计时，而不必拥有相同的内部几何。

---

## 十一、为什么可信边界不会因多 Track 而降低

### 11.1 每条 Track 都有独立身份

候选提交身份与构建产物身份分开：

```text
candidate source/frontend/options
→ Build 时 probe 真实 compiler/device/OS
→ 冻结执行身份与 artifact set
→ 真实 correctness / benchmark / confirmation
```

frontend 升级即使源不变，也必须改变 build identity；不能沿用旧缓存或旧确认。

### 11.2 搜索与确认不能由 DSL 自证

每条 DSL 的 build adapter 都要能独立重建 expected。不能用 MSL `probe()` 或
`combined_candidate_source()` 给另一种 DSL 作证。

Graph 若未来加入，也必须先解决 `evaluate_task`/`CanonicalBundle` 的几何接缝：冻结 compiled
executable digest、binding table 和 encode plan，而不是伪造一个 stage/grid=1。

### 11.3 输入相关工作不能藏出计时

对 JIT/mutable cache：

- warmup 只能复用 input-independent 构建结果；
- input-dependent plan 必须失效并重算；
- fresh-rotation/mixture 中记录缓存键和命中；
- 不同地址与 `cu_seqlens` 的交错输入必须属于预登记评测计划；
- 不能用闭源 runtime 的内部行为作豁免。

### 11.4 批量脚本不是新的发布权威

它只读取各 Track 已有证据，按冻结规则汇总：

- 不重新发明 speedup；
- 不挑选幸运一轮；
- 不删除失败实例；
- 不把搜索 best 当 fresh confirmation；
- 不把 exploration summary 当正式发布；
- 不让 geomean 掩盖局部退化。

---

## 十二、首期明确不做什么

完成 MT0–MT7 后，以下仍然是后续工作，不应被误认为已经拥有：

1. **Graph/JIT Track**：需要先补齐 CanonicalBundle/tagged ExecutionPlan/artifact/expected 接缝；
2. **真正同输入的方案 B**：需要候选无关 seed 派生和新确认 epoch；
3. **跨 Track runtime portfolio**：不在不同 Track 之间按 shape 拼赢家；
4. **统一 Host Scheduler 或统一 DSL IR**：与平等路线目标相反；
5. **未经隔离的嵌入式宿主 DSL**：不能因为 worker 有 timeout 就标 available；
6. **清微真实 NPU Track**：模拟仍是 `simulated=true/publication_allowed=false`，NPU-P0 blocked；
7. **恢复 preserved Triton v1 或引入 CUDA/CANN 生产路线**：不允许；
8. **所有 DSL 语法与所有 shape 的一般正确性证明**：只覆盖已声明、已验证的能力和实例。

---

## 十三、给实习生的决策清单

### 我要加新算子

只写 `opspec.json + reference.py`，按需加 `baselines.*`。不要新增 Track。

### 我要写一份候选

选择一条已验收 Track，用其 DSL 和交付格式。人或 LLM 都可以，判决不因作者改变。

### 我要小/大 shape 使用不同实现

优先候选内部 variants + exact dispatch + 独立 fallback，不要新增 `small-track/large-track`。

### 我要多种 DSL 各跑各的，最后选最好

使用目标完成态的轻量批量工具。每条 Track 独立生成、构建、评测、确认，最后按冻结规则汇总。

### 我要从不同 Track 各拿一个 shape 的最优实现

这不是首期功能。需要额外的跨 Track runtime portfolio 设计与验收。

### 我要一种 Graph/JIT DSL

先确认构建组合根支持真实 graph 身份，不能拿一个假 grid/threads 混入 kernel pipeline。

### 我要清微 NPU 真机性能

当前不能。模拟只能证明协议/数据/身份，不能发布真实性能。

---

## 十四、最常见的误解

### “Track 就是 DSL。”

不对。DSL 是写法，Track 是交付与执行路线。当前两条 MSL Track 已证明同一种 DSL 可以有不同 Track。

### “所有 DSL 都必须生成 `msl-bundle`。”

不对。后端生成 MSL 不等于必须经过现有 MSL Track。独立与否看 frontend、调度、合同和身份。

### “路线独立，所以每条 DSL 可以自带安全和计时。”

不对。frontend/调度自由，Metal 执行审计和裁判必须统一。

### “KDA 公开工具自动跑完 CuTe/TIRx/PTX 并选最佳。”

不对。公开 `bench.py` 一次一个 kernel，外部再比较。

### “批量比较需要一个复杂中央竞速系统。”

不需要。逐条运行和读取结果的轻量脚本即可；复杂的是各 DSL 的真实构建执行路径。

### “多实例就是每个实例选不同 Track 的冠军。”

不对。首期是每条 Track 自己的实例向量整包比较，最终只选一条 Track。

### “相同 run_seed 就意味着跨 Track 使用同一 fresh 输入。”

不对。freeze digest 含候选身份，会改变 fresh seed。首轮只能诚实称为同分布探索比较。

### “完成计划后 Graph、清微真机和跨 Track 拼盘也都完成。”

不对。它们都明确不在首期完成条件内。

---

## 十五、这份完成态是否符合目标：请重点看这十条

如果以下描述就是你要的，说明计划方向一致；如果不同，应先改计划：

1. 多种 DSL 是一等、平等 Track，不经过现有 MSL Track 才能存在；
2. 至少两条真实 DSL Track 有独立 Metal 构建执行证据；
3. 每条 DSL 保留自己的 frontend、调度、work-list/生命周期和构建身份；
4. `msl-single/msl-bundle` 只是平等路线，不是统一 IR；
5. LLM 或人按各 Track 分别生成候选，Track 不要求作者必须是 LLM；
6. 所有路线共享任务、正确性、安全、计时和证据规则；
7. 轻量脚本逐条运行，各 Track 独立出结果，再选唯一获胜 Track；
8. 多实例按每条 Track 的实例向量整包比较，不跨 Track 拼盘；
9. 首轮结果是 exploration winner，不冒充同 fresh 输入或正式发布；
10. Graph、同输入复测、跨 Track portfolio、清微真机仍是后续工作。

---

## 十六、核对来源

### 本仓库

1. [多 DSL、平等 Track 与轻量批量探索实施计划](../架构/多DSL-Track实施计划.md)：本文对应的未来态范围。
2. [v2 协议参考](../架构/v2协议参考.md) 与 [AGENTS.md](../../AGENTS.md)：Track、候选自由和证据不变量。
3. `backends/metal/tracks.json`、`tcenv/track/registry.py`：当前真实 Track 登记与封闭能力集合。
4. `tcenv/taskcontract/runner.py`、`tcenv/track/bundle.py`：当前真实 GeometryPair/CanonicalBundle 组合根。
5. `tcenv/portfolio/candidate.py`、`tcenv/backends/metal/search_candidate.py`、
   `tcenv/backends/metal/portfolio_confirmation.py`：当前单候选搜索与确认接缝。
6. `tcenv/portfolio_validation/validation.py`、`taic/portfolio_confirmation.py`：实际 freeze/fresh seed 派生。
7. [Native Host Program ABI](../架构/native-host-program-ABI.md)：当前 MSL Bundle 的 native64 Host 路径。
8. [Host 调度成本对比](KDA-CAKE2-与ShapeForge的Host调度成本对比.md)：Host 生命周期与计时口径。
9. [ABI 设计深度对比](KDA-CAKE2-与ShapeForge的ABI设计深度对比.md)：任务/交付/执行/证据的分层。
10. [Shape Portfolio 研究](../架构/shape-portfolio研究.md)：候选内部 variants、exact dispatch 和 fallback。
11. [MT0-tilelang-probe](../../tests/support/golden/shared/多DSL-Track-阶段验收/MT0-tilelang-probe/acceptance.md)：
    首轮 TileLang 受信安装、源码/wheel 身份、外层 source-only 隔离 probe（6/6 产物、0 dispatch）、
    4 项真实 outer fixture 负例，以及 4 项真实 CPU 身份漂移 mutation（pinned version、package manifest、
    METADATA、RECORD，均在 lowering 前以 `IdentityDrift` 拒绝）与 r1–r4 失败历史；75 个固定节点无
    skip/xfail。该包本身不回改。现行 TileLang `/2` 与 `/3` 已 available，见文首当前事实。
12. [MT1b](../../tests/support/golden/shared/多DSL-Track-阶段验收/MT1b/acceptance.md)：TileLang-first 的 task-agnostic
    Track 合同（`tilelang-pipeline/1`，`kernel.py`/`text/x-python`/`dsl_source` + 候选自有
    kernel_pipeline 几何 + 前端/options 身份 + source-only/Metal 边界）、`(track_id, track_version)`
    多版本 registry（schema 2）、build 身份与 portfolio/search/preflight/confirmation 接缝。只做
    CPU/协议验收。当时 Track 保持 `interface_only`。后来的 `/2` `/3` 与 MT2 真实 Metal 见文首，包本身不回改。

### 外部只读源码快照

- **KDA workflow**：`NVlabs/kda`，`acdd79343c5e9d0c7cb96289f26fd3f1051ce785`，
  `README.md`、`docs/agent-flow.md`。
- **KDA kernels**：审查提交 `f539baab8e46712eb03c88562fc9d9c4b0798aa3`，
  `README.md`、`bench.py`、`kda-cake-cute/impl/{forward.py,planner.py}`、
  `kda-tirx/{kernel.py,kda_fwd_varlen.py}`、`kda-cake-ptx/kda_ptx/{api.py,scheduler.py}`。
- **CAKE2 / CANNBot**：`https://gitcode.com/cann/cannbot.git`，审查提交
  `b699c3247241dbe0884f0ea95c619d0cc85d30f4`，
  `plugins-community/collaborative-agent-kernel-evolution/` 下的
  `docs/EVALUATION_INSTRUCTION.md`、`agents/cake-evo.md`、`skills/cake-evo/`、
  `skills/op-desc-generation/` 和 `skills/ascendc-evaluation/`。

这些材料只用于架构比较，不是 ShapeForge 的生产依赖或本机性能证据。
