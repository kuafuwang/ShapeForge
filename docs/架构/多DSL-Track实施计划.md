# 多 DSL、平等 Track 与轻量批量探索实施计划

状态：**本轮按 MSL + TileLang、`kernel_pipeline` 范围的多 DSL Track 实施已完成。**
MT1a/MT5/MT1b 已按各自范围完成，MT2 已完成真实 Metal 验收与独立 RLAR 终审，MT4 已完成
correctness-only 生成/评测接入与独立 review，MT6 已按 MSL + TileLang 范围完成，MT7 文档、
只读可观测性与最终独立终审均已签署。MT0 初审的第二条新 DSL 目标已由独立范围复核取消；
MT3 Graph 未启动，不属于本轮完成条件。`publication_allowed=false`。
除已明确授权的 GPU、DeepSeek 生成与独立 reviewer 外，本计划不自动授权调用付费模型、运行 GPU/NPU、
启用 GitHub Actions 或改写历史证据。

## 本轮完成边界

本轮交付 MSL 单文件、MSL bundle 和 TileLang `/3` 三条真实路线，完成独立 frontend/构建/执行、
fresh confirmation、activation/resume、三 Track 批量探索、文档和只读观测。r15 是 exploration
winner，不是同一 fresh 输入比较或 speedup 发布。

以下项目**不是本轮未完成的阻塞项**，如需开展必须另开阶段和授权：第三条 DSL、Graph/JIT（MT3）、
候选无关的同 fresh 输入方案 B、无界生产搜索、性能发布和 P3-B small GEMM。不得为了“原计划 100%”
而重新把第三条 DSL 或 Graph 列为当前必须实现的任务。

目标共识来源：[KDA、昇腾 CAKE2 与 ShapeForge 的多 Track 交付对比](../可信harness/KDA-CAKE2-与ShapeForge的多Track交付对比.md)，先读该文第〇节。

## 1. 目标与一句话方案

### 1.1 基本目标

让多种 DSL 分别成为**一等、平等的 Track**，与现有 MSL Track 共享任务和可信验收底座，但各自保留：

- DSL 源文件形式；
- 候选 Host planner / scheduler；
- work list、handoff 和内部执行计划；
- frontend/compiler 和构建路径；
- Device 交付形式与执行生命周期。

`msl-single` 与 `msl-bundle` **不是所有 DSL 的必经下游或统一 IR**。现行第三条交付是独立的 `tilelang-pipeline`：`/1` 为 `interface_only`，`/2` 与 `/3` 为 `available`。登记见 `backends/metal/tracks.json`。

这里的“独立路线”指**候选面对的 frontend、优化/调度模型、交付合同与构建身份独立**，不经过现有
`msl-single/msl-bundle` 的 Track adapter；它不禁止 DSL 后端在 Apple GPU 上生成 MSL、AIR 或
`.metallib`，也不禁止调用 Apple Metal 编译链。MSL source、AIR 与 `.metallib` 是不同层次，不能
把 MSL 简单类比成 PTX。反过来，绕开现有 Python 类、用一个玩具 JSON/S-expression 脚本吐出
`.metal`，也不会仅凭“没调用旧类”就成为合格 DSL Track。

在此基础上增加一个**轻量批量探索与结果汇总工具**：

```text
Track A 独立生成、构建、评测、确认 → result A ┐
Track B 独立生成、构建、评测、确认 → result B ├→ 汇总比较 → 最佳有效候选及其 Track
Track C 独立生成、构建、评测、确认 → result C ┘
```

这个工具只是外层 Shell/Python 级别的批量运行器与结果读取器，不是新的中央 Host Scheduler、
统一 DSL IR、TrackRegistry 自动选路器或第二套性能判决。首轮跨 Track 比较采用**各自现有
confirmation 的同分布探索性比较**，输出 exploration winner，不冒充同 fresh 输入或正式发布；
真正同输入复测属于后续可选的新确认 epoch，仍需回现有发布链。

### 1.2 最终选择单位

选择的是：

```text
(track_id, track_version, candidate identity, build/evaluation identity)
```

不是抽象宣布某种 DSL 或某条 Track 永远最好。

基本目标是**一次批量探索后选一个最佳有效候选及其 Track**。多实例时选择单位实际是
`(winning_track, per-instance candidate identity vector)`，最终仍是唯一获胜 Track，不是跨 Track
拼盘。不同 shape 混合来自不同 Track 的获胜候选属于跨 Track runtime portfolio，不在首个实现范围内。

### 1.3 两项能力分别验收

1. **平等 Track 能力**：原计划要求至少两条非 MSL DSL，且首期只收 `kernel_pipeline`。本轮签署范围是两条 MSL Track 加上一条 TileLang。`graph_executable` / `command_encoder` 没有交付。第二条非 MSL DSL 已取消，不再作为完成条件。
2. **批量探索能力**：轻量工具逐条运行现有单 Track 流程，读取各自独立结果并确定性地汇总。已按 `msl-single`、`msl-bundle`、TileLang `/3` 做完，见文首。

不能用第二项的简单脚本掩盖第一条路线没有真实 Metal 证据，也不能用多个 DSL 目录或 `tracks.json` 名字冒充真实 Track。CPU/协议或清微模拟只能作补充证据。

---

## 2. 非目标与不可突破的边界

### 2.1 非目标

首个实现不做：

- 所有 DSL → `msl-bundle` 的统一 lowering 总线；
- 通用 PatternResolver、LoweringRuleRegistry、StagePartitioner、TileConnector；
- 按 `op_name`、shape 或算法划分的 Track；
- 所有 DSL 共用一个中央 Host planner；
- 跨 Track runtime variant portfolio；
- TrackRegistry 暗中自动切换 Track；
- 每条 DSL 自己的 correctness、baseline、计时或发布口径；
- 修改六 RPC、field number、HMAC、BlobChannel；
- 恢复 `task.py`、`track.json`、`kernel_triton.py`、逐题 LoweringProfile 或 preserved v1 生产路径；
- 重新启用当前已停用的 GitHub Actions。

### 2.2 硬件与上游边界

目标平台仍只有：

- Apple GPU / Metal 生产执行；
- 清微 NPU 离线模拟，`simulated=true`、`publication_allowed=false`，NPU-P0 保持 blocked。

CuTe、TIRx、PTX、CUDA、AscendC、CANN、EasyAsc、msopgen、ACL/PyBind、msprof 等只能作为外部
方法论参考，不能进入生产、pack 或 `agent_guidance`。不能为了凑够两种 DSL 而引入这些路线，
也不能把 preserved Triton v1 重新包装成新生产 Track。

### 2.3 候选自由与 Harness 责任

候选（人或 LLM）继续拥有：

- 算法、融合、Tiling、Host Tiling、ExecutionGeometry；
- DSL 内部 Host planner、work-list 编码、handoff 和状态机；
- 公开原生矩阵、同步、循环、persistent 与多阶段能力。

Harness 继续拥有：

- 任务合同、外部 ABI、安全与资源边界；
- 有界 worker、构建与真实执行审计；
- correctness、baseline、测量边界与证据身份；
- fresh confirmation 和唯一发布谓词。

DSL 可以不同，裁判不能不同。**候选表达力优先于共享抽象**：共享 Metal 执行底座只统一
buffer/resource 安全、实际执行观察、计时与证据，不得被扩大成统一 DSL、统一 Host Scheduler、
统一编译路线或统一 ExecutionGeometry。Track 拥有自己的 Candidate Source、Frontend/Compiler、
Host Scheduler、Internal ExecutionPlan 和 Build/Encode Strategy；现有 GeometryPair 只是
`kernel_pipeline` 的一种实现，不是所有 Track 的统一上限。未来可扩展带标签的 ExecutionPlan，
禁止为了适配旧接口伪造 stage/grid 或强迫 DSL 套用 `msl-bundle` 的 HostTilingPlan。

DSL 的候选执行模型必须在 MT0 分类为 `declarative` 或 `embedded_host_code`。前者只由受信
frontend 解析/编译；后者会被 import/执行为宿主 Python/C++ 等代码。当前 bounded worker 主要
提供进程组看门狗和环境变量清理，**不是完整 OS 沙箱**：它仍可能保留 HOME/PATH、仓库 cwd 和
同 UID 文件/网络能力。嵌入式候选没有经实证的独立进程、临时 HOME/cache、写集白名单、无网络
与无凭据可达隔离时，最高只能 `interface_only`，不得标 `available`。import/ctypes/subprocess
黑名单只能补充检查，不能替代真实隔离。

**MT0-isolation-profile 已确定的最终隔离实现决策（不等于 MT0 通过）：**不修改、配置或 fork hmz，
不另写 Seatbelt renderer，也不削弱嵌入式 DSL 的宿主代码表达自由。唯一生产路径由受信 driver 使用
hmz uv-tool 自己的解释器在 sandbox 外运行，导入源码审计过的底层 `hmz.coganchor.fence.proxy.Proxy`
与 `hmz.coganchor.darwin.seatbelt.Profile`；项目 Python 不跨环境 import hmz。不可信候选构建固定
`online=false`：driver 启动空 host allowlist 的 `Proxy(())`，并把实际 loopback port 传给
`Profile(port=...)`。`Profile(port=None)` 不是离线路径；proxy/loopback 的存在也不证明 Unix socket、
Mach service 或所有本地 IPC 已消失。受信依赖安装/准备属于另一个受信阶段，不与候选执行共用在线能力。
首个 TileLang 受信安装与 source-only probe 见
[`MT0-tilelang-probe/`](../../tests/support/golden/shared/多DSL-Track-阶段验收/MT0-tilelang-probe/acceptance.md)；该包已完成受信安装、身份核对、冻结探针计划、真实 source-only 隔离构建（6/6 产物，0 dispatch）与 4 项真实 outer fixture 负例，另有 4 项真实 CPU 身份漂移 mutation（pinned version、package manifest、METADATA、RECORD）在 lowering 前以 `IdentityDrift` 拒绝；保留 r1–r4 失败历史，75 个固定节点无 skip/xfail，完成了独立只读复核范围内的收尾。该包本身不回改。文首状态覆盖本段当时的结论：第二条 DSL 已取消，TileLang `/2` 与 `/3` 已 `available`。

旧的 `wrap._seatbelted()` internal-fence 路径已整体替换，不保留双路径。该包装会额外授予 per-user
`C/mds`；lower-level `Profile` 不经过它，因此严格 attempt/build 写集不需要 mds，也不存在 mds 合同例外。
ShapeForge 保留 attempt 环境/HOME/cache、只读输入与工具链、写集声明、rlimit、进程组 watchdog、artifact
和证据；受控 Python/Clang 及其后代只能经同一 `Profile.command()` 启动。固定解释器、uv-tool 布局、hmz
版本、相关源码 SHA、driver/shim/protocol 身份；缺 hmz、任何版本/源码漂移或 Seatbelt 不可用均 typed
`backend_capability_unavailable`，禁止无隔离 fallback。RLAR/嵌套工具上下文只做源码审计、生产适配、
静态/CPU mock 和普通 shim/watchdog 测试，不嵌套调用 hmz/Agent/subagent，不直接运行 `sandbox-exec`
或真实攻击 probe。随后由真实外层控制机留下 r1 `outer_probe_failed` 与 r2 `outer_probe_observed` 两份不可变
raw observation，并在新的 `mt0-hmz-lower-level-profile-outer-review-1` epoch 完成只读独立复核；复核不重跑
probe。r2 的 raw `checks` 精确集合实际为 18/18（不是旧文字中的 17 项），loopback/Unix socket/Mach service
残余和 `setsid` 脱离进程组缺口仍明确保留。该结果只覆盖 lower-level Profile 外层隔离证据。它当时不表示 frontend、生产 DSL 或真实 Metal 已经完成。文首状态覆盖本段：第二条 DSL 已取消，TileLang `/2` 与 `/3` 已 `available`，MT3 Graph 未启动。旧
`tests/support/golden/shared/多DSL-Track-阶段验收/MT0-isolation/` 是封闭的 internal-fence 历史包，不回改；新决策、固定节点、raw probe
结果与独立复核位于独立的 `tests/support/golden/shared/多DSL-Track-阶段验收/MT0-isolation-profile/`。

### 2.4 历史与工作树纪律

- 不删除或改写 R0–R6、G0、A1–A6、P0–P2 等历史证据。
- 现有 `msl-single/msl-bundle` 的题面、candidate/geometry/artifact identity 和真实 dispatch 是
  兼容锚点；如 registry/schema 身份必须前进，建立新 epoch 和新证据，不回填旧包。
- 每阶段记录 HEAD、tracked/untracked 工作树清单与 SHA-256。当前工作树可能有其他会话改动，
  不得 `reset --hard`、`git clean`、`stash` 或覆盖他人工作。
- P3-B small GEMM 的既有性能阻断与本计划分开，不能用新 DSL Track 宣称它已经解决。

---

## 3. 当前实现盘点与主要改造点

### 3.1 已有可复用底座

| 资产 | 当前位置 | 复用方式 |
| --- | --- | --- |
| 正式 TrackRegistry | `tcenv/track/registry.py`、`backends/metal/tracks.json` | 登记 task-agnostic 的 DSL Track、版本、能力、实现和证据 |
| Candidate/Geometry/Bundle | `tcenv/track/plane.py`、`delivery.py`、`bundle.py` | 复用通用打包与身份思想，不等于复用 `msl-bundle` 内容格式 |
| Track 选择与冻结 | `tcenv/taskcontract/runner.py` | 保持正式 `track_id/version` 和跨 worker 冻结审计 |
| 有界构建与执行 | `tcenv/v2_worker.py` | 新 DSL 首次编译、加载和执行也必须在 worker 内 |
| 唯一构建/报告/确认链 | `tcenv/remote/artifact_builder.py`、schema 4、`taic/policy.py`、confirmation 模块 | 新 Track 汇入同一身份与发布规则，不旁路自报 |
| 单 Track LLM 搜索 | `taic/orchestrator.py`、`kernel_agent.py`、`task_binding.py`、`track_prompt.py`（按登记的 renderer 解析 prompt/提交风格，未知 renderer fail closed） | 每条 Track 独立运行，批量工具只调用，不合并内部调度 |
| 单 Track CLI | `taic/cli.py` | 复用 `--track`、run/resume 记录校验；另加轻量批量入口 |

### 3.2 接入前盘点

下表是 TileLang 接入前的硬编码盘点，不是未做清单。现行 `--track` 可选项来自 `taic/cli.py` 的 `_run_track_choices()`，也就是 `load_track_registry().available()` 的 track id：`msl-single`、`msl-bundle`、`tilelang-pipeline`。`/1` 仍是 `interface_only`，新 run 选这个 id 时拿到最高 available 版本 `/3`。MSL 子包仍按有无 host 几何在 `msl-single` 与 `msl-bundle` 之间选择；调用方传入已登记 DSL Track 时，`tcenv/portfolio/candidate.py` 的 `project_provider` 保留该 DSL 身份，并拒绝把 DSL 源收成 MSL。

| 位置 | 接入前的问题 | 计划方向 |
| --- | --- | --- |
| `tcenv/track/registry.py` 与 schema | delivery、roles、capabilities 是封闭集合 | MT1 先证明现有角色是否足够；不足时版本化扩展，禁止伪装 MSL |
| `tcenv/track/plane.py` | delivery 校验和 single-source 身份规则与现有两条 Track 绑定 | 保持旧规则，新增 DSL delivery 的明确身份和角色校验 |
| `tcenv/taskcontract/runner.py` | builder/evaluator/geometry resolver 由调用方注入 | backend 侧 adapter 工厂按登记 identity 注入，不能按 op 分支；若改本文件，pack digest 需按实际变化重验 |
| `tcenv/portfolio/candidate.py` | `EvaluationBinding` 只认两个 Track；`project_provider()` 由 geometry 二选一，并检查 MSL include/launch/HostTilingPlan | MT1 冻结新 DSL 的投影与 Track 绑定；不得让 DSL 被二元投影吞掉，P2 按新身份重验 |
| `tcenv/backends/metal/search_candidate.py` | submission/preflight/evaluate/confirm 全部按两个 MSL Track 特判 | 按 Track 的 candidate/validator/build adapter 委托，保留单候选 portfolio 语义，不串用证据 |
| `tcenv/backends/metal/portfolio_confirmation.py` | expected 用 `MetalSourceBuilder.probe` 和 `combined_candidate_source` 重建 | 每种 DSL 独立重建 expected；不能使用 MSL 自证 DSL 构建，confirmation 保持 fresh |
| `tcenv/remote/artifact_bundle.py`、`tcenv/build_projection.py`、`tcenv/track/delivery.py` | `NATIVE_EXECUTABLE_KINDS` 只有 `compute-pipeline`，single-source 构造固定 `msl-single`，descriptor 有闭集/上限 | 版本化新增 graph/JIT executable 类型与 descriptor 审计；不把图句柄序列化进 bundle，不虚构 grid/threads |
| `tcenv/v2_worker.py` | 当前请求与执行围绕现有 Metal v2 路径 | 保持冻结合同，在 worker 内接入 DSL build/execute，禁止控制机加载原生候选 |
| `tcenv/backends/metal/track_adapter.py` | 现有两条 Metal Track 的构建/执行适配 | 抽清共享 Metal 边界与特定 Track 逻辑，新增独立 DSL adapter |
| `taic/cli.py` | `--track` choices 固定为两个名字 | 改为从正式 registry 获取可选值，resume 仍严格拒绝 Track 漂移 |
| `taic/orchestrator.py` | Track 解析、workspace 检查和 evaluator parity 含现有 Track 假设 | 改为按登记的 delivery/capability 判断，保留默认 `msl-single` 兼容性 |
| `taic/task_binding.py`、`prompts.py` | 题面和提交说明按当前 delivery 特判 | 按 Track 渲染 DSL 交付事实，不提供逐题算法 |
| `taic/kernel_agent.py` | envelope、quick/check 命令和文件说明含 `msl-bundle` 特判 | 改为 Track 提供的提交/工具合同，保持原有两条 Track 输出不变 |

**MT1 的回归目标：在输入不变时，现有两条 Track 的 statement、候选解析、执行结果和兼容行为
不因去特判而改变。**但 `task_pack_digest` 是文件内容哈希，不是仅文件列表哈希；修改闭包中的
`taskcontract`/portfolio/schema 文件就可能使 digest 前进。每次实现前必须记录实际闭包与 digest
差异，不能预承诺逐位不变。registry/schema/build/evaluation identity 若前进，开启新 epoch 并
重录必要证据，不回填旧包。

---

## 4. 分阶段实施

阶段编号使用 **MT0–MT7**，避免与既有 R/G/A/P 阶段混淆。每阶段均在
`tests/support/golden/shared/多DSL-Track-阶段验收/MT<n>/` 新建证据包，包含：

- `verify.py`；
- `manifest.sha256`；
- 固定正例、负例和 node id；
- 当前阶段验收文档；
- 必要的结构化证据与独立 reviewer 结论。

MT 编号是工作包身份，不是机械的文件排列顺序。MT1 分为 **MT1a（DSL 无关）**与
**MT1b（依赖 MT0 的具体 DSL 合同）**，实际依赖顺序为
**MT1a → MT5**、**MT0 → MT1b → MT2 → MT3 → MT4 → MT6 → MT7**；MT0 与 MT1a 可并行。
MT5 先用现有两条 MSL Track 打通，后续新 DSL 直接复用其输入/输出协议。当时的依赖规则是：MT0 未通过不妨碍 MT1a/MT5，但 MT1b 不得冻结猜测的 DSL schema，MT2/MT3 不得启动。后来在单独授权下完成了 TileLang 的 MT2、MT4、MT6、MT7。MT3 Graph 仍未启动。
阶段未通过不得进入依赖它的下一阶段。CPU/协议通过不能写成真实 Metal 通过，source-only 或模拟
Track 不得标为 `available`。独立 RLAR 使用不同的 actor/reviewer，禁止嵌套
`hmz_flow`/hmz/Agent/subagent；RLAR 内不执行 `git add/commit/reset/clean/stash`，主会话不自签。

### MT0：DSL 选型、来源与独立路线可行性门

**目标：**在写适配器之前，确定至少两种真正可在本地目标交付的非 MSL DSL。**MT1b 开始前必须
填实具体 DSL 名称、版本和构建/执行路线，不能只写假想的 DSL X/Y。**找不到就停止新 DSL 接入，
不伪造多 DSL。首期只接受 `kernel_pipeline` 类路线；graph 路线可记录为后续候选，但不能用来
满足首期两条 DSL 的完成条件。

**只做只读审计和本机工具链盘点，不生成 kernel、不跑 GPU、不调用模型。**如本机已有 frontend，
可做不产生/不执行设备 kernel 的离线版本与帮助信息探测；真实编译和 Metal 执行留到后续授权阶段。

每种候选 DSL 记录：

```text
dsl_id / version
repo / commit / license
source form and media type
candidate execution_model = declarative | embedded_host_code
candidate roles
Host scheduler ownership
frontend/compiler and version
build artifacts
Device execution path（区分 MSL source / AIR / metallib / runtime）
worker isolation requirements
cache/lifecycle requirements
whether it reuses the current msl-single/msl-bundle Track contract or adapter
independent frontend/optimizer/scheduler evidence
why it is a genuinely different delivery mechanism
```

**入选硬条件：**

1. LLM 或人可以编写该 DSL 候选；
2. 有明确、可冻结的 frontend/compiler 和构建产物；
3. 能在有界 worker 内完成构建与执行；
4. 能接入 Apple GPU 的真实本地执行路径，或仅被明确归为清微模拟路线；
5. 能记录实际 dispatch/执行与 Host 工作，不接受只有自报耗时；
6. 能满足 correctness、资源、安全和 fresh evidence；
7. 至少一条新 DSL Track **不经过现有 `msl-single/msl-bundle` Track 合同或 adapter**，证明平等路线不是换皮；后端为 Apple GPU 生成 MSL/AIR/metallib 不自动否决，但必须存在独立的候选级 frontend、优化/调度语义与身份，不能只是薄包装；
8. 两条 DSL 不能只是同一 frontend 改个名字，也不能用一个玩具文本格式加简单字符串拼接来凑数；
9. MT0 必须列出具体 DSL 名称、版本、实际工具和可核来源，不能只给“Tile DSL/MLIR/C++ 模板”等类别名；
10. 必须声明 `execution_model`。`embedded_host_code` 只有在独立实证了文件写集、网络、凭据与
    子进程隔离后才可进入真实候选构建；没有该证据只能 `interface_only`，不能用 bounded worker
    或 import/ctypes 黑名单冒充安全；
11. 首期必须是 `kernel_pipeline`：能提供真实 kernel/stage/grid/threads 与 CanonicalBundle 兼容的
    几何身份。这是控制首期工程范围，不是把所有 Track 永久限制为现有 GeometryPair；若 DSL 的
    优势需要更丰富的执行表示，应进入后续 tagged ExecutionPlan 扩展，而不是削足适履。graph
    路线只有补齐 MT1b 的 tagged ExecutionPlan、CanonicalBundle 与 artifact 接缝后才能另立
    后续工作包，不能交假的一阶段/grid=1 几何。

**待探测的三类方向（只是调查清单，不是已证实的 DSL 名单）：**

1. 有真实 AST/tile/layout 推导的 Python 或其他 frontend；MLX `mx.fast.metal_kernel`、模板/宏
   机制只有在具备独立编译语义且不是字符串拼接时才可进一步评估；Triton/TileLang/TVM 等需核对
   实际后端、许可证和目标边界，不能凭名字入选。
2. Apple 原生图/张量编译路线，如 MPSGraph、MLX compile/custom graph。必须现场核实 API 是否支持
   候选自定义生成、bounded worker、所需的外部 CommandBuffer 编码或等价的可审计执行边界；不能
   把第三方 runtime 黑盒直接调 GPU 当成合格交付。
3. 受限 C++ 模板微内核/布局代数路线，须有真实模板/布局推导、公开原生能力和可冻结构建身份；
   不能用 100 行文本拼接器制造一个玩具 DSL。

**输出：**`dsl-selection.json`、MT0 自己的 `upstream-map.json`（新增来源台账，不改写 G0 封闭包）、能力/风险矩阵、两种 DSL 的垂直切片规格。嵌入式宿主代码现有封闭的 `MT0-isolation` 历史包与独立的 `MT0-isolation-profile` 工作包；后者的 lower-level Profile 适配、静态/普通 CPU 验证及保留 r1 失败的 r2 真实外层 observed 证据已完成独立复核，但这仍不代表 DSL 选型、frontend、真实 Metal 或 MT0 整体完成。

**退出/阻断（起草时的规则；现行边界见文首）：**若没有两种合格 DSL，当时把 MT0 记为 blocked，MT2/MT3 不得启动，也不得用 MSL 改名
凑数。MT1 的现有 Track 去特判与 MT5 的现有两 Track 批量工具可以作为独立有用工作继续，但不因此
解除整体多 DSL 阻断。只有一条真实 Metal DSL 加一条清微模拟 DSL，同样不能算整体达标；后者只能
证明模拟/协议，保持 `simulated=true/publication_allowed=false`。MT0 通过也不等于 DSL Track 已 available。后来的范围复核取消了第二条 DSL，本轮按一条 TileLang 签署，不再用这段阻断把本轮记成未完成。

**MT0 当前进展（TileLang 首轮，用户已决定只选 TileLang）：** 首轮只推进 TileLang
`0.1.15`（上游审计 commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`，TVM 子模块
`4211874e9e4b8770fd20ad0d98ab80afd1df5028`）。**不安装、不构建、不探测 Halide**。写这段时第二条
非 MSL DSL 还留着以后评估；文首的范围复核已经把它取消。受信准备已把固定的 macOS arm64 wheel 与 19 个依赖 wheel 安装到
受控 staging（`/tmp/shapeforge-frontends/tilelang-0.1.15`，不污染项目 `.venv`、hmz uv-tool、
系统 Python 或全局配置），并记录 license、逐文件 PyPI sha256、RECORD 21,199/21,199、安装树与
`tilelang/` 包摘要；tag 与 TVM 子模块的 Python 源和 wheel 内副本逐字节一致，native 库未重建，
不冒充完整二进制可复现。

真实 source-only frontend probe **已在外层控制机完成**（第 5 轮，`probe-result-r5.json`）：最低
Seatbelt 能力探测 `returncode=0`，lower-level `Proxy(()) + Profile` 实际应用，`online=false`、空
host allowlist、无 mds 写根；六个构建产物（`kernel.metal` 732 B、device IR、kernel params、host
C source、`frontend_identity.json`、`build_identity.json`）按原字节归档在包内 `evidence/`，
`artifact_set_sha256=bc6a7300…`。`full_build=true`、`device_dispatch_calls=0`：未编译 Metal
library、未创建 pipeline/buffer、未 dispatch；`host_module.txt` 是未编译的生成源码（其 `__OBJC__`
分支文本上含 MPS command-buffer 调用，这不是执行事实）。前四轮失败没有被掩盖：r1–r3 是候选
无限定 `@prim_func` 触发 parser 拒绝并叠加诊断渲染崩溃，r4 是 Metal `LayoutInference` 需要 Target
context；修正仅为 `@prim_func` → `@T.prim_func` 与把同一 `tilelang_lower(...)` 放进 `with target:`，
算法、grid/threads、options 与隔离参数不变，原始结果/输入/日志与 `probe-history.json` 逐字节保留。
4 项真实负例（文件读/越界写/非 proxy loopback/未声明 artifact）全部 matched，另 5 项由冻结的
adapter 合同与固定节点覆盖，两类分开记录。

因此 MT0 包按当时的两条新 DSL 门槛仍是 blocked 记录，包本身不回改。文首已取消第二条 DSL；TileLang `/2` 与 `/3` 已 `available`。
不得用 Halide、玩具格式、MSL 改名或清微模拟解除当时的记录。旧 `MT0`、`MT0-isolation`、
`MT0-isolation-profile`、`MT1a`、`MT5` 包均未回改。

用户在 MT0 仍按当时门槛 blocked 的前提下显式批准先做 **TileLang-first 的 MT1b**（只冻结一条 DSL 的合同/身份/
接缝，当时登记 `interface_only`）。该批准不改写 MT0 包。后续 `/2` `/3`、MT2、MT4、MT6 与 MT7 见文首。MT1b 的完成范围见 §9 末尾。

### MT1：通用 Track 合同、去特判与现有路径兼容（MT1a / MT1b）

**目标：**让 TrackRegistry 和生成/执行链接收新的交付机制，同时保持现有两条 Track 行为稳定。

- **MT1a（可与 MT0 并行）**：仅做 DSL 无关的 registry/CLI 动态读取、adapter 延迟注入骨架、
  旧 prompt/TASK 黄金快照、MT5 汇总协议与现有路径回归，不冻结任何 DSL 特有 role/kind/schema。
- **MT1b（硬依赖 MT0）**：根据已选定 DSL 冻结 delivery/role/capability、ExecutionKind、
  native descriptor、多版本 registry 与对应 evidence；MT0 没有真实 DSL 源形态和构建产物时不得开始。

**主要工作（MT1a + MT1b）：**

1. 冻结 DSL CandidateEntry、media type、delivery、frontend/build/execution identity。
2. 优先复用 `device_kernel`、`host_tiling`、`schedule` 等通用角色；无法表达时才升级版本化
   schema。不得把 DSL 内容伪装成 `kernel.metal`。
3. 明确 renderer/validator/build adapter 的 implementation identity；adapter 工厂和 backend/
   DSL lookup 放在 `tcenv/backends/` 等已隔离侧，由调用方/worker 延迟注入，不向 hardware-neutral
   `taskcontract/opspec` 引入 backend 顶层依赖。执行 adapter 若无法由现有身份派生，需在评审后版本化
   补齐，不能增加按 op_name 的分支；通用合同确需修改时如实记录闭包与 digest 前进，不能一律禁改。
4. 去除第 3.2 节所列生成与 CLI 特判，改为读取登记的 delivery/capability/adapter。
5. 保持 `TrackRegistry` 只负责指定 Track，不负责批量竞速或自动替换。
6. 新 DSL Track 初始只登记 `interface_only`，直到真实证据完成后才能改为 `available`。
   登记为 `available` 还不够：DSL Track 必须同时声明 `metal_dispatch`，`TrackSpec.selectable` /
   `require_available` 才允许新 run 选中它，source-only 构建不构成设备执行证据。
   `available` 只覆盖声明且有真实证据的 capabilities；未验收语法/资源/ExecutionKind 必须由
   validator 返回 typed `unsupported_contract_capability`，不能表现为编译错误或 crash。扩能力
   必须升 `track_version`，不能靠一个全局 available 布尔暗示全域支持。
   Registry 在 MT1b 升级为允许 `(track_id, track_version)` 多版本共存：新 run 选择登记的可用版本，
   旧版本保留给 frozen reconstruction/resume/replay；默认 get 的版本解析规则、冻结 worker 选择、
   旧 run 不漂移均有固定测试。若不采用多版本共存，只能显式宣布旧版退役且旧 run 只读不可 resume，
   本计划默认选择前者，不暗中丢失历史恢复能力。
7. pack 闭包继续由静态 import/注册关系派生，不手写逐题 `pack_files`。
8. 明确拆分 **Track Frontend** 与 **Target Execution Substrate**：前者负责 DSL 源校验、frontend/
   lowering、候选 Host 准备和内部计划；后者统一负责 bounded worker、Metal buffer 生命周期、只读
   输入保护、output/workspace guard/poison、fresh-rotation、baseline/candidate 交错计时、
   CommandBuffer commit/wait 和硬件状态。所有 Apple GPU Track 必须汇入这套底座，不能各自复制
   安全逻辑或用闭源 runtime 黑盒绕过它。
9. 执行编码与审计按登记的 **ExecutionKind** 分派，而不是强迫所有路线交 grid/threads：
   - `kernel_pipeline`：沿用 per-stage kernel/grid/threads/threadgroup_bytes、反射绑定和实际
     dispatch parity；
   - `graph_executable/command_encoder`：底座持有 MTLBuffer/MTLCommandBuffer 生命周期，审计 compiled
     executable digest、真实输入/输出/workspace 绑定与权限、实际 encode/commit 次数和执行状态，
     不虚构 `(grid, threads)`。graph/JIT 的 `NativeExecutable.kind` 与 descriptor 需要版本化扩展，
     live handle 仍不进入序列化 bundle；
   - 新 kind 必须有对应的独立 expected 重建、负例与真实执行证据，未知 kind fail closed。
   - `graph_executable`（后续扩展）必须在 worker 的 **Build** 阶段显式完成 AOT/lowering 或触发
     首次编译，取得已编译 executable 并计算其 digest；不能在 execute/timing 阶段才进行首次
     tracing/JIT。编译期 shape 变化通常是新实例、新 build；同一 executable 只服务声明的实例与
     输入域，域外由 dispatch/validator 拒绝。L2 capacity 内 resolved total/count 变化继续按已验收
     runtime-resolved 规则处理，不能一概要求重编。需要重编时必须回到 Build 并更新身份，禁止
     在计时外静默重编后沿用旧身份。编译完成不豁免 input-dependent Host 准备的计时责任。
10. 共享执行底座不等于复用 `msl-bundle` 的候选合同、HostTilingPlan 或 DSL 前端；路线平等体现在
    frontend/调度自由，安全与测量统一体现在最底层执行与证据根。
11. `tcenv/portfolio/candidate.py`、Metal `search_candidate.py` 与 `portfolio_confirmation.py`
    必须一起设计：新 DSL 投影显式绑定正式 Track，由 Track validator/build adapter 处理，不被
    geometry 有无的二元 MSL 分支吞掉；搜索、preflight、confirmation expected 同样委托对应
    frontend。复用 PortfolioFreeze/subject 文档语义不等于 P2 代码和证据零修改，必须新增 DSL
    负例、独立 expected 与必要的新 epoch 重录。
    当前 `evaluate_task`/`CanonicalBundle` 强制 `GeometryPair + ResolvedGeometry` 和冻结几何 entry。
    首期 DSL 必须走真实 `kernel_pipeline` 兼容路径。若以后启用 graph，必须先引入版本化 tagged
    ExecutionPlan：kernel 沿用 GeometryPair/selection_parity；graph 冻结 compiled executable digest、
    输入/输出/workspace binding table 与 encode plan，并替换对应 artifact entry、expected 和
    dispatch 校验，不合成 stage/grid。未补这条构建缝前，graph 不得进入 MT2/MT3。
    每个 build adapter 必须在 Build 时 probe 真实 `{frontend/compiler 名称与版本, 关键 options,
    OS/device 事实}`，进入 compile profile、expected 重建和缓存键；frontend 升级即使源不变也要
    改变 build identity，MT3 增加编译器版本漂移负例。
12. 修改 prompt 指纹覆盖的 `signature_stub`、`render_contract`、`_check_command` 等函数时，
    `current_prompt_sha1` 会按设计因源码字节变化而前进，即使旧输出字符串不变。必须记录新
    指纹和授权影响，同时用黄金快照固定旧 `msl-single/msl-bundle` 的 TASK.md 与
    `TaskBinding.presentation_sha256()`；不能把输出未变误写成 prompt 指纹也不变。

**固定负例：**未知 DSL、未知 media type、角色缺失、伪造 MSL 后缀、op/shape 字段混入 registry、
未登记 frontend、执行 adapter 漂移、worker 读取默认 registry、旧 Track resume 改 Track。

**验收：**现有两条 Track 的 CPU/真实 Metal 锚点通过；新旧 schema 不双读；registry digest 变化按
新身份处理，旧证据文件字节不变。

### MT2：第一条 DSL Track 的真实垂直切片

**目标：**用固定人工候选打通第一条非 MSL DSL Track，不调用 LLM。

**最小正例：**

1. 一个简单 elementwise/normalization 候选，证明源解析、构建、执行和 correctness；
2. 一个能证明该 DSL **独立调度所有权**的小例子，可以是 Host work-list、Device 端调度或与其
   ExecutionKind 相符的阶段/调度表示；输入驱动部分全部计时。graph 若后续获批，可用图划分证明，
   不能强迫每条 DSL 都模仿 `msl-bundle` 的 Host planner 文件；
3. 一个与现有 MSL 候选语义相同但内部 Host/work-list 不同的对照例子。

**必须证明：**

- DSL 源和 frontend 身份进入候选/构建身份；
- 首次编译/加载/执行只在 bounded worker；
- 编译失败只有 partial build，超时/崩溃无 full build；graph/JIT 的首次 tracing/lowering/编译
  必须发生在该 Build 阶段，已编译 executable digest 进入身份，timed 首次编译负例必须被拒绝；
- 生成物和冻结几何进入 artifact set 后再计算 artifact identity；
- 实际执行有独立记录，不能只采用 DSL 自报 launch/time；
- correctness 使用 `tolerances + scaled_error`，包括 `read_write` 终值；
- 输入驱动的 Host 工作进入 candidate E2E；
- 未声明资源、路径逃逸、任意宿主 include、非法 import、guard/span 违规被拒绝；
- 不 import 或调用 `msl-single/msl-bundle` 的 Track adapter、候选合同或 HostTilingPlan 来冒充独立路线；必须复用的是 MT1 拆出的 Metal 执行审计底座，而不是另写一套无审计 runtime；
- 对带 JIT/mutable cache 的 DSL runtime，先证明首次图编译已在 Build 完成；shape/分段变化需要
  重编时必须回到 Build 并更新身份。随后使用已登记的 fresh-rotation、stateful timing 或 L2 ordered
  mixture 协议，交错不同输入地址和不同 `cu_seqlens` 值，证明 timed 调用仍读取本次输入；
- warmup 与 timed 使用相同输入时，只能复用 input-independent 构建结果；input-dependent plan 必须失效并重算。fresh-rotation/mixture 使用不同值或地址时，缓存键、命中和每次 input-dependent
  prepare 都要可审计，不能用“闭源 runtime 内部行为”作豁免；
- 交错输入必须属于预先登记的评测实例与计时计划，不能为了抓缓存临时改变 shape、baseline 或测量口径。

**通过后才可将该 Track 标为 available，且只覆盖已验收能力，不宣称所有 DSL 程序都可运行。**

**MT2 当前进展（TileLang /2，真实 Metal 与独立终审均已完成）：** 用户已单独授权本轮真实 Metal
GPU 验收。新增 `tilelang-pipeline/2` 并保留 `/1` 的 `interface_only` frozen replay；`/2` 仅对固定人工
candidate-owned `kernel_pipeline` 开放。单阶段、候选自有两阶段 work-list 和同语义 MSL 对照均通过真实
build/ABI/read_write correctness/dispatch/fresh rotation；shipped registry 也直接通过。r4 与 r6 各 **7/7**
GPU 节点通过，r4/r6 candidate/bundle/artifact identity 一致；最终聚焦 CPU 回归 **105 passed**。frontend/
沙箱失败为 partial，越界为 unsafe，generated-artifact/native 篡改被独立审计拒绝，r5 noisy 历史保留。
`device_ir` 的 attempt 绝对 SourceName 已规范化为逻辑 `kernel.py`，两次独立构建 artifact 可复现。
验收包 [`MT2/`](../../tests/support/golden/shared/多DSL-Track-阶段验收/MT2/acceptance.md) 的 verifier/manifest 已通过，独立 DeepSeek Flash
reviewer 已返回 `done=true`，日志摘要见 `MT2/review.json`。`/2` 的 `available` 只覆盖当时验收的 `kernel_pipeline`。LLM 生成与固定切片 fresh 已在 MT4、MT6 完成。Graph 未做，speedup publication 仍关闭。

### MT3：第二条 DSL Track 与平等性证明

**状态：未启动，且不再是本轮工作。** 独立范围复核已取消第二条非 MSL DSL。以后若另开 MT3，指的是 Graph/JIT，不是再接一条 DSL。下面的清单是原计划，不要当成当前任务。

**目标：**用第二条非 MSL DSL 证明 MT1/MT2 不是一次性专用硬编码。

**主要工作：**

1. 按 MT2 相同合同和测试层级接入第二条 DSL；
2. 复用共享 Harness 与 Metal 执行审计底座，不复用第一条 DSL 的 Host planner、内部 work-list 或私有执行格式；
3. 在同一语义任务上形成“MSL、DSL X、DSL Y”三个独立候选对照；
4. 结构测试证明三条路线可以平等注册，没有主从 lowering 关系；
5. 对相同候选字节，在相同合同/Track 下得到作者中立的结构判决。

**固定负例：**DSL X 调 DSL Y evaluator、两条 Track 共用缓存但身份未隔离、第一条 Track 证据给
第二条背书、只改文件扩展名冒充第二 DSL、缺少 Host step 却自报完整执行。

**通过条件：**两条新 DSL Track 都有真实 Metal 机制证据；只通过 CPU 模拟的路线只能保留
`interface_only` 或模拟身份。

### MT4：LLM/人工候选生成与工具合同接入

**目标：**让 LLM 能按不同 Track 生成对应 DSL 候选，同时允许人工提交同格式候选。

**主要工作：**

1. TrackRenderer 渲染统一 OpSpec/reference/ABI，以及该 Track 的 DSL 交付说明；不提供逐题 tiling。
2. `taic/task_binding.py`、`prompts.py`、`kernel_agent.py` 使用 Track 提供的提交说明和工具合同，
   不再只认识 `msl-bundle` envelope。
3. DSL 源、Host scheduler、schedule 和辅助文件在抽取、quick check、evaluate、replay 全程保留。
4. 相同任务可由不同 Track 分别启动独立 `taic run`；默认 committee/模型预算/agent turns/seed 一致，
   差异必须明确记录，不能把不同生成条件的结果直接解释为 DSL 本身优劣。编译快慢不需要伪装成相同，
   但每条 Track 必须有有限、登记的 per-track build timeout 和总搜索 wall/cost 上限，不能无限构建；
   搜索报告同时记录 best 指标、agent turns、构建耗时和 total_search_wall_sec，区分算法效果与搜索成本。
5. resume/replay 继续使用记录的 Track 和完整身份，拒绝 CLI 改 Track。

**测试策略：**先用确定性 fake agent/固定提交字节完成无付费测试。真实 LLM 垂直切片需要后续
单独授权，未授权时只能标“协议/fixture 通过”，不能写成真实模型生成已验收。

**MT4 当前进展（TileLang /2，correctness-only 生成/评测与独立 review 已完成）：** prompt、ABI→PrimFunc
签名、非执行 packer、`.taic` envelope、显式 DSL projection、production evaluate、Orchestrator、snapshot
restore 与 resume dry-run 已接入，fresh confirmation 仍属 MT6。真实 DeepSeek Flash 一次 ReLU 初始生成
（seed 4、max_rounds=0）花费 **$0.295137132**，6 个 case（含 IEEE）通过；`best_speedup=1.0` 只是占位，
没有 fresh performance/publication。生产 `evaluate_candidate` 与 fake-agent snapshot restore 均通过；
真实 replay 记录 `relu_kernel_kernel grid=(64,1,1) threads=(512,1,1)`。固定 CPU 5、fake-agent GPU 2、
MSL 聚焦回归 114、完整 CPU 安全套件 4963/26/13 均按原边界保留，26 个失败仅为已知 P2 epoch。
`test_tolerance_verdict` 已按 frozen 组合并行化（102 节点/132 秒，组合与断言不变）。
`track_adapter._evaluate_in_process` 的 MSL 测试接缝回归已最小修复，actor/reviewer 分离的独立 review
返回 **done=true**（`tests/support/golden/shared/多DSL-Track-阶段验收/MT4/review.json`）。签署范围仍仅 correctness-only
生成/评测/replay，不代表 MT0/MT3、第二条 DSL、Graph、fresh confirmation 或 speedup publication 完成。

### MT5：轻量多 Track 批量探索与结果汇总（提前到 MT1a 后）

**目标：**先用现有 `msl-single/msl-bundle` 实现一个外层工具，逐条调用已有单 Track 流程并读取
结果；不重写 TrackRegistry、Orchestrator 的内部搜索或性能确认。新 DSL 在 MT2–MT4 完成后只需
加入参与列表并复用同一 summary schema，不等到全部 DSL 完成才发现协议缺字段。

**建议接口：**新增 `taic/track_explore.py` 与 `taic track-explore` 子命令：`--config` 读版本化配置，
`--summarize-only` 只读已有 run 证据、不启动 `taic run`/模型/设备评测。名称只是实施时的
接口建议，MT5 评审可调整，但不能变成新的中央调度框架。

**输入文档：**版本化 `track_explore.json`，至少包含：

```text
schema_id / task / target / participating_tracks
benchmarks: 非空、去重的 [{dtype, size, usage}]；len==1 是单实例，len>1 是多实例探索
run_seed + master_seed + 输入生成器/seed 派生算法版本（相同；不要求跨候选字面 confirmation_seed 相同）
comparability_digest（由下方共同可比元组计算）
shared correctness/baseline/evaluation/confirmation config
per-track generation config、独立 run_dir 与 build_timeout_sec（每项都是 participating_tracks[] 里的字段，
可经 TAIC_TRACK_BUILD_TIMEOUT_SEC 传给子 run；必须有限，可按 frontend 能力登记）
generation_policy（schema 1 只接受 serial，见执行规则 2）
max_agent_turns（每个 track×instance 默认一致）
budget_scope=per_track_instance 或 per_track（必填、无默认；两种不得混算）
max_total_search_wall_sec / budget_usd（按 budget_scope 有限，summary 另记实际总和）
output path / selection policy version
```

**执行规则：**

1. 每条 Track 使用独立 run_dir、cache、build、evaluation 和 confirmation；
2. 默认串行，避免并发 GPU 测量扰动与不受控费用；quiet confirmation 串行。现行 schema 1 的
   `generation_policy` 只接受 `"serial"`：单 Track 接口还没有把生成与 quiet confirmation 拆开，
   所以并行生成会被 typed 拒绝，不会被默认开启；
3. 首轮采用 **方案 A：各自现有 confirmation 的同分布探索比较**。现有 fresh seed 的 HMAC 消息
   含 `portfolio_freeze_digest`，候选/Track 不同时，即使 run/master seed 相同，freeze、
   confirmation_seed 和实际 fresh 输入也会不同。因此必须相同的是 run/master seed、派生算法版本、
   task/instance、输入生成器及 `comparability_digest`；**不得要求或宣称跨 Track 字面
   confirmation_seed 相同，也不得声称同一 fresh 输入。**search/fresh 仍分离，master/unseen 不进
   候选视图。summary 只能给 exploration winner，正式发表回现有发布链。
   后续若需真正同输入，可另立 **方案 B**：候选无关的 track-explore fresh seed 派生、独立确认
   epoch 与同输入复测；它不是现有唯一发布谓词，不能伪装成“只读现有结果”，winner 仍回正式链。
4. `comparability_digest` 由共同元组 canonical digest 计算：`opspec_digest`、实例
   `(dtype,size,usage)`、reference implementation digest、baseline implementation/environment
   digest、target/device、protocol_version、timing discipline、输入生成器与 seed 派生算法版本。
   共同摘要不一致只展示、不参与选最佳；每条 Track 的完整 build/evaluation identity 仍分别保留，
   不能因为 comparability 相同就省略或假定完整身份相同。
5. 用于性能排名的 `benchmarks[]` 必须有 `usage="benchmark"`；golden/correctness-only 可单独
   探索和报告，但不能制造性能 winner；没有 baseline 的任务只能 correctness-only；
6. 汇总器只接受现有唯一发布谓词认可的有效候选；自报 best、timing uncertain、失败 batch、
   fallback 混入或身份不符均淘汰；
7. 汇总器不重新计算一套另一口径 speedup，不直接修改 catalog 或历史 best。

**默认选择规则（MT5 冻结，可经独立评审调整）：**

- 保持唯一口径：`latency_ratio = candidate/baseline`、`speedup = baseline/candidate`。现有
  `latency_ratio <= 1.10` 是最大回归容忍门，**不是** `<= 1/1.10`，也不是要求至少 1.10x 加速。
- **单实例（len==1）**：沿用该实例所属运行类型的现有有效性与唯一发布谓词；correctness、identity、
  fresh confirmation、quiet timing 任一失败先淘汰，然后按确认的 `latency_ratio` 升序。
  若要额外要求 `speedup > 1.0`（而非允许 10% 回归），必须在 MT5 实施评审中单独批准为更严格的
  探索准入门，不能声称当前 `verify_subject_document` 已经包含它，也不能改写现有发布门。
- **多实例（len>1）**：先明确是“每条 Track 一个覆盖全部实例的候选”，还是“每条 Track 每个实例
  各选一个最佳候选”；默认采用后者并命名为 **per-track per-instance best bundle**：每条 Track
  先用自己在各实例上的最佳确认候选组成一个参赛向量，再与其他 Track 的完整向量整包比较，最终
  仍只选一条获胜 Track。它不是每个实例从不同 Track 拼冠军，也不是跨 Track runtime portfolio，
  不改变 §1.2 的非目标。每条路线必须覆盖全部声明实例，缺实例、incorrect、identity/confirmation 失败、
  `timing_uncertain` 或任何实例 effective/specialized `latency_ratio > 1.10` 都淘汰整条路线。
- 多实例过门后的默认排序使用**预注册、等权的 `geomean(per-instance latency_ratio)` 升序**，使不同
  时间尺度的实例有同等话语权；等权 geomean 只在全部实例过硬门后排序，不能用来掩盖局部失败。
  不默认使用未归一化的 `sum(candidate_us)/sum(baseline_us)`，因为它会被 paper 等长耗时实例主导；
  若业务明确需要总工作量口径，必须在运行前冻结权重与理由并进入 evaluation identity，不得事后
  按结果调权或截尾。
- `worst-shard latency_ratio` 仅作硬门与并列时的 tie-break，之后以候选 identity 确定性打破完全并列；
  没有有效候选时输出 `no_winner`，不得降低门限制造胜者。
- 当前单 Track `taic run` 主要产出一个 `(dtype,size)` 结果，并不天然含多 shard。MT5 必须逐个
  benchmark 调用/汇总对应 run_dir、result、confirmation 与 identity，不得假装单 run 已有多分片证据；
  若复用 P2 多 shard portfolio 激活路径，则必须按其全域、fallback 和发布语义另行验收，不能混用。

**输出：**`track-explore-summary.json`，记录参与 benchmarks、每条 Track 的 run identity、候选/构建/
评测身份、comparability_digest、run/master seed 与派生算法版本、各自实际 confirmation_seed 与
fresh 输入身份、确认状态、逐实例指标、淘汰原因、winner、selection policy digest、agent turns、
构建耗时和 `total_search_wall_sec`；最佳性能与搜索成本分开呈现。同目录还会写
`<stem>-observations.json`（`taic.track-explore-observations.1`），逐 run 记录 wall/cost/returncode，
summary 与它同期生成并交叉核对（observe 读到不一致即拒绝展示）。

多实例 winner 记录为 `(winning_track, per-instance candidate identity vector)`：按 `instance_key`
排序，每项至少含 `(instance_key, candidate_sha256, build/evaluation identity)`，其 canonical digest
为 `exploration_bundle_digest`。它不是 runtime PortfolioCandidate，也不启用跨 Track dispatch。
首轮 winner 是 exploration winner；summary 不冒充同 fresh 输入或正式发布权威，正式发表仍走现有策略。

**固定测试：**全成功、部分 incorrect、部分 unsafe、无 baseline、no_winner、不同 baseline、
timing uncertain、cache 串用、Track 漂移、同 master/算法但不同 freeze 而产生不同 confirmation_seed
的方案 A 正例、禁止声称同 fresh 输入、comparability_digest 不一致只展示、方案 B 不冒充现有确认、
非 fresh seed、单实例 1.10 边界、
额外 `speedup>1.0` 探索门（若获批）、缺实例、per-track per-instance best bundle、禁止跨 Track 拼冠军、过门后等权 geomean 优先、
长耗时实例不再垄断、worst-shard 仅作硬门/tie-break、geomean 掩盖失败反例、不同 build timeout
但相同 agent turns、串行 quiet confirmation、重复运行幂等和 resume 互锁。

### MT6：真实 Metal 批量验收与性能边界

**当前进展（真实 Metal fresh 已通过，范围已收口）：** `/3` 使用稳定
`expected_fresh_subject.cache_key` 进入 BuildProfile；新的 `tcenv.track.tilelang.metal-build.2` descriptor
只含稳定 candidate/frontend/artifact identity，完整 attempt 证据保存在 BuildEvidence.log，避免 native_digest
随 attempt 漂移。`/2` 合同、prompt、registry 与 task-pack closure 未改。builder/probe、confirmation/
activation、V2 expected、confirm/_evaluate、run_config/Orchestrator/Snapshot dry-run 和身份负例均已接通。
固定 DeepSeek ReLU 候选的真实 Metal 两次独立构建、dispatch、fresh identity 均通过，BuildProfile、
artifact set 与 native digest 稳定；已执行 golden 均 confirmed。二十一次完整 fresh sidecar 全部保留，
累计 12 confirmed + 9 timing_uncertain。每轮 barrier 的 r7 仍全部 ratio >1.10，未采用；改为
**native concurrent + 每 2 轮 barrier** 后，r8 三次均 confirmed，ratio 为 0.9596/0.8347/0.8963，
全部 ≤1.10，达到 `stable_fresh_timing=true`、`fresh_confirmation_stable=true`、
`performance_gate_passed=true`。`/2`、MSL 和 P3 不增加 barrier。`/3` 已按 r8 证据登记 available，
默认 build/evaluate、fresh benchmark（ratio 0.803613）、golden、activation 和非 dry-run resume 均通过；
新 registry digest `9b85fa11…`、prompt SHA-1 `162d027f…` 已记录，旧 MT2/MT4 包未改写。
`/3` 为 `registered=true/selectable=true/default_registry_validated=true/reviewer_done=true`，
固定 `/3` ReLU 切片已通过独立 review；`publication_allowed=false`。同一段后面记录完整 MT6 `done=true`。
`affine_ramp` 的三 Track 真实批量探索已完成：`msl-single`、`msl-bundle`、TileLang `/3` 均
confirmed/eligible，ratio 为 0.156119/0.153255/0.074707，TileLang 为 `exploration_winner`，
`formal_publication=false`、`same_fresh_input=false`、`cross_track_runtime_portfolio=false`，
`track_explore_reviewer_done=true`。r10 因 observation 后补字段、fallback 缺执行证据和 build duration
漏算而被 reviewer 拒绝，`review-r10-rejected.json` 保留；r15 重新生成同期 observation/summary，补齐
`observed-execution.json`、真实 `native_batch_calls=0`、完整 build evidence 和成本/turn 校验。
证据位于 `MT6/evidence/track-explore-r15/`，MT6 verifier 通过。r15 已通过独立 review（`review-r15.json`），
Reviewer 独立重算 summary 与归档完全一致，并重跑 142 个兼容节点、55 个合同/registry 节点、
MT0/MT5 verifier。`scope-decision.json` 已正式将 MT6 范围调整为 **MSL + TileLang**，不再新增第三条 DSL，
`scope-review.json` 已独立签署范围复核通过，完整 MT6 `done=true`，`publication_allowed=false`。

**目标：**用固定候选在真实 Metal 上验证多 DSL Track 和轻量批量工具，而不是只做 mock。

下面的验收清单是原计划。已签署的范围以上面的「当前进展」和文首为准：真实路线是 `msl-single`、`msl-bundle` 与 TileLang `/3`。

**验收内容：**

1. 原计划写的是现有两条 MSL Track 与两条新 DSL Track。本轮实际独立构建执行的是两条 MSL Track 与 TileLang `/3`；
2. 同一任务、输入、baseline、种子、计时和确认规则；
3. 批量工具生成各 run_dir 和 summary，live 结果与归档一致；
4. 至少一个场景中有多于一个有效候选，证明确实能按冻结规则选最佳；
5. 另有不正确、无有效候选和计时噪声场景，正确输出淘汰或 `no_winner`；
6. 每个 shard 独立满足现有性能门，不能用 paper/geomean 隐藏 small 或其他 shard 失败；
7. 没有真实 fresh evidence 时保持 blocked，不挑选幸运一轮。

**性能纪律：**candidate wall / fastest registered baseline wall；诊断 GPU-only、上游公开倍数、
模拟器和构建成功都不能替代生产 fresh evidence。

**清微：**若做 DSL parity，只复用离线模拟，输出 `simulated=true/publication_allowed=false`；
不启动真实 NPU 宣称，不解除 NPU-P0。

### MT7：文档、可观测性、回归与独立终审

**当前进展（文档、只读可观测性与终审已完成）：** `README.md`、
`README.zh-CN.md`、`backends/metal/wiki/tilelang.md`、`wiki/INDEX.md` 和 `docs/agent-skills.md`
已更新 `/1`–`/3` 事实与边界；`observe/track_explore.py` 和 `/api/v2/track-explore` 提供只读
r15 winner/ratio/成本展示，不是发布权威。`tests/support/golden/shared/多DSL-Track-阶段验收/MT7/` 的 verifier
与 manifest 通过，3 个 observe 节点加结构共 9 个节点、45 个 track-explore 节点通过。
r15 已通过独立 review，r10 保持 rejected。范围已收成 MSL + TileLang，
`done=true`、`publication_allowed=false`。Graph 与无界搜索未开放。

**主要工作：**

1. 更新多 Track 对比文档：把已完成的 DSL Track 从“目标”移到“当前事实”，未完成项继续保留目标。
2. 更新 `AGENTS.md`、`.pi/skills/shapeforge/SKILL.md` 和必要的 backend DSL 指南；只写已验证能力。
3. 如公开使用方式变化，同步 `README.md`、`README.zh-CN.md`、`CONTRIBUTING.md` 和
   `docs/agent-skills.md`，不创建 `docs/README.md`。
4. `observe` 可选增加按 Track 只读筛选与批量 summary 展示；不得让 dashboard 成为发布权威。
5. 精简多 Track 对比文档中重复的目标声明：详细定义保留在第〇节与常见误解，正文重点讲机制；不再反复强调同一否定句。
6. 所有 MT 阶段 verify/manifest、固定负例、真实 Metal 证据和独立 reviewer 结论齐全。
7. 最终独立 RLAR reviewer 检查候选自由、路线平等、证据不串用、无弱化测试、无未授权费用。
   reviewer 最终只输出合法 `{done, notes}`；主会话不得自签。

---

## 5. 测试与证据要求

### 5.1 每阶段固定测试层

| 层级 | 内容 | 不得冒充 |
| --- | --- | --- |
| 纯 CPU/schema | registry、candidate、frontend identity、CLI、批量汇总和负例 | 真实构建或 Device 执行 |
| worker/协议 | 编译失败、trap、timeout、资源、身份、replay、fresh 输入 | Metal correctness/performance |
| 真实 Metal | DSL build、Host+Device、dispatch、correctness、计时、fresh confirmation | 所有 shape/所有 DSL 程序的一般证明 |
| 清微模拟 | 语义/协议/身份 parity | 真实 NPU correctness/performance |
| fake agent | DSL 生成、文件保真、批量调用和失败处理 | 真实 LLM 生成质量 |

### 5.2 常规命令

只读/CPU 阶段使用：

```bash
uv run pytest -q -m 'not gpu and not llm and not slow'
```

单失败串行诊断：

```bash
uv run pytest -q -n 0 <nodeid>
```

真实 Metal 只在用户明确授权后运行相应固定节点；`taic run`、`taic resume`、真实
`taic track-explore` 和真实 LLM 评测均需单独授权并设置费用/时间上限。

不得通过删除测试、skip/xfail GPU 节点、删除 shard、降低容差/性能门或挑选较快一轮来完成阶段。

### 5.3 每阶段证据身份

至少记录：

- HEAD、工作树 manifest 和基线 epoch；
- TrackSpec/registry/schema digest；
- DSL source/frontend/compiler/options identity；
- candidate/source/geometry/bundle/artifact/native identity；
- actual build/execution/dispatch；
- baseline/oracle/environment/evaluation identity；
- fresh confirmation 与 quiet timing；
- `simulated/publication_allowed` 边界；
- 固定 node id、负例和 verifier 自身 self-test。

Build evidence 不代替 evaluation evidence，Raw Observation 或自报 JSON 不代替真实执行。

---

## 6. 主要风险与对应阻止条件

| 风险 | 阻止条件 |
| --- | --- |
| 找不到两种适合本地目标的 DSL | MT0 blocked；不引入 CUDA/CANN，不用 MSL 改名凑数 |
| 名为多 DSL，实际都强制转成现有 `msl-single/msl-bundle` Track | 至少一条路线结构证明不经过现有 Track 合同/adapter；后端生成 MSL 不自动否决，但玩具薄包装不算独立 frontend |
| 为凑第二条路线使用玩具 DSL 或清微模拟 | 玩具自研格式不得计入；模拟永远 `simulated=true/publication_allowed=false`，整体 Metal 目标继续 blocked |
| 每条 DSL 重造安全/计时执行器 | MT1 未拆出共享 Metal 执行审计底座则不得进入 MT2/MT3；闭源 runtime 不得绕过 |
| DSL JIT/global cache 把输入相关准备藏到 warmup | fresh-rotation/mixture 交错与缓存命中审计失败则阻断，不能只看 timed 自报 |
| 把 Track 变成逐题算法注册表 | schema/结构测试拒绝 op/shape/tile/planner 字段 |
| 新 DSL 绕过 worker 或计时 | 控制机编译/加载负例、输入驱动 Host 计时审计失败即阻断 |
| DSL 源/frontend 身份丢失 | source/frontend/options/artifact digest 负例，不能 replay 则阻断 |
| 各 Track 证据串用 | cache/build/confirmation identity 隔离测试，旧证据不得直接继承 |
| 把不同候选的 fresh 输入误称为同一输入 | 方案 A 只声明同分布探索比较，记录实际 seed/input identity；同输入须另做方案 B，不改旧派生规则 |
| 不同评测口径导致假竞速 | comparability_digest 不一致只展示，各 Track 完整身份另存，不选 winner |
| 为产生 winner 挑选幸运运行 | `timing_uncertain`、quiet fresh、no_winner 负例；禁止删除失败 shard |
| 批量脚本膨胀成中央调度器 | 代码边界只允许批量启动、读取、校验和汇总；TrackRegistry 不做竞速 |
| 真实 LLM 未授权 | fake agent 只能证明协议；真实生成状态不得标完成 |
| 清微模拟被当成真实 NPU | 强制 `simulated=true/publication_allowed=false`，NPU-P0 不解除 |

---

## 7. 完成定义

下面 10 条是起草时的完成定义。本轮实际签署的边界在文首：两条 MSL 交付加上 TileLang `/2`、`/3`，以及批量探索、文档和终审。第 1 条里的「至少两条非 MSL DSL」已由范围复核取消，不能用来把本轮记成未完成。Graph、第三条 DSL、无界搜索和 speedup publication 仍未开放。

起草时的完成条件是：

1. 至少两条非 MSL DSL Track 各自具有独立、真实的 Metal 构建执行证据；
2. 它们与现有 MSL Track 平等共存，至少一条不经过现有 `msl-single/msl-bundle` Track 合同或 adapter；是否由后端生成 MSL 不是判据，独立 frontend/优化/调度与身份才是判据；
3. 各路线可以拥有独立 Host 调度，未引入中央逐题 planner 或 pattern 编译体系；
4. LLM/人工候选生成、抽取、构建、执行、resume/replay 全链身份完整；
5. 轻量工具能逐条运行多个 Track，按冻结规则输出最佳有效候选或诚实的 `no_winner`；
6. 正确性、安全、真实 dispatch、fresh confirmation 和唯一发布谓词未弱化；
7. 现有 MSL Track 回归通过，历史证据未改写；
8. 所有 MT 阶段包、固定负例、manifest 和独立 reviewer 验收完成；
9. 真实 LLM、真实 Metal 与性能结论只覆盖实际获授权并验收的范围；
10. 清微继续只作模拟，未宣称真实 NPU 能力。

若只有批量比较脚本完成，只能宣称“多 Track 结果汇总工具完成”；若只有 DSL adapter 完成，
只能宣称对应 DSL Track 机制完成。两者都不能单独冒充整个目标完成。

---

## 8. 审核意见处理记录

针对 `.pytest_cache/argon11.md` 的审核，本计划作如下取舍（不改变历史审核原文；后续
`.pytest_cache/argon12.md` 的采纳与修正在本节末尾另列）：

| 审核建议 | 处理 | 理由 |
| --- | --- | --- |
| “独立路线”不应禁止后端生成 MSL，应看 frontend/优化/调度独立性 | **采纳并收窄** | 独立判据是不经过现有 Track 合同/adapter，而非禁止 Apple 后端；同时拒绝玩具薄包装 |
| MSL 相当于 PTX、99% 路线都走 MSL | **不采纳为事实依据** | MSL source/AIR/metallib 层级不同，PTX 类比不准确；百分比没有可核来源 |
| 若只有一种成熟 DSL，可用清微模拟或 C++ 模板路线凑第二条 | **部分拒绝** | 可以做补充机制证据，但不能顶替两条真实 Metal Track，不能用玩具 DSL 凑数 |
| MT5 提前到 MT1 后，先用现有两 Track 验证 | **采纳** | 降低依赖风险，提前冻结批量协议，新 DSL 后续直接加入 |
| 拆分 Track Frontend 与 Target Execution Substrate | **采纳** | frontend/调度独立与 Metal 执行审计统一并不矛盾，避免复制安全逻辑 |
| winner 使用硬门 + 过门后总有效耗时排序 | **采纳硬门，后续按 argon12 修正排序** | 先逐实例过硬门；单实例比 `latency_ratio`，多实例采用等权 `geomean(latency_ratio)`。总有效耗时仅可作预注册可选口径，不截尾、不删除实例 |
| `latency_ratio <= 1/1.10` 且要求至少 1.10x 加速 | **拒绝** | 与现有公式和门限不符；保持 candidate/baseline <= 1.10，不能改成另一门 |
| 生成轮数与构建超时分开，记录搜索成本 | **采纳** | 公平比较搜索深度，同时为慢 frontend 设置有限构建上限，记录 wall/cost |
| 防止 JIT/mutable cache 把输入相关准备藏到 warmup | **采纳并收窄** | 使用已登记 fresh-rotation/mixture/stateful timing，禁止临时改变 shape 或测量口径 |
| 精简对比文档的重复防御性声明 | **纳入 MT7** | 目标确认与实现完成后统一精简，避免现在反复改动同一文档 |

针对 `.pytest_cache/argon12.md` 的代码级审核，追加处理如下：

| 审核建议 | 处理 | 理由 |
| --- | --- | --- |
| 补入 portfolio candidate、Metal search_candidate、portfolio_confirmation 三处硬编码 | **采纳** | 直接构建通过不代表单候选搜索与 fresh confirmation 能跑通；三处必须共同设计、重验 |
| 推荐 project_provider 扩展并承诺 P2 零修改 | **部分拒绝** | 可复用 PortfolioFreeze/subject 语义，但代码路径与身份会前进，需新 epoch、负例与必要重录 |
| Metal 底座区分 kernel_pipeline 与 graph_executable/command_encoder | **采纳** | 统一 buffer/计时/证据根，不要求图路线虚构 grid/threads；graph kind/descriptor 版本化 |
| MPSGraph/MLX 可直接注入 CommandBuffer 的既定事实 | **收窄为 MT0 核实项** | 需用实际 API/执行验证外部编码、Host 审计与 worker 边界，不把黑盒 runtime 当合格 Track |
| 区分单实例 taic run 与多实例探索 | **采纳** | 当前 run 不天然含多 shard；显式 benchmarks 与 per-track per-instance best bundle，禁止冒充 |
| 多实例排序不用未归一化 sum/sum，改等权 geomean | **采纳** | 避免 paper 长耗时主导；等权 geomean 只在每实例过硬门后排序，不掩盖失败 |
| 单实例必须 `confirmed_speedup>1.0` 是现有规则 | **拒绝该事实断言** | 当前 subject verifier 没卡此门；如要更严格探索准入，单独批准，不修改唯一发布谓词 |
| 禁止修改 taskcontract/opspec 且 pack digest 必须逐位不变 | **部分拒绝** | backend 工厂隔离是对的，但 digest 含文件内容；通用合同修改需登记实际差异并重验，不能承诺零变化 |
| prompt 指纹可能前进，但旧 TASK/presentation 字节需固定 | **采纳** | `inspect.getsource` 覆盖函数字节，源码改即指纹前进；旧授权影响需显式记录 |
| MT0 给出三类 Apple DSL 探测方向 | **采纳为调查清单** | 不是已验证 DSL 名单，均需过 frontend/Host/CommandBuffer/worker/证据可行性门 |
| 跨 Track 固定 run/master/confirmation seed | **采纳可共享部分，按 argon14 修正** | run/master seed、算法与输入生成器一致；但 freeze 含候选使 confirmation_seed 不同，首轮只能声明同分布探索，不声称同 fresh 输入 |
| NativeExecutable.kind、single-source helper、descriptor 闭集纳入盘点 | **采纳** | graph/JIT 需要新 kind/descriptor 与 schema-4 负例，live handle 不进 bundle |

针对 `.pytest_cache/argon14.md` 的审核，作如下修订（其 `REQUEST_CHANGES` 只针对当时的计划草案，
不自动改变本计划的授权状态）：

| 审核建议 | 处理 | 理由 |
| --- | --- | --- |
| 跨 Track 字面 confirmation_seed 相同与现有 freeze 派生冲突 | **采纳，首轮选方案 A** | 同 master/算法/实例/输入生成器，各自 fresh 输入，输出 exploration winner；真正同输入留方案 B 新 epoch，不改旧派生或冒充现有确认 |
| graph 在 evaluate_task/CanonicalBundle 缺几何接缝 | **采纳并缩小首期** | 首期只做 kernel_pipeline；graph 必须先有 tagged ExecutionPlan、冻结绑定/encode plan 与 artifact/expected 替代，不假造 grid |
| 嵌入式 Python/C++ 候选超出现有 worker 威胁模型 | **采纳** | MT0 分类 declarative/embedded_host_code；后者无 OS 级文件/网络/凭据隔离只能 interface_only，黑名单不代替沙箱 |
| TrackRegistry 同 ID 多版本共存 | **采纳** | `(track_id,track_version)` 共存，旧版用于 frozen/resume/replay，新版用于新 run；固定版本恢复负例 |
| MT0/MT1 依赖矛盾 | **采纳，拆 MT1a/MT1b** | DSL 无关工作与 DSL-specific schema 分离，MT0 blocked 时可做 MT1a/MT5，但不能冻结 MT1b |
| comparability_digest | **采纳并纠正理由** | 明确共同可比元组，同时保留各自完整 build/evaluation identity；不声称完整 identity 必含 candidate |
| available 按 capability 证据作用域、扩能力升版本 | **采纳** | 未验收子集 typed 拒绝，避免全局布尔暗示全域支持 |
| frontend 真实版本/options/device 进 Build identity | **采纳** | 每次 Build probe、expected 和缓存键均绑定，覆盖编译器版本漂移 |
| benchmarks 限 benchmark、预算 scope 与多实例身份向量 | **采纳并收窄** | golden/correctness-only 可探索但不能性能排名；默认 budget 每 track×instance；多实例为获胜 Track+身份向量及 digest |
| MT2 不强制 Host work-list，graph shape/Build 边界 | **采纳** | 独立调度所有权按 DSL 形态证明；区分编译期 shape 与 L2 runtime resolved total/count |

针对 `.pytest_cache/argon13.md` 的最后一致性审核，追加三项修订：

| 审核建议 | 处理 | 理由 |
| --- | --- | --- |
| 修正 argon11 表中的“总有效耗时排序”与 MT5 等权 geomean 冲突 | **采纳** | 审核决议表与正文统一为：逐实例硬门，单实例 ratio、多实例等权 geomean，总耗时仅预注册可选 |
| Graph/JIT 首次编译必须归 Build，重编更新构建身份 | **采纳并补足** | 禁止 timed 首次 tracing/JIT 和计时外静默重编；编译完成不豁免 input-dependent Host 计时 |
| `per-instance best bundle` 改名为 `per-track per-instance best bundle` | **采纳** | 每条 Track 自己的实例最佳向量整包比较，最终一个获胜 Track；不跨 Track 拼冠军或建立 runtime portfolio |

## 9. 建议实施顺序与授权点

```text
MT0 DSL 选型与可行性（只读，填实两个 kernel_pipeline DSL）──┐
  ↓ 用户确认两种 DSL 与目标边界                              │
MT1b DSL 合同、多版本 registry、ExecutionKind/native 身份       │
  ↓                                                          │
MT2 DSL X 垂直切片                                           │
  ↓                                                          │
MT3 DSL Y 垂直切片与平等性                                    │
  ↓                                                          │
MT4 新 DSL 的 LLM/人工生成接入（先 fake，真实模型另授权）      │
  ↓ 将新 DSL 加入已验收的 MT5 批量协议                         │
MT6 真实 Metal 批量与性能验收（GPU 另授权）                    │
  ↓                                                          │
MT7 文档、回归与独立终审                                      │
                                                             │
MT1a DSL 无关去特判、CLI/prompt/adapter 骨架（可与 MT0 并行）  │
  ↓                                                          │
MT5 先用现有 msl-single/msl-bundle 打通批量探索与汇总 ──────────┘

Graph/embedded_host_code 若没有各自构建缝与隔离证据，只保留后续工作，不进入首期完成定义。
```

MT1b 的入口条件原为两项同时满足：MT0 选型确认，且外层 TileLang source-only probe（含负例与独立复核）完成。第二项已达成。第一条「两种 kernel_pipeline DSL」没有达成；独立范围复核随后取消了第二条 DSL，不再把这件事记成当前阻断。TileLang 的 MT1b、MT2、MT4、MT6、MT7 已按文首范围结束。MT3 Graph 未启动。不得用 Halide、玩具格式、MSL 改名或清微模拟再开一条 DSL 来补原计划。

**MT1b 已完成范围（TileLang-first，显式用户批准，2026-11）：** 用户当时批准先推进 **TileLang 一条** DSL Track 的 MT1b。当时写入的范围只覆盖 CPU/协议：

- 冻结 task-agnostic `kernel_pipeline` TileLang Track 合同（`tilelang-pipeline/1`）：候选源
  `kernel.py`/`text/x-python`/`dsl_source`、候选自有 `host/execution_geometry.json` 几何、
  前端/入口/options 身份、source-only 与 Metal build 的显式边界；Python DSL 不再伪装成
  `kernel.metal`，也不要求走 msl-bundle 的 HostTilingPlan；
- TrackRegistry 升级为 `(track_id, track_version)` 多版本共存（schema 2，新 epoch）：新 run 取
  最高可用版本，冻结恢复/resume/replay 按记录版本精确重建，版本漂移/重复 (id, version)/未知
  版本全部 typed 拒绝；schema-1 文档降为 replay-only，旧 registry 字节归档后仍重建原 digest；
  `(track_id, track_version)` 已贯穿 `run_config` → CLI → Orchestrator → TaskBinding →
  renderer/evaluator/确认闸门，记录版本在 v1/v2 同时可选时仍恢复 v1（端到端负例），记录版本
  不再被 `int()` 强转，v1 run 若版本行已退役则拒绝续跑而不是漂移；
- Candidate/CanonicalBundle/ExecutionGeometry 走真实 `kernel_pipeline`（没有 graph、没有伪造
  stage/grid）；build 身份绑定**完整候选 manifest（含几何）** + 前端/options + 设备无关环境 +
  真实尝试终态 + 闭集工件内容，缓存键与 expected 主语同源，同源不同几何不再碰撞；
- source-only build 接缝接到真实 bounded 尝试：控制机先对 staged 安装做逐字节观测（MT0 冻结的
  `06f3b0ad…`/`380d3486…`/`6ba83c2e…`、3651 文件/134219728 字节，格式正确但内容漂移即
  `IdentityDrift`），再以现有 lower-level hmz Profile 运行 harness-owned entry，回读并核验工件
  内容后才产生 `full_build`；空转/垃圾字节/自报摘要都不能铸造 full build，Metal build 仍是
  typed 能力拒绝；
- portfolio/search/preflight/confirmation 三个接缝按 Track 合同拒绝 DSL，不再用 geometry 有无
  决定家族；
- 新 Track 在 MT1b 结束时登记为 `interface_only`：当时不可被新 run 选择、没有 Metal build/dispatch、不可确认、不可发布。
  现行 `tilelang-pipeline/2` 与 `/3` 已是 `available`；`/1` 仍是 `interface_only`。见文首。
  source-only codegen 是真实实现并可执行（bounded 围栏、0 dispatch、`publication_allowed=false`），
  但它不是 Metal 证据，也不把当时的 `/1` 变成 `available`。
- **r3 复核修正（仍在同一 epoch）**：(a) 围栏内生产入口的 prim_func 类型改为当前 TVM 的
  `tvm.tirx.PrimFunc`——pinned `tilelang 0.1.15` 的 TVM 只有 `tvm/tirx`，没有 `tvm/tir`，`@T.prim_func`
  实际产出 `tvm.tirx.function.PrimFunc`，旧检查会 `AttributeError`；(b) 构建正例改用 MT0-r5 的合法程序，
  历史 r3 parser 失败输入逐字节保留（两者只差旧 import 与 `@T.prim_func` 装饰器）；(c) 冻结
  `(track_id, track_version)` 显式进入 preflight、evaluator 独立 expected 重建与 confirmation/activation
  全链（`Orchestrator._confirm_v2` → `confirm_candidate` → `confirm_subject` → `activation_expectation` /
  `DispatchManifest.from_confirmations` / `validate_confirmation_for_activation` 及
  `track_factory._confirmation_identity`），同 id 的新可选版本不会改变旧 run 的投影、确认或激活；(d) 归档 B1–B4 GPU 中位值在复核窗口内被一次真实 Metal 后端会话重录——只有
  `median_us` 变化、候选/几何/工件集身份不变，不属 MT1b、未被回滚，单独记录在
  `MT1b/b-evidence-drift.json`；后续已在独立 `B-evidence-rerecord-2026-11-07` 保存新观察并恢复封闭包，
  原 incident record 未改，写出边界已改为不再自动覆盖历史。

**MT1b 当时明确未完成：** 真实 Metal build/dispatch/correctness/timing、生产 DSL、`available` 状态、
第二条非 MSL DSL、真实模型/付费搜索、GPU/NPU 证据。source-only 不等于 Metal 或生产可用。
这些未完成项里，Metal、`/2` `/3` 的 available、MT2/MT4/MT6/MT7 与 P2-F 后来完成。第三条 DSL 与 Graph 没有做。`publication_allowed=false`。
本轮执行环境拒绝嵌套 `sandbox-exec`，因此那次真实围栏 codegen 尝试以 typed
`backend_capability_unavailable` 结束（尝试请求/驱动记录/限制报告均保留为证据）；代码路径、工件
核验与身份派生已按真实尝试实现并有固定负例，但"真实产出一次新的 source-only 工件集"仍未在本机
观察到，见 MT1b acceptance 的 “环境阻断”。
MT1b 验收包见 [`MT1b/`](../../tests/support/golden/shared/多DSL-Track-阶段验收/MT1b/acceptance.md)。**MT2 已在其单独 GPU 授权下
完成实现、真实证据与独立 RLAR 终审（done=true）**，验收包见 [`MT2/`](../../tests/support/golden/shared/多DSL-Track-阶段验收/MT2/acceptance.md)。
MT3 Graph 未启动。MT4、MT6、MT7 与 P2-F 后来完成，见文首。第三条 DSL 不再新增。`publication_allowed=false`。
