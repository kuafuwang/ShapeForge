# varlen 与输入驱动 Host Program：GPU / NPU 跨后端实施计划

状态：**V0–V6 的 Metal varlen 机制、L2 continuous batching 与真实规模验收已完成；Host Program
已从 V3 Wasm 迁移为 native64 bounded-worker，P2 已在安静窗口完成最终重录**。V4 Dyn-1 按实测
跳过。V6 六档证据与 baseline 边界见 `docs/架构/v6-chunked验收/README.md`，P2 重录及保留的
历史 start-surface 锚点见 `docs/架构/shape-portfolio-阶段验收/P2/F.md`。此前 V3 Wasm 段落仅作
历史设计记录，不是当前执行路径；付费搜索、P3-B/C 与 Qingwei NPU-P0 不因本结论完成。

本文是 varlen 工作的唯一施工图。目标不是把某个 KDA kernel 搬进仓库，而是把 KDA、昇腾
CAKE2 与 CANNBot FA 技能中经过核对的机制，抽象成 **Metal 与 NPU 都能复用的候选交付和评测
模型**。

当前生产后端仍是 **Apple GPU / Metal**。NPU 侧当前只允许离线协议/数据/调度模拟，结果必须
`simulated=true`、`publication_allowed=false`；清微 NPU-P0 保持 blocked。昇腾 CAKE2、CANN 与
CANNBot FA 只作为机制参考，不引入其源码、工具链、专用 guidance，也不形成 Ascend 生产后端。

---

## 0. 设计结论与边界

### 0.1 不再采用的简化表述

“varlen 不需要运行时形状”是不准确的。准确说法是：

> 已实现的 V2a 把每个声明分布的总形状静态化，因此能复用固定 shape ABI；完整 varlen 仍需处理
> **同一形状下任意合法分段**，以及更进一步的 **每次调用 total_tokens/num_seqs 都变化**。

V2a 的 `seq_lens -> total_tokens/num_seqs/NSP1`、分段 fp64 oracle、逐段 `final_state` 和基础
Metal 正/负例保留为回归夹具，但不再把 V2a 称为完整 varlen 支持。**V2 Dyn-0 已在同一静态 ABI 下
证明任意合法分段**：device 按实际 `cu_seqlens` 建 work list、partial tail 掩码、逐段终态，以及
预处理 + 顺序链和真实 piece handoff 两类多阶段 fixture（§6.2）。**Host Program 把这份输入驱动的
work list 改由候选侧受限 C99 生成**。历史 V3 用 Wasm 验收过该机制；当前 native64 bounded-worker
实现已通过 CPU/协议、NPU 模拟 parity 与真实 Metal 复验。L2（total/count 也变化）仍是 V5。

### 0.2 两种“动态”必须分开

| 层级 | 一次调用中什么会变 | 需要的能力 | 是否 KDA 已覆盖 |
| --- | --- | --- | --- |
| L0 固定 shape | 不变 | 当前 OpSpec v2 | 固定长度是退化情形 |
| L1 packed varlen | `total_tokens/num_seqs` 固定，`cu_seqlens` 的合法分段可变 | 任意分段正确性、输入驱动 host work list、tail、handoff | 是，核心学习目标 |
| L2 runtime varlen | `total_tokens/num_seqs` 也可变，但不超过声明 capacity | logical/resolved shape 分离、capacity ABI、resolved grid/scalars | ShapeForge 为 continuous batching 增加的泛化，不冒充 KDA 原机制 |

KDA 每次调用也会先拿到实际 `T`、分配实际张量并读取实际 `cu_seqlens`；它没有要求我们用
“一个 max-capacity buffer 服务所有调用”。因此 **L1 是学习 KDA 的核心，L2 是后续生产泛化**。
最终完整验收必须包含 L2，不能用 L1 代替。

### 0.3 不变的架构纪律

1. **候选拥有算法**：分段策略、chunk、LPT/range-cut/自定义 planner、fusion、persistent kernel、
   单/多 stage、handoff、Tiling、ExecutionGeometry 均由候选决定。harness 不内置 KDA 调度器，
   不新增按任务名/shape 查表的 registry。
2. **harness 拥有边界**：任务合同、输入校验、ABI/capacity、Host Program 安全子集、workspace
   读写范围、计时、fallback、证据和发布门。
3. **不发明第三份 schedule 文件**：继续只有 device source、`host/tiling.*` 与
   `host/execution_geometry.json`。Host Program 是 `host_tiling` 的另一种表示，不是新 schedule。
4. **固定 shape 不退化**：非 varlen 实例的 ABI、冻结投影、父包字节、既有证据身份保持不变；
   单文件与既有 JSON linear bundle 继续工作。
5. **不改历史协议来凑功能**：R5、`verify_subject_document`、六 RPC/字段号/HMAC/BlobChannel
   不动。需要新证据时开新 epoch/文档版本，不把模拟写成实测。
6. **真实 GPU、真实 NPU、付费搜索分别授权**。CPU 合同测试可以先行；NPU 模拟不等于真机验收。

### 0.4 独立审核后的四项架构修订

本计划已按独立审核意见推翻初稿的四处设计，理由是不应为了抽象统一牺牲真实性能或硬件安全：

1. **不在计时圈内解释执行自研 Python AST，也不再使用 Python↔Wasm**。最新方案把候选受限 C99
   编译为 native64 `.dylib`，但只允许在 bounded v2 worker 内编译、加载和执行；控制机不加载
   native library、不接触指针。最终 dylib 做未解析符号/导出检查，外部 watchdog 处理 runaway。
   候选算法仍由候选书写，不把 KDA 的 LPT/range-cut 变成 harness 专用 intrinsic。
2. **Metal 不以跨 threadgroup 自旋等待作为主验收路径**。Apple GPU 没有可移植的跨 TG 前向进度
   承诺，首选现有 serial stage + `memoryBarrierWithScope_(MTLBarrierScopeBuffers)` 的多阶段 handoff。
3. **不按 capacity 全量写 workspace**。统一使用有效 span + poison/sentinel 尾部，避免小请求为
   max-capacity 的无效区付出序列化和拷贝成本。
4. **resolved scalar/动态 stage 不作为 V2 前置**。即使 L1 的 total/count 固定，分段也会产生不同的
   chunk 数与 handoff 需求；Dyn-0 先用静态 dispatch + 运行时 count 证明语义，Dyn-1 再按实测决定
   是否启用动态 grid/stage-active。

审核建议中的“device kernel 必须达标 + host 时间低于固定预算”**不作为新的发布谓词**：纯 device
时间可诊断，host 时间也可设反馈预算，但发布仍只能使用同一 E2E wall 与现有 unique predicate，
不能用 device-only 达标掩盖端到端回退。

### 0.5 第二轮代码契约审核后的落地修订

对照当前实现再次核对后，进一步冻结以下衔接规则，避免计划与现有构建/证据门冲突：

1. 若启用 Dyn-1，几何明确拆成构建期 **`ResolvedEnvelope`** 与逐 case **`LaunchGeometry`**；
   Dyn-0 与旧静态几何的 `selection_parity` 行为不变，动态 launch 走新的逐 case parity，而不是放松旧审计。
2. native64 使用系统 Clang 生成 worker-local `.dylib`，不进入 Schema-4 `ArtifactBundle.entries` 或
   `BuildOutcome.payloads`；`NativeExecutable.descriptor["host_program"]` 绑定 native digest 与真实
   assembled digest，executor 复用 library，异进程重建必须逐字节一致。编译/加载只在 bounded worker。
3. `host_inside_timing` 路径先修复只读输入重拷、计时圈内重建参数和 zero-fill 双 CommandBuffer
   问题，再测 Host Program；双提交的成本以本机实测为准，不引用未复现的固定微秒数。
4. L1 保留 canonical `cu_seqlens` 在 `SemanticInstanceV2` 指纹中，避免现有 SemanticDomainPlan
   身份碰撞；L2 capacity 实例才将其置 `None` 并启用独立 workload profile。
5. 静态 `sizes.seq_lens` 继续要求每段 ≥1；零长度只由运行时 `cu_seqlens` case 构造。

### 0.6 第三轮审核后的分级与工程收口

为避免把最贵的 Runtime/工具链改造放在正确性证据之前，施工进一步分为 **Dyn-0 / Dyn-1**：

- **Dyn-0 先行（V2 已完成，见 §8）**：静态 grid、静态 stage dispatch、静态 parity 不变，多余
  TG/stage 读运行时 count 后退出。用它证明任意分段、partial tail、final_state 和多 stage handoff，
  不依赖 Wasm/lld。
- **C99/native64 Host Program 是当前路径**：它表达候选自有的通用 planner，不用 Python oracle 的
  耗时决定是否值得存在；Dyn-0 只证明 device-side/静态路线也是合法候选。历史 Wasm 已删除。
- **Dyn-1 是可选优化**：只有实测空 TG/空 stage 成本显著时，才启用 `ResolvedEnvelope`、动态 grid
  和 stage skip。它不是所有候选的前置门，Dyn-0 与 Dyn-1 在 geometry 中显式声明并存。
- 大输入不直接映射进 native worker；只有小张量给 typed pointer，大输入走白名单 typed read callback。
- workspace 初始化改为 `host/device_zero/none(poison)` 三态互斥；按 possible stages 的并集分配，
  不按某个 case 的 active set 临时分配。
- L2 dispatch 主体仍是 capacity 实例，resolved shape 只进 input/launch 证据；NPU parity 分为同
  target 字节一致与异 target item 全集等价，不再要求 core 分配相同。

### 0.7 当前 native64 迁移状态（覆盖历史 V3 执行方式）

- 生产构建/执行使用 `native_abi.py`、`native_toolchain.py`、`native_runtime.py` 和中立
  `results.py`；Python 不计算 C 结构体偏移，唯一调用入口是无参数 `sf_native_invoke()`。
- `assembled_host_sha256` 哈希真实完整 C 编译单元，`native_sha256` 哈希实际 `.dylib`；异进程
  重建必须一致。最终二进制有未解析宿主符号或候选自定义导出时，在加载前拒绝。
- mapped input、typed read callback、有效 span、pointer/capacity/shape/target/metadata 调用后校验、
  guard/read-only 检查和原子失败均已由 CPU 负例固定。候选源码不能调用 `sf_native_*` 内部入口。
- `tcenv host-probe` 使用独立 bounded probe worker；控制机禁止 CDLL 仍可运行，候选 segfault
  返回 crash，runaway 由外部 deadline 杀死。
- Metal evidence 为 `tcenv.hostprogram.run.2`，记录 native digest/wall time；NPU 模拟复用同一
  native C99，继续 `simulated=true/publication_allowed=false`。
- 旧 Wasm runtime/toolchain/probe、Wasm32 renderer 与 `wasmtime` 依赖已删除。pack closure 现为
  **73 个文件**（后续 V5 新增 `taskcontract/runtime_shape.py`）；所有 task pack digest 前进，
  最终 P2 重录见 P2-F，不能拿旧 freeze 冒充当前身份。
- **真实 Metal correctness/timing 已通过**：native Host 10/10、V1 11/11、V2 19/19；CPU/协议
  128 passed。三次独立未插桩 full evaluation 的 Host wall 中位数 **7.583 µs**、native Host
  **4.584 µs**、candidate median **238.994 µs**、device **64.374 µs**。C 批量校验、Metal
  地址绑定与 C 原子直接提交消除了 timed 中间对象；没有删除 fresh-input/read-only/guard/span
  检查。完整数据见 `docs/架构/native-host-program-数据流耗时报告.md`，仍不是 P2/V6 结论。
- **V5 L2 capacity / continuous batching 已落地（CPU/协议与真实 Metal 机制验收，非 V6）**：OpSpec v2 档位可以声明
  `capacity` + `workload_profiles`，`SemanticInstanceV2.cu_seqlens=None`、`capacity`/`workload_profiles`
  进入语义身份；`ResolvedShape` 把 capacity/resolved/logical 三套形状分开，runtime total/count
  经 `SfHostResolved` 进入候选 Host Program 与 HostRunEvidence；capacity 尺寸的 `cu_seqlens` 前缀
  加声明 sentinel 尾部、logical 区域外逐字节 padding poison 由 harness 在调用前后校验；dispatch
  仍以 capacity 实例为唯一域点。**L1/fixed 的生成 header、tensor 绑定与 derived ABI digest 与 HEAD
  逐字节相同**，因此既有 assembled digest、descriptor 字节数与 Metal 记录不受影响；`varlen_kda`
  新增的 `capacity` 档改变了该题 opspec/instance/pack digest（未改档位的任务如 `varlen_scale` 语义
  身份不变；闭包 72→73，P2 继续 deferred）。真实 Metal 新增 `test_varlen_capacity_gpu.py` 3/3，
  覆盖 9 个 resolved batches、连续 mixture 和 resolved 篡改拒绝；既有 native/V1/V2 复验 40/40，
  最终定向 CPU 回归 282 passed。一次 full mixture candidate median 193.07 µs（cv 0.0445）、
  native Host 7.375 µs、Host wall 10.333 µs、device 22.791 µs，680 次 native call 对应 680 个
  CommandBuffer；mixture speedup withheld，不是 P2/V6 发布结论。V6 真实规模验收见 §8 V6。

---

## 1. 考察记录：KDA / 昇腾 CAKE2 到底做了什么

这一节区分**已逐行核对的实现事实**与**上游论文自报结果**。KDA 文件行号来自本地
`kda-260927-kda`；CAKE2 内容来自 arXiv 2610.31034v1 与 CANNBot 公开 PR。上游性能数字均未在
本仓库复现，不进入 ShapeForge 的发布证据。

### 1.1 KDA：一个围绕真实 varlen KDA forward 的高度协同实现

#### 1.1.1 算子与数值语义

KDA 的逐 token 权威递推（`kda-cake-ptx/kda_ptx/recurrent_kernel.py`）是：

1. q/k 分别做带 `1e-6` 的 L2 normalize，q 再乘 `scale`；
2. 每个 channel 计算 `log_decay = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`；
3. V-first fp32 state 先按 key channel 乘 `exp(log_decay)`；
4. `residual = (v - state @ k) * sigmoid(beta)`；
5. `state += residual ⊗ k`，`output = state @ q`。

TIRx 不是简单把这条循环重复 64 次就结束，而是以 `BT=64, D=128` 组织 chunk 内 G-form chain：
`S -> G=S·K^T -> v_new=U-G·T'^T -> S+=v_new·K^T -> decay`，其中 chunk 内还包含严格下三角
矩阵的求逆/对角缩放（`kda_fwd_varlen.py:24-30,957+`）。这说明两件事：

- **分段调度与 chunk 数值算法是两个独立自由度**；我们可以先复用调度机制，再让候选选择
  token-recurrent 或 chunked/matrix 算法。
- correctness oracle 必须覆盖完整递推。只比较 output、省略三角求解、常数化 norm、丢初值或
  零状态续传，都可能“看起来很快”但语义错误；详见
  `docs/可信harness/案例-KDA²-算子作弊与可信评测.md`。

#### 1.1.2 ABI 与每次调用的 host setup

`fused_setup`（`:1936-1987`）每次调用都会：

- 读取实际 `T=q.shape[1]` 和实际 `cu_seqlens`；单段时补 `[0,T]`；
- 将 cu 转为连续 int64 用于 host 调度，并检查 `T < 2^21`、`T*H*D < 2^31`；
- 由 `num_ctas = min(SM_count, H*num_seqs)` 决定 persistent CTA 数；
- 分配 item table、item count、handoff state 和 flags；
- 按 H、nseq、handoff dtype、max_items 等选择/编译相应 kernel variant。

因此 KDA 的关键不是“编译期知道某个固定分布”，而是**同一套实现根据本次输入现场生成调度**。它
也不要求所有调用共用一个 max-capacity buffer；我们的 L2 capacity 是面向连续调用的进一步泛化。

#### 1.1.3 `_host_item_table`：工作项、成本模型与两种切分策略

`_host_item_table`（`:1746-1823`）先将序列按长度降序排名，再把 `(head, sequence-rank, chunk)`
展开为工作流。其参数为 chunk cost 1、`BETA=1`、`START_COST=0.9`、`HANDOFF_COST=0.4`、
`SNAP=1`（`:57-70`）。

- 最长链相对平均负载足够短时，用 `_optimal_cuts` 做**等成本 range-cut**：二分最大 range cost，
  切点尽量吸附到序列末尾，减少不必要 handoff（`:1826-1884`）。
- 否则按整条 `(sequence, head)` 链做 **LPT**：最长链先放到当前最轻 CTA，再用 move/swap 局部搜索
  降低 makespan（`:1801-1813,1887-1924`）。
- 每个 item 用 `(word0, word1)` 表达 token offset、valid rows、src/dst handoff、first/last、
  sequence 和 head；`nvalid` 让最后一个 chunk 不必凑满 64 行。

这些是**候选算法**，不是 varlen 合同唯一合法答案。我们要保证候选能表达它们，也要允许完全不同的
planner 或纯 device scheduler。

#### 1.1.4 SMEM/global item table 与 partial tail

TIRx 默认 `MAX_ITEMS=160`：warp 0 将本 CTA 的工作项从 global 复制到 SMEM，随后所有 role 按同一
flat item index 工作（`:129-145,214-281`）；列表更长时直接读 global。每个 item 的 `nvalid`
驱动 prep、norm、beta、log-decay 与输出的有效行掩码（`:1288-1318,1435+`）。序列内部的尾块不能
再按完整 64 行做固定 box store，否则会把下一条序列当成当前序列的尾部。

这对应我们的后端中立能力：**固定容量 scratch + 运行时 item count + valid-row mask + 谓词化写回**；
不是要求 Metal/NPU 复制 CUDA 的 TMA/warp 指令。

#### 1.1.5 跨 CTA recurrent state handoff

状态 role 的规则（`:643-765,863-955`）是：

- 序列首 piece 从 `initial_state[seq]` 读取；中间 continuation 等待 producer flag 后读取 handoff；
- 序列末 piece 写 `final_state[seq]`；链中被切开的 head piece 写 handoff slot，再以 release flag
  发布，consumer 以 acquire 读取并清 flag；
- 权威 handoff 是 fp32；某些 H64 mixed/uniform 配置用 bf16 continuation buffer，但 `final_state`
  仍完整 materialize 为 fp32；
- fixed/mixed/uniform 使用不同调度选择，例如 H96 mixed 强制 whole-item LPT，其他配置保留
  chunk-linear split（`:40-43,1959-1962`）。

所以正确性不能只检查 output：必须逐序列检查终态、检查 continuation 的初值来源、检查 flag 清理和
partial tail 不推进状态。

#### 1.1.6 KDA 仓库内其他候选的补充经验

`kda-cake-ptx` / `kda-cake-cute` 是 **KDA-for-KDA 的候选名字，不是华为昇腾 CAKE2**。它们进一步
展示了 piece planner、LPT 装载、切分点搜索、单 launch/两 launch native-fp32 handoff 和按 workload
选择具体 schedule。可借鉴“多种调度共存并由候选选择”，但不能把这些名字混为 Ascend 技术。

#### 1.1.7 KDA 的 correctness 与 timing 口径

`bench.py:57-64` 的六组负载是 H96/H64 × fixed `[8192]` / mixed 六段 / uniform 八段，每组固定
seed；output 和 final_state 都检查，baseline 也承担 final_state 写回。

**计时差异不能照抄**：`prepare()` 在计时前完成 host scheduling/device tables/graph，`bench.py`
只测返回的 launch。ShapeForge 中，候选读取当前输入所做的事情属于本次求值，必须
`input_dependent=true` 且进入要求的计时边界。我们可以复用算法，不能复用“把 host work 移到
计时外”的口径。

### 1.2 昇腾 CAKE2：算子生成、交付与进化流程

#### 1.2.1 host+device 是基本交付单元

CAKE2 把 Ascend 算子看成 **host-side tiling program + device-side kernel**：host 决定数据如何
分块和搬运，device 决定计算与传输调度。完整交付还包括 CANN package、Python binding、编译、
安装和评测记录，而不是一段 AscendC 源码。

这与 ShapeForge 的 CandidateArtifact + Host Tiling + ExecutionGeometry + build/evaluation evidence
同构，也是我们应继续加强的方向。

#### 1.2.2 合同、DSL 路由与可检查 artifact

CAKE2 的流程是 request → `op_desc.json` → PyTorch reference → stateless functional form → DSL
route → host tiling/device code → CANN package/binding → evaluation。`op_desc` 保存 dtype/shape、
属性、case table、输出 shape 派生等接口事实，后续阶段不能各自猜测。

其 router 有多条 DSL track，并用 capability gate、结构特征和历史评测选择路线；CAKE-evo 并行产生
变体，Advisor 根据 profile 提出修改，artifact 与验证记录贯穿全流程。论文自报 53 个 CANN-Bench
算子、标准集 1060/1060、hidden set 98.28%、平均 speedup 1.334x；这些是上游结果，**本仓库未复现**。

#### 1.2.3 Host-level 策略与知识复用

CAKE2 的 H 类策略包括：TilingKey priority-chain dispatch、按 shape regime 选择特化 kernel、
多核负载均衡、L1/L2 memory tiering，以及针对 straggler core 的 host dispatch。CAKE Book 与
Self-Evolving Agent 则把经过验证的经验沉淀、复核并反哺后续生成。

这些机制主要解决“**如何为不同 shape/算子选择实现并组织生成经验**”，不是 KDA 那种“**如何按本次
`cu_seqlens` 将变长递推链切到各核**”。CAKE2 核心未给出通用 `cu_seqlens` work list 或 recurrent
state handoff，初始 CAKE2 插件也以 Vector 算子为主，attention 适配属于后续路线。

### 1.3 CANNBot FA 技能：昇腾 varlen/跨核机制的补充

CANNBot 的 FA/catlass 技能（不是 CAKE2 核心通用框架）提供了更具体的参考：TND varlen、累加
seqlen 与批界截断、TND 直读不转块、非整块尾处理、`needL1Carry/SameAB`、KV split host 贪心与
前缀和、CrossCore flag、AIC/AIV 协作。

它们说明 NPU 侧除了静态 TilingKey，还需要考虑序列边界、跨核状态和尾块；但这些属于特定 FA
技能，不应被误写成 CAKE2 核心已经具备通用 varlen。我们只把机制纳入 NPU 模拟检查项，不引入
AscendC/CANN 代码。

### 1.4 ShapeForge 与 KDA / CAKE2 的异同

| 维度 | KDA | 昇腾 CAKE2 | ShapeForge 当前与本计划 |
| --- | --- | --- | --- |
| 首要目标 | 一个 KDA forward 在指定 CUDA 负载上的正确性与速度 | 多算子 Ascend 交付、生成路线选择与进化 | 候选生成、可信正确性、精确 dispatch、证据绑定发布；本计划补 varlen |
| 生产硬件 | CUDA/B200 实现 | Ascend NPU/CANN | 当前只有 Metal 真机；NPU 仅模拟 |
| 交付物 | task-specific host setup + CUDA kernel + graph launch | contract/reference/host/device/CANN package/binding/evaluation | OpSpec/reference + device source + host tiling + geometry + evidence |
| host 职责 | 读取实际 cu、生成 work list、准备 handoff/flags | 按 shape/算子生成 tiling、选择 kernel 与核间划分 | 既有 linear 太弱；新增候选 Host Program 统一表达两类工作 |
| varlen 模型 | 实际 `cu_seqlens` 是一等输入 | 核心主要是 shape case/TilingKey，无通用 varlen 模型 | L1 任意分段，L2 total/count + capacity；fixed 是单段退化 |
| 调度所有权 | 具体 KDA 候选拥有 planner | agent/DSL route 与生成候选共同产生 | 候选拥有 planner/tiling/geometry，harness 不内置 KDA 路由 |
| 核间状态 | 长链可拆分，handoff state + acquire/release flag | FA 技能有 CrossCore 经验，CAKE2 核心无通用协议 | 后端中立 workspace/flag/事件语义；Metal 实测，NPU 模拟 |
| shape 特化 | fixed/mixed/uniform 选不同 schedule | TilingKey/shape regime 特化 | 现有 shape portfolio + exact one-of-N + 独立 fallback |
| 正确性 | output + final_state，固定负载与 seed | contract/reference + case 验证 | 同 tolerant/scaled_error，另加任意分段、终态、作弊和 guard 负例 |
| 计时 | prepare 在计时外，launch/graph 计时 | on-device evaluation，具体口径随任务 | 输入驱动 host work 强制计时，host/device 时间分别记录 |
| 证据与发布 | 特定 benchmark 结果，不是通用发布门 | artifact/validation/CAKE Book，上游自报性能 | candidate/input/launch/baseline 身份绑定，fresh quiet confirmation 才可发布 |
| 模拟边界 | 不涉及 | 真实 Ascend 为目标 | NPU 模拟固定 `simulated=true/publication_allowed=false` |

### 1.5 这套计划具体解决什么问题

1. **解决“固定分布档被误认为 varlen”**：L1 让同一 ABI 接受任意合法分段，permuted/skewed/fresh
   直接识别写死 canonical 边界的候选；L2 再解决 total/count 每次变化。
2. **解决 linear Host Tiling 表达力不足**：Host Program 让候选能实现 KDA 的 heap/LPT、二分、成本
   模型和位打包，也能实现 CAKE2 式 shape tiling，而不由 harness 内置任何一种算法。
3. **解决不等长序列的核间负载不均**：work list、persistent core、长链切分和 handoff 成为可评测的
   候选机制；候选可以选择 LPT、range-cut 或更好的方案。
4. **解决长链拆分后的正确性风险**：首/中/末 piece 的 state 来源、acquire/release flag、逐段
   final_state 和独立作弊负例进入合同，不再只比 output。
5. **解决 partial tail 与跨批越界**：valid rows、尾块 mask、空序列和谓词化写回成为统一要求，
   padding 不会被拿来掩盖错误。
6. **解决 host 派生 launch 与证据脱节**：静态 geometry envelope 约束候选自由，resolved launch 记录
   本次真实 grid/scalars；实际执行不能再与证据中的几何各说一套。
7. **解决调度成本被移到计时外**：输入驱动 host work 强制进入 candidate 时间，并记录 input digest
   和 host wall time；KDA prepare 的数字不会被直接拿来比较。
8. **解决 GPU/NPU 各写一套造成的语义漂移**：同一受限 C99/Wasm Host Program、workload、reference、
   case 和证据字段贯穿 Metal 与 NPU simulator，后端只负责执行与设备事实。
9. **解决 shape 过拟合与伪造通用性**：canonical profile 用于性能，permuted/fresh 用于正确性，
   tune/unseen 分布隔离，exact dispatch 之外仍有独立 fallback。
10. **解决模拟结果冒充真实能力**：NPU 只产生协议/调度模拟证据，清微与 Ascend 真机没有验收前，
    不出现 NPU speedup 或发布结论。

### 1.6 统一抽象，而不是三条实现路径

只实现一套语义：

```text
OpSpec / WorkloadProfile
        │
        ▼
CandidateArtifact
  ├─ device source(s)
  ├─ host/tiling.json（既有 linear）或 host/tiling.c（受限 C99/Wasm Host Program）
  └─ host/execution_geometry.json（静态 launch envelope）
        │
        ▼
Backend adapter
  ├─ Metal：真实输入 + 真实 workspace + 真实 dispatch + 真实计时
  └─ NPU simulator：同一 Host Program/工作项/状态协议模拟，禁止发布
        │
        ▼
EvaluationEvidence（语义实例 + workload + 实际输入 + 实际 launch + host/device 时间）
```

KDA 的 range-cut/LPT、候选自己的 planner、纯 device scheduler 都是同一合同下的合法候选；
CAKE2 的 shape/TilingKey 特化由现有 exact shape portfolio 表达，不另建第二条运行路径。

### 1.7 明确不照抄、也不承诺解决的部分

- 不复制 CUDA 的 warp specialization、TMA、tcgen05、PTX 指令或 KDA 的固定形状特判；Metal/NPU
  只复用调度与状态语义，具体指令优化仍归各后端候选。
- 不引入 CAKE2 的 DSL router、CANN package、AscendC skills 或其性能数字；本计划不是 Ascend 生产
  后端接入。
- 不把 KDA prepare/launch 与 ShapeForge end-to-end 直接比速，也不把上游 1.334x 等数字当本地证据。
- 不用 NPU 模拟回答真实 NPU 指令、ABI、容量、时序或性能问题；这些仍需对应真机与独立阶段。
- 不采用计时圈内的 Python AST 解释器，也不为了提速改为加载不可信原生库。Host Program 使用
  受限 C99/Wasm；Wasm 的线性内存、import 白名单、fuel 与现有外部 watchdog 共同构成边界。

---

## 2. 后端中立合同模型

### 2.1 分离三种身份：语义、workload、实际输入

V2a 把 `cu_seqlens` 直接放进实例指纹，解决了“不同声明分布不能串证据”，但还把“声明的
benchmark 分布”和“本次运行时输入”混在了一起。新模型分开：

| 身份 | 内容 | 用途 |
| --- | --- | --- |
| `semantic_instance_digest` | dtype、rank、静态 shape 或 capacity、tensor ABI、attrs、数值合同；**L1 还含 canonical `cu_seqlens`** | 判断这是不是同一个算子 ABI；L1 同时避免同总形状分布撞身份 |
| `workload_profile_digest` | 声明的 canonical `seq_lens`、profile id、usage、case-domain 版本 | 性能档、exact dispatch、fresh seed、候选覆盖计划；L2 的必需身份 |
| `evaluation_input_digest` | 本次实际 `cu_seqlens` 字节、case、seed、必要的输入身份；L2 另绑定声明的 workload profile（scope 是边界字节+形态+profile，不是整输入内容哈希） | 重放、Host Program cache、审计这次到底跑了什么 |

规则：

- fixed-shape 实例不增加新键，既有 `semantic_instance_digest` 逐位不变。
- **L1 兼容规则**：`SemanticInstanceV2.cu_seqlens` 继续表示该 size 档的 canonical 分段并留在
  fingerprint 中，`SemanticDomainPlan` 的闭集 schema 不变。这样 `[8,8]` 与 `[15,1]` 虽然总形状相同，
  仍是不同 domain point。
- 同一 L1 档下的 permuted/skewed/fresh 等是 **Evaluation Cases**，不是新的 domain point；它们的
  实际输入由 `evaluation_input_digest` 区分，reference 始终读取实际 `inputs["cu_seqlens"]`。
- **L2 演进规则**：capacity 档只声明 max total/count，不再绑定单一 `seq_lens`，其实例的
  `cu_seqlens=None`，semantic digest 只绑定 capacity ABI；具体 benchmark 分布由
  `workload_profile_digest` 标识。需要新 domain document 时开显式版本，不悄悄改变 L1 schema。
- benchmark 证据必须绑定 canonical workload；correctness 证据还必须能审计实际分段。
- unseen profile 与 tune profile 按规范化 `seq_lens` 去重且互不相交；unseen key/master seed 不进入
  候选视图。
- 不允许只凭 `total_tokens/num_seqs` 判断两次 varlen 性能观测相同。

### 2.2 `cu_seqlens` 是运行时输入，不是候选常量

统一约束：

- **静态声明与运行时输入分层**：OpSpec `sizes.<tier>.seq_lens` 继续要求每段为正整数（≥1），
  保留现有 `[0]` 拒绝规则；零长度序列只允许由运行时 `cu_seqlens[s] == cu_seqlens[s+1]` 形成，
  由 V1 的 `zero_length` case 在 `gen_inputs()` 中构造，且整批 `total_tokens >= 1`。
- `cu_seqlens` 是一维整数 input；第一阶段使用 `i32`，并显式检查 `total_tokens <= INT32_MAX`，
  需要更大范围时另开 int64 全链路阶段。
- 长度为 `num_seqs + 1`，首项 0、单调不减、末项等于本次 `total_tokens`。
- 允许长度 0 的序列（整批至少一个 token）；其 output 行为空，`final_state` 必须等于该序列
  `initial_state`。只在 canonical benchmark 中是否出现零长度由 profile 决定，correctness 必须覆盖。
- harness 在 host/device 执行前校验这些不变量；reference 永远读取实际输入，不能读取实例里
  另一份边界副本来代替它。
- L2 的 `cu_seqlens` 按 max capacity 分配时，`num_seqs+1` 之后的未使用区域必须填声明的 poison
  sentinel（不是便利的 0），并额外覆盖 `cu_seqlens[num_seqs+1]`/预取下一项的越界负例；外层
  GuardedBuffer 不能替代逻辑 capacity 内部的尾部检查。
- 候选可以读取声明的 canonical profile 来选择调度策略，但不能假设 correctness 只会给它这一组
  边界。

### 2.3 L1 的 OpSpec 表达

保留 V2a 的 `sizes.<tier>.seq_lens` 作为 **canonical workload profile**，并补充：

- `case_domain` 声明 canonical/permuted/skewed/chunk-boundary/zero-length 等分段 case；
- `gen_inputs(case, seed, instance)` 可按 case 产生同 total/count 的其他合法分段；
- 派生 `total_tokens/num_seqs/NSP1` 仍是该档编译期常量，因此 L1 复用现有 buffer ABI；
- `reference` 按实际 `inputs["cu_seqlens"]` 逐段计算；
- statement 显示 canonical profile 供候选做调度选择，同时明确其他合法分段也会测试。

### 2.4 L2 的 logical / resolved / capacity 模型

统一的 Capacity + Resolved 概念模型中，L1 是 `resolved_total_tokens == capacity_total_tokens`
且 `resolved_num_seqs == capacity_num_seqs` 的特例；Dyn-0 不需要动态几何，Dyn-1 是可选优化。
L2 再放开 total/count，不再把 `total_tokens/num_seqs` 只当编译期常量：

- **capacity shape**：`max_total_tokens/max_seqs`，用于 buffer/workspace 分配和 launch envelope；
  该档的 `SemanticInstanceV2.cu_seqlens=None`，semantic identity 不再绑定某个 canonical 分段；
- **resolved shape**：本次从输入张量形状与 `cu_seqlens` 得到的实际 total/count，具体 benchmark
  分布另由 `workload_profile_digest` 标识；
- **logical shape**：reference 和结果比较使用的有效区域；
- `TensorBinding`、ABI span、Metal buffer、workspace span、正确性 shape 检查都明确各自使用哪一套；
- output 的 padding 区必须通过独立 guard/poison 规则检查，不能用“ragged 比较”掩盖越界写；
- **dispatch 主体仍是 capacity 实例**：resolved total/count 只进入 `evaluation_input_digest` 与
  `LaunchGeometry`/evaluation evidence，不直接进入现有有限域 guard。若未来要按 resolved 区间再分
  variant，必须另开区间 guard 与 overlap 分析，不在 V5 偷换域模型。
- **logical 比较清单**：`gen_inputs` 在 capacity 缓冲的未用区填 poison；reference 只计算 cu 覆盖的
  序列；比较 output 的 logical 前缀与 `final_state[:num_seqs]`，并断言未用 padding 仍为 poison。
  不接受 empty batch（total_tokens=0），只接受 empty sequence；性能仍按固定 resolved 形状的
  workload profile 逐档测量，只是它们共享 capacity ABI。

---

## 3. 后端中立 Host Program：受限 C99 / native64 bounded worker

当前权威细节见 `docs/架构/native-host-program-ABI.md`。本节只保留与 varlen 总图直接相关的合同。

### 3.1 交付形式与候选自由

允许且仅允许一个 host entry：`host/tiling.json`（既有 linear，语义不变）或 `host/tiling.c`
（受限 C99 Host Program）。`host/execution_geometry.json` 仍是唯一几何声明。C 源码属于候选，
候选自行选择循环、heap、二分、成本模型、LPT/range-cut 或其它 planner；harness 不内置 KDA
算法，也不引入按任务名查询的 tiling registry。

### 3.2 强类型 native64 ABI

候选入口保持：

```c
int32_t host_tiling(const SfHostInputs *inputs,
                    const SfHostShapes *shapes,
                    const SfHostTarget *target,
                    SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved);
```

`host_contract.h` 由 OpSpec/ABI 生成。C 编译器拥有结构体自然对齐和 64 位指针布局；Python 不计算
`sizeof/offsetof`，也不把五个不同子结构体都误当成同一个 ABI 地址。worker 调用无参数
`sf_native_invoke()`，由 C 内部传递五个子结构体。输入/workspace 地址、span、shape/target 和
metadata 都通过生成的 `sf_native_*` accessor 交互；这些内部入口不允许候选调用。

- 小张量直接映射到独立 native region，具有强类型指针和前后 guard；
- 大输入只有 `id/numel/elem_nbytes`，使用
  `sf_input_read_i32/u32/f32/u16(id, elem_off, (uintptr_t)dst, count)` 按需读取；
- fp16/bf16 继续是 `uint16_t` raw-bit ABI，不静默转换；
- 每个 workspace 只有一个连续有效 span，`offset/count` 以元素计并满足 capacity；
- host/device_zero/poison 三态互斥，只提交有效 span，未写尾部不按 max-capacity 全量初始化。

### 3.3 bounded worker 与二进制边界

候选 `.dylib` 只在 bounded v2 worker 内编译、加载和执行；控制机不 `CDLL`、不接收 native
指针。`tcenv host-probe` 也走独立 bounded probe worker。候选 segfault 返回 crash，runaway 由
外部 deadline 杀死；native 不是内存沙箱，不能声称 guard 取代进程隔离。

受限 C99 预检拒绝 libc/process/file/network、inline assembly、函数指针、宿主 import 和
`sf_native_*` 内部入口。macOS 的 dylib 虽需链接 `libSystem`，最终 `nm` 检查仍要求：

- 未解析宿主符号为零，防止绕开文本黑名单调用 ambient API；
- 导出仅为 harness 的 `sf_native_*` 函数，候选不能添加自定义逃逸导出；
- `size -m` 的 `__DATA` 静态存储不得超过合同 `memory_budget_bytes`，否则加载前拒绝。

### 3.4 调用、校验与原子提交

每次调用重设 span/guard、写入 shape/target、复制 fresh mapped inputs，再执行候选。只有返回
0 才进入收集；收集前重新核对所有 input/workspace pointer、capacity、elem_nbytes、shape、target、
metadata-only 输入、guard 和 read-only 内容。任何非 0、非法 span、合同改写、guard 命中、
crash/timeout 都丢弃全部 span/resolved scalar，不能部分写入真实 Metal workspace。

### 3.5 构建身份与证据

- `program_sha256`：候选 `host/tiling.c`；
- `assembled_host_sha256`：**实际完整 C 编译单元**，包含生成 ABI、候选和 candidate 后的
  worker 支持代码；
- `native_sha256`：实际 `.dylib` bytes；异进程重建必须逐字节一致；
- `execution_backend=native-c99`，compiler/platform/used workspaces/memory budget 进入 descriptor；
- `.dylib` 不进入 Schema-4 entries，也不偷塞进 `BuildOutcome.payloads`。

Metal HostRunEvidence 为 `tcenv.hostprogram.run.2`，记录 `native_module_digest`、
`native_wall_ns`、实际 reads/spans、输入依赖、E2E 计时和 typed error。Host 工作始终
`input_dependent=true + preprocesses_inputs=true + inside_timing=true`，不能学 KDA 把 prepare
放到计时外。device-only 时间只作诊断，发布仍使用原有 fresh confirmation 和 E2E 谓词。

### 3.6 Metal 与 NPU 的一致性

Metal 与 NPU simulator 复用同一 native C99 语义：同 target 比较全部 span/reads/resolved，
异 target 比较 item 全集、唯一认领和语义依赖边，允许 core/grid 不同。NPU 继续
`simulated=true/publication_allowed=false`，不构成真机证据。

当前 CPU/协议/NPU 模拟与真实 Metal 复验均已通过。pack closure 为 **73**（`native_abi.py` 与
V5 的 `taskcontract/runtime_shape.py`），P2 最终重录 deferred，所有 task pack digest 前进。旧 Wasm
runtime/toolchain/probe、Wasm32 renderer 和 `wasmtime` 依赖已删除；后续 V3 标题下的 Wasm 内容
仅为历史验收记录。

---

## 4. Device 与执行几何能力

### 4.1 Dyn-0 默认、Dyn-1 可选：不要让所有候选先过动态几何

现有 `resolve_geometry()` 在构建前没有输入时就把 grid/active/scalars 求成静态整数，
`MetalV2Runtime._bind()` 只绑定 active stages，`selection_parity()` 又要求实际 dispatch 与静态
active set 逐字相等。动态 grid/stage skip 确实需要新机制，但它不是 varlen 正确性的前置条件。

#### Dyn-0：静态几何 + 运行时数据（默认）

- grid 和 stage dispatch 保持静态，现有编译、绑定和静态 parity 完全不变；
- 多余 TG 读 `item_count` 后立即返回；无需 handoff 的 case 中，对应 stage 照常 dispatch 但读 scalar/
  workspace 门后空转；
- device 可直接从 workspace 读 item count/flags，不必引入 runtime scalar binding；
- 先用它完成 arbitrary 分段、partial tail、逐段 final_state 和多 stage handoff 的正确性证据，
  不依赖 Wasm/lld，也不改变现有 `selection_parity`。

#### Dyn-1：`ResolvedEnvelope` + 逐 case `LaunchGeometry`（可选优化）

只有在真实规模测得空 TG/空 stage 开销值得优化时才启用，且 geometry 显式声明 `dyn_level=1`：

- 构建期对全部 `possible_stages` 做名称、资源、`max_grid`、runtime-scalar 合法域和 `when`
  校验并进入 artifact-set digest，全部完成编译与 `bind_stage_arguments`；possible coverage 不替代
  每次 launch 的 output coverage。
- 新增独立的 `runtime_scalars: [{name, dtype, min, max}]`，**不与构建期静态 `derived_scalars`
  混用**；stage 声明使用哪些 runtime scalar，绑定时按名字核对 kernel `constant int/uint` 参数。
- 动态只允许 grid 与 stage-active；`threads/threadgroup_bytes` 始终静态。每 stage 显式给出
  `max_grid`，运行期只检查 `resolved_grid <= max_grid`，不做含动态表达式区间证明。
- 每次 Host Program 后由 `ResolvedEnvelope + HostResolved + resolved shape` 得到 `LaunchGeometry`，
  检查 result-tensor 写入位掩码，未激活 stage 零 dispatch；预构建 BufferArg 槽位，仅更新 runtime
  scalar bytes。
- 逐 case/trial 使用新 parity：
  `observed_dispatch == evaluate_launch(envelope, host_run.resolved)`，并再次检查 envelope 上限。
  旧静态 parity 行为不变，provider 错误、inactive stage 执行、漏 stage 或 grid 漂移都是 mismatch。

Dyn-0 与 Dyn-1 长期共存，由候选在 geometry 中声明；只有测量证明空 TG/空 stage 成本显著时才做
Dyn-1，阈值在真实规模测量后冻结，不预先宣称“个位数百分比”或固定 5%。L2 若采用 Dyn-1，只把
resolved total/count 放开到 capacity 以下，复用同一套两层几何与证据。

### 4.2 工作项与存储层级

合同不规定 item 的位布局，但候选必须能表达：

- token offset、valid rows、sequence/head、first/last、handoff source/destination；
- per-core item count 与实际 work-list digest；
- 小型 item table 复制到固定容量的 threadgroup/UB/L1，大表留 global workspace；
- 固定容量 scratch + 运行时 count，而不是未受控的动态 threadgroup 参数；
- chunk 内累计 log-decay、partial tail mask 和谓词化输出。

**Workspace 必须在 OpSpec 中给出容量公式，不能到 V3 才临时发现没有资源**：

- `work_items <= num_ctas * max_items_per_cta * 2`（u32，双 word item）；
- `item_counts = num_ctas`（i32），`handoff_flags >= num_ctas+1`（i32）；
- `handoff_slots <= min(实际切分边界数, MAX_HANDOFF_SLOTS)`，`handoff_bytes = slots*D*D*dtype_size`；
- 所有 capacity 进入 ABI 和证据，Host Program 只写有效 span。`V2ArgSet` 按 possible stages 的
  workspace 并集一次性分配，完全不可能使用的 workspace 不分配；逐 case active set 只决定 dispatch，
  绝不能在计时中临时分配。
- 现有 `log_decay_tab` 在 T=8192、H=96、D=128 时约 384 MiB，只能作为“候选选择使用后”的
  scratch，不能因为 OpSpec 声明就无条件分配/清零；work_items/handoff 也必须按上面的显式公式预算，
  不得在真实规模默认再复制一份同量级中间表。

**V2 Dyn-0 冻结的容量公式与分配范围**（`tasks/varlen_kda/opspec.json`，`MAX_SPLIT_PER_CHAIN = 3`）：

| workspace | dtype / dims | 容量公式 | 用途 |
| --- | --- | --- | --- |
| `work_items` | u32 `[num_ctas, 2]` | `num_ctas = num_seqs*H*MAX_SPLIT_PER_CHAIN` | 双 word 工作项（边界与位布局由候选决定） |
| `item_counts` | i32 `[num_ctas]` | 同上 | 运行时 item count；多余 TG 读它后退出 |
| `handoff_flags` | i32 `[num_seqs*H*4]` | `>= num_ctas + 1` | ready/consumed flag（取值由候选决定） |
| `handoff_states` | fp32 `[slots, D, D]` | `slots = num_seqs*H*(MAX_SPLIT_PER_CHAIN-1)`，`handoff_bytes = slots*D*D*4` | 首/中/末 piece 之间的终态 |
| `token_stats` | fp32 `[total_tokens, H, 3]` | `total_tokens*H*3*4` bytes | 预处理表：`|q|^2`、`|k|^2`、`sigmoid(beta)` |
| `log_decay_tab` | fp32 `[total_tokens, H, D]` | `total_tokens*H*D*4` bytes | 可选 log-decay 表（384 MiB @ V6 规模） |

- `MAX_SPLIT_PER_CHAIN = 3` 是**声明的存储上界**，不是 harness 规定的切分算法：候选可以在容量内
  选择任意 piece 划分或完全不切分。一个按 64-token chunk 存 D×D 的实现会超出声明的 handoff 容量
  并在 `workspace_size_problems`/ABI 处被拒绝，而不是悄悄吃掉数百 MB。
- 分配范围是**全部 possible stages（含 `when` 为假的 stage）的并集**，在任何 dispatch 之前算好。
  完全没有 stage 绑定的 workspace 不分配、其正确性 fill 也跳过（naive 候选不再为 384 MiB 的
  `log_decay_tab` 付费）；同一 workspace 的 zero 策略跨 possible stages 必须唯一。逐 case 的 active
  set 只决定 dispatch，绝不在计时中触发分配。

### 4.3 状态 handoff 与同步：Multi-stage 优先

允许候选选择：

1. 单 CTA/单核处理整条 `(sequence, head)`；
2. persistent work list，整链 LPT；
3. **长链拆分 + 多 stage/two-launch handoff（Metal 首选）**：stage 之间用 Metal serial encoder 与
   `memoryBarrierWithScope_(MTLBarrierScopeBuffers)` 建立真实完成顺序，并区分两种形态：
   - **预处理 + 顺序链（正确性基线）**：stage 1 计算 per-token norm/log-decay/beta 表，stage 2
     每个 `(seq, head)` 一个 TG 顺序递推。它证明多阶段预处理与分段语义，**不冒充跨 piece handoff**。
   - **真实 piece handoff**：stage 0 规划 piece，stage 1/2/3 分别处理首/中/末 piece；中/末 piece 的
     初值来自 `handoff_states`，由 `handoff_flags` 的 ready/consumed 协议管理，末 piece 写
     output/final_state。每条链的切分边界数受 `MAX_SPLIT_PER_CHAIN` 限制，handoff slot 数与字节
     按 §4.2 公式进入 capacity；不允许无上限地为每个 chunk 存一份 `D×D` state。
4. 无跨 TG 依赖的树状规约或原子累加；
5. 纯 device scheduler 或其他算法。

**Metal 不把单 Launch 内跨 threadgroup 自旋等待作为通用或主验收路径**。Apple GPU 没有可移植的
CUDA 式跨 TG 前向进度与物理驻留保证，consumer 可能占住硬件而 producer 尚未调度。外部 watchdog
只能终止 hang，不能证明安全。若未来确需保留该实验路径，必须针对具体 target 另有前向进度证据、
consumer-first/驻留压力测试和独立发布授权；默认候选与 fallback 不使用它。

NPU 的 CrossCore 路径也必须由 target 的同步语义与对抗性调度模拟证明，不能把 CUDA 的 acquire/
release 自旋直接映射过去。所有路径中，`final_state` 始终以合同 dtype 和逐段终值验收；中间
handoff 若使用 bf16/fp16，只能作为候选数值路径，由现有 tolerances + scaled_error 判定。

---

## 5. 正确性、作弊负例与性能口径

### 5.1 必测分段 case

每个 varlen 任务至少覆盖：

- `canonical`：声明的 fixed/mixed/uniform 分布；
- `permuted`：同 total/count，长度顺序改变；
- `skewed`：同 total/count，一条长序列 + 多个极短序列；
- `all_unaligned_primes`：全质数短序列（1、3、5、7、13、17、67、131…），连续多个
  partial tail 紧邻，检查指针偏移、log-decay 重置和 mask 串位；
- `max_handoff_stress`：一条长度为 `T-(num_seqs-1)`，其余全为 1，同时触发超长链切分与
  极短碎片聚集，覆盖 LPT/range-cut 切换和最坏 handoff；
- `chunk_boundary`：1、BT−1、BT、BT+1、2BT±1，覆盖 partial tail；
- `zero_length`：空序列 output 为空且终态等于初态；
- `single_sequence`：单段全长退化——单段档就是它本身；多段档用“一条非空序列承载整批 + 其余
  空序列”实现，保持该档 total/count；
- `fresh_distribution`：fresh seed 派生的合法分段，与 tune 不相交。

每个 case 在报告里带 realization（`exact/partial/not_constructible/deviates/unclassified`，
含实际 segments 与未实现项）：该档 total/count 放不下的形状（单段档的多段 case、均匀档的
`permuted` 顺序、不存在的全素数分段）记为 `partial`/`not_constructible`；本可实现却没实现的
布局（非均匀 canonical 的假 `permuted`、一个申报 BT 目标都没出现的 `chunk_boundary`）记为
`deviates`。`partial` 的 `missing` 只表示**本次分段未实现**，不等于该档无法承载（同一
total/count 的另一合法分段可能实现它）。以上都**不计作该 case 的完整覆盖**。

`initial_state` 必须逐序列不同且非零，能识别状态串段。reference 始终按实际 `cu_seqlens` 独立
分段计算，output 与 final_state 都比较。

### 5.2 必须拒绝的候选

- 把 canonical 边界写死，permuted/fresh case 应失败；
- 完全忽略 `cu_seqlens`，把整包当一条序列；
- 丢弃 `initial_state` 或从零状态续传；
- output 正确但 `final_state` 错（独立负例，证明终态真的被比较）；
- head piece 写 handoff 后，continuation 从 initial/zero 开始；
- handoff flag 未清/旧 flag 导致读取上一轮 state；
- partial tail 的无效行改变 state 或写 output；
- 空序列错误推进状态；
- 跨序列/跨 head 共用 state；
- 依赖 workspace 初始零但未声明 zero，poison trial 应失败；
- host 写越界、读未声明输入、resolved grid 超 envelope、input-dependent host work 被移到计时外；
- 越界预取 `cu_seqlens[num_seqs+1]` 或读取 L2 capacity 尾部 sentinel；
- 在有限 core 槽位下 consumer-first 形成等待环，或 NPU 非对齐 DMA 写回踩到下一条序列。

**V2 Dyn-0 的实际覆盖**（`tests/backends/metal/test_varlen_gpu.py` 的 V1 负例保留不动，
`test_varlen_dyn0_gpu.py` 补新增项）：

- Metal 真机：空序列推进状态、partial tail 上取整读进下一条序列（两个 fixture 各一）、跨 head 共用
  state、work list 写出声明容量（guard 判 unsafe）、consumer-before-producer（两个 fixture）、
  continuation 从 initial_state / 零状态开始、producer 不发布、consumer 不清 flag、跨链共用 handoff
  slot、依赖未初始化的 flag（poison trial），以及 V1 的写死 canonical/丢 `initial_state`/
  `final_state` 错。
- NPU 模拟：有限 core + consumer-first 等待环、非对齐 DMA 踩踏、scratch 超预算全部由模拟器显式
  报缺陷；它们不是 Metal 真机证据。
- **V3 补上的 Host Program 负例（真实 Metal）**：host 返回非 0（incorrect，无 partial span）、
  span 超 capacity（incorrect）、plan 不覆盖分段（incorrect/unsafe）、引用了没有 stage 绑定的
  workspace（envelope 拒绝）、host 写与 device_zero 同块（envelope 拒绝）、危险 `#include`
  （投影期预检拒绝）、input-dependent host work 谎报在计时外（build 拒绝，report 为
  `compile_error`）。这些在 `tests/backends/metal/test_varlen_hostprogram_gpu.py`。
- **仍未在本阶段构造**：L2 capacity 尾部 sentinel 与 `cu_seqlens[num_seqs+1]` 越界预取（需要 V5 的
  capacity 缓冲与 poison 尾部；L1 的 `cu_seqlens` 是普通输入，越界读落在 guard 区、不能作为
  “读到声明外数据”的测试）；“resolved grid 超 envelope”属于 Dyn-1（V4），Dyn-0 由静态 launch
  解析与 ABI 上限拒绝代替，已在 `tests/backends/metal/test_varlen_workspace.py` 钉住“没有
  runtime_scalars/LaunchGeometry”。

### 5.3 性能与发布

- fixed/mixed/uniform 各自独立计时、独立通过 1.10 gate；不能用 geomean 隐藏某一档回退。
- candidate 时间包含输入驱动 Host Program、有效 span 写入、声明的 workspace 清零、所有 stage 和
  handoff；baseline 必须使用相同输入、output/final_state 责任和测量边界。
- 同时记录 `host_tiling_wall_ns`、`device_kernel_ns`、`end_to_end_wall_ns` 与 `timing_mode` 供诊断，
  **发布仍只用同一 E2E wall 与 unique publication predicate**；device-only 达标不能掩盖 E2E 回退。
  host-overhead 预算只能作为目标策略中的诊断/反馈，不能变成旁路发布门。
- serial 与未来的 ring-buffer overlap 是不同测量模式，必须预先声明、分别出证据，candidate/baseline
  使用同一模式；不能在看到结果后选择更有利的一种。
- `specialized_hit_speedup` 只统计生成 variant；`effective_end_to_end_speedup` 包含真实 fallback。
- KDA 的 prepare/launch 数字只能作算法参考，不能与我们的 end-to-end 数字直接比较。
- naive `float state[D*D]` 单线程 kernel 仅是正确性参考；真实性能验收使用 D=128、H=64/96 和
  KDA fixed/mixed/uniform 规模，不能由小档正确性推断可扩展性。
- NPU 模拟只证明协议/数据流/调度身份，不产生 NPU 时间、speedup、confirmed、catalog 或发布结论。

---

## 6. Metal 落地方案

### 6.1 运行时接线

- `tcenv/backends/metal/v2_runtime.py`：Wasm module 执行、声明输入视图、workspace 有效 span 写、
  host run evidence、resolved scalar/stage-active 绑定。
- `tcenv/taskcontract/portfolio_runner.py`：由 plan 是否读输入派生 lifecycle，不再恒写
  `input_dependent=false`；输入驱动计划强制进入计时。
- `V2ArgSet`：只读输入快照、workspace capacity/valid span 与实际写 digest；L2 的 `cu_seqlens`
  未用尾部填 poison sentinel；继续检查输入不可变和 guard。
- Metal stage binding：Dyn-0 保持现有静态 active-stage 绑定；Dyn-1 才对 envelope 的全部
  `possible_stages` 一次编译/反射绑定。静态 threadgroup 容量 + 运行时 item count；runtime scalar
  只走按名绑定的 4/8 字节整数参数，不绑 host pointer；动态 `when` 决定每次 case 的 stage-active
  集合并逐 case 做 output coverage。
- Metal 同步：stage 之间使用现有 serial encoder + `memoryBarrierWithScope_(MTLBarrierScopeBuffers)`；
  不以跨 TG 自旋等待作为验收或默认实现。
- 计时实现：旋转集参数槽位预构建，只读输入只恢复一次；Host-written 与 device-zeroed workspace
  分开，避免 zero-fill 双 CommandBuffer 和每轮重复构造 `stage_args`。

### 6.2 Metal 垂直切片候选

保留 V2a naive `known_good.metal` 作为最简正确性候选，并按 Dyn 级别提供两类**候选拥有的** fixture：

**Dyn-0（先做，不依赖 Wasm/lld）——已实现并验收（见 §8 V2）**：

- 静态 grid/stage/parity；device 按 `cu_seqlens` 建 work list，多余 TG/stage 读 count 后退出；
- stage 1 预处理 per-token norm/log-decay/beta，stage 2 每个 `(seq, head)` 一个 TG 顺序递推，证明
  多阶段预处理、partial tail 和逐段 final_state；
- 另有一个真实 piece-handoff fixture：stage 1 存局部终态，stage 2 顺序消费，
  `MAX_SPLIT_PER_CHAIN` 和 handoff bytes 符合 §4.2 公式，不得依赖同 launch 自旋；
- 小 item table 的 threadgroup 缓存与超大表 global fallback；
- 与 naive oracle 的 output/final_state 差分，以及 §5.2 全部负例。

两类 fixture 是 `tests/support/golden/metal/varlen/` 下**候选拥有**的固定源码，不是 harness 规则：

| fixture | 形态 | 证明什么 | 不声称什么 |
| --- | --- | --- | --- |
| `prep_chain.metal` | stage 1 预处理 + 建 work list；stage 2 每 `(seq,head)` 一个 TG 顺序递推 | 多阶段预处理、partial tail、逐段终态、work list、threadgroup 缓存/global 回退 | **不是**跨 piece handoff：整条链始终在同一个 TG 里 |
| `piece_handoff.metal` | stage 0 规划 piece；stage 1/2/3 处理首/中/末 piece；`handoff_states` + ready/consumed flags | 真实跨 stage 状态传递、首/中/末 piece 的 state 来源、flag 发布/清除与后置条件、有界 handoff 存储 | **不声称**这是 KDA 的最优切分，也不声称性能 |

两者都只用 OpSpec 声明的 workspace；fixture B 不绑定 `log_decay_tab`/`token_stats`，所以证据里这几个
workspace 不会出现在 `workspace_bytes`（分配范围是 possible-stage 并集，见 §4.2）。

**C99/Wasm Host Program（随后）**：

- host planner 覆盖 range-cut、LPT 与 handoff cost，复用 Dyn-0 的 device kernel和静态 parity；
- 只有 Dyn-1 启用时，才增加 resolved scalar 控制活跃 core/handoff stage 的正例，以及未激活 stage
  不得执行的负例。

这些 fixture 证明机制可表达，不是 harness 为所有 varlen 任务指定的标准算法；预处理 + 顺序链和
真实 piece handoff 不能互相冒充。

---

## 7. NPU 模拟落地方案

### 7.1 复用，而不是重新写一套

NPU simulator 使用相同的：

- OpSpec/WorkloadProfile/case 生成与 fp64 reference；
- 同一份受限 C99/Wasm Host Program 与输入/Workspace 校验；
- work-list、item count、resolved scalar、resolved launch 和 evidence 身份；
- LPT/range-cut/handoff/tail/zero-length 负例。

模拟器将 work items 映射到模拟 core，把固定容量 threadgroup scratch 映射为模拟 UB/L1，把 flag
状态转移和 handoff 依赖映射为显式事件图。可以用后端中立的 Python 参考执行器验证调度结果，但
**不能执行 MSL 来声称 NPU kernel 正确**。

V2 已实现的模拟器在 `tests/support/npu_sim/varlen_worklist.py`（仅测试组合根，生产包不 import），
验收在 `tests/shared/test_varlen_npu_sim.py`；V3 的同一 module parity 在
`tests/shared/test_hostprogram_npu_parity.py`。V2 的 parity 是两条**独立** Python 路径之间的差分：`metal_device_plan()`（`piece_handoff.metal` 规划 stage 的逐行转写）与
`npu_host_plan()`（前缀和 + 累加 piece 大小的另一份实现），分别由 `metal_adapter()`（静态
grid、一项一个 TG 槽位）与 `npu_adapter()`（host-neutral 规划 + 有限 core 调度）输出。同一个
`HostTarget` 桩下要求两者的 **work-list bytes 与依赖边**一致，并且**两侧各自**满足“每项恰好被
认领一次”；**core 分配（claim map）与 launch 几何不比较、也不要求相等**——同一组 target 参数下
Metal 的静态 grid 槽位数与桩的 core 数本来就不同（实测 18 vs 64），槽位到 item 的映射也不同。
parity 本身另有负对照（字节对调、单 bit 翻转、丢边、换 max_split、重复认领、有项未认领）证明它
真的会抓到漂移，而不是比较同一个函数。**V3 已完成真正的 Wasm module parity**：同一 build 的
module 在 `METAL_STUB`/`NPU_STUB` 两种 HostTarget 视图下输出逐字节相同，且与两个 adapter 的
work-list bytes 一致；换成 18-core Metal-like 与 64-core NPU-like 真实 target 后，item 全集、
“每项恰好一个 core 认领”和依赖边仍等价，claim map/grid 不作比较。该验证只证明协议/数据流身份，
不产生 NPU 时间或发布结论。

事件图不能只做对理想拓扑排序有利的执行：

- **Adversarial Core Scheduler**：在有限 core 槽位中故意优先调度带 `wait_flag` 的 consumer，注入
  consumer-first、producer 反序和驻满等待者等排列，检测等待环、死锁和活锁；只证明“顺序执行恰好
  能过”不算通过。
- **Alignment Clobber Check**：按 target 声明的 DMA 对齐粒度（例如 32B/512B，不把一个数字当成所有
  NPU 的通用事实）模拟写回。非对齐 partial tail 若未声明 masked write 或 read-modify-write 策略，
  必须判定可能覆盖下一条 TND 序列的开头；不能用 GPU 的逐线程 predicate 语义代替 NPU DMA 语义。

**Parity 分两层**：V2 已实现第一层——同一个 `HostTarget` 桩输入时，两条独立 planner
（`metal_device_plan`/`npu_host_plan`）给出的 **work-list bytes 与依赖边**必须一致，且两侧**各自**
每项恰好被认领一次；core 分配（claim map）与 launch 几何属于 target-specific，不在比较范围内，
也不要求相等。**V3 已加 Wasm module 层**（`tests/shared/test_hostprogram_npu_parity.py`）——同一
module 在两个 adapter 上的有效 span 与 resolved scalar 逐字节一致。使用各自真实 target 核数时同样
只要求 item 全集相同、每项恰好被一个 core 认领、依赖边等价，不要求 `num_ctas`、grid 或 core 分配
相同。

### 7.2 证据与能力边界

所有 NPU 报告固定：

```json
{
  "simulated": true,
  "publication_allowed": false,
  "measurement_kind": "simulated",
  "evidence_scope": "host-plan/work-list/state-protocol only"
}
```

- 不生成 NPU latency/speedup，不进入 fresh confirmation、发布、catalog/brief。
- 清微真实 target/toolchain/ABI/profiler 未接入，NPU-P0 继续 blocked。
- 昇腾 CAKE2/CANNBot 只提供设计参考；若未来需要真实 Ascend backend，必须另开阶段，提供真实
  target、设备、编译器、ABI、计时和独立验收，不在本计划中预先宣称。

---

## 8. 分阶段施工与验收

### V0：合同纠偏与兼容基线（先做，CPU）

**状态：实现、CPU 回归与后续 P2 重录均已完成。** 本阶段先独立落地，未与 V1 的 case 生成/
输入 digest 混为一次验收。

修复 V2a 审核发现的问题：

1. `_validate_size` 是 v1/v2 共用，不能让 v1 接受 `seq_lens`；派生符号豁免只对 v2 varlen 档生效，
   普通固定档缺符号必须在校验期报 `OpSpecError`，不能延后成 `KeyError`。
2. varlen 题必须声明符合合同的 `cu_seqlens` 输入；派生符号仍需声明并满足 min；检查 i32 溢出；
   同一题的 fixed/varlen 档混用必须明确定义，否则拒绝。
3. statement 不再对没有 `initial_state/final_state` 的 varlen_scale 打印不存在的状态合同。
4. 指纹测试用同一 `opspec_digest` 隔离 `cu_seqlens` 字段，而不是只靠不同 opspec 文档间接通过。
5. reference 文档不再把 token-by-token oracle 称为 chunked，也不再把 tensor input `scale` 说成 attr。

验收：v1 归档拒绝 `seq_lens`；固定档身份不变；非法 varlen ABI 全部 typed refusal；
`tests/shared/test_opspec_{v2,varlen}.py`、文档完整性通过。

### V1：L1 任意合法分段与三层身份（CPU + GPU 验收另授权）

**状态：已完成（CPU + 真实 Metal）。** 主会话独立实跑 V1 定向 CPU **469 passed**、
Metal GPU **11/11 passed**，必要非 GPU 回归 **68 passed**。P2 证据已按新的 task-pack digest
重录：14 个域实例、0 个 timing-blocked，P2 三文件 **36 passed**。flow 的最终 exit 1 是
reviewer 多输出 `_warning:null` 的 schema 收尾错误，不是实现失败。

`code_import_closure` 因本阶段新增 `tcenv/taskcontract/varlen.py` 并改动
`reference.py`/`contract_errors.py`/`opspec/v2.py`/`evaluate.py` 前进；P2 已按约定重录，
没有放宽或伪造证据。`tcenv/taskcontract/varlen_cases.py`（realization 分类）属于 backend/证据面，
不在 pack closure 内；pack 侧的运行时校验仍只有 `varlen.py`。

落地要点（与下文逐条对应）：

- case 生成在两道题自己的 `reference.py::gen_inputs`（reference 边界禁止 import 仓库/相对兄弟，
  `tasks/` 级 helper 也进不了 pack，因此两个 planner 独立实现；共享合同由
  `tests/shared/test_varlen_cases.py` 对两题跑同一组断言）。`varlen_scale` 增加 golden-only 的
  `single` 档，使 `single_sequence` 在该题上真的可构造；多段档的 `single_sequence` 用“一条非空
  序列 + 若干空序列”实现（不改 total/count，空序列是合法运行时事实），不是 canonical 重放。
- **可审计覆盖口径**：harness 的 `tcenv/taskcontract/varlen_cases.py` 从实际分段独立判断每个 case
  实现了什么，闭集为 `exact/partial/not_constructible/deviates/unclassified`，并记录实际
  `segments`、已实现/缺失的性质面。单段档只有一种合法分段，其 7 个多段 case 记为
  `not_constructible`；小档放不下整组 BT 目标时 `chunk_boundary` 记为 `partial` 并逐条列出本次
  未实现的 BT 组合（`missing` 不等于该档不可承载：例如 total=16/count=2 的 `(7,9)` 同样合法）；
  BT 目标按**实际分段全集**核算（含末段余量），末段恰好是 `BT`/`BT+1`/`2BT+1` 时它就是已实现的
  目标，六项目标全部出现才记 `exact`；一个目标都没出现的布局（如 `(2,2)`，同档 `(1,3)` 可实现）
  记 `deviates` 而不是 `not_constructible`。`total_tokens === num_seqs` 的全 1 档没有
  第二种正长度分段，`permuted`/`fresh`/`skewed`/`primes`/`handoff` 记为 `not_constructible`
  而不是 `deviates`（`chunk` 例外：分段里确实包含目标 1，记 `partial`）。
  `total_tokens === 2 * num_seqs` 时长段与 2-token 碎片同长（`[2,2]`/`[2,2,2]`），不构成偏斜：
  planner 构造 `[3,1]`/`[4,1,1]` 这类真实偏斜并记 `partial`（缺 2-token 碎片），只有长段严格长于
  2 才能 `exact`；handoff 也只有在存在长于 1-token 碎片的链时才是 `exact`。不存在的全素数分段由
  planner 退回该档 public 分布（不裸中止整次 correctness，也不用非质数冒充），再由 harness 独立
  搜索判定 `not_constructible`。每个 case 的 realization 进 correctness 报告，聚合在
  `correctness.coverage.case_coverage`；`partial`/`not_constructible` **不计作该 case 的完整覆盖**。
- 运行时边界校验归 harness：`tcenv/taskcontract/varlen.py` 的首 0、单调不减、末 total、长度、i32、
  total>=1，经 `ReferenceModule.generate/.execute` 同一漏斗；任务的 generator 不能自证。`offsets`
  对二维/嵌套/NaN/Inf 等非法边界总是返回“不可读”（不抛裸异常），且 cu 校验在通用 mapping 校验
  **之前**执行，所以长度/rank/dtype 错误都是 `varlen_contract_invalid`，其他张量的闭集校验不变。
  静态 `seq_lens` 每段 >= 1 仍由 `_validate_size` 负责，零长度只来自运行时 cu。
- 三层身份：`SemanticInstanceV2.cu_seqlens` 仍是 canonical 分布并留在 fingerprint（不新增 domain
  point）；`portfolio/domain.py::evaluation_input_digest` 绑定 case + seed + 实际 cu 字节 +
  输入张量身份，并写进 correctness 的 `input_sha256` 供重放。固定 shape 实例该字段保持空串。
- varlen 题的 `cases.names` 必须覆盖 §5.1 的 case 域且 `canonical` 排第一（L1 的 benchmark 用第一个
  case；L2 的 benchmark 按声明顺序轮转**多个** resolved 批次，见 §8 V5，绝不只跑第一个 case 的满
  capacity 点）；这是 harness 对下限的校验，题目可以追加 case。

- 实现 §2.1–2.3 与 §5.1/5.2 的 correctness case；
- canonical profile 用于 benchmark，permuted/skewed/fresh 用于正确性；
- L1 保留 canonical `cu_seqlens` 在 SemanticInstanceV2/fingerprint/SemanticDomainPlan 中，避免同
  total/count 的分布撞身份；permuted/skewed/fresh 只作为 Evaluation Cases，由 input digest 区分；
  L2 capacity 档才将实例 `cu_seqlens` 置 `None` 并启用独立 workload profile；固定 shape 投影不变；
- statement 展示 canonical profile，同时声明其他分段会测试；
- output-only final_state 负例、hardcoded-canonical 负例进入 Metal GPU 测试。

验收：同一 ABI 下换分段仍正确；写死边界失败；每个实际 case 都能由 input digest 重放。
V0/V1 涉及 pack closure 的修改集中完成后，**只重录一次**受影响 P2 证据。

### V2：Dyn-0 Metal + NPU 语义垂直切片（GPU 验收另授权；无 Wasm 依赖）

**状态：已实现并通过 CPU + 真实 Metal 验收（机制正确，不声称 KDA 性能）。**

- 静态 grid/stage/parity；device 按 `cu_seqlens` 建 work list或 per-seq 静态映射，多余 TG/stage 读
  count 后退出，不做动态 binding、不引入 `LaunchGeometry`。
- OpSpec 明确 work_items/item_counts/handoff_slots 等容量公式；按 possible stages 的并集分配，未使用
  workspace 不分配，active set 不在计时中触发分配。
- 完成“预处理 + 顺序链”和“真实 piece handoff”两类 fixture；后者以 `MAX_SPLIT_PER_CHAIN` 和
  handoff_bytes 公式限制存储，不得用数百 MB 的每 chunk `D×D` state 冒充可落地实现。
- partial tail、per-sequence final_state、threadgroup/global item table、§5.2 负例全部通过；stage 间
  使用真实 buffer barrier，不依赖跨 TG 自旋。
- NPU simulator 同步验证 core 事件、consumer-first 等待环和 DMA 对齐踩踏，全部
  `simulated=true/publication_allowed=false`。

验收：Dyn-0 在 Metal/NPU simulator 上证明任意分段与多阶段状态协议，旧静态 parity 不变；只标
“机制正确”，不标达到 KDA 性能。

落地与验收记录：

- 容量公式：`tasks/varlen_kda/opspec.json` 新增 `token_stats`/`work_items`/`item_counts`/
  `handoff_flags`/`handoff_states`，`MAX_SPLIT_PER_CHAIN = 3`（§4.2 表格）；`log_decay_tab` 保持声明
  但改为按绑定分配。ABI 从 13 个张量增到 18 个，仍在 `apple-m4-max` 的 31 个 buffer slot 内。
- 分配范围：`possible_workspace_uses()`（`tcenv/backends/metal/v2_runtime.py`）取全部 stage（含
  `when` 为假者）的并集，在任何 dispatch 前分配；无 stage 绑定的 workspace 不分配且跳过 fill。
- 错误报告：候选留下 poison（NaN/Inf）时，最坏元素诊断使用 `"inf"`/`"-inf"`/`"nan"` 拼写，
  使报告仍是可指纹的评测身份，判 `incorrect` 而不是让 `IdentityError` 崩掉 evaluator。
- Metal fixture：`tests/support/golden/metal/varlen/{prep_chain,piece_handoff}.metal`，由
  `tests/backends/metal/test_varlen_dyn0_gpu.py` 在真实设备上跑正例与 §5.2 负例；fixture A 的
  global item-table 回退用 80 条链的探针档证明（> 64 项缓存）。
- NPU 模拟：`tests/support/npu_sim/varlen_worklist.py` + `tests/shared/test_varlen_npu_sim.py`，
  覆盖 core 认领、UB/L1 预算、handoff 事件图、consumer-first 死锁、DMA 对齐踩踏与两层 parity。
- 本阶段**没有**引入 Wasm Host Program、runtime scalar/`LaunchGeometry`、Dyn-1、L2 capacity 或性能
  数字；V3–V6 仍未完成。
- 收口补记（Dyn-0 机制证据，均为真实 Metal + CPU 实测）：
  - **均衡切分**：`piece_handoff.metal` 与 NPU 规划都用 `base, rem = divmod(len, P)`，前 `rem` 段
    各多 1 个 token。`len=4` 的链是 `2/1/1`，不再是 `ceil` 公式下的 `2/2/0`——空 piece 没有
    DMA span，不该被当成“对齐踩踏”或被写成零长度写回。设备写出的 work list 与 NPU adapter 的
    字节逐字一致（`test_the_device_work_list_matches_the_npu_adapter_byte_for_byte`：同一份
    `varlen_plan` 在 GPU 上跑、读回 buffer，再与 `npu_adapter` 的规划差分）。
  - **stale READY + 旧 handoff state**：首 piece 在 slot 1 留下上一轮的 READY 与
    `initial_state + 0.5`，中 piece 本轮不发布，末 piece 通过 READY 读到旧 state。三个控制实验钉住
    归因：旧值存在但本轮正常发布时必须 `ok`；负例必须是有限中等量级误差（实测 scaled_err≈468，
    `nan_count=0`）而不是协议 poison 路径（不发布 flag 的负例实测 `scaled_err="inf"`）；只改旧值
    （0.5→0.25）误差随之变（≈331），证明末 piece 真的读了这份旧字节。构造不新增 ABI 绑定，也不
    依赖 harness 的 workspace 清零策略。
  - **native descriptor 去重（已实施）**：原实现把 kernel 0 的 bindings 同时写进顶层
    `bindings` 与 `kernels[0].bindings`，四阶段 fixture 因此达到 7980 B / 8192 B。现改为
    顶层保存 kernel 0 的完整 bindings，`kernels[0]` 只保留名称与资源字段，后续 stage 仍各自
    保存 bindings；**不删 ABI 信息、不扩大上限、不做旧 digest 兼容**。同一 fixture 实测降为
    **7209 B**，native digest 与受影响报告/证据按新结构重新生成。`source_builder.py` 与
    `test_build_path_gpu.py`、`test_varlen_dyn0_gpu.py` 固定了这一结构与实测体积。
- 证据边界：`code_import_closure` 未变（V2 只改了 `tasks/varlen_kda/opspec.json` 与不在闭包内的
  `tcenv/backends/metal/v2_runtime.py`、tests）；P2 归档确认证据（14 个域实例、0 timing-blocked、
  activation generation=1）用确定性重放复核仍通过。P2 的 `verify_baseline` 所钉的 45 文件 start
  surface 在 V2 之前（HEAD=0557ef3）就已漂移 4 个文件 + 活计划，属既有状态，需要按安静窗口流程重录；
  V2 没有新增该 surface 的漂移。varlen task pack digest 因 opspec 变更而前进（varlen 评测身份随之
  更新），没有改写任何已归档证据。

### V3：受限 C99 / Wasm Host Program 与 NPU parity（CPU 先行，GPU 另授权）

**状态：已完成（CPU + 真实 Metal 机制验收；不是性能结论）。** 工具链已冻结并通过
`scripts/doctor.py` 的 `host.toolchain`：Homebrew LLVM clang 22.1.2 + `/opt/homebrew/bin/wasm-ld`
+ `wasmtime==43.0.0`（`pyproject.toml` / `uv.lock`）。早期 WABT 探针
（`tests/support/host_tiling_probe.c/.py`）只用于确认标准 Clang + lld 能生成合法 Wasm、module 可
复用、每调用新 Store、非法边界返回非零、死循环被 fuel trap；正式实现不再依赖 WABT。探针钉住的
ABI 事实保留：输入/输出必须写入 **module 自己 export 的线性内存**，且地址不能重叠；另建外部
`Memory` 会得到全零假象。

本阶段门项与落地：

- **工具链/沙箱**：受限 C99（`stdint.h`/`stddef.h`/`stdbool.h` 的纯类型能力），标准 Clang →
  `wasm32`，`wasm-ld` 链接，`-nostdlib -fno-builtin --no-entry`，唯一 exports
  `{memory, host_tiling, sf_host_abi_offset}`，唯一 imports `sf_input_read_{i32,u32,f32}`；不用
  WASI、不自研 object loader、不加载原生库、不退回 Python AST。`_validate_module()` 与静态预检
  在投影期拒绝 `#include/#pragma/import`、libc/进程/文件/网络、函数指针、varargs、constructor/
  destructor/import attribute。真正 Clang 只在有界 worker 的 build 内执行（90 s 超时、受限 env/
  PATH）；`host-probe` 是显式的 CPU 旁路，不进入评测。
- **合同与身份**：`host_contract_spec()` 从 OpSpec/ABI 派生强类型 `host_contract.h`（Wasm32
  布局、`offsetof` static assert、C99 负数组 static assert、`SF_HOST_USE_<WS>` 宏、32 B 0xA5
  guard、返回码），`assembled_sha256(header, source)`；`assembly` 全文进构建证据。小输入
  （≤64 KiB，合计 ≤1 MiB）给 typed pointer，大输入给 `{id, numel, elem_nbytes}` + 白名单 read
  import。`NativeExecutable.descriptor["host_program"]` 记录
  `{schema_id, entry_path, program_sha256, assembled_host_sha256, wasm_sha256, entry,
  used_workspaces, fuel, memory_bytes, engine/engine_version, compiler_version}`，进入
  `native_digest`；Wasm bytes 不进 Schema-4 `entries`，不滥用 `payloads`。
- **执行**：build 编译一个 Module，评测期每次调用新建 Store（fuel + `set_limits(memory_size,
  memories=1, instances=1)`）；`MetalV2Executor` 独持 `host_runtime`。`from_descriptor` 在同进程
  复用 module，跨进程重编译时必须逐字节复现 `wasm_sha256`/`assembled_host_sha256`，否则拒绝。
- **原子提交**：`host_tiling()==0` 才提交；非 0/trap/fuel/span 越界/guard 哨兵被破坏/只读输入被
  改写全部丢弃 span 与 resolved scalar。host 非 0 或 span 越界在正确性里记该 case `incorrect`
  （带 typed run document），不是 evaluator crash；benchmark 侧抛 `GeometryError`。
- **三态初始化与计时**：workspace 每块声明 `host` / `device_zero` / `none(poison)`，host 写与
  device_zero 同块在 envelope 阶段拒绝；未被任何 possible stage 绑定的 workspace 不分配；容量
  仍按 possible stages 并集。`host_inside_timing` 路径不再重拷只读输入、`stage_args`/fill plan
  预建、`repeat_plans` 只恢复 `read_write`、host-written 块不做 device zero fill——真实 Metal
  实测每次迭代 **1 次 Wasm 调用 + 1 次 CommandBuffer**。device-only 时间只作诊断，不正确替代
  E2E 发布谓词。
- **C99 planner（候选侧）**：`tests/support/golden/shared/hostprogram/kda_planner.c`。mode 1 用
  `divmod` 均衡切分 + 两 word 位打包（`word0=token_begin|valid<<16`、`word1=chain|piece<<8|
  pieces<<16`），并重置 `handoff_flags`；mode 2 用整数/定点成本键（`size<<8 + 230 +
  (非末 piece?102:0)`）、range-cut、手写二叉 max-heap LPT 到 ≤8 worker、按 worker 分组发射。
  Python oracle（`tests/support/hostprogram_oracle.py`，复用 npu_sim 的独立 `plan_pieces`、
  `heapq` LPT）只在测试里做差分，不进 candidate/timed path。
- **题面/quick/离线修复**：`tcenv/task_presentation.py` 只对 `cu_seqlens` 任务插入 Host Program
  小节，包含完整 `host_contract.h`、C99 子集、返回码和最小 host+device 示例；固定 shape 题面
  逐字不变。`uv run python -m tcenv.cli host-probe <task> --source host/tiling.c` 可单独编译/试跑
  并输出 JSON span；缺工具链返回 `backend_capability_unavailable`。`tests/shared/
  test_hostprogram_contract.py` 用脚本化假候选根据 Clang 诊断修一处错误，完全离线，不花模型费。
- **NPU parity**：模拟器复用同一 Wasm module 与 ABI；同一 HostTarget 桩下 Wasm span 与
  `metal_adapter`/`npu_adapter` 的 work-list 逐字节一致；换成 18-core Metal-like / 64-core
  NPU-like 真实 target 后 item 全集相同、每项恰好一个 core 认领、依赖边等价，claim map/grid
  允许不同。Adversarial Core Scheduler 与 DMA 对齐踩踏检查保留。所有 NPU 报告固定
  `simulated=true/publication_allowed=false/measurement_kind=simulated`，不进发布/confirmed/
  catalog/brief，Qingwei NPU-P0 仍 blocked。
- **生命周期**：读输入就派生 `input_dependent=true + preprocesses_inputs=true +
  inside_timing=true`；谎报（如 `inside_timing=false`）在 build 被拒，报告 `compile_error`
  并带 `assembled_host_sha256`。`host/tiling.c` 与 `host/tiling.json` 互斥，恰好一条 entry。
- **资源**：`fuel = 2_000_000 + 512*chains + 64*tokens + 1024*max_items`；内存 = mapped 输入 +
  workspace 容量 + guard + 64 KiB 栈 + 256 KiB 余量 + `16*4*max_items`，64 KiB page 对齐。
  这是包络上界加固，不是 max-scale p95；mixed/canonical 实测只需 ~2.35k / 2,022,400 fuel。

**实测（本机 Apple M4 Max，真实 Metal；命令均串行 `-n 0`）**

- `tests/shared/test_hostprogram_contract.py` 24 passed；`test_hostprogram_planner.py` 11 passed
  （8 个 L1 case 的 mode-1 bytes 与独立 `plan_pieces` 相同；skew probe 的 mode-2 bytes 与独立
  oracle 相同，LPT makespan ≤ round-robin）；`test_hostprogram_npu_parity.py` 2 passed。
- `tests/backends/metal/test_varlen_hostprogram_gpu.py` 9 passed：9/9 L1 cases 正确；
  `host_runs` 记录 `input_reads`（`cu_seqlens` 4×i32）、`workspace_writes`、`unwritten_policy`
  （`work_items=host`、`handoff_states=poison-sentinel`）、`geometry/launch` 摘要；full 模式
  候选中位数 ≈443 µs，其中 host ≈200 µs、device ≈65–67 µs、E2E ≈439–443 µs，`timing_mode=timed`
  （**只证明分解与 E2E 口径，不发布 speedup**）。native descriptor 7,514 B < 8,192 B。
  负例：非 0 返回/span 越界 → `incorrect`（无 partial span）；plan 不覆盖分段 →
  `incorrect/unsafe`；host 写 device_zero 冲突、未绑定 workspace、危险 `#include`、生命周期
  谎报全部在 envelope/build 拒绝。
- V1/V2 未退化：`test_varlen_gpu.py` + `test_varlen_dyn0_gpu.py` 30 passed；相关 CPU 回归
  （workspace/report-identity/cases/tasks/opspec）469 passed；固定 shape 题面与 descriptor
  去重（fixture B 7,209 B）原样保留。
- 提交后的最终全量快测：4,342 passed / 13 skipped / 66 failed。分类：26 个 P2
  manifest/resume/confirmation 归档证据失败由本阶段闭包与题面身份变化引起（见下）；其余 40 个
  环境失败包括 `AF_UNIX path too long`、workspace/remote/selfcheck 环境，以及
  `test_busy_process_check_ignores_stopped_and_unrelated_processes` 的 `ps` 权限错误。
  26 + 40 = 66。期间另一次运行是 67/4,341，多出的 1 个是
  `test_remote_job_worker_bridge::test_a_descendant_holding_the_pipe_cannot_block_the_bridge`
  （环境/时序），不是 V3 回归。

**闭包与 P2 影响（未伪造、未改写 P2 证据）**

- `code_import_closure` 70 → 71 个文件：`tcenv/task_presentation.py` 新增对
  `tcenv/hostprogram/contract.py` 的 import，后者进闭包；`portfolio_runner.py`（不在闭包内）的
  `hostprogram.delivery` 与 Metal 侧文件不改变闭包。`task_presentation.py` 自身字节也变了。
- 因此**每个任务**的 `deployment_pack_digest` 前进（例：varlen_kda `c47c361b…` →
  `c5b3bc7f…`、rmsnorm `2bc112d2…` → `8a123cf1…`、gemm `8b30f539…` → `2491791a…`）。
  P2 归档确认/恢复/清单证据绑定旧 digest，26 个归档测试不再通过；没有删除、跳过或放宽任何
  P2 断言，也没有改写归档文件。**需要按安静窗口流程重录受影响 P2 证据，由主会话处理**；
  V3 本身不宣称 P2 通过。

**复核补记（第二轮，针对第一轮验收提出的问题）**

- **空 dispatch 例外收紧（第二轮再收紧）**：`pre_dispatch_failure` 不再信任
  `execution['host_program']` 是 Mapping 或自报字段，而是从**投影候选**取唯一的
  `host/tiling.c` entry，要求 native descriptor 的 `host_program` 绑定同一 `entry_path`/
  `program_sha256`/`assembled_host_sha256`/`wasm_sha256`，并要求失败 run 文档带
  `schema_id=tcenv.hostprogram.run.1`、同一 `candidate_sha256`/`program_sha256`/
  `assembled_host_sha256`/`wasm_module_digest`/`native_digest` 和 typed 失败字段；
  `PortfolioRunResult._check_identity` 用同一验证函数复核，不信任布尔字段。msl-single、linear
  JSON、以及“伪造 host_program + 伪造 failure”的空 dispatch 全部按 frozen geometry mismatch
  拒绝，`test_shape_portfolio_runner.py` 为此新增 msl-single/linear/两个伪造反例；真实 Metal 上
  C host 非 0 失败仍被接受为正例。
- **调用后 pointer 复核**：`HostRuntime.execute` 在状态 0 后、收集 span 前重新读取每个输入/workspace
  pointer 及 capacity/elem_nbytes，与调用前保存值比较，并在当前值上重做边界与 disjointness 检查；
  pointer 被改写（包括 `workspaces->work_items = (uint32_t *)inputs->cu_seqlens` 和
  import 的 `dst_off` 写回 ABI pointer）一律原子丢弃。CPU 合同测试覆盖这两种负例。workspace 参数
  的别名/赋值/取地址在投影预检显式拒绝；helper 形参别名由“未分配 workspace 非空 span 拒绝”和
  device-zero 运行期检查兜底。
- **真正的 build-once 与完整身份缓存**：`HostBuild` 保存 build 校验时创建的 Engine/Module，
  `HostRuntime` 复用它们，每次调用只新建 Store；跨进程才重编译并校验 digest。缓存键是
  `(wasm_sha256, assembled_host_sha256, program_sha256, fuel, memory_bytes, used_workspaces)`
  的完整身份，而不是仅 `wasm_sha256`：仅加注释的源码编译出相同 Wasm、不同 assembled/source，
  此时 A 已缓存、B 的 prebuilt descriptor 必须重编译 B 并校验，而不是被错误拒绝。CPU 测试用
  `wasmtime.Module`/`Store` 构造计数器证明 1 个 Module、N+1 个 Store；Metal 测试固定 A/B 顺序。
- **NPU parity 重写（并区分同/异 target 语义）**：同一 HostTarget 下比较**全部**有效 span
  （名字/offset/count/elem_nbytes/字节/digest）与 resolved scalars；真实异 target 下分别执行
  同一 module，把各自输出交给各自 scheduler，只比较**去重后的 item 全集**、每项唯一认领、完成
  集合和用 `(chain,piece)` 稳定键表达的语义依赖边；发射顺序、字节布局和 claim map 允许不同。
  `SF_PLANNER_MODE 3` 是一个合法的 target-dependent 顺序正例（大 threadgroup target 下反转
  work_items），测试证明字节不同但语义 parity 通过。V2 的对抗调度/DMA 检查保留，全部报告仍
  固定 simulated/publication_allowed=false。Dyn-0 的 `SfHostResolved` 只有 `reserved`，因此
  resolved scalars 断言为空字典（不是省略该比较）。
- **缺依赖的 typed outcome**：`MetalTrackBuilder.build` 在编译 MSL 前先发现 Host toolchain，缺
  clang/wasm-ld/engine 时抛 `HostCapabilityBuildError`（`toolchain_profile` +
  `backend_capability_unavailable`）；`v2_worker` 将其投影为 build-free `rejected` 报告，executor
  cache-miss 重编译缺依赖也返回 `rejected` 并保留已构建的完整 build。`test_hostprogram_capability.py`
  分别模拟三种缺依赖，验证 C 路径得到 typed environment outcome、linear JSON 路径仍可构建，
  且报告不出现 `host_program_invalid`（不怪候选源码）。
- **计时计数证据**：`benchmark_protocol` 新增 `host_wasm_calls`、`host_audit_calls`、
  `command_buffers`（仅候选）、`native_batch_calls`、`stage_args_rebuilds`、`read_only_restores`。
  真实 Metal full 实测：145 次 Wasm 调用 / 145 个候选 CommandBuffer / 1 次 audit / 0 次
  stage_args 重建 / 0 次只读输入重拷；`host_audit_calls < host_wasm_calls` 证明 timed 样本不计算
  digest。`command_buffers == host_wasm_calls` 证明每个 launch 一次 Wasm + 一次提交（native batch
  与 Objective-C encoder 都计数，baseline 的交错提交不计入候选）。
- **第三轮验收修复（5 项 + 文档算术）**：
  1. 空 dispatch 例外改为从投影候选的 `host/tiling.c` entry 与 native descriptor 的
     `host_program` 身份推导，逐项校验 candidate/program/assembled/wasm/native digest 和失败 run
     schema；`PortfolioRunResult._check_identity` 复用同一验证函数。新增 linear/msl-single
     “伪造 host_program + 伪造 failure”反例，仍按 frozen geometry mismatch 拒绝。
  2. build 缓存键改为 `(wasm, assembled, program, fuel, memory, used_workspaces)` 完整身份；
     仅注释差异的 B 不再命中 A，`from_descriptor` 重编译 B 并校验。新增 CPU 缓存键测试和 Metal
     A/B 顺序回归。
  3. 只暴露 `access=read` 的输入（`read_write` 由 device 独占，新增负例）；fp16/bf16 用
     `uint16_t` raw-bit typed pointer，大输入新增 `sf_input_read_u16`（证据记录语义 dtype 与
     import 名）；mixed 的 10 个只读输入全部 mapped，bf16 档也覆盖。修正文档“18 个输入”的不实
     表述为实测 10 个。
  4. 题面新增最小 host 函数体 + 配套 MSL kernel + `host/execution_geometry.json`；CPU 测试编译并
     执行 host 示例、解析/求值几何，Metal 测试用 `xcrun metal -c` 编译 MSL 示例。
  5. 异 target parity 只要求去重 item 全集、每项唯一认领、完成集合和 `(chain,piece)` 语义依赖边；
     新增 `SF_PLANNER_MODE 3` target-dependent 顺序正例（字节顺序不同但语义 parity 通过），同
     target 的全 span/resolved 比较保留。
  6. 文档失败算术按最终实测统一为 **26 P2 + 40 环境 = 66**（4,342 passed）；期间另一次
     67/4,341 多出的 1 个是 remote-job-worker 桥的时序环境失败，不是 V3 回归。
- **题面示例**：`host_program_statement_section` 现在包含一个真正最小、可编译/可运行的
  `host_tiling` 函数体（读 `cu_seqlens`、写 one-piece `work_items`/`item_counts`/`handoff_flags`）、
  配套 MSL kernel 和 `host/execution_geometry.json`；CPU 测试编译并执行 host 示例、解析并求值
  几何，Metal 测试用标准 `xcrun metal -c` 编译 MSL 示例。它不是 KDA 正确 kernel，题面明确标注。
- **复核后测试数（第三轮）**：CPU host 合同/planner/parity 49 个（新增 read_write 排除、fp16/bf16
  raw-bit、示例、完整缓存身份、伪造空 dispatch、target-dependent parity）；P1 runner 22 个；
  真实 Metal host+device 10 个 + capability 8 个；V1/V2 30 个保持通过。提交后的最终全量快测
  **4,342 passed / 13 skipped / 66 failed**，失败分类严格为 **26 个 P2 归档重录项 + 40 个既有
  环境失败**（26 + 40 = 66）。40 个环境失败包括 `AF_UNIX path too long`、workspace/remote/
  selfcheck，以及 `test_busy_process_check_ignores_stopped_and_unrelated_processes` 的
  `[Errno 1] Operation not permitted: 'ps'`（沙箱禁止 `ps`）。期间另一次运行为 67/4,341，多出的
  1 个是 `test_remote_job_worker_bridge::test_a_descendant_holding_the_pipe_cannot_block_the_bridge`
  （环境/时序），不是 V3 回归。

**未完成**：V4 Dyn-1（`ResolvedEnvelope`/逐 case `LaunchGeometry`/动态 parity）、V5 的真实
Metal L2 验收与 P2 重录（CPU/协议切片已完成）、V6 真实规模性能与搜索；真实 NPU 设备/工具链/
ABI/profiler 仍未接入；最大
规模 planner 的 p95 host 时间与 target-specific 诊断预算尚未冻结（留到 V6 前）。本阶段没有
发布任何 speedup、没有改写 KDA 性能结论。

### V4：Dyn-1 动态 grid/stage skip（可选优化，实测门）

**结论：跳过（实测不支持引入 Dyn-1）**。新增 `tests/backends/metal/test_varlen_dyn1_diagnostic_gpu.py`
在真实 Metal 上对 V3 host+device fixture 做 6 次独立 full run，得到：canonical 的静态 piece
grid 18、实际 item 16、空槽位 2；max_handoff_stress 的 grid 18、实际 item 10、空槽位 8。
candidate 中位数分别为 **548.09 µs / 547.85 µs**，host 约 298–305 µs、device 约 65–70 µs、
E2E 约 540–550 µs，噪声 cv 0.017–0.040（均在 0.05 内）。空槽位增加没有可辨识成本，主要成本
是 Host Program，不是空 TG/空 stage。因此不引入 `ResolvedEnvelope`/`LaunchGeometry`、动态
runtime_scalars 或新的动态 parity；Dyn-0 保持不变，直接进入 V5。该诊断不发布 speedup，
原始结果由测试写入 `/tmp/varlen_dyn0_idle_diagnostic.json`。

- 先用 V2/V3 在真实规模测量空 TG/空 stage 的 E2E 占比；只有显著时才实施，阈值以测量结果冻结，
  不预写个位数百分比或固定 5%。
- 实现 `ResolvedEnvelope`：全部 possible stages 编译/反射绑定；`runtime_scalars` 独立于静态
  `derived_scalars`，动态只允许 grid/stage-active，`threads/threadgroup_bytes` 静态，逐 stage
  `max_grid` 明确上限。
- 实现逐 case `LaunchGeometry`、动态 output coverage 和新 parity，旧静态 parity 不变；Dyn-0 与
  Dyn-1 在 geometry 中显式声明并存，不是所有候选的默认门。

验收：有量化数据证明收益，动态 output coverage 与逐 case parity 全过；没有数据则保持 Dyn-0，
不为了计划完整而做 Dyn-1。

### V5：L2 runtime total/count、capacity 与 overlap（新模型阶段）

- 实现 logical/resolved/capacity shape 与 max-capacity ABI；L1 只是 resolved total/count == capacity；
  capacity 实例的 `SemanticInstanceV2.cu_seqlens=None`，具体 benchmark 分布由 workload profile 标识；
  Dyn-1 若启用则复用其 runtime scalars，不强制先做。
- **dispatch 主体仍是 capacity 实例**，resolved total/count 只进 evaluation input/launch 证据；
  不在现有有限域 guard 中使用动态值。resolved 区间 variant 若未来需要，另开 overlap 设计。
- 按 §2.4 清单比较 output logical 前缀和 `final_state[:num_seqs]`，padding/guard 与 `cu_seqlens`
  capacity 尾部 sentinel 必须保持；拒绝超 capacity，**不接受 empty batch，只测 empty sequence**；
- Host Program cache 若加入，必须绑定 evaluation input digest 并记录 cold/miss；
- 可加入 ring-buffer host/device overlap 测量，serial/overlap 分别标识并使用同一测量根；
- 用同一候选在连续多个不同 total/count 调用上验证无重编译语义（是否重编译由候选与缓存身份决定，
  证据必须如实记录）。

验收：continuous batching 场景不再依赖为每个 total/count 建一个静态档；L1 证据不冒充 L2。

**本轮实施结果（Dyn-0 最小垂直切片，CPU/协议验收；不是 GPU/性能结论）**

合同面：

- OpSpec v2 档位新增第三种形状：`capacity: {total_tokens, num_seqs, cu_seqlens_sentinel,
  padding_poison_byte}` + `workload_profiles: [{id, seq_lens}]`。它与 `seq_lens`（L1）互斥，固定档与
  变长档仍不能同题混用；sentinel 必须为负（不能是便利的 `0`）、poison byte 必须非 0、至少两个
  profile 且至少两个不同 total 与两个不同 count、并且必须有一个 profile 正好覆盖 capacity。
- `SemanticInstanceV2` 增加 `capacity`/`workload_profiles`，并加 `variable_length_kind`
  (`fixed`/`l1`/`l2`)：**判别不能用 `cu_seqlens is None`**，L2 故意没有 canonical 分段。L2 的
  symbols 是容量（buffer/ABI span/workspace/Host 存储都按它分配），`cu_seqlens=None`，
  `capacity`+`workload_profiles` 进 `semantic_instance_digest`。
- 三层形状在 `tcenv/taskcontract/runtime_shape.py`：`capacity`、`ResolvedShape`（本次 total/count +
  profile id + sentinel + padding byte + 声明的 canonical `seq_lens`）、以及由声明 dim 词决定的
  logical/padding 区域。case→profile 的分配是确定且公开的（profile 按 id 排序，第 i 个 case 用
  `i % len(profiles)`），所以一次 case sweep 就是 shrink/grow 序列；期望值来自声明 profile，
  `gen_inputs` 只收到它、不能自证。
- 边界合同：`cu_seqlens` 按 capacity 分配；前缀是本次偏移（i32、从 0 开始、非降、末项等于 resolved
  total、prefix 长度 `num_seqs+1`），尾部必须逐项等于声明 sentinel。empty batch（total=0）、负值、
  非单调、错误末项、非 sentinel 尾部、便利 0 尾部、超 capacity 全部 typed 拒绝；empty sequence 合法。
- 正确性：只比较 `output[:resolved_total]` 与 `final_state[:resolved_num_seqs]`；logical 区域之外
  逐**字节**校验声明 poison（fp16 padding 的位型在 float32 展开后不是同一值，所以检查必须落在存储
  字节上）。输入侧由 task `gen_inputs` 写、harness 在跑前校验；纯 `write` 输出侧由 harness 每个 trial
  重新写 poison 再校验。**error scale 也只在 logical 区域计算**：默认 RMS 只在
  `output[:resolved_total]`/`final_state[:resolved_num_seqs]` 上取，自定义 `error_scale` 钩子拿到的
  也只是 logical 视图（边界张量保持 capacity 长度），返回的尺度必须能广播到 logical 结果形状；一个
  覆盖整 capacity 缓冲区的尺度被 typed 拒绝。

执行/证据面：

- `SfHostResolved` 现在有真字段（L2 的 `total_tokens`/`num_seqs`）+ 生成 accessor，Python 每次调用
  在 `_before()` 之后、contract capture 之前写入；C 的 `sf_native_contract_error` 把改写 resolved
  视为合同违规（整次调用丢弃）。HostRun 的 `resolved_scalars` 是从 ABI **回读**的值而不是调用方的
  期望。**L1/fixed 的 `SfHostResolved` 仍是 `{ reserved; }`，生成 header 与 HEAD 逐字节相同**，
  因此既有 assembled digest、descriptor 长度（7505/7209）与 Metal 记录不受影响。
- HostRunEvidence（`tcenv.hostprogram.run.2`）新增 `resolved_shape` 与 `evaluation_input_digest`
  （L1/fixed 为 `None`，字段集不缩水）。成功与失败文档字段集相同：失败的那一次也带它被调用时的
  resolved 形状与输入身份（`resolved_scalars` 仍为空——失败 run 整次丢弃，不回读），因此失败批次
  仍可定位/重放。按 ordered case 的连续调用全部保留（同一 case 的重复 correctness trial 按内容
  去重），并在 `launch_runs` 里每**次**调用各记一条（phase/ordinal/case/trial/resolved 形状/digest/
  provider）：Dyn-0 的静态 grid 跨批次完全相同，按 dispatch 签名去重会把“同一个 build 跑了多个
  批次”吞掉。
- `evaluation_input_digest` 对 L2 显式绑定**声明的** `workload_profile_id` 与 canonical
  `seq_lens`（含 `workload_profile_digest`），并**校验**由真实 `cu_seqlens` 字节推出的
  resolved total/count 等于该 profile；resolved total/count 本身仍从字节读，不从 case 名或 profile
  推断。它的 scope 在 fingerprint 文档里写明为
  `boundary-bytes+shape-metadata+declared-profile`：只 hash 边界字节，不是 byte-complete 的整输入
  内容哈希（payload 变化由数值比较与 padding 检查负责），证据不得把它当成后者的证明。
- dispatch 不变：域点仍是 capacity 实例，guard 名字空间只有 capacity symbols + attrs，resolved 值
  不出现；fallback 仍按域点独立，resolved 不产生新的 variant 或 overlap 域。
- 同一 build 连续执行不同 total/count **不重编译**：resolved 不在 contract/header/编译单元里，
  `program_sha256`/`assembled_host_sha256`/`native_sha256` 跨调用不变，CPU 测试对每个 case 断言
  build 身份不变且计划随 resolved 改变。本轮**没有加入** Host Program cache（因此没有 cold/miss
  语义需要记录），也没有 overlap 测量。
- benchmark 的 rotating sets 对 L2 就是声明的 continuous-batching 顺序：set `i` 跑第
  `i % len(cases)` 个声明 case（数量 `max(rotation, len(cases))`，上限 `RING_MAX`），所以一次计时
  同时覆盖满 capacity 与更小的批次；每个 set 的输入/padding/Host 存储都在 timed 之前建好，native
  直接提交与 fresh-input 语义不变。证据写入 `benchmark_batches`（每批次的 resolved 形状与
  evaluation input digest）并用 `benchmark_mixture_problems` 自检/复核：顺序必须与声明一致、形状
  必须等于该 case 的 profile、至少两个不同 total 与两个不同 count、必须包含满 capacity 点、每批
  必须有输入身份、provider 必须是 candidate（fallback 的时间不进 variant 的 mixture）。
- NPU：模拟器继续消费真实 host work list，只是每个 resolved 批次各调度一次；报告仍固定
  `simulated=true`/`publication_allowed=false`。

**未做/仍待授权**：

- 真实 Metal 上的 L2 正确性、L2 timed 证据与 P2 freeze/golden/evidence/manifest 重录（P2 仍
  **deferred**；本轮未改 `docs/架构/shape-portfolio-阶段验收/P2/` 任何文件）。本轮只固定了
  benchmark mixture 的**顺序/证据合同与自检**（CPU），真实设备上 mixture 的计时证据要在 GPU
  授权后补；在那之前不得把任何 mixture 数字当成速度结论；
- Dyn-1、overlap、V6 真实规模性能与搜索；
- Host Program 按 resolved 缓存（若将来加入，必须绑定 evaluation input digest 并记录 cold/miss）。

**review 修复（五个 release blocker，均为 CPU/协议证据）**：

1. logical 尺度：`prepare_v2_case` 的默认 RMS 与自定义 `error_scale` 都只在 logical 区域计算，
   `_check_v2_case` 不再对 per-row 尺度做 3-D 切片；负例证明“`output` 全对 + `final_state` 全错”
   会失败，改变 padding poison 只改 padding 判决、不改 logical 误差，整缓冲区尺度被 typed 拒绝。
2. resolved 轴别名：`tcenv/opspec/model.py::resolve_axis_kind` 是文档与运行时共用的唯一规则——
   canonical `total_tokens`/`num_seqs` 是 resolved 轴，独立符号积（各因子取值都不等于容量
   total/count）是 invariant；**opaque 符号恰好取容量值**（`T2=total_tokens`）、融合词
   （`total_tokens*H`）与缺词都在校验时拒绝，运行时 `require_capacity_axes`/`padding_slabs` 对
   手搓实例同样 fail closed。input/output/final_state 三个位置都有负例；`dt_bias` 的 `H*D`（恰好
   等于容量 total）仍合法，说明规则没有误伤独立符号积。

   复核追加（形状代数本身无法闭合的那一类）：仅靠“符号值等于容量”判别名会被**字面量/参数积**绕过——
   `final_state.dims[0] = "4" / "2*2" / "H*2" / "H*D"` 都能通过文档校验，此时整条轴被当 invariant，
   logical 区域变成整个 capacity 缓冲区（实测 scale 由 padding poison 主导：1.3306672482102278e16），
   一个全错的 logical `final_state` 会以 `ok=True, n_bad=0` 通过。纯数值规则无法区分这种写法和
   `dt_bias` 的合法 `H*D`，所以判据改为**角色**：`opspec/model.py::result_capacity_axis_problem` 要求
   结果张量的 capacity 尺寸轴必须命名 `total_tokens`/`num_seqs`；文档层（`opspec/v2.py`）与手搓实例
   运行时层（`runtime_shape.capacity_axis_problems`）共用同一函数，非结果张量的 `H*D` 不受影响。
   同一函数还关闭了 canonical 名字的歧义：capacity 的 `total_tokens == num_seqs == N`（N>1）时，
   结果轴上两个 canonical 名字求值相同，无法证明它是 token 轴还是 sequence 轴（误标会把真实 token 行
   当成 padding 要求），因此这种 capacity 必须声明不同的两个界；1/1 因两个 resolved 值恒为 1 而豁免。
   新增负例在修复前的代码上确实失败（文档层被接受 / 比较器返回 ok / `result_capacity_axis_problem`
   返回 None），修复后两层都 typed 拒绝。
3. 输入身份：`evaluation_input_digest` 绑定声明的 profile id/canonical seq_lens 并校验字节推出的
   total/count 与该 profile 一致；`prepare_v2_case` 也会拒绝“同一个 case 名跑另一个 total”的生成器。
   digest 的 scope 写进文档，不再暗示 byte-complete。`_check_v2_case` 的 L2 门控是**实例种类**而不是
   `resolved_shape is not None`：缺失的 resolved 形状是它自己的 typed 拒绝，不能再借 `None` 掉进
   整缓冲区比较、跳过边界校验/digest 重算/padding 检查（修复前用 reference 字节可得到 `ok=True`）。
4. HostRun 证据：`correctness` 返回全部 ordered case 的 HostRun 列表（不再只返回最后一个），失败
   文档带同一字段集（含 resolved 形状/输入身份），`record()` 幂等合并内容相同的 run 并保留每次
   调用的 `launch_runs`。
5. benchmark：按声明 profile 顺序计时多个 resolved 批次并写 `benchmark_batches`；mixture 自检 +
   负例（只跑满 capacity、乱序、伪造 resolved、fallback 混入、缺输入身份全部拒绝）。mixture 中位数
   与单形状基线不可比，所以 `evaluate` 不给它发布 speedup（`speedup_withheld`，数字仍在报告里
   供诊断）；报告的 `comparable_single_shape_measurement` 跟随 scope，而不是固定 `False`。exact-dispatch
   负例（伪造 resolved symbols → `semantic_unresolved`，未知 size →
   `unsupported_contract_capability`）确认 resolved 不是域点。

   复核撤回：**单 profile 不是可声明的合法边界**——`_validate_capacity_tier` 要求至少两个 profile 且
   至少两个不同 total 与两个不同 count（一个只有单一 resolved 形状的档就是换个名字的 L1），真实
   `opspec_v2_from_document` 会 typed 拒绝。之前用 `dataclasses.replace` 手搓单 profile 实例并把它
   称作“合法边界”的测试已撤回：现在文档层断言真实拒绝，运行时只保留**防御性**属性（若这种不可声明
   的实例真的到达 benchmark，则按单形状计时、不自造 mixture、不自伤，且 `benchmark_mixture_problems`
   明确断言为空）。新增验收测试不含 `pytest.skip`/`xfail`：native 工具链缺失以 typed capability 失败
   暴露，不让用例静默消失。

### V6：真实规模性能（已完成；付费搜索不在本次范围）

- D=128、H=64/96、fixed/mixed/uniform 的六档 KDA 均为 `total_tokens=8192`，任务档位见
  `tasks/varlen_kda/opspec.json`；
- `tests/support/golden/metal/varlen/chunked_gform.metal` 是 **msl-single** 的真实
  chunked/`simdgroup_float8x8` 候选：BT=8，每个 `(sequence, head, v-tile)` 一个 threadgroup，
  G-form `A/Base` 使用矩阵乘，下三角 residual 后保留 oracle 的 fp32 token-order state/output；
- 独立审查修复了 residual 漏掉当前 token decay 的错误：KD/QD 使用 `exp(L)` 而非 `exp(L_prev)`；
  小规格 9/9 L1 case、旧顺序负例和六档 canonical 由
  `tests/backends/metal/test_varlen_v6_chunked_gpu.py` 固定；
- 六档都完成 search + fresh 的 9-case correctness、15 次 E2E 计时、benchmark/golden 两个用途点和
  active DispatchManifest；candidate cv 均 <=0.0473，六档相对登记的朴素 Torch-MPS 逐 token
  baseline 的 fresh speedup 为 8.7205×–10.3821×，每档独立过 1.10 gate；
- **比较边界**：该 baseline 在 Python 中逐 sequence/token 调用大量 Torch-MPS 算子，候选是融合
  Metal kernel，所以接近 10× 主要反映消除框架/dispatch 开销；不能外推为相对优化 KDA/CAKE2、
  CUDA/CANN 或 vendor kernel 的通用加速。六档均 `provider=candidate`、无 fallback 计时混入，
  NPU 模拟数字未参与比较；
- V4 已实测跳过 Dyn-1；本候选 grid 精确对应实际 `(sequence, head, v-tile)`，不引入空 stage。
  msl-single 无 Host Program，不用另造的 device-only/host 预算替代 E2E；
- 证据、原始 15 样本、baseline 限制、约 58.2 分钟耗时分解和只读重放见
  `docs/架构/v6-chunked验收/README.md`；`verify.py --check` 已从 sidecar 独立重建并激活六份
  manifest，digest 完全一致；
- V1/V2/native Host/V5/V6 Metal 回归 51/51，定向 CPU/协议/结构 633 passed。P2 安静窗口重录后，
  完整 CPU 快速套件 **4549 passed、13 skipped**；
- 最终 P2 重录已完成（P2-F，保留历史 start-surface 拒绝）；固定长度 KDA 标准任务与复杂
  bundle 的离线/CPU 示例见 `docs/架构/fixed-kda-bundle示例.md`，真实 Metal、付费 msl-bundle
  搜索和 Qingwei NPU-P0 仍是独立事项。

---

## 9. 文件级改动地图

| 层 | 文件/位置 | 改动 |
| --- | --- | --- |
| OpSpec | `tcenv/opspec/model.py`、`opspec/v2.py`、schema | varlen 校验、profile、L2 shape/capacity |
| 身份 | `portfolio/domain.py`、`taskcontract`、validation | 三层身份、dispatch/fresh seed/input replay |
| 题面/可写性（V3 已实施） | `tcenv/task_presentation.py`、`tcenv/cli.py`、协议参考、quick/self-check | 完整 `host_contract.h`、C99 规则/返回码、`host-probe` 编译试跑、最小 host+device 示例与离线修复测试；固定 shape 题面逐字不变 |
| Host Program（V3 已实施） | `tcenv/hostprogram/{contract,precheck,toolchain,runtime,delivery,errors}.py`（不引入 Metal import）、`tcenv/backends/metal/host_program.py`、`pyproject.toml`、`scripts/doctor.py` | 受限 C99、自动 header、大小输入分流、Wasm ABI、Module 复用/每调用 Store、capacity fuel/内存公式、有效 span、trap/返回值原子丢弃；标准 lld 与 engine 依赖和 typed 拒答 |
| 构建身份（V3 已实施） | `tcenv/backends/metal/track_adapter.py`、`tcenv/remote/artifact_bundle.py`、`execution_geometry.py` | `NativeExecutable.descriptor["host_program"]` 进入 native digest；executor 独持 `host_runtime`；不改 Schema-4 闭集，不滥用 payloads |
| Portfolio 解耦 | `tcenv/portfolio/candidate.py` | 改为调用 `tcenv.hostprogram` 中立校验器，去除对 `backends.metal.v2_runtime` 的反向 import |
| Metal（V3 已实施） | `backends/metal/v2_runtime.py`、`track_adapter.py`、`host_program.py` | Dyn-0 静态路径、possible-workspace 并集分配、有效 span、host/device_zero 冲突拒绝、host evidence、一次 Wasm + 单 CommandBuffer 计时路径 |
| 生命周期（V3 已实施） | `taskcontract/portfolio_runner.py`、`track/lifecycle.py` | C99 Host Program 派生 input_dependent/preprocesses_inputs/inside_timing；三态 workspace 初始化；audit/timed 分离；pre-dispatch 候选失败不伪造 dispatch |
| 几何（可选 Dyn-1） | `track/geometry.py`、backend stage binding | `ResolvedEnvelope`、独立 runtime_scalars/max_grid、逐 case `LaunchGeometry`、动态 output coverage 与逐 case parity |
| 任务 | `tasks/varlen_kda`、`tasks/varlen_scale` | 分段 case、V2 workspace 容量公式（`varlen_kda`）、真实规模档 |
| Metal 测试 | `tests/backends/metal/` | V2：`test_varlen_dyn0_gpu.py`（两类 fixture + 新负例）、`test_varlen_workspace.py`（容量/分配范围，CPU）、`test_varlen_report_identity.py`（错误报告仍可指纹）；V3：`test_varlen_hostprogram_gpu.py`（host+device 正例、计时证据、host 失败/越界/envelope 负例）；V1 的 `test_varlen_gpu.py` 保留 |
| Metal fixture 数据 | `tests/support/golden/metal/varlen/{prep_chain,piece_handoff,host_planned_handoff,chunked_gform}.metal`、`tests/support/golden/shared/hostprogram/kda_planner.c` | 候选拥有的固定源码，不是 harness 规则；`host_planned_handoff` 用候选 C99 计划替换 device plan stage；`chunked_gform` 是 V6 msl-single 真实矩阵候选 |
| V6 验收 | `tests/backends/metal/test_varlen_v6_chunked_gpu.py`、`docs/架构/v6-chunked验收/` | 当前 token decay 正/负例、六档 D=128 canonical、完整 fresh sidecar、active manifest、逐档摘要和只读重放 |
| 共享测试 | `tests/shared/test_opspec_varlen.py`、`test_varlen_tasks.py`、`test_varlen_npu_sim.py`、`test_hostprogram_{contract,planner,npu_parity}.py` 等 | 合同、身份、容量公式、固定 shape 兼容、模拟器与负例；V3 的 ABI/预检/编译诊断/fuel/span/planner oracle/同 module parity |
| L2 形状合同（V5 已实施） | `tcenv/taskcontract/runtime_shape.py`（新，闭包 +1）、`tcenv/opspec/{model,v2}.py`、`tcenv/taskcontract/{reference,varlen}.py`、`tcenv/portfolio/domain.py` | capacity/resolved/logical 三套形状、workload profile 分配与期望重建、capacity 尺寸 `cu_seqlens` 前缀/尾部校验、padding slab 与逐字节 poison、`workload_profile_digest`、`gen_inputs(..., resolved=)` |
| L2 执行/证据（V5 已实施） | `tcenv/hostprogram/{contract,native_abi,native_runtime,native_probe_worker}.py`、`tcenv/backends/metal/{host_program,v2_runtime,testing}.py` | `SfHostResolved` 真字段/accessor/C 合同检查、每调用写入与回读、HostRunEvidence 的 `resolved_shape`/`evaluation_input_digest`、logical 区域比较与 padding 校验、`V2ArgSet.load_capacity_padding` |
| L2 测试（V5 已实施） | `tests/shared/test_varlen_capacity.py`、`tests/backends/metal/test_varlen_capacity_compare.py`、`tests/shared/test_hostprogram_npu_parity.py`（capacity parity）、`tests/support/hostprogram_fixtures.py` | OpSpec 正/负例、buffer 尺寸、连续调用无重编译、resolved 篡改/未声明拒绝、逻辑区域与 padding 比较、NPU 模拟 parity（全部 `simulated=true`） |
| NPU 模拟 | `tests/support/npu_sim/varlen_worklist.py`、`tests/shared/test_hostprogram_npu_parity.py` | core/UB-L1/事件图/对抗调度/DMA 对齐/两层 parity；V3 加同一 Wasm module 在两种 HostTarget 视图下逐字节一致、异 target item/唯一认领/依赖边等价；生产组合根不 import，全部 simulated |

每阶段开始先重新计算 `code_import_closure`，闭包改动集中处理并按授权重录受影响证据；不要假设
某个文件“上次不在闭包，这次仍不在”。

---

## 10. 完成定义

只有以下条件全部满足，才能说“GPU/NPU 通用的 varlen 机制已完成”：

- [x] 固定 shape 身份、既有 linear bundle、单文件、R5/远程协议均未退化（V3 回归保持）；
- [x] 同一 ABI 下任意合法 `cu_seqlens` 正确，写死 canonical 边界会被拒绝（V1/V2，V3 未改）；
- [x] 候选可用受限 C99/Wasm Host Program 实现 KDA 级 planner，harness 没有内置任务调度器，
  计时圈内没有 Python AST 或不可信原生库（V3）；
- [x] Metal 真机验证 work-list、partial tail、多 stage 跨核 handoff、逐段 final_state 与全部作弊负例，
  默认路径不依赖跨 TG 自旋等待（V2/V3）；
- [x] Dyn-0 静态几何与静态 parity 可独立完成正确性；Dyn-1 经 V4 实测没有可辨别收益而跳过，
  未把可选动态几何当成完成 varlen 的前置条件；
- [x] L1 canonical 身份不碰撞，L2 total/count 变化有 capacity/resolved-launch/dispatch/fallback 的完整语义
  （V5 的 CPU/协议与真实 Metal 机制切片：capacity 域点、resolved scalar、logical/padding、连续调用
  无重编译、ordered launch/batch 证据均已固定；后续最终 P2 重录见 P2-F）；
- [x] 所有输入驱动 host work 进入计时与证据，KDA prepare 口径没有被偷运进来（V3）；
- [x] NPU simulator 复用同一 Wasm module，具备 consumer-first 对抗调度与 DMA 对齐踩踏检查；同 target
  输出逐字节一致，异 target item 全集/唯一认领/依赖边等价，且所有结果
  `simulated=true/publication_allowed=false`（V3）；
- [x] V6 六档真实性能只在 Metal 真实设备、D=128/H=64/96、登记的同边界 Torch-MPS baseline、
  fresh quiet confirmation 和 active manifest 后发布；比较边界与外推限制见 V6 验收报告；
- [x] Qingwei NPU-P0 仍如实 blocked，Ascend/CANN 参考资料未被写成生产能力。

V0–V6 的 Metal varlen 机制、L2 continuous batching 与真实规模验收现已完成；Dyn-1 是实测后
跳过的可选优化，不是缺口。V2/V3 的早期 Metal 证据本身仍只证明机制，V6 性能只相对登记的朴素
Torch-MPS baseline，不能外推到优化 vendor KDA。NPU 结果仍固定 `simulated=true`/
`publication_allowed=false`，Qingwei NPU-P0 仍 blocked；最终 P2 重录已在安静窗口完成并记录于
P2-F，付费模型搜索仍需独立授权。
