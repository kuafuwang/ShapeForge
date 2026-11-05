# varlen 与输入驱动 Host Program：GPU / NPU 跨后端实施计划

状态：**V0 + V1（L1 任意合法分段）与 V2 Dyn-0（work list / 两类多阶段 fixture / NPU 模拟）已完成；
V3–V6 未实施**。

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
预处理 + 顺序链和真实 piece handoff 两类多阶段 fixture（§6.2）。L2（total/count 也变化）仍是 V5。

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

1. **不在计时圈内解释执行自研 Python AST**。Host Program 改为受限 C99 编译到 Wasm，候选算法
   仍由候选书写；禁止加载候选原生 `.so/.dylib`，也不把 KDA 的 LPT/range-cut 变成 harness
   专用 intrinsic。
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
2. Wasm 使用标准 Clang/lld 与正规 runtime，不自研 object 重定位加载器；`.wasm` 不加入 Schema-4
   `ArtifactBundle.entries`，也不偷塞进只服务 bundle entries 的 `BuildOutcome.payloads`。本地以
   `NativeExecutable.descriptor["host_program"]` 绑定 module digest，executor 独持 `host_runtime`
   管理 Module/Store，远程传输另开版本化 artifact schema。
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
- **C99/Wasm Host Program 仍保留**：它是表达 KDA host planner 的通用能力，不用 Python oracle 的
  耗时决定是否值得存在；Dyn-0 只证明 device-side/静态路线也是合法候选。
- **Dyn-1 是可选优化**：只有实测空 TG/空 stage 成本显著时，才启用 `ResolvedEnvelope`、动态 grid
  和 stage skip。它不是所有候选的前置门，Dyn-0 与 Dyn-1 在 geometry 中显式声明并存。
- 大输入不直接映射进 Wasm；只有小张量给 typed pointer，大输入走白名单区间 read import。
- workspace 初始化改为 `host/device_zero/none(poison)` 三态互斥；按 possible stages 的并集分配，
  不按某个 case 的 active set 临时分配。
- L2 dispatch 主体仍是 capacity 实例，resolved shape 只进 input/launch 证据；NPU parity 分为同
  target 字节一致与异 target item 全集等价，不再要求 core 分配相同。

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
| `evaluation_input_digest` | 本次实际 `cu_seqlens` 字节、case、seed、必要的输入身份 | 重放、Host Program cache、审计这次到底跑了什么 |

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

## 3. 后端中立 Host Program：受限 C99 / Wasm

### 3.1 交付形式：复用 `host_tiling`，不增加 schedule

允许且仅允许一个 host entry：

| 文件 | 角色 | 语义 |
| --- | --- | --- |
| `host/tiling.json` | `host_tiling` | 既有 linear writes；字节与执行语义不变 |
| `host/tiling.c` | `host_tiling` | 新 Host Program；受限 C99，编译为 Wasm，输入驱动地生成 work list/resolved scalars |

`host/execution_geometry.json` 仍是唯一几何声明。Host Program 不是另一个 planner registry，也不是
harness 选出的 KDA 算法；C 源码属于 candidate artifact，摘要进入候选身份。

**不采用**计时圈内的 Python AST 解释器，不采用加载候选原生 `.so/.dylib` 的方案，也不把
`lpt_pack_with_swap`/`range_cut_pack` 等 KDA 算法固化为 harness 专用算子。候选自己书写循环、
heap、二分和成本模型，保留真正的算法自由。

### 3.2 C ABI、资源视图与有效 Span

逻辑 ABI 如下（字段布局在 V3 冻结，指针均为 **Wasm 线性内存内的偏移**，不是 host 地址）：

```c
int32_t host_tiling(const HostInputs *inputs,
                    const HostShapes *shapes,
                    const HostTarget *target,
                    HostWorkspaces *workspaces,
                    HostResolved *resolved);
```

**harness 在编译前自动生成并前置拼接 `host_contract.h`**：根据 OpSpec 中声明的只读输入、
workspace 与 geometry 中声明的 scalar/stage knob，生成具名、强类型字段和 Wasm32 布局 static
assert。候选写 `inputs->cu_seqlens`、`workspaces->work_items.count`、`resolved->num_ctas`，
未声明的名字由 Clang 编译期直接拒绝，不手写宿主 64 位地址或 Wasm32 偏移。该头只包含 ABI 名称、
类型、capacity、offset 与 stage knob，不含任何调度策略；拼接后的全文计算 `assembled_host_sha256`
进入构建证据。

- `inputs` 只含声明为 `access=read` 的 global input，并分两类：
  - **小张量直接映射**：按 target policy 的 capacity 字节阈值（例如 ≤64 KiB）映射 `cu_seqlens`、
    `scale` 等到 Wasm 线性内存，header 给 typed pointer；解码/复制成本计入 host 时间。
  - **大输入按需读取**：`q/k/v/g` 等只在 header 中提供 `id/numel/dtype`，**不给指针**；候选必须
    调用生成的强类型白名单 import（如 `sf_input_read_i32(id, elem_off, dst_off, count)`），由
    harness 校验 id/dtype/范围并记录实际读取区间。未声明读取的大输入不可能被误复制。
- `shapes` 含 symbols/attrs、capacity 和 resolved total/count；`target` 只含 target profile 公开且
  进入身份的设备整数参数。
- `workspaces` 每项含 Wasm 内 typed pointer、capacity、元素 dtype 和候选填写的 `offset/count`。
  **每个 workspace 只有一个连续有效 span**；per-core offset 必须打包在这段内容内部，不在 ABI 中
  引入多段写。未来确需多段时升级 ABI 版本。
- `resolved` 只含 geometry 预先声明的 `runtime_scalars`/grid/stage-active knob，供 Dyn-1 使用；
  Dyn-0 可让 device 直接读 workspace 中的 item count，不需要动态 grid 或 stage skip。
- 相同 candidate + input + target 必须得到相同输出。候选不能读 global output、原输入的写权限、
  host 指针、文件、网络、时钟、随机数或环境变量。

**返回码与原子提交**：`host_tiling()==0` 才表示成功。非 0、Wasm trap 或 fuel 耗尽都丢弃本次全部
span/resolved scalar，返回 typed candidate failure，禁止部分写入生效；缺 clang/lld/runtime 则是
`backend_capability_unavailable`，不是候选代码错误。

**有效 Span 规则**：`offset >= 0`、`count >= 0`、`offset + count <= capacity`，按元素 dtype 检查
字节数；可以返回 `count=0`。harness 只复制有效区间并在 audit 执行中记录其 digest。未写尾部在
correctness trial 和初始化/warmup 中填 poison/sentinel，device 必须按 item count/resolved shape
停止；一旦错误读取尾部，正确性应失败，而不是靠全量零填掩盖。计时路径不为“保险”每轮全量清零
capacity。

**确定性成本键**：planner 的比较键使用整数或定点，平局按 `(seq, head, chunk)` 的整数序打破；
跨 backend 的逐字节一致由合同构造，不依赖浮点排序或编译器收缩行为。

### 3.3 C99 子集与 Wasm 沙箱

Host Program 使用可静态检查的 C99 子集：

- 允许普通函数、循环、条件、整数/浮点运算、位运算、固定大小数组和候选自己实现的 heap/LPT/
  range-cut；足以表达 KDA planner，不需要图灵不完备的“安全简化算法”。
- 只允许 `stdint.h`/`stddef.h`/`stdbool.h` 的纯类型能力；禁止 libc 调用、malloc/free、函数指针、
  inline asm、可变参数、文件/网络/进程/环境访问和未登记 host import。
- 编译目标为 Wasm，导出固定入口与线性内存；**不使用 WASI**。唯一可用 import 是 harness 预先
  登记的输入/target metadata 视图，所有读写都由 offset + capacity 校验。
- **工具链明确依赖**：使用标准 Clang + LLVM `lld/wasm-ld` 生成最终 module，并在 `pyproject.toml`
  声明维护中的 Wasm engine（V3 选型 probe 后冻结，例如 wasmtime）。当前 Xcode Clang 虽能生成
  Wasm object，但本机 PATH 没有 `wasm-ld/lld`，`.venv` 也没有 Wasm runtime，因此 V3 必须先补齐
  并验收这些依赖；不得自研 object 重定位/加载器来绕过它们。
- Wasm module 在 candidate build 阶段编译并进入 build evidence，执行时不按输入重新编译。编译器、
  Wasm engine、module digest、内存上限和 import 清单都进入构建身份。
- **Schema-4 兼容的 module 身份**：`.wasm` 不加入 `ArtifactBundle.entries`，避免破坏
  `candidate entries + manifest + execution_geometry` 闭集；也不放进 `BuildOutcome.payloads`，
  因为现有 payloads 只允许命名 bundle entries。本地构建把
  `{source_path, source_sha256, assembled_host_sha256, wasm_sha256, engine, compiler}` 写入
  `NativeExecutable.descriptor["host_program"]`，使 module digest 进入 `native_digest` 并接受交叉
  审计。executor 另持 `host_runtime`，不让 live module 充当身份证据；每次执行前校验实际 module
  bytes 的 sha256 与 descriptor 一致。若需远程传输 Wasm 字节，另开版本化 artifact schema，不改变
  既有 Schema-4/六 RPC 字段。
- **实例生命周期**：build 时编译 Module 一次并在评测期复用；每次调用新建 Store，保证 fuel、线性
  内存和调用状态干净，禁止每轮重新 Instantiate。`NativeExecutable.handle` 继续只表示后端 device
  executable，Host Module/Store 归 `MetalV2Executor.host_runtime`（或对应 backend executor）。
- **资源公式**：fuel 使用 `base + a*num_seqs + b*total_tokens + c*max_items` 随 capacity 缩放；
  线性内存页数按“直接映射输入 + workspace 有效 span + planner 私有栈/堆余量”计算。系数由
  envelope 给出上限，V3 用最大规模 KDA planner 的 p95 反推并冻结，engine 的 ResourceLimiter 落实。
- **doctor 与拒答**：`scripts/doctor.py` 检查 clang/lld/Wasm runtime，descriptor 记录 `clang
  --version` 首行与 engine 的确切版本。缺工具链时 `host/tiling.c` 返回 typed
  `backend_capability_unavailable`，JSON linear 不受影响；不是基础设施 crash，也不怪候选源码。
- 使用维护中的 Wasm engine；V3 先对本机 toolchain/engine 做安全与性能 probe。fuel 不替代现有外部
  watchdog。projection 只做路径/角色/危险语法静态预检，真正的 Clang 编译只在有界 worker 的
  `build()` 内发生，编译器本身不放进控制机 prepare 攻击面。
- probe 不通过则 V3 blocked，**不得退回候选原生 `.so/.dylib`**，也不得退回计时圈内的自研 Python
  解释器。测试里的 Python KDA planner 只能作独立 oracle，不能进入 candidate/timed path。

Metal 与 NPU simulator 执行**同一份 Wasm module 和 ABI**。未来真实 NPU backend 可将同一受限 C99
语义 lower 到其 host tiling ABI，但必须另做差分与安全验收，不能声称 Wasm artifact 可直接部署到
CANN/清微。

### 3.4 生命周期、计时与重叠测量

- 只读 symbols/attrs 的计划可以保持 `input_dependent=false`，按现有规则缓存。
- 只要读取任何 tensor value，就必须：
  `input_dependent=true + preprocesses_inputs=true + inside_timing=true`。
- 输入驱动计划在每次 correctness trial 和每个计时迭代执行；不能像 KDA `prepare` 一样提前到
  计时外。workspace 初始化所有权按每块 buffer 互斥声明：
  - `host`：CPU 清零（若声明 zero）并写有效 span，清零计入 host 时间，device 不再 fill；
  - `device_zero`：host 绝不写，fill 与 device stages 编进同一个 CommandBuffer；
  - `none/poison`：谁都不清，用 correctness poison trial 抓未初始化依赖。
  旧的 `WorkspaceUse.zero` 布尔值按显式兼容规则映射；host 写与 device_zero 同块冲突在 envelope
  阶段拒绝，不再等到运行时被覆盖。
- **先修现有 `host_inside_timing` 路径再计时**：只读输入不能每轮重拷；旋转集的 `BufferArg`
  槽位在计时圈外预构建，计时圈内只更新有效 span 和 resolved scalar bytes。目标是一次 Wasm 调用
  + 一次 GPU CommandBuffer 提交；真实开销以本机测量为准，不写未复现的固定微秒数。
- **audit 与 timed 分离**：correctness 和首次 warmup 记录 input ranges、span digest、LaunchGeometry
  与实际 dispatch；timed 迭代只累计 host/device 时间，不计算 sha256、不生成 JSON，和 device 侧只在
  warmup 记录 dispatch 的现有边界一致。
- 初版不做输入驱动 schedule cache。以后若增加，cache key 必须含 candidate + target + resolved
  shape + `evaluation_input_digest`；fresh/unseen 输入不能沿用 tune 的缓存，cold/miss 成本必须计入。
- Host Program 的 wall time（含 Wasm 调用和有效 span 拷贝）单独记录，同时包含在 candidate 的总计时
  中；device time 不能拿来替代 end-to-end 时间。
- 初版采用严格串行 `host -> device -> synchronize`，这是保守的 E2E 上界，不假装模拟 serving 的
  CPU/GPU overlap。V5 可增加 ring-buffer overlap 模式：host 准备下一组 rotation set 时，GPU 执行上一组，
  但所有 host work 仍计入端到端 wall，candidate/baseline 必须使用同一模式和同一测量根。
- `device_kernel_ns` 与 `host_tiling_wall_ns` 是**诊断分解**，不是第二套发布门。可以为目标策略增加
  host-overhead 提示/预算，但不得用 device-only 达标或任意固定的 20us/15% 规则替代 E2E verdict。

### 3.5 HostRunEvidence

新增本地 v2 证据，不改六 RPC/HMAC/BlobChannel：

```text
candidate_sha256 / program_sha256 / assembled_host_sha256 / wasm_module_digest
backend_id / target_id / native_digest
input_reads[]      = name, dtype, elem_off, count, sha256(实际读取区间)
workspace_writes[] = name, offset, count, dtype, nbytes, sha256(valid span bytes)
unwritten_policy   = poison-sentinel / declared-zero（逐项记录）
resolved_scalars   = 实际绑定值与实际 stage-active 集合
resolved_envelope_digest / launch_geometry_digest（逐 case/trial）
input_dependent / preprocesses_inputs / inside_timing
host_status / typed_error（非 0/trap/fuel 时无部分提交）
host_tiling_wall_ns / device_kernel_ns / end_to_end_wall_ns / timing_mode
simulated / publication_allowed
```

Dyn-1 的 frozen geometry envelope 在计算候选 artifact-set digest 前进入 artifact set；输入驱动的
实际 `LaunchGeometry` 是 **evaluation evidence**，不能事后回写并改变候选身份。Dyn-0 继续使用现有
冻结几何与静态 parity。

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
- **未在本阶段构造**：L2 capacity 尾部 sentinel 与 `cu_seqlens[num_seqs+1]` 越界预取（需要 V5 的
  capacity 缓冲与 poison 尾部；L1 的 `cu_seqlens` 是普通输入，越界读落在 guard 区、不能作为
  “读到声明外数据”的测试）；Host Program 相关项（读未声明输入、host 写越界、input-dependent
  host work 被移出计时）属于 V3；“resolved grid 超 envelope”属于 Dyn-1（V4），Dyn-0 由静态 launch
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
验收在 `tests/shared/test_varlen_npu_sim.py`。V2 尚无 Wasm Host Program，所以 parity 是两条**独立**
Python 路径之间的差分：`metal_device_plan()`（`piece_handoff.metal` 规划 stage 的逐行转写）与
`npu_host_plan()`（前缀和 + 累加 piece 大小的另一份实现），分别由 `metal_adapter()`（静态
grid、一项一个 TG 槽位）与 `npu_adapter()`（host-neutral 规划 + 有限 core 调度）输出。同一
`HostTarget` 桩下要求两者的 work-list bytes、依赖边与认领逐字节/逐边一致；真实异 target 下还允许
core 分配与 launch 几何不同。parity 本身另有负对照（字节对调、单 bit 翻转、丢边、换 max_split、
重复认领、有项未认领）证明它真的会抓到漂移，而不是比较同一个函数。真正的 Wasm module parity
（同一 module 在两个 adapter 上的有效 span 与 resolved scalar）属于 V3。

事件图不能只做对理想拓扑排序有利的执行：

- **Adversarial Core Scheduler**：在有限 core 槽位中故意优先调度带 `wait_flag` 的 consumer，注入
  consumer-first、producer 反序和驻满等待者等排列，检测等待环、死锁和活锁；只证明“顺序执行恰好
  能过”不算通过。
- **Alignment Clobber Check**：按 target 声明的 DMA 对齐粒度（例如 32B/512B，不把一个数字当成所有
  NPU 的通用事实）模拟写回。非对齐 partial tail 若未声明 masked write 或 read-modify-write 策略，
  必须判定可能覆盖下一条 TND 序列的开头；不能用 GPU 的逐线程 predicate 语义代替 NPU DMA 语义。

**Parity 分两层**：V2 已实现第一层——同一个 `HostTarget` 桩输入时，两条独立 planner
（`metal_device_plan`/`npu_host_plan`）给出的 work-list bytes、依赖边与认领必须逐字节/逐边一致；
V3 再加 Wasm module 层——同一 module 在两个 adapter 上的有效 span 与 resolved scalar 逐字节一致。
使用各自真实 target 核数时，work list 的 core 分配和 launch 可以不同，只要求 item 全集相同、每项
恰好被一个 core 认领、依赖边等价，不要求 `num_ctas` 或 grid 相同。

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
- varlen 题的 `cases.names` 必须覆盖 §5.1 的 case 域且 `canonical` 排第一（运行时用第一个 case 做
  benchmark）；这是 harness 对下限的校验，题目可以追加 case。

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
  - **native descriptor 头部空间**：四阶段 fixture 的 descriptor 实测 7980 B / 8192 B 上限
    （`kernels` 表把 kernel 0 的 bindings 在顶层重复了一份）。再添一个 stage/绑定会让**合法**候选在
    worker 里以 `BundleError: descriptor is too large` 崩溃（基础设施错误冒充结论）。本阶段不放宽
    闭集、不改 descriptor 形状（那会改 native digest），只把限额钉进 fixture 的 GPU 测试；扩大头部
    空间需要单独的 epoch 决定。
- 证据边界：`code_import_closure` 未变（V2 只改了 `tasks/varlen_kda/opspec.json` 与不在闭包内的
  `tcenv/backends/metal/v2_runtime.py`、tests）；P2 归档确认证据（14 个域实例、0 timing-blocked、
  activation generation=1）用确定性重放复核仍通过。P2 的 `verify_baseline` 所钉的 45 文件 start
  surface 在 V2 之前（HEAD=0557ef3）就已漂移 4 个文件 + 活计划，属既有状态，需要按安静窗口流程重录；
  V2 没有新增该 surface 的漂移。varlen task pack digest 因 opspec 变更而前进（varlen 评测身份随之
  更新），没有改写任何已归档证据。

### V3：受限 C99 / Wasm Host Program 与 NPU parity（CPU 先行，GPU 另授权）

- 完成 C99 子集、标准 Clang + LLVM `lld/wasm-ld`、正规 Wasm engine、线性内存/import/fuel 和外部
  watchdog 的选型与 probe；在 `pyproject.toml` 声明冻结后的 engine 依赖。probe 失败则 blocked，
  不加载原生库、不退回 Python AST、不自研 object 重定位加载器。
- 根据 OpSpec/geometry 自动生成强类型 `host_contract.h` 与 `assembled_host_sha256`；小输入直接映射，
  大输入只给元数据并走白名单区间 read import；实现单连续 span + poison 尾部和返回码原子提交。
- Module build 一次复用、每调用新 Store，executor 独持 `host_runtime`；descriptor 记录
  `{assembled_host_sha256, wasm_sha256, engine/compiler 确切版本}` 并进入 native digest，不把 live
  module 当身份，不新增 Schema-4 entry，不滥用 `BuildOutcome.payloads`。
- 实现 `host/device_zero/none(poison)` 三态初始化；修复只读输入重拷、计时圈内 `stage_args` 重建和
  zero-fill 双 CommandBuffer，目标一次 Wasm + 一次 GPU 提交。
- 用候选侧 C99 KDA planner 验证 heap、二分、LPT、位打包和自定义整数/定点成本模型；Python 版本仅作
  测试 oracle，不进 candidate/timed path；projection 只做静态预检，真正 Clang 编译只在有界 worker。
- 题面给完整 `host_contract.h`、C99 子集和返回码；quick 路径可单独编译/试跑 Host Program；提供最小
  host+device 示例，并用脚本化假候选根据编译诊断修复一处错误（离线，不花模型费）。
- `scripts/doctor.py` 检查工具链；缺依赖返回 `backend_capability_unavailable`，JSON linear 不受影响。
- audit 采 digest，timed 只计时；fuel/内存按 capacity 公式缩放并用最大规模 p95 冻结系数。

验收：同一 HostTarget 桩下 Wasm 输出逐字节一致；真实异 target 下 item 全集/唯一认领/依赖边等价，
core 分配和 launch 允许不同。HostRunEvidence 可重放；Schema-4 不放宽；记录复杂 planner 的
median/p95 host 时间，并在 V6 前冻结 target-specific 诊断预算。

### V4：Dyn-1 动态 grid/stage skip（可选优化，实测门）

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

### V6：真实规模性能与搜索（单独 GPU/付费授权）

- D=128、H=64/96、fixed/mixed/uniform 的 KDA 规模；
- 候选至少有一种真正的 chunked/matrix 实现，naive 单线程只作对照；
- host+device end-to-end 与 vendor/独立 baseline 同边界；逐档 fresh confirmation 与独立 fallback；
- 实测空 TG/空 stage 占比，决定 Dyn-1 是否值得启用；不能用理论百分比代替测量；
- 报告 host/device 分解，但不以 device-only 或固定 host 预算替代 E2E 发布谓词；
- 只有安静、fresh、unique publication predicate 全部满足才发布；
- NPU 模拟数字永不参与比较。

---

## 9. 文件级改动地图

| 层 | 文件/位置 | 改动 |
| --- | --- | --- |
| OpSpec | `tcenv/opspec/model.py`、`opspec/v2.py`、schema | varlen 校验、profile、L2 shape/capacity |
| 身份 | `portfolio/domain.py`、`taskcontract`、validation | 三层身份、dispatch/fresh seed/input replay |
| 题面/可写性 | `tcenv/task_presentation.py`、协议参考、quick/self-check | 完整 `host_contract.h`、C99 规则/返回码、Host-only 编译试跑、最小 host+device 示例与离线修复测试 |
| Host Program | `tcenv/hostprogram/`（不引入 Metal import）、`pyproject.toml`、`scripts/doctor.py` | 受限 C99、自动 header、大小输入分流、Wasm ABI、Module 复用/每调用 Store、capacity fuel/内存公式、有效 span；标准 lld 与 engine 依赖和 typed 拒答 |
| 构建身份 | `remote/artifact_builder.py`、`artifact_bundle.py`、`execution_geometry.py` | `NativeExecutable.descriptor["host_program"]` 进入 native digest；executor 独持 `host_runtime`；不改 Schema-4 闭集，不滥用 payloads |
| Portfolio 解耦 | `tcenv/portfolio/candidate.py` | 改为调用 `tcenv.hostprogram` 中立校验器，去除对 `backends.metal.v2_runtime` 的反向 import |
| Metal | `backends/metal/v2_runtime.py`、`track_adapter.py` | Dyn-0 静态路径、possible-workspace 并集分配、有效 span、host evidence、单 CommandBuffer 计时路径 |
| 生命周期 | `taskcontract/portfolio_runner.py`、`track/lifecycle.py` | input-dependent prepare、三态 workspace 初始化、audit/timed 分离与计时边界 |
| 几何（可选 Dyn-1） | `track/geometry.py`、backend stage binding | `ResolvedEnvelope`、独立 runtime_scalars/max_grid、逐 case `LaunchGeometry`、动态 output coverage 与逐 case parity |
| 任务 | `tasks/varlen_kda`、`tasks/varlen_scale` | 分段 case、V2 workspace 容量公式（`varlen_kda`）、真实规模档 |
| Metal 测试 | `tests/backends/metal/` | V2：`test_varlen_dyn0_gpu.py`（两类 fixture + 新负例）、`test_varlen_workspace.py`（容量/分配范围，CPU）、`test_varlen_report_identity.py`（错误报告仍可指纹）；V1 的 `test_varlen_gpu.py` 保留 |
| Metal fixture 数据 | `tests/support/golden/metal/varlen/{prep_chain,piece_handoff}.metal` | 候选拥有的固定源码，不是 harness 规则 |
| 共享测试 | `tests/shared/test_opspec_varlen.py`、`test_varlen_tasks.py`、`test_varlen_npu_sim.py` 等 | 合同、身份、容量公式、固定 shape 兼容、模拟器与负例 |
| NPU 模拟 | `tests/support/npu_sim/varlen_worklist.py` | core/UB-L1/事件图/对抗调度/DMA 对齐/两层 parity；生产组合根不 import，全部 simulated |

每阶段开始先重新计算 `code_import_closure`，闭包改动集中处理并按授权重录受影响证据；不要假设
某个文件“上次不在闭包，这次仍不在”。

---

## 10. 完成定义

只有以下条件全部满足，才能说“GPU/NPU 通用的 varlen 机制已完成”：

- [ ] 固定 shape 身份、既有 linear bundle、单文件、R5/远程协议均未退化；
- [ ] 同一 ABI 下任意合法 `cu_seqlens` 正确，写死 canonical 边界会被拒绝；
- [ ] 候选可用受限 C99/Wasm Host Program 实现 KDA 级 planner，harness 没有内置任务调度器，
  计时圈内没有 Python AST 或不可信原生库；
- [ ] Metal 真机验证 work-list、partial tail、多 stage 跨核 handoff、逐段 final_state 与全部作弊负例，
  默认路径不依赖跨 TG 自旋等待；
- [ ] Dyn-0 静态几何与静态 parity 可独立完成正确性；若启用 Dyn-1，则 `ResolvedEnvelope` 与逐 case
  `LaunchGeometry`、possible stages 编译/绑定、动态 parity 和 output coverage 全部可审计，且有实测
  数据证明空调度优化值得；
- [ ] L1 canonical 身份不碰撞，L2 total/count 变化有 capacity/resolved-launch/dispatch/fallback 的完整语义；
- [ ] 所有输入驱动 host work 进入计时与证据，KDA prepare 口径没有被偷运进来；
- [ ] NPU simulator 复用同一 Wasm module，具备 consumer-first 对抗调度与 DMA 对齐踩踏检查；同 target
  输出逐字节一致，异 target item 全集/唯一认领/依赖边等价，且所有结果
  `simulated=true/publication_allowed=false`；
- [ ] 真实性能只在 Metal 真实设备、真实规模、同边界 baseline、fresh quiet confirmation 后发布；
- [ ] Qingwei NPU-P0 仍如实 blocked，Ascend/CANN 参考资料未被写成生产能力。

在此之前，只能分别宣称“V2a 固定分布切片”“L1 任意分段正确性”“V2 Dyn-0 Metal work-list/handoff
机制正确”“NPU 协议/调度模拟通过”，不能统称完整 varlen 已支持。V2 的 Metal 证据只证明机制，不构成
KDA 性能结论；NPU 结果仍固定 `simulated=true`/`publication_allowed=false`。
