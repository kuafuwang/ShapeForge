# varlen 与输入驱动 Host Program：GPU / NPU 跨后端实施计划

状态：**设计冻结候选；V2a 有限切片已实现，完整方案未实施**。

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
Metal 正/负例保留为回归夹具，但不再把 V2a 称为完整 varlen 支持。

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
4. **resolved scalar/动态 stage 提前到 V2**。即使 L1 的 total/count 固定，分段也会产生不同的
   chunk 数、活跃 core 数和 handoff stage 需求；L1 只是 resolved total/count 等于 capacity 的特例。

审核建议中的“device kernel 必须达标 + host 时间低于固定预算”**不作为新的发布谓词**：纯 device
时间可诊断，host 时间也可设反馈预算，但发布仍只能使用同一 E2E wall 与现有 unique predicate，
不能用 device-only 达标掩盖端到端回退。

### 0.5 第二轮代码契约审核后的落地修订

对照当前实现再次核对后，进一步冻结以下衔接规则，避免计划与现有构建/证据门冲突：

1. 几何明确拆成构建期 **`ResolvedEnvelope`** 与逐 case **`LaunchGeometry`**；旧静态几何的
   `selection_parity` 行为不变，动态 launch 走新的逐 case parity，而不是放松旧审计。
2. Wasm 使用标准 Clang/lld 与正规 runtime，不自研 object 重定位加载器；`.wasm` 不加入 Schema-4
   `ArtifactBundle.entries`，也不偷塞进只服务 bundle entries 的 `BuildOutcome.payloads`。本地以
   `NativeExecutable.descriptor["host_program"] + live handle` 绑定 module digest 与执行对象，
   远程传输另开版本化 artifact schema。
3. `host_inside_timing` 路径先修复只读输入重拷、计时圈内重建参数和 zero-fill 双 CommandBuffer
   问题，再测 Host Program；双提交的成本以本机实测为准，不引用未复现的固定微秒数。
4. L1 保留 canonical `cu_seqlens` 在 `SemanticInstanceV2` 指纹中，避免现有 SemanticDomainPlan
   身份碰撞；L2 capacity 实例才将其置 `None` 并启用独立 workload profile。
5. 静态 `sizes.seq_lens` 继续要求每段 ≥1；零长度只由运行时 `cu_seqlens` case 构造。

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

统一的 Capacity + Resolved 模型从 V2 就落地，L1 是 `resolved_total_tokens == capacity_total_tokens`
且 `resolved_num_seqs == capacity_num_seqs` 的特例；L2 再放开 total/count。L2 不再把
`total_tokens/num_seqs` 只当编译期常量：

- **capacity shape**：`max_total_tokens/max_seqs`，用于 buffer/workspace 分配和 launch envelope；
  该档的 `SemanticInstanceV2.cu_seqlens=None`，semantic identity 不再绑定某个 canonical 分段；
- **resolved shape**：本次从输入张量形状与 `cu_seqlens` 得到的实际 total/count，具体 benchmark
  分布另由 `workload_profile_digest` 标识；
- **logical shape**：reference 和结果比较使用的有效区域；
- `TensorBinding`、ABI span、Metal buffer、workspace span、正确性 shape 检查都明确各自使用哪一套；
- output 的 padding 区必须通过独立 guard/poison 规则检查，不能用“ragged 比较”掩盖越界写；
- dispatch request 携带 resolved total/count 与 workload 身份；候选 guard 只能使用声明可读的元数据，
  exactly-one 规则、unknown/unmatched 和独立 fallback 不变。

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

逻辑 ABI 如下（字段布局在 V2 冻结，指针均为 **Wasm 线性内存内的偏移**，不是 host 地址）：

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

- `inputs` 只含声明为 `access=read` 的 global input；每个视图有 typed pointer、元素数与 dtype。
  `cu_seqlens` 等实际读取的输入按 ABI 解码并复制/映射进 Wasm 线性内存，相关成本计入
  `host_tiling_wall_ns`；未声明读取的大输入不做无谓复制。
- `shapes` 含 symbols/attrs、capacity 和 resolved total/count；`target` 只含 target profile 公开且
  进入身份的设备整数参数。
- `workspaces` 每项含 Wasm 内 typed pointer、capacity、元素 dtype 和候选填写的 `offset/count`。
  支持紧凑有效前缀，也支持 per-core offset + valid span；不要求按 max capacity 全量写。
- `resolved` 只含 geometry 预先声明的 scalar/grid/stage-active knob。L1 的 total/count 等于
  capacity，但 chunk 数、活跃 core 数、handoff stage 是否激活仍可由本次输入决定。
- 相同 candidate + input + target 必须得到相同输出。候选不能读 global output、原输入的写权限、
  host 指针、文件、网络、时钟、随机数或环境变量。

**有效 Span 规则**：`offset >= 0`、`count >= 0`、`offset + count <= capacity`，按元素 dtype 检查
字节数；可以返回 `count=0`。harness 只复制有效区间并记录其 digest。未写尾部在 correctness trial
和初始化/warmup 中填 poison/sentinel，device 必须按 item count/resolved shape 停止；一旦错误读取
尾部，正确性应失败，而不是靠全量零填掩盖。计时路径不为“保险”每轮全量清零 capacity。

### 3.3 C99 子集与 Wasm 沙箱

Host Program 使用可静态检查的 C99 子集：

- 允许普通函数、循环、条件、整数/浮点运算、位运算、固定大小数组和候选自己实现的 heap/LPT/
  range-cut；足以表达 KDA planner，不需要图灵不完备的“安全简化算法”。
- 只允许 `stdint.h`/`stddef.h`/`stdbool.h` 的纯类型能力；禁止 libc 调用、malloc/free、函数指针、
  inline asm、可变参数、文件/网络/进程/环境访问和未登记 host import。
- 编译目标为 Wasm，导出固定入口与线性内存；**不使用 WASI**。唯一可用 import 是 harness 预先
  登记的输入/target metadata 视图，所有读写都由 offset + capacity 校验。
- **工具链明确依赖**：使用标准 Clang + LLVM `lld/wasm-ld` 生成最终 module，并在 `pyproject.toml`
  声明维护中的 Wasm engine（V2 选型 probe 后冻结，例如 wasmtime）。当前 Xcode Clang 虽能生成
  Wasm object，但本机 PATH 没有 `wasm-ld/lld`，`.venv` 也没有 Wasm runtime，因此 V2 必须先补齐
  并验收这些依赖；不得自研 object 重定位/加载器来绕过它们。
- Wasm module 在 candidate build 阶段编译并进入 build evidence，执行时不按输入重新编译。编译器、
  Wasm engine、module digest、内存上限和 import 清单都进入构建身份。
- **Schema-4 兼容的 module 身份**：`.wasm` 不加入 `ArtifactBundle.entries`，避免破坏
  `candidate entries + manifest + execution_geometry` 闭集；也不放进 `BuildOutcome.payloads`，
  因为现有 payloads 只允许命名 bundle entries。本地构建把
  `{source_path, source_sha256, assembled_host_sha256, wasm_sha256, engine, compiler}` 写入
  `NativeExecutable.descriptor["host_program"]`，live Wasm module 与 Metal pipeline handle 一起由
  `NativeExecutable.handle` 持有，使 module digest 进入 `native_digest` 并接受现有交叉审计。
  是否新增独立 executable kind 在 V2 冻结；若需远程传输 Wasm 字节，另开版本化 artifact schema，
  不改变既有 Schema-4/六 RPC 字段。
- 使用维护中的 Wasm engine；V2 先对本机 toolchain/engine 做安全与性能 probe。线性内存有上限，
  fuel 提供确定性的指令预算，现有外部 watchdog 仍负责进程级终止；fuel 不替代 watchdog。
- probe 不通过则 V2 blocked，**不得退回候选原生 `.so/.dylib`**，也不得退回计时圈内的自研 Python
  解释器。测试里的 Python KDA planner 只能作独立 oracle，不能进入 candidate/timed path。

Metal 与 NPU simulator 执行**同一份 Wasm module 和 ABI**。未来真实 NPU backend 可将同一受限 C99
语义 lower 到其 host tiling ABI，但必须另做差分与安全验收，不能声称 Wasm artifact 可直接部署到
CANN/清微。

### 3.4 生命周期、计时与重叠测量

- 只读 symbols/attrs 的计划可以保持 `input_dependent=false`，按现有规则缓存。
- 只要读取任何 tensor value，就必须：
  `input_dependent=true + preprocesses_inputs=true + inside_timing=true`。
- 输入驱动计划在每次 correctness trial 和每个计时迭代执行；不能像 KDA `prepare` 一样提前到
  计时外。zero workspace 的 fill 与 host write 顺序必须保持“先 fill、后 host write、再 device”。
- **先修现有 `host_inside_timing` 路径再计时**：只读输入不能每轮重拷；旋转集的 `BufferArg`
  槽位在计时圈外预构建，计时圈内只更新有效 span 和 resolved scalar bytes。Host-written workspace
  与 device-zeroed workspace 分开管理：CPU 不写后者时，其 fill 可与 device stages 编进同一个
  CommandBuffer，避免“fill 等一次、host write 后再等一次”的双同步。目标是一次 Wasm 调用 + 一次
  GPU CommandBuffer 提交；真实开销以本机测量为准，不写未复现的固定微秒数。
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
input_reads[]      = name, dtype, resolved_shape, sha256(raw bytes)
workspace_writes[] = name, offset, count, dtype, nbytes, sha256(valid span bytes)
unwritten_policy   = poison-sentinel / declared-zero（逐项记录）
resolved_scalars   = 实际绑定值与实际 stage-active 集合
resolved_envelope_digest / launch_geometry_digest（逐 case/trial）
input_dependent / preprocesses_inputs / inside_timing
host_tiling_wall_ns / device_kernel_ns / end_to_end_wall_ns / timing_mode
simulated / publication_allowed
```

Frozen geometry envelope 在计算候选 artifact-set digest 前进入 artifact set；输入驱动的实际
resolved launch 是 **evaluation evidence**，不能事后回写并改变候选身份。

---

## 4. Device 与执行几何能力

### 4.1 `ResolvedEnvelope` + 逐 case `LaunchGeometry`

现有 `resolve_geometry()` 在构建前没有输入时就把 grid/active/scalars 求成静态整数，
`MetalV2Runtime._bind()` 只绑定 active stages，`selection_parity()` 又要求实际 dispatch 与静态
active set 逐字相等。若直接让 Host Program 改 grid/stage-active，同一 `correctness()` 的不同 case
必然撞上现有 parity。因此几何显式拆成两层：

#### 构建期：`ResolvedEnvelope`

- 对 `ExecutionGeometry` 中所有**静态可能激活的 `possible_stages`** 做名称、资源、grid 上限、
  scalar 合法域和 `when` 依赖校验，进入 CanonicalBundle/artifact-set digest；
- 对 possible_stages 全部完成编译与 `bind_stage_arguments`，不能只绑定构建期碰巧 active 的子集；
- possible coverage 只证明“存在某个 stage 能写 result”，不替代每次 launch 的 output coverage。

#### 每次 Host Program 后：`LaunchGeometry`

- 由 `ResolvedEnvelope + HostResolved + 本次 resolved shape` 求值实际 grid、derived scalars 和
  stage-active 集合；L1 的 resolved total/count 恰好等于 capacity，但 chunk 数、活跃 core 数和
  handoff stage 仍可随 case 改变；
- 检查 grid/scalars 落在 envelope 合法域内，并针对实际 active set 做 result-tensor 写入位掩码
  检查；未激活 stage 零 dispatch；
- 预构建的 `BufferArg` 槽位复用，只更新本次 resolved scalar bytes。

#### 逐 case/trial parity

- 旧静态几何的 `selection_parity` 与 `_dispatch_mismatches` 行为**保持不变**；
- 新动态路径逐 case/trial 核对：
  `observed_dispatch == evaluate_launch(envelope, host_run.resolved)`，并再次检查 envelope 上限；
  worker 选择错误 provider、偷偷执行 inactive stage、漏 stage 或修改 grid 都是 mismatch；
- `LaunchGeometry` 是 evaluation evidence，不回写并改变已冻结的 candidate/artifact identity。

L2 只是把 resolved total/count 放开到 capacity 以下，复用同一套两层几何、绑定和证据，不再推倒重改。

### 4.2 工作项与存储层级

合同不规定 item 的位布局，但候选必须能表达：

- token offset、valid rows、sequence/head、first/last、handoff source/destination；
- per-core item count 与实际 work-list digest；
- 小型 item table 复制到固定容量的 threadgroup/UB/L1，大表留 global workspace；
- 固定容量 scratch + 运行时 count，而不是未受控的动态 threadgroup 参数；
- chunk 内累计 log-decay、partial tail mask 和谓词化输出。

### 4.3 状态 handoff 与同步：Multi-stage 优先

允许候选选择：

1. 单 CTA/单核处理整条 `(sequence, head)`；
2. persistent work list，整链 LPT；
3. **长链拆分 + 多 stage/two-launch handoff（Metal 首选）**：stage 1 并行计算各 piece 的局部变换并
   写中间态，stage 2 规约/传递 state，stage 3 写最终结果；stage 之间用 Metal serial encoder 与
   `memoryBarrierWithScope_(MTLBarrierScopeBuffers)` 建立真实完成顺序；
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
- `single_sequence`：单段全长退化；
- `fresh_distribution`：fresh seed 派生的合法分段，与 tune 不相交。

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
- Metal stage binding：对 envelope 的全部 `possible_stages` 一次编译/反射绑定；静态 threadgroup
  容量 + 运行时 item count；resolved scalar 只走按名绑定的 4/8 字节整数参数，不绑 host pointer；
  动态 `when` 决定每次 case 的 stage-active 集合，并逐 case 做 output coverage。
- Metal 同步：stage 之间使用现有 serial encoder + `memoryBarrierWithScope_(MTLBarrierScopeBuffers)`；
  不以跨 TG 自旋等待作为验收或默认实现。
- 计时实现：旋转集参数槽位预构建，只读输入只恢复一次；Host-written 与 device-zeroed workspace
  分开，避免 zero-fill 双 CommandBuffer 和每轮重复构造 `stage_args`。

### 6.2 Metal 垂直切片候选

保留 V2a naive `known_good.metal` 作为最简正确性候选，另在测试 support 中提供**候选拥有的** KDA
work-list fixture：

- host planner 覆盖 range-cut、LPT 与 handoff cost；
- device 按 item table 执行，包含首/中/末 piece、partial tail、per-sequence final_state；
- 至少一个长序列被切分后，以 **stage 1 中间态 → barrier → stage 2/3 传递与写回** 完成 fp32
  handoff 的正例；不得依赖 consumer/producer 同 launch 自旋；
- resolved scalar 控制活跃 core 和 handoff stage 的正例，以及未激活 stage 不得执行的负例；
- 小 item table 的 threadgroup 缓存与超大表 global fallback；
- 与 naive oracle 的 output/final_state 差分，以及 §5.2 全部负例。

该 fixture 证明机制可表达，不是 harness 为所有 varlen 任务指定的标准算法。

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

事件图不能只做对理想拓扑排序有利的执行：

- **Adversarial Core Scheduler**：在有限 core 槽位中故意优先调度带 `wait_flag` 的 consumer，注入
  consumer-first、producer 反序和驻满等待者等排列，检测等待环、死锁和活锁；只证明“顺序执行恰好
  能过”不算通过。
- **Alignment Clobber Check**：按 target 声明的 DMA 对齐粒度（例如 32B/512B，不把一个数字当成所有
  NPU 的通用事实）模拟写回。非对齐 partial tail 若未声明 masked write 或 read-modify-write 策略，
  必须判定可能覆盖下一条 TND 序列的开头；不能用 GPU 的逐线程 predicate 语义代替 NPU DMA 语义。

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

- 实现 §2.1–2.3 与 §5.1/5.2 的 correctness case；
- canonical profile 用于 benchmark，permuted/skewed/fresh 用于正确性；
- L1 保留 canonical `cu_seqlens` 在 SemanticInstanceV2/fingerprint/SemanticDomainPlan 中，避免同
  total/count 的分布撞身份；permuted/skewed/fresh 只作为 Evaluation Cases，由 input digest 区分；
  L2 capacity 档才将实例 `cu_seqlens` 置 `None` 并启用独立 workload profile；固定 shape 投影不变；
- statement 展示 canonical profile，同时声明其他分段会测试；
- output-only final_state 负例、hardcoded-canonical 负例进入 Metal GPU 测试。

验收：同一 ABI 下换分段仍正确；写死边界失败；每个实际 case 都能由 input digest 重放。
V0/V1 涉及 pack closure 的修改集中完成后，**只重录一次**受影响 P2 证据。

### V2：受限 C99 / Wasm Host Program（CPU）

- 完成 C99 子集、标准 Clang + LLVM `lld/wasm-ld`、正规 Wasm engine、线性内存/import/fuel 和外部
  watchdog 的选型与 probe；在 `pyproject.toml` 声明冻结后的 engine 依赖。probe 失败则 blocked，
  不加载原生库、不退回 Python AST、不自研 object 重定位加载器。
- 根据 OpSpec/geometry 自动生成强类型 `host_contract.h` 与 `assembled_host_sha256`；实现输入/输出
  资源视图、有效 span + poison 尾部、resolved scalar/stage-active 与输入/输出校验。
- Wasm module 以 `NativeExecutable.descriptor["host_program"] + live handle` 进入 native digest；
  不新增 Schema-4 bundle entry，不滥用 `BuildOutcome.payloads`；远程字节传输另开版本化 schema。
- 实现 `ResolvedEnvelope`：全部 possible stages 编译/反射绑定；实现逐 case `LaunchGeometry`、动态
  output coverage 和新 parity，旧静态 parity 不变。
- 保持 JSON linear 与单文件字节/行为不变；生命周期 input-dependent/timing/evidence 接线。
- 修复 `host_inside_timing` 的只读输入重拷、计时圈内 `stage_args` 重建和 zero-fill 双 CommandBuffer；
  Host-written 与 device-zeroed workspace 分离，目标一次 Wasm + 一次 GPU 提交。
- 用候选侧 C99 KDA planner 验证 heap、二分、LPT、位打包和自定义成本模型都可表达；Python 版本仅作
  测试 oracle，不进 candidate/timed path；preflight 拒绝危险语法/import、未声明资源、错误 span
  和非法 resolved scalar。

验收：同一 Wasm planner 在两个 backend adapter 上产生逐字节一致的有效 span 与 launch 记录；
HostRunEvidence 可重放；Schema-4 既有校验不放宽；记录复杂 planner 的 median/p95 host 时间，并在
V6 前冻结 target-specific 诊断预算，不能只报 device time 或靠解释器硬撑。

### V3：Metal KDA work-list 垂直切片（需显式 GPU 授权）

- 完成 §6 的 host+device fixture；
- range-cut、LPT、长链多 stage fp32 handoff、partial tail、threadgroup/global item table 都有正例；
- stage 间使用真实 buffer barrier，resolved scalar/stage-active 与实际 dispatch 一致；
- §5.2 负例全部通过“必须拒绝”验收；
- 与 naive varlen KDA oracle 在 canonical/permuted/skewed/boundary 上对齐 output/final_state；
- host wall time 进入 candidate 时间，逐 case/trial 的 `observed_dispatch` 与
  `evaluate_launch(envelope, host_resolved)` 一致，旧静态 parity 不退化。

验收只能标“Metal 机制正确”，还不能标“达到 KDA 性能”。

### V4：NPU 协议模拟 parity（不需要真实 NPU）

- V2 的同一 Host Program 在 NPU simulator 产生相同 work list；
- core 分配、事件依赖、handoff flag、UB/L1 容量与尾块规则可审计；
- consumer-first/逆序调度可检测等待环，非对齐 DMA 写回可检测相邻序列踩踏；
- 所有报告 `simulated=true/publication_allowed=false`；
- 模拟不进入 Metal 的发布链，也不解除 Qingwei NPU-P0。

验收：GPU 与 NPU simulator 的 Host Program/Worklist/ResolvedLaunch 差分为零；证据类型严格区分。

### V5：L2 runtime total/count、capacity 与 overlap（新模型阶段）

- 在 V2 已统一的 resolved scalar/stage-active 之上，实现 logical/resolved/capacity shape 与
  max-capacity ABI；L1 只是 resolved total/count == capacity，无需重写 Runtime；capacity 实例的
  `SemanticInstanceV2.cu_seqlens=None`，具体 benchmark 分布由 workload profile 标识；
- dispatch request 携带 resolved total/count，exact one-of-N 与独立 fallback 保持；
- padding/guard、`cu_seqlens` capacity 尾部 sentinel、empty batch/empty sequence、超 capacity typed
  refusal；
- Host Program cache 若加入，必须绑定 evaluation input digest 并记录 cold/miss；
- 可加入 ring-buffer host/device overlap 测量，serial/overlap 分别标识并使用同一测量根；
- 用同一候选在连续多个不同 total/count 调用上验证无重编译语义（是否重编译由候选与缓存身份决定，
  证据必须如实记录）。

验收：continuous batching 场景不再依赖为每个 total/count 建一个静态档；L1 证据不冒充 L2。

### V6：真实规模性能与搜索（单独 GPU/付费授权）

- D=128、H=64/96、fixed/mixed/uniform 的 KDA 规模；
- 候选至少有一种真正的 chunked/matrix 实现，naive 单线程只作对照；
- host+device end-to-end 与 vendor/独立 baseline 同边界；逐档 fresh confirmation 与独立 fallback；
- 报告 host/device 分解，但不以 device-only 或固定 host 预算替代 E2E 发布谓词；
- 只有安静、fresh、unique publication predicate 全部满足才发布；
- NPU 模拟数字永不参与比较。

---

## 9. 文件级改动地图

| 层 | 文件/位置 | 改动 |
| --- | --- | --- |
| OpSpec | `tcenv/opspec/model.py`、`opspec/v2.py`、schema | varlen 校验、profile、L2 shape/capacity |
| 身份 | `portfolio/domain.py`、`taskcontract`、validation | 三层身份、dispatch/fresh seed/input replay |
| 题面 | `tcenv/task_presentation.py`、协议参考 | canonical profile + 可变分段 + Host Program 合同 |
| Host Program | `tcenv/hostprogram/`（不引入 Metal import）、`pyproject.toml` | 受限 C99、自动生成 `host_contract.h`、Wasm ABI、线性内存/import/fuel、有效 span；标准 lld 与 engine 依赖 |
| 构建身份 | `remote/artifact_builder.py`、`artifact_bundle.py`、`execution_geometry.py` | `NativeExecutable.descriptor["host_program"] + live handle`；不改 Schema-4 闭集，不滥用 payloads |
| Portfolio 解耦 | `tcenv/portfolio/candidate.py` | 改为调用 `tcenv.hostprogram` 中立校验器，去除对 `backends.metal.v2_runtime` 的反向 import |
| Metal | `backends/metal/v2_runtime.py`、`track_adapter.py` | possible stages 编译/绑定、有效 span、resolved scalar、host evidence、单 CommandBuffer 计时路径 |
| 生命周期 | `taskcontract/portfolio_runner.py`、`track/lifecycle.py` | input-dependent prepare 与计时边界 |
| 几何 | `track/geometry.py`、backend stage binding | `ResolvedEnvelope`、逐 case `LaunchGeometry`、动态 output coverage 与逐 case parity |
| 任务 | `tasks/varlen_kda`、`tasks/varlen_scale` | 分段 case、必要 workspace、真实规模档 |
| Metal 测试 | `tests/backends/metal/` | Host Program、work-list、handoff/tail、GPU 负例 |
| 共享测试 | `tests/shared/test_opspec_varlen.py` 等 | 合同、身份、固定 shape 兼容、负例 |
| NPU 模拟 | `tests/support` 的模拟组合根 | 同 Wasm/core event/alignment/evidence；生产组合根不 import |

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
- [ ] 几何拆成 `ResolvedEnvelope` 与逐 case `LaunchGeometry`，possible stages 全部编译/绑定，
  动态 parity 与 output coverage 可审计，旧静态 parity 不退化；
- [ ] L1 canonical 身份不碰撞，L2 total/count 变化有 capacity/resolved-launch/dispatch/fallback 的完整语义；
- [ ] 所有输入驱动 host work 进入计时与证据，KDA prepare 口径没有被偷运进来；
- [ ] NPU simulator 复用同一 Wasm module，具备 consumer-first 对抗调度与 DMA 对齐踩踏检查，且所有结果
  `simulated=true/publication_allowed=false`；
- [ ] 真实性能只在 Metal 真实设备、真实规模、同边界 baseline、fresh quiet confirmation 后发布；
- [ ] Qingwei NPU-P0 仍如实 blocked，Ascend/CANN 参考资料未被写成生产能力。

在此之前，只能分别宣称“V2a 固定分布切片”“L1 任意分段正确性”“Metal work-list 机制正确”
或“NPU 协议模拟通过”，不能统称完整 varlen 已支持。
