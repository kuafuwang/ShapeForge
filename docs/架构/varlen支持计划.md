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
8. **解决 GPU/NPU 各写一套造成的语义漂移**：同一 HostProgramIR、workload、reference、case 和证据
   字段贯穿 Metal 与 NPU simulator，后端只负责执行与设备事实。
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
  ├─ host/tiling.json（既有 linear）或 host/tiling.py（新 Host Program）
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
- HostProgramIR 初版以语义一致和安全为先；若解释执行成为瓶颈，只能通过对同一 IR 做差分验证的
  native 优化解决，不能再发明一套简化 planner 来规避。

---

## 2. 后端中立合同模型

### 2.1 分离三种身份：语义、workload、实际输入

V2a 把 `cu_seqlens` 直接放进实例指纹，解决了“不同声明分布不能串证据”，但还把“声明的
benchmark 分布”和“本次运行时输入”混在了一起。新模型分开：

| 身份 | 内容 | 用途 |
| --- | --- | --- |
| `semantic_instance_digest` | dtype、rank、静态 shape 或 capacity、tensor ABI、attrs、数值合同 | 判断这是不是同一个算子 ABI |
| `workload_profile_digest` | 声明的 canonical `seq_lens`、profile id、usage、case-domain 版本 | 性能档、exact dispatch、fresh seed、候选覆盖计划 |
| `evaluation_input_digest` | 本次实际 `cu_seqlens` 字节、case、seed、必要的输入身份 | 重放、Host Program cache、审计这次到底跑了什么 |

规则：

- fixed-shape 实例不增加新键，既有 `semantic_instance_digest` 逐位不变。
- L1 的 canonical mixed/uniform/fixed 是不同 workload profile；同一 profile 下的 permuted/skewed
  correctness case 仍是同一 ABI 和同一 dispatch subject，但 `evaluation_input_digest` 不同。
- benchmark 证据必须绑定 canonical workload；correctness 证据还必须能审计实际分段。
- unseen profile 与 tune profile 按规范化 `seq_lens` 去重且互不相交；unseen key/master seed 不进入
  候选视图。
- 不允许只凭 `total_tokens/num_seqs` 判断两次 varlen 性能观测相同。

### 2.2 `cu_seqlens` 是运行时输入，不是候选常量

统一约束：

- `cu_seqlens` 是一维整数 input；第一阶段使用 `i32`，并显式检查 `total_tokens <= INT32_MAX`，
  需要更大范围时另开 int64 全链路阶段。
- 长度为 `num_seqs + 1`，首项 0、单调不减、末项等于本次 `total_tokens`。
- 允许长度 0 的序列（整批至少一个 token）；其 output 行为空，`final_state` 必须等于该序列
  `initial_state`。只在 canonical benchmark 中是否出现零长度由 profile 决定，correctness 必须覆盖。
- harness 在 host/device 执行前校验这些不变量；reference 永远读取实际输入，不能读取实例里
  另一份边界副本来代替它。
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

L2 不再把 `total_tokens/num_seqs` 只当编译期常量：

- **capacity shape**：`max_total_tokens/max_seqs`，用于 buffer/workspace 分配和 launch envelope；
- **resolved shape**：本次从输入张量形状与 `cu_seqlens` 得到的实际 total/count；
- **logical shape**：reference 和结果比较使用的有效区域；
- `TensorBinding`、ABI span、Metal buffer、workspace span、正确性 shape 检查都明确各自使用哪一套；
- output 的 padding 区必须通过独立 guard/poison 规则检查，不能用“ragged 比较”掩盖越界写；
- dispatch request 携带 resolved total/count 与 workload 身份；候选 guard 只能使用声明可读的元数据，
  exactly-one 规则、unknown/unmatched 和独立 fallback 不变。

---

## 3. 后端中立 Host Program

### 3.1 交付形式：复用 `host_tiling`，不增加 schedule

允许且仅允许一个 host entry：

| 文件 | 角色 | 语义 |
| --- | --- | --- |
| `host/tiling.json` | `host_tiling` | 既有 linear writes；字节与执行语义不变 |
| `host/tiling.py` | `host_tiling` | 新 Host Program；纯函数、输入驱动、可生成 work list 和 resolved scalars |

`host/execution_geometry.json` 仍是唯一几何声明。Host Program 不是另一个 planner registry，也不是
harness 选出的 KDA 算法；其源码属于 candidate artifact，摘要进入候选身份。

### 3.2 纯函数 API

逻辑接口如下（名字在实施时固定，语义先冻结）：

```python
def host_tiling(inputs, shapes, target):
    # inputs：只能读取声明为 access=read 的 global input，按名称惰性解码
    # shapes：实例 symbols/attrs；L2 还含 resolved total/count/capacity
    # target：target profile 中公开、进入身份的设备整数参数
    return {
        "workspaces": {
            "items": [...],          # 类型和元素数由 ABI 校验
            "item_counts": [...],
            "handoff_flags": [...],
        },
        "resolved_scalars": {"num_ctas": ...},  # 只能覆盖几何预先声明的 knob
    }
```

约束：

- 只能读声明输入，只能写声明 workspace；不能读写 global output、输入原缓冲区、host 指针、文件、
  网络、时钟、随机数或环境变量。
- 返回的 workspace 内容必须有确定 dtype/span；初版要求 host-written workspace 全量返回，避免未写
  区域依赖 poison/zero 的隐式语义。后续若允许 span write，必须使用同一套边界校验函数。
- `resolved_scalars` 只能来自 geometry 预先声明的名字、dtype 和范围；L1 不需要它，L2 才允许
  resolved grid/scalar。
- 相同 candidate + input + target 必须得到相同输出。每次实际读取和写入都进证据。

### 3.3 安全语言与后端中立 IR

不使用 Python `exec/eval`。 restricted-Python 仅作为候选书写前端，解析成后端中立 HostProgramIR：

- 支持 KDA planner 真正需要的纯函数、嵌套函数/闭包、默认参数、`if/for/while`、二分搜索、
  list/dict/tuple、list comprehension/generator、lambda、算术/比较/位运算、切片；
- 仅提供白名单 intrinsic：`len/range/min/max/sum/sorted/enumerate/reversed`、list/dict 方法、
  `heapq` 的 push/pop/heapify、少量 `math`；
- 禁止 import 其他模块、属性双下划线、class、异常逃逸、文件/网络/进程/反射、任意 host include；
- runaway 由现有外部 watchdog 处理，不用一个隐藏的小循环上限误杀合法 planner；
- AST/资源校验在 preflight 完成，输入相关执行错误规范化为 typed contract failure。

Metal 与 NPU simulator 执行**同一份 IR**。未来若提供 native host compiler，它只是同一 IR 的优化
执行器，必须与参考解释器做差分一致性验证；不能把性能优化变成另一套调度语义。

### 3.4 生命周期与计时

- 只读 symbols/attrs 的计划可以保持 `input_dependent=false`，按现有规则缓存。
- 只要读取任何 tensor value，就必须：
  `input_dependent=true + preprocesses_inputs=true + inside_timing=true`。
- 输入驱动计划在每次 correctness trial 和每个计时迭代执行；不能像 KDA `prepare` 一样提前到
  计时外。zero workspace 的 fill 与 host write 顺序必须保持“先 fill、后 host write、再 device”。
- 初版不做输入驱动 schedule cache。以后若增加，cache key 必须含 candidate + target + resolved
  shape + `evaluation_input_digest`；fresh/unseen 输入不能沿用 tune 的缓存，cold/miss 成本必须计入。
- Host Program 的 wall time 单独记录，同时包含在 candidate 的总计时中；device time 不能拿来替代
  end-to-end 时间。

### 3.5 HostRunEvidence

新增本地 v2 证据，不改六 RPC/HMAC/BlobChannel：

```text
candidate_sha256 / program_sha256 / backend_id / target_id
input_reads[]      = name, dtype, resolved_shape, sha256(raw bytes)
workspace_writes[] = name, span, nbytes, sha256(written bytes)
resolved_scalars   = 实际绑定值
geometry_envelope_digest / resolved_launch_digest
input_dependent / preprocesses_inputs / inside_timing
host_wall_ns / device_time（若后端可提供）
simulated / publication_allowed
```

Frozen geometry envelope 在计算候选 artifact-set digest 前进入 artifact set；输入驱动的实际
resolved launch 是 **evaluation evidence**，不能事后回写并改变候选身份。

---

## 4. Device 与执行几何能力

### 4.1 静态 envelope + resolved launch

`ExecutionGeometry` 声明：

- stage 顺序、kernel、允许 grid/threads 公式、静态 threadgroup bytes、workspace、when；
- persistent grid 的上限或静态公式；
- Host Program 可决定的 scalar/grid knob 及其范围。

L1 多数情况下 `grid=min(target_core_count, H*num_seqs)` 已可在现有表达式中求值，不需要动态几何。
L2 才由 Host Program 给出 resolved grid，并由 harness 检查不超过 envelope；实际 dispatch 记录必须
与 resolved launch 一致。

### 4.2 工作项与存储层级

合同不规定 item 的位布局，但候选必须能表达：

- token offset、valid rows、sequence/head、first/last、handoff source/destination；
- per-core item count 与实际 work-list digest；
- 小型 item table 复制到固定容量的 threadgroup/UB/L1，大表留 global workspace；
- 固定容量 scratch + 运行时 count，而不是未受控的动态 threadgroup 参数；
- chunk 内累计 log-decay、partial tail mask 和谓词化输出。

### 4.3 状态 handoff 与同步

允许候选选择：

1. 单 CTA/单核处理整条 `(sequence, head)`；
2. persistent work list，整链 LPT；
3. 长链拆分，单 launch 内 handoff；
4. 多 stage/two-launch handoff；
5. 纯 device scheduler 或其他算法。

跨核续传必须使用声明的 handoff state/flag workspace 和公开的原子/同步能力，明确 acquire/release
语义。自旋等待由外部 watchdog 限制；死锁是候选失败，不能变成 harness 默认重试。

`final_state` 始终以合同 dtype 和逐段终值验收；中间 handoff 若使用 bf16/fp16，只能作为候选数值
路径，由现有 tolerances + scaled_error 判定，不能宣称“等价 fp32”。

---

## 5. 正确性、作弊负例与性能口径

### 5.1 必测分段 case

每个 varlen 任务至少覆盖：

- `canonical`：声明的 fixed/mixed/uniform 分布；
- `permuted`：同 total/count，长度顺序改变；
- `skewed`：同 total/count，一条长序列 + 多个极短序列；
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
- host 写越界、读未声明输入、resolved grid 超 envelope、input-dependent host work 被移到计时外。

### 5.3 性能与发布

- fixed/mixed/uniform 各自独立计时、独立通过 1.10 gate；不能用 geomean 隐藏某一档回退。
- candidate 时间包含输入驱动 Host Program、workspace 清零、所有 stage 和 handoff；baseline 必须使用
  相同输入、output/final_state 责任和测量边界。
- `specialized_hit_speedup` 只统计生成 variant；`effective_end_to_end_speedup` 包含真实 fallback。
- KDA 的 prepare/launch 数字只能作算法参考，不能与我们的 end-to-end 数字直接比较。
- naive `float state[D*D]` 单线程 kernel 仅是正确性参考；真实性能验收使用 D=128、H=64/96 和
  KDA fixed/mixed/uniform 规模，不能由小档正确性推断可扩展性。
- NPU 模拟只证明协议/数据流/调度身份，不产生 NPU 时间、speedup、confirmed、catalog 或发布结论。

---

## 6. Metal 落地方案

### 6.1 运行时接线

- `tcenv/backends/metal/v2_runtime.py`：Host Program 输入惰性解码、workspace 全量写、host run
  evidence、resolved scalar 绑定。
- `tcenv/taskcontract/portfolio_runner.py`：由 plan 是否读输入派生 lifecycle，不再恒写
  `input_dependent=false`；输入驱动计划强制进入计时。
- `V2ArgSet`：只读输入快照、workspace ABI span 与实际写 digest；继续检查输入不可变和 guard。
- Metal stage binding：静态 threadgroup 容量 + 运行时 item count；resolved scalar 只走按名绑定的
  4/8 字节整数参数，不绑 host pointer。
- Metal atomics/同步：用公开 MSL 能力实现 acquire/release flag；不可用时明确报告能力不足，不能
  用无同步全局读写冒充 handoff。

### 6.2 Metal 垂直切片候选

保留 V2a naive `known_good.metal` 作为最简正确性候选，另在测试 support 中提供**候选拥有的** KDA
work-list fixture：

- host planner 覆盖 range-cut、LPT 与 handoff cost；
- device 按 item table 执行，包含首/中/末 piece、partial tail、per-sequence final_state；
- 至少一个长序列跨 threadgroup 的 fp32 handoff 正例；
- 小 item table 的 threadgroup 缓存与超大表 global fallback；
- 与 naive oracle 的 output/final_state 差分，以及 §5.2 全部负例。

该 fixture 证明机制可表达，不是 harness 为所有 varlen 任务指定的标准算法。

---

## 7. NPU 模拟落地方案

### 7.1 复用，而不是重新写一套

NPU simulator 使用相同的：

- OpSpec/WorkloadProfile/case 生成与 fp64 reference；
- HostProgramIR 与输入/Workspace 校验；
- work-list、item count、resolved scalar、resolved launch 和 evidence 身份；
- LPT/range-cut/handoff/tail/zero-length 负例。

模拟器将 work items 映射到模拟 core，把固定容量 threadgroup scratch 映射为模拟 UB/L1，把 flag
状态转移和 handoff 依赖映射为显式事件图。可以用后端中立的 Python 参考执行器验证调度结果，但
**不能执行 MSL 来声称 NPU kernel 正确**。

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
- 拆分 semantic/workload/evaluation-input 身份；固定 shape 投影不变；
- statement 展示 canonical profile，同时声明其他分段会测试；
- output-only final_state 负例、hardcoded-canonical 负例进入 Metal GPU 测试。

验收：同一 ABI 下换分段仍正确；写死边界失败；每个实际 case 都能由 input digest 重放。
V0/V1 涉及 pack closure 的修改集中完成后，**只重录一次**受影响 P2 证据。

### V2：后端中立 Host Program（CPU）

- 实现 HostProgramIR、restricted AST、白名单 intrinsic、输入/输出资源校验；
- 保持 JSON linear 与单文件字节/行为不变；
- 生命周期 input-dependent/timing/evidence 接线；
- 用 KDA `_host_item_table` 的等价测试 planner 验证嵌套函数、heap、二分、LPT、位打包全部可表达；
- preflight 拒绝危险语法、未声明资源、错误 workspace span 和非法 resolved scalar。

验收：同一 planner 在两个 backend adapter 上产生逐字节一致的 workspace 与 launch 记录；
HostRunEvidence 可重放；不跑 GPU/NPU 也能完成本阶段大部分验收。

### V3：Metal KDA work-list 垂直切片（需显式 GPU 授权）

- 完成 §6 的 host+device fixture；
- range-cut、LPT、长链 fp32 handoff、partial tail、threadgroup/global item table 都有正例；
- §5.2 负例全部通过“必须拒绝”验收；
- 与 naive varlen KDA oracle 在 canonical/permuted/skewed/boundary 上对齐 output/final_state；
- host wall time 进入 candidate 时间，实际 dispatch 与 resolved launch 一致。

验收只能标“Metal 机制正确”，还不能标“达到 KDA 性能”。

### V4：NPU 协议模拟 parity（不需要真实 NPU）

- V2 的同一 Host Program 在 NPU simulator 产生相同 work list；
- core 分配、事件依赖、handoff flag、UB/L1 容量与尾块规则可审计；
- 所有报告 `simulated=true/publication_allowed=false`；
- 模拟不进入 Metal 的发布链，也不解除 Qingwei NPU-P0。

验收：GPU 与 NPU simulator 的 Host Program/Worklist/ResolvedLaunch 差分为零；证据类型严格区分。

### V5：L2 runtime total/count 与 capacity（新模型阶段）

- 实现 logical/resolved/capacity shape、max-capacity ABI、resolved grid/scalars；
- dispatch request 携带 resolved total/count，exact one-of-N 与独立 fallback 保持；
- padding/guard、workspace capacity、empty batch/empty sequence、超 capacity typed refusal；
- Host Program cache 若加入，必须绑定 evaluation input digest 并记录 cold/miss；
- 用同一候选在连续多个不同 total/count 调用上验证无重编译语义（是否重编译由候选与缓存身份决定，
  证据必须如实记录）。

验收：continuous batching 场景不再依赖为每个 total/count 建一个静态档；L1 证据不冒充 L2。

### V6：真实规模性能与搜索（单独 GPU/付费授权）

- D=128、H=64/96、fixed/mixed/uniform 的 KDA 规模；
- 候选至少有一种真正的 chunked/matrix 实现，naive 单线程只作对照；
- host+device end-to-end 与 vendor/独立 baseline 同边界；逐档 fresh confirmation 与独立 fallback；
- 只有安静、fresh、unique publication predicate 全部满足才发布；
- NPU 模拟数字永不参与比较。

---

## 9. 文件级改动地图

| 层 | 文件/位置 | 改动 |
| --- | --- | --- |
| OpSpec | `tcenv/opspec/model.py`、`opspec/v2.py`、schema | varlen 校验、profile、L2 shape/capacity |
| 身份 | `portfolio/domain.py`、`taskcontract`、validation | 三层身份、dispatch/fresh seed/input replay |
| 题面 | `tcenv/task_presentation.py`、协议参考 | canonical profile + 可变分段 + Host Program 合同 |
| Host IR | 新后端中立模块（不引入 Metal import） | AST/IR、intrinsics、资源与差分语义 |
| Metal | `backends/metal/v2_runtime.py`、`track_adapter.py` | 输入解码、workspace、resolved scalar、host evidence |
| 生命周期 | `taskcontract/portfolio_runner.py`、`track/lifecycle.py` | input-dependent prepare 与计时边界 |
| 几何 | `track/geometry.py`、backend stage binding | launch envelope/resolved launch 校验 |
| 任务 | `tasks/varlen_kda`、`tasks/varlen_scale` | 分段 case、必要 workspace、真实规模档 |
| Metal 测试 | `tests/backends/metal/` | Host Program、work-list、handoff/tail、GPU 负例 |
| 共享测试 | `tests/shared/test_opspec_varlen.py` 等 | 合同、身份、固定 shape 兼容、负例 |
| NPU 模拟 | `tests/support` 的模拟组合根 | 同 IR/core event/evidence；生产组合根不 import |

每阶段开始先重新计算 `code_import_closure`，闭包改动集中处理并按授权重录受影响证据；不要假设
某个文件“上次不在闭包，这次仍不在”。

---

## 10. 完成定义

只有以下条件全部满足，才能说“GPU/NPU 通用的 varlen 机制已完成”：

- [ ] 固定 shape 身份、既有 linear bundle、单文件、R5/远程协议均未退化；
- [ ] 同一 ABI 下任意合法 `cu_seqlens` 正确，写死 canonical 边界会被拒绝；
- [ ] 候选可用后端中立 Host Program 实现 KDA 级 planner，harness 没有内置任务调度器；
- [ ] Metal 真机验证 work-list、partial tail、跨核 handoff、逐段 final_state 与全部作弊负例；
- [ ] L2 total/count 变化有 capacity/resolved-launch/dispatch/fallback 的完整语义；
- [ ] 所有输入驱动 host work 进入计时与证据，KDA prepare 口径没有被偷运进来；
- [ ] NPU simulator 复用同一 IR 且所有结果 `simulated=true/publication_allowed=false`；
- [ ] 真实性能只在 Metal 真实设备、真实规模、同边界 baseline、fresh quiet confirmation 后发布；
- [ ] Qingwei NPU-P0 仍如实 blocked，Ascend/CANN 参考资料未被写成生产能力。

在此之前，只能分别宣称“V2a 固定分布切片”“L1 任意分段正确性”“Metal work-list 机制正确”
或“NPU 协议模拟通过”，不能统称完整 varlen 已支持。
