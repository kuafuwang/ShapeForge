# varlen（可变长序列）支持计划

状态：**咨询稿，未授权实施**。本文基于 KDA-for-KDA 仓库的**真实 varlen 实现**（本地
`/Users/kuafu/CLionProjects/kda-260927-kda`）与 CAKE2 论文（arxiv 2610.31034）的核对结果制定，
不是凭空设计。所有 KDA 机制均给出仓库内位置；ShapeForge 侧接口以 `docs/架构/v2协议参考.md` 为准。

## 0. 先回答「KV cache 下不支持动态 shape 能行吗」

取决于 ShapeForge 是否要做**自回归推理**的算子：

- **现有题（训练、固定 shape 推理、benchmark）**：不需要 varlen，静态特化（shape portfolio + 精确
  one-of-N dispatch）就是最优解。
- **自回归推理（KV cache、continuous batching、varlen attention）**：**不支持 varlen 不行**。
  prefill 各 prompt 长度不同、decode 每 step KV 长度增长、batch 内各序列长度不同，固定 shape 特化
  只能靠「为每个 length 出 variant（不现实）/ padding 到 max（浪费）/ bucket（牺牲性能）」凑合。

varlen 是「实例模型」的问题，不是「候选自由度」的问题。KDA/CAKE2 的候选自由度已吸收进
msl-bundle；varlen 需要的是 OpSpec 的**实例与张量模型扩展**。

## 1. 学习对象核实：KDA 是真 varlen，CAKE2 不是

- **KDA（KDA-for-KDA）**：真正的 varlen。`cu_seqlens` 是 ABI 输入，host 派生 work list，跨 CTA
  状态 handoff，分档计时。这是 varlen 的主要学习对象。
- **CAKE2（CANNBot-CAKE，arxiv 2610.31034）**：论文**没有** varlen/cu_seqlens/ragged/动态 shape
  机制。它的实例模型是「固定 contract 下的多个声明 shape case + host tiling + shape-driven
  kernel selection」——是**静态 shape 集合**，与 ShapeForge 的 shape portfolio 同构，不是 varlen。
  之前把「KDA/CAKE2 的 ABI 天生是 varlen」一起说是**错误的**，已纠正。

## 2. KDA 的 varlen 真实机制（学习的核心）

### 2.1 实例模型：cu_seqlens 的取值分布

`cu_seqlens = [0, *cumsum(seq_lens)]`（`bench.py:87-91`）。KDA 的六档 workload 就是三种长度分布
× 两种 head 数（`bench.py:58-63`）：

| 档 | seq_lens | 含义 |
| --- | --- | --- |
| `fixed` | `[8192]` | 单段全长（varlen 的退化） |
| `mixed` | 多段不等长 | 长度差异大，调度难 |
| `uniform` | `[1024]*8` | 多段等长 |

`initial_state` 是 `[num_seqs, H, D, D]`（**每段一个初始状态**）。这直接对应 ShapeForge 的
「varlen size 档 = cu_seqlens 的一种取值分布」。

### 2.2 调度：host work list 派生（核心，不是「kernel 内循环」一句话）

KDA 的 varlen 调度是一个**负载均衡问题**，由 host 侧 `_host_item_table`
（`kda-tirx/kda_fwd_varlen.py:1746+`）把 `(head, sequence, chunk)` 的工作切成 `num_ctas` 个 range：

- **成本模型**（`:64-69`）：chunk cost 1 + 段开始成本 `BETA=1` + LPT 单位开始成本 `START_COST=0.9`
  + 跨 CTA 状态 handoff 成本 `HANDOFF_COST=0.4`（都以 chunk 为单位）。
- **两种策略**（`:1773-1775`）：最长序列的 chunk 数 `maxnch <= l_min - (2*SNAP+BETA)`（最长序列短于
  平均负载）时用**等成本 range 切分**（`_optimal_cuts`，最小化最大 range 成本）；否则用**整段 LPT**
  （把整段 `(sequence, head)` 放到最空闲的 CTA）。
- **item 打包**：每个 work item 打包成 `(word0, word1)`，含 token offset、valid rows、跨 CTA
  handoff 指针、first/last flags、seq、head（`:1750-1753`）。

**关键**：range 在序列中间开始时，该 CTA 用**前一个 CTA 的 fp32 state handoff** 继续（不是从零开始）。
这是 varlen 正确性的核心——hacking_example 的「零状态续传」作弊正是破坏了它。

### 2.3 SMEM item table

prologue 把 cu_seqlens 展开成 SMEM item table（token offset、valid rows、seq id、first/last flags），
所有 role 按 flat item index 循环以保持 barrier parity（`:31-34`）。尾块（partial tail）在 prep
role 里把越界行的 k/q/log-decay/beta 清零，state 不被污染（`:32-36`）。

### 2.4 状态 handoff 与 partial tail

- 每段第一个 chunk 读 `initial_state[seq]`，最后一个 chunk 写 `final_state[seq]`（`:36-38`）。
- 跨 CTA 的序列用 **fp32 state handoff**（不是 bf16，H96 uniform 保留 fp32 handoff，`:41-44`）。
- 输出用 **predicated 16-byte global store**（不能用固定 box 的 TMA store，序列边界在内部，`:38-40`）。

### 2.5 varlen shape portfolio

不同 varlen 分布用**不同调度策略**（`:41-44`）：H96 mixed-varlen 强制整段 LPT；H64 mixed 与
uniform 保留 chunk-linear splitting + bf16 continuation buffer。**这正好印证 ShapeForge 的
shape portfolio 思想可以延伸到 varlen**：不同长度分布是不同「档」，每档选不同实现。

### 2.6 分档计时

fixed/mixed/uniform 各是一档 benchmark，分别报 speedup（`bench.py:206+`，README 的结果表按
fixed/mixed/uniform 分列）。**不是把 varlen 当成一个 shape，而是把长度分布当成多个档**。

## 3. 对 ShapeForge 的扩展设计（基于上述真实机制）

### 3.1 实例模型：varlen size 档 = cu_seqlens 的取值分布

当前 `sizes[size]` 给 symbols 固定值。varlen 的 size 档改为给 **cu_seqlens 的取值**：

- `fixed`: `[0, 8192]`；`mixed`: `[0, 100, 600, 2600, …]`；`uniform`: 8×1024。
- `initial_state` 等按 `num_seqs` 声明（每段一个）。
- **证据身份**：varlen instance 的 `instance_key` 必须包含长度分布标识（`mixed` 与 `uniform`
  是不同问题），不能沿用固定 shape 的 `dtype/size/usage`。

### 3.2 张量声明：packed sequence + cu_seqlens

- 引入**运行时维度** `total_tokens`（所有序列的 token 总和，运行时才知道，不是编译期 symbol）。
  张量 `dims` 允许 `["total_tokens", "H", "K"]`。
- 新增合同输入 `cu_seqlens`：`role: input`、`dtype: int64`、`dims: ["num_seqs+1"]`、`space: global`。
- `num_seqs` 也是运行时值。

### 3.3 调度：Host Tiling work list（复用 msl-bundle 的 Host Tiling）

varlen 的 work list 派生是 **Host Tiling plan 的自然延伸**：Host 侧读 cu_seqlens，按成本模型
（chunk + 段开始 + handoff）把 `(head, seq, chunk)` 切成 CTA range，写进 workspace 的 item table。
这正是 KDA 的 `_host_item_table` 做的事，也是 msl-bundle 已有的「Host Tiling 写 workspace」能力。
候选也可以用 kernel 内 cu_seqlens 循环（固定长度 KDA 的 recurrent 路径）。

### 3.4 状态 handoff：fp32 + 逐段 final_state

- 跨 stage/CTA 的序列状态用 **fp32 workspace** 传递（不能降精度）。
- 正确性必须**逐段对齐 final_state**（每段第一个 chunk 读 initial、最后 chunk 写 final），
  不能只比 output——hacking_example 的「零状态续传」就是靠「不看 final state」漏过的。
  这对应 ShapeForge 已有的「正确性含最终 `read_write` 状态与 workspace 清空规则」。

### 3.5 正确性/计时

- reference 按 cu_seqlens **分段**计算（KDA 的 delta rule、varlen attention 的掩码都是段内语义）。
- `gen_inputs` 按该档 cu_seqlens 生成打包序列。
- 计时按 fixed/mixed/uniform **分档**报 speedup，不合并。

### 3.6 与 shape portfolio 的关系

varlen 不是取代 shape portfolio，而是**把「档」从固定 shape 扩展到长度分布**：fixed/mixed/uniform
各是一档，候选为不同长度分布写不同 variant（KDA 的 H96 vs H64 策略差异正是这样）。fallback 仍需
独立通用实现。

## 4. 分阶段实施

### V1：固定长度 varlen 算子（当前已可行，作为退化验证）

用**固定长度 KDA**（单段）验证「多阶段状态递推 + workspace + 多源」。这是 varlen 的退化
（`cu_seqlens` 只有一段），当前固定 shape 模型下已可做，**不需要等 V2**，是 msl-bundle 的合理
测试例子。

### V2：packed varlen（cu_seqlens）——核心扩展

引入 §3 的全部扩展：运行时 `total_tokens` 维度、`cu_seqlens` 输入、varlen size 档（含证据身份）、
Host work list 派生、fp32 状态 handoff、分段 reference 与输入生成、分档计时。以 **varlen KDA**
作为首个端到端算子（ABI 就是 varlen，有公开精确 recurrence 作 oracle，有 KDA 的真实调度作参照）。

### V3：动态 shape 泛化

把「运行时维度」从 sequence length 泛化到任意维度（动态 batch/head/分辨率）。这是 OpSpec v3
级别的泛化，不是 V2 的前置。

## 5. 风险与边界

- **不是开关**：varlen 牵动 opspec、实例派生、几何、reference、评测、证据身份。
- **证据身份必须含长度分布**：否则 mixed 与 uniform 会共享 instance 身份，证据错配。
- **状态 handoff 是正确性高危区**：跨 CTA/stage 的状态递推必须 fp32 且逐段验证，这是
  hacking_example 作弊被抓的地方，也是 varlen 最容易出错的地方。
- **不破坏固定 shape**：现有题的固定 shape 语义、pack 闭包 digest、P2 证据身份保持不变；
  varlen 是新增能力，固定 shape 作为其退化继续工作。
- **与 msl-bundle 正交**：varlen 是「算什么 shape」，msl-bundle 是「候选怎么由多文件组成」，
  两者独立，varlen 算子可用 single 或 bundle 交付。

## 6. 与当前工作的关系

- 本计划**不阻塞**固定长度 KDA 作为 msl-bundle 测试例子（V1 当前可做）。
- varlen 完整支持（V2）是独立工作线，需单独授权与阶段验收，不与 msl-bundle 接入或 v1→v2 迁移
  混在一起。
- KDA 的 varlen 机制（`_host_item_table`、SMEM item table、fp32 handoff、分档计时）是 V2 的
  直接实现参照；CAKE2 的静态 shape case 模型与现有 shape portfolio 同构，不是 varlen 的学习对象。
