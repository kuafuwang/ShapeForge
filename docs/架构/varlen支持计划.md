# varlen（可变长序列）支持：施工方案

状态：**施工方案（待授权实施）**。本方案基于 KDA-for-KDA 仓库的**真实 varlen 实现**
（`/Users/kuafu/CLionProjects/kda-260927-kda`）、CAKE2 论文（arxiv 2610.31034）核对，以及对本仓库
OpSpec v2 的 24 个「固定 shape 假设点」的逐项调查（文件+行号随附）。**模型可以修改**——本方案
明确哪些模型要改、怎么改，而不是绕过现有模型。

## 0. 先回答「KV cache 下不支持动态 shape 能行吗」

- **现有题（训练、固定 shape 推理、benchmark）**：不需要 varlen，静态特化就是最优解。
- **自回归推理（KV cache、continuous batching、varlen attention）**：不支持 varlen 不行。固定 shape
  只能靠「为每个 length 出 variant（不现实）/ padding 到 max（浪费）/ bucket（牺牲性能）」凑合。
varlen 是「实例模型」的问题，不是「候选自由度」的问题。候选自由度已吸收进 msl-bundle；varlen
需要的是 OpSpec 的实例与张量模型扩展，**本方案就改这个模型**。

## 1. 学习对象核实

- **KDA（KDA-for-KDA）**：真 varlen。`cu_seqlens` 是 ABI 输入，host 派生 work list，跨 CTA fp32
  状态 handoff，按长度分布分档计时。**主要学习对象**。
- **CAKE2（CANNBot-CAKE）**：论文无 varlen/cu_seqlens/动态 shape 机制。它的实例模型是「固定
  contract 下的多个声明 shape case + host tiling + shape-driven selection」——**静态 shape 集合**，
  与 ShapeForge 的 shape portfolio 同构，不是 varlen。之前「两者都天生 varlen」的说法是错的。

## 2. KDA 的 varlen 机制（逐项 → 我们要学什么）

### 2.1 实例规约：长度分布 + 稳定身份

KDA 的六档 workload 用**名字**区分长度分布（`bench.py:58-63`）：

```python
"h96-fixed":          (96, [8192])          # 单段全长（varlen 退化）
"h96-mixed_varlen":   (96, MIXED)           # 多段不等长
"h96-uniform_varlen": (96, [1024]*8)        # 多段等长
# h64 同三组
```

`cu_seqlens = [0, *cumsum(seq_lens)]`（`bench.py:87-91`）；`initial_state` 是 `[num_seqs, H, D, D]`
（每段一个初始状态）。**每个 workload 有固定 seed**（`bench.py:29`），output 与 final_state 都进
正确性（`bench.py:129,226`）。

→ **学什么**：varlen instance = 「长度分布 + 稳定 ID + 固定 seed」。固定 shape 是「单段全长」的退化。

### 2.2 调度：host work list 派生（成本模型 + 两种策略）

`_host_item_table`（`kda-tirx/kda_fwd_varlen.py:1746+`）把 `(head, sequence, chunk)` 切成 `num_ctas`
个 range：

- **成本模型**（`:64-69`）：chunk cost 1 + 段开始 `BETA=1` + LPT 单位开始 `START_COST=0.9` +
  跨 CTA handoff `HANDOFF_COST=0.4`。
- **策略**（`:1773-1775`）：最长序列 chunk 数 `maxnch <= l_min - (2*SNAP+BETA)` 用**等成本 range
  切分**（`_optimal_cuts`），否则**整段 LPT**。
- **item 打包**（`:1750-1753`）：`(word0, word1)` = token offset、valid rows、handoff 指针、
  first/last flags、seq、head。

→ **学什么**：varlen 的 work list 是 **host 侧按成本模型派生**的，不是「kernel 内循环」一句话。
跨 CTA 的序列用**前一个 CTA 的 fp32 state handoff** 继续（不是从零开始）。

### 2.3 SMEM item table 与 partial tail

prologue 把 cu_seqlens 展开成 SMEM item table（token offset、valid rows、seq id、first/last flags），
所有 role 按 flat item index 循环（`:31-34`）。尾块在 prep role 把越界行的 k/q/log-decay/beta 清零，
state 不被污染（`:32-36`）；输出用 predicated 16-byte store（不能用固定 box TMA，`:38-40`）。

→ **学什么**：varlen 需要「有效行/尾块掩码」机制，不是简单按总长算。

### 2.4 状态 handoff：fp32 + 逐段 final_state

每段第一个 chunk 读 `initial_state[seq]`、最后一个 chunk 写 `final_state[seq]`（`:36-38`）；跨 CTA
用 fp32 handoff（H96 uniform 保留 fp32，`:41-44`）。hacking_example 的「零状态续传」作弊正是破坏了
这个（`docs/可信harness/案例-KDA²-算子作弊与可信评测.md`）。

→ **学什么**：varlen 的**正确性高危区是状态递推**。必须 fp32 + 逐段 final_state 验证，不能只比 output。

### 2.5 varlen shape portfolio

不同长度分布用不同调度策略（`:41-44`）：H96 mixed 用整段 LPT，H64 mixed/uniform 用 chunk-linear
splitting + bf16 continuation buffer。

→ **学什么**：shape portfolio 思想可延伸到 varlen——**不同长度分布是不同「档」，每档选不同实现**。

## 3. ShapeForge 的模型扩展（24 个假设点 → 改动）

调查发现的「固定 shape 假设点」按改动分组如下。每条给出 KDA 机制对应与当前文件位置。

### 3.1 varlen 实例规约（改实例模型）

在 `sizes` 的 size 档中引入 **varlen profile**：一个带稳定 ID 的长度分布规约（`seq_lens` 或
`cu_seqlens`），固定 shape 作为「单段全长」的退化。

- 改 `sizes` 校验（`tcenv/opspec/model.py:556-584`）：size 档除 symbols 外，允许一个
  `varlen`/`seq_lens` 字段；有它时该档是 varlen 实例。
- 实例化（`tcenv/opspec/v2.py:496-576`）：先由 varlen profile 生成 cu_seqlens，解析出
  `total_tokens = sum(seq_lens)` 与 `num_seqs = len(seq_lens)`，再派生完整实例。
- `instance_key`（`tcenv/portfolio/domain.py:153-170`）：varlen 实例用 `dtype/<profile-id>/usage`，
  profile-id 含长度分布（`mixed`/`uniform`/`fixed`），不能只靠字符串——digest 要含完整 cu_seqlens。

### 3.2 逻辑 shape / resolved shape / buffer capacity 分离（改张量与 ABI）

varlen 不能把运行时长度强转成 symbol。引入三者分离：

- **逻辑 shape**：`total_tokens`（运行时维度，非编译期 symbol）。
- **resolved shape**：本次输入的实际长度（由 cu_seqlens 决定）。
- **buffer capacity**：分配上限（max-capacity，实例规约里给出，或按该档最大值）。

改动：
- `dims` 文法（`tcenv/opspec/v2.py:309-325`、`model.py:329-367`）：允许 `total_tokens` 这类
  运行时维度标记（区别于编译期 symbol），`num_seqs` 同理。`cu_seqlens.dims = ["num_seqs+1"]`。
- `TensorBinding`（`model.py:96-137`）：区分逻辑维度与 capacity；`numel/nbytes` 按 capacity 算。
- ABI（`tcenv/track/abi.py:153-307`）：span 按 capacity 算（分配），但记录逻辑维度供运行时校验。
- Metal buffer 分配（`v2_runtime.py:338-352`）：按 capacity 分配，运行时长度做边界检查。
- launch workspace 字节（`v2_runtime.py:580-603`）：从「必须等于固定 span」改为「capacity 相等 +
  运行时长度 ≤ capacity」。

### 3.3 dtype：补 int64 或改用 i32

`cu_seqlens` 需要整型索引。当前注册表只有 fp16/bf16/fp32/i32/u32/i8/u8（`tcenv/dtypes.py:48-60`）。
- 方案 A：新增 `int64` 全链路支持（dtype 注册、ABI、Metal `long` 映射、reference）。
- 方案 B：`cu_seqlens` 用 `i32`（KDA 用 int64，但 token 数 < 2^31 时 i32 够用）。**建议先用 i32**，
  避免全链路加 int64 的工作量；在 opspec 里注明 cu_seqlens 的 dtype 约束。

### 3.4 几何：两条路线（改几何派生）

- **路线 A：固定上限 grid + device 内按 cu_seqlens 早退**。保留预解析几何（grid 按 capacity 派生），
  kernel 内按 cu_seqlens 处理实际长度。适合 KDA 的 recurrent 路径。改动小：几何表达式仍按
  symbols/capacity 求值（`tcenv/track/geometry.py:261-318`），运行时长度在 kernel 内。
- **路线 B：host 读 cu_seqlens 派生 work list/grid**。扩展 Host Tiling 使其能读声明输入
  （cu_seqlens）、在计时边界内执行、把实际解析几何纳入证据。适合 KDA 的 work list 路径。改动大：
  Host Tiling 目前只支持 symbols/attrs 驱动的 linear writes（`v2_runtime.py:65-82,157-243`），
  且 portfolio_runner 把 host_tiling 固定为 `input_dependent=False`（`portfolio_runner.py:638-660`）。

**V2 先走路线 A**（改动小、能跑通 varlen KDA 的 recurrent 形式），路线 B 作为 V2 的后续增强
（KDA 的 work list 优化）。这样 V2 不必一次改完 Host Tiling 的输入依赖模型。

### 3.5 reference / 输入生成 / 正确性（改评测语义）

- `gen_inputs`（`tcenv/taskcontract/reference.py:111-132`）：varlen 必须自定义——先按 profile 生成
  cu_seqlens（首项 0、单调、末项 = total_tokens），再按 total_tokens 生成打包张量、按 num_seqs 生成
  per-sequence initial_state。默认采样（`sampling.py:194-217`）不适用。
- `reference(inputs, attrs)`：可读 cu_seqlens，按段独立计算（KDA delta rule、varlen attention 掩码）。
  注意 reference 的输入/输出**形状校验**（`reference.py:309-344`）要支持「resolved shape」而非固定 shape。
- 正确性（`v2_runtime.py:1033-1088`、`testing.py:140-171`）：引入 **valid-length mask / ragged
  比较**——按 cu_seqlens 的有效区域比较，尾块/padding 不计入误差（KDA 的 partial tail 清零语义）。
  `error_scale` 不能用来「掩盖 padding」，需要正式的 ragged 比较语义。
- **逐段 final_state**：final_state 是 global `read_write` 张量（`[num_seqs, H, D, D]`），逐段验证，
  不能只比 output（hacking_example 的零状态续传就是靠「不看 final state」漏过的）。

### 3.6 长度分布身份贯穿证据层（改身份模型）

规范化 varlen profile（cu_seqlens）必须进入：
- `semantic_instance_digest`（`opspec/v2.py:216-273`）：digest 含完整 cu_seqlens，不只 profile 名。
- domain point / DispatchRequest（`portfolio/domain.py:153-170`、`dispatch.py:756-807`）：varlen
  instance 的 key 含 profile-id，dispatch 绑定该 profile。
- confirmation seed（`portfolio_validation/validation.py:77-110`、`taic/portfolio_confirmation.py:50-63`）：
  seed 派生输入含规范化 cu_seqlens，保证不同长度分布有不同 fresh seed。
- evaluation fingerprint（`tcenv/evaluate.py:1448-1507`）：varlen profile 进 fingerprint。
- pack digest（`taskcontract/pack.py:262-308`）：varlen profile 声明在 opspec 内 → 间接进 digest。

### 3.7 题面与 op_desc 投影

`task_presentation.py:48-69`、`opspec/op_desc.py:28-52` 目前只打印整数 shape。varlen 要打印
「逻辑维度 + capacity + cu_seqlens 规约」，让 LLM 看到 varlen 的真实 ABI 与调度自由。

## 4. 分阶段实现路径

### V1：固定长度 varlen 算子（当前可做，退化验证）

用**固定长度 KDA**（单段全长）验证「多阶段状态递推 + workspace + 多源」。这是 varlen 的退化，
当前固定 shape 模型下已可做，**不需要 V2 的任何模型改动**。它是 msl-bundle 的合理测试例子。

### V2：packed varlen（cu_seqlens）——核心扩展

按 §3 改模型，以 **varlen KDA** 作为首个端到端算子：

1. **实例规约**（§3.1）：`sizes` 支持 varlen profile（`seq_lens`），KDA 声明 fixed/mixed/uniform 三档。
2. **张量与 ABI**（§3.2、§3.3）：`total_tokens`/`num_seqs` 运行时维度 + capacity 分离；`cu_seqlens`
   用 i32。
3. **几何**（§3.4 路线 A）：固定上限 grid + kernel 内按 cu_seqlens 逐段 recurrence。
4. **reference/正确性**（§3.5）：分段 reference + ragged 比较 + 逐段 final_state。
5. **证据身份**（§3.6）：varlen profile 贯穿 digest/dispatch/seed/fingerprint。
6. **Host work list**（§3.4 路线 B，后续增强）：KDA 的等成本 range 切分 + fp32 handoff。

**V2 验收**：varlen KDA 在 fixed/mixed/uniform 三档上，候选 kernel 正确（含逐段 final_state）、
可分档计时、证据身份区分三档；hacking_example 的「零状态续传」「丢弃初始状态」两类作弊被
逐段 final_state 验证挡住。

### V3：动态 shape 泛化

把「运行时维度」从 sequence length 泛化到任意维度（动态 batch/head/分辨率）。OpSpec v3 级别，
不是 V2 前置。

## 5. 风险与边界

- **不是开关**：varlen 牵动 opspec、实例派生、ABI、几何、reference、评测、证据身份（§3 全部）。
- **证据身份必须含长度分布**：否则 mixed/uniform 共享 instance 身份，证据错配。
- **状态 handoff 是正确性高危区**：跨 CTA/stage 状态递推必须 fp32 + 逐段验证（hacking_example）。
- **不破坏固定 shape**：现有题的固定 shape 语义、pack 闭包 digest、P2 证据身份保持不变；
  varlen 是新增能力，固定 shape 作为其退化继续工作。
- **与 msl-bundle 正交**：varlen 是「算什么 shape」，msl-bundle 是「候选怎么由多文件组成」，
  varlen 算子可用 single 或 bundle 交付。
- **路线 B 的工作量**：Host Tiling 读输入 + 计时边界 + 实际几何入证据是较大改动，V2 先用路线 A
  跑通，路线 B 作为增强，避免一次改完所有模型。

## 6. 与当前工作的关系

- **不阻塞**固定长度 KDA 作为 msl-bundle 测试例子（V1 当前可做）。
- varlen 完整支持（V2）是独立工作线，需单独授权与阶段验收，不与 msl-bundle 接入或 v1→v2 迁移
  混在一起。
- KDA 的 varlen 机制（`_host_item_table`、SMEM item table、fp32 handoff、分档计时）是 V2 的直接
  实现参照；CAKE2 的静态 shape case 模型与现有 shape portfolio 同构，不是 varlen 的学习对象。
