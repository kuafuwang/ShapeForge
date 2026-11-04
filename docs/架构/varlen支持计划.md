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

## 1. 学习到的机制，以及哪些不该照抄

### 1.1 KDA：输入驱动的 varlen 工作项与状态协议

核对对象：本地 `kda-260927-kda/kda-tirx/kda_fwd_varlen.py` 与 `bench.py`。

必须保留的经验：

- **每次调用读取实际 `cu_seqlens`**，不是只识别一个固定分布名称。
- `_host_item_table` 将 `(head, sequence, chunk)` 展开成 per-CTA 工作项；工作项包含 token
  offset、valid rows、seq/head、first/last、src/dst handoff。
- 成本模型考虑 chunk、序列开始、pipeline fill/drain 和 handoff；等成本 range-cut 与整段 LPT
  是两种候选策略，不是 harness 应硬编码的唯一答案。
- 长序列可以跨 CTA 切开；后续 piece 通过 handoff state + acquire/release flag 接续，不能从零
  状态或独立重算。
- SMEM 只缓存有限工作项，超长列表从 global 读取；partial tail 用 valid-row mask，输出按有效行
  写回，不能用过界固定 box store。
- `initial_state[seq]` 只在序列首 piece 读取，`final_state[seq]` 只在末 piece 写出；中间 piece
  读 handoff。fp32 是权威路径，dtype 特化 handoff 必须另有容差证据。
- fixed/mixed/uniform 分别计时；output 和 final_state 都检查。

**计时差异不能照抄**：KDA 的 `prepare()` 在计时前完成 host scheduling/device tables/graph，
`bench.py` 只测返回的 launch。ShapeForge 中，候选读取当前输入所做的事情属于本次求值，必须
`input_dependent=true` 且进入要求的计时边界。我们可以复用算法，不能复用“把 host work 移到
计时外”的口径。

### 1.2 昇腾 CAKE2：host+device 交付与 shape 驱动调度

CAKE2 的可借鉴点是：

- 一个可交付算子是 **host tiling + device kernel + package/binding/evaluation** 的协同产物，
  不是孤立的 device 源码。
- host-level 策略包括 TilingKey 分发、按 shape 选择特化 kernel、多核负载均衡和 straggler core
  修复。
- 每种机制都有可检查 artifact 与验证记录。

CAKE2 核心没有给出 KDA 式通用 `cu_seqlens` work list 和 recurrent state handoff；不能把它的
shape-driven tiling 直接称为 varlen 支持。

### 1.3 CANNBot FA 技能：昇腾特有的 varlen/跨核处理

CANNBot 的 FA/catlass 技能（不是 CAKE2 核心通用框架）提供了额外参考：TND varlen、批界截断、
非整块尾处理、KV split host 贪心与前缀和、CrossCore flag、AIC/AIV 协作、`needL1Carry`。

这些机制只进入我们的**后端中立能力清单**与 NPU 模拟检查项；不引入 AscendC/CANN 代码。

### 1.4 统一抽象，而不是三条实现路径

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
