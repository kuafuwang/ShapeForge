# V5 rlar 任务包：runtime total/count、capacity 与 continuous batching

状态：这是当时的实施输入，不是完成证据。下面的「不授权」只约束写这份任务包的那一轮。V5 的 CPU/协议与真实 Metal、V6 六档和 P2-F 后来都已落地，现状以 `docs/架构/varlen支持计划.md` 文首为准。

原授权说明：本轮授权 rlar actor/reviewer 消耗 token；**不授权 GPU kernel、付费模型搜索或 P2 重录**。

## 1. 目标

让同一个候选在同一个 max-capacity ABI 下处理连续多次不同 `total_tokens/num_seqs` 的调用，
不再为每个 total/count 创建静态档。L1 是 resolved total/count 等于 capacity 的退化情形。

## 2. 不可变边界

1. 候选拥有算法、planner、tiling、work list 和 ExecutionGeometry；harness 不内置 KDA 专用调度器。
2. dispatch 主体是 capacity 实例；resolved total/count 只进入 evaluation input/launch 证据，
   不把动态值塞进现有有限域 guard。
3. L2 `SemanticInstanceV2.cu_seqlens=None`，benchmark 分布由独立 workload profile 标识；L1
   canonical `cu_seqlens` 身份保持不变，固定 shape 投影不变。
4. `cu_seqlens` 每个调用都必须是 i32、从 0 开始、单调、末项等于本次 total_tokens；
   `total_tokens>=1`，允许 empty sequence，拒绝 empty batch 和超 capacity。
5. 比较 logical output 前缀和 `final_state[:num_seqs]`；未用 padding、`cu_seqlens` capacity 尾部
   和 guard/sentinel 必须保持声明的 poison。正确性只使用现有 tolerances + scaled_error。
6. Host Program 继续 `input_dependent=true + preprocesses_inputs=true + inside_timing=true`，
   不能把 prepare 移出 E2E。
7. 保留 native64 批量校验、Metal 地址绑定和 C 原子直接提交；不得通过删除 fresh-input、
   read-only/guard/span 检查来降低耗时。
8. 保持 Dyn-0 静态 capacity grid；本阶段不启用 Dyn-1，不把 overlap 当成前置条件。
9. NPU 仅模拟，始终 `simulated=true/publication_allowed=false`；清微 NPU-P0 继续 blocked。
10. 不改六 RPC/HMAC/BlobChannel、R5、历史验收文件或 P2 golden/evidence/manifest。

## 3. 必做实现

### 3.1 形状与容量合同

- 在 OpSpec/schema 中声明 max total/count 及 capacity 派生规则；
- 区分 logical、resolved、capacity shape，buffer 按 capacity 分配；
- L2 instance 的 `cu_seqlens=None`，独立 workload-profile identity；
- 校验 fixed/L1/L2 不得混用，非法 capacity/alias/shape 公式给出 typed error。

### 3.2 Runtime resolved scalar

- 每次调用向 device 和候选 Host Program 提供真实 total/count；
- 扩展 native `SfHostResolved` 与生成 accessor，不再只有 reserved 字段；
- resolved scalar 进入 HostRun/evaluation evidence，并在调用前后校验其合同；
- 保持 C 编译器拥有 native64 布局，Python 不猜偏移。

### 3.3 Continuous-batching 输入与正确性

- 同一候选/同一 build 连续执行多个不同 total/count 和不同合法分段，不能按 total/count 重编译；
- 记录每次实际 `evaluation_input_digest`、resolved shape 和 case 顺序；
- poison 未使用 input、`cu_seqlens` capacity 尾部、output padding 和 final_state padding；
- 覆盖 total=1、count=1、count=capacity、total=capacity、空序列、skewed、permuted、fresh、
  连续缩小再增大 batch；拒绝 total=0、负值、非单调、错误末项和超 capacity。

### 3.4 Dispatch / fallback / 证据

- exact dispatch 使用 capacity 实例，不把 resolved total/count 当作另一个静态 domain point；
- fallback 仍独立，fallback 时间与正确性不得冒充 variant 证据；
- build/candidate/native digest 不因 resolved total/count 改变；
- HostRunEvidence 保留 run.2/native 身份，补充 resolved total/count 和实际 input digest；
- 如加入 cache，必须绑定 evaluation input digest 并记录 cold/miss；否则明确不加入。

### 3.5 文档

- 更新 varlen 计划与 native ABI 当前状态。当时这一轮不宣称 GPU/性能验收已完成；后来的 V5 Metal 与 V6 见 varlen 计划文首；
- 重算 `code_import_closure`，记录 digest 前进。当时 P2 继续 deferred；P2-F 后来完成；
- 不删除或改写历史 V0–V3 验收证据。

## 4. 必须新增的 CPU 测试

1. OpSpec/schema fixed/L1/L2 正例与混用负例；
2. logical/resolved/capacity shape 与 buffer sizing；
3. 连续不同 total/count 的输入生成、runtime 校验、digest 和 poison；
4. native Host Program resolved scalar 读写/重放/篡改拒绝；
5. 同一 build 连续执行不重新编译，resolved 输入变化产生不同正确计划；
6. output/final_state 只比较 logical 区域，padding/sentinel 不被误计入或修改；
7. empty batch、超 capacity、非法 cu、尾部踩踏、伪造 resolved、伪造 dispatch 均被拒绝；
8. 现有 native Host Program、V1/V2 CPU、fixed-shape projection、linear JSON bundle 回归；
9. NPU simulator parity（只模拟，不声称真机）。

## 5. 本轮明确不做

- 不运行任何 GPU 测试或 kernel；
- 不运行 `taic run/resume` 或其它付费模型搜索；
- 不修改 `tests/support/golden/shared/shape-portfolio-阶段验收/P2/`；
- 不做 Dyn-1、overlap 或最终 P2 重录。V6 与 P2-F 不在这一轮的授权里；它们后来的记录见 varlen 计划文首；
- 不删除/skip/xfail 既有负例，不放宽断言、容差或发布门。

## 6. rlar 工作方式与交付

Actor：实现上述最小完整 V5 垂直切片，优先合同和连续调用正确性，不做表面兼容壳。
Reviewer：独立检查候选自由、身份、padding、resolved、fallback、timed 边界和 native 优化是否退化，
重点寻找“把 L1 换名冒充 L2”的实现。

交付：实际 diff、测试结果、未解决风险。主 session 随后复核 diff 并决定 GPU 授权验收和提交。
