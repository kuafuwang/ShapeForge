# 案例：开发者工具请求被当成写核 overlay/经验

状态：**已观察的路由误判，修复尚未实施。** 本文是决策证据，不是新计划，也不是对原提案人/评审人的能力评价。
核对日期：事实链观察于 2026-10-26；2026-11-03 按代码现状补了「历史盘上」与「今天重放」两层。未运行测试、模型或 GPU。`inherit_prompt`、`developer_queue`、`route_proposal` 仍是文档概念，代码零命中。

## 1. 事实链

提案：`b4131be13c4750c8`，来源 run
`runs/apple-m4-max/harness-pilot-active-20261022`，`first`/deepseek-flash，`conv2d`。

action：

> Add a `tcenv.cli ab` subcommand that compiles several kernel files, times them interleaved
> in one process over rotating buffer sets, and prints per-file median wall time and cv.

原始记录：

- proposal `surface=tip`、`status=valid`，probe 为 `agent_work/r1.p0/bench_ab.py`，
  SHA-256 `0f86570b…`，`trusted_record=false`。
- review：`useful=true`、confidence `0.8`、reviewer `moonshot/kimi-k3`；
  review 承认探针可核，但明确指出没有保存 13 次交错的原始输出、探针不做正确性检查，
  risk 为 `wrong-metric (mild)`；若进入 `tcenv.cli`，必须标明 timing-only。
- decision：`reviewed_active`，reviewer 同上，随后生成 `shadow=false` 的 overlay 事件。
- 试点事实：后续 r2/r3 TASK 没有真正送达该 overlay；所以本案例**没有观测到**下一场 agent 因这段文本
  产生的实际动作，风险是在路由层面被证实的，不是一次已完成的攻击。
- 人工 promotion：patch `737cb574…`、commit `178de7c`，该 action 被写进
  `targets/apple-m4-max/experience.md:99` 的 `[harness b4131…]` bullet。
- 当前检索未发现 `tcenv.cli ab` 子命令实现；它仍是一条面向开发者的工具请求，而不是写核 agent
  能在 workdir 内使用的陈述。

### 1.1 历史盘上状态与今天重放（2026-11-03）

这两层不能写成同一句「零 overlay」。

| 层面 | 事实 |
| --- | --- |
| 历史盘上 | 提案 `status=valid`、`surface=tip`，无 `kind`，无 `probe_time_verified`，`trusted_record=false`。review `useful=true`，无 `protocol_version`。决策走到 `reviewed_active` → promotion → `promotion_commit`（`178de7c`）。`harness_overlay.jsonl` 有 `shadow=false`、`reviewed=true`、`protocol_version="3"` 的行。`experience.md:99` 仍有这条裸 bullet，没有 `Confidence` |
| 今天重放 | `settle` 在 `taic/harness.py:944` 因 review 缺 `protocol_version` 直接跳过；即便补上，`:975` 还要求 `probe_time_verified is True` 且 `probe_sha256` 非空。按今天的门，这条历史 review **不会新发** overlay。已有的 overlay 行不会因此消失 |

人审已决定：从 `experience.md` 删除该 bullet，事实留在本文与开发者队列，不做「保留 + 加注」。删除会改变 `experience.md`，从而改变 grant 指纹；与指纹扩面、overlay 标题措辞修正合并为一次性重签。本文不改 `experience.md`。

新路由的验收是：**新决策**零 overlay、零 `inherit_prompt`。不要把验收写成「盘上已经零 overlay 行」。

## 2. 为什么这是错误路由

这条建议对**开发者**可能有用：交错计时能减少两次独立 `--timed` 的漂移，review 也验证了探针使用的
API。问题在接收者：

- 它需要修改 `tcenv.cli` 和评测命令，写核会话不能写仓库，只能写自己的 workdir；
- 下一场写核 agent 即使读到，也无法执行“加子命令”，最多写一个本地近似脚本或浪费回合；
- 它被推进成 target experience 后，会作为知识反复出现在后续 TASK；但它不是设备事实、MSL 用法或
  已验证的 kernel 经验；
- review 的 `useful` 回答的是“这条建议是否有价值”，没有回答“谁能在当前权限和范围内执行它”。

所以建议本身未必无价值，**有价值但接收者错了**。

## 3. 当时为什么会被判成可用

1. `acceptance.kind_of` 按关键词把它归到 `tool` 类；
2. `validate_proposal` 允许 `tool/reporting` 作为 `surface=tip` 进入 P1，只拦截评测/预算/信任/协议等词；
3. review prompt 要求判断 useful/actionable/safe，但没有 `recipient`、`requires_repo_change` 或
   `actionable_within_session` 字段；
4. `settle` 只要 review `useful=true`、trust、probe 等门通过，就能生成 `reviewed_active` overlay；
5. 人工 promotion 又把这条 tip 写进经验库，进一步放大了错误对象的生命周期。

这不是“模型说了危险词”的问题，而是**有用性和可执行对象没有被分开建模**。

## 4. 修改 harness 时的决策建议

### 路由规则

- **先判断接收者在当前权限内能否执行，再判断建议是否有用。** 给后来队友的建议必须是它当场能够做到、用得上的内容：当前 scope 内的避坑、路径、探针、测量方法、MSL/算法小知识或边界。证据状态照实保留。要求新增工具、修改 harness/权限/评分、扩大 scope 或长期写入经验的建议，属于系统/开发者需求，不能包装成队友提示。能跑的小命令或路径若读凭据、读错材料或改评分输入，同样不能当队友建议。
- 给队友的建议由宿主先计算 `recipient`、`requires_repo_change`、`actionable_within_session`、scope 和证据状态；后续 LLM 只表示同意、拒绝或无法判断。宿主字段通过且同意后，**在当前任务内自动 `inherit_prompt`**，不逐条人审。渲染保留宿主前缀、scope 和证据状态。拒绝、无法判断或证据不足则不继承。这个同意只把提示放进当前任务，不自动变成系统行为，不授予额外权限，不把 `reported` 变成 `verified`，不恢复 `stale`/`retired`，也不把经验库里的 `reported`/`stale`/`retired` 条目放回提示。工作假设和计划 fixture 见 [plan-06 §3.3](plan-06-注入面与运行时安全.md)。
- 给系统/开发者的建议必须进开发者队列；长期/跨任务采纳由人审核、必要时离线验证后决定。后续 LLM 的同意不代替这道人审。
- `kind_of` 只是证据/验收分类，不是接收者分类；`tool`/`reporting` 只能作为 developer 强先验，不能单独决定路由。路由应由 `recipient`、`requires_repo_change`、`actionable_within_session`、scope 与本场证据决定；`tool`/`reporting`/文档类请求通常进入开发者队列或 P2，**reviewer 的 `useful` 不能覆盖路由**。
- `28c247a285f8a1e3` 是必要反例：它虽被 `kind_of` 归为 `tool`，但 full-unroll pragma 与恒真 thread guard 可直接用于写核，不能只凭 kind 拒绝；它只通过新增路由门，仍须过 scope/probe/trust/never-auto 等既有门，不保证激活。
- 只有同时满足以下条件才可进 overlay：
  - `recipient=kernel_agent`；
  - `requires_repo_change=false`；
  - `actionable_within_session=true`；
  - scope 匹配当前 target/task，且有本场可核证据。
- 进入 overlay 或经验库之前，自由文本先经人审转结构化；转换时由宿主填写接收者，不采用模型自报的“这是给下一场用的”。`inherit_prompt` 不把逐条人审当准入；宿主算不出字段时失败关闭。
- 设备事实/硬件前提走 premise/probe 路径，不以“有用提示”名义绕过测量。

### 数据与观察

开发者队列至少保留：proposal key、action、原始证据、`requires_repo_change`、目标模块、状态
（pending/accepted/rejected/deferred）、关联 P2/commit、后续是否被真正使用。指标应是
“开发者建议 → P2 提交 → 后续使用”，不是 overlay 送达率。

### 验收 fixture

下列期望是计划中的离线 fixture，当前没有对应测试。不要把 `tests/shared/test_proposal_routing.py`、`tests/support/attack_fixtures.py`、`tests/support/golden/metal/attack_fixtures.json`、`tests/backends/metal/test_acceptance_attacks.py` 写成既有基线：2026-11-03 工作区里这四个文件不存在；2026-11-02 现状文档记录它们从未提交。本轮没有重跑 `git log`。

- `b4131…` 的**新决策**必须被路由为 developer，零 overlay、零 `inherit_prompt`，出现在开发者队列，不能自动写 experience。这不要求把盘上已有的 `shadow=false` overlay 行说成不存在。历史 review 的 `useful=true` 已经演示：同意和有用性都不改变这条路由。
- `28c247a285f8a1e3` 作为直接写核正例，只允许通过新增路由门，不保证激活为 overlay。它不证明 `kind=tool` 的请求都能进队友提示。若宿主字段未通过，它也不得 `inherit_prompt`。
- 可执行且 scope 匹配的避坑或探针，按 plan-06 的 `pitfall-probe-in-scope`：宿主字段通过且后续 LLM 同意后，当前任务自动提示采纳；零 overlay，零 experience 写入。
- 有用但本场不可执行的提示不进 overlay，也不 `inherit_prompt`。过期或 scope 不匹配则拒绝。LLM 同意不扩大权限，不把 `reported` 写成事实。
- `tool/reporting` 提案即使 review `useful=true`、有 probe，也不得生成 `reviewed_active` overlay。需要改仓库的请求不因 kind 以外的理由回到 overlay。
- 若开发者工具最终被采纳，走静态 manifest/registry、人审、测试和 digest；不接受模型动态插件。
- `experience.md:99` 的删除已经拍板：删掉这条 bullet，不做「保留 + 加注」。事实留在本文与开发者队列。本文不直接改经验库文件。

## 5. 后续讨论与证据位置

模型分级与常驻审核的完整讨论已写入 `plan-05-独立评审与反共识.md`；`plan-03` 记录 scope/指纹授权边界，`plan-08` 记录 shadow 指标与重启门槛，`plan-06` 记录按接收者而非 kind 的路由。rlar 日志：`/Users/kuafu/.pi/agent/hmz-flow/2026-10-26T15-07-20-741Z-rlar-5fc78b56.log`。

## 6. 证据位置

- `runs/apple-m4-max/harness-pilot-active-20261022/harness_proposals.jsonl`：提案与 probe SHA。
- 同目录 `harness_reviews.jsonl`：`useful=true`、confidence、wrong-metric risk。
- 同目录 `harness_decisions.jsonl`：`reviewed_active`、promotion、approval、commit。
- 同目录 `harness_overlay.jsonl`：`shadow=false` overlay 事件。
- `targets/apple-m4-max/experience.md:99`：被推进成 harness bullet 的 action。
- 相关路由代码：`taic/acceptance.py::kind_of`、`taic/harness.py` 的 proposal/settle/reviewed_active 路径。

## 7. 修复落点（2026-11-03）

`taic/proposal_routing.py::route_proposal` 现已按 action 的动词与对象计算接收者、仓库改动需求与当前可执行性；b413 即使自报 `audience=peer` 也会得到 `developer_queue` 并记录 `intent_conflict`。`settle` 的 quorum、`pending_tip_reviews`、`candidate_patch` 都接同一道门；plan-05 又移除了单来源 `reviewed_active` 的新激活路径。所以这条工具请求不会再因 quorum、有用 review 或人工 promotion 进入 overlay / 自动评审 / `experience.md`。

新增 `tests/support/injection_cases.json` 与 `tests/shared/test_injection_boundaries.py`，并回放本目录真实提案：b413 → developer queue，28c → overlay candidate。历史盘上的 overlay、promotion 与经验库 bullet 没有被这次代码修改抹掉；b413 源条目的删除仍归一次性重签那次处理。
