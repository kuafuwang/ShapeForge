# plan-08：观测、采纳成本与运营节奏

> 状态：设计计划，不是实验结果或付款授权。默认只做只读报表、离线回放与 dry-run；任何付费模型、GPU、apply/commit、grant 都必须另行具名批准。
## 当前决定

- **本批只做**：首版只读周报，仅含 delivery（含 omitted/无后续）、成本 unknown、五轴原始计数和 `as_of`；分母只保留一张 2 合法行 + 2 坏行示例表。
- **输入**：run 的 TASK、proposals/reviews/decisions/overlay、实验记录、ledger 与既有 adoption 文件。
- **本批不做**：完整质量统计、Brier/ROI、自动运营动作或付费试点。$40 若另获授权，只证明送达与账单完整，不能产出“值得继续”的确认性证据。
- **完成产物**：可下钻 run/proposal/TASK 的只读周报；探索窗口结束默认暂停新付费，后续确认另开授权单。
- **失败去向**：先读 `logs/pytest/triage/triage_summary.md` 及 handoff；周报不得隐式刷新索引或追加 adoption。

## ① 角度与问题

现在这个方向能回答“发生过哪些提案、review、overlay、promotion”，却不能稳定回答“本周是否值得继续投入”。过程记录不等于实际送达，实际送达不等于搜索效果，搜索效果也不等于扣除实现与人工运营成本后仍值得采纳。本计划只建立一个运营闭环：**按 target/task 分层看产出、过程、质量、效果和完整成本；每周由人决定继续观察、授权一个小试点，还是暂停。** 当前证据支持的默认决定是保留 P0/shadow 与离线建设，不扩大 active；这不是“方向已失败”的结论，因为尚没有一次可核验的 active 完整 run。

## ② 目标 / 非目标

### 目标

- 做一份带 `as_of`、数据覆盖和 unknown 的周报，原始计数与比率并列，能下钻到 run、proposal、experiment 和 TASK。
- 对每条提案分开记录 schema 有效、shadow/active/reviewed_active、实际注入、人工采纳、固化、效果结论及分阶段成本。
- 用最小看板改动暴露“决定了但没送进 TASK”、失败、缺账单、人工分钟和 TASK 字节，而非再做一个总分。
- 形成每 run/每周复核清单、信任与证据待办，以及一个默认 dry-run、分段解锁的付费试点授权草案。

### 非目标

- 不重做 plan-02 的配对统计、效应阈值或功效分析；本文只引用其 verdict 和成本语义。
- 不制定 plan-03 的授权/撤权策略、plan-05 的 review rubric、plan-04 的证据签名或 plan-09 的文档治理。
- 不自动重签、采纳、固化、提交或恢复 overlay，不把历史 run 拼成对照，不将 speedup 强行折算成美元；不默认联网核对供应商账单（需单独授权），账本覆盖不足只报已知部分和 unknown。

## ③ 现状证据

- 默认 `harness_auto=false`、`harness_shadow=true`、`harness_review=true`（`taic.toml:22-24`）。即使 auto 关闭，结构化提案仍会记录；只有打开后才 settle/review（`taic/orchestrator.py:2483,2491,2497`）。所以默认是“可观察、未自动结算”，不是已有 active 线上效果。
- 单场 `board()` 已汇总 proposals/reviews/decisions/overlay、acceptance milestones、adoption 和自由文本建议（`taic/harness.py:1654-1691`）；`GET /api/v2/harness/<run>` 只是转发这份文件事实（`observe/query.py:757-769`、`observe/app.py:363-374,722`）。
- `taic harness status|shortlist` 可读已有事实；`harness adoption` 虽不调模型/GPU，却会追加 `harness_adoption.jsonl`（`taic/cli.py:925-930,1395-1398`、`taic/adoption.py:154-160`），不能冒充严格只读周报命令。
- `taic/adoption.py` 已有 review/实现/实验成本、五个独立里程碑和阶段耗时；未发生阶段为 0，已发生但缺账为 unknown（`taic/adoption.py:55-108`）。`known_adoption_cost_usd` 只是已知部分，不是完整总成本；shortlist 的 evidence score/known cost 也只适合排队（`taic/adoption.py:133-150`）。
- `observe/trust.py` 已按 target/task/provider/model 给人工审查计数、Wilson 下界，并分开系统建议合理率与 premise 实测率（`observe/trust.py:27-103`）。`brief` 已有搜索花费、模型汇总、发布状态等紧凑事实，但还没有 harness 摘要（`taic/brief.py:104-154`）。
- 模型费用页已有 `sessions_with_cost/sessions` 覆盖；非空 KPI 不证明账单完整（`docs/架构/可观测.md:127-129`、`docs/架构/数据层.md:215-231`）。当前没有人工分钟、逐场 TASK/overlay 字节和宿主投递回执；检索未发现这些字段，属于**已核对的缺失**，不是 0。

| run | 已记录成本（快照口径） | 能说明什么 |
| --- | --- | --- |
| `harness-pilot-shadow-20261022` | $2.6658，4 场 | 0 条结构化提案；只说明 shadow 未误触发，不说明 active（`docs/可信harness/试点结果.md:11,21,37-44`）。 |
| `harness-pilot-shadow-20261022b` | 搜索 $0.7584，review $0.1063，2 场 | 有 1 条 valid tip；`reviewed_active` 生成时 run 已结束，没有后续 TASK（`docs/可信harness/试点结果.md:65-91`）。 |
| `harness-pilot-active-20261022` | 初次停止 $0.7996，随后快照 $2.4661；续跑当前 `result.json` 为 $7.1034873 | 有 review/promotion 链，但没有 active 质量结论（`docs/可信harness/实现交接.md:116-132`；`runs/apple-m4-max/harness-pilot-active-20261022/result.json:7-13`）。 |

唯一较完整的端到端例是 `b4131be13c4750c8`：增加 `tcenv.cli ab` 交替计时命令的 tip，经独立 review useful（实际 $0.118）后成为 `reviewed_active`，再本地 promotion，最终记录 `commit:178de7c`；当前是 target 经验的一条 bullet（`runs/apple-m4-max/harness-pilot-active-20261022/harness_reviews.jsonl:1`、`runs/apple-m4-max/harness-pilot-active-20261022/harness_decisions.jsonl:1-4`、`targets/apple-m4-max/experience.md:99`）。这证明“建议可以被人固化”，不证明工具已经实现，也不证明 overlay 改善搜索。

关键反例就在磁盘上：该 key 有 `shadow=false` 的 overlay（`runs/apple-m4-max/harness-pilot-active-20261022/harness_overlay.jsonl:1`），但 r2/r3 的 TASK 明确列出“本运行 harness overlay”被省略（`runs/apple-m4-max/harness-pilot-active-20261022/agent_work/r2.p0/TASK.md:155-157`、`runs/apple-m4-max/harness-pilot-active-20261022/agent_work/r3.p0/TASK.md:155-157`）；续跑 r4–r6 也同样省略（`runs/apple-m4-max/harness-pilot-active-20261022/agent_work/r4.p0/TASK.md:146-148`、`runs/apple-m4-max/harness-pilot-active-20261022/agent_work/r5.p0/TASK.md:146-148`、`runs/apple-m4-max/harness-pilot-active-20261022/agent_work/r6.p0/TASK.md:146-148`）。旧汇总文案却写“Only active entries reached TASK.md”（`experiments/harness_pilot.py:121`），因此 active 事件绝不能当投递证明。当前渲染器虽已优先保留 overlay 且有满契约合成回归（`taic/kernel_agent.py:720-798`、`tests/shared/test_kernel_agent.py:452-482`），也不能事后把旧 TASK 改写成已送达，更不能替代新 run 的逐场回执。

## ④ 指标体系：五个轴，不合成分数

统一口径：周报同时给 `proposal_records` 与去重 `proposal_key`；版本变化用 `proposal_key + spec_sha256 + change_sha256` 绑定。每个比率都显示分子/分母、target/task、provider/model、protocol epoch、时间窗和缺失数；空分母显示 `—`，不是 0。

为避免“修好分母”，每个比率同时报 `universe_n / eligible_n / observed_n / excluded_n` 与 `exclusion_reason_counts`，并满足两个不变式：**`universe_n = eligible_n + not_applicable_n`** 且 **`eligible_n = observed_n + excluded_n`**；**只有该指标自己的 `observed_n` 进它的分母**，`not_applicable_n` 两份都不进。分类是**「指标 × 记录」**的函数：同一批文件在 `schema_valid_rate` 与 `parse_ok_rate` 下会落进不同格子，所以下面三条规则要按「对哪个指标」来读。进入/离开规则必须冻结：

- 某记录对**该指标**不适用（例：无 review 的提案不计 `review_useful_rate`，非实验 run 不计 effect cell 指标）→ 进 `not_applicable_n`，永不计入 `eligible_n`。同一记录可以对指标 A 是 N/A、对指标 B 是 eligible，所以 N/A 是「指标 × 记录」的属性，不是记录级属性。
- 该指标的 eligibility **尚未**满足（指标窗口未关、artifact 还不该存在）→ 记 `not_yet_observed`：它是 `not_applicable_n` 的**子计数**（属于「现在还不适用」），不进 `eligible_n`/`observed_n`/`excluded_n`，分母显式标「未就绪」；窗口关闭后仍缺 → 移出 N/A，进 `excluded_n` 并带 reason（如 `missing`），不允许无限期挂在 `not_yet_observed` 里。这样 `eligible_n = observed_n + excluded_n` 始终成立，不必引入第三项。
- 在**该指标 eligibility 已满足**、且该指标把这类行当「不可用」的前提下，`missing`/`parse_error`/`infrastructure_failure` 进该指标的 `excluded_n`，按冻结政策计 failure 还是 unknown（`parse_ok_rate` 的例外见下：坏行对它是「已观测到解析失败」，进 `observed_n`）；eligibility 未满足时不得提前计入 `excluded_n`，只能按 `not_yet_observed`/N/A 处理。`not_applicable`（含子计数 `not_yet_observed`）与这三类分开统计，不得互相吸收，也不得与 `eligible_n` 重叠。原因枚举固定为 `missing`（应存在但文件/字段缺）、`parse_error`（行不可解析）、`infrastructure_failure` 三类，不增第四类文件缺失原因。必要输出或轨迹缺失、基础设施失败不得静默删除，计 failure 还是 unknown 必须按冻结的指标定义，effect cell 的 invalid 仍归 plan-02。**这是四种可选政策之一，不是唯一正确做法**：Harmony App. C（p.18）把基础设施失败的任务从该次 run 排除；RRSI App. B.3（p.17）让缺交付物使其应满足的 criterion 全失败；RRSI App. B.6（p.18）用完整 480 分母并把基础设施失败计失败。周报必须写明采用哪一种、为什么，以及跨 run 比较时口径是否一致。

### 4.1 产出

- `schema_valid_rate = valid 记录数 / 成功解析的提案记录数`，分母是该指标的 `observed_n`；`parse_ok_rate = 成功解析行数 / 总行数`（含空行与坏行），是另一个指标。坏行在前者进 `excluded_n`，在后者是已观测失败，两个分母不得互换，数值只按下面的唯一示例表核对。来源是 `harness_proposals.jsonl`，解析器当前会跳坏行（`taic/adoption.py:13-24`），故“0 提案”必须先区分“无记录”和“不可读”。

**同一批数据在两个指标下的分类（数值验收，2 条合法行 + 2 条坏行、无空行）：**

| 指标 | universe_n | eligible_n | observed_n | excluded_n | N/A | 比率 |
| --- | --- | --- | --- | --- | --- | --- |
| `schema_valid_rate` | 4 | 4 | 2 | 2（`parse_error`） | 0 | 2/2 = 100% |
| `parse_ok_rate` | 4 | 4 | 4 | 0 | 0 | 2/4 = 50% |

读法：`schema_valid_rate` 只统计**成功解析**的提案记录，坏行对它不可用，进 `excluded_n`；`parse_ok_rate` 统计**解析结果本身**，坏行「已观测到解析失败」，进 `observed_n` 而不是 `excluded_n`。两种指标各自都满足 `eligible_n = observed_n + excluded_n` 与 `universe_n = eligible_n + not_applicable_n`，但**分母不得互换**。

边界补充分类（`schema_valid_rate` 的 `universe_n` 基数 = 提案记录行，即合法行 + 坏 JSON 行，不含空行，与上表一致）：①**空行**：不是提案记录，不进 `schema_valid_rate` 的 `universe_n`；但它是 `parse_ok_rate` 的一个被观测行（解析结果为空/失败），所以同一数据加 1 条空行时，parse 指标变 `universe=eligible=observed=5, excluded=0`、比率 `2/5=40%`，schema 指标仍 `4/4/2/2`、`2/2`。②**整个文件 `missing`**：两个指标都进 `excluded_n`+`missing`。③**`infrastructure_failure`**：按冻结政策计 failure 或 unknown，仍留在该指标的 `excluded_n`。

注意：本表只是**口径示例**，不是对某次真实 run 的测量；数值验收应固化成 fixture（§⑨）。
- 每 session 的去重提案数按 surface/kind 分层，分母来自 `feedback.jsonl`/observe sessions；有效结构化率、review useful、人工采纳不得混称“有效率”。它不能回答建议是否有价值，也不能回答模型为何没写提案。
- `review_useful_rate = useful=true 的独立 review / 已有二值 verdict 的 review`，来源是 `harness_reviews.jsonl`；现有 `adoption_summary.useful_rate` 则是“到 useful review 且账单已知”的 milestones/valid records（`taic/harness_experiment.py:595-603`、`taic/adoption.py:113-128`），两个分母不得互换。

### 4.2 过程

- 按 key 报 `shadow_only`、`active_quorum`、`active_reviewed`、`converted`、`implemented`、`experiment_validated`、`local`、`committed`；来源为 `harness_decisions.jsonl`、`harness_overlay.jsonl`、`harness_proposals.jsonl`、`harness_reviews.jsonl`、`code_promotions/*.json`、`harness_promotions/` 与 `harness_control.json`。
- shadow/active/reviewed_active 是决策状态，converted 是 `origin=conversion` 的人工转换，local/commit 是不同固化事实；现有 `adoption_rate` 实为 committed/valid records（`taic/adoption.py:113-128`），不是完整漏斗转化率。legacy 可出现 committed=true 而 local=false，不能强行补阶段（`taic/harness_experiment.py:595-603`）。
- 新增 `delivery` 轴：逐 `(proposal_key, session_label)` 报 `delivered / no_subsequent_session / omitted / disabled / scope_mismatch / unknown`，来源为实际 `agent_work/<label>/TASK.md` 及待补的宿主回执。它能回答文本是否到达，不能证明模型读过、理解或遵从。
- 每 key 另报 `origin_target`、`origin_task`、`declared_scope`、`delivered_scopes`、`scope_evidence=origin_only|cross_scope_observed|unknown` 和 `out_of_scope_delivery_count`；scope 证据缺失写 unknown。`cross_scope_observed` 只表示送达，不表示泛化有效；只有 `origin_only` 时不得称 general。

### 4.3 质量（首版只报原始计数，其余指标后续）

- 自由文本 claim 的人工采纳率沿用 `reasonable/reviewed`，其中 reasonable 是 adopted/done；被推翻率用 `refuted/reviewed`，系统建议另用 `system_*` 同口径；来源是 `feedback_review.json` 投影后的 SQLite（`observe/feedback.py:14-15,1603-1663`、`observe/trust.py:45-96`）。pending 不进分母，Wilson 下界只提示小样本不确定，不是质量分。
- `measured_rate=verified_premise/premise` 只表示前提实测覆盖，不是建议有效率或搜索收益（`observe/trust.py:97-103`）。proposal 级人工采纳率只消费 plan-02 的具名 adopt/reject/defer/rollback adoption verdict，落地前为 unknown；现有 `harness_adoption.jsonl` 是汇总快照，不能反推出人审结论。
- review 与人工一致率只消费 plan-05 的 `review_quality_observation`：报 dataset/rubric 版本、exact agreement、selective agreement 与 coverage、matched_n/unlinked_n；uncertain 不硬塞二分类，defer/pending/效果 inconclusive 也不算“不一致”。人工 adopt/reject 是采纳决定，不冒充 reviewer 质量真值；该观察落地前此项为 unknown（`docs/可信harness/plan-05-独立评审与反共识.md:45-59,92`）。
- 另报“采纳后推翻/回滚数 ÷ 曾被人工采纳且已有事前定义后续观察窗的提案数”；`effect_validation.validated=false` 不自动等于命题被 refuted。上述质量轴不能回答长期正确性、reviewer 是否真独立或跨任务泛化。
- 轨迹审计按 issue 类型报 `frame_n / sampled_n / audited_n / unavailable_n / pending_n`、`sampled_n/frame_n`、`audited_n/sampled_n` 和 `confirmed_issue_trace_n/audited_n`；空分母为 `—`，unavailable 不算无问题。这些覆盖率/问题率不合成质量分，也不表示 trace 可用于训练。

### 4.4 效果（只对接 plan-02）

- 周报引用 experiment id、主指标、完整 pair 数、`supports_effect / refuted / inconclusive / invalid`、护栏和独立效果复核；来源为授权 plan、`harness_experiments/*.result.json`、`harness_decisions.jsonl` 及发布门的 `result.json/confirm_report.json`。判定、配对和阈值由 plan-02 所有（`taic/harness_experiment.py:483-577`；`docs/可信harness/plan-02-可证伪预测与效果实验.md:65-117`）。
- 不把同 run 激活前后、search best、单次 confirmed 或历史方便样本当 treatment 效果。没有匹配 experiment 时写“未测”，不是 0，也不由 adoption committed 推断有效。

### 4.5 成本与上下文开销

- 一次性采纳成本分 `C_review`、`C_implementation`、`C_experiment`：分别读 `harness_reviews.jsonl`、`code_promotions/*.json`、去重后的 effect/experiment 记录；实现规则沿用 `taic/adoption.py:55-108`。任一已发生阶段缺账，则完整成本 unknown；已知子总额只能标“已知下界”。
- `C_experiment` 是双臂总 `spent_usd`，其中 additional review 是分解/护栏，不能再加一次；搜索总花费与采纳成本也分行，避免把同一 ledger 重复相加。效果成立后仍需另做成本采纳判断，口径遵循 plan-02（`docs/可信harness/plan-02-可证伪预测与效果实验.md:109-117`）。
- run 搜索成本读 `result.json/state.json`、`ledger.json`、`transcript.jsonl` 的 agent 行与 `budget_events.json`；后两者回答不同问题，budget event 不是完整账单。`run_config.cost_note` 明确预算不是硬发票上限（`runs/apple-m4-max/harness-pilot-active-20261022/run_config.json:83-84`）。
- 人工分钟按 review、diff/实现、实验复核、周运营分列：review 复核时长消费 plan-05 的观察，其余来自具名 adoption 记录，由人补录而非模型自报，同人同段不双算；未填是 unknown。现有 `elapsed_s/stage_times_s` 是墙钟，不可当人工工时（`taic/adoption.py:38-52,99-108`）。
- TASK 成本报每场 UTF-8 `task_bytes`、`overlay_bytes`、截断/省略块及占比，来源为 TASK 与宿主回执；同时保存 task/overlay SHA，而非把正文放进周报。字节不是 token，更不能直接推断质量或美元。
- 仅在成本全已知且分母非零时给“每去重提案/每人工采纳/每效果验证”的已知成本；美元与人工分钟双轴呈现。任一项 unknown 或效果未测时不报 ROI、回本期或“便宜”。

**为什么不合成：** 提案数会奖励 JSON spam，useful/采纳率会奖励琐碎安全建议与宽松评审，commit 数会奖励易写文档，低成本会奖励漏账和少 review，单看 speedup 又会漏掉搜索成本与异常轨迹。加权总分还会把 unknown 变成隐性 0、把小样本和不同单位压成可游戏的目标；因此保留五轴原始向量、分母和例外清单，shortlist 只用于排队，Wilson 只提示不确定性，二者都不触发授权（`docs/可信harness/架构与理论.md:411-419`）。

## ⑤ 看板与报告的最小改动建议

1. **先加只读周报命令。** 建议 `taic harness weekly --since ISO_TIME --target TARGET --task TASK --format md|json`（拟新增，当前不存在）：默认 stdout、不追加 adoption、不改 trust/control/run；以 run 文件为事实源，记录 `as_of`、源指纹和覆盖。若使用 SQLite，显式刷新与生成报告分成两步并记录 index revision；`taic query`/trust metrics 本就可能更新派生索引（`docs/架构/数据层.md:413-415`）。
2. **周报固定七块：** 人工作出的继续/暂停决定（含五轴逐项 `continue|hold|stop|unknown` 与理由）、target/task 分层计数、提案阶段与 delivery/scope、质量 matched set、plan-02 效果链接、美元/人工分钟/TASK 字节、异常与轨迹抽查。运行中的 run 与已结束 run 分开，不能混成分母。
3. **复用现有单场 endpoint。** `board()` 已带 adoption，先在面板增加“决策 / 实际送达 / 未送达原因”“known cost / unknown”“人工 verdict / local / commit / effect”列，不新增写按钮；delivery 数据补齐后再进同一 JSON（`observe/app.py:363-374`）。
4. **brief 只增紧凑异常摘要。** 可加 proposals、active、delivered、cost_unknown、pending_human 计数和 harness 文件路径，不塞建议正文，不替代 `harness status`；保持现有无模型/GPU 语义（`taic/brief.py:86-104`）。
5. **显式呈现坏数据。** missing、parse_error、cost_unknown、minutes_unknown、delivery_unknown、index_stale、失败 cell 各自成行；不能用 `or 0.0` 抹平。旧 pilot 的 review 汇总正有这个缺口（`experiments/harness_pilot.py:81`）。
6. **轨迹抽查而非只抽查赢家。** 每 run 全查 active、review/人工分歧、缺账和失败；其余在 `as_of` 冻结的抽样框按 `target×task×arm×run_outcome` 分层（非实验记 `arm=non_experiment`），事前固定周 seed、人审预算和各层 `sample_n`，成功与 0 提案 run 也可入样。逐层报 frame/sampled/audited/unavailable；非空层抽 0 或未完成只表示未覆盖，不表示无异常。

   人工本地打开对应 TASK、transcript/session、candidate 与评审证据，核对 `TASK/overlay SHA → 工具命令与结果 → candidate/评测证据 → 最终声明`，登记 `evidence_chain=complete|incomplete|not_applicable|unknown`、issue 类型与处置；另查工具误用、找基准答案、凭据/支付异常和“分数好但过程错”。聚合 JSON 不复制正文或秘密；incomplete 先列审计异常，不自动改效果 verdict。

## ⑥ 运营节奏

### 每 run 结束后

- 先 `taic brief RUN`：看 stop_reason、spent/budget、cost 覆盖、published 与 search_best 的区别；再 `harness status RUN`：看提案、review、decision、overlay、adoption，不为“生成周报”追加 adoption。
- 对每个 active key 先构造预期 overlay，再核对后续每场 TASK 的规范化文本或 SHA；没有后续场次与确实被省略分开记账。只要 delivery 未证实，该 run 不进入 active 效果样本。
- 人复核全部新 proposal、转换和 review 分歧，记录 verdict、证据与人工分钟；错误或有害条目立即用现有 `harness disable`，信任暂停/撤销另走 plan-03。
- 对齐 search ledger、review/实现/实验账单；失败与 unknown 保留，不能为了周报“结账”而填 0。发布数字仍只走既有确认门（`docs/架构/可观测.md:3-8`）。

### 每周一次（建议固定同一工作日，具体日期由运营人定）

- 冻结 `as_of` 后生成周报，先处理 unknown/失败/待审，再看五轴和 Wilson；不得先看最好 run 再选择分层或分母。
- 人工决定前逐轴填写 `continue|hold|stop|unknown`、证据和取舍；任一轴的改进不得抵消另一轴的显式 stop 或未满足的授权门，不投票、不求均值。尚待试点采集的证据可保持 P0/dry-run，但不能视为已通过 paid gate。
- 完成第⑤节的轨迹抽查，登记发现、处置和未覆盖部分；重复出现的已 done 建议、local 未 commit、commit 未效果复核各进待办，不自动晋升。
- 对 `declared_scope` 超出 origin 的 key 核对 `scope_evidence` 与 `out_of_scope_delivery_count`；delivery 只证明到达，`transfer` 没有匹配效果证据就为 `not_measured`，`origin_only` 不得称 general；越 scope 投递进入暂停流程。
- 运行 `trust status`/`trust audit`，逐项检查 target/task、provider/model/effort、scope、prompt/protocol、expires、evidence_run；`trust metrics` 需要刷新索引，单列为运维步骤。提醒重签/证据更新不等于自动重签，身份和恢复规则归 plan-03（`taic/trust.py:540-625`）。
- 证据提醒只看可核差异：prompt/协议/目标文档变化、artifact 丢失、旧 spec/patch、后续反证；生命周期影响报告与退场归 plan-07，时间不新本身不证伪，不能按年龄自动作废或续期（`docs/可信harness/plan-07-经验知识生命周期.md:76-85`）。
- 人写下一周决定与上限：继续 P0、授权哪个 dry-run、是否申请付费、何时复评。没有“超过某总分就自动继续”。

### 何时暂停

- **立即停 active：** 错误/有害 overlay、越 scope、合同或安全/正确性回归、疑似凭据或基准作弊；先 disable 相关 run，再按 plan-03 暂停授权，保留全部证据。
- **停止后续付费 cell：** 任一 treatment 缺投递证明、control 被污染、配置/patch/协议漂移、必要账单或授权要求的轨迹缺失、`out_of_scope_delivery_count>0`、超过授权承诺或无人可做独立复核。修好并重新 dry-run 前不得续跑。
- **暂停原授权下的跨 scope 扩展：** 到预设复评点仍只有 `origin_only`，或 scope/效果证据 unknown，就保持 `transfer=not_measured`，不得把原授权扩展到新 target/task；如需测量只能另建具名 grant 并先 dry-run。这不等于 refuted，也不停止原 scope 的 P0/shadow。
- **长期价值证据与本次探索窗口分开。** `offline_only` 的长期价值证据是宿主验收与人工实用采纳；`paired_required` 则是 plan-02 的 `supports_effect`，它要求另行完成确认性校准并达到 `confirmatory_min_pairs`。本次 $40 探索窗口**没有被设计成能产出该证据**：不得因窗口里没有 `supports_effect` 就宣称方向被反驳，也不得因窗口完成就宣称值得继续。窗口结束后默认暂停新的付费 cell；若以后要继续，必须另开真实校准/确认实验的具名授权单，不能把本次 $40 当成那张单的预算或证据。完整美元/人工分钟超过授权上限，或探索性 inconclusive 达到事前上限，也停止后续付费，但同样不等于 `refuted`。保留 P0/shadow 和只读分析；小样本 Wilson 很低本身不是“证明无效”。

## ⑦ 下一次付费试点设计（默认不执行）

**本节全部真实执行均为「需人工授权：付费模型 + GPU」。本文不授权执行；默认只做离线回放和 `execute-experiment` dry-run。现有 runner 主要物化 `treatment.patch`，run-overlay delivery 仍需先按 plan-02 补齐并验收（`experiments/harness_runner.py:23-83`、`docs/可信harness/plan-02-可证伪预测与效果实验.md:37-40`）。**

### 前置与假设

- 选一条**新的**、scope 明确的 `paired_required` 提案：reproduction/expected 完整，独立 useful review 账单已知，实现经宿主验收，patch/overlay 固定，trust 当前有效。不得直接复用 `b4131be13c4750c8`：它已进经验文件，可能同时污染 control；旧 run 也不能当对照。
- H0（送达）：treatment 从首个适用 session 起含同一 overlay SHA，control 不含；无 overlay 被省略，必需合同块保留，TASK 不超既有字节预算，逐场账单完整。H0 未过，不许谈 H1。
- H1（探索性观察）：在冻结 target/task/dtype/protocol、base/config、委员会/effort 和 seeds 下，仅改变该 overlay，按 plan-02 事前冻结的主指标、方向/最小效应与护栏报告原始配对点和描述性区间。指标与阈值只用于防止事后挑选，**不是本窗口的成功线**；本文不编造数值，也不要求 3 对达到确认性结论。
- 首个 live cell 前，授权单还须冻结 `permitted_target`、`permitted_task`、必需 TASK/receipt/transcript/ledger artifacts、轨迹抽样框与 seed、审计人及独立效果复核人；任一空缺只能 dry-run。审计默认人工本地完成并计入人工分钟上限，不新增付费 judge；所有 S0/S1 live cell 均「需人工授权：付费模型 + GPU」。

### 样本、对照与预算（均为建议上限，未授权、不是成本预测）

- **S0 送达冒烟：** 1 对 fresh seed、2 个 cell，只验证 H0 和账本，不作效果结论，seed 不再用于 S1。建议每 cell 记录账本上限 $5（只沿用此前小试的初始授权量级，不是花费预测，`docs/可信harness/试点结果.md:12-13,65,95`），S0 上限 $10。
- **S1 探索性效果试点：** S0 全过后另授权并冻结 3 对新的 fresh seeds（6 个 cell），沿用 AB/BA；建议每 cell $5，S1 上限 $30，第 4 对必须重新授权。3 对是 `exploratory_min_pairs`，只能产出原始配对点与描述性区间，**唯一允许的效果结论是 `inconclusive`/不确定**（这是 plan-02 的规划规则；当前实现只有 `min_pairs`，3 对仍可能被判 `supports_effect`，所以本节不能当现状引用）；执行、送达或账单不可信仍按 `invalid`，二者都不能解锁采纳。三对零宽或极窄的 bootstrap 区间没有可声称的覆盖率，既不得记 `supports_effect`，也不得记 `refuted`，更不得作为采纳依据；真实确认性校准尚不存在时，S1 不解锁任何确认性结论或采纳链；inconclusive 不得为追显著性自动加样本（`docs/可信harness/plan-02-可证伪预测与效果实验.md` §4、§7.7、§9）。
- 新模型花费总授权上限建议为 **$40（S0 $10 + S1 $30）**，包含 cell 内搜索及任何 review；不另设隐藏 review 池。试点前的 review/实现成本仍进采纳账，若还需新付费实现/review，则停止并重报全口径预算，不能绕到 cap 外。
- 人工分钟上限、单笔/总超支处置也必须在授权单填写；未填不执行。$40 是记录账本的授权上界，不是供应商发票保证；缺账或超额即停止后续 cell，供应商对账另行授权。
- 两臂同 base commit、初始 kernel、完整配置、预算和协议，control 无该 overlay，treatment 仅有固定 digest 的 overlay；不能只改 `harness_auto` 让其他 review/提案行为也成为差量。任何额外 active overlay 都使该对 invalid。

### 成功、失败与中止

- S0 成功要求每个 cell 的 host receipt 与实际 TASK SHA/字节一致且 control 阴性，授权要求的轨迹与账本可读；任一缺失、错配、截断、越 scope 或缺账即失败，不解锁 S1。
- **S1 的完成标准只是证据包完整**：S0 的送达条件在全部 6 个 cell 持续成立，control 阴性断言成立，必需轨迹齐全，且账单没有 unknown；随后如实保存原始配对点、描述性区间和 `inconclusive`。它**不能**产生“效果成功”或“值得继续”的结论。以后若另开获授权的校准与确认实验，plan-02 `supports_effect` + 护栏通过 + 独立效果复核才可称效果成立；“值得采纳”还需完整成本、人工分钟和具名 adoption 决定，不能自动 commit。
- 本授权单不产生 `supports_effect` 或 `refuted`。反向点估计只作描述；样本不足、区间未过线或 3 对零宽/极窄区间都记 `inconclusive`。delivery、绑定或账单不可信记 `invalid`。失败 cell 不得删除后再算；任何反向或支持性确认词都留给那张尚不存在的确认实验授权单。
- 出现安全/正确性回归、未授权内容、轨迹中的支付/凭据/找答案行为、预算或人工上限触发，立即中止并 disable；恢复需要新的具名授权。执行批准人不能成为唯一效果复核人（`taic/harness_experiment.py:606-625`）。
- 只声明该 scope/time window 的结果；seed holdout 不是 task holdout，跨 task/target 必须另建 grant、scope 和实验，不在这次 $40 内顺手外推。

## ⑧ 分阶段步骤

1. **A（本批）：首版只读周报。** 只实现 delivery（含 omitted/no_subsequent/unknown）、成本 unknown、五轴原始计数和 `as_of`，可下钻 run/proposal/TASK；不改 run、不追加 adoption、不隐式刷新索引。
2. **后续阶段。** host receipt、完整质量/轨迹面板、dry-run live gate、S0/S1 和运营复评分别立项。$40 若另获授权，窗口只以送达与账单完整为完成标准；确认性校准/确认实验按[确认性效果实验-另纸.md](确认性效果实验-另纸.md)另行申请。

## ⑨ 验收标准（离线测试 / 命令）

本批只验收首版周报：

- delivery 区分 delivered/omitted/no_subsequent/disabled/scope_mismatch/unknown；active 事件不能当送达，历史 replay 只能标 replayed。
- 成本 missing/unknown 不变 0，未发生阶段与已发生缺账分开；缺账不报 ROI。
- 五轴保留原始计数和 `as_of`，不合成总分；N/A、not-yet、missing/parse_error/infrastructure_failure 只按 §④ 的唯一示例表与边界验收。
- 同一 `as_of` 输出确定，不写 run、不追加 adoption、不隐式刷新索引；失败先读 `logs/pytest/triage/triage_summary.md` 及 handoff。

```bash
# 现有回归；新周报测试与命令实现后才可运行
uv run pytest -q -n 0 tests/shared/test_adoption.py tests/shared/test_observe_trust.py tests/shared/test_kernel_agent.py
uv run pytest -q -n 0 tests/shared/test_harness_weekly.py tests/shared/test_harness_delivery.py
# 实现后：uv run python -m taic.cli harness weekly --since ... --format md
```

完整轨迹抽样、S0/S1、真实付费/GPU、Brier/ROI 与后续运营动作均不是本批验收。

## ⑩ 风险与回滚、边界、参考论文

| 风险 | 防线与回滚 |
| --- | --- |
| Goodhart、选择性分层、总分驱动刷提案 | 五轴不合成、分母/as_of 固定、轨迹抽查；撤销排序或面板不改变原始数据，也不自动改变信任。 |
| unknown 被当 0、成本双算、人工工时被墙钟替代 | unknown 阻断 ROI 和下一付费阶段；回滚只下线报表，append-only 原始账不删不改。 |
| active 事件或文本子串被误当送达 | 实际 TASK + SHA + host receipt 双重核对，control 阴性；误投递立即 disable，新证据不足则暂停付费。 |
| overlay 挤掉合同/成绩/失败反馈 | 报 TASK 字节和 omitted 必需块；越界即停止注入，回滚是关闭 overlay，不重写旧 TASK。 |
| 索引陈旧、历史 schema 漂移、提醒疲劳 | 文件为源、显式 as_of/index revision、legacy unknown；周报合并重复展示但保留总数和 backlog，不能静默丢弃。 |
| 轨迹含秘密或敏感行为 | 周报只存路径/SHA/计数，人工本地查看；敏感发现按安全流程处置，不进入公开示例或模型上下文。 |

**边界：** plan-01 管正确性/安全/发布门；plan-02 管实验设计、统计 verdict、成本采纳语义与 rollback；plan-03 管 grant、重签、暂停和恢复；plan-04 管证据完整性/签名，plan-08 只消费投递与证据 id；plan-05 管 review 质量和人工一致性的判定方法；plan-06 管注入与运行时隔离；plan-07 管经验生命周期、退场和迁移；plan-09 管文档新鲜度与治理。本文拥有聚合口径、覆盖/成本呈现、周报、运营节奏和付费授权清单，不在接口处另造平行事实。

**论文只提供方法警示，不提供本仓库收益数字：**

- HarnessDev（2609.01437，§4.3 pp.11–14；§5 是 Related Work）：Evolution 有增益但不稳定、迁移是混合的——五个 self-runtime 声明在 held-out 上提高 +1.43~+4.44（均值 +3.11），固定 Gemini 后只有 Opus 提高、其余三者回退；同一 commit 的 pair-score 波动约 ±4.75，可见反馈分数很少选中 held-out 最优版本。因此 S1 不把「多迭代几轮」当处理变量，也不把某次多轮增益当方向有效的证据；但也不能简写成「Evolution 是负结果」。
- AI Agents That Matter（2407.01502）：成本—效果要联合报告，holdout 不足会过拟合；因此 S0 seed 不复用于 S1，也不把采纳数当收益（`docs/可信harness/papers/导读.md`）。
- HAL（2510.11977）：21,730 次 rollouts 中，更高 reasoning effort 经常更差；LLM 辅助日志检查发现找基准答案和滥用信用卡。这里只借鉴“按 effort 分层 + 看轨迹”，其规模不是本试点目标，也不暗示本仓库已发生这些行为（`docs/可信harness/papers/导读.md`）。
- RuVerBench（2606.29920）：judge 噪声明显，多数投票收益递减；因此周报保留 review/人工分歧并抽查，不靠增加 reviewer 票数刷一致率，新增票必须先证明边际价值（`docs/可信harness/papers/导读.md`）。

## 附录：文献与变更索引

只作查阅，不再逐轮重述正文；论文数字均未在本仓库复现。

| 日期 | 影响位置 | 证据文件 |
| --- | --- | --- |
| 2026-10-25 | 文献对应的正文规则与边界 | `2609.01437_harnessdev.pdf`、`2610.15150_harmony-harness-multiobjective.pdf`、`2609.24972_rrsi-regularized-harness-rsi.pdf`、`2607.15524_recursive-harness-self-improvement.pdf`、`2407.01502_ai-agents-that-matter.pdf`；其余见 `papers/导读.md` |
| 2026-10-26 | 当前决定、阶段、验收与跨 plan 归属 | `审核意见-plan-01至09.md` |
