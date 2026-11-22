# Harness 收益计分卡实施计划

## 父任务

目标不是优化某个算子，而是判断 ShapeForge 的 harness 改动能否在多个任务、多个 backend 上带来可复核收益。Metal 使用真实设备证据；清微 NPU 当前仅允许 offline simulation，不能据此声明设备正确性或性能提升。

本计划不调用付费模型、不跑 GPU、不启动搜索。后续是否开展受控实验，必须根据本计分卡的结果另行授权。

**当前状态（2026-11-21）**：H0 已完成，见 [H0 数据盘点与口径冻结](harness-scorecard-H0-2026-11-21.md) 和 [当次 inventory](harness-scorecard-inventory-2026-11-21.json)。H1/H2 已完成只读实现与自然 run 基线，见 [H1/H2 四张表与基线](harness-scorecard-H1-H2-2026-11-21.md) 和 [计分卡快照](harness-scorecard-2026-11-21.json)；`publication_allowed=false`，自然 run 前后差异不是因果。H3 尚未授权、未开始。

## 非目标

- 不用 conv2d 或任何单一算子的结果代表 harness 收益。
- 不把不同任务、shape、dtype、测量边界下的 speedup 或延迟直接求平均。
- 不把 NPU 模拟当成真实 NPU 证据。
- 不把 search best 当发布加速比；fresh confirmation 与搜索期数字分开。
- 不为了得到“改进”而调整分母、剔除失败 run、把中断 run 排出分母，或把缺失写成 0。
- 不把 `delivered` 当成模型已理解或该信息有效。
- 不打开 `observe` 索引、不写派生库、不新建阶段验收包、不改 79 文件任务包闭包。

## 数据入口

只读使用现有材料：

- `runs/**/result.json`、`state.json`、`run_config.json`、`transcript.jsonl`、`ledger.json`
- `candidates/*/report.json`
- `sessions/*/task_delivery.json`
- `paired_comparison.json`、`paired_comparison_recheck.json`、`paired_host_selection.json`
- `harness_proposals.jsonl`、`harness_overlay.jsonl`、`harness_route_m.jsonl`
- `feedback.jsonl`、自测账本和宿主权威报告
- `targets/*/experience.md`、`targets/*/trust.toml`
- 已有只读审计：`audit-n1`、`audit-seed-decisions`、`audit-seed-recheck`、`audit-route-m`、`audit-dynamic-budget`、`audit-omission-impact`、`audit-omission-content`、`audit-experience`

不重建或修改历史 run。原始 run 缺失字段必须是 `null` / `unavailable`，不得写成 0。

H0 先定义 **run root**，再分桶；不能把任意含标记文件的子目录当成一场 run。规则写进合成测试，目录个数只出现在当次 inventory 里，不写死成验收数字：

- 只有 run root 直属的 `result.json`、`run_config.json`、`state.json`、`transcript.jsonl`、`ledger.json` 或 `candidates/` 是运行标记。根级 `experience_overlay.json` / `render-check/` 只作准备标记。路径中位于 `agent_work/`、`sessions/`、`candidates/`、`.selfcheck/`、`__pycache__` 之内的文件不再向上制造一个 run root。
- 实验父容器没有自己的运行标记时不是 run；嵌套 cell 各自是 run。父容器的 summary、plan 和成本不能与子 cell 再算一次。恢复 cell 只在有显式记录时写 `recovery_of`；其成本归属到记录指定的 cell，归属不明时标 `unavailable`，不猜、不双算。
- `finalized`：run root 有可解析的 `result.json`，且含 `trace` 或 `task`。它只表示结果已经落盘，**不表示搜索成功**。`stop_reason`、候选状态与确认状态另列；`agent_failure` 不得写成 `completed_ok`。这是 `taic/n1_metrics_audit.py` 今天实际计入的集合。
- `incomplete_or_aborted`：run root 没有 `result.json`，但有 `state.json`、`transcript.jsonl` 或 `candidates/`。有账本就计入已知成本与中断，不从搜索分母里拿掉。
- `dry_run`：run root 有根级 `--render-only` 记录或等价准备回执，且没有 `result.json`、`state.json`、`transcript.jsonl`、`candidates/`。它不进搜索分母，单独计数。`agent_work/` 里的 `experience_overlay.json` 不是 dry-run 证据。
- 不符合任何规则的布局进入 `unclassified`，列出路径，不强行归桶。

每个 run root 还必须有 `cohort`：`natural`、`controlled`、`recovery` 或 `unknown`，以及 `experiment_kind=paid|smoke|paid_smoke|unknown`（付费与冒烟可以同时成立）。它们由 `run_config.json` / `harness_experiment` / 实验 plan 的显式记录决定；目录名只用于检索，不是分类证据。只有普通 target 下按编排器标准时间戳布局生成的 run root 且没有实验元数据时才记 `natural`，自定义、嵌套或归属不清时记 `unknown`。paid 只由明确批准/实验记录证明，smoke 只由显式 smoke 标记证明，不用成本或名字猜。旧 plan 没有 `experiment_id` 时保持 `experiment_id=null`，另记 `experiment_fingerprint=plan 字节 SHA-256` 作分组键，不伪造原实验 ID。受控实验、配对冒烟、恢复和嵌套 cell 与自然搜索分开；同一份文件不能既算自然 run，又算实验 cell。

## 计分维度

五维都保留。能从现有文件重算的才出数；不能重算的标 `unavailable`，不发明代理指标。

### A. 搜索效率

- 会话数、候选数、明确 `ok` 候选数与比例。分母沿用 `observe/health.py` 的候选状态：`ok` 进分子；`rejected`、`compile_error`、`abi_error`、`unsafe`、`incorrect`、`runtime_error`、`timeout`、`crash` 分列。`old_format` / `unreadable` / `missing` 是读取状态，`unchanged` / `duplicate` 未送评测，都不进 ok 率分母。
- 到第一颗明确 ok 的轮次。`label == "prior"` 的 ok 不算模型 first-ok；一场里没有模型 ok 则为 `null`。first-ok 是 run 级，不按委员会拆开后各算一次。
- 未交核只认显式 `agent_no_submission`。`agent_failed`、`provider_failed`、`delivery_rejected`、`cost_incomplete` 另列，不并进未交核。
- run 级成本只取 `result.json.spent_usd`；模型级成本只取 `model_totals`。二者分别进入各自的表，不能相加，也不能让实验父容器与 cell 双算。已知账本除以非 `prior` 的模型 ok 候选数得到的是 `ledger_cost_per_ok`；ok 数为 0 时比值是 `null`，旁边给出 `ledger_spent_without_ok_usd`。组内只要有一个纳入统计单位的 run 缺 `spent_usd`，账本分子和已知成本每 ok 分母都只取成本已知的 run，缺失 run 的 ok 不进该分母，`cost_complete` 不得为 true；`model_totals` 同样按每个有会话 run 是否取得匹配的**有限** `cost_usd` 判断完整性（有匹配条目但缺 `cost_usd` 也算缺成本，不能用 completed-ok 缺失间接代替），已知账本估值只用成本已知 run 的 completed-ok 分母。只要所取 run 带 `cost_may_be_incomplete=true`，`exact_cost_per_ok` 就是 `null`，账本均值必须带 `incomplete` 标记；允许当前所有 run 都没有精确成本，不能为出数而剔除它们。恢复 cell 的成本按显式 `recovery_of` 归属到目标 run/cell 所在分组，归属不明时不并入任何组账本；已归属的恢复成本不再在恢复自己的分组重复计入账本或每 ok 分母。`report.json` 的 `elapsed_s` 是评测时间，不是模型墙钟；两种时间分开，缺哪一种就标哪一种 `unavailable`。
- 同窗口可判配对数与不可判配对数。`too_noisy` 是不可判，不是错误。applied pair 的已知结论只包括真实存在且可识别的 `diff`/`no_diff`/`regressed`/`too_noisy`；前三者算可判，`too_noisy` 算已知不可判，缺 conclusion 或未知值计缺失、不进 `known_units`，也不当已知不可判。`paired_comparison_recheck.json` 与自然 `paired_comparison.json` 分开；重测的 seed-decision 可判数不复用 applied-pair 的测量可判率。普通配对只有 `elapsed_s`，不得标成 `gpu_seconds`；GPU 秒只来自 `paired_host_selection.json`，且 session 正本优先于 candidate 副本。

### B. 测量可信度

- `timed_by` 分布。旧报告里的 `gpu=true` 不得冒充 GPU 时间戳；没有 `timed_by` 就是 `unavailable`。
- A/A `--control`、配对比区间、`too_noisy` 与不可判覆盖。历史 `report.json` 没有 `benchmark.control` 时记覆盖 `unavailable`，不记成 0 次不匹配。字段覆盖以**报告**为独立单位：`control_field_reports / reports_total` 全缺为 `unavailable`，部分且低于 H0 的 0.8 覆盖率或不足最小单位数为 `insufficient`，达到阈值才是 `measured`；实际 control 运行（`control_runs`、区间是否含 1）另算，不能因任意一份报告有字段就把覆盖写成 `measured`。
- 身份不匹配只在同一字段两侧都真实存在且冲突时计数。单核确认看 `result.json` 的 `confirmations[*].identity`；六元身份（candidate / geometry / bundle / artifact / native / evaluation）只在报告或确认记录里真有这些字段时核对。`identity_scope=full` 且全部协议要求字段一致才叫 `match`；只有部分字段时记 `partial`，缺失或没有 `build` 的旧报告记 `unavailable`，部分字段一致不得冒充完整 match。
- fallback、不可比 speedup、跨窗口比值和混合测量边界单独标记。这些标记存在时，该行 speedup 不进入任何汇总。

### C. 反馈与知识价值

- 证据等级用 `taic/evidence_level.py`：`model_claim`、`model_script`、`host_tool_in_fence`、`host_process`、`human_confirmed`、`simulated`。等级挂在单条主张、提案或测量上，不挂在整场 run 上。
- 前提分 `verified` / `unverified` / `downgraded`。空白前提是 `null`，不能写成 0%。
- 被否决前提及其数字是否仍留在原记录里，按记录在不在计。跨 run「同一主张又出现」只对有精确 `proposal_key` 或精确 digest 的结构化提案计数。自由文本没有这个键时，该项是 `unavailable`；不得用现场聚类临时定义重复。
- `verified` 经验沿用 `audit-experience`：有效指针、缺失指针、模拟声明、target/任务错配。这是仓库里 `targets/*/experience.md` 的作用域审计，按 target 与任务汇总，不按 model 或 size 拆。

### D. 信息送达

- 静态合同省略必须为 0。做不到就失败，不改成比例。
- 动态实际字节、省略字节和分布，直接读 `task_delivery.json`。态势层名字以 `taic/dynamic_budget_audit.py` 的 `overlay`、`host_ledger`、`peer_unverified` 为准；提示里的 peer tips 文案对应该第三层，不另开第四层。
- NOTES、证据、历史等其他块单独计省略。`situation delivered` 不能代替这些块。
- 自然 run 没有预登记的「指定保留项」。这一项只在带保留清单的受控记录上只读复述已归档结论；自然 run 标 `not_applicable`。字节送达不是已理解，也不是该信息有效。

### E. 治理与后端边界

- 授权是否过期只审计当前工作区 `targets/*/trust.toml`：逐 grant 对照 `taic.trust.current_prompt_sha1(target_id)`、`current_protocol_version()`、`status` 与 `expires_at`，并保留 target/task/provider/model/effort/scope。历史 run 的 `run_config.json` 只有 `system_prompt_sha1`，那是编排器对当次 `self.system` 的 SHA-1，与 trust 里名叫 `prompt_sha1` 的 SHA-256 提示面指纹不是同一个值。两者不得互相比较。历史 run 没有授权留痕时记 `unavailable`。不得生成 target 级 `authorized=true/false`，一个模型的 active grant 不能代表整个 target。
- Metal 专属功能与通用 harness 功能分栏。只有输入文件自身明确声明模拟时，NPU 才输出 `simulated=true`、`publication_allowed=false` 的协议或模拟指标。`runs/` 里没有 NPU 自然 run 时，本计分卡的 NPU 设备指标是 `unavailable`，不从测试 fixture 补模拟行，也不补成设备数字。
- 本计分卡不自动停权、不自动晋升、不把一条经验传播到另一个 target。

## 输出与分组

新增只读入口，与现有 `taic harness audit-*` 对齐。`scorecard` 这个词已经用于 `tcenv/report.py` 的「外层成绩单」和 `tests/shared/test_scorecard.py`，本命令不叫 `harness scorecard`。

```bash
uv run python -m taic.cli harness audit-scorecard
uv run python -m taic.cli harness audit-scorecard --json
```

模块是 `taic/harness_scorecard.py`。测试是 `tests/shared/test_harness_scorecard.py`。输出 schema 是 `shapeforge.harness-scorecard.1`。实现只解析 `runs/` 与 `targets/` 上的文件，不调用 `observe.index.borrow()`，也不调用会借这条写连接的 `observe.health.collect()` / `overlay_activity()`。可以复用 `observe/health.py` 里的状态常量；会话与候选计数在 `taic/` 里从文件重算。

禁止把下面四张表压成一张笛卡尔积。`evidence_level` 只出现在主张表里。`protocol_version` 与 `system_prompt_sha1` 有则记录，没有则为 `null`，不叫 `prompt_sha1`。

1. `run_efficiency`：键是 `target_id / backend / task / dtype / size / committee_signature / cohort / experiment_kind / experiment_id / simulated / lifecycle`。`experiment_id` 为空时使用 `experiment_fingerprint` 作分组键；原始行保留 `cell_id` 与 `recovery_of`。装 first-ok、finalized 与中断、`stop_reason`、确认状态、search best 与 confirmed speedup 的分列覆盖。`committee_signature` 来自 `run_config.json` 的 first 与 second，不把整场 run 算到 `result.json["model"]` 一个模型上。
2. `model_session_efficiency`：键是 `target_id / backend / task / dtype / size / provider / model / committee / cohort / experiment_kind / experiment_id / simulated`，同样以 `experiment_fingerprint` 兜底分组；原始行保留 `cell_id`。会话结局来自 `feedback.jsonl` 的 `label / outcome / candidate_status / provider / model / committee`；`sessions/` 只提供送达回执，`model_totals` 只提供该模型的聚合成本。`feedback.label` 与候选/result trace 的 join、`.retryN` 和恢复会话去重规则必须在 H0 固定；`candidate_status` 与外层 `report.json` 冲突时以外层报告为准并记录冲突。一场双委员会 run 可以在两个模型行里各有会话，但 `runs` 计数留在第 1 张表。
3. `delivery_and_feedback`：保留三个逻辑 record 类型，不共享分母：`delivery_receipt` 的单位是 session，`claim_evidence` 的单位是 proposal/claim，`route_m_trace` 的单位是 trace。共同分组键至少有 target/backend/task/cohort/experiment_id；session label、proposal key、trace id 是各自的行身份。送达字节、主张证据等级和路线 M 门迹分别进自己的 counts/ratios，不能拼成一张混合分母表。
4. `experience_and_governance`：经验指针按 target/task 汇总；trust 必须另列逐 grant 明细，再按 target/backend 卷 active/stale/revoked 等状态数量。不能输出 target 级授权布尔值。

每组有 `coverage`、`counts`、`ratios`、`notes`，以及原始路径和生成时间。`evidence_kind=real|simulated` 与 `metric_status=measured|unavailable|undecidable|not_applicable|insufficient` 是两个独立维度，不能写成互斥四选一。H0 按**指标与独立统计单位**分别冻结具名 insufficient 阈值和最小覆盖条件，交给合成测试；看完自然 run 之后不再改阈值。run 数达到阈值但相应字段覆盖不足时仍是 insufficient。

可以向上卷的只有同一 record 类型、同一 cohort、同质分母的运行比例：省略率、`too_noisy` 率、`host_process` 占比、未交核率，且卷动时保留下层 `insufficient` 标记。speedup、延迟、每 ok 成本不跨 task、shape、dtype 或测量边界合并。

## 实施步骤

### H0：数据盘点与口径冻结

- 先固定 run root 发现、子目录排除、实验父容器/cell、恢复归属和 cohort 判定，再按 finalized / incomplete_or_aborted / dry_run / unclassified 列出当前 `runs/`。
- 写出缺失字段、协议版本、真实或模拟 backend；finalized 不等于成功，`stop_reason` 与确认状态独立保留。
- 把 ok、未交核、first-ok、成本覆盖、配对可判、证据等级、身份 full/partial 字段路径、feedback join/retry 去重和每项 `insufficient` 阈值固定成测试。冻结值：run/session/claim/delivery-receipt/paired-conclusion 都至少需要 5 个独立已知单位，已知字段覆盖率至少 0.8；跨任务至少 2 个 task，跨 backend 至少 2 个真实 backend；静态合同省略仍要求 0，不能套用 0.8。
- H0 的可复现只读命令是 `uv run python -m taic.cli harness audit-scorecard-inventory [--json]`；模块为 `taic/scorecard_contract.py` 与 `taic/scorecard_inventory.py`，当次 inventory 保存为 `docs/可信harness/harness-scorecard-inventory-YYYY-MM-DD.json`。它只输出 run root/lifecycle/cohort/字段覆盖，不生成 H1 四张计分表。
- 核对 `system_prompt_sha1` 与 trust `prompt_sha1` 不会被写成同一列；trust 按 grant 而不是 target 布尔值输出。
- 某指标无法从现有文件重算时，标 `unavailable`；当前全部成本不完整或没有 NPU 自然 run 都是合法结果。

### H1：只读计分卡实现

- 模块只放在 `taic/harness_scorecard.py`。shared 测试不 import Metal 或 NPU backend。
- 复用而不重写：`n1_metrics_audit` 的 run 覆盖与配对隔离，`dynamic_budget_audit` 的层名与字节，`omission_content_audit` 的省略正文，`experience_scope` 的指针审计，`seed_decision_audit` 与 `route_m_recheck_audit` 的可判和门迹。候选结局与前提空值沿用 `observe/health.py` 已有常量，不从索引再算一遍。
- JSON 与人类摘要都分别展示 `evidence_kind=real|simulated` 和 `metric_status=measured|unavailable|undecidable|not_applicable|insufficient`，允许 real+unavailable、simulated+insufficient 等组合，不压成互斥四栏。

### H2：自然 run 分析

- 按四张表描述当前基线。只写关联，不把自然 run 的前后差异写成因果。
- 列出 0–2 个适合做后续受控实验的 harness 改动。0 个是合法结果：此时写明该先补哪几道题的自然观测，而不是为了凑数指定实验。
- 计分卡命令只打到 stdout。当次快照另存 `docs/可信harness/harness-scorecard-YYYY-MM-DD.json`。人工摘要写入 `docs/可信harness/实验索引.md`、`docs/可信harness/架构与理论.md` 的「理论与实践对照」，以及 `docs/代理反馈.md`。蓝图里说的 N5 就是这张对照，不另建同名文件。
- 不新建 `tests/support/golden/shared/` 阶段验收包，不改任务包闭包。

### H3：后续受控实验（不在本计划授权内）

只有 H2 写出至少 1 个跨任务候选后才设计：同任务、同候选、同 seed、同模型，按模型分层 AB/BA。Metal 与 NPU 分开。NPU 仍要真实设备授权才能谈设备效果。预算、停止条件、主指标预先登记。

## 测试与验收

- 合成 run 覆盖：有 ok、无 ok、未交核、first-ok 只有 `prior`、未知成本、全部 `cost_may_be_incomplete`、`too_noisy`、fallback、缺失回执、双委员会、无 `result.json` 的中断目录、根级 `--render-only`、模拟 NPU 声明、没有 NPU 自然 run。
- 增加回归：`agent_work/` 里的 overlay 不得变成 dry-run；实验父容器/cell 与恢复成本不得双算；`agent_failure` 不得写成 completed_ok；`.retryN` 不得双算会话；feedback 与外层报告冲突必须显式记录；部分身份一致不得写 full match。
- 比例分母、缺失值、四张表及 delivery/claim/route-M 三种 record 不串线、speedup 不被平均、`system_prompt_sha1` 不与 trust 指纹比较、`peer_unverified` 不另计成 peer tips；没有 NPU 自然 run 时不得从 fixture 补行。
- trust 必须逐 grant 区分 task/model/scope/prompt/protocol/status，不能因同 target 的一条 active grant 而把其他模型写成已授权。
- 真实 `runs/` 只读，并与 `audit-n1`、`audit-dynamic-budget`、`audit-experience` 的重叠计数一致。不钉死当前 run 数。
- shared 测试不 import backend，也不产生 `observe` 数据库写入。
- 文档完整性、结构、相关审计测试通过。
- 最终交付包括：实现提交、测试数字、真实计分卡摘要、仍不可判的项、下一步是否需要付费及理由。

## 父任务汇报要求

每个阶段结束都同时说明：父任务总目标、相关已完成项、剩余项、本阶段改变的是机制、数据质量、证据还是效果，以及下一步和证据位置。不能只汇报改了哪些函数。
