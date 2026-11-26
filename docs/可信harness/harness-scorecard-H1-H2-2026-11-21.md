# Harness 收益计分卡 H1/H2：四张只读表与自然 run 基线

日期：2026-11-21。计划：[Harness 收益计分卡实施计划](harness收益计分卡-plan.md)。前序：[H0 数据盘点与口径冻结](harness-scorecard-H0-2026-11-21.md)。

## 父任务与边界

父任务是判断 ShapeForge 的 harness 改动能否在多个任务、多个 backend 上带来可复核收益。H1 只改变**观测机制**：新增 `taic/harness_scorecard.py`，复用 H0 的 run-root/cohort/成本/身份口径和现有只读审计，输出四张互不共享分母的表。H2 只改变**数据质量与证据整理**：分析现有自然 run，保存快照并写这份人类摘要。

本阶段没有调用模型、没有跑 GPU、没有启动付费搜索，也没有启动 H3。`publication_allowed=false`；自然 run 的前后差异只写关联，不写因果。清微 NPU 仍只有 offline simulation，不是设备正确性或性能证据。

计分卡里的 `h3_started=false` 只描述**这次只读审计自己**不启动 H3（另有一个语义更明确的 `h3_started_by_this_audit=false`），不表示仓库从未做过 H3。2026-11-22 的配对窗口测量 smoke 是用户另行授权的独立实验，见下方候选 1 与 [预登记](h3-measurement-window-plan-2026-11-22.json)／[结果](h3-measurement-window-result-2026-11-22.json)；本审计不重写、不延伸那份证据。

## 可复现入口与快照

```bash
uv run python -m taic.cli harness audit-scorecard
uv run python -m taic.cli harness audit-scorecard --json
```

实现：`taic/harness_scorecard.py`。测试：`tests/shared/test_harness_scorecard.py`。schema 为 `shapeforge.harness-scorecard.1`。完整快照：[harness-scorecard-2026-11-21.json](harness-scorecard-2026-11-21.json)。

命令只读 `runs/` 与 `targets/`，不调用 `observe.index.borrow()`、`observe.health.collect()` 或 `overlay_activity()`，不写派生库。会话与候选从文件重算；可以复用 `observe/health.py` 的状态常量和现有审计的层名、阈值。

**2026-11-22 重算（同一命令，不是新数据）**：这次只改缺失值与分母口径，run/inventory 计数、账本 `$106.804398`、`model_totals` 已知合计 `$95.398035` 都不变。具体四件事：(1) 一份报告都没有 `benchmark.control` 时 `control_field_coverage_rate` 保持 `null`，旧输出写成 `0.0`；有字段但部分覆盖才按 `reports_total`／H0 最小单位／0.8 覆盖判。(2) `model_totals` 的 `cost_usd=0.0` 在 `sessions_with_cost=0` 时不再是已知成本（两个 unknown `gemm_silu` 分组从 `0.0` 变 `null`），`sessions_with_cost < sessions` 的部分覆盖不再标完整，精确比率的分子改用只含完全覆盖 run 的 `model_totals_cost_usd_fully_covered`。(3) `evaluated_rate`／`no_submission_rate` 的单位是该组全部逻辑会话，未知 outcome 只降低覆盖。(4) 经验比例应用 H0 的 claim 级最小单位 5。文本摘要同时改为：缺失/零回执打印 `null`，只有真实完整零省略才打印 `0`。下文的数字已按新口径更新，并与 `harness-scorecard-2026-11-21.json` 一致。

## 当次盘点

| 维度 | 实测 |
| --- | ---: |
| 合法 run root | 148 |
| finalized / incomplete_or_aborted | 104 / 44 |
| dry_run / unclassified | 0 / 0 |
| natural / controlled / recovery / unknown | 66 / 8 / 0 / 74 |
| evidence_kind real / simulated / unavailable | 63 / 0 / 85 |
| experiment_kind paid / smoke / paid_smoke / unknown | 7 / 0 / 0 / 141 |
| 实验父容器（不计 run） | 3 |

这些数字与 H0 inventory 的 lifecycle/cohort 计数逐项相等。`finalized` 只表示 `result.json` 已落盘，不表示搜索成功；`stop_reason`、候选状态与确认状态分别保留。

## 1. run_efficiency（单位 run）

自然 run 的只读基线：

- 66 个自然 run root：39 finalized、27 incomplete_or_aborted。27 个中断 run 没有 `result.json`，候选计数是 `null`，不补 0。
- 48 个 run 有 `feedback.jsonl`，共 288 个逻辑会话；18 个自然 run 没有反馈文件，因此自然会话总数是**已知部分**，`sessions_complete=false`。
- 39 个有 trace 的 finalized run 中，候选状态：248 `ok`、2 `incorrect`、20 `unchanged`、6 `duplicate`。`unchanged`/`duplicate` 未送评测，不进入 ok 率分母。ok 率 **248/250 = 99.2%**。
- 到第一颗模型 ok：39/39 个 finalized 自然 run 都有非 `prior` 的模型 ok；`prior`-only first-ok 为 0。first-ok 是 run 级，没有按委员会拆开后各算一次。
- `confirmed_status=ok` 有 39 个 run。确认记录文件只有 35 份；这 35 份**全部是 partial identity**，没有一份同时具备 candidate/geometry/bundle/artifact/native/evaluation 六元字段。旧记录里的 `identity.kind=match` 单独计为 legacy match，不冒充 full match。
- 自然配对：12 份未 applied、27 份 applied；可判 **1**、`too_noisy` **26**（`too_noisy` 率 26/27 = 96.3%，同质分母可卷动；可判 1 只是计数，不单独当稳定比例）。结论分级是 `diff`/`no_diff`/`regressed` 算可判、`too_noisy` 算已知不可判、缺失/未知不计 known；自然 27 份 applied 的 conclusion 全部可识别（`paired_conclusion_known=27`、unknown=0）。按 run 分组的 `metric_status` 是 conv2d small（19 个 paired conclusion、0 可判）与 conv2d paper（7 个、1 可判）为 `measured`，gemm_silu（1 个）为 `insufficient`，所以 1/27 不能当成单一指标的充足标记。普通配对没有 `timed_by` 时记 `unavailable`，27/27 都是 unavailable；不把旧 `gpu=true` 冒充 GPU 时间戳。
- 精确 SHA 重测单列：27 份，13 diff、4 no_diff、10 too_noisy；按 seed-decision 口径可判 13/27 = 48.1%（只有 `diff`+方向能支持或反对 replace；`no_diff` 是已知结论但不单独支持 replace），全部 `timed_by=gpu_timestamp`。重测不与自然配对混合，也不把这里的 seed-decision 可判数与 applied-pair 的测量可判率混用。
- A/A 对照：字段覆盖按**报告独立单位**分组判定，实际对照运行另算。自然 289 份候选报告里 **0 份**有 `benchmark.control` 字段，覆盖为 `unavailable`、比例是 `null`（不是 `0.0`），不能记成“0 次不匹配”。全库 16 份报告有该字段但值为 null，分布在 3 个 unknown 分组：10/10 达到 5 个独立单位，覆盖为 `measured`；2/2 与 4/4 不足 5 个单位，覆盖为 `insufficient`；实际 A/A 对照运行仍是 0 次。任意一份报告有字段不再把整组写成 `measured`。
- 成本：66 个自然 run 都有 `spent_usd`；39 个 finalized 合计 **$99.179895**，27 个中断合计 **$7.624504**，全部自然 run 合计 **$106.804398**。65 个 `cost_may_be_incomplete=true`，1 个完整性未知。所有自然分组的 `exact_cost_per_ok` 都是 `null`。9 个自然分组没有模型 ok 候选，只报告 `ledger_spent_without_ok_usd`（合计 $7.624504），不产生每 ok 比值。已知成本 per-ok 的分母只来自成本已知 run：全库 5 个 run 缺 `spent_usd`（都在 unknown cohort、0 个模型 ok），不进入分子或分母，所以自然 run 的数字不变。`model_totals` 同理：31 个有会话 run 缺对应模型的 `model_totals`/`cost_usd`（自然 14），58 个 completed-ok 会话在已知成本分母之外（自然 17）；所有 `model_totals_cost_per_ok` 仍为 `null`，不能把已知账本除以全部会话 ok。
- 账本每 ok 值只作单行参考，不跨 task/shape 求平均：conv2d small（32 run 组）0.329、conv2d paper 三个单 run 分别 0.379/1.322/1.868、flash_attention paper 0.791、gemm small 0.360、gemm_silu small 0.168、relu paper 0.009（美元/模型 ok）。这些是账本估值，不是 provider 对账单，也不是精确成本。
- 27 个自然 run 的 speedup 行带 `missing_confirmation` 标记；1 个带 `mixed_measurement_boundary`。所有 speedup 只列单 run 值，`included_in_average=false`，不进入任何汇总。

受控/未知 cohort 不与自然 run 合并：8 个 controlled run（7 个 paid）保留 experiment_id/fingerprint、cell_id 与 recovery_of；74 个 unknown 仍是 unknown，不按目录名冒充自然搜索或恢复运行。恢复成本按 `recovery_of` 归入目标 run/cell 所在分组：目标组的账本和已知成本 per-ok 分母接收该成本，恢复自己组只保留原始描述、不再计一次；目标缺失或歧义时只记 `unattributed_recovery_cost_usd`，不进任何组账本。本快照没有 recovery run，这些字段全为 0/`null`。

## 2. model_session_efficiency（单位 session）

自然 run 的 288 个逻辑会话，按 provider/model/committee 汇总。底层 19 个自然分组还按 task/dtype/size/lifecycle 等键分开；下表只汇总会话计数，不把不同组的 ok 率合并：

| provider / model | committee | sessions | evaluated | unchanged | agent_failed | unknown | ok 会话 | model_totals 已知成本 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek / deepseek-flash | first | 174 | 162 | 9 | 1 | 2 | 157 | $44.030415 |
| deepseek / deepseek-flash | second | 94 | 82 | 11 | 0 | 1 | 80 | $27.486571 |
| moonshot / kimi-k3.1-pro | second | 13 | 12 | 0 | 1 | 0 | 11 | $20.339772 |
| kimi-code / k3.1-pro | second | 6 | 6 | 0 | 0 | 0 | 6 | $3.541277 |
| moonshot / kimi-k3 | second | 1 | 0 | 0 | 1 | 0 | 0 | 无 |

- 会话结局合计：262 evaluated、20 unchanged、3 agent_failed、3 unknown；**0** 个显式 `agent_no_submission`。`agent_failed` 不等于未交核；即使一条失败会话带 `candidate_status=ok`，也只计入 `failed_outcome_with_ok_status`，不计 `completed_ok_sessions`。`unknown` 不是失败：它留在单位分母里，只降低覆盖。全库 `evaluated_rate` 的单位因此有 6 个分组变化：自然 deepseek/first conv2d paper 中断组 2→3、deepseek/second conv2d small finalized 84→85、relu 中断组 0→1，以及 controlled/unknown 的 3 个纯 unknown 分组 0→1。状态不变（已知单位仍 ≥5 且覆盖 ≥0.8 的三组仍是 `measured`，其余本来就不足）；“5 个已知、2 个 unknown”这种覆盖不足由合成回归固定。第 1 张表的 `no_submission_rate` 用同一分母规则，5 个分组的单位随之变化。
- 会话表候选族：254 ok、2 non_ok、26 not_evaluated、6 unavailable。它与第 1 张表的候选分母**不是同一个单位**：一些 feedback 行没有对应的外层 trace；不把两张表的 ok 数强行对齐。
- `feedback.candidate_status` 与外层 report 冲突 0 次；`.retryN` 去重 0 行。候选 `report.json` 存在时优先于 `result.json` 的 trace，再退回 feedback；若出现冲突，外层报告优先并保留冲突计数。
- `model_totals` 已知聚合成本合计 **$95.398035**，覆盖 269 个会话、11 个模型分组；它只来自 `result.json.model_totals`，与全部自然 run 账本 $106.804398（finalized 部分 $99.179895）是不同单位，**不能相加**。有 `model_totals` 的分组都带 incomplete 标记或完整性未知；所有 `model_totals_cost_per_ok` 为 `null`。已知成本分母只覆盖成本已知的 run：自然 14 个有会话 run 缺对应模型的 `model_totals`/`cost_usd`（其中 1 个是账单样本只覆盖部分会话的 run），17 个 completed-ok 自然会话在分母之外（全库 31 个 run、58 个 completed-ok 会话）。269 个 `model_totals` 会话与 288 个逻辑会话不是同一个分母。完整性按每个有会话 run 是否取得匹配的**有限** `cost_usd` 判定：有匹配条目但缺 `cost_usd` 也算缺成本，不因该 run 只有失败会话而放行；`sessions_with_cost=0` 的 `cost_usd=0.0` 是缺样本，`sessions_with_cost < sessions` 是部分覆盖，两者都不算成本完整。已知合计 `$95.398035` 仍含部分覆盖行的已开单金额；精确比率的分子另用 `model_totals_cost_usd_fully_covered`（自然 **$94.730156**，全库 $125.645637），只含完全覆盖的 run，与 completed-ok 分母同一集合。真实快照中 `runs_with_model_total_without_cost=2`（全库，均为 `sessions_with_cost=0` 的 p2u 恢复行）、`runs_with_partial_model_cost=3`（自然 1、unknown 2），自然分组的条目缺失仍为 0；这三条分支都由合成回归固定。自然 deepseek/second conv2d small 分组的 completed-ok 已知成本分母因此从 74 降到 71，其完全覆盖成本为 $24.769156（已开单 $25.437035）。

## 3. delivery_and_feedback（三种 record 类型，不共享分母）

### delivery_receipt（单位 session）

自然 233 份 `sessions/*/task_delivery.json`：

文本摘要的省略字节只在每个分组都带真实整数时才求和：任一分组因缺字段而是 `null` 时打印 `null`，不补 0；只有完整且真实为 0 的省略才打印 `0`。底层表里的 `None` 没有被改写成 0。

- 静态合同：delivered **3,726,732 B**，omitted **0 B**，违例 0。静态省略仍是必须为 0 的硬条件，没有改写成比例。
- 动态区：delivered **1,022,772 B**，omitted **292,213 B（22.22%）**；145/233 份回执有动态省略；`over_budget=0`，字节账不闭合 0。
- 统一态势块：177 份 delivered、56 份 empty；delivered **597,683 B**，omitted **0 B**。
- 态势三层（`dynamic_budget_audit` 的层名）：`overlay` 0 B、`host_ledger` 0 B、`peer_unverified` delivered 588,656 B / omitted 0。态势块送达不等于模型已理解，也不等于该信息有效；这里没有另开第四层。
- 其他块单独计省略：上一轮 NOTES **176,304/353,521 B（49.87%）**、当前现场 5,386/228,855 B（2.35%）、最近的历史 24,135/26,196 B（92.13%）、现场证据 35,460/44,461 B（79.76%）、自测对照 50,928/64,269 B（79.24%）。
- 自然 run 没有预登记的指定保留项，`specified_retention=not_applicable`。受控记录若在 `runs/` 里带保留清单，只读复述；没有则 `unavailable`。
- 全库 `render-check/` 回执单独计 1 份，不进入会话分母；`agent_work/` 里的 overlay 不是 dry-run 证据。

### claim_evidence（单位 proposal/claim）

- 自然 250 条结构化提案；208 条有显式 evidence_level。按冻结规则归一后：202 条 `model_claim`（其中 42 条未标注，沿用 `summarize_levels` 的默认）、46 条 `model_script`、2 条 `host_process`；0 条 `human_confirmed`、0 条 `simulated`。显式 `host_process` 占显式标签 2/208 = 0.96%。
- 自然 1,586 条反馈 claim：premise `verified` 500、`unverified` 82、`downgraded` 13、空白 991。空白保持 `null`，不写成 0% 已验证；在 595 条显式前提中，verified 84.0%、unverified 13.8%、downgraded 2.2%。
- 13 条被否决前提全部仍在原记录里带正文（13/13），没有因状态变化删除数字。
- 跨 run 重复只按精确 `proposal_key`/`claim_digest`，并按 cohort/实验分组隔离（结构化/不同 key/重复 key/出现次数）：自然 250/244/6/12；controlled 三个实验分组分别 12/10/1/3、5/5/0/0、2/2/0/0；unknown 18/16/1/3；0 条无结构键自由文本。`0445afadadc5ca4d` 在 natural/unknown 各 1 个 run，`9a10fcbfdca2b3ec` 在 controlled 的 3 个 run 和 unknown 的 1 个 run，跨 cohort 不互计。自由文本没有该键时该项是 `unavailable`；本快照没有这种行。
- evidence_level 只挂在单条提案上，不挂在整场 run 上。

### route_m_trace（单位 trace）

- 自然 run 的真实 `harness_route_m.jsonl` 为 **0 条**，自然覆盖是 `unavailable`，不是成功率 0%。
- 全库只有 1 条真实门迹，来自 unknown cohort 的 route-M shadow run：结论 `too_noisy`、五门均 `not_run`、`shadow_written=false`、注入 0 B。它是机制证据，不是路线 M 效果，也不是可交付 shadow。

## 4. experience_and_governance（experience 指针 / trust 逐 grant）

### 经验指针

- 2 个 `targets/*/experience.md`（含空模板），15 节：14 verified、1 reported。
- 16 个指针全部格式有效且存在；0 missing、0 invalid、0 warning、0 simulated section。
- 比例也按 H0 的 claim 级最小独立单位 5 判定：gemm 与 gemm_silu 各只有 1 节/1 指针，`pointer_valid_rate`、`pointer_existing_rate`、`verified_rate` 都是 `insufficient`；只有 unknown-task 分组（13 节/14 指针）仍是 `measured`。少量指针不再算充分证据。
- 13 节是 legacy（缺题目/芯片/落点），任务作用域为 unknown；只有 gemm 与 gemm_silu 各 1 节能落到具体 task。2 节进入人工晋升评审输入，但本阶段不晋升、不改 Confidence、不跨 target 传播。

### 信任

逐 grant 对照当前 `current_prompt_sha1(target_id)`、`current_protocol_version()`、`status` 与 `expires_at`：

| target / backend | task | provider / model | effort | 状态 | scopes |
| --- | --- | --- | --- | --- | --- |
| apple-m4-max / metal | conv2d | deepseek / deepseek-flash | high | active | overlay:premise, overlay:tip |
| apple-m4-max / metal | conv2d | moonshot / kimi-k3 | high | stale_prompt | overlay:premise, overlay:tip |
| apple-m4-max / metal | conv2d | moonshot / kimi-k3.1-pro | high | stale_prompt | overlay:premise, overlay:tip |

- 0 expired、0 suspended、0 revoked。两条 stale_prompt 只说明提示面指纹已变化，不自动停权、不自动重签。
- 没有 target 级 `authorized=true/false`；一条 active grant 不代表同 target 的其他模型已授权。
- `run_config.system_prompt_sha1` 是编排器对当次 `self.system` 的 SHA-1，与 trust 的 `prompt_sha1`（提示面 SHA-256）不是同一个值；本计分卡不互相比较。

## 后端边界与不可判项

- 真实 backend 只有 Metal：63 个 run 声明 real，50 个 Metal run 未声明，35 个 run 连 backend 都缺失（unknown）。跨真实 backend 只有 **1** 个，`cross_backend_status=insufficient`。
- `runs/` 没有 NPU 自然 run；`simulated` run 为 0。NPU 设备指标是 `unavailable`，不从测试 fixture 补模拟行。Metal 专属功能（GPU 时间戳、A/A 对照、fallback provider）与通用 harness 功能（送达字节、信任、经验审计）分栏。
- `effect_assessment=insufficient`：自然 run 有 5 个 task 的 ok 率可判（conv2d、flash_attention、gemm、gemm_silu、relu），但只有 1 个真实 backend；`effect_claim_allowed=false`。
- 明确不可判/缺失：所有自然分组的 `exact_cost_per_ok=null`；自然 A/A 覆盖 `unavailable`；自然配对 timed_by `unavailable`；35/35 确认记录 partial identity；自然 route-M 0 条；NPU 设备指标 `unavailable`；13/15 经验节 legacy/unknown task；18/66 自然 run 缺反馈文件；27/66 自然 run 中断。
- speedup、延迟和每 ok 成本没有跨 task、shape、dtype 或测量边界求平均。自然 run 的前后差异不是因果。

## 后续候选（0–2，均未执行）

**候选 1（测量覆盖，不是 speedup 效果）：配对窗口/轮数 A/B。** 同任务、同候选、同 seed、同模型，在 `conv2d/fp16/paper` 与 `gemm/fp16/small` 各做一组 AB/BA：对照组（不加这项改动的一侧）用现有自然配对协议，实验组（加上要验证的改动的一侧）只加宽同窗口轮数/缓冲区组数。主指标是**可判配对比**与区间半宽；`too_noisy` 不算成功，不发布 speedup。它只验证“当前配对能否做出可判结论”，不验证 harness 提高搜索速度。需要另行批准，预算、停止条件和样本量预先登记；本阶段不设计、不执行。

**候选 1 已于 2026-11-22 经用户 H3 授权后执行**：使用现有候选、无模型调用，control 16 轮、treatment 64 轮，两对固定 SHA 各做 AB/BA，共 8 个 cell。全部完成且两组均为 diff；conv2d 的区间在 AB 中变宽、BA 中变窄，gemm 两种顺序均变窄，但两对 control 本就已全部可判。因此只得到单对 smoke 的 `inconclusive`，`adopt_protocol=false`，不改变默认 16 轮，不产生 speedup 或跨任务效果结论。见 [H3 预登记](h3-measurement-window-plan-2026-11-22.json) 与 [H3 结果](h3-measurement-window-result-2026-11-22.json)。

**不列 speedup 效果候选。** 现有自然 run 只有一个真实 backend，NPU 只有模拟；任何跨 backend 的 harness 收益结论都不可判。应先补第二颗真实 backend（或真实 NPU 设备）上的自然观测，并让新确认记录带上六元身份字段；否则 H3 只能在单一 Metal 任务上重复单算子结论，不能回答父任务。

## 测试与下一步

- `tests/shared/test_harness_scorecard.py`：**52 passed**。覆盖有/无 ok、未交核、prior-only first-ok、未知/不完整成本、已知与缺失成本混合时的每 ok 分母、`model_totals` 条目缺失、条目存在但缺 `cost_usd`、`cost_usd=0.0` 且 `sessions_with_cost=0`、`sessions_with_cost < sessions` 的部分覆盖、完全覆盖/部分覆盖混合时精确分子与分母同一集合、too_noisy、缺失 pair conclusion、no_diff 可判、fallback、缺失回执、双委员会、中断、render-only、模拟 NPU、agent_work 不误为 dry-run、父容器/cell/恢复不双算、resolved+unresolved 恢复归属、agent_failure 不算成功、retry 不双算、feedback 与外层报告冲突、candidate report 优先于 trace、partial identity、静态合同省略、受控保留清单、cross-cohort 提案重复不互计、control 字段缺失/1/5 与 5/5 覆盖、未知 outcome 不离开分母、经验比例的最小单位、文本摘要的 null 与真实零（含零回执时 unavailable）、CLI、真实 runs 只读与现有审计重叠。
- 相关聚焦回归（H0 inventory、N1、动态预算、省略影响/内容、经验作用域、种子决策/重测、路线 M、harness CLI、结构、文档完整性）：**86 passed**；理论记录（`test_feedback_evolution.py`）：**50 passed**。
- `prepare_session` 的顺序回归：`tests/shared/test_experience_overlay_injection.py` 新增 3 个节点——非 seedless 经验摘要被拒时工作目录里没有 seed kernel/report.json/experience.md/device.md，正常渲染时 seed、report.json、files 列表与摘要是原字节，seedless 保留已有的 report.json（旧规则）且不获得种子。旧顺序下第一个节点失败（已独立复现），另两个是保字节/保 seedless 的守门测试。
- 全量安全 CPU 套件跑过一次：**5,677 passed、13 skipped、38 failed**。其中 2 个失败由本工作引入并已修复：计分卡真实 run 重叠断言在并行测试临时出现额外回执时假设了固定文件集；理论记录复核人使用新会话而未沿用测试要求的 self-review 谱系。其余 36 个是既有/环境失败，分类见下。没有删除测试、skip/xfail GPU 节点或放宽断言。
- 口径说明：实现会话留下的记录是聚焦 33、相关回归 95、全量 5,677/13/38。上一轮独立审查新增 2 个回归后（计分卡 + H0 inventory）为 41 passed；本轮按 4 个可复现 blocker 做最小修复，新增 7 条回归（混合成本 run、`model_totals` 部分缺失、条目存在但缺 `cost_usd`、recovery 混合归属、缺失 pair conclusion、no_diff 可判、control 1/5 与 5/5），计分卡 + H0 inventory 为 **48 passed**；另复跑 N1/动态预算/省略/经验/种子/路线 M/harness CLI（**45 passed**）、结构/文档完整性/理论记录（**71 passed**），没有重跑全量安全套件；因此 36 个既有/环境失败仍按实现会话的分类记录，不写成独立复验。
- 重算核对：用当前绝对路径 `runs/` 与 `targets/` 重生成快照后，除 `generated_at_unix`、新增指标字段/说明和两处 unknown 分组的 `control_field_coverage`（2/2 与 4/4 因不足 5 个独立单位从 `measured` 修正为 `insufficient`）外，自然 run 的 run 数、候选数、配对、账本和 `model_totals` 数字不变。没有为“预计不变”保留错误数字。
- 最后一处 recovery coverage 由父会话直接修复：成本比率的 units 现在随 `cost_owner_root` 归属，resolved recovery 的 ok units 只进目标组，unresolved 不进任何成本 units，原始 `model_ok_candidates` 不变。合成回归直接断言目标组 ledger/exact coverage 都是 `{units:2,known_units:2}`、恢复组都是 `{units:0,known_units:0}`；父会话复跑计分卡+H0 为 **48 passed**。真实快照只新增 `cost_units_model_ok` 审计字段，移除该字段后与修复前逐字段一致。
- 既有/环境失败分类：acceptance governance 的 `worktree_head` 未登记（HEAD 33c9c01 无对应 epoch）；bitdelta/gemm_silu 的 trust.toml 锚点漂移（未修改 trust）；observe 模板数值标记、kernel_agent 自检摘要为既有失败；selfcheck/selfcheck_bridge 的 AF_UNIX、进程权限与宿主环境失败；children/run_detached/triton_baseline 的进程权限；track_mt1b 缺 `hmz` 模块；remote_job_engine 在并行套件中超时但单独通过。它们不是本计分卡的代码路径。
- `git diff --check` 通过。没有 GPU、模型、付费搜索或 H3；没有修改 `tasks/`、`tcenv` 任务包、prompt、trust、历史 run、历史验收证据或 H0 冻结阈值。
- 候选 1 已按上述边界执行完毕，不再追加样本。下一步是补第二真实 backend/NPU 自然观测；没有其他 H3 候选被批准或启动。
