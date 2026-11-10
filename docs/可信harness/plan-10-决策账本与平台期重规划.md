# plan-10：决策账本与平台期重规划

状态：**阶段 1 与阶段 2 已在 CPU 与 fake session 落地，开关默认关闭**（2026-11-10 授权这两段。`task_ledger` 默认关，`plateau_after` 默认 0。这不是搜索变快的证据，也没有 GPU 证据。不占 README 第 3 节的 8 个工作包）。核对日期：2026-10-29 的规格，2026-11-03 按当时代码改了复验与账本接入两句，2026-11-10 落地阶段 1、2。默认离线。不调用模型、不使用 GPU/NPU、不跑 `taic`。

## 当前决定（本批做 / 不做 / 完成产物 / 失败去向）

- **本批做**：阶段 1、2 的 CPU 与 fake session 实现和测试。`task_ledger` 默认关，`plateau_after` 默认 0。
- **本批不做**：不把这两个开关改成默认开，不改 `patience`，不打开 `diversity_archive`，不新增常驻 Agent 或第三委员会，不自动重写架构，不建知识库，不加 NCU/msprof 门，不自动 overlay，不扩大 trust，不跑 GPU，不调模型。任务书里出现账本不是搜索变快。
- **完成产物**：`nodes_from_candidates`、`SearchConfig.task_ledger`、`SearchConfig.plateau_after`，以及 `tests/shared/test_task_ledger.py`。
- **失败去向**：离线测试失败时先读 `logs/pytest/triage/triage_summary.md` 及 handoff。

## 1. 目标与非目标

| 目标（阶段 1、2 已在 CPU / fake session 落地，默认关闭） | 非目标 |
| --- | --- |
| 在 fake session 上检查：现有账本文本会不会出现在下一场任务书 | 不把「文本出现了」写成搜索变好、GPU 变快或架构已重写 |
| 检查 `falsified` 提示不要重复，`untested_impl_fail` 提示可以带 hint 重试 | 不新增分类规则，不把编译失败改称为方向已被证伪 |
| 平台期只规划一段架构假设提示；触发由 fixture 显式给出 | 不新增常驻架构 Agent；不改 `patience`；不调用 `select_diverse_parent` |

## 2. 现状证据

2026-11-03 只读复核下面几条，未跑搜索。`classify_outcome` 的分支顺序沿用 2026-10-29 的阅读，本轮未逐行重读该函数；若与代码不符，以符号为准再核，不要把下面的分类顺序当成新测量。

- `SearchConfig.plan_ledger` 默认 `True`。2026-11-03 看到的任务书是最近 5 行历史，看板会读 `plan_ledger`，那不是下一场任务书。2026-11-10 另加了默认关闭的 `task_ledger`：关着时有种子的任务书仍是这 5 行、400 字节；打开后才换成 `format_ledger_for_planner`。`docs/架构/搜索.md` 的字段表记了这个默认。方向行不依附多初稿：`init_samples` 大于 1 或 `task_ledger` 打开时都记 `direction:`。只开账本、初稿仍是 1 时，接力场的标题用记下的方向。两个都关时任务书不要求这一行。900 字节预算内保留最新的完整节点和结尾重试句。未决复验保持未计时，不推进父种子。
- `tests/shared/test_architecture_plan.py` 断言了实现失败、`within_noise=False` 且更慢的 `falsified`，以及 `validated_win`。2026-11-10 在该文件和 `tests/shared/test_task_ledger.py` 补了 `status=ok`、`improved=False`、`within_noise=True` → `validated_tie`。这覆盖分类，不是四类都已经在真实搜索里被计时证明。
- `diversity_archive` 默认 `False`（`taic/llm.py:317`）。`select_diverse_parent` 定义在 `taic/policy.py:292`。本轮在 `taic/` 里没有看到搜索循环调用它，调用点仍以测试为准。`patience` 只负责停止。这两处都还不是平台期重开架构。
- 复验不要写成「计后换输入门没有」。① `differential_recheck` 默认关（`taic/llm.py:315`）。② `confirm_batches=3` 已默认开（`taic.toml:30`，`taic/llm.py:298`，`taic/orchestrator.py:3608`）。③ 秘密 holdout 在未跟踪的 `tcenv/portfolio_validation/`，未接入默认搜索。本文不改这三层。
- 上列是仓库现状。ForgeKernel 的外环重规划、自晋升目标与加速比是**论文口径 / 未在本仓库复现**，不写入验收。数字只以 [导读 §2J](papers/导读.md) 为准，本文不另抄。

## 3. 分阶段步骤

阶段 1 和阶段 2 已按 2026-11-10 的授权写进 CPU 与 fake session。`task_ledger` 默认关，`plateau_after` 默认 0。关着时有种子的任务书仍是最近 5 行、400 字节。这只证明分类和提示文本，不是搜索变快，也没有 GPU 证据。

0. **规格。** 输入是预置 trace 与 history 行，不读真实 `runs/`。退出条件：人确认本文没有把论文加速比写成事实。规格仍留在这里；阶段 1、2 已经往下做了，默认关闭。
1. **三条确定性检查（已落地，开关默认关）。**
   - 分类：不改 `classify_outcome` 的判断顺序。`improved` 为 `validated_win`；`within_noise=True` 为 `validated_tie`；噪声内平局不记成 `falsified`。CPU fixture 是 `status=ok`、`improved=False`、`within_noise=True` → `validated_tie`，在 `tests/shared/test_architecture_plan.py` 和 `tests/shared/test_task_ledger.py`。
   - 任务书：同一份 fake history。对照臂只有最近 5 行历史；处理臂打开 `task_ledger`，换成 `format_ledger_for_planner` 的文本。断言处理臂含账本、对照臂不含。这只证明文本在不在。
   - 重试纪律：处理臂里 `falsified` 带不要重复，`untested_impl_fail` 带可重试。12 条长方向仍要在 900 字节内留下最后一轮和这句。不调用模型观察它是否真的换了方向。
2. **平台期只改提示（已落地，默认关）。** `plateau_after` 默认 0。大于 0 且停滞轮数达到它时，有种子的任务书多一段。论文里的「连续 3 次无新最优」是作者的工程选择，未在本仓库复现，不采用为默认阈值。这段只写：假设是什么、依赖哪几条记录、什么观察会否定它。委员会数量、`harness_auto`、`diversity_archive` 与 `patience` 保持原值。不启动第二个进程。

## 4. 离线验收

可运行项是 `tests/shared/test_task_ledger.py`，以及 `tests/shared/test_architecture_plan.py` 里的 `validated_tie` 断言。通过只说明分类和文本。下表的「明确不算通过」仍然有效。

| 检查 | 输入 | 通过条件 | 明确不算通过 |
| --- | --- | --- | --- |
| 分类回归 | 原有三类断言，加上 `within_noise=True` 的 CPU fixture | `improved` 仍为 `validated_win`；`within_noise=False` 且更慢仍为 `falsified`；新 fixture 要求 `within_noise=True` → `validated_tie` | 把噪声内平局写成 `falsified`，或写成这些 fixture 已经证明真实搜索覆盖了四类 |
| 任务书含账本 | fake history，无 GPU | 处理臂含 ledger 段，对照臂只有最近 5 行历史 | 真实搜索变快，或确认加速比 |
| 重试提示 | 同一份 fake ledger | 两类提示与 `format_ledger_for_planner` 末句一致 | 模型实际没有重复某方向 |
| 平台期提示 | `plateau=true` 的 fake plan | 只多一段假设文字；无新 Agent、无配置扩权 | 自动改内核架构，或打开 `diversity_archive` |

## 5. 风险、回滚与其他 plan 边界

- 账本若被读成事实来源，一次 `ok` 会被当成方向已证实。账本文本只能当规划提示，不能替代 `tcenv` 报告或确认门。
- 把平台期做成第二个 Agent 会扩大执行面。回滚是关掉 `task_ledger` 和把 `plateau_after` 设回 0；默认已经是这样，有种子的任务书仍用最近 5 行历史。
- plan-02 本批改成本账本读法、修好 repetitions、按 pair 身份配对，不收本文实验，也不实现「提高 TASK 预算」。
- plan-05 继续禁止把双 Agent 当独立评审；本文不新增评审角色。
- plan-07 本批只做过滤与人工去重；配方字段不在这里建库。
- plan-08 的回执只证明送达；账本条数和架构介入次数不是效果。
- plan-01 的三层复验保持原状：`differential_recheck` 默认关，`confirm_batches=3` 已开，holdout 未接入。①不是计后换种子确认。分类里已有「第二次种子算错不算 `falsified`」的分支，本文不改该门。
- 已有交错 A/B、`required_gain` / `noise_z` / 确认批次、`stop_at_roofline` 保持原样，不写成新需求。

## 附录：文献与变更索引

| 日期 | 内容 | 来源 |
| --- | --- | --- |
| 2026-10-29 | 登记为研究候选；未授权、未运行 | [ForgeKernel 调研](ForgeKernel-调研.md) §4.2、§7.4；数字以 [导读 §2J](papers/导读.md) 为准 |
| 2026-11-10 | 阶段 1、2 按 CPU 与 fake session 授权落地。`task_ledger` 默认关，`plateau_after` 默认 0。测试在 `tests/shared/test_task_ledger.py`。不是搜索变快，也没有 GPU 证据 | 提升 LLM 内核加速比计划 |

ForgeKernel（[papers/59558_ForgeKernel_Self_Evolvin.pdf](papers/59558_ForgeKernel_Self_Evolvin.pdf)，ICLR 2027 双盲投稿，匿名，无官方代码/artifact）描述了外环：连续无新最优、课程走完或门控僵局时重开架构分析。**论文口径 / 未在本仓库复现。** 摘要是 22 个专家库单元中 18 个；正文另有 29 个 cell 中 25 个；7 个生成系统核另报几何均值 +39.4%。这是三套口径，不能混写，也不能当本文验收。K-Search 有公开的高层/低层分工代码，不能拿它的结果代替 ForgeKernel，本文也不搬入它的难度门或效用函数。
