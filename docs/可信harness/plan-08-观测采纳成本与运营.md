# plan-08：观测、采纳成本与运营节奏

状态：**本批只做渲染时回执与离线 S0；周报和 $40 试点冻结。** 默认不调用模型、不使用 GPU。

## 当前决定

- **输入**：`render_contract` 的实际 TASK/overlay 组装结果、`TASK_MAX_BYTES` 与各块 cap、run 配置、proposal key 与 `omitted` 列表。`_cap_task` 只负责截断，省略清单在 `render_contract` 生成。
- **本批只做**：在渲染当场写 host delivery receipt。回执包含 run/session/proposal keys、TASK/overlay SHA、bytes、omitted、reason，以及预算和各块 cap。实际生效的 TASK 预算（`TASK_MAX_BYTES`）与各块 cap 另写入本场 `run_config`，delivery receipt 记录同一组值。每场动态内容的 `prompt_surface_digest` 也写在回执里，不写进 grant 指纹。用构造好的 run 状态离线验证 treatment 含目标 overlay、control 不含。
- **成本口径**：只记录仓库里真有的数：美元、token，以及自检 GPU 秒。不把活跃小时、wall-hours 或 GPU 小时汇总写成已有指标。`first_eligible_*` 是到首个合格候选的累计量。美元的权威来源是 `transcript.jsonl` / `ledger.json`，不从 `feedback.jsonl` 读 `cost_usd`，因为该字段不存在，也不在这里双写。
- **本批不做**：每周报告、S1 的 3 对付费搜索、$40 授权单、Brier/ROI 或自动运营动作。不新建 `runs/*/events.jsonl`。不按窗口把 TASK 预算填满。完整旧设计见[归档](archive/plan-08-观测采纳成本与运营.md)。
- **完成产物**：上述回执规格、本场 `run_config` 中的预算与各块 cap 记录，以及离线 S0。回执代码和测试当前不存在，不得写成已通过。失败先读 `logs/pytest/triage/triage_summary.md` 及 handoff。

## 当前代码核对（2026-11-02 事实，2026-11-03 复核，供审阅，未实施）

- `render_contract`（`taic/kernel_agent.py:614`）在 `:763-793` 生成 `omitted`，并在 `:790-793` 写中文省略注记。`TASK_MAX_BYTES` 在 `:56`，是 12000。各块 cap 写在 `:734`（成绩单 1000）、`:736`（bound 300）、`:739`（失败 1000）、`:743`（NOTES 1800）、`:746`（peer 1500）、`:748`（历史 400）、`:750`（证据 600）、`:757`（自测 400）。预留是 `:767` 的 200 或 420。这些数没有写进 append-only 宿主回执，也没有写进 `run_config`。
- 本地在 `taic/kernel_agent.py:1097-1099` 把 `TASK.md` 写入会话目录，并在 `:1100-1115` 调用 `prompt_surface_digest`（`taic/workspace_env.py:1079`），得到 `PreparedSession.prompt_surface_sha256`。这个哈希覆盖 TASK.md 与 flow 的 `skills/`，不是 append-only 回执，也不记录 omitted、预算、各块 cap 或 overlay 行键。远程 `WorkspaceReceipt.prompt_surface_sha256`（`taic/workspace_env.py:497-537`，本轮未逐字段重读）在 2026-11-02 现状里同样不含这些字段。本批回执补的是预算、各块 cap、omitted/reason 和 overlay 行键；每场哈希仍不进 grant 指纹。
- `runs/*/events.jsonl` 不存在。真实事件流是 `transcript.jsonl`、`budget_events.json`、`ledger.json`，以及 `harness_proposals.jsonl`、`harness_overlay.jsonl`、`harness_decisions.jsonl`、`harness_reviews.jsonl`、`harness_adoption.jsonl`。不要为了观测新造一条 `events.jsonl`。机全局的 `gpu_events.jsonl` 和 `data/review_events.jsonl` 是别的账，不是 run 事件流。
- 成本只有美元和 token，外加自检 GPU 秒：上限 `kernel_agent.py:115` 的 120 秒，累计在 `:1294`。`first_eligible_cost_usd` 与 `first_eligible_elapsed_s`（`taic/harness_experiment.py:21`）是到首个合格候选为止的累计量。没有活跃小时汇总，没有 wall-hours 汇总。`taic/brief.py` 全文没有 `cost_may_be_incomplete`；`taic/orchestrator.py:1099` 把该字段写成 `True`。
- `feedback.jsonl` 没有 `cost_usd`（写入在 `taic/orchestrator.py:2846-2881`）。回执和 S0 不从那里取美元。改读 transcript/ledger 是 plan-02 的事。
- `brief.harness_overlay` 与历史 active 事件只能作为观测输入，不能当作 overlay 改善搜索的效果证据。`shadow=false` 也不等于已送达。

## 为什么回执先于周报和付费试点

`render_contract` 在组装时已经生成 `omitted` 列表。回执在渲染当场写下预算、各块 cap、实际字节和省略原因，比事后解析 TASK.md 更直接，也能区分 `delivered / omitted / no_subsequent_session / disabled / scope_mismatch / unknown`。active 事件本身不等于送达。`runs/*/events.jsonl` 不存在，回执不靠新建这条文件来证明送达。

S0 是确定性的渲染检查：构造冻结 run 状态即可在本地验证，不需要模型或 GPU。S1 按当前规则只能得到
`inconclusive`，且没有决策能合法挂在结果上，因此不应花 $30 去执行；$40 的其余部分也不能被包装成
确认性校准预算。

当前 P1 流量很少，周报大多会是空的。真正已有改进的是自由文本经人审进入 P2 的路径。回执以后可以观察这条链是否真被送到下一场；本批不另建使用情况追踪，也不把 overlay 事件计数当成效果。

## shadow 评估指标与重启门槛（2026-10-26 讨论结论）

以下指标不进入本批实现。本批仍只有渲染回执与离线 S0，不写周报，也不花 $40。路由 replay 若要做，归 plan-06 的离线分流，不由本文另开实施。

在决定是否给模型分级或引入常驻审核前，可以做不激活的离线路由 replay 和人工标注。`false overlay` 是**反事实指标**：在冻结 replay/激活模拟中，把人工标为 developer/unknown/不可本场执行的条目判为“会生效”的比例，分母只含该类条目；shadow 实际注入恒为 0，不能用实际 0 证明路由有效。

其余指标为 recipient/actionability 准确率与漏掉有效 tip、人工推翻率（分母为已有最终人工标签的全部 shadow 判定）、人工分钟、每有效 P2 成本；均附小样本区间。与确定性路由的 κ 只作诊断，不能当正确性硬门。人工标注与真实流量分开计数，不能用当前 12 条提案支持模型能力排名。

shadow 对照必须是同一冻结 bundle：确定性路由、fresh reviewer（无记忆）、常驻分类器查先例前盲判、常驻分类器查先例后判断；隔离被测条目及推翻结论，明确各臂是否共享上下文，并对顺序 anchor 设对照。当前提案全部来自 deepseek-flash，故 shadow 不设 model 因子；若要模型分级，另需 role 分层多模型盲对照和单独授权。

本批周报仍冻结。重启周报需先有稳定渲染回执和真实 P1 流量；重启自动 review 另需满足 plan-05 的四项条件；模型分级或常驻审核的上线均须单独具名批准，不能从 replay 直接晋升。

## 新增文献对回执字段的补充（2026-10-27）

本批最小回执**设计**是 `run/session/proposal keys + task/overlay SHA + bytes + omitted + reason + TASK_MAX_BYTES + 各块 cap`。每场动态内容只进 `prompt_surface_digest`，不进 grant 指纹。下表是以后评审回执 schema 时的候选字段，**不现在扩建周报、S1 或自动运营动作**，也不把论文里的活跃小时写成仓库已有的列。

| 后续候选字段 | 参考文献 | 可解决的问题与限制 |
| --- | --- | --- |
| `track/skill_version`、device/compiler/资源指纹 | [CANNBot-CAKE](papers/2610.31034_cannbot-cake.pdf) | 判断旧观测是否仍适用；版本变化后应降权或重测，而不是沿用旧收益 |
| `validation_stage`、`fallback`、`abstain` | [CANNBot-CAKE](papers/2610.31034_cannbot-cake.pdf)、[AgentCompile](../paper/2606.07665_agentcompile-an-llm-guided-compiler-for-direct-cuda-inference.pdf) | 区分生成核、库回退与无法选择；回退不能被记成生成 kernel 的收益 |
| `predicted_effect` 与下一轮 `observed_effect` | [AHE](papers/2604.25850_agentic-harness-engineering.pdf) | 判断建议是否按预测起作用，避免只统计 active/overlay 数量 |
| 每个 target 的正确性、性能与跨机 max/min speedup 差异 | [KernelBenchX](papers/2605.04956_kernelbenchx.pdf) | 防止跨机平均掩盖失效；该比值不是重复测量的统计方差，只作分账建议，不把论文结果当本仓库测量 |
| 成本、能力回退、infra/unknown 与不可判 | [SoL-Pi](papers/2609.20519_sol-pi.pdf)、[CANNBot-CAKE](papers/2610.31034_cannbot-cake.pdf) | 成本不能只记 token；infra 失败与模型失败分账，unknown 不进入成功率分子 |
| `search_best` 与绑定 kernel/target 的 `confirmed` | [AgentCompile](../paper/2606.07665_agentcompile-an-llm-guided-compiler-for-direct-cuda-inference.pdf)、[CANNBot-CAKE](papers/2610.31034_cannbot-cake.pdf) | 只借生成/运行准入与 local/official 分账的类比；本仓库仍按既有确认门区分 search best 与发布值，不搬论文阈值 |
| 显式分母、forbidden-effect 与正常场景误伤 | [ContextOS 协议笔记](papers/articles/contextos_agent-harness-benchmark-protocol.md)、[evals 与 harness 工程笔记](papers/articles/hugobowne_how-evals-are-central-to-harness-engineering.md) | 博客只提供指标写法建议，不作为独立效果证据 |
| token、GPU 小时、活跃小时、wall 小时分列；参考强度与「每组取最优战役」的选择效应分开记 | [ForgeKernel](papers/59558_ForgeKernel_Self_Evolvin.pdf) | **论文口径 / 未在本仓库复现。** 作者例子：DSA score recompute 3.1 活跃小时；循环 KDA prefill 97.8 活跃小时 / 34 GPU 小时；2D 卷积前向 19 次架构介入。这些单位彼此不能换算，也不能换成 CAKE 的 token 账。论文把对 CAKE persistent KDA 的 +0.4% 称为 parity。同一 DSA fp8 indexer 对强/弱参考得到的内核相差 18%，打赢弱基线不能写成接近强实现。每组取最优战役是选择效应，不能当效应量。回执仍只证明送达；成本、复用条数和架构介入次数都不是效果。不据此做周报、$40 或真实效果校准。昇腾 8 个四级算子的几何均值 +33.1% 只能在 CANN-Bench 托管服务复现，不能当本仓库回执的校准值 |

## 重启条件

先有稳定的渲染回执与真实 P1 流量，再考虑周报。只有当一个新的决策确实需要探索性配对结果、
且该结果有合法用途时，才重新评估付费搜索；真实确认性校准另见[确认性效果实验-另纸.md](确认性效果实验-另纸.md)，
不由本文授权。
