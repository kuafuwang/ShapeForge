# plan-05：独立评审与反共识

状态：**自动评审已降为 advisory 并默认关闭（2026-11-03）；不建参考集、不建常驻批准者。** 默认不新增付费 reviewer、不调用 GPU。

## 实施记录（2026-11-03）

- **决定**：选择“降级为 advisory + 默认不自动调用 reviewer”，不是只改配置掩盖机制问题。`settle` 已移除单来源 `reviewed_active` 的**新激活路径**；有用 review 仍写入 `harness_reviews.jsonl` 供人阅读，但不再生成 overlay。历史 `reviewed_active` 行仍可按原证据读取与渲染。
- `SearchConfig.harness_review` 与 `taic.toml` 均改为 `false`：在 plan-05 的四个重启条件满足前，默认搜索不自动花评审费。人工显式 review 仍可使用。
- `build_review_prompt` 明确告诉 reviewer：它的结论只供人参考，不会自行激活 overlay 或改变权限。双来源 quorum 的既有 scope/probe/trust 门不变；plan-06 的 recipient 路由仍先于 quorum。
- **离线验收**：相关 harness/conversion/injection/review/llm/report_schema 共 104 项通过；workflow/CLI/kernel_agent/trust 另有 50 项通过。唯一失败仍是已记录的 `test_harness_workflow.py` 历史 fixture 写 `task='relu'` 而当前 `V1_TASK='rope'`，与本次机制修改无关。
- **未做**：模型分级、常驻审核、参考集、Brier、语义 quorum、自动轮换、更多 debate、自动重签与指纹扩面实施。`inherit_prompt` 已由 plan-06 实现为 TASK 内队友复核；review 摘要只作参考，不恢复自动激活。

## 当前决定

- **输入**：历史 5 次 review、201 条人工 claim、提案内容与实际 overlay 路由。
- **现状**：人工 rejected/refuted 为 0，参考集没有负例；quorum 从未成立。历史 pilot 里，自动评审曾把开发者需求推成 `reviewed_active`。当前默认 `harness_auto=false`，settle/review 不会跑；机制仍在，一旦打开就会再注入错误对象。例子与 `useful=true` 记录见[案例-开发者提案误入overlay.md](案例-开发者提案误入overlay.md)。
- **本批只做**：配合 plan-06 按接收者/可执行性修路由，不按 `kind` 一刀切。`b4131be13c4750c8` 的新决策进 developer queue，零 overlay、零 `inherit_prompt`；不把盘上已有 overlay 行写成已经消失。`28c247a285f8a1e3` 只通过新增路由门，不自动激活。并让人决定是关闭 `harness_review`，还是把 review 仅作为人看的建议。未决定前不自动改变配置，也不把 `useful=true` 或模型等级当授权。
- **渲染措辞**：`overlay_text` 的固定标题写成两个模型各自提出，与单来源 `reviewed_active` 不符。修正该标题是本次提示面变更的一部分，与指纹扩面、b413 从 `experience.md` 删除合并为一次性重签，不做 old→new hash 别名。
- **审核指纹**：grant 指纹增加的是决定送达文本的代码与模板，包括 `overlay_text` 函数和固定标题。每场 overlay 行、peer_tips 正文不进 grant 指纹，进每场 `prompt_surface_digest` 与 plan-08 的 delivery receipt。`experience.md` / `device.md` / 技能文档继续留在指纹内。
- **本批不做**：分层指标、Brier、自动轮换、语义 quorum、更多 debate、参考集或常驻批准者。不把动态内容哈希进 grant。ForgeKernel 的架构/编码双 Agent 不能当作 fresh reviewer、自动激活或模型分级的依据。完整旧设计见[归档](archive/plan-05-独立评审与反共识.md)。

## 当前代码核对（2026-11-02 事实，2026-11-03 复核，供审阅，未实施）

- quorum 的实现要求两个委员会、不同模型产生同一 `proposal_key`。`reviewed_active` 是另一条路：`taic/harness.py:939-982` 允许单来源 tip，并在 `:982` 硬写 `"shadow": False`。quorum 从未成立，所以现有 overlay 行理论上都是单来源。
- `overlay_text`（`taic/harness.py:1129`）在 `:1144-1145` 写「以下条目由两个模型在本硬件上各自提出并经人授予的信任放行」。这句与单来源门不一致。本批要改的是这句固定标题，不是把每场 overlay 正文写进 grant 指纹。
- 当前 grant 指纹的组成在 `taic/trust.py:77-95`：若干 `prompts.*`、`kernel_agent.render_contract`、`kernel_agent._cap_task`、`harness.write_template`、若干常量、`taic-tools/SKILL.md`，以及 `:128-147` 的 `experience.md` / `device.md` / 后端技能。`overlay_text`、`peer_tips_for`、`stage_guidance` 的策略都不在里面。`build_review_prompt` 也不在这张表里；本次扩面决定没有把它加进去，不要顺手加。
- `kernel_agent.py` 不调用 `prompts.system_prompt`。`taic/orchestrator.py:679` 仍计算它，`:1111` 写入 `system_prompt.txt`。它是否仍被 hmz 发送，本轮无法确认。已不发送的 chat 面先逐个核对再从指纹剔除，本次不删。
- `--explain` 不存在。`inherit_prompt` 仍是计划，代码零命中。单来源 review 若保留，只能作为人工建议，不能自动激活 overlay。这句只管 overlay。plan-06 的 `inherit_prompt` 是另一处置，同意不构成独立证据。

## 为什么先停自动通道

quorum 目前要求两个委员会、不同模型产生同一 `proposal_key`，而 key 由归一化后的整句 action 计算。
 轻微的措辞差异已经会产生不同 key；second 至今没有结构化提案。继续优化这条路径的
“证据独立性”没有现实流量。单来源 review 若保留，也只能作为人工建议，不能自动激活 overlay。这句只管 overlay 与 `reviewed_active`。[plan-06 §3.3](plan-06-注入面与运行时安全.md) 的 `inherit_prompt` 是另一处置：宿主字段通过后，后续任务 LLM 的同意可以让当前任务提示引用该建议，并且不逐条人审。该同意不激活 overlay，也不构成独立证据。

人工 adopted 主要记录 `valid_measured` 或批量恢复，不等于建议被采纳，也不支持用接近 100% 的
reasonable/reviewed 证明评审有效。没有人工负例时，不计算误报率或模型排名。

## 模型分级与常驻审核（2026-10-26 rlar 讨论结论）

### 分级只能是受限委派，不能是智力特权

不按厂商声望、一次榜单、review confidence 或自信语气给模型分级。若未来确有证据，只能按 `role + provider/model/version/effort + target/task + scope + expiry` 分别评估写核、建议分类和评审能力；写核成绩不能迁移成评审能力。当前 `trust.py` 的 `SCOPES` 没有 reviewer role，grant 也没有部署 version 维度，新增 role scope 或身份维度必须由 plan-03 具名扩权，不能把 `overlay:tip` 改名冒充分类授权。

高能力模型即使差异确实存在，也只能影响**非权限性事项**：developer queue 的排序标注、已批准队列内部的优先级、评审侧重点。不得改变 recipient、workdir 可执行性、trust/scope、围栏、评分材料、容差/预算/协议、是否新启动 code-agent、上下文可见面或 P2 执行。`tcenv.cli ab` 无论由谁提出，都因需要改仓库而进 developer queue；模型强弱不改变接收者。

“聪明模型更少提蠢建议、不会被蠢建议带偏”只能作为待测假设。要分别用正常建议、诱饵建议和对抗建议测 recipient/actionability、证据纪律与抗暗示，不能用写核成绩或表达自信替代。

队友提示继承使用的工作假设以 [plan-06 §3.3](plan-06-注入面与运行时安全.md) 的四条为准，与上一句同属待测假设，不是已测安全保证。该节同时列出仍可能发生的失败和计划 fixture。同源（同模型或同上下文）同意记入复核记录，仍可能是相关错误；它可以被当前任务提示采纳，但不是独立认证，也不能改写宿主字段。常驻审核、单一批准者和自动 overlay 通道不因 `inherit_prompt` 而放开。`b4131be13c4750c8` 即使获得同意，也只进 developer queue。

### 常驻审核只能是 shadow 分类器/顾问

常驻上下文可能带来分类一致性、被推翻清单和 developer queue 后续追踪，但不等于更聪明或更独立：它也会带来陈旧先例、记忆注入、anchor/从众、持续 token/延迟成本和单点偏差。只要 review 还有激活能力，就不得引入常驻记忆；不得让常驻模型成为单一批准者或自动 overlay 通道。

若未来试验常驻分类，必须先盲分类再查先例。先例库只保存**经人工批准、版本化的决策记录**，每条带原文 digest、核验状态、scope、provenance、expiry；未验证 FEEDBACK/旧经验/模型记忆永远只是 untrusted candidate，不因被记住就成为事实。过期、冲突、疑似注入、首见类型、b413 类边界、reviewer 与 proposer 同 model/同上下文，必须 fresh 或人审。`overlay_text` 的函数与固定标题按已拍板决定进入 grant 指纹；每场 overlay 行不进。`build_review_prompt` 不在本次扩面表内，继续留在指纹外，不要顺手加。两者现在都不在 `current_prompt_sha1` 里。把静态模板纳进指纹是 `control_plane_change`，按 plan-03 具名审批，不能宣称是自动收窄，也不做 hash 别名。

### 当前不做自动审核，先离线验证

当前 12 条提案、5 次 review、人工 rejected/refuted 为 0，样本不足且没有模型因子，不支持任何模型分级或常驻审核上线。本批只离线做：路由规范 + `b4131be13c4750c8`/`28c247a285f8a1e3` 双向 fixture + 既有数据 replay。由具名人做正负标注，以及比较确定性路由、fresh reviewer、常驻分类器查先例前/后，都在另获授权之后；未获授权不建设参考集。届时对照须隐藏 proposer 身份、隔离被测条目及推翻结论、控制顺序 anchor，不能先给常驻臂看答案。

另获授权后的比较，评价指标是 recipient/actionability 准确率、反事实 false-overlay、漏掉有效 tip、人工推翻率、人工分钟和每有效 P2 成本，附小样本区间；实际 shadow 注入恒为 0，不能把“实际 0”当收益。重启自动 review 仍须满足下节的四个条件，模型分级与常驻审核另行授权，不由本段开通。

讨论记录：`/Users/kuafu/.pi/agent/hmz-flow/2026-10-26T15-07-20-741Z-rlar-5fc78b56.log`（actor=deepseek-flash，reviewer=kimi-code/k3.1-pro）。

## 新增文献对评审边界的补充（2026-10-27）

这些文献只帮助解释为什么当前仍需 fresh/人审，**不增加付费 reviewer、不建参考集、不重启自动激活**。

- [Humanize](papers/2610.08900_humanize-judgement-engineering.pdf)：可借鉴机械路由与语义评审分离、fresh reviewer 和公开 postmortem 语料；但其 reviewer 拥有完成判定权，不能据此证明 advisory 或自动激活有效，也不支持常驻批准者。
- [Humanize Olympiad Agents](../paper/2610.13632_humanize-olympiad-agents-a-multi-session-workflow-for-olympiad.pdf)：语义 review 与客观 verifier 是两个独立门，候选在评审期只读；领域是奥赛/Lean，不能当 kernel 效果证据。
- [ModularRSI](papers/2609.14857_modularrsi.pdf)：模块限改与三道验证门失败后的按 diff 回滚，只作为人工隔离与回滚参考；论文没有 `quarantine` 机制，也不是继续辩论或自动激活的依据。
- [Meta-Harness](../paper/2603.28052_meta-harness-end-to-end-optimization-of-model-harnesses.pdf)：同集搜索与评审可能互相放大过拟合，是自动 review 的警示，不是新增模型配额的依据。
- [ForgeKernel](papers/59558_ForgeKernel_Self_Evolvin.pdf)（ICLR 2027 双盲投稿，匿名作者，无官方代码/artifact；**论文口径 / 未在本仓库复现**）：架构 Agent 与编码 Agent 是同一场优化里的写核分工。前者只反推参考行为并拆里程碑，不直接改代码；后者逐里程碑实现。这是分工，不是独立评审，不能当作 fresh reviewer、自动激活、模型分级或第三委员会的依据。论文没有给出二者独立性的统计证据；本仓库也没有文中的 Claude Code 执行面。性能数字至少有三套分母，都要分开写：摘要是 22 个专家库单元中 18 个追平/超越；正文另有 29 个 cell 中 25 个；7 个生成系统核全部追平/超越，几何均值 +39.4%、最高 +170.6%。三套都不能混写，不能搬到 Metal，也不能当成评审质量。数字以 [导读 §2J](papers/导读.md) 为准。

因此，新文献加强的是“为什么不能把模型自评或双 Agent 分工当独立证据”的判断，而不是放松本节重启条件。`harness_auto=false`、`harness_review=true` 是否关闭或降为 advisory，仍由人决定。

## 重启条件

同时满足以下条件再评估自动 review：开发者路由已修正、确有写核工作目录可直接使用的 tip、提案量超过人工可处理水平，并且已有覆盖正负例的人工标注。失败或样本不足先读计数，不调用更多模型。`logs/pytest/triage/triage_summary.md` 仍是测试失败的第一入口。
