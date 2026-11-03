# plan-07：经验知识生命周期

状态：**staging 过滤已实现（2026-11-03），人工去重完成盘点；完整生命周期归档。** 不调用模型或 GPU。实施与验收见[经验盘点-2026-11-03.md](经验盘点-2026-11-03.md)。

## 实施记录（2026-11-03）

- 新增 `taic/experience_prompt.py`：`filter_experience_for_prompt` 只保留前言与 `Confidence: verified` 小节，排除 `reported`/`stale`/`retired`/未标注小节，并删除裸 `[harness …]` bullet。
- `stage_guidance`（`taic/kernel_agent.py`）对 `experience.md` 改为过滤后写入会话目录；`SkillIndex`（`taic/tools.py`）对 `experience` 先过滤再建 outline/lookup。运行级快照仍保存原始文件。
- 新增 `tests/shared/test_experience_staging.py`。本机离线结果：新增+wiki 11 passed、`test_kernel_agent.py` 32 passed、`test_trust.py`+`test_taic_llm.py` 48 passed。
- **未做**：b413 源条目未删（留给一次性重签那次）；grant 指纹未因本项过滤改变，过滤策略纳入指纹仍属 plan-03 的 `control_plane_change`；device/experience 经盘点维持现状，无需合并。`inherit_prompt` 后来由 plan-06 实现为 TASK 内队友复核，不写经验库。

## 当前决定

- **输入**：`targets/<id>/experience.md`、device/skills、staging 与 TASK 渲染结果。
- **本批只做**：让经验库中的 `reported`/stale/retired 不进入 staging、outline 或 TASK；人工合并 device 与 experience 的近似重复，保留来源与较窄边界。这不禁止 plan-06 的 `inherit_prompt` 在当前任务提示里引用一条状态仍为 `reported` 的当场建议。那条提示不得取自经验库里待排除的 `reported`/`stale`/`retired` 条目，也不得写回经验库。上述排除尚未在 `stage_guidance` 生效。
- **本批不做**：不透明 `entry_id`、revision/CAS、artifact/composition、跨 target 迁移、单 key 撤回或自动写入。ForgeKernel 的测量认证配方只作下表的以后评审字段，不建知识库，也不自动写入。完整设计见[归档](archive/plan-07-经验知识生命周期.md)。
- **已知误入条目**：`experience.md:99` 的 `[harness b4131…]` 是 developer tool request，不是 kernel 可用经验。人审已决定**删除**这条 bullet，事实留在[案例](案例-开发者提案误入overlay.md)与开发者队列，不做「保留 + 加注」。删除会改变 `experience.md`，从而改变 grant 指纹；与指纹扩面、overlay 标题措辞修正合并为一次性重签，不做 hash 别名。本文不直接改经验库文件。
- **队友提示与经验库分开**：当前任务的 `inherit_prompt` 由 [plan-06 §3.3](plan-06-注入面与运行时安全.md) 规定。本批不实现该通道，也不把它当成经验入库。同意不把 `reported` 改成 `verified`，不恢复 `stale`/`retired`，不扩权。
- **指纹**：`experience.md` / `device.md` / 技能文档继续留在 grant 指纹内。`stage_guidance` 的拷贝/过滤策略进入 grant 指纹；本场实际拷入的字节进 `prompt_surface_digest`，不进 grant。是否把知识库移出指纹仍是独立扩权，本次不碰。
- **完成产物**：一份简短盘点和 staging 过滤回归；失败先读 `logs/pytest/triage/triage_summary.md` 及 handoff。

## 当前代码核对（2026-11-02 事实，2026-11-03 复核，供审阅，未实施）

- `stage_guidance`（`taic/kernel_agent.py:862-881`）对 `device.md`、reference skill 和 `experience.md` 做 `shutil.copyfile`，没有按条过滤。运行级冻结同样整文件写入 `agent_guidance/`（`taic/orchestrator.py:1132-1136`）。`taic/` 没有经验 staging 目录。
- `targets/apple-m4-max/experience.md` 为 8326 字节。文件头 `:6-7` 写 `reported` 不进提示词，代码没有对应实现。正文 9 节、9 条 `Confidence: verified`（`:12,23,33,43,53,63,73,83,93`）。没有 `reported` / `stale` / `retired` 实例。`:99` 是 b413 裸 bullet，没有 `Confidence`。过滤缺口目前是潜在的，不是已经把这三种状态送进了 TASK。
- grant 指纹仍包含 `experience.md` / `device.md` / 后端技能（`taic/trust.py:128-147`）。过滤策略本身还不在指纹里；按已拍板决定，策略进 grant 指纹，每场拷入内容不进。不把 `experience.md` 移出指纹。
- b413 的历史 promotion 已 commit `178de7c`，盘上 overlay 行仍在。删除的是经验库 bullet。新路由决策零 overlay、零 `inherit_prompt`，见案例 §1.1。不因曾有 review 就继续渲染这条 bullet。

## 为什么缩到过滤与去重

先区分两类建议，避免把“给队友的提示”和“给系统的需求”混成一条经验：

| 建议对象 | 典型内容 | 当前任务内如何继承 | 长期/跨任务采纳 |
| --- | --- | --- | --- |
| 后来的队友 LLM | 当前 scope 内、既有权限下可执行的避坑、路径、探针、测量方法、小知识、边界。证据状态照实保留 | 宿主字段先通过，后续队友同意后自动 `inherit_prompt`。不逐条人审。规则、工作假设与计划 fixture 见 [plan-06 §3.3](plan-06-注入面与运行时安全.md) | 同意只停留在当前任务提示。写入经验库或变成系统行为仍须人工审核验证 |
| 系统/开发者 | 新工具、改 harness/权限/评分、路由、报告字段、跨任务经验库结构 | 不进入 overlay，也不 `inherit_prompt`。后续 LLM 的同意不实现、不批准 | 进开发者队列，人审与必要的离线验证后决定 |

**给队友的建议必须在既有权限内、当场能做、当前任务用得上。** “用本场允许的探针看这个形状”“这个形状会触发已知 tail bug”可以在宿主字段通过后被当前任务继承。“请给 `tcenv` 加子命令”以及任何要改 harness、权限或评分的请求交给开发者。能执行的小命令或路径若会读凭据、读错 target 的材料或改评分输入，同样不得继承。可执行不等于可以扩权。

当前 `experience.md` 是 8326 字节、9 节、9 条 verified，外加 `:99` 的 b413 裸 bullet。`reported` / `stale` / `retired` 没有实例，主要仍在文件头说明中。过滤仍未实现。
过滤规则便宜且能防未来误注入；给少量条目建立 ID/revision/CAS  主要是在重复 Git 历史，不能解决当前主要问题。

真实重复更接近同一设备事实被多次记载，以及 `device.md` 与 `experience.md` 重复。人审入库时去重即可，
不需要先建知识图谱或跨 target 迁移框架。

## 信任指纹是前置决策

`current_prompt_sha1` 当前包含 `experience.md`、`device.md`、后端 skills 与 `taic-tools/SKILL.md`。
因此一次经验晋升、删除 b413 bullet 或 ID 迁移都可能让现有 grant 变成 stale prompt。删除 b413 不把文件移出指纹；它与指纹扩面、渲染措辞修正合并为一次性重签。是否把知识库移出提示指纹属于扩权，须由 plan-03 具名决定，不能在 plan-07 里顺手改变。动态的每场拷贝不进 grant 指纹。

## 新增文献给出的后续候选判据（2026-10-27）

这些只作为以后评审经验条目的参考，**本批仍只做 `reported`/stale/retired 过滤与人工去重，不新增 entry_id、revision/CAS、自动写入或跨 target 迁移**。

| 后续可讨论的判据 | 参考文献 | 当前使用边界 |
| --- | --- | --- |
| 知识升格需真机验证与人工评审；`skill_version` 变化后旧观测降权 | [CANNBot-CAKE](papers/2610.31034_cannbot-cake.pdf) | 支持 staging 记录来源、scope、版本和验证状态；不直接自动入库 |
| 移除一个优化后，语义等价、正确且实测退化超过噪声，才承认其经验价值 | [AscendOptimizer](papers/2603.23566_ascendoptimizer.pdf) | 可把“去掉它会怎样”写成后续可证伪预测；其传导式评测与代码占位不能当泛化证据 |
| 训练期在匹配验证子任务上比较有/无技能的 Δk，再过滤、排序并受逐步衰减的预算约束 | [SKILL0](papers/2604.02268_skill0.pdf) | 只借筛选/排序思路；其 RL、改权重与最终零技能推理不属于本仓库，不据此启动 staging/TASK 自动 A/B |
| 新技能在原任务重放验证后才入库，按版本保存快照 | [daVinci-kernel](../paper/2606.16497_davinci-kernel-co-evolving-skill-selection-summarization-and-utilization.pdf) | 只借“先验证再入库”，不采用其 RL 训练或自由文本直接注入方式 |
| 编辑前写预测，下一轮核对；组件、经验和决策分开记账 | [AHE](papers/2604.25850_agentic-harness-engineering.pdf) | 支持人工去重时保留 decision/provenance，不能把单条编辑的归因当成已解决 |
| 三道验证门失败后按 diff 回滚，而不是继续叠加未过门经验 | [ModularRSI](papers/2609.14857_modularrsi.pdf) | 论文没有 `quarantine` 机制；这里只作 stale/retired 的人工处置参照，不自动触发新 overlay |
| 以后评审字段可含机制、触发症状、最小例子、坑点、provenance、no-go；入库前先在目标架构做测量认证 | [ForgeKernel](papers/59558_ForgeKernel_Self_Evolvin.pdf) | **论文口径 / 未在本仓库复现。** 只作以后人工评审条目时的字段清单。当前仍只做 `reported`/stale/retired 过滤与人工去重。不建知识库，不自动写入，不自动晋升，不把作者自报的 71 条或 553 个（条目, 运行）配对当因果效果或本仓库条目数。编码 Agent 写的是 CuTe DSL，平台是 H100/B200/910C，不能当成 Metal 经验已认证 |

这些判据的共同前提是**经验不能因被记住或被 reviewer 复述就变成事实**；当前主要工作仍是阻止未验证、过期和错误接收者的条目进入 staging/TASK。`stage_guidance` 仍整文件拷贝 `experience.md`，过滤尚未实现。

**队友复核是当前任务的提示采纳，不是长期认证，也不是事实升级。** 后续 LLM 可以拒绝不适用的提示。宿主字段通过后，它的同意使该提示自动进入当前任务的后续提示，无需逐条人审。渲染保留宿主前缀、scope 和证据状态，不写成已验证指令或系统消息。经验库里待排除的 `reported`/`stale`/`retired` 条目不能借这次同意进入提示。复核记录沿用 plan-06 的轻量字段：scope、证据指针与证据状态、同源标记、task/run 有效期。这不是知识库。同意不把 `reported` 变成 `verified`，不恢复 `stale`/`retired`，不授予额外权限，也不把当前任务的采纳延长为跨任务经验或系统行为。后继队友换措辞再提交时另记一条，原状态和有效期不升级。长期采纳、经验库写入和系统行为变化仍由人审核，并在需要时做离线验证。四条工作假设及其仍可能发生的失败（误解 scope、陈旧、半真半假、不可执行、可执行但越界、注入、同源、连锁）以 plan-06 §3.3 为准，这里不另写成已证明的安全性质。

## 重启条件

条目数量明显增长、P1 确有流量，并且「是否把 `experience.md` 移出指纹」与撤回语义另批拍板之后，才重启 entry_id、单 key 撤回和迁移设计。本次已经拍板的是：文件留在 grant 指纹内，每场动态内容不进。
`reported`/stale/retired 过滤、b413 删除与人工去重不依赖这些后续机制。
