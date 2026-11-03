# ForgeKernel 调研：架构先行、测量认证与决策记忆

状态：2026-10-29 只读调研备忘。**不改变 plan-01～plan-09 的授权、优先级或配置**，也不是论文复现。
本文没有运行模型、GPU、NPU 或 `taic`，没有修改代码。所有“文章说”与“本仓库已有”分开写。

## 0. 一句话结论

ForgeKernel 值得借鉴的不是“再加一个更聪明的常驻 Agent”，而是三件事：

1. **把架构决策与代码调优分开记录**，避免搜索很早就锁死在一个顶层设计里；
2. **用同一任务内的配对测量与机制证据判定收益**，不把模型自报或单次最快值当结论；
3. **经验必须经过测量认证、带 scope/provenance，并可审计地复用**，而不是自由文本直接进下一场提示。

本仓库的测量与确认链已经覆盖第 2 点的大部分；第 1、3 点只有部分基础，且当前数据量很小。
因此这篇材料**不要求现在实现双 Agent、知识库或自动晋升**。

## 1. 来源与可核实性

| 项目 | 当前状态 |
| --- | --- |
| 公众号文章 | 本地：[打平 CAKE、领先 KDA！ForgeKernel打破 Agent 优化天花板，覆盖 H100、B200 与昇腾 910C.pdf](../一个公众号的论文解读/打平%20CAKE、领先%20KDA！ForgeKernel打破%20Agent%20优化天花板，覆盖%20H100、B200%20与昇腾%20910C.pdf)，22 页，2026-10-28 |
| 论文 | 用户已手动放入：[59558_ForgeKernel_Self_Evolvin.pdf](papers/59558_ForgeKernel_Self_Evolvin.pdf)，18 页，ICLR 2027 双盲投稿（Anonymous authors，Paper under double-blind review），PDF 创建于 2026-09-26。OpenReview 网页 `https://openreview.net/forum?id=EtV3f5BPD4` 仍被浏览器挑战拦截，但本地 PDF 可读。 |
| 官方代码 | 论文正文、参考文献和 PDF 链接中没有 ForgeKernel 自己的代码/artifact 地址；GitHub 搜索也没有找到与面壁智能/ModelBest 对应的官方实现。同名 `DaTouJun/ForgeKernel` 是 LLM 推理 megakernel，`SergeiMe-dotcom/ForgeKernel` 是操作系统内核，均不是本文系统；`forgekernel` PyPI 包是 CAD 几何库 |
| 官方组织 | `OpenBMB` 组织可查 `ForgeTrain`、`ForgeStencil`，未查到这个 ForgeKernel；`ModelBest` 组织搜索无结果 |
| 结论 | 现在有**论文原文**可核对，但它仍是**匿名双盲投稿 + 无官方代码/复现 artifact**。文中的数字只能当作者主张，不能当本仓库已验证结果，也不能作为新的效果证据 |

公众号文章本身有一处内部张力：它说 ForgeKernel “不设置固定流程”，但随后描述的是明确的“架构 Agent + 编码 Agent + 嵌套双环 + 三类门控”固定流程。应以后者（可描述的结构）为准，不把“无固定流程”当机制事实。

## 2. 文章主张的结构化摘要

| 机制 | 文章的主张 | 需要保留的怀疑 |
| --- | --- | --- |
| 架构/编码分工 | 架构 Agent 反推参考核的顶层设计，编码 Agent 逐里程碑实现；架构 Agent 不直接改代码 | 两个 Agent 不等于独立评审；文章没有给出二者独立性的统计证据 |
| 黑盒参考 | 默认通过 NCU/时间线探针反推分块、流水线、线程束分工，并标记无法判断的点。参考源码不跨边界。参考实现可以有可读源码（CUDA/Triton/TileLang/ThunderKittens），也可以是密封二进制 | 依赖 NCU/msprof；这不是“所有参考都无源码”。公众号写成“仅凭二进制”偏强；Metal 没有同等级计数器，不能直接搬 |
| 里程碑课程 | 先架构决策（M0），后参数调优（M1…Mk）；每步有正确性、带来源性能、机制验证三类门 | “机制验证”要具体到可用探针；文章没有给出每类机制的完整反例集 |
| 外环重规划 | 连续 3 次无新最优、课程走完仍未达标、或门控僵局时，重开架构分析 | 触发阈值是工程选择，不是论文证明的最优值 |
| 知识配方 | 每条含机制、触发症状、最小例子、坑点；按 GPU 架构分类；只引用公开资料/探针，不含参考核专属命名；79 次运行积累 71 条、9 类、553 个（条目, 运行）配对 | 这是作者自己的账本；没有公开配方库或独立复现，不能把 553 当因果效果 |
| 测量认证 | 任何配方入库前在目标架构跑探针；日志记录“本轮发明/复用/落地了多少” | 认证仍由系统自身执行；没有第三方审计 |
| 自进化 | 把当前最优核封存为下一代参考，目标增益 5%，源码不跨代、知识跨代 | “平台期”是估计值；连续晋升会放大选择偏差和成本 |
| 主要数字 | 27 个算子、29 个形状；22 个专家库单元中 18 个追平/超越（另 2 个在 1% 内），几何均值 +1.6%，最高 +7.6%；7 个生成系统核全部追平/超越，几何均值 +39.4%，最高 +170.6%；昇腾 8 个四级算子几何均值 +33.1% | 正文另有“29 个 cell 中 25 个追平/超越”的口径，和“22 个专家库单元”不是同一个分母；每组取“最优优化战役”；不同硬件/基线/成本单位；CAKE 只快 0.4%；不能当统一效应量 |
| 成本 | 简单 DSA 分数重计算 3.1 活跃小时；循环 KDA 前向 97.8 活跃小时、34 GPU 小时；2D 卷积前向 19 次架构介入 | 文章自己承认 CAKE 报 token、ForgeKernel 报 GPU/Agent 时间，不能直接换算 |
| 局限 | 必须有强参考核；平台期只是估计；只覆盖四类算子、NVIDIA 两代和一款昇腾 | 与“通用自进化”标题相比，覆盖范围明显更窄 |

这些数字可以用于提问和设计实验，不能写成本仓库的测量结果。尤其“每组取最优战役”意味着**选择效应**没有被单独报告。

### 2.1 原文核对（2026-10-29）

本地 PDF 的摘要、正文和附录给出的关键口径：

- 实验平台是 H100 80 GB（SXM，132 SM）、B200（148 SM）和 Ascend 910C；编码 Agent 写的是 **CuTe DSL**，不是 Triton→MSL。
- 公众号文章的“仅凭二进制反推”说法偏强：论文的参考实现可以是 CUDA/Triton/TileLang/ThunderKittens，也可以只有二进制；真正规则是**参考源码不跨边界**，不是所有参考都没有源码。
- 架构 Agent 是 Claude Fable 5.1，编码 Agent 是 Claude Opus 5，都是 stock Claude Code agents；本仓库没有这两个执行面，不能把结果直接迁移到 Metal。
- 正确性门是每个算子的独立 FP32 judge，且拒绝任何调用/转发到参考核的候选；最终性能声明取最多 3 次同任务读数的中位数。
- 参考消融：无参考时三个算子全部停滞；只给密封二进制时几何均值约 +1.5%，可读源码约 +2.5%；说明“参考的行为”比“参考的源码”更关键。
- 参考强度会传播：同一个 DSA fp8 indexer，对强参考 DeepGEMM 2.6.1 与弱参考 cuDNN-frontend 1.28 得到的内核相差 18%；对弱参考的加速比不能当成对强实现的接近程度。
- 成本：DSA score recompute 3.1 活跃小时、循环 KDA prefill 97.8 活跃小时 / 34 GPU 小时、2D 卷积前向 19 次架构介入；wall 时间还含共享集群排队。
- 可复现性声明明确说有两类结果不能只靠论文复现：Ascend 分数只能在 CANN-Bench 托管服务复现；campaign 用前沿模型采样，重跑会得到不同轨迹，应复现的是测量而非轨迹。

这些是论文原文，不是本仓库复现。尤其“最好战役”与“最多 3 次同任务读数”让跨系统比较仍有选择效应。

## 3. 与本仓库逐项对照

| ForgeKernel 的点 | 本仓库现状（只读核对） | 判断 |
| --- | --- | --- |
| 同一任务内配对计时 | `tcenv/bench.py` 已按 A/B 交错测量候选与基线，报中位数、CV 和逐次样本；确认阶段另有 `confirm_batches=3` | **已有，不需要新增** |
| 噪声门与确认 | `taic/policy.py` 的 `required_gain` 同时用 `min_improvement` 和 `noise_z`；默认 `min_improvement=1.005`、`noise_z=2.0`；搜索最优与确认胜者分开 | **已有，且比文章的 1%–5% 口径更明确** |
| roofline 停机 | `search.stop_at_roofline=True`；种子进入 roofline 噪声门且无基线更快时停止 | **已有**；它是启发式，不是物理最优证明 |
| profiler 证据 | `profile_best` 在结尾追赢家；`profile_poor` 在种子慢于基线时追 Instruments；`hw_feedback` 是静态硬件特征，默认关 | **部分已有**；Metal 没有 NCU 计数器，不能照搬“机制计数器门” |
| 结构化决策记录 | `taic/plan_ledger.py` 已能把尝试分为 `validated_win / validated_tie / falsified / untested_impl_fail`，且明确“编译失败不算方向被证伪” | **有模块**；但本次只读检索没有发现 `format_ledger_for_planner` 被当前 `kernel_once` 的 TASK 路径调用，当前 brief 仍主要用 `history[-5:]`。不能把它当成已接入的架构记忆 |
| 平台期重开架构 | `patience` 只负责停止；`diversity_archive` 在配置/观察层存在，本次检索未发现它实际调度“换一颗特征不同的核”的分支 | **不能当现有能力**；若要试，应先做离线实验，不应直接加第二个常驻 Agent |
| 经验 staging | `stage_guidance` 把 `experience.md` 整文件拷进会话；文件头写了 `reported` 不进提示词，但本次阅读没看到按 `reported`/stale/retired 过滤的代码 | **已知缺口**，正是 plan-07 的当前任务；也是 b413 条目的风险来源 |
| 知识配方/provenance | 经验条目已有观察、来源、本机实测、边界、反例，但没有统一的测量证书、scope、expiry、复用审计 | **方向一致，规模不够**；当前 12 条提案、0 条 `trusted_record`，不应先建知识库 |
| 双 Agent 独立评审 | 当前是一轮一场新会话、first/second 两个委员会；没有常驻架构 Agent，也没有把“架构 Agent”当独立 reviewer | **不采纳“双 Agent 即独立评审”**；这会把架构建议和评审混成一个角色 |
| 黑盒参考、源码隔离 | 本仓库的任务规格就是 Triton 源码；编码 Agent 需要语义规格，不能把源码藏掉 | **不适用**；黑盒二进制是 IP/反推场景，不是当前 lowering 场景 |
| 跨硬件（NCU→msprof） | 本仓库目标固定为 Apple Metal；没有 NPU 执行面 | **不现在抽象**；跨后端属于 dsl/执行运行时议题，不是本 harness 的当批改动 |
| 成本可比性 | `result.json` 有轮数、花费、GPU 时间；文章把 token 与 GPU/Agent 小时混在一起比较 | **提醒**：plan-08 的回执只证明“送达”，不能把成本或事件数当效果 |

## 4. 对 harness 的具体启发

### 4.1 先做已有的，不重复造

ForgeKernel 的“配对计时、噪声门、确认、roofline”在本仓库已经有对应实现。**不要因为这篇文章再加一套 A/B 计时代码**；先核对现有 `tcenv/bench.py`、`taic/policy.py` 和确认路径是否被正确使用。文章最多是对现有方向的旁证，不是新需求。

### 4.2 最值得补的是“决策记录”，不是“第二个 Agent”

本仓库已有 `plan_ledger.py` 的分类逻辑，但当前写核会话主要看到最近几条 `history`，看不到结构化的“哪些方向已验证胜利、哪些只是实现失败、哪些还未测”。缺少这段文本可能让下一场重复试错，也可能把编译失败误当成方向失败；两种说法都尚待验证，不是已观察到的搜索结果。

可考虑的最低成本实验（**仅离线、需另批**；未授权。断言与 [plan-10](plan-10-决策账本与平台期重规划.md) 相同，只查文本和分类）：

- 用现有的 `PlanNode` 生成一段短 ledger，只检查它是否出现在 fake TASK 的“历史”段；对照臂只有当前 `history[-5:]`；
- 同一批 fake session/预置报告上，只断言分类和提示文本：`falsified` 写明不要重复，`untested_impl_fail` 写明可重试。不调用模型。据此否决“比较下一场是否还会重复已被证伪的方向”：只渲染 fake TASK 测不到这个行为；
- 不新增模型角色，不改评分、不碰 GPU，不把 ledger 当事实来源。

这与 ForgeKernel 的“里程碑课程”只共享一个原则：**让下一轮看到上一轮真正确立了什麼**；不需要复制它的 NCU 反推或自进化。

### 4.3 经验过滤应借“测量认证”，不借“自进化”

ForgeKernel 的配方结构（机制、症状、最小例子、坑点）比当前自由文本更可审计，但它的准入仍由系统自身完成。对本仓库更有用的映射是：

- `reported`/stale/retired 不进入 staging/TASK；
- 每条经验保留来源、scope、验证状态、边界与 expiry；
- b413 这类 developer tool request 不进写核经验；
- 不自动把一次成功写成通用配方，不做跨 target 迁移。

这正是 plan-07 已收窄的方向。**现在不要建完整知识库**；条目数量和 trust 记录都不够。

### 4.4 机制证据可以做，但不能冒充 NCU

论文强调“不是只看变快，还要看机制是否真的落地”。本仓库可用的替代证据是有限的：

- launch/grid/threads、静态 threadgroup 内存、源码声明的 simdgroup 槽位；
- `tcenv profile` 的 spill/code size/GPU idle；
- dispatch 计数或 no-op/回退探针（属于 plan-01 的攻击语料，不是常规收益门）。

这些证据可以写进候选记录或 NOTES，作为**非阻塞**解释；不应在没有 Metal 计数器的情况下声称“机制已验证”。也不应把机制证据替代 fresh-input 或确认门。

### 4.5 成本与选择效应要分开记

论文承认 CAKE 的 token 成本与 ForgeKernel 的 GPU/Agent 时间不可直接换算，但正文仍用“最优战役”做跨系统比较。对本仓库的提醒是：

- 报告搜索最优时，同时保留轮数、会话数、GPU 秒、模型花费；
- 报告确认胜者时，不把搜索阶段的 best-of-N 当确认值；
- plan-08 的 delivery receipt 只回答“有没有送到”，不回答“有没有效果”；
- 不要用“active 事件数”或“复用条数”代替因果效果。
- 参考强度会传播：论文对同一 DSA fp8 indexer 用强参考与弱参考得到的内核相差 18%。我们的加速比分母必须用当前最强 vendor baseline 与 roofline；打赢弱基线不能当成接近强实现。

## 5. 明确的“不采纳”

| 不采纳 | 原因 |
| --- | --- |
| 增加常驻架构 Agent / 第三委员会 | 当前一轮一场会话的预算和信任边界已经够复杂；论文没有给出独立评审证据，双 Agent 也不等于独立 |
| 自动知识库、自动配方晋升 | 当前 12 条提案、0 条 `trusted_record`，样本不足；自动晋升会把模型自述变成事实 |
| 参考核二进制反推 / NCU/msprof 门 | 目标平台是 Metal，没有同等级计数器；本仓库的任务规格是 Triton 源码，隐藏源码会丢掉语义 |
| 自进化“晋升最优为参考”直接上线 | 会放大 best-of-N 选择偏差和成本；我们已有确认门，应先证明它能被正确使用 |
| 把论文的 +39.4% / +170.6% 当效应量 | 论文有原文，但官方代码/复现 artifact 未找到；每组取最优战役、不同硬件/基线/成本单位 |
| 把双 Agent 写进 plan-05 作为“独立评审”依据 | 架构 Agent 与编码 Agent 是分工，不是对抗性独立评审；plan-05 的 fresh/人审边界不能因此放松 |

## 6. 可下载/可核对的材料

| 材料 | 位置 | 状态 |
| --- | --- | --- |
| ForgeKernel 公众号 PDF | `docs/一个公众号的论文解读/打平 CAKE、领先 KDA！ForgeKernel打破 Agent 优化天花板，覆盖 H100、B200 与昇腾 910C.pdf` | 本地已有；二手文章 |
| ForgeKernel 论文 | [papers/59558_ForgeKernel_Self_Evolvin.pdf](papers/59558_ForgeKernel_Self_Evolvin.pdf) | 本地已有，18 页，ICLR 2027 双盲投稿；SHA-256 `4f0251783647cb4be97ba57e921f4e6bef2bcae02f82fab6bd8271cc83bd0872`；官方代码未找到 |
| ForgeKernel 中译 | [papers/59558_ForgeKernel_Self_Evolvin.zh-CN.md](papers/59558_ForgeKernel_Self_Evolvin.zh-CN.md) · [papers/59558_ForgeKernel_Self_Evolvin.zh.pdf](papers/59558_ForgeKernel_Self_Evolvin.zh.pdf) | AI 辅助全文译文，不是作者官方译本；数字以英文 PDF 为准 |
| ForgeKernel 官方代码 | — | 未找到；同名仓库均非本文系统 |
| K-Search 论文 | `docs/paper/2602.19128_k-search-llm-kernel-generation-via-co-evolving-intrinsic-world-model.pdf` | 已下载，15 页；SHA-256 `bb7c52042664173039bf8c453f105c19ae3a1ce6116d7307bfaed2e806926cef` |
| K-Search 代码 | `/Users/kuafu/CLionProjects/dsl_for_llm/K-Search` | 源码快照；上游 commit `53c8fab9a5e8fab2c86610d24fbec5067f90e115`，Apache-2.0，见目录内 `SOURCE.md` |
| CAKE | `docs/可信harness/papers/2608.12629_cake-compiler-agent-codesign.pdf` | 已有；文章的直接对照之一 |
| KDA | `/Users/kuafu/CLionProjects/kda`（及 worktree） | 已有；文章的直接对照之一 |
| Kernel Forge | `docs/paper/2607.24762_kernel-forge-an-agent-harness-for-llm-based-generation.pdf` | 已有；**不同系统**，不要与 ForgeKernel 混名 |

K-Search 之所以作为补充材料，是因为它有公开论文和代码，且同样把“高层规划”与“低层实现”分开；它不是 ForgeKernel 的官方实现，不能拿它的结果替代 ForgeKernel 的未公开代码/结果。它的 `WorldModelSelectionPolicy` 用难度门和确定性效用函数选下一个动作，`refine` 只在 `passed` 且有性能字段时更新，这些是**代码可核对**的做法，但仍需要在本仓库离线复现后才谈采纳。

## 7. 建议的下一步（按现有优先级）

1. **不改 plan-01/02**：ForgeKernel 的配对计时、噪声门、确认已被本仓库覆盖；当前仍先做 plan-01 的 recheck 核对与 plan-02 的两个 pair bug。
2. **plan-07 继续只做 staging 过滤与人工去重**：把 `reported`/stale/retired、b413 类 developer request 挡在写核经验之外；配方 schema 只作为以后评审字段，不现在建库。
3. **plan-08 继续只做 delivery receipt 与离线 S0**：token、GPU 小时、活跃小时、wall 小时只作以后的口径旁注。本批不把这些单位写进最小回执，也不把送达或复用计数写成效果。
4. **决策账本 / 平台期**：plan-02/07/08 的本批范围容纳不下。2026-10-29 另写 [plan-10](plan-10-决策账本与平台期重规划.md)，状态是研究候选、未授权实施。另获批准前不接入 TASK，不跑 GPU，不新增常驻 Agent。
5. **不做**：自动知识库、自动配方晋升、NCU/msprof 门、跨平台抽象、把文章数字当本仓库结果。

## 8. 对 plan 的落点

2026-10-29 只加了文献旁注，并新建未授权的 plan-10。plan-01～plan-09 的本批授权和优先级没有改。

| plan | 是否改了本批 | 说明 |
| --- | --- | --- |
| plan-01 | 否 | 配对计时/机制证据不替代 fresh-input；本文只作旁证 |
| plan-02 | 否 | 仍只修 pair 身份与 `pair_count`，不新增实验 |
| plan-03 | 否 | 不因文章放宽 trust/scope；双 Agent 不是授权依据 |
| plan-05 | 只加旁注 | 双 Agent 是分工，不是独立 review；不因此建参考集或重启自动激活 |
| plan-06 | 否 | 黑盒参考是 IP/反推问题，不是本仓库注入面 |
| plan-07 | 只加旁注 | 测量认证/配方 schema 只作以后字段，当前仍只做过滤与去重 |
| plan-08 | 只加旁注 | token、GPU 小时、活跃小时、wall 小时不可直接换算；回执不证明效果 |
| plan-09 | 此前已做 | K-Search 论文与代码已登记在 `docs/paper/README.md` 的“相关对照”区，不计入原 56 篇 |
| plan-10 | 新建研究候选 | 未授权实施；不占 README 第 3 节的 8 个工作包 |

本文仍是调研记录，不改上述优先级。
