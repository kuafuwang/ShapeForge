# CAKE2 机制清单与证据分级

日期：2026-10-27

> **目的**：在制定任何改造 plan 之前，先分清 CANNBot-CAKE / CAKE2 哪些机制值得借鉴、哪些必须改造、哪些应后置、哪些不能照抄。本文是机制清单，不是实现计划，也不授权运行模型、GPU/NPU、安装脚本或真实算子实验。

## 0. 快照、来源与核查边界

### 0.1 一手来源

| 来源 | 版本与用途 |
| --- | --- |
| CANNBot-CAKE 论文 | [arXiv:2610.31034v1](https://arxiv.org/abs/2610.31034) · [本地 PDF](../可信harness/papers/2610.31034_cannbot-cake.pdf)，重点 §2–7、Appendix A |
| `cann/cannbot` | commit `2b60dd86aa90`；本轮核查时工作树干净 |
| `cann/cannbot-skills` | commit `67e021abad3e`；本轮核查时工作树干净 |
| 相关编程样例 | `cann/cannbot-dsl` commit `755cd63b9e6f`；用于理解 DSL 表达，不用于反推 CAKE2 的完整 router |

路径记号：

```text
CAKE2 = /Users/kuafu/CLionProjects/dsl_for_llm/cannbot/
  plugins-community/collaborative-agent-kernel-evolution/
A = agents/
S = skills/
H = hooks/
```

### 0.2 核查方式与不能推出的结论

- 已读论文方法、路由、Evolution、CAKE Book、评测与局限，并只读检查上述源码。
- 对 `item-response`、`skill_version`、`CAKE Book`、`abstain` 等做定向检索，当前树未定位到论文所述统一实现。
- **没有运行 CAKE2、测试、生成、编译或真实设备评测。**
- 代码或测试文件存在，只证明其存在；不证明执行通过、论文结果复现或生产安全。
- 当前 checkout 未必等于论文实验版本。其他仓库、私有服务、分支或后续版本可能另有实现，缺失项标 **unknown**，不写成“论文没有”。

## 1. 分级与决策规则

### 1.1 证据等级（不是性能分，也不是采用优先级）

| 等级 | 含义 | 本文使用情况 |
| --- | --- | --- |
| **E0 未知** | 材料不足，无法确认机制或效果 | 仅标 unknown，不得自动放行 |
| **E1 文献主张** | 论文、SKILL、README 或项目文档描述了机制，但未核到统一确定性实现 | 多数 router、CAKE Book、Evolution 效果属于此级 |
| **E2 代码可核** | 已在指定 commit 中读到具体文件和逻辑，但未执行 | 当前多数生成、构建、评测脚本属于此级 |
| **E3 离线验证** | 确定性离线测试/fixture 已验证合法接受、攻击拒绝和诊断 | 本文没有项目达到此级，后续 P0/P1 首批目标 |
| **E4 目标验证** | 绑定 target 的编译、数值、内存/同步与真实路径检查通过 | 需要另行授权 CANN/NPU |
| **E5 确认与治理** | 绑定 artifact 的性能确认、独立复核、成本评估和人审发布完成 | 需要真实测量与发布流程，不能由论文数字代替 |

**一个机制可以是 E2 实现存在，而其论文效果仍只有 E1。** E2 不代表有效，更不代表性能提升已证实。

### 1.2 借鉴决策

| 决策 | 含义 |
| --- | --- |
| **A：优先借鉴** | 可进入 P0–P2，但仍需按我方合同和验收重写 |
| **B：改造后借鉴** | 保留机制，重做确定性 gate、权限、诊断或版本边界 |
| **C：后置/待证据** | 先积累正确候选、观测或使用场景，暂不做复杂实现 |
| **D：不借鉴** | 与最小权限、可信验收或成本目标冲突 |
| **U：待核实** | 当前证据不足，不能决定是否采用 |

## 2. 总览：先学什么，暂不学什么

| 优先方向 | 机制 ID | 原因 |
| --- | --- | --- |
| **第一批：合同、artifact、交付与验收** | M01–M07、M13–M17、M21、M34 | 直接减少语义猜测、定位失败并形成可部署算子，能先离线验收 |
| **第二批：局部诊断与 profiling** | M22–M27 | 已有正确候选和可信测量后才有价值 |
| **第三批：多 track 路由与知识复用** | M08–M12、M28–M31 | 需要足够可比较观测、知识治理和版本账，不能提前堆复杂度 |
| **目标特定扩展** | M32–M33 | 仅在实际选择 950/Regbase/SIMT 或模拟路线时研究 |
| **明确拒绝或必须替换** | M18–M20、M35 | 直接执行不可信代码、权限过宽、fail-open、把搜索最好值当确认结果 |

## 3. 机制清单

### 3.1 合同、artifact 与阶段状态

| ID | 机制与论文依据 | 当前代码事实 | 证据 | 决策与 Plan | 验收与边界 |
| --- | --- | --- | --- | --- | --- |
| **M01** | **`op_desc` 作为接口权威**：论文 §2 将身份、公式、I/O、属性、case 表作为单一事实源 | `S/op-desc-generation/SKILL.md:12-48` 要求生成名称、类别、数学描述、shape/dtype、attributes 和 test cases；生成仍由 Agent 完成 | E2 | **A / P0** | 用 schema 校验未知字段、dtype/shape/layout/输出签名；文本生成不等于合同已验证 |
| **M02** | **reference + functional form**：论文要求保留参考实现并转成无状态形式，供 oracle 和 lowering 共用 | 主流程在 `A/cake.md:75-156` 明确 reference-generation、functional-conversion | E2 | **A / P0–P1** | 参考实现只读、输入可复现；生成者不得修改 golden 或把 reference 当性能 baseline 的唯一选择 |
| **M03** | **durable artifact 链**：合同→参考→DSL/host/device→package/binding/evaluation record | `A/cake.md:75-156` 有阶段流程；`S/ascend-call-generation/scripts/gen_project.py:57-105` 有工程生成逻辑 | E2 | **A / P1–P2** | 每阶段有 artifact ID、输入输出、hash 和状态；缺 artifact 不得继续 |
| **M04** | **阶段成功才 commit，失败不 commit**：论文 §2.1 | `S/git-version-management/SKILL.md:104-166` 写清规则、path-scoped add 和 hash 绑定；主要是 skill 指令，未见统一强制层 | E2 | **B / P1** | 用确定性状态机验证失败不写成功 commit；禁止 `git add -A`，并发工作树不得混入他人修改 |
| **M05** | **commit hash 与 evaluation 绑定** | `S/git-version-management/SKILL.md:145-166` 描述 stage 9 hash 写入与再次 commit | E2 | **B / P1–P2** | hash 必须同时绑源码、合同、binary、target、协议，不能只绑一次 Git 提交 |
| **M06** | **skill trace**：记录调用顺序、状态、耗时、输出和重试 | `S/skill-trace/SKILL.md:87-170` 提供 JSON 结构和 TRACE-INIT/START/END；含概念性伪代码和文本命令 | E2 | **B / P1、P6** | 由实际 runner 自动写，不依赖 Agent 自觉；验证乱序、缺 END、重复 retry、损坏 JSON 的处理 |
| **M07** | **op_desc 机器 gate**：论文称在生成前检查 dtype 覆盖、签名和 shape metadata | 当前 `op-desc-generation` 主要是字段与流程说明，未核到统一 schema validator | E1 + 部分 E2 | **A / P0–P1** | 这是应补强的确定性层：完整/错误/未知合同 fixture 必须先于生成通过 |

### 3.2 Skill、生成与路由

| ID | 机制与论文依据 | 当前代码事实 | 证据 | 决策与 Plan | 验收与边界 |
| --- | --- | --- | --- | --- | --- |
| **M08** | **多 DSL/track 路由**：论文 §3 有 DSL-A/B/C 与 SIMT | 当前主 Agent 明示 Vector 算子；可明确识别 AscendDSL、easyasc 等路线，但论文匿名 track 无法全部映射 | E1 + 部分 E2 | **C / P5** | 先证明至少两个 track 都可交付，再做路由；不得虚构四轨对应关系 |
| **M09** | **L0 capability gate 与弃权**：不可行组合先排除，无可行 track 不猜 | 论文文本同时出现 abstain 与交完整 track menu；当前 `S/dsl-easyasc-codegen/scripts/check_device.sh:84-123` 只有特定设备分支，主 Agent 在无环境时询问用户 | E1 + 部分 E2 | **B：先借简单能力门 / P0，复杂路由后置 P5** | 未知 target、无 track、无 oracle 必须显式 unknown/弃权；当前无统一行为可照抄 |
| **M10** | **item-response 难度校正与 per-dimension ability** | 当前树定向检索未定位到该模型实现 | E1 | **C / P5** | 先积累同任务、同预算、同口径观测；数据不足时不建复杂模型 |
| **M11** | **append-only ledger + `(track, skill_version)`：旧观测降权、infra/作弊剔除** | 当前树未定位到统一能力账本 | E1 | **C / P5–P6** | 先设计最小观测 schema；版本变化、infra、作弊、自相矛盾记录必须有独立状态 |
| **M12** | **探索/利用与重测奖励**：信息代理、上界和 skill 更新后重测 | 当前未定位到统一 router | E1 | **C / P5** | 探索预算与生成预算分离；不得用上界或模型偏好替代真实评测 |
| **M13** | **track-specific lowering 与可观察状态** | `S/dsl-easyasc-codegen/scripts/run_codegen.py:121-148,208-236` 有 smoke、`codegen_status.json`、raw AscendC、post/bridge 和失败出口 | E2 | **A–B / P1–P2** | 优先复用“每阶段可检查 artifact”；Vector 之外的 Cube 覆盖不得由当前主流程推定 |

### 3.3 构建、评测、权限与状态契约

| ID | 机制与论文依据 | 当前代码事实 | 证据 | 决策与 Plan | 验收与边界 |
| --- | --- | --- | --- | --- | --- |
| **M14** | **项目构建是交付的一部分**：论文要求 build/package/install/bind，而非 kernel 片段 | `S/ascend-call-generation/scripts/gen_project.py:57-105,173-225` 有 msopgen 项目生成；`S/dsl-easyasc-post/scripts/transform.py:321-390` 有后处理 | E2 | **A–B / P2** | 在隔离环境验证生成、安装、绑定和卸载；不可信项目脚本不得直接获得系统路径写权限 |
| **M15** | **逐 case 数值与性能记录** | `S/ascendc-evaluation/scripts/evaluate.py:448-450,559-656` 写 `PASS/FAIL/ERROR`、`total_cases`、`passed_cases`、`geometric_mean_speedup` 和 `all_passed` | E2 | **A–B / P2** | 保留逐 case 证据，增加 hidden/fresh shape、输出毒化、多次启动、路径审计和失败最小化 |
| **M16** | **性能异常时保留 precision-only 路径** | 评测代码在性能阶段异常时可继续精度路径并汇总结果 | E2 | **B / P2** | 只能标 `performance_unknown/inconclusive`，不能因精度通过就写成有性能证据或自动发布 |
| **M17** | **bridge/ABI 由接口决定参数顺序** | `S/easyasc-dsl-bridge/scripts/gen_bridge.py:362-394,433-472` 依据 aclnn 头生成 cpp/ModelNew | E2 | **B / P2** | ABI schema 必须独立校验；额外参数、错误 constness、workspace 和 tiling 不得由 wrapper 隐式改变 |
| **M18** | **直接 `exec` 生成代码** | `S/ascendc-evaluation/scripts/evaluate.py:287-301` 读取 reference/custom 文件后在当前进程 Python `exec` | E2 | **D：不照抄 / P0** | 改为最小权限、受控子进程/沙箱、只读评分材料、资源限额和超时；本轮未执行该路径 |
| **M19** | **`permissionMode: bypassPermissions`** | `A/cake.md:6` 明确声明；其他 Agent 配置也需逐个审查 | E2 | **D：不照抄 / P0** | 候选生成者不得继承该权限；外层 runner 是否另有限制仍须运行时验证 |
| **M20** | **stop hook 部分分支 fail-open** | `H/check_eval_done.sh:18-19` 未设输出目录时 approve；`:42-44` 解析失败或 total=0 时 approve，注释称避免 deadlock | E2 | **D：改为 fail-closed / P0–P1** | 缺目录、损坏 JSON、total=0、未知字段应 `block/error/inconclusive`，不能当通过 |
| **M21** | **Evo 与 evaluation 的字段契约不一致** | `A/cake-evo.md:371-372` 期待 `compilation_success/precision_passed/speedup/base_time_ms/gen_time_ms`；`evaluate.py:647-652` 写 `passed_cases/geometric_mean_speedup` 等 | E2 | **B / P1–P2** | 先统一 versioned result schema 和适配器；静态不一致已确认，是否另有运行时适配尚未核实 |

### 3.4 诊断、Advisor 与 Evolution

| ID | 机制与论文依据 | 当前代码事实 | 证据 | 决策与 Plan | 验收与边界 |
| --- | --- | --- | --- | --- | --- |
| **M22** | **Advisor 用确定性规则评分/路由瓶颈建议** | `S/code-performance-advisor/scripts/analysis_engine/cli.py:504-534` 有 weighted-Jaccard、coverage、冗余降权；`workflow.py:130-170` 有 raw score 到 fast/moderate/deep 的确定性路由 | E2 | **B / P4** | 这是 profiling 规则路由，不是论文 item-response DSL router；先测规则误报、漏报和 target 适用性 |
| **M23** | **roofline 与 fusion 分析**：论文 §4.1 以 Tmin、管道瓶颈和融合流量指导优化 | 当前 Advisor 有分析框架，但未核到与论文公式完全一致的统一实现 | E1 + 部分 E2 | **B–C / P4** | Tmin 是模型下界而非实测最优；target 未标定或模型不适用时返回 unknown，不强行停判 |
| **M24** | **Evolution 变体、Good/Medium/Poor 分层与精英迁移** | `S/cake-evo/SKILL.md:102-105,185-276` 和 `A/cake-evo.md` 描述流程，主要是 Agent/Skill 编排 | E2（文本） | **C / P4** | 先有正确候选、稳定 baseline、噪声门和确认协议；低速/profile 失败不得被排名逻辑隐藏 |
| **M25** | **AIC top-1 咨询与 PMR 经验提炼** | 论文 §4.2–4.3；当前未核到统一的因果验证或经验准入门 | E1 | **C / P4、P6** | top-1 建议只是假设；PMR 必须经正确性、移除/复测、适用域和人审 |
| **M26** | **headroom≈1 停止** | 论文明确 Tmin 是乐观分析下界，案例也可是预算耗尽而非达到 roofline | E1 | **C / P4** | 只能作为停止建议；模型错误、依赖/争用未覆盖或测量不确定时继续/停止都不得伪装成确定结论 |
| **M27** | **有界重试、诊断待闭合、同一失败两次回推 lowering/skeleton** | `A/cake.md` 与相关 skill 文本描述，未见统一确定性归因器 | E2（文本） | **B / P3** | 用稳定错误码、artifact/字段路径、最小证据和 repair owner；根因不明允许 unknown，禁止无限重试 |

### 3.5 知识、治理与 target 扩展

| ID | 机制与论文依据 | 当前代码事实 | 证据 | 决策与 Plan | 验收与边界 |
| --- | --- | --- | --- | --- | --- |
| **M28** | **CAKE Book：带来源、版本、trust/recency 的跨用户 wiki** | 当前树未定位到完整实现 | E1 | **C / P6** | 先建小型分类知识库；模型自述不得自动成为 trusted reference |
| **M29** | **Self-Evolving Agent 读取 CI、分析 skill、开 issue/PR** | 当前树未定位到完整 CI 平台和持续 Agent | E1 | **C / P6** | 只允许提案，不允许自扩权限、改评分或自动合并；先积累可审计 run evidence |
| **M30** | **知识经重复证据 + target 版本 + 真机 + 跨用户人审后升格** | 论文 §5.4；当前 `cake-review` 不等于该独立人审/真机升格链 | E1 + 部分 E2 | **B–C / P6** | 原则值得借鉴，但实现必须另建 reviewer、证据状态和退役；不能将自动 REVIEW/commit 当人审 |
| **M31** | **版本化知识与 stale-rule 处理：冲突新增 scoped 记录而非覆盖** | 当前未定位到统一知识生命周期实现 | E1 | **C / P5–P6** | 先规定 source、target、shape、baseline、测量、review 和 expiry 字段；旧 skill 降权不等于验证通过 |
| **M32** | **950 Regbase/membase/SIMT 路径选择** | 论文 Appendix A；本地有 a5/easyasc 设备分支，但完整论文路径未核到 | E1 + 部分 E2 | **C：仅目标需要时 / P5、P7** | Regbase、SIMT、a5 是 target 路径，不是新 DSL 数量；需要真实 950 能力指纹与独立验证 |
| **M33** | **simulator-only / report-only** | `check_device.sh` 对非 a5 host 可告警放行模拟；另一 codegen 路径有 report-only | E2 | **B / P2** | 结果必须标 `measurement_kind=simulation`，永远不能冒充真机、进入确认或自动发布 |
| **M34** | **无环境时询问用户、远程/本地/仅生成代码** | `A/cake.md:130-134` 明确要求 AskUserQuestion，而不是自行猜配置 | E2 | **A / P0** | 值得保留“显式询问/仅源码/弃权”三态；仅生成代码不得写成编译或数值通过 |
| **M35** | **排行榜/ Evolution 的最好值** | 论文主结果跨 benchmark 修订按算子取最好；案例有轮内最好候选和最终官方修复包 | E1 | **D：不作确认口径 / P4** | search best、官方快照与 confirmation 分账；不得把最终修复结果回填到所有中间轮 |
| **M36** | **开发机 48 向量核、评测机 40 核导致首轮 12 cases 失败** | 论文 §4.4 自曝 host UB 记账与 kernel layout 不一致，最终用共享 header 修复 | E1 | **B / P0、P4** | 这是 target 漂移的直接教训：device/core/UB/compiler/skill/binary 必须绑定并重测，本地通过不能外推 |

## 4. 明确不照抄的实现模式

| 模式 | 替换原则 |
| --- | --- |
| `bypassPermissions` | 候选最小权限；生成、评测、发布分权；评分材料只读 |
| 当前进程直接 `exec` 生成代码 | 受控子进程/沙箱、文件系统与网络限制、资源限额、超时、输出隔离 |
| 缺目录/坏 JSON/total=0 放行 | fail-closed；缺证据是 `inconclusive/error`，不是 pass |
| 把 SKILL 文本当强制 gate | 由 runner/compiler/schema 确定性执行，Agent 只能提案 |
| 先实现完整 router、CAKE Book、Self-Evolving Agent | 先有一个正确、可复测算子和可信观测，再逐步增加复杂度 |
| 把模拟器通过当真机通过 | 模拟只作预检，真机确认是独立证据层 |
| 把最高加速比或最终修复包当全过程表现 | 记录完整漏斗、失败分母、搜索与确认、成本和适用域 |
| 每遇到一个失败就增加一条 pitfall | 先判断表示/lowering/verifier/diagnostics 的根因，防知识库无界膨胀 |

## 5. 从 E2 到 E3：首批离线 fixture

这些 fixture 不需要 NPU，可在制定 P0/P1 后实现；目前只是设计，尚未执行。

| fixture | 应命中的机制 | 预期结果 |
| --- | --- | --- |
| 缺 dtype/layout/output shape、未知字段、非法 eps 的 op_desc | M01、M07 | schema 拒绝并定位字段，不进入生成 |
| reference 被候选修改、输入 seed 改变、baseline 未冻结 | M02 | 权限或合同门拒绝，不能继续验收 |
| 缺中间 artifact、hash 不符、阶段失败却出现成功 commit | M03–M05 | 状态机拒绝，保留上次成功状态 |
| skill 未写 END、乱序、重复 retry、trace 损坏 | M06 | runner 标记 infra/inconclusive，不伪造成 skill 成功 |
| 无可行 track、未知设备、模拟结果申请发布 | M09、M32–M34 | 明确 abstain/unknown；模拟不得进入真机确认 |
| 缺输出目录、结果 JSON 损坏、total=0、未知结果字段 | M20–M21 | fail-closed，不批准停止或发布 |
| 生成代码尝试读取 golden、写计时文件、访问网络或越权路径 | M18–M19 | 沙箱拒绝并记录安全事件，不影响评分材料 |
| API、tail、buffer、alias、缺 wait、错误并行变异 | M07、M13、M15、M27 | 静态门能拒绝的提前拒绝；其余由数值/最小反例路由到正确 owner |
| profile 失败、低速、headroom 模型不适用 | M16、M22–M26 | 标性能 unknown，不进入排名或确认；不得把低速当正确性失败 |
| 未验证经验、旧 target/skill 经验、相互冲突经验 | M11、M28–M31 | 不进入 trusted context；保留来源和冲突，不自动覆盖或升格 |

## 6. 后续 plan 映射与解锁条件

| Plan | 对应机制 | 何时可以启动 | 解锁下一阶段的最低证据 |
| --- | --- | --- | --- |
| **P0：合同、权限、可信基线** | M01、M02、M07、M18–M20、M34、M36 | 现在可做离线设计 | 合同、oracle、权限和 target 指纹完整；坏合同/未知状态 fail-closed |
| **P1：Artifact 与阶段门** | M03–M07、M13、M21 | P0 schema 通过 | 阶段失败无成功 commit；artifact/hash/trace 可重放，字段契约统一 |
| **P2：构建、绑定与独立评测** | M13–M17、M33 | P1 + 获得真实环境授权 | 能部署调用；tail/fresh/路径通过；模拟与真机分账 |
| **P3：局部诊断与修复** | M27、M21 | P2 有真实失败样本 | 注入 bug 能定位到字段/阶段/owner；无证据时 unknown |
| **P4：Profiling 与受控优化** | M22–M26、M35–M36 | 已有正确候选和可信 baseline | 正确但慢不发布；search best 与 confirmation 分离，成本受控 |
| **P5：多 DSL 路由** | M08–M12、M32 | 至少两个 track 都有同口径交付记录 | 无可行 track 会弃权；版本/infra/作弊正确处理，路由优于固定 track 或保持简单固定路由 |
| **P6：Wiki 与经验生命周期** | M06、M11、M25、M28–M31 | 有版本化、可复核的成功与失败证据 | 未验证经验不进 trusted context；移除检验、人审、退役与冲突记录有效 |
| **P7：可选 Typed IR** | M07、M13、M32 暴露的表示缺口 | 复发失败已证明不是知识、harness 或后端问题 | 与现有 DSL+精选上下文同预算比较有净收益，否则不增加这一层 |

## 7. 当前结论

1. **先借工程结构，不借权限与评分实现。** M01–M07、M13–M17 是最有现实价值的主线。
2. **先补确定性合同与验收，再谈 router 和知识自演化。** M08–M12、M28–M31 当前主要是论文机制，且需要更多观测基础。
3. **M18–M20 是必须替换的安全边界，不是可复用功能。**
4. **当前论文与代码差距本身是重要输入。** 后续每个 plan 都应先列 commit、代码证据和未决项，不能用论文完整系统名义掩盖当前实现缺口。
5. **下一步只建议展开 P0/P1。** 其余 plan 应按解锁条件逐个评审，而不是一次性全线复刻 CAKE2。
