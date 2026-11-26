# plan-05：独立评审与反共识

状态：设计计划，尚未实施。本文只规划评审质量、来源独立性、成本度量与人工队列；默认离线，不调用模型、不使用 GPU。文中新增命令、字段和测试均为规划目标，当前不存在；本文编写时未运行测试。

## 当前决定

- **本批只做**：先数历史中有多少提案还能对上原 prompt SHA、多少只能标 `reconstructed/current`，并冻结盲标与抽样规范。
- **输入**：历史 proposals/reviews、REVIEW/prompt、证据包与反馈记录；没有原始 SHA 就只标 reconstructed。
- **本批不做**：样本不足时的分层区间、Brier、自动轮换或全套 `test_review_quality.py`；不新增付费 reviewer、debate 或 GPU。
- **完成产物**：一份样本覆盖/可回放计数和是否足以建集的 go/no-go；不足就停在计数与标注规范。
- **失败去向**：先读 `logs/pytest/triage/triage_summary.md` 及 handoff；冻结 JSON 重放不得冒称重新评审。

## 1. 角度与问题：quorum 是召回，review 也需要被评审

- 双来源 quorum 的回答是「两个受信模型提出了同一 action」，只是候选召回机制；它既不证明证据独立，也不证明建议有用，更不证明 harness 改后有效。现有文档已明确这一边界（`docs/可信harness/架构与理论.md:290-298`）。
- 单来源只读 review 降低了召回门槛，但 reviewer 也会受 shared context、作者身份、证据缺口和自身偏差影响；一次 `useful=true` 不是人工真值。两条路径目前都没有以人工参考标签为基准的质量、分层误差和成本—质量报告。
- 本文评价的是「对一份冻结证据包，评审判断是否可靠」，不是内核加速效果。后者必须交 plan-02；人工判 useful 也不能改写成 `supports_effect`。
- 小额试点只证明链路曾运行：第二次 quorum 为 0 且 review 后无后续场次，第三次 TASK 未实际含 overlay（`docs/可信harness/试点结果.md:76-91,106-122`）。它们不能充当 reviewer 准确率或 quorum 有效性的基线。

## 2. 目标 / 非目标

| 目标 | 非目标 |
| --- | --- |
| 建人工标注、版本化、可回放的离线评审集，报告一致率、漏报/误报、风险漏判、覆盖率和成本 | 把固定标注集当成永久真值，或用它自动 hill-climb prompt / reviewer |
| 记录证据与上下文血缘，在关键分歧点比较证据，而非累加票数 | 增加辩论、多轮互相说服或多数投票自动裁决 |
| 在付费前做 reviewer 资格、回避、轮换和预算选择，并保留完整选择审计 | 重新设计 trust 生命周期、授权、撤权或自动升降信任（plan-03） |
| 让人工只处理不确定、风险、冲突和高影响项，并把复核结论形成可消费观察 | 自动 P2、改验收器、做效果实验、设计签名或运行时注入防线 |

## 3. 现状证据：独立性、事实边界与成本到哪一步

| 现状 | 代码证据与质量缺口 |
| --- | --- |
| quorum 要求不同 `provider+model`、两个委员会、双方 active trust、同 key、不同 probe SHA；conversion 与未核时间行不计票 | `taic/harness.py:856-884`。不同 SHA 只排除字节完全相同的探针；不检查探针是否测不同内容，也不检查共同 NOTES、experience、overlay 或文字来源。相同 probe 的回归测试只覆盖这一层（`tests/shared/test_harness.py:344-362`）。 |
| 自动 review 从无决策、无 review 的 valid tip 中取首条，要求 probe 时间已核且有 SHA | `taic/harness.py:780-802`、`taic/orchestrator.py:2518-2521`。无 probe 的 tool/reporting/wording 合法提案不在自动 review 召回内；按 key 去重也不表达「证据变化后重审」或「同 key 多 reviewer」。 |
| reviewer 默认取另一委员会第一个成员；回避按 `provider+model`，conversion 保留原始作者 | `taic/orchestrator.py:2524-2532`、`taic/harness.py:706-777`。自动路径只查当前 row，手工路径才查同 key 全组；最终 `settle` 再排除所有作者并查 reviewer active `overlay:tip` trust（`taic/harness.py:937-967`），所以资格预检与付费选择尚未统一。 |
| 自由文本转换必须显式人工 `confirm`，原始文本与 `source_author` 留痕 | `taic/harness.py:1489-1541,1610-1645`。AI 转换目前只记 `filled_by=ai`，未记 converter 的 provider/model/effort；原作者回避已有测试，转换者回避无法完整核对。 |
| review 输出只有布尔 useful、confidence 和自由文本 reason/evidence/risk/test | `taic/review.py:475-520`。没有独立 `uncertain` 结论；`settle` 看 useful、协议、作者、trust 和源探针，不把 confidence、risk 或 `missing` 作为独立硬门（`taic/harness.py:928-991`）。现有测试证明 useful review 可激活，不证明判断正确（`tests/shared/test_harness.py:453-474`）。 |

证据包还要分清「宿主计算」与「宿主读到代理可写内容」：

- 宿主直接计算/筛选：run 内路径约束、probe 是否存在/大小/当前 SHA/是否变化、mtime 时间状态、candidate report 数字摘要、`trusted_record` 与 `missing` 列表（`taic/review.py:158-219,275-377`）。这些是有界检查，不是 probe 已执行或命题为真的证明。
- 仍是提案者/转换者文本：action、rationale、reproduction、expected、measurement、原始自由文本、probe 正文；它们被截断并放在 untrusted 区（`taic/review.py:385-477`）。source author 也会进入 prompt，虽不用于判真，却可能带来作者/委员会偏差。
- `_self_check_view` 从 agent work 下的 JSONL 选行，只把数字放入 Outer checks（`taic/review.py:223-272`）；这比自由文本更窄，但不等于 host authority 或签名记录。测试只保证自检自由文本不进入该段（`tests/shared/test_review.py:190-209`）。证据来源升级交 plan-04，本文要求把 `source_class=host|workspace|unknown` 显式展示，不把「Outer checks」一概当独立执行事实。
- 预算按实际单价、prompt 字节、输出上限和缓存 allowance 估算，再乘余量（`taic/review.py:25-75`）；成功行记录 expected/budget、由 usage 与配置价格换算的 `cost_usd`、tokens、耗时及 prompt/review SHA（`taic/review.py:601-627,632-669`）。它不是供应商发票；缺 usage 时成本为 unknown，自动路径不采纳该 verdict，失败花费进 ledger（`taic/orchestrator.py:2566-2574`），但失败 attempt 尚无统一结构化质量记录。
- flow 只执行一次有预算的 reviewer 回合（`taic/flows/feedback_review/__init__.py:31-50`）。现有测试覆盖价格缩放、有界 prompt、旧 verdict 不复用等（`tests/shared/test_review.py:17-30,159-170,243-262`），没有人工一致性、校准、轮换公平性或边际成本测试。

## 4. 评审质量基线：人工参考集与离线回放

### 4.1 建集、标注与防过拟合

1. 先盘点历史 `harness_proposals.jsonl`、`harness_reviews.jsonl`、feedback 与仍存在的证据。量少则全量，量足则按 kind、origin、target/task、quorum/单源、review 正负/未评、证据完整/缺失、reviewer 分层随机；风险、conversion、分歧、quorum 成立且 review useful、verifier 通过或 host 无 blocking 但未人审、上下文截断/scope 受限案例强制纳入，避免只从失败或分歧估计质量，并保存抽样概率以还原总体估计。host-invalid 提案只做路由 fixture，不混入「valid proposal 是否 useful」的分母。
2. 每个 case 冻结 `dataset_version/case_id/split/run/proposal_key/source_seq/claim_digest`、规范化 bundle 与 SHA、原 prompt 与 SHA、review 输出、价格/成本、rubric 版本和分层字段。历史 `REVIEW.md` 可能被后续复用覆盖（`taic/review.py:558-559`）；只有其 SHA 与日志 `prompt_sha256` 相符才可称原输入，否则标 `reconstructed/current`，不得冒充当时证据。
3. 标注主类固定为 `useful / not-useful / uncertain`，另用一个正交维度标风险：`risk_severity=none|low|blocking` 加可选 `risk_types`。第一轮把 `risk` 当互斥主类，结果一个「有用但需护栏」的提案只能被标成 risk，于是 reviewer 说 useful 的案例在混淆矩阵里全变成误报，指标会系统性惩罚正确判断。现在的规则：blocking → 主类仍可为 useful，但 `risk_severity=blocking` 单独计入风险漏判，不得自动激活；low → `useful_but_risky`，与「reviewer 误报」分开报。仍标 `evidence_sufficient`、`scope_ok`、关键证据引用和一句理由；需要新执行才能判断时选 uncertain，不以想象补结果。
4. 另标与主类正交的 `failure_attribution=harness/model/environment_or_protocol/mixed/undetermined` 和 `contract_complete=yes/no/not_applicable`；可行动提案的 contract 至少含 `target/trigger/expected_observable/at_risk/falsifier/evidence_refs`。归因须核对环境合同、轨迹行为与影响范围，不能由失败分数、prompt/code 形式或复发次数决定；这里只评可证伪性，不做 plan-02 的效果验证。
5. 标注界面先隐藏 reviewer/quorum 结论、provider/model/committee、confidence 和 commit/合入/采用状态，随机展示顺序；标注者不得把任务转交给被评模型，也尽量避开自己提出、转换或审批过的 case。每个标签记录 `label_provenance=human_blind/human_adjudicated`；LLM-reviewed commit 的结论只作被评 route 结果，不作人工参考标签或独立证据。重叠样本由第二人盲标，分歧保留原标签后 adjudicate，并报告标注者一致率与裁决率；标签和 holdout 不进入 REVIEW、TASK、NOTES、experience 或 few-shot。
6. 按 proposal group 与 run/时间切开发集和锁定确认集，防止同 key 近重复跨 split。任何针对确认集结果的 prompt、阈值或选择器修改都把它降级为开发集，再换新确认集；禁止自动搜索最优标签规则。**默认只承诺公开的 dev/frozen 两个 split**；私密 holdout 是未解决的可选项，需 plan-06 的 host-owned、agent 不可读的最小设计与读取回执，落地前本计划不声称存在秘密 holdout。

### 4.2 回放与报告指标（样本与原始证据足够后才启用）

离线回放只消费冻结的模型输出和人工标签；新路由用 scripted/fake reviewer，不能把「重放旧 JSON」写成重新评审。没有对应冻结输出的策略只能评路由，不得补造模型准确率。报告至少包含：

| 维度 | 口径 |
| --- | --- |
| 一致与遗漏 | 全量 exact agreement、各类 recall/precision、macro 指标和 coverage；uncertain 不硬塞进二分类，同时单列 selective agreement，避免只报容易案例。 |
| 误报 / 漏报 | 误报 = reviewer useful、人工 not-useful（不含 `useful_but_risky`）；漏报 = reviewer not-useful、人工 useful；风险按 `risk_severity` 单列，且必须同时报告两种映射：四类映射（原口径，risk 互斥）与三主类 + 风险维度（新口径），否则同类数据换个标签就会多出或消掉一批误报。quorum/reviewed_active 也分别报告人工 useful 比例与候选覆盖，不把激活率当准确率。 |
| 一致通过与覆盖 | `concordant_positive_error` = quorum 成立、review useful 且 host 无 blocking 时，人工 not-useful/risk 的比例；另按 `review_scope` 报 `unreviewed_surface` 与风险漏判，一致不当作正确。 |
| 校准与分歧 | confidence 的可靠性图/Brier 等仅在样本足够时报告；缺失 confidence 不补造成绩。按分歧类型、shared-context/evidence 状态报告冲突解决结果。 |
| 分层与不确定性 | 按 kind、origin、target/task、provider/model/effort、证据完整性和 rubric 版本分层；区间按 proposal/run 聚类重采样，小样本单元写 `insufficient`，不强行排名。 |
| 成本—质量 | 每次 attempt 的预算、已知实际成本、unknown、tokens、耗时；每案成本、每个人工一致结论成本、人工队列量与复核时长（若采集）。按唯一 claim 报 `cost_per_unique_claim`，按 recurrence group 报组成本，额外 reviewer 另报每新增人工确认风险和每解决分歧的增量成本与人工分钟；quorum 的搜索生产成本与 review 增量成本分栏，无法归因记 unknown。 |

第一版只形成基线与误差清单，不预设准确率、成本或轮换权重阈值；是否达到可自动路由的水平由人另批。该基线也不评估采纳后的搜索效果。

## 5. 反共识：核验分歧点，不再造一场投票

1. 出现以下任一情形，进入 `divergence_check`，而不是加第二票：两个来源对测量含义、expected、scope 或风险给出不兼容分支；quorum 与 reviewer 结论冲突；不同 probe 被用来支持同一关键命题但内容/输入/指标不可比；reviewer useful 与 host blocking fact 冲突。无法形成可核分支时也进人工队列。
2. 分歧记录采用 `claim_id/claim_type -> branches[{source, observation, attribution, evidence_refs, falsifier}] + host_facts + competing_signals + supports/contradicts/unknown + missing + adjudication`。宿主确定信号、模型归因和人工观察分栏，模型 tag 不得升级为 host fact；保留 secondary/competing signals，无证据归因填 `undetermined`，现有类型不匹配填 `novel`。宿主先比较结构化字段与布尔事实；rationale 的语义分支必须由有界原文引用支持，不能靠无出处摘要。模型辅助抽取若将来加入，同样需离线标注验证和真实调用授权。
3. reviewer 先独立看证据，不看来源身份、票数、他人 verdict/confidence；host 仍用隐藏身份做回避。输出针对每个关键分支的证据判断，而非「多数人因此对」。第二 reviewer 仅用于预声明的离线校准、抽样审计或人工要求，保留全部结论，不做多数表决。
4. **证据引用必须宿主可核。** 评审、标注与人工复核提出的证据引用分两类，都带 `path + quote`，引文（≥ 24 字符）在空白归一后必须逐字出现：
   - **run artifact**：路径必须落在本次 run 的 artifact 树内（Harmony Harness App. E.1 的原始约束）。
   - **仓库代码/文档**：记 `commit`；若引用的是未提交的工作树，则记 `HEAD + dirty` 与工作树文件 SHA，并标明该引用只在本次工作树有效、不能当已提交事实。另记仓库相对路径与行窗口，宿主在离线 checkout 上核文件级 SHA、行窗口与引文。**不能要求仓库引用落在 run 树内**，否则评审引用 `taic/acceptance.py`、`tests/` 这类证据会被全部判失败。
   不通过时允许一次重试；这次重试是**又一次付费 reviewer 调用**，与 `search.provider_retries` 一样单独计费和计数，usage/预算未知时不重试、直接记 `evidence_unverifiable` 并进人工，不计作支持。**注意这是本仓库更严的本地路由**：Harmony App. E.1（p.20）在证据仍不可核时是抑制该 plan 的 guidance、operator 继续 unguided，并不拒绝整个候选；本文选择「进人工且不得激活」，理由是本仓库的自动通道比论文的 operator 循环更窄。它**只证明引文确实在那份文件里**，不证明引文支持命题，后者仍是人审与他人证据比较的工作。
5. 证据独立性新增版本化血缘：`probe_sha256`、可读时的规范化文本 SHA、有界内容/摘录、测量对象/输入域/expected、执行证据 ID，以及每项是 host-derived 还是 agent-declared。相似度只能提示复查，不能证明语义独立；执行绑定本身交 plan-04。引文核验只复用第 4 条的 `path + quote`，不另建第二套证据格式。
6. 对 NOTES、target experience、wiki、TASK/overlay 和 guidance 快照记录 `artifact_id/path/SHA/source_session/read_provenance`。只有宿主轨迹能证明时才记 read；否则是 unknown，不能从「没记录」推出「没读过」。两个来源共享材料时标 `shared-context`，共享或派生探针标 `shared-evidence`；这不等于建议必错，但不得再宣称两条独立证据。每对 source/reviewer 另记 `author_disjoint/context_disjoint/evidence_disjoint=yes/no/unknown`；不同 provider/model 只说明身份不同，不得据此把任一维度填 yes。
7. 每个 bundle 记录 `review_scope{context_mode,included_surfaces,excluded_or_truncated}` 与 `coverage_missing`；`no finding` 只适用于 included surfaces，关键 surface 未审或截断时 `scope_ok=unknown` 并进人工，不得写成 clean 或无风险。
8. 复发只生成 `recurrence_hint{pattern,evidence_refs,distinct_tasks,distinct_runs,distinct_models,shared_context}`，用于分层抽样、去重与队列优先级；同模型跨任务复发可更新 `distinct_tasks`，但不增加模型/评审独立性，共享上下文或派生证据不增加证据独立性；复发不得单独改变 `failure_attribution/useful/risk` 或路由结果。
9. 不加 debate：Judging with Many Minds 报告首轮后多类偏差会被 debate 放大，且额外回合增加费用与从众机会（`docs/可信harness/papers/导读.md`）。本计划采用一次性、证据有界的 meta-review；不以更多 CoT、更长理由或互相反驳作为质量代理。

## 6. reviewer 选择、轮换与成本约束

- 付费前统一硬门：属于配置委员会；当前 target/task/prompt/protocol/effort 下具有 active `overlay:tip` trust；与同 key 所有原作者及 conversion 原作者不冲突；预算可入 ledger。自动与手工路径都先用 `reviewer_conflicts_with_group`，无合格者则不调用、进人工/待审状态，而非先花钱再由 settle 否决。
- AI 转换补记 converter 的 provider/model/effort，并纳入回避；人工转换保留具名确认人。自由文本转换始终由人触发，reviewer 不能代替转换批准，也不能审自己的原文或结构化文本。
- 在合格池内优先不同 provider/model，记录可知的模型家族与 effort 多样性；effort 只作分层，不预设越高越准。用可复现 seed 做分层轮换，避免永远 `members[0]`，并记录候选、排除原因、选择器版本和估算价格；没有基线前不按自报 confidence 或微小榜单做质量路由。
- trust 是人工授予的参与资格，不是评审能力证明。离线质量只用于合格池内的抽样、轮换和人工决策建议，不自动扩权；同名模型的评审能力与提案能力也不得互相推断。
- 每个候选 reviewer 都按自身价格、实际 prompt、输出和缓存 allowance 重算预算，不得回退固定 $0.2；一批一案保持现状。批量评审只做离线对照，按 `claim_digest/recurrence_hint` 比较按案与按组的准确率、跨案污染、`cost_per_unique_claim` 和人工队列量，不因省 token 默认开启；额外 reviewer 必须证明边际价值，任何真实模型调用均为 **需人工授权：付费模型**。§5.4 的引文重试也计入同一份 review 授权上限与单案预算；授权单未列这一项时不得重试。

## 7. 人机分工、最小队列信息包与信任接口

| 情形 | 路由上限 |
| --- | --- |
| useful、`risk_severity=none`、无 blocking missing、关键分支一致、独立性满足当前策略、仅同 run P1 tip | 可沿现有 reviewed_active 候选路径；仍不是效果验证或 P2，启用新策略前先 shadow |
| uncertain、`risk_severity>=low`、关键 surface 未审/截断、probe 改变/缺失/不可读、关键证据冲突、复发但归因 undetermined、转换原文缺失或保真有争议 | 只能进人工队列，不能由第二票覆盖；宿主 blocking fact 不依赖 reviewer 是否忽略它 |
| quorum/review 冲突，或需要依赖 shared/unknown 独立性才能激活 | 人工裁决证据；不得写成「两个模型同意所以通过」 |
| 跨 run/target、P2、合同/预算/信任/协议或评测语义 | 维持人工批准，不属于自动 review 放行范围 |
| 完整证据下的 concordant not-useful | 不激活、归档并抽样人审；review 超时/解析失败/成本 unknown 是 attempt 状态，不是 not-useful 标签 |

人工最小信息包只含裁决所需内容：原自由文本与结构化差异、scope/kind/origin、source/reviewer/converter 身份及回避结果、冻结 bundle/prompt/rubric SHA、probe 双 SHA 与摘录/截断状态、self-check 来源类和匹配数字、report 摘要、host missing、分歧分支与 NOTES/experience 血缘、reviewer 理由/risk/recommended test、成本/unknown 和历史人工标签。默认不重跑 probe 或 GPU。

复核结论追加为版本化 `review_quality_observation`：case/review/proposal 身份、subject 类型与身份、人工主类与风险、证据/scope 判断、annotator/adjudicator、理由、rubric/dataset 版本；对 proposer 保留人工判定，对 reviewer 由宿主推导 `aligned/false_positive/false_negative/risk_missed`，另记 route 结果。plan-03 可消费这些观察作为 grant 审查材料；本计划不设置挂起阈值、不自动 grant/revoke，也不把 useful 写成 confirmed effect。原标注、裁决和后续改判全部保留，不覆盖历史。

## 8. 分阶段步骤：先度量与回放，再改流程

- **Phase 0（本批）：样本盘点与标注规范。** 统计历史 `harness_proposals.jsonl`、`harness_reviews.jsonl` 与证据中，多少 case 的原 prompt SHA 可对上、多少只能标 `reconstructed/current`、多少已不可回放；冻结抽样、盲标、adjudication 与 split 规则。
- **退出条件。** 只产出覆盖计数和 go/no-go。样本不足或原 prompt 留存不足时停止，不建分层区间、Brier、自动轮换或全套 `test_review_quality.py`；数量足够后才另批离线 instrumentation、路由回放、shadow 与可选真实评审。任何真实模型/GPU 仍另行授权。

## 9. 验收标准（离线、遵循 tests/README.md）

本批只验收 Phase 0：schema/抽样规则可复现；prompt SHA 相符才称原输入，不符只能标 reconstructed；host-invalid 只做路由 fixture，不混入 useful 分母；每个分层给出实际样本数与未知数；go/no-go 能解释为什么继续或停止。只有样本与原始证据足够时，才启用后文的数据集、指标、独立性与成本验收；它们不是当前必须创建的全套测试。

```bash
# 现有回归；新增覆盖统计工具与测试实现后才可运行
uv run pytest -q -n 0 tests/shared/test_review.py tests/shared/test_harness.py
uv run pytest -q -n 0 tests/shared/test_harness_conversion.py tests/shared/test_structure.py
# 后续：uv run python -m taic.cli harness review-quality --dataset DATASET --replay RECORDED_JSONL
```

失败先读 `logs/pytest/triage/triage_summary.md` 及 handoff。真实模型评测、第二 reviewer 和 GPU 均不属于本批。

## 10. 风险、回滚、边界与参考

| 风险 | 控制与回滚 |
| --- | --- |
| 人工标签本身有噪声或把偏好当真值 | 盲标、重叠标注、uncertain、裁决留痕；按 rubric 版本重放，不删除旧结论。风险主类改成正交维度后，旧四类标签仍按原值保留，两种映射并列报告而不是重写历史 |
| 历史样本少、正例/ quorum 选择偏差、分层伪精确 | 加权与聚类区间、小单元 insufficient；回退到描述性报告，不选「最佳模型」 |
| 标注集成为新 reward、确认集反复泄露 | group/time split、新确认集、禁止自动调参；撤回受污染比较并更换 split |
| 血缘缺失被误当独立，或语义分歧抽取错误 | unknown 失败关闭到人工；新路由仅 shadow，按 `selector/rubric_version` 切回 legacy，保留全部审计 |
| 队列过载、轮换和第二评审增加成本 | 风险/冲突优先，concordant negative 抽样；停止额外 reviewer 并恢复单 review，不降低 host 硬门 |

边界：plan-01 负责评测反作弊，plan-02 负责效果与成本实验，plan-03 负责 trust 生命周期，plan-04 负责证据来源/签名/执行绑定，plan-06 负责注入与运行时隔离（私密 holdout 的 host-owned 设计也归它，且尚未落地）。本文只提供质量观察、证据独立性与路由；active 回滚沿用现有 run 控制，不设计新的放权通道。

方法参考而非本仓库效果承诺：AgentAuditor 的 confabulation consensus 与关键分歧点分支证据比较（`docs/可信harness/papers/导读.md`）；Judging with Many Minds 对 debate 偏差放大的结果（`docs/可信harness/papers/导读.md`）；RuVerBench 的 2,458 条人工 rubric 判定、评委噪声、batch 权衡和多数投票收益递减（`docs/可信harness/papers/导读.md`）。论文数字属于原研究，本仓库尚未复现；仓库侧问题定义见 `docs/可信harness/架构与理论.md:257-344,391-423`。

## 附录：文献与变更索引

只作查阅，不再逐轮重述正文；论文数字均未在本仓库复现。

| 日期 | 影响位置 | 证据文件 |
| --- | --- | --- |
| 2026-10-25 | 文献对应的正文规则与边界 | `acl2026_auditing-multiagent-reasoning-trees.pdf`、`acl2025_judging-with-many-minds.pdf`、`2606.29920_llm-judge-rubrics-agentic.pdf`、`2609.11677_ecdysis-runtime-harness-training.pdf`、`2610.15150_harmony-harness-multiobjective.pdf` |
| 2026-10-26 | 当前决定、阶段、验收与跨 plan 归属 | `审核意见-plan-01至09.md` |
