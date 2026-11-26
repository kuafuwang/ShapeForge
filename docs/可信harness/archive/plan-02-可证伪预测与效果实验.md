# plan-02：可证伪预测与效果实验

> 状态：设计计划，不是实验结果。默认只做离线与 dry-run；任何真实模型调用或 GPU cell 都标记为「需人工授权」，绝不成为默认步骤。

## 当前决定

- **本批只做**：预测合同、dry-run/mock、按 pair ID 的采集、`run_overlay` 宿主送达函数，以及零宽/窄区间不得出确认词的离线保护测试。
- **输入**：acceptance spec、现有 runner、`treatment.patch`、mock cells 与 TASK/overlay fixture。
- **本批不做**：确认性 `supports_effect`/`refuted`、真实 unchanged-control 校准、付费或 GPU 实验；`confirmatory_min_pairs` 在有效校准授权出现前是**不存在**，不是待填参数。
- **完成产物**：可 dry-run 的计划、宿主送达/阴性回执、探索性原始点与 `inconclusive`；不产生采纳结论。
- **失败去向**：先读 `logs/pytest/triage/triage_summary.md` 及 handoff；注入路径未实现时 overlay 计划只能 dry-run。

## 1. 角度与问题：review 有用，不等于 harness 有效

本计划只回答一个问题：**某条提案对应的某个确定改动，在冻结条件下能否产生预先声明、足以覆盖代价且可复测的任务级效果。**

- `useful=true` 是 reviewer 对建议价值的判断，可作为进入实现/实验的先验，不是任务结果；现有验收模块也明确把二者分开（`taic/acceptance.py:1-6`）。
- `reviewed_active` 只表示独立 review 后允许单来源 tip 在本 run 后续场次生效，不表示效果已验证，更不是 P2 晋升（`taic/harness.py:947-991`）。
- 同一 run 的“激活前/后”不能自然归因： incumbent、NOTE/FEEDBACK、剩余预算、委员会轮换和失败历史同时变化；overlay 事件也不证明 TASK 实际注入了文本。
- 三次小额试点没有补上这个缺口：首次没有验证 active 影响（`docs/可信harness/试点结果.md:37-44`）；第二次 `reviewed_active` 生成时已无后续场次（`docs/可信harness/试点结果.md:87-91`）；第三次 r2/r3 的 TASK 没有 overlay（`docs/可信harness/试点结果.md:106-119`）。人工应用 target 经验 patch 也“不是测量结论”（`docs/可信harness/试点结果.md:122`）。

因此要把“建议有用”改写为一条能在执行前冻结、执行后可判为支持/反驳/不确定的预测，而不是把 review 置信度当成效果大小。

## 2. 目标 / 非目标

**目标**

1. 复用 acceptance spec 与实验 v2，为每个提案形成唯一的预测对象、对照、方向、最小效应、护栏和失效条件。
2. 建立按 seed/pair 身份配对的采集与推断，区分 P1 run overlay 和 P2 仓库 patch。
3. 让风险等级决定所需证据，让效果、采纳成本、持续运行成本共同决定 adopt/defer/reject/rollback。
4. 先以模板、dry-run、mock 和统计单测闭环，再允许具名授权的小额真实配对实验。

**非目标**

- 不重写判分、正确性、容差、安全验收器，不设计 reviewer rubric，不做看板 UI。
- 不把 search best 当 confirmed speedup，不承诺跨 task、跨模型或跨硬件泛化。
- 不自动 apply/commit，不把 legacy 人工采纳追认为实验验证，不修改信任授予逻辑。

## 3. 现状证据与真实缺口

| 现有层 | 已有事实 | 本计划要补的缺口 |
| --- | --- | --- |
| 验收合同 | spec 已绑 proposal/review/scope，并有 `effect_metric_id`、`guardrail_metrics`、`control`、`rollback`（`taic/acceptance.py:82-128`）；实现记录另绑 patch（`taic/acceptance.py::implementation_report`，当前 `:225-232`）。 | metric 缺省会落到成本（`taic/acceptance.py:109`），还不是每条提案都显式声明的可证伪预测。 |
| 宿主验收 | host check 只能引用登记 ID，skip 不算通过（`taic/acceptance_checks.py:27-31,62-85`）。 | 实现验收只证明“改动做到了”，不能替代任务级效果。 |
| 实验计划 v2 | 已冻结 base commit、配置、委员会、protocol、seeds、预算、方向、最小效应与护栏；默认 dry_run（`taic/harness_experiment.py:198-249`）。 | `direction=lower`、`min_effect=0` 仍是 CLI 缺省（`taic/cli.py:1400-1413`），授权前应显式确认而非事后挑选。 |
| 执行 | 同 commit 两臂 worktree，按 seed AB/BA；真实执行另需双授权与具名人（`experiments/harness_runner.py:23-83,123-159`）。 | treatment 目前就是 `treatment.patch`（`experiments/harness_runner.py:40-43`），没有直接表达 run overlay 的送达与注入证据；也没有任何把固定 overlay 只装进 treatment 的宿主路径。 |
| 采集 | 已核对 config/protocol/change/base/seed，并读取五类指标、护栏和账单（`taic/harness_experiment.py:362-466`）。 | 独立 `collect-experiment` 只接一个 control 目录和一个 treatment 目录（`taic/cli.py:994-1001`）；多 cell 的计划驱动恢复采集需补齐。 |
| 推断 | 成本/时间用 log ratio，其余用差值；少于 `min_pairs` 不确定，bootstrap 后仍由人复核（`taic/harness_experiment.py:483-529,606-625`）。 | 当前按数组位置 `zip` 配对（`taic/harness_experiment.py:493-505`），且反向效果/护栏越界也归入 `inconclusive`，缺少明确 refuted。另一个缺口：现有 `min_pairs` 单一门槛让 3 对零宽/极窄区间就能越过“支持”判据；本批只交付探索性保护，真实确认与校准另见[确认性效果实验-另纸.md](确认性效果实验-另纸.md)，不把确认参数写成近期待填项。 |
| 重复与历史 | `repetitions` 有计划字段，最新 verdict 按 experiment 去重（`taic/harness_experiment.py:581-600`）。 | `pair_count=len(seeds)//repetitions` 与读取时 `pair_count==len(seeds)`、seeds 唯一约束不一致（`taic/harness_experiment.py:232,276,296-300`）；重复 seed 也不能当独立对。跨 experiment 的旧 true 仍可让里程碑为 true。 |
| 采纳成本 | review/实现/实验分阶段计费，已发生但缺账为 unknown（`taic/adoption.py:55-108`）。 | 尚未计算效果相对成本的收益；shortlist 按 evidence score 和 known cost 排序（`taic/adoption.py:133-150`），unknown 不能因此显得“便宜”。 |

## 4. 提案级可证伪预测：复用合同，不另造平行字段

把 **ExperimentPlan 视为预测的冻结信封**，预测文本由它生成；不要再建一套与 spec/plan 可漂移的 `prediction.metric`、`prediction.baseline` 字段。

三个 plan 都有「双向预测」，但层次不同，不能互相顶替：plan-01 的 `should_reject / should_still_pass / at_risk_solvers` 是 **fixture 级**（候选应被拒、合法解应通过），plan-04 的 `should_fix / at_risk` 是 **证据/修订级**，本计划的 `predicted_observables` 是 **实验级**。实验级的确认不能当 fixture 级或证据级的验证。

| 预测语义 | 唯一事实来源 | 需要的收紧 |
| --- | --- | --- |
| 预测对象 | `proposal_key + spec_sha256 + change_sha256 + scope` | treatment 描述增加 `delivery=run_overlay/repo_patch`；overlay 的规范内容仍由 change digest 绑定。`delivery=run_overlay` 还须冻结注入方式：由宿主在首个适用场次前把一条已 settled、固定 digest 的 overlay 写入 treatment run，control 不写；两臂除该条外配置逐字段相同。 |
| 证据与改动单元 | spec 增加 `evidence_refs`，treatment 增加 `change_atoms` | 均为计划字段并纳入 experiment 绑定；失败提案至少记录原始 trace 与去重 task/kernel ID，其他提案链接原始 artifact；每个 atom 记录 `component/patch_hunks/regression_watch`，汇总诊断不替代原始证据。 |
| 双向预测（AHE §3.3/App. B.2、One Recipe A.7） | 计划新增 `predicted_observables.host_facts` 与 `predicted_observables.directional` | 两类**不同权**：`host_facts` 是宿主可核的确定事实：treatment 的逐场送达回执（TASK/overlay SHA、字节、`omitted` 列表）存在且与冻结 digest 一致，两臂的**运行期** config digest 相等，逐场字节/轮次记录完整。静态的 `config_sha256` 已由现有计划校验，这里新增的是运行期回执；它证明「改动确实送达」，不证明效果。`directional` 是机制方向期望（例：被拒核数下降），**只报告、不改判**——单 task、三对样本不足以把方向性期望当成假设检验。AHE §4.4.2 自己也只做到 fix-precision 33.7%、regression-precision 11.8%（89 个 task 的口径，regression 约为随机的 2 倍），所以「预期受损」本来就不能当硬门。多 task/case 的扩展计划才点名具体单位；**禁止用 seed 编号冒充逐项预测**。`run_overlay` 的宿主注入路径（§5 阶段 B）不存在时，`host_facts` 无法生成，该计划只能 dry-run，不能标 authorized。 |
| 基线 | spec `control`，plan 的 `base_commit/config_sha256/protocol_version/seeds` | 对照只能是“同条件无此改动”，不能用方便取得的历史 run 替代。 |
| 主指标 | spec `effect_metric_id`，plan `primary_metric` | 一个计划只有一个主指标；其他指标只作护栏或探索性结果。 |
| 方向与最小效应 | `evaluation_rules.direction/min_effect`（`taic/harness_experiment.py:186-195`） | 授权前显式冻结；成本/时间/invalid rate 为 lower，tool success/confirmed speedup 为 higher，并校验指标—方向兼容。 |
| 样本与缺失 | 本批只用 `min_pairs/max_missing_fraction/bootstrap_samples` 与 `resampling_unit=seed` | 冻结后按 seed 成对聚类重采样，完整对不足不得改成非配对；区间只表示单 task 内 between-seed run variability（输入/搜索/执行混合），不是纯测量噪声，也不是 task 级泛化。`confirmatory_min_pairs` 与真实校准另见[确认性效果实验-另纸.md](确认性效果实验-另纸.md)，不进入本批 schema。 |
| 护栏 | spec `guardrail_metrics` 与 rules `guardrail_limits` | 将前者规范为受控 ID，直接对应 `GUARDRAILS`（`taic/harness_experiment.py:21-22`）；旧自由文本只读兼容。`regression_watch` 只增加观察，不缩小护栏集合；未预见回归同样判负。 |
| 成本/效益判据 | 独立 `adoption_terms`，与 evaluation rules 同计划冻结 | 写明成本口径、可接受一次性成本或回本情景；不是第二个效果指标，成本未知则不给正 ROI。 |
| 失效与回滚 | 本批只用 `invalid` 与 `inconclusive` | 绑定、送达、配置、护栏记录或账单不可信记 `invalid`；样本不足、区间跨阈值或未经有效校准记 `inconclusive`。`supports_effect`/`refuted` 及其确认性回滚条件另见[确认性效果实验-另纸.md](确认性效果实验-另纸.md)，本批不生成。 |

本批预测句式固定为：

> 在 `{base_commit, config_sha256, protocol_version, seeds}` 下，相对 `{control}`，仅改变 `{delivery, change_sha256}`，主指标 `{effect_metric_id}` 按事前冻结的方向与 `{min_effect}` 报告；送达事实 `{predicted_observables.host_facts}` 必须成立，机制方向 `{predicted_observables.directional}` 只作事后报告。达到探索性样本要求后，只输出原始配对点、描述性区间和 `inconclusive`，不得输出确认性效果词或自动采纳。任何字段改变都生成新 experiment，不能覆盖原预测。

确认性支持、反驳与校准另见[确认性效果实验-另纸.md](确认性效果实验-另纸.md)，本阶段不实现。`invalid` 表示执行或证据不可信，`inconclusive` 表示效果证据不足；两者都不能写成成功。



## 5. 分阶段方案

**阶段 A：离线预测模板与合同校验**

1. 从现有 spec/plan 生成上述预测，不执行模型/GPU；授权前强制确认主指标、方向、最小效应、样本、区间口径和护栏。
2. 为五类指标固定变换与合法方向，为 `run_overlay/repo_patch` 建立不同 treatment 描述；校验计划字段 `evidence_refs/change_atoms/regression_watch/predicted_observables` 与原始证据、patch 的绑定；旧计划保持只读升级或明确拒绝。
3. 本批只明确 `invalid/inconclusive`；修改预测必须生成新 ID，保留原计划和全部尝试。`supports_effect`/`refuted` 及确认性状态机另纸，不在本批实现。

**阶段 B：dry-run、mock 采集与统计单测**

1. 用临时仓库和至少三个 mock seed 对验证 AB/BA、每臂配置、预算上限及“未授权时 runner 零调用”；dry-run 会物化 worktree，不把它写成只读操作。
2. 新增以 `cells.json` 为输入的多 cell 采集，输出带 `{seed, repetition, pair_index, order, run_digest}` 的 pair 记录；独立 collect 与 execute 后汇总共用一条实现。
3. mock 覆盖缺 cell、乱序、重复 seed、配置/effort 漂移、缺账单、缺主指标、护栏缺失与越界；按 pair ID 连接，禁止位置 zip。
4. 单测覆盖 log ratio/差值、描述性区间、少于三对只报方向、三对与未校准情形只得 `inconclusive`、零宽与极窄区间的伪支持/伪反驳反例、按 seed 聚类的确定性 bootstrap、重复测量不增独立对、多重比较、跨 experiment 后续否定结论和成本 unknown；确认性正对照另纸，不在本批实现。
5. **补上 overlay 的送达路径，否则 `delivery=run_overlay` 计划一律 `invalid`。** 现有 runner 只物化 `treatment.patch`：`build_cells` 在 `change_sha256` 非空时要求 patch 存在且哈希相符（`experiments/harness_runner.py:40-43`），没有任何 overlay 注入入口。规划一条宿主路径：在 treatment run 的首个场次会话开始前，把已绑定 digest 的 overlay 条目写入该 run 的 overlay 文件并冻结 run 级配置，control 同位置留空；两臂的 `harness_auto/harness_shadow/harness_review`、委员会与预算均相同。**不得**用只给 treatment 打开 `--harness-auto` 制造差量——那同时改变了 settle/ review 行为，差量不再单一。该路径不存在时，本阶段的 overlay 类计划只能 dry-run，不能标记 authorized。

**阶段 C：真实校准与确认实验（另纸，本阶段不实施）**

本批不进入阶段 C。真实校准、确认性 `supports_effect`/`refuted`、付费 cell 与 GPU 均按[确认性效果实验-另纸.md](确认性效果实验-另纸.md)另行申请；当前 $40 探索窗口也不能代替那张授权单。



## 6. 风险分级决定证据强度

| 风险级 | 必要证据 | 能否用搜索 A/B 晋升 |
| --- | --- | --- |
| `offline_only` | 实际 patch 仍为附加、只读、宿主检查覆盖的路径；隔离离线回归通过，保留人工 apply 确认。 | 不需要，也不得声称有任务级效果；`tcenv/report.py` 反馈或缺 host check 会升级（`taic/acceptance.py:218-222`）。 |
| `paired_required` | 宿主验收 + 冻结配对实验 + 独立效果复核 + 成本/效益采纳决定。 | 可以，但 `supports_effect` 只是必要条件之一。 |
| `human_gated` | 具名人工、领域专属离线/探针/对抗证据和明确回滚；实际 patch 可升级风险，模型自报不能降级（`taic/acceptance.py:196-222`）。 | A/B 不能替代人工与领域证据；硬件 premise 继续走 probe/premise 路径（`taic/harness_experiment.py:219-222`）。 |

失败证据只决定诊断优先级，不自动证明 harness 根因：单 task/kernel 失败可作辅助证据，跨 task recurrence 也需保留原始 trace 并审查改动 scope。由局部失败推导全局禁用、动作门控或跨模型通用约束时，至少升级为 `human_gated`，补充合法行为与既有成功用例的领域证据；recurrence 不能替代这些证据或降低风险级。

还要理顺现有采纳路由：`apply-code` 默认对所有非 offline patch 要效果验证（`taic/code_agent.py:1047-1051`），而计划只收 `paired_required`；实现时应按 tier 明确分流，不能让 human-gated 借一次 A/B 或兼容绕过获得错误证据含义。`--legacy-unvalidated` 只保留人工兼容事实（`taic/cli.py:1093-1096,1489-1496`）。

**永远不该靠 A/B “证明”的改动**：判分与正确性语义、容差、安全/反作弊规则、信任/权限、协议与预算记账、验收/实验统计规则本身。它们要靠人工审查、golden/property/对抗测试和独立探针；“ treatment 没更差”不能证明新尺子正确，也不能在实验中同时修改尺度和被测对象。

## 7. 归因与混淆控制

1. **同 seed 配对不是同数组下标。** 每对绑 seed/repetition、两臂 run digest 与执行顺序；失败或缺失保留原因，不删除不利对后重算。这里的禁令针对**看过结果之后**按好坏取舍；它与第 10 条事前声明的对称污染剔除不是同一件事。
2. **AB/BA 只减轻时间趋势。** 保留现有交错（`experiments/harness_runner.py:50-52`），同时报告 order；不能把交错当作远端模型漂移已消失的证据。
3. **模型/effort 先冻结再比较。** 现有 config digest 能发现记录内漂移，但同名 provider 模型后端是否变化：未验证；若响应无可靠代次信息，结论必须限于该时间窗。
4. **P1 与 P2 分开归因。** P2 对同 base commit 应用/不应用同一 patch；P1 用新 paired runs 从起点注入/不注入同一 overlay，并核验 TASK 内容摘要。单个 run 的中途前后对比只能算探索；更准确地说，它不是「全局效果的噪声版本」而是**另一个估计量**——RHI §3.1 把自己的目标定义为 trajectory-local 放松，并明说「不是原 μ 下的无偏估计」。同一 run 的前后差只回答「这一场之后是否比这一场之前好」，不能外推成「这条 overlay 相对无 overlay 更好」；后者必须用新的配对 run。
5. **调参与确认、迁移与确认分开。** 产生提案和试跑用过的 seeds 不作为唯一确认集；seed holdout 也不是 task holdout。若要把已支持结论外推到另一 task/model/effort/timeout/step budget，必须新开预测，冻结同一 patch 并在目标条件同窗口对照；源模型调过的 operating point 若未单独交叉设计，结论必须保留 harness × operating-point 耦合，不能声称纯可迁移性。真实迁移实验需人工授权。
6. **多重比较不摘最好结果。** 一个主指标在看结果前冻结；同批多提案、重复查看或换 metric 只报探索性。若要做确认性家族结论，须先实现并冻结 family 与区间校正，否则不得自动 validated。
7. **小样本不升级。** 少于三对只报方向；三对及未达到有效确认门槛时只报原始点、描述性区间和 `inconclusive`。零宽或极窄 bootstrap 区间没有可声称的 nominal 覆盖率，不能产生 `supports_effect`/`refuted` 或采纳。合成 null 只能测实现；真实校准、覆盖率/功效推导与 `confirmatory_min_pairs` 另见[确认性效果实验-另纸.md](确认性效果实验-另纸.md)，本阶段不实施。重复同 seed 不增加独立对数。
8. **bundle 只产生整包归因。** 同一 candidate 的多个 `change_atoms` 共享一次测量，不能把主效应或采纳结果分摊成单 atom 证据；交互/消融须另冻结实验。
9. **评估不确定性与搜索不确定性分开。** 固定 patch 的 paired/bootstrap 只支持该改动的任务级结论；若主张提案机制可重复，须另立协议，从同一起点独立重跑搜索并评价各次冻结产物，不能用同 seed 重复或单次 hill-climb 历史替代。真实搜索重复需人工授权。当前设计每个 seed 只测一次，所以只能估**种子间 run variability**（输入/搜索/执行的混合），不能估同一 cell 内的重复测量噪声；要分离后者得另加重复 cell，成本翻倍，属单独的校准决定。
10. **污染必须按臂对称。** 「污染」指任一臂能读到本实验的评分材料或一个已证明更快的同题产物（期望输出/测试、缓存报告、历史 run 的胜者、已入库的优化核）。两臂共用的 seed 本身不是差分污染，但由共享 seed 或历史 run 派生出的评分材料仍需单独判污染，不能整类豁免。出现污染时该实例作废；Ouroboros §6 的做法是**任一臂触到参考材料即对称剔除该实例**，并报告「剔除后配对比较的解释反转」（原文没有给数值符号翻转）。
    **本仓库额外加的三条是本地设计，不是论文规则**：规则在看结果前写定、同时报未剔除/剔除后两个结果、两者结论相反时记 `invalid`。加它们的理由是防止事后挑拣；不能把它们写成 Ouroboros 的做法。
11. **只在被选中的提案上测到的效果不是搜索本身的证据。** 采纳样本是选择后的子集；报告必须同时给出同一批次里未被选中/被反驳的提案数，否则不得从 `supports_effect` 推断“harness 搜索有效”。两篇的发布规则都带**best-on-evolve 选择偏差**：AHE 的 Algorithm 1 第 3 行评估的是 `H(t−1)`、第 14 行却可能把 `Ht` 赋给 `Hbest`，并没有非负保证；One Recipe App. A.4 的非回归门只保证 **evolution split** 上「不过 H0 就退回 H0」，其 `Dtest` 是 disjoint、最后才评。因此这些增益不能独立确认效果或泛化，但仍可说明分布内优化发生了——不能反过来说「搜索无效」。

## 8. 采纳、回滚与成本结论

- 效果复核继续只回答“预测是否成立”；扩展 `adoption_review` 记录 adopt/defer/reject/rollback、证据 digest、成本完整性、具名人和理由，不把成本判断塞进 `validated` 布尔值。
- `paired_required` 的 adopt 候选必须同时满足：同 spec/patch 的 supports_effect、无实测护栏回归、成本非 unknown、预声明效益条件成立、独立效果与人工采纳复核。
- 主效应阈值与护栏是非补偿性门槛：结构新颖、atom 更少或 token/美元节省不能补过未达标主效应；只有事先将成本设为 `primary_metric` 时才判定成本效果。
- `refuted` 或伤害性护栏越界：未采纳则 reject；已启用则按 spec rollback。`inconclusive/invalid` 默认 defer，不得把“未证明有害”写成有效，也不自动抹掉历史证据。
- 后续同 scope 的反证必须阻断旧 true 的自动采纳；不同 config/scope 不能简单互相覆盖，应由新采纳决定明确适用范围。回滚追加事件并保留原 prediction/result。
- P1 用现有人工 disable 通道，重新启用只能新开 run（`taic/harness.py:1006-1019`）；P2 由人反向应用已绑定 patch，不自动 commit。

成本至少分列：一次性 `C_adopt = C_review + C_implementation + C_experiment`（其中 `C_experiment` 是支撑该决定的所有预注册确认实验实际账单，含双臂、失败/无效 cell 和已执行的独立重复，不只记最后成功实验），以及采用后的每 run 成本差；总成本与其组成不得再相加，同一 review 事件不得在一次性和经常性成本中重复计数。仅当收益与成本同口径、全已知且持续美元节省为正时，才报告 `break_even_runs = C_adopt / 每 run 节省`，并给出不确定性/情景范围；否则给成本—效果双轴结论，不把时间或 speedup 强行折成美元。效果成立但不值得其成本应为 `reject_cost`，不是 supports_effect 的反面。

## 9. 验收标准（全部先离线）

1. 合同：预测 round-trip/digest 稳定；缺指标、方向不兼容、事后改阈值或区间口径、失败提案缺 trace/task IDs、atom-patch 未绑定、无护栏或旧 schema 不明升级均拒绝；`paired_required` 另要求 `predicted_observables` 两类齐全，`offline_only` 不要求（它不跑实验）。
2. 采集：乱序仍按 pair ID 配对；缺 cell/账单/注入证明不会变成有利零值；多 cell 独立采集与 execute 汇总一致；污染剔除只能按事前冻结的对称规则执行，且 raw/filtered 两个结果同时存在。
3. 统计：两对只能方向；3 对及未校准情形只报描述性区间 + `inconclusive`，不得 `supports_effect`/`refuted` 或采纳。规划反例只覆盖 `[1,1,1]`、`[1,1,0.9]` 的伪支持与 `[0.2,0.2,0.2]` 的伪反驳；合成 null 不得解锁确认，缺覆盖率或功效/MDE 的校准记录无效。真实确认正对照另见[确认性效果实验-另纸.md](确认性效果实验-另纸.md)，不属本批验收。重复 seed 不增独立样本，`host_facts` 不成立的对不得进入统计，`directional` 与主区间相反时必须写「机制未获支持」。
4. 采纳：旧 experiment 的 true 不能盖过同 scope 新反证；成本 unknown 不报正 ROI；新颖性/节省不补过主效应或护栏；reject/rollback 可追踪且不会自动 apply/commit。
5. 范围与账本：bundle 不拆记单 atom 效果；固定 patch 的 supports 不升级为提案机制、CMP 或迁移结论；跨 scope 无新 experiment 不得 adopt；失败/无效及已执行重复的账单缺失均为 cost unknown。
6. 测试放 `tests/shared/test_acceptance.py`、`test_harness_experiment.py`、`test_harness_runner.py`、`test_adoption.py`；共用 mock helper 才放 `tests/support/`，遵守 `tests/README.md:71-116`。这些都是 backend 无关逻辑，不加 GPU/LLM marker。
7. 送达：`delivery=run_overlay` 的 dry-run 计划必须能列出 treatment 预置 overlay 的 digest、写入位置与 control 的阴性断言；只靠 `harness_auto` 差异或只靠 run 内自行 settle 的 plan 被拒。注入路径实现前，overlay 类 `execute-experiment` 仍只能 dry-run。

```bash
uv run pytest -q -n 0 -m 'not gpu and not llm and not slow' \
  tests/shared/test_acceptance.py tests/shared/test_harness_experiment.py \
  tests/shared/test_harness_runner.py tests/shared/test_adoption.py
uv run pytest -q -m 'not gpu and not llm and not slow'
```

真实付费/GPU 冒烟测试不属于默认验收；若以后增加，必须同时标 `llm`、`gpu`（必要时 `slow`），单独「需人工授权」运行。本文编写时未执行上述测试、付费实验或 GPU。

## 10. 风险、边界与方法论参考

**主要风险**：小样本 bootstrap 给人虚假确定性；远端模型漂移破坏配置同一性；多提案反复试验造成选择性报告；overlay 未送达却被记为 treatment；把搜索成本、确认加速和人工时间混成一个 ROI。对应防线分别是保守不确定结论、交错与适用范围、冻结 family、注入摘要和分栏账本。

**边界**：plan-01 拥有判分/容差/抗作弊及安全护栏的产生逻辑，本计划只消费结果；plan-05 拥有 review 质量与校准，本计划只把 useful 当待证伪假设；plan-08 拥有看板、聚合展示、成本告警和试点的分段授权，本计划只定义判定与成本语义。plan-08 §7 的 S0/H0 送达冒烟是 `predicted_observables.host_facts` 的操作化执行，不是第二套送达语义；两边冲突时以本计划的字段定义为准，plan-08 只负责授权与执行。证据签名、信任授权、知识生命周期和文档治理仍各归其 plan。

**论文只作方法参考，不作收益承诺**：AHE（`docs/可信harness/papers/2604.25850_agentic-harness-engineering.pdf`）提供 decision observability；其 10 轮 Terminal-Bench 2 的 69.7%→77.0% 是原论文场景结果（`docs/可信harness/papers/导读.md`），不是本仓库预测。AI Agents That Matter（`docs/可信harness/papers/2407.01502_ai-agents-that-matter.pdf`）提醒联合看成本—效果并避免无 holdout 过拟合（`docs/可信harness/papers/导读.md`）。HAL（`docs/可信harness/papers/2510.11977_holistic-agent-leaderboard.pdf`）提供标准化 harness 与大规模 rollout 的方法参照；其规模不是本计划样本目标（`docs/可信harness/papers/导读.md`）。

## 附录：文献与变更索引

只作查阅，不再逐轮重述正文；论文数字均未在本仓库复现。

| 日期 | 影响位置 | 证据文件 |
| --- | --- | --- |
| 2026-10-25 | 文献对应的正文规则与边界 | `2604.25850_agentic-harness-engineering.pdf`、`2609.24972_rrsi-regularized-harness-rsi.pdf`、`2608.10178_one-recipe-many-harnesses.pdf`、`2608.08311_ouroboros-reviewed-core-evolution.pdf`、`2504.15228_self-improving-coding-agent.pdf`；其余见 `papers/导读.md` |
| 2026-10-26 | 当前决定、阶段、验收与跨 plan 归属 | `审核意见-plan-01至09.md` |
