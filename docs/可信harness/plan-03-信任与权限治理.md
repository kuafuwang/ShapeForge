# Plan 03：信任与权限治理——身份、单调收敛与授权生命周期

本文只规划信任身份、权限判定和授权运营，不改变当前开关，不调用付费模型，不运行 GPU。核心原则是：**权限只能被自动收窄；任何扩大或无法比较的变化都必须由具名人批准，unknown 一律不自动放行。**

## 当前决定

- **本批只做**：只读 `--explain`，逐维列出 status/target/task/provider/model/effort/scope/expiry/protocol/prompt 的实际拒因，并覆盖四套身份字符串与三条 conv2d-only grant。
- **输入**：`taic.toml`、`targets/<id>/trust.toml` 与 audit、`AGENTS.md`、当前 trust/committee 实现。
- **本批不做**：别名表、v2 generation/policy digest、`control_plane_change` 接线、自动恢复或任何扩权；未批准前当前身份继续拒绝。是否把知识库移出提示指纹、是否把授权从 task 放宽到 target，都是另需具名拍板的扩大，不能顺手做。
- **完成产物**：不刷新索引、不改 registry 的诊断报告与离线 fixture；§⑥ 是信任状态转移的唯一正文。
- **失败去向**：先读 `logs/pytest/triage/triage_summary.md` 及 handoff；`--explain` 与新测试未实现前不得写成现有命令。

## 当前代码核对（2026-10-27；路径清单与指纹句于 2026-11-03 按已拍板决定改正。其余句子本轮未重读，供审阅，未实施）

- `--explain` 仍是规划中的只读命令，不得写成已存在。先做 status/target/task/provider/model/effort/scope/expiry/protocol/prompt 的逐维拒因，不自动修复身份。
- 四套身份字符串与三条 conv2d-only grant 只用于诊断。`kimi-code` 与 `moonshot`、两个 model 字符串是否同部署仍未验证；不得用名称相似度授权。
- `tasks/` 已被 `code_agent.forbidden_path`（`taic/code_agent.py:161`）禁止，因此 P2 评分材料补丁不重复添加它。`SECURITY_CRITICAL`（`:43-87`）按评分材料角色还要补 `tcenv/taskcontract/`、`tcenv/opspec/`、`tcenv/task.py`、`tcenv/distributions.py`；`report.py` / `taic/parse.py` / `taic/evidence.py` 按角色复核，不一刀切。具体补丁归 plan-06，本 plan 不改这些路径。
- Humanize、CANNBot-CAKE、AKG 等只能帮助解释边界，不能替代 `trust.check`，也不能让 `skill_version`、reviewer 或模型等级成为新的自动授权依据。

## ① 角度与问题

信任不是模型声望，而是一份可过期、可暂停、可撤销、可审计的能力委托（`docs/可信harness/架构与理论.md:228-256`）。本计划回答三个问题：

- **身份**：配置中的 provider/model 名称变化时，怎样既不让 P1 无声失效，也不靠模糊别名误授权。
- **策略**：怎样把 P0/P1/P2、never-auto 和 run 控制写成机器可判定的单调权限规则，而不只依赖约定。
- **生命周期**：grant、重签、过期、自动暂停、人工恢复各需要什么证据，如何减少反复重签和误暂停的运营负担。

不变式是：信任只是自动生效的必要条件，不替代提案结构、证据、双来源或独立 review；P2 的 propose 也不包含 apply/commit/push。

## ② 目标 / 非目标

### 目标

- 建立保留原始字符串、解析到规范身份、带策略版本的身份模型，并让 grant、check、统计和 quorum 使用同一套解析结果。
- 用允许集包含关系判定「收窄 / 扩大 / 不可比较」，为每次自动决定输出稳定拒绝原因和策略摘要。
- 将 registry 的持久状态与某时刻的有效资格分开，补齐证据、到期、恢复和审计字段。
- 先做只读诊断与迁移 dry-run，再做纯策略测试，最后才接线任何行为变化。

### 非目标

- 不证明两个远端模型实际同权重、同路由或同行为；不自动发现别名，不根据名称相似度授权。
- 不设计证据签名、review 质量、注入/沙箱或 harness 效果实验；这些分别属于 plan-04、05、06、02。
- 不默认调整 8 条 / 2 run / 25% 的暂停阈值，不默认开启 `harness_auto`，不新增自动 apply、commit 或 push。

## ③ 现状证据

- 当前配置是 `harness_auto=false`、`harness_shadow=true`、`harness_review=false`（`taic.toml` 的 `[search]`）；因此默认并未开启自动 settle，也不自动调用 reviewer。下面的身份问题是启用 P1 前的真实缺口，不宣称已有线上损失。
- **授权粒度也是独立决定。** 三条 grant 都只覆盖 `conv2d`，所以 P1 在其他 task 上按现状永远关闭；应明确写成「P1 目前只是 conv2d 实验」，不能把 task 放宽到 target 当作修复名称的副作用。放宽属于扩大，必须另开具名批准。
- **配置与 grant 仍是不同字面量**：`taic.toml` 的 second 与 `AGENTS.md` 的 Search 节都是 `kimi-code` / `k3.1-pro`。grant 仍是另一套：`moonshot/kimi-k3.1-pro` 与 `moonshot/kimi-k3`，另有 `deepseek/deepseek-flash`；三条 grant 的 task 全是 `conv2d`（`targets/apple-m4-max/trust.toml`）。`trust.is_trusted('apple-m4-max', task='conv2d', provider='kimi-code', model='k3.1-pro', effort='high', scope='overlay:tip')` 对不上任何 active grant。`kimi-code` 与 `moonshot`、两个 model 字符串是否同一部署仍是**未验证**；未批准前不做任何别名。
- **拒因不止一个维度，而且顺序有优先级。** `check` 依次比 status → target → **task** → provider/model/effort → scope → expiry → protocol → prompt（`taic/trust.py:569-592`）；在 conv2d 上复现得到 `trust is for another provider/model/effort`，而在任何非 conv2d 任务（例如 paper 预设的 `flash_attention`）上，第一条不匹配的是 `trust is for another task`，即使 provider 已改成 `moonshot` 也依旧 deny。原因是三条 grant 的 `task` 都是 `conv2d`（`targets/apple-m4-max/trust.toml:6,23,40`）。所以「改名就能修好」只对 conv2d 成立；其他任务需要的是新增 task 维度的 grant，那是扩大，不是别名能替代的事。
- 这不是越权，而是字面量比较产生的 fail-closed 静默不可用；`check` 精确比较 provider/model/effort（`taic/trust.py:576-577`），`is_trusted` 最后又丢掉逐条原因，只返回 `no active grant`（`taic/trust.py:595-625`）。**未验证**：`kimi-code` 与 `moonshot`、两个 model 字符串是否代表同一部署，不能据名字推断。
- 委员会配置只做非空与 `strip`，随后把原字符串放进 `Member` 和 `run_config`（`taic/committees.py:73-86,198-202,237`），没有规范身份或别名层。
- 三条现有 grant 的 scopes 都是 `overlay:tip + overlay:premise`、protocol 3、`prompt_sha1=aa71abd3…`，且复核时该聚合指纹仍等于 `trust.current_prompt_sha1()`；同时 `evidence_run=""`、`expires_at=0.0`（`targets/apple-m4-max/trust.toml:4-52`）。所以文件里的 `status=active` 不等于已具备证据、未过期且对当前身份有效。
- `prompt_sha1` 字段名有历史误导：实现实际用 SHA-256 对标签、提示函数源码、模板常量、全局/目标/backend 文档逐段哈希（`taic/trust.py:77-99,172-199`）；同模块的非提示代码已不触发重签。**其中包含 `experience.md`、`device.md`、后端 skills 与 `taic-tools/SKILL.md`**，所以一次经验晋升或 ID 迁移也会使现有 grant 变成 stale prompt。旧 grant 只保存聚合值，无法事后可靠分解是哪一段变化；是否把知识库移出指纹属于扩权，须单独具名决定。
- scope 是封闭集合，但 `propose:skill/propose:task_wording` 仅可登记，没有自动生效路径（`taic/trust.py:30`；`taic/harness.py:303-305`）；scope、状态、过期、协议、prompt 任一不合都拒绝（`taic/trust.py:569-592`）。
- `harness_control.json` 的 API 拒绝 `disabled=false`，只允许关闭（`taic/harness.py:1006-1027`）。但通用 JSON 读取把缺失与损坏都变成 `None`（`taic/control.py:93-105`），而 `_overlay_disabled` 将其视为未禁用（`taic/harness.py:996-1003`）；后续应区分「本无控制文件」与「文件存在但不可解析」。
- `sync_trust_status` 在至少 8 条 reviewed claims、至少 2 个 run、confirmed-refuted 率至少 25% 时自动 suspended（`taic/harness.py:1030-1043,1054-1085`）。当前 201 条人工 claim 中 `status=rejected` 与 `verification=refuted` 都是 0，因此自动暂停的输入目前为零，不能把该阀门当成已生效的安全网，也不先调阈值。SQL 只按 target/task/provider/model 聚合，未按 effort、prompt、protocol 或 grant 代次切片，`prompt_sha1` 参数也未参与；索引未追平会滞后（`docs/可信harness/实现交接.md:138`）。
- CLI 的 grant 允许默认 `--by=cli`、空 reason、空 evidence-run，status 只打印 registry，metrics 会先刷新派生索引（`taic/cli.py:858-899,1363-1389`）。严格只读报告不能直接复用会写索引的 metrics 路径。
- 现有测试已覆盖 scope/prompt/protocol/expiry、精确撤销和指纹组成（`tests/shared/test_trust.py:13-278`），但没有覆盖别名迁移、策略差集、grant 代次和恢复流程。

## ③.1 分级委派与审核指纹（2026-10-26 讨论结论）

- **模型能力不是授权。** 若未来按 role 分级，只能是 `role + provider/model/version/effort + target/task + scope + expiry` 的受限委派；新增 role scope 或部署 version 维度都是具名扩权。当前 `SCOPES` 没有 reviewer role，grant 没有部署 version，`overlay:tip` 不能改名当分类/评审授权。
- **审核控制面也要绑版本。** `overlay_text` 与 `build_review_prompt` 当前都不在 `current_prompt_sha1`（组成见 `taic/trust.py:77-95`）。2026-11-02 已决定把 `overlay_text` 函数及其固定标题纳入 grant 指纹；每场 overlay 行不纳入。`build_review_prompt` 不在该扩面表内，本次不加。纳入仍是 `control_plane_change`，须 P2 人审和允许集差分，不在本批实现，也不做 old→new hash 别名。获批部署后旧 grant 变 stale 才是收窄效果，后续重签仍须另行人审。具体边界以 §③.1.1 为准。

### ③.1.1 指纹扩面决定（2026-11-02，plan 层面，未实施）

人审已拍板：**指纹扩到“决定送达文本的代码与模板”，而不是每场实际内容。** 纳入 `peer_tips_for` 的选取/过滤逻辑与 `_PEER_HEADER`、`overlay_text` 及其固定标题文案、`workspace_prompt.render_remote_contract`、`code_agent` 的 IMPLEMENT 提示模板、`stage_guidance` 的拷贝/过滤策略、`TASK_MAX_BYTES` 与各块 cap 及注入顺序。

**每场变化的内容不进 grant 指纹**：实际选中的 peer_tips、生效的 overlay 行、远程 TASK 字节、IMPLEMENT.md 正文、各块实际字节与 `omitted`，全部进每场 `prompt_surface_digest` 与 plan-08 的 delivery receipt。把动态内容哈希进 grant 会让信任每场 stale，这不是安全加强而是系统不可用。

`experience.md`/`device.md`/技能文档**继续留在指纹内**；是否移出仍是独立扩权决定，本次不碰。已不在 kernel_once 发送的 chat 提示面，先逐个核对哪些 `prompts.*` 函数仍被使用再决定剔除，不误删。

指纹变化、b413 经验条目删除、渲染措辞修正**合并为同一次提示面变更，一次性重签**（当前基线 `aa71abd3…`）；仍按上面的 `control_plane_change` 流程给前后允许集差分，不做 old→new hash 别名，不用 LLM 判断“语义没变”。完整边界表见[代码现状-2026-11-02.md](代码现状-2026-11-02.md) §9.1。
- **知识库边界不变。** 把 `experience.md`/`device.md` 移出指纹会扩大有效授权，须具名批准；审核记忆若被允许进入生效路径，也必须绑定内容快照，不能用“模型记得”替代证据。分级与常驻审核的实施讨论归 plan-05。

## ④ 身份模型方案：规范化、别名与迁移（v2 冻结）

定义原始身份 `R=(raw_provider, raw_model, raw_effort, raw_route)`，规范身份 `C=(provider_id, model_id, effort_id, route_id)`；`raw_route` 仅保存宿主配置的非秘密 route/deployment/profile 标识，`unspecified` 不是通配。v2 授权主键建议为 `K=(target_id, task, C)`。prompt、protocol、scopes、expiry 仍是 K 上的约束，不能被别名消解；每次重签另有单调递增的 `grant_generation`，避免把不同授权期的统计混在一起。

- **原文可追**：run、proposal、grant、audit 都保留 R；C、identity policy version/digest 是宿主解析结果，不采纳模型自报的 canonical id。
- **身份解析只碰身份，不碰 task。** 别名表、effort/route 归一与大小写处理只能影响 `provider/model/effort/route` 四元组；`target_id` 与 `task` 永远精确匹配，不允许别名、前缀、同义任务或「同 kernel 名」把它们拉宽。给已有主体新增 task 维度的 grant、或把 grant 的 task 改名去匹配当前 run，都属于扩大，需要独立具名批准。
- **最小规范化**：保留现有去首尾空白，拒绝空串、控制字符和非法结构；不默认做大小写、Unicode、标点、版本后缀或模糊归一。effort/route 的空值只表示 unspecified，不等于 high/off 或任何默认路由；未知 effort/route 不是通配符。
- **精确别名**：人工维护版本化映射，规则以完整 R 为单位映射到唯一 C，禁止 `*`、前缀/正则、自动传递闭包和独立 provider/model 别名自由组合，避免造出未批准的交叉组合。
- **别名证据**：每条规则记录 `approved_by`、reason、evidence_run、适用上下文和策略摘要。它只声明本仓库授权身份等价，不证明远端实现相同；当前 Kimi 案例必须先由人核实，再显式批准，不能由迁移器自动补上。
- **全链路一致**：trust check、指标、暂停统计及 `distinct_models` 都使用同一份 C 和策略版本，历史行不重写 R。别名合并可能减少独立作者数、同时扩大某 grant 的匹配范围，必须评估整个 settle 允许集，而非只看 `trust.check`。
- **v2 registry**：保存 K、R 示例、`identity_policy`、generation、带算法标签的 prompt manifest；稳定 id 可由 K 派生，generation 区分 re-grant。旧 id 与新 id 的谱系写入 append-only audit。
- **迁移默认 dry-run**：逐条展示旧/新身份、命中的别名、有效允许集差集和缺失证据；无别名时保持精确匹配。旧 v1 行只按旧字面量语义读取，不因安装别名表而自动变 active。
- **人工迁移**：任何新增匹配都是扩大，须逐 grant 具名批准并追加审计；不得伪造历史空 `evidence_run`，不得顺手延长 `expires_at`。冲突、重复目标、未知版本或不可比较映射一律拒绝。

这样既能把「换名」变成可见的待批准事项，又不会让「看起来相似」成为隐式授权；无别名或解析失败时回退到 P0/人工队列，而不是无信任 active。

## ⑤ 权限策略的形式化：收窄自动、扩大具名批准

参考 Progent，把宿主动作看成有限能力集合，把请求记为 `x=(target,task,run,C,effort,prompt,protocol,scope,capability)`，策略 P 的允许集为 `A(P)={x | allow(P,x)}`。P0/P1/P2 不是必须全序继承的称号，而是不同能力集；never-auto 集合 N 对所有自动策略恒有 `A(P)∩N=∅`。

| 表面 | 自动能力上界 | 明确不包含 |
| --- | --- | --- |
| P0 observe | 写本场 FEEDBACK、进入人工队列 | overlay、repo patch 生效 |
| P1 shadow | 记录满足既有门槛的候选 | 因 shadow 本身获得注入权 |
| P1 active | 在有效 scope/grant 及其他证据门全过时，仅注入本 run 后续场次 | 跨 run/target/task、仓库写入 |
| P2 propose | 生成 skill/task_wording patch 候选 | apply、commit、push、自批 |
| 人工 P2 | 具名人审 diff/测试后 apply/commit | 把人的批准转授给模型 |
| never-auto | 无自动能力 | 合同、ABI、容差、预算、价格、acceptance、删除 audit、git push，以及自动修改 provider/model/effort/route、identity/approval 规则，永远不因 grant 解锁 |

`harness_shadow=true` 也不能简化成「绝不会注入」：独立 useful review 的 `reviewed_active` 分支在 quorum 的 shadow 分支之外（`taic/harness.py:904-905,936-993`）。策略测试必须把 `harness_auto/harness_shadow/harness_review` 三个开关及这条路径一起建模。

- 对更新 `P→P'`，仅当能证明 `A(P')⊆A(P)` 且 never-auto 不变式仍成立，才属于自动可接受的收窄；扩大、不可比较、解析失败都进入「需具名批准或拒绝」。
- 收窄例子：active→suspended/revoked、缩短到期、删除 scope、禁用 overlay、auto/review 由开转关、quorum 由 active 转 shadow。 revoke 仍可要求人触发，但绝不会因触发者是自动系统而扩权。
- 扩大例子：新 grant（含为已有主体新增 `task`、`target` 或 scope 的新 grant）、恢复 active、增加 scope、延长到期、重绑新 prompt/protocol 后恢复资格、新增别名匹配、auto/review 由关转开、shadow 转 active、重新启用已 disable 的 run。**`revoked→suspended` 不在任何允许集内**（不同于上列「需具名批准」的扩大），run 内出现即拒绝；重建资格只能新开 grant/新 generation。
- **判定器不能自证收窄**：自动收窄只适用于同一版本判定器下的授权数据；身份解析、grant generation、暂停/恢复或批准规则的更新必须标记 `control_plane_change=true`，自动分类为不可比较并交具名人批准，不能凭 `A(P')⊆A(P)` 自动生效。
- review 的 useful 只是证据门，不是批准人；code agent 是实现者也不是批准人。模型或 code agent 提交的策略/检查器改动即使看似收窄，也仍走 P2，不能自我应用。
- **批准独立且绑定快照**：audit 分记 `proposer_principal`、`reviewer_refs`、`approved_by`、`approval_channel`；批准者须为独立于 proposer 和 LLM review panel 的具名宿主操作员，模型/quorum 只能作 `reviewer_refs`，`proposer_principal=approved_by` 或 principal/channel 不可验证均拒绝扩大。
- **以下代次/策略字段属 v2（未实现）**：批准绑定 `base_generation`、before/after `policy_digest`、`proposal_digest` 和 `policy_test_ref`；写入前重算，任一不符即 stale，拒绝 grant/renew/recover/alias 迁移（v1 不要求这些字段，见 plan-06 §5.10）。`policy_test_ref` 指向的离线权限差集与 fail-closed 测试必须存在且通过，行为分数只能补充，不能替代。
- 这里不需要 SMT：scope、capability、状态和别名规则是有限的，身份是精确元组，其余是哈希/版本相等及时间比较；用结构规则和有限 fixtures 的包含检查即可。未来若加入通配、正则或复合符号条件，应另做求解器证明，不能偷偷放宽本判定器。
- 每个决定保留 `allow/deny + reason + grant_id/generation + policy_digest`；诊断可以新增解释，但不能改变同输入下的布尔结果。

## ⑥ 授权生命周期与运营

**本节是 trust/control 状态转移的唯一正文。** plan-06 与[评分可信计算基-另纸.md](评分可信计算基-另纸.md)只引用本节并记录偏离时的处置，不另抄 `suspend`/`suspended` 映射、字段表或恢复规则。

持久状态只表达 `active/suspended/revoked`；`expired/stale_prompt/stale_protocol/identity_unknown` 是每次 check 计算的有效资格，不靠后台重写 registry 假装状态迁移。

| 事件 | 触发与证据要求 | 权限结果 |
| --- | --- | --- |
| grant | 具名 `--by`、非空 reason、非空 evidence_run；核对本地 run 的 target/task/原始或规范身份，绑定当前 prompt/protocol、最小 scopes 和明确到期 | 人批准的新 generation；默认不继承别的主体/scope |
| 重签/续期 | 到期前、prompt/protocol/identity policy 变化后逐条展示差异和新证据；不得只传一个新哈希覆盖旧行 | 属扩大，人批准后才 active，audit 保存 previous |
| 过期 | `now>=expires_at`；新 grant 要求有限未来时间，旧 `0.0` 标为 legacy_no_expiry | 自动拒绝，不删除、不自动续期、不伪造 revoked |
| 自动暂停 | 先保留现阈值。决定记录只绑**该次决策实际使用的 index as-of/快照摘要**；n/runs/bad/rate **由宿主在该快照上重算**，不要求写进决定行（v1）。**当前实现**的决定行有 9 个键：`schema_version`/`event="suspend"`/`t`/`target_id`/`task`/`provider`/`model`/`grant_id`/`reason`（`taic/harness.py:1086-1090`），不记录快照，也没有调用回执；**跨记录映射**：该 `event="suspend"` 对应审计行的 `event="suspended"`（`set_status` 写 `event=status`，`taic/trust.py:506-510`），两者是同一逻辑事件，不得要求字面相等；将来统一改名必须带 schema 迁移与旧记录兼容读法。effort/prompt_sha1/protocol_version 不在决定行，但可由 `grant_id` 对应的审计 `grant` 对象导出（该对象写时冻结、前缀不可改写）；该审计行不可用时只能记 `host_unverified`。v1 待新增的只有快照字段，v2 才加 generation/policy_digest（见 §⑤ 与 §⑥ 的代次说明）。外文档不得把待实现项当现状。 | 仅 `active→suspended`；不得自动恢复或撤销（`taic/harness.py:1052-1076` 只筛 active 再调 `trust.suspend`） |
| 人工暂停/撤销 | 具名人+reason + 宿主观测到的调用回执或具名批准记录（`--by` 是自报字符串，不能单独当身份；CLI 无 run_dir、不写决策行）；别名失实、越 scope、证据失效等可提前 revoke | 对 C 的全部已批准别名生效，不能靠改名逃逸；`active→suspended/revoked`、`suspended→revoked` 与重复 suspend/revoke 都是合法收窄（`set_status` 不检查原状态，`taic/trust.py:469-536`）；**`revoked→suspended` 不在任何允许集内**：run 内出现即 `invalid`，不接受普通具名批准覆盖；重建资格只能新开 grant（新 generation）并重新冻结 |
| 人工恢复 | 查看暂停证据，关联暂停事件；提供其后新证据，或具名说明为何旧证据仍适用 | 新 generation 必须显式 grant；不得因比率回落或换 prompt 自动复活 |

**代次写入必须比较当前值（v2 计划，未实现；v1 无 generation，不得要求）。** 自动暂停携带 `expected_generation`，扩大批准携带 `base_generation`；不匹配只追加 `stale_event`，不改变当前 grant。旧 generation 的延迟自动暂停不得暂停新 grant；人工 revoke 不受此豁免，仍对 C 的全部代次生效。

**恢复与止损失效分离。** 索引、统计或模型服务不可用可以阻断 recover/renew，但不得阻断基于 registry 的 suspend/revoke，全局 disable 也必须仍可用；这些收窄操作不调用付费模型或 GPU。恢复先输出 dry-run 差异，再经独立批准创建新 generation，不能因旧批准或比率回落直接 active。

**prompt 降噪不等于放宽绑定。** 新 grant 保存排序后的 `label→sha256` parts manifest 和聚合算法标签，报告只列实际变化的 prompt 函数、模板或文档；旧聚合值无法分解时标 `attribution=unavailable`。不做空白归一化、不用 LLM 判断「语义没变」、不自动建立 old→new hash 别名；人看完 diff 后仍按重签处理。现有「排除非提示代码」继续保留（`taic/trust.py:172-199`）。

**阈值先评估再谈调整。** 只读回放应按 legacy 全历史口径与 grant generation 口径并列报告会暂停的条目、重签后是否立即再触发、索引覆盖和缺失字段；没有这些数据时，阈值合适性均为**未验证**，不做自动调参。缩小历史统计窗口可能减少暂停、扩大有效权限，不能冒充 fail-closed 改动。

运营上，`trust status --explain` 应集中列出 stored/effective 差异、逐 grant 拒绝原因、证据缺口和按时间排序的到期项；`renew` 只可预填当前指纹和旧 scopes，不能一键批量复活所有 target/task/model。

## ⑦ 分阶段步骤

1. **本批：只读诊断。** 实现 `--explain` 与逐维拒因报告，覆盖四套身份字符串、三条 conv2d-only grant、task 与身份不匹配的先后顺序；不写 registry、不刷新索引、不启用别名。
2. **本批：离线 fixture。** 固定当前 deny、各维度拒因与不变布尔结果；`--explain` 和新测试实现后才可运行。
3. **后续另纸，本阶段不实现。** 别名、v2 registry/generation/policy digest、暂停/恢复的接线与任何扩权分别申请；§⑥ 的状态表只作以后接线的唯一依据。

## ⑧ 验收标准（离线测试 / 命令）

本批只验收只读诊断：每个 grant 的 status/target/task/identity/scope/expiry/protocol/prompt 拒因能一次列全，不只报 `no active grant`；conv2d 与 flash_attention 的拒因顺序正确；四套字符串未经批准仍全部按字面量处理，布尔 `is_trusted` 结果不变；报告不改 registry、run 或索引。失败先读 `logs/pytest/triage/triage_summary.md` 及 handoff。

当前已有命令与后续命令分开：

```bash
# 当前可运行（只读，不刷索引）
uv run python -m taic.cli trust audit apple-m4-max
uv run pytest -q -n 0 tests/shared/test_trust.py
uv run pytest -q -n 0 tests/shared/test_harness.py tests/shared/test_harness_workflow.py

# 本批实现后才可运行：--explain 与新增诊断测试
uv run python -m taic.cli trust status apple-m4-max --explain
uv run pytest -q -n 0 tests/shared/test_trust_explain.py
```

别名、v2 `grant_generation`/`policy_digest`/`base_generation`、自动暂停/恢复接线和 `test_trust_policy.py` 均属后续，**不是本批验收项**。

## ⑨ 风险与回滚

- **fail-open 绝对禁止。** 每批行为变化都附旧/新策略同一 fixture 全集的差集；出现任何未具名批准的 `after_allow && !before_allow` 即失败，而不是只展示几个 deny 示例。
- 别名的有意扩权不能宣称 fail-closed：它只能作为单独、具名、可审计的扩大变更存在，且即使获批也不能突破 never-auto。别名过窄造成可用性下降时，走 P0/人工审查，不以自动放宽来止损。
- 自动暂停口径变窄、旧 schema 兼容、代码回滚都可能恢复权限；回滚同样先做策略差集，不能默认安全。操作回滚优先 disable P1 / suspend 相关 grant，再具名处理代码或 registry。
- 不删除或截断 audit，不自动恢复旧 trust.toml 快照，不让新旧 writer 混写；部署 v2 前必须有版本拒绝与回滚读法，旧读取器不能解释不了新字段却继续放行。
- 主要残余风险是远端身份等价无法由字符串证明、历史证据归属不完整，以及 fail-closed 带来的重签负担；分别用显式批准、标 unknown 和只读待办报告处理，不用隐式豁免解决。

## ⑩ 与其他 plan 的边界、参考论文

- **plan-02**：负责权限/overlay 是否改善搜索的对照实验；本计划只决定谁可以做什么，不把授权指标当效果结论。
- **plan-04**：负责 evidence/run/prompt/audit 的签名与密码学 provenance；本计划只消费这些绑定并规定授权事件字段，parts manifest 也不是签名。
- **plan-05**：负责 claim/review/quorum 的判定质量；本计划只规定已审统计如何触发暂停与人工恢复，不重新裁判 claim。
- **plan-06**：负责运行时、工具执行和注入防线；本计划的 never-auto 是授权上界，宿主动作分类失败仍须 deny。plan-06 与[评分可信计算基-另纸.md](评分可信计算基-另纸.md)只引用 §⑥ 的状态转移表，不再复制 suspend/suspended 映射、字段表或恢复规则；它们只记录偏离时的处置，以及合法收窄与非法扩权对内核发布和 harness 证据的不同影响。
- **Progent（arXiv:2504.11703）**：工具名/参数上的符号权限、确定性检查、SMT 区分 narrowing 与 expansion，即 monotonic confinement；本地 PDF 为 `docs/可信harness/papers/2504.11703_progent.pdf`，摘要见 `docs/可信harness/papers/导读.md`。本计划借其判定原则，但以有限封闭策略代替求解器。
- **Toward Secure LLM Agents（arXiv:2606.10749）**：247 篇综述提出显式 trust boundary、principled privilege control、provenance-aware state；本地 PDF 为 `docs/可信harness/papers/2606.10749_secure-llm-agents.pdf`，索引见 `docs/可信harness/papers/导读.md`。它支持本计划的身份、版本和审计边界，但不替代仓库内的离线证明。

### 新增文献对只读诊断的补充（2026-10-27）

以下只帮助丰富 `--explain` 的拒因解释与人工审查线索，**不新增 grant 字段、不扩大 scope、不自动授权**。

| 可解释的边界 | 参考文献 | 使用方式与限制 |
| --- | --- | --- |
| recipient、路由与 fresh review | [Humanize](papers/2610.08900_humanize-judgement-engineering.pdf) | 区分确定性路由与内容评审；review 有用不能改变接收者或权限。只作 plan-06 路由分层的旁证；plan-05 是否降级为 advisory 仍由人决定 |
| device、compiler、`skill_version` 与观测有效期 | [CANNBot-CAKE](papers/2610.31034_cannbot-cake.pdf) | 版本变化后的旧观测降权/重测只作人工审查提示；不进入本批 `--explain` 字段，不在 trust check 自动降权或重签，也不把 `skill_version` 塞进 trust schema |
| 评分材料与 profile 路径只读 | [CUDA Agent](papers/2602.24286_cuda-agent.pdf) | 只作“评测与 profile 路径应对优化者只读”的外部类比。`tasks/` 已由 `forbidden_path` 禁止，本批不把它再写入 `SECURITY_CRITICAL`。仍缺的四个路径是 `tcenv/taskcontract/`、`tcenv/opspec/`、`tcenv/task.py`、`tcenv/distributions.py`，归 plan-06。不等于本仓库已完成隔离 |
| 后端过滤与错误改派 | [AKG](../paper/2512.23424_akg-kernel-agent-a-multi-agent-framework-for-cross.pdf) | 论文支持 DSL/backend 精确过滤及实现错误回 Coder、结构错误回 Designer；其 Conductor 用 LLM 诊断，不能替代宿主确定性路由或作为授权依据。只作 recipient 参考，不新增 role scope |

## 附录：文献与变更索引

只作查阅，不再逐轮重述正文；论文数字均未在本仓库复现。

| 日期 | 影响位置 | 证据文件 |
| --- | --- | --- |
| 2026-10-25 | 文献对应的正文规则与边界 | `2504.11703_progent.pdf`、`2606.10749_secure-llm-agents.pdf`、`2608.08311_ouroboros-reviewed-core-evolution.pdf`、`2607.07663_rsi-survey-bounded-to-autonomous.pdf`、`2606.09498_self-harness.pdf` |
| 2026-10-26 | 当前决定、阶段、验收与跨 plan 归属 | `审核意见-plan-01至09.md` |
| 2026-10-27 | 新增文献对只读诊断的补充 | Humanize、CANNBot-CAKE、CUDA Agent、AKG；只用于解释边界，不新增授权字段 |
