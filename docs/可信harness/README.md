# 可信 harness：当前决定与 plan 索引

这个目录存放可信 harness 的当前计划与归档。主线不是自动改模型权重，而是让下一次会话看到正确的
harness；实测成绩、人工授权和可核证据决定哪些建议能试用，仓库级改动仍由人看 diff、跑测试、提交。

所有 `plan-NN` 都是规划，不是付款或 GPU 授权。默认路径不调用付费模型、不使用 GPU。
`archive/` 只保存历史设计，不是当前施工要求。

## 0. 先知道的现状

| 事实 | 当前含义 |
| --- | --- |
| `harness_auto=false`、`harness_shadow=true`、`harness_review=false`、`inherit_prompt=true` | 默认记录提案，不自动 settle 或调用 reviewer；`harness_auto` 打开时才有 settle/overlay 渲染。`inherit_prompt` 只把少量候选交给后续队友复核，不激活 overlay |
| 先前盘点为 12 条结构化提案、10 条 valid、0 条 premise、0 条 `trusted_record=true` | 本轮未重数。P1 流量很小，且多数是开发者工具/报告需求，不应注入写核 workdir |
| quorum 从未成立 | 单来源 review 已只作 advisory，不再生成新的 `reviewed_active`。盘上仍有 b413 的历史 `shadow=false` overlay 行；试点后续 TASK 没有观测到这段文本被送达 |
| second 配置与 `AGENTS.md` 都是 `kimi-code` / `k3.1-pro`；grant 仍是 `moonshot/kimi-k3.1-pro`、`moonshot/kimi-k3`，外加 `deepseek/deepseek-flash`；三条 grant 全是 `conv2d` | 配置名与 grant 名不得自动别名；P1 当前只是 conv2d 实验，放宽 task/target 需具名批准 |
| `experience.md` / `device.md` / 技能文档留在 grant 指纹内 | 是否移出是独立扩权，本次不碰。每场 peer_tips、overlay 行、远程 TASK 字节不进 grant 指纹，进每场 `prompt_surface_digest` 与 delivery receipt |
| 复验是三层，不是一个缺口 | ① `differential_recheck` 默认关 ② `confirm_batches=3` 已默认开 ③ 秘密 holdout 在未跟踪的 `tcenv/portfolio_validation/`，未接入默认搜索 |
| `collect_arm` 向 `feedback.jsonl` 要 `cost_usd` | 真实记录没有该字段，任何真实 run 的 collect-experiment 必失败。改读 `transcript.jsonl` / `ledger.json`，不双写 |
| `pytest.mark.llm` | 保留该 marker。2026-11-03 检索 `tests/` 下 0 处 live 用例。不加「必须 >0」的结构检查 |
| 201 条人工 claim 中 rejected/refuted 为 0；adopted 多表示测量有效 | 自动暂停没有当前输入，人工采纳率也不能解释为建议质量。本行沿用既有盘点，本轮未重数 |

真正已产生改进的是「自由文本 → 人审 → P2 提交」路径。`b4131be13c4750c8` 的 promotion 已 commit `178de7c`，说明这条链能把一条建议写进仓库；它不是 overlay 改善搜索的效果证据。人审已决定从 `experience.md` 删除该 bullet，事实留在案例文档与开发者队列，不做「保留 + 加注」。盘上 overlay 行仍在；验收说的是新决策零 overlay、零 `inherit_prompt`。当前任务里的队友提示采纳已由 plan-06 实现：宿主字段通过后，后续 LLM 在 TASK 内复核并决定 adopt/reject/unknown，不逐条人审、不额外调用 reviewer。同意不扩权、不把 `reported` 改成 `verified`、不恢复 `stale`/`retired`，也不是 overlay 激活或效果证据。

## 1. 阅读顺序

0. [代码现状-2026-11-02.md](代码现状-2026-11-02.md)：行号级事实底稿。plan-01/02/05/06/07/08/10 的「当前代码核对」已按该文改到 2026-11-02，并在 2026-11-03 用符号加行号复核过漂移。plan-03 只改了与已拍板路径清单冲突的两处。plan-04 与 plan-09 本轮未改。
1. [合理性审核-plan-01至09.md](合理性审核-plan-01至09.md)：流量、代码与成本判断，决定当前优先级。
2. [审核意见-plan-01至09.md](审核意见-plan-01至09.md)：施工图与早期跨 plan 矛盾；其中论文抽查保留证据边界。
3. 各 plan 文首的「当前决定」。正文只保留当前批次需要的规则；完整旧设计去[归档索引](archive/README.md)。
4. [架构与理论.md](架构与理论.md)、[实现交接.md](实现交接.md)、[试点结果.md](试点结果.md)：区分设计、实现和历史证据。
5. [papers/导读.md](papers/导读.md)：外部文献索引；论文数字不是本仓库结果。
6. [plan-10-决策账本与平台期重规划.md](plan-10-决策账本与平台期重规划.md)：研究候选，未授权实施。它不计入第 3 节的 8 个工作包。

决策案例：[案例-开发者提案误入overlay.md](案例-开发者提案误入overlay.md)。它记录 `b4131…` 如何从 developer tool request 变成 `reviewed_active` overlay 与 experience bullet，供路由修复决策使用。

外部案例：[案例-KDA²-算子作弊与可信评测.md](案例-KDA²-算子作弊与可信评测.md)。只读证据，记录 KDA-for-KDA 仓库里一颗被 agent 作弊的 Kimi Delta Attention kernel（judge 接受 3.74x 后代码审查取消资格）及其对应的可信验证正面做法，供 plan-01 验收器抗作弊的攻击语料与三层复验方向参考。

Host 成本与 ABI 对比：[KDA、CAKE2 与 ShapeForge 的 Host 调度成本对比](KDA-CAKE2-与ShapeForge的Host调度成本对比.md)。只学习通用 Host+Device、输入驱动调度、状态 handoff 与原生交付机制，不复制 CUDA/CANN 平台 ABI；当前 native64 状态以 `../架构/native-host-program-ABI.md` 为准。

ABI 分层设计与无 ABI 失败模式：[KDA、昇腾 CAKE2 与 ShapeForge 的 ABI 设计深度对比](KDA-CAKE2-与ShapeForge的ABI设计深度对比.md)。从任务语义、外部调用、Host+Device 数据、生命周期、安全和证据身份解释三方为何采用不同 ABI，并说明 ABI 如何约束交接而不替候选决定算法；当前以 native64 路径为准。

多 Track 入门与深度对比：[KDA、昇腾 CAKE2 与 ShapeForge 的多 Track 交付对比](KDA-CAKE2-与ShapeForge的多Track交付对比.md)。用同一算子的不同交付方式解释 `msl-single`、`msl-bundle`、Candidate、Variant、Agent 和 Target 的区别。现行 available 的 Metal Track 是 `msl-single`、`msl-bundle` 和 `tilelang-pipeline` 的 `/2`、`/3`；`/1` 仍是 `interface_only`。清微 NPU 模拟不构成可用 Track。`publication_allowed=false`。

相关调研：[ForgeKernel 调研](ForgeKernel-调研.md)。只读备忘，记录论文可核实状态与可借鉴点。它不改变第 3 节 8 个工作包的优先级或授权。决策账本与平台期重规划只以 [plan-10](plan-10-决策账本与平台期重规划.md) 的研究候选身份存在，当前未授权实施。

后续调研：[Evolving Agent Teams 调研](Evolving-Agent-Teams-调研.md)。论文与 EAT 工程已下载并只读核对；它支持“根据已验证失败改变提示/角色”的研究方向，但不授权新增监督者、自动团队演化、自动知识库或全权限执行。

相关文档：[../代理反馈.md](../代理反馈.md)、[../架构/评测.md](../架构/评测.md)、
[../架构/数据层.md](../架构/数据层.md)、[../KDA框架与本工程对比.md](../KDA框架与本工程对比.md)、[../security.md](../security.md)。

## 2. plan 的当前落点

plan-01～plan-09 是这一批计划。下表按行区分已经落地的行为和仍未授权的部分，不要把整批读成尚未实现。表末的 plan-10 只为和 `plan-NN` 文件集合对账：它是研究候选，未授权实施，不进入第 3 节的 8 个工作包。

| 编号 | 当前只做 | 当前不做 / 重启条件 |
| --- | --- | --- |
| [plan-01](plan-01-验收器抗作弊.md) | 核对三层复验，再审阅离线 CPU fixture 设计。① 默认关 ② 已开 ③ 未提交且未接入 | 不把②写成缺口，不把①写成计后换输入门，不实现单次 bench 内换输入，不接入 holdout，不跑 GPU |
| [plan-02](plan-02-可证伪预测与效果实验.md) | 先改 `collect_arm` 的账本读法；修好 `--repetitions`；配对改为 pair 身份。494 与 526 不是同一种截断 | 不在 `feedback.jsonl` 双写 `cost_usd`。提高 TASK 预算只作可证伪候选，本批不实现。其余实验设计仍归档 |
| [plan-03](plan-03-信任与权限治理.md) | 只读 `--explain`，逐维拒因与四套身份/conv2d-only grant。指纹扩面边界已写入 §③.1.1 | 别名、v2、自动恢复冻结。不实现重签，不做 old→new hash 别名。移出 `experience.md`、放宽 task 仍须另批 |
| [plan-04](plan-04-证据链与可信执行记录.md) | 保持现有 premise fail-closed | 本轮未改。全部新框架冻结；出现真实 premise 或 trusted_record 后重启 |
| [plan-05](plan-05-独立评审与反共识.md) | 配合路由，决定 review 关闭或只给人看。修正 overlay 固定标题的双模型措辞；该措辞与指纹扩面、b413 删除合并为一次性重签 | 参考集、Brier、轮换、常驻批准者冻结。动态 overlay 行不进 grant 指纹 |
| [plan-06](plan-06-注入面与运行时安全.md) | 按接收者/可执行性分流；`inherit_prompt` 已实现为 TASK 内队友复核。`SECURITY_CRITICAL` 补四个评分材料路径。b413 拆成历史盘上与今天重放。围栏由 hmz 提供，本仓库不验证 | 完整评分 TCB、observer、排序白名单、真实红队归档。不把四个不存在的旧测试文件写成基线，不把队友复核写成长期认证 |
| [plan-07](plan-07-经验知识生命周期.md) | `reported`/stale/retired 过滤已实现，staging 与 SkillIndex 共用。b413 bullet 决定删除但源文件未改。`experience.md` 留在指纹内 | entry_id/CAS、artifact、迁移冻结。不做「保留 + 加注」。`inherit_prompt` 不写经验库、不升级 `reported` |
| [plan-08](plan-08-观测采纳成本与运营.md) | 渲染时 delivery receipt 含预算与各块 cap，同一组值写入本场 `run_config`，加离线 S0。成本只按美元、token 与自检 GPU 秒 | 周报、$40、活跃小时、wall-hours 不在本批。`runs/*/events.jsonl` 不存在，不新建这条日志来凑观测 |
| [plan-09](plan-09-文档一致性与可核验性.md) | 窄链接/索引、常量与 `_permission` 边界核对，提出 `AGENTS.md` 修正 | 本轮未改。大型 registry、默认行号 warning、联网检查不做 |
| [plan-10](plan-10-决策账本与平台期重规划.md) | 无实施项。文件只登记离线/CPU/fake-session 规格，并写明 `confirm_batches` 已开 | 研究候选，未授权实施。不接任务书、不改搜索、不加 Agent；不进入下面 8 个工作包 |

另纸 stub 只说明归档与重启条件：[评分可信计算基-另纸.md](评分可信计算基-另纸.md)、
[确认性效果实验-另纸.md](确认性效果实验-另纸.md)。完整版本均在 `archive/`。

## 3. 当前实施顺序

下面按行标明哪些已经落地、哪些仍是计划。已实现的行不是 GPU 或付费授权。其它 agent 的并发改动不纳入这些工作包。

| 顺序 | 当前缺口（已只读核对） | 计划中的最小修改与离线验收 | 本批明确不做 |
| --- | --- | --- | --- |
| 1. 路由规范与 replay | **已实现（2026-11-03）**：`route_proposal` 计算 `recipient` / `requires_repo_change` / `actionable_within_session`；`kind_of` 仍是证据分类。TEMPLATE 新增 `audience` / `needs` / `peer_action` 教模型表明意图，自报不算事实 | b413 新决策进 developer queue 且零 overlay、零 `inherit_prompt`；28c 类只过新增路由门。quorum、自动送评审与写入经验库均已接门；单来源 review 只作 advisory。真实 run 记录回放通过。`inherit_prompt` 已实现为最多两条候选的 TASK 内复核，无额外 reviewer 调用 | 不按 kind 一刀切；不把队友复核写成长期认证。提示面变化后指纹为 `49ace437…`，未自动重签 |
| 2. 复验审查与离线 fixture | 三层：① `SearchConfig.differential_recheck`（`taic/llm.py` 的字段，默认 `False`）② `confirm_batches=3`（`taic.toml`，读取点在 `taic/orchestrator.py` 的确认批次）已开，`timing_uncertain` 不发布 ③ `tcenv/portfolio_validation/` 未跟踪、未接入默认搜索 | 核对①②与 `tests/shared/test_differential_recheck.py`（该文件已存在）。tail/shape、no-op/回退、退化题只作为计划新增的离线 CPU fixture 规格 | 不改已开启的 confirm 默认值，不打开 differential_recheck，不实现单次 bench 内换输入，不接入 holdout，不跑 GPU。不引用四个从未存在的攻击测试文件 |
| 3. P2 评分材料 | **已实现（2026-11-03）**：`SECURITY_CRITICAL` 补入 `tcenv/taskcontract/`、`tcenv/opspec/`、`tcenv/task.py`、`tcenv/distributions.py`；`tasks/` 仍由 `forbidden_path` 禁止 | `test_security_boundaries.py` / `test_code_agent.py` / `test_acceptance.py` 共 30 项通过；`docs/security.md` 已同步 | 不把 `tasks/` 再写入常量。`report.py` / `taic/parse.py` / `taic/evidence.py` 按角色复核，不一刀切禁改。不开放自动 patch，不扩建完整评分 TCB |
| 4. 实验账本与配对 | **已实现（2026-11-03）**：`collect_arm` 不再向 `feedback.jsonl` 要 `cost_usd`；逐场明细改读 `modelcost.session_rows`（`transcript.jsonl` 的 `agent`/`chat` 行），按模型汇总用现有 `Totals`，`ledger.json` 为含 `review:*` 非会话扣费的总额权威，三层对账失败即 `ExperimentError`，不补 0。`make_plan`/`read_plan`/v1 升级/runner 共用 `_plan_cells`，`pair_count` 是独立 seed 数，`len(seeds) == pair_count * repetitions`，`repetitions>1` 合法计划可读。每个样本带 `{seed, repetition, pair_index, order, arm, run_digest}`，`paired_effect` 按 `pair_index+seed+repetition` 连接并在 pair 内收拢 repetition，`:526` 等长护栏保留 | `test_harness_experiment.py` + `test_harness_runner.py` 42 passed，`test_modelcost.py` 19 passed（全部离线 CPU）。新增真实形状账本对账、缺账本/对不上、真实 collector→summary 跨臂、只打乱一臂的身份配对、缺 cell/重复 cell/错 arm/错 digest、未声明重复 seed 等用例 | 不加预测字段、不做确认性实验、不付费、不用 GPU，未跑真实 run，效果未验证。不双写 `cost_usd`，不改 orchestrator 写入端与 evaluator。不按窗口把 TASK 预算填满；「提高预算是否更好」只登记为可证伪候选 |
| 5. 渲染 delivery receipt | `render_contract`（`taic/kernel_agent.py`）已生成 `omitted`。`prompt_surface_digest`（定义在 `taic/workspace_env.py`，调用在 `taic/kernel_agent.py`）只哈希 TASK.md 与 flow skills，不是 append-only 回执，不含 omitted、预算、各块 cap 或 overlay 行键 | 渲染当场写 run/session/proposal keys、TASK/overlay SHA、bytes、omitted/reason，以及 `TASK_MAX_BYTES` 与各块 cap；同一组预算与 cap 写入本场 `run_config`。离线 S0 验 treatment 含目标 overlay、control 不含。这些是计划，当前没有回执测试 | 不做周报、$40 试点、真实确认性校准或自动运营。不把活跃小时或 wall-hours 写成已有指标。不把每场哈希写进 grant 指纹 |
| 6. staging 过滤与人工去重 | **已实现（2026-11-03）**：`filter_experience_for_prompt` 只保留 `Confidence: verified` 小节并删除裸 `[harness …]` bullet；`stage_guidance` 与 `SkillIndex` outline/lookup 共用。`targets/apple-m4-max/experience.md` 现为 14 个二级标题、13 条 `verified`、1 条 `reported`（affine_ramp）。b413 的 `ab` 子弹仍留在源文件末尾，过滤后不进提示面 | 新增 `tests/shared/test_experience_staging.py`；本机离线 11 + 32 + 48 passed。b413 源条目删除、过滤策略纳入指纹留待一次性重签；device/experience 盘点结论为维持现状。详见[经验盘点-2026-11-03.md](经验盘点-2026-11-03.md) | 不自动写经验，不做 entry_id/CAS/artifact/跨 target 迁移，不把 `experience.md` 移出指纹，不做「保留 + 加注」 |
| 7. 只读诊断与 review 决定 | **review 已决定并实现（2026-11-03）**：单来源 review 只作 advisory，`reviewed_active` 不再产生新行；`harness_review=false`，默认不自动调用 reviewer。`--explain` 仍不存在 | 相关 104 项测试通过；历史 reviewed_active 行只作证据读取。剩余计划是逐维只读拒因：离线验收拒因一次列全，`is_trusted` 布尔结果不变，报告不写 registry、run 或索引 | 不自动别名，不放宽 task/target，不建常驻批准者或参考集，不做 old→new hash 别名。本批不实现重签；plan-05 四个重启条件未满足前不重开自动 review |
| 8. 文档窄核对 | 新文献已增加书目与不同来源层级，plan 与代码证据仍需可核。本轮未改 plan-09 | 跑相对链接、plan 集合、表格与来源检查；保留 `papers/导读.md` 为唯一可信书目 | 不建大型 registry、不联网下载、不把学习笔记或博客当独立证据 |

上表按行区分已实现和仍是计划。不要把整表读成尚未授权。决策账本与平台期重规划放不进第 4 项的实验账本与配对、plan-07 的过滤与人工去重、或 plan-08 的回执与离线 S0，因此另登记为 [plan-10](plan-10-决策账本与平台期重规划.md)。plan-10 不增加第 9 个工作包。

冻结项及可数重启条件见[归档索引](archive/README.md)。没有满足条件时，不把归档中的字段或测试名当成本批要求。任何工作包进入实现前，仍须逐项人工确认；发生并发修改时先停下来核对 diff，不覆盖其它会话。

## 4. 共同纪律

1. **plan 不是批准。** `taic run/resume`、付费 review、真实 GPU 验证都要另行具名授权。
2. **收窄可以直接做，扩大必须具名批准。** 不改信任、容差、预算、评测门槛或审批逻辑来让测试变绿。
3. **区分现状、规划与归档。** 未实现的 `suite_id`、observer、确认性 verdict 等不得写成现有能力。
4. **发布语义分开。** 合法信任收窄只停止后续 overlay，内核确认发布走原确认门；`host_unverified` 只禁止 harness 采纳/效果证据。非法扩权或评分篡改才同时作废两类结果。
5. **测试失败先看** `logs/pytest/triage/triage_summary.md` 和 handoff，不先翻完整 transcript。
6. 新 plan 使用 plan-09 的模板与编号；写完跑相对链接检查并登记。另纸 stub 不占 `plan-NN` 序列。
