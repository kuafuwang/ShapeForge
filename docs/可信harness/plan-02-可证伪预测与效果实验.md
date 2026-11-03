# plan-02：可证伪预测与效果实验

状态：**本批的成本账本、repetitions 与配对身份已实现（2026-11-03），离线测试通过；未跑真实实验，效果未验证。** 其余设计归档。默认不调用模型、不使用 GPU。提高 TASK 预算只登记为可证伪候选，本批不实现。

## 审核补修（2026-11-03）

独立审核发现的两处对账漏洞已修复：`ledger.json` 缺失或为空 `requests` 时，不再把无凭据差额记成非会话 review 成本；`transcript.jsonl` 没有任何计费会话时，即使 `spent=0` 也抛 `ExperimentError`。新增两项回归，保留“transcript 总额大于 ledger”和“ledger 多出会话”原有错误顺序。历史 resume 产生的 `unaccounted` 仍按 fail-closed 处理，没有为兼容旧账放宽。

## 当前决定

- **输入**：`taic/harness_experiment.py`、`taic/modelcost.py`、`transcript.jsonl` / `ledger.json`、现有 mock runs，以及 `_session_brief` 的 overlay 渲染开关。
- **本批先做成本账本**：`collect_arm` 改为复用 `taic/modelcost.py` 的会话聚合，读 `transcript.jsonl` 的 `agent` 行和 `ledger.json`。**不在 `feedback.jsonl` 双写 `cost_usd`。** 这是真实 run 的硬阻塞，不是可选优化。
- **本批接着修 `--repetitions`**：统一 `len(seeds) == pair_count * repetitions`。v1 升级按 repetitions 重算 `pair_count`，不再写成 `len(seeds)`。补乱序、缺失、重复的离线测试；这些测试当前不存在。
- **配对身份**：control/treatment 按 `pair_index + seed + repetition` 连接。`:494` 与 `:526` 不是同一种截断，但现在都按位置配对，一起改，不另开第三种实验。
- **本批不做**：新预测字段、`paired_required` 实验、真实校准、确认性 verdict、付费或 GPU。不按窗口把 `TASK_MAX_BYTES` 填满。「提高预算是否更好」只作以后的可证伪候选。确认性设计见[确认性效果实验-另纸.md](确认性效果实验-另纸.md)，完整旧设计见[归档](archive/plan-02-可证伪预测与效果实验.md)。
- **完成产物**：账本改读后的离线对账，以及按 pair 身份连接的离线记录。失败先读 `logs/pytest/triage/triage_summary.md` 及 handoff。测试未落地前不得写成已通过。

## 当前代码核对（2026-11-02 事实，2026-11-03 复核的**修改前**状态；修复结果见 §1.6）

- **硬阻塞**：`collect_arm`（`taic/harness_experiment.py:362`）在 `:399` 读取 `_experiment_feedback_records`（`:347`，文件是 `feedback.jsonl`）。有记录时 `:406-411` 逐行要求有限的 `cost_usd`，并与 `result.spent_usd` 对账；没有记录但 `spent > 0` 或有 trace 时，`:402` 抛出 `an experiment needs a complete per-session cost ledger`。写入端 `taic/orchestrator.py:2846-2881` 的 feedback 记录没有 `cost_usd`。因此任何真实 run 都会失败。只有合成测试数据能过。
- **权威账本**：`taic/modelcost.py:45` 的 `Totals` 只聚合 `event=agent` 行（`:61`）；`from_events` 在 `:105`，`from_path` 在 `:117`。`docs/架构/数据层.md:87` 把 `ledger.json` 定为汇总权威，review 等非会话扣费也在 `ledger.json`。本批复用这套聚合，不在 `feedback.jsonl` 再写一份美元。
- **口径**：只有美元和 token，外加自检 GPU 秒（`taic/kernel_agent.py:115` 上限 120 秒，累计在 `:1294`）。`first_eligible_cost_usd` / `first_eligible_elapsed_s`（`harness_experiment.py:21`，取值约 `:436`）是到首个合格候选为止的累计量，不是单颗内核成本。没有活跃小时，也没有 wall-hours 汇总。
- **`:494` 与 `:526` 不是同一形态**。`paired_effect` 在 `:494` 裸 `zip(left["samples"], right["samples"])`，长度不等时靠 `:506-509` 的 `max_missing_fraction`（默认 `0.0`，`:186`）判 inconclusive。护栏在 `:523` 先要求两侧样本等长，不等就记入 `guards` 并 `continue`，`:526` 的 zip 不再静默截断。两处仍然按位置配对。`pair_index` 只出现在 `experiments/harness_runner.py:76` 的 cell 记录里；`collect_arm` 与 `paired_effect` 不读它。samples 仍是裸数值列表。
- **`--repetitions>1` 不可用**。`make_plan`（`:232`）写 `pair_count = len(seeds) // repetitions`。v1 升级（`:260`）把 `pair_count` 写成 `len(seeds)`。`read_plan` 在 `:276` 要求 `pair_count == len(seeds)`，又在 `:300` 要求 `len(seeds) == pair_count * repetitions`。后两条同时成立只在 `repetitions == 1`。因此 `repetitions>1` 必被 `:276` 拒绝。`tests/shared/test_harness_experiment.py:25` 只构造 `repetitions: 1`，没有乱序、缺失、重复或 `>1` 的用例。这些用例是计划新增，当前不存在，不得写成已通过。

## 1. 先修现有错误，不扩建实验框架

### 1.1 成本账本

`collect_arm` 停止从 `feedback.jsonl` 逐行读 `cost_usd`。会话美元与 token 改由 `modelcost.Totals` 聚合 `transcript.jsonl` 的 `agent` 行；review 等非会话扣费读 `ledger.json`，与 `docs/架构/数据层.md:87` 的权威一致。不在 `feedback.jsonl` 增加 `cost_usd` 字段，也不为旧 run 补写一份平行账单。

对账失败、transcript 缺失或 `spent_usd` 对不上时，结果是 `invalid/inconclusive`，不得用 0 填充。`first_eligible_cost_usd` 继续表示到首个合格候选为止的累计美元，不改成单颗内核成本，也不引入活跃小时或 wall-hours。

### 1.2 配对身份

`paired_effect` 的 `:494` 与护栏的 `:526` 都还按数组位置配对，但护栏已经有等长前置检查，主指标没有。修的是身份，不是把两处说成同一种静默截断。

计划中的记录形状是 `{seed, repetition, pair_index, order, arm, run_digest}`。现状 samples 没有这些字段；`pair_index` 只存在于 runner 的 cell 记录。

- control/treatment 必须按同一 `pair_index + seed + repetition` 连接；数组下标不是身份。
- 缺失、重复、无法匹配或 digest 不符的条目必须显式 `invalid/inconclusive`，不得静默丢弃后重算。`:494` 不再只靠 `max_missing_fraction` 消化错位。
- AB/BA 的 `order` 只用于报告时间趋势，不能用于配对。

### 1.3 `pair_count`、seeds 与 repetitions

冻结一个自洽定义：`len(seeds) == pair_count * repetitions`，`pair_count` 是独立 seed 对的数量。
重复 seed 只能来自事前声明的 repetition，不能增加独立对数；`repetitions=1` 时 seeds 必须唯一，
`repetitions>1` 时也只允许冻结的重复模式。`make_plan`、`read_plan` 与 v1 升级使用同一约束。
v1 升级按 repetitions 重算 `pair_count`，不再把 `pair_count` 写成 `len(seeds)` 之后又在 `:276` 把 `repetitions>1` 拒掉。

### 1.4 离线验收

下列测试是计划新增，当前不存在，不得写成已通过：
- 真实形状的 feedback 记录（没有 `cost_usd`）不再让 `collect_arm` 失败；美元来自 transcript/ledger 的合成夹具，与 `spent_usd` 对得上才算完整。
- 乱序仍能按 pair 身份配对；缺 cell、重复 cell、错误 arm、错误 digest 不成有利零值。
- 合法 repetition 不增加独立对数；未声明重复 seed、长度不符均拒绝。`repetitions>1` 且 `len(seeds) == pair_count * repetitions` 的计划能被 `read_plan` 接受。
- 单测放在已有的 `tests/shared/test_harness_experiment.py` 与 `tests/shared/test_harness_runner.py`（这两个文件已存在；新增的是用例，不是新路径）。不需要真实搜索、模型或 GPU。

### 1.5 TASK 预算：只登记为可证伪候选

人审已决定：不按模型窗口把 `TASK_MAX_BYTES`（`taic/kernel_agent.py:56`，12000 字节）动态填满。预算与各块 cap 写入 `run_config` 和 plan-08 的 delivery receipt，本批不改渲染器。

「提高预算是否更好」若以后要测，control 固定当前 12000 字节，treatment 固定一个事前声明的更高预算。各臂把自己的预算和各块 cap 写入自己的 `run_config` 与 delivery receipt，其余配置一致。不许按窗口临时填满。那是另批实验，不是本批的账本或配对修复，本批不实现，也不授权付费或 GPU。

### 1.6 实施记录（2026-11-03）

本批只做账本读法与配对身份；下列是已完成的代码与离线测试事实，**不包含任何真实实验效果**。

**成本账本**

- `collect_arm` 不再从 `feedback.jsonl` 逐行要 `cost_usd`。逐场明细来自新增的只读辅助 `taic.modelcost.session_rows`（逐条 `event=agent` / `chat` 行，不合并、不补 0）；按模型汇总仍用 `modelcost.from_path` 的既有 `Totals`。
- `ledger.json` 的 `summary.total_usd` 是含非会话扣费（`review:*`）的权威总额。对账分三层：ledger 总额对 `result.spent_usd`；ledger 请求合计对其 summary；每个非 `review:` 请求必须与 transcript 的一条会话按 `label` + 费用逐条对上。
- 以下情况全部抛 `ExperimentError`，不用 0 或默认值补齐：transcript 缺失或没有计费会话、ledger 缺失或没有 summary、`cost_incomplete=true`、费用非有限、请求对不上、transcript 与 ledger 谁多出一条会话。`cost_incomplete=false` 且无费用是写入端声明的零账单（与 `taic/orchestrator.py:2638` 一致），按 0 接受。
- 只用只读抽样核对，未写任何 run：3 个真实 run（`conv2d-flame-20261020`、`20261024-065432_…` 为 `agent` 行，`plan_cover/relu_recheck` 为 `chat` 行）逐条对账通过，非会话扣费为 0。
- `first_eligible_cost_usd` 语义不变：仍是到首个合格候选为止的累计美元；新增一条护栏，首个合格 trace 的费用不得超过整场支出。
- 臂记录的 `change_sha256` 取计划里的写法：真实 control 在 result/config 里写空串（`taic/llm.py` 的 `SearchConfig` 默认值），计划里是 `None`。`collect_arm` 先验证记录值与计划预期一致（control 空串、treatment 哈希），再返回计划值。`summarize_result` 的严格校验未放宽。

**配对身份**

- 每个样本带 `{seed, repetition, pair_index, order, arm, run_digest}`。`paired_effect` 与 guardrail 比较都按 `pair_index + seed + repetition` 连接，`order` 只写报告，不参与配对。
- 同一 pair 的 repetition 先在 pair 内取均值，独立对数只按 `pair_index` 计，所以合法 repetition 不会把 3 对抬成 6 对。
- 重复 cell、错 arm、错 digest、计划外的身份、缺 `cells` 记录都是显式错误；缺计划内 cell 仍由 `max_missing_fraction`（默认 `0.0`）判 inconclusive。`:526` 的等长护栏保留，身份覆盖检查更严。
- 旧 result 文档没有 `cells`，`summarize_result` 现在直接拒绝。`runs/` 下当前没有任何 `harness_experiments/*.result.json`，无需迁移。

**repetitions 与 pair_count**

- 新增 `_plan_cells` / `plan_cells`：`pair_index` 按 seed 首次出现顺序分配，`repetition` 是同一 seed 的第几次出现；每个 seed 必须恰好出现 `repetitions` 次，于是 `len(seeds) == pair_count * repetitions`。
- `make_plan`、`read_plan`、v1 升级与 `experiments/harness_runner.build_cells` 共用这一处推导。`pair_count` 是独立 seed 数（等于 `len(seeds)//repetitions`，且先验证整除与重复模式），v1 升级按 repetitions 重算，`read_plan` 用同一约束校验。
- 多一次重复、少一次重复、同一 seed 出现次数超过 repetitions、长度与 `pair_count * repetitions` 不符都拒绝；`repetitions>1` 且长度自洽的合法计划现在能被 `read_plan` 接受。

**runner**

- `build_cells` 用 `plan_cells` 写 `pair_index` / `repetition` / `order`；cell 目录仍按计划顺序编号，`repetitions>1` 时不再互相覆盖。
- `execute_plan` 把 cell 身份传给 `collect_arm`，合并 samples 的同时合并 `cells` / `run_digests`，并把 `model_totals`、`session_cost_usd`、`non_session_cost_usd` 汇总进臂记录。

**测试实测数字**（全部离线 CPU，无模型、无 GPU）

| 命令 | 结果 |
| --- | --- |
| `uv run pytest -q -n 0 tests/shared/test_harness_experiment.py tests/shared/test_harness_runner.py` | 42 passed |
| `uv run pytest -q -n 0 tests/shared/test_modelcost.py`（动了 `modelcost`） | 19 passed |
| `uv run pytest -q -n 0 tests/shared/test_structure.py` | 6 passed |

新增用例覆盖：真实形状 transcript/ledger 对账；缺 transcript / 缺 ledger / 总额不符 / 请求合计不符 / 未知账单 / 单边多一条会话均失败；真实 `collect_arm` 输出（control 空 digest 规范化为 `None`）能直接进 `summarize_result`；`repetitions>1` 合法计划被 `read_plan` 接受、v1 升级重算 `pair_count`；只打乱一臂、每格数值不同时仍得到按身份配对的退化区间（位置 zip 会得到非退化区间）；缺 cell、重复 cell、错 arm、错 digest、计划外身份不成有利零值；未声明重复 seed 与长度不符被拒；runner 端 12 个 cell 合并成 3 对。

**未做**

- 没有运行真实 `taic run/resume`，没有付费模型，没有 GPU/NPU；没有做真实实验、校准或确认性 verdict。**离线测试通过不等于效果已验证。**
- 没有在 `feedback.jsonl` 双写 `cost_usd`，没有改 `taic/orchestrator.py` 的写入端，没有改 evaluator。
- `taic experiment collect` 仍只接两个 run 目录，`repetitions>1` 时没有 CLI 参数传 cell 身份；该命令对这种计划会显式报错（`this run seed is repeated in the plan; collecting it needs pair_index and repetition`），runner 路径正常。
- 预算语义未重新定价：`max_total_budget_usd` 与 `summarize_result` 的上限仍按 `len(seeds)`（即全部 cell 数）乘单臂预算。

## 2. 只有决定测 overlay 时：冻结渲染模式

当前 `_session_brief` 只在 `harness_auto=true` 时渲染 overlay；settle、自动评审和 `sync_trust_status`
也在同一开关后。两臂都关则预置 overlay 不显示，两臂都开则各自产生新提案和评审，差量不再单一。
因此不能只在 runner 里预写 overlay 文件。

若未来确有必要测 overlay，新增的是 **orchestrator 的冻结渲染模式**，不是 runner 偷改开关：

- 宿主只渲染预先冻结、digest 明确的 overlay；control 同位置为空。
- 该模式不 settle、不自动评审、不触发 `sync_trust_status`，也不允许 run 内新增 active overlay。
- 两臂其余配置逐字段相同，包括冻结的 `TASK_MAX_BYTES` 与各块 cap。渲染时写 plan-08 的回执（TASK/overlay SHA、字节、omitted、预算与 cap），不是事后猜 TASK。本批不实现该回执。
- 模式不存在时，overlay 实验只能 dry-run。是否值得测，先看 plan-06 的路由修正后还剩多少真实 tip。

## 3. 评估单位暂不定案

按 seed 配对不能共享 LLM 采样与搜索路径，且每个 cell 都是完整搜索，成本与单条提示的小效应不匹配。
“按 harness 版本在固定小任务集上比较 N 与 N−1”是待评估替代方案，但也有时间趋势、远端模型漂移和
版本选择偏差，不能直接当成新真相。先做离线近端验收与版本级观测，暂不承诺确认实验或样本量。

## 4. 归档与重启

完整预测合同、确认性统计和运营设计保留在[归档](archive/plan-02-可证伪预测与效果实验.md)。
只有账本改读、repetitions 与 pair 身份都已修、路由后确有可在写核工作目录直接使用的 tip、且另行批准评估单位与预算时才重启。
真实校准和确认性实验仍按[确认性效果实验-另纸.md](确认性效果实验-另纸.md)另行授权。

## 5. 新增文献仅作重启参考（2026-10-27）

以下只用于以后满足重启条件时查阅，**不新增预测字段、不建立新实验、不调用模型或 GPU**。当前仍是：改账本读法、修好 repetitions、按 pair 身份配对。提高 TASK 预算不在此列。

| 文献 | 若以后重启，可借鉴的设计 | 当前必须保留的限制 |
| --- | --- | --- |
| [AHE](papers/2604.25850_agentic-harness-engineering.pdf) | 编辑前写可证伪预测，下一轮核对并按文件回滚；组件/经验/决策可观测 | 不能把单条编辑的归因当成已解决；当前不加 schema |
| [SoL-Pi](papers/2609.20519_sol-pi.pdf) | 指标在优化者控制之外冻结、双门、Pareto、留出单向验收 | kernel 证据含模拟 cycles 且有能力回退；不能据此开启付费 S1 |
| [ModularRSI](papers/2609.14857_modularrsi.pdf) | benchmark-disjoint、模块限改、三道验证门与失败后的按 diff 回滚 | 只用部分进化实例，自评自审仍需独立复核；不重启自动 overlay |
| [Meta-Harness](../paper/2603.28052_meta-harness-end-to-end-optimization-of-model-harnesses.pdf) | 同集搜索与评测可能过拟合，提示确认集要独立 | 只作警示，不据此新增 harness 搜索或真实校准 |
| [AscendOptimizer](papers/2603.23566_ascendoptimizer.pdf) | 经验的移除检验和实测退化可以变成可证伪预测 | 其设定为传导式 autotuning，且代码尚未释放；不把效果外推到新任务 |
| [KernelBenchX](papers/2605.04956_kernelbenchx.pdf) | 按机器分别报告正确性与性能，并单列跨机 max/min speedup 差异，避免只报跨机平均值 | 该比值不是重复测量的统计方差；本仓库样本仍少，不把论文结果当成本地测量 |

这些资料加强的是“重启时怎样设计对照”的判断，不改变当前账本、repetitions、配对身份与离线验收的完成定义。账本改读、repetitions 与配对身份已按 §1.6 落地并离线验收；重启仍须另行具名批准。
