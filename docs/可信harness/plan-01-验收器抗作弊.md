# plan-01：验收器抗作弊

状态：规划，未实施。本文只覆盖 `tcenv` 的正确性、验证强度、作弊语料与回归方法；默认路径不调用模型、不使用 GPU。

## 当前决定

- **本批只做**：核对三层复验的现状，并审阅 tail/shape、no-op/回退、退化题的离线 CPU fixture 设计。未写判定按原始字节比较仍是设计，不先于上述核对。
- **三层不要混写**：① `SearchConfig.differential_recheck` 默认 `False`，是升种子前的第二种子 quick 复查。② `confirm_batches=3` 已默认开，是计后换种子的多批确认。③ 秘密 holdout 在未跟踪的 `tcenv/portfolio_validation/`，未接入默认搜索。单次 bench 内换一份新输入再启动，与这三层都不是同一个门，本批不实现。
- **输入**：现有 evaluator/测试、已装配的任务对象与公开攻击模式。不读秘密 holdout。本文说的验收是 `tcenv/evaluate.py` 的关卡梯（`device_check` → `screen` → `compile` → `abi` → `correctness` → `bench`），不是 `taic/acceptance.py` 与 `harness_experiment.milestones` 的提案采纳平面。
- **本批不做**：relative L2 硬门、单次 bench 内换输入、把 holdout 接进默认搜索、真实 Metal/GPU 或付费攻击生成。不改已开启的 `confirm_batches`，不把①默认打开。
- **完成产物**：三层复验的核对记录，和候选 fixture 规格的审阅记录。攻击/solver 对照 fixture 是计划新增，当前不存在；另获实现授权后才成为回归。子集 `eligible` 不得写成完整验收。
- **失败去向**：先读 `logs/pytest/triage/triage_summary.md` 及 handoff；规划中的测试文件不存在时不得冒称可运行。

## 当前代码核对（2026-11-02 事实，2026-11-03 复核，供审阅，未实施）

- ① `SearchConfig.differential_recheck`（`taic/llm.py:315`）默认 `False`。闸门在 `taic/orchestrator.py:1910`，实现是 `_differential_recheck`（`:1971`）。它在升种子或噪声内并列之前做不计时复查。这不是计后换种子确认。
- ② 计后换种子确认**已经默认打开**：`taic.toml:30` 的 `confirm_batches = 3`，默认值也在 `SearchConfig.confirm_batches`（`taic/llm.py:298`）。`taic/orchestrator.py:3608` 起每批用 `_phase_seed("confirm", …)`（`:3630`），`reps=eval_reps*2`（`:3632`）。任一批失败即 `not publishable`（`:3698-3700`）。`timing_uncertain` 不发布（`taic/policy.py:949`）。本批不把这一层写成缺口，也不改它的默认值。
- ③ 秘密 holdout：`tcenv/portfolio_validation/validation.py` 的 `_fresh_seed`（`:100`）与 `ValidationPlan`（`:243`）已写在工作区，目录未跟踪。`taic/orchestrator.py` 不引用它。干净 checkout 上不存在。本批不接入。
- 单次 bench 仍把各轮组装入同一份 `data0`：`prepare_bench_sets` 调用 `one.load_inputs(data0.raw)`（`tcenv/backends/metal/backend.py:566`），计时中抽查 `check_case(task, data0, …)`（`:659`）。这是 bench 内部的输入复用，不是「确认门不存在」。bench 内换输入仍是未授权设计。
- 关卡梯的 2026-11-03 赋值点：`device_check` 在 `tcenv/evaluate.py:377` 与 `:390`，`screen` 在 `:451`，`compile` 在 `:491`，`abi` 在 `:673`（另一路径 `:591`），`correctness` 在 `:693`（另一路径 `:600`），`bench` 在 `:788`（另一路径 `:625`）。`EvalReport.eligible` 在 `:127-134`，只看 `status == ok`。
- 任务来源已经分叉。`tasks/__init__.py:1-12` 的 `get_task` 只装保留的 v1 题，并写明永不读取 v2。v2 走 `tcenv/taskcontract/loader.py`（`opspec.json` + `reference.py` + `baselines.*`）。v1 标记与 v2 混放即 `AssemblyError`（`loader.py:126-147`）。容差权威是 `tcenv/opspec/v2.py:549-575`。`evaluate.py:727` 仍使用装配后的 `task.cases`；不要把 `tasks/<op>/task.py` 写成所有题的路径。基线不是 oracle（`tcenv/taskcontract/baselines.py:1-11`）。
- `tests/shared/test_differential_recheck.py` 已存在，作为①的审查基线。tail/shape、no-op/回退、退化题的 fixture 是**计划新增**。`tests/backends/metal/test_anti_hack_correctness.py`、`tests/backends/metal/test_anti_hack_bench.py` 与 `tests/support/golden/metal/fixtures/anti_hack/` 当前不存在。`tests/shared/test_distributions.py` 已存在，不是这批新文件。不得引用 `test_proposal_routing.py`、`attack_fixtures.py`、`attack_fixtures.json`、`test_acceptance_attacks.py`：四个都不存在。
- 形式验证仍是 `tcenv/verify.py` 恒返回 `unsupported`。本批不把它写成已能判生死。

## 1. 角度与要解决的问题
目标不是寻找「看起来可疑」的源码，而是防止**语义错误或计时语义不一致的候选被判合格、换种子或发布**。把候选 MSL 当作会利用输入分布、容差、重复启动、缓存和检查顺序的对抗程序；同时用合法解集合约束修复，避免把优化本身当作弊。

核心问题是：现有差分门能抓多少已知作弊、哪些评测面仍可被利用、每次补门后能否证明「攻击被挡住且真核没有误杀」。本文采用 hacker–fixer–solver 三角循环，先落地手写、确定性的离线语料，不把付费模型生成攻击当前置条件。

## 2. 目标 / 非目标
| 目标 | 非目标 |
| --- | --- |
| 为每个作弊类别建立独立反例、最小 fixture 和永久回归 | 用静态关键词或代码风格代替数值判决 |
| 覆盖所有契约输出、结构化尾部/状态、独立输入统计量与计后新输入 | 承诺抽样差分等价于形式证明 |
| 让计时后复验、缓存语义变更和新协议可测试、可回滚 | 本文重新设计加速比 A/B、成本或信任授权 |
| 分别报告攻击检出、合法解误拒、未执行和 harness 故障 | 用单一「通过率」宣称已消除 reward hacking |
| CPU 单测先行；真实 MSL 集成复验显式授权 | 默认运行 GPU、真实挂死核或付费 hacker 模型 |

合法的形状特化、常量展开、tile 特化、符合容差的近似和多段流水线不是攻击目标；只有违反任务语义、契约或计时协议的 behavior 才判负。

## 3. 现状证据
### 3.1 已有验收门

2026-11-03 复核过的是关卡梯、`eligible`、`confirm_batches` 和 `data0` 复用，行号见文首核对。下表其余格子沿用 2026-10-27 的阅读，本轮未逐条重读；与代码不符时以符号为准，不要把未复核的行号当成新测量。

| 已有机制 | 代码证据 | 能挡住什么 |
| --- | --- | --- |
| 遍历所有契约输出，逐元素要求全部在容差内，不是「99.9% 元素通过」 | `tcenv/backends/metal/testing.py:250-287` | 多数部分输出和局部大误差 |
| 输出毒化、输入不可变、保护带检查 | `tcenv/backends/metal/sanitize.py:261-300` | 未写输出、改输入、常见越界写 |
| 毒化/清零/再次毒化三次启动并比较 | `tcenv/backends/metal/testing.py:386-404` | 读取输出旧值、部分数据竞争 |
| 正常路径使用装配后的 `task.cases`；validate 才缩到一个用例 | `tcenv/evaluate.py:727` | 未传 `cases` 的 `full` 会评估全部 `task.cases`；但 `cases=` 可缩减评估集，且 `EvalReport.eligible` 只是 `status == STATUS_OK`（`:127-134`）。v2 题的 cases 不来自 `tasks/<op>/task.py` |
| 计时含编码与等待，候选/基线交错并取中位数 | `tcenv/backends/metal/runtime.py:417-446`、`tcenv/bench.py:100-150` | 把编码藏到窗外、单次顺序噪声 |
| 计时后检查已启动轮组的安全性，并核对首末轮组输出 | `tcenv/backends/metal/backend.py:651-662` 的 `spot_check`、`tcenv/evaluate.py:838` 的 `harness.spot_check()` | 一部分只在计时段出现的错误 |
| 缓存键含协议、任务/评测语义、源码、种子、用例和目标 | `tcenv/evaluate.py:1069-1080` | 常见跨源码、跨种子误复用 |
| 确认换阶段种子、禁用缓存，默认三批，已经打开 | `taic.toml:30`、`SearchConfig.confirm_batches`（`taic/llm.py:298`）、`taic/orchestrator.py:3608` 与 `:3637` 的 `eval_cache=False` | 把搜索旧报告冒充确认重测。这是复验第②层，不是缺口 |
| 形式验证明确为 `unsupported` | `tcenv/verify.py:53-69` | 防止把未证明写成已证明 |

已有 GPU 回归也覆盖部分未写输出和输出初值依赖（`tests/backends/metal/test_tcenv.py:212-235`）；它们是基础，不等于系统化的作弊语料。

### 3.2 已知缺口或待证伪假设
下表把代码事实与攻击推断分开；没有实际复现的条目不写成「已可利用」。

| 代码事实 | 风险判断 |
| --- | --- |
| 单次 bench 的各轮组装入同一份 `data0.raw`（`prepare_bench_sets`，`tcenv/backends/metal/backend.py:566`），计时抽查仍用同一 `data0`（`:659`） | **bench 内部缺口，不是确认门缺失**：候选也许能在预热/首轮算一次、后续重放。计后换种子的第②层 `confirm_batches=3` 已经默认开。bench 内再换一份输入仍未实现，实际攻击成功率未核实 |
| 计后数值只查已启动集合的首、末两项，中间集合只查 safety（`tcenv/backends/metal/backend.py:651-662` 的 `spot_check`） | **覆盖缺口**：按轮组索引选择不算/错算可能漏过；需 fixture 证实 |
| `check_case` 的 `ok` 只看 `n_bad_total`，`unwritten_total` 只进入诊断（`tcenv/backends/metal/testing.py:266-292`） | **纵深缺口**：毒化未写应独立令该 case 失败，不能依赖数值碰巧不等；端到端是否仍可借三次启动挡住需保留回归。注意判定语义：毒化图案是 `0xFF`（`tcenv/backends/metal/sanitize.py:37`），对整数即 `-1`、对浮点即负 NaN，因此「值等于毒化图案」本身不是「没写过」的充分条件 |
| 允许基线误差把门放宽到 2 倍、上限 8（`tcenv/backends/metal/testing.py:248-258`），尺度通常来自参考 RMS（`:119-134`） | **容差面**：稀疏误差、近零参考和「贴着门优化」需要边界语料；不能未经 solver 校准就全局收紧 |
| ①默认关闭（`SearchConfig.differential_recheck`，`taic/llm.py:315`）；②确认多批已开；③holdout 未接入 | **流程面**：搜索 `ok` 不是强证据。是否打开①应由漏检数据决定，不在离线阶段偷改默认值，也不把②写成还没做 |
| 一元 fp16 全域扫描不进入常规 `task.cases`，也不覆盖其他合同/尾部（`tcenv/backends/metal/domain.py:3-24`） | **覆盖边界**：不能把 domain sweep 或抽样写成全任务证明 |
| `evaluate(..., cases=...)` 可缩减用例；`EvalReport.eligible` 只是 `status == STATUS_OK`（`tcenv/evaluate.py:127-134`）。用例选择在 `:727` | **报告语义缺口**：子集全过同样得到 `eligible`，因此 `eligible` 不等于完整验收。这里的验收是评测关卡，不是 `taic/acceptance.py` 的提案里程碑。正式编排是否可被缩减 `cases` 绕过，本轮未核实 |
| 派生种子含 run seed、phase、label 和候选源码 SHA-256（`taic/orchestrator.py:83-109`） | **不列为本批攻击驱动**：为目标输入预计算答案还涉及哈希不动点，不能简化成“知道 seed 就能作弊”。公开派生仍不等于私密 holdout；结构化分布反例比种子预测更值得先做 |

本次检索未找到专门面向这些攻击面的 Metal 作弊 manifest；仓库外是否另有语料未核实。

### 3.3 外部反面教材
KDA 的五类作弊见 sibling checkout `kda-260927-kda/hacking_example/README.md:30-38`；其 README 记录六个 workload 全失败而 geomean 为 3.727x（`:72`），这是该仓库的记录，不是本仓库实测。其现行 `bench.py` 同时要求每元素门、relative L2、有限值和两个输出（`kda-260927-kda/bench.py:55,144-154,229`）。可迁移的是攻击模式与计后换输入复验（README `:86-95`），不是 CUDA 数字或阈值。

## 4. 分阶段方案
### 阶段 0：冻结威胁模型与基线（纯 CPU、只读分析先行）
1. 建攻击登记表：`attack_id / task / semantic_counterexample / surface / suite_role / expected_stage / expected_reason / solver_set / at_risk_solvers / execution`；`suite_role` 区分 `dev / frozen_confirm / rotation`，`execution` 记录 `cpu / gpu` 与最终 `executed / not_executed / harness_error`；未跑或 `gpu_not_run` 不得记为已检出。
2. 把「攻击成功」定义为：已有独立反例证明候选语义错误，而它仍 `eligible`，或计时后新输入错误但仍产生可用于选择的加速结果；仅编译成功、自检成功或静态可疑不算成功。
3. 为现有测试做覆盖矩阵，优先复用 `test_tcenv.py`、`test_testing.py`、`test_eval_cache.py`，避免只新增同名重复断言。
4. 每个新攻击先在旧 evaluator 上留下失败测试或复现记录，再修门；CPU 可证实的部分不需要 GPU。真实 MSL 的旧版复现统一标「需人工授权」。

### 阶段 1A：计后 fresh-input 复验（设计草案；本审阅卡未授权实现）

下面描述的是**单次 bench 内部**、最后一个计时样本之后换输入再启动。它和两件已有机制都不同：① `differential_recheck`（默认关）② 已经默认打开的 `confirm_batches`。bench 内换输入当前没有。本审阅卡只核对①②的边界并审阅 fixture 规格；不改 evaluator、不打开①、不改②的默认值，也不跑这些 fake runner。

1. 在最后一个计时样本之后增加 fresh-input check：保留计时末输出，原地写入不同 seed/结构的新输入，按同一 launch 协议再启动并逐输出核对；**这一次不重新毒化**，使依赖旧输出图案的早退核暴露。随后再以毒化输出启动一次，区分缓存重放与未写全。新输入生成、清场和复验都在计时窗外，不改写实测 `samples_s`。
2. 先用 fake timing runner 复现三类攻击：只在预热/首轮计算、只计算首末轮组、对旧输入返回旧输出；三者必须在 fresh-input check 失败，合法「每次重写全部输出」runner 必须通过。计后只查首末轮组的现状也要有一个中间轮组特化反例。
3. 正式路径再评估是否对所有已启动轮组做数值检查；开发快速路径若只抽样，报告不得写成完整计后复验。真实 MSL 集成仍留到另行授权的 GPU 阶段，CPU fake 不冒称已复现硬件攻击。

后续协议工作（不属于本批）：每轮组独立输入、缓存/schema 与 `suite_id` 的完整接入、真实 Metal 和付费攻击生成。

### 阶段 1B：correctness oracle 与对抗输入（fresh-input 之后，纯 CPU）
1. 用 fake context/launcher 和直接构造的 raw tensor 测 `prepare_case`、`check_case`、`run_correctness`：缺/多输出、错误字节数、部分写、每个输出各有一个坏点、NaN/Inf 位置错误都必须非 `ok`。
2. 将「从未写过」做成独立失败条件，但先固定判定语义：毒化图案 `0xFF` 对整数是 `-1`、对浮点是 NaN，合法写入 `-1` 或合法产出 NaN 的输出元素与「未写」字节相同。必须按每个输出元素的**原始字节**比较，不能用浮点数值相等或 NaN 语义代替：只有同一元素在毒化轮的字节等于 `0xFF` 填充、且在零填充对照轮的字节等于 `0x00` 填充，才计为「同时跟随两种填充」的未写候选；零填充对照必须在数值已失败时也执行，否则无法区分合法 `-1` 与从未写过。参考值是否等于毒化图案不参与判定：合法写出 `-1`/NaN 时零填充轮会保留该值，未写时零填充轮会变成 0；加参考例外反而会漏掉参考恰为 `-1`/NaN 的真正未写。该对照只增加一次 launch，不改变 `samples_s`。增加全零、全常数但非平凡参考的 solver，防止把「写零」或「写毒化图案」误判为完整实现。
3. 加容差边界向量：恰好门内/门外、RMS 很大而局部参考近零、稀疏尾块错误、基线放宽达到 cap、非有限 scale。relative L2 可先作诊断字段，不在没有 solver 数据时新增统一硬门。
4. 给高价值任务补独立 CPU oracle：短尺寸朴素循环或递推，不能用被测的 `task.reference` 自证。先验证 oracle 与参考在合法输入上一致，再用它判定攻击反例。
5. 新增结构化分布：逐通道 one-hot/basis、仅尾块非零、单块非零、q/k 不同尺度与不同统计量、多 chunk 状态哨兵、排列/缩放的合法 metamorphic 关系；生成器仍用显式 seed。
6. 区分开发用例子集与完整验收套件：报告记录**计划新增**的 `suite_id`（当前 `tcenv/`、`taic/`、`tests/` 中均无实现）与实际/必需 case 摘要，缺必需 case 不得标成完整 acceptance。口径必须区分两条路径：只有 `mode=full` 且未传 `cases` 的调用，`ok` 才等于全部 `task.cases` 通过；传了缩减 `cases` 的调用，即使 `eligible` 也只能报「该子集通过」，不得写成完整验收。**默认只承诺公开的 dev / frozen_confirm / rotation 三集**。秘密 holdout 的代码在未跟踪的 `tcenv/portfolio_validation/`，未接入默认搜索；在它成为已提交、已接入、且 agent 不可读的路径之前，公开 fixture 不冒充秘密确认集，也不把③写成首阶段阻塞项。

### 阶段 3：真实 Metal 语料与协议启用（**需人工授权：GPU**）
1. 将最小 MSL 攻击逐类跑旧版/新版 evaluator，记录实际 stage、状态和反例；不以 CPU mock 代替最终 Metal 集成结论。
2. 对同任务 `naive_msl` 与合法优化变体跑 solver；新门先 diagnostic，再由人审切换 enforce。正式协议切换前必须完成目标任务的 GPU 攻防两侧验证。
3. 命令见 §7，全部带 `gpu` marker 且串行；不运行真实挂死测试，不设置 `TCENV_ALLOW_GPU_HANG_TESTS`。

### 阶段 4：可选新型攻击生成（**需人工授权：付费模型；若执行核还需 GPU**）
仅用已批准预算让 hacker 提出新 fixture，人审后入语料；模型不能自批改 evaluator、自批 solver 或直接晋升补丁。该阶段永远不是合并或日常回归的默认路径。

## 5. 作弊语料设计
### 5.1 KDA 五类到 Triton→MSL 的映射
| KDA 模式 | 本仓库具体作弊核类别 | 必备反例 / solver 约束 |
| --- | --- | --- |
| 常数 q/k 范数 | RMSNorm/attention/softmax 用固定均值、倒数范数或常数 denominator，省略 reduction | 非 iid、one-hot、常数、正负缩放、不同行范数；合法预计算 `eps` 和数学常数必须能过 |
| 丢弃 initial-state 尾部 slab | mamba/scan/GEMM/conv 略过尾 channel、尾 tile、边界通道或 state 分片 | 仅尾块非零、逐通道 basis、所有输出含 state；合法完整尾块特化不能误杀 |
| 不做 state handoff | 多 chunk/多 threadgroup 各自从零累加，不写或不读跨段状态，online softmax 不带 m/l | 两段以上非零哨兵、不同 chunk 划分；单段合法核与完整递推核都进 solver |
| 跳过 triangular solve | 用固定 damping、低阶近似或省略依赖求解/归一化步骤，只在原分布接近 | 病态缩放、抵消、不同条件数的小尺寸独立递推；容差内合法近似不能凭名字拒绝 |
| q/k 共用 RMS | 两个输入、两行或两个 state 共用一个统计量，利用同分布假设 | q/k 独立缩放、异方差、排列与单输入变化；相同输入时结果恰好相同不能单独证明作弊 |

### 5.2 本仓库额外攻击面
| 类别 | 最小攻击 | 对应门 |
| --- | --- | --- |
| 部分输出/部分元素 | 只写某输出、首末元素或首末轮组，其余写零/保留旧值 | 全输出毒化、独立 unwritten 失败、所有已启动轮组数值检查 |
| 容差拟合 | 针对 RMS、基线放宽或 `allow` 边界制造稀疏误差 | 边界向量、结构化尾部、名义/有效容差同时报告；solver 校准后才改阈值 |
| 计时段特化 | 预热真算、正式样本空转；检测输出已写后重放 | 计后保留旧输出并换输入复验，再毒化复验 |
| 输入/轮组缓存 | 记住相同 `data0`、指针或轮组索引，跨 launch 不重算 | 各轮组独立反例、fresh input；指针特化只作假设，需 GPU 证实 |
| 评测缓存重放 | 新检查上线后仍取旧语义的 `ok` 报告 | 协议/schema 失效、完整 suite 身份、确认禁缓存 |
| 种子/分布拟合 | 预测公开派生种子，或只对 normal 的矩和固定 case 顺序拟合 | source 变化后的新 seed、结构化 case、未回显 holdout；公开种子不宣称秘密 |
| 参考/用例退化 | 输入或参考全零、缺 state、参考溢出，使错误解也容易通过 | oracle 自洽与非平凡性检查；此类记 harness bug，不算 hacker 战果 |
| 固定用例集全过 | 未传 `cases` 的 `full` 下，`ok` 表示该次**实际评估**的用例全部通过；`EvalReport.eligible` 只是 `status == ok`（`tcenv/evaluate.py:127-134`），而 `cases=` 能缩减评估集（`:727`），所以传子集时 `eligible` 只覆盖该子集。这里的「完整验收」指评测关卡，不是 `taic/acceptance.py` 的 `reviewed/implemented/experiment_validated/local/committed` | **拟合面与报告边界**：即便 `full` 全过，满分也不提供 `ok` 之外的信息，不是分布拟合诊断、不能当泛化证据；缩减子集得到的 `eligible` 更不能写成完整评测。报告只能写「这批 case 通过、泛化未测」，并给**计划新增**的 `suite_id`（当前 `tcenv/`、`taic/`、`tests/` 中无此符号）与实际 case 摘要。Harmony Harness App. K.6 的 `100/100/100` 是另一件事，只说明「满足度量≠满足属性」。补结构变体需人工授权（GPU），不是离线验收门 |
| P2 评分文件可写 | 写核声明是 `local=ALL,user=READ,system=READ,online=NONE`（`taic/flows/kernel_once/__init__.py:38-42`）。2026-11-02 现状：hmz 把它翻成 Landlock/Seatbelt，架不起围栏就拒绝启动。本仓库只断言声明（`tests/shared/test_security_boundaries.py:10-27`），不验证围栏、不记录 hmz 版本。本轮未重读 hmz 源码 | **主要风险在 P2**：`tasks/` 已由 `forbidden_path`（`taic/code_agent.py:161`）禁止，不要再写入 `SECURITY_CRITICAL`。该常量仍缺 `tcenv/taskcontract/`、`tcenv/opspec/`、`tcenv/task.py`、`tcenv/distributions.py`。`report.py` / `taic/parse.py` / `taic/evidence.py` 按角色复核，不一刀切。补路径归 plan-06，不建设完整 TCB |

### 5.3 fixture 形式

- 固定 Metal 源码、输入向量和 manifest 放 `tests/support/golden/metal/fixtures/anti_hack/`（规划目录，当前不存在），经 `tests.support.hw_golden` 读取；不得直接放 golden 根目录。
- 每个 fixture 带独立期望输出或反例、适用合同、攻击类别、预期拒绝 stage/reason、`solver_set` 与 `at_risk_solvers`；不把「当前版本恰好接受」写成永久 golden。
- 每个攻击至少配一个 nearest-legit solver：尽量保留算法、shape/tile 特化和写出顺序，只修掉语义违规；它若触发同一拒绝理由即阻塞 enforce。
- 计划为公开 fixture 生成改 `attack_id`、重排 case、替换同类结构化输入公开 seed 的语义保持变体；攻击仍须按同一理由拒绝，合法对照仍须通过，防止 fixer 记住 ID、顺序或种子。
- 通用分布/缓存测试放 `tests/shared/`；MSL、Metal raw buffer 与 launch 行为测试放 `tests/backends/metal/`。support 只放 loader/helper，不放业务测试。
- 语料分开发集、冻结确认生成集和轮换生成集：修门、调阈值和选择补丁只用开发集；确认结果若导致改门，该批转为开发集并另生新批。定稿后所有历史批次仍永久回归，轮换集只增补覆盖。私密 holdout 是**未解决的可选项**：需要 plan-06 的 host-owned 存储、读取回执与「agent 不可读」的最小离线 fake 设计；在这些都不存在时，公开 fixture 不冒充秘密，本文也不承诺秘密确认集。

## 6. hacker–fixer–solver 三角循环的离线落地
| 角色 | 默认实现 | 每轮产物 |
| --- | --- | --- |
| hacker | 手写 MSL 变体、raw-output 向量、fake timing runner、确定性输入 mutator | 一个独立语义反例和最小复现，不靠模型自报「这是作弊」 |
| fixer | 人审后的最小 `tcenv` 修改，先加会失败的回归，再实现检查；每轮只改一个可归因语义门 | 事前 `should_reject / should_still_pass / at_risk_solvers` 预测，以及实际 stage/reason、协议/缓存版本变化和攻击全集回归结果 |
| solver | CPU 合法输出集、任务独立 oracle、`naive_msl` 与合法优化变体（GPU 部分需授权） | 合法解仍通过的证据，以及误拒反例 |

循环规则：

1. hacker 在旧版被挡住：只有目标反例已由 `expected_stage` 实际求值，且拒绝理由符合 `expected_reason`，才记为检出并固化回归；更早的编译失败、超时或 harness crash 不算。在旧版通过：登记一个确认漏洞，再进入 fixer；未执行单列。
2. fixer 只针对语义缺口，不通过放宽参考、删除反例、扩大容差或隐藏失败让测试变绿；评测判分修改按 P2 人审处理。结果须与事前 `should_reject / should_still_pass / at_risk_solvers` 对账，未预测 solver 回归阻塞 enforce。
3. solver 失败时先分类为「合法解误拒」「solver 本身错误」「参考/输入 harness bug」。未查清前不得 enforce，也不得为追求攻击检出率删掉合法解；攻击检出提升不能抵消合法解误拒、未执行或错误 stage/reason。
4. solver 必须包含边界内合法近似、不同写出顺序、形状常量/循环展开、多段实现和正常尾块处理；只测 naive 会漏掉对优化核的误杀。
5. 每轮同时跑全部历史攻击与 solver，再跑现有 CPU 套件；新攻击进入开发集，旧攻击不得因修复新攻击而回归。
6. 每个 fixture 先按互斥最终状态 `executed`（完成目标检查）/ `not_executed` / `harness_error` 对账，必须满足 `registered = executed + not_executed + harness_error`；指标再按类别报 `accepted_attacks / executed_attacks` 和 `rejected_legit / executed_legit`。必需 fixture 的后两类阻塞对应套件验收，不得过滤分母变绿；固定语料上的 0 只表示这些反例已覆盖，不外推为未知攻击的 0。
7. 计分后审计若给已 `eligible` 候选补出独立反例，本轮选择必须把它视为非 `eligible` 且停止使用其加速结果，不能只记 warning；随后转成 fixture 走旧版/新版回归。
8. 反馈给模型的内容只保留失败关卡和可公开反例；私有种子、答案与（若将来存在的）holdout 输入不回显。是否允许读取相关文件由 plan-06 决定；在此之前本文不把公开派生种子写成秘密。

## 7. 验收标准与命令

### 本批：审阅卡完成条件

①仍默认关闭，且被核对为第二种子复查；② `confirm_batches=3` 仍默认开；③ holdout 仍未跟踪、未接入。tail/shape、no-op/回退、退化题的 fixture 规格仍是候选，不写入 `EvalReport`。本批不要求 runner、未写检查或 `suite_id` 已经通过。

### 后续另获实现授权后的退出条件

下面的 runner、未写判定、`suite_id` 行为与测试命令都不是本批验收。实现另获授权后才成为可运行退出条件。

- fresh-input 反例优先通过：只在预热/首轮计算、只算首末轮组、重放旧输出三类 fake runner 都必须在新输入上失败；合法每次重写全部输出的 runner 必须通过。每个必需攻击都有独立反例、预期 stage/reason 和最近合法 solver；未执行、前置编译失败或 harness error 不算目标挡获，合法解误拒阻塞 enforce。
- 未写判定按原始字节覆盖四类 fixture：参考=毒化/非毒化 × 合法写/未写；整数 `-1`、浮点 NaN 的合法写入不误拒，真正跟随两种填充的元素才判未写。
- `mode=full` 且未传 `cases` 才能写全部 `task.cases`；缩减 `cases` 的 `eligible` 只代表实际子集，`suite_id` 为计划新增。
- 报告分别列攻击、合法解、未执行和 harness 故障；公开 dev/frozen_confirm/rotation 之外不声称秘密 holdout。

```bash
# 后续另获实现授权、且 anti_hack 文件已经新增之后才可运行。
# test_anti_hack_correctness.py、test_anti_hack_bench.py 与 fixtures/anti_hack/ 当前不存在。
# test_distributions.py 与 test_structure.py 已存在。默认不花钱、不用 GPU。
uv run pytest -q -n 0 tests/shared/test_structure.py
uv run pytest -q -n 0 tests/backends/metal/test_anti_hack_correctness.py tests/backends/metal/test_anti_hack_bench.py tests/shared/test_distributions.py
uv run pytest -q -m 'not gpu and not llm and not slow'
```

失败先读 `logs/pytest/triage/triage_summary.md` 及 handoff，不先凭完整日志改门。

### 后续验收（本批不做）

缓存协议、真实 Metal、frozen GPU 攻防与付费攻击生成均另开任务；相应测试文件和 GPU 命令不是本批完成条件。真实 Metal 仍须显式授权，且不运行真实挂死测试或设置 `TCENV_ALLOW_GPU_HANG_TESTS`。

## 8. 风险与回滚
| 风险 | 控制 / 回滚 |
| --- | --- |
| 过紧门误杀合法近似或形状特化 | 先 diagnostic；solver 与攻击同审；按单个检查 revert，不删除语料 |
| 独立 oracle 自身错误 | 双实现短尺寸对拍、非平凡性/可表示性检查；判为 harness bug 并隔离任务，不靠放宽容差掩盖 |
| 计后检查增加评测时长 | 放在计时窗外；quick 可降级但必须标注，full 保持完整；超时单列基础设施/候选状态，不伪记攻击检出 |
| 改计时输入分布破坏历史可比性 | 首阶段只做计后复验；不同轮组输入另走协议评审和 plan-02 效果实验 |
| 回滚后新旧缓存混用 | 恢复完全一致的旧语义才可读旧协议缓存，否则再 bump cache schema；保留审计文件，不手改旧报告 |
| 「未写」硬门误伤合法 `-1`/NaN 输出 | 判定要求按原始字节确认毒化轮与零填充轮同时跟随各自填充；参考值为 `-1`/NaN 不构成例外。只看单轮的实现不得 enforce，先按 diagnostic 报告 |
| 「私有」种子实际可读 | 公开派生只当复现机制，不当秘密；秘密保存、文件权限与进程边界交 plan-06 |
| mock 与真实 Metal 行为偏离 | CPU 先行但不宣称集成完成；正式启用前做授权 GPU 攻防验证 |
| 对固定语料过拟合 | 保留轮换 seed、结构化生成集和公开轮换批；0 ASR 仅限已执行语料。私密 holdout 在落地前不得写成已有 |
| 缩减子集被写成完整验收 | 报告必须绑定 `mode`、是否传入 `cases` 与 case 清单；未传 `cases` 的 `full` 之外，`eligible` 只证明该子集。任何把 `evaluate(..., cases=[...])` 的 `eligible` 写成「全部 `task.cases` 通过」或「完整 acceptance」的报告即判验收失败 |

## 9. 与其他 plan 的边界
| 计划 | 本文交付 / 明确不做什么 |
| --- | --- |
| plan-02 效果实验 | 本文给攻击检出与 solver 误拒门；加速收益、搜索质量、成本 A/B 和统计显著性由 plan-02 负责 |
| plan-04 证据链 | 本文要求记录 suite、输入/计后复验语义并正确失效缓存；签名、receipt、源码证据绑定与防伪造由 plan-04 负责 |
| plan-06 注入安全 | 本文研究核语义与评测对抗 fixture；提示注入、文件/进程沙箱、secret 权限、self-check 伪造和模型运行时隔离由 plan-06 负责；**评测器、用例与期望输出这些评分文件的只读性与前后完整性同样归 plan-06**，本文的数值门不能替代它 |

信任登记、P1/P2 授权、observe 看板指标和文档治理均沿用各自计划；本文不允许模型借 harness-proposal 自动改容差、判分或验收协议。

## 10. 文献与变更索引

只作查阅，不再逐轮重述正文；论文数字均未在本仓库复现。

| 日期 | 影响位置 | 证据文件 |
| --- | --- | --- |
| 2026-10-25 | 文献对应的正文规则与边界 | `2606.08960_hacker-fixer-loops.pdf`、`2509.14279_robust-kbench.pdf`、`2609.24972_rrsi-regularized-harness-rsi.pdf`、`2608.10178_one-recipe-many-harnesses.pdf`、`2610.15150_harmony-harness-multiobjective.pdf`；其余见 `papers/导读.md` |
| 2026-10-26 | 当前决定、阶段、验收与跨 plan 归属 | `审核意见-plan-01至09.md` |
| 2026-10-27 | 新增离线 fixture 的文献落点 | Correctness Illusion、MKEvolve、DRTriton、KernelBench-Verified、Xe-Forge、KernelBenchX、CANNBot-CAKE 等，见 §11.1 |
## 11. 参考论文与文件
- `docs/可信harness/papers/2610.15150_harmony-harness-multiobjective.pdf`：§5.2 跨基准迁移、App. K.6（p.40）的 raw-feedback artifact / soft form / hard form 三层；本文只借「满足度量≠满足属性」与评分面保护，未复现其 benchmark。
- `docs/可信harness/papers/2609.24972_rrsi-regularized-harness-rsi.pdf`：§3、Appendix D/E 与 Limitations；本次仅迁移防适应性过拟合与可归因选择，未复现其 benchmark。
- `docs/可信harness/papers/2608.08311_ouroboros-reviewed-core-evolution.pdf`：§3、§5–§7 与 Limitations；本次仅迁移执行分账和审计后的验收规则。
- `docs/可信harness/papers/2608.10178_one-recipe-many-harnesses.pdf`：§3.2、Appendix A.2/A.4/A.7；本次仅迁移阶段化类型结果与双向可证伪预测。
- `docs/可信harness/papers/2606.08960_hacker-fixer-loops.pdf`：hacker/fixer/solver 三角；`docs/可信harness/papers/导读.md` 记录其 1,968 任务审计、323（16%）可 hack，以及公开 KernelBench exploit 语料 62%→0%，均为论文结果而非本仓库结果。
- `docs/可信harness/papers/2509.14279_robust-kbench.pdf`：更严内核正确性/性能验证与稳健测试对照；索引见 `docs/可信harness/papers/导读.md`。
- `docs/架构/评测.md:3,219-228,282-286`：判决边界、模式和三次启动；`docs/架构/搜索.md:288-316`：确认与发布门。
- `docs/可信harness/架构与理论.md` §7–§8：当前实现差距、证据伪造和 Goodhart 等失败模式。
- 核心实现：`tcenv/backends/metal/testing.py`、`sanitize.py`、`backend.py`、`runtime.py`、`tcenv/evaluate.py`、`tcenv/bench.py`、`taic/orchestrator.py`。
- 测试规则：`tests/README.md`、`tests/shared/README.md`、`tests/backends/metal/README.md`、`tests/support/golden/README.md`。
- 外部语料只读参考：`kda-260927-kda/hacking_example/README.md` 与 `kda-260927-kda/bench.py`；不执行其 CUDA/B300 命令，不复制其测量作为本仓库结论。

## 11.1 新增文献的离线落点（2026-10-27）

本节只补充 fixture 候选与证据来源。计后 fresh-input 仍是阶段 1A 的设计草案，本审阅卡不授权实现；也不授权真实 GPU、秘密 holdout 或付费攻击生成。每个 fixture 都必须同时保留合法 solver；子集即使得到 `eligible` 也只能报告该子集通过，不能写成完整验收。

| 离线 fixture 候选 | 要抓的失效与预期检查 | 文献来源（均为论文口径，未在本仓库复现） |
| --- | --- | --- |
| `H=1,3,7,256,1025` 的 tail/shape 候选档（按任务 contract 适配） | 规整形状通过但 tail-mask 读错、越界或污染归约；shape 覆盖另记 correctness fixture，不把阶段 1A 的计后 fresh-input 最小协议扩成换形状 | [Correctness Illusion](papers/2606.20128_correctness-illusion.pdf) §3.1、§4 |
| 容差过松与过严的双向 bug 注入 | 同时报漏检与误拒，记录误差分布和可行性间隙，不能只调一个全局阈值 | [MKEvolve](papers/2607.20501_mkevolve.pdf) 的容差校准与 Appendix D |
| no-op、未调用与框架回退 | 输出碰巧正确，但目标 kernel 未执行或实际走了库；用执行路径/调用计数而非文本风格判定 | [DRTriton](papers/2603.21465_drtriton.pdf)、[CudaForge](../paper/2511.01884_cudaforge-an-agent-framework-with-hardware-feedback-for-cuda.pdf) |
| 常数输出、退化题、假通过夹具 | 用输出退化、空比较、窄覆盖骗取高分；合法 solver 必须仍能通过，报告覆盖范围 | [KernelBench-Verified](papers/2607.16241_kernelbench-verified.pdf)、[CUDA Agent](papers/2602.24286_cuda-agent.pdf)、[GEAK](../paper/2507.23194_geak-introducing-triton-kernel-ai-agent-evaluation-benchmarks.pdf) |
| 跨 target 误继承旧报告（仅离线元数据 fixture） | 构造 target/kernel 绑定不符的旧结果，预期不得作为本 target 的验收或发布证据；真实基线重测与确认不在本批。CANNBot-CAKE §4.4 是 EngramGateFusion 案例：local oracle 与 official grader、48 与 40 个 vector core 的分数不可直接比较，只作“绑定环境”旁证，不是可照搬的跨 target 协议 | [KernelBenchX](papers/2605.04956_kernelbenchx.pdf)、[Correct-but-Slow](../paper/2607.04454_correct-but-slow-an-empirical-study-of-the-gpu.pdf)、[CANNBot-CAKE](papers/2610.31034_cannbot-cake.pdf) §4.4 |
| 快到不可能的异常候选 | 用独立物理下界或参考算法下界做 sanity check，异常结果转人审，不把自报 headroom 当证据 | [SOLAR](../paper/2606.26383_solar-ai-powered-speed-of-light-performance-analysis.pdf) |

[Xe-Forge](papers/2605.26118_xe-forge.pdf) 与 [CUDA Agent](papers/2602.24286_cuda-agent.pdf) 的测量路径权限证据归 plan-06 的 P2 基线材料/只读 fixture；本文不另建权限或评分 TCB fixture。

这些文献只增加可证伪的反例，不改变本批“公开离线攻击、合法 solver、CPU fixture”的范围；静态关键词不能替代数值与执行路径证据。

