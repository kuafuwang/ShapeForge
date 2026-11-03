# Evolving Agent Teams（EAT）调研：动态团队、持久经验与可信边界

状态：2026-10-30 只读调研备忘。本文依据**公众号 PDF、arXiv 原文、EAT 工程源码**，不把公众号转述当论文结论；没有运行模型、GPU/NPU 或 `taic`，没有安装依赖、执行任务或评测。本文是资料调研，**不改变当前 plan 的优先级、授权或代码**。

## 0. 一句话结论

EAT 最有价值的不是“多开几个 Agent”，而是把优化对象拆成：

- **团队配置** `T=(R,Π,M)`：角色 `R`、协作协议 `Π`、知识管理 `M`；
- **实际解**：kernel 或 policy；
- **持久文件系统**：learnings、演化历史、每代 artifacts，支撑跨代诊断。

它确实比 K-Search 更接近“随阶段改变工作方式”，但代码实现也说明：**监督者主要靠提示词与 Claude Code 生成团队配置，确定性部分只负责解析、启动、计时和记账**。主实验每个设置只跑两次、跨领域只跑一次，不能把曲线当成稳定效应量。

对当前仓库最有用的启发是：**先记录并复用已验证的决策证据，必要时改变提示/角色，而不是先增加常驻多 Agent 或自动知识库。** 这与当前未授权的 `plan-10` 同向；不意味着应直接移植 EAT。

## 1. 来源与下载状态

| 材料 | 位置/链接 | 状态 |
| --- | --- | --- |
| 公众号 PDF | `docs/一个公众号的论文解读/让 GPU 优化 Agent 灵活就业！伯克利开源 Evolving Agent Teams：从零生成超越生产级的 GPU 内核，比 KDAgent 省钱！.pdf` | 本地已有，22 页，2026-10-29 |
| 原始论文 | [arXiv:2610.36848](https://arxiv.org/abs/2610.36848) | 已下载：`docs/paper/2610.36848_evolving-agent-teams.pdf`，17 页；arXiv v1，2026-10-27；SHA-256 `a2a15b5f0a0aa36797452a2413c59077b74cd3d5966fea6ea39711adbcad0a4d` |
| EAT 工程 | [caoshiyi/EAT](https://github.com/caoshiyi/EAT) | 已下载到 `/Users/kuafu/CLionProjects/dsl_for_llm/EAT`；commit `737e1665eeeffd3fce35d319174f88ea4613d7aa`，Apache-2.0；源码快照无 `.git`，见 `SOURCE.md` |
| 任务数据 | [eva98/eat-gym](https://huggingface.co/datasets/eva98/eat-gym) | 公开数据集；只核对 README/API，没有下载大文件、没有 setup |
| 演化轨迹 | [eva98/eat-evolution-traces](https://huggingface.co/datasets/eva98/eat-evolution-traces) | 公开数据集；只核对 README/API，没有下载 transcripts/kernels |
| 生成的 kernels | `caoshiyi/eat-kernels` | **当前 API 返回 Not Found**，不能当作已下载或可核代码；README 中链接的该仓库未能访问 |
| RoboDojo policy | [caoshiyi/eat-robodojo-policy](https://github.com/caoshiyi/eat-robodojo-policy) | 只核对 API：存在，commit `a905684d83ef8e0024ee9f658b4b7f68bb34b833`，Apache-2.0；未克隆，避免扩大研究范围 |

本次实际保存的是论文和 EAT 主工程；没有下载 HuggingFace 大文件、模型权重或 RoboDojo 数据。`eat-kernels` 的 404 需要明确记录，不能把 README 的“everything is released”当作所有产物都已可访问。

## 2. 论文到底做了什么

### 2.1 两个层次的优化对象

论文把开放程序优化写成：给定目标描述、代码库和正确性约束 `C`，在 token 预算内寻找可执行程序 `s`，使 `F(s)` 最大。传统系统主要改 `s`；EAT 同时改变团队配置：

| 配置轴 | 含义 | 例子 |
| --- | --- | --- |
| `R` roles | 谁负责什么 | 纪律员、性能剖析员、知识整合者、并行调试员 |
| `Π` coordination | 角色如何交互 | 串行研究→实现→剖析；并行隔离分支；混合 |
| `M` knowledge | 经验如何保存/共享 | `LEARNINGS.md`、`SYNTHESIS.md`、演化历史、per-team artifacts |

监督者不是直接写 kernel 的“强模型”，而是根据上一代的失败模式设计下一代团队配置。论文声称角色不是从固定池中选择，而是现场生成。

### 2.2 双层闭环

```text
上一团队的输出、官方分数、成本、记录
        ↓
监督者 REFLECT：定位停滞原因
        ↓
监督者 EVOLVE：改角色 / 提示词 / 阶段 / 知识管理
        ↓
启动下一团队（可串行、并行）
        ↓
团队写代码、跑编译/剖析/自测
        ↓
宿主运行官方 eval，记录 SCORE 与成本
        ↺
```

持久文件系统 `K` 保存三类记录：

- **学习文档**：经验观察、技术结论、失败记录；
- **团队产出**：每代 kernel、角色记录、分析报告；
- **演化历史**：监督者为什么重组、改变了什么、预期解决什么。

### 2.3 MLA 案例：重组解决的是流程问题，不是单纯模型问题

论文 Figure 1 的长期轨迹中，几个典型变化：

| 阶段 | 观察到的失败 | 团队变化 | 论文报告的结果 |
| --- | --- | --- | --- |
| V1→V2 | 放弃 WGMMA、转向容易的标量路线；多变量提交 | 增加纪律员；每次提交只改一处；引入 `LEARNINGS.md` | 首次得到正确 WGMMA kernel |
| V5→V6 | 多代卡在同一个 P@V WGMMA 根因 | 调试员与主优化师并行、隔离分支 | 确认一条路线是死胡同；主分支 12 次提交，+44% |
| V7→V8 | V4 的关键建议没有进入公共经验 | 增加知识整合者；引入 `SYNTHESIS.md`，要求后续团队阅读 | 首次通过 P@V WGMMA 独立测试 |
| V9→V10 | 架构变了，旧瓶颈分析过期 | 重新 profiling，再用性能驱动优化师 | 9 次提交 +32%，Split-K 重调 +4.1% |

可迁移的是**“根据已观测失败改变工作方式”**，不是“监督者角色天然有效”或“所有平台期都必须继续烧钱”。

## 3. 结果、口径与限制

### 3.1 主实验

| 任务 | GPU/语言 | 负载 | 基线 | 论文报告 |
| --- | --- | --- | --- | --- |
| FA4 | B200 / CuTe DSL | 24 | FlashAttention-4 | 同预算 EAT 优于 Claude Code；没有完整统一的绝对收益表 |
| GDN prefill | B200 / CUDA | 100 | FlashInfer | 同预算 EAT 优于 Claude Code；扩展预算平均 4.06×，最差负载 0.40×，37/100 低于基线 |
| MLA decode | H100 / CUDA | 47 | FlashInfer | 同预算 EAT 约 0.74×；扩展预算平均 1.21×，36/47 胜出，剩余 11 个 0.86–0.99× |
| GEMM+AllReduce | H100 / CUDA | 5 | CUTLASS+NCCL | 同预算约 1.18×；扩展预算 1.56×，接近 ThunderKittens 1.54× |

**重要的基线口径**：MLA Figure 1 的 `1.0×` 是排除 `plan()` 开销的 FlashInfer `run()` 阶段；生产 FlashInfer 缓存 `plan()` 后，实线基准为 `0.77×`。论文最终 `0.96×`（相对排除 plan 的基线）对应生产基线的 `1.21×`。因此“超过生产级”依赖具体计时定义，不能只摘一个 1.21。

### 3.2 与其他系统的对比

MLA decode 上，论文 Table 2：相同 Claude Opus 4.6、相同 $800 模型推理预算、两次运行：

- CudaForge：0.448× / 0.283×，标量/SIMT 路线；
- K-Search：0.602× / 0.598×，WMMA 路线；
- EAT：0.721× / 0.685×，WGMMA 路线。

论文将差距归因于系统最终收敛的技术路线不同，而不只是“更多 Agent”。这是有启发性的对照，但**每个设置只有两次运行**，不能当成大样本证明。

固定四角色团队在 MLA 上为 0.31×，Codex 单 Agent 为 0.28×，EAT 为 0.74×；固定团队与 EAT 的模型相同但角色固定，说明**单纯加角色可能因信息壁垒更差**。不过这些是单次/少量运行的结果，统计显著性有限。

### 3.3 跨领域结果

论文报告：

- AMD MI350X KDA prefill：相对 vLLM 平均 3.01×；73 次迭代、186 小时、约 $3.0K；相对 Triton FLA 核 7.3×。
- B200 KDA prefill：相对 FlashKDA 3.44×；不是从零，而是先移植已有强核，初始约 1.96×。
- Apple M4 Pro MoE offload：朴素引擎 13.4 tokens/s，初次论文口径最终 27.3 tokens/s；第二次运行后重测为 30.3 tokens/s。README 又给出 GGUF/MLX 的不同比较，不能混成同一数字。
- RoboDojo：42 个任务，held-out seeds 0/2 得分 62.1，超过当时榜单 55.3；$472（不含运行时 LLM 调用）。

这些是不同硬件、不同起点、不同评分函数的案例，**不能合成一个“通用优化能力提升”数字**。B200 与 MoE 还明确是从已有强实现开始，不是从零生成。

### 3.4 论文自己承认的限制

- 主设置每个跑两次，跨领域设置每个跑一次；
- 最佳两次结果的选择会引入选择效应；
- 成本-性能 Pareto 分析和更嘈杂环境的受控研究尚未完成；
- 多次架构变化、知识库复用与监督者干预同时发生，单因素归因并不充分。

因此本文只把结果当作**设计假设与可复核材料**，不把公众号的“全面领先”“演化才是核心护城河”当已在本仓库成立的结论。

## 4. 工程实现：论文机制在代码里长什么样

### 4.1 文件与职责

本地工程很小，主要是 shell、Python 和提示词：

| 文件 | 实际作用 |
| --- | --- |
| `SKILL.md` | 面向 Claude Code 的安装/运行说明；定义 `eat/task.md`、`eat/eval.sh`、`eat/env.sh` |
| `agent_team/SUPERVISOR.md` | 监督者行为哲学：必须 REFLECT→EVOLVE→LAUNCH→MONITOR，不接受团队自报的“硬件极限” |
| `agent_team/launch_stateless.sh` | 主循环；调用监督者、提取配置、记录成本、启动团队、跑官方 eval、每轮提交 |
| `agent_team/parse_team_config.py` | 从监督者输出解析角色 prompt、timeout/model、阶段计划 |
| `agent_team/launch_team.sh` | 按阶段并行/串行启动 Claude Code；运行官方 eval；汇总成本和团队输出 |
| `agent_team/extract_result.py` | 从 Claude JSON 输出取 `result` 文本 |
| `ops/verify_iter.py` | 对某轮 commit 在 worktree 里重跑 3 次官方 eval，并写验证记录 |
| `ops/curve_loop.sh` | 每 5 分钟生成曲线；新 best 后调用独立重测 |
| `cleanroom/sandbox.sh` | bubblewrap 隔离；限制可见路径、网络、GPU、环境变量 |
| `cleanroom/allowlist_proxy.py` | 只允许 Anthropic/Claude API 域名，记录拒绝请求 |
| `cleanroom/audit_clean_traces.py` | 扫描 transcript 中的 Web、网络命令、越界路径、禁止引用、held-out 读取 |
| `cleanroom/audit_code_overlap.py` | 检查候选源码与禁止来源的行重合 |
| `ops/fetch_task.sh` | 从 EAT Gym 下载任务起始状态并运行 setup |

### 4.2 监督者与团队的实际分工

**监督者**：

- `launch_stateless.sh` 用 `claude --print` 调用；
- 系统提示由 `SUPERVISOR.md` + 任务 `eat/task.md` + 迭代预算/上一团队输出构成；
- 输出必须是 `---BEGIN TEAM_CONFIG---` 块，包含角色、prompt、workflow、success criteria；
- 监督者被禁止 `Agent,Write,Edit,NotebookEdit`，但仍用 `--dangerously-skip-permissions` 调用 Claude CLI；
- 可用 `--persistent` 保留同一监督者会话，也可用 `--stateless` 每次新会话。

**团队**：

- `parse_team_config.py` 从自由文本提取角色，最多 `max_team_size`；
- 每个角色得到独立 prompt，并可声明 timeout/model；
- 同一阶段的角色并行，不同阶段串行；
- 角色实际是新的 Claude Code session，不是本仓库的 hmz `kernel_once`；
- 官方 eval 在团队结束后由宿主运行，输出 `SCORE: <x>`；
- 成本来自 Claude JSON 的 `total_cost_usd`，监督者成本单独累计；
- 每轮执行 `git add -A` 和 `git commit --allow-empty`。

**确定性验证**：

- `verify_iter.py` 在指定 commit 的 worktree 中重跑三次 `eat/eval.sh`；
- `curve_loop.sh` 在新 best 时调用它；
- 这是“提交后重测”，不是搜索候选替代确认门；也没有本仓库的 `confirm_top/noise_z/confirm_batches` 语义。

### 4.3 可核的 harness 优点

1. **官方分数与 Agent 自报分离**：最终分数由宿主 `eval.sh` 给出，不直接相信角色文本。
2. **角色配置是显式工件**：每次监督者重组都有 `team_config.md`，便于审计“为什么改角色”。
3. **阶段与并行由宿主调度**：不是让 Agent 自己决定是否并行，避免角色之间失控。
4. **重测与曲线分离**：可以按迭代 commit 重测，记录验证时间和分数。
5. **clean-room 是实际代码**：有 bubblewrap、只读白名单、网络 allowlist、transcript 审计、源码重合检查。
6. **成本有账**：监督者和团队分开记，并记录官方分数。

### 4.4 与本仓库边界的冲突与风险

- `claude --print --dangerously-skip-permissions` 不是本仓库可接受的路径级受限沙箱；不能直接复制到写核会话。
- 团队共享任务仓库，角色可写代码、运行命令；官方 eval 的隔离主要靠 `eat/task.md` 规则、宿主重测与可选 cleanroom，不是所有运行都强制 VM 隔离。
- 每轮 `git add -A` 提交，会包含 Agent 产生的所有变更；本仓库有并发会话和未跟踪工作，**绝不能照抄**。
- `TEAM_CONFIG` 是监督者生成的自由文本；`parse_team_config.py` 能解析格式，但不能验证提示内容的授权、权限或真实性。
- 知识管理主要是文档继承，不是本仓库要求的 `verified/reported/stale/retired` 过滤；学习文档可能把未验证断言带给下一代。
- “监督者永不接受平台期”的提示语与我们的 `patience`、`stop_at_roofline`、预算门不同；不能用“继续重组”绕过停止或预算。
- cleanroom 只在显式配置时使用；不是默认安全能力，且它的网络允许列表只覆盖 Anthropic，不是通用 harness 沙箱。
- 代码库中的官方 eval 是否真正不可改，需要逐任务检查；不能把“有 eval.sh”当作评分材料天然在 TCB 外。

## 5. 对本仓库 harness/flow 的启发

### 5.1 已经有对应物，不重复做

| EAT 的概念 | 本仓库已有 | 判断 |
| --- | --- | --- |
| 官方 eval 与 Agent 自报分离 | `tcenv` 外层评估、`EvalReport`、自测不进种子 | 已有，保持 |
| 同一任务内配对测量 | `tcenv/bench.py` 交错候选/基线、median/CV | 已有，不需要新增 A/B |
| 搜索最优与确认胜者分开 | `confirm_top`、`confirm_batches=3`、`required_gain` | 已有；EAT 的迭代重测不是等价替代 |
| 阶段化任务/技能 | `SessionBrief`、seed report、`taic-tools`、profile/hw feedback | 已有部分；不必复制完整团队框架 |
| 跨会话经验 | `NOTES.md`、`FEEDBACK.md`、`experience.md`、plan ledger | 已有记录，但准入和路由仍是缺口 |
| 停止条件 | `patience`、`max_rounds`、预算、`stop_at_roofline` | 已有；EAT 的“永不接受平台期”不能覆盖它们 |

### 5.2 值得借鉴的最小点：决策证据与角色变更可审计

EAT 的监督者不是简单的“每轮换模型”，而是把**上一代的失败模式**变成下一代的团队配置。对当前仓库，最小、可信的映射是：

1. 先把每轮真正确立的证据写清：赢、噪声内并列、方向被否证、实现失败未计时；
2. 下一场只在有证据时改变提示/角色，例如“需要并行隔离调试”或“架构改变后重新 profiling”；
3. 角色变更本身记录 provenance：依据哪一轮、哪份报告、哪条失败；
4. 保持 recipient/scope/trust 不变，角色变更不自动授权、不改变评分材料；
5. 先做 fake session/CPU fixture，不证明“实际搜索变快”。

这与 `plan-10` 的研究候选相同，不应直接复制 EAT 的监督者 Agent。

### 5.3 持久经验要学习“传递已验证事实”，不是“传所有文档”

EAT 的 V7→V8 案例很有用：V4 的关键建议存在，但没有进入共享 `LEARNINGS.md`，后续团队看不到；这提示我们应区分：

- 已验证事实与测量证据；
- 方向性建议；
- 未验证断言；
- 过期或不适用的经验。

对当前仓库，`plan-07` 的 staging 过滤仍应优先：不能让 `reported`/stale/retired 或 developer tool request 进入写核 TASK。EAT 的“强制读 SYNTHESIS.md”不能替代过滤；文档越大不代表知识越可信。

### 5.4 并行可以隔离失败，但不是免费收益

EAT 的 V5→V6 说明：主优化和难题调试并行，可以避免一个根因卡住整条路线；固定四角色却可能因信息壁垒更差。对本仓库的映射应是：

- 只有明确、独立、可合并的工作才并行；
- 每个分支有独立 workdir/写权限、独立预算和产物；
- 主分支不被调试分支的未完成修改污染；
- 最终仍由宿主验收，不因并行就降低确认门；
- 不把多角色当作独立 reviewer 或多模型共识。

### 5.5 clean-room 与验证思路可借鉴，不能直接搬实现

EAT 的实际 clean-room 代码值得参考：

- 只读允许的参考源码；
- 网络只允许 API；
- transcript 扫描网络/路径/禁止来源；
- 源码行重合检查；
- 每轮 commit 后独立重测。

但当前仓库的 `local=ALL` 只允许 workdir/临时目录、仓库只读，且不是 VM；不能把 EAT 的 `--dangerously-skip-permissions` 当作等价实现。真正可迁移的是**“评分材料、参考源码、候选写入面分离，事后有审计”**这条原则。

### 5.6 成本与基线口径必须先写清

EAT 的主要对比是 token/美元预算，而不是 GPU 小时；扩展任务还同时有 186 小时、$3.0K 等成本。对本仓库：

- 记录 token/美元、GPU 秒、Agent 活跃时间、wall 时间，不能混算；
- 记录基线是否含 plan/preprocessing/cache；
- 记录起点是 naive reference 还是已有强实现；
- 记录主实验运行次数与是否取最好；
- 不把单次跨领域结果当成本地效果。

这与 `plan-08` 的成本口径旁注一致，不新增周报或付费实验。

## 6. 与 ForgeKernel / K-Search 的关系

| 系统 | 主要改变什么 | 主要记忆 | 与本仓库最有用的对照 |
| --- | --- | --- | --- |
| K-Search | 在固定 world-model/search tree 中选择下一步 | 结构化决策树、节点与方案 | `plan_ledger` 的结构化记录；代码公开，含 MLX 适配 |
| ForgeKernel | 重开架构、参考晋升、配方认证 | 里程碑、硬件配方、no-go | 架构决策与测量认证；官方代码未找到，CuTe DSL/NVIDIA 为主 |
| EAT | 重组角色、协作和知识管理 | learnings、演化历史、per-team artifacts | 根据失败模式改变工作方式；工程公开，但默认权限/提交方式不适合本仓库 |

三者都反对“只在固定架构里调参”，但**EAT 不是 ForgeKernel 的代码实现，K-Search 也不是 EAT 的替代品**。不能把三个系统的数字拼成一个排行榜。

## 7. 当前建议与不做事项

### 可登记为后续研究候选（未授权）

1. 在 `plan-10` 的离线规格里增加一条对照：普通 history vs. 带证据分类的决策记录，检查下一场 TASK 是否准确继承已验证事实。
2. 为“角色/提示变更”设计 provenance 字段：`source_round`、`source_report`、`failure_mode`、`scope`、`expires`；只作为规划字段，不现在实现。
3. 评估是否需要**受控的隔离并行调试**：先在 fake session 上定义分支产物、冲突和合并规则，不启动真实多 Agent。
4. 对 EAT clean-room 的“参考/评分材料/候选写入面分离”做只读对照，保留本仓库路径级沙箱和人审，不引入默认全权限。

### 当前不做

- 不运行 EAT、EAT Gym、RoboDojo 或任何 GPU/NPU/模型实验；
- 不安装 Claude Code 插件、依赖、驱动、模型或数据集；
- 不复制 `--dangerously-skip-permissions`、`git add -A`、共享工作目录或自动提交；
- 不新增常驻监督者、第三委员会、自动知识库、自动 overlay 或自动团队演化；
- 不把 EAT 的“平台期必须重组”当作绕过 `patience`、预算或确认门的理由；
- 不把 EAT 的小样本曲线、跨领域数字或 MoE/AMD/B200 结果当本仓库效果；
- 不因为 `eat-kernels` 链接存在就声称已下载全部 kernel。

## 8. 资料索引

| 类别 | 本地位置 |
| --- | --- |
| 本调研 | `docs/可信harness/Evolving-Agent-Teams-调研.md` |
| 论文 | `docs/paper/2610.36848_evolving-agent-teams.pdf` |
| EAT 工程 | `/Users/kuafu/CLionProjects/dsl_for_llm/EAT` |
| EAT 源码快照信息 | `/Users/kuafu/CLionProjects/dsl_for_llm/EAT/SOURCE.md` |
| K-Search 对照论文 | `docs/paper/2602.19128_k-search-llm-kernel-generation-via-co-evolving-intrinsic-world-model.pdf` |
| ForgeKernel 对照论文 | `docs/可信harness/papers/59558_ForgeKernel_Self_Evolvin.pdf` |
| EAT Gym / traces | 仅保留 HuggingFace 链接，未下载大文件 |
| eat-kernels | API 404，未下载；保留待复核状态 |

本文件只是调研记录。若以后要把某一条接入 plan 或实现，应另行人审并明确离线验收、权限和预算；本文不自动修改 plan。
