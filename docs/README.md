# ShapeForge 文档入口

ShapeForge 是面向多后端的算子生成、shape portfolio 精确分派与证据化性能治理系统。候选可以拥有算法、融合、Tiling、Host Tiling 与 ExecutionGeometry；harness 拥有任务合同、ABI、安全、测量边界和发布证据。当前生产目标是 Apple GPU / Metal；清微 NPU 只有离线模拟，不能宣称真机 correctness/performance。

Triton 只是 preserved v1 的题面格式和可选对照，不再是整个项目的边界。当前主线是 OpSpec v2、候选自有实现、stable variant、exact one-of-N dispatch、独立 fallback 和 fresh confirmation。

模型负责写内核源码（kernel text）。本仓库的程序负责编译、查安全、对答案（差分测试，differential test）、计时，并决定留哪一版。模型看不见显卡，也不能自己宣布写对了或更快了。

能对外说的倍数（published speedup），要四个条件同时成立：搜索已经结束、换一批输入重新测过、源码哈希对得上，而且计时没有噪声干扰。搜索过程里看起来最快的那次，是另一个数（search best），不能引用。权威记录在 `runs/` 里该次搜索的 `result.json`，还要过 `experiments/common.py` 的 `conclusion_of_run` 判定。下面各篇里的对照表是写作时的快照，不是最新结论。

默认一次新搜索大约 15 美元（预设名叫 `paper`）。初始稿和第 1 轮由 DeepSeek 的 `deepseek-flash` 写，第 2 轮起轮到月之暗面的 `kimi-k3`，之后按 `(r - 1) % 2` 轮流。只配一家的密钥不够。`--preset base` 才读 `taic.toml` 自己的上限，那是 50 美元。

写内核走本机的 `hmz` 启动器（`uv tool install hmz`），由它调用 `pi`；缺 `hmz` 时 `taic run` 在启动前拒绝（`docs/架构/hmz.md`）。

| 命令 | 会花钱 | 会占用这块同时驱动显示器的显卡 |
| --- | --- | --- |
| `taic run` | 会 | 会 |
| `taic resume` | 会 | 会 |
| `uv run python -m taic.cli brief` | 不会 | 不会 |

只看摘要用：

```
uv run python -m taic.cli brief
uv run python -m taic.cli brief RUN_DIR
```

每一轮都是一场新的 `hmz` 会话（session）：模型拿到的是这一轮新写的任务书，不是上一轮的对话。这一轮写错了，评测不合格的反馈会写进下一场任务书，不留在同一段对话里，也不另发规划请求。两个委员会（committee，一组负责一轮写内核的模型配置）仍轮流。模型相同也换一场。

## 这几句是常识

| 词 | 在这个仓库里指什么 |
| --- | --- |
| 题目 | 当前是 OpSpec v2 任务合同、reference、已登记 baselines 与 tolerances；preserved v1 可以使用 Triton 题面，主环境不拿 Triton 来编译 |
| 内核/候选 | 交给目标后端的实现。候选可拥有算法、融合、Tiling、Host Tiling 与执行几何，不限于 Triton 直译 |
| Shape portfolio | 候选规划 coverage buckets，多个稳定 variant 按 exact dispatch 选择；未覆盖输入由独立通用 fallback 处理 |
| 评测（`tcenv`） | 编译、查接口、对答案、计时。它不写程序 |
| 搜索（`taic`） | 安排模型写、看成绩、只留下又安全、抽查算对、而且明显更快的一版 |
| 种子 | 下一轮接着改的那一版合格程序 |
| 确认 | 搜索结束后的复测。对外倍数来自这里 |
| 上界 | 用这道题的运算量和这台机器测到的峰值，估一个「按当前算法最快大概多久」。它不是物理极限 |

算对是抽查，不是证明。留下的那一版可以比现成库更慢。死循环仍可能把电脑冻住。这些边界写在 [诚实说明.md](诚实说明.md)。

## 当前主线状态

- R0–R6、G0、A1–A6 和 shape-portfolio P0–P2 已完成；P3-A 已完成。
- P3-B 仍 blocked：paper GEMM 通过，small GEMM 尚未稳定满足逐 shard 1.10 与 fresh timing 门。
- 清微 NPU-P0 仍 blocked；模拟证据不解除真机阻断。
- 权威实施计划：[架构/shape-portfolio实施计划.md](架构/shape-portfolio实施计划.md)；跨硬件任务合同与迁移计划：[架构/跨硬件算子生成分工_9aef8d3a.plan.md](架构/跨硬件算子生成分工_9aef8d3a.plan.md)。
- 历史文档中的 Triton-only、逐题 LoweringProfile、`@variant` 或“计划未实施”描述，只代表写作时的阶段，不是当前 ShapeForge 的定位。

## 每篇在回答什么

先读这三篇，其余按需要查。

| 文档 | 回答什么 |
| --- | --- |
| [诚实说明.md](诚实说明.md) | 现在哪些话不能说，每个数字到哪核对。 |
| [架构说明.md](架构说明.md) | 总览和索引，写搜索、评测、看板怎样分工、常量和命令。逐题、逐进程细节在 `docs/架构/`；两者冲突时，以细节文档和代码为准。 |
| [维护决策.md](维护决策.md) | 影响数据/协议、但不需要每次改代码的决定，新条目在最上面。 |

### 架构细节（进程分工）

| 文档 | 回答什么 |
| --- | --- |
| [架构/题目.md](架构/题目.md) | 一道题目带着什么编译约定，缓冲区两头的保护带是什么。 |
| [架构/评测.md](架构/评测.md) | 一次评测怎样判决。 |
| [架构/搜索.md](架构/搜索.md) | 一次搜索怎样循环，什么数字可以发布，怎样续跑。 |
| [架构/数据层.md](架构/数据层.md) | 一次搜索留下什么、SQLite 索引怎样建、`taic query` 怎样查、协议版本/成本/保留期。 |
| [架构/可观测.md](架构/可观测.md) | 实验怎样分组，看板怎样读。 |
| [架构/hmz.md](架构/hmz.md) | `hmz` 是别人写的启动器。搜索写内核只走它，再由它调用 `pi`。`--kernel-agent off` 拒绝启动。 |
| [架构/执行运行时与设备扩展.md](架构/执行运行时与设备扩展.md) | 当前的执行运行时：本地/远程评测、`execution.json`、outbox、epoch、自检桥、P2 静态/隔离动态验收与设备 backend 边界。 |
| [架构/多厂商NPU评测通信抽象-plan.md](架构/多厂商NPU评测通信抽象-plan.md) | 未来多厂商 NPU infra 的通信抽象：统一 gRPC 控制面、厂商 driver、工件/能力/计时证据与验收；未实现。 |
| [架构/构建执行与证据语义升级-plan.md](架构/构建执行与证据语义升级-plan.md) | R0–R6 已完成：构建/评测身份分账、schema 4、fresh 三方确认与唯一发布谓词；历史证据不可改写。 |
| [架构/跨硬件算子生成分工_9aef8d3a.plan.md](架构/跨硬件算子生成分工_9aef8d3a.plan.md) | G0、A1–A6 已完成：OpSpec v2 标准任务入口、Track/CanonicalBundle、16 题迁移与旧逐题 lowerer 退役；shape portfolio 后续计划见下。 |
| [架构/shape-portfolio实施计划.md](架构/shape-portfolio实施计划.md) | 当前主线：候选 ShapePlanner、stable variants、exact dispatch、独立 fallback、未见分片验证与性能治理；P0–P2/P3-A 完成，P3-B blocked。 |
| [架构/远程评测模拟验收-plan.md](架构/远程评测模拟验收-plan.md) | Phase 2.5 历史验收证据与边界；旧 SSH 数据面和 runner 已删除，当前远程评测见执行运行时文档。 |

### 一条内核与一道题的具体问题

| 文档 | 回答什么 |
| --- | --- |
| [conv2d约束排查.md](conv2d约束排查.md) | 2026-10-20 查 conv2d 为什么过不了加速比 1。torch-mps 跑的是 Winograd。哪些约束合理、哪些不合理。 |
| [hardware_features.md](hardware_features.md) | 论文归功的硬件特性，在这颗苹果芯片上怎么认。检测器只判断「在不在」，不判断「快了多少是因为它」。 |
| [算子工程师边界.md](算子工程师边界.md) | 这条路替掉了哪些人手，哪些仍要人做。 |
| [直调与hmz取舍.md](直调与hmz取舍.md) | 写内核只走 `hmz`。`taic/llm.py` 的直调留给实验脚本。 |

### 可信 harness 与反馈（P2）

| 文档 | 回答什么 |
| --- | --- |
| [可信harness/README.md](可信harness/README.md) | 这个方向的文档索引与九份独立角度的 plan（plan-01…09），以及三条当前现状。 |
| [可信harness/架构与理论.md](可信harness/架构与理论.md) | RSI 外层循环、证据/harness 两条流水线、权限梯度、信任、双来源、影子与 P2 提升的理论和流程图。 |
| [可信harness/实现交接.md](可信harness/实现交接.md) | 可信 harness 的实现交接：提交、已验证行为、已知边界和审阅清单。 |
| [可信harness/试点方案.md](可信harness/试点方案.md) | 当时试点的设计与授权边界；设计新试点前配合 plan-08 读。 |
| [可信harness/试点结果.md](可信harness/试点结果.md) | 2026-10-22 的三次 conv2d 试点：第一次 4 场/0 提案；第二次 1 条 tip + kimi review；第三次自动 review 生成 `reviewed_active`，但 4 场中 r2 未交核、确认值因批次差异未发布。 |
| [写核代理读什么.md](写核代理读什么.md) | 写核代理要守的每条规则写在哪个文件里，对应哪一次真实反馈。 |
| [代理反馈.md](代理反馈.md) | 一条反馈怎么编号、怎么过期、人怎么复核。 |
| [架构改进评估.md](架构改进评估.md) | 相对公开框架还缺什么，以及已经写进代码的开关。计划账本默认开着。 |
| [可信harness/papers/导读.md](可信harness/papers/导读.md) | 可信 harness 的外部论文导读（34 篇 PDF + 2 篇博客 + 1 篇未取得全文的笔记，含 RSI/harness 自我改进专题 14 篇），逐篇给出结论、口径与对应机制。 |
| [实验批准单-搜索开关.md](实验批准单-搜索开关.md) | 搜索开关这组对照，磁盘上有什么、名义费用是多少。它不是开工批准。 |

### 别人的框架与理论笔记

| 文档 | 回答什么 |
| --- | --- |
| [KDA框架与本工程对比.md](KDA框架与本工程对比.md) | 别人怎样用编程代理轮流写内核，本仓库怎样公布倍数。 |
| [外部框架经验.md](外部框架经验.md) | 别的搜索框架里，哪几件事可以搬到这台一次只测一份的苹果显卡上。 |
| [代理理论.md](代理理论.md) | 哪些老说法对得上这个仓库。这是笔记，不是操作步骤。 |
| [pi编排与异常.md](pi编排与异常.md)、[opencode编排与异常.md](opencode编排与异常.md)、[deepseek-harness编排与异常.md](deepseek-harness编排与异常.md)、[deepseek-harness-web可观测.md](deepseek-harness-web可观测.md) | 别人的编程代理怎样收失败、超长、截断和取消。主体对的是他们的源码。每篇末节才写本仓库采纳了哪几条。 |

### 论文与访谈原文

* [`2609.36800v1.pdf`](2609.36800v1.pdf) 与 [`2609.36800v1.zh.pdf`](2609.36800v1.zh.pdf)：论文原文和中文 PDF。他们在 NVIDIA 上做的是同一类循环，目标语言不同。
* [`跑掉四五千万人民币等值 Token 之后，两位英伟达工程师重新理解了 Agent：代码不是护城河，时间才是.pdf`](跑掉四五千万人民币等值 Token 之后，两位英伟达工程师重新理解了 Agent：代码不是护城河，时间才是.pdf)：KDA / Humanize 的对谈整理。他们怎么把 harness 当 test-time scaling、三层结构（流程 / 知识库 / 性能反馈）、把钱花在验证与知识库上、以及新模型来了先拆 harness。本仓库对照它写的取舍在 [直调与hmz取舍.md](直调与hmz取舍.md) 与 [外部框架经验.md](外部框架经验.md)。
