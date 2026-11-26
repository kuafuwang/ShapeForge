# 《Trustworthy LLM Harnesses》笔记：来源、摘要与获取状态

检索与整理日期：2026-10-25。**这份笔记只依据出版方/聚合器的元数据与完整摘要，没有读过全文。**
凡涉及正文命题、证明、实验细节的地方都标注为「摘要级」，不能当作已核实的内容引用。

## 1. 我是怎么知道它的

在为本仓库「可信 harness」方向做文献检索时，第一轮查询
`trustworthy agent harness scaffolding LLM agents evaluation paper 2025 2026` 的搜索结果里出现了这条记录，
带标题、日期和 DOI。之后我尝试下载 PDF 失败（见第 5 节），所以在论文索引里只留了链接并标注「未读原文」。
本次又补做了作者、摘要与二次文献核查。

## 2. 已核实的元数据

| 字段 | 值 | 来源 |
| --- | --- | --- |
| 标题 | Trustworthy LLM Harnesses: A Measurement Framework for Correctness, Quality, Efficiency, Reliability, Robustness, Safety, and Fairness | OpenAlex / preprints.org |
| 作者 | Shudong Sun（单一作者） | OpenAlex `authorships` |
| 单位 | Statistics and Data Science GIDP, University of Arizona（页面摘要署名 shudongsun@arizona.edu） | preprints.org 页面摘要 |
| 发布日期 | 2026-10-14 | OpenAlex `publication_date` |
| 类型 / 状态 | preprint，**未同行评审** | OpenAlex `type=preprint`；preprints.org 标 "Not peer-reviewed version" |
| 引用数 | 0（2026-10-25 查询） | OpenAlex `cited_by_count` |
| DOI | 10.20944/preprints202610.1366.v1 | 搜索结果与 OpenAlex |
| 关键词 | large language model agents; AI harness evaluation; trustworthy AI; robustness; fairness; reproducibility | preprints.org 页面摘要 |

## 3. 完整摘要（OpenAlex 返回，英文原文）

> Large language model (LLM) agents are no longer deployed as bare models. They run inside an execution
> harness, the control layer that assembles context, exposes tools, manages memory and state, schedules
> retries, and verifies intermediate results. Because the harness determines what the model sees and what its
> outputs are allowed to do, observed agent behavior is jointly produced by the model and the harness, yet
> prevailing benchmarks attribute it to the model alone. We study the problem of harness evaluation: holding
> the model, task distribution, environment, budget, and evaluator fixed, how can the contribution of the
> harness be isolated, and what does it mean for a harness to be trustworthy? We formalize the harness as a
> trajectory-generating policy and define the harness effect as a paired, metric-indexed contrast. We then
> develop measurement principles along seven dimensions—outcome correctness, quality, efficiency, reliability
> and reproducibility, robustness, safety, and fairness—each targeting the specific mechanisms through which a
> harness influences agent behavior, and grounded in established results from statistical evaluation,
> efficient inference, certified robustness, and fair learning. A set of elementary propositions clarifies when
> paired designs reduce variance, how approximate verification bounds correctness loss, why marginal subgroup
> gaps can mislead, and how differential evaluator bias contaminates harness comparisons. We conclude with a
> reference protocol and argue that harness quality should be reported as a Pareto profile rather than a single
> leaderboard score.

## 4. 摘要级要点（不是全文结论）

1. **问题定义与本仓库一致**：观测到的 agent 行为是「模型 × harness」的联合产物，但常见 benchmark 把功劳/锅都记给模型。
2. **隔离方法**：固定模型、任务分布、环境、预算和评估器，把 harness 形式化为 trajectory-generating policy，
   把 **harness effect 定义成配对的、按指标索引的对比量**。这和我们 plan-02 的「同 seed 配对 + 冻结配置」同向。
3. **七个测量维度**：结果正确性、质量、效率、可靠性与可复现性、鲁棒性、安全、公平。
   每个维度针对 harness 影响行为的具体机制，而不是堆一个总分。
4. **若干命题（摘要只给了主题）**：配对设计何时降方差；近似验证如何界定正确性损失上界；
   为什么边际子群差距会误导；**差异化评估者偏差如何污染 harness 之间的比较**。
5. **报告形式**：给一个参考协议，主张用 **Pareto profile** 而不是单一 leaderboard 分数。

对本仓库最有用的两点（摘要级推断，需读全文才能确认细节）：

- **「差异化评估者偏差会污染比较」**直接对应 plan-05：如果两条 arm 的裁判/自检/评测路径不一致，
  比出来的 harness effect 就含裁判差。我们的 `reviewer`、self-check、远程评测端都应视为潜在偏差源。
- **Pareto profile 而非单一分数**正是 plan-08「五轴指标不合成分数」的外部依据；
  「近似验证界定正确性损失」则对应 plan-01：抽查式差分不是证明，要能给误差上界。

## 5. 全文与其他介绍的获取状态

**PDF 仍未取得。** 尝试记录（都是正常 HTTP 请求，没有绕过反爬、验证码或付费墙）：

| 尝试 | 结果 |
| --- | --- |
| `https://www.preprints.org/manuscript/202610.1366/v1/download` | 403（Akamai） |
| `https://www.preprints.org/manuscript/202610.1366/v1`（带浏览器 UA） | 403 |
| `https://www.preprints.org/frontend/manuscript/f14d1abba2f739cb110f1a94fa191eb1/download_pub` | 403 |
| OpenAlex `locations` | 只有 DOI 落地页，`pdf_url=None`，无其他镜像 |
| Semantic Scholar Graph API | 429 限流，未继续重试 |

**没有找到任何独立介绍它的博客或技术文章。** 使用的检索式包括：
`"Trustworthy LLM Harnesses" measurement framework`、`"preprints202610.1366" OR "10.20944/preprints202610.1366"`、
`"Trustworthy LLM Harnesses" correctness quality efficiency reliability robustness safety fairness author`、
`LLM harness measurement framework … blog 2026`、`Shudong Sun University of Arizona harness LLM agents preprint`、
`"harness effect" "trajectory-generating policy"`。
命中结果全部是 preprints.org 页面、DOI 或聚合器本身；没有其他作者的评述、机构新闻、教程或引用（OpenAlex 引用数 0）。
这只能说明「本次检索未发现」，不等于证明不存在。

## 6. 因此补下的相邻材料（不是对这篇论文的介绍）

这些材料讲的是同一个问题域（harness 的测量与审计），可读性/可得性更好：

| 文件 | 是什么 |
| --- | --- |
| `2605.14271_auditing-agent-harness-safety.pdf` | HarnessAudit：审计完整执行轨迹的边界合规、执行保真与系统稳定；210 个任务、8 个领域；指出「任务完成」与「安全执行」不一致，违规随轨迹变长累积 |
| `2605.27922_harness-bench.pdf` | Harness-Bench：在共享任务环境/预算/协议下比较**配置级** harness 效应，106 个沙箱离线任务、5,194 条轨迹 |
| `2609.05736_beyond-prompts-tool-agent-harness.pdf` | 固定模型下的资源受限 harness 选择；报告 held-out lift、最差条件提升、可重复性与 RelLift95(B)，与本仓库「效果要保守估计」最接近 |
| `articles/contextos_agent-harness-benchmark-protocol.md` | 博客：Agent Harness Benchmark 协议，要求显式分母（trusted completion、forbidden-effect rate、benign overblocking 等） |
| `articles/hugobowne_how-evals-are-central-to-harness-engineering.md` | 博客（Antaripa Saha / Hamel Husain / Hugo Bowne-Anderson，2026-09-01）：evals 是 harness 工程的核心反馈机制 |

已有索引里最接近的三篇仍是 `2604.25850`（AHE）、`2609.00006`（harness 源码解剖）、`2510.11977`（HAL）。

## 7. 引用建议

- 可以把它当作「harness 作为独立测量对象」这一问题形式化的参考来引，**不要**把它当实验证据或方法已被验证的依据。
- 它是单作者、未同行评审、发布 11 天内零引用的预印本；命题与证明需读全文核对后才能使用。
- 若之后能拿到全文，先核对第 4 节列出的四个命题的实际假设与证明，再决定是否写进 `docs/可信harness/` 的任何设计。
