# SOL-ExecBench、KDA 与 SOLAR 如何组合

> 性质：外部论文与公开源码的只读调研，不是 ShapeForge 已完成的集成，也不是新的性能证据。
> 本次没有运行 GPU、调用付费模型或下载完整 Hugging Face 数据集。
> 这里的 KDA 默认指 **Kernel Design Agents**；Kimi Delta Attention 的同名缩写另见文末。

## 0. 结论先行

“KDA 负责生成和优化 kernel，SOL-ExecBench 负责评测，SOLAR 负责计算硬件理论下界”是正确的角色概括，但不能据此推断三者已经组成一个公开的一键流水线。

实际有两种已经成立的连接，以及一个不存在的直接连接：

1. **SOLAR → SOL-ExecBench**：在题目或排行榜版本构建期，从参考程序、workload shape 和硬件配置推导 `t_sol`。
2. **KDA → SOL-ExecBench**：KDA 生成的候选可以交给 SOL-ExecBench 或官方排行榜评测；公开的 Databricks 结果证明了这种实际使用。
3. **KDA ↛ SOLAR**：本次核对的公开仓库里没有直接调用关系，也没有证据表明 KDA 每优化一轮就重新运行 SOLAR。

因此，更准确的图不是三个程序在线依次调用，而是：

```text
━━━━━━━━━━━━━━ 题目 / 排行榜版本构建期 ━━━━━━━━━━━━━━

definition.json + workload.jsonl
  PyTorch reference + 具体 shape
                  │
                  ▼
                SOLAR
  Graph → Einsum → Work/Bytes Analysis → Roofline
                  │
                  ▼
          每个 workload 的 t_sol
                  │
                  └──────────────┐
                                 │
未命名的多 agent 优化器           │
  多轮生成候选                    │
        │                       │
        ▼                       │
远程 SOL-ExecBench 沙箱          │
  正确性 + 反作弊 + 计时          │
        │                       │
        ▼                       │
选择最快的合法候选作为 t_b ──────┤
                                 │
━━━━━━━━━━━━━━ 后续候选优化期 ━━━━│━━━━━━━━━━━━━━
                                 │
KDA / 其他优化系统                │
  研究并生成 solution.json + 源码 │
        │                       │
        ▼                       │
SOL-ExecBench / 官方评测服务      │
  正确性 C + 候选实测 t_k         │
        │                       │
        └───────────┬───────────┘
                    ▼
        S = 1 / (1 + (t_k-t_sol)/(t_b-t_sol))
                    │
                    ▼
             SOL Score / 排行榜
```

这张图表达逻辑数据流，不表示当前公开仓库已经提供把全部步骤自动连接起来的驱动程序。

## 1. 核对范围与证据边界

### 1.1 一手资料

- SOL-ExecBench 技术报告：[arXiv:2603.19173](https://arxiv.org/abs/2603.19173)，重点是 §4.2–4.5。
- SOLAR 技术报告：[arXiv:2606.26383](https://arxiv.org/abs/2606.26383)，重点是 §3 和附录 B。
- KDA 官方仓库：[NVlabs/kda](https://github.com/NVlabs/kda)。
- SOL-ExecBench 官方仓库：[NVIDIA/SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench)。
- SOLAR 官方仓库：[NVlabs/SOLAR](https://github.com/NVlabs/SOLAR)。
- KDA 实际使用 SOL-ExecBench 的公开记录：[L1 记录](https://humanfia.ai/news/2026-07-02-solexec-l1)与[批量优化记录](https://humanfia.ai/news/2026-06-22-sol-bench-batch)。

公众号 PDF 只作为本次调研入口，不用它替代论文和源码。该 PDF 位于另一个本地 checkout：

```text
docs/一个公众号的论文解读/AI 写 kernel 的时代来了：NV 给算子生成定了一条硬件满分线.pdf
```

### 1.2 本地源码版本

| checkout | 核对版本 |
| --- | --- |
| `/Users/kuafu/CLionProjects/kda` | `acdd79343c5e9d0c7cb96289f26fd3f1051ce785` |
| `/Users/kuafu/CLionProjects/SOL-ExecBench` | `a9fa0804c793d438e70850c33fe34426e66d53dd` |
| `/Users/kuafu/CLionProjects/SOLAR` 当前 `master` | `1a1bbecac3683541e1ca5081a825ffed2191ced8` |
| SOLAR 历史 `origin/execbench-solar-fixes` | `71750b07952baccea499efd4e08bbd90d025e03f` |

本地 SOL-ExecBench 的 `data/` 只有 `.gitkeep`，没有完整题目和真实 trace。因此，生产排行榜服务或 Hugging Face 原始行中额外保存了什么，不能仅凭本地文件断言。本文关于“当前公开 schema 没有什么”的结论，只限于上述公开源码版本。

## 2. 三个项目分别负责什么

| 项目 | 输入 | 输出 | 不负责什么 |
| --- | --- | --- | --- |
| KDA / Kernel Design Agents | 任务合同、工作区、参考实现、工具和历史证据 | 候选源码、验证/计时/profile 记录、晋级决定 | 不天然拥有某个 benchmark，也不计算 SOLAR bound |
| SOL-ExecBench | definition、workload、solution、硬件与计时配置 | 正确性状态、候选延迟和评测 trace；官方服务另给排行榜结果 | 当前公开 harness 不调用 SOLAR，也不从候选推导理论下界 |
| SOLAR | 参考程序、shape、dtype、目标硬件和建模假设 | MAC/FLOP、访存字节、bottleneck、解析 SOL runtime | 不测量候选，不生成 KDA 优化策略 |

最容易混淆的一点是：**SOLAR 内部的 agentic translator 是把参考程序翻译成可验证 DSL 的 agent，不等于 KDA 这个 kernel 优化工作流。**

## 3. SOLAR 与 SOL-ExecBench 的官方语义连接

### 3.1 SOLAR 提供 `t_sol`

SOL-ExecBench 论文 §4.2 明确说，题目的 SOL runtime 由 SOLAR 推导。其 Roofline 形式为：

\[
t_{sol}=\max\left(\frac{\text{Total FLOPs}}{\text{Compute Throughput}},
                  \frac{\text{Total Fused Bytes}}{\text{Memory Bandwidth}}\right).
\]

这里有两层含义：

- 计算项和访存项取 `max`，对应计算与访存理想重叠的模型；
- 使用 fused bytes，意味着中间结果的融合和复用假设会直接影响下界。

SOLAR 分析的是参考程序的语义、shape 和建模假设，不是每份候选的实际指令、调度或实测 latency。

### 3.2 `reference`、`t_b` 和 `t_sol` 是三种不同的东西

SOL-ExecBench 特意把功能参考实现和评分 baseline 分开：

- **reference**：定义“算什么”，用于正确性；通常可读、可移植，但不追求高性能。
- **scoring baseline `t_b`**：优化实现的延迟，用作 SOL Score 的 `0.5` 锚点。
- **SOLAR `t_sol`**：解析下界，用作 `1.0` 锚点。

因此不能拿 reference latency 代替 `t_b`，也不能把候选延迟当成 `t_sol`。

### 3.3 分数是在三者汇合后计算的

对正确候选：

\[
S(t_k)=\frac{1}{1+\dfrac{t_k-t_{sol}}{t_b-t_{sol}}}.
\]

锚点是：

- `t_k = t_b` 时，`S = 0.5`；
- `t_k = t_sol` 时，`S = 1`；
- 候选越慢，分数越接近 `0`；
- 错误候选没有性能信用，记为 `0`。

论文假设 `t_b > t_sol` 且 `t_k >= t_sol`。如果候选反而快过 `t_sol`，论文要求把它当成审计信号，检查 SOLAR bound 和 reward hacking，而不是直接宣布突破物理极限。

## 4. 当前公开 SOL-ExecBench 源码实际输出什么

### 4.1 trace 提供候选时间，不自动提供完整评分三元组

当前 `src/sol_execbench/core/data/trace.py:58-70` 的 `Performance` 只有：

```text
latency_ms
reference_latency_ms
speedup_factor
```

`src/sol_execbench/driver/templates/eval_driver.py:584-653` 的执行顺序是：

1. 对候选 `user_fn` 计时；
2. 可选地对 definition reference 计时；
3. 写出 `latency_ms`、reference latency 和二者比值。

其中：

- `latency_ms` 可映射为 `t_k`；
- `reference_latency_ms` 是原始 reference 的时间，不是优化 scoring baseline；
- trace 没有 SOLAR 生成的 `t_sol`。

代码内部的局部变量 `_sol_latency_ms` 名字有误导性：它保存的是 **solution latency**，不是 Speed-of-Light latency。

### 4.2 当前 workload schema 不带评分时间

`src/sol_execbench/core/data/workload.py:102-117` 的字段是：

```text
axes, inputs, uuid, tolerance
```

没有 `t_sol`、`t_b` 或等价字段。`scripts/download_solexecbench.py:19-60` 也只是把 Hugging Face 行组装成 definition、reference 和 workload 文件。

### 4.3 `sol_score()` 当前是独立工具函数

`src/sol_execbench/sol_score.py` 定义了评分公式，并把 `t_sol` 描述为 raw SOLAR estimate。但在上述 SOL-ExecBench checkout 中，`sol_score(` 只有定义，没有调用点。

这与论文的描述相容：harness 报告绝对 runtime，SOL Score 随后结合该 benchmark release 的 scoring baseline 计算。公开本地 CLI 不应被描述成“跑完就必然得到官方 SOL Score”。

## 5. 当前 SOLAR 如何读取 SOL-ExecBench 题目

当前 SOLAR 有：

```text
scripts/gen_solexecbench_adapter.py
```

它读取用户提供的 `definition.json` 和可选的 `workload.jsonl`，解析 const/var/expr axes，并生成带 `Model`、`get_model()`、`get_inputs()` 的 PyTorch adapter。

但这个脚本只做 **schema/source adapter**：

- 不生成候选 `solution.json`；
- 不调用 SOL-ExecBench；
- 不自动跑完整 SOLAR pipeline；
- 不调用 `sol_score()`；
- 与 KDA 没有连接。

当前文档中的完整路线还需要先为参考程序准备并验证 SOLAR DSL，再在原始 shape 上分析：

```bash
$PY scripts/verify_saved_ir.py $M/adapter.py $M/solar_model_einsum_lib.py \
    --source-type cuda --mem-budget 30 [tolerance flags]

$PY scripts/trace_saved_ir.py $M/solar_model_einsum_lib.py \
    --definition <bench>/<cat>/<p> --out-dir $M/dsl_full \
    --fallback-adapter $M/adapter.py [--custom-py $M/full_inputs.py]

$PY -m solar.einsum_dsl.cli_einsum $M/dsl_full \
    --output-dir $M/einsum_full

$PY -m solar.cli.analyze_model \
    --einsum-graph-path $M/einsum_full/*_einsum_graph.yaml \
    --output-dir $M/analysis_full --no-copy-graph

$PY -m solar.cli.predict_perf_model \
    --analysis-path $M/analysis_full/analysis.yaml \
    --output-dir $M/perf_full --arch-config B200 \
    --precision fp16|fp8|nvfp4
```

来源是 SOLAR `skills/verified-translation/SKILL.md:91-124`。adapter 的 capped verification 与 original-shape trace 是两个步骤，不能只验证小 shape 就把小 shape 的工作量当成原题下界。

当前 `skills/verified-translation/reference.md:349-354` 还明确写着：

> Collection and join tooling is not shipped.

也就是说，把各 workload 的 perf YAML 与候选计时按身份连接，仍需外部工具完成。

## 6. 历史分支中最具体的组合实现

SOLAR 的 `origin/execbench-solar-fixes` 分支包含当前 `master` 已移除的：

```text
scripts/run_execbench_problem.py
scripts/export_execbench_csv.py
configs/execbench/
```

### 6.1 `run_execbench_problem.py`

这个脚本完成了真实的格式桥接：

1. 读取一道 SOL-ExecBench 题和一个 workload；
2. 解析轴和标量；
3. 把 reference `run()` 包装成 SOLAR model；
4. 调用 graph → einsum → analysis → perf 四阶段；
5. 写出 `sol_summary.json`。

摘要字段包括：

```text
problem, workload_uuid, workload_axes, resolved_axes
arch, precision, setup_config
total_macs, total_flops
sol_ms.{unfused,fused,fused_prefetched}
bottleneck.{unfused,fused,fused_prefetched}
memory_bytes.{unfused,fused,fused_prefetched}
perf_yaml
```

它还接受：

```bash
--t-k <候选实测毫秒> --t-b <优化 baseline 毫秒>
```

随后直接执行：

```python
t_sol = summary["sol_ms"]["fused"]
summary["sol_score"] = sol_score(args.t_k, args.t_b, t_sol)
```

这证明该历史实现选择 **SOLAR fused runtime** 作为 `t_sol`，并真正调用了 SOL-ExecBench 的评分函数。

历史分支上的典型调用形式是：

```bash
python scripts/run_execbench_problem.py \
  ../SOL-ExecBench/data/benchmark/L1/<problem> \
  --workload-index 0 --arch-config B200 \
  --t-k <candidate-ms> --t-b <optimized-baseline-ms>
```

但它仍不是完整自动闭环：

- `t_k` 由外部传入；
- `t_b` 由外部传入；
- 不自动消费 SOL-ExecBench trace；
- 不调用 KDA。

所以它是“SOLAR + 外部时间 + SOL Score”的具体实现，不是三项目统一 runner。

### 6.2 `export_execbench_csv.py`

历史 exporter 按 `(problem, workload_uuid)` 汇集 `sol_summary.json`，导出 SOLAR fused/unfused/prefetched 结果。它允许用户另给 comparison CSV，其中可带：

```text
reference_latency_ms
optimized_baseline_latency_ms
```

文档明确说比较数据不随仓库发布。这再次说明：bound 生成和实际计时汇合是独立步骤。

当前 SOLAR `scripts/README.md` 说明，开发期间使用的 benchmark-campaign runners、collectors 和 comparison helpers 有意不放进当前公开 distribution。因此，历史分支可以用于理解设计，但不能冒充当前 `master` 的可用命令。

## 7. KDA 与 SOL-ExecBench 如何实际组合

### 7.1 公开 KDA 仓库提供的是通用任务合同

`kda/docs/agent-flow.md` 定义的最小循环是：

1. 定义任务合同；
2. agent 检查工作区；
3. 写计划；
4. 实现候选；
5. 验证正确性；
6. 测量目标指标；
7. 记录证据；
8. 决定保留、修改或拒绝；
9. 重复，直到满足晋级条件或 blocker 明确。

任务合同包含：

```text
Objective
Inputs / outputs
Correctness requirements
Implementation constraints
Validation command
Evaluation command
Promotion criteria
```

因此 KDA 接入 SOL-ExecBench 的自然接口是：把 `sol-execbench` 或远程评测调用写成任务合同的 validation/evaluation command。它不需要 KDA 理解 SOLAR 内部 IR。

当前 KDA 仓库没有 SOL-ExecBench 专用 adapter、runner 或 `sol_score()` 调用；README 也明确说 workflow 独立于单一 benchmark harness。

### 7.2 有真实使用证据，但没有完整公开编排实现

Humanfia 的公开记录说明，KDA 生成的 kernel 以 Databricks 名义进入 SOL-ExecBench 排行榜；2026 年 7 月的 L1 记录给出 SOL Score `0.76394`。另一篇记录描述了从一台 8×B200 节点开始的无人值守批量优化队列。

这些材料证明：

```text
KDA 生成候选 → SOL-ExecBench 外部评测 / 排行榜
```

但它们没有公开足够细节来证明以下更强说法：

- KDA 每一轮都调用 SOLAR；
- SOL Score 是直接在线奖励，还是只使用绝对 latency 直到最后提交；
- 使用本地公开 harness、私有服务还是二者组合；
- 排行榜结果以什么频率和格式回灌给 agent；
- Databricks 的完整私有 orchestration 与公开 KDA 文档逐字节相同。

因此，不能把角色图进一步编成未公开的内部实现。

### 7.3 论文的 `0.732` 不是已证实的 KDA 成绩

SOL-ExecBench 论文 §4.5 的 baseline 生成器只称为“agentic kernel optimization system”，没有命名为 KDA。

它的工作方式是：

- 多 agent 在固定时间和成本预算内独立优化；
- 每轮收集合法提交，并暴露给下一轮 agent；
- 后续 agent 可以从已有正确候选继续；
- baseline 生成阶段的 agent 被限制为 PyTorch 和标准 Python 包；
- 候选必须经过编译、正确性和反作弊检查；
- 每题选择最快的合法候选作为最终优化 baseline；
- 新 scoring baseline 被采纳前还要人工审查。

这套流程证明了“agent 生成 + harness 验证 + 最快合法候选晋级”的组合，但不能把论文中的整体中位 SOL Score `0.732` 自动归到 KDA。KDA 的明确证据来自后续实际排行榜使用记录。

## 8. 三个时间值必须怎样连接

| 量 | 权威生产者 | 当前公开 artifact | 不能误认为 |
| --- | --- | --- | --- |
| `t_sol` | SOLAR 对参考程序、原始 workload shape 和硬件模型的分析 | SOLAR perf YAML / 历史 `sol_summary.json` | 候选实测时间 |
| `t_k` | SOL-ExecBench 对候选的实际计时 | trace `evaluation.performance.latency_ms` | SOLAR 输出 |
| `t_b` | 该 benchmark release 指定的优化 scoring baseline | 当前本地 harness 不产生 | `reference_latency_ms` |
| `S` | `t_k/t_b/t_sol` 经 `sol_score()` 组合 | 当前仓库里的独立函数或官方排行榜结果 | 单纯 `t_sol/t_k` |

连接键至少需要：

```text
problem identity + workload UUID + benchmark/evaluator release
+ hardware/clock + precision + SOLAR modeling assumptions
```

如果任何一项变化，就不能默认三个时间仍属于同一个评分合同。

## 9. 如果要自己搭一套可运行的组合

以下是根据公开接口推导出的实施方案，不是对现有官方私有流程的描述。

### 9.1 冻结评测合同

先固定：

- SOL-ExecBench 数据和 evaluator 版本；
- 每道题的 workload UUID 与 shape；
- B200 等目标硬件和时钟政策；
- 正确性、反作弊、warmup 和计时规则；
- SOLAR fused/unfused、dense/sparse、精度和 bytes 假设；
- 优化 baseline 的身份和 `t_b`。

当前 SOLAR 文档特别提醒：默认 sparse peak 对应 2:4 结构化稀疏假设；普通 dense 工作应区分 dense SOL。fused SOL 也不是无条件可达的实测保证。

### 9.2 离线生成并保存 `t_sol`

为每个 `(problem, workload_uuid)` 运行 SOLAR，保存：

- 参考实现和 adapter/DSL 身份；
- 数值验证范围；
- 原始 shape trace 身份；
- perf YAML；
- 选择的 `t_sol` 及其 fused/dense/sparse 等假设。

不要让候选修改这份下界。

### 9.3 用同一计时边界建立 `t_b`

`reference_latency_ms` 只能作为 reference 的测量，不能自动成为优化 scoring baseline。`t_b` 必须来自声明的优化实现，并尽量与候选采用相同的硬件、输入、计时和包含项。

### 9.4 把 SOL-ExecBench 写进 KDA 任务合同

概念上的合同可以是：

```text
Objective:
  在固定 benchmark release 的全部 workload 上降低合法候选延迟，
  或在自有 t_b/t_sol 数据齐全时提高离线复算 SOL Score。

Validation / evaluation:
  sol-execbench <problem_dir> --solution <solution.json> -o <trace.jsonl>

Promotion:
  所有要求的 workload 均为 PASSED；
  fresh 复测确认提升超过噪声；
  身份、正确性和计时边界未改变；
  t_k < t_sol 或 t_b <= t_sol 时停止晋级并审计。
```

这是一份集成合同示例，不是 KDA 仓库已有配置。

### 9.5 在 harness 外做评分 join

从 trace 读取：

```text
t_k = evaluation.performance.latency_ms
```

按 problem + workload UUID 连接冻结的 `t_b/t_sol`，再调用 `sol_score()`。失败候选记零分，不用异常或缺失 latency 参与平均。

当前公开代码没有自动完成这一步；历史分支也只是通过 `--t-k/--t-b` 手工注入两个时间。

### 9.6 将证据反馈给下一轮优化

每轮至少保存：

- 候选源码与 solution；
- trace 和 correctness 状态；
- 原始 `t_k`，不只保存派生分；
- profile 和候选变更；
- 晋级/拒绝原因；
- 所用 benchmark、baseline 和 SOLAR 配置身份。

需要官方成绩时，再按官方提交流程送评测服务。本地自算分不能冒充官方排行榜分。

## 10. 对 ShapeForge 的可借鉴点与限制

可借鉴的是职责分离：

1. **候选生成器不拥有理论下界**；
2. **harness 保存原始正确性和计时证据**；
3. **reference、优化 baseline、SOL bound 分开管理**；
4. **按稳定身份连接候选、workload、硬件和评分版本**；
5. **候选快过 bound 时先审计，不直接发布**。

但当前 ShapeForge 的生产后端是 Apple Metal，而 SOL-ExecBench 的官方成绩基于 B200。不能：

- 直接把 Metal candidate 交给 SOL-ExecBench 跑官方题；
- 用 B200 的 `t_sol` 给 M4 Max 上的 KDA 打分；
- 把 KDA-for-KDA 相对 FlashKDA 的加速比当成 SOL Score；
- 在没有 Metal 硬件模型和相同计时边界时声称已经复刻该评分体系。

若要将同样思想用于 Metal，需要另建并验证对应硬件的峰值、带宽、精度和融合假设，同时保持 ShapeForge 自己的候选自由、fresh confirmation 和证据身份合同。

## 11. 同名 KDA 的附注

本文的 KDA 是 **Kernel Design Agents**。另一个 KDA 是 **Kimi Delta Attention** 算子。

`KDA-for-KDA` 表示用前者优化后者。本地 `/Users/kuafu/CLionProjects/kda-260927-kda` 使用自己的 FlashKDA/CUPTI benchmark，没有 SOLAR 或 SOL-ExecBench 直接集成。它的 CuTe/TIRx/PTX 加速比不能在没有同合同 `t_b/t_sol` 的情况下换算成 SOL Score。

## 12. 最终判断

最准确的一句话是：

> **SOLAR 在题目构建期“造尺子”，SOL-ExecBench 在候选优化期“量候选”，KDA 根据验证、计时和 profile 证据“改候选”；三者通过按版本冻结的 `t_sol/t_b` 和每份候选的 `t_k` 汇合，而不是通过当前公开仓库里的一个三合一可执行程序汇合。**
