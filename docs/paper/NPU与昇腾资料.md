# NPU / 昇腾算子生成资料

本文是公众号论文下载任务的补充检索记录，回答两个问题：AscendCraft 的 DSL 工程是否公开，以及当前资料里是否只有 GPU 工作。结论是：**AscendCraft 未找到官方实现，但 NPU/昇腾相关工作远不止一篇；CANNBot-CAKE、AscendKernelGen、AKG 等有真实工程可读。**

本轮只下载论文和源码，没有运行 GPU/NPU、安装 CANN 或执行这些 Agent。论文数字均为原论文口径，不是本仓库复现结果。

## 1. 先区分两个名字

### AscendCraft：论文有，官方代码未找到

- 论文：[AscendCraft: Automatic Ascend NPU Kernel Generation via DSL-Guided Transcompilation](https://arxiv.org/abs/2601.22760)。
- 本地：[英文 PDF](2601.22760_ascendcraft-automatic-ascend-npu-kernel-generation-via-dsl-guided.pdf)。
- 方法：LLM 先写类似 Triton 的 Ascend DSL，再经 host、kernel 初始化、kernel 计算、对齐/padding 四趟 lowering 生成 AscendC。
- 代码核查：论文和 arXiv HTML 没有代码链接；GitHub 仓库搜索 `AscendCraft` 返回 0。截至本轮检索，**未找到作者或华为发布的官方 AscendCraft 仓库**。这不等于证明永远不会公开，也不能把同名或第三方项目当成官方实现。
- 当前可用材料：论文第 3 节的 DSL 定义、Softmax 示例、第 4 节的多趟 lowering 与编译反馈；本地没有 AscendCraft 可运行工程。

### KernelCraft：空仓库不是 AscendCraft

- 论文：[KernelCraft: Benchmarking for Agentic Close-to-Metal Kernel Generation on Emerging Hardware](https://arxiv.org/abs/2603.08721)。
- 本地：[英文 PDF](2603.08721_kernelcraft-benchmarking-for-agentic-close-to-metal-kernel-generation.pdf)。
- 平台：PLENA、AMD NPU、Coral NPU，另有 Sonic BOOM CPU 对照；不是华为 AscendCraft。
- 代码核查：克隆时 `https://github.com/AICrossSim/KernelCraft` 没有任何 Git refs，本地只有空 `.git`，所以已删除空目录；项目主页为 <https://kernelcraft-cam.github.io/>，本轮未找到迁移后的完整实现。
- 已有相关代码：`/Users/kuafu/CLionProjects/dsl_for_llm/PLENA_Compiler` 与 `PLENA_Simulator` 已保留，但它们是 PLENA 工具链，**不等于完整 KernelCraft benchmark/agent 实现**。

## 2. 已定位的 NPU / 昇腾论文与工程

| 工作 | 主要价值 | 论文 | 工程状态与本地位置 |
| --- | --- | --- | --- |
| **CANNBot-CAKE / CAKE2** | 多 DSL 路由、Generation/Evolution、部署级 CANN 算子、验证记录与经验复用；是目前最完整的昇腾 Agent 工程链之一 | [arXiv:2610.31034](https://arxiv.org/abs/2610.31034) · [本地 PDF](../可信harness/papers/2610.31034_cannbot-cake.pdf) | **官方、有实际内容**：`/Users/kuafu/CLionProjects/dsl_for_llm/cannbot` 与 `cannbot-skills`；CAKE2 入口在 `cannbot/plugins-community/collaborative-agent-kernel-evolution/` |
| **AscendKernelGen** | Ascend-CoT、领域模型 SFT+RL、生成与评测一体；直接研究低语料 AscendC 生成 | [arXiv:2601.07160](https://arxiv.org/abs/2601.07160) · [本地 PDF](2601.07160_ascendkernelgen-a-systematic-study-of-llm-based-kernel-generation.pdf) | **官方、有实际内容**：`/Users/kuafu/CLionProjects/dsl_for_llm/NPUKernelBench`；模型/数据集只留链接，未下载权重 |
| **AKG kernel Agent** | Designer/Coder/Verifier/Conductor，多 DSL、多后端、Unified Sketch；适合比较跨 GPU/NPU 的中间表示 | [arXiv:2512.23424](https://arxiv.org/abs/2512.23424) · [本地 PDF](2512.23424_akg-kernel-agent-a-multi-agent-framework-for-cross.pdf) | **官方、有实际内容**：`/Users/kuafu/CLionProjects/dsl_for_llm/akg`，其中 `akg_agents/` 是 Agent 子项目 |
| **AgenticCANN** | 知识分层注入 + 阶段自适应进化，直接面向 Ascend 910B 低语料生成 | [arXiv:2607.26661](https://arxiv.org/abs/2607.26661) · [本地 PDF](2607.26661_agenticcann-automated-ascend-c-operator-generation-via-knowledge-augment.pdf) | **仅找到论文**；截至本轮检索未找到官方仓库，未用第三方仓库冒充 |
| **AscendOptimizer** | host tiling 进化搜索 + optimization rewind 制造可验证经验；对经验生命周期有参考价值 | [arXiv:2603.23566](https://arxiv.org/abs/2603.23566) · [本地 PDF](../可信harness/papers/2603.23566_ascendoptimizer.pdf) | `KernelHive/AscendOptimizer` 目前只有“代码、实验、文档尚未公开”的 release notice；**占位克隆已删除，只保留论文** |
| **AscendCraft** | AI 友好 Ascend DSL + 约束式多趟 transcompilation | [arXiv:2601.22760](https://arxiv.org/abs/2601.22760) · [本地 PDF](2601.22760_ascendcraft-automatic-ascend-npu-kernel-generation-via-dsl-guided.pdf) | **仅找到论文**，未找到官方工程 |
| **AscendOpGenAgent** | Triton/AscendC 算子生成、Benchmark、多轮优化；可作为独立工程比较 | 本轮未绑定唯一论文 | **独立实现，非 AscendCraft 官方实现**：`/Users/kuafu/CLionProjects/dsl_for_llm/AscendOpGenAgent` |
| **MultiKernelBench** | CUDA、AscendC、TileLang-Ascend、Pallas、MUSA、SYCL 等同任务生成评测 | [arXiv:2507.17773](https://arxiv.org/abs/2507.17773) · [本地 PDF](2507.17773_multikernelbench-a-multi-platform-benchmark-for-kernel-generation.pdf) | 已保留：`/Users/kuafu/CLionProjects/dsl_for_llm/MultiKernelBench`；含 AscendC direct-launch 和 anti-hack 检查 |
| **KernelGenBench** | 多芯片 LLM/agent 生成评测，覆盖 Ascend NPU、NVIDIA、MUSA、Hygon、Iluvatar、MetaX | 本轮未绑定唯一论文 | 已保留：`/Users/kuafu/CLionProjects/dsl_for_llm/KernelGenBench` |
| **KernelCraft** | 新 ISA 汇编生成、工具调用、模拟器/真机反馈；覆盖 AMD NPU 与 Coral NPU | [arXiv:2603.08721](https://arxiv.org/abs/2603.08721) · [本地 PDF](2603.08721_kernelcraft-benchmarking-for-agentic-close-to-metal-kernel-generation.pdf) | 官方主仓为空；仅保留 PLENA 编译器和模拟器 |

**名字警告**：CANNBot-CAKE（arXiv:2610.31034，Ascend NPU）不是 GPU 论文 CAKE（arXiv:2608.12629，Compiler–Agent Co-Design）。两者可以比较，但不能混为同一系统或同一组结果。

## 3. CANNBot / DSL 资料的实际内容

已下载：

- `/Users/kuafu/CLionProjects/dsl_for_llm/cannbot`
  - CANN 官方 Agent 应用与插件入口。
  - CAKE2：`plugins-community/collaborative-agent-kernel-evolution/`。
  - 该插件文档明确包含 AscendDSL/AscendC 生成、项目构建、正确性/性能评估和进化优化。
- `/Users/kuafu/CLionProjects/dsl_for_llm/cannbot-skills`
  - 技能、评测、设计、调试与经验工作流。
- `/Users/kuafu/CLionProjects/dsl_for_llm/cannbot-dsl`
  - Agent 亲和 DSL 的**样例集合、文档和精度测试**，包含 DeepSeek V4.1、Sparse Flash Attention 等样例。
  - README 表明编译器后端通过 `cannbot-dsl` wheel 分发；因此这个仓库不应被误写成“完整 DSL 编译器源码”。

相关官方资料：

- [CANNBot 应用仓](https://gitcode.com/cann/cannbot)
- [CANNBot skills](https://gitcode.com/cann/cannbot-skills)
- [CANNBot-DSL 样例与文档](https://gitcode.com/cann/cannbot-dsl)
- [CANNBot-DSL API 文档](https://cannbot-dsl.gitcode.com/api/)
- [CANN-Bench](https://gitcode.com/cann/cann-bench)（本轮未克隆）
- [AscendC 官方编程资料](https://www.hiascend.com/cann/ascend-c)
- [CANN 开源项目列表](https://www.hiascend.com/cann/open-map)

## 4. 哪些最值得我们先读

1. **想看“AI 友好 DSL 怎么设计”**：AscendCraft §3–4 + CANNBot-DSL 样例 + GPU CAKE 的 Cake IR。比较它们分别隐藏什么、显式保留什么、如何验证承诺。
2. **想看“完整 NPU 生成工程”**：CANNBot-CAKE/CAKE2，其次 AscendKernelGen + NPUKernelBench。前者偏 Agent 工作流与部署级交付，后者偏数据、领域模型和生成评测。
3. **想看“跨 GPU/NPU 的表示与后端”**：AKG kernel Agent + MultiKernelBench/KernelGenBench。不要把同一 DSL 在不同后端的成绩直接合并。
4. **想看“经验怎样从测量中产生”**：CANNBot-CAKE 的 CAKE Book + AscendOptimizer 论文。AscendOptimizer 目前只有论文，不能声称已有可运行代码。
5. **想看“新 ISA/汇编而非 AscendC”**：KernelCraft + PLENA 工具链；但完整 KernelCraft agent 仓目前为空。

## 5. 数据、权重与运行边界

- AscendKernelGen 的模型与数据入口：[KernelGen-LM-8B](https://huggingface.co/AscendKernelGen/KernelGen-LM-8B)、[KernelGen-LM-32B](https://huggingface.co/AscendKernelGen/KernelGen-LM-32B)、[Ascend-CoT-v3-json](https://huggingface.co/datasets/AscendKernelGen/Ascend-CoT-v3-json)。本轮未下载模型权重或大型数据集。
- CANNBot/CAKE、AscendOpGenAgent、AKG 等都需要相应 CANN/Ascend 环境；本地 macOS 目录只是源码参考，**没有进行真机验证**。
- 是否运行任何 NPU/GPU 实验、安装驱动或拉取大模型，需要另行明确授权。

## 6. 代表性“原本给人用、后来被 Agent 使用”的 DSL（仅记录，本轮未下载）

这类并没有多到需要全收。形成“DSL + Agent 生成 + 编译/数值/性能反馈”闭环的代表主要是下面这些。它们与 AscendCraft/CANNBot-DSL 不同：**DSL 本身先面向人类开发者，Agent 工作流是后加的。**

### 6.1 DSL 与工程

| DSL / 编程面 | 硬件 | 已经被哪些 Agent 或论文使用 | 本轮状态（未下载不等于不存在） |
| --- | --- | --- | --- |
| **AscendC** | 华为昇腾 | CANNBot-CAKE、AscendKernelGen、AgenticCANN、AscendOpGenAgent 等 | 论文和部分 Agent 已下载；`asc-devkit` 未克隆，只保留[官方资料](https://www.hiascend.com/cann/ascend-c)与[开源组件入口](https://gitcode.com/cann/asc-devkit) |
| **Triton-Ascend** | 华为昇腾 | AKG、AscendOpGenAgent、MultiKernelBench、CANNBot `triton-op-generator` | [`triton-lang/triton-ascend`](https://github.com/triton-lang/triton-ascend) 与 [`Ascend/triton-ascend-kernels`](https://gitcode.com/Ascend/triton-ascend-kernels) 未克隆；使用它们的部分工程已在本地 |
| **TileLang-Ascend** | 华为昇腾 | CANNBot 的 TileLang 编排、`tilelang2ascendc-ops-generator`，以及 CAKE 的多 DSL 路线 | [`tile-ai/tilelang-ascend`](https://github.com/tile-ai/tilelang-ascend) 未克隆；相关 CANNBot 插件已随仓库下载 |
| **PyPTO** | 华为昇腾 | CANNBot `pypto-op-orchestrator` / `pypto-pro-op-orchestrator`、PyPTO-Gym、`pypto-skills` | [`hw-native-sys/pypto`](https://github.com/hw-native-sys/pypto)、[`pypto-lib`](https://github.com/hw-native-sys/pypto-lib)、[`pypto-skills`](https://github.com/hw-native-sys/pypto-skills)、[`cann/pypto-gym`](https://gitcode.com/cann/pypto-gym) 均未克隆；CANNBot 编排已在本地 |
| **EasyAsc** | 华为昇腾 | `ops-easyasc-dsl`：Python DSL → 指令 IR → 模拟器/AscendC，面向 Agent skills | **不是完全未下载**：插件已随 `cannbot-skills/plugins-community/ops-easyasc-dsl` 获得；`agent/assets/` 中的 runtime/docs/examples 尚未用 `init.sh` 展开，也没有另找独立仓库克隆 |
| **NKI（Neuron Kernel Interface）** | AWS Trainium / Inferentia | NKI-Agent、ATX Custom NKI Agent、AWS Neuron Agentic Development | [`aws-neuron/neuron-agentic-development`](https://github.com/aws-neuron/neuron-agentic-development) 与 [`aws-samples/sample-ATX-custom-NKI-Agent`](https://github.com/aws-samples/sample-ATX-custom-NKI-Agent) 未克隆；NKI-Agent 论文未下载 |
| **AMD AIE C++ API / IRON / MLIR-AIE** | AMD NPU（Ryzen AI） | NPUEval：LLM 生成 AIE C++ kernel，再用开源编译器与真机反馈 | [`AMDResearch/NPUEval`](https://github.com/AMDResearch/NPUEval) 和 [`Xilinx/mlir-aie`](https://github.com/Xilinx/mlir-aie) 未克隆；NPUEval 论文未下载。这里是领域专用 C++/intrinsics，不是 Python DSL |
| **Pallas** | Google TPU（严格说不是 NPU，但问题同构） | JAXBench、MaxKernel、Autocomp 生成 Pallas kernel | [`AI-Hypercomputer/accelerator-agents`](https://github.com/AI-Hypercomputer/accelerator-agents) 未克隆；JAXBench 与 MaxKernel 论文未下载 |

### 6.2 只登记、未下载的代表论文

| 论文 | 为什么值得留作代表 | 状态 |
| --- | --- | --- |
| [NKI-Agent（arXiv:2607.04395）](https://arxiv.org/abs/2607.04395) | AWS Trainium/Inferentia 的 NKI 生成；SFT + compile-verify-fix 工具闭环，直接对应“低语料 NPU DSL” | **PDF 未下载；arXiv 网页版已读**，Table 1/3 的数字已在学习笔记中引用并标注为 `[arXiv HTML]` |
| [NPUEval（arXiv:2507.14403）](https://arxiv.org/abs/2507.14403) | AMD NPU、102 个算子、真机功能与向量化效率评测；说明编译器反馈和 RAG 仍不足以解决全部问题 | **PDF 未下载；arXiv 网页版已读**（摘要、§4、Appendix A/D） |
| [JAXBench（arXiv:2607.20466）](https://arxiv.org/abs/2607.20466) | TPU/Pallas 的 50 任务基准；用于比较文档注入、迭代 agent 与 Autocomp | **PDF 未下载；arXiv 网页版已读**（§3.2、§3.3、§4） |
| [MaxKernel（arXiv:2609.04523）](https://arxiv.org/abs/2609.04523) | TPU/Pallas 的多 agent 生成与搜索，含 HITL、Autonomous、Graph Search 三种路径 | 未下载 |
| [CANN Bench（arXiv:2607.20518）](https://arxiv.org/abs/2607.20518) | CANN 官方的 agent-generated kernel 基准；53 个 L1–L4 算子，规范计划覆盖 Triton、PyPTO、TileLang 等 DSL | 未下载；[`cann-bench`](https://gitcode.com/cann/cann-bench) 也未克隆 |
| [EvoKernel（arXiv:2603.10846）](https://arxiv.org/abs/2603.10846) | value-driven memory 用于冷启动和持续优化 NPU kernel；与经验检索/生命周期直接相关 | 未下载；[项目页](https://evokernel.zhuo.li)与[数据集](https://huggingface.co/datasets/noahli/EvoKernel)只留链接 |

### 6.3 对“昇腾是不是最热门”的判断

从本轮公开资料的密度看，**华为昇腾确实是 LLM-for-NPU-kernel 最集中的一支**：AscendC、Triton-Ascend、TileLang-Ascend、PyPTO、EasyAsc、CANNBot 系列都有公开材料。AWS Trainium 的 NKI、AMD NPU 的 AIE/NPUEval、Google TPU 的 Pallas 也各有代表工作，但数量和整合程度目前更分散。

这只是“公开论文与开源工程可见度”的判断，**不等于市场份额、硬件销量或性能优劣**。对我们的工程而言，更值得优先比较的是：Triton-Ascend / TileLang / PyPTO 的抽象层级、Agent 能否获得确定性验证，以及 DSL 的硬件承诺是否能被静态检查，而不是单纯追逐“NPU 热门”这个标签。

## 7. 进一步学习：DSL 为什么被选中、如何为 LLM 设计

已单独整理学习笔记：[LLM 与 NPU DSL：为什么被选中，又如何为 LLM 设计](LLM与NPU-DSL：为什么被选中、如何为LLM设计.md)。

笔记重点解释：人类 DSL 被 Agent 选中的工程原因，以及 LLM/Agent 面向的 DSL 如何通过 canonical 表示、显式硬件承诺、分层 lowering、局部诊断、模拟器和独立验证来降低生成难度。

笔记里另有两处与本文直接相关：一是把“对付低语料”拆成四个杠杆（换表示／给上下文／训模型／挖经验）并各自配可核证据，用于判断该先投哪一条；二是统一了来源标注，`[本地 PDF]` 与 `[arXiv HTML]` 分开写，所有引用的数字都连同基线、模型与题集一起记录——因为这批工作里口径的影响普遍大于方法的影响。
