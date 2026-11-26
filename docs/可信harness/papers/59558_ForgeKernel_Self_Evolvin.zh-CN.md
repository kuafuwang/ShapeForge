# ForgeKernel：自我演化的 Kernel——追上并超越专家与 Agent

> **原文标题**：ForgeKernel: Self-Evolving Kernels Catching and Surpassing Experts and Agents  
> **状态**：ICLR 2027 会议论文，双盲评审中（Anonymous authors; Paper under double-blind review）  
> **原文**：[`59558_ForgeKernel_Self_Evolvin.pdf`](59558_ForgeKernel_Self_Evolvin.pdf)，18 页，LaTeX / pdfTeX，创建于 2026-09-26  
> **原文 SHA-256**：`4f0251783647cb4be97ba57e921f4e6bef2bcae02f82fab6bd8271cc83bd0872`

## 翻译说明

- 本文件是 **AI 辅助制作的中文译文，不是作者官方译本**。正文、附录、表和图注按 PDF 文本层翻译；关键数字已对照原文。歧义以原始 PDF 为准。
- 文中所有加速比、时间、分数和成本都是 **论文作者的主张**，不是本仓库的测量结果。
- 图 1、图 2、图 7 是结构示意图，下面用文字转写其关系；图 3–6、图 8 的数值已收成表。色块、弧线和未标出的 Public best 散点以原 PDF 为准。
- 英文技术名词在必要处保留：kernel、harness、campaign、CTA、NCU、MMA、TMA、roofline、`/goal`。算子名、库名、模型名不翻译。
- 参考文献按原文著录，不逐条翻译。

## 术语对照

| 英文 | 本译文 |
| --- | --- |
| self-evolving | 自我演化 |
| campaign | 战役 |
| sealed reference | 密封参考（只可运行、计时、剖析，不提供源码） |
| arch agent | 架构 agent |
| coding agent | 编码 agent |
| milestone / curriculum | 里程碑 / 课程 |
| decision row | 决策行 |
| gate | 门 |
| probe-certified | 经探针认证 |
| promote | 升格（把已接受 kernel 封成下一场的参考） |
| knowledge base / recipe | 知识库 / 配方 |
| noise floor | 噪声底 |
| same-job gain | 同一 GPU 作业内的增益 |
| foreign entry | 外来条目（其他战役写下、本场落地的配方） |
| no-go | 否定记录 |
| geometric mean | 几何平均 |
| persistent loop | 持久循环 |

---

## 摘要

前沿编程 agent 已经能写出与专家相当的 GPU kernel：放任它跑，即便是一条通用的 agent 循环，也能在许多算子上胜过手工调优的参考实现。但每一条这样的循环都会停在它自己造成的天花板上。它在最初几次尝试里就定死 kernel 的架构，之后整场都在这个架构之下做调参；而且每一场都从和上一场相同的知识出发，把已经测到的东西丢掉。

我们提出 ForgeKernel，一个用来打破这层天花板的自我演化 kernel agent。它只从硬件行为读取参考 kernel 的架构——运行、计时并剖析一个密封二进制，从不读它的源码——再把它变成一门架构优先的课程，交给编码 agent 去实现；一旦测量表明瓶颈在架构而不是调参，这门课程会被重新打开。

每一个被接受的 kernel 都会被密封，并升格为下一场战役的参考，于是优化可以越过当前已知的最好实现继续往前走；每一条被硬件探针认证的发现，进入一个跨算子共享的知识库。

在 Hopper 与 Blackwell 上的 27 个算子中，ForgeKernel 追平或超过 22 个专家库 kernel 里的 18 个，其中包括 FlashAttention-4、cuDNN 和 DeepGEMM；也追平或超过三个 kernel 生成系统已发表的全部 7 个 kernel，后者最高 +170.6%（几何平均 +39.4%）。对 DeepSeek-V4.1 的 sparse-MLA prefill 与融合 attention kernel——本稿提交前 15 天才发布，没有论文、也没有第二份实现——它分别落到 1.3% 与 3.7% 以内。只把剖析器换成昇腾的之后，它在 Ascend 上领先 CANN-Bench 全部 8 个 level-4 算子的最佳公开提交，几何平均 +33.1%。把已知最好的 kernel（包括一条通用循环写出的 kernel）升格为参考，还能再拿到最多 +9.9%。

---

## 1 引言

现代机器学习系统对 GPU kernel 性能的要求越来越高，新算子也在不断出现：稀疏 attention indexer（DeepSeek-AI, 2025b）、门控 delta 网络（Yang et al., 2024）、潜在 attention 变体（DeepSeek-AI, 2024）。好消息是，大语言模型已经能自己写出高效的 GPU kernel。已有一批工作专门做 kernel 优化 agent（Chen et al., 2025; Wang et al., 2025a; Dong et al., 2025; Wei et al., 2025; Han et al., 2026; Luo et al., 2026），在某些设定下这些 agent 胜过专家实现。更刺眼的是一个开源结果（vincent317, 2026）：前沿模型 Claude Fable 5.1 只用 Claude Code 的 `/goal` 模式，就已经在许多任务上追平专家 kernel。

作为算子 agent 的最小形态，这个结果迫使我们问：在今天的前沿模型之上，一个面向算子的 agent 应该怎么设计。这种 agent 的存在，是为了补上模型还剩的能力缺口，而不是在模型已经交付得很好的地方再去锦上添花。它携带的流程是临时的：随着 agentic kernel 数据进入训练语料，模型会把这些流程内化，harness 要一块一块拆掉，因为留下来的会变成过度指导。Kernel Design Agents（Liu et al., 2026）是代表案例：一套曾经塞满优化细节指导的系统，已经被削成一条轻量工作流，而 `/goal` 轨迹显示模型自己就会用它、并且改得更好。这并不意味着算子 agent 失去了价值。我们测量了 `/goal` 停在哪里（§2），发现两个值得补上的缺口。在一些算子上，它仍然达不到最好的人类实现，障碍是架构性的而不是战术性的：架构在最初几次尝试里就定了，填满剩余运行时间的局部调参改不了它。在它确实胜过人类专家的算子上，也没有任何东西说明它已经碰到硬件极限。

这些观察引出 ForgeKernel，一个自我演化的算子 agent 系统（图 1）。它的目标是继续协助基座模型前进，并去探硬件极限，直到模型能自己把每个算子都定下来。两处核心设计承担这个目标。第一处是分析一份参考实现并复现它的性能，让每个算子都从当前最好水平起步；分析的同时，ForgeKernel 形成自己的判断，可以接受或改进参考的架构决策，所以架构在战役进行中是可调的，而不是一开始就钉死。第二处是把结果喂回去的自我演化：ForgeKernel 把自己写出的 kernel 当成新的参考，再分析一次，再超越一次。这也是搜索能在最好水平之后仍然保持开放的原因：增益朝硬件 roofline 收缩，直到某一代的增益掉到测得的噪声底以下，循环停止，它所到达的平台期本身就是对天花板的一个估计。这两处设计都不需要参考的代码，只需要能够剖析它。于是开源代码很少的领域——训练便是其中之一——也能分到这份收益。

自我演化要付得起，还需要第三样东西：活过一场运行的知识。一场 `/goal` 运行留下的是一个 kernel，不是知识，所以在自我演化循环里，每一代都要为上一代已经学会的东西再付一次钱。ForgeKernel 在设计上是有状态的。挺过测量的发现积累进知识库，按 GPU 组织，并在算子和形状之间共享，于是每一代从上一代表结束的地方起步，而不是从模型预训练结束的地方起步。

在 Hopper 与 Blackwell 上的 27 个算子、29 个输入形状中，ForgeKernel 在 29 个单元里的 25 个上追平或超过其专家参考，并追平或超过三个近期生成系统已发表的全部 7 个 kernel。相对 `/goal` agent，比较是沿着 agent 自己的结局切开的，而不是沿着算子切开的：agent 没达到参考的那些 kernel，ForgeKernel 每一个都达到了；agent 领先的那些，它收敛后的 kernel 成为下一场参考并被超过，相对那些作者已经停止改进的 kernel 再高出 +9.5%。

因此 ForgeKernel 贡献三件事，按它使用它们的顺序：从一个它从不阅读的参考上读出 kernel 架构；一个经探针认证的硬件机制知识库，可在共享同一 GPU 的算子之间迁移；以及通过把自己接受的 kernel 升格为下一场战役的参考来自我演化。

**图 1：自我演化的两种机制。**（a）一场 `/goal` 运行很早定死架构，停在一个离天花板距离未知的平台期，并丢弃它测到的东西。（b）ForgeKernel 把每个被接受的 kernel 升格为下一场参考（上），并把经认证的发现锻进知识库（下）。图中关系：设备时间越低越快；架构早早定死之后，局部调参填满剩余运行且不能改架构；硬件天花板未知。知识被丢弃：每一场运行结束后，下一代要为上一代已经学会的东西再付一次。ForgeKernel 一侧：第 0 代是密封参考，升格后第 1 代、第 2 代各自成为下一场参考；当某一代增益掉到噪声底以下即停。知识库按 GPU 组织、跨算子共享；每一代从上一代表结束的地方开始。处方（prescribe）与收获（harvest）把知识库和各算子连起来。

---

## 2 前沿 agent 独自能走多远

在加入任何机制之前，我们先测量模型单独能做什么。`/goal` agent 是 Claude Fable 5.1，由未修改的 Claude Code CLI 以默认 `/goal` 模式驱动，每个 kernel 最多 24 小时，提示固定：算子签名、输入形状，以及一块 H100。唯一扣下的是代码。它不能读参考的源码，也不能从网上拷代码，但可以用 ForgeKernel 的架构 agent 后来使用的同一套工具去运行、计时、剖析和调查参考。分开二者的是机制，不是信息。它的目标是开放式的，循环在会话之间无状态，所以下面每个数字都是收敛后的读数，不是初稿；它学到的任何东西都活不过一场运行。一个 kernel 只有在每个张量上的误差落在专家参考自身误差的 2 倍以内才计数；我们逐个审计了用精度换速度的交易。十个任务覆盖 attention、GDN，以及 DSA 稀疏 attention 链的各个成员，前向和反向都有，参考来自 FlashAttention-4、FlashInfer、FlashQLA、DeepGEMM、FlashMLA 和 cuDNN-frontend。表 5（附录 I）是我们起步时的天花板。

**十个任务，七场胜利。** agent 在十个任务中的 7 个上胜过参考，全集几何平均 +17%。胜利包括：GQA attention 前向与反向上的 FlashAttention-4；DSA 反向链的全部三个 cuDNN-frontend 参考；cuDNN-frontend 的 radix top-k；以及 FlashInfer 的融合 GDN prefill kernel——一个几乎没有公开先例的年轻算子，并在第二块 H100 和第二个 FlashInfer 版本上独立复核过。它没胜过的三个是：DSA 的 fp8 indexer 前向、DSA 稀疏 attention 前向，以及 GDN prefill 反向；前两个面对的是这组里工程化程度最高的参考。短板既不跟着算子走，也不跟着前向/反向走：单是 DSA 稀疏 attention 链，就同时有一场前向失利和一场反向胜利。

**不动点。** 无论 `/goal` agent 停在参考之上还是之下，停法都一样：停在一个它自己的循环已经无法再改进的收敛 kernel 上。表 5 的每一行，胜的和负的一样，都是一个无界优化器停下来的地方。本文其余部分拿来比较的，是这个不动点，不是参考。有两件事在那里结束。搜索结束了，因为 agent 在最初几次尝试里定死架构，填满剩余运行的局部调参无法重新打开它。知识也结束了，因为无状态会话丢掉它花钱测到的每一条硬件事实。§3 就是为了把这两件事都继续下去。

---

## 3 方法

### 3.1 总览

一个 kernel 的性能由两层决定：固定其全局结构的架构决策——分块、流水深度、warp 角色、张量布局——以及执行该结构的战术知识。在 Hopper 和 Blackwell 上，第一层设定一个第二层抬不上去的天花板（NVIDIA Corporation, 2022; 2026b; 2024; Thakkar et al., 2026; Shah et al., 2024）（附录 A）。架构层也可以不靠源码观察到：剖析器签名给出线索，有针对性的探针再消歧（§3.2）。

一场战役接收一个算子、一个输入形状、一个密封参考 \(R\) 和相对它的目标增益 \(\tau\)，返回一个 kernel \(K\) 以及产出它的里程碑课程 \(M_0 \to M_k\)。记 \(t_K\) 为设备时间，增益 \(g(K, R) = t_R / t_K\) 始终在同一 GPU 作业里对两个 kernel 一起计时；当 \(K\) 通过该算子的正确性判定且 \(g(K, R) \ge \tau\) 时，战役成功。自我演化（§3.5）精确地迭代这个映射，令 \(R_{n+1} = K_n\)，并在确认增益掉到测量噪声底时停止：\(g(K_{n+1}, K_n) - 1 < \epsilon\)，\(\epsilon = 1\%\)–\(5\%\)。这个映射不依赖于谁写了 \(R\)，所以专家库、另一个系统的 kernel，以及 ForgeKernel 自己的输出，都是可接受的参考。

顺着这两层，架构 agent 分析参考、撰写并修订课程、诊断轨迹；编码 agent 写下每一行 kernel 代码。三个制品居中调解：按算子存放的里程碑，只存决策和测得的目标、不存实现正文；算子无关的硬件机制知识库，唯一含有代码的制品；以及把前者变成后者的参考分析策略。harness 和提示里不放任何算子专用信息。

两条规则支配设计：代码不许越过边界，由一个确定性驱动器在提示之外强制执行——参考作为载荷送出，只有 GPU 作业才能拆封；agent 自己的声称不是证据，所以每一个数字、每一道门、每一条知识库条目都落在一次测量上（附录 B、C）。

**图 2：ForgeKernel 架构。** 架构 agent 把参考 kernel 的硬件可观察行为变成里程碑序列；编码 agent 一次实现一行决策，每一行都在同一 GPU 作业里相对参考计时。门的结果分三路：重试该行；重新分析并修订课程；或者交付。一个已交付的 kernel 成为下一场战役的参考。图中：NCU / 时间线探针 → 参考分析 → 里程碑构建（\(M_0\) 为架构决策，\(M_1\)、\(M_2\) 其后）→ 编码 agent 实现决策行 → 候选在同一 GPU 作业中相对参考计时 → 里程碑门：失败则回到该行，平台期则重新分析并重新蒸馏，通过则交付。收获把经探针认证的配方写入知识库。

### 3.2 参考分析与里程碑构建

参考分析从硬件可观察行为恢复参考的架构。一次 NCU 扫描给出启动结构、占用率、流水线利用率和停顿分解，架构 agent 据此推断 kernel 数量、warp 角色、流水深度和受限资源；串行、延迟受限的 kernel 还会收到一次时间线探针；仪器无法分辨的东西保持为显式的未决问题。架构 agent 把这些观察变成有序的里程碑序列 \(M_0 \to M_k\)，每个决策用自然语言写出，带上动作、目的、触发条件、代价和依赖。自然语言是载体，因为参考之间没有共同的实现语言——它们用 CUDA C++（Thakkar et al., 2026）、Triton（Tillet et al., 2019）、TileLang（Wang et al., 2025b）或 ThunderKittens（Spector et al., 2024）写成，或者只以二进制交付（Chetlur et al., 2014; NVIDIA Corporation, 2026a）——于是从 DeepGEMM 的 CUDA 上读出的决策，会在编码 agent 的 CuTe DSL 里实现，为一个算子锻出的配方也可以开给另一个算子。没有触发条件的决策——也就是没有激活它的剖析器症状——是一条没有支撑的声称。

\(M_0\) 携带最要紧的决策：算法合同（分解、数据流、中间张量 ABI、精度）和硬件架构（tile 形状、warp 角色、寄存器预算、共享内存布局、MMA atom），并对照一份从参考签名誊写来的资源拓扑合同做检查；\(M_1\) 到 \(M_k\) 再在这个拓扑上调节阶段数、swizzle 模式、等待深度这类杠杆，因为过晚改变拓扑会迫使活性计划返工。\(M_0\) 最先到来，但不是终局：一个需要改变拓扑才能越过的天花板是 \(M_0\) 的缺陷，架构 agent 会重新打开它（§3.3）。每个里程碑以一道门结束，门里有正确性行、带来源的性能行，以及一条机制验证行——一个锚定在剖析器上的谓词，用来把真正落地的机制和碰巧出现的加速分开。未解决的行必须以自己的证据了结为 DEFERRED、INFEASIBLE 或 N/A（附录 B）；决策图在目标之下耗尽，则作为里程碑缺陷退回架构 agent。

### 3.3 ForgeKernel 的嵌套循环

内循环里，编码 agent 一次实现一行决策，单个 GPU 作业用一次独立的 FP32 正确性检查、配对计时和剖析来评估它；正确性失败会被回退，\(M_0\) 作为一个架构单元来评估。外循环是单回合 agent 所没有的：连续 3 次迭代没有新的最优、决策图耗尽，或门陷入僵局时，架构 agent 重新进入，给杠杆定价，跑裁决探针以分离单个效应和耦合效应，并修订课程。增量补丁就地继续；根基补丁从最早的依赖决策重新开始——\(M_0\) 里做出的拓扑决策就是这样被重新打开，而不是在它下面继续调。一个候选只有在同一作业内的增益清过噪声底 1%–5% 时才被接受；没有任何测量能记到它账上的补丁会被回退，并记成一条否定记录。

### 3.4 知识的锻造与复用

知识库里的每条配方记录一个机制、它的触发症状、一个最小可运行示例，以及已知的坑，按 GPU 架构组织。agent 可读的来源只引用文档、公开 API 符号和探针裁决；参考专用的名字留在仅供人读的档案里，由确定性 lint 隔开。因此一条配方不保留它被锻时所针对的参考的痕迹，这才允许 attention 反向战役里写下的配方，被一场只共享硬件的 GEMM 战役读到。

准入要求测量：来自任一 agent 的提案，只有在目标架构上的一次 GPU 探针认证之后才能入库。一场达到目标的战役，并不因此确立其中提出的每一个解释；认证把真正起作用的机制和仅仅伴随出现的机制分开，所以积累下来的是硬件行为，而不是某一次成功配置的记录。处方是回流路径：架构 agent 在需要配方的行里按名字引用它们，战役账本同时记录一场战役发明的配方，以及它交付的 kernel 落地了哪些外来配方，于是知识库的贡献是可审计的，而不是被断言的。死路以同样的方式积累，进一张按症状和重试前提索引的否定表。

### 3.5 用自我参考做迭代性能改进

因为参考只被当作一个可测量的二进制使用，ForgeKernel 已经接受的 kernel 可以像厂商二进制一样充当这个角色。ForgeKernel 把每一个被升格的 kernel——它自己的，或 `/goal` agent 的——密封起来，作为架构参考供给下一场战役。那场战役对着它重做分析：重读 NCU 签名，把目标和机制行重新锚定到被升格的参考上，并施加分阶段目标增益 +5%。源码不越过代际边界；知识库越过。这就是图 2 的两条反馈边并不冗余的原因。升格抬高性能锚点，知识库降低到达锚点的成本；当确认的分阶段增益掉到同一作业噪声底以下，或预算用尽时，链条停止。

---

## 4 实验

### 4.1 实验设置

Hopper 实验跑在 H100 80 GB（SXM，132 个 SM）上，Blackwell 跑在 B200（148 个 SM）节点上；编码 agent 写 CuTe DSL。效率是相对参考的设备时间，二者在同一作业里计时，每个候选在计时之前都要通过该算子专用的正确性判定。架构 agent 跑 Claude Fable 5.1，编码 agent 跑 Claude Opus 5，二者都是原装 Claude Code agent。战役在达到目标增益时停止，即使还有更多空间。因此图 3 测量的是一个密封二进制能否被追平，而不是系统能超出它多远；`/goal` agent 超出参考更多的地方，它的 kernel 是更好的起点，§4.3 在两个这样的任务上升格了它。可复现性声明和附录 G 给出确认协议、判定器、版本钉扎和仪器；附录 H 给出每场战役的成本。

后面三小节的增益不能互换：§4.2 是相对密封参考，§4.3 是相对上一代（其累计列把它们复利起来），§4.4 是相对一个固定的专家库。

### 4.2 追上专家库，超过 SOTA 系统

这一块里的每场战役得到的东西和 `/goal` agent 完全一样：可以运行和剖析参考，唯一被拒绝的是源码。因此比较隔离的是 §3 的机制，不是信息。它们还共享同一套 harness——同一个驱动器、里程碑模板、判定器和测量协议，Hopper 和 Blackwell 都用，GEMM 和 attention 都用，对库用，也对另一个系统的 kernel 用。其中没有任何东西专为一个算子或一块 GPU 特化；一场战役和下一场的差别，只在任务卡（算子、形状、密封参考、目标）以及开场时知识库的状态。

**图 3：ForgeKernel 对专家库。** 每个点是一个算子、形状和 GPU，每格取最好的战役，按密封参考所属的库分组。增益是停止后确认的、相对该参考的加速；竖线是持平。形状见表 3。

Hopper H100（蓝点）与 Blackwell B200（方点）相对密封专家库的增益：

| 库 | 算子 | GPU | 增益 |
| --- | --- | --- | --- |
| FlashAttention-4 | attn fwd（GQA 16:1） | H100 | +0.3% |
| FlashAttention-4 | attn bwd（MHA） | H100 | +5.1% |
| FlashAttention-4 | attn bwd（GQA 8:1） | H100 | +2.7% |
| FlashAttention-4 | attn bwd（MHA） | B200 | +2.5% |
| DeepGEMM | fp8 gemm | H100 | +0.8% |
| DeepGEMM | fp8 grouped gemm | H100 | +0.3% |
| DeepGEMM | dsa indexer fwd | H100 | +2.9% |
| DeepGEMM | fp4 grouped gemm | B200 | +0.6% |
| cuBLAS | gemm（fp16） | H100 | +0.9% |
| cuDNN 9 | conv2d fwd | H100 | −0.5% |
| cuDNN 9 | conv2d wgrad | H100 | +1.0% |
| cuDNN 9 | conv2d dgrad | H100 | +0.7% |
| cuDNN 9 | conv3d fwd | H100 | +7.6% |
| cuDNN-frontend | dsa indexer topk | B200 | +2.2% |
| cuDNN-frontend | dsa sparse bwd | B200 | +1.4% |
| cuDNN-frontend | dsa score recompute | B200 | +6.2% |
| cuDNN-frontend | dsa indexer bwd | B200 | +2.5% |
| FlashMLA | dsa sparse fwd | B200 | +2.0% |
| FlashInfer | gdn prefill fwd | B200 | −0.2% |
| FlashQLA | gdn prefill bwd | B200 | +1.1% |
| DeepSeek-V4.1 | sparse mla prefill | B200 | −1.3% |
| DeepSeek-V4.1 | fused attn rope cast | B200 | −3.7% |

**面对最强的人类专家。** 相对专家库，ForgeKernel 在图 3 的 22 格里追平或超过 18 格，另有 2 格落在 1% 以内，几何平均 +1.6%，最好 +7.6%。这在两代 GPU、每一个算子族上都成立：GEMM 和卷积格，面对 cuBLAS、DeepGEMM 和 cuDNN 多年的调优，落在 −0.5% 到 +7.6% 之间；FlashAttention-4 在其 Hopper 前向与反向、以及 Blackwell 反向上被超过；DSA 链的每一个反向 kernel 都胜过 cuDNN-frontend，最高 +6.2%。边际很小，是因为被测的对象——这个领域里工程化程度最高的一批 kernel——只以二进制被观察。两格停在参考之下不到 1%；其中之一，FlashInfer 的 GDN prefill（−0.2%），是 §4.3 会回到的任务。离得最远的两个是最新的：DeepSeek-V4.1 的 sparse MLA prefill（−1.3%），以及把 RMSNorm、两遍 RoPE、稀疏 attention 和一次 FP8 cast 融进同一次启动的 kernel（−3.7%）。DeepSeek 随 V4.1 发布它们（DeepSeek-AI, 2025c），在本稿提交前 15 天，没有描述它们的论文，也没有别的库重新实现它们。ForgeKernel 需要从参考得到的是它在 GPU 上的行为；一个才发布几天的 kernel，和一个被研究了多年的 kernel，在这一点上暴露得一样充分。

**面对最强的 kernel agent。** 追上专家库并不能说明 ForgeKernel 好过其他自动化系统，那些系统也许也能追上。图 4 直接检验这一点，把三个生成系统已发表的最强 kernel 当作密封参考：K-Search（Cao et al., 2026）在它自己的 FlashInfer-Bench 负载和 GPUMode TriMul 任务上的获胜 kernel，跑在 Hopper 上；Kernel Design Agents（Liu et al., 2026）和 CAKE（Ye et al., 2026）的 kernel，跑在 Blackwell 上。ForgeKernel 在全部 7 个上处于持平或更高，几何平均增益 +39.4%，最好 +170.6%。边际跟着每个系统留下的余量走。K-Search 的四个 kernel 留下的最多，其中包括其作者报告为快过 GPUMode 排行榜上每一个顶尖提交的 TriMul kernel（Cao et al., 2026）。

**图 4：ForgeKernel 对 SOTA 生成系统的 kernel。** 每格最好战役；坐标轴含义同图 3，但每块 GPU 用自己的刻度。

| 系统 | GPU | 算子 | 相对密封系统 kernel 的增益 |
| --- | --- | --- | --- |
| K-Search | H100 | gqa paged decode | +24.5% |
| K-Search | H100 | mla paged decode | +170.6% |
| K-Search | H100 | mla paged prefill | +146.1% |
| K-Search | H100 | trimul | +8.7% |
| Kernel Design Agents | B200 | dsa sparse attn paged | +7.0% |
| Kernel Design Agents | B200 | gdn prefill varlen | +5.8% |
| CAKE | B200 | recurrent kda prefill | +0.4% |

Kernel Design Agents 的 KDA-v0.5 kernel（Zou et al., 2026）留下的较少——其作者测得它们相对 MLSys 2026 FlashInfer 竞赛获胜提交，在 DSA 稀疏 attention 和 GDN prefill 上分别为 41% 和 69%——ForgeKernel 两者都超过了。CAKE 的持久 KDA kernel 被追平在 +0.4%，是持平而不是一场胜利。

**跨架构测试。** 在 Ascend 910C 上，我们把 NCU 换成 msprof，循环照原样跑，每个算子的参考是华为当前算子库里最快的实现。在 CANN-Bench 1.1.2 的全部 8 个 level-4 算子上，最终 kernel 领先排行榜上的最佳公开提交，加速比的几何平均为 +33.1%（图 6，附录 I）。

### 4.3 自我演化

与参考持平是 §4.2 停下的地方，不是系统停下的地方。ForgeKernel 需要参考只是为了有一个二进制可以剖析，所以任何 kernel 都能充当参考，包括一个没有人研究过其源码的、机器写出的 kernel。自我演化就是把这条规则反复使用：取一个任务上迄今已知最好的 kernel，不管谁写的，把它升格为参考，再对它跑同一条未改的流水线。把 `/goal` 再跑得更长并不是替代方案，因为它的读数已经收敛，而且它不跨会话携带任何东西（§2）；把它的 kernel 密封成下一场参考，并把更早的发现往前带，恰恰是 ForgeKernel 加上的两种机制，表 1 测量的就是它们。

**表 1：通过升格参考实现的自我演化。** 每个任务在其链条开始时的已知最好 kernel，被密封为下一场参考，目标 +5%。括号是被升格 kernel 相对专家库的增益；累计列把它和新战役的增益复利起来。底行连续升格两次，先 gen-1 再 gen-2。

已知最好的是 `/goal` agent 的，升格它：

| 算子 | 参考（相对专家） | 增益 | 累计 |
| --- | --- | --- | --- |
| gdn prefill fwd | agent（+82.6%） | +9.5% | +100% |
| attn fwd（GQA 16:1） | agent（+22.8%） | +5.0% | +29% |

已知最好的是 ForgeKernel 自己的，升格我们自己的：

| 算子 | 参考（相对专家） | 增益 | 累计 |
| --- | --- | --- | --- |
| dsa sparse fwd | gen-0（+2.0%） | +7.2% | +9% |
| gdn prefill bwd | gen-0（+1.1%） | +2.0% | +3% |

连续两次升格 ForgeKernel 自己的 kernel，从左到右：

| 算子 | 参考（相对专家） | 增益 | 累计 |
| --- | --- | --- | --- |
| attn fwd（GQA 16:1） | gen-0（+0.3%） | +9.9% | +10% |
| attn fwd（GQA 16:1） | gen-1（+10.3%） | +4.2% | +15% |

**在 `/goal` agent 领先的地方，它的 kernel 就是参考。** 在 GDN prefill 和 GQA attention 前向上，agent 的 kernel 是已知最好的（表 5）。密封后交给 ForgeKernel 当参考，每一个都被超过（表 1）：结果分别站在 FlashInfer 之上 +100%，以及 FlashAttention-4 之上 +29%。ForgeKernel 看到的并不比 agent 多，二者都不读专家源码。它带来的是 agent 的循环所缺的：一个由其他算子的战役付钱测得的、经探针认证的硬件行为知识库（§3.4），以及一条在进展停滞时从架构和编排这一层重新进入的循环，而不是在它最初选定的架构下面调参。增益不大，但 agent 自己的循环已经找不到它了。

**在 ForgeKernel 领先的地方，它升格自己的 kernel。** 在 DSA 稀疏 attention 前向和 GDN 反向上，ForgeKernel 的 kernel 是已知最好的，于是我们把它升格为它自己任务的参考，再跑一遍流水线。稀疏 attention 链相对前代 +7.2%，GDN 反向链 +2.0%（表 1）。ForgeKernel 在完全没有外部参考的情况下自我演化。升格也可以重复：在 GQA attention 前向上，我们连续两次升格 ForgeKernel 自己的 kernel，每一代密封为下一代的参考，结束时比 FlashAttention-4 高 +15%（表 1）。

**为什么升格还能继续找到增益。** 升格的作用像迭代局部搜索里的一次重启：每一代从迄今找到的最好点出发，并面对一根更高的杠，于是搜索不会停在上一场战役的停止规则让它停下的地方。kernel 优化需要这次重启，因为一旦选定一个架构，它内部的每个杠杆都可以估价为零，而一个更好的拓扑仍然存在。DSA 稀疏 attention 那一代是写开的案例：增益来自把 kernel 的第一个 \(M_0\) 决策重新打开为持久循环，而不是来自任何被调过的参数（附录 I）。

**知识库的使用。** 升格能复利，只是因为一场战役学到的东西活进了下一场。战役之间 harness 没有变（§4.2），所以知识库是唯一在增长的输入，早场战役和晚场战役的差别必须从这里来。它的账本有 71 个条目、9 个类别，诞生于 79 次运行；有 553 个（条目, 运行）配对，其中被开出的条目落地进了交付的 kernel。流量随时间增长：切成连续三段，最早的运行每场落地 4.9 个外来条目，中间 4.1 个，最近 9.4 个。发明集中在缺口所在之处——attention 反向（17 条）和 DSA 稀疏前向（14 条）几乎贡献了知识库的一半——而图 3 里边际最大的两格，conv3d 前向和 DSA 分数重计算，来自什么都没发明、只落地外来条目的战役。图 8（附录 I）按战役追踪这两个方向。

### 4.4 消融

**参考消融。** 参考是 ForgeKernel 得到的唯一一项任务专用输入，所以图 5 在 3 个算子上按三档消融它：没有参考（一份 PyTorch 实现，加上参考设备时间这一个数字）；架构 agent 可以运行、剖析和调查的密封二进制；以及这份二进制再加上可读源码。没有参考时，ForgeKernel 在三个算子上全部停滞。它们都没有多少公开设计可以回忆——一个年轻的循环算子、一个反向 pass、一个稀疏 kernel——所以携带信息的是参考；给了参考，在同样封顶的协议下，三个都恢复了。让源码可读有适度帮助（几何平均 +2.5% 对 +1.5%；源码可读的 attention 反向战役在 11 次迭代后撞上元步骤上限），但密封二进制已经在三个算子上都到达库的 1% 以内。架构 agent 主要需要从参考得到的是它的行为。

**图 5：参考消融。** 每个算子一组，每组三根柱：无参考、可剖析的密封二进制（默认）、源码可读。每根柱都是相对同一个专家库的增益；前两臂封顶为三次元步骤。下表按图中标注转写（从左到右：无参考、密封二进制、源码可读）。几何平均与正文「源码可读 +2.5%、密封 +1.5%」一致；「密封二进制已在三个算子上到达库的 1% 以内」指的是没有任何一格慢过库超过 1%（唯一低于库的密封结果是 GDN 的 −0.2%）。

| 算子 | 无参考 | 密封二进制（默认） | 源码可读 |
| --- | --- | --- | --- |
| Attn bwd GQA | −29.8% | +2.7% | +3.1% |
| GDN prefill fwd | −14.2% | −0.2% | +4.0% |
| DSA sparse fwd | −22.1% | +2.0% | +0.4% |
| 几何平均 | −22.3% | +1.5% | +2.5% |

**参考强度则会传播。** 表 2 用同一条流水线，分别对着 DSA fp8 indexer 前向的两个真实参考各跑一场：DeepSeek 在 DeepGEMM 里的生产 kernel，以及 NVIDIA 在 cuDNN-frontend 里的 kernel，后者慢 29%。得到的 kernel 跟着各自的参考走，同一算子、同一 GPU 上两者相差 18%。两场都超过了自己的参考，较弱的那场超过得更多，但对着 cuDNN-frontend 建出的 kernel 仍然慢于 DeepGEMM 本身：弱参考上方的余量，不够用来找到强参考所体现的东西。ForgeKernel 升到它的参考的高度，这就是为什么升格已知最强的 kernel（§4.3）是它继续往上爬的方式。

**表 2：DSA fp8 indexer 前向上的参考强度。** 对该算子的两个真实参考各一场战役。ForgeKernel vs ref 是配对的同一作业增益；ForgeKernel（ms）是三次确认作业的中位数。

|  | 参考 | 参考（ms） | ForgeKernel（ms） | ForgeKernel vs ref |
| --- | --- | --- | --- | --- |
| 强 | DeepGEMM 2.6.1 | 1.861 | 1.808 | +2.9% |
| 弱 | cuDNN-frontend 1.28 | 2.408 | 2.138 | +12.7% |

---

## 5 相关工作

**专家 kernel 与 kernel 基础设施。** ForgeKernel 吸收的参考来自专家 kernel 工程：FlashAttention 一系（Dao et al., 2022; Dao, 2023; Shah et al., 2024; Zadouri et al., 2026）、服务库（Ye et al., 2025; DeepSeek-AI, 2025c;a）、厂商库（Chetlur et al., 2014; NVIDIA Corporation, 2026a），以及它们下面的基础设施（Thakkar et al., 2026; Tillet et al., 2019; Spector et al., 2024; Wang et al., 2025b）；超优化器和 megakernel 编译器（Wu et al., 2025; Cheng et al., 2025）在没有 LLM 的情况下探索这个空间的一部分。ForgeKernel 消费这个生态，而不是与它竞争。

**单回合生成。** 大多数生成系统（Yu et al., 2026）把优化知识留在一个回合之内：对着编译器和剖析器反馈的测试时 agent 循环（Chen et al., 2025; Wang et al., 2025a; Dong et al., 2025; Wei et al., 2025; Han et al., 2026; Guo et al., 2026），在 kernel 种群上的进化搜索（Lange et al., 2025; Novikov et al., 2025; Wiedemann et al., 2026; Yan et al., 2026; Du et al., 2026b），以及把知识写进权重的强化学习或微调（Baronio et al., 2025; Li et al., 2026; 2025; Tehrani et al., 2026; Du et al., 2026a; Fu et al., 2026; AI at Meta, 2025; Yuksekgonul et al., 2026）。回合结束时这些知识消失，训练停止时则冻结。ForgeKernel 改为把它留在一个显式的、经测量认证的制品里，并补上模型这一轴：更强的编码 agent 会改进每一场战役。

**知识积累与基于参考的学习。** 近期工作分享了这一前提的若干部分（Sun et al., 2026; Spoczynski et al., 2026; Ma et al., 2026; Luo et al., 2026; Chen et al., 2026; Cao et al., 2026; Zhang et al., 2026; Ye et al., 2026; Liu et al., 2026）。AVO（Chen et al., 2026）把知识隐含在种群谱系里，CAKE（Ye et al., 2026）把编译器规则积累在一个手工搭建的 schedule IR 里，Kernel Design Agents（Liu et al., 2026）在我们的若干任务上发布了已发表的、由 agent 写出的最强 kernel，我们把它们当作要去超越的参考。此前没有系统同时做到这三件事：只通过硬件可观察行为吸收一个参考，跨算子积累经测量认证的知识，以及把被接受的 kernel 升格为下一场战役的参考。

---

## 6 讨论与局限

ForgeKernel 给编程 agent 一个外部的架构层：它从密封参考的硬件测量推断架构决策，用锚定在测量上的目标引导编码 agent 走过里程碑，并把被接受的 kernel 和经认证的发现变成后续战役的参考和配方。在 27 个算子上，它相对专家库和其他系统的 kernel 达到 +10% 的几何平均增益，自我演化再最多加上 +9.9%。

三个局限框住这个结果。必须有参考：只给一份 PyTorch 实现和一个设备时间数字，ForgeKernel 在 §4.4 的三个算子上全部停滞；它超越的是一份已经存在的实现，而不是写出第一份。平台期是估计，不是证明：自我演化停在同一作业的噪声底，所以平台期表明这条循环没有再找到东西，并不表明不存在更多东西。覆盖面是两代 NVIDIA 加一块 Ascend 上的四个算子族；与这些不同的稀疏模式和存储层次没有被测试。

引言所描述的那条轨迹也适用于 ForgeKernel 自己：harness 里那些编码「模型还不会做什么」的部分，应当随着模型变强而变得不必要；应当比它们活得更久的，是一份经测量认证的、关于某一块具体 GPU 如何行为的记录。

---

## 可复现性声明

本文的每一条声称都是一个设备时间之比，由本文描述的 harness 产生。其背后的测量程序写在 §4.1 和附录 G：参考与候选在同一 GPU 作业里计时；最终声称取最多 3 次同一作业读数的中位数；算子专用的正确性判定——对照一个独立于专家 kernel 的 fp32 参考，并有一道围栏拒绝任何调用或转发到该 kernel 的候选——在任何计时之前运行。每一个参考都钉在作业镜像里，并在我们自己的作业中重测。按数值类别的判定容差在附录 D，钉扎的参考版本和入口点在附录 E，每个任务和每场战役的问题形状在附录 F，每场战役的实测成本——架构 agent 再次进入次数、GPU 小时、活跃小时和墙钟小时——在附录 H。§2 的 `/goal` agent 基线是默认模式下的原装 Claude Code CLI，协议如该节所述；其中没有任何部分是我们自己的代码。有两类结果不能仅凭本文复现。昇腾数字由 CANN-Bench 的托管评分服务计算，只能在那里复现。战役本身跑的是我们不控制其权重的前沿模型，所以重跑会采样到另一条轨迹；重跑应当复现的是测量，不是轨迹。这就是为什么这里的每个数字都是相对钉扎参考的同一作业之比，而不是一个绝对时间。

## AI 使用声明

本工作研究的是 AI 生成的代码，因此生成式 AI 既是研究对象，也是研究工具。每一个被报告为 ForgeKernel 输出的 GPU kernel 都由 §3 描述的编码 agent 写成，每一门里程碑课程都由架构 agent 写成；没有作者手写或手调任何一个出现在结果里的 kernel。§2 的 `/goal` agent kernel 同样由一个未修改的商业编程 agent 产出。所有这些制品只有在 §4.1 的自动正确性判定和计时协议接受之后才被计入，作者还审计了每一个被计为胜利的 kernel，检查用精度换速度的交易。

生成式 AI 也被用来协助 harness 实现，以及本稿的起草和编辑。作者核验了实验代码，重跑了每一个被报告数字背后的测量，并审阅了全部文本。我们对这项工作的最终内容负责，包括其文本、声称和制品。

---

## 参考文献

以下条目按原文著录，不翻译。访问日期与版本号保留论文写法。

- AI at Meta. KernelLLM. https://huggingface.co/facebook/KernelLLM, 2025. Model card. Accessed 2026-09.
- Carlo Baronio, Pietro Marsella, Ben Pan, Simon Guo, and Silas Alberti. Kevin: Multi-turn RL for generating CUDA kernels, 2025.
- Shiyi Cao, Ziming Mao, Joseph E. Gonzalez, and Ion Stoica. K-Search: LLM kernel generation via co-evolving intrinsic world model, 2026.
- Terry Chen, Bing Xu, and Kirthi Devleker. Automating GPU kernel generation with DeepSeek-R1 and inference time scaling. NVIDIA Developer Blog, 2025. https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/.
- Terry Chen, Zhifan Ye, Bing Xu, et al. AVO: Agentic variation operators for autonomous evolutionary search, 2026. NVIDIA.
- Xinhao Cheng, Zhihao Zhang, Yu Zhou, et al. MPK: A compiler and runtime for mega-kernelizing tensor programs, 2025.
- Sharan Chetlur, Cliff Woolley, Philippe Vandermersch, Jonathan Cohen, John Tran, Bryan Catanzaro, and Evan Shelhamer. cuDNN: Efficient primitives for deep learning, 2014.
- Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning, 2023.
- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In NeurIPS, 2022.
- DeepSeek-AI. DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model, 2024.
- DeepSeek-AI. DeepGEMM: Clean and efficient FP8/FP4 GEMM kernels with fine-grained scaling. https://github.com/deepseek-ai/DeepGEMM, 2025a. Accessed 2026-09.
- DeepSeek-AI. DeepSeek-V3.2-Exp: Boosting long-context efficiency with DeepSeek sparse attention. https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf, 2025b. Tech report, September 2025.
- DeepSeek-AI. FlashMLA: Efficient attention kernels for DeepSeek models. https://github.com/deepseek-ai/FlashMLA, 2025c. SM90/SM100 dense and sparse attention kernels; the DeepSeek-V4.1 sparse MLA prefill and fused norm+RoPE+attention+cast kernels are upstream ba89a34. Accessed 2026-09.
- Juncheng Dong, Yang Yang, Tao Liu, et al. STARK: Strategic team of agents for refining kernels, 2025.
- He Du, Qiming Ge, Jiakai Hu, et al. Kernel-smith: A unified recipe for evolutionary kernel optimization, 2026a.
- Weihua Du, Jingming Zhuo, Yixin Dong, et al. AdaExplore: Failure-driven adaptation and diversity-preserving search for efficient kernel generation, 2026b.
- Dayuan Fu, Mohan Jiang, Tongyu Wang, et al. davinci-kernel: Co-evolving skill selection, summarization, and utilization via RL for GPU kernel optimization, 2026.
- Jihu Guo, Sitian Lu, Tenghui Ma, et al. SpecGen: Accelerating agentic kernel optimization with speculative generation, 2026.
- Yaochen Han, Ke Fan, Hongxu Jiang, et al. EGG: An expert-guided agent framework for kernel generation, 2026.
- Kimi Team. Kimi Linear: An expressive, efficient attention architecture, 2025. Introduces Kimi Delta Attention (KDA).
- Robert Tjarko Lange, Aaditya Prasad, Qi Sun, Maxence Faldor, Yujin Tang, and David Ha. The AI CUDA Engineer: Agentic CUDA kernel discovery, optimization and composition. Sakana AI technical report, 2025. https://pub.sakana.ai/ai-cuda-engineer/paper/.
- Shangzhan Li, Zefan Wang, Ye He, et al. AutoTriton: Automatic Triton programming with reinforcement learning in LLMs, 2025.
- Xiaoya Li, Albert Wang, Guoyin Wang, Jiwei Li, and Chris Shum. CUDA-L1: Improving CUDA optimization via contrastive reinforcement learning. In ICLR, 2026. arXiv:2507.14111.
- Sihao Liu, Ligeng Zhu, et al. Kernel Design Agents (KDA): An agent-centric workflow for high-performance CUDA kernels. https://github.com/NVlabs/kda, 2026. NVIDIA and MIT HAN Lab; software release.
- Mao Luo, Hongbin Li, Feng Lin, Hanling Yi, and Zhe Huang. Agentic kernel optimization: Generating state-of-the-art GPU kernels without hand-written CUDA, 2026.
- Xing Ma, Yangjie Zhou, Wu Sun, et al. CuBridge: An LLM-based framework for understanding and reconstructing high-performance attention kernels, 2026. ACL 2026.
- Alexander Novikov, Ngân Vũ, Marvin Eisenberger, et al. AlphaEvolve: A coding agent for scientific and algorithmic discovery, 2025.
- NVIDIA Corporation. NVIDIA H100 Tensor Core GPU architecture whitepaper. https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c, 2022. Accessed 2026-09.
- NVIDIA Corporation. NVIDIA Blackwell architecture technical overview. https://resources.nvidia.com/en-us-blackwell-architecture, 2024. Accessed 2026-09.
- NVIDIA Corporation. cuDNN frontend API. https://github.com/NVIDIA/cudnn-frontend, 2026a. Version 1.29. Accessed 2026-09.
- NVIDIA Corporation. Parallel Thread Execution ISA, version 9.4. https://docs.nvidia.com/cuda/parallel-thread-execution/, 2026b. wgmma.mma_async, cp.async.bulk.tensor (TMA), and tcgen05 sections. Accessed 2026-09.
- Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. FlashAttention-3: Fast and accurate attention with asynchrony and low-precision, 2024.
- Benjamin F. Spector, Simran Arora, Aaryan Singhal, Daniel Y. Fu, and Christopher Ré. ThunderKittens: Simple, fast, and adorable AI kernels, 2024.
- Marcin Spoczynski, Daniel Fleischer, Moshe Berchansky, et al. Xe-Forge: Multi-stage LLM-powered kernel optimization for Intel GPU, 2026.
- Qitong Sun, Jun Han, Tianlin Li, et al. KernelSkill: A multi-agent framework for GPU kernel optimization, 2026.
- Ali Tehrani, Yahya Emara, Essam Wissam, et al. Fine-tuning GPT-5 for GPU kernel generation, 2026. Makora.
- Vijay Thakkar, Pradeep Ramani, Cris Cecka, Aniket Shivam, Honghao Lu, Ethan Yan, Jack Kosaian, Mark Hoemmen, Haicheng Wu, Andrew Kerr, et al. CUTLASS: CUDA templates for linear algebra subroutines. https://github.com/NVIDIA/cutlass, 2026. Version 4.8; includes CuTe layout algebra and the Python CuTe-DSL. Accessed 2026-09.
- Philippe Tillet, Hsiang-Tsung Kung, and David Cox. Triton: An intermediate language and compiler for tiled neural network computations. In MAPL, 2019.
- vincent317. Crossing the expert line without expert machinery. https://github.com/vincent317/Crossing-The-Expert-Line-Without-Expert-Machinery, 2026. GitHub repository. Accessed 2026-09.
- Jianghui Wang, Vinay Joshi, Saptarshi Majumder, et al. GEAK: Introducing Triton kernel AI agent & evaluation benchmarks, 2025a.
- Lei Wang, Yu Cheng, Yining Shi, Zhengju Tang, Zhiwen Mo, Wenhao Xie, Lingxiao Ma, Yuqing Xia, Jilong Xue, Fan Yang, and Zhi Yang. TileLang: A composable tiled programming model for AI systems, 2025b.
- Anjiang Wei, Tianran Sun, Yogesh Seenichamy, et al. Astra: A multi-agent system for GPU kernel performance optimization, 2025.
- Nina Wiedemann, Quentin Leboutet, Michael Paulitsch, Diana Wofk, and Benjamin Ummenhofer. KernelFoundry: Hardware-aware evolutionary GPU kernel optimization. In ICML, 2026. arXiv:2603.12440.
- Mengdi Wu, Xinhao Cheng, Shengyu Liu, et al. Mirage: A multi-level superoptimizer for tensor programs. In OSDI, 2025. arXiv:2405.05751.
- Minghao Yan, Bo Peng, Benjamin Coleman, et al. PacEvolve: Enabling progress-aware consistent evolution, 2026.
- Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving Mamba2 with delta rule, 2024. ICLR 2025.
- Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, and Luis Ceze. FlashInfer: Efficient and customizable attention engine for LLM inference serving. In MLSys, 2025. Best Paper Award. arXiv:2501.01005.
- Zihao Ye, Yingyi Huang, Hongyi Jin, Bohan Hou, Junru Shao, Zhongming Yu, Jinqi Chen, Meghan Cowan, Shiyi Cao, Shanli Xing, Hanfeng Chen, Vinod Grover, Tianqi Chen, and Luis Ceze. CAKE: Compiler-agent co-design for frontier kernel evolution, 2026.
- Yang Yu, Peiyu Zang, Chi Hsu Tsai, et al. Towards automated kernel generation in the era of LLMs, 2026. Survey.
- Mert Yuksekgonul, Daniel Koceja, Xinhao Li, et al. Learning to discover at test time, 2026. TTT-Discover.
- Ted Zadouri, Markus Hoehnerbach, Jay Shah, Timmy Liu, Vijay Thakkar, and Tri Dao. FlashAttention-4: Algorithm and kernel pipelining co-design for asymmetric hardware scaling, 2026.
- Shuoming Zhang, Qiuchu Yu, Yangyu Zhang, et al. Learning when to optimize: Verified optimization skills from expert GPU-kernel lineages, 2026. KLineage; skills extracted from open reference source, static one-shot.
- Dongyun Zou, Yixin Dong, Junxian Guo, et al. KDA-v0.5 kernels for the MLSys 2026 FlashInfer AI kernel generation contest. https://github.com/mit-han-lab/mlsys2026-flashinfer-contest-solution, 2026. Speedups over the contest winners re-measured by the authors under the flashinfer-bench CUPTI protocol, not by the contest evaluator. Accessed 2026-09.

---

## 附录 A 硬件背景

**两层的完整内容。** 架构决策通过以下各项决定全局结构：（i）分块层次；（ii）流水阶段数与缓冲区放置；（iii）warp 角色；（iv）共享内存和寄存器中的张量布局；（v）重计算与存储的边界。战术知识通过以下各项决定一个固定结构的执行：（i）warp-group MMA 的发射；（ii）TMA 描述符的构造；（iii）mbarrier 协调；（iv）按角色的寄存器预算。

**硬件行为中的架构签名。**（i）占用率和每 warp 寄存器数量揭示角色专门化；（ii）TMA 与 L2 的字节流揭示操作数暂存；（iii）屏障停顿分布揭示流水深度；（iv）启动结构、block 维度和 grid 维度揭示 kernel 分解，以及重计算与存储的边界。

## 附录 B 门的义务与里程碑校验

跨角色交接、收集-供给纪律，以及消费方 warpgroup 的相位，接受封闭词表标签；每个标签都要求锚定在停顿上的验证，以及记录下来的顺序证据。每一条未解决的门行收到三种机器检查裁决之一：（i）DEFERRED，要求证据表明参考达到了该指标、当前杠杆改变不了它，以及一个点名的下游杠杆来接收这项未解决的测试；（ii）INFEASIBLE，要求逐指标的论证，以及排除该目标的证据；（iii）N/A，只适用于结构上不适用。校验器检查每个里程碑版本的各节、五元组、无环推导、来源、机制行和标签义务。补丁在后续迭代上评估，若无法把改进归因于它们则回退；一份预定义合同指定当一个指标上的增益与另一个指标上的损失对换时所使用的不对称代价。

## 附录 C 里程碑构建细节

**架构参考与库基线。** 每场战役区分架构参考和库基线。被测量的架构参考设定停止阈值，并锚定每一条机制行；没有它，harness 拒绝这场战役。库基线是最好的现成调用，提供一个比较点，但不决定终止。这一区分防止对着碰巧方便的那条基线宣布成功——早期一次 FlashAttention 反向循环犯过这种锚定错误。

**决策图。** 里程碑决策之间的推导边构成一个 DAG，从而允许补丁之后的部分重执行。原子组标注标出必须在一次提交里一起落地的决策集合，因为任何一个严格子集都会回退。

**证据规则。** 每个里程碑数字带一个来源标记，`[MEASURED, job N]` 或 `[ESTIMATE]`，并带上它的推导；每个架构决策点名验证其执行的机器可读通道；里程碑校验器检查这两者（附录 B）。运行图 2 两条循环的驱动器，在每一回合之前重写每个 agent 的权限策略。门的锚只由参考测量、硬件规格，或从这些来源导出的表达式组成。工作区隔离、里程碑校验和知识库 lint 独立于提示文本强制执行「无代码」规则，ForgeKernel 把自我打分的探针当作一个单独的缺陷类别。\(M_0\) 的门检查正确性、架构谓词和机制行，不设绝对性能阈值；相对参考的比值在该阶段提供跨作业比较。里程碑校验器和知识库 lint 在战役开始、认证和发布时运行。

## 附录 D 正确性判定

每个判定器施加的逐张量容差遵循算子的数值类别：GEMM 用 DeepGEMM 的聚合准则，attention 梯度用相对最大绝对值界，GDN 和 KDA（Kimi Delta Attention，Kimi Team, 2025）的状态 kernel 用缩放容差加余弦界——它们的输出很小，固定绝对容差会让它们轻易通过。GDN 判定器另外加入相关键和长记忆输入，用来抓住状态衰减的捷径。

## 附录 E 参考版本与入口点

库参考钉在作业镜像里：FlashAttention-4（CuTe DSL）、cuDNN 9 与 cuDNN-frontend 1.28、DeepGEMM、FlashMLA、FlashInfer 和 FlashQLA。测量入口点为：FlashAttention-4 的 forward 与 flash attn bwd；FlashInfer 的 fused chunk gated delta rule；FlashQLA 的 TileLang backward；DeepGEMM 的 fp8 / fp4 mqa logits；FlashMLA 的 flash mla sparse fwd；cuDNN-frontend 的 CuTe DSL radix top-k、稀疏 attention 反向、indexer 反向（三个 kernel 求和），以及 sparse attn 与 sparse indexer 分数重计算的组合。在昇腾上，每个算子的参考是华为当前算子库里最快的实现；CANN-Bench 1.1.2 提供评分基线，其公开仓库在本文写作时发布到 v1.0.0。

系统参考是：K-Search（Cao et al., 2026）在它自己的 FlashInfer-Bench 负载和 GPUMode TriMul 任务上的获胜 kernel；Kernel Design Agents（Liu et al., 2026; Zou et al., 2026）为 MLSys 2026 FlashInfer 竞赛定义发布的 KDA-v0.5 kernel（GDN prefill 与 DSA 稀疏 attention）；以及 CAKE（Ye et al., 2026）的 Kimi Delta Attention prefill kernel。

## 附录 F 战役形状

表 3 列出正文中每个任务和每场战役的问题形状。

**表 3**

`/goal` agent 天花板：

| 算子 | GPU | 形状 |
| --- | --- | --- |
| attn fwd gqa | H100 | B=2，S=4096，H Q=16 / H KV=1，D=128，bf16，causal |
| fa bwd gqa | H100 | B=10，N=4096，H Q=16 / H KV=2，D=64，fp16，causal |
| gdn prefill fwd | H100 | N=8192，h qk=16 / h v=32，d=128，bf16 |
| gdn prefill bwd | H100 | N=8192，h qk=16 / h v=32，d=128，bf16 |
| dsa indexer fwd | H100 | s q=4096，s kv=32768，H=64，D=128，fp8 |
| dsa indexer topk | H100 | logits fp32 [4096 × 32768]，k=1024 |
| dsa sparse fwd | H100 | s q=4096，s kv=32768，h=128，d=512，topk=1024 |
| dsa sparse bwd | H100 | s q=32768，s kv=8192，h=128，d=512，topk=1024 |
| dsa score recompute | H100 | s q=32768，s kv=8192，topk=1024 |
| dsa indexer bwd | H100 | s q=32768，s kv=8192，h=64，d=128，topk=1024 |

ForgeKernel 战役：

| 算子 | GPU | 形状 |
| --- | --- | --- |
| attn fwd（GQA 16:1） | H100 | B=2，S=4096，H=16/1，D=128，bf16，causal |
| attn bwd（MHA） | H100 | B=10，N=4096，H=16，D=64，fp16，causal |
| attn bwd（GQA 8:1） | H100 | B=10，N=4096，H=16/2，D=64，fp16，causal |
| fp8 gemm | H100 | M=N=K=4096，blk 128 |
| fp8 grouped gemm | H100 | G=8，N=K=4096，blk 128 |
| gemm（fp16） | H100 | M=N=K=8192，fp16 |
| conv2d fwd | H100 | N=128，C=K=256，28×28，3×3，s1 |
| conv2d wgrad | H100 | N=128，C=K=256，28×28，3×3，s1 |
| conv2d dgrad | H100 | N=128，C=K=256，56×56，3×3，s2 |
| conv3d fwd | H100 | N=8，C=K=256，8×28×28，3×3×3 |
| dsa indexer fwd | H100 | s q=4096，s kv=32768，H=64，D=128，fp8 |
| dsa indexer topk | H100 | 4096×32768，fp32，k=1024 |
| dsa sparse fwd | H100 | s q=4096，s kv=32768，h=128，d=512，topk=1024 |
| dsa sparse bwd | H100 | s q=32768，s kv=8192，h=128，d=512，topk=1024 |
| dsa score recompute | H100 | s q=32768，s kv=8192，topk=1024 |
| dsa indexer bwd | H100 | s q=32768，s kv=8192，H=64，D=128，topk=1024 |
| gdn prefill fwd | H100 | N=8192，h=16/32，d=128，bf16 |
| gdn prefill bwd | H100 | N=8192，h=16/32，d=128，bf16 |
| gqa paged decode | H100 | workload 7a7fc288（61535 pages） |
| mla paged decode | H100 | workload 5bef8d88（B=64；75145 tokens） |
| mla paged prefill | H100 | workload fe63f292（B=18；15883 q） |
| trimul | H100 | S=1024，D=384，H=128（case 6） |
| attn bwd（MHA） | B200 | B=8，N=4096，H=16，D=128，bf16，causal |
| fp4 grouped gemm | B200 | G=8，M=4096/group，N=K=4096 |
| sparse mla prefill | B200 | s q=4096，s kv=32768，H Q=128，H KV=1，D=512，topk=1024，bf16，attn sink |
| fused attn rope cast | B200 | s q=4096，s kv=32768，H Q=128，H KV=1，D=512，rope=64，topk=1024，n wv group=16，bf16→fp8，attn sink，q norm |
| dsa sparse attn paged | B200 | workload 564007ac（T=8；8462 pages） |
| gdn prefill varlen | B200 | workload a9540651（T=8192，N=20） |
| recurrent kda prefill | B200 | T=8192，H=96，K=V=128，bf16 |

## 附录 G 测量协议

计时使用 CUDA event，覆盖一批背靠背启动；在主机派发会占主导时，改用 CUDA graph 重放或 CUPTI kernel 区间。每个正确性判定器的逐张量容差按算子的数值类别设定，使输出很小的 kernel 不能轻易通过（附录 D）；隔离按 §3.1 执行。编码 agent 在 Hopper 上以 sm 90a 为目标，在 Blackwell 上以 sm 100 为目标。

在 Blackwell 上，GDN 和 DSA 是相对 Kernel Design Agents 比较，而不是相对一个库：它的 GDN kernel 快于 FlashInfer，且没有库实现 DSA。昇腾战役由 CANN-Bench 1.1.2 的托管 harness 计时，而不是由我们的 harness 计时——设备时间只取 kernel，来自 msprof 的 kernel 详情，对照基准自己为 910C 上每个用例定义的基线与硬件界来评分，编译和精度折进分数——它在自己的托管服务上计算，所以这些数字只能在那里复现。

Level 4 是该基准最难的一档，即它的融合 attention 与 RNN kernel。相对最佳公开提交，平均基准分数上的领先（3.87 分）小于加速比上的领先，因为该量表的一半是编译与正确性的地板，其性能的另一半会在候选接近硬件界时饱和。gru 和 lstm 是相对基准基线低于 1× 的两格：CANN-Bench 没有为二者提供参考实现，所以它们的基线由硬件界导出。每一个公开提交也都低于它——领先者达到 0.324× 和 0.410×——而这正是我们相对公开最好成绩的边际最宽的两个算子。

## 附录 H 战役成本

**表 4：每场 ForgeKernel 战役的实测成本。** Arch agent 是架构 agent 的再次进入次数；活跃小时是 agent 小时加 GPU 小时；墙钟小时包含共享集群上的排队。「—」表示原文该格为空。

| 算子 | GPU | 架构 agent | GPU 小时 | 活跃小时 | 墙钟小时 |
| --- | --- | --- | --- | --- | --- |
| attn fwd（GQA 16:1） | H100 | 6 | 0.9 | 9.9 | 10.5 |
| attn bwd（MHA） | H100 | 4 | 0.7 | 9.0 | 44.3 |
| attn bwd（GQA 8:1） | H100 | — | 7.3 | 56.0 | 122.0 |
| fp8 gemm | H100 | 3 | 0.5 | 4.3 | 9.0 |
| fp8 grouped gemm | H100 | 6 | 2.0 | 14.1 | 22.7 |
| conv2d fwd | H100 | 19 | 1.7 | 30.3 | 64.1 |
| conv2d wgrad | H100 | 3 | 0.3 | 6.6 | 18.2 |
| conv2d dgrad | H100 | 3 | 0.5 | 11.3 | 13.1 |
| conv3d fwd | H100 | 2 | 2.0 | 8.7 | 27.0 |
| dsa indexer topk | H100 | 12 | 0.5 | 15.2 | 20.2 |
| dsa sparse fwd | H100 | 4 | 0.9 | 8.3 | 8.7 |
| dsa sparse bwd | H100 | 3 | 2.3 | 13.8 | 15.5 |
| dsa score recompute | H100 | 1 | 0.2 | 3.1 | 4.1 |
| dsa indexer bwd | H100 | 6 | 3.8 | 18.4 | 34.4 |
| gdn prefill fwd | H100 | 7 | 3.0 | 22.7 | 26.9 |
| gdn prefill bwd | H100 | 5 | 6.3 | 28.9 | 36.3 |
| gqa paged decode | H100 | 1 | 0.3 | 7.5 | 8.8 |
| mla paged decode | H100 | 1 | 0.3 | 10.7 | 47.4 |
| mla paged prefill | H100 | 1 | 1.0 | 5.3 | 5.9 |
| trimul | H100 | 3 | 4.5 | 31.1 | 42.0 |
| attn bwd（MHA） | B200 | 9 | 10.1 | 39.2 | 58.7 |
| fp4 grouped gemm | B200 | 3 | 4.5 | 19.0 | 36.8 |
| dsa sparse attn paged | B200 | 2 | 3.4 | 12.5 | 16.9 |
| gdn prefill varlen | B200 | — | 6.7 | 30.1 | 41.6 |
| recurrent kda prefill | B200 | 10 | 34.0 | 97.8 | 134.5 |

## 附录 I 补充结果

正文用散文陈述这些结果，并指向这里的图或表。

**表 5：`/goal` agent 在 Hopper 上的天花板，按 kernel（§2，协议见该节）。** 增益是该 kernel 比其钉扎专家参考快多少；未胜过参考的 kernel 在原文中以粗体标出，下表同样加粗。入口点见附录 E，形状见附录 F。

| 前向算子 | 参考 | 增益 | 反向算子 | 参考 | 增益 |
| --- | --- | --- | --- | --- | --- |
| attn fwd gqa | FlashAttention-4 | +22.8% | fa bwd gqa | FlashAttention-4 | +3.4% |
| gdn prefill fwd | FlashInfer | +82.6% | gdn prefill bwd | FlashQLA | **−25.1%** |
| dsa indexer fwd | DeepGEMM | **−6.0%** | dsa sparse bwd | cuDNN-frontend | +22.0% |
| dsa indexer topk | cuDNN-frontend | +125.9% | dsa score recompute | cuDNN-frontend | +22.4% |
| dsa sparse fwd | FlashMLA | **−22.2%** | dsa indexer bwd | cuDNN-frontend | +12.9% |

**案例：DSA 稀疏 attention 前向。** 产出 ForgeKernel 的 DSA 稀疏 attention kernel 的那场战役，在追平 FlashMLA 的同时按自己的目标停下。那时是单个非持久 kernel，每个 SM 一次只持有一个 CTA，于是每个 CTA 的 Q 加载、第一次 KV 填充和输出存储都在没有 MMA 可以重叠的情况下运行（图 7）。升格为参考之后，kernel 按其测得的签名被重建，在该拓扑内调参停滞在 +4.5%，每个杠杆的估价都是零。架构 agent 的消融随后把暴露出来的时间放在 kernel 的边界上——单单去掉输出存储就获得 6.0%——于是它把第一个 \(M_0\) 决策重新打开为一个持久循环，让每个工作项的写出与下一项的加载重叠。这个改动不是一个被调过的参数，它把 kernel 带到相对前代 +7.2%，并在第二个节点上确认（§4.3）。

**图 6：ForgeKernel 在 Ascend 910C 上，CANN-Bench 1.1.2 的全部 8 个 level-4 算子。** 分数（左）是基准自己的指标：0.2 编译 + 0.3 精度 + 0.5 硬件可达性能。加速比（右）把两组点都除以同一个基准基线，竖线为 1×。Public best 是对全部公开提交取的逐算子包络，2026-09-24 从排行榜读出，从不是循环的输入。下表是图上标在 ForgeKernel 点旁的读数。Public best 是同图的另一组点；正文只点名 GRU、LSTM 的公开领先成绩为 0.324× 与 0.410×。右栏几何平均 1.13× 是相对基准基线，与正文「相对最佳公开提交的加速比几何平均 +33.1%」不是同一个分母。平均分数领先为 3.87 分。

| 算子 | CANN-Bench 分数（0–100） | 相对基准基线的加速比 |
| --- | --- | --- |
| GQA | 78.9 | 1.24× |
| GMM-SwiGLU-Quant | 85.7 | 2.53× |
| GRU | 64.8 | 0.53× |
| LSTM | 64.8 | 0.54× |
| MHA | 77.5 | 1.18× |
| MLA | 75.8 | 1.13× |
| MLA-Prolog | 72.3 | 1.49× |
| SparseFA | 73.6 | 1.47× |
| 均值 / 几何平均 | 74.2 | 1.13× |

**图 7：自我演化在 DSA 稀疏 attention 前向上改变了什么（示意图，不按比例）。** Gen-0 是 ForgeKernel 自己的 kernel，非持久，一次每个 SM 一个 CTA，grid 为 8192。时间线是：load，KV 行程（MMA），store；CTA 边界处没有在飞的 MMA，随后又是 load、KV 行程、store。Gen-1 把该 \(M_0\) 决策（图中标为 D0.1）重新打开为持久循环，grid 等于 SM 数：第 \(n\) 项的写出与第 \(n+1\) 项的 Q 加载和第一次 KV 填充重叠。箭头标出在一个 SM 上省下的时间。

**图 8：79 场战役上的知识来源（每场一列，按启动顺序）以及 9 个硬件知识类别，对应 §4.3 的汇总。** 每个格子里，左下三角计数该战役发明的条目，右上三角计数被复用的条目——由其里程碑开出、由其编码 agent 交付，但由另一个算子的战役发明。紫色弧表示知识在算子之间流动；红色弧表示一场较晚的战役把它还了回去。色块和弧线以原 PDF 第 18 页为准，此处不逐格转录。
