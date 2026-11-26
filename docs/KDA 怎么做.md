状态：早期设计备忘，针对当时的多 Track 计划草稿（草稿本身已不在工作区，这里引用的 §4.1 / §4.3 / §5.2 在今天没有对应文件）。文中出现的 `opspec-to-msl`、`triton-spec-to-msl`、`host_device_bundle` 都不是当前 Track；现行注册表见 [backends/metal/tracks.json](../backends/metal/tracks.json)（`msl-single`、`msl-bundle`，以及 `tilelang-pipeline` 的 `/1`、`/2`、`/3`，其中只有 `/1` 是 `interface_only`）。第 1 节提出的派生标量、按阶段绑定和按 shape 切换特化今天都已实现，见该节里的现状注。

对照 **KDA（NVIDIA Kernel Design Agents）** 和 **CAKE2（华为 CANNBot-CAKE）** 的真实实现，它们之所以能做到“完全不限制候选内核能力”，核心秘诀在于：**系统只锁“算子对外调用边界（Call Boundary）”，绝不伸手去管“候选内部怎么拆核函数、怎么定义 Tiling 结构体、怎么根据 Shape 分支调度”。**

顺着你“学习工业化先进经验、不限制内核候选能力”的原则，我把 KDA / CAKE2 在这 4 个关键环节的做法，以及你当前 Plan 里**还残留的 4 处隐形束缚**对比如下：

---

### 1. 核函数参数与 Host Tiling：CAKE2 / KDA 允许候选自定义 Tiling 参数，而你目前的 §4.1/§4.3 把标量参数锁死在了 `opspec` 原始字段上

* **KDA / CAKE2 怎么做**：
* **CAKE2（昇腾）**：算子合同只锁定外部输入输出张量和原始属性，但候选自己定义 **`TilingData` 结构体** 和 **`TilingFunc`（Host C++）**。候选可以在 Host CPU 上预先算好任意派生参数（例如 `1.0f / sqrt(d)`、快速整数除法魔数 `magic/shift`、Split-K 的每核起止偏移、尾块大小 `tail_m`、动态选择 `tiling_key` 分支），打包塞进 `TilingData` 传给核函数，避免每个 GPU/NPU 线程在核函数内重复做昂贵的标量除法和分支判断。
* **KDA（CUDA C++ / PTX）**：对外只锁一个 `launch(...)` 入口，候选在内部可以写多个不同签名的 `__global__` 辅助核（比如 Kernel 1 负责 Transpose/Pad 到 Workspace，Kernel 2 负责主矩阵乘，Kernel 3 负责 Split-K 归约），每个子核传什么参数结构体完全由候选自己决定。


* **你当前 Plan 还残留的限制**：
* 看你的 **§4.1（契约常量收敛）** 和 **§4.3（标量参数宽度/类型必须与契约一致）**：你规定内核的标量槽位只能是 `opspec.json` 里的原始维度符号（如 `M, N, K`）和原始 `attrs`（如 `EPS`），且多阶段的每个内核都必须严格匹配这套固定签名。
* 这意味着在 Metal 上：
1. **候选无法向内核传入任何 Host 预计算的派生标量**（如 `num_tiles`、`inv_N`、`stride_batch`）；
2. **多阶段流水里的辅助核（如初始化核、归约核）被迫声明一大串它根本不用的契约张量和标量槽位**；
3. **候选无法根据运行时的 `(M, N, K)` 大小动态切换不同的 Kernel 特化（`tiling_key` / 模板分支）**（比如小 `M` 走 GEMV 核，大 `M` 走 Tile GEMM 核）。

> 现状注：这三条今天都已经放开，只剩浮点派生标量还要绕一下：stage 的 `derived_scalars` 能按名 `setBytes` 传整数派生标量（MSL 的 `int` / `uint` / `long` / `ulong`，按各自宽度 4 或 8 字节，且必须是只读参数），浮点派生量不在其中，只能写进任务给的 workspace 或候选自己的 scratch，或者让核里自己算；每条 stage 只绑自己读写的张量；`when` 按实例的符号取值决定该 stage 是否激活；`taic-launch` 的 `grid` / `threads` 本来就吃符号算术表达式（如 `grid=C*OC/64`）。出处见 [v2协议参考.md](架构/v2协议参考.md) 的几何闭集一节与 `tcenv/backends/metal/stage_binding.py`。




* **怎么按 KDA / CAKE2 彻底放开（当时写给草稿的提案；第 1、2 条的主体已由上文现状注所述的机制实现，`role: output` 张量跨 stage 先写后读仍需任务把它声明成 `read_write`；第 3 条没有采纳：现行注册表里没有 `host_device_bundle`）**：
1. **`taic-launch` 支持符号表达式、条件分支与候选自定义派生标量（Derived Scalars）**：允许候选在启动块里用 `symbols/attrs` 算术表达式计算 `grid/threads/threadgroup/workspace`（如 `grid=(ceil_div(N, 256), M, 1)`），允许声明派生标量传入核函数，并允许按维度条件（如 `when M <= 16`）路由到候选文件内的不同特化核函数；
2. **多段流水支持按阶段绑定参数（Per-Stage Argument Binding）**：允许某一段辅助核只绑定它实际读写的契约张量/workspace/标量（省略时才默认按全量槽位绑定），且 `role: output` 张量在多阶段间天然允许先写后读（可读写中间累加）；
3. **在 Mac 上也允许 `host_device_bundle`（或 C++ Launch 入口）作为高阶 Track**：不必把 `host_device_bundle` 局限在 NPU。当未来需要极致压榨 Metal（或接 C++/汇编混合调度）时，候选同样可以交 `Host Tiling + Metal Kernel` 束，Mac 与 NPU 架构完全同构。



---

### 2. 题面规格（Prompt 输入）：KDA / CAKE2 直接拿 `reference.py` 当精确数学规格，不维护两套代码描述

* **KDA / CAKE2 怎么做**：
* CAKE2 给 Agent 的入口是 `op_desc_template.json`（固定版本的字段清单）加 `<op>_reference.py` / `<op>_custom.py`；KDA 的合同是文字条目（Objective / Inputs / Correctness / Constraints / Validation / Evaluation / Promotion）加 eval 命令。两者都不使用本仓库的 `opspec.json`；本仓库用 `opspec.json` + `reference.py` 是另一套设计。
* 原因很简单：到了复杂算子（FlashAttention、Mamba2 状态扫描、MoE 路由、RoPE），自然语言或单行 `formula` 根本说不清边界条件（比如掩码对角线偏移、哪个中间变量升 FP32 累加、广播轴顺序），而作者必填的 `reference.py` 本身就是最精确、无歧义的数学定义。


* **你的 Plan 怎么改（已完成）**：
* v2 任务书已经把 OpSpec 题面与 `reference.py` 一起渲染进去：`taic/kernel_agent.py` 写的是「规格是上面的 OpSpec 题面与派生 ABI；没有 Triton 源。下面的只读参考实现是判分 oracle」（仅 `sem_*` 语义恢复题用 `display.json.spec_text` 覆盖隐藏）。
* `triton-spec-to-msl` 这条 Track 不再需要：v1 的 Triton 题面（`track.json` / `kernel_triton.py`）已退役，v2 题目录里出现这些标记文件会被拒绝，见 [v2协议参考.md](架构/v2协议参考.md)。



---

### 3. 原地更新（In-Place `read_write`）的评测隔离：KDA / CAKE2 严格区分“单步验真”与“多步测速”

* **KDA / CAKE2 怎么做**：
* 工业界大量高性能算子是原地更新的（KV Cache 增量写入、RoPE 原地旋转、残差原地累加 `x += f(x)`、优化器步进）。
* 评测这类算子时，它们把流程拆成互不干扰的两步：
1. **Correctness Pass（单步验真）**：克隆干净输入 → 跑 **1 次** 候选核 → 检查 Guard Band 越界 + 检查输出/原地张量是否含非法 `NaN/Inf` + 与 `reference.py` 做误差比对；
2. **Benchmark Pass（连续多轮测速）**：预热后在单个 CommandBuffer 里连续发射 $N$ 次（如 100 次）测量纯硬件延迟 → **只检查 Guard Band 是否越界，绝不对被原地迭代了 100 次的张量做数值或溢出检查**（因为 `x *= 1.5` 连续原地跑 100 次在 FP16 下必然溢出成 `Inf`——乘法是指数增长；`x += y` 只是线性增长，溢出与否取决于 `y` 和初值——但这些都不是内核 Bug）。




* **你的 Plan 怎么改**：
* 在 **§4.3 最后一段（`read_write` 的重放与差分隔离）** 补上一句：**“正确性差分与结果有限值/NaN 检查仅在单次执行的独立输入副本上进行；连续多轮计时（benchmark reps）后不对被原地迭代的 `read_write` 张量重做数值差分，不因数值累积/溢出误判候选失败；guard band、scratch 的残留探针/全轮探针和只读输入快照复查照旧执行。”**



---

### 4. 编译选项与硬件指令解锁：不要把编译 Flag 锁死在单一默认档

* **KDA / CAKE2 怎么做**：
* 在平台 C / 汇编生成中，很多极限优化（例如 Metal 的 `simdgroup_matrix` 硬件矩阵乘、fast-math 融合乘加、或者特定编译宏开关）受编译器选项影响。

> 本节在原文里没有写完，缺「你当前 Plan 还残留的限制」与「怎么改」两段。现行实现里候选能选的编译选项只有 `taic-launch` 的数学模式 `math fast` / `math safe` / `math relaxed`，没有别的 flag，也不能自选 MSL 版本或 `<metal_*>` 以外的 include，见 [v2协议参考.md](架构/v2协议参考.md) 与 [评测.md](架构/评测.md)；`simdgroup_matrix` 这类硬件矩阵乘不是编译开关，候选直接在 MSL 里用内建。


