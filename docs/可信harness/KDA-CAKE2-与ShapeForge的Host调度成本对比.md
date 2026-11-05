# Host 调度、输入准备和 Kernel 启动成本：KDA、昇腾 CAKE2 与 ShapeForge 的对比

> **这份说明写给不一定做内核开发的人。**先不讲 Metal、CUDA 或 C99 的细节，只回答一个问题：
>
> **一个 GPU/NPU 算子在真正开始计算之前，谁来做准备工作？这些准备工作花多少时间，应该算在谁头上？**

> **实施状态更新（native64）**：本文前部的 V3 Wasm 数字保留为成本归因历史。当前 ShapeForge
> 已删除 Wasm 执行路径，改为 bounded worker 内的 native64 C99 Host Program；CPU/协议/NPU
> 模拟与真实 Metal 复验均已通过。最新合同见 `docs/架构/native-host-program-ABI.md`。
> 最新小 fixture 的 native Host 为 62.67 µs、Host wall 为 65.94 µs、candidate median 为
> 299.58 µs（cv 0.021），此前 Wasm 的 Host 约 200 µs、E2E 约 443 µs。这是当前 fixture 的
> 实测改善，不是 P2/V6 发布结论；输入复制、reader 和校验成本仍需继续优化。

---

## 一、先把问题说成人话

可以把一次算子执行想象成“让厨房做菜”：

1. **Host 侧调度**：经理先决定谁做哪几道菜、每道菜分给哪个厨师；
2. **输入解析**：经理看订单，知道每桌来了多少人、每个人点多少；
3. **Workspace 准备**：经理准备案板、盘子、中间半成品和交接便签；
4. **Kernel 启动**：真正开火，让厨师开始炒菜。

厨房最慢的有时不是炒菜，而是**经理反复写订单、搬盘子、传话**。

在我们的场景里，订单不是固定的一张菜单，而是变长的：

- 有的序列只有 1 个 token；
- 有的序列有 8 个 token；
- 有的序列有 12 个 token；
- 还有一个空序列。

这些长度放在一个叫 `cu_seqlens` 的数组里，例如：

```text
[0, 4, 12, 14]
```

它表示三段序列：

```text
第 1 段：token 0–3，长度 4
第 2 段：token 4–11，长度 8
第 3 段：token 12–13，长度 2
```

如果经理每次都要把订单从一种格式翻译成另一种格式，再准备一堆中间表格，准备工作就可能比真正计算还贵。

**KDA、昇腾 CAKE2 和 ShapeForge 都要解决这个问题，但它们把经理放在不同位置，采用的工作方式也不同。**

---

## 二、KDA 是怎么解决的？

### 2.1 先把一次准备工作做完，后面反复使用

KDA 的典型流程是：

```text
订单（cu_seqlens）
        ↓
prepare：安排工作、准备工作表、准备交接信息、建立启动计划
        ↓
得到一份可以直接执行的 launch
        ↓
之后反复 replay（重放）这份启动计划
```

KDA 的 `prepare()` 会做这些事情：

- 根据每段长度，把 `(sequence, head, chunk)` 切成工作项；
- 决定每个 CTA（可以粗略理解为一个厨师小组）负责哪些工作项；
- 使用 LPT、range-cut 等策略，让各组工作尽量均匀；
- 准备长序列跨组时的中间状态和 handoff 信息；
- 建立 CUDA Graph 或等价的可重放启动计划。

**关键点是：准备工作只做一次，后面主要重复执行已经准备好的启动计划。**

### 2.2 它解决的是“经理反复准备”的成本

如果同一张订单要处理很多次，KDA 不需要每次都重新：

- 读订单；
- 算切分；
- 填工作表；
- 重建启动计划。

所以它的成本结构更像：

```text
第一次：花较多时间准备
后续很多次：只执行准备后的计划
```

这对 benchmark 很有利，因为后续测量的是“重放启动计划”的时间。

### 2.3 但这里有一个重要的计时差异

KDA 的 `bench.py` 是先调用 `prepare()`，再对返回的 launch 计时。

也就是说，**KDA 报告里的速度，主要不是“从收到订单到全部完成”的总时间，而是“准备完成后的执行时间”。**

这并不意味着 KDA 的做法有问题，它回答的是另一个问题：

> **一份已经准备好的 KDA kernel，执行起来有多快？**

ShapeForge 要回答的问题更严格：

> **从拿到这次真实输入开始，包括调度、准备、执行，整个端到端过程有多快？**

所以我们可以学习 KDA 的切分和状态管理，但**不能简单把 `prepare()` 挪到计时外，来假装我们的整个系统也一样快。**

---

## 三、昇腾 CAKE2 是怎么解决的？

### 3.1 Host Tiling 本身就是原生算子工程的一部分

CAKE2 不把 host 侧工作当成一个需要每次解释执行的小脚本，而是把它做成算子工程里的原生代码。

它的结构大致是：

```text
算子合同（op_desc）
        ↓
Host Tiling（原生 C/C++）
        ↓
Device Kernel（AscendC）
        ↓
编译、打包、绑定、安装、评测
```

Host Tiling 负责：

- 根据 shape 和属性求 tiling 参数；
- 选择合适的 TilingKey/特化 kernel；
- 决定数据如何分块、如何搬运；
- 做核间的工作分配；
- 写 tiling metadata 或 workspace。

Device Kernel 负责：

- 真正的计算；
- 片上存储和流水；
- 核内同步和数据传输。

**它和我们最大的区别是：Host Tiling 是编译好的原生代码，不是每次通过 Python 或虚拟机去解释执行。**

### 3.2 它解决的是“host 代码本身要执行得快”

CAKE2 的 host 侧仍然可能有成本：

- tiling 计算；
- shape 特化选择；
- metadata 写入；
- CANN 调用和启动。

但这些工作是在原生算子工程中完成的，不需要经过：

```text
Python → 虚拟机 → 再回 Python → 再回设备
```

它更接近：

```text
输入元数据 → 原生 host tiling → workspace/metadata → device launch
```

所以 CAKE2 的经验是：

> **Host Tiling 应该是可以编译、部署、检查的原生交付物，而不是运行时临时解释的小程序。**

### 3.3 CAKE2 的 shape 特化和 KDA 的 varlen 调度不是一回事

CAKE2 的 Host-level 策略主要包括：

- 按 shape regime 选择不同 TilingKey；
- 为不同 shape 选择特化 kernel；
- 多核负载均衡；
- 处理 straggler core；
- L1/L2 memory tiering。

它主要解决：

> **不同形状的任务，应该选哪种实现和哪套切分参数？**

KDA 的 varlen 主要解决：

> **这次输入中每条序列长度不同，应该怎样把工作项切开，并处理跨段状态？**

两者都有 host 调度，但调度对象不同。**不能把 CAKE2 的 TilingKey 直接当成 KDA 的 varlen work list。**

### 3.4 CANNBot FA 的补充：复杂 varlen 也放在原生工程里

CANNBot 的 FA/catlass 技能里，还能看到更具体的做法：

- 根据 seqlen 做前缀和；
- KV split；
- TND varlen 处理；
- 非整块尾块处理；
- CrossCore flag；
- AIC/AIV 跨核协作；
- `needL1Carry` 等状态续传。

这些也都在原生 host+device 工程中完成，不是通过一个通用解释器每轮执行。

---

## 四、ShapeForge 原来是怎么做的？

### 4.1 我们也想保留候选的自由度

ShapeForge 的目标不是只优化一个固定 KDA kernel，而是希望：

- 候选可以决定自己的调度算法；
- 可以选择 LPT、range-cut 或其他 planner；
- 可以写多阶段 kernel；
- 可以处理 handoff、workspace 和状态；
- harness 负责 ABI、安全、正确性、计时和证据。

V3 选择让候选写受限 C99，再编译成 Wasm，在 Python 中通过 Wasm runtime 执行。

它的路径大致是：

```text
候选 C99
   ↓
Clang + wasm-ld
   ↓
Wasm module
   ↓
Python / Wasmtime
   ↓
读取输入、执行 planner、读出 span
   ↓
再把 span 写入 Metal workspace
   ↓
执行 Metal kernel
```

### 4.2 这条路的问题：准备工作被多层搬来搬去

我们实际测到，Host Program 的成本大约 **300 µs**，而真正的 C99 planner 计算本身只花很小一部分。

在一个更细的分解里，约 169 µs 的 HostRuntime 执行时间包括：

| 工作 | 大约耗时 |
|---|---:|
| 新建 Store | 10 µs |
| 建 Linker、注册 imports | 45 µs |
| 实例化 module | 9 µs |
| 读取 exports 和 ABI 基址 | 16 µs |
| 写入 mapped inputs | 50 µs |
| C99 planner 调用 | 26 µs |

可以看到，**最大的成本不是 planner 算法，而是 Python 与 Wasm 之间的边界管理和数据搬运。**

用厨房来比喻：

> 厨师其实很快就能切好菜，但经理每次都要重新拿订单、翻译订单、找盘子、把盘子从一个房间搬到另一个房间，还要重复登记。

### 4.3 我们的约束比 KDA 更严格

ShapeForge 还要求：

- 读取当前输入的 host work 必须计时；
- fresh/unseen 输入不能偷用 tune 的调度缓存；
- 不能把准备时间藏在 prepare 里；
- 每次实际分段都要能重放和审计；
- 候选失败不能留下部分 workspace 写入；
- NPU 模拟结果不能冒充真实设备性能。

所以我们不能简单地说：

> “像 KDA 一样，先把 prepare 做好，后面都不算它。”

那会破坏我们的端到端证据口径。

---

## 五、三者到底怎么解决同一个成本问题？

### 5.1 一句话对比

| 对象 | 主要做法 | 它省掉的是什么成本 |
|---|---|---|
| **KDA** | `prepare()` 一次性做 host scheduling 和 graph，再 replay | 重复调度、重复准备工作 |
| **昇腾 CAKE2** | Host Tiling 编译成原生 C/C++，和 device kernel 一起交付 | Python/虚拟机边界、解释执行和重复翻译 |
| **ShapeForge 原 V3** | C99 编译成 Wasm，通过 Python/Wasmtime 执行 | 原本想换到安全沙箱，但引入了 FFI、Store、复制和实例化成本 |

### 5.2 KDA 的成本在哪里？

KDA 并不是没有 host 成本，它有：

- 读取 `cu_seqlens`；
- 计算 work list；
- LPT/range-cut；
- handoff 准备；
- graph 准备。

只是它把这些工作主要放在 `prepare()`，而不是每次 replay 都重复。

**它的优化重点是：一次准备，反复利用。**

### 5.3 CAKE2 的成本在哪里？

CAKE2 的 host tiling 也有成本：

- tiling 参数计算；
- shape 选择；
- metadata 写入；
- 原生算子启动。

但它没有我们这种：

- Python 创建 Store；
- 创建 Linker；
- Wasm 实例化；
- Python bytes 与 Wasm memory 多次复制。

**它的优化重点是：host 侧本身原生执行。**

### 5.4 我们的成本在哪里？

我们的成本主要是：

- Python ↔ Wasm 的 FFI；
- Store/Linker/实例化；
- 输入输出的多次复制；
- 每次重新读取 ABI 和视图信息。

真正 planner 的 C99 计算反而不是主要成本。

**我们的优化重点应该是：去掉虚拟机边界，改成原生 Host Program。**

---

## 六、为什么不能只说“学 KDA 就好”？

因为三者面对的问题不完全相同。

### 6.1 KDA 教我们什么？

可以学：

- 按实际 `cu_seqlens` 生成工作项；
- 用成本模型做 LPT/range-cut；
- 长序列切分与 handoff；
- partial tail；
- 每段独立 `final_state`；
- fixed/mixed/uniform 分档计时。

不能简单照搬：

- `prepare()` 不计时的测量口径。

### 6.2 CAKE2 教我们什么？

可以学：

- host tiling 和 device kernel 一起交付；
- host tiling 编译成原生代码；
- shape/TilingKey 特化；
- 用 artifact 和验证记录管理交付；
- 用真实测量反馈优化实现。

不能简单照搬：

- CANN/Ascend 的工具链和专用 API；
- 把 CAKE2 的 shape 特化直接说成 KDA 的 varlen 机制。

### 6.3 ShapeForge 应该保留什么？

必须保留：

- 候选拥有算法和调度自由；
- harness 拥有合同、ABI、安全和证据；
- 输入驱动 host work 计时；
- fresh/unseen 的正确性覆盖；
- 失败不部分提交；
- NPU 模拟不发布为真实性能。

需要改变的是：

- **Metal 生产路径不再经过 Python ↔ Wasm；**
- 改成**受限 C99 → 本机代码 + 有界 worker + 薄 C ABI**；
- 让 Host Program 的输入输出尽量用原生指针和有效 span；
- 让 NPU 模拟复用同一 C99 语义，而不是维护一套独立的 Python/Wasm 执行语义。

---

## 七、最后的差异比较：我们应该选哪条路？

| 比较项 | KDA | 昇腾 CAKE2 | ShapeForge 应该采用的方向 |
|---|---|---|---|
| Host 代码形态 | Python setup + 原生 CUDA/TIRx/graph | 原生 C/C++ Host Tiling | 受限 C99 → 本机代码 |
| Host 执行方式 | prepare 后 replay | 编译后原生执行 | worker 内原生执行 |
| 主要成本来源 | 调度和图准备 | tiling 和原生启动 | 目前是 Python↔Wasm 边界，需去掉 |
| 是否可复用准备结果 | 可以，prepare 一次后复用 | 可以，构建后固化 | 可以复用编译结果和 ABI 元数据，但不能复用 fresh 输入的结果 |
| 是否计入 Host 成本 | prepare 通常不计时 | 取决于算子评测口径 | **必须计入** |
| 是否支持任意合法分段 | 是，KDA 的核心 | 核心主要是 shape case | 是，L1/L2 目标 |
| 是否保留候选自由 | 具体 KDA 实现内自由 | agent/DSL/候选生成自由 | 候选自己写 planner，harness 不内置 KDA 算法 |
| 证据重点 | kernel 正确性和速度 | 交付、编译、验证、性能和知识复用 | 端到端正确性、输入身份、实际 launch、计时和发布门 |
| 真实设备边界 | CUDA 真机 | Ascend 真机 | Metal 真机；NPU 只模拟，不能冒充实测 |

**最终结论：**

> **KDA 教我们如何切工作和传状态，CAKE2 教我们如何把 Host Tiling 做成原生交付物。**
>
> **ShapeForge 不应该继续在 Python↔Wasm 的边界上花主要成本，而应该把 Host Program 编译成本机代码，在有界 worker 里执行，同时保留我们自己的端到端计时、正确性和证据纪律。**

这不是“虚拟机永远不好”，而是：

- 如果目标是安全参考、教学或跨平台解释执行，虚拟机有价值；
- 如果目标是 **Metal 生产路径的真实 Host 性能**，当前这层虚拟机边界已经成了主要成本中心，应该移除。

---

## 八、为什么不整体照抄 KDA / CAKE2 的 ABI？

读到上面，很多人自然会问：既然 KDA 和 CAKE2 已经有成熟做法，**为什么不把它们的 ABI 原样拿过来，省得自己再设计？**

简短的回答是：**ABI 不只是“几个参数的名字和排列”，它背后还绑定了硬件、生命周期、算法假设和证据规则。整体照抄会把别人的假设一起搬过来，反而不适合 ShapeForge。**

### 8.1 三家的 ABI 本来就不是为同一件事设计的

| 对比项 | KDA | 昇腾 CAKE2 | ShapeForge |
| --- | --- | --- | --- |
| 主要硬件 | CUDA / NVIDIA GPU | Ascend NPU / CANN | Apple Metal；NPU 目前仅模拟 |
| Host 的生命周期 | `prepare()` 一次准备，后续 replay | 原生算子工程编译、安装、绑定 | 每次真实输入都要审计，Host work 必须计时 |
| 主要调度对象 | `(sequence, head, chunk)` 工作项 | shape case、TilingKey、核间分块 | 候选自己的 work list、stage、workspace |
| 状态与证据 | output/final_state 与 benchmark | 算子交付、验证、性能、CAKE Book | 输入身份、实际 dispatch、HostRunEvidence、发布门 |
| 可接受的抽象 | CUDA CTA、SM、CUDA Graph | CANN TilingData、AIC/AIV、TilingKey | OpSpec v2、Metal buffer、NPU 模拟语义 |

因此，**KDA 的 ABI 是为“某个 KDA kernel 在 CUDA 上高效运行”服务的；CAKE2 的 ABI 是为“Ascend 原生算子工程”服务的；ShapeForge 的 ABI 是为“候选可自由生成、可信评测和证据绑定”服务的。**它们并不是可以直接互换的标准件。

### 8.2 KDA 的 ABI 不能直接拿来的原因

KDA 里有几类东西看起来很诱人，但不能直接当作 ShapeForge 的 ABI：

- **CUDA/C++ 的 raw pointer、tensor map、device descriptor**：它们依赖 CUDA 的地址、内存和执行模型，Metal 不能直接使用。
- **CTA item 的 bit packing**：`word0/word1` 里塞 token offset、valid rows、first/last、handoff 指针等，是 KDA 特定 kernel 的压缩格式，不是所有候选都必须采用的格式。
- **handoff slot 和 flag 布局**：它依赖 CUDA 的 CTA 调度和 acquire/release 语义。Metal 默认更偏向 serial stage + buffer barrier，不能假装有同样的跨 threadgroup 前向进度保证。
- **固定 KDA 形状的假设**：例如 D=128、chunk=64、V-first fp32 state。我们的候选可以选别的算法、别的 chunk、别的状态布局。
- **`prepare()` 的生命周期**：KDA 可以把准备放在计时外，ShapeForge 要求当前输入驱动的 host work 进入端到端计时。

如果直接抄，结果不是“得到 KDA 的性能”，而是**把我们的候选自由、Metal 约束和计时口径都绑到 KDA 的特定实现上**。

### 8.3 CAKE2 的 ABI 也不能直接拿来的原因

CAKE2 的 ABI 也不是一个可以脱离 CANN 独立使用的通用接口：

- **CANN host tiling 结构体和 TilingData**：它们是 Ascend 原生工程的一部分，和 Metal buffer、Metal stage binding 不是一回事。
- **TilingKey**：它服务于 shape regime/特化 kernel 选择，不等于 KDA 的 `cu_seqlens` work list，也不等于 ShapeForge 的 evaluation case 或 varlen 实例。
- **AscendC kernel 参数和 workspace 描述**：它们假定 AIC/AIV、片上 UB/L1、CANN 的调用和资源模型，不能直接映射到 Metal threadgroup 或我们的 NPU 模拟器。
- **CANN package/binding 的交付 ABI**：它解决的是 Ascend 的构建、安装和 Python binding，不解决 ShapeForge 的 OpSpec、fresh evidence、exact dispatch 和发布门。

照抄这些内容，既不能在 Metal 上运行，也会把我们的 OpSpec、NPU 模拟语义和证据身份弄混。

### 8.4 我们确实应该改自己的 ABI，但理由是自己的问题

“不要照抄”不等于“我们的 ABI 永远不改”。以下改动有明确的本地理由：

1. **本机 64 位布局**：原来的 Wasm32 指针和偏移不能直接用于本机 `.dylib`，必须改成真实 64 位指针和 C 自然对齐。
2. **减少边界复制**：输入和有效 span 应尽量通过原生指针传递，避免 Python ↔ Wasm 的多次复制。
3. **继续由 OpSpec 生成强类型合同头**：未声明的输入、workspace、scalar 在编译期就拒绝，不靠运行时猜。
4. **保留原子提交**：返回 0 才提交，非 0/trap/越界全部丢弃，不能因为改成本机执行就放松。
5. **worker 隔离**：原生代码只在有界 worker 内编译和加载，控制机不接触原生指针。
6. **保留证据身份**：`HostRunEvidence`、`evaluation_input_digest`、实际 dispatch 和端到端计时仍然是 ShapeForge 的责任，不能由外部 ABI 代替。

这些修改是因为**我们要去掉 Python↔Wasm 的成本，同时守住自己的合同**，不是因为 KDA 或 CAKE2 的 ABI 长什么样。

### 8.5 合理的关系是“学机制，不抄 ABI”

| 可以学习的机制 | ShapeForge 自己的对应方式 |
| --- | --- |
| `cu_seqlens` 是一等运行时输入 | 保留我们的 i32 `cu_seqlens` input 和 L1/L2 语义 |
| Host 根据输入生成 work list | 候选自己的 Host Program 生成 `work_items/item_counts` |
| Host+Device 协同 | 保留 `host/tiling.c` + device source + execution geometry |
| 中间状态和 handoff | 保留 `handoff_states`、flag workspace 和逐段 `final_state` |
| 按 shape 做特化 | 用现有 shape portfolio / exact dispatch |
| 原生 Host Tiling | 受限 C99 → 本机代码 + 有界 worker，不再 Python↔Wasm |
| 有效行和尾块 | 用我们的 valid span、poison/sentinel、partial-tail 检查 |
| 输入身份与重放 | 保留 `evaluation_input_digest` 和 HostRunEvidence |

**一句话：不是抄 KDA/CAKE2 的 ABI，而是让 ShapeForge 的 ABI 具备同等重要的能力。**

---

## 九、给领导的一句话总结

**KDA 是“先把工作安排一次，后面反复用”；CAKE2 是“把安排工作的人直接编译成正式员工”；我们之前是“每次请一个翻译员反复解释工作安排”，所以慢在翻译和传话，而不是慢在真正做菜。下一步应该把翻译环节去掉，让安排工作的人直接在本机执行，同时保留我们对真实输入、总耗时和正确性的严格要求。**
