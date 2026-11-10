# ABI 到底固定了什么：KDA、昇腾 CAKE2 与 ShapeForge 的设计深度对比

> **这份文档写给不一定做内核开发的技术实习生。**
>
> 我们要回答三个问题：
>
> 1. KDA、昇腾 CAKE2 和 ShapeForge 的 ABI 分别为什么设计成这样？
> 2. 固定 ABI 会不会限制候选的算法自由？
> 3. 如果干脆不要 ABI，让候选自己带输入、输出和启动代码，会发生什么问题？
>
> 先给结论：
>
> **ABI 一定会形成约束。合理的 ABI 只固定“两个独立组件怎样交接”，不应固定“候选内部怎样计算”。**
> 三家的差异，不只是参数列表不同，而是它们对**硬件、生命周期、信任边界、性能测量和交付责任**的假设不同。

> **当前状态边界：**
>
> - ShapeForge 以当前 **native64 C99 Host Program + Metal** 路径为准，不再使用 Wasm/Wasmtime 虚拟机。
> - 当前两条已验收的生产 Track 都是 Metal；清微 NPU 只有流程、数据、协议和身份模拟，
>   必须保持 `simulated=true`、`publication_allowed=false`。
> - 本文的 CAKE2 面向**昇腾 NPU / CANN**，不是清微 NPU。
> - KDA 的公开 kernel 数字来自 NVIDIA 设备，CAKE2 的交付面向昇腾；本文不比较它们的性能倍数，
>   也不把上游结果当成本机实测。
> - 外部判断限定在文末列出的固定源码审查版本，不是对上游未来版本的永久断言。

---

## 一、先建立一个最重要的直觉

把一次算子执行想成“餐厅让外部厨师做一道菜”。

餐厅至少需要和厨师约定：

- 订单上每项食材是什么；
- 哪些食材不能改；
- 成品装在哪里；
- 半成品能放在哪里；
- 厨房什么时候算开始工作；
- 最后怎样验收这道菜。

这些就是交接边界。

餐厅不需要规定：

- 厨师使用什么菜谱；
- 切几刀；
- 分几个小组；
- 哪一组先做；
- 私人流转标签怎样编码。

**ABI 是订单、餐盘和交接规则，不应该是菜谱。**

但ABI不只是“函数有几个参数”。在真实 GPU/NPU 系统里，它还包括：

- 参数顺序和名称；
- dtype、shape、stride；
- 内存地址和对齐；
- 32/64 位指针宽度；
- buffer 的读写权限；
- workspace 的大小和有效区域；
- 对象什么时候创建、复用和销毁；
- Host 与 Device 谁负责哪一步；
- 出错时哪些写入生效；
- 这次调用对应哪份代码、输入和证据。

所以，**两个实现即使都叫 `run()`，也可能有完全不同的 ABI。**

---

## 二、先分清五层：它们经常被一起叫成“ABI”

为了不做“苹果对橘子”的比较，先把五层分开。

| 层次 | 回答的问题 | ShapeForge 的主要载体 |
| --- | --- | --- |
| **任务语义合同** | 要算什么，合法输入和正确结果是什么 | `opspec.json + reference.py` |
| **外部内存/调用合同** | 张量怎样传、谁能写、输出在哪里 | `TensorMemoryContract` |
| **交付与执行合同** | 候选交哪些文件、有哪些 stage、怎样启动 | CandidateArtifact、Track、ExecutionGeometry |
| **原生 Host ABI** | Python/worker 怎样调用候选 C planner，指针和 span 怎样校验 | 生成的 `host_contract.h`、`sf_native_invoke()` |
| **证据身份** | 这次构建和测量究竟对应什么代码、输入、几何和环境 | candidate/build/evaluation identity、HostRunEvidence |

KDA 和 CAKE2 也有这些层次，但未必都有 ShapeForge 这样的独立、版本化类型。

因此：

- KDA 的 `prepare/run` 主要是**候选生命周期和调用边界**；
- KDA 的 `(word0, word1)` 是**某个候选内部的 Host→Device work-item 格式**；
- CAKE2 的 `TilingData` 是**CANN Host Tiling→Device Kernel 的数据交接**；
- CAKE2 的 `TilingKey` 是**实现变体选择键**；
- ShapeForge 的 `OpSpec` 是**任务语义主合同**，不是 KDA work-item 的替代品；
- ShapeForge 的 `host_contract.h` 是**worker 调候选 Host Program 的 ABI**，不是一份必须直接塞进 Metal kernel 的通用结构体。

**这些名字不能直接画等号。**

---

## 三、三家的 ABI 为什么长成不同样子

一张表先看设计目标：

| 系统 | 首要设计目标 | ABI 的主要服务对象 | 主要代价 |
| --- | --- | --- | --- |
| **KDA kernels** | 在已知 KDA 算子和 NVIDIA 硬件上做到极致性能 | 特定 `kda_forward` 调用、CUDA kernel、graph replay、CTA 调度 | 与任务形状、CUDA 指针/描述符、同步模型和 prepare 生命周期紧绑定 |
| **昇腾 CAKE2** | 让 Agent 生成可编译、安装、绑定、评测的昇腾原生算子工程 | CANN Host Tiling、AscendC Device、TilingKey、package/PyBind | 与 CANN/Ascend 工具链、工程结构和变体交付方式紧绑定 |
| **ShapeForge** | 让人或 LLM 自由提交候选，同时由独立 Harness 统一校验、测量和留证 | OpSpec、通用 Candidate/Track、worker-native Host ABI、Metal runtime、证据链 | 必须显式声明资源、接受 worker/ABI 检查，并承担更严格的身份和计时责任 |

这不是“谁设计得绝对更好”，而是：

> **KDA 优先服务特定高性能实现，CAKE2 优先服务昇腾原生工程，ShapeForge 优先服务候选自由和独立可信验收。**

设计目标不同，ABI 自然不同。

---

## 四、KDA：为特定 KDA 任务和 CUDA 高性能执行而设计

### 4.1 外部调用边界已经很明确

KDA 前向实现的共同任务入口是：

```python
run(
    q, k, v, g, beta,
    A_log, dt_bias, scale,
    initial_state, cu_seqlens,
) -> (output, final_state)
```

公开 README 还固定了代表性 ABI 事实：

- `q/k/v/g` 是 bf16，形状为 `[1, T, H, 128]`；
- `beta` 是 bf16；
- `A_log`、`dt_bias` 是 fp32；
- `initial_state` 和 `final_state` 是 fp32；
- `cu_seqlens` 是 int64 `[N+1]` 或 `None`；
- `D=128`，状态采用特定 V-first recurrent layout。

这是一份**特定算子的任务 ABI**。它让三种完全不同的实现——CuTe、TIRx、static PTX——可以围绕同一个数学任务进行比较。

它没有要求三种实现使用同一个内部 kernel、同一种调度器或同一个 work-item 格式。

### 4.2 `scale` 说明“参数名相同”也不等于 ABI 相同

TIRx wrapper 中，`scale` 会转换成 Python `float`：

```python
data = {
    # ...
    "scale": float(scale),
}
```

而在 ShapeForge 当前 `varlen_kda` OpSpec 中：

- `cu_seqlens` 是 **i32 张量**；
- `scale` 是 **fp32 输入张量 `[1]`**；
- `q/k/v/g` 可以是声明档位中的 fp16 或 bf16；
- 形状使用 `total_tokens/H/D` 等符号，而不是把 `D` 永久固定为 128。

所以即使两边都写着：

```text
q, k, v, scale, cu_seqlens
```

也不能把调用字节或 Python 参数直接互转而不做适配。

**名字相同不是 ABI 相同；类型、容器、形状、精度和生命周期都必须一致。**

### 4.3 `prepare/run` 把“准备”和“重放”分开

TIRx 的典型过程是：

```text
prepare(*inputs)
   ├─ 分配 output / final_state
   ├─ 编译和准备 tensor maps
   ├─ 建立 Host work list
   ├─ warmup + synchronize
   └─ 记录私有 CUDA Graph
          ↓
       launch()
          ↓
       replay，并返回准备好的输出 buffer
```

PTX 路线也提供 prepared launch，并强调 Host scheduling、descriptor table 和 graph 生命周期。

这非常适合回答：

> **一份已经准备好的 kernel，在固定绑定和启动计划下执行有多快？**

它的 ABI 因而不是单纯一次函数调用，还包括：

- 哪些对象在 `prepare` 中创建；
- 哪些输入地址被 launch 闭包或 graph 捕获；
- 哪些描述符可以复用；
- 输出 buffer 是否每次返回同一个；
- 当前输入变化时哪些准备必须重做。

这些都是 ABI 生命周期的一部分。

### 4.4 KDA 的 work-item 位布局是候选内部 ABI

TIRx 的 `_host_item_table` 为每个 CTA 生成 `(word0, word1)`：

```text
word0:
  token offset 的拆分 bits
  + valid rows
  + handoff source/destination
  + first/last

word1:
  sequence
  + head
```

其中 token offset 被压进 21 bits，代码还显式检查：

```python
assert T < (1 << 21)
assert T * H * D < (1 << 31)
```

这是非常优秀、也非常具体的工程决策：

- 用少量字宽压缩调度信息；
- 减少 Host→Device 表的数据量；
- 让 Device kernel 用固定位运算解码；
- 与固定 chunk、CTA 调度和 handoff 协议共同优化。

但它不是通用 ABI，因为：

- bit 位置由这组 Host 和 Device 源码共同约定；
- 21-bit token 上限来自该布局；
- first/last/handoff 的含义依赖该算法；
- Metal 或其他候选可能选择完全不同的 work item；
- 换成别的 chunk、状态布局或调度器后，这套编码未必仍然合理。

**它是 KDA 候选内部的高性能私有 ABI，不应该上升成 ShapeForge 的全局调度格式。**

### 4.5 KDA 的底层 ABI 深绑 CUDA 执行模型

PTX shims 中可以看到：

- `TensorView`；
- raw pointer；
- TMA descriptor/table；
- kernel argument array；
- grid/block/stream；
- handoff flag 地址；
- CUDA graph 与 stream capture。

Host 和 Device 不只是共享几个 Python 张量，还共享：

- 指针值；
- descriptor 生命周期；
- kernel 参数布局；
- 同步与 stream 语义；
- flag/state 的所有权。

这些设计能让特定 NVIDIA 实现高效工作，但不能直接搬到 Metal。

### 4.6 KDA 为什么这样设计

因为它主要优化的是：

```text
已知任务 ABI
+ 已知 NVIDIA 硬件
+ 已知高性能候选
+ 已准备计划的重复执行
```

它优先消除：

- 重复 Host 调度；
- graph 重建；
- descriptor 重复上传；
- CTA 工作分配不均衡；
- Device 解码成本。

它没有义务为 ShapeForge 的所有算子提供统一 OpSpec、候选沙箱、冻结几何或 fresh confirmation。

**这不是缺失或错误，而是设计目标不同。**

---

## 五、昇腾 CAKE2：为 CANN 原生 Host+Device 工程而设计

### 5.1 `op_desc` 是入口摘要，不是完整通用 ABI

固定版本的 `op_desc_template.json` 包含：

```text
op_name
category
description
shape_info.input_shapes / output_shapes
attributes
test_cases
```

它适合告诉 Agent：

- 算子叫什么；
- 代表性输入输出是什么；
- 有哪些属性；
- 从哪些测试用例开始。

但该模板本身没有完整表达 ShapeForge 需要的：

- `read_write` 终值；
- alias；
- workspace 生命周期；
- 完整语义域；
- tolerance/error-scale 权威；
- work accounting；
- baseline 身份；
- build/evaluation identity。

这不代表 CANN 本身不能表达这些概念，而是说：**这份 CAKE2 示例模板不能直接成为 ShapeForge 的第二份主合同。**

### 5.2 Python 侧通过 Model / ModelNew 接入评测

CAKE2 的 `evaluate.py` 会分别加载：

```text
<op>_reference.py  → Model
<op>_custom.py     → ModelNew
```

然后：

- 根据 test case 创建输入；
- 运行 reference 和 custom；
- 还用 CPU fp64 golden 做三方数值比较；
- 再进行性能测量。

这里的 Python 调用约定服务于 Agent 工程和评测流水线。它使不同 Evo variants 能复用相同的 reference、wrapper 和测试入口。

但 Python `ModelNew(*inputs)` 只是外层 API；真正的 Host+Device 交接在下面的 CANN 工程中。

### 5.3 Host Tiling 是原生 C++ 工程的一部分

CAKE2 的 FlashAttention migration 示例中，Host 侧有：

```cpp
static ge::graphStatus TilingFunc(gert::TilingContext* context)
```

它负责：

- 读取 attrs；
- 设置 block dim；
- 填 `FlashAttentionCustomTilingData`；
- `SaveToBuffer(...)` 写入 raw tiling buffer；
- 设置实际 tiling data size；
- 计算并填写 workspace sizes；
- 返回执行状态。

其结构是：

```text
输入 shape / attrs / platform
           ↓
     Host TilingFunc
           ↓
 TilingData + workspace sizes + block dim (+ TilingKey)
           ↓
       CANN runtime
```

这是一种原生算子 ABI：Host 不再是运行时临时解释的小脚本，而是正式编译、安装和交付的一部分。

### 5.4 Device 通过固定入口和 tiling buffer 接收数据

对应 Device 入口类似：

```cpp
extern "C" __global__ __aicore__ void flash_attention_custom(
    GM_ADDR q,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling);
```

然后：

```cpp
GET_TILING_DATA(tiling_data, tiling);
int S1 = tiling_data.S1;
int S2 = tiling_data.S2;
// ...
```

这里发生了两次关键交接：

1. Python/CANN 把外部输入输出和 workspace 交给 Device；
2. Host Tiling 把自定义派生参数通过 tiling buffer 交给 Device。

`TilingData` 的字段由算子工程定义，CANN 宏和序列化机制保证 Host/Device 对它的理解一致。

这与 KDA 的 `(word0, word1)` 很不一样：

- KDA 示例强调手工位打包和 CUDA 调度；
- CAKE2 强调原生结构体、CANN 序列化和算子工程集成。

### 5.5 TilingKey 连接“当前情况”和“特化实现”

CAKE2 策略材料中的典型模式是：

```text
Host:
  根据 dtype / shape / mode / applicability
  计算 tilingKey
       ↓
Device:
  TILING_KEY_IS(...)
  选择对应的实现类或模板实例
```

例如：

- 普通维度与大 HeadDim 使用不同算法；
- fp16/bf16/fp32 使用不同模板实例；
- 传统 two-pass 与 Welford 使用不同 tiling key；
- 不同实现拥有自己的 TilingData 字段。

这解决的是：

> **同一算子工程内，当前输入应采用哪套 tiling 和特化实现？**

它不是：

- KDA 的 varlen work list；
- ShapeForge 的 Track；
- 通用 VariantRegistry；
- ShapeForge 的 fresh evidence。

它是 CANN Host+Device 协同分发机制。

### 5.6 构建、安装和 PyBind 也是交付 ABI 的一部分

CAKE2 工程还包括：

```text
op_host / op_kernel
   ↓
host library + device object + .run package
   ↓
安装到算子工作目录
   ↓
生成/加载 PyBind extension
   ↓
ModelNew 调用与 evaluate.py 评测
```

因此它的 ABI 不只是内存里的结构体，还包含：

- 算子工程目录约定；
- 编译产物；
- 安装位置；
- runtime library；
- Python binding；
- variant 切换方式。

固定文档中的 Evo variant 切换会把对应 `.run` 安装到共享 base 目录、覆盖当前 runtime，再评测。它首先是一条工程交付与搜索流程，不是 ShapeForge 这种多 Track 身份协议。

### 5.7 CAKE2 为什么这样设计

因为它要解决的是：

> **怎样让 Agent 生成的结果成为 CANN 能真正构建、安装、调用和评测的昇腾原生算子？**

所以它优先：

- 保留 Host Tiling 的原生执行效率；
- 使用 CANN 的 TilingData/TilingKey；
- 对齐 AscendC 的 GM_ADDR、AIC/AIV、workspace 模型；
- 让构建、安装、绑定、测试形成完整工程闭环。

把它直接搬到 Metal，不会自动得到同样的性能或安全性，只会把 CANN 的假设一起搬来。

---

## 六、ShapeForge：为“未知候选 + 独立可信 Harness”而分层设计

ShapeForge 面对的不是一个固定的高性能 KDA 实现，也不是一个已经确定的 CANN 工程。

它可能收到：

- 人写的候选；
- LLM 写的候选；
- 单文件 MSL；
- Host+Device Bundle；
- 不同算法和不同内部资源；
- 正确、错误、不安全或故意作弊的实现。

因此它不能只说“调用候选的 `run()` 就好”，而要把边界拆成几层。

### 6.1 第一层：OpSpec + reference 固定数学语义

`opspec.json` 固定：

- 张量名称和顺序；
- role、dtype、layout、stride、access；
- 符号 shape 和 attrs；
- cases、语义域和容差；
- 必要的 workspace 容量声明；
- work accounting；性能 baseline 由独立的 `baselines.*` 登记。

`reference.py` 给出独立数学答案。

它**不固定**：

- tile；
- grid；
- threads；
- num_warps；
- planner；
- StagePartitioner；
- 候选优化参数。

例如 `varlen_kda` 的 `work_items` 声明了容量和两个 u32 words，但 OpSpec 文档明确说明：

> token range、valid rows、sequence/head、first/last、handoff source/destination 的**位布局属于候选**；容量公式属于 ABI。

这条区分非常重要：

```text
Harness 需要知道：最多多少项、每项多少字节、怎样保证不越界。
Candidate 自己决定：这些字节里的每一位是什么意思。
```

### 6.2 第二层：TensorMemoryContract 固定外部内存事实

`TargetAbiDeriver` 从语义实例和 target limits 派生：

- tensor identity；
- role/dtype/access；
- concrete shape/strides；
- global/workspace；
- accumulator dtype 和容差；
- buffer slot/resource 上限。

它明确不包含：

- tile；
- grid；
- threads；
- launch 常量；
- fallback geometry。

**这就是“固定交接、不固定算法”的代码级边界。**

### 6.3 第三层：CandidateArtifact / Track / ExecutionGeometry 固定交付

候选可以走：

- `msl-single`：单文件 MSL + 声明式几何；
- `msl-bundle`：候选 Host Program + Device Kernel + schedule/resources。

ExecutionGeometry 属于候选，用来描述：

- stage；
- kernel；
- grid/threads；
- `when`；
- workspace；
- derived scalars。

构建后，完整几何会被冻结为 `execution_geometry.json`，加入 artifact set，再重新计算 `provenance.artifact_set_digest`。

因此 ABI 不只说明“数据怎么传”，还和“实际怎样启动”绑定。

### 6.4 第四层：native64 Host ABI 固定 worker 与候选 planner 的握手

当前 Host 入口是：

```c
int32_t host_tiling(
    const SfHostInputs *inputs,
    const SfHostShapes *shapes,
    const SfHostTarget *target,
    SfHostWorkspaces *workspaces,
    SfHostResolved *resolved);
```

相关 C 结构体由 Harness 根据 OpSpec/ABI 生成。

候选在函数内部仍然可以写：

- 自己的辅助函数；
- 成本模型；
- LPT、range-cut、divmod 或其他 planner；
- 循环和状态机；
- 自己的 work-item 位编码；
- handoff 生成逻辑。

固定的是入口和安全交接，不是 planner 算法。

### 6.5 为什么让 C 编译器拥有布局，而不是 Python 猜偏移

当前 native ABI 明确要求：

- 指针宽度为 64 bit；
- C 编译器拥有 `sizeof`、对齐和 `offsetof`；
- Python 不读取裸结构体；
- Python 只调用生成的 scalar/address accessors 和 `sf_native_invoke()`；
- 控制机不 `dlopen` 候选库，也不接触 native pointer。

实际编译和加载只发生在有界 worker 内。

原因是：

```text
Python  dict/JSON 的字段顺序
≠ C struct 的内存布局
≠ Metal buffer 的字节布局
≠ 另一种硬件上的自然对齐
```

如果 Python 自己猜：

- `uint32_t` 后面是否 padding；
- 指针是 4 字节还是 8 字节；
- 一个结构体对齐到 4、8 还是 16 字节；
- 数组末尾有没有 tail padding；

那么 Host 写入的字段可能被 Device 或另一段 C 代码在不同位置读取。

**让编译器拥有布局，是消除一类整层的隐蔽 ABI bug。**

### 6.6 输入 ABI：小输入映射，大输入 typed read

当前规则是：

| 输入 | Host 看到什么 | 原因 |
| --- | --- | --- |
| `cu_seqlens`、小 scalar 等 mapped input | 强类型只读指针和元数据 | planner 通常需要直接读取 |
| `q/k/v/g` 等大输入 | `id/numel/dtype` 元数据 | 避免默认把大输入复制进 Host |
| 大输入确实需要读取 | `sf_input_read_<dtype>(...)` 白名单 callback | 控制 id、dtype、范围并记录实际读取区间 |
| `read_write` 输入 | 不进入候选 Host 的可写视图 | Device 独占，避免 Host 偷偷改变输入/终值语义 |

每次 fresh 调用仍读取本次输入，不能按 build 缓存输入值或复用旧 plan。

### 6.7 Workspace ABI：容量和有效 span 分开

每个 workspace 有：

- 实际地址；
- `capacity`；
- `offset`；
- `count`；
- `elem_nbytes`。

必须满足：

```text
offset + count <= capacity
```

这里有两个不同问题：

1. **最多能放多少**：capacity，用于分配和越界检查；
2. **这次实际写了多少**：有效 span，用于提交和 Device 消费。

如果只固定 capacity，不固定 span，Host 可能只写了 4 项，Device 却按最大容量读取 100 项，于是读到旧数据或 poison。

如果只固定 span 字段名，不固定所有权，候选可能把 workspace 指针改指到只读输入，造成越界或输入篡改。因此调用后还要重新核对 pointer、metadata、guard 和只读输入。

### 6.8 L1 与 L2：静态 shape 和本次 resolved shape 分开

当前 ABI 还区分：

- `SfHostShapes`：编译期/容量形状；
- `SfHostResolved`：本次调用真实的 resolved scalar。

对于 L2 capacity 实例：

- buffer 按 capacity 分配；
- `total_tokens/num_seqs` 由每次真实输入解析；
- Python 每次写入 resolved 值；
- 候选不得改写这些字段；
- HostRun 从 ABI 回读实际值；
- correctness 只比较 logical 区域，padding 按协议检查；
- 同一个 build 可服务不同 resolved batch，无需因为 total/count 改变而重编译。

V5 L2 的 CPU/协议与真实 Metal 机制已经验收。continuous-batching mixture 不发布 speedup。V6 六档与 P2-F 已另有记录；V6 倍数只相对登记的朴素 Torch-MPS baseline。

这说明：

> **ABI 固定的是形状的含义和存储边界，不是把每次输入形状永久写死。**

### 6.9 原子提交：失败不能留下“半份答案”

Host 调用只有返回 0 且全部检查通过，才能提交有效 span 和 resolved scalar。

以下情况必须丢弃本次写入：

- 非 0 返回；
- native trap；
- span 越界；
- guard 破坏；
- read-only 修改；
- 非法 import；
- worker 超时。

否则可能出现：

```text
workspace A 已写入新计划
workspace B 仍是旧计划
resolved scalar 又是第三次的值
Device 却把三者拼在一起执行
```

这种“部分新、部分旧”的状态比直接失败更危险，因为它可能产生貌似合理的错误结果。

### 6.10 证据身份也是 ABI 设计的一部分

ShapeForge 不只关心“能调用”，还关心“调用的是哪份东西”。

核心分账是：

```text
candidate_sha256 == provenance.source_sha256

候选 entries
→ 编译
→ 冻结 ExecutionGeometry 加入 artifact set
→ 重算 artifact_set_digest
→ correctness / benchmark / fresh confirmation
```

Bundle 的 candidate digest 来自仅含候选源 entries 的 canonical manifest，不是传输 envelope 的哈希。

Native Host 另有：

- `program_sha256`；
- `assembled_host_sha256`；
- `native_sha256`；
- compiler/platform；
- `execution_backend=native-c99`。

`native.descriptor/native_digest` 用于交叉审计，不替代 canonical candidate/artifact/evaluation identity。

因此，**改 ABI 不是只改一个头文件**：它可能改变派生 header、真实编译单元、native 产物、执行行为和评测身份，旧证据不能自动继承。

---

## 七、把三方放在同一张 ABI 对照表里

| 比较维度 | KDA kernels | 昇腾 CAKE2 | ShapeForge |
| --- | --- | --- | --- |
| 主要抽象层级 | 特定 KDA 任务 API + CUDA 候选内部 ABI | CANN 原生算子工程 ABI | 通用任务/候选/执行/证据分层 ABI |
| 外部入口 | `run(...)`，部分实现另有 `prepare(...)` | `ModelNew` / custom wrapper / CANN 调用 | OpSpec instance + 正式 Track + CandidateArtifact |
| 任务语义来源 | `kda_forward` 任务 ABI 和 benchmark | op_desc、reference、工程接口、test cases | 版本化 OpSpec v2 + 独立 `reference.py` |
| 代表性 `cu_seqlens` | int64 tensor 或 `None` | 取决于具体算子工程 | 当前 `varlen_kda` 为 i32 tensor，有明确边界/sentinel 规则 |
| 代表性 `scale` | Python scalar，TIRx wrapper 转 `float` | 取决于 op_desc/Host Tiling | 当前任务为 fp32 输入张量 `[1]` |
| 输出所有权 | TIRx `prepare` 示例分配 output/final_state 并由 launch 返回 | CANN/custom wrapper 工程管理 | Harness 按合同分配、初始化、恢复并比较 |
| Host 生命周期 | prepare 一次，prepared launch/replay | 原生 TilingFunc + 编译安装运行 | 有界 worker 内 native64 Host Program；fresh 调用按真实输入执行 |
| Host→Device 调度数据 | Python scheduler + work tables + raw pointer/descriptors | TilingData + raw tiling buffer + TilingKey | 候选 workspace 有效 span + frozen geometry + typed binding |
| work-item 内部布局 | 特定实现固定，如 21-bit token packing | 具体 TilingData/工程自行定义 | 候选自己定义；Harness 管容量、span、安全和身份 |
| shape/variant 选择 | shape planner、retained implementation、fallback | TilingKey、shape regime、模板实例 | 候选 ShapePlanner、one-of-N dispatch、独立 fallback |
| workspace | CUDA buffer、TMA descriptor、handoff state/flags | CANN workspace size 与 GM_ADDR | 声明容量、有效 span、初始化三态、guard/poison |
| 原生指针边界 | CUDA/C++ shims 与具体 kernel 内部使用 | CANN/AscendC 工程内部使用 | 只在 bounded worker 内；控制机不见 native pointer |
| 结构体布局责任 | 具体 Python/C++/CUDA/TVM 组件共同约定 | CANN TilingData 宏和工程类型系统 | C 编译器拥有 C 布局；Python 只用生成 accessor |
| 失败提交语义 | 由具体候选、runtime 和 fallback 协议决定 | 由 CANN 返回状态和工程实现决定 | 非 0/trap/非法 span/guard/read-only 等全部丢弃 |
| 正确性判决 | 具体 benchmark 的 output/final_state 与数值门 | reference/custom/CPU fp64 golden 等评测 | 只认 `tolerances + scaled_error`，含 `read_write` 终值 |
| 计时边界 | 常见 benchmark 先 prepare，再测 launch/replay | 取决于 CAKE2 算子评测口径 | 输入驱动的必要 Host work 计入 E2E；可复用构建按身份复用 |
| 构建与证据身份 | 有打包、快照和 benchmark 记录，不是本地整套协议 | 有工程、package、评测和 Evo 报告，不是本地整套协议 | candidate/geometry/artifact/native/evaluation identity 分账 |
| 真实硬件边界 | NVIDIA CUDA / Blackwell 等具体实现 | 昇腾 NPU / CANN | Metal 生产；清微 NPU 只模拟 |
| 最适合回答 | 准备好的特定 KDA kernel 多快、怎样调度 | 怎样交付可构建安装的昇腾原生算子 | 未知候选怎样被独立、正确、可复现地接收和验收 |

---

## 八、同一个 varlen 例子，三种 ABI 各自关心什么

假设输入有：

```text
cu_seqlens = [0, 4, 12, 14]
```

即三条序列长度为 `4、8、2`。

### KDA 关心

- 把 `(sequence, head, chunk)` 组织成 CTA 工作；
- 长序列怎样 split；
- LPT/range-cut 怎样均衡；
- `(word0, word1)` 怎样压缩；
- CTA 之间怎样 handoff；
- descriptor、graph 和 launch 怎样复用。

### CAKE2 关心

如果由类似原生工程表达，它更关心：

- Host Tiling 怎样读取 shape/attrs；
- TilingData 放哪些派生参数；
- workspace 需要多少；
- block dim/TilingKey 怎么选；
- AIC/AIV 或具体 Device 实现怎样消费这些数据。

这不表示 CAKE2 自带一个通用 KDA varlen planner。

### ShapeForge 关心

- 本次 `cu_seqlens` 是否符合合同；
- 候选实际读取了哪些输入区间；
- planner 是不是候选自己写的；
- work list 是否超过声明容量；
- 有效 span 是多少；
- handoff state/flags 是否按候选协议执行；
- output/final_state 是否只在 logical 区域正确；
- Host 工作是否进入应有计时边界；
- 这次执行绑定的 candidate/geometry/artifact/input identity 是什么。

### 同一个“切分”在不同层的位置

```text
len=4 切成 2/1/1 还是 4
```

这是候选 planner 的算法决定，不应由 ABI 固定。

```text
最多允许 3 个 piece，每个 work item 是 2 个 u32
```

这是当前具体 ShapeForge 任务/候选声明的容量和存储合同，需要 ABI 固定。

```text
第 28 bit 表示 handoff destination
```

如果候选选择这种编码，它是该候选内部 Host+Device 的共同约定，不应成为所有候选的全局规定。

---

## 九、固定 ABI 到底会不会限制候选？

**会。任何 ABI 都会限制。**问题是有没有限制在正确的位置。

### 9.1 合理的限制

| 固定内容 | 为什么必须固定 |
| --- | --- |
| 外部输入输出和 dtype/layout | 否则无法生成输入、分配 buffer、比较结果 |
| read/read_write/write 权限 | 否则无法保护输入和检查最终状态 |
| workspace capacity/span/lifetime | 否则无法防止越界、旧数据和部分提交 |
| Host 调用入口 | 否则 Harness 不知道怎样调用候选 |
| 原生指针/结构体布局 | 否则不同语言和组件会读错字节 |
| stage 与实际 dispatch 绑定 | 否则候选自报和硬件执行可能不同 |
| 计时边界 | 否则不同候选测的不是同一件事 |
| 候选与构建身份 | 否则旧证据可以套到新实现上 |

### 9.2 不合理的限制

如果 ABI 开始固定以下内容，就越过了候选自由边界：

- 所有候选必须使用 KDA 的 `(word0, word1)`；
- 所有候选必须使用 CAKE2 的 TilingData/TilingKey；
- `D` 永远只能是 128；
- chunk 永远只能是 64；
- 必须使用 LPT；
- 必须拆成固定 stage；
- 某种 shape 必须用某个 tile；
- 不允许 persistent、循环、状态机或公开原生矩阵能力；
- 按 `op_name` 给候选塞一套中心算法。

那就不再是 ABI，而是**伪装成 ABI 的算法注册表**。

### 9.3 当前合同不能表达新机制时怎么办

不能假装 ABI 永远不限制，也不能让候选偷偷绕过。

正确做法是：

1. 明确判断是算法变化、资源声明变化，还是外部语义变化；
2. 普通算法/tiling 变化不需要改 ABI；
3. 新 workspace 或几何由候选显式声明并重新构建身份；
4. 真正新的调用/资源机制需要版本化 ABI 或新 Track；
5. 独立验证安全、正确性、计时和证据；
6. 未知版本或不支持能力 fail closed，不静默回退到另一套 ABI；
7. 不用旧 ABI 的 performance evidence 给新 ABI 背书。

**候选自由不等于“任何想法都已经由当前系统支持”；它意味着系统不为方便而暗中替候选规定算法。**

---

## 十、如果不用 ABI，会发生什么问题？

“不要 ABI”通常有两种含义：

1. 不写显式 ABI，但靠大家默契；
2. 让每个候选自己决定所有调用和内存约定。

第一种只是**隐式 ABI**，第二种会让独立 Harness 失去统一判断依据。

下面分别看实际后果。

### 10.1 连“结果对不对”都无法统一定义

假设候选只交：

```python
def run(*args):
    ...
```

没有合同就无法知道：

- 第 3 个参数是 `v` 还是 `g`；
- `scale` 是 Python float 还是 fp32 tensor；
- `cu_seqlens` 是 int32 还是 int64；
- 返回值的第二个元素是不是 `final_state`；
- `initial_state` 是否会被原地修改；
- workspace 是不是错误地混进 reference outputs。

此时即使运行没有报错，也无法可靠和 `reference.py` 比较。

**ABI 首先让“正确”有共同对象。**

### 10.2 Host 和 Device 可能读同一块内存，却读出不同含义

例如 Host 认为：

```c
struct WorkItem {
    uint16_t offset;
    uint16_t count;
};
```

Device 却认为：

```c
struct WorkItem {
    uint32_t offset;
    uint32_t count;
};
```

两边都能编译，内存也都存在，但 Device 会把相邻字节拼成不同字段。

再例如 native64 下把指针塞进 `uint32_t`：

```text
真实地址：0x00000001_23456780
截断之后：0x23456780
```

这不是“性能稍差”，而是直接访问错误地址。

没有共享 ABI，就没有可靠办法发现这种不一致。

### 10.3 无法可靠分配和检查 workspace

如果候选只说“我需要 scratch”，却没有：

- dtype；
- capacity；
- alignment；
- lifetime；
- offset/count；
- initialization policy；

Harness 就不知道：

- 要分配多少；
- 是否需要清零；
- 哪些区域有有效数据；
- Device 应该读取多少；
- 候选是否越界；
- 下一次调用是否能复用旧内容。

最终常见结果是：

- 分配过小导致越界；
- 分配过大导致性能测量失真；
- 未初始化数据被当成状态；
- Device 按最大容量读取垃圾尾部；
- 上一个 case 的 handoff 泄漏到下一个 case。

### 10.4 `read_write` 和 final state 很容易被漏掉

很多有状态算子不只是“输入 → 输出”：

```text
initial_state + 当前输入
       ↓
output + final_state
```

如果没有明确 ABI：

- 候选可能只返回 output；
- benchmark 可能忘记检查 final_state；
- 多次 correctness trial 可能不恢复 initial_state；
- 连续 timing 后的状态可能被误当成原始输入结果；
- Host 可能偷偷修改 read_write 输入。

ShapeForge 把 `read_write` 终值纳入 `tolerances + scaled_error`，就是为了防止“主输出对了，状态却错了”。

### 10.5 Host+Device 同步协议会失控

Handoff 至少要求双方知道：

- state 在哪里；
- flag 在哪里；
- 什么值表示 READY；
- 谁写、谁读；
- 什么时候清空；
- 失败后是否丢弃；
- 上一轮 READY 是否属于当前 generation。

没有共同协议，可能出现：

- consumer 先启动并永远等待；
- producer 写了 state，却还没发布 flag；
- consumer 读到上一轮的 READY；
- flag 已 READY，但 state 还没对另一方可见；
- stage 顺序与自旋假设不匹配。

这类问题不一定产生编译错误，可能表现为偶发错误结果或死锁。

### 10.6 计时口径会被人为或无意地掏空

如果候选自己决定 prepare/run 边界，却不声明：

- 哪些工作依赖本次输入；
- 哪些结果可以缓存；
- 哪些复制发生在计时内；
- workspace 何时分配和清零；
- output 何时同步可读；

它就可能把重工作全部放在计时外：

```text
计时前：
  读取输入
  计算 work list
  预处理数据
  建 graph
  清空 workspace

计时内：
  只按一下 replay
```

这个结果不是端到端执行成本。

ABI 和生命周期合同的作用，是让 Harness 知道哪些工作不能从测量中消失。**否则不同候选测的根本不是同一件事。**

### 10.7 只能哈希文件，却无法证明“以什么方式执行”

没有 ABI 时，仍然可以对源码算 SHA-256，但无法完整回答：

- 使用了哪个 Track/ABI 版本；
- 哪份 ExecutionGeometry 被冻结；
- 哪些 workspace 和 stage 被激活；
- 哪份输入和 resolved shape 被消费；
- 实际 dispatch 是否与选择一致；
- 哪个 baseline/environment 参与比较。

所以“源码没变”不足以复用旧证据。

**身份不仅是代码字节，也是代码怎样被构建和调用。**

### 10.8 Variant 和 fallback 可能选错实现

假设候选有：

```text
小形状 variant A
大形状 variant B
通用 fallback C
```

没有明确 guard 和 dispatch 合同，就可能：

- 同一个请求同时命中 A 和 B；
- 没有命中却静默使用未验证实现；
- fallback 的结果和时间被算成 variant 的成绩；
- Host 选择 A，Device 实际启动 B；
- tune 的 shape 被冒充 unseen 泛化证据。

ABI/调度合同需要保证 exactly-one dispatch，并把 fallback 单独记账。

### 10.9 跨硬件“字节一致”会被误当成“语义一致”

同一组字节在不同硬件上可能因为：

- dtype 不同；
- endianness/对齐不同；
- pointer width 不同；
- work-item 顺序不同；
- target 资源不同；
- 同步模型不同；

而具有不同含义。

ShapeForge 的 NPU 模拟 parity 因此区分：

- 同 target：比较完整 span 元数据、字节和 resolved scalars；
- 不同 target：比较去重 item 全集、唯一认领、完成集合和语义依赖边，允许 target-dependent 顺序不同。

而且这些都仍是 `simulated=true/publication_allowed=false`，不是实际 NPU 证据。

### 10.10 安全边界会退化为“相信候选自觉”

如果候选可以任意传指针、include 宿主文件、声明隐藏资源或改写输入，Harness 没有 ABI 就缺少检查对象。

当然，**有 ABI 也不等于绝对安全**：

- native ABI 本身不是 sandbox；
- runaway 仍需外部 watchdog；
- MSL/C 编译和加载仍需 worker 隔离；
- guard/span 检查不能证明所有数学错误。

但没有 ABI，连这些有限而有针对性的检查都很难统一实现。

---

## 十一、“我不用 ABI”的几种替代方案，其实仍然需要合同

### 方案 A：每个候选自己带一个 Python `run()`

这是最容易想到的方案。

它适合固定任务原型，但仍然隐式规定：

- 参数顺序；
- tensor 类型；
- 输出形式；
- buffer 生命周期；
- prepare 是否在计时内。

如果没有独立合同，Harness 只能执行候选自报的流程，无法公平比较不同候选。

### 方案 B：让候选自己写一份 JSON 描述

这比完全没有描述好，但仍然是 ABI，只是由候选提供。

还必须回答：

- 哪些字段由 Harness 独立派生；
- 哪些字段允许候选拥有；
- 自报与实际执行不一致怎么办；
- 未知字段和版本怎么办；
- digest 是否覆盖它；
- 是否能用 JSON 证明真实硬件行为。

**自报 JSON 是声明，不是硬件事实。**

### 方案 C：靠 MSL/C 反射自动推断一切

反射可以告诉 Harness 部分事实：

- kernel 名称；
- 参数名；
- 部分类型和绑定位置。

但它通常不能独立推导：

- 哪个参数是 `cu_seqlens`；
- 哪个 tensor 是 read_write；
- `final_state` 的数学语义；
- workspace 的初始化策略；
- Host prepare 是否该计时；
- tolerance 和 error scale；
- fresh evidence 属于哪个语义域。

反射是 ABI 绑定的工具，不是任务合同的替代品。

### 方案 D：所有东西都序列化后 RPC

序列化只是把 ABI 换成线上格式：

- 字段名；
- 类型；
- 版本；
- 顺序；
- 可选值；
- 错误语义；
- 字节编码；

这些仍然是 ABI/协议。它还可能增加复制和延迟，并不会消除合同。

### 方案 E：每个候选自带 launcher + evaluator

这是最自由、也最危险的方式：

- 候选能做任何优化；
- 但也能自己决定测什么、何时开始计时、比较哪些输出；
- Harness 不再是独立裁判。

这适合某些上游固定任务工作区，却不符合 ShapeForge 的可信发布目标。

**所以真正的问题不是“要不要 ABI”，而是“ABI 由谁拥有、固定多少、怎样验证”。**

---

## 十二、三方 ABI 可以怎样互相学习

| 上游机制 | ShapeForge 应该学习 | 不应照搬 |
| --- | --- | --- |
| KDA `prepare/run` | 把准备与执行生命周期说清，缓存必须有身份 | 把输入相关 prepare 一律放到计时外 |
| KDA work list | Host 根据真实输入生成调度工作 | 固定位布局、21-bit 上限、CTA/SM 规则 |
| KDA handoff | 长链切分、状态续传、fallback 和负例 | CUDA 跨 CTA 自旋/acquire-release 假设 |
| CAKE2 Host Tiling | Host+Device 原生协同交付 | CANN TilingContext/TilingData 原样移植 |
| CAKE2 TilingKey | applicability、稳定变体键、精确分发 | 把它当通用 Track 或 varlen work list |
| CAKE2 工程交付 | 编译、安装、绑定、评测形成闭环 | CANN package、ACL/PyBind、msopgen、msprof |
| ShapeForge 本地底座 | OpSpec、worker、canonical identity、fresh evidence | 为模仿上游再维护 KDA/CAKE2 双运行路径 |

最终不是：

```text
KDA ABI + CAKE2 ABI + ShapeForge ABI
```

而是：

```text
一条去重后的 ShapeForge 合同与执行链
  ├─ 学习 KDA 的生命周期和调度方法
  ├─ 学习 CAKE2 的原生 Host+Device 工程方法
  └─ 保留本地候选自由、Metal/清微边界和可信证据
```

---

## 十三、给实习生的 ABI 审查清单

以后看到一项 ABI 改动，可以按顺序问。

### A. 它属于哪一层？

- 任务语义？
- 外部 tensor ABI？
- Candidate/Track 交付？
- Host native ABI？
- 候选内部 work-item？
- 执行几何？
- 证据身份？

不要把不同层的字段混在一起比较。

### B. 它固定的是交接还是算法？

合理固定：

- 类型、权限、容量、生命周期、错误语义、身份。

警惕固定：

- tile、planner、chunk、stage 算法、某个上游位布局。

### C. 谁拥有它？

- OpSpec/reference 由任务合同拥有；
- ABI、安全、测量和证据由 Harness 拥有；
- 算法、Host Tiling、ExecutionGeometry 和内部布局由候选拥有；
- Track 只拥有交付机制。

### D. 改完之后哪些身份必须变化？

可能需要变化：

- opspec/instance digest；
- candidate manifest；
- derived header / assembled source；
- native binary；
- frozen geometry；
- artifact set；
- evaluation identity。

不能只改实现而继续复用旧证据。

### E. 有没有对应负例？

至少考虑：

- 参数顺序或 dtype 错；
- pointer 被改写；
- span 超 capacity；
- read-only 被修改；
- workspace 未声明；
- stage/grid 不一致；
- 非 0/trap 后部分提交；
- fresh 输入复用旧 plan；
- fallback 冒充 variant；
- NPU 模拟冒充真实性能。

### F. 它是否真的需要进入全局 ABI？

如果只是某个候选自己的编码方式，就留在候选内部。

**不要把一个候选的好实现，上升成所有候选必须服从的格式。**

---

## 十四、最后的一句话总结

**KDA 的 ABI 为“特定 KDA 算子在 CUDA 上高性能 prepare/replay 和调度”服务；CAKE2 的 ABI 为“Agent 生成 CANN 原生 Host+Device 算子工程并交付评测”服务；ShapeForge 的 ABI 为“让人或 LLM 自由写候选，同时由独立 Harness 统一分配、校验、执行、计时和留证”服务。**

再压缩成一句：

> **没有 ABI，候选看似更自由，实际上连输入输出、内存、生命周期、计时和证据都无法可靠共享；ABI 过大又会把算法自由吃掉。ShapeForge 的选择是：外部交接强类型、内部实现归候选、所有变更进身份、所有声明靠真实执行验证。**

---

## 十五、核对来源

### 15.1 ShapeForge 本仓库

1. [Native Host Program ABI](../架构/native-host-program-ABI.md)：当前 native64 Host ABI、输入/span/原子提交、L2 resolved shape 与证据。
2. [Native Host 数据流耗时报告](../架构/native-host-program-数据流耗时报告.md)：当前 Host 路径和计时边界；本文不引用其数字作为跨系统性能比较。
3. `tcenv/track/abi.py`：`TargetAbiDeriver` 固定 tensor/ABI 事实，明确不包含 tile/grid/threads。
4. `tcenv/track/plane.py`、`tcenv/track/delivery.py`：TensorMemoryContract、CandidateArtifact、ExecutionGeometry 和 candidate manifest。
5. `tcenv/hostprogram/native_abi.py`、`native_toolchain.py`、`native_runtime.py`：C 编译器拥有布局、worker-only native64 构建/加载与真实调用。
6. `tasks/varlen_kda/opspec.json`：当前 i32 `cu_seqlens`、fp32 `scale` tensor、符号形状、workspace 容量和候选位布局边界。
7. [v2 协议参考](../架构/v2协议参考.md) 与 [AGENTS.md](../../AGENTS.md)：候选自由、Track、身份和计时不变量。
8. [上游移植台账](../../tests/support/golden/shared/跨硬件算子生成分工-G0/upstream-map.json)：固定上游版本和 reference/reuse/port/adapt/local 决策。
9. [Host 调度成本对比](KDA-CAKE2-与ShapeForge的Host调度成本对比.md)：Host 成本与 prepare/run 计时口径。
10. [多 Track 交付对比](KDA-CAKE2-与ShapeForge的多Track交付对比.md)：为什么 Track、Variant、Agent 和 Target 不是同一个概念。

### 15.2 KDA 外部只读快照

- 仓库：`NVlabs/kda` 的 kernel 分支快照；
- 审查提交：`f539baab8e46712eb03c88562fc9d9c4b0798aa3`；
- 主要文件：
  - `README.md`：外部 `run` 参数、dtype/shape 和 prepare/run 生命周期；
  - `kda-tirx/kernel.py`：TIRx prepare、输出分配和 graph replay wrapper；
  - `kda-tirx/kda_fwd_varlen.py`：CTA work-item `(word0, word1)`、21-bit token packing 和范围断言；
  - `kda-cake-ptx/kernel.py`、`kda_ptx/scheduler.py`：prepare、scheduler、fallback、descriptor/handoff 生命周期；
  - `kda-cake-ptx/kda_ptx/shims/*.cc`：TensorView、raw pointer、kernel args 和 CUDA launch 边界；
  - `bench.py`：固定 workload、模块加载与 correctness/timing 入口。

### 15.3 CAKE2 外部只读快照

- 仓库：`https://gitcode.com/cann/cannbot.git`；
- 审查提交：`b699c3247241dbe0884f0ea95c619d0cc85d30f4`；
- 根目录：`plugins-community/collaborative-agent-kernel-evolution/`；
- 主要文件：
  - `skills/op-desc-generation/references/op_desc_template.json`：op_desc 示例字段；
  - `skills/ascendc-evaluation/scripts/evaluate.py`：Model/ModelNew、输入生成与三方数值评测；
  - `docs/EVALUATION_INSTRUCTION.md`：op_host/op_kernel、构建安装、PyBind 和 Evo variant 切换；
  - `skills/dsl-easyasc-post/references/migration_examples/flash_attention/`：`TilingFunc`、workspace、GM_ADDR 和 `GET_TILING_DATA` 示例；
  - `skills/cake-evo/references/meta_prompts/strategies/dtype_03_tilingkey_dispatch.md`：TilingData/TilingKey 与特化分发策略。

这些外部材料仅用于架构比较。CUDA/CANN 的代码、工具链、芯片知识和专用 ABI 不进入 ShapeForge 生产、pack 或 `agent_guidance`。
