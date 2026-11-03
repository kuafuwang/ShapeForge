# 外部咨询：small GEMM 的公平 GPU 测量与 Metal 执行路径优化

## 我希望你解决什么问题

请给出**可落地、可复现、可审查**的技术方案，解决下面两个相互独立的问题：

1. 如何公平比较我们生成的 Metal kernel 与 fastest registered baseline（当前是 MLX）的
   **kernel/GPU 性能**，而不是拿 candidate GPU-only 时间除以 baseline wall 时间；
2. 如何降低我们当前 Python/PyObjC Metal 执行路径的 launch/encoding 成本，使小算子的
   **端到端 wall 性能**也有机会与 MLX 比较。

请不要只回答“用 Instruments / xctrace / 优化 tiling”。我们已经做了下述实验。需要具体到：

- 可用的 Apple/MLX API 或执行器结构；
- candidate 和 baseline 各自的测量边界；
- 如何归因到具体进程、command buffer、encoder、kernel；
- 如何控制 tracing perturbation、GPU 频率和 command coalescing；
- 为什么该方案不会把不同口径的数字混成一个 speedup；
- 最小验证实验与预期观察。

---

## 1. 系统与任务

- 设备：Apple M4 Max；Metal/MSL；没有 CUDA。
- Xcode/xctrace：`xctrace version 27.0 (27A266a)`。
- MLX：`0.32.3`。
- Python：3.13，PyObjC Metal binding。
- 当前仓库：`/Users/kuafu/CLionProjects/ai_lowerling`。
- 任务：标准 `tasks/gemm`，fp16 输入，fp32 累加，fp16 输出。
- small GEMM：`M=256, N=256, K=128`，即
  `C[256,256] = A[256,128] @ B[128,256]`。
- paper GEMM：`M=N=K=2048`，已经通过当前性能门；这里只咨询 small。
- fastest registered baseline：MLX；另一个 Torch MPS baseline 明显更慢，不应拿它替代 fastest baseline。

当前性能政策要求每个 shard：

```text
latency_ratio = candidate_wall / baseline_wall <= 1.10
```

不能因为当前候选不达标而修改 1.10、删除 small shard、挑选较快一轮，或者把 correctness-only
证据当成性能证据。

---

## 2. 为什么现在有两种不同的“公平”

### 系统端到端公平

如果用户实际通过我们当前的 Python/PyObjC runtime 调 kernel，那么：

```text
encoding → commit → GPU execution → waitUntilCompleted
```

都是系统真实成本。MLX 的原生路径开销更低也是它的真实优势。因此当前 wall ratio 可以描述
“当前系统端到端性能”。

### kernel 算法公平

当前 candidate 支付 Python/PyObjC 编码成本，MLX 支付其原生框架成本。因此 wall ratio 不能解释为
“生成的 kernel 算法本身比 MLX 慢多少”。

我们**不能**采用：

```text
candidate GPU-only / baseline wall
```

这仍然是不公平比较。若建立 kernel-only 指标，candidate 和 fastest baseline 必须都有同一测量根的
GPU-only 时间，而且必须证明 profiling 本身没有改变结论。

---

## 3. 当前实测结果

### 正常生产 wall 计时

经过第一轮 runtime 编码优化后，一次 fresh small 测量约为：

```text
candidate wall = 4.56 us
MLX wall       = 2.29 us
ratio          = 1.99
```

仍存在 `timing_uncertain`，所以 P3-B 保持 blocked。

paper GEMM 当前 ratio 约 1.015，correctness/dispatch/performance 均通过；paper 不能掩盖 small 失败。

### candidate 自身 wall/GPU 分解

生产 `MTLCommandBuffer.GPUStartTime/GPUEndTime` 可用。一次诊断运行：

```text
candidate GPU timestamp median = 3.30 us / launch
candidate wall median          = 5.64 us / launch（诊断捕获运行）
wall - GPU                     = 2.40 us / launch
同轮 MLX wall                  = 2.77 us
```

诊断捕获与无插桩门运行不能逐位比较，但结论稳定：small 既有 kernel 成本，也有明显路径成本。
只删除路径统计既不公平，也不必然让 candidate 达标。

证据文件：

- `docs/架构/shape-portfolio-阶段验收/P3/evidence/gemm-performance.json`
- `docs/架构/shape-portfolio-阶段验收/P3/evidence/small-timing-diagnostic.json`
- `docs/架构/shape-portfolio-阶段验收/P3/evidence/kernel-only-feasibility.json`

---

## 4. 已完成的 runtime 优化

修改：`tcenv/backends/metal/runtime.py`。

已正式落地且通过 correctness/dispatch 锚点：

1. 缓存 launch 的 `MTLSize`；
2. 单阶段 kernel 复用相同 pipeline state；
3. 执行路径使用等价 operation iterator，避免定时窗内创建大 list；
4. workspace fill 后仍强制重新绑定 candidate pipeline；
5. **没有改变计时边界**：编码、commit、sync 仍计入 wall。

效果：small candidate median 从约 5.85us 降到约 4.56us，但 ratio 仍约 1.99。

固定测试：`tests/backends/metal/test_launch.py`，18 个节点通过，覆盖单阶段复用与 fill 后重绑。

批量 `setBuffers_offsets_withRange_` 实验没有进一步收益，未采用。

---

## 5. 已失败的执行器路线，请不要原样重复

### Indirect Command Buffer（ICB）

目标：预编码重复 dispatch，减少 Python 热路径。

结果：

1. `MTLIndirectComputeCommand.setComputePipelineState_` 在最小复现中发生原生段错误；
2. 改为 `inheritPipelineState=True`，由外层 encoder 设置 pipeline 后，可以完成编码；
3. 首次 benchmark 仍出现 `GPU Address Fault / Page Fault`；
4. 添加 `useResources_count_usage_` 后仍未解决；
5. 普通非 ICB correctness 路径正常，因此没有把该原型带入生产。

原型在 `/tmp/p3_icb_experiment.py`（临时文件，不是生产代码）。

如果你建议 ICB，请具体说明：

- descriptor 的 `commandTypes`、inherit flags 应怎样设置；
- PyObjC 下应使用 `concurrentDispatchThreads_threadsPerThreadgroup_` 还是其它 API；
- buffer 绑定、residency、`useResource(s)` 和 command execution range 的正确组合；
- 如何解释并修复上述 address fault；
- 是否应该放弃 ICB，改用 Objective-C/C++ 原生批量 encoder。

### small kernel 变体

已尝试：

- 4 种已有 simdgroup tiling；
- 两份归档优化 GEMM 候选；
- 直接全局加载；
- 64×64 全 K staging；
- 32×32 小 tile；
- 16×16/warp；
- golden A/B pipeline + 直接写回 epilogue，删除 C staging 尾部 barrier。

均未稳定达到 1.10。直接 epilogue 约 2.10；32×32 在新 runtime 下约 2.05。当前保留简单且正确的
golden 候选作为 blocked 证据，不继续盲调参数。

---

## 6. Metal System Trace：已解决录制，但还没有公平门

### 最初错误

```text
Failed to start the recording: Could not set the recording priority.
No configuration information received, will have to guess.
```

当时机器上有 Instruments/DTServiceHub 正在运行。关闭 Instruments 后，同一个 `--launch` 命令成功，
所以录制问题至少在本机已解决。旧输出目录还存在时需要使用新路径或删除旧目录。

成功命令：

```bash
cd /Users/kuafu/CLionProjects/ai_lowerling
xcrun xctrace record \
  --template 'Metal System Trace' \
  --no-prompt \
  --output /tmp/p3-small-fresh.trace \
  --time-limit 90s \
  --launch -- .venv/bin/python /tmp/p3_profile_once.py
```

导出：

```bash
xcrun xctrace export \
  --input /tmp/p3-small-fresh.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
  --output /tmp/p3-fresh-gpu.xml
```

结果：

- 2,366 个 compute rows；
- 其中 2,026 个属于目标 Python 进程；
- XML 大约 11 MB；
- process 名在 `event-label` 中，可能是 `Python (pid)` 或 `python (pid)`，过滤时应忽略大小写。

### 仍然不能直接用 full interleaved trace 做 kernel-only ratio

这次 full benchmark 本身交错执行 candidate 和 MLX，但 export 中：

- interval 主要是 command-buffer/encoder 级；
- 没有可靠的 kernel 名；
- candidate 一个 command buffer 可包含许多 launch；
- MLX 也可能合并操作；
- 因而不能安全地把每一行归因成 candidate 或 MLX kernel。

请特别注意：不要建议简单按 `event-label` 或 duration 大小猜测 candidate/MLX。

### standalone MLX trace

脚本：`/tmp/p3_mlx_baseline_profile.py`，只执行 small MLX GEMM。

```bash
MLX_MAX_OPS_PER_BUFFER=1 xcrun xctrace record \
  --template 'Metal System Trace' \
  --no-prompt \
  --output /tmp/p3-mlx.trace \
  --time-limit 30s \
  --launch -- .venv/bin/python /tmp/p3_mlx_baseline_profile.py

xcrun xctrace export \
  --input /tmp/p3-mlx.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
  --output /tmp/p3-mlx-gpu.xml
```

结果：

- 18,893 个目标 Python compute rows；
- GPU interval median 约 **9.58us**；
- 但同一 MLX GEMM 正常 wall median 只有约 **2.5us**。

这说明 tracing 和 one-op-per-buffer 明显扰动了测量。**不能拿 9.58us 当 MLX 的真实 GPU-only 时间，
更不能为了让 candidate 通过而使用这个偏慢 baseline。**

---

## 7. `.gputrace` 路线

以下可以生成 capture：

```bash
METAL_CAPTURE_ENABLED=1 uv run python /tmp/p3_capture_once.py
```

内部调用：

```python
mx.metal.start_capture('/tmp/p3-small.gputrace')
# candidate + baseline evaluation
mx.metal.stop_capture()
```

但：

```text
xctrace export: Cannot import files of type 'com.apple.gputools.gputrace'
```

所以目前理解是：

- `.gputrace` 用 Xcode GPU Frame Debugger 打开；
- `.trace` 用 xctrace export；
- 不希望自研私有 `.gputrace` 二进制解析器。

如果你知道 Apple 官方支持的 `.gputrace` headless 导出/转换方式，请给出确切工具和命令；否则请明确
它只能作为 Xcode 诊断，不要建议不存在的 xctrace 转换。

---

## 8. 公共 API 检查结果

- candidate 自己的 `MTLCommandBuffer.GPUStartTime/GPUEndTime` 可读取；
- MLX Python 暴露 `start_capture/stop_capture`，但没有找到 command-buffer GPU timestamp accessor；
- 安装包包含 MLX C++ headers/`libmlx.dylib`，但未找到公开的 GPU timestamp 接口；
- Torch MPS 有 OS Signpost/Metal capture，但 Torch 不是 fastest baseline，不能拿它替代 MLX；
- Metal System Trace 可用但扰动和归因问题尚未解决。

---

## 9. 我希望你重点回答的方案选择

### A. 公平的 baseline GPU-only 测量

是否存在不显著扰动生产的 MLX GPU timestamp 获取方式？例如：

- 官方 C++/Objective-C hook；
- command-buffer completed handler；
- 可审查的 MLX 扩展点；
- 与 candidate 相同的原生 Metal command buffer 包装；
- 或其它不依赖慢速 tracing 的方法。

如果必须改 MLX 源码/使用私有符号，请明确代价、版本绑定和为什么仍可作为可信 baseline，而不要把它
描述成零成本公共 API。

### B. 如果只能用 Metal System Trace

请给出严格设计：

- candidate 和 MLX 分开录还是同录；
- 怎样设置 command queue/encoder 标签；
- 是否以及何时使用 `MLX_MAX_OPS_PER_BUFFER=1`；
- 如何处理 command-buffer 多 launch 与 encoder coalescing；
- 怎样控制 GPU 时钟、trace 开销与测量扰动；
- 怎样证明 tracing 下 candidate/baseline 的相对关系能代表非 tracing 生产关系。

如果无法证明，请直接说 xctrace 只能诊断，不能作为该性能门。

### C. 原生执行路径

如果建议 Objective-C/C++ 原生批量 encoder，请给出最小架构：

- 哪些工作一次性预计算；
- 每个 benchmark batch 仍需计时的实际工作；
- 如何保持 rotation、read_write restore、workspace zero fill、barrier 和 dispatch parity；
- 如何从 Python 调用而不重新引入同等开销；
- 如何验证不是通过把应计时工作移出测量窗来“提速”。

如果建议继续 ICB，请优先解释本例 address fault，而不是只给官方概念链接。

### D. 如果公平 kernel-only 暂时做不到

请评价以下保守做法是否合理：

- 保留 wall 作为唯一发布性能门；
- GPU timestamp/xctrace 只作诊断；
- small P3-B 继续 blocked；
- 原生执行器作为独立工程；
- 不发布“kernel 算法比 MLX 慢多少”的结论。

---

## 10. 不可接受的答案

- 直接删除 candidate 路径成本，但仍除以 baseline wall；
- 使用 tracing 下约 9.58us 的 MLX 数字来让 candidate 显得更快；
- 改用更慢的 Torch MPS baseline 替代 fastest MLX；
- 修改 1.10、删除 small shard、选择偶然较快的一轮；
- 只说“用 Instruments”“优化 tiling”“用 C++”而没有 API、测量边界和验证步骤；
- 把 `.gputrace` 与 Instruments `.trace` 当成同一种可直接 xctrace export 的格式；
- 把任何模拟、估计或单轮自报数字当成 fresh 硬件证据。

---

## 11. 当前仓库状态与相关文件

- P3-A 已完成：两本账、1.10 门、policy 绑定和独立 reviewer ACCEPT。
- P3-B 仍 blocked：paper 通过，small 未达到端到端 1.10。
- 已归档诊断：`docs/架构/shape-portfolio-阶段验收/P3/evidence/`。
- 阶段说明：`docs/架构/shape-portfolio-阶段验收/P3/B.md`。
- 权威计划：`docs/架构/shape-portfolio实施计划.md`。
- 本咨询文档只描述问题，不构成新的性能证据或许可。

请优先给出一个**最小可验证方案**，再讨论长期原生执行器架构。

---

## 12. 追加：外部建议复核与原型结果

已收到一份外部建议（`.pytest_cache/argon7.md`），不能照单全收：

1. **“每 rep 重建 encoder”的前提不成立。** 当前 `run_pipeline` 在整个 command buffer 外只创建一次
   compute encoder，`endEncoding()` 也在 rep 循环之后；故其 MVE-1 不是新的优化。
2. **“ICB fault 是 M/N/K 位于 buffer(3..5) 的 setBytes 参数未继承”不适用于当前 v2 GEMM。**
   本候选 M/N/K 是编译期常量，真实 stage 参数只有 A/B/C 三个 buffer。ICB address fault 根因仍未成立，
   不应按该断言修复。
3. **原生批量编码器方向已被原型验证，但尚未过门。** /tmp Objective-C 原型将 Python 每批调用降为
   一次 FFI，native wall 包含 commandBuffer 创建、全部编码、commit 和 waitUntilCompleted。
   golden/32×32/direct-epilogue 的 candidate 约 2.83–2.86us，wall-GPU 约 0.34–0.37us，
   ratio 约 1.20–1.32，仍高于 1.10，且尚未生产化。
4. **commit-hook 可以获得 MLX GPU intervals，但不能直接求和。** MLX 一批可提交多个重叠 command
   buffer，需对 GPUStartTime/GPUEndTime 区间取 union。当前 hook off/on wall 中位数比约 1.105，
   扰动不是建议中所称的可忽略；因此仍只作诊断，不作 kernel-only 门。

新增证据：`P3/evidence/native-batch-feasibility.json`。

下一步如果继续，应审查并生产化原生批量编码器，同时保持 small blocked；不能用原型数字直接替换
当前生产门，也不能把 hook 的 MLX GPU union 与 candidate wall 混算。

---

## 13. 追加：第二份外部建议的验证结果

### Chunked commit

已用 `MTLSharedEvent` 保留 rotation 依赖，并把全部编码、commit 和最终等待留在 wall 内：

| 路径 | candidate | 同轮 MLX | ratio |
|---|---:|---:|---:|
| 单 native command buffer | 2.80us | 2.23us | 1.25 |
| CHUNK=8 | 6.74us | 2.10us | 3.20 |
| CHUNK=16 | 8.32us | 2.54us | 3.28 |
| CHUNK=32 | 4.72us | 2.17us | 2.17 |

因此“chunk 后 wall 会自然降到约 2.58us”在当前实现中不成立。额外 encoder/commit/shared-event
成本大于 overlap 收益，未采用。

### MLX 真实几何与 NAX

Geometry hook 读到：`grid=(4,4,1)`、`threads=(32,2,2)`、静态 threadgroup=5376B、PSO label 为空。
候选也是 4×4 grid/128 threads，故不是简单的 grid 或 40 核波次问题。

single-buffer K16 版本 ratio 约 1.48，未采用。

MLX metallib 中存在 `steel_gemm_fused_nax_*` 符号。使用公开 `NAXTile/gemm_loop` 生成了 38,745B
自包含候选，但当前 wrapper 在修正漏写的 `c_col` 后仍有 65,454/65,536 元素超差，不能用于性能
证据。完整预处理版本为 1,699,390B，超过当前 1,000,000B 候选上限；保留 MPP SDK include 的
版本则尚未获得正式 include 白名单扩展。

请下一轮建议优先回答：

1. 原生单 command buffer 已有约 0.34us 路径成本时，剩余 1.20–1.32 ratio 的 kernel 侧优化应
   如何基于真实 MLX NAX 实现完成，而不是继续猜 tiling；
2. 若要复用 `NAXTile/gemm_loop`，wrapper 还需要满足哪些未说明的 ABI/布局/函数常量条件；
3. 是否值得正式允许唯一的 SDK `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`
   include，同时继续禁止所有宿主和 MLX 项目路径；
4. 如果 NAX 适配继续失败，是否应停止 kernel-only 路线，保留 wall 门和 small blocked，
   只生产化原生批量编码器。

后续已关闭 NAX 路线：MLX 用 `metal::is_nax_available()` 分派 NAX，NAX 属于 M5 Neural
Accelerator；M4 Max 不走该路径。普通 M4 `GEMMKernel/BlockLoader/BlockMMA` 自包含候选已通过
真实 ABI/correctness，但没有稳定优于 golden，未采用。后续建议不要再把 M5 NAX 当作 M4 的
优化手段；应聚焦普通 Steel/MSL 内核与原生单 command-buffer 执行器，或保留 small blocked。
