# P3-B：第二个 tiled GEMM 任务（当前 blocked）

状态：**blocked（paper 已通过；small 的真实性能仍不满足 1.10）**。

## 当前实测结果

- `fp16/paper/benchmark`：使用 golden 归档的 tiled/simdgroup 候选，correctness 与 dispatch 均通过，
  candidate/baseline median ratio ≤1.10；证据为 `P3/evidence/paper-performance.json`；
- `fp16/small/benchmark`：当前正确候选仍慢于 fastest registered baseline，出现
  `latency_ratio > 1.10` 与 `timing_uncertain`；`PerformanceReport.publishable=False`，证据为
  `P3/evidence/gemm-performance.json`；
- 早期 paper 的 `identity_error` 来自错误的 naive 候选，已通过替换为正确 golden 候选关闭；
  旧 `paper-blocked.json` 只保留为该次失败的记录，不再是当前 paper 状态。

## 已做过的 small 手工优化

- 4 种已有 simdgroup tiling；
- 两份归档优化 GEMM 候选；
- 直接全局加载的 simdgroup kernel；
- 64×64 全 K staging、32×32 小 tile、16×16/warp 版本。

这些实现 correctness 可构造，但 small candidate median 多轮仍约 4.5–8us，未稳定达到 fastest baseline
的 1.10 以内。不得通过挑选较快一轮、删除 small 分片或修改阈值来宣称完成。

## 时间分解诊断（非发布证据）

`P3/evidence/small-timing-diagnostic.json` 使用生产 `run_pipeline` 返回的真实 GPU timestamp 与墙钟，
明确标记 `diagnostic_only=true`、`publication_allowed=false`：

- candidate GPU median：约 **3.30us/launch**；
- candidate wall median：约 **5.64us/launch**（诊断捕获运行，不能与无插桩门数字逐位比较）；
- wall-GPU 差额（编码、commit/sync 等）：约 **2.40us/launch**；
- 同轮 MLX baseline wall：约 **2.77us**。

runtime 编码优化已正式落地：缓存 `MTLSize`、单阶段复用 pipeline state、执行路径使用等价
operation iterator；fill 后仍强制重绑 candidate pipeline。普通 P3-B 门运行中的 small candidate
median 已从约 5.85us 降至约 4.56us，但 ratio 仍约 1.99，继续 blocked。批量 buffer binding
实验没有进一步收益，未采用。small 直接写回 epilogue、删除 C staging 尾部 barrier 的实验 ratio 约 2.10，
同样未采用。Indirect Command Buffer 原型在 inherit-pipeline 后仍于首次 benchmark 出现 GPU address
fault，未进入生产；普通 correctness 路径不受影响。

因此 small 不是单纯的 kernel 问题，也不是单纯的 launch 问题：即使把 wall-GPU 差额全部消除，
当前 GPU 时间仍高于该轮 baseline wall；而只优化 kernel 也无法在约 2.40us 固定差额下达到 3.0us
左右的总时间目标。后续必须分别评估 runtime 编码路径与 kernel，不得通过排除应计时工作来过门。

如果后续决定增加 kernel-only 口径，必须为 candidate 与 baseline 建立同一 GPU-only 测量根，开启新
performance epoch 并独立复核；不能只删除 candidate 的路径成本后继续除以 baseline wall。当前 GPU
时间本身也未稳定低于现有目标，因此“不统计路径”不是可直接宣告通过的捷径。

已实际检查可行性（`P3/evidence/kernel-only-feasibility.json`）：最初 Metal System Trace 因 Instruments
占用/录制优先级失败，关闭 Instruments 后同一 `--launch` 命令已成功并导出非空 intervals；
standalone MLX 也得到 18,893 个 compute rows。但 trace 下 MLX median 约 9.58us，远高于正常约
2.5us wall，且 interleaved trace 不能按 kernel 名安全区分 candidate/MLX，故只能诊断，不能作门。
MLX `.gputrace` 仍应用 Xcode GPU Frame Debugger，xctrace 不支持导入。当前
`baseline_gpu_only_available=false`、`kernel_only_gate_allowed=false`，不得发布 kernel-only ratio。
原生执行器仍是独立的后续工程，不能把这些工具结果写成公平 kernel-only 测量已经建立。

## 原生批量执行器、官方 full-run 与 commit-hook 诊断（性能仍未发布）

外部建议经代码核对有两处不适用：`run_pipeline` 本来就在整个 batch 外只创建一个 encoder，
故“每 rep 重建 encoder”不是当前事实；当前 v2 GEMM 的 M/N/K 是编译期常量，真实参数只有
A/B/C 三个 buffer，不能将 ICB fault 断言为 `buffer(3..5)` 标量未继承。

但原生批量编码方向有效：`P3/evidence/native-batch-feasibility.json` 记录 /tmp Objective-C
原型，native wall 包含 commandBuffer 创建、全部编码、commit 和 wait。golden/32×32/direct-epilogue
三种测量的 candidate 约 2.83–2.86us，wall-GPU 差额约 0.34–0.37us，但 ratio 仍约 1.20–1.32，
故原型没有使 P3-B 通过。随后固定 harness 原生批量编码器已生产化：仅接受单阶段、buffer-only、
无 fill batch，其它路径回退 PyObjC；launch hooks 保留，Python/原生输出一致性有固定 GPU 节点，
预编译在计时窗外，且不信任预存 dylib。生产化不等于性能通过。

commit-hook 诊断可以读取 MLX command-buffer GPU intervals；重叠区间必须取 union，不能累加。
hook off/on wall 中位数比约 1.105，扰动不可忽略，所以仍只允许诊断，
`kernel_only_gate_allowed=false`。

后续建议也已实测：带 `MTLSharedEvent` 的 CHUNK=8/16/32 分段 commit 反而约 6.74/8.32/4.72us，
均差于单个原生 command buffer 约 2.80us，未采用。MLX 几何 hook 读到 `grid=4×4`、
`threads=32×2×2`、静态 threadgroup 5,376B，与候选 grid/threads 相同；single-buffer K16 仍约
1.48×，未采用。NAX 自包含适配修正输出指针后仍有 65,454/65,536 个元素超差，未用于性能证据，
也未正式扩大 MPP include 白名单。进一步确认 NAX 由 `metal::is_nax_available()` 分派，属于 M5
Neural Accelerator 路径，**M4 Max 不应继续投入 NAX wrapper**。普通 M4
`GEMMKernel/BlockLoader/BlockMMA` 自包含版本已通过真实 ABI/correctness，但性能未稳定优于 golden，
未采用。详见 `P3/evidence/native-batch-feasibility.json`。

进一步直接调用官方 `GEMMKernel::run`、使用编译期 `GEMMParams` 的 small full-run 已通过真实
ABI/correctness/dispatch。去掉冗余 `TransformNone` 预转换后，best fresh 原型 candidate 2.201us、
MLX 2.199us（ratio 1.001，但 noisy）；同协议官方 binary 对照 candidate/official=0.949。
这些只证明路线与局部最好效果，不能替代验收。最新生产证据 ratio 约 1.329，虽
`timing_uncertain=false` 仍超 1.10，**不得挑选 quiet 一轮标完成**。当前保留 full-run 为正确但未达标的
small 候选，P3-B 继续 blocked。

## 仍需完成

- 先降低 candidate 编码/commit/sync 路径的固定成本，再提供 small 专用候选；最终仍须 fresh correctness
  与 timing 全过且 ratio ≤1.10，不能用诊断中的 GPU-only 时间代替端到端门；
- 在此之前 P3-B 不得标 completed，P3 不得进入整体性能发布；paper 的通过不能掩盖 small 的失败。

外部复核所需的完整问题描述、复现命令和不可接受答案清单见
`P3/small-gemm-performance-consultation.md`。MLX small 官方 AIR 反汇编与 hot path 分析见
`P3/mlx-small-disassembly-analysis.md`：官方是普通 BK16/fp32-fragment Steel，非 NAX；golden
是 BK32/fp16-fragment，不能用总指令数简单判断性能。
