# per-launch 与 shape-portfolio 性能归因（2026-11-25）

## 结论先行

此前的两个直觉解释需要修正：

- **per-launch 不是 Host 固定成本增加。** 它的提交/编码/等待外部成本反而更低；真正的问题是并发 CommandBuffer 的 GPU busy 总和增加约 **68.18%**，重叠带来的资源竞争吃掉了大部分收益。
- **shape-portfolio 不是单纯 Host 抖动。** candidate GPU 时间戳本身的 CV 高于 wall CV；Host/outside 部分只有 **0.18–0.34 µs**，不是主要噪声源。

这两个结论来自逐样本匹配的真实 `RunTiming`，不是根据中位数猜测。更底层的 GPU 受限资源（compute、memory、scheduler、clock）尚未由这些时间戳区分，不能指定其中一个作为根因。

## 测量方法

没有模型调用、付费搜索或生产代码改动，也没有重跑整套测试求通过。

### per-launch

对 `softmax/fp16/paper` 的同一候选 SHA-256 `205c44412be8d6251822fce922fe8f32d6c2edb798be7cd0dd84392f57ec72b6`，在三个既定 seed 下，将报告选中的每条 wall sample 一对一匹配回 `RunTiming`：

- `wall`：每次 launch 的端到端时间；
- `gpu_busy`：逐 CommandBuffer `GPUEnd-GPUStart` 的总和；
- `gpu_span`：`max(GPUEnd)-min(GPUStart)`，包含重叠但不重复计算重叠区间；
- `wall-span`：GPU span 之外的编码、提交和等待成本；
- `busy-span`：GPU busy 中被端到端重叠消去的部分。

### shape-portfolio

对七个 pointwise shard，将选中的 15 条 candidate wall sample 匹配回融合单 CommandBuffer 的 GPU 时间，分别重算 wall CV、GPU CV 和 outside CV。MLX CV 来自同一交错窗口。

## per-launch：并发存在，但 contention 更大

三个 seed 的中位分解：

| 指标 | single buffer | per-launch SharedEvent |
| --- | ---: | ---: |
| wall / launch | 147.292 µs | 141.986 µs |
| GPU busy / launch | 140.039 µs | 235.515 µs |
| GPU span / launch | 140.039 µs | 136.421 µs |
| wall − span / launch | 7.253 µs | 5.279 µs |
| 相对 MLX 的中位 speedup | 0.9534× | 0.9965× |

因此：

1. per-launch 确实发生重叠：每次 launch 的 `busy-span` 约 **99.1–102.1 µs**。
2. wall 相对 single buffer 降低约 **3.60%**，不是“机制完全没有收益”。
3. 但 GPU busy 增加约 **68.18%**。相同数量的 kernel 在重叠时各自执行更慢，说明并发成本很大；时间戳不能进一步指定是哪种硬件资源。
4. GPU span 只从约 140.039 降到 136.421 µs。加上约 5.279 µs 的外部成本后，candidate wall 仍略慢于 MLX wall，所以 `new_median > 1.0` 仍失败。

这否决了“主要是 SharedEvent/CommandBuffer Host 固定成本导致不达标”的解释。当前直接原因是：**重叠收益真实存在，但 GPU 执行膨胀抵消了大部分收益，最终只达到与 MLX 接近、略慢的水平。**

## shape-portfolio：GPU 本身不稳定

本次诊断六个 shard 为 `timing_uncertain`：

| 指标 | 范围 |
| --- | --- |
| candidate wall 中位 | 0.415–0.943 µs |
| candidate GPU 中位 | 0.232–0.581 µs |
| wall − GPU 中位 | 0.178–0.343 µs |
| 不稳定 shard 的 wall CV | 7.28%–18.60% |
| 不稳定 shard 的 GPU CV | **11.70%–28.84%** |
| MLX CV | 约 1.78%–3.07% |
| latency ratio | 0.334–0.875 |

安静门要求 CV 不超过 **5%**。触发 `noise_cv` 的是 candidate wall CV，而同一批样本的 GPU CV 更高；因此不能把失败归因为 Host 提交抖动。亚微秒 kernel 的绝对波动很小，但相对 CV 很容易超过 5%。

所有 shard 的 correctness、dispatch 和 latency-ratio 数值仍满足各自功能/1.10 检查；失败的是**测量可信度**，不是“候选已经比 baseline 慢”。

本次数据没有区分 GPU clock、系统调度或具体执行资源。若继续追到底层硬件原因，需要单独的 counter/trace 实验；trace 会扰动计时，不能把 profile 窗口直接拿来替换这批 fresh 数字。

## 决策边界

- 不采用 per-launch 作为默认提交方式；本次诊断不覆盖完整 GPU 套件中的失败。
- 不改变 5% CV、1.10 latency ratio、>1.0 或额外 3% 的门。
- 不把重复诊断窗口当成独立效果样本，不追加样本求绿。
- `publication_allowed=false`，无 speedup 发布。

逐行数据见 [performance-attribution-2026-11-25.json](performance-attribution-2026-11-25.json)。
