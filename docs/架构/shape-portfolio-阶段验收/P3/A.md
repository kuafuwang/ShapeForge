# P3-A：性能模型与两本账

状态：**completed（独立 reviewer ACCEPT；20 个 CPU 节点通过，policy 与失败聚合边界闭合）**。

## 范围

- 仅 `usage=benchmark` 的实例进入性能比较；golden-only 不进入；
- `latency_ratio = candidate_us / baseline_us`（越低越快）；`speedup = baseline_us / candidate_us`
  （与全仓同向）；`specialized_hit_latency_ratio` 为逐 shard ratio 的中位数，
  `effective_end_to_end_latency_ratio` 为总候选时间/总基线时间；
- 每个 shard 的 specialized/effective ratio 都不得超过 **1.10**，geomean 只能描述，不能掩盖局部退化；
  未知 usage、重复 instance、缺预期 shard 或全 fallback 无 variant 也阻断；
- `timing_uncertain`、unknown、缺 baseline、correctness/dispatch 失败均阻断发布；
- 原始测量不是 fresh proof，性能发布仍需 fresh timing/confirmation。

## 实现

`tcenv/portfolio_performance/performance.py`：

- `ShardMeasurement` 严格校验 instance/provider/时间与状态；
- `PerformanceReport.evaluate` 计算两本账、fallback time/count fraction、逐 shard 失败原因；
- `MAX_REGRESSION_RATIO = 1.10`；
- 不使用 P2 已完成的 `tcenv/portfolio_validation` 包，避免破坏其封闭验收面。

## 固定 CPU 证据

`tests/shared/test_shape_portfolio_performance.py` 20 个节点：

1. golden 被忽略，benchmark 才比较；
2. fallback 只进入 effective 账；
3. geomean 不能隐藏 specialized 局部回归；
4. fallback 回归只影响 effective 门，不给 variant 记功；
5. timing_uncertain/unknown/correctness/dispatch/missing baseline 全阻断；
6. fallback 不进入 specialized_hit_speedup；
7. bool、零、负数、NaN、Inf、字符串时间拒绝；
8. golden-only 与空输入都不是性能证据。

P3 起点基线在首个性能模型通过 17 节点后记录，未声称它早于首个 P3 文件：71 个合同面文件、
2128 个完整工作树文件、HEAD `4a93ed2a4934fe526eedfcf4577de5dbd366e24a`。
冻结的 `evidence/performance-policy.json` 固定 1.10、ratio 方向、两本账聚合、baseline 选择与 fresh timing 要求；
它不是生成的性能结果。
