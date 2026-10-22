# 可信 harness shadow 试点结果（2026-10-22）

## 运行

- run：`runs/apple-m4-max/harness-pilot-shadow-20261022`
- 配置：`conv2d`，fp16，paper；两个委员会 deepseek-flash / kimi-k3；`harness_auto = true`，`harness_shadow = true`；没有 active overlay。
- 用户授权上限：**$15**。实际执行 4 场后停在 **$2.6658**；为了让“没有结构化提案”的状态不再继续烧钱，把运行预算降到已花费附近，`stop_reason = budget_exhausted`。
- 没有调用额外模型做验证，没有手动改 TASK。

## 结果

| 场次 | 委员会 | 模型 | 搜索加速比 | 费用 | 反馈 |
| --- | --- | --- | ---: | ---: | --- |
| init | first | deepseek-flash | 1.1389 | $0.2548 | 7 条，`kinds=none` |
| r1.p0 | first | deepseek-flash | 0.9791 | $0.7030 | 6 条，`kinds=none` |
| r2.p0 | second | kimi-k3 | 1.0479 | $2.1803 | 3 条，`kinds=none` |
| r3.p0 | first | deepseek-flash | 1.1605（新种子） | $2.6658 | 8 条，`kinds=none` |

确认复测：`r3.p0` 三批中位数 **1.1389x**，确认状态 `ok`，发布状态 `confirmed`。搜索最好值仍是 `1.1605x`，两者不能混写。

## 这次试点发现了什么

- **0 条结构化提案、0 条 overlay、0 条决策**。四场反馈都只有「给下一棒的避坑提示」和「被否决的前提」，没有「对系统的建议」或「需要帮助」，所以模型没有机会写 `harness-proposal` 块。
- 因此这次只验证了**反馈收集、信任登记、shadow 路径未误触发**；没有验证模型真的会使用结构化提案，也没有验证 active overlay 对搜索的影响。
- 空占位为 0；说明 fbv3 的占位过滤在这几场正常。
- 确认复测与搜索值的差距（1.1605 → 1.1389）符合已有测量门槛：搜索期看快，确认期用独立批次。

## 结论

**shadow 基础设施可用，但“模型主动提出结构化 harness 建议”的频率为 0/4。** 这不是 overlay 代码失败的证据；是没有输入可测。下一步不应直接打开 active，而应二选一：

1. 在会产生系统层问题的题目/环境上再跑一次 shadow（例如工具缺失、预算耗尽、契约限制），看模型是否写出「对系统的建议」并附带结构化块；
2. 给 TASK/HARNESS.md 更强的结构化示例和一句“若有系统建议，必须用块”的提示，再做 A/B；这属于 prompt 变更，应单独评估，不要混进本次试点结论。

## 运行命令（可复现）

```bash
uv run python -m taic.cli run conv2d \
  --dtype fp16 --size paper --budget 15 \
  --harness-auto --harness-shadow \
  --run-dir runs/apple-m4-max/harness-pilot-shadow-20261022

uv run python -m experiments.harness_pilot \
  runs/apple-m4-max/harness-pilot-shadow-20261022
```

`targets/apple-m4-max/trust.toml` 已记录两个模型的 `overlay:tip` 和 `overlay:premise` 授权，`trust_audit.jsonl` 有 grant 行；本次没有打开 active，所以授权没有被用于注入 TASK。
