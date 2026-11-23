# softmax ledger / sweep 开关的真实 run 证据

这份记录纠正“这些机制只有 CPU 测试”的过宽说法。两场真实 run 已存在于 `runs/apple-m4-max/`；**本次只读登记，没有新 GPU、付费模型或搜索。** 它们证明相应配置在真实模型与 Metal 搜索中启用并完成了 run，不证明长期收益，也不证明每个开关都实际改变了行为。

## 共同配置

- 任务：softmax / fp16 / paper / `apple-m4-max`；run_seed=7。
- 两场 `run_config.json` 都记录：`plan_ledger=true`、`task_ledger=true`、`plateau_after=2`、`sweep_tool=true`、`diversity_archive=false`。
- 委员会为 DeepSeek `deepseek-flash` 与 `kimi-code/k3.1-pro`。
- 只读检索没有找到独立的 sweep 调用记录，因此这里**不声称实际使用了 sweep**；名称中的 `diversity` 也不能覆盖配置里的 `diversity_archive=false`。

## 结果

| run | 候选 SHA / winner | 实际成本 | search best | fresh confirmation | 停止原因 |
| --- | --- | ---: | ---: | --- | --- |
| `softmax-ledger-verify-20261110` | `ad1968a4ca6f3429def82e7d85d45e21ac82c1bdc5c6ce0b8e7452dcf53f8e5d` / init | $1.8904313679999998 | 0.9670966595172557 | ok / 0.9559082757127974 | max_rounds |
| `softmax-diversity-verify-20261110` | `205c44412be8d6251822fce922fe8f32d6c2edb798be7cd0dd84392f57ec72b6` / init2 | $1.412710488 | 0.9837530533596897 | ok / 0.9677433041219036 | at_roofline |

两场都是单次、特定任务和窗口的机制证据；confirmed 均小于 1，不构成加速发布，也不支持把 task ledger、plateau、sweep 或 diversity 说成收益来源。`publication_allowed=false`。

后续若要回答效果问题，需要单因素 AB/BA、多对独立样本和 fresh confirmation；不能把这些既有 run 与混合多个开关的计划拼成因果证据。G7 所需的真实账本误并/误拆仍应另外检查，不能仅凭开关为 true 就关闭。
