# Harness 收益计分卡 H0：数据盘点与口径冻结

日期：2026-11-21。计划：[Harness 收益计分卡实施计划](harness收益计分卡-plan.md)。

## 父任务与边界

父任务是判断 harness 改动能否跨任务、跨 backend 带来可复核收益。本阶段只改变**数据质量与统计口径**：只读盘点现有 `runs/`，不实现 H1 四张计分表，不分析 H2 效果，不调用模型、不跑 GPU，也不启动 H3。

## 可复现入口

```bash
uv run python -m taic.cli harness audit-scorecard-inventory
uv run python -m taic.cli harness audit-scorecard-inventory --json
```

实现：`taic/scorecard_contract.py`、`taic/scorecard_inventory.py`。测试：`tests/shared/test_scorecard_inventory.py`。

## 当次 inventory

完整快照：[harness-scorecard-inventory-2026-11-21.json](harness-scorecard-inventory-2026-11-21.json)。

| 维度 | 实测数量 |
| --- | ---: |
| 合法 run root | 148 |
| finalized（仅表示 result 落盘） | 104 |
| incomplete_or_aborted | 44 |
| dry_run / unclassified | 0 / 0 |
| natural / controlled / recovery / unknown | 66 / 8 / 0 / 74 |
| real / simulated / unavailable 证据声明 | 63 / 0 / 85 |
| 实验父容器（不计入 run） | 3 |

字段覆盖：protocol 91、system prompt 120、spent_usd 143、stop_reason 104、confirmed_status 90。它们是覆盖计数，不是成功率；缺失没有补 0。

三个实验父容器只负责归组，子 cell 才计 run。历史 `agent_work/` 中的 overlay 文件不再制造 dry-run。自定义且没有显式实验记录的目录保持 `unknown`，不按名字冒充自然搜索或恢复运行。旧 plan 缺 `experiment_id` 时保留 null，另用 plan 字节 SHA-256 作 `experiment_fingerprint`。

当前没有符合 run-root 合同的 dry-run，也没有 NPU 自然 run；NPU 设备指标为 `unavailable`，不从模拟 fixture 补行。

## 冻结口径

- run root 只认根级运行/准备标记；排除 `agent_work/`、`sessions/`、`candidates/`、`.selfcheck/` 与 `__pycache__`。
- `finalized` 不等于成功；`stop_reason`、候选状态和确认状态独立保存。
- `cohort=natural|controlled|recovery|unknown`；`experiment_kind=paid|smoke|paid_smoke|unknown`。paid/smoke 只由显式记录证明。
- run 成本与模型成本分开；`ledger_cost_per_ok` 不是精确成本，`cost_may_be_incomplete=true` 时 `exact_cost_per_ok=null`。
- 六元身份只有全字段一致才叫 full match；部分一致是 partial，冲突是 mismatch。
- 会话 retry 映射回逻辑标签，原始标签保留；feedback 与外层报告冲突时外层优先且记录冲突。
- run/session/claim/delivery-receipt/paired-conclusion 的最低独立已知单位均为 5，字段覆盖率至少 0.8；跨任务至少 2 个 task，跨真实 backend 至少 2 个 backend。静态合同省略仍必须为 0。

## 测试与下一步

H0 聚焦测试：**19 passed**。真实 inventory 的只读不变量也在测试中：执行前后 JSON 文件 mtime 不变，自然 run 必须符合标准生成布局，H1 标记仍为 false。

下一步是 H1：按这些冻结口径实现四张只读计分表。H0 没有产生任何 harness 收益或加速结论，`publication_allowed=false`。
