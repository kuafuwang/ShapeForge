# 可信 harness 试点

这份文档是试点协议，不是自动执行器。它不发模型请求、不启动 GPU；`taic run`/`resume` 会产生费用，只有明确批准后才跑。

## 前置

1. 先完成 `experiments.apply_feedback_audit`，让存量 174 条有审查标签。
2. 在目标硬件的 `targets/<target_id>/trust.toml` 里，由人给两个委员会的两个不同 provider/model 授予 `overlay:tip`。信任绑定 task、effort、prompt sha 和协议版本。
3. 运行配置：`harness_auto = true`、`harness_shadow = true`。影子期只写 `harness_overlay.jsonl`，不注入 TASK。`premise` 只有在本场探针、active `overlay:premise` 信任和匹配的 harness 自检记录或外层候选报告同时存在时才可能 active；否则只写 shadow。

## 试点范围

- 2–3 道题：一道已经产生过反馈的题，一道 held-out（未参与规则设计）。每题约 5 个 run，两个委员会都出现。
- 记录：空占位、结构化提案、双来源一致数、shadow/active 条目、人工否决数、unchanged 率、每 run 成本。
- 人复核每一条 shadow 条目；只有确认没有把合同/命令/未验证事实带进 TASK 后，才用 `--no-harness-shadow` 跑 active。

## 离线汇总

```bash
uv run python -m experiments.harness_pilot \
  runs/apple-m4-max/RUN_A runs/apple-m4-max/RUN_B \
  --out /tmp/pilot.json --markdown /tmp/pilot.md
```

或看单场：

```bash
uv run python -m taic.cli harness status runs/apple-m4-max/RUN
```

## 停止条件

- 任一条 active 条目被人判为错误或有害：`uv run python -m taic.cli harness disable RUN --by 人名 --reason ...`。
- 同一模型在该 target/task 上出现确认推翻率越线，`sync_trust_status` 会暂停它的投票权；人仍可随时 revoke。
- 不要把试点结论复制到另一个硬件；信任、overlay 和 P2 补丁都按 target 分开。

## 阶段 A 的当前门槛

阶段 A 只做离线加固，不跑付费搜索：

- `premise` 需要本场探针、active `overlay:premise` 信任和匹配的 harness 自检记录或外层候选报告；没有记录时只写 shadow。
- 空占位已有索引墓碑；7 条审计项均已核对，当前所有空占位小节共 18 条墓碑。
- `tests/test_harness_workflow.py` 覆盖 tip 从双来源/信任到 active TASK，以及 disable 后消失；`premise` 无自检记录时保持 shadow。
- 自检记录是本机记录，不是密码学签名；阶段 B 仍需人工复核，不能把 `premise` 当形式证明。

阶段 B 才考虑 2–3 道题的 shadow/active 运行，并需要单独的预算与 GPU 批准。
