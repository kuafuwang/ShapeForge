# 可信 harness 反馈实现交接

这份文件给下一个审阅 agent：列出提交、验证过的行为、已知边界和检查清单。实现没有调用外部 LLM，也没有跑 GPU 内核；试点只准备了离线工具。

## 提交

| commit | 内容 |
| --- | --- |
| `90b7517` | fbv3 空占位：所有小节的缺席句不再生成 claim，7 条审计原文做夹具 |
| `7ae00d5` | `info`/`obsolete` 审查状态、`feedback_claim_removed` 墓碑、174 条审计写回工具 |
| `397d806` | 按 target/task 的信任登记、prompt/协议绑定、审计行、Wilson 指标 |
| `9c79d61` | 结构化提案、双来源 quorum、影子优先 overlay、`HARNESS.md`、P2 patch、CLI/API、文档 |
| `4b433ff` | 系统建议率与前提实测率分开；P2 apply 可选跑离线测试 |
| `887c80f` | 空占位率；两个模型必须使用不同 SHA 的探针 |
| `bb7c51f` | 索引记录 `placeholder_items`，旧记录在 repair 时补计数 |
| `1c1eac6` | 本交接文件 |
| `158db49` | 拒绝带 shell/代码语法的 `action` |
| `caf16f1` | 收紧 never-auto：合同、容差、ABI、预算、价格、阈值的词不进 P1 |
| `e05f662` | 把 ABI 改为词边界匹配，避免 `capability` 被误伤 |
| `4738a94` | P2 patch 前校验 target id |
| `4cade5f` | 看板顶栏 Harness 按钮，打开当前 run 的只读 JSON |

基础是 `c9f7862`（数据层文档 + 反馈审计）。

## 已实现的行为

- `taic/agent_feedback.py`：`fbv3` 在所有小节和逐条两处识别缺席句；带动作请求的条目保留。`_repair_record` 为旧记录补 `placeholder_items`。
- `observe/feedback.py`：状态加入 `info` 和 `obsolete`；重建时读 `FEEDBACK.md` 只用于把探针已恢复的降级前提升为 `verified`，保留旧 digest 和审查键；删除的 claim 墓碑保留 `status`/`reason`。
- `taic/trust.py`：`targets/<id>/trust.toml`，键含 target/task/provider/model/effort、prompt sha、协议、scope、过期；`trust_audit.jsonl` 追加 grant/suspend/revoke。
- `taic/harness.py`：解析 ` ```harness-proposal ` JSON；`tip` 需要两个不同 provider/model、两个委员会、两份不同 SHA 的本场探针和 active 信任；`premise` 只需本场探针。默认只写影子 overlay；`TASK.md` 才注入 active 条目，且带未验证标题、受 12KB 预算裁剪。
- `taic/cli.py`：`trust grant|revoke|suspend|status|audit|metrics`，`harness status|disable|settle|patch|apply`。`apply` 默认拒绝；`--check` 只检查，`--run-tests` 跑反馈/合同离线测试，`--confirm` 才改 `targets/<id>/experience.md`，不自动 commit。
- `observe/query.py` / `observe/app.py`：`GET /api/v2/harness/{run_id}` 返回提案、决策和 overlay；顶栏 Harness 按钮打开当前 run 的只读 JSON，没有编辑面板。
- `experiments/apply_feedback_audit.py`：dry-run 默认；`--apply` 写 `feedback_review.json`；`--merge-topic` 显式合并同 topic 的系统建议（默认不做）。
- `experiments/harness_pilot.py`：只读汇总影子/生效、空占位、提案和 unchanged 率。

## 验证

- 最近全量离线测试：**884 passed, 5 skipped, 133 deselected**（`uv run pytest -q -m 'not gpu and not llm'`）。
- 新增测试覆盖：7 条占位原文、探针恢复后的 digest 保留、`info`/`obsolete`、信任 scope/prompt/协议/过期、单模型/同探针不达 quorum、跨 target/协议隔离、P2 `git apply --check`、审计写回、试点汇总。
- 数据侧：174 条审计已写回；60 条实测 `adopted`、15 条已修 `done`、35 条未验证 `info`、空占位/同主题重复 `obsolete`，其余 pending。39 条采集时缺探针的 claim 中，当前索引有 31 条 premise 升为 `verified`、8 条仍 `downgraded`，但审查状态仍为 pending，等待人复核。
- `targets/apple-m4-max/trust.toml` 为空；没有 active 信任，因此即使 `harness_auto` 打开也不会自动通过任何 tip。

## 已知边界（请重点审）

1. **没有跑付费试点。** `harness_pilot.py` 和文档已就绪，但 `taic run`/`resume` 未执行；不要把它们当成已验证的质量结论。
2. **P1 `premise` 不要求双模型**，只要求本场探针；这是“证据流水线”和“harness 流水线”分开的代价。若认为实测前提也必须投票，需要改 `settle`。
3. **“独立”只到探针 SHA 不同。** 两个模型若各自写一份相同内容的探针，SHA 相同会被拒绝；但复制文字后各写不同探针仍可能通过。没有做语义抄袭检测。
4. **7 条空占位**已经从 claim 表消失；审查文件保留了 `obsolete`，但当前索引的 `feedback_claim_removed` 没有这 7 行（数据库在 fbv3 后重建，墓碑表只记录重建时仍存在的旧 claim）。审计 JSON 和 review 文件仍是证据。
5. **同主题合并是显式选择。** 实际执行用了 `--merge-topic`，因此 10 条 active 系统建议被标为 duplicate；fixed 的重复仍保留 `done`。topic 不是命题等价，审阅时要看 note 的 `duplicate of`。
6. **看板只有 API，没有按钮**；`taic harness status` 是当前的查看入口。
7. **`sync_trust_status` 自动暂停**在 `harness_auto` 打开时每次 session 后调用，读取 observe 索引；若索引未追平，暂停判断会滞后。它只按 confirmed refuted 率，样本门槛 8 条 / 2 个 run。
8. **`HARNESS.md` 代替了预建 FEEDBACK 空模板**：预建五节会制造新的空占位；合同仍要求只有有内容才写小节。

## 审阅清单

```bash
git show --stat 90b7517 7ae00d5 397d806 9c79d61 4b433ff 887c80f bb7c51f 1c1eac6 158db49 caf16f1 e05f662 4738a94 4cade5f
uv run pytest -q -m 'not gpu and not llm'
uv run python -m taic.cli trust status apple-m4-max
uv run python -m taic.cli harness status RUN_DIR
git status --short   # 只应有一个无关的 docs/target_profiles_*.plan.md
```

重点看：`taic/harness.py` 的 quorum/覆盖条件、`observe/feedback.py` 的 digest 保留与墓碑、`taic/trust.py` 的 TOML 写出、`taic/orchestrator.py` 的 `_record_feedback` 钩子，以及 `experiments/apply_feedback_audit.py` 的 `--merge-topic` 是否应默认关闭。
