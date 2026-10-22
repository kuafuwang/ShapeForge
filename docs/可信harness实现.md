# 可信 harness 反馈实现交接

这份文件给下一个审阅 agent：列出提交、验证过的行为、已知边界和检查清单。理论、RSI 架构和流程图见 [架构/可信harness.md](架构/可信harness.md)。实现没有调用外部 LLM，也没有跑 GPU 内核；试点只准备了离线工具。

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

基础是 `c9f7862`（数据层文档 + 反馈审计）。完整提交列表用 `git log --oneline c9f7862..HEAD`；上表只列有代码/行为变化的提交，另有几次只改本交接文件。

## 已实现的行为

- `taic/agent_feedback.py`：`fbv3` 在所有小节和逐条两处识别缺席句；带动作请求的条目保留。`_repair_record` 为旧记录补 `placeholder_items`。
- `observe/feedback.py`：状态加入 `info` 和 `obsolete`；重建时读 `FEEDBACK.md` 只用于把探针已恢复的降级前提升为 `verified`，保留旧 digest 和审查键；删除的 claim 墓碑保留 `status`/`reason`。
- `taic/trust.py`：`targets/<id>/trust.toml`，键含 target/task/provider/model/effort、prompt sha、协议、scope、过期；`trust_audit.jsonl` 追加 grant/suspend/revoke。
- `taic/harness.py`：解析 ` ```harness-proposal ` JSON；`tip` 需要两个不同 provider/model、两个委员会、两份不同 SHA 的本场探针和 active 信任；`premise` 需要本场探针、active `overlay:premise` 信任，以及匹配的 harness 自检记录或外层候选报告（同 target/协议/内核 SHA，测量数字一致）；否则只写影子。默认只写影子 overlay；`TASK.md` 才注入 active 条目，且带未验证标题、受 12KB 预算裁剪。
- `taic/review.py` + `taic/flows/feedback_review`：独立只读 review；预算按模型单价和 prompt 长度算；写 `harness_reviews.jsonl`；useful review 产生 `reviewed_active`，只对本 run 后续场次生效，运行结束失效。
- `taic/cli.py`：`trust grant|revoke|suspend|status|audit|metrics`，`harness status|review|disable|settle|patch|apply`。`apply` 默认拒绝；`--check` 只检查，`--run-tests` 跑反馈/合同离线测试，`--confirm` 才改 `targets/<id>/experience.md`，不自动 commit。
- `observe/query.py` / `observe/app.py`：`GET /api/v2/harness/{run_id}` 返回提案、决策和 overlay；顶栏「结构化提案」按钮在看板内直接列表显示这三样（只读），原始 JSON 在面板底部留了链接；没有编辑面板。
- `experiments/apply_feedback_audit.py`：dry-run 默认；`--apply` 写 `feedback_review.json`；`--merge-topic` 显式合并同 topic 的系统建议（默认不做）。
- `experiments/harness_pilot.py`：只读汇总影子/生效、空占位、提案和 unchanged 率。

## 验证

- 最近全量离线测试：**896 passed, 5 skipped, 133 deselected**（`uv run pytest -q -m 'not gpu and not llm'`）。
- 新增测试覆盖：7 条占位原文、探针恢复后的 digest 保留、`info`/`obsolete`、空占位墓碑、信任 scope/prompt/协议/过期、单模型/同探针不达 quorum、跨 target/协议隔离、P2 `git apply --check`、审计写回、试点汇总、 review 预算随价格变化与 verdict 解析，以及合成的 feedback→proposal→trust→overlay→TASK 端到端流程。
- 数据侧：174 条审计已写回；60 条实测 `adopted`、15 条已修 `done`、35 条未验证 `info`、7 条空占位 `obsolete`，其余 57 条 pending。10 条按 topic 合并的 `obsolete` 已由 `experiments.revert_topic_merges` 改回 pending；39 条采集时缺探针的 claim 中，31 条 premise 升为 `verified`、8 条仍 `downgraded`，等待人复核。7 条审计空占位现在都有 `feedback_claim_removed` 墓碑；重建会为所有空占位小节写墓碑，当前共 18 条。
- `targets/apple-m4-max/trust.toml` 为空；没有 active 信任，因此即使 `harness_auto` 打开也不会自动通过任何 tip。

## 审阅响应

外部审阅的总评是 **PASS-WITH-CHANGES**（审阅文件已按用户要求删除，结论保留在此）。已处理：

- `premise` 现在除本场探针外，还要求 active `overlay:premise` 信任；空信任表不会自动放行（`taic/harness.py` + `tests/test_harness.py`）。
- 交接第 6 条已改成“顶栏有只读的面板，没有编辑面板”。
- 10 条按 topic 标为 `obsolete` 的系统建议已用 `experiments.revert_topic_merges` 改回 `pending`，备注改为“possible same topic”；7 条空占位仍保留 `obsolete`。
- 仍保留的已知差距：`self_checks.jsonl` 是本机记录，不是密码学签名；还没有外层 `report.json` 的签名链。

## 阶段 A 的收紧与可信记录

- `self_checks.jsonl` 现在记录 `target_id`、`protocol_version`、内核 SHA、任务信息；这些行由评测命令追加，不是提案者自报。
- `premise` 只有同时满足三件事才可能 active：本场探针、active `overlay:premise` 信任、匹配的 `status=ok` 自检行或外层候选报告（同 target/协议/内核 SHA，且 `evidence.measurement` 的数字与 `candidate_us` 或 `speedup` 一致）。否则只写 shadow。
- 7 条审计空占位现在都有 `feedback_claim_removed` 墓碑；重建会为所有空占位小节写墓碑，当前 18 条。
- 新增 `tests/test_harness_workflow.py`，验证 tip 从两个模型/信任/active overlay 进入 TASK，以及 disable 后消失；`premise` 在无自检记录时即使有信任也不进入 active。
- 仍不等同于密码学签名：自检文件在本机磁盘上，恶意代理理论上仍可伪造。阶段 B 的付费试点仍应保持人工复核。

## 试点结果

试点报告见 [可信harness试点结果.md](可信harness试点结果.md)：第一次 4 场、$2.6658、0 提案；第二次 $0.7584、1 条 tip、kimi review $0.106；第三次 $0.7996、自动 review useful、`reviewed_active` 生成，但 run 在下一场前停掉。\n\nTASK 的 12KB 预算现在为 active overlay 预留空间；没有这个预留，满成绩单会把 overlay 静默裁掉。

## 独立 review

- **不是 quorum**：review 把单来源 tip 推到 `reviewed_active`，只对**本 run 的后续场次**生效；运行结束失效，不是永久固化。
- **只读**：review 不写 kernel/TASK，只写 `HARNESS_REVIEW.json`；prompt 把建议标成 untrusted data。
- **动态预算**：按 reviewer 的 input/output 单价、prompt 长度、最大输出 token，以及 pi/skill 上下文的 cache read/write allowance 计算；不写死 $0.2。实测 kimi review：旧预算 $0.252、实际 $0.106；加入缓存 allowance 后同一输入预算约 $0.588，实际仍只按用量计费。
- **实测**：deepseek 寄存器报告 tip → kimi review `useful=true, confidence=0.6`；当时的代码记录为旧名 `reviewed_shadow`，新代码会对同一 review 产生 `reviewed_active`（仅本 run）。

## 已知边界（请重点审）

1. **没有跑付费试点。** `harness_pilot.py` 和文档已就绪，但 `taic run`/`resume` 未执行；不要把它们当成已验证的质量结论。
2. **P1 `premise` 不要求双模型**，但要求本场探针、active `overlay:premise` 信任和匹配的 harness 自检记录或外层候选报告；否则只写 shadow。
3. **“独立”只到探针 SHA 不同。** 两个模型若各自写一份相同内容的探针，SHA 相同会被拒绝；但复制文字后各写不同探针仍可能通过。没有做语义抄袭检测。
4. **空占位墓碑**：7 条审计里的空占位已经从 claim 表消失，审查文件保留 `obsolete`，索引现在也有对应墓碑。重建时会把所有空占位小节写入 `feedback_claim_removed`，当前共 18 条（7 条审计项 + 11 条其他空小节），reason 为 `fbv3 empty placeholder`。
5. **同主题合并没有保留。** 曾用 `--merge-topic` 把 10 条 active 系统建议标为 duplicate；复核后用 `experiments.revert_topic_merges` 改回 pending，只保留「possible same topic」备注。topic 不是命题等价。
6. **看板有只读入口，没有编辑面板**；顶栏「结构化提案」按钮打开当前 run 的面板，节点来自 `GET /api/v2/harness/<run>`（原始 JSON 也留了链接），`taic harness status` 是命令行入口；信任授予、settle、P2 apply 仍需 CLI/人。
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
