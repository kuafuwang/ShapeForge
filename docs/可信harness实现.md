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

- `taic/agent_feedback.py`：`fbv4` 在所有小节和逐条两处识别缺席句；带动作请求的条目保留。`_repair_record` 为旧记录补 `placeholder_items`，并先按旧 digest 从 `sections` 恢复被擦除的实测正文；不能按 item 位置与 section 片段直接 zip。
- `observe/feedback.py`：状态加入 `info` 和 `obsolete`；重建时读 `FEEDBACK.md` 只用于把探针已恢复的降级前提升为 `verified`，保留旧 digest 和审查键；删除的 claim 墓碑保留 `status`/`reason`。多命题前提支持 `subclaims`：父 claim 保持 `info`/`pending`，子命题各自校验并显示状态；旧审查文件无需迁移。
- `taic/trust.py`：`targets/<id>/trust.toml`，键含 target/task/provider/model/effort、`taic/prompts.py`/`taic/kernel_agent.py`/`taic/harness.py` 三个文件的整体 sha256、协议、scope、过期；`trust_audit.jsonl` 追加 `grant`/`suspended`/`revoked`。
- `taic/harness.py`：解析 ` ```harness-proposal ` JSON；新增 `free_text_suggestions` / `convert_feedback`，把人工确认的自由文本写成带 `origin=conversion`、`claim_digest`、`confirmed_by`、`source_author`、`structured_by` 的提案；转换使用记录里的 `session_started` 判断探针时间，旧记录缺该字段时只核对路径/哈希并标记时间无法验证；转换行顶层 provider/model 为空，不冒充模型作者；`tip` 需要两个不同 provider/model、两个委员会、两份不同 SHA 的本场探针和 active 信任；`premise` 需要本场探针、active `overlay:premise` 信任，以及匹配的 harness 自检记录或外层候选报告（同 target/协议/内核 SHA，测量数字一致）；没有匹配记录时不写 overlay，只保留 `trusted_record=false` 的提案。默认只写影子 overlay；`TASK.md` 才注入 active 条目，且带未验证标题；overlay 自身最多 1800 字符。
- `taic/review.py` + `taic/flows/feedback_review`：独立只读 review；输入是有界证据包（原始自由文本、scope、探针节选、self-check、候选报告、已验证/缺失），外层检查与提案正文分开标注；预算按最终 prompt 和模型单价算；写 `harness_reviews.jsonl` 与 `prompt_sha256`；useful review 仅在提案为 tip、探针本场时间已核验、reviewer 有该 `(target, task)` 的 active `overlay:tip` 信任且独立于同 key 的所有作者时产生 `reviewed_active`，只对本 run 后续场次生效，运行结束失效。
- `taic/cli.py`：`trust grant|revoke|suspend|status|audit|metrics`，`harness status|review|disable|settle|patch|apply|commit|implement|apply-code`。`apply` 默认拒绝；`--check` 只检查，`--run-tests` 跑离线测试，`--by` 记录确认人；`apply` 只改本地 checkout（`status=local`，不写最终 `change_ref`），`commit --message` 才提交这些路径并回填 `commit:<sha>`。`implement` 让 code agent 在 worktree 生成代码 patch；`apply-code` 默认先在另一个隔离 worktree 应用 patch 并跑离线套件，再改主 checkout，且拒绝 tasks/trust/control/凭据路径；它记录 `code_promotion` 与 `paths`，`commit` 会同时提交 `promotion` 和 `code_promotion`，提交前校验补丁哈希，在独立 Git index 中从 HEAD 顺序重放并比较暂存快照。
- `observe/query.py` / `observe/app.py`：`GET /api/v2/harness/{run_id}` 返回提案、自由文本建议、决策和 overlay；顶栏「结构化提案」按钮在看板内列出这三样，并对 review 有用的提案给出「固化到本地」按钮（走 `POST /api/runs/{id}/harness/{key}/promote`）；没有内容编辑框。看板这条路径直接本地 apply，不跑离线测试或 prompt golden；要跑测试用 CLI `harness apply --run-tests`。
- `experiments/apply_feedback_audit.py`：dry-run 默认；`--apply` 写 `feedback_review.json`；`--merge-topic` 显式合并同 topic 的系统建议（默认不做）。
- `experiments/harness_pilot.py`：只读汇总影子/生效、空占位、提案和 unchanged 率。

## 验证

- 最近全量离线测试：**945 passed, 1 skipped, 133 deselected**（`uv run pytest -q -m 'not gpu and not llm'`，本轮修改后）。
- 新增测试覆盖：7 条占位原文、探针恢复后的 digest 保留、`info`/`obsolete`、空占位墓碑、信任 scope/prompt/协议/过期、单模型/同探针不达 quorum、跨 target/协议隔离、P2 `git apply --check`、审计写回、试点汇总、 review 预算随价格变化与 verdict 解析，以及合成的 feedback→proposal→trust→overlay→TASK 端到端流程。
- 数据侧：174 条审计已写回；60 条实测 `adopted`、15 条已修 `done`、35 条未验证 `info`、7 条空占位 `obsolete`，其余 57 条 pending。10 条按 topic 合并的 `obsolete` 已由 `experiments.revert_topic_merges` 改回 pending；39 条采集时缺探针的 claim 中，31 条 premise 升为 `verified`、8 条仍 `downgraded`，等待人复核。7 条审计空占位现在都有 `feedback_claim_removed` 墓碑；重建会为所有空占位小节写墓碑，当前共 18 条。
- `targets/apple-m4-max/trust.toml` 的两个 conv2d grant（deepseek-flash、kimi-k3）status 仍是 `active`，但记录的 `prompt_sha1` 为 `c270d5d3750e…`；本次修改后当前指纹为 `3236e0adce76…`。离线 `trust.check` 对两者均返回 `trust was granted for another prompt`。再次启用 active overlay 前须复核并重新授予信任；本轮不修改信任文件。

## 审阅响应

外部审阅的总评是 **PASS-WITH-CHANGES**（审阅文件已按用户要求删除，结论保留在此）。已处理：

- `premise` 现在除本场探针外，还要求 active `overlay:premise` 信任；空信任表不会自动放行（`taic/harness.py` + `tests/test_harness.py`）。
- 交接第 6 条已改成“顶栏有只读的面板，没有编辑面板”。
- 10 条按 topic 标为 `obsolete` 的系统建议已用 `experiments.revert_topic_merges` 改回 `pending`，备注改为“possible same topic”；7 条空占位仍保留 `obsolete`。
- 仍保留的已知差距：`self_checks.jsonl` 是本机记录，不是密码学签名；还没有外层 `report.json` 的签名链。

## 阶段 A 的收紧与可信记录

- `self_checks.jsonl` 现在记录 `target_id`、`protocol_version`、内核 SHA、任务信息；这些行由评测命令追加，不是提案者自报。
- `premise` 只有同时满足三件事才可能 active：本场探针、active `overlay:premise` 信任、匹配的 `status=ok` 自检行或外层候选报告（同 target/协议/内核 SHA，且 `evidence.measurement` 的数字与 `candidate_us` 或 `speedup` 一致）。没有匹配记录时不写 overlay，只保留 `trusted_record=false` 的提案。
- 7 条审计空占位现在都有 `feedback_claim_removed` 墓碑；重建会为所有空占位小节写墓碑，当前 18 条。
- 新增 `tests/test_harness_workflow.py`，验证 tip 从两个模型/信任/active overlay 进入 TASK，以及 disable 后消失；`premise` 在无自检记录时即使有信任也不进入 active。
- 仍不等同于密码学签名：自检文件在本机磁盘上，恶意代理理论上仍可伪造。阶段 B 的付费试点仍应保持人工复核。

## 试点结果

试点报告见 [可信harness试点结果.md](可信harness试点结果.md)：第一次 4 场、$2.6658、0 提案；第二次 $0.7584、1 条 tip、kimi review $0.106；第三次 $0.7996、自动 review useful、`reviewed_active` 生成，但 run 在下一场前停掉。

TASK 的 12KB 预算为 active overlay 预留空间；overlay 自身最多 1800 字符，超出的条目不进 TASK，TASK 不逐条记录被裁掉的条目。没有预留时，满成绩单会把 overlay 静默裁掉。

## 独立 review

- **不是 quorum**：review 仅把探针本场时间已核验的单来源 tip 推到 `reviewed_active`，只对**本 run 的后续场次**生效；运行结束失效，不是永久固化。reviewer 自己还必须在该 `(target, task)` 上有 active `overlay:tip` 信任，且独立于同 key 所有作者；否则 review 只存档。
- **只读且可核**：review 不写 kernel/TASK，只写 `HARNESS_REVIEW.json`；prompt 把建议标成 untrusted data，并附外层计算的有界证据（探针存在/哈希/时间、self-check、候选报告、缺失项）。旧反馈记录缺 `session_started`、或旧 proposal 行缺 `probe_time_verified` 时，明确显示“无法验证”。
- **动态预算**：按 reviewer 的 input/output 单价、prompt 长度、最大输出 token，以及 pi/skill 上下文的 cache read/write allowance 计算；不写死 $0.2。实测 kimi review：旧预算 $0.252、实际 $0.106；加入缓存 allowance 后同一输入预算约 $0.588，实际仍只按用量计费。
- **实测**：deepseek 寄存器报告 tip → kimi review `useful=true, confidence=0.6`；当时的代码记录为旧名 `reviewed_shadow`，新代码会对同一 review 产生 `reviewed_active`（仅本 run）。

## 已知边界（请重点审）

- **code-agent P2 已提供 worktree worker，但仍不是自动合并。** `taic/code_agent.py` 在 detached worktree 里运行 code agent、收集 patch；`taic harness implement` 生成 patch，`taic harness apply-code` 由人确认后 apply，默认在隔离 worktree 里确认导入该 worktree 的代码并跑离线套件。AI 是默认实现者，但不能继承 overlay 信任或自动 commit/merge。
- **本地固化与 git 提交分离。** 看板固化改本地 checkout（`local_patch_sha256`，无最终 `change_ref`）；`taic harness commit` 只提交被 promote 的路径，覆盖 `promotion` 与 `code_promotion`，提交前校验补丁哈希，在独立 Git index 中从 HEAD 顺序重放并比较暂存快照，拒绝漂移，成功后才写 `commit:<sha>`。

1. **没有跑付费试点。** `harness_pilot.py` 和文档已就绪，但 `taic run`/`resume` 未执行；不要把它们当成已验证的质量结论。
2. **P1 `premise` 不要求双模型**，但要求本场探针、active `overlay:premise` 信任和匹配的 harness 自检记录或外层候选报告；没有匹配记录时不写 overlay，只保留提案。
3. **“独立”只到探针 SHA 不同。** 两个模型若各自写一份相同内容的探针，SHA 相同会被拒绝；但复制文字后各写不同探针仍可能通过。没有做语义抄袭检测。
4. **空占位墓碑**：7 条审计里的空占位已经从 claim 表消失，审查文件保留 `obsolete`，索引现在也有对应墓碑。重建时会把所有空占位小节写入 `feedback_claim_removed`，当前共 18 条（7 条审计项 + 11 条其他空小节），reason 为 `fbv4 empty placeholder`。
5. **同主题合并没有保留。** 曾用 `--merge-topic` 把 10 条 active 系统建议标为 duplicate；复核后用 `experiments.revert_topic_merges` 改回 pending，只保留「possible same topic」备注。topic 不是命题等价。
6. **看板已有本地固化按钮，不是 git 提交器**；顶栏「结构化提案」按钮打开当前 run 的面板，节点来自 `GET /api/v2/harness/<run>`。review 有用后点「固化到本地」立即改本地 checkout；`taic harness commit` 才写 git 和 `change_ref`。信任授予、settle、代码 patch 仍走 CLI/人。
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
