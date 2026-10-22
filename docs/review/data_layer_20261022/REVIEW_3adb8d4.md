# 实现审核：`69bbe49..3adb8d4`

**对照**：`IMPLEMENTATION.md`、`REVIEW.md`
**仓库**：`3adb8d4`（7 个提交：`e670aa1` `1a8ac9a` `109cad1` `004027c` `ac04199` `6f0528b` `3adb8d4`）
**日期**：2026-10-22
**约束**：未跑 `taic run` / `taic resume`，未启动 GPU，未调用 LLM。

## 审核结论

- 总评: **PASS-WITH-CHANGES**
- 离线门禁：`uv run pytest -q -m "not gpu and not llm"` → **843 passed, 5 skipped, 133 deselected**（74.76s）。与执行报告的 843 一致。
- `repro.sh` 退出码 0。和报告不一致的现场数字：`agent_work` **2.6MB / 214 文件**（报告写 0.5MB），`runs/` **1131.3MB / 4233 文件**（报告写 1129.2MB）。差额和「把旧 TASK.md 从 scratch 恢复回来」对得上：报告里的 0.5MB 是恢复之前。`brief` **1479B**、无 `gpu` 键，和报告一致。
- 仍未做的 8 项（10^5 压测、F10、F12、sessions 表、eval 级 protocol、topic 写入、看板页面、`taic query`/MCP）这轮 diff 里没有对应实现，和报告一致。

| 重点 | 结论 | 证据 | 要改的地方 |
|---|---|---|---|
| 计费记全 | PASS，resume 会把 token 列丢掉 | `_priced`（`taic/committees.py:200`）与 `load_config`（`taic/llm.py:252`）同一表达式。`charge`（`llm.py:433-449`）写入 `write` 和 `basis`。现场调用在 `orchestrator.py:1222`：`charge(..., tokens=result.spent, basis=prices.as_dict())`。三份 `cost_recomputed.json` 与 REVIEW.md 逐分一致：FA-2 `$18.20559006`（+$4.872156），feedback `$20.539063128`（+$5.319141），flame `$10.73508804`（+$2.8608）。`result.json` 的 `spent_usd` 未改。 | `taic/resume.py:349` 从 transcript 重建 agent 请求时仍写 `hit=miss=out=0`，不读 `event.tokens`。`load_snapshot` 以 transcript 为准、不用 `ledger.json`（`:484-491`）。续跑之后，已完成会话的 cache-write 计数从账本里消失；美元 `cost_usd` 还在，预算门不会少算钱。 |
| TASK 12KB | PASS，真实题目上成绩单被整块丢掉 | 必需段不裁：`render_contract` 先拼合同、签名、两条命令、证据规则（`kernel_agent.py:314-410`），再按块塞动态段。用真实 `get_task` 渲染、动态段拉满时：conv2d paper 必需段 10,654B，成稿 **12,259B**；flash_attention paper 必需段 10,598B，成稿 **12,194B**。两者的省略清单都是「当前现场、上一轮 NOTES、避坑提示、现场证据」。失败反馈、最近历史、自测对照还在。relu small 必需段 8,571B，当前现场留得下。测试 `test_task_md_keeps_its_dynamic_blocks_inside_the_budget` 允许 `TASK_MAX_BYTES + 400`，用的是十几字节的假合同，看不出这点。 | 块是整块进或整块丢。conv2d / flash 的剩余大约 1.3KB，装不下带 1000 字成绩单的「当前现场」，后面更短的块却能塞进来。外层成绩单因此不在 TASK.md 里。省略注记加在预算检查之后（`:450-453`），所以成稿超过 12,000。旧的 91 份 TASK.md 仍是中位 21,664B、最大 48,612B，那是归档恢复，不是新渲染器的输出。 |
| candidates / keyset | PASS | 表主键 `(run_id, idx)`（`observe/index.py:237-248`）。磁盘 748 个 `report.json`，索引 **741**。差的 7 个全在 `pilot_suite_mini_old_protocol/conv2d_fp16`，这个 run 不在 `runs` 表里。已索引 run 没有重复 `idx`。`abi_error` 索引 130；另外 3 个在那个未索引 run 里。`runs_page` 的 keyset 是 `(排序列, run_id)` 同一方向（`observe/query.py:268-289`）。测试 `test_candidates_are_indexed_and_a_list_page_carries_the_next_cursor` 两页 run_id 不交。`protocol_version` 列在，但现有 52 行全是 `legacy`：回填没有读 `loop_protocol`，4 个 `flame_chase_1` 和 50 个 chat 在这一列上仍是同一桶。 | 满页时总会给 `next`，最后一页可能多一次空请求。不丢行、不重行。 |
| resume 不重开一格 | PASS | `run_one` 在 `run_config.json` + `state.json` + `transcript.jsonl` 都在时走 `_resume_one`（`experiments/common.py:176-186`），`resuming=True`，目录非空不再 `RunStartError`。`load_snapshot` 拒绝非 `flame_chase_1`（`resume.py:584-586`）。已有 `r{n}.p0` 候选的轮次 `next_round = n+1`（`:543-551`）；没有候选的当前轮换一场新会话，不从第 1 轮重评。`taic resume` 的 `snapshot_results` 只在 `cli.py:340`，批量这条路径不搬 `result.json`。有 `result.json` 的格子套件仍不调用 `run_one`。 | 测试 `test_an_interrupted_cell_is_resumed_not_refused` 用 `max_rounds = 0`，证明的是「轮次用尽也能写回 result、且不启动会话」。它没有断言「第 5 轮中断后从第 6 轮继续、已有候选不重评」。那条在 `resume.py` 的 `next_round` 里。三文件还没写出来就崩溃的格子，仍会因为目录非空被拒绝。 |
| transcript 默认不落正文 | PASS | `LLMConfig.keep_transcript_bodies` 直接构造默认 True（`llm.py:114`）；`load_config` 默认 False（`:243`）。`chat()` 把这个开关传进 `chat_transcript_event`（`:854`）。关着时只有 `messages_sha256/bytes` 和 `reply_sha256/bytes`（`:507-510`）。hmz 的 `agent` 事件本来就没有 prompt 正文。`flame_chase_1` 的 `infer_settings` 在有 `run_config.json` 时直接返回（`resume.py:277-284`），续跑不读 `messages`。 | 没有 `run_config.json` 的旧 chat run，若以后写出不带正文的 `chat` 行，`infer_settings` 恢复不出 system prompt（`:273-276`、`:306-320`）。当前协议的续跑和计费不靠这些正文。磁盘上的 759.7MB 旧 transcript 还在，这个开关只约束新的 chat 行。 |

## 和执行报告不一致的测量

- 索引 schema 是 `5`。`turns` **0**，`turn_bytes>0` **0**。v5 重建之后 turn 仍是懒投影，要打开 run 才写入。transcript 文件还在（54 个，759.7MB）。这不是把对话删了。
- `experiments` 2 行，`arms` **13**。`p1/experiment.json` 11 个 arm，`repeat_20261007-015329/experiment.json` 2 个 arm，`frozen` 都是 `{"backfilled": true}`。
- `repro.sh` 第 11 节抽到的旧 ledger 仍没有 `input_tokens_cache_write`。那三份是改代码之前写的；新的 `charge` 只影响之后的运行。历史数字在 `cost_recomputed.json`。

## 建议的后续（仍属这 7 个提交的收尾）

1. `resume._requests_from_transcript` 的 `agent` 行写入 `event.tokens` 的 input/output/cache_read/cache_write，`restore` 才能把 write 加回去。
2. TASK 预算：装不下时先把「当前现场」收到剩余字节，不要整块跳过；省略注记算进 12,000。真实的 conv2d / flash_attention 应用例放进测试，替换现在的短假合同。
3. 索引回填 `protocol_version` 时，已有 `loop_protocol=flame_chase_1` 的 run 不要和 chat 一起标成 `legacy`。

10^5 压测、`candidate_serial`、`data/deleted` TTL、sessions 表、eval 级 protocol、`feedback_claims.topic`、看板页面、`taic query`/MCP 维持报告里的「下一轮」。
