# 数据层修改计划（依据 REVIEW.md）

**审核意见**：`docs/review/data_layer_20261022/REVIEW.md`（PASS-WITH-CHANGES）
**执行版本**：`69bbe49`（审核包抬头 `ace4ac6` 的三个未提交文件已进入该提交）
**约束**：不跑 `taic run`/`resume`、不启动 GPU、不调用 LLM；所有验证用离线测试和只读脚本。数据回填只写 `runs/`（gitignored）和 `data/` 的本地文件，不改历史 `spent_usd`。

## 批次与提交

| # | 主题 | 主要文件 | 验收 |
|---|---|---|---|
| 1 | 计费完整性 + 审核包数字修正 | `taic/committees.py`、`taic/llm.py`、`taic/orchestrator.py`、`taic/kernel_agent.py`、`experiments/reprice_runs.py`、`docs/review/...` | `_priced` 带 `cache_write`；`ledger` 记 write token 和 `cost_basis`；三个 run 写 `cost_recomputed.json`；离线测试全绿 |
| 2 | TASK 预算 + agent_work 上限 + golden 隔离 | `taic/kernel_agent.py`、`taic/agent_feedback.py`、`taic/orchestrator.py`、`tests/hw_golden.py` | 每个 TASK.md ≤12KB（必需段落不可裁）；`v/` 归档；重录 golden 不写仓库 `kernels/` |
| 3 | 索引 v5：candidates + keyset + 请求路径 + 特征缓存 + protocol | `observe/index.py`、`observe/query.py`、`observe/app.py`、`observe/static/index.html`、`taic/artifacts.py` | candidates 入索引；列表 keyset；`history()` 不再每请求扫描；特征按源码 sha 缓存；`protocol_version` 落盘 |
| 4 | 批量幂等 + experiment manifest 回填 | `experiments/common.py`、`experiments/manifest.py`、`experiments/reprice_runs.py` | 非空目录的 interrupted 格自动 resume；p1/repeat 有 `experiment.json`；`arms` 可读 |
| 5 | P1：brief、gzip/ETag、transcript 正文、前端 | `taic/brief.py`、`taic/llm.py`、`observe/app.py`、`observe/static/index.html` | `brief` 无 GPU 噪声/绝对路径、单次读 transcript；JSON 有 gzip/ETag；chat 正文默认不落盘；列表有 URL 状态/stop_reason |
| 6 | 执行报告 + 最终验证 | `docs/review/.../IMPLEMENTATION.md` | 测试、repro、回填结果、commit ID、剩余风险 |

## 审核意见中明确要修正的点

1. **F1/N1/N2**：`_priced` 与 `llm.py:249` 用同一表达式；`add`/`charge` 记 write token 与 `cost_basis`；三个已结束 run 用 `transcript.agent.tokens` 重算，写单独的 `cost_recomputed`，**不改** `spent_usd`（feedback 已按旧值 `budget_exhausted`）。
2. **F8**：NOTES 不是 12.7KB；取样里 NOTES 约 4.6KB、tips 约 8.2KB（两个块合起来才是 12.7KB 级别的跨度）。TIP 后段会缩短，瘦身按「有上限的摘要」写，不假设单调增长。
3. **F11**：审查页已按 `feedback_revision` 增量、单卡替换；P1 只做**主列表**、gzip/ETag，不重排审查页。
4. **F4/F6**：25s/10.7s 仍标 U；本轮先做结构和 `/tmp` 合成压测脚本，未测到的秒数不写进验收。
5. **F14**：kimi 在 60 场里 28 场、记录金额约 80%；`runs.model` 仍是第一委员会，另加 `committees_json`/会话级字段，不把 `model` 当全会话。
6. **F5**：`status != 'missing'` 使复合索引失效的说法过了一步；schema 里还没有那个复合索引。做 candidates 表和缺失的列索引即可。
7. **§8 文件数**：`398` 不可复现；以 `repro.sh` 为准（全集中位 40、agent_work-run 中位 29、当前大 run 17–32/场）。

## 验收指标（本轮）

- `brief` 输出 ≤1.5KB（不含 `--gpu`），不读 transcript；路径相对化。
- TASK.md 每场 ≤12KB；必需段落（合同/签名/命令）不裁；裁掉的块留下路径。
- `run_detail` 不再每次重算 `feature_grid`（按源码 sha 缓存）。
- `runs_page` 支持 keyset `next`、`sort`、`publication`、`min_speedup`，SQL 有对应索引。
- 批量 interrupted 格可以 resume，不因非空目录放弃整批。
- `p1`/`repeat` 有 `experiment.json`，`arms > 0`。
- 计费记录：`ledger` 有 write token 和 `cost_basis`；三个 run 的 `cost_recomputed.json` 与审核重算一致。
- 离线门禁：`uv run pytest -q -m "not gpu and not llm"` 全绿；`repro.sh` exit 0。

## 明确不在本轮

- 不改历史 `result.spent_usd`；不回写旧 `ledger.json`。
- 不做 10^5 真实目录压测（只提供 `/tmp` 合成脚本与 10^3 结果）。
- 不做全量内容寻址迁移（3.5% 收益）；只做 `v/` 归档和后续写入上限。
- 不改确认批次/搜索策略；unchanged_patience/委员会性价比等 S2 结论等试点。
- 不重排审查页；只做主列表、gzip/ETag、brief。

## 数据回填（只本地，不进 commit）

- `runs/apple-m4-max/{flash_attention-20261022-2,conv2d-feedback-20261021,conv2d-flame-20261020}/cost_recomputed.json`
- `runs/apple-m4-max/{p1,repeat_20261007-015329}/experiment.json`
- 现有 `agent_work/*/v/` 归档到 `runs/<run>/scratch/<label>.tar.gz`（保留 NOTES/FEEDBACK/self_checks/submission）
- 删除 `kernels/apple-m4-max/{relu,gemm,flash_attention}` 测试夹具（`repro.sh` 已证明是 golden 录制污染）
