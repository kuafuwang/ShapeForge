# 数据层修改执行报告

**依据**：`docs/review/data_layer_20261022/REVIEW.md`（PASS-WITH-CHANGES）
**起点**：`69bbe49`
**约束**：未跑 `taic run` / `taic resume`，未启动 GPU kernel，未调用 LLM。验证只用离线测试、`repro.sh` 和本地回填。

## 提交

| commit | 主题 |
|---|---|
| `e670aa1` | Charge the second committee its own cache-write price, and record session tokens. |
| `1a8ac9a` | Keep a session's task file inside a byte budget, and archive its scratch. |
| `109cad1` | Index candidate facts and page runs by keyset instead of scanning every request. |
| `004027c` | Resume an interrupted batch cell, and backfill experiment manifests. |
| `ac04199` | Keep prompt bodies out of the transcript, and give the dashboard ETag, gzip and cursor pages. |
| `6f0528b` | Keep eight peer tips; the character budget is the stop, not a shorter list. |
| （本报告） | Execution report and final verification. |

## 按审核意见的落实

### 1. 计费（F1 + N1/N2）— `e670aa1`

- `taic/committees.py:_priced` 补上 `price_cache_write`，表达式与 `taic/llm.py:249` 相同。
- `CostLedger.add` 保留 `write` token；`charge(usd, label, tokens=..., basis=...)` 记录 hmz 的 token 分解和当时的价表；`summary()` 增加 `input_tokens_cache_write` / `cost_basis`；`restore()` 恢复 write token。
- `Orchestrator` 把 `result.spent` 和 `prices.as_dict()` 传给 `charge`。
- 新增 `experiments/reprice_runs.py`：从 `transcript.jsonl` 的 `agent.tokens` 重算，写 `cost_recomputed.json`，**不改 `spent_usd`**。
- 三个 run 的结果与二次审核一致：

| run | 记录 | 重算 | 差额 |
|---|---:|---:|---:|
| `flash_attention-20261022-2` | $13.33343406 | **$18.20559006** | +$4.872156 |
| `conv2d-feedback-20261021` | $15.21992213 | **$20.53906313** | +$5.319141 |
| `conv2d-flame-20261020` | $7.87428804 | **$10.73508804** | +$2.860800 |

### 2. TASK 预算（F8）— `1a8ac9a` / `6f0528b`

- `render_contract(brief, max_bytes=12_000)`：合同、签名、命令、证据规则是必需段；成绩单、失败反馈、NOTES、tips、历史、现场证据按优先级加入，超预算的块被省略并**留下文件路径**。
- 上游预算也收紧：`_notes_for_next` 4000→2500，scorecard 2000→1000，history 8→5 行，tips 1200 字符 + verified 1800 字符。`TIP_KEEP` 保持 8（字符预算才是停止条件，N4）。
- 新增测试：动态块超预算时 TASK.md ≤ 12KB + 省略标记，必需段和两条命令仍在。
- 重新录制 `tests/golden/hw_off.json`（只有 4 个 orchestrator 哈希变化）。

### 3. 候选索引 + 请求路径（F4/F5/F6/F15）— `109cad1`

- `INDEX_SCHEMA_VERSION = 5`，新增 `candidates` 表（status/stage/speedup/within_noise/eligible/kernel_sha256），`runs` 增加 `kernel_sha256` / `protocol_version` / `committees_json`。
- `runs_page` 支持 `cursor`（keyset）、`sort=started|speedup|spent`、`publication`、`min_speedup`、`since`；返回 `next`。`offset` 保留为旧调用方的回退。
- `history()` 不再每个请求 `sync_known()`，按 `(db path, revision)` 缓存；`tick()` 默认全进程 0.5s 节流（`OBSERVE_TICK_INTERVAL=0` 供测试）。
- `feature_grid` 的 `feature_signature` 按源码 sha 缓存；state/result 变化仍会重建格子，但不再重算相同源码。
- 新增读接口：`/api/v2/runs/{id}/candidates`、`/api/v2/failures`、`/api/v2/best`。
- `protocol_version = "2"` 写入 `run_config.json` 和 `result.json`；旧 run 索引为 `legacy`。
- 索引实测：**candidates 741 行，arms 13**。

### 4. 批量幂等 + experiment manifest（F9/F13）— `004027c`

- `experiments/common.run_one`：非空目录若是 flame-chase（`run_config.json` + `state.json` + `transcript.jsonl`）就 resume，不再抛 `RunStartError`；只有 stray 文件的目录仍按旧行为拒绝。
- resume 保留运行记录里的 `committees`；没有剩余轮次时也能完成收尾。
- `experiments/manifest.backfill`：为 p1/repeat 这类旧目录按子目录建 arm、attach 现有 run；delta 不含 budget（避免以后 `ensure` 因预算变化拒绝）。
- 回填结果：`runs/apple-m4-max/p1` **11 arms**，`repeat_20261007-015329` **2 arms**；`observe` 的 `arms` 从 0 变 **13**。

### 5. agent_work 上限（F3）— `1a8ac9a`

- `prune_work` 把会话里非权威文件（`v/`、探针脚本、seed kernel、report、guidance 副本）打进 `<run>/scratch/<label>.tar.gz` 并删除；`NOTES.md`、`FEEDBACK.md`、`self_checks.jsonl`、`TASK.md`、交卷文件保留。
- 超过 50MB 的归档直接丢弃并写 JSON manifest（硬上限）。
- `experiments/prune_agent_work.py` 对已有 run 应用同一规则。回填：**57 + 91 个会话，共约 111.5MB**；`agent_work` 从 **112.0MB** 降到 **0.5MB**；`runs/` 从 1,234.4MB 降到 **1,129.2MB**（大头仍是 legacy chat transcript 759.7MB 和 XML 332.4MB）。
- 旧 TASK.md 从 scratch 归档恢复到 `agent_work/<label>/TASK.md`，`repro.sh` 的 91 个样本仍可测。

### 6. P1（F7/F11/F2）— `ac04199`

- `taic/brief`：一次遍历 transcript；默认不读全局 GPU 锁事件（`--gpu` 才给）；路径相对仓库；新增 `schema_version` / `protocol_version` / `stamp`；`brief --kernels` 改为紧凑 JSON。
- `observe/app.py`：`ETagMiddleware` + `GZipMiddleware`；`If-None-Match` 返回 304；SSE 不缓冲。
- `transcript.jsonl`：`LLMConfig.keep_transcript_bodies` 默认 **True**（直接构造，测试用），`load_config` 默认 **False**；chat 行默认只写 `messages_sha256/bytes`、`reply_sha256/bytes`。
- 看板主列表：cursor 翻页、`stop_reason`/花费/模型列、筛选写入 URL、`content-visibility` 行级虚拟化；`runQuery` 支持 sort。

## 验证

| 项 | 结果 |
|---|---|
| 离线门禁 | `uv run pytest -q -m "not gpu and not llm"` → **843 passed, 5 skipped, 133 deselected, 72.61s** |
| `repro.sh` | exit 0；`brief` **1,479B**（原 2,454B），无 GPU 事件，路径相对 |
| `agent_work` | **0.5MB**（原 112.0MB） |
| `runs/` | **1,129.2MB**（原 1,234.4MB） |
| index | runs 52 / candidates 741 / arms 13 |
| 计费重算 | 三个 run 与 REVIEW.md 的数字逐分一致 |
| 测试新增 | 计费、重算、TASK 预算、prune、candidates/keyset、resume、backfill、ETag/gzip、transcript 哈希 |

## 仍未做（明确留给下一轮）

1. **10⁵ 合成压测**：F4 的 25s 仍是外推（U）；没有在 `/tmp` 造 10³/10⁴ 个 run。
2. **F10 的 GPU 节省**：`candidate_serial` 默认移除、baseline 固定、eval 缓存都未做；unchanged_patience 仍等试点。
3. **F12 保留策略**：`data/deleted` 266MB 的 TTL、manifest、`taic gc`、kernel blob/refcount 未做。
4. **F14 的会话表**：只加了 `committees_json` 和 transcript 的 agent 行；`runs.model` 仍是第一委员会，没有独立 `sessions` 表。
5. **F15 的 eval 级 epoch**：`protocol_version` 只落在 run 级，`EvalReport` 还没有。
6. **`feedback_claims.topic`**：仍然 0/320；写入端未补。
7. **Leaderboard / failure 页面 / 图表**：只加了 API，没有前端页面；主列表仍是整表 innerHTML（cursor 降低了页大小）。
8. **`taic query` / MCP**：没有新增命令层；AI 目前用 `brief` + HTTP 读接口。

## 风险与回滚

- `candidates` 首次建索引会读 748 份 report/kernel；一次性成本，已实测可接受。
- `history()` 的缓存键是 `(db path, revision)`；`tick()` 0.5s 节流意味着页面最多延迟半秒看到新 run（SSE 也是）。
- TASK 预算可能省略 tips/历史；省略块写明 `NOTES.md`/`FEEDBACK.md`/`feedback.jsonl`/`state.json`，不是静默丢失。
- `prune_work` 归档 seed kernel/report/guidance；`agent_work` 不再保留会话现场，scratch 是第二份。旧 run 的 `TASK.md` 已从归档恢复。
- `ETagMiddleware` 缓冲非 SSE 响应；大 HTML 仍是一次性内存，但 84KB 量级可接受。
- 回滚单个批次：按上表 revert 对应 commit；数据回填（`cost_recomputed.json`、`experiment.json`、`scratch/`）都在 gitignored 的 `runs/` 下，不影响代码回滚。
