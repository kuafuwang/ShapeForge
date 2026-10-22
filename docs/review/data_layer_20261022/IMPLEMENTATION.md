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
| `7389851` | Keep resumed token counts, a partial scorecard, and the flame-chase protocol epoch. |
| `a877c13` | Measure the index at 100k runs, index sessions and the eval protocol. |
| `519b93c` | Complete the TASK omission note instead of cutting it at the byte cap. |
| `7adf797` | Add read-only query and gc commands, and best/failure views. |
| `6eb8f24` | Turn off the unread serial runner, cache identical evaluations, and set a 30-day deleted TTL. |
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
- `experiments/prune_agent_work.py` 对已有 run 应用同一规则。回填：**57 + 91 个会话，共约 111.5MB**；`agent_work` 先从 **112.0MB** 降到 **0.5MB**。
- 为了保留审核样本，旧 TASK.md 又从 scratch 归档恢复到 `agent_work/<label>/TASK.md`（91 份）。恢复后现场为 `agent_work` **2.6MB / 214 文件**、`runs/` **1,131.3MB / 4,233 文件**；报告此前的 0.5MB / 1,129.2MB 是恢复前的数字。新的会话保留 TASK.md，未来的增长不会再出现这种恢复差异。

### 6. P1（F7/F11/F2）— `ac04199`

- `taic/brief`：一次遍历 transcript；默认不读全局 GPU 锁事件（`--gpu` 才给）；路径相对仓库；新增 `schema_version` / `protocol_version` / `stamp`；`brief --kernels` 改为紧凑 JSON。
- `observe/app.py`：`ETagMiddleware` + `GZipMiddleware`；`If-None-Match` 返回 304；SSE 不缓冲。
- `transcript.jsonl`：`LLMConfig.keep_transcript_bodies` 默认 **True**（直接构造，测试用），`load_config` 默认 **False**；chat 行默认只写 `messages_sha256/bytes`、`reply_sha256/bytes`。
- 看板主列表：cursor 翻页、`stop_reason`/花费/模型列、筛选写入 URL、`content-visibility` 行级虚拟化；`runQuery` 支持 sort。

## 验证

| 项 | 结果 |
|---|---|
| 离线门禁 | `uv run pytest -q -m "not gpu and not llm"` → **853 passed, 5 skipped, 133 deselected, 75.40s** |
| `repro.sh` | exit 0；`brief` **1,479B**（原 2,454B），无 GPU 事件，路径相对 |
| `agent_work` | **2.6MB / 214 文件**（恢复 91 份 TASK.md 后；恢复前 0.5MB） |
| `runs/` | **1,131.3MB / 4,233 文件**（原 1,234.4MB） |
| 索引 | schema **8**；`protocol_version` = 4 `flame_chase_1` + 48 `legacy`；`protocol_epoch` 把前者归到 `2`；sessions **85**；candidates 741；arms 13 |
| `turns` | 重建后 **0**：仍是懒投影，打开某个 run 才写入；54 个 transcript（759.7MB）没有删，`turn_bytes>0` 要等一次详情请求 |
| 10^5 合成压测 | cold sync **157.6s**、hot tick **6ms**、history **0.83s**、first list **4ms**、second page **2ms**、best **0.5s**、DB **167MB**（`SCALE_PROBE.json`） |
| index | runs 52 / candidates 741 / arms 13 |
| 计费重算 | 三个 run 与 REVIEW.md 的数字逐分一致 |
| 测试新增 | 计费、重算、TASK 预算、prune、candidates/keyset、resume、backfill、ETag/gzip、transcript 哈希 |

## 7. 二次实现审核（REVIEW_3adb8d4）修复 — `7389851`

`REVIEW_3adb8d4.md` 的 PASS-WITH-CHANGES 三条收尾已落地：

1. **resume 保留 token**：`taic/resume.py:_requests_from_transcript` 的 `agent` 行读取 `event.tokens` 的 input/output/cache_read/cache_write，chat 行也保留 cache-write；`CostLedger.restore` 因此能把 write 列加回去。测试 `test_agent_tokens_survive_a_resume_ledger_restore` 和 `test_chat_usage_write_tokens_survive_a_resume`。
2. **TASK 预算优先保留当前现场**：装不下时把「当前现场」裁到剩余字节（保留种子行和部分成绩单），不再整块丢掉；省略注记计入 12,000 字节上限。测试改为真实的 `conv2d` 和 `flash_attention` paper 任务，断言 `<= TASK_MAX_BYTES`、`外层成绩单` 仍在、`本段已截断` 和省略注记存在。golden 重录（只有 flash_attention plan_mix=0 一个键变化）。
3. **protocol epoch 回填**：索引 schema 升到 6；`_upsert_run` 在没有显式 `protocol_version` 时读 `loop_protocol`，`flame_chase_1` 不再和 chat 一起标成 `legacy`。重建后的现有 52 行：**4 个 `flame_chase_1` + 48 个 `legacy`**；测试 `test_a_legacy_flame_chase_run_is_not_labelled_chat`。

## 8. 补齐 §7 中不需要新政策的部分 — `a877c13` / `519b93c` / `7adf797`

`REVIEW_5375077.md` 的两条残留（省略注记被裁、`protocol_version` 两代分桶）也已处理：

- 省略注记预留 420 字节，真实 conv2d / flash_attention 测试断言「需要时读取」在文末；golden 重录。
- `runs_page` 增加 `protocol_epoch`：`flame_chase_1` 与 `2` 归为同一代，`legacy` 单独一桶。

| 原「下一轮」项 | 现在的状态 |
|---|---|
| 10^5 压测 | **做了**。`experiments/scale_probe.py` 在 `/tmp` 造 100k 个最小 run；cold sync **157.6s**、hot tick **6ms**、history **0.83s**、first page **4ms**、second page **2ms**、best **0.5s**、DB **167MB**。结果在 `SCALE_PROBE.json`。请求路径已移出目录扫描；`best` 的 0.5s 仍是 100k 全表，留作下一步的聚合表。 |
| sessions 表 | **做了**。schema 8 新增 `sessions`（committee/model/tokens/cost）；现有 **85 行**（kimi 35、deepseek 50，含旧 run 的空 committee）。`taic query sessions|models` 和 `/api/v2/{sessions,model-stats}` 可读。 |
| eval 级 protocol | **做了**。`tcenv/schema.PROTOCOL_VERSION` 写入 `EvalReport.protocol_version`；`candidates.protocol_version` 一并索引。 |
| data/deleted TTL | **做了工具和默认政策**。默认保留 **30 天**；`taic gc` 只报告，`--apply` 才删除并写 `gc_log.jsonl`。真实 266MB 没有删，等一次人工 dry-run；政策写入 `docs/维护决策.md`。 |
| leaderboard/failure 页面 | **做了入口**。看板头部「最佳 / 失败」打开两个表，分别读 `/api/v2/best` 与 `/api/v2/failures`。 |
| taic query/MCP | **做了 CLI**。`taic query runs|best|failures|sessions|models|capabilities`，支持 `--jsonl`。MCP 传输层没有做；CLI 是同一读模型的最小面。 |
| feedback_claims.topic 写入 | **未做**。agent 输出没有 topic 字段，现有 `topic` 是人工 review 时填的；要自动填必须先定义分类法（谁产生、允许哪些值、是否 LLM 打标）。加一个空字符串之外的猜测会让 320 行变得不可审。 |
| candidate_serial 移除 / eval 缓存 | **未做**。两者都改测量协议：移除 serial 会改变每颗候选的计时，eval 缓存要决定哪些 phase/seed 可以复用。需要一次 GPU A/B 和 `protocol_version` bump；没有你的预算批准不能跑。 |

## 9. 用户批准后的 A/B 与政策决定 — `6eb8f24`

| 决定 | 证据 | 落地 |
|---|---|---|
| `candidate_serial` 默认关闭 | 3 轮 × 3 个已确认内核（conv2d / flash_attention / relu，paper，seed 12345，15 reps）：关闭后墙钟 −5.3% / −8.3% / −4.0%；候选中位数 **−0.82% / +0.08% / +1.14%**，cv 2–4%，无系统偏差。`SERIAL_AB.json` | `SearchConfig.measure_serial=False`、`taic.toml` 同值；`tcenv eval --serial` 可重新打开；`PROTOCOL_VERSION="3"` |
| `eval_cache` 默认开启 | 同一颗 relu、同一 seed/reps/mode/target/protocol：第一次 wall **4.41s**，第二次 **0.68s**，`cache_hit=true`，`candidate_us`/`speedup` 逐字段相同。`CACHE_AB.json` | 缓存键含 kernel/task fingerprint/seed/reps/mode/target/protocol/measure_serial；不同 seed 的确认批次仍上 GPU；`--eval-cache` 供 CLI，搜索默认 `eval_cache=true` |
| `data/deleted` 30 天 | 当前 266MB，无 TTL；删除不可逆 | 默认 `older_than_days=30`；`taic gc` 只报告，`--apply` 才删并写 `gc_log.jsonl`；政策写入 `docs/维护决策.md` |
| `feedback_claims.topic` 维持现状 | 320 条中 0 条非空；写入端只有人工 review | 决定与重启条件写入 `docs/维护决策.md`；不自动生成 |

## 仍未做

1. **MCP server**：`taic query` 已有同一读模型；MCP 的传输、鉴权、工具 schema 未定，等接入形态。
2. **10^5 的 `best` 全表 0.5s**：需要聚合表/物化视图；列表页/历史已经 2–4ms/0.83s，不影响日常。

## 风险与回滚

- `candidates` 首次建索引会读 748 份 report/kernel；一次性成本，已实测可接受。
- `history()` 的缓存键是 `(db path, revision)`；`tick()` 0.5s 节流意味着页面最多延迟半秒看到新 run（SSE 也是）。
- TASK 预算可能省略 tips/历史；省略块写明 `NOTES.md`/`FEEDBACK.md`/`feedback.jsonl`/`state.json`，不是静默丢失。
- `prune_work` 归档 seed kernel/report/guidance；`agent_work` 不再保留会话现场，scratch 是第二份。旧 run 的 `TASK.md` 已从归档恢复。
- `ETagMiddleware` 缓冲非 SSE 响应；大 HTML 仍是一次性内存，但 84KB 量级可接受。
- 回滚单个批次：按上表 revert 对应 commit；数据回填（`cost_recomputed.json`、`experiment.json`、`scratch/`）都在 gitignored 的 `runs/` 下，不影响代码回滚。
