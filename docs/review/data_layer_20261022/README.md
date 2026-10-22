# 数据层审核包（供二次审核）

> **执行状态（2026-10-22）**：已按 `REVIEW.md` 完成 P0 + P1 的代码修改，提交与验证见同目录 `IMPLEMENTATION.md`；本文件保留审核时的口径与修正记录。

**仓库**：`ai_lowerling`
**审核版本**：`ace4ac6` + 3 个未提交文件（`docs/架构/可观测.md`、`observe/static/index.html`、`tests/test_observe_feedback.py`）；另有未跟踪的 `docs/target_profiles_6f03f75e.plan.md`
**日期**：2026-10-22
**审核范围**：数据生成、存储、索引/检索、AI 接口、人类看板
**方法**：5 个只读 agent 分审 + 交叉测量；未运行 `taic run`/`resume`、未启动 GPU、未调用 LLM。离线门禁 `uv run pytest -q -m "not gpu and not llm"` = **823 passed, 5 skipped, 133 deselected**。
**证据等级**：`✔` 直接测量/复现；`A` agent 测量且抽查；`U` 外推或单一 run，需实验验证。

---

## 0. 给审核 agent 的阅读说明

你不需要重跑整个项目。建议顺序：

1. 读 §1（一句话结论）、§3（样本量现实）、§4（A/B/C/D 分类）。
2. 运行 `./repro.sh`（只读，无 GPU/LLM）。它会打印 §2 的所有基线数字。
3. 对 §5 的 F1–F15 逐条给 **PASS / FAIL / UNKNOWN**，附证据；对 §7 的 12 条逐条挑战。
4. 用 §9 的模板输出。不要把 `runs/`、`data/observe.sqlite` 写坏；需要写就用 `/tmp`。

**禁止**：`taic run`、`taic resume`、任何 GPU kernel、任何 LLM API 调用。`taic brief` 和 `repro.sh` 是只读的，允许。
**行号**：基于 `ace4ac6` 工作树，±5 行内；以 `grep -n` 为准。
**重要口径**：不要把旧 chat 协议（legacy）和当前 hmz 协议（`flame_chase_1`）的数据混在一起算；不要把 `runs.model` 当成会话模型（它只记第一个委员会）。

---

## 1. 一句话结论

数据层当前是「为约 50 个 run 写的脚本集合」，不是「为 10^5 个 run 写的系统」。真正的瓶颈不是磁盘，而是：

1. **每次请求重算 / 扫描**（`history()`、`sync_known()`、`tick()`、`feature_grid`）；
2. **每场会话重灌上下文**（取样 TASK.md 26,616B：tips 8,201B、上一轮 NOTES 4,565B；两个块合起来才是 12.7KB 级；每搜索重复 21–64 次）；
3. **候选级事实没进索引**（741 个 candidate、133 个 `abi_error` 无法查询）；
4. **计费漏项**（`_priced` 漏 `cache_write`：FA-2 记录 $13.33 / 实际 $18.21、feedback $15.22 / $20.54、flame $7.87 / $10.74；见 §5 F1）；
5. **批量不可重入**（suite 第一格 interrupted 就整批失败）。

「hmz 数据少」**一半是真样本少**（当前协议 4 个 run、85 场会话、60 场集中在 3 个 run），**一半是指标口径造成的假象**（两委员会被压成 `runs.model` 一列，kimi 实际跑了约一半会话但索引只显示 2 个 run）。

---

## 2. 可复现基线

> 运行 `./repro.sh` 会打印下列数字；也可手工执行 §10 的命令。

| 项 | 数值 | 证据 |
|---|---|---|
| `runs/` | **1.23GB / 5273 文件 / 54 个 run 目录**（另有 3 个实验容器目录）；索引 **52** 个 | ✔ |
| 文件构成（st_size 合计 1,234.4MB） | transcript **759.7MB（61.5%）**、xctrace export **332.4MB（26.9%）**、`agent_work` **112.0MB（9.1%）**、`candidates` **20.5MB（1.7%）**、其它 9.8MB | ✔ |
| `data/` | 269MB，其中 `data/deleted/` **266MB**、`observe.sqlite` 2.3MB | ✔ |
| 最大单文件 | `runs/apple-m4-max/p1/conv2d_paper/transcript.jsonl` **92.82MB** | ✔ |
| XML | `counter_teaching/export/counter_values.xml` 326MB → gzip **21.7MB（15×）** | ✔ |
| 索引表 | 52 runs / 719 turns / 36 feedback_items / 320 claims / experiments **2** / arms **0** | ✔ |
| transcript 构成 | 旧 chat run：99% 是 `chat`（messages+reply，平均 260KB/条）；hmz run：`agent` 元数据 **10–13KB/run** | ✔ |
| 协议 | 50 个 legacy chat、**4 个 `flame_chase_1`**；有 `agent` 会话的 run **12** 个，共 **85 场** | ✔ |
| 会话集中度 | conv2d-feedback 25 场 + fa-20261022-2 21 场 + conv2d-flame 14 场 = **60/85** | ✔ |
| confirmed | 10 个；当前协议只有 **3** 个；**28** 个是「旧版未绑定源码」 | ✔ |
| 模型口径 | 索引 `runs.model`：deepseek-flash 49 / kimi-k3 **2**；`run_config.committees` = first deepseek + second kimi | ✔ |
| `brief` | 2,454B（conv2d-committees）；**8 个全局 GPU 锁事件 = 1,039B（42%）**；3 个 run 的占比 39/42/73%；读 transcript 两次；旧 run 305–445ms / 149MB | ✔ |
| `run_detail` | 44.7–259KB；conv2d-committees 44.7KB / 363ms；`feature_grid` 10.7s（25 颗 kernel；**U，二次审核未复测**）；95.5MB 读 / 187KB 返回（**U**） | ✔/A/U |
| 请求路径 | `history()` 全量 `row_json`（54KB/14ms）；`sync_known()` 1093 stat / 14.8ms（52 runs）；SSE 每连接 1Hz `tick()` | A |
| 看板 | 84KB 内联 HTML；首屏 3 请求+SSE ≈106KB；每 revision 再发 36KB+；feedback 快照 341KB；无 gzip/ETag | A |
| TASK.md（91 个会话副本） | 中位 **21,664B**；当前树最大 **48,612B**（无 tips 的旧会话）；有 tips 的最大 **19,628B**；审核样本 `r20.p0` 26,616B（tips 8,201B + NOTES 4,565B）；`init` 10,946B；≈7.2k tokens/场；21–64 场/搜索 | ✔/A |
| 文件数/run | 全集中位 **40**；有 agent_work 的 13 个 run 中位 **29**；当前大 run：**797**（conv2d-feedback，31.9 文件/场）、558（FA-2，26.6/场）、239（conv2d-flame，17.1/场）；`v/` 296 文件 / 83.7MB | ✔ |
| 生产 | 42 个 run ≈55.6h / **$151.79**；计时 GPU 实际 0.68h → **1.2% 利用率**（U，定义见 §8） | A |
| 计时构成 | 454 个 ok full eval：candidate 33% / `candidate_serial` **34%** / baselines 33%；`candidate_serial_us` 无人读 | A |
| 原子性 | 31/31 `resume.lock` 是死 pid；candidate `kernel.metal` 先于 report 写；事件 temp 名无 per-call 计数 | A |

---

## 3. 样本量现实：hmz 到底有多少

| 指标 | 数值 | 含义 |
|---|---|---|
| 当前协议 run | 4（`flame_chase_1`）；其中 confirmed **3** | 策略结论最多 3 个样本 |
| `agent` 会话 | 85 场 / 12 个 run | 会话级统计的总分母 |
| 集中度 | 60/85 在 3 个 run | 任一「率」基本由 2–3 个 run 决定 |
| `publication` | confirmed 10、in_progress 10、不确定 2、旧版未绑定源码 28、未确认 2 | 只有 10 个可信发布，且混合协议 |
| 索引 model | kimi 只 2 个 run | **口径假象**（见 F14） |
| `topic` | 0/320 非空 | 写入缺口，不是数据少（见 F15） |
| `arms` | 0（p1/repeat 无 `experiment.json`） | 写入缺口（见 F13） |
| `turns` | 11/52 个 run | 懒投影，不是缺失 |

**判据**：
- **S0 = 与样本量无关**（固定成本/逻辑错误）→ 现在修。
- **S1 = 随规模恶化**（O(N) 扫描、每 run 文件数）→ 现在修，并用合成数据验证。
- **S2 = 需要更多数据才能下结论**（率、分布、A/B、策略调参）→ 先试点，不要用 n=2–3 拍板。

---

## 4. 分类结论

### A 类（S2，等试点）
confirmed 只有 10 个（当前协议 3 个）；leaderboard、best-by-task、failure taxonomy 阈值、unchanged 率、agent_failed 率、每场成本、kimi vs deepseek 的性价比、FTS 收益、A/B 归因——**都还没有统计功效**。

### B 类（S0/S1，现在修）
F1 计费漏项、F2 transcript 正文、F3 agent_work 复制、F4 O(N) 请求扫描、F5 候选级索引缺失、F6 detail 重算、F7 brief 噪声、F8 TASK.md 每场 7.2k tokens、F9 批量非幂等、F10 GPU/LLM 浪费（部分）、F11 看板规模化、F15 协议 epoch 不可比。

### C 类（假稀疏，写入/建模问题）
`topic` 0/320、`arms=0`、`ledger.charge()` token 全 0、`turns` 11/52（懒投影）、`runs.model` 压平委员会、`feature_grid` 每次重算。**加数据不会好**。

### D 类（历史包袱，冷归档）
759MB 旧 chat transcript（99% 正文）、326MB 单个 XML、266MB `data/deleted`。这些让「存储 1.2GB」「brief 读 149MB」「zstd 6%」看起来像 hmz 的问题；实际 hmz transcript 10–13KB。**不要用来推断 hmz**。

---

## 5. 发现清单（F1–F15）

> 每条都可被 `repro.sh` 或 §10 的命令复现。**请逐条给 PASS/FAIL/UNKNOWN。**

### F1 · blocker · S0 · 计费漏 cache-write ✔
- **证据**：`taic/committees.py:190 _priced()` 只设 `price_in/out/cache_read`，**没有** `price_cache_write`，第二个委员会继承 deepseek 的 `0.0`。`flash_attention-20261022-2` 记录 **$13.3334**，重算 **$18.2056**（1,624,052 cache-write × $3/M = $4.872，漏 27%）。`conv2d-feedback-20261021`：记录 $15.2199，重算 **$20.5391**（35%）。`conv2d-flame-20261020`：记录 $7.8743，重算 **$10.7351**。当前协议 3 个 run 的 `ledger.json` 请求行全部 `hit=miss=out=0`（`charge()` 路径）；更早的 hmz run 仍走 chat 的 `add()`，所以有 token 数——两者不要混算。重算脚本 `experiments/reprice_runs.py` 已写 `cost_recomputed.json`，`spent_usd` 不变。
- **影响**：预算门放行超支；历史成本不可用于决策。
- **修复**：一行补 `price_cache_write=float(price.get("cacheWrite", price.get("input", 0.30)) or 0.0)`；历史按 token 重算并写 `cost_recomputed`；记录 `cost_basis`。

### F2 · blocker · S1/L · transcript 存正文，O(n²) ✔
- **证据**：`taic/llm.py:769-776` 原样记录 `messages`+`reply`；`transcript.jsonl` **759.7MB / 54 文件 = 61.5%**；最大 92.82MB；zstd 后约 **6%**；system prompt 在 64 个请求里重复，白存 **~64MB**。hmz 只写 `agent` 元数据，**10–13KB/run**。
- **影响**：旧数据主导磁盘；新数据没有这个问题，但 guardrail 缺失。
- **修复**：正文默认不落盘，只留元数据；需要审计时 `--keep-agent-io` + zstd；`sys_sha` 引用 system prompt。

### F3 · blocker · S0/S1 · agent_work 无上限 + 每场复制 ✔/A
- **证据**：`agent_work` **112MB / 1409 文件**；全 run 中位 40 文件、有 `agent_work` 的 run 中位 29；但当前大 run 是 **797 文件 / 25 场 ≈31.9/场**（conv2d-feedback）、558/21≈26.6（FA-2）、239/14≈17.1（conv2d-flame）。每场复制 `msl_reference.md` 16KB×21、`TASK.md` 23KB×21、`experience.md` 5.4KB×21、seed kernel/report ×20；`v/` 296 文件 / **83.7MB** 消融 kernel，无代码读取。
- **影响**：文件数随会话线性膨胀；1000 runs ≈1GB 静态副本。
- **修复**：hmz cwd 移到 `$TMPDIR/taic/<run>/<label>`；只回拷 NOTES、self_checks、交卷 kernel；scratch 打 `tar.zst` + 50MB 上限。

### F4 · critical · S1 · 每个请求 O(全部 run)
- **证据**：`observe/query.py:41-52` `history()` 加载全部 `row_json`（今天 54KB/14ms）；`observe/index.py:419 sync_known()` 1093 stat/14.8ms（52 runs）；`index.py:407 tick()` 被 `runs_page` 和每帧 SSE 调用；SSE 每连接 1Hz。10^5 runs 外推 **~25s/请求（U）**。
- **修复**：写穿 journal（`runs/.index_journal.jsonl`）；请求路径禁止 `sync_known/tick`；聚合表；共享 1Hz ticker；keyset 分页。

### F5 · critical · S1 · 候选级事实不进索引
- **证据**：741 个 candidate、748 份 `report.json`；**133 abi_error、84 incorrect、29 compile_error** 无法从索引查询；`runs` 没有 `kernel_sha256` 列；无 FTS5；`task/operator/model/dtype/publication` 无索引；`runs_page` 只有 `LIMIT/OFFSET`，无排序/min_speedup/since。（二次审核修正：`status != 'missing'` 使复合索引失效的说法过了一步；schema 里原本没有那个复合索引，缺的是候选表和这些列的索引。）
- **修复**：`candidates(run_id,label,stage,status,kernel_sha,speedup,within_noise,cost)` + 部分索引 + FTS5 + keyset；`stop_class` 枚举。

### F6 · high · S0 · detail 全量重算
- **证据**：`run_detail` 44.7–259KB（conv2d-committees 实测 44.7KB / 363ms）；`query.feature_grid` 对 25 颗 kernel 逐个 `feature_signature` → **10.7s（U，二次审核未复测）**；服务端读 95.5MB 返回 187KB（**U**）；`state`/`result` **23 键重复**；`timeline/llm/confirmations/trace` 前端 0 引用（`index.html`）。
- **修复**：ingest 时算 `feature_grid`/`kernel_writer`；detail 拆 `/summary`（≤2KB）+ `/candidates` `/turns` `/timeline` `/feedback` 懒加载。

### F7 · high · S0/L · brief 噪声 + 双读 transcript ✔
- **证据**：`taic/brief.py:100,116` 读 transcript 两次；`:119` 把全局 GPU 锁事件（1,039B，来自别的 run）塞进 `gpu` 键——conv2d-committees 实测 2,454B / 8 个事件 / 1,039B / 0.14s；三个 run 占 **39/42/73%**；绝对路径 260–456B；`cands` 截 8 行且丢 `cost_usd/improved/within_noise`。旧 run 305–445ms / 149MB。
- **修复**：从索引出 summary；`--gpu` 才给事件；相对路径；`schema_version`/`stamp`；cands 分页。

### F8 · high · S0 · TASK.md 每场重灌
- **证据**：`init` 10,946B；审核样本 `r20.p0` **26,616B**，其中 peer tips **8,201B**、上一轮 NOTES **4,565B**（源文件 `NOTES.md` 4,564B）。二次审核修正：**“NOTES 12,751B”不成立**——12,766B 是从 NOTES 标题到 `# 证据` 的跨度，已经把 tips 算进去。tips 从 1.9KB 长到 8.2KB，中间有一次缩短（r15 8,068B → r16 7,947B），不是单调追加。≈7.2k tokens/场；21–64 场/搜索。
- **修复**：常量契约只给一次（或 hash + 工具查询）；NOTES 给 L1 摘要 + 路径；tips 给 digest；目标 ≤12KB/场（−53%），按「有上限的摘要」算，不假设单调增长。

### F9 · high · S0 · 批量不可重入
- **证据**：`experiments/run_suite.py:66`/`p1_runs.py:149` 把「无 result.json」当「没跑过」；遇到 interrupted 的格子 Orchestrator 拒绝非空目录 → `RunStartError` 杀掉整批；resume 的 `snapshot_results` 把 `result.json` 移成 `.before_topup_N.json`，再死一次就像从没跑过；`_confirm` 每次 resume 全量重测；FA-2 的 winner 被测 5 次。
- **修复**：`queue.jsonl`（queued/running/needs_resume/done/skipped）；`run(cell)` 必须 resume；`confirm/<sha>.json` 幂等；`evalkey` 缓存。

### F10 · high · S0/S2 · GPU/LLM 浪费
- **证据**：`candidate_serial` 占 **34%** 计时 GPU，`candidate_serial_us` 无人读；baselines 33% 每候选重测（mlx cv 0.8%，49 次重测）；FA-2 winner 测 5 次；`unchanged` 15/25 会话 $10.46/$15.22（**单 run，S2**）。
- **修复**：默认去掉 `candidate_serial`；baseline 按 `(task,target,calibration)` 锚定；unchanged_patience **试点后**再定；eval 缓存去重（不同 seed 的确认批次保留）。

### F11 · high · S0/S1 · 展示层不工业化
- **证据**：84KB 内联 HTML、无 gzip/ETag；首屏 3 请求 + SSE ≈106KB；每 revision 再发 36KB+；主列表整表 `innerHTML` 重建；筛选不进 URL；无排序/FTS/分页大小；不显示 `stop_reason`；只渲染 16 列中 6 列；无图表；页面带写 token + 手写 `esc`（XSS 面）。二次审核修正：审查页已按 `feedback_revision` 增量、单卡替换（`feedback.py:735-750`、`index.html:450`）；「341KB 全量快照」只对首次快照成立，不应再按它排审查页工期。
- **修复**：拆 `app.js/css` + gzip/ETag；URL 即状态；keyset + 虚拟列表（本期先做 keyset + content-visibility）；聚合图表；CSP + 统一 `esc`；`Runs/Leaderboard/Experiments/Reviews/Cost/Catalog`。

### F12 · medium · S0 · 保留/删除不可持续
- **证据**：`data/deleted/` **266MB** 只 `shutil.move`、无 TTL/压缩；`data/deletion_log.jsonl` 在被 `.gitignore` 排除的 `data/` 下；`kernels/.../best.metal` 被更快的覆盖，历史 confirmed 源码只留 sha；**31/31** `resume.lock` 是死 pid。
- **修复**：`deleted/` 存 `manifest.json` + `tar.zst`；`taic gc --apply` TTL；日志移出 `data/`；kernel 用 blob + refcount，`best.metal` 变指针。

### F13 · medium · S0 · 实验 provenance 在丢
- **证据**：`runs/apple-m4-max/p1`（12 arms、**418MB、$35.34**）和 `repeat_20261007-015329`（163MB）**没有 `experiment.json`**；`observe.sqlite` 里 `experiments=2, arms=0`；17/29 `run_config` 是 `git_dirty=true` 但无 diff；报告无 calibration sha / macOS-Metal 版本；`run_config` 只记第一个委员会的价格。
- **修复**：每个 experiment 落 manifest；arm delta 不写 Python 常量；manifest 记价格表/校准/工具链；回填 p1/repeat。

### F14 · medium · S0 · 身份/口径不统一
- **证据**：3 种 run_id（文件路径 / `target/name` / catalog basename）；`publication` 5 个自由文本含中文（`confirmed/in_progress/不确定/旧版未绑定源码/未确认`）；`brief` 无 `schema_version`；4 套 read model（index/detail/brief/plots）；`runs.model` 取 `[llm].model`。二次审核修正：kimi 在 60 场里占 **28 场**，记录金额约 **80%**（$28.98/$36.43）；场次接近一半，金额是大头（输出 $15/M vs deepseek $1.2/M）。
- **修复**：统一 `RunSummary` + `schema_version`；publication 枚举；per-session `committee/model`；catalog 用同一 run_id。

### F15 · medium · S0 · 协议 epoch 不可比
- **证据**：`run_config` 有 v1/v2；确认批次 1→3；`d3bd2dc` 把 encode 移进计时窗口；target 重构改了 contract/limits；没有 `protocol_version`/`metric_protocol` 列。跨这些变更聚合速度/成本/confirmed 数会得到混合分布。
- **修复**：每条 run/eval 记录 `protocol_version`；所有聚合默认按 epoch 分组；旧数据回填 `epoch=legacy`。

---

## 6. 建议与验收指标

### P0（与数据量无关，先做；顺序按二次审核）
1. **计费（F1 + N1/N2）**：补 `_priced` 的 cache_write；`add`/`charge` 记 write token 和 `cost_basis`；三个已结束 run 用 transcript 重算，单独写 `cost_recomputed`，不改 `spent_usd`。
2. **TASK 瘦身（F8，按修正后的体积）**：常量契约约 11KB/场；随场变大的是 tips（取样 8.2KB）和 NOTES（约 4.6KB）；目标 ≤12KB/场，按这两块减。
3. **候选索引 + 请求路径不扫全表（F4/F5）**：先做 `candidates` 表和 keyset；10^5 的 25s 用 `/tmp` 合成目录测完再写进验收表。
4. **批量幂等和 experiment manifest（F9/F13）**：目录非空即拒绝、`experiment.json` 缺失，这两条与样本量无关。
5. **agent_work 上限（F3）**：`v/` 的 83.7MB / 296 文件没有读者，先归档。
6. `brief`/`detail` summary-first + 去 GPU/绝对路径 + `schema_version` + 懒加载 tab。
7. 存储卫生：transcript 正文默认不落盘；blobs + manifest；retention/GC；旧数据冷归档。

### 试点（需要预算批准）
- **20–30 个 run**：2–3 task × 2 模型 × 5 run，约 **$300–600**（按当前 $13–20/run）。
- 产出：每场成本 P50/P90、unchanged 率、失败分类、cost/confirmed kernel 的置信区间；验证 F10/F13 的实际收益。
- 同步做 **0 成本合成压测**：只读复制/硬链现有 run 到 `/tmp`，造 10^3/10^4 个 run，测索引/查询延迟，验证 F4。

### P1（试点后）
Leaderboard / failure 页面 / 图表 / FTS 调参 / A-B 归因 / patience / unchanged_patience / 委员会轮换 / 导出与论文快照。

### 验收指标

| 指标 | 现状 | 目标 |
|---|---|---|
| `brief` | 2.4KB、旧 run 305–445ms、读 149MB | ≤500B、<10ms、不读 transcript |
| `detail` 首屏 | 45–259KB、10.7s（U） | summary ≤2KB、<20ms；tab 分页 |
| 10^5 runs 列表 / tick | 外推 25s / ~570ms（U） | <5ms / <2ms |
| 文件/场（当前大 run） | 17–32 | ≤5；单 run ≤150 |
| 现有语料存储 | 1.23GB | **40–110MB**（取决于旧正文删/冷存，见 §8） |
| TASK.md / 场 | 取样 26.6KB / 当前中位 21.7KB | ≤12KB，token −50% |
| 计费 | 记录 vs 真实差 27–35% | <1%，per-committee 可见 |
| 计时 GPU | 100% | −67%（去 serial + 固定 baseline） |
| 评测次数 | 100% | −60%（eval 缓存；不同 seed 的确认批次保留） |

---

## 7. 请二次审核重点挑战的 12 条

> 每条给了反证方法；**能推翻就推翻**。

1. **F1 计费**：`_priced` 是否真的没设 `price_cache_write`？用 FA-2 的 1.62M cache-write × kimi `$3/M` 重算是否 $18.20？对比 `result.spent_usd`。
2. **F2 transcript**：759MB 是否 99% 来自旧 chat？hmz 是否 10–13KB？用 `kind` + `loop_protocol` 交叉验证。
3. **F3 文件数**：全集中位 40、agent_work-run 中位 29，但当前大 run 17–32 文件/场；这个分母是否合理？`agent_work` 里哪些文件有读者？`v/`（296 文件 / 83.7MB）是否真的无人读？
4. **F4 O(N)**：25s 是外推（U）。用 `/tmp` 合成 10^3/10^4 run 实测；若 <5s 则降级。
5. **F5 候选索引**：133 个 abi_error 是否真的无法从现有索引回答？`feedback_items.kinds` 是否已部分覆盖？`task` 索引是否真的被删？
6. **F6 feature_grid**：在 `conv2d-feedback-20261021` 上复测是否仍 10.7s？缓存 key 是否含 `state.json` mtime？前端是否 0 引用 timeline/llm/confirmations？
7. **F7 brief**：GPU 事件占比是否 39–73%？是否读 transcript 两次？hmz run 上是否也慢（预期不是）？
8. **F8 TASK.md**：26.6KB 是否 = 12.7KB NOTES + 8.2KB tips？tips 是否每场增长、无去重？
9. **F10 unchanged 69%**：这是单个 run；请用 3 个当前协议 run 重算率与成本占比。
10. **F10 GPU 1.2%**：GPU 工作的定义（计时窗口/墙钟/命令缓冲 GPU 时间）？换定义后结论是否成立？
11. **F14 model 压平**：`runs.model` 是否只取 `[llm].model`？kimi 的实际 token/成本占比是否约一半？
12. **F12 存储目标**：40–60MB 与 106MB 的差别是否来自「删旧正文」vs「压缩旧正文」？`data/deleted` 266MB 是否计入？XML 是否必须保留原始？

---

## 8. 已知分歧、口径与不确定性

| 分歧 | 解释 |
|---|---|
| 全树内容去重 **3.5%** vs metal 重复 **32%** | 分母不同：前者是 1.23GB 全树，后者是 121.5MB metal 文件。结论：内容寻址省不了总字节，但能省文件数和 scratch 复制。 |
| 存储目标 **40–60MB** vs **106MB** | 前者默认旧 chat 正文不保留（只留元数据）；后者默认 zstd 压缩保留；需拍板。 |
| `brief` 慢 vs 快 | 旧 chat run（92MB transcript）慢；hmz run 快。不要把旧数据成本算到 hmz 头上。 |
| `history()` 14ms vs 25s | 前者是 52 run 实测；后者是 10^5 外推（U），必须合成压测。 |
| `turns` 11/52 | 懒投影（设计），不是数据缺失；但首次查询仍全量重读。 |
| `publication=confirmed` 10 个 | 混合协议（chat + hmz），不能跨 epoch 聚合。 |
| 近期协议变更 | 确认批次 1→3、encode 移入计时窗口、target 重构、self-check 路径；跨 `d3bd2dc` 前后指标不可比（F15）。 |
| 文件数“中位 398” | 早期 agent 报告的数字不可复现；`repro.sh` 实测全集中位 40、agent_work-run 中位 29，当前大 run 17–32 文件/场。指标改用 files/session。 |

---

## 9. 审核输出模板

请按下面格式返回，便于合并：

```markdown
## 审核结论
- 总评: PASS / PASS-WITH-CHANGES / FAIL
- 覆盖范围: F1–F15 + 12 条挑战

| ID | 结论 | 证据（命令/行号/数字） | 修正 |
|----|------|------------------------|------|
| F1 | PASS/FAIL/UNKNOWN | ... | ... |
...

## 被推翻/降级/升级的结论
- ...

## 新发现
- ...

## 建议优先级调整
- ...
```

---

## 10. 复现命令（与 `repro.sh` 等价）

```bash
cd /Users/kuafu/CLionProjects/ai_lowerling
find runs -type f | wc -l; du -sh runs data
find runs -name transcript.jsonl -print0 | xargs -0 du -ch | tail -1
find runs -name '*.xml' -size +100M -exec ls -lh {} \;
sqlite3 'file:data/observe.sqlite?mode=ro' \
  "select count(*) from runs; select count(*) from turns; select count(*) from experiments; select count(*) from arms;"
uv run python -m taic.cli brief runs/apple-m4-max/conv2d-committees-20261020 | wc -c
```

完整脚本见同目录 `repro.sh`；机器可读发现见 `findings.json`。

---

## 11. 数据层文件地图（给审核者定位）

| 事实 | 权威位置 | 现有读取方 |
|---|---|---|
| 花钱 | `runs/<run>/transcript.jsonl`（`result` 行） | `resume.py`、`brief.py`、`observe/detail.py` |
| 发布 | `runs/<run>/result.json` + `confirm_report.json` + `best.metal` | `experiments/common.py:conclusion_of_run`、`observe` |
| 候选/评测 | `runs/<run>/candidates/NNN_label/{kernel.metal,report.json,feedback.txt}` | `resume.py`、`observe/detail.py` |
| 会话 | `transcript.jsonl` 的 `agent` 行 + `agent_work/<label>/` | `brief.py`、`observe/detail.py` |
| 索引 | `data/observe.sqlite`（v4：runs/turns/feedback/...） | `observe/query.py`、`observe/app.py` |
| AI 摘要 | `taic/brief.py`（现读 state/result/transcript） | CLI、agent |
| 计划/反馈 | `feedback.jsonl`、`feedback_claims` | `observe/feedback.py` |
| 确认内核 | `kernels/<target>/<task>/<dtype>/<size>/{best.metal,meta.json,history.jsonl}` | `taic/kernel_catalog.py`、`--from @catalog` |
| 实验 | `experiments/*.py` + 缺失的 `experiment.json` | `experiments/manifest.py`、`observe/index.py` |

---

**一句话给审核 agent**：请优先验证 **F1（钱）**、**F3/F8（每 run/每场固定成本）**、**F4/F5（规模化检索）**，并明确区分「统计上还不够」与「结构上已经错了」。前者等试点，后者现在修。

---

## 12. 二次审核（REVIEW.md）修正与升级

> 这一节以 `REVIEW.md` 为准；上面的正文若与它冲突，按这一节读。

- **F1 重算入口升级**：三个当前协议 run 的差额可以只从 `transcript.jsonl` 的 `agent.tokens` 算出来，与 `result.spent_usd` 对得上（FA-2 分毫不差：记录 13.33343406，agent 成本之和相同）。`ledger.requests` 补不上 token，历史重算不要依赖它。
- **N1**：`taic/committees.py:190` 与 `taic/llm.py:249` 两条计价路径不一致；活动配置写了 `price_cache_write`，委员会成员换 provider 时丢掉。
- **N2**：`CostLedger.add` 入账时丢弃 write token；`charge` 连 hit/miss/out 都写 0。hmz 的 token 只活在 transcript 的 `agent` 事件里。
- **N3**：`experiments.frozen_json` 两行都是 `{}`，`arms=0`，磁盘上 `experiment.json` 为 0；`experiments=2` 不表示有两份可回放的实验说明。
- **N4**：tips 后段会缩短（r15 8,068B → r16 7,947B → r20 8,201B）；瘦身按有上限的摘要写，不假设单调追加。
- **F8 降级**：NOTES 是 4,565B，不是 12,751B；12,766B 的跨度含 tips。
- **F11 降级**：审查页已增量、单卡替换；只留主列表、gzip/ETag。
- **F4/F6 降级**：25s 与 10.7s 未复测，保持 U。
- **F10 口径**：unchanged 在三个当前协议 run 上是 69%、0%、70% 金额占比（合计 22/60 场、44% 金额），不是总体 69%。
- **F14 升级**：kimi 在 60 场里 28 场，记录金额约 80%（$28.98/$36.43）。

### 二次审核对 §7 十二条的判定

1 成立；2 成立；3 分母可用（中位 40/29，大 run 17–32/场）；4 未推翻也未坐实（25s 保持 U）；5 133 个 abi_error 不在索引；6 缓存键含 state/result mtime，10.7s 未复测；7 双读和全局 GPU 事件成立（抽到 1,039/2,454 = 42%）；8 26,616B 与 tips 8,201B 成立，NOTES 12,751B 不成立；9 69% 只属于 conv2d-feedback；10 1.2% 维持 U；11 `runs.model` 只取 `[llm].model`，场次接近一半、金额约 80%；12 40–60MB 与 106MB 是两种保留政策。
