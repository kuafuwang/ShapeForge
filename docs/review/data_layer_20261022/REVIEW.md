# 数据层二次审核

**审核对象**：`docs/review/data_layer_20261022/`（README §5 F1–F15、§7 十二条、§8 口径）
**审核时仓库**：`69bbe49`（审核包抬头写的是 `ace4ac6` + 3 个未提交文件；这 3 个文件已进入 `69bbe49`。工作区只剩未跟踪的 `docs/review/` 和 `docs/target_profiles_6f03f75e.plan.md`）
**日期**：2026-10-22
**约束**：未跑 `taic run` / `taic resume`，未启动 GPU kernel，未调用 LLM。`repro.sh` 退出码 0，写入只在临时目录。`SLOW=1`（`feature_grid` 10.7s）本轮没跑。

## 审核结论

- 总评: **PASS-WITH-CHANGES**
- 覆盖范围: F1–F15 + 12 条挑战
- 一句话：钱的漏洞、每场复制、候选事实不进索引、协议不能混算，这几条在当前代码和现有 run 上成立，现在就该修。`10^5` 次请求 25s、`feature_grid` 10.7s、GPU 利用率 1.2% 仍是外推或未复测。NOTES「12.7KB」把 tips 算进去了。

| ID | 结论 | 证据（命令/行号/数字） | 修正 |
|----|------|------------------------|------|
| F1 | PASS | `taic/committees.py:190-199` `_priced` 只写 `price_in/out/cache_read`。`load_config`（`taic/llm.py:249`）会写 `price_cache_write`，第二委员会走 `_priced` 时 `replace` 留下第一委员会的价。价表：deepseek-flash `cacheWrite=0`，kimi-k3 `cacheWrite=3`。FA-2 记录 `$13.3334`，kimi `cache_write=1,624,052` × `$3/M` = `$4.872`，重算 `$18.206`。conv2d-feedback 记录 `$15.220`，重算 `$20.539`。conv2d-flame 记录 `$7.874`，重算 `$10.735`。三份 `ledger.json` 的请求行全是 `hit=miss=out=0`（FA-2 21/21，feedback 25/25，flame 14/14）。 | 漏的是第二委员会的 cache-write 单价，第一委员会 deepseek 的 0 和价表一致。历史重算要读 transcript 的 `agent.tokens`，ledger 请求行里没有 token。 |
| F2 | PASS | `repro.sh`：transcript 759.7MB / 61.5%，最大 `p1/conv2d_paper/transcript.jsonl` 92.82MB，其中 chat 91.95MB。`taic/llm.py:769-776` 仍把 `messages`+`reply` 写入 `event=chat`。4 个 `flame_chase_1` transcript：1038B、5514B、13626B、10683B，事件种类只有 `agent`。 | 「hmz 10–13KB/run」只盖住两个大 run。四个当前协议 run 的范围是 1.0–13.6KB。zstd「约 6%」本轮未再压。 |
| F3 | PASS | `repro.sh`：全集中位 40、有 `agent_work` 的 13 个 run 中位 29；conv2d-feedback 797/25=31.9，FA-2 558/21=26.6，conv2d-flame 239/14=17.1。全部名为 `v` 的目录合计 296 文件 / 83.7MB。仓库 Python 里没有读取这些 `agent_work/**/v/` 的调用。 | `p1` 有 1523 个文件且 `agent_work=False`，全集中位 40 会被大量小 run 拉低。当前协议用「文件/场」这组数字。 |
| F4 | PASS | `observe/query.py:41-49` `history()` 先 `sync_known()`，再 `SELECT row_json` 后整表 `json.loads`。`observe/index.py:407-417` `tick()` 每帧扫目录 mtime；`query.py:464` 在流里调用。52 run 时的 14ms / 25s 外推本轮没有造 `10^3`/`10^4` 目录去测。 | 结构是 S1，25s 仍是 U。优先级留在规模化项里，不要把它写成已经测到的秒数。 |
| F5 | PASS | `candidates/**/report.json` 748 份：`abi_error` 133、`incorrect` 84、`compile_error` 29。索引表没有 `candidates`。`feedback_items` 36 行，`kinds LIKE '%abi%'` 为 0。索引有 `runs_status/target/experiment`，没有 `runs(task/model/dtype/publication)`，没有 FTS5。 | 「`status != 'missing'` 使复合索引失效」过了一步：schema 里还没有那个复合索引。缺的是候选表和这些列的索引。 |
| F6 | PASS | `feature_grid`（`observe/query.py:93-108`）缓存键含 `state.json` 与 `result.json` 的 `mtime_ns+size`（`:445-452`）。`index.html:1091` 渲染 `feature_grid`。同文件搜 `timeline` / `confirmations` / `.llm` / `trace` 无匹配。`repro.sh`：conv2d-committees 的 `run_detail` 44.7KB / 356ms。 | 10.7s 与「读 95.5MB」本轮未复测（没跑 `SLOW=1`）。 |
| F7 | PASS | `taic/brief.py:100` 与 `:116` 各走一遍 transcript（`_transcript_counts`、`_truncation_retries`）。`:119` 把 `tcenv` 全局 GPU 事件塞进 `gpu`。`repro.sh`：conv2d-committees 2454B，8 个事件 1039B（42%）。 | 39/73% 那两个 run 本轮没再跑 brief。hmz transcript 只有 1–14KB，双读在 hmz 上不是 149MB 那个量级。 |
| F8 | PASS | FA-2 `run_config.search.peer_tips=true`。`init/TASK.md` 10946B；`r20.p0/TASK.md` 26616B。其中 peer 块（「本任务前序避坑提示」到 `# 证据`）**8201B**；`# r19.p0 NOTES` 到 peer 头 **4565B**，对应 `r19.p0/NOTES.md` 4564B。同 run tips：r1 1875B → r15/r16 约 8.0KB → r20 8201B；r16（7947）小于 r15（8068）。 | 「NOTES 12,751B」不成立。12,766B 是 NOTES 嵌入段加 tips 整段。tips 大体变长，后段有替换，不是每一场都比上一场更长。 |
| F9 | PASS | `experiments/run_suite.py:66-69`、`p1_runs.py:149-157`：`read_result` 为空就 `run_one`。`taic/orchestrator.py:376-377`：非 resume 且目录非空则 `RunStartError`。 | 「FA-2 winner 测了 5 次」本轮未逐次对账，记 UNKNOWN。控制流本身成立。 |
| F10 | PASS | 431 份带 `benchmark.candidate_serial_us` 的 ok report，时间加总：candidate 33.8% / serial 35.2% / baselines 31.1%（审核包写 33/34/33）。`candidate_serial_us` 的读者在 `tcenv/evaluate.py` 写出，搜索决策路径不读它。unchanged：feedback 15/25 场、`$10.46/$15.22`；FA-2 0/21、`$0`；flame 7/14、`$5.51/$7.87`。三 run 合计 22/60 场、记录金额 `$15.97/$36.43`（44%）。 | 69% 是 feedback 这一场的金额占比。三个当前协议 run 合在一起是 37% 场次、44% 金额。GPU 1.2% 未复测，保持 U。 |
| F11 | PASS | `index.html` 仍内联，`observe/app.py` 无 gzip/ETag。主列表 `host.innerHTML = html`（约 1186 行）。审查页已按条替换：`observe/feedback.py:735-750` 在 `since` 等于当前 revision 时返回空，否则只给 `c.rev > since`；`index.html:450` 只改一张卡。 | 审核包里的「feedback 整表 341KB、每条都重建」对审查页已经过时。gzip、ETag、主列表整表重绘还在。 |
| F12 | PASS | `du`：`data/deleted` 266MB。`resume.lock` 31 个，31 个 pid 都已不在（`os.kill(pid,0)` → `ProcessLookupError`）。 | 删除日志是否被 gitignore、`best.metal` 覆盖历史，本轮未再开目录核对，沿用审核包。 |
| F13 | PASS | `runs/**/experiment.json` 数量为 0。`experiments` 表仍有 2 行：`apple-m4-max/p1`（driver `p1_runs`，`frozen_json={}`，status done）和 `repeat_20261007-015329`（driver `repeatability`，`frozen_json={}`，interrupted）。`arms` 0。 | 这两行是目录壳，不是一份可复现的 manifest。 |
| F14 | PASS | `observe/index.py:794`：`runs.model` 取 `run_config.llm.model`，否则 result/state。`repro.sh`：deepseek-flash 49、kimi-k3 2、deepseek-v4-pro 1。三个大 run 的 agent 事件：kimi 28/60 场，记录金额 `$28.98/$36.43`（80%）。 | 「大约一半」说的是场次。金额上 kimi 是这 60 场里的大头，因为单价是 `$15/M` 输出对 deepseek `$1.2/M`。 |
| F15 | PASS | `runs` 表无 `protocol_version` / `metric_protocol`。`repro.sh`：50 个 `legacy/chat`、4 个 `flame_chase_1`。publication：confirmed 10、in_progress 10、不确定 2、旧版未绑定源码 28、未确认 2。 | 跨 `d3bd2dc` 的计时窗口差异本轮未逐行 diff。缺列这一点足够禁止把速度和 confirmed 数混在一个平均数里。 |

## 被推翻/降级/升级的结论

- **降级 F8 的 NOTES 数字。** `r20.p0` 文件 26,616B 和 tips 8,201B 成立。嵌进 TASK 的上一轮 NOTES 是 4,565B，源文件 `NOTES.md` 是 4,564B。12,751B 对不上任何一个单独块；从 NOTES 标题到 `# 证据` 的 12,766B 已经把 tips 算进去了。
- **降级 F11 的审查页描述。** 主列表仍整页 `innerHTML`，响应仍无 gzip/ETag。审查 feed 已按 `feedback_revision` 增量，单卡替换。不要再按「341KB 全量快照」排审查页的工期。
- **降级 F4/F6 的秒数。** O(全部 run) 的 `history()`/`sync_known()`，以及 detail 上现算 `feature_grid`，代码都在。25s 和 10.7s 没有在这次审核里重测。
- **升级 F1 的重算入口。** 三个当前协议 run 的差额可以只从 `transcript.jsonl` 的 `agent.tokens` 算出来，和 `result.spent_usd` 对得上（FA-2 分毫不差：记录 13.33343406，agent 成本之和相同）。`ledger.requests` 补不上 token。
- **升级 F14 的成本口径。** 索引里 kimi 只有 2 个 run；那 60 场里 kimi 是 28 场、记录花费的 80%。
- **§8「中位 398」这条修正成立。** `repro.sh`：全集中位 40，有 `agent_work` 的 run 中位 29，三个大 run 17.1–31.9 文件/场。

## 新发现

- **N1 · `taic/committees.py:190` 与 `taic/llm.py:249` 两条计价路径不一致。** 活动配置加载时写入 `price_cache_write`；委员会成员换成另一 provider 时丢掉这个字段，继承 deepseek-flash 的 0。kimi 的 `price_in/out/cache_read` 是对的（FA-2 上用继承的 0 去重算，和记录值一致，补上 `$3/M` 才出现 `$4.87`）。
- **N2 · `CostLedger.add`（`taic/llm.py:387-394`）入账时丢弃 write token。** `cost_of` 乘了 `price_cache_write`，请求行只存 `hit/miss/out`。`charge`（`:397-401`）连这三项都写 0。hmz 的 token 只活在 transcript 的 `agent` 事件里。
- **N3 · 实验行是空壳。** `experiments.frozen_json` 两行都是 `{}`，`arms` 为 0，磁盘上 `experiment.json` 为 0。`experiments=2` 不表示有两份可回放的实验说明。
- **N4 · tips 后段会缩短。** FA-2：r15 与 r14 都是 8068B，r16 降到 7947B，r20 再到 8201B。瘦身方案要按「有上限的摘要」来写；只说「单调追加」会估错节省。

## 建议优先级调整

P0 保持，顺序改成：

1. **计费（F1 + N1/N2）。** 给 `_priced` 补上 `price_cache_write`（与 `llm.py:249` 同一表达式）。新的 `charge`/`add` 记下 write token 和 `cost_basis`。三个已结束 run 用 transcript 重算，单独写下 `cost_recomputed`，不要改原来的 `spent_usd`（预算门已经按旧数停过：feedback 在记录值 `$15.22` 上 `budget_exhausted`，重算是 `$20.54`）。
2. **TASK 瘦身（F8，按修正后的体积）。** 常量契约约 11KB 每场都在；真正随场变大的是 tips，r20 为 8.2KB。NOTES 全文约 4.6KB，不是 12.7KB。目标仍可以是 ≤12KB/场，账要按这两块来减。
3. **候选索引 + 请求路径不要扫全表（F4/F5）。** 先做 `candidates` 表和 keyset。`10^5` 的 25s 用 `/tmp` 合成目录测完再写进验收表。
4. **批量幂等和 experiment manifest（F9/F13）。** 目录非空即拒绝、`experiment.json` 缺失，这两条与样本量无关。
5. **agent_work 上限（F3）。** `v/` 的 83.7MB / 296 文件没有读者，可以先归档再挪 cwd。

试点保持 S2，不要用这 4 个 `flame_chase_1` run 定 unchanged_patience 或委员会性价比。unchanged 的金额占比在三个 run 上是 69%、0%、70%。GPU 利用率 1.2% 在有新定义下的测量之前，不作为 P0 验收。

P1 里把「审查页全量刷新」拿掉。留下：主列表虚拟化、gzip/ETag、`brief` 的全局 GPU 事件和双读、transcript 正文默认不落盘（只影响以后的 chat 路径；现有 760MB 是 legacy）。

## §7 十二条，逐条

1. **成立。** `_priced` 没设 `price_cache_write`。FA-2：1,624,052 × $3/M = $4.872，记录 $13.333，重算 $18.206，与 `result.spent_usd` 13.33343406 一致。
2. **成立。** 759.7MB 里最大文件 99% 是 `chat`。四个 hmz run 的 transcript 是 1–14KB 的 `agent` 行。
3. **分母可用。** 中位 40 / 29，大 run 17–32 文件/场，与 `repro.sh` 一致。`v/` 296 文件、83.7MB，代码侧无读者。
4. **25s 未推翻，也未坐实。** 本轮没造合成 run。代码路径是全表 `row_json` 加每帧 `tick`。秒数保持 U。
5. **133 个 abi_error 不在索引里。** `feedback_items.kinds` 没有 abi。`runs(task)` 索引不存在。
6. **缓存键含 state/result 的 mtime 和 size。** `feature_grid` 有前端读者；timeline/llm/confirmations/trace 没有。10.7s 未复测。
7. **双读和全局 GPU 事件成立。** 抽到的 brief 是 1039/2454 = 42%。另两个百分比未复测。hmz transcript 很小，慢的是旧 chat 文件。
8. **26,616B 与 tips 8,201B 成立。NOTES 12,751B 不成立**（见上）。tips 从 1.9KB 长到 8.2KB，中间有一次缩短。
9. **69% 只属于 conv2d-feedback 的金额。** 三个当前协议 run：22/60 场，记录金额 44%。
10. **1.2% 维持 U。** 本轮没重算墙钟和计时窗口。serial 约占计时和的 35%，这条与利用率定义无关，成立。
11. **`runs.model` 只取 `[llm].model`。** 场次接近一半（28/60）；记录金额约 80% 在 kimi。
12. **40–60MB 与 106MB 是两种保留政策，本轮没有重新压缩。** `data/deleted` 266MB 在 `data/` 下，不在 `runs/` 的 1,234MB 里。XML 332.4MB 在 `runs/` 里；要不要留原始 XML 是保留政策，不是这次测量能决定的。

## 复现

```bash
cd /Users/kuafu/CLionProjects/ai_lowerling
bash docs/review/data_layer_20261022/repro.sh   # exit 0，约 4s
```

计费重算读的是 `runs/apple-m4-max/{flash_attention-20261022-2,conv2d-feedback-20261021,conv2d-flame-20261020}/transcript.jsonl` 的 `event=agent`，单价来自本机 `~/.pi/agent/models.json` 的 `cost` 字段（deepseek-flash cacheWrite 0，kimi-k3 cacheWrite 3）。脚本没有打印密钥。
