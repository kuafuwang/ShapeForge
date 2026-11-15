# 已验证经验与配对搜索准备

P2-S 只做 Plan 3A 的准备工作：把 P2-I 到 P2-R 里核对过的经验收成可审查的搜索输入，并留下一份小额配对实验的 dry-run。没有启动 `taic run`、`taic resume` 或 `taic track-explore --config`。实弹搜索未授权、未执行。Plan 3 没有完成。`publication_allowed=false`。真实 NPU 仍禁止。

`docs/架构/shape-portfolio实施计划.md` 被 P2-I 的 manifest 钉住。本阶段不改那个文件，状态写在 `AGENTS.md` 和本文。

## 为什么不改实时 prompt

`targets/apple-m4-max/experience.md` 在 `current_prompt_sha1()` 里。P2-M 把它接进提示面之后，current 是 `e9d28ff61819a041610d763f5c1464ea860265be2d414f78c15a1f8aea1fdbcf`。P2-N 到 P2-R 的验收都要求现场指纹等于这个值。

如果把新经验写进这个文件，对照臂和既有锚点会一起移动，配对就不再是「只差这一份经验」。所以 P2-S 不新建一个改写指纹的 prompt epoch。记录是：

| 项 | 值 |
| --- | --- |
| previous_prompt_sha1 | `e9d28ff61819a041610d763f5c1464ea860265be2d414f78c15a1f8aea1fdbcf` |
| current_prompt_sha1 | 同上，`prompt_changed=false` |
| experience.md | `3bc1760a4373329dc633a5193286ca41dbee82f8930ecccab959017675b81dba`，未改 |
| 过滤后的对照经验 | 13415 字节，`e972450e333550739378afe9362e17470bc9f59fba089a849835e28402a8fea2` |
| 过滤后的处理经验 | 25469 字节，`8e5adbecd3f764904ceab72c2e35fc7534d08931fde0a24210292cdf23513304` |

处理文本是「现有过滤结果 + 冻结叠加层」的反事实，只存在于 `P2-S/evidence/experience-overlay.md`。现场 `system_prompt_v2` 在两臂之间逐字节相同，而且不含叠加层标记。P2-M 归档的 `gemm` fp16 small 渲染（msl-single 35927 字节、msl-bundle 41218 字节）重新渲染后仍然相符。实验题 `gemm_silu` fp16 small 的渲染是另一份任务文本（36031 / 41322 字节），两臂相同，不能拿去和 P2-M 的 gemm 金样比长度。

旧锚点的行为不变：`tests/shared/test_scratch_prompt.py` 和 P2-M 到 P2-R 的 verifier 继续要求现场指纹等于 P2-M 的 current。本阶段没有改那些测试或历史 manifest。

## 六条可教经验

每条可教经验都要能回到完整 fresh subject，或回到明确标成非候选的报告。标量三臂，以及叠加层里标成 probe 下界或 paired probe 的数字，作为反例上下文保留，但不是独立 teachable lesson，也不是 confirmed speedup。`kda tail` 的决策测量、`kda single` 的 N=1 自检和 P2-R 的 harness 修复写在叠加层里作边界：数值不采用，不作为独立经验。occupied retry 的墙钟和搜索最优倍数不进入叠加层。

1. **conv2d paper 全局 U。** `P2-I/evidence/conv2d-mma-f4-confirmation.json` 里两份 fresh subject：全局 U 617.29 µs，每 threadgroup 重算 1474.42 µs，精确比 2.388525，记到四位是 2.3885 倍。confirmed、identity match、confirmation seed 不是搜索种子 7。这是同一套 F(4×4) simdgroup MMA 的复用，不是发布加速比。更早的标量三臂（17622.82 / 19480.31 / 17473.25 µs）也是 fresh，写在叠加层里说明不能当成「scratch 一定更快」。
2. **conv2d small 保持单阶段。** `P2-J/evidence/small-confirmation.json` 的 fresh 是 3.2373 µs，相对最快基线 0.9858。`launch-floor.json` 的两阶段无 fill 下限 6.1287 µs，`not_a_candidate=true`，是 probe。P2-N 的权重缓存候选 4.0056 µs，比这份 fresh 慢 23.7%，决策是停止。
3. **gemm_silu 把 SiLU 留在片段里。** fresh 来自 `P2-O/evidence/reg-small-confirmation.json`（1.8252 µs，相对最快基线 1.6721 倍）和 `reg-paper-confirmation.json`（1160.87 µs，1.0071 倍）。paper 不要写成稳定的生产加速。GLOBAL 候选没有实现。small 的两阶段下限 8.7713 µs（约 4.8056 倍于 REG）是 probe；paper 高 4.4% 是 stage-2 的流量下界，不是一份 GLOBAL 核的 fresh 时间。occupied retry 不进入叠加层。
4. **bitdelta 在 tile 里解包。** fresh 来自 `P2-P/evidence/reg-small-confirmation.json`（2.1044 µs）和 `reg-paper-confirmation.json`（1219.06 µs）。paper 的同轮配对中位比 1.0259 可以保留。small 的两个最省形态在归档、重放和评审重跑之间翻过，只能教大约一个数量级，不能教精确倍数或排序。这是 paired probe。paper 上 REG 高于硬件下限约 4.6%，不是另一套算法的上限。搜索最优倍数不进入叠加层。
5. **GEMM Split-K 不做。** `P2-L/evidence/launch-floor.json` 的 small 两阶段下限 7.0390 µs，约 3.7514 倍于归档锚点，`not_a_candidate=true`。paper 已有 1024 个输出 threadgroup。10.98% 不是字节流量比，它是 `split2_partial_write_read_us_at_peak` / `existing_candidate_us` 的峰值带宽时间等价。既有 candidate 微秒数是归档锚点，不是本阶段新的 fresh subject。没有新增源码。
6. **SEQ 不要为了并发切链。** 受控变量按路径分开。KDA 六份 fresh subject 的 `device/host_piece_handoff.metal` 都是 `124c7485cbf2feb073e6a245042e09e7f7b9400ee1fd1b17c7543ced5e347b54`。切链在 Host Program `host/tiling.c`：三 piece 是 `369e3b44c4fde94a3c3110b40d78df1f4e2d5722ac15dad22940d29d44bf1d17`，两个单 piece 臂是 `e66406d7ef14ac14c98e41af5c7c124af755558c0233f4ef927d03ee8db7f8a9`。stage 数在 `host/execution_geometry.json`：三 stage 是 `686ce257cdab5e13f217cec008beb143a11fc6f3cf4a68974660979547be5cf4`，单 stage 是 `0b8cfe071fb7ea77c7e15ef0941018c6be016d295b9feb520fa56fd83ff0ca96`。varlen 两份 fresh subject 没有 Host Program。`device/candidate.metal` 分别是 pieces_three `0d214416afa02296691b66083898c1e158b17c4d3b818c8104151006e59631e5` 和 pieces_one `aaa42e0ae1ddfb3a6563c3b3cbcbe783ee7d4f92dbba934a3fc4da3670036f58`。`varlen_source` 的唯一文本差别在 `kernel void varlen_plan` 内，把 `uint pieces = min(MAX_SPLIT, len);` 换成 `uint pieces = 1u;`。P2-Q 合同 §3.1 把 varlen 也写进「三臂 device 源码逐字节相同」；那份文件被 P2-Q manifest 钉住，这里不改。KDA 那句成立，varlen 不成立。八个 fresh subject 都经过确认读取。kda small 与 paper 各有 A、B、C 三臂，三份 confirmation seed 不同。A−C 同时改了 piece 数和 stage 数，是跨 confirmation 的总开销，不能解释为隔离的 handoff 成本，也不是同一次输入的 paired run。small 的 fresh A/B/C 是 246.64 / 224.81 / 205.51 µs，A−C 为 20.0%；paper 是 188422.04 / 187407.45 / 187275.13 µs，A−C 为 0.6%。这个 0.6% 不能外推成 SEQ 便宜。varlen mixed 只覆盖 L1，两臂是 54.9266 µs 对 33.9083 µs（62.0%），stage 数相同，device 字节不同，不含 L2 capacity。决策测量是另一份样本：small 23.0%（handoff 字段 26.0449 µs、stage floor 20.2645 µs）、paper 0.5%、varlen 56.4%。`kda tail` 在 decision.sizes 里，数值不采用、不作为独立经验。`kda single` 不在 decision.sizes，只在 state_algebra 和测量文件的 N=1 自检里（A−B 约 0.8%），数值同样不采用。范围是 Apple M4 Max、fp16。P2-R 的 harness 修复写在叠加层里作边界，`evaluation_device_verdict=false`，不作为独立经验。`handoff_states` 仍在 OpSpec workspace。

叠加层里的每节都写 `publication_allowed=false`。

## 配对方案

这不是 `docs/实验批准单-搜索产出.md` 里那张四开关表。那张表的 δ 仍然空白，名义 $36 的 conv2d + flash_attention 计划也没有获批。P2-S 的唯一变量是经验叠加层。

| 项 | 对照 | 处理 |
| --- | --- | --- |
| 任务 / dtype / size | gemm_silu / fp16 / small | 相同 |
| 搜索种子 | 7 | 相同 |
| 委员会 | taic.toml 的 first（deepseek / deepseek-flash）和 second（kimi-code / k3.1-pro） | 相同 |
| init_samples / task_ledger / plateau_after / sweep_tool | 1 / 关 / 0 / 关 | 相同 |
| 经验文本 | 现场过滤后的 experience.md | 同一过滤结果加上冻结叠加层 |

选择这题是因为 REG fresh 已经快于最快登记基线（1.6721 倍），单次求值只有几微秒，而当前提示已经教了 ScratchSpec 语法，却没有这句「逐元素 epilogue 不要写全局 scratch」。conv2d paper 的正例是真的，但一次确认要几百微秒到几十毫秒。conv2d small 的 fresh 没有快过基线。kda paper 的单 stage fresh 约 187 ms。bitdelta small 的基线差距更大，但短搜索更不容易自己走到 tile 内解包。

计划由这条命令生成，没有 `--execute`：

```bash
uv run python -m experiments.search_yield_ab --dry-run \
  --factor experience-overlay \
  --overlay tests/support/golden/shared/shape-portfolio-阶段验收/P2-S/evidence/experience-overlay.md \
  --tasks gemm_silu --dtype fp16 --size small \
  --pairs 2 --budget-usd 3 --max-total-usd 12 \
  --out runs/p2s-experience-overlay-dry-run
```

归档计划在 `P2-S/evidence/dry-run-plan.json`。2 对、先 AB 后 BA、4 个 cell。每 cell 软预算 $3。`max_total_usd` 与 `hard_cap_usd` 都是 $12，这是是否启动后续 cell 的承诺上限。已经发出的请求可以超过单次软预算，实际账单也可以超过 $12。这不是一张账单绝不会越过的天花板。可容忍超支仍未批准。名义合计仍是 $12。

主指标：完整配对上 `treatment.confirmed_speedup - control.confirmed_speedup` 的中位数。`best_speedup` 不能填补缺口。次指标沿用 `experiments/search_yield.py` 的 `lift_median`、`unchanged_fraction`、`spend_after_best_median`、`self_checks_median`。预先写下、但 dry-run 没有测量的量是「求值候选里声明了 scratch 的比例」。

停止条件：δ 空白，或完整配对少于 2，结论都是 inconclusive。汇总器默认的 `allowed_regression=0.0` 不是 owner 批准，`allowed_regression_approved` 保持 null。`P2-S/evidence/approval.json` 的批准人、δ、实验 ID、GPU 窗口和执行记录都是 null。`paid_search` 是「未授权/未执行」。

`--execute` 在四道人工门禁都齐的时候仍然返回 2。原因是叠加层没有安装，两臂会看到同一段提示。本阶段没有添加一条绕过 fingerprint 的注入路径。

## 还不能做的事

- 不能把这份 dry-run 说成一次已花费的搜索，也不能把复核本身的成本算进实验。
- 不能填写四开关批准单的 δ，也不能用那张 $36 的表代替本协议。
- 不能把这份文件直接写进 `experience.md` 然后开跑。叠加层的 verified 小节已经带 `本机实测:`、边界，以及 `docs/` 来源，反事实合并能通过现有的 verified 条目合同。装进实时提示仍会移动 `current_prompt_sha1`，对照臂必须继续看到现在的过滤经验。本阶段没有这条注入路径，所以 `--execute` 仍拒绝。
- 不重签 trust，不改六 RPC / HMAC / BlobChannel，不放宽正确性、安全或 1.10 性能门。
