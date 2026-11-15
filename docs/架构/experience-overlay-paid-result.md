# 经验叠加层实弹配对的结果

P2-U 记录已经跑完的经验叠加层配对。题目是 `gemm_silu`、fp16、small，两对，冻结协议。主指标是完整配对上 `treatment.confirmed_speedup - control.confirmed_speedup` 的中位数。δ = 0.01，`allowed_regression=0`。两对差是 -0.10295850321802247 和 -0.19655229305915878，中位差是 -0.14975539813859062，`excluded_pairs=[]`。`experiment_complete=true`，`complete_pairs=2`，`verdict=no_gain`，`adopt_overlay=false`，`publication_allowed=false`。

这次 gemm_silu/fp16/small、两对、冻结协议下未获增益。不是「经验永远无用」的普遍证明。处理臂的 `lift_median` 更高，主指标仍然失败，次指标不能改写 `verdict=no_gain`。Plan 3 的这次实验已执行且未通过。Plan 4 未开始。NPU 仍是 simulation-only。

验收从 `runs/p2u-experience-overlay-paid-20261114` 的 run 文件重算。`summary.json` 只作为对照，不能单独充当结论。P2-I 到 P2-T 的 manifest 保持原字节。本文不改 P2-S、P2-T 的历史文档。

## 四格 subject

| 格 | confirmed | 来源 SHA-256 | identity | run_seed | 花费 |
| --- | --- | --- | --- | --- | --- |
| control rep1 | 1.8415238580482542 | `e890effe3df919c67aa23858dc2a28327eb4edc2387c3d3c803ceada4b37b910` | match | 7 | 2.92720628 |
| treatment rep1 | 1.7385653548302318 | `b9b34b66f7eb3e3e56fe2bab82668a68d335f24fce63d831c09f61d10debacd6` | match | 7 | 3.665203884 |
| treatment rep2 | 1.564200200680351 | `d40fbf255f3cded6d75393ca1024131ddf3a2134a80fd8d9b3abe75032aa7f45` | match | 7 | 4.5453007240000005 |
| control rep2 | 1.7607524937395098 | `ea555f09478262f05d85700cbcc9efe8523e930058d2eda1013548f7d5595c17` | match | 7 | 2.9565494880000003 |

四格 `confirmed_status=ok`。来源 SHA-256 等于该格 `best.metal`，也等于 `confirmations/fp16_small_benchmark.json` 的 `candidate_sha256`。subject 是 `identity.kind=match`、`differences=[]`、`provider=candidate`、`credits_variant=true`、`fallback_id=null`、`status=confirmed`。胜者分别是 control rep1 的 `r3.p0`、treatment rep1 的 `r1.p0`、treatment rep2 的 `r1.p0`、control rep2 的 `init`。搜索期的 `best_speedup` 与 fresh confirmation 的 `confirmed_speedup` 不是同一个数，配对只用后者。

每格有 5 条账本请求，scratch 里有 init 到 `r4.p0` 五份会话归档。`result.json` 的 `rounds` 字段在两格 control 是 5，在两格 treatment 是 4；这个字段不是请求数，也不是归档数。

## 种子

两臂 `run_config.json` 的 `seed` 和 `run_seed`，以及 `result.json` 的 `run_seed`，都是 7。trace 里 `eval_seed_from=search` 的 `eval_seed` 全部是字面量 7。四格 subject 的 `confirmation_seed` 分别是 4035910615072932、8117866098860839、4608897102258473、5487670343179405。这是 fresh confirmation 的派生种子。`eval_seed(run_seed, phase, label, source)` 的返回值也不是这颗搜索字面量。确认种子不拿来冒充 `run_seed`。

## 叠加层

会话正文留在每格 `scratch/*.tar.gz` 的 `experience.md`。对照臂五份归档都是 13415 字节、标记 0 份、SHA-256 `e972450e333550739378afe9362e17470bc9f59fba089a849835e28402a8fea2`，包内没有 `experience_overlay.json`。处理臂五份归档都是 25469 字节、标记恰好 1 份、SHA-256 `8e5adbecd3f764904ceab72c2e35fc7534d08931fde0a24210292cdf23513304`，包内 sidecar 与该 run 的 `experience_overlay.json` 和 `run_config.json` 顶层记录一致。

四格系统提示相同，36031 字节，SHA-256 `b8446c3bfcececdd708561bc8d3e34365044320c9c291efed3812b0da8965f35`，不含标记。`agent_guidance/experience.md` 与全局 `targets/apple-m4-max/experience.md` 相同，也不含标记。冻结叠加层文件仍是 SHA-256 `8e3b9f975cb3988e03af09a12e131bfdb8a699631c8780d544d3836f4bdb97cc`。搜索开关在四格都是 `init_samples=1`、`task_ledger=false`、`plateau_after=0`、`sweep_tool=false`。委员会仍是 first `deepseek-flash` 与 second `k3.1-pro`。

## 次指标

| 臂 | lift_median | unchanged_fraction | spend_after_best_median | self_checks_median |
| --- | --- | --- | --- | --- |
| control | 0.03607214678631787 | 0.0 | 0.7204149125972352 | 2.0 |
| treatment | 0.1331584579867846 | 0.0 | 0.9120297976368839 | 2.5 |

处理臂 `lift_median` 较高，只说明它相对自己的 init 抬得更多。两对 confirmed 差值都为负，中位差低于 δ，`verdict=no_gain`。

## 费用

四格 `result.json` 的 `spent_usd` 与各自 `ledger.json` 的 `total_usd` 相等。按 JSON 数值的十进制文本相加是 14.0942603760000008，与账单口径 14.094260376 的差小于 1e-12。后两格文本里的 `0000005` 和 `0000003` 是这两个浮点的原始字面量。`experiment_commitment` 读正式根得到 14.094260376000001，与这四格之和的差小于 1e-9。

中断根 `runs/p2u-experience-overlay-paid-20261114-interrupted` 的 commitment 是 0。`recovery-failed-1` 的 `spent_usd` 是 0，账本请求是空的，也是 0。这两处不重复计入 14.094260376。treatment rep1 的恢复费用 0.21717865199999994 是该格账本的 init 行，已经含在 3.665203884 里，不再另加。

## 恢复链

treatment rep1 的父编排器在 init 期间被工具中断。原目录整份移到中断根。那里的 `state.json` 是 `spent_usd=0`、`evaluations=0`、`best_label=null`、`trace=[]`，没有 `result.json`。init 目录里有 self-check 日志和候选，候选 SHA-256 是 `1a3056224676f7fbc3208ba98349781bc0974d3465a68338d9bd0c3a18833cb4`。self-check 不是正式结果。

`recover_init.py` 对这同一份候选字节调用 `prepare_session` 和 `collect_session`，`new_model_call_for_init=false`。重建账单是 cost `0.21717865199999994`，input 99045、output 96468、cache_read 11950592、cache_write 0。这次调用做出了正式评测和 fresh confirmation，然后因预算耗尽停下。停下来的快照在 `result.before_topup_1.json`：init 的搜索 `best_speedup` 是 1.4571061913744205，fresh `confirmed_speedup` 是 1.429754492820249，来源 SHA 仍是 `1a305622…`，identity match。这是恢复后的中间确认，不是配对使用的最终 subject。

随后 `resume --add-budget 2.782821348` 把该格从 0.21717865199999994 恢复到原来的 $3。最终 subject 是 `r1.p0`，confirmed 1.7385653548302318，来源 `b9b34b66…`。

第一次恢复脚本的 `copytree` 撞上已经存在的 `.selfcheck`，现场在 `recovery-failed-1`。transcript 记录 `File exists`，第二委员会随后被「只接受第一次 init」拒绝。该目录 `stop_reason=agent_failure`，`evaluations=0`，`confirmed_speedup=null`，花费 0。没有 GPU 评测，没有计费。

这条恢复是非标准的孤儿 init 回放。配对仍然成立：候选字节在孤儿、失败尝试和正式 init 提交里相同；处理臂每次会话归档恰好一份叠加层，对照臂为零；恢复账单只在 treatment rep1 的 init 行出现一次；配对数字是 resume 之后的 fresh confirmation；没有恢复的 rep2 差值同样为负。`recovery.breaks_pairing=false`。

## 承诺上限

原批准的承诺上限是 $12，单格软预算 $3，δ = 0.01，`allowed_regression=0`，批准人是 kuafu。`plan.before-cap-increase.json` 仍是 $12。前三格实际承诺是 11.137710888。再按整格 $3 启动 control rep2 会到 14.137710888，`fits_cap(..., 12)` 拒绝，这就是当时的 `over_cap`。

用户只批准把承诺上限提到 $15，用来完成剩下的 control rep2。`approval-cap-increase.json` 写明 `previous_cells_topped_up=false`、`no_additional_cells=true`。当前 `plan.json` 和 `cap.json` 是 $15。`fits_cap(..., 15)` 允许启动最后一格。实际总账单低于 $15。前三格没有因为这次提额再加预算。treatment rep1 更早的 add-budget 只是把被恢复预算截断的格子补回原来的 $3。

两格 treatment 的在途超支都发生在 `r4.p0`：rep1 从 2.3429444839999998 加到 3.665203884，rep2 从 2.092686724 加到 4.5453007240000005。两格 control 停在 $3 以内。这是原来的软预算规则已经允许的在途超支，文件里留着。

`plan.json` 仍写着 `executed=false`、`paid_search=未授权/未执行`、`delta=null`。原始 `approval.json` 仍写着 `executed=false`，同时保留 δ = 0.01 和 $12。执行记录是四份 `confirmed_status=ok` 的 `result.json`。汇总使用批准单上的 δ = 0.01，不读取 plan 里的空 δ。

## 仍受的限制

- `publication_allowed=false`。confirmed 数字是这次配对的主指标，不是发布速度。
- Plan 3 的这次实验已执行且未通过。Plan 4 未开始。
- NPU 仍是 simulation-only。
- 负结果的范围只到这次题目、两对和冻结协议。
- 恢复路径不是普通编排器收尾。候选、叠加层、账单和正式评测可以追溯，所以这一对保留；它仍然要和最终 subject 分开读。
- 提示面未改。`current_prompt_sha1` 仍是 `e9d28ff61819a041610d763f5c1464ea860265be2d414f78c15a1f8aea1fdbcf`。任务包仍是 79 个文件，`b5b74185710e5c64cf55ed20cc472f3258e0fe2b1c631bb7f850eeb8dcbb069e`。

重算后的完整 subject、文件 SHA-256 和恢复判断在 `tests/support/golden/shared/acceptance-epochs/fixtures/p2u/conclusion.json`。
