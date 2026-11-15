# P2-U：经验 overlay 实弹配对

- epoch：`p2u-experience-overlay-paid-1`
- parent HEAD：`a9c1dccce636107c2852995578b05611bf19865b`
- prompt：`e9d28ff61819a041610d763f5c1464ea860265be2d414f78c15a1f8aea1fdbcf`，`prompt_changed=false`
- pack：79 文件，`b5b74185710e5c64cf55ed20cc472f3258e0fe2b1c631bb7f850eeb8dcbb069e`
- `experiment_complete=true`，`complete_pairs=2`，`verdict=no_gain`，`adopt_overlay=false`
- `publication_allowed=false`；NPU 仍是 simulation-only
- Plan 3 的这次实验已执行且未通过。Plan 4 未开始。

实验根是 `runs/p2u-experience-overlay-paid-20261114`。中断现场是 `runs/p2u-experience-overlay-paid-20261114-interrupted`。题目是 gemm_silu、fp16、small，两对。两对差是 -0.10295850321802247 和 -0.19655229305915878，中位差是 -0.14975539813859062。总账单是 14.094260376。δ = 0.01，`allowed_regression=0`。

这次 gemm_silu/fp16/small、两对、冻结协议下未获增益。不是「经验永远无用」的普遍证明。处理臂较高的 `lift_median` 不改写主指标。treatment rep1 的孤儿 init 恢复可以追溯，候选 SHA 从 `1a305622` 起；self-check 和 resume 前的 init 确认都不是配对数字。原承诺上限 $12 挡住了最后一格，批准后的上限是 $15，前三格不提额。

合同见 [经验叠加层实弹配对的结果](../../../../../../docs/架构/experience-overlay-paid-result.md)。

## 验收

```bash
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-U/verify.py
```

验证器从四份 `result.json`、确认 subject、账本、scratch 会话归档、恢复文件和两份 plan 重算。`summary.json` 必须与这次重算一致。6 个 CPU 节点，不跑 GPU，不调用模型，不提高预算：

- `tests/shared/test_p2u_experience_overlay_paid.py::test_paired_confirmed_speedups_are_no_gain`
- `tests/shared/test_p2u_experience_overlay_paid.py::test_secondary_lift_does_not_override_the_primary`
- `tests/shared/test_p2u_experience_overlay_paid.py::test_identity_run_seed_and_search_seed`
- `tests/shared/test_p2u_experience_overlay_paid.py::test_overlay_copies_come_from_session_archives`
- `tests/shared/test_p2u_experience_overlay_paid.py::test_cost_excludes_interrupted_and_failed_recovery`
- `tests/shared/test_p2u_experience_overlay_paid.py::test_cap_history_and_recovery_chain`

独立复核重算配对、费用、恢复和批准边界。满足后 `review.json` 的 `done=true`。P2-I 到 P2-T 的 manifest 不改。
