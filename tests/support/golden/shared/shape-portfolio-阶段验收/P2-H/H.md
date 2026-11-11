# P2-H：ABI 失败必须是候选判决，不能打崩整个搜索

## 范围

P2-G 的付费配对冒烟暴露了两层问题：候选 stage 的 buffer 槽位没有从 0 连续编号，触发正常
`abi_error`；随后 runner 又因为 `NO_DISPATCH_STATUSES` 漏掉 `abi_error`，把候选失败升级成
`TaskContractError`，使 control run 在 r2 评测中异常退出，treatment 没有启动。

本阶段只修复 runner 的分类：`abi_error` 与 `compile_error` 一样，是首次 dispatch 之前的正常候选
判决，worker 的 `bounded-v2-worker` provider 和空 dispatch 不再被解释为执行了几何 fallback。
新增 `tests/shared/test_shape_portfolio_runner.py::test_an_abi_error_is_a_typed_candidate_failure_not_a_runner_crash`，
覆盖有完整 build、零 dispatch、worker provider 的真实形状。

## pack 与历史证据

`tcenv/taskcontract/portfolio_runner.py` 不在 74 文件的 deployment pack 闭包内，因此本修复的 pack digest
仍为 `2ebc22ecc0b598577c3984f45af4e1facc785ca057752e041267a0d1308786eb`。
`acceptance-baseline.json` 以 `3b83e2773914063ecbe1590d407640be56722db9` 为父提交，建立
`p2-h-abi-error-1` 验收 epoch。P2-G/P2-F/P2-A–E、P3、P4 的历史文件未修改。

P2-G 的 14 个默认域实例与 softmax epoch-4 证据在该 pack 下仍然有效，本目录保留同一证据集并另行执行
live 验证，不重写旧目录。录制和 verifier 统一使用与 `confirmation_master_seed(7)` 相同的公开验收 key，
避免整数 7 与派生字节 key 混用造成 validation-plan 漂移。新增 CPU 回归使 P0/P1 anchors 从 58 变为 59，
总 CPU 节点为 **61**；真实 Metal 节点仍为 **9**。

## 候选侧复现与修正（不属于生产性能发布）

出错的 r2 候选位于本次付费 run 的工作目录。`conv2d_wgrid` 的 `[1,2]`、`conv2d_relocate` 的 `[2]`
已分别改为从 0 开始的连续槽位。槽位修正后又发现 restore 的 IEEE 快速路径会产生 NaN；强制 restore
走直接卷积后，正式 `--quick` 六例全部通过，才允许计时。

修正版正式 full evaluation：candidate 894.993 µs、Torch MPS 768.269 µs、ratio **0.858407×**，
correctness/safety 均通过。该值只用于分析候选结构，不作为生产 speedup 发布，也不填补付费实验的
未确认样本。其性能解释是：restore 的直接回退牺牲快速路径，五阶段 U 搬运和没有独立 workspace 的
重组装成本仍在，因此虽然优于 r1 的 0.750611×，仍未达到 1.0。

## 验证

默认命令：

```bash
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-H/verify.py
```

验证 manifest、pack 闭包、61 个 CPU 节点、9 个真实 Metal 节点、默认 14 subjects 与 softmax 2 subjects
的 active manifest、live/archive 稳定身份、P4 历史重放和 checkpoint 生命周期。默认 verifier 已完整通过；
归档配对中位数为 0.973020× → 1.012535×，live 配对中位数为 0.972692× → 1.009667×，fresh benchmark
confirmation 为 1.014475×。这些是 P2-G 协议证据的复验，不是新增 speedup 发布。完整 CPU 快速套件另行运行。
