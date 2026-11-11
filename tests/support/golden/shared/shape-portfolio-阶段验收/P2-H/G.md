# P2-G：逐 launch CommandBuffer + 逐组 SharedEvent 计时协议

## 范围

本阶段把此前只在探针里成立的提交结构改进接入真实 evaluator、worker、`taic run/resume` 配置和
fresh confirmation。新路径名为 `per_launch_set_event`，默认 `single_buffer` 不变；fresh confirmation
使用独立的 **protocol epoch 4** 与 `3+per_launch_set_event` 报告身份，不改写 P2 A–F、P3、P4 的历史证据。

只有单阶段、无计时内 fill、纯 BufferArg、无动态 threadgroup 的候选进入新路径。每次 launch 使用一个
CommandBuffer，第 `i+nsets` 次通过 SharedEvent 等待同一轮转组的第 `i` 次；不同组可以重叠，同一组仍
严格按序。多阶段、workspace 清零、Host Program 和动态 threadgroup 回退原路径，报告记录
`batch_submission=single_buffer_fallback`。事件创建、编码、wait/signal、提交和全部完成等待都在计时窗内。

本阶段是协议与正确性验收，不把某个 speedup 登记为生产发布；两份 softmax 证据均带
`publication_allowed=false`。

## 新 epoch 与历史包边界

- 阶段起点：`37e80f3aaa9c29ccfbb8cd58265f8598437ac578`，新 epoch `p2-g-per-launch-1`。
- 当前任务包仍为 **74 个代码文件**，pack digest 为
  `2ebc22ecc0b598577c3984f45af4e1facc785ca057752e041267a0d1308786eb`，记录在
  `acceptance-baseline.json` 与 `pack-closure.json`。
- 旧 `P2/`、`P3/`、`P4/` 文件未修改。当前 P2 CPU 测试读取本目录的新证据；P3/P4 的历史重放仍读旧目录。
- `activation_expectation` 从 freeze 的 epoch 4 重建同协议 expected，避免 fresh sidecar 正确但激活时
  被 epoch-1 expected 误拒。`V2LocalEvaluator` 的 provenance/protocol 与搜索配置同样区分两条路径。

## 录制的真实证据

依次执行：

```bash
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-G/verify.py --write-golden
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-G/verify.py --record-evidence
```

结果：

- 既有 P2 真实 Mac 节点 **5 passed**，P2-G SharedEvent 节点 **4 passed**；
- 默认域的 12 个 unseen + 2 个 tune 全部 fresh confirmed，**0 timing-blocked**；
- 默认 active DispatchManifest + pointer：generation=1、14 subjects；
- 新协议的 softmax `fp16/paper/golden` 与 `fp16/paper/benchmark` 均 confirmed、identity match，
  新 active DispatchManifest + pointer：generation=1、2 subjects；
- benchmark fresh confirmation：candidate 140.955 µs、MLX 141.605 µs，ratio 1.004609；
- 三组相同 seed 的完整配对评测，中位数从 **0.967986×** 提升到 **1.009056×**：

| seed | 默认单 CB | 逐 launch + 逐组 SharedEvent |
| --- | ---: | ---: |
| 2026111023 | 0.971433× | 1.009056× |
| 2026111024 | 0.967986× | 1.009053× |
| 2026111025 | 0.947949× | 1.012440× |

这些比值只用于证明新协议消除了旧提交结构的损失，不是相对优化 kernel 的通用性能声明。

## 固定负例与运行时检查

- 多阶段、fill、ScalarArg、动态 threadgroup 不具备新路径资格；
- 未知 `batch_submission` 被拒绝，默认 evaluator 仍是 `single_buffer`；
- `taic resume --batch-submission ...` 不能改变旧 run 的协议 epoch，跨协议续跑被明确拒绝，必须新开 run；
- epoch 1 与 epoch 4 的 freeze/validation digest 不同，task-pack digest 相同；
- CPU 假对象逐 launch 检查 SharedEvent 的 wait/signal 代数；
- 真实 Metal 对三组 read_write 缓冲执行六次 `+=1`，每组最终必须恰好为 2，防止把丢更新当成提速；
- 带 fill 的调用在编码前被拒绝，workspace 清零不能被实验路径跳过；
- 配对报告检查 correctness、safety、实际 batch_submission 与 protocol_version，live/archive 比较排除
  微秒数后的稳定身份，不要求两次计时逐字节相同。

## 最终复验

依次执行 `--write-manifest` 与默认 verifier，实测完整通过：

- manifest 覆盖 **50 个文件**，新 epoch baseline 与 74 文件 pack digest 均匹配；
- P0/P1 anchors 58、结构 6、P2-0 anchor 1、P2-A/B/C/D 9/11/16/10，加上本阶段 4 个 CPU 节点，
  合计 **60 个 CPU 节点**（含 `taic` 配置映射、禁止 resume 跨 epoch，以及 P4 历史离线重放 8 个节点）；
- 既有 5 个真实 Mac 节点与本阶段 4 个真实 Metal 节点全部通过，合计 **9 个真实 Mac 节点**；
- archived/live 的默认 12 unseen + 2 tune 均 confirmed、0 timing-blocked，live==archived stable identity；
- 默认 active manifest 为 14 subjects、generation=1，新协议 active manifest 为 2 subjects、generation=1；
- live 配对中位数为 **0.969951× → 1.005571×**，归档配对中位数为 **0.967986× → 1.009056×**，
  两侧均通过 >1 且相对默认提升至少 0.03 的阶段门；
- checkpoint/resume/observe 返回 14 个完成实例，revoke generation=2 后拒绝 resume；
- P4 的 8 个离线重放节点通过，`simulation.json` 未修改；其 freeze 显式固定到 P4 阶段起点 pack，
  不再随 live evaluator digest 改变；
- 无 skip、xfail、deselect、模型调用或付费搜索，旧 P2/P3/P4 文件未修改。

完整 CPU 快速套件最终实测 **5135 passed、13 skipped、0 failed**；既有真实 Metal 搜索回归另行通过，
不以本目录录制代替全套回归。
