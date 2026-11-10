# P2-F：native64 / V5 / V6 后的最终证据重录

## 范围

本次在用户退出其他程序后的安静窗口重录 P2 golden、14 个域实例的 fresh confirmation、
DispatchManifest 与 active pointer。它绑定当前 native64 Host Program、L2 continuous batching、
V6 task-pack 以及 varlen case-domain confirmation 修复后的代码身份；不调用模型或付费搜索，
不进入 P3，不发布 NPU 模拟性能。

A–E 及其历史结论不改写。本文件只记录这次重录和当前计数。

## 重录结果

依次执行：

```bash
uv run python docs/架构/shape-portfolio-阶段验收/P2/verify.py --write-golden
uv run python docs/架构/shape-portfolio-阶段验收/P2/verify.py --record-evidence
uv run python docs/架构/shape-portfolio-阶段验收/P2/verify.py --write-manifest
```

实测：

- `--write-golden` 成功，candidate view 无 unseen key/fresh seed 泄漏；
- `--record-evidence` 的 5 个 real-Mac confirmation 节点全部通过；
- 12 个 unseen + 2 个 tune，共 14/14 域实例 fresh confirmed，**0 timing-blocked**；
- active DispatchManifest 与 pointer 均为 generation=1、14 subjects；
- manifest 最初写为 27 个文件，加入本 F.md 后为 28 个文件。

当前测试面相对旧验收有真实增长，因此同步 verifier 计数而不是删减测试：

- P0/P1 anchors：54 → **58**；
- P2-B CPU：10 → **11**（新增 varlen 任务按声明 case-domain 确认、不得硬编码 `normal` 的回归）；
- P2-A/B/C/D：9/11/16/10，共 **46** 个 CPU 节点；真实 Mac 节点仍为 5。

## 完整余下复验

默认 verifier 先通过 manifest，随后在历史 start-surface 锚点停止（见下一节）。在不修改该锚点的
前提下，按默认函数的相同顺序逐项执行其余检查，实测全部通过：

- P0/P1 fixed CPU anchors：58 passed；
- repository structure：6 passed；P2-0 anchor：1 passed；
- P2-A/B/C/D：9/11/16/10 passed，合计 46；
- archived 12 unseen + 2 tune：全部 confirmed，0 timing-blocked；
- 新一轮 live 5 个 real-Mac 节点：全部通过；
- live 与 archived 均为 14/14 confirmed，stable confirmation identities 完全一致；
- active DispatchManifest + pointer：generation=1、14 subjects；
- checkpoint/resume/observe 返回 14 个完成实例，revoke generation=2 后拒绝 resume。

没有 skip、xfail、deselect、容差放宽或删除负例。随后完整 CPU 快速套件复跑为
**4549 passed、13 skipped**，此前 partial evidence 导致的 26 个 seed/plan 失败全部消失。

## 有意保留的历史锚点

默认 `verify.py` 仍在 `verify_baseline()` 拒绝，因为 P2 阶段开始时的 45 文件合同面与当前工作树
存在以下 drift：

```text
docs/架构/shape-portfolio实施计划.md  （唯一原本允许漂移的 living plan）
tcenv/backends/metal/launch.py
tcenv/portfolio/candidate.py
tcenv/portfolio/domain.py
tcenv/taskcontract/portfolio_runner.py
tests/shared/test_shape_portfolio_runner.py
tests/support/README.md
```

这是此前已经定位并有意保留的 pre-existing anchor。`acceptance-baseline.json`、
`start-worktree.sha256` 和 `verify_baseline()` 均未修改；不能为了让默认命令退出 0 而扩大允许
漂移集合。因此本次准确结论是：**P2 新证据与全部后续验收通过，历史 start-surface 锚点仍按原样
拒绝当前工作树**。不宣称该历史 drift 已修复，也不把默认 verifier 的这次停止伪报为全绿。

## 信任边界

sidecar/archive 仍不是外部硬件信任根；本轮另有 live fresh execution 与 stable identity 对照。
Qingwei NPU-P0 仍 blocked，NPU 模拟继续保持 `simulated=true`、
`publication_allowed=false`。V6 性能数字只属于 `tests/support/golden/shared/varlen-v6-chunked验收/` 登记的
Torch-MPS baseline 比较，不由本 P2 fixture 发布。
