# 运维与事故记录

这里放**运行期事实**：某次长任务为什么卡住、某个配置为什么改了、当时看到的是哪些数字。
它不是设计文档，也不是验收证据——设计在 `docs/架构/`；验收证据在 `docs/可信harness/` 和 `tests/support/golden/shared/acceptance-epochs/`，已移出工作区的阶段包按 `tests/support/golden/shared/retired.json` 里的恢复命令取回。

放这里而不是删除的理由：这类记录里写的是"当时真实发生了什么"，删掉之后下一个遇到同样现象的人
只能重新踩一遍。但它也不该待在仓库根目录冒充入口文件。

## 约定

1. 一篇一次事故，文件名 `<slug>-YYYYMMDD.md`（`<slug>` 用一句话点名故障，现有 `agent_debug`、`gpu_bif0_freeze`、`cursor-agent-http2-ping-timeout`），日期用事发当天。
2. 正文保持原样归档：不美化、不补写当时不知道的结论。要更新就新写一篇并互相链接。
3. 每篇必须在这一页有一行，写清**现象一句话**和**结论一句话**，否则它又是一份没人会翻到的文件。
4. 涉及凭据、token、内部地址的内容先脱敏再放进来。
5. 时间线要带**可复核的时钟锚点**：当场贴 `date`、`sysctl -n kern.boottime`、`last reboot`，并 `stat` 最后写过的文件。「等了约 N 分钟」这类事后回忆不能单独当观测值——[gpu_bif0_freeze-timeline-20261126.md](gpu_bif0_freeze-timeline-20261126.md) 就是一条回忆与三个锚点对不上的实例。
6. 正文按第 2 条冻结；发现事实错误时新写一篇复核并互相链接，不就地改旧正文。

## 索引

（术语：表里的 RLAR 是一个 agent 写、每轮换一个新 reviewer 只读复核的循环；actor 是其中写代码的那个会话，turn 是这个会话里的一次完整作答。）

| 日期 | 文件 | 现象 | 结论 |
| --- | --- | --- | --- |
| 2026-10-29 | [agent_debug-20261029.md](agent_debug-20261029.md) | RLAR 的 actor 反复自报 "context window is exhausted"，133 turns 后 `no turn produced an answer` | pi 的自动压缩按约 983K 阈值判断（`contextWindow 1M - reserveTokens 16K`），而 actor 在约 446K 就开始自报耗尽；改为给 `deepseek-flash` 配 `reserveTokens: 500000`，并明确"fresh actor = 新 run 开新会话，不是每轮新建"，进度落在工作树和阶段文档里 |
| 2026-11-22 | [cursor-agent-http2-ping-timeout-20261122.md](cursor-agent-http2-ping-timeout-20261122.md) | Grok actor 长时间 RLAR 反复失败，只有 `[unavailable] PING timed out` | 普通 HTTPS、登录和模型目录正常，失败在持久 HTTP/2 Agent/worker 桥接；优先升级并重启，必要时测 HTTP/1.1、缩短 turn 和降低并行，不能把无答案当审核完成 |
| 2026-11-17 | [gpu_bif0_freeze-20261117.md](gpu_bif0_freeze-20261117.md) | 整机 GUI 完全冻结、只能强制关机；事后发现当天 GPU 被 `conv2d --dtype fp16` 的 kernel 打崩 5 次，累计 8 次 | GPU 执行时读地址 `0x0` 触发 `BIF0 page fault`（signature 563）→ GPU 硬件 reset → WindowServer 主线程阻塞在 `IOGPU/AGXG16X` 提交路径 80 秒 → 看门狗杀掉 WindowServer → 桌面冻结。不是硬件坏；本机跑新候选前必须开 `MTL_DEBUG_LAYER=1`，并发压到 1–2，`budget.duration` 压到 1–2 分钟 |
| 2026-11-26 | [gpu_bif0_freeze-timeline-20261126.md](gpu_bif0_freeze-timeline-20261126.md) | 复核 11-17 冻结的时间线：原文「等了约 10 分钟」与「14:48 重启」不能同时成立 | 用 `last reboot`、冻结前 uptime（→14:45:48）与最后一次写盘（14:45:51）定出：系统在 14:45:51 仍活着，14:48 已重启，等待最多两分钟出头；原文因果链与处置不变，缺的是当场记录而不是复现事故 |
