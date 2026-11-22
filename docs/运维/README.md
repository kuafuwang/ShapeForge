# 运维与事故记录

这里放**运行期事实**：某次长任务为什么卡住、某个配置为什么改了、当时看到的是哪些数字。
它不是设计文档，也不是验收证据——设计在 `docs/架构/`，验收在各自的阶段目录。

放这里而不是删除的理由：这类记录里写的是"当时真实发生了什么"，删掉之后下一个遇到同样现象的人
只能重新踩一遍。但它也不该待在仓库根目录冒充入口文件。

## 约定

1. 一篇一次事故，文件名 `agent_debug-YYYYMMDD.md`，日期用事发当天。
2. 正文保持原样归档：不美化、不补写当时不知道的结论。要更新就新写一篇并互相链接。
3. 每篇必须在这一页有一行，写清**现象一句话**和**结论一句话**，否则它又是一份没人会翻到的文件。
4. 涉及凭据、token、内部地址的内容先脱敏再放进来。

## 索引

| 日期 | 文件 | 现象 | 结论 |
| --- | --- | --- | --- |
| 2026-10-29 | [agent_debug-20261029.md](agent_debug-20261029.md) | RLAR 的 actor 反复自报 "context window is exhausted"，133 turns 后 `no turn produced an answer` | pi 的自动压缩按约 983K 阈值判断（`contextWindow 1M - reserveTokens 16K`），而 actor 在约 446K 就开始自报耗尽；改为给 `deepseek-flash` 配 `reserveTokens: 500000`，并明确"fresh actor = 新 run 开新会话，不是每轮新建"，进度落在工作树和阶段文档里 |
| 2026-11-22 | [cursor-agent-http2-ping-timeout-20261122.md](cursor-agent-http2-ping-timeout-20261122.md) | Grok actor 长时间 RLAR 反复失败，只有 `[unavailable] PING timed out` | 普通 HTTPS、登录和模型目录正常，失败在持久 HTTP/2 Agent/worker 桥接；优先升级并重启，必要时测 HTTP/1.1、缩短 turn 和降低并行，不能把无答案当审核完成 |
| 2026-11-17 | [gpu_bif0_freeze-20261117.md](gpu_bif0_freeze-20261117.md) | 整机 GUI 完全冻结、只能强制关机；事后发现当天 GPU 被 `conv2d --dtype fp16` 的 kernel 打崩 5 次，累计 8 次 | GPU 执行时读地址 `0x0` 触发 `BIF0 page fault`（signature 563）→ GPU 硬件 reset → WindowServer 主线程阻塞在 `IOGPU/AGXG16X` 提交路径 80 秒 → 看门狗杀掉 WindowServer → 桌面冻结。不是硬件坏；本机跑新候选前必须开 `MTL_DEBUG_LAYER=1`，并发压到 1–2，`budget.duration` 压到 1–2 分钟 |
