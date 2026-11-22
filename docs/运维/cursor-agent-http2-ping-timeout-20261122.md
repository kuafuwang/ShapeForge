# Cursor Agent `[unavailable] PING timed out` 排查（2026-11-22）

## 现象

使用 `cursor-agent/grok-4.7-high-fast` 作为 RLAR actor 时，actor 多次在长时间审查中失败，未产生可交给 reviewer 的答案。hmz 只显示 `Error: [unavailable]`。

底层 worker 日志：

```text
ConnectError: [unavailable] PING timed out
Resetting HTTP/2 session after bridge connection failure
failureReason: rpc_unavailable
```

日志位于 `~/Library/Application Support/Cursor/User/globalStorage/anysphere.cursor-agent-worker/`。同一 worker 反复重连，短时能收到帧，随后再次出现 PING timeout。

## 已排除

- **模型名存在**：`cursor-agent --list-models` 包含 `grok-4.7-high-fast`。
- **已登录且订阅有效**：`cursor-agent status/about` 正常。
- **普通 API 连通**：直连 `https://api2.cursor.sh/` 与 `https://agentn.global.api5.cursor.sh/` 均可得到 HTTP 200。
- **DNS/IPv6 错误不是本次证据**：当时两个 endpoint 都只返回 IPv4，没有 AAAA 结果。
- **不是 hmz 在执行中文任务文本**：`Command '效果边界。'` 是 hmz 对最后一个 `--` 后参数的错误摘要，不是实际命令。

## 当前最可信的解释

失败位于 **Cursor Agent 的持久 HTTP/2 streaming / worker bridge**，而不是普通 HTTPS、账号或模型目录。

证据：

- 短请求（status、list-models、普通 HTTP 200）成功，不代表长流式连接可用。
- worker 反复出现 `PING timed out`、HTTP/2 reset、`ECONNRESET`、TLS 建连中断与 CANCEL。
- 本机 `network.useHttp1ForAgent=false`，即使用默认 HTTP/2 Agent 路径。
- Cursor 官方/员工在论坛中将此错误解释为长连接 keepalive 未按时收到回应；也有多例不同网络的相同报告。
- 当前 CLI 为 `2026.11.13-eacade1`。官方 changelog 在 `2026.11.17` 才加入“HTTP/2 keepalive pings，快速检测并重试静默中断”；本机低于该版本。更新可能改善检测/恢复，但不证明能消除底层断流。

## 与其他因素的边界

- 本地代理 `127.0.0.1:7897` 可用，代理 curl 比部分直连样本更快；但 shell 与系统代理未开启，不能据此断言 hmz/Cursor 一定走了该代理，也不能证明代理会修复持久桥接。
- 多个并行 agent/子 agent 会增加长连接与上传压力；Cursor 论坛有相关失败报告。并行审查可能放大问题，不应把它当作无条件加速。
- 大上下文与长 turn 更容易遇到上传/keepalive 超时；新会话和较短 turn 只能降低暴露，不修复网络。
- 无法仅凭本机证据把责任唯一归为本地网络、ISP、代理、Cursor 服务端或当前 CLI；持续失败应附 request id 给 Cursor support。

## 建议的处理顺序

1. 更新到当前官方 CLI 并彻底重启 Cursor/agent worker。官方 changelog：<https://cursor.com/docs/cli/changelog>。
2. 若仍失败，使用 Cursor 官方网络指引检查 HTTP Compatibility Mode / HTTP/1.1：<https://cursor.com/help/troubleshooting/network>。该模式是绕过 keepalive 的候选，不是已验证修复。
3. 如需代理，只对诊断命令设置一次性代理并核对实际生效，不修改全局 git/系统配置。
4. 降低单 turn 长度与并行度，使用新会话/短任务；需要恢复时用同一 chat 的 checkpoint/resume，不把失败当代码结果。
5. 若仍复现，保存 request id、worker 日志和时间窗口，联系 Cursor support。官方论坛中员工也建议检查网络路径，并确认过长连接/ACP 重试缺口。

参考案例：

- <https://forum.cursor.com/t/connection-error-connection-failed-please-try-again-or-contact-support-if-the-issue-persists/162561>
- <https://forum.cursor.com/t/agent-keeps-a-dead-system-proxy-then-large-chats-die-with-http-2-ping-timeout/178812>
- <https://forum.cursor.com/t/unattended-work-impossible-due-to-cli-failures/173810>
- <https://forum.cursor.com/t/self-hosted-workers-lose-bridge-connection-repeatedly-half-open-ping-timeouts-two-workers-drop-at-once/177847>

## 对 ShapeForge 审核任务的影响

在连接恢复前，Grok actor 的长时间 RLAR 审核不可靠。不能把无答案当成“没有代码问题”，也不能把超时部分结果当完成。可暂时改用其他 actor，或先缩短每个 actor 的阅读/复现任务再让 reviewer 复核。
