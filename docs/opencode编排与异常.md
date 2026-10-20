# OpenCode 2.0.32：一轮怎么编排，交互异常怎么收

这篇记的是别人的编程代理 OpenCode，不是本仓库的搜索循环。本仓库读者带走三句：直调按流接收，空闲超时，还没结束或正文为空时整份重发，截断的工具调用不执行。从中断处接着写、以及先把整段历史压成摘要，没有搬进来。细节在文末。

这篇对过 `/Users/kuafu/workspace/opencode-2.0.32`，包版本 `2.0.32`。那个检出不在本仓库里。下面的路径、重试次数和压缩窗口是他们的。

OpenCode 是一个编程代理框架。用户发一句话，框架调模型，模型可以再调工具，工具结果回到同一条会话，直到这一轮不再需要模型。一次交互拆成三层：

| 层 | 目录 | 这一层做完就停在哪 |
| --- | --- | --- |
| 请求 | `packages/ai` | 把供应商的 HTTP / SSE / WebSocket 收成事件流。失败收成 `AIError`，这里不重试 |
| 循环 | `packages/core/src/session/runner` | 一步一次模型请求。工具在同一步里并行。瞬时失败、输出截断、溢出补压都留在同一步里 |
| 会话 | `packages/core/src/session` | 收件箱、压缩、选模型。失败后不会自动换下一个目录模型 |

会话里的结束原因只有这几种：`stop`、`length`、`tool-calls`、`content-filter`、`error`、`unknown`（`packages/schema/src/llm.ts`）。循环是否再问，看的是有没有未结束的本地工具，不看 `tool-calls` 这个字符串本身。

## 一轮怎么走

入口是 `packages/core/src/session/session.ts` 的 `Session.prompt`。它先准备附件，再把一条 user 放进 inbox，然后 `execution.wake`。同一条会话同时只有一个执行 fiber：忙的时候只记一次 doorbell，当前 drain 结束再 drain，不开第二条并行执行（`packages/core/src/session/run-coordinator.ts`）。

消息先只在 inbox 里。`SessionRunner.drain`（`packages/core/src/session/runner/llm.ts`）把可以提升的 input 写成会话消息，再进循环。默认 delivery 是 `steer`，插进当前意图；`queue` 是下一轮，要等这次 continuation 结束才提升。压缩会插到同边界更早的 steer 前面。

每一次发给模型之前，顺序是：

1. 先试自动压缩。压成功就换一条新的 assistant id，再转一圈；压失败则这一步直接 `StepFailedError`，不再调主模型。
2. `SessionModelRequest.baseTranscript` 组 system 和消息。历史经 `toLLMMessages` 降成模型能看的格式。
3. `session.context` hook 可以改 system、messages、tools、`maxTokens`。不支持的媒体换成错误文本。内联图片超过 25 MiB，从旧到新删到约 15 MiB。
4. `LLM.stream` 编译请求并发出。主循环用的是流，不是把流折成一次 `generate` 的旁路。

`step` 从 1 起。新的一轮用户输入会把它打回 1。外层 `while (true)` 在 `runStep` 返回「还要继续」时 `step++`。自动压缩、重试、输出截断续写、溢出补压都留在同一次 `runStep` 里，不增加 `step`。

一轮 assistant 回来之后：

| 条件 | 接着做什么 |
| --- | --- |
| 本步有本地工具调用，且 `toolChoice` 不是 `"none"` | 外层再走一步，把工具结果交给模型 |
| `finish` 是 `length`，并且还有未完成的本地工具 | 同上，算新的一步。半截工具不执行 |
| `finish` 是 `length`，没有未完成工具 | 同一步里插入一句 synthetic，要求把剩下的工作拆小，换新的 assistant id 再问 |
| 最近 3 条 user/assistant 里，assistant 的 finish 全是 `length` | 停。文案是两次续写之后仍然截断 |
| 流在已有输出之后被打断，且判定可重试 | 这条 assistant 标失败，退避后插入「从上次中断处继续」，换新 assistant |
| 还没有任何输出，且可重试 | 同一条 assistant、同一份消息再发 |
| 溢出且这次补压成功 | 同一步重来。这一 step 里溢出只补压一次 |
| `content-filter` | 助手标失败，本步结束 |
| 正常 `stop`，且没有本地工具 | 回到收件箱。没有待提升项，这一轮结束 |

没有进程级的步数上限。`Agent.Info.steps` 是可选正整数，内置 agent 不写死它。配了 `steps: N` 时，第 N 步仍带工具定义（为了缓存前缀），但 `toolChoice` 设成 `"none"`，并加一句只许用文字总结的提示。模型若仍发出 tool call，执行函数返回工具错误，并且不再继续。

工具在同一步的流里并行启动：每来一个非托管的 `tool-call` 就 fork 一条 fiber，runner 没有并发上限。下一步请求要等 `Fiber.awaitAll`。托管工具（`providerExecuted`）由流里的 `tool-result` 记账，本地不执行。流被打断时，正在跑的工具 fiber 一并打断。

## 模型怎么交给这一次请求

每一步都重新解析，不缓存上一次的模型对象。会话没选模型就走默认模型，仍没有则 `ModelNotSelectedError`。选了但当前目录里找不到，是 `ModelUnavailableError`，不会改用另一个能用的。variant 不存在也直接失败，不退回 default。

思考档和 effort 来自 variant 叠进模型的 settings / headers / body，不是 runner 每个请求单独传一个参数。会话层每次请求显式放进 generation 的，主要是 `maxTokens`：夹在目录输出上限、1 024 和 256 000 之间，再按上下文剩余空间缩小。

切模型是显式的 `Session.switchModel`。provider、id、variant 都没变就直接返回。下一次 `load` 读到新的 `session.model`。

失败之后不会自动换下一个目录模型。相近的只有传输降级：WebSocket 连接失败，或同一条 socket 连续丢交换 5 次，这条会话改走 HTTP，直到重启或 move。同一条 socket 每 55 分钟轮换一次，空闲 30 分钟关掉。这换的是连接，不是模型。

## 错误怎么分类，什么会重试

分类在 `packages/ai/src/provider-error.ts` 的 `classifyProviderFailure`。证据按这个顺序用，先命中先返回：HTTP 状态，JSON 里的 code / type，再是文案正则。`x-should-retry` 不参与分类，只在 `isRetryable` 里覆盖：`true` 强制可重试，`false` 强制不重试。

| 类别 | 结果 | 会话层 |
| --- | --- | --- |
| 限流 | `RateLimit`（429，或限流 code / 文案；429 上的额度文案先被收成额度） | 重试 |
| 服务端暂时故障 | `ProviderInternal`（408、409、5xx，或无状态时的过载文案） | 重试 |
| 传输 | `Transport`。未送出、或已接受但读失败，可重试；对端拒绝则不可 | 按上面这条 |
| 流没收到终态 | `InvalidProviderOutput` + `incomplete-stream` | 重试 |
| 认不出 | `UnknownProvider` | 重试。注释写明：认不出的形状默认当暂时故障 |
| 鉴权 | `Authentication`（401、403，或错 key 文案） | 停 |
| 额度 | `QuotaExceeded`（402，或额度 code；仅 429 再加额度文案） | 停 |
| 内容策略 / 拒绝 | `ContentPolicy` | 停 |
| 非法请求、上下文溢出、请求体过大 | `InvalidRequest` | 停。溢出另走压缩 |
| 调用方截止时间 | `Timeout` 这个 tag | 停。这是媒体轮询的截止，不是 HTTP 空闲 |

HTTP 层发一次就停。`packages/ai` 里没有 LLM 的 `maxRetries`。重试发生在这一轮流结束之后，由 `packages/core/src/session/runner/retry.ts` 决定。

默认：

| 项 | 值 |
| --- | --- |
| 第一次请求之后最多再试 | 10 次（`Schedule.recurs(10)`） |
| 退避 | 2 秒、4 秒、8 秒，之后每次封顶 10 秒 |
| 抖动 | `Schedule.jittered`，大约落在名义间隔的 0.8 到 1.2 |
| `Retry-After` | 只对限流和服务端错误。取指数间隔和它的较大者，封顶 15 分钟。10 秒的间隔上限压不住这个头 |
| 传输且 `code === "Timeout"` | 最多再批准 3 次 |
| 关掉某一次 | `session.retry` hook 可以把 `decision.retry` 改成 false |

等响应头的空闲默认 300 000 毫秒，SSE 块与块之间的空闲默认 900 000 毫秒（`packages/ai/src/schema/options.ts` 的 `DEFAULT_HTTP_TIMEOUT_MS`、`DEFAULT_CHUNK_TIMEOUT_MS`）。整次请求的 `timeout` 不设就没有墙钟。单独吃「最多 3 次」的，是 `Transport` 且 `code === "Timeout"` 的那一种。

还没有任何 text、reasoning 或 tool 输出时，可重试失败复用同一条 assistant、同一份消息再发。等待期间错误记在 `assistant.retry` 上，下一次 `step.started` 清掉 error / finish / retry。这条当时是空的，组请求时模型看不见这次失败。

已经出了字再遇到可重试失败：半截留在这条 assistant 上并标失败，退避后插入 synthetic：「The previous response was interrupted. Continue from where you left off without repeating completed content.」然后换新的 assistant。这是让模型接着写，不是从最后一个 SSE 事件续传。限流出现在半截输出之后也走这条。

OpenResponses 的 WebSocket 续传被拒时另有一次立即重发：`previous_response_not_found` 记 `retry-full`，连接数打满记 `rotate-and-retry-full`。还没出字时不走退避，立刻整包重发，并清掉 checkpoint、关掉这条 socket。这一 step 只给这一次。第二次再被拒就失败，不进入那 10 次退避。

## 取消和半截消息

用户取消是 fiber interrupt，不是 `AIError`。`Step.Started` 之前、起始 snapshot 也还没好，什么都不记。已经开始写 assistant 之后：已发布的 text / reasoning / tool input 留在会话里；未结算的工具写成 `Tool execution interrupted`；assistant 写成 `Step interrupted`。默认 interrupt 不 `resume`，不会自动再跑模型。退避睡眠中途取消，不再发下一次请求。

下次用户请求会带上这条已经结束的半截 assistant。它带 `error` 时，思考正文改成普通 text 再发给模型，provider metadata 不复用。工具错误仍是 tool result。

进程死后还停在 streaming / running 的工具，下一次 `drain` 开头写成 `Tool execution interrupted: <name>`。组请求前，`normalizeToolHistory`（`packages/ai/src/tool-history.ts`）给还没有 result 的本地调用补一条错误结果，正文是 `Tool result missing`。托管工具不补。空输出改成 `(no tool output)`。

## 上下文和截断

溢出检测在分类器里：code 如 `context_length_exceeded`、`model_context_window_exceeded`、`request_too_large`，或 “prompt too long / context length” 一类文案，收成 `InvalidRequest` + `context-overflow`。限流、以及「`max_tokens` 设得比模型上限还大」的 `too many tokens`，不收成溢出。`packages/ai` 只打标，不压缩。

压缩在 `packages/core/src/session/compaction.ts`。运行时初值是自动压缩开着，最近原文保留 15 000 token。配置可以改 `auto`、`keep.tokens`、`buffer`。

每一步模型调用之前先看用量。天花板是：

- 配了 `buffer`：窗口减去 `buffer`。
- 没配：窗口至少 32 000 时，留出 `max(窗口的 10%, 16 000)`；更小的窗口只留 10%。
- 窗口未知（输入上限和上下文都 ≤ 0）：天花板是无穷大，自动压缩不触发，只能等服务商拒绝。

估计量用最近一条本服务商、且没有 error 的助手消息的 input、cache、output、reasoning，加上这之后的文本估计。最后一条已经是完成的压缩就跳过，避免连着压。

`auto: false` 时，自动压缩和溢出补压都不做，只剩用户的 `/compact`。

服务商在还没有可见输出时返回溢出：同一步里 `compact({ reason: "overflow" })` 一次。成功就重来；这一 step 里第二次溢出不再压，错误写上助手消息。已经写出 reasoning 或正文之后，不再为这次溢出去压。溢出本身不进那 10 次重试。

摘要默认用当前会话模型。隐藏 agent `compaction` 若单独配了模型且能解析，才用那个。本地摘要要求回复带模板里的 `##` 标题。第一次没有，就再发一句提醒，只此一次。第二次仍不对、正文为空、`length` 或内容过滤，记一条 `failed`，不把半截摘要入库。可重试的服务商错误在压缩发送循环里走同一套 10 次退避。压缩请求自己再被溢出或请求体过大拒绝时，按第一次被拒请求的估计缩到 70%、50%、35%，三次缩完再被拒就放弃。

压缩不删 `session_message` 行。之后的模型请求从最近一条已完成压缩读起。本地摘要变成一条 user 消息，里面是 `<summary>`，可选再加 `<recent-context>`（`keep` 范围内的最近原文，序列化成文本）。原始行仍留在表里。

`finish === "length"` 不走压缩。半截正文已经写进当前助手消息。没有未完成的本地工具时，插入 synthetic，要求把剩下的工作拆小，同一步里再问。只允许两次这样的续写；第三次助手消息仍是 `length` 就失败。半截 tool call 不执行，标 `tool.input-incomplete`，附上最多 2048 字符的参数摘录，然后进入下一步，把失败的工具结果交给模型，不再插那句拆小的 synthetic。

空的成功 `stop` 当正常结束，主对话不会因此再请求。只有 reasoning、以 `stop` 结束，也当这一步完成；reasoning 已把 `outputStarted` 设为 true，所以不能再走「没出字就透明重试」。压缩若只有 reasoning、没有模板正文，提醒一次后按「没有摘要」失败。

Anthropic 的 `stop_reason === "refusal"` 在进会话之前映射成 `content-filter`。OpenAI 的 refusal 增量写成普通 text。`content-filter` 把助手标失败，这一步结束，不续写，也不重试。

## 工具失败、权限、子代理

未知工具、参数不是合法 JSON、JSON 合法但 schema 不对、工具抛错、规则 `deny`，都写成 `status: "error"` 的 tool result，下一步再问模型。发出去之前会做一轮确定的类型修补（字符串数字、多余字段）；修不好才落到 schema 错误。

用户拒绝且没有反馈：工具记成「The user declined this tool call」，assistant 标 `Step interrupted`，然后 interrupt 本轮，不立刻再问模型。拒绝带反馈则收成工具错误，模型继续。

Shell 的非零退出和命令超时是成功的 tool result，正文里带退出码或超时说明。job 变成 `error` 或 `cancelled` 才是工具失败。

子代理是另一个会话。前台等它结束：失败或取消写成当前工具的错误结果，带上 child 的 session id，父回合继续，工具里没有自动重跑。后台立刻返回「正在后台工作」，结束时往父会话写一条 synthetic。父工具被取消时会 interrupt 孩子。进程重启时，还在跑的后台孩子可以 resume，最多 10 次；超限就通知父会话不再自动恢复。这是把被挂起的执行接上，不是孩子失败后再跑一遍。

## 一张总表

| 发生了什么 | 会话里留下什么 | 接着做什么 |
| --- | --- | --- |
| 还没出字的限流、5xx、不完整流、认不出的错误 | 同一条空 assistant，错误记在 `retry` 上 | 同一请求再发，最多 10 次，2s / 4s / 8s 后封顶 10s |
| 已经出了字之后的同类错误 | 半截标失败 | 退避后加一句「从中断处继续」，换新 assistant |
| 鉴权、额度、内容过滤、非法请求 | 失败 assistant | 这一轮结束 |
| 上下文超窗，且还没有可见输出 | 历史从新的摘要检查点读起，旧行仍在表里 | 同一步补压一次再请求；再溢出就停 |
| 用量越过天花板 | 同样换成摘要加最近原文 | 这一步在调主模型之前就压 |
| 输出顶到 `length`，没有未完成工具 | 半截留着 | 同一步再问，最多两次；第三次仍截断就停 |
| `length` 且工具参数被截断 | 工具错误结果，不执行 | 下一步把结果交给模型 |
| 工具抛错、参数不对、未知工具、规则拒绝 | 一条错误 tool result | 下一步再问模型 |
| 用户拒绝且无反馈、或按停止 | 半截 assistant 标 `Step interrupted` | 本轮结束，等下一条用户消息 |
| 孤儿 tool call | 会话里可能仍缺结果 | 下次组请求时补 `Tool result missing` |
| WebSocket 连不上或连续失败 5 次 | 模型不变 | 这条会话改走 HTTP |

## 本仓库后来采纳的部分

上面是 OpenCode 在 2.0.32 上的做法。2026-10-20，直调 `taic/llm.py` 采纳了其中能放进一次聊天请求的几条：按流接收，空闲超时，还没有结束原因或正文为空时再发同一份消息，截断的工具调用不执行。自动换模型、整段历史摘要和「从中断处继续」没有搬进来。现行规则在 [架构说明.md](架构说明.md) 的 §2.4。这篇不批准一次新的付费搜索。
