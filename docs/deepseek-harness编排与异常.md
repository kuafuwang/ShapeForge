# DeepSeek Harness：一轮怎么编排，交互异常怎么收

这篇记的是别人的编程代理 DeepSeek Harness，不是本仓库的搜索循环。本仓库的回合、预算和评测仍由 `taic` 排。实验脚本里的直调采纳的是按流接收、900 秒块间空闲、空回复整份重发、顶到输出上限时留下正文且不执行截断的工具调用。他们的重试次数、压缩比例和取消协议没有搬进来。

这篇对过 `/Users/kuafu/workspace/deepseek-harness` 本地检出 `687f282e86`（2026-10-17，合并进 `dsh` 0.2.1-rc.2）。那个检出不在本仓库里。行号指那一版。

下面说的「模型」是被那个框架调用的服务。用户发一句话，框架调模型，模型可以再调工具，工具结果回到同一条会话，直到这一轮不再需要模型。

## 一轮怎么走

入口在 `packages/core/agent-loop/src/agent.ts`。

用户消息走 `followup`，放进下一轮的队列并唤醒驱动器。驱动器空闲时才开工；已经在跑就不再开第二个。驱动器的 `kick` 反复调用 `turn`，`turn` 再一步步调用 `step`。

一步里面做这些事：

1. `preStep` 从收件箱领消息。一轮的第一步把已经排好的「下一步」消息全部领走，再加一条新的用户回合；之后的步只领「下一步」。
2. 组装系统提示。`pre-step` 可以拒绝这一步，拒绝则本轮以 `blocked` 结束，不调模型。
3. 把系统提示和已经接纳的用户消息写入会话，再从会话表面导出这次请求的消息。
4. `prepareCall` 按 provider 字符串找到适配器，然后只消费流。DeepSeek 适配器把线协议写成 `stream: true`（`packages/llm/llm-deepseek/src/serialize.ts`）。
5. 流正常结束，整段压成一条 `assistant/message` 写入会话。流上的每个块只在进程内广播，不按块落盘。
6. 正文里的 `tool-call` 交给 `executeToolCalls`。每个结果写成 `tool/result`。下一步导出消息时，它们变成 `role: 'tool'`。
7. 还有工具调用、且没有工具声明 `concludesTurn` 时，同一次 `turn` 再走一步。没有工具调用，或某个工具声明这一轮到此结束，并且「下一步」队列是空的，本轮结束，原因是 `completed`。

循环里没有步数上限。`packages/core/agent-loop/src/constants.ts` 里和调度有关的默认值只有并行工具上限：同时在飞最多 10 个（`DEFAULT_MAX_PARALLEL_TOOL_CALLS`）。这是并行池的大小，不是一条回复里工具调用个数的上限。

工具是否并行，看工具自己的 `isConcurrencySafe`。返回值不是严格的 `true`，或者分类函数抛错，都按独占处理。独占调用会先等前面的并行池排空。

工具抛错、参数校验失败、未知工具名、权限拒绝，都写成 `isError: true` 的工具结果，正文以 `Error:` 开头，然后同一步结束后再请求模型。这些失败不换模型，也不重发刚才那次模型请求。

系统提示、历史、工具清单分三处进入请求：

| 内容 | 进请求的方式 |
| --- | --- |
| 系统提示 | `SystemPrompt.assemble` 后写入 `system/message`，再进 `messages`。空的系统消息不进模型历史 |
| 历史 | 只来自会话表面上的 system、user、assistant、tool、developer。失败尝试 `assistant/attempt` 不进历史 |
| 工具清单 | 放在请求的 `tools` 字段。工具增减另记一条 developer 消息。不写进系统提示正文 |

编排层不 import 具体供应商。每一步都重新 `prepareCall`，注册表里这个 provider 字符串指向谁，就用谁的 `stream()`。`llm-deepseek` 和 `llm-pi-ai` 都是适配器。循环自己不保存「当前模型」覆盖；换模型是 `packages/core/agent/src/model-selection.ts` 里的 `installModelSelection`，它改下一步请求的 provider / model。`DEFAULT_MODELS` 里的名字是目录，不是循环自动选中的默认模型。

## 传输失败：什么重试，什么立刻停

重试插件在 `packages/llm/llm-retry`。默认策略 `normal` 的数字在 `packages/llm/llm/src/retry-policy.ts`：

| 项 | 默认 |
| --- | --- |
| 第一次请求之后最多再试 | 5 次（一共 6 次请求） |
| 本地退避起点 | 500 毫秒 |
| 本地退避上限 | 10000 毫秒 |
| 抖动 | ±0.1 |
| 可重试的码 | `EMPTY_RESPONSE`、`RATE_LIMIT`、`SERVER`、`TIMEOUT`、`TRANSPORT` |

计数在 `step/start` 或 `turn/end` 清零。`always` 策略除了调用方已经取消、插件已经销毁，什么码都重试，并且没有次数上限。

退避是指数的，指数封在 1024。服务商的 `Retry-After` 只在 DeepSeek 的 HTTP 头里读：纯数字当秒，否则当日期。解析出来的等待落在 10000 毫秒以内就原样等，不加抖动；比上限还长时，`normal` 不再重试。`llm-pi-ai` 的失败对象没有这段等待，重试只用本地退避。

DeepSeek 的 HTTP 分类在 `packages/llm/llm-deepseek/src/transport.ts`，先匹配先生效：

| 条件 | 码 | `normal` 下 |
| --- | --- | --- |
| 401、403，或认证 / 权限类错误 | `AUTH` | 停 |
| 402，或余额、配额文案 | `QUOTA` | 停。429 若正文是余额不足，也走这里，不走限流 |
| 429，或限流类错误 | `RATE_LIMIT` | 重试 |
| 上下文超窗文案 | `CONTEXT_WINDOW_EXCEEDED` | 停，交给压缩插件 |
| 400、413，或非法请求 | `INVALID_REQUEST` | 停 |
| 5xx，或过载 | `SERVER` | 重试 |
| 流里的 error 事件、上面都没中 | `SERVER` | 重试 |
| 其余，例如裸 404 | `HTTP_<状态>` | 停，不换模型 |

两条适配器在失败后都不改模型、不改 endpoint。

超时方面，DeepSeek 传输层没有单独的连接超时、首字节超时或整次请求超时。`fetch` 只带取消信号。真正的时钟是流空闲看门狗：这一次 `next()` 还没交出下一块，就一直计时，建连、等响应头、等第一块、块与块之间的静默都算。默认 `300000` 毫秒（`packages/llm/llm-deepseek/src/defaults.ts` 的 `DEFAULT_STREAM_IDLE_TIMEOUT_MS`）。SSE 事件和注释会重新计时。到点是 `TIMEOUT`，按默认可重试。`llm-pi-ai` 同样只有这只空闲看门狗，默认也是 300000 毫秒。图片走 Files API 时另有 60000 毫秒截止；超时后同一请求改走 base64，不进 llm-retry。

流要走到 `finish`，并且原因不是 `error` 或 `aborted`，才算成功回合。失败写 `assistant/attempt`，不写进模型能看见的历史，也不执行工具。重试时再从会话表面导出消息，请求体与上一次相同，半截输出不塞回去。

几种断开的差别：

| 现象 | 码 | 默认 |
| --- | --- | --- |
| 套接字掐断、连接重置 | `TRANSPORT` | 重试 |
| 对端干净结束，但还没有 `message_stop` | `STREAM_CLOSED` | 不重试 |
| SSE 帧 JSON 非法、字段坏 | `MALFORMED_RESPONSE` | 不重试 |
| 已经正常结束，工具参数不是合法 JSON | `MALFORMED_RESPONSE` | 整次丢掉，不写成工具结果 |
| 正常 `stop` 但没有任何内容 | `EMPTY_RESPONSE` | 重试 |
| 用户取消 | `ABORTED` | 不重试 |

空闲超时先于取消信号判断。信号已经 abort 时，不会把用户取消收成 `TRANSPORT` 再重试。退避等待中途取消，不再发下一次请求。

## 上下文和这一轮的正文

### 超窗

两条路都有，实现在 `packages/compaction/compaction-basic`。

请求之前，`pre-step` 按压力决定要不要压缩。默认：窗口的 0.8 作为压力阈值，再留出 65536 token 的余量；保留尾部约占 0.16；摘要调用的 `maxTokens` 默认 65536。token 数优先用上一次成功请求的 usage，否则用估算。低于阈值就不压。

仍超阈值时，可以先剪工具结果的中间，再量一次。还超，就调用摘要模型（`purpose: 'compaction'`）。摘要重试默认 `compactionRetries = 1`，加上第一次一共两次。摘要自己被截断、没有正文、带图片，或摘要并不比被替换的那段更短，就不落地。这条路径上的失败只记日志，回合继续去请求模型。

服务商明确返回 `CONTEXT_WINDOW_EXCEEDED` 之后，压缩插件不看阈值，再压一次并重试原请求。`maxOverflowRetries` 默认 1：成功恢复后再遇到超窗，就把原来的错误抛出，回合以 `error` 结束。表面没有任何变化时，不占用这次重试额度。

压缩改的是会话表面：被替换的那一段换成一条摘要用户消息。旧事件仍留在日志里，下次发给模型的视图不再带它们。系统消息若在表面的第 0 条，不进入被替换范围。尾部至少留 1 个节点，并且不把一次工具调用和它的结果切开。

### 输出被 max_tokens 截断

半截正文和 reasoning 写入 `assistant/message`。这一步直接以 `max-tokens` 结束，不自动续写，也不执行工具。所有工具调用块，包括已经完整的，都从落盘正文里丢掉。若丢掉之后正文为空，这条仍在日志里，但下次请求看不到它。

### 空回复和只有推理

空回复是 `EMPTY_RESPONSE`，走上面的传输重试，最多再试 5 次。不把「你刚才没说话」写回给模型。用尽后回合以 `error` 结束。

只有 reasoning、没有正文，算成功的 `stop`。没有工具调用就结束这一轮，不会再问一次。正文是散文、不是工具调用时，同样当普通完成。

工具参数在本层 `JSON.parse` 失败时，保留原始字符串，校验失败后写成 `Error: invalid arguments: ...` 的工具结果，下一步发给模型。DeepSeek 在不是 `max_tokens` 的情况下，工具 JSON 非法则是 `MALFORMED_RESPONSE`，回合直接错误结束，不写成工具结果。

### 工具结果太长

三套阈值各写各的：

| 机制 | 默认 | 留下什么 |
| --- | --- | --- |
| 压缩前的字符剪枝 | 超过 8192 个码位才剪；头 4096，尾 1024 | 中间换成 `[... tool result middle pruned ...]`。改的是会话表面 |
| spill | 基础部署 `maxInlineTokens: 12500`。不配置则不挂 | 全文落盘，上下文留头尾，并写上文件位置。落盘失败则保留原文 |
| 图片预算 | DeepSeek 默认上下文窗 1000000、输出上限 256000、单次 inline base64 20MiB、最多 600 张 | 超预算时把最老的仍保留图片换成占位文字再重试 |

`read` 不走 spill。

## 人按停止、权限、子代理、命令

用户取消调用 `cancel({ kind: 'user' })`：清空收件箱，abort 当前信号。已经生成的非空正文和 reasoning 写入 `assistant/message`，并标 `interrupted: true`。半截工具调用丢掉。什么都还没生成时，只写不进模型历史的 `assistant/attempt`。

已经开始的工具会收到 abort，调度器等它停完再提交结果。成功返回会被换成 `Error: tool call aborted`。还没派发的写成 `Error: tool call aborted before dispatch`。前台 shell 会杀掉进程。本回合到此结束，不会因为这次停止再请求模型。用户下一条消息才会开新回合，并把留下的前缀和工具结果带上。

权限在 `packages/interaction/user-approval`，不在 `packages/guard`。拒绝是工具结果，回合继续，下一步再调模型：

| 情况 | 模型看到的正文 |
| --- | --- |
| 用户拒绝，或策略是 `never` | `the user rejected tool "..."` |
| 没有应答通道 | `tool "..." requires approval, but no approval channel is available` |
| 批准服务没装 | `tool "..." requires approval (not yet supported)` |
| 应答被取消，且整回合没有 abort | `approval for tool "..." was cancelled` |
| 整回合已经停止 | `tool call aborted before dispatch`，本回合不再调模型 |

批准询问没有「超时不答」。`ask_user_question` 的 timed 模式默认等 120 秒，超时是成功结果，正文说明这是尚未回答，不是许可。

`packages/guard/timeout-policy` 在自己的计时器赢时，把工具结果换成 `Error: tool call timed out after ${timeoutMs}ms`。重复同一调用时，`repeat-tool-reminder` 不拦截，只在下一步前插入提醒。

shell 的非零退出、命令超时、输出过长，都是普通成功的工具结果，正文里带 `[exit code: ...]`、`[timed out after ...]` 或 `[output truncated; ...]`。模型自己读这些标记。只有 spawn 失败、调用方 abort，才是 `isError`。

子代理是另一个 `Agent`、另一份会话。委托工具立刻返回 `started subagent <id>`，完成时不改这条结果，事后另发一条用户消息。默认继承父回合的 provider 和 model；调用里同时给出两者才换路由。`spawn` 是空上下文，`fork` 只种入父会话已经结束的轮次。

启动失败（例如深度超限）变成当前工具的错误结果，父回合继续。孩子跑起来之后，父回合按停止不会自动杀掉子代理，要显式 `interrupt`。没有自动重跑子代理。孩子失败、被停、超上下文、拒绝任务，都写成后续用户消息，例如 `Background subagent <id> failed before it finished.` 父回合已经空闲时，这条消息会再开一轮；父回合还在跑，就插进下一步。子代理运行超时在这次阅读的路径里没有找到。

崩溃恢复会给悬空的工具调用补结果。还没记成已开始的，正文是 `The tool call was interrupted before the Harness recorded it as started. Retry it if it is still needed.` 已经开始但没有持久结果的，会写 `Its outcome is unknown.`，并要求不要盲目重试（`packages/core/session/src/repair.ts`）。

## 一张总表

| 发生了什么 | 模型历史里留下什么 | 接着做什么 |
| --- | --- | --- |
| 5xx、限流、流空闲超时、连接重置、空回复 | 不留半截 | 同一请求再试，最多 5 次 |
| 认证失败、余额不足、非法请求、模型 404 | 不留半截 | 回合 `error` 结束 |
| 上下文超窗 | 压缩成功则历史表面被摘要替换 | 再请求一次；再超窗就结束 |
| 输出顶到 max_tokens | 留下半截正文，丢掉工具调用 | 这一轮结束，不自动续写 |
| 工具抛错、参数不对、用户拒绝权限 | 一条 `Error:` 工具结果 | 下一步再问模型 |
| shell 非零退出或命令超时 | 一条带标记的普通工具结果 | 下一步再问模型 |
| 用户按停止 | 留下已生成的正文，标中断 | 本回合结束，等下一条用户消息 |
| 子代理随后失败 | 另发一条用户消息 | 父回合空闲时再开一轮 |

## 和本仓库已经碰到的事

本仓库的搜索仍由 `taic` 排回合、预算和评测；写内核现在只走 `hmz` 再调 `pi`，不经过直调。2026-10-20，实验脚本里的直调 `taic/llm.py` 采纳了下面几条，空闲改成 900 秒，写在 [架构说明.md](架构说明.md) 的 §2.4。这些规则不描述 `hmz` 会话。

流空闲看门狗在有新的一块数据时重新计时，整段回复不再另设一只更短的墙钟。空回复会重试，半截不写进下一次请求。顶到输出上限时留下正文，截断的工具调用不执行。

工具失败和「交回的文件与当前种子逐字相同」是两类事。工具出错会写成结果再问模型。和当前种子逐字相同走的是 `Orchestrator._eval` 里的 `unchanged`：不评测，不把这份结论放进缓存，也不把上一份加速比再报一次。和种子不同、但源码哈希已经测过，才复用旧成绩，那才是评测缓存。
