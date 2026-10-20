# DeepSeek Harness 的 Web 可观测

这篇记的是别人的网页怎样看见一场会话，不是本仓库的看板。本仓库的看板仍由 `observe` 读 `runs/` 里的目录，实时靠轮询，没有他们那条 WebSocket。

这篇对过 `/Users/kuafu/workspace/deepseek-harness` 本地检出 `687f282e86`（2026-10-17，`dsh` 0.2.1-rc.2）。那个检出不在本仓库里。浏览器入口是 `apps/web`。`packages/web/web` 是搜索和抓取，没有页面事件通道。

## 架构

页面用 HTTP 打开，事件走一条 WebSocket，不是 SSE。

| 通道 | 路径 | 作用 |
| --- | --- | --- |
| 页面 | `GET /` | 静态页。默认 `127.0.0.1:3080`，可用配置改主机和端口。拒绝绑 `0.0.0.0` / `::` |
| 实时 | WebSocket `/api/remote.mux` | 一条物理连接，上面复用多条逻辑流 |
| 一元调用 | `POST /api/<namespace>/<method>` | 翻更早的历史走这里，不走 WebSocket |

启动 URL 带 `?token=`。校验通过后种 `dsh-auth-*` cookie，再 303 到页面。之后的 WebSocket 和 `POST /api` 靠这张 cookie。请求还要过 Host / Origin：loopback、监听地址本身，或配置里的可信主机。`sec-fetch-site: cross-site` 拒绝。

一条已经落盘的会话事件，路径是：

1. agent 循环 `session.append`，写入会话日志。
2. 日志提交后发出 `session/event`。
3. 这条会话的 `follow` 若 `session.id` 对得上，就 `yield` 这一条。
4. 网关把帧写进 `/api/remote.mux`。
5. 浏览器里的 journal stream 收成 `append`，会话窗口 `appendLive`。
6. 对话装配订阅这份窗口，`useSyncExternalStore` 只通知变了的节点。

还没落盘的流式 token 不进日志。循环发 `agent/assistant-stream`，同一条 `follow` 另送 `{type:'assistant-stream'}` 帧。

每条 `follow` 只跟一个会话地址。第一帧是当前尾窗快照（头、游标、一页记录、`hasMore`、投影、助手基线）。之后持久事件按 `seq` 逐条增量。助手流式帧不占 `seq`。整页还有一条 `$events` 流给会话列表，客户端再按 `sessionId` 分。

断线后再打开，不是从上次的 `seq` 续传。`follow` 请求里没有恢复游标。重连再读当前日志，发一帧新的尾窗，客户端整窗替换。落在这扇尾窗里的事件因此补上；更早的靠 `hasMore`，用 HTTP `session.page` 往前翻。之前已经向前翻进来的页，会被这次替换清掉。心跳连续丢失就断开。重连退避从最多 500 毫秒起，倍数 2，封顶 10000 毫秒。

## 数据结构

持久化是每个会话一个目录。当前代文件是 `session.v4.jsonl`，可以再压成 `.zstd`。首行是头，其后每行一个对象。

头：`{ type:'session', version, id, createdAt, isSeeded, delegationDepth, cwd?, parentSession?, origin?, agentPreset? }`。

事件信封：`{ type, seq, time, data, ignorable?, surfaceOp?, sourceEventSeqs? }`。

和页面相关的 `data`：

| 事件 | 页面用它看什么 |
| --- | --- |
| `turn/start`、`turn/end` | 一轮的边界。结束原因有 `completed`、`aborted`、`blocked`、`error`、`max-tokens`、`interrupted`、`forked` |
| `step/start`、`step/end` | 一轮里的一步 |
| `user/message`、`assistant/message` | 对话正文。助手条带 `message`、压缩过的 `stream`、可选 `usage`、`interrupted` |
| `assistant/attempt` | 失败、重试、取消，未上屏的尝试 |
| `tool/call`、`tool/result` | 工具名、模型原文参数字符串、结果、`isError`、`error` |
| `request/header` | 这一串请求的 `provider`、`model`、工具清单 |
| `llm/retry`、`llm/retry-started` | 传输重试和失败原因 |
| `compaction/start`、`summary`、`end`、`prune` | 压缩。摘要里有模型和 token 估价 |

`assistant/live-chunk` 只活在浏览器窗口里，不写日志。

浏览器不另存一份消息文档。Chat 和轨迹页都从事件投影。壳上只在 `localStorage` 的 `dsh.conversation.<sessionId>` 里放草稿和当前视图。

投影出来的形状：

| 对象 | 字段里和可观测有关的 |
| --- | --- |
| 用户消息 | `seq`、`time`、`content`、`source` |
| 助手消息 | `turn`、`step`、`blocks`（`text` / `reasoning` / `tool-call`）、`usage`、`interrupted` |
| 工具 | 进行中是 `preparing` 或 `start`；结果是 `content`、`isError`、`error`、`meta`、子调用 |
| 一轮 | 位置、状态、步；页脚有 token 合计；过程条有消息数、工具数、子代理数 |
| 轨迹 | 系统提示、节点、助手、工具、请求头、压缩、回合结束。单元格另有 input / output / cache / think |

不进画面的：

- 思考签名留在日志的 `replayState`，投影只拷 `reasoning.text`。
- 流式工具参数增量不进助手块，归到工具节点。过程行摘要截到 160 个字素。
- JSON 展示超过 20000 字符截断。
- `AUTH` 错误在页面上只留错误码，消息清空。
- 用户文本的打码只改绘制，节点里仍是原文。日志本身不脱敏。
- 会话结构里没有金额。token 在 `usage`。模型名在 `request/header` 和助手消息的 `source`。

## 动态更新

流式输出不是每个 token 重画整页。

- 除 `usage` 和 `finish` 外，助手增量按动画帧发布。装配处嵌套 3 次 `requestAnimationFrame` 再刷。没有按毫秒节流。
- 刷完只发布脏了的节点。引用没变就跳过。顺序没变就留原数组。每一行只订自己的 key。
- Markdown 增量解析，已经冻结的块留下，只重画尾部。
- 工具参数增量走同一套三帧合批。`tool/call` 和 `tool/result` 是立即刷，并取消还没刷的帧。进行中和结果落在同一个 `tool-call` 节点上，不重排消息列表。

列表：

| 列表 | 做法 |
| --- | --- |
| 消息 | 没有虚拟列表。已加载的全部画出来。首屏和「加载更早」同一窗口：至少 50 条 user/assistant，且至少 2 个回合，硬顶 500 条。翻进来的留在窗口里，不裁掉 |
| 回合轨 | 条目不少于 2 时用虚拟列表。提问预览 50 字，回答预览 120 字 |
| 会话列表 | 没有虚拟列表。每组默认 5 行，再展开一次加 5。自动水位最多 50 |

重连用尾页整窗替换，不是补缺口。窗口的 revision 没变，订阅方直接返回。已经打开的会话再切回去，不再请求。第一次打开才拉尾页。

大段工具输出默认折叠，点开才画。展开后的行数上限：read 8，search 8，diff 9，超了就头尾各一半、中间收起。终端在对话里展开后不做中间折叠。未知 JSON 默认折叠，展开后超过 20000 字符截断。

## 关注点

默认打开的是「对话」页。左栏是会话列表，中间是对话和输入，右栏默认收起。右栏展开后是指南、文件、终端，不是回合检查器。工具详情在对话里的那一行点开，不在右栏。「检查」才跳到轨迹页。

Web 默认是「详细」，并且「回答结束后收起」。

| 人要看的 | 页面上 |
| --- | --- |
| 正在生成 | 末尾状态「深度求索中，用时 …」。输入按钮变成停止。当前回合的工具和思考铺开，不收成组 |
| 正文 | Markdown，定稿后留在折叠外面 |
| 思考 | 标题「思考」，默认折叠。详细模式收起时露第一行 |
| 工具 | 一行：图标、名字、摘要，默认折叠。失败时摘要换成错误首行 |
| 正常结束 | 收成一行「已完成，用时 …」，点开才见步骤 |
| 用户取消 | 过程头变成「已停止」，截断的正文末尾再写「已停止」。不画失败行 |
| 传输重试 | 对话里一行「正在重试 / 等待重试（n/m）」。原因默认折在 `<details>` 里 |
| 本轮失败 | 对话里一行「本轮运行失败」，带消息和可选错误码 |
| 输出顶满 | 同一位置的警告行，不当成失败 |
| token | 输入条上方的药丸，点开才有分项。每轮尾部还有「消耗」。没有金额 |
| 上下文占用 | 输入条上的圆环，点开才拆系统、工具、消息 |
| 额度用尽 | 全屏浮层，不进对话 |

故意不画进对话流的：系统提示词、普通上下文、名为 `permission` 的命令。系统提示词要到轨迹页选中某次请求才有单独一页。更早的已完成回合折在「已完成」后面。设置里才能改工作步骤的详略、收起时机、性能与用量。

轨迹是顶栏的另一页，默认不打开。插件市场和设置页的大部分与回合可观测无关。Session Log 只是设置里的上传开关，对话里不显示日志。

## 这篇不改本仓库的看板

本仓库的看板仍由 `observe` 读运行目录。上面是 DeepSeek Harness 在 `687f282e86` 上的 Web 做法。
