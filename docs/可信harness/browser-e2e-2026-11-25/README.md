# 真实浏览器 E2E：Edge 布局、点击、焦点与自动重连（2026-11-25）

## 结论

**通过。** 使用真实 Microsoft Edge **158.0.4370.33**（Chromium，headless）和真实 Uvicorn/ASGI observe 服务完成自动化 E2E。没有执行 `safaridriver --enable`，没有修改 Safari 设置，没有模型或 GPU 调用。

本轮是新的自动化浏览器证据，不改写 [可观测](../../架构/可观测.md) 中既有的 Safari/Edge 人工窗口验收。Safari 本轮没有运行，Chrome stable 也没有安装，不能写成 Safari/Chrome 自动化通过。

## 实际验收项

### 真实布局与点击

在 **1280、760、620、480、320 px** 五档宽度下：

- `scrollWidth <= innerWidth`，无整页横向溢出；
- header/main 不重叠，详情主列铺满；
- 点击 `e2e/one` 运行行，URL 进入该 run，详情异步加载完成；
- `#runs-toggle`、`#detail-toggle`、`#side-toggle` 都能真实收起/展开，并同步 `aria-expanded`。

逐档截图：`layout-1280.png`、`layout-760.png`、`layout-620.png`、`layout-480.png`、`layout-320.png`。

### 真实焦点

- 点击 `#search-open` 后，真实 `document.activeElement` 为 `#search-input`；
- Shift+Tab 到 `#search-close`，Tab 回到输入框，Escape 关闭并把焦点还给 `#search-open`；
- 运行表恰好一个 `tabindex=0`；从首行 `e2e/two` 按 ArrowDown 到 `e2e/one`，可见焦点轮廓为 `solid 2px`。

### 真实断线重连

测试停止并重启同一端口的真实服务，保留浏览器页面、URL 和派生索引。只观察原生 EventSource，不调用 `close()`，不伪造 reconnect 回调：

| 指标 | 结果 |
| --- | --- |
| EventSource constructor | 1 |
| open 事件 | 2 |
| error 事件 | 1 |
| 最终 readyState | 1 (`OPEN`) |
| 重连后相同 revision 的 detail GET | 0 |
| 新 revision 的变化帧 | 1 |
| 新 revision 触发的 detail GET | 1 |
| 更新后的重复运行行 | 0（仍恰好 1 行） |
| 页面导航 | 未发生，URL 不变 |

重连后发布 `spent_usd: 0.20 → 0.35`，`#detail-budget` 真实更新。相同快照在新连接上可以重复一次，这是服务端每连接基线；断言的是**没有重复 revision 驱动的 detail 请求或 DOM 行**，不是禁止重连快照本身。

## 安全与预期网络事件

真实 ASGI guard **仅允许 GET，HEAD 也拒绝**。`HEAD /api/budget/reset`、`HEAD /api/stop`、`POST /api/budget/reset`、`POST /api/stop` 四个负例都返回 403。没有点击加预算、停止、删除、保存设置、review/promote 等控件，也没有启动搜索、hmz、付费评审或 GPU。

只分类两项预期网络事件，其他 console/page error 均判失败：

1. `/favicon.ico` 的 404，与被测流程无关；
2. 主动停服造成的 `/api/v2/stream` `ERR_INCOMPLETE_CHUNKED_ENCODING`，即本次真实断线条件；只接受确切 stream URL、确切文本、恰好一次；每条 console error 在捕获时记录 `restartWindow`，停服前和确认重连后的同名错误均不接受。

重启前后真实导航计数均为 4（初始加载和正常 pushState），没有新增导航。最终 `console_errors=[]`、`page_errors=[]`，逐项结果见 [evidence.json](evidence.json)。

## 重跑

```bash
TAIC_BROWSER_E2E=1 \
NODE_PATH=/tmp/shapeforge-browser-e2e/node_modules \
TAIC_BROWSER_E2E_EVIDENCE_DIR="$PWD/docs/可信harness/browser-e2e-2026-11-25" \
uv run pytest -q -n 0 tests/shared/test_observe_browser_e2e.py
```

`playwright-core` 只安装在 `/tmp/shapeforge-browser-e2e`，不进入项目依赖或系统 Python。测试入口是 `tests/shared/test_observe_browser_e2e.py`，真实服务与只读 guard 在 `tests/support/observe_e2e_server.py`，浏览器驱动在 `tests/support/observe_e2e.js`。

未设置 `TAIC_BROWSER_E2E=1` 时，该环境专属节点显式 skip；本次实际执行为 **1 passed / 27.73s**，不是 skip 形成的通过。独立只读复核已返回 `done=true`（[review.json](review.json)）。
