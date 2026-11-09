# Agent 调试记录：RLAR actor 反复报告上下文耗尽

日期：2026-10-29

## 1. 现象

在连续执行构建执行与证据语义升级计划的 RLAR 中，actor 多次报告：

> My context window is exhausted.

随后部分轮次只复述剩余任务，没有实际代码增量。最后一次 RLAR 报告：

- exit 143，由 `/flow stop` 停止；
- 总耗时 337 分 32 秒；
- 133 turns、1710 tool calls；
- actor 实际为 67 轮，其余为 reviewer；
- `no turn produced an answer`。

相关日志：

`/Users/kuafu/.pi/agent/hmz-flow/2026-10-28T18-14-45-667Z-rlar-53163518.log`

当时阶段状态：

- R0、R1 已完成；
- R2 进行中，部分实现与聚焦 CPU 测试已落地；
- R3–R6 未开始；
- R7 真实设备工作仍未授权、保持 blocked。

## 2. 查到的实际原因

### 2.1 RLAR 的 actor 默认跨轮复用同一会话

`hmz/flows/builtin/rlar/__init__.py` 的结构是：

```python
working = await actor.spawn()
while True:
    await actor.run(..., session=working)
    reading = await reviewer.spawn()
```

因此 actor 会保留所有历史，reviewer 每轮新建。长任务中不断累积：

- 每轮 reviewer 的长修改意见；
- 重复出现的剩余任务清单；
- 大文件读取、diff 和测试输出；
- R2 的大量迁移与验收工作。

初始任务提示词偏长，但不是唯一或主要原因。主要问题是把 R1–R6 的长期工程放在一个不断增长的 actor 会话里。

### 2.2 pi 支持自动压缩，但这次没有达到触发阈值

pi 默认自动压缩条件为：

```text
contextTokens > contextWindow - reserveTokens
```

本次配置和记录：

- `deepseek-flash` 的 `contextWindow = 1,000,000`；
- 默认 `reserveTokens = 16384`，约 983K tokens 才触发；
- actor session 中 compaction 记录为 0；
- 最后 usage 约 519K tokens；
- actor 在约 446K tokens 时已开始自报耗尽；
- 返回为正常 `stopReason=stop`，不是 provider 的 context-overflow 或 `length` 错误。

所以：**pi 支持自动压缩，且没有关闭；但它按约 983K 的阈值判断还不需要压缩，模型却提前开始自报耗尽。**

是否存在服务商侧更小的实际窗口，当前日志无法证明。模型自己写“context exhausted”也不会直接触发 pi 的自动压缩。

实际 actor session：

`/Users/kuafu/.hmz/epics/-Users-kuafu-CLionProjects-ai-lowerling/20261028T181445.868Z-7d8ca8/sessions/pi/sessions/--Users-kuafu-CLionProjects-ai_lowerling--/2026-10-28T18-14-46-086Z_2fba4b50-9f20-4b26-b2d8-3e425d0b3ee1.jsonl`

## 3. 已采用的建议与配置

已在 `/Users/kuafu/.pi/agent/settings.json` 增加：

```json
{
  "compaction": {
    "enabled": true,
    "modelOverrides": {
      "deepseek/deepseek-flash": {
        "reserveTokens": 500000,
        "keepRecentTokens": 40000
      }
    }
  }
}
```

作用：DeepSeek Flash 在约 500K tokens 时提前自动压缩，并保留最近 40K tokens。其他模型和原有设置不变。

注意：这项配置只影响后续压缩判断，不会 retroactively 清理旧的 519K 会话。

### 3.1 “从检查点使用 fresh actor”到底是什么意思

**fresh actor = 开一个新的模型会话（新的 session），不带旧对话历史。不是每轮新建 actor，也不是换模型、清空代码或每次工具调用都重启进程。**

本次具体做法是：从旧 RLAR 切换到一个新的 RLAR run（`#0318fbdd`），actor 仍然使用 DeepSeek Flash，
但启用新的 actor session。旧会话约 519K tokens 的对话不再作为输入；工作树、已有代码改动和阶段验收文档
原样保留，由它们而不是旧会话记忆承接进度。因此这是**新的 RLAR run 从检查点继续**，不是旧会话原样续聊。

当前 run 内各部分的生命周期如下：

| 对象 | 当前行为 |
| --- | --- |
| Actor | 只在本 run 启动时 fresh 一次；后续 RLAR 轮次默认复用同一个 actor session，不是每轮新建 |
| Reviewer | 按 RLAR 默认行为，每轮新建 reviewer session |
| 自动压缩 | 在同一个 actor session 内把较早历史总结成摘要，不等于换 actor，也不新建 run |
| 再次 fresh actor | 只在以后明确决定轮换会话时发生，不因进入下一阶段或下一轮自动发生 |

所以当前策略准确说是：**启动时 fresh 一次 → 后续在同一个 RLAR 中靠小增量、状态写盘和自动压缩继续。**

## 4. 后续执行建议

1. **从检查点使用 fresh actor**：这里指本次启动新 run 时开一个新会话，不是每轮新建。R1–R6 已经做完，现行身份规则在 `docs/架构/构建与评测身份.md`。
2. **每轮只做一个小而完整的增量**：实际实现、focused CPU 测试、记录结果，再进入下一项。不要在一轮里规划整阶段全部工作。
3. **状态外置**：把当前阶段、下一点、决策和测试证据写进仓库里的现行文档，不要依赖会话记忆。不要重建已经删掉的 R1–R6 阶段验收目录。
4. **Reviewer notes 只写增量**：仅报告本轮新发现的实质问题、下一点和验证结果，不重复整份旧阻塞清单。
5. **一个 RLAR 内连续执行，不在阶段边界停**：阶段通过后自动进入下一阶段。只有 R2–R6 全部通过才允许 reviewer `done=true`，不要每阶段让用户重新发“继续”。
6. **保留权限边界**：不运行 GPU/NPU/真实设备，不运行 gpu/llm/slow 标记测试，不运行 `taic run` 或真正的 `taic resume`；模拟证据不能冒充真实证据。
7. **不用“上下文耗尽”当完成条件**：自报耗尽不是阶段通过，也不能据此结束整个 RLAR。应通过压缩、fresh actor 和检查点接续解决。

## 5. 当前状态

- 压缩配置已写入并验证。
- 已启动新的后台 RLAR `#0318fbdd`，从当前 R2 工作树继续；actor/reviewer 都是新会话，不恢复旧对话历史。
- 本次启动日志：`/Users/kuafu/.pi/agent/hmz-flow/2026-10-29T00-04-29-495Z-rlar-0318fbdd.log`。
- 当前 actor 只在本 run 启动时 fresh 一次；后续在同一个 RLAR 中按小增量和自动压缩继续，不逐轮换 actor。
- R2 仍是当前执行阶段，不能把“已启动新 run”当成 R2 或后续阶段已完成；R7 继续未授权 blocked。
