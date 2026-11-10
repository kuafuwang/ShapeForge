# hmz

`hmz` 是别人写的启动器，用来驱动已经登录的编程代理。搜索写内核只走 `hmz exec`，跑一场 `kernel_once`。`--kernel-agent off` 拒绝启动。换种子和确认仍在 `taic`。搜索不再单独发规划请求。`flame_chase` 是 `hmz` 自带的双代理轮流，本仓库的搜索不用它。

## 这篇讲什么

- `hmz` 只负责**启动与围栏**：它不写内核，也不判卷；判卷在 `taic`。
- 真正的模型调用发生在 `pi` 里，`pi` 由 `hmz` 拉起。链子是 `taic` → `hmz` → `pi` → 模型。
- 本仓库只给 `hmz` 一份流程 `kernel_once`：一个角色 `assistant`，写完一颗内核就停；两个委员会的轮换由搜索进程负责。
- `loop_protocol: flame_chase_1` 只是续跑协议标签，不代表调用 `flame_chase`。
- 生产搜索与实验脚本直调是两条路径。对照图在下面；传输重试的细则在 [直调与hmz取舍.md](../直调与hmz取舍.md)。
- 一场 `kernel_once` 的先后顺序见文中的时序图：`taic` 准备任务书，`pi` 写交卷文件和笔记，`taic` 只把候选和记账元数据收回 `runs/`。

```mermaid
flowchart LR
  T["taic 搜索进程<br/>准备 TASK.md、判卷、换种子"] --> E["hmz exec -f kernel_once"]
  E --> F["权限围栏<br/>可读/可写路径、可连主机"]
  E --> P["pi --mode rpc<br/>代理会话"]
  P --> M["模型<br/>provider/model[:effort]"]
  P --> W["工作目录<br/>读 TASK.md、写交卷文件"]
  E --> EP["epic<br/>~/.hmz/epics/.../epic.jsonl"]
```

上图是 `hmz` 这一场里面有什么。下面两节补上它和直调的差别，以及一场从准备任务书到收回候选的顺序。传输重试的逐条规则仍在 [直调与hmz取舍.md](../直调与hmz取舍.md)。

## 两条路径并排

路径 A 是生产搜索。路径 B 只留给实验脚本。搜索写内核只走路径 A。

```mermaid
flowchart LR
  subgraph A["路径 A · 生产搜索"]
    T["taic"] --> H["hmz exec -f kernel_once"]
    H --> P["pi"]
    P --> W["agent_work<br/>read 与 bash quick"]
    H --> F["Seatbelt 或 Landlock"]
    H --> R["JSON 回执"]
  end
  subgraph B["路径 B · 实验脚本"]
    S["实验脚本"] --> L["taic/llm.py"]
    L --> HTTP["一次 HTTP 流"]
    HTTP --> N["工具最多 6 轮"]
    HTTP --> Retry["断流则重发同一份消息"]
  end
```

| | 路径 A：`taic` → `hmz exec -f kernel_once` → `pi` | 路径 B：`taic/llm.py` 直调 |
| --- | --- | --- |
| 谁在用 | 每一次写核搜索 | 实验脚本。`--kernel-agent off` 不能把搜索改回这条路 |
| 工作区 | 隔离的 `agent_work/<场次>/`。仓库对这场只读。`device.md`、`experience.md`、`msl_reference.md` 在开跑时冻进 `agent_guidance/`，续跑用这份快照 | 没有工作区会话。一次请求结束就回到脚本 |
| 工具 | `pi` 的 `read` 和 `bash`。快速自查是任务书里的 `--quick`，由 `bash` 执行，不是 `taic/llm.py` 里的 `quick_check` | `chat_with_tools` 最多 `MAX_TOOL_TURNS = 6`。搜索不注册那两个工具 |
| 围栏 | `hmz` 的 Landlock（Linux）或 Seatbelt（macOS）。立不起来就拒绝启动。声明里的新会话是 `online=NONE` | 工具在调用进程里执行，没有这道围栏 |
| 收回的记录 | `--json` 回执：花费、会话结束原因、工具次数。候选是交卷文件。完整对话留在 epic 下的 `pi` 会话目录 | 正文。连接失败、5xx、限流和流静默按同一份消息重发 |

[security.md](../security.md) 记下了围栏合同，也记下了边界：本仓库断言「立不起来就拒绝启动」，还没有一份自己做的操作系统级围栏证明。上表因此只比较两条路径的形状。

## 一场 kernel_once 的时序

`taic` 准备任务书和冻住的指导文件，`hmz` 拉起 `pi`，`pi` 把核和笔记写在 `agent_work/`，`taic` 只把候选和记账元数据放进 `runs/`。

```mermaid
sequenceDiagram
  participant Taic as taic
  participant Guide as agent_guidance
  participant Hmz as hmz
  participant Pi as pi
  participant Work as agent_work
  participant Runs as runs
  Taic->>Guide: 冻住 device.md、experience.md、msl_reference.md
  Taic->>Work: 写本场 TASK.md 和种子 kernel.metal
  Taic->>Hmz: hmz exec -f kernel_once --json
  Hmz->>Pi: 启动会话，套上 Seatbelt 或 Landlock
  Pi->>Work: 读 TASK.md 和冻住的指导文件
  Pi->>Work: read，以及 bash 执行任务书里的 quick
  Pi->>Work: 写交卷文件、NOTES.md、FEEDBACK.md
  Hmz-->>Taic: JSON 回执：花费、结束原因、工具次数
  Taic->>Work: 读交卷文件
  Taic->>Runs: 写入候选和记账事件
  Note over Runs: 完整对话留在 hmz epic，不进入 runs
```

按时间往下看：

1. 新运行把 `device.md`、`experience.md`、`msl_reference.md` 拷进 `runs/<这次>/agent_guidance/`，并在 `run_config.json` 记下这三份和技能 `SKILL.md` 的 sha256。续跑读这份快照。
2. 每一场在 `agent_work/<场次>/` 写新的 `TASK.md`。外层成绩单和上一场失败反馈各封顶约 1000 字节。种子文件名是 `kernel.metal`，外层不拿它当交卷。
3. `taic` 调用 `hmz exec -f taic/flows/kernel_once --json`。`hmz` 启动 `pi --mode rpc`，并套上围栏。
4. `pi` 读任务书，用 `read` 看文件，用 `bash` 跑任务书里的 `--quick` / `--timed`。核写进 `TASK.md` 点名的交卷文件，笔记追加到 `NOTES.md` 和 `FEEDBACK.md`。任务书要求：整份反馈都没内容时，不要创建 `FEEDBACK.md`。
5. `hmz` 把 `spent`、会话结束和工具次数打成一行行 JSON。`taic` 据此记账，并把交卷文件收成候选。
6. `pi` 的逐字对话留在 `~/.hmz/epics/...` 的会话目录。`runs/` 里的 `transcript.jsonl` 记下的是记账事件（花费、token、`runtime` 诊断），不存放完整聊天。正常结束却没有交卷时，`agent_diagnostics/` 最多留 240 字脱敏预览，转录只记路径、长度和 sha256。更早的直调运行会在 `transcript.jsonl` 里留下聊天正文，那是路径 B 的旧记录。

`taic` 现在经 `taic/session_runtime.py` 的 `LocalHmzKernelSessionRunner` 调用上面这一场；当前 runner 只接受 local harness/workspace。选择 `split-control` 时，`pi` 仍在 control_host，只有正式评测和自测经 `EvaluationExecutor` 去设备侧受信服务；`hmz` 本身不搬进容器或 VM。详见 [执行运行时与设备扩展.md](执行运行时与设备扩展.md)。

源码不进入本仓库。2026-10-19 装上的发布包是 PyPI 的 `hmz` 0.2.0；2026-10-24 已升级到 0.2.2（PyPI 2026-10-24 的最新发布）。本机源码已快进到 `19b3dc59`（2026-10-24）；PyPI 0.2.2 的 release tag 是 `c2946aa`，两者是不同分支。其中 `flame_chase` 与 `e2013a4` 相同，仍是 48 行。搜索仍只调用 `kernel_once`；`run_config.json` 的 `loop_protocol: flame_chase_1` 只是续跑协议标签。搜索循环、上下文和判卷仍在 [搜索.md](搜索.md)。

## 它是什么

`hmz` 是一个 Python 程序。它不写内核，也不判卷。它启动已经装在本机的编程代理，给这场套上会话、权限围栏和一份运行记录。编程代理再去调用模型。本工程选用的代理是 `pi`。`pi` 是 Node 程序，也不是本仓库写的。

一条写内核的调用是：

```text
hmz exec -f taic/flows/kernel_once
    -a assistant=pi/<provider>/<模型>[:effort]
    -p turn_output_tokens=<这一场的输出 token>
    -p turn_minutes=<主回合分钟>
    -p wrap_output_tokens=<收尾回合的输出 token>
    -p wrap_minutes=<收尾回合分钟>
    -p submit_name=<交卷文件名>
    -p seed_name=kernel.metal
    -p budget.duration=<整场分钟>m
    --json
    -- "Read TASK.md in this directory and do it. Write the finished kernel to <交卷文件名>. The search does not read kernel.metal."
```

参数要点：

- `submit_name` 是 `{题目}-{dtype}-{规模}-{场次}.metal`。少了它，`kernel_once` 会回落到 `submit.metal`，而搜索只读 `TASK.md` 里写的那份交卷文件。
- `seed_name` 默认也是 `kernel.metal`；它只用于收尾提示里点名种子文件，实际写入的种子文件由搜索按本场契约放置，交卷文件仍由 `submit_name` 决定。
- `--json` 让 `hmz` 把 `spent` 等记录写成一行行 JSON，搜索据此记账。

单场美元上限是剩余预算和 `relay_budget_usd`（默认 2 美元）里较小的那个，再按该委员会的输出单价换成 `turn_output_tokens`。不是把剩余美元原样交给 `budget.cost`。

`kernel_once` 是本仓库的流程：一个角色 `assistant`，写完一颗内核就停。两个委员会的轮换在搜索进程里。轮到谁，就把那个委员会的模型写进上面的 `pi/...`。

## 它内部的四层

| 层 | 代码落点 | 一场里发生什么 |
| --- | --- | --- |
| 流程 | `flame_chase` 或本仓库的 `kernel_once` | 决定几个角色、每场是否新开会话、任务句子是什么。`flame_chase` 只有一个循环：两个角色轮流，每场 `spawn` 再 `run` 同一句任务 |
| 代理会话 | `pi --mode rpc` | 模型在这里。`pi` 用自己的 `read`、`write`、`bash` 改文件、跑命令 |
| 权限围栏 | `hmz internal fence` | 限制这场能读哪些路径、能写哪些路径、能连哪些主机。`/tmp` 可写。`~/.cache` 和 Clang 的默认模块缓存不可写。`TMPDIR` 指到这场自己的临时目录 |
| epic | `~/.hmz/epics/<工作区>/<时间>-<编号>/epic.jsonl` | 一行一件事：开始、打开了哪个会话、花费、结束。不另抄 `pi` 说过的话。那些话在 `<epic>/<场次>/sessions/pi/sessions/` 下（0.2.2 起；更早的 hmz 把它们放在 `~/.pi/agent/sessions`） |

`hmz` 的网页按 epic 显示一场：谁上场、何时开始和结束、怎么结束、花了多少。搜索看板读同样的四行，并另读 `pi` 会话的末尾。两边都不把 epic 当成成绩单。

## 技能怎么交到 pi

流程目录下的 `skills/<名字>/SKILL.md` 要由角色用 `_skills` 点名。`hmz` 才把它放进这场进程的虚拟家目录。不放进工作区。`pi --mode rpc` 不信任工作区里的 `.pi/skills`，也不加载工作区里的 `.pi/mcp.json`。

本仓库的技能是 `taic/flows/kernel_once/skills/taic-tools/SKILL.md`。它让代理读 `msl_reference.md`，并按 `TASK.md` 里的命令做 `--quick` 和 `--timed`。计时用机器上那把 GPU 锁。主回合有输出 token 和时长上限；到了上限，同一会话再写一次 NOTES，把核写进 `TASK.md` 里的交卷文件。`kernel.metal` 是种子，外层不读。

KDA 教 Nsight 是同一条路。`skills/ncu-report-skill/SKILL.md` 写明先测、再对症状、再写计划，并给出可照抄的 `ncu` 命令。`flame_chase` 自己不跑剖析。代理用终端执行技能里的命令。

## 本工程用了什么，不用什么

| 用 | 不用 |
| --- | --- |
| `hmz exec`、`kernel_once`、本机 `pi`、权限围栏、代理会话、epic 的四行摘要、技能目录 | `flame_chase` 排整次搜索；`hmz web` 当搜索看板；Claude、Codex、Kimi Code、litellm；把 Humanize 源码搬进本仓库 |

`taic/` 没有 import 或调用 `flame_chase`；上面那个 `flame_chase_1` 字符串不是调用它。不用 `flame_chase` 排班，是因为两个委员会已经在搜索里轮换。下一场读到的是新写的 `TASK.md`：当前核、成绩单和上一场失败反馈。由它排整次搜索，会把计时和换种子交回代理。写内核的那一场不弱：上场的仍是 `pi`。判卷比 `flame_chase` 多，它不判卷。

搜索不再单独规划。模型调用只发生在 `hmz` 拉起的 `pi` 里。`lookup_msl` 和 `quick_check` 不注册进当前搜索。

可观测层按需启动一个 `hmz web --no-open --port 0 --attached` sidecar，作为查看 epic/session 的辅助窗口。它只读 Humanize 的会话视图，不读搜索候选、确认门或 trust，也不作为发布证据；主看板仍是 `python -m observe`。详见 [可观测.md](可观测.md) 的「hmz web 辅助窗口」。
