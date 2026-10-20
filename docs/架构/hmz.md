# hmz

这一篇说明 Humanize（命令 `hmz`）是什么，以及本工程调用了它的哪一层。本工程的搜索循环、上下文、工具定义和判卷仍在 [搜索.md](搜索.md)。谁做、调用谁、模型在哪一层，写在 [外部框架经验.md](../外部框架经验.md) 的「我们做了什么，调用了什么，模型在哪一层」。

`hmz` 不是本仓库写的。源码不进入本仓库。2026-10-19 装上的发布包是 PyPI 的 `hmz` 0.2.0。本机检出 `e2013a4` 里的 `flame_chase` 与该发布包相同。当天 GitHub `main` 的 `737f438` 不改这个文件。

## 它是什么

`hmz` 是一个 Python 程序。它不写内核，也不判卷。它启动已经装在本机的编程代理，给这场套上会话、权限围栏和一份运行记录。编程代理再去调用模型。本工程选用的代理是 `pi`。`pi` 是 Node 程序，也不是本仓库写的。

一条写内核的调用是：

```text
hmz exec -f taic/flows/kernel_once
    -a assistant=pi/<provider>/<模型>
    -p budget.cost=<还剩的美元>
```

`kernel_once` 是本仓库的流程：一个角色 `assistant`，写完一颗内核就停。两个委员会的轮换在搜索进程里。轮到谁，就把那个委员会的模型写进上面的 `pi/...`。

## 它内部的四层

| 层 | 代码落点 | 一场里发生什么 |
| --- | --- | --- |
| 流程 | `flame_chase` 或本仓库的 `kernel_once` | 决定几个角色、每场是否新开会话、任务句子是什么。`flame_chase` 只有一个循环：两个角色轮流，每场 `spawn` 再 `run` 同一句任务 |
| 代理会话 | `pi --mode rpc` | 模型在这里。`pi` 用自己的 `read`、`write`、`bash` 改文件、跑命令 |
| 权限围栏 | `hmz internal fence` | 限制这场能读哪些路径、能写哪些路径、能连哪些主机。`/tmp` 可写。`~/.cache` 和 Clang 的默认模块缓存不可写。`TMPDIR` 指到这场自己的临时目录 |
| epic | `~/.hmz/epics/<工作区>/<时间>-<编号>/epic.jsonl` | 一行一件事：开始、打开了哪个会话、花费、结束。不另抄 `pi` 说过的话。那些话在 `~/.pi/agent/sessions` |

`hmz` 的网页按 epic 显示一场：谁上场、何时开始和结束、怎么结束、花了多少。搜索看板读同样的四行，并另读 `pi` 会话的末尾。两边都不把 epic 当成成绩单。

## 技能怎么交到 pi

流程目录下的 `skills/<名字>/SKILL.md` 要由角色用 `_skills` 点名。`hmz` 才把它放进这场进程的虚拟家目录。不放进工作区。`pi --mode rpc` 不信任工作区里的 `.pi/skills`，也不加载工作区里的 `.pi/mcp.json`。

本仓库的技能是 `taic/flows/kernel_once/skills/taic-tools/SKILL.md`，角色上写了 `_skills = ("taic-tools",)`。文件头点名两个工具。正文说明用法：`lookup_msl` 是用 `read` 读工作区里的 `msl_reference.md`；`quick_check` 是用 `bash` 跑 `TASK.md` 里那条 `tcenv eval --quick`，锁文件在 `/tmp/ai_lowerling-gpu.lock`。这条不计时。

KDA 教 Nsight 是同一条路。`skills/ncu-report-skill/SKILL.md` 写明先测、再对症状、再写计划，并给出可照抄的 `ncu` 命令。`flame_chase` 自己不跑剖析。代理用终端执行技能里的命令。

## 本工程用了什么，不用什么

| 用 | 不用 |
| --- | --- |
| `hmz exec`、`kernel_once`、本机 `pi`、权限围栏、代理会话、epic 的四行摘要、技能目录 | `flame_chase` 排整次搜索；`hmz web` 当搜索看板；Claude、Codex、Kimi Code、litellm；把 Humanize 源码搬进本仓库 |

不用 `flame_chase` 排班，是因为两个委员会已经在搜索里轮换，交接的是当前内核和成绩单。由它排整次搜索，会把计时和换种子交回代理。写内核的那一场不弱：上场的仍是 `pi`。判卷比 `flame_chase` 多，它不判卷。

规划不经过 `hmz`。搜索进程直接调用模型接口。省略 `--kernel-agent pi` 时，写内核也不经过 `hmz`，模型在搜索进程自己的聊天里，`lookup_msl` 和 `quick_check` 是注册进那次请求的函数。
