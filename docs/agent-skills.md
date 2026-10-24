# Agent skills

## 这篇讲什么

项目的技能（skill）只有一份权威源，放在仓库里，而不是某个 agent 的私有目录。各个 agent 工具看到的 `.claude`、`.codex`、`.cursor`、`.agents` 目录只是指向权威源的相对符号链接。改技能时改权威源，不要复制出一份新的技能树。

## 权威源与符号链接

权威源（canonical source）在仓库内：

```text
.pi/skills/harness-feedback/                 # feedback triage
taic/flows/kernel_once/skills/taic-tools/    # kernel-session tool guidance
```

下面这些是指向上述权威源的相对符号链接：

```text
.claude/skills/{harness-feedback,taic-tools}
.codex/skills/{harness-feedback,taic-tools}
.cursor/skills/{harness-feedback,taic-tools}
.agents/skills/{harness-feedback,taic-tools}
```

各 agent 的入口文件和技能发现位置：

| Agent | 入口文件 Entry point | 技能发现位置 Skill discovery |
| --- | --- | --- |
| Claude Code | `CLAUDE.md` | `.claude/skills/` |
| Codex | `AGENTS.md` | `.codex/skills/` / `.agents/skills/` |
| Cursor | `.cursor/rules/project.mdc` | `.cursor/skills/` and rules |
| Pi | `AGENTS.md` | `.pi/skills/` |

一句话概括这套布局：入口文件负责告诉 agent“项目是什么”，技能目录负责给它“怎么做某类任务”的说明；两者都指向同一份仓库内容。

## Adding or changing a skill

新增或修改一个技能时，按顺序做四件事：

1. 改权威源：编辑 `.pi/skills/` 或 `taic/flows/kernel_once/skills/` 下的文件。
2. 保留现有的相对符号链接；不要创建复制出来的技能树。
3. 如果某个工具要求单独的规则文件，就加一条简短的指针规则，写明权威源在哪里，例如 `.cursor/rules/harness-feedback.mdc` 的做法。
4. 跑一遍离线测试套件；如果映射关系变了，同步更新本文档。

```mermaid
flowchart LR
    A["改权威源 .pi/skills 或 taic/flows"] --> B["保留相对符号链接"]
    B --> C["必要时加指针规则文件"]
    C --> D["跑离线测试套件"]
    D --> E["映射有变则更新本文档"]
```

## 信任边界

只有来自 run 反馈的技能文本才可能被当作不可信输入；技能说明本身属于仓库代码，应当像代码一样 review。
