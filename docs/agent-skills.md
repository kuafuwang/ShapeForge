# Agent skills

The canonical project skills live in the repository, not in one agent's private
directory:

```text
.pi/skills/harness-feedback/                 # feedback triage
taic/flows/kernel_once/skills/taic-tools/    # kernel-session tool guidance
```

The following are relative symlinks to those sources:

```text
.claude/skills/{harness-feedback,taic-tools}
.codex/skills/{harness-feedback,taic-tools}
.cursor/skills/{harness-feedback,taic-tools}
.agents/skills/{harness-feedback,taic-tools}
```

Agent entry points:

| Agent | Entry point | Skill discovery |
| --- | --- | --- |
| Claude Code | `CLAUDE.md` | `.claude/skills/` |
| Codex | `AGENTS.md` | `.codex/skills/` / `.agents/skills/` |
| Cursor | `.cursor/rules/project.mdc` | `.cursor/skills/` and rules |
| Pi | `AGENTS.md` | `.pi/skills/` |

## Adding or changing a skill

1. Edit the canonical source under `.pi/skills/` or
   `taic/flows/kernel_once/skills/`.
2. Keep the existing relative symlinks; do not create copied skill trees.
3. If a tool requires a rule file, add a short pointer rule that names the
   canonical source, as `.cursor/rules/harness-feedback.mdc` does.
4. Run the offline suite and update this document if the mapping changes.

Skill text can be untrusted input only when it comes from run feedback; the skill
instructions themselves are repository code and should be reviewed like code.
