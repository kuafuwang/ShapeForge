@AGENTS.md

# Claude Code project entry

Read `AGENTS.md` first. The repository-level tests, safety rules, GPU/LLM
permission rules and backend/target layout are authoritative there.

## Skills

This repository exposes the same skills through `.claude/skills/`, `.codex/skills/`,
`.cursor/skills/` and `.agents/skills/`. They are symlinks to the canonical
sources:

- `harness-feedback` → `.pi/skills/harness-feedback`
- `taic-tools` → `taic/flows/kernel_once/skills/taic-tools`

Read the matching `SKILL.md` before using a skill. Do not fork the skill text;
edit the canonical source and rerun the offline tests.

## Before changing code

```bash
uv run pytest -q -m 'not gpu and not llm and not slow'
```

Use `-n 0` for a single-process reproduction. A GPU test or paid LLM call requires
explicit user permission. Preserve unrelated working-tree changes.
