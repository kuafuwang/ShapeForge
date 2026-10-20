# Agents

This repository lowers Triton kernels to Metal Shading Language on an Apple GPU. `taic` searches. `tcenv` compiles, checks, and times each candidate.

Every search uses two committees, `first` and `second`, one member each. In `taic.toml`, first is DeepSeek `deepseek-flash` and second is Moonshot `kimi-k3`. Handoff between them is mandatory even when both members use the same model. Repair stays in the conversation that started the candidate. When `--kernel-agent pi` writes the kernel, each attempt is a new `hmz` session.

Do not spend money or run a GPU kernel unless the user asks. `taic run` and `taic resume` do both. The read-only summary is:

```
uv run python -m taic.cli brief
uv run python -m taic.cli brief RUN_DIR
```

A speedup is published only when the confirmation gate binds the remeasured kernel; the search best is a separate number.

Credentials are never committed.
