# Agents

测试目录与 golden 放置规则见 `tests/README.md`。

This repository lowers Triton kernels to Metal Shading Language on an Apple GPU. `taic` searches. `tcenv` compiles, checks, and times each candidate.

Every search uses two committees, `first` and `second`, one member each. In `taic.toml`, first is DeepSeek `deepseek-flash` and second is Moonshot `kimi-k3`. Kernels are written only through `hmz`: one new session a round, even when both members use the same model. A session that does not leave a new kernel (`agent_failed`) hands the next session to the other committee. Two of those in a row stop the search. An evaluated kernel that fails the checks, or a file identical to the seed, clears that count.

Do not spend money or run a GPU kernel unless the user asks. `taic run` and `taic resume` do both. The read-only summary is:

```
uv run python -m taic.cli brief
uv run python -m taic.cli brief RUN_DIR
```

A speedup is published only when the confirmation gate binds the remeasured kernel; the search best is a separate number.

The data layer (run layout, SQLite index, retrieval, protocol epochs, cost and retention) is documented in `docs/架构/数据层.md`. `taic query` is the CLI for it: no model call and no GPU, but `best`/`failures`/`sessions`/`models` first refresh the derived index, so it is not a strictly read-only process.

Credentials are never committed.
