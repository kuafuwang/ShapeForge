# Agents

测试目录与 golden 放置规则见 `tests/README.md`。

Tests:

- Fast safe suite (no GPU, no model, no long checks): `uv run pytest -q -m 'not gpu and not llm and not slow'`. Drop `and not slow` for the exhaustive CPU suite. Plain `uv run pytest -q` also runs `gpu` tests; run those only when the user allows GPU use. `-m gpu` / `-m llm` need the same explicit permission.
- Debug one failure serially: `uv run pytest -q -n 0 <nodeid>`.
- Read `tests/README.md` first: layout (`shared/` vs `backends/<id>/` vs `support/`), placement rules, markers, parallelism, and failure logs.
- Layer rules: `tests/shared/README.md`, `tests/backends/README.md`, `tests/backends/metal/README.md`, `tests/support/README.md`, `tests/support/golden/README.md`. The executable version of those rules is `tests/shared/test_structure.py`.
- A failure writes `logs/pytest/triage/triage_summary.md`; read it, then the Markdown it points at. The JSON block between `---BEGIN_AI_FAILURE_HANDOFF---` and `---END_AI_FAILURE_HANDOFF---` is the ground truth and carries a single-process reproduce command.
- To understand what an assertion means: `docs/架构/评测.md` (compile, safety, tolerance, timing) and `docs/架构/搜索.md` (selection, confirmation). `docs/架构/数据层.md` covers the observe/index tests.

This repository lowers Triton kernels to Metal Shading Language on an Apple GPU. `taic` searches. `tcenv` compiles, checks, and times each candidate. Backend code lives under `tcenv/backends/<id>/`; backend metadata, skills and wiki under `backends/<id>/`; chip profiles, measurements and trust under `targets/<id>/`. The old top-level `tcenv/*` backend shims are gone.

Every search uses two committees, `first` and `second`, one member each. In `taic.toml`, first is DeepSeek `deepseek-flash` and second is Moonshot `kimi-k3`. Kernels are written only through `hmz`: by default one new session a round, even when both members use the same model. `search.provider_retries` (0–3, default 0) allows bounded same-committee attempts for a structured provider `turn_failed` with no submission; attempts are billed separately and do not advance the round. A normal answer without a submission, a timeout/process failure, or an available checkpoint is not retried. A final no-kernel outcome (`agent_failed`, `provider_failed`, or `agent_no_submission`) hands the next session to the other committee. Two final no-kernel rounds in a row stop the search with `stop_reason=agent_failure`. An evaluated kernel that fails the checks, or a file identical to the seed, clears that count.

Do not spend money or run a GPU kernel unless the user asks. `taic run` and `taic resume` do both. The read-only summary is:

```
uv run python -m taic.cli brief
uv run python -m taic.cli brief RUN_DIR
```

A speedup is published only when the confirmation gate binds the remeasured kernel; the search best is a separate number.

The data layer (run layout, SQLite index, retrieval, protocol epochs, cost and retention) is documented in `docs/架构/数据层.md`. `taic query` is the CLI for it: no model call and no GPU, but `best`/`failures`/`sessions`/`models` first refresh the derived index, so it is not a strictly read-only process.

Credentials are never committed.
