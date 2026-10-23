# Tests

## Layout

```text
tests/
├── conftest.py          # shared fixtures and the GPU/observe isolation
├── support/             # fakes, failure handoff, golden data, worker stand-ins
├── shared/              # backend-neutral tests (migration target)
├── backends/metal/      # Metal-specific tests (migration target)
└── test_*.py            # existing flat tests, moved in batches
```

`support` and the nested packages have `__init__.py`; pytest collects them recursively.

## Commands

```bash
uv run pytest -q                         # all offline tests, parallel by default
uv run pytest -q -n 0                    # serial reproduction
uv run pytest -q -m 'not gpu and not llm'
uv run pytest -q --no-test-file-log      # no per-test log files
```

The default is `-n auto --dist worksteal`. GPU tests still serialize on the
machine-wide GPU lock; LLM tests remain behind their marker.

## Logs and AI handoff

Each xdist worker writes its own files:

```text
logs/pytest/session-<worker>.log
logs/pytest/tests/<worker>/<nodeid>.log
logs/pytest/triage/<worker>/<nodeid>.md
logs/pytest/triage/<worker>/manifest.jsonl
```

The controller also writes `logs/pytest/triage/triage_summary.md`, aggregating
worker manifests. A failed test's markdown contains a fenced JSON block between
`---BEGIN_AI_FAILURE_HANDOFF---` and `---END_AI_FAILURE_HANDOFF---`, plus the
per-test log path and a capped tail. Use `-n 0` for the single-process reproduce
command printed in that block.
