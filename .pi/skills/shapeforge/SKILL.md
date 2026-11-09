---
name: shapeforge
description: Understand and safely contribute to ShapeForge's OpSpec v2, shape portfolio, Metal runtime and evidence pipeline. Use when onboarding to this repository, modifying candidates/dispatch/performance evidence, or running its acceptance workflow.
---

# ShapeForge contributor skill

This skill is an onboarding router, not a second architecture document. Load the authoritative context instead of reconstructing the project from old Triton notes.

## 1. Read in this order

1. `AGENTS.md` — candidate freedom, evidence invariants, current status, tests and cost/device boundaries.
2. `README.md` or `README.zh-CN.md` — the only project entrypoints.
3. `docs/架构/shape-portfolio实施计划.md` — living phase status and acceptance scope.
4. The specific architecture file for your task listed in `AGENTS.md`.
5. `tests/README.md` and the relevant layer README before adding tests.

If the task is about variable-length sequences, read `docs/架构/varlen支持计划.md` (the header is the status; the V3 Wasm section is a historical snapshot) and `docs/架构/v2协议参考.md` §3.1. For Host Program work, read `docs/架构/native-host-program-ABI.md` first.

Current execution facts. Phase narratives in `AGENTS.md` are the longer status; when an older paragraph disagrees with that file and the code, follow the code.

- Live tasks are `tasks/<op>/opspec.json` plus `reference.py`. `tasks.get_task` does not assemble those directories.
- The Host Program is native64 C99. `tcenv/hostprogram/native_toolchain.py` compiles `host/tiling.c` to a worker-local `.dylib`. Python calls `sf_native_invoke()`. Evidence schema is `tcenv.hostprogram.run.2`. The Wasm / `wasmtime` toolchain is gone.
- Varlen V0–V3 mechanism, V5 L2 capacity, and V6 six-tier Metal acceptance are in tree. V4 Dyn-1 was measured and skipped. V6 speedup is only versus the registered naive Torch-MPS baseline.
- P2-F re-recording is done. The historical start-surface check in the P2 verifier still rejects drift. Do not edit that verifier or its acceptance baseline to make the check pass.
- The task-pack code closure is 74 files from `code_import_closure` in `tcenv/taskcontract/pack.py`, including `tcenv/hostprogram/native_abi.py` and `tcenv/taskcontract/runtime_shape.py`.
- TileLang `/1` is `interface_only`. `/2` and `/3` are `available`. A new run selects the highest available version. Default search delivery remains `msl-single` unless `--track` names another available track. Exploration on that track is not speedup publication. There is no third DSL and no Graph track.
- Qingwei NPU is simulation only (`simulated=true`, `publication_allowed=false`).

Do not describe CPU protocol success as GPU or performance success. Do not paste phase status back into this skill.

Do not create `docs/README.md` or another project overview.

## 2. Establish the task boundary

Before editing, write down:

- whether the change belongs to the **candidate** or the **harness**;
- which identity, correctness, timing and evidence invariants apply;
- which phase/epoch owns the change;
- which fixed positive and negative tests prove it;
- whether GPU or paid-model authorization exists.

Candidates own algorithms, fusion, coverage plans, Tiling, Host Tiling and ExecutionGeometry. The harness owns task contracts, ABI, safety, measurement boundaries and evidence. Do not introduce a per-task pattern/tiling registry to make an implementation easier.

## 3. Preserve the evidence contract

- Search best is not published speedup.
- Correctness uses only `tolerances + scaled_error`, including final `read_write` state.
- Fallback time and correctness never count as variant success.
- Every shard must independently meet the 1.10 gate; geomeans cannot hide a regression.
- Do not rewrite completed phase evidence or select a lucky timing run.
- NPU simulation never establishes real-device correctness/performance.

If evidence is insufficient, leave the phase blocked and record it. Do not skip tests, delete shards or weaken the gate.

## 4. Run only authorized checks

Default safe suite:

```bash
uv run pytest -q -m 'not gpu and not llm and not slow'
```

Only with explicit GPU permission:

```bash
uv run pytest -q -m gpu
```

Single serial failure:

```bash
uv run pytest -q -n 0 <nodeid>
```

Use `logs/pytest/triage/triage_summary.md` and its JSON handoff for diagnosis. Do not run paid search (`taic run`/`taic resume`) without explicit user authorization.

## 5. Finish with a coherent change

A complete contribution normally includes:

- focused implementation and fixed tests/negative cases;
- current acceptance documentation and manifest when required;
- fresh evidence rather than edited performance numbers;
- aligned `README.md` and `README.zh-CN.md` only when the public contract changed;
- `git diff --check`, a clean scoped commit and no credentials or unrelated artifacts.

For general collaboration rules, read `CONTRIBUTING.md`. For other existing skills and aliases, use `docs/agent-skills.md`; do not duplicate their text here.
