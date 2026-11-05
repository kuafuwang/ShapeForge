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

If the task is about variable-length sequences, also read `docs/架构/varlen支持计划.md` and `docs/架构/v2协议参考.md` §3.1. V0 + V1 are complete on CPU and real Metal: an L1 tier accepts any legal `cu_seqlens` under one total/count, with auditable case realization and replayable evaluation-input identity. V2 Dyn-0 is also complete (mechanism only, not a performance claim): `tasks/varlen_kda` declares the work-list/handoff workspace capacities, `V2ArgSet` allocates the possible-stage union, and two separate real-Metal fixtures prove "preprocessing + sequential chain" and "real piece handoff" (never one posing as the other). The NPU side is a test-only simulator with `simulated=true/publication_allowed=false`; its work list is compared byte for byte against the buffer the device plan stage really writes, and its consumer-first spin model is a host-scheduling argument, never Metal safety evidence. Dyn-0 splits a chain evenly (`divmod`, every piece non-empty), and fixture B's four-stage native descriptor measures 7980 B of the 8192 B closed bound, so one more stage or binding would crash a legitimate candidate as an infrastructure error — widening that needs its own epoch decision. V3 C99/Wasm Host Program, V4 Dyn-1, V5 L2 capacity and V6 real-scale performance are **not** implemented. `tasks/varlen_kda` is the reference task.

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
