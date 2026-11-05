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

If the task is about variable-length sequences, also read `docs/架构/varlen支持计划.md` and `docs/架构/v2协议参考.md` §3.1. V0 + V1 are complete on CPU and real Metal: an L1 tier accepts any legal `cu_seqlens` under one total/count, with auditable case realization and replayable evaluation-input identity. V2 Dyn-0 is also complete (mechanism only, not a performance claim): `tasks/varlen_kda` declares the work-list/handoff workspace capacities, `V2ArgSet` allocates the possible-stage union, and two separate real-Metal fixtures prove "preprocessing + sequential chain" and "real piece handoff" (never one posing as the other). The NPU side is a test-only simulator with `simulated=true/publication_allowed=false`; its work list is compared byte for byte against the buffer the device plan stage really writes, and its consumer-first spin model is a host-scheduling argument, never Metal safety evidence. Dyn-0 splits a chain evenly (`divmod`, every piece non-empty), and the multi-stage native descriptor is deduplicated: kernel 0's full bindings live only at the top level, later stages keep their own, reducing fixture B from 7980 B to 7209 B without dropping ABI information or widening the closed bound; native digests and affected evidence are regenerated. V3 (restricted C99 / Wasm Host Program) is complete as a *mechanism*: `host/tiling.c` is compiled once with standard Clang + `wasm-ld` into a `wasmtime` module (new Store per call), the generated `host_contract.h` is prepended from OpSpec/ABI, and `assembled_host_sha256`/`wasm_sha256` enter `NativeExecutable.descriptor["host_program"]` and therefore native identity (Wasm bytes never join Schema-4 entries). Only `access=read` inputs enter the contract (`read_write` stays device-owned); the 10 mixed read inputs are all typed-mapped, with fp16/bf16 using a `uint16_t` raw-bit ABI, and large inputs use only metadata plus `sf_input_read_*` whitelisted imports (including `sf_input_read_u16` raw bits); `host_tiling()==0` commits atomically, and non-zero/trap/fuel/span/guard/read-only-input violations discard everything. Workspaces are `host` / `device_zero` / `none(poison)`; a host write plus device-zero on the same block is refused at the envelope, and the timed path no longer recopies read-only inputs, rebuilds `stage_args` or double-submits zero-fill — real Metal is one Wasm call + one CommandBuffer. The candidate C99 planner (divmod split, fixed-point cost + range-cut + heap LPT, bit packing) matches the independent V2 plan byte for byte on every L1 case; the Python planner is a test-only oracle. The NPU simulator reuses the same module/ABI: same-target spans are byte-identical and different targets keep the item set, one-claim-per-item and dependency edges (claim maps/grid may differ); all reports stay `simulated=true/publication_allowed=false`. After the acceptance review, the empty-dispatch exemption is derived from the projected `host/tiling.c` entry and the native descriptor's `host_program` identity (candidate/program/assembled/wasm/native digests + run schema), never from a boolean or a self-reported mapping; post-call pointer/capacity/elem_nbytes revalidation rejects repointing a workspace (including via an import write-back), workspace-parameter aliasing is refused by the precheck, the build Engine/Module is reused in-process (constructor counters pin one Module and one new Store per call) and the cache key is the full build identity (comment-only B recompiles instead of hitting A), NPU parity compares every span and resolved scalar on one target and the deduplicated item set/unique claims/completion/semantic `(chain,piece)` edges across targets (a mode-3 target-dependent order is a positive case), a missing clang/wasm-ld/engine becomes a `backend_capability_unavailable` rejected outcome (the linear JSON path still builds), the statement carries a compilable/executable minimal host+device example, and timed evidence counts Wasm calls/command buffers/audits/stage-args rebuilds/read-only recopies. The closure grew by `tcenv/hostprogram/contract.py` and the statement bytes changed, so every task pack digest moved and P2 archived evidence needs re-recording — no P2 file was edited. V4 Dyn-1, V5 L2 capacity and V6 real-scale performance are **not** implemented. `tasks/varlen_kda` is the reference task.

**Latest override:** the historical V3 Wasm path has been replaced by native64 C99 in the bounded v2 worker. Read `docs/架构/native-host-program-ABI.md` first for Host Program work. The C compiler owns ABI layout; Python uses generated accessors and `sf_native_invoke()`, real assembled/native digests enter identity, and evidence is `tcenv.hostprogram.run.2`. Wasm runtime/toolchain/probe and the `wasmtime` dependency are removed. CPU/contract/NPU-sim and real Metal checks pass (native Host 10/10, V1 11/11, V2 19/19); three fresh uninstrumented full evaluations measure Host wall 7.583 µs, native Host 4.584 µs, candidate 238.994 µs and device 64.374 µs (medians), with 40 Metal and 128 focused CPU tests passing; see `docs/架构/native-host-program-数据流耗时报告.md`, not a P2/V6 result. P2 re-recording has not run. Pack closure is 72 (`tcenv/hostprogram/native_abi.py` added). Do not describe CPU protocol success as GPU or performance success.

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
