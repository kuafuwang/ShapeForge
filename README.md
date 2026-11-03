# ShapeForge

[English](README.md) | [简体中文](README.zh-CN.md)

**Generate specialized kernels. Prove every shard.**

ShapeForge is an evidence-first system for generating, evaluating and dispatching specialized GPU kernels. It supports kernels written by an LLM or by a person; candidate implementation freedom is not restricted to Triton translation or a fixed pattern library.

The current production target is **Apple GPU / Metal Shading Language**. Qingwei NPU support is limited to offline protocol and data simulation: no real-device correctness or performance is claimed.

## What ShapeForge does

- **Task contracts**: standard OpSpec v2 directories define semantics, inputs/outputs, tolerances and registered baselines.
- **Candidate-owned implementation**: the candidate chooses algorithms, fusion, Tiling, Host Tiling and ExecutionGeometry.
- **Shape portfolio**: a candidate ShapePlanner declares coverage buckets; multiple stable variants coexist and exact one-of-N dispatch selects at most one.
- **General fallback**: uncovered inputs use an independently validated implementation, never a disguised variant success.
- **Unseen-shard validation**: required boundary, tail and resource instances remain outside candidate tuning.
- **Evidence-bound activation**: canonical bundles, build/evaluation identity, fresh confirmation and one publication predicate decide what may run and what may be quoted.

The harness owns the task contract, ABI, safety, measurement boundary and evidence. It does not prescribe a closed set of kernel patterns.

## Current status

| Area | Status |
| --- | --- |
| R0–R6 evidence and build identity | Complete |
| G0 and A1–A6 OpSpec v2 migration | Complete |
| Shape portfolio P0–P2 | Complete |
| P3-A performance accounting | Complete |
| P3-B second tiled GEMM task | **Blocked**: paper passes; small still fails the stable fresh 1.10 gate |
| Qingwei NPU | Offline simulation only; NPU-P0 remains blocked |

P3-B is intentionally not marked complete: a good individual measurement is not a substitute for stable fresh evidence.

## Candidate and harness boundary

```text
Candidate (person or LLM)
  algorithm / fusion / coverage plan / tiling / Host Tiling / ExecutionGeometry
                              │
                              ▼
Harness
  OpSpec + ABI + pack closure + safety + bounded execution
                              │
                              ▼
Evaluation and evidence
  correctness + dispatch parity + fresh timing + canonical identity
                              │
                              ▼
Activation and publication
  exact dispatch + fallback + confirmation + revocation/resume
```

Candidates may use persistent kernels, multi-stage state machines and public native matrix/synchronization features. Arbitrary host includes, undeclared resources and sandbox escapes are refused. Runaway execution is handled by an external watchdog rather than by limiting candidate loop style.

## Quick start

Requirements: macOS on Apple Silicon, Xcode/Metal toolchain, `uv`, and the Python environment described by `pyproject.toml`.

```bash
uv sync

# Discover standard task contracts
uv run python -m tcenv.cli tasks

# Exercise a hand-written kernel through the evaluation pipeline
uv run python -m tcenv.cli naive gemm

# Read-only run summaries
uv run python -m taic.cli brief
uv run python -m taic.cli brief RUN_DIR
```

### Tests

Start with the safe CPU suite (no GPU, model calls or long checks):

```bash
uv run pytest -q -m 'not gpu and not llm and not slow'
```

Run GPU tests only when you explicitly allow real GPU use:

```bash
uv run pytest -q -m gpu
```

For one serial failure, read `logs/pytest/triage/triage_summary.md` and the Markdown/JSON handoff it points to, then reproduce with:

```bash
uv run pytest -q -n 0 <nodeid>
```

Test layout, markers and evidence rules are in `tests/README.md`.

### Paid model search and GPU use

`taic run` and `taic resume` can spend money and execute kernels on the GPU that also drives the display. They are never required for reading evidence or running the CPU suite.

```bash
# Example only; this spends money and uses the GPU
uv run python -m taic.cli run softmax --budget 1.0
```

Kernel writing goes through the local `hmz` launcher and `pi`; see `docs/架构/hmz.md`. Credentials are read from the environment or the configured credential file and are never committed.

## Evidence and performance claims

Two numbers must not be mixed up:

- **Search best**: the fastest observation during a search. It is not publishable.
- **Published speedup**: a fresh remeasurement with matching source/evaluation identity, quiet timing and a passing publication gate.

Correctness uses the repository's `tolerances + scaled_error`, including final `read_write` state. Fallback correctness and fallback time are not credited to a specialized variant.

For shape portfolios there are two accounts:

- `specialized_hit_speedup`: only shards handled by generated variants;
- `effective_end_to_end_speedup`: total work including the real fallback path.

Every shard must independently satisfy the frozen regression policy. Geomeans are descriptive only and cannot hide a local regression.

Raw observations, JSON fields and hashes alone do not prove real hardware execution. Activation also requires independent expected reconstruction and fresh evidence.

## Repository layout

| Path | Purpose |
| --- | --- |
| `tasks/` | OpSpec v2 task directories: `opspec.json`, `reference.py`, baselines and optional display/reference metadata |
| `tcenv/` | Task contracts, Track/CanonicalBundle, evaluation, Metal runtime and backend-neutral evidence |
| `tcenv/portfolio/` | Semantic domains, candidate portfolios and exact dispatch |
| `tcenv/portfolio_evidence/` | Deterministic evidence projection and strict comparison |
| `tcenv/portfolio_validation/` | Freeze, tune/unseen plans and candidate views |
| `tcenv/portfolio_performance/` | Per-shard verdicts and the two performance accounts |
| `taic/` | Search, confirmation, activation, persistence and portfolio-aware resume |
| `backends/` | Backend metadata, skills and wiki |
| `targets/` | Chip profiles, measurements and trust boundaries |
| `tests/` | `shared/`, `backends/<id>/` and support/golden suites; see `tests/README.md` |
| `docs/` | Architecture, implementation plans, security and acceptance evidence |
| `kernels/` | Confirmed kernel examples where present |

Historical v1 task fixtures are test-only. The old per-task `task.py`, `track.json`, `kernel_triton.py`, per-task LoweringProfile and generated-task registry are not the current authoring path.

## Documentation map

Start here:

- `docs/架构说明.md` — process and runtime overview;
- `docs/架构/shape-portfolio实施计划.md` — authoritative portfolio plan and P0–P4 status;
- `docs/架构/跨硬件算子生成分工_9aef8d3a.plan.md` — OpSpec v2 and migration architecture;
- `docs/架构/构建执行与证据语义升级-plan.md` — build/evaluation identity and publication evidence;
- `docs/架构/执行运行时与设备扩展.md` — local/remote execution and evaluator boundaries;
- `docs/security.md` — model-written kernel and P2 security boundaries;
- `docs/agent-skills.md` — agent skill entry points.

Stage acceptance material is under:

- `docs/架构/跨硬件算子生成分工-阶段验收/`
- `docs/架构/shape-portfolio-阶段验收/`

These are evidence records, not claims that every planned stage is complete.

## For AI-assisted contributors

Start with `AGENTS.md`, the canonical project context and safety boundary. `CONTRIBUTING.md` describes the human/AI workflow, and `.agents/skills/shapeforge/SKILL.md` provides a short onboarding skill for other agents. These files explain candidate freedom, evidence invariants, current status, tests and authorization boundaries so another AI does not have to infer the project from its Triton history.

## Historical origin

ShapeForge began as an Apple M4 Max re-implementation of *AI as a compiler: compiling Triton kernels without the Triton compiler*. The original path lowered Triton text directly to MSL. That work remains visible in the Git history and in the optional `triton-msl` comparison path.

It is no longer the project boundary. Current candidates may originate from Triton, handwritten MSL, new algorithms or a shape planner. The current architecture centers on OpSpec v2, candidate-owned implementation, shape portfolios and evidence-bound activation.

## Security and responsible use

A Metal kernel can hang the GPU that drives the display. Use the bounded evaluator instead of launching arbitrary candidate source manually. GPU hang tests are guarded by `TCENV_ALLOW_GPU_HANG_TESTS=1` and are not part of the normal suite.

No CUDA, PTX, CuTe, TIRx, NVIDIA scheduling rules, AscendC, CANN or related vendor-specific knowledge is used in the production task packs or candidate guidance.
