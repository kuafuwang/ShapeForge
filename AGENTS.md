# ShapeForge agent context

Start here before editing. This file is the canonical project context for AI-assisted contributors. Do not infer the project from its old Triton-only description.

## What this repository is

ShapeForge generates, evaluates and dispatches specialized GPU kernels. Candidates may be written by an LLM or a person. The current production backend is **Apple GPU / Metal Shading Language**; Qingwei NPU is **offline simulation only**, not a real-device correctness or performance target.

The project is not a Triton-to-MSL converter. Triton survives only as historical task text in read-only old runs: the v1 task entry and the Triton→Metal comparison baseline are both retired. Current comparison baselines are the registered Torch-MPS/MLX entries in `tasks/<op>/baselines.json`. Current work uses OpSpec v2, candidate-owned algorithms and tiling, shape portfolios, exact dispatch, independent fallback and evidence-bound activation.

Read the default English entrypoint `README.md`, or `README.zh-CN.md` for Chinese. `docs/README.md` does not exist; do not recreate a second project entrypoint.

## Non-negotiable invariants

1. **Candidate freedom**
   - Candidates own algorithms, fusion, coverage plans, Tiling, Host Tiling and ExecutionGeometry.
   - Do not introduce PatternResolver, LoweringRuleRegistry, StagePartitioner, TileConnector or per-task tiling registries.
   - Persistent kernels, loops, multi-stage state machines and public native matrix/synchronization features are allowed.
   - Arbitrary host includes, undeclared resources and sandbox escapes remain forbidden. Runaway execution belongs to the external watchdog.

2. **Harness authority**
   - The harness owns the task contract, ABI, pack closure, safety, measurement boundary and evidence.
   - Correctness uses only the repository's `tolerances + scaled_error`, including final `read_write` state and declared workspace clearing rules.
   - Host Tiling/prepare work that belongs to a candidate must stay inside its required timing boundary.

3. **Identity and evidence**
   - `candidate_sha256 == provenance.source_sha256`.
   - Frozen execution geometry enters the artifact set before recomputing `provenance.artifact_set_digest`.
   - Build evidence is not evaluation evidence. Baseline/oracle/environment identity is part of evaluation identity.
   - Native descriptors/digests are cross-audits, not a replacement for canonical identity.
   - Search best is not published speedup. Publication requires matching fresh confirmation, quiet timing and the unique publication predicate.
   - Raw observations and self-reported JSON are not hardware proof. Independent expected reconstruction and fresh execution have separate trust roles.

4. **Shape portfolio**
   - Candidates own the ShapePlanner and coverage plan; the harness validates and selects, it does not invent the plan.
   - Exact one-of-N dispatch, stable variant identity and independent general fallback are mandatory.
   - Required boundary/tail/resource instances must remain unseen and disjoint from tune. Master seeds and unseen keys never enter candidate views.
   - Fallback correctness and fallback time never count as variant correctness or speedup.
   - `specialized_hit_speedup` covers generated variants only; `effective_end_to_end_speedup` includes real fallback work. Every shard independently satisfies the 1.10 gate. Geomeans cannot hide a local regression.

5. **History**
   - Do not rewrite the bytes of a closed acceptance package while it remains in the working tree. A package may be removed from the working tree when no active test or active code opens it by path. Record the removal in `tests/support/golden/shared/retired.json` with the last complete commit, tree hash and restore command. Git history keeps the bytes. There is no per-name ban and no count cap.
   - Packages that left the working tree stay listed in `tests/support/golden/shared/retired.json`. Later tests must not open those directories by path.
   - Completed phase packages and their verifier/manifests are closed boundaries. Put new evidence in a new phase/epoch rather than silently changing an old one.
   - Do not modify six-RPC/field-number/HMAC/BlobChannel contracts to make a feature fit.

## Current status

- **当前身份（2026-11-26 核对）**：`tests/support/live_anchors.json` 记着任务包闭包 **78 个文件** / `99acb3a63f4abceabd73ad80ef457f1a1eb7ed02e740dec95756b7497269a8c9`，以及 `prompt_sha1 = 25d2e428b91c105179599871da05fd811647a9d2764fa944381504f859859364`（现场值用 `uv run python -c "from taic import trust; print(trust.current_prompt_sha1())"` 复核）。判断今天的状态请看 `tests/support/live_anchors.json`、`tests/support/golden/shared/acceptance-epochs/` 里最新的纪元文件，以及自己跑出来的结果。**按阶段记的历史账本已拆到 [docs/架构/阶段状态账本.md](docs/架构/阶段状态账本.md)**，那里的文件数、摘要、指纹与测试计数只对当时成立。
- R0–R6, G0, A1–A6, P0–P2, P3 (P3-A/B/C) and P4 are complete. P4 is the offline Qingwei NPU protocol boundary; NPU-P0 stays blocked.
- **今天仍然成立的**（2026-11-26 核对，以代码为准）：
  - 生产路径只有 v2。v1 任务入口、`tcenv eval` / `naive` / `triton-baseline`、Triton→Metal 比较基线都已退役，`MetalBackend.compiler_baseline()` 恒返回 `{}`。
  - 可用 Track：`msl-single/1`、`msl-bundle/1`、`tilelang-pipeline/2`、`tilelang-pipeline/3`；`tilelang-pipeline/1` 仍是 `interface_only`。默认搜索交付仍是 `msl-single`。
  - 发布：`publication_allowed=false` 是唯一对外口径；只有 fresh confirmation 能发布，search best 不算。付费搜索与真实 GPU 实验都要具名授权。
  - 确认：一次 fresh `confirm_subject`（先搜索种子 `reps=3`、再 fresh 种子 `reps=15`）；`confirm_batches`/`confirm_top` 是遗留字段。
  - harness 默认：`harness_auto=false`、`harness_shadow=true`、`harness_review=false`。
  - 设备：生产只有 Apple GPU / `metal`；清微 NPU 只有离线模拟，NPU-P0 仍 blocked。
- **历史账本**：[docs/架构/阶段状态账本.md](docs/架构/阶段状态账本.md)（P2-I…P2-V、G10 / G10 follow-up、H0–H3、varlen V0–V6、MT2/MT4/MT6/MT7、P2-F 等，按阶段与日期混记）。新阶段往下追加，不回填旧条目。

The living plan is `docs/架构/shape-portfolio实施计划.md`. Read it before assuming a phase is pending or complete.

## Where to look

| Need | Authoritative location |
| --- | --- |
| Project overview | `README.md`, `README.zh-CN.md` |
| Current portfolio plan/status | `docs/架构/shape-portfolio实施计划.md` |
| Phase-by-phase status ledger (historical, not current) | `docs/架构/阶段状态账本.md` |
| varlen (packed sequence) support | `docs/架构/varlen支持计划.md` |
| Multi-DSL Tracks and their delivery registry | `docs/架构/多DSL-Track实施计划.md`, `backends/metal/tracks.json` |
| OpSpec v2 and task directories | `docs/架构/v2协议参考.md` |
| Build/evaluation identity | `docs/架构/构建与评测身份.md` |
| Local/remote execution runtime | `docs/架构/执行运行时与设备扩展.md` |
| Evaluation, tolerance and timing semantics | `docs/架构/评测.md` |
| Search, selection and confirmation | `docs/架构/搜索.md` |
| Run layout, index and retrieval | `docs/架构/数据层.md` |
| Model/P2 security | `docs/security.md` |
| New contributor tutorials | `docs/教程/` |
| Offline documentation reader | `docs/visual/index.html` (rebuild with `scripts/build_doc_visual.py`) |
| Test layout and failure protocol | `tests/README.md` and layer READMEs |
| Agent skill entry points | `docs/agent-skills.md` |
| Kernel-writing agent reading list | `docs/写核代理读什么.md` |
| Feedback and harness review | `docs/代理反馈.md` |
| Trusted-harness plan index | `docs/可信harness/README.md` |
| Experiment/attempt evidence index | `docs/可信harness/实验索引.md` (update after every paid/GPU experiment) |
| Typed failure codes and what to do next | `docs/失败码.md` |
| Run-time and agent incident notes | `docs/运维/README.md` |
| Environment self-check | `scripts/doctor.py` |

Backend code lives under `tcenv/backends/<id>/`; backend metadata, skills and wiki under `backends/<id>/`; chip profiles, measurements and trust under `targets/<id>/`.

The current task path is `tasks/<op>/opspec.json` + `reference.py` (with optional `reference.json`, `baselines.json` + `baselines.py`, `display.json`, `known_good.metal`). Preserved v1 is test-only. Do not revive `task.py`, `track.json`, `kernel_triton.py`, per-task LoweringProfile or handwritten central task registries.

## Tests and failure handling

Read `tests/README.md` first, then the applicable layer README:

- `tests/shared/README.md`
- `tests/backends/README.md`
- `tests/backends/metal/README.md`
- `tests/support/README.md`
- `tests/support/golden/README.md`

The executable placement rules are `tests/shared/test_structure.py`. Shared tests must not import a backend; Metal-specific tests belong under `tests/backends/metal/`. Entry docs and the skill alias trees are checked by `tests/shared/test_docs_integrity.py`: dead links, dead `#anchor`s, missing repository paths, and a retired entrypoint that only survives as a tombstone.

Fast safe suite (no GPU, model or long checks):

```bash
uv run pytest -q -m 'not gpu and not llm and not slow'
```

Drop `and not slow` for the exhaustive CPU suite. Plain `uv run pytest -q` also runs GPU tests; only use it with explicit GPU permission. `-m gpu` and `-m llm` need the same permission.

Debug one failure serially:

```bash
uv run pytest -q -n 0 <nodeid>
```

A failure writes `logs/pytest/triage/triage_summary.md`. Read it and the Markdown it points at. The JSON block between `---BEGIN_AI_FAILURE_HANDOFF---` and `---END_AI_FAILURE_HANDOFF---` is the ground truth and includes a single-process reproduction command. Classify old/environment failures separately from new regressions; never delete tests, skip/xfail GPU nodes or weaken assertions to force acceptance.

Run environment-mutating commands from the repository root, not a temporary directory. Do not hide progress with `| tail`. Use bounded timeouts and live output for long commands.

## Search, cost and devices

Do not spend money or run GPU kernels unless the user explicitly asks. `taic run`, `taic resume` and `taic track-explore --config` all do; `track-explore` starts a real single-Track run for every Track whose `run_dir` has no `result.json` yet, while `--summarize-only` reads existing evidence and starts nothing. Read-only summaries are:

```bash
uv run python -m taic.cli brief
uv run python -m taic.cli brief RUN_DIR
```

`taic query` makes no model/GPU call, but `best`/`failures`/`sessions`/`models` refresh the derived index, so it is not strictly read-only.

Every search uses two committees, `first` and `second`, one member each. In `taic.toml`, first is DeepSeek `deepseek-flash`, second is `kimi-code` / `k3.1-pro`. Kernels are written only through `hmz`, normally one new session per round even when both members use the same model. Provider retries are bounded and only for structured provider `turn_failed` with no submission; ordinary no-submission answers, timeouts/process failures and available checkpoints are not retried.

A speedup is published only when the confirmation gate binds the remeasured kernel; search best is separate.

The execution runtime (local vs split-control evaluation, `execution.json`, outbox, evaluator epoch, self-check bridge and P2 isolated verification) is documented in `docs/架构/执行运行时与设备扩展.md`. Local remains the default. Remote evaluation requires explicit `--remote-evaluator-config` and a human-registered site allowlist.

## Agent skills and collaboration

- General agent onboarding for this repository: `.pi/skills/shapeforge/SKILL.md` (also linked from `.agents/skills/shapeforge`).
- Existing canonical skills: `.pi/skills/shapeforge/`, `.pi/skills/harness-feedback/` and `taic/flows/kernel_once/skills/taic-tools/`.
- Aliases and entry points: `docs/agent-skills.md`. Edit canonical sources instead of duplicating skill text.
- Human/AI contribution workflow: `CONTRIBUTING.md`.

Credentials are never committed. Do not add API keys, tokens, local caches, raw model transcripts or unrelated large artifacts to the repository.
