# ShapeForge agent context

Start here before editing. This file is the canonical project context for AI-assisted contributors. Do not infer the project from its old Triton-only description.

## What this repository is

ShapeForge generates, evaluates and dispatches specialized GPU kernels. Candidates may be written by an LLM or a person. The current production backend is **Apple GPU / Metal Shading Language**; Qingwei NPU is **offline simulation only**, not a real-device correctness or performance target.

The project is not a Triton-to-MSL converter. Triton is a preserved v1 input format and an optional comparison path. Current work uses OpSpec v2, candidate-owned algorithms and tiling, shape portfolios, exact dispatch, independent fallback and evidence-bound activation.

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
   - Do not delete or rewrite R0–R6, G0, A1–A6, P0–P2 or other acceptance evidence.
   - Completed phase packages and their verifier/manifests are closed boundaries. Put new evidence in a new phase/epoch rather than silently changing an old one.
   - Do not modify six-RPC/field-number/HMAC/BlobChannel contracts to make a feature fit.

## Current status

- R0–R6, G0, A1–A6, P0–P2 and P3-A are complete.
- v1→v2 task migration (B4/B5) and msl-bundle delivery (B0–B6) are complete; `tasks/<op>/` v2 entries are the only live task path.
- **varlen V0 + V1（L1 任意合法分段）已完成**：canonical `seq_lens` 派生总形状并用于调度/性能档，同一 ABI 下的 permuted/skewed/prime/chunk-boundary/max-handoff/zero-length/single-sequence/fresh 是 Evaluation Cases，实际分段由 `evaluation_input_digest` 重放。CPU 与真实 Metal 验收通过；P2 证据已按新 pack digest 重录。
- **varlen V2 Dyn-0 已完成（机制正确，不是性能结论）**：`tasks/varlen_kda` 在 OpSpec 里声明 work_items/item_counts/handoff_flags/handoff_states/token_stats 的容量公式（MAX_SPLIT_PER_CHAIN=3），workspace 按 possible-stage 并集惰性分配；静态 grid/stage/parity 不变，多余 TG/stage 读运行时 count 后退出。两类候选 fixture 在真实 Metal 上分开验收——“预处理 + 顺序链”（`prep_chain.metal`，不冒充 handoff）与“真实 piece handoff”（`piece_handoff.metal`，handoff_states + flag 协议，stage 间 buffer barrier，不跨 TG 自旋），并覆盖 partial tail、空序列、跨 head 共用、越界 work list、consumer-first、未发布/未清 flag 等负例。规划用均衡切分（`divmod`，每 piece 非空，`len=4` → `2/1/1`），真实设备写出的 work list 与 NPU adapter 逐字节一致；“上一轮 READY + 旧 handoff state”负例用三个控制实验钉住归因（正常发布必须 `ok`；负例是有限中等误差而非 protocol poison；改旧值误差跟着变）。多阶段 native descriptor 已去重：kernel 0 的完整 bindings 只在顶层保存，后续 stage 各自保存 bindings，四阶段 fixture 从 7980 B 降为 7209 B，不删 ABI 信息、不扩大上限，digest 与报告按新结构重生成。NPU 只有模拟器（`tests/support/npu_sim/varlen_worklist.py`，simulated=true/publication_allowed=false），其 consumer-first 自旋死锁是 host 调度论据，不作为 Metal 安全证据（Metal fixture 是 serial stage + buffer barrier）。
- **varlen V3（受限 C99 / Wasm Host Program + NPU parity）已完成（机制验收，不是性能结论）**：`host/tiling.c` 是候选拥有的受限 C99 Host Program，build 时用标准 Clang + `wasm-ld` 编译一次为 Wasm（`wasmtime==43.0.0`），每次调用新建 Store；`host_contract.h` 由 OpSpec/ABI 自动生成并前置拼接，`assembled_host_sha256` 与 `wasm_sha256` 进入 `NativeExecutable.descriptor["host_program"]` → native digest，Wasm bytes 不进 Schema-4 entries。小输入（≤64 KiB/合计 ≤1 MiB）typed 映射，大输入只给元数据并走 `sf_input_read_*` 白名单 import；`host_tiling()==0` 才原子提交，非 0/trap/fuel/span 越界/guard/只读改写全部丢弃。workspace 三态 `host/device_zero/none(poison)`，host 写与 device_zero 同块在 envelope 拒绝；`host_inside_timing` 不再重拷只读输入、`stage_args`/fill 预建、host-written 块不 device zero，真实 Metal 实测每迭代一次 Wasm + 一次 CommandBuffer。题面只对 `cu_seqlens` 任务给完整 header/C99 子集/返回码，`host-probe` 可单独编译试跑，缺工具链是 `backend_capability_unavailable`。候选侧 C99 planner（divmod 均衡切分 / 定点成本 + range-cut + heap LPT + 位打包）在 8 个 L1 case 上与独立 V2 `plan_pieces` 逐字节一致，Python 只作测试 oracle；NPU 模拟器复用同一 module/ABI，同 target span 逐字节一致、异 target item 全集/唯一认领/依赖边等价，全部 `simulated=true/publication_allowed=false`。真实 Metal 9/9 case、timing 证据与 host 失败/envelope 负例由 `test_varlen_hostprogram_gpu.py` 固定（mixed full 候选中位数 ≈443 µs，host ≈200 µs、device ≈65–67 µs、E2E ≈439–443 µs，仅机制）。闭包 70→71（`hostprogram/contract.py`）且题面字节变化使所有 task pack digest 前进，P2 归档证据需按安静窗口重录（未改写、未放宽）。**V4 Dyn-1、V5 capacity、V6 真实规模性能仍未实施**，方案见 `docs/架构/varlen支持计划.md`。
- P3-B is **blocked**: paper GEMM passes, small GEMM does not have stable fresh evidence at or below the 1.10 latency-ratio gate. Do not select a lucky run or delete the small shard to claim completion.
- P3-C has not started final acceptance.
- Qingwei NPU-P0 remains blocked. Simulation must carry `simulated=true` and `publication_allowed=false`.

The living plan is `docs/架构/shape-portfolio实施计划.md`. Read it before assuming a phase is pending or complete.

## Where to look

| Need | Authoritative location |
| --- | --- |
| Project overview | `README.md`, `README.zh-CN.md` |
| Current portfolio plan/status | `docs/架构/shape-portfolio实施计划.md` |
| varlen (packed sequence) support | `docs/架构/varlen支持计划.md` |
| OpSpec v2 and migration architecture | `docs/架构/跨硬件算子生成分工_9aef8d3a.plan.md` |
| Build/evaluation identity | `docs/架构/构建执行与证据语义升级-plan.md` |
| Local/remote execution runtime | `docs/架构/执行运行时与设备扩展.md` |
| Evaluation, tolerance and timing semantics | `docs/架构/评测.md` |
| Search, selection and confirmation | `docs/架构/搜索.md` |
| Run layout, index and retrieval | `docs/架构/数据层.md` |
| Model/P2 security | `docs/security.md` |
| Test layout and failure protocol | `tests/README.md` and layer READMEs |
| Agent skill entry points | `docs/agent-skills.md` |
| Kernel-writing agent reading list | `docs/写核代理读什么.md` |
| Feedback and harness review | `docs/代理反馈.md` |
| Trusted-harness plan index | `docs/可信harness/README.md` |
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

The executable placement rules are `tests/shared/test_structure.py`. Shared tests must not import a backend; Metal-specific tests belong under `tests/backends/metal/`.

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

Do not spend money or run GPU kernels unless the user explicitly asks. `taic run` and `taic resume` do both. Read-only summaries are:

```bash
uv run python -m taic.cli brief
uv run python -m taic.cli brief RUN_DIR
```

`taic query` makes no model/GPU call, but `best`/`failures`/`sessions`/`models` refresh the derived index, so it is not strictly read-only.

Every search uses two committees, `first` and `second`, one member each. In `taic.toml`, first is DeepSeek `deepseek-flash`, second is Moonshot `kimi-k3`. Kernels are written only through `hmz`, normally one new session per round even when both members use the same model. Provider retries are bounded and only for structured provider `turn_failed` with no submission; ordinary no-submission answers, timeouts/process failures and available checkpoints are not retried.

A speedup is published only when the confirmation gate binds the remeasured kernel; search best is separate.

The execution runtime (local vs split-control evaluation, `execution.json`, outbox, evaluator epoch, self-check bridge and P2 isolated verification) is documented in `docs/架构/执行运行时与设备扩展.md`. Local remains the default. Remote evaluation requires explicit `--remote-evaluator-config` and a human-registered site allowlist.

## Agent skills and collaboration

- General agent onboarding for this repository: `.pi/skills/shapeforge/SKILL.md` (also linked from `.agents/skills/shapeforge`).
- Existing canonical skills: `.pi/skills/harness-feedback/` and `taic/flows/kernel_once/skills/taic-tools/`.
- Aliases and entry points: `docs/agent-skills.md`. Edit canonical sources instead of duplicating skill text.
- Human/AI contribution workflow: `CONTRIBUTING.md`.

Credentials are never committed. Do not add API keys, tokens, local caches, raw model transcripts or unrelated large artifacts to the repository.
