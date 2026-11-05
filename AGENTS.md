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
- **varlen V3（受限 C99 / Wasm Host Program + NPU parity）已完成（机制验收，不是性能结论）**：`host/tiling.c` 是候选拥有的受限 C99 Host Program，build 时用标准 Clang + `wasm-ld` 编译一次为 Wasm（`wasmtime==43.0.0`），每次调用新建 Store；`host_contract.h` 由 OpSpec/ABI 自动生成并前置拼接，`assembled_host_sha256` 与 `wasm_sha256` 进入 `NativeExecutable.descriptor["host_program"]` → native digest，Wasm bytes 不进 Schema-4 entries。只有 `access=read` 的输入进入合同（`read_write` 由 device 独占）；mixed 的 10 个只读输入全部 typed 映射（q/k/v/g/beta 为 fp16，用 `uint16_t` raw-bit ABI；fp32/i32 用原生 C 类型），大输入只给元数据并走 `sf_input_read_*` 白名单 import（含 `sf_input_read_u16` 的 fp16/bf16 raw bits）；`host_tiling()==0` 才原子提交，非 0/trap/fuel/span 越界/guard/只读改写全部丢弃。workspace 三态 `host/device_zero/none(poison)`，host 写与 device_zero 同块在 envelope 拒绝；`host_inside_timing` 不再重拷只读输入、`stage_args`/fill 预建、host-written 块不 device zero，真实 Metal 实测每迭代一次 Wasm + 一次 CommandBuffer。题面只对 `cu_seqlens` 任务给完整 header/C99 子集/返回码，`host-probe` 可单独编译试跑，缺工具链是 `backend_capability_unavailable`。候选侧 C99 planner（divmod 均衡切分 / 定点成本 + range-cut + heap LPT + 位打包）在 8 个 L1 case 上与独立 V2 `plan_pieces` 逐字节一致，Python 只作测试 oracle；NPU 模拟器复用同一 module/ABI，同 target span 逐字节一致、异 target item 全集/唯一认领/依赖边等价，全部 `simulated=true/publication_allowed=false`。真实 Metal 9/9 case、timing 证据与 host 失败/envelope 负例由 `test_varlen_hostprogram_gpu.py` 固定（mixed full 候选中位数 ≈443 µs，host ≈200 µs、device ≈65–67 µs、E2E ≈439–443 µs，仅机制）。复核后额外收紧：空 dispatch 的例外不再信任布尔字段或自报 `host_program`，而是从投影候选的 `host/tiling.c` entry 与 native descriptor 的 `host_program` 身份逐项校验 candidate/program/assembled/wasm/native digest 和失败 run schema；msl-single/linear JSON/伪造 host_program+failure 仍按 parity mismatch 拒绝；调用成功后重新核对所有输入/workspace pointer 与 capacity/elem_nbytes（含把 workspace 指向输入、import 写回 ABI pointer 两种负例），workspace 参数别名/赋值/取地址在预检拒绝；build 的 Engine/Module 由同进程 executor 复用，构造计数器固定“1 个 Module + 每次新 Store”，缓存键是完整 build 身份（仅注释差异的 B 会重编译而不是命中 A）；NPU parity 同 target 比较全部 span 元数据/字节/resolved scalars，异 target 只比较去重 item 全集/唯一认领/完成集合/`(chain,piece)` 语义依赖边（`SF_PLANNER_MODE 3` 固定 target-dependent 顺序正例）；缺 clang/wasm-ld/engine 在 build 边界得到 `backend_capability_unavailable` 的 `rejected` 结果（linear JSON 不受影响，不怪候选）；timed 证据新增 Wasm 调用/CommandBuffer/audit/stage_args/只读重拷计数（真实 Metal 145/145/1/0/0）。题面另含一个可编译/执行的最小 host+device 示例；native descriptor 实测 7,514 B。闭包 70→71（`hostprogram/contract.py`）且题面字节变化使所有 task pack digest 前进，P2 归档证据需按安静窗口重录（未改写、未放宽）。**V4 Dyn-1、V5 capacity、V6 真实规模性能仍未实施**，方案见 `docs/架构/varlen支持计划.md`。
- **Host Program 已完成 native64 迁移（CPU/协议与真实 Metal 验收）**：当前生产路径不再使用 Wasm；`host/tiling.c` 由系统 Clang 编译为 worker-local `.dylib`，编译/加载/执行只在 bounded v2 worker 内，控制机只见结构化结果。独立 `native_abi.py` 让 C 编译器拥有布局，Python 只用生成的 accessor 和 `sf_native_invoke()`；`assembled_host_sha256` 覆盖真实完整编译单元，`native_sha256` 覆盖 dylib，HostRunEvidence 为 `tcenv.hostprogram.run.2`。mapped inputs、typed reads、有效 span、pointer/metadata/shape/target/guard 调用后校验、最终符号边界、fresh replay、NPU native parity 和 crash/timeout 均有 CPU 测试；旧 Wasm runtime/toolchain/probe 与 `wasmtime` 依赖已删除。pack closure 为 **72**（新增 `hostprogram/native_abi.py`），P2 最终重录继续 deferred。真实 Metal native Host 10/10、V1 11/11、V2 19/19 与 CPU/协议 128 passed 已通过；三次独立未插桩 full evaluation 的 Host wall 中位数 **7.583 µs**、native Host **4.584 µs**、candidate median **238.994 µs**、device **64.374 µs**，C planner 插桩约 0.167 µs。C 批量校验、Metal 地址绑定与 C 原子直接提交消除了 timed 中间对象，未删除 fresh-input/read-only/guard/span 检查。完整数据流见 `docs/架构/native-host-program-数据流耗时报告.md`。该结果不是 P2 发布或 V6 结论；P2 最终重录与 V5/V6 尚未完成。历史 V3 Wasm 验收记录仅作历史，不是当前执行路径。
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
