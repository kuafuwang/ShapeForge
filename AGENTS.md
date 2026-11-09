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
- **varlen V3（受限 C99 Host Program）已完成，执行路径随后换成 native64**：当时的 Wasm（`wasm-ld` + `wasmtime`）验收是历史机制。现行编译、加载和证据见下一条。仍然有效的合同（只读输入进 host、返回 0 才提交、workspace 三态、NPU 只模拟）由 native64 继承。V4 Dyn-1 已按实测跳过。方案见 `docs/架构/varlen支持计划.md`。
- **V5 L2 capacity / continuous batching 已完成（CPU/协议与真实 Metal 机制验收，非 V6 性能结论）**：OpSpec v2 档位可以声明 `capacity` + `workload_profiles`（与 `seq_lens` 互斥，固定档仍不能与变长档同题混用），`SemanticInstanceV2` 新增 `capacity`/`workload_profiles` 和 `variable_length_kind`（判别不能用 `cu_seqlens is None`，L2 故意没有 canonical 分段）；L2 的 symbols 是容量，buffer/ABI span/workspace/Host 存储都按容量分配。`tcenv/taskcontract/runtime_shape.py` 分开 capacity/resolved/logical 三套形状，case→profile 的分配确定（profile 按 id 排序、case 按序取模），`gen_inputs` 只收到声明 profile 的期望值、不能自证。`cu_seqlens` 按容量分配：前缀是本次偏移（i32、从 0、非降、末项等于 resolved total、长度 `num_seqs+1`），尾部必须逐项等于声明 sentinel（负数，拒绝便利 0）；empty batch/负值/非单调/错误末项/非 sentinel 尾部/超容量全部 typed 拒绝，empty sequence 合法。正确性只比较 `output[:resolved_total]` 与 `final_state[:resolved_num_seqs]`，logical 区域外逐字节校验声明的 padding poison（fp16 位型在 float32 展开后不同，检查必须落在存储字节上；输入侧 task 写、harness 跑前校验，`write` 输出侧 harness 每 trial 重写再校验）。`SfHostResolved` 对 L2 有真字段 `total_tokens`/`num_seqs` + 生成 accessor，Python 每次调用写入、C 合同检查拒绝候选改写、HostRun 从 ABI 回读；**L1/fixed 生成 header、tensor 绑定与 derived ABI digest 与 HEAD 逐字节相同**，既有 assembled digest、descriptor 长度与 Metal 记录不受影响；`varlen_kda` 的 opspec/instance digest 因新增档位前进（未改档位的任务如 `varlen_scale` 语义身份不变）。HostRunEvidence 新增 `resolved_shape` 与 `evaluation_input_digest`（L1/fixed 为 `None`）。dispatch 仍以 capacity 实例为唯一域点，resolved 不进 guard，fallback 仍按域点独立。同一 build 连续不同 total/count 不重编译（resolved 不进 contract/header/编译单元，CPU 测试逐 case 断言 build 身份不变且计划随 resolved 变化）；本轮**没有**加 Host Program cache，也没有 overlap 测量。NPU 模拟器仍只调度真实 host work list，全部 `simulated=true/publication_allowed=false`。`varlen_kda` 新增 `capacity` 档使该题 opspec/pack digest 前进，闭包 72→73（`taskcontract/runtime_shape.py`），当时 P2 仍 deferred；最终重录见 P2-F。复核后补齐五个 release blocker（`tests/backends/metal/test_varlen_capacity_evidence.py` 逐项固定）：(1) L2 的 error scale（含自定义 `error_scale` 钩子）只在 logical 区域计算，钩子只拿到 logical 视图，`output` 全对而 `final_state` 全错会失败，换 padding poison 不改变 logical 误差；“整缓冲区尺度”被 typed 拒绝。(2) resolved 轴的别名（第二个符号取值等于容量 `total_tokens`/`num_seqs`）在文档校验与 prepare/比较两处 fail closed，input/output/final_state 都有负例；**结果张量的 capacity 尺寸轴必须命名**（`"4"`/`"2*2"`/`"H*2"`/`"H*D"` 这类能通过 shape 代数但会让 logical 区域变成整个 capacity 缓冲区的写法被拒，非结果张量的 `dt_bias` `H*D` 仍合法），同一规则在文档层与手搓实例运行时层都有负例；capacity 的 `total_tokens == num_seqs`（>1）时结果轴上的 canonical 名字本身无法证明，这种容量声明被拒（1/1 豁免）；`V2Case.resolved_shape` 缺失不再是“无 L2 合同”，比较器按实例种类无条件进入校验并对缺失值 typed 拒绝（没有绕过 digest 重算/padding 的分支）；独立符号积（`dt_bias` 的 `H*D`）仍合法，不会误伤。(3) `evaluation_input_digest` 显式绑定声明的 `workload_profile_id` 与 canonical `seq_lens`，并校验字节推出的 total/count 等于该 profile；resolved total/count 仍来自字节；digest 自述 `identity_scope=boundary-bytes+shape-metadata+declared-profile`，不再暗示 byte-complete。(4) HostRunEvidence 每个 ordered case 都保留（同一 case 的重复 trial 按内容去重），失败文档字段集与成功一致并带 resolved 形状/输入身份；新增 per-call `launch_runs`，不被 Dyn-0 静态 dispatch 去重吞掉。(5) benchmark 按声明 profile 顺序计时多个 resolved 批次（`benchmark_batches` + `benchmark_mixture_problems` 自检/负例），padding 与 setup 仍在 timed 外；mixture 与单形状基线不可比，因此不发布 speedup（`speedup_withheld`）。exact-dispatch/伪造 resolved/fallback 混入都有负例；resolved 仍不是域点。真实 Metal 新增 `test_varlen_capacity_gpu.py` 3/3：同一 build 执行 9 个 resolved batches、每 case 三次 trial 的 shape/digest/ordinal 全部一致，continuous-batching benchmark 按声明顺序覆盖四种 `(total,count)`，resolved 篡改候选被 `incorrect` 拒绝且 HostRunEvidence 保留。既有 native Host/V1/V2 复验仍为 40/40，最终定向 CPU 回归 282 passed。一次 full mixture 实测 candidate median 193.07 µs（cv 0.0445）、native Host 7.375 µs、Host wall 10.333 µs、device 22.791 µs，680 次 native call 对应 680 个 CommandBuffer；mixture speedup 按合同 withheld，不是 P2/V6 发布数字。
- **Host Program 已完成 native64 迁移（CPU/协议与真实 Metal 验收）**：当前生产路径不再使用 Wasm；`host/tiling.c` 由系统 Clang 编译为 worker-local `.dylib`，编译/加载/执行只在 bounded v2 worker 内，控制机只见结构化结果。独立 `native_abi.py` 让 C 编译器拥有布局，Python 只用生成的 accessor 和 `sf_native_invoke()`；`assembled_host_sha256` 覆盖真实完整编译单元，`native_sha256` 覆盖 dylib，HostRunEvidence 为 `tcenv.hostprogram.run.2`。mapped inputs、typed reads、有效 span、pointer/metadata/shape/target/guard 调用后校验、最终符号边界、fresh replay、NPU native parity 和 crash/timeout 均有 CPU 测试；旧 Wasm runtime/toolchain/probe 与 `wasmtime` 依赖已删除。pack closure 的代码文件现为 **74**（含 `tcenv/hostprogram/native_abi.py` 与 `tcenv/taskcontract/runtime_shape.py`），P2 最终重录见 P2-F。真实 Metal native Host 10/10、V1 11/11、V2 19/19 与 CPU/协议 128 passed 已通过；三次独立未插桩 full evaluation 的 Host wall 中位数 **7.583 µs**、native Host **4.584 µs**、candidate median **238.994 µs**、device **64.374 µs**，C planner 插桩约 0.167 µs。C 批量校验、Metal 地址绑定与 C 原子直接提交消除了 timed 中间对象，未删除 fresh-input/read-only/guard/span 检查。完整数据流见 `docs/架构/native-host-program-数据流耗时报告.md`。这组微秒数是小 fixture 测量，不是 V6 六档结论，也不是 speedup 发布；V6 与 P2-F 见下述条目。历史 V3 Wasm 验收记录仅作历史，不是当前执行路径。
- **varlen V6 真实规模验收已完成**：`tasks/varlen_kda` 新增 D=128、H=64/96、total_tokens=8192 的 fixed/mixed/uniform 六档；`tests/support/golden/metal/varlen/chunked_gform.metal` 是 **msl-single** 的真实 BT=8 chunked/`simdgroup_float8x8` 候选，G-form `A/Base` 用矩阵乘，residual 后保留 fp32 token-order state/output。独立审查修复 residual 漏掉当前 token decay 的根因（KD/QD 用 `exp(L)`，不是 `exp(L_prev)`），小规格 9/9 L1 case、旧顺序负例和六档 canonical 均由 `test_varlen_v6_chunked_gpu.py` 固定。六档全部完成 9-case search+fresh correctness、15 次 E2E、benchmark/golden sidecar 和 active manifest；candidate cv≤0.0473，相对登记的**朴素 Torch-MPS 逐 token baseline** fresh speedup 为 8.7205×–10.3821×。该比值包含消除 Python/MPS 框架与 dispatch 开销，不能外推为相对优化 KDA/CAKE2/CUDA/CANN/vendor kernel 的通用加速；六档均 candidate provider、无 fallback/NPU 模拟计时混入。六档正确性由 `tests/backends/metal/test_varlen_v6_chunked_gpu.py` 固定。V1/V2/native Host/V5/V6 Metal 回归 51/51，完整 CPU 快速套件 4549 passed、13 skipped。未启动付费搜索。
- **P2-F 最终重录已完成，但历史 start-surface 锚点仍按原样拒绝**：安静窗口依次执行 `--write-golden`、`--record-evidence`、`--write-manifest`；14/14 archived 与 live 域实例均 fresh confirmed、0 timing-blocked，active manifest/pointer generation=1，live==archived stable identity，checkpoint/resume/revoke 通过。P0/P1 anchors 更新为 58，P2-A/B/C/D 为 9/11/16/10（46 CPU）+5 live Mac，manifest 28 文件，完整 CPU 4549 passed。默认 verifier 仍在既有 `verify_baseline()` 的 45 文件 start-surface drift 停止；`acceptance-baseline.json`、`start-worktree.sha256` 和允许漂移集合未修改，不能宣称该历史锚点已修复。新增 `P2/F.md` 记录，不改写 A–E。
- **固定长度 KDA 标准任务与复杂 bundle 的离线/CPU 示例已完成**：`tasks/kda` 是独立固定 `[N,H,D]` ABI（无 `cu_seqlens`/varlen case-domain），含 fp64 oracle、final_state、naive known_good 和登记的 Torch-MPS baseline。候选 fixture 由 native64 C99 planner、三个 Metal piece stages 和 fp32 handoff workspace 组成；planner 对 N 做 min(3,N) 均衡切分，single/small/tail 的 span 逐字节固定，bundle 预检保留三 stage，单文件冒充 bundle/缺 Host Program 均被拒绝。聚焦测试 12 passed，任务发现/迁移锚点 78 passed，完整 CPU 快速套件 4561 passed、13 skipped；`tcenv tasks` 对没有 paper 的任务按默认档回退的 OpSpecError 也已修复。示例与边界见 `docs/架构/fixed-kda-bundle示例.md`。**真实 Metal、fresh 性能与付费搜索尚未执行**。
- **多 DSL / TileLang MT2 已完成实现、真实 Metal 证据、只读 verifier 与独立 reviewer/RLAR 终审（done=true）**：新增 `tilelang-pipeline/2` 且保留 `/1` 的 `interface_only` frozen replay；`/2` 仅对固定人工 candidate-owned `kernel_pipeline` 开放。单阶段、候选自有两阶段 work-list 与同语义 MSL 对照均通过真实 build/ABI/read_write correctness/dispatch/fresh rotation，shipped registry 直接通过；r4/r6 各 7/7 GPU 节点，r4/r6 candidate/bundle/artifact identity 一致，最终聚焦 CPU 回归 105 passed。frontend/沙箱失败为 partial，越界为 unsafe，native/生成物篡改被独立审计拒绝；r5 noisy 与早期 ABI/poison/路径失败均保留。`device_ir` 的随机 attempt SourceName 已规范化，独立构建 artifact 可复现。验收包 `tests/support/golden/shared/多DSL-Track-阶段验收/MT2/` 的 verifier/manifest 通过，独立 DeepSeek Flash reviewer 返回 `done=true`，日志摘要见 `MT2/review.json`；签署范围仍只限固定人工真实 Metal 机制。`available` 不代表生产 `taic run`、LLM 生成、Graph、正式 fresh confirmation 或 publication 已开放；MT0 初审时第二条 DSL deferred；后续独立范围复核已决定不再新增第三条 DSL，MT6 已按 MSL + TileLang 范围完成，MT3 Graph 未开始。P2-F 已重录。`/3` 可被 `--track` 选中，默认搜索交付仍是 `msl-single`；这不是 speedup publication。
- **MT4（TileLang LLM/人工候选生成与工具链）已完成并通过独立 review**：`tilelang-pipeline/2` 的 prompt、ABI→PrimFunc 签名、非执行 packer、`.taic` envelope、显式 DSL projection、production evaluate、Orchestrator、snapshot restore 与 resume dry-run 均已接入；`/1` 继续 interface_only。真实 DeepSeek Flash 一次 correctness-only ReLU 生成花费 $0.295137132，6 个 case（含 IEEE）通过，replay dispatch 为 `relu_kernel_kernel grid=(64,1,1) threads=(512,1,1)`；`best_speedup=1.0` 只是占位，非性能结论。真实 `evaluate_candidate` 和 fake-agent snapshot restore 通过；MSL 兼容回归 114、完整 CPU 安全套件 4963/26/13（26 个均为已知 P2 epoch）。`test_tolerance_verdict` 已按 frozen 组合并行化，102 节点/132 秒。`track_adapter._evaluate_in_process` 回归已由 actor 最小修复，Reviewer 独立重跑 5/15/8/6/148/102 节点并返回 `done=true`，见 `tests/support/golden/shared/多DSL-Track-阶段验收/MT4/review.json`。本条只记录当时的 correctness-only 接入。fresh confirmation、TileLang `/3`、MT6/MT7 见后面的条目；Graph 与 speedup publication 仍未开放。
- **MT6 固定 `/3` ReLU 切片 fresh 已通过，并已登记、终审完成（`publication_allowed=false`）**：`/3` 稳定 fresh subject 进 BuildProfile，`metal-build.2` descriptor 只含稳定 identity，attempt 证据在 BuildEvidence.log。worker 诊断定位 native batch 抖动；每轮 barrier 的 r7 仍全部 ratio >1.10，未采用；改为 **native concurrent + 每 2 轮 barrier** 后，r8 三次 confirmed，ratio 为 0.9596/0.8347/0.8963，全部 ≤1.10，达到 `stable_fresh_timing=true`、`fresh_confirmation_stable=true`、`performance_gate_passed=true`。`/2`、MSL 和 P3 不增加 barrier；`/3` 已按 r8 证据登记 available，固定 `/3` ReLU 切片已通过独立 review（`reviewer_done=true`）。`affine_ramp` 三 Track 真实批量探索已完成：`msl-single`、`msl-bundle`、TileLang `/3` 均 confirmed/eligible，ratio 为 0.156119/0.153255/0.074707，TileLang 为 `exploration_winner`，`formal_publication=false`、`same_fresh_input=false`、`cross_track_runtime_portfolio=false`，`track_explore_reviewer_done=true`。r10 因 observation 后补字段、fallback 缺执行证据和 build duration 漏算而被 reviewer 拒绝，`review-r10-rejected.json` 保留；r15 重新生成同期 observation/summary，补齐 `observed-execution.json`、真实 `native_batch_calls=0`、完整 build evidence 和成本/turn 校验，已通过独立 review（`review-r15.json`）。证据位于 `tests/support/golden/shared/多DSL-Track-阶段验收/MT6/evidence/track-explore-r15/`，MT6 verifier 通过。`scope-decision.json` 已正式将 MT6 范围调整为 **MSL + TileLang**，不再新增第三条 DSL，`scope-review.json` 已独立签署范围复核通过，完整 MT6 `done=true`，`publication_allowed=false`。新 registry digest `9b85fa11…`、prompt SHA-1 `162d027f…` 已记录，旧 MT2/MT4 包未改写。重启后经授权恢复固定 TileLang/Torch，source-only 双构建通过，不冒称原整树相同。21 次归档 benchmark 为 12 confirmed + 9 timing_uncertain，20 次 golden 均 confirmed；19 个新 CPU 节点加兼容 41、扩展 75、MSL 148、默认 registry CPU 40、track-explore 45 和 2 个真实 Metal 节点通过。见 `tests/support/golden/shared/多DSL-Track-阶段验收/MT6/review.json`。
- **本轮多 DSL Track 实施已按 MSL + TileLang、`kernel_pipeline` 范围完成，MT7 已通过最终独立评审**：`README*`、`backends/metal/wiki/tilelang.md`、`docs/agent-skills.md` 已更新；`/api/v2/track-explore` 只读展示 r15 winner/成本，非发布权威。`tests/support/golden/shared/多DSL-Track-阶段验收/MT7/` verifier、137 个 CPU 节点和 r15 重算通过。MT6 已按 MSL + TileLang 范围完成，第二条 DSL 不再新增。MT3 Graph 未启动，不属于本轮完成条件。`publication_allowed=false`，不开放第三条 DSL、Graph、无界生产搜索或 speedup publication；后续这些工作必须另开阶段和授权。
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
| OpSpec v2 and task directories | `docs/架构/v2协议参考.md` |
| Build/evaluation identity | `docs/架构/构建与评测身份.md` |
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

Every search uses two committees, `first` and `second`, one member each. In `taic.toml`, first is DeepSeek `deepseek-flash`, second is `kimi-code` / `k3.1-pro`. Kernels are written only through `hmz`, normally one new session per round even when both members use the same model. Provider retries are bounded and only for structured provider `turn_failed` with no submission; ordinary no-submission answers, timeouts/process failures and available checkpoints are not retried.

A speedup is published only when the confirmation gate binds the remeasured kernel; search best is separate.

The execution runtime (local vs split-control evaluation, `execution.json`, outbox, evaluator epoch, self-check bridge and P2 isolated verification) is documented in `docs/架构/执行运行时与设备扩展.md`. Local remains the default. Remote evaluation requires explicit `--remote-evaluator-config` and a human-registered site allowlist.

## Agent skills and collaboration

- General agent onboarding for this repository: `.pi/skills/shapeforge/SKILL.md` (also linked from `.agents/skills/shapeforge`).
- Existing canonical skills: `.pi/skills/harness-feedback/` and `taic/flows/kernel_once/skills/taic-tools/`.
- Aliases and entry points: `docs/agent-skills.md`. Edit canonical sources instead of duplicating skill text.
- Human/AI contribution workflow: `CONTRIBUTING.md`.

Credentials are never committed. Do not add API keys, tokens, local caches, raw model transcripts or unrelated large artifacts to the repository.
