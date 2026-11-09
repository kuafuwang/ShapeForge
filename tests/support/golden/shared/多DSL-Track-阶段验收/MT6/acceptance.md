# MT6：TileLang fresh confirmation 与性能边界

状态：**MT6 按 MSL + TileLang 范围已完成。固定 `/3` ReLU 切片及独立 review 已通过；`affine_ramp` 的三 Track 真实批量探索 r15 已通过独立 review；MT6 范围调整已通过独立范围复核。r10 保持 rejected，`publication_allowed=false`。**

## 已确认的 `/2` 阻断

`tilelang-pipeline/2` 的真实构建把完整 `tilelang_source_only_identity` 放入
`BuildProfile.compile_profile`。该身份包含 attempt 事实，所以同一候选的 search 与 fresh 两次独立构建会得到
不同 provenance；R5 comparator 拒绝它们是正确行为，不能通过忽略字段或放宽比较来制造确认通过。

## 当前 `/3` 合同切片

- 新增独立 `tcenv/track/tilelang_confirmation.py`，定义 `TILELANG_CONFIRMATION_TRACK_VERSION=3`、
  `/3` 合同提案与 `FreshBuildSubject`；不修改旧 `tilelang.py`、registry 或 task-pack closure；
- 可注册/可构建的 `SUPPORTED_TRACK_VERSIONS` 仍是 `(1,2)`，registry 没有 `/3`；
- `/1` 合同 digest 保持 `689f86802ed8e040a37461677c4ffb922b9be1b8ea4c3e73b18faeeed9100022`；
- `/2` 合同 digest 保持 `03541a31c94ef815de38c43e1ed260fd31c7c74955598c3652031d641429d1df`；
- `/3` 合同 digest 为 `6c1f379f45d97db23571c15f2e6de2e57b4ab80184741684909b548b7d928997`；
- registry digest 保持 `596474db2d9132c16998d6f406eb1f10286acf86e501da734caa958cb4350fec`；
- `expected_fresh_subject.cache_key` 包装完整 `/2` pre-build subject 并加入 `/3` 身份，在 attempt 变化时稳定；
  完整 source-only identity 随 attempt 前进。稳定 subject 进入 BuildProfile；`/3` native descriptor 只保存稳定的
  candidate/frontend/artifact/fresh subject，完整 attempt 证据保存在 BuildEvidence.log；
- subject 绑定完整 candidate manifest（包括 geometry）、frontend/environment 与 Track version。

固定测试：`tests/shared/test_tilelang_mt6_contract.py` 3 个节点，加 MT1b/MT2 合同兼容节点共 **25 passed**。
没有 frontend 执行、Metal device、模型或付费调用。

补充边界：当前工作树的 MT2 只读 verifier 仍因**本切片之前已经存在**的 task-pack digest 漂移拒绝
（归档 `b02ec81e…`，当前观察 `527c8d8a…`）。`/3` 新模块不在 closure 内，旧 `tilelang.py` 已恢复；
本切片不重录、不修改 MT2 历史证据，也不把该既有漂移伪装成全绿。

## 当前 builder/probe CPU 接缝

- `TileLangFreshMetalBuilder` 的外层 Track 是 `/3`，nested `TileLangSourceOnlyBuild` 明确使用冻结 `/2` TrackSpec；
  `audit_fresh_registration` 固定外层 build_adapter=3、source_only_track=/2、interface_only 和无 metal_dispatch；
- `/3` BuildProfile 只含稳定 `tilelang_fresh_subject`，两个不同 attempt 的完整 source-only identity 产生相同
  BuildProfile；`/2` 仍含原 `tilelang_source_only_identity`，旧行为未改；
- `/3` 使用新的 `tcenv.track.tilelang.metal-build.2` 稳定 descriptor：保存 outer `/3`、稳定 artifact/frontend
  identity 与 fresh_subject，不包含 attempt 或 attempt-specific source-only digest；完整 attempt JSON 保存在
  BuildEvidence.log，避免 search/fresh 的 native_digest 必然漂移。控制端 artifact audit 独立重算并拒绝
  fresh_subject/artifact 漂移，`/2` descriptor schema 与行为不变；
- probe 只观察 staged frontend/environment 和固定 Metal 工具链，不 import/执行候选；真实候选构建仍属
  bounded worker。`/3` 固定 math=safe，生成物请求其他 math 时 typed 拒绝，避免 probe/build 身份猜测；
- 5 个结构化 CPU 节点通过；加上述合同与 MT1b/MT2 兼容共 **30 passed**，另有 MT4 **5 passed**。
  这些测试使用固定 probe 替身，没有打开 Metal device。

## 当前 confirmation/activation CPU 路由

- `confirmation_input` 对 `/3` 使用独立 `DSL_PROTOCOL_EPOCH=3`；`/1`、`/2` 仍 typed 拒绝，MSL epoch 不变；
- `/3` geometry resolution 直接读取 candidate-owned kernel_pipeline geometry，不进入 MSL launch parser；
- independent expected 通过 `TileLangFreshMetalBuilder.probe` 重建，不调用 `combined_candidate_source` 或
  MSL probe；`V2LocalEvaluator.confirmation_expected` 复用同一路径；
- `confirm_candidate → confirm_subject → _evaluate` 全程显式传递 track_id/version，projection 不可换成其他 Track；
- `_evaluate` 先 `require_available`，再取 `/3` adapter builder/evaluator/geometry，保留合同声明的全部 cases；
  interface_only 在任何 `run_portfolio_task` 前拒绝；
- `activation_expectation` 的 cache key、preflight、projection 和 expected 均绑定 `/3`，真实 dispatch replay
  选中 candidate provider；resume 专项验收仍未完成；
- 路由测试使用 **test-only available registry fixture**，生产 registry 没有 `/3`。fixture 不是真实 Metal 证据；
- 7 个新节点通过；MT6 新节点加 MT1b/MT2 兼容共 **37 passed**，加 MT4 共 **42 passed**，另有
  MSL/MT2 聚焦兼容 **148 passed**。没有真实 search/fresh 两次构建、GPU dispatch 或计时。

## 当前 resume 与身份负例

- `track_factory` 登记 `/3` 的精确 implementation identity；resume 构造历史 system text 时显式使用
  `/2` 的相同 DSL/ABI 题面，实际 binding/evaluator 始终为 `/3`。`track_prompt.py` 与 prompt digest 未改，
  新 `/3` 生成仍需独立 prompt epoch，不能借 resume 路径开放；
- run_config `/3` → Orchestrator → binding/evaluator 全部保持 `/3` 和 `kernel.taic`；注入 `/2` evaluator 会拒绝；
- 内存 Snapshot `restore(dry_run=True)` 恢复同一 `.taic` candidate identity，不写 state/candidate 文件、不调用
  agent。它证明 dry-run 接缝，不冒充真实继续一轮搜索或 GPU replay；
- independent expected 拒绝 search/confirmation 协同修改 `tilelang_fresh_subject`，拒绝把 attempt-only
  `tilelang_source_only_identity` 偷渡回 compile_profile，也拒绝两次 artifact_set 不一致；
- environment、candidate source 和 geometry 漂移都会改变 fresh subject；interface_only 仍在任何真实 run 前拒绝；
- MT6 现有 19 个新节点，加 MT1b/MT2 合同兼容共 **41 passed**，加 MT4 共 **46 passed**；
  扩展 resume/MSL seam 集合 **75 passed**，另有 MSL 聚焦 **148 passed**。全部未执行 GPU/模型。

## 真实 Metal 观察（已授权，固定 DeepSeek ReLU 候选，无模型调用）

- 两次独立 `evaluate_candidate` 均 correctness `ok`，真实 dispatch `relu_kernel_kernel`；BuildProfile、
  artifact set、assembled/native digest 相同，descriptor 为稳定 schema 2；
- 三次完整 fresh sidecar 全部归档于 `evidence/real-metal-r1/`，identity 均 `match`，search/fresh 的
  BuildProfile、artifact set 与 native digest 均一致；已执行的 golden 均 `confirmed`；

| 观察 | candidate µs | MLX µs | noise CV | benchmark | golden |
| --- | ---: | ---: | ---: | --- | --- |
| r1 | 1.077661 | 2.061041 | 0.049014 | confirmed | confirmed |
| r2 | 1.022240 | 2.096011 | 0.052326 | rejected: timing_uncertain | 未执行（当时 benchmark 断言先停止） |
| r3 | 1.059930 | 2.164056 | 0.055401 | rejected: timing_uncertain | confirmed |

### 第二次三次连续复测（`real-metal-quiet-r2`）

`record_quiet.py` 固定同一 freeze/validation，每次无论 benchmark verdict 都继续 golden 并立即落盘：

| 观察 | candidate µs | MLX µs | noise CV | benchmark | golden |
| --- | ---: | ---: | ---: | --- | --- |
| quiet r1 | 1.032957 | 2.138829 | 0.049599 | confirmed | confirmed |
| quiet r2 | 1.066524 | 2.002200 | 0.050415 | rejected: timing_uncertain | confirmed |
| quiet r3 | 1.066645 | 2.043678 | 0.042570 | confirmed | confirmed |

新序列三次 golden/identity 全部通过，但仍有一次 quiet timing 失败，`stable_fresh_performance=false`。
累计六次归档 benchmark 为 3 confirmed + 3 timing_uncertain，五次已执行 golden 均 confirmed；没有
删除旧序列或选择 quiet r1/r3 作为稳定证据。

### 第三次三次连续复测（`real-metal-quiet-r3`）

用户关闭高负载窗口后，loadavg 降为 `1.70/1.94/2.09`，仍完整执行三次：

| 观察 | candidate µs | MLX µs | noise CV | benchmark | golden |
| --- | ---: | ---: | ---: | --- | --- |
| quiet-r3 r1 | 1.086352 | 2.137161 | 0.056823 | rejected: timing_uncertain | confirmed |
| quiet-r3 r2 | 1.109033 | 2.070269 | 0.068143 | rejected: timing_uncertain | confirmed |
| quiet-r3 r3 | 1.305472 | 2.055951 | 0.049178 | confirmed | confirmed |

累计九次归档 benchmark 为 4 confirmed + 5 timing_uncertain，八次已执行 golden 均 confirmed。
第三轮只是恰好低于 0.05，不能覆盖前两次，也不能用来宣称稳定性能。

- 两个 GPU 节点最终通过：一个验证独立构建身份，一个验证 fresh 机制并保留 typed timing_uncertain；
  这不是把 timing rejection 改成通过，性能门仍明确失败；
- 三次归档中两次 quiet timing 未通过，所以 `stable_fresh_performance=false`、
  `performance_gate_passed=false`、`publication_allowed=false`。不得选择 r1 冒充稳定性能或登记 available；
- 更早只有 triage/控制台摘要的观察见 `history-notes.md`，不用于结论。

### 重启后的前端恢复与 r5 复测

重启删除 `/tmp` 中的固定安装。离线恢复失败后，经用户授权联网重装，最终固定 TileLang wheel、
3651 个包文件、METADATA、RECORD、Torch 2.15.0 wheel 和 20 个 dependency 版本均通过核验；source-only
双构建通过。恢复记录为 `evidence/frontend-recovery-r1.json`，使用独立的
`/tmp/shapeforge-frontends/tilelang-0.1.15-recovered` 路径，不冒称整棵树与原 MT0 完全相同，
也不占用原 MT0 staging 路径；Metal 测试通过显式环境变量选择该恢复 epoch。

| 观察 | candidate µs | MLX µs | noise CV | benchmark | golden |
| --- | ---: | ---: | ---: | --- | --- |
| quiet-r5 r1 | 1.132360 | 2.068670 | 0.066357 | rejected: timing_uncertain | confirmed |
| quiet-r5 r2 | 1.113377 | 2.059194 | 0.058081 | rejected: timing_uncertain | confirmed |
| quiet-r5 r3 | 1.121835 | 2.109227 | 0.069574 | rejected: timing_uncertain | confirmed |

累计二十一次归档 benchmark 为 12 confirmed + 9 timing_uncertain，二十次已执行 golden 均 confirmed。
r4 的 frontend-unavailable 失败也保留，未计入完成观察。

## native batch 诊断与 r6 稳定 timing

worker 内 GPU timestamp/execution 诊断显示 native batch 覆盖 60/60 个 CommandBuffer；禁用后该计数为 0，
同一候选的 noise_cv 从约 0.055 降到 0.0342。`/3` 因此选择现有的 PyObjC fallback，`/2`、MSL 和 P3 不变。
CommandBuffer commit/wait、dispatch、guard、rotation 和 fresh 身份保持，CPU 节点覆盖 `/3` 禁用、`/2` 保留。

| 观察 | candidate µs | MLX µs | latency ratio | noise CV | benchmark | golden |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| quiet-r6 r1 | 2.677774 | 2.167233 | 1.235573 | 0.036855 | confirmed | confirmed |
| quiet-r6 r2 | 2.666922 | 2.040350 | 1.307090 | 0.048499 | confirmed | confirmed |
| quiet-r6 r3 | 2.666083 | 2.144223 | 1.243379 | 0.035570 | confirmed | confirmed |

**quiet timing 已稳定**：三次均 confirmed、golden/identity 全通过，`stable_fresh_timing=true`。
但 PyObjC fallback 的候选 wall 明显变慢，三次 ratio 全部超过 1.10，所以
`performance_gate_passed=false`、`stable_fresh_performance=false`，不能登记 available 或发布。
初始把 confirmed 当成性能通过的摘要保存在 `summary-initial.json`，修正后的 `summary.json` 明确区分
quiet timing 与性能门。

## r7 失败实验与 r8 通过

每轮 barrier 虽安静但候选过慢，r7 全部 ratio >1.10 且第三轮 timing_uncertain，因此不采用：

| 观察 | candidate µs | MLX µs | latency ratio | noise CV | benchmark | golden |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| r7 r1 | 2.627028 | 2.143071 | 1.225824 | 0.045787 | confirmed | confirmed |
| r7 r2 | 2.568027 | 2.100909 | 1.222341 | 0.036359 | confirmed | confirmed |
| r7 r3 | 2.585115 | 2.055990 | 1.257358 | 0.061118 | rejected: timing_uncertain | confirmed |

改为 **native concurrent + 每 2 轮 barrier**，保留并发收益但限制调度宽度。r8 三次全部 confirmed，且
全部 ratio ≤1.10：

| 观察 | candidate µs | MLX µs | latency ratio | noise CV | benchmark | golden |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| r8 r1 | 1.980827 | 2.064181 | 0.959619 | 0.035107 | confirmed | confirmed |
| r8 r2 | 1.686507 | 2.020542 | 0.834681 | 0.031840 | confirmed | confirmed |
| r8 r3 | 1.895005 | 2.114146 | 0.896345 | 0.047098 | confirmed | confirmed |

固定 `/3` ReLU 切片因此达到 `stable_fresh_timing=true`、`fresh_confirmation_stable=true`、
`performance_gate_passed=true`。`/2`、MSL 和 P3 不增加 barrier；/3 仍未注册、
`publication_allowed=false`，完整 MT6 仍需后续登记/恢复、范围验收与独立 review。

## 默认 registry、activation 与真实 resume

`backends/metal/tracks.json` 已登记 `/3` available，证据为 MT6 GPU 测试、本文件和 r8 summary。新 registry digest
为 `9b85fa111f7ccde5c95dfc61b84c2c2c6a142ae5cc8e171c0fae340ac042d138`，新 prompt SHA-1 为
`162d027fb7a2f23e0c279cdf98b0c2d2e323ab7fb032321440d2abcc0119da11`。旧 MT2/MT4 封闭包未改写，
`registry-epoch.json` 分类记录其 verifier epoch drift；40 个 registry/MT2/MT4 CPU 节点通过。

`evidence/default-registry-r3/` 的默认路径验证全部通过：

- 默认 `/3` build/evaluate correctness `ok`，dispatch `relu_kernel_kernel grid=(64,1,1) threads=(512,1,1)`；
- benchmark fresh `confirmed`、identity `match`，`latency_ratio=0.8036134694700262`；
- golden fresh `confirmed`、identity `match`；
- activation `resolved`、provider `candidate`，fresh subject 与候选一致；
- fake-agent 真实构建 + **非 dry-run** resume 恢复原 `.taic` 和 `/3`，state.json 落盘，模型调用为 0；
- r1 的 staging 失败和 r2 的摘要序列化错误分别保留，未被覆盖。

因此 `/3` 为 `registered=true/selectable=true/default_registry_validated=true`，但
`publication_allowed=false`、`done=false`、`reviewer_done=false`。下一步是独立 review；完整多 Track 批量和
第二条 DSL 范围仍另行处理。

## 独立 review（固定 `/3` ReLU 切片已通过）

独立 Reviewer 接受 `review.json`，并保存 Actor 的 `actor-review-request.json`。Reviewer 独立重跑：

- 47 个 MT6/MT1b/MT2/MT4 核心与兼容节点；
- 40 个 registry/factory 节点；
- 21 个 native batch 节点；
- 163 个 MT0/MT1b 隔离、resume 与 seam 节点；
- 51 个 MT5/结构节点；
- MT6 verifier 与 `git diff --check`。

Actor 的 4 个测试文件修复被确认是必要的版本适配：避免默认 `/3` 与测试 fixture 重复，保留 pre-/3
和 live `/3` registry digest 双向核对，未注册负例改为 `/4`，没有削弱断言。Native C/ctypes 的
`barrier_interval` 一致，`/3` 为 native concurrent + interval 2，`/2` 与 MSL 为 0。

签署范围仅 `/3` 固定 ReLU 登记切片；`reviewer_done=true`，但完整 MT6 `done=false`，第二条 DSL、
Graph、无界 track-explore 和 speedup publication 均未签署，`publication_allowed=false`。

## 三 Track 真实批量探索（`track-explore-r15`）

任务改为 `affine_ramp`，因为 ReLU 无 workspace、不能合法进入 `msl-bundle`。三条固定候选分别使用
MSL launch、bundle Host Tiling + schedule 和 TileLang `/3`，均通过各自 preflight。运行使用 fake agent，
模型调用与费用均为 0，独立 run_dir/cache 和现有 fresh confirmation。

最终三条路线均 `confirmed`、identity `match`、`latency_ratio <= 1.10`：

| Track | ratio | 结果 |
| --- | ---: | --- |
| msl-single | 0.156119 | confirmed / eligible |
| msl-bundle | 0.153255 | confirmed / eligible |
| tilelang-pipeline/3 | 0.074707 | confirmed / eligible / **exploration winner** |

`summary.json` 为 `exploration_winner`，不是正式发布：

- `formal_publication=false`、`same_fresh_input=false`、`cross_track_runtime_portfolio=false`；
- 每条 Track 保留自己的 portfolio freeze、confirmation seed 和 fresh input identity；
- `track_explore_reviewer_done=false`，完整 MT6 `done=false`，第二条 DSL 仍 deferred；
- 全部 run_dir、confirmation、候选 manifest、缓存观察和配置保存在 `evidence/track-explore-r15/`。

r10 因 observation 后补字段、fallback 缺执行证据和 TileLang build duration 漏算而被 reviewer 拒绝，
`review-r10-rejected.json` 保留，未改写。r15 重新生成同期 observation/summary，补充
`observed-execution.json`、真实 `native_batch_calls=0`、完整 build.evidence.duration_ms 和成本/turn 校验，
并优先使用 build evidence 统计真实 lowering 时间。所有路线使用 `pyobjc_fallback`，未修改 0.05 噪声门或
1.10 latency gate。`tests/shared/test_track_explore.py` 45 个节点和 MT5 verifier 通过。

## r15 独立 review（已通过）

`review-r15.json` 与 `actor-review-request-r15.json` 记录了本次独立复核。Reviewer 独立重算
`summarize_track_explore`，与归档 summary 完全一致，并重跑 142 个兼容节点、55 个合同/registry 节点、
MT0/MT5 verifier 和 MT6 verifier。确认 observation/summary 同期一致、真实 fallback、build duration 成本、
turn cap、`.taic` 身份分离、attempt/batch 和 r10 拒绝历史均成立。Actor 未修改仓库文件。

`track_explore_reviewer_done=true` 仅表示 r15 批量探索切片通过，完整 MT6 `done=false`，第二条 DSL、
Graph、无界 track-explore 和 speedup publication 仍未签署，`publication_allowed=false`。

## 测量环境诊断（`timing-analysis-r6.json`）

- 二十一次归档噪声归因：candidate 14 次、torch_mps 7 次、MLX 0 次。说明噪声门并不总是在惩罚
  candidate，不能把当前现象简单归因于 TileLang kernel 本身；
- candidate 与 baselines 仍使用同一 interleaved wall、15 reps、最多 3 attempts、16 rotation sets，
  `QUIET_CV=0.05` 未改，identity 全 match；
- 诊断时 loadavg 为 `2.61/2.47/2.32`，Activity Monitor、iTerm2、pi、WindowServer 仍有明显 CPU 活动；
  AC 供电、无热警告、GPU lock wait 为 0。未关闭用户应用、未改计时代码；
- 关闭高负载窗口后的第三轮仍有两次 timing_uncertain；重启并关闭 Spotlight 后，r5 在约 88% CPU idle
  下仍三次 timing_uncertain，说明仅降低交互负载不足以形成稳定证据。
  当前 `environment_ready_for_rerecord=false`，不再盲目挑选轮次；需要新的安静环境证据或独立的
  计时诊断结论，且不得修改阈值。

## 范围决定：不再新增第三条 DSL

`scope-decision.json` 正式将 MT6 范围调整为 **MSL + TileLang**，且 `scope-review.json` 已独立签署。
MSL 与 TileLang 已提供两种独立 DSL 和三条真实交付/构建/执行路线，继续新增第三条 DSL 只会增加安装、
隔离和证据成本，没有新的工程价值。范围复核接受该缩小，完整 MT6 `done=true`，但
`publication_allowed=false`，不把探索结果当成 speedup 发布。r10 拒绝仍保留。

## 后续实现顺序

1. 用 r8 证据评估 `/3` 登记，并在默认 registry 下验证真实 build/evaluate、activation 和非 dry-run resume；
   登记前保持未发布，随后进行独立 review。完整多 Track 批量与第二条 DSL 范围另行处理；
2. 真实稳定证据齐备前生产 registry 不注册 `/3`；证据复核后再决定是否登记 available，不能借 `/2` 的证据；
3. 最后独立 reviewer 签署。当前 `done=false/reviewer_done=false/publication_allowed=false`。

本阶段不启动第二条 DSL、Graph、无界 `track-explore`、真实模型搜索或性能发布，也不改变 P3-B blocked 状态。
