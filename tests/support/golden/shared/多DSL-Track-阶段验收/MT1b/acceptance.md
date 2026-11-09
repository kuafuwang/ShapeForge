# MT1b：TileLang Track 合同、构建身份与 adapter 接缝（CPU/协议范围完成，`interface_only`）

## 1. 范围结论

本包只完成获授权的 **MT1b：TileLang-first 的 Track 合同、多版本 registry、构建身份与
portfolio/search/preflight/confirmation 接缝**。用户已明确确认进入 MT1b，并在 MT0 整体仍 blocked
（唯一剩余阻断是第二条非 MSL DSL，按决定 deferred）的前提下批准先推进 TileLang 一条 DSL。

本包是 **CPU/协议验收**：没有运行 GPU/NPU/Metal kernel，没有调用模型或付费搜索，没有安装 Halide
或任何新依赖，没有修改 hmz，没有嵌套 hmz/Agent/subagent，没有 `git add/commit/reset/clean/stash`。

**TileLang Track 登记为 `interface_only`：不可被新 run 选择、没有 Metal build/dispatch、不可确认、
不可发布。** 它的 source-only codegen 是真实实现（bounded 围栏、0 dispatch、
`publication_allowed=false`），但静态 registry 登记与 source-only 工件都不构成 Metal、正确性、
计时、性能或生产可用性结论。

### 1.1 r3 收尾复核的结论（本轮）

r3 只做收尾与独立复核，不扩大范围。发现并处理了四类事实：

1. **两个真实的生产 source-only 缺陷（已修，不是能力缺失）。**
   - `tcenv/backends/metal/tilelang_build.py` 的 PrimFunc 类型检查用的是 `tvm.tir.PrimFunc`，而
     pinned `tilelang 0.1.15` 自带 TVM 只有 `tvm/tirx`（没有 `tvm/tir`），`@T.prim_func` 实际产出
     `tvm.tirx.function.PrimFunc`。旧检查在围栏内会 `AttributeError`。已改为 `tvm.tirx.PrimFunc`。
     在 staged 安装上做了**只解析不 lowering** 的核对：`hasattr(tvm, "tir") == False`、
     `hasattr(tvm, "tirx") == True`、`isinstance(scale_kernel, tvm.tirx.PrimFunc) == True`。
   - 构建接缝的正例原先使用 MT0 历史 **r3** 的 parser 失败输入（`tilelang.language.tir` 已不存在）。
     现在正例使用 MT0-**r5** 的合法程序（`3ebead19…`），并在测试里断言“只修 r3 的旧 import 与
     `@T.prim_func` 装饰器就逐字节等于 r5”，**r3 归档本身一个字节没动**（`1c839a81…`）。
2. **冻结 `(track_id, track_version)` 的投影与确认/激活全链已全部钉住（已收紧）。**
   `search_candidate.preflight_submission`、`V2LocalEvaluator.confirmation_expected`、
   `Orchestrator._confirm_v2` → `search_candidate.confirm_candidate` →
   `portfolio_confirmation.confirm_subject` → `activation_expectation` →
   `DispatchManifest.from_confirmations` → `validate_confirmation_for_activation` 以及
   `track_factory._confirmation_identity` 之前存在按 registry 默认（最高可选版本）解析或漏传
   `track_version` 的缺口（其中前次 `ec9a6b95` 会话被 `/flow stop` 中断时，`confirm_candidate`
   调 `confirm_subject` 仍停留在去版本负例变异态，本轮已补回 `track_version=track_version`）。
   现在全链都显式传冻结版本，并由 `test_the_msl_single_file_preflight_still_works` 与第 65 个节点
   `test_a_v1_run_confirms_and_activates_through_the_same_frozen_version` 逐跳固定（注册一个**可选**
   的 `msl-single/2` 后，冻结在 1 的预检、fresh confirmation、activation expectation、manifest
   activation 与 MT5 scheme-A seed 派生全部保持 `track_version == 1`）。
3. **MT1b 自己的记录与 `code_files`/`manifest` 已同步收尾。** 前次会话中断在更新确认/激活版本链与第
   65 个节点之后、重算 `identity.json`/`nodes.json`/`manifest.sha256` 之前，导致主会话运行
   `verify.py --check` 报 `taic/orchestrator.py` 等文件摘要不一致。现已修复 `confirm_candidate`
   漏传版本并按当前源码逐项重算（见 §3、§5）。
4. **归档 B1–B4 的 9 个 GPU 证据文件在复核窗口内被改过**（每个文件只有 `median_us` 一行变化）。
   这不是 MT1b 的改动，也不是 MT1b 需要的东西。现已将新观察独立归档到
   `B-evidence-rerecord-2026-11-07/raw/`，并恢复封闭包原字节；原始漂移记录保持不变。
   独立处置记录见 §6、`b-evidence-resolution.json` 与该新 epoch。

### 1.2 r2 五项复核缺陷保持修复（未回退）

1. **旧 run 的 `track_version` 在真正 resume 时不再丢失。** 原先 CLI 只用记录版本做一次查询再只
   返回 id，`Orchestrator` 又以无版本的 `require_available()` 重新选择，`TaskBinding` 也只存 id。
   现在 `(track_id, track_version)` 贯穿 `run_config` → `taic.cli` → `Orchestrator` →
   `TaskBinding` → renderer/evaluator/确认闸门；`require_msl_track`/`project_provider`/
   `run_portfolio_task`/`evaluate_candidate`/`preflight_submission` 与 evaluator 的独立 expected
   重建都显式传版本；注入评测器的 binding 版本与 run 不一致时启动拒绝。端到端负例：registry 同时
   有可选的 `msl-single/1` 与 `/2`，记录 version=1 的 run 恢复 v1（默认选择仍是 v2，断言非空转）。
   `_resume_track_choices()` 按 `track_ids` 去重；`int(block["track_version"])` 的强转被删除
   （`True`/`1.0`/`"1"`/`0`/`None`/`-1` 全部 typed 拒绝）。
2. **前端身份漂移检查不再接受“格式正确”的摘要。** `package_manifest_sha256`/
   `dist_info_metadata_sha256`/`dist_info_record_sha256`/文件数/字节数都与 MT0 实测冻结值
   （`06f3b0ad…`/`380d3486…`/`6ba83c2e…`、3651/134219728）逐项比较；`"d"*64` 被 `IdentityDrift` 拒绝。
3. **构建身份/缓存键绑定完整候选与 ExecutionGeometry。** `candidate_sha256` 用完整候选 manifest
   （路径/角色/media/字节摘要，含几何条目），并保留 `dsl_source_sha256` 溯源；
   `expected_build_subject()` 可在 build 前独立重建；同源 grid=1/999 的 cache key 与 digest 必须不同。
4. **生产 adapter 有真实 source-only build/probe 接缝。** 控制机对 staged 安装逐字节观测 → 组闭集
   `AttemptSpec` → 经现有 lower-level hmz Profile 运行（离线、空 host、attempt 内写、rlimit + 看门狗）
   → 回读并核验**内容** → 终态 `completed`/0/未超时才推导 `full_build=true`。
   `source_only_build_identity()` 不能对任意字节铸造 full build。Metal build 仍是 typed
   `unsupported_contract_capability`。
5. **对比文档与计划文档的措辞已同步**，并把 “interface_only 不可 build” 改精确：source-only
   codegen 可跑但 0 dispatch；没有 Metal build/dispatch，不可确认/发布。

## 2. 实际实现

### 2.1 冻结的 task-agnostic `kernel_pipeline` 合同

- 正式身份 `tilelang-pipeline/1`，`delivery=dsl_source_pipeline`，`execution_kind=kernel_pipeline`；
  合同文本见 [`contract.md`](contract.md)，可执行定义在 `tcenv/track/tilelang.py`。
- 候选源是 `kernel.py` / `text/x-python` / role `dsl_source`，可选一条候选自有的
  `host/execution_geometry.json`（现有 kernel-pipeline `ExecutionGeometry`）。`device_kernel`/
  `host_tiling`/`launch` 角色、`.metal/.metallib/.air` 路径、MSL media type 全部拒绝：
  **Python DSL 不会被伪装成 `kernel.metal`**。
- 不要求 `msl-single`/`msl-bundle` 的合同：没有 `taic-launch` 块、没有 HostTilingPlan、没有
  `CandidateLifecycle`；不使用 `msl_single_geometry_resolver`，几何来自共享的 `candidate_geometry`。
- 前端身份 `tilelang 0.1.15` / entry `tilelang.engine.lower` / options 摘要 `10248461…`
  （设备无关 options：target kind/keys/host、host codegen、`enable_device_compile=false`）。
  registry 只钉摘要，冻结模块拥有 options，两侧每次构造 adapter 时交叉核对。
- source-only 与 Metal build 是**两个显式模式**：本 epoch 只实现 `source_only`（0 dispatch），
  `metal_build` 请求是 `unsupported_contract_capability`。
- 闭集工件：`frontend_identity.json`、`kernel.metal`（**前端产出的 MSL 源**）、`device_ir.json`、
  `kernel_params.json`、`host_module.txt`；摘要规则与 MT0 探针逐字节一致。
- build 身份（`tcenv.track.tilelang.build-identity.2`）绑定：track 身份、模式、完整候选 manifest、
  DSL 源摘要、前端安装身份、options、设备无关环境（闭集四键）、尝试终态、工件行/工件集；
  `full_build` 由“完整尝试 + 内容核验”推导；`device_dispatch_calls=0`、device library/pipeline/
  buffer 全 false。
- 生产构建接缝（`tilelang_build.TileLangSourceOnlyBuild`）真的组并运行围栏尝试；成功才产出
  source-only build 记录（`tcenv.track.tilelang.source-only-build.1`，`publication_allowed=false`）。
- 隔离复用 `tcenv.worker_isolation` 的 lower-level hmz Profile；缺前端 →
  `backend_capability_unavailable`，身份漂移 → `track_frontend_identity_drift`，codegen
  超时/非零/缺工件 → `codegen_failed`（typed codegen 错误**不**被当作能力缺失）。控制机不 import
  TileLang/torch/hmz（有静态测试）。

### 2.2 `(track_id, track_version)` 多版本 registry（schema 2，新 epoch）

- `tcenv/track/registry.py` 以 `(track_id, track_version)` 为键；`backends/metal/tracks.json` 升到
  `schema_version: 2`（新 `tracks.v2.schema.json` 闭集）。同一 id 多版本共存；重复 `(id, version)` 拒绝。
- **默认选择规则冻结**：无版本参数时取该 id 的**最高可选**版本；没有可选版本时退回最高已登记版本，
  只为给出精确状态错误。冻结恢复/resume/replay 总是显式传版本，缺版本是 typed 拒绝并列出已登记版本。
- `available()` 每个 id 至多一条（最高可用版本），顺序按首次登记；`tracks` 保留全部版本。
- schema-1 文档降为 **replay-only**：`registry_from_document` 仍按原字节重建（同一 registry digest、
  同一 MSL Track digest），`load_track_registry` 拒绝把它当 live registry。
- DSL Track 的结构规则：必须声明 `candidate_source` + `frontend` + `kernel_pipeline_geometry`，
  不得借用 MSL 角色/能力；`interface_only` 不得声明 `metal_dispatch`；即使文档写 `available`，
  没有 `metal_dispatch` 的 DSL Track 也 `selectable=False`。
- **版本贯穿**：`run_config.track.track_version` → `_recorded_track_selection` →
  `Orchestrator(track_version=)` → `replace(binding, track_id=, track_version=)` →
  `TaskBinding.track`（版本化解析）→ renderer/evaluator/`require_msl_track`/`project_provider`/
  `run_portfolio_task`/`preflight_submission`/evaluator 的独立 expected 重建。记录的版本行已退役时
  拒绝续跑（列出已登记版本），不回退；注入评测器解析到别的版本时启动拒绝。worker 侧的冻结选择由
  `_frozen_track` 从冻结文档的 `track_version` 重建，不接受 host 默认 registry。

### 2.3 接缝（都只拒绝，不放宽）

| 接缝 | 行为 |
| --- | --- |
| `tcenv/portfolio/candidate.py::project_provider` | 投影前用 registry 声明的 DSL 源（role/path/media）识别内容，typed 拒绝；`track_version` 决定 `validate_for` 的 TrackSpec，并记入 `CandidateProjection.track_version` |
| `Metal search_candidate.portfolio_for_submission` | 先解析 MSL Track（带版本）；DSL Track/未知 id 在选路前 typed 拒绝，不会落入单文件分支 |
| `Metal search_candidate.preflight_submission` | 新增 `stage="track"` + 稳定 `code` 字段（`unsupported_contract_capability`），在任何域/几何/launch 解析前返回；内部投影带冻结版本 |
| `Metal search_candidate.evaluate_candidate` | 仍走带版本的 `require_available`；interface_only 在 MSL 投影前拒绝 |
| `Metal portfolio_confirmation._resolved / confirmation_expected` | `require_msl_track(version=projection.track_version)` 在 `combined_candidate_source`/`MetalSourceBuilder.probe` 之前拒绝 |
| `Metal track_factory.adapter_for_track` | 按登记实现身份返回 TileLang adapter（validator/builder/共享几何解析器）；evaluator/confirmation 解析即 typed 拒绝；identity 漂移 → “no Metal adapter implements” |
| `taic/track_prompt` | 未登记 renderer 的 prompt 适配器 → typed 拒绝（不会收到 MSL bundle 说明） |
| `taic/cli` | 新 run choices 仍只有 `msl-single/msl-bundle`；resume 解析全部 id（按 `track_ids` 去重）并返回记录版本；`_resolve_run_track` 按 `(id, version)` 选 `require_available` 并拒绝 TileLang |
| `taic/orchestrator` + `TaskBinding` | 版本进入 binding 与所有下游选择；注入评测器版本不一致即 `RunStartError`；新 run 把所选版本写入 `run_config` |
| `tcenv/v2_worker` / `select_track(frozen=...)` | 按冻结的 registry/版本重建；伪造 TileLang 冻结选择在 Track 边界即拒绝 |

## 3. 身份与 digest 证据

`identity.json` 是唯一结构化证据；`verify.py` 重算全部字段（不采信结论文本）。

| 项目 | MT1b 前（HEAD 归档） | MT1b 后（r3 实测） |
| --- | --- | --- |
| `backends/metal/tracks.json` sha256 | `00c1f53c…`（schema 1，2 track） | `92e32667…`（schema 2，3 track） |
| registry document digest | `9d568eec…` | `25f0f477…` |
| `msl-single` / `msl-bundle` track digest | `7d19a788…` / `31b10ee5…` | **逐字节不变** |
| `tilelang-pipeline/1` track digest | — | `8c491ee9…` |
| 冻结合同 digest | — | `689f8680…` |
| 前端安装身份（MT0 实测） | — | manifest `06f3b0ad…`、METADATA `380d3486…`、RECORD `6ba83c2e…`、3651 文件 / 134219728 字节 |
| 围栏内 `frontend_entry.py` sha256 | — | `5c6bedc2…`（r3 重算；r2 记录的 `a24cce38…` 是 `tvm.tirx` 修复前的 entry 文本） |
| task pack 闭包 | 73 文件 | **74**（新增 `tcenv/track/schema/tracks.v2.schema.json`；无删除） |
| 任务 pack digest | 35 个 | **35/35 全部前进**（registry.py 在闭包内） |
| prompt/TASK 源指纹 | `3fc1b835…` | **不变**（MT1b 没有改任何 prompt 覆盖的函数/文本；`identity.json` 记录交集为空） |
| 归档 schema-1 registry 字节 | `00c1f53c…`（= `git show HEAD:backends/metal/tracks.json`） | 逐字节归档在 `evidence/tracks-v1-epoch.json`，仍重建 `9d568eec…` |

MSL Track 文档不变意味着 `TrackSelection.track_digest`、presentation 语句、TASK.md、quick/timed
命令都保持原字节（MT1a golden 节点仍通过）。`task_pack_digest` 前进是 registry 文件在静态闭包内的
直接后果，不掩饰、不回填。

r3 之后 `identity.json.code_files` 逐项等于当前源码字节；verifier 的 `verify_identity` 因此不再仅因
`tcenv/taskcontract/portfolio_runner.py`、`tcenv/backends/metal/{search_candidate,tilelang_build}.py`
等后续改动失败。

### 3.1 归档 MT0 产物的 CPU 重放

`verify.py` 只用归档字节重放 `evidence/source-only-r5/` 与 `probe-result-r5.json`：

- 工件集摘要 `bc6a7300…` 与候选摘要 `3ebead19…` 逐字节重建；
- `enable_device_compile=false`、`dispatch_calls=0`、device library/pipeline/buffer 全 false、
  `online=false`、空 host allowlist、无内部 fence/wrap；
- `metal_correctness_or_performance_claim=false`、`production_ready=false`、`publication_allowed=false`。

负例：改动任一工件字节、伪造 dispatch、把 `policy_online` 改成 true、改 wheel sha、改
`artifact_set_sha256`、颠倒工件行顺序，全部 typed 拒绝。**这不是重新运行探针。**

## 4. 固定正负例与 node id

新增 **65** 个 CPU 节点（r3 原 64 个 + 确认/激活链冻结版本节点 1 个，完整 id 见 `nodes.json`）；逐个重跑 **65 passed**：

- `tests/shared/test_track_mt1b_registry.py`（15）：schema 2、schema-1 replay-only、多版本共存、
  默认/显式版本、版本漂移、重复 (id, version)、闭集（op/shape/未知 delivery/role/capability/
  media）、DSL 不伪装 MSL、必填 source/frontend/kernel_pipeline、interface_only 不得声明
  metal_dispatch、CLI choices、冻结选择往返与伪造拒绝、interface_only 不可选。
- `tests/shared/test_track_mt1b_tilelang_contract.py`（14）：冻结合同、注册⇄合同漂移、候选源
  闭集、前端缺失/漂移（含“格式正确但内容不同”的三个摘要 + 文件数/字节数 + bool/浮点）、隔离声明、
  source-only vs Metal 模式、build 身份（完整 manifest/几何/前端/环境/尝试终态）、闭集、
  垃圾/自报工件不能铸造 full build、codegen 终态、归档 r5 重放及其负例、合同模块无 frontend/
  sandbox import、不可支持的 DSL 语法是能力拒绝而不是编译错误。
- `tests/shared/test_track_mt1b_resume_version.py`（13）：真实 `run_config` → CLI → `Orchestrator`
  → `TaskBinding`/evaluator 的版本恢复（v1 记录在 v2 可选时仍恢复 v1，且默认选择确为 v2）、
  版本行退役拒绝、bool/浮点/字符串/0/-1/None 不强转、v1/schema<3 重建、choices 去重、注入评测器
  版本不符启动拒绝。
- `tests/backends/metal/test_track_mt1b_build_seam.py`（7）：真实尝试被组装并启动（成功或 typed
  能力缺口，两者都断言请求/驱动记录/离线/空 host/entry 字节）、staged 观测或能力缺口、伪造安装
  是 `IdentityDrift`、缺 staged 是 `backend_capability_unavailable`、adapter 服务表接线与 Metal
  拒绝、expected 主语绑定 manifest、构建模块不在控制机 import 前端。正例程序是 MT0-r5 字节
  （并断言“r3 只差旧 import 与 `@T.prim_func`”）。
- `tests/backends/metal/test_track_mt1b_seams.py`（16）：工厂构造/无默认 registry 读取/未知
  identity、validator、真实尝试（成功或 typed 能力缺口）与 Metal 拒绝、expected 独立性、preflight
  track 关卡（未知 id 不再落入单文件）、MSL 单文件仍通过且冻结版本进入内部投影、portfolio 投影
  拒绝 DSL 内容、confirmation 不触发 MSL probe、prompt 不给 MSL bundle 文本、
  `evaluate_candidate`/`evaluate_task` 边界、MT5 confirmation 派生拒绝、伪造 resume、v1 记录在可选
  v2 已登记时通过同一冻结版本完成 fresh confirmation 与 manifest activation（6 跳全链）。

兼容回归 16 个文件 **172 passed**（含 5 个 B 证据写出/处置节点）：`test_track_registry`、`test_track_factory`、
`test_a6_metal_config`、`test_search_v2_bundle_preflight`、`test_track_plane`、`test_track_mt1a`、
`test_shape_portfolio_candidate`、`test_evaluate_task`、`test_search_v2_bundle_track`、
`test_cross_hardware_boundary(_a2)`、`test_a6_retirement`、`test_failure_codes`、`test_structure`、
`test_shape_portfolio_runner`（投影签名新增冻结版本参数，其 spy 同步跟进；断言未弱化）。

前置包适用节点（直接运行其固定节点，不改包）：

| 包 | 节点 | r3 结果 |
| --- | --- | --- |
| MT1a | 86 regression + 23 new | 86/86、23/23 通过 |
| MT5 | 45 | 45/45 通过 |
| MT0-tilelang-probe | 75 | **73/75**；2 个 node 因必需的 living-document 更新而失败（见 §5） |

`verify.py` 另有 **10** 个文档级负变异：registry digest、合同 digest、闭包大小、归档 schema-1
字节、r5 工件集摘要、把 prompt 覆盖文件声明为 MT1b 改动、“格式正确但内容不同”的前端摘要、
围栏 entry digest、几何盲 build 身份、**把 B1 归档 GPU 证据声明为 MT1b 改动**——每一个都有对应
拒绝路径。另有 6 个**行为**探针（`build_identity_probes`）每次运行都真实复跑：同源不同几何必须
不同、同输入必须可复现、合法摘要漂移必须拒绝、自报/不完整工件必须拒绝、超时尝试必须拒绝、
entry 摘要必须由源码复算。`verify_external_b_evidence` 另核验 §6 的记录本身（当前字节、`median_us`
与每个“未变字段”）。

## 5. 历史锚点与 P2 漂移（未回填、未弱化）

MT1b 必须更新两份 living document（本计划与多 Track 对比），并改动被封闭验收器 pin 的文件；
封闭包本身**逐字节未改**，也没有为了让它们通过而改断言。r3 实测它们的停止点：

- MT0-tilelang-probe：`verify.py` 停在 `verify_history`（`docs/架构/多DSL-Track实施计划.md` 的
  living-document hash 未重算）；固定节点 73/75。它自己的 `history-anchor-drift.json` 记录的是同一
  类漂移，旧 `current_sha256` 未被改写。
- MT0-isolation-profile：`verify.py` 停在 `verify_results` 的 scoped source digest（同一份计划文档，
  既有漂移类）。
- MT1a：`recompute_surface()` 在为已登记的 `interface_only` TileLang Track 渲染题面时
  `require_available` 拒绝；包内 before/after 断言仍成立，固定节点 86+23 全通过。
- MT5：功能部分通过、45/45 节点通过；`--check` 停在 manifest——第一处是 `taic/cli.py`，另两个
  变动 pin 是 `taic/orchestrator.py` 与 `tcenv/backends/metal/track_factory.py`（r2 只写了 factory）。
- P0/P1/P3 在 MT1b 之前就因其 manifest pin 失败（涉及未被我改动的文件）；P2 的 manifest 现在额外
  pin 到 `taic/portfolio_confirmation.py`、`taic/portfolio_manifest.py` 与
  `tcenv/backends/metal/portfolio_confirmation.py`。

`history-anchor-drift.json` 的 living-document 摘要在 r3 重新计算（r2 写下的值已过期），并记录了
上述实测停止点与 manifest mismatch 清单；它的 `unchanged: true` 只指封闭包自己的文件。

**P2 归档证据漂移**（`p2-epoch-drift.json`）：registry 文件在静态 pack 闭包内，闭包 73→74 使 35 个
任务 pack digest 全部前进；P2 归档 confirmation 文档的 seed 由绑定旧 `task_pack_digest` 的
validation plan 派生，因此同样的 **26** 个 P2 replay 节点以 “confirmation seed is not derived from
this validation plan” 失败。r3 重新做了两次独立验证：

- 当前工作树：失败的 26 个 node id 与记录集合**逐 id 相同**；
- `git archive HEAD` 的 pristine 副本（旧 73 文件闭包、无 MT1b 文件）：同一 26 个节点 **26 passed**
  （159.17 s）。

修复方式仍是**在真实 Metal 安静窗口重录新 P2 epoch**（本轮未授权 GPU），不是改归档文档、不是
skip/xfail。P2–F 与 P3 性能/发布结论不受影响。

每个 MT1b 文件改动前后的边界记录在 `identity.json.changed_files`；AGENTS.md 的“不删除/不改写
R0–R6、G0、A1–A6、P0–P2 证据”未被触碰。

## 6. 归档 B1–B4 GPU 证据漂移的处置（独立 epoch，已解决）

复核窗口内发现 9 个 B1–B4 `evidence/gpu-*.json` 的 `median_us` 被一次真实 Metal
`test_task_migration_b*_gpu.py` 运行重写。原始事实仍完整保存在 `b-evidence-drift.json`，该文件
**未被修改**：它记录的是当时的漂移，不是当前归档字节。

本次处置不删除这次真实观察，也不让它继续污染封闭验收包：

1. **新观察独立保存**：9 个漂移后的 JSON 原样保存在
   `docs/架构/跨硬件算子生成分工-阶段验收/B-evidence-rerecord-2026-11-07/raw/`；
2. **封闭包恢复**：对应 9 个 `B*/evidence/gpu-*.json` 已恢复为 HEAD 原字节，候选/几何/dispatch/
   correctness 及原始 median_us 均与归档一致；
3. **写出边界修复**：新增 `tests/support/migration_gpu_evidence.py`。B1–B4 GPU 测试不再写
   `B*/evidence/`，默认每次写入独立临时 run 目录；需要形成新 epoch 时，必须通过
   `SHAPEFORGE_B_EVIDENCE_RUN_DIR` 显式指定封闭 evidence 之外的新目录，指定回旧归档会拒绝；
4. **可重放验证**：`b-evidence-resolution.json` 指向该独立处置包。verifier 逐项检查：原始
   incident record SHA 未变、raw 副本等于漂移观察、当前归档等于 HEAD、identity 字段不变、
   median_us 恢复原值。测试还固定 pending 处置和 incident record 被篡改两类负例。

因此这里既有完整的新观察，也有未被改写的历史证据；没有回滚删除、没有冒认 MT1b、没有 GPU/
模型重跑。后续若需新的 B 性能 epoch，必须在新的输出目录独立验收，不再自动覆盖历史包。

## 7. 全量 CPU 快速套件分类（r3 实测）

`uv run pytest -q -m 'not gpu and not llm and not slow'`：**61 failed / 4812 passed / 13 skipped /
1 warning**（525.58 s）。逐类核对（并用 pristine HEAD 副本交叉验证）：

| 类别 | 数量 | 说明 |
| --- | --- | --- |
| MT1b epoch 导致 | 28 | 26 个 P2 归档证据 replay（与 `p2-epoch-drift.json` 逐 id 相同）+ 2 个 MT0 living-document 漂移节点 |
| 环境既有失败 | 32 | 18 个 `AF_UNIX path too long`（selfcheck_bridge）+ 14 个 sandbox 拒绝（`ps` 被拒、timeout/权限类断言）；在 pristine HEAD 副本上逐文件复现同样的 14 + 18 |
| 并行计时 flake | 1 | `test_remote_job_engine.py::test_formal_priority_allows_aged_cross_run_selfcheck` 在并行负载下 8 s 队列期限超时（`queue_timeout`），单进程重跑通过（连同 `test_children.py`、`test_harness.py`、`test_run_detached.py` 一起跑：8 failed / 17 passed / 3.90 s，失败集与全量运行里的同名节点一致） |

第一版 epoch 的 4 个“已修正节点”（`test_a6_*`、`test_failure_codes`、`test_structure`、
`test_track_plane`/`test_search_v2_bundle_track`）继续保持通过；`test_shape_portfolio_runner`
（投影签名新增冻结版本参数）也全绿。没有删除测试、没有 skip/xfail、没有弱化任何断言。

## 8. 环境阻断：本机不能执行嵌套 Seatbelt

生产的 source-only 接缝已实现为“控制机观测 → 组闭集 `AttemptSpec` → 现有 hmz lower-level
Profile → 回读核验”，并且固定节点真的启动了一次尝试。但本执行环境本身已在 Seatbelt 内，
`sandbox-exec` 再嵌套会被拒绝：

```
sandbox-exec -p '(version 1)(allow default)' /bin/true
sandbox-exec: sandbox_apply: Operation not permitted   (exit 71)
```

r3 又独立跑了一次真实尝试（`attempt_base=/tmp/mt1b-attempt-review`），结果同样是 typed
`backend_capability_unavailable`，记录可核对：

- `hmz-profile-request.json`：`online=false`、`hosts=[]`、read 只含 `/usr`、`/bin`、staged 安装与
  仓库只读根，write 只含 attempt 根的 `build/{work,home,cache,config,data,tmp,pycache,artifacts}`
  与 `/dev/{null,zero,random,urandom,tty,ptmx,fd,dtracehelper}`；
- `input/frontend_entry.py` = `5c6bedc2…`（当前 entry 文本）、`input/kernel.py` = `3ebead19…`
  （= MT0-r5 候选）、`input/pinned.json` 带冻结安装摘要与候选摘要；
- `driver-report.json`：`fenced_returncode=71`、`lower_level_route=hmz.coganchor.fence.proxy.Proxy +
  hmz.coganchor.darwin.seatbelt.Profile`、`internal_fence_or_wrap_used=false`；
- `stderr.txt`：`sandbox-exec: sandbox_apply: Operation not permitted`；
- `limits.json` 存在但为空：shim 在应用 Seatbelt 之前就被拒，来不及写资源上限报告——这正是
  “the trusted shim did not return a valid resource-limit report” 这条 typed 结论的来源；
- 失败发生在**围栏建立**这一层，不是 codegen：没有 70（device required）/72（unsupported）/73
  （unexpected）终端码，没有被当成 “候选错误” 或 “能力已实现”。

这意味着：

- **已证明**：生产代码路径真的拼装并启动了围栏尝试；请求/策略/input 字节与冻结身份一致；工件内容
  核验、尝试终态门、身份派生、几何绑定、前端安装字节核对、Metal 拒绝全部按固定节点执行；归档 r5
  的真实字节仍可 CPU 重放；pinned frontend 上 `tvm.tirx.PrimFunc` 的类型门已被只解析核对证明可用。
- **未证明**：在这次执行环境里真实产出新的 source-only 工件集（`kernel.metal` 等）。MT0 的 r5
  归档来自 sandbox 之外的控制机；本轮没有、也不能在该环境下重跑。这是环境限制，不是用 helper 代替
  构建，也没有放宽任何断言。
- 在允许嵌套（或不在沙箱内）的宿主上，同一路径应产出 full source-only build；本包不把“未观察到”
  写成“已完成”。

## 9. 未做事项与下一授权点

- 未运行真实 Metal、GPU/NPU、Metal kernel、真实模型或付费搜索；未安装 Halide/新依赖；未改 hmz。
- TileLang Track 不是 `available`：没有真实 Metal build/dispatch、没有 correctness/timing/性能证据，
  不可被新 run 选择，也不可确认/发布。MT0 仍 blocked（第二条 DSL deferred），**整体多 DSL/Metal/
  生产 DSL 仍未完成**。
- B\* 历史证据处置已完成（§6：raw 保存 + 归档恢复 + 写出边界修复）；P2 新 epoch 安静窗口重录仍待授权。后续 B\* 新性能 epoch 仍需独立 GPU 授权。
- 下一授权点是 **MT2**：第一条 DSL Track 的真实 Metal 垂直切片（最小 elementwise/normalization、
  证明独立调度所有权、构建/执行/正确性/计时、首次编译只在 bounded worker、Host 工作进入 E2E）。
  MT2 需要单独批准；它不因 MT1b 完成而自动开始，也不允许把 source-only 结果升级为性能结论。
- 本轮最终 reviewer 结论只通过合法 `{done, notes}` JSON 返回：`done=true` 仅表示 MT1b 的合同/
  身份/adapter 与 CPU/协议验收完成，不表示 MT2、真实 Metal、生产 DSL 或整个计划完成。
