# MT1b：TileLang Track 合同、构建身份与 adapter 接缝（冻结文本）

本文件冻结 MT1b 交付的 **task-agnostic、`kernel_pipeline`** TileLang Track 合同。它不含任何
算子名、shape、算法、tiling 或性能结论，也不授权 MT2。可执行定义在
`tcenv/track/tilelang.py`；本文件只把**身份、边界与失败语义**固定成可引用的文本。

## 1. 正式身份

| 项 | 值 |
| --- | --- |
| `track_id` | `tilelang-pipeline` |
| `track_version` | `1` |
| `delivery` | `dsl_source_pipeline` |
| `execution_kind` | `kernel_pipeline` |
| `status` | `interface_only`（新 run 不可选；无 Metal build/dispatch，不可确认/发布；source-only codegen 可执行但 0 dispatch、`publication_allowed=false`） |
| renderer / validator / build adapter | `metal.track.tilelang-pipeline.{renderer,validator,build}` v1 |
| registry schema | `tracks.v2.schema.json`（`schema_version: 2`） |

Track 只描述**交付机制**：注册表与合同都没有 `op_name`/`shape`/`tile`/`grid`/`threads`/
`lowering_profile` 字段，也没有算法模板。候选自己拥有算法、融合、tiling、执行几何与 DSL 内部
调度。

## 2. 候选源合同（不伪装成 kernel.metal）

| 项 | 值 |
| --- | --- |
| 源路径 | `kernel.py` |
| media type | `text/x-python` |
| role | `dsl_source` |
| 允许的第二条目 | `host/execution_geometry.json` / role `schedule` / `application/json` |
| 禁止的 role | `device_kernel`、`host_tiling`、`launch` |
| 禁止的路径后缀 | `.metal`、`.metallib`、`.air` |
| 禁止的 media type | `text/x-metal-source`、`text/x-msl`、`application/x-metallib`、`application/x-metal-air` |

约束由两处共同执行：`CandidateArtifact.validate_for()`（registry 声明）与
`tcenv/track/tilelang.py::validate_candidate()`（经 `TileLangTrackValidator` 调用：冻结合同 +
条目集闭集）。**Python DSL 不会被改写成 `kernel.metal`**；MSL 文本是**构建产物**（见 §5），
永远不是候选条目。

轨道不要求 `msl-single`/`msl-bundle` 的合同：没有 `taic-launch` 块、没有 Host Tiling Plan、
没有 `CandidateLifecycle`（Host 准备/调度由 DSL 前端与候选自己的 host 代码拥有，MT2 才谈
计时边界）。

## 3. 执行几何（真实 kernel_pipeline，不伪造 graph）

- 候选可以（也应当）提交 `host/execution_geometry.json`，它是现有 kernel-pipeline 的闭集
  `ExecutionGeometry` 文档（`schema_version`/`source`/`stages[kernel,grid,threads,...]`）。
- 同一个 `candidate_geometry` 解析器被 MSL bundle 与 TileLang 共用；TileLang 不借用
  `msl-bundle` 的 HostTilingPlan，也不进入 `graph_executable`/`command_encoder`。
- 无几何时 Track 仍合法（MT1b 只校验交付形状）；是否要求几何属于 MT2 的切片设计，不能在本
  epoch 预先承诺。
- 本 Track 不引入 graph kind、不改 `NATIVE_EXECUTABLE_KINDS`（仍只有 `compute-pipeline`）、
  不改六 RPC / field number / HMAC / BlobChannel。

## 4. 前端/编译器身份

| 项 | 值 |
| --- | --- |
| `frontend_id` | `tilelang` |
| `frontend_version` | `0.1.15`（固定 wheel `dcffc9aa…`，上游 commit `a35f8ddf…`） |
| `entry` | `tilelang.engine.lower` |
| `options_sha256` | `10248461d099f3be090173904e662b5a9f7d8fa5860d6bb6f94315bd9046eeab` |
| source-only options | `target kind=metal, keys=[metal,gpu], target_host=c, enable_host_codegen=true, enable_device_compile=false, host_artifact_format=c-source` |

`options_sha256` 是**设备无关**构建 options 的 RFC8785 摘要；registry 只钉摘要，options 文档由
冻结合同模块拥有，两侧在每次构造 adapter 时交叉核对（不一致即 `IdentityDrift`）。

一次 build 必须观测到并记录前端安装身份，且必须**等于** MT0 冻结的实测值（不是“格式正确的
任意摘要”）：

| 项 | 冻结值 |
| --- | --- |
| `package_manifest_sha256` | `06f3b0adee7abce77af1111cb36e704b4b23c3cbe38d095dd62cce86509c11cc` |
| `dist_info_metadata_sha256` | `380d3486286aeb708dc4846cfdc4b438985c84199a160b21238cadf3e4e3fa19` |
| `dist_info_record_sha256` | `6ba83c2ed267154eedc9dec6d916e315c553a8e515744f0feb7ca05dfa388c34` |
| `package_files` / `package_bytes` | `3651` / `134219728` |
| manifest 规则 | `sha256(join("\n", f"{rel}  {bytes}  {sha256}"))`，排序、排除 `__pycache__`/`.pyc` |

缺任何一项是 `backend_capability_unavailable`（typed：本机没有已登记前端）。版本/入口/options
不同、或三个摘要/文件数/字节数任一不同，都是 `IdentityDrift`——**包括“64 位十六进制但内容不同”**。
frontend 升级即使候选源不变也改变 build 身份与缓存键。

## 5. source-only 与 Metal build 的边界

| | `source_only`（本 epoch 实现） | `metal_build`（MT2，未实现） |
| --- | --- | --- |
| options | `enable_device_compile=false` | `enable_device_compile=true` |
| 产物 | `frontend_identity.json`、`kernel.metal`（MSL 源）、`device_ir.json`、`kernel_params.json`、`host_module.txt` | 加上 device library / pipeline / 真实 dispatch |
| dispatch | 0（闭集校验） | 真实执行，需要 Metal 证据 |
| 状态 | 机制已由 MT0 r5 归档证据证明 | **不存在**；请求即 `unsupported_contract_capability` |

`source_only` 的工件集是闭集，摘要规则固定为
`sha256(RFC8785([{path,bytes,sha256}] 按声明顺序))`；缺件/多件/空字节都拒绝。

**生产接缝**（`tcenv/backends/metal/tilelang_build.py`，`TileLangSourceOnlyBuilder.build`）：

1. 控制机对 staged 安装做逐字节观测（不 import 前端），比对 §4 的冻结值；
2. 组一个闭集 `AttemptSpec`（harness-owned `frontend_entry.py` + 候选 `kernel.py` + pinned 身份
   + 声明工件），经现有 lower-level hmz Profile 运行（离线、空 host、attempt 内写、rlimit +
   外部看门狗）；
3. 回读工件并核验**内容**：`frontend_identity.json` 的 `verified`/`build_options`/
   `device_execution`/`candidate.source_sha256`、`kernel.metal` 是 MSL 且声明了候选几何里的
   stage 内核、`device_ir.json` 非空、`kernel_params.json` 非空、`host_module.txt` 是生成的宿主源；
4. 尝试终态必须是 `completed` + `returncode=0` + 未超时 + 离线 + 空 host；超时/非零/未完成
   一律 `codegen_failed`。

build 身份文档（`tcenv.track.tilelang.build-identity.2`）绑定：track 身份、build mode、
**完整候选 manifest 摘要**（`candidate_sha256`，覆盖 `kernel.py` **与**候选几何的路径/角色/
media/字节摘要）与 `dsl_source_sha256`、`candidate_entries` 行、前端安装身份、options 摘要、
设备无关环境（闭集：`python_version`/`python_implementation`/`platform`/`machine`）、尝试终态
（`status/returncode/timed_out/online/hosts/route/request_sha256/hmz_version/policy_sha256`）、
工件行与工件集摘要、`full_build`（由上述全部核验**推导**，不是调用方布尔）、
`device_dispatch_calls=0`、`device_library_created/pipeline_state_created/buffer_allocated=false`。

- `candidate_sha256` 就是候选身份（`candidate_sha256 == provenance.source_sha256` 规则不变）；
  同源不同几何 ⇒ 不同 manifest ⇒ 不同 build 身份/缓存键。
- 缓存键（expected **主语**）= 候选 manifest + DSL 源 + 前端/options + 环境的指纹，不含结果
  工件与尝试终态；`expected_build_subject()` 可在 build 之前独立重建，`full_build=false`。
- 只哈希 `kernel.py`、或让调用方传 `full_build=true`、或用任意字节直接造身份，都是本 epoch
  明确拒绝的旧缺陷；verifier 的 `build_identity_probes` 逐项复跑这些不变量。

## 6. 隔离与失败语义

- 复用 `tcenv.worker_isolation` 的 lower-level hmz Profile（`Proxy(()) + Profile`），
  **不建第二套 Seatbelt**，也不改 hmz。控制机不 import TileLang/torch/hmz。
- build 请求必须声明：`route`（固定为该 lower-level 路线）、`online=false`、`hosts=[]`、
  `external_reads=[]`、`external_writes=[]`、非空 attempt 写根、`internal_fence_or_wrap=false`。
  任一不满足即 typed 拒绝。
- 失败闭集：缺前端/hmz/解释器 → `backend_capability_unavailable`（typed 能力缺口，无无隔离
  fallback）；身份/options/安装字节漂移 → `IdentityDrift`；非声明 media/role/path → 合同错误；
  外部读写/online → 合同错误；codegen 超时/非零/缺工件/工件内容不符 →
  `codegen_failed`/合同错误（只可能是 partial，绝不升级为 full build）；请求 `metal_build` →
  `unsupported_contract_capability`。
- 真实围栏运行依赖宿主允许 `sandbox-exec`。在拒绝嵌套 Seatbelt 的执行环境里，生产路径同样以
  `backend_capability_unavailable` 结束，并保留 attempt 请求/驱动记录/限制报告；这不是“用
  helper 代替构建”，也不是放宽边界（见 acceptance §环境阻断）。

## 7. 选择与证据边界

- `interface_only` 的 DSL Track **不可被新 run 选择**：`TrackSpec.selectable` 要求 DSL Track
  声明 `metal_dispatch` 能力；`available()`/`require_available()`/`select_track()` 都会拒绝。
- 伪造的冻结选择（把 TileLang 写成 available）在 Track 边界即拒绝，不进入任何 build。
- 证据范围：CPU/协议（合同、身份、接缝、归档 r5 字节重放）。**不是** Metal 正确性、dispatch、
  计时、性能或可用性证据；`publication_allowed=false`。
