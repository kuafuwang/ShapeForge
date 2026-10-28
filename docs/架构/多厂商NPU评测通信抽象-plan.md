# 多厂商 NPU 评测通信抽象计划

状态：设计计划，2026-10-27。依据 `docs/tsm/` 的清微 TX81/TX82 资料与当前
`tcenv.remote.v1` 实现整理。**本计划不授权安装 SDK、访问真实设备、运行 GPU/NPU 或产生
性能/发布结论。** 当前先确定抽象与验收；任何真实站点接入都要另行授权并重新登记
site/device/evaluator epoch。

## 0. 核心结论

多厂商适配不是把每家厂商的 CLI、SDK 或 shell 再包一层 RPC，而是把通道拆成两层：

1. **统一控制面**：继续使用现有 gRPC + TLS + Protobuf 六 RPC、mTLS、每 RPC grant、
   持久 job/attempt、lease fence、Submit/GetJob/Watch/Cancel/Retry、结果签名和 outbox。
2. **可插拔设备执行面**：设备侧受信适配层由 worker 内 `VendorDriver` 与 worker 外
  `DeviceController` 组成，共同把能力发现、工件校验、加载/启动、同步/取消、清理证明、
  健康检查和计时证据映射到共同语义。厂商 SDK、固件、驱动、卡侧处理器和组网工具只出现在
  这一层及其部署 profile 中。
3. **独立 blob 数据面**（P0b，见 §4.5）：控制元数据继续走 gRPC/TLS/Protobuf，候选大工件改走
  受信、content-addressed 的分块通道。六 RPC 与 `EvaluationRequest` 不再承载整份大模型/大图
  字节；候选身份仍由 `CandidateDigest` + request digest 绑定，blob 只是同一身份的字节来源。

外部 API 不新增 `TsmSubmit`、`VendorBSubmit` 之类的厂商专用入口；agent 也不能指定任意
vendor shell、路径、MPI 命令或裸设备地址。**通信统一的是请求、能力、工件、结果、错误与
计时证据，不是假装所有 NPU 都有相同的 kernel ABI。**

构建也必须是统一的一层：`ArtifactBuilder` 在 control_host 或 evaluator 内把 source 变成
工件，产出相同的 `ArtifactBundle`、`BuildProvenance` 和 `BuildEvidence`。local Metal 与
远程 NPU 不再各写一套“先编译再执行”的隐式流程；传输与设备执行分别交给
ArtifactTransport 和 VendorDriver。

## 1. 清微资料揭示的实际形态

| 形态 | 资料依据 | 对抽象的要求 |
| --- | --- | --- |
| host runtime + 卡侧处理器 | `tx_runtime.cn_v1.4.0.pdf` §4.3：kernel launch 经 Host → AP → NPU Core Scheduler；`tsm-shell使用指南`：AP/SCP/RMCU、KIQ/共享内存描述符 | 设备操作必须留在受信设备侧；不能把卡侧 shell 当作通用评测协议 |
| 多种执行物与启动方式 | runtime §4.3：`txModuleLoad` 接受 Triton Kernel ELF/专有 bin；SIMT、Cluster、`txLaunchModel`（BPM 表）、`txLoadGraph` | 单 `source bytes` 不足以覆盖所有形态；需要版本化工件清单与执行规格 |
| stream/event/IPC | runtime §4.7–4.9：stream 同步、event elapsed time、IPC memory handle | 同步与计时是独立能力，不能把 launch 成功当执行完成；`txDeviceSynchronize` 在文档中仍标为未来实现，当前以 stream/event 同步为准；共享内存不能变成隐式跨主机通道 |
| 多卡、die/tile、rank | runtime §4.1/4.4：device/tile 信息、物理/逻辑 rank；SMI FAQ：user/mesh/soft/slot ID 含义不同 | 逻辑设备组与物理拓扑必须分开；Linux 枚举 soft id 不能直接当稳定身份 |
| 单机 mesh/torus 与多机 C2C/MPI | `TX81平台软件安装部署指南` §4.1–4.5：discovery、`node_connections.json`、fabric_manager、MPI 网卡绑定 | 组网是站点 provisioning 与 lease 配置，不是 agent 请求字段；跨节点 fabric 不穿过 control_host |
| 容器设备映射 | 安装指南 §3.2：`host-hal`、`--network=host`、按需映射 `/dev/accel_drv_mgr` 与 `/dev/accel/dev-*`，非特权为推荐方式 | 可复用容器生命周期，但设备映射、host network 与挂载必须是受信部署 profile，不能由候选配置 |
| SMI 健康/拓扑/reset | `SMI工具使用指南`：TxIPC 获取健康、利用率、拓扑；`--reset` 对所有设备且需先结束占用；FAQ 要求物理机 root 运行 discovery_host | 健康检查与 reset 是独立运维能力；reset 必须限权并评估整机/整组影响，厂商只支持全设备 reset 时不能假装按组隔离，也不能作为每个 job 的默认动作 |
| Profiler 独立通道（HLD §3 阶段表；LLD 有部分实现/运行样例） | `tx82_profiler_hld` §1–3、`lld-tx82-profiler-ch01`、`raPTI_LLD` §1–4：Host/API span、Intercept Queue、per-die 与 launch 汇总、PMC buffer、correlation id | Host API EXIT 不是设备完成；计时证据必须区分 host wall、device event、profiler dispatch，并保留关联与完整性信息 |

Profiler 的阶段表只在 `tx82_profiler_hld` §3（HLD Phase 1 Host/Device 时间、HLD Phase 2
Intercept、HLD Phase 3 双 buffer、HLD Phase 4 memory 生命周期）；`lld-tx82-profiler-ch01`
记录 LLD 的 Phase 1 部分实现（换 Create/Destroy、intercept 建队、Register WI），`raPTI_LLD`
附录 A 有 Host/API span 与 per-die/launch 设备时间的运行日志样例。这些说明厂商侧有部分实现，
但不是本项目在目标设备/版本上的验收；只有对应能力交付、核验并进入 `TimingEvidence` 后
才能使用。

### 1.1 清微 Docker 的真正边界

清微 Docker 是**客户软件环境、依赖与操作界面的隔离**，不是硬件隔离。每个客户可以有自己的
Linux 镜像、runtime、配置和文件视图，并通过设备映射、共享文件、IPC/共享内存等机制与
host 交互；最终操作驱动和物理 NPU 的仍是 host 上的运行时与设备。客户既可在 host 上操作，
也可在自己的 Docker 内操作，但共享同一个 host kernel 与物理设备。

因此“每客户一个 Docker”不等于每客户一张 NPU，也不构成独立 lease 或强安全边界。客户环境
可以隔离，设备组、discovery、健康检查、reset 和资源清理仍必须按 host/设备组统一管理。

另外两条必须正面处理，而不能照抄进生产：

- 多机部署资料出现“免密 root 登录其他机器”和物理机 root discovery 的要求。它们只能进入
  受信站点 provisioning/运维评审，**不得成为 agent 可调用的能力，也不得把 root SSH 私钥
  放进 frontend/workspace**。需要由设备方提供最小权限的组网方式，或由 operator 预配并核验。
- 资料中部分 serial/UUID、稳定拓扑信息仍标为待实现。没有稳定硬件身份前，不能用 soft id
  或容器内枚举顺序冒充发布级设备身份。

## 2. 现有通道保留什么，缺什么

### 2.1 直接保留并冻结语义

- 六 RPC、`JobStore`/`JobEngine`、单写者 SQLite、attempt/fence、quota、终态不可变、
  显式 Retry、Watch/GetJob 对账。
- mTLS、每 RPC grant、API/evidence contract、request digest、attempt receipt、HMAC、
  host site/epoch 发布白名单。
- `AttemptSupervisor` 的 create/start/wait/kill/delete/status、cleanup/absence/residue
  证明，以及超时、取消、恢复不自动重跑。
- `LocalEvaluator`、Unix selfcheck bridge、私有 worker 帧与现有 Metal 默认路径。

### 2.2 需要抽象而不是硬编码的部分

- `TargetCapability` 目前只有 target/backend/modes/features，无法表达设备组、执行物、
  启动方式、计时来源与设备组可用/隔离状态。
- `EvaluationRequest.source` 只描述单份待编译源码，不能无损表达 ELF/bin、graph、model、
  参数表及辅助文件。P0b 先补独立 blob 通道与单个 blob 候选引用（§4.5）；多件 bundle 与
  graph/model 语义仍属 P1。
- `Backend` 负责源码/语言/launch 规则，`AttemptSupervisor` 负责 worker 生命周期；在
  Backend/worker 与设备之间还缺一个明确的**厂商设备驱动层**，不能直接往 gRPC server 或
  Docker supervisor 里塞 SDK。
- 编译目前隐含在 evaluator worker 内，无法表达 control_host 本地编译、evaluator 端编译和
  预编译工件三种时序。需要统一的 `ArtifactBuilder`，把构建产物与构建证据从传输、执行中
  拆出来（§4.2.1）。
- `EvalReport` 的标量耗时不足以解释多 rank、profiler 缓冲、时钟域与证据缺失；厂商错误也
  不能只靠字符串归类。

## 3. 目标架构

```text
control_host（编译服务器）
  LLM / pi / hmz / taic / git / 编译工具链
  SourceBundle → ArtifactBuilder → ArtifactBundle + BuildProvenance
  taic / GrpcEvaluatorFactory / outbox / publication gate
                 │  同一 tcenv.remote.v1（或兼容扩展）
                 ▼
设备站点 single-active frontend + 受信设备控制区
  auth → JobEngine → attempt/fence → AttemptSupervisor
                 │ 私有 worker 协议          │ typed + fence 绑定
                 ▼                          ▼
受信 worker                         DeviceController / Reconciler（worker 外）
  request/artifact 校验                probe / health / cancel / cleanup proof
       → VendorDriver（设备语义）                │
           ├─ ArtifactCodec / LaunchAdapter      │
           └─ TimingProvider / EvidenceMapper    │
                 └──────────────┬─────────────────┘
                                ▼
                  厂商 runtime / driver / device group

独立 blob 数据面（P0b，与控制面并列、不共享消息）
  agent ──gRPC BlobChannel（grant 绑定、分块、content-addressed）──▶ BlobStore（单写者 state 目录）
                                                                        ▲
                                          受信 worker 按 content id 分块读取 ┘
```

- control_host 同时是编译服务器：LLM、git、源码管理、编译与打包都留在富环境一侧；NPU
  服务器只保留受信执行所需的最小 runtime/driver/evaluator，不要求 LLM、git 凭据或完整
  编译环境。control_host 不直接接触 `/dev/*`、卡侧 AP 或 fabric。

构建、传输和执行严格分层：

```text
SourceBundle
  → ArtifactBuilder（control_host 或 evaluator）
  → ArtifactBundle + BuildProvenance + BuildEvidence
  → ArtifactTransport（本地引用 / inline / BlobChannel）
  → VendorDriver（校验、加载、执行、计时）
```
- 每个逻辑设备组仍对应一个 single-active frontend、一份持久状态和一个 lease owner；
  多设备/多厂商使用独立 frontend，不在 v1 暗做全局调度或 HA。
- `Backend` 继续负责源码、语言、参考实现、任务指纹和 launch 校验；`VendorDriver` 负责
  infra；`AttemptSupervisor` 只负责 worker 生命周期，三者不互相吞并。

建议落点：

- `tcenv/backends/<backend_id>/`：源码/契约/参考/评测语义。
- `tcenv/remote/drivers/<vendor_id>/`：厂商 infra adapter。
- `backends/<id>/`、`targets/<id>/`：公开元数据、skills、芯片/设备 profile。
- `deploy/vendors/<vendor_id>/`：受信镜像、依赖清单、provision/health 脚本与部署 profile。
- `tcenv/remote/blob_channel.py`、`tcenv/remote/blob_service.py`：P0b 的 blob 数据面与独立
  gRPC 服务（§4.5）；不进六 RPC。
- `tests/support/vendor_sim/`：至少两种不同形态的 fake vendor 与 golden；不进入生产组合根。
- `tests/support/blob_sim/`：blob 通道的故障注入 IO、崩溃/重建与配额场景；只在测试组合根注入。
- `tests/support/tsm_sim/`：P2a 的清微 runtime/SMI/profiler 模拟器与故障注入（见 §8 P2a），
  只在测试 composition root 注入。

## 4. 厂商中立模型

### 4.1 Site、DeviceGroup 与 VendorProfile

`device_id` 继续是对外不透明标识；其下由受信配置定义一个逻辑 `DeviceGroup`：

- `site_id/device_id/vendor_id/backend_id/target_id`；
- 逻辑设备/卡数、每卡 die/tile 数、逻辑 rank 形状与拓扑摘要；这些字段按厂商形态可为
  不适用或空，不能强制填 1 来假装存在 rank/die；
- driver/firmware/runtime/SDK/profiler/OS kernel/adapter 版本；
- 支持的执行物、launch 方式、计时来源、reset/health 能力和资源上限；
- worker 形态：容器设备映射、受信 host runtime，或厂商专用受信 RPC（若未来确有此形态）。

物理 BDF、user/mesh/soft/slot id、die/tile/rank 坐标由站点 profile 和 probe 解析，agent
不直接指定。profile 同时保存从逻辑 ID 到物理身份的映射及其证据；soft id 只能作当次
定位，不能单独进入发布身份。

以下 DeviceGroup 绑定仅适用于新的 NPU contract；现有 Metal/source v1 不增加这些字段。
稳定的 `group_id`、`profile_digest`、`topology_digest` 与资源代际（设备组分配、reset 或
拓扑变化时递增，与 grant generation 无关）必须进入 capability、request digest、
evaluator epoch/provenance/receipt 和 lease owner；这些 digest 与 `artifact_set_digest`
一样使用 schema 固定的 canonical document + SHA-256，不对 Protobuf 序列化字节取摘要；
Submit/Retry 逐项复核；拓扑变化先刷新 capability，影响测量的变化必须新 epoch，不能拿旧
capability 在另一组设备上执行或重试。同一物理设备不能同时分给两个 frontend；若必须跨
frontend 共享，先另做站点级 lease，否则 Cancel/reset 可能误伤其他 job。

### 4.2 CapabilityDescriptor

在现有 GetCapabilities 上扩展结构化能力，而不是让客户端猜测：

- `artifact_kinds`：支持的 bundle schema、artifact role、最大单件/总字节、是否支持外部 blob；
- `launch_kinds`：`KERNEL`、`GRAPH`、`MODEL` 等公共执行形态；SIMT/Cluster/BPM 等厂商
  模式放在受信 profile 的子类型，不进入公共枚举；
- `device_groups`：逻辑设备、拓扑摘要、rank/die 约束（若适用）与可用状态；
- `timing_sources`：`HOST_WALL`、`DEVICE_EVENT`、`PROFILER_DISPATCH`；只有
  `DEVICE_EVENT`、`PROFILER_DISPATCH` 这类设备侧来源才能支撑后续性能发布，
  `HOST_WALL` 单独只能作诊断；
- `versions`：完整版本集合与 adapter schema。

health 由 `DeviceController` 负责，reset 与 worker 形态属于服务端受信 profile/运维，不作为
客户端可请求能力；capability 只报告设备组可用性、探测结果和限额。公共字段必须是明确
Protobuf message，允许另带 `schema_id + digest` 的厂商说明 blob 供诊断，但**请求执行不得
依赖任意 vendor key-value**；vendor-specific 配置固定在服务端受信 profile。客户端缺少
必需 feature 时不 Submit，服务端在 create/start 前同样拒绝。

### 4.2.1 ArtifactBuilder：统一 local/remote 构建

`ArtifactBuilder` 负责把 source 变成可执行工件，是 Backend/任务语义层的一部分，不属于
BlobChannel 或 VendorDriver。建议边界：

```text
ArtifactBuilder:
  probe(build_profile) -> BuildCapabilities
  build(source_bundle, task_contract, build_profile, deadline) -> BuildOutcome
```

`BuildOutcome` 统一包含：

- `ArtifactBundle`：可执行 module/graph/model 及 manifest/digest；
- `BuildProvenance`：builder id、构建主机/profile、工具链与 SDK 版本、source digest、
  编译 profile、artifact_set_digest；
- `BuildEvidence`：compile/pipeline 状态、日志、耗时和错误分类。

工件载荷必须同时支持两种表示，不能让抽象只适用于文件：

- **bytes**：ELF/bin/graph/model 等可序列化字节，可 inline 或经 BlobChannel 传输；
- **backend-native executable handle**：例如 Metal runtime 的 pipeline handle。它只在创建它的
  后端/进程语义内有效，manifest 记录 handle 类型、source/build 身份与后端元数据，不能拿
  空字节或伪造 digest 冒充 file artifact，也不能未经专门合约就交给 BlobChannel/另一个后端。

因此 ArtifactTransport 还必须区分“可传输字节”和“仅限本地的后端原生句柄”；Metal 不需要
为了统一而序列化 pipeline，NPU 的 bytes/blob 路径也不需要理解 Metal handle。

placement 只改变构建位置，不改变产物语义：

| 模式 | 构建位置与时序 | 后续路径 |
| --- | --- | --- |
| 预编译 | control_host 先构建，成功后 Submit；失败则不 Submit | ArtifactTransport → 远程 NPU，仅加载/执行 |
| 构建并评测 | source 提交给 evaluator，在同一 attempt 内构建并执行 | evaluator worker 内构建后进入 VendorDriver |
| 本地构建执行 | control_host 本地构建并执行，如现有 Metal | 无远程传输，保持现有 local 语义 |

`BuildProvenance` 不改变 `artifact_set_digest` 只描述工件 manifest/字节的规则；它作为独立
字段进入 canonical request、outbox/resume 和签名 BuildEvidence。若构建环境变化会影响测量
可比性，host 还必须按新测量基线处理，不能只换 artifact digest 就继续比较。

builder 只能使用登记的 build_profile/toolchain，不接受任意 shell、路径或构建命令。接收方
仍要复核 artifact digest、厂商签名/ABI 与任务契约，BuildProvenance 不是免验证证明。

### 4.3 ArtifactBundle 与 ExecutionSpec

新增版本化 `ArtifactBundle`，不把压缩包或 JSON 塞进旧 `source`：

- `schema_id`、manifest、root digest（即 `artifact_set_digest`）；
- 每项只含 bundle 相对 POSIX 路径、`role`、`media_type`、实际字节 sha256、大小；
- role 使用公共枚举：`SOURCE`、`EXECUTABLE_MODULE`、`GRAPH_DEFINITION`、
  `MODEL_DEFINITION`、`PARAMETER_TABLE`、`AUXILIARY`；role 是否必需由 `artifact_kinds`
  声明，未声明的 role 不要求每个厂商都提供；
- 拒绝绝对路径、`..`、符号链接、设备文件、重复路径、超总限额和 digest 不符；执行时按
  实际字节复核，不做换行/Unicode 规范化。

`ExecutionSpec` 描述任务族（kernel source/module、graph、model）、launch kind、逻辑
device_group、必要的资源数量、同步策略与计时策略；cache policy 继续作为 request 级字段，
不进入工件身份。物理设备分配由服务端完成；agent 只能申请受策略约束的逻辑资源，不能指定
BDF、rank 坐标或 fabric 命令。

摘要与持久化规则：

- 现有 Metal 单 source 请求的 `candidate_sha256` 与算法保持不变；
- `artifact_set_digest = SHA256(RFC8785(canonical_manifest))` 只覆盖 bundle 的 `schema_id`、
  schema 版本、相对路径、role、media_type、size 与每项 sha256，manifest 按路径稳定排序；
  它是候选工件集的稳定身份，不含设备组、placement、ExecutionSpec 或 cache policy；
- `ExecutionSpec`、task fingerprint、device group/profile/topology、placement/timing 与
  cache policy 进入新 contract 的 canonical request document；`CandidateDigest(algorithm,
  value)` 对 source 取 sha256(source bytes)，对 bundle 取 `artifact_set_digest`，再由
  request digest 同时绑定 CandidateDigest 与上述执行/放置字段；
- **bundle 必须启用新的 API/evidence contract**，不是向旧 v1 偷加可选字段。带 bundle 的
  新 contract 版本用 oneof 明确 source 与 bundle 恰好一个，并给 request/receipt/result
  document 新的 `document_type` 或 `document_version`；旧 `tcenv.remote.v1.*` 与
  `schema_version=1` 字段集不变。`CandidateDigest` 贯穿 canonical request、request digest、
  新的 receipt（使用 `candidate_digest`，不双写 `candidate_sha256`）、outbox、结果签名和
  resume；不能留空、各算一份或在旧字段中暗换含义；
- P0 可以先只发布 source 分支的新 contract；bundle 在 P1 以同一新 contract 的后续版本
  加入，不能把 bundle 塞回旧 v1；
- 实际 bundle 字节与 manifest 按 source 的同一 job 生命周期持久化，支持恢复和幂等；终态后
  按既有清理策略删除。旧 contract 只接受现有 Metal/source 及旧验证语义；所有 NPU 请求
  （含 simulated/fake 与单 source）一律使用新 contract，混合、缺失或 digest 不符均拒绝。

首版只允许有界 inline bundle。大模型/大图需要独立的 content-addressed blob 通道（§4.5，
P0b 已先落地单 blob 候选引用）；不得以宿主路径、共享目录或 `source` 伪装大工件传输。
ArtifactBundle 进入 blob 形态仍属 P1，不能借 P0b 提前宣称。

### 4.4 Result、TimingEvidence 与错误

现有 `EvalReport.status/stage`、`JobFailure.domain/code/retryable` 保持。证据增加共同
`TimingEvidence`：

- `source`、`clock_domain`、`calibration_digest`、同步/flush 是否完成；
- 汇总 start/end/elapsed，以及声明拓扑所要求的 device/rank/die span 与 completeness；
  无 rank/die 的单设备或服务式厂商不因这些字段非适用而判缺失；
- profiler correlation/launch id（如支持）和 profiler 开销/缓冲丢失信息；
- 声明必需的 device/rank/span 缺失、时钟域未知或缓冲不完整时，结果只能诊断，不能发布。

在新的 NPU contract 中，`artifact_set_digest`（若有 bundle）、device group/profile/topology
digest、`TimingEvidence` 与厂商诊断必须纳入同一个签名 result envelope/evidence contract，
不能只签 `EvalReport` 的标量耗时。旧 Metal/source 的结果 envelope 与验证规则不变。

多设备结果不能用平均值冒充完成时间：同步语义确定后按定义汇总，并保留每个 rank 的证据。
Host launch 返回耗时单独记录，不能写成 device elapsed。

厂商错误码与诊断原文（脱敏、有界）保留在 evidence；共同分类仍只有候选 compile/incorrect/
crash/timeout、基础设施、协议/策略。C2C down、discovery 失败、驱动/固件不兼容、设备 reset、
license/配额等必须映射为正确的基础设施类别，不能变成候选 timeout 或 ok。

### 4.5 独立 content-addressed blob 通道（P0b）

目标：大模型/大图不进入单条 `EvaluationRequest`/`OpaqueBlob`/Protobuf 大消息。控制元数据仍是
六 RPC + TLS + Protobuf，大二进制走独立的受信分块通道：`BlobChannel` gRPC 服务，与 Evaluator
控制面是不同 service、不同 proto package、不同消息，六 RPC 名称、field number、digest 与 HMAC
算法一个字节都不变。

**边界（本轮只做小文件模拟）**

- 只证明协议、分块、完整性、恢复、授权与清理；不证明大文件吞吐、容量、跨站点性能或真实
  NPU 可用性。真实对象存储/模型下载/设备访问一律不做。
- 不实现 P1 的 graph-only fake、ArtifactBundle 或复杂 ExecutionSpec；blob 引用只挂在
  P0 已有的 `artifact_kind=source` + `KERNEL`/`SIMT` 形状上，多件/多角色 bundle 留给 P1。
- 不改旧 Metal/source contract：`tcenv.remote.v1.*` 的请求、receipt、outbox 与签名语义不动；
  P0 的 `npu-source` contract 也不动，blob 形态是它的**后续 contract 版本**。

**BlobDescriptor（稳定身份）**

| 字段 | 规则 |
| --- | --- |
| `content_id` | `blob-sha256-v1:<64 hex>`；算法固定为内容 sha256 的算法标签化表示，同字节同 id |
| `sha256` | 实际内容 SHA-256，必须与 `content_id` 的 hex 逐字节相同，否则 fail closed |
| `size_bytes` | 声明大小；提交时按实际落盘字节复核 |
| `media_type` | 必须等于请求 `execution.source_media_type`；不做内容嗅探或转码 |
| `chunk_size` | 声明分块大小（1..上限），upload/read 的边界都由它决定 |

原始二进制不透明：不做 Base64、不做 UTF-8 校验、不逐字段序列化模型。descriptor 是 canonical
request 的一部分，因此 `CandidateDigest`/request digest 绑定的是**内容身份**，不是校验和备注。

**状态机与持久化**

```text
accepted ──begin──▶ uploading ──commit(复核 size+sha256)──▶ ready
                        │                                   │
                        ├──abort/fail/超限/复核不符──▶ failed
                        └──cleanup 失败──────────────▶ quarantine（不假装已清理）
```

- 每个 blob 一个 manifest（0600，原子写 + fsync + rename）记录 descriptor、状态、已验证 chunk
  集合与 `received_bytes`、scope 绑定、TTL、lease 列表、cleanup 结果与 quarantine 原因；
  chunk 各自成文件，写入用临时文件 + rename，崩溃不会产生"半块已确认"。
- `ready` 是唯一可被 Submit 引用的状态；`uploading`/`failed` 一律拒绝，`quarantine` 永不自动
  复用。重复 `commit` 在 descriptor 与字节一致时幂等返回 `ready`；`ready` 之后任何 chunk 写入
  都被拒绝。
- 崩溃/重启后按 `content_id` + manifest 里已验证的 offset 续传；未验证的临时 chunk 被丢弃，
  续传从第一个未验证 offset 继续。

**分块与内存边界**

- `WriteChunk(content_id, offset, data)` 只接受 `offset == received_bytes` 的顺序块（乱序、越界、
  超 `chunk_size`、超过声明 `size_bytes` 都拒绝）。`offset < received_bytes` 的重放按**声明 chunk
  边界**定位第 `offset / chunk_size` 块（不是只认最后一块）：该块必须已在已验证集合内，长度必须
  等于声明长度（最后一块可短），字节完全相同时幂等 ack、不推进进度；非边界、超出已验证范围、
  长度不符或字节不同一律冲突。
- `ReadChunk(content_id, offset, length)` 只返回 `length <= chunk_size` 的区间，读取路径按实际
  区间读文件，不把整份内容读入内存；commit 复核 digest 用固定大小缓冲流式哈希。
- 背压是协议的一部分：每个 chunk 一个 RPC ack，客户端最多一个 chunk 在途；服务端不缓存后续
  块。慢消费者不会让服务端堆内存。

**授权、配额与生命周期**

- 每个 upload/read/commit/lease 都必须携带该 RPC 的 session grant；服务端把 token 解析成
  `BlobScope`（site/device/target/subject/run/session），与 blob 记录的 scope 逐字段比较。
  未认证、过期、已撤销的 grant 在触碰存储之前就被拒绝；跨 scope 读取（越权）即使知道
  `content_id` 也拒绝。
- agent 只提供 descriptor 与 offset/长度，不提供本地路径、共享目录、URL 或任意 vendor 参数；
  服务端把 blob 放在自己 state 目录下，worker 由引擎注入受信 blob 根，不接受 agent 输入。
- 限额：单 blob 声明大小、`chunk_size` 上下界、chunk 数量、每 scope 并发上传数、每 scope 与
  全局字节配额都显式配置。字节配额按**声明大小预留**：`accepted`/`uploading` 的 blob 从 `begin`
  起就占住整份 `size_bytes`（不是只算已落盘 chunk），所以两个未完成上传不能各自通过检查后再一起
  写超；`ready` 按声明（已验证）字节计，`failed`/`quarantine` 按仍占盘的残留字节计。`write_chunk`
  在落盘前按同一账本再复查一次（配额被调小或账本不一致时 fail closed），超限返回
  `resource_exhausted`，不落半份数据。
- 生命周期：upload/commit/abort 一旦记录 attempt/fence 就绑定它，旧 attempt 既不能完成也不能 abort 新上传；
  lease 以 `request_id` 为键（Submit 早于 attempt 分配），只由引擎在终态释放，agent 没有任何清理入口。
  `Submit` 成功后 blob 与 `request_id` 建立 lease；job/attempt 进入终态后释放 lease，
  最后一个 lease 释放且策略允许时删除字节；上传未完成就终态/断链的临时 blob 由 sweep 按 TTL
  清理。清理失败必须把 blob 标成 `quarantine` 并保留证据，绝不返回"已清理"。

**与 NPU contract 的集成（新 contract，旧 contract 不动）**

- 新 contract：`tcenv.remote.npu-blob.evaluator-api.1` / `tcenv.remote.npu-blob.evidence.1`，
  document version 3，仍复用六 RPC、job/attempt/fence、device group 与 `DEVICE_EVENT` 约束。
- 互斥规则：`npu-source` 只接受 inline source；`npu-blob` 只接受 blob 引用且**禁止**携带任何
  source 字节。两种形态由 contract 身份选择，不由请求形状猜测；混用、缺 blob 引用、带 source
  字节都 fail closed。
- `CandidateDigest`：blob 形态用 `algorithm=sha256-blob-v1`，`value` 等于 descriptor 的 `sha256`
  并等于实际内容哈希；inline 形态保持 `sha256`。request digest 同时绑定 `candidate_digest`、
  `blob_ref`（content id/size/media type/chunk size）与 `execution`。
- 链：canonical request → request digest → receipt（`candidate_digest` + device group）→ outbox 行
  （`schema_version=2`）→ HMAC result signature → resume（终态无字节时按持久化字段重建）。任一
  环节改写 descriptor/digest 都被拒。
- 未完成 blob、非本 scope 的 blob、digest/size 与 descriptor 不符的 blob 不许 Submit；attempt
  dispatch 时 worker 再按 content id 分块读取并复核整份 digest，读失败按基础设施失败处理。

**小文件模拟验收矩阵（本轮）**

| 维度 | 覆盖 |
| --- | --- |
| 分块 | `chunk_size` 极小强制多块；空文件、1 字节、正好一个 chunk、chunk+1、含 NUL/非 UTF-8 随机二进制 |
| 字节一致 | 上传后逐 chunk 读回与原文逐字节相同；无整文件内存缓冲（记录单次读写峰值） |
| 续传/幂等 | 中断后按已验证 offset 续传；重复 chunk、重复 commit、重复 begin 幂等；同 id 不同字节/大小/digest 冲突 |
| 拒绝面 | 乱序、越界 offset、超 chunk/超大小/超配额、未 ready 使用、重复 ready 后写入、越权/过期/撤销 grant |
| 故障 | 存储写入/清理失败、进程崩溃后重启、并发相同 content id、慢消费者背压、cleanup 失败必须 quarantine |
| 契约链 | descriptor/digest 在 request digest、receipt、outbox、签名、resume 全链一致；篡改任一环节被拒 |
| 回归 | 旧 Metal/source 的 candidate_sha256/receipt/outbox 验证不变；六 RPC、field number、HMAC 不变 |

**变异可证伪性**：跳过 commit 的 digest 复核、允许 `uploading` 被 Submit 使用、把整份文件读入
内存、忽略 cleanup 失败、旧 fence/旧 attempt 提交或清理新 blob、跨 scope 读取放行——每一条都
必须让对应测试失败。

**模拟无法证明（留待真实环境）**：真实对象存储/共享存储的吞吐与容量、跨站点带宽与尾延迟、
大模型（GB 级）的并发与断点续传、服务端进程崩溃时的存储原子性、真实 NPU 上大工件加载路径、
multipart 校验与运维配额。本轮结果一律 `simulated=true`、`publication_allowed=false`，不进入
confirmed/catalog/brief，不产生性能或"大模型已支持"的结论。

## 5. VendorDriver 与生命周期

建议接口（是设计边界，不要求机械照搬函数名）：

```text
# worker 内，负责协作式执行
VendorDriver:
  prepare(spec, bundle, attempt_context) -> PreparedAttempt
  execute(prepared, progress, deadline) -> VendorOutcome
  close(prepared) -> CooperativeCleanup

# worker 外受信 helper（不是 gRPC frontend 主进程），负责探测与故障对账
DeviceController:
  probe(profile) -> VendorFacts
  capabilities(facts) -> CapabilityDescriptor
  cancel(attempt_id, lease_fence) -> None
  reconcile(attempt_id, lease_fence) -> CleanupProof
  health_check(profile) -> HealthProof
```

约束：

- VendorDriver 不负责定义编译流程：预编译工件只做校验/加载；构建并评测模式才在 worker
  内先调用 ArtifactBuilder，再进入 prepare。两种模式必须产出同样的 ArtifactBundle、
  BuildProvenance 和 BuildEvidence，不得在 driver 内另藏一套 vendor 专用编译协议。
- frontend 在 worker create/start 前只做纯 schema/policy/digest/capability 校验，不加载
  厂商 SDK；worker 启动后、执行前再由 driver `prepare` 校验设备与工件并准备资源，这一阶段
  同样计入 attempt deadline，且必须可 Cancel、可被后续 reconcile 清理；
- `VendorDriver` 在 worker 内不是安全边界：它只做协作式生命周期，不能阻止被攻破的 worker
  继续使用设备；真正的取消/隔离依赖 worker 外 `DeviceController` 与厂商按 attempt 的权限。
  `DeviceController` 是独立于 frontend 主进程和 attempt 容器的受信进程/服务，按设备组
  single-active 部署；frontend 重启不能把它当作已清理，controller 不可用时 job 只能
  quarantine，不能回退到 worker 内清理；这些调用必须有界并可重试，超时按基础设施失败处理；
- `prepare` 在创建模块/graph/stream/IPC 等设备资源前，必须先把 attempt_id/lease_fence 写进
  worker 外可读的持久资源台账（或由 controller 代建句柄）；否则 worker 在 prepare 中被杀后，
  controller 只能 quarantine，不能声称 reconcile 成功。容器内临时文件随容器删除可证明消失；
  host-visible 的 attempt 目录/句柄必须由 controller 作用域持有或登记，不能假设删容器会清掉；
- `execute` 只做 typed load/launch/synchronize/timing，不接受 shell、任意路径或 vendor 自定义
  命令字符串；路径只能由 driver 从已校验工件物化；deadline 覆盖 setup 与执行；
- 清理分两层：worker 存活时 driver 做协作式 close；timeout/Cancel/worker 被杀后由 worker 外
  `DeviceController.reconcile` 按 attempt/fence 核对模块/graph/stream/IPC/profiler/临时资源。
  删除容器只证明 worker 消失，不证明设备资源已释放；厂商无法提供这种外部控制或证明时，
  只能 quarantine + 人工恢复，不能验收为自动取消/超时清理成功；
- `health_check` 是轻量 known-good 探针，与完整候选评测分开，并尊重设备组 lease，不能与
  候选执行并发；reset 不属于客户端或候选请求路径，只能由受信运维在明确作用域、lease
  持有和策略允许时执行；
- `AttemptSupervisor` 不 import 厂商 SDK，仍复用容器 create/start/kill/delete/verify 语义；
  DeviceController 也必须只接受 typed、fence 绑定的操作，不能变成新的任意 shell 通道。

worker profile 的取舍：

1. **容器设备映射（优先）**：复用现有容器执行面，按 profile 映射必要设备节点、只读系统
   视图与网络策略。对清微而言，这首先表示每客户一套软件环境，不是设备隔离；host 与容器
   入口都映射到同一个 host 设备组，DeviceController/lease 在 host 侧统一，删容器只结束
   worker，不能当作设备资源或共享状态的清理证明。
2. **受信 host runtime**：仅当厂商明确不支持容器时另做生产级 supervisor；当前 process
   supervisor 是本地开发路径，不能直接宣称满足 NPU 生产隔离。
3. **厂商专用受信 RPC**：若设备只提供远程服务，新增的是共享受信设备控制适配层的内部
   传输，worker 与 DeviceController 都只能经 typed 接口使用，凭据留在控制区；否则外部控制器
   无法取消和对账异步服务。本站 JobEngine 仍负责 attempt、清理与证据，不把第三方服务直接
   当成本项目 trusted evaluator。

## 6. 清微 TSM 的映射方案

| 清微能力 | 适配位置与规则 |
| --- | --- |
| `txModuleLoad`/ELF/bin | `ArtifactCodec` 校验 executable module 的 manifest/digest；厂商 runtime 仍做模块签名校验。清微主路径由 control_host 的 ArtifactBuilder 预编译并交接，NPU 服务器不拉源码、不配完整编译环境；若选择 evaluator 端编译，也必须复用同一 BuildOutcome。不能假定 Metal/Triton 源码天然兼容 |
| SIMT/Cluster launch | `LaunchAdapter` 把公共 `KERNEL` 的受信 profile 子类型（SIMT/Cluster）映射到 `txLaunchKernel*`；grid/block/cluster 参数来自已校验 ExecutionSpec，AP 调度细节不外露 |
| graph/model/BPM | 映射到 `GRAPH_DEFINITION`/`MODEL_DEFINITION` role 与 `GRAPH`/`MODEL` typed launch；`txLoadGraph` 的 case 目录由 driver 从已校验 bundle 在 attempt 作用域内物化，agent 不提供路径；BPM 使用公共 `PARAMETER_TABLE`，不增加任意路径字段 |
| stream/event | 基础同步与 `DEVICE_EVENT` 计时；stream 创建/同步/销毁纳入 cleanup 证明 |
| raPTI/profiler | `PROFILER_DISPATCH` 计时适配，保留 launch correlation id、per-die 与全 die 汇总、flush/完整性；Host EXIT 不当设备时间 |
| `/dev/accel*` 与 host-hal | 每客户 Docker 提供独立软件环境/文件视图，通过设备映射、共享文件/IPC 与 host 交互；最终由 host runtime/driver 操作设备，不是每客户独立 NPU。受信 profile 精确映射所需节点；默认非特权。`--network=host`、`/tmp`、`/sys`、`/lib/modules` 都是显式例外，逐项最小化、只读优先并单独验收 |
| SMI/TxIPC | probe/topology/health；health 失败先隔离设备组。`tsm_smi --reset` 只支持全设备、需先结束占用，不能按 job/单卡 reset；按整机影响评估并限权，不在每个 job 默认执行 |
| discovery/C2C/MPI | 仅在站点 provisioning、能力发布前和故障恢复时由受信运维路径处理；发现失败即 capabilities 不可用/quarantine，不进入候选执行 |
| user/mesh/soft/slot id | 持久映射并记录证据；soft id 不单独作为稳定身份，serial/UUID 缺失列为发布阻塞问题 |
| tsm-shell、fabric_manager、免密 root SSH | 不向 agent/frontend 请求开放。只能在独立运维边界内使用，或要求设备方提供最小权限替代；凭据不得进入 workspace |

## 7. 部署、信任与发布门

- 每个 vendor profile 都要记录镜像 digest、adapter 版本、driver/firmware/runtime/SDK/profiler、
  OS/kernel、任务包、拓扑与 calibration；任一影响测量语义的变化都形成新 evaluator epoch。
- 容器默认非特权、只读 rootfs、无 host HOME/凭据、最小设备映射。厂商 host-hal 或多机组网
  所需的 host network、`/tmp`、`/sys`、`/lib/modules` 等宿主挂载是显式例外，逐项最小化、
  只读优先并单独验收，不能扩展成任意 egress 或通用宿主路径。
- NPU 服务器只保留最小受信执行环境：runtime/driver/evaluator、必要任务包与设备工具；
  LLM、git 凭据、源码仓库和完整编译工具链默认留在 control_host，不随工件下发。
- 每客户 Docker 只隔离其软件环境。设备组 lease 必须跨该 host 上所有客户容器与 host 入口
  统一判定；一个客户的容器删除不能释放其他客户的 lease，reset 也必须评估对所有客户及
  in-flight attempt 的影响，不能写成容器级 reset。
- 站点 provisioning 与每次 attempt 分开：root discovery、固件升级、fabric 配置不进 job；
  这些状态未验证时 GetCapabilities 必须失败或标为不可用。
- `simulated=true` 永不可发布。真实结果还需 measured、正确设备侧计时、完整证据、签名、
  site/epoch allowlist 和确认门；不同厂商的结果不共用 eval cache，也不因 task 名相同就默认可比。
- 第三方 device SDK、日志、profiler trace 和源码 bundle 同样受数据接收方、保留期、可读
  人员与二次用途约束。

## 8. 分阶段实施与验收

### P0：最小 source-only NPU 新 contract 与六 RPC 通信闭环

只证明最小闭环，不先扩 bundle、graph 或复杂计时：

- 新 NPU API/evidence contract 先只有 source 分支；bundle 在 P1 作为同一新 contract 的
  后续版本加入，不回到旧 v1；
- 最小 capability/feature 协商：新 contract ID、source artifact、`KERNEL`、单 fake 设备组；
  不展开 bundle/多 rank/die/profiler；
- 单 source fake vendor A：source → ELF（公共 `KERNEL`，profile 子类型 SIMT），device
  event；最小 `TimingEvidence` 只记录 source、clock domain、同步完成和单设备 elapsed；
- 六 RPC 全链路：GetCapabilities → Submit → GetJob/Watch；Cancel 覆盖显式取消，
  Retry 覆盖 interrupted attempt 的新 capability/fence 恢复；
- `CandidateDigest` 的必要持久链：canonical request、request digest、新 receipt
  （`candidate_digest`）、outbox、结果签名和 resume；新的 document_type/version 与 outbox
  行版本同时落地，旧 Metal/source contract 的字段和验证不变；
- 路径/digest/feature/限额 fail closed；旧 attempt/fence 结果拒绝；simulated/异常结果不可发布。

验收：同一客户端经六 RPC 跑通 A 的提交、观察、取消、恢复/重试；`CandidateDigest` 在
canonical request/request digest、receipt、outbox、签名和 resume 全链一致，篡改任一环节被拒；
现有 Metal/local 测试不变；`sim-npu` 等 fake gRPC 用例迁移到新 contract，不能继续走旧
contract。P0 不验收 bundle、graph-only 异步 fake、多 rank/die 或 profiler 计时。

**P0 实施记录（2026-10-27，代码已落地，未接入真实设备）**

- contract 身份：新增 `tcenv.remote.npu-source.evaluator-api.1` /
  `tcenv.remote.npu-source.evidence.1`，document_type 为 `.request-digest` /
  `.receipt-id` / `.result-signature`，`document_version=2`；旧
  `tcenv.remote.v1.*`（version 1）字段与摘要字节不变（同源对照脚本核对过
  `request_digest_document`/`request_digest`/`attempt_receipt_id`/`provenance_document`/
  `receipt_document`）。`contracts.ContractDialect` 选择身份字符串与 receipt 形状，不由
  请求形状推断；`dialect_for` 拒绝未知或混合的 API/evidence 配对。
- proto 增补：`EvaluationRequest` 字段 23/24（`CandidateDigest`、`NpuExecutionSpec`），
  `TargetCapability` 5–9（artifact/launch/subtype/device_groups/timing_sources），
  `EvaluatorProvenance` 18–22，`Receipt` 9/10。`scripts/generate_evaluator_proto.py
  --check` 通过（protoc 29.3 / grpc_python_plugin 1.69.0）。六 RPC 名字与语义未变，
  未新增厂商专用 RPC。
- P0 执行形状：`artifact_kind=source`、`launch_kind=KERNEL`、
  `launch_subtype=SIMT`、`synchronization=stream-event`、`required_timing_sources` 只允许
  `DEVICE_EVENT`、单逻辑设备组；bundle/graph/model/多 rank/die/profiler 一律 `ContractError`。
- 持久化迁移：job store schema v4 的版本白名单含 v3，v3→v4 原位迁移（加
  `attempts.device_binding_json`）与 v3 备份恢复都有独立回归，迁移后任务/结果/attempt 保留，
  旧 attempt 的绑定为空因此永不被别的设备组 reconcile；v1/v2→v4 路径保持。
- 终态读路径（source 已删除）：`contracts.npu_stored_request_document` 从持久化字段重建
  canonical request（精确字段集、逐字段校验、CandidateDigest 不依赖 source 字节、拒绝
  legacy `candidate_sha256`），重算并比对 `jobs.request_digest`；`_validate_result_envelope`
  与 store 打开时的 `_validate_durable_state` 都执行该检查，并交叉核对 accepted base
  provenance、`attempts.device_binding_json` 与签名 result 的 device group/artifact/launch/
  timing。因此改 `request_json` 的 `task`/`execution`/加 `candidate_sha256`、改
  `jobs.provenance_json` 或 attempt 绑定后，直接 `_job_by_id` 读与关闭后重开都 fail closed
  （`from_stored` 也不再静默丢弃 legacy 字段）。旧 Metal 读路径不变。
- `CandidateDigest` 链：canonical request → request digest（绑定 source sha256、execution
  spec 与 device group）→ receipt（`candidate_digest`，不双写 `candidate_sha256`）→
  outbox 行 `schema_version=2` + result envelope → HMAC result signature → resume。
  `GrpcEvaluator.wire_request_digest()` 给出 wire digest，host 侧唯一接线点是
  `taic.evaluation_outbox.bind_wire_digest()`（`taic/orchestrator.py` 在 `begin` 前调用，
  由 `tests/shared/test_orchestrator_runtime.py` 的 NPU host-adapter 回归覆盖 begin→Submit→
  complete→resume）；
  wire digest **不许回退**：缺字段时 client 在 Submit/GetJob 前、outbox 在
  begin/complete 时都以 `request_digest_missing` 拒绝，已记录值与服务端重算不一致则以
  `request_digest_mismatch` 在 Submit 前拒绝。outbox 在 complete 与 replay 共用的校验里按
  NPU dialect 重算 wire receipt（剔除宿主附加的 signature/measurement 元字段），所以
  retained result 的 `receipt_id`/规范字段被改写时 replay 直接拒绝。
- selfcheck 的宿主链路也绑定同一身份：`taic/selfcheck_bridge.py` 在提交前用
  `bind_wire_digest(..., npu_contract=...)` 绑定（selfcheck 没有 outbox 行），
  `GrpcEvaluator.selfcheck`/`selfcheck_resume` 在请求缺 wire digest 时用
  `wire_request_digest()` 绑定即将提交并返回的同一身份；旧 Metal selfcheck 路径不变。
- 设备组占用：是一个带引用计数的 in-flight 状态而不是普通锁。`GetCapabilities` 为整个
  probe/health/capabilities 序列占一个槽，每个 controller 调用再加一个引用，只有底层调用
  真正返回才减少引用——RPC 超时只结束等待，卡住的调用仍占着设备，scheduler 因此不会在
  探测未结束时 claim 或启动候选（bounded 超时后仍不得出现 running attempt/device
  resource/running marker）；调用真正返回后候选才启动并正常完成。scheduler 的 claim 与
  “attempt 占用设备组”在同一临界区内完成，probe 不会插进两者之间；`cancel`/`reconcile`
  属于该 attempt 自己的 owner 调用，不占 probe 槽。controller 也会在 running marker 存在时
  拒绝 probe/known-good 探针且不记录 probe。
- 清理证据必须存在：fake ledger 被删除时 `reconcile` 返回 `complete=False`
  （「ledger is missing」），读取/释放不会重建文件，`clear()` 只有测试装配
  （`operator=True`）可用；结果提交、Cancel、恢复路径都因此 quarantine。
- outbox 每个写读路径都核对既有行：`begin`/`complete`/`unavailable` 在任何 state 或
  result 写入前统一校验 `schema_version`、`transport`/API/evidence contract（NPU 缺字段
  直接拒绝，不按“旧行”放行）、`request_id`、`request_digest`、`CandidateDigest` 与状态；
  `complete` 拒绝改写已 completed 的行且失败时不写 result 文件，`unavailable` 不会重建或
  覆盖损坏的 NPU 行。旧 Metal 的宽松兼容语义保持。host 编排从 outbox 重放结果时直接返回
  该结果、不再重复 `complete`（也就不会尝试改写已完成行）。
- retained envelope 必须可认证：NPU outbox 强制要求 host 结果签名 key（缺失则构造
  `ValueError`），complete 与 replay 都要求 `result.signature` 并用该 key 做 HMAC 验证，
  覆盖 report/provenance/evidence/receipt 全字段。因此“伪造 attempt/fence 后用公开算法
  重算合法 `receipt_id`”“把 evidence 提成 measured/可发布”“改写 report 或 provenance”
  以及换 key/撤销 key 后重放都被拒；无 key 的 outbox 不再能保留 NPU 结果。旧 Metal 的
  host digest 与可选签名语义不受影响。
- 设备组绑定：`device_group_id/vendor_id/profile_digest/topology_digest/resource_generation`
  进入 capability（target.device_groups + provenance）、request digest、receipt、provenance
  与 Retry（Retry capability 的 device group 必须与 job 记录一致，串组即
  `failed_precondition`）。Submit 也在写入任何 job 之前逐项复核 capability 的完整绑定与
  request execution、accepted provenance 一致（含 artifact/launch/subtype/timing），幂等回放
  还要求与既有 job 的绑定一致，因此旧 generation 的 capability 不能为新 generation 的
  请求建立 job，也不会创建 attempt 或启动 worker。attempt/lease 把接受时的完整绑定持久化到
  `attempts.device_binding_json`（store schema v4），`CleanupProof` 携带同一组
  vendor/profile/topology/generation；cancel/reconcile 只对“该 lease 接受时的绑定”生效，
  重启换组时既不向新组 controller 发取消，也不能把旧 attempt 标成 interrupted，只能
  quarantine。
- 设备事实进入 capability：`GetCapabilities` 先用只读的 grant 预检
  （`JobStore.require_grant`，不消费 nonce、不写 capability 行）确认调用方已认证，
  未通过时直接返回 `unauthenticated`/`permission_denied` 且完全不触发设备操作；随后才用
  worker 外 controller 做有界
  `probe`/`health_check`/`capabilities`（`device_call_timeout_s`，默认 2s），并先确认该
  设备组没有活跃 attempt 或仍在 in-flight 的探测（scheduler 的 claim 与 attempt 占用同临界区，
  controller 再按 ledger 的 running marker 复核，busy 时设备侧连 probe 都不记录），核对
  事实里的 `logical_devices == 1` 与 `DEVICE_EVENT` timing，再逐项核对
  vendor/profile/topology/resource generation 与 P0 执行形状；探测失败、超时、不健康
  （含 `ok=True` 但没有真正执行 known-good 探针的 `executed_known_good=False`）或
  绑定不一致都 `failed_precondition`，且在消耗 nonce、写入 capability 之前失败。NPU 组合根
  缺 controller 属于装配错误（构造即 `ValueError`），运行期 controller 消失同样只能
  quarantine。`cancel`/`reconcile` 复用同一有界机制：超时按基础设施失败处理，job 保持
  quarantine、不提交结果、也不宣告 cancelled/interrupted（Cancel RPC 自身有界返回）。
- 超时与诊断证据：candidate deadline 仍产出 `status=timeout` 的 completed 报告，但 NPU 证据
  新增 `complete` 字段。`complete=false` 必须未同步、零 device 时间、零 launch、时钟域为
  `unobserved`、不可发布；`complete=true` 必须已同步、至少一次 launch、正 device 耗时且
  时钟域已观测；`status=ok` 必须 `complete=true`，`status=timeout` 不得 `complete=true`。
  `status=ok` 的报告还要满足 `candidate_us ≈ device_elapsed_ns/1000`（容差 1µs 级），
  由 `contracts.npu_report_timing_check` 在 store（worker result 与已存 envelope）、host
  client 与 outbox replay 三处执行，所以“candidate_us=50000 而 device_elapsed_ns=1000”
  这种自相矛盾结果会被拒。Host API 时间与 device event 时间继续分开，绝不把 host 等待写成
  device elapsed。
- driver 边界：`tcenv/remote/drivers/base.py` 只有 typed 协议与数据类（`VendorFacts`、
  `PreparedAttempt`、`VendorOutcome`、`CleanupProof`、`HealthProof`）；无 shell、路径、BDF、
  vendor key-value。fake vendor A（source → fake ELF/SIMT、device event、跨进程资源台账）只在
  `tests/support/vendor_sim/`，生产组合根不 import。
- 生产入口与测试组合根：`python -m tcenv.remote.grpc_service` 对 `npu-source` contract
  直接 `ValueError`（P0 没有生产 NPU adapter/controller），并拒绝把旧 contract 服务给非
  Metal target/backend。测试 side 只有一个 NPU gRPC 组合根（`tests/support/vendor_sim/`，
  新 contract）；`tests/support/npu_sim/grpc_service_entry.py` 明确降级为 legacy
  Metal/source transport 验收根，backend 默认 `sim-cpu` 并且拒绝任何 NPU 形态身份，
  `tests/support/job_engine.py` 的通用旧路径 fixture backend 改为 `sim-fake`。local、
  selfcheck bridge、私有 worker 帧、容器执行面、Metal 默认路径未改
  （`tests/support/npu_sim/fake_backend.py` 是本地 Backend、不发 gRPC 请求，不在 contract
  范围内）。
- 清理语义：worker 被杀/超时/取消后由 worker 外 controller reconcile；`complete=False` →
  job quarantine、不提交结果、不允许 Retry、`close()` 报错。终态 job 的 Cancel 不再向
  controller 发取消。Host API 时间与 device event 时间分开记录，`candidate_us` 取自
  device event。fake vendor 的 `prepare` 全程使用已校验的当前绑定（`PreparedAttempt` 与
  ledger 行都写 vendor/profile/topology/generation），`reconcile`/`release_attempt` 在释放前
  逐项核对绑定：异组或异代际的资源既不删也不证明已清理，proof 保持 `complete=False` 并把
  它们列为 live resources 供 quarantine。运行状态同样属于租约：`mark_device_running` 记录当时
  的完整绑定，只有匹配绑定的 driver `close` 或 controller `reconcile` 才能清除它，且清除后
  必须复核为空才可返回 `complete=True`；仍有 marker（或只剩 marker）时 proof 一律
  `complete=False`，异组/异代际的 marker 原样保留。成功 proof 还必须是
  `complete=True` 且 `live_resources` 为空：自相矛盾（声称完成却仍列 live resource）的
  proof 按基础设施清理失败处理，结果不提交、Cancel 不终结、Retry 被拒。
- 终态 request_id：`request_id` 不属于 canonical digest，因此
  `npu_stored_request_document` 单独校验其标识符形状，`_validate_result_envelope` 与
  `_validate_durable_state` 都把它与 `jobs.request_id` 比对；篡改后直接读与重开都
  fail closed。
- 验证证据（每条命令各自单独跑）：
  `tests/shared/test_remote_npu_{contract,grpc,cleanup}.py` **88 passed**
  （wire receipt 与 host HMAC 篡改、时钟/候选时间自相矛盾、hanging cancel/reconcile、
  终态 request_id/task/execution/provenance/attempt 绑定篡改、Submit 跨 generation
  capability、fake vendor 的代际 prepare/台账、运行 marker 与异组/异代际
  reconcile 拒绝并保留对方 running 证据、无效/过期/撤销 grant 下零设备调用、
  自相矛盾 proof（complete=True 且有 live resource）在提交/Cancel/Retry 三路 quarantine、
  未执行 known-good 探针的健康证明不广告 available、outbox 行篡改/已完成行不可改写/
  损坏行不被静默重建、活跃 lease 下 GetCapabilities 零设备调用、宿主 selfcheck/adapter
  绑定 wire digest、probe 序列与候选启动互斥、缺失台账不算清理、组内 timing/设备数矛盾
  被拒）、探测超时后设备仍被占用直到真实调用返回、capability 缺 launch_subtypes/事实
  多设备或 HOST_WALL 被拒）；
  `test_remote_grpc_{adapter,tls,contract,revocation}.py` 与
  `test_remote_{conformance,go_conformance}.py` 六个旧文件 **62 passed**；
  `test_orchestrator_runtime.py` **20 passed**（含 NPU host 接线回归与 blob wire-candidate 证据
  校验：合法 blob 结果通过、证据换 candidate 被拒）；
  `test_remote_{job_engine,job_recovery,legacy_worker_bridge,container_engine,
  container_supervisor,runtime}.py` + `test_evaluation_outbox.py` **265 passed, 1 skipped**
  （跳过项是 Linux-only 的 `PF_KTHREAD` 检查；v3→v4 原位迁移与 v3 备份恢复回归在
  `test_remote_job_recovery.py`，`_validate_durable_state` 的终态读校验同文件覆盖）；
  `test_selfcheck_bridge.py` 在本机围栏默认 TMPDIR 下 **18 failed, 1 passed**（全部是
  短 TMPDIR 导致的 AF_UNIX 路径超长），`TMPDIR=/tmp` 下 **19 passed**。
  `uv run pytest -q -m 'not gpu and not llm and not slow'`：**2287 passed / 6 skipped /
  32 failed**。这批失败全部
  是本机 hmz 围栏环境造成：`test_selfcheck_bridge.py` 的 AF_UNIX 路径超长（短 TMPDIR），
  `ps` 被禁（`test_children`、`test_run_detached`、`test_triton_baseline`）与 HOME 权威
  日志不可写（`test_selfcheck`、`test_harness`、`test_kernel_agent`）；失败文件不 import
  remote/NPU/新测试。
- 未做（不得据此宣称已支持）：bundle/oneof/`artifact_set_digest`、graph-only fake B、P2a
  TSM simulator、P2 真实设备、profiler 计时、多 rank/die、device group 并入 evaluator
  epoch、跨 contract 的滚动升级（legacy state 与 npu-source state 不兼容，换 contract
  需要新 state 目录与新 evaluator epoch，Docker 滚动重启验收因此未重跑）、
  `tests/support/npu_sim/grpc_sim_runner.py` 的 Docker 验收（本机无 Docker 权限，只做了
  import/入口拒绝的回归）。所有结果为 `simulated=true`、`publication_allowed=false`，
  不产生性能或发布结论。

### P0b：独立 content-addressed blob 通道与候选 blob 引用（已实现，无真实设备）

在 P0 闭环之上只加数据面与必要集成，不改 P0 语义：

- 设计、状态机、授权/配额/清理与模拟矩阵见 §4.5；
- 新 contract `tcenv.remote.npu-blob.*`（document version 3）只增加“候选字节来自 blob”这一形态，
  `artifact_kind=source`、`KERNEL`/`SIMT`、单设备组、`DEVICE_EVENT` 与 P0 完全一致；
- 独立 gRPC `BlobChannel` 服务（新 proto package），不新增 Evaluator RPC，不改六 RPC 与旧
  field number/digest/HMAC；
- 验收不覆盖 P1 的 bundle/graph-only fake、P2a 的 TSM simulator 或任何真实设备。

**P0b 实施记录（2026-10-28，代码已落地，未接入真实设备/对象存储）**

- 数据面：`tcenv/remote/blob_channel.py`（manifest 状态机、chunk 文件、scope 绑定、配额/TTL、
  lease 与 sweep、quarantine）与 `tcenv/remote/blob_service.py`（`BlobChannel` servicer）。proto 为
  `proto/tcenv/remote/blob/v1/blob.proto`，生成物在 `tcenv/remote/blob/v1/`，由
  `scripts/generate_evaluator_proto.py` 一起生成并 `--check`；`evaluator.proto` 只新增字段 25
  `EvaluationRequest.blob_ref`，六 RPC 与 1..24 号字段不变，生成物字节可复现。
- 契约：`contracts.NPU_BLOB_DIALECT`（`tcenv.remote.npu-blob.evaluator-api.1` /
  `tcenv.remote.npu-blob.evidence.1`，document version 3）。blob 形态 `candidate_digest.algorithm`
  固定 `sha256-blob-v1`，`blob_ref` 必填且禁止 source 字节；`npu-source` 拒绝 `blob_ref`。
  receipt、provenance、timing evidence 与 result signature 形状与 P0 相同，digest 通过
  `candidate_digest` 进入，descriptor 通过 request digest 进入。
- 集成：`Submit` 在写入任何 job 之前要求 blob `ready`、scope 相同、descriptor 与请求一致
  （未 ready 或跨 scope 一律拒绝，且不建 job、不建 attempt、不取 lease）；attempt dispatch 由
  引擎注入受信 blob 根（`<state_dir>/blobs`，agent 不能提供路径），fake worker 按 content id
  分块读取并复核整份 digest，chunk 数/最大 chunk/摘要写进 fake 台账；终态释放 lease 并按策略
  清理，cleanup 失败 → blob `quarantine` + job 记录 hazard；崩溃/重启后 `reconcile_blobs` 按
  `JobStore.live_request_ids()` 释放孤儿 lease。
- host 侧：`EvaluationRequest.wire_candidate_digest`（host-only，不进 `to_dict`/旧 digest）
  记录上传产物的 digest，`bind_wire_digest` 同时绑定 wire digest 与 wire CandidateDigest；
  blob outbox 缺该字段即 `request_digest_missing`，row/result 的 `candidate_digest` 按 wire 值
  校验，篡改 row、receipt、evidence 或 host 签名都被拒。
- 配额与重放的复审修复（`af575be` 之后）：字节配额原先只统计已落盘 chunk，两个 8 字节声明在
  `max_scope_bytes=max_total_bytes=12` 下都能通过并各自写满；现按声明大小在 `begin` 预留、在
  `write_chunk` 复查，账本另给出 `reserved_bytes`/`quota_bytes`。重放原先只认最后一块，写完
  offset 0/4/8 后重放 offset 0 被误拒；现按声明边界定位任意已确认块并核对完整长度与字节。
  新增 store 用例（两个未完成声明同 scope / 跨 scope 超全局配额、abort 释放预留、配额调小后
  写入 fail closed 且零落盘、非整除末尾块重放）与 gRPC 用例（BeginUpload 预留 scope/全局配额、
  跨多个后续 chunk 重放 offset 0/4、非边界与不同字节拒绝）；并发相同 content id 在幂等重放下
  不再报错，断言改为前缀连续、可提交、逐字节一致。
- 幂等 replay 与 outbox 身份的复审修复（`35d8364` 之后）：
  `JobEngine.submit` 先查持久 job 行，只有**新** accepted job 才要求 ready blob 并取 lease；终态
  清理已删除字节的 job 再以相同 request/digest `Submit` 时从已提交行回放（同一 attempt/receipt，
  不新建 lease、不重建字节），不同 digest 或另一 grant scope 仍按原语义拒绝。原先先取 lease 再查
  重放，导致 completed 之后重复 `Submit` 得到 `not_found: blob does not exist`。
  `EvaluationOutbox._expected_candidate` 对 blob contract 不再回退到 host `candidate_sha256`
  （空 source 摘要），缺 `wire_candidate_digest` 即 `request_digest_missing`；`unavailable` 在
  新建或改写任何 blob 行之前执行同一身份门（缺 wire digest 或缺 wire CandidateDigest 都拒绝；
  无持久接受记录时不写行、静默返回，沿用既有 missing-wire 语义）。原先 `unavailable` 只检查
  wire digest，会写入一行伪造的 `sha256-blob-v1` 空内容摘要。
- 编排层与并发提交的复审修复（`946cb34` 之后）：
  host 编排的 NPU evidence 校验改用 wire CandidateDigest（blob 形态下 host `candidate_sha256`
  是空 source 摘要，不能作为期望值；缺 wire candidate 即 `request_digest_missing`，证据
  candidate 的 algorithm/value 必须与 outbox dialect 一致）。`JobEngine.submit` 把「重放判定 +
  取 lease + 持久提交」放进同一临界区，两个同 request id、不同 blob/digest 的并发 Submit 只会
  留下胜方的 lease（败方 `already_exists`，既不新增孤儿 lease 也不误删胜方 lease）。
  `bind_wire_digest` 先检查已有 `wire_candidate_digest` 是否被 evaluator 改变，再做首次 blob
  绑定（矛盾即 `request_digest_mismatch`，evaluator 不再给出 candidate 即
  `request_digest_missing`），并且不再用「candidate 是否等于 host 摘要」推断形态，零字节 blob
  （其摘要恰等于空 source 摘要）也能正确绑定。
- 变异核验（临时改动后复原）：跳过 commit 摘要校验、忽略 cleanup 失败、接受旧 attempt/fence、
  跳过 scope 校验、同时去掉两道 ready 守卫，对应测试全部转为失败；把配额改回只算落盘字节会让
  预留用例失败；把 submit 改回先取 lease 会让 completed/failed/重启后的 replay 用例失败；把
  outbox 的候选回退与 `unavailable` 门去掉会让缺 candidate 的用例失败；把编排层改回比较
  host `candidate_sha256` 会让 blob 证据用例被误拒；把 Submit 的临界区换成空上下文会让并发用例
  重新出现两个 lease；把 `bind_wire_digest` 改回先按值比较会让重绑定篡改用例通过篡改。
  FaultIO 记录的 chunk/offset、台账与 `io.crashes` 证明故障确实触发。
- 验证证据（每条命令单独跑）：
  `tests/shared/test_remote_blob_channel.py` **36 passed**（边界、二进制、读写峰值、崩溃续传、
  任意已确认块重放与短尾块、乱序、错误 digest、同 id 不同大小、未 ready、跨 scope、旧 attempt、
  未完成声明按声明大小预留 scope/全局配额、abort 释放预留、配额调小后写入零落盘、并发、过期/
  sweep、cleanup quarantine、孤儿 release、corrupt chunk 检测）；
  `tests/shared/test_remote_blob_grpc.py` **18 passed**（真实 gRPC 上传/流式下载逐字节一致、
  空文件、缺失/过期/撤销/越权 grant 零落盘或零读取、越界/超 chunk/乱序拒绝、断点续传、
  重复 chunk/commit、digest 不符 fail closed、同 id 不同大小、abort、慢消费者有界预取、
  message 无路径/URL/命令、BlobChannel 与六 RPC 分离且 field number 不变）；
  `tests/shared/test_remote_blob_contract.py` **21 passed**（未 ready/跨 scope 不可 Submit、
  端到端 digest 链与 fake worker 分块读取台账、completed/failed/重启后的同 request 重复 Submit
  从持久行回放且不新建 lease、同 request id 不同 blob 的并发 Submit 只留胜方 lease、outbox
  绑定/replay/重绑定篡改与缺字段拒绝、`unavailable`
  缺 wire CandidateDigest 不写行/不伪造空内容摘要、store 内 `blob_ref` 的 sha/size/chunk/media
  篡改在直读与重开都 fail closed、interrupted attempt 用同一 blob Retry、cleanup 失败
  quarantine、重启释放孤儿 lease、npu-source 拒绝 `blob_ref` 而旧 Metal digest 仍是 version 1）；
  `tests/shared/test_remote_blob_go_conformance.py` **7 passed**（Go 独立重建 canonical
  request/receipt/签名并拒绝 candidate/blob size/canonical JSON/receipt id/HMAC 五类篡改）。
  既有 P0 回归：`test_remote_npu_{contract,grpc,cleanup}.py` **88 passed**；旧
  `test_remote_grpc_{adapter,tls,contract,revocation}.py` + `test_remote_{conformance,
  go_conformance}.py` **62 passed**；`test_remote_{job_engine,job_recovery,
  legacy_worker_bridge,container_engine,container_supervisor,runtime}.py` +
  `test_evaluation_outbox.py` + `test_orchestrator_runtime.py` **283 passed, 1 skipped**
  （Linux-only 跳过项不变）；`test_selfcheck_bridge.py` 在 `TMPDIR=/tmp` 下 **19 passed**，
  在本机 hmz 围栏默认 TMPDIR 下仍是 **18 failed / 1 passed**（AF_UNIX 路径超长，未改测试）。
  `uv run pytest -q -m 'not gpu and not llm and not slow'`：**2371 passed / 6 skipped /
  32 failed**，32 项仍全是本机围栏环境（selfcheck AF_UNIX 路径、`ps` 被禁、HOME 权威日志
  不可写），失败文件不 import remote/NPU/blob 新代码。
- 未做（不得据此宣称）：大文件/GB 模型吞吐与容量、真实对象存储、跨站点性能、真实 NPU 大工件
  加载、multipart/运维配额、服务端 SIGKILL 时的存储原子性、bundle/graph-only fake、P2a TSM
  simulator。所有结果 `simulated=true`、`publication_allowed=false`。

### P1：bundle、graph-only 异步 fake 与 conformance

在 P0 闭环上扩展，不改 P0 的 source-only 语义：

- 用同一新 contract 的后续版本启用 source/bundle oneof 与 ArtifactBundle/ExecutionSpec；
  bundle 必须新版本化，不能作为旧 v1 的可选字段；
- 第二个 fake vendor B：graph-only（`GRAPH`）+ 异步服务式执行，无 ELF/stream/rank/die，
  以服务返回的 completion token 映射同步与 timing evidence；
- 扩展 `TimingEvidence`：按声明拓扑判定完整性，无 rank/die 不判缺失；Host launch 时间单独
  记录，不能写成 device elapsed；bundle 的 `artifact_set_digest` 进入签名 result envelope；
- 完成 bundle 的 CandidateDigest/receipt/outbox/签名/恢复闭环，以及 Python/Go golden；
- 旧 contract 只接受现有 Metal/source 及旧验证语义；所有 NPU 请求（含 simulated/fake 与
  单 source）一律走新 contract；`sim-npu` 等 fake 不再有旧 contract 例外；旧 contract 只对
  现有 Metal target 开放，不靠请求形状判断；服务端按 contract + required_features fail-closed；
- unknown Protobuf field 由服务端按 contract 明确拒绝；不得改变旧 Metal source 请求、六 RPC
  或旧结果验证。

**P1 实施记录（2026-10-29，代码已落地，未接入真实设备/SDK/对象存储）**

- contract 身份：新增 `tcenv.remote.npu-bundle.evaluator-api.1` /
  `tcenv.remote.npu-bundle.evidence.1`，document_type 为 `.request-digest` /
  `.receipt-id` / `.result-signature`，`document_version=4`。P0（v2）、P0b（v3）与旧
  `tcenv.remote.v1.*`（v1）的 document/字段/摘要字节不变；`ContractDialect.artifact_form`
  增加 `bundle`，`candidate_algorithms` 在 v4 为 `("sha256", "sha256-bundle-v1")`。
- source/bundle oneof：v4 request 的 `artifact` 必须恰好一个分支 ——
  `{"source": {size, sha256}}`（inline 源）或 `{"bundle": <ArtifactBundle>}`（工件集）；两个或
  零个分支、以及把 bundle 塞进 v3、把 v4 请求喂给旧 contract，全部 `ContractError`。v4 拒绝
  request 级 `blob_ref`（大件按 entry 走 P0b 通道）。分支与 contract 版本的双向选择都是
  fail closed，不靠请求形状猜 contract。
- ArtifactBundle（`tcenv/remote/artifact_bundle.py`）：
  `schema_id=tcenv.remote.npu-bundle.artifact-bundle.1`；每项只有相对 POSIX 路径、公共 role
  （`SOURCE`/`EXECUTABLE_MODULE`/`GRAPH_DEFINITION`/`MODEL_DEFINITION`/`PARAMETER_TABLE`/
  `AUXILIARY`）、media type、实际字节 size 与 sha256；
  `artifact_set_digest = SHA256(RFC8785(canonical_manifest))`，manifest 只覆盖 schema_id、
  schema_version、path/role/media_type/size/sha256 并按 path 稳定排序，不含字节、传输形态、
  device group、placement、ExecutionSpec 或 cache policy。实际字节复核不做换行/Unicode 规范化
  （CRLF/BOM 是另一个工件）。拒绝绝对路径、`..`/`.`/空段、反斜杠、控制字符、前后空白、
  重复与大小写冲突路径、符号链接、设备节点/FIFO、超 inline 单件 262144/总 1000000/64 件限额、
  size 或 sha256 与字节不符、blob 引用与 manifest 身份不符；小件不允许 blob、大件不允许 inline。
- ExecutionSpec：v4 文档显式含 `schema_id`、`task_family`（`kernel`/`graph`）、
  `artifact_kind`、`launch_kind`、`launch_subtype`、media type、逻辑 `device_group`、
  `resources`（`logical_devices` + `required_rank_spans`/`required_die_spans`，`null` 表示该
  维度不存在）、必需 timing source 与 synchronization；形状由 task family 推导
  （kernel→source/KERNEL/SIMT/stream-event，graph→bundle/GRAPH/SERVICE/
  service-completion-token），cache policy 仍是 request 级字段；BDF/物理 rank/fabric 没有任何
  字段。`VendorFacts.ranks/dies`（0 = 无该维度）来自受信 profile，引擎在 GetCapabilities 的
  probe 阶段逐项比对，agent 声明 rank span 而 profile 没有该维度时在 create/start 前
  `failed_precondition`（不建 job、不建 attempt、不碰设备）。
- 第二个 fake vendor B（`tests/support/vendor_sim/vendor_b.py` + `graph_service.py`）：bundle/
  `GRAPH`/`SERVICE`，**无 ELF、无 stream、无 rank/die**；`prepare` 把 graph job 交给文件支撑的
  异步服务并立即返回 `queued`（`launch_returned_before_completion=true`），`execute` 轮询服务
  直到它返回本次提交的 completion token，`device_elapsed_ns` 只来自服务记录，
  `host_api_elapsed_ns` 是 host 轮询耗时；它与 A 的形态差异是编译产物/同步语义/完成观察三者
  同时不同，而不是改名复用 A 的 SIMT 行为。A 也被同一 v4 contract 的 source 分支服务
  （`BundleSourceGrpcHarness`），证明 oneof 两分支都真实可跑。
- TimingEvidence v2（`tcenv.remote.npu-bundle.timing-evidence.2`）：新增
  `artifact_set_digest`、`topology`（`logical_devices`/`declared_devices`/
  `required_rank_spans`/`required_die_spans`）、`spans`（每设备一条，rank/die 不适用时为
  `null`，且不得凭空编造声明里没有的维度）与可选 `buffer`（flushed/written/delivered/
  dropped/overwritten）。完整性按声明拓扑判定：缺声明设备 span、缺声明 rank/die span、时钟域
  未知、`device_elapsed_ns` 不等于最慢 span、buffer 丢弃/覆盖/未 flush/少交付、span 时钟域
  不一致 → 只能 `complete=false` + `publication_allowed=false` 的诊断行；`status=ok` 必须完整，
  `status=timeout` 不得完整；Host 时间与 device 时间始终两个字段，v1 evidence 一字未改。
- 链与持久化：v4 canonical request（含 `artifact` oneof）→ request digest → receipt
  （`candidate_digest` + device group，无 `candidate_sha256`）→ outbox 行 `schema_version=2` →
  HMAC result signature → resume。bundle 的 inline 字节以 canonical payload 存进 job 的
  `source` 列（与旧 source 同一生命周期与终态清理），`from_stored` 重建 typed bundle 并复核
  payload 与 `artifact` 文档一致。host 侧新增 `wire_candidate_algorithm`（host-only，不进
  `to_dict`/旧 digest）：v4 两个分支的 CandidateDigest 算法不同，只绑 value 会让一个 bundle
  摘要被当成 inline 摘要，outbox 缺 algorithm 即 `request_digest_missing`。旧 Metal/source 与
  P0/P0b 的 `candidate_sha256`/receipt/outbox/HMAC 语义不变。
- worker 内/外边界：`FakeVendorBDriver.close` 只做协作式 collect；真正的释放由 worker 外
  `FakeVendorBController.reconcile` 按 attempt/fence/完整绑定核对服务状态后给出
  `CleanupProof`。丢失/重复/错误 completion token、`never_complete`、worker 在 submit 后被
  SIGKILL、服务拒绝释放（`keep_service_job`）→ 都没有 `ok` 结果；清理未证明即 quarantine 且
  Retry 被拒；服务把提交留在 live 时证据保持可观察。
- 六 RPC 与 wire：`evaluator.proto` 新增 `ArtifactBundle`/`ArtifactBundleEntry`（entry 内容是
  真 oneof：`data` 或 `blob_ref`）、`EvaluationRequest.bundle = 26`、`NpuResourceRequirements`
  与 `NpuExecutionSpec` 8..11；六 RPC 名字、1..25 号字段、旧 digest/HMAC 不变，
  `scripts/generate_evaluator_proto.py --check` 通过（protoc 29.3）。旧 contract 收到 v4
  execution/bundle 字段按 contract 明确拒绝；未知 Protobuf field 在 request、execution、
  bundle、bundle entry 四级拒绝（`reject_unknown_fields`）。
- Python/Go golden：`tests/support/golden/remote/npu_bundle_v4.json` 由生产函数生成，独立 Go
  verifier（`tests/support/remotecontract`）只读原始字段重建 canonical manifest、
  artifact_set_digest、request document、receipt、result envelope 与 HMAC，并独立复算 v2 证据的
  完整性规则（声明设备 span、聚合 device 时间、host/device 分离、无 rank/die）；旧
  `evaluator_v1.json` 与 `npu_blob_v3.json` 向量继续通过。
- 验证证据（每条命令单独跑）：`test_remote_npu_bundle_contract.py` **42 passed**、
  `test_remote_npu_bundle_grpc.py` **8 passed**、`test_remote_npu_bundle_cleanup.py`
  **14 passed**、`test_remote_npu_bundle_go_conformance.py` **12 passed**（合计 76）；P0/P0b
  回归 `test_remote_npu_{contract,grpc,cleanup}.py` +
  `test_remote_blob_{channel,contract,grpc,go_conformance}.py` **170 passed**；旧
  `test_remote_grpc_{adapter,tls,contract,revocation}.py` + `test_remote_{conformance,
  go_conformance}.py` **62 passed**；`test_remote_{job_engine,job_recovery,
  legacy_worker_bridge,container_engine,container_supervisor,runtime}.py` +
  `test_evaluation_outbox.py` **265 passed, 1 skipped**；`test_selfcheck_bridge.py` 在本机
  围栏默认 TMPDIR 下 **18 failed / 1 passed**（AF_UNIX 路径超长，未改测试），`TMPDIR=/tmp` 下
  **19 passed**。`uv run pytest -q -m 'not gpu and not llm and not slow'`：**2447 passed /
  6 skipped / 32 failed**，32 项仍是本机围栏环境（selfcheck AF_UNIX 路径、`ps` 被禁、HOME
  权威日志不可写），失败文件不 import P1 新代码。
- 变异核验（临时改动后复原，对应测试转失败）：跳过 bundle manifest 的 digest 复核、允许
  `..`/`.` 路径段、允许 artifact 同时带两个分支、跳过声明设备 span 检查、让 fake B 用 host
  时间当 device 时间、让 driver 接受任意 completion token、让 `_reconcile_device` 直接返回
  成功。故障注入先证明触发（服务记录 `fault_fired`/`device_failed`、或提交仍在 live）再断言
  结果。
- 未做（不得据此宣称）：P2a TSM simulator、真实 TX SDK/driver/firmware、真实 graph 包格式与
  签名校验、多 die/rank 聚合与 profiler 计时、GB 级工件与对象存储吞吐、容器设备映射、
  生产发布；`python -m tcenv.remote.grpc_service` 仍对 NPU contract 直接报错，fake vendor 只在
  `tests/support/vendor_sim/`。所有结果 `simulated=true`、`publication_allowed=false`。

**P1 复核修复（2026-10-29，Cursor 中断后的独立复核；仍无真实设备）**

每条都先写出能失败的复现，再改代码，最后做变异核验（把修复临时改回旧行为，确认对应测试转
失败）。逐项结论：

1. **`materialize` 的父目录符号链接逃逸（确认，已修）**。复现：在 attempt 目录里放
   `graph -> <bundle 根之外>`，旧实现 `target.parent.mkdir(parents=True, exist_ok=True)` 成功
   跟随该链接，`open("xb")` 把字节写到了根之外（实测外层目录真的出现 `main.json`）。旧代码只
   检查最终 target 与 root 本身，没检查中间段。修复：`_descend_without_symlinks` 逐段 `lstat`
   （链接/普通文件/设备节点一律拒绝），最终文件用 `O_CREAT|O_EXCL|O_NOFOLLOW` 打开，回读也用
   `O_NOFOLLOW` 并 `fstat` 复核是普通文件；root 本身也按 `lstat` 拒绝链接或非目录。回归：
   `test_materialize_refuses_a_symlinked_parent_instead_of_writing_through_it`（断言根外目录仍为
   空）、`..._at_the_target_name`、`..._a_symlinked_root_or_a_file_in_the_way`。同时把
   「一个条目路径是另一个的父目录」在 manifest 校验期就拒绝（否则 materialize 会写一半）。
2. **v4 result evidence descriptor 误标 v1（确认，已修）**。复现：走真实六 RPC 跑完一个
   graph job，直接读 `EvaluationResult.evidence`（不经过内部 document）——旧值
   `schema_id=tcenv.remote.npu-source.timing-evidence.1` + `media_type=application/json`，而
   实际字节是 v2 document。两端都用同一个错值（`job_store` 按 `npu_source` 选 v1，
   `grpc_client` 也按 `npu_source` 期待 v1），所以互不报警；Python golden 里却是 v2 schema +
   v2 media，签名文档也用 v2，于是 wire 与 golden 互相矛盾。修复：`ContractDialect` 显式带
   `evidence_schema_id`/`evidence_media_type`，v4 取 v2/v2；`job_store` 生成与持久读、
   `grpc_client` 校验、`result_signature_document`（两端与持久读共用的签名入口）都按 dialect
   核对，report descriptor 也只能是 `tcenv.eval-report.v3`。v4 是第一个「强制自己写出的身份」的
   版本（`enforce_evidence_identity=True`）；v1–v3 仍写历史 `application/json` 但不再由
   `npu_source` 猜 schema，而是每个 dialect 显式声明，且 P0/P0b 的真实 envelope 断言把这两个
   身份钉住（否则同样会被下一次改动悄悄带上 v1/legacy id——本轮复核时确实先写坏过一次，
   被新增断言当场抓住）。回归：六 RPC 测试直接断言 `final.result.evidence` 的 schema/media
   等于 golden 的 `unsigned_result.evidence`，`test_the_signed_evidence_descriptor_is_the_version_two_identity`
   证明三种误标都被签名入口拒绝，P0/P0b 各加一条 descriptor 身份断言。v1–v3 的 media type 与
   旧 golden 的不一致仍在（冻结），风险写进剩余风险。
3. **TimingEvidence 的设备×rank 唯一性与完整性（确认，已修）**。复现：一设备两 rank、
   `required_rank_spans=2` 的完整证据被旧规则「timing evidence reports a device span twice」
   拒绝——即 `required_rank_spans>1` 永远无法 complete；而「2 设备只报 device-0 的两个 rank」
   这类缺设备组合旧规则只会给出「重复设备」，rank 集合 + `len(spans)%R` 的启发式根本看不到
   device×rank 坐标。修复：span 唯一性改成 `(device_id, rank, die)` 坐标；完整性改成声明拓扑的
   精确笛卡尔积（每设备 × 每 rank × 每 die 恰好一条，多一条/少一条都拒）；span 只能命名
   `declared_devices` 里的设备，rank/die 超出声明范围即拒；`null` 维度仍不要求、也不许编造，
   graph-only 厂商不受 rank/die 惩罚。`device_group_requirements`/`_topology_document` 增加
   「坐标数 ≤ 64」上界，一个永远无法覆盖的拓扑在 request 阶段就被拒。回归：
   `test_a_device_with_two_ranks_reports_one_span_per_rank`、
   `test_completeness_is_the_cartesian_product_of_the_declared_topology`（缺设备/缺某设备某
   rank/重复坐标/超范围 rank/未声明设备）、
   `test_a_graph_only_vendor_is_not_penalised_for_absent_rank_or_die_spans`、
   `test_a_topology_no_evidence_row_could_cover_is_refused_up_front`；Go verifier 用同一组
   golden 向量（`npu_bundle.completeness_vectors`，6 个含 4 个应拒）独立复算，Python 侧也断言
   两边判决一致。
4. **多 blob entry attach 失败的回滚范围（部分证伪，已收紧；复核后补全第二条路径）**。复现
   尝试：job A 已持有 X 的 lease，再提交引用 X + 未 commit 的 Y 的 job B——旧实现 `release(B)`
   只删 B 的 lease，A 的 lease 保留（按 request id 释放，不会跨 request）。真正的隐患是同一
   request id 上**不同 attempt** 的 lease（P0b 通道把 lease 绑 attempt/fence，P2a 会用到）：
   旧回滚按 request id 释放会把旧 attempt 的 lease 一起删掉。修复：`BlobStore.detach(scope,
   content_id, request_id)` 只删单个 (blob, request) 的 lease（字节保留、不删文件），
   `_lease_blob` 记录本次循环真正新增的 content id 并只回滚这些。回归：
   `test_a_rejected_multi_entry_submit_only_rolls_back_its_own_leases`（旧 attempt 的绑定 lease
   存活、本次新 lease 回收、被拒 Submit 不留 job、字节仍可用）。诚实结论：按字面「释放同
   request 已接受 job 的 lease」在现实现里不可达（`job_exists` 为真时根本不 lease，且
   `_submit_lock` 串行化），已在此说明而不是当成已发生的 bug。
   第二轮复核又发现同一条 attach 之后还有**第二条全 request `release`**：`JobEngine.submit`
   在 `_lease_blob` 成功、`store.submit`（例如 capability 已失效）随后失败时，仍调
   `blob_store.release(request_id, cleanup=False)`。复现：同一 request 先持 `older` 的
   attempt/fence lease，再提交只引用已 ready `fresh` 的 bundle 并给一个从未签发的
   capability——调用前 `older.leases == (request_id,)`，调用后变成 `()`，`fresh` 也被清空。
   修复：`_lease_blob` 返回 `_ArtifactLease(scope, content_ids)`（本次真正新增的 lease），
   `submit` 的失败路径改为 `_drop_attached_blobs(lease.scope, request_id, lease.content_ids)`；
   `job_engine` 里已不存在按 request id 的整体 `release`（只剩终态清理 `_release_blob_lease`，
   那本来就要释放该 request 的全部 lease）。回归：
   `test_a_store_rejection_after_attach_keeps_an_older_attempt_lease`（旧 attempt lease 存活且
   仍绑 attempt/fence、本次新 lease 回收、无 job row、两份 blob 仍 ready），保留原 attach 失败
   回归；变异核验：把该路径改回 `release(request_id)` 时新回归立即转失败。
5. **没有 blob 通道时 blob 形态 bundle 仍被接受（确认，已修）**。复现：无 `blob_store` 的 v4
   frontend 提交带 blob entry 的 bundle，旧实现照常建 job/attempt、lease 设备，worker 才因缺
   `TCENV_BLOB_ROOT` 返回 17，被记成可重试的 infrastructure failure。修复：`submit` 在
   `_validate_submission` 里先拒绝（`failed_precondition`），`retry` 在重派前用持久化 request
   做同一判断，`_lease_blob` 也不再无声跳过。回归：
   `test_a_bundle_that_needs_the_blob_channel_is_refused_before_it_exists`（断言无 job row、服务
   无 submission，同前端 inline bundle 仍可完成）。
6. **同类变体排查**：async completion token（丢失/重复/错误/never_complete，worker 被杀，服务
   拒绝释放）与资源账本本轮复核未发现新漏洞，对应回归在原文件保持；report descriptor 的
   schema 身份此前无人校验，已并入第 2 项的签名入口检查；bundle 终态 blob 清理失败原先只有
   P0b 单 blob 的回归，现补 `test_a_failing_bundle_blob_cleanup_is_quarantined_not_hidden`
   （注入 `remove` I/O 失败 → blob quarantine + job 记录 hazard，不伪装成干净清理）。“回滚只
   覆盖本次 lease”这条同类漏洞确实还有第二条路径（见第 4 项第二轮），已一并修掉并补回归。

**P1 复核修复（第二轮，2026-10-29；仍无真实设备）**

第二轮独立复核又逐项复现了 7 个问题，全部先留下能失败的回归再修（第 3 项的 dispatch 守卫在
第三轮被补全，见该项末尾；其余各项在第二轮即闭合）：

1. **`materialize` 仍有父目录 symlink 的 TOCTOU（确认，已修）**。复现：在「逐段检查」之后、
   写入之前把已检查的 `graph` 目录换成指向根外的链接（脚本注入），旧实现虽随后因第二项报
   `BundleError`，根外已经真的出现 `main.json`——`O_NOFOLLOW` 只保护最后一段，路径名在
   `open` 时被重新解析。修复：改用 **dir-fd**：root 相对其父目录 `open(O_DIRECTORY|O_NOFOLLOW)`
   打开（不存在则 `mkdirat`），每段父目录用 `openat(O_DIRECTORY|O_NOFOLLOW)` 逐级下钻、
   缺失则 `mkdirat`，文件用 `openat` + `O_CREAT|O_EXCL|O_NOFOLLOW` 创建，写入后从**同一个 fd**
   `fstat`/回读复核；任何一步都不再做路径名二次解析。回归：
   `test_materialize_cannot_be_raced_into_writing_through_a_swapped_parent`（确定性换名竞争，
   断言根外目录为空、字节留在真正被打开的目录），旧实现下该竞争会真实写出根外文件。
2. **无通道守卫破坏终态幂等 replay（确认，已修）**。复现：blob-backed bundle 跑完 `completed`
   后关闭，以无 `blob_store` 的前端重开，同一 request/digest 的 Submit 得到
   `failed_precondition no blob channel`，而不是 completed + 原 receipt。修复：artifact 通道
   检查从 `_validate_submission` 移到「这是新 job」的分支（`not job_exists`）里，既有 job 的
   replay 仍从持久行回答；新 job 仍 fail closed。回归：
   `test_a_terminal_bundle_job_replays_without_the_blob_channel`（completed 与 cancelled 两种
   终态都 replay 出原 receipt/状态，同一前端上新的 blob job 仍被拒且不留 job）。
3. **create/start 前的 fail-closed 仍不全（确认，已修）**。(a) `retry` 只查通道，不查
   ready/descriptor：复现是首次 attempt `interrupted` 后释放并删除 blob，Retry 仍写新 queued
   attempt、worker 退 17。修复：Retry 在写新 attempt 前用持久化 request 重新 `_lease_blob`
   （逐项 ready + descriptor 一致 + 重取 lease），失败即拒且不建 attempt；正常路径仍只回滚本次
   新增的 lease。回归：`test_retry_refuses_a_blob_that_is_no_longer_ready`（job 仍是同一个
   interrupted attempt）。(b) 崩溃留下 `accepted`、尚未 enqueue 的 blob job，无 `blob_store`
   重开时仍会调度并真的 create 出 worker。修复：`_run_worker` 在 `supervisor.create` 前再查一次
   artifact，不可服务就把 attempt 以 `artifact_unavailable` 失败掉，不建进程、不碰设备。回归：
   `test_a_recovered_accepted_bundle_job_never_spawns_a_worker_without_the_channel`（直接建
   accepted 行再重开，断言 `worker_pid is None`、服务无 submission、failure code 正确）。
   第三轮复核补：该 dispatch 守卫当时只查 `blob_store is None`，没有逐项复验 blob。复现：上传
   ready blob → 按真实 Submit 顺序 `_lease_blob` 后直接建 accepted 行（崩溃窗口）→
   `store.release(request_id)` 删除 artifact → 关闭并在**保留 channel** 的前端重开，实际仍
   create 并启动了 worker（`failure.code=worker_failed`、message 为 worker exited 17、
   `worker_pid` 非空）。修复：`_unservable_artifact_error` 通过新增的
   `JobStore.job_blob_scope` 取 job 行的 blob scope，对每个 descriptor 复核 ready、descriptor
   一致、lease 仍属本 request，失败原因写进 `artifact_unavailable` 的 message。回归：
   `test_a_recovered_accepted_job_with_a_deleted_blob_is_refused_at_dispatch` 与
   `test_a_recovered_accepted_job_with_a_quarantined_blob_is_refused_at_dispatch`（都断言
   `worker_pid is None`、服务无 submission、failure 为 `artifact_unavailable`）；新 job 的
   ready/scope/descriptor/lease 校验与终态无 channel replay 语义保持不变。
4. **Go conformance 两处语义缺口（确认，已修）**。(a) `verifyEvidenceCompleteness` 不检查
   `topology.logical_devices` 与 declared 列表长度、不拒重复设备：把主 evidence 的
   `logical_devices` 改成 2（仍只有一个设备/span）后 Python 拒绝而 `go run .` 仍 `ok`。修复：
   Go 增加数量一致与重复设备检查，并把 `logical-devices-disagree-with-the-list` /
   `duplicate-declared-device` 两个负向量放进共享 golden（Python 侧同样拒绝）。(b) `verifyBundle`
   不校验 v4 evidence descriptor 身份：把 unsigned/document 两处 descriptor 一致改成 v1 schema +
   `application/json`、重算 canonical 并用 golden key 重签后，Python 以 identity 拒绝而 Go 仍
   `ok`。修复：Go 固定校验 v2 schema 与 v2 media type，回归
   `test_the_go_verifier_rejects_a_consistently_resigned_version_one_descriptor`（只保留合法
   向量，确保失败原因是 identity 而不是 HMAC 或向量）。
5. **queued/accepted Cancel 的 lease 清理缺口（确认，已修）**。复现：停止 scheduler 后 Submit
   再 Cancel，snapshot 已 `cancelled`，但 `blob_store.leases()` 仍保留 request/blob，直到
   shutdown/restart。修复：`cancel()` 的快路径在终态时释放该 request 的 lease；`revoke_grant`
   的 `cancel_scope` 只返回 running 的那些、其余在 scope 内直接终态，因此补
   `_release_terminal_leases(scope_id=...)` 扫描已终态 job 的 lease。回归：
   `test_cancelling_a_queued_bundle_releases_its_artifact_lease`、
   `test_revoking_a_grant_releases_the_leases_of_the_jobs_it_cancels`（lease 与字节都被释放）。
6. **`reconcile_blobs` 忽略 quarantine 记录（确认，已修）**。复现：孤儿 lease 注入 `remove`
   失败后 `close(wait=5)` 返回 True、`close_error=None`，manifest 实际已是 quarantine。修复：
   逐条检查 `reconcile` 返回的记录，`cleanup=quarantine` 转成显式 hazard（`close_error`）；
   同时把 `live_request_ids()` 的异常也收进 `close_error` 而不是抛出。回归：启动与关闭两条
   `test_a_quarantining_{startup,shutdown}_reconciliation_*`。
7. **post-attach 的致命 store 错误路径（确认，已修）**。复现：attach 后 `job_exists` 因 store
   `mark_unavailable` 抛错，原 `submit` 错误被新的 `unavailable` 掩盖；同时 close 线程在
   `live_request_ids` 抛 `JobError` 时因没有兜底而不关 store、不置 `closed` 事件。修复：存在性
   未知时**不 detach**（保留 lease 交后续 reconcile）并把未验证回滚记进 hazard，原异常原样抛出；
   close 路径对 reconcile 和其它清理一律兜底，无论如何都执行 `store.close()` 并置
   `closed`。回归：`test_an_unknown_store_state_keeps_the_lease_and_the_original_error`、
   `test_close_finishes_even_when_the_store_cannot_list_live_jobs`。

第二轮证据（每条命令单独跑）：`test_remote_npu_bundle_contract.py` **57 passed**、
`test_remote_npu_bundle_grpc.py` **8 passed**、`test_remote_npu_bundle_cleanup.py` **38 passed**、
`test_remote_npu_bundle_go_conformance.py` **18 passed**（合计 126）；P0/P0b 171、旧
gRPC/conformance/evidence 87、job/recovery/container/outbox 265 passed/1 skipped 与本轮一致；
`selfcheck_bridge` 默认围栏 18 failed/1 passed（未改测试）、`TMPDIR=/tmp` 19 passed；
`generate_evaluator_proto.py --check` 通过（未改 proto）；全套安全套件 **2474 passed / 6 skipped
/ 32 failed**，32 项仍是既有围栏环境（`ps` 被禁、AF_UNIX、HOME 权威日志）。第二轮变异核验：
把 artifact 守卫放回 `_validate_submission`（replay 回归转失败）、去掉 Retry 的重新 lease
（retry 回归转失败）、去掉 dispatch 守卫（恢复回归转失败）、去掉 cancel/revoke 的 lease 释放
（两条 lease 回归转失败）、去掉 reconcile 的 quarantine 转 hazard（两条 quarantine 回归转
失败）、把 `job_exists` 改回裸调用（致命 store 回归转失败）、去掉 `live_request_ids` 的兜底
（close 回归转失败）、把 Go 换回上一版（两个 Go 回归分别转失败）。

**P1 复核修复（第四轮，2026-10-29；仍无真实设备）**

第三轮之后又独立复现出 11 类缺口（多数是「同一语义只修了一条路径」），仍全部先写成会失败的
回归再修：

1. **Go 验的是脱离签名字节的另一份 evidence（确认，已修）**。复现：把 `evidence_base64`
   解码后的 `complete` 改成 false，同步更新 descriptor、canonical JSON 并用 golden key 重签，
   分离副本仍保持 true——旧 Go 返回 0。修复：解析签名里的真实 payload，对它执行全部语义检查
   （`verifyBundleEvidence`），并要求它与分离副本 canonical 相等；回归
   `test_the_go_verifier_reads_the_signed_evidence_payload`、`..._payload_with_a_contradictory_aggregate`。
2. **Go 把 `rank:-1`/`die:-1` 当成 null（确认，已修）**。`-1` 原本只是内部哨兵却被当成合法输入；
   Python 以 out-of-bounds 拒绝而 Go 接受。修复：区分 `null` 与整数，负数一律拒绝；共享向量新增
   `graph-only-rank-negative`、`graph-only-die-negative`，以及 `die-only-two-dies`、
   `rank-and-die-two-by-two` 两个合法形状。
3. **`materialize` 把未绑定 FD 的路径名交回调用方（确认，已修）**。单条目真实换名复现：FD 写入的
   正确字节在 `graph-real/main`，但返回的 Path 读到外部 attacker 字节。修复：返回
   `MaterializedEntry`（root + 相对路径 + size/sha），`read_bytes`/`verify` 用 dir-fd 逐段
   `O_NOFOLLOW` 重新读取并复核 digest；`vendor_b` 消费端改用 handle 复核，不再 `is_file()`。
   回归 `test_materialize_handles_cannot_be_talked_into_reading_a_swapped_path`。
   第五轮复核发现该 handle 仍未闭合：`read_bytes` 只按保存的路径重新解析、返回前不校验，把
   `root` 换成普通目录并在原位置放普通攻击者文件后原样返回外部字节（`verify` 才另报错、且是
   第二次读取）。修复：读取与校验合并为同一次原子操作，`read_bytes` 只返回已校验的字节，
   `verify` 复用同一操作；回归 `test_materialize_handles_verify_the_bytes_they_return` 覆盖
   普通文件替换与「同尺寸不同内容」两种变体。同时修掉 `vendor_b` 里 `BundleError` 未导入导致
   校验失败抛 `NameError` 的问题，回归 `test_the_fake_vendor_b_driver_reports_a_verification_failure_as_value_error`。
4. **`entry_from_file` 仍有 check/read TOCTOU（确认，已修）**。lstat 之后按路径 `read_bytes()`，
   换成外部 symlink 后返回并哈希了外部字节。修复：逐组件 dir-fd、最终 `O_NOFOLLOW`，`fstat` 与
   读取都来自同一 FD；回归父目录与最终组件两种换名。
5. **祖先冲突只看精确前缀、没有 Unicode alias（确认，已修）**。`Graph` 与 `graph/child`、组合/分解
   `café` 都能通过 manifest，实际写一半后 EEXIST。修复：`_alias_key`（NFC + casefold）同时用于
   全路径重复与祖先检查，manifest/digest 仍保留原路径；回归在 manifest 期拒绝且零部分输出。
6. **dispatch 守卫只证明 manifest 可读（确认，已修）**。保留 ready manifest、删掉首个 chunk 或改
   一个字节后恢复 accepted job，旧守卫仍 create/start worker（`worker_failed`、pid 非空）。修复：
   在 `supervisor.create` 前用 `iter_chunks` 流式读取并整体 sha256/size 复核；回归缺失与损坏
   两种 chunk 变体，都断言 `artifact_unavailable`、`worker_pid is None`、服务无 submission。
7. **Retry 回滚没有未知提交保护（确认，已修）**。让 `store.retry` 提交新 queued attempt 后再抛错，
   旧实现仍 detach，把 live job 的 lease 删掉。修复：与 Submit 共用
   `_drop_lease_if_uncommitted`（retry 用「是否换了 attempt」判定），只有证明未提交才回滚，未知则
   保留 lease 并记录 hazard；回归 `test_a_retry_that_committed_before_failing_keeps_its_lease`。
8. **仍有 cancelled 状态绕过 lease 清理（确认，已修）**。`_interrupt_row`/`fail_attempt`/
   `recovered_attempt`/cancelled-Retry 都可能把行改成 cancelled，而旧代码只在自己预期的终态上
   释放。修复：新增 `_release_if_terminal`，按**提交后的行**统一决定释放，接在 interrupt、fail、
   finish_cancellation、recover、retry 与 cancel 快路径上；三条回归分别覆盖 interrupt 竞态、
   恢复、cancelled Retry。
9. **清理 hazard 两处丢失（确认，已修）**。(a) 启动 reconcile 用 `attempt=None`，终态 job 的残留
   lease 删除失败只写 `close_error`，job 仍 `quarantined=False`；现按 job 是否存在持久化 hazard。
   (b) `_release_blob_lease` 只捕 `BlobError`，注入 listdir `OSError` 后 Cancel 已终态却裸抛；
   现捕获所有异常并记录 hazard。两条真实故障回归。
10. **leased ready blob 仍按 TTL 过期（确认，已修）**。`ready_ttl_ms` 的语义是「无 lease 的
    committed blob 存活时间」，但读路径不看 lease，`sweep` 却看，于是长期 attempt 在 TTL 后失去
   自己的工件。修复：只在无 lease 时过期；对照回归覆盖 leased（state/iter_chunks/read_chunk 仍可用、
    `sweep` 跳过）与 unleased（仍过期）。未改 P0b 授权或清理语义。
11. **跨 job 回滚回归（补齐）**。新增两个真实 durable job 的控制回归：
   `test_a_rejected_bundle_submit_leaves_another_jobs_lease_and_bytes`——A queued 持 X，B 引用
   X+未 ready Y 被拒后，A 的 job/lease/字节不变且随后仍能完成；它是对「按 request id 释放会误伤
   别的 job」这一字面场景的证伪控制，不能由同 request 的旧 attempt 测试替代。

第四轮证据（每条命令单独跑）：`test_remote_npu_bundle_contract.py` **58**、
`test_remote_npu_bundle_grpc.py` **8**、`test_remote_npu_bundle_cleanup.py` **42**、
`test_remote_npu_bundle_go_conformance.py` **18**（合计 **126**）；P0/P0b **171**（blob_channel
新增 leased-TTL 对照）；旧 gRPC/conformance/evidence 87；job/recovery/container/outbox 265
passed/1 skipped；`selfcheck_bridge` 默认围栏 18 failed/1 passed、`TMPDIR=/tmp` 19 passed；
`generate_evaluator_proto.py --check` 通过（未改 proto）；全套安全套件 **2492 passed / 6 skipped /
32 failed**（32 项仍是既有围栏环境）。变异核验（复原后重跑）：materialize 返回 Path、entry_from_file
改回路径读取、祖先检查改回精确匹配、去掉 dispatch 内容复核、Retry 改回无条件 detach、去掉
interrupt/recover/retry 的终态释放、reconcile 改回 attempt=None、release 只捕 BlobError、过期
判定忽略 lease、Go 换回上一版——对应回归分别转失败。

**P1 复核修复（第六轮，2026-10-29；仍无真实设备）**

e14c758 之后又确认三处同类缺口，仍先复现再修：

1. **回滚失败被静默吞掉（确认，已修）**。复现：ready blob attach 后以无效 capability 让
   `store.submit` 拒绝，并让 `store.detach` 抛 `OSError`——Submit 正确报 `failed_precondition`、
   `job_exists=False`，但 lease 仍在该 request 名下且 `close_error is None`；由于 leased blob
   不过期，这条无 job 的 lease 会一直留着。修复：`_drop_attached_blobs` 对每次失败的 detach 记
   一条显式 hazard（不扩大回滚范围、不掩盖原始 Submit/attach 异常）。回归：
   `test_a_failing_rollback_after_an_attach_error_is_recorded`（attach 循环在半途失败）与
   `test_a_failing_rollback_after_a_store_rejection_is_recorded`（post-attach store 拒绝），
   两者都断言原异常、无 job、lease 仍在、`close_error` 记录回滚失败，并断言随后 shutdown
   reconcile 会释放该 lease 与字节。
2. **默认 `close()` 快路径绕过 reconcile（确认，已修）**。复现：孤儿 lease + `FaultIO` 的
   `remove` 失败，`close()`（无 wait）返回 True、`closed=True`、`close_error=None`，lease 仍在、
   blob 仍 ready；只有 `close(wait=...)` 的后台路径会 reconcile。修复：抽出
   `_finish_shutdown`（兜底 reconcile → `store.close()` → 置 `closed`），快路径与后台路径共用。
   回归 `test_the_default_close_reconciles_an_orphan_lease`（先停 scheduler 确保走快路径）。
3. **Go 未校验 report descriptor 身份（确认，已修）**。复现：把 `unsigned_result` 与 `document`
   两处 report schema 同时改成 v1 evidence schema、重算 canonical 并重签，Go 返回 0，而 Python
   `result_signature_document` 抛 `result report descriptor is not the candidate report schema`。
   修复：Go `verifyBundle` 固定 report descriptor 为 `tcenv.eval-report.v3`（与 Python 同一规则，
   不额外发明 media type 约束）；回归
   `test_the_go_verifier_rejects_a_consistently_resigned_report_schema`（只保留合法向量、断言失败
   原因是 report 身份而不是 HMAC）。

第六轮证据（每条命令单独跑）：`test_remote_npu_bundle_contract.py` **58**、
`test_remote_npu_bundle_grpc.py` **8**、`test_remote_npu_bundle_cleanup.py` **42**、
`test_remote_npu_bundle_go_conformance.py` **18**（合计 **126**）；P0/P0b **171**；旧
gRPC/conformance/evidence 87；job/recovery/container/outbox 265 passed/1 skipped；
`generate_evaluator_proto.py --check` 通过（未改 proto）；安全全套 **2496 passed / 6 skipped /
32 failed**（32 项仍是既有围栏环境）。变异核验：把 detach 异常改回静默 `pass`（两条回滚回归转
失败）、把快路径改回直接 `store.close()`（默认 close 回归转失败）、Go 换回上一版（report 身份
回归转失败）。

修复后证据（每条命令单独跑）：`test_remote_npu_bundle_contract.py` **57 passed**、
`test_remote_npu_bundle_grpc.py` **8 passed**、`test_remote_npu_bundle_cleanup.py` **38 passed**、
`test_remote_npu_bundle_go_conformance.py` **18 passed**（合计 126）；P0/P0b 回归
`test_remote_npu_{contract,grpc,cleanup}.py` +
`test_remote_blob_{channel,contract,grpc,go_conformance}.py` **171 passed**；
`test_remote_grpc_{adapter,tls,contract,revocation}.py` + `test_remote_{conformance,
go_conformance,evidence_contract}.py` **87 passed**；`test_remote_{job_engine,job_recovery,
legacy_worker_bridge,container_engine,container_supervisor,runtime}.py` +
`test_evaluation_outbox.py` **265 passed, 1 skipped**；`test_selfcheck_bridge.py` 在默认围栏
TMPDIR 下 **18 failed / 1 passed**（AF_UNIX 路径超长，未改测试），`TMPDIR=/tmp` **19 passed**；
`scripts/generate_evaluator_proto.py --check` 通过（未改 proto）；全套安全套件 **2474 passed /
6 skipped / 32 failed**（32 项仍是既有围栏环境）。

剩余风险（不得据此宣称）：v1–v3 的 evidence descriptor media type 与旧 golden 仍不一致（冻结，
不修）；per-device span 的**设备 id 命名**仍由 vendor 自报、frontend 只核对数量与坐标，没有
profile 侧的物理 id 绑定；`declared_devices` 与 profile 的 rank/die 事实只核对「有没有该维度 +
数量」，没有逐 die 的拓扑摘要绑定；P2a/真实设备、真实 rank/die 聚合、真实 SDK 计时仍全部未做。

### P1c：统一 ArtifactBuilder 与 control_host 工件交接（无真实 NPU，使用真实 Mac/Metal）

位置：P1 之后、P2a 之前。用统一 ArtifactBuilder 把 local/remote 构建与传输/执行分开，
再用小文件证明 control_host 预编译工件可以进入 NPU 评测。此阶段不访问真实 NPU，也不要求
在 NPU 服务器上安装编译器、git 或 LLM 凭据；但会使用真实 Mac/Metal 编译与 pipeline，
检验 backend-native artifact 抽象，不能用 fake Metal 代替。

**实现与映射**

- 实现 `ArtifactBuilder` 的 probe/build/BuildOutcome 与 deadline/错误分类，构建过程只接受
  登记的 build_profile/toolchain；source digest、builder/toolchain/profile 与实际 artifact
  字节都进入 BuildProvenance/BuildEvidence，并接入 canonical request、outbox、签名和 resume。
- **Metal 必须是具体生产实现，不是概念映射**：在 `tcenv/backends/metal/` 实现
  `MetalSourceBuilder`，probe 返回真实 macOS/Metal runtime/device 能力与版本；build 复用
  现有 `MTLDevice.newLibraryWithSource_options_error_` 的 MSL 编译和 pipeline 创建路径，
  返回 backend-native pipeline handle + manifest + BuildProvenance + BuildEvidence。旧
  candidate_sha256、compile/pipeline 报告字段、缓存、执行与发布语义不变，不改走 NPU
  contract，不把 pipeline 强行序列化成文件。若现有 ArtifactBundle 只支持文件，按 §4.2.1
  的 backend-native payload 扩展中立模型，而不是让 Metal 迁就文件模型。
- **真实 Mac 验收**：有效 MSL 实际编译并创建 pipeline，无效 MSL 映射为 compile failure；
  核对真实 source digest、builder/runtime/device 版本、BuildEvidence，以及与旧路径相同的
  行为和错误分类。可做小规模正确性验证，但不跑性能 benchmark、不做发布加速比、不跑完整
  GPU suite。真实 Mac/Metal 不可用或权限不足必须标 blocked，不能用 fake 代替。
- TSM 主路径使用 control_host 的 TsmModuleBuilder：小 source → 小 ELF/module →
  ArtifactBundle/manifest → BlobChannel → fake NPU VendorDriver。NPU 侧只校验并加载，
  不拉源码、不调用编译器。
- 另测 evaluator 端构建：source 经六 RPC 提交后由 evaluator 内 ArtifactBuilder 构建并
  进入 VendorDriver；两种时序的 ArtifactBundle、BuildProvenance、BuildEvidence 和错误
  分类一致。

**模拟验收**

| 场景 | 必须证明 |
| --- | --- |
| control_host 预编译 | build 成功后才 Submit；NPU 端收到的 ELF 与本地构建字节一致，无编译器/git/LLM 依赖 |
| evaluator 端构建 | source 在同一 attempt 内构建并执行，digest/provenance/evidence 与预编译路径语义一致 |
| 构建失败 | 预编译失败不 Submit；attempt 内构建失败映射为 compile failure，不伪装成基础设施 timeout 或 ok |
| 工具链/profile 不匹配 | builder/probe 返回与 profile 不符时 Submit/create/start 前拒绝 |
| 工件篡改 | 修改 manifest、ELF 字节、blob digest、builder 版本或 BuildProvenance 任一环节均被拒 |
| 中断与恢复 | 构建/上传中断后续传与 resume 不重复构建、不混用旧 artifact；旧 attempt/fence 结果被拒 |
| 清理 | 构建临时文件、blob 引用与设备 module 都按 job 生命周期清理；无法证明时 quarantine |
| 具体 MetalSourceBuilder | 真实 Mac 上有效 MSL 编译+pipeline、无效 MSL compile failure、真实 provenance/evidence；旧 Metal/local 编译、缓存、报告和发布语义不变 |

验收必须包含真实六 RPC 与 BlobChannel，而不是只测 builder 函数；编译日志、artifact
manifest、BuildProvenance 与 fake NPU 的加载记录形成闭环。小文件只证明抽象和状态机，
不证明真实 TX 编译器 ABI、跨版本兼容、真实 NPU 加载或构建性能。

### P2a：TSM 模拟验收（无真实设备）

位置：P1 之后、P2 之前。近期没有清微设备，因此先假设设备存在，用 TSM simulator 模拟
runtime/SMI/profiler 与设备资源状态，验证 GetCapabilities → Submit → Watch/GetJob →
Cancel/Retry、attempt/fence、超时/取消/恢复与证据闭环。P2a 只覆盖通信、状态机与清理逻辑，
不验证真实 NPU 计算、驱动 ABI、容器设备映射或硬件性能。它是 P2 的前置模拟，**不替代**真实
P2；通过后也不得把任何 capability 写成“TSM 已支持”或“可发布”。

P0/P1 的 fake 在 worker 内直接给报告；P2a 的 sim 位于 `VendorDriver` 之下，也就是厂商 SDK
边界。P2a 要实现真正的 TSM adapter（`prepare`/`execute`/`close`、资源登记、错误映射、
evidence 映射），只把 SDK 调用与设备状态换成 sim；其上的 `VendorDriver` 骨架、
`DeviceController`、gRPC 六 RPC、`JobEngine`/`JobStore`、`AttemptSupervisor`、worker 进程、
签名/outbox/resume 都是真实代码。P2 把 sim 绑定换成真实 SDK 绑定并重新验证；若真实 SDK
迫使改变上层语义，按设计变更处理，不静默改写。sim 不是返回固定 JSON 的假函数：它维护
句柄、设备时间线、资源台账和故障状态机，未知或越界调用一律 typed 失败。

**实现边界**

- `tests/support/tsm_sim/`（规划，当前不存在）：`runtime/`（module/function/launch/
  stream/event/IPC）、`smi/`（probe/health/topology/reset/discovery）、`profiler/`（Host 与
  Device span、correlation、双 buffer、flush）、`ledger/`（attempt 资源账本）、`faults/`、
  `clock.py`（虚拟时钟）、`known_good/`（由 control_host ArtifactBuilder 产出的 module
  字节 + manifest + digest + BuildProvenance）、`facade.py`
  （对 driver 暴露的 typed 协议）。另设 `host/`（共享 runtime/设备组状态、lease 与 reset）
  和 `customers/`（A/B 两套独立 Docker 软件环境、镜像/文件视图与入口）；两个客户入口和
  host 入口必须映射到同一份 host 设备状态，不得在 sim 里虚拟成三张独立 NPU。
- 注入点只能是测试 composition root：复用 P0/P1 的 test-only entry 骨架与
  `tests/support/npu_sim/`、`tests/support/remote_pki.py`、`tests/support/remotegrpc/`；生产
  组合根不得 import `tests.support.tsm_sim`，由 import/结构检查断言，而不是靠约定。
- 不允许为迁就 sim 修改生产错误分类、放宽校验或加测试专用分支。错误映射测试以生产分类为
  准，sim 只产生厂商错误码与原始诊断（脱敏、有界）。
- sim 不新增厂商专用 RPC、不新增任意 vendor key-value、不接受路径/命令/BDF/rank。
- 测试文件按现有分层放 `tests/shared/test_tsm_sim_*.py`：driver 不是 backend，与
  `tests/shared/test_remote_*.py` 同类；sim 本体仍在 `tests/support/`。

**状态机覆盖**

1. module：`txModuleLoad` → `txModuleGetFunction` → `txModuleUnload`；重复 load、卸载后
   launch、失效句柄必须被拒。
2. launch：入队立即返回，设备时间线独立推进；完成只能经 `txStreamSynchronize`、
   `txEventSynchronize` 或 `txEventQuery` 观察。资料中 `txDeviceSynchronize` 仍标为未来实现，
   sim 不提供这条捷径。
3. stream/event：create/destroy/sync/query 与 event record/elapsed/query 的完整状态迁移；
   跨 attempt 复用必须被 attempt/fence 绑定拒绝。
4. IPC：`txIpcGetMemHandle`/`txIpcOpenMemHandle`/`txIpcCloseMemHandle` 计入台账；open 未
   close 即泄漏。
5. 资源：module/function/stream/event/IPC handle/device memory/profiler buffer 都带
   `attempt_id`、`lease_fence`、`generation` 与创建/销毁时间；`close` 只释放本 attempt 的。
6. probe：报告 driver/firmware/runtime/SDK/profiler/OS 版本、device/die/rank 形状与
   topology digest，供 `capabilities` 判定可用性。
7. worker 已死、设备仍 running：sim 保留设备侧状态；`DeviceController.reconcile` 必须按
   attempt/fence 核对并给出 `CleanupProof`；不能证明就 quarantine，不能报告自动清理成功。
8. 客户环境/host：A/B 各自 Docker 有独立镜像、runtime 配置和文件视图，但 host 设备组、
   driver/firmware 与 lease 共享；host 与 Docker 两种入口都能看到同一设备时间线。A 的容器
   删除只结束 A 的 worker，host 资源与 B 的运行状态不被误删；B 不能在 A 持有排他 lease 时
   启动另一路正式计时。reset 对 A/B 与 host 入口的所有 in-flight attempt 生效，不能伪装成
   单容器 reset。

**故障注入**

每个注入都必须先证明故障真的发生，再断言结果：fault 提供 `assert_fired()`，注入未触发即
测试失败，不允许只把错误字符串写进预期。

| 注入 | 位置 | 先断言的触发证据 | 期望 |
| --- | --- | --- | --- |
| AP/SDK hang | `prepare` 内加载或 launch 前 | 调用记录有 entered、无 returned | JobEngine deadline 到期，基础设施 timeout；不自动重跑 |
| launch 后 hang | launch 已返回、event 永不 signal | 状态含 `launch_started` 与 `device_running` | 设备侧超时；Host elapsed 不进入 device elapsed |
| worker SIGKILL | execute（prepare/close 另各一例） | execute：kill 前已有 `launch_started`，kill 后、reconcile 前仍能读到 `device_running` 与 live 资源 | reconcile 成功或 quarantine，二者必居其一且附证据 |
| C2C down | 执行中对端链路 up→down | 链路状态确实为 down，执行已开始 | 基础设施错误；不映射成候选 crash/timeout/ok |
| discovery 失败 | probe / capability 计算 | discovery 调用已执行并失败 | capabilities 不可用；Submit 在 create/start 前拒绝 |
| 版本或 profile 不匹配 | probe 返回值 vs profile | 逐项比对失败（driver/firmware/runtime/profiler） | capabilities 不可用；Submit 在 create/start 前拒绝；不记候选错误 |
| 拓扑代际变化 | attempt 中途 generation++ | 旧与新 generation 同时可观察 | 旧 capability/fence 的提交与结果被拒；影响测量则新 epoch |
| IPC/stream/module 泄漏 | close 吞掉 close/destroy/unload | close 后 ledger live 计数仍 > 0 | `CleanupProof` 不完整 → quarantine |
| 缺必需 span | profiler 已记录设备完成与 correlation | `device_completed` 与 correlation 存在，但必需 span 查询为空 | 证据不完整，只作诊断，不发布 |
| buffer 实际丢弃/覆盖 | Buffer-HW 半区或 Buffer-CPU ring | drop/overwrite 计数 > 0，或记录序号出现缺口 | 证据不完整，只作诊断，不发布 |
| flush 记录未完整交付 | flush 已触发、记录投递 | flush 调用已记录，但交付记录少于已写记录 | 证据不完整，只作诊断，不发布 |
| 健康探针失败 | known-good module 在健康设备上失败 | 探针确实被执行 | 设备组 quarantine，暂停接新 job |
| reset 的全设备影响 | 运行中 reset | 占用检查与 generation++ 均被记录 | 有占用时拒绝 reset；reset 后 in-flight 结果作废，无单卡/单 job 语义 |
| 客户 A 容器消失、B 仍在运行 | A 有未完成 host 资源，B 有独立 attempt | A 的容器已删除，A 的 host 资源仍 live，B 的设备时间线未改变 | 只 reconcile A 的资源；不得误删 B 或宣称整组清理完成 |
| 两客户争用同一 host 设备组 | A 持排他计时 lease，B 从自己的 Docker 提交 | B 确实使用不同客户环境，但 host group 相同 | B 排队/按公平策略执行，不能并行取得同一计时 lease |

worker SIGKILL 的 execute 一例必须先证明设备仍在执行：kill 前已有 `launch_started`，kill 后、
reconcile 前仍读到 `device_running` 与未释放资源；prepare/close 阶段另各一例，只要求 pid 消失
且台账仍有 live 资源。SIGKILL 用两个变体覆盖两种结论：资源仍可按 attempt/fence 归属时，
reconcile 必须成功，`CleanupProof` 完整且台账清零；归属或证明被破坏时，只能 quarantine。
这样既不会始终 quarantine 空过，删除 reconcile 也必定让成功变体失败。

**计时与发布**

- `host_api_elapsed` 与 `device_elapsed` 是分开的字段；Host 只作诊断，不能写成设备耗时。
- sim 的“完整证据”= 同步完成 + event 或 profiler span + 声明拓扑要求的 rank/die span +
  flush 完成 + 时钟域已知；缺 rank/die（仅当 profile 声明需要）、缺 span、丢缓冲、flush
  不完整、时钟域未知，任一项都只能诊断，不得发布。
- 所有 P2a 结果固定 `simulated=true`、`publication_allowed=false`：不进入 confirmed/catalog/
  brief，不产生性能数字，不绑定确认门。

**安全负例**

- agent 请求不能携带 `tsm-shell`、MPI、`fabric_manager`、root discovery、任意路径、裸
  BDF/rank 或任意 vendor 参数；三层断言：schema 拒绝、策略在 create/start 前拒绝、sim
  台账中这类调用为零。
- 不能借 simulator 放行 privileged、host 凭据或 host HOME；测试 composition root 不放松
  容器策略，sim 拒绝来自 env 的提权开关。客户 B 默认也看不到客户 A 的私有文件、attempt
  目录或凭据，只有 profile 明确允许的 host 共享路径/IPC 才能访问；“同组设备可共享”不能被
  解释成“客户文件可互读”。
- sim 接口只接受 typed 值：越界路径、root 操作、未知 API 一律 typed 失败，不得默默成功。

**可证伪性**

| 变异 | 必须失败的测试 |
| --- | --- |
| 删除 `reconcile` | worker 被杀后的资源对账 |
| `close` 跳过 cleanup | 泄漏检测与 `CleanupProof` |
| 用 Host elapsed 充当 device elapsed | Host/Device 分离 |
| 接受旧 fence 的提交或旧 fence 的结果 | fence 与恢复 |
| 跨 attempt 复用 module/stream | 资源归属 |
| 单卡/单 job 语义做 reset | reset 全设备影响 |
| 证据缺 span 仍发布 | 发布门 |

**模拟无法证明**

- 真实 TX SDK/driver/firmware ABI、真实错误码取值与 AP 内部调度；
- 容器 `/dev/accel_drv_mgr`、`/dev/accel/dev-*`、host-hal、`--network=host` 与 `/tmp`、
  `/sys`、`/lib/modules` 挂载是否真的可用；每客户 Docker 的真实文件/IPC 共享机制、跨客户
  隔离与 host 最终执行路径也不能由 sim 证明；
- 物理机 root discovery、fabric_manager、真实 C2C 链路与 MPI 绑定；
- `tsm_smi --reset` 的真实全设备影响、驱动卸载/重装与同站其他 job 误伤；
- 真实设备内存/句柄是否真的释放（sim 只证明台账与调用序列自洽）；
- 真实时钟域、event 精度、raPTI 的 intercept queue、双 buffer、flush 与丢缓冲行为；
- source→TX ELF 的编译器/ABI/正确性，以及硬件性能与 profiler 开销。

因此真实 P2 必须重新验证以上全部；P2a 通过只说明抽象、状态机、清理与证据判定自洽。

**测试清单与验收**

| 组 | 内容 | 来源 | 关键断言 |
| --- | --- | --- | --- |
| G1 注入契约 | sim 只在测试根注入；未知 API typed 失败；无新 RPC/字段 | P2a 新增（复用 P0/P1 contract 版本化） | 生产根 import 检查为空；proto 无新增 RPC |
| G2 module/launch/stream/event | 句柄生命周期基本闭环 | P2a 新增 | 校验顺序与失效句柄拒绝 |
| G3 Host 先返回 | 异步完成语义 | P2a 新增（P1 已单列 Host launch） | Host 完成不等于执行完成 |
| G4 资源归属 | module/stream/IPC/profiler buffer 账本 | P2a 新增 | close 只释放本 attempt；泄漏可检出 |
| G5 reconcile | 三阶段 SIGKILL；execute 路径先证 `launch_started` 与 kill 后 `device_running` | P2a 新增（复用 supervisor kill/absence） | 可证明：`CleanupProof` 完整且台账清零；不可证明：quarantine 且无自动清理声明 |
| G6 topology/version | probe、generation、epoch | P1 已有 topology digest | 不匹配即不可用；代际变化拒旧 fence |
| G7 Cancel/Retry/fence | 取消与 interrupted 恢复 | P0/P1 已覆盖 JobEngine 语义 | P2a 只补设备资源随取消/重试释放 |
| G8 计时与证据 | event/profiler/rank-die；缺 span、丢缓冲、flush 不全三类注入分别判定 | P1 已有完整性判定 | 缺项只作诊断；`simulated=true` |
| G9 安全负例 | tsm-shell/MPI/fabric/root/路径/BDF/rank/vendor 参数 | P0/P1 已有 shell/path/feature 负例 | 三层拒绝且台账零调用 |
| G10 conformance 与回归 | Python/Go conformance、Metal/local | P1 已建立 | 旧 contract 与旧验证不变 |
| G11 客户 Docker/host 共享 | A/B 独立软件环境、同一 host 设备组；host 与 Docker 两入口 | P2a 新增 | 同一设备时间线；排他 lease 互斥；删 A 容器不误删 B；reset 影响所有客户 |

G7、G9 与 P0/P1 重叠的部分直接复用，不重复建设：P2a 只加设备侧资源、异步完成与证据部分。

验收条件：

1. G1–G11 离线通过，不需要 SDK、GPU/NPU 或真实 provider，可带 `-m 'not gpu and not llm'`；
   虚拟时钟，无长 sleep 轮询，结果确定。
2. 每条故障注入先证明触发再断言结果；变异表逐条能复现失败。
3. 跑完 P2a 后 confirmed/catalog/brief 与基线一致，没有新增性能数字。
4. G10 回归与 conformance：以下三条命令通过，且旧 contract/旧验证未改：

   ```bash
   uv run pytest -q -n 0 tests/shared/test_remote_conformance.py tests/shared/test_remote_go_conformance.py tests/shared/test_remote_grpc_tls.py
   uv run pytest -q -m 'not gpu and not llm and not slow'
   uv run pytest -q -n 0 tests/shared/test_structure.py
   ```
5. sim 调用轨迹、状态转换、台账与故障计数留在 pytest tmp，不新建 review/report 文档。

### P2：清微单节点通信试点（另行授权真实设备）

P2a 的模拟结果不构成本阶段证据：下面每一项仍须在真实设备与真实 driver/firmware/runtime 上
重新验证。

先用厂商提供的 known-good executable module，不先假设 Triton 源码工具链已通：

- provision 核验 driver/firmware/runtime、设备映射、拓扑与任务包；
- 在 control_host 用登记的 TsmModuleBuilder 构建 known-good module，记录工具链/SDK、source
  digest、BuildProvenance，并经 BlobChannel 上传；NPU 端只复核工件与构建证据，不重新编译；
- 在 host 与每客户 Docker 两种入口分别执行 known-good 流程，确认它们指向同一 host 设备组；
  两个客户环境争用排他计时 lease 时不能并行，删除一个客户容器不能误清另一个客户的资源；
- GetCapabilities → Submit → Watch/GetJob；Cancel 与 Retry 分别覆盖显式取消与 interrupted
  attempt 的新 capability/fence 恢复；
- load/launch/sync、超时/取消清理、旧 fence 拒绝、异常后 known-good 健康探针；
- worker 被杀后由 worker 外 DeviceController 按 attempt/fence 完成设备侧 reconcile；若无法
  证明模块/stream/IPC/profiler 已释放，只能 quarantine + 人工恢复，不算自动清理通过；
- 验证 agent 无法触发 tsm-shell、MPI、fabric_manager、root discovery 或任意路径。

这一阶段只证明通信与生命周期，不做性能发布。随后再补 source→module 的编译器/ABI/参考
实现契约，不能让通信试点冒充 kernel 任务已适配。

### P3：清微计时与多 die/card

先 device event。raPTI/profiler 不按 HLD/LLD 整体放行，而按 `tx82_profiler_hld` §3 的能力
逐项设门：HLD Phase 1 Host/Device 时间（LLD 的 Phase 1 队列层部分已做，raPTI 日志样例可见
Host/API 与 device timing）、HLD Phase 2 Intercept Queue、HLD Phase 3 双 buffer、HLD Phase 4
memory 生命周期。目标版本上某项能力可用，且 `TimingEvidence` 能核验 correlation/per-die/launch
汇总/flush/丢缓冲后，才开启该项；厂商日志样例或 HLD/LLD 描述不等于本项目验收。未核验项
保持 blocked，P3 仍可只用 device event 独立验收。之后扩到多 die/card，按声明拓扑保留所需
device/rank/die 证据，并验证租约互斥、C2C 故障、取消/恢复与 reset 作用域。

### P4：第二家真实厂商与多机扩展

第二家必须选与清微形态不同者（例如 graph-only、异步服务式或无 rank/die 的厂商），完整跑同一 conformance
矩阵。只有它也能复用控制面、artifact/error/timing 语义且没有厂商专用 RPC 漏到 agent，才说明
抽象成立。多机 rank/MPI/fabric 最后接入，仍保持站点内封装与 single-active lease，不宣称 HA。

## 9. 跨厂商验收矩阵

每一家都必须逐项提供证据，不能以“另一个厂商已过”豁免。P2a 的模拟结果（§8）不能填任何
一行：`simulated=true` 只说明抽象、状态机与清理逻辑自洽。

| 维度 | 必须证明 |
| --- | --- |
| 能力 | vendor/device group/版本/拓扑真实；不支持的能力在 create/start 前拒绝 |
| 工件 | 所有角色、大小、digest、路径、bundle root 与 receipt 闭环；大工件无隐式本地路径 |
| 执行 | 只走 typed load/launch/sync；agent 无 shell、MPI、裸地址或任意 vendor 参数 |
| 故障 | timeout/cancel/驱动异常/C2C 或厂商对应故障分类正确；不伪装成候选 ok |
| 清理 | 模块/graph/stream/IPC/profiler/临时文件/worker 都有清理证明；失败即 quarantine |
| 恢复 | 不自动重跑；旧 attempt/fence 结果被拒；Retry 使用新 capability 与 fence |
| 计时 | 设备侧来源、同步/flush/完整性、calibration 与声明拓扑所需证据齐全；Host EXIT 不冒充设备时间 |
| 安全 | 最小设备映射、无 agent 凭据、无 root SSH、日志脱敏、数据保留与第三方授权 |
| 兼容 | Python/Go conformance 与既有 Metal/local/gRPC 回归通过；无 vendor 专用外部 API |
| 发布 | simulated 不可发布；真实 measured 仍须签名、site/epoch 与确认门，不跨厂商混缓存/混 epoch |

## 10. 接入清微前必须回答的问题

1. Triton/其他源码到 TX ELF/bin 的正式编译器、ABI、grid/block/cluster 契约、正确性参考实现
   与容差是什么？control_host 的 BuildProfile、工具链/SDK 版本、source digest 与
   BuildProvenance 如何登记并传入结果证据？没有这些契约，只能先做 known-good module 通信试点。
2. serial/UUID 等稳定硬件身份如何取得？user/mesh/soft/slot 的映射在重启、换卡、容器重建后
   如何复核？
3. frontend/worker/DeviceController/host-hal 的实际部署边界是什么？每客户 Docker 与 host
   共享哪些文件/IPC，最终由哪个 host 组件执行？`/dev/accel_drv_mgr` 与 dev-* 各自必要吗？
   host network、`/tmp`、`/sys`、`/lib/modules` 哪些只需只读，哪些确实不需要？
4. 物理机 root discovery 能否由设备方 provision 一次完成？组网变化如何通知 frontend？
   免密 root SSH 能否被最小权限的受管替代完全消除？
5. reset 的真实作用域、已占用设备处理与 known-good 探针内容是什么？全卡 reset 如何防止
   误伤同站其他 job？
6. device event 与 raPTI 的版本、开销、时钟域、flush/丢缓冲语义是什么？多 rank 如何定义
   完整测量与汇总？
7. graph/model/BPM 的正式包格式、签名/兼容校验、大小与外部 blob 需求是什么？

这些答案未落实前，不能把 TSM 适配写成“已支持”，更不能产生性能或发布结论。

## 11. 非目标

- 不在本计划中实现任何厂商 SDK、安装固件/驱动、访问真实 NPU 或重写 kernel 编译器。
- 不统一不同厂商的 kernel ABI、graph 格式或容差；这些仍属于 Backend/task contract。
- 不引入全局多租户调度、跨 frontend HA、自动 failover 或跨厂商统一性能排名。
- 不把 tsm-shell、免密 root、MPI discovery 或设备 passthrough 当成 agent 可请求的能力。
- 不以 fake vendor、known-good module 或 Host wall time 的结果宣称真实 NPU 性能或发布许可。
