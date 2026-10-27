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

外部 API 不新增 `TsmSubmit`、`VendorBSubmit` 之类的厂商专用入口；agent 也不能指定任意
vendor shell、路径、MPI 命令或裸设备地址。**通信统一的是请求、能力、工件、结果、错误与
计时证据，不是假装所有 NPU 都有相同的 kernel ABI。**

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
  参数表及辅助文件，也没有大工件传输通道。
- `Backend` 负责源码/语言/launch 规则，`AttemptSupervisor` 负责 worker 生命周期；在
  Backend/worker 与设备之间还缺一个明确的**厂商设备驱动层**，不能直接往 gRPC server 或
  Docker supervisor 里塞 SDK。
- `EvalReport` 的标量耗时不足以解释多 rank、profiler 缓冲、时钟域与证据缺失；厂商错误也
  不能只靠字符串归类。

## 3. 目标架构

```text
control_host
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
```

- control_host 不需要安装任何厂商 SDK，也不直接接触 `/dev/*`、卡侧 AP 或 fabric。
- 每个逻辑设备组仍对应一个 single-active frontend、一份持久状态和一个 lease owner；
  多设备/多厂商使用独立 frontend，不在 v1 暗做全局调度或 HA。
- `Backend` 继续负责源码、语言、参考实现、任务指纹和 launch 校验；`VendorDriver` 负责
  infra；`AttemptSupervisor` 只负责 worker 生命周期，三者不互相吞并。

建议落点：

- `tcenv/backends/<backend_id>/`：源码/契约/参考/评测语义。
- `tcenv/remote/drivers/<vendor_id>/`：厂商 infra adapter。
- `backends/<id>/`、`targets/<id>/`：公开元数据、skills、芯片/设备 profile。
- `deploy/vendors/<vendor_id>/`：受信镜像、依赖清单、provision/health 脚本与部署 profile。
- `tests/support/vendor_sim/`：至少两种不同形态的 fake vendor 与 golden；不进入生产组合根。
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

首版只允许有界 inline bundle。大模型/大图需要后续独立的 content-addressed blob 通道
（上传前声明 size/digest、落盘复核、receipt 绑定、配额与清理）；不得以宿主路径、共享目录
或 `source` 伪装大工件传输。

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
   视图与网络策略。
2. **受信 host runtime**：仅当厂商明确不支持容器时另做生产级 supervisor；当前 process
   supervisor 是本地开发路径，不能直接宣称满足 NPU 生产隔离。
3. **厂商专用受信 RPC**：若设备只提供远程服务，新增的是共享受信设备控制适配层的内部
   传输，worker 与 DeviceController 都只能经 typed 接口使用，凭据留在控制区；否则外部控制器
   无法取消和对账异步服务。本站 JobEngine 仍负责 attempt、清理与证据，不把第三方服务直接
   当成本项目 trusted evaluator。

## 6. 清微 TSM 的映射方案

| 清微能力 | 适配位置与规则 |
| --- | --- |
| `txModuleLoad`/ELF/bin | `ArtifactCodec` 校验 executable module 的 manifest/digest；厂商 runtime 仍做模块签名校验。来源编译另属 Backend/toolchain，不能假定 Metal/Triton 源码天然兼容 |
| SIMT/Cluster launch | `LaunchAdapter` 把公共 `KERNEL` 的受信 profile 子类型（SIMT/Cluster）映射到 `txLaunchKernel*`；grid/block/cluster 参数来自已校验 ExecutionSpec，AP 调度细节不外露 |
| graph/model/BPM | 映射到 `GRAPH_DEFINITION`/`MODEL_DEFINITION` role 与 `GRAPH`/`MODEL` typed launch；`txLoadGraph` 的 case 目录由 driver 从已校验 bundle 在 attempt 作用域内物化，agent 不提供路径；BPM 使用公共 `PARAMETER_TABLE`，不增加任意路径字段 |
| stream/event | 基础同步与 `DEVICE_EVENT` 计时；stream 创建/同步/销毁纳入 cleanup 证明 |
| raPTI/profiler | `PROFILER_DISPATCH` 计时适配，保留 launch correlation id、per-die 与全 die 汇总、flush/完整性；Host EXIT 不当设备时间 |
| `/dev/accel*` 与 host-hal | 受信 profile 精确映射所需节点；默认非特权。厂商非特权 recipe 仍带 `--network=host` 与 `/tmp`、`/sys`、`/lib/modules` 挂载，host network 与这些宿主挂载都是显式例外，逐项最小化、只读优先并单独验收 |
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
- retained envelope 必须可认证：NPU outbox 强制要求 host 结果签名 key（缺失则构造
  `ValueError`），complete 与 replay 都要求 `result.signature` 并用该 key 做 HMAC 验证，
  覆盖 report/provenance/evidence/receipt 全字段。因此“伪造 attempt/fence 后用公开算法
  重算合法 `receipt_id`”“把 evidence 提成 measured/可发布”“改写 report 或 provenance”
  以及换 key/撤销 key 后重放都被拒；无 key 的 outbox 不再能保留 NPU 结果。旧 Metal 的
  host digest 与可选签名语义不受影响。
- 设备组绑定：`device_group_id/vendor_id/profile_digest/topology_digest/resource_generation`
  进入 capability（target.device_groups + provenance）、request digest、receipt、provenance
  与 Retry（Retry capability 的 device group 必须与 job 记录一致，串组即
  `failed_precondition`）。attempt/lease 把接受时的完整绑定持久化到
  `attempts.device_binding_json`（store schema v4），`CleanupProof` 携带同一组
  vendor/profile/topology/generation；cancel/reconcile 只对“该 lease 接受时的绑定”生效，
  重启换组时既不向新组 controller 发取消，也不能把旧 attempt 标成 interrupted，只能
  quarantine。
- 设备事实进入 capability：`GetCapabilities` 先用 worker 外 controller 做有界
  `probe`/`health_check`/`capabilities`（`device_call_timeout_s`，默认 2s），逐项核对
  vendor/profile/topology/resource generation 与 P0 执行形状；探测失败、超时、不健康或
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
  device event。
- 验证证据（每条命令各自单独跑）：
  `tests/shared/test_remote_npu_{contract,grpc,cleanup}.py` **48 passed**
  （含 v3→v4 迁移/备份恢复、wire receipt 与 host HMAC 篡改、时钟/候选时间自相矛盾、
  hanging cancel/reconcile、终态 request/provenance/attempt 绑定篡改回归）；
  `test_remote_grpc_{adapter,tls,contract,revocation}.py` 与
  `test_remote_{conformance,go_conformance}.py` 六个旧文件 **62 passed**；
  `test_orchestrator_runtime.py` **18 passed**（含 NPU host 接线回归）；
  `test_remote_{job_engine,job_recovery,legacy_worker_bridge,container_engine,
  container_supervisor,runtime}.py` + `test_evaluation_outbox.py` **265 passed, 1 skipped**
  （跳过项是 Linux-only 的 `PF_KTHREAD` 检查）；`test_selfcheck_bridge.py` 在本机围栏里
  **18 failed, 1 passed**（全部是短 TMPDIR 导致的 AF_UNIX 路径超长，非围栏环境应全通过）。
  `uv run pytest -q -m 'not gpu and not llm and not slow'`：**2249 passed / 6 skipped /
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
  `clock.py`（虚拟时钟）、`known_good/`（module 字节 + manifest + digest）、`facade.py`
  （对 driver 暴露的 typed 协议）。
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
  容器策略，sim 拒绝来自 env 的提权开关。
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
  `/sys`、`/lib/modules` 挂载是否真的可用；
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

G7、G9 与 P0/P1 重叠的部分直接复用，不重复建设：P2a 只加设备侧资源、异步完成与证据部分。

验收条件：

1. G1–G9 离线通过，不需要 SDK、GPU/NPU 或真实 provider，可带 `-m 'not gpu and not llm'`；
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
   与容差是什么？没有这份契约，只能先做 known-good module 通信试点。
2. serial/UUID 等稳定硬件身份如何取得？user/mesh/soft/slot 的映射在重启、换卡、容器重建后
   如何复核？
3. frontend/worker/DeviceController/host-hal 的实际部署边界是什么？`/dev/accel_drv_mgr` 与
   dev-* 各自必要吗？host network、`/tmp`、`/sys`、`/lib/modules` 哪些只需只读，哪些确实
   不需要？
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
