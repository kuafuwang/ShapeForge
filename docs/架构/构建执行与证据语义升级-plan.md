# 构建、执行与证据语义升级计划

状态：设计计划，2026-10-28。独立于 `多厂商NPU评测通信抽象-plan.md`：那份文档负责
通信与设备接入，本文负责**构建入口、构建产物、执行结果、缓存和发布数据的语义演进**。

基本原则：**旧数据只读保留，未来数据按新语义生成和使用；不回填、不改写历史，也不为了
兼容历史而保留两条生产构建路径。**

## 0. 为什么单独做这份计划

当前 Metal 的构建、执行、缓存和发布语义是逐步形成的，存在三类问题：

- `backend.compile()` 与新的 `ArtifactBuilder` 概念并存，`MetalSourceBuilder` 如果只被测试
  直接实例化而没有进入 `tcenv.evaluate`，就不是实际生产实现；
- candidate、构建环境、构建产物和执行环境的身份混在一起，容易让“同 source、不同工具链”
  错误共享缓存或被拿去比较；
- compile/pipeline 报告、执行证据和发布 provenance 对多后端而言信息不足，但直接替换旧
  字段又会破坏历史证据。

本计划不要求一次性统一所有厂商的 ABI，也不要求重写旧 run。它先建立新语义和新数据版本，
再用真实 Mac/Metal 与 fake vendor 验证。

## 1. 目标与非目标

### 目标

1. 生产构建只有一个入口：`backend.artifact_builder()`，evaluator 不得绕过它直接调用
   `MetalContext.compile` 或其他 backend-specific 编译入口。
2. 明确区分四种身份：source、candidate build profile、artifact、execution environment。
3. 新版本数据能表达构建过程、构建产物、执行证据和发布身份，且比旧数据更适合多 backend。
4. 旧数据只读保留；新数据必须有明确 schema/document 版本，不能被旧消费者误读。
5. 真实 Mac/Metal 成为新构建接口的实际实现和验收，而不是测试专用概念类。

### 非目标

- 不统一各厂商的 kernel ABI、graph 格式、launch 参数、参考实现或容差。
- 不把 Metal pipeline 强行转成文件，也不要求 backend-native handle 走 BlobChannel。
- 不要求旧 run 按新规则 resume、重新确认或重新发布。
- 不在本阶段访问真实 NPU、安装 SDK、产生性能数字或发布加速比。
- 不用“新旧两套生产构建实现”换取兼容；版本并存只发生在**数据读取**层。

## 2. 目标语义模型

```text
SourceBundle
  ├─ source bytes + candidate_sha256
  └─ task/source manifest
        │
        ▼
backend.artifact_builder()
  ├─ ArtifactBundle / backend-native executable handle
  ├─ BuildProvenance
  └─ BuildEvidence
        │
        ▼
ExecutionDriver（现有执行路径的语义映射）
  ├─ ExecutionOutcome（report/status/stage/error）
  └─ TimingEvidence（设备、时钟域、同步与完整性）
        │
        ▼
EvaluationResult（版本化 envelope）
  ├─ CandidateDigest / ArtifactDigest / BuildProvenance
  ├─ ExecutionProvenance / Receipt / Signature
  └─ CachePolicy / PublicationPolicy
```

各层只统一必要语义：

- **构建层**：输入、构建过程、构建证据、构建产物；
- **执行层**：生命周期、状态、错误、同步与计时证据，不统一厂商内部 ABI；
- **缓存层**：身份和复用规则；
- **发布层**：结果身份、签名、确认和可比性规则。

## 3. 身份与新数据语义

### 3.1 `candidate_sha256`：只表示候选源码

- 继续是**实际候选源码字节的 SHA-256**，不把 builder、工具链或 pipeline 混进去。
- 旧 `candidate_sha256` 数据不变；新版本继续沿用同一算法，避免源码身份无意义漂移。
- assembled source（含 contract header）如需记录，另存 `assembled_sha256`，不得替换
  candidate_sha256。

### 3.2 `build_profile_digest`：表示构建环境与构建方式

新增版本化 profile document，至少包含：

- builder id/version、build host/profile；
- toolchain/SDK 版本、编译 profile、任务契约与 build flags；
- 必要的 backend/target 信息。

`build_profile_digest` 用于区分“同一 source 应该怎样构建”，不是候选源码的替代身份。
构建环境变化是否影响测量可比性，由 publication/epoch 规则另行判断。

### 3.3 `artifact_set_digest`：只表示构建产物

- 对文件型 artifact，仍是 canonical manifest + 每项实际字节 sha256 的摘要；
- 不包含 cache policy、placement 或执行环境；
- 对 Metal pipeline 等 backend-native handle，使用独立 descriptor + source/build 身份，
  不得拿空字节或伪造文件 digest 冒充。

### 3.4 `BuildEvidence`：新构建证据，旧报告保留

新数据可写结构化 `BuildEvidence`：

- `status`、`mode`（prebuilt / build-in-attempt / local）；
- `builder/toolchain/profile`、source/artifact digest；
- 分阶段耗时（assemble/compile/pipeline 等，按 backend 能力）；
- 有界编译日志、错误分类和诊断；
- backend-native artifact descriptor（不含不可序列化的 live handle）。

旧 `compile_s`、`pipeline_s`、`language_version` 等字段继续按旧规则读取。新版本不再把
所有 backend 的阶段硬塞进同一组耗时字段；如果 Metal 仍输出这些字段，它们必须来自同一次
真实构建，不能由另一路径补算。

## 4. 执行、缓存与发布语义

### 4.1 执行语义

统一的是生命周期和证据，不是厂商内部执行细节：

- `prepare/execute/cancel/cleanup/health` 的 attempt/fence 约束；
- `status/stage/error_domain/error_code` 与候选 compile_error/incorrect/crash/timeout 和
  infrastructure/protocol/policy 的分类；
- 设备侧同步完成、计时来源、时钟域、声明拓扑所需的 span 和完整性；
- backend-native artifact 只在创建它的后端语义内执行，不能跨后端消费。

Metal 的 grid/threadgroup、正确性检查、设备锁、超时和发布执行行为不在本阶段重写。

### 4.2 缓存语义（新版本命名空间）

新缓存键由以下部分组成：

- candidate_sha256、task fingerprint；
- build_profile_digest、artifact format/schema；
- execution environment（site/device/backend/target/worker build/calibration/protocol/epoch）。

收益是避免“同 source、不同工具链/后端/构建方式”错误复用。旧缓存只读，不双写；新缓存
可以冷启动，不要求旧 key 继续命中。`reuse/refresh/disabled` 的策略含义不变。

### 4.3 发布语义

新版本结果继续要求：真实测量、`publication_allowed=true`、site/epoch allowlist、签名和
确认门。新增的只是更完整的身份约束：

- BuildProvenance 与 ExecutionProvenance 必须一致绑定到同一 attempt/result；
- 执行环境或计时语义变化形成新 evaluator epoch，不拿新旧数字直接比较；
- 不同 backend/vendor/clock domain 默认使用各自基线，不因 task 名相同就宣称可比；
- simulated、诊断性或不完整证据永远不能发布。

这不是放宽发布，而是让新数据更不容易被误用。

## 5. 数据版本与兼容策略

1. 新 document/schema 必须显式版本化，版本路由覆盖 request、receipt、outbox、result、
   BuildEvidence、TimingEvidence 和 cache record。
2. 旧数据只读：旧 golden、旧报告和旧 outbox 继续按原规则验证，不回填、不重签、不改写。
3. 新 run 使用新 contract/epoch；旧 run 若必须 resume，只按旧规则处理，不能让它悄悄切换
   到新构建或新执行语义。
4. 生产构建只允许 ArtifactBuilder 一条路径。读取旧数据与保留旧生产路径是两回事，禁止
   以兼容为名恢复 `backend.compile()` 双轨。
5. 所有读写入口必须拒绝“版本未知、混合版本、版本与字段不匹配”，不能默默按旧格式猜。
6. 如果新语义改变执行或计时口径，必须建立新测量基线；旧数据只用于审计和历史对照。

## 6. 分阶段实施

### P0：接口与数据版本

- 定义 ArtifactBuilder、BuildOutcome、BuildProvenance、BuildEvidence 和版本路由；
- 写清新旧字段映射、digest 规则、缓存键和未知版本拒绝；
- 为 Metal source、文件型 NPU artifact、backend-native handle 各写 golden；
- 旧 Metal 数据逐字节保持不变。

验收：新旧 golden 各自通过，混合/未知版本 fail closed；没有生产 `backend.compile()` 调用。

### P1：Metal 单路径与真实 Mac 验收

- `tcenv/evaluate.py` 改为 `backend.artifact_builder()`，移除 `MetalBackend.compile`；
- `MetalSourceBuilder` 复用真实 MSL 编译/pipeline，返回 native handle 和真实 BuildEvidence；
- 有效 MSL 编译+pipeline、无效 MSL compile failure、真实 source digest 和诊断；
- 与旧路径对照 candidate digest、compile/pipeline 字段、缓存、执行和发布行为。

验收：真实 evaluator 只经 builder，Metal 旧行为不变；不跑性能 benchmark 或发布数字。

### P2：新数据写入、缓存与恢复

- 新 request/receipt/outbox/result/BuildEvidence 按新版本写入；
- 缓存使用新命名空间，旧缓存只读；
- 验证 resume、篡改、版本混合、构建环境变化、cache policy 和旧 run 不被切换；
- 更新 brief/observe/查询的版本分派，避免新旧数字误比较。

验收：新数据全链一致，旧数据只读可验证；没有双生产构建路径。

### P3：fake vendor 与清微模拟

- fake TSM builder 在 control_host 生成小 ELF，BlobChannel 交给 fake NPU；
- evaluator 端构建和 control_host 预编译两种时序产生一致的构建语义；
- 覆盖工具链不匹配、artifact/provenance 篡改、异步完成、清理和旧 fence；
- TSM simulator 复用同一 BuildOutcome/ExecutionOutcome，不另造编译语义。

验收：真实六 RPC/blob 闭环；所有结果 simulated=true、publication_allowed=false。

### P4：真实设备与第二 backend（另行批准）

- 真实 TSM SDK/runtime/driver 上验证 BuildProvenance 与设备执行；
- 第二个不同形态 backend 验证接口不是 Metal/TSM 专用；
- 真实计时、设备拓扑和发布许可另行验收，不使用本计划中的模拟证据替代。

## 7. 验收矩阵

| 维度 | 必须证明 |
| --- | --- |
| 单路径 | evaluator 只经 ArtifactBuilder；没有旧 `backend.compile()` 或绕过 builder 的 runtime 调用 |
| 身份 | candidate、build profile、artifact、execution environment 四种身份不混淆、不互相替代 |
| 构建证据 | 真实 compile/pipeline 信息、诊断、耗时与错误分类来自同一次构建，无后补伪造 |
| 执行 | backend-native artifact 不被跨后端消费；取消、超时、清理和旧 fence 正确 |
| 缓存 | 不同 builder/toolchain/artifact/epoch 不错误复用；策略含义不变 |
| 发布 | simulated/不完整证据不发布；新 epoch/基线不与旧数据直接比较 |
| 数据版本 | 旧数据只读可验证，新数据版本明确，未知/混合版本拒绝，无历史回填 |
| 真实 Mac | 有效/无效 MSL、真实 pipeline、真实诊断与旧行为对照通过 |
| 多 backend | 至少两种不同执行形态复用接口，无厂商专用生产分支或 RPC |

## 8. 成本、收益与边界

主要成本是**新接口、新数据版本、读写路由和两组 golden**，不是迁移旧数据。额外成本包括
新缓存冷启动和一次性的新基线确认，不应扩大成永久双轨维护。

预期收益：构建入口唯一、缓存身份更准确、构建证据更完整、未来多 backend 不再依赖隐式
编译流程，同时旧证据仍可审计。

仍未决定或需要真实证据的问题：新 BuildEvidence 是否覆盖所有 backend 所需阶段、不同
backend 的执行/计时语义在哪一层真正可比、旧 run resume 的保留期限，以及真实 NPU 的构建
和加载契约。上述问题未验证前，不得宣称新语义已在所有设备上成立。
