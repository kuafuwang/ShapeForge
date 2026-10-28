# 构建、执行与证据语义升级计划

状态：设计计划，2026-10-28。**尚未实施。** 代码基线是 HEAD `f325181`。

前提：`runs/` 里的历史数据**只是本地测试数据，没有投入生产**，本计划不要求兼容。
因此不再保留“旧报告只读、旧 run 续跑、版本并存”这一整套复杂度：新语义直接成为
唯一语义，旧数据丢弃或用当前代码重新生成。

范围：本地 Metal 的构建入口、报告里的构建证据、缓存键与确认门的数据语义。
六 RPC、BlobChannel、VendorDriver、fake TSM、真实 NPU 的 wire contract 归
`多厂商NPU评测通信抽象-plan.md`。

## 0. 一句话

**构建只剩一条路径，报告只剩一个新版本，旧 run 不再续跑。**

## 1. 已定方向

- 生产构建只有 `backend.artifact_builder()` → `ArtifactBuilder.build`。
- 删除 `Backend.compile` / `compile_all`、`MetalBackend` 上的同名方法，以及
  `xcworker`、`probes`、`profile` 里绕过 builder 的调用；**不做**委托包装或兼容分支。
- `MetalContext.compile` 只作为 `MetalSourceBuilder` 的内部 primitive。
- 旧 run 不再续跑；旧报告不再读入。丢弃历史数据比维护双版本简单得多。
- 允许改 interface 语义，但不统一各厂商 ABI、launch 参数或容差。
- Metal 必须有真实实现并在真实 Mac 上验收，不能是概念类或测试专用类。

## 2. HEAD 上实际有什么

| 位置 | HEAD 行为 |
| --- | --- |
| `tcenv/evaluate.py:417-421` | 调 `backend.compile` / `compile_all`；没有 `artifact_builder()` |
| `tcenv/backends/base.py:332,343` | 协议上仍有 `compile` / `compile_all` |
| `tcenv/backends/metal/backend.py:389,400` | 同上，委托给 `ctx.compile` / `ctx.compile_all` |
| `tcenv/xcworker.py:50,60` | 直接 `ctx.compile` / `ctx.compile_all` |
| `tcenv/backends/metal/probes.py:39`、`profile.py:87` | 直接 `ctx.compile` |
| `tcenv/backends/metal/source_builder.py:209` | `build` 调 `self.backend.compile`；evaluator 不调用它 |
| `tcenv/evaluate.py`、`tcenv/worker.py` | 报告写 `REPORT_SCHEMA_VERSION = 3`；缓存键 `EVAL_CACHE_SCHEMA = 2`（`evaluate.py:981`） |
| `tcenv/remote/job_engine.py:1830` | 超时报告同样走 `to_json()` |

已有、本计划不重做：`BuildProfile.capability_document()`、`artifact_set_digest`、
`NativeExecutable.digest`、`SourceBundle.sha256`、NPU `document_version` 1–5。

工作区里有与本文相关但**尚未提交、也未验收**的草稿（构建入口、`source_builder`、
`test_single_build_path.py` 等）。它们不是基线，`test_single_build_path.py` 还会让
`test_structure.py` 变红（shared 测试里 import backend）；P0 按本文重写测试，
不照搬这份草稿。

## 3. 单一构建路径

```text
tcenv/evaluate.py
  → backend.artifact_builder()
  → ArtifactBuilder.build(source, task_contract, deadline)
  → MetalSourceBuilder
  → MetalContext.compile           # 仅 builder 内部
  → BuildOutcome(native handle, provenance, evidence)
```

- `Backend` 协议用 `artifact_builder()` 取代 `compile`/`compile_all`。
- `evaluate.py`、`xcworker.py`、`probes.py`、`profile.py` 全部改走 builder；
  不再有任何“先 backend.compile 再 builder”的第二条路。
- `MetalSourceBuilder` 复用现有 `assemble` + `MetalContext.compile` + pipeline 创建，
  返回 backend-native handle；旧 `compile_s` / `pipeline_s` / `language_version`
  报告字段仍来自同一次真实构建，不从 `BuildEvidence` 反推。
- `MetalContext` 由 backend 持有（`get_context()` 单例），不进中立
  `ArtifactBuilder.build` 签名；`MetalSourceBuilder` 可构造注入，`FakeNpuBackend`
  用普通对象满足同一签名。

## 4. 三种构建身份

| 身份 | 算法 | 含义 |
| --- | --- | --- |
| `candidate_sha256` | `SourceBundle.sha256` | 候选源码字节；不是 assembled 文本 |
| `build_profile_digest` | 对 `capability_document()` 做 RFC 8785 + SHA-256 | builder / toolchain / compile profile |
| `artifact_set_digest` | `ArtifactBundle.artifact_set_digest` | 工件集；native handle 不进 digest |
| `native_digest` | `NativeExecutable.digest` | 仅 Metal 本地；不写进 NPU provenance |

`assembled_sha256` 记录契约头合并后的文本摘要，但不替代 `candidate_sha256`。
执行环境仍是 `EvaluationResult.provenance`，不放进 `BuildOutcome`，也不参与
`build` 身份。

## 5. 报告 schema 4：唯一版本

`REPORT_SCHEMA_VERSION` 直接改为 4。**只写 4，只读 4。**

`build` 三种形状，8 个键：`shape`、`candidate_sha256`、`assembled_sha256`、
`build_profile_digest`、`artifact_set_digest`、`native_digest`、`builder_id`、
`build_profile`。

| 形状 | 约束 | 生产者 |
| --- | --- | --- |
| `null` | — | 编译前 `rejected`、父进程 timeout/crash、框架 crash、`unchanged`/`duplicate`、`JobEngine._timeout_report` |
| `partial` | artifact / native 两个 digest 为 `null`；profile 三件套要么全 `null`，要么前两个非空且 digest 为 64 位小写 hex；值只能来自 `BuildError` 已附材料，`except` 里禁止再 probe/编译 | `compile_error`、`deadline` 转成的 `timeout` |
| `full` | 8 个值全非空，且全部来自同一次 `BuildOutcome` | 编译成功后的 `ok` / `abi_error` / `unsafe` / `incorrect` / `runtime_error` / 分配失败 |

规则：

- `status=ok` 必须是 `full`。
- `validation` 与 `timed` 是两个 worker，各构建一次，各自带自己的 outcome；
  不合并耗时，也不合并 digest。
- 读入时按同一套形状规则拒绝不合格文件；没有“按旧版本宽容”的分支。

## 6. 读取与续跑：只接受当前版本

- `from_json` 只接受整数 `4` 和上面三种形状；`0–3`、`5+`、未知键一律拒绝，
  报错里写清“旧报告不再支持”。
- `taic resume` 遇到旧 schema 的 run 直接拒绝，给出明确提示（重新跑一场），
  不做内存兼容、不改写文件、不尝试升级。
- outbox 已完成行如果版本不是当前版本，同样拒绝重放；需要保留的结论只能重跑。
- 不再有 `local-resume` / `remote-resume` 的兼容分支：续跑只处理本计划之后
  产生的 schema 4 报告。

## 7. 缓存与确认门

**缓存**：`EVAL_CACHE_SCHEMA` 升到 3，键含 `candidate_sha256`、
`build_profile_digest`、`artifact_set_digest`、target/backend/协议与既有任务身份。
旧键不读、不迁移，冷启动重编重测。`reuse` / `refresh` / `disabled` 语义不变。

**确认门**：只对 `local-*` 候选在 `if rep.status == "ok" and rep.speedup` 之后、
`values.append` 之前检查：

- 搜索期报告必须是 schema 4 `full`，且每个成功 batch 也是 `full`；
- 五个 digest（`candidate_sha256`、`assembled_sha256`、`build_profile_digest`、
  `artifact_set_digest`、`native_digest`）与搜索期一致；
- `candidate_sha256` 必须等于当前 `source_sha256`；
- `null` / `partial` 不得 append。

不满足时 `entry["status"] = "build_identity_mismatch"`，走现有失败出口
（`_failure_record`、`rechecks/`、`failed_batches`、`stop=True`），不改
`rep.status`，不计 `hangs`。

远端候选不新增 digest 门：沿用今天的 `ok` + `speedup` 路径，`build` 只保存不比较。

## 8. 数据重置

既然旧数据不要了：

- `runs/` 不迁移；需要历史结论时用当前代码重跑。
- 测试夹具 `tests/support/golden/metal/fixtures/*.report.json` 由 schema 3 重建为
  4，旧的删除，不保留双份向量。重建后仍需覆盖 `ok` / `compile_error` /
  `incorrect` 等既有语义。
- `docs/架构/backend-phase0-baseline.json` 是历史基线记录，不改写；schema 4 的
  对照另建新基线文件，不覆盖旧文件、不宣称旧基线被替代。
- NPU `document_version` 1–5、`tcenv.eval-report.v3` 字符串与 `PROTOCOL_VERSION`
  不动。

## 9. 受影响的下游：observe、brief、catalog

schema 4 不只影响 worker 和确认门。以下位置直接读候选 `report.json` 或确认报告，
必须同步，否则会静默跳过新报告或把错误字段当旧 schema 解释：

| 位置 | 现在读什么 | schema 4 需要什么 |
| --- | --- | --- |
| `observe/index.py` 候选行 | `candidates/<tag>/report.json` 的 status/stage/speedup/协议字段 | 接受 schema 4；`build` 只索引 `shape` 与四个 digest（是否建列由 P1 决定）；不因新字段报错或整行丢弃 |
| `observe/detail.py`、`observe/query.py` | 展示候选与详情 | 能展示 `build` 摘要；旧 run 目录不在索引里时无特殊处理 |
| `taic/brief.py` | 扫描 `candidates/*/report.json`、`confirm_report.json` | schema 4 的 `ok`/失败分类照旧；`build` 缺失不影响摘要，不当作旧版本兼容 |
| `taic/kernel_catalog.py` | 从 run 目录取 best/候选 | 只依赖文件与确认结果；若断言 schema 版本则同步到 4 |
| `observe` 索引 schema | `index_schema_version` 与派生列 | 若新增 `build` 列/过滤，需升 `INDEX_SCHEMA_VERSION` 并重建；若只展示 JSON，不强行升版 |

约束：

- 这些模块**不读取旧 schema 0–3**；遇到旧 run 目录按“不可索引/不可展示”处理，
  不猜字段、不伪装成 4。
- 不要求 observe 为构建身份做发布判断；发布门仍在 `taic`。
- 相关测试（`test_observe*`、`test_brief*`、`test_kernel_catalog*`）的 fixture 若使用
  schema 3 报告，随 P1 的夹具重建一起改为 4；不要靠放宽容错让两边都过。
- 若 P1 决定给 `build` 建索列，索引升版与全量重建写入同一个提交，并在验收里
  包含一次旧索引重建。

## 10. 阶段与验收

### P0：单路径构建（可独立合入）

- Backend 协议改为 `artifact_builder()`；删除 `compile` / `compile_all` 及
  `MetalBackend` 同名方法；`evaluate` / `xcworker` / `probes` / `profile` 迁移。
- `MetalSourceBuilder` 接真实 `assemble` + `MetalContext.compile`，返回 native handle。
- 结构检查：`tcenv/` 内除 `runtime.py` 与 `source_builder.py` 外没有
  `.compile(` / `.compile_all(`；`Backend` 没有 `compile`；`MetalBackend` 没有
  `compile`。
- 真实 Mac：有效 MSL 编译 + pipeline、无效 MSL 分类为 `compile_error`、诊断与
  耗时来自同一次构建；`candidate_sha256`、`compile_s`、`pipeline_s` 与旧行为一致。
- 无 GPU 回归：`test_backend_neutral`、`FakeNpuBackend` 走同一签名。

### P1：schema 4 与身份

- `build_profile_digest` 函数 + golden；报告写 4，`build` 三形状校验。
- 读入拒绝 0–3 与未知键；`taic resume` 对旧 run 明确拒绝（有专门测试）。
- 缓存键升到 3，旧键不命中。

### P2：确认门

- 本地确认门加 digest 检查与 `build_identity_mismatch` 失败出口。
- 远端路径不加门，保留 `ok` + `speedup`。
- 夹具重建为 schema 4；旧向量删除。

三者可以 P0 → P1 → P2 顺序合入；P2 依赖前两者，单独合入无法验收。

P1 还包含 observe/brief/catalog 的读取适配与夹具重建；P2 包含确认门。
下游适配若单独提交，必须同时给出“读旧 schema 被明确拒绝”的测试，
不能只改正向路径。

## 11. 不做的事

- 不保留旧 run 兼容、不回填、不重签历史。
- 不统一各厂商 ABI、launch 参数、容差或计时内部实现。
- 不改 Metal 的 grid、设备锁、超时、容差与 launch；不改 `PROTOCOL_VERSION`。
- 不改 NPU 六 RPC、BlobChannel、`document_version` 1–5 与 HMAC。
- 不在本文决定 build profile 是否进入 evaluator epoch；本轮 `build` 只做本地
  证据与确认门，不进 epoch。
- 不宣称新 schema 让已发布数字更准确；旧基线文件不重算。
