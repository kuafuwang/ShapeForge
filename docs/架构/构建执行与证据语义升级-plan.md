# 构建、执行与证据语义升级计划

状态：设计计划，2026-10-28。对照当时工作区代码写成，**尚未实施**。
通信、六 RPC、BlobChannel、VendorDriver、fake TSM 与真实 NPU 属于
`多厂商NPU评测通信抽象-plan.md`。本文只规定本地评测里构建身份如何进入
报告、缓存和确认门。

已定方向：生产构建只保留 `backend.artifact_builder()` → `ArtifactBuilder.build`；
不恢复 `backend.compile`。旧文件只读，不回填、不改写、不重签。版本差异只在
读取时分支。不统一各厂商 ABI、launch 或容差。Metal 的验收必须跑真实
`MetalSourceBuilder`，不能用替身类代替。

## 1. 代码里已经有的，不要重做

下列事实以当前源码为准。本计划不把它们再列成待办。

生产构建已经是一条路径。`Backend` 只有 `artifact_builder()`，没有
`compile` / `compile_all`。`MetalBackend` 同样没有这两个方法。
`tcenv/evaluate.py` 与 `tcenv/xcworker.py` 都调用 `backend.artifact_builder()`
然后 `builder.build(SourceBundle(...))`。`MetalContext.compile` /
`compile_all` 只出现在 `MetalSourceBuilder._compile_assembled` 和
`compile_source`。有一份未跟踪的 `tests/shared/test_single_build_path.py` 用正则扫生产树，
这是对的方向。但它在函数里 import 了 `tcenv.backends`，而
`tests/shared/test_structure.py` 禁止 `tests/shared/` 出现这种 import。
因此 `test_backend_free_tests_do_not_import_a_backend` 在这份文件存在时就会失败。
这不是报告 schema 的问题。新测试不要照这个位置再 import 一次 backend。
只读源码的断言留在 `tests/shared/`；要 import `MetalBackend` 或跑 `gpu` 的断言
放到 `tests/backends/metal/`。

`MetalSourceBuilder` 会真实调用 `MetalBackend.assemble` 与
`MetalContext.compile`，并返回带 live handle 的 `BuildOutcome`。
`evaluate` 随后丢掉这份 outcome，只把 `executable_handles()` 交给后面的
执行，并把 **kernel 对象上的** `compile_s`、`pipeline_s`、`language_version`
抄进 `rep.compile`。`BuildEvidence`、native digest、profile 都没有进入
`EvalReport`。

`compile_source` 仍被 `tcenv/backends/metal/probes.py` 和
`profile.measure_peak_bandwidth` 使用。它们编译的是固定探针源，不写候选
报告、不进缓存、不能当发布证据。保持这个例外。不要把探针改接 `build()`，
也不要让 `evaluate` / `xcworker` 改走 `compile_source`。

`full` 以及带校验的 `timed` 会先跑一个 validation worker，再跑测量 worker。
两次都调用 `build()`。`xcworker` 在另一个进程里再构建一次。pipeline handle
不能跨进程复用，也不能序列化进 BlobChannel。填写身份时只许使用**本 worker
这次** `build()` 的 outcome，禁止父进程为了补字段再编译一次，也禁止把
validation 那次的耗时抄进测量报告。

## 2. 目标与非目标

### 目标

1. 新产生的成功构建，在同一次 `BuildOutcome` 上留下四种身份，并写入新的
   `EvalReport`。
2. 读取按文档自己的版本分派。旧报告仍能按今天的规则读入。新版本写错字段或
   版本号过高时拒绝。
3. 确认门在新报告上核对构建身份。旧报告仍只核对源码字节。
4. 缓存键升级后，不再把升级前的 ok 报告当作命中。这不等于已经把工具链编进键。

### 非目标

- 不改 NPU `document_version` 1–5 的字段、digest 或 HMAC。不新增第 6 版
  wire contract。`build_profile_digest` 先只出现在本地报告。
- 不改 `PROTOCOL_VERSION`（现为 `"3"`），不把 build profile 塞进
  `EvaluatorProvenance` 的 epoch 输入。本变更不改变计时窗口。
- 不改 Metal 的 grid、容差、设备锁、超时和 launch。
- 不把 `rep.compile["compile_s"]` / `pipeline_s` 改成从
  `BuildEvidence.duration_ms` 反推。成功路径的 duration 是
  `(compile_s + pipeline_s)` 的毫秒四舍五入，失败路径另有下限 1，两者都不是
  报告字段的来源。
- 不把 `SourceBundle` 的默认路径 `kernel.tx` 改成 `kernel.metal`。该路径进入
  manifest，改了会改变所有 Metal `artifact_set_digest`。
- 不改 `evaluate` 里编译失败文案 `Metal compiler rejected the kernel`。
- 不规定旧 run 的保留期限，不写批量审计或迁移脚本，不改 `runs/`。
- 不在缓存命中路径上调用 `MetalContext.info()` / `probe()`。那一次 info
  会编译一小段固定 MSL，仍是设备访问。见 §7。

## 3. 四种身份

算法都用已经存在的函数。不另做一套 profile schema。

| 身份 | 现有算法 | 今天落在哪 | 本计划写入新报告的规则 |
| --- | --- | --- | --- |
| 候选源码 | `SHA-256(候选源码的 UTF-8 字节)`，即 `SourceBundle.sha256`。与 `hashlib.sha256(kernel_source.encode("utf-8"))` 相同 | 缓存键字段名是 `kernel_sha256`。旧远程请求字段名是 `candidate_sha256`。NPU receipt 用 `candidate_digest`，不双写 `candidate_sha256` | 键名 `candidate_sha256`。不得改用 assembled 文本 |
| 构建 profile | **尚无此 digest。** 材料是 `BuildProfile.capability_document()`：`builder_id`、`builder_version`、`build_host`、`build_profile`、`toolchain`、`compile_profile`。不含 source、不含 artifact | 只活在当次 `BuildProvenance` 里，报告和缓存键都没有 | `build_profile_digest = SHA-256(RFC8785(capability_document()))`，走 `contracts.canonical_digest_document` |
| 工件集 | `artifact_set_digest`：`SHA-256(RFC8785(canonical_manifest))`。manifest 含 `schema_id`、`schema_version=1`、按 path 排序的 path/role/media_type/size/sha256 | Metal 的唯一 entry 是候选源码本身（路径仍是 `kernel.tx`）。native handle 不在这个 digest 里 | 原样抄 `outcome.artifact_set_digest`。禁止用空字节或伪造文件冒充 |
| 执行环境 | `EvaluatorProvenance` schema 1。epoch 输入是 site、device、backend、`backend_version`、target、worker_build、task_pack、calibration、`protocol_version`。本地后三项里 worker_build 与 calibration 是空串 | `execution.json` 的 observed、发布白名单。本地缓存键只有其中的 backend、backend_version、target、protocol | **不新增字段，不改 epoch 公式** |

另外两个容易混进去、但必须分开的值：

- `assembled_sha256` 是契约头加候选源码之后的 SHA-256，今天在
  `NativeExecutable.descriptor["assembled_sha256"]` 和构建日志里。它不是
  `candidate_sha256`。
- native digest 是 `NativeExecutable.digest`，即
  `SHA-256(RFC8785(native document))`。`artifact_bundle.py` 的注释写它由
  `BuildProvenance` 绑定，**字段并没有**。`provenance_document` 只有
  `source_sha256` 与 `artifact_set_digest`。以字段为准。本计划不把 native
  digest 补进 NPU provenance（那会改变已签名的 v5 文档）。本地报告单独记录它。

Metal 的 `compile_profile` 今天含 `language`、`math`、`device`、`gpu_family`、
`unified_memory`。因此 `build_profile_digest` 会随设备名字符串变化，它不是纯
工具链 id。不要为了「看起来干净」删掉这些键：`capability_document()` 已进入
NPU v5 的逐字段比较，改键就是另一份 contract。

`MetalBackend.version` 是字符串 `"1"`，不是 MSL 版本。MSL 版本在
`toolchain.version`（来自 `MetalContext.info()` 的 `language_version`）。

## 4. 版本：谁可读，谁可写

| 文档 | 当前版本 | 读取 | 本计划的写入 |
| --- | --- | --- | --- |
| `EvalReport` | `REPORT_SCHEMA_VERSION = 3`。缺 `schema_version` 视为 0。高于进程能读的版本，或出现不认识的字段，`SchemaError` | `migrate_report` 只在内存里丢掉派生字段 `eligible`，并在只有 `eligible: true` 时补 `status=ok`。它会丢掉 `schema_version`，所以版本规则必须写在丢掉之前。不写回文件 | 升到 **4**。0–3 仍按今天读入。5 以及未知字段仍拒绝 |
| worker 请求 | `REQUEST_SCHEMA_VERSION = 4` | 缺版本视为 0；未知字段拒绝 | 不动 |
| 本地评测缓存 | `EVAL_CACHE_SCHEMA = 2`，放在键的 JSON 里再做 SHA-256。只缓存 `status=ok` | 键不中就当没有。损坏文件当没有 | 升到 **3**，使升级前的文件不再被这条键打开。不删除、不改写旧文件 |
| host outbox | NPU contract 为 2，其余为 1 | 行上版本与当前 contract 不一致则 `outbox_conflict` | 不动。本地 Metal 搜索不走这份 outbox 的 build 行 |
| 远程 contract | legacy 1，npu-source 2，blob 3，bundle 4，build 5 | 未知或混配的 API/evidence 对拒绝 | 不动 |
| `EvaluatorProvenance` | schema 1 | 缺字段或未知字段拒绝 | 不动 |

写入规则：

- 新评测的 `to_json()` 写 schema 4。已有的 `report.json`、`result.json`、
  outbox、receipt 不因被读入而写回。`from_json` 再 `to_json` 会盖上当前版本号
  （今天对 schema 3 已经如此）。resume 不得借这条往返改历史文件。
- 已完成的结果只重放存储内容，不按新语义重建，不重签。
- 旧 run 若继续评**新**候选：仍用同一个 `ArtifactBuilder`，新候选写 schema 4。
  这不是第二条构建实现。已完成候选的文件保持原字节。
- 不在旧 run 里把已有报告升级成 schema 4，也不为了续跑保留一个只写 schema 3
  的生产写入器。
- 一个 run 里可以同时存在 schema ≤3 与 schema 4 的不同文件。比较时看加载后的
  `build` 字段，不看 run 的年龄。

`from_json` 在调用 `migrate_report` 之前先看版本，因为版本号随后会被丢掉：

- 版本 ≤3：`build` 不算已知字段。文件里若出现这个键，按未知字段拒绝。
  构造出的对象里 `build is None`。
- 版本 =4：顶层必须有 `build`。缺键拒绝。
- 版本 >4：拒绝。

`build` 为 JSON `null`，或为一个闭集对象，多一个键就拒绝。对象的键只有
`candidate_sha256`、`assembled_sha256`、`build_profile_digest`、
`artifact_set_digest`、`native_digest`、`builder_id`、`build_profile`。

- `status=ok`：七个键都在，四个 digest 都是 64 位小写 hex。不允许 `null`。
- `compile_error`：对象必须在。`candidate_sha256` 是输入字节的 SHA-256，不需要
  第二次编译。`assembled_sha256`、`build_profile_digest`、`artifact_set_digest`、
  `native_digest` 为 JSON `null`。不要用 `_fallback_profile()` 造一个看起来像
  真实环境的 digest。
- `rejected` 与 `pending`：`build` 为 JSON `null`（还没构建）。
- 其余状态（构建已成功，之后才失败）：与 `ok` 相同的满对象。
- `toolchain_profile` 与 `deadline` 今天在 `evaluate` 里继续向上抛，不写成候选
  的 `compile_error`。保持。

成功报告上的七个值必须来自同一次 `BuildOutcome`：`source_bundle.sha256`、
descriptor 里的 `assembled_sha256`、`provenance.profile` 的
`build_profile_digest`、`outcome.artifact_set_digest`、`outcome.bundle.native.digest`、
以及 profile 的 `builder_id` 与 `build_profile`。在 `finish()` 之前、`build()`
成功返回之后写入 `rep.build`，这样后面的失败出口也会带上它。

## 5. 缓存与确认

缓存键今天包含：`schema`、`protocol`、task/dtype/size/mode、task fingerprint、
evaluation fingerprint、`backend`、`backend_version`、`kernel_sha256`、seed、
reps、cases、fast_bench、require_contract_launch、validate、measure_serial、
`target_id`。不含 MSL、macOS、`build_profile`、site 或 epoch。远程结果本来就不
进宿主这份缓存。

`EVAL_CACHE_SCHEMA` 改为 3 **只为了**让命中结果一定是带 `build` 的新报告。
键的其余字段不增加。增加 live toolchain 就必须在父进程 probe，本计划不做
（§2、§7）。`reuse` / `refresh` / `disabled` 的含义不变。确认阶段已经是
`eval_cache=False`，保持。

确认门保持现有发布条件：`measurement_kind=measured`、`publication_allowed`、
site/epoch 白名单。simulated 仍不能进搜索。另外：

- 搜索期报告的 `build is None`：维持现状。`confirmation_source_sha256` 仍是
  `SHA-256(优胜者源码 UTF-8)`，在 `taic/orchestrator.py` 收尾处由源码计算，
  不从报告反读。
- 搜索期报告的 `build` 是对象：同一次确认复测（`eval_cache=False`）的
  `rep.build` 必须在 `candidate_sha256`、`build_profile_digest`、
  `artifact_set_digest`、`native_digest` 上与搜索期报告一致，且
  `candidate_sha256` 等于 `confirmation_source_sha256`。不一致则这次确认失败，
  不发布加速比。失败沿用确认失败的既有停法，不新增发布通道。
- 不因为多了 `build` 就重测历史优胜者，也不把旧 speedup 标成失效。

## 6. 实施顺序

P1 可以单独合入。P2 依赖 P1，且 P2 列出的改动必须在**同一个提交**，不能先写
schema 4 再补确认门。

### P1：只增加 digest 函数

在 `BuildProfile` 上增加 `build_profile_digest()`，实现为
`SHA-256(canonical_digest_document(self.capability_document()))`。
不改 `capability_document()` 的键，不读文件，不碰设备。

验收（无 GPU、无模型）：

```bash
uv run --no-sync pytest -q -n 0 \
  tests/shared/test_build_profile_digest.py \
  tests/shared/test_structure.py
```

断言：同一 `capability_document()` 得到稳定的 64 位 hex；改 `toolchain.version`
或 `compile_profile.device` 都会变；文档不含 `source_sha256` 与
`artifact_set_digest`。该测试只 import `tcenv.remote`，不 import
`tcenv.backends`。文件还没写时，第一条路径会收集失败；写完后两条都应通过。
`test_structure.py` 若仍因 `tests/shared/test_single_build_path.py` 的 backend
import 失败，那是该文件的既有问题，不属于 P1。P1 不得为了让结构测试变绿去改它。

### P2：报告、缓存键与确认门一次落地

同一提交里：

- `REPORT_SCHEMA_VERSION` 改为 4，`EvalReport` 增加默认 `null` 的 `build`。
- `evaluate` 按 §4 从当次 outcome 填写。不增加编译次数。
- `EVAL_CACHE_SCHEMA` 改为 3。
- 确认门按 §5 比较。`build is None` 的旧报告路径行为不变。
- 更新 `tests/shared/test_report_schema.py`：今天用 `schema_version=4` 表示
  「过新」。改后过新的例子必须是 5；0–3 无 `build` 仍能读，0–3 若带上 `build`
  键则拒绝；schema 4 缺键、多键、`status=ok` 但 digest 为空，或 digest 非 hex，
  都拒绝。

验收（无 GPU、无模型）：

```bash
uv run --no-sync pytest -q -n 0 \
  tests/shared/test_report_schema.py \
  tests/shared/test_build_profile_digest.py \
  tests/shared/test_structure.py
```

必须同时满足：

- 生产树除 `runtime.py` 与 `source_builder.py` 外，没有 `backend.compile(` /
  `ctx.compile(`。这条用读文件完成，放在 `tests/shared/`，不 import backend。
  若 `test_single_build_path.py` 仍留在 `tests/shared/` 且继续 import
  `tcenv.backends`，上面的 `test_structure.py` 会失败。P2 要把它拆开：
  读文件的断言留在 shared，import `MetalBackend` 或带 `gpu` 标记的断言移到
  `tests/backends/metal/`。拆分与 schema 4 写在同一个提交，避免验收命令和
  目录规则互相矛盾。
- 新增一条无 GPU、且不 import backend 的测试：缓存键 body 的 `schema` 为 3，
  且 body **没有** `build_profile_digest` 键（避免以后有人把 probe 悄悄加进父进程）。
- 新增一条无 GPU、且不 import backend 的测试：`build is None` 时确认不要求
  四个 digest；`build` 为对象且复测 digest 不同则确认失败；digest 相同且源码
  哈希一致才沿用成功路径。
- `git diff` 不改 `runs/`、proto、部署脚本。

真实 Mac（要另获 GPU 许可；做不到就标 blocked，不许用 fake Metal 填结果）。
测试放在 `tests/backends/metal/`，不要放进 `tests/shared/`：

```bash
uv run --no-sync pytest -q -n 0 \
  tests/backends/metal/test_build_identity.py -m gpu
```

断言：`relu` / `fp16` / `small` 的 quick 评测走真实 `MetalSourceBuilder`，
`report.build["candidate_sha256"]` 等于源码 UTF-8 的 SHA-256，`native_digest`
等于该次 `NativeExecutable.digest`，`report.compile["compile_s"]` 等于该
kernel 对象的 `compile_s` 且大于 0。无效 MSL 仍是 `compile_error`，且
`build.artifact_set_digest` 与 `native_digest` 为 `null`。不要跑 benchmark，
不要写发布加速比。文件名可以不同，目录必须是 `tests/backends/metal/`。

## 7. 成本与尚未被证据支持的说法

会付出的成本：

- 缓存冷启动。schema 2 的文件还在磁盘上，新键不会读它们，ok 候选要重编、重测。
  这是一次性的，不是第二条缓存实现。
- 确认变严只作用于带 `build` 的新报告。机器在搜索和确认之间换了设备名或 MSL
  版本时，profile digest 不同，确认失败。这是正确性方向的失败，不是新的加速比。
- 读旧报告的成本不变：v0 仍走 `migrate_report`。不要另做审计作业来「补齐」digest。

尚未被证据支持，实施时不要写成收益：

- 没有测量证明已经发生过「同类源码、不同工具链命中同一缓存」。schema 3 的键
  **仍然**不含 MSL / macOS。`backend_version="1"` 不变时，系统升级后仍可能命中
  升级前写入的 schema 3 报告。确认阶段因为 `eval_cache=False` 会重测并比对
  digest，所以这条洞不应变成已确认的发布数字。未跑确认时，`best_speedup` 仍可能
  是另一次工具链的缓存。要堵上命中路径，必须接受每次查缓存都 probe 设备；那是
  另一次行为变更，不在本计划。
- 单路径构建已经在树上。本计划不产生「从此只有一个编译入口」这项新收益。
- 本地 `build` 对象没有第二个生产 backend 在用。NPU 的构建证据已经在
  document_version 5 的签名文档里，不由这次 schema 4 代替。
- 本文件没有跑 `gpu` 测试，不能当作 Mac 验收记录。

## 8. 仍不在本文决定的事

- 缓存命中前是否 probe。决定是本计划不做，见 §7。
- native digest 要不要进入某一版 NPU provenance。要做就必须新的 document
  version，并且不能改 v5 的字节。归通信计划。
- 旧 run 目录留多久。本文不删数据。
