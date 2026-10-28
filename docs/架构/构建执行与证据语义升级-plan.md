# 构建、执行与证据语义升级计划

状态：设计计划，2026-10-28。**尚未实施。** 代码基线是 HEAD `f325181`；
构建相关文件与 `evaluate.py` 等于 `2e02088`（其后的 `65717ad`、`b95043b`
只改文档）。下面的行号都按 HEAD 核对，不按工作区里未提交的 diff。

范围：本文只规定本地 Metal 的一条构建路径、本地新报告的 schema 4、缓存键
和确认门的数据语义。六 RPC、BlobChannel、VendorDriver、fake TSM、真实 NPU
的 wire contract 以及 NPU 证据能否发布，归通信计划。

已定方向：生产构建只保留 `backend.artifact_builder()` →
`ArtifactBuilder.build`；删除 `Backend.compile` / `compile_all` 与
`MetalBackend` 上的同名方法，不做委托包装。旧数据只读：不回填、不改写、
不重签，不改历史 golden 与基线文件。允许改 interface 语义，但不统一各厂商
ABI、launch 参数或容差。版本并存只发生在读取层，生产构建只有一套实现。
Metal 必须有真实实现并在真实 Mac 上验收，不能是概念类或测试专用类。

工作区当前有未提交改动：构建入口草稿（`tcenv/evaluate.py`、`backends/base.py`、
`metal/{backend,source_builder,probes,profile}.py`、`remote/artifact_builder.py`、
`xcworker.py`）、属于通信计划的 `remote/{artifact_bundle,blob_channel}.py`、以及
未跟踪的 `tests/shared/test_single_build_path.py`。这些都不是基线，也没有验收
记录：本文的「已有什么」只按 HEAD 写，「要改什么」按 §6 写。

## 1. HEAD 上实际有什么

| 位置 | HEAD 行为 |
| --- | --- |
| `tcenv/evaluate.py:417-421` | 调 `backend.compile` / `backend.compile_all`；没有 `artifact_builder()` |
| `tcenv/backends/base.py:332,343` | 协议上仍有 `compile` 与 `compile_all` |
| `tcenv/backends/metal/backend.py:389,400` | 二者委托给 `ctx.compile` / `ctx.compile_all` |
| `tcenv/xcworker.py:50,60` | 直接 `ctx.compile` / `ctx.compile_all` |
| `tcenv/backends/metal/probes.py:39`、`profile.py:87` | 直接 `ctx.compile` |
| `tcenv/backends/metal/source_builder.py:209` | `build` 调 `self.backend.compile`；evaluator 不调用这个类 |
| `tcenv/remote/artifact_builder.py:457,563-566,640-642` | 已有 `BuildOutcome`（bundle/provenance/evidence），没有 `executable_handles`；中立 `build` 是三参数签名；Registry 就这样调用 |
| `tcenv/evaluate.py`、`tcenv/worker.py` | 报告一律 `to_json()` 写 `REPORT_SCHEMA_VERSION = 3`；缓存键 `EVAL_CACHE_SCHEMA = 2`（`evaluate.py:981`） |
| `tcenv/remote/job_engine.py:1830`、`job_store.py:2345` | 超时报告用 `to_json()`；`_validate_report` 只做 `from_json` 与绑定检查，不特别处理 `build` |

已有、本计划不重做：`BuildProfile.capability_document()`、`artifact_set_digest`、
`NativeExecutable.digest`、`SourceBundle.sha256`、NPU `document_version` 1–5、
`from_json` 的「新版本 / 未知字段直接拒绝」策略。

`tests/shared/test_single_build_path.py` 是未跟踪文件，在函数里 import
`tcenv.backends`，因此只要它在 `tests/shared/`，`test_structure.py` 的这一条
就是红的。它断言的是草稿，不是 HEAD。不要提交它，也不要改它的 import 消红；
P0 另写测试。

## 2. 目标与非目标

目标：

1. P0 之后，evaluator 与 xcworker 只经 `ArtifactBuilder.build` 构建。
2. 一次成功的 `BuildOutcome` 带三种构建身份：候选源码、build profile、工件集。
   执行环境仍是 `EvaluationResult.provenance`，不放进 `BuildOutcome`。
3. 本计划新写的本地 Metal 报告用 schema 4，记下本 worker 有没有拿到
   `BuildOutcome`。读入的旧文件保持原版本。
4. 缓存键升到 schema 3：升级前的 ok 报告不再被这条键打开，键里仍无现场工具链。

非目标：

- 不改 NPU `document_version` 1–5 的字段、digest 或 HMAC，不新增第 6 版 wire
  contract。字符串 `tcenv.eval-report.v3`（`REPORT_SCHEMA_ID`）与报告里的整数
  `schema_version` 不是一回事，不要跟着改成 v4。
- 不改 `PROTOCOL_VERSION`（`"3"`），不把 build profile 塞进 `EvaluatorProvenance`
  的 epoch 输入；计时窗口不变，所以不重录 `backend-phase0-baseline.json`，
  也不重签旧 run。
- 不改 Metal 的 grid、容差、设备锁、超时和 launch；不把 `rep.compile` 的
  `compile_s` / `pipeline_s` 改成从 `BuildEvidence.duration_ms` 反推。
- 不改 `SourceBundle` 的默认路径 `kernel.tx`，不改文案
  `Metal compiler rejected the kernel`。
- 不规定旧 run 的保留期限，不写迁移脚本，不改 `runs/`；不在缓存命中路径上
  调用 `MetalContext.info()` / `probe()`。

## 3. 三种构建身份，加上已有的执行环境

构建身份来自同一次 `BuildOutcome`。执行环境不在这个对象里。

| 身份 | 算法（HEAD 已有） | 放在哪 |
| --- | --- | --- |
| 候选源码 | `SourceBundle.sha256` | 报告 `build.candidate_sha256`。不是 assembled 文本 |
| build profile | **P1 的新函数**，材料是 `capability_document()`：`builder_id`、`builder_version`、`build_host`、`build_profile`、`toolchain`、`compile_profile`；不含 source，不含 artifact | 报告 `build.build_profile_digest` |
| 工件集 | `outcome.artifact_set_digest = SHA-256(RFC8785(canonical_manifest))`；native handle 不在里面 | 报告 `build.artifact_set_digest`；`build.native_digest` 另抄 `NativeExecutable.digest` |

`assembled_sha256` 是契约头加候选源码之后的 SHA-256：记在报告里，但不代替
`candidate_sha256`。Metal 的 `compile_profile` 今天含 `device` / `gpu_family`，
digest 会随设备名变；不要为了「看起来干净」删键，该文档已进入 NPU v5 的逐字段
比较。HEAD 的 `provenance_document` 没有 native digest，不要把它补进 v5。

执行环境是 `tcenv/evaluation.py` 的 `EvaluatorProvenance`（schema 1），epoch
输入含 site、device、backend、`backend_version`、target、worker_build、
task_pack、calibration、`protocol_version`。发布门继续用这份 provenance
（`measurement_kind`、`publication_allowed`、site/epoch 白名单）；`build` 不参与
epoch，也不替换它。simulated 仍不能进搜索。

## 4. schema 4 的 `build`

`from_json` 先读整数版本（缺键为 0），再按版本决定是否接受 `build`，且发生在
构造 `EvalReport` 之前。通过之后版本留在对象上（运行期属性，不进 JSON）。

- 版本 ≤3：不接受 `build` 键（按未知字段拒绝）；对象上 `build` 为 `None`。
- 版本 4：必须有 `build`；`status=ok` 必须是 full；非 ok 可以是 `null`、
  partial 或 full。
- 版本 5 或任何多余键：拒绝。

三种形状：

- `null`：这份报告的生产者没有本 worker 的 `BuildOutcome`。
- partial 与 full 用同一组 8 个键：`shape`、五个 digest（`candidate_sha256`、
  `assembled_sha256`、`build_profile_digest`、`artifact_set_digest`、
  `native_digest`）和 `builder_id`、`build_profile`。full 全非空；partial 的
  artifact / native 两个 digest 为 `null`，profile 三件套要么全 `null`，要么
  前两个非空且 digest 是 64 位小写 hex，`assembled_sha256` 可 hex 可 `null`。
- partial 里非空的值只能来自同一次失败里已附在 `BuildError` 上的东西，`except`
  里禁止再 `probe()` 或再编译；full 的 8 个值只从这一次 `BuildOutcome` 抄。

谁是哪个形状：

| 生产者 | `build` |
| --- | --- |
| 编译前 `rejected`、父进程新建的 timeout / crash、`worker.py` 的框架 crash、`unchanged` / `duplicate` / `external_duplicate`、`JobEngine._timeout_report` | `null` |
| 编译失败 `compile_error`（含非 UTF-8 源码）与 `deadline` 变成的 `timeout` | partial 或 `null`，按 `BuildError` 上实际带了什么 |
| 编译成功之后的每个 `finish`：`ok`、`abi_error`、`unsafe`、`incorrect`、`runtime_error`、分配失败的那几个 `rejected` | full |
| 父进程读回的子进程报告（`evaluate.py:949`，以及随后被改判成 `unsafe` 的路径） | 原样保留，不改 `build`，也不清成 `null` |
| 远端 Metal worker（走同一份 `tcenv/evaluate`，见 `remote/production.py`）与厂商报告 | 由生产者决定；读入后按 §4.1 保存 |
| outbox 已完成行 | 不自己构建，只重放存储的字节 |

`full` 与带校验的 `timed` 是两个 worker：validation 先跑、measurement 再跑，
各构建一次。测量报告只带测量 worker 的 outcome；validation 失败被返回时带的是
它自己那份。两次耗时与两份 digest 不合成一条。

## 4.1 写入版本、远端边界、确认从哪分支

写入只有一条规则，版本由对象来源决定，没有按调用点选版本的参数：

- 本进程新建的报告：写 `REPORT_SCHEMA_VERSION`（4），`build` 按 §4；
  `JobEngine._timeout_report` 也在这里，它的 `build` 是 `null`。
- `from_json` 读入的报告：写出它被读进来的那个整数版本；0–3 仍然不写
  `build`，4 照抄 `build`。旧文档不会因为被重新序列化就升级。
- 写版本 4 时 `status=ok` 必须已经是 full；读入用同一套形状规则拒绝不合格的
  文件。测试夹具若要 `ok` 报告能读回并走到确认成功，必须自带一份合成的 full
  `build`（固定 hex，不编译 Metal）。生产代码不为夹具开旁路。

远端边界（本文选定的那一套规则，其余说法作废）：

- 远端 Metal worker 跑的就是这份 `tcenv/evaluate`，所以它写出的报告与本地
  worker 同版本、同形状。`JobStore._validate_report` 保持今天的行为（`from_json`
  加绑定检查），**不新增**对顶层 `build` 的拒绝：旧 worker 交回的 schema ≤3 与
  新 worker 交回的 schema 4 都照收。
- 远端报告的 `build` 只保存，不比较。远端**不新增** digest 门，也**不**因为
  身份而失败；也不把远端结果排除在确认池外。
- 远端新报告的落盘版本就是它自己的（见 §5.1），不升级、不补 `build`。

确认只看 §5.1 写在报告上的 `origin`：`local-new` 与 `local-resume` 走 §5 的
digest 门，`remote-new` 与 `remote-resume` 走今天的 ok 路径（`ok` 且有
speedup 就进中位数）。进入搜索仍由 `_check_publication`
（`taic/orchestrator.py:1443-1453`）决定，那不是确认算法。

无 GPU 回归放在 `test_confirm_build_identity.py`：`origin="remote-new"`、
schema 3、无顶层 `build`、两个 ok batch → 确认成功，中位数是其中一份。

## 5. 确认门：每个候选、每个成功 batch

对照 `taic/orchestrator.py` 的 `_confirm`（约 3191–3256 行）与下面的 `_finish`
（约 3441 行）：只有 `ok` 且有 `speedup` 的 batch 进 `values` / `reports`；全部
成功后中位数是 `statistics.median(values)`，留下最接近中位数的那份。
`confirmation_source_sha256` 要到 `_finish` 才赋值，循环里用局部 `source_sha256` 重算。

在 `if rep.status == "ok" and rep.speedup` 为真、`values.append` 之前，只对
`c.report.origin` 为 `local-*` 的候选做身份检查；本地分支先看搜索期报告有没有
`build`（即文件版本是不是 4），再看它是不是续跑读入的：

- 搜索期报告是 schema 4（新评的或续跑读回的）：必须 full；每个 batch 也必须
  schema 4 full，五个 digest（`candidate_sha256`、`assembled_sha256`、
  `build_profile_digest`、`artifact_set_digest`、`native_digest`）与搜索期相同，
  两边 `candidate_sha256` 都等于 `source_sha256`。`local-new` 与 `local-resume`
  都走这条，续跑不会因为 origin 变宽。
- 搜索期报告是 schema ≤3（无 `build`）：只有 `local-resume` 走兼容分支——不与
  旧文件比 digest；本轮 batch 必须是 schema 4 full、`candidate_sha256` 等于
  `source_sha256`，且同一候选各 batch 的五个 digest 彼此相同。
- `local-new` 且搜索期报告是 schema ≤3：失败（`build_identity_mismatch`）。
- `null` 或 partial 的 batch 不得 append。

不符时 `entry["status"]` 写 `build_identity_mismatch`，走现有失败出口：
`_failure_record`、`rechecks/confirm_<label>.json`、`failed_batches` 加一、
`stop = True` 并 `break`。不要写成 `ok`，不要改 `rep.status`，也不要因此增加
`hangs`（只在 timeout 时加）。外层 `if stop: break` 停掉后续候选，与今天某个
batch 失败相同；前面成功的 batch 不能把这颗核标成确认成功。

每颗候选只和自己的搜索期报告比；`choose_confirmed` 换 winner 时不做跨候选的
digest 比较——胜者进入 `confirmed` 前，它自己的每个成功 batch 已经对过它自己
的 `c.report`。`eval_cache=False` 保持。

## 5.1 来源标记与禁止写回

`origin` 是报告对象上的运行期属性，不写进 JSON：

- `taic/resume.py:563` 读入后写 `origin`：`run_dir/execution.json` 的
  `evaluation`（`orchestrator.py:761` 写下的 `"local"` / `"remote"`）决定
  `local-resume` / `remote-resume`。缺这份字段就 `NeedsInspection`，不要猜。
- `_evaluate_report` 拿到新报告时写 `local-new`（本地执行器）或 `remote-new`。

`_save`（`orchestrator.py:1292`）在 `1296` 行重写候选源码之前返回：`origin`
以 `-resume` 结尾时，不写源码、`report.json`、`feedback.txt`，也不改内存报告后
再序列化；`_session_brief` 的提示副本不得写回该路径。只有 `local-new` /
`remote-new` 才创建新文件，版本按 §4.1。续跑验收对源码文件和 `report.json` 都
比较 sha256，读入并调用 `_save` 之后两份字节都不变。

outbox 已完成行只重放。`complete()` 不得改写已完成行，重放字节与写入时相同，
不经过 `to_json` 再存回。

## 6. 阶段

依赖：P0 与 P1 互相独立，可分别合入；P2 依赖两者（分开合入会出现 schema 4
报告却仍按源码哈希确认）。P0 只改构建入口、诊断文本和失败分类，`EvalReport` 上
还没有 `build`，P0 的代码和测试不得读写它。三条验收命令都先把未跟踪的
`test_single_build_path.py` 移出 `tests/shared/`；文件还在时验收还没开始。

### P0：单路径与真实 Metal

从 HEAD 改，不把未提交草稿当作已经完成。

- `Backend` 删除 `compile` / `compile_all`，增加
  `artifact_builder(*, kernel_contract, math="fast", kernel_names=None)`；
  `MetalBackend` 删除这两个方法，把三项交给 `MetalSourceBuilder`（HEAD 的构造
  函数已有 `kernel_contract` 与 `math`，要补 `kernel_names`）。
- `build()` 自己 `assemble`，然后只调一次 `context.compile` 或一次
  `context.compile_all`。`spec is None` 时 `math="fast"`、只编译契约核名；有
  spec 时 `math=spec.math`、名字去重保序（与 HEAD `evaluate.py:419` 相同）。
  `evaluate.py` 与 `xcworker.py` 都改走这个入口；`xcworker` 不再自己 `assemble`，
  但仍按 `spec.stages` 的顺序取 pipeline 做 `run` / `run_pipeline`。
- 中立协议不接收 context：`ArtifactBuilder.build` 与
  `ArtifactBuilderRegistry.build` 保持 HEAD 的三参数，`MetalContext` 只出现在
  Metal 的实现里。上下文归后端所有：`MetalBackend.context()` 返回
  `get_context()` 的单例（`evaluate.py:347` 的 `ctx` 就是同一个对象），builder
  直接用；构造时也可显式收下一份，供测试与探针定向使用。唯一固定源码入口是
  `compile_fixed(source, kernel_names, *, context, math="fast")`：Metal 专用、
  context 必填、不 `assemble`、不返回 `BuildOutcome`。`probes.py` / `profile.py`
  把已有的 `ctx` 传进来，不把 HEAD 的两次单独编译合成一次。生产树里除
  `runtime.py` 与 `source_builder.py` 外不再有 `ctx.compile(` / `backend.compile(`。
- 一个 `BuildOutcome` 只带一个 `NativeExecutable`（`kind=compute-pipeline`）。
  单核 handle 就是那个 pipeline；多阶段 handle 是只含 `kernels`（按
  `kernel_names` 顺序的 tuple）的进程内容器，不序列化、不进 digest。descriptor
  必须覆盖每个阶段（少记一个阶段要改变 `native_digest`），bindings 按
  `(index, kind)` 排序后写入，顶层字段仍描述 `kernels[0]`、`pipeline_s` 仍是各核
  之和。超过现有的 32 字段 / 8192 字节上限时抛 `BuildError("toolchain_profile")`，
  不返回 outcome、不截断。P0 新增 `executable_handles()`：有 `kernels` 时返回
  这个 tuple，否则返回 `(handle,)`。
- 编译失败只分类一次：builder 调用一次 `rewrite_compiler_lines`，把改写后的
  文本放进新增的 `BuildError.report_detail`（evaluator 用它填
  `compile["errors"]`，与 HEAD 逐字一致），`detail` 仍是带类名的有界日志行。
  evaluator 只捕获 `BuildError`：`compile` → `compile_error`（文案不变）；
  `toolchain_profile` 重新抛出（由 `worker.py:136` 变成框架 `crash`）；
  `deadline` → `timeout`；其他异常照旧落到框架 `crash`。evaluator 自己不再改写
  诊断，也不把 `str(exc)` 写进 `compile.errors`。
- `tests/support/npu_sim/fake_backend.py` 的 `FakeNpuBackend` 实现同一个中立
  `artifact_builder` / `build`，context 是它自己的普通对象，不注解
  `MetalContext`。`tests/shared/test_backend_neutral.py` 在评测过程中仍不得
  import Metal。
- 不增加编译次数，也不为填 `build` 在父进程重编（见 §7）。

验收（无 GPU）：

```bash
uv run --no-sync pytest -q -n 0 -m 'not gpu and not llm and not slow' \
  tests/shared/test_structure.py \
  tests/shared/test_build_path_structure.py \
  tests/shared/test_backend_neutral.py \
  tests/backends/metal/test_source_builder.py
```

`test_build_path_structure.py` 只读源码，不 import `tcenv.backends`，断言没有
第二个生产编译入口、`artifact_builder` 收下 `kernel_names` 与 `spec.math`、
`compile_fixed` 的 context 归属。`test_source_builder.py` 里不带 `gpu` 的用例按
新入口改写，至少保留：只有 Metal 的编译错误算 `compile` 失败；handle 不进
digest 与序列化；一个 outcome 只有一个 `NativeExecutable`；改写过的诊断只出现
一次且不带类名；`compile_fixed` 只编译传入的 context；超限 descriptor 得
`toolchain_profile` 且不启动设备。

真实 Mac 另获 GPU 许可后跑 `tests/backends/metal/test_build_path_metal.py`
（`-m gpu`），三条都要有，不能只测单核：

- 无 `taic-launch` 的有效 MSL 经 `tcenv.evaluate` 得到一个 pipeline；无效 MSL
  得到 `compile_error`。
- 两段 `taic-launch`（两个核名，`math safe`）经 quick 路径：`executable_handles()`
  长度为 2，descriptor 的 `math` 为 `safe`，`kernels` 覆盖这两段。
- 同一份两段源码经 `tcenv.xcworker` 构建并启动一次（不必套 `xctrace`）：两个核
  都进入 `run_pipeline`，输出里的核名与去重列表一致。

同一份许可下再跑 `tests/backends/metal/test_source_builder.py -m gpu`：不带
context 的 `registry.build(...)` 也要能编过，
`test_invalid_msl_is_a_typed_compile_failure_not_infrastructure` 仍是
`BuildError.error_class == "compile"`，并新增跨进程一致性（两个独立进程对同一
份单核与两段源码各 `build` 一次，`native.digest` 与 `artifact_set_digest` 相等，
handle 不跨进程比较）。做不到就标 blocked，不用假 Metal 填；不写加速比。

### P1：digest 函数

在 `BuildProfile` 上增加 `build_profile_digest()`：
`SHA-256(canonical_digest_document(self.capability_document()))`。不改
`capability_document()` 的键，不读文件，不碰设备。测试对
`canonical_digest_document` 的字节钉一份已知答案，并断言改 `toolchain.version`
或 `compile_profile.device` 后 digest 变、文档不含 `source_sha256` 与
`artifact_set_digest`。

```bash
uv run --no-sync pytest -q -n 0 \
  tests/shared/test_build_profile_digest.py \
  tests/shared/test_structure.py
```

测试只 import `tcenv.remote`。

### P2：报告、缓存键、确认门，同一提交

- `REPORT_SCHEMA_VERSION` 改为 4；`EvalReport` 增加 `build`，读入按 §4、写出按
  §4.1（不新增按调用点选版本的参数）。`origin` 与 `_save` 按 §5.1，`_confirm`
  按 §5；`_timeout_report` 按 §4.1（默认写入 4 + `null`），`_validate_report`
  不动。
- `EVAL_CACHE_SCHEMA` 改为 3；键的其余字段不增加，尤其不要加入 `build_profile_digest`。
- 更新被整数版本钉住的断言：`test_report_schema.py` 里过新改成 5，旧文档往返
  保持原版本（0–3 无 `build`、0–3 带 `build` 拒绝、4 缺键/多键/full 含 `null`/
  partial 含非空 `native_digest` 都拒绝）；`test_orchestrator.py:68` 的候选
  `report.json` 改成 4（`result.json` 的 schema 1 不改）；
  `test_backend_phase0.py:49-52` 拆成两套断言——文件里的
  `report_schema_version` 仍是 3、`top_level_fields` 仍是那份列表，当前
  `REPORT_SCHEMA_VERSION` 是 4、`fields(EvalReport)` 比它多出 `build`。不改
  `docs/架构/backend-phase0-baseline.json` 的任何字段；要记 schema 4 的字段表就
  另起新文件，写明它不是 2026-10-23 那次测量。

```bash
uv run --no-sync pytest -q -n 0 -m 'not gpu and not llm and not slow' \
  tests/shared/test_report_schema.py \
  tests/shared/test_build_profile_digest.py \
  tests/shared/test_eval_cache.py \
  tests/shared/test_confirm_build_identity.py \
  tests/shared/test_structure.py \
  tests/shared/test_orchestrator.py \
  tests/shared/test_evaluation_outbox.py \
  tests/shared/test_publication.py \
  tests/backends/metal/test_backend_phase0.py \
  tests/shared/test_remote_npu_grpc.py \
  tests/shared/test_remote_npu_build_grpc.py
```

`test_confirm_build_identity.py` 不 import backend，至少覆盖：`local-resume`
且搜索期是 schema 4 时仍要比五个 digest（不同即失败，两份相同则中位数是其中
一份），不能因 origin 绕开；`local-new` 的搜索期是 schema 3、或 batch 为 `null`
时 `build_identity_mismatch`；换 winner 只对过该候选自己的搜索期报告；
`remote-new` 的确认按 §4.1；`test_eval_cache.py` 断言键的 `schema` 为 3 且没有
`build_profile_digest`。

另外在本提交里：

```bash
git grep -n 'schema_version"\] == 3' tests
git grep -n 'REPORT_SCHEMA_VERSION == 3' tests docs
```

命中项在同一提交改掉，或说明它指的是历史基线文件里的 3 而不是当前写入版本。
两条 grpc 测试里，新建的 JobEngine 超时报告断言 schema 4 + `build: null`，证据
里的 `build.status == "not_run"` 不变；重放的旧报告不改字节（§5.1）。
`test_publication.py` 仍只断言 `_check_publication`。
`tests/support/golden/` 里已有向量的字节和 HMAC 必须与 HEAD 逐字节相同（含
`tcenv.eval-report.v3` 这个字符串）；schema 4 的覆盖用新文件，不重算、不替换
旧 golden。旧测试若因新字段变红，修的是生产路径不要把 schema 4 写进旧向量。

真实 Mac 的 gpu 测试仍是 P0 那几条，外加一条：quick 评测的 full `build` 五个
digest 来自同一次 outcome，`compile_s` 仍是 kernel 对象上的值。

## 7. 成本，以及什么时候才重做基线

已有编译次数不变：`mode=full` 与「`timed` 且要求校验」先 validation 再
measurement worker，各编译一次；经 `xcworker` 做 profiling 是第三个进程；带宽
探针是另一段固定源码。P0 只替每次的入口，不在父进程再编译一次来填 `build`。

缓存升到 schema 3 后，旧 ok 文件还在磁盘上，新键不读它们：新搜索要重编、重测。
这是缓存冷启动，不是发布基线推倒重来；旧 run 已确认的 speedup 保持原字节。
旧数据审计就是 §5.1 的双文件 sha256、§4.1 的无设备远端回归、
`test_evaluation_outbox.py` 的完成行重放字节不变。

不重建测量基线的理由：本文不改计时窗口、确认 batch 的统计定义、目标限额或
`PROTOCOL_VERSION`，`backend-phase0-baseline.json` 整份保持原字节（含 `gpu`
段、`report_schema_version: 3` 和当时的 `top_level_fields`）。必须重做基线的
条件是其中之一变了、需要升 `PROTOCOL_VERSION` 或 evaluator epoch；在那之前，
不要把新数字写成旧基线的替代，也不要宣称 schema 4 让已发布加速比更准确。

尚未被证据支持的说法：没有测量证明「同类源码、不同 MSL 命中同一缓存」发生过。
schema 3 的键仍不含 MSL / macOS。确认阶段因 `eval_cache=False` 会重测，所以只有
搜索期报告是 full 时 digest 不一致才会挡住发布；未跑确认时，`best_speedup` 仍
可能来自另一套工具链的缓存。单路径在 HEAD 上还不存在，不能把草稿说成已经
收益。本文没跑 gpu 测试。

## 8. 仍不在本文决定的事

- 缓存命中前是否 probe 设备：决定是不做（§7）。要做是另一次行为变更。
- native digest 要不要进入某一版 NPU provenance：要做得新的 document version，
  且不能改 v5 的字节，归通信计划。
- 旧 run 目录留多久：本文不删数据。
- 实现级细节留到开工时用测试定：partial 具体附哪些 `BuildError` 字段、多阶段
  descriptor 的键序与命名、`to_json` 在 4 + ok + 非 full 时是否主动抛错、远端
  将来要不要自己的 digest 门（本文的答案：不要）。
