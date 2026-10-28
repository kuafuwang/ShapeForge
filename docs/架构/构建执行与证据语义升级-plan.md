# 构建、执行与证据语义升级计划

状态：设计计划，2026-10-28。**尚未实施。** 代码基线是
`2e02088909799d801cc7c07696335ac2e33b42c5`。其后的 `65717ad` 与 `b95043b`
只改了别的文档，没有改 `tcenv/`、`taic/`、`tests/`。行号按这份代码核对，
不按工作区里未提交的 diff。

通信计划拥有六 RPC、BlobChannel、VendorDriver、fake TSM、真实 NPU 的 wire
contract，以及 NPU 证据能不能发布。本文只规定本地 Metal：一条构建路径，
以及本地新报告的 schema 4。不编造 Metal native full 去填远端报告。

已定方向：生产构建只保留 `backend.artifact_builder()` → `ArtifactBuilder.build`；
删除 `Backend.compile` / `compile_all` 与 `MetalBackend` 上的同名方法，不做
委托包装。旧文件只读，不回填、不改写、不重签。已经落盘的文件只在读取时按
它自己的整数版本解释。本计划新写的数据只有本地 Metal 的 schema 4。远端 /
NPU 不在这个写入范围内，也不把「继续写 schema 3」称作本文的新数据路由。
不统一各厂商 ABI、launch 或容差。Metal 验收必须跑真实设备上的
`MetalSourceBuilder`，不能用替身类代替。

工作区里已有一份未提交草稿（`evaluate.py`、`base.py`、`metal/backend.py`、
`source_builder.py`、`artifact_builder.py`、`xcworker.py`，以及未跟踪的
`tests/shared/test_single_build_path.py`）。它**不是**本计划的前置依赖，
也没有验收记录。实施从 HEAD 的行为写起。草稿可以当参考，但不能当成已经落地；
其中的 `BuildOutcome.executable_handles` 在 HEAD 的 `BuildOutcome` 上不存在。

## 1. HEAD 上实际有什么

用 `git show 2e02088:路径` 核对。下面这些还在，P0 之前不能写「单路径已落地」。

| 位置 | HEAD 行为 |
| --- | --- |
| `tcenv/evaluate.py:417-421` | `backend.compile` / `backend.compile_all`。没有 `artifact_builder()` |
| `tcenv/backends/base.py:332,343` | 协议上仍有 `compile` 与 `compile_all` |
| `tcenv/backends/metal/backend.py:389,400` | 二者委托给 `ctx.compile` / `ctx.compile_all` |
| `tcenv/xcworker.py:50,60` | 直接 `ctx.compile` / `ctx.compile_all` |
| `tcenv/backends/metal/probes.py:39`、`profile.py:87` | 直接 `ctx.compile`。HEAD 没有 `compile_source` |
| `tcenv/backends/metal/source_builder.py:209` | `MetalSourceBuilder.build` 仍调用 `self.backend.compile`。evaluator 不调用这个类 |
| `tcenv/remote/artifact_builder.py:457` | 已有 `BuildOutcome`（bundle、provenance、evidence）。没有 `executable_handles` |

已有、本计划不重做的部分：`BuildProfile.capability_document()`、
`artifact_set_digest`、`NativeExecutable.digest`、`SourceBundle.sha256`、
NPU `document_version` 1–5。本地 `EvalReport` 仍是
`REPORT_SCHEMA_VERSION = 3`，缓存键里的 `EVAL_CACHE_SCHEMA = 2`。

`tests/shared/` 在 HEAD 上不包含 `test_single_build_path.py`。那份未跟踪文件
在函数里 import `tcenv.backends`，所以只要它留在 `tests/shared/`，
`tests/shared/test_structure.py::test_backend_free_tests_do_not_import_a_backend`
就是红的。它断言的是草稿，不是 HEAD。不要把它提交进 `tests/shared/`。
P0 另写测试。开始 P1 或 P2 的验收时，`tests/shared/test_*.py` 不得 import
`tcenv.backends`；否则验收视为还没开始，而不是「已知失败可忽略」。

## 2. 目标与非目标

### 目标

1. P0 之后，evaluator 与 xcworker 只经 `ArtifactBuilder.build` 构建。
2. 一次成功的 `BuildOutcome` 带三种构建身份：候选源码、build profile、工件集
   （native digest 属于这次工件，不另算一种执行环境）。执行环境仍是
   `EvaluationResult.provenance`，不放进 `BuildOutcome`。
3. 本计划新写的本地 Metal 报告用 schema 4 记下本 worker 是否拿到了
   outcome。读入的旧本地文件保持原版本。远端 / NPU 的报告与发布不在这个
   目标里，见 §4.1。
4. 缓存键升到 schema 3，使升级前的 ok 报告不再被这条键打开。键里仍然没有
   现场工具链。

### 非目标

- 不改 NPU `document_version` 1–5 的字段、digest 或 HMAC，不新增第 6 版
  wire contract。字符串 `tcenv.eval-report.v3`（`REPORT_SCHEMA_ID`）与报告文件
  里的整数 `schema_version` 不是同一个版本，不要跟着改成 v4。
- 不改 `PROTOCOL_VERSION`（`"3"`），不把 build profile 塞进
  `EvaluatorProvenance` 的 epoch 输入。计时窗口不变，所以不重录
  `docs/架构/backend-phase0-baseline.json` 的 `gpu` 段，也不重签旧 run。
- 不改 Metal 的 grid、容差、设备锁、超时和 launch。
- 不把 `rep.compile["compile_s"]` / `pipeline_s` 改成从
  `BuildEvidence.duration_ms` 反推。
- 不把 `SourceBundle` 的默认路径 `kernel.tx` 改成 `kernel.metal`。
- 不改编译失败文案 `Metal compiler rejected the kernel`。
- 不规定旧 run 的保留期限，不写迁移脚本，不改 `runs/`。
- 不在缓存命中路径上调用 `MetalContext.info()` / `probe()`。

## 3. 三种构建身份，加上已有的执行环境

构建身份来自同一次 `BuildOutcome`。执行环境不在这个对象里。

| 身份 | 算法（HEAD 已有） | 放在哪 |
| --- | --- | --- |
| 候选源码 | `SourceBundle.sha256`，即 `SHA-256(候选 UTF-8)` | 报告 `build.candidate_sha256`。不是 assembled 文本 |
| build profile | **新函数**，见 P1。材料是已有的 `capability_document()`：`builder_id`、`builder_version`、`build_host`、`build_profile`、`toolchain`、`compile_profile`。不含 source，不含 artifact | 报告 `build.build_profile_digest`。Metal 的 `compile_profile` 今天可以含 `device` / `gpu_family`，digest 会随设备名变。不要为了「看起来干净」删键：该文档已进入 NPU v5 的逐字段比较 |
| 工件集 | `outcome.artifact_set_digest`：`SHA-256(RFC8785(canonical_manifest))`。native handle 不在这个 digest 里 | 报告 `build.artifact_set_digest`。`build.native_digest` 另抄 `NativeExecutable.digest`。HEAD 的 `provenance_document` 只有 `source_sha256` 与 `artifact_set_digest`，**没有** native digest。不要把 native digest 补进 v5 provenance |

`assembled_sha256` 是契约头加候选源码之后的 SHA-256，记在报告里，但不代替
`candidate_sha256`。

执行环境是 `tcenv/evaluation.py` 的 `EvaluatorProvenance`（schema 1）。
`local_provenance` 的 epoch 输入是 site、device、backend、`backend_version`、
target、worker_build、task_pack、calibration、`protocol_version`。本地
worker_build 与 calibration 是空串。`MetalBackend.version` 是 `"1"`，不是
MSL 版本。发布门继续用这份 provenance（`measurement_kind`、
`publication_allowed`、site/epoch 白名单）。`build` 对象不参与 epoch，
也不替换它。simulated 仍不能进搜索。

## 4. schema 4 的 `build`：按有没有拿到 outcome

不按 `status` 推断。`rejected` 在编译前和编译后都会出现。`timeout` / `crash`
经常是父进程新建的空报告。

形状只有三种，外加 schema ≤3 的「键不存在」：

- **键不存在**：只允许 schema ≤3。`from_json` 先读整数版本（缺键为 0），
  再按该版本决定是否接受 `build`。≤3 若带 `build` 键，按未知字段拒绝。
  拒绝发生在构造 `EvalReport` 之前。通过之后，文件版本留在对象上，供
  `to_json()` 和确认使用；内存里的 `build` 字段是 `None`。
- **`null`**：这份报告的生产者没有本 worker 的 `BuildOutcome`，也不是下面的
  partial。JSON `null`。
partial 与 full 都是下面这 8 个键，不能多也不能少。`shape` 算一个键。

| 键 | partial | full |
| --- | --- | --- |
| `shape` | 字符串 `partial` | 字符串 `full` |
| `candidate_sha256` | 64 位小写 hex | 64 位小写 hex |
| `assembled_sha256` | hex，或 JSON `null` | hex |
| `build_profile_digest` | 见下 | hex |
| `artifact_set_digest` | JSON `null` | hex |
| `native_digest` | JSON `null` | hex |
| `builder_id` | 见下 | 非空字符串 |
| `build_profile` | 见下 | 非空字符串 |

`builder_id`、`build_profile`、`build_profile_digest` 在 partial 里要么三个都是
JSON `null`，要么前两个是非空字符串且 digest 是 hex。不允许只填其中一两个。
partial 里非空的 profile / assembled 只能来自同一次失败里已经附在
`BuildError` 上的值。`except` 里禁止再 `probe()` 或再编译。P0 可以给
`BuildError` 加可选附件；不附则这三个 profile 字段与 `assembled_sha256` 都是
`null`，但 8 个键仍然都在。full 的 8 个值只从这一次 `BuildOutcome` 抄。

schema 4 的顶层必须有 `build` 键。schema 5，或对象里多出来的键，都拒绝。
`null` 只用于这份报告没有 outcome：编译前拒绝、父进程新建的 timeout/crash、
以及 `unchanged` / `duplicate` / `external_duplicate`。这些 status 不是 `ok`。
schema 4 且 `status=ok` 必须是 full。`build: null` 的 schema 4 `ok` 不能进入
本地确认。schema ≤3 且没有 `build` 键的兼容路径只用于续跑读入的历史本地
文件，见 §5。远端 / NPU 新报告不走这条兼容路径。测试里的假
`EvalReport(status="ok")` 不是生产例外：要让编排测试仍能走到确认成功，夹具
必须自带一份合成的 full `build`（固定 hex，不编译 Metal）。生产 `_confirm`
不判断自己是不是测试。

### 生产者对照（HEAD 行号）

| 生产者 | 拿到 outcome 了吗 | schema 4 写什么 |
| --- | --- | --- |
| `evaluate_in_process` 编译前返回：`evaluate.py:389,406,408`，以及父进程 `evaluate.py:1224,1260,1268` 的 device_check / screen | 否 | `null` |
| 编译失败：`evaluate.py:425` `finish("compile_error")` | 否，失败记录 | partial |
| 编译已经成功之后的所有 `finish`，包括 `abi_error`、`unsafe`、`incorrect`、`runtime_error`、`ok`，以及分配失败的 `rejected`：`evaluate.py:465,559,575` | 是 | full。这三处是构建之后的 `rejected`，不能写成 `null` |
| 父进程杀掉 worker：`evaluate.py:941-945` 新建 `base`，status `timeout` | 父进程没有 outcome | `null`。不要读半份 report.json 来补 |
| 父进程发现没有报告或报告损坏：`evaluate.py:950-967` | 否 | `null` |
| 父进程 `EvalReport.from_json` 成功：`evaluate.py:949`，以及随后可能把 status 改成 `unsafe` 的 `969-976` | 沿用子进程写好的 `build` | 不改 `build`，不清成 `null` |
| `tcenv/worker.py:73` 请求读失败；`worker.py:141` 框架异常 | 否 | `null` |
| `tcenv/remote/job_engine.py:1836` `_timeout_report` | 否。本文不把这里当成 Metal outcome | 本文不改这个生产者的报告版本。隔离约束见 §4.1：不要被全局 schema 4 带成带 `build` 的新报告。证据 JSON 里的 NPU `build`（例如 `status=not_run`）仍属通信计划 |
| `taic/orchestrator.py:2110,2118,2130` 的 `unchanged` / `duplicate` / `external_duplicate` | 否，不上 GPU | `null` |
| `tests/support/fake_worker_*.py`、`tests/support/orchestrator.py`、`tests/support/backend_neutral_driver.py` 以及各测试里直接 `EvalReport(...)` | 否 | 不进入 `_confirm` 的报告可以是 `"build": null`。凡是 `status=ok` 且测试期望确认成功的夹具，在测试里写上合成 full（固定 hex）。生产代码不为这些夹具开旁路 |
| 厂商 / NPU worker 交回的成功报告 | 本文不读取它的 `BuildEvidence` 来填 `EvalReport.build` | 本文不改这份报告的 schema，也不编 Metal 的 8 键。`JobStore._validate_report`（`job_store.py:2345`）继续接受它今天的形状 |
| outbox（`taic/evaluation_outbox.py`） | 不自己构建 | 保存调用方给的报告字节。不发明 `build`，不把 schema 3 行改写成 4 |

### 4.1 新语义的边界：只写本地 Metal

`REPORT_SCHEMA_VERSION` 升到 4 之后，它是两件事：`from_json` 接受的最大
版本，以及**本地 Metal 新报告的写入版本**。读取旧文件时按文件里的整数
分支。这不是两条都在写新数据的生产格式。

本计划新写的报告只有这些：

- `evaluate.py` 新建的本地 Metal 报告：版本 4，顶层始终有 `build`
  （`null` / partial / full，按 §4）。`status=ok` 只能是 full。本地搜索里
  不上 GPU 的复制报告（`unchanged` / `duplicate` / `external_duplicate`）
  同样写 schema 4 且 `build` 为 `null`；它们的 status 不是 `ok`。
- 从历史 schema ≤3 读入的对象：`to_json()` 仍写出那个整数，且不加入 `build`。
  这只约束序列化结果，避免误写时升级。它不是写回旧路径的许可，见 §5.1。
  本地实施之后不得再新写 schema ≤3 的候选报告。

远端 / NPU **不在本计划的新语义里**。不要把它们的 `BuildEvidence` 映射成
Metal 的 8 键 full，也不要新规定它们「继续写 schema 3」作为本文的路由。
它们的报告字节保持各自今天的生产者。P2 只加一条隔离：`EvalReport.to_json()`
的默认版本不得把 `JobEngine._timeout_report` 或厂商交回的文档改成 schema 4
或加上顶层 `build`。证据 JSON 里已有的 NPU `build` 对象保持通信计划的字段。

这些结果的发布门也不换成 §5：

- 能不能进入搜索，仍只由 `Orchestrator._check_publication`
  （`taic/orchestrator.py:1443-1453`）决定：`measurement_kind == "measured"`、
  `provenance.publication_allowed`、site 白名单、epoch 白名单。这个函数不看
  `EvalReport.build`。
- NPU 设备证据能不能当作可发布结果，仍由通信计划里的证据
  `publication_allowed`、receipt，以及 `JobStore._validate_report` 决定。

若某条远端执行调用的就是这份 `evaluate.py`（Metal worker，不是 NPU 适配器），
那份报告是本地 Metal 语义，写 schema 4。`_timeout_report` 不是这条路径。

`from_json` 同时接受历史 schema ≤3（没有 `build`）和 schema 4（`build` 符合
闭集）。schema 3 带 `build`、schema 4 缺 `build`、schema 4 `ok` 不是 full，
都拒绝。§5 的确认只处理本地 Metal 的 schema 4，以及续跑读入的历史本地文件。
不把远端新报告放进「schema ≤3 所以走旧发布」这一支。

`full` 与 `timed` 且需要校验时，HEAD 的 `evaluate()` 先跑 validation worker，
再跑测量 worker。两份进程各构建一次。测量报告只带测量 worker 的 outcome。
validation 那次如果失败并被返回，带的是 validation worker 自己的 outcome。
不要把两次耗时或两份 digest 合成一条。

## 5. 确认循环：每个候选、每个成功 batch

对照 `taic/orchestrator.py` 的 `_confirm`（约 3191–3256 行）。一个候选有
`batches` 次复测。只有 `rep.status == "ok"` 且有 `speedup` 的 batch 才会进入
`values` / `reports`。全部 batch 都成功后，中位数是 `statistics.median(values)`，
留下的报告是 `min(reports, key=lambda r: abs(float(r.speedup) - median))`。
然后 `choose_confirmed` 可能把 winner 从 `self.best` 换成池里另一颗。

`confirmation_source_sha256` 要到 `_finish` 的 3438–3441 行、且
`confirmed_status == "ok"` 时才赋值。确认循环里还没有这个字段。循环内用
`hashlib.sha256(c.source.encode("utf-8")).hexdigest()` 重算，命名为局部变量
`source_sha256`。`_finish` 里那一行保持不动，继续从
`_confirm_winner.source` 计算。

在 `if rep.status == "ok" and rep.speedup` 为真、`values.append` 之前插入
身份检查。搜索期报告是 `c.report`。`from_json` 必须把读到的整数版本留在
报告上（缺 `schema_version` 记为 0），并记住这份对象是不是续跑时从已有
`report.json` 读入的。不能只看 `build is None`。

本节只处理本地候选。远端 / NPU 结果的发布门是 §4.1 的
`_check_publication` 与通信计划，不进入下面的 digest 比较，也不因文件版本
≤3 被当成历史本地文件放行。本进程从远端执行器拿到的报告若出现在确认池里：
不 append，走下面的失败出口，`entry["status"]` 写 `not_local_build_report`。

- 搜索期报告是续跑读入的历史本地文件（schema ≤3，没有 `build`）：不拿它和
  batch 比 digest。本轮本地复测写出的 batch 必须是 schema 4 full，
  `candidate_sha256` 等于 `source_sha256`，且同一候选各个 batch 的五个
  digest 彼此相同：`candidate_sha256`、`assembled_sha256`、
  `build_profile_digest`、`artifact_set_digest`、`native_digest`。schema 4
  的 `null` 或 partial 不得 append。本计划不新写 schema ≤3 的确认 batch。
- 搜索期报告是 schema 4 且曾以 `ok` 写入：必须已是 full。`build` 为 `null`
  或 partial 时，这个候选直接走下面的失败出口，不得跳过检查。batch 也必须
  是 schema 4 full，五个 digest 与搜索期报告相同，且两边的
  `candidate_sha256` 都等于 `source_sha256`。schema ≤3 的 batch 不能配一份
  schema 4 的搜索期报告。
- 搜索期报告缺失，或 schema ≤3 但不是续跑读入的历史文件：不当作旧数据。
  走失败出口，`entry["status"]` 写 `not_local_build_report`。
- digest 不符时 `entry["status"]` 写 `build_identity_mismatch`。上面已经
  写明 `not_local_build_report` 的情况不要改成这个字符串。两种都走现有的
  失败出口，不要新开发布通道：调用已有的 `_failure_record`，写入
  `rechecks/confirm_<label>.json`，`failed_batches` 加一，`stop = True`
  并 `break`。不要写成 `ok`。不要改 `rep.status`：
  `_failure_record` 只在 `incorrect` / `unsafe` / `timeout` / `crash` 时
  把记录的 `class` 设成该 status，其余（包括仍为 `ok` 的身份不符）已经落成
  `"other"`。不要因此增加 `hangs`（`hangs` 只在 `rep.status == "timeout"`
  时加）。外层 `if stop: break` 会停掉后续候选，与今天某个 batch 失败的
  行为相同。前面已经成功的 batch 不能把这颗核标成确认成功，这段现有逻辑保持。

每颗候选只和自己的搜索期报告比。`choose_confirmed` 换 winner 时，不再做
跨候选的 digest 比较：胜者进入 `confirmed` 之前，它自己的每个成功 batch
已经对过它自己的 `c.report`。留下的那份中位数报告是这些已通过 batch 之一。

`eval_cache=False` 保持。

### 5.1 旧 run 续跑：禁止写回，以及 outbox

`taic/resume.py:563` 用 `EvalReport.from_json` 读 `candidates/*/report.json`。
schema 0–3（缺版本视为 0）读入后只保留旧语义：内存对象没有 `build`，
文件版本记为读到的那个整数。这次读入不写盘。

已有候选的 `report.json` 不得重写。版本仍是 3、没有 `build`，也不允许写回。

- `resume.py` 只读。读入后不要调用 `to_json()` 把结果写回同一路径。
- `orchestrator._save`（`taic/orchestrator.py:1302`）若目标 `report.json`
  已存在且当前对象是从该文件读入的，直接返回，不写 `report.json`。禁止先改
  `benchmark["seed_gpu_us_median"]` 再序列化回这份旧文件。其他路径同样不得
  替换这份字节，包括先读入再原子替换。
- `_session_brief`（约 2908 行）可以在内存里序列化一份给会话提示用的副本。
  该副本不得写到 `candidates/*/report.json`。
- 只有新的本地 Metal worker 结果才能写**新的** `candidates/*/report.json`
  （该路径上还没有文件），版本为 4。不要打开旧目录覆盖旧报告。远端 /
  NPU 执行器自己的结果文件不由本文规定新 schema；本地续跑也不得把它们
  抄成一份新的 schema 3 候选来走 §5 的历史分支。
- 同一 run 里因此可以同时有续跑留下的历史 schema ≤3 文件，和本次新写的
  schema 4 文件。确认按 §5：历史文件必须是续跑读入的那一份；新的本地
  `ok` 必须是 full。不按 run 的年龄，也不按「凡是 schema 3 都兼容」。

outbox 里 `status=completed` 的行只重放存储内容。`complete()` 不得改写
已完成行（现有 `outbox_conflict` / immutable completed）。重放出来的报告
保持它写入时的 schema，不经过「读入再 `to_json()` 存回」。pending /
unavailable 也不得在本地把已存报告升级成 schema 4；若以后重新执行，那是
一次新的 worker 结果，写新的结果文件，不覆盖已完成行。

## 6. 阶段

P0 可以先于 P2 合入。P0 只改构建入口、诊断文本和 `status`。`EvalReport`
上还没有 `build` 字段，P0 的代码和测试都不得读写它。partial / `null` /
full 要到 P2 才写入。P1 只依赖 HEAD 里已有的 `BuildProfile`，可以在 P0
之前合入，但验收命令包含 `test_structure.py` 且必须全绿。工作区若仍有
未跟踪的 `tests/shared/test_single_build_path.py`，先把它移出 `tests/shared/`
（不要提交进该目录）再跑 P1。这是 P1 的开始条件，不是 P1 里允许的红。

### P0：单路径与真实 Metal

从 HEAD 改，不把未提交草稿当作已经完成。

- `Backend` 删除 `compile` / `compile_all`，增加 `artifact_builder()`。
- `MetalBackend` 删除这两个方法。`MetalSourceBuilder.build` 改为调用
  `MetalContext.compile` / `compile_all`，不再调用 `backend.compile`。
- `Backend.artifact_builder(*, kernel_contract, math="fast", kernel_names=None)`
  把这三项交给 `MetalSourceBuilder`。HEAD 的构造函数只有 `math`，没有
  `kernel_names`；P0 要补上。`evaluate.py:416-420` 的分支原样变成这两项，
  不再调用 `backend.compile`：
  - `spec is None`：`math="fast"`，`kernel_names=None`。builder 只编译
    契约里的 `kernel_name`，一次 `MetalContext.compile`。
  - `spec is not None`：`math=spec.math`，
    `kernel_names=list(dict.fromkeys(stage.kernel for stage in spec.stages))`
    （与现在 419 行的去重顺序相同）。builder 对拼好的源码调用一次
    `MetalContext.compile_all(assembled, names, math=math)`。
- `xcworker.py` 使用同一调用。HEAD 在 39 行先 `assemble`，50/60 行再
  `ctx.compile` / `compile_all`。P0 改为把候选原文放进 `SourceBundle`，
  由 builder 自己 `assemble`。`math` 与 `kernel_names` 的规则与 evaluator
  相同（无 spec 时 math 为 `fast`、names 为空）。随后仍按 `spec.stages`
  的顺序取 pipeline 做 `run` / `run_pipeline`，顺序不改成去重顺序。
- 一个 `BuildOutcome` 只带一个 `NativeExecutable`（`kind=compute-pipeline`）。
  单核时 `handle` 就是那一个 `CompiledKernel`，descriptor 保持 HEAD
  `source_builder.py:264-281` 已有的键，不另加 `kernels`。
  多阶段时 `handle` 是进程内对象，只含属性 `kernels`：按 `kernel_names`
  顺序的 tuple，不序列化、不进 digest。descriptor 在单核那组键之外再加且只加
  两个键：`kernel_names`（去重后的名字列表）和 `kernels`（等长数组，每项有
  `name`、`max_total_threads`、`execution_width`、`static_tg_bytes`、
  `bindings`）。顶层的 `max_total_threads` / `bindings` 等仍描述
  `kernels[0]`，与今天 `kernel = kernels[0]` 一致；`pipeline_s` 仍是各核
  `pipeline_s` 之和。少记一个阶段必须改变 `native_digest`。
  `BuildOutcome.executable_handles()` 在 HEAD 不存在，P0 添加：有
  `handle.kernels` 时返回该 tuple，否则返回 `(handle,)`。handle 不进任何
  digest。descriptor 顶层字段数必须仍低于现有的 32 个上限。
- `build()` 与下面的 `compile_fixed` 都接收调用方已经拿着的
  `MetalContext`，类型为必填参数 `context`，没有默认值。编译只调用这个
  对象的 `compile` / `compile_all`。函数体内禁止 `get_context()`、
  `backend.context()` 和 `self.backend.context()`。HEAD `source_builder.py:201`
  的 `self.backend.context()` 删掉。`evaluate.py:347` 已经取出的 `ctx` 原样
  传入 `build(..., context=ctx)`。`xcworker.py:37` 的 `ctx` 同样传入，随后
  `ArgSet(ctx, ...)` 与 `ctx.run` 用的是同一个对象。
- `evaluate.py:421` 今天捕获的是 `backend.compile_error()`（Metal 上即
  `MetalCompileError`），并在 423 行用当时的 `header_lines` 调用
  `rewrite_compiler_lines`。改由 builder 内部 assemble 之后，evaluator 不再
  持有 `header_lines`。HEAD `source_builder.py:218-235` 已经在构造
  `BuildError` 之前改写过一次。evaluator 不得再改写。
  `BuildError` 增加 `report_detail: str`。`error_class == "compile"` 且来自
  `MetalCompileError` 时：builder 只调用一次 `rewrite_compiler_lines`，
  `report_detail` 等于改写后的编译器文本，不含异常类前缀，也不使用
  `detail` 上 `[:500]` 的截断。`detail` 仍是
  `f"{type(exc).__name__}: {report_detail}"[:500]`，只给日志和 `str()`。
  非 UTF-8（HEAD 183 行）没有编译器行号，`report_detail` 就是
  `Metal source is not UTF-8 text: ...`，同样不带类前缀，也不调用 rewrite。
  evaluator 只捕获 `BuildError`：
  - `compile`：`rep.compile = {"ok": False, "errors": exc.report_detail}`，
    然后 `finish("compile_error", "Metal compiler rejected the kernel")`。
    不调用 `rewrite_compiler_lines`，不把 `str(exc)` 或 `exc.detail` 写入
    `compile.errors`。无效 MSL 走这里。P0 不写 `build`。
  - `toolchain_profile`：不是候选失败。不要写成 `compile_error` 或
    `rejected`。重新抛出，由 `worker.py:141` 变成框架 `crash`。设备不存在、
    契约头拼不出来、编译器语言版本与 probe 不一致，都属这一类（HEAD
    187–198 行与 248–252 行）。P0 不写 `build`。
  - `deadline`：`finish("timeout", …)`，不是 `compile_error`。P0 不写 `build`。
  - `context.compile` 抛出的其他异常不要包成 `BuildError("compile")`。它们
    继续落到 `worker.py:141` 的框架 `crash`。
- 探针不走 `build()`。`build()` 必须 `assemble` 且需要 contract，不能编译
  `probes.py` 的原始 MSL，也不能编译 `profile.py` 里没有契约头的固定源码。
  在 `MetalSourceBuilder` 上增加唯一的固定源码入口：

  `compile_fixed(self, source: str, kernel_names: Sequence[str], *, context: MetalContext, math: str = "fast") -> tuple[CompiledKernel, ...]`

  `context` 必填。不 `assemble`，不接收 contract，不返回 `BuildOutcome`，
  不写 `EvalReport`。`kernel_names` 为空时抛 `ValueError`。长度为 1 时调用
  一次 `context.compile(source, kernel_names[0], math=math)`；多于一个时
  调用一次 `context.compile_all(source, list(kernel_names), math=math)`，
  顺序与参数相同。异常原样抛出 `MetalCompileError`，不翻译成 `BuildError`。
  `probes.py` 在 35 行已经取出的那个 `ctx` 继续用于 `ctx.info()`、
  `compile_fixed(..., context=ctx)` 和 `ctx.new_buffer`。不要为编译再取一次
  singleton。`profile.py` 的 `measure_peak_bandwidth(ctx)` 保持「调用方可以
  传入非默认 context」：`ctx = ctx or get_context()` 之后，`new_buffer` 与
  两次 `compile_fixed(_COPY_SRC, (name,), context=ctx)` 都用这个 `ctx`。
  不要合并成一次 `compile_all`：HEAD 87 行是两次单独编译。除此之外不要再
  添编译入口。生产树里除 `runtime.py`（定义）和 `source_builder.py`
  （`build` 与 `compile_fixed` 内部，且只编译传入的那个 `context`）外，
  不再出现 `ctx.compile(` / `context.compile(` / `backend.compile(`。
- 不增加编译次数。今天 `full` 与带校验的 `timed` 已经是两个 worker，各编译
  一次；xcworker 再编译一次。P0 只替换每一次的入口。带宽探针仍是两次
  `compile_fixed`，次数与 HEAD 的两次 `ctx.compile` 相同。P0 不把 `build`
  写进 `EvalReport`。`compile` / `toolchain_profile` / `deadline` 对应的
  partial 或 `null` 由 P2 在字段存在之后再写，见该阶段。

验收（无 GPU）：

```bash
uv run --no-sync pytest -q -n 0 -m 'not gpu and not llm and not slow' \
  tests/shared/test_structure.py \
  tests/shared/test_build_path_structure.py \
  tests/backends/metal/test_source_builder.py
```

`test_build_path_structure.py` 只读源码，不 import `tcenv.backends`。断言
上一段的调用关系，包括 `kernel_names` 与 `spec.math` 被传进
`artifact_builder`，`compile_fixed` 的签名，以及多阶段 descriptor 含
`kernels`。`test_source_builder.py` 里不带 `gpu` 标记的用例同一条命令跑，
并按新入口改写，至少包括：

- `test_only_the_metal_compile_error_is_classified_as_a_compile_failure`
  （约 168 行）：失败从 `StubContext.compile` 抛出真实的 `MetalCompileError`，
  断言变成 `BuildError("compile")`；其他异常不得变成 `compile`。不再通过
  `backend.compile_error()` 注册一个替身异常类。
- `test_the_native_handle_is_never_hashed_serialized_or_transported`
  （约 592 行）：handle 仍不进 digest、不进序列化。多阶段时 digest 覆盖
  descriptor 里的 `kernels`，不覆盖 live handle。
- `test_one_build_outcome_must_name_one_live_executable`（约 691 行）：
  一个 outcome 仍只有一个 `NativeExecutable`。多阶段的 handle 是那个
  容器对象；`executable_handles()` 返回其中的 pipeline。
- 新增无 GPU 用例 `test_report_detail_is_rewritten_once`：stub 的
  `rewrite_compiler_lines` 对
  `program_source:12:1: error: expected unqualified-id`、`header_lines=11`
  只被调用一次，结果精确等于
  `kernel.metal:1:1: error: expected unqualified-id`（`contract.py:271-274`
  的规则：12−11=1）。`report_detail` 与该字符串逐字相同，且不含
  `MetalCompileError`。`detail` 以 `MetalCompileError: ` 开头且含这段文本
  一次。evaluator 把 `compile["errors"]` 设成 `report_detail` 之后，文本
  仍是这一份，不得变成 `kernel.metal` 被再减一次行号，也不得带上类前缀。
  `evaluate.py` 的 `except BuildError` 分支源码里不得出现
  `rewrite_compiler_lines`。
- 新增无 GPU 用例 `test_compile_fixed_uses_the_given_context`：两个假
  context，`A.compile` 记录调用，`backend.context()` 返回 `B`。
  `compile_fixed(..., context=A)` 只调用 `A.compile`，不调用 `B`，也不调用
  `backend.context()`。`build(..., context=A)` 同样只编译 `A`。覆盖
  `measure_peak_bandwidth` 传入非默认 context 时 pipeline 与 buffer 同属
  一个对象这条约束；测试本身不创建 Metal 设备。

该文件里已有的 `@gpu` 用例不要放进这条无 GPU 命令。真实 Mac 另获 GPU
许可后跑 `tests/backends/metal/test_build_path_metal.py`（`-m gpu`），三条
都要有，不能只测单核：

- 无 `taic-launch` 的有效 MSL（如 relu/fp16/small 的 naive）经
  `tcenv.evaluate` 得到一个 pipeline；无效 MSL 为 `compile_error`。
- 两段 `taic-launch`（两个不同 kernel 名，`math safe`）经 evaluator 的
  quick 路径：`executable_handles()` 长度为 2，descriptor 的 `math` 为
  `safe`，`kernel_names` 与去重后的 stage 名一致，`kernels` 两项的 `name`
  覆盖这两段。
- 同一份两段源码经 `tcenv.xcworker` 的构建与一次启动（不必套 `xctrace`）：
  两个核都进入 `run_pipeline`，输出里的 kernel 名与去重列表一致。

做不到就整段标 blocked，不用假 Metal 填。不要跑 benchmark，不要写加速比。
同一份 GPU 许可下再跑 `tests/backends/metal/test_source_builder.py -m gpu`。
`test_invalid_msl_is_a_typed_compile_failure_not_infrastructure` 仍断言
`BuildError.error_class == "compile"`；经 evaluator 的那条则断言报告
`status` 为 `compile_error`，文案仍是 `Metal compiler rejected the kernel`。

### P1：digest 函数

在 `BuildProfile` 上增加 `build_profile_digest()`：
`SHA-256(canonical_digest_document(self.capability_document()))`。
不改 `capability_document()` 的键，不读文件，不碰设备。

已知答案（对 `canonical_digest_document` 的字节，不是对字段插入顺序）：

```text
{"build_host":"control_host","build_profile":"metal-msl-control-host-v1","builder_id":"metal-source-builder","builder_version":"1","compile_profile":{"device":"Apple M4 Max","gpu_family":"apple9","language":"msl","math":"fast","unified_memory":true},"toolchain":{"id":"apple-metal","sdk_version":"macos-26.0","version":"4.0"}}
```

digest 必须等于 `b45eef1b41395ea37d520aa3097a867b9a13ff9d2cd8b851eb74f63f1c8ba440`。
再断言改 `toolchain.version` 或 `compile_profile.device` 后 digest 变，且文档
不含 `source_sha256` 与 `artifact_set_digest`。

验收：

```bash
uv run --no-sync pytest -q -n 0 \
  tests/shared/test_build_profile_digest.py \
  tests/shared/test_structure.py
```

两条都要通过。测试只 import `tcenv.remote`。

### P2：报告、缓存键、确认门，同一提交

依赖 P0 与 P1。分开合入会出现 schema 4 报告却仍按源码哈希确认。

同一提交：

- `REPORT_SCHEMA_VERSION` 改为 4，含义见 §4.1。`EvalReport` 增加 `build`。
  本地 Metal 按 §4 填写：`compile_error` 写 partial；`toolchain_profile`
  变成的框架 `crash` 与 `deadline` 变成的 `timeout` 写 `null`；成功的
  outcome 写 full。这些是 P2 才有的字段，不回写到 P0。远端生产者按 §4.1
  隔离，不改成 schema 4，也不新增顶层 `build`。
- worker 与父进程按 §4 填写本地报告。历史文件的读入按 §5.1，不回写。
- `EVAL_CACHE_SCHEMA` 改为 3。键的其余字段不增加，尤其不要加入
  `build_profile_digest`。
- `_confirm` 按 §5 改。`from_json` / `to_json` 按 §5.1 保留旧文件版本。
- 更新被整数版本钉住的现有断言，见下一节。
- 不改 `docs/架构/backend-phase0-baseline.json` 的任何字段，包括
  `report_schema_version`、`top_level_fields`、`git_commit`（`ef2a048`）和
  `recorded_at`。那是历史记录。改它来匹配当前代码会伪造基线。

验收（无 GPU、无模型）：

```bash
uv run --no-sync pytest -q -n 0 \
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
  tests/shared/test_remote_npu_build_grpc.py \
  -m 'not gpu and not llm and not slow'
```

`test_confirm_build_identity.py` 不 import backend。至少覆盖：续跑读入的
历史 schema ≤3 上，两个 schema 4 full 且 digest 彼此相同的 batch 可以进入
中位数；其中一份 `build` 为 `null` 则该候选失败；本进程新来的 schema 3
报告（不是续跑历史文件）得到 `not_local_build_report`，不得进入中位数；
搜索期已是 schema 4 `ok` 但 `build` 为 `null` 时直接
`build_identity_mismatch`；搜索期为 full 时，一个 batch digest 不同则失败；
两个 batch 都匹配时，中位数报告是这两份之一；换 winner 时胜者只对过自己的
搜索期报告。另加续跑测试：
准备一份 schema 3 的 `report.json`，记下整文件 sha256；`resume.py` 读入后
走 `_save`（以及会给内存报告补 `seed_gpu_us_median` 的路径）。断言该文件
字节与读入前一致，而不是只断言版本仍为 3。内存里给会话提示用的副本可以
含别的字段，但不得写回这个路径。

`test_eval_cache.py` 增加：键 body 的 `schema` 为 3，且 body 没有
`build_profile_digest`。现有的「改 seed / 源码 / backend 则键变化」保持。

`test_report_schema.py`：今天用 `schema_version=4` 表示过新。改后过新的例子
是 5。0–3 无 `build` 仍能读；0–3 带 `build` 拒绝。schema 4 缺键、多键、
full 里有 digest 为 `null`、partial 里 `native_digest` 非 `null`，都拒绝。
`tests/shared/test_orchestrator.py:68` 的候选 `report.json` 整数改为 4。
假评测若要确认成功，报告里是合成的 full `build`，不是 `null`。同文件里
`result.json` 的 `schema_version == 1` 不要改。

`tests/backends/metal/test_backend_phase0.py:49-52` 改为两套断言，不改
基线文件：文件里的 `report_schema_version` 仍是 3，`top_level_fields` 仍
等于文件里记下的那份列表；当前 `REPORT_SCHEMA_VERSION` 是 4，
`dataclasses.fields(EvalReport)` 比那份历史列表多出 `build`。协议版本与
index 版本的断言保持对历史文件的原样比较。不要新写一份「当前清单」去替换
`git_commit=ef2a048` 那份文件。若要记录 schema 4 的字段表，另增一个新
文件，并写明它不是 2026-10-23 那次测量。

另外在本提交里跑：

```bash
git grep -n 'schema_version"\] == 3' tests
git grep -n 'REPORT_SCHEMA_VERSION == 3' tests docs
```

命中的 EvalReport 整数断言在同一提交改掉，或说明它指的是历史基线文件里的
3，而不是当前写入版本。`result.json` 的 schema 1 不要改。
无设备回归只检查隔离，不把远端报告说成 schema 4 或历史兼容发布。上面
两条现有 grpc 测试不新开设备：
`test_npu_candidate_timeout_completes_a_diagnostic_timeout_report` 与
`test_a_candidate_timeout_reports_a_not_run_build_not_a_compile_failure`
断言 `_timeout_report` 的 EvalReport 与升级前一样没有顶层 `build`，也没有
被写成 schema 4。证据里的 NPU `build.status == "not_run"` 保持原断言。
`JobStore._validate_report` 仍接受厂商今天交回的那份报告，拒绝一份编造的
Metal full。不把 NPU `BuildEvidence` 抄进 `EvalReport.build`，不导入 Metal
backend。`test_publication.py` 继续只断言 `_check_publication` 的
provenance 条件，不为远端结果增加五个 digest 的要求。

`tests/support/golden/` 里已有向量的字节和 HMAC 必须与 HEAD 逐字节相同，
包括 `tcenv.eval-report.v3` 这个字符串。schema 4 的覆盖用新文件，不重算、
不替换旧 golden。旧测试若因新报告字段变红，修的是生产路径不要把 schema 4
写进旧向量，而不是更新签名。

真实 Mac 的 gpu 测试仍是 P0 那条，外加一条：quick 评测的 full `build`
五个 digest 来自同一次 outcome，`compile_s` 仍是 kernel 对象上的值。
许可不足则 blocked。

## 7. 成本，以及什么时候才重做基线

已经存在的编译次数，P0 不会减少：

- `mode=full`，以及 `timed` 且要求校验时，`evaluate()` 先
  `_validation_pass()` 再 `_run_worker`。两个进程各调用一次编译。
- 通过 `xcworker` 做 profiling 时，第三个进程再编译一次。
- `probes.py` / `profile.py` 的带宽探针是另一段固定源码，不是候选报告。

P0 若在父进程再编译一次来填 `build`，就会在上述次数之外多一次。禁止。

缓存从 schema 2 升到 3 之后，旧的 ok 文件还在磁盘上，新键不会读它们。
新搜索要重编、重测。这是缓存冷启动，不是发布基线推倒重来。旧 run 里已经
确认的 speedup 保持原字节。

混合版本旧数据的审计就是下面三件可重复的检查，不另做人工盘点：

- 旧 `report.json` 字节不变：§6 续跑测试比较读入前后的整文件 sha256。
  `_save` 之后仍然相同。
- 写入边界：本地新报告是 schema 4。远端 / NPU 生产者的报告字节不被改成
  schema 4，也不被送进 §5。用 P2 的无设备 grpc 断言做隔离，不跑真实 NPU。
  历史 schema ≤3 只在续跑读取时分支。
- outbox 已完成行只重放：`test_evaluation_outbox.py` 断言完成行重放出来的
  报告字节与写入时相同，schema 3 行不会变成 schema 4。

不重建测量基线的原因：本计划不改计时窗口、确认 batch 的统计定义、目标
限额或 `PROTOCOL_VERSION`。`backend-phase0-baseline.json` 整份保持原字节，
包括 `gpu` 段、`report_schema_version: 3` 和当时的 `top_level_fields`。
字段表与当前代码的差异由测试分开断言（§6 的 P2），不回写这份历史文件。

必须另做测量基线的条件（本计划不做）：计时窗口、确认批次定义或目标限额
变了，因而要升 `PROTOCOL_VERSION` 或 evaluator epoch。在那之前，不要把
新跑出来的数字写成旧基线的替代，也不要宣称 schema 4 让已发布加速比更准确。

尚未被证据支持的说法：没有测量证明已经发生过「同类源码、不同 MSL 命中
同一缓存」。schema 3 的键仍不含 MSL / macOS。确认阶段因为
`eval_cache=False` 会重测；只有搜索期报告是 full 时，digest 不一致才会
挡住发布。未跑确认时，`best_speedup` 仍可能来自另一套工具链的缓存。
要在命中路径上堵住这一点，必须每次查缓存都 probe 设备。那是另一次行为
变更，不在本计划。单路径在 HEAD 上还不存在，不能把草稿说成已经收益。
本文件没有跑 gpu 测试。

## 8. 仍不在本文决定的事

- 缓存命中前是否 probe。决定是不做，见 §7。
- native digest 要不要进入某一版 NPU provenance。要做就必须新的 document
  version，并且不能改 v5 的字节。归通信计划。
- 旧 run 目录留多久。本文不删数据。
