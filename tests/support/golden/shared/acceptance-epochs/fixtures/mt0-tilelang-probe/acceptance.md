# MT0-tilelang-probe：TileLang 固定版本受信安装与 source-only 隔离探针（安装 + 源码级 probe 完成，MT0 仍 blocked）

## 1. 结论与严格范围

本轮按用户决定只推进 **TileLang 0.1.15**，不安装、不构建、不探测 Halide；第二条非 MSL DSL
留待后续单独决定。当前结论是：

| 项 | 结果 |
| --- | --- |
| 固定版本 macOS arm64 wheel | **存在且已受信安装**：`tilelang-0.1.15-cp39-abi3-macosx_11_0_arm64.whl` |
| wheel/依赖哈希 | **全部与 PyPI 发布值逐一核对通过**：19 个 wheel、230,410,258 B |
| 安装身份 | **RECORD 21,199/21,199 文件全部验证通过**；0 mismatch、0 missing |
| 上游源码绑定 | TileLang tag 的 390 个 `tilelang/**/*.py` 与 wheel 逐字节相同；TVM 子模块 745 个 `python/tvm/**/*.py` 与 wheel 内捆绑副本逐字节相同 |
| source-only frontend probe | **已真实执行并完成**（第 5 轮）：6/6 构建产物、`full_build=true`、`probe_executed=true` |
| 设备执行 | **0 dispatch**：未创建 library/pipeline/buffer，未编译或执行任何生成的 kernel |
| 真实负例 | **4/4 matched**：文件读、越界写、非 proxy loopback、未声明 artifact |
| 静态/adapter 负例 | 5 项由冻结的 adapter 合同与固定节点覆盖，其中 4 个身份漂移由真实 entry 的固定 CPU mutation 拒绝，与真实 outer fixture 分开记录 |
| 失败历史 | r1–r4 原始结果与输入/日志全部保留；**没有删除或用 r5 覆盖** |
| Verifier | `verify.py` 重算证据字节、身份、负例、历史、修复链与 reviewer 结论；75 个固定节点 0 skip/xfail |
| Reviewer 结论 | `review.json`：`done=true` 仅限“固定安装 + source-only 隔离构建 + 真实负例”；`not_done` 明确含 MT0/MT1b/第二条 DSL/生产 DSL/真实 Metal/发布 |
| MT0 | **仍 blocked**：唯一剩余阻断是第二条非 MSL DSL 按用户决定 deferred；MT1b 另因硬依赖 MT0 且选型未确认而 blocked |
| Metal / 生产 DSL / 发布 | **未开始**：`metal_or_performance=not_started`、`production_ready=false`、`publication_allowed=false` |

**本包的 `done=true` 只覆盖“固定 TileLang 受信安装 + source-only 隔离构建与负例经独立复核且
verifier 通过”。** 它不表示 MT0、MT1b、生产 DSL、真实 Metal、整个多 DSL 计划或第二条 DSL
完成。`probe-status.json#done_semantics` 与 `verify.py` 的输出都固定了这条边界。

## 2. 用户决定与轮次边界

- 首轮只选 TileLang；不做 Halide。
- 允许受信准备阶段联网下载/安装固定版本 TileLang 及必要依赖到受控目录。
- 只做 source-only frontend 探针：解析、lower、生成 host/device/MSL 构建产物；不执行生成的
  kernel，不调用 Metal GPU，不做 correctness/performance。
- 不运行 GPU/NPU、真实模型搜索或付费实验。
- 候选代码即使简单也按 `embedded_host_code` 处理，只能在 hmz lower-level Profile 已应用后运行。
- 第二条非 MSL DSL 留待后续用户批准轮次；不用 Halide、玩具格式或 MSL 改名顶替。

`probe-status.json#round1_dsl_decision.halide` 固定
`installed=false / built=false / probed=false / code_or_dependency_added=false`，防止以后把
“只选 TileLang”写成两条 DSL 已经就绪。

## 3. 受信安装：来源、license、哈希与实际身份

### 3.1 上游与 license

```text
dsl_id            tilelang
version           0.1.15
tag               v0.1.15
upstream commit   a35f8ddf45eba16c21211ec8822d56ce5363036f
TVM fork          https://github.com/TileLang/tvm
TVM commit        4211874e9e4b8770fd20ad0d98ab80afd1df5028
license           MIT（wheel LICENSE 另记 2024-12-01 至 2025-03-14 的 Microsoft 历史协作条款）
LICENSE sha256    f70e5cdc24c3797af741930440086f9a0ad36930b05a3dfc25b420936dc2cfa1
```

### 3.2 wheel 与依赖

- wheel：`tilelang-0.1.15-cp39-abi3-macosx_11_0_arm64.whl`，33,667,284 B，
  sha256 `dcffc9aa8d4cd086dd93090f5afeeaa38aa621d2bb80998c4e5f5ee07be116aa`，
  与 PyPI JSON 的 `digests.sha256` 相同。
- 依赖闭包：19 个 wheel（含 `apache-tvm-ffi==0.1.12`、`torch==2.15.0`、`z3-solver==4.15.4.0`、
  `torch-c-dlpack-ext==0.1.5`、`numpy==2.5.4` 等），总 230,410,258 B；每个文件的 sha256 都与
  `https://pypi.org/pypi/<project>/<version>/json` 的发布值核对。
- 下载走清华 PyPI 镜像：本机直连 `files.pythonhosted.org` 多次 stall；镜像返回的是同一
  content-addressed 文件，且逐文件对 PyPI 官方 sha256 复核，故来源与完整性仍可审计。
- Python 使用 `/opt/homebrew/bin/python3.13` 创建**独立 venv**，版本 3.13.1，pip 24.3.1。
  `HOME`、`TMPDIR`、pip cache 全部重定向到 staging 目录内。

### 3.3 安装位置与不污染边界

staging 根：`/tmp/shapeforge-frontends/tilelang-0.1.15/`（临时受控目录，本轮 HOME 不可写时唯一
可写的受控位置）。

- 未修改项目 `.venv`、hmz uv-tool、系统 Python、全局 pip/uv 配置。
- 未把 wheel、依赖或缓存写入仓库；仓库只保存文件名/版本/sha256 文本与小型证据。
- staging 根权限为 `0700`（仅 owner 可访问）；它是**临时**目录，外层可按 `frontend-install.json`
  的 URL、版本、sha256 原样重建。
- 安装命令：先 `pip download` 到 `wheels/`，再
  `pip install --no-index --find-links wheels --no-compile tilelang==0.1.15`；`pip check` 无
  broken requirement。

### 3.4 实际安装身份

- 20 个 distribution 的 RECORD 共列出 21,199 个带 sha256 的文件，**21,199/21,199 验证一致**。
- 安装树：21,186 个文件、853,505,126 B，manifest digest
  `ed87747484c00473356eae644062bf37e137a55ef208dacc940cfb075ed55dad`。
- `tilelang/` 包：3,651 个文件、134,219,728 B，manifest digest
  `06f3b0adee7abce77af1111cb36e704b4b23c3cbe38d095dd62cce86509c11cc`；
  与 RECORD 文件集完全一致，无 symlink，无额外文件。
- dist-info：METADATA `380d3486286aeb708dc4846cfdc4b438985c84199a160b21238cadf3e4e3fa19`，
  RECORD `6ba83c2ed267154eedc9dec6d916e315c553a8e515744f0feb7ca05dfa388c34`。
- 诚实边界：安装树聚合摘要（21,186 文件 / 853,505,126 B）是**安装时记录**，其聚合规则没有在本包
  重述，`verify.py` 不重算这个聚合；它重算的是主 wheel、19 个依赖 wheel 的 bytes/SHA 与
  PyPI 发布摘要绑定、`tilelang/` 包 manifest、dist-info 与全部 21,199 条 RECORD。
  probe 的身份锚点因此是这些可逐文件重算的身份，不是这个 tree 摘要。

### 3.5 源码身份与诚实边界

从 `codeload.github.com` 下载固定 tag / 固定 TVM commit 的 tarball（sha256 已记录）并逐一比较
Python 源文件：

- TileLang tag：390 个 `.py`，390 个与 wheel 相同，**0 mismatch、0 只在 tag 侧**；
  wheel 另有 747 个文件来自捆绑的 `3rdparty/tvm`/cutlass 子树。
- TVM 子模块：745 个 `python/tvm/**/*.py`，745 个与 wheel 内副本相同，**0 mismatch、0 单侧差异**。

`frontend-install.json#execution_boundary` 描述的是**安装阶段**（未 import、未 codegen、未生成
kernel）；§4 记录的外层 probe 是之后独立的一步，两者不矛盾。

**边界：**`libtilelang`/`libtvm` 的 native 字节没有用本地源码重建并逐字节比较。
`source_provenance.native_libraries_rebuilt_from_source=false`，不得把 Python 源一致扩大成
“整个二进制已从该 commit 可复现构建”。

## 4. 真实 source-only probe（第 5 轮）

### 4.1 执行命令与上下文

```bash
cd <repo>
uv run python 'docs/架构/多DSL-Track-阶段验收/MT0-tilelang-probe/probe/run_probe.py' \
  --frontend-install 'docs/架构/多DSL-Track-阶段验收/MT0-tilelang-probe/frontend-install.json' \
  --attempt-root /tmp/shapeforge-mt0-tilelang-attempt-r5 \
  --result 'docs/架构/多DSL-Track-阶段验收/MT0-tilelang-probe/probe-result-r5.json'
```

`probe-result-r5.json`：`status=source_only_frontend_probe_completed`、`probe_executed=true`、
`full_build=true`、`gpu_npu_executed=false`、`device_dispatch_calls=0`、
`metal_correctness_or_performance_claim=false`、`mt0_status=blocked`、
`mt1b_status=not_started_blocked`、`production_ready=false`、`publication_allowed=false`。

隔离：最低能力的 `sandbox-exec` 能力探测 `returncode=0`；唯一 lower-level 路线
`hmz.coganchor.fence.proxy.Proxy + hmz.coganchor.darwin.seatbelt.Profile` 已真实应用
（`profile_command_spawned=true`、`fenced_returncode=0`）；`online=false`、`hosts=[]`、
proxy 已停止、**无 mds 写根**、`internal_fence_or_wrap_used=false`、无无隔离 fallback。
Profile 的写集只有 `/dev/*` 与 `attempt/build/{work,home,cache,config,data,tmp,pycache,artifacts}`。
`attack_containment_proven=false`：本轮只观察了下列 4 个负例，不宣称外层攻击面被完整证明。

trusted collector 只在 driver report 通过 adapter 的 `_valid_driver_report`（`online=false`、`hosts=[]`、
proxy 已停止、profile 读/写与请求逐项一致、`mds_write_root_added=false`、`internal_fence_or_wrap_used=false`）
且 shim 的 limit report 为 `ready` 后才收集 artifact；`completed` + 六个文件本身就是这些检查的产物，
不是 runner 的自述。

嵌套 hmz/RLAR 上下文曾观测到 `sandbox-exec: sandbox_apply: Operation not permitted`，该事实原样
保留在 `context-capability.json`，并作为 `probe-status.json#prior_typed_code` 的历史记录；它描述的是
当时的嵌套上下文，不是 r5 的失败。

### 4.2 六个构建产物（原字节归档）

`evidence/source-only-r5/` 与 `probe-result-r5.json#build.artifact_rows` 逐字节一致：

| 文件 | bytes | sha256 |
| --- | --- | --- |
| `frontend_identity.json` | 2,338 | `f31474453cd08dd58fa557568a5b8cfa3cc13caadead791427df1deec07c8ed7` |
| `kernel.metal` | 732 | `a53f916fac4806462ab541ba9d06b9ad8a03415a58f3f270c984b483dfcf1ca8` |
| `device_ir.json` | 15,447 | `16b53c656ca94b6506a97e5898fee0ca4c7f2af0087fcf757a50d0748b94c3e0` |
| `kernel_params.json` | 778 | `51e90320a5f7057ca3f93fe61fbe4fbae711524f10d7820e608f5a66de19f58f` |
| `host_module.txt` | 12,598 | `a3152a4b5d5bcd89461f36ef2ede118d64c86cafa8494fe6af4ba632ffc5792e` |
| `build_identity.json` | 1,760 | `e0d3b94c919ce68f7b173f47b6384bf7663079cee21ecc5e0730b48a71cafab5` |

`artifact_set_sha256 = bc6a7300ee1010be1c7667b187c951cd7ec1e6961b87d288886e3776698e821f`，
范围为前 5 个 build artifact 的 `{path, bytes, sha256}` 行，规范化规则
`json.dumps(rows, sort_keys=True, separators=(",", ":"))`；`build_identity.json` 保存这 5 行与该摘要，
自身由 trusted collector 单独哈希。

### 4.3 产物内容与“未执行设备”边界

- `kernel.metal`：真实 MSL source。`scale_kernel_kernel`、`[[kernel,
  max_total_threads_per_threadgroup(256)]]`、`a [[ buffer(0) ]]`、`b [[ buffer(1) ]]`，
  函数体为 `b[blockIdx.x * 256 + threadIdx.x] = a[blockIdx.x * 256 + threadIdx.x] * 2.0f`。
  不含 `MTLCommandBuffer`/`dispatchThreadgroups`/`newLibraryWithSource`/`makeComputePipelineState`。
- `device_ir.json`：TVM JSON（`tvm_version=0.25.dev0`），`global_symbol`、`thread_extent`
  的四个 `IntImm` 依次为 `[4, 256, 1, 1]`，指针类型为 `global` scope，buffer 元素为 `float32`。
- `kernel_params.json`：`KernelParam` 图，`dtype=float32`、`shape=[1024]`。
- `host_module.txt`：host codegen 的 **C source**（`host_artifact_format="c-source"`），含
  `__tvm_ffi_scale_kernel`/`scale_kernel_kernel`。必须如实指出：只有早期 headers 位于
  `#ifdef __OBJC__` 内；后面的 `torch::mps::get_dispatch_queue()`、`dispatch_sync`、
  `get_command_buffer()` 与 `metal.SetStream` 在该 guard **之外**，并包含
  `id<MTLCommandBuffer>` 文本。该文件没有被编译、加载或执行；`device_execution` 的四项均为
  false/0，`build_identity.device_execution.dispatch_calls=0`。因此“没有 dispatch”是**执行事实**，
  不是“生成源码里不出现 device 符号”，这些计数也不是 OS-level trace。
- 导入期 `torch.backends.mps.is_available` 被固定版本 TileLang 调用了 2 次（`torch.mps.is_available`
  为 0），由 entry 的计数器记录；entry 自己不主动发起设备查询，也不创建任何 device 对象。
- 运行期未调用 `tilelang.compile`/`tilelang.jit`/JITKernel、未做带 device 编译的 `tvm.build`、
  未调用 `device_codegen(compile_device=True)`，未创建 `MTLDevice`/`MTLLibrary`/
  `MTLComputePipelineState`/buffer/encoder/command buffer。
  这些 device 计数由冻结 entry 自己写入（runner 也固定 `device_dispatch_calls: 0`），是
  **代码路径事实而非 OS 级 trace**：可信度来自入口只调用 `tilelang.engine.lower` 且
  `enable_device_compile=False`、入口 AST 禁止面、没有任何 Metal 库/字节码被编译或加载，
  以及 host artifact 只是未编译的 C source。

## 5. 失败历史：r1–r4 原样保留

原始结果与归档位置（**未删除、未改写**）：

| 轮次 | 结果文件 | returncode | 根因 | 该轮之后的修正 |
| --- | --- | --- | --- | --- |
| r1 | `probe-result.json` | 73 | 候选使用无限定 `@prim_func`；固定版本 parser 不识别该装饰器，错误渲染又在 `tvm::Source::GetLine` 崩溃（`line_map size: 9` vs `requested line: 25`） | 仅诊断：r2 在 lowering 异常处理里加 `traceback` |
| r2 | `probe-result-r2.json` | 73 | 同 r1；失败发生在候选 module 求值阶段，r2 的 handler 还没覆盖到 | 仅诊断：r3 用 `try/except` 包住 `load_candidate` 并打印 traceback |
| r3 | `probe-result-r3.json` | 73 | 同 r1，traceback 显示 `The parser does not understand the decorator` | 仅候选调用点：`@prim_func` → `@T.prim_func`，删除对应 import |
| r4 | `probe-result-r4.json` | 72 | 候选已解析；`tilelang.metal.pipeline` 的 `LayoutInference` 调用 `Target::Current(true)`，缺少 Target context（`Target context required`） | 仅前端调用点：同一 `tilelang_lower(...)` 放进 `with target:` |
| r5 | `probe-result-r5.json` | 0 | — | 完成 |

r1–r4 的 raw `status` 都是 `full_build_rejected`（runner 在 `validate_full_build` 对非 0 退出走该
分支），`typed_code=backend_capability_unavailable`，`isolation.status=process_failed`；
`probe-history.json` 与 `verify.py` 逐字段对比 raw 的 `status`/`typed_code`，不再把派生状态写死。

每轮的候选输入与 frontend entry 输入归档在 `evidence/failure-inputs/rN-*.py`，stderr 归档在
`evidence/failure-logs/rN-stderr.txt`；`probe-history.json` 用 SHA-256 绑定全部结果、输入与日志。

### 5.1 修正只是前端调用，不是算法或性能改动

- 候选 r1 → r5 的文本 diff 只有三行：
  `-from tilelang.language.tir import prim_func`、`-@prim_func`、`+@T.prim_func`。
  去掉该 import 并归一化 `scale_kernel` 的装饰器列表后，r1 与 r5 的 AST **完全相同**：
  同名函数、同 `float32` 输入输出、同 `N=1024`/`THREADS=256`、同
  `T.Kernel(N // THREADS, threads=THREADS)`（4×256）、同 `T.Parallel` 循环体与同
  `a[...] * 2.0` 表达式。`verify.py#verify_fix_chain` 机械重算这一结论。
- frontend entry 的 r1→r2、r2→r3 只是增加 traceback 诊断；r3→r4 无变化；r4→r5 只是把同一个
  `tilelang_lower(...)`（同 target、同 `target_host="c"`、同 `enable_host_codegen=True`、
  同 `enable_device_compile=False`）放进 `with target:` 并缩进 4 空格。删除该包装并反缩进可
  逐字节还原 r4 文件。
- 没有改 grid、threads、tiling、融合、调度、测量口径或隔离参数；runner 与
  `frontend-install.json` 在 5 轮中不变（`probe/run_probe.py` sha256
  `205e82532f6fc420bd39ca0425c8ea94fb0573842f37d5aac073f9264cf6df3c`）。

### 5.2 过程事件（不隐藏）

第一次 `--fixture network_connect` 调用因缺少 `--outside-port` 返回 2；runner 在创建 listener 与
attempt 目录之前就退出，所以那次调用没有 result JSON、也没有 attempt 目录。重跑（受控 listener
实际端口）的结果才是归档的 `negative-network.json`，其 stdout/stderr 路径
`/private/tmp/shapeforge-mt0-tilelang-negative-network-r2` 也印证了这是重跑。
`probe-history.json#process_incidents` 固定了这一历史。

## 6. 负例矩阵：4 项真实 outer fixture + 5 项静态/adapter 覆盖（含 4 项真实 CPU identity mutation）

### 6.1 真实执行（raw 结果在包根）

| id | 结果 | typed outcome |
| --- | --- | --- |
| `candidate_sensitive_read` | `negative-read-outside.json` | 读被拒（returncode 13，0 artifact，outside canary 未变） |
| `candidate_outside_write` | `negative-write-outside.json` | 写被拒（returncode 13，0 artifact，outside canary 未变） |
| `candidate_network_connect` | `negative-network.json` | 非 proxy loopback 未成功（returncode 13，0 artifact） |
| `undeclared_artifact` | `negative-artifact.json` | trusted collector 拒绝整包：`artifact_contract_rejected` |

四者的 `fixture_observation.matched=true`、`probe_executed=false`、`full_build=false`，且文档中
没有 `build` 对象；读/写/网络三项的 `isolation.implicit_mds_write_roots=[]`、
`policy_online=false`、`driver_report.mds_write_root_added=false`。注意 `artifact_count=0` 也有
adapter 层保证：只有 `completed` 才会按 declared set 收集 artifact，非 0 退出根本不收集。

**证据强度边界：**三个 fixture 只在**成功**时返回 14，否则把 `PermissionError` 与一般 `OSError`
（如 connection refused/ENOENT）都映射为 13；canary 在运行前已实际创建并核对，网络 listener 也
已先绑定，所以这些结果与“被 Profile 拒绕”一致，但归档文档本身**只精确记录到“失败而非成功”、
而非异常类型**。不要把它写成观察到 `PermissionError` 的 OS 级 trace；这正是下一个小节要求 fixture
记录异常类型或 listener accept 计数的理由。

**顶层 status 的已知形态：**runner 的 fixture 路径复用通用结果文档，所以这四个 raw 文件的顶层
`status` 仍是 `blocked_backend_capability_unavailable`（`negative-artifact.json` 也是），即使
fixture 已被正确拒绝。`fixture_observation` 才是权威；顶层 status 既不是构建成功，也不是一次新的
能力失败。`verify.py` 按这个冻结形态严格检查，不允许把 raw 文件改成“成功”。

### 6.2 静态或 adapter 合同覆盖（明确区分）

| id | 覆盖方式 |
| --- | --- |
| `frontend_version_drift` | 真实 `frontend_entry.main()` 的 `[pinned_version]` 固定 CPU mutation：`pinned.version=0.1.16` 与安装 METADATA `Version: 0.1.15` 不一致，命中 `IdentityDrift`；plan 的 9 项矩阵 |
| `frontend_source_drift` | 真实 `frontend_entry.main()` 的 `[package_manifest]`、`[metadata]`、`[record]` 固定 CPU mutation：分别篡改 `installed.tilelang_package_manifest_sha256`、METADATA、RECORD 身份，均命中 `IdentityDrift`；plan 的 9 项矩阵 |
| `codegen_timeout` | 继承的 adapter watchdog 合同与固定节点 |
| `candidate_nonzero` | 继承的 adapter `process_failed` 合同与固定节点 |
| `missing_toolchain` | adapter/hmz 的 typed `backend_capability_unavailable` 合同与固定节点 |

这 5 项仍与 4 个真实 outer fixture 分开记录，合计 9 项，全部要求 no full build。前两类现在不是
“只把 ID 写入 coverage list”：四个固定 CPU 测试都调用真实 `frontend_entry.main()`，在 lowering 前
返回 typed `IdentityDrift`，artifact 目录为空，不产生 full-build 或 production-ready 结果；它们不是
新的 outer probe，也不生成 raw probe JSON。其余 3 项仍由继承的 adapter/Watchdog 合同覆盖。

## 7. 验证与固定节点

`nodes.json` 固定 **75 个节点，0 skip/xfail**：

- `tests/shared/test_mt0_tilelang_probe.py`：**62** 个。除安装/身份/上下文/计划/源哈希外，
  新增 4 个真实 entry 身份漂移拒绝节点（pinned version、package manifest、METADATA、RECORD），以及
  host guard 实际布局/误判拒绝；还包括归档产物逐字节与 result+identity 一致、`device_dispatch_calls=0`/offline/no-mds、
  r1–r4 失败历史与根因、修复链只改装饰器与 Target context、4+5 负例分层、artifact/负例/历史
  变异拒绝、status 过度声明拒绝（MT0/MT1b/生产/第二条 DSL/dispatch/device compile/online/mds/
  无证据）、reviewer 结论必须窄且写明残余限制。
- `tests/shared/test_worker_isolation_hmz.py`：**9** 个继承节点，固定 offline/no-mds 策略、
  undeclared/symlink artifact 拒绝、timeout 回收、non-zero 非成功、缺 hmz/Seatbelt typed 拒绝、
  在线/控制字段拒绝、无 internal fence。
- `tests/shared/test_mt0_isolation_profile.py`：**3** 个继承节点，固定 r2 18/18 raw 保留、
  文件/网络/mds 拒绝与 negative controls、以及“外层观测不完成 MT0/frontend/Metal”。
- `tests/shared/test_mt0_dsl_selection.py`：**1** 个继承节点，固定 MT0 仍 blocked 且依赖阶段未授权。

`verify.py` 不是“查字符串成功”：

1. 重算 `evidence/source-only-r5/` 六个文件的字节/SHA，与 `probe-result-r5.json` 的行、
   `build_identity.json` 的行和 `artifact_set_sha256` 对齐；检查无 symlink/hardlink、集合精确。
2. 解析 MSL（kernel 签名、buffer 0/1、4×256 索引与 `* 2.0f`、禁止执行侧构造）、device IR
   （`global_symbol`、`[4,256,1,1]` thread extent、`float32`、1024 shape）、kernel params、
   host source（明确只有早期 headers 在 `#ifdef __OBJC__` 内，后续 MPS/command-buffer 调用文本在 guard
   外，不把文本缺失当证据，并用 guard 误判 mutation 拒绝错误布局）。
3. 安装身份不是只查字符串：重算主 wheel 与 19 个依赖 wheel 的 bytes/SHA，并要求每个依赖的
   `sha256 == pypi_published_sha256`；重算 `tilelang/` package manifest（3,651 文件）、
   dist-info METADATA/RECORD，以及 20 个 distribution 的 21,199 条 RECORD（urlsafe-base64）。
   随后校验 `frontend_identity.json`/`build_identity.json` 与安装身份、target/options、
   `enable_device_compile=false`、四项 device 计数为 false/0、candidate SHA 与当前候选一致。
4. 校验 r5 的隔离事实：`online=false`、空 host allowlist、无 mds、无 internal wrap、
   `fenced_returncode=0`、proxy 已停止、`attack_containment_proven=false`；写集必须**恰好等于**
   8 项 `/dev/*` 与 `…attempt-r5/build/{work,home,cache,config,data,tmp,pycache,artifacts}`，
   额外根、缺失根或别的 attempt 根都会被拒绝。
5. 校验 r1–r4 原始结果/输入/日志的 SHA，并逐字段要求派生历史与 raw 的
   `status=full_build_rejected`、`typed_code=backend_capability_unavailable` 一致；r3 日志须含
   `The parser does not understand the decorator`、r4 日志须含 `Target context required`。
6. 校验 4 个真实 fixture 负例的 `fixture_observation`（不只顶层 status）与 5 项静态/adapter 覆盖的
   分离；另执行 4 个真实 entry 身份 mutation，分别固定 pinned version、package manifest、METADATA、RECORD
   漂移在 lowering 前以 `IdentityDrift` 拒绝，且无 full build/production-ready。
7. 机械校验修复链：候选 AST 等价、entry 诊断 diff 与 `with target:` 包装可从 r4 逐字节重建。
8. 校验 `review.json`：`done=true` 的 done_scope 只覆盖安装 + source-only 构建 + 负例，
   `not_done` 必须含 MT0/MT1b/第二条 DSL/生产 DSL/真实 Metal/发布，findings 必须写出
   生成 host source 的真实 guard 布局与设备文本、device 计数非 OS trace、fixture 13 的异常类型合并、首次
   network 调用 rc=2、未重算的安装树聚合与 75 节点结果。
9. `self_test()` 对上述每类检查做内存变异，任何“只改字符串/改状态”都必须被拒绝。

运行方式：

```bash
uv run python 'docs/架构/多DSL-Track-阶段验收/MT0-tilelang-probe/verify.py'           # 结构/证据/历史/自测
uv run python 'docs/架构/多DSL-Track-阶段验收/MT0-tilelang-probe/verify.py' --check   # 另跑 75 节点并重算 manifest
```

## 8. 未完成项与边界

`probe-status.json#current_status` 与 `probe-plan.json#round_status` 记录：

1. **MT0 仍 blocked**，且唯一剩余阻断就是第二条非 MSL DSL：用户决定本轮只做 TileLang，两 DSL
   门槛未满足。MT1b 不是 MT0 的阻断项，它自身因硬依赖 MT0 与缺少显式 DSL 选型确认而 blocked。
   **不再**把“probe 未执行”列为阻断。
2. MT1b/MT2–MT7、真实 Metal correctness/dispatch/performance、生产 DSL 与发布均未开始。
3. 本轮 probe 是 source-only：不编译 Metal library、不创建 pipeline/buffer、不 dispatch；
   `host_module.txt` 只是未编译的 C source。
4. tilelang/tvm native 库未从源码重建，不宣称二进制可复现。
5. 外层 Seatbelt 攻击面只观察到这四个 fixture，不宣称整体 containment。

## 9. 历史纪律与复核边界

- 未回改 `MT0/`、`MT0-isolation/`、`MT0-isolation-profile/`、`MT1a/`、`MT5/`；
  其 manifest/raw/review 摘要由 `start-state.json#other_closed_sha256` 与 `verify.py` 固定。
- 未修改六 RPC/field number/HMAC/BlobChannel、R0–R6/G0/A1–A6/P0–P2。
- 未新增 pattern/tiling registry，未限制候选算法或调度自由。
- 未执行 Git 写操作；未安装 Halide 或新依赖；未运行 GPU/NPU/真实模型/付费搜索；未修改 hmz。
- 未嵌套 hmz/hmz_flow/Agent/subagent；没有为了“通过”而放宽 Profile、改 hmz 或接受 Xcode license。
- `probe-result*.json`、`negative-*.json`、`evidence/` 与旧包证据均未被改写；r5 成功不掩盖 r1–r4。

### 9.1 旧 epoch 的 living-document 锚点漂移（如实记录，不回改旧包）

本轮按用户要求更新了两份 living 文档：`docs/架构/多DSL-Track实施计划.md` 和
`docs/可信harness/KDA-CAKE2-与ShapeForge的多Track交付对比.md`。封闭的 MT0-isolation-profile
在其 `verification.json#scoped_source_sha256` 中保留了这两份文档的旧摘要，因此它的 verifier
现在在 `verify_results()` 处对这两项报 `scoped source digest mismatch`。

这是**预期的 living-document 锚点漂移**，不是本轮改写旧证据：

- 旧包的 `manifest.sha256`、r1/r2 raw、`outer-probe-review.json`、`verification.json` 与
  `verify.py` 均逐字节未改；`history-anchor-drift.json` 记录旧/新摘要，`verify.py` 逐项重算并
  固定旧包文件的 SHA-256。
- 没有修改旧 verifier 去接受新摘要，也没有 skip/xfail/弱化它的任何节点。
- 失败范围仅为这两项 `scoped_source_sha256`；旧 epoch 的 raw、结论和 reviewer 语义不变。
- 本包**不宣称**旧 verifier 仍端到端通过。单独执行旧包的 `verify_document`、
  `verify_source_identity`、`verify_outer_plan`、`verify_cpu_evidence`、`verify_history`、
  `verify_outer_review`、`verify_nodes`、`self_test` 与 `verify_manifest` 均通过；唯一失败是
  `verify_results()` 的这两项 living-document 摘要（`history-anchor-drift.json#isolated_old_verifier_checks`）。
- 要恢复旧 verifier 端到端通过，只能由后续独立 epoch 决定是否把 living 文档从 closed-package
  的源摘要锚点中移除；本轮不回填旧包。
