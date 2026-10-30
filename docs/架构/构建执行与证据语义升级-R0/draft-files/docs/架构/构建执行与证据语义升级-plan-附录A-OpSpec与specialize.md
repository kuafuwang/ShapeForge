# 附录 A：OpSpec 示例与负例

日期：2026-10-28（按主 plan 修订稿对齐）

> **定位**：[`构建执行与证据语义升级-plan.md`](构建执行与证据语义升级-plan.md) 的附录，只放 RMSNorm 的
> OpSpec 与 track 绑定示例、不属于 OpSpec 的内容，以及关键负例。阶段划分、身份模型、构建路径、报告
> schema 与确认门的规则只在主 plan；两者冲突时以主 plan 为准。本文描述的都尚未实施。

## 1. RMSNorm 的 OpSpec

示例只含语义字段。`spec_text`、`description` 与 dtype 拼写属于 track 绑定（§2），不在这里，也不进入
`opspec_digest`。

```json
{
  "spec_version": 1,
  "op_name": "rmsnorm",
  "category": "normalization",
  "formula": "y[r,:] = x[r,:] * rsqrt(mean(x[r,:]^2, axis=H) + EPS) * w",
  "uses": ["EPS"],
  "type_vars": {
    "T": ["fp16", "bf16"]
  },
  "symbols": {
    "M": { "min": 0 },
    "H": { "min": 1, "note": "允许非整除；tail 必须被覆盖" }
  },
  "tensors": {
    "x": {
      "role": "input",
      "dims": ["M", "H"],
      "dtype": "T",
      "layout": "row_major",
      "strides": ["H", "1"],
      "space": "global",
      "access": "read"
    },
    "w": {
      "role": "input",
      "dims": ["H"],
      "dtype": "T",
      "layout": "contiguous",
      "strides": ["1"],
      "space": "global",
      "access": "read"
    },
    "y": {
      "role": "output",
      "dims": ["M", "H"],
      "dtype": "T",
      "layout": "row_major",
      "strides": ["H", "1"],
      "space": "global",
      "access": "write"
    }
  },
  "numerics": {
    "acc_dtype": "fp32",
    "rounding": "single_round_at_output",
    "tolerances": {
      "fp16": { "rtol": 0.002, "atol_rel": 0.001 },
      "bf16": { "rtol": 0.016, "atol_rel": 0.008 }
    },
    "exact": false
  },
  "attrs": {
    "EPS": {
      "dtype": "float",
      "default": 0.000001,
      "domain": { "gt": 0 }
    }
  },
  "cases": {
    "names": ["normal", "wide_range", "large", "tiny", "outlier", "ieee"]
  },
  "sizes": {
    "paper": { "M": 4096, "H": 4096, "usage": ["golden", "benchmark"] },
    "small": { "M": 64, "H": 512, "usage": ["golden", "benchmark"] },
    "tail_odd": { "M": 1, "H": 3, "usage": ["golden"] }
  }
}
```

逐项说明：

- `type_vars.T` 显式列出 dtype 覆盖，`instantiate(opspec, dtype, size)` 把请求 dtype 绑到 `T`；不在列表里的
  dtype 拒绝。不写成“由 task 决定”，否则 OpSpec 依赖它本该取代的 task。
- `numerics.tolerances` 按 dtype 物化，数值与今天 `tcenv/task.py::default_tolerances` 相同。不引用函数名，
  否则改那个函数不会改变 `opspec_digest`。
- `uses` 声明公式需要的 attr。引用完整性按这个列表检查，不解析 `formula` 字符串；`formula` 只供人和模型阅读。
- `attrs.*.domain` 用登记的结构化约束（如 `{"gt": 0}`），不写自由文本表达式；JSON Schema 只校验形状，
  语义校验器只接受已登记的约束文法。`formula` 可以自由书写，`domain` 不行。
- `tail_odd` 只用于正确性（`usage` 不含 `benchmark`）。上方的 JSON 块是示例，不是 R1 的
  `tasks/rmsnorm/opspec.json`；R1 复制时必须删掉 `tail_odd`（F-5），否则 `opspec_digest` 会覆盖一个
  R0 没有的 size。
- `flops`/`min_bytes` 的写法（语义推导器或唯一显式表达式）由 R1 选定，示例不预设。
- `opspec_digest` 不是只对上例某些“语义字段”哈希：按主 plan §3.1，它哈希包含完整 `document`、
  `semantic_rules_version` 与 `work_accounting_version` 的 canonical envelope。上例仅改键顺序或空白不
  改变 digest；NFC/NFD 两种写法不归一化，是不同身份；重复键与未知键拒绝。改规则版本即使当前实例相同
  也会改变 digest。SemanticInstance 与 Specialization 的完整派生投影另进入 `task_fingerprint`，不回写成
  第二份 OpSpec。

## 2. RMSNorm 的 Metal track 绑定

track 绑定承载给模型看的文本与评测装配。它由 task-pack digest 与任务指纹绑定，不进入 `opspec_digest`。

```json
{
  "track": "triton-spec-to-msl",
  "opspec": "tasks/rmsnorm/opspec.json",
  "lowering_profile": "metal-rmsnorm",
  "spec_text": {
    "template": "tasks/rmsnorm/kernel_triton.py",
    "placeholders": { "M": "spec_params", "H": "spec_params", "TL": "dtype_spelling" }
  },
  "known_good": {
    "template": "tasks/rmsnorm/naive.metal",
    "placeholders": { "T": "dtype_spelling" }
  },
  "description": "Row-wise RMSNorm: y[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * w; the sum of squares is accumulated in fp32 and the result rounded once.",
  "dtype_spelling": {
    "fp16": { "T": "half", "TL": "tl.float16" },
    "bf16": { "T": "bfloat", "TL": "tl.bfloat16" }
  },
  "reference": {
    "implementation_id": "rmsnorm.reference",
    "implementation_version": 1,
    "params": {}
  },
  "baselines": ["mlx", "torch_mps"]
}
```

- `placeholders` 只列模板实际出现的占位符，每个声明一个来源（主 plan §4.4 的三类之一）。RMSNorm 的
  `backend_params` 为空。
- `kernel_triton.py` 里写死的 `num_warps = 8`（对应 threads=256）与 `eps = 1e-6` 是 R1 为逐字节等价保留的
  字面量。LoweringProfile 将来改 threads 前，先把 `num_warps` 改成 `backend_params` 占位符；OpSpec 的
  EPS、default 或容差改变前，先把 `eps` 等语义字面量改成 `spec_params` 占位符，不能只改一边。
- 契约默认的 `grid=(M,1,1)`、`threads=(256,1,1)`、常量 `M`/`H`（`uint`）与 `EPS`（`float`），以及
  `Contract.gpu`，都由 `metal-rmsnorm` 这份 LoweringProfile 经 `lower` 派生，不写在 OpSpec 或 track 里。
  `lowering_profile_digest` 还覆盖登记的 lowerer 版本和完全解析后的 profile；variant family 另覆盖
  factory id/version，不能只哈希一份配置字典。
- `reference` 直接写登记的 `{implementation_id, implementation_version, params}`；不要写
  `tasks/rmsnorm/task.py::reference` 路径，因为 `reference` 是 `build` 内的闭包，不是模块级符号。
  注册表负责把 implementation_id 解析到实现；可执行身份按主 plan §7.2 的 `canonical_callable_identity`。
- `baselines` 绑定完整登记的 `Task.baselines` 集合，不随 fast-bench 的实际选择子集改变。登记实现进入
  `baseline_implementation_digest`（host 可重算），可用性与实际库版本进入
  `baseline_environment_digest`（测量进程声明；远端确认时冻结搜索期的值）；可用但版本不可识别时环境
  摘要为 `null`，不能缓存或确认。`compiler_baseline`/triton-msl 是后端从缓存估计出的诊断列，不是
  `Task.baselines` 的 runner；只有它的 estimator/缓存身份能由 host 从 task pack 重算时才进入这两份
  摘要，否则只作测量侧诊断，不替代 `speedup`。gen/reference/error_scale/baseline 的 callable 身份使用
  `{implementation_id, implementation_version, params}`，实际代码与依赖由测量一侧的 task-pack 身份
  绑定，不递归序列化 AST、局部 class 或闭包。

## 3. 不属于 OpSpec 的内容

| 内容 | 归属 |
| --- | --- |
| 契约默认 grid/threads、常量目标类型、启动形式、variant 参数、`Contract.gpu` | LoweringProfile |
| 候选 `taic-launch` 自选的 math、核名、每段 grid/threads、workspace | 候选源码，由 `candidate_sha256` 覆盖；math 记入 BuildProvenance 的 `compile_profile` |
| 影响生成代码的 host tiling | LoweringProfile |
| 工具链、SDK、编译选项、build_host | BuildProfile / BuildProvenance |
| 只影响执行的设置 | ExecutionSpec |
| DeviceGroup、driver/firmware、物理拓扑、资源代际 | DeviceGroup profile / topology / resource generation |
| `spec_text` 模板、`description`、dtype 拼写表 | track 绑定 |
| 厂商 baseline 的可执行 runner 与登记实现 | `Task.baselines` 与版本化 baseline registry；摘要归 `baseline_implementation_digest`，不归 build。`compiler_baseline` 是派生诊断估计，另登记其 estimator/缓存或不纳入发布身份 |
| baseline 的可用性与实际库版本 | 测量进程声明；摘要归 `baseline_environment_digest`，远端确认时冻结搜索期值，host 不用自己的库版本 |
| 输入生成、reference/error scale 的可执行函数身份 | 登记 `{implementation_id, implementation_version, params}`；与实际代码/依赖的测量一侧 task-pack 身份一起进入 `evaluation_fingerprint`，不用含地址的 `repr` |
| lowerer 与 variant factory 的实现身份/版本 | `lowering_profile_digest` 的 canonical envelope |
| 参考实现 Python 代码 | `task.py` / reference artifact，不变成 OpSpec 字符串字段；可执行身份另进入 evaluation fingerprint |
| Agent 写的 kernel 源 | candidate artifact，不回写 OpSpec |
| `flops/min_bytes` 的第二份手写值 | 不存在；只保留一个推导器或唯一显式字段 |

## 4. 关键负例

编号供测试与评审引用。F-16 至 F-18 属于后续的 R5b（主 plan §10），不在 R0–R7 验收内。

| 编号 | 输入/变异 | 预期 |
| --- | --- | --- |
| F-1 | 两个 OpSpec 域不同，但在 `paper(H=4096)` 上实例相同 | `opspec_digest` 不同，缓存与确认不得串联 |
| F-2 | 只改 LoweringProfile 的 `threads` | `opspec_digest` 不变；`lowering_profile_digest` 与依赖它的 task/evaluation/cache 身份改变 |
| F-3 | 候选 `taic-launch` 改 threads | OpSpec、LoweringProfile、task/evaluation 指纹不变；candidate 及下游 assembled/artifact/native/cache 身份按实际结果改变 |
| F-4 | `uses` 含 EPS，但 EPS 既无实例值也无 default | 实例化拒绝，不补 0 |
| F-5 | R1 的 OpSpec 里出现 R0 快照没有的任何 size（golden 或 benchmark） | R1 验收失败：R1 不扩大声明域/性能域 |
| F-6 | `H=3` 的 tail 实例 | 能正常实例化并进入 golden；mask/tail 缺失必须被捕获 |
| F-7 | capability 未知或 target 不支持声明的 space/layout | `inconclusive`，不自动换 target |
| F-8 | native build 缺 live handle | `BuildOutcome` 构造即失败，不可能成为 full/native |
| F-9 | simulator/source_only 结果申请确认 | `confirmation_ineligible`，不计 measured |
| F-10 | 纯比较函数收到缺失的远端 BuildProvenance/context；同一份输入若已违反 report/RPC 必填合同 | 前者 `build_identity_unknown`；后者可在更早边界 `result_invalid`，两条路径都不进入确认，不能为得到 unknown 放宽 schema |
| F-11 | 48→40 核：分别改构建期 tiling、编译选项、设备分配 | 至少分别改变 LoweringProfile、BuildProvenance、device group/topology；对应下游身份按依赖更新，三类都使旧测量不可复用 |
| F-12 | 安装 generation 改变但 source 未变 | 身份不同，防止加载旧 `.so` |
| F-13 | Agent 修改 generated/baseline/oracle 路径 | 权限拒绝并记录，不影响评分材料 |
| F-14 | 本地报告 0–3、未知键、缺 `build`、损坏 JSON | 明确拒绝，不当作新报告或未运行 |
| F-15 | 旧 NPU contract `document_version` 测试向量 | 明确拒绝或删除，不以“测试兼容”为理由继续接受 |
| F-16 | 两个 dispatch guard 同时匹配、没有 guard 匹配或 guard 语义未知 | 确定性排序、回退或拒绝，不能猜一个变体；unknown 不得算通过 |
| F-17 | 请求 shape 在声明域外或所需资源超限 | 走已验证 fallback 或明确拒绝，不能把生成 kernel 当通过 |
| F-18 | 只有热门 shape 通过，hidden/tail shape 失败或 fallback 未测 | portfolio 不通过，不能用单点最好值冒充全域成功 |
| F-19 | `flops/min_bytes` 与语义推导不一致 | 校验失败，不选择其中一个静默继续 |
| F-20 | 首个确认候选 `build_identity_mismatch` | `confirmed_status` 不是 `"ok"`，不写 `confirm_report.json`，`kernel_error=False` |
| F-21 | 本地构建抛 `BuildError(deadline)` | 不得落成 crash；本地调用方本就不传 deadline，编译超时由父进程 SIGKILL 记 timeout |
| F-22 | 题面占位符未声明来源，或同名占位符在两个来源都有值 | 拒绝 |
| F-23 | `sem_int_<seed>` 以 `fp32` 请求；以 `fp16`/`bf16` 请求 | 前者拒绝；后者得到 i32 实例（登记的别名） |
| F-24 | 同一 OpSpec 仅改 canonical 键顺序/空白；另改 NFC/NFD 写法、重复键；另分别改 semantic/work-accounting 规则版本 | 键顺序/空白 digest 不变；NFC/NFD 是不同身份，重复键拒绝；规则版本改变 `opspec_digest` 与下游任务/缓存身份，即使当前实例相同 |
| F-25 | LoweringProfile 不变，只改 lowerer 或 variant factory 版本 | `opspec_digest` 不变，lowering 与下游 task/evaluation/cache 身份失效；不得哈希 Python `repr` 代替版本 |
| F-26 | build 缺公共字段/含表外键；partial 带 artifacts；full 带 error_class；缺 evaluation/baseline 顶层键 | 严格 `from_json` 拒绝；不是合法 unknown，不得补默认值 |
| F-27 | candidate、artifact、BuildProvenance 完全相同，只改 baseline runner 或实际库版本 | build 可一致，但 `evaluation_identity_mismatch`，不发布 |
| F-28 | `evaluation_fingerprint`/`baseline_implementation_digest`/`baseline_environment_digest` 为 `null`，或合同允许的 expected 分量不完整；host 部署缺文件/缺登记 | 前者 `evaluation_identity_unknown`，不缓存、不进入 values、不发布；后者是装配/部署错误，不是可发布的 unknown；字符串 `"unknown"` 不得当成相等 |
| F-29 | 同一候选第一个确认批次 match，第二个批次 build 或 evaluation mismatch | 该候选整体失败；第一批不能抵消；若没有其他完整确认的候选，最终 `confirmed_status` 不得为 ok |
| F-30 | 搜索 Candidate 缺 remote context 引用/completed outbox 行；引用存在但签名/result/receipt 损坏（含 `validate_receipt` 的基类 `EvaluationError`，以及重放报告/契约的 `SchemaError`/`ContractError`；outbox 不存 wire descriptor，descriptor 在首次收结果时校验） | 前者 unknown；后者 NeedsInspection/result_invalid；均在 `Orchestrator.restore` 的 outbox 校验或 `_confirm` 入口拒绝，不发布，不重放或补写上下文 |
| F-31 | 搜索与确认报告彼此相同，但当前 task/oracle/baseline expected 已改变 | 三方比较判 evaluation mismatch；不能只比较两份旧报告就发布 |
| F-32 | Orchestrator 记录 allowed，但读盘时缺某批 identity/测量资格记录，或两处批次记录不同 | 共同 `confirmation_publication` 谓词拒绝，catalog 不收录 |
| F-33 | 已确认 run 在当前机器升级 baseline 库后读盘；另一份 confirm_report 缺冻结 expected | 前者只按当时冻结 expected 判定，不 retroactively 改写历史结论；后者不可判、不发布，也不用当前环境补值 |
| F-34 | generated baseline 闭包含 AST、Mamba reference 闭包含局部 class；另一 task 缺实现登记或 task-pack 漏依赖文件 | 前两者用登记实现 + canonical params + task-pack 得到跨进程稳定身份；后两者是装配/部署错误，不得用对象 `repr` 或 unknown 冒充可确认证据 |
| F-35 | 指纹文档含 EPS、容差、`flops` 等浮点；同一数值以 `1e-06` 与 `0.000001` 两种写法出现在源 JSON；另含 `-0.0`；另测 `1` vs `1.0` 与整数值浮点 `16777216` vs `16777216.0` | 按主 plan §3.1 的 canonical 值编码（非整数数值转十进制字符串，round-trip）后摘要一致且可复算；`0.0` 与 `-0.0`、`1` 与 `1.0` 是不同值；同一身份字段若被规则固定为 float，必须先在实例化时统一类型；不得把 float 直接送入既有 canonicalizer |
| F-36 | 远端 `evaluation_context.json` 的 `request_id`/`candidate`/epoch/device_group/build_provenance 与 outbox 行或 result receipt/provenance/evidence 不一致；或 `baseline_environment_digest` 与已接受报告顶层字段不一致；或保留报告非 schema 4/契约非法（`SchemaError`/`ContractError`/`EvaluationError`） | 损坏：NeedsInspection/result_invalid，不发布；指针文件或 completed 行缺失才是 unknown；completed 行保留非 schema 4 报告必须使 `prepare_resume` 返回 `ResumeRefused`，`snapshot_results` 未被调用（或第二次 restore 失败时 `rs.restore_snapshot` 被调用），run 目录未被改写 |
| F-37 | evaluator 在搜索与确认之间升级 baseline 库；host 本地未装该库 | 确认报告的环境分量与冻结值不同，判 evaluation mismatch；host 不得以自身库版本生成 expected 或因此判 unknown |
| F-38 | 某个确认批次计时不确定（本地 `timing_is_uncertain`） | 该批 `status="timing_uncertain"`，整颗候选不确认，`kernel_error=False` |
| F-39 | 确认报告 full 但 `build_kind=simulator/source_only`；确认报告 `cache_hit=true`；本地缓存命中值 `build_kind≠native` | 前两者记 `confirmation_ineligible`，不发布；最后一项按未命中处理，不能作为本地测量；native 的搜索期缓存命中仍是合法左端 |
| F-40 | 非 UTF-8 源到达 builder，而 builder 尚未持有 typed BuildProvenance | 不得生成 partial；要么在 builder 入口改判编译前 `rejected`，要么先取得 provenance 再按 compile 处理 |
| F-41 | 远端 outbox 行的历史 evaluator epoch 与当前 evaluator epoch 不同；投影/行/result 自身一致 | 不是 outbox 损坏：`load_completed_context` 返回历史上下文，R5 比较时按 identity mismatch/unknown 处理 |
