# 学习 KDA/CAKE2 的 Shape Portfolio：域规划、精确 Dispatch 与未见分片验证

状态：**研究结论，未实施**。本文不修改历史 R0–R6 计划，也不改变已完成的 A1–A6、Track、
schema 4、fresh confirmation 或 NPU-P0 状态。清微 NPU 仍只能做流程、数据、协议和身份模拟。

## 1. 结论：模仿组合方式，而不是照搬源码

应当学习 KDA/CAKE2 的 **planner → specialization portfolio → exact dispatch → correct fallback →
domain validation** 组合方式，并把这套运行方式接到本地 Candidate/ExecutionGeometry 与证据底座上。

- **向 KDA 学习**：通用 shape planner 与精确特化共存；每个 retained variant 由规范化
  shape/layout/device/mode key 标识；没有合适 retained variant 时，独立、语义正确的通用实现接管。
- **向 CAKE2 学习**：variant 带 applicability 和稳定 tiling key；候选搜索/演化可以产生多种实现，
  但只有经过独立验证的 variant 才能进入可用组合；按域记录结果，而不是只留下一个全局 best。
- **本地补齐通用控制面**：上游没有可直接搬来的统一 VariantRegistry、DispatchManifest 和
  held-out shape gate，因此这些通用接口由本地实现，但运行行为模仿上述上游模式。

本地已有 Candidate、ExecutionGeometry、CanonicalBundle、独立 fallback geometry、dispatch parity、
TaskLocator 和 schema-4/fresh confirmation。新增工作集中在**候选 ShapePlanner、候选级 variant
组合、精确 dispatch 和未见分片验证**，不需要改变候选内核的表达力。

## 2. 上游源码审计摘要

### 2.1 KDA workflow

- `docs/agent-flow.md:14-48` 是任务、候选、证据和 promotion 工作流，不是 shape registry 或
  dispatcher。
- `skills/ncu-report-skill/helpers/list_flashinfer_workloads.py:131-192` 按 workload tuple 做覆盖
  统计并选择代表点，适合借鉴“每个 dispatch path 都要有证据”的方法，但不是隐藏形状验证。
- `scripts/export_huggingface.py:70-90,271-273` 提供不可变快照/包身份方法，不提供 variant
  applicability 或域外执行语义。

### 2.2 KDA kernels

- `kda-cake-cute/impl/planner.py:179-340` 有实际 shape-only 规划；`impl/forward.py:902-1006`
  用 shape/layout/device/mode 形成 plan cache key。可借鉴规划与缓存身份，不能据此宣称存在通用
  registry。
- `kda-cake-ptx/kda_ptx/scheduler.py:1425-1437` 与 `fallback.py:17-107` 展示“无 retained module
  就走独立正确实现”的清晰边界；`static_runtime.py:199-230` 仅让选定配置返回 retained module。
- `kda-tirx-bwd/kda_backward_packed.py:12446-12540` 按整除、完整 chunk、head 关系等语义条件选择
  特化或通用路径，是较完整的 portfolio 结构示例。
- `bench.py:54-64` 使用固定形状；`--seed` 只改变输入值，不改变 shape。可见 tail/boundary 测试也
  不等于 held-out shape 证明。

### 2.3 CAKE2

- `agents/cake-evo.md:288-297,356,565-581` 的 variant 主要是搜索分支、trace 和最终 best 路径，
  不是 shape-indexed persistent registry。
- `skills/ascendc-evaluation/scripts/evaluate.py:228-249,634-685` 对固定 test cases 评测并聚合，
  没有训练/隐藏形状划分或未见组合验证。
- `skills/cake-evo/references/meta_prompts/strategies/perf_02_adaptive_tiling.md:477-491,621-648`
  和 `dtype_03_tilingkey_dispatch.md:182-214` 展示 applicability、稳定 key、精确分发的方法；
  它们是策略示例，不是可移植的通用基础设施。

## 3. 本地已有资产与真实缺口

| 能力 | 可复用资产 | 缺口 |
|---|---|---|
| 语义域 | OpSpec v2 完整 digest、`sizes.usage`、显式 dtype/size 拒绝（`tcenv/opspec/v2.py`） | 没有把 shape/layout/资源规划成 tune/validation shards |
| 候选与变体 | `CandidateArtifact`、`CanonicalBundle`、candidate/geometry/bundle/artifact 身份 | 没有 variant guard、variant_id 或组合注册 |
| 几何与执行 | `ExecutionGeometry.when`、closed expression evaluator、实际 dispatch/provider parity | `when` 只控制单个候选内部 stage，不是 one-of-N variant dispatch |
| Fallback | 独立 fallback ExecutionGeometry 和独立身份 | 它是单个构建的几何备选，不是声明域未命中时的厂商/旧实现 fallback |
| 验证 | 固定 case、fresh input batch、G0 和真实 Mac 评测 | fresh seed 不换 shape；没有 unseen shard、tail/资源边界和 fallback-rate 验收 |
| 恢复与证据 | TaskLocator、run_config、resume、schema 4、fresh confirmation | 当前恢复单个 best，不恢复 domain plan、variant registry 和 dispatch manifest |

`tcenv/track/registry.py` 仍必须只描述交付机制。不得把 op/shape/tiling 塞进 Track，也不得引入
`PatternResolver`、`LoweringRuleRegistry`、`StagePartitioner` 或 `TileConnector`。v2 对 `@variant`
的拒绝应继续保留：它防止第二份任务合同，并不排斥**候选 Bundle 内部的 portfolio**。

## 4. 模仿模型：候选规划，harness 验证，通用实现兜底

运行方式直接对应 KDA/CAKE2 的方法：

1. 候选 `ShapePlanner` 读取 Semantic Domain，按 shape/layout/资源与候选自己的算法结构规划
   coverage buckets，并决定每桶使用什么实现、Tiling、Host Tiling 和 ExecutionGeometry。
2. 每个 bucket 形成带 applicability guard 与稳定 key 的 variant；多个 variant 可以共存，不再只保留
   一个全局 best。
3. 请求到来时按规范化 key 精确匹配 retained variant；匹配失败时走候选声明且经过验证的通用正确
   实现，或明确拒绝。
4. harness 不替候选设计 tile，而是独立检查 domain、guard、identity、exactly-one dispatch、fallback
   和证据，并用候选未用于规划的 shape shards 验证泛化。

这相当于把 KDA 的“retained specializations + generic correct path”和 CAKE2 的“applicability +
variant key + 独立评测”组合到同一个候选交付模型中。

### 三个“域”必须分开

1. **Semantic Domain（语义接受域）**
   - 由 OpSpec 定义：什么 dtype、shape、layout、attrs 和数值语义合法。
   - 域外请求只能 `unsupported_contract_capability`，不能用厂商实现把非法语义变成合法。
2. **Candidate Coverage（候选声明覆盖域）**
   - 由候选 portfolio 的 variant guards 与 fallback 声明。
   - 候选可以自由决定算法、融合、Tiling、Host Tiling 和 ExecutionGeometry；harness 不替候选
     选择这些实现。
3. **Validated Coverage（证据覆盖域）**
   - 由独立验证计划与实际 fresh evidence 决定。
   - 声明覆盖不等于已验证；有限采样不等于连续全域证明。未验证区域只能拒绝或走已登记、已
     测试的 fallback，不能计作生成 kernel 成功。

这三个域的身份必须分别进入 `opspec_digest`、`portfolio_candidate_digest` 和
`dispatch_manifest_digest`，不得混算。

## 5. 推荐组件

### 5.1 SemanticDomainPlan

由 OpSpec 与 TargetAbiLimits 派生，不读取候选 tiling：

```json
{
  "schema_id": "tcenv.shape-domain-plan.1",
  "planner_version": 1,
  "opspec_digest": "...",
  "target_abi_digest": "...",
  "normalization_version": 1,
  "domain_kind": "finite|sampled",
  "validation_shards": [
    {"shard_id": "...", "usage": "golden|benchmark"}
  ]
}
```

职责：

- 规范化 shape/layout/stride/dtype/attrs 和资源边界；
- 区分 correctness 与 benchmark，不让 golden-only shape 进入性能声明；
- 明确有限枚举域与不可穷举域；
- 为独立验证提供域描述，不为候选选择 tile/grid/threads。

### 5.2 Candidate ShapePlanner + PortfolioCandidate（候选 Bundle）

候选先由自己的 ShapePlanner 生成 `coverage_plan`：它模仿 KDA 的 shape-only planner，把语义域分为
候选认为有意义的 buckets，并为每桶选择实现。这个规划输出属于候选，不属于 Track 或 harness 的
算法库。harness 随后只验证它是否明确、可求值、可构建和可证明。

候选新增可选 `portfolio.json`，不需要新 Track，也不改变 OpSpec：

```json
{
  "schema_id": "tcenv.candidate-portfolio.1",
  "domain_plan_digest": "...",
  "coverage_plan": {
    "planner_id": "candidate-owned-planner",
    "planner_version": 1,
    "buckets": [{"bucket_key": "...", "guard_digest": "..."}]
  },
  "variants": [
    {
      "variant_key": "diagnostic-name-only",
      "guard": {"expr": "...", "digest": "..."},
      "entries": ["device/a.metal", "host/execution_geometry.json"]
    }
  ],
  "fallback": {"kind": "registered_implementation|refuse", "implementation_id": "..."}
}
```

候选拥有 coverage plan、guard 和实现。Guard 只描述适用域，可使用现有 `track/expr.py` 的闭集求值器；
它不是 kernel pattern，也不限制循环、状态机或 persistent 实现。多阶段 `ExecutionGeometry.when`
继续控制 variant 内部 stage，不与 portfolio guard 混用。

### 5.3 VariantRecord 与身份

每个 variant 至少绑定：

- `domain_plan_digest`、canonical `guard_digest`；
- `candidate_sha256`、`geometry_identity`、`bundle_digest`；
- 含冻结几何的 `artifact_set_digest`；
- evaluation identity、target/device generation 和独立 fresh confirmation。

`variant_id` 应由上述 canonical 内容派生，人类名字只作诊断。v2 不再使用 LoweringProfile 或
variant factory 作为 variant 身份。

候选提交不能直接成为 active registry。先生成 `PortfolioCandidate`，通过构建、正确性、dispatch
和隐藏分片验收后，才派生可激活的 `DispatchManifest`。

### 5.4 ExactDispatchResolver

输入不只是 shape，还包括规范化语义实例、layout/stride/attrs、target_id 和 ABI/resource
generation。解析顺序必须固定：

1. 请求不在 Semantic Domain：`semantic_unresolved` 或 `unsupported_contract_capability`；
2. 在 Candidate Coverage 内求值所有 active guards；
3. 多匹配：`dispatch_overlap`，不得靠优先级猜；
4. 恰好一个：核对 variant/evidence 身份，执行该 variant；
5. 零匹配：只有已登记且适用于该请求原因的 fallback 才能执行，否则 `dispatch_unmatched`；
6. guard/请求字段未知或证据不可判定：`dispatch_unknown`，不是候选成功。

注册期也要检查 guard 重叠、未知字段、空 variant、不可规范化 guard 和缺证据。运行期不能因为
注册期检查过就跳过实际 identity/selection/dispatch parity。

### 5.5 DomainFallback

它与当前独立 fallback ExecutionGeometry 不是同一职责，必须单独建模：

- `kind=registered_implementation`：有实现 ID/version、canonical params、task-pack/evaluation 身份；
- `kind=refuse`：明确拒绝；
- 触发原因区分 `outside_candidate_coverage`、`resource_unavailable`、`variant_unavailable`；
- Semantic Domain 之外永远拒绝，不触发 fallback；
- fallback 自身必须过 correctness 和必要的 negative/tail 测试；
- 结果记录 `provider=fallback`、原因和实现身份，不得把 fallback 时间或成功计为生成 variant 的
  correctness/speedup。

现有 baseline registry 是计时基线，不自动成为 correctness fallback oracle。

### 5.6 UnseenShardValidation

验证分片由 harness 独立产生，不能由候选 planner 选择，也不能进入 task pack 或 agent guidance：

- 训练/调优分片与最终验证分片分开；
- 验证使用 fresh seed、guard threshold±1、奇数/tail、资源边界和未见 shape/layout 组合；
- `hidden` 表示未用于生成和调参，事后必须记录 planner_version、seed 和确切 shards 以供 replay；
- 有限离散域可声明 `exhaustive`；不可穷举域只能声明 `sampled`，不得写成全域证明；
- 每个声明 fallback 触发原因都必须实际构造负例；
- 未见分片失败、未匹配、多匹配或证据 unknown 都不能被平均值掩盖。

### 5.7 PortfolioConfirmation 与 DispatchManifest

逐 variant 仍使用现有唯一 builder、schema 4、evaluation identity 和 fresh confirmation。组合层新增
独立 `portfolio_confirmation.json`，不向 schema 4 塞自由字段。

建议派生：

```text
portfolio_set_digest = SHA256(canonical({
  domain_plan_digest,
  ordered [guard_digest, candidate_sha256, geometry_identity, artifact_set_digest],
  fallback_identity,
  dispatch_rules_version
}))

dispatch_manifest_digest = SHA256(canonical({
  portfolio_set_digest,
  validation_plan_digest,
  per_variant_confirmation_digests,
  activation_generation
}))
```

`DispatchManifest` 只包含已通过的组合证据与明确状态（candidate/validated/active/revoked）。
resume/replay/observe 以 `dispatch_manifest_digest` 为主键之一，并记录每次实际 resolution 的
variant、provider、fallback reason 和证据状态。

性能结果按 shape/shard 和 variant 展示；可以另给聚合值，但必须同时给出最差项、失败、unknown、
fallback rate 和覆盖类型。不得用单点最好值或一个 geomean 代表全域。

## 6. 建议的最小垂直切片

先用一个真实 Metal 题做两条候选 variant + 一个独立 fallback：

1. 同一 OpSpec/reference/ABI，候选 A 覆盖小维度，候选 B 覆盖另一段维度；
2. threshold-1/threshold/threshold+1 与奇数 tail 验证 exactly-one dispatch；
3. 域内零匹配走已登记 fallback，域外请求明确拒绝；
4. 构造 guard overlap、未知 shape 字段、失效 artifact、fallback 误记 candidate 等负例；
5. 最后加入未参与 variant 设计的 fresh shape shard，并做 fresh rebuild/confirmation。

优先复用 `msl-bundle` 的多文件交付能力，不新增 `msl-portfolio` Track。`msl-single` 的现有单
候选语义保持不变。

## 7. 推荐实施顺序

- **P0：合同与负例（离线）**：DomainPlan、PortfolioCandidate、VariantRecord、guard/identity、
  exact resolver 和 fallback 类型；不跑 GPU/NPU。
- **P1：真实 Mac dispatch 垂直切片**：两 variant + fallback，真实 build/correctness/actual dispatch
  parity；先固定 CPU 负例，再跑真实 Metal。
- **P2：隐藏分片与组合证据**：独立 validation plan、portfolio confirmation、dispatch manifest、
  resume/replay/observe。
- **P3：有限任务推广与性能治理**：逐 shape 性能对照和回归门，不能用总平均值隐藏局部退化。
- **P4：清微离线模拟**：只复用域/身份/协议测试；真实设备、工具链、baseline 和计时合同齐备前，
  NPU-P0 继续 blocked。

## 8. 实施前仍需明确的问题

1. Guard 规范化需要覆盖哪些 layout/stride/attrs，现有表达式是否足够，缺失时应扩展表达式而不是
   放任自由文本。
2. 部分 Candidate Coverage 是否允许激活；若允许，未覆盖区域必须有什么 fallback 证据。
3. registered fallback 的实现注册与厂商 baseline 如何区分，避免把计时 baseline 误当 correctness
   fallback。
4. 隐藏分片种子由谁保管、何时向候选公开，以及如何证明没有进入 task pack/guidance。
5. DispatchManifest 的激活/撤销代际如何进入 run_config/resume，而不修改六 RPC、field number、
   HMAC、BlobChannel 或 schema 4 自由字段。

解决这些问题后，实现应看起来像 KDA/CAKE2 的 shape portfolio：候选规划并保有多个特化实现，
精确 key 决定执行路径，通用正确实现处理未覆盖输入，独立未见分片决定这些能力是否真正可信。
