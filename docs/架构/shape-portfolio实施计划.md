# Shape Portfolio 实施与验收计划

状态：**P0 已完成（completed：新验收 epoch `p0-reacceptance-baseline-1` 下从头复验通过并已独立复验签署
`done=true`）；P1 已完成（completed：RLAR 以 `pi/kimi-code/k3.1-pro:xhigh` 为 actor、
`pi/deepseek/deepseek-flash:xhigh` 为 reviewer 跑满 5 轮，逐轮发现并修复阻断 20–24，reviewer 独立重写攻击脚本后
签署 `done=true`，主会话再独立重跑完整默认验收（189/189 负例、18 CPU + 6 GPU 无 skip、live==archived）；
二十四项历史阻断均保留在 P1.md §8 未回改）；
**P2-0 已完成（completed：严格证据模型、确定性重投影、信任根扰动与自动变异门均通过独立复核）；P2 主体已完成（P2-A–E 全部完成，P2-F 安静窗口重录已完成；历史 start-surface 锚点仍拒绝漂移）；P3 已完成（P3-A/B/C 全部完成：两个任务的完整有限域 9 个 benchmark 分片 ratio 均 ≤1.10，两本账与完整设备/dispatch/四重身份已归档并通过 live 对照；历史阶段起点基线按原样保留）；P4 已完成（有限域 exact dispatch 接入现有 fake-NPU source/bundle 协议；证据保持 simulated=true、publication_allowed=false、npu_p0.status=blocked，不解除 NPU-P0）。** 目标行为学习 KDA/CAKE2：候选 ShapePlanner 规划 coverage
buckets，多个特化 variant 按稳定 key 共存，exact dispatch 选择唯一实现，通用正确实现处理未覆盖
输入，独立未见分片验证泛化。

本计划不恢复 LoweringProfile、`@variant` 或逐题 tiling registry，不修改既有 R0–R6/A1–A6 历史证据。
基线为当前 post-A6 工作树；每个阶段开始时记录完整 HEAD、工作树 manifest 和 SHA-256，不能用旧 HEAD
冒充当前基线。P0 的**原始开始时**快照当时未记录且不可重建（见 P0.md §7.2），因此按 P0.md §7.2 的
第二条路重建了真实新基线（`P0/acceptance-baseline.json`，epoch `p0-reacceptance-baseline-1`）并把 P0
验收作为新阶段从头重做；**不是**书面豁免，也不声称原始起始基线存在。

## 1. 不变量与范围

- 候选拥有 ShapePlanner、coverage plan、算法、融合、Tiling、Host Tiling 与 ExecutionGeometry；
  harness 拥有语义域、规范化请求、精确 dispatch 校验、测量边界和证据。
- 不执行候选 planner 的任意宿主代码。生产输入是 planner 已生成的 `coverage_plan/portfolio.json`；
  harness 只解析、验证并执行闭集 guard。
- 不新增 `PatternResolver`、`LoweringRuleRegistry`、`StagePartitioner`、`TileConnector`，不按
  op_name/shape 给 Track 添加算法。v2 `@variant` 继续拒绝。
- 首个实现使用**外层 portfolio envelope + 子 CandidateArtifact 投影**，不新增
  `msl-portfolio` Track。母包先由 `tcenv/portfolio/candidate.py` 解包，绝不直接交给
  `CandidateArtifact.validate_for` 或 TrackRegistry。P0 不改两条 MSL Track 文档，也不把
  op/shape 写进 Track。live `backends/metal/tracks.json` 后来在 MT epoch 增加了
  `tilelang-pipeline`；P0 当时的 registry 字节留在 MT1b 归档 `tracks-v1-epoch.json`，不要回填。
  两条 MSL Track 文档的 digest 仍由 `tests/shared/test_shape_portfolio_candidate.py` 的
  `TRACK_DIGESTS` 钉住。
- 复用唯一 `artifact_builder`、CandidateArtifact/CanonicalBundle、schema-4 EvalReport、evaluation
  identity、fresh confirmation 和唯一发布门。六 RPC、field number、HMAC、BlobChannel 不变。
- 每阶段使用固定人工候选，不调用模型搜索，不运行 `taic run/resume` 或 `Orchestrator.run`，不做
  付费实验。真实 Mac GPU 仅在对应阶段获得执行许可后运行；清微只做离线模拟，NPU-P0 始终 blocked。
- 每个阶段都有 `verify.py`、`manifest.sha256`、稳定正负例 node id、验收文档和独立 reviewer；
  必要 GPU 节点不得 skip。阶段未通过不得进入下一阶段。

## 2. 先冻结三个域和三个身份

### 2.1 三个域

1. **Semantic Domain**：OpSpec 接受的 dtype/size/shape/layout/attrs；域外只能拒绝。
2. **Candidate Coverage**：候选 variants 与 fallback 的 guards 声明能覆盖什么。
3. **Validated Coverage**：独立验证和 fresh evidence 实际证明了什么；声明不等于验证。

首个实现只支持**有限、显式命名的 OpSpec instances**。未声明 shape 不会因为数值接近就进入语义域。
`domain_kind` 只登记 `finite|non_finite` 两个值，激活集合只有 `finite`，所以 `non_finite` 现在被
`UnsupportedContractCapability` 直接拒绝，不会被当成 sampled-finite（`tcenv/portfolio/domain.py`）。
验证侧没有 `validation_kind` 字段；实际落地的是 `coverage_kind`：portfolio 注册报告
（`RegistrationReport`）记 `exhaustive_finite_point_check`，confirmation/manifest subject 记
`sampled_unseen|tune_confirmation`。
非有限域的激活留到以后真正的 OpSpec 域扩展，不在本计划偷跑。采样证据永远不得宣称全域证明。

### 2.2 三个身份

- `variant_id`：候选声明身份，含 domain plan、canonical guard、variant entries、geometry declaration
  和 planner 身份；不含测量结果，也不含 instance-resolved geometry。
- `evaluation_binding_digest`（代码里是 `EvaluationBinding.identity`）：某个 variant/fallback 投影后的
  构建前身份。P0 只绑定 provider、variant/fallback id、domain/portfolio digest、parent portfolio SHA、
  `candidate_sha256`、track、几何原路径与内容 SHA 和 entries；bundle/artifact/native/evaluation/target
  身份由 runner 与 confirmation sidecar 追加，不在这一个摘要里。
- `dispatch_manifest_digest`：已验证组合、validation plan、逐 variant confirmation 和 activation
  generation 的身份。

`GeometryPair.identity` 含实例解析结果，不能代替稳定 `variant_id`。人类 `variant_key` 只作诊断。

## 3. 合同与模块边界

### 3.1 SemanticDomainPlan

新增 `tcenv/portfolio/domain.py` 与 `schema/semantic-domain-plan.v1.json`：

```json
{
  "schema_id": "tcenv.semantic-domain-plan.1",
  "normalization_version": 1,
  "opspec_digest": "...",
  "target_generation_digest": "...",
  "domain_kind": "finite",
  "instances": [
    {
      "instance_key": "...",
      "dtype": "fp16",
      "size": "small",
      "usage": "golden|benchmark",
      "semantic_instance_digest": "...",
      "target_abi_digest": "..."
    }
  ]
}
```

只含语义实例和 ABI，不含候选 tile，也不含事后隐藏分片。规范化 dtype/layout/stride/target 是精确
key 字段；初版 guard 只对 symbols 和 integer attrs 使用现有 `track/expr.py` 闭集整数表达式。

Guard 统一使用 `expr.py` 已有的**函数式/JSON 语法**，例如
`and(ge(N, 64), eq(rem(N, 2), 1))`，不新增中缀解析器。`guard_digest` 必须哈希规范化后的
`expr.Call` AST，不得哈希原始字符串。规范化只允许纯结构重排：`gt(a,b) → lt(b,a)`、
`ge(a,b) → le(b,a)`，扁平化并排序 `and`/`or`，可交换操作数按 canonical bytes 排序；不得把
`a-b<0` 改成 `a<b`，不得消项、展开或改变求值顺序。交换合取项等句法等价形式必须得到同一
digest；canonicalization 不宣称一般逻辑等价或 SMT 可判定性。

现有 `/` 的**精确整除语义保持不变**，不得为 portfolio 改成截断除法。P0 在 `expr.py` 新增显式
`trunc_div(a,b)` 与 `rem(a,b)`：向零截断、余数符号跟被除数，除零抛 `GeometryError`，dispatch
层转换为 `dispatch_unknown`。负数、零除数和边界值 golden 验证同一套跨后端语义；奇数 tail 和
对齐判断只能使用 `rem`，不能借用 `/`。

### 3.2 PortfolioCandidate

新增 `tcenv/portfolio/candidate.py` 与 `schema/candidate-portfolio.v1.json`。外层 portfolio
envelope 的固定入口为 `coverage_plan/portfolio.json`：

```json
{
  "schema_id": "tcenv.candidate-portfolio.1",
  "domain_plan_digest": "...",
  "planner": {"id": "candidate-owned", "version": 1},
  "coverage_plan": {
    "buckets": [{"bucket_key": "...", "guard_digest": "..."}]
  },
  "variants": [
    {
      "variant_key": "diagnostic-name",
      "bucket_key": "...",
      "guard": {"expr": "..."},
      "entries": ["device/a.metal"],
      "geometry_entry": "host/variant_a_geometry.json"
    }
  ],
  "fallback": {
    "kind": "registered_implementation|refuse",
    "implementation_id": "candidate-local-generic",
    "implementation_version": 1,
    "entries": ["device/generic.metal"],
    "geometry_entry": "host/generic_geometry.json",
    "reasons": ["outside_candidate_coverage", "resource_unavailable"]
  }
}
```

`registered_implementation` 是候选 Bundle 内登记的通用实现，不是全局 lowering registry。
`refuse` 不得携带 entries。

母包不是 `msl-bundle`，也不含 Track 可识别的特殊 role；它在任何 Track 校验之前由 portfolio
parser 解包。每个 variant/fallback 再机械投影成普通 `CandidateArtifact`，其 geometry entry 投影为
`host/execution_geometry.json`，再复用现有 `evaluate_task`：

- 含 `host_tiling` 的子包走 `msl-bundle`，必须恰有一个 host entry，且对应测试 OpSpec 显式声明
  workspace；
- 不含 `host_tiling` 的 variant/generic fallback 走 `msl-single`，且只能有一个自包含 device entry；
  多源文件 variant 必须满足 `msl-bundle` 的 Host Tiling/workspace 条件，否则拒绝，不得伪造 host plan、
  workspace 或把多个 entries 塞进 single-source track；
- 两条路径的 `.metal` 源码都必须保留候选自己的 `taic-launch`，因为现有 MetalV2Executor 还会用它
  交叉检查 workspace/launch；
- MSL 子包的 track 只由实际交付内容决定，不按 op_name/shape 选择实现。调用方传入已登记 DSL Track 时，`project_provider` 保留该 DSL 身份；DSL 源若被收成 MSL，抛 `UnsupportedContractCapability`。portfolio 仍不改 registry 文件，也不新增 `msl-portfolio`。

`entries` 必须列出该 variant 编译所需的全部设备源码/公共源文件；除指定 `geometry_entry` 外，其他
entries 的相对路径、角色、media_type 和字节完全不变，不得以只拷一个主源文件的方式丢掉公共依赖。
映射后路径冲突、重复 geometry、未声明依赖或原始路径逃逸均拒绝。`EvaluationBinding` 同时记录原
geometry_entry 路径与内容 SHA-256。投影不得修改候选字节、丢失 Host Tiling、放宽既有 include/宿主
路径安全边界或复用 parent portfolio SHA。legacy 非 portfolio bundle 不进入该路径。

### 3.3 Exact dispatch

新增 `tcenv/portfolio/dispatch.py` 与 `schema/portfolio-dispatch-record.v1.json`，组合层记录类型
命名为 `PortfolioDispatchRecord`，不得与 `v2_runtime.DispatchRecord` 的单 stage 记录混用。解析顺序
固定：

1. 请求不可解析：`semantic_unresolved`；
2. 请求超出 OpSpec：`unsupported_contract_capability`，永不 fallback；
3. guard 字段/求值/证据不可判定：`dispatch_unknown`；
4. 多匹配：`dispatch_overlap`，不得按优先级猜；
5. 恰好一个：选择 candidate variant；
6. 零匹配：选择已注册且原因适用的已验证 fallback，否则 `dispatch_unmatched`。

fallback 原因必须来自实际执行阶段，而非候选自报：所有 guards 均为 false 时固定为
`outside_candidate_coverage`。恰好一个 guard 命中后，`portfolio_runner.py` 必须先调用
`resolve_geometry(variant_geometry, instance=..., contract=abi, limits=limits)` 做纯 CPU 预检；若超出
`TargetAbiLimits`，原因固定为 `resource_unavailable`。只有 fallback 已声明该原因，才切换为投影后的
fallback CandidateArtifact 并调用 `evaluate_task(..., fallback_geometry=None)`；否则该 variant 直接
失败，不得进入 `evaluate_task` 后等待默认 `select_geometry` 抛错。多个 guard 命中仍是 overlap；
fallback 自身资源不足不得再次链式回退。每次记录 matched variant、失败原因和最终 provider。

恰好一个 variant 命中但预检不是 `ok` 时，记录 `variant_unavailable`：`resource_unavailable` 而
fallback 未声明该原因、几何 `invalid`（候选缺陷，不切换 fallback），或 fallback 自身预检失败，都落在
这里，且都不会进入 `evaluate_task`。`variant_unavailable` 同时也登记在 fallback 原因词表里，但现行
resolver 只从执行阶段派生 `outside_candidate_coverage` 与 `resource_unavailable` 两种原因。

有限域注册时枚举所有 instances，检查 overlap、空 variant、未知 bucket、guard 与 entries；这只是
**有限点集查重**，不引入 SMT 或一般静态终止/互斥证明。运行期仍逐个求值所有 guards，多匹配始终
`dispatch_overlap`。非有限域不得借有限测试宣称静态无重叠，本计划不激活。

### 3.4 DomainFallback

它不是当前 `GeometryPair` 的执行几何备选，也不是计时 baseline。注册角色必须显式区分
`timing_baseline` 与 `domain_fallback`：只有通过 OpSpec golden reference 精度验收、声明自身适用域
和独立身份的实现才能成为 `domain_fallback`。它是独立投影、独立 correctness/confirmation 的实现。
结果必须记录 `provider=candidate|fallback` 与触发原因；fallback 的成功和时间不得记为生成 variant
的 correctness 或 speedup。Semantic Domain 外永远拒绝。

### 3.5 ValidationPlan / EvaluationBinding / Confirmation / Manifest

新增 `tcenv/portfolio_validation/validation.py` 与 `taic/{portfolio_manifest.py,portfolio_confirmation.py}`，
各自带闭集 schema id（`tcenv.portfolio.freeze.1` / `.validation-plan.1` / `.candidate-tune-view.1` /
`.subject-confirmation.1` / `.dispatch-manifest.1`）：

- `ValidationPlan`：`derivation_version`、`selection_mode`（`hmac-sha256-master-seed-v1`）、
  `portfolio_freeze_digest`、`opspec_digest`、`tune_shards`、`unseen_instance_keys`、
  `required_instance_keys` 和逐 unseen key 的 `fresh_seeds`。生成与调优阶段的 `tune_shards` 是另一份
  显式视图，不得包含 unseen shards；候选只能拿到 `CandidateTuneView`。没有 `generator_version`、
  `validation_kind` 或 `shards[]` 字段。
- `EvaluationBinding`：variant/fallback identity、投影 candidate SHA、几何原路径与内容 SHA、entries、
  parent portfolio SHA 与 track；bundle、artifact_set/native digest、evaluation fingerprint、
  target/device 身份由 runner 和 confirmation sidecar 记录。
- `PortfolioConfirmation`：原子记录单位是 **`(variant_or_fallback, instance_key)`**，保存原始
  search/confirmation 两份 schema-4 report、`search_seed`、`confirmation_seed`、`confirmation_attempts`、
  身份比较、`coverage_kind` 和覆盖全文档的 `confirmation_digest`。`status`/`reason` 由两份原始 report
  与 expected 重算，不采信归档值；sidecar 没有 `cache_hit` 字段，判定要求两份原始 report 的
  `cache_hit` 都为 `false`。
  不允许跨 instance_key 比较 lowering/evaluation identity。原 EvalReport 原样保存，组合证据是 sidecar。
- `DispatchManifest`：`portfolio_freeze_digest`、`validation_plan_digest`、按 `instance_key` 排序的
  `subjects`（每个带 provider、variant/fallback id、`candidate_sha256`、`credits_variant`、
  `coverage_kind` 和 `stable_confirmation_digest`）、`activation_generation`、
  `validated|active|revoked` 状态和 `revocation_reason`（只有 revoked 非 null）；
  `dispatch_manifest_digest` 覆盖上述全部。
  没有 `portfolio_set_digest` 或 `dispatch_rules_version` 字段（后者在 `PortfolioDispatchRecord` 上）。

只有 `active` 可用于新执行；旧 digest 可 replay，但 revoked active pointer 不可继续执行。

## 4. 分阶段实施

### P0：闭集合同、候选投影与离线 exact resolver

**状态：completed（2026-11-01，新验收 epoch `p0-reacceptance-baseline-1` 下完整复验通过，独立复验签署
done=true）。** 验收：`tests/support/golden/shared/shape-portfolio-阶段验收/P0.md` 与 `P0/verify.py`（epoch 基线 44 文件 +
36 个固定 CPU 节点 + 14 项合同自检 + 相关 safe regression；无 GPU/性能/active 证据）。

阻断项（reviewer 2026-11-01，均已在代码与测试中修复，等待独立复验）：

1. `DispatchRequest.document()` 输出 `instance_key` 但闭集字段不含它，导致合法文档不能重放；
2. 请求字段未与权威 `SemanticInstanceV2` 绑定（`str()` 静默强转、op_name/symbols/attrs 未核对，
   矛盾请求可能 resolved/fallback）；
3. 上述“不可解析”负例被共同的前置字段集错误掩盖，未真正覆盖各自分支。

第二轮复验又发现（均已修复，待复验）：

4. 域计划未绑定权威 OpSpec/target/instances，伪造 `opspec_digest`/`target_generation_digest`/
   逐点 ABI 摘要后可重新绑定并继续 dispatch；
5. canonical AST 只排序 `eq/ne`，`+`、`*`、`min`、`max` 未排序；
6. 缺少“旧 bundle 无 `coverage_plan/portfolio.json` 不得进入 portfolio 路径”的固定负例，且
   P0.md 节点数与摘要漂移的结果描述不一致。

第三轮复验又发现（均已修复，待复验）：

7. `project_provider` 的 `workspace_declared` 只是调用方自述，未从权威 OpSpec/ABI 验证 workspace，
   无 workspace 的 OpSpec 仍可伪造 `msl-bundle`；
8. `verify_historical` 未校验 A4–A6 manifest 中列出的 evidence 文件；
9. P0.md §1/§3.2 仍写只排序 `eq/ne`，与 `{eq,ne,+,*,min,max}` 实现不一致。

第四轮复验又发现（前两项已修复，第三项保持开放）：

10. fallback 缺少 “geometry_entry 不得同时在 entries” 检查，且 variant 可把其它 provider 的
    geometry 作为普通 source 交付，同一几何被复制并冒充 Host Tiling；
11. 投影边界只检查引号 include，忽略角括号 include（`#include </etc/passwd>` 可通过）；
12. **P0 起始基线未记录**（计划要求开始时的完整 HEAD + 工作树 manifest + SHA-256），事后无法真实
    重建，不能用当前清单或旧 HEAD 冒充。

第五轮复验又发现（已修复）：

13. `_require_launch_retained` 只做 `b"taic-launch"` 子串匹配，`// taic-launch` 可伪造 launch；已改为
    复用 `tcenv.backends.metal.launch.unique_launch_block` 的扫描语义（行注释/字符串/重复块拒绝）。

第六轮复验又发现（已修复）：

14. `unique_launch_block` 仍用前缀匹配，`/* taic-launch-fake */` 可冒充 launch；已改为首 token 精确
    等于 `taic-launch`，非 launch 块中的任何标记提及都拒绝（含真实块旁的非首 token 提及）；
15. `DELIVERABLES` 漏掉被改的 `tcenv/backends/metal/launch.py`，相关回归漏掉
    `tests/backends/metal/test_launch.py`；已补入（manifest 18 文件、回归 10 文件 70 passed）。

阻断 12（P0 起始基线未记录）在第四至第九轮**一直开放**：只能由计划 owner/reviewer 明确接受缺口
并书面豁免，或重新建立基线后作为新阶段重做验收；不得用当前清单/审计快照/旧 HEAD 冒充。本轮（第十轮）
按第二条路关闭：用户只说“继续”而未给书面豁免，故**不使用豁免**；先在任何新代码改动之前用
`P0/baseline_epoch.py --write` 对当前真实工作树建立 epoch 基线 `p0-reacceptance-baseline-1`
（完整 HEAD `4a93ed2a4934fe526eedfcf4577de5dbd366e24a`、`git status` 快照、44 个合同面文件 SHA-256、
`tracks.json` 摘要与现场重算的 registry/Track digest），记录里明确写明“这是新验收阶段的真实起点，
**不是**此前缺失的 P0 起始基线”；然后用 `verify.py` 独立校验该基线（记录字节被 pin、逐文件重哈希、
文件集/路径指纹/分组钉死、且要求每个覆盖文件与**更早的**审计点 `baseline_audit.sha256` 逐条相等），
最后把 P0 验收作为新阶段从头完整重做。生产与测试合同面本轮零改动，所以没有第二个 epoch；原始起始基线
仍然不存在，历史缺口在 P0.md §7.2 如实保留。

第七轮复验又发现（已修复）：

16. `_pytest` 用子串判断通过数（`"36 passed" in "136 passed"` 为真）；已改为从 pytest summary 解析
    整数并按相等校验，合同自检加入 `136 != 36` 负例（自检项由 11 增至 12）。

第八轮复验又发现（已修复）：

17. `DispatchRequest` 未执行登记 schema 的 op_name/dtype/size 长度上限，可产生 `$.request.size` 非法
    的记录；已改为超限请求 `semantic_unresolved`，并把 detail 截到 schema 上限（自检项 12→13）。

第九轮复验又发现（已修复）：

18. `msl-bundle` 只检查 `host_tiling` role，未复用 `HostTilingPlan.from_candidate`，且正例 fixture 不是
    合法 host tiling JSON；已改为真实 JSON 并在投影边界复用既有闭集校验（含非 JSON/未知字段/未知
    workspace/dtype 负例）；
19. 投影接受 quoted include，与既有 `screen_v2_source` 只允许 `<metal_...>` 冲突；已改为拒绝 quoted
    include 并保持公共源为普通 entry。

第九轮修复后 `verify.py`（manifest 18 个交付文件 + 9 个历史记录 + 32 个 evidence + 13 项合同自检 +
36 固定节点 + 10 个回归文件 70 passed）通过；当时阻断 12 仍开放。

第十轮（本轮）关闭阻断 12 后，新验收 epoch `p0-reacceptance-baseline-1` 下从头复验通过：manifest 21 个
交付文件 + epoch 基线（44 个覆盖文件）+ 9 个历史记录 + 32 个 evidence + 14 项合同自检 + 36 固定节点 +
10 个回归文件 70 passed，无 skip；独立复验已签署 done=true。P0 因此标为 completed；在 P1 获得自己的许可以前
不开始 P1。

两个小 RLAR 阶段，先合同后 resolver，不跑 GPU/模型。

**新增**

- `tcenv/portfolio/{__init__.py,domain.py,candidate.py,dispatch.py}`；
- `tcenv/portfolio/schema/{semantic-domain-plan.v1,candidate-portfolio.v1,portfolio-dispatch-record.v1}.json`；
- `tests/shared/test_shape_portfolio_{domain,candidate,dispatch}.py`；
- `tests/support/golden/shared/shape-portfolio-阶段验收/P0/{verify.py,manifest.sha256}` 与 `P0.md`；
- 验收 epoch 基线（阻断 12，第二条路）：`P0/{acceptance-baseline.json,acceptance-baseline.sha256,baseline_epoch.py}`。

**正例**

- 有限 domain normalization、guard canonicalization、variant/fallback projection；
- threshold−1/threshold/threshold+1、奇数 tail、fallback hole、refuse fallback；
- 相同候选字节不因作者/会话改变 variant_id；不同 guard/entries 改变 variant_id；
- 合取交换/扁平化后的同义 AST 得到同一 guard_digest，不同语义 guard 得到不同 digest；
- 含 host_tiling 的子包走 msl-bundle、无 host 的子包走 msl-single；母包不经过 Track 校验，
  tracks.json 与 track digest 不变；
- 负数除法/余数、零除数、公共源文件投影、geometry 原路径/摘要和映射路径冲突均有固定用例。

**固定负例**

- overlap、空 variant、未知 guard/name/bucket、缺 entry、重复/伪造 digest、非规范化 guard digest；
- `a-b<0` 被错误代数改写、负数 `rem` 宿主语义漂移、除零、公共依赖遗漏、投影路径碰撞；
- 母包被误交给 TrackRegistry、子包缺 host_tiling/workspace/taic-launch、Track digest 被 portfolio 改动；
- dtype/size 域外、请求字段 unknown、证据 unknown、多匹配、零匹配且无 fallback；
- instance-resolved geometry 被误当稳定 variant key；
- parent portfolio SHA 被复用为 variant candidate SHA；
- v2 `@variant`、非 portfolio 旧 bundle 和禁止架构名仍存在时拒绝。

**退出门**

阶段开始时先建可核验基线（完整 HEAD、工作树 manifest、合同面 SHA-256、tracks/registry/Track 摘要），
并在验收里独立重算（不是重生快照）：固定 CPU 节点全过、无 skip；有限域可穷举检查正确；相关 safe
regression 通过；P0 verifier 和独立 reviewer 通过。P0 不产生 active manifest 或性能声明。

### P1：真实 Mac 两特化 variant + 通用 fallback 垂直切片

**状态：completed（已完成）。** RLAR（actor `pi/kimi-code/k3.1-pro:xhigh`、reviewer
`pi/deepseek/deepseek-flash:xhigh`）跑满 5 轮，actor 逐轮红队并修复阻断 20–24（计时字段的 JSON 类型/`int()`
截断、bool 冒充 0/1、空 list 冒充 object、摘要重算与分层拒绝的注释失实），reviewer 独立重写攻击脚本对六份
证据逐叶子篡改并配平派生摘要后签署 `done=true`；主会话随后独立重跑完整默认 `verify.py` 退出 0（189/189
负例、P1 CPU 18 + GPU 6 无 skip、P0 70、A6 55、A4 4、live==archived，HEAD 仍 `4a93ed2`，证据与测试文件未
被改动）。剩余非阻断项已写入 P1.md §7：历史 167 组负例仍以“任何异常”记拒（新增组已断言理由）、每组 run 的
`slice` 标签只由 live 比较守护（防御纵深缺口）、原始计时只做类型/协议边界检查（P1 不发布 speedup）。详见
`tests/support/golden/shared/shape-portfolio-阶段验收/P1.md` §8 阻断 1–24。

**历史 pending 记录（已关闭，保留可审计）**：独立复验先后提出二十四项实质阻断，均已在代码/测试/验收器中修复。

1. 生命周期绑定了默认 locator 的实例合同而不是请求实例（`threshold`/`above` 的
   `lifecycle.instance_key` 相同，而 `above` 的 `lowering_profile_digest` 不同）——已改为在 dispatch
   之前派生请求实例的 `run_contract`，两档实例身份互异且等于合同指纹；
2. 验收器没有重算 bundle/native/evaluation/run 身份，把四个字段全改成 0 仍能通过——已加入独立重算
   （候选清单+几何对+生命周期、`SHA256(RFC8785(descriptor))`、结果/组合文档）与 17 个伪造负例；
3. 单源 variant/fallback 的 `resource_unavailable` 语义被收窄：源内 launch 超限时抛错而不分流——已给
   `ExactDispatchResolver` 加 opt-in `source_preflight`（默认关闭、P0 行为不变），用真实 Metal launch
   解析器在构建前预检源内几何，并补 CPU/GPU 负例；
4. `single_source_overflow.json` 的 `refused` 负例没有被验收、也没有参与 live==archived——已改为完整拒绝
   记录（请求 + 完整 dispatch record + narrow-portfolio 身份 + 零构建/执行），由验收器闭集校验并独立
   重放，`refused` 参与完全相等比较，另加 8 个伪造负例；
5. 缺真实 GPU 的语义域外拒绝验证——已加 GPU 节点 `test_p1_gpu_an_out_of_domain_request_is_refused_without_
   fallback_or_device_work`（真实 Metal 适配器、`fp32` 与未命名 size、零构建执行），证据
   `out_of_domain.json` 由验收器重放并纳入 live 比较；
6. `_observed_dispatch()` 按五元组折叠相同签名的两个 active stage，限制多阶段候选表达力——去重身份改为
   `(stage, 五元组)`，stage 索引进入证据并按 stage 逐条对齐，新增"两个相同签名 stage 都被执行"的
   CPU 节点与"多一条不存在 stage 必须失败"的负例；
7. 验收器未重算 candidate/geometry 身份（协同改多个字段并同步重算派生摘要后仍通过）——已由固定切片重新
   投影候选、重建 `geometry_pair` 并重算候选/几何身份，新增 5 组协同伪造负例；
8. 文档计数与证据清单过期（三项/5 节点/5 份证据/4 份证据/15-16 项 manifest）——已全面核对为
   18 CPU/6 GPU/6 份证据/18 项 manifest；
9. 计时/派遣协议可从归档伪造（`report.benchmark` 整块被当 volatile）——只排除测量数值，保留并显式校验
   `protocol`/`dispatch`/`measure_serial`/`rotation_sets`，新增 2 组协同伪造负例；
10. 请求合同与 schema-4 evaluation 身份未独立绑定（`evaluation_fingerprint`、`lowering_profile_digest`、
    `task_pack_digest`、`presentation_sha256` 可协同改伪）——由请求实例重新派生并核对，新增 6 组协同负例；
11. P0/A6 回归没有精确通过数门——钉死 70/55，并把 15 个锚点测试文件字节钉住，新增计数门负例。
   `baseline_epoch.py` 的独立复核也已与唯一声明漂移规则一致（退出 0，只报 `dispatch.py`）。

12. 资源负例只数下游适配器、未证明组合器没进 `evaluate_task`；验收器文件头/输出仍虚报 P0 范围；
    同源预检的"零投影"表述过强——三处已按实际行为修正（拒绝路径在组合器边界 spy 计数为 0，
    声明回退恰好一次且只收到最终 fallback；文字改为"临时投影可用于预检，拒绝时不返回最终 projection"）。
13. "多字段协同篡改 + 重算全部派生摘要"仍能绕过验收器（构建证据只比同源摘要、只看 entries"像几何"；
    Host Tiling 只查非空与最后一个 path；sidecar 只比三块并用归档字段重算归档摘要）——已改为从固定候选/
    DomainAuthority/选中投影/请求合同**独立重建**期望（`ReconstructedRun`），构建证据复用
    `verify_evaluator_build` + `metal_contract(...).assemble()`，Host Tiling 用 `HostTilingPlan` 重建并逐次
    核对 `N*4`，`projection`/`evaluation`/`record`/`run_identity`/报告字段整份相等，证据与运行组键集改为
    闭集；默认负例从 48 增到 81（38 单字段 + 43 组协同，全部先重算派生摘要再验），本轮 16 组新协同攻击
    全部被拒。
14. native 描述符、`evidence.loaded` 与 ABI/correctness/safety 的内部证据仍未独立检查（改 `backend_id`、
    换掉 `descriptor.bindings`、改 SIMD 宽度/线程上限/线程组内存、谎报已加载 artifact、`correctness.ok=true`
    却把 case 改成失败、`safety.violations`/`abi.violations` 非空仍算 ok 均可通过）——已补：native 记录与
    descriptor 逐字段独立核对（backend/kind/compiler/MSL 版本/线程上限/SIMD 宽度/静态线程组内存/
    `kernel_names`），`bindings` 由固定候选源 kernel 签名按生产 `DTYPES` 表重建后整表比较，
    `evidence.loaded` 必须为空，ABI/correctness/safety 的 `ok` 必须由 `violations`/`cases` 内部证据蕴含并
    与请求实例的容差/case 集合/seed/元素数一致，`verification` 必须是 harness 的"无检查器"原文，相关块改为
    闭集；默认负例 81 → **120**（38 单字段 + 82 组协同），本轮 37 组独立重算摘要的攻击全部被语义检查拒绝。
15. build provenance、MSL 版本、计时协议文案、阶段时间线、compile/benchmark 未知字段与编译日志主张没有独立
    语义门（`language_version` 由两个待审字段互证；`compile_profile`/`toolchain`/`builder_id` 无生产重建；
    `coverage.scope`、worst 索引、`scaled_err` 一致性只做弱检查；`protocol`/`dispatch` 只查子串、
    `rotation_sets` 只查下界；`timeline` 整块 volatile、compile/benchmark 的未知字段既不比较也不拒绝；日志可写成
    0 字节）——已补：MSL 版本钉为工具链事实并在真实 GPU 阶段用生产探测核对，provenance 由注册 builder/目标重建
    且闭集，coverage/worst/scaled_err 收敛，protocol/dispatch 与 v2 原文逐字相等、rotation_sets 由 ABI 重建，
    timeline 按模式固定序列（只排除时长），compile/benchmark 闭集，编译日志按生产模板定界；默认负例 120 →
    **141**（38 单字段 + 103 组协同），本轮 18 组独立重算摘要的攻击全部被语义检查拒绝。
16. 测量证据的派生结论未按生产公式核对（样本与 median/mean/min/cv 可矛盾、noise_cv 与 candidate.cv 脱钩、
    attempts 无上限、roofline 的 bound 可改并可塞未知字段、worst 的 got/want 可任意改而误差为 0）——已补：
    candidate 统计量由 `interleaved` 公式从样本重建并要求整块相等，`noise_cv == attempt_noise(...)`、
    `noisy` 由其推出、`1 ≤ attempts ≤ BENCH_ATTEMPTS`，roofline 由 `roofline(task, median_s, 标定资产)` 重建
    整字典相等，worst 的 `want` 由固定 case/seed 的生产 `prepare_v2_case` 参考输出核对、`scaled_err` 用生产
    `scaled_error` 重算；默认负例 141 → **152**（38 单字段 + 114 组协同），本轮 10 组独立重算摘要的攻击全部
    被语义检查拒绝。
17. 测量协议/固定切片事实仍有洞（`noisy=true` 却 `attempts=1`、repetition 数由报告自选、域外拒绝的 request
    身份采信 sidecar、编译日志摘要被误当不可复算、`host.machine` 只查前缀）——已补：`noisy ⇒ attempts ==
    BENCH_ATTEMPTS`、`EXPECTED_REPS=3` 固定 repetition 数、拒绝请求由固定实例独立重建并整份比较、
    编译日志按生产模板逐字节重建（size/sha256/duration_ms 精确）、host 块用本机 `platform` 事实逐字段相等
    （run 级与文档级共用）；默认负例 152 → **164**（38 单字段 + 124 组协同，新增 12 组含两组按生产公式完整
    配平统计量/noise/roofline/摘要的测量伪造）。
18. 浮动字段与固定候选输出未封严（`device_warnings` 任意文本、`elapsed_s=-1`、容差内改写
    `worst.got` 并配平 scaled_err）——主会话已收敛为生产 probe 重算 warning 且移出 volatile、
    elapsed 有限非负校验、固定 exact candidate 的 `got==want==reference` 与 worst 首元素规则；新增
    三组协同负例并同时检查 stable live 比较，默认负例 164 → **167**（38 单字段 + 127 组协同），
    完整默认 P1 verify 已通过，等待独立复验。
19. 计时批量 `iters` 无人校验（只要求 `>=1`，改为 `10**9` 或 `1` 并重算摘要后验收器接受）——主会话
    复审发现。生产 `interleaved` 记录每次启动的样本，所以 `iters × min_s` 就是一批墙钟，`choose_iters`
    又给出 `kmax=4000`、`BATCH_WALL_BUDGET_S=1.6 s`、慢核封顶 `kmin`（本切片 16 个轮转组）三条规则；
    已按这三条约束 `iters`，并用 `inspect.signature` + `BATCH_WALL_BUDGET_S` 交叉核对钉住的常量，
    新增两组协同负例（`10**9`、`1`），默认负例 167 → **169**（38 单字段 + 129 组协同）。P1 不发布
    speedup，这一项不阻断 P1 自身发布谓词，但属同一类协议声明缺陷，已在 P2/P3 复用归档计时前关闭。
20. 计时字段 JSON 类型仍可伪造（`attempts=1.5`、`gpu_lock_wait_s=false` 重算三个摘要后均同时通过
    语义验收和 stable live 比较）——本轮独立红队发现；attempts 现在必须是非 bool 整数，lock wait 必须是
    非 bool 有限非负数值。新增两组完整配平且断言具体拒绝理由的默认负例，169 → **171**
    （38 单字段 + 129 协同 + 4 完整配平测量攻击）；伪造负例移入 GPU 阶段，CPU 诊断不再打开 Metal 设备。
    最初对 quiet retry 的怀疑被真实 fallback 证据与生产循环否证（第二次才 quiet 合法），本轮新加的错误
    `quiet ⇒ attempts == 1` 已撤回，原有 noisy 规则未弱化。
21. 整数语义字段仍可靠 `int()` 截断或 bool 冒充（n_bad=-0.5 时 case.ok=true 仍通过，另含
    nan_count/unwritten/n_elems/seed、worst.index、document.n、rotation_sets、candidate.iters/reps）——
    第二轮 reviewer 发现并由 actor 独立复现；上述字段现全部先要求非 bool 整数再做值/范围检查，
    `_rebuilt_bench_stats` 不再先 int() 再比较。新增 10 组重算摘要且断言具体理由的默认负例，
    171 → **181**（38 单字段 + 139 协同 + 4 完整配平测量攻击）。target_s 使用未归档 probe，
    文档明确只声称 kmax/watchdog/rotation 可证不变量，不伪造最终 min_s 对 probe 的精确重建；
    worst.got 文档同步改为已由固定 reference 重建。
22. bool 冒充 0/1 数值仍可穿过两道证据门（allow=true、case/worst scaled_err=false、pipeline.stages=true；
    native static_tg_bytes=false 在归档层也接受）——第三轮 reviewer 发现；新增递归严格 JSON 比较并用于
    report/build/record/projection/evaluation/refusal/stable live，correctness 数值与 descriptor 的全部同类
    整数/数值字段也在转换前检查类型。新增 5 组具体理由负例，181 → **186**（38 单字段 + 144 协同 +
    4 完整配平测量攻击）。附带将 target_s/watchdog 钉值改为精确相等，并用生产 choose_iters AST
    交叉核对 0.1 慢核阈值，关闭常量门只查正数的失实声明。
23. 空 list 可冒充空 JSON object（baselines/baseline_us 同时穿两道门，speedups 穿过归档门），且 §7
    错称所有 volatile 字段都会重算——第四轮 reviewer 发现；三个字段现必须先为 Mapping 再要求为空，
    新增 3 组具体理由负例，186 → **189**（38 单字段 + 147 协同 + 4 完整配平测量攻击）。文档已明确
    原始计时只做类型/协议边界，三个报告派生摘要才每次重算，不把可伪造的原始测量包装成离线真值。
24. 摘要重算注释指向不存在的 `_check_identity_documents`，coordinated docstring 又绝对化为只有独立
    重建能拒绝——第五轮 reviewer 发现；现改指 `_reconstruct_run` / `_check_one_evidence` 的真实赋值
    与比较，并明确生产 canonical/闭集 reader 先拒一部分、其余由独立重建拒。只修文字，不改测试、
    证据或断言，默认负例保持 **189**，完整默认验收重跑。

P1 的验收门**不含** P0 的 14 项合同自检（`P0/verify.py: verify_contracts()`）：其第 11 项要求当前工作树
整份通过 P0 epoch 基线，而 P1 已声明的 `dispatch.py` 扩展必然违反它（P1 不改写 P0 验收器）。替代保障是
P0 的 36 个固定节点（含 13 个 dispatch 节点）、10 个回归文件的精确 70、P0 manifest（仅允许那一个声明
漂移）、P0 历史记录与 44 个共享文件的逐字节交叉验证，以及 P1 自己的 79 文件合同面检查。

当前证据：真实 Mac Apple M4 Max，18 个固定 CPU 节点 + 6 个固定 GPU 节点（无 skip）+ live==archived；
`tcenv/portfolio/dispatch.py` 是 P1 起点基线中唯一变化的文件（已在 `P1/manifest.sha256` 重新钉住，
P0 的 36 个固定节点继续通过）。P1 仍不激活 portfolio（无 active manifest、无 `portfolio_run.json`、
无 speedup 发布）。

**新增**

- `tcenv/taskcontract/portfolio_runner.py`：resolve → CPU geometry preflight → project →
  `evaluate_task`，返回 `PortfolioDispatchRecord + TaskEvaluationResult`，不创建 active manifest；
- `tests/shared/test_shape_portfolio_runner.py`；
- `tests/support/shape_portfolio_metal.py`：固定人工候选与有限测试 domain；
- `tests/backends/metal/test_shape_portfolio_gpu.py`；
- `P1/{verify.py,manifest.sha256,evidence/}` 与 `P1.md`。

**真实切片**

- 一个有限 elementwise 题，两个不同实现/几何的 variant，一个独立 generic fallback；
- 至少一个 variant 使用候选 Host Tiling 且测试 OpSpec 声明 workspace，走 `msl-bundle`；无 Host
  Tiling 的 generic fallback 走 `msl-single`；所有 MSL 源都带自己的 `taic-launch`；
- 声明两个 dispatch 边界、一个奇数 tail 和一个候选覆盖空洞。

**必须验证**

- threshold−1/threshold/threshold+1、tail、覆盖空洞、语义域外拒绝；
- 实际 provider/grid/threads/workspace 与 selection/dispatch parity；
- variant candidate/geometry/artifact 身份不同且与投影一致；
- fallback 真实执行但 `provider=fallback`，不进入 variant 性能或 correctness 记功；
- Host Tiling、read_write、workspace、guard 和计时边界保持既有语义。

**固定负例**

伪造 provider、错误 grid、parent/variant 身份串用、fallback 冒充 candidate、失效 artifact、
Host Tiling 缺失、域外请求触发 fallback；guard 命中后资源超限但 fallback 未声明
`resource_unavailable`，以及 fallback 自身再次资源不足。资源预检负例必须证明在调用
`evaluate_task` 前已经分流，且只构建最终选中的投影候选。

**退出门**

P0 已通过；必要真实 GPU 节点全部执行且无 skip；证据包含真实设备、schema-4 report、实际 dispatch
与身份；A4 几何锚点和 A6 preserved-v1 锚点不回归；独立 reviewer 通过。P1 仍不激活 portfolio。

### P2-0：证据权威图、严格模型与最小 Projector 原型

**状态：completed（CPU-only 原型已完成并经独立 reviewer 两轮 ACCEPT）。** 最终证据：39 个手写理由
负例、1238 个自动变异（1232 个结构化拒绝 + 6 个合法 volatile + 0 unchecked）、P0/P1 54 个固定
CPU 节点 + 结构 6 节点 + P2-0 扩展 1 节点全部通过且无 skip；R1–R8 与最终 F1–F6 均已关闭。
默认验收命令为 `uv run python -u tests/support/golden/shared/shape-portfolio-阶段验收/P2-0/verify.py`。
本阶段验证了 `docs/架构/证据验收方法-问题与架构建议.md` 的架构能消除整类问题，而不是先把 P1 的 5105 行
验收器重构一遍。P1 的 verifier、证据、manifest 与签署保持冻结；P2-0 不改 schema-4、六 RPC、
Track/registry digest，也不创建 active manifest 或性能声明。

实际顺序固定为：

1. **P2-0A：强类型证据边界与权威图。** 先为 P2 新证据定义严格类型、闭集字段和
   `frozen_input / derived / observed_stable / observed_volatile / external_secret` 来源分类；
   列出候选字节、OpSpec、DomainAuthority、ABI/设备限制、master seed、fresh execution 与协议 epoch
   等信任根。生产写入和验收读取使用同一严格模型；未知字段、bool/数值/容器变形必须在统一边界拒绝，
   不能靠后续逐字段 `int()/float()` 补救。现有 schema-4 report 先作为冻结输入/原始观察处理，不返工
   全部历史模型。
2. **P2-0B：最小 Deterministic Projector。** 用现有 P1 人工 fixture 的 CPU 可重建部分做新原型：
   `FrozenInputs + RawObservations -> 完整派生视图`。派生报告由唯一投影生成并整树严格比较，不再逐字段
   手写 A==B；identity/canonical 规则继续复用唯一生产函数，并配冻结 golden vectors。完整可读报告可以
   保存，但只是投影缓存，不是第二权威。Raw Observation 在离线归档中仍是自述，绝不能标成天然不可伪造。
3. **P2-0C：信任根扰动与分类变异门。** 默认负例至少覆盖候选源、OpSpec/DomainAuthority、ABI/设备
   限制、freeze/plan 绑定和协议 epoch 的扰动；每个根都声明哪些结论必须随之变化，拒绝遗漏也要区分合法
   不敏感。派生字段变异必须由重投影拒绝，volatile 字段按显式豁免处理；负例必须断言语义理由，不能只
   捕获任意异常。原始计时和未见泛化明确移交 P2 fresh confirmation，不在离线 checker 堆启发式。
4. **进入 P2 主体后再同步建立 Hypothesis 状态机测试。** 对真实 DispatchManifest 激活/撤销/恢复转换
   生成序列，补充而不是替代固定崩溃恢复负例、真实 GPU 节点与 fresh confirmation。

**原型交付与验收**：`tcenv/portfolio_evidence/projection.py`（严格模型与确定性投影基础设施）、
`tests/support/golden/shared/shape-portfolio-阶段验收/P2-0/{evidence-model.md,verify.py,mutate.py,golden.json,mutation-report.json,manifest.sha256}`，以及本阶段明确登记的共享测试扩展
`tests/shared/test_shape_portfolio_evidence.py::test_p2_0_core_api_strict_json_and_numeric_boundaries`。
P0 的固定测试明确要求 `tcenv/portfolio/` 只含 `__init__/candidate/dispatch/domain` 四文件，因此 P2
证据层独立为 `tcenv/portfolio_evidence/`：它只调用 P0/P1 的生产 resolver/projection，不形成第二 dispatch，
也不通过修改 P0 测试或给新文件改名来绕过锚点。P2 主体则使用独立 `tcenv/portfolio_validation/`，
因为 P2-0 已将自己的证据包封成两个文件；继续往其中加模块会破坏已完成的 P2-0 验收。新包只复用
P0/P1/P2-0 的生产函数，P3 performance 另起 `tcenv/portfolio_performance/`，并各自建立封闭文件清单与 forbidden-architecture 检查。
P2-0 不通过把新测试改成别的名字来绕过 P0/P1 的封闭测试集。上述新增节点已作为 **P2-0 自己的
确切扩展**写入 P2-0 verifier 与 manifest；默认验收精确运行 P0/P1 54 个固定 CPU 节点、
`tests/shared/test_structure.py` 6 节点和该扩展 1 节点，全部禁止 skip/xfail/deselect。新测试只调用
核心 API，不调用 P2-0 verify，因而不会形成 pytest 递归。不回改 P0/P1 verifier，而是继续钉住其
manifest/证据并运行固定 node id 与精确计数锚点。进入 P2 主体时的后续扩展仍须另行声明。

**退出门**：严格类型攻击在统一边界拒绝；每个声明信任根都有扰动负例；派生字段无需逐字段比较即可被
重投影拦截；原始观察与派生结论的信任边界写入 evidence-model.md；P0/P1 历史文件不变；独立 reviewer
通过后才能开始 P2 主体。原型行数不是验收目标，也不预设“必须降到 500 行”。

**上游已有做法（见 `docs/架构/上游证据防伪测试盘点.md`）**：cannbot 的 npubench 与 ops-perf 已实现并测试过
同类机制——私钥 HMAC 回执 + `build_attempt_id` 防重放、接受结果前重新哈希（TOCTOU）、
陈旧证据必须留带 reason 的失效账本、不信自报 speedup 而用原始 latency 重算。P2 证据设计应把这些
列为候选机制，与“外部钉住的 freeze/expected digest + 重投影”组合，而不是只做其中一项：
HMAC 证明回执出自该次受控构建，重投影证明证据是该输入的确定性投影。私钥必须由上层 harness 持有，
不得放在 agent 可写目录；未收到真实验回执前不得声称已具备该能力。

### P2：独立未见分片、fresh confirmation、激活与恢复

**实施子阶段（逐段验收，不一次性混合推进）**：

- **P2-A（已完成）**：`PortfolioFreeze / ValidationPlan / CandidateTuneView`，冻结候选与两阶段分片隔离；
- **P2-B（已完成）**：独立未见分片的 fresh rebuild/confirmation 与原子确认证据；
- **P2-C（已完成）**：DispatchManifest、激活/撤销代次与覆盖门；
- **P2-D（已完成）**：portfolio resume/observe sidecar、崩溃恢复与幂等；
- **P2-E（已完成）**：完整真实 Mac 验收、性能边界复核与独立签署。
- **P2-F（最终重录完成）**：native64/V5/V6 后的安静窗口 golden/evidence/manifest 重录；14/14
  archived 与 live fresh confirmations 均 confirmed、0 timing-blocked，46 个 P2 CPU + 5 个 live Mac
  及 checkpoint 生命周期通过，完整快速 CPU 4549 passed。历史 start-surface 锚点仍按原样拒绝
  当前漂移，未为拿绿改写 baseline；详见 `P2/F.md`。

**新增**

- `tcenv/portfolio_validation/validation.py`（schema id 是代码里的闭集常量，不是 JSON schema 文件）、
  `tcenv/backends/metal/portfolio_confirmation.py`（fresh 重建与激活期望），以及
  `taic/{portfolio_confirmation.py,portfolio_manifest.py,portfolio_store.py}`；
- `taic/shape_portfolio.py` 专用 checkpoint/resume 入口；
- `observe/shape_portfolio.py` 显式 sidecar reader；
- `P2/{verify.py,manifest.sha256,evidence/}` 与 `P2.md`。

**独立分片协议**

1. **两阶段隔离**：生成与调优阶段只向候选挂载完整有限命名池中的 `tune_shards`，保留的
   unseen instance_keys 不落候选工作目录；
2. `opspec.json["sizes"]` 从一开始就说明完整有限实例池（含 boundary/tail/hole 的命名 size），因此
   opspec/task-pack digest 在后续阶段无需也不允许改变；本计划不动态生成新的 shape 数值；
3. 先冻结 candidate bytes、domain plan、task-pack 和 guidance digest，形成 `portfolio_freeze_digest`，
   此后 candidate/Bundle 只读；
4. Confirmation 由独立进程使用 harness 保管的 `master_seed + opspec_digest + portfolio_freeze_digest`，
   **从已声明命名池中选择未用于调优的 instance_keys，并派生各分片 fresh 输入种子**；plan/seed 不进入
   task pack、agent_guidance 或候选可读上下文；
5. 验证覆盖 threshold±1、tail、资源边界和未用于规划的 instance 组合；所有 instance 都能通过现有
   `instantiate_v2` 装配，不允许事后改 opspec 或偷偷注入 symbol overrides；
6. resume 只能使用已保存 plan/seed；候选字节变化必须产生新的 freeze 和 validation plan；
7. 前后快照证明 task-pack/guidance 未变，不能用“隐藏”名义把分片提前交给候选。真正隐藏 shape 数值
   的 `symbol_overrides` 域扩展不在本计划范围内。

**证据与激活**

- 每个 `(variant_or_fallback, instance_key)` 都独立 fresh rebuild + fresh confirmation，沿用现有
  identity comparator；
- `usage=benchmark`：执行 `compare_confirmation_identity` 与既有计时确认门；`usage=golden`：执行
  `compare_confirmation_identity`，并要求 `status=ok`、`cache_hit=false`、`build.shape=full`，不得
  调用要求 speedup/benchmark block 的 `confirmation_publication`；
- 保存原 schema-4 reports，PortfolioConfirmation/DispatchManifest 只做 sidecar；
- 逐 freeze、plan、evaluation、confirmation、activation 原子 checkpoint；
- 缺、失败、过期、unknown、stale 或 cache_hit 证据均不得 active；
- 每种 fallback reason 都要真实构造并验证；
- **允许部分特化覆盖激活**，但每个 Semantic Domain instance 必须恰好命中一个 active variant，或
  命中一个适用且已验证的 `registered_implementation` fallback；有覆盖空洞且 `fallback.kind=refuse`
  的组合不得 active；
- sampled 永远保留 sampled 标签。

**run/resume 兼容**

- 现有 schema-3 `run_config.json` 不改；同目录新增 `portfolio_run.json` 并绑定其 digest；
- `active_portfolio_pointer.json` 只存 `dispatch_manifest_digest` 与 `activation_generation`，作为
  active 指针，不复制或改写单 variant 的 schema-4 证据；
- `taic/resume.py::inspect_run` 最前面检查 `run_dir / "portfolio_run.json"`，存在即返回
  `{"resumable": false, "kind": "portfolio_run", "reason": "use portfolio resume"}`；
  `load_snapshot` 对同一 sidecar 抛 `ResumeError`。`taic/cli.py::prepare_resume` 再把不可续跑结果
  转为 `ResumeRefused(code=3)`，不得在 resume.py 反向 import cli。两条路径均为 P2 固定 CPU 负例；
- portfolio 使用自己的 resume 入口；旧 schema 1–3 runs 的恢复行为不变；
- observe 显式读取 sidecar，记录 per-shard provider、`credits_variant`、coverage 和失效记录；
  fallback reason / failure / unknown 在分片证据与确认 sidecar 里，不在 `portfolio_run.json`。
  不做 SQLite migration。

**退出门**

真实未见 Mac shards 与 fresh rebuild/confirmation 通过；每个 checkpoint 崩溃恢复负例通过；不会重复
完成已确认 fresh batch；revoked/stale/unknown manifest 不可执行；独立 reviewer 通过。

### P3：有限域性能治理与第二个任务

**实施子阶段（状态及时同步）**：

- **P3-A（已完成）**：性能模型与两本账。`latency_ratio=candidate/baseline`、`speedup=baseline/candidate`，
  specialized 使用逐 shard ratio 中位数，effective 使用 sum/sum；usage 白名单、实例去重、缺/多 shard
  与无 variant 阻断已落地，`performance-policy.json` 已冻结 1.10，20 个 CPU 节点通过，独立 reviewer ACCEPT。
- **P3-B（性能门已过）**：第二个 tiled GEMM 任务。fresh 证据里 `fp16/paper/benchmark` ratio 1.021、
  `fp16/small/benchmark` ratio 0.817，两者都是 candidate provider 且 `publishable=true`。small 此前约 1.33
  是因为 `torch_mps` 与候选、MLX 放在同一次交错里，把 GPU 频率留在对 launch-bound 候选更不利的档。
  现行口径仍用端到端墙钟：候选与 MLX 同一次交错，Torch MPS 缓冲和它自己的批次都在这次比较之后。
  1.10 没有放宽。早期 paper `identity_error`
  已通过替换 naive 候选关闭。runtime 编码优化（缓存 `MTLSize`、复用单阶段 pipeline state、
  等价 operation iterator）已落地，small candidate median 从约 5.85us 降至约 4.56us；最新诊断
  GPU 约 3.30us、wall-GPU 差额约 2.40us，仍需同时处理剩余编码成本与 kernel。诊断见
  `P3/evidence/small-timing-diagnostic.json`，仅作诊断、不可发布。ICB 原型出现 address fault、
  直接 epilogue 实验仍约 2.10，均未采用；kernel-only 若未来新增，必须为 candidate/baseline 建立
  同一 GPU 测量根并开新 epoch，不能只删除 candidate 路径成本。关闭 Instruments 后 xctrace 已成功，
  standalone MLX 也有非空 GPU intervals，但 tracing 扰动显著且 interleaved 归因不唯一，当前只允许诊断、
  `P3/evidence/kernel-only-feasibility.json`；完整外部咨询问题与复现命令见
  `P3/small-gemm-performance-consultation.md`。固定 harness 原生批量编码器已生产化（单阶段 buffer-only
  无 fill，其余回退），Python 输出一致性与 launch hooks 有固定节点；官方 `GEMMKernel::run` small
  full-run correctness/dispatch 通过。那些约 1.33 的 ratio 是 `torch_mps` 仍挤在同一次交错里的旧测量，保留在
  `native-batch-feasibility.json`，不是当前门。P3-B 当时的门是上面的 0.817 / 1.021（P3-C 的重录数字见 `P3/C.md`，以上为历史记录）。分段 commit 实测更差，NAX wrapper 已按 M5-only 边界停止。
- **P3-C（已完成）**：两个任务（`shape_portfolio_pointwise` 7 个有限域分片含 fallback，`tasks/gemm` `small` + `paper`）的完整有限域 fresh 性能证据、两本账与逐分片设备/dispatch/四重身份已归档（`P3/evidence/{pointwise-domain-performance,gemm-domain-performance,gemm-performance,paper-performance}.json`），含真实 GPU 局部回归负例与 live vs archived 对照；详见 `P3/C.md`。P3 阶段起点历史合同面漂移按原样保留。

**新增**

- `tcenv/portfolio_performance/performance.py`；
- `tcenv/backends/metal/portfolio_performance.py`：真实 Metal 分片测量与逐分片证据投影；
- `tests/shared/test_shape_portfolio_performance.py`；
- `tests/backends/metal/test_shape_portfolio_performance_gpu.py`；
- 修改 `tcenv/backends/metal/runtime.py` 与 `tests/backends/metal/test_launch.py`：不改变计时边界的
  launch 编码优化，以及单阶段复用/fill 后重绑的固定测试。
- 新增 `tcenv/backends/metal/native_batch.{py,m}`、`tests/backends/metal/test_native_batch.py` 与
  `tests/support/golden/metal/fixtures/gemm_small_steel.metal`；修改 `backend.py` 的组合根与
  `track_adapter.py` 的执行侧 native-call 计数。原生路径只覆盖单阶段 buffer-only 无 fill，
  其余回退。现行计时口径下 small 与 paper 的 fresh ratio 都已 ≤1.10。
- `P3/{A.md,B.md,C.md,verify.py,manifest.sha256,evidence/performance-policy.json,evidence/gemm-performance.json,evidence/paper-performance.json,evidence/gemm-domain-performance.json,evidence/pointwise-domain-performance.json,evidence/small-timing-diagnostic.json,evidence/kernel-only-feasibility.json,evidence/native-batch-feasibility.json}`（共 32 个封闭交付文件）。

**范围与口径**

- 完整穷举一个有限 domain，并加入第二个固定候选任务（一个 tiled GEMM 类切片）；
- 仅 `usage=benchmark` 的 instances 进入性能比较；golden-only 不得进入；
- candidate 与已注册 baseline 使用现有 interleaved、rotation、first-quiet-attempt 和同设备口径；
- fallback timing 单独报告，永不计入生成 variant speedup。

**性能门**

P3 开始前冻结 `performance-policy.json`，不得实施中放宽。Mac portfolio 的每 shard 回归比例固定为
**1.10**；`timing_uncertain`、unknown、缺 baseline、任一 correctness/dispatch 失败均阻断发布。
`PerformanceReport` 逐 shard 给 verdict（失败与 unknown 理由），并按两本账报 latency ratio 与 fallback 的
时间/计数比例（`fallback_time_fraction`、`fallback_count_fraction`）；每分片的 candidate/baseline 中位数、
`noise_cv` 与 `worst_scaled_err` 记在 `shard_evidence_document` 的证据文档里，coverage kind 记在
confirmation/manifest 的 subject 上，性能报告里没有这一项。
性能必须分成两本账：`specialized_hit_speedup` 只描述特化命中分片；
`effective_end_to_end_speedup` 在相同语义、输入和计时边界下包含实际 fallback 执行，描述端到端有效
结果。后者不是 variant speedup，也不得用来粉饰前者。Geomean 只能作描述，不能作为唯一门，不能掩盖
局部退化。

**退出门**

两个任务的 per-shard correctness/dispatch/performance 全过；归档与 fresh 性能证据一致；P0–P2
锚点不回归；独立 reviewer 通过。

### P4：清微 NPU 离线数据与协议模拟

**新增**

- `tests/support/npu_sim/shape_portfolio.py`；
- `tests/shared/test_shape_portfolio_npu_sim.py`；
- `P4/{verify.py,manifest.sha256,evidence/}` 与 `P4.md`。

复用 P0 的 domain/identity/resolver 与 P2 的 validation/manifest 负例，接入现有 fake-NPU source/
bundle 协议和 `docs/架构/清微NPU数据模拟/` 的重放机制。不得改 proto、六 RPC、field number、HMAC、
BlobChannel 或 remote EvalReport。

**退出门**

离线模拟与现有 remote conformance 全过；证据固定 `simulated=true`、`publication_allowed=false`、
`npu_p0.status=blocked`，无真实设备 correctness、timing、speedup 或发布字段。P4 不解除 NPU-P0。

## 5. 每阶段验收纪律

每阶段 verifier 必须：

1. 校验完整 manifest 和文件 SHA-256；
2. 固定精确 node id 与通过数，必要节点出现 skip 即失败；
3. 包含稳定负例，防止删节点、摘组合、伪 digest、fallback 记功和身份串用；
4. 对真实 GPU 证据检查设备、实际 dispatch、provider、candidate/geometry/artifact/evaluation 身份；
5. 对归档与 live 证据逐字段或按明确容差比较，不能只信 actor 自报；
6. 只能通过显式 `--record-evidence` 生成新证据，再显式 `--write-manifest`；普通 verify 不改归档；
7. GPU/skipped 阶段只能诊断，不能写 completed；
8. reviewer 独立检查实现和证据，不被测试改名、skip、xfail 或弱断言欺骗。

每阶段同时运行相关聚焦回归；P2/P3 运行完整快速 CPU 与相关完整 Metal GPU。既有沙箱
ps/AF_UNIX/host 权限失败逐项分类，新失败不得冒充环境问题。

## 6. 最终完成定义

只有 P0–P3 全部通过、P4 离线边界也通过，才可宣称 Mac 侧具备初版 shape portfolio：

- 候选 planner 能产生多个可共存、身份稳定的 variants；
- 每个合法请求 exactly-one dispatch，overlap/unknown/unmatched 不猜；
- 语义域外拒绝，候选覆盖域外有独立验证的 generic fallback 或明确拒绝；
- 未见分片独立、可 replay，有限域与采样域结论不混淆；
- variant/fallback 都有独立 fresh evidence，active/resume/revoke 可恢复；
- per-shard correctness、performance、failure、unknown 和 fallback rate 可审计；
- 既有单候选、preserved v1、schema 4、fresh confirmation 和发布门未退化；
- 清微仍只有模拟证据，NPU-P0 仍 blocked。
