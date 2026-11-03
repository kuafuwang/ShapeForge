# P2-0 证据权威图与最小原型边界

状态：**completed（R1–R8 与最终 F1–F6 均经独立 reviewer 复核接受）**。本文件解释 `tcenv/portfolio_evidence/projection.py` 的原型合同；它不是 P2 的
ValidationPlan/PortfolioConfirmation/DispatchManifest，也不替代 P1 的历史证据。

## 1. 原型要回答的问题

P1 的验收器需要逐字段检查大量派生视图。P2-0 验证另一条路：

> 能否把完整证据视为 `Projector(FrozenInputs, RawObservation)` 的缓存，让验收器严格解析最小输入、
> 重放唯一生产 dispatch/projection，再做整树严格比较？

原型只使用 CPU：P1 人工 fixture 的 `below`（单源 variant B）和 `hole`（单源 generic fallback）。
不运行 GPU、模型、搜索、fresh confirmation 或激活。`msl-bundle` 接口允许调用方传入同一冻结包的
权威 TaskContract，但 P2-0 不复制 P1 的 Host Tiling/真实构建验收。

## 2. 数据流

```text
FrozenPortfolioInputs                         RawPortfolioObservation
- OpSpec 原文                                 - provider             [stable]
- opspec_tensor_order（声明顺序权威）          - track_id             [stable]
- SemanticDomainPlan 原文                     - candidate_sha256     [stable]
- 全部候选 entry 原始字节                     - instance_key         [stable]
- provider evidence 状态                      - elapsed_s            [volatile]
- target / ABI 参数
- DispatchRequest
- protocol_epoch
          │                                           │
          └───────────────────┬───────────────────────┘
                              ▼
                 唯一生产函数确定性重放
                 ExactDispatchResolver.resolve
                 project_provider (需要时)
                              │
                              ▼
                 Derived Dispatch View
                 - dispatch record
                 - candidate projection / null
                 - credits_variant
                 - protocol_epoch
                              │
                              ▼
            view_digest / freeze_digest / root_digest
```

完整 JSON 同时保留冻结输入、观察和派生视图，便于审计；验收时它不是权威来源，而是必须等于
重投影结果的缓存。

## 3. 字段来源与证明方式

| 字段/子树 | 来源 | 验收方式 |
|---|---|---|
| `freeze.opspec` | `frozen_input` | v2 OpSpec 解析、digest、DomainAuthority 重算；object key 顺序不承担语义 |
| `freeze.opspec_tensor_order` | `frozen_input` | 显式 tensor 声明顺序权威；名字闭集、无重复，Projector 按它重建 ABI 顺序，绝不排序 tensors |
| `freeze.domain_plan` | `frozen_input` | 从 OpSpec/target/命名实例重算，不能自报 ABI/语义摘要 |
| `freeze.candidate_entries` | `frozen_input` | canonical base64 解码、原始字节进入 parse/projection |
| `freeze.evidence` | `frozen_input` | provider 闭集与 `validated/unknown` 状态词 |
| `freeze.target_id/max_buffer_length` | `frozen_input` | 生产 target/ABI 派生 |
| `freeze.request` | `frozen_input` | DispatchRequest 闭集解析与生产 resolver 重放 |
| `freeze.protocol_epoch` | `frozen_input` | 非 bool 正整数；进入 freeze 和 view 身份 |
| `observation.provider/track_id/candidate_sha256` | `observed_stable` | 必须等于独立重放投影；live 可比较 |
| `observation.instance_key` | `observed_stable` | 必须等于冻结请求的 `dtype/size/usage` key；below/tail 的稳定观察不再相同 |
| `observation.elapsed_s` | `observed_volatile` | 只检查 JSON 闭集内有限、非负、安全范围数值；整数不先转 float；排除 stable core |
| `view` | `derived` | 由生产 resolver/projection 重建，整树比较 |
| `freeze_digest/view_digest/root_digest` | `derived` | 唯一 `tcenv.identity.fingerprint` 重算 |
| P2 master seed / unseen keys | `external_secret` | **本原型不序列化**；由未来独立 harness 进程持有 |

`EVIDENCE_FIELD_ORIGINS` 是机器检查的闭集：新增顶层或 observation 字段而没有来源分类时，
验收直接失败。除了键集，生产验收还把 `ORIGIN_INVENTORY_VERSION` 与每个精确字段分类钉到固定
`EVIDENCE_FIELD_ORIGIN_INVENTORY_SHA256`；仅把 `view` 改标 volatile 或把 `freeze` 改标
external_secret 会破坏这个语义 pin 并被拒绝。`view` 与 `freeze` 的来源标签覆盖整个子树。
整份证据文档的 JSON 嵌套深度上限为 **32**，超过限制统一以 `strict_parse` 拒绝；正常原型文档
深度不足 10。

## 4. 外部信任根：expected freeze digest

`verify_document(document, expected_freeze_digest=...)` 的 expected 值必须来自调用方受保护的
freeze 记录，不能从归档读取。否则攻击者可以换掉整份合法的 OpSpec/候选/请求，再生成一份完全
自洽的新证据。

公开 SHA-256 只提供完整性和寻址，不提供“这次运行真实发生”的证明。Root digest 也不签名；
如果攻击者能改全部输入和摘要，它仍能构造另一份文档，所以外部 expected digest 必须独立。

生产验收顺序固定为 `strict_parse → freeze_self_consistency → external_pin →
semantic_reprojection → derived_view`。所有预期拒绝都是 `EvidenceRejection`，显式携带
`stage/rejected_by`，不靠错误消息猜归因。先对归档 freeze 的 canonical 值核对自身摘要和外部
pin，再做语义重放，因此不同 `opspec_tensor_order` 会由信任根本身拒绝，而不会被下游 ABI
错误掩盖。

文件与文本生产入口分别是 `verify_json(path, ...)` 与 `loads_document(text, ...)`，唯一调用
仓库的 `load_strict_json/loads_strict_json`，重复 key 在生成 Mapping 之前拒绝。`verify_document`
保留为进程内 Mapping API，但**无法追回之前宽松 `json.loads` 已丢弃的重复键**；生产文件读取
不得先 `json.loads` 再调用它。入口递归只接受 JSON 类型闭集：object key 必须是字符串，array
必须是 list，拒绝 tuple、非有限 float 与超出 `MAX_SAFE_INTEGER` 的整数。

## 5. Raw Observation 的信任边界

原型中的 observation 是 runner 自述的最小稳定观察 + 一个 volatile 时长：

- 离线重投影能证明 observation 与冻结候选的 dispatch/projection 一致；
- 它不能证明 observation 真由硬件产生；
- `elapsed_s` 只证明数值形状，不能证明计时真实；
- `instance_key` 把稳定观察绑定到冻结请求的 shape 用途，避免未来 fresh confirmation 只凭
  provider/candidate 混淆 below/tail；它仍是 runner 自述，不是硬件真实性证明；
- P2 的 fresh execution/confirmation 是另一个信任根，不能由本原型取代。

这也是为什么不把 RawObservation 称为“不可伪造 telemetry”。最小化减少的是自述与冗余，
不是取消真实执行的需求。

## 6. 生产函数复用与 golden

- dispatch 使用唯一 `ExactDispatchResolver`；
- projection 使用唯一 `project_provider` 与 DomainAuthority；
- canonical/digest 使用唯一 `tcenv.identity.fingerprint`；
- `golden.json` 冻结 below/hole 的 freeze/view/root/stable-core 摘要，生产规则漂移会显式失败；
- golden 不是性能基线，也不证明原始观察真实。

## 7. 默认攻击类别

`verify.py` 默认执行并断言拒绝理由：

1. 严格类型：bool/integral-float 冒充 epoch，空 list 冒充 evidence object，负/布尔/`10**400`/
   无限 elapsed，tuple dims、非字符串 key、非 canonical base64、未知字段；严格比较另行拒绝
   tuple、非字符串 key 与 `-0.0/+0.0` 混淆；
2. 序列化：反转 `opspec.tensors` object key 顺序不改变 pin/裁决，`sort_keys=True` 文件和文本
   往返一致；反转显式 `opspec_tensor_order` 改变 pin 并被旧信任根拒绝；重复 formula key 的
   文件/文本在严格入口拒绝，而同一文本经宽松 `json.loads` 的正控制证明攻击构造有效；
3. 来源语义：把 view 错标 volatile 或 freeze 错标 external_secret，即使字段键集完整也被固定
   来源清单摘要拒绝；
4. 派生视图：改 record/provider/credits_variant/未知 view 字段并重算 view/root digest；单独
   伪造 freeze/root digest 分别归入 freeze 自洽和 derived view；
5. 稳定观察：伪造 candidate/provider/instance_key 不能由重放投影接受；below/tail 的稳定观察
   必须具有不同 instance_key；
6. volatile：elapsed 的合法范围内改变可重投影，stable core 不变，但不会被误报为身份变化；
7. 信任根：候选源、OpSpec、DomainAuthority/ABI、evidence 状态、请求、protocol epoch 变化，
   旧的外部 expected freeze digest 必须拒绝；对合法替代根，还要求语义 view 确实改变或权威重建拒绝；
8. golden：两个固定实例的身份摘要逐值相等；
9. 历史边界：P0/P1 manifest 除 living plan 与 P1 已声明的 dispatch.py 外不得漂移，P2-0 不回改
   已完成 verifier/证据；固定运行 P0/P1 54 个 CPU 节点、`test_structure.py` 6 节点以及本阶段
   明确登记的 `test_shape_portfolio_evidence.py` 1 节点，全部无 skip 且精确计数。

### 独立红队后的序列化决定（人类冻结，不交给实现者发明）

独立红队复现了 `opspec.tensors` 的顺序盲区：对象 key 顺序会被 canonical JSON 丢弃，但 v2 的
tensor 声明顺序决定 ABI 绑定和 semantic instance 身份。P2-0 因此采用以下合同：

1. freeze 必须显式携带 `opspec_tensor_order` 列表；该列表是 tensor 声明顺序的权威；
2. `opspec.tensors` 的 JSON object key 顺序**不再是语义**，Projector 按列表重建声明顺序；
3. 不得通过简单排序 tensors 修复，因为排序会改变 ABI；
4. 同一份 freeze 必须允许 `json.dumps(sort_keys=True)` 文件往返，pin、重投影和裁决保持一致；
5. 文件入口必须使用仓库唯一 `load_strict_json`，重复 key 拒绝；进程内 Mapping 入口只接受
   JSON 类型闭集，tuple/非字符串 key/非有限数/越界整数不得进入；
6. 这些变化仍属于 P2-0 原型迭代，不重写 P0/P1 的历史验收资产。

## 8. 自动变异门（不是手写负例的替代品清单）

`mutate.py` 对 **below/candidate、hole/fallback、projection=null refusal** 三份完整证据递归
生成攻击，而不是由人逐字段挑目标：

- 合计 761 个标量叶子：字符串改写、None→0、bool 翻转、整数 +1、数值类型变形，以及针对
  R3 的 `10**400` 和正无穷量级攻击；
- 合计 279 个容器节点：每个 object 注入未声明字段，每个 array 逆序或追加元素；数组和标量
  no-op 都使用 `strict_json_equal`，不再使用会合并 bool/int 的 Python `==`；
- 每份文档的 3 个摘要叶子不参与自动配平（否则会被配平步骤修回），freeze/root/view 三个摘要
  另由手写负例覆盖；
- 可 canonical 化的变异先重算 freeze/view/root 公开摘要；无 canonical 表示的极值仍必须进入
  生产 verifier，由严格入口拒绝，配平异常本身不算覆盖。

当前 `mutation-report.json`：below 429、hole 429、refusal 380，合计 **1238** 个有效变异，
**1232** 个结构化拒绝；每份文档只有 `observation.elapsed_s` 的两种合法范围内变化被接受，合计
6 个 volatile positive，**unexpected_accepted = 0**。按来源为 derived 242、frozen_input 966、
observed_stable 15、observed_volatile 15；按阶段为 strict_parse 79、freeze_self_consistency 0、
external_pin 906、semantic_reprojection 21、derived_view 226。自动门总是先配平，所以 freeze
自洽层由手写摘要伪造覆盖，报告保留其 0 计数而不虚构贡献。`by_rejected_by` 另行钉住
`expected_freeze_digest=906`、`json_type_closure_and_schema=79`、
`frozen_inputs_observation_projector=21`、`strict_json_equal=226`。报告同时保留三份文档 digest、
逐文档计数、逐文档 `by_stage/by_rejected_by` 与全部层级计数，能发现改变三份 fixture 的 Projector
退化；分支级放松由手写信任根与纵深防御负例补充，不把 0 unchecked 解释成所有实现分支都被证明。

只有带合法 `stage/rejected_by` 的 `EvidenceRejection` 计入覆盖；AssertionError、TypeError、
KeyError、OverflowError 等内部错误会直接让门失败。默认验收还用一个故意“全部接受”的 verifier
做小子集负控制，证明门会报告未检查叶子，而不是只会打印通过总数。

局限：自动变异证明“这些变异会影响 verdict”，不证明每条检查的期望值来自正确的信任根，
也不证明 Raw Observation 真实。后者仍靠信任根扰动与未来的 fresh execution。

## 8.1 独立红队后第一轮真实修复记录

本轮只处理 R1–R8，不进入 P2 主体：显式 tensor 顺序权威与 sort_keys 往返、严格文件/文本入口、
JSON 类型闭集和安全整数、严格 signed-zero/container 比较、来源分类固定摘要、结构化拒绝阶段、
三文档自动变异、共享 pytest 精确扩展、observation.instance_key 均已落地。真实 CPU 证据为：
34 个手写理由负例通过；自动变异 1238/1238 无未检查接受（1232 拒绝 + 6 合法 volatile）；
P0/P1 54 节点、结构 6 节点、新增 1 节点全部通过且无 skip。golden 与 mutation report 均由显式
录参重生；随后独立 reviewer 对 R1–R8 给出 ACCEPT。

## 8.2 最终独立复核与 F1–F6 收尾

独立 reviewer 对 R1–R8 给出 **ACCEPT**，随后对最终加固再次给出 **ACCEPT**。主会话依据其建议补齐：
JSON 深度上限 32、view_digest 负例、self-pin 下重复 tensor 名称与额外 evidence provider 两个纵深
负例，以及逐文档 stage 保留。mutation suite schema 随 per-document 载荷扩展升到 `.3`。

最终默认验收：39 个手写理由负例（34 + 2 深度 + 1 view_digest + 2 纵深防御）、1238 个自动变异
（1232 拒绝 + 6 合法 volatile + 0 unchecked）、P0/P1 54 + 结构 6 + 新 P2-0 节点 1，全部通过且
无 skip。独立 reviewer 逐项复现 F1–F5 并确认关闭，F6 的计数和表述已由本文同步。

流程说明：一次 RLAR 因 reviewer 最终返回内部分析而非规定 JSON 技术性退出，不是测试失败；
P2-0 的完成依据是随后独立 reviewer 的 ACCEPT 与主会话重跑，而非该 RLAR 的退出状态。

## 9. 明确不在原型范围内

- fresh/未见分片、master seed 保管与派生；
- schema-4 GPU report、Host Tiling 真实执行、构建/计时身份；
- DispatchManifest 激活/撤销、active pointer、resume/observe；
- speedup、性能发布或 fallback 计时归账；
- 用第二个独立实现替代生产 dispatch/projection。

这些边界必须在 P2 主体另行验收，不能从 P2-0 的通过推断。
