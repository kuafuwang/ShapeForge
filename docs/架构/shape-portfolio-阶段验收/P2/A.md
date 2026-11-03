# P2-A：冻结与两阶段分片隔离

状态：**completed（9 个固定 CPU 节点与独立 reviewer 阻断项全部关闭）**。

## 范围

本阶段只建立 fresh confirmation 的前置信任根：

- `PortfolioFreeze`：domain plan、候选父 entries 原始字节、task-pack/guidance digest 与 protocol epoch；
- `ValidationPlan`：tune_shards、mastered unseen instance_keys、逐 unseen fresh input seeds；
- `CandidateTuneView`：候选唯一可读投影，只有 freeze digest 与 tune_shards。

不运行 GPU、模型或搜索；不做 fresh rebuild/confirmation、DispatchManifest 激活、resume sidecar 或性能声明。

## 分片与种子协议

1. 完整有限命名池来自已冻结 domain plan，不动态生成 shape，不改 OpSpec；
2. `tune_shards` 必须是非空、排序、去重的域内 keys；`required_instance_keys` 中的
   boundary/tail/resource 点**不得出现在 tune 中，必须全部进入 unseen**，不能只满足
   `tune ∪ unseen` 覆盖；当前 fixture 同时覆盖这些点的 golden 与 benchmark 两种 usage；
3. required 点先进入 unseen，剩余 unseen 名额才按 HMAC-SHA256 排名选择；required 策略本身写入
   plan，resume 时必须同时提供外部 `expected_plan_digest`，不能只重算 seed 就声称 plan 未变；
4. master seed 由 harness 外部传入，只能是安全非负整数或至少 16 字节密钥材料；
5. unseen 选择使用 `purpose=select-unseen-instance`，fresh 输入种子使用
   `purpose=fresh-input-seed`，消息包含 `opspec_digest + portfolio_freeze_digest + instance_key`；
6. master seed 不进入 `ValidationPlan.document()`、`CandidateTuneView.document()`、错误文本或 golden；
7. 完整 plan 只在 harness 侧保存，候选视图不包含 unseen keys、fresh seeds 或 master seed；
   plan 的 tune/unseen/required 都是不可变 tuple，fresh_seeds 是 `MappingProxyType`，候选视图不共享
   可变 list/dict；
8. resume 使用已保存 plan/seed；`verify_fresh_seeds` 要求外部 expected plan digest，并重算拒绝错误
   master、plan 或 freeze 漂移。

## 身份与漂移

- `PortfolioFreeze.digest` 由唯一 `tcenv.identity.fingerprint` 计算；构造时必须持有权威
  `DomainAuthority`，自洽但非 OpSpec/target 派生的 domain plan 不能冻结；
- task-pack/guidance digest 是**外部信任前提**：调用方必须从真实 pack/guidance 字节计算后传入，
  P2-A 不声称从两个 opaque hex 字符串反证内容；
- 候选 entry 使用 canonical base64 保留原始字节，entry 顺序由 document 规范化；
- task-pack/guidance/候选字节变化必须产生新 freeze 与新 plan，旧 plan 不得继续；
- `PortfolioFreeze.from_document` 会解析 `domain_plan` 并强制其 digest 等于权威 authority.plan；
  生产 freeze 文件/文本入口 `load_portfolio_freeze` / `loads_portfolio_freeze` 与 plan 入口一样使用唯一
  严格 JSON loader，重复 key 在进入 Mapping 前拒绝；
- `ValidationPlan.from_document` 只做形状、类型和选择自洽检查；域内绑定由 `validate_for(freeze)`
  完成。生产文件/文本入口 `load_validation_plan` / `loads_validation_plan` 使用唯一严格 JSON loader，
  重复 key 在进入 Mapping 前拒绝；
- freeze 在构造时即计算并缓存 digest，候选 entry 的 `bytearray` 会复制为不可变 bytes；超过 canonical
  大小/深度预算的 bundle 在构造时以 `TaskContractError` 拒绝，不会留下无法计算身份的 freeze。

## 固定 CPU 证据

`tests/shared/test_shape_portfolio_validation.py` 9 个节点：

1. freeze 严格类型、确定性、sort_keys 往返与字节绑定；
2. required boundary/tail/resource 全部且只能在 unseen，并覆盖 golden+benchmark；
3. 候选视图不包含任何 unseen key、fresh seed 或 master seed；
4. resume 同时要求相同 master/freeze/plan digest；
5. plan/seeds 与候选视图不可变，from_document 不与调用者共享可变 mapping；
6. required∩tune、非域内实例、bool unseen_count、过多 unseen、未知字段、bool seed、非权威 authority
   与重复 JSON key 均拒绝；
7. 候选字节变化产生新 freeze/plan，旧 plan 不得复用；
8. 超过 canonical 预算的 freeze 在构造时以 typed contract error 拒绝；
9. required 先占名额后，HMAC 从剩余候选中确定性补足 unseen；不同 master 产生不同 plan。

`P2/golden.json` 冻结固定 fixture 的 freeze/plan/candidate-view 身份、unseen 选择与 fresh seeds。
`P2/acceptance-baseline.json` 与 `start-worktree.sha256` 是在任何 P2 生产文件创建前记录的真实起点：
HEAD `4a93ed2a4934fe526eedfcf4577de5dbd366e24a`、45 个合同面文件、2098 个完整工作树文件。

默认验收还运行 P0/P1 54 个固定 CPU 节点、结构门 6 节点、P2-0 扩展 1 节点与本阶段 9 节点，
全部禁止 skip/xfail/deselect。

## 边界

- 这里只证明分片隔离和确定性派生，不证明候选没见过历史任务包中的所有 shape 名；
- master seed 的保密性依赖上层 harness 的进程/权限边界，不能靠把 seed 放进 agent 可读环境变量；
- HMAC 派生用于不可预测选择与 fresh inputs，不等同于构建回执的硬件真实性证明；
- P2 主体后续仍需 fresh confirmation、激活/撤销、崩溃恢复与独立 reviewer。
