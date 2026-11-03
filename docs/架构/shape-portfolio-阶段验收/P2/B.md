# P2-B：独立未见分片 fresh confirmation

状态：**completed（独立 reviewer ACCEPT；10 个 CPU 节点 + 5 个真实 Mac 节点与 12 份 unseen + 2 份 tune sidecars 通过）**。

## 范围

对 P2-A ValidationPlan 中每个 `(variant_or_fallback, instance_key)`：

1. 用 search seed=7 经 `run_portfolio_task` 做一次真实构建/执行；
2. 用该 unseen 的 fresh seed 再创建新的 builder/evaluator，经有界 v2 worker 做第二次真实构建/执行；
   search 使用 3 reps，fresh confirmation 使用 15 reps。若 benchmark 测量 noisy，按预先声明的
   `BENCH_ATTEMPTS` 上限做最多 3 次独立 fresh rebuild，首次 confirmed 即停；sidecar 记录
   `confirmation_attempts`，只有达到预算上限仍可保留 `timing_uncertain`，不放宽既有噪声门；
3. 从请求实例合同、选中 projection/geometry 和当前注册 Metal builder probe 重建 expected，
   不从 search report 复制 expected；
4. 用现有 R5 `compare_confirmation_identity` 比较 search/confirmation/expected 三方身份；
5. 保存原始 schema-4 reports 与 sidecar，不改六 RPC、field number、HMAC 或既有发布谓词。

本阶段不激活 portfolio，不创建 active manifest，不发布 speedup。

## 资格判定

- `identity.kind == "match"`；
- 两次报告均 `status=ok`、`cache_hit=false`、`build.shape=full`、`build_kind=native`；
- expected 与两个报告自身身份一致且完整；
- search/confirmation seeds 不同，且 confirmation seed 等于 ValidationPlan 中该 unseen 的派生 seed；
- `usage=benchmark` 必须通过现有 `timing_is_uncertain` 门；若真实测量持续噪声，sidecar 必须保持
  `status=rejected, reason=timing_uncertain` 并阻断后续激活，不能改成 confirmed 或拿旧证据顶替；
  本 fixture 无 baseline，仍不发布 speedup；
- `usage=golden` 不调用要求 speedup 的 `confirmation_publication`；
- fallback 真实执行，但 `credits_variant=false`，不进入 variant correctness/performance 记功；
- 所有证据保留 `coverage_kind=sampled_unseen`，不把 sampled 冒充全域证明。

`verify_subject_document` 会从原始 reports/expected 重算 R5 identity、verdict 与 confirmation digest，
并做严格 JSON 整树比较。篡改后的“rejected”记录可以如实归档，但只有 `status=confirmed` 才能进入后续激活。
sidecar 自身不是外部信任根：`verify_confirmation_suite` 还使用同一 TaskLocator/Freeze/ValidationPlan
重放 dispatch/preflight/projection，核对 provider/variant/fallback/candidate、usage/mode、plan-derived seed 与
`locator.deployment_pack_digest == freeze.task_pack_digest`；还从请求合同/几何/当前 builder probe 重算
`confirmation_expected`，所以 expected 与两份 reports 协同伪造也不能靠归档自洽通过。usage 翻转、非 plan seed、
provider 记功篡改、pack 漂移和 expected/report 协同伪造均有独立 CPU 负例。

## 真实 Mac 证据

`tests/backends/metal/test_shape_portfolio_confirmation_gpu.py` 5 个真实 Mac/GPU 节点：

1. unseen golden：fresh seed、full/native build、R5 identity 与 sidecar 重投影；
2. unseen benchmark：保留 full timing gate，无 speedup 发布；
3. 全部 12 个 unseen instance_keys：24 次独立真实构建/执行，12 份不同 confirmation digest；
4. unseen fallback：正确执行但不计 variant 功劳，篡改 credit 必须拒绝；
5. `test_p2_c_tune_instances_have_independent_fresh_confirmation_for_activation`：两个 tune instances 也各自
   独立 fresh rebuild/confirmation，保留 `coverage_kind=tune_confirmation`，产出激活所需的 tune 证据。

`P2/evidence/unseen-confirmations.json` 保存全部 12 份 unseen subject sidecars，
`P2/evidence/tune-confirmations.json` 保存第 5 个节点产出的 2 份 tune subject sidecars。默认验收逐份重算语义，
并运行 10 个 CPU 负例（pack/locator 字节漂移、跨 freeze expectation、usage/mode、plan seed、provider/credit、
expected/report 协同伪造、严格 attempts 等）及全部 5 个真实 Mac/GPU 节点，比较 subject/freeze/plan、build/native/artifact、evaluation、设备与
correctness/safety 等稳定身份；计时数值、quiet/uncertain 状态和由计时影响的 confirmation digest
不要求逐字相等，但每次都必须由当前报告重新判定，非 timing 的拒绝仍是失败。live 的 timing block
不是归档漂移，但也绝不能被后续激活当成 confirmed；激活只消费当前 `status=confirmed` 的 subject 与
自己的 manifest，不把 sidecar 的 confirmation_digest 当外部信任根。

## 边界

- 真实执行证明这些固定 unseen 实例，不证明任意未见 shape；
- expected 的 builder probe 只建立当前注册工具链身份，不是候选编译缓存；
- 每次 `MetalTrackEvaluator` 拥有本进程首次真实编译，两次运行不共享 full build；
- P2-C/D 仍需激活/撤销、checkpoint/resume、失效账本与独立复核。
