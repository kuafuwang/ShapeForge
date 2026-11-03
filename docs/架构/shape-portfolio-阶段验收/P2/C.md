# P2-C：DispatchManifest、激活/撤销与覆盖门

状态：**completed（独立 reviewer ACCEPT；16 个 Manifest 节点与完整激活绑定通过）**。

## 范围

- `ManifestSubject` 只保留 instance/provider/candidate/stable confirmation identity，不复制 schema-4 reports；
- `DispatchManifest.from_confirmations` 只接受当前 `status=confirmed` 且 R5 identity 为 match 的 sidecars；
  派生前还逐份调用 `taic.portfolio_confirmation.verify_subject_document`，重算 R5/verdict/digest，伪造 status/report/
  attempts/confirmation_digest 不得通过；另由 `validate_confirmation_for_activation` 统一校验 plan-derived seed、
  freeze target 与冻结 projection，变体串用、非 fresh seed、异 target 均拒绝；
- 激活要求 **每一个 Semantic Domain instance** 都恰好有一个 subject；本切片为 14/14，包含 unseen 与 tune；
  同时校验 ValidationPlan digest、plan 选择集、coverage_kind，以及 variant/fallback 是否属于冻结 portfolio；
- `active_portfolio_pointer.json` 只保存 `dispatch_manifest_digest` 与 `activation_generation`；
- `validated → active → revoked` 代次单调递增，revoked/stale/unknown 不可执行，revocation 必须带 reason。
  `ActivePortfolioPointer.read` 另接受 `minimum_generation`，可阻止旧 manifest/pointer 回滚；**持久化代次地板
  与失效账本仍是 P2-D 的硬门**，本阶段只证明单份 manifest/pointer 自洽与显式 floor 拒绝。

稳定 confirmation digest 由 subject 语义、expected、R5 identity 以及两份报告的 build/evaluation identity
组成，不含计时、noise、attempts 或浮动 verdict。原始 `confirmation_digest` 仍是完整 sidecar 指纹，
但不直接作为 active manifest 的稳定身份。

## 拒绝边界

- 缺实例、重复实例、非 confirmed subject、identity 非 match；
- provider/variant/fallback 身份不一致，fallback 记 variant 功劳；
- 非当前 freeze、错误 manifest digest、缺 revocation reason；
- 从文档反序列化但没有 confirmed sidecars 的 manifest 不得 `activate()`；`verify_manifest_document` 只做
  canonical/identity 一致性检查，不能代替 sidecar 存在性与 fresh replay；
- active pointer 与 manifest digest/generation 不一致，或 manifest 已 revoked；
- 不用 active pointer 的文本代替 manifest，也不发布 speedup。

## 证据

`tests/backends/metal/test_shape_portfolio_manifest.py` 16 个 CPU 节点：

1. 14/14 全量确认可激活并逐字重放；
2. 缺/重复/unconfirmed subject 拒绝；
3. revoked 是新代次且不能重新激活；
4. active pointer 原子写读与 stale/revoked 拒绝；
5. manifest digest/status 伪造拒绝；
6. fallback credit 与 subject 身份闭集；
7. sidecar verdict/digest 重算，伪造 attempts/report 拒绝；
8. variant/fallback 与 plan 选择、coverage_kind 绑定；
9. validation_plan_digest 不匹配拒绝；
10. generation floor 阻止旧 active pointer 回滚；
11. 文档-only manifest 不得激活；
12. variant 串用时冻结投影 candidate bytes 不匹配，拒绝；
13. 非 plan-derived confirmation seed 拒绝；
14. 非 freeze target 拒绝；
15. 缺/多独立 expectation 拒绝；
16. document-only active manifest 不得获得 active pointer。

原始计时块仍是自报观察：归档读取器不能证明它真实。quiet/uncertain 的差异可在 live 中出现，但不构成
Manifest 的稳定身份，也不能用来伪造 fresh speedup；P3 的性能确认必须继续使用 fresh measurement，
而不是相信归档 benchmark 文本。

`P2/evidence/dispatch-manifest.json` 与 `active_portfolio_pointer.json` 由全部真实 sidecars 派生。
默认验收先重算完整 fresh suite，再激活并比较归档 manifest/pointer，不由 sidecar 自报身份直接发布。

## 尚未覆盖

- P2-D 的 checkpoint/resume、崩溃恢复、observe sidecar 与失效账本；
- P3 的性能门、第二任务与 fallback 两本账；
- 生产 master seed/HMAC 回执持有方式（当前仍为固定测试 master）。
