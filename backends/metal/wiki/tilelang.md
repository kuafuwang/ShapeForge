# TileLang Track：当前已验证边界

TileLang 是独立 DSL Track，不是 MSL 的包装器，也不是其他 DSL 的必经中间层。候选拥有 Python DSL 源、work-list 和 ExecutionGeometry（候选包只接受 `kernel.py` 与 `host/execution_geometry.json` 两个路径，stray 一律拒绝）；frontend 版本/入口/options 由 Track 冻结、不在候选手里；harness 继续拥有 bounded worker、ABI、安全、dispatch、计时与证据。

## 版本与当前事实

| 版本 | 状态 |
| --- | --- |
| `/1` | `interface_only` source-only replay，不可选择，不进入生产 Metal build |
| `/2` | 固定人工真实 Metal 机制、人工/LLM correctness-only 生成与 replay 已完成 |
| `/3` | 固定 ReLU 登记、fresh identity、activation/resume 和 `affine_ramp` 批量探索通过 |

`/3` 的主要变化不是“更强的 DSL”：

- 稳定 `tilelang_fresh_subject` 进入 BuildProfile，attempt-specific 证据留在 BuildEvidence.log；
- `metal-build.2` descriptor 只含稳定 candidate/frontend/artifact identity，避免 native_digest 随 attempt 漂移；
- 真实构建继续使用 `/2` 的 fenced source-only 子构建，不能把它误称为 `/2` fresh confirmation；
- `/3` 使用 native concurrent + `barrier_interval=2` 控制调度抖动，`/2` 和 MSL 为 0；
- `/1`、`/2` 的合同与历史证据不因此被改写。

## 已执行的验证

- 固定 ReLU 候选：fresh identity、golden correctness、activation 和非 dry-run resume；
- `affine_ramp` 三 Track 探索：`msl-single`、`msl-bundle`、TileLang `/3` 均 `confirmed/eligible`，TileLang 以最低 ratio 0.074707 成为 exploration winner；
- r15 observation 与 summary 同期一致：`observations.json` 逐条记录 `encoder_strategy=pyobjc_fallback` 与 `native_batch_calls=0`，`observed-execution.json` 保留 binding/dispatch 细节；
- 使用完整 `build.evidence.duration_ms` 统计真实 lowering 成本；
- 所有路线均满足 `latency_ratio <= 1.10`，但不等同于发布。

## 不得外推

- 不把 r15 当成同一 fresh 输入、跨 Track runtime portfolio 或 speedup publication；
- MT6 已按 MSL + TileLang 完成（`full_mt6_done=true`，`additional_dsl_not_required=true`），`publication_allowed=false`；
- 不把该完成外推为 Graph、无界 `track-explore`、生产 LLM 搜索或 speedup publication；
- 不把 `/3` 的固定候选结果外推为所有 TileLang 程序、所有形状或其他硬件的结论；
- 不修改历史 MT0/MT1b/MT2/MT4/P2 证据来消除 registry/prompt epoch drift。

权威证据：`tests/support/golden/shared/多DSL-Track-阶段验收/MT6/`，尤其是 `evidence/track-explore-r15/`、`review-r15.json` 和 `review-r10-rejected.json`。observe 的 `/api/v2/track-explore` 只提供只读展示，不是发布权威。
