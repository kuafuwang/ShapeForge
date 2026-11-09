# MT7：文档、可观测性、回归与独立终审

状态：**MT7 已完成：文档与只读可观测性完成，MT6 已按 MSL + TileLang 范围完成，MT7 最终独立终审已签署。**

## 已完成

- `README.md` 与 `README.zh-CN.md` 增加多 DSL Track 当前事实：三条路线、r15 exploration winner、不是发布。
- `backends/metal/wiki/tilelang.md` 记录 `/1`–`/3` 的能力边界；`wiki/INDEX.md` 链接到该说明。
- `docs/agent-skills.md` 增加 TileLang 当前事实入口，避免从旧 Triton-only 描述推断项目。
- `observe/track_explore.py` 提供只读 summary reader；`/api/v2/track-explore` 返回 r15 winner、ratio、构建/执行成本和边界，不修改 run、模型或证据，也不是发布权威。
- `tests/shared/test_track_explore_observe.py` 三个节点覆盖 winner、执行/成本证据和只读 API；与 `test_structure.py` 共 9 个节点通过。
- r15 独立 review 已完成（`review-r15.json`），r10 rejection 保留。

## 未完成与边界

- 原目标是“两条 MSL + 两条新 DSL”。独立范围复核已把完整 MT6 收窄为 MSL + TileLang：`scope-review.json` 为 `done=true`，`full_mt6_done=true`，`additional_dsl_not_required=true`。
- 该完成保持 `publication_allowed=false`。Graph、无界 `track-explore` 和生产搜索未纳入本次签署。
- MT7 最终独立 review 已签署：`done=true`、`reviewer_done=true`。本包只证明文档、只读观测与当前回归，不开放发布。

## 最终独立终审

Actor 发现并修正当前文档仍把已签署 MT6 范围写成 deferred，以及 verifier 硬编码
`second_dsl_scope_decision_pending=true` 的问题；没有修改 ABI、阈值、历史证据或 MT7 自签字段。

Reviewer 独立重放 `summarize_track_explore` 与归档 summary 完全一致，核对三路线均
`confirmed/eligible`、native_batch_calls=0、完整 build duration、agent_turns=1 和三个 false 边界；
r10 保持 rejected。独立运行 MT7/MT6/MT5 verifier、137 个 CPU 节点、交接 hash 和
`git diff --check` 均通过。`review.json` 签署 MT7 `done=true`；`publication_allowed=false`，
不包含第三条 DSL、Graph、无界生产搜索或 speedup publication。
