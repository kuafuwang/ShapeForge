# P2-T：经验 overlay 的 treatment-only 注入

- epoch：`p2-t-experience-overlay-injection-1`
- parent HEAD：`a9c1dccce636107c2852995578b05611bf19865b`
- prompt：`e9d28ff61819a041610d763f5c1464ea860265be2d414f78c15a1f8aea1fdbcf`，`prompt_changed=false`
- pack：79 文件，`b5b74185710e5c64cf55ed20cc472f3258e0fe2b1c631bb7f850eeb8dcbb069e`
- `publication_allowed=false`；NPU 仍 simulation-only
- 实弹搜索：**未授权/未执行**。Plan 3 没有完成。Plan 4 没有开始。

处理臂的真实会话由 `prepare_session` 注入 P2-S 冻结叠加层。对照臂和默认搜索只看到原来的过滤经验。全局 `targets/apple-m4-max/experience.md` 和 `taic/orchestrator.py` 保持原字节。`taic/orchestrator.py` 的 SHA-256 仍是 P2-I manifest 里的 `35832a11e22bf3a5a222f541157ac85e02a5a2d17d2ad925dc575fb2511e9ef1`。

绑定写在这一轮的 `experience_overlay.json`。已有 `run_config.json` 时，同一份不含正文的记录进入顶层 `experience_overlay`。会话 sidecar 再记路径、字节摘要、渲染后摘要、臂和 run。

experience-overlay 的执行把计划里的种子 7 作为两臂共同的 `run_seed` 传给 `run_one`。v2 搜索期评测请求仍是编排器里的字面量 7。`eval_seed(run_seed, ...)` 派生出来的值不是这颗字面量，汇总器不拿它充当搜索种子。四开关执行不传 `seed`。`run_config.json` 与 run 文件同时存在时，缺 `experience_overlay` 键会在写会话文件前拒绝。执行前核对同一 pair 的 task/dtype/size/rep、AB/BA，以及 `committees` 与 `taic.toml` 一致。

`--render-only` 写出四个 cell 的会话文件。它不产生候选，不调用模型，不是搜索结果。`--execute` 仍要四道人工门禁。本阶段批准人、δ 和可容忍超支是空的。

P2-S 的 manifest 和证据保持原字节。现场相对那份 manifest 有漂移的只有 `experiments/search_yield_ab.py` 和 `tests/shared/test_experience_search_prep.py`。

合同见 [经验叠加层的 treatment 注入](../../../../../../docs/架构/experience-overlay-injection.md)。

## 验收

```bash
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-T/verify.py
```

12 个注入测试、13 个 P2-S 准备测试、6 个四开关测试、5 个既有 experience staging 测试，再加 1 个既有指导快照测试，共 37 个 CPU 节点。不跑 GPU，不调用模型。第一次 `done=true` 已作废，见 `review-r1-rejected.json`。独立复核 `done=true`，见 `review.json`；`review-r1-rejected.json` 仍作废；批准人、δ 和可容忍超支仍为空；实弹搜索仍是未授权/未执行；Plan 3 和 Plan 4 没有完成。
