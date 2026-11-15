# P2-V：按任务检索的紧凑经验 overlay

- epoch：`p2v-compact-experience-overlay-1`
- parent HEAD：`d3388497aabaf2a687f3f1d65801c87edd9fdd9c`
- 上一份 parent HEAD：`a9c1dccce636107c2852995578b05611bf19865b`
- orchestrator：`458cd29093fd36fd87fb2f500d8bcbe935bfa2d1d6a0fb583bde4b7c868795bd`
- prompt：`e9d28ff61819a041610d763f5c1464ea860265be2d414f78c15a1f8aea1fdbcf`，`prompt_changed=false`
- pack：79 文件，`b5b74185710e5c64cf55ed20cc472f3258e0fe2b1c631bb7f850eeb8dcbb069e`
- `adopt_compact_overlay=false`；`publication_allowed=false`；NPU 仍 simulation-only
- 实弹搜索：**未授权/未执行**。这是新的未批准实验。Plan 4 未开始。

P2-U 保持 `verdict=no_gain`、`adopt_overlay=false`。紧凑叠加层只投影 `gemm-silu-reg` 的 small 行，974 字节，SHA-256 `fd72abf078d9d5a7cd2118ee5263a3c83e5695c044c0bc0cc6ca86758a0aa3d1`。对照会话 13415 字节。完整处理会话 25469 字节。紧凑处理会话 14389 字节，渲染 SHA-256 `93b65a30a3b1de5addaabda1e1249cf72295e7396d6627981baaa5511c748c81`。会话增量从 12054 字节减到 974 字节，去掉 11080 字节，比例 0.9191969470715116。

注入仍走 `prepare_session`。对照 0 份，处理恰好 1 份。experience 渲染字节和渲染 SHA 没有因 `d338849` 改变。该提交把 v2 的 `selfcheck_allow_validate` 设为 False，TASK.md 不再要求 `eval-v2 --validate`。`seedless_kernel_brief` 对 v2 使用同一个值，未付费渲染里两臂归一化 TASK.md 都是 `7ad7dfe8a3be4ed0245f07ffa8bd27abc50e52507a41bb47f2d8b7fb73d028bb`。未来 compact 配对内部仍只有 overlay 差别。和 P2-U 跨实验比较时，任务书也变了，不能把 overlay 长度说成唯一变量。

`approved_by`、`delta`、`tolerable_overrun` 和 `executed` 为空。P2-U 的批准不沿用。P2-T 和 P2-U 的 verifier 仍钉着旧 HEAD 和旧 orchestrator pin `35832a11e22bf3a5a222f541157ac85e02a5a2d17d2ad925dc575fb2511e9ef1`；在当前 HEAD 上重跑失败是历史锚点漂移，那两份 manifest 和证据不改。

合同见 [按任务检索的紧凑经验 overlay](../../../../../../docs/架构/compact-experience-overlay.md)。

## 验收

```bash
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-V/verify.py
```

11 个 CPU 节点，见 `tests/shared/test_experience_compact_overlay.py`。不跑 GPU，不调用模型。`review-r1-superseded.json` 原样保留 d338849 之前的 `done=true`，那份签名不覆盖当前 HEAD。当前 `review.json` 的 `done=true` 只在新 HEAD、新 orchestrator、紧凑正文、真实渲染、共享的修正任务书和未执行实弹都被验证器重算通过之后成立。
