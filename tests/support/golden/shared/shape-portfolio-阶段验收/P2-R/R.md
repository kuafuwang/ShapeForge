# P2-R：修复 msl-bundle + Host Program + scratch 的 worker 几何身份

- epoch：`p2-r-bundle-scratch-geometry-1`
- parent HEAD：`a9c1dccce636107c2852995578b05611bf19865b`
- pack：79 文件，`b5b74185710e5c64cf55ed20cc472f3258e0fe2b1c631bb7f850eeb8dcbb069e`
- 上一纪元摘要：`5512395852ea7608a6075e2b6b45dca9bd85f112210af336cf6f8f77f5fd811f`
- `publication_allowed=false`；NPU 仍 simulation-only

本阶段只修 worker 重建 `ResolvedGeometry` 时丢掉非空 `scratch` 的问题。不改 P2-I 到 P2-Q 的 manifest 或证据，不 `git commit` / `reset` / `clean`。`handoff_states` 仍在 OpSpec workspace。

## 结论

控制机冻结的 lifecycle 文档在 scratch 非空时带 `scratch`。修复前 `_lifecycle_from_document` 不读这个键，`runner.py` 的文档比较因此拒绝一份本来合法的 bundle。比较没有删。

空 scratch 和没有该字段的旧文档仍然不写出这个键。`ResolvedScratch.from_document` 保留冻结的 `numel`/`nbytes`，不重算。闭包文件数仍是 79；变的是闭包内的 `tcenv/scratch/spec.py`。`tcenv/v2_worker.py` 不在这份闭包里，但 P2-I 的 manifest 钉着它的旧字节 `39ce5932…`。那个 manifest 不改写；当前文件是 `a083a362…`。P2-J 到 P2-Q 的 manifest 不含这个文件，仍然逐字节相符。

同一候选字节下，空 scratch 与非空 scratch 的 `TaskEvaluationResult.identity` 不同（`evaluation_changed=true`）。`candidate_sha256` 和 `task_pack_digest` 不变。变的是 geometry、bundle、artifact、native，以及报告里绑住这些构建身份的字段。两端摘要在 `evidence/identity-delta.json`。这份 CPU 报告的状态是 `rejected`，不是设备结论。

gpu 文件里 5 个节点都标了 gpu。行宽和单块 `read_write` 预检是 backend/geometry。后三个才是真实 Metal 求值：两块 write-texture 乒乓跑完 `kda small` 的三个 case；Host Program 返回 1 判 `incorrect`；P2-Q 的 device 指针乒乓停在反射 `readwrite`。

合同见 [bundle scratch 的 worker 几何身份](../../../../../../docs/架构/bundle-scratch-worker-geometry.md)。

## 验收

```bash
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-R/verify.py
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-R/verify.py --static
```

11 个 CPU 节点 + 2 个 backend/geometry 节点 + 3 个真实 Metal 求值节点。
