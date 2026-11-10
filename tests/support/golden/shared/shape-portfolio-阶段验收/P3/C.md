# P3-C：两个任务的完整有限域性能验收与终审

状态：**completed（两个任务的完整有限域 per-shard correctness/dispatch/performance 全过，两本账与完整身份归档并支持 live 对照；P3 阶段起点历史合同面漂移按原样保留）**。

## 1. 范围与完成摘要

P3-C 在已完成的 P3-A（性能模型与两本账，`performance-policy.json` 冻结 `1.10`）和已过门的 P3-B（第二个 tiled GEMM 任务）之上，补齐两个任务的完整有限域端到端性能证据、逐分片设备/dispatch/四重身份绑定、真实 GPU 局部回归负例，以及 `P3/verify.py` 的完整复核闭环：

1. **任务一（`shape_portfolio_pointwise` 有限域 7 个 benchmark 分片 + fallback）**：
   - 覆盖 `tail (N=7)`、`below (N=63)`、`threshold (N=64)`、`above (N=65)`、`hole (N=100)`、`boundary (N=128)`、`large (N=1024)` 全部 7 个 `usage=benchmark` 实例；
   - 包含 `variant-a-bundle`（`msl-bundle`，Host Tiling 写 `scratch` + 1D kernel）、`variant-b-single`（`msl-single`）与独立通用 `fallback`（`hole` 分片，`outside_candidate_coverage`）；
   - 归档证据：`P3/evidence/pointwise-domain-performance.json`；
   - 7/7 分片 `correctness_ok=true`、`dispatch_ok=true`、`timing_uncertain=false`、`latency_ratio` 为 `0.4969`–`0.5774`（全部 $\le 1.10$）；
   - `hole` 分片 `provider="fallback"`、`credits_variant=false`；它不进入 `specialized_hit_speedup`，只进入 `effective_end_to_end_speedup`（`fallback_count_fraction = 1/7 ≈ 0.1429`，`fallback_time_fraction ≈ 0.1432`）；
   - 两本账：`specialized_hit_latency_ratio = 0.5722`（`speedup = 1.7476×`），`effective_end_to_end_latency_ratio = 0.5594`（`speedup = 1.7875×`），`publishable = true`。

2. **任务二（`tasks/gemm` 有限域 2 个 benchmark 分片：`small` 与 `paper`）**：
   - 覆盖 `fp16/small/benchmark`（`M=256, N=256, K=128`）与 `fp16/paper/benchmark`（`M=N=K=2048`）；
   - 归档证据：`P3/evidence/gemm-performance.json`、`P3/evidence/paper-performance.json` 与完整域汇总 `P3/evidence/gemm-domain-performance.json`；
   - `fp16/small/benchmark`：candidate `1.876 µs`、MLX `2.244 µs`、`latency_ratio = 0.8360`（$\le 1.10$），`noise_cv = 0.0128`；
   - `fp16/paper/benchmark`：candidate `1160.095 µs`、MLX `1155.358 µs`、`latency_ratio = 1.0041`（$\le 1.10$），`noise_cv = 0.0067`；
   - 完整域两本账：`specialized_hit_latency_ratio = 0.9200`（`speedup = 1.0869×`），`effective_end_to_end_latency_ratio = 1.0038`（`speedup = 0.9962×`），`fallback_count_fraction = 0.0`，`publishable = true`。

3. **真实 GPU 回归阻断负例**：
   - `test_p3_c_real_gpu_local_regression_blocks_publication_despite_fast_peer`：在真实 M4 Max 上运行正确但未优化的 `small` 候选，实测 `latency_ratio > 1.10`；`PerformanceReport` 拒绝发布（`publishable=false`），同时给出 `fp16/small/benchmark:specialized_ratio>1.10` 与 `fp16/small/benchmark:effective_ratio>1.10`，两本账 speedup 均为 `None`。

4. **逐分片身份与独立算术复核**：
   - `tcenv/backends/metal/portfolio_performance.py` 新增 `shard_evidence_document` 与 `measure_shard_with_evidence`，每个分片记录真实 `device_name`、`gpu_family`、`dispatch`、`expected_dispatch`、`variant_id`/`fallback_id`、`credits_variant`、`candidate_sha256`、`geometry_identity`、`bundle_digest`、`artifact_set_digest`、`native_digest`、`evaluation_fingerprint`、`task_pack_digest`、`best_baseline`、`baselines_measured`、`noise_cv` 与 `worst_scaled_err`；
   - `P3/verify.py` 从每个分片的原始 `ShardMeasurement` 独立重算 `PerformanceReport.evaluate(...)`，逐字段核对两本账、fallback 比例与 64 位十六进制摘要，并在 `--live-compare` 下核对 live 与 archived 的全部稳定身份一致。

## 2. 闭合文件清单与测试节点

`P3/verify.py` 的 `DELIVERABLES` 现包含 **32 个文件**（新增 `tcenv/backends/metal/portfolio_performance.py`、`tcenv/backends/metal/baselines.py`、`tcenv/backends/metal/v2_runtime.py`、`P3/C.md`、`P3/evidence/gemm-domain-performance.json`、`P3/evidence/pointwise-domain-performance.json`）：

- P3-A 性能模型 CPU 节点：`tests/shared/test_shape_portfolio_performance.py`（**20 passed**）；
- P3-B runtime 编码 CPU 节点：`tests/backends/metal/test_launch.py`（**18 passed**）；
- P3-B native/Python 一致性节点：`tests/backends/metal/test_native_batch.py`（**5 passed**，含真实 Metal 单阶段与多阶段一致性）；
- P3-B/C 真实 Mac 性能门节点：`tests/backends/metal/test_shape_portfolio_performance_gpu.py`（**4 passed**，无 skip/xfail/deselect）；
- P0/P1/P2-0/P2 CPU 锚点：58 + 6 + 1 + 9 + 11 + 16 + 10 = **111 passed**，加上 P2 归档的 14/14 confirmed、active DispatchManifest（generation=1）与 checkpoint/resume/revoke 生命周期验证全部通过。

## 3. 历史阶段起点基线说明

`P3/acceptance-baseline.json` 记录于 commit `4a93ed2a4934fe526eedfcf4577de5dbd366e24a`（71 个合同面文件）。其后仓库完成了 varlen V5/V6、native64 Host Program、P2-F 安静窗口重录，以及将已完成的验收目录从 `docs/架构/` 迁入 `tests/support/golden/shared/`，使 71 个历史合同面文件中的 19 个 P2/portfolio 相关文件相对 `4a93ed2` 发生了预期演进。

与 `P2/F.md` 相同，我们**不篡改** `P3/acceptance-baseline.json` 或 `start-worktree.sha256` 的历史字节，也不伪造原始起点基线不存在的豁免：`P3/verify.py` 精确校验 `acceptance-baseline.json` 与 `start-worktree.sha256` 本身未变，并固定这组已知的历史演进集合，任何集合外的未声明漂移仍立即失败。
