# P2-I

Scratch workspace 把任务包闭包从 74 个文件推进到 79 个：scratch 两个文件，加上生产求值导入 Host Tiling 入口后的 `delivery.py`、`errors.py`、`precheck.py`。本目录用与 P2-H 相同的公开 master key 重录闭包、14 个默认域实例和 softmax per-launch 证据。P2-H、P2-G、P2-F、P3、P4 的历史文件没有改。

- epoch：`p2-i-scratch-1`
- parent HEAD：`72a9c3a08986b51a366bfc489e34038b5983dec6`
- pack：79 个文件，`5512395852ea7608a6075e2b6b45dca9bd85f112210af336cf6f8f77f5fd811f`
- 记录时 softmax 配对 fresh 中位数：0.967099× → 1.012472×
- `publication_allowed=false`

conv2d paper / fp16 的三臂 fresh confirmation 在 `evidence/conv2d-ablation.json`，每臂保留完整 `confirm_candidate` subject。`verify.py` 重算候选摘要和确认身份，核对搜索与确认的融合 GPU 中位数，并重跑 `test_conv2d_scratch_gpu.py` 的 6 个真实 Metal 节点。搜索种子是 7。确认种子：重算 `F(2×2)` 为 4067513467619784，scratch `F(2×2)` 为 1220837283154464，scratch `F(4×4)` 为 2163249565235418，都不是 7，三臂都是 `confirmed`。按当前 benchmark safety identity 重录后的墙钟中位数：17622.82 µs、19480.31 µs、17473.25 µs。同一份确认的融合 GPU 中位数：17576.44 µs、19425.36 µs、17416.40 µs。同一次 Torch-MPS 约 771–777 µs，MLX 约 978–1013 µs。`publication_allowed=false`。丢掉 subject 的上一版摘要留在 `evidence/conv2d-ablation-r1-summary.json`，不参与验收。

漏写最后一个 OC 的 quick 报告是 `unsafe`：三轮 `y` 随 `uwin` 初值变化，安全门先于数值比较。M2 计划文本写的是 `incorrect`。安全门不改，这一例不改判。部分写 `U` 仍是 `incorrect`，越界仍是 `unsafe`。

字段、生命周期和身份链见 `docs/架构/中间结果复用与Scratch Workspace.md`。

## F(4×4) simdgroup MMA consumer

旧三臂标量 consumer 的负结果保留，不改写。新的独立候选是 `tests/support/golden/metal/fixtures/conv2d_f4_mma_scratch.metal`，SHA-256 为 `5fbfea6c501f6439e5abc69a18633d1d1f6c8205759d4add39ce723c749b5be2`。它沿用历史 F(4×4) 的 interleaved U 布局和 `simdgroup_float8x8`，但使用独立 `ScratchSpec` buffer，不借用 `y`，也没有把 `uflag` 的 poison 初值当成算法分支。consumer 扫描完整 6×6 patch 与每个 filter 的全部 `K_TOTAL` taps；IEEE 输入进入直接 fp32 回退，普通输入保持 MMA fast path。六个 case 和 out1/out2/out3 均通过。

`evidence/conv2d-mma-f4-confirmation.json` 保留完整 `confirm_candidate` subject、候选路径和当前 benchmark safety identity。搜索侧为 602.93 µs、相对 Torch-MPS 1.2924×；fresh 侧为 617.29 µs、相对 Torch-MPS 1.2461×，fresh seed `818286732287391`，与搜索 seed 7 不同，identity 为 `match`，第一次确认即 `confirmed`。fresh 侧 MLX 为 994.27 µs、Torch-MPS 为 769.20 µs，candidate cv 0.01594。`verify.py` 重放 candidate SHA、六 case、identity 和 fresh margin，真实 Metal 回归在 `tests/backends/metal/test_conv2d_mma_gpu.py`。同一 evidence 还保留 `recompute_subject`。`metal_source_recompute()` 只改变 U 的来源：每个 threadgroup 重算相同的变换，consumer tiling、MMA、epilogue 和 IEEE 回退不变。recompute fresh 为 1474.42 µs、融合 GPU 1427.08 µs，identity `match`；与全局 U fresh 617.29 µs、融合 GPU 561.25 µs 相比，墙钟收益为 2.3885×，GPU 约 2.54×。两臂 fresh seed 不同，都通过六个 case。至此，全局 U 复用在 M4 Max paper 的同算法 A/B 中有明确净收益，不再是仅凭厂商 baseline 的推测。

这是受限 paper fixture 的实验性 fresh 证据，`publication_allowed=false`，不是生产发布或全形状结论。

最终 `verify.py` 完整通过：66 个 CPU 节点、18 个真实 Metal 节点、14 个域实例的 live/archived 稳定身份、两份 active manifest、checkpoint/resume/revoke、P4 离线重放，以及新旧 conv2d subject 重放。没有运行模型搜索，也没有改六 RPC、字段号、HMAC、BlobChannel 或 prompt SHA。

## small 与 paper 的候选 portfolio

上面那次 66+18 是 scratch 与 MMA 验收当时打印的标题。脚本里其实已经还有 P2-A/B/C/D、scratch 探针和 P4 这些 CPU 组。把那些组的 `expected` 加起来是 131 个 CPU 和 18 个 Metal。本轮把 conv2d small+paper portfolio 纳入同一份 P2-I verifier：候选树、`tests/support/conv2d_portfolio.py`、记录器、CPU/GPU 测试、`full-eval.json`、blocker，以及 `evidence/conv2d-portfolio-small.json` 和 `evidence/conv2d-portfolio-paper.json` 都在 `DELIVERABLES` 和 `manifest.sha256` 里。删掉其中任何一个，manifest 校验失败。

新增节点是 7 个 CPU 测试、1 个 launch-grid 测试和 5 个真实 Metal 测试。验收行因此改成脚本实际运行的 pytest 组之和：139 个 CPU 节点和 23 个真实 Metal 节点。

两份 run 是搜索种子 7、reps 15 的真实 portfolio 运行。small 是 fallback，墙钟 1264.88 µs，相对 Torch-MPS 的延迟比 65.63。paper 是 MMA 变体，墙钟 608.83 µs，相对 Torch-MPS 1.2647×，`candidate_sha256` 等于 `5fbfea6c501f6439e5abc69a18633d1d1f6c8205759d4add39ce723c749b5be2`。`full-eval.json` 由这两份 run 派生。Metal 节点重跑后核对 provider、身份、六案、安全和 1.10 门的方向，不要求墙钟微秒逐值相等。同一协议的前一次样本是 1255.56 µs / 64.73 和 612.69 µs / 1.2628×；这一次没有归档 run 文件，只作参考，不参与验收。归档 paper fresh 仍是 617.29 µs、1.2461×。`publication_allowed=false`。small 没有专用 MMA，性能推广未完成。

## 最终验收（r4）

`manifest.sha256` 原来还是新增 portfolio 之前的 59 行旧集合，缺 12 个文件、另有 4 个摘要过期。现在重写为 71 行，覆盖全部 `DELIVERABLES`。用临时副本和 monkeypatch 做反向检查：从 `DELIVERABLES` 去掉任一新增项、或在副本里改一个摘要，`verify_manifest()` 都抛 `AssertionError`，真实 manifest 未被改写。

完整 `verify.py` 在最终工作树上通过：139 个 CPU 节点 + 23 个真实 Metal 节点。新增的是 7 个 CPU、1 个 launch-grid、5 个真实 Metal。

真实 Metal 重跑（不是读自述 JSON）：paper 612.28 µs，相对 Torch-MPS 0.7897×、MLX 0.6257×，`candidate_sha256` 仍是 `5fbfea6c…`；small 1272.53 µs，相对 Torch-MPS 62.78×、MLX 398.69×，`provider=fallback`。两次重跑的 `stable_identity` 与归档逐字段相等，1.10 门方向不变。独立 reviewer 另跑一次，得到 paper 611.24 µs / 0.7838×、small 1271.70 µs / 63.36×，身份同样 match。

反向检查证据链：把归档 small 的 `outcome` 改成 `resolved`，Metal full-eval 节点在 `stable_identity(small) != stable_identity(archived_small)` 处失败，随后按原字节恢复（`61746ced…`，manifest 仍通过）。删掉候选树、把 small run 换成 paper run、改 `full-eval.json` 的 shard 数值，`verify_conv2d_portfolio()` 都会失败。

r3 指出的安全漏洞已经修：覆盖探针重新 dispatch 之后，quick 路径对同一 argset 重跑 `safety_violations()`，benchmark 路径用 `collect_benchmark_safety(sets, coverage)` 在探针之后收集 guard 和只读输入违例；两个负例节点在 verifier 里通过。

独立 reviewer 复核后给出 `done=true`，记录在 `review.json`（r1/r2/r3 的作废结论仍保留在各自的 `review-*-rejected.json`）。范围只是 P2-I scratch 机制和 conv2d small+paper portfolio 重放：`publication_allowed=false`，small 专用 MMA 仍未完成，没有发布任何加速数字。
