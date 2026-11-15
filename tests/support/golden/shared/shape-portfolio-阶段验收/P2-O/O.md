# P2-O：gemm_silu REG vs 全局 Scratch 正反对照

- epoch：`p2-o-gemm-silu-reg-vs-global-1`
- parent HEAD：`a9c1dccce636107c2852995578b05611bf19865b`
- pack：79 文件，`5512395852ea7608a6075e2b6b45dca9bd85f112210af336cf6f8f77f5fd811f`
- `publication_allowed=false`；NPU 仍 simulation-only

本阶段只做 Plan 2A，不改 P2-I/J/K/L/M/N 的任何历史文件，也没有 `git commit/reset/clean`。

## 结论

`gemm_silu` 的乘加/激活中间结果应留在寄存器/片段上，**REG 路线保留，GLOBAL 两阶段不实现**。

| 档 | REG fresh | 最快基线 | 延迟比 | 两阶段 floor | stage-2 流量下界 | 保守 lower bound | probe 外推 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `small` | 1.8252 µs | MLX 3.0520 µs | 0.5980 | 8.7713 µs | 0.8752 µs | **8.7713 µs = 4.806×** | 11.7960 µs |
| `paper` | 1160.8681 µs | MLX 1169.0564 µs | 0.9930 | 9.9254 µs | 52.9730 µs | **1211.4809 µs = 1.0436×** | 1240.6939 µs |

- `small`：两阶段启动下限已经是 REG 的 4.8 倍，任何全局中间结果都过不了 1.10 门。
- `paper`：合法的 fp32 中间结果必须在 stage 2 读回并写输出，保守的 24 MiB stage-2 流量在实测峰值下是 52.97 µs；扣掉最多能省的 2.36 µs，净增 50.61 µs（4.4%）。全序列化的 32 MiB 估计是 70.63 µs（6.1%），但不是下界。GLOBAL 没有省任何 FLOP，stage-2 流量是融合代价的约 22 倍（全序列化 30 倍）。
- 因此 `same_algorithm_ab_run=false`，没有为使用 ScratchSpec 写慢核。完整推导见 [gemm-silu-scratch-headroom.md](../../../../../../docs/架构/gemm-silu-scratch-headroom.md)。

## REG 候选与 fresh

候选树 `tests/support/golden/metal/gemm_silu_reg/`，两份都是单 stage、零 scratch、片段上的 fp32 SiLU：

- `small` SHA-256 `d5203e459ac2c8bca4d1d59a0fb22ba5910e568fde98fe9b83da7e43f9f07d4c`，六案通过；
- `paper` SHA-256 `461cff911579129e2a6209fc31c2a14db68eac867c1d15d18d6378c713cc5330`，六案通过。

fresh 用 P2-I 的公开 master key，seed 分别为 `3775173618909035`（small）和 `6349172394694722`（paper），与搜索 seed 7 不同；identity `match`，第一次确认即 `confirmed`，两档都过 1.10 延迟门。`paper` 的 1.0071× 只是本协议位置，不是发布。

`paper` 第一次 fresh 被外部负载抬到 1315.23 µs（MLX 同时 1330.83 µs），原样留在 `evidence/reg-paper-confirmation-occupied.json`，不进入决策；安静窗口重跑 1160.87 µs（cv 0.0047），seed 与身份与第一次相同。

两档都通过真实 `evaluate_candidate` 的六案 quick 路径：每个 case 都跑 harness 的三轮初值（out1/out3 填毒、out2 填零），`safety.ok` 与空的 `nondeterminism` 说明结果不依赖输出初值，且只读输入未被修改。REG 候选是单 stage、无 scratch、无 Host Program、无 prepare 缓存；每个计时迭代都重新执行完整 GEMM 和 epilogue，没有可跳过的中间状态。本阶段没有新增 shape portfolio：两个源码按登记实例分别评测，不对其它 M/N/K 声明覆盖，也没有 variant/fallback 分派需要演练；因此不把这两份源码写成跨形状交付候选。

## 融合代价与 probe

同算法 plain GEMM control 与 REG 的差值是“激活留在片段上”的代价，但两者是独立 full run，不是同一次运行内的 A/B：`small` −0.177 µs 在这个尺度上不可分辨，`paper` +2.360 µs（0.203%，跨 run；独立的 `5*M*N` flop 下界约 1.42 µs）。

`experiments/gemm_silu_scratch_probe.py` 的真实 Metal probe（`not_a_candidate=true`）：

| 臂 | small | paper |
| --- | ---: | ---: |
| 单阶段写满 fp16 | 3.6150 µs | 24.3896 µs |
| 两阶段 floor（无 fill） | 8.7713 µs | 9.9254 µs |
| fp32 纹理写+读（无 fill） | 13.5858 µs（+9.9708） | 104.2154 µs（+79.8258） |
| fp32 buffer 写+读 + `device_zero` fill | 24.0550 µs（+20.4400） | 496.3208 µs（+471.9312） |

纹理臂不是只声明字节数：每个逻辑 fp32 有独立的 R32 texel 写/读，producer 写 `float((i & 1023) + 1)`，consumer 写同一值到输出 `i`，运行后读回整块 scratch 与输出并逐个比较。两档的 texture/buffer 臂都是 `elements=M*N`、`scratch_elements_match=true`、`output_elements_match=true`；旧的 `float4` 单坐标 quarter 版本被同一检查在 `index 1` 抓到（`quarter_coverage_negative.detected=true`）。这是新增的回归门，旧版本不能通过。

`paper` 的纹理路径约 402 GB/s，buffer+fill 路径约 118 GB/s；16 MiB 的逐字节 fill 单独就把 buffer 路线推到近 500 µs。决策用的是更保守的硬下界（两阶段 floor 与 `6*M*N / 实测峰值带宽` 的 stage-2 流量，并扣掉融合代价），不是最差 probe；`8*M*N` 的全序列化只是参考估计，不是下界。

## 证据与验收

`evidence/`：

- `reg-small-confirmation.json`、`reg-paper-confirmation.json`：完整 fresh subject；
- `reg-paper-confirmation-occupied.json`：负载污染的重试，保留不删；
- `plain-gemm-small-control.json`、`plain-gemm-paper-control.json`：同算法 control；
- `scratch-probe.json`：四个 probe 臂，`not_a_candidate=true`；
- `decision.json`：由上述文件计算，verifier 重算并逐字段比较。

`verify.py` 检查 manifest、79 文件闭包、提示 SHA、trust SHA、P2-I..N 历史字节、REG 源码/六案/fresh/control/probe/decision/coverage，并运行 6 个 CPU 节点与 6 个真实 Metal 节点：

```bash
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-O/verify.py
```

预期最终行：`P2-O acceptance (6 CPU + 6 real Metal nodes; REG kept, GLOBAL rejected, publication_allowed=false)`。

本阶段尚未声称独立 reviewer 签署；reviewer 的结论另存 `review.json`（不进入历史 manifest 的交付集合）。
