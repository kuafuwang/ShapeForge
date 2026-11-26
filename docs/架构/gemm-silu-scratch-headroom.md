# gemm_silu：REG 融合 vs 全局 Scratch 的停止结论

P2-O 只回答一个问题：`C = silu(A @ B)` 的乘加与激活中间结果，应该留在寄存器/片段上，还是写成一块全局 fp32 scratch、屏障之后由第二个 kernel 读回并激活？本阶段用真实 Metal 测量了完整成本，结论是 **保留 REG**。GLOBAL 两阶段没有可补偿的计算节省，且在 small 上连两阶段启动下限都过不了，在 paper 上仅 stage-2 的必需流量就超过整个融合代价一个数量级以上。因此没有实现 GLOBAL 候选，也没有为了使用 `ScratchSpec` 写一个已知更慢的核。

`publication_allowed=false`。下面所有微秒数都是本阶段证据，不是生产 speedup 发布。

## 形状与 roofline

`tasks/gemm_silu/opspec.json` 声明两个 fp16 档，六个 case（含 `ieee`）：

| 档 | M | N | K | 声明 flops | 最小字节 | 计算下界 | 内存下界 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `small` | 256 | 256 | 128 | 17,104,896 | 262,144 B | 1.158 µs | 0.584 µs |
| `paper` | 2048 | 2048 | 2048 | 17,200,840,704 | 25,165,824 B | 1,164.9 µs | 52.97 µs |

`paper` 是计算受限，现有 REG 候选已经贴着计算下界；`small` 是启动受限档。`flops` 含激活的 `5*M*N` 项，计算下界用报告里实测的 fp16 峰值（14.7659 TFLOPs）。

输出并行度不是问题：`paper` 的 grid 是 `32×32 = 1024` 个 threadgroup，`small` 是 `4×4 = 16` 个。没有任何需要 Split-K 或跨块归约的输出并行度缺口。这里的中间结果只属于单个输出元素，做完乘加就在同一线程的片段里，不需要被另一个执行单元再读。

## REG 候选

候选树是 [`tests/support/golden/metal/gemm_silu_reg`](../../tests/support/golden/metal/gemm_silu_reg)：

| 档 | 来源 | 单阶段 | SHA-256 |
| --- | --- | --- | --- |
| `small` | `gemm_small_steel.metal`（MLX v0.32.3 steel GEMM，BM=BN=64/BK=16） | `grid=4,4,1 threads=32,2,2` | `d5203e459ac2c8bca4d1d59a0fb22ba5910e568fde98fe9b83da7e43f9f07d4c` |
| `paper` | `gemm_paper_best.metal`（P3 的 2048³ 候选） | `grid=32,32,1 threads=128,1,1` | `461cff911579129e2a6209fc31c2a14db68eac867c1d15d18d6378c713cc5330` |

两份源码都只加了一个片段上的 fp32 SiLU epilogue，`math fast`，一次 fp32→fp16 存储：

- `small`：新增 `mlx::steel::TransformSiluFused`，并且补上 stock fixture 在 MN-aligned 快路径漏掉的 `Epilogue::apply`。不改这一步，`256×256×128` 会静默丢掉激活，实测得到未激活的 GEMM。
- `paper`：在 `thread_elements()` 上就地算出 SiLU，再走原来的 `(half)` 转换。**不能把结果写回 `te[0]/te[1]`**：写回 simdgroup 片段会让编译器把 16 个累加片段整体溢出，paper 从 1177 µs 变成 2464 µs。这是本阶段唯一一次实现返工，原因和最终写法都保留在源码注释里。

六案全部通过。`small` 的 `tiny` scaled error 是 0.9778（门限 1.0，靠输出格式的亚正规 floor 通过），其余 case 最大 0.2138；`paper` 的 `tiny` 是 0.9759，其余最大 0.2233。`ieee` 的 NaN/inf 与参考逐位置一致（fresh 采样里 `small` 4686 个 NaN，`paper` 399815 个）。

## 基线与 fresh

两份 fresh subject 都用 P2-I 的公开 master key 派生 fresh seed，搜索 seed 仍是 7，identity `match`，第一次确认即 `confirmed`：

| 档 | REG fresh | 最快基线 | 延迟比 | 相对最快基线 | 次快基线 | 相对次快 |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| `small` | **1.8252 µs** | MLX 3.0520 µs | 0.5980 | 1.6721× | Torch-MPS 15.8080 µs | 8.66× |
| `paper` | **1160.8681 µs** | MLX 1169.0564 µs | 0.9930 | 1.0071× | Torch-MPS 1219.4722 µs | 1.0505× |

`small` 是 `fp16/small/benchmark`，fresh seed `3775173618909035`；`paper` 是 `fp16/paper/benchmark`，fresh seed `6349172394694722`。两档都过 1.10 延迟门。`paper` 相对最快基线只快 0.7%，仍不足以当成稳定的生产加速结论。

`paper` 第一次 fresh 采样被外部负载抬到 1315.23 µs（MLX 同时被抬到 1330.83 µs，cv 0.0411）。它原样保存在 [`reg-paper-confirmation-occupied.json`](../../tests/support/golden/shared/acceptance-epochs/fixtures/p2o/evidence/reg-paper-confirmation-occupied.json)，只作重试记录，不进入决策；安静窗口重跑得到上表的 1160.87 µs（cv 0.0047），seed 和 build/evaluation identity 与第一次相同。

## 同算法融合代价

REG 相对同一 GEMM 算法、只去掉 SiLU 的 control，差值就是“把激活留在片段上”的代价。control 在 `tasks/gemm` 上跑，因为 `gemm_silu` 的正确性要求 SiLU；两侧的 ABI、形状和计时协议相同，候选源码只差 SiLU epilogue 那几行：

| 档 | REG | plain GEMM control | 融合代价 | 占 REG |
| --- | ---: | ---: | ---: | ---: |
| `small` | 1.8252 µs | 2.0022 µs | −0.177 µs | −9.7% |
| `paper` | 1160.868 µs | 1158.508 µs | **+2.360 µs** | **+0.203%** |

这是两次独立 full run 的跨 run 差，不是同一次运行内的 A/B。`small` 的 −0.177 µs 在这个尺度上无法分辨是融合收益还是运行间偏移，只能记为“未分辨”；不能读成“融合更快”。`paper` 的 +2.36 µs 同样带跨 run 不确定度；独立的 flop 下界是 `5*M*N / 14.766 TFLOPs = 1.42 µs`。两个估计都约为 REG 的 0.2%（paper 的 2.36 µs 是 0.203%）。决策只需要一个上界：**把 SiLU 移出 REG 最多只能省回约 1.4–2.4 µs（paper），small 则是无法分辨的零点几微秒**。

## GLOBAL 的完整成本

GLOBAL 的合法形态必须是：GEMM 把 fp32 累加器写进全局 scratch → buffer barrier → 第二个 kernel 读 fp32、做 SiLU、写 fp16 输出。不能写 fp16 中间结果，因为 OpSpec 的 `rounding=single_round_at_output` 只允许在输出处舍入一次；先舍到 fp16 再算 SiLU 会引入第二次舍入。

因此 GLOBAL 相对 REG 的增量是固定的：

- scratch 写入 + 读回 = `2 * M * N * 4` 字节；
- 多一个 stage 的启动和 barrier；
- 如果选 buffer + `device_zero`，还要在计时窗内清零整块 fp32 scratch（16 MiB at paper）。

`experiments/gemm_silu_scratch_probe.py` 在真实 Metal 上测了四个不完整 probe（`not_a_candidate=true`，没有 GEMM、不参与正确性）：

| probe 组 | 做什么 | small | paper |
| --- | --- | ---: | ---: |
| `single_stage_fp16_write` | 单阶段写满 fp16 输出（REG 的 store 基线） | 3.6150 µs | 24.3896 µs |
| `two_stage_floor` | 4 个 fp32 纹理元素，producer → barrier → consumer，无 fill | **8.7713 µs** | 9.9254 µs |
| `two_stage_texture_roundtrip` | 每个逻辑 fp32 一个 R32 texel，写+读，无 fill | 13.5858 µs（比单阶段 +9.9708） | 104.2154 µs（+79.8258） |
| `two_stage_buffer_roundtrip` | 同样字节走 buffer，`device_zero` fill 在计时窗内 | 24.0550 µs（+20.4400） | 496.3208 µs（+471.9312） |

纹理组不是只按 JSON 字节数自述：producer 把 `float((i & 1023) + 1)` 写进元素 `i`，consumer 把同一值写到输出 `i`；一次运行后 probe 读回整块 scratch 和整块输出，要求 `M*N` 个元素逐个相等。`small` 与 `paper` 的 texture/buffer 组都是 `scratch_elements_match=true`、`output_elements_match=true`、`elements=M*N`。旧的 `float4` 在单个 R32 坐标上写的 quarter 版本会留下 3/4 的 poison，并以同一检查在 `index 1` 处被抓到（`quarter_coverage_negative.detected=true`）。这个检查是 P2-O 的回归门，而不是只比较声明的字节字段。

`paper` 的纹理路径达到约 402 GB/s（`10 * M * N / 时间`，40 MiB），buffer+fill 路径只有约 118 GB/s：16 MiB 的 `device_zero` 由 harness 的逐字节 fill kernel 清零，`16 MiB / 256 B = 65536` 个 threadgroup，单这一项就把时间抬到近 500 µs。纹理路径是更公平的上界，它没有 fill；但它仍然比单阶段写多 79.83 µs。

决策使用两个保守下界，而不是拿最差的 probe 当结论：

- 两阶段启动下限 `two_stage_floor`：任何 GLOBAL 都至少有两个 stage；
- **stage-2 的读+输出流量** `6*M*N / 实测峰值带宽`（fp32 读 4 B + fp16 写 2 B；paper 475.07 GB/s，small 449.27 GB/s）。producer 多写的 `2*M*N` fp32 可能被 compute-bound GEMM 吸收，因此不计入下界；GLOBAL 的 stage 1 不付 SiLU，所以再减去测到的融合代价（取 0 与实测值的较大者）。

`8*M*N / 峰值` 的全序列化数字仍保留为参考估计（paper 70.63 µs），但它**不是**下界：它假设 producer 的额外写完全串行，并重复计入了 stage 1 不做的 SiLU。

| 档 | REG | stage-2 流量下界 | 两阶段 floor | 保守 lower bound | probe 外推 | 保守 / REG | probe / REG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `small` | 1.8252 µs | 0.8752 µs | 8.7713 µs | **8.7713 µs** | 11.7960 µs | **4.806×** | 6.463× |
| `paper` | 1160.868 µs | 52.9730 µs | 9.9254 µs | **1211.4809 µs** | 1240.6939 µs | **1.0436×** | 1.0688× |

`paper` 的保守 lower bound 是 `REG + 52.97 µs − 2.36 µs 融合代价 = 50.61 µs`，即 1.0436×；如果完全不给融合代价信用，是 1.0456×；如果按独立 flop 下界 1.42 µs 给信用，是 1.0444×。producer 写和纹理低效都没有拿来当必要条件；实测 texture probe 外推 1240.69 µs（1.0688×）比这个下界更高，符合预期。`small` 的保守 lower bound 是两阶段 floor 本身，已经是 REG 的 4.8 倍；probe 外推 11.7960 µs 高于 floor 8.7713 µs，两者都远高于 REG，因此不是靠外推撑结论。

两个档都满足 `global_would_win=false`。完整推导、每一级数字和 stop reason 在 [`evidence/decision.json`](../../tests/support/golden/shared/acceptance-epochs/fixtures/p2o/evidence/decision.json)，由记录脚本从 fresh subject、control 和 probe 计算，verifier 逐字段重算。

## 为什么没有做 GLOBAL A/B

按计划第 1 条，先测/算再实现。两条停止条件都已触发：

1. **两阶段下限已排除 small**：8.77 µs 的 floor 已经是 1.83 µs REG 的 4.8 倍，任何真实的 Split/两阶段算法都不可能把它拉回 1.10 门以内。
2. **scratch 流量已排除 paper**：合法的 fp32 中间结果必须在 stage 2 读回并写输出，保守的 stage-2 流量 24 MiB 在实测峰值带宽下是 52.97 µs；扣掉把激活移出 REG 最多能省的 2.36 µs，净增 50.61 µs（REG 的 4.4%）。全序列化的 32 MiB 估计是 70.63 µs（6.1%），但那不是必要条件。stage-2 流量是融合代价的约 22 倍，且没有省掉任何 FLOP。

即使允许一个不合法的 fp16 scratch（会引入第二次舍入），写+读也只有 16 MiB，约 35.3 µs（paper）或 0.58 µs（small）；paper 仍大于 2.36 µs 的融合代价，small 仍需再加上两阶段 floor。结论不依赖数值精度上的让步。

所以本阶段没有写 GLOBAL 候选，`implementation.global_candidate_source_added=false`、`same_algorithm_ab_run=false`。这里记录的是 **negative result**：GLOBAL 被数据否决，REG 保留。不是为了使用 ScratchSpec 而实现一个慢核。

## REG、SHARED 与 conv2d 的边界

这次是 `REG`：中间结果属于单个线程的片段，做完就消费，零全局 scratch。conv2d 的全局 `U` 是 `SHARED`：一份权重变换被许多 tile 读，producer 的固定成本可以被大量 consumer 摊薄，所以 P2-I 在 paper 上测到 2.39× 的全局复用收益。两者的判据不同：

- `REG` 的对照是“把激活移出片段”，省下的是几个 flop；任何全局往返都会输。
- `SHARED` 的对照是“每个 consumer 重算同一份变换”，省下的是重复计算；只有当 producer + 流量小于重复计算时才会赢。

不能把 conv2d 的正例机械套到 gemm_silu，也不能把这里的负例推广成“全局 scratch 永远没价值”。`gemm_silu` 没有第二个消费者，也没有可复用的跨 tile 变换。

## 证据、重放与边界

- 两档 fresh subject：[`reg-small-confirmation.json`](../../tests/support/golden/shared/acceptance-epochs/fixtures/p2o/evidence/reg-small-confirmation.json)、[`reg-paper-confirmation.json`](../../tests/support/golden/shared/acceptance-epochs/fixtures/p2o/evidence/reg-paper-confirmation.json)；被负载污染的重试在 `reg-paper-confirmation-occupied.json`。
- 同算法 control：[`plain-gemm-small-control.json`](../../tests/support/golden/shared/acceptance-epochs/fixtures/p2o/evidence/plain-gemm-small-control.json)、[`plain-gemm-paper-control.json`](../../tests/support/golden/shared/acceptance-epochs/fixtures/p2o/evidence/plain-gemm-paper-control.json)。
- probe：[`scratch-probe.json`](../../tests/support/golden/shared/acceptance-epochs/fixtures/p2o/evidence/scratch-probe.json)，`not_a_candidate=true`。
- 决策与重算：[`decision.json`](../../tests/support/golden/shared/acceptance-epochs/fixtures/p2o/evidence/decision.json)。

重录与重放：

重录入口仍是 `uv run python -m tests.backends.metal.gemm_silu_scratch_record --confirmation --control --probe --decision`。验收目录已移出工作区，恢复命令在 `tests/support/golden/shared/retired.json`。上面的证据链接指向仍被测试读取的副本。

单个节点：

```bash
uv run pytest -q -n 0 tests/shared/test_gemm_silu_scratch_decision.py
uv run pytest -q -n 0 -m gpu tests/backends/metal/test_gemm_silu_scratch_gpu.py
```

边界：

- `publication_allowed=false`，不发布 speedup；`paper` 的 1.0071× 和 `small` 的 1.6721× 只说明本协议的相对位置，不是生产数字。
- 没有启动 `taic` 付费搜索，没有模型调用；没有改提示面、`current_prompt_sha1`、六 RPC、字段号、HMAC 或 BlobChannel。
- 清微 NPU 仍只有模拟器；本阶段没有跑真实 NPU，也没有改 `tests/support/npu_sim/`。
- P2-I/J/K/L/M/N 的 manifest 逐文件保持不变；P2-N 的未提交工作树原字节保留。任务包闭包仍是 79 个文件，摘要 `5512395852ea7608a6075e2b6b45dca9bd85f112210af336cf6f8f77f5fd811f`。

## 剩余限制

- 结论只覆盖登记的 `fp16/small` 与 `fp16/paper` 两个形状，不外推到其它 M/N/K、bf16 或非 MLX steel 算法。
- `paper` 的 1.0071× 离 1.0× 只有 0.7%，不是稳定加速的发布证据；本阶段也没有做多轮 fresh 的置信区间。
- `small` 的 `tiny` case 的 scaled error 0.9778 接近 1.0，原因是输出落在 fp16 亚正规区、靠格式 floor 通过；后续若改激活公式需要重新检查这一档。
- probe 的纹理路径实测约 402 GB/s，低于 buffer 峰值；因此 `paper` 的 79.83 µs 增量是这套纹理 probe 的外推，不是理论最小值，也不是 GLOBAL candidate 的实测。决策不依赖它：保守下界 50.61 µs（1.0436×）已经足以否决 GLOBAL。
- 历史 `runs/triton_baselines/` 里还有一组 Triton GLOBAL 观察（plain GEMM 1381.63 µs，GLOBAL 1440.56 µs，+58.93 µs），但 `runs/` 不入库、也不属于 P2-O 证据，只作为旁证，不进入决策或 verifier。
