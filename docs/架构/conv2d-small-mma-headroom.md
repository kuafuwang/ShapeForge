# conv2d small：单阶段 F(4×4) MMA 与 headroom 决策

本文记录 P2-J 的 small 专用变体。P2-I 的候选树、文档和 71 文件 manifest 保持不变；新候选树是 `tests/support/golden/metal/conv2d_small_mma/`，包含 small 与 paper 两个精确 guard 的 MMA variant，以及独立 general fallback。

## 先测提交下限，不直接写大 kernel

small 的最快登记基线是 MLX，P2-I 归档为 3.1786 µs；1.10 延迟门要求 candidate 不超过约 3.4964 µs。`/tmp` 中的不完整 probe kernel 只测 launch floor，没有正确性或性能发布含义，原始 JSON 归档在 P2-J `evidence/launch-floor.json`：

| 路径 | 墙钟中位数 | 结论 |
| --- | ---: | --- |
| 单阶段、无 fill | 1.6904 µs | 有有限优化空间 |
| 两阶段、无 fill | 6.1287 µs | 已超过 1.10 门，全局 U 两阶段路线不值得投入 |

两阶段数字连清零都没有计入，因此是乐观下限，不是可交付 candidate。small 必须使用单阶段，在每个 threadgroup 内重算 U；不能照搬 paper 的 producer/consumer 全局 U 结构。

## 正确实现与调优轨迹

第一个正确的直接卷积 `simdgroup_float8x8` 为 15.3339 µs，六 case 通过；相对 Torch-MPS 为 1.3051×，但相对 MLX 只有 0.2071×。随后用历史 F(4×4) 生成器的结构参数化出 C32/OC64 单阶段 Winograd，保留完整 6×6 输入检测和全部权重检测，IEEE 走顺序 fp32 直接卷积。中间结果均保留正式报告：

| 版本 | 墙钟 | 说明 |
| --- | ---: | --- |
| 初始单阶段 recompute-U | 4.2248 µs | 六 case 通过，已超过 Torch-MPS，但未过 MLX 门 |
| 合并输入检测、每个 U 只检查一次 | 3.9662 µs | 不减少检测覆盖 |
| `UGRP=4` | 3.5937 µs | 改善 K-block 调度 |
| `LDV=32` | 3.5236 µs | 缩小 V 行 stride |
| `FUSE=1, MF=0` | **3.2373 µs fresh** | 最终候选，identity `match` |

T/OCG、TS、CGRAN8、NOC2 等变体更慢，没有为拿绿选择错误或不安全实现。最终配置为 NFA=2、TS=1、OS=8、NOC=1、CGRAN=4、UGRP=4、LDV=32、FUSE=1、MF=0，单 stage，grid `(4,1,1)`，threads `(256,1,1)`。candidate SHA-256 是 `50ac29d1bc5c0ebcb983f16d5faedeaa0869324b10a979613900b29a640486f2`。

## Fresh 与 portfolio 证据

`evidence/small-confirmation.json` 是完整 `confirm_candidate` subject：搜索侧 3.4349 µs，fresh 侧 **3.2373 µs**，fresh seed `1431016671809190`，与搜索 seed 7 不同，identity `match`，第一次确认即 `confirmed`。fresh 相对 Torch-MPS 为 **5.8795×**，相对 MLX 为 **0.9858×**，即相对最佳基线仍约慢 1.4%，但延迟比 1.0144，满足 1.10 门。

P2-J 的两份搜索种子、reps 15 的真实 portfolio run 是 `small-run.json` 和 `paper-run.json`：small candidate 3.3004 µs，paper candidate 608.736 µs。两档都 `provider=candidate`、`outcome=resolved`，small 不再走 fallback。`full-eval.json` 只由这两份 run 派生，不冒充第二份 fresh 证据。paper 的旧 fresh 617.29 µs / 1.2461× 不改写。

独立 general fallback 仍注册并保留在候选树里；P2-I 已验证其 small/paper 正确性。新 CPU 测试覆盖 exact dispatch、guard 交叉相等、weak-guard overlap、单字节身份变化和归档重放。真实 Metal 测试覆盖两档六 case、partial tail、out1/out2/out3，以及两份 full eval 的稳定身份和 1.10 门。

本阶段 `publication_allowed=false`：small 相对最佳基线尚未大于 1×，且结论只覆盖登记的 small/paper fp16 点，不是全形状或生产 speedup 发布。
