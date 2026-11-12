# P2-J：conv2d small 单阶段 MMA

- epoch：`p2-j-conv2d-small-mma-1`
- parent HEAD：`2462ef4e64e923cdd5653f22c0500301c036b6c6`
- pack：79 文件，`5512395852ea7608a6075e2b6b45dca9bd85f112210af336cf6f8f77f5fd811f`
- `publication_allowed=false`

P2-I 已关闭，本阶段没有修改其候选树、文档、证据或 manifest。`verify.py` 逐文件重算 P2-I 的 71 个历史摘要，并单独验证 P2-J 的 22 个交付文件。

## 候选与 headroom

small 最快基线 MLX 约 3.18 µs，两阶段无 fill 的 launch-floor probe 已为 6.1287 µs，因此最终采用单阶段 recompute-U，而不是 paper 的全局 U 两阶段结构。单阶段 probe 下限为 1.6904 µs。probe kernel 故意不完整，只用于可行性判断，不能作为 candidate 或发布证据。

最终 small variant 是 `tests/support/golden/metal/conv2d_small_mma/variants/small/kernel.metal`，SHA-256 `50ac29d1bc5c0ebcb983f16d5faedeaa0869324b10a979613900b29a640486f2`。NFA=2、TS=1、OS=8、NOC=1、CGRAN=4、UGRP=4、LDV=32、FUSE=1、MF=0；单 stage，grid 4×1×1、threads 256×1×1。六个 case、IEEE 和三轮初值均通过。

paper variant 与原确认 fixture 逐字节一致，SHA-256 `5fbfea6c501f6439e5abc69a18633d1d1f6c8205759d4add39ce723c749b5be2`。small/paper 都有九符号精确 guard、稳定 variant identity 和独立 general fallback。weak-guard overlap、单字节身份变化和归档身份漂移会被拒绝。

## 证据

- `evidence/launch-floor.json`：不完整 probe 的单/双阶段下限；`not_a_candidate=true`。
- `evidence/direct-mma-report.json`：正确的直接 MMA，15.3339 µs。
- `evidence/initial-recompute-report.json`：初始单阶段 Winograd，4.2248 µs。
- `evidence/headroom.json`：由报告派生的路线决策，不是另一次测量。
- `evidence/small-confirmation.json`：完整 fresh subject。fresh **3.2373 µs**，Torch-MPS **5.8795×**，MLX **0.9858×**，延迟比 **1.0144**，identity `match`，fresh seed `1431016671809190`。
- `evidence/small-run.json`：搜索种子真实 portfolio run，small candidate **3.3004 µs**。
- `evidence/paper-run.json`：搜索种子真实 portfolio run，paper candidate **608.736 µs**。
- `conv2d_small_mma/full-eval.json`：仅由上述两份 run 派生，`fresh_confirmation=false`。

small 相对最佳基线 MLX 仍约慢 1.4%，所以这里只声明满足 1.10 延迟门，不声明相对最佳基线加速或生产发布。paper 的既有 fresh 617.29 µs / 1.2461× 保持不变。

## 验收

`verify.py` 检查 manifest、79 文件 pack 闭包、P2-I 历史字节、新鲜 subject、两份 run 和派生 summary，并运行：

- 5 个 shared CPU 节点 + 1 个 launch 节点；
- 3 个真实 Metal 节点：small 六案、paper 六案、small+paper full-eval 重放。

预期通过行：`P2-J acceptance (6 CPU + 3 real Metal nodes; no model/search; publication_allowed=false)`。本阶段尚未声称独立 reviewer 签署。
