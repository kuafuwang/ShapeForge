# Backend GPU 回归（2026-10-23）

> 本文是 2026-10-23 的一次性本地回归记录。下面的数字和结论只对应那天的代码、机器状态和本地日志，不随代码更新，也不能当作当前基线。后续复跑应按文末条件重新记录。

范围：用户授权的两项：

1. 每个 task 跑一次编译 + 正确性 + 计时 smoke；
2. `flash_attention` 与 `docs/架构/backend-phase0-baseline.json` 的 P0 基线对比。

没有模型调用；只使用本机 Metal。

## 1. 全任务 smoke

命令模式：

```bash
uv run python -m tcenv.cli naive <task> --dtype fp16 --size paper --seed 12345 --reps 5
```

共 16 个 task，全部返回 `STATUS: ok (stage reached: done)`：

```text
sigmoid relu silu swiglu rmsnorm softmax reduction_sum rope
gemv gemm gemm_silu conv2d flash_attention lion bitdelta_gemm
mamba2_chunk_scan
```

原始输出在本地 `logs/gpu-smoke-20261023-204514/`，未提交。

## 2. Phase 0 对比

使用与基线相同的条件：

```text
task    flash_attention
dtype   fp16
size    paper
seed    105886635628786
mode    full
reps    15
source  tests/support/golden/metal/backend_phase0_kernel.metal
```

本地共跑 6 次，取全部 6 次的中位数；原始 JSON 在
`logs/gpu-phase0-20261023-204713/`，未提交。

| 指标 | P0 old | P0 recorded current | 本次 6 次中位 | 相对 old | 相对 recorded current | 2% / 0.03 门 |
|---|---:|---:|---:|---:|---:|---|
| candidate wall µs | 322.4375 | 320.3125 | 321.1730 | -0.392% | +0.269% | pass |
| candidate GPU µs | 299.6367 | 298.3711 | 305.7129 | **+2.028%** | **+2.461%** | **fail by 0.028 pp** |
| MLX µs | 331.4359 | 331.0750 | 337.9208 | +1.957% | +2.069% | pass |
| speedup vs MLX | 1.027907 | 1.033634 | 1.052796 | +0.024889 | +0.019162 | pass |

## 结论

- 16 个 task 的 smoke 全部通过；没有编译、ABI、安全、正确性或计时阶段失败。
- `flash_attention` 的 wall time 和 speedup 在 P0 门内；GPU-timestamp 中位数比 P0 old 高 2.028%，
  超出固定 2% 门 0.028 个百分点，属于临界失败。
- 单次运行有 GPU 时间戳和 MLX 基线的明显漂移（6 次中 GPU 为 298.8–318.7 µs，MLX 为 329.0–343.0 µs）；
  wall time 中最慢一次为 340.4 µs，其余为 319.1–323.3 µs。
- 因此本次**不更新 P0 基线、不把该差异写成确认回归，也不提交任何 kernel catalog 改动**。
  若要做正式发布门，应在机器更安静、GPU 温度稳定时重跑同一 seed/reps，并重新记录一份独立样本。
