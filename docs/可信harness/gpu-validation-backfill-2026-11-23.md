# GPU 验证补验：native guard、计时、recheck、AB 与 scratch

本次获得用户明确 GPU 授权，串行执行预定 Metal 补验；**没有付费模型、搜索或 NPU 设备调用**。脚本模型与既有候选的模型成本为 $0。结论只覆盖下列真实 Metal 机制，不是 speedup 发布，也不是全部 299 个 GPU 节点的全量结果。

## 执行的 pytest 节点

最终共 **15 个真实 GPU 节点通过**：

| 范围 | 节点与真实结果 |
| --- | --- |
| native Host 绑定存储 | `test_varlen_hostprogram_gpu.py` 新增两例：真实绑定 `cu_seqlens` 的位型改写判 `incorrect` 且报告 `input ... was modified`；越起点一个 byte 判 `incorrect`、`guard sentinel`，没有 workspace write 提交。正向 C99 planner 的 9 个 L1 case 也通过。 |
| per-launch 与 A/A control | `test_per_launch_preserves_read_write_generation_dependencies` 在真实 SharedEvent 路径上验证 `gpu_s>0`、`gpu_span_s>=gpu_s`、`gpu_per_iter_s=gpu_s/6`；`test_real_a_a_control_is_wall_clock_and_not_a_baseline` 验证 control 为 wall-clock、seed 7、非 baseline，候选 GPU 中位数非空。 |
| 选中 attempt 的 GPU 样本 | `test_p2_g_softmax_paired_fresh_evaluations` 的 single-buffer / per-launch 三对 fresh 评测通过；两条路径都新增 `candidate_gpu_us_median>0` 和实际 attempts 边界断言。 |
| differential recheck | `test_search_v2_gpu.py` 正向 GEMM：脚本修复后的候选在 seed 7 真实 recheck 通过。新增 ReLU 反例：同一份字节 full=`ok`、派生 fresh quick=`incorrect`，不成为种子；`rechecks/init.json` 两份 provenance 的 SHA 都是 `53807ecd8798fee93b34d3877bf024cef12ca91f3d087acf2763d3c9d9c9a060`。两臂均 `seed=7`，无付费模型。 |
| AB 停止路径 | `test_ab_gpu.py` 用真实 Metal runner 与时间戳，在安全钩子注入计时后 guard 违规；结果 `unsafe`、无 rows、无排名，坏臂真实启动一次，另一臂 0 次。注入只验证停止控制流；真实越界写由上面的 Host/MTLBuffer 负例证明。 |
| 资源 fallback | `test_p1_gpu_a_single_source_launch_overflow_runs_only_the_declared_fallback` 通过：真实源内 launch 超限只执行已声明 `resource_unavailable` 的 fallback。 |
| conv2d scratch | 漏写 OC=`unsafe`、部分写=`incorrect`、越界=`unsafe`，以及 F(4×4) 含部分尾块的全 case 正向，共 4 节点通过。 |

相关非 GPU 回归为 **92 passed**，`git diff --check` 通过。

## 真实 AB smoke

执行：

```bash
uv run python -m tcenv.cli ab \
  kernels/apple-m4-max/conv2d/fp16/paper/best.metal \
  tests/support/golden/metal/conv2d_portfolio/variants/paper/kernel.metal \
  --task conv2d --dtype fp16 --size paper --reps 4 --json
```

两臂均 `status=ok`、`timed_by=gpu_timestamp`、`measurement_scope=fused_command_buffer_per_iteration`：

| label | SHA-256 | median | cv |
| --- | --- | ---: | ---: |
| `best.metal` | `285f90f945b8d21865c3284f03b77ddec07960bcd4a4f6b77a3c3c255c7278fc` | 390.44791871371365 µs | 0.02855015772784958 |
| `kernel.metal` | `5fbfea6c501f6439e5abc69a18633d1d1f6c8205759d4add39ce723c749b5be2` | 565.9054508074545 µs | 0.008508446887802988 |

`paired_ratio_to_first=0.6951075987844966` 只是这次只读 N 路消融，不是 fresh publication，也不消耗 `--timed` 晋级额度。

## 真实 v2 profile

对 conv2d paper 候选 `5fbfea6c…` 执行 `tcenv profile ... --json`，真实 Instruments Metal System Trace 返回：

- 19.103166209 秒窗口、7511 次 dispatch；
- 1161 次 spill、538704 spilled bytes；
- shader code 42620 bytes；
- thermal state `Nominal`；median dispatch 159.542 µs，p10/p90 = 66.0/1517.25 µs。

Instruments 没导出 process 列，因此是 **machine-wide** 窗口，且 tracing 会扰动计时。这里只证明 v2 profile 能真实记录压力信号，不把上述时间当中位数性能或加速证据。

## 短尾探针的合同边界

曾尝试用 `u8[5]` + device pointer 构造短尾 scratch。真实 ABI 合法拒绝：`uninitialized_poison` 的首个 producer 必须反射为 write-only，device pointer 只能反映 readwrite；合法纹理路线是 fp32。故当前生产合同没有这条 u8 短尾执行路径，新增测试已撤回，未为得到结果放宽 ABI。非对齐/短尾逐字节算法仍由 `test_scratch_boundary_cpu.py` 覆盖；真实 GPU 上本次确认的是 conv2d 的部分写、越界和部分尾块。

## 边界与下一步

- 本批修改测试与断言，没有改生产内核、任务、trust、prompt 或历史验收包。
- 没有运行完整 GPU 套件、付费搜索、真实 SSH/远程站点或 NPU；`publication_allowed=false`。
- 远程真实基础设施与第二设备仍是独立阶段，不能由本机 Metal 结果外推。
