# MLX small GEMM 官方反汇编分析（M4 Max）

## 范围与证据级别

本文分析 MLX `0.32.3` 自带 `.metallib` 的 **AIR LLVM 反汇编**，不是最终 AGX 机器码。
最终 AGX ISA 还需要从 Metal shader cache 提取 GPU Mach-O/DWARF；不能把本文的指令数直接当成
硬件 cycle 数。

分析对象：`M=256,N=256,K=128`、fp16 的 small GEMM。MLX geometry hook 已观测
`grid=(4,4,1)`、`threads=(32,2,2)`、静态 threadgroup 5,376B。

## 官方实际 kernel

符号表：

```text
steel_gemm_fused_nn_float16_float16_bm64_bn64_bk16_wm2_wn2
```

它是普通 `steel_gemm_fused`，**不是 NAX**。这与 M4 Max 不走
`metal::is_nax_available()` 的设备边界一致。

静态内存与 fragment：

```text
As = 1536 half
Bs = 1152 half
合计 2688 half = 5376 bytes

BlockMMA accumulator = [16 x <2 x float>]
```

官方 AIR 标识：`air64_v28-apple-macosx26.2.0`，Metal 4.0。

## 反汇编命令

```bash
xcrun metal-nm --defined-only .venv/lib/python3.13/site-packages/mlx/lib/mlx.metallib

xcrun metal-objdump --metallib \
  --disassemble-symbols=steel_gemm_fused_nn_float16_float16_bm64_bn64_bk16_wm2_wn2 \
  .venv/lib/python3.13/site-packages/mlx/lib/mlx.metallib \
  > /tmp/p3-mlx-small-disassembly.txt

# 从完整导出中截取该函数（完整导出包含所有 module，不能只读文件开头）
sed -n '11056848,11060048p' /tmp/p3-mlx-small-disassembly.txt \
  > /tmp/p3-mlx-small-kernel.air
```

用 `metal-perf air` 汇总整个函数的静态 intrinsic：

```text
2411 instructions
13x air.wg.barrier
18x air.simdgroup.barrier
6x air.simdgroup_matrix_8x8_multiply_accumulate (fp32 operands)
6x llvm.memcpy.p3i8.p1i8.i64
2x air.min.s.i32
```

**这 6 个 MMA/6 个 memcpy 是不同通用分支中的静态调用点，不表示 small 每次只执行 6 次。**

## small 的实际热路径

`align_K=true` 时，控制流从函数中的 label 291 进入 label 545，再因
`gemm_k_iterations_aligned=K/BK=8>0` 进入 label 559 的 aligned hot loop：

1. 每次 K iteration 对 A/B 各做一次 16-byte global→threadgroup copy；
2. `air.wg.barrier` 后由 BlockLoader/BlockMMA 读取 threadgroup fragments；
3. half fragment 转 float fragment；
4. 一个静态 `air.simdgroup_matrix_8x8_multiply_accumulate` 调用点，fp32 累加；
5. `label 581` 内层 fragment loop 后回到 K loop，动态执行 8 次；
6. K 结束后进入输出路径，累加器 `fptrunc float -> half`，再写回。

所以官方 small 的关键结构是：

```text
BK=16，8 次 K iteration
16B A/B tile copy + threadgroup barrier
fp32 fragment MMA
最终单次 fp16 舍入
```

## 与当前 golden 的 AIR 对比

当前 `tests/support/golden/metal/fixtures/gemm_paper_best.metal` 以 small 常量编译：

```text
283 instructions
3x air.wg.barrier
2x air.simdgroup_matrix_8x8_load (fp16)
1x air.simdgroup_matrix_8x8_multiply_accumulate (fp16 operands)
1x air.simd_shuffle_xor
```

普通 M4 Steel wrapper 编译结果：

```text
288 instructions
4x air.wg.barrier
3x air.simdgroup.barrier
2x 16B memcpy（静态调用点）
1x fp32 MMA（静态调用点）
```

差异：

- golden：`BK=32`，small 4 次 K iteration，fp16 operand fragments；
- 官方：`BK=16`，8 次 K iteration，fp32 operand fragments；
- 普通 Steel wrapper：结构接近官方，但用 small 编译期常量，静态指令数远小于官方完整通用函数。

**不能由 283 < 2411 推断 golden 更快。** 官方大部分指令属于 align/batch/axpby 等其它分支，
small 热路径只使用其中一个 fp32 MMA 调用点。真正应比较的是 hot loop、AGX code、GPU 占用与实际 wall。

## 对后续优化的意义

1. 官方 small 不是 M5 NAX，也没有神秘的 split-K 或不同 grid；
2. golden 的 fp16 fragment 路线本身已经是合理的 M4 优化，不应简单回退成官方 fp32 fragment；
3. 普通 Steel wrapper correctness 通过但未稳定提速，与 AIR 结构一致：单纯复制官方模板不能自动获得
   baseline 性能；
4. 下一步若继续，应提取 small hot path 的 **AGX 机器码和 profiler shader 统计**，比较真实指令、
   寄存器/occupancy、memory stall，而不是继续看总 AIR 指令数或猜 NAX/tiling；
5. 原生单 command-buffer 批量编码器仍是独立且已验证有效的路径优化方向。

本文只提供编译结构诊断，不构成性能验收或通过证据。

## 直接复用后的实际结果

后续已直接复用 `GEMMKernel::run`，并删除 plain-output 下冗余的
`TransformNone float -> half -> float` 预转换，让 `Ctile.store<half>` 完成单次输出舍入：

- correctness/ABI/dispatch 通过；
- best fresh 原型 candidate 2.201us、MLX 2.199us（ratio 1.001，但 noisy）；
- 同协议官方 binary 对照 candidate/official=0.949；
- 最新生产证据 ratio 约 1.329，虽然非 noisy，仍超过 1.10。

因此官方源码复用和直接输出优化确实有效，但尚不能宣称稳定达到官方效果。详细证据见
`evidence/native-batch-feasibility.json` 与 `evidence/gemm-performance.json`。
