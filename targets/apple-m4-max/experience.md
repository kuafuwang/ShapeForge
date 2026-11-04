# 经验条目（本仓库实测过的那一部分）

这份文件是 KernelWiki 里「经验层」的入口：每条都给出观察、来源、本机实测、边界，以及什么时候不成立。
和 `INDEX.md` 里那两份 Apple PDF 的区别是：PDF 说语言与设备能力，这里说**我们自己测到过什么**。

证据等级写在每条的 `Confidence:` 上。`verified` = 本机复现过，`来源` 里必须有 `runs/` 或 `docs/` 的路径；
`reported` = 只有上游出处（别人的 PR、博客、WWDC）还没在本机复现，**不进提示词**，只留在库里。

## 累加器片段到 32 会溢出

Keywords: gemm accumulator register spill simdgroup fragment tile occupancy
Confidence: verified
观察: 每组 SIMD 的累加器片段达到 32 个及以上时，寄存器溢出，同一题的耗时是一千多微秒变到一万七到三万五微秒。
来源: `runs/hw_features/REPORT.md`（同一批契约的实测对照）；`runs/apple-m4-max/harness-pilot-shadow-20261022b/agent_work/init/v3b.metal`、`v2_pad.metal`、`FEEDBACK.md`；proposal `43c462b0b4f70803`。
本机实测: fp16 paper，32 片段以上 17880–35006 µs，16 片段及以下 1361–1942 µs，约 12–26 倍。
补充（conv2d shadow-b 同核对照）: 16 槽 690 µs，32 槽 1069–1079 µs（1.55x）；两版报告都是 `maxTotalThreadsPerThreadgroup=1024`，静态 threadgroup 内存也正常，所以这个字段区分不了。`tcenv` 现在打印源码声明的 `simdgroup_float8x8` 槽数和最大数组，作为源码级启发式，不冒充编译器寄存器数或 spill 数据；要确认 spill 用 `tcenv profile` 的 spill events。
边界: 只覆盖 fp16 与本机（M4 Max）的寄存器文件；换成半分块或更小的 BLOCK 时片段数会变。惩罚幅度随核和契约变化，同核 conv2d 对照是 1.55x，不等于上一条的 12–26x。
反例: 把 K 维切得更细、用 threadgroup 做中间累加，可以既保留大 tile 又不让片段数越界。

## 矩阵单元与 SIMT ALU 的 fp32 吞吐相近，工作相加

Keywords: matrix simdgroup fp32 fma simt throughput additive operand reuse
Confidence: verified
观察: 本机 fp32 8x8x8 simdgroup MMA 的吞吐与 fp32 SIMT FMA 相近；两种工作混跑时时间相加，不重叠。矩阵单元对 fp32 不是额外的一份算力，真正的红利是操作数复用。
来源: `runs/apple-m4-max/conv2d-feedback-20261021/agent_work/r11.p0/probe_mix.py`、同目录 `NOTES.md`
本机实测: 8 warp 全 MMA 12.62 TFLOP/s，4 warp 全 MMA 14.34，8 warp 全 FMA 13.7；4 MMA + 4 FMA warp 混跑 `t_mix/(t_mma+t_fma)=1.01`，另两组配比也相加。
边界: 只覆盖 M4 Max 的 fp32 8x8x8 路径与这组 warp 配比；不能据此推断 fp16、其他矩阵形状或所有 ALU 指令的比例。
反例: 若核的瓶颈是 DRAM、threadgroup 带宽或指令发射而不是 fp32 乘加吞吐，换矩阵单元/ALU 的配比不会改变总时间。

## 启动契约能把同一颗核拉开 1.6 倍

Keywords: launch contract grid threads flash attention tile choose variant
Confidence: verified
观察: 同一道题、同一颗核的写法、同一个模型与预算，只把启动契约从 64x32t256 换成 32x16t128，确认加速比从 0.610 升到 0.970。
来源: `runs/apple-m4-max/p1/fa_64x32t256/result.json`、`runs/apple-m4-max/p1/fa_32x16t128/result.json`
本机实测: 两次独立搜索的确认值，1.59 倍相对差。
边界: 这是两次独立随机搜索的对照，不是配对实验；换题或换 shape 结论可能反过来。
反例: 大 tile 在更长的序列上通常更好，因为每次加载复用得更多。

## 逐元素与归约已经贴着带宽屋顶

Keywords: elementwise reduction bandwidth roof saturated relu silu softmax rmsnorm
Confidence: verified
观察: 一元逐元素与归约类题目的自研核基本在 0.95–1.03 倍之间，再改也挤不出空间。
来源: `runs/apple-m4-max/suite_full/*_fp16/result.json`
本机实测: relu 0.971、silu 0.968、sigmoid 0.951、swiglu 0.989、rmsnorm 0.995、softmax 0.974、reduction_sum 0.974、gemv 1.006、rope 1.025。
边界: fp16 paper 尺寸；换成小尺寸时启动开销占比上升，结论会变。
反例: 把相邻的两三个算子融进一段（省掉一遍读写）会把屋顶本身挪走，这时仍有空间。

## 库没调好的融合算子是真机会

Keywords: fused operator mamba scan library gap mlx half compile opportunity
Confidence: verified
观察: 逐元素之外的融合算子，如果现成库没有为这个 shape 调过，自研能明显超过它。
来源: `runs/apple-m4-max/p1/mamba2/result.json`
本机实测: mamba2_chunk_scan fp16 确认 1.459 倍（对 `mlx_half_compile` 183.19 µs，候选 125.53 µs），对原版 `mlx` 498.92 µs 约 3.97 倍。
边界: 同一批 shape 与 dtype；库更新之后要重测。
反例: 逐元素饱和题没有这种空间（见上一条）。

## 有的库已经换了算法

Keywords: conv2d winograd algorithm library faster than roofline conv
Confidence: verified
观察: torch-mps 的 conv2d 比按直接卷积算出的 roofline 下界还快约 17%，System Trace 显示它跑的是 `ndArrayConvWinogradBatched`；在直接卷积这一层做微调打不过它。
来源: `docs/conv2d约束排查.md`、`runs/apple-m4-max/p1/conv2d_paper/candidates/005_r2.p2/report.json`
本机实测: 候选 951.9 µs，torch-mps 773.5–775.4 µs，直接卷积下界 931.8 µs。
边界: paper 尺寸、fp16；下界是按直接卷积的乘加数算的，换算法后这个数不再是天花板。
反例: Winograd 在输入含 NaN 或无穷大时与「逐抽头相加」的参考不一致，数值用例会判错——这是题目要求的，不是缺陷。

## threadgroup 内存上限是 32768 字节

Keywords: threadgroup memory limit static allocation staging 32768 device
Confidence: verified
观察: 本机 `maxThreadgroupMemoryLength` 是 32768 B；静态占用 12288 B 的核合法。插桩的着色器校验会把静态占用报成大约两倍，别照那个数调参。
来源: `tcenv/backends/metal/runtime.py` 的设备信息、`tcenv/evaluate.py` 里只报静态上限时的例外分支
本机实测: M4 Max（Apple9）32768 B。
边界: 换机型要重新读设备属性，不要把这个数写死在核里。
反例: 动态 `threadgroup` 参数在 `taic-launch` 里声明后可用，但它和静态数组共享同一个上限。

## 静态 threadgroup 占用降低常驻线程组数

Keywords: threadgroup memory occupancy resident threadgroups flash attention pad
Confidence: verified
观察: 同一颗 flash-attention 核只增加 8 KB 静态 pad，把静态占用从 9216 B 提到 17408 B，每核常驻线程组从 3 降到 1；同输出配对对照 wall 321.48→360.17 µs（+12.0%），GPU 时间戳 300.03→342.94 µs（+14.3%）。
来源: `runs/apple-m4-max/flash-attention-pilot-20261022/agent_work/r1.p0/repro_tgmem_base.metal`、`repro_tgmem_pad17k.metal`；proposal `8bd186122ecde37e`。
本机实测: M4 Max，fp16，paper，seed 12345，15 reps；base 9216 B → 3 resident，pad 17408 B → 1 resident；两个版本都通过 correctness。
边界: 同一颗核、同一 launch、只改静态 pad；12–14% 是这道 flash-attention shape 的配对结果，不推广成所有核的固定比例。
反例: 核已经受寄存器或线程数限制，或静态占用没有跨过 32768/2 的驻留阈值时，增加 pad 不一定降低 resident，也不一定同幅变慢。

## 被长度截断后把重试的思考降下来

Keywords: truncation retry reasoning effort low max tokens empty reply
Confidence: verified
观察: 回复被 `max_tokens` 截断且没有内核时，把重试的推理强度从 high 降到 low，二次截断变少、交出内核变多、单价更低。
来源: `runs/apple-m4-max/truncation_replay/REPORT.md`
本机实测: `no_reasoning`（当时仍是 high）20 次里 14 次再截断、4 颗核；`low_effort` 10 次、6 颗核，均价 $0.0474 对 $0.0490。
边界: deepseek-flash、48000 token 上限。换模型要重测。
反例: 换到本来就少思考的模型时这条不适用。
## small GEMM 与 MLX 的 steel kernel 性能相当，不是「打不过」

Keywords: gemm small launch bound mlx steel baseline probe buffer slot measurement artifact
Confidence: verified
观察: `gemm` 的 small 档（M=256,N=256,K=128，fp16）加速比在 0.74–0.96 之间剧烈波动，中心约 0.8。但这**不是**因为 MLX 的 steel kernel 更强：把 MLX 的 steel kernel 和我们的复刻版放进**同一进程、同一 GPU 窗口**对比，两者 GPU 时间基本一样。
来源: `runs/apple-m4-max/20261103-103008_gemm_fp16_deepseek-flash_mwfqjgna/`；探针 `targets/apple-m4-max/probes/`。
本机实测: （同窗口、各自正确的 grid/threads、独立输出 buffer、带正确性检查）
- MLX `steel_gemm_fused_nn_float16_float16_bm64_bn64_bk16_wm1_wn2` ≈ 2.68 µs，我们的复刻版（同一编译器 clang -O3）≈ 2.57 µs，**性能相当甚至复刻版略快**。
- 本机编译器与 MLX 的 metallib 是**同一个 Apple metal 编译器家族**（本机 32023.921，MLX 32023.883），不存在「MLX 用了更好的工具链」。
- 官方 `eval-v2` 的加速比在这个尺寸随 GPU 时钟大幅波动（同一份 init 候选多次测得 0.74、0.79、0.96），单次数值不能当结论。
边界: 早先我们一度误判「MLX kernel 快 20–40%」，那是**探针把输出槽位写错**导致的假象——MLX fused kernel 的输出 D 在 **buffer(3)**（buffer(2) 是可选输入源 C）、params 在 **buffer(4)**，放错槽位会让它提前 return、什么都没算，于是显得「快」。用独立输出 buffer + 正确性检查才暴露这一点。
反例: 这个尺寸加速比不稳，不等于 paper（2048³）也打不赢；paper 时固定成本可忽略，比较才有意义。

## 手写 32 fragment 的 GEMM 在本机 clang 下溢出，8 fragment 反而最快

Keywords: gemm fragment tile register spill simdgroup accumulator occupancy
Confidence: verified
观察: 手写每 SIMD 组 64×32（32 个 fp32 累加器 fragment）的 GEMM，在本机 clang 下寄存器溢出（roofline 读数归零/结果异常）；降到 init 的 8 fragment（每 SG 16×32）才正确且最快。
来源: `runs/apple-m4-max/20261103-103008_gemm_fp16_deepseek-flash_mwfqjgna/candidates/000_init/`；`targets/apple-m4-max/probes/steel_direct_16x32.metal`。
本机实测: 32 fragment 版本 roofline NaN、结果错误；8 fragment 版本 6 个用例全过、官方 candidate ≈2.5–3.6 µs。
边界: 这是**本机 clang 的寄存器分配**限制，不是「32 fragment 这个 tile 结构本身不行」——MLX 预编译的 32 fragment kernel 能跑，说明不同编译产物下它可行。
反例: 用 MLX 的编译器/配置编译 32 fragment 源码能达到可用的性能，所以不能据此排除大 tile 方向。

## 多阶段批次的计时固定成本可降约 75%（native 编码器已支持多阶段）

Keywords: timing multi stage pipeline native batch fixed cost launch bound flash attention
Confidence: verified
观察: 多阶段、纯 buffer、无 fill 的批次现在和单阶段一样走 harness 自有的 ObjC 编码器，固定成本大幅下降。这是 launch-bound 小算子（gemm、flash-attention 等）计时改善的主要来源，与 kernel 写法无关。
来源: `tcenv/backends/metal/native_batch.{m,py}`（`taic_metal_run_pipeline`）、`tcenv/backends/metal/runtime.py`。
本机实测: 同一 2 段 pipeline（纯 buffer、无 fill），PyObjC 编码 fixed ≈5.26 µs/iter，native 编码 fixed ≈1.33 µs/iter（约 -75%）；单阶段 GEMM fixed ≈1.0 µs/iter。正确性与 PyObjC 逐值一致（`tests/backends/metal/test_native_batch.py` 的多阶段对照）。
边界: 只覆盖「纯 buffer、无 fill、无动态 threadgroup 参数」的批次；带 fill/workspace 或动态 threadgroup 的批次仍走 PyObjC，那部分固定成本还在。
反例: 给 `ctx.run` 加 native 快路径对官方发布数字没有影响（官方走 run_pipeline），且曾引入 unsafe，已回退，不要再走这条路。

## launch-bound 小算子的计时路径：run_pipeline 走 native，ctx.run 走 PyObjC

Keywords: timing dispatch overhead pyobjc native batch fixed cost launch bound harness
Confidence: verified
观察: 候选计时有两条路径。官方 `run_pipeline`（评测/发布用）对「单阶段、纯 buffer、无 fill」的批次本来就走 harness 自有的 ObjC 编码器（`tcenv/backends/metal/native_batch.m`），固定成本低；而 `MetalContext.run`（探针、自测、部分 v1 路径）走 PyObjC，空核固定成本约 2.2 µs。
来源: `tcenv/backends/metal/runtime.py`、`native_batch.py`、`tcenv/v2_worker.py`。
本机实测: 同脚本同轮，空核 PyObjC ≈2.35 µs，直接调 native 编码器 ≈0.95 µs。
边界: `v2_worker` 此前漏了 `backend.prepare_context`，导致 worker 里的 run_pipeline 也只能走 PyObjC（已补上，见 `tcenv/v2_worker.py`）。给 `ctx.run` 加 native 快路径对官方发布数字没有影响（官方走 run_pipeline），且曾引入 unsafe，已回退，不要再走这条路。
反例: 带 fill/workspace 的批次不走 native（`eligible` 拒绝），那部分的固定成本仍在 PyObjC 侧；要降它得再扩展 fill 支持（本次尝试后已回退，见 `native_batch.m` 的接口预留）。

## msl-bundle：读 ramp 表的 affine_ramp 比 torch 快，但这条还没进提示词

Keywords: bundle workspace host tiling affine ramp baseline speedup unverified
Confidence: reported
观察: `msl-bundle`（Host Tiling 写 workspace + device 读表）在 elementwise 上明显快过 `torch_mps`。这是一条**未验证**的观察（not reproduced）：它来自一次在忙碌机器上的自测（cv 10.5%），没有经过安静窗口，也没有走过真实搜索 + 确认，所以按 `reported` 记录，不进提示词。
来源: `tasks/affine_ramp/`（第一道声明 `space: workspace` 的标准题）、`tests/support/msl_bundle_affine_ramp/evidence.json`。
本机实测: fp16/paper（N=8192, NS=128），`eval-v2 --timed --track msl-bundle --reps 5`：candidate 22.14 µs，`torch_mps` 83.45 µs，speedup 3.769x；同一次 `--quick` 判 ok（compiled, ABI ok, safe, correct）。
边界: 证据口径是 **steady-state table-reuse** —— `zero=false` 时 Host Tiling 在计时外，而 baseline 每次调用都自己构造 ramp，两者做的准备工作不同，不能写成「端到端同样的工作」。动态 `[[threadgroup(k)]]` 参数不支持；`threadgroup_bytes` 必须写 0。
反例: 依赖初始零 workspace 却不声明 `{"zero": true}` 的候选会在 poison 试次上判错 —— 正确性对每个 workspace 都填，计时只清声明 `zero` 的。

- [harness b4131be13c4750c8] Add a tcenv.cli ab subcommand that compiles several kernel files, times them interleaved in one process over rotating buffer sets, and prints per-file median wall time and cv.
