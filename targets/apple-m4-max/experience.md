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

宿主重跑: runs/hw_features/REPORT.md
## 矩阵单元与 SIMT ALU 的 fp32 吞吐相近，工作相加

Keywords: matrix simdgroup fp32 fma simt throughput additive operand reuse
Confidence: verified
观察: 本机 fp32 8x8x8 simdgroup MMA 的吞吐与 fp32 SIMT FMA 相近；两种工作混跑时时间相加，不重叠。矩阵单元对 fp32 不是额外的一份算力，真正的红利是操作数复用。
来源: `runs/apple-m4-max/conv2d-feedback-20261021/agent_work/r11.p0/NOTES.md`（实测数字）；同目录 `probe_mix.py`（探针源码）
本机实测: 8 warp 全 MMA 12.62 TFLOP/s，4 warp 全 MMA 14.34，8 warp 全 FMA 13.7；4 MMA + 4 FMA warp 混跑 `t_mix/(t_mma+t_fma)=1.01`，另两组配比也相加。
边界: 只覆盖 M4 Max 的 fp32 8x8x8 路径与这组 warp 配比；不能据此推断 fp16、其他矩阵形状或所有 ALU 指令的比例。
反例: 若核的瓶颈是 DRAM、threadgroup 带宽或指令发射而不是 fp32 乘加吞吐，换矩阵单元/ALU 的配比不会改变总时间。

宿主重跑: runs/apple-m4-max/conv2d-feedback-20261021/agent_work/r11.p0/NOTES.md
## 启动契约能把同一颗核拉开 1.6 倍

Keywords: launch contract grid threads flash attention tile choose variant
Confidence: verified
观察: 同一道题、同一颗核的写法、同一个模型与预算，只把启动契约从 64x32t256 换成 32x16t128，确认加速比从 0.610 升到 0.970。
来源: `runs/apple-m4-max/p1/fa_64x32t256/result.json`、`runs/apple-m4-max/p1/fa_32x16t128/result.json`
本机实测: 两次独立搜索的确认值，1.59 倍相对差。
边界: 这是两次独立随机搜索的对照，不是配对实验；换题或换 shape 结论可能反过来。
反例: 大 tile 在更长的序列上通常更好，因为每次加载复用得更多。

宿主重跑: runs/apple-m4-max/p1/fa_32x16t128/result.json
## 逐元素与归约已经贴着带宽屋顶

Keywords: elementwise reduction bandwidth roof saturated relu silu softmax rmsnorm
Confidence: verified
观察: 一元逐元素与归约类题目的自研核基本在 0.95–1.03 倍之间，再改也挤不出空间。
来源: `runs/plan_type_analysis/REPORT.md` 的 suite_full 表（逐项 speedup 与 roofline 比例）
本机实测: relu 0.971、silu 0.968、sigmoid 0.951、swiglu 0.989、rmsnorm 0.995、softmax 0.974、reduction_sum 0.974、gemv 1.006、rope 1.025。
边界: fp16 paper 尺寸；换成小尺寸时启动开销占比上升，结论会变。
反例: 把相邻的两三个算子融进一段（省掉一遍读写）会把屋顶本身挪走，这时仍有空间。

宿主重跑: runs/plan_type_analysis/REPORT.md
## 库没调好的融合算子是真机会

Keywords: fused operator mamba scan library gap mlx half compile opportunity
Confidence: verified
观察: 逐元素之外的融合算子，如果现成库没有为这个 shape 调过，自研能明显超过它。
来源: `runs/apple-m4-max/p1/mamba2/result.json`
本机实测: mamba2_chunk_scan fp16 确认 1.459 倍（对 `mlx_half_compile` 183.19 µs，候选 125.53 µs），对原版 `mlx` 498.92 µs 约 3.97 倍。
边界: 同一批 shape 与 dtype；库更新之后要重测。
反例: 逐元素饱和题没有这种空间（见上一条）。

宿主重跑: runs/apple-m4-max/p1/mamba2/result.json
## 有的库已经换了算法

Keywords: conv2d winograd algorithm library faster than roofline conv
Confidence: verified
观察: torch-mps 的 conv2d 比按直接卷积算出的 roofline 下界还快约 17%，System Trace 显示它跑的是 `ndArrayConvWinogradBatched`；在直接卷积这一层做微调打不过它。
来源: `docs/conv2d约束排查.md`、`runs/apple-m4-max/p1/conv2d_paper/candidates/005_r2.p2/report.json`
本机实测: 候选 951.9 µs，torch-mps 773.5–775.4 µs，直接卷积下界 931.8 µs。
边界: paper 尺寸、fp16；下界是按直接卷积的乘加数算的，换算法后这个数不再是天花板。
反例: Winograd 在输入含 NaN 或无穷大时与「逐抽头相加」的参考不一致，数值用例会判错——这是题目要求的，不是缺陷。

宿主重跑: runs/apple-m4-max/p1/conv2d_paper/candidates/005_r2.p2/report.json
## threadgroup 内存上限是 32768 字节

Keywords: threadgroup memory limit static allocation staging 32768 device
Confidence: verified
观察: 本机 `maxThreadgroupMemoryLength` 是 32768 B；报告中的静态占用 9216 B 的核合法。插桩的着色器校验会把静态占用报成大约两倍，别照那个数调参。
来源: `runs/apple-m4-max/flash-attention-pilot-20261022/confirm_report.json`（设备上限与静态占用）；`tcenv/backends/metal/runtime.py` 的设备信息、`tcenv/evaluate.py` 里只报静态上限时的例外分支（插桩行为的代码观察，不是报告字段）
本机实测: M4 Max（Apple9）32768 B。
边界: 换机型要重新读设备属性，不要把这个数写死在核里。
反例: 动态 `threadgroup` 参数在 `taic-launch` 里声明后可用，但它和静态数组共享同一个上限。

宿主重跑: runs/apple-m4-max/flash-attention-pilot-20261022/confirm_report.json
## 静态 threadgroup 占用降低常驻线程组数

Keywords: threadgroup memory occupancy resident threadgroups flash attention pad
Confidence: verified
观察: 同一颗 flash-attention 核增加 8 KB 静态 pad，把静态占用从 9216 B 提到 17408 B；按 32768 B 上限计算，每核常驻 threadgroup 上限从 3 降到 1。2026-11-20 同窗口 `ab` 的 GPU 中位从 331.6442 增至 370.8340 µs（+11.8168%）。
来源: `runs/host-rerun-threadgroup-residency-20261120/stdout.txt`（16 轮交错 GPU 计时）、同目录 `static_bytes.jsonl`（PSO 反射的静态占用）。
本机实测: M4 Max，fp16 paper；base 9216 B / 3 resident / 331.6442 µs，pad 17408 B / 1 resident / 370.8340 µs；两版 quick 与 ab 都正确，worst scaled error 都是 0.9138155，cv 约 0.20%/0.21%。
边界: 同一颗核、同一 launch，只比较这两个具体文件。pad 版为防止优化写入 pad 并加入 1e-30 的 `pad_bias`，所以不是形式上“只改声明”；两版输出误差相同。11.82% 是当前窗口结果，不沿用旧稿的 12–14%，也不推广为所有核的固定比例。
反例: 核已经受寄存器或线程数限制，或静态占用没有跨过 32768/2 的驻留阈值时，增加 pad 不一定降低 resident，也不一定同幅变慢。
宿主重跑: runs/host-rerun-threadgroup-residency-20261120/stdout.txt
## 被长度截断后把重试的思考降下来

Keywords: truncation retry reasoning effort low max tokens empty reply
Confidence: verified
观察: 回复被 `max_tokens` 截断且没有内核时，把重试的推理强度从 high 降到 low，二次截断变少、交出内核变多、单价更低。
来源: `runs/apple-m4-max/truncation_replay/REPORT.md`
本机实测: `no_reasoning`（当时仍是 high）20 次里 14 次再截断、4 颗核；`low_effort` 10 次、6 颗核，均价 $0.0474 对 $0.0490。
边界: deepseek-flash、48000 token 上限。换模型要重测。
反例: 换到本来就少思考的模型时这条不适用。
宿主重跑: runs/apple-m4-max/truncation_replay/REPORT.md
## small GEMM 与 MLX 的 steel kernel 性能相当，不是「打不过」

Keywords: gemm small launch bound mlx steel baseline probe buffer slot measurement artifact
Confidence: verified
观察: `gemm` small（M=256,N=256,K=128，fp16）的官方 speedup 仍低于 1，但同一次 `--timed` 交错测量里，候选 GPU 时间与 MLX 相当，不能把 wall speedup 直接解释成 MLX 的 kernel 快 20–40%。
来源: `runs/host-rerun-gemm-small-mlx-20261120/direct8.json`、同目录 `steel32.json`；候选源码 `targets/apple-m4-max/probes/`。
本机实测: direct8：candidate GPU 1.7654 µs、MLX 1.9336 µs，candidate wall 2.0112 µs、官方 speedup 0.96139；steel32：candidate GPU 1.8254 µs、MLX 2.0187 µs，candidate wall 2.0876 µs、speedup 0.96699。两次 status 都是 ok。
边界: 每个比值只在自己那次 `--timed` 窗口里成立，direct8 与 steel32 不跨窗口互比。GPU 时间戳相近不等于 wall 固定成本相同，也不代表 paper 尺寸结论。旧稿的 2.68/2.57 µs 是另一套探针和窗口，本次没有复用那些数。
反例: `tcenv ab` 的一次启动一个命令缓冲边界会给出另一组绝对值；要讨论发布 speedup 必须看官方 `--timed` 边界，不能拿不同计时边界的绝对微秒数拼接。
宿主重跑: runs/host-rerun-gemm-small-mlx-20261120/direct8.json
## 32 fragment 在当前工具链下可正确运行，但计时边界会改变快慢结论

Keywords: gemm
Confidence: verified
题目: gemm
芯片: apple-m4-max
落点: targets/apple-m4-max/experience.md
观察: 32 fragment 在当前工具链下可正确运行，但计时边界会改变快慢结论
来源: `runs/host-rerun-gemm-fragments-20261120/stdout.txt`
本机实测: fp16 small：steel_gemm_64x64（每 SG 32 个 fp32 累加器 fragment）与 steel_direct_16x32（8 fragment）都通过 --quick；tcenv ab 同窗口 GPU 中位分别为 13.0347 与 7.8531 µs，32/8=1.6598。另一次 --timed 的生产批量边界下两者 candidate_gpu_us_median 为 1.8254/1.7654 µs。
边界: 只覆盖本机 2026-11-20 的工具链、这两个具体文件和 fp16 small。ab 每次 run_once 一个命令缓冲，--timed 把多次启动融进一批；两种边界给出不同比值，不能只拿 1.66 倍当纯 kernel 结论。旧的“32 fragment 必然 roofline NaN/结果错误”已被本次正确性结果推翻。
反例: 换编译器、tile 或启动批次后 32 fragment 可能打平或更快；本次结果只否定“当前必然溢出”，不否定其他编译产物曾发生溢出。
宿主重跑: runs/host-rerun-gemm-fragments-20261120/stdout.txt
## 多阶段批次的编码固定成本可降约 65%（native 编码器已支持多阶段）

Keywords: timing multi stage pipeline native batch fixed cost launch bound flash attention
Confidence: verified
观察: 两段、纯 buffer、无 fill 的批次走 harness 自有 ObjC 编码器时，编码固定成本明显低于 PyObjC。2026-11-20 重跑没有再得到旧稿的 75%，本批估计降幅是 64.96%。
来源: `runs/host-rerun-native-timing-20261120/stdout.txt`（原始逐轮输出）与同目录 `probe.py`。
本机实测: 每次调用 64 iterations、15 个计入样本的交错轮次。native：wall 11.2969、GPU 7.4323、wall−GPU 3.8216 µs/iter；PyObjC：18.5052、7.4264、10.9075 µs/iter。固定成本估计从 10.9075 降到 3.8216 µs/iter（-64.9636%），输出逐值一致。
边界: `wall−GPU` 是编码/提交/等待固定成本的估计，不是独立硬件计数器；只覆盖这两段 kernel、64-iteration 批次和当前宿主。wall 总时间降幅不是 65%，不能混用。
反例: fills 非空或存在动态 threadgroup 绑定时 `eligible()` 退回 PyObjC；改变每批 iteration 数、GPU  kernel 时间或命令缓冲边界都会改变这个比例。
宿主重跑: runs/host-rerun-native-timing-20261120/stdout.txt
## launch-bound 小算子的计时路径：run_pipeline 走 native，ctx.run 走 PyObjC

Keywords: timing dispatch overhead pyobjc native batch fixed cost launch bound harness
Confidence: verified
观察: 候选计时有两条路径。官方 `run_pipeline` 对符合条件的批次走 harness 自有 ObjC 编码器；`MetalContext.run` 走 PyObjC。当前空核重跑仍显示固定成本差别，但幅度比旧稿小。
来源: `runs/host-rerun-native-timing-20261120/stdout.txt`、同目录 `probe.py`；实现位于 `tcenv/backends/metal/runtime.py` 与 `native_batch.py`。
本机实测: 空核每次 64 iterations、15 个交错样本。`ctx.run`：wall 6.8105、GPU 2.1250、wall−GPU 4.7116 µs/iter；native `run_pipeline`：5.1504、2.1270、3.0280 µs/iter。固定成本估计下降 35.7329%，不是旧稿的 2.35/0.95 µs。
边界: `wall−GPU` 是固定成本估计；空核仍有约 2.12 µs/iter 的 GPU 时间，不能把 wall 全算作 Python。只覆盖当前宿主、这个空核和 64-iteration 批次。
反例: 非空 fills 或动态 threadgroup 绑定不走 native（`eligible` 拒绝），那部分固定成本仍在 PyObjC 侧。给 `ctx.run` 加 native 快路径曾引入 unsafe 且已回退，不要据此重试。
宿主重跑: runs/host-rerun-native-timing-20261120/stdout.txt
## msl-bundle：读 ramp 表的 affine_ramp 比 torch 快，但这条还没进提示词

Keywords: bundle workspace host tiling affine ramp baseline speedup unverified
Confidence: reported
观察: `msl-bundle`（Host Tiling 写 workspace + device 读表）在 elementwise 上明显快过 `torch_mps`。这是一条**未验证**的观察（not reproduced）：它来自一次在忙碌机器上的自测（cv 10.5%），没有经过安静窗口，也没有走过真实搜索 + 确认，所以按 `reported` 记录，不进提示词。
来源: `tasks/affine_ramp/`（第一道声明 `space: workspace` 的标准题）、`tests/support/msl_bundle_affine_ramp/evidence.json`。
本机实测: fp16/paper（N=8192, NS=128），`eval-v2 --timed --track msl-bundle --reps 5`：candidate 22.14 µs，`torch_mps` 83.45 µs，speedup 3.769x；同一次 `--quick` 判 ok（compiled, ABI ok, safe, correct）。
边界: 证据口径是 **steady-state table-reuse** —— `zero=false` 时 Host Tiling 在计时外，而 baseline 每次调用都自己构造 ramp，两者做的准备工作不同，不能写成「端到端同样的工作」。动态 `[[threadgroup(k)]]` 参数不支持；`threadgroup_bytes` 必须写 0。
反例: 依赖初始零 workspace 却不声明 `{"zero": true}` 的候选会在 poison 试次上判错 —— 正确性对每个 workspace 都填，计时只清声明 `zero` 的。

- [harness b4131be13c4750c8] Add a tcenv.cli ab subcommand that compiles several kernel files, times them interleaved in one process over rotating buffer sets, and prints per-file median wall time and cv.
## 一次性 SiLU 留在寄存器里，不走 GLOBAL scratch

Keywords: gemm_silu
Confidence: verified
题目: gemm_silu
芯片: apple-m4-max
落点: targets/apple-m4-max/experience.md
观察: 一次性 SiLU 留在寄存器里，不走 GLOBAL scratch
来源: `runs/host-rerun-gemm_silu-reg-small-20261119/stdout.txt`
本机实测: fp16 small：candidate 1.6864 µs，mlx 2.9060 µs，speedup 1.4991x（cv 0.0297，reps 15，candidate_gpu_us_median）。种子字节是 tests/support/golden/metal/gemm_silu_reg/variants/small/kernel.metal，sha256 d5203e45…
边界: 只覆盖 fp16 small 的 64x64x32 t128 steel GEMM；SiLU 在 fp32 累加器上做、最后只向 fp16 舍入一次。P2-S 冻结叠加层里记的是 1.8252 µs / 1.6721x，那是另一次会话的 fresh subject，不是同一段窗口，两者不能互换。
反例: GLOBAL scratch 版省不下 FLOPs，还要多一次 fp32 写读、一次 barrier 和一次 dispatch；两阶段 no-fill 下限是 8.7713 µs，比这一版慢一个量级。把激活拆成第二个 kernel 是把寄存器收益换回带宽成本。
宿主重跑: runs/host-rerun-gemm_silu-reg-small-20261119/stdout.txt

