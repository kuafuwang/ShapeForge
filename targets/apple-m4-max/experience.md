# 经验条目（本仓库实测过的那一部分）

这份文件是 KernelWiki 里「经验层」的入口：每条都给出观察、来源、本机实测、边界，以及什么时候不成立。
和 `INDEX.md` 里那两份 Apple PDF 的区别是：PDF 说语言与设备能力，这里说**我们自己测到过什么**。

证据等级写在每条的 `Confidence:` 上。`verified` = 本机复现过，`来源` 里必须有 `runs/` 或 `docs/` 的路径；
`reported` = 只有上游出处（别人的 PR、博客、WWDC）还没在本机复现，**不进提示词**，只留在库里。

## 累加器片段到 32 会溢出

Keywords: gemm accumulator register spill simdgroup fragment tile occupancy
Confidence: verified
观察: 每组 SIMD 的累加器片段达到 32 个及以上时，寄存器溢出，同一题的耗时是一千多微秒变到一万七到三万五微秒。
来源: `runs/hw_features/REPORT.md`（同一批契约的实测对照）
本机实测: fp16 paper，32 片段以上 17880–35006 µs，16 片段及以下 1361–1942 µs，约 12–26 倍。
边界: 只覆盖 fp16 与本机（M4 Max）的寄存器文件；换成半分块或更小的 BLOCK 时片段数会变。
反例: 把 K 维切得更细、用 threadgroup 做中间累加，可以既保留大 tile 又不让片段数越界。

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
来源: `tcenv/metal_runtime.py` 的设备信息、`tcenv/evaluate.py` 里只报静态上限时的例外分支
本机实测: M4 Max（Apple9）32768 B。
边界: 换机型要重新读设备属性，不要把这个数写死在核里。
反例: 动态 `threadgroup` 参数在 `taic-launch` 里声明后可用，但它和静态数组共享同一个上限。

## 被长度截断后把重试的思考降下来

Keywords: truncation retry reasoning effort low max tokens empty reply
Confidence: verified
观察: 回复被 `max_tokens` 截断且没有内核时，把重试的推理强度从 high 降到 low，二次截断变少、交出内核变多、单价更低。
来源: `runs/apple-m4-max/truncation_replay/REPORT.md`
本机实测: `no_reasoning`（当时仍是 high）20 次里 14 次再截断、4 颗核；`low_effort` 10 次、6 颗核，均价 $0.0474 对 $0.0490。
边界: deepseek-flash、48000 token 上限。换模型要重测。
反例: 换到本来就少思考的模型时这条不适用。
- [harness b4131be13c4750c8] Add a tcenv.cli ab subcommand that compiles several kernel files, times them interleaved in one process over rotating buffer sets, and prints per-file median wall time and cv.
