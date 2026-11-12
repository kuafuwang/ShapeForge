# GEMM Split-K：不实施的可行性结论

P2-L 只回答 Split-K 是否值得实现，不新增 candidate，也不把 probe 当成性能发布。结论依据是实测提交下限、现有 P3 candidate、输出并行度和 scratch 流量。

## small

P3 的现有 candidate 为 **1.8763 µs**，MLX baseline 为 **2.2444 µs**。GEMM ABI 的真实运行时 probe 在不完整 kernel 上测得：

- 单阶段墙钟中位数：2.9669 µs；
- 两阶段、无 fill 墙钟中位数：**7.0390 µs**；
- 两阶段 GPU 中位数：4.7752 µs。

这些 kernel 故意不完整，`not_a_candidate=true`，没有正确性或 candidate 性能含义。两阶段下限已经是现有 candidate 的 **3.75×**，并且超过 MLX baseline；真实的 Split-K partial 计算和归并只会继续增加时间。Split-2 的 fp32 partial 写+读还至少需要 1 MiB 流量，按标定峰值也约 1.991 µs。small 因此不实施 Split-K。

## paper

现有 paper candidate 为 **1160.09 µs**，MLX baseline 为 **1155.36 µs**，实测 fp16 roofline 约 **1163.48 µs**，已经接近计算下界。当前 grid 为 32×32，即 **1024 个输出 threadgroup**，不存在需要通过 Split-K 增加的输出并行度。

每个 split 的 fp32 partial 写+读为 32 MiB；Split-2 合计至少 **64 MiB / 127.42 µs**，约相当于当前 candidate 的 **10.98%**，尚未计入归并输出和额外 stage。它没有可补偿的计算余量，因此 paper 同样不实施 Split-K。

`experiments/gemm_splitk_probe.py` 可以重跑真实 Metal floor。`P2-L/evidence/analysis.json` 由该 floor 和固定 P3 锚点通过 `tests/support/gemm_splitk.py` 计算，不能手改结论。CPU 测试复核公式与拒绝原因，真实 Metal 节点重跑 probe 并只检查拒绝条件，不钉墙钟浮点逐值相等。

本阶段 `implementation_started=false`、`candidate_source_added=false`、`publication_allowed=false`。通用 ScratchSpec 的 SPLIT 类型保留给其它确有输出并行度不足且归约成本可覆盖的场景，不因为 conv2d 的 SHARED 成功就强行推广。
