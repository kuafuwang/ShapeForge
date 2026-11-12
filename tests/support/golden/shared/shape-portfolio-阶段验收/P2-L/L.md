# P2-L：GEMM Split-K 负结果

- epoch：`p2-l-gemm-splitk-negative-1`
- parent HEAD：`521d9d5322058fee14478a066388be3410960b60`
- pack：79 文件，`5512395852ea7608a6075e2b6b45dca9bd85f112210af336cf6f8f77f5fd811f`
- `implementation_started=false`
- `candidate_source_added=false`
- `publication_allowed=false`

small 的现有 candidate 为 1.8763 µs，GEMM ABI 两阶段无 fill probe 已需 7.0390 µs，是 candidate 的 3.75×，也超过 MLX baseline 2.2444 µs。probe kernel 故意不完整，仅测 launch floor，不能当作 candidate。

paper 已有 1024 个输出 threadgroup，candidate 1160.09 µs 接近 fp16 roofline 1163.48 µs。Split-2 的 fp32 partial 写+读至少 64 MiB / 127.42 µs，额外增加约 10.98% 流量，没有输出并行度或计算余量可以补偿。

因此 small 和 paper 均不实现 Split-K。`launch-floor.json` 保存真实 Metal probe，`analysis.json` 由固定公式和 P3 锚点派生。`verify.py` 还逐文件核对 P2-I 71、P2-J 22、P2-K 13 个历史交付文件，运行 1 个 CPU 分析节点和 1 个真实 Metal floor 重放节点。预期最终行：

`P2-L acceptance (1 CPU + 1 real Metal negative replay; no candidate, no publication)`。

本阶段尚未声称独立 reviewer 签署。
