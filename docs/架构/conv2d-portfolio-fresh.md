# conv2d small+paper：portfolio-level fresh confirmation

P2-K 确认的是 P2-J 提交的**同一棵** small+paper 候选树，而不是把两个独立单核报告拼成 portfolio 结论。P2-I 与 P2-J 的历史 manifest 均保持逐字节不变。

冻结身份：

- portfolio freeze：`db264da7e7a4c389e120f740ea1e71806c7655e0d70c820453afe0d1cc2f2bb0`
- validation plan：`8ceace0e53013cdf7d2c06b32db1225e3f766dfa7a565cd9bea854d5d8ea6d64`
- validated DispatchManifest：`f31ebdc787783b0ff7d1cff138b22f52992b2ff53666acf1ac3d755169fd88a6`，generation=0，没有激活。

`tests/support/golden/shared/shape-portfolio-阶段验收/P2-K/record.py` 用同一 coverage、small/paper/fallback 字节、task-pack digest 和公开 master key 重建 freeze/validation。四个 subject 都由 `confirm_subject` 分别做搜索侧与 fresh 侧的独立真实构建：

| 域点 | fresh seed | 结果 | fresh 墙钟 |
| --- | ---: | --- | ---: |
| small / benchmark | 2213214225460092 | confirmed，identity match | 3.4935 µs |
| paper / benchmark | 3017626063546626 | confirmed，identity match | 611.2739 µs |
| small / golden | 2983258450447635 | confirmed，identity match | correctness-only |
| paper / golden | 8939627534293646 | confirmed，identity match | correctness-only |

small benchmark 相对 Torch-MPS 为 5.7140×、相对 MLX 为 0.9111×；paper benchmark 相对 Torch-MPS 为 1.2739×、相对 MLX 为 1.5940×。两档 benchmark 都满足 1.10 延迟门。small 仍未稳定超过最佳基线，不作最佳基线加速或生产发布声明。

`validated-manifest.json` 由四个 subject 和逐点 `activation_expectation` 重建，覆盖完整、状态为 `validated`。真实 Metal 重放节点再次执行四个 fresh subject，并要求重建 manifest 与归档逐字段相同；计时不比较微秒逐值相等。该 manifest 没有 active pointer，也没有 activation generation 1。

packed-half 指数位检查是最后一次 small 微优化实验。三次交替 fresh-seed A/B 的新版中位数为 3.3685 µs，旧版为 3.4220 µs；新版相对 MLX 的中位数只有 0.9483×，且并非每轮超过 1.0×。因此没有采用该变体，P2-J 的 small 源码保持原 SHA-256。原始三轮报告与 `adopt_candidate=false` 保存在 `packed-half-probe.json`，没有把首次单次的 1.013× 当作稳定结论。
