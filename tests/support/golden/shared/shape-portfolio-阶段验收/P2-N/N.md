# P2-N：small 优化停止决定

- epoch：`p2-n-conv2d-small-stop-1`
- parent HEAD：`a9c1dccce636107c2852995578b05611bf19865b`
- `adopt_candidate=false`
- `stop_small_optimization=true`
- `publication_allowed=false`

最后一个结构实验让每个 lane 用私有 `half[72]` 缓存八组权重，在 36 个 Winograd 位置间复用，试图消除重复设备读取。实验源码和正式 `eval-v2` 报告分别保存在 `evidence/weight-cache.metal` 与 `weight-cache-report.json`，SHA-256 为 `62e1ac92b7bbc15f936d819e3a41da9aca979cadf88a2af265430351b76651eb`。

实验六个 case（含 IEEE）均正确，但 candidate 为 **4.0056 µs**，相对 P2-K 最新 fresh 3.4935 µs 慢 **14.66%**，相对此前 3.2373 µs 慢 **23.73%**。私有 cache 带来的寄存器或本地内存压力超过了减少设备权重读取的收益，未达到预设的至少 5% 改善，反而明显回退。

因此按“做不到就适可而止”的约定停止 small 微优化，不将该源码加入 portfolio，也不再启动新的 small 结构实验或付费搜索。P2-J 的生产实验 candidate 保持原 SHA-256 `50ac29d1…`，P2-K 的 validated-only 状态不变。

**停止不是因为没有预算或单纯模型费用。** 根本原因是技术收益已经递减且不稳定：容易获得的优化已经完成，最新缺口需要结构性收益，而实际结构实验反而回退。继续投入工程时间、验证维护、机会成本，以及可选的模型费用，都缺乏合理的预期回报。这个停止决定是综合投入产出判断，不是金钱限制。

**偶尔超过也不等于可靠交付。** 微秒级 kernel 的计时波动可以达到数个百分点，某一次超过 1.0× 并不能证明稳定加速。可靠交付必须同时满足全部正确性 case、candidate/build/evaluation 身份一致、重复独立 fresh 执行和声明的延迟/噪声门。研究可以继续探索，但一次性好成绩不能作为生产证据。

`verify.py` 校验实验源码与报告摘要、六案正确性、停止决定、当前 candidate 未变化，以及 P2-I/J/K/L/M 的历史 manifest。验收为 1 个 CPU 节点，不需要 GPU 或模型。
