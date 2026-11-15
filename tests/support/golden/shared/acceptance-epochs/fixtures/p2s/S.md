# P2-S：已验证经验整理与配对搜索的 dry-run 准备

- epoch：`p2-s-experience-search-prep-1`
- parent HEAD：`a9c1dccce636107c2852995578b05611bf19865b`
- prompt：`e9d28ff61819a041610d763f5c1464ea860265be2d414f78c15a1f8aea1fdbcf`，与 P2-M 的 current 相同，`prompt_changed=false`
- pack：79 文件，`b5b74185710e5c64cf55ed20cc472f3258e0fe2b1c631bb7f850eeb8dcbb069e`，与 P2-R 记录相同
- `publication_allowed=false`；NPU 仍 simulation-only
- 实弹搜索：**未授权/未执行**。Plan 3 没有完成。

本阶段把 P2-I 到 P2-R 里核对过的经验收成一份冻结叠加层，并写出 `gemm_silu` fp16 small 的 AB/BA dry-run。叠加层没有写入 `targets/apple-m4-max/experience.md`，也没有写入其它进入 prompt fingerprint 的文件。P2-M 的 prompt-epoch 与 P2-I 到 P2-R 的 manifest 保持原字节。

## 教给搜索的六条

| 经验 | 证据类 | 可以怎么用 |
| --- | --- | --- |
| conv2d paper 全局 U | 一对 fresh subject | 只对这套 F(4×4) MMA：重算 1474.42 µs / 全局 U 617.29 µs = 2.3885 倍。同算法复用，不是发布加速比 |
| conv2d small | fresh + probe 下界 | 单阶段 3.2373 µs，相对最快基线 0.9858。两阶段下限 6.1287 µs 是 `not_a_candidate` |
| gemm_silu REG | fresh subject | small 1.8252 µs、1.6721 倍；paper 1160.87 µs、1.0071 倍。GLOBAL 没有实现 |
| bitdelta REG | fresh + paired probe | paper 的配对中位比 1.0259 可以保留。small 只教大约一个数量级，不教精确倍数或排序。没有 SHARED 候选 |
| GEMM Split-K | probe 下界 + 归档锚点 | small 下限 7.0390 µs，`not_a_candidate`。paper 的 10.98% 是峰值带宽时间等价，不是字节流量比。不要做 |
| SEQ 切链 | 八个 fresh subject，含 B 臂 | KDA 的 `device/host_piece_handoff.metal` 六臂同为 `124c7485…`。A−C 是跨 confirmation 的总开销，不能解释为隔离的 handoff。范围是 Apple M4 Max、fp16、varlen 仅 L1 mixed |

标量三臂，以及文中标成 probe 下界或 paired probe 的数字，作为反例上下文保留，但不是独立 teachable lesson，也不是 confirmed speedup。`kda tail` 的决策测量、`kda single` 的 N=1 自检和 P2-R 的 harness 修复写在叠加层里作边界：数值不采用，不作为独立经验。occupied retry 的墙钟和搜索最优倍数不进入叠加层。

## 配对实验

唯一差别是处理臂多这一份冻结叠加层。`init_samples`、`task_ledger`、`plateau_after`、`sweep_tool` 两臂都是对照默认值。任务是 `gemm_silu` / fp16 / small，搜索种子 7，委员会仍是 `taic.toml` 的 first/second。2 对、AB 然后 BA、4 个 cell。每 cell 软预算 $3。$12 是是否启动后续 cell 的承诺上限，不是账单天花板。可容忍超支仍未批准。名义合计 $12。δ、批准人和执行记录都是空白。

`--execute` 即使四道门禁都填上也会拒绝：叠加层没有装进实时提示，两臂会看到同一段文字。这不是一次已执行的搜索。

合同见 [已验证经验与配对搜索准备](../../../../../../docs/架构/verified-experience-search-prep.md)。

## 验收

```bash
uv run python tests/support/golden/shared/shape-portfolio-阶段验收/P2-S/verify.py
```

13 个准备测试加 6 个既有四开关测试，共 19 个 CPU 节点。不跑 GPU，不调用模型。全量安全 CPU 套件里已有的环境失败和历史锚点失败不在本阶段消掉，也不写成全绿。
